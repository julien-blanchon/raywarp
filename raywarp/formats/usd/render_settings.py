"""`rw.formats.usd.settings(stage)`: a stage's authored Omniverse RTX render settings as
RenderSettings.

Omniverse stores its renderer settings as `rtx:*` keys in the root layer's
`customLayerData["renderSettings"]` and as `omni:rtx:*` attributes of UsdRender RenderSettings /
RenderProduct prims. The mapping (docs/settings.md) starts from base settings (by
`rtx:rendermode` unless given: the RTX look recipe, the default or `FAST`), applies the keys it
knows and reports the others (`report.dropped["usd.settings.<key>"]`). Its keys never touch
`outputs` or `performance`, and it is only applied when called (a stage load never changes the
settings).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp.scene.model.report import LoadReport
from raywarp.settings import FAST, RenderSettings

if TYPE_CHECKING:
  from pxr import Usd

# The look of Omniverse RTX RealTimePathTracing (Isaac Sim, Newton ViewerRTX) over the default
# (docs/settings.md "Recipes"): RTX's light transport choices (sun through window glass, its ACES
# approximation and exposure; the dome's sun a shadowed directional light, `SceneOptions.env_sun`,
# as every upload by default), per frame.
# Dotted changes of the default (applied per call: the default is read when the map runs).
_RTX_LOOK: Final[dict[str, object]] = {
  "gi.normal_bias": 0.3,
  "lighting.max_exact": None,
  "lighting.light_grid_cell": 0.5,
  "reflections.max_exact": None,
  "reflections.shadows": "sun",
  "shadows.max_casters": 8,
  "shadows.translucent": "clear",
  "post.tone_mapper": "aces_rtx",
}
# rtx:rendermode -> the changes of the default it starts from.
_MODES: Final[dict[str, Mapping[str, object]]] = {
  "RealTimePathTracing": _RTX_LOOK,
  "PathTracing": _RTX_LOOK,
  "RaytracedLighting": {},
  "MinimalRendering": FAST,
}
# rtx:post:tonemap:op (Kit's tonemapper enum, or its token) -> post.tone_mapper.
_TONE_MAPPERS: Final = {0: "clamp", 6: "aces_rtx", "Clamp": "clamp", "Aces": "aces_rtx", "ACES": "aces_rtx"}
# RTX's camera defaults for the exposure keys: f/5, 1/50 s, ISO 100.
_EXPOSURE_DEFAULTS: Final = {"post:tonemap:fNumber": 5.0, "post:tonemap:cameraShutter": 50.0, "post:tonemap:filmIso": 100.0}
# Setting changes by dotted field name (`RenderSettings.replace`; None: the field's "unset", e.g.
# every light casting shadows).
type Updates = dict[str, bool | int | float | str | tuple[float, float, float] | None]
# An authored setting value (Python scalars, tokens, Gf vectors; this module never imports pxr
# at load time).
type UsdValue = object


def _float(v: UsdValue) -> float:
  """A numeric setting value (ValueError for anything else: the key is reported)."""
  if isinstance(v, (bool, int, float)):
    return float(v)
  return float(str(v))


def _tone_mapper(v: UsdValue, out: Updates) -> None:
  """post:tonemap:op (KeyError: a tone mapper the core does not have, reported)."""
  out["post.tone_mapper"] = _TONE_MAPPERS[v if isinstance(v, str) else int(_float(v))]


def _background(v: UsdValue, out: Updates) -> None:
  """background:source:type: the dome (sky) or a constant color behind the scene."""
  kind = str(v)
  if kind in ("domeLight", "sky", "0"):
    out["environment.background"] = True
  elif kind in ("color", "1"):
    out["environment.background"] = False
  else:
    raise KeyError(kind)


def _background_color(v: UsdValue, out: Updates) -> None:
  """background:source:color."""
  r, g, b = (float(c) for c in np.asarray(v, np.float64).reshape(3))
  out["environment.background_color"] = (r, g, b)


# RTX keys copied to a setting field, converted by type.
_FIELDS: Final = {
  "shadows:enabled": ("shadows.enabled", bool),
  "reflections:enabled": ("reflections.enabled", bool),
  "rtpt:maxRoughness": ("reflections.max_roughness", float),
  "translucency:enabled": ("transparency.enabled", bool),
  "indirectDiffuse:enabled": ("gi.enabled", bool),
  "directLighting:sampledLighting:samplesPerPixel": ("shadows.samples", int),
  "directLighting:sampledLighting:maxLights": ("shadows.max_casters", int),
  "hydra:faceCulling:enabled": ("surfaces.backface_culling", bool),
}
# RTX keys with their own conversion.
_KEYS: Final = {
  "post:tonemap:op": _tone_mapper,
  "background:source:type": _background,
  "background:source:color": _background_color,
}


def _apply(key: str, v: UsdValue, out: Updates) -> None:
  """Applies RTX key `key` (KeyError: one the mapping does not have, or a value it does not map)."""
  if key in _FIELDS:
    field, kind = _FIELDS[key]
    value = bool(v) if kind is bool else kind(_float(v))
    # RTX's 0 lights means every light (the setting's None).
    out[field] = None if field == "shadows.max_casters" and value <= 0 else value
  else:
    _KEYS[key](v, out)


def _ev100(keys: dict[str, UsdValue], report: LoadReport) -> float | None:
  """EV100 of the camera exposure keys, taken out of `keys` (None when none is usable).

  A key that is not a positive finite number is reported and left at RTX's default.
  """
  exposure: dict[str, float] = {}
  for key in _EXPOSURE_DEFAULTS:
    if key not in keys:
      continue
    try:
      value = _float(keys.pop(key))
    except (TypeError, ValueError):
      value = math.nan
    if math.isfinite(value) and value > 0.0:
      exposure[key] = value
    else:
      report.drop(f"usd.settings.{key}")
  if not exposure:
    return None
  # EV100 = log2(N^2 / t) - log2(ISO / 100), t = 1 / cameraShutter.
  n, shutter, iso = ({**_EXPOSURE_DEFAULTS, **exposure}[k] for k in _EXPOSURE_DEFAULTS)
  return math.log2(n * n * shutter) - math.log2(iso / 100.0)


def authored(stage: Usd.Stage) -> dict[str, UsdValue]:
  """The stage's RTX settings by key without the `rtx:` prefix (prims override the layer)."""
  from pxr import UsdRender  # noqa: PLC0415  # pxr on the first stage read only

  keys: dict[str, UsdValue] = {}
  layer = stage.GetRootLayer().customLayerData.get("renderSettings", {})
  for k, v in layer.items():
    if k.startswith("rtx:"):
      keys[k[4:]] = v
  for prim in stage.Traverse():
    if prim.IsA(UsdRender.Settings) or prim.IsA(UsdRender.Product):
      for attr in prim.GetAuthoredAttributes():
        name = attr.GetName()
        value = attr.Get() if name.startswith("omni:rtx:") else None
        if value is not None:
          keys[name[9:]] = value
  return keys


def settings(
  stage: str | os.PathLike[str] | Usd.Stage, base: RenderSettings | None = None, report: LoadReport | None = None
) -> RenderSettings:
  """RenderSettings of a stage's authored Omniverse RTX settings over `base`.

  Args:
    stage: the stage (path or Usd.Stage).
    base: the settings the stage's keys apply to (None: the base of `rtx:rendermode`,
      the RTX look recipe of docs/settings.md when unset).
    report: receives the keys that are ignored (`usd.settings.<key>`).
  """
  from pxr import Usd  # noqa: PLC0415  # pxr on the first stage read only

  from raywarp.formats.usd.util import open_stage  # noqa: PLC0415  # the readers import pxr

  if not isinstance(stage, Usd.Stage):
    stage = open_stage(stage)
  keys = authored(stage)
  report = report if report is not None else LoadReport()
  mode = str(keys.pop("rendermode", "RealTimePathTracing"))
  if base is None:
    base = RenderSettings().replace(_MODES.get(mode, _RTX_LOOK))
    if mode not in _MODES:
      report.drop(f"usd.settings.rendermode.{mode}(rtx_look)")
  updates: Updates = {}
  ev100 = _ev100(keys, report)
  if ev100 is not None:
    updates |= {"exposure.mode": "manual", "exposure.ev100": ev100}
  for key, value in sorted(keys.items()):
    one: Updates = {}
    try:
      _apply(key, value, one)
      base.replace(**one)  # validates the key's values: an invalid one is dropped, not fatal
      updates |= one
    except (KeyError, TypeError, ValueError):
      report.drop(f"usd.settings.{key}")
  return base.replace(**updates)
