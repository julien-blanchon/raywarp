"""UsdLux lights -> the canonical `Light` / `EnvironmentLight` records (UsdLux inputs one-to-one).

The records keep UsdLux's inputs and units (`raywarp.scene.model.lights`): `intensity * 2^exposure
* color * blackbody(colorTemperature)` is the surface radiance [nits] of an area light, or the
illuminance [lux] of a DistantLight, derived at upload
(`raywarp.core.pipeline.upload.lights.derive_lights`).
With `normalize`, the upload divides the radiance by the light's world-space surface area in stage
units (`meters_per_unit` is the stage's), as the UsdLux spec says, so the emitted power does not
depend on the light's size (Omniverse RTX agrees). Sizes are scaled to world metres by the light's
world transform (including metersPerUnit).

Two dialects differ (`dialect`): the UsdLux spec (the default) and Omniverse RTX's legacy one
(`omni:rtx:usdluxVersion` < 2505, which Kit assumes when the attribute is absent). In the legacy
dialect a DistantLight ignores `normalize` (its intensity is always lux) and a DomeLight's pole is
its local +Z; in the spec, an unnormalized DistantLight's intensity is a radiance (lux = intensity
* pi sin^2(angle / 2)) and a DomeLight's pole is +Y with OpenEXR's latlong layout. A light reads
in the legacy dialect when its authored version says so, or, without one, when it is authored in
a Kit layer (Kit's `customLayerData`), or carries Kit's attribute naming (reported as ambiguous).

A light prim is read once (`read_light`: attributes, report entries) and placed under each of its
instancing transforms (`emit_light`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import warnings
from typing import Final, TypedDict

import numpy as np
from pxr import Sdf, Usd, UsdLux

from raywarp.formats.usd import util
from raywarp.formats.usd.options import Dialect
from raywarp.formats.usd.stage import dome_formats, physical
from raywarp.formats.usd.stage.dome_formats import EXR_LATLONG
from raywarp.formats.usd.stage.physical import PhotometricLight
from raywarp.formats.usd.util import Matrix3, Matrix4, Time, Vec3
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.images import decode, files
from raywarp.scene.images.colorspace import ColorSpace, auto_colorspace
from raywarp.scene.lighting import ies_profile
from raywarp.scene.lighting.blackbody import scaled_colors
from raywarp.scene.model import ies as ies_lib
from raywarp.scene.model.lights import MAX_ANGLE_DEG, EnvironmentLight, Light, Shaping
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import NO_TEXTURE, EnvFormat, LightKind, TextureId, Wrap
from raywarp.scene.model.xform import orthonormalize

LIGHT_TYPES: Final = {
  "DistantLight": LightKind.DISTANT,
  "SphereLight": LightKind.SPHERE,
  "RectLight": LightKind.RECT,
  "DiskLight": LightKind.DISK,
  "CylinderLight": LightKind.CYLINDER,
}
DOME_TYPES: Final = ("DomeLight", "DomeLight_1")


def input_names(name: str) -> tuple[str, str]:
  """The attribute names of light input `name`: with the `inputs:` prefix of UsdLux >= 21.02,
  then without."""
  return f"inputs:{name}", name


# Attributes read (shadow and shaping inputs: under `_IGNORED_PREFIXES`).
_CONSUMED: Final = {
  attr
  for name in (
    *("intensity", "exposure", "color", "enableColorTemperature", "colorTemperature", "normalize", "diffuse"),
    *("specular", "radius", "width", "height", "length", "angle", "treatAsPoint", "treatAsLine", "texture:file"),
    *("texture:format", "poleAxis"),
  )
  for attr in input_names(name)
} | {"visibility", "purpose", "extent", "xformOpOrder", "visibleInPrimaryRay"}
# Omniverse RTX's camera visibility of a light's surface (no UsdLux input: un-namespaced, false
# by default for every light but domes and distant lights, per the ovrtx lights documentation).
_PRIMARY_RAY: Final = "visibleInPrimaryRay"
# The inputs a DomeLight's environment reads (`_photometric`, `_dome`, `_dome_frame`).
_DOME_CONSUMED: Final = {
  attr
  for name in (
    *("intensity", "exposure", "color", "enableColorTemperature", "colorTemperature"),
    *("texture:file", "texture:format", "poleAxis"),
  )
  for attr in input_names(name)
} | {"visibility", "purpose", "extent", "xformOpOrder"}
# Namespaces of authored attributes that are not light inputs (or are reported one by one).
_IGNORED_PREFIXES: Final = (
  "xformOp:",
  "omni:",
  "physics:",
  "collection:",
  "light:",
  "shadow:",
  "inputs:shadow:",
  "inputs:shaping:",
  "shaping:",
  "physical:",
  "photometric:",
)
# Map +Z (the core's environment pole) -> local +Y: a -90 degree turn about X.
_POLE_Y: Final = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# Omniverse's latlong has the core's handedness (longitude growing to the left of the image)
# with the image centre column towards -x: a half turn about the pole (checked against RTX
# renders of a tilted sunflowers dome, sun included).
_HALF_TURN: Final = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
# The UsdLux spec's latlong is OpenEXR's (`ImfEnvmap.cpp`): pole +Y, centre column +Z, u = 0.25
# towards +X (`dome_formats.EXR_LATLONG`, in the dome's local frame).
# DomeLight_1 `poleAxis = "Z"`: the +Y frame turned +90 degrees about X (+Y -> +Z), as Hydra does.
_EXR_Z: Final = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]) @ EXR_LATLONG
# The first RTX UsdLux version that follows the spec (`omni:rtx:usdluxVersion`), and the legacy
# version Kit assumes when the attribute is absent.
_RTX_SPEC_VERSION: Final = 2505.0
_RTX_LEGACY_VERSION: Final = 2411.0
_RTX_VERSION_ATTR: Final = "omni:rtx:usdluxVersion"
# customLayerData keys every layer saved by Omniverse Kit carries.
_KIT_LAYER_KEYS: Final = frozenset({"omni_layer", "renderSettings", "cameraSettings"})
# The light inputs whose pre-21.02 names (without `inputs:`) Kit still writes next to the new ones.
_LIGHT_INPUTS: Final = ("intensity", "exposure", "color", "angle", "radius", "width", "height", "texture:file")
# pxr's blackbody table spans these temperatures [K] (it clamps outside: lossless here).
_KELVIN_RANGE: Final = (1000.0, 10000.0)


def _get_float(prim: Usd.Prim, name: str, default: float, time: Time) -> float:
  """Float input `name` with or without the `inputs:` prefix."""
  return util.float_value(prim, input_names(name), default, time)


def _get_bool(prim: Usd.Prim, name: str, *, default: bool, time: Time) -> bool:
  """Bool input `name` with or without the `inputs:` prefix."""
  return util.bool_value(prim, input_names(name), default=default, time=time)


def dialect(report: LoadReport, prim: Usd.Prim, time: Time, default: Dialect | None) -> Dialect:
  """The dialect of a light prim: its authored `omni:rtx:usdluxVersion`, else `default` (None:
  detected): RTX legacy when the layer with its strongest opinion is a Kit layer, else the spec.
  Weaker Kit evidence reads as RTX legacy but is ambiguous (reported and warned): another layer
  of the light (or the stage's root) saved by Kit, or Kit's attribute naming (`omni:` inputs, or
  pre-21.02 names next to the `inputs:` ones, which Kit writes both). Pre-21.02 names alone read
  as the spec, also reported."""
  version = _rtx_version(report, prim, time)
  if version is not None:
    return Dialect.SPEC if version >= _RTX_SPEC_VERSION else Dialect.RTX_LEGACY
  if default is not None:
    return default
  if _kit_layer(_defining_layer(prim)):
    report.count("UsdLux.dialect.rtx_legacy(kit layer)")
    return Dialect.RTX_LEGACY
  layers = [spec.layer for spec in prim.GetPrimStack()] + [prim.GetStage().GetRootLayer()]
  if any(_kit_layer(layer) for layer in layers):
    _ambiguous(report, prim, "a Kit layer composes a light authored in another layer: read as RTX legacy")
    return Dialect.RTX_LEGACY
  unprefixed = [n for n in _LIGHT_INPUTS if util.authored(prim, n)]
  kit_named = any(util.authored(prim, f"inputs:{n}") for n in unprefixed) or any(
    a.GetName().startswith("omni:") for a in prim.GetAuthoredAttributes()
  )
  if kit_named:
    _ambiguous(report, prim, "Kit attribute naming without Kit layer metadata: read as RTX legacy")
    return Dialect.RTX_LEGACY
  if unprefixed:
    _ambiguous(report, prim, "pre-21.02 input names: read as the UsdLux spec")
  return Dialect.SPEC


def _rtx_version(report: LoadReport, prim: Usd.Prim, time: Time) -> float | None:
  """The light's authored `omni:rtx:usdluxVersion` (None: none, or not a finite number:
  reported)."""
  if not util.authored(prim, _RTX_VERSION_ATTR):
    return None
  try:
    version = util.float_value(prim, _RTX_VERSION_ATTR, _RTX_LEGACY_VERSION, time)
  except TypeError:
    version = math.nan
  if math.isfinite(version):
    return version
  report.approx("UsdLux.usdluxVersion.invalid(ignored)", prim)
  return None


def _ambiguous(report: LoadReport, prim: Usd.Prim, why: str) -> None:
  """Reports (each light) and warns (once per reason) a light whose dialect is inferred from weak
  evidence."""
  report.approx(f"UsdLux.dialect.ambiguous({why})", prim)
  # The message names only the reason, so Python's default filter shows it once per reason.
  warnings.warn(f"UsdLux light dialect ambiguous: {why} (see the load report)", UserWarning, stacklevel=3)


def _defining_layer(prim: Usd.Prim) -> Sdf.Layer | None:
  """The layer of the strongest opinion on the light's intensity, else of its strongest spec."""
  for name in input_names("intensity"):
    attr = prim.GetAttribute(name)
    if attr and attr.HasAuthoredValue():
      stack = attr.GetPropertyStack(Usd.TimeCode.Default())
      if stack:
        return stack[0].layer
  specs = prim.GetPrimStack()
  return specs[0].layer if specs else None


def _kit_layer(layer: Sdf.Layer | None) -> bool:
  """Whether `layer` was saved by Omniverse Kit (its customLayerData keys)."""
  return layer is not None and not _KIT_LAYER_KEYS.isdisjoint(layer.customLayerData)


class _Photometric(TypedDict):
  """The UsdLux inputs every light and dome has (`LightAPI`): `Light` / `EnvironmentLight`
  keywords."""

  intensity: float
  exposure: float
  color: tuple[float, float, float]
  enable_color_temperature: bool
  color_temperature: float


def _photometric(report: LoadReport, prim: Usd.Prim, time: Time) -> _Photometric:
  """A light's intensity, exposure, color and color temperature; with `enableColorTemperature`, a
  temperature outside pxr's 1000-10000 K table is clamped to it (reported; NaN: 6500 K)."""
  enabled = _get_bool(prim, "enableColorTemperature", default=False, time=time)
  kelvin = _get_float(prim, "colorTemperature", 6500.0, time)
  if not _KELVIN_RANGE[0] <= kelvin <= _KELVIN_RANGE[1]:
    kelvin = _clamped(report if enabled else None, prim, "UsdLux.colorTemperature.clamped", kelvin, _KELVIN_RANGE)
  return _Photometric(
    intensity=_get_float(prim, "intensity", 1.0, time),
    exposure=_get_float(prim, "exposure", 0.0, time),
    color=util.color_value(prim, input_names("color"), (1.0, 1.0, 1.0), time),
    enable_color_temperature=enabled,
    color_temperature=6500.0 if math.isnan(kelvin) else kelvin,
  )


def _clamped(report: LoadReport | None, prim: Usd.Prim, key: str, value: float, bounds: tuple[float, float]) -> float:
  """`value` clamped to `bounds` (NaN stays NaN: the caller's default), reported as `key`
  unless `report` is None (an input the light does not use)."""
  if report is not None:
    report.approx(key, prim)
  return value if math.isnan(value) else min(max(value, bounds[0]), bounds[1])


def radiance(report: LoadReport, prim: Usd.Prim, time: Time) -> Vec3:
  """Intensity * 2^exposure * color * blackbody(colorTemperature) (linear RGB): what a
  MeshLightAPI surface emits (the area lights' records derive it at upload)."""
  p = _photometric(report, prim, time)
  rgb = scaled_colors(
    np.array([p["color"]]),
    intensity=np.array([p["intensity"]]),
    exposure=np.array([p["exposure"]]),
    kelvin=np.array([p["color_temperature"]]),
    enabled=np.array([p["enable_color_temperature"]]),
  )
  return rgb[0]


def read_light(  # noqa: PLR0913  # the reader's context, then the stage's frame keyword-only
  builder: SceneBuilder,
  report: LoadReport,
  prim: Usd.Prim,
  *,
  time: Time,
  up_z: bool,
  mpu: float,
  lux_dialect: Dialect | None,
) -> Light | EnvironmentLight | PhotometricLight | None:
  """The light (or dome) of a UsdLux prim, its attributes read and reported once (None: dropped);
  `mpu`: the stage's metersPerUnit; `lux_dialect`: `usd.Options.lux_dialect`. A light's sizes
  are its local ones (`emit_light` scales them to world metres, then solves a photometric target:
  `physical`)."""
  type_name = prim.GetTypeName()
  p = _photometric(report, prim, time)
  if type_name in DOME_TYPES:
    _report_unread_dome(report, prim, type_name)
    dome = _dome(builder, report, prim, p, time=time, frame=_dome_frame(report, prim, time, up_z=up_z, default=lux_dialect))
    return physical.read(report, prim, dome, time)
  kind = LIGHT_TYPES.get(type_name)
  if kind is None:
    report.drop(f"light.{type_name}", prim)
    return None
  _report_unread(report, prim, kind, type_name)
  angle = _angle(report, prim, kind, time)
  normalize = _get_bool(prim, "normalize", default=False, time=time)
  if kind == LightKind.DISTANT and not normalize and dialect(report, prim, time, lux_dialect) == Dialect.SPEC:
    p["intensity"] *= _distant_size_factor(angle)
  light = Light(
    kind=kind,
    **p,
    normalize=normalize,
    meters_per_unit=mpu,
    radius=_get_float(prim, "radius", 0.5, time),
    width=_get_float(prim, "width", 1.0, time),
    height=_get_float(prim, "height", 1.0, time),
    length=_get_float(prim, "length", 1.0, time),
    angle=angle,
    treat_as_point=kind == LightKind.SPHERE and _get_bool(prim, "treatAsPoint", default=False, time=time),
    treat_as_line=kind == LightKind.CYLINDER and _get_bool(prim, "treatAsLine", default=False, time=time),
    diffuse=_get_float(prim, "diffuse", 1.0, time),
    specular=_get_float(prim, "specular", 1.0, time),
    shaping=_shaping(builder, report, prim, time),
    shadow_enable=_get_bool(prim, "shadow:enable", default=True, time=time),
    visible_in_primary_ray=util.bool_value(prim, (_PRIMARY_RAY,), default=False, time=time),
    path=str(prim.GetPath()),
  )
  return physical.read(report, prim, light, time)


def _distant_size_factor(angle: float) -> float:
  """UsdLux's distant-light size factor: the illuminance [lux] at normal incidence per unit
  radiance of an unnormalized DistantLight of angular diameter `angle` [deg] in [0, 180]:
  pi sin^2(angle / 2), and 1 at zero diameter. The core still approximates the diffuse/specular
  response at the disc's central direction."""
  if angle == 0.0:
    return 1.0
  return math.pi * math.sin(math.radians(angle) * 0.5) ** 2


def _angle(report: LoadReport, prim: Usd.Prim, kind: LightKind, time: Time) -> float:
  """A DistantLight's `angle` (angular diameter [deg]); an authored value outside [0, 180]
  (SimReady thumbnail rigs author 260) is clamped and reported (NaN: the 0.53 default)."""
  angle = _get_float(prim, "angle", 0.53, time)
  if 0.0 <= angle <= MAX_ANGLE_DEG:
    return angle
  used = report if kind == LightKind.DISTANT else None
  angle = _clamped(used, prim, "DistantLight.angle.clamped", angle, (0.0, MAX_ANGLE_DEG))
  return 0.53 if math.isnan(angle) else angle


def _shaping(builder: SceneBuilder, report: LoadReport, prim: Usd.Prim, time: Time) -> Shaping | None:
  """ShapingAPI (None when not applied); the cone angle and softness clamped to UsdLux's ranges
  (reported; NaN: no cone, no softness). UsdLux documents a softness in [0, 1]; content authors
  more (80 on some Isaac spots): its effect saturates at 1 (the falloff starts at the axis).
  The IES profile (`_ies`) is read with its angle scale and normalisation."""
  shaped = ("shaping:cone:angle", "shaping:ies:file")
  if not (prim.HasAPI(UsdLux.ShapingAPI) or any(_authored_input(prim, name) for name in shaped)):
    return None
  angle = _get_float(prim, "shaping:cone:angle", 90.0, time)
  if not 0.0 <= angle <= MAX_ANGLE_DEG:
    angle = _clamped(report, prim, "ShapingAPI.coneAngle.clamped", angle, (0.0, MAX_ANGLE_DEG))
  softness = _get_float(prim, "shaping:cone:softness", 0.0, time)
  if not 0.0 <= softness <= 1.0:
    softness = _clamped(report, prim, "ShapingAPI.coneSoftness.clamped", softness, (0.0, 1.0))
  focus = _get_float(prim, "shaping:focus", 0.0, time)
  tint = util.color_value(prim, input_names("shaping:focusTint"), (0.0, 0.0, 0.0), time)
  if focus > 0.0 and tint != (0.0, 0.0, 0.0):
    report.drop("ShapingAPI.focusTint", prim)
  scale = _get_float(prim, "shaping:ies:angleScale", 0.0, time)
  return Shaping(
    cone_angle=MAX_ANGLE_DEG if math.isnan(angle) else angle,
    cone_softness=0.0 if math.isnan(softness) else softness,
    focus=focus,
    focus_tint=tint,
    ies=_ies(builder, report, prim, time),
    ies_angle_scale=0.0 if math.isnan(scale) else scale,
    ies_normalize=_get_bool(prim, "shaping:ies:normalize", default=False, time=time),
    ies_mode="measured" if physical.illuminant_api(prim) else "mask",
  )


def _authored_input(prim: Usd.Prim, name: str) -> bool:
  """Whether light input `name` is authored, under either of its names (`input_names`)."""
  return any(util.authored(prim, attr) for attr in input_names(name))


def _ies(builder: SceneBuilder, report: LoadReport, prim: Usd.Prim, time: Time) -> str:
  """The text of the light's `shaping:ies:file` ("" none; an unresolved or malformed file is
  reported and dropped). The file is a cache dependency of the scene."""
  if not _authored_input(prim, "shaping:ies:file"):
    return ""
  path = util.asset_value(prim, input_names("shaping:ies:file"), time)
  if not path:
    report.drop("ShapingAPI.ies.unresolved", prim)
    return ""
  builder.dependencies.add(path)
  if prim.GetTypeName() == "DistantLight":
    report.drop("DistantLight.ies(no angular profile)", prim)
    return ""
  try:
    return _read_ies(path, os.path.getmtime(path))
  except (ies_lib.IesError, ValueError, OSError) as e:
    report.drop("ShapingAPI.ies.invalid", f"{prim.GetPath()}: {e}")
    return ""


@functools.lru_cache(maxsize=64)
def _read_ies(path: str, mtime: float) -> str:
  """An IES file's checked text (`scene.model.ies.read`, then gridded once: `ies_profile.profile`),
  read once per (path, modification time): a fixture used by many lights."""
  del mtime
  text = ies_lib.read(path)
  ies_profile.profile(text)
  return text


def _report_unread(report: LoadReport, prim: Usd.Prim, kind: LightKind, type_name: str) -> None:
  """Reports the authored light inputs the core cannot represent."""
  if kind == LightKind.RECT and util.authored(prim, "inputs:texture:file"):
    report.drop("RectLight.texture", prim)
  for name in ("inputs:shadow:color", "inputs:shadow:distance", "inputs:shadow:falloff", "inputs:shadow:falloffGamma"):
    if util.authored(prim, name):
      report.drop(f"ShadowAPI.{name.split(':')[-1]}", prim)
  for rel in ("collection:lightLink:includes", "collection:lightLink:excludes", "collection:shadowLink:includes"):
    r = prim.GetRelationship(rel)
    if r and r.GetTargets():
      report.drop("UsdLux.light_linking", prim)
  for attr in util.valued_attributes(prim):
    name = attr.GetName()
    if name not in _CONSUMED and not name.startswith(_IGNORED_PREFIXES):
      report.drop(f"{type_name}.{name}", prim)


def _report_unread_dome(report: LoadReport, prim: Usd.Prim, type_name: str) -> None:
  """Reports the authored DomeLight inputs an environment cannot represent: diffuse / specular
  scales, shadow switch, guide radius, portals, light linking (it lights every surface, both
  lobes, and is not a shadow caster)."""
  for rel in ("collection:lightLink:includes", "collection:lightLink:excludes", "collection:shadowLink:includes", "portals"):
    r = prim.GetRelationship(rel)
    if r and r.GetTargets():
      report.drop("UsdLux.light_linking" if rel.startswith("collection:") else f"{type_name}.portals", prim)
  for attr in util.valued_attributes(prim):
    name = attr.GetName()
    if name not in _DOME_CONSUMED and not name.startswith(
      ("xformOp:", "omni:", "collection:", "light:", "physical:", "photometric:")
    ):
      report.drop(f"{type_name}.{name}", prim)


def _to_world(light: Light, scales: Vec3) -> Light:
  """A light read in local units with its sizes scaled to world metres by the transform's
  `scales` (the sizes its kind does not have are left: the upload ignores them)."""
  match light.kind:
    case LightKind.SPHERE:
      return dataclasses.replace(light, radius=light.radius * float(np.cbrt(np.prod(scales))))
    case LightKind.RECT:
      return dataclasses.replace(light, width=light.width * float(scales[0]), height=light.height * float(scales[1]))
    case LightKind.DISK:
      return dataclasses.replace(light, radius=light.radius * float(0.5 * (scales[0] + scales[1])))
    case LightKind.CYLINDER:
      length, radius = light.length * float(scales[0]), light.radius * float(0.5 * (scales[1] + scales[2]))
      return dataclasses.replace(light, length=length, radius=radius)
    case _:
      return light


def emit_light(  # noqa: PLR0913  # the reader's context, then where the light goes keyword-only
  builder: SceneBuilder,
  report: LoadReport,
  record: Light | EnvironmentLight | PhotometricLight,
  *,
  world: Matrix4,
  path: str,
  body: str | None,
) -> None:
  """Adds a light read by `read_light` at `world` (4, 4) (a photometric target solved with the
  light's world sizes)."""
  xform, scales = orthonormalize(world)
  target = None
  if isinstance(record, PhotometricLight):
    record, target = record.light, record.target
  try:
    if isinstance(record, EnvironmentLight):
      builder.add_environment(dataclasses.replace(record, rotation=xform[:, :3] @ record.rotation, path=path))
      return
    if record.kind != LightKind.DISTANT and not np.allclose(scales, scales[0], rtol=1e-3):
      report.approx("light.non_uniform_scale", path)
    light = dataclasses.replace(_to_world(record, scales), xform=xform, path=path, body=body)
    builder.add_light(physical.resolve(light, target) if target is not None else light)
  except ValueError as e:
    # One bad authored light (a negative intensity, a NaN color, ...) must not abort the stage.
    report.drop("light.invalid", f"{path}: {e}")


def _dome_frame(report: LoadReport, prim: Usd.Prim, time: Time, *, up_z: bool, default: Dialect | None) -> Matrix3:
  """The dome's map frame in its local frame (map -> local). DomeLight_1 `poleAxis` (or one
  authored on a DomeLight) names the pole ("scene": the stage's up axis); a DomeLight's pole is
  +Y in the spec, +Z in RTX legacy (whatever the stage's up axis: the Marbles sample (Y-up)
  turns its dome upright with a 270 degree turn about X, and its RTX render shows the HDRI's
  floor below a camera looking down).
  The spec's layout is OpenEXR's latlong, RTX legacy's the core's turned half about the pole."""
  legacy = dialect(report, prim, time, default) == Dialect.RTX_LEGACY
  axis = "Z" if legacy else "Y"
  # Houdini (Karma) authors DomeLight_1's poleAxis on a plain DomeLight: honoured there too.
  if prim.GetTypeName() == "DomeLight_1" or util.authored(prim, "poleAxis"):
    axis = util.token_value(prim, input_names("poleAxis"), "scene", time)
    if axis == "scene":
      axis = "Z" if up_z else "Y"
  if legacy:
    return (np.eye(3) if axis == "Z" else _POLE_Y) @ _HALF_TURN
  return _EXR_Z if axis == "Z" else EXR_LATLONG


def _dome(  # noqa: PLR0913  # the reader's context, then the dome's values keyword-only
  builder: SceneBuilder, report: LoadReport, prim: Usd.Prim, p: _Photometric, *, time: Time, frame: Matrix3
) -> EnvironmentLight:
  """A DomeLight's environment: latlong texture (its authored colorSpace, else by file type; other
  `texture:format`s resampled to latlong: `dome_formats`), photometric inputs and map frame
  `frame` (`rotation`: map frame -> the dome's local frame; `emit_light` turns it to world).

  Raises:
    ValueError: for a `texture:format` that is not UsdLux's or an image that cannot hold it.
  """
  file = util.asset_value(prim, input_names("texture:file"), time)
  texture = NO_TEXTURE
  if file:
    attr = next((a for a in map(prim.GetAttribute, input_names("texture:file")) if a and a.HasAuthoredValue()), None)
    cs = util.colorspace_of(attr, report=report, where=prim) or auto_colorspace(file, color=True)
    fmt = util.token_value(prim, input_names("texture:format"), "automatic", time)
    if fmt in dome_formats.LATLONG_FORMATS:
      texture = builder.add_texture(file, colorspace=cs, wrap=(Wrap.REPEAT, Wrap.CLAMP))
    else:
      texture = _resampled_texture(builder, report, prim, file, cs, fmt)
  elif _authored_input(prim, "texture:file"):
    report.drop("DomeLight.texture:file.unresolved", prim)
  # The GPU upload resamples what is not a yaw about +z (a Y-up stage's dome, a tilted one).
  return EnvironmentLight(
    texture=texture,
    **p,
    rotation=frame,
    format=EnvFormat.LATLONG,
    path=str(prim.GetPath()),
  )


def _resampled_texture(  # noqa: PLR0913  # the reader's context, then the texture's values
  builder: SceneBuilder, report: LoadReport, prim: Usd.Prim, file: str, cs: ColorSpace, fmt: str
) -> TextureId:
  """The texture of a dome image `file` of format `fmt` resampled to latlong (decoded once, here;
  the file is a cache dependency). An undecodable file: reported, no texture (as a latlong one).

  Raises:
    ValueError: for an unknown format or an image whose shape cannot hold it (no mapping is right).
  """
  builder.dependencies.add(files.container(file))
  texels, space, error = dome_formats.read_linear(file, cs)
  if texels is None:
    report.drop(decode.report_key(error), f"{file}: {error}")
    return NO_TEXTURE
  latlong = dome_formats.to_latlong(texels, fmt, where=f"DomeLight {prim.GetPath()}")
  return builder.add_texture(np.ascontiguousarray(latlong[..., :3]), colorspace=space, wrap=(Wrap.REPEAT, Wrap.CLAMP))
