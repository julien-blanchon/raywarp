"""Omniverse's codeless physical-lighting schemas -> the canonical UsdLux record, converted once.

* `PhysicalLightIlluminantAPI` (applied by each photometric schema too): `physical:illuminant`
  "white" (the UsdLux colour as is), "blackbody" (`colorTemperature` without needing
  `enableColorTemperature`: UsdLux's table), "illuminant D" (the CIE daylight series at
  `colorTemperature`, folded into the colour), "custom" (`physical:customIlluminant` (nm, value)
  pairs, integrated to a colour). It also makes an IES profile's candela values the light's
  intensity (`Shaping.ies_mode` "measured") unless `ies:normalize`.
* `PhotometricAreaLightAPI` (rect, sphere, disk, cylinder): `photometric:illuminance` [lux] at
  `photometric:illuminance:distance` [m] when both are positive, else `photometric:power` [lm]
  (schema fallback 1600 lm): the light's `intensity` is solved from its world-size record
  (`raywarp.scene.lighting.photometry`; the flux into a cone: "reflector"); an IES profile then
  only shapes.
* `PhotometricDistantLightAPI`: `photometric:illuminance` [lux] (fallback 10000) on a surface
  perpendicular to the light, the canonical illuminance itself.
* `PhotometricDomeLightAPI`: `photometric:illuminance` [lux] (fallback 10000) on an upward-facing
  surface (`EnvironmentLight.illuminance`: the map scaled at upload,
  `raywarp.core.pipeline.upload.environment`).

Both targets keep the chromaticity of the colour and illuminant; the exposure is part of the
solved value (the target is met exactly).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, final

import numpy as np
from pxr import Usd

from raywarp.formats.usd import util
from raywarp.formats.usd.util import Time
from raywarp.scene.images.colorspace import REC709_Y
from raywarp.scene.lighting import photometry
from raywarp.scene.lighting.blackbody import scaled_colors
from raywarp.scene.model.lights import EnvironmentLight, Light
from raywarp.scene.model.report import LoadReport

ILLUMINANT_API: Final = "PhysicalLightIlluminantAPI"
AREA_API: Final = "PhotometricAreaLightAPI"
DISTANT_API: Final = "PhotometricDistantLightAPI"
DOME_API: Final = "PhotometricDomeLightAPI"
_APIS: Final = (ILLUMINANT_API, AREA_API, DISTANT_API, DOME_API)
# Schema fallbacks (Omniverse docs).
AREA_POWER_LM: Final = 1600.0
DISTANT_LUX: Final = 10000.0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Target:
  """A photometric target of a light, solved when its world size is known (`resolve`).

  Attributes:
    power: luminous flux [lm] (None: the illuminance target).
    illuminance: [lux] at `distance` [m] (used when power is None).
  """

  power: float | None
  illuminance: float = 0.0
  distance: float = 0.0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PhotometricLight:
  """A light read with a photometric target (`lux.read_light`; `lux.emit_light` solves it)."""

  light: Light
  target: Target


def applied(prim: Usd.Prim) -> set[str]:
  """The API schemas applied to `prim`, registered or codeless (from its `apiSchemas` list op)."""
  names = set(prim.GetAppliedSchemas())
  ops = prim.GetMetadata("apiSchemas")
  if ops is not None:
    for items in (ops.explicitItems, ops.prependedItems, ops.appendedItems, ops.addedItems):
      names.update(str(i) for i in items)
  return names


def illuminant_api(prim: Usd.Prim) -> bool:
  """Whether the prim has PhysicalLightIlluminantAPI (alone or through a photometric schema)."""
  return bool(applied(prim) & set(_APIS))


def _luminance_of(record: Light | EnvironmentLight) -> float:
  """Rec.709 luminance of intensity 1 x 2^exposure x colour x blackbody of a record."""
  rgb = scaled_colors(
    np.array([record.color], np.float64),
    intensity=np.array([1.0]),
    exposure=np.array([record.exposure]),
    kelvin=np.array([record.color_temperature]),
    enabled=np.array([record.enable_color_temperature]),
  )[0]
  return float(np.dot(rgb, REC709_Y))


def _illuminant[R: (Light, EnvironmentLight)](report: LoadReport, prim: Usd.Prim, record: R, time: Time) -> R:
  """`record` with the `physical:illuminant` colour folded in (module doc)."""
  mode = util.token_value(prim, "physical:illuminant", "white", time)
  if mode == "white":
    return record
  if mode == "blackbody":
    return dataclasses.replace(record, enable_color_temperature=True)
  try:
    if mode == "illuminant D":
      lo, hi = photometry.DAYLIGHT_RANGE
      kelvin = min(max(record.color_temperature, lo), hi)
      if kelvin != record.color_temperature:
        report.approx("physical:illuminant.D.clamped", prim)
      tint = photometry.cct_color(kelvin, "daylight")
    elif mode == "custom":
      pairs = np.asarray(prim.GetAttribute("physical:customIlluminant").Get(time), np.float64).reshape(-1, 2)
      tint = photometry.spectrum_color(pairs[:, 0], pairs[:, 1])
    else:
      report.drop(f"physical:illuminant.{mode}", prim)
      return record
  except (ValueError, TypeError) as e:
    report.drop("physical:illuminant.invalid", f"{prim.GetPath()}: {e}")
    return record
  color = tuple(float(c * t) for c, t in zip(record.color, tint, strict=True))
  return dataclasses.replace(record, color=(color[0], color[1], color[2]), enable_color_temperature=False)


def read[R: (Light, EnvironmentLight)](report: LoadReport, prim: Usd.Prim, record: R, time: Time) -> R | PhotometricLight:
  """The record of a UsdLux prim with its physical-lighting schemas applied: the illuminant
  colour, a distant light's illuminance, or an area light's target (`PhotometricLight`)."""
  apis = applied(prim)
  if not apis & set(_APIS):
    return record
  record = _illuminant(report, prim, record, time)
  if isinstance(record, EnvironmentLight):
    if DOME_API in apis:
      lux = util.float_value(prim, "photometric:illuminance", DISTANT_LUX, time)
      return dataclasses.replace(record, illuminance=lux if math.isfinite(lux) and lux > 0.0 else 0.0)
    return record
  if DISTANT_API in apis:
    lux = util.float_value(prim, "photometric:illuminance", DISTANT_LUX, time)
    return dataclasses.replace(record, intensity=lux / max(_luminance_of(record), 1e-30))
  if AREA_API not in apis:
    return record
  lux = util.float_value(prim, "photometric:illuminance", 0.0, time)
  distance = util.float_value(prim, "photometric:illuminance:distance", 0.0, time)
  if lux > 0.0 and distance > 0.0:
    return PhotometricLight(light=record, target=Target(power=None, illuminance=lux, distance=distance))
  power = util.float_value(prim, "photometric:power", AREA_POWER_LM, time)
  return PhotometricLight(light=record, target=Target(power=power if math.isfinite(power) else AREA_POWER_LM))


def resolve(light: Light, target: Target) -> Light:
  """`light` (world sizes) with the intensity meeting `target`; an IES profile only shapes it.

  Raises:
    ValueError: for a light that cannot meet it (no flux, nothing along its axis, a negative
      target).
  """
  if light.shaping is not None and light.shaping.ies:
    light = dataclasses.replace(light, shaping=dataclasses.replace(light.shaping, ies_mode="mask"))
  if target.power is not None:
    return photometry.with_power(light, target.power)
  return photometry.with_illuminance(light, target.illuminance, target.distance)
