"""Validation of the canonical UsdLux records (`lights.Light`, `lights.EnvironmentLight`):
ValueError naming the record, the field, its value and the expected range, before a record enters a
scene."""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from raywarp.scene.lighting.blackbody import F32_MAX, scaled_colors
from raywarp.scene.lighting.ies_profile import profile
from raywarp.scene.model import checks
from raywarp.scene.model.lights import IES_MODES, MAX_ANGLE_DEG, EnvironmentLight, Light, LightFrame, Lights, Shaping
from raywarp.scene.model.types import EnvFormat, LightKind

# Exposures span at most this many stops either way (2^128 x a float32 color overflows float32).
MAX_EXPOSURE: Final = 128.0
# Light frames whose axes' Gram matrix is this close to the identity are orthonormal (float32
# frames from float64 sources).
_ORTHONORMAL_TOL: Final = 1.0e-4


def check_light(light: Light, *, ntexture: int, ninstance: int, where: str) -> None:
  """Validates a light record: photometric inputs (`_check_photometric`), lobes, sizes (a rect
  with area), angles [0, 180] deg, shaping, an orthonormal finite xform, ids that exist, a positive
  `ext.range`, an absolute intensity, a known falloff.

  Raises:
    ValueError: naming the light, the field, its value and the expected range.
  """
  kind = checks.member("kind", light.kind, LightKind, where=where)
  at = f"{where} ({kind.name.lower()} light)"
  peak = _check_photometric(light, at)
  for name, value in (("diffuse", light.diffuse), ("specular", light.specular)):
    checks.in_range(name, value, at, low=0.0)
  checks.positive("meters_per_unit", light.meters_per_unit, at)
  for name, size in (("radius", light.radius), ("width", light.width), ("height", light.height), ("length", light.length)):
    checks.in_range(f"{name} [m]", size, at, low=0.0)
  if kind == LightKind.RECT and light.width * light.height <= 0.0:
    raise ValueError(f"{at}: a rect light needs a positive width and height, got {light.width} x {light.height}")
  if not peak * 4.0 * math.pi * max(light.radius, light.width, light.height, light.length) ** 2 <= F32_MAX:
    raise ValueError(f"{at}: its radiance {peak:g} x its area (the point intensity) exceeds float32 ({F32_MAX:g})")
  checks.in_range("angle [deg]", light.angle, at, low=0.0, high=MAX_ANGLE_DEG)
  if light.shaping is not None:
    _check_shaping(light.shaping, at)
  _check_frame(light.xform, at)
  checks.index("texture", light.texture, ntexture, where=at)
  checks.index("instance", light.instance, ninstance, where=at)
  if light.ext.range is not None:
    checks.positive("ext.range [m]", light.ext.range, at)
  if light.ext.relative:
    raise ValueError(f"{at}: a display-relative light (no intensity given) is added through `Lighting`, which resolves it")
  if light.ext.falloff not in Lights.FALLOFFS:
    raise ValueError(f"{at}: ext.falloff must be one of {Lights.FALLOFFS}, got {light.ext.falloff!r}")


def _check_shaping(shaping: Shaping, where: str) -> None:
  """Checks a ShapingAPI record: cone half angle in [0, 180] deg, softness in [0, 1], focus >= 0,
  a finite non-negative focus tint.

  Raises:
    ValueError: for a value out of range.
  """
  checks.in_range("shaping.cone_angle (half angle) [deg]", shaping.cone_angle, where, low=0.0, high=MAX_ANGLE_DEG)
  checks.in_range("shaping.cone_softness", shaping.cone_softness, where, low=0.0, high=1.0)
  checks.in_range("shaping.focus", shaping.focus, where, low=0.0)
  checks.color("shaping.focus_tint", shaping.focus_tint, where)
  checks.finite("shaping.ies_angle_scale", shaping.ies_angle_scale, where)
  if shaping.ies_mode not in IES_MODES:
    raise ValueError(f"{where}: shaping.ies_mode must be one of {IES_MODES}, got {shaping.ies_mode!r}")
  if shaping.ies:
    try:
      profile(shaping.ies)
    except ValueError as e:
      raise ValueError(f"{where}: shaping.ies: {e}") from None


def _check_frame(frame: LightFrame, where: str) -> None:
  """Checks a light frame: (3, 4), finite, orthonormal axes (sizes are given in world units).

  Raises:
    ValueError: for another shape, a NaN / inf or axes that are not orthonormal.
  """
  m = checks.finite_array("xform", frame, where, shape=(3, 4))
  axes = m[:, :3]
  if not np.allclose(axes.T @ axes, np.eye(3), atol=_ORTHONORMAL_TOL):
    raise ValueError(f"{where}: xform axes must be orthonormal (sizes are world sizes), got {axes.tolist()}")


def _check_photometric(record: Light | EnvironmentLight, where: str) -> float:
  """The peak radiance scale (intensity x 2^exposure x color x blackbody) of checked photometric
  inputs: intensity >= 0, |exposure| <= MAX_EXPOSURE stops, color >= 0, temperature > 0, and the
  scale finite in float32 (the device's precision).

  Raises:
    ValueError: for a value out of range, or a radiance scale beyond float32.
  """
  checks.in_range("intensity", record.intensity, where, low=0.0)
  checks.in_range("exposure [stops]", record.exposure, where, low=-MAX_EXPOSURE, high=MAX_EXPOSURE)
  color = checks.color("color", record.color, where)
  checks.positive("color_temperature [K]", record.color_temperature, where)
  scale = scaled_colors(
    color.reshape(1, 3),
    intensity=np.array([record.intensity]),
    exposure=np.array([record.exposure]),
    kelvin=np.array([record.color_temperature]),
    enabled=np.array([record.enable_color_temperature]),
  )
  if not float(scale.max()) <= F32_MAX:
    raise ValueError(f"{where}: intensity x 2^exposure x color = {scale.ravel().tolist()} exceeds float32 ({F32_MAX:g})")
  return float(scale.max())


def check_environment(env: EnvironmentLight, *, ntexture: int, where: str) -> None:
  """Validates an environment record (finite intensity / exposure, color >= 0, temperature > 0,
  a finite rotation, a texture that exists, a known format).

  Raises:
    ValueError: naming the environment, the field, its value and the expected range.
  """
  checks.index("texture", env.texture, ntexture, where=where)
  _check_photometric(env, where)
  checks.finite_array("rotation", env.rotation, where, shape=(3, 3))
  checks.member("format", env.format, EnvFormat, where=where)
  checks.in_range("illuminance [lux]", env.illuminance, where, low=0.0)
