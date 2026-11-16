"""MuJoCo lights -> the core's light model, following MuJoCo's own two light models.

MuJoCo's Filament renderer decides per model (`model_lights.cc`): when the lights' intensities
sum to more than 0 the model is **photometric**, otherwise it is a classic (OpenGL) model.

* Photometric (`intensity` > 0 somewhere): every light is physical, as in Filament: color
  `diffuse` (aliased, live), intensity `intensity` [candela; lux for directional lights], inverse
  square falloff windowed at `range`, `bulbradius` its emitter radius (soft shadows; with
  `lighting.area_lights="ltc"` a point light with a radius and a range > 0 is a sphere light;
  range 0, no falloff, keeps it a point light), a spot's inner cone 1 - `softness` of its
  `cutoff`. A light of intensity 0 is dark. The headlight is off and
  there is no ambient (Filament's).
* Classic (every `intensity` 0, MuJoCo's default): OpenGL lights mapped exactly. With default
  attenuation (1, 0, 0), a point/spot light has no distance falloff (`light.range` 0): it gives
  a facing surface illuminance pi / exposure at any distance (`lighting.ev100`). Nondefault
  constant/linear/quadratic attenuation is read live from the MuJoCo Warp model, including
  per-world rows. The headlight (`visual/headlight`) is the camera light of
  `lighting.camera_light`, its `ambient` plus the lights' `ambient` a constant ambient radiance
  (display-relative).

A classic model without lights and with its headlight off shows `NO_LIGHT_AMBIENT` (MuJoCo's own
renderers' fallback), unless `rw.Lighting` adds an environment or an ambient (the model is lit).

Deliberate differences from MuJoCo's OpenGL renderer (`mujoco.Renderer`), which draws a classic
model darker and with more contrast:
* its fixed-function pipeline drops point lights (`render_gl3.c`: only spot and directional
  lights are supported); MuJoCo Warp's renderer, Filament and raywarp light with them.
* it shows its lighting sums unencoded (no sRGB transfer): a surface lit at a fifth of full
  shows at 0.2 there, at about 0.48 here (display-encoded), and its unlit faces are near black.
* it has no indirect light (GI) and no automatic exposure.
Render a classic model with `exposure.mode="manual"` at the lighting's `ev100` and `gi.enabled`
off for MuJoCo Warp's direct look.

A classic spot's `exponent` sets its smooth cone's inner angle (`classic_cone_inner`: the falloff
crosses one half where cos^exponent does): MuJoCo's default exponent draws a pool on the floor.

Not rendered (recorded in the docs' compatibility checklist): `specular` (one color drives both
lobes). `type="image"` lights light the scene through the environment
(`raywarp.integrations.mujoco.environment`), not as lights (intensity 0 here).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, Literal, final

import mujoco
import numpy as np

from raywarp._base.types import LightShape, LightType
from raywarp.core.pipeline.gpu_scene import gpu_scene as core_scene
from raywarp.scene.lighting.rig import Headlight, Lighting
from raywarp.scene.model.lights import MAX_ANGLE_DEG, Light
from raywarp.scene.model.types import Column, LightKind

# The ambient a classic model without lights and without headlight shows (MuJoCo's OpenGL
# renderer's global ambient then, `render_gl3.c` initLights, and MuJoCo Warp's
# `NO_LIGHT_AMBIENT_FALLBACK`): it is never black.
NO_LIGHT_AMBIENT: Final = 0.3
# The light kinds an MJCF light holds (`add_lights`).
type MjcfKind = Literal["directional", "point", "spot"]


def photometric(mjm: mujoco.MjModel) -> bool:
  """Whether the model's lights are physical (Filament's rule: their intensities sum above 0)."""
  return bool(mjm.nlight) and float(np.sum(mjm.light_intensity)) > 0.0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LightTables:
  """The light tables MuJoCo Warp does not hold (`m` has no intensity, range or bulb radius).

  Attributes:
    intensity: (nlight,) candela (point, spot) or lux (directional, and lights without falloff).
    range: (nlight,) falloff window radius [m]; 0: no falloff.
    radius: (nlight,) emitter radius (directional: angular radius, 0 for MuJoCo's).
    shape: (nlight,) int32 LightShape.
    cone_inner: (nlight,) inner / outer spot cone angle: a photometric model's Filament
      1 - `softness`, a classic one's from the spots' `exponent` (`classic_cone_inner`); None
      without spots (the core's 0.75).
  """

  intensity: Column[np.float32]
  range: Column[np.float32]
  radius: Column[np.float32]
  shape: Column[np.int32]
  cone_inner: Column[np.float32] | None


def light_tables(mjm: mujoco.MjModel, exposure: float, *, off: bool = False) -> LightTables:
  """The core light tables of the model's lights (module docstring), at `exposure` (classic);
  `off`: every light at intensity 0 (a `Lighting(replace=True)`)."""
  kind = np.asarray(mjm.light_type)
  image = kind == int(LightType.IMAGE)
  directional = kind == int(LightType.DIRECTIONAL)
  bulb = np.asarray(mjm.light_bulbradius, np.float32)
  radius = np.where(directional, 0.0, bulb).astype(np.float32)
  cone_inner = None
  if photometric(mjm):
    intensity = np.asarray(mjm.light_intensity, np.float32).copy()
    light_range = np.asarray(mjm.light_range, np.float32).copy()
    # Range 0 is "no falloff", which the sphere (LTC) form cannot express (it would window the
    # light to nothing): such lights stay point lights (soft shadows from `radius` still).
    sphere = (kind == int(LightType.POINT)) & (bulb > 0.0) & (light_range > 0.0)
    shape = np.where(sphere, int(LightShape.SPHERE), int(LightShape.NONE)).astype(np.int32)
    if np.any(kind == int(LightType.SPOT)):
      cone_inner = (1.0 - np.asarray(mjm.light_softness, np.float32)).astype(np.float32)
  else:
    intensity = np.full(mjm.nlight, math.pi / exposure, np.float32)
    light_range = np.zeros(mjm.nlight, np.float32)
    shape = np.full(mjm.nlight, int(LightShape.NONE), np.int32)
    if np.any(kind == int(LightType.SPOT)):
      cone_inner = classic_cone_inner(np.asarray(mjm.light_exponent), np.asarray(mjm.light_cutoff))
  intensity[image] = 0.0  # environment lights (the kernels would light them as point lights)
  if off:
    intensity[:] = 0.0
  return LightTables(intensity=intensity, range=light_range, radius=radius, shape=shape, cone_inner=cone_inner)


def headlight(mjm: mujoco.MjModel, spec: Headlight | None = None, *, off: bool = False) -> core_scene.GpuHeadlight:
  """The camera headlight: `spec` when given (its diffuse color), else `visual/headlight` in a
  classic model, off in a photometric one or when the model's lighting is replaced (`off`)."""
  if spec is not None:
    r, g, b = (float(c) for c in np.asarray(spec.diffuse, np.float32))
    return core_scene.GpuHeadlight(active=True, color=(r, g, b), intensity=float(math.pi))
  vis = mjm.vis.headlight
  r, g, b = (float(c) for c in vis.diffuse)
  return core_scene.GpuHeadlight(
    active=bool(vis.active) and not photometric(mjm) and not off, color=(r, g, b), intensity=float(math.pi)
  )


def ambient(
  mjm: mujoco.MjModel, exposure: float, *, with_headlight: bool = True, fallback: bool = True
) -> tuple[float, float, float]:
  """Constant ambient radiance, display-relative: the classic model's headlight (unless
  `with_headlight` is False: a replaced headlight) and active lights' `ambient` (none in a
  photometric model); a classic model without lights and with its headlight off,
  `NO_LIGHT_AMBIENT` (unless `fallback` is False: light added from Python lights it)."""
  out = np.zeros(3, dtype=np.float32)
  if not photometric(mjm):
    if mjm.vis.headlight.active and with_headlight:
      out += mjm.vis.headlight.ambient
    elif mjm.nlight == 0 and with_headlight and fallback:
      out += NO_LIGHT_AMBIENT  # nothing lights the model: MuJoCo's fallback
    active = np.asarray(mjm.light_active, bool) & (np.asarray(mjm.light_type) != int(LightType.IMAGE))
    out += np.asarray(mjm.light_ambient, np.float32)[active].sum(axis=0) if mjm.nlight else 0.0
  r, g, b = (float(v) for v in out / exposure)
  return r, g, b


def add_lights(spec: mujoco.MjSpec, lighting: Lighting) -> Lighting:
  """Adds `lighting`'s lights to a MuJoCo spec (before `compile`) and returns the rest of it, for
  `rw.integrations.mujoco.scene(..., lighting=)`: MuJoCo Warp's light tables are the model's
  (aliased), so added lights must be MJCF lights: a `Light.sun` (directional), `Light.point` (point)
  or `Light.spot` (spot), without an IES profile.

  Each light keeps its meaning in the model's light model (module docstring): in a classic
  model (every intensity 0) a display-relative light is a classic light of its color; in a
  photometric one it gets the intensity that shows its color at `lighting.ev100` (pi / exposure
  lux; point and spot lights at their distance to the origin, at least 1 m).
  `lighting.replace` deletes the spec's lights and turns its headlight off first.

  A model without lights (after `replace` too) takes the light model of the added lights: classic
  when they are all display-relative, photometric when one has an intensity.

  Raises:
    ValueError: for a light MJCF cannot hold (a rect / disk / tube / UsdLux sphere of radiance,
      an IES profile, a soft or partial cone), or a physical light added to a classic model with
      lights: it would turn every classic light of the model photometric (dark).
  """
  if lighting.replace:
    for light in list(spec.lights):
      spec.delete(light)
    spec.visual.headlight.active = 0
  mapped = [_mjcf(light) for light in lighting.lights]
  physical = any(not light.ext.relative for light in lighting.lights)
  own = sum(float(light.intensity) for light in spec.lights)
  if physical and len(spec.lights) and own == 0.0:
    raise ValueError("add_lights: a classic MuJoCo model (every light intensity 0) takes display-relative lights only")
  # The model's light model after the addition (Filament's rule: intensities summing above 0).
  photometric = physical or own > 0.0
  for light, (kind, intensity) in zip(lighting.lights, mapped, strict=True):
    pos = np.asarray(light.xform[:, 3], np.float64)
    if light.ext.relative:
      intensity = 0.0
      if photometric:
        intensity = math.pi / lighting.exposure * (1.0 if kind == "directional" else max(float(pos @ pos), 1.0))
    spec.worldbody.add_light(
      type=_MJ_LIGHT_TYPES[kind],
      pos=pos.tolist(),
      dir=(-np.asarray(light.xform[:, 2], np.float64)).tolist(),
      diffuse=list(light.color),
      intensity=intensity,
      cutoff=light.shaping.cone_angle if light.shaping is not None else _MJ_POINT_CUTOFF,
      # Not MJCF's defaults (a 10 m falloff window, softness 0.2): the inverse square law at every
      # distance and a hard cone (`_mjcf` takes no range and no softness).
      range=math.inf,
      softness=0.0,
      bulbradius=light.radius if kind != "directional" else 0.0,
      castshadow=light.shadow_enable,
    )
  return dataclasses.replace(lighting, lights=(), replace=False)


def classic_cone_inner(exponent: np.ndarray, cutoff_deg: np.ndarray) -> Column[np.float32]:
  """(nlight,) inner / outer cone angle fractions standing for classic (OpenGL) spots' `exponent`:
  the cone's smoothstep from the inner angle to `cutoff` falls to one half where cos^exponent
  does (theta_half = acos(0.5^(1 / exponent))), i.e. the inner angle is 2 theta_half - cutoff,
  within [0, cutoff]. Exponent 0 (a uniform cone) keeps a hard edge (fraction 1); MuJoCo's default
  10 at a 40 degree cutoff gives a pool fading from its centre (fraction 0.07)."""
  e = np.maximum(np.asarray(exponent, np.float64), 0.0)
  cutoff = np.radians(np.maximum(np.asarray(cutoff_deg, np.float64), 1.0e-6))
  half = np.where(e > 0.0, np.arccos(np.power(0.5, 1.0 / np.maximum(e, 1.0e-12))), np.pi / 2)
  return np.clip((2.0 * half - cutoff) / cutoff, 0.0, 1.0).astype(np.float32)


# The MJCF light type of each mapped light kind.
_MJ_LIGHT_TYPES: Final[dict[MjcfKind, mujoco.mjtLightType]] = {
  "directional": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  "point": mujoco.mjtLightType.mjLIGHT_POINT,
  "spot": mujoco.mjtLightType.mjLIGHT_SPOT,
}
# MJCF's default `cutoff` [deg] (unused by a point light).
_MJ_POINT_CUTOFF: Final = 45.0


def _unheld(light: Light) -> list[str]:
  """The fields of `light` set away from what an MJCF light holds (a light keeps them only in a
  UsdLux scene)."""
  checks = {
    "exposure": light.exposure != 0.0,
    "normalize": light.normalize,
    "color temperature": light.enable_color_temperature,
    "diffuse / specular scales": light.diffuse != 1.0 or light.specular != 1.0,
    "a range": light.ext.range is not None,
    "a falloff other than inverse square": light.ext.falloff != "inverse_square",
  }
  return [name for name, unheld in checks.items() if unheld]


def _mjcf(light: Light) -> tuple[MjcfKind, float]:
  """(MJCF light type, intensity [lux or cd]) of a light MJCF can hold.

  Raises:
    ValueError: for another light (see `add_lights`).
  """
  where = f"rw.integrations.mujoco.add_lights {light.path or light.kind.name.lower()!r}"
  unheld = _unheld(light)
  if unheld:
    raise ValueError(f"{where}: MJCF lights cannot hold {', '.join(unheld)}")
  if light.kind == LightKind.DISTANT:
    return "directional", light.intensity
  point = light.kind == LightKind.SPHERE and (light.treat_as_point or light.radius > 0.0)
  if not point or (light.shaping is not None and light.shaping.ies):
    raise ValueError(f"{where}: MJCF holds directional, point and spot lights without an IES profile")
  area = math.pi * light.radius * light.radius if light.radius > 0.0 else 1.0
  intensity = light.intensity * area  # candela (`Light.point`: radiance x projected area)
  if light.shaping is None or light.shaping.cone_angle >= MAX_ANGLE_DEG:
    return "point", intensity
  if light.shaping.cone_softness > 0.0:
    raise ValueError(f"{where}: an MJCF spot has no cone softness (MuJoCo's `softness` is its photometric inner cone)")
  return "spot", intensity
