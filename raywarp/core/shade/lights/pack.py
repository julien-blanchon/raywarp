"""Packed lights: the frame's light tables gathered into one `punctual.Light` each.

The light slots (`light.pos`, `light.strength`, ...) are separate tables of 1 or nworld rows,
shared or per world, aliased to the simulator's (MuJoCo's moving lights) or written by the user
(`Renderer.per_world`). Reading them field by field in the light loop cost one `worldid % rows`
and one scattered load per field and light at every shading point. One kernel per frame
(`LightPack.pack`, graph-capturable: no host transfer or allocation once sized) gathers them into
a (rows, nlight) `Light` table, rows = the most rows of any input, with what the shading derives
per light precomputed: the spot cone's outer cosine and smooth-start angle, the area form's
frame and surface radiance; and the same table's hot records (`punctual.LightHot`), which the
ranking of capped light loops reads.

The environment suns (`core.pipeline.gpu_scene.gpu_scene.EnvSuns`: a map's sun cut out as a shadowed
directional light) are packed after the tables' lights (columns nsource, nsource + 1, ...: no table
holds them) and follow their world's environment: direction rotated by the world's `env.yaw`,
irradiance times its `env.intensity`, off where the world shows another map (`env.id`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.tables import batched, batched1d, in_table
from raywarp.core.pipeline.platform.launches import CachedLaunch
from raywarp.core.shade.environment import envmap
from raywarp.core.shade.lights import area as area_lights
from raywarp.core.shade.lights import ies as ies_lib
from raywarp.core.shade.lights import punctual
from raywarp.core.shade.lights.ies import IesLights, ies_table
from raywarp.core.shade.lights.ies_table import IesTable
from raywarp.core.shade.lights.punctual import Light, LightHot
from raywarp.core.shade.materials.texture import orthonormal_basis

if TYPE_CHECKING:
  from raywarp._base.dims import Table
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import EnvSuns

# The shading kernels' arithmetic options (the packed values equal what they computed per point).
wp.set_module_options({"enable_backward": False, "default_grid_stride": False, "fast_math": True})

# Floor of a sun's peak irradiance its colour is divided by (a black sun: colour 0).
_MIN_PEAK: Final = 1.0e-30

# A (1 or nworld, nlight) light table of a frame (`Frame.light_*`).
type LightTable = (
  Table[int, int, int] | Table[bool, int, int] | Table[float, int, int] | Table[wp.vec3, int, int] | Table[wp.vec2, int, int]
)


@wp.func
def cone(
  # In:
  ltype: int,
  cutoff: float,
  inner: float,
) -> wp.vec3:
  """(cos outer, smooth-start angle [rad], 1 / (outer - start) [1/rad]) of a light's spot cone
  (`punctual.cone_factor`); (NO_CONE, 0, 1) without one.

  Spot lights have one; cosine lights (rects / disks in point form) when `cutoff` < 90 degrees.
  The smooth start is `inner` x the cutoff angle (UsdLux: 1 - softness).
  """
  coned = ltype == wp.static(punctual.LIGHT_SPOT) or (ltype == wp.static(punctual.LIGHT_COSINE) and cutoff < 90.0)
  if not coned:
    return wp.vec3(wp.static(punctual.NO_CONE), 0.0, 1.0)
  outer = cutoff * wp.static(wp.pi / 180.0)
  start = outer * wp.clamp(inner, 0.0, 1.0)
  return wp.vec3(wp.cos(outer), start, 1.0 / wp.max(outer - start, wp.static(punctual.CONE_MIN_WIDTH)))


@wp.func
def area_radiance(
  # In:
  shape: int,
  radius: float,
  half: wp.vec2,
  color: wp.vec3,
  intensity: float,
) -> wp.vec3:
  """Uniform surface radiance of an area light of point-light `intensity` [cd] (radius >= 1e-4)."""
  r = wp.max(radius, 1.0e-4)
  area = r * r * wp.pi
  if shape == wp.static(area_lights.SHAPE_RECT):
    area = 4.0 * half[0] * half[1]
  elif shape == wp.static(area_lights.SHAPE_TUBE):
    area = 4.0 * r * half[0]
  return color * (intensity / wp.max(area, 1.0e-12))


@wp.struct
class Suns:
  """The environment suns (`EnvSuns`) and the per-world environment fields they follow."""

  # (nsun,) the map each sun was cut out of, its direction in that map's frame, its colour
  # (irradiance / peak), peak irradiance and disc's angular radius [rad].
  env: wp.array[int]
  dir: wp.array[wp.vec3]
  color: wp.array[wp.vec3]
  peak: wp.array[float]
  radius: wp.array[float]
  env_id: wp.array[int]
  env_intensity: wp.array[float]
  env_yaw: wp.array[float]
  nenv: int
  # Whether the environment lights the scene (`Features.env_lit`): off, the suns are off too.
  lit: bool


@wp.func
def sun_light(
  # In:
  suns: Suns,
  row: int,
  k: int,
) -> Light:
  """Sun `k` in row `row`: a shadowed directional light on where the row's world shows its map,
  turned by its yaw, scaled by its intensity."""
  light = Light()
  light.type = wp.static(punctual.LIGHT_DIRECTIONAL)
  light.shape = wp.static(area_lights.SHAPE_NONE)
  light.cast_shadow = True
  light.active = suns.lit and in_table(batched1d(suns.env_id, row), suns.nenv) == suns.env[k]
  light.dir = -envmap.rotate_yaw(suns.dir[k], batched1d(suns.env_yaw, row))
  light.color = suns.color[k]
  light.intensity = suns.peak[k] * batched1d(suns.env_intensity, row)
  light.range = 1.0
  light.attenuation = wp.vec3(1.0, 0.0, 0.0)
  light.radius = suns.radius[k]
  light.cos_outer = wp.static(punctual.NO_CONE)
  light.cone_start = 0.0
  light.cone_scale = 1.0
  axis_x, axis_y = orthonormal_basis(light.dir)
  light.axis_x = axis_x
  light.axis_y = axis_y
  return light


@wp.kernel
def pack(
  # In:
  ltype: wp.array2d[int],
  cast_shadow: wp.array2d[bool],
  active: wp.array2d[bool],
  color: wp.array2d[wp.vec3],
  cutoff: wp.array2d[float],
  cone_inner: wp.array2d[float],
  intensity: wp.array2d[float],
  light_range: wp.array2d[float],
  light_attenuation: wp.array2d[wp.vec3],
  radius: wp.array2d[float],
  pos: wp.array2d[wp.vec3],
  direction: wp.array2d[wp.vec3],
  shape: wp.array2d[int],
  axis: wp.array2d[wp.vec3],
  size: wp.array2d[wp.vec2],
  suns: Suns,
  ies_profile: wp.array[int],
  ies: IesTable,
  # Out:
  out: wp.array2d[Light],
  hot: wp.array2d[LightHot],
):
  """Light li of row `row` (a world, or every world when no input is per world): a table's light,
  or past the tables' columns an environment sun. Area lights keep their shape under
  `lighting.area_lights="point"` too: no area code is compiled then, and the light-size widening
  of their point form reads it (`evaluators.source_widening`)."""
  row, li = wp.tid()
  nsource = ltype.shape[1]
  if li >= nsource:
    sun = sun_light(suns, row, li - nsource)
    out[row, li] = sun
    hot[row, li] = punctual.hot_of(sun)
    return
  light = Light()
  light.type = batched(ltype, row, li)
  light.shape = batched(shape, row, li)
  light.cast_shadow = batched(cast_shadow, row, li)
  light.active = batched(active, row, li)
  light.pos = batched(pos, row, li)
  light.dir = batched(direction, row, li)
  light.color = batched(color, row, li)
  power = batched(intensity, row, li)
  light.range = batched(light_range, row, li)
  light.attenuation = batched(light_attenuation, row, li)
  light.radius = batched(radius, row, li)
  c = cone(light.type, batched(cutoff, row, li), batched(cone_inner, row, li))
  light.cos_outer = c[0]
  light.cone_start = c[1]
  light.cone_scale = c[2]
  light.axis_x = batched(axis, row, li)
  light.axis_y = wp.cross(light.dir, light.axis_x)
  light.ies = ies_profile[li] + 1
  if light.ies != 0:
    # TILT=INCLUDE at the light's current orientation (moving lights; 1 without a table).
    nadir = wp.where(light.type == wp.static(punctual.LIGHT_TUBE), light.axis_x, light.dir)
    power = power * ies_lib.tilt(ies, light.ies, nadir)
  light.intensity = power
  light.half = batched(size, row, li)
  if light.shape != wp.static(area_lights.SHAPE_NONE):
    light.radiance = area_radiance(light.shape, light.radius, light.half, light.color, power)
  out[row, li] = light
  hot[row, li] = punctual.hot_of(light)


@wp.kernel
def derive_hot(
  # In:
  packed: wp.array2d[Light],
  # Out:
  hot: wp.array2d[LightHot],
):
  """The hot record of every packed light."""
  row, li = wp.tid()
  hot[row, li] = punctual.hot_of(packed[row, li])


@final
class LightPack:
  """A Renderer's packed light table (`pack` once per frame; see the module doc).

  Attributes:
    lights: (rows, max(nlight, 1)) the packed lights (`Lights.lights`).
    hot: their hot records (`Lights.hot`).
    ies: the IES profile table (`Lights.ies`).
    visible: (max(k, 1),) (light, LightShape) of the k lights camera rays see (`Lights.visible`).
    nvisible: k.
  """

  def __init__(
    self,
    nlight: int,
    suns: EnvSuns,
    *,
    ies: IesLights,
    visible: np.ndarray[[int, 2], np.dtype[np.int32]],
    device: wp.Device,
  ) -> None:
    """Sized for one row, on `device`; `nlight`: the lights shaded (the tables' then the suns,
    `GpuScene.nlight`); `suns`: the environment suns (`Environment.suns`); `ies`: the lights' IES
    profiles (`GpuScene.ies`); `visible`: the lights camera rays see (`GpuScene.light_visible`)."""
    self.nlight = nlight
    n = max(nlight, 1)
    self.has_suns = bool(len(suns))
    self._suns = _sun_arrays(suns, device)
    profile = np.full(n, -1, np.int32)
    profile[: len(ies.light_profile)] = ies.light_profile
    self.ies_profile = wp.array(profile, dtype=int, device=device)
    # The profile table (shared by every world; `Lights.ies`).
    self.ies = ies_table(ies, device)
    self.nvisible = len(visible)
    rows = np.asarray(visible, np.int32).reshape(-1, 2) if len(visible) else np.zeros((1, 2), np.int32)
    self.visible = wp.array(rows, dtype=wp.vec2i, device=device)
    self.lights: wp.array[Light, [int, int]] = wp.zeros((1, n), dtype=Light, device=device)
    self.hot: wp.array[LightHot, [int, int]] = wp.zeros((1, n), dtype=LightHot, device=device)
    self._launch = CachedLaunch()

  def rows(self, frame: Frame) -> int:
    """Rows of the packed table: the most rows of a light table (and of an environment field a sun
    follows)."""
    rows = max(t.shape[0] for t in _tables(frame))
    if self.has_suns:
      rows = max(rows, frame.env_id.shape[0], frame.env_intensity.shape[0], frame.env_yaw.shape[0])
    return rows

  def pack(self, frame: Frame, nenv: int, *, env_lit: bool) -> bool:
    """Packs `frame`'s lights (`nenv` environment maps; the suns on with `env_lit`, the
    environment lighting the scene); True when the table was reallocated."""
    rows = self.rows(frame)
    dev = self.lights.device
    grown = self.lights.shape[0] != rows
    if grown:
      self.lights = wp.zeros((rows, max(self.nlight, 1)), dtype=Light, device=dev)
      self.hot = wp.zeros((rows, max(self.nlight, 1)), dtype=LightHot, device=dev)
    if self.nlight == 0:
      return grown
    suns = self._suns
    suns.env_id, suns.env_intensity, suns.env_yaw, suns.nenv = frame.env_id, frame.env_intensity, frame.env_yaw, max(nenv, 1)
    suns.lit = env_lit
    self._launch.launch(
      pack,
      (rows, self.nlight),
      inputs=[*_tables(frame), suns, self.ies_profile, self.ies],
      outputs=[self.lights, self.hot],
      device=dev,
    )
    return grown


def _sun_arrays(suns: EnvSuns, device: wp.Device) -> Suns:
  """The `Suns` of `suns` on `device` (at least one row each: an empty array is no kernel
  argument; the per-world fields are the frame's, set at every pack)."""
  n = max(len(suns), 1)
  irradiance = np.zeros((n, 3), np.float32)
  irradiance[: len(suns)] = suns.irradiance
  peak = irradiance.max(axis=1)
  out = Suns()
  out.env = wp.array(np.r_[suns.env, np.full(n - len(suns), -1)].astype(np.int32), dtype=int, device=device)
  out.dir = wp.array(np.r_[suns.direction, np.zeros((n - len(suns), 3))].astype(np.float32), dtype=wp.vec3, device=device)
  out.color = wp.array(irradiance / np.maximum(peak, _MIN_PEAK)[:, None], dtype=wp.vec3, device=device)
  out.peak = wp.array(peak, dtype=float, device=device)
  out.radius = wp.array(np.r_[suns.angle, np.zeros(n - len(suns))].astype(np.float32), dtype=float, device=device)
  return out


def _tables(f: Frame) -> tuple[LightTable, ...]:
  """The frame's light tables in `pack`'s input order."""
  return (
    f.light_type,
    f.light_cast_shadow,
    f.light_active,
    f.light_color,
    f.light_cutoff,
    f.light_cone_inner,
    f.light_strength,
    f.light_range,
    f.light_attenuation,
    f.light_radius,
    f.light_pos,
    f.light_dir,
    f.light_shape,
    f.light_axis,
    f.light_size,
  )


def hot_records(packed: wp.array[Light, [int, int]]) -> wp.array[LightHot, [int, int]]:
  """The hot records of a packed light table built elsewhere (`derive_hot`; tests, bakes)."""
  hot = wp.empty(packed.shape, dtype=LightHot, device=packed.device)
  wp.launch(derive_hot, dim=packed.shape, inputs=[packed], outputs=[hot], device=packed.device)
  return hot


def sun_discs(suns: EnvSuns, nenv: int, device: wp.Device) -> tuple[wp.array[wp.vec4, [int]], wp.array[wp.vec3, [int]]]:
  """(`kernels.abi.structs.EnvironmentArrays.env_sun`, `env_sun_radiance`; drawn by
  `background.sun_disc`) of `nenv` maps (at least one row) with `suns`: per map its sun's direction
  (map frame) and angular radius (0: no sun), and the radiance of the uniform disc holding its
  irradiance (E / (2 pi (1 - cos r)) = E / (4 pi sin^2(r / 2))). The disc covers the region cut out
  of the map (`EnvSuns.extent`), not the light's smaller disc (`angle`): a smaller disc left a ring
  of the refill around it, darker than the map's glow there (the map's sun spans that region
  too)."""
  n = max(nenv, 1)
  disc = np.zeros((n, 4), np.float32)
  radiance = np.zeros((n, 3), np.float32)
  for k, e in enumerate(suns.env.tolist()):
    r = float(suns.extent[k])
    if r > 0.0:
      disc[e] = (*suns.direction[k], r)
      radiance[e] = suns.irradiance[k] / (4.0 * np.pi * np.sin(0.5 * r) ** 2)
  return wp.array(disc, dtype=wp.vec4, device=device), wp.array(radiance, dtype=wp.vec3, device=device)
