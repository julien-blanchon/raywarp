"""IES photometric profiles on the GPU: the profile table (`IesTable`, one copy shared by every
world and light using a profile) and the per-light lookup (`factor`).

The host grids (`scene/lighting/ies_profile.py`) are concatenated into one float array; each profile
has a header per instance (a grid with a light's UsdLux angle scale, its coarse profile and TILT
table: `IesProfile`). A light points at its instance with `Light.ies` (1 + its index; 0, a
zeroed `Light`: none); the light's intensity already holds the
profile's peak and its normalisation (`pipeline/upload/lights.py`), so `factor` returns the profile
relative to its peak, in [0, 1]. Each instance's profile is baked at upload (`ies_profile.sample`,
the angle scale applied) into an OCTA x OCTA octahedral map of the emission direction in the light
frame (nadir `Light.dir` at the map's centre, reference axis `Light.axis_x`; a tube's are swapped):
the lookup is three dot products, the octahedral fold and one bilinear read (no inverse
trigonometry, one code path for every photometric type).

Compiled in only with `Features.ies` (a scene with an IES light): without one, no kernel reads
the table or the light fields.
"""

from __future__ import annotations

import dataclasses
from typing import Final, final

import numpy as np
import warp as wp

from raywarp._base.warp_structs import warp_struct
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.shade.lights import punctual
from raywarp.core.shade.lights.ies_table import IesProfile, IesTable
from raywarp.core.shade.lights.punctual import Light
from raywarp.scene.lighting import ies_profile
from raywarp.scene.lighting.ies_profile import IesGrid

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

KIND_C: Final = ies_profile.KIND_C


# The octahedral map's texels per side (a profile instance: OCTA^2 floats, 64 KB): the bake
# resamples the photometric grid at the texel centres (~1.4 degrees apart at the nadir: a sharp
# cut-off is blurred over about one texel).
OCTA: Final = 128


@wp.func
def _octa_uv(
  # In:
  v: wp.vec3,
) -> wp.vec2:
  """The octahedral map coordinates in [0, 1]^2 of direction `v` (any length; +Z at the centre)."""
  p = wp.vec2(v[0], v[1]) / wp.max(wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2]), 1.0e-20)
  if v[2] < 0.0:
    p = wp.vec2(
      (1.0 - wp.abs(p[1])) * wp.where(p[0] >= 0.0, 1.0, -1.0),
      (1.0 - wp.abs(p[0])) * wp.where(p[1] >= 0.0, 1.0, -1.0),
    )
  return p * 0.5 + wp.vec2(0.5)


# Inlined: the table struct is not copied into a call per light.
@wp.func(inline=True)
def factor(
  # Model:
  table: IesTable,
  # In:
  light: Light,
  l: wp.vec3,
) -> float:
  """The profile of `light` toward the point it lights (`l`: the direction to the light),
  relative to its peak; 1 for a light without one."""
  if light.ies == 0:
    return 1.0
  tube = light.type == wp.static(punctual.LIGHT_TUBE)
  nadir = wp.where(tube, light.axis_x, light.dir)
  ref = wp.where(tube, light.dir, light.axis_x)
  d = -l
  uv = _octa_uv(wp.vec3(wp.dot(d, ref), wp.dot(d, wp.cross(ref, nadir)), wp.dot(d, nadir)))
  # One bilinear fetch between texel centres, clamped at the map's edges (its strip of the atlas).
  half = wp.static(0.5 / float(OCTA))
  x = wp.clamp(uv[0], half, 1.0 - half)
  y = (float(light.ies - 1) + wp.clamp(uv[1], half, 1.0 - half)) * table.inv_rows
  return wp.texture_sample(table.maps, wp.vec2(x, y), dtype=float)


def octa_directions(n: int) -> np.ndarray[[int, int], np.dtype[np.float64]]:
  """(n * n, 3) the unit directions at the texel centres of an n x n octahedral map (row-major:
  row = v, column = u; +Z at the centre), `_octa_uv`'s inverse."""
  c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
  px, py = np.meshgrid(c, c, indexing="xy")
  z = 1.0 - np.abs(px) - np.abs(py)
  lower = z < 0.0
  x = np.where(lower, (1.0 - np.abs(py)) * np.sign(px), px)
  y = np.where(lower, (1.0 - np.abs(px)) * np.sign(py), py)
  v = np.stack([x, y, z], axis=-1).reshape(-1, 3)
  return v / np.linalg.norm(v, axis=1, keepdims=True)


def octa_profile(g: IesGrid, angle_scale: float) -> np.ndarray[[int], np.dtype[np.float32]]:
  """(OCTA^2,) the octahedral map of grid `g` with `angle_scale`: `factor`'s frame (x along the
  reference axis, y, z = cos from the nadir) is `ies_profile.sample`'s light-local frame with its
  Z flipped (the nadir is local -Z)."""
  d = octa_directions(OCTA)
  local = np.stack([d[:, 0], d[:, 1], -d[:, 2]], axis=1)
  return ies_profile.sample(g, local, angle_scale).astype(np.float32)


@wp.func
def coarse(
  # Model:
  table: IesTable,
  # In:
  instance: int,
  cos_nadir: float,
) -> wp.vec2:
  """(bound, azimuthal mean) of profile instance `instance` (1 + index) at cos(theta) from the
  nadir, relative to the peak (`ies_profile.coarse`): the capped lights' ranking and remainder."""
  p = table.profiles[instance - 1]
  u = (wp.clamp(cos_nadir, -1.0, 1.0) + 1.0) * wp.static(0.5 * float(ies_profile.COARSE_KNOTS - 1))
  i = wp.min(int(u), wp.static(ies_profile.COARSE_KNOTS - 2))
  t = u - float(i)
  k = p.coarse + 2 * i
  return wp.vec2(
    wp.lerp(table.data[k], table.data[k + 2], t),
    wp.lerp(table.data[k + 1], table.data[k + 3], t),
  )


@wp.func
def tilt(
  # Model:
  table: IesTable,
  # In:
  instance: int,
  nadir: wp.vec3,
) -> float:
  """The TILT=INCLUDE factor of profile instance `instance` (1 + index; 0, or no table: 1) for a
  light whose nadir (type C) or photometric axis (A / B) is `nadir` (world, +Z up): the angle
  [deg] of the nadir from straight down, or of the axis from the horizon; linear between the
  table's angles, clamped at its ends. Evaluated per frame (`pack.pack`): moving lights."""
  if instance == 0:
    return 1.0
  p = table.profiles[instance - 1]
  if p.ntilt == 0:
    return 1.0
  down = wp.clamp(-nadir[2], -1.0, 1.0)
  angle = wp.degrees(wp.acos(down))
  if p.kind != wp.static(KIND_C):
    angle = wp.abs(wp.degrees(wp.asin(down)))
  if angle <= table.data[p.tilt]:
    return table.data[p.tilt + p.ntilt]
  for k in range(1, p.ntilt):
    a1 = table.data[p.tilt + k]
    if angle <= a1:
      a0 = table.data[p.tilt + k - 1]
      f0 = table.data[p.tilt + p.ntilt + k - 1]
      f1 = table.data[p.tilt + p.ntilt + k]
      return wp.lerp(f0, f1, (angle - a0) / wp.max(a1 - a0, 1.0e-6))
  return table.data[p.tilt + 2 * p.ntilt - 1]


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IesLights:
  """The IES profiles of an uploaded scene and which light uses which.

  Attributes:
    grids: the distinct profiles (a profile used by several lights is stored once).
    instances: (grid index, angle scale) of each profile instance (one per distinct pair).
    light_profile: (source lights,) int32 index into `instances` (-1: none; the environment suns,
      packed after them, have none).
  """

  grids: tuple[IesGrid, ...]
  instances: tuple[tuple[int, float], ...]
  light_profile: np.ndarray[[int], np.dtype[np.int32]]

  @property
  def any(self) -> bool:
    """Some light has a profile."""
    return bool(np.any(self.light_profile >= 0))

  @staticmethod
  def none(nlight: int) -> IesLights:
    """No light has a profile."""
    return IesLights(grids=(), instances=(), light_profile=np.full(nlight, -1, np.int32))


def ies_table(ies: IesLights, device: wp.Device) -> IesTable:
  """The device table of `ies`: per instance its octahedral map (`octa_profile`, one strip of the
  `maps` atlas), coarse profile and TILT table (one empty entry without any: arrays and the atlas
  are never empty)."""
  headers = np.zeros(max(len(ies.instances), 1), dtype=warp_struct(IesProfile).numpy_dtype())
  chunks: list[np.ndarray[[int], np.dtype[np.float32]]] = [np.zeros(1, np.float32)]
  maps = [np.ones((OCTA, OCTA), np.float32)] if not ies.instances else []
  offset = 1
  for i, (gi, scale) in enumerate(ies.instances):
    g = ies.grids[gi]
    maps.append(octa_profile(g, scale).reshape(OCTA, OCTA))
    rough = ies_profile.coarse(g, scale).ravel()
    tilt_at, ntilt = offset + rough.size, len(g.tilt_angles)
    tilt = np.concatenate([g.tilt_angles, g.tilt_factors]).astype(np.float32)
    headers[i] = (g.kind, offset, tilt_at, ntilt)
    chunks += [rough, tilt]
    offset = tilt_at + tilt.size
  table = IesTable()
  table.profiles = wp.array(headers, dtype=IesProfile, device=device)
  table.data = wp.array(np.concatenate(chunks), dtype=wp.float32, device=device)
  texels = np.ascontiguousarray(np.concatenate(maps))
  atlas = wp.Texture2D(texels, filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP, device=device)
  # Its content digest: the bakes that read the lights (GI, the light grid) key on it.
  table.maps = bake_cache.register_texture(atlas, bake_cache.texture_digest(texels, "ies-atlas", "linear", "clamp"))
  table.inv_rows = 1.0 / float(len(maps))
  return table
