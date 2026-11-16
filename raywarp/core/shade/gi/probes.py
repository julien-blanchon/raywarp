"""The GI probe grid as the shading reads it: the probes' maps and the irradiance lookup at a
surface point (device side of the GI feature; package doc: `shade/gi/__init__.py`).

Each probe stores, in octahedral maps (written by the blends as (row, texel) arrays, sampled
by the lookup from bilinear texture atlases with one-texel borders: `atlas.py`):
- `irradiance` (IRR_TEXELS² texels): rgb = cosine-weighted mean incoming radiance around the
  texel's direction (E / pi: the units of the split-sum diffuse ambient, so the diffuse term is
  `diffuse_color * E / pi`), a = the fraction of that cosine lobe that reaches the sky
  (environment / ambient fill; used to occlude the environment's specular term);
- `distance` (DIST_TEXELS² texels): mean and mean² of the hit distance around the texel
  (power-50 cosine lobe), for the Chebyshev visibility test of the lookup.

Lookup of the probes (`lookup`: the resample of the ambient cube; `irradiance_at`, the debug views):
trilinear weights over the 8 probes of the cell around the point offset by the normal / view
biases, times the wrap-shading backface weight and the Chebyshev visibility weight (RTXGI's
`DDGIGetVolumeIrradiance`; distance moments and irradiance bilinear: nearest distance texels made
the visibility weight a step function of the direction, sharp leak patches at coarse spacing).
The resample also drops the probes its static sight test found hidden from the texel and reports
the visible weight (the cube's validity). Camera, reflection and probe-ray hits read `irradiance`:
the lookup resampled after each update into the ambient-cube texture (`cube.py`): the Chebyshev
test alone passes probes behind room corners, the cube's static sight mask does not.

Maps are float16 (the blend runs in float32). Their rgb holds E / pi times `Volume.scale` (the
exposure at the volume's creation: display-relative values, white ~ 1): in physical units a
display-white irradiance is ~4e4 (EV100 15), at float16's limit (65504), so a lamp's bounce
would overflow to inf, and inf times the blend's zero hysteresis is NaN, which every later
bounce spreads to the whole volume and image. The blend also drops non-finite ray results and
clamps to the float16 range (`cube.MAP_MAX`), so the maps always stay finite.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.kernels.abi.tables import table_row
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.shade.gi import cube as cube_lib
from raywarp.core.shade.gi.atlas import Atlas, tile_uv
from raywarp.core.shade.materials.octahedral import oct_decode, oct_encode

wp.set_module_options({"enable_backward": False})

# Octahedral map resolutions (interior texels per side; RTXGI: 6 and 14).
IRR_TEXELS: Final = 6
DIST_TEXELS: Final = 12
# Exponent of the cosine lobe of the distance maps (RTXGI's probeDistanceExponent).
DISTANCE_EXPONENT: Final = 50.0
# Ray-result distance of a miss.
MISS_DISTANCE: Final = 1.0e27
# Lookup weights: backface wrap floor, Chebyshev floor, and the RTXGI weight crush below which
# tiny weights are cubed (less light leaking through thin occluders).
_WRAP_FLOOR: Final = 0.2
_CHEBYSHEV_FLOOR: Final = 0.05
_CRUSH: Final = 0.2
# Floor of a corner's trilinear factor (a probe at the far corner still counts a little).
_TRILINEAR_FLOOR: Final = 0.001


@wp.struct
class RelightField:
  """Per-world tint of a shared volume's irradiance (continuous domain randomisation:
  `relight.Relight`): a coarse grid of cells over the probe box, each world's RGB ratio of its
  probe-ray radiance to the shared bake's in each cell, per axis direction (an ambient cube: the
  cosine lobes about +x, -x, +y, -y, +z, -z, blended by the squared normal like `cube.ambient`).

  Attributes:
    texture: the ratios minus 1 (float16 rgb; 0 filters to exactly 0: an untinted world reads
      exactly 1), one slab of (cz, cy, cx) cells per (world, face) (slab world x 6 + face), tiled
      like the ambient cube (`cube.layout`), filtered trilinearly.
    rows: worlds of the texture (0: no tint: a shared or grouped volume).
    origin: corner of cell (0, 0, 0) [m].
    inv_cell: inverse cell size per axis [1/m].
    counts: cells per axis.
    tiles: slabs per texture depth row.
  """

  texture: wp.Texture3D
  rows: int
  origin: wp.vec3
  inv_cell: wp.vec3
  counts: wp.vec3i
  tiles: int


def no_relight(device: wp.Device) -> RelightField:
  """The field of a volume without per-world tint."""
  field = RelightField()
  texels = wp.zeros((1, 1, 1, 4), dtype=wp.float16, device=device)
  field.texture = bake_cache.register_texture(cube_lib.texture3d(texels, linear=True), "gi-relight:none")
  field.rows = 0
  field.origin = wp.vec3(0.0)
  field.inv_cell = wp.vec3(1.0)
  field.counts = wp.vec3i(1, 1, 1)
  field.tiles = 1
  return field


@wp.func
def relight_texel(
  # Model:
  field: RelightField,
  # In:
  slab: int,
) -> wp.vec2i:
  """(y, z) texel offsets of slab `slab` of the tint texture."""
  return wp.vec2i((slab // field.tiles) * field.counts[1], (slab % field.tiles) * field.counts[2])


@wp.func(inline=True)
def _relit_face(
  # Model:
  field: RelightField,
  # In:
  q: wp.vec3,
  slab: int,
) -> wp.vec3:
  """Trilinear ratio - 1 of slab `slab` at cell coordinates q (`cube.fetch`'s snapping: the
  filter weights do not depend on where the slab lies)."""
  at = relight_texel(field, slab)
  y = float(at[0]) + cube_lib.snap(wp.clamp(q[1], 0.5, float(field.counts[1]) - 0.5))
  z = float(at[1]) + cube_lib.snap(wp.clamp(q[2], 0.5, float(field.counts[2]) - 0.5))
  v = wp.texture_sample(field.texture, wp.vec3(cube_lib.snap(q[0]), y, z), dtype=wp.vec4)
  return wp.vec3(v[0], v[1], v[2])


@wp.struct
class Volume:
  """The probe grid and its maps (kernel input of the shading; a field of `Environment`).

  Attributes:
    origin: position of probe (0, 0, 0) before relocation.
    spacing: probe spacing per axis.
    counts: probes per axis.
    normal_bias: lookup offset along the surface normal [m].
    max_distance: distances are clamped to this in the distance maps [m].
    scale: map units per radiance unit of the irradiance maps' rgb (keeps them in float16 range).
    nprobe: probes of the grid (one group's rows of `irradiance`).
    world_group: (1 or nworld,) the irradiance group of each world (row `world % len`): group g's
      maps are rows [g nprobe, (g + 1) nprobe) of `irradiance` (`groups.py`).
    irradiance: (ngroup nprobe, IRR_TEXELS²) rgb: E / pi times `scale`, a: sky visibility.
    distance: (nprobe, DIST_TEXELS²) mean and mean² of the hit distance in units of max_distance
      (float16: squares in metres overflow past 256 m; geometry only: shared by the groups).
    offset: (nprobe,) relocation offset of each probe.
    active: (nprobe,) 1 for probes used by the lookup.
    irradiance_atlas, distance_atlas: the maps as bilinear texture atlases (`atlas.Atlas`: what
      the lookup samples, unnormalised texel coordinates); irradiance_tiles / distance_tiles:
      tiles per atlas row.
    cube: the groups' irradiance resampled into ambient cubes (`cube.Cube`: what camera and
      reflection hits read).
    relight: the per-world tint of a relit volume (`RelightField`; `rows` 0: none).
  """

  origin: wp.vec3
  spacing: wp.vec3
  counts: wp.vec3i
  normal_bias: float
  max_distance: float
  scale: float
  nprobe: int
  world_group: wp.array1d[wp.int32]
  irradiance: wp.array2d[wp.vec4h]
  distance: wp.array2d[wp.vec2h]
  offset: wp.array1d[wp.vec3]
  active: wp.array1d[wp.int32]
  irradiance_atlas: wp.Texture2D
  distance_atlas: wp.Texture2D
  irradiance_tiles: int
  distance_tiles: int
  cube: cube_lib.Cube
  relight: RelightField


def attach(vol: Volume, irradiance: Atlas, distance: Atlas) -> None:
  """Points `vol`'s atlas fields at these atlases."""
  vol.irradiance_atlas, vol.irradiance_tiles = irradiance.texture, irradiance.tiles
  vol.distance_atlas, vol.distance_tiles = distance.texture, distance.tiles


def empty_volume(device: wp.Device) -> Volume:
  """The placeholder of renderers without GI on `device` (never read: the lookup is compiled
  out)."""
  vol = Volume()
  vol.origin = wp.vec3(0.0)
  vol.spacing = wp.vec3(1.0)
  vol.counts = wp.vec3i(1, 1, 1)
  vol.normal_bias = 0.0
  vol.max_distance = 1.0
  vol.scale = 1.0
  vol.nprobe = 1
  vol.world_group = wp.zeros(1, dtype=wp.int32, device=device)
  vol.irradiance = wp.zeros((1, 1), dtype=wp.vec4h, device=device)
  vol.distance = wp.zeros((1, 1), dtype=wp.vec2h, device=device)
  vol.offset = wp.zeros(1, dtype=wp.vec3, device=device)
  vol.active = wp.zeros(1, dtype=wp.int32, device=device)
  attach(vol, Atlas(1, 1, 4, device), Atlas(1, 1, 2, device))
  vol.cube = cube_lib.CubeTexture((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1), 1, device).struct
  vol.relight = no_relight(device)
  return vol


@wp.func
def texel_direction(
  # In:
  texel: int,
  n: int,
) -> wp.vec3:
  """Direction of the centre of texel `texel` (row-major) of an n x n octahedral map."""
  tx = texel % n
  ty = texel // n
  inv = 2.0 / float(n)
  return oct_decode(wp.vec2((float(tx) + 0.5) * inv - 1.0, (float(ty) + 0.5) * inv - 1.0))


@wp.func(inline=True)
def _sample_irradiance(
  # Model:
  vol: Volume,
  # In:
  row: int,
  uv: wp.vec2,
) -> wp.vec4:
  """Bilinear irradiance of map row `row` at octahedral coordinates `uv` (map units; one atlas
  fetch)."""
  at = tile_uv(uv, row, wp.static(IRR_TEXELS), vol.irradiance_tiles)
  return wp.texture_sample(vol.irradiance_atlas, at, dtype=wp.vec4)


@wp.func(inline=True)
def _sample_distance(
  # Model:
  vol: Volume,
  # In:
  probe: int,
  d: wp.vec3,
) -> wp.vec2:
  """Bilinear distance moments [m, m^2] of `probe` towards d (one atlas fetch; stored in units of
  the max distance)."""
  at = tile_uv(oct_encode(d), probe, wp.static(DIST_TEXELS), vol.distance_tiles)
  m = wp.texture_sample(vol.distance_atlas, at, dtype=wp.vec2)
  return wp.vec2(m[0] * vol.max_distance, m[1] * vol.max_distance * vol.max_distance)


@wp.func
def _unscale(
  # Model:
  vol: Volume,
  # In:
  g: wp.vec4,
) -> wp.vec4:
  """Map value -> (E / pi, sky visibility)."""
  inv = 1.0 / vol.scale
  return wp.vec4(g[0] * inv, g[1] * inv, g[2] * inv, g[3])


# Grid ---------------------------------------------------------------------------------------------


@wp.func
def probe_index(
  # Model:
  vol: Volume,
  # In:
  c: wp.vec3i,
) -> int:
  """Index of the probe at grid coordinates c."""
  return (c[2] * vol.counts[1] + c[1]) * vol.counts[0] + c[0]


@wp.func
def probe_coords(
  # Model:
  vol: Volume,
  # In:
  probe: int,
) -> wp.vec3i:
  """Grid coordinates of a probe."""
  x = probe % vol.counts[0]
  y = (probe // vol.counts[0]) % vol.counts[1]
  z = probe // (vol.counts[0] * vol.counts[1])
  return wp.vec3i(x, y, z)


@wp.func(inline=True)
def grid_position(
  # Model:
  vol: Volume,
  # In:
  probe: int,
) -> wp.vec3:
  """World position of a probe's grid point (before relocation)."""
  c = probe_coords(vol, probe)
  return vol.origin + wp.cw_mul(vol.spacing, wp.vec3(float(c[0]), float(c[1]), float(c[2])))


@wp.func
def probe_position(
  # Model:
  vol: Volume,
  # In:
  probe: int,
) -> wp.vec3:
  """World position of a probe (grid position + relocation offset)."""
  return grid_position(vol, probe) + vol.offset[probe]


@wp.func
def probe_cell(
  # Model:
  vol: Volume,
  # In:
  biased: wp.vec3,
) -> tuple[wp.vec3i, wp.vec3]:
  """(base probe coordinates, trilinear fractions) of the lookup at the biased point."""
  q = wp.cw_div(biased - vol.origin, vol.spacing)
  base = wp.vec3i(int(wp.floor(q[0])), int(wp.floor(q[1])), int(wp.floor(q[2])))
  alpha = wp.vec3(
    wp.clamp(q[0] - float(base[0]), 0.0, 1.0),
    wp.clamp(q[1] - float(base[1]), 0.0, 1.0),
    wp.clamp(q[2] - float(base[2]), 0.0, 1.0),
  )
  return base, alpha


@wp.func
def probe_corner(
  # Model:
  vol: Volume,
  # In:
  base: wp.vec3i,
  alpha: wp.vec3,
  k: int,
) -> tuple[int, wp.vec3]:
  """(probe, per-axis trilinear factors) of corner k of the lookup cell (clamped to the grid)."""
  off = wp.vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1)
  c = wp.vec3i(
    wp.clamp(base[0] + off[0], 0, vol.counts[0] - 1),
    wp.clamp(base[1] + off[1], 0, vol.counts[1] - 1),
    wp.clamp(base[2] + off[2], 0, vol.counts[2] - 1),
  )
  tx = wp.where(off[0] == 1, alpha[0], 1.0 - alpha[0])
  ty = wp.where(off[1] == 1, alpha[1], 1.0 - alpha[1])
  tz = wp.where(off[2] == 1, alpha[2], 1.0 - alpha[2])
  floor = wp.static(_TRILINEAR_FLOOR)
  return probe_index(vol, c), wp.vec3(wp.max(tx, floor), wp.max(ty, floor), wp.max(tz, floor))


@wp.func
def _weight(
  # Model:
  vol: Volume,
  # In:
  point: wp.vec3,
  normal: wp.vec3,
  biased: wp.vec3,
  base: wp.vec3i,
  alpha: wp.vec3,
  k: int,
  sight: int,
) -> tuple[int, float, float]:
  """(probe, weight, visible weight) of corner k of the lookup cell: the weight is 0 for an
  inactive probe and for a corner whose bit k of `sight` is clear (a probe known to be hidden
  from the point); the visible weight is its trilinear factor times its unfloored Chebyshev
  visibility (how much of the lookup the probes that see the point carry)."""
  probe, t = probe_corner(vol, base, alpha, k)
  if vol.active[probe] == 0 or (sight >> k) & 1 == 0:
    return probe, 0.0, 0.0
  ppos = probe_position(vol, probe)
  to_probe = wp.normalize(ppos - point)
  biased_to_probe = ppos - biased
  dist = wp.length(biased_to_probe)
  # Smooth backface: probes behind the surface count less (but never zero, against holes).
  wrap = (wp.dot(to_probe, normal) + 1.0) * 0.5
  w = wrap * wrap + wp.static(_WRAP_FLOOR)
  # Chebyshev visibility from the probe's distance moments towards the point.
  moments = _sample_distance(vol, probe, -biased_to_probe / wp.max(dist, 1.0e-6))
  chebyshev = float(1.0)
  if dist > moments[0]:
    variance = wp.abs(moments[0] * moments[0] - moments[1])
    v = dist - moments[0]
    chebyshev = variance / (variance + v * v)
    chebyshev = chebyshev * chebyshev * chebyshev
  w *= wp.max(wp.static(_CHEBYSHEV_FLOOR), chebyshev)
  w = wp.max(w, 1.0e-6)
  # Crush tiny weights (RTXGI): reduces light leaking through thin occluders.
  crush = wp.static(_CRUSH)
  if w < crush:
    w *= w * w / (crush * crush)
  trilinear = t[0] * t[1] * t[2]
  return probe, w * trilinear, chebyshev * trilinear


@wp.func
def group_row(
  # Model:
  vol: Volume,
  # In:
  worldid: int,
) -> int:
  """First irradiance map row of world `worldid`'s group."""
  return vol.world_group[table_row(worldid, vol.world_group.shape[0])] * vol.nprobe


@wp.func
def irradiance(
  # Model:
  vol: Volume,
  # In:
  worldid: int,
  point: wp.vec3,
  normal: wp.vec3,
  outward: wp.vec3,
) -> wp.vec4:
  """(E / pi, sky visibility) at `point` of world `worldid` for the hemisphere around unit
  `normal` (the shading normal; `outward`: the geometric normal the lookup is offset along), from
  the ambient-cube texture (camera, reflection and probe-ray hits: `cube.ambient`), times the
  world's tint (`relit`).

  E / pi multiplies a Lambertian albedo directly (the split-sum diffuse ambient's units).
  """
  group = vol.world_group[table_row(worldid, vol.world_group.shape[0])]
  e = _unscale(vol, cube_lib.ambient(vol.cube, group, point, normal, outward))
  k = relit(vol.relight, worldid, point, normal)
  return wp.vec4(e[0] * k[0], e[1] * k[1], e[2] * k[2], e[3])


@wp.func
def beyond(
  # Model:
  vol: Volume,
  # In:
  point: wp.vec3,
) -> float:
  """How far `point` lies outside the probe box, in probe spacings along the farthest axis (0
  inside). Every bounded instance lies inside the box (planes and unbounded scenery do not
  bound it): a point beyond it sees the environment past them, where the box's boundary texels
  (what the lookup clamps to) hold the light next to them."""
  cube = vol.cube
  q = wp.cw_mul(point - cube.origin, cube.inv_cell)
  out = float(0.0)
  for a in range(3):
    over = wp.max(wp.max(-q[a], q[a] - float(cube.counts[a])), 0.0)
    out = wp.max(out, over / (cube.inv_cell[a] * vol.spacing[a]))
  return out


@wp.func
def relit(
  # Model:
  field: RelightField,
  # In:
  worldid: int,
  point: wp.vec3,
  normal: wp.vec3,
) -> wp.vec3:
  """World `worldid`'s tint of the shared irradiance at `point` around unit `normal` (the three
  faces the normal faces, trilinear, blended by the squared normal; three texture fetches);
  exactly 1 without per-world tint or where the world's ratios are 1."""
  if field.rows == 0:
    return wp.vec3(1.0)
  first = (worldid % field.rows) * 6
  q = wp.cw_mul(point - field.origin, field.inv_cell)
  dx = _relit_face(field, q, first + wp.where(normal[0] >= 0.0, 0, 1))
  dy = _relit_face(field, q, first + wp.where(normal[1] >= 0.0, 2, 3))
  dz = _relit_face(field, q, first + wp.where(normal[2] >= 0.0, 4, 5))
  return wp.vec3(1.0) + dx * (normal[0] * normal[0]) + dy * (normal[1] * normal[1]) + dz * (normal[2] * normal[2])


@wp.func
def lookup(
  # Model:
  vol: Volume,
  # In:
  first_row: int,
  point: wp.vec3,
  biased: wp.vec3,
  cell_point: wp.vec3,
  normal: wp.vec3,
  sight: int,
) -> tuple[wp.vec4, float]:
  """(probe lookup in map units (rgb times `Volume.scale`), visible weight) of the group whose
  maps start at row `first_row`, looked up at an explicitly offset point `biased` (no bias
  added), over the probe cell around `cell_point` (`biased` itself, but the texel centre for the
  ambient cube's resample, whose `sight` is of that cell).

  `sight`: bit k clear for a cell corner hidden from the point (all set: unknown; the ambient
  cube's resample knows them: `cube.Cube.walls`). The visible weight (0-1) is the share of the
  trilinear weight carried by probes that see the point (`_weight`); with no active, visible probe
  the value is 0.
  """
  total = wp.vec4(0.0)
  wsum = float(0.0)
  visible = float(0.0)
  base, alpha = probe_cell(vol, cell_point)
  # Every probe's irradiance is read towards the normal: one octahedral encoding for the 8 probes.
  uv = oct_encode(normal)
  for k in range(8):
    probe, w, seen = _weight(vol, point, normal, biased, base, alpha, k, sight)
    if w > 0.0:
      total += _sample_irradiance(vol, first_row + probe, uv) * w
      wsum += w
      visible += seen
  if wsum <= 0.0:
    return wp.vec4(0.0), 0.0
  return total / wsum, visible


@wp.func
def maps_at(
  # Model:
  vol: Volume,
  # In:
  first_row: int,
  point: wp.vec3,
  biased: wp.vec3,
  normal: wp.vec3,
) -> wp.vec4:
  """The value of `lookup` with every probe of the cell possibly visible."""
  value, _visible = lookup(vol, first_row, point, biased, biased, normal, 255)
  return value


@wp.func
def irradiance_at(
  # Model:
  vol: Volume,
  # In:
  first_row: int,
  point: wp.vec3,
  biased: wp.vec3,
  normal: wp.vec3,
) -> wp.vec4:
  """`maps_at` as (E / pi, sky visibility)."""
  return _unscale(vol, maps_at(vol, first_row, point, biased, normal))


@wp.func
def fibonacci_direction(
  # In:
  i: int,
  n: int,
) -> wp.vec3:
  """Direction i of n spherical Fibonacci points (uniform on the sphere)."""
  golden = 0.6180339887498949
  phi = 2.0 * wp.pi * (float(i) * golden - wp.floor(float(i) * golden))
  cos_theta = 1.0 - (2.0 * float(i) + 1.0) / float(n)
  sin_theta = wp.sqrt(wp.max(0.0, 1.0 - cos_theta * cos_theta))
  return wp.vec3(wp.cos(phi) * sin_theta, wp.sin(phi) * sin_theta, cos_theta)
