"""Probe lookups resampled into ambient-cube 3D textures: what camera and reflection hits read
(a few hardware-filtered fetches instead of the 8-probe lookup).

After every update the probe lookup (`probes.lookup`: 8 probes, Chebyshev visibility) is
evaluated at the centre of every texel of a regular grid over the probe box (`subdiv` texels per
probe spacing and axis) for the six axis normals, each looked up from a point biased along its
normal by the volume's normal bias. Shading blends the three axis values facing its normal by the
squared normal components (Valve's ambient cube), sampled at the point pushed one texel along the
normal. One slab of (nz, ny, nx) texels per (group, axis direction), rgb = E / pi times
`Volume.scale`, a = sky visibility; three fetches (`ambient`).

Leaks (bright corners and edges) are kept out by three parts:
1. Wall-aware filtering (`wall_aware`): a static texture of each texel's free distance along the
   six axes (`walls`, traced once at placement: `kernels.walls_kernel`). Per axis, the lookup
   point moves onto a texel centre when a wall lies between it and the other texel of its
   trilinear footprint (or between it and the nearest texel), so hardware filtering never mixes
   in a texel on the other side of a wall, whatever the wall's thickness; a point on a wall's plane
   (a crease) reads the texel on its open side.
2. Validity: a texel face none of whose probes sees it (visible weight below `VISIBLE_MIN`: inside
   geometry, behind furniture, a closet without probes) is invalid instead of a renormalised
   average of occluded probes; invalid faces take the mean of their valid neighbours that no wall
   separates from them (`dilate`: DILATE_PASSES passes, more for a probe volume's cube), texels
   inside solids that of their outside neighbours (a probe volume's cube), then any texel the mean
   of its valid neighbours across walls (one pass); the rest keep their own value (deep inside
   geometry).
3. Placement: the default probe box spans all static geometry and one spacing more (`baked`).

Layout: slab s of (nz, ny, nx) texels, tiled along the texture's depth then height (`layout`); x
spans the slab, so the hardware clamps it, y and z are clamped in the lookup to the slab's texel
centres (no filtering across slabs). The texel budget lowers the subdivision for many slabs
(continuous randomisation, `subdivision`); a cube that cannot fit raises.

Map units, float16. Invalid texel faces are flagged in the staging texels only, by a negative
flag channel (`-1 - value`) during the resample and dilation.
"""

from __future__ import annotations

import enum
import math
import warnings
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import host_copy, launch_bands
from raywarp.core.shade.gi.groups import group_used

if TYPE_CHECKING:
  from collections.abc import Mapping

  from raywarp._base.dims import AnyHostArray

wp.set_module_options({"enable_backward": False})

# (gate, rep, nworld): which groups ran this update (`groups.group_used`).
type GroupGate = tuple[wp.array1d[int], wp.array1d[wp.int32], int]

# Texels per probe spacing and axis: at least SUBDIV, more for texels of at most the scene's
# texel size (`baked._probe_grid`: its derived spacing / SUBDIV, so a coarser gi.spacing or a grid
# coarsened by gi.max_probes keeps it), at most MAX_SUBDIV (it drops until the cube fits its
# budget).
SUBDIV: Final = 4
MAX_SUBDIV: Final = 8
# Texel budget of the slabs of a cube (8 B per texel; the staging texels hold as many: a 128 MB
# texture).
MAX_TEXELS: Final = 1 << 24
# Largest 3D texture side (CUDA).
MAX_SIDE: Final = 16384
FACES: Final = 6
# Largest map value written: the probe maps' and the cube's (float16's largest finite value is
# 65504).
MAP_MAX: Final = 6.0e4
# A texel face is valid when the probes that see it carry at least this share of the trilinear
# weight: the lookup's Chebyshev floor (`probes._CHEBYSHEV_FLOOR`), the weight a fully occluded
# probe keeps; below it the value is the renormalised mix of probes that do not see the point.
VISIBLE_MIN: Final = 0.05
# Dilation passes: an invalid face up to this many texels from a valid one that no wall separates
# from it takes its neighbours' mean (probe-less pockets up to about this wide); then one pass
# across walls fills the texels inside geometry next to valid ones (a lookup at a convex corner
# can reach one through the diagonal of its footprint).
DILATE_PASSES: Final = 3
# Fewest texel rows per CPU band of the dilation passes (`CubeTexture.finish`).
ROW_BAND: Final = 8
# Free distances are stored in 1/255 texel steps (8 bits), two per 16-bit channel; 255: at least
# one texel free.
WALL_STEPS: Final = 255
# The texture unit's linear filter weights have 8 fractional bits: lookups are snapped to that
# grid (`fetch`).
FILTER_STEPS: Final = 256
# Walls channel 3: bits 0-7 the probe sight mask (`kernels.walls_kernel`); bit INSIDE_BIT the texel
# centre lies inside a closed solid; bit COMPACT_BIT it lies inside a compact solid (an object in a
# room: `kernels.compact_solids`); bit SOLID_SHIFT + d (d = 2 a + (0: +a, 1: -a)) the neighbour
# along direction d is inside a compact solid. A compact solid's texels hold the space around it
# (`dilate`): a footprint may reach them without a leak.
INSIDE_BIT: Final = 8
SIGHT_MASK: Final = (1 << INSIDE_BIT) - 1
SOLID_SHIFT: Final = 9
COMPACT_BIT: Final = 15


class Stage(enum.IntEnum):
  """The texels a dilation pass fills (`dilate`): outside solids, from the neighbours no wall
  separates from them; inside solids, from any neighbour; any texel still invalid, walls ignored
  (the last pass)."""

  OUTSIDE = 0
  INSIDE = 1
  ACROSS = 2


@wp.struct
class Cube:
  """The ambient-cube textures of a probe volume (a field of `probes.Volume`).

  Attributes:
    texture: float16 rgba, trilinear, clamped, unnormalised texel coordinates: the slabs.
    walls: uint16 rgba, nearest, clamped, unnormalised: per texel of a slab, the free distance
      [texels, 0-1] along +a / -a in the high / low byte of channel a; channel 3 the probe sight
      mask and the solid bits (`SIGHT_MASK`, `INSIDE_BIT`, `COMPACT_BIT`, `SOLID_SHIFT`).
    origin: lower corner of the texel box (the probe box).
    inv_cell: texels per metre per axis.
    counts: (nx, ny, nz) texels of a slab.
    tiles: slabs per row of the texture's depth.
    bias: lookup offset along the normal (one texel) [m].
  """

  texture: wp.Texture3D
  walls: wp.Texture3D
  origin: wp.vec3
  inv_cell: wp.vec3
  counts: wp.vec3i
  tiles: int
  bias: float


@wp.func(inline=True)
def slab_origin(
  # Model:
  cube: Cube,
  # In:
  slab: int,
) -> wp.vec2i:
  """(y, z) texel offsets of slab `slab` in the texture."""
  return wp.vec2i((slab // cube.tiles) * cube.counts[1], (slab % cube.tiles) * cube.counts[2])


@wp.func(inline=True)
def fetch(
  # Model:
  cube: Cube,
  # In:
  q: wp.vec3,
  slab: int,
) -> wp.vec4:
  """Trilinear value of slab `slab` at texel coordinates q (centres at +0.5, y / z clamped to the
  slab's texel centres). y and z are snapped to 1/FILTER_STEPS texel before the slab's offset is
  added: the hardware filter weights (FILTER_STEPS steps) then do not depend on where the slab
  lies in the texture (a group's lookups equal those of the same world rendered alone)."""
  at = slab_origin(cube, slab)
  y = float(at[0]) + snap(wp.clamp(q[1], 0.5, float(cube.counts[1]) - 0.5))
  z = float(at[1]) + snap(wp.clamp(q[2], 0.5, float(cube.counts[2]) - 0.5))
  return wp.texture_sample(cube.texture, wp.vec3(snap(q[0]), y, z), dtype=wp.vec4)


@wp.func(inline=True)
def snap(
  # In:
  x: float,
) -> float:
  """`x` rounded down to a multiple of 1 / FILTER_STEPS (exact in float32 below 2^15 texels)."""
  return wp.floor(x * wp.static(float(FILTER_STEPS))) * wp.static(1.0 / float(FILTER_STEPS))


@wp.func
def free_distance(
  # In:
  raw: int,
  positive: bool,
) -> float:
  """Free distance [texels, 0-1] towards +axis (`positive`) or -axis of a 16-bit walls channel."""
  d = wp.where(positive, raw >> 8, raw & 255)
  return float(d) / float(wp.static(WALL_STEPS))


@wp.func
def wall_channel(
  # Model:
  cube: Cube,
  # In:
  i: int,
  j: int,
  k: int,
  c: int,
) -> int:
  """Raw 16-bit channel `c` of texel (i, j, k)'s walls (the walls texture, nearest)."""
  v = wp.texture_sample(cube.walls, wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5), dtype=wp.vec4)
  return int(wp.round(v[c] * 65535.0))


@wp.func
def sighted(
  # Model:
  cube: Cube,
  # In:
  cell: wp.vec3,
  a: int,
  at: float,
) -> bool:
  """Whether a probe is in sight of the centre of the texel `cell` moved to `at` along axis `a`."""
  other = wp.vec3(cell[0], cell[1], cell[2])
  other[a] = wp.clamp(at, 0.5, float(cube.counts[a]) - 0.5)
  walls = wp.texture_sample(cube.walls, other, dtype=wp.vec4)
  return (int(wp.round(walls[3] * 65535.0)) & wp.static(SIGHT_MASK)) != 0


@wp.func
def wall_aware(
  # Model:
  cube: Cube,
  # In:
  q: wp.vec3,
) -> wp.vec3:
  """Texel coordinates `q` clamped to the box, moved per axis onto a texel centre where a wall
  separates them from a texel of their trilinear footprint (one walls fetch, two on a crease).

  Along axis a the footprint holds the texel c containing q and its neighbour on q's side. With
  the free distance d from c's centre towards q (texels): d >= 1, nothing in between: q stays;
  |q - c| <= d, the wall lies beyond q: q moves onto c's centre; else the wall lies between c and
  q (c is behind it): q moves onto the neighbour's centre. A q on the wall itself (within the
  stored distances' rounding: a receiver at the edge of a perpendicular wall, a crease) stays on
  c's side, unless no probe is in sight of c's centre and one is of the neighbour's: c then lies
  inside the wall and holds a dilated value (possibly the other side's light), the neighbour the
  open side's.
  """
  out = wp.vec3(0.0)
  cell = wp.vec3(0.0)
  for a in range(3):
    n = float(cube.counts[a])
    out[a] = wp.clamp(q[a], 0.5, n - 0.5)
    cell[a] = wp.min(wp.floor(out[a]), n - 1.0) + 0.5
  walls = wp.texture_sample(cube.walls, cell, dtype=wp.vec4)
  bits = int(wp.round(walls[3] * 65535.0))
  # c inside a compact solid holds the nearest values of the space around it (`dilate`): nothing
  # to keep apart.
  inside = (bits >> wp.static(COMPACT_BIT)) & 1 != 0
  for a in range(3):
    s = out[a] - cell[a]
    d = free_distance(int(wp.round(walls[a] * 65535.0)), s >= 0.0)
    # The texel beyond the wall on q's side holds this space's values when it lies inside a compact
    # solid (`SOLID_SHIFT`): no snap (smooth filtering around objects, at curved and slanted
    # surfaces, and where the offset point entered one).
    ahead = (bits >> (wp.static(SOLID_SHIFT) + 2 * a + wp.where(s >= 0.0, 0, 1))) & 1 != 0
    if d < 1.0 and not inside and not ahead:
      # The stored distances are rounded down by at most one step: within two of q, q is on the
      # wall.
      beyond = wp.abs(s) - d
      across = cell[a] + wp.where(s >= 0.0, 1.0, -1.0)
      move = beyond >= wp.static(2.0 / float(WALL_STEPS))
      if not move and beyond > -wp.static(2.0 / float(WALL_STEPS)) and (bits & wp.static(SIGHT_MASK)) == 0:
        move = sighted(cube, cell, a, across)
      out[a] = wp.where(move, across, cell[a])
  return out


@wp.func
def lookup_point(
  # Model:
  cube: Cube,
  # In:
  point: wp.vec3,
  outward: wp.vec3,
) -> wp.vec3:
  """Wall-aware texel coordinates of a surface point with unit geometric normal `outward` (pushed
  one texel along it: a mapped or bent shading normal would push it through the surface)."""
  return wall_aware(cube, wp.cw_mul(point + outward * cube.bias - cube.origin, cube.inv_cell))


@wp.func
def ambient(
  # Model:
  cube: Cube,
  # In:
  group: int,
  point: wp.vec3,
  normal: wp.vec3,
  outward: wp.vec3,
) -> wp.vec4:
  """Map value (E / pi times the volume's scale, sky visibility) of group `group` at `point`
  around unit `normal` (the shading normal; `outward`: the geometric normal the lookup is offset
  along) (a walls fetch and three trilinear fetches)."""
  q = lookup_point(cube, point, outward)
  first = group * wp.static(FACES)
  vx = fetch(cube, q, first + wp.where(normal[0] >= 0.0, 0, 1))
  vy = fetch(cube, q, first + wp.where(normal[1] >= 0.0, 2, 3))
  vz = fetch(cube, q, first + wp.where(normal[2] >= 0.0, 4, 5))
  return vx * (normal[0] * normal[0]) + vy * (normal[1] * normal[1]) + vz * (normal[2] * normal[2])


@wp.func
def store_texel(
  # In:
  value: wp.vec4,
  valid: bool,
  flag: int,
) -> wp.vec4:
  """A resampled texel value with channel `flag` flagged (`-1 - value`) when invalid; channels
  clamped to the float16 range."""
  out = value
  for c in range(4):
    out[c] = wp.clamp(out[c], 0.0, wp.static(MAP_MAX))
  if not valid:
    out[flag] = -1.0 - out[flag]
  return out


@wp.func
def write_texel(
  # Model:
  cube: Cube,
  # In:
  slab: int,
  k: int,
  j: int,
  i: int,
  value: wp.vec4,
  # Out:
  texels: wp.array4d[wp.float16],
):
  """Writes the staging texel (k, j, i) of slab `slab`."""
  at = slab_origin(cube, slab)
  for c in range(4):
    texels[at[1] + k, at[0] + j, i, c] = wp.float16(value[c])


@wp.kernel
def dilate(
  # Model:
  cube: Cube,
  # Data in:
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  # In:
  nworld: int,
  stage: int,
  row0: int,
  # Out:
  texels: wp.array4d[wp.float16],
):
  """One dilation pass: each invalid texel (k, j, i) of slab s (row row0 + r =
  (s * nz + k) * ny + j; row0: the first row of a CPU band) takes the mean of the six
  neighbours' valid values: `Stage.OUTSIDE`, of the texels outside solids, from the neighbours no
  wall separates from it; `Stage.INSIDE`, of the texels inside a solid (`INSIDE_BIT`), from the
  outside neighbours across its surface and the inside ones no surface separates from it (filled
  from outside inwards, after the outside: each takes its nearest outside values, never another
  solid's); `Stage.ACROSS`, of any texel, walls ignored. Filled values are flagged +2: a pass only
  reads values valid before it, so the result does not depend on the thread order (nor on the
  bands). Only the slabs of the groups that ran (`groups.group_used`)."""
  r, i = wp.tid()
  row = row0 + r
  j = row % cube.counts[1]
  k = (row // cube.counts[1]) % cube.counts[2]
  s = row // (cube.counts[1] * cube.counts[2])
  if not group_used(gate, rep, nworld, s // wp.static(FACES)):
    return
  at = slab_origin(cube, s)
  z, y = at[1] + k, at[0] + j
  idx = wp.vec3i(i, j, k)
  solid = (wall_channel(cube, i, j, k, 3) >> wp.static(INSIDE_BIT)) & 1 != 0
  if (stage == wp.static(int(Stage.OUTSIDE)) and solid) or (stage == wp.static(int(Stage.INSIDE)) and not solid):
    return
  across = stage != wp.static(int(Stage.OUTSIDE))
  if float(texels[z, y, i, 3]) < 0.0:
    total = wp.vec4(0.0)
    count = float(0.0)
    for nb in range(6):
      a = nb // 2
      step = wp.where(nb % 2 == 0, 1, -1)
      nidx = wp.vec3i(idx[0], idx[1], idx[2])
      nidx[a] = idx[a] + step
      if nidx[a] < 0 or nidx[a] >= cube.counts[a]:
        continue
      walled = free_distance(wall_channel(cube, i, j, k, a), step > 0) < 1.0
      if walled and not across:
        continue
      if walled and stage == wp.static(int(Stage.INSIDE)):
        # Inside a solid: an outside neighbour across its surface, or an inside one of the same
        # solid (no surface between); never another solid's interior.
        if (wall_channel(cube, nidx[0], nidx[1], nidx[2], 3) >> wp.static(INSIDE_BIT)) & 1 != 0:
          continue
      nz, ny = at[1] + nidx[2], at[0] + nidx[1]
      v = float(texels[nz, ny, nidx[0], 3])
      if v < 0.0 or v >= 2.0:
        continue
      for e in range(4):
        total[e] += float(texels[nz, ny, nidx[0], e])
      count += 1.0
    if count > 0.0:
      mean = total / count
      mean[3] += 2.0
      for e in range(4):
        texels[z, y, i, e] = wp.float16(mean[e])


@wp.kernel
def settle(
  # Model:
  cube: Cube,
  # Data in:
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  # In:
  nworld: int,
  final: int,
  row0: int,
  # Out:
  texels: wp.array4d[wp.float16],
):
  """After a dilation pass: filled values (flag >= 2) become valid; with `final`, the faces still
  invalid keep their own value (`-1 - flag` undone). Only the slabs of the groups that ran. Texel
  row row0 + r = z * Y + y of the staging array (row0: the first row of a CPU band)."""
  r, i = wp.tid()
  z, y = (row0 + r) // texels.shape[1], (row0 + r) % texels.shape[1]
  slab = (y // cube.counts[1]) * cube.tiles + z // cube.counts[2]
  # The last depth row of slabs can be partly padding (`layout`): no group, nothing to settle.
  if slab // wp.static(FACES) >= rep.shape[0]:
    return
  if not group_used(gate, rep, nworld, slab // wp.static(FACES)):
    return
  v = float(texels[z, y, i, 3])
  if v >= 2.0:
    texels[z, y, i, 3] = wp.float16(v - 2.0)
  elif final != 0 and v < 0.0:
    texels[z, y, i, 3] = wp.float16(-1.0 - v)


def subdivision(
  probes: tuple[int, int, int], spacing: tuple[float, float, float], nslab: int, budget: int, cell: float | None = None
) -> int:
  """Texels per probe spacing (texels of at most `cell` [m], SUBDIV to MAX_SUBDIV; None: SUBDIV),
  lowered until the cube's `nslab` slabs fit `budget` texels and MAX_SIDE (warned below
  SUBDIV).

  Raises:
    ValueError: when even one texel per spacing does not fit.
  """
  cell = max(spacing) / SUBDIV if cell is None else cell
  sub = min(MAX_SUBDIV, max(SUBDIV, math.ceil(max(spacing) / cell - 1.0e-6)))
  while sub > 1 and not _fits(probes, nslab, sub, budget):
    sub -= 1
  if not _fits(probes, nslab, sub, budget):
    n = math.prod(probes) * nslab
    raise ValueError(
      f"ambient cube (GI / sky visibility): {math.prod(probes)} probes x {nslab} slabs ({n} texels at one "
      f"texel per probe spacing) exceed its budget of {budget} texels (texture side <= {MAX_SIDE}): "
      "lower gi.max_probes or the number of distinct per-world GI inputs (irradiance groups)"
    )
  if sub < SUBDIV:
    warnings.warn(
      f"ambient cube (GI / sky visibility): {sub} texels per probe spacing (of {SUBDIV}) to fit "
      f"{math.prod(probes)} probes x {nslab} slabs in {budget} texels: coarser indirect light",
      RuntimeWarning,
      stacklevel=3,
    )
  return sub


def _fits(probes: tuple[int, int, int], nslab: int, sub: int, budget: int) -> bool:
  """Whether `nslab` slabs of `sub` texels per spacing fit `budget` and the texture limits."""
  nx, ny, nz = (p * sub for p in probes)
  if nx * ny * nz * nslab > budget:
    return False
  _tiles, height, depth = layout((nx, ny, nz), nslab)
  return max(nx, height, depth) <= MAX_SIDE


def layout(counts: tuple[int, int, int], nslab: int) -> tuple[int, int, int]:
  """(slabs per depth row, texture height, texture depth) [texels] of `nslab` slabs of `counts`
  (nx, ny, nz) texels."""
  _nx, ny, nz = counts
  tiles = max(1, min(nslab, MAX_SIDE // nz))
  return tiles, math.ceil(nslab / tiles) * ny, tiles * nz


def texture3d(
  texels: wp.array[wp.float16, [int, int, int, 4]] | wp.array[wp.uint16, [int, int, int, 4]], *, linear: bool
) -> wp.Texture3D:
  """A clamped 3D texture holding `texels` (unnormalised coordinates)."""
  return wp.Texture3D(
    texels,
    filter_mode=wp.TextureFilterMode.LINEAR if linear else wp.TextureFilterMode.CLOSEST,
    address_mode=wp.TextureAddressMode.CLAMP,
    normalized_coords=False,
    device=texels.device,
  )


@final
class CubeTexture:
  """The cube textures of a volume and their staging texels (`kernels.resample_cube` writes the
  staging texels, `finish` dilates and uploads them).

  Args:
    lo: lower corner of the probe box [m]; spacing: probe spacing per axis [m]; probes: probes
    per axis; nslab: slabs (6 per group); budget: texels of all slabs at most; cell: the largest
    texel [m] (`subdivision`); passes: the dilation passes of `finish` outside solids and inside
    them (`Stage`).
  """

  def __init__(
    self,
    lo: tuple[float, float, float],
    spacing: tuple[float, float, float],
    probes: tuple[int, int, int],
    nslab: int,
    device: wp.Device,
    *,
    budget: int = MAX_TEXELS,
    cell: float | None = None,
    passes: tuple[int, int] = (DILATE_PASSES, 0),
  ) -> None:
    """Allocates the staging texels, the walls (no wall anywhere) and the textures; `struct` is
    the lookup's `Cube`. The staging arrays are what the bake writes; a frozen volume releases
    them (`release`) and keeps the textures only."""
    self.device = device
    self.nslab = nslab
    self.subdiv = subdivision(probes, spacing, nslab, budget, cell)
    nx, ny, nz = (p * self.subdiv for p in probes)
    tiles, height, depth = layout((nx, ny, nz), nslab)
    self._texel_shape = (depth, height, nx, 4)
    self._walls_shape = (nz, ny, nx, 4)
    self.texels: wp.array[wp.float16, [int, int, int, 4]] | None = wp.zeros(self._texel_shape, dtype=wp.float16, device=device)
    self.walls: wp.array[wp.uint16, [int, int, int, 4]] | None = wp.full(
      self._walls_shape, 0xFFFF, dtype=wp.uint16, device=device
    )
    # Contact occlusion per texel face (`kernels.ao_kernel`, open until the placement writes it;
    # six faces, padded to 8 bytes): what every resample reads, so a release keeps it.
    self.ao: wp.array[wp.uint8, [int, int, int, 8]] = wp.full((*self._walls_shape[:3], 8), 255, dtype=wp.uint8, device=device)
    self.passes = passes
    cube = Cube()
    # Derived from the maps and the placement (hashed as arrays by the bake keys): named by
    # their layout only.
    cube.texture = bake_cache.register_texture(texture3d(self.staging(), linear=True), f"gi-cube:{self._texel_shape}")
    cube.walls = bake_cache.register_texture(texture3d(self.wall_array(), linear=False), f"gi-walls:{self._walls_shape}")
    texel = [s / self.subdiv for s in spacing]
    cube.origin = wp.vec3(*lo)
    cube.inv_cell = wp.vec3(*(1.0 / t for t in texel))
    cube.counts = wp.vec3i(nx, ny, nz)
    cube.tiles = tiles
    cube.bias = float(max(texel))
    self.struct = cube

  @property
  def nbytes(self) -> int:
    """Device memory of the textures and of the staging arrays still held."""
    textures = 8 * (math.prod(self._texel_shape[:3]) + math.prod(self._walls_shape[:3]))
    return textures + sum(int(a.capacity) for a in (self.texels, self.walls, self.ao) if a is not None)

  def staging(self) -> wp.array[wp.float16, [int, int, int, 4]]:
    """The staging texels (allocated again after `release`: a resample writes every texel)."""
    if self.texels is None:
      self.texels = wp.zeros(self._texel_shape, dtype=wp.float16, device=self.device)
    return self.texels

  def wall_array(self) -> wp.array[wp.uint16, [int, int, int, 4]]:
    """The walls array the placement writes and a restore fills (allocated again after
    `release`; the kernels read the walls texture)."""
    if self.walls is None:
      self.walls = wp.full(self._walls_shape, 0xFFFF, dtype=wp.uint16, device=self.device)
    return self.walls

  def release(self) -> None:
    """Frees the staging texels and walls (a frozen bake: the textures and the contact
    occlusion hold everything the lookups and a later resample read); `staging` /
    `wall_array` allocate them again."""
    self.texels = None
    self.walls = None

  def set_walls(self) -> None:
    """Uploads the walls (`walls`, written by `kernels.walls_kernel` or a restore) into their
    texture.

    Raises:
      RuntimeError: the walls array was released.
    """
    if self.walls is None:
      raise RuntimeError("the cube's walls array was released (a frozen bake): nothing to upload")
    self.struct.walls.copy_from(self.walls)

  def finish(self, runs: GroupGate) -> None:
    """Dilates the invalid faces of the resampled staging texels of the groups that ran (`runs`:
    gate, rep, nworld of `groups.group_used`): the wall-aware passes outside solids and inside
    them (`passes`), one across walls, then the rest keep their own value; uploads
    them into the texture."""
    c = self.struct.counts
    dev = self.device
    gate, rep, nworld = runs
    texels = self.staging()
    module = wp.get_module(dilate.__module__)
    outside, inside = self.passes
    stages = [Stage.OUTSIDE] * outside + [Stage.INSIDE] * inside + [Stage.ACROSS]
    for stage in stages:
      common = [self.struct, gate, rep, nworld]

      def run_dilate(a: int, b: int, inputs: tuple[object, ...] = (*common, int(stage))) -> None:
        """Dilates texel rows [a, b) of every slab (row (s * nz + k) * ny + j)."""
        wp.launch(dilate, dim=(b - a, c[0]), inputs=[*inputs, a], outputs=[texels], device=dev)

      def run_settle(a: int, b: int, inputs: tuple[object, ...] = (*common, int(stage == Stage.ACROSS))) -> None:
        """Settles staging rows [a, b) (row z * Y + y)."""
        wp.launch(settle, dim=(b - a, texels.shape[2]), inputs=[*inputs, a], outputs=[texels], device=dev)

      launch_bands(module, device=dev, n=self.nslab * c[2] * c[1], launch=run_dilate, min_band=ROW_BAND)
      launch_bands(module, device=dev, n=texels.shape[0] * texels.shape[1], launch=run_settle, min_band=ROW_BAND)
    self.struct.texture.copy_from(texels)

  def to_arrays(self) -> dict[str, AnyHostArray]:
    """The cube as host arrays (a bake cache entry: `from_arrays`).

    Raises:
      RuntimeError: the staging arrays were released.
    """
    if self.texels is None or self.walls is None:
      raise RuntimeError("the cube's staging arrays were released (a frozen bake): its state is in the bake cache")
    s = self.struct
    return {
      "texels": host_copy(self.texels),
      "walls": host_copy(self.walls),
      "origin": np.asarray(s.origin, np.float32),
      "inv_cell": np.asarray(s.inv_cell, np.float32),
      "tiles": np.asarray(s.tiles, np.int32),
      "bias": np.asarray(s.bias, np.float32),
    }


def from_arrays(a: Mapping[str, AnyHostArray], device: wp.Device, digest: str) -> Cube:
  """The `Cube` of the arrays of `CubeTexture.to_arrays` on `device` (bit-identical; its
  textures registered under `digest`: an input of other bakes)."""
  texels = wp.array(a["texels"], dtype=wp.float16, device=device)
  walls = wp.array(a["walls"], dtype=wp.uint16, device=device)
  cube = Cube()
  cube.texture = bake_cache.register_texture(texture3d(texels, linear=True), f"{digest}:texels")
  cube.walls = bake_cache.register_texture(texture3d(walls, linear=False), f"{digest}:walls")
  cube.origin = wp.vec3(*a["origin"])
  cube.inv_cell = wp.vec3(*a["inv_cell"])
  nz, ny, nx = walls.shape[:3]
  cube.counts = wp.vec3i(nx, ny, nz)
  cube.tiles = int(a["tiles"])
  cube.bias = float(a["bias"])
  return cube
