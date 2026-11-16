"""Static light visibility grid (light culling).

The scene's bounding box is divided into cubic cells; a light is listed in a cell when it is in
range of the cell and visible (an unoccluded segment) from at least one of 27 points of the cell (a
3 x 3 x 3 lattice including its corners); a cell that sees no light (e.g. inside a solid object
whose surface crosses it) lists the lights its 26 neighbours see, none when they see none either (a
closed room without lights, above a ceiling: listing every light there made each shading point rank
every light and leak the unranked ones' light, unshadowed, through the walls). Directional lights
are listed in every cell: a sun reaches surfaces through openings narrower than the lattice (an
alpha cut-out louvre, a slit, a gap under a roof) whose lattice points see none of it, its penumbra
(a sun disc) reaches cells whose points see no centre ray, and window glass blocks the test's
any-hit rays but not the shading; its shadow rays decide. A light that emits nothing at the build
(zero intensity or color: lamps authored off) is listed in no cell and costs the bake nothing; lit
later, it invalidates its row like a moved light (`check_reach`), unless the lights' emission is
declared per world or written by a simulator (the dark lights are then flooded too). Shading then
loops over the lights of the point's cell only: lights behind walls neither cost a loop iteration
nor leak light through the walls. Lights have no range cutoff (the inverse square law at any
distance, `pipeline.upload.lights`): a cell moves its weakest distant positional lights, together at
most FAR_FRACTION of its estimated illuminance, into its far field (their unshadowed irradiance at
the cell's centre on the six axis normals, an ambient cube, shaded through the lobes' albedo like
the capped loops' remainder, `sources.far_field`), so a long light list costs no loop
iterations and no light is dropped by a distance cutoff. Built once from world 0 (lights
and the scene static geometry at that time), on the first render. Every frame checks packed-light
reach on the device: changed rows evaluate all lights until they return to their baked configuration
or the user requests `Renderer.rebuild_light_grid()`. Declared movable-light IDs are excluded from
the static grid and evaluated once per shading point; their changes do not invalidate rows. Points
outside the grid also evaluate all lights. Moving shared walls still require an explicit rebuild.
Instances that move or change per world (robots, per-world poses / sizes / meshes / visibility) do
not occlude the grid's rays, software or OptiX (`baked._see_through_dynamic`: world 0's occlusion
would cull lights elsewhere); they cast shadows through the shadow rays.

The 27 points of a cell are the half-cell lattice over the grid, shared with its neighbours; the
test floods each light's lattice outwards from the light (`light_flood`) instead of testing every
(point, light) pair.
"""

from __future__ import annotations

import bisect
import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.records import record_field
from raywarp.core.kernels.abi.rays import (
  UNBOUNDED_SHADOW_DIST,
  GeometryFields,
  SoftwareGeometry,
  Trees,
  software_geometry_type,
)
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.shade.lights import area as area_lights
from raywarp.core.shade.lights import flood as light_flood
from raywarp.core.shade.lights import punctual as lights
from raywarp.core.shade.lights.flood import MARKED, Corner, Pass, lattice_ijk
from raywarp.core.shade.lights.punctual import Light, Lights
from raywarp.core.trace.accel import query
from raywarp.scene.images.colorspace import REC709_Y

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyHostArray
  from raywarp.core.trace.backend import GridTracer


wp.set_module_options({"enable_backward": False})

# Host point sets of the grid's bounds: (n, 3) positions, (n,) values.
type Points[N: IntVar] = np.ndarray[[N, 3], np.dtype[np.float32]]
type Values[N: IntVar] = np.ndarray[[N], np.dtype[np.float32]]

# Cells per axis are capped: the cell size grows to fit large scenes.
MAX_CELLS_PER_AXIS: Final = 512
MAX_CELLS: Final = 1 << 22
# Lattice point x flooded light tests a build is capped at: the cell grows (`_cell_size`), so the
# bake's cost stays bounded whatever the scene's extent and light count. Measured: interiors of
# 1-16 lights need <= 0.3 M tests at 1 m cells (5 USD interiors); a 640 x 600 m town over 228
# lit lights 6.5 G at 1.25 m cells, and 14.3 G tests over 496 lights took 260-324 s of first
# frame on an RTX PRO 6000; a 16-core CPU tests ~15 M / s (128 M in 9.9 s). 2^29: seconds on
# RTX, ~36 s on the CPU; a tiny requested cell (0.05 m over 256 lights took minutes) is grown
# the same way.
MAX_TESTS: Final = 1 << 29
# Margin around the positional lights' box (the grid covers no more than that), in cells of the
# requested size (scale-free; 2 m with the default 1 m cell, as the former fixed margin).
LIGHT_MARGIN_CELLS: Final = 2.0
# Far field: a positional light at least FAR_CELLS cells from a cell's centre (its direction and
# distance vary little over the cell) may leave the cell's list, the weakest first, while those
# that left hold at most FAR_FRACTION of the cell's estimated illuminance (every light it sees,
# directional ones included). 2^-6: light-list length against image difference measured on two
# many-light interiors and a many-light street (2^-2 and 2^-6 compared).
FAR_CELLS: Final = 2.0
# ... and its emitter (half extents + radius) at most this fraction of its distance (a large panel
# close to the cell is not a point).
FAR_SIZE: Final = 0.25
FAR_FRACTION: Final = 2.0**-6
# `Grid.dynamic_mask` values: a movable light (no reach check), a light of some cell's far field,
# a light dark at the bake and not flooded (listed nowhere: emitting invalidates its row).
MOVABLE: Final = 1
FAR_MEMBER: Final = 2
UNFLOODED: Final = 3
# Rec. 709 luminance weights (`post.grading.color.luminance`).
_LUMA: Final = np.asarray(REC709_Y, np.float64)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Grid:
  """A built light grid (`assemble`), as the `Lights.grid_*` fields hold it (`wire`).

  `origin` / `cell` / `dims` of the ncell cells; cell c lists `lights[start[c]:start[c + 1]]`;
  `start[ncell]:start[ncell + 1]` of `lights` are the movable light IDs (excluded from the static
  slices, evaluated at every point; `dynamic_mask`: MOVABLE for them (no reach check), FAR_MEMBER
  for the lights in some cell's far field (an emission check), UNFLOODED for the lights dark at
  the bake and listed nowhere (emitting invalidates the row), 0 for the others). `snapshot`
  stores the build-time light row (every light on). `valid` (2 x light rows): per live light row,
  its reach (rows 0..n-1: 0 when a static light moved, the row then evaluates every light) and its
  far fields (rows n..2n-1: 0 when a far-field light's color or intensity differs from the build's
  or it is off, the far fields then read as zero: at most FAR_FRACTION of a cell's light);
  refreshed before shading (`checked`; transient, not persisted with the cached grid). `far` (3 x
  ncell, or empty when no cell has a far field): per cell its far-field color (the lights' summed
  illuminance at its centre) and its luminance on the +x, +y, +z and on the -x, -y, -z normals (an
  ambient cube).

  One array per role, merged: the OptiX launch parameters hold the `Lights` struct, under a byte
  limit (`trace.optix.device.params.PARAMS_LIMIT`).
  """

  origin: wp.vec3
  cell: float
  dims: wp.vec3i
  start: wp.array[int, [int]]
  lights: wp.array[int, [int]]
  snapshot: wp.array[Light, [int, int]]
  valid: wp.array[wp.int32, [int]]
  dynamic_mask: wp.array[wp.int32, [int]]
  far: wp.array[wp.vec3, [int]]


def assemble(
  *,
  origin: wp.vec3,
  cell: float,
  dims: wp.vec3i,
  start: np.ndarray[[int], np.dtype[np.int32]],
  lights: np.ndarray[[int], np.dtype[np.int32]],
  dynamic: np.ndarray[[int], np.dtype[np.int32]],
  snapshot: wp.array[Light, [int, int]],
  far: np.ndarray[[int, 3], np.dtype[np.float32]],
  far_members: np.ndarray[[int], np.dtype[np.int32]],
  nlight: int,
  unflooded: np.ndarray[[int], np.dtype[np.int32]] | None = None,
) -> Grid:
  """The `Grid` of the cells' lists `start` (ncell + 1) / `lights`, the movable IDs `dynamic`, the
  far fields `far` (3 ncell x 3, or empty) of the lights `far_members`, the lights `unflooded`
  (dark at the bake, listed nowhere; None: none) and the build-time row `snapshot` (nlight
  lights), on `snapshot`'s device."""
  dev = snapshot.device
  static = np.asarray(start, np.int32)
  merged = np.concatenate([static, [static[-1] + len(dynamic)]]).astype(np.int32)
  ids = np.concatenate([np.asarray(lights, np.int32)[: static[-1]], np.asarray(dynamic, np.int32)])
  mask = np.zeros(nlight, np.int32)
  mask[np.asarray(far_members, np.intp)] = FAR_MEMBER
  if unflooded is not None:
    mask[np.asarray(unflooded, np.intp)] = UNFLOODED
  mask[np.asarray(dynamic, np.intp)] = MOVABLE
  return Grid(
    origin=origin,
    cell=cell,
    dims=dims,
    start=wp.array(merged, dtype=int, device=dev),
    lights=wp.array(ids if ids.size else np.zeros(1, np.int32), dtype=int, device=dev),
    snapshot=snapshot,
    valid=wp.ones(2, dtype=wp.int32, device=dev),
    dynamic_mask=wp.array(mask, dtype=wp.int32, device=dev),
    far=wp.array(np.asarray(far, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev),
  )


def wire(lit: Lights, grid: Grid) -> None:
  """Points the `Lights` grid fields at `grid`."""
  lit.grid_origin, lit.grid_inv_cell, lit.grid_dims = grid.origin, 1.0 / grid.cell, grid.dims
  lit.grid_start, lit.grid_lights = grid.start, grid.lights
  lit.grid_valid, lit.grid_far = grid.valid, grid.far


def make_visit(trace_any: Callable, area: bool):
  """The visibility test of pair i of a `Pass` with ray query `trace_any`: the pair becomes
  -1 - pair where its light reaches its lattice point (`light_flood.spread_kernel` reads it).
  Shared by the software kernel (`visibility_kernel`) and the OptiX program
  (`trace.optix.device.programs`).
  """
  point_visible = make_point_visible(trace_any, area)

  @wp.func
  @device_generic
  def visit[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    g: Pass,
    i: int,
  ):
    """Traces pair i of `g`."""
    pair = g.pairs[i]
    # Cells marked already (`light_flood.settle_kernel`): the centre ray only (it decides the flood)
    samples = wp.where(pair >= wp.static(MARKED), 1, 5)
    pair = pair & wp.static(MARKED - 1)
    q = pair // g.nchunk
    li = g.light0 + pair - q * g.nchunk
    if not lights.fetch(lit, 0, li).active:
      return
    ijk = lattice_ijk(g.dims, q)
    p = g.origin + wp.vec3(float(ijk[0]), float(ijk[1]), float(ijk[2])) * (0.5 * g.cell)
    if point_visible(geo, lit, li, p, g.cell, samples):
      g.pairs[i] = -1 - pair

  return visit


def make_point_visible(trace_any: Callable, area: bool):
  """point_visible(geo, lit, li, p, cell, samples) -> whether light li of world 0 reaches point p.

  Punctual lights: an unoccluded segment (directional: towards the light), in range with a cell
  of slack; area lights (`area`): a segment to the centre or one of four extreme points, in front
  of a rect / disk.
  """

  @wp.func
  @device_generic
  def area_visible[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    li: int,
    p: wp.vec3,
    cell: float,
    samples: int,
  ) -> bool:
    """Whether area light `li` reaches point `p`.

    An unoccluded segment to its centre or (`samples` 5) one of four extreme points, in front of
    a rect / disk, in range with a cell of slack (range 0: unbounded).
    """
    a = area_lights.area_of(lights.fetch(lit, 0, li))
    one_sided = a.shape == wp.static(area_lights.SHAPE_RECT) or a.shape == wp.static(area_lights.SHAPE_DISK)
    # Range 0: unbounded reach (no window), as for punctual lights.
    if a.range > 0.0 and wp.length(a.center - p) > a.range + cell:
      return False
    if one_sided and wp.dot(p - a.center, a.normal) <= 0.0:
      return False
    for s in range(samples):
      to_light = area_lights.shape_points(a, s) - p
      dist = wp.length(to_light)
      if dist <= 0.0 or not trace_any(geo, 0, p, to_light / dist, dist * wp.static(area_lights.SHADOW_END)):
        return True
    return False

  @wp.func
  @device_generic
  def point_visible[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    li: int,
    p: wp.vec3,
    cell: float,
    samples: int,
  ) -> bool:
    """Whether light `li` (any form) reaches point `p` (area lights: by their first `samples`)."""
    light = lights.fetch(lit, 0, li)
    if wp.static(area):
      if light.shape != wp.static(area_lights.SHAPE_NONE):
        return area_visible(geo, lit, li, p, cell, samples)
    if light.type == wp.static(lights.LIGHT_DIRECTIONAL):
      return not trace_any(geo, 0, p, -light.dir, wp.static(UNBOUNDED_SHADOW_DIST))
    to_light = light.pos - p
    dist = wp.length(to_light)
    # Out of range from this sample point, with a cell of slack (range 0: no falloff, any distance).
    if light.range > 0.0 and dist > light.range + cell:
      return False
    return dist <= 0.0 or not trace_any(geo, 0, p, to_light / dist, dist * wp.static(area_lights.SHADOW_END))

  return point_visible


@specialize
def visibility_kernel(
  instance_ray_types: tuple[int, ...],
  *,
  area: bool,
  trees: Trees,
  alpha_cutout: bool,
  live_uv: bool = False,
  world_uv: bool = False,
):
  """The grid's software visibility kernel (`make_visit`), one thread per pair of a `Pass`.

  Cached: `warmup` compiles the same one ahead. `trees`: whether the per-world and the
  shared-instance BVHs hold primitives; `alpha_cutout`: the rays pass cut-out texels.
  """
  _, trace_any = query.unculled_queries(
    instance_ray_types, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv
  )
  visit = make_visit(trace_any, area)

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees))
  @concrete(geo=software_geometry_type(live_uv=live_uv))
  def visibility(
    # Model:
    geo: SoftwareGeometry,
    lit: Lights,
    # In:
    g: Pass,
  ):
    """Visibility of pair i of `g` (`make_visit`)."""
    visit(geo, lit, g, wp.tid())

  return visibility


def _bounds[NInstance: IntVar](
  pos: Points[NInstance], radius: Values[NInstance], boxes: light_flood.Boxes, cell: float
) -> tuple[Corner, Corner]:
  """Grid box: world 0's instances (centres +- bounding radii), clipped to the lights' box.

  The light box gets LIGHT_MARGIN_CELLS cells of `cell`: far scenery such as a backdrop would
  otherwise stretch the cells.
  """
  finite = np.isfinite(radius) & (radius < 1.0e5)  # planes have an infinite bounding radius
  if np.any(finite):
    lo, hi = (pos[finite] - radius[finite, None]).min(0), (pos[finite] + radius[finite, None]).max(0)
  else:  # No bounded static instances: use the lights' box, else one cell either side of the origin
    lo, hi = np.full(3, -np.inf), np.full(3, np.inf)
  positional = ~boxes.directional
  if np.any(positional):
    lp = boxes.centre[positional].astype(np.float32)  # the packed float32 positions
    lo = np.maximum(lo, lp.min(0) - LIGHT_MARGIN_CELLS * cell)
    hi = np.minimum(hi, lp.max(0) + LIGHT_MARGIN_CELLS * cell)
  lo, hi = np.where(np.isfinite(lo), lo, -cell), np.where(np.isfinite(hi), hi, cell)
  return lo, np.maximum(hi, lo + 1.0e-3 * cell)


def _fill_blind(
  mask: np.ndarray[[int, int], np.dtype[np.bool_]], dims: list[int]
) -> np.ndarray[[int, int], np.dtype[np.bool_]]:
  """`mask` (ncell, nlight) with each cell that sees no light listing the lights its 26
  neighbours see (a cell whose lattice points are all inside an object can hold lit surfaces of
  that object); a cell whose neighbourhood sees nothing either (above a ceiling, a closed room
  without lights) lists none."""
  blind = ~mask.any(axis=1)
  if not blind.any():
    return mask
  nx, ny, nz = dims
  cube = np.pad(mask.reshape(nz, ny, nx, -1), ((1, 1), (1, 1), (1, 1), (0, 0)))
  z, y, x = np.unravel_index(np.nonzero(blind)[0], (nz, ny, nx))
  seen = np.zeros((len(z), mask.shape[1]), bool)
  for dz, dy, dx in np.ndindex(3, 3, 3):
    seen |= cube[z + dz, y + dy, x + dx]
  out = mask.copy()
  out[blind] = seen
  return out


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Pairs:
  """(cell, light) pairs of a grid mask with world 0's light at each cell's centre.

  `rgb` (n, 3): the light's unshadowed illuminance at the centre (color x intensity x its
  emission lobe over the squared distance, not below a cell's; directional lights: their lux);
  `key` (n,): its luminance (inf for directional lights: strongest first); `toward` (n, 3): the
  unit direction to the light; `farable` (n,): the light may move to the cell's far field
  (positional, inverse square at any distance, no classic attenuation or IES profile, at least
  FAR_CELLS cells away).
  """

  cells: np.ndarray[[int], np.dtype[np.intp]]
  ids: np.ndarray[[int], np.dtype[np.intp]]
  rgb: np.ndarray[[int, 3], np.dtype[np.float32]]
  key: np.ndarray[[int], np.dtype[np.float32]]
  toward: np.ndarray[[int, 3], np.dtype[np.float32]]
  farable: np.ndarray[[int], np.dtype[np.bool_]]


def _lobe(
  world0: np.ndarray,
  ids: np.ndarray[[int], np.dtype[np.intp]],
  toward: np.ndarray[[int, 3], np.dtype[np.float32]],
  margin: np.ndarray[[int], np.dtype[np.float32]],
) -> tuple[np.ndarray[[int], np.dtype[np.float32]], np.ndarray[[int], np.dtype[np.bool_]]]:
  """((n,) emission lobe of lights `ids` toward -`toward` (`lights._lobe`, a spot / cosine cone as a
  step at its outer angle; other lights 1), (n,) whether that lobe holds over a cell seen within
  `margin` [rad] of the direction: the cell is wholly inside or wholly outside the cone / the
  emitter's horizon)."""
  ltype = record_field(world0, "type")[ids]
  axis = record_field(world0, "dir")[ids].astype(np.float32)
  cos_t = np.clip(np.sum(-toward * axis, axis=1), -1.0, 1.0)
  lobe = np.ones(len(ids), np.float32)
  lobe = np.where(ltype == lights.LIGHT_COSINE, np.maximum(cos_t, 0.0), lobe)
  lobe = np.where(ltype == lights.LIGHT_TUBE, np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0)), lobe)
  cos_outer = record_field(world0, "cos_outer")[ids]
  coned = (ltype == lights.LIGHT_SPOT) | ((ltype == lights.LIGHT_COSINE) & (cos_outer > lights.NO_CONE))
  lobe = np.where(coned & (cos_t < cos_outer), 0.0, lobe)
  # The edge the lobe steps or falls to zero at: the cone, else a cosine emitter's horizon.
  edge = np.where(coned, np.arccos(np.clip(cos_outer, -1.0, 1.0)), np.pi / 2)
  bounded = coned | (ltype == lights.LIGHT_COSINE)
  clear = ~bounded | (np.abs(np.arccos(cos_t) - edge) > margin)
  return lobe.astype(np.float32), clear


def _pairs(
  mask: np.ndarray[[int, int], np.dtype[np.bool_]], lit: Lights, *, lo: Corner, cell: float, dims: list[int]
) -> _Pairs:
  """The `_Pairs` of `mask` (ncell, nlight) (a host read of world 0's packed lights; float32)."""
  world0 = host_copy(lit.lights)[0][: lit.nlight]
  cells, ids = np.nonzero(mask)
  color = record_field(world0, "color")[ids] * record_field(world0, "intensity")[ids, None]
  directional = record_field(world0, "type")[ids] == lights.LIGHT_DIRECTIONAL
  nx, ny, _nz = dims
  ijk = np.stack([cells % nx, (cells // nx) % ny, cells // (nx * ny)], axis=1)
  centre = (np.asarray(lo, np.float64) + (ijk + 0.5) * cell).astype(np.float32)
  to = record_field(world0, "pos")[ids] - centre
  d2 = np.sum(to * to, axis=1)
  dist = np.sqrt(np.maximum(d2, 1.0e-30))
  toward = to / dist[:, None]
  toward = np.where(directional[:, None], -record_field(world0, "dir")[ids], toward)
  # The cell's bounding sphere (radius sqrt(3) / 2 cells) as seen from the light.
  margin = np.arcsin(np.minimum(0.87 * cell / dist, 1.0)).astype(np.float32)
  lobe, clear = _lobe(world0, ids, toward, margin)
  rgb = color * (lobe / np.maximum(d2, cell * cell))[:, None]
  rgb = np.where(directional[:, None], color, rgb).astype(np.float32)
  key = np.where(directional, np.inf, rgb @ _LUMA.astype(np.float32)).astype(np.float32)
  plain = np.all(record_field(world0, "attenuation")[ids] == np.array([1.0, 0.0, 0.0], np.float32), axis=1)
  # An emitter that is small seen from the cell (its point form is its far field).
  size = np.linalg.norm(record_field(world0, "half")[ids], axis=1) + record_field(world0, "radius")[ids]
  farable = (
    ~directional
    & np.isinf(record_field(world0, "range")[ids])
    & plain
    & (record_field(world0, "ies")[ids] == 0)
    & (d2 >= (FAR_CELLS * cell) ** 2)
    & (size <= FAR_SIZE * dist)
    & clear
  )
  return _Pairs(cells=cells, ids=ids, rgb=rgb, key=key, toward=toward.astype(np.float32), farable=farable)


def _far_drop(p: _Pairs, ncell: int) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(n,) pairs that move to their cell's far field: farable, the weakest first, while their sum
  stays at most FAR_FRACTION of the cell's estimated illuminance (directional lights: their lux)."""
  lum = np.where(np.isinf(p.key), p.rgb.astype(np.float64) @ _LUMA, p.key.astype(np.float64))
  total = np.bincount(p.cells, weights=lum.astype(np.float64), minlength=ncell)
  order = np.lexsort((lum, p.cells))  # by cell, the weakest first
  weight = np.where(p.farable[order], lum[order], 0.0)
  cum = np.cumsum(weight)
  cells = p.cells[order]
  first = np.searchsorted(cells, cells, side="left")  # each cell's first sorted pair
  before = np.where(first > 0, cum[np.maximum(first - 1, 0)], 0.0)
  drop = np.zeros(len(order), bool)
  drop[order] = p.farable[order] & (cum - before <= FAR_FRACTION * total[cells])
  return drop


def _far_fields(
  p: _Pairs, drop: np.ndarray[[int], np.dtype[np.bool_]], ncell: int
) -> tuple[np.ndarray[[int, 3], np.dtype[np.float32]], ...]:
  """(ncell, 3) far-field color, luminance on the +axis normals and on the -axis normals (an
  ambient cube) of the dropped pairs: each light adds its illuminance x max(+-l_i, 0)."""
  cells, key, toward = p.cells[drop], p.key[drop].astype(np.float64), p.toward[drop].astype(np.float64)

  def sums(w: np.ndarray[[int, 3], np.dtype[np.float64]]) -> np.ndarray[[int, 3], np.dtype[np.float32]]:
    """(ncell, 3) per-cell sums of the dropped pairs' rows of `w`."""
    return np.stack([np.bincount(cells, weights=w[:, k], minlength=ncell) for k in range(3)], 1).astype(np.float32)

  return (
    sums(p.rgb[drop].astype(np.float64)),
    sums(key[:, None] * np.maximum(toward, 0.0)),
    sums(key[:, None] * np.maximum(-toward, 0.0)),
  )


def _strongest_first(p: _Pairs, keep: np.ndarray[[int], np.dtype[np.bool_]]) -> np.ndarray[[int], np.dtype[np.int32]]:
  """The cells' light lists of the kept pairs (row-major: grouped by cell), each strongest first at
  its cell's centre (`_Pairs.key`: the directional lights' first of all); ties keep the light order.

  The capped light loops rank a cell's lights in this order (`sources._capped`): its K
  strongest fill the slots in the first iterations, so the later ones seldom take the slot update
  (divergent across a warp).
  """
  cells, ids, key = p.cells[keep], p.ids[keep], p.key[keep]
  order = np.lexsort((-key, cells))
  return ids[order].astype(np.int32)


def _boxes(lit: Lights, *, flood_dark: bool) -> light_flood.Boxes:
  """World 0's packed lights (row 0 of a shared table) as the flood seeds them (a host read); a
  light that emits nothing (`emits`) is not flooded unless `flood_dark`: it is listed in no cell."""
  world0 = host_copy(lit.lights)[0][: lit.nlight]
  light_type, light_pos = record_field(world0, "type").astype(np.int32), record_field(world0, "pos").astype(np.float64)
  reach = np.linalg.norm(record_field(world0, "half"), axis=1) + record_field(world0, "radius")
  bright = (record_field(world0, "intensity") > 0.0) & (record_field(world0, "color").max(axis=1) > 0.0)
  on = record_field(world0, "active").astype(bool) & (bright | flood_dark)
  return light_flood.Boxes(
    active=on, directional=light_type == lights.LIGHT_DIRECTIONAL, centre=light_pos, extent=reach.astype(np.float64)
  )


def _cell_size(extent: np.ndarray[[3], np.dtype[np.float64]], cell: float, nflood: int) -> float:
  """The cell [m] of a grid over a box of `extent` [m], at least `cell`: grown (x1.25 steps) to
  at most MAX_CELLS_PER_AXIS / MAX_CELLS cells and at most MAX_TESTS lattice point x light tests
  over `nflood` flooded lights."""
  cell = max(cell, float(extent.max()) / MAX_CELLS_PER_AXIS)

  def cells(c: float) -> list[int]:
    return [int(math.ceil(e / c)) for e in extent]

  while math.prod(cells(cell)) > MAX_CELLS or math.prod(2 * d + 1 for d in cells(cell)) * nflood > MAX_TESTS:
    cell *= 1.25
  return cell


def _chunks(flooded: np.ndarray[[int], np.dtype[np.bool_]], nchunk: int) -> list[int]:
  """The first light of each chunk of `nchunk` consecutive lights the flood runs: every chunk
  starts at a flooded light (lights that are off or dark cost no chunk, hence no flood rounds)."""
  starts: list[int] = []
  light0 = 0
  ids: list[int] = np.flatnonzero(flooded).tolist()
  while (i := bisect.bisect_left(ids, light0)) < len(ids):
    starts.append(ids[i])
    light0 = ids[i] + nchunk
  return starts


def _flood_lights(
  trace: Callable[[Pass, int], None],
  boxes: light_flood.Boxes,
  *,
  lo: Corner,
  cell: float,
  dims: list[int],
  device: wp.Device,
) -> np.ndarray[[int, int], np.dtype[np.bool_]]:
  """(ncell, nlight) whether each light reaches each cell of the grid (`lo`, `cell`, `dims`): the
  active lights of `boxes` flooded in chunks of consecutive lights (`light_flood.flood`, the
  chunk's pairs traced by `trace`)."""
  nlight = len(boxes.active)
  visible = wp.zeros((math.prod(dims), nlight), dtype=wp.int32, device=device)
  npoint = math.prod(2 * d + 1 for d in dims)
  g = Pass()
  g.origin, g.cell, g.dims = wp.vec3(*lo), cell, wp.vec3i(*dims)
  nchunk = max(1, min(nlight, light_flood.CHUNK_PAIRS // npoint))
  for light0 in _chunks(boxes.active, nchunk):
    g.light0, g.nchunk = light0, min(nchunk, nlight - light0)
    first = light_flood.seeds(boxes, light0=light0, nchunk=g.nchunk, lo=lo, cell=cell, dims=dims)
    light_flood.flood(trace, g, first=first, npoint=npoint, visible=visible)
  return host_copy(visible) != 0


def build[NInstance: IntVar](
  *,
  geo: SoftwareGeometry,
  lit: Lights,
  shape_types: tuple[int, ...],
  pos: Points[NInstance],
  radius: Values[NInstance],
  cell: float,
  area: bool,
  trees: Trees,
  tracer: GridTracer | None,
  alpha_cutout: bool,
  live_uv: bool = False,
  world_uv: bool = False,
  dynamic_lights: tuple[int, ...] = (),
  flood_dark: bool = False,
) -> Grid:
  """The light visibility grid (`Grid`, module doc) of world 0, with cells of at least `cell` [m].

  `geo` and `lit` are world 0's current kernel inputs (the scene BVH refitted, every light on),
  `pos` / `radius` world 0's instance centres and bounding radii (infinite: not bounding the
  grid); `shape_types`, `live_uv`, `world_uv`: the software kernel's ray queries
  (`visibility_kernel`); `area`: area lights are compiled in (tested against their centre and
  four extreme points, from their emitting side); directional lights are listed in every cell
  (module doc); lights that emit nothing in world 0 (zero intensity or color: street lamps
  authored off) in none, at no bake cost, unless `flood_dark` (lit later, they invalidate their
  row: `check_reach`);
  `trees`: whether the per-world and the shared-instance BVHs hold primitives (Features);
  `tracer`: traces the pairs of a `Pass` elsewhere (the OptiX visibility program,
  `trace.optix.launch`), None: the software kernel; `alpha_cutout`: the rays pass cut-out texels
  (Features.alpha_cutout); `dynamic_lights`: the movable light IDs (no flood, listed once after
  the cells' lists); `flood_dark`: the lights' emission is written later (per world, a simulator):
  the dark ones are flooded too, so lighting one keeps its row's lists.
  """
  nlight = lit.nlight
  boxes = _boxes(lit, flood_dark=flood_dark)
  # Skip flood work for movable lights; they are appended once at each shading point.
  boxes.active[list(dynamic_lights)] = False
  lo, hi = _bounds(pos, radius, boxes, float(cell))
  extent = np.maximum(hi - lo, 1.0e-3)
  nflood = int(boxes.active.sum())
  cell = _cell_size(extent, float(cell), nflood)
  dims = [max(int(math.ceil(e / cell)), 1) for e in extent]
  if tracer is None:
    tracer = functools.partial(
      light_flood.software_trace,
      visibility_kernel(shape_types, area=area, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv),
      geo,
    )
  visible = _flood_lights(functools.partial(tracer, lit), boxes, lo=lo, cell=cell, dims=dims, device=lit.lights.device)
  mask = _fill_blind(visible, dims)
  mask[:, boxes.directional & boxes.active] = True
  # Movable lights (directional ones too) are evaluated once, from the movable list only.
  mask[:, list(dynamic_lights)] = False
  ncell = mask.shape[0]
  pairs = _pairs(mask, lit, lo=lo, cell=cell, dims=dims)
  drop = _far_drop(pairs, ncell)
  keep = ~drop
  start = np.concatenate([[0], np.cumsum(np.bincount(pairs.cells[keep], minlength=ncell))]).astype(np.int32)
  far = np.stack(_far_fields(pairs, drop, ncell), axis=1).reshape(-1, 3) if drop.any() else np.zeros((0, 3), np.float32)
  return assemble(
    origin=wp.vec3(*lo),
    cell=cell,
    dims=wp.vec3i(*dims),
    start=start,
    lights=_strongest_first(pairs, keep),
    dynamic=np.asarray(dynamic_lights, np.int32),
    snapshot=wp.clone(lit.lights[:1]),
    far=far,
    far_members=np.unique(pairs.ids[drop]).astype(np.int32),
    nlight=nlight,
    unflooded=np.flatnonzero(~boxes.active).astype(np.int32),  # movable ones: MOVABLE wins
  )


def to_arrays(grid: Grid) -> dict[str, AnyHostArray]:
  """A grid as host arrays (the bake cache's entry, `from_arrays`)."""
  return {
    "origin": np.asarray(grid.origin, np.float32),
    "cell": np.asarray(grid.cell, np.float64),
    "dims": np.asarray(grid.dims, np.int32),
    "start": host_copy(grid.start),
    "lights": host_copy(grid.lights),
    "snapshot": host_copy(grid.snapshot),
    "dynamic_mask": host_copy(grid.dynamic_mask),
    "far": host_copy(grid.far),
  }


def from_arrays(a: Mapping[str, AnyHostArray], *, device: wp.Device) -> Grid:
  """The grid of the arrays of `to_arrays` (bit-identical), on `device`."""
  return Grid(
    origin=wp.vec3(*a["origin"]),
    cell=float(a["cell"]),
    dims=wp.vec3i(*a["dims"]),
    start=wp.array(a["start"], dtype=int, device=device),
    lights=wp.array(a["lights"], dtype=int, device=device),
    snapshot=wp.array(a["snapshot"], dtype=lights.Light, device=device),
    valid=wp.ones(2, dtype=wp.int32, device=device),
    dynamic_mask=wp.array(a["dynamic_mask"], dtype=wp.int32, device=device),
    far=wp.array(np.asarray(a["far"], np.float32).reshape(-1, 3), dtype=wp.vec3, device=device),
  )


@wp.func
def emits(
  # In:
  light: lights.Light,
) -> bool:
  """Whether a packed light emits (positive intensity and color; `_boxes` tests the same)."""
  return light.intensity > 0.0 and wp.max(wp.max(light.color[0], light.color[1]), light.color[2]) > 0.0


@wp.func
def same_reach(
  # In:
  current: lights.Light,
  built: lights.Light,
) -> bool:
  """Whether a packed light still has the footprint used by the visibility bake.

  All lights are baked active. Color/intensity and disabling a light cannot enlarge its
  geometric reach; position, orientation, size, range and cone changes invalidate it (a light
  dark at the bake and listed nowhere is `check_reach`'s UNFLOODED case). A directional light is
  listed in every cell (`build`): turned (an environment sun under a per-world `env.yaw`), it
  keeps its reach.
  """
  if current.type == wp.static(lights.LIGHT_DIRECTIONAL) and built.type == wp.static(lights.LIGHT_DIRECTIONAL):
    return True
  return (
    current.type == built.type
    and current.shape == built.shape
    and current.pos == built.pos
    and current.dir == built.dir
    and current.range == built.range
    and current.radius == built.radius
    and current.cos_outer == built.cos_outer
    and current.cone_start == built.cone_start
    and current.cone_scale == built.cone_scale
    and current.axis_x == built.axis_x
    and current.axis_y == built.axis_y
    and current.half == built.half
  )


@wp.func
def same_power(
  # In:
  current: lights.Light,
  built: lights.Light,
) -> bool:
  """Whether a far-field light still emits what the far fields were built with (on, the build's
  color and intensity)."""
  return current.active == built.active and current.color == built.color and current.intensity == built.intensity


@wp.kernel
def check_reach(
  # Model:
  snapshot: wp.array2d[lights.Light],
  # In:
  current: wp.array2d[lights.Light],
  dynamic_mask: wp.array[wp.int32],
  # Out:
  valid: wp.array[wp.int32],
):
  """One thread per (light row, light): a static light whose reach changed, or an unflooded one
  (dark at the bake, listed nowhere) that emits, invalidates its row (undeclared motion uses every
  light, including on graph replay), a far-field light whose emission changed its row's far fields
  (`Grid.valid`'s second half); `valid` starts at 1."""
  row, light = wp.tid()
  kind = dynamic_mask[light]
  if kind != wp.static(MOVABLE) and not same_reach(current[row, light], snapshot[0, light]):
    valid[row] = wp.int32(0)
  if kind == wp.static(UNFLOODED) and current[row, light].active and emits(current[row, light]):
    valid[row] = wp.int32(0)
  if kind == wp.static(FAR_MEMBER) and not same_power(current[row, light], snapshot[0, light]):
    valid[current.shape[0] + row] = wp.int32(0)


def checked(grid: Grid, current: wp.array[Light, [int, int]], nlight: int) -> Grid:
  """`grid` with `Grid.valid` refreshed for the packed light rows `current` (`nlight` lights):
  `check_reach`, without host reads (a fill and one launch per frame). The same grid, unless the
  row count changed (a new `valid` array: the caller re-wires `Lights`)."""
  rows = current.shape[0]
  if grid.valid.shape[0] != 2 * rows:
    grid = dataclasses.replace(grid, valid=wp.empty(2 * rows, dtype=wp.int32, device=current.device))
  grid.valid.fill_(1)
  if nlight:
    wp.launch(
      check_reach,
      dim=(rows, nlight),
      inputs=[grid.snapshot, current, grid.dynamic_mask],
      outputs=[grid.valid],
      device=current.device,
    )
  return grid
