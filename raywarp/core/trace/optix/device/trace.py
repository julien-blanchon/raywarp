"""Ray queries through OptiX: the `trace_closest` / `trace_any` pair of `query.make_trace`, and the
shadow rays through translucent casters (`make_shadow`).

All worlds share one instance acceleration structure (IAS). Each world has one cell on a fixed
grid holding the per-world instances it draws (`trace.accel.reach`: within `reach.NEAR_RADIUS` of
the world's reference position; `instances.world_bounds`). The cell's bounds define a uniform
power-of-two scale into a unit cube. A body parked or teleported beyond `reach.NEAR_RADIUS` is
hidden: it would otherwise stretch the cell and quantise the world's other geometry (to
millimetres at 1e4 m), and a second cell per world cost every query a second trace site in the
raygen (+30 % on a robot scene with per-world instances at fast). Rays are clipped in
scene coordinates, start at their entry into the cell and are transformed into it; returned
distances are converted back to scene units. Moving another world cannot change this transform,
and a world's bounds are its current geometry's (see `accel.py`): neither another world nor an
earlier frame changes its trace coordinates.

A world's results do not depend on its index either: every cell lies in one binade per axis
(`world_offset`: coordinates in [b, 2b), one float spacing), its scale is a power of two (exact)
and rays start inside it. Two identical worlds then hold the same cell coordinates up to their
offsets, which are multiples of that spacing: differences of two cell points (the analytic hit
programs' ray minus centre, `hits.py`) are exact (Sterbenz) and equal in both worlds. Only
OptiX's own world-to-instance transform of a ray (its hardware triangle test) still rounds at
the offset's magnitude: box hits are re-solved here in scene coordinates (`refine_box`); mesh
triangle hits keep that float noise (distances and barycentrics within a few spacings, and a
grazing silhouette ray can hit in one world and miss in another).
The price: cell coordinates sit at |x| in [b, 2b) instead of around 0, a spacing of b · 2^-23
cell units (1 world: 2.4e-7 of the cell's extent; 512 worlds: 1.9e-6).

Planes are infinite or larger than the rest of the scene, so they are not instanced: they are
intersected analytically here with MuJoCo Warp's `ray_plane`, before the OptiX traces (a plane hit
bounds them): their ids are run-time data (`GeometryFields.plane_ids`), their loop form a build's
(`Features.planes`: a scene without planes compiles no plane code).
Hit distances are in the units of the (unit) ray direction, as in the software traversal.

Hits by instance type (see `accel.py`); every hit program writes the hit's instance index into
the payload (a triangle instance's OptiX id is the instance itself: `hits.instance_of`):
- meshes and boxes: hardware triangle hits. A mesh hit carries its face, barycentrics and world
  face normal (packed, `core/shade/materials/octahedral.py`); a box (a scaled unit cube) gets the
  normal of the cube face it hit.
- spheres and capsules: built-in linear curves. The closest-hit program re-runs MuJoCo Warp's
  `ray_sphere` / `ray_capsule` on the hit instance (`hits.make_curve_hit`), so distances and
  normals are the software traversal's.
- ellipsoids and cylinders: custom primitives whose intersection programs run MuJoCo Warp's
  `ray_*` functions (`hits.make_intersection`).
- shared scenery (`OptixFields.rt_shared`, traced before the world's cell, in scene
  coordinates): the same programs over the shared structures (`shared=True`: shared poses,
  world 0's sizes). Its traces mark the payload's world with `SHARED_RAY`.
"""

from __future__ import annotations

import dataclasses
from typing import Final, final

import warp as wp

from raywarp._base.types import TWO_SIDED, ShapeType
from raywarp.core.kernels import native
from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi.features import PLANES_NONE
from raywarp.core.kernels.abi.rays import FLEX_HIT, AlphaTest, Hit, OptixFields, instance_pose, no_hit, world_twosided
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.kernels.generic import compiled_out, device_generic, specialize
from raywarp.core.shade.materials.octahedral import unpack_unit
from raywarp.core.trace.accel import cutout
from raywarp.core.trace.accel.instances import closer
from raywarp.core.trace.accel.primitives import RAY_TOL_REL, ray_capsule, ray_sphere
from raywarp.core.trace.accel.query import PlaneQueries, make_plane_hit, make_plane_queries

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Entries of OptixFields.rt_cell.
CELL_LOWER: Final = 0  # cell lower corner, scene coordinates
CELL_UPPER: Final = 1  # cell upper corner
CELL_SCALE: Final = 2  # uniform scene-to-cell scale, repeated in xyz
CELL_CENTER: Final = 3  # cell centre
CELL_OFFSET: Final = 4  # cell grid centre, computed with world_offset in update_cell
CELL_REF: Final = 5  # the world's reference position (`trace.accel.reach.reference`)
CELL_SIZE: Final = 6
# Entries after the worlds' cells: the shared IAS's box (lower, upper; scene coordinates), which its
# builds and refits emit (`group.InstanceGroup` `bounds`): a shared trace runs only for a ray that
# meets it (`shared_span`).
SHARED_BOX: Final = 2

# Shape types traced as built-in linear curves (linear swept spheres).
CURVE_TYPES: Final = (int(ShapeType.SPHERE), int(ShapeType.CAPSULE))
# Shape types intersected by custom-primitive programs.
CUSTOM_TYPES: Final = (int(ShapeType.ELLIPSOID), int(ShapeType.CYLINDER))
# Ray direction components below this are treated as parallel to the cell's slabs.
_PARALLEL_EPS: Final = 1.0e-20
# A world cell's and the shared box's padding (relative to the largest extent, plus absolute [scene
# units]): the slab tests' rounding never drops a hit on their faces.
CELL_PAD_REL: Final = 1.0e-3
CELL_PAD_ABS: Final = 1.0e-3
# OPTIX_RAY_FLAG_DISABLE_ANYHIT and OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES (optix_types.h).
RAY_FLAG_DISABLE_ANYHIT: Final = 1
RAY_FLAG_CULL_BACK_FACING_TRIANGLES: Final = 1 << 4
# Added to the payload's world by traces of the shared IAS (worlds stay below it).
SHARED_RAY: Final = 1 << 30
# OptiX instance id of deformable part k's identity instance (`dynamic.DeformableGases`):
# FLEX_INSTANCE + k. The instances' own ids (the instances themselves) stay below it, within
# OptiX's 28 bits of instance id (OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID).
FLEX_INSTANCE: Final = 1 << 27
# Visibility-mask bit of the culling queries' rays: the custom-primitive intersection programs cull
# back faces only for them (`hits.make_intersection`), so one build serves culling and unculled
# queries (the light grid's, translucent shadows, the alpha test's). No instance has it (below).
RAY_MASK_CULL: Final = 1 << 7
# Visibility-mask bits of the instances: per-world mesh and box instances on `WORLD_BITS`, shared
# ones on `SHARED_BITS`, the identity instances of the curve and custom-primitive GASes on
# `GAS_BITS`, the deformables' on `FLEX_BITS`. The render's queries trace every bit (255, but for
# `RAY_MASK_CULL`); the shared bakes' (`BAKE_RAY_MASK`) leave the deformables out.
WORLD_BITS: Final = 1
SHARED_BITS: Final = 2
GAS_BITS: Final = 4
FLEX_BITS: Final = 8
# Visibility mask of the shared bakes' rays (the GI programs, the light grid's visibility test):
# deformables never occlude a bake, on either backend (the software bakes hide their parts:
# `baked.Baked._hide_flex`).
BAKE_RAY_MASK: Final = 255 & ~FLEX_BITS


@wp.func
def cube_face_normal(
  # In:
  face: int,
) -> wp.vec3:
  """Outward normal of face `face` (triangles 2 face, 2 face + 1) of the unit cube
  (`mesh_inputs.unit_cube`). Selected per component: an indexed store would put the vector in
  local memory (a stack frame of 12 B per call site, kept by a module whose word has no box)."""
  axis = face // 2
  sign = wp.where(face % 2 == 0, 1.0, -1.0)
  return wp.vec3(wp.where(axis == 0, sign, 0.0), wp.where(axis == 1, sign, 0.0), wp.where(axis == 2, sign, 0.0))


@wp.func
def _binade(
  # In:
  cells: int,
) -> float:
  """The power of two b >= 2 · `cells` (>= 2): `cells` unit cells two apart fit in [b, 2b)."""
  b = float(2.0)
  while b < 2.0 * float(cells):
    b = 2.0 * b
  return b


@wp.func
def world_offset(
  # In:
  grid: wp.vec3i,
  c: int,
) -> wp.vec3:
  """Fixed centre of cell `c` in the IAS: odd integers b + 1, b + 3, ... per axis, unit cells with
  a unit gap, all inside [b, 2b) (`_binade`; see the module doc)."""
  ix = c % grid[0]
  iy = (c // grid[0]) % grid[1]
  iz = c // (grid[0] * grid[1])
  return wp.vec3(
    _binade(grid[0]) + 1.0 + 2.0 * float(ix),
    _binade(grid[1]) + 1.0 + 2.0 * float(iy),
    _binade(grid[2]) + 1.0 + 2.0 * float(iz),
  )


@wp.func
def world_reference(
  # In:
  cell: wp.array1d[wp.vec3],
  worldid: int,
) -> wp.vec3:
  """World `worldid`'s reference position (`trace.accel.reach.reference`, cached in its cell)."""
  return cell[worldid * CELL_SIZE + CELL_REF]


@wp.func
def world_scale(
  # In:
  cell: wp.array1d[wp.vec3],
  c: int,
) -> float:
  """Uniform scale of cell `c` only (a power of two); its normalized geometry fits in a unit
  cube."""
  return cell[c * CELL_SIZE + CELL_SCALE][0]


@wp.func
def cell_offset(
  # In:
  cell: wp.array1d[wp.vec3],
  c: int,
) -> wp.vec3:
  """The grid offset cached by the same per-world write that updates bounds and scale."""
  return cell[c * CELL_SIZE + CELL_OFFSET]


@wp.func
def to_cell(
  # In:
  cell: wp.array1d[wp.vec3],
  c: int,
  point: wp.vec3,
) -> wp.vec3:
  """Scene-coordinate `point` in cell `c`'s coordinates (centred before scaling, so that two
  large translated coordinates do not cancel)."""
  center = cell[c * CELL_SIZE + CELL_CENTER]
  return (point - center) * world_scale(cell, c) + cell_offset(cell, c)


@wp.func
def from_cell(
  # In:
  cell: wp.array1d[wp.vec3],
  c: int,
  point: wp.vec3,
) -> wp.vec3:
  """Cell `c`'s coordinates `point` in scene coordinates (`to_cell`'s inverse)."""
  return (point - cell_offset(cell, c)) / world_scale(cell, c) + cell[c * CELL_SIZE + CELL_CENTER]


@wp.func
def clip_to_cell(
  # In:
  cell: wp.array1d[wp.vec3],
  c: int,
  origin: wp.vec3,
  direction: wp.vec3,
  max_dist: float,
) -> wp.vec2:
  """Parameter interval [t0, t1] of the ray inside cell `c` (empty if t0 > t1)."""
  lower = cell[c * CELL_SIZE + CELL_LOWER]
  upper = cell[c * CELL_SIZE + CELL_UPPER]
  t0 = float(0.0)
  t1 = max_dist
  for k in range(3):
    dk = direction[k]
    if wp.abs(dk) < wp.static(_PARALLEL_EPS):
      if origin[k] < lower[k] or origin[k] > upper[k]:
        t0 = 1.0
        t1 = 0.0
    else:
      inv = 1.0 / dk
      a = (lower[k] - origin[k]) * inv
      b = (upper[k] - origin[k]) * inv
      t0 = wp.max(t0, wp.min(a, b))
      t1 = wp.min(t1, wp.max(a, b))
  return wp.vec2(t0, t1)


@wp.func
def shared_span(
  # In:
  cell: wp.array1d[wp.vec3],
  origin: wp.vec3,
  direction: wp.vec3,
  max_dist: float,
) -> bool:
  """Whether the ray to `max_dist` meets the shared IAS's box (the last `SHARED_BOX` entries of
  `cell`, padded as a world's cell): a shared trace that cannot hit is not launched (a sun's
  shadow ray from the floor past a robot's fixed base: one OptiX trace call less per pixel)."""
  first = cell.shape[0] - wp.static(SHARED_BOX)
  lower = cell[first]
  upper = cell[first + 1]
  pad = wp.static(CELL_PAD_REL) * wp.max(wp.max(upper[0] - lower[0], upper[1] - lower[1]), upper[2] - lower[2])
  pad = pad + wp.static(CELL_PAD_ABS)
  t0 = float(0.0)
  t1 = max_dist
  for k in range(3):
    lo = lower[k] - pad
    hi = upper[k] + pad
    dk = direction[k]
    if wp.abs(dk) < wp.static(_PARALLEL_EPS):
      if origin[k] < lo or origin[k] > hi:
        return False
    else:
      inv = 1.0 / dk
      a = (lo - origin[k]) * inv
      b = (hi - origin[k]) * inv
      t0 = wp.max(t0, wp.min(a, b))
      t1 = wp.min(t1, wp.max(a, b))
  return t0 <= t1


@specialize
def make_triangle_hit(*, has_mesh: bool, has_box: bool, compact: bool, bound: int = 0):
  """triangle_hit(geo, worldid, r, hit) -> the Hit of trace result `r` (on its instance
  `r.instance`).

  Triangles carry their barycentrics (a, b) and packed world face normal; curves and custom
  primitives a world normal (a, b, c). A mesh (or SDF) hit gets its mesh, face and barycentrics,
  a height field's its face and barycentrics (no mesh, as the software traversal's), a deformable
  part's (`FLEX_INSTANCE` + part) the software traversal's flex hit: `FLEX_HIT`, the part
  as mesh id, its face in the part's every-world mesh. `bound`: Features.bound (`instance_pose`).
  """
  pose = instance_pose(compact=compact, bound=bound)
  gate = toggles.bound_on(bound)  # the box's shape bit (`toggles.SHAPE_BOX`)

  @wp.func
  @device_generic
  def triangle_hit[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    r: native.TraceResult,
    hit: Hit,
  ) -> Hit:
    """`hit` set from trace result `r` (see `make_triangle_hit`)."""
    gi = r.instance
    n = wp.vec3(r.a, r.b, r.c)
    hit.bary_u = 0.0
    hit.bary_v = 0.0
    hit.face_id = -1
    hit.mesh_id = -1
    if wp.static(has_mesh):
      gtype = int(ShapeType.FLEX)
      if gi < wp.static(FLEX_INSTANCE):
        gtype = geo.instance_type[gi]
      # Meshes, SDFs, deformables (MESH and above) and height fields; curves and custom
      # primitives carry their world normal, boxes get theirs below.
      if gtype >= int(ShapeType.MESH) or gtype == int(ShapeType.HFIELD):
        hit.face_id = r.primitive
        # OptiX (b1, b2) weight vertices 1 and 2; MuJoCo Warp's (u, v) vertices 0 and 1.
        hit.bary_u = 1.0 - r.a - r.b
        hit.bary_v = r.a
        n = unpack_unit(r.normal)  # world face normal from the hit program (`core/shade/materials/octahedral.py`)
        if gtype == int(ShapeType.FLEX):
          hit.mesh_id = gi - wp.static(FLEX_INSTANCE)
          gi = wp.static(FLEX_HIT)
        elif gtype != int(ShapeType.HFIELD):
          hit.mesh_id = batched(geo.instance_mesh, worldid, gi)
    if wp.static(has_box):
      if gi >= 0 and gate(geo.bits, wp.static(toggles.SHAPE_BOX)):
        if geo.instance_type[gi] == int(ShapeType.BOX):
          _pb, rot = pose(geo, worldid, gi)
          n = rot @ cube_face_normal(r.primitive // 2)
    hit.instance_id = gi
    hit.dist = r.t
    hit.normal = wp.normalize(n)
    return hit

  return triangle_hit


@specialize
def make_refine_box(*, compact: bool, bound: int = 0):
  """refine_box(geo, worldid, origin, direction, side, hit): a per-world box hit on cube side
  `side` (`cube_face_normal`) gets the distance of the scene-space ray to that face's plane.

  OptiX tests the triangle with the ray in the box's frame, which its hardware computes from the
  cell coordinates (rounding at the world's offset); the scene-space distance is the same in every
  world. Other instances' hits are returned unchanged. `bound`: Features.bound (`instance_pose`).
  """
  pose = instance_pose(compact=compact, bound=bound)

  @wp.func
  @device_generic
  def refine_box[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    side: int,
    hit: Hit,
  ) -> Hit:
    """`hit` with a box hit's distance on its face plane (see `make_refine_box`)."""
    gi = hit.instance_id
    if gi < 0:  # a deformable's (`FLEX_HIT`)
      return hit
    if geo.instance_type[gi] != int(ShapeType.BOX):
      return hit
    pos, rot = pose(geo, worldid, gi)
    # The face's axis selected per component (an indexed read puts the matrix and the half sizes
    # in local memory: a 48 B stack frame, kept by a module whose word has no box).
    face = cube_face_normal(side)
    outward = rot @ face
    denom = wp.dot(outward, direction)
    if denom != 0.0:
      half = wp.dot(batched(geo.instance_size, worldid, gi), wp.cw_mul(face, face))
      t = (half - wp.dot(outward, origin - pos)) / denom
      if t > 0.0:
        hit.dist = t
    return hit

  return refine_box


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TraceSpec:
  """What the queries of one `Tracer` entry compile in (`make_trace`).

  Attributes:
    planes: `Features.planes` (PLANES_NONE: no plane code).
    has_mesh, has_box: any mesh (triangle GAS) / box (unit-cube triangle GAS) instance.
    has_instances: the per-world IAS holds any instance.
    cull: drop hits whose normal faces away from the ray (as MuJoCo Warp): triangles by ray flag,
      planes here, custom primitives in the intersection programs (`RAY_MASK_CULL`).
    has_shared: the shared IAS (`OptixFields.rt_shared`) holds any instance.
    compact: Features.compact_poses (how poses are indexed).
    visibility: Features.visibility (the per-world visibility table hides planes).
    mask: OptiX visibility mask of the queries: the instances whose mask shares a bit with it
      (255: all). Planes are tested regardless.
    anyhit: the alpha test's any-hit programs run on the cut-out instances' candidate hits
      (`programs.build`); the other instances disable them (`group.InstanceTables.flags`).
    alpha: the alpha test of the analytic planes (Features.alpha_cutout: its texture coordinates;
      None: none).
    shared_anyhit: the shared scenery's traces run the any-hit programs (the alpha test's, or the
      per-world visibility test of shared instances, `accel.OptixScene.shared_visible`) and carry
      the ray's world (the any-hit and intersection programs read it).
    two_sided: culling skips the planes flagged in `GeometryFields.instance_twosided`
      (Features.two_sided; triangles: `accel.OptixScene` instance flags, custom primitives: their
      intersection programs).
    world_optics: the planes' two-sided flags and shadow transmittance are the world's
      (Features.world_optics, `trace.world_entry`).
    counting: the occlusion queries count their outcomes (`kernels/abi/counters.py`;
      Features.counters).
    bound: Features.bound: the structures, planes, pose indexing and visibility compiled in are
      used where the build word says so (`toggles.GEO_*`, `geo.bits`: an OptiX module binds it).
      The shared traces' any-hit programs, compiled in for the shared instances' visibility alone
      (`shared_anyhit` without `anyhit`), run only where it is on (`GEO_SHARED_VISIBLE`).
    switches: Features.switches: culling compiled in (`cull`) read at run time (`toggles.setting`:
      the software path's `geo.rt.flags`, an OptiX module's word, `bound`): off, the rays take the
      mask and flags without it.
  """

  planes: int
  has_mesh: bool
  has_box: bool
  has_instances: bool
  cull: bool
  has_shared: bool
  compact: bool
  visibility: bool
  mask: int = 255
  anyhit: bool = False
  alpha: AlphaTest | None = None
  shared_anyhit: bool = False
  two_sided: bool = False
  world_optics: bool = False
  counting: bool = False
  bound: int = 0
  switches: int = 0

  @property
  def ray_mask(self) -> wp.uint32:
    """The rays' visibility mask: `mask`, with `RAY_MASK_CULL` set when culling, cleared if not."""
    return wp.uint32(self.mask | RAY_MASK_CULL if self.cull else self.mask & ~RAY_MASK_CULL)

  def ray_flags(self, *, anyhit: bool) -> wp.uint32:
    """OptiX ray flags of the closest-hit traces: triangle culling, any-hit programs disabled
    unless `anyhit` (`anyhit` per world, `shared_anyhit` shared)."""
    culling = RAY_FLAG_CULL_BACK_FACING_TRIANGLES if self.cull else 0
    return wp.uint32(culling | (0 if anyhit else RAY_FLAG_DISABLE_ANYHIT))

  @property
  def unculled(self) -> TraceSpec:
    """This spec without culling: its rays' mask and flags where the run-time switch is off."""
    return dataclasses.replace(self, cull=False)


def _planes(spec: TraceSpec) -> PlaneQueries | None:
  """The analytic planes' queries of `spec` (`query.make_plane_queries`; two-sided planes are
  skipped only when culling; `TraceSpec.alpha`: alpha-tested, a cut-out plane unculled on the
  occlusion and shadow queries), None: PLANES_NONE."""
  if spec.planes == PLANES_NONE:
    return None
  alpha = spec.alpha
  plane = (int(ShapeType.PLANE),)
  cut_away = None
  if alpha is not None:
    cut_away = cutout.make_word_cut_away(
      plane, compact=spec.compact, live_uv=alpha.live_uv, world_uv=alpha.world_uv, bound=spec.bound
    )
  hits = [
    make_plane_hit(
      cull_backfaces=spec.cull,
      compact=spec.compact,
      visibility=spec.visibility,
      two_sided=spec.two_sided and spec.cull,
      cut_away=cut_away,
      first_hit=first,
      bound=spec.bound,
      switches=spec.switches,
      world_optics=spec.world_optics,
    )
    for first in (False, cut_away is not None)
  ]
  return make_plane_queries(hits[0], hits[1], world_optics=spec.world_optics)


def _seen_only(spec: TraceSpec) -> bool:
  """Whether the shared traces run their any-hit programs for the shared instances' visibility
  alone, switched by the build word (`TraceSpec.bound`, `toggles.GEO_SHARED_VISIBLE`)."""
  return spec.shared_anyhit and not spec.anyhit and bool(spec.bound & toggles.GEO_SHARED_VISIBLE)


@wp.func(inline=True)
def tie_cutoff(
  # In:
  dist: float,
) -> float:
  """Distance beyond which every hit loses to a hit at `dist` (`trace.accel.instances.closer`).

  A farther hit `t` loses when `dist < t - RAY_TOL_REL * max(t, 1)`; the margin (3 and 2 tie
  windows instead of 1) covers the float rounding of both sides.
  """
  return dist * (1.0 + 3.0 * wp.static(RAY_TOL_REL)) + 2.0 * wp.static(RAY_TOL_REL)


@specialize
def make_closest(spec: TraceSpec):  # noqa: C901 (each structure's bound gate)
  """trace_closest(geo, worldid, origin, direction, max_dist) -> Hit (`query.make_trace`'s
  signature).

  The queries run nearest-bound first, each one's hit bounding the next one's ray: the planes
  (analytic), the shared scenery, then the world's cell, whose trace is skipped when the cell lies
  beyond the hit so far (`clip_to_cell`'s span is then empty). Every merge uses `closer`'s tie
  window (the higher instance wins), as the software traversal does.
  """
  has_instances, has_shared = spec.has_instances, spec.has_shared
  planes = _planes(spec)
  has_planes = planes is not None
  planes_hit = planes.closest if planes is not None else compiled_out
  triangle_hit = make_triangle_hit(has_mesh=spec.has_mesh, has_box=spec.has_box, compact=spec.compact, bound=spec.bound)
  refine_box = make_refine_box(compact=spec.compact, bound=spec.bound)
  flags, shared_flags = spec.ray_flags(anyhit=spec.anyhit), spec.ray_flags(anyhit=spec.shared_anyhit)
  # The shared traces' flags where the build word turns their visibility any-hit off.
  unseen_flags = spec.ray_flags(anyhit=False)
  seen_only = _seen_only(spec)
  ray_mask = spec.ray_mask
  # Culling switched off at run time: the flags and mask without it.
  open_spec = spec.unculled
  open_flags, open_shared, open_unseen = (open_spec.ray_flags(anyhit=a) for a in (spec.anyhit, spec.shared_anyhit, False))
  open_mask = open_spec.ray_mask
  setting = toggles.setting(spec.switches, spec.bound)  # culling: a tier build reads it at run time
  gate = toggles.bound_on(spec.bound)

  @wp.func
  @device_generic
  def trace_closest[G: OptixFields](  # noqa: C901 (each structure's bound gate)
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> Hit:
    """Closest hit of the planes, the shared scenery and the world's cell (see `make_closest`)."""
    rflags, sflags, uflags, mask = flags, shared_flags, unseen_flags, ray_mask
    if not setting(geo.rt.flags, geo.bits2, wp.static(toggles.BACKFACE_CULLING)):
      rflags, sflags, uflags, mask = open_flags, open_shared, open_unseen, open_mask
    plane = no_hit(max_dist)
    bound = max_dist
    if wp.static(has_planes):
      if gate(geo.bits, wp.static(toggles.GEO_PLANES)):
        plane = planes_hit(geo, worldid, origin, direction, plane)
        if plane.instance_id >= 0:
          bound = wp.min(max_dist, tie_cutoff(plane.dist))
    hit = no_hit(max_dist)
    if wp.static(has_shared):
      # The shared scenery, in scene coordinates (a ray that misses its box: no trace).
      if gate(geo.bits, wp.static(toggles.GEO_SHARED)) and shared_span(geo.rt_cell, origin, direction, bound):
        if wp.static(seen_only):
          if not gate(geo.bits, wp.static(toggles.GEO_SHARED_VISIBLE)):
            sflags = uflags
        s = native.optix_trace_closest(geo.rt_shared, origin, direction, 0.0, bound, sflags, mask, worldid + SHARED_RAY)
        if s.instance >= 0:
          hit = triangle_hit(geo, worldid, s, hit)
          # Cell hits within the tie window still reach `closer` below.
          bound = wp.min(bound, tie_cutoff(hit.dist))
    if wp.static(has_instances):
      if gate(geo.bits, wp.static(toggles.GEO_WORLD)):
        span = clip_to_cell(geo.rt_cell, worldid, origin, direction, bound)
        if span[0] <= span[1]:
          scale = world_scale(geo.rt_cell, worldid)
          start = to_cell(geo.rt_cell, worldid, origin + span[0] * direction)  # inside the cell
          r = native.optix_trace_closest(
            geo.rt_handle, start, direction, 0.0, (span[1] - span[0]) * scale, rflags, mask, worldid
          )
          if r.instance >= 0:
            r.t = span[0] + r.t / scale
            cell = triangle_hit(geo, worldid, r, no_hit(max_dist))
            if wp.static(spec.has_box):
              if gate(geo.bits, wp.static(toggles.SHAPE_BOX)):
                cell = refine_box(geo, worldid, origin, direction, r.primitive // 2, cell)
            if hit.instance_id < 0:
              hit = cell
            else:
              closer(hit, cell)  # the software traversal's tie rule: the higher instance wins
    if wp.static(has_planes):
      if hit.instance_id < 0:
        hit = plane
      else:
        closer(hit, plane)
    return hit

  return trace_closest


@specialize
def make_any(spec: TraceSpec):  # noqa: C901 (counted branches keep the disabled trace path unchanged)
  """trace_any(geo, worldid, origin, direction, max_dist) -> bool (`query.make_trace`'s
  signature)."""
  planes = _planes(spec)
  has_planes = planes is not None
  any_plane = planes.any if planes is not None else compiled_out
  has_instances, has_shared, anyhit, shared_anyhit = spec.has_instances, spec.has_shared, spec.anyhit, spec.shared_anyhit
  counting = spec.counting
  ray_mask, open_mask = spec.ray_mask, spec.unculled.ray_mask
  setting = toggles.setting(spec.switches, spec.bound)  # culling: a tier build reads it at run time
  seen_only = _seen_only(spec)
  gate = toggles.bound_on(spec.bound)

  @wp.func
  @device_generic
  def trace_any[G: OptixFields](  # noqa: C901 (optional path atomics stay beside each OptiX call)
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> bool:
    """Whether the planes, the shared scenery or the world's cell hold a hit before max_dist."""
    mask = ray_mask
    if not setting(geo.rt.flags, geo.bits2, wp.static(toggles.BACKFACE_CULLING)):
      mask = open_mask
    if wp.static(has_planes):
      if gate(geo.bits, wp.static(toggles.GEO_PLANES)):
        if any_plane(geo, worldid, origin, direction, max_dist) != -1:
          if wp.static(counting):
            counters.add(geo, worldid, wp.static(counters.PLANE_ANY_TERMINATED), 1)
          return True
    # The shared scenery first: it occludes most shadow rays.
    if wp.static(has_shared):
      if gate(geo.bits, wp.static(toggles.GEO_SHARED)) and shared_span(geo.rt_cell, origin, direction, max_dist):
        if wp.static(counting):
          counters.add(geo, worldid, wp.static(counters.SHARED_ANY_CALLS), 1)
        # With its any-hit programs (for the shared instances' visibility alone: where the build
        # word has it, `TraceSpec.bound`).
        blocked = False
        if wp.static(seen_only):
          if gate(geo.bits, wp.static(toggles.GEO_SHARED_VISIBLE)):
            world = worldid + SHARED_RAY
            blocked = native.optix_trace_any_world(geo.rt_shared, origin, direction, 0.0, max_dist, mask, world)
          else:
            blocked = native.optix_trace_any(geo.rt_shared, origin, direction, 0.0, max_dist, mask)
        elif wp.static(shared_anyhit):
          world = worldid + SHARED_RAY
          blocked = native.optix_trace_any_world(geo.rt_shared, origin, direction, 0.0, max_dist, mask, world)
        else:
          blocked = native.optix_trace_any(geo.rt_shared, origin, direction, 0.0, max_dist, mask)
        if blocked:
          if wp.static(counting):
            counters.add(geo, worldid, wp.static(counters.SHARED_ANY_HITS), 1)
          return True
    if wp.static(has_instances):
      if gate(geo.bits, wp.static(toggles.GEO_WORLD)):
        span = clip_to_cell(geo.rt_cell, worldid, origin, direction, max_dist)
        if wp.static(counting):
          if span[0] > span[1]:
            counters.add(geo, worldid, wp.static(counters.WORLD_CELL_REJECTED), 1)
        if span[0] <= span[1]:
          scale = world_scale(geo.rt_cell, worldid)
          start = to_cell(geo.rt_cell, worldid, origin + span[0] * direction)  # inside the cell
          tmax = (span[1] - span[0]) * scale
          blocked = False
          if wp.static(anyhit):
            blocked = native.optix_trace_any_world(geo.rt_handle, start, direction, 0.0, tmax, mask, worldid)
          else:
            blocked = native.optix_trace_any(geo.rt_handle, start, direction, 0.0, tmax, mask)
          if wp.static(counting):
            counters.add(geo, worldid, wp.static(counters.WORLD_ANY_CALLS), 1)
            if blocked:
              counters.add(geo, worldid, wp.static(counters.WORLD_ANY_HITS), 1)
          return blocked
    return False

  return trace_any


@specialize
def make_shadow(spec: TraceSpec):
  """trace_shadow(geo, worldid, origin, direction, max_dist) -> the RGB transmittance of a shadow
  ray through translucent casters (0: blocked): the planes (analytic: a translucent one multiplies,
  an opaque one blocks), then one `optix_trace_shadow` per structure (the shared scenery, the
  world's cell), whose any-hit and intersection programs multiply in the translucent candidates
  and whose opaque ones end it (`intrinsics.trace_shadow`). An unoccluded ray that meets no
  translucent caster costs what an occlusion query does."""
  planes = _planes(spec)
  has_planes = planes is not None
  planes_shadow = planes.shadow if planes is not None else compiled_out
  has_instances, has_shared = spec.has_instances, spec.has_shared
  ray_mask, open_mask = spec.ray_mask, spec.unculled.ray_mask
  # Culling as the occlusion queries' (`TraceSpec.cull`: back-facing triangles by ray flag, custom
  # primitives by the mask's `RAY_MASK_CULL`, planes in `plane_hit`); the any-hit programs run.
  culling = wp.uint32(RAY_FLAG_CULL_BACK_FACING_TRIANGLES if spec.cull else 0)
  setting = toggles.setting(spec.switches, spec.bound)  # culling: a tier build reads it at run time
  gate = toggles.bound_on(spec.bound)

  @wp.func
  @device_generic
  def trace_shadow[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> wp.vec3:
    """The transmittance along a shadow ray before max_dist (see `make_shadow`)."""
    cull, mask = culling, ray_mask
    if not setting(geo.rt.flags, geo.bits2, wp.static(toggles.BACKFACE_CULLING)):
      cull, mask = wp.uint32(0), open_mask
    t = wp.vec3(1.0)
    if wp.static(has_planes):
      if gate(geo.bits, wp.static(toggles.GEO_PLANES)):
        t = planes_shadow(geo, worldid, origin, direction, max_dist)
        if wp.max(t[0], wp.max(t[1], t[2])) <= 0.0:
          return t
    if wp.static(has_shared):
      if gate(geo.bits, wp.static(toggles.GEO_SHARED)) and shared_span(geo.rt_cell, origin, direction, max_dist):
        t = native.optix_trace_shadow(geo.rt_shared, origin, direction, 0.0, max_dist, cull, mask, worldid + SHARED_RAY, t)
        if wp.max(t[0], wp.max(t[1], t[2])) <= 0.0:
          return t
    if wp.static(has_instances):
      if gate(geo.bits, wp.static(toggles.GEO_WORLD)):
        span = clip_to_cell(geo.rt_cell, worldid, origin, direction, max_dist)
        if span[0] <= span[1]:
          scale = world_scale(geo.rt_cell, worldid)
          start = to_cell(geo.rt_cell, worldid, origin + span[0] * direction)  # inside the cell
          t = native.optix_trace_shadow(
            geo.rt_handle, start, direction, 0.0, (span[1] - span[0]) * scale, cull, mask, worldid, t
          )
    return t

  return trace_shadow


@wp.func
def _slab_exit(
  # In:
  p: float,
  v: float,
  half: float,
) -> float:
  """Distance along a box-frame axis at which a ray from `p` inside [-half, half] (direction
  component `v`) leaves the slab (1e30 when parallel)."""
  return wp.where(v > 0.0, (half - p) / v, wp.where(v < 0.0, (-half - p) / v, 1.0e30))


@wp.func
def _box_exit(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  half: wp.vec3,
  origin: wp.vec3,
  direction: wp.vec3,
) -> tuple[float, wp.vec3]:
  """(distance, outward world normal) of the face a ray from inside the box (`pos`, `rot`, half
  sizes `half`) leaves through: the nearest slab exit, per component (no indexed reads: local
  memory in the OptiX raygen)."""
  frame = wp.transpose(rot)
  p = frame @ (origin - pos)
  v = frame @ direction
  tx = _slab_exit(p[0], v[0], half[0])
  ty = _slab_exit(p[1], v[1], half[1])
  tz = _slab_exit(p[2], v[2], half[2])
  t = wp.min(tx, wp.min(ty, tz))
  n = wp.vec3(wp.where(t == tx, wp.sign(v[0]), 0.0), 0.0, 0.0)
  if t != tx:
    n = wp.vec3(0.0, wp.where(t == ty, wp.sign(v[1]), 0.0), wp.where(t == ty, 0.0, wp.sign(v[2])))
  return t, rot @ n


@specialize
def make_leave(spec: TraceSpec, types: tuple[int, ...]):
  """leave(geo, worldid, gi, origin, direction, hit) -> `hit` (the closest hit of a ray from
  `origin` inside instance gi), or the face the ray leaves gi through where it is nearer or ties
  and wins as the software traversal's would (`closer`: the higher instance), for the shape types
  `types` the build traces: a sphere's or a capsule's (MuJoCo Warp's `ray_sphere`, `ray_capsule`:
  the curve intersector reports none from inside), a box's (`_box_exit`: one OptiX trace leaves
  coincident faces unordered, a glass box's bottom on the floor). Back faces culled as the
  closest queries cull them (`spec`: the culling switch, the two-sided instances); other shapes
  keep `hit`. Small and scalar: it is inlined in the OptiX raygen's layer loop
  (`pixel.make_trace_pixel`), whose state across traces it must not grow."""
  pose = instance_pose(compact=spec.compact, bound=spec.bound)
  setting = toggles.setting(spec.switches, spec.bound)  # culling: a tier build reads it at run time
  cull, two_sided, world_optics = spec.cull, spec.two_sided, spec.world_optics
  sphere, box = int(ShapeType.SPHERE) in types, int(ShapeType.BOX) in types
  capsule = int(ShapeType.CAPSULE) in types

  @wp.func
  @device_generic
  def leave[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    origin: wp.vec3,
    direction: wp.vec3,
    hit: Hit,
  ) -> Hit:
    """`hit`, or gi's exit face where nearer (see `make_leave`)."""
    gtype = geo.instance_type[gi]
    if gtype != int(ShapeType.SPHERE) and gtype != int(ShapeType.BOX) and gtype != int(ShapeType.CAPSULE):
      return hit
    if wp.static(cull):
      if setting(geo.rt.flags, geo.bits2, wp.static(toggles.BACKFACE_CULLING)):
        if wp.static(two_sided):
          if wp.static(world_optics):
            bits = world_twosided(geo, worldid, gi)
          else:
            bits = geo.instance_twosided[gi]
          if (bits & wp.static(TWO_SIDED)) == 0:
            return hit
        else:
          return hit
    pos, rot = pose(geo, worldid, gi)
    size = batched(geo.instance_size, worldid, gi)
    d = float(-1.0)
    n = wp.vec3(0.0)
    if wp.static(sphere):
      if gtype == int(ShapeType.SPHERE):
        d, n = ray_sphere(pos, size[0] * size[0], origin, direction)
    if wp.static(box):
      if gtype == int(ShapeType.BOX):
        d, n = _box_exit(pos, rot, size, origin, direction)
    if wp.static(capsule):
      if gtype == int(ShapeType.CAPSULE):
        d, n = ray_capsule(pos, rot, size, origin, direction)
    tol = wp.static(RAY_TOL_REL) * wp.max(hit.dist, 1.0)
    tie = wp.abs(d - hit.dist) <= tol and (hit.instance_id < 0 or gi > hit.instance_id)
    if d >= 0.0 and (d < hit.dist - tol or tie):
      hit.instance_id = gi
      hit.dist = d
      hit.normal = n
      hit.bary_u = 0.0
      hit.bary_v = 0.0
      hit.face_id = -1
      hit.mesh_id = -1
    return hit

  return leave


@specialize
def make_trace(spec: TraceSpec):
  """(trace_closest, trace_any) of `spec`, with the signatures of `query.make_trace`."""
  return make_closest(spec), make_any(spec)
