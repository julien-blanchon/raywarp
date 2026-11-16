"""What a scene traversal does with one BVH primitive: the pieces shared by `cast.py` and
`ordered.py`.

A traversal (`cast.make_cast_ray`, `ordered.make_ordered_cast`) only decides which primitive comes
next (its `next candidate` func). Each candidate then goes through `make_candidate`'s func:

1. `flex_primitive`: a deformable primitive of the world tree goes to `intersect_flex`;
2. `resolve` (`make_resolve`): primitive -> instance; transparent or hidden instances are skipped;
3. `tree_pose` (`make_tree_pose`): the instance's pose (per-world, or shared by every world);
4. `intersect` (`make_intersect`): per-type ray intersection, branches of absent types removed
   with `wp.static`, back faces culled as the ray asks;

and the traversal merges the candidate `Hit` into its closest one with `closer` (or returns it at
once for any-hit queries). A skipped candidate is a miss (distance -1): `continue` cannot cross a
func.

Trees: 0 is the world tree (`SoftwareFields.bvh_id`, one group per world: instances then deformable
primitives), 1 the tree of the instances shared by every world (`SoftwareFields.shared_bvh_id`).
`instance_pose` / `pose_column` are the pose rules of the shading code, OptiX and the analytic
planes (`query.py`), which read `GeometryFields.instance_rot` rather than the trace frames.

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0, `_src/ray.py` `ray_bvh`'s
loop body): the per-type dispatch, alpha skip, back-face rule and tie-break are unchanged (hits are
identical). Changed: one func per step over geometry / `Ray` / `Hit` (it was inlined twice, in
the unordered and the ordered traversal), the shared tree, compact pose slots, per-world
visibility and two-sided instances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import warp as wp

from raywarp._base.types import TWO_SIDED, ShapeType
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.counters import BLAS_NODES
from raywarp.core.kernels.abi.rays import (
  FLEX_HIT,
  GeometryFields,
  Hit,
  Ray,
  SoftwareFields,
  no_hit,
  offset_floor,
  ray_offset,
  world_twosided,
)
from raywarp.core.kernels.abi.tables import batched, in_table, instance_rgba, instance_visible, material_rgba, table_row
from raywarp.core.kernels.generic import compiled_out, device_generic, specialize
from raywarp.core.trace.accel import cutout, meshes
from raywarp.core.trace.accel.nodes import MeshWalk, counter_address, mesh_walk_next, mesh_walk_start
from raywarp.core.trace.accel.primitives import (
  RAY_TOL_ABS,
  RAY_TOL_REL,
  ray_box,
  ray_capsule,
  ray_cylinder,
  ray_ellipsoid,
  ray_plane,
  ray_sphere,
)

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# The tree ids of the traversals.
WORLD_TREE: Final = 0
SHARED_TREE: Final = 1
# The instance types intersected through a BLAS (the others analytically).
_BLAS_TYPES: Final = frozenset((int(ShapeType.MESH), int(ShapeType.SDF), int(ShapeType.HFIELD), int(ShapeType.FLEX)))

# (world_tree, shared, compact_poses, visibility) of `Features.trees` (no analytic planes).
type BvhTrees = tuple[bool, bool, bool, bool]
# tree_on(geo, tree) -> whether the traversal walks BVH tree `tree` (`make_tree_on`).
type TreeOn = Callable[[SoftwareFields, int], bool]
# candidate(geo, worldid, tree, prim, ray, max_t) -> Hit.
type Candidate = Callable[[SoftwareFields, int, int, int, Ray, float], Hit]


@specialize
def make_counts(*, counted: bool):
  """counts(geo, worldid): the address the mesh queries add their traversal counts to
  (`nodes.counter_address`), 0 when not `counted` (Features.counters)."""

  @wp.func(inline=True)
  @device_generic
  def counts[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
  ) -> wp.uint64:
    """World `worldid`'s mesh-query counters, or 0."""
    if wp.static(counted):
      return counter_address(geo.counters, worldid, wp.static(BLAS_NODES))
    return wp.uint64(0)

  return counts


@specialize
def pose_column(*, compact: bool, bound: int = 0):
  """column(geo, instance): per-world pose column of a per-world instance (`compact`:
  Features.compact_poses; `bound`: Features.bound, `instance_pose`)."""
  gate = toggles.bound_on(bound)

  @wp.func
  @device_generic
  def column[G: GeometryFields](
    # In:
    geo: G,
    gi: int,
  ) -> int:
    """Per-world pose column of instance `gi` (its pose slot, or `gi` without compact poses)."""
    if wp.static(compact):
      if gate(geo.bits, wp.static(toggles.GEO_COMPACT)):
        return geo.pose_slot[gi]
    return gi

  return column


@wp.func(inline=True)
@device_generic
def tree_root[G: SoftwareFields](
  # In:
  geo: G,
  worldid: int,
  tree: int,
) -> tuple[wp.uint64, wp.uint64, int]:
  """(BVH id, packed children, root node) of `tree` for world `worldid`."""
  if tree == SHARED_TREE:
    return geo.shared_bvh_id, geo.shared_nodes, geo.shared_root
  return geo.bvh_id, geo.bvh_nodes, geo.group_root[worldid]


@wp.func(inline=True)
@device_generic
def flex_primitive[G: SoftwareFields](
  # In:
  geo: G,
  worldid: int,
  tree: int,
  prim: int,
) -> int:
  """Deformable primitive index of BVH primitive `prim`, or -1 (an instance, or the shared tree)."""
  if tree == SHARED_TREE:
    return -1
  # World w's group holds its instances, then its deformable primitives.
  local = prim - worldid * (geo.bvh_ninstance + geo.bvh_nflexprim)
  if local < geo.bvh_ninstance:
    return -1
  return local - geo.bvh_ninstance


@specialize
def make_tree_on(*, bound: int):
  """tree_on(geo, tree) -> whether the traversal walks BVH `tree` (`WORLD_TREE`, `SHARED_TREE`):
  the build word's structure bit where `bound` (Features.bound) binds it (`toggles.GEO_WORLD`,
  `GEO_SHARED`: a build compiling both trees for every scene, `kernels/probe.relight_features`),
  else every tree compiled in (a constant the compiler folds)."""
  gate = toggles.bound_on(bound)

  @wp.func(inline=True)
  @device_generic
  def tree_on[G: SoftwareFields](
    # In:
    geo: G,
    tree: int,
  ) -> bool:
    """Whether this frame's scene has BVH `tree`."""
    if tree == SHARED_TREE:
      return gate(geo.bits, wp.static(toggles.GEO_SHARED))
    return gate(geo.bits, wp.static(toggles.GEO_WORLD))

  return tree_on


@specialize
def make_resolve(*, visibility: bool, bound: int = 0):
  """resolve(geo, worldid, tree, prim) -> instance id, or -1 to skip the primitive; `bound`
  (Features.bound): the visibility table compiled in is read where the build word says so
  (`toggles.GEO_VISIBLE`)."""
  gate = toggles.bound_on(bound)

  @wp.func(inline=True)
  @device_generic
  def resolve[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    tree: int,
    prim: int,
  ) -> int:
    """Instance of BVH primitive `prim` of `tree`; -1: deformable, transparent or hidden in this
    world."""
    gi = int(0)
    if tree == SHARED_TREE:
      gi = geo.shared_ids[prim]
    else:
      local = prim - worldid * (geo.bvh_ninstance + geo.bvh_nflexprim)
      if local >= geo.bvh_ninstance:
        return -1
      gi = geo.enabled_instance_ids[local]
    # Alpha 0 hides an instance (MuJoCo's rule), in every tree.
    rgba, _ = instance_rgba(geo.color_rule, geo.instance_material, geo.instance_color, geo.material_base_color, worldid, gi)
    if rgba[3] == 0.0:
      return -1
    # Per-world visibility masks shared and per-world instances alike.
    if wp.static(visibility):
      if gate(geo.bits, wp.static(toggles.GEO_VISIBLE)):
        if not instance_visible(geo.visible_slot, geo.instance_visible, worldid, gi):
          return -1
    return gi

  return resolve


@specialize
def make_tree_pose(*, shared_tree: bool, compact: bool, meshes: bool, bound: int = 0):
  """tree_pose(geo, worldid, tree, instance) -> (position, rotation) in the trace frame.

  `meshes`: mesh (or SDF) instances are present; their rotation is read from the trace frames
  (`instance_rot_trace`, written for mesh columns only: `trace.mesh_trace_frames`), every other
  instance's from its pose. `bound` (Features.bound): compact pose slots compiled in are read
  where the build word says so (`toggles.GEO_COMPACT`).
  """
  gate = toggles.bound_on(bound)

  @wp.func(inline=True)
  @device_generic
  def tree_pose[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    tree: int,
    gi: int,
  ) -> tuple[wp.vec3, wp.mat33]:
    """Pose of instance `gi` met in `tree`: its shared slot, or its per-world column."""
    is_mesh = False
    if wp.static(meshes):
      gtype = geo.instance_type[gi]
      is_mesh = gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF)
    slotted = False
    if wp.static(compact):
      slotted = gate(geo.bits, wp.static(toggles.GEO_COMPACT))
    if wp.static(shared_tree):
      if tree == SHARED_TREE:
        if slotted:
          slot = -1 - geo.pose_slot[gi]
          return geo.shared_pos[slot], geo.shared_rot[slot]
        if is_mesh:
          # Without compact poses the shared slots are world 0's columns (`Frame.shared_poses`).
          return geo.shared_pos[gi], geo.instance_rot_trace[0, gi]
        return geo.shared_pos[gi], geo.shared_rot[gi]
    col = gi
    if slotted:
      col = geo.pose_slot[gi]
    if is_mesh:
      return geo.instance_pos[worldid, col], geo.instance_rot_trace[worldid, col]
    return geo.instance_pos[worldid, col], geo.instance_rot[worldid, col]

  return tree_pose


@wp.func(inline=True)
def culled(
  # In:
  direction: wp.vec3,
  cull: bool,
  d: float,
  n: wp.vec3,
) -> float:
  """`d`, or -1 for a back-face hit of a ray along `direction` when culling (as MuJoCo Warp: exit
  faces of an enclosing shape).

  Strict `> 0` keeps tangent hits and the zero normals of any-hit queries.
  """
  if cull and d >= 0.0 and wp.dot(direction, n) > 0.0:
    return -1.0
  return d


@specialize
def make_analytic(types: tuple[int, ...]):
  """analytic(gtype, pos, rot, size, ray) -> (distance, normal) of the analytic shape types
  present."""
  present = frozenset(types)

  @wp.func(inline=True)
  def analytic(
    # In:
    gtype: int,
    pos: wp.vec3,
    rot: wp.mat33,
    size: wp.vec3,
    ray: Ray,
  ) -> tuple[float, wp.vec3]:
    """Distance (-1: none or not an analytic shape) and normal of `ray` on a shape of type
    `gtype`."""
    d = float(-1.0)
    n = wp.vec3(0.0)
    o = ray.origin
    v = ray.direction
    if wp.static(int(ShapeType.PLANE) in present):
      if gtype == ShapeType.PLANE:
        d, n = ray_plane(pos, rot, size, o, v)
    if wp.static(int(ShapeType.SPHERE) in present):
      if gtype == ShapeType.SPHERE:
        d, n = ray_sphere(pos, size[0] * size[0], o, v)
    if wp.static(int(ShapeType.ELLIPSOID) in present):
      if gtype == ShapeType.ELLIPSOID:
        d, n = ray_ellipsoid(pos, rot, size, o, v)
    if wp.static(int(ShapeType.CAPSULE) in present):
      if gtype == ShapeType.CAPSULE:
        d, n = ray_capsule(pos, rot, size, o, v)
    if wp.static(int(ShapeType.CYLINDER) in present):
      if gtype == ShapeType.CYLINDER:
        d, n = ray_cylinder(pos, rot, size, o, v)
    if wp.static(int(ShapeType.BOX) in present):
      if gtype == ShapeType.BOX:
        d, n = ray_box(pos, rot, size, o, v)
    return d, n

  return analytic


# cut_away(geo, worldid, hit, point, direction) -> whether the alpha test discards `hit`
# (`cutout.make_cut_away`).
type CutAway = Callable[[GeometryFields, int, Hit, wp.vec3, wp.vec3], bool]
# cut_mesh(geo, worldid, instance, BLAS ids, packed nodes, mesh, pos, frame, ray, max_t, cull) ->
# (distance, normal, u, v, face) of an alpha-tested mesh.
type CutMesh = Callable[
  [SoftwareFields, int, int, wp.array1d[wp.uint64], wp.array1d[wp.uint64], int, wp.vec3, wp.mat33, Ray, float, bool],
  tuple[float, wp.vec3, float, float, int],
]


@specialize
def make_cut_mesh(*, first_hit: bool, cut_away: CutAway, counted: bool):
  """The `CutMesh` query: the mesh's BLAS walked once (`nodes.mesh_walk_next`), each candidate
  triangle alpha-tested where it is found and the discarded ones skipped (OptiX's any-hit rule):
  the closest kept hit, or with `first_hit` the first one (shadow rays). Distance -1: none."""
  counts = make_counts(counted=counted)

  @wp.func
  @device_generic
  def cut_mesh[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    ids: wp.array1d[wp.uint64],
    halves: wp.array1d[wp.uint64],
    mesh: int,
    pos: wp.vec3,
    frame: wp.mat33,
    ray: Ray,
    max_t: float,
    cull: bool,
  ) -> tuple[float, wp.vec3, float, float, int]:
    """(distance, world normal, u, v, face) of the nearest (first_hit: first) triangle of mesh
    `mesh` of instance gi at (pos, frame) the alpha test keeps."""
    lpnt, lvec = meshes.mesh_ray(pos, frame, ray.origin, ray.direction)
    walk = MeshWalk()
    mesh_walk_start(walk, ids[mesh], max_t, -1)
    candidate = no_hit(-1.0)
    candidate.instance_id = gi
    candidate.mesh_id = mesh
    d = float(-1.0)
    n = wp.vec3(0.0)
    bu = float(0.0)
    bv = float(0.0)
    face = int(-1)
    while mesh_walk_next(walk, ids[mesh], halves[mesh], lpnt, lvec, cull, counts(geo, worldid)):
      candidate.dist = walk.t
      candidate.normal = wp.normalize(frame @ walk.normal)
      candidate.bary_u = walk.u
      candidate.bary_v = walk.v
      candidate.face_id = walk.face
      if not cut_away(geo, worldid, candidate, ray.origin + ray.direction * walk.t, ray.direction):
        d = walk.t
        n = candidate.normal
        bu = walk.u
        bv = walk.v
        face = walk.face
        if wp.static(first_hit):
          break
        walk.limit = walk.t
    return d, n, bu, bv, face

  return cut_mesh


@specialize
def make_cut_analytic(*, analytic: Callable[[int, wp.vec3, wp.mat33, wp.vec3, Ray], tuple[float, wp.vec3]], cut_away: CutAway):
  """cut_analytic(geo, worldid, instance, type, pos, rot, size, ray, cull, d, n) -> (distance,
  normal) past the cut-out texels of an alpha-tested analytic shape whose first crossing is (d, n):
  a convex shape has at most two crossings (in, out), each alpha-tested where it is found."""

  @wp.func
  @device_generic
  def cut_analytic[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    gtype: int,
    pos: wp.vec3,
    rot: wp.mat33,
    size: wp.vec3,
    ray: Ray,
    cull: bool,
    d: float,
    n: wp.vec3,
  ) -> tuple[float, wp.vec3]:
    """The first crossing the alpha test keeps (distance -1: none)."""
    candidate = no_hit(-1.0)
    candidate.instance_id = gi
    beyond = ray
    for _crossing in range(2):
      if d < 0.0:
        break
      if culled(ray.direction, cull, d, n) < 0.0:
        return -1.0, n
      candidate.dist = d
      candidate.normal = n
      point = ray.origin + ray.direction * d
      if not cut_away(geo, worldid, candidate, point, ray.direction):
        return d, n
      # The next crossing, from just past this one.
      skip = d + ray_offset(point, offset_floor(geo, gi))
      beyond.origin = ray.origin + ray.direction * skip
      d, n = analytic(gtype, pos, rot, size, beyond)
      d = wp.where(d >= 0.0, d + skip, -1.0)
    return -1.0, n

  return cut_analytic


@specialize
def make_blas_hit(*, types: tuple[int, ...], first_hit: bool, counted: bool, cut_mesh: CutMesh | None):
  """blas_hit(geo, worldid, instance, type, pos, rot, ray, max_t, cull) -> (distance, normal, u,
  v, face, mesh id) of a mesh or height-field instance (distance -1: none or another type), for
  the mesh and height-field types present: their BLAS queried in the instance
  frame (`make_intersect`'s arguments). With `cut_mesh` (the alpha test), a hit on an alpha-tested
  instance is re-walked candidate by candidate (`make_cut_mesh`): only instances actually hit read
  their material, and a missed one costs no more than without the test (the walk's candidates are
  the plain query's, so no plain hit means no kept hit)."""
  has_hfield = int(ShapeType.HFIELD) in types
  counts = make_counts(counted=counted)
  alpha = cut_mesh is not None
  # Read only in `wp.static(alpha)` branches.
  cut_out = cutout.cut_out if alpha else compiled_out
  cut_walk = cut_mesh if cut_mesh is not None else compiled_out

  @wp.func(inline=True)
  @device_generic
  def blas_hit[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    gtype: int,
    pos: wp.vec3,
    rot: wp.mat33,
    ray: Ray,
    max_t: float,
    cull: bool,
  ) -> tuple[float, wp.vec3, float, float, int, int]:
    """Hit of `ray` with mesh or height-field instance gi (distance -1: none or another type)."""
    d = float(-1.0)
    n = wp.vec3(0.0)
    bu = float(0.0)
    bv = float(0.0)
    face = int(-1)
    mesh_id = int(-1)
    ids = geo.mesh_bvh_id
    halves = geo.mesh_nodes
    mesh = int(0)
    is_hfield = False
    if wp.static(has_hfield):
      if gtype == ShapeType.HFIELD:
        is_hfield = True
        ids = geo.hfield_bvh_id
        halves = geo.hfield_nodes
        mesh = in_table(batched(geo.instance_mesh, worldid, gi), geo.hfield_bvh_id.shape[0])
    if not is_hfield:
      mesh = in_table(batched(geo.instance_mesh, worldid, gi), geo.nmesh)
    # Any-hit queries take the unculled any-hit query of meshes; height fields always the closest
    # one (culled as the ray asks).
    any_query = False
    if wp.static(first_hit):
      any_query = not is_hfield
    if any_query:
      any_hit = meshes.ray_mesh_with_bvh_anyhit(
        ids, halves, mesh, pos, rot, ray.origin, ray.direction, max_t, counts(geo, worldid)
      )
      d = wp.where(any_hit, 0.0, -1.0)
    else:
      d, n, bu, bv, face, mesh_id = meshes.ray_mesh_with_bvh(
        ids, halves, mesh, pos, rot, ray.origin, ray.direction, max_t, cull, counts(geo, worldid)
      )
    if wp.static(alpha):
      if d >= 0.0:
        if cut_out(geo, worldid, gi):
          # Any-hit queries cull no mesh triangle (`mesh_any`): neither does their walk.
          d, n, bu, bv, face = cut_walk(
            geo, worldid, gi, ids, halves, mesh, pos, rot, ray, max_t, cull and not wp.static(first_hit)
          )
          mesh_id = wp.where(d >= 0.0, mesh, -1)
          if wp.static(first_hit):
            n = wp.vec3(0.0)
    if is_hfield:
      mesh_id = -1
    return d, n, bu, bv, face, mesh_id

  return blas_hit


@specialize
def make_intersect(
  *,
  types: tuple[int, ...],
  first_hit: bool,
  two_sided: bool,
  counted: bool,
  cut_away: CutAway | None,
  world_optics: bool = False,
):
  """intersect(geo, worldid, instance, pos, rot, ray, max_t) -> Hit (distance -1: none).

  `types`: the instance types present (others compile out); `first_hit`: meshes use the any-hit
  query (no surface data); `two_sided`: instances flagged in `instance_twosided` are never culled
  (`world_optics`: the world's flags, Features.world_optics);
  `counted`: the mesh queries count their nodes and triangles (Features.counters); `cut_away`: the
  alpha test of cut-out instances (Features.alpha_cutout; None: none), applied to each candidate
  surface where the traversal finds it (`make_cut_mesh`, `make_cut_analytic`).
  """
  has_blas = any(t in _BLAS_TYPES and t != int(ShapeType.FLEX) for t in types)
  has_analytic = any(t not in _BLAS_TYPES for t in types)
  analytic = make_analytic(types)
  alpha = cut_away is not None
  # Read only in `wp.static(alpha)` branches (`compiled_out` without the alpha test).
  cut_out = cutout.cut_out if cut_away is not None else compiled_out
  cut_mesh = make_cut_mesh(first_hit=first_hit, cut_away=cut_away, counted=counted) if cut_away is not None else None
  cut_analytic = make_cut_analytic(analytic=analytic, cut_away=cut_away) if cut_away is not None else compiled_out
  blas_hit = make_blas_hit(types=types, first_hit=first_hit, counted=counted, cut_mesh=cut_mesh)

  @wp.func(inline=True)
  @device_generic
  def intersect[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    pos: wp.vec3,
    rot: wp.mat33,
    ray: Ray,
    max_t: float,
  ) -> Hit:
    """Hit of `ray` with instance `gi` at (pos, rot) before `max_t`."""
    hit = no_hit(-1.0)
    hit.instance_id = gi
    gtype = geo.instance_type[gi]
    cull = ray.cull
    if wp.static(two_sided):
      if wp.static(world_optics):
        cull = cull and (world_twosided(geo, worldid, gi) & wp.static(TWO_SIDED)) == 0
      else:
        cull = cull and (geo.instance_twosided[gi] & wp.static(TWO_SIDED)) == 0
    size = batched(geo.instance_size, worldid, gi)
    d, n = analytic(gtype, pos, rot, size, ray)
    if wp.static(alpha and has_analytic):
      # Only a hit instance within the bound reads its material. Any-hit queries test the
      # crossings unculled (as their confirmations did, and OptiX's wrapped any-hit queries).
      if d >= 0.0 and d < max_t:
        if cut_out(geo, worldid, gi):
          d, n = cut_analytic(geo, worldid, gi, gtype, pos, rot, size, ray, cull and not wp.static(first_hit), d, n)
          if wp.static(first_hit):
            n = wp.vec3(0.0)
    if wp.static(has_blas):
      if gtype == ShapeType.MESH or gtype == ShapeType.SDF or gtype == ShapeType.HFIELD:
        d, n, bu, bv, face, mesh_id = blas_hit(geo, worldid, gi, gtype, pos, rot, ray, max_t, cull)
        hit.bary_u = bu
        hit.bary_v = bv
        hit.face_id = face
        hit.mesh_id = mesh_id
    hit.dist = culled(ray.direction, cull, d, n)
    hit.normal = n
    return hit

  return intersect


@specialize
def make_intersect_flex(*, first_hit: bool, counted: bool):
  """intersect_flex(geo, worldid, flex primitive, ray, max_t) -> Hit (instance FLEX_HIT, mesh = flex
  id); `counted` as `make_intersect`."""
  counts = make_counts(counted=counted)

  @wp.func(inline=True)
  @device_generic
  def intersect_flex[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    fp: int,
    ray: Ray,
    max_t: float,
  ) -> Hit:
    """Hit of `ray` with deformable primitive `fp` (an edge capsule or a whole flex mesh) of world
    `worldid`."""
    hit = no_hit(-1.0)
    hit.instance_id = wp.static(FLEX_HIT)
    flexid = geo.flex_prim_flexid[fp]
    part = geo.flex_part[flexid]
    rgba = material_rgba(geo.color_rule, geo.material_base_color, worldid, part.material, part.rgba)
    if rgba[3] == 0.0:  # invisible (alpha 0, its material's for the default rgba: shade/materials/flex.py)
      return hit
    edge_id = geo.flex_prim_edgeid[fp]
    d = float(-1.0)
    n = wp.vec3(0.0)
    bu = float(0.0)
    bv = float(0.0)
    face = int(-1)
    if edge_id >= 0:
      # 1D flex: a capsule along the edge.
      edge = geo.flex_edge[edge_id]
      adr = geo.flex_vertadr[flexid]
      row = table_row(worldid, geo.flex_vert.shape[0])  # a batched table: 1 or nworld rows
      epos, emat, esize = meshes.edge_frame(
        geo.flex_vert[row, adr + edge[0]], geo.flex_vert[row, adr + edge[1]], geo.flex_radius[flexid]
      )
      d, n = ray_capsule(epos, emat, esize, ray.origin, ray.direction)
      hit.mesh_id = flexid
    else:
      root = geo.flex_group_root[worldid, flexid]
      if wp.static(first_hit):
        any_hit = meshes.ray_flex_with_bvh_anyhit(
          geo.flex_bvh_id, geo.flex_nodes, flexid, root, ray.origin, ray.direction, max_t, counts(geo, worldid)
        )
        d = wp.where(any_hit, 0.0, -1.0)
      else:
        d, n, bu, bv, face = meshes.ray_flex_with_bvh(
          geo.flex_bvh_id, geo.flex_nodes, flexid, root, ray.origin, ray.direction, max_t, ray.cull, counts(geo, worldid)
        )
        if d >= 0.0:
          hit.mesh_id = flexid
    hit.dist = culled(ray.direction, ray.cull, d, n)
    hit.normal = n
    hit.bary_u = bu
    hit.bary_v = bv
    hit.face_id = face
    return hit

  return intersect_flex


@specialize
def make_candidate(
  *,
  types: tuple[int, ...],
  first_hit: bool,
  trees: BvhTrees,
  two_sided: bool,
  counted: bool,
  cut_away: CutAway | None,
  bound: int = 0,
  world_optics: bool = False,
):
  """candidate(geo, worldid, tree, prim, ray, max_t) -> Hit of BVH primitive `prim` (distance -1:
  none or skipped); `counted`: the mesh queries count their work (Features.counters); `cut_away`,
  `world_optics`: the alpha test, the world's two-sided flags (`make_intersect`); `bound`
  (Features.bound): the visibility table and compact poses compiled in, read where the build word
  says so (`make_resolve`, `make_tree_pose`)."""
  _world_tree, shared_tree, compact, visibility = trees
  has_flex = int(ShapeType.FLEX) in types
  resolve = make_resolve(visibility=visibility, bound=bound)
  meshes = int(ShapeType.MESH) in types or int(ShapeType.SDF) in types
  tree_pose = make_tree_pose(shared_tree=shared_tree, compact=compact, meshes=meshes, bound=bound)
  intersect = make_intersect(
    types=types, first_hit=first_hit, two_sided=two_sided, counted=counted, cut_away=cut_away, world_optics=world_optics
  )
  intersect_flex = make_intersect_flex(first_hit=first_hit, counted=counted)

  @wp.func(inline=True)
  @device_generic
  def candidate[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    tree: int,
    prim: int,
    ray: Ray,
    max_t: float,
  ) -> Hit:
    """Hit of `ray` with BVH primitive `prim` of `tree` before `max_t`."""
    if wp.static(has_flex):
      fp = flex_primitive(geo, worldid, tree, prim)
      if fp >= 0:
        return intersect_flex(geo, worldid, fp, ray, max_t)
    gi = resolve(geo, worldid, tree, prim)
    if gi < 0:
      return no_hit(-1.0)
    pos, rot = tree_pose(geo, worldid, tree, gi)
    return intersect(geo, worldid, gi, pos, rot, ray, max_t)

  return candidate


@wp.func(inline=True)
@device_generic
def translucent_caster[G: GeometryFields](
  # In:
  geo: G,
  gi: int,
) -> bool:
  """Whether instance gi passes some light to shadow rays (a shadow transmittance > 0)."""
  t = geo.instance_shadow_transmit[gi]
  return wp.max(t[0], wp.max(t[1], t[2])) > 0.0


@wp.func(inline=True)
def closer(
  # Out:
  best: wp.ref[Hit],
  # In:
  hit: Hit,
):
  """Keeps `hit` in `best` when it is nearer.

  Within the tie window the higher instance index wins, like MuJoCo's GL depth test (a deformable
  hit, instance FLEX_HIT, loses every tie).
  """
  tol = RAY_TOL_REL * wp.max(best.dist, 1.0)
  nearer = (hit.dist < best.dist - tol) or (
    wp.abs(hit.dist - best.dist) <= tol and (best.instance_id < 0 or hit.instance_id > best.instance_id)
  )
  if hit.dist >= 0.0 and nearer:
    best = hit


@wp.func(inline=True)
def widen(
  # In:
  dist: float,
) -> float:
  """Traversal cut-off of a closest distance: exclusive bounds widened so coincident hits reach the
  tie-break."""
  return dist * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS
