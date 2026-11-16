"""Unordered scene-BVH ray cast (closest hit / any hit), specialised on the shape types present.

`make_cast_ray` builds `cast_ray(geo, worldid, ray) -> Hit`: it walks the BVH trees of the
traced instances (`Features.trees`: the world tree's group of this world, the shared tree) in
Warp's unordered order and sends each primitive whose leaf box the ray enters to the candidate
pieces of `instances.py`. One node stack is reused across trees (`next_unordered`, nodes read
through `nodes.py`). Keeping the current child outside that stack fits the depth-32 trees
Warp's SAH builder can produce; its public query pushes both children and can overflow there.

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0, `_src/ray.py` `ray_bvh`);
the per-primitive arithmetic, alpha skip, back-face rule and tie-break are MuJoCo Warp's (hits are
identical). Changed: geometry / `Ray` / `Hit` structs instead of 37 arguments and a 7-tuple; the
body split into named pieces (`instances.py`); the shared-instance tree; per-world visibility and
compact pose slots; two-sided instances; back faces culled inside the mesh queries (`meshes.py`).
"""

from __future__ import annotations

import warp as wp

from raywarp.core.kernels import native
from raywarp.core.kernels.abi import counters
from raywarp.core.kernels.abi.rays import Hit, Ray, SoftwareFields, no_hit, world_transmit
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.trace.accel.instances import (
  SHARED_TREE,
  WORLD_TREE,
  BvhTrees,
  Candidate,
  CutAway,
  TreeOn,
  closer,
  make_candidate,
  make_tree_on,
  translucent_caster,
  tree_root,
  widen,
)
from raywarp.core.trace.accel.nodes import STACK, IntStack, ray_aabb, reciprocal, unordered_int_at, unordered_store

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class WalkStack:
  """Pending nodes of the unordered traversal, as the slots of their halves in the packed children
  (`packed.py`; dynamically indexed: shared or local memory).

  CUDA blocks <=128 use one shared stack per thread, so a second walk may start only after
  the first finishes. Scene candidates enter mesh queries with distinct stacks, never a walk's.
  """

  node: IntStack


@wp.struct
class Walk:
  """Scalar state of the unordered traversal, read at constant offsets only (registers): the stack
  depth, the node popped next, the current leaf's slots and the traversed tree's arrays.

  `next` is the packed slot of the node visited next (-1: pop the stack): an inner node's second
  child is visited right away, so it never needs a stack entry, and one free entry keeps both
  children. `nodes` / `drops`: nodes tested and children dropped by a full stack, read only by
  counted builds (`Features.counters`; dead stores otherwise).
  """

  count: int
  next: int
  slot: int
  slot_end: int
  halves: wp.uint64
  prims: wp.uint64
  nodes: int
  drops: int


@wp.func(inline=True)
def start_unordered(
  # Out:
  walk: wp.ref[Walk],
  s: wp.ref[WalkStack],
  # In:
  tree_id: wp.uint64,
  halves: wp.uint64,
  root: int,
  origin: wp.vec3,
  rcp_dir: wp.vec3,
  cut: float,
):
  """Resets the walk to the tree below node `root` (`halves`: the tree's packed children) and
  visits the root, which is nobody's packed child, from its own node."""
  walk.halves = halves
  walk.prims = native.bvh_prims(tree_id)
  walk.count = 0
  walk.next = -1
  walk.slot = 0
  walk.slot_end = 0
  node = native.bvh_node_at(native.bvh_lowers(tree_id), native.bvh_uppers(tree_id), root)
  walk.nodes += 1
  if ray_aabb(origin, rcp_dir, node.lower, node.upper, cut) >= 0.0:
    if node.leaf != 0:
      walk.slot = node.left
      walk.slot_end = node.right
    else:
      unordered_store(s.node, 0, 4 * root)
      walk.count = 1
      walk.next = 4 * root + 2


@wp.func(inline=True)
def next_unordered(
  # Out:
  walk: wp.ref[Walk],
  s: wp.ref[WalkStack],
  # In:
  origin: wp.vec3,
  rcp_dir: wp.vec3,
  cut: float,
) -> int:
  """Next primitive whose leaf box the ray enters before `cut`, -1 when done (Warp's
  `bvh_query_next` order)."""
  while walk.slot >= walk.slot_end:
    # Next node: the second child of the last inner node, else the top of the stack.
    slot = walk.next
    if slot >= 0:
      walk.next = -1
    else:
      if walk.count == 0:
        return -1
      walk.count -= 1
      slot = unordered_int_at(s.node, walk.count)
    node = native.packed_child(walk.halves, slot)
    walk.nodes += 1
    if ray_aabb(origin, rcp_dir, node.lower, node.upper, cut) >= 0.0:
      if node.leaf != 0:
        # A leaf hands out its primitives.
        walk.slot = node.left
        walk.slot_end = node.right
      else:
        # An inner node (`left`: its index): left pushed, right next (Warp's order: right
        # first). A popped node always frees a slot, so only a chain of `next` nodes at full
        # depth can drop a child.
        if walk.count < wp.static(STACK):
          unordered_store(s.node, walk.count, 4 * node.left)
          walk.count += 1
        else:
          walk.drops += 1
        walk.next = 4 * node.left + 2
  prim = native.bvh_primitive_at(walk.prims, walk.slot)
  walk.slot += 1
  return prim


@wp.func(inline=True)
@device_generic
def count_walk[G: SoftwareFields](
  # In:
  geo: G,
  worldid: int,
  walk_column: int,
  nodes: int,
  candidates: int,
  drops: int,
):
  """Adds one scene walk (`walk_column`), its nodes, candidates and dropped children to world
  `worldid`'s counters (Features.counters)."""
  counters.add(geo, worldid, walk_column, 1)
  counters.add(geo, worldid, wp.static(counters.SCENE_NODES), nodes)
  counters.add(geo, worldid, wp.static(counters.INSTANCE_TESTS), candidates)
  if drops > 0:
    counters.add(geo, worldid, wp.static(counters.STACK_DROPS), drops)


@specialize
def make_cast_ray(
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
  """cast_ray(geo, worldid, ray) -> Hit: the unordered traversal of `trees`, specialised at compile
  time.

  `types`: the `ShapeType` values present (absent types compile out, sparing registers).
  `first_hit`: any-hit query (shadow rays): cheaper mesh / flex queries, returns the first hit
  before `ray.max_dist` (callers test `instance_id != -1`). `trees` (`Features.trees` without the
  planes): the BVHs traversed, shared one first (its hits bound the world's query; it occludes
  most shadow rays), pose indexing, per-world visibility. `two_sided`, `world_optics`: see
  `make_intersect`.
  `counted`: the walk counts its work (`kernels/abi/counters.py`, Features.counters). `cut_away`:
  the candidates' alpha test (`instances.make_intersect`; None: none). `bound` (Features.bound): the
  trees, visibility table and compact poses compiled in, each walked or read where the build word
  says so (`toggles.GEO_WORLD`, `GEO_SHARED`, `GEO_VISIBLE`, `GEO_COMPACT`).
  """
  world_tree, shared_tree = trees[0], trees[1]
  candidate = make_candidate(
    types=types,
    first_hit=first_hit,
    trees=trees,
    two_sided=two_sided,
    counted=counted,
    cut_away=cut_away,
    bound=bound,
    world_optics=world_optics,
  )
  ntree = int(world_tree) + int(shared_tree)
  first_tree = SHARED_TREE if shared_tree else WORLD_TREE
  tree_on = make_tree_on(bound=bound)
  return _stack_cast(candidate, tree_on, ntree=ntree, first_tree=first_tree, first_hit=first_hit, counted=counted)


def _stack_cast(candidate: Candidate, tree_on: TreeOn, *, ntree: int, first_tree: int, first_hit: bool, counted: bool):
  """`make_cast_ray`'s depth-safe traversal, shared tree first when present, with one node stack;
  `tree_on`: which trees the scene has (`instances.make_tree_on`)."""

  @wp.func
  @device_generic
  def cast_ray[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    ray: Ray,
  ) -> Hit:
    """Closest (or, any-hit, first) hit of `ray` in this world's trees (instance -1: none)."""
    best = no_hit(ray.max_dist)
    column = wp.static(counters.SCENE_ANY_WALKS if first_hit else counters.SCENE_CLOSEST_WALKS)
    if wp.static(ntree > 0):
      rcp_dir = reciprocal(ray.direction)
      walk = Walk()
      s = WalkStack()
      ncandidate = int(0)
      # A runtime loop over the trees: one inlined copy of the walk and its candidate (a
      # `range(wp.static(...))` loop is unrolled: one copy per tree).
      k = int(0)
      while k < wp.static(ntree):
        tree = wp.where(k == 0, wp.static(first_tree), WORLD_TREE)
        k += 1
        if not tree_on(geo, tree):
          continue
        tree_id, halves, root = tree_root(geo, worldid, tree)
        start_unordered(walk, s, tree_id, halves, root, ray.origin, rcp_dir, widen(best.dist))
        prim = int(-1)
        while True:
          cut = widen(best.dist)
          prim = next_unordered(walk, s, ray.origin, rcp_dir, cut)
          if prim < 0:
            break
          if wp.static(counted):
            ncandidate += 1
          # Any-hit queries keep the exact bound (their meshes test it directly).
          hit = candidate(geo, worldid, tree, prim, ray, wp.where(wp.static(first_hit), best.dist, cut))
          if wp.static(first_hit):
            if hit.dist >= 0.0 and hit.dist < best.dist:
              if wp.static(counted):
                count_walk(geo, worldid, column, walk.nodes, ncandidate, walk.drops)
              return hit
          else:
            closer(best, hit)
      if wp.static(counted):
        count_walk(geo, worldid, column, walk.nodes, ncandidate, walk.drops)
    elif wp.static(counted):
      counters.add(geo, worldid, column, 1)
    return best

  return cast_ray


@specialize
def make_shadow_cast(  # noqa: C901 (the walk's compile-time forms: counted, world_optics)
  *,
  types: tuple[int, ...],
  trees: BvhTrees,
  two_sided: bool,
  counted: bool,
  cut_away: CutAway | None,
  bound: int = 0,
  world_optics: bool = False,
):
  """cast_shadow(geo, worldid, ray) -> RGB transmittance: the any-hit walk of `make_cast_ray`
  whose translucent candidates (`instances.translucent_caster`) multiply their shadow transmittance
  in and are passed (each instance is one BVH primitive: met once per walk), an opaque one ending
  it with 0 (the OptiX shadow trace's any-hit rule, `trace/optix/device/trace.make_shadow`);
  `bound`, `world_optics` (the world's transmittance, `trace.world_entry`): as `make_cast_ray`'s."""
  world_tree, shared_tree = trees[0], trees[1]
  candidate = make_candidate(
    types=types,
    first_hit=True,
    trees=trees,
    two_sided=two_sided,
    counted=counted,
    cut_away=cut_away,
    bound=bound,
    world_optics=world_optics,
  )
  ntree = int(world_tree) + int(shared_tree)
  first_tree = SHARED_TREE if shared_tree else WORLD_TREE
  tree_on = make_tree_on(bound=bound)

  @wp.func
  @device_generic
  def cast_shadow[G: SoftwareFields](  # noqa: C901 (compile-time forms)
    # In:
    geo: G,
    worldid: int,
    ray: Ray,
  ) -> wp.vec3:
    """Transmittance of `ray` through this world's trees (see `make_shadow_cast`)."""
    t = wp.vec3(1.0)
    column = wp.static(counters.SCENE_ANY_WALKS)
    if wp.static(ntree > 0):
      rcp_dir = reciprocal(ray.direction)
      walk = Walk()
      s = WalkStack()
      ncandidate = int(0)
      k = int(0)
      while k < wp.static(ntree):
        tree = wp.where(k == 0, wp.static(first_tree), WORLD_TREE)
        k += 1
        if not tree_on(geo, tree):
          continue
        tree_id, halves, root = tree_root(geo, worldid, tree)
        start_unordered(walk, s, tree_id, halves, root, ray.origin, rcp_dir, widen(ray.max_dist))
        prim = int(-1)
        while True:
          prim = next_unordered(walk, s, ray.origin, rcp_dir, widen(ray.max_dist))
          if prim < 0:
            break
          if wp.static(counted):
            ncandidate += 1
          hit = candidate(geo, worldid, tree, prim, ray, ray.max_dist)
          if hit.dist >= 0.0 and hit.dist < ray.max_dist:
            if wp.static(world_optics):
              # The world's transmittance read once (the per-instance form's two reads measured
              # slower with per-world entries on A100).
              passed = wp.vec3(0.0)
              if hit.instance_id >= 0:
                passed = world_transmit(geo, worldid, hit.instance_id)
              if wp.max(passed[0], wp.max(passed[1], passed[2])) <= 0.0:
                if wp.static(counted):
                  count_walk(geo, worldid, column, walk.nodes, ncandidate, walk.drops)
                return wp.vec3(0.0)
              t = wp.cw_mul(t, passed)
            else:
              pass_through = False
              if hit.instance_id >= 0:
                pass_through = translucent_caster(geo, hit.instance_id)
              if not pass_through:
                if wp.static(counted):
                  count_walk(geo, worldid, column, walk.nodes, ncandidate, walk.drops)
                return wp.vec3(0.0)
              t = wp.cw_mul(t, geo.instance_shadow_transmit[hit.instance_id])
            if wp.static(counted):
              counters.add(geo, worldid, wp.static(counters.TRANSLUCENT_PASSES), 1)
      if wp.static(counted):
        count_walk(geo, worldid, column, walk.nodes, ncandidate, walk.drops)
    elif wp.static(counted):
      counters.add(geo, worldid, column, 1)
    return t

  return cast_shadow
