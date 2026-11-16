"""Ordered (nearest-first) traversal of MuJoCo Warp's scene BVH.

`wp.bvh_query_ray` visits children in a fixed order and tests a node against the current
closest hit only when it is popped. This traversal tests both children when a node is
expanded, descends into the nearer one first and stores each pushed child's entry distance, so a
subtree behind the closest hit found so far is skipped without loading it. Each primitive goes
through the same candidate pieces as the unordered traversal (`instances.py`: MuJoCo Warp's
intersections, alpha skip, back-face rule and tie-break), so hits are identical.

BVH nodes are read through the native accessors of `nodes.py` (kernels need
`nodes.BUILD_OPTIONS`). Scenes with deformables use the unordered traversal
(`core/build/variants/resolve.py`).
"""

from __future__ import annotations

import warp as wp

from raywarp.core.kernels import native
from raywarp.core.kernels.abi import counters
from raywarp.core.kernels.abi.rays import Hit, Ray, SoftwareFields, no_hit
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.trace.accel.cast import count_walk
from raywarp.core.trace.accel.instances import (
  SHARED_TREE,
  WORLD_TREE,
  BvhTrees,
  CutAway,
  closer,
  make_candidate,
  make_tree_on,
  tree_root,
  widen,
)
from raywarp.core.trace.accel.nodes import (
  STACK,
  FloatStack,
  IntStack,
  float_at,
  float_store,
  int_at,
  int_store,
  ray_aabb,
  reciprocal,
)

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class NearStack:
  """Pending nodes of the ordered traversal (the slot of their halves in the packed children,
  `packed.py`) with their entry distances.

  Dynamically indexed, so it lives in local memory; the scalar state is the `Cursor`'s (registers).
  Warp's SAH builder permits leaves at depth 32: keeping the current child in the cursor fits its
  32 pending siblings without dropping the last far subtree.
  """

  node: IntStack
  t: FloatStack


@wp.struct
class Cursor:
  """Scalar state of the ordered traversal, read at constant offsets only (registers): the stack
  depth, the next child kept outside the stack (its reference, as `native.packed_child` gives it,
  and its entry distance), the current leaf's slots, and the traversed tree's packed children and
  primitive array. `nodes` / `drops`: nodes tested and children dropped by a full stack, read only
  by counted builds (`Features.counters`; dead stores otherwise)."""

  count: int
  has_next: int
  next_t: float
  next_left: int
  next_right: int
  next_leaf: int
  slot: int
  slot_end: int
  halves: wp.uint64
  prims: wp.uint64
  nodes: int
  drops: int


@wp.func(inline=True)
def start_ordered(
  # Out:
  c: wp.ref[Cursor],
  # In:
  tree_id: wp.uint64,
  halves: wp.uint64,
  root: int,
  origin: wp.vec3,
  rcp_dir: wp.vec3,
  cut: float,
):
  """Resets `c` to the tree below node `root` (empty when the ray misses its box before `cut`);
  `halves`: the tree's packed children."""
  c.halves = halves
  c.prims = native.bvh_prims(tree_id)
  c.count = 0
  c.has_next = 0
  c.slot = 0
  c.slot_end = 0
  # The root is nobody's packed child: its own node, read once.
  node = native.bvh_node_at(native.bvh_lowers(tree_id), native.bvh_uppers(tree_id), root)
  c.nodes += 1
  if ray_aabb(origin, rcp_dir, node.lower, node.upper, cut) >= 0.0:
    c.has_next = 1
    c.next_t = 0.0
    c.next_left = wp.where(node.leaf != 0, node.left, root)
    c.next_right = node.right
    c.next_leaf = node.leaf


@wp.func(inline=True)
def expand(
  # Out:
  c: wp.ref[Cursor],
  s: wp.ref[NearStack],
  # In:
  index: int,
  origin: wp.vec3,
  rcp_dir: wp.vec3,
  cut: float,
):
  """Tests both children of inner node `index` (one 64-byte packed read) against the ray before
  `cut`: stacks the entered far one, keeps the entered near one for immediate descent.

  The validated builder caps each traced subtree at depth 32: an inner node has at most 31
  pending siblings, so its far child fits even when descent did not pop a stack entry.
  """
  slot = 4 * index
  a = native.packed_child(c.halves, slot)
  b = native.packed_child(c.halves, slot + 2)
  ta = ray_aabb(origin, rcp_dir, a.lower, a.upper, cut)
  tb = ray_aabb(origin, rcp_dir, b.lower, b.upper, cut)
  c.nodes += 2
  # The right child first when it is entered nearer (or alone).
  swap = tb >= 0.0 and (ta < 0.0 or tb < ta)
  near_t = wp.where(swap, tb, ta)
  far_t = wp.where(swap, ta, tb)
  if far_t >= 0.0:
    if c.count < wp.static(STACK):
      int_store(s.node, c.count, wp.where(swap, slot, slot + 2))
      float_store(s.t, c.count, far_t)
      c.count += 1
    else:
      c.drops += 1
  if near_t >= 0.0:
    c.has_next = 1
    c.next_t = near_t
    c.next_left = wp.where(swap, b.left, a.left)
    c.next_right = wp.where(swap, b.right, a.right)
    c.next_leaf = wp.where(swap, b.leaf, a.leaf)


@wp.func(inline=True)
def next_ordered(
  # Out:
  c: wp.ref[Cursor],
  s: wp.ref[NearStack],
  # In:
  origin: wp.vec3,
  rcp_dir: wp.vec3,
  cut: float,
) -> int:
  """Next primitive in nearest-first order whose leaf the ray enters before `cut`, -1 when done."""
  while c.slot >= c.slot_end:
    entry = c.next_t
    left = c.next_left
    right = c.next_right
    leaf = c.next_leaf
    if c.has_next != 0:
      c.has_next = 0
    else:
      if c.count == 0:
        return -1
      c.count -= 1
      entry = float_at(s.t, c.count)
      # A node entered beyond the closest hit found since it was pushed is skipped unread.
      if entry < cut:
        node = native.packed_child(c.halves, int_at(s.node, c.count))
        left = node.left
        right = node.right
        leaf = node.leaf
    if entry < cut:
      if leaf != 0:
        c.slot = left
        c.slot_end = right
      else:
        expand(c, s, left, origin, rcp_dir, cut)
  prim = native.bvh_primitive_at(c.prims, c.slot)
  c.slot += 1
  return prim


@specialize
def make_ordered_cast(
  *,
  types: tuple[int, ...],
  trees: BvhTrees,
  two_sided: bool,
  counted: bool,
  cut_away: CutAway | None,
  bound: int = 0,
  world_optics: bool = False,
):
  """cast_ray(geo, worldid, ray) -> closest Hit in nearest-first order; the arguments of
  `cast.make_cast_ray` (closest hits only: any-hit queries walk unordered, `query.make_trace`).

  Tree order: the world's own tree first (its near hits cut the shared traversal short).
  """
  world_tree, shared_tree = trees[0], trees[1]
  ntree = int(world_tree) + int(shared_tree)
  order = (WORLD_TREE, SHARED_TREE) if world_tree else (SHARED_TREE, SHARED_TREE)
  candidate = make_candidate(
    types=types,
    first_hit=False,
    trees=trees,
    two_sided=two_sided,
    counted=counted,
    cut_away=cut_away,
    bound=bound,
    world_optics=world_optics,
  )
  tree_on = make_tree_on(bound=bound)

  @wp.func
  @device_generic
  def cast_ray[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    ray: Ray,
  ) -> Hit:
    """Closest hit of `ray` in this world's trees (instance -1: none)."""
    best = no_hit(ray.max_dist)
    rcp_dir = reciprocal(ray.direction)
    # One stack for both trees (constructing one zero-fills it: per tree, that made the two-tree
    # traversal up to 70% slower).
    s = NearStack()
    c = Cursor()
    ncandidate = int(0)
    # A runtime loop over the trees: one inlined copy of the traversal and its candidate (a
    # `range(wp.static(...))` loop is unrolled: one copy per tree).
    k = int(0)
    while k < wp.static(ntree):
      tree = wp.where(k == 0, wp.static(order[0]), wp.static(order[1]))
      k += 1
      if not tree_on(geo, tree):
        continue
      tree_id, halves, root = tree_root(geo, worldid, tree)
      start_ordered(c, tree_id, halves, root, ray.origin, rcp_dir, widen(best.dist))
      while True:
        cut = widen(best.dist)
        prim = next_ordered(c, s, ray.origin, rcp_dir, cut)
        if prim < 0:
          break
        if wp.static(counted):
          ncandidate += 1
        closer(best, candidate(geo, worldid, tree, prim, ray, cut))
    if wp.static(counted):
      count_walk(geo, worldid, wp.static(counters.SCENE_CLOSEST_WALKS), c.nodes, ncandidate, c.drops)
    return best

  return cast_ray
