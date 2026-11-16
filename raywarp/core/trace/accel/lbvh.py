"""In-place LBVH rebuild of the world BVH (`bvh.SceneBvh`), with scratch owned by the caller and
every launch gated on the device.

The world BVH groups its primitives by world, `group_size` per world, world w's at
[w · group_size, (w + 1) · group_size). A rebuild gives it a new topology from the current boxes:
Warp's grouped LBVH (Karras 2012, `warp/native/bvh.cu`: key = world << 32 | 30-bit Morton code of
the box centre, the same bottom-up hierarchy with its group-preserving parent choice, leaves
packed below `BVH_QUERY_STACK_SIZE` levels), written into the `wp.Bvh`'s own node buffers. Two
differences from `wp.Bvh.rebuild`:
* no allocation: Warp's builder allocates its scratch on every call, which a CUDA graph's
  conditional body cannot hold, so a captured frame could not rebuild. The scratch is allocated
  here once;
* every kernel reads the cadence's gate (`cadence.Cadence.due`) and returns when it is closed:
  an eager frame launches the rebuild every frame and it runs on the device's schedule, without
  `wp.capture_if`'s per-frame host read of the gate. A captured frame records the launches as a
  `wp.capture_if` body: replays skip them on the frames without a rebuild.
The Morton codes are relative to each world's own centre bounds (Warp's span every world: worlds
far apart, as in tiled environment grids, then share few code bits); the sort within a world is
a rank count over its `group_size` boxes (no global sort, no sort scratch).

The new topology carries no bounds: the caller refits after it (`wp.Bvh.refit`, which also
recomputes the packed leaves), then repacks its children (`packed.PackedNodes.refresh`).
"""

from __future__ import annotations

from typing import final

import warp as wp

from raywarp.core.pipeline.platform.device import device_caps, host_copy

wp.set_module_options({"enable_backward": False})


@wp.func_native(
  """
  const wp::BVH bvh = wp::bvh_get(bvh_id);
  if (bvh.num_leaf_nodes != bvh.num_items || bvh.max_nodes < 2 * bvh.num_items - 1) return 0;
  return 1;
  """
)
def _rebuildable(
  # In:
  bvh_id: wp.uint64,
) -> int:
  """1 when the tree has one leaf per item and room for an LBVH's 2 n - 1 nodes (Warp's refit
  then walks every leaf of the new topology), else 0."""
  ...


@wp.func_native(
  """
  const wp::BVH bvh = wp::bvh_get(bvh_id);
  bvh.primitive_indices[slot] = item;
  range_left.data[slot] = slot;
  range_right.data[slot] = slot;
  if (slot + 1 < bvh.num_items) children.data[slot] = 0;
  """
)
def _leaf(
  # In:
  bvh_id: wp.uint64,
  slot: int,
  item: int,
  # Out:
  range_left: wp.array1d[int],
  range_right: wp.array1d[int],
  children: wp.array1d[int],
):
  """Leaf `slot` holds item `item`; its key range is itself; inner node n + slot (if any) has no
  child yet."""
  ...


@wp.func_native(
  """
  const wp::BVH bvh = wp::bvh_get(bvh_id);
  const int n = bvh.num_items;
  volatile int* lefts = range_left.data;
  volatile int* rights = range_right.data;
  volatile int* parents = bvh.node_parents;
  volatile wp::BVHPackedNodeHalf* lowers = bvh.node_lowers;
  volatile wp::BVHPackedNodeHalf* uppers = bvh.node_uppers;
  const int* prims = bvh.primitive_indices;
  // Key of slot k: world << 32 | Morton code; delta(k): leading equal bits of keys k and k + 1.
  #define MJPBR_KEY(k) ((((wp::uint64)((k) / group_size)) << 32) | (wp::uint64)(unsigned int)codes.data[k])
  #if defined(__CUDA_ARCH__)
  #define MJPBR_CLZ(x) __clzll(x)
  #else
  #define MJPBR_CLZ(x) __builtin_clzll(x)
  #endif
  #define MJPBR_DELTA(k) ((MJPBR_KEY(k) ^ MJPBR_KEY((k) + 1)) == 0 ? 64 : MJPBR_CLZ(MJPBR_KEY(k) ^ MJPBR_KEY((k) + 1)))
  int index = leaf;
  for (;;) {
    const int left = lefts[index];
    const int right = rights[index];
    if (left == 0 && right == n - 1) {
      *bvh.root = index;
      parents[index] = -1;
      break;
    }
    // Warp's parent choice: stay in the node's world while one neighbour does, else the side of
    // the longer common key prefix (ties: the primitives' parity).
    bool parent_right = false;
    if (left == 0) {
      parent_right = true;
    } else {
      bool decided = false;
      const int group = left / group_size;
      if (group == right / group_size) {
        const bool same_right = right < n - 1 && (right + 1) / group_size == group;
        const bool same_left = (left - 1) / group_size == group;
        if (same_right != same_left) {
          parent_right = same_right;
          decided = true;
        }
      }
      if (!decided) {
        if (right != n - 1 && MJPBR_DELTA(right) >= MJPBR_DELTA(left - 1)) {
          if (MJPBR_DELTA(right) == MJPBR_DELTA(left - 1))
            parent_right = ((prims[left - 1] % 2) ^ (prims[right] % 2)) != 0;
          else
            parent_right = true;
        }
      }
    }
    int parent;
    if (parent_right) {
      parent = right + n;
      parents[index] = parent;
      lowers[parent].i = (unsigned int)index;
      lefts[parent] = left;
    } else {
      parent = left + n - 1;
      parents[index] = parent;
      uppers[parent].i = (unsigned int)index;
      rights[parent] = right;
    }
    #if defined(__CUDA_ARCH__)
    __threadfence();
    #endif
    // The second child to arrive completes the parent and goes on up.
    if (wp::atomic_add(children.data + (parent - n), 1) == 1) {
      lowers[parent].b = 0;
      index = parent;
    } else {
      break;
    }
  }
  #undef MJPBR_DELTA
  #undef MJPBR_CLZ
  #undef MJPBR_KEY
  """
)
def _climb(
  # In:
  bvh_id: wp.uint64,
  leaf: int,
  group_size: int,
  codes: wp.array1d[wp.int32],
  # Out:
  range_left: wp.array1d[int],
  range_right: wp.array1d[int],
  children: wp.array1d[int],
):
  """Links leaf `leaf`'s ancestors bottom-up (Warp's `build_hierarchy`, topology only)."""
  ...


@wp.func_native(
  """
  const wp::BVH bvh = wp::bvh_get(bvh_id);
  int depth = 1;
  for (int p = bvh.node_parents[node]; p != -1; p = bvh.node_parents[p]) depth++;
  const int left = range_left.data[node];
  const int right = range_right.data[node] + 1;
  if (left / group_size == (right - 1) / group_size && (right - left <= bvh.leaf_size || depth >= BVH_QUERY_STACK_SIZE)) {
    bvh.node_lowers[node].b = 1;
    bvh.node_lowers[node].i = (unsigned int)left;
    bvh.node_uppers[node].i = (unsigned int)right;
  }
  """
)
def _pack_leaf(
  # In:
  bvh_id: wp.uint64,
  node: int,
  group_size: int,
  range_left: wp.array1d[int],
  range_right: wp.array1d[int],
):
  """Node `node` becomes a leaf over its slots when they are one world's and at most the leaf
  size, or when it lies `BVH_QUERY_STACK_SIZE` levels deep (Warp's `mark_packed_leaf_nodes`)."""
  ...


@wp.func
def _part1by2(
  # In:
  v: int,
) -> int:
  """The 10 low bits of `v`, two zero bits after each."""
  x = v & 0x3FF
  x = (x | (x << 16)) & 0x030000FF
  x = (x | (x << 8)) & 0x0300F00F
  x = (x | (x << 4)) & 0x030C30C3
  x = (x | (x << 2)) & 0x09249249
  return x


@wp.func
def _cell(
  # In:
  x: float,
) -> int:
  """Morton cell (0..1023) of a coordinate normalised to [0, 1] (0 for NaN: inf x 0 of a
  degenerate world box)."""
  if not (x >= 0.0):
    return 0
  return int(wp.min(x * 1024.0, 1023.0))


@wp.func
def _finite(
  # In:
  v: wp.vec3,
) -> bool:
  """All components finite (a diverged pose's box is not)."""
  return wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])


@wp.kernel
def world_bounds(
  # In:
  due: wp.array1d[int],
  group_size: int,
  lower: wp.array1d[wp.vec3],
  upper: wp.array1d[wp.vec3],
  # Out:
  low_out: wp.array1d[wp.vec3],
  inv_out: wp.array1d[wp.vec3],
):
  """World w's box of its primitives' finite centres: its lower corner and inverse extent."""
  w = wp.tid()
  if due[0] == 0:
    return
  lo = wp.vec3(1.0e30)
  hi = wp.vec3(-1.0e30)
  for k in range(group_size):
    c = 0.5 * (lower[w * group_size + k] + upper[w * group_size + k])
    if _finite(c):
      lo = wp.min(lo, c)
      hi = wp.max(hi, c)
  if lo[0] > hi[0]:
    lo = wp.vec3(0.0)
    hi = wp.vec3(0.0)
  extent = hi - lo
  low_out[w] = lo
  inv_out[w] = wp.vec3(
    wp.where(extent[0] > 0.0, 1.0 / extent[0], 0.0),
    wp.where(extent[1] > 0.0, 1.0 / extent[1], 0.0),
    wp.where(extent[2] > 0.0, 1.0 / extent[2], 0.0),
  )


@wp.kernel
def morton_codes(
  # In:
  due: wp.array1d[int],
  group_size: int,
  lower: wp.array1d[wp.vec3],
  upper: wp.array1d[wp.vec3],
  low: wp.array1d[wp.vec3],
  inv: wp.array1d[wp.vec3],
  # Out:
  code_out: wp.array1d[wp.int32],
):
  """30-bit Morton code of primitive i's centre in its world's centre box (0: not finite)."""
  i = wp.tid()
  if due[0] == 0:
    return
  w = i // group_size
  c = 0.5 * (lower[i] + upper[i])
  code_out[i] = wp.int32(0)
  if _finite(c):
    u = wp.cw_mul(c - low[w], inv[w])
    code = (_part1by2(_cell(u[2])) << 2) | (_part1by2(_cell(u[1])) << 1) | _part1by2(_cell(u[0]))
    code_out[i] = wp.int32(code)  # pyrefly: ignore[unnecessary-type-conversion]  # Warp int -> int32 store


@wp.kernel
def sort_leaves(
  # In:
  due: wp.array1d[int],
  bvh_id: wp.uint64,
  group_size: int,
  code: wp.array1d[wp.int32],
  # Out:
  sorted_out: wp.array1d[wp.int32],
  range_left_out: wp.array1d[int],
  range_right_out: wp.array1d[int],
  children_out: wp.array1d[int],
):
  """Primitive i's leaf slot: its rank by (code, index) in its world (a stable sort)."""
  i = wp.tid()
  if due[0] == 0:
    return
  first = (i // group_size) * group_size
  mine = code[i]
  rank = int(0)
  for k in range(group_size):
    other = code[first + k]
    if other < mine or (other == mine and first + k < i):
      rank += 1
  slot = first + rank
  sorted_out[slot] = mine
  _leaf(bvh_id, slot, i, range_left_out, range_right_out, children_out)


@wp.kernel
def hierarchy(
  # In:
  due: wp.array1d[int],
  bvh_id: wp.uint64,
  group_size: int,
  code: wp.array1d[wp.int32],
  # Out:
  range_left_out: wp.array1d[int],
  range_right_out: wp.array1d[int],
  children_out: wp.array1d[int],
):
  """Leaf i's ancestors (`_climb`)."""
  i = wp.tid()
  if due[0] == 0:
    return
  _climb(bvh_id, i, group_size, code, range_left_out, range_right_out, children_out)


@wp.kernel
def pack_leaves(
  # In:
  due: wp.array1d[int],
  bvh_id: wp.uint64,
  group_size: int,
  range_left: wp.array1d[int],
  range_right: wp.array1d[int],
):
  """Node k's packed-leaf marking (`_pack_leaf`)."""
  k = wp.tid()
  if due[0] == 0:
    return
  _pack_leaf(bvh_id, k, group_size, range_left, range_right)


@wp.kernel
def group_roots(
  # In:
  due: wp.array1d[int],
  bvh_id: wp.uint64,
  # Out:
  group_root_out: wp.array1d[int],
):
  """World w's root node in the new topology."""
  w = wp.tid()
  if due[0] == 0:
    return
  group_root_out[w] = wp.bvh_get_group_root(bvh_id, w)


@wp.kernel
def check(
  # In:
  bvh_id: wp.uint64,
  # Out:
  ok_out: wp.array1d[int],
):
  """`_rebuildable` of the tree."""
  ok_out[0] = _rebuildable(bvh_id)


@final
class Rebuild:
  """The scratch of the in-place rebuilds of one grouped `wp.Bvh` (`nworld` groups of
  `group_size` primitives) and their gated launches.

  Raises (constructor):
    RuntimeError: for a tree whose node buffers do not fit an LBVH of its items (a tree built with
      leaves of several primitives).
  """

  def __init__(self, bvh: wp.Bvh, nworld: int, group_size: int) -> None:
    """Allocates the scratch (once: `rebuild` allocates nothing)."""
    device = bvh.device
    ok = wp.zeros(1, dtype=int, device=device)
    wp.launch(check, dim=1, inputs=[bvh.id], outputs=[ok], device=device)
    if not int(host_copy(ok)[0]):
      raise RuntimeError(
        "the world BVH cannot be rebuilt in place: its build packed several primitives into a leaf (leaf_size > 1, "
        "or leaves packed at the traversal stack's depth), and Warp's refit walks one leaf per primitive"
      )
    self.bvh = bvh
    self.nworld = nworld
    self.group_size = group_size
    n = nworld * group_size
    self._n = n
    self._low = wp.zeros(nworld, dtype=wp.vec3, device=device)
    self._inv = wp.zeros(nworld, dtype=wp.vec3, device=device)
    self._code = wp.zeros(n, dtype=wp.int32, device=device)
    self._sorted = wp.zeros(n, dtype=wp.int32, device=device)
    self._left = wp.zeros(2 * n - 1, dtype=int, device=device)
    self._right = wp.zeros(2 * n - 1, dtype=int, device=device)
    self._children = wp.zeros(max(n - 1, 1), dtype=int, device=device)

  def rebuild(
    self,
    due: wp.array[int, [int]],
    lower: wp.array[wp.vec3, [int]],
    upper: wp.array[wp.vec3, [int]],
    roots: wp.array[int, [int]],
  ) -> None:
    """A new topology from the boxes `lower` / `upper` (the tree's items) and the worlds' `roots`,
    on the frames `due` opens (no work otherwise; nothing allocated). The caller refits next.

    A captured frame records the launches as a conditional body (`wp.capture_if`): replays skip
    them on the frames without a rebuild. Eagerly each kernel tests the gate (`capture_if` would
    read it back to the host, a synchronisation)."""
    if device_caps(self.bvh.device).capturing():
      wp.capture_if(due, on_true=lambda: self._launch(due, lower, upper, roots))
    else:
      self._launch(due, lower, upper, roots)

  def _launch(
    self,
    due: wp.array[int, [int]],
    lower: wp.array[wp.vec3, [int]],
    upper: wp.array[wp.vec3, [int]],
    roots: wp.array[int, [int]],
  ) -> None:
    """The rebuild's gated launches (`rebuild`)."""
    device, n, g, bvh_id = self.bvh.device, self._n, self.group_size, self.bvh.id
    wp.launch(world_bounds, dim=self.nworld, inputs=[due, g, lower, upper], outputs=[self._low, self._inv], device=device)
    wp.launch(morton_codes, dim=n, inputs=[due, g, lower, upper, self._low, self._inv], outputs=[self._code], device=device)
    scratch = [self._left, self._right, self._children]
    wp.launch(sort_leaves, dim=n, inputs=[due, bvh_id, g, self._code], outputs=[self._sorted, *scratch], device=device)
    wp.launch(hierarchy, dim=n, inputs=[due, bvh_id, g, self._sorted], outputs=scratch, device=device)
    wp.launch(pack_leaves, dim=2 * n - 1, inputs=[due, bvh_id, g, self._left, self._right], device=device)
    wp.launch(group_roots, dim=self.nworld, inputs=[due, bvh_id], outputs=[roots], device=device)
