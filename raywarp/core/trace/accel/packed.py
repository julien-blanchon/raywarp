"""Packed BVH2 children of the mesh BLASes and the scene BVHs: what the queries read instead of
Warp's two node arrays (and, for meshes, its vertex / index arrays).

For every inner node, the boxes and references of both children sit in 64 contiguous bytes
(`nodes.py`: `mesh_pack_node`), so one node expansion reads one cache line instead of four
sectors of two arrays; a mesh's triangles follow, gathered in leaf-slot order (48 bytes each), so
a leaf reads its triangles contiguously instead of through the primitive and vertex indices. The
layout holds the same boxes and vertices as Warp's: the tree, its visit order and the hits are
unchanged. `packed_nodes(ids)` gives the (n,) addresses for a table of mesh ids (0: no mesh),
built once per table (static scene meshes, which are never refit); every tree refit at runtime
owns its `PackedNodes` and calls `refresh` after the refit (height fields, deformables, the scene
BVHs).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Final, cast, final

import numpy as np
import warp as wp

from raywarp.core.kernels import native
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.trace.accel import nodes

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Bytes of one node half (`wp::BVHPackedNodeHalf`); a packed node is four.
_HALF: Final = 16


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _mesh_max_nodes(
  # In:
  ids: wp.array1d[wp.uint64],
  # Out:
  count_out: wp.array1d[int],
):
  """Node capacity of mesh i's BVH."""
  i = wp.tid()
  count_out[i] = native.mesh_max_nodes(ids[i])


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _mesh_num_items(
  # In:
  ids: wp.array1d[wp.uint64],
  # Out:
  count_out: wp.array1d[int],
):
  """Triangle count of mesh i's BVH."""
  i = wp.tid()
  count_out[i] = native.mesh_num_items(ids[i])


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _mesh_depth(
  # In:
  ids: wp.array1d[wp.uint64],
  # Out:
  depth_out: wp.array1d[int],
):
  """Recorded depth of mesh i's BVH (0: unrecorded: cuBQL's fast path, which keeps the tree within
  the stack; Warp's LBVH caps the depth of single-group ranges only, and no grouped LBVH tree is
  packed here: the scene and deformable trees are SAH, which records it)."""
  i = wp.tid()
  depth_out[i] = native.mesh_max_depth(ids[i])


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _bvh_depth(
  # In:
  ids: wp.array1d[wp.uint64],
  # Out:
  depth_out: wp.array1d[int],
):
  """Recorded depth of BVH i (0: unrecorded)."""
  i = wp.tid()
  depth_out[i] = native.bvh_max_depth(ids[i])


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _mesh_pack(
  # In:
  ids: wp.array1d[wp.uint64],
  addresses: wp.array1d[wp.uint64],
):
  """Packed children of node `index` of mesh i (nothing past its node count or for a leaf) and
  its triangle of leaf slot `index` (nothing past its triangle count)."""
  i, index = wp.tid()
  native.mesh_pack_node(ids[i], addresses[i], index)


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _bvh_max_nodes(
  # In:
  ids: wp.array1d[wp.uint64],
  # Out:
  count_out: wp.array1d[int],
):
  """Node capacity of BVH i."""
  i = wp.tid()
  count_out[i] = native.bvh_max_nodes(ids[i])


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _bvh_pack(
  # In:
  ids: wp.array1d[wp.uint64],
  addresses: wp.array1d[wp.uint64],
):
  """Packed children of node `index` of BVH i (nothing past its node count or for a leaf)."""
  i, index = wp.tid()
  native.bvh_pack_node(ids[i], addresses[i], index)


@final
class PackedNodes:
  """The packed children of a table of meshes' BVHs (`meshes`) or of Warp BVHs.

  Attributes:
    ids: (n,) the mesh (or BVH) ids (0: none).
    addresses: (n,) device address of each tree's packed halves (0: none), what the queries read;
      `host_addresses` the same values on the host.
    halves: (total, 4) the packed halves `addresses` point into (kept alive with them).
  """

  def __init__(self, ids: wp.array[wp.uint64, [int]], *, meshes: bool) -> None:
    """Allocates every tree's packed nodes (four halves per node) and packs them.

    Raises:
      RuntimeError: for a tree deeper than the traversal stacks hold (`nodes.STACK` levels): its
        queries would skip subtrees. Warp's builders stop splitting at that depth.
    """
    self.ids = ids
    depths = _counts(_mesh_depth if meshes else _bvh_depth, ids)
    deepest = int(depths.max()) if len(depths) else 0
    if deepest > nodes.STACK:
      raise RuntimeError(
        f"a BVH of depth {deepest} exceeds the {nodes.STACK}-entry traversal stacks (its queries would skip "
        "subtrees); Warp's builders cap the depth: use a validated warp-lang"
      )
    self._pack = _mesh_pack if meshes else _bvh_pack
    n = ids.shape[0]
    device = ids.device
    counts = _counts(_mesh_max_nodes if meshes else _bvh_max_nodes, ids)
    # A mesh's gathered triangles follow its packed children: three 16-byte vertices each.
    items = _counts(_mesh_num_items, ids) if meshes else np.zeros_like(counts)
    offsets = np.concatenate([[0], np.cumsum(4 * counts + 3 * items)])
    self.halves = wp.zeros((max(int(offsets[-1]), 1), 4), dtype=wp.float32, device=device)
    ptr = self.halves.ptr
    if ptr is None:
      raise RuntimeError("packed BVH children need an allocated device array")
    base = int(ptr)
    addresses = np.where(counts > 0, base + _HALF * offsets[:-1], 0).astype(np.uint64)
    self.addresses: wp.array[wp.uint64, [int]] = wp.array(addresses, dtype=wp.uint64, device=device)
    self.host_addresses = addresses  # the same, on the host (struct fields of one tree)
    # Threads per tree: one per node (its children) and per leaf slot (its triangle).
    self._nodes = int(max(counts.max(), items.max())) if n else 0
    self.refresh()

  def refresh(self) -> None:
    """Repacks every tree from its current BVH and vertices (after a refit)."""
    if self._nodes:
      wp.launch(self._pack, dim=(self.ids.shape[0], self._nodes), inputs=[self.ids, self.addresses], device=self.ids.device)


def _counts(
  kernel: Callable[[wp.array[wp.uint64, [int]], wp.array[int, [int]]], None], ids: wp.array[wp.uint64, [int]]
) -> np.ndarray[[int], np.dtype[np.int64]]:
  """(n,) per-tree counts of `kernel` (node capacity, triangle count or depth) on the host."""
  n = ids.shape[0]
  if not n:
    return np.zeros(0, np.int64)
  out = wp.zeros(n, dtype=int, device=ids.device)
  wp.launch(kernel, dim=n, inputs=[ids], outputs=[out], device=ids.device)
  return np.asarray(host_copy(out), np.int64)


# The packed nodes of the static mesh-id tables (scene BLASes, never refit), per table: their
# addresses and the buffer they point into. Not the `PackedNodes`, which holds its table: a value
# referencing its weak key keeps the entry (and its buffer) for the life of the process.
# Final[...]: an empty table, its key and value types come from the annotation.
_PACKED: Final[
  weakref.WeakKeyDictionary[wp.array[wp.uint64, [int]], tuple[wp.array[wp.uint64, [int]], wp.array[wp.float32, [int, 4]]]]
] = weakref.WeakKeyDictionary()


def packed_nodes(ids: wp.array[wp.uint64, [int]]) -> wp.array[wp.uint64, [int]]:
  """(n,) packed-node addresses of the static meshes `ids` (built on first use, kept as long as
  the table). A table whose BLASes are refit must own a `PackedNodes` and refresh it instead."""
  found = _PACKED.get(ids)
  if found is None:
    packed = PackedNodes(ids, meshes=True)
    found = _PACKED[ids] = (packed.addresses, packed.halves)
  return found[0]


# The packing kernels (`warmup.first_frame`: the software scene BVH a probe / grid bake builds).
KERNELS: Final = cast(
  "tuple[wp.Kernel, ...]",
  (_bvh_depth, _bvh_max_nodes, _bvh_pack, _mesh_depth, _mesh_num_items, _mesh_max_nodes, _mesh_pack),
)
