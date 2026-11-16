"""Depth-32 SAH trees preserve every candidate and masked analytic hit after refitting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.features import PLANES_NONE
from raywarp.core.kernels.abi.rays import SoftwareGeometry
from raywarp.core.trace.accel import cast, nodes, ordered, query
from raywarp.core.trace.accel.packed import PackedNodes


@wp.func_native("return wp::bvh_get(tree).max_depth;")
def _depth(tree: wp.uint64) -> int: ...


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _tree_info(
  # In:
  tree: wp.uint64,
  # Out:
  info: wp.array1d[int],
) -> None:
  info[0] = wp.bvh_get_group_root(tree, 0)
  info[1] = _depth(tree)


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _candidates(
  # In:
  tree: wp.uint64,
  halves: wp.uint64,
  root: int,
  # Out:
  counts: wp.array2d[int],
) -> None:
  origin = wp.vec3(-2.0, 0.0, 0.0)
  rcp = nodes.reciprocal(wp.vec3(1.0, 0.0, 0.0))
  near = ordered.NearStack()
  cursor = ordered.Cursor()
  ordered.start_ordered(cursor, tree, halves, root, origin, rcp, 10.0)
  primitive = ordered.next_ordered(cursor, near, origin, rcp, 10.0)
  while primitive >= 0:
    counts[0, primitive] += 1
    primitive = ordered.next_ordered(cursor, near, origin, rcp, 10.0)
  walk = cast.Walk()
  stack = cast.WalkStack()
  cast.start_unordered(walk, stack, tree, halves, root, origin, rcp, 10.0)
  primitive = cast.next_unordered(walk, stack, origin, rcp, 10.0)
  while primitive >= 0:
    counts[1, primitive] += 1
    primitive = cast.next_unordered(walk, stack, origin, rcp, 10.0)


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _stack_threads(
  # Out:
  errors: wp.array2d[int],
) -> None:
  """Each thread retains all its own pending siblings across divergent reads."""
  row, column = wp.tid()
  identity = row * errors.shape[1] + column
  stack = cast.WalkStack()
  for depth in range(nodes.STACK):
    nodes.unordered_store(stack.node, depth, identity * nodes.STACK + depth)
  failures = int(0)
  for step in range(nodes.STACK):
    depth = (step + identity) % nodes.STACK
    if nodes.unordered_int_at(stack.node, depth) != identity * nodes.STACK + depth:
      failures += 1
  errors[row, column] = failures


@pytest.mark.parametrize("device,block_dim", [("cpu", 64), ("cuda:0", 64), ("cuda:0", 128), ("cuda:0", 256), ("cuda:0", 512)])
def test_unordered_stack_storage_is_private_to_each_thread(device: str, block_dim: int) -> None:
  """CUDA shared stack and large-block local fallback preserve independent thread state."""
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  errors = wp.empty((3, 173), dtype=int, device=device)
  wp.launch(_stack_threads, dim=errors.shape, outputs=[errors], device=device, block_dim=block_dim)
  np.testing.assert_array_equal(errors.numpy(), np.zeros(errors.shape, dtype=np.int32))


type TraceCases = Callable[[SoftwareGeometry, wp.array1d[int], wp.array1d[float], wp.array1d[int]], None]


def _trace_cases(*, nearest_first: bool) -> TraceCases:
  trees = (False, True, False, False, PLANES_NONE)
  closest, any_hit = query.make_trace(
    types=(int(ShapeType.BOX),), cull_backfaces=False, ordered=nearest_first, trees=trees, two_sided=False
  )

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees, ordered=nearest_first))
  def run(
    # In:
    geo: SoftwareGeometry,
    # Out:
    ids: wp.array1d[int],
    distances: wp.array1d[float],
    occluded: wp.array1d[int],
  ) -> None:
    world = wp.tid()
    origin = wp.vec3(-2.0, 0.0, 0.0)
    direction = wp.vec3(1.0, 0.0, 0.0)
    hit = closest(geo, world, origin, direction, 10.0)
    ids[world] = hit.instance_id
    distances[world] = hit.dist
    occluded[world] = int(any_hit(geo, world, origin, direction, 10.0))

  return run


def _boxes(n: int, *, device: str) -> SoftwareGeometry:
  # World i keeps only instance i opaque: every tree leaf must remain reachable by real queries.
  colors = np.zeros((n, n, 4), dtype=np.float32)
  colors[np.arange(n), np.arange(n), 3] = 1.0
  geo = SoftwareGeometry()
  geo.shared_ids = wp.array(np.arange(n).astype(np.int32), dtype=int, device=device)
  geo.shared_pos = wp.zeros(n, dtype=wp.vec3, device=device)
  geo.shared_rot = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33, device=device)
  geo.instance_type = wp.full(n, int(ShapeType.BOX), dtype=int, device=device)
  geo.instance_size = wp.full((1, n), wp.vec3(0.5), dtype=wp.vec3, device=device)
  geo.instance_material = wp.full((1, n), -1, dtype=int, device=device)
  geo.instance_color = wp.array(colors, dtype=wp.vec4, device=device)
  geo.material_base_color = wp.zeros((1, 0), dtype=wp.vec4, device=device)
  return geo


def test_empty_bvh_tracer_needs_no_private_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(wp, "__version__", "1.20.0")
  trees = (False, False, False, False, PLANES_NONE)
  options = query.module_options(trees, ordered=False)
  assert options == {}
  closest, any_hit = query.make_trace(types=(), cull_backfaces=False, ordered=False, trees=trees, two_sided=False)

  @wp.kernel(module="unique", enable_backward=False, module_options=options)
  def empty(
    # In:
    geo: SoftwareGeometry,
    # Out:
    results: wp.array1d[int],
  ) -> None:
    origin = wp.vec3(0.0)
    direction = wp.vec3(1.0, 0.0, 0.0)
    results[0] = closest(geo, 0, origin, direction, 10.0).instance_id
    results[1] = int(any_hit(geo, 0, origin, direction, 10.0))

  results = wp.zeros(2, dtype=int, device="cpu")
  wp.launch(empty, dim=1, inputs=[SoftwareGeometry()], outputs=[results], device="cpu")
  np.testing.assert_array_equal(results.numpy(), [-1, 0])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("mirrored", [False, True])
def test_depth32_sah_keeps_candidates_and_masked_box_hits(device: str, mirrored: bool) -> None:
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  nodes.require_compatible()
  # Highly nonuniform but finite boxes reach the actual SAH builder's depth limit. Mirroring
  # swaps the deep branch: ordered traversal formerly lost leaves; public unordered queries
  # could overflow their unchecked stack. Refit then gives all boxes ordinary metre-scale size.
  x = np.exp2(np.arange(161) - 80).astype(np.float32)
  n = len(x)
  lo = np.zeros((n, 3), dtype=np.float32)
  hi = np.full((n, 3), 1.0e-20, dtype=np.float32)
  lo[:, 0] = -1.1 * x if mirrored else x
  hi[:, 0] = -x if mirrored else 1.1 * x
  lower = wp.array(lo, dtype=wp.vec3, device=device)
  upper = wp.array(hi, dtype=wp.vec3, device=device)
  groups = wp.zeros(n, dtype=int, device=device)
  tree = wp.Bvh(lower, upper, groups=groups, constructor="sah")
  info = wp.zeros(2, dtype=int, device=device)
  wp.launch(_tree_info, dim=1, inputs=[tree.id], outputs=[info], device=device)
  root, depth = info.numpy().tolist()
  # The builders' depth boundary: 32 levels in Warp 1.18; 1.19 stops one level earlier (a free
  # stack slot for queries pushing both children).
  boundary = 32 if wp.__version__.startswith("1.18.") else 31
  assert depth == boundary, "fixture must reach the builder's actual stack-depth boundary"
  lower.fill_(wp.vec3(-0.5))
  upper.fill_(wp.vec3(0.5))
  tree.refit()
  counts = wp.zeros((2, n), dtype=int, device=device)
  halves = PackedNodes(wp.array([tree.id], dtype=wp.uint64, device=device), meshes=False)
  wp.launch(
    _candidates, dim=1, inputs=[tree.id, wp.uint64(int(halves.host_addresses[0])), root], outputs=[counts], device=device
  )
  np.testing.assert_array_equal(counts.numpy(), np.ones((2, n), dtype=np.int32))
  geo = _boxes(n, device=device)
  geo.shared_bvh_id = tree.id
  geo.shared_nodes = wp.uint64(int(halves.host_addresses[0]))
  geo.shared_root = root
  ids = wp.empty(n, dtype=int, device=device)
  distances = wp.empty(n, dtype=float, device=device)
  occluded = wp.empty(n, dtype=int, device=device)
  for nearest_first in (False, True):
    wp.launch(_trace_cases(nearest_first=nearest_first), dim=n, inputs=[geo], outputs=[ids, distances, occluded], device=device)
    np.testing.assert_array_equal(ids.numpy(), np.arange(n).astype(np.int32))
    np.testing.assert_array_equal(distances.numpy(), np.full(n, 1.5, dtype=np.float32))
    np.testing.assert_array_equal(occluded.numpy(), np.ones(n, dtype=np.int32))


def test_packing_rejects_a_tree_deeper_than_the_stacks(monkeypatch: pytest.MonkeyPatch) -> None:
  """A tree deeper than the traversal stacks raises where its packed nodes are built, instead of
  queries silently skipping the subtrees a full stack drops (here: a depth-32 SAH tree against
  stacks of 16 entries)."""
  x = np.exp2(np.arange(161) - 80).astype(np.float32)
  lo = np.zeros((len(x), 3), dtype=np.float32)
  hi = np.full((len(x), 3), 1.0e-20, dtype=np.float32)
  lo[:, 0], hi[:, 0] = x, 1.1 * x
  tree = wp.Bvh(wp.array(lo, dtype=wp.vec3, device="cpu"), wp.array(hi, dtype=wp.vec3, device="cpu"), constructor="sah")
  ids = wp.array([tree.id], dtype=wp.uint64, device="cpu")
  PackedNodes(ids, meshes=False)
  monkeypatch.setattr(nodes, "STACK", 16)
  with pytest.raises(RuntimeError, match="exceeds the 16-entry traversal stacks"):
    PackedNodes(ids, meshes=False)
