"""`trace.accel.lbvh.Rebuild`: the in-place grouped LBVH of the world BVH is a valid tree (every
finite item found by Warp's own query after the refit, each world under its own root), on degenerate
inputs too (one item, one world, coincident centres, NaN boxes)."""

import numpy as np
import pytest
import warp as wp

from raywarp.core.trace.accel import lbvh


@wp.kernel
def _found(
  # In:
  bvh_id: wp.uint64,
  lower: wp.array1d[wp.vec3],
  upper: wp.array1d[wp.vec3],
  # Out:
  found_out: wp.array1d[int],
):
  """1 where item i's own box query returns item i."""
  i = wp.tid()
  query = wp.bvh_query_aabb(bvh_id, lower[i], upper[i])
  k = int(0)
  while wp.bvh_query_next(query, k):
    if k == i:
      found_out[i] = 1


@wp.kernel
def _below_root(
  # In:
  bvh_id: wp.uint64,
  roots: wp.array1d[int],
  group_size: int,
  lower: wp.array1d[wp.vec3],
  upper: wp.array1d[wp.vec3],
  # Out:
  found_out: wp.array1d[int],
):
  """1 where item i's box query from its world's root returns item i (and no other world's)."""
  i = wp.tid()
  query = wp.bvh_query_aabb(bvh_id, lower[i], upper[i], roots[i // group_size])
  k = int(0)
  while wp.bvh_query_next(query, k):
    if k == i:
      found_out[i] = 1
    if k // group_size != i // group_size:
      found_out[i] = -1


def _rebuilt(centres: np.ndarray, moved: np.ndarray, nworld: int) -> tuple[np.ndarray, np.ndarray]:
  """(found by a whole-tree query, found under the world's root) per item after a rebuild of the
  SAH tree of `centres` from the boxes around `moved` (half size 0.1)."""
  n = len(centres)
  group = n // nworld
  lower = wp.array(centres - 0.1, dtype=wp.vec3, device="cpu")
  upper = wp.array(centres + 0.1, dtype=wp.vec3, device="cpu")
  groups = wp.array(np.repeat(np.arange(nworld), group).astype(np.int32), dtype=int, device="cpu")
  bvh = wp.Bvh(lower, upper, groups=groups, constructor="sah")
  rebuild = lbvh.Rebuild(bvh, nworld, group)
  lower.assign(moved - 0.1)
  upper.assign(moved + 0.1)
  roots = wp.zeros(nworld, dtype=int, device="cpu")
  rebuild.rebuild(wp.array([1], dtype=int, device="cpu"), lower, upper, roots)
  bvh.refit()
  found = wp.zeros(n, dtype=int, device="cpu")
  wp.launch(_found, dim=n, inputs=[bvh.id, lower, upper], outputs=[found], device="cpu")
  below = wp.zeros(n, dtype=int, device="cpu")
  wp.launch(_below_root, dim=n, inputs=[bvh.id, roots, group, lower, upper], outputs=[below], device="cpu")
  return found.numpy(), below.numpy()


@pytest.mark.parametrize(
  ("nworld", "group", "spread"),
  [(1, 1, 1.0), (3, 1, 1.0), (1, 2, 1.0), (2, 13, 1.0), (4, 64, 1.0), (2, 500, 1.0), (1, 3000, 1e-3), (2, 2000, 0.0)],
)
def test_rebuilt_tree_finds_every_item_under_its_world(nworld: int, group: int, spread: float) -> None:
  """Worlds 100 m apart, items scattered (or all at one point: `spread` 0, the deepest ladders,
  packed below the stack depth): every item is found, from the whole tree and from its world's
  root only."""
  rng = np.random.default_rng(group)
  n = nworld * group
  offsets = np.repeat(np.arange(nworld)[:, None] * 100.0, group, 0)
  centres = (rng.normal(size=(n, 3)) + offsets).astype(np.float32)
  moved = (rng.normal(size=(n, 3)) * 3.0 * spread + offsets).astype(np.float32)
  found, below = _rebuilt(centres, moved, nworld)
  assert found.all() and (below == 1).all()


def test_rebuild_skips_non_finite_items() -> None:
  """A NaN box (a diverged pose) gets Morton code 0 and stays out of the world's centre bounds:
  the other items are all found."""
  rng = np.random.default_rng(0)
  centres = rng.normal(size=(80, 3)).astype(np.float32)
  moved = (rng.normal(size=(80, 3)) * 3.0).astype(np.float32)
  moved[5] = np.nan
  found, below = _rebuilt(centres, moved, 2)
  finite = np.isfinite(moved).all(1)
  assert found[finite].all() and (below[finite] == 1).all()


def test_closed_gate_leaves_the_tree() -> None:
  """With the gate closed every launch returns at once: the topology is the SAH build's."""
  rng = np.random.default_rng(1)
  centres = rng.normal(size=(32, 3)).astype(np.float32)
  lower = wp.array(centres - 0.1, dtype=wp.vec3, device="cpu")
  upper = wp.array(centres + 0.1, dtype=wp.vec3, device="cpu")
  groups = wp.zeros(32, dtype=int, device="cpu")
  bvh = wp.Bvh(lower, upper, groups=groups, constructor="sah")
  rebuild = lbvh.Rebuild(bvh, 1, 32)
  roots = wp.full(1, -7, dtype=int, device="cpu")
  rebuild.rebuild(wp.zeros(1, dtype=int, device="cpu"), lower, upper, roots)
  assert int(roots.numpy()[0]) == -7  # nothing written
  assert not rebuild._code.numpy().any()
