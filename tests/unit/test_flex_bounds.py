"""`trace.optix.dynamic.flex_bounds` (a block reduction per world) widens each world's cell box to
exactly the min / max of its deformable points, keeping the box already there."""

import numpy as np
import pytest
import warp as wp

from raywarp.core.trace.optix import dynamic
from raywarp.core.trace.optix import instances as ops


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("per_world", [1, 37, 1000])
def test_block_bounds_are_the_points_min_max(device: str, per_world: int) -> None:
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("no CUDA device")
  nworld = 3
  points = np.random.default_rng(per_world).normal(size=(nworld * per_world, 3)).astype(np.float32)
  bounds = np.zeros((nworld, ops.BOUNDS), np.float32)
  bounds[:, :3], bounds[:, 3:6], bounds[:, 6:] = 0.5, -0.5, 7.0  # a box already there; the reference
  out = wp.array(bounds.ravel(), dtype=float, device=device)
  wp.launch_tiled(
    dynamic.flex_bounds,
    dim=(nworld,),
    inputs=[wp.array(points, dtype=wp.vec3, device=device), per_world],
    outputs=[out],
    block_dim=dynamic.FLEX_BLOCK,
    device=device,
  )
  got = out.numpy().reshape(nworld, ops.BOUNDS)
  w = points.reshape(nworld, per_world, 3)
  np.testing.assert_array_equal(got[:, :3], np.minimum(w.min(axis=1), 0.5))
  np.testing.assert_array_equal(got[:, 3:6], np.maximum(w.max(axis=1), -0.5))
  np.testing.assert_array_equal(got[:, 6:], 7.0)
