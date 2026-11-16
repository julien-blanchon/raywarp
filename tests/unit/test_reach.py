"""`core/trace/accel/reach.py`: the per-world draw rule both traversals apply (reference position,
reach, CELL_LIMIT)."""

import numpy as np
import pytest
import warp as wp

from raywarp.core.trace.accel import reach


@wp.kernel
def _reference_and_drawn(
  # In:
  pos: wp.array2d[wp.vec3],
  rot: wp.array2d[wp.mat33],
  size: wp.array2d[wp.vec3],
  ids: wp.array1d[int],
  # Out:
  ref_out: wp.array1d[wp.vec3],
  drawn_out: wp.array1d[int],
):
  """World 0's `reach.reference` (instance k in column k), and whether each instance is drawn."""
  k = wp.tid()
  ref = reach.reference(pos, rot, size, ids, ids, 0)
  if k == 0:
    ref_out[0] = ref
  drawn_out[k] = wp.where(reach.drawn(ref, pos[0, k], rot[0, k], size[0, k]), 1, 0)


def _run(positions: list) -> tuple[np.ndarray, list[int]]:
  """(reference, drawn flags) of one world whose instances sit at `positions`."""
  p = np.asarray(positions, np.float32)[None]
  n = p.shape[1]
  ref = wp.zeros(1, dtype=wp.vec3, device="cpu")
  drawn = wp.zeros(n, dtype=int, device="cpu")
  inputs = [
    wp.array(p, dtype=wp.vec3, device="cpu"),
    wp.array(np.tile(np.eye(3, dtype=np.float32), (1, n, 1, 1)), dtype=wp.mat33, device="cpu"),
    wp.array(np.full((1, n, 3), 0.1, np.float32), dtype=wp.vec3, device="cpu"),
    wp.array(np.arange(n).astype(np.int32), dtype=int, device="cpu"),
  ]
  wp.launch(_reference_and_drawn, dim=n, inputs=inputs, outputs=[ref, drawn], device="cpu")
  return ref.numpy()[0], drawn.numpy().tolist()


@pytest.mark.parametrize(
  ("positions", "expected", "drawn"),
  [
    # Narrow: the box centre, every instance drawn.
    ([(0.0, 0.0, 0.0), (2.0, 4.0, 0.0), (10.0, 0.0, 6.0)], (5.0, 2.0, 3.0), [1, 1, 1]),
    # Wide: the lower median to 1 m; the far one hidden.
    ([(0.0, 0.0, 0.0), (5.0, 1.0, 0.0), (3.0e3, -3.0e3, 2.0)], (5.0, 0.0, 0.0), [1, 1, 0]),
    # Diverged (beyond CELL_LIMIT): out of the reference and hidden.
    ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0e20, 0.0, 0.0)], (0.5, 0.0, 0.0), [1, 1, 0]),
  ],
)
def test_reference_is_the_box_centre_or_the_lower_median(positions: list, expected: tuple, drawn: list) -> None:
  """A world at most `reach.NEAR_RADIUS` wide takes its placed instance positions' box centre as
  its reference, a wider one their per-axis lower median (to `reach.MEDIAN_RESOLUTION`); an
  instance beyond `reach.NEAR_RADIUS` of it, or beyond `reach.CELL_LIMIT`, is not drawn."""
  ref, flags = _run(positions)
  np.testing.assert_allclose(ref, expected, atol=reach.MEDIAN_RESOLUTION)
  assert flags == drawn
