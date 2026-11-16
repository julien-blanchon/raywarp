"""Instance entry points reject transform shapes that would truncate data or change counts."""

from __future__ import annotations

import numpy as np
import pytest

from raywarp.scene import SceneBuilder


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rows", [3, 4])
def test_instance_rejects_float32_overflow_before_mutation(batch: bool, rows: int) -> None:
  """Finite host values must stay finite in stored transforms, regardless of NumPy error policy."""
  matrix = np.eye(4)
  matrix[0, 3] = 1e40
  builder = SceneBuilder()
  builder.add_instance("box")
  before = builder.instance_xforms().copy()
  with np.errstate(over="raise"):
    with pytest.raises(ValueError, match="xform"):
      if batch:
        builder.add_instances("box", xforms=matrix[None, :rows])
      else:
        builder.add_instance("box", xform=matrix[:rows])
    assert np.geterr()["over"] == "raise"
  assert builder.ninstance == 1
  np.testing.assert_array_equal(builder.instance_xforms(), before)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rows", [3, 4])
def test_instance_preserves_largest_finite_float32_values(batch: bool, rows: int) -> None:
  """Validation follows storage representability instead of imposing a scene-size limit."""
  matrix = np.eye(4)
  largest = np.asarray(np.finfo(np.float32).max, np.float64).item()
  matrix[:2, 3] = [largest, -largest]
  builder = SceneBuilder()
  if batch:
    builder.add_instances("box", xforms=matrix[None, :rows])
  else:
    builder.add_instance("box", xform=matrix[:rows])
  result = builder.instance_xforms()
  assert result.shape == (1, 3, 4)
  assert np.isfinite(result).all()
  np.testing.assert_array_equal(result[0], matrix[:3].astype(np.float32))


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
  "last_row", [(0.1, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 2.0), (float("nan"), 0.0, 0.0, 1.0), (0.0, 0.0, float("inf"), 1.0)]
)
def test_instance_rejects_unrepresentable_last_row(batch: bool, last_row: tuple[float, float, float, float]) -> None:
  """The affine storage cannot silently discard projective or nonfinite fourth-row values."""
  matrix = np.eye(4)
  matrix[3] = last_row
  builder = SceneBuilder()
  builder.add_instance("box")
  before = builder.instance_xforms().copy()
  with pytest.raises(ValueError, match="xform"):
    if batch:
      builder.add_instances("box", xforms=matrix[None])
    else:
      builder.add_instance("box", xform=matrix)
  assert builder.ninstance == 1
  np.testing.assert_array_equal(builder.instance_xforms(), before)


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 5), (6, 4)])
def test_single_instance_rejects_batch_or_oversized_transform(shape: tuple[int, ...]) -> None:
  """A single-instance call cannot silently emit a batch or discard extra matrix entries."""
  builder = SceneBuilder()
  builder.add_instance("box")
  before = builder.instance_xforms().copy()
  with pytest.raises(ValueError, match=r"xform.*shape"):
    builder.add_instance("box", xform=np.ones(shape))
  assert builder.ninstance == 1
  np.testing.assert_array_equal(builder.instance_xforms(), before)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 5), (2, 6, 4), (1, 2, 3, 4), (0, 3, 5)])
def test_instance_batch_rejects_wrong_rank_or_matrix_shape(shape: tuple[int, ...]) -> None:
  """Only the documented leading instance axis is accepted, including for an empty batch."""
  builder = SceneBuilder()
  builder.add_instance("box")
  before = builder.instance_xforms().copy()
  with pytest.raises(ValueError, match=r"xforms.*shape"):
    builder.add_instances("box", xforms=np.ones(shape))
  assert builder.ninstance == 1
  np.testing.assert_array_equal(builder.instance_xforms(), before)


@pytest.mark.parametrize("rows", [3, 4])
def test_single_scaled_reflected_transform_preserves_values(rows: int) -> None:
  """Validation does not orthonormalize an authored transform or lose its translation."""
  matrix = np.diag(np.array([-2.0, 3.0, 0.5, 1.0], np.float64))
  matrix[:3, 3] = [4.0, -5.0, 6.0]
  builder = SceneBuilder()
  assert builder.add_instance("box", xform=matrix[:rows]) == 0
  assert builder.ninstance == 1
  np.testing.assert_array_equal(builder.instance_xforms()[0], matrix[:3].astype(np.float32))


@pytest.mark.parametrize("rows", [3, 4])
@pytest.mark.parametrize("count", [0, 2])
def test_valid_instance_batch_preserves_count_and_values(rows: int, count: int) -> None:
  """Empty batches and both authored transform layouts retain the requested row mapping."""
  matrix = np.diag(np.array([-2.0, 3.0, 0.5, 1.0], np.float64))
  batch = np.broadcast_to(matrix, (count, 4, 4)).copy()
  batch[:, 0, 3] = np.arange(count) + 4.0
  builder = SceneBuilder()
  assert builder.add_instances("box", xforms=batch[:, :rows]) == range(count)
  assert builder.ninstance == count
  result = builder.instance_xforms()
  assert result.shape == (count, 3, 4)
  np.testing.assert_array_equal(result, batch[:, :3].astype(np.float32))
