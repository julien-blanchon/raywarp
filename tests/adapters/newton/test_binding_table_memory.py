"""Newton host preparation keeps independent compact rows without deep-copy amplification."""

from __future__ import annotations

import numpy as np
import pytest

from raywarp.integrations.newton.binding import BindingTables, NewtonBinding


def _tables(rows: int, count: int = 3) -> BindingTables:
  local = np.zeros((rows, count, 7), dtype=np.float32)
  local[:, :, 6] = 1
  return BindingTables(
    body=np.full((rows, count), -1, np.int32),
    local=local,
    world=np.full((rows, count), -1, np.int32),
    shape=np.full((rows, count), -1, np.int32),
  )


def _owned_bytes(array: np.ndarray[tuple[int, ...]]) -> int:
  while isinstance(array.base, np.ndarray):
    array = array.base
  return array.nbytes


def test_retained_first_row_owns_only_one_row() -> None:
  source = _tables(64, 32)
  binding = NewtonBinding(tables=source, nsource=64, device="cpu", shape_transform=None)
  actual = binding._row0  # retained allocation is the regression contract
  for result, original in zip(
    (actual.body, actual.local, actual.world, actual.shape),
    (source.body, source.local, source.world, source.shape),
    strict=True,
  ):
    assert result.shape[0] == 1 and result.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(result, original)
    assert _owned_bytes(result) == result.nbytes
    np.testing.assert_array_equal(result, original[:1])


@pytest.mark.parametrize("source_rows,target_rows", [(0, 3), (2, 3), (2, 4), (3, 2), (1, 0)])
def test_broadcast_rejects_invalid_world_mapping(source_rows: int, target_rows: int) -> None:
  with pytest.raises(ValueError, match="rows"):
    _tables(source_rows)._broadcast(target_rows)  # concat's row check


def test_concat_rejects_distinct_nonshared_world_counts() -> None:
  with pytest.raises(ValueError, match="rows"):
    BindingTables.concat((_tables(2), _tables(4)))


def test_concat_accepts_noncontiguous_inputs_and_empty_instance_parts() -> None:
  source = _tables(4)
  source.body[:] = np.arange(4)[:, None]
  strided = BindingTables(
    body=np.asfortranarray(source.body[:, ::2]),
    local=source.local[:, ::2],
    world=source.world[:, ::2],
    shape=source.shape[:, ::2],
  )
  result = BindingTables.concat((strided, _tables(1, 0)))
  for actual, original in zip(
    (result.body, result.local, result.world, result.shape),
    (strided.body, strided.local, strided.world, strided.shape),
    strict=True,
  ):
    assert actual.flags["C_CONTIGUOUS"] and not np.shares_memory(actual, original)
    np.testing.assert_array_equal(actual, original)


def test_concat_and_merged_preserve_world_order_and_independence() -> None:
  shared, worlds = _tables(1, 1), _tables(3, 2)
  shared.body[:] = 10
  worlds.body[:] = [[20, 21], [30, 31], [40, 41]]
  worlds.local[:, :, 0] = [[2, 3], [4, 5], [6, 7]]
  combined = BindingTables.concat((shared, worlds))
  merged = combined.merged()
  np.testing.assert_array_equal(combined.body, [[10, 20, 21], [10, 30, 31], [10, 40, 41]])
  np.testing.assert_array_equal(merged.body, [[10, 20, 21, 10, 30, 31, 10, 40, 41]])
  for result in (combined.body, combined.local, combined.world, combined.shape):
    assert result.flags["C_CONTIGUOUS"]
  shared.body[:] = 99
  worlds.local[:] = 99
  assert combined.body[0, 0] == 10 and combined.local[0, 1, 0] == 2
  combined.body[:] = 88
  assert merged.body[0, 0] == 10


def test_selected_dynamic_rows_do_not_alias_inputs() -> None:
  source = _tables(4, 3)
  source.body[:, 1] = 7
  source.local[:, 1, 0] = [1, 2, 3, 4]
  binding = NewtonBinding(tables=source, nsource=4, device="cpu", shape_transform=None)
  actual = binding._rows  # selected allocation must not alias source arrays
  assert actual.body.shape == (4, 1) and actual.body.flags["C_CONTIGUOUS"] and actual.local.flags["C_CONTIGUOUS"]
  source.body[:] = -1
  source.local[:] = 0
  np.testing.assert_array_equal(actual.body, [[7]] * 4)
  np.testing.assert_array_equal(actual.local[:, 0, 0], [1, 2, 3, 4])
