"""Property tests of the scene boundary checks and the byte budget (skipped without hypothesis)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from raywarp.scene.images import blob
from raywarp.scene.model import checks


@given(value=st.floats(allow_nan=True, allow_infinity=True), low=st.floats(-10.0, 10.0), span=st.floats(0.0, 10.0))
def test_in_range_accepts_exactly_finite_values_in_range(value, low, span):
  """`checks.in_range` accepts a value iff it is finite and in [low, low + span]."""
  high = low + span
  if math.isfinite(value) and low <= value <= high:
    assert checks.in_range("v", value, "test", low=low, high=high) == value
  else:
    with pytest.raises(ValueError, match="must be in"):
      checks.in_range("v", value, "test", low=low, high=high)


@given(total=st.integers(1, 1 << 40), requests=st.lists(st.integers(-5, 1 << 41), max_size=20))
def test_byte_budget_never_exceeds_its_total(total, requests):
  """Sequential holds clamp to the total; try_acquire never takes more than is free."""
  budget = blob.ByteBudget(total)
  for n in requests:
    with budget.hold(n):
      assert budget.peak <= total
    assert budget.try_acquire(n) == (0 <= n <= total)
    if 0 <= n <= total:
      budget.release(n)
  assert budget.peak <= total
