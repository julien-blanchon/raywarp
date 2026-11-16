"""Small host-only contracts: measurements must reject invalid images; strings index like lists."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures.conformance import Tolerance, ViewRender, measure

from raywarp.scene.model.table import StringTable


def _view(height: int = 2, width: int = 3) -> ViewRender[int, int]:
  return np.ones((height, width), np.float32), np.zeros((height, width, 3), np.float32)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("which", [0, 1])
def test_nonfinite_hit_albedo_cannot_pass(bad: float, which: int) -> None:
  views = [_view(), _view()]
  views[which][1][0, 0, 0] = bad
  with pytest.raises(ValueError, match="albedo must be finite"):
    measure(*views)


@pytest.mark.parametrize("bad", [math.nan, -math.inf, -1.0])
def test_invalid_depth_cannot_be_treated_as_a_matching_miss(bad: float) -> None:
  a, b = _view(), _view()
  a[0][0, 0] = b[0][0, 0] = bad
  with pytest.raises(ValueError, match="depth must be nonnegative"):
    measure(a, b)


def test_zero_and_infinite_misses_compare_without_albedo() -> None:
  a, b = _view(), _view()
  a[0][:] = 0.0
  b[0][:] = math.inf
  a[1][:] = b[1][:] = math.nan  # backgrounds have no albedo surface
  result = measure(a, b)
  assert result.coverage == 1.0 and result.depth == 0.0 and result.albedo == 0.0
  assert not result.failures


@pytest.mark.parametrize("height,width", [(0, 3), (2, 0), (0, 0)])
def test_empty_images_cannot_pass(height: int, width: int) -> None:
  with pytest.raises(ValueError, match="nonempty"):
    measure(_view(height, width), _view(height, width))


def test_broadcastable_but_mismatched_views_cannot_pass() -> None:
  with pytest.raises(ValueError, match="shapes must match"):
    measure(_view(2, 1), _view(2, 3))


def test_depth_and_albedo_dimensions_must_match() -> None:
  with pytest.raises(ValueError, match="albedo"):
    measure((_view(2, 3)[0], _view(3, 2)[1]), _view())


@pytest.mark.parametrize("field", ["coverage", "depth", "albedo"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_tolerance_rejected(field: str, bad: float) -> None:
  with pytest.raises(ValueError, match=field):
    Tolerance(**{field: bad})


def test_coverage_fraction_cannot_exceed_one() -> None:
  with pytest.raises(ValueError, match="coverage"):
    Tolerance(coverage=1.01)


def test_finite_differences_keep_the_expected_metrics() -> None:
  a, b = _view(), _view()
  b[0][:] = 1.25
  b[1][:] = 0.5
  result = measure(a, b)
  assert result.coverage == 1.0
  assert result.depth == 0.25
  assert result.albedo == 0.5
  assert len(result.failures) == 2


@pytest.mark.parametrize("strings", [[], [""], ["alpha", "", "bé"]])
def test_string_table_index_contract_matches_list(strings: list[str]) -> None:
  table = StringTable.from_list(strings)
  for index in range(-len(strings), len(strings)):
    assert table[index] == strings[index]
  for index in (-len(strings) - 1, -len(strings) - 2, len(strings), len(strings) + 1):
    with pytest.raises(IndexError):
      table[index]
