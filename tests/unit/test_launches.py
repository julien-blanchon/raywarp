"""`CachedLaunch` replays its recorded launch only while every argument is the same: a swapped
array, a struct edited in place or a changed scalar launches anew (on every available device).
`BandedLaunch` covers its launch once in bands, each band recorded and replayed."""

import os
from typing import Final

import numpy as np
import pytest
import warp as wp

from raywarp.core.pipeline.platform.launches import BandedLaunch, CachedLaunch

_DEVICES: Final = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


@wp.struct
class _Offset:
  value: float


@wp.kernel(module="unique", enable_backward=False)
def _affine(
  # In:
  x: wp.array[float],
  scale: float,
  offset: _Offset,
  # Out:
  y_out: wp.array[float],
):
  """y = scale x + offset."""
  i = wp.tid()
  y_out[i] = scale * x[i] + offset.value


@pytest.mark.parametrize("device", _DEVICES)
def test_cached_launch_follows_every_argument(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """Same arguments: the recorded launch is replayed (no new `wp.launch`); each change below
  records a new one, whose output shows the new argument."""
  with wp.ScopedDevice(device):
    x, y = wp.array([1.0, 2.0], dtype=float), wp.zeros(2, dtype=float)
    offset = _Offset()
    offset.value = 0.5
    calls = []
    launch = wp.launch

    def counted(*args: object, **kwargs: object) -> object:
      calls.append(1)
      return launch(*args, **kwargs)  # pyrefly: ignore[no-matching-overload]  # forwards any call

    monkeypatch.setattr(wp, "launch", counted)
    site = CachedLaunch()

    def run(scale: float, x: wp.array, offset: _Offset, y: wp.array) -> np.ndarray:
      site.launch(_affine, (2,), inputs=[x, scale, offset], outputs=[y], device=device)
      return y.numpy()

    np.testing.assert_allclose(run(2.0, x, offset, y), [2.5, 4.5])
    x.assign([3.0, 4.0])  # contents change in place: the same array, replayed
    np.testing.assert_allclose(run(2.0, x, offset, y), [6.5, 8.5])
    assert len(calls) == 1
    swapped = wp.array([10.0, 20.0], dtype=float)  # a swapped array
    np.testing.assert_allclose(run(2.0, swapped, offset, y), [20.5, 40.5])
    assert len(calls) == 2
    offset.value = -1.0  # the struct edited in place
    np.testing.assert_allclose(run(2.0, swapped, offset, y), [19.0, 39.0])
    assert len(calls) == 3
    np.testing.assert_allclose(run(3.0, swapped, offset, y), [29.0, 59.0])  # a changed scalar
    assert len(calls) == 4
    other = wp.zeros(2, dtype=float)  # a swapped output
    np.testing.assert_allclose(run(3.0, swapped, offset, other), [29.0, 59.0])
    assert len(calls) == 5
    np.testing.assert_allclose(run(3.0, swapped, offset, other), [29.0, 59.0])
    assert len(calls) == 5  # unchanged again: replayed


@wp.kernel(module="unique", enable_backward=False)
def _mark(
  # In:
  first: int,
  # Out:
  hits_out: wp.array2d[wp.int32],
):
  """Counts a visit of (i, first + j)."""
  i, j = wp.tid()
  hits_out[i, first + j] += 1


def test_banded_launch_covers_its_launch_once(monkeypatch: pytest.MonkeyPatch) -> None:
  """Every (row, index) of the banded axis is launched once per call, in several bands; a second
  call replays the recorded bands (no new `wp.launch`), a new shape records anew."""
  calls = []
  launch = wp.launch

  def counted(*args: object, **kwargs: object) -> object:
    calls.append(1)
    return launch(*args, **kwargs)  # pyrefly: ignore[no-matching-overload]  # forwards any call

  monkeypatch.setattr(wp, "launch", counted)
  site = BandedLaunch()
  hits = wp.zeros((2, 1000), dtype=wp.int32, device="cpu")
  site.launch(_mark, (2, 1000), inputs=[], outputs=[hits], device="cpu", min_band=1, per_core=4)
  np.testing.assert_array_equal(hits.numpy(), 1)
  nband = len(calls)
  assert nband > 1 or os.cpu_count() == 1
  site.launch(_mark, (2, 1000), inputs=[], outputs=[hits], device="cpu", min_band=1, per_core=4)
  np.testing.assert_array_equal(hits.numpy(), 2)
  assert len(calls) == nband
  small = wp.zeros((2, 10), dtype=wp.int32, device="cpu")
  site.launch(_mark, (2, 10), inputs=[], outputs=[small], device="cpu", min_band=1, per_core=1)
  np.testing.assert_array_equal(small.numpy(), 1)
