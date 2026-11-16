"""`device.launch_bands`: CPU launches split in bands cover the range once, on every OS."""

import os
import sys
import threading
import time

import numpy as np
import pytest
import warp as wp
from conftest import no_default_device
from warp._src import context

from raywarp.core.pipeline.platform import device

CPU = wp.get_device("cpu")


@wp.kernel
def _count(
  # In:
  first: int,
  # Out:
  hits: wp.array1d[wp.int32],
):
  """Counts a visit of index first + tid."""
  i = first + wp.tid()
  wp.atomic_add(hits, i, 1)


def _visits(n: int, min_band: int, per_core: int = 1) -> np.ndarray:
  """Visit count per index of a banded launch over [0, n) on the CPU."""
  hits = wp.zeros(n, dtype=wp.int32, device="cpu")

  def run(a: int, b: int) -> None:
    """Indices [a, b)."""
    wp.launch(_count, dim=b - a, inputs=[a], outputs=[hits], device="cpu")

  device.launch_bands(wp.get_module(__name__), device=CPU, n=n, launch=run, min_band=min_band, per_core=per_core)
  return hits.numpy()


@pytest.mark.parametrize(("n", "min_band", "per_core"), [(1, 1, 1), (1000, 1, 1), (1000, 64, 1), (37, 100, 1), (1000, 1, 4)])
def test_bands_cover_the_range_once(n: int, min_band: int, per_core: int):
  """Every index is launched exactly once, whatever the band count."""
  np.testing.assert_array_equal(_visits(n, min_band, per_core), np.ones(n, np.int32))


def test_bands_leave_the_default_device_alone(monkeypatch: pytest.MonkeyPatch):
  """The band threads set no default device: Warp's is process-wide, and one `wp.ScopedDevice`
  per band raced (the last restore could leave a CUDA process on the CPU device: the L4 suite
  then rendered every later OptiX test of that worker in software)."""

  def scoped(*_args: object) -> None:
    raise AssertionError("a band changed the process's default device")

  monkeypatch.setattr(wp, "ScopedDevice", scoped)
  monkeypatch.setattr(wp, "set_device", scoped)
  with no_default_device():  # any other write of the default device replaces the stand-in
    stand_in = context.runtime.default_device  # pyrefly: ignore[missing-attribute]
    np.testing.assert_array_equal(_visits(1000, 1), np.ones(1000, np.int32))
    assert context.runtime.default_device is stand_in  # pyrefly: ignore[missing-attribute]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs a CUDA default device beside the CPU")
def test_cpu_bands_keep_a_cuda_default_device():
  """CPU bands under a CUDA default device leave it in place (the race above, end to end)."""
  with wp.ScopedDevice("cuda:0"):
    for _ in range(20):
      _visits(4096, 1)
      assert wp.get_device().alias == "cuda:0"


def test_concurrent_callers_share_the_process_cores():
  """Worker threads baking at once share one band per core of the process (each call had its own
  pool of a thread per core: N workers ran N x cores threads)."""
  cores = device.process_cores()
  lock = threading.Lock()
  running, peak = [0], [0]

  def run(a: int, b: int) -> None:
    """A band that holds its thread for a while and records how many run at once."""
    with lock:
      running[0] += 1
      peak[0] = max(peak[0], running[0])
    time.sleep(0.05)
    with lock:
      running[0] -= 1

  callers = [
    threading.Thread(
      target=device.launch_bands,
      args=(wp.get_module(__name__),),
      kwargs={"device": CPU, "n": 64 * cores, "launch": run, "min_band": 1},
    )
    for _ in range(3)
  ]
  for t in callers:
    t.start()
  for t in callers:
    t.join()
  assert peak[0] <= cores


@pytest.mark.skipif(sys.platform != "linux" or len(os.sched_getaffinity(0)) < 2, reason="needs Linux affinity, 2+ cores")
def test_pool_follows_the_process_mask_not_the_first_caller(monkeypatch: pytest.MonkeyPatch):
  """A worker thread pinned to one core creates the pool: the bands still get the process's core
  count and mask (the pool was sized and pinned by the creating thread's mask)."""
  monkeypatch.setattr(device, "_POOL", device._BandPool())  # a fresh pool
  process = os.sched_getaffinity(0)
  masks: list[set[int]] = []

  def run(a: int, b: int) -> None:
    """Records the band thread's mask."""
    masks.append(os.sched_getaffinity(0))
    time.sleep(0.02)

  def narrow() -> None:
    """A worker pinned to one core that creates the pool (its first banded launch would)."""
    os.sched_setaffinity(0, {min(process)})
    device._POOL.get().submit(int).result()  # executors start their threads on submit

  worker = threading.Thread(target=narrow)
  worker.start()
  worker.join()
  masks.clear()
  device.launch_bands(wp.get_module(__name__), device=CPU, n=64, launch=run, min_band=1)
  assert len(masks) == min(len(process), 64)
  assert all(m == process for m in masks)


def test_a_band_calling_launch_bands_runs_serially():
  """A call from inside a band runs as one launch (no wait on the threads it occupies)."""
  inner = []

  def run(a: int, b: int) -> None:
    """Each band calls launch_bands again."""
    device.launch_bands(wp.get_module(__name__), device=CPU, n=100, launch=lambda c, d: inner.append((c, d)), min_band=1)

  device.launch_bands(wp.get_module(__name__), device=CPU, n=2, launch=run, min_band=1)
  assert inner == [(0, 100)] * min(2, device.process_cores())


def test_without_affinity_mask(monkeypatch: pytest.MonkeyPatch):
  """An OS without `os.sched_getaffinity` (macOS) uses the CPU count (the light grid's CPU trace
  called it unconditionally: AttributeError on macOS)."""
  monkeypatch.setattr(sys, "platform", "darwin")
  monkeypatch.delattr(os, "sched_getaffinity", raising=False)
  assert (_visits(500, 8) == 1).all()  # (np.testing imports sysconfig, which the patched platform breaks)


def test_concurrent_first_callers_share_one_pool(monkeypatch: pytest.MonkeyPatch):
  """Threads that need the band pool at once get the same one (an unlocked `functools.cache` let
  each build its own: 4 pools whose threads idled for the process's life)."""
  monkeypatch.setattr(device, "_POOL", device._BandPool())
  real = device._process_mask
  monkeypatch.setattr(device, "_process_mask", lambda: (time.sleep(0.05), real())[1])  # a slow creation
  pools: list[object] = []
  threads = [threading.Thread(target=lambda: pools.append(device._POOL.get())) for _ in range(4)]
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  assert len({id(p) for p in pools}) == 1


@pytest.mark.parametrize("min_band", [0, -4])
def test_min_band_below_one_is_rejected(min_band: int):
  """`min_band` < 1 raises (they silently became 1)."""
  with pytest.raises(ValueError, match="min_band"):
    device.launch_bands(wp.get_module(__name__), device=CPU, n=10, launch=lambda a, b: None, min_band=min_band)


@pytest.mark.skipif(device.process_cores() < 2, reason="needs 2+ cores (two bands)")
def test_a_raising_band_waits_for_the_others():
  """When a band raises, launch_bands re-raises only after every other band ended (`Executor.map`
  raised at once, cancelled the pending bands and left running ones writing the caller's arrays)."""
  done: list[int] = []

  def run(a: int, b: int) -> None:
    """Band 0 fails at once; band 1 ends later."""
    if a == 0:
      raise RuntimeError("band 0")
    time.sleep(0.2)
    done.append(a)

  with pytest.raises(RuntimeError, match="band 0") as failed:
    device.launch_bands(wp.get_module(__name__), device=CPU, n=2, launch=run, min_band=1)
  assert done == [1]
  assert failed.value.__notes__ == ["launch_bands: 1 of 2 bands failed (this is the first)"]
