"""Bound encoded reads and retained decode inputs; subprocess deadlines expose deadlocks."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from raywarp.scene.images import blob
from raywarp.scene.images import decode as imaging
from raywarp.scene.model.rayw import Dependency
from raywarp.scene.model.types import PixelType, Texels


def encoded_reads() -> None:
  """A full read takes its encoded-byte reservation before allocating input bytes."""
  active = 0
  peak = 0
  lock = threading.Lock()
  info = imaging.ImageInfo(width=1, height=1, channels=4, pixel=PixelType.U8, error="probe only")

  def read(source: imaging.ImageSource) -> tuple[bytes, Dependency | None]:
    nonlocal active, peak
    with lock:
      active += 128
      peak = max(peak, active)
    time.sleep(0.005)  # allow the other pending reads to contend
    return bytes(128), None

  def probe(source: imaging.ImageSource, data: bytes | None = None) -> imaging.ImageInfo:
    nonlocal active
    with lock:
      active -= 128
    return info

  def encoded_size(source: imaging.ImageSource) -> int:
    return 128

  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(blob, "AHEAD_BYTES", 512)
    patch.setattr(blob, "_READ_BYTES", 128, raising=False)
    patch.setattr(blob, "probe_bytes", encoded_size, raising=False)
    patch.setattr(blob, "read_file", read)
    patch.setattr(blob, "probe", probe)
    ahead = blob.Ahead()
    try:
      for i in range(8):
        ahead.submit(imaging.ImageSource(path=f"image-{i}.png"))
      assert len(ahead.results()) == 8
    finally:
      ahead.close()
  assert peak <= 128
  assert active == 0


def retained_inputs(*, oversized: bool = False, fail: bool = False) -> None:
  """Unprocessed retained images share the decode workspace limit; failure releases it."""
  info = imaging.ImageInfo(width=2, height=2, channels=4, pixel=PixelType.U8, decode_work=64)
  sources = [imaging.ImageSource(path=str(i)) for i in range(3)]
  probed = [blob.Probed(info=info, base=np.full((2, 2, 4), i, np.uint8)) for i in range(3)]
  memory = blob.ByteBudget(64 if oversized else 128)
  decoded: list[str] = []
  written: list[int] = []

  def decode(source: imaging.ImageSource, info: imaging.ImageInfo) -> tuple[Texels, str]:
    decoded.append(source.path)
    return np.full((2, 2, 4), int(source.path), np.uint8), ""

  def sink(index: int, levels: list[Texels]) -> None:
    retained = sum(p.base.nbytes for p in probed if p.base is not None)
    required = min(blob.decode_bytes(info, 1), memory.total)
    assert memory.held >= required + retained
    assert memory.held <= memory.total
    np.testing.assert_array_equal(levels[0], index)
    if fail:
      raise ValueError("sink failed")
    written.append(index)

  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(blob, "decode_base", decode)
    if fail:
      with pytest.raises(ValueError, match="sink failed"):
        blob.decode_all(sources, probed, [1] * 3, sink=sink, threads=1, budget=memory)
    else:
      assert blob.decode_all(sources, probed, [1] * 3, sink=sink, threads=3, budget=memory) == [""] * 3
      assert sorted(written) == [0, 1, 2]
      assert decoded  # at least one ahead base did not fit alongside the largest workspace
  assert memory.held == 0
  assert all(p.base is None for p in probed)


# The deadline of one scenario once the worker has imported raywarp: a scenario takes <= 0.26 s
# (3000 in-process runs of each, idle 16-core host); the deadline only has to tell a hang from
# that. Imports are not timed by it: they took 2 s idle and over 15 s once under a loaded suite.
_SCENARIO_DEADLINE_S = 15.0
_IMPORT_DEADLINE_S = 300.0
_READY = "ready"


@pytest.mark.parametrize("scenario", ["reads", "retained", "oversized", "failure"])
def test_bounded_image_workspace_finishes(scenario: str) -> None:
  """A regression must fail within the deadline, even if a worker waits on memory forever (a
  subprocess, killed then); the deadline starts once the worker has imported raywarp."""
  cmd = [sys.executable, str(Path(__file__).resolve()), scenario]
  with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    assert proc.stdin is not None and proc.stdout is not None
    try:
      ready = threading.Thread(target=proc.stdout.readline, daemon=True)
      ready.start()
      ready.join(_IMPORT_DEADLINE_S)
      assert not ready.is_alive(), "the worker did not finish its imports"
      _, err = proc.communicate(input="", timeout=_SCENARIO_DEADLINE_S)  # stdin's EOF starts the scenario
    finally:
      proc.kill()
    assert proc.returncode == 0, err


if __name__ == "__main__":
  print(_READY, flush=True)
  sys.stdin.read()  # the parent's start signal (EOF)
  match sys.argv[1]:
    case "reads":
      encoded_reads()
    case "retained":
      retained_inputs()
    case "oversized":
      retained_inputs(oversized=True)
    case "failure":
      retained_inputs(fail=True)
    case _:
      raise ValueError("unknown image-budget scenario")
