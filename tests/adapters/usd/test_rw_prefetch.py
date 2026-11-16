"""Prefetch must not hang on replaced source types or hide worker programming failures."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from raywarp.scene.images import prefetch
from raywarp.scene.model.rayw import Dependency


@pytest.mark.parametrize("threads", [0, -1])
def test_invalid_thread_count_is_rejected_even_without_sources(threads: int) -> None:
  with pytest.raises(ValueError, match="at least 1"):
    prefetch.Prefetch([], threads=threads)


def _wait_for_result(reads: prefetch.Prefetch) -> dict[str, Dependency]:
  deadline = time.monotonic() + 2.0
  while time.monotonic() < deadline:
    if result := reads.done():
      return result
    time.sleep(0.001)
  raise AssertionError("prefetch did not return a result or propagate its failure")


def test_worker_programming_error_is_propagated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path = tmp_path / "source"
  path.write_bytes(b"input")

  def fail_read(fd: int, count: int) -> bytes:
    raise RuntimeError("broken reader")

  monkeypatch.setattr(os, "read", fail_read)
  reads = prefetch.Prefetch([Dependency(path=str(path), mtime_ns=0, size=5)])
  try:
    with pytest.raises(RuntimeError, match="broken reader"):
      _wait_for_result(reads)
  finally:
    reads.close()


def test_missing_file_is_a_recorded_dependency(tmp_path: Path) -> None:
  path = str(tmp_path / "missing")
  reads = prefetch.Prefetch([Dependency(path=path, mtime_ns=0, size=5)])
  try:
    assert _wait_for_result(reads) == {path: Dependency(path=path, mtime_ns=-1, size=-1)}
  finally:
    reads.close()


def test_replaced_file_type_cannot_block_open(tmp_path: Path) -> None:
  path = tmp_path / "source"
  os.mkfifo(path)
  subprocess.run(
    [
      sys.executable,
      "-c",
      "import sys, threading; from raywarp.scene.images.prefetch import _read; "
      "state = _read(sys.argv[1], threading.Event()); assert state is not None and state.size == -1",
      str(path),
    ],
    check=True,
    timeout=15,
    capture_output=True,
    text=True,
  )
