"""tests/kernel_snapshot.py: the test jobs' kernel snapshot per kernel-code digest (offline)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import warp as wp

from tests import kernel_snapshot


def _session(log: Path, rows: list[tuple[str, bool]]) -> None:
  """A session's records: (module directory, compiled) Warp loads."""
  log.mkdir(parents=True, exist_ok=True)
  with (log / "compiles-1.jsonl").open("w") as f:
    for name, compiled in rows:
      f.write(
        json.dumps({"kind": "warp", "module": name, "dir": name, "s": 1.5, "compiled": compiled, "test": "t.py::a"}) + "\n"
      )


def test_a_snapshot_holds_what_the_session_loaded_and_restores_exact_then_near(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """`save` keeps the loaded modules (not the others the cache holds) under this tree's digest;
  `restore` takes that snapshot (exact), else the newest one (near); a session that compiled
  nothing keeps the snapshot as it is; one that loaded nothing saves none; the newest `KEEP` are
  kept."""
  monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp"))
  cache = tmp_path / "warp" / wp.config.version
  for name in ("wp_a_1", "wp_b_2", "wp_stale_3"):
    (cache / name).mkdir(parents=True)
    (cache / name / f"{name}.o").write_bytes(b"obj")
  bucket = tmp_path / "bucket"
  _session(tmp_path / "log", [("wp_a_1", True), ("wp_b_2", False)])
  kernel_snapshot.save(bucket, tmp_path / "log")
  (target,) = bucket.glob("tests.*.tar")
  assert target.name == f"tests.{kernel_snapshot.key()}.tar"
  import shutil

  shutil.rmtree(tmp_path / "warp")
  assert kernel_snapshot.restore(bucket) == "exact"
  assert sorted(p.name for p in cache.iterdir()) == ["wp_a_1", "wp_b_2"]
  mtime = target.stat().st_mtime_ns
  _session(tmp_path / "log", [("wp_a_1", False)])
  kernel_snapshot.save(bucket, tmp_path / "log")
  assert target.stat().st_mtime_ns == mtime
  _session(tmp_path / "log2", [])
  target.unlink()
  kernel_snapshot.save(bucket, tmp_path / "log2")
  assert not list(bucket.glob("tests.*.tar"))
  for i in range(kernel_snapshot.KEEP + 2):
    (bucket / f"tests.{i:016x}.tar").write_bytes(b"")
    os.utime(bucket / f"tests.{i:016x}.tar", (i, i))
  _session(tmp_path / "log", [("wp_a_1", True)])
  kernel_snapshot.save(bucket, tmp_path / "log")
  kept = sorted(p.name for p in bucket.glob("tests.*.tar"))
  assert len(kept) == kernel_snapshot.KEEP and target.name in kept
  target.rename(bucket / "tests.00000000000000ff.tar")  # another tree's, the newest
  assert kernel_snapshot.restore(bucket) == "near (tests.00000000000000ff.tar)"
