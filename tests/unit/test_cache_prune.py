"""`raywarp.cache.prune`: the caches bounded by deleting their least recently used entries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from raywarp.cache import prune


def _file(path: Path, size: int, used: float) -> Path:
  """A cache file of `size` bytes last used at `used`."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"x" * size)
  os.utime(path, (used, used))
  return path


def test_the_least_recently_used_entries_go_first(tmp_path: Path) -> None:
  """Oldest first until the caches fit; a PTX and its entry go together, a glTF conversion as a
  whole; OptiX's cache, the kernel store, locks and unknown files stay."""
  old_bake = _file(tmp_path / "bakes" / "gi" / "a.npz", 100, 1000.0)
  ptx = _file(tmp_path / "ptx" / "k.ptx", 100, 2000.0)
  ptx_entry = _file(tmp_path / "ptx" / "k.json", 10, 2000.0)
  gltf = _file(tmp_path / "gltf" / "duck-1" / "duck.usda", 100, 3000.0)
  scene = _file(tmp_path / "office-1.wpbr", 100, 4000.0)
  kept = [
    _file(tmp_path / "optix" / "optix7cache.db", 1000, 1.0),
    _file(tmp_path / "kernels" / "sm120" / "d0" / "files.json", 1000, 1.0),
    _file(tmp_path / "ptx" / "k.lock", 0, 1.0),
    _file(tmp_path / "hf" / "state.json", 1000, 1.0),
  ]
  done = prune.prune(250, tmp_path)
  assert (done.removed, done.freed, done.kept) == (2, 210, 200)
  assert not old_bake.exists() and not ptx.exists() and not ptx_entry.exists()
  assert gltf.exists() and scene.exists() and all(p.exists() for p in kept)
  assert prune.prune(0, tmp_path).removed == 2
  assert not gltf.parent.exists() and not scene.exists()


def test_caching_off_or_a_negative_size_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("RAYWARP_CACHE_DIR", "")
  with pytest.raises(ValueError, match="caching is off"):
    prune.prune(0)
  with pytest.raises(ValueError, match=">= 0"):
    prune.prune(-1, Path("."))
