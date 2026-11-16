"""Host-only cache failure and atomic-replacement regressions; no rendering or USD required."""

from __future__ import annotations

import dataclasses
import json
import os
import struct
import time
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pytest

from raywarp.cache import code_version, paths
from raywarp.scene.model import wpbr
from raywarp.scene.model.types import ALIGN, Json


def _file(path: Path, arrays: dict[str, Json], *, meta: dict[str, Json] | None = None) -> None:
  """Write a controlled cache header over a 256-byte array region."""
  data = json.dumps({"version": wpbr.VERSION, "arrays": arrays, "meta": meta}).encode()
  path.write_bytes(
    wpbr.MAGIC + bytes(2 * ALIGN - len(wpbr.MAGIC)) + data + struct.pack("<QQ8s", 2 * ALIGN, len(data), wpbr.END)
  )


def test_load_keeps_one_generation_during_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path, replacement = tmp_path / "scene.wpbr", tmp_path / "replacement.wpbr"
  wpbr.save(path, {"a": np.array([11, 12], np.int32)}, lambda: {"generation": "old"})
  wpbr.save(replacement, {"a": np.array([21, 22], np.int32)}, lambda: {"generation": "new"})
  memmap = np.memmap

  def replace_before_mapping(
    file: str | os.PathLike[str] | BinaryIO, *, dtype: type[np.uint8], mode: Literal["r"]
  ) -> np.memmap:
    replacement.replace(path)
    return memmap(file, dtype=dtype, mode=mode)

  with monkeypatch.context() as patch:
    patch.setattr(np, "memmap", replace_before_mapping)
    arrays, header = wpbr.load(path)
  assert header["generation"] == "old"
  np.testing.assert_array_equal(arrays["a"], [11, 12])
  with pytest.raises(ValueError, match="read-only"):
    arrays["a"][0] = 0
  assert isinstance(arrays["a"], np.memmap)
  current, header = wpbr.load(path)
  assert header["generation"] == "new"
  np.testing.assert_array_equal(current["a"], [21, 22])


@pytest.mark.parametrize(
  "entry",
  [
    ["u1", [-1], ALIGN],
    ["u1", [True], ALIGN],
    ["u1", [1], -ALIGN],
    ["u1", [1], ALIGN + 1],
    ["u1", [1], 2 * ALIGN],  # header bytes, not array storage
    ["u1", [ALIGN + 1], ALIGN],
    ["u1", [1 << 65], ALIGN],
    ["O", [1], ALIGN],
    ["V0", [1], ALIGN],
    ["not-a-dtype", [1], ALIGN],
  ],
)
def test_malformed_array_rejected(tmp_path: Path, entry: Json) -> None:
  path = tmp_path / "scene.wpbr"
  _file(path, {"a": entry})
  with pytest.raises(ValueError):
    wpbr.load(path)


def test_empty_and_scalar_arrays_round_trip(tmp_path: Path) -> None:
  path = tmp_path / "scene.wpbr"
  wpbr.save(path, {"scalar": np.array(3.5), "empty": np.empty((0, 3), np.float32)}, dict)
  arrays, _ = wpbr.load(path)
  assert arrays["scalar"].shape == () and arrays["scalar"].item() == 3.5
  assert arrays["empty"].shape == (0, 3)


@pytest.mark.parametrize("tail", [b"", b"broken", struct.pack("<QQ8s", 1 << 63, 1 << 63, wpbr.END)])
def test_truncated_or_out_of_bounds_header_rejected(tmp_path: Path, tail: bytes) -> None:
  path = tmp_path / "scene.wpbr"
  path.write_bytes(wpbr.MAGIC + (bytes(ALIGN) if len(tail) > 8 else b"") + tail)
  with pytest.raises(ValueError):
    wpbr.read_header(path)
  assert not wpbr.fresh(path, wpbr.CacheKey(adapter="test", source="source", options={}, code="test"))
  assert wpbr.stale_dependencies(path, tmp_path) == []


@pytest.mark.parametrize("dependencies", [None, {}, "missing", [None], [["source", True, 1]]])
def test_malformed_dependency_record_is_not_fresh(tmp_path: Path, dependencies: Json) -> None:
  path = tmp_path / "scene.wpbr"
  key = wpbr.CacheKey(adapter="test", source="source", options={}, code="test")
  _file(path, {}, meta={"cache": {"key": key.to_json(), "dependencies": dependencies}})
  assert not wpbr.fresh(path, key)


def test_freshness_includes_missing_and_changed_sources(tmp_path: Path) -> None:
  path, source = tmp_path / "scene.wpbr", tmp_path / "source"
  key = wpbr.CacheKey(adapter="test", source=str(source), options={}, code="test")
  meta: dict[str, Json] = {}
  wpbr.record(meta, key, [str(source)])
  _file(path, {}, meta=meta)
  assert wpbr.fresh(path, key)
  source.write_text("new")
  assert not wpbr.fresh(path, key)
  wpbr.record(meta, key, [str(source)])
  _file(path, {}, meta=meta)
  assert wpbr.fresh(path, key)
  source.write_text("changed")
  assert not wpbr.fresh(path, key)


def test_a_source_edited_during_the_read_is_recorded_unknown(tmp_path: Path) -> None:
  """A dependency modified after the build started reading may have been read before its edit:
  it is recorded unknown, so the next load rebuilds (an untouched one stays fresh)."""
  path, edited, kept = tmp_path / "scene.wpbr", tmp_path / "edited", tmp_path / "kept"
  kept.write_text("kept")
  started = time.time_ns()
  edited.write_text("edited during the read")
  later = started + 1_000_000_000  # stamped after the start whatever the file system's tick
  os.utime(edited, ns=(later, later))
  key = wpbr.CacheKey(adapter="test", source=str(edited), options={}, code="test")
  meta: dict[str, Json] = {}
  wpbr.record(meta, key, [str(edited), str(kept)], since_ns=started)
  _file(path, {}, meta=meta)
  assert not wpbr.fresh(path, key)
  wpbr.record(meta, key, [str(kept)], since_ns=started)
  _file(path, {}, meta=meta)
  assert wpbr.fresh(path, key)


def test_load_fresh_reads_the_header_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  path, source = tmp_path / "scene.wpbr", tmp_path / "source"
  key = wpbr.CacheKey(adapter="test", source=str(source), options={}, code="test")
  meta: dict[str, Json] = {}
  wpbr.record(meta, key, [str(source)])
  wpbr.save(path, {"a": np.arange(5).astype(np.int32)}, lambda: {"meta": meta})
  reads = []
  original = wpbr._any_header
  monkeypatch.setattr(wpbr, "_any_header", lambda file: reads.append(1) or original(file))
  found = wpbr.load_fresh(path, key)
  assert found is not None and len(reads) == 1
  np.testing.assert_array_equal(found[0]["a"], np.arange(5))
  assert wpbr.load_fresh(path, dataclasses.replace(key, code="other")) is None
  source.write_text("new")
  assert wpbr.load_fresh(path, key) is None
  assert wpbr.load_fresh(tmp_path / "missing.wpbr", key) is None


def test_constructor_failure_closes_and_removes_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  opened: list[int] = []

  def fail_truncate(fd: int, length: int) -> None:
    opened.append(fd)
    raise OSError("no space")

  monkeypatch.setattr(os, "ftruncate", fail_truncate)
  with pytest.raises(OSError, match="no space"):
    wpbr.save(tmp_path / "scene.wpbr", {"a": np.zeros(1, np.uint8)}, dict)
  assert not list(tmp_path.iterdir())
  with pytest.raises(OSError, match="Bad file descriptor"):
    os.fstat(opened[0])


def test_zero_byte_write_fails_without_leaking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  calls = 0

  def no_progress(fd: int, data: object, offset: int) -> int:
    nonlocal calls
    calls += 1
    # Makes a pre-fix regression fail promptly rather than hanging its pwrite loop.
    if calls > 1:
      raise RuntimeError("retried a write that made no progress")
    return 0

  monkeypatch.setattr(os, "pwrite", no_progress)
  with pytest.raises(OSError, match="no progress"):
    wpbr.save(tmp_path / "scene.wpbr", {"a": np.zeros(1, np.uint8)}, dict)
  assert not list(tmp_path.iterdir())


def test_code_digest_ignores_comments_docstrings_and_formatting() -> None:
  base = b'"""Module doc."""\n\ndef f(x):\n  """Doc."""\n  return x + 1  # one\n'
  same = b"def f(x):\n    # a comment\n    return (x +\n            1)\n"
  other = b"def f(x):\n  return x + 2\n"
  digest = code_version.code_digest
  assert digest([("p/a.py", base)]) == digest([("p/a.py", same)])
  assert digest([("p/a.py", base)]) != digest([("p/a.py", other)])
  # 1, 1.0 and True are different code; names and file contents both count.
  assert len({digest([("p/a.py", f"x = {v}\n".encode())]) for v in ("1", "1.0", "True")}) == 3
  assert digest([("p/a.py", base)]) != digest([("p/b.py", base)])
  # Other files (MDL modules) count byte for byte; the order of the files does not matter.
  files = [("p/a.py", base), ("p/m.mdl", b"mdl 1.7;")]
  assert digest(files) == digest(files[::-1]) != digest([("p/a.py", base), ("p/m.mdl", b"mdl 1.7; ")])


def test_code_digest_memo_is_used_and_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(paths.CACHE_ENV, str(tmp_path))
  files = [("p/a.py", b"x = 1\n")]
  first = code_version.code_digest(files)
  assert list(tmp_path.glob("code-digests-*.json"))
  monkeypatch.setattr(code_version, "_python_digest", lambda source: pytest.fail("parsed again"))
  assert code_version.code_digest(files) == first


def test_code_version_reads_python_and_mdl_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  package = tmp_path / "pkg_w17"
  (package / "lib").mkdir(parents=True)
  (package / "__init__.py").write_text("X = 1\n")
  (package / "lib" / "m.mdl").write_text("mdl 1.7;")
  (package / "notes.txt").write_text("ignored")
  monkeypatch.syspath_prepend(str(tmp_path))
  assert [name for name, _ in code_version.code_files("pkg_w17")] == ["pkg_w17/__init__.py", "pkg_w17/lib/m.mdl"]
  with pytest.raises(ModuleNotFoundError):
    code_version.code_files("pkg_w17_missing")


def test_installed_versions_reads_dist_info() -> None:
  found = wpbr.installed_versions("numpy", "pytest", "no-such-distribution-w17")
  assert found["numpy"] == np.__version__ and found["pytest"] == pytest.__version__
  assert found["no-such-distribution-w17"] == "none"


def _key(source: Path, *, options: int = 1, code: str = "c", library: str = "1") -> wpbr.CacheKey:
  return wpbr.CacheKey(adapter="test", source=str(source), options={"o": options}, code=code, libraries={"lib": library})


def test_moved_scene_stays_fresh_and_other_libraries_do_not(tmp_path: Path) -> None:
  """Dependencies are recorded relative to the source's directory: the tree read elsewhere (another
  machine's mount) is fresh; a library version, option or code change is not."""
  tree = tmp_path / "a" / "scene"
  (tree / "tex").mkdir(parents=True)
  source = tree / "scene.usda"
  source.write_text("root")
  (tree / "tex" / "t.png").write_bytes(b"png")
  path = tmp_path / "scene.wpbr"
  meta: dict[str, Json] = {}
  wpbr.record(meta, _key(source), [str(source), str(tree / "tex" / "t.png"), str(tmp_path / "a" / "absent.png")])
  _file(path, {}, meta=meta)
  assert wpbr.fresh(path, _key(source))
  moved = tmp_path / "b" / "scene"
  moved.parent.mkdir()
  (tmp_path / "a" / "scene").rename(moved)  # rename keeps the mtimes
  assert wpbr.fresh(path, _key(moved / "scene.usda"))
  assert not wpbr.fresh(path, _key(moved / "other.usda"))
  assert not wpbr.fresh(path, _key(moved / "scene.usda", library="2"))
  assert not wpbr.fresh(path, _key(moved / "scene.usda", options=2))
  assert not wpbr.fresh(path, _key(moved / "scene.usda", code="d"))
  (tmp_path / "a" / "absent.png").write_bytes(b"new")  # missing at build time (outside: absolute)
  assert not wpbr.fresh(path, _key(moved / "scene.usda"))


def test_list_files_matches_stat_files(tmp_path: Path) -> None:
  (tmp_path / "d").mkdir()
  (tmp_path / "d" / "a").write_bytes(b"1")
  (tmp_path / "b").write_bytes(b"22")
  (tmp_path / "link").symlink_to(tmp_path / "b")
  paths = [str(tmp_path / n) for n in ("d/a", "b", "link", "d", "missing", "gone/x")]
  assert wpbr.list_files(paths) == [next(d for d in wpbr.stat_files(paths) if d.path == p) for p in paths]


def test_dependencies_outside_the_source_tree_stay_absolute(tmp_path: Path) -> None:
  """A file outside the source's directory is recorded by its absolute path: the tree mounted
  elsewhere still checks the same file (a relative `../..` path would point somewhere else)."""
  tree = tmp_path / "a" / "scene"
  tree.mkdir(parents=True)
  source = tree / "scene.usda"
  source.write_text("root")
  outside = tmp_path / "shared" / "lib.mdl"
  meta: dict[str, Json] = {}
  wpbr.record(meta, _key(source), [str(source), str(outside)])
  entry = meta["cache"]
  assert isinstance(entry, dict)
  assert entry["dependencies"] == [
    ["scene.usda", *wpbr.stat_files([str(source)])[0].to_json(tree)[1:]],
    [str(outside), -1, -1],
  ]
  deeper = tmp_path / "b" / "c" / "scene"
  deeper.parent.mkdir(parents=True)
  tree.rename(deeper)
  path = tmp_path / "scene.wpbr"
  _file(path, {}, meta=meta)
  assert wpbr.fresh(path, _key(deeper / "scene.usda"))
  outside.parent.mkdir()
  outside.write_text("now present")
  assert not wpbr.fresh(path, _key(deeper / "scene.usda"))


def test_a_known_state_taken_after_the_start_is_recorded_unknown(tmp_path: Path) -> None:
  """A prefetch's state (`known`) of a file edited after the build started: the stage may have
  read the file before the edit while the prefetch read (and stat-ed) it after, so the edited
  state must not be recorded as what the scene was built from (a stale cache read as fresh)."""
  path, layer = tmp_path / "scene.wpbr", tmp_path / "layer.usdc"
  layer.write_text("edited after the stage read it")
  started = time.time_ns()
  later = started + 1_000_000_000
  os.utime(layer, ns=(later, later))
  st = layer.stat()
  known = {str(layer): wpbr.Dependency(path=str(layer), mtime_ns=st.st_mtime_ns, size=st.st_size)}
  key = wpbr.CacheKey(adapter="test", source=str(layer), options={}, code="test")
  meta: dict[str, Json] = {}
  wpbr.record(meta, key, [str(layer)], known=known, since_ns=started)
  _file(path, {}, meta=meta)
  assert not wpbr.fresh(path, key)


def test_mdl_cache_write_leaves_no_temporary_file(tmp_path: Path) -> None:
  """An MDL cache write that fails with something else than an OSError (a bad array for
  `np.save`) leaves no temporary file behind; the cache files get the user's umask, as the other
  caches' (not mkstemp's 0600: a cache directory shared by users)."""
  from raywarp.cache.paths import write_atomic
  from raywarp.formats.usd.materials.mdl import bake

  def fails(_f: object) -> None:
    raise ValueError("not an array")

  with pytest.raises(ValueError, match="not an array"):
    write_atomic(tmp_path / "entry.npy", fails, "bake cache")
  assert list(tmp_path.iterdir()) == []
  bake.save(tmp_path / "entry.npy", np.zeros((1, 1, 1), np.float32))
  umask = os.umask(0)
  os.umask(umask)
  assert [p.name for p in tmp_path.iterdir()] == ["entry.npy"]
  assert (tmp_path / "entry.npy").stat().st_mode & 0o777 == 0o666 & ~umask
