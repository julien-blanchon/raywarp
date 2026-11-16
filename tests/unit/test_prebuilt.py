"""Prebuilt kernel bundles (`core/build/kernel_cache/prebuilt.py`, `tools/prebuilt`): checked once,
seeded into the user's caches, read in place; missing, damaged or mismatched entries warn and
compile once; concurrent first starts copy and compile once; the fill records the census's modules
per tier; collect, merge and the module packs. The OptiX module service:
tests/unit/test_optix_modules.py."""

from __future__ import annotations

import concurrent.futures
import errno
import hashlib
import json
import lzma
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import time
import types
import warnings
from pathlib import Path
from typing import cast

import pytest
import warp as wp

from raywarp.core.build.kernel_cache import kernel_code, prebuilt
from raywarp.core.build.kernel_cache import store as kernel_store
from raywarp.core.kernels.abi import toggles
from raywarp.core.trace.optix import pipeline

_SCHEMA = (
  "CREATE TABLE cache_info (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, optix_version VARCHAR(32), "
  "driver_version VARCHAR(32), size INTEGER, timestamp INTEGER)",
  "CREATE TABLE cache_data (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, value BLOB)",
)
_RECORD: dict[str, object] = {
  "has_curves": False,
  "has_custom": False,
  "opacity_micromaps": False,
  "max_registers": 96,
  "bound": [],
}
_PTX = b"// ptx\n.entry e()\n{\n}\n"


def _entry(ptx: str = "", groups: dict[str, list[dict[str, object]]] | None = None, *, code: str = "") -> dict[str, object]:
  """An index (or PTX cache) entry's JSON of PTX file `ptx`: generated from every kernel-code
  function of this code (or of `code`: other kernel code)."""
  return {"ptx": ptx, "code": code or kernel_code.version(), "reach": ["*"], "groups": groups or {}}


def _optix_db(path: Path, keys: list[str]) -> None:
  """An OptiX disk cache (its two tables) holding `keys`."""
  with sqlite3.connect(path) as db:
    for sql in _SCHEMA:
      db.execute(sql)
    for k in keys:
      db.execute("INSERT INTO cache_info VALUES (?, '9.0', '580', 1, 0)", (k,))
      db.execute("INSERT INTO cache_data VALUES (?, ?)", (k, k.encode()))


def _install(store: Path, *, warp: str = "", code: str = "", arch: int = 120) -> Path:
  """A complete bundle in the kernel store `store` (sm120/d0, pinned with its file list): one Warp
  module, the PTX of key "k" and its module record (default: this Warp release, the entry of this
  kernel code)."""
  b = store / "sm120" / "d0"
  (b / "warp" / "wp_mod_abc").mkdir(parents=True)
  (b / "warp" / "wp_mod_abc" / "wp_mod_abc.sm120.cubin").write_bytes(b"cubin")
  (b / "warp" / "wp_mod_abc" / "wp_mod_abc.meta").write_text("{}")
  (b / "ptx").mkdir()
  (b / "ptx" / f"c0{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(_PTX))
  entry = _entry(f"c0{prebuilt.PTX_SUFFIX}", {"fast": [_RECORD]}, code=code)
  (b / prebuilt.entry_path("k")).write_text(json.dumps(entry))
  prebuilt.Manifest(arch=arch, warp=warp or wp.config.version, content="d0").write(b / prebuilt.MANIFEST)
  _seal(b)
  return b


def _seal(b: Path) -> None:
  """(Re)writes the file list of bundle `b` and pins it."""
  listing = kernel_store.Listing.of(b)
  (b / kernel_store.FILES).write_bytes(listing)
  pins = {"bundles": {"120": {"content": b.name, "files": hashlib.sha256(listing).hexdigest()}}}
  kernel_store.PIN.write_text(json.dumps(pins))


def _install_pack(store: Path, driver: str, rows: dict[str, list[str]]) -> Path:
  """The OptiX module pack of `driver` of the store's bundle: per module name its rows' keys."""
  p = store / "sm120" / "d0" / "optix" / driver
  (p / prebuilt.MODULE_DIR).mkdir(parents=True)
  for name, keys in rows.items():
    _optix_db(p / prebuilt.module_path(name), keys)
  (p / kernel_store.FILES).write_bytes(kernel_store.Listing.of(p))
  return p


def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
  """A new process's view: nothing checked, seeded or warned yet."""
  for name, empty in (("_SEEDED", set()), ("_BUNDLES", {}), ("_PACKS", {}), ("_WARNED", set()), ("_UNREACHABLE", [])):
    monkeypatch.setattr(prebuilt, name, empty)


@pytest.fixture
def user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Empty user caches (raywarp's directory, Warp's kernel cache), an empty kernel store
  (`tmp_path`/store) whose remote is `tmp_path`/remote (file://), no pins, nothing checked or
  seeded yet, driver 575.1."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path / "user"))
  monkeypatch.setenv(kernel_store.STORE_ENV, str(tmp_path / "store"))
  monkeypatch.setenv(kernel_store.URL_ENV, (tmp_path / "remote").as_uri())
  monkeypatch.delenv(prebuilt.BUNDLE_ENV, raising=False)
  monkeypatch.delenv("OPTIX_CACHE_PATH", raising=False)
  monkeypatch.setattr(kernel_store, "PIN", tmp_path / "kernels.json")
  monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))
  _fresh(monkeypatch)
  monkeypatch.setattr(prebuilt, "driver_version", lambda: "575.1")
  return tmp_path / "user"


def _remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """The store's bundle (and packs) laid out on the remote (`tools/release/kernels.py layout`),
  the store emptied: everything is fetched."""
  from tools.release import kernels

  store = tmp_path / "store" / "sm120" / "d0"
  packs = sorted((store / "optix").iterdir()) if (store / "optix").is_dir() else []
  out = tmp_path / "remote"
  bundle = tmp_path / "bundle"
  shutil.copytree(store, bundle, ignore=shutil.ignore_patterns("optix", kernel_store.FILES))
  kernels.layout(bundle, out, pack=None, workers=2)
  for pack in packs:
    kernels._lay(pack, out / "sm120" / "d0" / "optix" / pack.name, 2)  # a pack without its manifest
  kernels.pin(out)
  shutil.rmtree(tmp_path / "store")
  _fresh(monkeypatch)
  return out / "sm120" / "d0"


def _keys(db: Path) -> list[str]:
  """The keys of an OptiX disk cache."""
  with sqlite3.connect(db) as c:
    return sorted(r[0] for r in c.execute("SELECT key FROM cache_data"))


def test_a_module_s_rows_come_from_its_pack_file(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`seed_module`: the running driver's pack merges one module's rows into an OptiX cache (the
  others' stay out; another driver's pack: none, warned once); a module the pack lacks merges
  nothing; seeding the bundle places the common Warp modules and merges no OptiX row."""
  _install(tmp_path / "store")
  _install_pack(tmp_path / "store", "580", {"m1": ["a"], "m2": ["b", "c"]})
  db = user / "optix" / prebuilt.OPTIX_DB
  db.parent.mkdir(parents=True)
  _optix_db(db, ["z"])
  prebuilt.seed(120)
  assert _keys(db) == ["z"]
  assert (tmp_path / "warp" / "wp_mod_abc" / "wp_mod_abc.sm120.cubin").read_bytes() == b"cubin"
  with pytest.warns(RuntimeWarning, match="no OptiX module pack for NVIDIA driver 575.1"):
    assert not prebuilt.seed_module(120, "m2", db)
  _fresh(monkeypatch)
  monkeypatch.setattr(prebuilt, "driver_version", lambda: "580")
  assert prebuilt.seed_module(120, "m2", db)
  assert _keys(db) == ["b", "c", "z"]
  assert not prebuilt.seed_module(120, "other", db)
  assert _keys(db) == ["b", "c", "z"]


def test_a_missing_optix_cache_starts_from_the_module_file(tmp_path: Path, user: Path) -> None:
  """No OptiX cache yet (OptiX creates it on its first module): the module's file becomes it."""
  _install(tmp_path / "store")
  _install_pack(tmp_path / "store", "575.1", {"m1": ["a"]})
  db = user / "optix" / prebuilt.OPTIX_DB
  assert prebuilt.seed_module(120, "m1", db)
  assert _keys(db) == ["a"]


class _OtherSqlite:
  """A connection through a second SQLite library of this process (the system's, loaded apart),
  as OptiX's own copy holds its disk cache open in WAL mode while the renderer merges into it."""

  def __init__(self, path: Path) -> None:
    """Opens `path` in WAL mode (skips without a system SQLite)."""
    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("sqlite3")
    if name is None:
      pytest.skip("no system SQLite library to stand in for OptiX's")
    self._lib = ctypes.CDLL(name, mode=os.RTLD_LOCAL)
    self._lib.sqlite3_errmsg.restype = ctypes.c_char_p
    self._db = ctypes.c_void_p()
    assert self._lib.sqlite3_open(str(path).encode(), ctypes.byref(self._db)) == 0
    self._lib.sqlite3_busy_timeout(self._db, 30_000)
    self.run("PRAGMA journal_mode=WAL")

  def run(self, sql: str) -> None:
    """Executes `sql`.

    Raises:
      AssertionError: SQLite refused it.
    """
    rc = self._lib.sqlite3_exec(self._db, sql.encode(), None, None, None)
    assert rc == 0, self._lib.sqlite3_errmsg(self._db).decode()

  def close(self) -> None:
    """Closes the connection."""
    self._lib.sqlite3_close(self._db)


def test_merges_keep_the_rows_optix_writes_meanwhile(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Seeds of several modules at once, while this process's OptiX (its own SQLite library) holds
  its WAL cache open and writes the modules it creates: every merge succeeds and keeps every row
  OptiX wrote before and after it (an in-process merge deleted OptiX's open WAL on close: its
  later rows were lost, and a later merge failed with "locking protocol")."""
  _install(tmp_path / "store")
  mods = {f"m{i}": [f"p{i}"] for i in range(6)}
  _install_pack(tmp_path / "store", "580", mods)
  _fresh(monkeypatch)
  monkeypatch.setattr(prebuilt, "driver_version", lambda: "580")
  db = user / "optix" / prebuilt.OPTIX_DB
  db.parent.mkdir(parents=True)
  _optix_db(db, ["z"])
  optix = _OtherSqlite(db)
  try:
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
      seeds = [pool.submit(prebuilt.seed_module, 120, m, db) for m in mods]
      for i, done in enumerate(seeds):
        optix.run(f"INSERT INTO cache_data VALUES ('o{i}', x'00')")  # a module OptiX created
        assert done.result()
    optix.run("INSERT INTO cache_data VALUES ('last', x'00')")
  finally:
    optix.close()
  want = sorted(["z", "last", *(f"o{i}" for i in range(6)), *(f"p{i}" for i in range(6))])
  assert _keys(db) == want


def test_a_failed_merge_warns_and_the_module_is_created(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A merge that fails (here: no space left) warns; the module is then OptiX's to create."""
  _install(tmp_path / "store")
  _install_pack(tmp_path / "store", "575.1", {"m1": ["a"]})

  def full(*_args: object) -> None:
    raise OSError(errno.ENOSPC, "No space left on device")

  monkeypatch.setattr(prebuilt, "merge_optix", full)
  with pytest.warns(RuntimeWarning, match="was not merged"):
    assert not prebuilt.seed_module(120, "m1", user / "optix" / prebuilt.OPTIX_DB)


def test_large_specialisation_families_are_fetched_by_name_only(tmp_path: Path, user: Path) -> None:
  """Seeding places every module but those of a family of more than `FAMILY_FETCHED_BY_NAME`
  variants (one kernel factory's per-Features modules): a renderer fetches those it lists."""
  b = _install(tmp_path / "store")
  family = [f"wp_relight__locals__rays_{i:08x}_{i:07x}" for i in range(prebuilt.FAMILY_FETCHED_BY_NAME + 1)]
  for name in family:
    (b / "warp" / name).mkdir()
    (b / "warp" / name / f"{name}.sm120.cubin").write_bytes(b"r")
  _seal(b)
  prebuilt.seed(120)
  assert sorted(p.name for p in (tmp_path / "warp").iterdir()) == ["wp_mod_abc"]
  assert prebuilt.fetch_modules(120, family[:2]) == 2


def test_a_renderer_s_modules_ptx_and_pack_are_fetched_on_use(
  tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """From a remote alone: the bundle check fetches only its file list and manifest; `fetch_modules`
  fetches the listed modules Warp's cache lacks (checked) and links them into it, a name the
  bundle lacks is left to Warp; the PTX is fetched when its build asks; of the driver's pack, its
  file list and the files of the modules created, nothing else; a second process fetches nothing."""
  _install(tmp_path / "store")
  _install_pack(tmp_path / "store", "575.1", {"m1": ["a"], "m2": ["b"]})
  _remote(tmp_path, monkeypatch)
  store = tmp_path / "store" / "sm120" / "d0"
  assert prebuilt.bundle(120) is not None
  assert sorted(p.relative_to(store).as_posix() for p in store.rglob("*") if p.is_file()) == ["files.json", "manifest.json"]
  assert prebuilt.fetch_modules(120, ["wp_mod_abc", "wp_own_123"]) == 1
  assert (tmp_path / "warp" / "wp_mod_abc" / "wp_mod_abc.sm120.cubin").read_bytes() == b"cubin"
  assert os.stat(tmp_path / "warp" / "wp_mod_abc" / "wp_mod_abc.sm120.cubin").st_nlink == 2  # linked, not copied
  assert prebuilt.fetch_modules(120, ["wp_mod_abc"]) == 0  # in Warp's cache: nothing to do
  assert prebuilt.ptx(120, "k", expected=True) == _PTX
  prebuilt.seed(120)
  assert prebuilt.seed_module(120, "m1", user / "optix" / prebuilt.OPTIX_DB)
  assert _keys(user / "optix" / prebuilt.OPTIX_DB) == ["a"]
  pack = store / "optix" / "575.1"
  assert sorted(p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file()) == ["files.json", "m/m1.db"]
  shutil.rmtree(tmp_path / "remote")
  _fresh(monkeypatch)
  shutil.rmtree(tmp_path / "warp")
  with warnings.catch_warnings():
    warnings.simplefilter("error")  # the store serves it all: no fetch, no warning
    assert prebuilt.fetch_modules(120, ["wp_mod_abc"]) == 1
    assert prebuilt.ptx(120, "k", expected=True) == _PTX


def test_the_common_modules_arrive_in_one_request(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Seeding from a remote fetches the common Warp modules as one archive (checked member by
  member against the pinned file list, then deleted) and links them into Warp's cache."""
  _install(tmp_path / "store")
  _remote(tmp_path, monkeypatch)
  opened: list[str] = []
  real = kernel_store._open  # the requests, counted
  monkeypatch.setattr(kernel_store, "_open", lambda url: opened.append(url) or real(url))
  prebuilt.seed(120)
  assert (tmp_path / "warp" / "wp_mod_abc" / "wp_mod_abc.meta").is_file()
  assert [u.rsplit("/d0/", 1)[1] for u in opened if "/warp/" in u] == ["warp/common.tar.xz"]
  assert not (tmp_path / "store" / "sm120" / "d0" / kernel_store.COMMON).exists()


def test_concurrent_unpacks_of_one_archive_both_succeed(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A second process's unpack of the same archive runs to its end between this one's download and
  its open (two first starts at once): each reads its own download, so neither loses the archive
  to the other's clean-up, and the members land once."""
  _install(tmp_path / "store")
  remote = _remote(tmp_path, monkeypatch)
  local, url = tmp_path / "store" / "sm120" / "d0", remote.as_uri()
  files = kernel_store.listing(local, url, None)
  real, others = kernel_store.tarfile.open, []

  def other_process_first(path: Path) -> object:
    if not others:
      others.append(None)  # the other process opens its own archive unhindered
      others[0] = kernel_store.unpack(local, url, files, kernel_store.COMMON)
    return real(path)

  monkeypatch.setattr(kernel_store.tarfile, "open", other_process_first)
  assert kernel_store.unpack(local, url, files, kernel_store.COMMON) == {}
  assert others == [{}]
  assert (local / "warp" / "wp_mod_abc" / "wp_mod_abc.meta").is_file()
  assert not [p for p in (local / "warp").iterdir() if p.is_file()]  # no archive or partial left


def test_without_network_or_store_the_kernels_compile_and_warn_once(
  tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A pinned bundle the store lacks and the remote cannot serve: one loud warning per process;
  its builds and modules then compile (nothing served)."""
  _install(tmp_path / "store")
  _remote(tmp_path, monkeypatch)
  shutil.rmtree(tmp_path / "remote")
  with pytest.warns(RuntimeWarning, match="COMPILES now") as record:
    assert prebuilt.bundle(120) is None
    assert prebuilt.ptx(120, "k", expected=True) is None
    assert prebuilt.fetch_modules(120, ["wp_mod_abc"]) == 0
  assert len([w for w in record if "COMPILES now" in str(w.message)]) == 1


def test_a_download_that_is_not_the_pinned_one_is_refused(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A remote file whose content is not what the pinned file list names (tampered, damaged) is
  not stored: it warns and its build compiles; a file list that is not the pinned one refuses the
  whole bundle."""
  _install(tmp_path / "store")
  remote = _remote(tmp_path, monkeypatch)
  (remote / "ptx" / f"c0{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(b"// other\n"))
  with pytest.warns(RuntimeWarning, match="sha256"):
    assert prebuilt.ptx(120, "k", expected=True) is None
  assert not (tmp_path / "store" / "sm120" / "d0" / "ptx" / f"c0{prebuilt.PTX_SUFFIX}").exists()
  _fresh(monkeypatch)
  shutil.rmtree(tmp_path / "store")
  (remote / kernel_store.FILES).write_text('{"files": {}}')
  with pytest.warns(RuntimeWarning, match="not this release.s"):
    assert prebuilt.bundle(120) is None


def test_prefetch_all_fills_the_store_for_offline_use(tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`prebuilt.prefetch`: every file of the bundle and the driver's pack in the store; offline
  afterwards (no remote), the store alone serves."""
  _install(tmp_path / "store")
  _install_pack(tmp_path / "store", "580", {"m1": ["a"], "m2": ["b"]})
  _remote(tmp_path, monkeypatch)
  assert prebuilt.prefetch(120, driver="580") == tmp_path / "store" / "sm120" / "d0"
  store = tmp_path / "store" / "sm120" / "d0"
  assert (store / "warp" / "wp_mod_abc" / "wp_mod_abc.meta").is_file()
  assert all((store / "optix" / "580" / prebuilt.module_path(m)).is_file() for m in ("m1", "m2"))
  shutil.rmtree(tmp_path / "remote")
  _fresh(monkeypatch)
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert prebuilt.ptx(120, "k", expected=True) == _PTX
    assert prebuilt.fetch_modules(120, ["wp_mod_abc"]) == 1


@pytest.mark.parametrize(
  ("manifest", "reason"),
  [({"warp": "0.1.0"}, "Warp 0.1.0"), ({"arch": 89}, "sm_89, not sm_120")],
)
def test_a_mismatched_bundle_is_not_used_and_says_why(tmp_path: Path, user: Path, manifest: dict, reason: str) -> None:
  """A bundle of another Warp release or architecture could never hit: it warns once, is not
  read, and its builds compile."""
  _install(tmp_path / "store", **manifest)
  with pytest.warns(RuntimeWarning, match=reason):
    prebuilt.seed(120)
  assert not (tmp_path / "warp").exists()
  assert prebuilt.bundle(120) is None
  with pytest.warns(RuntimeWarning, match="no prebuilt kernels for sm_120"):
    assert prebuilt.ptx(120, "k", expected=True) is None


def test_a_damaged_manifest_is_not_used(tmp_path: Path, user: Path) -> None:
  """A manifest that does not parse warns and leaves the bundle unused."""
  b = _install(tmp_path / "store")
  (b / prebuilt.MANIFEST).write_text('{"arch": 1')
  with pytest.warns(RuntimeWarning, match="damaged manifest"):
    assert prebuilt.bundle(120) is None


def test_without_a_bundle_an_expected_build_warns_once(tmp_path: Path, user: Path, recwarn: pytest.WarningsRecorder) -> None:
  """No pinned bundle: nothing is seeded; a build the bundle would hold warns once per process
  that the builds compile on first use; the scene's own builds say nothing."""
  prebuilt.seed(120)
  assert not user.exists()
  assert prebuilt.ptx(120, "k", expected=False) is None
  assert not recwarn.list
  for key in ("k", "other"):
    assert prebuilt.ptx(120, key, expected=True) is None
  assert [str(w.message) for w in recwarn.list if "no prebuilt kernels for sm_120" in str(w.message)] != []
  assert len(recwarn.list) == 1


def test_bundled_ptx_is_read_in_place_and_a_missing_build_warns(tmp_path: Path, user: Path) -> None:
  """The bundle's compressed PTX is read by key through its entry; a key the bundle should hold
  but lacks warns that it compiles once; an unexpected key (a scene's own build) is silent."""
  _install(tmp_path / "store")
  assert prebuilt.ptx(120, "k", expected=True) == _PTX
  assert prebuilt.ptx(120, "own", expected=False) is None
  with pytest.warns(RuntimeWarning, match="lacks the build other of this kernel code: compiling it once"):
    assert prebuilt.ptx(120, "other", expected=True) is None


def test_a_bundled_build_of_other_kernel_code_is_not_used(tmp_path: Path, user: Path) -> None:
  """An entry generated from other kernel code (its digest differs from the installed
  code's over its reach) reads as absent: warned when expected, compiled once; the bundle's other
  parts (its Warp modules, by content) still serve."""
  _install(tmp_path / "store", code="other")
  assert prebuilt.bundle(120) is not None
  with pytest.warns(RuntimeWarning, match="lacks the build k of this kernel code"):
    assert prebuilt.ptx(120, "k", expected=True) is None
  bundle = prebuilt.bundle(120)
  assert bundle is not None and bundle.keyed() == {}


@pytest.mark.parametrize(
  ("damage", "message"), [("truncated", "is damaged"), ("garbage", "is damaged"), ("missing", "could not be fetched")]
)
def test_a_damaged_bundle_entry_warns_and_compiles(tmp_path: Path, user: Path, damage: str, message: str) -> None:
  """A truncated or corrupt compressed PTX in the store (xz's integrity check) warns and reads as
  absent, and so does a missing one the remote cannot serve: the build compiles once and is
  cached."""
  b = _install(tmp_path / "store")
  entry = b / "ptx" / f"c0{prebuilt.PTX_SUFFIX}"
  data = entry.read_bytes()
  if damage == "missing":
    entry.unlink()
  else:
    entry.write_bytes(data[: len(data) // 2] if damage == "truncated" else data[:20] + b"x" * 16 + data[36:])
  with pytest.warns(RuntimeWarning, match=message):
    assert prebuilt.ptx(120, "k", expected=False) is None


def test_an_entry_that_does_not_parse_warns_and_compiles(tmp_path: Path, user: Path) -> None:
  """An entry file that does not parse warns; its build then compiles."""
  b = _install(tmp_path / "store")
  (b / prebuilt.entry_path("k")).write_text("{")
  with pytest.warns(RuntimeWarning, match="PTX of k .* is damaged"):
    assert prebuilt.ptx(120, "k", expected=False) is None


def test_a_start_reads_only_its_own_builds_entries(tmp_path: Path, user: Path) -> None:
  """A build's PTX reads its own entry file; the other builds' entries are not opened (a start
  parses its few builds' entries, not the whole bundle's), and `keyed` lists every build."""
  b = _install(tmp_path / "store")
  (b / prebuilt.entry_path("unread")).write_text("{")  # damaged: opening it would raise
  assert prebuilt.ptx(120, "k", expected=True) == _PTX
  bundle = prebuilt.bundle(120)
  assert bundle is not None and bundle.keys() == ["k"]


@pytest.mark.parametrize(
  "entry",
  [
    [],
    {"ptx": 3, "groups": {}},
    {"ptx": "c0.ptx.xz", "modules": []},
    {"ptx": "c0.ptx.xz", "groups": {"fast": [{"bound": []}]}},
    {"ptx": "c0.ptx.xz", "groups": {"fast": {}}},
  ],
)
def test_a_damaged_entry_warns_and_compiles(tmp_path: Path, user: Path, entry: object) -> None:
  """An entry of the wrong shape (not an object, a field of the wrong type or missing, a record
  missing its fields) warns and reads as absent; a tier's module list refuses the bundle."""
  b = _install(tmp_path / "store")
  (b / prebuilt.entry_path("k")).write_text(json.dumps(entry))
  with pytest.warns(RuntimeWarning, match="entry k is damaged"):
    assert prebuilt.ptx(120, "k", expected=True) is None
  bundle = prebuilt.bundle(120)
  assert bundle is not None
  with pytest.raises(ValueError, match="entry k is damaged"):
    bundle.entries()


def test_a_truncated_cached_ptx_is_compiled_again(tmp_path: Path, user: Path) -> None:
  """A cached PTX cut short (it does not end with a closing brace) warns, is deleted and reads as
  absent: compiled again."""
  path = user / "ptx" / "k.ptx"
  path.parent.mkdir(parents=True)
  path.with_suffix(".json").write_text(json.dumps(_entry()))
  path.write_bytes(_PTX[:-4])
  with pytest.warns(RuntimeWarning, match="truncated"):
    assert pipeline._read(path) is None  # the cache read under test
  assert not path.exists()
  path.write_bytes(_PTX)
  assert pipeline._read(path) == _PTX


def test_a_warp_ptx_with_its_nul_terminator_reads_whole(tmp_path: Path, user: Path) -> None:
  """A PTX as Warp writes it (`}\\n\\n\\0`: the NUL terminator after the last brace) is whole: read
  as it is, kept, no warning (the truncation check once deleted every cached PTX on each start)."""
  path = user / "ptx" / "k.ptx"
  path.parent.mkdir(parents=True)
  path.with_suffix(".json").write_text(json.dumps(_entry()))
  warp_style = _PTX.rstrip() + b"\n\n\0"
  path.write_bytes(warp_style)
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert pipeline._read(path) == warp_style  # the cache read under test
  assert path.exists()


def test_an_interrupted_store_never_pairs_this_code_s_entry_with_an_old_ptx(
  tmp_path: Path, user: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A store of a key whose cached PTX is an older kernel code's (a key holds across code edits)
  that stops between its two files leaves no entry of this code beside the old PTX: the entry is
  written last (written first, a reader of this code meanwhile took the old PTX as fresh)."""
  path = user / "ptx" / "k.ptx"
  path.parent.mkdir(parents=True)
  path.with_suffix(".json").write_text(json.dumps(_entry(code="older")))
  path.write_bytes(b"// older\n.entry e()\n{\n}\n")
  replace, calls = Path.replace, []

  def second_fails(self: Path, target: Path) -> Path:
    """The store's second file fails (a full disk; other caches' writes go through)."""
    if Path(target).parent == path.parent:
      calls.append(target)
    if len(calls) == 2:
      raise OSError(errno.ENOSPC, "no space left")
    return replace(self, target)

  with monkeypatch.context() as m, pytest.warns(UserWarning, match="cannot be written"):
    m.setattr(Path, "replace", second_fails)
    pipeline._store(path, _PTX)  # the cache write under test
  assert pipeline._read(path) is None  # stale: compiled again under the lock


def _compile_in_a_process(path: str, log: str) -> bytes:
  """A process's first use of build `path`: `pipeline.compile_once` with a slow compile that logs
  itself (the test's worker)."""

  def slow() -> bytes:
    with Path(log).open("a") as f:
      f.write(f"{os.getpid()}\n")
    time.sleep(1.0)
    return _PTX

  return pipeline.compile_once(Path(path), slow)


def test_concurrent_first_use_compiles_once(tmp_path: Path) -> None:
  """Two processes needing the same missing build at once: one compiles and stores it, the other
  waits on the lock and reads it (one compile)."""
  path, log = tmp_path / "ptx" / "k.ptx", tmp_path / "compiles.log"
  path.parent.mkdir()
  with concurrent.futures.ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
    got = list(pool.map(_compile_in_a_process, [str(path)] * 2, [str(log)] * 2))
  assert got == [_PTX, _PTX]
  assert len(log.read_text().split()) == 1


def test_module_records_round_trip() -> None:
  """A build's module options (`pipeline.module_record`) survive the index's JSON (`Entry.json`,
  `Entry.of`) and give the compile options back (`pipeline.record_options`): what a scene-free
  module creation compiles."""
  flags = types.SimpleNamespace(
    has_world_curves=False, has_shared_curves=True, primitive_types=(), shared_primitive_types=(4,), has_omm=True
  )
  record = pipeline.module_record(flags, 96, ((40, b"\x04\x00\x00\x00"),), shapes=toggles.SHAPE_ALL)
  groups: dict[str, tuple[prebuilt.ModuleRecord, ...]] = {"fast": (record,), "default": (record,)}
  entry = prebuilt.Entry(ptx="p", code="c", reach=frozenset({"a.py:f", "b.py:g"}), groups=groups)
  back = prebuilt.Entry.of(json.loads(json.dumps(entry.json())))
  assert back == entry and back.modules == (record,)
  assert (record.has_curves, record.has_custom, record.opacity_micromaps, record.max_registers) == (True, True, True, 96)
  mco = pipeline.record_options(record, calls=()).module
  assert mco.maxRegisterCount == 96 and mco.numBoundValues == 1


def test_collect_stores_ptx_by_content_with_the_census_records(tmp_path: Path) -> None:
  """`tools/prebuilt/build.py collect`: the Warp modules of the architecture, each distinct PTX
  once (two builds with the same PTX share its file) with their entries (records per
  tier: none for a build no census scene uses); a PTX without an entry of this kernel code is
  left out; the manifest sealed with its content digest."""
  from tools.prebuilt import build

  root = tmp_path / "root"
  module = root / "warp" / wp.config.version / "wp_mod"
  module.mkdir(parents=True)
  (module / "wp_mod.sm120.cubin").write_bytes(b"c")
  (module / "wp_mod.sm89.cubin").write_bytes(b"other")
  (root / "raywarp" / "ptx").mkdir(parents=True)
  other = _PTX + b"\n.entry f()\n{\n}\n"
  for key, ptx in (("k1", _PTX), ("k2", _PTX), ("k3", other), ("stale", other), ("bare", other)):
    (root / "raywarp" / "ptx" / f"{key}.ptx").write_bytes(ptx)
    groups = {"fast": [_RECORD]} if key in ("k1", "k2") else {}
    if key != "bare":
      (root / "raywarp" / "ptx" / f"{key}.json").write_text(
        json.dumps(_entry("", groups, code="old" if key == "stale" else ""))
      )
  bundle = build.collect(root, 120, tmp_path / "out", workers=2)
  assert sorted(p.name for p in (bundle / "warp" / "wp_mod").iterdir()) == ["wp_mod.sm120.cubin"]
  index = build.read_entries(bundle)
  assert sorted(index) == ["k1", "k2", "k3"] and index["k1"]["ptx"] == index["k2"]["ptx"] != index["k3"]["ptx"]
  assert index["k1"]["groups"] == {"fast": [_RECORD]} and index["k3"]["groups"] == {}
  assert index["k1"]["code"] == kernel_code.version() and index["k1"]["reach"] == ["*"]
  assert len(list((bundle / "ptx").glob(f"*{prebuilt.PTX_SUFFIX}"))) == 2
  assert lzma.decompress((bundle / "ptx" / str(index["k1"]["ptx"])).read_bytes()) == build.normalise(_PTX)
  assert prebuilt.Manifest.read(bundle / prebuilt.MANIFEST).content


def test_normalised_ptx_keeps_only_what_optix_reads() -> None:
  """`build.normalise` drops comments, blank lines, line info and debug sections (keeping a local
  depot: OptiX refuses a module without it) and numbers each function's registers and labels by
  first use: two builds differing only in those agree."""
  from tools.prebuilt import build

  def ptx(r: int, label: int) -> bytes:
    body = ".entry e()\n{\n\t.local .align 4 .b8 \t__local_depot0[12];\n\t.reg .b32 %r<12>;\n\t.loc 1 2 3\n"
    body += f"$L__BB0_{label}:\n\tmov.b32 %r{r}, 1;  // set\n"
    body += f"\tadd.s32 %r{r + 1}, %r{r}, 1;\n\tbra $L__BB0_{label};\n}}\n"
    return f'// header\n.version 9.0\n\n.file 1 "a.cu"\n{body}.section .debug_info\n{{\n.b8 1\n}}\n'.encode()

  out = build.normalise(ptx(7, 3))
  assert out == build.normalise(ptx(2, 9))
  assert out == (
    b".version 9.0\n.entry e()\n{\n\t.local .align 4 .b8 \t__local_depot0[12];\n\t.reg .b32 %r<12>;\n$L__0:\n"
    b"\tmov.b32 %r0, 1;\n\tadd.s32 %r1, %r0, 1;\n\tbra $L__0;\n}\n"
  )


def test_normalised_ptx_names_do_not_vary_between_fills() -> None:
  """`build.normalise`: NVRTC's file-scope prefix (it names Warp's source file and its module hash)
  and the programs' Warp source-hash suffixes become fixed names, so two fills of the same code
  (or of code differing only in comments) give the same bytes; the runtime still finds each
  program by its semantic name (`pipeline.entry_names`); two programs one name apart are refused."""
  from raywarp.core.trace.optix import pipeline
  from tools.prebuilt import build

  def ptx(module: str, program: str) -> bytes:
    sym = f"_ZN58_INTERNAL_00000000_27_wp_raywarp_optix_{module}_cu_8d30c1fb2wp9SLICE_ENDE"
    return (
      f".version 9.0\n.global .align 4 .u32 {sym} = 1;\n.visible .entry __raygen__probe_ao_{program}()\n{{\n"
      f"\tld.global.u32 %r1, [{sym}];\n}}\n.visible .entry __miss__render_{program[::-1]}()\n{{\n}}\n"
    ).encode()

  out = build.normalise(ptx("bdefb90", "f86e48f3"))
  assert out == build.normalise(ptx("22ae3f2", "0a1b2c3d"))
  assert b"wp_raywarp_optix" not in out and b"_ZN_INTERNAL_rw_2wp9SLICE_ENDE" in out
  names = pipeline.entry_names(out, ["__raygen__probe_ao", "__miss__render"])
  assert names == {"__raygen__probe_ao": "__raygen__probe_ao", "__miss__render": "__miss__render"}

  # Warp orders a module's kernels by their source hash (a comment moves them), NVRTC numbers each
  # local-memory depot by position and ends the text with a NUL: one order and numbering.
  def program(name: str, depot: int) -> str:
    return f".visible .entry {name}()\n{{\n\t.local .align 4 .b8 \t__local_depot{depot}[8];\n\tmov.u64 %SPL, __local_depot{depot};\n}}\n"

  first = ".version 9.0\n" + program("__raygen__render_0a1b2c3d", 0) + program("__miss__layer_1b2c3d4e", 1) + "\x00"
  second = ".version 9.0\n" + program("__miss__layer_9b2c3d4e", 0) + program("__raygen__render_8a1b2c3d", 1) + "\x00"
  assert build.normalise(first.encode()) == build.normalise(second.encode())
  assert build.normalise(first.encode()).endswith(b"}\n\x00\n")
  twins = b".entry __miss__render_0a1b2c3d()\n{\n}\n.entry __miss__render_1a1b2c3d()\n{\n}\n"
  with pytest.raises(ValueError, match="share a name"):
    build.normalise(twins)


def test_merge_unites_the_shards_bundles(tmp_path: Path) -> None:
  """`tools/prebuilt/build.py merge`: the union of parallel fill shards' bundles (PTX and entries)
  under one sealed manifest; shards of different manifests are refused."""
  from tools.prebuilt import build

  shards = []
  for i in range(2):
    b = tmp_path / f"s{i}"
    (b / "ptx").mkdir(parents=True)
    (b / "ptx" / f"c{i}{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(_PTX))
    build.write_entries(b, {f"k{i}": _entry(f"c{i}{prebuilt.PTX_SUFFIX}")})
    prebuilt.Manifest(arch=120, warp=wp.config.version, content=f"d{i}").write(b / prebuilt.MANIFEST)
    shards.append(b)
  merged = build.merge(shards, tmp_path / "out")
  assert sorted(build.read_entries(merged)) == ["k0", "k1"]
  assert len(list((merged / "ptx").glob(f"*{prebuilt.PTX_SUFFIX}"))) == 2
  assert prebuilt.Manifest.read(merged / prebuilt.MANIFEST).content not in ("d0", "d1")
  prebuilt.Manifest(arch=120, warp="0.1.0", content="d1").write(shards[1] / prebuilt.MANIFEST)
  with pytest.raises(SystemExit, match="differ"):
    build.merge(shards, tmp_path / "out2")


def _module_record(registers: int) -> prebuilt.ModuleRecord:
  """A module record of register cap `registers`."""
  return prebuilt.ModuleRecord(
    has_curves=False, has_custom=True, opacity_micromaps=False, max_registers=registers, bound=((8, "04000000"),)
  )


def test_merge_unites_the_module_records_of_a_shared_key(tmp_path: Path) -> None:
  """A key two shards hold (a GI build two shards' builds share) keeps its PTX file and per tier
  the union of their module records, in order."""
  from tools.prebuilt import build

  def shard(name: str, index: dict[str, object]) -> Path:
    b = tmp_path / name
    (b / "ptx").mkdir(parents=True)
    for entry in index.values():
      (b / "ptx" / entry["ptx"]).write_bytes(lzma.compress(_PTX))  # pyrefly: ignore[bad-index]
    build.write_entries(b, index)
    prebuilt.Manifest(arch=120, warp=wp.config.version, content=name).write(b / prebuilt.MANIFEST)
    return b

  r96, r128 = (_module_record(n).json() for n in (96, 128))
  c0, c1 = f"c0{prebuilt.PTX_SUFFIX}", f"c1{prebuilt.PTX_SUFFIX}"
  first = shard("first", {"k0": _entry(c0, {"fast": [r96]})})
  second = shard("second", {"k0": _entry(c0, {"fast": [r96, r128], "default": [r128]}), "k1": _entry(c1, {"fast": [r128]})})
  merged = build.merge([first, second], tmp_path / "out")
  index = prebuilt.Bundle(merged, prebuilt.Manifest.read(merged / prebuilt.MANIFEST))
  k0 = index.entry("k0")
  assert k0 is not None and k0.ptx == c0
  assert k0.groups == {"fast": (_module_record(96), _module_record(128)), "default": (_module_record(128),)}
  assert index.entry("k1") is not None


def test_the_census_records_each_tier_s_modules_most_used_first(tmp_path: Path) -> None:
  """`fill.census_records`: the census rows' units (`scene_keys.py`) per PTX key and tier, most
  used first, each once; rows of an older census (no units) are counted, not used."""
  from tools.prebuilt import fill

  r96, r128 = (_module_record(n).json() for n in (96, 128))
  rows = [
    {
      "scene": "a",
      "units": [{"key": "k", "group": "fast", "module": r96}, {"key": "g", "group": "fast", "module": r96}],
    },
    {"scene": "b", "units": [{"key": "k", "group": "fast", "module": r128}]},
    {
      "scene": "c",
      "units": [{"key": "k", "group": "fast", "module": r128}, {"key": "k", "group": "default", "module": r96}],
    },
    {"scene": "old", "class": [0]},
  ]
  (tmp_path / "c.json").write_text(json.dumps(rows))
  census, nrow, old = fill.census_records([tmp_path / "c.json"])
  assert (nrow, old) == (4, 1)
  assert census == {
    "k": {"fast": (_module_record(128), _module_record(96)), "default": (_module_record(96),)},
    "g": {"fast": (_module_record(96),)},
  }


def test_the_census_groups_order_and_filter_the_units(tmp_path: Path) -> None:
  """`fill.census_ranked(groups)`: the units group by group (a row's own units are its first `own`,
  the others its neighbours'), only the rows of each group's recipes, a unit once (in its first
  group); `groups_of` parses the pack's `--groups`."""
  from tools.prebuilt import fill

  r96, r128, r160, r192 = (_module_record(n).json() for n in (96, 128, 160, 192))
  rows = [
    {
      "scene": "a",
      "recipe": "fast",
      "own": 1,
      "units": [{"key": "k", "group": "fast", "module": r96}, {"key": "k", "group": "fast", "module": r128}],
    },
    {"scene": "a", "recipe": "quality", "own": 1, "units": [{"key": "q", "group": "default_ss2_counts", "module": r160}]},
    {"scene": "b", "recipe": "default", "own": 1, "units": [{"key": "d", "group": "default", "module": r192}]},
  ]
  (tmp_path / "c.json").write_text(json.dumps(rows))
  groups = fill.groups_of("quality:own,fast+default:own,fast:near")
  assert groups == ((frozenset({"quality"}), "own"), (frozenset({"fast", "default"}), "own"), (frozenset({"fast"}), "near"))
  ranked, nrow, old = fill.census_ranked([tmp_path / "c.json"], groups)
  assert (nrow, old) == (3, 0)
  assert [r.max_registers for _, _, r in ranked] == [160, 192, 96, 128]  # a tie: by key
  census, _, _ = fill.census_records([tmp_path / "c.json"])  # every row: own units, then neighbours
  assert census["k"]["fast"] == (_module_record(96), _module_record(128))
  with pytest.raises(ValueError, match="own"):
    fill.groups_of("fast:all")


def test_merge_takes_the_census_records(tmp_path: Path) -> None:
  """`build.py merge --census`: the census's module records join the shards' entries (the shards
  compile without the census), united with what the shards recorded; a census key no shard holds
  is reported, not added."""
  from tools.prebuilt import build

  b = tmp_path / "s0"
  (b / "ptx").mkdir(parents=True)
  (b / "ptx" / f"c0{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(_PTX))
  build.write_entries(b, {"k0": _entry(f"c0{prebuilt.PTX_SUFFIX}", {"fast": [_module_record(96).json()]})})
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="d0").write(b / prebuilt.MANIFEST)
  units = [
    {"key": "k0", "group": "fast", "module": _module_record(128).json()},
    {"key": "gone", "group": "fast", "module": _module_record(96).json()},
  ]
  (tmp_path / "c.json").write_text(json.dumps([{"scene": "s", "recipe": "fast", "own": 2, "units": units}]))
  merged = build.merge([b], tmp_path / "out", (tmp_path / "c.json",))
  bundle = prebuilt.Bundle(merged, prebuilt.Manifest.read(merged / prebuilt.MANIFEST))
  k0 = bundle.entry("k0")
  assert k0 is not None and k0.groups == {"fast": (_module_record(96), _module_record(128))}
  assert bundle.entry("gone") is None


def test_a_pack_holds_every_record_or_the_selected_census_s(tmp_path: Path) -> None:
  """`build.py pack --select`: without it every record of the index is packed; with census files
  only the (key, tier, record) they reach (the registry census and a tier's neighbours: the rest
  is created on first use)."""
  from tools.prebuilt import build

  r96, r128 = (_module_record(n).json() for n in (96, 128))
  b = tmp_path / "b"
  (b / "ptx").mkdir(parents=True)
  index = {
    "k0": _entry("c0.ptx.xz", {"fast": [r96, r128], "default": [r128]}),
    "k1": _entry("c1.ptx.xz", {"fast": [r96]}),
  }
  build.write_entries(b, index)
  bundle = prebuilt.Bundle(b, prebuilt.Manifest(arch=120, warp=wp.config.version, content="b"))
  every = build._packed(bundle, ())
  assert set(every) == {
    ("c0.ptx.xz", _module_record(96)),
    ("c0.ptx.xz", _module_record(128)),
    ("c1.ptx.xz", _module_record(96)),
  }
  (tmp_path / "c.json").write_text(json.dumps([{"scene": "s", "units": [{"key": "k0", "group": "default", "module": r128}]}]))
  assert build._packed(bundle, (tmp_path / "c.json",)) == [("c0.ptx.xz", _module_record(128))]


def test_a_seeded_refill_bundles_only_its_listed_keys_and_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`fill --seed B`: a listed key whose entry in B is of this kernel code is taken (its PTX and
  reach; not compiled), a key of other code is not; only the listed Warp modules are copied; so
  `collect` bundles exactly the listed keys and modules (B's dead key and module are left out)."""
  from tools.prebuilt import build, fill

  b = tmp_path / "prev"
  (b / "ptx").mkdir(parents=True)
  for name in ("wp_mod_abc", "wp_mod_dead"):
    (b / "warp" / name).mkdir(parents=True)
    (b / "warp" / name / f"{name}.sm120.cubin").write_bytes(b"cubin")
  (b / "ptx" / f"c0{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(_PTX))
  index = {"k": _entry(f"c0{prebuilt.PTX_SUFFIX}"), "dead": _entry(f"c0{prebuilt.PTX_SUFFIX}")}
  index["stale"] = _entry(f"c0{prebuilt.PTX_SUFFIX}", code="old")
  build.write_entries(b, index)
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="b").write(b / prebuilt.MANIFEST)
  root = tmp_path / "root"
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(root / "raywarp"))
  (root / "raywarp" / "ptx").mkdir(parents=True)
  build_k = cast(
    "fill.Build", types.SimpleNamespace(key="k", features=None, flags=None, supersampled=False, arch=89)
  )  # the fields `_compile` reads
  groups: dict[str, tuple[prebuilt.ModuleRecord, ...]] = {"fast": (prebuilt.ModuleRecord.of(_RECORD),)}
  assert fill._compile(build_k, [fill.Unit("k", False, groups)], str(b), None)[2] == "seeded"
  assert fill._take(fill.seed_bundle(str(b)), "stale", root / "raywarp" / "ptx" / "stale.ptx") is None
  cache = root / "warp" / wp.config.version
  assert fill._take_modules(fill.seed_bundle(str(b)), cache, {"wp_mod_abc"}, 120) == 1
  bundle = build.collect(root, 120, tmp_path / "out", workers=1)
  out = build.read_entries(bundle)
  assert list(out) == ["k"] and out["k"]["groups"] == {"fast": [_RECORD]}
  assert lzma.decompress((bundle / "ptx" / str(out["k"]["ptx"])).read_bytes()) == build.normalise(_PTX)
  assert [p.name for p in (bundle / "warp").iterdir()] == ["wp_mod_abc"]
  assert fill._compile(build_k, [fill.Unit("k", False, groups)], str(b), None)[2] == "kept"  # a resumed shard


_FILL_ALL = Path(__file__).resolve().parents[2] / "tools" / "prebuilt" / "fill_all.sh"


def test_the_fill_watch_rides_out_a_failed_job_list(tmp_path: Path) -> None:
  """`fill_all.sh watch`: a failed `tools.hf ps` (a Hub hiccup) is retried; under `set -e` it ended
  the watch, and with it every relaunch. A job whose run exited 0 is then done."""
  calls = tmp_path / "calls"
  stub = tmp_path / "py"
  # The local interpreter of `tools.hf` (`PY`): `PY -m tools.hf <command> ...`.
  stub.write_text(
    "#!/usr/bin/env bash\n"
    f'echo "$3" >> {calls}\n'
    "case $3 in\n"
    f'  ps) [ "$(grep -c ^ps {calls})" -gt 1 ] || exit 2 ;;\n'
    "  status) echo '{\"exit\": 0}' ;;\n"
    "esac\n"
  )
  stub.chmod(0o755)
  state = tmp_path / "state"
  state.write_text("day 2026-11-10\np0 2026-11-10-t-p0\n")
  env = {**os.environ, "PY": str(stub), "STATE": str(state), "WATCH_EVERY": "0", "SEED": "", "HF_CLI": "true"}
  done = subprocess.run(
    ["bash", str(_FILL_ALL), "watch", "t"], env=env, capture_output=True, text=True, timeout=60, check=False
  )
  assert done.returncode == 0, done.stdout + done.stderr
  assert "job list failed" in done.stdout and "p0 ok" in done.stdout
  assert (tmp_path / "state.done").read_text() == "p0\n"


def test_interactively_a_job_out_of_relaunches_leaves_the_others_running(tmp_path: Path) -> None:
  """FAIL_FAST=0 (no CI): a job out of relaunches is reported, nothing is killed, the watch follows
  the others to their end and then fails (the operator fixes that job; the waiting ones resume)."""
  calls = tmp_path / "calls"
  stub = tmp_path / "py"
  stub.write_text(
    "#!/usr/bin/env bash\n"
    f'echo "$3 $4" >> {calls}\n'
    "case $3 in\n"
    f'  ps) [ "$(grep -c ^ps {calls})" -gt 2 ] || echo "x 2026-11-10-t-p1" ;;\n'
    '  status) [ "$4" = 2026-11-10-t-p1 ] && echo \'{"exit": 0}\' || echo \'{"exit": 1}\' ;;\n'
    "esac\n"
  )
  stub.chmod(0o755)
  state = tmp_path / "state"
  state.write_text("day 2026-11-10\np0 2026-11-10-t-p0-r2\np1 2026-11-10-t-p1\n")
  (tmp_path / "state.tries").write_text("p0 a\np0 b\n")
  env = {k: v for k, v in os.environ.items() if k != "CI"}
  env |= {"PY": str(stub), "STATE": str(state), "WATCH_EVERY": "0", "SEED": "", "HF_CLI": "true"}
  done = subprocess.run(
    ["bash", str(_FILL_ALL), "watch", "t", "p0", "p1"], env=env, capture_output=True, text=True, timeout=60, check=False
  )
  assert done.returncode == 1 and "p0: 2026-11-10-t-p0-r2 failed after 2 relaunches" in done.stdout
  assert "p1 ok" in done.stdout and "FILL FAILED" not in done.stdout and "kill" not in calls.read_text()


def test_the_refill_job_commands_hand_over_archives(tmp_path: Path) -> None:
  """`fill_all.sh spec`: every job waits in place for its inputs and hands over one archive (no
  `cp -r` of a bundle, no seed directory read in place); no compiler cache is restored; the pack
  checks the release driver and packs the default and fast recipes' own census modules. The other
  architectures: tests/tooling/test_fill_all_archs.py."""
  state = tmp_path / "state"
  state.write_text("day 2026-11-11\n")
  scenes = tmp_path / "scenes.txt"
  scenes.write_text("anymal_c primitives bistro\n")
  env = {
    **os.environ,
    "STATE": str(state),
    "SCENES_FILE": str(scenes),
    "ARCHS": "120",
    "PTX": "2",
    "WARP": "1",
    "CENSUS_JOBS": "1",
    "SEED": "",
    "HF_CLI": "true",  # an empty build cache
    "CACHE": "/cache",
  }

  def spec(job: str) -> tuple[str, str]:
    out = subprocess.run(
      ["bash", str(_FILL_ALL), "spec", "t", job], env=env, capture_output=True, text=True, timeout=60, check=True
    )
    head, command = out.stdout.split("\n", 1)
    return head, command

  base = "/data/runs/2026-11-11-t"
  assert (
    spec("census-heavy")[0].startswith("cpu-performance") and "tools.prebuilt.word_reach --scenes $s" in spec("census-heavy")[1]
  )
  assert "for s in anymal_c primitives;" in spec("census-0")[1]
  # A public checkout (no development tree: no benchmark registry) takes the feature-space scenes
  # only.
  scenes.write_text("anymal_c fs_point1_plain_none_mesh_n bistro\n")
  public = subprocess.run(
    ["bash", str(_FILL_ALL), "spec", "t", "census-0"],
    env=env,
    cwd=tmp_path,
    capture_output=True,
    text=True,
    timeout=60,
    check=True,
  )
  assert "for s in fs_point1_plain_none_mesh_n;" in public.stdout
  scenes.write_text("anymal_c primitives bistro\n")
  head, p0 = spec("sm120-p0")
  assert head == "cpu-upgrade 75m" and "/cache/sm120/list/" in p0 and f"{base}-sm120-final-p0.tar.partial" in p0
  assert "--cache /cache" in p0  # the build cache: taken from, kept in
  assert "cp -r" not in p0 and "--workers 4 --shard /tmp/lists/ptx-0.json" in p0
  assert "lens" not in p0  # lens-camera PTX is not filled (user, 2026-11-11): first use / warm
  env["FILL_ARGS"] = "--exclude-x"  # the fill's exclusions reach the listing and every shard
  assert spec("sm120-p0")[1].count("--exclude-x") == 1 and "--exclude-x --out /tmp/list" in spec("sm120-list-0")[1]
  merge = spec("sm120-merge")[1]
  assert "-ge 3 ]" in merge and "--census /tmp/census/sm120/*.json" in merge and f"{base}-sm120-merged-ptx.tar" in merge
  assert "plan-pack" in merge and "--keep /cache/sm120/optix/580.178.04" in merge
  head, pack = spec("sm120-pack-0")
  assert head == "rtx-pro-6000 35m" and "580.178.04" in pack and "--shard 0/9" in pack
  assert "--groups quality:own,fast+default:own" in pack
  assert "--workers 32" in pack and "--deadline $((HF_DEADLINE - 600))" in pack and "--keep /cache/sm120/optix/$drv" in pack
  assert "--seed" not in pack
  env |= {"SEED": "c0@r0", "SEED_GUARD": ""}  # the published pack of bundle c0 (driver 580.178.04) seeds every pack
  seeded = "--seed https://huggingface.co/buckets/blanchon/raywarp-kernels/resolve/sm120/c0/optix/580.178.04 "
  assert all(seeded in spec(f"sm120-pack-{i}")[1] for i in (0, 8))
  release = spec("sm120-release")[1]
  assert "merge-packs" in release and "kernels layout /tmp/b/raywarp_kernels/sm120 /tmp/layout" in release


def test_parts_and_shards_are_checked_at_the_command_line() -> None:
  """`fill list --part` and `build.py pack --shard` take I/N with 0 <= I < N (an index past the
  count listed or packed nothing, silently)."""
  import argparse

  from tools.prebuilt.fill import fraction

  assert fraction("3/16") == (3, 16)
  for bad in ("16/16", "3", "a/b", "-1/4"):
    with pytest.raises(argparse.ArgumentTypeError):
      fraction(bad)


def test_rekey_moves_entries_to_the_new_keys_without_compiling(tmp_path: Path) -> None:
  """`rekey` writes a bundle again after a change of the PTX key's parts and of a host-only
  kernel-code unit: an entry generated from the old tree whose covered changes are only the
  declared ones moves to its new key with this tree's digest (fresh, its PTX file kept, read by
  key); an entry of other code, one covering another change and one no build reaches are dropped
  and reported."""
  import dataclasses

  from fixtures.optix_codegen import preset_features, scene_flags

  import raywarp as rw
  from raywarp.core.trace.optix.device import programs
  from tools.prebuilt import build as bundles
  from tools.prebuilt import rekey
  from tools.prebuilt.fill import Build

  f = dataclasses.replace(preset_features(rw.RenderSettings()), scene_free=True)
  build = Build(key="", features=f, flags=scene_flags(), supersampled=False, registers=(128, 0), arch=89)
  old = {"warp": (("warp-lang", wp.config.version), ("mujoco-warp", "3.17.0"))}
  keys = rekey.key_map([build], old)
  render = programs.ptx_key(f, build.flags, supersampled=False, arch=89)
  (old_render,) = [k for k, v in keys.items() if v == render]
  assert old_render != render
  # The fill's tree: the key's host function and another function differed.
  now = kernel_code.units()
  host = "raywarp/core/trace/optix/device/programs.py:ptx_key"
  other = next(u for u in sorted(now) if kernel_code.function(u) and u != host)
  since = {**now, host: "old", other: "old"}

  def entry(reach: str, code: str = "") -> dict[str, object]:
    """An index entry of the PTX file, reach `reach`, generated from the fill's tree (or `code`)."""
    reached = frozenset({reach})
    return {**_entry(f"c0{prebuilt.PTX_SUFFIX}", code=code or kernel_code.digest(reached, since)), "reach": [reach]}

  index = {old_render: entry(host), "stale": entry(host, "other code"), "changed": entry(other), "unmapped": entry(host)}
  touched = {**keys, "stale": "new-stale", "changed": "new-changed"}
  b = tmp_path / "in" / "kernels" / "sm120"
  (b / "ptx").mkdir(parents=True)
  (b / "ptx" / f"c0{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(_PTX))
  bundles.write_entries(b, index)
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="d0").write(b / prebuilt.MANIFEST)

  target, report = rekey.rekey(b, tmp_path / "out", touched, since, frozenset({host}))
  assert report.kept == 1 and report.stale == ["stale"] and report.unmapped == ["unmapped"]
  assert report.changed == {"changed": [other]}
  bundle = prebuilt.Bundle(target, prebuilt.Manifest.read(target / prebuilt.MANIFEST))
  entry = bundle.entry(render)
  assert entry is not None and entry.fresh() and bundle.entry(old_render) is None
  assert lzma.decompress((target / "ptx" / entry.ptx).read_bytes()) == _PTX
  assert bundle.manifest.content != "d0"
