"""The module pack's builder (`tools/prebuilt/build.py pack`), OptiX faked: one module file per
module (its own context and disk cache: exactly its rows), a terminated pack resumed from its keep
directory (only the rest made), shards merged, the census order and the deadline."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import lzma
import shutil
import sqlite3
import threading
import types
from pathlib import Path

import pytest
import warp as wp

from raywarp.core.build.kernel_cache import kernel_code, prebuilt
from raywarp.core.trace.optix import context, modules
from tools.prebuilt import build, fill

_SCHEMA = (
  "CREATE TABLE IF NOT EXISTS cache_info (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, optix_version VARCHAR(32), "
  "driver_version VARCHAR(32), size INTEGER, timestamp INTEGER)",
  "CREATE TABLE IF NOT EXISTS cache_data (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, value BLOB)",
)
_KEYS = [f"k{i}" for i in range(6)]


def _record(bits: int, *, curves: bool = False) -> prebuilt.ModuleRecord:
  """A module record binding class bits `bits`."""
  return prebuilt.ModuleRecord(
    has_curves=curves, has_custom=False, opacity_micromaps=False, max_registers=80, bound=((8, f"{bits:08x}"),)
  )


def _bundle(root: Path) -> Path:
  """A kernels bundle of six PTX keys, one fast record each."""
  b = root / "sm120"
  (b / "ptx").mkdir(parents=True)
  for i, key in enumerate(_KEYS):
    (b / "ptx" / f"{key}{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(key.encode()))
    entry = prebuilt.Entry(
      ptx=f"{key}{prebuilt.PTX_SUFFIX}", code=kernel_code.version(), reach=kernel_code.ALL, groups={"fast": (_record(i),)}
    )
    (b / prebuilt.entry_path(key)).write_text(json.dumps(entry.json()))
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="d0").write(b / prebuilt.MANIFEST)
  return b


def _jobs(b: Path) -> list[tuple[str, prebuilt.ModuleRecord]]:
  """Every module of bundle `b`, in key order."""
  return build._packed(prebuilt.Bundle(b, prebuilt.Manifest.read(b / prebuilt.MANIFEST)), ())


def _keys(db: Path) -> list[str]:
  """The keys of an OptiX disk cache (a module file)."""
  with sqlite3.connect(db) as c:
    return sorted(r[0] for r in c.execute("SELECT key FROM cache_data"))


def _files(pack: Path) -> list[str]:
  """The module names of a pack directory."""
  return sorted(f.name.removesuffix(prebuilt.MODULE_SUFFIX) for f in (pack / prebuilt.MODULE_DIR).glob("*"))


@dataclasses.dataclass
class FakeContext:
  """An OptiX device context whose module creation adds the module's row (and the curve module's)
  to the disk cache at its location; closed explicitly."""

  location: str = ""
  closed: bool = False

  def set_cache_location(self, location: str) -> None:
    self.location = location

  def set_cache_sizes(self, _low: int, _high: int) -> None:
    pass

  def close(self) -> None:
    self.closed = True

  def add(self, key: str, driver: str = "580") -> None:
    """One row of the cache at the location."""
    db = Path(self.location) / prebuilt.OPTIX_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as c:
      for sql in _SCHEMA:
        c.execute(sql)
      c.execute("INSERT INTO cache_info VALUES (?, '9.0', ?, 1, 0)", (key, driver))
      c.execute("INSERT INTO cache_data VALUES (?, ?)", (key, key.encode()))


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[FakeContext]:
  """The pack worker's OptiX faked (each `DeviceContext` a `FakeContext`, `modules.make` adding its
  rows); the contexts made."""
  contexts: list[FakeContext] = []

  def new_context(_cuda: int, _arch: int) -> FakeContext:
    contexts.append(FakeContext())
    return contexts[-1]

  def make(ctx: FakeContext, ptx: bytes, record: prebuilt.ModuleRecord) -> types.SimpleNamespace:
    if record.has_curves:
      ctx.add("curves")
    ctx.add(ptx.decode())
    return types.SimpleNamespace(destroy=lambda: None)

  monkeypatch.setattr(context, "DeviceContext", new_context)
  monkeypatch.setattr(modules, "make", make)
  monkeypatch.setattr(build.wp, "init", lambda: None)
  monkeypatch.setattr(build.wp, "get_device", lambda _name: types.SimpleNamespace(context=1, arch=120))
  return contexts


def test_a_module_file_holds_exactly_its_module_s_rows(tmp_path: Path, made: list[FakeContext]) -> None:
  """`_module_file`: each module on a context of its own, closed, whose cache is a fresh directory:
  its file holds its row and its curve module's, nothing of the module made before it; named by
  `modules.digest` (what `modules.create` looks up)."""
  b = _bundle(tmp_path / "bundle")
  target = tmp_path / "pack"
  (target / prebuilt.MODULE_DIR).mkdir(parents=True)
  first, _ = build._module_file(b, f"k0{prebuilt.PTX_SUFFIX}", _record(0).json(), target, tmp_path / "s")
  second, _ = build._module_file(b, f"k1{prebuilt.PTX_SUFFIX}", _record(1, curves=True).json(), target, tmp_path / "s")
  assert first == modules.digest(b"k0", _record(0))
  assert _keys(target / prebuilt.module_path(first)) == ["k0"]
  assert _keys(target / prebuilt.module_path(second)) == ["curves", "k1"]
  assert all(c.closed for c in made) and len(made) == 2
  assert not (tmp_path / "s").exists() or not any((tmp_path / "s").iterdir())


def test_a_module_file_of_two_drivers_or_none_is_refused(tmp_path: Path) -> None:
  """`_fold`: a module file holds one driver's rows; a cache with none (OptiX's cache off) or two
  drivers' is an error."""
  ctx = FakeContext(str(tmp_path / "c"))
  ctx.add("a")
  ctx.add("b", driver="590")
  with pytest.raises(RuntimeError, match="drivers"):
    build._fold(tmp_path / "c" / prebuilt.OPTIX_DB, tmp_path / "m.db")


def _run(b: Path, target: Path, todo: list, **kwargs: object) -> tuple[int, float]:
  """`create_kept` on a thread pool of 2 (the worker processes' stand-in)."""
  (target / prebuilt.MODULE_DIR).mkdir(parents=True, exist_ok=True)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
    return build.create_kept(pool, b, todo, target, workers=2, scratch=target.parent / "scratch", **kwargs)  # pyrefly: ignore[bad-argument-type]


def test_a_terminated_pack_resumes_from_its_keep(
  tmp_path: Path, made: list[FakeContext], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Terminated after 3 modules: the keep holds the files made; the relaunch makes only the rest,
  and the pack (and the keep) hold all six."""
  b, keep = _bundle(tmp_path / "bundle"), tmp_path / "keep"
  todo = _jobs(b)
  real, calls = build._module_file, []
  lock = threading.Lock()

  def failing(*args: object) -> tuple[str, float]:
    with lock:
      calls.append(args[1])
      if len(calls) > 3:
        raise RuntimeError("terminated")
    return real(*args)  # pyrefly: ignore[bad-argument-type]

  monkeypatch.setattr(build, "_module_file", failing)
  with pytest.raises(RuntimeError, match="terminated"):
    _run(b, tmp_path / "out1" / "p", todo, keep=keep)
  kept = _files(keep)
  assert 1 <= len(kept) <= 3
  monkeypatch.setattr(build, "_module_file", real)
  before = len(made)
  done, _ = _run(b, tmp_path / "out2" / "p", todo, keep=keep)
  assert done == 6 and len(made) - before == 6 - len(kept)
  names = sorted(build._names(b, todo).values())
  assert _files(tmp_path / "out2" / "p") == names == _files(keep)


def test_shards_partition_the_modules_and_merge(tmp_path: Path, made: list[FakeContext]) -> None:
  """Two shards (`--shard I/2`: every second module): each module in exactly one shard's pack;
  `merge-packs` unites the files under one sealed manifest."""
  b = _bundle(tmp_path / "bundle")
  todo = _jobs(b)
  packs = []
  for i in range(2):
    target = tmp_path / f"out{i}" / build.PACK / "sm120-drv580"
    _run(b, target, todo[i::2], keep=None)
    prebuilt.Manifest(arch=120, warp=wp.config.version, content="", driver="580").write(target / prebuilt.MANIFEST)
    packs.append(target)
  assert len(_files(packs[0])) == len(_files(packs[1])) == 3 and not set(_files(packs[0])) & set(_files(packs[1]))
  merged = build.merge_packs(packs, tmp_path / "merged")
  assert _files(merged) == sorted(_files(packs[0]) + _files(packs[1]))
  assert prebuilt.Manifest.read(merged / prebuilt.MANIFEST).driver == "580"
  assert len(made) == 6


def test_a_pack_takes_the_census_s_own_words_first_and_stops_at_its_deadline(
  tmp_path: Path, made: list[FakeContext], monkeypatch: pytest.MonkeyPatch
) -> None:
  """`pack --select C --deadline T`: the census's own words first, then their single-change
  neighbours, each part most used first; past the deadline no module starts and the pack holds
  what is done (here the first two: the own words)."""
  b = _bundle(tmp_path / "bundle")
  unit = {i: {"key": key, "group": "fast", "module": _record(i).json()} for i, key in enumerate(_KEYS)}
  rows = [
    {"scene": "a", "recipe": "fast", "own": 1, "units": [unit[5], unit[0], unit[1]]},
    {"scene": "b", "recipe": "fast", "own": 1, "units": [unit[4], unit[1]]},
    {"scene": "c", "recipe": "quality", "own": 1, "units": [unit[3]]},
  ]
  (tmp_path / "c.json").write_text(json.dumps(rows))
  order = build._packed(
    prebuilt.Bundle(b, prebuilt.Manifest.read(b / prebuilt.MANIFEST)),
    (tmp_path / "c.json",),
    fill.groups_of("fast:own,fast:near"),
  )
  assert [p.split(".")[0] for p, _ in order] == [_KEYS[4], _KEYS[5], _KEYS[1], _KEYS[0]]
  clock = iter([0.0, 0.0, 0.0, 0.0])  # the first loop starts two modules in time, the next none
  monkeypatch.setattr(build.time, "time", lambda: next(clock, 100.0))
  done, _ = _run(b, tmp_path / "out" / "p", order, keep=None, deadline=50.0)
  assert done == 2 and _files(tmp_path / "out" / "p") == sorted(build._names(b, order[:2]).values())


def _published(b: Path, root: Path, todo: list, extra: list) -> str:
  """A published pack (`tools/release/kernels.py` layout, driver 580) of the modules of `todo` and
  `extra` (one another bundle's: no longer wanted) made from bundle `b`; its URL."""
  from tools.release import kernels

  pack = root / "made" / build.PACK / "sm120-drv580"
  _run(b, pack, [*todo, *extra], keep=None)
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="", driver="580").write(pack / prebuilt.MANIFEST)
  kernels._lay(pack, root / "remote", 2)
  return (root / "remote").as_uri()


def test_a_pack_fetches_the_seed_s_modules_and_makes_only_the_rest(tmp_path: Path, made: list[FakeContext]) -> None:
  """`pack --seed URL`: the wanted modules the published pack holds are fetched (byte for byte its
  files), only the others made; a published module no longer wanted is not carried over."""
  b = _bundle(tmp_path / "bundle")
  todo = _jobs(b)
  stale = (f"k0{prebuilt.PTX_SUFFIX}", _record(7))  # a record the census no longer reaches
  url = _published(b, tmp_path / "pub", todo[:4], [stale])
  before = len(made)
  seed = build.open_seed(url, tmp_path / "seed", 120, "580")
  target = tmp_path / "out" / build.PACK / "sm120-drv580"
  done, _ = _run(b, target, todo, keep=None, seed=seed)
  assert done == 6 and len(made) - before == 2
  names = build._names(b, todo)
  assert _files(target) == sorted(names.values())
  published = tmp_path / "pub" / "made" / build.PACK / "sm120-drv580"
  for job in todo[:4]:
    rel = prebuilt.module_path(names[job])
    assert (target / rel).read_bytes() == (published / rel).read_bytes()


def test_a_seed_of_another_driver_is_refused(tmp_path: Path, made: list[FakeContext]) -> None:
  """`open_seed`: a published pack of another driver (or architecture) is no seed."""
  b = _bundle(tmp_path / "bundle")
  url = _published(b, tmp_path / "pub", _jobs(b)[:1], [])
  with pytest.raises(SystemExit, match="driver 580, not sm120 driver 590"):
    build.open_seed(url, tmp_path / "seed", 120, "590")
  assert made


def test_the_runtime_finds_a_fill_s_modules_by_the_pack_s_names(
  tmp_path: Path, made: list[FakeContext], monkeypatch: pytest.MonkeyPatch
) -> None:
  """A fill root of NVRTC-style PTX (Warp's module hash in NVRTC's file-scope prefix, the programs'
  source-hash suffixes) collected, packed (OptiX faked), then read as the runtime does
  (`prebuilt.ptx`, `modules.digest`, `prebuilt.seed_module`): each module the pack made is found
  under the name the runtime computes, and the program by its semantic name; a second fill whose
  only differences are those names gives the same bundle PTX and pack names."""
  from raywarp.core.build.kernel_cache import store as kernel_store
  from raywarp.core.trace.optix import pipeline

  def fill_root(root: Path, module: str, suffix: str) -> Path:
    sym = f"_ZN58_INTERNAL_00000000_27_wp_raywarp_optix_{module}_cu_8d30c1fb2wp9SLICE_ENDE"
    ptx = f"// NVRTC\n.version 9.0\n.global .align 4 .u32 {sym} = 1;\n.visible .entry __raygen__render_{suffix}()\n{{\n}}\n"
    (root / "wpbr" / "ptx").mkdir(parents=True)
    (root / "wpbr" / "ptx" / "k0.ptx").write_text(ptx)
    entry = prebuilt.Entry(ptx="", code=kernel_code.version(), reach=kernel_code.ALL, groups={"fast": (_record(1),)})
    (root / "wpbr" / "ptx" / "k0.json").write_text(json.dumps(entry.json()))
    return build.collect(root, 120, root.parent / f"{root.name}_out", workers=1)

  bundles = [fill_root(tmp_path / "a", "bdefb90", "f86e48f3"), fill_root(tmp_path / "b", "22ae3f2", "0a1b2c3d")]
  names = [build._names(b, _jobs(b)) for b in bundles]
  assert list(names[0].values()) == list(names[1].values())
  store = tmp_path / "store" / "sm120" / "b0"
  shutil.copytree(bundles[0], store)
  pack = store / "optix" / "575.1"
  _run(store, pack, _jobs(store), keep=None)
  monkeypatch.setenv(kernel_store.STORE_ENV, str(tmp_path / "store"))
  monkeypatch.setenv(prebuilt.BUNDLE_ENV, "b0")
  monkeypatch.setattr(prebuilt, "driver_version", lambda: "575.1")
  for name, empty in (("_BUNDLES", {}), ("_PACKS", {})):
    monkeypatch.setattr(prebuilt, name, empty)
  code = prebuilt.ptx(120, "k0", expected=True)
  assert code is not None and pipeline.entry_names(code, ["__raygen__render"]) == {"__raygen__render": "__raygen__render"}
  name = modules.digest(code, _record(1))
  assert [name] == _files(pack) == list(names[0].values())
  db = tmp_path / "optix" / prebuilt.OPTIX_DB
  db.parent.mkdir()
  assert prebuilt.seed_module(120, name, db)
