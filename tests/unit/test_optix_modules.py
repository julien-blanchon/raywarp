"""The OptiX module service (`core/trace/optix/modules.py`) on the CPU, OptiX faked: one creation
per module across threads, a packed module's rows merged and OptiX's cache reloaded first, a loud
warning for an expected build's module outside the bundle, OptiX's compile tasks stopped between
tasks (a disk-cache hit runs one), and the class bits as an opaque key of the launch parameters'
bound words."""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import lzma
import sqlite3
import threading
import time
import types
from pathlib import Path

import pytest
import warp as wp

from raywarp.core.build.kernel_cache import kernel_code, prebuilt
from raywarp.core.build.kernel_cache import store as kernel_store
from raywarp.core.trace.optix import api, context, modules
from raywarp.core.trace.optix.device import params

_PTX = b"// ptx\n.entry e()\n{\n}\n"


def _record(registers: int, bits: int = 0, *, curves: bool = False) -> prebuilt.ModuleRecord:
  """A module record of register cap `registers` binding class bits `bits` (at offset 8)."""
  return prebuilt.ModuleRecord(
    has_curves=curves, has_custom=False, opacity_micromaps=False, max_registers=registers, bound=((8, f"{bits:08x}"),)
  )


@dataclasses.dataclass
class FakeContext:
  """An OptiX context whose module compiles take `seconds`, recording each (PTX, registers) and
  the most compiles at once."""

  seconds: float = 0.0
  made: list[tuple[bytes, int]] = dataclasses.field(default_factory=list)
  curves: list[int] = dataclasses.field(default_factory=list)  # the built-in curve modules' register caps
  running: int = 0
  most: int = 0
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
  arch: int = 120
  cache_db: Path | None = None  # OptiX's disk cache (None: no merge)
  reloads: int = 0

  def reload_cache(self) -> None:
    self.reloads += 1

  def module(self, mco: object, _pco: object, ptx: bytes, stop: threading.Event | None = None) -> types.SimpleNamespace:
    with self.lock:
      self.running += 1
      self.most = max(self.most, self.running)
    time.sleep(self.seconds)
    with self.lock:
      self.running -= 1
      self.made.append((ptx, mco.maxRegisterCount))  # pyrefly: ignore[missing-attribute]
    if stop is not None and stop.is_set():
      raise RuntimeError("stopped")
    return types.SimpleNamespace(destroy=lambda: None)

  def builtin_is_module(self, mco: object, _pco: object, options: object) -> int:
    assert options.builtinISModuleType == api.PRIMITIVE_TYPE_ROUND_LINEAR  # pyrefly: ignore[missing-attribute]
    with self.lock:
      self.curves.append(mco.maxRegisterCount)  # pyrefly: ignore[missing-attribute]
    return 1


def _install(store: Path, groups: dict[str, dict[str, list[prebuilt.ModuleRecord]]]) -> None:
  """A complete bundle in the kernel store `store` (sm120/d0, pinned): per PTX key (its PTX the
  key's bytes) its records per tier."""
  b = store / "sm120" / "d0"
  (b / "ptx").mkdir(parents=True)
  index = {}
  for key, tiers in groups.items():
    (b / "ptx" / f"{key}{prebuilt.PTX_SUFFIX}").write_bytes(lzma.compress(key.encode()))
    entry = prebuilt.Entry(
      ptx=f"{key}{prebuilt.PTX_SUFFIX}",
      code=kernel_code.version(),
      reach=kernel_code.ALL,
      groups={g: tuple(rs) for g, rs in tiers.items()},
    )
    index[key] = entry.json()
  for key, raw in index.items():
    (b / prebuilt.entry_path(key)).write_text(json.dumps(raw))
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="d0").write(b / prebuilt.MANIFEST)
  listing = kernel_store.Listing.of(b)
  (b / kernel_store.FILES).write_bytes(listing)
  pins = {"bundles": {"120": {"content": "d0", "files": hashlib.sha256(listing).hexdigest()}}}
  kernel_store.PIN.write_text(json.dumps(pins))


@pytest.fixture
def user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """Empty user caches and kernel store (`tmp_path`/store, no remote), no pins, nothing considered
  yet, driver 575.1."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path / "user"))
  monkeypatch.setenv(kernel_store.STORE_ENV, str(tmp_path / "store"))
  monkeypatch.setenv(kernel_store.URL_ENV, (tmp_path / "remote").as_uri())
  monkeypatch.delenv(prebuilt.BUNDLE_ENV, raising=False)
  monkeypatch.setattr(kernel_store, "PIN", tmp_path / "kernels.json")
  monkeypatch.setattr(prebuilt, "_BUNDLES", {})
  monkeypatch.setattr(prebuilt, "_PACKS", {})
  monkeypatch.setattr(prebuilt, "driver_version", lambda: "575.1")
  monkeypatch.setattr(modules, "_WARNED", set())
  return tmp_path / "user"


def test_a_module_is_created_once_at_a_time_across_threads(user: Path) -> None:
  """Threads needing the same module wait for one creation (OptiX's disk cache then serves the
  others), a lock file per module beside raywarp's cache; other modules create in parallel."""
  ctx = FakeContext(seconds=0.2)
  same = [threading.Thread(target=modules.create, args=(ctx, _PTX, _record(96))) for _ in range(3)]
  for t in same:
    t.start()
  for t in same:
    t.join()
  assert ctx.most == 1 and ctx.made == [(_PTX, 96)] * 3
  assert len(list((user / "prebuilt" / "modules").glob("*.lock"))) == 1
  ctx = FakeContext(seconds=0.2)
  other = [threading.Thread(target=modules.create, args=(ctx, _PTX, _record(r))) for r in (96, 128)]
  for t in other:
    t.start()
  for t in other:
    t.join()
  assert ctx.most == 2


def test_a_module_with_curves_creates_the_built_in_curve_module_too(tmp_path: Path, user: Path) -> None:
  """A module whose record traverses curves (spheres, capsules) creates OptiX's built-in curve
  intersection module with its options (OptiX's disk cache: the pack and the pre-creation hold
  it, a pipeline finds it); one without curves none."""
  ctx = FakeContext()
  modules.create(ctx, _PTX, _record(96))  # pyrefly: ignore[bad-argument-type]
  assert ctx.curves == []
  modules.create(ctx, _PTX, _record(80, curves=True))  # pyrefly: ignore[bad-argument-type]
  assert ctx.curves == [80]


def test_a_packed_module_s_rows_are_merged_and_the_cache_reloaded_first(tmp_path: Path, user: Path) -> None:
  """`create`: a module the running driver's pack holds (by `digest`) has its rows merged into the
  context's OptiX cache and the cache reloaded before OptiX is asked for it (then a hit); one the
  pack lacks is OptiX's alone, nothing reloaded; without a cache location nothing is merged."""
  _install(tmp_path / "store", {"k": {"fast": [_record(96)]}})
  pack = tmp_path / "store" / "sm120" / "d0" / "optix" / "575.1" / prebuilt.MODULE_DIR
  pack.mkdir(parents=True)
  name = modules.digest(_PTX, _record(96))
  prebuilt.merge_optix(_db(tmp_path / "rows.db", "row"), pack / f"{name}{prebuilt.MODULE_SUFFIX}")
  (pack.parent / kernel_store.FILES).write_bytes(kernel_store.Listing.of(pack.parent))
  ctx = FakeContext(cache_db=user / "optix" / prebuilt.OPTIX_DB)
  modules.create(ctx, _PTX, _record(96))  # pyrefly: ignore[bad-argument-type]
  assert ctx.reloads == 1 and ctx.made == [(_PTX, 96)]
  with sqlite3.connect(user / "optix" / prebuilt.OPTIX_DB) as db:
    assert [r[0] for r in db.execute("SELECT key FROM cache_data")] == ["row"]
  modules.create(ctx, _PTX, _record(128))  # pyrefly: ignore[bad-argument-type]
  assert ctx.reloads == 1
  bare = FakeContext()
  modules.create(bare, _PTX, _record(96))  # pyrefly: ignore[bad-argument-type]
  assert bare.reloads == 0


def _db(path: Path, key: str) -> Path:
  """An OptiX disk cache holding row `key`."""
  with sqlite3.connect(path) as db:
    db.execute(
      "CREATE TABLE cache_info (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, optix_version VARCHAR(32), "
      "driver_version VARCHAR(32), size INTEGER, timestamp INTEGER)"
    )
    db.execute("CREATE TABLE cache_data (key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, value BLOB)")
    db.execute("INSERT INTO cache_info VALUES (?, '9.0', '575.1', 1, 0)", (key,))
    db.execute("INSERT INTO cache_data VALUES (?, ?)", (key, b"v"))
  return path


def test_a_failed_creation_says_so() -> None:
  """OptiX rejecting a module raises with its cause."""

  class Rejecting:
    cache_db: Path | None = None
    arch: int = 120

    def module(self, *_args: object) -> object:
      raise ValueError("bad PTX")

  with pytest.raises(RuntimeError, match="could not create the program module: bad PTX"):
    modules.create(Rejecting(), _PTX, _record(96))  # pyrefly: ignore[bad-argument-type]


def test_an_expected_module_outside_the_bundle_warns_once(tmp_path: Path, user: Path, recwarn: pytest.WarningsRecorder) -> None:
  """A prebuilt build's module its entry records is silent; one it does not (a class-bits value no
  census scene uses) warns once, naming the bound words that differ; a scene's own build, or a key
  the bundle lacks (`prebuilt.ptx` warned), is silent."""
  _install(tmp_path / "store", {"k": {"fast": [_record(96, 0)]}})
  modules.check(120, "k", _record(96, 0), expected=True)
  modules.check(120, "k", _record(96, 5), expected=False)
  modules.check(120, "other", _record(96, 5), expected=True)
  assert not recwarn.list
  for _ in range(2):
    modules.check(120, "k", _record(96, 5), expected=True)
  (w,) = recwarn.list
  assert "not prebuilt ([(8, '00000005')]" in str(w.message) and "no NVRTC" in str(w.message)


def test_stopping_an_optix_compile_runs_no_new_task(monkeypatch: pytest.MonkeyPatch) -> None:
  """`context._run_tasks`: OptiX's compile tasks run until none is left; once `stop` is set the
  running ones end and the tasks they spawned are not run (the module stays incomplete)."""
  ran: list[int] = []
  stop, armed = threading.Event(), threading.Event()

  def execute(task: int, _room: int) -> list[int]:
    ran.append(task)
    if task == 3 and armed.is_set():
      stop.set()
    return [task * 2, task * 2 + 1] if task < 8 else []

  monkeypatch.setattr(context, "_execute", execute)
  assert context._run_tasks(1, None) == 15  # the tasks executed (a disk-cache hit: 1)
  assert sorted(ran) == list(range(1, 16))
  ran.clear()
  armed.set()
  context._run_tasks(1, stop)
  assert 3 in ran and len(ran) < 15 and not {6, 7} & set(ran)


def test_the_build_word_is_an_opaque_bound_word() -> None:
  """`params.set_words`: the launch parameters' bound words with word `word` placed as a frame
  places it: only the word's copies differ between two values, each holding its bits (the
  geometry copy every bit, the others all but the geometry's); a geometry-only layout has the
  geometry copy alone."""
  from raywarp.core.kernels.abi import toggles

  def words(struct: type, word: int) -> params.BoundWords:
    p = struct()
    params.set_words(p, word)
    return params.bound_words(p)

  high = toggles.MAT_REFLECTION
  word = toggles.CLASS_IES | toggles.GEO_SHARED | toggles.high(high)
  struct = params.params_type(params.ParamsLayout(mesh=True))
  at = params.word_offsets(struct)
  copies = (("geo",), ("geo", "cut"), ("lit",), ("env",))
  # The material carrier holds the first half (its second: the geometry's, `GeometryFields.bits2`).
  assert set(at) == {(c, half) for c in copies for half in ("bits", "bits2")} - {(("geo", "cut"), "bits2")}
  changed = dict(sorted(set(words(struct, word)) - set(words(struct, 0))))
  expected = {at[("geo",), "bits"]: toggles.CLASS_IES | toggles.GEO_SHARED}
  expected |= {at[c, "bits"]: toggles.CLASS_IES for c in copies[1:]}
  expected |= {at[c, "bits2"]: high for c in copies if c != ("geo", "cut")}
  assert changed == {o: bytes(ctypes.c_int32(v)) for o, v in expected.items()}
  assert params.geometry_word(words(struct, word), struct) == word
  plain = params.params_type(params.ParamsLayout(shaded=False))
  assert set(params.word_offsets(plain)) == {(("geo",), "bits"), (("geo",), "bits2")}


def test_every_launch_binds_both_halves_of_the_render_s_word() -> None:
  """The light-grid visibility launch (and the GI / bake launches, the same `toggles.load`) set
  every copy from the geometry copy's both halves: a copy set from `geo.bits` alone bound the
  second half 0 (every per-field bit off: the GI and grid modules shaded without the scene's maps
  and materials)."""
  from raywarp.core.kernels.abi import toggles
  from raywarp.core.kernels.abi.structs import OutputArrays
  from raywarp.core.shade.lights.punctual import Lights
  from raywarp.core.trace.optix import launch

  word = toggles.CLASS_MATERIALS | toggles.GEO_SHARED | toggles.high(toggles.MAT_REFLECTION | toggles.MAP_NORMAL)
  struct = params.params_type(params.ParamsLayout(mesh=True, grid=True))
  p = struct()
  geo = p.geo
  toggles.store(geo, word)
  launch.set_grid(p, geo, Lights(), OutputArrays())
  assert params.geometry_word(params.bound_words(p), struct) == word
  for copy in (p.lit, p.env):
    assert toggles.load(copy) == word & ~toggles.GEO_ALL
