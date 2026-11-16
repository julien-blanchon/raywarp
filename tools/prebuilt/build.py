"""The prebuilt kernels of one GPU architecture (`raywarp/core/build/kernel_cache/prebuilt.py`):
the kernels bundle, the OptiX module pack of the NVIDIA driver that filled it.

1. `fill` (tools/prebuilt/fill.py, CPU): the PTX of every code split under a root R, with the
   kernel code each was generated from and the module records the census reaches.
2. `collect --root R --arch 120 --workers 8 --out dist/`: `R/warp` (Warp's kernel cache) and
   `R/wpbr/ptx` (the OptiX programs' PTX, compressed and stored by content, each build's entry
   beside them: `prebuilt.entry_path`; only those of this kernel code, `prebuilt.Entry.fresh`: a
   root holds only its fill's keys and modules) into `dist/raywarp_kernels/sm<arch>/`, with its
   manifest. The shards of a sharded fill (`fill_all.sh`) are each collected, then `merge --bundle
   B1 --bundle B2 ... --out dist/` makes one bundle.
3. `pack --bundle dist/raywarp_kernels/sm120 --workers 8 --out dist/ [--select C ...] [--shard I/N]
   [--keep K] [--seed URL]`
   (a GPU of that architecture, the driver the pack is for): every module the bundle index records
   (or those the census files `--select` reach), created by OptiX alone (no NVRTC), `workers` at a
   time, each into a fresh OptiX disk cache of its own: one module's rows per file,
   `dist/raywarp_optix_modules/sm<arch>-drv<driver>/m/<module>.db`, with its manifest (a first
   frame fetches its own modules' files, not the pack). `--shard I/N` on N GPU jobs (a job holds
   ~one cap of module creation), then `merge-packs --pack P1 --pack P2 ... --out dist/`. `--keep
   K` (a bucket directory per shard): each module file is copied there when made, and a relaunch
   with the same `K` resumes from it. `--seed URL` (a published pack of the same driver,
   `<kernels>/sm120/<content>/optix/<driver>`): each wanted module that pack holds is fetched, not
   made (a module's name covers its PTX and record, `modules.digest`: a refill makes only the
   modules whose PTX or record changed).
4. `rekey --bundle dist/raywarp_kernels/sm120 --out dist2/ --since REV [--old NAME=VALUE ...]
   [--ignore UNIT ...]` (tools/prebuilt/rekey.py, CPU, no compile): the bundle again with its
   entries under the keys and kernel-code digests of this tree, after a change of the PTX key's
   parts or of host code in kernel-code modules since the fill's tree REV.

A release lays the bundle and its packs out for download (`tools/release/kernels.py layout`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import lzma
import multiprocessing
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final, final

import warp as wp

from raywarp.core.build.kernel_cache import prebuilt, store
from tools.prebuilt import fill

# The output directories of a bundle and of a module pack (under `--out`).
NAME: Final = "raywarp_kernels"
PACK: Final = "raywarp_optix_modules"


# A PTX register (`%r12`, `%fd3`, ...) and a label (`$L__BB0_4`): renumbered per function.
_REG: Final = re.compile(r"%(rd|rs|r|fd|f|p|h|hh|b)(\d+)\b")
_LABEL: Final = re.compile(r"\$L__[A-Za-z_]*\d+(?:_\d+)?")
_FUNCTION: Final = (".visible .entry", ".entry", ".func", ".visible .func", ".weak .func")
# Line info (`.loc`, `.file`), dropped; not `.local` (a function's local-memory depot: OptiX refuses
# the module without it, OPTIX_ERROR_INVALID_INPUT).
_LINE_INFO: Final = re.compile(r"\.(loc|file)\s")
# NVRTC's file-scope prefix of internal-linkage symbols: it names the source file Warp wrote,
# `wp_<module>_<Warp module hash>.cu`, and a digest of it (`_ZN58_INTERNAL_00000000_27_wp_raywarp_
# optix_bdefb90_cu_8d30c1fb2wp6volume7CLOSESTE`): one per module, varying from fill to fill.
_INTERNAL: Final = re.compile(r"_ZN\d+_INTERNAL_[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]{8}")
_INTERNAL_AS: Final = "_ZN_INTERNAL_wpbr_"
# A function's local-memory depot (`__local_depot<N>`, numbered by the function's position).
_DEPOT: Final = re.compile(r"__local_depot\d+\b")
# An OptiX program's Warp name suffix (`__raygen__render_<8 hex>`: its source text's hash, comments
# included); the runtime finds a program with or without it (`trace/optix/pipeline.entry_names`).
_PROGRAM: Final = re.compile(
  r"\b(__(?:raygen|miss|closesthit|anyhit|intersection|exception|direct_callable|continuation_callable)__\w*?)_[0-9a-f]{8}\b"
)


def _renumber(body: list[str]) -> list[str]:
  """One function's lines, its registers (per class) and labels numbered by first use."""
  regs: dict[tuple[str, str], int] = {}
  counts: dict[str, int] = {}
  labels: dict[str, int] = {}

  def reg(m: re.Match[str]) -> str:
    k = (m.group(1), m.group(2))
    if k not in regs:
      regs[k] = counts.get(k[0], 0)
      counts[k[0]] = regs[k] + 1
    return f"%{k[0]}{regs[k]}"

  def label(m: re.Match[str]) -> str:
    return f"$L__{labels.setdefault(m.group(0), len(labels))}"

  return [_LABEL.sub(label, _REG.sub(reg, line)) for line in body]


def normalise(ptx: bytes) -> bytes:
  """The same program as PTX `ptx`, as OptiX reads it, in fewer and more shared bytes: no
  comments, blank lines, line info (`.loc`, `.file`) or debug sections, and each function's
  registers and labels numbered by first use (RTX PRO 6000, OptiX 9: accepted, identical images;
  xz -7 %: `internal/benchmarks/lab/ptx_normal_check.py`); the names that vary from fill to fill
  (NVRTC's file-scope prefix, the programs' Warp source-hash suffix, the programs' order and their
  depots' numbers) made fixed, so equal code is equal bytes in every fill and its OptiX modules keep
  their names (`trace/optix/modules.digest`): a module pack seeds the next refill (`pack --seed`).

  Raises:
    ValueError: two programs share a name once their suffixes are dropped.
  """
  text = _PROGRAM.sub(r"\1", _INTERNAL.sub(_INTERNAL_AS, ptx.decode()))
  entries = re.findall(r"\.entry\s+(\w+)", text)
  if len(entries) != len(set(entries)):
    raise ValueError(f"PTX programs share a name without their Warp suffix: {sorted(entries)}")
  head: list[str] = []
  functions: list[list[str]] = []
  tail: list[str] = []  # NVRTC's closing NUL
  body: list[str] | None = None
  debug = False
  for raw in text.splitlines():
    line = raw.split("//", 1)[0].rstrip()
    s = line.strip()
    if s.startswith(".section") and ".debug" in s:
      debug = True
    if debug:
      debug = s != "}"
      continue
    if not s or _LINE_INFO.match(s):
      continue
    if body is None and s.startswith(_FUNCTION) and not s.endswith(";"):
      body = []
    if body is None and s == "\x00":
      tail.append(line)
      continue
    if body is None:
      if functions:  # module-scope text between functions: their order is kept as written
        functions[-1].append(line)
      else:
        head.append(line)
      continue
    body.append(line)
    if line == "}":
      functions.append(_renumber(body))
      body = None
  return ("\n".join(head + _ordered(functions) + (body or []) + tail) + "\n").encode()


def _ordered(functions: list[list[str]]) -> list[str]:
  """The lines of a module's functions (each a list of lines, its header first) in a fixed order:
  the device functions as written, then the programs by name (Warp orders a module's kernels by
  their source hash: a comment moves them), each local-memory depot numbered in that order (NVRTC
  numbers them by position). Unchanged when module-scope text follows a function (an order PTX
  might need)."""
  entries = [f for f in functions if ".entry" in f[0] and f[-1] == "}"]
  if any(f[-1] != "}" for f in functions):
    order = functions
  else:
    order = [f for f in functions if ".entry" not in f[0]] + sorted(entries, key=lambda f: f[0])
  depots: dict[str, str] = {}

  def depot(m: re.Match[str]) -> str:
    return depots.setdefault(m.group(0), f"__local_depot{len(depots)}")

  return [_DEPOT.sub(depot, line) for f in order for line in f]


def _compress(src: Path, dst: Path) -> None:
  """PTX file `src` normalised and xz-compressed into `dst` (a worker process: xz 9e)."""
  if not dst.exists():
    dst.write_bytes(lzma.compress(normalise(src.read_bytes()), preset=9 | lzma.PRESET_EXTREME))


def _content_name(ptx: Path) -> str:
  """A PTX file's name in a bundle: its normalised content's digest (identical programs share one
  file)."""
  return hashlib.blake2b(normalise(ptx.read_bytes()), digest_size=16).hexdigest() + prebuilt.PTX_SUFFIX


def _seal(bundle: Path, manifest: prebuilt.Manifest) -> prebuilt.Manifest:
  """`manifest` with the digest of `bundle`'s entries (names and sizes), written to the bundle."""
  h = hashlib.blake2b(digest_size=8)
  for path in sorted(bundle.rglob("*")):
    if path.is_file() and path.name != prebuilt.MANIFEST:
      h.update(f"{path.relative_to(bundle)}:{path.stat().st_size}".encode())
  sealed = dataclasses.replace(manifest, content=h.hexdigest())
  sealed.write(bundle / prebuilt.MANIFEST)
  return sealed


def collect(root: Path, arch: int, out: Path, workers: int) -> Path:
  """The kernels bundle of the caches under `root` for SM version `arch` (`out`/NAME/sm<arch>).
  The PTX are compressed by `workers` processes (xz 9e)."""
  bundle = out / NAME / f"sm{arch}"
  shutil.rmtree(bundle, ignore_errors=True)
  bundle.mkdir(parents=True)
  warp_cache = root / "warp" / wp.config.version
  for module in sorted(warp_cache.iterdir()) if warp_cache.is_dir() else []:
    # What loads: the cubins of `arch`, the host (CPU) objects of the modules a renderer launches
    # on the host, metas; no sources (the OptiX programs are PTX: kept below).
    files = [f for f in module.iterdir() if f.suffix in (".meta", ".o") or (f.suffix == ".cubin" and f".sm{arch}." in f.name)]
    if not any(f.suffix in (".cubin", ".o") for f in files):
      continue
    (bundle / "warp" / module.name).mkdir(parents=True)
    for f in files:
      shutil.copy2(f, bundle / "warp" / module.name / f.name)
  found = sorted((root / "wpbr" / "ptx").glob("*.ptx"))
  entries = {ptx: e for ptx in found if (e := _entry(ptx)) is not None}
  if len(entries) != len(found):
    print(f"collect: {len(found) - len(entries)} PTX without an entry of this kernel code left out", flush=True)
  (bundle / "ptx").mkdir(exist_ok=True)
  # One compressed file per distinct PTX (read in place), and one entry per key naming it.
  names = {ptx: _content_name(ptx) for ptx in entries}
  unique = {name: ptx for ptx, name in names.items()}
  index = {ptx.stem: dataclasses.replace(e, ptx=names[ptx]).json() for ptx, e in entries.items()}
  spawn = multiprocessing.get_context("spawn")  # no fork of a threaded process
  with concurrent.futures.ProcessPoolExecutor(workers, mp_context=spawn) as pool:
    targets = [bundle / "ptx" / name for name in unique]
    for i, _ in enumerate(pool.map(_compress, unique.values(), targets), 1):
      if i % 100 == 0 or i == len(unique):
        print(f"collect: {i}/{len(unique)} PTX compressed", flush=True)  # the heartbeat
  write_entries(bundle, index)
  _seal(bundle, prebuilt.Manifest(arch=arch, warp=wp.config.version, content=""))
  print(f"bundle {bundle}: {_modules(bundle)}, {len(index)} builds, {len(unique)} PTX, {_size(bundle) / 2**20:.1f} MB")
  return bundle


def _entry(ptx: Path) -> prebuilt.Entry | None:
  """The fill's entry of PTX file `ptx` (`<key>.json` beside it) when it is of this kernel code
  (`prebuilt.Entry.fresh`; None: absent, damaged or stale)."""
  try:
    entry = prebuilt.Entry.of(json.loads(ptx.with_suffix(".json").read_text()))
  except (OSError, ValueError, KeyError, TypeError):
    return None
  return entry if entry.fresh() else None


def merge(bundles: list[Path], out: Path, census: tuple[Path, ...] = ()) -> Path:
  """One kernels bundle (`out`/NAME/sm<arch>) from the bundles of parallel fill shards (`fill.py
  fill --shard`, each collected): the union of their Warp modules, PTX and indexes (`_unite`);
  their manifests must agree (architecture, Warp release). `census`: the census files whose module
  records the index takes (`fill.census_records`, united with the shards'): a fill's shards then
  compile without waiting for the census."""
  manifests = {dataclasses.replace(prebuilt.Manifest.read(b / prebuilt.MANIFEST), content="") for b in bundles}
  if len(manifests) != 1:
    raise SystemExit(f"the shards' manifests differ: {sorted(map(repr, manifests))}")
  (manifest,) = manifests
  target = out / NAME / f"sm{manifest.arch}"
  shutil.rmtree(target, ignore_errors=True)
  target.mkdir(parents=True)
  index: dict[str, object] = {}
  for b in bundles:
    for part in ("warp", "ptx"):
      if (b / part).is_dir():
        shutil.copytree(b / part, target / part, dirs_exist_ok=True)
    _unite(index, read_entries(b))
  if census:
    _record(index, census)
  write_entries(target, index)
  _seal(target, manifest)
  nptx = len(list((target / "ptx").glob(f"*{prebuilt.PTX_SUFFIX}")))
  print(
    f"bundle {target}: {len(bundles)} shards, {_modules(target)}, {len(index)} builds, {nptx} PTX, "
    f"{_size(target) / 2**20:.1f} MB"
  )
  return target


def _modules(bundle: Path) -> str:
  """The Warp modules of `bundle` by kind (`fill.module_kind`: its renderer's, MuJoCo Warp's)."""
  names = [p.name for p in (bundle / "warp").iterdir()] if (bundle / "warp").is_dir() else []
  return f"{len(names)} Warp modules {fill.kind_counts(names)}"


def read_entries(bundle: Path) -> dict[str, dict[str, object]]:
  """The entries of a complete bundle directory `bundle` as parsed JSON, PTX key -> entry
  (`prebuilt.entry_path`)."""
  suffix = prebuilt.ENTRY_SUFFIX
  return {f.name[: -len(suffix)]: json.loads(f.read_text()) for f in sorted((bundle / "ptx").glob(f"*{suffix}"))}


def write_entries(bundle: Path, entries: Mapping[str, object]) -> None:
  """Writes `entries` (PTX key -> entry JSON) as the entry files of bundle directory `bundle`,
  replacing those it held (`prebuilt.entry_path`)."""
  for f in (bundle / "ptx").glob(f"*{prebuilt.ENTRY_SUFFIX}"):
    f.unlink()
  (bundle / "ptx").mkdir(parents=True, exist_ok=True)
  for key, raw in entries.items():
    (bundle / prebuilt.entry_path(key)).write_text(json.dumps(raw))


def _unite(index: dict[str, object], other: Mapping[str, object]) -> None:
  """Adds bundle entries `other` to `index` (both PTX key -> entry JSON): per key its PTX file
  (the first shard's: equal PTX, equal content name) and per tier the union of the module records
  (in order, without repeats; a GI key two shards' builds share holds both shards' records)."""
  for key, raw in other.items():
    entry = prebuilt.Entry.of(raw)
    mine = index.get(key)
    if mine is None:
      index[key] = raw
      continue
    have = prebuilt.Entry.of(mine)
    groups = dict(have.groups)
    for g, records in entry.groups.items():
      groups[g] = tuple(dict.fromkeys((*groups.get(g, ()), *records)))
    index[key] = dataclasses.replace(have, groups=groups).json()


def _record(index: dict[str, object], census: tuple[Path, ...]) -> None:
  """Adds the module records the census files `census` reach to `index`'s entries (`_unite`'s
  union per tier), and prints the census keys no bundled build holds (those scenes compile)."""
  records, nrow, old = fill.census_records(list(census))
  for key, groups in records.items():
    if key in index:
      _unite(index, {key: dataclasses.replace(prebuilt.Entry.of(index[key]), groups=groups).json()})
  missing = sorted(set(records) - set(index))
  nmodule = sum(len(rs) for groups in records.values() for rs in groups.values())
  print(
    f"merge census: {nrow} rows ({old} without units), {nmodule} module records over {len(records)} PTX keys; "
    f"{len(missing)} keys no bundled build holds{f' (e.g. {missing[0]})' if missing else ''}",
    flush=True,
  )


def _size(path: Path) -> int:
  """Bytes of the files under `path`."""
  return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def pack(
  bundle: Path,
  out: Path,
  workers: int,
  *,
  shard: tuple[int, int] = (0, 1),
  select: tuple[Path, ...] = (),
  groups: tuple[fill.Group, ...] = (),
  keep: Path | None = None,
  seed: str | None = None,
  deadline: float | None = None,
) -> Path:
  """The OptiX module pack of `bundle` for this machine's GPU and driver (`out`/PACK/
  sm<arch>-drv<driver>/m/<module>.db, `prebuilt.module_path`): every module its index records
  (`select`: only those these census files reach, in `groups`' order, `_packed`; `shard` (I, N):
  every N-th of them from the I-th, `merge_packs` unites the N packs), each created by OptiX from
  the bundled PTX in a worker process of its own (`workers` at a time, `_module_file`) into a fresh
  disk cache: one module's rows per file. `keep`: a directory (on the bucket) each module file is
  copied to as it is made; a run with the same `keep` resumes from it (a cap or a termination
  loses the modules in flight). `seed`: the URL of a published pack of this driver whose modules
  are fetched, not made (`open_seed`). `deadline` (epoch seconds): no module starts after it; the
  pack then holds the modules done (in `_packed`'s order: group by group, the most used first).
  Prints the module count, the wall time and the size."""
  b = prebuilt.Bundle(bundle, prebuilt.Manifest.read(bundle / prebuilt.MANIFEST))
  driver = prebuilt.driver_version()
  if not driver:
    raise SystemExit("no NVIDIA driver version (/proc/driver/nvidia/version): a pack is per driver")
  index, count = shard
  target = out / PACK / f"sm{b.manifest.arch}-drv{driver}"
  shutil.rmtree(target, ignore_errors=True)
  (target / prebuilt.MODULE_DIR).mkdir(parents=True)
  scratch = Path(tempfile.mkdtemp(prefix="wpbr_pack_"))
  todo = _packed(b, select, groups or fill.EVERY)[index::count]
  spawn = multiprocessing.get_context("spawn")  # one CUDA / OptiX context per worker, no fork
  try:
    published = None if seed is None else open_seed(seed, scratch / "seed", b.manifest.arch, driver)
    with concurrent.futures.ProcessPoolExecutor(max(1, workers), mp_context=spawn, initializer=_worker_env) as pool:
      done, wall = create_kept(
        pool, bundle, todo, target, workers=workers, scratch=scratch, keep=keep, seed=published, deadline=deadline
      )
    _seal(target, dataclasses.replace(b.manifest, content="", driver=driver))
  finally:
    shutil.rmtree(scratch, ignore_errors=True)
  print(f"pack {target}: {done} OptiX modules in {wall:.0f} s on {workers} processes, {_size(target) / 2**20:.1f} MB")
  return target


def _worker_env() -> None:
  """A pack worker process: OptiX's disk cache is the one each module's own context names
  (`_module_file`), never OPTIX_CACHE_PATH's (which would override it)."""
  os.environ.pop(prebuilt.OPTIX_CACHE_ENV, None)


def _module_file(bundle: Path, ptx: str, record: dict[str, object], target: Path, scratch: Path) -> tuple[str, float]:
  """One module of the pack (a worker process): the module of bundled PTX file `ptx` with
  `record` (`ModuleRecord` JSON) created by OptiX on a device context of its own whose disk cache is
  a fresh directory under `scratch` (`modules.make`: the curve module too), then that cache (every
  row its creation wrote, nothing else) folded into `target`'s module file; (module name, seconds).

  Raises:
    RuntimeError: OptiX rejected it, or wrote no row (its disk cache is off).
  """
  from raywarp.core.trace.optix import context, modules  # the OptiX binding: a GPU host's

  wp.init()
  code = lzma.decompress((bundle / "ptx" / ptx).read_bytes())
  rec = prebuilt.ModuleRecord.of(record)
  name = modules.digest(code, rec)
  cache = scratch / name
  shutil.rmtree(cache, ignore_errors=True)
  device = wp.get_device("cuda:0")
  ctx = context.DeviceContext(int(device.context), int(device.arch))  # pyrefly: ignore[bad-argument-type]
  t0 = time.perf_counter()
  try:
    ctx.set_cache_location(str(cache))
    ctx.set_cache_sizes(context.CACHE_LOW, context.CACHE_HIGH)
    modules.make(ctx, code, rec).destroy()
  finally:
    ctx.close()  # OptiX has written every row
  seconds = time.perf_counter() - t0
  _fold(cache / prebuilt.OPTIX_DB, target / prebuilt.module_path(name))
  shutil.rmtree(cache, ignore_errors=True)
  return name, seconds


def _fold(db: Path, dst: Path) -> None:
  """OptiX disk cache `db` (one module's rows, one driver's) folded into module file `dst` (one
  file, the WAL folded in; written under a temporary name, then renamed).

  Raises:
    RuntimeError: `db` holds no row, or rows of several drivers.
  """
  with sqlite3.connect(db) as c:
    drivers = sorted(r[0] for r in c.execute("SELECT DISTINCT driver_version FROM cache_info").fetchall())
    if len(drivers) != 1:
      raise RuntimeError(f"{db}: OptiX rows of drivers {drivers}: a module file holds one driver's rows")
    partial = dst.with_name(f".{dst.name}.{os.getpid()}.partial")
    partial.unlink(missing_ok=True)
    c.execute("VACUUM INTO ?", (str(partial),))
  partial.replace(dst)


def create_kept(
  pool: concurrent.futures.Executor,
  bundle: Path,
  todo: list[tuple[str, prebuilt.ModuleRecord]],
  target: Path,
  *,
  workers: int,
  scratch: Path,
  keep: Path | None,
  seed: Seed | None = None,
  deadline: float | None = None,
) -> tuple[int, float]:
  """Makes the module files of `todo` ((bundled PTX file, record), `_module_file`) into `target`
  in order, `workers` at a time on `pool` (a module ending starts the next: a long one never holds
  the others back); the modules `keep` already holds are copied, those the published pack `seed`
  holds fetched (`_take_seeded`), neither made; each new file is copied to `keep`; no module
  starts past `deadline` (epoch seconds), nor one of a PTX whose earlier module took longer than
  the time left (a build's modules take alike: one of minutes is left to first use, not cut by the
  job's cap); (modules in `target`, wall seconds)."""
  names = _names(bundle, todo)
  kept = _resume(keep, target, set(names.values()))
  kept |= _take_seeded(seed, target, set(names.values()) - kept) if seed is not None else set()
  left = iter([j for j in todo if names[j] not in kept])
  t0 = time.perf_counter()
  flight: dict[concurrent.futures.Future[tuple[str, float]], tuple[str, prebuilt.ModuleRecord]] = {}
  took: dict[str, float] = {}  # PTX file -> the longest creation of its modules so far
  made = len(kept)
  while True:
    stopped = deadline is not None and time.time() >= deadline
    while not stopped and len(flight) < max(1, workers) and (job := next(left, None)) is not None:
      if deadline is not None and time.time() + took.get(job[0], 0.0) > deadline:
        continue  # its build's modules take longer than the time left
      flight[pool.submit(_module_file, bundle, job[0], job[1].json(), target, scratch)] = job
    if not flight:
      break
    ended, _ = concurrent.futures.wait(flight, return_when=concurrent.futures.FIRST_COMPLETED)
    for future in ended:
      job = flight.pop(future)
      name, seconds = future.result()
      took[job[0]] = max(took.get(job[0], 0.0), seconds)
      made += 1
      if keep is not None:
        _copy(target / prebuilt.module_path(name), keep / prebuilt.module_path(name))
      if made % 50 == 0:
        print(f"pack: {made}/{len(names)} modules ({time.perf_counter() - t0:.0f} s)", flush=True)
  if deadline is not None and time.time() >= deadline:
    print(f"pack stopped at the deadline: {made}/{len(names)} modules done", flush=True)
  return made, time.perf_counter() - t0


def _names(bundle: Path, todo: list[tuple[str, prebuilt.ModuleRecord]]) -> dict[tuple[str, prebuilt.ModuleRecord], str]:
  """The module name (`modules.digest`) of each job of `todo`."""
  from raywarp.core.trace.optix import modules

  codes: dict[str, bytes] = {}
  out = {}
  for ptx, record in todo:
    if ptx not in codes:
      codes[ptx] = lzma.decompress((bundle / "ptx" / ptx).read_bytes())
    out[ptx, record] = modules.digest(codes[ptx], record)
  return out


def _resume(keep: Path | None, target: Path, wanted: set[str]) -> set[str]:
  """The modules of `wanted` whose files `keep` holds (an earlier run's), copied into `target`."""
  if keep is None or not (keep / prebuilt.MODULE_DIR).is_dir():
    return set()
  found = {f.name.removesuffix(prebuilt.MODULE_SUFFIX) for f in (keep / prebuilt.MODULE_DIR).glob(f"*{prebuilt.MODULE_SUFFIX}")}
  kept = found & wanted
  for name in kept:
    _copy(keep / prebuilt.module_path(name), target / prebuilt.module_path(name))
  if kept:
    print(f"pack resumed from {keep}: {len(kept)} modules done", flush=True)
  return kept


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Seed:
  """A published module pack whose modules a pack fetches instead of making them (`open_seed`).

  Attributes:
    url: its remote directory (`<kernels>/sm<arch>/<content>/optix/<driver>`).
    files: its file list.
  """

  url: str
  files: store.Listing


def open_seed(url: str, local: Path, arch: int, driver: str) -> Seed:
  """The published pack at `url` (its file list and manifest fetched into `local`) as a seed of a
  pack of SM version `arch` and NVIDIA driver `driver`: an earlier release's modules of unchanged
  PTX and record (a module's name is `modules.digest` of both) are the same modules.

  Raises:
    SystemExit: it cannot be fetched, or it is of another architecture or driver.
  """
  try:
    files = store.listing(local, url, None)
    failed = store.fetch(local, url, files, [prebuilt.MANIFEST])
  except store.FetchError as e:
    raise SystemExit(f"the seed pack {url}: {e}") from e
  if failed:
    raise SystemExit(f"the seed pack {url}: {failed}")
  manifest = prebuilt.Manifest.read(local / prebuilt.MANIFEST)
  if (manifest.arch, manifest.driver) != (arch, driver):
    raise SystemExit(f"the seed pack {url} is of sm{manifest.arch} driver {manifest.driver}, not sm{arch} driver {driver}")
  return Seed(url, files)


def _take_seeded(seed: Seed, target: Path, wanted: set[str]) -> set[str]:
  """The modules of `wanted` that pack `seed` holds, fetched into `target` (each checked against
  its file list).

  Raises:
    SystemExit: a module could not be fetched.
  """
  held = sorted(n for n in wanted if prebuilt.module_path(n) in seed.files.files)
  failed = store.fetch(target, seed.url, seed.files, [prebuilt.module_path(n) for n in held])
  if failed:
    raise SystemExit(f"the seed pack {seed.url}: {len(failed)} modules not fetched, e.g. {next(iter(failed.items()))}")
  print(f"pack seeded from {seed.url}: {len(held)} of {len(wanted)} modules", flush=True)
  return set(held)


def _copy(src: Path, dst: Path) -> None:
  """`src` copied to `dst` (under a temporary name, then renamed: a reader never sees a part)."""
  dst.parent.mkdir(parents=True, exist_ok=True)
  partial = dst.with_name(f".{dst.name}.{os.getpid()}.partial")
  shutil.copyfile(src, partial)
  partial.replace(dst)


def _packed(
  b: prebuilt.Bundle, select: tuple[Path, ...], groups: tuple[fill.Group, ...] = fill.EVERY
) -> list[tuple[str, prebuilt.ModuleRecord]]:
  """(PTX file, record) of every module bundle `b` records (by key, tier, record), or only those
  the census files `select` reach, in `fill.census_ranked`'s order of `groups` (e.g. the default
  and fast recipes' own words, then their neighbours; the others are created on first use or by
  `Renderer.warm`): a pack cut at its deadline holds the first; each module once."""
  entries = b.keyed()
  if not select:
    return sorted({(e.ptx, r) for e in entries.values() for rs in e.groups.values() for r in rs}, key=repr)
  ranked = fill.census_ranked(list(select), groups)[0]
  held = {(k, g, r) for k, e in entries.items() for g, rs in e.groups.items() for r in rs}
  return list(dict.fromkeys((entries[k].ptx, r) for k, g, r in ranked if (k, g, r) in held))


def merge_packs(packs: list[Path], out: Path) -> Path:
  """One OptiX module pack from the packs of `pack --shard` jobs (same architecture, Warp release,
  and driver: their manifests agree): the union of their module files."""
  manifests = {dataclasses.replace(prebuilt.Manifest.read(p / prebuilt.MANIFEST), content="") for p in packs}
  if len(manifests) != 1:
    raise SystemExit(f"the packs' manifests differ: {sorted(map(repr, manifests))}")
  (manifest,) = manifests
  target = out / PACK / f"sm{manifest.arch}-drv{manifest.driver}"
  shutil.rmtree(target, ignore_errors=True)
  (target / prebuilt.MODULE_DIR).mkdir(parents=True)
  for p in packs:
    for f in (p / prebuilt.MODULE_DIR).glob(f"*{prebuilt.MODULE_SUFFIX}"):
      shutil.copyfile(f, target / prebuilt.MODULE_DIR / f.name)
  _seal(target, manifest)
  n = len(list((target / prebuilt.MODULE_DIR).iterdir()))
  print(f"pack {target}: {len(packs)} shards, {n} modules, {_size(target) / 2**20:.1f} MB")
  return target


def main(argv: list[str] | None = None) -> None:
  """`collect`, `merge`, `pack`, `merge-packs`, `rekey` (see the module doc)."""

  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("collect")
  c.add_argument("--root", type=Path, required=True, help="the cache root a fill filled (tools/prebuilt/fill.py)")
  c.add_argument("--arch", type=int, required=True)
  c.add_argument("--out", type=Path, required=True)
  c.add_argument("--workers", type=int, required=True, help="xz processes (a job's cores, not the host's)")
  k = sub.add_parser("pack")
  k.add_argument("--bundle", type=Path, required=True)
  k.add_argument("--out", type=Path, required=True)
  k.add_argument("--workers", type=int, default=4, help="modules at a time (worker processes, each in parallel OptiX tasks)")
  k.add_argument(
    "--shard", type=fill.fraction, default=(0, 1), help="I/N: every N-th module from the I-th (N jobs, then merge-packs)"
  )
  k.add_argument("--select", type=Path, action="extend", nargs="+", default=[], help="pack only what these census files reach")
  k.add_argument(
    "--groups",
    type=fill.groups_of,
    default=(),
    help="the --select units in this order, e.g. quality:own,fast+default:own,fast:near (default: own words, then neighbours)",
  )
  k.add_argument("--keep", type=Path, default=None, help="copy each module file here; resume from it (a bucket path)")
  k.add_argument("--seed", default=None, help="a published pack's URL (this driver): fetch the modules it holds, not made")
  k.add_argument("--deadline", type=float, default=None, help="epoch seconds: no module batch starts after it")
  g = sub.add_parser("merge")
  g.add_argument("--bundle", type=Path, action="append", required=True, help="a shard's collected bundle (repeat)")
  g.add_argument("--out", type=Path, required=True)
  g.add_argument(
    "--census", type=Path, action="extend", nargs="+", default=[], help="census files whose module records the index takes"
  )
  mp = sub.add_parser("merge-packs")
  mp.add_argument("--pack", type=Path, action="append", required=True, help="a `pack --shard` output (repeat)")
  mp.add_argument("--out", type=Path, required=True)
  r = sub.add_parser("rekey")
  r.add_argument("--bundle", type=Path, required=True, help="a kernels bundle (<package>/sm<arch>/)")
  r.add_argument("--out", type=Path, required=True)
  r.add_argument("--since", required=True, help="the git revision the bundle was filled from")
  r.add_argument("--old", action="append", default=[], help="NAME=VALUE: a key part as it was (programs.PtxParts)")
  r.add_argument("--ignore", action="append", default=[], help="a kernel-code unit changed since that generates no code")
  args = p.parse_args(argv)
  if args.cmd == "rekey":
    _rekey(args.bundle, args.out, args.since, args.old, frozenset(args.ignore))
  elif args.cmd == "pack":
    pack(
      args.bundle,
      args.out,
      args.workers,
      shard=args.shard,
      select=tuple(args.select),
      groups=args.groups,
      keep=args.keep,
      seed=args.seed,
      deadline=args.deadline,
    )
  elif args.cmd == "merge-packs":
    merge_packs(args.pack, args.out)
  elif args.cmd == "merge":
    merge(args.bundle, args.out, tuple(args.census))
  else:
    collect(args.root, args.arch, args.out, args.workers)


def _rekey(bundle: Path, out: Path, since: str, old: list[str], ignore: frozenset[str]) -> Path:
  """`rekey.rekey` of `bundle` over every fill point's build (`fill._points`), its report printed;
  the bundle written."""
  from tools.prebuilt import rekey

  builds = [b for _, b in fill._points("optix", modules=False) if b is not None]
  keys = rekey.key_map(builds, rekey.parse_old(old))
  target, report = rekey.rekey(bundle, out, keys, rekey.units_at(since), ignore)
  print(
    f"rekey {target}: {report.kept} entries kept; dropped (a refill compiles them): {len(report.unmapped)} no fill "
    f"build reaches, {len(report.stale)} not of {since}'s code, {len(report.changed)} changed since",
    flush=True,
  )
  for key, units in sorted(report.changed.items())[:20]:
    print(f"rekey changed {key}: {units[:5]}", flush=True)
  return target


if __name__ == "__main__":
  main(sys.argv[1:])
