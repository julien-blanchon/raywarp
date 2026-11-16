"""Prebuilt kernels: the compiled caches of a release (`performance.kernels="prebuilt"`),
downloaded on first use.

A bundle, one per GPU architecture and kernel code, built for one Warp release (`Manifest`):
- `warp/`: Warp's kernel cache entries (one directory per module: cubin and meta, named by the
  content hash of its generated code);
- `ptx/`: the OptiX programs' PTX, one per true code split, xz-compressed and stored by content
  (`<sha>.ptx.xz`: builds with the same PTX share one file), and one entry per PTX key
  (`programs.ptx_key`), `<key>.json`: its file, the kernel code it was generated from (its reach
  and digest: `kernel_code.digest`; an entry of other code is not used) and, per tier, the module
  records the census reaches (`Entry`, `ModuleRecord`: compile options and bound values, the class
  bits among them; most used first), so that the modules compile without a scene. A start reads
  (fetches) only its own builds' entries, never all of them;
- `optix/<driver>/`, optional, one per NVIDIA driver release: the OptiX module pack, the index's
  modules as rows of OptiX's own disk cache, one small database per module (`module_path`:
  `m/<module>.db`, the module named by `trace/optix/modules.digest` of its PTX and record; valid for
  that driver only; `tools/prebuilt/build.py pack` on a GPU host).

Where they live (`kernel_store`): a remote (the releases' public Hugging Face storage bucket, or a
mirror: `RAYWARP_KERNELS_URL`, file:// too) and the local store
(`RAYWARP_KERNELS`, else `<cache dir>/kernels`), both laid out `sm<arch>/<content>/`: the
bundle's files and `files.json` (every file's sha256 and size; the remote holds each file
xz-compressed, `<path>.xz`, except the PTX, already compressed). The release pins each
architecture's bundle (`kernels.json` beside this module: its content key and the sha256 of its
`files.json`), so every byte that reaches the caches is checked against the installed package.
One loader: a file the store lacks is downloaded into it (checked, then renamed into place), so a
store filled ahead (`prefetch`, a copied directory) is an offline mirror, read the same way.

Only what is used is fetched, never the whole bundle: the common Warp modules once (`seed`:
`common_modules`, 13 MB on sm120), a renderer's modules of the large specialisation families in
one parallel batch before they load (`fetch_modules`, from `warmup`), each OptiX build's PTX when
its pipeline is built (`ptx`), each OptiX module's rows of the running driver's pack just before
OptiX looks for that module (`seed_module`: merged into OptiX's disk cache, a cache hit then);
Warp modules are hard-linked from the store into Warp's kernel cache.
A bundle is checked once per process (`bundle`): a damaged manifest, another Warp release or
architecture warns once and the bundle is not used. A file that cannot be fetched (no network, a
damaged download) warns once per process, loudly: what it held compiles, once (cached) - the only
compile path left, and zero compiles is the goal. An entry the bundle should hold but lacks warns
too. A module the pack lacks (another driver, a combination outside the census) is created by
OptiX alone, never NVRTC, once per machine and driver (`trace/optix/modules.py`: on first use under
a lock); OptiX keeps it in its disk cache like a pack's.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Final, final

import warp as wp

from raywarp.cache.paths import cache_dir
from raywarp.core.build.kernel_cache import kernel_code, optix_merge
from raywarp.core.build.kernel_cache import store as kernel_store

MANIFEST: Final = "manifest.json"
# A build's entry file in a bundle's `ptx/` (`entry_path`: `<key>.json`, the `Entry` JSON).
ENTRY_SUFFIX: Final = ".json"
# A bundled PTX file's suffix (xz-compressed PTX, its integrity checked by xz's CRC64).
PTX_SUFFIX: Final = ".ptx.xz"
# OptiX's disk cache file inside its cache directory (`optix_cache_dir`).
OPTIX_DB: Final = "optix7cache.db"
# OptiX's own variable naming its module cache directory (it overrides the API's location).
OPTIX_CACHE_ENV: Final = "OPTIX_CACHE_PATH"
# A module pack's directory of module files (`module_path`) and their suffix (an OptiX disk-cache
# database holding one module's rows).
MODULE_DIR: Final = "m"
MODULE_SUFFIX: Final = ".db"
# A bundle the pins do not name, by its content key (a fill's own bundle in the store: the fill
# tooling and benchmarks; its files.json is not pinned).
BUNDLE_ENV: Final = "RAYWARP_KERNELS_BUNDLE"
# A Warp module family (one kernel factory: its variants differ by a content hash) of more variants
# than this is fetched by name only (`common_modules`).
FAMILY_FETCHED_BY_NAME: Final = 500
# A Warp module directory's hashes: the kernel closure's (unique modules) and the module's content.
_FAMILY: Final = re.compile(r"(_[0-9a-f]{8})?_[0-9a-f]{7}$")
# The NVIDIA kernel module's version line (Linux).
_DRIVER_FILE: Final = Path("/proc/driver/nvidia/version")
# Guards the checked bundles, the bundles' lazily read entries and the warned set.
_LOCK: Final = threading.Lock()
# Held for a whole `seed`: a thread that finds the bundle being seeded waits for the copy (one that
# returned at once loaded its first module before the copy held it, and compiled it).
_SEED_LOCK: Final = threading.Lock()
# The checked bundle of each architecture (`bundle`; None: none usable), the module pack of each
# (architecture, driver) (`module_pack`), the seeded architectures, those whose missing bundle
# `ptx` warned about, and whether a failed fetch warned (once per process).
_BUNDLES: Final[dict[int, Bundle | None]] = {}
_PACKS: Final[dict[tuple[int, str], Pack | None]] = {}
_SEEDED: Final[set[int]] = set()
_WARNED: Final[set[int]] = set()
_UNREACHABLE: Final[list[str]] = []
# raywarp's package directory (as its code objects name their files): `warn` reports at the
# first caller outside it.
_PACKAGE_DIR: Final = str(Path(__file__).parents[__name__.count(".") - 1])


def warn(message: str) -> None:
  """A RuntimeWarning `message`, reported at the first caller outside raywarp."""
  warnings.warn(f"raywarp: {message}", RuntimeWarning, skip_file_prefixes=(_PACKAGE_DIR,))


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
  """What a bundle or pack was built for.

  Attributes:
    arch: SM version of its cubins (e.g. 120).
    warp: the Warp release.
    content: a digest of its entries (`tools/prebuilt/build.py`): its store directory's name.
    driver: a module pack's NVIDIA driver ("" a kernels bundle's).
  """

  arch: int
  warp: str
  content: str
  driver: str = ""

  @staticmethod
  def read(path: Path) -> Manifest:
    """The manifest file at `path` (OSError, ValueError, KeyError: missing or damaged)."""
    data = json.loads(path.read_text())
    return Manifest(
      arch=int(data["arch"]),
      warp=str(data["warp"]),
      content=str(data["content"]),
      driver=str(data.get("driver", "")),
    )

  def write(self, path: Path) -> None:
    """Writes the manifest to `path`."""
    path.write_text(json.dumps(dataclasses.asdict(self), indent=1))

  def mismatch(self, arch: int) -> str | None:
    """Why a bundle of this manifest cannot serve this process on SM version `arch` (None: it
    can): every key and module of another Warp release differs, so it would never hit (the
    kernel code is checked per entry: `Entry.fresh`)."""
    if self.arch != arch:
      return f"built for sm_{self.arch}, not sm_{arch}"
    if self.warp != wp.config.version:
      return f"built for Warp {self.warp}, this is Warp {wp.config.version}"
    return None


@final
@dataclasses.dataclass(frozen=True, slots=True)
class ModuleRecord:
  """The options one OptiX module of a PTX is compiled with
  (`trace/optix/pipeline.compile_options`), what compiles it without a scene: the fill writes those
  the census reaches into the bundle entry (`tools/prebuilt/fill.py`). Opaque to the packaging: the
  bound words (array strides, the class bits) are bytes at launch-parameter offsets.

  Attributes:
    has_curves, has_custom, opacity_micromaps, max_registers: as `pipeline.compile_options`.
    bound: the bound launch-parameter words, (offset in bytes, hex bytes).
  """

  has_curves: bool
  has_custom: bool
  opacity_micromaps: bool
  max_registers: int
  bound: tuple[tuple[int, str], ...]

  @staticmethod
  def of(data: object) -> ModuleRecord:
    """The record of its JSON form (TypeError, KeyError: damaged)."""
    d = _json(data, dict)
    return ModuleRecord(
      has_curves=_json(d["has_curves"], bool),
      has_custom=_json(d["has_custom"], bool),
      opacity_micromaps=_json(d["opacity_micromaps"], bool),
      max_registers=_json(d["max_registers"], int),
      bound=tuple((_json(o, int), _json(v, str)) for o, v in (_json(w, list) for w in _json(d["bound"], list))),
    )

  def json(self) -> dict[str, object]:
    """Its JSON form (`of`)."""
    return dataclasses.asdict(self) | {"bound": [list(w) for w in self.bound]}


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Entry:
  """One build of a bundle (its `entry_path`), or of a fill's PTX cache (`<key>.json` beside
  `<key>.ptx`).

  Attributes:
    ptx: its compressed PTX file in the bundle's `ptx/` (shared by the builds of equal PTX; "" in
      a PTX cache).
    code: the digest of the kernel code its PTX was generated from (`kernel_code.digest(reach)`).
    reach: the kernel-code functions its generation used (`kernel_code.Reach`).
    groups: per tier (a `tiers.TIERS` name), the modules the census reaches in it (the module
      pack's rows), most used first. Empty: no census scene uses the build.
  """

  ptx: str
  code: str
  reach: kernel_code.Reach
  groups: dict[str, tuple[ModuleRecord, ...]]

  @property
  def modules(self) -> tuple[ModuleRecord, ...]:
    """Every recorded module, once (in group order)."""
    return tuple(dict.fromkeys(r for records in self.groups.values() for r in records))

  def fresh(self) -> bool:
    """Whether its PTX is the installed kernel code's: the code its reach covers is unchanged."""
    return kernel_code.digest(self.reach) == self.code

  @staticmethod
  def of(data: object) -> Entry:
    """The entry of its JSON form (TypeError, KeyError: damaged)."""
    d = _json(data, dict)
    groups = {
      _json(g, str): tuple(ModuleRecord.of(r) for r in _json(records, list)) for g, records in _json(d["groups"], dict).items()
    }
    reach = frozenset(_json(u, str) for u in _json(d["reach"], list))
    return Entry(ptx=_json(d["ptx"], str), code=_json(d["code"], str), reach=reach, groups=groups)

  def json(self) -> dict[str, object]:
    """Its JSON form (`of`)."""
    return {
      "ptx": self.ptx,
      "code": self.code,
      "reach": sorted(self.reach),
      "groups": {g: [r.json() for r in rs] for g, rs in self.groups.items()},
    }


def _json[T](value: object, kind: type[T]) -> T:
  """`value` (parsed JSON) as a `kind`; TypeError when it is not one."""
  if not isinstance(value, kind):
    raise TypeError(f"{value!r} is not a {kind.__name__}")
  return value


@final
@dataclasses.dataclass(slots=True)
class Bundle:
  """A kernels bundle this process can use (`bundle`), in the store or a complete local directory.

  Attributes:
    root: its directory (`<store>/sm<arch>/<content>/`, or a fill's complete bundle).
    manifest: its manifest.
    url: its remote directory (None: `root` is complete, nothing is fetched).
    files: its file list (None with `url` None).
  """

  root: Path
  manifest: Manifest
  url: str | None = None
  files: kernel_store.Listing | None = None
  _entries: dict[str, Entry | None] = dataclasses.field(default_factory=dict, repr=False, compare=False)
  _modules: dict[str, list[str]] | None = dataclasses.field(default=None, repr=False, compare=False)

  def path(self, rel: str) -> Path:
    """The local file of bundle path `rel`, fetched first when the store lacks it.

    Raises:
      kernel_store.FetchError: it cannot be fetched (warned once per process: `unreachable`).
    """
    return _fetched(self.root, self.url, self.files, rel)

  def fetch(self, rels: Iterable[str]) -> dict[str, str]:
    """Fetches the bundle paths `rels` the store lacks, in parallel; {path: error} of the failed."""
    if self.url is None or self.files is None:
      return {}
    return kernel_store.fetch(self.root, self.url, self.files, rels)

  def module_files(self) -> dict[str, list[str]]:
    """Its Warp modules: directory name -> the bundle paths of its files."""
    with _LOCK:
      if self._modules is None:
        modules: dict[str, list[str]] = {}
        for rel in self._names():
          parts = rel.split("/")
          if len(parts) == 3 and parts[0] == "warp":
            modules.setdefault(parts[1], []).append(rel)
        self._modules = modules
      return self._modules

  def _names(self) -> list[str]:
    """Every bundle path: its file list's, else (a complete local bundle) its files'."""
    return list(self.files.files) if self.files is not None else _local_files(self.root)

  def keys(self) -> list[str]:
    """The PTX key of every build it holds an entry for (`entry_path`)."""
    return [rel[4 : -len(ENTRY_SUFFIX)] for rel in self._names() if _ENTRY_PATH.fullmatch(rel)]

  def entry(self, key: str) -> Entry | None:
    """The entry of PTX key `key` (`entry_path`: read once, fetched when the store lacks it; None:
    the bundle lacks it, or holds it for other kernel code: `Entry.fresh`).

    Raises:
      ValueError: the entry is damaged.
      kernel_store.FetchError: it cannot be fetched (warned once per process).
    """
    with _LOCK:
      if key in self._entries:
        return self._entries[key]
    rel = entry_path(key)
    held = rel in self.files.files if self.files is not None else (self.root / rel).is_file()
    entry = _parsed(key, json.loads(self.path(rel).read_text())) if held else None
    fresh = entry if entry is not None and entry.fresh() else None
    with _LOCK:
      return self._entries.setdefault(key, fresh)

  def entries(self) -> list[Entry]:
    """Every entry it holds (fetched in one parallel batch).

    Raises:
      ValueError: an entry is damaged.
    """
    return list(self.keyed().values())

  def keyed(self) -> dict[str, Entry]:
    """Every entry by PTX key (those of the installed kernel code: `entry`), fetched in one
    parallel batch; an entry that cannot be fetched is left out (warned once per process).

    Raises:
      ValueError: an entry is damaged.
    """
    keys = self.keys()
    failed = self.fetch(entry_path(k) for k in keys)
    if failed:
      unreachable(str(self.url), next(iter(failed.values())), f"{len(failed)} PTX entries")
    entries = {k: self.entry(k) for k in keys if entry_path(k) not in failed}
    return {k: e for k, e in entries.items() if e is not None}


def _fetched(root: Path, url: str | None, files: kernel_store.Listing | None, rel: str) -> Path:
  """The local file of path `rel` of the store directory `root` (remote `url`, file list `files`;
  None: complete locally), fetched first when the store lacks it.

  Raises:
    kernel_store.FetchError: it cannot be fetched (warned once per process: `unreachable`).
  """
  local = root / rel
  if url is None or files is None or local.is_file():
    return local
  failed = kernel_store.fetch(root, url, files, [rel])
  if failed:
    unreachable(url, failed[rel], f"{rel} (and what needs it)")
    raise kernel_store.FetchError(failed[rel])
  return local


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Pack:
  """The OptiX module pack of one NVIDIA driver a process can use (`module_pack`): in the store,
  its files fetched one module at a time, or a complete local directory.

  Attributes:
    root: its directory (`<bundle>/optix/<driver>/`).
    url: its remote directory (None: `root` is complete, nothing is fetched).
    files: its file list (None with `url` None).
  """

  root: Path
  url: str | None = None
  files: kernel_store.Listing | None = None

  def holds(self, rel: str) -> bool:
    """Whether it has pack path `rel` (its file list's, else the local directory's)."""
    return rel in self.files.files if self.files is not None else (self.root / rel).is_file()

  def path(self, rel: str) -> Path:
    """The local file of pack path `rel`, fetched first when the store lacks it.

    Raises:
      kernel_store.FetchError: it cannot be fetched (warned once per process: `unreachable`).
    """
    return _fetched(self.root, self.url, self.files, rel)

  def fetch_all(self) -> dict[str, str]:
    """Fetches every file the store lacks, in parallel; {path: error} of the failed."""
    if self.url is None or self.files is None:
      return {}
    return kernel_store.fetch(self.root, self.url, self.files, self.files.files)


def module_path(name: str) -> str:
  """The pack path of OptiX module `name` (`trace/optix/modules.digest`): a database of its rows."""
  return f"{MODULE_DIR}/{name}{MODULE_SUFFIX}"


def entry_path(key: str) -> str:
  """The bundle path of PTX key `key`'s entry (`Entry`)."""
  return f"ptx/{key}{ENTRY_SUFFIX}"


# A bundle path of an entry (`entry_path`): `ptx/<key>.json` (the only JSON files of `ptx/`).
_ENTRY_PATH: Final = re.compile(r"ptx/[^/]+" + re.escape(ENTRY_SUFFIX))


def _local_files(root: Path) -> list[str]:
  """The paths of the files under a complete bundle directory `root`."""
  return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]


def _parsed(key: str, raw: object) -> Entry:
  """The entry `raw` of key `key` (ValueError naming the key: damaged)."""
  try:
    return Entry.of(raw)
  except (KeyError, TypeError) as e:
    raise ValueError(f"the PTX entry {key} is damaged ({e!r})") from e


def unreachable(url: str, error: str, what: str) -> None:
  """Warns, once per process, that prebuilt kernels could not be fetched: what they hold compiles
  (the one compile path left; zero compiles is the goal)."""
  with _LOCK:
    first = not _UNREACHABLE
    _UNREACHABLE.append(what)
  if first:
    warn(
      f"prebuilt kernels could not be fetched from {url} ({error}): {what} COMPILES now (minutes, once, "
      "cached). Zero compiles needs them once: warm the scene online once, fill a store with "
      f"`raywarp.core.build.kernel_cache.prebuilt.prefetch`, or point {kernel_store.STORE_ENV} at a filled store"
    )


def bundle(arch: int) -> Bundle | None:
  """The kernels bundle of SM version `arch` when this process can use it (checked once per
  process; None: none pinned, unreachable or unusable, which warned)."""
  with _LOCK:
    if arch in _BUNDLES:
      return _BUNDLES[arch]
  found = _resolve(arch)
  with _LOCK:
    return _BUNDLES.setdefault(arch, found)


def _resolve(arch: int) -> Bundle | None:
  """`bundle`'s lookup: the pinned bundle's file list and manifest, from the store, else fetched;
  with `BUNDLE_ENV`, that complete bundle of the store, read as it is."""
  store, own = kernel_store.store_root(), os.environ.get(BUNDLE_ENV)
  pinned = (own, "") if own else kernel_store.pin(arch)
  if store is None or pinned is None:
    return None
  content, sha = pinned
  rel = f"sm{arch}/{content}"
  root, url = store / rel, f"{kernel_store.remote()}/{rel}"
  try:
    if own:
      return _checked(Bundle(root, Manifest.read(root / MANIFEST)), arch)
    files = kernel_store.listing(root, url, sha)
    path = Bundle(root, Manifest(arch=arch, warp="", content=content), url, files).path(MANIFEST)
    return _checked(Bundle(root, Manifest.read(path), url, files), arch)
  except kernel_store.FetchError as e:
    unreachable(url, str(e), f"every kernel of sm_{arch}")
  except (OSError, ValueError, KeyError) as e:
    warn(f"the prebuilt kernels at {root} are not used: damaged manifest ({e})")
  return None


def _checked(b: Bundle, arch: int) -> Bundle | None:
  """`b` when it can serve SM version `arch` (else a warning saying why, and None)."""
  reason = b.manifest.mismatch(arch)
  if reason is not None:
    warn(f"the prebuilt kernels at {b.root} are not used: {reason}")
  return b if reason is None else None


def module_pack(arch: int, driver: str) -> Pack | None:
  """The OptiX module pack of SM version `arch` and NVIDIA driver `driver` (its file list fetched
  once per store; None: none for this driver, or unreachable, warned: OptiX creates each module on
  first use)."""
  with _LOCK:
    if (arch, driver) in _PACKS:
      return _PACKS[arch, driver]
  found = _pack(arch, driver)
  with _LOCK:
    return _PACKS.setdefault((arch, driver), found)


def _pack(arch: int, driver: str) -> Pack | None:
  """`module_pack`'s lookup: `optix/<driver>/` of the bundle and its file list."""
  b = bundle(arch)
  if b is None or not driver:
    return None
  root = b.root / "optix" / driver
  if b.url is None:
    return Pack(root) if root.is_dir() else None
  url = f"{b.url}/optix/{driver}"
  try:
    return Pack(root, url, kernel_store.listing(root, url, None))
  except kernel_store.FetchError as e:
    warn(
      f"no OptiX module pack for NVIDIA driver {driver} at {url} ({e}): OptiX creates each module on first use "
      "(seconds to minutes, once per machine and driver)"
    )
    return None


def seed_module(arch: int, name: str, db: Path) -> bool:
  """Merges OptiX module `name`'s rows (`module_path`, `trace/optix/modules.digest`) of the running
  driver's pack into OptiX's disk-cache database `db`, fetched first when the store lacks them:
  True, OptiX then finds the module (a cache hit, once it reads its cache again:
  `trace/optix/context.DeviceContext.reload_cache`). False when the pack lacks it (another driver, a
  combination outside the census: OptiX creates it) or when its file cannot be fetched (warned
  once per process) or merged (warned)."""
  p = module_pack(arch, driver_version())
  rel = module_path(name)
  if p is None or not p.holds(rel):
    return False
  try:
    merge_optix(p.path(rel), db)
  except kernel_store.FetchError:
    return False  # warned once (`unreachable`)
  except OSError as e:
    warn(f"the OptiX module {name} of the pack at {p.root} was not merged ({e}): OptiX creates it now")
    return False
  return True


def driver_version() -> str:
  """The running NVIDIA driver's version (e.g. "580.95.05"; "" when unknown: no pack is used)."""
  try:
    text = _DRIVER_FILE.read_text()
  except OSError:
    return ""
  for word in text.split():
    parts = word.split(".")
    if len(parts) >= 2 and all(p.isdigit() for p in parts):
      return word
  return ""


def ptx(arch: int, key: str, *, expected: bool) -> bytes | None:
  """The bundled PTX of key `key` for SM version `arch`'s device (None: none), fetched when the
  store lacks it. `expected`: the build is one the bundle holds (`tiers.prebuilt`); then a missing
  bundle (once per process) or entry warns that it compiles once. A damaged entry always warns."""
  b = bundle(arch)
  if b is None:
    with _LOCK:
      first = expected and arch not in _WARNED
      if first:
        _WARNED.add(arch)
    if first:
      warn(f"no prebuilt kernels for sm_{arch}: each kernel build compiles once on first use (cached)")
    return None
  try:
    entry = b.entry(key)
    if entry is None:
      if expected:
        warn(
          f"the prebuilt bundle {b.root} lacks the build {key} of this kernel code: compiling it once (cached); a "
          "supported configuration should never miss (please report it)"
        )
      return None
    return lzma.decompress(b.path(f"ptx/{entry.ptx}").read_bytes())
  except kernel_store.FetchError:
    return None  # warned once (`unreachable`)
  except (OSError, lzma.LZMAError, ValueError) as e:
    warn(f"the prebuilt PTX of {key} in {b.root} is damaged ({e}): compiling it once (cached); delete it to fetch it again")
    return None


def prefetch(arch: int, *, driver: str = "") -> Path:
  """Fills the kernel store with the whole bundle of SM version `arch` and NVIDIA driver `driver`'s
  OptiX module pack (default: this machine's; none without a driver): an offline mirror for
  machines without network (copy the store, point RAYWARP_KERNELS at it). No GPU needed. What a
  scene uses alone is fetched by warming it (`Renderer.warm`: the renderer's own loads). The
  bundle's store directory.

  Raises:
    RuntimeError: no usable bundle for `arch`, or files that could not be fetched (warned).
  """
  b = bundle(arch)
  if b is None:
    raise RuntimeError(f"no usable prebuilt kernels for sm_{arch} (warned above)")
  failed = b.fetch(b.files.files if b.files is not None else ())
  driver = driver or driver_version()
  p = module_pack(arch, driver) if driver else None
  failed |= p.fetch_all() if p is not None else {}
  if failed:
    raise RuntimeError(f"{len(failed)} prebuilt kernel files not fetched, e.g. {next(iter(failed.items()))}")
  return b.root


def fetch_modules(arch: int, names: Iterable[str]) -> int:
  """Places the bundled Warp modules `names` (Warp's module directory names) that Warp's kernel
  cache lacks into it: fetched into the store in one parallel batch, then hard-linked (copied
  across file systems); how many were placed. Names the bundle lacks are left to Warp (they
  compile); a failed fetch warned once (`unreachable`)."""
  b, kernels = bundle(arch), wp.config.kernel_cache_dir
  if b is None or kernels is None:
    return 0
  listed = b.module_files()
  wanted = [n for n in dict.fromkeys(names) if n in listed and not (Path(kernels) / n).is_dir()]
  if not wanted:
    return 0
  failed = b.fetch(rel for n in wanted for rel in listed[n])
  if failed and b.url is not None:
    unreachable(b.url, next(iter(failed.values())), f"{len(failed)} Warp module files")
  placed = 0
  for name in wanted:
    if not any(rel in failed for rel in listed[name]):
      placed += _place(b.root / "warp" / name, Path(kernels) / name)
  return placed


def _place(src: Path, dst: Path) -> int:
  """Links the files of module directory `src` into Warp's cache as `dst` (a directory complete or
  absent: built under a temporary name, then renamed); 1 when placed, 0 when another process did."""
  dst.parent.mkdir(parents=True, exist_ok=True)
  scratch = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent))
  try:
    for f in src.iterdir():
      try:
        os.link(f, scratch / f.name)
      except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
          raise
        shutil.copy2(f, scratch / f.name)  # another file system: hard links cannot cross it
    scratch.rename(dst)
    return 1
  except OSError:
    if dst.is_dir():
      return 0  # another process placed it first
    raise
  finally:
    shutil.rmtree(scratch, ignore_errors=True)


@contextlib.contextmanager
def locked(path: Path, *, wait: bool = True) -> Generator[bool]:
  """Holds an exclusive lock on file `path` (created) for the block, across processes: True; with
  `wait` False, False at once when another holder has it."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as f:
    try:
      fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BlockingIOError:
      yield False
      return
    try:
      yield True
    finally:
      fcntl.flock(f, fcntl.LOCK_UN)


def seed(arch: int) -> None:
  """Checks the bundle of SM version `arch` (its file list and manifest, fetched once per store)
  and places its common Warp modules in Warp's kernel cache (`common_modules`: fetched once per
  store), once per process (nothing without a bundle or a cache directory). Every caller returns
  after it (`_SEED_LOCK`): a thread loading a module meanwhile finds it placed. OptiX modules are
  seeded one at a time, when created (`seed_module`)."""
  with _SEED_LOCK:
    if arch not in _SEEDED:
      _seed(arch)
      _SEEDED.add(arch)


def common_modules(b: Bundle) -> list[str]:
  """The Warp modules of `b` every renderer may load: all but the specialisation families of more
  than `FAMILY_FETCHED_BY_NAME` variants (a module per Features combination: the GI relight
  kernels, 2.3 GB of a 2.4 GB bundle), which are fetched by name when a renderer lists them
  (`fetch_modules`, from `warmup`). The common ones are small (sm120: 556 modules, 13 MB
  compressed), so no first frame compiles one that its renderer's list missed."""
  families: dict[str, list[str]] = {}
  for name in b.module_files():
    families.setdefault(_FAMILY.sub("", name), []).append(name)
  return [n for names in families.values() if len(names) <= FAMILY_FETCHED_BY_NAME for n in names]


def _seed(arch: int) -> None:
  """`seed`'s common modules (its caller holds `_SEED_LOCK`)."""
  b = bundle(arch)
  if cache_dir() is None or b is None:
    return
  common = common_modules(b)
  if b.url is not None and b.files is not None and kernel_store.COMMON in b.files.files:
    missing = [rel for n in common for rel in b.module_files()[n] if not (b.root / rel).is_file()]
    failed = kernel_store.unpack(b.root, b.url, b.files, kernel_store.COMMON) if missing else {}
    if failed:
      unreachable(b.url, next(iter(failed.values())), f"{len(failed)} common Warp module files")
  fetch_modules(arch, common)


def optix_cache_dir(user: Path) -> Path:
  """OptiX's module cache directory under raywarp's cache directory `user` (where
  `trace/optix/context.device_context` points OptiX): OPTIX_CACHE_PATH, else `user`/optix."""
  return Path(os.environ.get(OPTIX_CACHE_ENV) or user / "optix")


def merge_optix(src: Path, dst: Path) -> None:
  """Adds the rows of OptiX disk cache `src` that `dst` lacks (`dst` created as a copy of `src`
  when absent), in a child process (`optix_merge`: this process's OptiX may hold `dst` open
  through its own SQLite, which a connection of Python's would break).

  Raises:
    OSError: the merge failed (the child's last error line attached) or did not end in time.
  """
  try:
    done = subprocess.run(
      [sys.executable, "-I", optix_merge.__file__, str(src), str(dst)],
      capture_output=True,
      text=True,
      timeout=_MERGE_TIMEOUT_S,
      check=False,
    )
  except subprocess.TimeoutExpired as e:
    raise OSError(f"merging {src.name} into {dst} took over {_MERGE_TIMEOUT_S} s") from e
  if done.returncode:
    last = (done.stderr.strip().splitlines() or [f"exit {done.returncode}"])[-1]
    raise OSError(f"merging {src.name} into {dst} failed: {last}")


# Seconds a merge may take: its busy timeout (`optix_merge.BUSY_TIMEOUT_S`) and the copy, with room.
_MERGE_TIMEOUT_S: Final = 4 * optix_merge.BUSY_TIMEOUT_S
