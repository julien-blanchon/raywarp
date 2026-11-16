r"""The `.wpbr` scene cache: one file of raw arrays, loaded with a single memory map.

Layout (little endian)::

  [0, 16)        magic b"WPBR" + format version (uint32) + 8 reserved bytes
  [256, ...)     every array of the scene, raw C-order bytes, each 256-byte aligned
  ...            JSON header: {"version", "arrays": {name: [dtype, shape, offset]}, "report",
  "meta"}
  last 24 bytes  header offset (uint64), header length (uint64), b"WPBREND\0"

The header is written last, so arrays (in particular the texel blob) can be written in place
while images decode, and a crash leaves no valid file: each writer writes its own temporary file
`<name>.tmp-<host>-<pid>-<token>` next to the destination and renames it into place on success;
`cache.paths.remove_stale` deletes the ones of crashed writers. A load reads the trailer and
header, maps the file once and takes a zero-copy view per array: its cost does not depend on the
object count.

Freshness (any adapter): a cached scene records what it was built from in its metadata, a
`CacheKey` (adapter, source file name, options, the converter's version string, the versions
of the libraries it reads with) and its `Dependency` list (every file read, missing
ones included, with mtime and size, relative to the source's directory). `fresh` compares both on
load, the files through one listing per directory (`list_files`: a network mount serves a listing
in one round trip, a stat each in one of their own). Read on another machine, a cached scene
stays fresh where its sources sit at another path but keep their names, sizes and mtimes.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import functools
import hashlib
import json
import math
import os
import stat
import struct
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import BinaryIO, Final, Literal, final

import numpy as np

from raywarp.cache.paths import DEFAULT_DIR, cache_dir, remove_stale, tmp_name
from raywarp.scene.model.types import ALIGN, Array, Json, itemsize

MAGIC: Final = b"WPBR"
END: Final = b"WPBREND\0"
# 2: Lights.instance; 3: second UV set, vertex colors; 4: meta["cache"] (key + dependencies),
# emissive panel lights no longer stored (added after the load); 5: Instances.source;
# 6: Cameras.exposure, Instances.label / labels (UsdSemantics); 7: canonical OpenPBR materials
# (Materials: the OpenPBR inputs, ext_*, extras; MaterialExtras); 8: Lights / Environments: the
# UsdLux inputs; 9: Materials.ext_normal_scale removed (normal-map
# strengths and flips are the GEOMETRY_NORMAL binding's scale / bias: scene/model/normal_map.py);
# 10: IES profiles (Lights shaping IES fields); 11: Environments.illuminance;
# 12: Images.digest (content digests of the images), the cache key's libraries, dependencies
# recorded relative to the source's directory; 13: Lights.visible_in_primary_ray; 14: images of
# equal content share their bytes (`Images.offset`), the texel blob is the last array, and the
# optional corner indices (`Meshes.uv1_idx` / `color_idx`) keep the rows of the meshes with them.
VERSION: Final = 14
_TRAILER: Final = struct.Struct("<QQ8s")
# Bytes per read of an in-file move (`Writer.move`).
_MOVE_CHUNK: Final = 64 << 20
# Parallel stats of a freshness check (network file systems serve several at a time).
STAT_THREADS: Final = 32

# What a freshness check compares with the files now: every dependency or the source (`fresh`).
type Verify = Literal["files", "source"]
# Array layout of a file: name -> [dtype string, shape, byte offset].
type Layout = dict[str, tuple[str, list[int], int]]


def align(n: int) -> int:
  """`n` rounded up to a multiple of ALIGN."""
  return (n + ALIGN - 1) // ALIGN * ALIGN


@final
class Writer:
  """Writes a `.wpbr` file whose arrays have known dtypes and shapes, possibly from threads.

  Arrays are written with `os.pwrite` (thread-safe, releases the GIL). Writing through a
  writable memory map would be slower on macOS: first-touch page faults, then ~0.75 s per GB
  to unmap the dirty pages.
  """

  def __init__(self, path: str | os.PathLike[str], arrays: Mapping[str, Array], last: str | None = None) -> None:
    """Creates the temporary file of `path`, sized for `arrays` (only their dtypes and shapes are
    read); array `last` is laid out after the others (it may then `shrink`)."""
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale(self.path)
    self.tmp = self.path.with_name(tmp_name(self.path))  # same file system: an atomic rename
    self.layout: Layout = {}
    self.last = last
    offset = ALIGN
    for name in sorted(arrays, key=lambda n: n == last):  # stable: `last` moves to the end
      a = arrays[name]
      self.layout[name] = (a.dtype.str, list(a.shape), offset)
      offset = align(offset + a.nbytes)
    self.end = offset
    self.fd = os.open(self.tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
      self.pwrite(MAGIC + struct.pack("<I", VERSION) + bytes(8), 0)
      os.ftruncate(self.fd, self.end)
    except BaseException:
      self.abort()
      raise

  def offset(self, name: str) -> int:
    """Absolute file offset of array `name`."""
    return self.layout[name][2]

  def write(self, name: str, array: Array, byte_offset: int = 0) -> None:
    """Writes `array`'s bytes at `byte_offset` inside array `name`."""
    self.pwrite(np.ascontiguousarray(array), self.offset(name) + byte_offset)

  def move(self, name: str, source: int, destination: int, nbytes: int) -> None:
    """Copies `nbytes` bytes of array `name` from byte `source` down to byte `destination` (below
    it: copied forward in chunks, overlaps included)."""
    if destination > source:
      raise ValueError(f"{name}: a move up ({source} -> {destination}) would overwrite its own source")
    base = self.offset(name)
    done = 0
    while done < nbytes:
      chunk = os.pread(self.fd, min(_MOVE_CHUNK, nbytes - done), base + source + done)
      if not chunk:
        raise OSError(f"{self.tmp}: read past the end at byte {base + source + done}")
      self.pwrite(chunk, base + destination + done)
      done += len(chunk)

  def shrink(self, name: str, nbytes: int) -> None:
    """Cuts the last array `name` (one dimension) to its first `nbytes` bytes."""
    dtype, shape, offset = self.layout[name]
    if name != self.last or len(shape) != 1:
      raise ValueError(f"{name}: only the last, one-dimensional array of a file shrinks")
    size = itemsize(np.dtype(dtype))
    count = nbytes // size
    self.layout[name] = (dtype, [count], offset)
    self.end = align(offset + count * size)
    os.ftruncate(self.fd, self.end)

  def pwrite(self, data: Array | bytes, offset: int) -> None:
    """Writes all of `data` at file `offset`."""
    view = memoryview(data).cast("B")
    while len(view):
      n = os.pwrite(self.fd, view, offset)
      if n == 0:
        raise OSError(f"{self.tmp}: write made no progress at byte {offset}")
      view, offset = view[n:], offset + n

  def close(self, header: Mapping[str, Json]) -> None:
    """Appends the JSON header (plus the version and the array layout) and moves the file into
    place (no fsync: it is a cache)."""
    data = json.dumps({**header, "version": VERSION, "arrays": self.layout}, separators=(",", ":")).encode()
    self.pwrite(data + _TRAILER.pack(self.end, len(data), END), self.end)
    os.close(self.fd)
    self.fd = -1  # closed: `abort` must not close the number again (another file may reuse it)
    try:
      self.tmp.replace(self.path)
    finally:
      self.tmp.unlink(missing_ok=True)

  def abort(self) -> None:
    """Deletes the partial file."""
    if self.fd >= 0:
      with contextlib.suppress(OSError):
        os.close(self.fd)
      self.fd = -1
    self.tmp.unlink(missing_ok=True)


def save(
  path: str | os.PathLike[str],
  arrays: Mapping[str, Array],
  header: Callable[[], Mapping[str, Json]],
  *,
  write: Mapping[str, Callable[[Writer], None]] | None = None,
) -> None:
  """Writes `arrays` and the JSON `header()` (called once the arrays are written) to `path`.

  `write`: arrays the caller writes itself, name -> `fn(writer)` (e.g. the texel blob, decoded
  straight into the file); `arrays` holds them as placeholders of their final dtype and shape
  (the first one's at most: it is laid out last and may `Writer.shrink`). The partial file is
  deleted on error.
  """
  callers = {} if write is None else write
  writer = Writer(path, arrays, last=next(iter(callers), None))
  try:
    for fn in callers.values():
      fn(writer)
    for name, array in arrays.items():
      if array.size and name not in callers:
        writer.write(name, array)
    writer.close(header())
  except BaseException:
    writer.abort()
    raise


def read_header(path: str | os.PathLike[str]) -> dict[str, Json]:
  """The JSON header of a `.wpbr` file (raises ValueError if it is not a complete one of this
  format version)."""
  with Path(path).open("rb") as file:
    header, _ = _any_header(file)
  if header.get("version") != VERSION:
    raise ValueError(f"{path}: .wpbr version {header.get('version')} (expected {VERSION})")
  return header


def _any_header(file: BinaryIO) -> tuple[dict[str, Json], int]:
  """JSON header and array-region end of an open `.wpbr`, any format version (else ValueError).

  Read and map the same descriptor: an atomic replacement of its path must not mix generations.
  """
  if file.read(4) != MAGIC:
    raise ValueError(f"{file.name}: not a .wpbr file")
  size = file.seek(0, os.SEEK_END)
  if size < ALIGN + _TRAILER.size:
    raise ValueError(f"{file.name}: truncated .wpbr file")
  file.seek(-_TRAILER.size, os.SEEK_END)
  offset, length, end = _TRAILER.unpack(file.read(_TRAILER.size))
  if end != END or offset < ALIGN or offset + length != size - _TRAILER.size:
    raise ValueError(f"{file.name}: invalid .wpbr header bounds")
  file.seek(offset)
  header = json.loads(file.read(length))
  if not isinstance(header, dict):
    raise ValueError(f"{file.name}: malformed .wpbr header")
  return header, offset


def _layout(header: Mapping[str, Json]) -> Layout:
  """The array layout of a header (ValueError when malformed)."""
  found = header.get("arrays")
  if not isinstance(found, dict):
    raise ValueError("a .wpbr header without arrays")
  arrays: dict[str, Json] = found  # annotated: pyrefly loses the element type of a narrowed Json
  layout: Layout = {}
  for name, entry in arrays.items():
    match entry:
      case [str(dtype), list(shape), int(offset)] if type(offset) is int and all(type(n) is int and n >= 0 for n in shape):
        layout[name] = (dtype, [n for n in shape if isinstance(n, int)], offset)
      case _:
        raise ValueError(f"a .wpbr header with a malformed array entry {name}")
  return layout


def load(path: str | os.PathLike[str]) -> tuple[dict[str, Array], dict[str, Json]]:
  """(arrays, header) of a `.wpbr` file: one memory map, one zero-copy view per array."""
  with Path(path).open("rb") as file:
    header, data_end = _any_header(file)
    if header.get("version") != VERSION:
      raise ValueError(f"{path}: .wpbr version {header.get('version')} (expected {VERSION})")
    mm = np.memmap(file, dtype=np.uint8, mode="r")
  return _views(path, header, data_end, mm), header


def load_fresh(
  path: str | os.PathLike[str], key: CacheKey, *, verify: Verify = "files"
) -> tuple[dict[str, Array], dict[str, Json]] | None:
  """`load` of `.wpbr` file `path` when it is `fresh` for `key` (`verify`: see `fresh`), else
  None: the header is read and parsed once, the file mapped through the descriptor read from."""
  try:
    with Path(path).open("rb") as file:
      header, data_end = _any_header(file)
      if header.get("version") != VERSION or not _fresh_meta(header.get("meta"), key, verify):
        return None
      mm = np.memmap(file, dtype=np.uint8, mode="r")
  except (OSError, ValueError):
    return None
  return _views(path, header, data_end, mm), header


def _views(path: str | os.PathLike[str], header: Mapping[str, Json], data_end: int, mm: Array) -> dict[str, Array]:
  """One zero-copy view per array of a header's layout into the file's memory map `mm`."""
  arrays: dict[str, Array] = {}
  for name, (dtype_str, shape, offset) in _layout(header).items():
    try:
      dtype = np.dtype(dtype_str)
    except TypeError as exc:
      raise ValueError(f"{path}: invalid dtype {dtype_str!r} of array {name}") from exc
    size = itemsize(dtype)
    nbytes = math.prod(shape) * size
    hasobject = dtype.hasobject  # pyrefly: ignore[missing-attribute]  # numpy stubs omit this dtype flag
    if hasobject or size == 0 or offset < ALIGN or offset % ALIGN or offset + nbytes > data_end:
      raise ValueError(f"{path}: invalid dtype or byte range of array {name}")
    arrays[name] = mm[offset : offset + nbytes].view(dtype).reshape(shape)
  return arrays


# Freshness ------------------------------------------------------------------------------------


@functools.cache
def installed_versions(*distributions: str) -> dict[str, str]:
  """Distribution name -> installed version ("none" when absent), from the `*.dist-info` (or
  `*.egg-info`) directories of `sys.path` (the first found, as the import system;
  `importlib.metadata` costs ~50 ms to import)."""
  want = {_normalized(d): d for d in distributions}
  found: dict[str, str] = {}
  for entry in sys.path:
    try:
      names = os.listdir(entry or ".")
    except OSError:
      continue
    for name in names:
      if name.endswith((".dist-info", ".egg-info")):
        project, _, version = name.removesuffix(".dist-info").removesuffix(".egg-info").partition("-")
        version = version.split("-py")[0]  # an .egg-info name may end in -py3.12
        key = _normalized(project)
        if key in want and want[key] not in found:
          found[want[key]] = version
  return {d: found.get(d, "none") for d in distributions}


def _normalized(name: str) -> str:
  """A distribution name normalized as in its `.dist-info` directory name (PEP 503 / 427)."""
  return "_".join(part for part in name.lower().replace("-", "_").replace(".", "_").split("_") if part)


@final
@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class CacheKey:
  """What a cached scene is built from, besides its files: compared on every load.

  Attributes:
    adapter: the adapter that read the source ("usd").
    source: absolute path of the source. Its directory is the root the dependencies are recorded
      relative to; only its file name is compared (a cached scene moved with its sources, or read
      on another machine mounting them elsewhere, stays fresh).
    options: the adapter and build options that change the scene (JSON values).
    code: the version string of the converter that builds the scene (bumped by hand when its
      output changes, e.g. `formats.usd.version.CONVERTER`).
    libraries: distribution -> version of the third-party libraries the build reads with.
  """

  adapter: str
  source: str
  options: dict[str, Json]
  code: str
  libraries: dict[str, str] = dataclasses.field(default_factory=dict)

  @property
  def root(self) -> Path:
    """The directory the dependencies are recorded relative to (the source's)."""
    return Path(self.source).parent

  def to_json(self) -> dict[str, Json]:
    """The compared JSON form (stored in the scene metadata, `meta["cache"]["key"]`)."""
    libraries: dict[str, Json] = dict(self.libraries)
    return {
      "adapter": self.adapter,
      "source": Path(self.source).name,
      "options": self.options,
      "code": self.code,
      "libraries": libraries,
    }

  def file_name(self) -> str:
    """`<source stem>-<digest of the adapter, source path and options>.wpbr`: a code or library
    change rebuilds the same file (`fresh` compares the whole key) instead of leaving a stale copy
    behind."""
    named = {"adapter": self.adapter, "source": self.source, "options": self.options}
    digest = hashlib.sha1(json.dumps(named, sort_keys=True).encode()).hexdigest()[:16]
    return f"{Path(self.source).stem}-{digest}.wpbr"


def cache_file(
  key: CacheKey,
  cache: str | os.PathLike[str] | bool,  # noqa: FBT001 (a path or True: the adapters' `cache` option)
) -> Path:
  """The `.wpbr` file of `key`: `cache` itself when it names one, else a file in directory `cache`.

  `cache` True means `cache_dir()` (DEFAULT_DIR when caching is off).
  """
  if isinstance(cache, bool):
    if not cache:
      raise ValueError("cache=False names no file")
    return (cache_dir() or Path(DEFAULT_DIR).expanduser()) / key.file_name()
  if os.fspath(cache).endswith(".wpbr"):
    return Path(cache).expanduser()
  return Path(cache).expanduser() / key.file_name()


@final
@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Dependency:
  """A file a scene was read from: absolute path, mtime and size (-1, -1: missing when read)."""

  path: str
  mtime_ns: int
  size: int

  def to_json(self, root: Path) -> list[Json]:
    """[path, mtime_ns, size]: the path relative to `root` for a file under it, else absolute (a
    file outside the source's tree is the same file wherever the tree is mounted)."""
    relative = os.path.relpath(self.path, root)
    inside = relative != os.pardir and not relative.startswith(os.pardir + os.sep)
    return [relative if inside else self.path, self.mtime_ns, self.size]


def _stat(path: str) -> Dependency:
  """The `Dependency` of `path` now (a missing or non-regular file is -1, -1)."""
  try:
    st = Path(path).stat()
  except OSError:
    return Dependency(path=path, mtime_ns=-1, size=-1)
  if not stat.S_ISREG(st.st_mode):
    return Dependency(path=path, mtime_ns=-1, size=-1)
  return Dependency(path=path, mtime_ns=st.st_mtime_ns, size=st.st_size)


def stat_files(paths: Iterable[str]) -> list[Dependency]:
  """`Dependency` of every path (sorted, unique), stat-ed in parallel (one synchronous pass)."""
  unique = sorted(set(paths))
  if len(unique) <= 1:
    return [_stat(p) for p in unique]
  with concurrent.futures.ThreadPoolExecutor(min(STAT_THREADS, len(unique))) as pool:
    return list(pool.map(_stat, unique))


def record(
  meta: dict[str, Json],
  key: CacheKey,
  dependencies: Iterable[str],
  *,
  known: Mapping[str, Dependency] | None = None,
  since_ns: int | None = None,
) -> None:
  """Stores `key` and the current state of `dependencies` (absolute paths) in scene metadata
  `meta`, the paths of files under `key.root` relative to it (`Dependency.to_json`).

  `known`: states already taken during this build (`prefetch.Prefetch.done`), not stat-ed again.
  `since_ns`: when the build started reading (`time.time_ns()`); a file modified since may have
  been read before its edit, so it is recorded as unknown (the next load rebuilds): a state taken
  here, or a known one (a prefetch's read is not the stage's: it may follow the edit the stage's
  own read preceded). File systems stamp times at their tick, never later: an edit made before the
  start is never flagged; one within a tick after it can be missed.
  """
  known = {} if known is None else known
  paths = sorted(set(dependencies))
  stats = {d.path: d for d in stat_files(p for p in paths if p not in known)}
  states = {p: known[p] if p in known else stats[p] for p in paths}
  if since_ns is not None:
    states = {p: Dependency(path=p, mtime_ns=-1, size=-1) if d.mtime_ns >= since_ns else d for p, d in states.items()}
  rows: list[Json] = [states[p].to_json(key.root) for p in paths]
  meta["cache"] = {"key": key.to_json(), "dependencies": rows}


def recorded_dependencies(meta: Mapping[str, Json], root: Path) -> list[Dependency]:
  """The dependencies `record` stored in `meta`, as absolute paths (relative ones under `root`;
  none when it holds no cache record)."""
  entry = meta.get("cache")
  found = entry.get("dependencies") if isinstance(entry, dict) else None
  rows: list[Json] = found if isinstance(found, list) else []  # annotated as in `_layout`
  out: list[Dependency] = []
  for row in rows:
    match row:
      case [str(path), int(mtime), int(size)] if type(mtime) is int and type(size) is int:
        out.append(Dependency(path=os.path.normpath(os.path.join(root, path)), mtime_ns=mtime, size=size))
  return out


def stale_dependencies(path: str | os.PathLike[str], root: Path) -> list[Dependency]:
  """The dependencies recorded in `.wpbr` file `path` (absolute under `root`), of any format
  version (none when it is missing or holds no record): what a rebuild of it will read again."""
  try:
    with Path(path).open("rb") as file:
      header, _ = _any_header(file)
    meta = header.get("meta")
  except (OSError, ValueError):
    return []
  return recorded_dependencies(meta, root) if isinstance(meta, dict) else []


def fresh(path: str | os.PathLike[str], key: CacheKey, *, verify: Verify = "files") -> bool:
  """Whether `.wpbr` file `path` exists, was built for `key` and none of its files changed.

  `verify`: "files" compares every recorded dependency with the files now; "source" only the source
  itself (one stat), for caches whose other inputs the caller has checked or knows immutable (e.g.
  prepared scenes checked against a read-only bucket listing): on a network mount a stat costs a
  round trip each (904 files: 0.25 to 14 s on a FUSE bucket mount).
  """
  try:
    meta = read_header(path).get("meta")
  except (OSError, ValueError):
    return False
  return _fresh_meta(meta, key, verify)


def _fresh_meta(meta: Json, key: CacheKey, verify: Verify) -> bool:
  """Whether a header's `meta` records `key` and dependencies that are unchanged now (all of them,
  or the source's only: `fresh`)."""
  entry = meta.get("cache") if isinstance(meta, dict) else None
  if not isinstance(meta, dict) or not isinstance(entry, dict) or entry.get("key") != key.to_json():
    return False
  recorded = recorded_dependencies(meta, key.root)
  rows = entry.get("dependencies")
  if not isinstance(rows, list) or len(recorded) != len(rows):
    return False
  if verify == "source":
    source = os.path.normpath(key.source)
    recorded = [d for d in recorded if d.path == source]
    if len(recorded) != 1:
      return False
  return all(state == d for d, state in zip(recorded, list_files(d.path for d in recorded), strict=True))


def list_files(paths: Iterable[str]) -> list[Dependency]:
  """`Dependency` of every path (in order), read from one listing per directory, the directories
  listed in parallel.

  A network file system (FUSE bucket mounts) serves a listing with its entries' attributes in one
  round trip but each first stat of a file in a round trip of its own, which it does not overlap:
  904 files in 6 directories take 0.25 s listed and ~16 s stat-ed on a bucket mount.
  """
  wanted = list(paths)
  by_dir: dict[str, set[str]] = {}
  for p in wanted:
    by_dir.setdefault(os.path.dirname(p), set()).add(os.path.basename(p))
  if not by_dir:
    return []
  workers = min(STAT_THREADS, len(by_dir))
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    listed = dict(zip(by_dir, pool.map(_listing, by_dir.items()), strict=True))
  missing = (-1, -1)
  out: list[Dependency] = []
  for p in wanted:
    mtime, size = listed[os.path.dirname(p)].get(os.path.basename(p), missing)
    out.append(Dependency(path=p, mtime_ns=mtime, size=size))
  return out


def _listing(item: tuple[str, set[str]]) -> dict[str, tuple[int, int]]:
  """The (mtime_ns, size) by name of the regular files among `names` in `directory` (symbolic
  links followed, as `_stat`); a directory that cannot be listed holds none."""
  directory, names = item
  out: dict[str, tuple[int, int]] = {}
  try:
    with os.scandir(directory) as entries:
      for entry in entries:
        if entry.name in names:
          with contextlib.suppress(OSError):
            st = entry.stat()
            if stat.S_ISREG(st.st_mode):
              out[entry.name] = (st.st_mtime_ns, st.st_size)
  except OSError:
    pass
  return out
