"""Disk caches of the expensive precomputations (light grid, GI probes, environment
prefilter), so that a second run with the same inputs loads them instead of
recomputing them.

The light grid, the GI probe volume and an environment map's prefilter / SH
are pure functions of their inputs: the kernel structs they read (geometry, materials, lights,
environment), a few host values (settings, volume placement) and the code. Each bake is stored as
`<cache dir>/bakes/<kind>/<digest>.npz`, the digest (`Digest`) over every one of those inputs,
the sources of `CODE_PACKAGES` (`code_version`), the Warp version (its code generation) and the
device kind (CPU and CUDA round differently), so a hit restores exactly the arrays an uncached
build computes (pixel-identical). Host values are hashed by value (Warp matrices by their
entries; sets and mappings in a fixed order, dataclasses field by field); a value whose repr holds
an object address (objects, functions) is not hashable (`UncacheableError`).

Struct arrays are hashed by content (a device -> host copy, once per bake), or by the content
digest their creator registered for an array it never writes again (`register_array`: the
upload's mesh tables); texture handles by the digest their creator registered
(`register_texture`); BVH / OptiX handles are not hashed (they derive from hashed arrays). A
struct holding an unregistered texture is not cacheable: the bake then runs as without a cache.
Writes go to a temporary file of their own (a unique name per writer: concurrent threads and
processes never share one) renamed into place, so a reader never sees a partial file. The
directory is `cache_dir()` (`RAYWARP_CACHE_DIR`; empty: caching off).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import operator
import re
import struct
import threading
import warnings
import weakref
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, final

import numpy as np
import numpy.lib.recfunctions as rfn
import warp as wp

from raywarp.cache.code_version import code_version
from raywarp.cache.paths import cache_dir, write_atomic
from raywarp.core.pipeline.platform.device import device_caps, host_copy

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray, AnyHostArray

# Bumped when the stored layout of a bake changes.
FORMAT: Final = 1
# The packages whose code keys every bake: the renderer core, the shared types and the settings.
CODE_PACKAGES: Final = ("raywarp.core", "raywarp._base", "raywarp.settings")
# A repr that names an object by its address (`<... object at 0x...>`, `<function f at 0x...>`):
# not a value (another process, or another object later at the same address, would share a key).
_ADDRESS: Final = re.compile(r" at 0x[0-9a-fA-F]+")
# Textures by handle id (resolves texture arrays) and their registered content digests (weak:
# entries live as long as their textures).
_textures: weakref.WeakValueDictionary[int, wp.Texture] = weakref.WeakValueDictionary()
_digests: weakref.WeakKeyDictionary[wp.Texture, str] = weakref.WeakKeyDictionary()

type Bake = dict[str, AnyHostArray]


def register_texture[T: wp.Texture](texture: T, digest: str) -> T:
  """Records `texture`'s content digest (texels and sampling options) for bake keys; returns it."""
  _digests[texture] = digest
  _textures[texture.id] = texture
  return texture


# id(array) -> (weak reference, content digest) of device arrays registered by their creator
# (`register_array`; by id: Warp arrays are not hashable by value).
# Re-entrant: a garbage collection the registering thread triggers runs the weak references'
# callbacks (`forget`) on that thread.
_arrays: dict[int, tuple[weakref.ref[wp.array], str]] = {}
_arrays_lock = threading.RLock()


def array_digest(host: AnyHostArray) -> str:
  """Content digest of a host array (dtype, shape, contents): what `register_array` records for
  a device copy of it."""
  a = np.ascontiguousarray(host)
  h = hashlib.blake2b(digest_size=16)
  h.update(repr((a.dtype.str, a.shape)).encode())
  h.update(a.data)
  return h.hexdigest()


def register_array[A: wp.array](array: A, digest: str) -> A:
  """Records the content digest (`array_digest` of the host array it was copied from) of a device
  array its creator never writes again, so bake keys hash the digest instead of copying the
  array back to the host and hashing its contents; returns it. Forgotten when the array dies."""
  key = id(array)

  def forget(_ref: weakref.ref[wp.array]) -> None:
    """Drops the entry of the dead array (unless the id was registered again since)."""
    with _arrays_lock:
      if _arrays.get(key, (None,))[0] is _ref:
        del _arrays[key]

  entry = (weakref.ref(array, forget), digest)  # allocated outside the lock (a collection may run)
  with _arrays_lock:
    _arrays[key] = entry
  return array


def _registered(array: wp.array) -> str | None:
  """The registered content digest of a device array (`register_array`), or None."""
  found = _arrays.get(id(array))
  return found[1] if found is not None and found[0]() is array else None


def texture_digest(texels: AnyHostArray, *options: object) -> str:
  """Digest of a texture's texels and creation options (`register_texture`)."""
  h = hashlib.blake2b(digest_size=16)
  h.update(repr((texels.dtype.str, texels.shape, options)).encode())
  h.update(np.ascontiguousarray(texels).data)
  return h.hexdigest()


# (table name, device) -> its `table_texture`: lookup tables of constants, kept for the process
# (a few small textures per device). The lock makes the first creation per key happen once.
_tables: dict[tuple[str, str], wp.Texture2D] = {}
_tables_lock = threading.Lock()


def table_texture(name: str, texels: Callable[[], AnyHostArray], device: wp.Device) -> wp.Texture2D:
  """The bilinear, edge-clamped 2D texture of lookup table `name` on `device` (`texels`: its
  (rows, columns, channels) values, read on the first call per device; registered for bake keys
  with its digest)."""
  key = (name, str(device))
  with _tables_lock:
    if key not in _tables:
      table = texels()
      texture = wp.Texture2D(
        table, filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP, device=device
      )
      _tables[key] = register_texture(texture, texture_digest(table, name))
    return _tables[key]


@final
class UncacheableError(Exception):
  """A bake input that cannot be hashed (an unregistered texture, a value without a value repr)."""


def _texture(handle: int | wp.Texture) -> str:
  """The registered digest of a texture (object or handle id).

  Raises:
    UncacheableError: for a texture without a registered digest.
  """
  tex = handle if isinstance(handle, wp.Texture) else _textures.get(handle)
  digest = None if tex is None else _digests.get(tex)
  if digest is None:
    raise UncacheableError(f"texture {handle!r} has no registered digest")
  return digest


class _StructVar(Protocol):
  """A field of a Warp struct type (`warp.codegen.Var`: not in the stubs)."""

  type: object


def _struct_vars(s: object) -> Mapping[str, _StructVar]:
  """Field name -> variable of a Warp struct instance's type, in declaration order."""
  # Warp has no public field table of a struct instance (stub gap WS8 hides the private one).
  return type(s)._cls.vars  # pyrefly: ignore[missing-attribute]  # stub gap WS8


@final
class Digest:
  """Incremental digest of a bake's inputs (see the module doc)."""

  def __init__(self, kind: str, device: wp.Device | None) -> None:
    """Starts with the bake kind, the format, the core sources and the device kind (`device`
    None: the kind only, the digest of one value inside another digest)."""
    self.kind = kind
    self._h = hashlib.blake2b(digest_size=20)
    if device is None:
      self.add(kind)
      return
    caps = device_caps(device)
    arch = f"sm{caps.arch}" if caps.kind == "cuda" else caps.kind
    self.add(kind, FORMAT, code_version(*CODE_PACKAGES), wp.config.version, arch)

  def add(self, *values: object) -> Digest:
    """Hashes host values by their repr (numbers, strings, enums, Warp vectors), host and device
    arrays and Warp matrices by their entries, tuples, lists and dataclasses element by element
    (the repr of a large array elides its middle), sets and mappings in a fixed order (a set's repr
    follows string hashes, which change per process; equal mappings may differ in insertion order).

    Raises:
      UncacheableError: for a value without a value repr (an object address would key the bake).
    """
    for v in values:
      if isinstance(v, np.ndarray | wp.array):
        self.array(v)
      elif isinstance(v, tuple | list):
        self.add(type(v).__name__, len(v), *v)
      elif isinstance(v, set | frozenset):
        self.add(type(v).__name__, len(v), *sorted(Digest("element", None).add(e).hexdigest() for e in v))
      elif isinstance(v, Mapping):
        # In the order of the keys' digests, like sets: equal mappings built in any order match.
        keyed = sorted(((Digest("key", None).add(k).hexdigest(), k, x) for k, x in v.items()), key=operator.itemgetter(0))
        self.add(type(v).__name__, len(v), *(e for _, k, x in keyed for e in (k, x)))
      elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        self.add(type(v).__qualname__, *(getattr(v, f.name) for f in dataclasses.fields(v)))
      elif isinstance(v, wp.Texture):
        self.add(_texture(v))  # its registered content digest (UncacheableError: none)
      elif wp.types.type_is_matrix(type(v)):
        self.add(type(v).__name__).array(np.asarray(v))
      else:
        self._text(repr(v), type(v))
    return self

  def _text(self, text: str, kind: type) -> None:
    """Hashes the repr `text` of a value of type `kind`.

    Raises:
      UncacheableError: for a repr naming an object address.
    """
    if _ADDRESS.search(text):
      raise UncacheableError(f"{kind.__name__} has no value repr: {text}")
    self._h.update(text.encode())
    self._h.update(b"\0")

  def array(self, a: AnyHostArray | AnyDeviceArray) -> Digest:
    """Hashes an array's dtype, shape and contents (texture arrays: their textures' digests)."""
    if isinstance(a, wp.array):
      registered = _registered(a)
      if registered is not None:  # its creator's content digest: no host copy, no rehash
        return self.add("registered", registered)
      if a.size == 0:  # the dtype by name: a struct type's repr holds its address
        return self.add("empty", wp.types.type_repr(a.dtype), a.shape)
      if isinstance(a.dtype, type) and issubclass(a.dtype, wp.Texture):
        # Each element starts with the texture's 64-bit handle id.
        records = np.ascontiguousarray(host_copy(a)).reshape(-1)
        ids = np.frombuffer(records.tobytes(), np.uint64).reshape(len(records), -1)[:, 0]
        return self.add([_texture(int(i)) for i in ids])
      if a.dtype == wp.uint64:
        return self.add("handles", a.shape)  # BVH ids: derived from hashed arrays
      a = host_copy(a)
    # Struct records by their fields' bytes: the padding between fields is not a value (a kernel
    # that stores a local struct copies whatever its stack held there).
    a = np.ascontiguousarray(rfn.repack_fields(a, recurse=True) if a.dtype.names else a)
    self.add(str(a.dtype.descr) if a.dtype.names else a.dtype.str, a.shape)  # field names and types, not `|V<n>`
    self._h.update(a.data)
    return self

  def struct(self, s: object) -> Digest:
    """Hashes every field of a Warp struct instance (nested structs, arrays, textures, scalars)."""
    for name, var in _struct_vars(s).items():
      value = getattr(s, name)
      self.add(name)
      if isinstance(value, wp.array):
        self.array(value)
      elif isinstance(value, wp.Texture):
        self.add(_texture(value))
      elif wp.types.is_struct(value):
        self.struct(value)
      elif var.type == wp.uint64:
        self.add("handle")
      else:
        self.add(value)
    return self

  def hexdigest(self) -> str:
    """The digest so far."""
    return self._h.hexdigest()


def fields(s: object) -> dict[str, object]:
  """Field name -> value of a Warp struct instance (the table `Digest.struct` hashes)."""
  return {name: getattr(s, name) for name in _struct_vars(s)}


def _file(kind: str, digest: str) -> Path | None:
  """`<cache dir>/bakes/<kind>/<digest>.npz`, or None when caching is off."""
  root = cache_dir()
  return None if root is None else root / "bakes" / kind / f"{digest}.npz"


def load(key: Digest | None) -> Bake | None:
  """The stored arrays of a bake, or None (no key, caching off, a miss, or an unreadable file:
  warned, then rebuilt and rewritten by the caller)."""
  path = None if key is None else _file(key.kind, key.hexdigest())
  return None if path is None else read_arrays(path)


def save(key: Digest | None, arrays: Mapping[str, AnyHostArray]) -> None:
  """Stores a bake's arrays (`store_arrays`)."""
  path = None if key is None else _file(key.kind, key.hexdigest())
  if path is not None:
    store_arrays(path, arrays)


def cached[T](
  key: Digest | None, compute: Callable[[], tuple[T, Mapping[str, AnyHostArray]]], restore: Callable[[Bake], T | None]
) -> T:
  """A bake's value: `restore` of its stored arrays (`load`), else the value `compute` returns,
  whose arrays it stores (`save`). `restore` returns None for arrays it cannot use (computed and
  stored again)."""
  stored = load(key)
  restored = None if stored is None else restore(stored)
  if restored is not None:
    return restored
  value, arrays = compute()
  save(key, arrays)
  return value


def read_arrays(path: Path) -> Bake | None:
  """The arrays of an `.npz` cache file, or None (a miss, or a damaged or unreadable file: the
  caller rebuilds and rewrites it)."""
  if not path.is_file():
    return None
  try:
    return _stored_members(path)
  except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
    warnings.warn(f"raywarp cache: unreadable {path} ({e}); rebuilding it", RuntimeWarning, stacklevel=2)
    return None


def _stored_members(path: Path) -> Bake:
  """The `.npy` members of an uncompressed `.npz` (`np.savez`) read straight from the file:
  `np.load` reads them through `zipfile`, which copies every byte through a CRC check (most of a
  bake hit's time; the files are written atomically, `store_arrays`).

  Raises:
    ValueError: for a compressed member or a truncated file.
  """
  out: Bake = {}
  with zipfile.ZipFile(path) as archive, path.open("rb") as f:
    for info in archive.infolist():
      if info.compress_type != zipfile.ZIP_STORED or not info.filename.endswith(".npy"):
        raise ValueError(f"member {info.filename} is not a stored .npy")
      f.seek(info.header_offset)
      local = f.read(_LOCAL_HEADER)
      if len(local) != _LOCAL_HEADER:
        raise ValueError("truncated local header")
      name_len, extra_len = struct.unpack("<HH", local[26:30])
      f.seek(info.header_offset + _LOCAL_HEADER + name_len + extra_len)
      out[info.filename[: -len(".npy")]] = np.lib.format.read_array(f, allow_pickle=False)
  return out


# Bytes of a zip local file header before its name and extra field.
_LOCAL_HEADER: Final = 30


def store_arrays(path: Path, arrays: Mapping[str, AnyHostArray]) -> None:
  """Writes `arrays` to the `.npz` cache file `path` atomically (`cache.paths.write_atomic`: a
  write that fails, a full or read-only disk, warns and leaves the cache without the entry)."""
  write_atomic(path, lambda f: np.savez(f, **{k: np.asarray(v) for k, v in arrays.items()}), "bake cache")  # pyrefly: ignore[bad-argument-type]  # stub gap NS4


def key(kind: str, device: wp.Device, *structs: object, host: tuple[object, ...] = ()) -> Digest | None:
  """The digest of a bake reading `structs` and the `host` values (None: caching off, or an input
  that cannot be hashed: warned once per bake kind and reason, the bake then runs every time)."""
  if cache_dir() is None:
    return None
  d = Digest(kind, device)
  try:
    d.add(*host)
    for s in structs:
      d.struct(s)
  except UncacheableError as e:
    _warn_uncached(kind, str(e))
    return None
  return d


@functools.cache
def _warn_uncached(kind: str, reason: str) -> None:
  """Warns (once per bake kind and reason) that a bake is not cached."""
  warnings.warn(f"raywarp bake cache: the {kind} bake is not cached ({reason})", RuntimeWarning, stacklevel=3)
