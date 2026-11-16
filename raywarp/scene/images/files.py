"""Source files on disk or inside a package, named by USD's package-relative paths.

A packaged file is `<package>[<member>]`: `/a/robot.usdz[textures/base.png]`, nested as
`/a/outer.usdz[inner.usdz[base.png]]` (Ar's syntax, `ArJoinPackageRelativePath`). A package is
a zip archive (USDZ: zip, members stored uncompressed and 64-byte aligned; any zip reads). The
image decoders read through this module, so a texture inside a `.usdz` decodes like a file. The
package file on disk is what a scene depends on (`container`).
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import dataclasses
import io
import os
import posixpath
import threading
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import Final, final

_OPEN: Final = "["
_CLOSE: Final = "]"
# Packages kept open (`_Archives`): a scene reads its textures from one or a few.
ARCHIVE_CACHE: Final = 8


@final
class PackageError(OSError):
  """A packaged file that cannot be read (the package or the member is missing / not a zip)."""


def split(path: str) -> tuple[str, str] | None:
  """(package, member) of a package-relative path (the outermost package: the member may itself
  be packaged); None for a plain path."""
  if not path.endswith(_CLOSE):
    return None
  depth = 0
  for i in range(len(path) - 1, -1, -1):
    c = path[i]
    depth += (c == _CLOSE) - (c == _OPEN)
    if depth == 0:
      return (path[:i], path[i + 1 : -1]) if i > 0 else None
  return None


def is_packaged(path: str) -> bool:
  """Whether `path` names a file inside a package."""
  return split(path) is not None


def absolute(path: str) -> str:
  """`path` with its file on disk made absolute and normalised (a packaged file's member names
  are kept as given)."""
  parts = split(path)
  if parts is None:
    return os.path.abspath(path)  # noqa: PTH100 (normalises "..", keeps links: one key per path)
  return f"{absolute(parts[0])}{_OPEN}{parts[1]}{_CLOSE}"


def container(path: str) -> str:
  """The file on disk holding `path`: the outermost package of a packaged file, else `path`."""
  parts = split(path)
  return path if parts is None else parts[0]


def suffix(path: str) -> str:
  """The lower-case extension of the file `path` names (a packaged file's own, not its
  package's)."""
  while (parts := split(path)) is not None:
    path = parts[1]
  return Path(path).suffix.lower()


def join(document: str, relative: str) -> str:
  """The path of file `relative` next to file `document`, normalised, as Ar anchors an asset path:
  an absolute `relative` as it is; else inside `document`'s package when it is packaged
  (`a.usdz[m/x.mtlx]` + `t.png` -> `a.usdz[m/t.png]`), else in its directory. A `..` past the
  package root stays in the member (`a.usdz[../t.png]`), which no package holds: Ar anchors it so
  too and does not resolve it (checked with usd-core 26.8)."""
  if Path(relative).is_absolute():
    return os.path.normpath(relative)
  parts = split(document)
  if parts is None:
    return os.path.normpath(Path(document).parent / relative)
  package, member = parts
  if is_packaged(member):
    return f"{package}{_OPEN}{join(member, relative)}{_CLOSE}"
  inner = posixpath.normpath(posixpath.join(posixpath.dirname(member), relative.replace(os.sep, "/")))
  return f"{package}{_OPEN}{inner}{_CLOSE}"


def _innermost(path: str) -> tuple[str, str] | None:
  """(package, member) of a packaged file with the member a plain name: `a.usdz[q.usdz[c.png]]`
  -> (`a.usdz[q.usdz]`, `c.png`); None for a plain path."""
  parts = split(path)
  if parts is None:
    return None
  package, member = parts
  while (inner := split(member)) is not None:
    package, member = f"{package}{_OPEN}{inner[0]}{_CLOSE}", inner[1]
  return package, member


@final
@dataclasses.dataclass(slots=True)
class _Archive:
  """An open zip package, the lock its member reads take (a ZipFile is read by one thread at a
  time: its file position is shared), and its readers: an evicted archive (`retired`) closes when
  its last reader is done, never under one (all guarded by `_Archives`' lock)."""

  zip: zipfile.ZipFile
  lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
  readers: int = 0
  retired: bool = False


# A package's identity: its path inside its container chain, and the container file's mtime and
# size (an edited container is a new archive).
type _Key = tuple[str, int, int]


@final
class _Archives:
  """The open packages, least recently used first, at most `ARCHIVE_CACHE` of them (evicted ones,
  and older states of an edited package, close once unread; `close` retires all, also at exit;
  a forked child drops its parent's). Thread-safe: decodes run on threads."""

  def __init__(self) -> None:
    """No archive open."""
    self._lock = threading.Lock()
    self._open: collections.OrderedDict[_Key, _Archive] = collections.OrderedDict()

  @contextlib.contextmanager
  def use(self, package: str) -> Generator[_Archive]:
    """The open archive of `package` (a file or a packaged file, opened on first use), held open
    while the block runs.

    Raises:
      PackageError: for a missing package, or one that is not a zip.
    """
    archive = self._acquire(package)
    try:
      yield archive
    finally:
      with self._lock:
        archive.readers -= 1
        if archive.retired and archive.readers == 0:
          archive.zip.close()

  def _acquire(self, package: str) -> _Archive:
    """The archive of `package` with one more reader."""
    try:
      st = Path(container(package)).stat()
    except OSError as e:
      raise PackageError(f"{package}: not a readable zip package ({e})") from e
    key = (package, st.st_mtime_ns, st.st_size)
    with self._lock:
      if (hit := self._open.get(key)) is not None:
        self._open.move_to_end(key)
        hit.readers += 1
        return hit
    opened = _Archive(zip=_open_zip(package))
    with self._lock:
      archive = self._open.get(key)
      if archive is None:
        for stale in [k for k in self._open if k[0] == package]:  # an older state of the package
          self._retire(stale)
        archive = self._open[key] = opened
        while len(self._open) > ARCHIVE_CACHE:
          self._retire(next(iter(self._open)))
      else:  # opened meanwhile by another thread
        opened.zip.close()
      archive.readers += 1
      return archive

  def _retire(self, key: _Key) -> None:
    """Drops `key` from the cache, closing its archive now or when its last reader is done (the
    caller holds the lock)."""
    archive = self._open.pop(key)
    archive.retired = True
    if archive.readers == 0:
      archive.zip.close()

  def close(self) -> None:
    """Retires every open archive."""
    with self._lock:
      while self._open:
        self._retire(next(iter(self._open)))

  def forget(self) -> None:
    """In a forked child: drops the parent's archives (their files are the parent's) and its lock
    (another thread may have held it at the fork)."""
    self._lock = threading.Lock()
    self._open = collections.OrderedDict()


def _open_zip(package: str) -> zipfile.ZipFile:
  """The zip archive `package` (a file, or itself a packaged file: read into memory once)."""
  try:
    if is_packaged(package):
      return zipfile.ZipFile(io.BytesIO(read(package)))
    return zipfile.ZipFile(package)
  except (OSError, zipfile.BadZipFile) as e:
    raise PackageError(f"{package}: not a readable zip package ({e})") from e


_ARCHIVES: Final = _Archives()
atexit.register(_ARCHIVES.close)
os.register_at_fork(after_in_child=_ARCHIVES.forget)


def close_archives() -> None:
  """Closes the packages kept open (they reopen on the next read; one being read closes after)."""
  _ARCHIVES.close()


@contextlib.contextmanager
def _member(path: str) -> Generator[tuple[_Archive, zipfile.ZipInfo]]:
  """(the open archive, the member's entry) of packaged file `path`, held open in the block.

  Raises:
    PackageError: for a missing package or member.
  """
  parts = _innermost(path)
  if parts is None:
    raise PackageError(f"{path}: not a packaged file")
  package, member = parts
  with _ARCHIVES.use(package) as archive:
    try:
      info = archive.zip.getinfo(member)
    except KeyError:
      raise PackageError(f"{path}: no member {member!r} in {package}") from None
    yield archive, info


def read(path: str, limit: int | None = None) -> bytes:
  """The bytes of file `path` (its first `limit` bytes when given).

  Raises:
    OSError: for a missing or unreadable file (PackageError for a packaged one).
  """
  if not is_packaged(path):
    with Path(path).open("rb") as f:
      return f.read() if limit is None else f.read(limit)
  with _member(path) as (archive, info), archive.lock, archive.zip.open(info) as f:
    return f.read() if limit is None else f.read(limit)


def read_stat(path: str) -> tuple[bytes, str, os.stat_result]:
  """(the bytes of file `path`, the file on disk holding it, that file's stat): the open file's
  fstat, or a package's stat taken before the read (an edit during it shows as a change).

  Raises:
    OSError: as `read`.
  """
  if is_packaged(path):
    package = container(path)
    st = Path(package).stat()
    return read(path), package, st
  with Path(path).open("rb") as f:
    return f.read(), path, os.fstat(f.fileno())


def size(path: str) -> int:
  """The size of file `path` [bytes] (a packaged file's uncompressed size).

  Raises:
    OSError: as `read`.
  """
  if not is_packaged(path):
    return Path(path).stat().st_size
  with _member(path) as (_, info):
    return info.file_size


def is_file(path: str) -> bool:
  """Whether `path` names an existing file (a packaged one: its package holds the member)."""
  if not is_packaged(path):
    return Path(path).is_file()
  try:
    size(path)
  except OSError:
    return False
  return True
