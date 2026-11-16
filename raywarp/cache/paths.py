"""The disk-cache directory and atomic cache writes, shared by every disk cache of the package (the
`.wpbr` scene files, the bakes, textures, kernels and conversion memos).

A writer writes its own temporary file `<name>.tmp-<host>-<pid>-<token>` next to the destination
and renames it into place on success (`write_atomic`), so a concurrent reader sees the old file or
the new one; `remove_stale` deletes the temporary files of crashed writers. The directory is
$RAYWARP_CACHE_DIR, else ~/.cache/raywarp (`cache_dir`; an empty value turns caching off).
"""

from __future__ import annotations

import contextlib
import functools
import glob
import hashlib
import os
import socket
import time
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Final

# A temporary name's "-"-separated parts after ".tmp-": host, pid, token.
_TMP_PARTS: Final = 3


# A temporary file last written longer ago than this is stale, whatever its writer (`remove_stale`).
STALE_SECONDS: Final = 6 * 3600.0


# Cache directory of the scene adapters' `.wpbr` files (their default `cache=True`) and of the
# renderer's bakes, unless the environment variable CACHE_ENV names another one (empty: no cache).
DEFAULT_DIR: Final = "~/.cache/raywarp"


CACHE_ENV: Final = "RAYWARP_CACHE_DIR"


@functools.cache
def _host() -> str:
  """This process's pid space: a digest of the host name, the kernel's boot id and the pid
  namespace (hex, no "-": `remove_stale` splits on it). Two containers or machines sharing a
  host name (a shared cache mount) differ in boot id or pid namespace, so a writer's pid is only
  checked where it means that process."""
  parts = [socket.gethostname()]
  with contextlib.suppress(OSError):
    parts.append(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
  with contextlib.suppress(OSError):
    parts.append(str(os.stat("/proc/self/ns/pid").st_ino))
  return hashlib.blake2b("\0".join(parts).encode(), digest_size=8).hexdigest()


def tmp_name(path: Path) -> str:
  """A temporary name of `path` unique to one writer: `<name>.tmp-<host>-<pid>-<token>`."""
  return f"{path.name}.tmp-{_host()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"


def _pid_alive(pid: int) -> bool:
  """Whether process `pid` of this host is running."""
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except OSError:
    return True
  return True


def remove_stale(path: str | os.PathLike[str]) -> list[Path]:
  """Deletes the stale temporary files of `.wpbr` file `path` (see the module docstring).

  A temporary file is stale when its writer (a pid of this pid space, `_host`) is dead, or when it
  was last written more than STALE_SECONDS ago (another pid space's, or one whose pid a later
  process reuses: a writer writes its file until it renames it). Returns the removed files.
  """
  path = Path(path)
  removed: list[Path] = []
  try:
    candidates = list(path.parent.glob(f"{glob.escape(path.name)}.tmp-*"))
  except OSError:
    return removed
  host, now = _host(), time.time()
  for tmp in candidates:
    parts = tmp.name[len(path.name) + len(".tmp-") :].split("-")  # host, pid, token
    try:
      ours = len(parts) == _TMP_PARTS and parts[0] == host and parts[1].isdigit()
      if (ours and not _pid_alive(int(parts[1]))) or now - tmp.stat().st_mtime > STALE_SECONDS:
        tmp.unlink()
        removed.append(tmp)
    except OSError:
      continue
  return removed


def write_atomic(path: Path, write: Callable[[BinaryIO], object], what: str) -> None:
  """Writes a disk cache file atomically: a concurrent reader sees the old file or the new one.
  The cache is best effort: a failed write warns (naming `what`) and leaves no file. The writer's
  own temporary file (`tmp_name`: `remove_stale` deletes those of crashed writers first) is
  removed whatever ends the write."""
  tmp = path.with_name(tmp_name(path))
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale(path)
    with tmp.open("wb") as f:
      write(f)
    tmp.replace(path)
  except OSError as error:
    warnings.warn(f"raywarp: {what} not written ({path}): {error}", RuntimeWarning, stacklevel=3)
  finally:
    tmp.unlink(missing_ok=True)


def cache_dir() -> Path | None:
  """The cache directory: $RAYWARP_CACHE_DIR, else DEFAULT_DIR; None when the variable is empty."""
  value = os.environ.get(CACHE_ENV)
  if value is None:
    return Path(DEFAULT_DIR).expanduser()
  return Path(value).expanduser() if value else None


def location(
  cache: str | os.PathLike[str] | bool | None,  # noqa: FBT001 (the adapters' `cache` option)
) -> str | os.PathLike[str] | None:
  """Where an adapter's `cache` option keeps its `.wpbr` file: True (the adapters' default) the
  cache directory (`cache_dir`: None when CACHE_ENV is set empty, caching off; None with a
  RuntimeWarning when it cannot be created or written, as the bake cache's failed writes: a
  read-only home must not fail every load), a path as given (its errors raise: asked for), None
  or False no cache."""
  if cache is None or cache is False:
    return None
  if cache is not True:
    return cache
  directory = cache_dir()
  if directory is not None and not _writable(directory):
    warnings.warn(
      f"raywarp cache: {directory} cannot be written; loading without the .wpbr cache "
      f"(set {CACHE_ENV} to a writable directory, or to an empty value to turn caching off)",
      RuntimeWarning,
      stacklevel=3,
    )
    return None
  return directory


def _writable(directory: Path) -> bool:
  """Whether `directory` exists (made here if missing) and files can be created in it."""
  try:
    directory.mkdir(parents=True, exist_ok=True)
  except OSError:
    return False
  return os.access(directory, os.W_OK | os.X_OK)
