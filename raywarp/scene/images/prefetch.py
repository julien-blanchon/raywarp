"""Background reads of the files a rebuild will read: the dependency list of the stale `.rayw`.

On a network file system (e.g. a bucket mount: ~25 ms per uncached lookup, ~0.2 s per first
read of a file, ~5 reads in flight) a cold build is mostly waiting: USD opens its layers one
after the other, resolves asset paths one at a time, the builder reads every image. A stale
cache file already lists every file the previous build read; `Prefetch` reads them all on a
thread pool (smallest first: layers and modules before images) while the stage opens, so the
build's own reads hit the file-system caches. Each read also records the file's `Dependency`
(fstat of the open file), which the cache record of the rebuilt scene reuses instead of a
second stat pass (`cache.record`).
"""

from __future__ import annotations

import concurrent.futures
import os
import stat
import threading
from collections.abc import Iterable
from typing import Final, final

from raywarp.scene.images.blob import worker_count
from raywarp.scene.model.rayw import Dependency

# Reads in flight (the mount serves ~5 at full speed; more hide its per-file latency).
THREADS: Final = 32
# Bytes per read call.
_CHUNK: Final = 8 << 20


def _read(path: str, stop: threading.Event) -> Dependency | None:
  """Reads all of `path` (into the file-system caches); its `Dependency` (-1, -1: missing or not
  a regular file), None when reading it failed or `stop` ended it (its state is then left to
  `cache.record`)."""
  try:
    # A cached regular file may since have become a FIFO; opening must not wait for a writer
    # before fstat can reject it. O_NONBLOCK does not change regular-file reads.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
  except OSError:
    return Dependency(path=path, mtime_ns=-1, size=-1)
  try:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
      return Dependency(path=path, mtime_ns=-1, size=-1)
    while os.read(fd, _CHUNK):
      if stop.is_set():
        return None
    return Dependency(path=path, mtime_ns=st.st_mtime_ns, size=st.st_size)
  except OSError:  # a failed file read only loses its prefetch; programming errors propagate
    return None
  finally:
    os.close(fd)


@final
class Prefetch:
  """Reads files on a thread pool in the background (see the module docstring)."""

  def __init__(self, files: Iterable[Dependency], threads: int = THREADS) -> None:
    """Starts reading `files` (smallest first); rejects a worker count below 1."""
    order = sorted(files, key=lambda d: (d.size, d.path))
    self._pool = concurrent.futures.ThreadPoolExecutor(max(1, min(worker_count(threads), len(order))), "raywarp-prefetch")
    self._stop = threading.Event()
    self._futures = {d.path: self._pool.submit(_read, d.path, self._stop) for d in order}

  def done(self) -> dict[str, Dependency]:
    """Completed dependencies by path (I/O failures left out; unexpected worker errors raised)."""
    out: dict[str, Dependency] = {}
    for path, f in self._futures.items():
      state = f.result() if f.done() and not f.cancelled() else None
      if state is not None:
        out[path] = state
    return out

  def close(self) -> None:
    """Drops the reads not started yet and stops the started ones at their next chunk."""
    self._stop.set()
    self._pool.shutdown(wait=False, cancel_futures=True)
