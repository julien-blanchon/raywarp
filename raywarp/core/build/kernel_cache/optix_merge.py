"""Merges the rows of one OptiX disk-cache database into another, in a process of its own.

  python -I optix_merge.py SRC DST

`prebuilt.merge_optix` runs it as a child process: a process whose OptiX has DST open never opens
DST itself. OptiX keeps its disk cache in an SQLite database in WAL mode through its own copy of
SQLite, and two SQLite libraries in one process do not see each other's POSIX locks: a connection
of Python's SQLite closing DST finds no other holder, checkpoints and deletes the WAL and its
index under OptiX's open connection (OptiX's later rows go to the unlinked files and are lost, and
its view of the index breaks: SQLITE_PROTOCOL, "locking protocol", on the next merge), and closing
its file descriptor drops OptiX's locks. Another process's connection sees OptiX's locks like any
other process's: SQLite's own locking then orders the two (the busy timeout below).

Standard library only, run by path: the child imports nothing of raywarp or Warp.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Final

# OptiX's disk-cache tables and the columns a merge copies (OptiX 9.0's schema; named, not `*`).
TABLES: Final = {
  "cache_data": ("key", "value"),
  "cache_info": ("key", "optix_version", "driver_version", "size", "timestamp"),
}
# Seconds a merge waits for OptiX's (or another merge's) write lock.
BUSY_TIMEOUT_S: Final = 30.0


def merge(src: Path, dst: Path) -> None:
  """Adds the rows of OptiX disk cache `src` that `dst` lacks, in one transaction (`dst` created
  as a copy of `src` when absent). OptiX's own triggers keep its size accounting."""
  dst.parent.mkdir(parents=True, exist_ok=True)
  if not dst.exists():
    scratch = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent))
    try:
      shutil.copy2(src, scratch / dst.name)
      os.link(scratch / dst.name, dst)  # fails if the file appeared meanwhile: then merged below
      return
    except FileExistsError:
      pass
    finally:
      shutil.rmtree(scratch, ignore_errors=True)
  db = sqlite3.connect(dst, timeout=BUSY_TIMEOUT_S, isolation_level=None)
  try:
    db.execute("ATTACH DATABASE ? AS bundle", (str(src),))
    db.execute("BEGIN IMMEDIATE")
    for table, columns in TABLES.items():
      names = ", ".join(columns)
      db.execute(
        f"INSERT OR IGNORE INTO main.{table} ({names}) SELECT {names} FROM bundle.{table} "  # fixed names
        f"WHERE key NOT IN (SELECT key FROM main.{table})"
      )
    db.execute("COMMIT")
    db.execute("DETACH DATABASE bundle")
  finally:
    db.close()


if __name__ == "__main__":
  merge(Path(sys.argv[1]), Path(sys.argv[2]))
