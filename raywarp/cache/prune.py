"""Bounds raywarp's disk caches: `prune(max_bytes)` deletes the least recently used entries of
the content-keyed caches under the cache directory (`cache_dir()`) until they fit.

The caches grow with every scene, setting and code version seen (bakes, textures, `.wpbr`
scenes, PTX), and nothing in them is ever needed: a deleted entry is a miss that rebuilds it. So
the size is bounded by a prune the user runs (or a job script calls), not by bookkeeping on
every write:

  raywarp cache prune --max-gb 20      # or raywarp.cache.prune.prune(max_bytes)

An entry's last use is its newest access or modification time (the PTX cache refreshes it on
every read; a file read through the kernel's `relatime` updates it at least once a day): the
oldest go first. Entries are files, a PTX's `.ptx` and `.json` together, a converted glTF's whole
directory. Left alone: OptiX's disk cache (`optix/`: its own size bound,
`core/trace/optix/context.py`), the prebuilt kernel store (`kernels/`: one directory per release
content, delete an old one by hand), the seeding stamps (`prebuilt/`), locks, temporaries and
whatever else lives in the directory (MDL's conversions prune themselves after a week). Warp's own
kernel cache is Warp's (`warp.clear_kernel_cache()`).

A process using an entry meanwhile keeps its open or mapped file (POSIX); its next start rebuilds
what was deleted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Final, final

from raywarp.cache.paths import cache_dir

# The pruned caches: directory -> whether an entry is one of its sub-directories (else its files).
CACHES: Final = {"bakes": False, "textures": False, "ptx": False, "emissive": False, "gltf": True}
_SKIPPED: Final = (".lock", ".partial", ".tmp")
_GB: Final = 1 << 30


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Entry:
  """One cache entry: its paths (deleted together), bytes and last use [s since the epoch]."""

  paths: tuple[Path, ...]
  size: int
  used: float


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Pruned:
  """What `prune` did: entries and bytes removed, bytes kept."""

  removed: int
  freed: int
  kept: int

  def __str__(self) -> str:
    """E.g. "removed 12 entries (3.40 GB), 19.90 GB kept"."""
    return f"removed {self.removed} entries ({self.freed / _GB:.2f} GB), {self.kept / _GB:.2f} GB kept"


def prune(max_bytes: int, root: Path | None = None) -> Pruned:
  """Deletes the least recently used entries of the caches under `root` (default `cache_dir()`)
  until they hold at most `max_bytes` (see the module doc).

  Raises:
    ValueError: a negative `max_bytes`, or caching off (`RAYWARP_CACHE_DIR=""`) without `root`.
  """
  if max_bytes < 0:
    raise ValueError(f"max_bytes must be >= 0, got {max_bytes}")
  root = root if root is not None else cache_dir()
  if root is None:
    raise ValueError("caching is off (RAYWARP_CACHE_DIR is empty): nothing to prune")
  entries = sorted(entries_of(root), key=lambda e: e.used)
  total = sum(e.size for e in entries)
  removed = freed = 0
  for e in entries:
    if total - freed <= max_bytes:
      break
    for p in e.paths:
      if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
      else:
        p.unlink(missing_ok=True)
    removed, freed = removed + 1, freed + e.size
  return Pruned(removed, freed, total - freed)


def entries_of(root: Path) -> Iterator[Entry]:
  """The entries of the pruned caches under `root` (`CACHES`, plus the `.wpbr` files in `root`)."""
  yield from (_entry((p,)) for p in _files(root, "*.wpbr"))
  for name, directories in CACHES.items():
    folder = root / name
    if directories:
      yield from (_entry((d,)) for d in _children(folder) if d.is_dir())
    elif name == "ptx":
      groups: dict[Path, list[Path]] = {}
      for p in _files(folder, "**/*"):
        groups.setdefault(p.with_suffix(""), []).append(p)
      yield from (_entry(tuple(ps)) for ps in groups.values())
    else:
      yield from (_entry((p,)) for p in _files(folder, "**/*"))


def _files(folder: Path, pattern: str) -> list[Path]:
  """The files of `folder` matching `pattern`, locks and temporaries aside (none when absent)."""
  with contextlib.suppress(OSError):
    return [p for p in folder.glob(pattern) if p.is_file() and not p.name.endswith(_SKIPPED)]
  return []


def _children(folder: Path) -> list[Path]:
  """The entries of `folder` (none when absent)."""
  with contextlib.suppress(OSError):
    return list(folder.iterdir())
  return []


def _entry(paths: tuple[Path, ...]) -> Entry:
  """The `Entry` of `paths` (a directory counts its files; a vanished path counts 0)."""
  size, used = 0, 0.0
  for p in paths:
    for f in [p] if p.is_file() else p.rglob("*"):
      with contextlib.suppress(OSError):
        st = f.stat()
        size += st.st_size if f.is_file() else 0
        used = max(used, st.st_atime, st.st_mtime)
  return Entry(paths, size, used)


def sizes(root: Path | None = None) -> dict[str, int]:
  """Bytes held under the cache directory `root` (default `cache_dir()`; {} with caching off), per
  top-level directory (the pruned caches, the kernel store `kernels`, OptiX's cache `optix`, ...)
  and "wpbr" for the `.wpbr` files beside them; largest first."""
  root = root if root is not None else cache_dir()
  if root is None:
    return {}
  out = {"wpbr": sum(e.size for e in (_entry((p,)) for p in _files(root, "*.wpbr")))}
  for d in _children(root):
    if d.is_dir():
      out[d.name] = _entry((d,)).size
  return dict(sorted(out.items(), key=lambda kv: -kv[1]))
