"""Compiles the test suite's megakernels once, before the pytest workers start.

A CPU suite run compiles one software megakernel per distinct `Features` (6-40 s each), and with
xdist every worker that meets a configuration on a cold kernel cache compiles it too. A run with
$RAYWARP_RECORD_FEATURES=DIR (conftest.py) records the configurations into DIR; this tool
loads them (pickles of an older code version or format are skipped, counted) and
compiles each distinct one once into Warp's kernel cache, spread over `--workers` processes, so
the suite's workers only load modules.

  python -m tests.precompile DIR [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing
import pickle
import sys
import time
from pathlib import Path


def load(directory: Path) -> tuple[list[tuple[object, object, bool]], int]:
  """(the distinct recorded `pixel.megakernel` calls of `directory`: (Features, camera model,
  banded), the files that did not load: an older code version's or format's)."""
  seen: dict[str, tuple[object, object, bool]] = {}
  bad = 0
  for path in sorted(directory.glob("features-*.pkl")):
    try:
      calls = pickle.loads(path.read_bytes())
      if not all(isinstance(c, tuple) and len(c) == 3 for c in calls):
        raise TypeError("not (Features, camera model, banded) calls")
    except (pickle.UnpicklingError, AttributeError, TypeError, ImportError, EOFError):
      bad += 1
      continue
    for call in calls:
      seen.setdefault(repr(call), call)
  return list(seen.values()), bad


def compile_some(calls: list[tuple[object, object, bool]]) -> int:
  """Compiles the megakernel module of each recorded call on the CPU; returns the count."""
  import warp as wp

  from raywarp.core.kernels import pixel

  for f, model, banded in calls:
    kernel = pixel.megakernel(f, model, banded=banded)
    wp.force_load(device="cpu", modules=[kernel.module])
  return len(calls)


def main() -> int:
  """Command line (module doc)."""
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("directory", type=Path)
  ap.add_argument("--workers", type=int, default=8)
  a = ap.parse_args()
  if not a.directory.is_dir():
    print(f"test_precompile: no {a.directory}: nothing to compile")
    return 0
  features, bad = load(a.directory)
  t0 = time.perf_counter()
  chunks = [features[i :: a.workers] for i in range(a.workers)]
  with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
    done = sum(pool.map(compile_some, [c for c in chunks if c]))
  print(f"test_precompile: {done} configurations compiled in {time.perf_counter() - t0:.1f} s ({bad} stale files skipped)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
