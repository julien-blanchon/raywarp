"""The test jobs' compile cache: a warm snapshot per kernel-code digest; where a suite compiles.

A CPU test job's kernels (Warp's CPU modules, the megakernels, the OptiX programs' PTX compiled
by NVRTC on the CPU) are keyed by their code. The job's generic `warp` cache family is ~25 GB of
every tree's modules (1.5-2 min to restore, too large to save: BUG-289), so a suite started
cold. Here a test job keeps its own snapshot, keyed by the digest of the code that makes kernels
(`key`: `raywarp/`, `tests/`, `tools/` sources and the Warp release), holding only what the
suite loaded:

  python -m tests.kernel_snapshot restore DIR    # DIR/tests.<key>.tar, else the newest (a near
  tree) python -m pytest -p tests.kernel_snapshot ...  # records each Warp module load, PTX build
  ($LOG_ENV) python -m tests.kernel_snapshot save DIR       # the loaded modules ->
  DIR/tests.<key>.tar

DIR is the job's cache key directory on the bucket (`/data/cache/<env>-cpu-warp<version>`: the
`tests` family of `tools.hf cache`; the newest `KEEP` are kept).
  python -m tests.kernel_snapshot report          # compile seconds per module family and per test
  file

A rerun of the same tree restores its own snapshot (every module a load); a changed tree restores
the newest snapshot, whose unchanged modules still load. (No CPU test of the render and contracts
areas compiles an OptiX programs' PTX: 0 of 1 597 loads, `w10r-suite-cold`; the recorder counts
any that does.)
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import hashlib
import json
import os
import sys
import tarfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
# Where the plugin writes its records (one JSON-lines file per process; unset: the plugin is off).
LOG_ENV: Final = "RAYWARP_TEST_COMPILES"
# The code whose change can change a kernel (`key`).
SOURCES: Final = ("raywarp", "tests", "tools")
# The snapshots' family name (`tools/hf/config.FAMILIES`) and how many are kept (each a tree's; the
# newest serves a changed tree; `config.KEEP_BY_FAMILY`).
FAMILY: Final = "tests"
KEEP: Final = 6


# --- the key and the snapshot ---------------------------------------------------------------------


def key(root: Path = ROOT) -> str:
  """The digest of every Python source under `SOURCES` of `root` and the Warp release: equal
  digests, equal kernels."""
  import warp

  h = hashlib.sha256(f"warp {warp.config.version}\n".encode())
  for top in SOURCES:
    for path in sorted((root / top).rglob("*.py")):
      h.update(f"{path.relative_to(root)}\n".encode())
      h.update(path.read_bytes())
  return h.hexdigest()[:16]


def kernel_dir() -> Path:
  """Warp's kernel cache directory of this environment (`WARP_CACHE_PATH`/<release>, as Warp
  resolves it; without the variable, Warp's own, from an initialisation)."""
  import warp as wp

  if os.environ.get("WARP_CACHE_PATH"):
    return Path(os.environ["WARP_CACHE_PATH"]) / wp.config.version
  wp.config.log_level = wp.LOG_WARNING  # no init banner
  wp.init()
  assert wp.config.kernel_cache_dir is not None
  return Path(wp.config.kernel_cache_dir)


def _snapshots(directory: Path) -> list[Path]:
  """The snapshots of `directory`, newest first."""
  return sorted(directory.glob(f"{FAMILY}.*.tar"), key=lambda p: p.stat().st_mtime, reverse=True)


def restore(directory: Path) -> str:
  """Extracts `directory`/tests.<key>.tar (else the newest snapshot) into Warp's kernel cache
  (its `warp/`); prints and returns what was restored (exact, near, none)."""
  t0 = time.perf_counter()
  exact = directory / f"{FAMILY}.{key()}.tar"
  found = exact if exact.is_file() else next(iter(_snapshots(directory)), None)
  how = "none" if found is None else "exact" if found == exact else f"near ({found.name})"
  cache = kernel_dir()
  if found is not None:
    with tarfile.open(found) as tar:
      for member in tar:
        top, _, rest = member.name.partition("/")
        if member.isfile() and rest and top == "warp" and ".." not in member.name:
          member.name = rest
          tar.extract(member, cache, filter="data")
  size = found.stat().st_size >> 20 if found else 0
  print(f"test_cache restore: {how}, {size} MB in {time.perf_counter() - t0:.0f} s", flush=True)
  return how


def _records(log: Path) -> list[dict[str, object]]:
  """The plugin's records of every process (`LOG_ENV` directory `log`)."""
  rows = []
  for path in sorted(log.glob("compiles-*.jsonl")):
    rows += [json.loads(line) for line in path.read_text().splitlines() if line]
  return rows


def save(directory: Path, log: Path) -> None:
  """`directory`/tests.<key>.tar from the Warp modules the session loaded
  (a partial name, then renamed), unless the snapshot exists and the session compiled nothing;
  the oldest snapshots past `KEEP` deleted."""
  rows = _records(log)
  target = directory / f"{FAMILY}.{key()}.tar"
  compiled = sum(1 for r in rows if r["compiled"])
  if target.is_file() and not compiled:
    print(f"test_cache save: {target.name} holds this tree's kernels (nothing compiled): kept", flush=True)
    return
  t0 = time.perf_counter()
  cache = kernel_dir()
  used = sorted({str(r["dir"]) for r in rows if r["kind"] == "warp" and (cache / str(r["dir"])).is_dir()})
  if not used:  # a session that loaded nothing (stopped at its start): no snapshot to prefer
    print("test_cache save: the session loaded no Warp module: nothing saved", flush=True)
    return
  directory.mkdir(parents=True, exist_ok=True)
  partial = directory / f"{target.name}.{os.uname().nodename}.{os.getpid()}.partial"
  with tarfile.open(partial, "w") as tar:
    for name in used:
      tar.add(cache / name, f"warp/{name}")
  partial.replace(target)
  for old in _snapshots(directory)[KEEP:]:
    old.unlink(missing_ok=True)
  print(
    f"test_cache save: {target.name} {target.stat().st_size >> 20} MB ({len(used)} Warp modules, {compiled} compiled this run) "
    f"in {time.perf_counter() - t0:.0f} s",
    flush=True,
  )


def report(log: Path, top: int = 15) -> None:
  """Compile seconds per module family (a module's name without its hash) and per test file."""
  rows = [r for r in _records(log) if r["compiled"]]
  loads = len(_records(log))
  total = sum(float(r["s"]) for r in rows)  # pyrefly: ignore[bad-argument-type]
  print(f"test_cache report: {loads} loads, {len(rows)} compiles, {total:.0f} s of compile summed over the workers")
  for title, field in (("family", "module"), ("test file", "test")):
    sums: collections.defaultdict[str, float] = collections.defaultdict(float)
    counts: collections.Counter[str] = collections.Counter()
    for r in rows:
      name = str(r[field]).split("::")[0] if field == "test" else str(r[field])
      sums[name] += float(r["s"])  # pyrefly: ignore[bad-argument-type]
      counts[name] += 1
    for name, s in sorted(sums.items(), key=lambda kv: -kv[1])[:top]:
      print(f"test_cache {title}: {s:7.0f} s {counts[name]:5d} x {name}")


# --- the pytest plugin (`-p tests.kernel_snapshot`)
# ----------------------------------------------------

_TEST: list[str] = ["(collection)"]
_LOCK = threading.Lock()


def _record(kind: str, name: str, where: str, seconds: float, compiled: bool) -> None:
  """One load (compiled or read from a cache) of the current test, appended to this process's
  records at once (a worker killed mid-session, e.g. out of memory, keeps what it loaded)."""
  event = {"kind": kind, "module": name, "dir": where, "s": round(seconds, 2), "compiled": compiled, "test": _TEST[0]}
  with _LOCK, (Path(os.environ[LOG_ENV]) / f"compiles-{os.getpid()}.jsonl").open("a") as f:
    f.write(json.dumps(event) + "\n")


@contextlib.contextmanager
def _timed() -> Generator[list[float]]:
  """The seconds of the block, in the yielded list."""
  out: list[float] = []
  t0 = time.perf_counter()
  try:
    yield out
  finally:
    out.append(time.perf_counter() - t0)


def pytest_configure(config: object) -> None:
  """With `LOG_ENV` set: times every Warp module build (`Module._compile`: called on each first
  load, compiled or cached) and every OptiX PTX build (`pipeline.compile_once`)."""
  del config
  if not os.environ.get(LOG_ENV):
    return
  Path(os.environ[LOG_ENV]).mkdir(parents=True, exist_ok=True)
  import warp._src.context as warp_context

  from raywarp.core.trace.optix import pipeline

  build = warp_context.Module._compile

  @functools.wraps(build)  # the contracts read Warp's signature (`test_warp_private.py`)
  def compile_module(self: object, device: object = None, output_dir: object = None, *args: object, **kwargs: object) -> bool:
    with _timed() as t:
      compiled = build(self, device, output_dir, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]
    _record("warp", self.name, Path(str(output_dir)).name, t[0], compiled)  # pyrefly: ignore[missing-attribute]
    return compiled

  once = pipeline.compile_once

  @functools.wraps(once)
  def compile_ptx(path: Path | None, make: object) -> bytes:
    made: list[bool] = []

    def tracked() -> bytes:
      made.append(True)
      return make()  # pyrefly: ignore[not-callable]

    with _timed() as t:
      ptx = once(path, tracked)
    _record("ptx", "optix.ptx", "" if path is None else path.name, t[0], bool(made))
    return ptx

  warp_context.Module._compile = compile_module
  pipeline.compile_once = compile_ptx  # pyrefly: ignore[bad-assignment]


def pytest_runtest_logstart(nodeid: str, location: object) -> None:
  """The test the next loads belong to."""
  del location
  _TEST[0] = nodeid


# --- command line ---------------------------------------------------------------------------------


def main() -> int:
  """`key`, `restore DIR`, `save DIR`, `report` (module doc)."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  sub.add_parser("key")
  for name in ("restore", "save"):
    s = sub.add_parser(name)
    s.add_argument("directory", type=Path)
  sub.add_parser("report")
  a = p.parse_args()
  log = Path(os.environ.get(LOG_ENV, "compiles"))
  if a.cmd == "key":
    print(key())
  elif a.cmd == "restore":
    restore(a.directory)
  elif a.cmd == "save":
    save(a.directory, log)
  else:
    report(log)
  return 0


if __name__ == "__main__":
  sys.exit(main())
