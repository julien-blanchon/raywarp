"""The bucket's compiler / environment caches: `cache/<key>/<family>.<listing digest>.tar`.

`key` is `apt-<image and package list digest>` for the system packages, `<env>-uv` for the uv
cache and `<env>-<arch>-warp<version>` for the compiler families (entry.sh); every job restores the
newest 2 snapshots of its key's families and saves one new snapshot when its cache grew. `prune`
keeps the newest 2 per (key, family) and deletes `.partial` uploads of killed jobs. Files directly
under `cache/` are the pre-layout snapshots (legacy).
"""

from __future__ import annotations

import collections
import dataclasses
import datetime as dt
import re
from typing import Final

from tools.hf import hub
from tools.hf.config import CACHE, FAMILIES, KEEP_BY_FAMILY, KEEP_SNAPSHOTS, PARTIAL_MAX_AGE, PREPARED, RUN_TTL, RUNS

_SNAPSHOT: Final = re.compile(
  rf"{CACHE}/(?P<key>[a-z0-9.-]+)/(?P<family>[a-z]+)\.(?P<digest>[0-9a-f]{{8,64}})\.tar(?P<partial>(\.[^/]+)?\.partial)?"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
  """One cache archive."""

  file: hub.BucketFile
  key: str
  family: str
  partial: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Inventory:
  """The cache prefix, classified."""

  snapshots: list[Snapshot]
  legacy: list[hub.BucketFile]  # pre-layout files (cache/<name> and unknown names)


def inventory() -> Inventory:
  """Lists and classifies `cache/`."""
  snapshots: list[Snapshot] = []
  legacy: list[hub.BucketFile] = []
  for f in hub.list_files(CACHE):
    found = _SNAPSHOT.fullmatch(f.path)
    if found and found.group("family") in FAMILIES:
      snapshots.append(Snapshot(f, found.group("key"), found.group("family"), bool(found.group("partial"))))
    else:
      legacy.append(f)
  return Inventory(snapshots, legacy)


def plan_prune(snapshots: list[Snapshot], now: dt.datetime) -> list[Snapshot]:
  """What a prune deletes: all but the newest KEEP_SNAPSHOTS (a family's KEEP_BY_FAMILY) complete
  archives per (key, family), and partial uploads older than PARTIAL_MAX_AGE."""
  groups: dict[tuple[str, str], list[Snapshot]] = collections.defaultdict(list)
  out: list[Snapshot] = []
  for snap in snapshots:
    if snap.partial:
      if now - snap.file.mtime > PARTIAL_MAX_AGE:
        out.append(snap)
    else:
      groups[snap.key, snap.family].append(snap)
  for (_, family), group in groups.items():
    group.sort(key=lambda s: s.file.mtime, reverse=True)
    out += group[KEEP_BY_FAMILY.get(family, KEEP_SNAPSHOTS) :]
  return out


def _gb(n: int) -> str:
  return f"{n / 1e9:7.2f} GB"


def show(inv: Inventory) -> str:
  """Per (key, family): snapshots, size, newest; then the legacy files."""
  lines: list[str] = []
  groups: dict[tuple[str, str], list[Snapshot]] = collections.defaultdict(list)
  for snap in inv.snapshots:
    groups[snap.key, snap.family].append(snap)
  for (key, family), group in sorted(groups.items()):
    newest = max(s.file.mtime for s in group)
    partial = sum(s.partial for s in group)
    lines.append(
      f"{key:<36} {family:<9} {len(group) - partial:>2} snapshot(s) {_gb(sum(s.file.size for s in group))}"
      f"  newest {newest:%Y-%m-%d %H:%M}" + (f"  {partial} partial" if partial else "")
    )
  if inv.legacy:
    lines.append(f"legacy (pre-layout) files: {len(inv.legacy)}, {_gb(sum(f.size for f in inv.legacy)).strip()}")
    lines += [f"  {_gb(f.size)}  {f.mtime:%Y-%m-%d}  {f.path}" for f in sorted(inv.legacy, key=lambda f: f.path)]
  return "\n".join(lines) or "cache/ is empty"


def verify(inv: Inventory, now: dt.datetime) -> list[str]:
  """Problems: stale partials, too many snapshots, empty files, newest archives without a tar header
  (a range read of their first block)."""
  problems = [
    f"stale partial upload {s.file.path}" for s in inv.snapshots if s.partial and now - s.file.mtime > PARTIAL_MAX_AGE
  ]
  problems += [f"empty archive {s.file.path}" for s in inv.snapshots if s.file.size == 0]
  extra = [s for s in plan_prune(inv.snapshots, now) if not s.partial]
  if extra:
    problems.append(f"{len(extra)} snapshot(s) beyond the newest {KEEP_SNAPSHOTS} per family (run `cache prune`)")
  newest: dict[tuple[str, str], Snapshot] = {}
  for snap in inv.snapshots:
    if not snap.partial and snap.file.size >= 512:
      best = newest.get((snap.key, snap.family))
      if best is None or snap.file.mtime > best.file.mtime:
        newest[snap.key, snap.family] = snap
  for snap in newest.values():
    if hub.read_range(snap.file.path, 257, 5) != b"ustar":
      problems.append(f"not a tar archive: {snap.file.path}")
  return problems


def prune(inv: Inventory, now: dt.datetime, *, dry_run: bool) -> list[Snapshot]:
  """Deletes (or with `dry_run` only lists) what `plan_prune` selects."""
  doomed = plan_prune(inv.snapshots, now)
  for snap in doomed:
    print(
      f"{'would delete' if dry_run else 'delete'} {_gb(snap.file.size)}  {snap.file.mtime:%Y-%m-%d %H:%M}  {snap.file.path}"
    )
  print(f"{len(doomed)} file(s), {sum(s.file.size for s in doomed) / 1e9:.2f} GB")
  if doomed and not dry_run:
    hub.delete([s.file.path for s in doomed])
  return doomed


def plan_prune_runs(files: list[hub.BucketFile], now: dt.datetime) -> list[hub.BucketFile]:
  """Files of runs older than RUN_TTL without a `keep` marker, and stale `.partial` files of runs /
  prepared scenes."""
  by_run: dict[str, list[hub.BucketFile]] = collections.defaultdict(list)
  for f in files:
    if f.path.startswith(f"{RUNS}/") and f.path.count("/") >= 2:
      by_run[f.path.split("/")[1]].append(f)
  out = [f for f in files if f.path.endswith(".partial") and now - f.mtime > PARTIAL_MAX_AGE]
  for group in by_run.values():
    created = min(f.mtime for f in group)
    if now - created > RUN_TTL and not any(f.path.count("/") == 2 and f.path.endswith("/keep") for f in group):
      out += [f for f in group if f not in out]
  return out


def prune_runs(now: dt.datetime, *, dry_run: bool) -> list[hub.BucketFile]:
  """Deletes (or lists) what `plan_prune_runs` selects under runs/ and prepared/."""
  doomed = plan_prune_runs(hub.list_files(RUNS) + hub.list_files(PREPARED), now)
  runs = sorted({f.path.split("/")[1] for f in doomed if f.path.startswith(f"{RUNS}/")})
  partials = [f for f in doomed if f.path.endswith(".partial")]
  print(f"{'would delete' if dry_run else 'delete'}: {len(runs)} expired run(s) {runs[:10]}, {len(partials)} stale partial(s)")
  if doomed and not dry_run:
    hub.delete([f.path for f in doomed])
  return doomed
