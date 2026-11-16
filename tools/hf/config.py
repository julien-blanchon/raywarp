"""Constants of the HF tooling: the bucket, its layout, the allowed flavors and the pins."""

from __future__ import annotations

import datetime as dt
import re
import shlex
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
HERE: Final = Path(__file__).resolve().parent
BUCKET: Final = "blanchon/mujoco-warp-pbr"
MOUNT: Final = "/data"
PROJECT_LABEL: Final = "raywarp"
# Local state (price table copy, fetched runs default under the worktree's internal/results/hf).
STATE: Final = Path.home() / ".cache" / "raywarp" / "hf"

# Flavors a lane may use (operating manual I18): CPU suites / preparation, the default GPU, headline
# RTX numbers, the Warp path without RT cores. l40sx1 is the L4 overflow (same sm_89 cache family,
# so warm); a10g-large the second overflow (sm_86: its own, colder cache family). cpu-performance
# (32 vCPU) runs the full CPU suite (`tools.hf test --full`); it shares cpu-upgrade's cache family.
FLAVORS: Final = ("cpu-upgrade", "cpu-performance", "l4x1", "l40sx1", "a10g-large", "rtx-pro-6000", "a100-large")
ENVS: Final = ("project", "mjlab")
EXTRAS: Final = ("ovrtx",)

# Bucket layout (README "Bucket layout").
RUNS: Final = "runs"
CACHE: Final = "cache"
PREPARED: Final = "prepared"
SPEND: Final = "spend"
LEDGER: Final = f"{SPEND}/ledger.jsonl"
BUDGETS: Final = f"{SPEND}/budgets.json"
CANCELS: Final = f"{SPEND}/cancels"  # <job id>: written when a job is cancelled (its end time)
# Cache families and how many snapshots per (environment key, family) a prune keeps. `tests`: the
# test jobs' own kernel snapshots, one per kernel-code digest (`tests/kernel_snapshot.py`, which
# keeps the newest `test_cache.KEEP`; a prune leaves those).
FAMILIES: Final = ("apt", "uv", "warp", "cuda", "wpbr", "ovrtx", "hyperlux", "tests")
KEEP_SNAPSHOTS: Final = 2
KEEP_BY_FAMILY: Final = {"tests": 6}
# A `.partial` upload older than this belongs to a killed job.
PARTIAL_MAX_AGE: Final = dt.timedelta(hours=2)
# Runs older than this are deleted by `cache prune` unless runs/<id>/keep exists.
RUN_TTL: Final = dt.timedelta(days=14)

_TAG: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")


def pins() -> dict[str, str]:
  """The KEY=value pairs of tools/hf/pins.env (comments and blank lines skipped)."""
  out: dict[str, str] = {}
  for line in (HERE / "pins.env").read_text().splitlines():
    if line.strip() and not line.lstrip().startswith("#"):
      key, _, value = line.partition("=")
      out[key.strip()] = " ".join(shlex.split(value))
  return out


def check_tag(tag: str) -> str:
  """`tag` if it is a valid run tag / lane name (lowercase, digits, '-'), else ValueError."""
  if not _TAG.fullmatch(tag):
    raise ValueError(f"invalid tag {tag!r}: lowercase letters, digits and '-', at most 48 characters")
  return tag


def bucket_uri(path: str = "") -> str:
  """The hf:// URI of `path` in the project bucket."""
  return f"hf://buckets/{BUCKET}/{path}".rstrip("/")
