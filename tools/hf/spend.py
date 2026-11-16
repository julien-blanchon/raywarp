"""Spend tracker: each project job's billed minutes × its flavor's price, per lane, vs budgets.

HF's own job records (label `project=raywarp`, including sandboxes) are the source of truth: the
ledger (`spend/ledger.jsonl` in the bucket) is a snapshot of them, rewritten whole by `sync`, so
concurrent writers cannot lose rows. Budgets (`spend/budgets.json`, $ per lane) are set by the
coordinator; a launch warns when a lane's spend plus the new job's worst case passes 80 % of its
budget and refuses when it passes the budget.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from typing import Final

from tools.hf import hub, jobs
from tools.hf.config import BUDGETS, LEDGER

# The consolidation plan's budget period starts here (operating manual: $50 total from this point).
PLAN_START: Final = dt.datetime(2026, 11, 2, 22, 0, tzinfo=dt.UTC)
WARN_FRACTION: Final = 0.8


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
  """One ledger row."""

  job: str
  run: str
  lane: str
  flavor: str
  status: str
  created: str
  minutes: float  # -1: a cancelled job of unknown end
  usd: float
  reserved_usd: float  # what a running job may still bill up to its cap


def rows(all_jobs: list[jobs.Job], price: dict[str, float], now: dt.datetime, cancelled: dict[str, dt.datetime]) -> list[Row]:
  """Ledger rows of `all_jobs` created since the plan start (a cancelled job of unknown end: -1
  minutes, flagged in the summary)."""
  out: list[Row] = []
  for job in all_jobs:
    if job.created < PLAN_START:
      continue
    known = job.minutes(now, cancelled.get(job.id))
    minutes = -1.0 if known is None else known
    rate = price.get(job.flavor)
    if rate is None:
      raise KeyError(f"no price for flavor {job.flavor} (job {job.id})")
    out.append(
      Row(
        job.id,
        job.run,
        job.lane or "untagged",
        job.flavor,
        job.status,
        job.created.isoformat(),
        round(minutes, 2),
        round(max(minutes, 0.0) * rate / 60, 4),
        round(job.remaining_minutes(now) * rate / 60, 4),
      )
    )
  return out


def ledger() -> list[Row]:
  """The current ledger, from HF's job records."""
  found = jobs.list_jobs(finished=True)
  open_ended = [j.id for j in found if j.status == "CANCELED" and not j.running_s and j.created >= PLAN_START]
  return rows(found, jobs.prices(), dt.datetime.now(dt.UTC), jobs.cancel_times(open_ended))


def sync() -> list[Row]:
  """Rewrites the bucket ledger from HF's records; returns it."""
  current = ledger()
  hub.write(LEDGER, "".join(json.dumps(dataclasses.asdict(row)) + "\n" for row in current).encode())
  return current


def budgets() -> dict[str, float]:
  """$ budget per lane ({} when none is set)."""
  if not hub.exists(BUDGETS):
    return {}
  return {k: hub.number(v) for k, v in hub.obj(json.loads(hub.read_bytes(BUDGETS))).items()}


def set_budget(lane: str, usd: float) -> None:
  """Sets one lane's budget."""
  current = budgets()
  current[lane] = usd
  hub.write(BUDGETS, json.dumps(current, indent=1, sort_keys=True).encode())


def per_lane(current: list[Row]) -> dict[str, tuple[float, float]]:
  """lane -> (minutes, $)."""
  out: dict[str, tuple[float, float]] = {}
  for row in current:
    minutes, usd = out.get(row.lane, (0.0, 0.0))
    out[row.lane] = (minutes + max(row.minutes, 0.0), usd + row.usd)
  return out


def check(lane: str, worst_case_usd: float, current: list[Row], limits: dict[str, float]) -> str:
  """An empty string when `lane` may spend `worst_case_usd` more, "warn: ..." near its budget or
  with cancelled jobs of unknown duration, "refuse: ..." when the launch could exceed it. Running
  jobs count at their full cap (`reserved_usd`): parallel launches cannot overshoot together."""
  if lane not in limits:
    return ""
  mine = [row for row in current if row.lane == lane]
  spent = sum(row.usd for row in mine)
  reserved = sum(row.reserved_usd for row in mine)
  budget = limits[lane]
  total = spent + reserved + worst_case_usd
  unknown = sum(1 for row in mine if row.minutes < 0)
  note = f" ({unknown} cancelled job(s) of unknown duration not counted)" if unknown else ""
  if total > budget:
    return (
      f"refuse: lane {lane} spent ${spent:.2f} + running ${reserved:.2f} + this job ${worst_case_usd:.2f}"
      f" > budget ${budget:.2f}{note}"
    )
  if total > WARN_FRACTION * budget or unknown:
    return (
      f"warn: lane {lane} at ${spent:.2f} + running ${reserved:.2f} + this job ${worst_case_usd:.2f} of ${budget:.2f}{note}"
    )
  return ""


def summary(current: list[Row], limits: dict[str, float]) -> str:
  """A per-lane table (minutes, $, budget) and the total since the plan start."""
  lines = [f"{'lane':<24} {'jobs':>5} {'minutes':>9} {'$':>8} {'budget':>8}"]
  lanes = per_lane(current)
  counts = {lane: sum(1 for row in current if row.lane == lane) for lane in lanes}
  for lane, (minutes, usd) in sorted(lanes.items(), key=lambda kv: -kv[1][1]):
    budget = f"{limits[lane]:.2f}" if lane in limits else "-"
    flag = "  <-- over 80 %" if lane in limits and usd > WARN_FRACTION * limits[lane] else ""
    unknown = sum(1 for row in current if row.lane == lane and row.minutes < 0)
    flag += f"  ({unknown} cancelled job(s) of unknown duration not counted)" if unknown else ""
    lines.append(f"{lane:<24} {counts[lane]:>5} {minutes:>9.1f} {usd:>8.3f} {budget:>8}{flag}")
  total = sum(usd for _, usd in lanes.values())
  lines.append(f"{'total since ' + PLAN_START.strftime('%Y-%m-%d %H:%M'):<30} {total:>18.3f}")
  return "\n".join(lines)
