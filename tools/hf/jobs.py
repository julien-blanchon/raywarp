"""HF Jobs of the project: listing, launch, logs, cancel, and the hardware price table."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from typing import Final

from tools.hf import hub
from tools.hf.config import CANCELS, PROJECT_LABEL, STATE

FINAL: Final = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


@dataclasses.dataclass(frozen=True, slots=True)
class Job:
  """One HF job (sandboxes are jobs too)."""

  id: str
  name: str
  flavor: str
  status: str
  labels: dict[str, str]
  created: dt.datetime
  started: dt.datetime | None
  running_s: float
  timeout_s: float  # the run's hard cap (HF_TIMEOUT_S of `tools.hf run`), 0 when unknown

  @property
  def lane(self) -> str:
    """The lane label ("" for jobs launched without one)."""
    return self.labels.get("lane", "")

  @property
  def run(self) -> str:
    """The run label (`tools.hf run`), the sandbox name, or the job name."""
    return self.labels.get("run") or self.labels.get("sandbox") or self.name

  def minutes(self, now: dt.datetime, cancelled: dt.datetime | None = None) -> float | None:
    """Billed minutes (running jobs: so far). A cancelled job has no duration in HF's record: its
    end is `cancelled` (the time `tools.hf` recorded when it cancelled it), else unknown (None)."""
    if self.started is None:
      return 0.0
    if self.status not in FINAL:
      return (now - self.started).total_seconds() / 60
    if self.running_s or self.status != "CANCELED":
      return self.running_s / 60
    return None if cancelled is None else max(0.0, (cancelled - self.started).total_seconds() / 60)

  def remaining_minutes(self, now: dt.datetime) -> float:
    """Minutes a running / scheduled job may still bill (its cap minus what it used; 0 when done
    or when the cap is unknown)."""
    if self.status in FINAL or not self.timeout_s:
      return 0.0
    used = (now - self.started).total_seconds() / 60 if self.started else 0.0
    return max(0.0, self.timeout_s / 60 - used)


def _job(row: dict[str, hub.Json]) -> Job:
  labels = {k: hub.text(v) for k, v in hub.obj(row.get("labels") or {}).items()}
  status = row.get("status")
  stage = hub.text(hub.obj(status).get("stage")) if isinstance(status, dict) else hub.text(status)
  durations = hub.obj(row.get("durations") or {})
  created = hub.timestamp(row.get("created_at"))
  if created is None:
    raise hub.HubError(f"job record without created_at: {row}")
  return Job(
    id=hub.text(row.get("id")),
    name=labels.get("name", hub.text(row.get("name"))),
    flavor=hub.text(row.get("flavor")),
    status=stage,
    labels=labels,
    created=created,
    started=hub.timestamp(row.get("started_at")),
    running_s=hub.number(durations.get("running_secs")),
    timeout_s=float(hub.text(hub.obj(row.get("environment") or {}).get("HF_TIMEOUT_S")) or 0),
  )


def list_jobs(*, finished: bool = False, label: str = f"project={PROJECT_LABEL}") -> list[Job]:
  """The project's jobs (running / scheduling; `finished`: all), oldest first."""
  args = ["jobs", "ps", "--label", label, "--limit", "0", *(["-a"] if finished else [])]
  rows = hub.items(hub.hf_json(*args) or [])
  return sorted((_job(hub.obj(row)) for row in rows), key=lambda job: job.created)


def inspect(job_id: str) -> Job:
  """The current record of one job."""
  rows = hub.items(hub.hf_json("jobs", "inspect", job_id) or [])
  return _job(hub.obj(rows[0]))


def logs(job_id: str, tail: int | None = None) -> str:
  """The job's log so far (the last `tail` lines)."""
  return hub.hf("jobs", "logs", job_id, *(["--tail", str(tail)] if tail else []), timeout=120)


def cancel(job_id: str) -> None:
  """Cancels a job (billing stops) and records when (HF keeps no end time for cancelled jobs)."""
  hub.hf("jobs", "cancel", job_id)
  hub.write(f"{CANCELS}/{job_id}", dt.datetime.now(dt.UTC).isoformat().encode())


def cancel_times(ids: list[str]) -> dict[str, dt.datetime]:
  """Job id -> when `tools.hf` (or a sandbox watchdog) cancelled it, for the ids with a marker."""
  marked = {f.path.rsplit("/", 1)[1] for f in hub.list_files(CANCELS)}
  out: dict[str, dt.datetime] = {}
  for job_id in set(ids) & marked:
    when = hub.timestamp(hub.read_bytes(f"{CANCELS}/{job_id}").decode().strip())
    if when is not None:
      out[job_id] = when
  return out


def launch(
  image: str,
  command: list[str],
  *,
  volumes: list[str],
  flavor: str,
  timeout_s: int,
  env: dict[str, str],
  labels: dict[str, str],
  name: str,
) -> str:
  """Starts a detached job with its volumes (`runs.volumes`: the bucket or a run directory, and
  any other bucket mounted); returns its id."""
  args = ["jobs", "run", "--detach", "--flavor", flavor, "--timeout", f"{timeout_s}s", "--attempts", "1", "--name", name]
  for volume in volumes:
    args += ["--volume", volume]
  for key, value in env.items():
    args += ["--env", f"{key}={value}"]
  for key, value in {"project": PROJECT_LABEL, **labels}.items():
    args += ["--label", f"{key}={value}"]
  out = hub.hf(*args, "-q", image, *command, timeout=180)
  found = re.findall(r"\b[0-9a-f]{24}\b", out)
  if not found:
    raise hub.HubError(f"hf jobs run printed no job id: {out[-500:]}")
  return found[-1]


def prices() -> dict[str, float]:
  """$ per hour of every flavor (`hf jobs hardware`; refreshed once a day into the local state)."""
  cached = STATE / "hardware.json"
  today = dt.datetime.now(dt.UTC).date().isoformat()
  if cached.exists():
    saved = hub.obj(json.loads(cached.read_text()))
    if saved.get("date") == today:
      return {k: hub.number(v) for k, v in hub.obj(saved.get("prices")).items()}
  rows = hub.items(hub.hf_json("jobs", "hardware") or [])
  table: dict[str, float] = {}
  for entry in rows:
    row = hub.obj(entry)
    table[hub.text(row.get("name"))] = float(hub.text(row.get("cost/hour")).lstrip("$"))
  STATE.mkdir(parents=True, exist_ok=True)
  cached.write_text(json.dumps({"date": today, "prices": table}))
  return table
