"""Runs: one HF job per question. Upload the committed tree, launch `entry.sh job`, follow, fetch.

A run is `runs/<YYYY-MM-DD>-<tag>/` in the bucket: `code.tgz` + `code.sha256` (git archive of a
worktree's HEAD), `sources/<name>.tgz` (extra committed trees for A/B), `cmd.json`, then from the
job `run.log`, `stages.txt`, `out/` (small files, mirrored while it runs), `evidence.tgz` and
`status.json` (written last). The job carries the labels project / run / lane, so HF's job list
maps runs to jobs (no local registry).
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Final

from tools.hf import hub, jobs, spend
from tools.hf.config import FLAVORS, RUNS, bucket_uri, check_tag, pins


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
  """What to run."""

  tree: Path
  tag: str
  lane: str
  flavor: str
  timeout_s: int
  command: str
  env: str = "project"
  extras: tuple[str, ...] = ()
  sources: tuple[tuple[str, str], ...] = ()  # (name, git ref)
  stall_s: int = 600
  no_save: tuple[str, ...] = ()  # cache families the job does not save (scratch: entry.sh HF_NO_SAVE)
  no_restore: tuple[str, ...] = ()  # cache families the job does not restore (unused: entry.sh HF_NO_RESTORE)
  # Isolated run (`isolated.sh`): only the run's own directory is mounted (no bucket, no
  # environment, no caches); `inputs` (name, local file or directory) ship in it as inputs/<name>.
  isolated: bool = False
  inputs: tuple[tuple[str, str], ...] = ()
  # More storage buckets mounted (OWNER/NAME, mount path), e.g. the kernel build cache at /cache.
  mounts: tuple[tuple[str, str], ...] = ()


def git(tree: Path, *args: str) -> str:
  """`git -C tree <args>` stdout (stripped)."""
  return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, check=True).stdout.strip()


# History never ships: the development tree's knowledge base and results.
_HISTORY: Final = ("research", "results")


def archive(tree: Path, commit: str, out: Path) -> None:
  """Writes `commit`'s committed files as the tar.gz `out`, the development-only tree included
  (its `export-ignore`, .gitattributes, keeps it out of a plain `git archive`: the public export)
  but not its history (`_HISTORY`). A commit without it (the public tree) ships as it is."""
  excludes = [f":(exclude){h}" for h in _HISTORY]  # top level: a tree from before the split
  base = ["git", "-C", str(tree), "archive", "--format=tar", commit, "--", ".", ":(exclude)internal", *excludes]
  parts = [subprocess.run(base, capture_output=True, check=True).stdout]
  if (
    subprocess.run(
      ["git", "-C", str(tree), "cat-file", "-e", f"{commit}:internal"], capture_output=True, check=False
    ).returncode
    == 0
  ):
    cmd = ["git", "-C", str(tree), "archive", "--format=tar", "--prefix=internal/", f"{commit}:internal", "--", ".", *excludes]
    parts.append(subprocess.run(cmd, capture_output=True, check=True).stdout)
  with tarfile.open(out, "w:gz") as dst:
    for part in parts:
      with tarfile.open(fileobj=io.BytesIO(part)) as src:
        for member in src:
          dst.addfile(member, src.extractfile(member) if member.isfile() else None)


def default_lane(tree: Path) -> str:
  """The lane label of a worktree: its branch name, made a valid tag."""
  branch = git(tree, "rev-parse", "--abbrev-ref", "HEAD")
  return check_tag(re.sub(r"[^a-z0-9-]+", "-", branch.lower()).strip("-")[:48] or "detached")


def parse_duration(text: str) -> int:
  """Seconds of "90s", "25m", "2h" (a bare number: minutes)."""
  found = re.fullmatch(r"(\d+)([smh]?)", text.strip())
  if not found:
    raise ValueError(f"bad duration {text!r}: e.g. 25m, 90s, 1h")
  return int(found.group(1)) * {"s": 1, "m": 60, "h": 3600, "": 60}[found.group(2)]


def run_id(tag: str, now: dt.datetime) -> str:
  """`<YYYY-MM-DD>-<tag>`."""
  return f"{now.date().isoformat()}-{check_tag(tag)}"


def package(request: Request, directory: Path) -> dict[str, str | int | list[str]]:
  """Writes code.tgz, code.sha256, sources/*.tgz, sources.txt and cmd.json into `directory`;
  returns cmd.json's content."""
  commit = git(request.tree, "rev-parse", "HEAD")
  code = directory / "code.tgz"
  archive(request.tree, commit, code)  # committed files only, no history
  (directory / "code.sha256").write_text(f"{hashlib.sha256(code.read_bytes()).hexdigest()}  code.tgz\n")
  lines: list[str] = []
  for name, ref in request.sources:
    check_tag(name)
    sha = git(request.tree, "rev-parse", f"{ref}^{{commit}}")
    (directory / "sources").mkdir(exist_ok=True)
    source = str(directory / "sources" / f"{name}.tgz")
    archive(request.tree, sha, Path(source))
    lines.append(f"{name} {sha}")
  if lines:
    (directory / "sources.txt").write_text("\n".join(lines) + "\n")
  for name, local in request.inputs:
    check_tag(name)
    src, dst = Path(local), directory / "inputs" / name
    if not src.exists():
      raise FileNotFoundError(f"input {name}: {src} does not exist")
    dst.parent.mkdir(exist_ok=True)
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
  cmd: dict[str, str | int | list[str]] = {
    "command": request.command,
    "commit": commit,
    "flavor": request.flavor,
    "timeout_s": request.timeout_s,
    "lane": request.lane,
    "env": request.env,
    "stall_s": request.stall_s,
    "with": list(request.extras),
    "sources": lines,
    "isolated": request.isolated,
    "inputs": [name for name, _ in request.inputs],
    "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
  }
  (directory / "cmd.json").write_text(json.dumps(cmd, indent=1))
  return cmd


def encode_command(command: str) -> str:
  """The job's command as one base64 word (`HF_CMD_B64`, decoded by entry.sh): `hf jobs run --env`
  parses a value as a dotenv line, which drops everything after a newline or a " #"."""
  return base64.b64encode(command.encode()).decode()


def job_command(rid: str, commit: str, *, isolated: bool = False) -> list[str]:
  """The container command: verify and extract the uploaded tree, then `entry.sh job` (an isolated
  run: `isolated.sh`)."""
  run = f"/data/{RUNS}/{rid}"
  entry = f"isolated.sh {rid}" if isolated else f"entry.sh job {rid}"
  script = (
    f"set -e; cd {run} && sha256sum -c --quiet code.sha256; mkdir -p /work/src; tar xzf {run}/code.tgz -C /work/src; "
    f"echo {commit} > /work/src/COMMIT; exec bash /work/src/tools/hf/{entry}"
  )
  return ["bash", "-c", script]


def volumes(rid: str, *, isolated: bool, mounts: tuple[tuple[str, str], ...] = ()) -> list[str]:
  """The job's volumes: the whole bucket at /data, or (isolated) the run's own directory alone,
  at the same path (`/data/runs/<run>`: nothing else of the bucket is reachable from the job);
  then each of `mounts` (bucket, path), read-write."""
  if isolated:
    if mounts:
      raise ValueError("an isolated run mounts its own directory only")
    return [f"hf://buckets/{hub.BUCKET}/{RUNS}/{rid}:/data/{RUNS}/{rid}"]
  return [f"hf://buckets/{hub.BUCKET}:/data", *(f"hf://buckets/{b}:{path}" for b, path in mounts)]


def launch(request: Request, *, dry_run: bool = False, force_budget: bool = False) -> tuple[str, str]:
  """Uploads and starts a run; returns (run id, job id) (job id "" for a dry run)."""
  if request.flavor not in FLAVORS:
    raise ValueError(f"flavor {request.flavor}: one of {', '.join(FLAVORS)}")
  if git(request.tree, "status", "--porcelain", "--untracked-files=no"):
    print("warning: uncommitted changes are NOT uploaded (the run uses HEAD)", file=sys.stderr)
  rid = run_id(request.tag, dt.datetime.now(dt.UTC))
  worst = request.timeout_s / 3600 * jobs.prices()[request.flavor]
  verdict = spend.check(request.lane, worst, spend.ledger(), spend.budgets())
  if verdict:
    print(verdict, file=sys.stderr)
    if verdict.startswith("refuse") and not force_budget:
      raise SystemExit(3)
  with tempfile.TemporaryDirectory() as tmp:
    cmd = package(request, Path(tmp))
    if dry_run:
      print(json.dumps(cmd, indent=1))
      print(shlex.join(job_command(rid, str(cmd["commit"]), isolated=request.isolated)))
      print("volumes", *volumes(rid, isolated=request.isolated, mounts=request.mounts))
      return rid, ""
    if hub.exists(f"{RUNS}/{rid}/cmd.json"):
      raise ValueError(f"run {rid} exists: pick another --tag")
    hub.sync_up(Path(tmp), f"{RUNS}/{rid}")
  env = {
    "HF_RUN": rid,
    "HF_CMD_B64": encode_command(request.command),
    "HF_ENV": request.env,
    "HF_WITH": " ".join(request.extras),
    "HF_TIMEOUT_S": str(request.timeout_s),
    "HF_STALL_S": str(request.stall_s),
    "HF_FLAVOR": request.flavor,
    "HF_NO_SAVE": ",".join(request.no_save),
    "HF_NO_RESTORE": ",".join(request.no_restore),
  }
  job = jobs.launch(
    pins()["IMAGE"],
    job_command(rid, str(cmd["commit"]), isolated=request.isolated),
    volumes=volumes(rid, isolated=request.isolated, mounts=request.mounts),
    flavor=request.flavor,
    timeout_s=request.timeout_s,
    env=env,
    labels={"run": rid, "lane": request.lane},
    name=rid,
  )
  print(f"run {rid}  job {job}  ({request.flavor}, cap {request.timeout_s // 60} min, {bucket_uri(f'{RUNS}/{rid}')})")
  return rid, job


def find_job(run: str) -> jobs.Job:
  """The job of a run id, a sandbox name or a job id."""
  if re.fullmatch(r"[0-9a-f]{24}", run):
    return jobs.inspect(run)
  found = jobs.list_jobs(finished=True, label=f"run={run}") or jobs.list_jobs(finished=True, label=f"sandbox={run}")
  if not found:
    raise ValueError(f"no job labelled run={run} or sandbox={run}")
  return found[-1]


def running_jobs(names: list[str], running: list[jobs.Job]) -> dict[str, list[jobs.Job]]:
  """Each of `names` (run ids, sandbox names, job ids) -> its jobs among `running` (`kill`: one
  job list for every name, no label query per name)."""
  return {name: [j for j in running if name in (j.id, j.labels.get("run"), j.labels.get("sandbox"))] for name in names}


def stall_of(rid: str) -> int:
  """The run's in-job stall limit (cmd.json)."""
  return int(hub.number(hub.obj(json.loads(hub.read_bytes(f"{RUNS}/{rid}/cmd.json"))).get("stall_s")) or 600)


def status(rid: str) -> dict[str, hub.Json] | None:
  """The run's status.json (None until the job wrote it)."""
  path = f"{RUNS}/{rid}/status.json"
  return hub.obj(json.loads(hub.read_bytes(path))) if hub.exists(path) else None


# Consecutive failed polls `wait` rides out (a Hub hiccup must not end the follow, and with it the
# stall auto-kill and the fetch); more is an outage, raised.
WAIT_RETRIES: Final = 10


def wait(rid: str, job_id: str, *, stall_s: int, poll_s: int = 30) -> str:
  """Follows a job until it ends; cancels it when its log has not moved for `stall_s` (bootstrap or
  finish hangs the in-job watchdog cannot see). A failed poll is retried (`WAIT_RETRIES` in a row).
  Returns the final HF stage."""
  last, moved, errors = "", time.monotonic(), 0
  while True:
    try:
      job = jobs.inspect(job_id)
      tail = jobs.logs(job_id, tail=3) if job.status == "RUNNING" else ""
    except hub.HubError as e:
      errors += 1
      if errors > WAIT_RETRIES:
        raise
      print(f"poll {errors}/{WAIT_RETRIES} failed: {e}", flush=True)
      time.sleep(poll_s)
      continue
    errors = 0
    if job.status in jobs.FINAL:
      return job.status
    if f"{job.status}|{tail}" != last:
      last, moved = f"{job.status}|{tail}", time.monotonic()
      line = tail.strip().splitlines()[-1] if tail.strip() else ""
      print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {job.status} {line[:160]}", flush=True)
    elif job.status == "RUNNING" and time.monotonic() - moved > stall_s:
      print(f"no log progress for {stall_s} s: cancelling {rid} ({job_id})", flush=True)
      jobs.cancel(job_id)
    time.sleep(poll_s)


def fetch(rid: str, destination: Path) -> Path:
  """Copies the run directory (not the code, sources or inputs) into destination/<run>."""
  out = destination / rid
  hub.sync_down(f"{RUNS}/{rid}", out, "code.tgz", "sources/*", "inputs/*")
  return out


def summary(state: dict[str, hub.Json]) -> str:
  """One line of a status.json."""
  stages = hub.obj(state.get("stages_s") or {})
  timing = " ".join(f"{k}={hub.number(v):.0f}s" for k, v in stages.items())
  return f"exit {state.get('exit')} ({state.get('reason')}) on {state.get('gpu')}: {timing}"
