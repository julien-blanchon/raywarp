"""`python -m tools.hf`: HF jobs, caches and spend (README.md).

The development repository's `python -m internal.tools.hf` adds the dev sandboxes and the test
selection (`sandbox`, `test`) on top of this parser (`main(extend=...)`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from tools.hf import cache, hub, jobs, runs, spend
from tools.hf.config import ENVS, EXTRAS, FLAVORS, ROOT


def job_command(rest: list[str]) -> str:
  """The command after `--` (one string, run by bash in the job)."""
  words = rest[1:] if rest[:1] == ["--"] else rest
  if not words:
    raise SystemExit("a command is required after --")
  return words[0] if len(words) == 1 else shlex.join(words)  # one argument: a shell line


def worktree(path: str | None) -> Path:
  """The top of the git worktree holding `path` (default: the current directory)."""
  return Path(runs.git(Path(path or "."), "rev-parse", "--show-toplevel"))


def cmd_run(a: argparse.Namespace) -> int:
  """`run`: package the committed tree, launch, and with `--wait` follow and fetch."""
  tree = worktree(a.tree)
  if a.isolated and (a.source or a.extra or a.env != "project" or a.no_cache_save):
    print(
      "--isolated runs without the bucket, environment and caches: drop --source / --with / --env / --no-cache-save",
      file=sys.stderr,
    )
    return 2
  if a.input and not a.isolated:
    print("--input ships with --isolated runs only (a normal run reads the bucket)", file=sys.stderr)
    return 2
  request = runs.Request(
    tree=tree,
    tag=a.tag,
    lane=a.lane or runs.default_lane(tree),
    flavor=a.flavor,
    timeout_s=runs.parse_duration(a.timeout),
    command=job_command(a.command),
    env=a.env,
    extras=tuple(a.extra),
    sources=tuple(tuple(s.split("=", 1)) for s in a.source),
    stall_s=runs.parse_duration(a.stall),
    no_save=tuple(f for f in a.no_cache_save.split(",") if f),
    no_restore=tuple(f for f in a.no_cache_restore.split(",") if f),
    isolated=a.isolated,
    inputs=tuple(tuple(i.split("=", 1)) for i in a.input),
  )
  rid, job = runs.launch(request, dry_run=a.dry_run, force_budget=a.over_budget)
  if not job or not a.wait:
    return 0
  return follow(rid, job, request.stall_s + 300, Path(a.fetch_to))


def follow(rid: str, job: str, stall_s: int, destination: Path) -> int:
  """Follows run `rid` (job `job`) to its end, fetches it, syncs the ledger; 0: it exited 0."""
  stage = runs.wait(rid, job, stall_s=stall_s)
  state = runs.status(rid)
  out = runs.fetch(rid, destination)
  spend.sync()
  print(f"{rid}: HF {stage}; " + (runs.summary(state) if state else "no status.json (killed before the finish)"))
  print(f"fetched -> {out}")
  return 0 if state and state.get("exit") == 0 else 1


def cmd_wait(a: argparse.Namespace) -> int:
  """`wait`: follow a launched run (`_follow`)."""
  job = runs.find_job(a.run)
  stall = runs.parse_duration(a.stall) if a.stall else runs.stall_of(job.run) + 300
  return follow(job.run, job.id, stall, Path(a.fetch_to))


def cmd_logs(a: argparse.Namespace) -> int:
  """`logs`: the job log (`-f`: streamed)."""
  job = runs.find_job(a.run)
  if a.follow:
    return subprocess.run(["hf", "jobs", "logs", "-f", job.id], check=False).returncode
  print(jobs.logs(job.id, tail=a.tail), end="")
  return 0


def cmd_fetch(a: argparse.Namespace) -> int:
  """`fetch`: copy the run directory locally and print its status."""
  out = runs.fetch(a.run, Path(a.to))
  state = runs.status(a.run)
  print(f"{out}" + (f"\n{runs.summary(state)}" if state else "\n(no status.json yet)"))
  return 0


def cmd_status(a: argparse.Namespace) -> int:
  """`status`: the run's status.json."""
  state = runs.status(a.run)
  print(json.dumps(state, indent=1) if state else "no status.json yet")
  return 0


def cmd_ps(a: argparse.Namespace) -> int:
  """`ps`: the project's jobs and sandboxes, their minutes and $."""
  now = dt.datetime.now(dt.UTC)
  price = jobs.prices()
  found = jobs.list_jobs(finished=a.all)
  if not found:
    print("no running raywarp job or sandbox")
  for job in found[-a.limit :]:
    minutes = job.minutes(now) or 0.0  # a cancelled job of unknown end: 0 here (the ledger flags it)
    print(
      f"{job.id}  {job.status:<10} {job.flavor:<13} {job.lane or '-':<16} {minutes:6.1f} min "
      f"${minutes * price.get(job.flavor, 0) / 60:6.3f}  {job.run}"
    )
  return 0


def cmd_kill(a: argparse.Namespace) -> int:
  """Cancels the running jobs of runs / sandboxes / job ids `a.runs` (an argument may hold several,
  whitespace-separated: a shell variable that did not split), resolved against one job list; an
  unknown one or a failed cancel (a job that ended meanwhile) is reported, the others still
  cancelled."""
  wanted = [w for arg in a.runs for w in arg.split()]
  found = runs.running_jobs(wanted, jobs.list_jobs())
  failed = 0
  for run in wanted:
    for job in found.get(run, []):
      try:
        jobs.cancel(job.id)
      except hub.HubError as e:
        print(f"{job.id} ({job.run}): {e}", file=sys.stderr)
        failed += 1
        continue
      print(f"cancelled {job.id} ({job.run})")
  missing = [run for run in wanted if not found.get(run)]
  for run in missing:
    print(f"{run}: no running job of that run, sandbox or id", file=sys.stderr)
  return 1 if missing or failed else 0


def cmd_cache(a: argparse.Namespace) -> int:
  """`cache show / verify / prune`."""
  inv = cache.inventory()
  now = dt.datetime.now(dt.UTC)
  match a.action:
    case "show":
      print(cache.show(inv))
    case "verify":
      problems = cache.verify(inv, now)
      print("\n".join(problems) or "cache OK")
      return 1 if problems else 0
    case "prune":
      cache.prune(inv, now, dry_run=a.dry_run)
      cache.prune_runs(now, dry_run=a.dry_run)
  return 0


def cmd_spend(a: argparse.Namespace) -> int:
  """`spend show / sync / budget`."""
  match a.action:
    case "budget":
      spend.set_budget(a.lane, a.usd)
      print(f"budget {a.lane}: ${a.usd:.2f}")
      return 0
    case "sync":
      current = spend.sync()
    case _:
      current = spend.ledger()
  limits = spend.budgets()
  if a.rows:
    for row in current:
      if not a.lane or row.lane == a.lane:
        print(
          f"{row.created[:16]}  {row.job}  {row.flavor:<13} {row.status:<10} {row.minutes:6.1f} min ${row.usd:6.3f}  "
          f"{row.lane:<14} {row.run}"
        )
  print(spend.summary([r for r in current if not a.lane or r.lane == a.lane], limits))
  return 0


type Sub = argparse._SubParsersAction[argparse.ArgumentParser]


def parser(prog: str, extend: Callable[[Sub], None] | None) -> argparse.ArgumentParser:
  """The CLI: runs, their jobs, caches and spend; `extend` adds subcommands."""
  p = argparse.ArgumentParser(prog=prog, description=__doc__)
  sub = p.add_subparsers(dest="cmd", required=True)
  _run_parser(sub)
  _job_parsers(sub)
  _scene_parsers(sub)
  _spend_parser(sub)
  if extend is not None:
    extend(sub)
  return p


FETCH_TO: Final = str(ROOT / "internal" / "results" / "hf")


def _run_parser(sub: Sub) -> None:
  """`run`."""
  r = sub.add_parser("run", help="upload the committed tree and run one command on an HF job")
  r.add_argument("--tag", required=True, help="run name: runs/<date>-<tag>/")
  r.add_argument("--flavor", choices=FLAVORS, default="cpu-upgrade")
  r.add_argument("--timeout", default="20m", help="hard cap of the whole job (e.g. 20m, 1h)")
  r.add_argument("--stall", default="10m", help="stop the command when its output stops for this long")
  r.add_argument("--env", choices=ENVS, default="project")
  r.add_argument("--with", dest="extra", action="append", choices=EXTRAS, default=[])
  r.add_argument("--source", action="append", default=[], metavar="NAME=REF", help="extra committed tree ($HF_SOURCES/NAME)")
  r.add_argument("--tree", help="worktree to upload (default: the current one)")
  r.add_argument("--lane", help="spend label (default: the branch name)")
  r.add_argument("--wait", action="store_true", help="follow, auto-kill on a stall, fetch")
  r.add_argument("--fetch-to", default=FETCH_TO)
  r.add_argument("--dry-run", action="store_true", help="package and print, launch nothing")
  r.add_argument("--over-budget", action="store_true", help="launch even when the lane budget would be exceeded")
  r.add_argument(
    "--no-cache-save",
    default="",
    metavar="FAMILIES",
    help="cache families not saved at the end (scratch), e.g. warp,cuda,wpbr",
  )
  r.add_argument(
    "--no-cache-restore",
    default="",
    metavar="FAMILIES",
    help="cache families not restored (the job does not read them: a fill compiles into its own root), e.g. warp",
  )
  r.add_argument(
    "--isolated",
    action="store_true",
    help="mount the run's own directory only (no bucket, environment or caches): third-party code (tools/hf/isolated.sh)",
  )
  r.add_argument(
    "--input",
    action="append",
    default=[],
    metavar="NAME=PATH",
    help="isolated runs: a local file or directory ($HF_INPUTS/NAME)",
  )
  r.add_argument("command", nargs=argparse.REMAINDER)
  r.set_defaults(fn=cmd_run)


def _job_parsers(sub: Sub) -> None:
  """`logs`, `fetch`, `status`, `wait`, `ps`, `kill`."""
  s = sub.add_parser("logs", help="job log")
  s.add_argument("run", help="run id (<date>-<tag>) or job id")
  s.add_argument("-f", "--follow", action="store_true")
  s.add_argument("--tail", type=int)
  s.set_defaults(fn=cmd_logs)
  s = sub.add_parser("fetch", help="copy runs/<run>/ locally")
  s.add_argument("run", help="run id (<date>-<tag>) or job id")
  s.add_argument("--to", default=FETCH_TO)
  s.set_defaults(fn=cmd_fetch)
  s = sub.add_parser("status", help="print status.json")
  s.add_argument("run", help="run id (<date>-<tag>) or job id")
  s.set_defaults(fn=cmd_status)
  s = sub.add_parser("wait", help="follow until done, then fetch")
  s.add_argument("run", help="run id (<date>-<tag>) or job id")
  s.add_argument("--stall", help="default: the run's --stall + 5 min")
  s.add_argument("--fetch-to", default=FETCH_TO)
  s.set_defaults(fn=cmd_wait)
  s = sub.add_parser("ps", help="running raywarp jobs (and sandboxes)")
  s.add_argument("-a", "--all", action="store_true")
  s.add_argument("--limit", type=int, default=30)
  s.set_defaults(fn=cmd_ps)
  s = sub.add_parser("kill", help="cancel runs / jobs")
  s.add_argument("runs", nargs="+")
  s.set_defaults(fn=cmd_kill)


def _scene_parsers(sub: Sub) -> None:
  """`cache`."""
  s = sub.add_parser("cache", help="bucket caches: show / verify / prune (newest 2 per family)")
  s.add_argument("action", choices=("show", "verify", "prune"))
  s.add_argument("--dry-run", action="store_true")
  s.set_defaults(fn=cmd_cache)


def _spend_parser(sub: Sub) -> None:
  """`spend show / sync / budget`."""
  money = sub.add_parser("spend", help="spend ledger and lane budgets").add_subparsers(dest="action", required=True)
  for action, doc in (("show", "per-lane totals (from HF's job records)"), ("sync", "show + rewrite spend/ledger.jsonl")):
    s = money.add_parser(action, help=doc)
    s.add_argument("--lane", help="only this lane")
    s.add_argument("--rows", action="store_true", help="one line per job")
    s.set_defaults(fn=cmd_spend)
  s = money.add_parser("budget", help="set a lane budget ($)")
  s.add_argument("lane")
  s.add_argument("usd", type=float)
  s.set_defaults(fn=cmd_spend)


def main(
  argv: list[str] | None = None, *, prog: str = "python -m tools.hf", extend: Callable[[Sub], None] | None = None
) -> int:
  """The CLI (`extend`: more subcommands); a Hub, value or subprocess error is one `error:` line
  and exit 2."""
  args = parser(prog, extend).parse_args(argv)
  try:
    return int(args.fn(args))
  except (hub.HubError, ValueError, KeyError) as e:
    print(f"error: {e}", file=sys.stderr)
    return 2
  except subprocess.CalledProcessError as e:
    print(f"error: {' '.join(map(str, e.cmd))} failed: {(e.stderr or '').strip()[-500:]}", file=sys.stderr)
    return 2
