"""tools/prebuilt/fill_all.sh across architectures: the job plan per architecture (OptiX: PTX
shards, census, packs on its GPU flavor; software: Warp modules only), and the DRY=1 orchestration
run end to end on this machine with stub jobs (each HF job a local process, each tool a stub
writing what the next job waits for; the keys, the build cache's units, the cache references, the
release's layout, pins and publish are the real tools/prebuilt/buildkey.py, build_cache.py and
tools/release/kernels.py; the build cache bucket a local directory): two architectures' bundles,
one pack, one pins file; then the same release again (nothing launched) and one with one unit
changed (only it compiled, the census taken from the cache)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILL_ALL = ROOT / "tools" / "prebuilt" / "fill_all.sh"

# The orchestrator's `PY -m tools.hf ...`: `run` starts the job's command as a local process (its
# /tmp a directory of its own, the bucket mount $DATA), `ps` lists the running ones, `status` an
# ended one's exit; anything else is the real interpreter (tools.release.kernels pin / publish).
_HF = r"""#!/usr/bin/env bash
if [ "$2" != tools.hf ]; then exec "$REAL_PY" "$@"; fi
jobs=$STUB_JOBS; mkdir -p "$jobs"
case $3 in
  run)
    shift 3; tag=""
    with=""
    while [ "$1" != -- ]; do [ "$1" != --tag ] || tag=$2; [ "$1" != --flavor ] || flavor=$2; [ "$1" != --with ] || with="$with $2"; shift; done
    rid="$(date -u +%F)-$tag"; dir=$jobs/$rid
    if [ -e "$dir" ]; then echo "run $rid exists: pick another --tag" >&2; exit 1; fi  # as tools.hf
    mkdir -p "$dir/tmp"
    echo "$flavor" > "$dir/flavor"; echo "$with" > "$dir/with"
    cmd=${2//$DATA/@DATA@}; cmd=${cmd//$CACHE/@CACHE@}; cmd=${cmd//\/tmp\//$dir/tmp/}; cmd=${cmd//@DATA@/$DATA}; cmd=${cmd//@CACHE@/$CACHE}
    ( cd "$STUB_REPO" && PATH="$STUB_BIN:$PATH" HF_DEADLINE=$(( $(date +%s) + 3600 )) bash -c "$cmd" > "$dir/log" 2>&1; echo $? > "$dir/exit" ) < /dev/null > /dev/null 2>&1 &
    echo "$rid" > "$dir/started"
    echo "run $rid" ;;
  ps) for d in "$jobs"/*/; do [ -f "$d/exit" ] || echo "x $(basename "$d")"; done ;;
  status) [ -f "$jobs/$4/exit" ] && echo "{\"exit\": $(cat "$jobs/$4/exit")}" ;;
  kill) shift 3; for r in "$@"; do echo "$r" >> "$jobs/killed"; pkill -f -- "$jobs/$r/tmp/" || true; echo 137 > "$jobs/$r/exit"; done ;;
esac
"""

# A job's `python`: the fill / build / census tools as stubs; tools.release.kernels the real one.
_PYTHON = r"""#!/usr/bin/env bash
if [ "$2" = tools.release.kernels ] || [ "$2" = tools.prebuilt.build_cache ]; then exec "$REAL_PY" "$@"; fi
exec "$REAL_PY" "$STUB_BIN/tools_stub.py" "$@"
"""

_TOOLS = r"""
import json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # the repository (the job's directory): tools.prebuilt, raywarp
a = sys.argv[1:]


def opt(name, default=None):
  return a[a.index(name) + 1] if name in a else default


def manifest(path, arch, driver=""):
  content = f"p{arch}" if driver else f"c{arch}"
  path.write_text(json.dumps({"arch": arch, "warp": WARP, "content": content, "driver": driver}))


def compiled(arch, kind, unit):
  with open(Path(os.environ["STUB_JOBS"]) / "compiled", "a") as f:
    f.write(f"sm{arch} {kind} {unit}\n")


def fill_units(root, cache, arch, warp):
    # Each unit through the real build cache: taken when held, else "compiled" and kept.
    from raywarp.core.build.kernel_cache import kernel_code, prebuilt
    from tools.prebuilt.build_cache import BuildCache

    units = json.loads(Path(opt("--shard")).read_text())["units"]
    kept = BuildCache(Path(cache), arch)
    if warp:
      wcache = root / "warp" / WARP
      wcache.mkdir(parents=True, exist_ok=True)
      names = {f"wp_{u}_0123456": u for u in units}
      taken = kept.take_warp(WARP, names, wcache)
      for name in sorted(set(names) - taken):
        (wcache / name).mkdir()
        (wcache / name / f"{name}.sm{arch}.cubin").write_bytes(name.encode())
        kept.keep_warp(WARP, wcache / name)
        compiled(arch, "warp", names[name])
    else:
      ptx = root / "raywarp" / "ptx"
      ptx.mkdir(parents=True, exist_ok=True)
      for u in units:
        path = ptx / f"k{u}.ptx"
        if kept.ptx_entry(f"k{u}", path) is None:
          path.write_text(f"ptx {u}")
          reach = frozenset()
          kept.keep_ptx(f"k{u}", path, prebuilt.Entry(ptx="", code=kernel_code.digest(reach), reach=reach, groups={}))
          compiled(arch, "ptx", u)
    (root / "units.json").write_text(json.dumps(units))


WARP = __import__("warp").config.version
if a[:2] == ["-m", "tools.prebuilt.fill"]:
  cmd = a[2]
  if cmd == "list":
    out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
    i = opt("--part").split("/")[0]
    # Three units; --niche (FILL_ARGS) changes the third.
    units = ["a", "b", "c2" if "--niche" in a else "c"]
    (out / f"part-{i}.json").write_text(json.dumps({"profile": opt("--profile"), "backend": opt("--backend"), "units": units}))
  elif cmd == "spread":
    out = Path(opt("--out"))
    units = sorted({u for p in out.glob("part-*.json") for u in json.loads(p.read_text())["units"]})
    for kind in ("ptx", "warp"):
      n = int(opt(f"--{kind}-shards"))
      for i in range(n):
        (out / f"{kind}-{i}.json").write_text(json.dumps({"units": units[i::n]}))
  elif cmd == "fill":
    root = Path(opt("--root")); root.mkdir(parents=True, exist_ok=True)
    (root / "args").write_text(" ".join(a))
    arch = {"rtx_pro_6000": 120, "l4": 89, "a10g": 86, "h100": 90, "a100": 80}[opt("--profile")]
    if "--warp" in a and os.environ.get("STUB_FAIL") == f"w{arch}":
      sys.exit("RuntimeError: no NVRTC / nvJitLink 13 libraries")  # a Warp shard that always fails
    fill_units(root, opt("--cache"), arch, "--warp" in a)
  print(f"FILL {cmd} done")
elif a[:2] == ["-m", "tools.prebuilt.build"]:
  cmd = a[2]
  out = Path(opt("--out"))
  if cmd == "collect":
    arch = int(opt("--arch"))
    b = out / "raywarp_kernels" / f"sm{arch}"
    (b / "warp" / "wp_m_0123456").mkdir(parents=True, exist_ok=True)
    (b / "warp" / "wp_m_0123456" / f"wp_m_0123456.sm{arch}.cubin").write_bytes(b"cubin")
    if arch in (120, 89, 86):
      (b / "ptx").mkdir(exist_ok=True)
      (b / "ptx" / "k.ptx.xz").write_bytes(b"ptx")
      shutil.copyfile(Path(opt("--root")) / "units.json", b / "ptx" / "units.txt")  # what the pack makes
    manifest(b / "manifest.json", arch)
    print(f"collect {b}")
  elif cmd == "merge":
    src = Path(a[a.index("--bundle") + 1])
    shutil.copytree(src, out / "raywarp_kernels" / src.name, dirs_exist_ok=True)
    print(f"merge {src.name} census {opt('--census')}")
  elif cmd == "plan-pack":
    # The bundle's modules (one per unit) the kept modules lack.
    units = json.loads((Path(opt("--bundle")) / "ptx" / "units.txt").read_text())
    keep = Path(opt("--keep"))
    todo = [u for u in units if not (keep / "m" / f"m{u}.db").is_file()]
    out.write_text(f"{len(todo)}\n")
    print(f"pack plan: {len(todo)} of {len(units)}")
  elif cmd == "pack":
    arch = int(Path(opt("--bundle")).name[2:])
    units = json.loads((Path(opt("--bundle")) / "ptx" / "units.txt").read_text())
    p = out / "raywarp_optix_modules" / f"sm{arch}-drv580.1"
    (p / "m").mkdir(parents=True, exist_ok=True)
    keep = Path(opt("--keep").replace("$drv", "580.1"))
    for u in units:
      if (keep / "m" / f"m{u}.db").is_file():
        shutil.copyfile(keep / "m" / f"m{u}.db", p / "m" / f"m{u}.db")
      elif not opt("--assemble"):
        (p / "m" / f"m{u}.db").write_bytes(b"rows")
        (keep / "m").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p / "m" / f"m{u}.db", keep / "m" / f"m{u}.db")
        compiled(arch, "optix", u)
    manifest(p / "manifest.json", arch, "580.1")
    print(f"pack {p}")
  elif cmd == "merge-packs":
    src = Path(a[a.index("--pack") + 1])
    shutil.copytree(src, out / "raywarp_optix_modules" / src.name, dirs_exist_ok=True)
    print(f"pack merged {src.name}")
elif a[:2] == ["-m", "tools.prebuilt.word_reach"]:
  scene = opt("--scenes")
  arches = {"rtx_pro_6000": 120, "l4": 89, "a10g": 86}
  for profile in opt("--profiles").split(","):
    units = Path(opt("--units").replace("{arch}", str(arches[profile])))
    units.parent.mkdir(parents=True, exist_ok=True)
    units.write_text("[]")
    print(f"REACH {profile} {scene}")
else:
  sys.exit(f"stub: unexpected {a}")
"""

# The bucket CLI of `publish` (hf buckets cp from the stub mount) and of kernels.py (list, sync,
# cp).
_HF_CLI = r"""#!/usr/bin/env bash
echo "$*" >> "$STUB_JOBS/hf_calls"
case "$1 $2" in
  "buckets cp")
    case $3 in hf://buckets/blanchon/mujoco-warp-pbr/runs/*) cp "$DATA/runs/${3##*/runs/}" "$4" ;; esac
    case $4 in hf://buckets/$CACHE_BUCKET/*) dest=$CACHE/${4#hf://buckets/$CACHE_BUCKET/}; mkdir -p "$(dirname "$dest")"; cp "$3" "$dest" ;; esac ;;
  "buckets list")
    case $3 in "$CACHE_BUCKET"/*) cd "$CACHE" && find "${3#$CACHE_BUCKET/}" -type f 2>/dev/null; exit 0 ;; esac
    exit 0 ;;
esac
"""


def _env(tmp: Path, **extra: str) -> dict[str, str]:
  """The orchestrator's environment: its stubs, state and mount under `tmp`."""
  bin_ = tmp / "bin"
  bin_.mkdir(exist_ok=True)
  for name, text in (("py", _HF), ("python", _PYTHON), ("tools_stub.py", _TOOLS), ("hf", _HF_CLI)):
    (bin_ / name).write_text(text)
    (bin_ / name).chmod(0o755)
  (bin_ / "nvidia-smi").write_text("#!/bin/sh\necho 580.1\n")
  (bin_ / "nvidia-smi").chmod(0o755)
  return {
    **os.environ,
    "PATH": f"{bin_}:{os.environ['PATH']}",
    "PY": str(bin_ / "py"),
    "REAL_PY": sys.executable,
    "STUB_BIN": str(bin_),
    "STUB_JOBS": str(tmp / "jobs"),
    "STUB_REPO": str(ROOT),
    "DATA": str(tmp / "data"),
    "CACHE": str(tmp / "cache"),
    "CACHE_BUCKET": "owner/test-build-cache",
    "STATE": str(tmp / "state" / "t"),
    "POLL": "1",
    "WATCH_EVERY": "1",
    "SEED": "",
    **extra,
  }


def _spec(env: dict[str, str], job: str) -> tuple[str, str]:
  out = subprocess.run(
    ["bash", str(FILL_ALL), "spec", "t", job], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True
  )
  head, command = out.stdout.split("\n", 1)
  return head, command


def test_each_architecture_gets_its_backend_census_and_pack_flavor(tmp_path: Path) -> None:
  """OptiX architectures: PTX shards on their own profile, the census's per-architecture units,
  packs on their GPU flavor (the release driver checked where pinned); A100 / H100: Warp modules
  only (no PTX shard, census or pack); the census probes each scene once for every OptiX
  profile; `cost` sums every job's cap."""
  env = _env(tmp_path, ARCHS="120 89 86 90 80", PTX="2", WARP="1", WARP_SOFTWARE="3", CENSUS_JOBS="1", OPTIX_PARTS="1")
  scenes = tmp_path / "scenes.txt"
  scenes.write_text("fs_point1_plain_none_mesh_n\n")
  env |= {"SCENES_FILE": str(scenes), "HEAVY": ""}
  assert "--profile l4 " in _spec(env, "sm89-p0")[1] and "--backend optix --profile a10g" in _spec(env, "sm86-list-0")[1]
  head, w0 = _spec(env, "sm90-w0")
  assert head == "cpu-upgrade 90m" and "--warp --backend software --profile h100" in w0 and "--ptx-shards 0" in w0
  assert "--limit" not in w0 and _spec(env, "sm120-w0")[0] == "cpu-upgrade 75m"  # a release: every module
  # A dry run: one software Warp shard of a handful of modules (two word links), in minutes.
  head, dry = _spec(env | {"DRY": "1", "ARCHS": "120 80", "WARP_SOFTWARE": ""}, "sm80-w0")
  assert head == "cpu-upgrade 30m" and "--warp-shards 1 --limit 6 " in dry
  assert "--limit" not in _spec(env | {"DRY": "1", "ARCHS": "120 80"}, "sm120-w0")[1]
  assert "--profiles rtx_pro_6000,l4,a10g" in _spec(env, "census-0")[1]
  assert "--census /tmp/census/sm89/*.json" in _spec(env, "sm89-merge")[1]
  assert "--census" not in _spec(env, "sm80-merge")[1]
  assert _spec(env, "sm120-pack-0")[0] == "rtx-pro-6000 35m" and "620.70.06" in _spec(env, "sm120-pack-0")[1]
  head, pack = _spec(env, "sm89-pack-0")
  assert head == "l4x1 35m" and "--shard 0/6" in pack and "--groups fast+default:own" in pack and "FATAL" not in pack
  assert _spec(env, "sm86-pack-0")[0] == "a10g-large 35m" and _spec(env, "sm80-release")[0].startswith("cpu-upgrade")
  plan = subprocess.run(
    ["bash", str(FILL_ALL), "cost", "t"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True
  )
  rows = {line.split()[0]: line for line in plan.stdout.splitlines()}
  assert set(rows) == {"120", "89", "86", "90", "80", "census", "total"}
  assert " 6 jobs" in rows["80"] and " 45 jobs" not in rows["120"]  # list, 3 Warp shards, merge, release
  assert "--warp-shards 3" in w0 and "--warp-shards 1" in _spec(env, "sm120-w0")[1]  # software: WARP_SOFTWARE


def test_the_dry_refill_lays_out_and_pins_two_architectures(tmp_path: Path) -> None:
  """DRY=1 with one pack on sm_120: every job runs (locally, stubbed), each waits for its inputs
  on the mount, the merges hand the census to the OptiX bundle only, the release archives join
  into one layout, one pins file names both bundles (sm_120 with its pack's driver, sm_80
  without) and the scratch publish uploads both units of sm_120 and sm_80's bundle."""
  env = _env(tmp_path, DRY="1", PUBLISH="1", PACKS_120="1", DRIVER_120="580.1")
  (tmp_path / "data" / "runs").mkdir(parents=True)  # the bucket mount
  done = subprocess.run(
    ["bash", str(FILL_ALL), "refill", "t"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False
  )
  logs = "".join(f"\n--- {p.parent.name}\n{p.read_text()}" for p in sorted((tmp_path / "jobs").glob("*/log")))
  assert done.returncode == 0, done.stdout + done.stderr + logs
  ran = {p.parent.name.split("-t-", 1)[1] for p in (tmp_path / "jobs").glob("*/exit")}
  assert ran == {"sm120-list-0", "sm120-p0", "sm120-w0", "sm120-merge", "sm120-pack-0", "sm120-release"} | {
    "sm80-list-0",
    "sm80-w0",  # one shard of a handful of modules (WARP_LIMIT_SOFTWARE)
    "sm80-merge",
    "sm80-release",
    "census-0",
  }
  # Every Warp shard (and only they) installs the fill's NVRTC / nvJitLink (sm_80's failed without).
  withs = {p.parent.name.split("-t-", 1)[1]: p.read_text().strip() for p in (tmp_path / "jobs").glob("*/with")}
  assert {j for j, w in withs.items() if w == "fillcuda"} == {"sm120-w0", "sm80-w0"}
  assert set(withs.values()) == {"", "fillcuda"}
  flavors = {p.parent.name.split("-t-", 1)[1]: p.read_text().strip() for p in (tmp_path / "jobs").glob("*/flavor")}
  assert flavors.pop("sm120-pack-0") == "rtx-pro-6000" and set(flavors.values()) == {"cpu-upgrade"}
  pins = json.loads((tmp_path / "state" / "t.kernels.json").read_text())
  assert pins["format"] == 4 and set(pins["bundles"]) == {"120", "80"}
  assert list(pins["bundles"]["120"]["packs"]) == ["580.1"] and pins["bundles"]["80"]["packs"] == {}
  calls = (tmp_path / "jobs" / "hf_calls").read_text()
  assert calls.count("buckets sync") == 3 and "hf://buckets/blanchon/raywarp-kernels/scratch/dry-t/sm80/" in calls


def _refill(tmp: Path, tag: str, **extra: str) -> tuple[str, set[str], list[str]]:
  """A DRY refill of release `tag` on the shared mount, cache and pins of `tmp`: (its output,
  the jobs it launched, the units it compiled)."""
  env = _env(tmp, DRY="1", PUBLISH="1", PACKS_120="1", DRIVER_120="580.1", STATE=str(tmp / "state" / tag))
  env |= {"PINS": str(tmp / "pins.json"), "STUB_JOBS": str(tmp / "jobs" / tag), **extra}
  (tmp / "data" / "runs").mkdir(parents=True, exist_ok=True)
  done = subprocess.run(
    ["bash", str(FILL_ALL), "refill", tag], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False
  )
  jobs = tmp / "jobs" / tag
  logs = "".join(f"\n--- {p.parent.name}\n{p.read_text()}" for p in sorted(jobs.glob("*/log")))
  assert done.returncode == 0, done.stdout + done.stderr + logs
  ran = {p.parent.name.split(f"-{tag}-", 1)[1] for p in jobs.glob("*/exit")}
  made = (jobs / "compiled").read_text().split("\n")[:-1] if (jobs / "compiled").exists() else []
  return done.stdout, ran, sorted(made)


def test_a_release_compiles_only_what_no_earlier_fill_made(tmp_path: Path) -> None:
  """Three DRY releases on one build cache and one pins file: the first compiles every unit (3 PTX
  on sm_120, 3 Warp modules on each architecture, 3 OptiX modules on one pack job) and pins each
  architecture with its key; the second, nothing changed, launches no job; the third changes one
  unit (a fill option: the bundle and listing keys change, the census key does not): only that
  unit compiles (1 PTX, 1 Warp module per architecture, 1 OptiX module), the census launches no
  job (its rows come from the cache) and the pins carry the new keys; the fourth reaches no unit:
  nothing compiles, no pack job."""
  _, ran, made = _refill(tmp_path, "t1")
  assert "census-0" in ran and "sm120-pack-0" in ran
  assert made == sorted(
    f"sm{a} {kind} {u}" for a, kind in ((120, "ptx"), (120, "warp"), (80, "warp"), (120, "optix")) for u in "abc"
  )
  pins = json.loads((tmp_path / "pins.json").read_text())
  keys = json.loads((tmp_path / "state" / "t1.keys.json").read_text())
  assert pins["format"] == 4 and {a: b["key"] for a, b in pins["bundles"].items()} == keys
  refs = json.loads((tmp_path / "data" / "runs" / f"{_day()}-t1-sm120-refs.json").read_text())
  assert refs["arch"] == 120 and refs["optix"] == ["580.1/ma", "580.1/mb", "580.1/mc"]

  out, ran, made = _refill(tmp_path, "t2")
  assert "nothing changed" in out and ran == set() and made == []

  _, ran, made = _refill(tmp_path, "t3", FILL_ARGS="--niche")
  assert made == ["sm120 optix c2", "sm120 ptx c2", "sm120 warp c2", "sm80 warp c2"]
  assert not any(j.startswith("census") for j in ran) and "sm120-pack-0" in ran
  pins = json.loads((tmp_path / "pins.json").read_text())
  assert {a: b["key"] for a, b in pins["bundles"].items()} == json.loads((tmp_path / "state" / "t3.keys.json").read_text())
  assert pins["bundles"]["120"]["key"] != keys["120"]

  # A change that reaches no unit (another point selection the stub lists alike): every CPU job
  # runs and compiles nothing, no pack job (the release assembles the pack from the cache).
  only = "scene=fs_point0_plain_none_mesh_n,tier=fast,layout=shared,gi=False,outputs=shaded,set=env"
  _, ran, made = _refill(tmp_path, "t4", FILL_ARGS="--niche", ONLY=only)
  assert made == [] and "sm120-release" in ran and not any("pack" in j or j.startswith("census") for j in ran)
  assert list(json.loads((tmp_path / "pins.json").read_text())["bundles"]["120"]["packs"]) == ["580.1"]


def _day() -> str:
  """Today (UTC), as the run ids name it."""
  import datetime as dt

  return dt.datetime.now(dt.UTC).date().isoformat()


def test_in_ci_a_job_out_of_relaunches_fails_the_fill_at_once(tmp_path: Path) -> None:
  """CI (FAIL_FAST=1): the sm_80 Warp shard fails every launch; after RETRIES relaunches the refill
  kills every other running job of the wave (sm_80's merge waits for that shard: without the kill
  it waited to its cap, the 6 h workflow with it) and exits nonzero with the reason, in seconds.
  Rerun after the fix, the same state resumes: the failed and killed jobs relaunch under new run
  names (GitHub run 43976510973 instead failed at once on the restored state's exhausted shard);
  a state of another build is refused."""
  env = _env(tmp_path, DRY="1", CI="true", RETRIES="1", STUB_FAIL="w80", PINS=str(tmp_path / "pins.json"))
  (tmp_path / "data" / "runs").mkdir(parents=True)
  done = subprocess.run(
    ["bash", str(FILL_ALL), "refill", "t"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240, check=False
  )
  assert done.returncode == 1, done.stdout + done.stderr
  assert re.search(r"FILL FAILED: sm80-w\d failed after 1 relaunches", done.stdout) and "wave 1 incomplete" in done.stdout
  killed = (tmp_path / "jobs" / "killed").read_text().split()
  assert any(k.endswith("-t-sm80-merge") for k in killed)
  tries = (tmp_path / "state" / "t.tries").read_text().split("\n")[:-1]
  assert tries and all(line.startswith("sm80-w") for line in tries)  # only the failing shards relaunched

  # The fix, then the same build again (the workflow resumes its saved state): the failed and the
  # killed jobs relaunch (relaunch counts reset) under run names never used, and the fill ends.
  env.pop("STUB_FAIL")
  again = subprocess.run(
    ["bash", str(FILL_ALL), "refill", "t"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240, check=False
  )
  assert again.returncode == 0, again.stdout + again.stderr
  assert "relaunch sm80-w0" in again.stdout and "relaunch sm80-merge" in again.stdout
  assert "exists" not in again.stdout + again.stderr
  assert (tmp_path / "state" / "t.stage").read_text().split() == ["wave1", "wave2"]

  # Another build (here: other architectures) never resumes this state.
  other = subprocess.run(
    ["bash", str(FILL_ALL), "refill", "t"],
    cwd=ROOT,
    env=env | {"ARCHS": "120"},
    capture_output=True,
    text=True,
    timeout=60,
    check=False,
  )
  assert other.returncode == 2 and "FILL REFUSED" in other.stderr and "belongs to another build" in other.stderr


def test_a_dry_dispatch_with_the_default_input_keeps_the_two_dry_architectures(tmp_path: Path) -> None:
  """kernels.yml's architecture step: a dry dispatch with the default input leaves ARCHS unset
  (fill_all.sh's dry pair, sm_120 + sm_80; dry run 43978938600 built all five), a real one or a dry
  one with chosen architectures sets it; the tag names the dispatch (dry) or the build (real)."""
  import yaml

  workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "kernels.yml").read_text())
  (step,) = [s for s in workflow["jobs"]["fill"]["steps"] if s.get("name", "").startswith("The fill's architectures")]

  def run(**env: str) -> dict[str, str]:
    out = tmp_path / "github_env"
    out.write_text("")
    base = {
      "GITHUB_ENV": str(out),
      "GITHUB_RUN_ID": "77",
      "GITHUB_REF_NAME": "main",
      "GITHUB_SHA": "0123456789abcdef",
      "FORCE": "0",
    }
    subprocess.run(["bash", "-euo", "pipefail", "-c", step["run"]], env={"PATH": os.environ["PATH"], **base, **env}, check=True)
    return dict(line.split("=", 1) for line in out.read_text().splitlines())

  default = "120 89 86 90 80"
  assert run(DRY="1", ARCHS_IN=default) == {"TAG": "kernels-dry-77"}
  assert run(DRY="1", ARCHS_IN="120")["ARCHS"] == "120"
  real = run(DRY="0", ARCHS_IN=default)
  assert real["ARCHS"] == default and real["TAG"].startswith("kernels-main-01234567-")
