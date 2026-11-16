"""tools/prebuilt/fill_all.sh across architectures: the job plan per architecture (OptiX: PTX
shards, census, packs on its GPU flavor; software: Warp modules only), and the DRY=1 orchestration
run end to end on this machine with stub jobs (each HF job a local process, each tool a stub
writing what the next job waits for; the release's layout, pins and publish are the real
tools/release/kernels.py): two architectures' bundles, one pack, one pins file."""

from __future__ import annotations

import json
import os
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
    while [ "$1" != -- ]; do [ "$1" != --tag ] || tag=$2; [ "$1" != --flavor ] || flavor=$2; shift; done
    rid="$(date -u +%F)-$tag"; dir=$jobs/$rid; mkdir -p "$dir/tmp"
    echo "$flavor" > "$dir/flavor"
    cmd=${2//$DATA/@DATA@}; cmd=${cmd//\/tmp\//$dir/tmp/}; cmd=${cmd//@DATA@/$DATA}
    ( cd "$STUB_REPO" && PATH="$STUB_BIN:$PATH" HF_DEADLINE=$(( $(date +%s) + 3600 )) bash -c "$cmd" > "$dir/log" 2>&1; echo $? > "$dir/exit" ) < /dev/null > /dev/null 2>&1 &
    echo "$rid" > "$dir/started"
    echo "run $rid" ;;
  ps) for d in "$jobs"/*/; do [ -f "$d/exit" ] || echo "x $(basename "$d")"; done ;;
  status) [ -f "$jobs/$4/exit" ] && echo "{\"exit\": $(cat "$jobs/$4/exit")}" ;;
esac
"""

# A job's `python`: the fill / build / census tools as stubs; tools.release.kernels the real one.
_PYTHON = r"""#!/usr/bin/env bash
if [ "$2" = tools.release.kernels ]; then exec "$REAL_PY" "$@"; fi
exec "$REAL_PY" "$STUB_BIN/tools_stub.py" "$@"
"""

_TOOLS = r"""
import json, shutil, sys
from pathlib import Path

a = sys.argv[1:]


def opt(name, default=None):
  return a[a.index(name) + 1] if name in a else default


def manifest(path, arch, driver=""):
  content = f"p{arch}" if driver else f"c{arch}"
  path.write_text(json.dumps({"arch": arch, "warp": WARP, "content": content, "driver": driver}))


WARP = __import__("warp").config.version
if a[:2] == ["-m", "tools.prebuilt.fill"]:
  cmd = a[2]
  if cmd == "list":
    out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
    i = opt("--part").split("/")[0]
    (out / f"part-{i}.json").write_text(json.dumps({"profile": opt("--profile"), "backend": opt("--backend")}))
  elif cmd == "fill":
    Path(opt("--root")).mkdir(parents=True, exist_ok=True)
    (Path(opt("--root")) / "args").write_text(" ".join(a))
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
    manifest(b / "manifest.json", arch)
    print(f"collect {b}")
  elif cmd == "merge":
    src = Path(a[a.index("--bundle") + 1])
    shutil.copytree(src, out / "raywarp_kernels" / src.name, dirs_exist_ok=True)
    print(f"merge {src.name} census {opt('--census')}")
  elif cmd == "pack":
    arch = int(Path(opt("--bundle")).name[2:])
    p = out / "raywarp_optix_modules" / f"sm{arch}-drv580.1"
    (p / "m").mkdir(parents=True, exist_ok=True)
    (p / "m" / "m0.db").write_bytes(b"rows")
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
  "buckets cp") case $3 in hf://buckets/blanchon/mujoco-warp-pbr/runs/*) cp "$DATA/runs/${3##*/runs/}" "$4" ;; esac ;;
  "buckets list") exit 0 ;;
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
  env = _env(tmp_path, ARCHS="120 89 86 90 80", PTX="2", WARP="1", CENSUS_JOBS="1", OPTIX_PARTS="1")
  scenes = tmp_path / "scenes.txt"
  scenes.write_text("fs_point1_plain_none_mesh_n\n")
  env |= {"SCENES_FILE": str(scenes), "HEAVY": ""}
  assert "--profile l4 " in _spec(env, "sm89-p0")[1] and "--backend optix --profile a10g" in _spec(env, "sm86-list-0")[1]
  head, w0 = _spec(env, "sm90-w0")
  assert head == "cpu-upgrade 75m" and "--warp --backend software --profile h100" in w0 and "--ptx-shards 0" in w0
  assert "--profiles rtx_pro_6000,l4,a10g" in _spec(env, "census-0")[1]
  assert "--census /tmp/census/sm89/*.json" in _spec(env, "sm89-merge")[1]
  assert "--census" not in _spec(env, "sm80-merge")[1]
  assert _spec(env, "sm120-pack-0")[0] == "rtx-pro-6000 35m" and "580.178.04" in _spec(env, "sm120-pack-0")[1]
  head, pack = _spec(env, "sm89-pack-0")
  assert head == "l4x1 35m" and "--shard 0/6" in pack and "--groups fast+default:own" in pack and "FATAL" not in pack
  assert _spec(env, "sm86-pack-0")[0] == "a10g-large 35m" and _spec(env, "sm80-release")[0].startswith("cpu-upgrade")
  plan = subprocess.run(
    ["bash", str(FILL_ALL), "cost", "t"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True
  )
  rows = {line.split()[0]: line for line in plan.stdout.splitlines()}
  assert set(rows) == {"120", "89", "86", "90", "80", "census", "total"}
  assert " 4 jobs" in rows["80"]  # list, Warp shard, merge, release


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
    "sm80-w0",
    "sm80-merge",
    "sm80-release",
    "census-0",
  }
  flavors = {p.parent.name.split("-t-", 1)[1]: p.read_text().strip() for p in (tmp_path / "jobs").glob("*/flavor")}
  assert flavors.pop("sm120-pack-0") == "rtx-pro-6000" and set(flavors.values()) == {"cpu-upgrade"}
  pins = json.loads((tmp_path / "state" / "t.kernels.json").read_text())
  assert pins["format"] == 3 and set(pins["bundles"]) == {"120", "80"}
  assert list(pins["bundles"]["120"]["packs"]) == ["580.1"] and pins["bundles"]["80"]["packs"] == {}
  calls = (tmp_path / "jobs" / "hf_calls").read_text()
  assert calls.count("buckets sync") == 3 and "hf://buckets/blanchon/raywarp-kernels/scratch/dry-t/sm80/" in calls
