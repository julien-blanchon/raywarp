#!/usr/bin/env bash
# The user's path, timed on a GPU: a fresh environment, `uv pip install` of the raywarp wheel (no
# build), then each scene's first frame in fresh caches (Warp, OptiX, CUDA, raywarp's, the kernel
# store), its prebuilt kernels downloaded from REMOTE on first use (first_frame.py).
#
#   tools/release/first_frame.sh OUT WHEELS REMOTE SCENE:BASE...
#
# WHEELS: a directory holding the raywarp wheel (release.yml's dist/). REMOTE: the kernels' remote,
# a URL, or a laid-out directory (tools/release/kernels.py layout), then served over HTTP on this
# machine. The checkout running this supplies the scene registry, and its uv.lock the versions of the
# mujoco and usd extras' packages (installed as constraints: a newer MuJoCo Warp from PyPI would
# compile its own modules and fail --expect-zero-compiles). Writes OUT/install.json and
# OUT/<scene>-<base>.json / .png. WARP_LANG: the Warp release to install (the bundle's: its
# kernels serve that release only). FIRST_FRAME_ARGS: more first_frame.py arguments (e.g.
# --expect-zero-compiles). Needs uv and a GPU.
set -euo pipefail
out=$1 wheels=$2 remote=$3
shift 3
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
mkdir -p "$out"
scratch=$(mktemp -d)
if [[ -d $remote ]]; then
  port=8765
  (cd "$remote" && exec python3 -m http.server "$port" --bind 127.0.0.1 > "$scratch/http.log" 2>&1) &
  server=$!
  trap 'kill $server 2>/dev/null || true' EXIT
  remote=http://127.0.0.1:$port
  sleep 1
fi
start=$(date +%s.%N)
uv venv -q --python 3.12 "$scratch/venv"
"$scratch/venv/bin/python" -I - "$repo/uv.lock" mujoco usd > "$scratch/constraints.txt" <<'PY'
# The locked versions of raywarp's EXTRAS (argv[2:]) from uv.lock (argv[1]): name==version lines.
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    packages = tomllib.load(f)["package"]
versions: dict[str, list[str]] = {}
for package in packages:
    versions.setdefault(package["name"], []).append(package["version"])
extras = next(p for p in packages if p["name"] == "raywarp")["optional-dependencies"]
for extra in sys.argv[2:]:
    for dep in extras[extra]:
        # A forked resolution (the newton-sim group's MuJoCo) names the extra's version.
        if "version" not in dep and len(versions[dep["name"]]) != 1:
            sys.exit(f"uv.lock: {dep['name']} has several versions and {extra} names none")
        print(f"{dep['name']}=={dep.get('version') or versions[dep['name']][0]}")
PY
echo "constraints: $(tr '\n' ' ' < "$scratch/constraints.txt")"
UV_CACHE_DIR=$scratch/uv uv pip install --python "$scratch/venv/bin/python" --no-build --find-links "$wheels" \
  --index-strategy unsafe-best-match --extra-index-url https://pypi.nvidia.com "raywarp[mujoco,usd]" pillow \
  --constraint "$scratch/constraints.txt" \
  ${WARP_LANG:+"warp-lang==$WARP_LANG"} \
  2>&1 | tail -3 | tee "$out/install.log"
end=$(date +%s.%N)
python3 -c "import json,sys; json.dump({'install_s': round($end - $start, 2)}, open('$out/install.json', 'w'))"
echo "install: $(cat "$out/install.json")"
for item in "$@"; do
  scene=${item%%:*} base=${item#*:}
  caches=$(mktemp -d -p "$scratch")
  (
    cd "$caches"
    WARP_CACHE_PATH=$caches/warp OPTIX_CACHE_PATH=$caches/optix CUDA_CACHE_PATH=$caches/cuda \
      RAYWARP_CACHE_DIR=$caches/wpbr RAYWARP_KERNELS=$caches/store RAYWARP_KERNELS_URL=$remote \
      "$scratch/venv/bin/python" -I "$here/first_frame.py" --out "$out" --repo "$repo" --scene "$scene" --base "$base" ${FIRST_FRAME_ARGS:-}
  ) 2>&1 | grep -E "^first_frame|^process|Error|error|raywarp:" | tee -a "$out/first_frame.log" || true
done
