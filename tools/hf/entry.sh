#!/usr/bin/env bash
# The one entry of every HF job and dev sandbox (`python -m tools.hf`; README.md).
#
#   entry.sh job RUN       inside a `tools.hf run` job: bootstrap, restore caches, run the command
#                          under a cap with a stall watchdog and live mirroring, then finish
#   entry.sh bootstrap     (sandbox) the same environment and caches; writes $HF_WORK/env.sh
#   entry.sh save-caches   (sandbox) saves the caches that changed since the bootstrap
#
# One environment per HF_ENV (project: this repository's uv lock; mjlab: the pinned mjlab + HyperLux
# competitor environment), one cache directory per family, every pin from pins.env. Finish order
# (no SIGKILL path loses results): small files mirrored, evidence.tgz, caches (.partial + rename),
# status.json LAST. A job's own HF timeout is the hard cap; the command gets what remains minus
# the finish reserve. Inputs (job env): HF_RUN, HF_CMD_B64 (the command, base64: HF's --env values
# are dotenv lines, which cut a command at a newline or a " #"), HF_ENV, HF_WITH, HF_TIMEOUT_S, HF_STALL_S,
# HF_MIN_CAP_S (the shortest command cap worth starting, default 30 s; the tests' short jobs lower it).
set -euo pipefail
mode=${1:?usage: entry.sh job RUN | bootstrap | save-caches}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
# shellcheck source=tools/hf/pins.env
. "$HERE/pins.env"
DATA=${HF_DATA:-/data}
WORK=${HF_WORK:-/work}
DRY=${HF_DRY:-0}  # tests: no system setup, no GPU, the caller's python3
ENV_NAME=${HF_ENV:-project}
WITH=" ${HF_WITH:-} "
OUT=$WORK/out
T_START=$(date +%s)
FINISH_RESERVE=${HF_FINISH_RESERVE:-300}  # seconds kept for mirror + evidence + cache save + status
SAVE_CAP=180
# Cache families this job does not save (`tools.hf run --no-cache-save`): a fill / list / census
# job's caches are scratch (a 16 GB Warp family tarred at the end evicted most fill shards).
NO_SAVE=" ${HF_NO_SAVE:-} "; NO_SAVE=${NO_SAVE//,/ }
# Cache families this job does not restore (`tools.hf run --no-cache-restore`): a fill compiles into
# its own root and a test job restores its own kernel cache (`tests/kernel_snapshot.py`); the CPU `warp`
# family is ~25 GB (1.5-13 min) of other trees' modules. Not restored is not saved either.
NO_RESTORE=" ${HF_NO_RESTORE:-} "; NO_RESTORE=${NO_RESTORE//,/ }
RAYWARP_CACHE_MB=${HF_RAYWARP_CACHE_MB:-6144}  # the raywarp family's cap (the oldest files go first)
# The CPU `warp` family's cap: every tree's modules accumulate there (a 25 GB family, 122 snapshots
# before the save-time prune: BUG-289); the oldest modules go first, a miss recompiles.
WARP_CPU_CACHE_MB=${HF_WARP_CPU_CACHE_MB:-12288}
# Minutes a snapshot that left the restored window (the newest 2, uv 1) stays readable: a restore
# that started before it left reads it for at most this long (restores take 1-15 min).
RESTORE_MAX_MIN=30
EVIDENCE_CAP=$((200 << 20))
MIRROR_MAX=$((2 << 20))
declare -A DUR=() RESTORED=() SAVED=()
STAGE_NAME="" STAGE_T0=$T_START
REASON=ok
GPU_NAME=cpu GPU_CC="" ARCH=cpu RT=0 PY="" CMD_PID=""
mkdir -p "$WORK" "$OUT" "$WORK/tmp"

stage() {  # NAME: closes the current stage's timing, prints the marker
  local now; now=$(date +%s)
  [[ -z "$STAGE_NAME" ]] || DUR[$STAGE_NAME]=$((now - STAGE_T0))
  STAGE_NAME=$1 STAGE_T0=$now
  echo "STAGE $1 $(date -u +%FT%TZ) disk $(df -h "$WORK" 2>/dev/null | awk 'NR == 2 {print $3 " used, " $4 " free"}')"
  echo "$1 $now" >> "$WORK/stages.txt"
}

# --- GPU capability (not the name): compute capability and RT cores -------------------------------
# RT cores: every NVIDIA GPU of compute capability >= 7.5 except the compute-only datacenter parts
# GV100 (7.0), GA100 (8.0), GH100 (9.0), GB100/GB200 (10.0), GB300 (10.3).
detect_gpu() {
  GPU_NAME=cpu GPU_CC="" ARCH=cpu RT=0
  if [[ "$DRY" != 1 ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    ARCH=sm${GPU_CC/./}
    local major=${GPU_CC%%.*} minor=${GPU_CC#*.}
    case $GPU_CC in
      7.0|8.0|9.0|10.0|10.3) RT=0 ;;
      *) if (( major > 7 || (major == 7 && minor >= 5) )); then RT=1; fi ;;
    esac
  fi
  echo "gpu: $GPU_NAME cc=${GPU_CC:-none} arch=$ARCH rt_cores=$RT"
  # A GPU flavor without a visible GPU is a broken machine, never a CPU run.
  if [[ "$ARCH" == cpu && -n "${HF_FLAVOR:-}" && "$HF_FLAVOR" != cpu-* ]]; then
    echo "FATAL: flavor $HF_FLAVOR but no GPU visible (nvidia-smi)"; return 1
  fi
}

# --- caches: $DATA/cache/<key>/<family>.<listing sha>.tar, newest 2 restored ------------------------
cache_dir() {  # FAMILY -> the local directory of that cache
  case $1 in
    uv) echo "$WORK/cache/uv" ;;
    warp) echo "$WORK/cache/warp" ;;
    cuda) echo "$WORK/cache/cuda" ;;
    raywarp) echo "$WORK/cache/raywarp" ;;
    hyperlux) echo "$WORK/cache/hyperlux" ;;
    apt) echo "$WORK/cache/apt" ;;  # lists/ + archives/: the system packages' indexes and .deb files
    ovrtx) "$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/ovrtx/bin/cache")' ;;
  esac
}
# Never in a snapshot: prepared `.rayw` scenes (bucket prepared/, staged per job) and in-flight writes.
SKIP=(! -name '*.rayw' ! -name '*.tmp*' ! -name '*.partial')
# Saved but not a reason for a new snapshot: raywarp's per-file code digest memo (it changes with
# every commit; a job recomputes what it lacks in about a second).
QUIET=(! -name 'code-digests-*')
files() {  # DIR -> its file list (relative path + size), sorted
  (cd "$1" && find . -type f "${SKIP[@]}" "${QUIET[@]}" -printf '%P %s\n' | LC_ALL=C sort)
}
cap_dir() {  # DIR MAX_MB: deletes the oldest saved files (modification time: a restored file keeps its
  # archive's) until what a snapshot saves fits (access times are useless: extraction sets them,
  # relatime then rarely updates them)
  local dir=$1 max=$(($2 << 20)) total
  total=$(find "$dir" -type f "${SKIP[@]}" -printf '%s\n' | awk '{s += $1} END {print s + 0}')
  (( total > max )) || return 0
  find "$dir" -type f "${SKIP[@]}" -printf '%T@ %s %p\n' | sort -n | while read -r _ size path; do
    (( total > max )) || break
    rm -f "$path"; total=$((total - size))
  done
}
keep_of() {  # FAMILY -> the snapshots a job restores (`restore`; tools/hf/config.KEEP_BY_FAMILY)
  [[ $1 == uv ]] && echo 1 || echo 2
}
prune_family() {  # KEY FAMILY: deletes the family's snapshots no restore reads any more: those with
  # `keep_of` newer ones, the newest of which (the one that pushed them out) is RESTORE_MAX_MIN old
  local key=$1 family=$2 keep i out_at
  keep=$(keep_of "$family")
  mapfile -t snaps < <(ls -t "$DATA/cache/$key/$family".*.tar 2>/dev/null)
  for ((i = keep; i < ${#snaps[@]}; i++)); do
    out_at=$(stat -c %Y "${snaps[i - keep]}" 2>/dev/null) || continue
    (( $(date +%s) - out_at > RESTORE_MAX_MIN * 60 )) && rm -f "${snaps[i]}" && echo "pruned $(basename "${snaps[i]}")"
  done
}
listing() {  # DIR -> digest of its file list: changes when anything is added or resized
  files "$1" | sha256sum | cut -c1-16
}
restore() {  # KEY FAMILY...
  local key=$1 family dir tar t0; shift
  for family in "$@"; do
    t0=$(date +%s)
    dir=$(cache_dir "$family"); mkdir -p "$dir" "$WORK/stamps"
    if [[ "$NO_RESTORE" == *" $family "* ]]; then
      RESTORED[$family]="skipped (--no-cache-restore)"; echo "restore $key/$family: skipped"; continue
    fi
    # The newest 2 snapshots (concurrent compiles each add coverage); uv: the newest only (every
    # snapshot is a superset of what its job restored, and uv snapshots are multi-GB).
    mapfile -t found < <(ls -t "$DATA/cache/$key/$family".*.tar 2>/dev/null | head -"$([[ $family == uv ]] && echo 1 || echo 2)")
    for ((i=${#found[@]}-1; i>=0; i--)); do tar xf "${found[i]}" -C "$dir"; done
    RESTORED[$family]="${#found[@]} snapshot(s) $(du -sm "$dir" | cut -f1) MB $(( $(date +%s) - t0 )) s"
    echo "restored $key/$family: ${RESTORED[$family]}"
    files "$dir" > "$WORK/stamps/$family.files"
    sha256sum < "$WORK/stamps/$family.files" | cut -c1-16 > "$WORK/stamps/$family"
    echo "$key" > "$WORK/stamps/$family.key"
  done
}
save_caches() {  # every restored family whose listing changed: local tar -> .partial -> rename
  local stamp family key dir digest out result
  for stamp in "$WORK"/stamps/*.key; do
    [[ -e "$stamp" ]] || continue
    family=$(basename "$stamp" .key); key=$(cat "$stamp"); dir=$(cache_dir "$family")
    [[ "$family" != raywarp ]] || cap_dir "$dir" "$RAYWARP_CACHE_MB"
    [[ "$family" != warp || "$ARCH" != cpu ]] || cap_dir "$dir" "$WARP_CPU_CACHE_MB"
    if [[ "$NO_SAVE" == *" $family "* ]]; then
      result="skipped (scratch: --no-cache-save)"
    elif [[ -z "$(find "$dir" -type f -print -quit 2>/dev/null)" ]]; then
      result=empty
    elif digest=$(listing "$dir") && [[ "$digest" == "$(cat "$WORK/stamps/$family")" ]]; then
      result=unchanged
    else
      out="$DATA/cache/$key/$family.$digest.tar"
      mkdir -p "$DATA/cache/$key"
      # A writer-unique partial name: two jobs saving the same snapshot never interleave bytes.
      local partial="$out.$(hostname).$$.partial"
      # What grew since the restore (the first new paths, to spot volatile cache entries).
      local grown
      grown=$(files "$dir" | LC_ALL=C comm -13 "$WORK/stamps/$family.files" - | awk '{print $1}')
      # Streamed to the bucket (no local copy: a second copy of a 16 GB family filled the disk).
      if [[ -e "$out" ]] || { (cd "$dir" && find . -type f "${SKIP[@]}" -print0 | tar cf - --null -T -) > "$partial" \
        && { [[ -e "$out" ]] && rm -f "$partial" || mv "$partial" "$out"; }; }; then
        result="$(basename "$out") $(( $(stat -c %s "$out") >> 20 )) MB, $(grep -c . <<< "$grown") new/resized files"
        result+=" e.g. $(head -3 <<< "$grown" | tr '\n' ' ')"
        echo "$digest" > "$WORK/stamps/$family"
      else
        result=failed
        rm -f "$partial"
      fi
    fi
    echo "$result" > "$WORK/stamps/$family.saved"
    echo "saved $key/$family: $result"
    [[ "$NO_SAVE" == *" $family "* ]] || prune_family "$key" "$family"
  done
}

# --- environments -------------------------------------------------------------------------------
# System packages. The image lacks a few; the Ubuntu mirror is slow at times (apt-get update /
# install past 8 min), so the packages' indexes and .deb files are a cache family (`apt`, keyed by
# the image and the package list): a job with a snapshot installs offline, and only a miss (first
# job, or a snapshot the image's packages outgrew) reaches the mirror, with retries.
APT_TRIES=${HF_APT_TRIES:-4}
APT_BACKOFF_S=${HF_APT_BACKOFF_S:-30}  # the wait before the second try; doubled at each further try
apt_step() {  # NAME CAP_S CMD...: CMD under CAP_S, retried with backoff; each try's time is logged
  local name=$1 cap=$2 try t0 rc; shift 2
  for ((try = 1; ; try++)); do
    t0=$(date +%s)
    if timeout "$cap" "$@" >> "$WORK/apt.log" 2>&1; then
      echo "apt $name: ok in $(( $(date +%s) - t0 )) s (try $try)"; return 0
    else
      rc=$?
    fi
    echo "apt $name: exit $rc$([[ $rc != 124 ]] || echo " (over ${cap} s)") after $(( $(date +%s) - t0 )) s (try $try of $APT_TRIES)"
    if (( try >= APT_TRIES )); then
      echo "FATAL: apt $name failed; the end of apt.log:"; tail -n 20 "$WORK/apt.log"; return "$rc"
    fi
    sleep $(( APT_BACKOFF_S << (try - 1) ))
  done
}

system_packages() {  # the image's missing packages: from the apt cache family, else the mirror
  [[ "$DRY" != 1 || "${HF_DRY_APT:-0}" == 1 ]] || return 0  # tests: HF_DRY_APT=1 with a stub apt-get / dpkg
  local pkgs=(git ca-certificates curl libegl1 libgl1 libgles2 libglvnd0 libxrender1 libxext6 libosmesa6 ffmpeg)
  [[ "$ENV_NAME" != mjlab ]] || pkgs+=(build-essential cmake)
  [[ "$WITH" != *" ovrtx "* ]] || pkgs+=(libopengl0)
  command -v g++ >/dev/null 2>&1 || pkgs+=(build-essential)
  ! dpkg -s "${pkgs[@]}" >/dev/null 2>&1 || return 0
  stage packages
  export DEBIAN_FRONTEND=noninteractive
  local dir key
  dir=$(cache_dir apt)
  key=apt-$(printf '%s\n' "$IMAGE" "${pkgs[@]}" | LC_ALL=C sort | sha256sum | cut -c1-16)
  restore "$key" apt
  mkdir -p "$dir/archives/partial" "$dir/lists/partial"
  # Our own directories: the image's docker-clean hook deletes /var/cache/apt/archives/*.deb only.
  local apt=(apt-get -qq -o "Dir::Cache::archives=$dir/archives" -o "Dir::State::lists=$dir/lists"
    -o APT::Keep-Downloaded-Packages=true)
  local install=(install -y --no-install-recommends "${pkgs[@]}")
  local t0; t0=$(date +%s)
  if timeout 300 "${apt[@]}" "${install[@]}" --no-download >> "$WORK/apt.log" 2>&1; then
    echo "apt install: ok from the cache in $(( $(date +%s) - t0 )) s"; return 0
  fi
  echo "apt install: the cache lacks packages ($(( $(date +%s) - t0 )) s): the mirror"
  apt_step update 180 "${apt[@]}" update
  apt_step install 420 "${apt[@]}" "${install[@]}"
}

system_setup() {
  [[ "$DRY" != 1 ]] || return 0
  install -m 755 "$UV_BIN" /usr/local/bin/uv
  # The Isaac assets reference their dome textures as absolute /NVIDIA/... paths.
  [[ ! -d "$DATA/assets/isaac51/NVIDIA" || -e /NVIDIA ]] || ln -s "$DATA/assets/isaac51/NVIDIA" /NVIDIA
}

export_common() {
  export UV_CACHE_DIR=$WORK/cache/uv UV_PYTHON_INSTALL_DIR=$WORK/cache/uv/python UV_NO_SYNC=1
  # Warp reads WARP_CACHE_PATH; the benchmark scripts set wp.config.kernel_cache_dir from WARP_KERNEL_CACHE_DIR.
  # Both resolve to $WORK/cache/warp/<warp version>.
  export WARP_CACHE_PATH=$WORK/cache/warp WARP_KERNEL_CACHE_DIR=$WORK/cache/warp
  # raywarp's cache directory: textures, bakes, MDL conversions and OptiX's module cache (`optix/`).
  export CUDA_CACHE_PATH=$WORK/cache/cuda RAYWARP_CACHE_DIR=$WORK/cache/raywarp
  export HYPERLUX_CACHE_DIR=$WORK/cache/hyperlux
  # The native build hook (tools/native/hook.py, run by the editable install): the pinned NVIDIA
  # artifacts from the bucket (no download), the compiled MDL binding shared by every job.
  export RAYWARP_NATIVE_ARCHIVES=$NATIVE_ARCHIVES RAYWARP_NATIVE_CACHE=$NATIVE_CACHE
  export MJPBR_ASSET_DIR=$DATA/assets MJPBR_HOST_CACHE=$WORK/raywarp_cache
  export MUJOCO_GL=$([[ "$ARCH" == cpu ]] && echo osmesa || echo egl)
  mkdir -p "$MJPBR_HOST_CACHE"
}

env_project() {
  export UV_PROJECT_ENVIRONMENT=$WORK/venv
  # The tree itself is installed outside the uv cache: its editable build differs per commit and would
  # make every job save a new multi-GB uv snapshot (measured: the only change between runs).
  (cd "$SRC" && uv sync -q --locked --no-install-project --python "$PYTHON" --group dev --group bench --extra usd --extra mujoco)
  PY=$WORK/venv/bin/python
  uv pip install -q --python "$PY" --no-deps --no-cache -e "$SRC"
  if [[ "$RT" == 1 ]]; then
    # RT cores: OptiX is required (no silent software downgrade): the driver's OptiX and the headers.
    "$PY" -c 'from raywarp.core.pipeline.platform.device import device_caps; s = device_caps("cuda:0").optix; assert s.available, s.reason' \
      || { echo "FATAL: the OptiX traversal cannot run on this RT-core GPU"; return 1; }
  fi
  if [[ "$WITH" == *" ovrtx "* ]]; then
    # shellcheck disable=SC2086  # the pinned package list
    uv pip install -q --python "$PY" --extra-index-url "$OVRTX_INDEX" $OVRTX_PACKAGES pyglet
  fi
  if [[ "$WITH" == *" fillcuda "* ]]; then
    # shellcheck disable=SC2086  # the pinned package list
    uv pip install -q --python "$PY" $FILL_CUDA_PACKAGES
  fi
}

env_mjlab() {
  local h=$WORK/hyperlux
  (cd "$MJLAB_PINS" && sha256sum -c --quiet pins.sha256)
  mkdir -p "$h/mjlab" "$WORK/optix-dev"
  git clone -q "$MJLAB_PINS/mjlab_hyperlux_plugin.bundle" "$h/plugin" && git -C "$h/plugin" checkout -q "$PLUGIN_COMMIT"
  tar xzf "$MJLAB_PINS/mjlab.tgz" -C "$h/mjlab"
  git -C "$h/mjlab" apply "$h/plugin/patches/mjlab-rgb-backends.patch"
  tar xzf "$OPTIX_HEADERS" -C "$WORK/optix-dev"
  uv sync -q --frozen --python "$PYTHON" --project "$h/mjlab" --extra cu130
  PY=$h/mjlab/.venv/bin/python
  uv pip install -q --python "$PY" -e "$h/plugin[examples]"
  uv pip install -q --python "$PY" --no-deps --no-cache -e "$SRC"
  uv pip install -q --python "$PY" --no-deps "$USD_CORE"
  export PATH="$h/mjlab/.venv/bin:$PATH" CMAKE_GENERATOR='Unix Makefiles' HYPERLUX_BACKEND=OPTIX
  # HyperLux's OptiX modules (and raywarp's, which honours the variable) in the hyperlux family;
  # OPTIX_INCLUDE_DIR is HyperLux's (raywarp reads its bundled headers).
  export OPTIX_CACHE_PATH=$HYPERLUX_CACHE_DIR/optix
  export OPTIX_INCLUDE_DIR=$WORK/optix-dev/include MJPBR_HYPERLUX_SOURCES=$MJLAB_PINS/pins.txt MJPBR_MJLAB_COMMIT=$MJLAB_COMMIT
  "$PY" -m mjlab_hyperlux.patch_hyperlux > "$WORK/patch_hyperlux.log" 2>&1 || { tail -20 "$WORK/patch_hyperlux.log"; return 1; }
  local version; version=$("$PY" -c 'import importlib.metadata as m; print(m.version("hyperlux"))')
  [[ "$version" == "$HYPERLUX_VERSION" ]] || { echo "FATAL: hyperlux $version, pinned $HYPERLUX_VERSION"; return 1; }
}

bootstrap() {  # the environment and its caches; sets PY
  stage bootstrap
  detect_gpu
  system_packages
  system_setup
  export_common
  if [[ "$DRY" == 1 ]]; then
    PY=$(command -v python3)
    restore "dry" warp raywarp
    return 0
  fi
  [[ "$ENV_NAME" == project || "$ENV_NAME" == mjlab ]] || { echo "unknown HF_ENV $ENV_NAME"; return 2; }
  stage restore-uv
  restore "$ENV_NAME-uv" uv
  stage environment
  "env_$ENV_NAME"
  PATH="$(dirname "$PY"):$PATH"
  export PATH
  stage restore-caches
  local warp key families=(warp)
  warp=$("$PY" -c 'import importlib.metadata as m; print(m.version("warp-lang"))')
  key="$ENV_NAME-$ARCH-warp$warp"
  # raywarp on GPUs: what a warm start reads (CPU jobs run suites, whose files are throwaway).
  [[ "$ARCH" == cpu ]] || families+=(cuda raywarp)
  [[ "$ENV_NAME" != mjlab ]] || families+=(hyperlux)
  [[ "$WITH" != *" ovrtx "* ]] || families+=(ovrtx)
  restore "$key" "${families[@]}"
}

write_env() {  # FILE: the exports a later shell sources (sandbox)
  {
    for v in PATH PY UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_NO_SYNC UV_PROJECT_ENVIRONMENT WARP_CACHE_PATH WARP_KERNEL_CACHE_DIR CUDA_CACHE_PATH \
      RAYWARP_CACHE_DIR OPTIX_CACHE_PATH HYPERLUX_CACHE_DIR MJPBR_ASSET_DIR MJPBR_HOST_CACHE MUJOCO_GL OPTIX_INCLUDE_DIR HYPERLUX_BACKEND RAYWARP_NATIVE_ARCHIVES RAYWARP_NATIVE_CACHE \
      CMAKE_GENERATOR MJPBR_HYPERLUX_SOURCES MJPBR_MJLAB_COMMIT; do
      [[ -z "${!v:-}" ]] || printf 'export %s=%q\n' "$v" "${!v}"
    done
    printf 'export HF_PY=%q HF_SRC=%q HF_OUT=%q HF_ARCH=%q HF_RT=%q\n' "$PY" "$SRC" "$OUT" "$ARCH" "$RT"
  } > "$1"
}

# --- job: run the command, mirror, watchdog -------------------------------------------------------
mirror() {  # small out/ files + the log + stages, copied to the run directory (never deletes)
  local run=$DATA/runs/$HF_RUN
  mkdir -p "$run/out"
  (cd "$OUT" && find . -type f -size -$((MIRROR_MAX / 1024))k -newer "$WORK/.mirrored" -print0 2>/dev/null \
    | xargs -0 -r cp --parents -t "$run/out/") || true
  touch -d "@$(( $1 - 1 ))" "$WORK/.mirrored"
  tail -c $((8 << 20)) "$WORK/run.log" > "$run/run.log" 2>/dev/null || true
  cp "$WORK/stages.txt" "$run/stages.txt" 2>/dev/null || true
}

run_command() {
  stage run
  local cap=$(( ${HF_TIMEOUT_S:-1500} - ($(date +%s) - T_START) - FINISH_RESERVE ))
  (( cap > ${HF_MIN_CAP_S:-30} )) || { echo "no time left for the command (cap ${cap}s): raise --timeout"; REASON=no-time; return 1; }
  # HF_DEADLINE: the epoch second the command is stopped at (a command keeps time for its own last step).
  export HF_RUN HF_OUT=$OUT HF_SRC=$SRC HF_PY=$PY HF_SOURCES=$WORK/sources HF_ARCH=$ARCH HF_RT=$RT HF_DEADLINE=$(( $(date +%s) + cap ))
  echo "command (cap ${cap}s, stall ${HF_STALL_S:-600}s): $HF_CMD"
  cd "$SRC"
  : > "$WORK/run.log"
  timeout --kill-after=30s "$cap" bash -c "$HF_CMD" > "$WORK/run.log" 2>&1 &
  CMD_PID=$!
  local pid=$CMD_PID last_mirror=0 now rc
  tail -n +1 -f --pid="$pid" "$WORK/run.log" &
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1  # the command's end is seen within a second
    now=$(date +%s)
    if (( now - $(stat -c %Y "$WORK/run.log") > ${HF_STALL_S:-600} )); then
      echo "STALL: no output for ${HF_STALL_S:-600} s: stopping the command"
      REASON=stalled
      kill -TERM "$pid" 2>/dev/null || true
      break
    fi
    if (( now - last_mirror >= 60 )); then mirror "$now"; last_mirror=$now; fi
  done
  set +e; wait "$pid"; rc=$?; set -e
  if [[ "$REASON" == ok ]]; then
    case $rc in 0) ;; 124|137) REASON=timeout ;; *) REASON=failed ;; esac
  fi
  echo "command exit $rc ($REASON)"
  return "$rc"
}

json_map() {  # NAME of an associative array -> a JSON object of its string values
  local -n map=$1; local k v sep=""
  printf '{'
  for k in "${!map[@]}"; do
    v=${map[$k]//\\/\/}; v=${v//\"/\'}; v=${v//[$'\t\n\r']/ }  # no JSON escapes needed afterwards
    printf '%s"%s": "%s"' "$sep" "$k" "$v"; sep=", "
  done
  printf '}'
}

finish() {
  local rc=$? run=$DATA/runs/$HF_RUN size i
  trap - EXIT; trap '' TERM INT; set +eu
  # A terminated job: stop the command (timeout forwards to its process group) before reading out/.
  if [[ -n "$CMD_PID" ]] && kill -0 "$CMD_PID" 2>/dev/null; then
    kill -TERM "$CMD_PID" 2>/dev/null
    for ((i = 0; i < 30; i++)); do kill -0 "$CMD_PID" 2>/dev/null || break; sleep 1; done
  fi
  [[ "$REASON" != ok || "$rc" == 0 ]] || REASON=failed
  stage mirror
  mirror "$(date +%s)"
  stage evidence
  size=$(du -sb "$OUT" | cut -f1)
  if (( size <= EVIDENCE_CAP )); then
    tar czf "$WORK/evidence.tgz" -C "$OUT" . && timeout 90 cp "$WORK/evidence.tgz" "$run/evidence.tgz.$$.partial" \
      && mv "$run/evidence.tgz.$$.partial" "$run/evidence.tgz"
  else
    echo "out/ is $((size >> 20)) MB (> $((EVIDENCE_CAP >> 20)) MB): no evidence.tgz, mirrored small files only"
  fi
  # The command's outcome is final before the cache save (a job lost while saving keeps it).
  SAVED[all]=pending
  write_status "$rc" "$run" >/dev/null
  stage cache-save
  unset 'SAVED[all]'
  save_caches_capped
  stage done
  write_status "$rc" "$run"
  exit "$rc"
}

write_status() {  # RC RUN_DIR: status.json (written last: `tools.hf status`), to the run directory
  {
    printf '{"run": "%s", "exit": %s, "reason": "%s", "commit": "%s", "env": "%s", "with": "%s",' \
      "$HF_RUN" "$1" "$REASON" "$(cat "$SRC/COMMIT" 2>/dev/null)" "$ENV_NAME" "${HF_WITH:-}"
    printf ' "gpu": "%s", "compute_capability": "%s", "rt_cores": %s, "wall_s": %s,' \
      "$GPU_NAME" "$GPU_CC" "$RT" "$(( $(date +%s) - T_START ))"
    printf ' "stages_s": %s, "restored": %s, "saved": %s, "finished": "%s"}\n' \
      "$(json_map DUR | sed 's/"\([0-9]*\)"/\1/g')" "$(json_map RESTORED)" "$(json_map SAVED)" "$(date -u +%FT%TZ)"
  } > "$WORK/status.json"
  cat "$WORK/status.json"
  cp "$WORK/status.json" "$2/status.json.$$.partial" && mv "$2/status.json.$$.partial" "$2/status.json"
}

save_caches_capped() {  # save_caches under SAVE_CAP seconds (a background subshell, then the wait)
  ( save_caches ) & local pid=$! t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    if (( $(date +%s) - t0 > SAVE_CAP )); then kill -TERM "$pid"; SAVED[all]="timeout after ${SAVE_CAP}s"; break; fi
    sleep 1
  done
  wait "$pid" 2>/dev/null
  local f family
  for f in "$WORK"/stamps/*.key; do
    [[ -e "$f" ]] || continue
    family=$(basename "$f" .key)
    SAVED[$family]=$(cat "$WORK/stamps/$family.saved" 2>/dev/null || echo "not saved")
  done
}

case $mode in
  job)
    HF_RUN=${2:?run id}
    touch -d "@$T_START" "$WORK/.mirrored"
    trap finish EXIT
    trap 'REASON=terminated; exit 143' TERM INT
    REASON=bootstrap-failed
    HF_CMD=$(printf %s "${HF_CMD_B64:?HF_CMD_B64}" | base64 -d)
    # The run's extra source trees (`run --source NAME=REF`): $HF_SOURCES/<name>/ (a committed tree) + COMMIT.
    for archive in "$DATA/runs/$HF_RUN"/sources/*.tgz; do
      [[ -e "$archive" ]] || continue
      name=$(basename "$archive" .tgz); mkdir -p "$WORK/sources/$name"
      tar xzf "$archive" -C "$WORK/sources/$name"
      sed -n "s/^$name \([0-9a-f]*\)$/\1/p" "$DATA/runs/$HF_RUN/sources.txt" > "$WORK/sources/$name/COMMIT"
    done
    bootstrap
    # Every source tree reads the native stack the hook bundled into the job's tree (OptiX headers,
    # MDL SDK: raywarp/_native, pinned by tools/native/hook.py).
    for tree in "$WORK"/sources/*/; do
      [[ -d "$tree/raywarp" ]] && cp -r "$SRC/raywarp/_native/." "$tree/raywarp/_native/"
    done
    REASON=ok
    run_command
    ;;
  bootstrap)
    bootstrap
    write_env "$WORK/env.sh"
    echo "bootstrap done in $(( $(date +%s) - T_START )) s: source $WORK/env.sh"
    ;;
  save-caches)
    # shellcheck disable=SC1091
    . "$WORK/env.sh"; PY=$HF_PY
    save_caches
    ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
