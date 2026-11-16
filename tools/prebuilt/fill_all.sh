#!/usr/bin/env bash
# The prebuilt kernels' refill on HF jobs for every GPU architecture, one command, resumable
# (tools/prebuilt/fill.py, build.py, tools/release/kernels.py):
#
#   tools/prebuilt/fill_all.sh refill TAG              # everything below in order; rerun to resume
#   PUBLISH=1 tools/prebuilt/fill_all.sh refill TAG    # ... and the upload to the kernels bucket + the pins
#   PREFIX=scratch/x PUBLISH=1 ... refill TAG          # the upload under a scratch place (a dry run)
#   DRY=1 PUBLISH=1 tools/prebuilt/fill_all.sh refill TAG   # the CPU-only dry run (below)
#   tools/prebuilt/fill_all.sh status TAG              # each job: state, run id
#   tools/prebuilt/fill_all.sh watch TAG [JOB...]      # follow / relaunch jobs as `refill` does
#   tools/prebuilt/fill_all.sh unpublish TAG           # delete a scratch upload (PREFIX only)
#   tools/prebuilt/fill_all.sh spec TAG JOB            # one job's flavor, cap and command
#   tools/prebuilt/fill_all.sh keys TAG                # the architectures to build (keys vs the pins), seconds
#   tools/prebuilt/fill_all.sh plan TAG                # ... and their worst case (reads the cache)
#   tools/prebuilt/fill_all.sh cost TAG                # the worst case per architecture (count x cap x $/h)
#
# Nothing compiled twice (tools/prebuilt/buildkey.py, build_cache.py):
# - SKIP: each architecture's bundle has a key (buildkey.py: the git tree ids of the code that
#   reaches it, the locked Warp / MuJoCo Warp releases, the job image's CUDA, the OptiX headers'
#   commit, the fill options, the pack's driver and groups, the census's key), recorded in the pins
#   (format 4 `key`). `refill` first computes every key (seconds) and builds only the architectures
#   whose key the pins (PINS) do not hold; the others keep their pins entry. None changed: nothing
#   launched, $0. FORCE=1: every architecture.
# - REUSE: every job mounts the build cache bucket (CACHE_BUCKET, tools/prebuilt/buckets.env) at
#   CACHE (/cache), content-addressed and immutable: a listing part is taken from it when its key
#   (the code, the profile, the options) is there (the job is not launched); the census probes only
#   the scenes whose rows of its key are missing (no census job when none: no cpu-performance);
#   a PTX / Warp shard takes every PTX key (of unchanged reached code) and Warp module the last
#   published bundle (BUNDLE_SEED) or the cache holds and compiles only the rest, keeping each in
#   the cache as it is made; a pack takes the modules the cache (`--keep`) or the published pack
#   (SEED) holds, and the merge counts the modules left (`build.py plan-pack`): none, no GPU job
#   (the release job assembles the pack on its CPU), else ceil(left / PACK_MODULES_<A>) jobs.
# - The release lists what it used in the cache (refs/TAG.json); the workflow prunes the cache to
#   the last releases' units (`build_cache.py prune`).
#
# Architectures (ARCHS, default "120 89 86 90 80"): one bundle each.
# The PTX and Warp-module shards run on CPU jobs for every architecture (NVRTC targets any SM; the
# OptiX PTX target is min(arch, 89), `trace/optix/context.ptx_arch`, so sm_89 and sm_120 compile the
# same PTX keys when their routes agree); the OptiX module packs run on a GPU of the architecture
# (a module pack is per architecture and NVIDIA driver): sm_120 rtx-pro-6000, sm_89 l4x1 (L4_FLAVOR:
# or l40sx1), sm_86 a10g-large. sm_80 (A100) and sm_90 (H100) have no RT cores: the software path's
# bundle (Warp modules only, `fill --backend software`), no census, no pack.
#
# Wave 1, launched at once (each job waits in place for its inputs on the bucket, so its bootstrap
# overlaps the jobs before it), per architecture A (job names sm<A>-...):
#   sm<A>-list-<i>  OPTIX_PARTS cpu-upgrade jobs: `fill list --profile P --part i/N` (every fill point
#                   listed once) -> CACHE/sm<A>/list/<list key>/part-<i>.json (not launched when there)
#   sm<A>-p<i>      PTX shards (cpu-upgrade, OptiX architectures): wait for the listing, `fill spread`
#                   locally (the same shard files in every job), `fill fill --shard --seed --cache`
#                   (what the published bundle or the cache holds is taken, the rest compiled and kept)
#                   -> runs/<day>-<tag>-sm<A>-final-p<i>.tar
#   sm<A>-w<i>      Warp module shards (cpu-upgrade), the same way -> runs/<day>-<tag>-sm<A>-final-w<i>.tar
#   sm<A>-merge     (cpu-upgrade) waits for its shards and (OptiX) the census: `build.py merge
#                   --census` -> runs/<day>-<tag>-sm<A>-merged.tar (the bundle), -merged-ptx.tar
#                   (its PTX: the packs'), -census.tar (its census files), -pack-todo (with a
#                   DRIVER_<A>: the modules neither the cache nor the seed pack holds)
# and once for every OptiX architecture:
#   census-<i>      up to CENSUS_JOBS cpu-upgrade jobs + census-heavy (cpu-performance: the scenes of
#                   HEAVY, large prepared stages: cpu-upgrade lost the /data mount on them), over the
#                   scenes the cache lacks only: `word_reach.py --units --profiles` over the registry,
#                   each scene probed once for every OptiX architecture's profile
#                   -> CACHE/sm<A>/census/<census key>/<scene>.json
# Wave 2, after the merges:
#   sm<A>-pack-<i>  PACKS_<A> jobs (planned from the merge's -pack-todo: none when the cache and the
#                   seed hold every module) on the architecture's GPU flavor: the OptiX module pack of the
#                   census modules in PACK_GROUPS_<A>'s order until a deadline, on that flavor's
#                   driver (DRIVER_<A>: checked in the job when set; empty: the job's, recorded in the
#                   pack's manifest, `merge-packs` refuses packs of two drivers), one file per module
#                   (each kept in CACHE/sm<A>/optix/<driver>/ as made: resumable, and reused by every
#                   later fill; those the published pack of the seed holds fetched from it, not made)
#                   -> runs/<day>-<tag>-sm<A>-pack-<i>.tar
#   sm<A>-release   (cpu-upgrade) waits for its packs: `build.py merge-packs` (no pack job planned:
#                   `build.py pack --assemble DRIVER` from the cache and the seed, no GPU),
#                   `tools/release/kernels.py layout` (bundle + pack), `build_cache.py refs`
#                   -> runs/<day>-<tag>-sm<A>-release.tar, -refs.json
# Then, here: with PUBLISH=1 every built architecture's release archive downloaded into one layout,
# `kernels.py pin --keys` (one pins file for every architecture: the built ones' new entries, the
# skipped ones' kept) and `kernels.py publish` (skips a unit published with the same bytes, refuses
# one with others); PINS written (raywarp/core/build/kernel_cache/kernels.json: commit it; with
# PREFIX $STATE.kernels.json); refs/TAG.json uploaded to the cache (not with PREFIX).
#
# DRY=1: the orchestration proved on CPU, tiny and cheap (~12 cpu-upgrade jobs of a few minutes,
# < $0.10): ARCHS "120 80" (an OptiX and a software architecture) unless set, one listing part, one
# PTX and one Warp shard per architecture of the ONLY points (default: one scene-free point), one
# census job over DRY_SCENES, no pack (PACKS 0), every job on cpu-upgrade, PREFIX scratch/dry-TAG,
# no bundle seed (BUNDLE_SEED ""). Its keys name ONLY and the scene list: they never match a real
# release's.
#
# Never `cp -r` or reads many files in place from /data: every hand-over is one archive, copied to
# local disk and unpacked (one sequential read). No job restores a compiler cache family
# (`--no-cache-restore warp,cuda,raywarp`: the fill compiles into its own root, a pack into a fresh
# OptiX cache; what is reused comes from the build cache, by content). Every job prints a HEARTBEAT
# each minute (the stall watchdog), shards snapshot their
# output to the bucket every SNAP_EVERY s (one tar) and resume from it; `refill` relaunches a
# failed / stalled / vanished job up to RETRIES times. State: $STATE (.fill_all/TAG: the day and each
# job's current run id; `.done`: jobs ended ok; `.stage`: waves passed). Run from the committed tree
# and keep that tree for the whole fill (the PTX keys and the shard files name its code). Bash only.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# Bash reads a script as it runs: a long `refill` runs from a private copy (a checkout or edit of
# this file meanwhile cannot change it).
# FILL_ALL_TOOLS: this tree's tools/prebuilt (the keys, the bucket names), wherever it runs from.
if [ -z "${FILL_ALL_COPY:-}" ]; then
  FILL_ALL_TOOLS=$(cd "$(dirname "$0")" && pwd)
  copy=$(mktemp --suffix=.fill_all.sh) && cp "$0" "$copy" && FILL_ALL_TOOLS=$FILL_ALL_TOOLS FILL_ALL_COPY=$copy exec bash "$copy" "$@"
fi
set -euo pipefail
cmd=$1 tag=$2
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then
  ARCHS=${ARCHS:-120 80} OPTIX_PARTS=${OPTIX_PARTS:-1} PTX=${PTX:-1} WARP=${WARP:-1} WARP_SOFTWARE=${WARP_SOFTWARE:-4} CENSUS_JOBS=${CENSUS_JOBS:-1}
  PACKS=${PACKS:-0} HEAVY=${HEAVY-} PREFIX=${PREFIX:-scratch/dry-$tag} SEED=${SEED-} BUNDLE_SEED=${BUNDLE_SEED-}
  ONLY=${ONLY:-scene=fs_point0_plain_none_mesh_n,tier=fast,layout=shared,gi=False,outputs=shaded}
  SHARD_CAP=${SHARD_CAP:-30m} MERGE_CAP=${MERGE_CAP:-30m} RELEASE_CAP=${RELEASE_CAP:-30m} RELEASE_FLAVOR=cpu-upgrade
  if [ -z "${SCENES_FILE:-}" ]; then
    SCENES_FILE=$(mktemp --suffix=.scenes) && echo "${DRY_SCENES:-fs_point1_plain_none_mesh_n}" > "$SCENES_FILE"
  fi
fi
# Sizing (from a probe run and the refill plan; the cost model: .github/workflows/kernels.yml), per
# architecture.
ARCHS=${ARCHS:-120 89 86 90 80}
PTX=${PTX:-40} WARP=${WARP:-16} OPTIX_PARTS=${OPTIX_PARTS:-12} CENSUS_JOBS=${CENSUS_JOBS:-8} PACKS=${PACKS:-}
# Warp shards of a software-path architecture (sm_80 / sm_90): its builds are Warp word modules (each
# a megakernel per word and literal twin, lens model: one point lists 1 546 modules / ~500 MB of source
# against 118 / 1.7 MB on OptiX, measured 2026-11-16 on the dry point), so 3x the OptiX count; the
# dry run's point spreads over 4 (~125 MB each; one shard of all 1 546 was the failed dry run's).
WARP_SOFTWARE=${WARP_SOFTWARE:-48}
WORKERS=${WORKERS:-4}  # NVRTC processes per PTX shard (cpu-upgrade: 4 at once is the best rate)
SHARD_FLAVOR=${SHARD_FLAVOR:-cpu-upgrade}  # cpu-performance: ~2x faster per build, ~60x the price
RETRIES=${RETRIES:-2} WATCH_EVERY=${WATCH_EVERY:-120} SNAP_EVERY=${SNAP_EVERY:-600} POLL=${POLL:-30}
# A job that failed RETRIES relaunches: FAIL_FAST=1 (default under CI, e.g. the workflow) kills every
# other running job of the wave and fails the fill at once (the jobs waiting for its output would
# otherwise wait to their caps, hours); 0 (interactive default) leaves them waiting in place for the
# operator's fix of that job (`watch TAG JOB` after a hand relaunch, or rerun `refill TAG`).
if [ -n "${CI:-}" ]; then FAIL_FAST=${FAIL_FAST:-1}; else FAIL_FAST=${FAIL_FAST:-0}; fi
# Caps per job kind (worst case = the count x the cap x the flavor's $/h: `cost`).
LIST_CAP=${LIST_CAP:-40m} CENSUS_CAP=${CENSUS_CAP:-45m} SHARD_CAP=${SHARD_CAP:-75m} MERGE_CAP=${MERGE_CAP:-120m}
# The release (xz of a ~2.4 GB bundle on 8 processes, + the pack) on cpu-upgrade: ~10-30 min,
# $0.06 at its cap (cpu-performance was $3.80 per architecture).
PACK_CAP=${PACK_CAP:-35m} RELEASE_CAP=${RELEASE_CAP:-120m} RELEASE_FLAVOR=${RELEASE_FLAVOR:-cpu-upgrade}
# The pack (user-approved for sm_120, 2026-11-11: 9 jobs x 35 min, worst $14.4): the census modules
# group by group (`build.py pack --groups`), each group most used first: the quality recipe's own
# words first in time (7-15 min each on the RTX PRO 6000, BUG-583: started at once, spread over the
# jobs, so they end before the deadline), then the fast and default recipes' own words, until
# PACK_MARGIN s before the job's end (`--deadline`: no module starts after it, nor one of a build
# whose modules took longer than the time left; what is done is the pack; the cap is the cost).
# The L4 / A10G (8 / 12 vCPU: OptiX module creation is host-bound, ~4x / ~3x slower than the RTX
# PRO 6000's 32 workers; unmeasured) pack the fast and default recipes' own words only.
PACK_MARGIN=${PACK_MARGIN:-600}
# The modules one pack job makes before its deadline (planned job count: ceil(left / this), at
# most PACKS_<A>): w15rel2 measured 413 per RTX PRO 6000 job; the L4 / A10G scaled by their vCPUs.
PACK_MODULES_120=${PACK_MODULES_120:-400} PACK_MODULES_89=${PACK_MODULES_89:-100} PACK_MODULES_86=${PACK_MODULES_86:-130}
L4_FLAVOR=${L4_FLAVOR:-l4x1}  # sm_89: l4x1 ($0.80/h) or l40sx1 ($1.80/h, more cores)
# The release driver of each architecture's module pack ("" : the pack job's own, recorded).
DRIVER_120=${DRIVER_120-580.178.04} DRIVER_89=${DRIVER_89-} DRIVER_86=${DRIVER_86-}
# The seed of the packs (`build.py pack --seed`): a published pack of the same architecture and
# driver whose modules the packs fetch instead of making them (a module's name is the digest of its
# PTX and record: unchanged ones keep their names; the pack stays the census's set). SEED=auto:
# for each OptiX architecture with a DRIVER_<A>, the bundle raywarp/core/build/kernel_cache/
# kernels.json pins for it, if its pins list a pack of that driver (format 1 pins list no packs:
# assumed), and the tree its refill ran from (the latest commit of kernels.json naming that bundle:
# "... bundle CONTENT ... (refill TAG from REV)"); SEED_<A>=CONTENT@REV names one; SEED="": none. The name does not cover the code that
# turns a record into OptiX's compile options: a refill whose SEED_GUARD files differ from REV's
# refuses the seed (a module of other options would be a cache miss at first use); read that diff,
# then narrow SEED_GUARD or set SEED="".
SEED=${SEED-auto}
# The bundle the PTX / Warp shards take unchanged units from (`fill --seed`): auto, the bundle PINS
# pins for the architecture (if its pins name this BUCKET); a URL; "": none (the cache alone).
BUNDLE_SEED=${BUNDLE_SEED-auto}
SEED_GUARD=${SEED_GUARD-raywarp/core/trace/optix/pipeline.py raywarp/core/trace/optix/device/intrinsics.py raywarp/core/trace/optix/context.py raywarp/core/trace/optix/api.py raywarp/core/trace/optix/gas.py raywarp/core/trace/optix/modules.py}
# Registry scenes with large prepared stages: their census runs on cpu-performance.
HEAVY=${HEAVY-bistro hospital full_warehouse office_usd a_beautiful_game assembly_studio infinigen_dining isaac_warehouse intel_sponza_full}
# The census scenes (whitespace separated): the benchmark registry's and feature-space scenes
# (`tools/prebuilt/word_reach.py`). A public checkout (no development tree: no registry) takes the
# feature-space ones (fs_*) only.
SCENES_FILE=${SCENES_FILE:-tools/prebuilt/census_scenes.txt}
ONLY=${ONLY:-}  # a dry run: `fill list --only` KEY=VALUE pairs (comma separated): only those points listed
# The fill's exclusions: passed to both `fill list` (the points listed, so no shard is given them)
# and every `fill fill` shard.
FILL_ARGS=${FILL_ARGS:-}
STATE=${STATE:-.fill_all/$tag}
PY=${PY:-python}  # the local interpreter of tools.hf and tools.release (the jobs run their own `python`)
HF_CLI=${HF_CLI:-hf}  # the huggingface_hub CLI (the bucket copies of `publish`, the cache's listing)
DATA=${DATA:-/data}  # the jobs' mount of the project bucket
CACHE=${CACHE:-/cache}  # the jobs' mount of the build cache bucket
LANE=${LANE:-release}
# The buckets (tools/prebuilt/buckets.env, the one place their names are written; each overridable).
eval "$(sed -n 's/^\([A-Z_]*\)=\(.*\)$/\1=${\1:-\2}/p' "$FILL_ALL_TOOLS/buckets.env")"
BUCKET=${BUCKET:-$KERNELS_BUCKET} PREFIX=${PREFIX:-} PUBLISH=${PUBLISH:-0} FORCE=${FORCE:-0}
# The pins compared against (SKIP) and written (publish): the package's, or a scratch run's own.
if [ -n "$PREFIX" ]; then PINS=${PINS:-$STATE.kernels.json}; else PINS=${PINS:-raywarp/core/build/kernel_cache/kernels.json}; fi
hf() { $PY -m tools.hf "$@"; }

# --- architectures -----------------------------------------------------------------------------

backend() {  # A: optix (RT cores) or software (sm_80 A100, sm_90 H100)
  case $1 in 80 | 90) echo software ;; 120 | 89 | 86) echo optix ;; *) echo "no fill plan for sm_$1" >&2; return 1 ;; esac
}

profile() {  # A: the census / fill GPU profile (tools/prebuilt/probe.PROFILES)
  case $1 in 120) echo rtx_pro_6000 ;; 89) echo l4 ;; 86) echo a10g ;; 90) echo h100 ;; 80) echo a100 ;; esac
}

pack_flavor() {  # A: the HF flavor of its module pack (an OptiX architecture's GPU)
  case $1 in 120) echo rtx-pro-6000 ;; 89) echo "$L4_FLAVOR" ;; 86) echo a10g-large ;; esac
}

per_arch() {  # VAR A DEFAULT: $VAR_A, else DEFAULT
  local v="${1}_$2"
  echo "${!v-$3}"
}

packs() {  # A: its pack jobs (0 on the software path; planned at wave 2: `plan_packs`)
  local d=0 planned
  if [ "$(backend "$1")" = optix ]; then
    case $1 in 120) d=9 ;; 89) d=6 ;; 86) d=4 ;; esac
    d=${PACKS:-$d}
  fi
  planned=$(sed -n "s/^$1 //p" "$STATE.packs" 2>/dev/null)
  if [ -n "$planned" ]; then echo "$planned"; else per_arch PACKS "$1" "$d"; fi
}

assembled() {  # A: its pack is assembled by the release job (planned: no module left to make)
  [ "$(sed -n "s/^$1 //p" "$STATE.packs" 2>/dev/null)" = 0 ] && [ -n "$(per_arch DRIVER "$1" "")" ]
}

pack_workers() {  # A: module creation processes of a pack job (the flavor's vCPUs)
  local d=32
  case $1 in 120) d=32 ;; 89) d=8 ;; 86) d=12 ;; esac
  per_arch PACK_WORKERS "$1" "$d"
}

pack_groups() {  # A: the census groups its packs make, in order
  local d="fast+default:own"
  [ "$1" != 120 ] || d="quality:own,fast+default:own"
  per_arch PACK_GROUPS "$1" "$d"
}

ptx_shards() {  # A: PTX shards (0 on the software path: Warp modules only)
  if [ "$(backend "$1")" = optix ]; then per_arch PTX "$1" "$PTX"; else echo 0; fi
}

warp_shards() {  # A: its Warp shards (WARP_<A>, else WARP, or WARP_SOFTWARE on the software path)
  local d=$WARP
  [ "$(backend "$1")" = optix ] || d=$WARP_SOFTWARE
  per_arch WARP "$1" "$d"
}

optix_archs() {  # the OptiX architectures of ARCHS
  local a
  for a in $ARCHS; do [ "$(backend "$a")" != optix ] || echo "$a"; done
}

# --- job commands ------------------------------------------------------------------------------
# Every job: a heartbeat each minute, stopped at the end.
BEAT='(while sleep 60; do echo "HEARTBEAT $(date +%T) $(free -g | awk "/Mem/ {print \$3 \"/\" \$2 \" GB\"}")"; done) & BEAT=$!; trap "kill $BEAT" EXIT; set -o pipefail;'

day() { sed -n 's/^day //p' "$STATE"; }
base() { echo "$DATA/runs/$(day)-$tag"; }

wait_for() {  # GLOB COUNT: a job-side loop until COUNT files match GLOB on the bucket mount
  echo "n=0; until [ \$(ls $1 2>/dev/null | wc -l) -ge $2 ]; do (( n++ % 10 )) || echo \"WAIT \$(ls $1 2>/dev/null | wc -l)/$2 $1\"; sleep $POLL; done;"
}

keep() {  # SRC_DIR DEST_TAR [MEMBERS]: SRC_DIR (or its MEMBERS) as one archive on the bucket (partial, then renamed)
  echo "{ tar -cf $2.partial -C $1 ${3:-.} && mv $2.partial $2 && echo \"kept $2 \$(du -h $2 | cut -f1)\"; }"
}

only() {  # the dry run's --only flags
  [ -z "$ONLY" ] || echo "$ONLY" | tr ',' '\n' | sed 's/^/--only /' | tr '\n' ' '
}

seed_url() {  # A: the seed pack's URL of architecture A ("" without one), its options code checked against REV's
  local seed content rev changed driver
  driver=$(per_arch DRIVER "$1" "")
  seed=$(per_arch SEED "$1" "$SEED")
  [ -n "$seed" ] && [ -n "$driver" ] || return 0
  if [ "$seed" = auto ]; then
    content=$($PY -c "
import json, sys
b = json.load(open('raywarp/core/build/kernel_cache/kernels.json'))['bundles'].get('$1')
print(b['content'] if b and ('packs' not in b or '$driver' in b['packs']) else '')")
    [ -n "$content" ] || return 0  # no published pack of this architecture and driver
    # The latest pins commit naming this bundle (a release that skipped the architecture kept it).
    seed=$(git log --format=%s -- raywarp/core/build/kernel_cache/kernels.json | sed -n "s/.*bundle \($content\)[ ,].*from \([0-9a-f]*\))\$/\1@\2/p" | head -1)
    [ -n "$seed" ] || { echo "SEED=auto: no commit of raywarp/core/build/kernel_cache/kernels.json names a refill of bundle $content: set SEED_$1=CONTENT@REV" >&2; return 1; }
  fi
  content=${seed%@*} rev=${seed#*@}
  changed=""
  [ -z "$SEED_GUARD" ] || changed=$(git diff --name-only "$rev" HEAD -- $SEED_GUARD) || return 1
  [ -z "$changed" ] || { echo "SEED $seed (sm$1): the options code changed since $rev ($changed): read the diff, then narrow SEED_GUARD or set SEED=\"\"" >&2; return 1; }
  echo "https://huggingface.co/buckets/$BUCKET/resolve/sm$1/$content/optix/$driver"
}

all_scenes() {  # every census scene, one per line (a public checkout: the fs_* ones)
  local all; all=$(tr -s ' \n' '\n' < "$SCENES_FILE" | grep -v '^$')
  [ -d internal/benchmarks ] || all=$(grep '^fs_' <<<"$all" || true)
  echo "$all"
}

census_scenes() {  # I: the scenes of census job I (census-heavy: the HEAVY ones), of those the cache lacks
  local all; all=$(grep -v '^$' "$STATE.census-todo" || true)
  if [ "$1" = heavy ]; then
    for s in $HEAVY; do if grep -qx "$s" <<<"$all"; then echo "$s"; fi; done | paste -sd,
  else
    { grep -vxF -f <(tr ' ' '\n' <<<"$HEAVY") <<<"$all" || true; } | awk -v i="$1" -v n="$CENSUS_JOBS" '(NR - 1) % n == i' | paste -sd,
  fi
}

job_spec() {  # NAME: "flavor cap" and, on the next line, the job's command
  local name=$1
  if [[ $name == census-* ]]; then
    local i=${name#census-} flavor=cpu-upgrade scenes profiles
    [ "$i" != heavy ] || flavor=cpu-performance
    scenes=$(census_scenes "$i")
    profiles=$(for a in $(optix_archs); do profile "$a"; done | paste -sd,)
    local ck dests="" have="true" a
    ck=$(state census-key)
    for a in $(optix_archs); do
      dests="$dests && mkdir -p $CACHE/sm$a/census/$ck && cp /tmp/near/sm$a/\$s.json $CACHE/sm$a/census/$ck/.\$s.json.partial && mv $CACHE/sm$a/census/$ck/.\$s.json.partial $CACHE/sm$a/census/$ck/\$s.json"
      have="$have && [ -f $CACHE/sm$a/census/$ck/\$s.json ]"
    done
    echo "$flavor $CENSUS_CAP"
    # Each scene probed unless the cache holds its rows for every architecture, then kept there.
    echo "$BEAT mkdir -p /tmp/near && for s in ${scenes//,/ }; do if $have; then echo \"REACH \$s cached\"; continue; fi; python -m tools.prebuilt.word_reach --scenes \$s --profiles $profiles --out /tmp/reach/\$s.sm{arch}.json --units /tmp/near/sm{arch}/\$s.json 2>&1 | grep --line-buffered -E '^REACH|Error|Traceback' || exit 1; true $dests || exit 1; done"
    return
  fi
  local arch=${name%%-*} job=${name#*-}
  arch=${arch#sm}
  local b; b="$(base)-sm$arch"
  local lk; lk=$(state "list-key-$arch")
  local be pr np nw lists="/tmp/lists" listed="$CACHE/sm$arch/list/$lk" parts="$CACHE/sm$arch/list/$lk/part-*.json"
  be=$(backend "$arch") pr=$(profile "$arch") np=$(ptx_shards "$arch") nw=$(warp_shards "$arch")
  local spread="mkdir -p $lists && cp $parts $lists/ && python -m tools.prebuilt.fill spread --ptx-shards $np --warp-shards $nw --out $lists | grep FILL &&"
  case $job in
    list-*)
      local i=${job#list-}
      echo "cpu-upgrade $LIST_CAP"
      echo "$BEAT if [ -f $listed/part-$i.json ]; then echo 'FILL listing part cached'; exit 0; fi; python -m tools.prebuilt.fill list --backend $be --profile $pr --part $i/$OPTIX_PARTS $(only)$FILL_ARGS --out /tmp/list 2>&1 | grep --line-buffered -E '^FILL|Error|Traceback' && mkdir -p $listed && cp /tmp/list/part-$i.json $listed/.part-$i.json.partial && mv $listed/.part-$i.json.partial $listed/part-$i.json" ;;
    pack-*)
      local i=${job#pack-} keep_dir="$CACHE/sm$arch/optix/\$drv" n driver check url
      n=$(packs "$arch") driver=$(per_arch DRIVER "$arch" "")
      url=$(per_arch SEED_URL "$arch" "")
      check=""
      [ -z "$driver" ] || check="[ \"\$drv\" = $driver ] || { echo \"FATAL: driver \$drv, the pack is for $driver\"; exit 3; };"
      echo "$(pack_flavor "$arch") $PACK_CAP"
      echo "$BEAT drv=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); echo \"pack driver \$drv\"; $check mkdir -p /tmp/b /tmp/census && cp $b-merged-ptx.tar /tmp/b.tar && tar -xf /tmp/b.tar -C /tmp/b && cp $b-census.tar /tmp/c.tar && tar -xf /tmp/c.tar -C /tmp/census && python -m tools.prebuilt.build pack --bundle /tmp/b/raywarp_kernels/sm$arch --workers $(pack_workers "$arch") --out /tmp/pack --shard $i/$n --select /tmp/census/*.json --groups $(pack_groups "$arch") --keep $keep_dir ${url:+--seed $url }--deadline \$((HF_DEADLINE - $PACK_MARGIN)) 2>&1 | grep --line-buffered -E '^pack|Error|Traceback' && $(keep /tmp/pack "$b-pack-$i.tar" raywarp_optix_modules)" ;;
    p* | w*)
      local i=${job:1} root=/tmp/fill part="$b-partial-$job.tar" final="$b-final-$job.tar" fill
      case $job in
        # Lens-camera PTX: the fast tier's distortion program only (`fill.LENS_TIERS`; every model's: `fill --lens`).
        p*) fill="--profile $pr --workers $WORKERS --shard $lists/ptx-$i.json" ;;
        w*) fill="--warp --backend $be --profile $pr --workers $WORKERS --shard $lists/warp-$i.json" ;;
      esac
      local seed; seed=$(bundle_seed "$arch")
      fill="$fill --cache $CACHE${seed:+ --seed $seed}"
      echo "$SHARD_FLAVOR $SHARD_CAP"
      # Resume from the last snapshot; snapshot every SNAP_EVERY s while the fill runs; then the
      # shard's bundle (`build.py collect`) as its final archive.
      local snap="(while sleep $SNAP_EVERY; do tar -cf $part.partial -C $root . 2>/dev/null && mv $part.partial $part && echo \"FILL snapshot \$(du -h $part | cut -f1)\"; done) & SNAP=\$!;"
      echo "$BEAT $(wait_for "$parts" "$OPTIX_PARTS") $spread mkdir -p $root && { [ ! -f $part ] || tar -xf $part -C $root; } && echo \"FILL resumed \$(find $root -name '*.ptx' -o -name '*.cubin' | wc -l) files\"; $snap python -m tools.prebuilt.fill fill --root $root $fill $FILL_ARGS 2>&1 | grep --line-buffered -E '^FILL|Error|Traceback'; rc=\$?; kill \$SNAP; [ \$rc = 0 ] && python -m tools.prebuilt.build collect --root $root --arch $arch --out /tmp/dist --workers 8 2>&1 | grep -E '^(collect|bundle)|Error|Traceback' && $(keep /tmp/dist "$final" raywarp_kernels)" ;;
    merge)
      local shards="" s census_wait="" census_get="true" census_arg="" census_keep="true" plan="true" ck driver url
      for s in $(shards "$arch"); do shards="$shards ${s#sm$arch-}"; done
      if [ "$be" = optix ]; then
        # The census rows of every scene, from the cache (this fill's census jobs keep them there).
        ck=$(state census-key) driver=$(per_arch DRIVER "$arch" "") url=$(per_arch SEED_URL "$arch" "")
        census_wait=$(wait_for "$CACHE/sm$arch/census/$ck/*.json" "$(all_scenes | grep -c .)")
        census_get="mkdir -p /tmp/census/sm$arch && cp $CACHE/sm$arch/census/$ck/*.json /tmp/census/sm$arch/"
        census_arg="--census /tmp/census/sm$arch/*.json"
        census_keep="$(keep /tmp/census/sm$arch "$b-census.tar")"
        # With a release driver: the modules left to make (neither in the cache nor in the seed pack).
        [ -z "$driver" ] || plan="python -m tools.prebuilt.build plan-pack --bundle /tmp/dist/raywarp_kernels/sm$arch --select /tmp/census/sm$arch/*.json --groups $(pack_groups "$arch") --keep $CACHE/sm$arch/optix/$driver ${url:+--seed $url }--driver $driver --out /tmp/todo 2>&1 | grep -E '^pack|Error|Traceback' && cp /tmp/todo $b-pack-todo"
      fi
      echo "cpu-upgrade $MERGE_CAP"
      echo "$BEAT $(wait_for "$b-final-[pw]*.tar" "$((np + nw))") $census_wait mkdir -p /tmp/s /tmp/census && for s in$shards; do cp $b-final-\$s.tar /tmp/s.tar && mkdir -p /tmp/s/\$s && tar -xf /tmp/s.tar -C /tmp/s/\$s && rm /tmp/s.tar || exit 1; done && $census_get && python -m tools.prebuilt.build merge \$(for s in$shards; do echo --bundle /tmp/s/\$s/raywarp_kernels/sm$arch; done) $census_arg --out /tmp/dist 2>&1 | grep -E '^(bundle|merge)|Error|Traceback' && $(keep /tmp/dist "$b-merged.tar" raywarp_kernels) && { [ ! -d /tmp/dist/raywarp_kernels/sm$arch/ptx ] || $(keep /tmp/dist "$b-merged-ptx.tar" "raywarp_kernels/sm$arch/ptx raywarp_kernels/sm$arch/manifest.json"); } && $census_keep && $plan" ;;
    release)
      local n wait="" got="true" pack="" refs driver url ck=""
      n=$(packs "$arch") driver=$(per_arch DRIVER "$arch" "") url=$(per_arch SEED_URL "$arch" "")
      [ "$be" != optix ] || ck=$(state census-key)
      if (( n )); then
        wait=$(wait_for "$b-pack-[0-9]*.tar" "$n")
        got="for i in \$(seq 0 $((n - 1))); do mkdir -p /tmp/p/\$i && cp $b-pack-\$i.tar /tmp/p.tar && tar -xf /tmp/p.tar -C /tmp/p/\$i || exit 1; done && python -m tools.prebuilt.build merge-packs \$(for d in /tmp/p/*/raywarp_optix_modules/sm$arch-drv*; do echo --pack \$d; done) --out /tmp/pack"
      elif assembled "$arch"; then
        # Every module is in the cache or the seed pack: the pack assembled here, no GPU.
        got="mkdir -p /tmp/census && cp $b-census.tar /tmp/c.tar && tar -xf /tmp/c.tar -C /tmp/census && python -m tools.prebuilt.build pack --bundle /tmp/b/raywarp_kernels/sm$arch --assemble $driver --out /tmp/pack --select /tmp/census/*.json --groups $(pack_groups "$arch") --keep $CACHE/sm$arch/optix/$driver ${url:+--seed $url} 2>&1 | grep --line-buffered -E '^pack|Error|Traceback'"
      fi
      if (( n )) || assembled "$arch"; then pack="--pack \$(ls -d /tmp/pack/raywarp_optix_modules/sm$arch-drv*)"; fi
      refs="python -m tools.prebuilt.build_cache refs --arch $arch --bundle /tmp/b/raywarp_kernels/sm$arch ${pack} --list-key $lk${ck:+ --census-key $ck} --out /tmp/refs.json && cp /tmp/refs.json $b-refs.json"
      echo "$RELEASE_FLAVOR $RELEASE_CAP"
      echo "$BEAT $wait mkdir -p /tmp/b /tmp/p && cp $b-merged.tar /tmp/b.tar && tar -xf /tmp/b.tar -C /tmp/b && rm /tmp/b.tar && $got && python -m tools.release.kernels layout /tmp/b/raywarp_kernels/sm$arch /tmp/layout $pack --workers 8 && du -sh /tmp/layout && $refs && $(keep /tmp/layout "$b-release.tar")" ;;
    *) echo "unknown job $name" >&2; return 2 ;;
  esac
}

shards() {  # A: every shard of architecture A: sm<A>-p<I> (PTX), sm<A>-w<I> (Warp modules)
  local i np nw
  np=$(ptx_shards "$1") nw=$(warp_shards "$1")
  for ((i = 0; i < np; i++)); do echo "sm$1-p$i"; done
  for ((i = 0; i < nw; i++)); do echo "sm$1-w$i"; done
}

censuses() {  # every census job that has scenes (none without an OptiX architecture)
  local i
  [ -n "$(optix_archs)" ] || return 0
  for ((i = 0; i < CENSUS_JOBS; i++)); do [ -z "$(census_scenes "$i")" ] || echo "census-$i"; done
  [ -z "$(census_scenes heavy)" ] || echo census-heavy
}

wave1() {  # every job of wave 1, in launch order: what the others wait for first (each `hf run`
  # uploads the tree: ~1-2 min per wave of 10, so the last of ~100 jobs starts ~15 min after the
  # first); no listing part the cache holds
  local a i
  for a in $ARCHS; do
    for ((i = 0; i < OPTIX_PARTS; i++)); do grep -qx "sm$a-list-$i" "$STATE.cached-jobs" 2>/dev/null || echo "sm$a-list-$i"; done
  done
  for a in $ARCHS; do shards "$a"; done
  censuses
  for a in $ARCHS; do echo "sm$a-merge"; done
}

wave2() {  # every job of wave 2
  local a i
  for a in $ARCHS; do for ((i = 0; i < $(packs "$a"); i++)); do echo "sm$a-pack-$i"; done; done
  for a in $ARCHS; do echo "sm$a-release"; done
}

# --- keys and the build cache ------------------------------------------------------------------

state() { cat "$STATE.$1"; }  # NAME: a value `prepare` wrote

key_inputs() {  # A: the --input arguments of architecture A's bundle key (beyond the code)
  local a=$1
  printf '%s\n' --input "backend=$(backend "$a")" --input "profile=$(profile "$a")" --input "only=$ONLY" --input "fill_args=$FILL_ARGS"
  if [ "$(backend "$a")" = optix ]; then
    printf '%s\n' --input "driver=$(per_arch DRIVER "$a" "")" --input "groups=$(pack_groups "$a")" --input "census=$(state census-key)"
  fi
}

cache_ls() {  # PREFIX: the cache bucket's files under PREFIX (paths from its root)
  command $HF_CLI buckets list "$CACHE_BUCKET/$1" -R --format quiet
}

keys() {  # once per STATE: every key, the architectures to build (ARCHS: those whose key the pins
  # lack; git and the standard library only: seconds)
  local a keys="{" sep="" scenes built
  if [ ! -f "$STATE.archs" ]; then
    scenes=$(mktemp) && all_scenes > "$scenes"
    $PY "$FILL_ALL_TOOLS/buildkey.py" census --scenes "$scenes" > "$STATE.census-key"
    rm -f "$scenes"
    for a in $ARCHS; do
      mapfile -t inputs < <(key_inputs "$a")
      keys="$keys$sep\"$a\": \"$($PY "$FILL_ALL_TOOLS/buildkey.py" bundle --arch "$a" "${inputs[@]}")\"" sep=", "
      $PY "$FILL_ALL_TOOLS/buildkey.py" list --arch "$a" --input "backend=$(backend "$a")" --input "profile=$(profile "$a")" \
        --input "only=$ONLY" --input "fill_args=$FILL_ARGS" --input "parts=$OPTIX_PARTS" > "$STATE.list-key-$a"
    done
    echo "$keys}" > "$STATE.keys.json"
    if [ "$FORCE" = 1 ]; then built=$ARCHS; else
      built=$($PY "$FILL_ALL_TOOLS/buildkey.py" changed --keys "$STATE.keys.json" --pins "$PINS" --bucket "$BUCKET")
    fi
    echo "$built" > "$STATE.archs"
  fi
  ARCHS=$(state archs)
}

prepare() {  # once per STATE: `keys`, then the census scenes the cache lacks, the listing parts it holds
  local a i lk ck s
  keys
  [ ! -f "$STATE.census-todo" ] || return 0
  ck=$(state census-key)
  : > "$STATE.cached-jobs"
  local cached; cached=$(mktemp)
  for a in $ARCHS; do
    lk=$(state "list-key-$a")
    cache_ls "sm$a/list/$lk" >> "$cached"
    for ((i = 0; i < OPTIX_PARTS; i++)); do grep -qx "sm$a/list/$lk/part-$i.json" "$cached" && echo "sm$a-list-$i" >> "$STATE.cached-jobs"; done
  done
  for a in $(optix_archs); do cache_ls "sm$a/census/$ck" >> "$cached"; done
  for s in $(all_scenes); do  # a scene some built OptiX architecture lacks
    for a in $(optix_archs); do grep -qx "sm$a/census/$ck/$s.json" "$cached" || { echo "$s"; break; }; done
  done > "$STATE.census-todo"
  rm -f "$cached"
}

bundle_seed() {  # A: the published bundle the shards of A take unchanged units from ("": none)
  [ "$BUNDLE_SEED" = auto ] || { echo "$BUNDLE_SEED"; return; }
  $PY -c "
import json, sys
p = json.load(open('$PINS')) if __import__('os').path.isfile('$PINS') else {}
b = p.get('bundles', {}).get('$1') if p.get('bucket') == '$BUCKET' else None
print(f'https://huggingface.co/buckets/$BUCKET/resolve/sm$1/{b[\"content\"]}' if b else '')"
}

plan_packs() {  # each OptiX architecture with a release driver: its pack jobs from the modules its
  # merge found left to make (none: the release assembles the pack), once
  local a todo n per most tmp
  tmp=$(mktemp)
  for a in $(optix_archs); do
    grep -q "^$a " "$STATE.packs" 2>/dev/null && continue
    [ -n "$(per_arch DRIVER "$a" "")" ] || continue
    command $HF_CLI buckets cp "hf://buckets/$PROJECT_BUCKET/runs/$(day)-$tag-sm$a-pack-todo" "$tmp" || continue
    todo=$(tr -dc 0-9 < "$tmp") per=$(per_arch PACK_MODULES "$a" 400) most=$(packs "$a")
    n=$(( (todo + per - 1) / per )); (( n <= most )) || n=$most
    echo "$a $n" >> "$STATE.packs"
    echo "sm$a: $todo pack modules left to make: $n pack jobs"
  done
  rm -f "$tmp"
}

# --- cost --------------------------------------------------------------------------------------

# $/h of the flavors (tools.hf.jobs.prices(), `hf jobs hardware`, 2026-11-16).
price() {
  case $1 in cpu-upgrade) echo 0.03 ;; cpu-performance) echo 1.90 ;; l4x1) echo 0.80 ;; l40sx1) echo 1.80 ;;
    a10g-large) echo 1.50 ;; rtx-pro-6000) echo 2.75 ;; a100-large) echo 2.50 ;; *) echo "no price for $1" >&2; return 1 ;; esac
}

minutes() { local c=${1%m}; echo "${c%h}"; }

cost() {  # the worst case (every job runs to its cap) per architecture, and the total
  local a job flavor cap total=0 sum line
  for a in $ARCHS census; do
    sum=0
    if [ "$a" = census ]; then set -- $(censuses); else set -- $(wave1 | grep "^sm$a-") $(wave2 | grep "^sm$a-"); fi
    for job in "$@"; do
      read -r flavor cap <<<"$(job_spec "$job" | head -1)"
      sum=$(awk -v s="$sum" -v m="$(minutes "$cap")" -v p="$(price "$flavor")" 'BEGIN {print s + m / 60 * p}')
    done
    line=$(printf '%-8s %3d jobs  worst $%.2f' "$a" "$#" "$sum")
    echo "$line"
    total=$(awk -v t="$total" -v s="$sum" 'BEGIN {print t + s}')
  done
  printf 'total    worst $%.2f\n' "$total"
}

# --- orchestration -----------------------------------------------------------------------------

record() {  # NAME RUNTAG: the job's current run id (before its launch; never from a background job)
  sed -i "/^$1 /d" "$STATE"
  echo "$1 $(date -u +%F)-$2" >> "$STATE"
}

# The extras of job NAME (`tools.hf run --with`): every Warp shard gets the fill's NVRTC / nvJitLink
# wheels (fillcuda: the software path's word modules link with them, `tools/prebuilt/software.py`;
# an OptiX architecture's Warp shards have no word module, which is why only sm_80 / sm_90 failed
# without them).
extras() { case $1 in sm*-w[0-9]*) echo "--with fillcuda" ;; esac; }

start() {  # NAME RUNTAG: launch one job (its run id recorded first)
  local spec flavor cap command
  spec=$(job_spec "$1")
  read -r flavor cap <<<"$(head -1 <<<"$spec")"
  command=$(tail -n +2 <<<"$spec")
  # No compiler cache is read (the fill compiles into its own root, the pack creates into a fresh
  # OptiX cache): none restored (~25 GB CPU, ~37 GB GPU, 1.5-6 min) or saved.
  # The build cache mounted read-write at CACHE (what earlier fills made, by content).
  # shellcheck disable=SC2046  # the extras' words
  hf run --flavor "$flavor" $(extras "$1") --no-cache-restore warp,cuda,raywarp --no-cache-save warp,cuda,raywarp --mount "$CACHE_BUCKET:$CACHE" --lane "$LANE" --tag "$2" --timeout "$cap" --stall 30m -- "$command" | tail -1
}

launch() {  # NAME...: every job not yet launched, in waves of 10 (the bucket mount fails under ~100 starts)
  local n=0 name
  for name in "$@"; do
    grep -q "^$name " "$STATE" && continue
    record "$name" "$tag-$name"
    start "$name" "$tag-$name" &
    (( ++n % 10 )) || wait
  done
  wait
}

ended_ok() {  # RUN: the run's status says exit 0
  hf status "$1" 2>/dev/null | grep -q '"exit": 0'
}

follow() {  # NAME...: until each job ended ok (relaunching a failed one up to RETRIES times); 1: one
  # gave up (FAIL_FAST=1: at once, the others killed)
  local running name rid tries left gave_up
  while true; do
    running=$(hf ps --limit 1000) || { echo "$(date -u +%H:%M) job list failed: retrying"; sleep "$WATCH_EVERY"; continue; }
    left=0 gave_up=0
    for name in "$@"; do
      grep -qx "$name" "$STATE.done" 2>/dev/null && continue
      rid=$(sed -n "s/^$name //p" "$STATE")
      if grep -q " $rid\$" <<<"$running"; then left=$((left + 1)); continue; fi
      if ended_ok "$rid"; then echo "$name" >> "$STATE.done"; echo "$(date -u +%H:%M) $name ok ($rid)"; continue; fi
      tries=$(grep -c "^$name " "$STATE.tries" 2>/dev/null) || tries=0
      if (( tries >= RETRIES )); then
        echo "$(date -u +%H:%M) $name: $rid failed after $tries relaunches: by hand (\`hf logs $rid\`)"
        gave_up=1
        if [ "$FAIL_FAST" = 1 ]; then abort "$name" "$running" "$@"; return 1; fi
        continue
      fi
      echo "$name $rid" >> "$STATE.tries"
      echo "$(date -u +%H:%M) relaunch $name (was $rid)"
      record "$name" "$tag-$name-r$((tries + 1))"
      start "$name" "$tag-$name-r$((tries + 1))"
      left=$((left + 1))
    done
    (( left )) || return "$gave_up"
    echo "$(date -u +%H:%M) $left running"
    sleep "$WATCH_EVERY"
  done
}

abort() {  # FAILED RUNNING NAME...: kill every running job of NAME (the `hf ps` lines RUNNING) and say why
  local failed=$1 running=$2 name rid
  shift 2
  for name in "$@"; do
    rid=$(sed -n "s/^$name //p" "$STATE")
    if [ -n "$rid" ] && grep -q " $rid\$" <<<"$running"; then
      echo "$(date -u +%H:%M) kill $name ($rid)"
      hf kill "$rid" || echo "$(date -u +%H:%M) kill $rid failed: cancel it by hand"
    fi
  done
  echo "FILL FAILED: $failed failed after $RETRIES relaunches; every other running job of the wave was killed (FAIL_FAST=1)."
  echo "Fix the cause, then rerun \`refill $tag\`: jobs done stay done, the failed and the killed ones relaunch."
}

publish() {  # every built architecture's release archive here, laid out as one, pinned (the skipped
  # ones' entries kept) and published (PREFIX: under that place); its cache references recorded
  local scratch a
  scratch=$(mktemp -d)
  mkdir -p "$scratch/layout"
  for a in $ARCHS; do
    command $HF_CLI buckets cp "hf://buckets/$PROJECT_BUCKET/runs/$(day)-$tag-sm$a-release.tar" "$scratch/sm$a.tar"
    tar -xf "$scratch/sm$a.tar" -C "$scratch/layout" && rm "$scratch/sm$a.tar"
    command $HF_CLI buckets cp "hf://buckets/$PROJECT_BUCKET/runs/$(day)-$tag-sm$a-refs.json" "$scratch/refs-$a.json"
  done
  $PY -m tools.release.kernels pin "$scratch/layout" --to "$PINS" --bucket "$BUCKET" --keys "$STATE.keys.json"
  $PY -m tools.release.kernels publish "$scratch/layout" --bucket "$BUCKET" ${PREFIX:+--prefix "$PREFIX"}
  if [ -z "$PREFIX" ]; then
    $PY -c "import json, sys; print(json.dumps({'tag': '$tag', 'archs': [json.load(open(f)) for f in sys.argv[1:]]}))" "$scratch"/refs-*.json > "$scratch/refs.json"
    command $HF_CLI buckets cp "$scratch/refs.json" "hf://buckets/$CACHE_BUCKET/refs/$tag.json"
  fi
  echo "$PINS written$([ -n "$PREFIX" ] || echo ': commit it')"
  cat "$PINS"
  rm -rf "$scratch"
}

case $cmd in
  refill | watch | spec | cost | plan | keys)
    case $cmd in spec | cost | plan)  # a look only: a fill not started keeps no state of it
      if [ ! -f "$STATE.archs" ]; then
        t=$(mktemp) && { cp "$STATE" "$t" 2>/dev/null || echo "day $(date -u +%F)" > "$t"; } && STATE=$t
      fi ;;
    esac
    mkdir -p "$(dirname "$STATE")"
    [ -f "$STATE" ] || echo "day $(date -u +%F)" > "$STATE"
    if [ "$cmd" = keys ]; then keys; else prepare; fi ;;
  status) ARCHS=$(state archs) ;;
esac
case $cmd in
  refill | watch | spec)
    for a in $(optix_archs); do
      url=$(seed_url "$a") || exit 1
      export "SEED_URL_$a=$url"
    done ;;
esac
case $cmd in
  refill)
    touch "$STATE.stage"
    # A rerun is the operator's (or the workflow's) go after a fix: every job gets RETRIES again.
    rm -f "$STATE.tries"
    if [ -z "$ARCHS" ]; then
      echo "nothing changed: every architecture's key is pinned in $PINS ($(cat "$STATE.keys.json")); nothing launched"
      exit 0
    fi
    echo "building sm$(echo $ARCHS | sed 's/ / sm/g') (keys $(cat "$STATE.keys.json")); census scenes to probe: $(grep -c . "$STATE.census-todo" || true)"
    if ! grep -qx wave1 "$STATE.stage"; then
      launch $(wave1)
      follow $(wave1) || { echo "wave 1 incomplete: fix the jobs named above, then rerun"; exit 1; }
      echo wave1 >> "$STATE.stage"
    fi
    if ! grep -qx wave2 "$STATE.stage"; then
      plan_packs
      launch $(wave2)
      follow $(wave2) || { echo "wave 2 incomplete: fix the jobs named above, then rerun"; exit 1; }
      echo wave2 >> "$STATE.stage"
    fi
    echo "release: $(for a in $ARCHS; do echo -n "$(base)-sm$a-release.tar "; done)"
    if [ "$PUBLISH" = 1 ] && ! grep -qx "publish$PREFIX" "$STATE.stage"; then
      publish && echo "publish$PREFIX" >> "$STATE.stage"
    elif [ "$PUBLISH" != 1 ]; then
      echo "not published: PUBLISH=1 tools/prebuilt/fill_all.sh refill $tag (after the approval)"
    fi ;;
  watch)  # watch TAG [JOB...]: follow these jobs (default: every recorded one) as `refill` does
    shift 2
    (( $# )) || set -- $(sed -n 's/^\([^ ]*\) .*/\1/p' "$STATE" | grep -v '^day$')
    follow "$@" ;;
  status)
    for name in $(wave1) $(wave2); do
      rid=$(sed -n "s/^$name //p" "$STATE")
      echo "$name ${rid:-not launched} $(grep -qx "$name" "$STATE.done" 2>/dev/null && echo ok)"
    done ;;
  unpublish)
    [ -n "$PREFIX" ] || { echo "unpublish removes a PREFIX (scratch) upload only" >&2; exit 2; }
    command $HF_CLI buckets rm "$BUCKET/${PREFIX%/}/" --recursive -y ;;
  spec)  # print one job's flavor, cap and command (a check of the generated scripts)
    job_spec "$3" ;;
  cost)
    cost ;;
  keys)  # the architectures to build, one line (the workflow's first step: no job, no install)
    echo "$ARCHS" ;;
  plan)  # the architectures to build and their worst case
    echo "keys $(cat "$STATE.keys.json")"
    echo "build: ${ARCHS:-nothing (every key pinned)}"
    [ -z "$ARCHS" ] || cost ;;
  *)
    echo "usage: $0 refill|watch|status|unpublish|spec|cost|keys|plan TAG [JOB...]" >&2
    exit 2 ;;
esac
