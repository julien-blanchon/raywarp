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
#   tools/prebuilt/fill_all.sh cost TAG                # the worst case per architecture (count x cap x $/h)
#
# Architectures (ARCHS, default "120 89 86 90 80"; internal/research/kb/lanes/K-ARCH.md): one bundle each.
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
#                   listed once) -> runs/<day>-<tag>-sm<A>-list/part-<i>.json
#   sm<A>-p<i>      PTX shards (cpu-upgrade, OptiX architectures): wait for the listing, `fill spread`
#                   locally (the same shard files in every job), `fill fill --shard`
#                   -> runs/<day>-<tag>-sm<A>-final-p<i>.tar
#   sm<A>-w<i>      Warp module shards (cpu-upgrade), the same way -> runs/<day>-<tag>-sm<A>-final-w<i>.tar
#   sm<A>-merge     (cpu-upgrade) waits for its shards and (OptiX) the census: `build.py merge
#                   --census` -> runs/<day>-<tag>-sm<A>-merged.tar (the bundle), -merged-ptx.tar
#                   (its PTX: the packs'), -census.tar (its census files)
# and once for every OptiX architecture:
#   census-<i>      CENSUS_JOBS cpu-upgrade jobs + census-heavy (cpu-performance: the scenes of HEAVY,
#                   large prepared stages: cpu-upgrade lost the /data mount on them): `word_reach.py
#                   --units --profiles` over the registry, each scene probed once for every OptiX
#                   architecture's profile -> runs/<day>-<tag>-census-<i>.tar (sm<A>/<scene>.json)
# Wave 2, after the merges:
#   sm<A>-pack-<i>  PACKS_<A> jobs on the architecture's GPU flavor: the OptiX module pack of the
#                   census modules in PACK_GROUPS_<A>'s order until a deadline, on that flavor's
#                   driver (DRIVER_<A>: checked in the job when set; empty: the job's, recorded in the
#                   pack's manifest, `merge-packs` refuses packs of two drivers), one file per module
#                   (each kept on the bucket as made: resumable; those the published pack of the seed
#                   holds fetched from it, not made) -> runs/<day>-<tag>-sm<A>-pack-<i>.tar
#   sm<A>-release   waits for its packs: `build.py merge-packs`, `tools/release/kernels.py layout`
#                   (bundle + pack) -> runs/<day>-<tag>-sm<A>-release.tar
# Then, here: with PUBLISH=1 every architecture's release archive downloaded into one layout,
# `kernels.py pin` (one pins file for every architecture and its packs) and `kernels.py publish`
# (refuses an existing content key or pack); raywarp/core/build/kernel_cache/kernels.json written
# (commit it; with PREFIX the pins go to $STATE.kernels.json only).
#
# DRY=1: the orchestration proved on CPU, tiny and cheap (~12 cpu-upgrade jobs of a few minutes,
# < $0.10): ARCHS "120 80" (an OptiX and a software architecture) unless set, one listing part, one
# PTX and one Warp shard per architecture of the ONLY points (default: one scene-free point), one
# census job over DRY_SCENES, no pack (PACKS 0), every job on cpu-upgrade, PREFIX scratch/dry-TAG.
#
# Never `cp -r` or reads many files in place from /data: every hand-over is one archive, copied to
# local disk and unpacked (one sequential read). No job restores a compiler cache family
# (`--no-cache-restore warp,cuda,wpbr`: the fill compiles into its own root, a pack into a fresh
# OptiX cache). Every job prints a HEARTBEAT each minute (the stall watchdog), shards snapshot their
# output to the bucket every SNAP_EVERY s (one tar) and resume from it; `refill` relaunches a
# failed / stalled / vanished job up to RETRIES times. State: $STATE (.fill_all/TAG: the day and each
# job's current run id; `.done`: jobs ended ok; `.stage`: waves passed). Run from the committed tree
# and keep that tree for the whole fill (the PTX keys and the shard files name its code). Bash only.
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
# Bash reads a script as it runs: a long `refill` runs from a private copy (a checkout or edit of
# this file meanwhile cannot change it).
if [ -z "${FILL_ALL_COPY:-}" ]; then
  copy=$(mktemp --suffix=.fill_all.sh) && cp "$0" "$copy" && FILL_ALL_COPY=$copy exec bash "$copy" "$@"
fi
set -euo pipefail
cmd=$1 tag=$2
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then
  ARCHS=${ARCHS:-120 80} OPTIX_PARTS=${OPTIX_PARTS:-1} PTX=${PTX:-1} WARP=${WARP:-1} CENSUS_JOBS=${CENSUS_JOBS:-1}
  PACKS=${PACKS:-0} HEAVY=${HEAVY-} PREFIX=${PREFIX:-scratch/dry-$tag} SEED=${SEED-}
  ONLY=${ONLY:-scene=fs_point0_plain_none_mesh_n,tier=fast,layout=shared,gi=False,outputs=shaded}
  SHARD_CAP=${SHARD_CAP:-30m} MERGE_CAP=${MERGE_CAP:-30m} RELEASE_CAP=${RELEASE_CAP:-30m} RELEASE_FLAVOR=cpu-upgrade
  if [ -z "${SCENES_FILE:-}" ]; then
    SCENES_FILE=$(mktemp --suffix=.scenes) && echo "${DRY_SCENES:-fs_point1_plain_none_mesh_n}" > "$SCENES_FILE"
  fi
fi
# Sizing (internal/research/kb/lanes/W10-REFILL.md: the probe and the plan), per architecture.
ARCHS=${ARCHS:-120 89 86 90 80}
PTX=${PTX:-40} WARP=${WARP:-16} OPTIX_PARTS=${OPTIX_PARTS:-12} CENSUS_JOBS=${CENSUS_JOBS:-8} PACKS=${PACKS:-}
WORKERS=${WORKERS:-4}  # NVRTC processes per PTX shard (cpu-upgrade: 4 at once is the best rate)
SHARD_FLAVOR=${SHARD_FLAVOR:-cpu-upgrade}  # cpu-performance: ~2x faster per build, ~60x the price
RETRIES=${RETRIES:-2} WATCH_EVERY=${WATCH_EVERY:-120} SNAP_EVERY=${SNAP_EVERY:-600} POLL=${POLL:-30}
# Caps per job kind (worst case = the count x the cap x the flavor's $/h: `cost`).
LIST_CAP=${LIST_CAP:-40m} CENSUS_CAP=${CENSUS_CAP:-45m} SHARD_CAP=${SHARD_CAP:-75m} MERGE_CAP=${MERGE_CAP:-120m}
PACK_CAP=${PACK_CAP:-35m} RELEASE_CAP=${RELEASE_CAP:-120m} RELEASE_FLAVOR=${RELEASE_FLAVOR:-}
# The pack (user-approved for sm_120, 2026-11-11: 9 jobs x 35 min, worst $14.4): the census modules
# group by group (`build.py pack --groups`), each group most used first: the quality recipe's own
# words first in time (7-15 min each on the RTX PRO 6000, BUG-583: started at once, spread over the
# jobs, so they end before the deadline), then the fast and default recipes' own words, until
# PACK_MARGIN s before the job's end (`--deadline`: no module starts after it, nor one of a build
# whose modules took longer than the time left; what is done is the pack; the cap is the cost).
# The L4 / A10G (8 / 12 vCPU: OptiX module creation is host-bound, ~4x / ~3x slower than the RTX
# PRO 6000's 32 workers; unmeasured) pack the fast and default recipes' own words only.
PACK_MARGIN=${PACK_MARGIN:-600}
L4_FLAVOR=${L4_FLAVOR:-l4x1}  # sm_89: l4x1 ($0.80/h) or l40sx1 ($1.80/h, more cores)
# The release driver of each architecture's module pack ("" : the pack job's own, recorded).
DRIVER_120=${DRIVER_120-580.178.04} DRIVER_89=${DRIVER_89-} DRIVER_86=${DRIVER_86-}
# The seed of the packs (`build.py pack --seed`): a published pack of the same architecture and
# driver whose modules the packs fetch instead of making them (a module's name is the digest of its
# PTX and record: unchanged ones keep their names; the pack stays the census's set). SEED=auto:
# for each OptiX architecture with a DRIVER_<A>, the bundle raywarp/core/build/kernel_cache/
# kernels.json pins for it, if its pins list a pack of that driver (format 1 pins list no packs:
# assumed), and the tree its refill ran from (the last commit of kernels.json: "... (refill TAG
# from REV)"); SEED_<A>=CONTENT@REV names one; SEED="": none. The name does not cover the code that
# turns a record into OptiX's compile options: a refill whose SEED_GUARD files differ from REV's
# refuses the seed (a module of other options would be a cache miss at first use); read that diff,
# then narrow SEED_GUARD or set SEED="".
SEED=${SEED-auto}
SEED_GUARD=${SEED_GUARD-raywarp/core/trace/optix/pipeline.py raywarp/core/trace/optix/device/intrinsics.py raywarp/core/trace/optix/context.py raywarp/core/trace/optix/api.py raywarp/core/trace/optix/gas.py raywarp/core/trace/optix/modules.py}
# Registry scenes with large prepared stages: their census runs on cpu-performance.
HEAVY=${HEAVY-bistro hospital full_warehouse office_usd a_beautiful_game assembly_studio infinigen_dining isaac_warehouse intel_sponza_full}
# The census scenes (whitespace separated): the benchmark registry's and feature-space scenes
# (`tools/prebuilt/word_reach.py`). A public checkout (no internal/: no registry) takes the
# feature-space ones (fs_*) only.
SCENES_FILE=${SCENES_FILE:-tools/prebuilt/census_scenes.txt}
ONLY=${ONLY:-}  # a dry run: `fill list --only` KEY=VALUE pairs (comma separated): only those points listed
# The fill's exclusions: passed to both `fill list` (the points listed, so no shard is given them)
# and every `fill fill` shard.
FILL_ARGS=${FILL_ARGS:-}
STATE=${STATE:-.fill_all/$tag}
PY=${PY:-python}  # the local interpreter of tools.hf and tools.release (the jobs run their own `python`)
HF_CLI=${HF_CLI:-hf}  # the huggingface_hub CLI (the bucket copies of `publish`)
DATA=${DATA:-/data}  # the jobs' mount of the project bucket
LANE=${LANE:-release}
BUCKET=${BUCKET:-blanchon/raywarp-kernels} PREFIX=${PREFIX:-} PUBLISH=${PUBLISH:-0}
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

packs() {  # A: its pack jobs (0 on the software path)
  local d=0
  if [ "$(backend "$1")" = optix ]; then
    case $1 in 120) d=9 ;; 89) d=6 ;; 86) d=4 ;; esac
    d=${PACKS:-$d}
  fi
  per_arch PACKS "$1" "$d"
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

warp_shards() { per_arch WARP "$1" "$WARP"; }

release_flavor() {  # A: the release job's flavor (an OptiX bundle + pack: large /data reads)
  [ -z "$RELEASE_FLAVOR" ] || { echo "$RELEASE_FLAVOR"; return; }
  if [ "$(backend "$1")" = optix ]; then echo cpu-performance; else echo cpu-upgrade; fi
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
    seed=$(git log -1 --format=%s -- raywarp/core/build/kernel_cache/kernels.json | sed -n "s/.*bundle \($content\)[ ,].*from \([0-9a-f]*\))\$/\1@\2/p")
    [ -n "$seed" ] || { echo "SEED=auto: the last commit of raywarp/core/build/kernel_cache/kernels.json names no refill of bundle $content: set SEED_$1=CONTENT@REV" >&2; return 1; }
  fi
  content=${seed%@*} rev=${seed#*@}
  changed=""
  [ -z "$SEED_GUARD" ] || changed=$(git diff --name-only "$rev" HEAD -- $SEED_GUARD) || return 1
  [ -z "$changed" ] || { echo "SEED $seed (sm$1): the options code changed since $rev ($changed): read the diff, then narrow SEED_GUARD or set SEED=\"\"" >&2; return 1; }
  echo "https://huggingface.co/buckets/$BUCKET/resolve/sm$1/$content/optix/$driver"
}

census_scenes() {  # I: the scenes of census job I (census-heavy: the HEAVY ones)
  local all; all=$(tr -s ' \n' '\n' < "$SCENES_FILE" | grep -v '^$')
  [ -d internal/benchmarks ] || all=$(grep '^fs_' <<<"$all" || true)
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
    echo "$flavor $CENSUS_CAP"
    echo "$BEAT mkdir -p /tmp/near && for s in ${scenes//,/ }; do python -m tools.prebuilt.word_reach --scenes \$s --profiles $profiles --out /tmp/reach/\$s.sm{arch}.json --units /tmp/near/sm{arch}/\$s.json 2>&1 | grep --line-buffered -E '^REACH|Error|Traceback' || exit 1; done && $(keep /tmp/near "$(base)-census-$i.tar")"
    return
  fi
  local arch=${name%%-*} job=${name#*-}
  arch=${arch#sm}
  local b; b="$(base)-sm$arch"
  local be pr np nw lists="/tmp/lists" parts="$b-list/part-*.json"
  be=$(backend "$arch") pr=$(profile "$arch") np=$(ptx_shards "$arch") nw=$(warp_shards "$arch")
  local spread="mkdir -p $lists && cp $parts $lists/ && python -m tools.prebuilt.fill spread --ptx-shards $np --warp-shards $nw --out $lists | grep FILL &&"
  case $job in
    list-*)
      local i=${job#list-}
      echo "cpu-upgrade $LIST_CAP"
      echo "$BEAT python -m tools.prebuilt.fill list --backend $be --profile $pr --part $i/$OPTIX_PARTS $(only)$FILL_ARGS --out /tmp/list 2>&1 | grep --line-buffered -E '^FILL|Error|Traceback' && mkdir -p $b-list && cp /tmp/list/part-$i.json $b-list/part-$i.json.partial && mv $b-list/part-$i.json.partial $b-list/part-$i.json" ;;
    pack-*)
      local i=${job#pack-} keep_dir="$b-pack-keep-${job#pack-}" n driver check url
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
      echo "$SHARD_FLAVOR $SHARD_CAP"
      # Resume from the last snapshot; snapshot every SNAP_EVERY s while the fill runs; then the
      # shard's bundle (`build.py collect`) as its final archive.
      local snap="(while sleep $SNAP_EVERY; do tar -cf $part.partial -C $root . 2>/dev/null && mv $part.partial $part && echo \"FILL snapshot \$(du -h $part | cut -f1)\"; done) & SNAP=\$!;"
      echo "$BEAT $(wait_for "$parts" "$OPTIX_PARTS") $spread mkdir -p $root && { [ ! -f $part ] || tar -xf $part -C $root; } && echo \"FILL resumed \$(find $root -name '*.ptx' -o -name '*.cubin' | wc -l) files\"; $snap python -m tools.prebuilt.fill fill --root $root $fill $FILL_ARGS 2>&1 | grep --line-buffered -E '^FILL|Error|Traceback'; rc=\$?; kill \$SNAP; [ \$rc = 0 ] && python -m tools.prebuilt.build collect --root $root --arch $arch --out /tmp/dist --workers 8 2>&1 | grep -E '^(collect|bundle)|Error|Traceback' && $(keep /tmp/dist "$final" raywarp_kernels)" ;;
    merge)
      local shards="" s census_wait="" census_get="true" census_arg="" census_keep="true"
      for s in $(shards "$arch"); do shards="$shards ${s#sm$arch-}"; done
      if [ "$be" = optix ]; then
        census_wait=$(wait_for "$(base)-census-*.tar" "$(censuses | wc -l)")
        census_get="for t in $(base)-census-*.tar; do tar -xf \$t -C /tmp/census || exit 1; done"
        census_arg="--census /tmp/census/sm$arch/*.json"
        census_keep="$(keep /tmp/census/sm$arch "$b-census.tar")"
      fi
      echo "cpu-upgrade $MERGE_CAP"
      echo "$BEAT $(wait_for "$b-final-[pw]*.tar" "$((np + nw))") $census_wait mkdir -p /tmp/s /tmp/census && for s in$shards; do cp $b-final-\$s.tar /tmp/s.tar && mkdir -p /tmp/s/\$s && tar -xf /tmp/s.tar -C /tmp/s/\$s && rm /tmp/s.tar || exit 1; done && $census_get && python -m tools.prebuilt.build merge \$(for s in$shards; do echo --bundle /tmp/s/\$s/raywarp_kernels/sm$arch; done) $census_arg --out /tmp/dist 2>&1 | grep -E '^(bundle|merge)|Error|Traceback' && $(keep /tmp/dist "$b-merged.tar" raywarp_kernels) && { [ ! -d /tmp/dist/raywarp_kernels/sm$arch/ptx ] || $(keep /tmp/dist "$b-merged-ptx.tar" "raywarp_kernels/sm$arch/ptx raywarp_kernels/sm$arch/manifest.json"); } && $census_keep" ;;
    release)
      local n wait="" got="true" pack=""
      n=$(packs "$arch")
      if (( n )); then
        wait=$(wait_for "$b-pack-[0-9]*.tar" "$n")
        got="for i in \$(seq 0 $((n - 1))); do mkdir -p /tmp/p/\$i && cp $b-pack-\$i.tar /tmp/p.tar && tar -xf /tmp/p.tar -C /tmp/p/\$i || exit 1; done && python -m tools.prebuilt.build merge-packs \$(for d in /tmp/p/*/raywarp_optix_modules/sm$arch-drv*; do echo --pack \$d; done) --out /tmp/pack"
        pack="--pack \$(ls -d /tmp/pack/raywarp_optix_modules/sm$arch-drv*)"
      fi
      echo "$(release_flavor "$arch") $RELEASE_CAP"
      echo "$BEAT $wait mkdir -p /tmp/b /tmp/p && cp $b-merged.tar /tmp/b.tar && tar -xf /tmp/b.tar -C /tmp/b && rm /tmp/b.tar && $got && python -m tools.release.kernels layout /tmp/b/raywarp_kernels/sm$arch /tmp/layout $pack --workers 8 && du -sh /tmp/layout && $(keep /tmp/layout "$b-release.tar")" ;;
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
  # uploads the tree: ~1-2 min per wave of 10, so the last of ~100 jobs starts ~15 min after the first)
  local a i
  for a in $ARCHS; do for ((i = 0; i < OPTIX_PARTS; i++)); do echo "sm$a-list-$i"; done; done
  for a in $ARCHS; do shards "$a"; done
  censuses
  for a in $ARCHS; do echo "sm$a-merge"; done
}

wave2() {  # every job of wave 2
  local a i
  for a in $ARCHS; do for ((i = 0; i < $(packs "$a"); i++)); do echo "sm$a-pack-$i"; done; done
  for a in $ARCHS; do echo "sm$a-release"; done
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

start() {  # NAME RUNTAG: launch one job (its run id recorded first)
  local spec flavor cap command
  spec=$(job_spec "$1")
  read -r flavor cap <<<"$(head -1 <<<"$spec")"
  command=$(tail -n +2 <<<"$spec")
  # No compiler cache is read (the fill compiles into its own root, the pack creates into a fresh
  # OptiX cache): none restored (~25 GB CPU, ~37 GB GPU, 1.5-6 min) or saved.
  hf run --flavor "$flavor" --no-cache-restore warp,cuda,wpbr --no-cache-save warp,cuda,wpbr --lane "$LANE" --tag "$2" --timeout "$cap" --stall 30m -- "$command" | tail -1
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

follow() {  # NAME...: until each job ended ok (relaunching a failed one up to RETRIES times); 1: one gave up
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
      if (( tries >= RETRIES )); then echo "$(date -u +%H:%M) $name: $rid failed after $tries relaunches: by hand"; gave_up=1; continue; fi
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

publish() {  # every architecture's release archive here, laid out as one, pinned and published (PREFIX: under that place)
  local scratch a pins
  scratch=$(mktemp -d)
  mkdir -p "$scratch/layout"
  for a in $ARCHS; do
    command $HF_CLI buckets cp "hf://buckets/blanchon/mujoco-warp-pbr/runs/$(day)-$tag-sm$a-release.tar" "$scratch/sm$a.tar"
    tar -xf "$scratch/sm$a.tar" -C "$scratch/layout" && rm "$scratch/sm$a.tar"
  done
  pins=raywarp/core/build/kernel_cache/kernels.json
  [ -z "$PREFIX" ] || pins=$STATE.kernels.json
  $PY -m tools.release.kernels pin "$scratch/layout" --to "$pins" --bucket "$BUCKET"
  $PY -m tools.release.kernels publish "$scratch/layout" --bucket "$BUCKET" ${PREFIX:+--prefix "$PREFIX"}
  echo "$pins written$([ -n "$PREFIX" ] || echo ': commit it')"
  cat "$pins"
  rm -rf "$scratch"
}

case $cmd in
  refill | watch | spec)
    for a in $(optix_archs); do
      url=$(seed_url "$a") || exit 1
      export "SEED_URL_$a=$url"
    done ;;
esac
case $cmd in
  refill)
    mkdir -p "$(dirname "$STATE")"
    [ -f "$STATE" ] || echo "day $(date -u +%F)" > "$STATE"
    touch "$STATE.stage"
    if ! grep -qx wave1 "$STATE.stage"; then
      launch $(wave1)
      follow $(wave1) || { echo "wave 1 incomplete: fix the jobs named above, then rerun"; exit 1; }
      echo wave1 >> "$STATE.stage"
    fi
    if ! grep -qx wave2 "$STATE.stage"; then
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
    [ -f "$STATE" ] || { STATE=$(mktemp) && echo "day $(date -u +%F)" > "$STATE"; }
    job_spec "$3" ;;
  cost)
    [ -f "$STATE" ] || { STATE=$(mktemp) && echo "day $(date -u +%F)" > "$STATE"; }
    cost ;;
  *)
    echo "usage: $0 refill|watch|status|unpublish|spec|cost TAG [JOB...]" >&2
    exit 2 ;;
esac
