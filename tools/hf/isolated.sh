#!/bin/bash
# The in-job entry of an isolated run (`python -m tools.hf run --isolated`), for third-party code
# that must not reach the bucket: the job mounts its own run directory alone, at
# /data/runs/<run> (`runs.volume`); nothing else of the bucket, no token, no project environment,
# no caches. The command (bash, in the uploaded tree) brings what it needs itself.
#
#   isolated.sh <run>
#
# Command environment: HF_RUN, HF_SRC (the tree), HF_OUT (= /data/runs/<run>/out: written in place,
# no mirror), HF_INPUTS (the run's inputs/: `--input NAME=PATH`), HF_TIMEOUT_S, HF_ISOLATED=1.
# The command's output goes to run.log in the run directory as it runs; status.json (exit, reason
# ok / failed / timeout, commit, gpu, wall_s, stages_s.run) is written last.
set -u
RID=${1:?usage: isolated.sh <run>}
RUN=${HF_DATA:-/data}/runs/$RID  # HF_DATA / HF_WORK: the tests' roots, as entry.sh
export HF_RUN=$RID HF_SRC=${HF_WORK:-/work}/src HF_OUT=$RUN/out HF_INPUTS=$RUN/inputs HF_ISOLATED=1
mkdir -p "$HF_OUT"
CMD=$(printf %s "${HF_CMD_B64:?}" | base64 -d)
unset HF_CMD_B64
START=$(date +%s)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
cd "$HF_SRC" || exit 1
# The cap leaves 90 s for status.json before HF's own timeout.
timeout --kill-after=30 $(( ${HF_TIMEOUT_S:-1800} - 90 )) bash -c "$CMD" 2>&1 | tee "$RUN/run.log"
rc=${PIPESTATUS[0]}
case $rc in 0) reason=ok;; 124|137) reason=timeout;; *) reason=failed;; esac
wall=$(( $(date +%s) - START ))
printf '{"run": "%s", "exit": %d, "reason": "%s", "commit": "%s", "gpu": "%s", "isolated": true, "wall_s": %d, "stages_s": {"run": %d}}\n' \
  "$RID" "$rc" "$reason" "$(cat "$HF_SRC/COMMIT" 2>/dev/null)" "${GPU:-cpu}" "$wall" "$wall" > "$RUN/status.json"
echo "isolated run $RID: exit $rc ($reason), $wall s"
exit "$rc"
