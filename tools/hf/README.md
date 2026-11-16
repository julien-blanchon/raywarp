# HF compute and bucket: `python -m tools.hf`

The one way to run work on Hugging Face Jobs, manage the bucket caches and prepared scenes, and
track spend (the kernel-build pipeline launches its jobs with it: tools/prebuilt). Developer
tooling outside the library (standard library + the `hf` CLI, logged in on the machine). The
maintainers' development repository extends this CLI with dev sandboxes, test runs and the
preparation of its benchmark scenes; those parts are not in the public repository. Run it from the
worktree whose HEAD you want to ship:

```sh
PYTHONPATH=. python -m tools.hf --help
```

## Rules

- **One job per question**, batched, from a **committed** tree (the run uploads `HEAD`; uncommitted
  edits are not shipped and a warning says so). Dry-run on `cpu-upgrade` first.
- **Flavors:** `cpu-upgrade` ($0.03/h: CPU suites, scene preparation, compile census), `l4x1`
  ($0.80/h: the default GPU: correctness, diagnostics, image diffs, relative A/B, dev sandbox),
  `rtx-pro-6000` ($2.75/h: headline numbers, RTX/OptiX-sensitive checks), `a100-large` ($2.50/h:
  the Warp path without RT cores), `cpu-performance` ($1.90/h, 32 vCPU: the full CPU suite),
  and two L4 overflows when no L4 is free: `l40sx1` (sm_89, the L4's warm
  cache family) and `a10g-large` (sm_86, a colder family of its own). Nothing else is accepted
  (`config.FLAVORS`).
- Every job has a **hard timeout** (`--timeout`, the HF job cap) and a **stall watchdog** (`--stall`:
  the command is stopped when its output stops moving; `--wait` also cancels a job whose log stops
  during bootstrap / finish). `python -m tools.hf ps` before launching; never leave a sandbox idle.
- Every job is labelled `project=raywarp run=<id> lane=<lane>` (lane: the branch name, or
  `--lane`), so the **spend tracker** sees it; a lane over its budget cannot launch.

## Runs

```sh
python -m tools.hf run --tag fix-env-cpu --flavor cpu-upgrade --timeout 30m --wait -- \
  'python -m pytest -q -n 8 tests/render/test_envmap.py'
python -m tools.hf run --tag fix-env-l4 --flavor l4x1 --timeout 20m --wait -- \
  'python -m pytest -q -n 2 tests/optix_backend && python -m benchmarks.run --scene franka --png $HF_OUT/franka.png'
python -m tools.hf ps [-a]            # running (all) raywarp jobs + sandboxes, minutes and $
python -m tools.hf logs <run> [-f]    # <run> = <date>-<tag> or a job id
python -m tools.hf status <run>       # status.json
python -m tools.hf wait <run>         # follow, auto-kill on a stall, fetch
python -m tools.hf fetch <run>        # runs/<run>/ -> a local results directory (--to; no code)
python -m tools.hf kill <run>...
```

What a run does:
1. `git archive HEAD` of the worktree (without the development repository's results and notes) → `runs/<YYYY-MM-DD>-<tag>/code.tgz`
   + `code.sha256` + `cmd.json`; `--source NAME=REF` adds `sources/NAME.tgz` (that ref's committed
   tree, same exclusions, extracted to `$HF_SOURCES/NAME/` with `COMMIT`: A/B trees).
2. The command reaches the job base64-encoded (`HF_CMD_B64`; HF parses `--env` values as dotenv lines,
   which cut at a newline or a ` #`), so multi-line commands, quotes and `#` need no workaround:
   `python -m tools.hf run --tag t -- "$(cat script.sh)"` runs the script as written.
3. `hf jobs run` on the pinned image with the bucket at `/data`; the job verifies the sha256 and runs
   [`entry.sh`](entry.sh) `job`.
4. `entry.sh`: **bootstrap** (system packages from the `apt` cache family, the pinned uv, `uv sync --locked` of the repo with the
   `dev` + `bench` groups and the `usd` + `mujoco` extras into `/work/venv`, the tree installed editable (its native build hook bundles the
   MDL SDK and OptiX headers from the pinned archives in `pins/native/`; `--source` trees get the same
   `raywarp/_native`); on GPUs **with RT cores** a check that the OptiX traversal can run, fatal on failure),
   **cache restore**, then the command (bash, in
   the uploaded tree, venv first on `PATH`) under `timeout` and the stall watchdog. Every 60 s, and at
   the end, small files of `$HF_OUT` (< 2 MB each), `run.log` and `stages.txt` are mirrored to the run
   directory.
5. **Finish** (also after a failure, timeout, stall or SIGTERM): final mirror → `evidence.tgz` (all of
   `$HF_OUT` when ≤ 200 MB) → cache save (≤ 180 s) → **`status.json` last** → exit (billing stops).

Command environment: `HF_OUT` (results: mirrored), `HF_SRC` (the tree), `HF_PY` (the env's python),
`HF_DEADLINE` (the epoch second the command is stopped at: a command keeps time for its last step),
`HF_SOURCES`, `HF_ARCH` (`cpu`, `sm89`, `sm120`, ...), `HF_RT` (1: RT cores), `MJPBR_ASSET_DIR=/data/assets`,
`MJPBR_HOST_CACHE=/work/raywarp_cache` (prepared scenes staged to local disk), the cache variables below,
`MUJOCO_GL` (`egl` on GPUs, `osmesa` on CPU).

`status.json`: `exit`, `reason` (`ok`, `failed`, `timeout`, `stalled`, `terminated`, `bootstrap-failed`,
`no-time`), `commit`, `gpu`, `compute_capability`, `rt_cores`, `wall_s`, `stages_s` (bootstrap,
packages (only when the image lacks some), restore-uv, environment, restore-caches, run, mirror, evidence, cache-save), `restored`, `saved`.

**GPU detection is by capability**, not by name: compute capability from `nvidia-smi`; RT cores on
every GPU of capability ≥ 7.5 except the compute-only GV100 / GA100 / GH100 / GB100 / GB300 (7.0, 8.0,
9.0, 10.0, 10.3). RT cores ⇒ OptiX must work.

**Environments** (`--env`): `project` (default: this repository's lock) and `mjlab` (the pinned mjlab +
HyperLux plugin competitor environment from `pins/hyperlux-office-v1`, with this tree installed in it;
HyperLux pinned to `HYPERLUX_VERSION`). `--with fillcuda` adds the kernel fill's NVRTC / nvJitLink wheels (`FILL_CUDA_PACKAGES`: every Warp
shard of tools/prebuilt/fill_all.sh). `--with ovrtx` adds Omniverse RTX (ovrtx/ovstage pins + its
shader cache).

**Pins**: [`pins.env`](pins.env) is the only place versions and bucket paths of pinned inputs are
written (image, Python, uv binary, ovrtx, mjlab), and the bucket paths of the native build hook's inputs
(`NATIVE_ARCHIVES`, `NATIVE_CACHE`; its versions, URLs and sha256 live in `tools/native/hook.py`).

### Isolated runs (third-party code): `run --isolated`

For code we do not trust (a competitor installed from its repository): the job mounts **its own run
directory alone**, at the same path (`hf://buckets/.../runs/<run>` -> `/data/runs/<run>`), and nothing
else of the bucket. No token is passed to the job; there is no project environment, no cache restore
or save; the entry is [`isolated.sh`](isolated.sh), not `entry.sh`.

```sh
python -m tools.hf run --tag alloy-x --flavor rtx-pro-6000 --isolated --timeout 20m \
  --input scenes=/path/to/scenes --input uv=/path/to/uv --wait -- 'bash competitor/isolated_job.sh'
```

- Inputs: `--input NAME=PATH` (a local file or directory, repeatable) ships in the run directory as
  `inputs/NAME` (`$HF_INPUTS/NAME`), next to the committed tree (`$HF_SRC`). `--input` needs `--isolated`;
  `--isolated` refuses `--needs-prepared`, `--source`, `--with`, `--env` and `--no-cache-save` (none of them
  is available without the bucket).
- Outputs: `$HF_OUT` is `runs/<run>/out` itself (written in place, no mirror); the command's output is
  `run.log` as it runs; `status.json` (exit, reason, commit, gpu, `isolated`, wall time) is written last, so
  `wait`, `fetch` (which skips `inputs/`) and `status` work as for a normal run.
- The command brings its own tools (system packages, uv from `--input`, the third-party package). Run
  the untrusted part as an unprivileged user with an empty environment, and copy its results into
  `$HF_OUT` from the job's own (root) process after it exits: it can then write nothing but its run
  directory, and nothing it writes there is taken as ours unless we copy it.

## Bucket layout (`hf://buckets/blanchon/mujoco-warp-pbr`, mounted at `/data`)

| Path | Content |
|---|---|
| `assets/` | scene sources only (isaac51, showcase, omniverse, unreal, mjcf, menagerie, mujoco, mujoco_warp, hdri) |
| `prepared/v<format>-c<converter>-l<libraries>/` | prepared `.rayw` scenes + sidecar JSON + `mdl/` bakes; key = `.rayw` VERSION + `raywarp.formats.usd.version.CONVERTER` + digest of the job environment's `LIBRARIES` versions |
| `refs/<renderer>-<version>/` | immutable reference renders |
| `wheels/` | pinned wheels only |
| `pins/` | pinned environments (`hyperlux-office-v1`: mjlab, plugin bundle, OptiX headers, uv); `native/`: the native build hook's artifacts (`python tools/native/hook.py fetch`) |
| `sdk/` | native build caches (ccache; `native-hook/`: the build hook's compiled MDL bindings by build key), MDL corpus |
| `cache/<key>/<family>.<digest>.tar` | warm caches, newest 2 per family (below) |
| `regression/` | regression baselines (newest tag) |
| `runs/<date>-<tag>/` | one per run: code, `cmd.json`, `run.log`, `stages.txt`, `out/`, `evidence.tgz`, `status.json`; deleted after 14 days unless `runs/<id>/keep` exists |
| `spend/` | `ledger.jsonl` (snapshot of HF's job records), `budgets.json` |

Keep the mount at `/data`: a prepared `.rayw` records its sources as absolute `/data/assets/...` paths.

## Caches

`cache/<key>/<family>.<listing digest>.tar`, key `apt-<digest of the image and the package list>` for
the `apt` family (the system packages' indexes and `.deb` files: a job installs them offline, and only
a miss reaches the Ubuntu mirror, each step retried with backoff and timed in the log as `apt <step>:`),
`<env>-uv` for the uv cache (packages + Python) and
`<env>-<arch>-warp<version>` for the families `warp`, `cuda` (GPUs), `raywarp` (GPUs: raywarp's cache
directory, `RAYWARP_CACHE_DIR`: texture levels, bakes, MDL conversions and OptiX's module cache,
which raywarp keeps in its `optix/`), `hyperlux` (mjlab), `ovrtx` (`--with ovrtx`). A job restores the
newest 2 snapshots of each family (uv: the newest only) into ONE directory set (`UV_CACHE_DIR`,
`WARP_KERNEL_CACHE_DIR`, `CUDA_CACHE_PATH`, `RAYWARP_CACHE_DIR`, `HYPERLUX_CACHE_DIR` under
`/work/cache/`) and saves one new snapshot per family that grew: a tar streamed to `<name>.partial`
on the bucket → rename, so a killed job never leaves a truncated archive (no local copy: a second
copy of the 16 GB CPU `warp` family filled fill shards' disks). `run --no-cache-save warp,cuda,raywarp`
skips families whose growth is scratch (fill, listing and census jobs); `run --no-cache-restore
warp,cuda,raywarp` does not restore families the job does not read (a fill compiles into its own root,
a pack into a fresh OptiX cache; the CPU `warp` family is ~25 GB, 1.5-6 min), and does not save them either.
`run --mount OWNER/NAME:PATH` mounts another storage bucket read-write at PATH (the kernel fill's build
cache at `/cache`: tools/prebuilt/build_cache.py; the job gets no token, the mount is the access). `status.json` is written
before the save (`"saved": {"all": "pending"}`) and again after it; every `STAGE` line logs the disk use. `raywarp` never saves `.rayw`
scenes (they live in `prepared/`), a changed code-digest memo alone makes no snapshot, and above
`HF_RAYWARP_CACHE_MB` (6 GB) the oldest files (modification time) are dropped before the save.

```sh
python -m tools.hf cache show                 # per key/family: snapshots, size, newest; legacy files
python -m tools.hf cache verify               # stale partials, extra snapshots, tar headers (range reads)
python -m tools.hf cache prune --dry-run      # newest 2 per family; partials > 2 h; runs > 14 days
```

## Prepared scenes

Prepared `.rayw` scenes live in the bucket's `prepared/v<VERSION>-c<CONVERTER>-l<libraries>/`, a
directory named from the importer's code and the installed library versions, so a converter or
library change points at a new, empty directory instead of a stale file, and two environments never
share a file. `MJPBR_PREPARED_DIR` overrides it. Jobs stage the `.rayw` to local disk before
memory-mapping it. Building and checking them is done by the development repository's tooling.

## Spend

```sh
python -m tools.hf spend show [--lane L] [--rows]   # per lane: jobs, minutes, $, budget (since the plan start)
python -m tools.hf spend budget w1-2 1.50           # coordinator: set a lane budget
python -m tools.hf spend sync                       # rewrite spend/ledger.jsonl (also after every --wait run)
```

$ = HF's billed running minutes × the flavor price of `hf jobs hardware`. A launch warns past 80 % of
the lane budget (spent + the cap of the lane's running jobs + this job's worst case = timeout × price)
and refuses past 100 % (`--over-budget` overrides; write the reason in your lane file). HF keeps no end
time for cancelled jobs: `tools.hf kill` / `sandbox down` / the sandbox watchdog record one in
`spend/cancels/<job>`; a job cancelled elsewhere shows as "unknown duration" (not counted, flagged).
