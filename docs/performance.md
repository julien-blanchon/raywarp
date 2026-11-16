# Performance

One `render` call shades the requested (world, camera, pixel) batch with the prebuilt kernel build
its settings select ([Compiled in or runtime](settings.md#compiled-in-or-runtime)). A complete call also includes preparation, derived tables, tracing and
temporal/output work as needed; it is not necessarily one kernel launch. Measure the complete
call as well as individual GPU stages when deciding what to optimize.

**Goal:** faster than every other renderer in every comparable setup, and significantly faster at
large batch (512-1024 px, hundreds to thousands of views per launch). Compare like with like:
matched features and native resolution, renderer time measured separately from physics,
randomisation, bindings and policy work, and startup and peak memory reported as their own
dimensions. Preserve hardware, resolved settings, scene / camera, batch / resolution, backend,
cache state and timing method when comparing runs (see [Measuring](#measuring)).

The numbers on this page are recorded measurements: each names its commit and configuration, and none is
a fresh measurement of the current tree. Settings, code and dependencies have changed since; do
not extrapolate an old speedup or startup time to today's setup.

## Batching

- **Put everything in one renderer.** `nworld x ncam x H x W` pixels per launch; throughput
  saturates from about 1-4 M pixels per launch (e.g. 256 worlds at 64^2), below that the
  launch overhead dominates.
- **Render at the policy's resolution.** Pixels cost linearly once the GPU is full; a 64^2
  observation is 16x cheaper than 256^2 resized down.
- **Static scenery is shared.** Instances that the simulator never moves are stored and traced
  once for every world (one shared BVH / OptiX instance tree); only the moving parts are per
  world (`r.layout` says what was shared). Per-world appearance tables keep this sharing;
  per-world poses, sizes or meshes take an instance out of it ([robotics](robotics.md)).
- **Identical images.** `performance.render_order="world"` maps the threads world-major, faster
  when every world renders nearly the same image (fixed cameras on mostly static scenes).

## Max-speed checklist

1. **The cheapest settings that show what the policy needs**: geometry-only `Outputs` (depth /
   seg) -> the unlit recipe -> `rw.FAST` -> the default ([settings](settings.md#fast-field-set)).
   Turn off what the task does not see (`reflections.enabled`: -29 % on Office, `shadows.enabled`,
   `transparency.enabled`).
2. **OptiX on RT-core GPUs**: `performance.traversal="auto"` (the default) uses OptiX on GPUs
   with RT cores: 3-7x faster on mesh scenes (DamagedHelmet, the default, 194k -> 626k, below),
   1.7-1.9x on primitive-only scenes (L4, 512 worlds x 512^2).
3. **CUDA graphs are automatic**: on a CUDA device `r.render(sim)` records its frame as a CUDA
   graph on the second frame whose inputs repeat (the same simulator arrays, seed array, settings
   and bound tables) and replays it afterwards; writes in place (simulator state, `camera_pose`,
   per-world tables, integer seeds) keep the graph, a few alternating simulator buffers get one
   graph each, and anything else runs one eager frame and records again. This matters at small
   batch, where an eager frame is launch-bound (viewers, video, data generation). To fuse the
   simulator step into the same graph, `graph = r.capture(lambda: mjw.step(m, d), sim=d)`, then
   `graph.launch(d)` every step ([robotics](robotics.md#step-and-render-in-one-cuda-graph)).
4. **Render only the worlds you need this step**: `r.render(d, active=ids)` with a `(k,)` int32
   device array of world ids (`-1`: a hole), written in place (also inside a graph); the frame
   time follows the listed worlds, the others keep their last outputs
   ([robotics](robotics.md#selected-worlds-only)).
5. **Write straight into the policy's tensor**: `rw.settings.ColorOutput(channels=3, planar=True)` (uint8)
   or `ColorOutput(channels=3, planar=True, dtype="float16")` writes `(nworld * ncam, 3, H, W)`
   channels-first in the kernel; `r.bind_outputs(rgb=tensor_as_warp_array)` makes it render into
   your buffer. Free on the
   RTX PRO 6000 (-0.3..+1.9 %, `internal/results/checkpoint/gpu4-r1/report.md`).
6. **Never read back per frame**: `out.torch()` / `wp.to_torch` views, no `.numpy()`.
   Recording a video too? Encode from device memory (NVENC) or read back NV12 into pinned buffers
   on a side stream: [recording video](recording.md).
7. **Warm up** before training (`Renderer.warm`, below), so the first step does not compile.
8. **Big scenes**: bake the `.wpbr` once (`rw.formats.usd.bake`), keep it on a local disk.

## OptiX

OptiX needs an NVIDIA driver serving OptiX 9.0 (R580 or newer; OptiX ships in the driver):
raywarp calls it through its own ctypes binding (`core/trace/optix/api.py`), and installing raywarp
fetches the OptiX SDK headers the programs compile against. `traversal="auto"` uses
OptiX on GPUs with RT cores and the software BVH elsewhere (CPU, A100 / H100) and for scenes
with dim-1 flex (segments); height fields, SDFs and flex surfaces are OptiX triangle structures
(refitted as they change); both render the same pixels (within a few edge pixels). There is no
fallback: `traversal="optix"` where OptiX cannot run or cannot trace the scene raises, and so does
"auto" on an RT-core GPU whose driver lacks OptiX 9.0. OptiX compiles its programs on the first
frame of a configuration, in parallel tasks on every CPU core (cached on disk after);
the OptiX register cap (an internal constant per tier family, `raywarp/core/pipeline/platform/tuning.py`
`OPTIX_REGISTERS`: 128 for the default, 96 for `FAST`) trades occupancy for spills (the Office is 2.3x faster
at 128 than at 64; light scenes 10-22 % faster at 64). `internal/results/rt_backend.md` lists what the
backend supports.

## First frame: prebuilt kernels, caches and warm

**Cold and warm.** A renderer's first frame does, once per configuration, what later frames
skip: it fetches its kernel build, loads its Warp modules, creates the OptiX modules its frames
bind (absent from the driver's module pack), uploads the scene (textures converted once, then
cached) and runs the bakes (light grid, GI probes, environment prefilter). With the prebuilt
bundle a cold first frame compiles nothing; what is left is downloads, module loads and creation,
upload and bakes. A **warm** start finds all of it in the disk caches below and starts at once:
fill them ahead with `Renderer.warm` (below), in this process or an earlier one.

**Prebuilt kernels.** The kernels are not compiled on your machine: a bundle per GPU architecture
(`performance.kernels="prebuilt"`, the default) holds every tier's Warp modules, the OptiX
programs' PTX and, per NVIDIA driver release, a pack of ready OptiX modules (one small file per
module, fetched when that module is first created). Nothing to install:
raywarp downloads only what a renderer uses, on first use (megabytes for a scene's first frame),
from the public Hugging Face bucket the release's pins name (`blanchon/raywarp-kernels`; plain
HTTPS, set `HF_TOKEN` for a higher rate limit), checks every
file against the hashes pinned in the installed package and keeps it in a local store
(`$RAYWARP_KERNELS`, default `<cache dir>/kernels`). The first renderer on a GPU picks its
architecture's bundle (sm_80, 86, 89, 90 and 120 are pinned; another architecture compiles) and the
running driver's module pack. `RAYWARP_KERNELS_URL` points at a mirror (`file://` works). For a
machine without network, fill a store on one that has it and copy it over:

```python
# docs: skip (downloads the bundle of an architecture)
import raywarp as rw

rw.kernels.fetch(120)  # the SM version (120: RTX PRO 6000 / RTX 50); no GPU needed (`raywarp kernels fetch --arch 120`)
# then copy $RAYWARP_KERNELS (default <cache dir>/kernels) to the offline machine, and there:
rw.kernels.configure(download=False)  # or RAYWARP_KERNELS_DOWNLOAD=0: the local store only
```

A file that cannot be fetched warns once and compiles instead (cached). Where compiling must
never happen (a fleet of training jobs), `rw.kernels.configure(compile=False)` (or
`RAYWARP_KERNELS_COMPILE=0`) makes a configuration that is not prebuilt raise
`rw.kernels.KernelsNotPrebuilt`, naming it, instead of compiling ([install](install.md#prebuilt-kernels)).

Three disk caches make the second run start fast:

- **kernels and OptiX programs**: Warp's kernel cache (`~/.cache/warp`), keyed by the generated
  code, so one entry per distinct `Features`; the OptiX programs' PTX in `$RAYWARP_CACHE_DIR/ptx`
  and OptiX's modules in `$RAYWARP_CACHE_DIR/optix`;
- **textures**: `$RAYWARP_CACHE_DIR/textures`, every texture's stored levels keyed by its content
  and an entry per texture keyed by its every input (source images, channels, color spaces,
  wraps, storage, the code), so a repeat upload reads the levels instead of converting, packing
  and filtering the images;
- **bakes** (light grid, GI probes, environment prefilter): `$RAYWARP_CACHE_DIR/bakes`
  (default `~/.cache/raywarp`; an empty value turns them off), keyed by the content of
  everything they read; a later build of the same scene loads them bit for bit.

They grow with every scene, setting and code version seen (a development machine reached 18 GB)
and are never pruned on their own. Bound them with a prune, deleting the least recently used
entries until they fit (a deleted entry is rebuilt on its next use):

```bash
raywarp cache prune --max-gb 20   # or raywarp.cache.prune.prune(max_bytes); `raywarp cache info`: the sizes
```

It covers the bakes, textures, PTX, emissive fits, glTF conversions and the `.wpbr` scenes in
`$RAYWARP_CACHE_DIR`; OptiX's module cache bounds itself, an old release's prebuilt kernels
(`$RAYWARP_CACHE_DIR/kernels/sm<arch>/<content>`) and Warp's kernel cache
(`warp.clear_kernel_cache()`) are deleted by hand.

Measured before the prebuilt bundle, so the cold column includes compiles the bundle now removes (RTX PRO 6000,
`internal/results/checkpoint/gpu4-r1/report.md`, OptiX; cold = empty caches):

| scene | cold first frame | warm first frame | where the cold time goes |
| --- | ---: | ---: | --- |
| DamagedHelmet | 30.9 s | 2.55 s | OptiX PTX compile 15.5 s (the light-grid bake waits for it) |
| Office (Isaac) | 65.6 s | 15.1 s | warm: host load 11.4 s + upload 5.1 s from a cached `.wpbr` |

Fill them ahead of a run with `Renderer.warm`: the constructor's arguments (one word changed
from the run's setup) plus the state its first frame renders. It returns a
`concurrent.futures.Future` at once and does in the background what that configuration's first
frame would: the Warp modules load (compiled when missing), the OptiX modules the frame binds are
created into OptiX's disk cache, the structures are built and the bakes stored. A renderer of the
same configuration then starts at once, in this process or a later one. The scene's own
preparation is the setup's usual code: `raywarp.formats.usd.load(path)` converts a stage into the `.wpbr`
cache once (on by default; a second load maps it: 0.4-0.6 s for an Isaac prop that converts in
3-9 s), `upload` fills the texture cache.

```python
import mujoco
import mujoco_warp as mjw
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco

mjm = mujoco.MjModel.from_xml_string("""
<mujoco><worldbody>
  <light pos="0 0 2" dir="0 0 -1"/>
  <geom type="plane" size="2 2 0.1"/><geom type="box" pos="0 0 0.2" size="0.2 0.2 0.2"/>
  <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
</worldbody></mujoco>""")
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=4)
mjw.forward(m, d)
scene = rw.integrations.mujoco.scene(mjm, m)
# Every configuration the run will use, warmed together.
warms = [
  rw.Renderer.warm(scene, nworld=4, resolution=(64, 64), settings=s, sim=d)
  for s in (rw.RenderSettings().replace(rw.FAST), rw.RenderSettings())
]
for w in warms:
  print(w.result())  # e.g. "warm 3.20 s: renderer 2.90 s (+2 cached, OptiX 2 compiled / 0 cached), frame 0.30 s (+3 cached)"
r = rw.Renderer(scene, nworld=4, resolution=(64, 64), settings=rw.RenderSettings())  # starts warm
```

Warm at the run's configuration: its world count (the bakes trace world 0, but a GI bake's world
groups depend on the worlds), its cameras and resolution and its `per_world` declaration (they
decide what is compiled in and what is shared). Many warms run at once: their CPU work (OptiX
module creation, Warp module loads) overlaps, their device steps take turns per device, and a
warm identical to one still running shares its future. Each result lists its steps' seconds and
the cache entries they added: an already warm configuration adds none. For an event loop,
`await asyncio.wrap_future(future)`.

The first `render` of a configuration that was not warmed waits for its compiles; `Renderer(...)`
starts them in the background, so scene setup and simulation overlap them
(`Renderer.wait_compiles` waits for them without device work).

## Zero compiles

What is guaranteed (OptiX, `performance.kernels="prebuilt"`, the default):

- **No NVRTC, PTX or Warp compile for any single setting or scene change.** The default, `FAST`,
  unlit and each recipe of [settings](settings.md#recipes), with every single value of every
  compile-time setting (each choice, the integers around each threshold the code compares them
  to) and of every scene property the kernel resolution reads (capabilities, lights, environment,
  deformables, material inputs, per-world declarations, sharing layouts) routes to a prebuilt
  tier build (127 PTX builds in the bundle), except a scene with deformables on OptiX: it takes
  its own build, compiled on first use (the tier programs miss deformable hits, a known issue;
  `Renderer.explain()` names it). The test suite proves it on the CPU through the real
  resolution (`tests/unit/test_zero_compile.py`: 1 236 points, `tools/prebuilt/zero.py`); the
  pairs of changes are walked too (`python -m tools.prebuilt.zero walk 2`, sharded on a CPU job:
  124 008 points, 30 holes besides the lens cameras, which no longer compile), their remaining holes listed in
  `internal/research/kb/lanes/W9-HOLES.md` and `internal/research/kb/lanes/W9-PRESET-IMPL.md`. Alpha cut-outs on
  any geometry kind (meshes, boxes, spheres, capsules, ellipsoids, cylinders, planes) share one
  PTX: the alpha test runs in each kind's hit programs and the plane queries (the 24 pair holes
  of cut-outs on analytic shapes are closed; `test_zero_compile.py` routes every recipe there).
- **No OptiX module created at frame time** once a configuration is warm: `Renderer.warm(scene,
  settings, ...)` creates, in parallel and from bundled or cached PTX only, exactly the modules
  (render and GI programs, each with its bound words) the configuration's frames bind, and builds
  their pipelines in its frame. A later renderer of that configuration (any process) finds them in
  OptiX's disk cache (`tests/unit/test_census_words.py`: the warm's modules are the frame's).
- **Settings one change away**: `Renderer.warm(..., nearby=64)` also creates the modules of every
  setting one change from the warmed one (the same domain the test suite walks, `raywarp/core/build/variants/nearby.py`),
  at most that many, after its frame; `Renderer.nearby_modules()` shows the plan first (its
  count and estimated seconds; 16-46 modules, ~5-60 s on a 192-core RTX host for the default and
  `FAST` on a robot scene). A later `update_settings` to any of them binds a cached module.
- **Per-driver module packs**: the bundle's census records every module a census scene's
  configuration binds (both halves of the build word and the output word, so a pack built from
  it is hit; MJCF scenes also as `raywarp.integrations.mujoco.scene` loads them with their own lights only,
  the showcase room also with its per-world poses); the running driver's pack holds one file per
  module, fetched and merged into OptiX's disk cache when the module is first created, so a census
  scene's first frame creates no module on a fresh machine either and downloads only its own. A
  scene outside the census creates its modules once per machine and driver (seconds;
  `Renderer.warm` does it ahead).

How settings that used to compile are served now: a light cap, shadow budget or texture tap count
below the tier's by the tier itself (a tier build reads `lighting.max_exact`,
`reflections.max_exact`, `shadows.max_casters` and `surfaces.anisotropy` at run time within its
compiled values, behind a word bit its own counts leave off: `FAST` with `max_casters=1`, the
default with `max_exact=2` or `anisotropy=4`), a shadow budget below the default's exact lights by its `casters`
variant (a 3-caster budget loop); larger caps and budgets (up to 16), `shadows.samples` and
`surfaces.anisotropy` by the counted and sampled tier variants, which read them at run time up to
their capacities (slower frames: 16 ranking and budget slots, run-time sampling); `transparency.max_layers` and the OptiX supersampling
factor (`aa.spp`) are run-time data; the grading runs in the deferred display pass; the
reflection hits' sun shadows are a word bit, as are the translucent layers
(`transparency.enabled=False` on a translucent scene keeps its tier build) and, on OptiX, the live
material UV mappings of declared material selection; unlit rendering has its own 2 x 2 and
anisotropic variants.

**The documented exception: lens cameras** (OpenCV distortion, fisheye, ray bundles).
A lens camera takes the pinhole camera's prebuilt tier build plus its model's render program, a
module of its own linked into the build's pipeline. The bundle holds the fast tier's OpenCV
distortion / fisheye programs (SLAM and visual odometry); the other lens programs are left out
(`tools/prebuilt/fill.py fill --lens` adds every model's): the first lens renderer of such a
build compiles its model's program once (NVRTC, measured 1-9 min on the
CPU of a job: about the build's own compile) and creates its module (seconds),
both cached afterwards. `Renderer.warm(scene, settings)` with the lens camera does this ahead of
the first frame. Pinhole renderers compile and create exactly what they did. On the software path
each camera model is a megakernel of its own, compiled once likewise. **Not included in the prebuilt kernels** (they compile once on first use, or ahead in
`Renderer.warm`, then are cached): the motion-vector output (`outputs.motion`) beyond the fast
tier; the sampled-shadow variants (`shadows.samples` > 1 without the counted recipes' other
values, on `FAST` or the default); two variants of the default tier: reflection-hit sun shadows
(`reflections.shadows="sun"` with the default's other settings) and a shadow budget below the
default's (`shadows.max_casters` < 4; light caps and budgets above it are the counted variant's,
prebuilt with the RTX look); 2 x 2 supersampling outside the quality
recipe (`aa.spp=2` on `FAST`, the default or unlit); and unlit shading with anisotropic filtering
(`surfaces.shading="unlit"` with `surfaces.anisotropy` > 1, e.g. on the default). A first use
compiles their programs on the CPU (NVRTC, about 10-25 min for the default tier's builds) and creates their OptiX modules (seconds to minutes). The default,
`FAST`, unlit and every documented recipe (quality, reference, RTX look) stay fully prebuilt
(`tools/prebuilt/fill.py fill --niche` adds them to a bundle). The counters build (a profiling
build, below) and `performance.kernels="scene"` compile by design.

## Running on the CPU

The renderer runs unchanged on Warp's CPU device, with images within a few 1/255 of the GPU's
(`internal/benchmarks/regression.py check --device cuda:0` against the CPU references). A Warp CPU launch
runs on one core; the renderer splits its render launch (the megakernel, in 4 bands of rays per
core) and the deferred display over every core of the process's affinity, so one renderer uses
the whole machine and the images are bit-identical to one launch. Several renderers in one
process share those cores (pin worker processes with `taskset` to split them).

```python
with wp.ScopedDevice("cpu"):
  d = mjw.make_data(mjm, nworld=4)
  mjw.forward(m, d)
  r = rw.Renderer(rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm)), nworld=4, resolution=(32, 32))
  r.render(d)
```

Measured against other CPU renderers (32-vCPU EPYC 7R13, Franka in a small room, 512^2, images
per second): raywarp FAST 22.0 (1 world) to 26.2 (4 worlds), default 21.0 to 24.9 (one core
before the bands: 1.7 / 1.6), MuJoCo's OpenGL renderer through OSMesa 7.7 (shadow maps on),
PyBullet's TinyRenderer 3.9, MuJoCo Warp's renderer on the CPU 3.3
(`internal/benchmarks/specs/w14_kernel/cpu.sh`, `internal/research/kb/lanes/W14-KERNEL.md`). The first CPU frame
compiles the CPU kernels once (~25 s wall on 32 cores).

## Developing kernel code

A change to kernel code invalidates the prebuilt builds that use it, so the next first frame
compiles. Compile only what you test: `performance.kernels="scene"` builds the scene's own,
specialised kernels instead of a tier's superset, at the same frame time. On an RTX PRO 6000 host
(fresh caches; Warp's helper modules already compiled):

| configuration | tier (`"prebuilt"`) | own build (`"scene"`) |
|---|---:|---:|
| Franka, FAST | 67 s | 13 s |
| Franka, default | 573 s | 31 s |
| showcase room, default | - | 195 s |

For feature-dense scenes, Warp's own optimisation level is the one further switch:
`wp.config.optimization_level = 2` before the first render compiles the big builds 2-5x faster
(showcase room 195 s -> 70 s) but renders 4-20x slower, so use it for correctness work only. It
is part of the kernels' cache keys: those builds never mix with the default ones. OptiX's own
optimisation level changes nothing measurable.

## Sensor effects

The `sensor` group's effects run as a pass after the display, only while one is on. Measured on an
RTX PRO 6000, Franka FAST at 512 worlds x 512^2 (a 19.4 ms frame): shot / read noise +3.5 ms,
range depth noise +1.2 ms, stereo depth +2.7 ms, noise + ADC + response + vignette +4.4 ms.

## Multi-GPU

One renderer per GPU, each with its own scene upload and its own simulator worlds (a process or
a thread per device): nothing is shared across devices.

## Against a rasteriser

On one RTX PRO 6000, at 512^2-1024^2 and 1-256 cameras, MuJoCo's OpenGL rasteriser reaches
0.03-0.58x the camera-frames/s of our `FAST` on robot scenes, and several processes
do not scale. Why we ray-trace, and where a rasteriser would still win:
[FAQ](faq.md#why-ray-tracing-and-not-rasterisation-or-path-tracing).

## Measuring

```bash
python internal/benchmarks/run.py run --spec internal/benchmarks/specs/smoke.json --out internal/results/smoke   # throughput, cold / warm
python internal/benchmarks/warmstart.py --scenes damaged_helmet --runs cold,warm --out internal/results/start   # start-up stages
```

For a Renderer created under `raywarp.core.pipeline.platform.tuning.override(counters=True)` (an internal debug
build), `r.counters()` gives per-pixel means of what the frame did
(samples, layers, candidate lights / cheap ranking entries, and shadow/reflection rays).
`lights` and `hit_lights` are candidates **before** evaluator rejection, not BRDF/LTC execution
counts. `light_terms` / `hit_light_terms` count nonzero unshadowed contributions;
`light_casters` / `hit_light_casters` count their cast-shadow eligibility, before sun-only
selection and budgeting. `hit_*` covers glossy secondary hits (MuJoCo mirrors included: their
reflection is the glossy ray). Capped diffuse remainder, ambient/environment and headlight terms
are outside these contribution counts. Use counters for attribution and a separate disabled
run for timing. All counting code compiles out when disabled; the existing always-retained
counter table uses int64 totals: 13 columns x 8 bytes = 104 bytes/world (52 KiB at 512 worlds).
This includes storage with counters disabled. Large pixel/sample/light products can exceed
32-bit totals; counter-enabled runs are diagnostic and have no throughput claim.

## Recorded numbers at policy sizes (512^2 and 1024^2)

16 cameras per launch (16 worlds x 1 camera; the G1: 8 worlds x 2 cameras), OptiX, CUDA graph,
ms per launch (all 16 images); code of `core-refactor` 0272b41, 2026-10-28
(`internal/results/history/post_release/preset_512/README.md`), under the presets of the time: "default"
is the then `balanced` preset, which still had coarse probe GI, `aces_rtx` and adaptive edge AA 2
(since removed); today's default uses probe GI resampled into the ambient cube, `pbr_neutral` and
light grid "auto". "unlit" was the `fastest` preset, `FAST` the `training_fast` preset. The
historical Office measurement attributed 18 % additional frame time to GI at 256^2
(`internal/results/history/benchmark_matrix/`); it does not establish the current default's net performance.

| scene | size | GPU | depth + seg | unlit | `FAST` | default | quality |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Office (Isaac) | 512^2 | RTX PRO 6000 | 0.29 | 0.45 | 2.4 | 12.4 | 68 |
| Office (Isaac) | 1024^2 | RTX PRO 6000 | 0.85 | 1.4 | 8.8 | 41 | 254 |
| Hospital (Isaac) | 512^2 | RTX PRO 6000 | 0.30 | 0.46 | 2.2 | 12.7 | 71 |
| Hospital (Isaac) | 1024^2 | RTX PRO 6000 | 0.90 | 1.4 | 7.8 | 43 | 263 |
| aloha (MuJoCo Warp) | 512^2 | RTX PRO 6000 | 0.26 | 0.34 | 0.92 | 2.4 | 6.4 |
| aloha (MuJoCo Warp) | 1024^2 | RTX PRO 6000 | 0.74 | 1.0 | 3.2 | 7.1 | 23 |
| G1 (Newton) | 512^2 | RTX PRO 6000 | 0.35 | 0.37 | 0.45 | 1.3 | 2.9 |
| G1 (Newton) | 1024^2 | RTX PRO 6000 | 0.89 | 0.96 | 1.3 | 4.0 | 10.7 |
| Office (Isaac) | 512^2 | L4 | 0.86 | 1.4 | 11.5 | 52 | 341 |
| Office (Isaac) | 1024^2 | L4 | 3.9 | 6.2 | 48 | 196 | 1341 |
| Hospital (Isaac) | 1024^2 | L4 | 4.0 | 6.3 | 46 | 206 | 1486 |
| aloha (MuJoCo Warp) | 1024^2 | L4 | 3.5 | 5.3 | 18 | 36 | 124 |
| G1 (Newton) | 1024^2 | L4 | 3.5 | 3.9 | 5.9 | 19 | 58 |

Renderer memory (Warp mempool growth, per camera): 1-3 MB at 512^2 and 4-9 MB at 1024^2 for
depth + seg / unlit / `FAST` (historical, before its GI); the historical default 11-15 / 31-34 MB
(26 MB of it at 1024^2 the edge buffers of the since removed adaptive AA), quality 20-24 / 69-75
MB; these historical memory figures do not describe the current default.

**Against Omniverse RTX** (the G1 scene of `internal/benchmarks/rtx_batched.py`, 8 worlds x 2 cameras,
RTX PRO 6000, ms per step of the 16 images): RTX RealTimePathTracing 15.2 (512^2) / 41.2
(1024^2), its Minimal mode 2.4 / 8.3; ours `FAST` 0.45 / 1.27, the default 1.32 / 4.04,
quality 2.95 / 10.7: the default is 10-12x RTX's path tracer, `FAST` 5-7x its Minimal
mode.

## Recorded numbers at 64^2-256^2 (an earlier checkpoint)

RTX PRO 6000 Blackwell, an earlier version of this code (`internal/results/checkpoint/gpu4-r1/report.md`,
`internal/results/history/m6/README.md`); camera-frames per second, one camera per world:

| scene | worlds x resolution | settings | software BVH | OptiX |
| --- | --- | --- | ---: | ---: |
| DamagedHelmet | 4096 x 64^2 | `FAST` | 309k | 1.29M |
| DamagedHelmet | 4096 x 64^2 | default | 194k | 626k |
| ANYmal C | 4096 x 64^2 | default | | 1.13M |
| DamagedHelmet | 1024 x 128^2 | RTX look (GI; measured with 64-frame accumulation, since removed) | | 88.6k |
| Office (Isaac, 16 x 256^2) | 16 x 256^2 | default | | 4.5k (10.7 ms / frame) |
| Hospital (Isaac, 16 x 256^2) | 16 x 256^2 | default | | 9.5k (5.1 ms / frame) |

The default rows predate its coarse GI (`internal/results/history/phase_a/decisions/`): on an L4
GI costs DamagedHelmet 4096 x 64^2 19 % (147k -> 119k frames/s) and Office / Sponza 960 x 540
+20-22 % per frame; the `FAST` row is unchanged.

On an L4 (M5 fly-through, 960 x 540, OptiX, `internal/results/history/m5/summary/README.md`):
Office / Warehouse / Hospital, the default, 8.8 / 6.9 / 4.7 ms per frame, `FAST`
2.0 / 1.5 / 1.3 ms.
