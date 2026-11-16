# Data generation over many scenes

The workload: hundreds of scenes with very different configurations (lights, textures, features),
each rendered for episodes with domain randomisation, scene after scene, with the frames saved to
disk. This page gives the recommended loop, what each part costs, and how to measure it on your
own scenes with [`internal/benchmarks/datagen/`](../internal/benchmarks/datagen/run.py).

## The loop

1. **Warm every scene once, ahead of the run** (a separate process, once per machine and
   driver). On a fresh machine, a scene's first frame compiles kernels, creates OptiX modules
   and runs its bakes. `Renderer.warm` stores all of them on disk: call it with the real run's
   constructor arguments (world count, resolution, cameras, per-world declaration and settings;
   [performance](performance.md#first-frame-prebuilt-kernels-caches-and-warm)):

   ```python
   rw.Renderer.warm(scene, nworld=16, resolution=(512, 512), settings=rw.RenderSettings().replace(rw.FAST)).result()
   ```

   Or run the loop below once with one step and saving off (`run.py --steps 1 --sink off`).
   USD and glTF scenes go through their memory-mapped `.wpbr` cache (on by default:
   `rw.formats.usd.load(path)`; `cache=dir` to keep it elsewhere), so later loads skip parsing.

2. **Iterate** over the scenes. For each scene: create one Renderer, run its episodes, then
   `close` it and drop it.
   - Load scene k + 1 **on the host** on a worker thread while scene k renders: map its
     `.wpbr`, or load its compiled MuJoCo model (below).
   - Upload it and create its Renderer on the render thread, or on the worker too (Renderers are
     safe on several threads; their device work then shares the GPU with the frames).
   - Save through a pinned readback ring with a writer thread, so the render loop never waits
     for the disk.
   - No `gc.collect()` between scenes: a closed Renderer and its scene are freed when dropped.
     Call `gc.freeze()` once after the warm-up, so Python's own occasional full collections only
     scan what was made since.

```python
# docs: skip (needs scenes, a GPU and disk; internal/benchmarks/datagen/ is the full, measured version)
import concurrent.futures
import gc

import raywarp as rw

NWORLD, RES = 16, (512, 512)
per_world = rw.PerWorld(material=["base_color", "specular_roughness"], light=["strength", "color"], env=["exposure"])

gc.collect()
gc.freeze()  # start-up objects are permanent: Python's full collections scan only the new ones
pool = concurrent.futures.ThreadPoolExecutor(1)
ahead = pool.submit(rw.Scene.load, wpbr_paths[0])  # host only: no device call on the worker
for i in range(len(wpbr_paths)):
  host = ahead.result()  # usually ready: loaded while the previous scene rendered
  if i + 1 < len(wpbr_paths):
    ahead = pool.submit(rw.Scene.load, wpbr_paths[i + 1])
  r = rw.Renderer(rw.upload(host), nworld=NWORLD, cameras=None, resolution=RES, per_world=per_world)
  for episode in range(EPISODES):
    randomise(r, rng)  # writes r.per_world[...] tables: nothing recompiles
    for t in range(STEPS):
      out = r.render(seed=t)
      ring.submit(out, t)  # pack + async copy to pinned memory; a writer thread saves it
  r.close()
  del r, host  # freed now: no reference cycle holds the scene's device memory
```

`internal/benchmarks/datagen/sinks.py` holds the ring (`Ring`) and the writers. `run.py` is the whole loop,
with the measurements. Copy them.

**Threads.** Several Renderers may be built, warmed, rendered and closed on several threads, beside
your own Warp work. raywarp never changes Warp's process-wide current stream or default device
(Warp's own `array.numpy()` does, for a moment: two of your threads calling it at once can leave
the null stream current for the process; read device arrays back on one thread, or with
`wp.copy(host_array, a, stream=...)`), and a Warp module loads on one thread at a time.

**MuJoCo scenes: cache the compiled model.** The host load of a MuJoCo scene is MuJoCo's: the spec
build (reading the XML and every mesh and texture file) and `MjSpec.compile` (meshes, convex hulls,
inertias). Save each compiled model once with `mujoco.mj_saveModel(model, "scene.mjb")` and load
it with `mujoco.MjModel.from_binary_path("scene.mjb")`: the same model bit for bit, in about the
compile's time, without the spec build and the asset reads (Franka 0.08 s / 40 MB, aloha_clutter
0.47 s / 266 MB, Unitree G1 0.17 s / 80 MB), one file instead of hundreds, which matters on
network storage. A `.mjb` is specific to the MuJoCo version: key the file by `mujoco.__version__`.
raywarp's own conversion (`rw.integrations.mujoco.scene`) takes 0.5-1 s on the CPU for these scenes.

## What a switch costs

RTX PRO 6000, 16 worlds x 512^2, `FAST`, OptiX with the prebuilt kernel bundle, 35
different scenes (internal/benchmarks/datagen `full` set; run `2026-11-10-w9dg-rtx`):

| stage | median | worst | what it is |
| --- | --- | --- | --- |
| host load | 0.02 s | 23.6 s (aloha_clutter; menagerie robots 4.5-8 s) | `.wpbr` map, code build, MJCF compile + meshes: **hidden by the host prefetch** |
| upload | 0.05 s | 5.1 s (free_boxes, 1000 bodies) | `rw.upload` / MuJoCo Warp `put_model` + `rw.integrations.mujoco.scene` |
| Renderer | 0.26 s | 2.6 s (first Renderer of the process) | tables, layout, features, backend |
| first frame, warm disk caches | 0.02-0.36 s | 1.8 s (Office), 15.6 s (free_boxes) | kernel / module loads, bakes from disk, frame 0 |
| first frame, first visit on the machine | 1.9 s | 38 s (Franka: 15 MuJoCo Warp compiles) | OptiX modules not in the bundle (1-9 s each), MuJoCo Warp kernels |
| teardown: `close` | 0.001 s | 0.008 s | releases the build; no longer waits for OptiX modules created ahead |
| teardown: `gc.collect()` | not needed | | a closed scene leaves no Warp array in a reference cycle (CPU check) |

- **Compiles:** zero raywarp kernel compiles over all 35 scenes, cold, thanks to the bundle.
  MuJoCo Warp compiled 45 kernels for its own models (Franka 15, Aloha 10, G1 9, ...). A warm
  start then loads them from Warp's cache.
- **Switch, warm process (disk caches filled by the warm-up):** upload + Renderer + first frame
  took 0.25-0.65 s for the tabletop / feature / procedural scenes and 3.0 s for Office; teardown
  adds ~1 ms (`close`; no collection). The Renderer takes 0.03-0.7 s of it: build it on the
  prefetch worker (below) to take it off the switch.
- **Memory:** device memory after each teardown stays at the high-water mark of the largest
  scene so far (Warp's memory pool keeps freed blocks), with no growth over 35 switches
  (6.3 GB after Bistro, then flat within +-0.2 GB). Live allocations after teardown were
  550-690 MB once Bistro had been loaded, about 50 MB of them the readback ring. That measurement
  still held the previous scene (fixed in `run.py`), and a cache kept each scene's packed mesh
  BVH for the process (fixed: a closed scene leaves no Warp array alive).

With the Renderer built on the prefetch worker too (`run.py --overlap renderer`, run
`2026-11-10-w9sw-rtx`), a warm switch is its first frame: 13 ms for the tabletop, 0.11-0.15 s for
primitives and the Cornell box, 0.47 s for a flex scene, 3.1 s for Office (its first frame).

## Saving

Same machine, Franka (MuJoCo Warp physics, 2 substeps per frame), 16 x 512^2, 64 steps. Images/s
are rendered images (worlds x cameras per second):

| sink | images/s | overhead | size per 512^2 image | notes |
| --- | --- | --- | --- | --- |
| no readback | 574 | - | - | render + physics only |
| readback ring only | 569 | 1 % | - | RGBA8 -> RGB8 pack kernel, async copy to pinned memory |
| **npy** (raw uint8 RGB, one `.npy` per output and episode) | 573 | **0 %** | 768 KB | written straight from the pinned buffer; writer 1.2 GB/s |
| jpeg (Pillow, quality 90, thread pool) | 491 | 10 % | 37 KB | encoding takes the GIL often enough to slow the render thread |
| mp4 (libx264 per world, ffmpeg processes) | 502 | 8 % | 1.7 KB | the render thread blocked 0.15 s at episode boundaries |

**Default: npy.** It is the fastest: there is no encode, so the cost is one pack kernel and
one copy, and the writer thread does a single `write` per frame. The limit is the disk. Raw
512^2 RGB is 0.75 MB per image, so a disk that sustains 1.2 GB/s keeps up with about 1,600
images/s. Above that rate, or when storage matters, record MP4: the video recorder
(`tools/video`, docs/recording.md) encodes on NVENC from device memory, and its
documented cost is a few tens of microseconds of the render thread per frame. JPEG costs about 10 %
of the frame rate and gives files 20 times smaller than raw.

## Reusing a Renderer across scenes

There is no `Renderer.set_scene`. Everything a Renderer builds depends on its scene:

- the camera rig (scene cameras);
- the shared / per-world instance layout;
- the per-world tables, sized by the scene's materials, lights and instances;
- the validation ranges;
- the resolved features, which select the kernels (the tier build);
- the backend's acceleration structures.

A `set_scene` would redo all of that. The only parts that do not depend on the scene are the
output buffers and the compiled kernels, and both are already reused:

- **Output buffers**: allocate them once and bind them to every Renderer with
  `Renderer(output_arrays={"rgb": rgb, ...})` (`run.py --bind`). The readback ring and anything
  else holding the arrays (a torch view, a recorder) keep pointing at the same memory across
  scenes.
- **Kernels**: scenes of the same tier share their compiled modules, both in the process and on
  disk. Scene data is run-time data, so a new scene never recompiles a kernel that another scene
  already compiled.

A captured CUDA graph belongs to one Renderer (it records that scene's pointers). When
replaying graphs, capture again for each scene.

## Measuring your own scene set

```bash
python internal/benchmarks/datagen/run.py --set full --nworld 16 --size 512 --steps 1 --sink off --overlap none --out out/warm   # warm pass
python internal/benchmarks/datagen/run.py --set full --nworld 16 --size 512 --steps 32 --sink npy --bind --cycles 2 --out out/run
python internal/benchmarks/datagen/report.py out/run
```

To add your scenes, add an `Entry` per scene and config to `internal/benchmarks/datagen/catalog.py`. The
`full` set has 75 configurations, interleaved so that any `--limit` prefix spans every group:

- MuJoCo menagerie robots and comparison scenes;
- USD Office, Bistro and the simple room;
- the glTF assets (through MJCF);
- the feature stages, procedural rooms and diagnostic rooms;
- 24 seeded random feature-space scenes;
- config variants: GI on / off, rgb + depth + seg, no DR / full DR, the default settings, lens cameras.

Each scene writes one JSON line to `<out>/scenes.jsonl`:

- the switch, split into teardown (close / gc), prefetch wait, load, upload, Renderer,
  randomisation and first frame;
- Warp module compiles (ours and others, e.g. MuJoCo Warp's) and OptiX module creations;
- images/s with and without saving, and the ring's blocked time;
- device memory after each teardown (the leak check across switches).

`job.sh` is the HF job that produced the numbers above.
