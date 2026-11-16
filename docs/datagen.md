# Data generation over many scenes

Render hundreds of different scenes one after another, randomise each for a few episodes, and
save the frames to disk without slowing the render loop. Costs are measured on 35 scenes.

## Quick start

One Renderer per scene: create it, render its episodes, save, close it, move on.

```python
import numpy as np

import raywarp as rw


def room(color: tuple[float, float, float]) -> rw.Scene:
  """A tiny host scene; a real generator loads one (rw.Scene.load("scene.rayw"), rw.formats.usd.load(path))."""
  b = rw.SceneBuilder()
  paint = b.add_material(rw.Material(base_color=color))
  b.add_instance("box", material=paint, size=(2.0, 0.05, 1.2), xform=np.c_[np.eye(3), [0.0, 2.0, 1.2]], path="wall")
  b.add_instance("box", material=paint, size=(0.3, 0.3, 0.3), xform=np.c_[np.eye(3), [0.3, 0.5, 0.3]], path="crate")
  b.add_light(rw.Light.point((0.0, 0.0, 2.2), intensity=80.0, radius=0.05))
  return b.build()


eye = np.tile(np.array([0.0, -1.9, 1.3, 0.45, 0.0, 0.0, 0.893], np.float32), (4, 1, 1))  # position, quaternion
for i, host in enumerate([room((0.8, 0.2, 0.2)), room((0.2, 0.4, 0.8))]):
  r = rw.Renderer(rw.upload(host), nworld=4, cameras=[rw.CameraSpec(fovy=70.0)], resolution=(32, 24))
  r.camera_pose.assign(eye)
  for t in range(2):
    out = r.render(seed=t)
    np.save(f"scene{i}_rgb_{t:06d}.npy", out.rgb.numpy()[..., :3])  # (nworld, ncam, H, W, 3) uint8
  r.close()
```

`.numpy()` waits for the GPU. A real generator saves through a readback ring instead
([below](#saving)).

## Switch between many scenes

1. **Warm every scene once, ahead of the run** (once per machine and driver). The first frame of
   a scene loads its GPU code and runs its one-time precomputations. `Renderer.warm` stores them
   on disk. Call it with the real run's arguments:

   ```python
   # docs: skip (needs a scene and a GPU)
   rw.Renderer.warm(scene, nworld=16, resolution=(512, 512), settings=rw.RenderSettings().replace(rw.FAST)).result()
   ```

   USD and glTF scenes are kept as `.rayw` files (on by default), so later loads skip parsing.

2. **Iterate** over the scenes:
   - Load scene k + 1 on the host, on a worker thread, while scene k renders.
   - Upload it and create its Renderer on the render thread, or on the worker too.
   - Save through a readback ring with a writer thread, so the render loop never waits for disk.
   - `close` each Renderer and drop it. No `gc.collect()` is needed between scenes.
   - Call `gc.freeze()` once after the warm-up, so Python's full collections skip start-up objects.

```python
# docs: skip (needs scenes, a GPU and disk)
import concurrent.futures
import gc

import raywarp as rw

NWORLD, RES = 16, (512, 512)
per_world = rw.PerWorld(material=["base_color", "specular_roughness"], light=["strength", "color"], env=["exposure"])

gc.collect()
gc.freeze()  # start-up objects are permanent: Python's full collections scan only the new ones
pool = concurrent.futures.ThreadPoolExecutor(1)
ahead = pool.submit(rw.Scene.load, rayw_paths[0])  # host only: no device call on the worker
for i in range(len(rayw_paths)):
  host = ahead.result()  # usually ready: loaded while the previous scene rendered
  if i + 1 < len(rayw_paths):
    ahead = pool.submit(rw.Scene.load, rayw_paths[i + 1])
  r = rw.Renderer(rw.upload(host), nworld=NWORLD, cameras=None, resolution=RES, per_world=per_world)
  for episode in range(EPISODES):
    randomise(r, rng)  # writes r.per_world[...] tables: nothing recompiles
    for t in range(STEPS):
      out = r.render(seed=t)
      ring.submit(out, t)  # pack + async copy to pinned memory; a writer thread saves it
  r.close()
  del r, host  # freed now
```

`ring` is your readback ring ([saving](#saving)); `randomise` writes the per-world tables
([domain randomisation](domain_randomisation.md)).

**Threads.** Renderers may be built, warmed, rendered and closed on several threads. Raywarp
never changes Warp's current stream or default device. Warp's own `array.numpy()` does, briefly:
read device arrays back on one thread, or with `wp.copy(host_array, a, stream=...)`.

**MuJoCo scenes: cache the compiled model.** A MuJoCo scene's host load is MuJoCo's: reading the
XML and assets, then compiling. Save each compiled model once with
`mujoco.mj_saveModel(model, "scene.mjb")` and load it with `mujoco.MjModel.from_binary_path("scene.mjb")`.
It is the same model, without the asset reads, in one file (Franka 0.08 s, aloha_clutter 0.47 s,
Unitree G1 0.17 s). A `.mjb` is specific to the MuJoCo version: key the file by
`mujoco.__version__`.

## What a switch costs

RTX PRO 6000, 16 worlds x 512², `FAST`, OptiX with the prebuilt kernels, 35 different scenes
(MuJoCo Menagerie robots, USD interiors, glTF assets, procedural scenes):

| stage | median | worst | what it is |
| --- | --- | --- | --- |
| host load | 0.02 s | 23.6 s (aloha_clutter; Menagerie robots 4.5-8 s) | `.rayw` map or MJCF compile: **hidden by loading ahead** |
| upload | 0.05 s | 5.1 s (free_boxes, 1000 bodies) | `rw.upload`, or `put_model` + `rw.integrations.mujoco.scene` |
| Renderer | 0.26 s | 2.6 s (first Renderer of the process) | tables, layout, acceleration structures |
| first frame, warm disk caches | 0.02-0.36 s | 1.8 s (Office), 15.6 s (free_boxes) | loads from disk, frame 0 |
| first frame, first visit on the machine | 1.9 s | 38 s (Franka: 15 MuJoCo Warp compiles) | OptiX setup (1-9 s each), MuJoCo Warp's own kernels |
| `close` | 0.001 s | 0.008 s | |

<!-- SHOWCASE: datagen/scene_switch.webp (docs/assets/SHOTS.md) -->
[![Scene after scene: Franka, the showcase room, Sponza, fixtures](assets/datagen/scene_switch.webp)](assets/datagen/scene_switch.mp4)

- **Compiles:** zero Raywarp compiles over all 35 scenes, cold. MuJoCo Warp compiled 45 kernels
  for its own models; a warm start loads them from Warp's cache.
- **Warm switch:** upload + Renderer + first frame took 0.25-0.65 s for tabletop and procedural
  scenes, 3.0 s for Office. With the Renderer built on the loading thread too, a switch is just
  its first frame: 13 ms for the tabletop, 3.1 s for Office.
- **Memory:** device memory stays at the high-water mark of the largest scene so far (Warp's
  memory pool keeps freed blocks). It did not grow over 35 switches (6.3 GB after Bistro, then
  flat within 0.2 GB).

<a id="saving"></a>

## Save frames without stalling the render loop

Same machine, Franka (MuJoCo Warp physics, 2 substeps per frame), 16 x 512², 64 steps. Images/s
counts worlds x cameras per second:

| sink | images/s | overhead | size per 512² image | notes |
| --- | --- | --- | --- | --- |
| no readback | 574 | - | - | render + physics only |
| readback ring only | 569 | 1 % | - | RGBA to RGB pack, async copy to pinned memory |
| **npy** (raw uint8 RGB) | 573 | **0 %** | 768 KB | written straight from the pinned buffer; writer 1.2 GB/s |
| jpeg (Pillow, quality 90, thread pool) | 491 | 10 % | 37 KB | encoding holds the GIL often enough to slow the render thread |
| mp4 (libx264 per world, ffmpeg processes) | 502 | 8 % | 1.7 KB | the render thread blocked 0.15 s at episode ends |

A readback ring is a few pinned host buffers. Asynchronous copies fill them on the render stream;
a writer thread empties them once each copy is done. The render thread waits only when every
slot is still being written:

```python
# docs: skip (needs a GPU, a renderer `r` and disk)
import queue
import threading

import numpy as np
import warp as wp

SLOTS = 4
slots = [wp.empty(r.outputs.rgb.shape, dtype=wp.vec4ub, device="cpu", pinned=True) for _ in range(SLOTS)]
free, full = queue.Queue(), queue.Queue()
for i in range(SLOTS):
  free.put(i)


def writer() -> None:
  while (item := full.get()) is not None:
    i, event, path = item
    wp.synchronize_event(event)  # this copy is done; the render stream ran on
    np.save(path, slots[i].numpy()[..., :3])  # (nworld, ncam, H, W, 3) uint8
    free.put(i)


thread = threading.Thread(target=writer, daemon=True)
thread.start()
for t in range(STEPS):
  out = r.render(sim)
  i = free.get()  # blocks only when every slot is still being written
  wp.copy(slots[i], out.rgb)  # asynchronous: device -> pinned host
  full.put((i, wp.record_event(), f"rgb_{t:06d}.npy"))
full.put(None)
thread.join()
```

Which format:

- **npy** is the fastest: no encoding, one copy, one `write` per frame. The disk is the limit:
  raw 512² RGB is 0.75 MB, so a 1.2 GB/s disk keeps up with about 1,600 images/s.
- **MP4** when storage matters: the [video recorder](recording.md) encodes on the GPU from device
  memory.
- **JPEG** costs about 10 % of the frame rate for files 20 times smaller than raw.

Record each frame's cameras with `r.camera_info()` ([ground truth](randomization.md#ground-truth)).

<a id="reusing-a-renderer-across-scenes"></a>

## Reuse output buffers across scenes

There is no `Renderer.set_scene`: a Renderer's cameras, tables and acceleration structures all
depend on its scene. What does not depend on the scene is already reused:

- **Output buffers:** allocate them once and pass them to every Renderer with
  `Renderer(output_arrays={"rgb": rgb, ...})`. A readback ring, a torch view or a recorder keeps
  pointing at the same memory across scenes.
- **GPU code:** scenes share it, in the process and on disk. A new scene never recompiles what
  another scene already loaded.

A captured CUDA graph belongs to one Renderer: capture again for each scene.

## See also

- [Dataset generation: every knob and ground truth](randomization.md): what to vary and export.
- [Domain randomisation](domain_randomisation.md): writing the per-world tables.
- [Recording video](recording.md): MP4 at render speed.
- [Performance and kernels](performance.md#how-the-kernels-work): what is prebuilt, the caches.
