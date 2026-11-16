# Concepts

Seven ideas cover how raywarp works: the scene, the renderer, worlds, cameras, settings,
per-world randomisation and the prebuilt kernels. Each section links to the page with the details.

## Scene

A scene exists in two forms:

- **`rw.Scene`**, the host scene: format-neutral geometry, materials (`rw.Material`, an OpenPBR
  Surface), lights (`rw.Light`), cameras and an environment, in NumPy arrays. USD, glTF and Newton
  loaders produce one (`rw.formats.usd.load`, `rw.formats.gltf.load`,
  `rw.integrations.newton.load`); `rw.SceneBuilder` builds one by hand. It can be inspected,
  edited (`host.with_lighting(...)`) and saved as a memory-mapped `.rayw` file that loads again
  in milliseconds.
- **`rw.GpuScene`**, the device scene the kernels read, made by `rw.upload(host)`. Every adapter
  has a one-call shortcut that loads and uploads: `rw.integrations.mujoco.scene(mjm, m)`,
  `rw.formats.usd.scene(path)`, `rw.formats.gltf.scene(path)`, `rw.integrations.newton.scene(model)`.

The MuJoCo Warp adapter is the exception to the two steps: it aliases MuJoCo Warp's model arrays
(no copy) and reads the poses from the `mjw.Data` passed to each render. Newton scenes are bound
to the model's bodies and read `state.body_q`. A loader reports what it dropped or approximated
(`gpu.report`). [Scenes](scenes.md) covers lights, environment maps, materials, textures and
cameras.

## Renderer

`rw.Renderer(scene, nworld=..., cameras=..., resolution=..., settings=..., per_world=...)` renders
`nworld` worlds x `ncam` cameras of one scene per `render` call, into device arrays it owns:

- `r.render(sim)` takes the simulator state (`mjw.Data`, a Newton `State`, or nothing for a
  static scene), refits the acceleration structure and returns `r.outputs`, the same arrays every
  frame, written in place. Each output is world-major: `(nworld, ncam, H, W)`.
- It never reads back to the host on its own; `out.torch()` gives zero-copy torch views.
- `r.capture(step, sim)` records the simulator step and the render in one CUDA graph.
- `r.explain()` prints what the settings resolve to on this scene and the frame's cost class.
- `r.close()` releases it.

A Renderer belongs to one scene. Switching scenes means a new Renderer
([data generation](datagen.md#reusing-a-renderer-across-scenes)).

## Worlds

A world is one copy of the scene: one environment of a vectorised simulator. All worlds render
in one launch. What does not move and is not randomised (static scenery) is stored and traced
once for every world, and only what differs between worlds gets its own copy, so memory and
build time grow with what varies, not with `nworld`.

<!-- SHOWCASE: concepts/worlds.webp (docs/assets/SHOTS.md) -->
![Worlds share static scenery; each has its own state and look](assets/concepts/worlds.webp)

`r.render(sim, active=ids)` renders only the listed worlds (a `(k,)` int32 device array, `-1`
for holes): the frame time follows the number of active worlds, and the others keep their last
image ([robotics](robotics.md#selected-worlds-only)).

## Cameras

`cameras=` names the scene's cameras (MJCF camera names, USD camera prims; `None`: every scene
camera) or gives `rw.CameraSpec` objects: free cameras posed per world through
`r.camera_pose`, or cameras that follow a scene camera (`mount=`, with an optional offset; a
camera on a robot's wrist moves with it). `resolution=(width, height)` is every camera's unless a
`CameraSpec` sets its own. Projections: pinhole, orthographic, fisheye, OpenCV and
Kannala-Brandt distortion (`CameraSpec.from_opencv(...)`, `CameraSpec.from_opencv_fisheye(...)`)
and per-pixel ray bundles. `r.camera_info()` exports each world's intrinsics and pose in OpenCV
conventions ([scenes](scenes.md#cameras)).

## Settings

`rw.RenderSettings` is a frozen dataclass of groups (`outputs`, `aa`, `shadows`, `lighting`,
`reflections`, `gi`, `exposure`, `sensor`, ...). A field is addressed as `"group.field"`:

```python
# docs: skip (an overview; the settings page runs these)
s = rw.RenderSettings()  # the default: PBR, probe GI, LTC area lights, glossy reflections
fast = rw.RenderSettings().replace(rw.FAST)  # maximum training throughput, still PBR
quality = rw.RenderSettings().replace({"aa.spp": 2, "shadows.samples": 2, "shadows.max_casters": 16})
s = s.replace({"outputs": rw.Outputs.of("rgb", "depth", "seg"), "exposure.mode": "manual"})
r.update_settings(**{"exposure.ev100": 12.0})  # on a live renderer
```

Three starting points:

| recipe | use it for |
| --- | --- |
| `rw.FAST` | training: point lights, 2 shadow casters, no reflection rays |
| the default, `RenderSettings()` | training that needs reflections and soft lighting, evaluation |
| quality (2 x 2 samples per pixel, soft shadows, more lights exact) | datasets, demos, high-resolution policies |

<!-- SHOWCASE: concepts/recipes.webp (docs/assets/SHOTS.md) -->
![FAST, default and quality on one scene](assets/concepts/recipes.webp)

Every recipe is deterministic per frame. The quality recipe's full field list, the unlit,
reference and RTX-look recipes and what each group costs: [settings](settings.md#recipes).

## Per-world randomisation

Declare what varies per world with `rw.PerWorld(...)` when building the Renderer, then write its
tables between frames:

```python
# docs: skip (an overview; the domain randomisation page runs these)
per_world = rw.PerWorld(material=["base_color", "specular_roughness"], light=["strength"], env=["exposure"])
r = rw.Renderer(scene, nworld=64, cameras=["front"], resolution=(512, 512), per_world=per_world)
r.per_world["material.base_color"].assign(colors)  # (nworld, nmaterial, 4): one row per world
r.render(d)  # world w reads row w: nothing recompiles
```

The declaration picks the kernel variant once; every later write is data: no compile, no host
round trip, and a captured CUDA graph keeps replaying. Materials, textures, lights, the
environment, exposure, fog, cameras, lens distortion, visibility, poses and meshes of static
objects can vary ([domain randomisation](domain_randomisation.md),
[randomising everything](randomization.md)). MuJoCo Warp's batched model tables (`m.geom_rgba`,
`m.light_diffuse`, ...) are read per world every frame as well.

## Kernels and first use

The renderer is a few megakernels (one per settings tier: default, fast, unlit, and their
variants). They are compiled ahead for each GPU architecture (sm_80, 86, 89, 90, 120). The first
Renderer on a machine downloads only the modules it uses from a public Hugging Face bucket,
checks them against hashes pinned in the package and keeps them in a local store
(`~/.cache/raywarp/kernels`). Scene data, settings values and per-world tables are run-time data:
a new scene, a recipe change or a randomised value compiles nothing. The documented exceptions
(lens-camera programs, the motion output on some tiers, a few rare variants) compile once and
are cached.

A cold first frame still creates OptiX modules (once per machine and driver), uploads the scene
and runs its bakes; `rw.Renderer.warm(...)` does all of it ahead, in the background, and a warm
start finds it in the disk caches. Offline machines and mirrors: [install](install.md#prebuilt-kernels),
[performance](performance.md#first-frame-prebuilt-kernels-caches-and-warm).

**Traversal.** On GPUs with RT cores the rays are traced with OptiX (bundled; it needs an NVIDIA
driver serving OptiX 9.0, R580 or newer). On GPUs without RT cores (A100, H100) and on the CPU
they use raywarp's own Warp BVH. There is no silent fallback: where OptiX cannot run on an
RT-core GPU, the renderer raises; `performance.traversal="software"` picks the Warp BVH
explicitly ([performance](performance.md#optix)).
