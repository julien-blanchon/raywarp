# Quickstart

Every source becomes a device scene with one call: `rw.integrations.mujoco.scene(mjm, m)`,
`rw.formats.usd.scene(path)`, `rw.formats.gltf.scene(path)`, `rw.integrations.newton.scene(model)`. A `rw.Renderer` then
renders `nworld` worlds x `ncam` cameras of it per call into device arrays. mjlab and Isaac Lab
users pick raywarp as their camera sensors' renderer ([mjlab](#mjlab), [Isaac Lab](#isaac-lab)).
Every snippet on this page runs as is on the CPU device (it is how `tests/tooling/test_docs.py` checks
them); on a GPU machine the same code runs on `cuda:0`, Warp's default device.

```bash
pip install "raywarp[mujoco]"   # Python >= 3.12; the core needs only warp-lang 1.18 and numpy
pip install "raywarp[newton]"   # Newton scenes and the SensorCamera drop-in
pip install "raywarp[usd]"      # USD stages (usd-core, pillow, materialx, openexr)
pip install "raywarp[gltf]"     # glTF scenes (the usd extra and pygltflib)
pip install "raywarp[all]"      # every simulator, format and the viewer
```

With Newton's SolverMuJoCo, install Newton's own `[sim]` extra beside it
(`pip install "raywarp[mujoco,newton]" "newton[sim]"`): it pins the MuJoCo / MuJoCo Warp pair the
solver supports (3.15 on Newton 1.7; MuJoCo Warp 3.17 dropped the `Model.stat` it reads). The
renderer works with either pair.

## MuJoCo Warp

The MuJoCo adapter aliases MuJoCo Warp's arrays: no copy at construction, none per frame. The
renderer reads the poses of the `mjw.Data` you pass to `render`, and the geom, material, light
and camera tables of the `mjw.Model` on every frame (write `m.geom_rgba` and the next frame
shows it).

```python
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco
import raywarp.formats.usd

XML = """
<mujoco>
  <worldbody>
    <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
    <light pos="0.5 -0.5 1.5" intensity="0" bulbradius="0.1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="ball" pos="0 0 0.4">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.2" rgba="0.8 0.3 0.2 1"/>
    </body>
    <geom name="crate" type="box" pos="0.6 0.2 0.2" size="0.2 0.2 0.2" rgba="0.2 0.4 0.9 1"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
    <camera name="side" pos="2 0 1" xyaxes="0 1 0 -0.4 0 0.9" fovy="60"/>
  </worldbody>
</mujoco>
"""
mjm = mujoco.MjModel.from_xml_string(XML)
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=8)

scene = rw.integrations.mujoco.scene(mjm, m)  # GpuScene: geoms, materials, lights, cameras (aliased)
r = rw.Renderer(
  scene,
  nworld=8,
  cameras=["front", "side"],  # MJCF camera names (or rw.CameraSpec); one name: cameras="front"
  resolution=(64, 48),  # (width, height) of every camera
  settings=rw.RenderSettings(outputs=rw.Outputs.of("rgb", "depth", "seg")),
)
mjw.step(m, d)
out = r.render(d)  # refits the BVH itself; syncs the host only once, on the first frame (its check)
assert out.rgb.shape == (8, 2, 48, 64)  # (nworld, ncam, H, W) vec4ub: RGBA bytes
rgba = out.rgb.numpy()  # (8, 2, 48, 64, 4) uint8 (a host copy: for looking, not for training)
depth = out.depth.numpy()  # float32 planar depth, 0 where nothing is hit
seg = out.seg.numpy()  # int32 geom id, -1 where nothing is hit
assert {0, 1, 2} <= set(np.unique(seg))
```

`render` returns `r.outputs`, the same arrays every frame (written in place). For training,
read them on the device: `out.torch()` gives zero-copy torch views ([robotics recipes](robotics.md)).

Already using MuJoCo Warp's own renderer? `rw.integrations.mujoco` has the same API: switch the import.

```python
from raywarp.integrations import mujoco as render  # instead of: import mujoco_warp as render

rc = render.create_render_context(mjm, nworld=8, cam_res=(64, 48), render_rgb=True, render_depth=True)
render.render(m, d, rc)
rgb = wp.zeros((8, 48, 64), dtype=wp.vec3)  # MuJoCo Warp's per-camera output arrays
render.get_rgb(rc, 0, rgb)
```

`create_render_context` takes MuJoCo Warp's arguments (`use_shadows`, `use_textures`,
`samples_per_pixel`, ...) plus `settings=` for anything else.

## USD stages and glTF

`rw.formats.usd.scene(path)` reads a USD stage into a format-neutral host scene, keeps it as a
memory-mapped `.wpbr` file and uploads it (`rw.formats.gltf.scene(path)` does the same for glTF 2.0).

```python
from pathlib import Path

Path("demo.usda").write_text("""#usda 1.0
(
  upAxis = "Z"
  metersPerUnit = 1
)
def Xform "World"
{
  def Cube "floor"
  {
    double size = 2
    float3 xformOp:scale = (3, 3, 0.02)
    double3 xformOp:translate = (0, 0, -0.02)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    color3f[] primvars:displayColor = [(0.7, 0.7, 0.7)]
  }
  def Sphere "ball"
  {
    double radius = 0.3
    double3 xformOp:translate = (0, 0, 0.3)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    color3f[] primvars:displayColor = [(0.8, 0.3, 0.2)]
  }
  def DistantLight "sun"
  {
    float inputs:intensity = 50000  # full sun [lux] at a sunny-day exposure (exposure.ev100 ~15; the default meters automatically)
    bool inputs:normalize = 1  # UsdLux: a normalized distant light's intensity is lux
    float inputs:angle = 1
    float3 xformOp:rotateXYZ = (30, 20, 0)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
  }
  def Camera "camera"
  {
    double3 xformOp:translate = (0, -3, 1.2)
    float3 xformOp:rotateXYZ = (70, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
  }
}
""")
gpu = rw.formats.usd.scene("demo.usda", cache="wpbr_cache")  # parsed once, then a memory-mapped .wpbr
print(gpu.report)  # what was dropped or approximated, counts, timings (upload's as gpu.*)
r = rw.Renderer(gpu, nworld=1, cameras=None, resolution=(64, 48))  # None: the stage's cameras
out = r.render()  # a static scene: no simulator state
assert out.rgb.numpy()[..., :3].mean() > 10
```

The step before the upload, `host = rw.formats.usd.load(path)`, returns the host scene (`rw.Scene`) to
inspect or edit (`host.with_lighting(...)`) before `rw.upload(host)`. A loader keeps the scene
it converts in a `.wpbr` file (`cache=True`, the default: the cache
directory, `~/.cache/raywarp` or `$RAYWARP_CACHE_DIR`), so a second load of an unchanged source
maps it in a fraction of a second. `cache=` a directory or a `.wpbr` path keeps it there,
`cache=None` converts every time, and `RAYWARP_CACHE_DIR=""` turns every cache off. A default
directory that cannot be written (a read-only home) warns and loads without the cache.

`rw.formats.usd.settings(stage)` maps the stage's Omniverse render settings onto a `RenderSettings`
([settings](settings.md#usd-stages)). Big stages: bake the `.wpbr` once, offline, with
`rw.formats.usd.bake(stage, "office.wpbr")`; later loads map it in a fraction of a second of host work.

## Newton

`rw.integrations.newton.scene(model)` is the device scene of a Newton model's visible shapes (plus the
lights, environment and headlight you pass as `lighting=`), bound to its bodies: `render(state)`
writes the instance poses from `state.body_q` with one kernel (no host copy), so swapping
`state_0` / `state_1` needs nothing. Shapes whose
`shape_color` differs between worlds show each world's color, re-read from `model.shape_color`
on every render (`rw.integrations.newton.Options(world_colors=False)`: world 0's everywhere).

```python
import newton

import raywarp.integrations.newton

world = newton.ModelBuilder()
body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
builder = newton.ModelBuilder()
builder.add_ground_plane()
builder.replicate(world, 4)  # 4 worlds
model = builder.finalize()
state = model.state()

scene = rw.integrations.newton.scene(model)
# The same in steps: host = rw.integrations.newton.load(model) (a host rw.Scene, cacheable with host.save),
# then rw.upload(host, model.device, binding=rw.integrations.newton.binding(host, model)). The stage the
# model was loaded from: host = rw.integrations.newton.load_usd(model, "robot.usd"), bound by
# rw.integrations.newton.binding(host, model, stage="robot.usd").
r = rw.Renderer(scene, nworld=model.world_count, cameras=[rw.CameraSpec(fovy=60)], resolution=(64, 48))
eye = wp.transformf((0.0, -3.0, 1.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.25))
r.camera_pose.fill_(eye)  # (nworld, ncam) poses of the free cameras, -z forward, +y up
out = r.render(state)
```

### SensorCamera: Newton's camera sensor, PBR shaded

`raywarp.integrations.newton.sensor.SensorCamera` has the constructor and `update()` of
`newton.sensors.SensorCamera`: one import changes.

```python
from raywarp.integrations.newton.sensor import SensorCamera  # instead of: from newton.sensors import SensorCamera

camera = SensorCamera(model)  # plus PBR options: lighting=rw.Lighting(...), **{"aa.spp": 2}
rays = SensorCamera.compute_camera_rays_pinhole(64, 48, camera_fov=1.0)  # or the OpenCV / fisheye helpers
views = wp.array([eye] * model.world_count, dtype=wp.transformf)  # one camera transform per world
color = camera.allocate_image_color(model.world_count, 64, 48)
depth = camera.allocate_image_depth(model.world_count, 64, 48)
camera.update(state, views, rays, color_image=color, depth_image=depth)  # Newton's update()
```

It supports Newton 1.7 and main: one view per camera transform, `camera_rays` from any of
Newton's `compute_camera_rays_*` helpers, `world_indices` with `WorldRenderFlag`, `ClearData`,
`RenderConfig` (mapped to `RenderSettings`), the `allocate_image_*` helpers,
`create_default_light(color=)`, `set_ambient_light(sky, ground)` and
`assign_checkerboard_material`; lights and ambient follow `model.up_axis`, shapes of opacity 0
are not rendered. Extra keyword-only options: `lighting=rw.Lighting(...)` (lights,
ambient, environment, headlight: [Lighting a scene](scenes.md#lighting-a-scene)) and any dotted
`RenderSettings` field (they take precedence over the `RenderConfig` mapping). `color_image` is
PBR shaded and tone mapped; the geometry outputs (depth, shape index, normal, albedo) match
Newton's.

Only the images passed to `update()` are rendered, straight into your arrays. On CUDA,
`graph = camera.capture(lambda: camera.update(...))` captures `update()` (and anything else in
the callable, e.g. physics) in a CUDA graph: `graph.launch()`. Differences from Newton:
cloth and soft-body surfaces render from `state.particle_q` (cloth thickened by the particle
radius), loose particles and Gaussians are not rendered, worlds may hold different shapes (a
different object per world: the scene pools them, each world shows its own), and without
`create_default_light` or `lighting` a key and fill light are used.

### ViewerRaywarpHeadless: Newton's viewer API, ray traced, no window

`ViewerRaywarpHeadless` is a Newton `ViewerNull` (same calls) that ray traces its frames with no
window and no OpenGL: Newton examples written against the viewer API render offscreen.

```python
# docs: requires pyglet
from raywarp.integrations.newton.headless import ViewerRaywarpHeadless

viewer = ViewerRaywarpHeadless(width=64, height=48, num_frames=2)
viewer.set_model(model)
viewer.set_camera(wp.vec3(3.0, -2.0, 1.5), pitch=-20.0, yaw=150.0)
while viewer.is_running():
  viewer.begin_frame(0.0)
  viewer.log_state(state)
  viewer.end_frame()
frame = viewer.get_frame()  # the last frame
```

Extra constructor options: `lighting`, `usd` (the stage's own materials and UsdLux lights),
`scene_options`
(`rw.SceneOptions`) and any dotted `RenderSettings` field. Without lights it reproduces Newton
ViewerGL's look from its own values (sun, hemispheric ambient, exposure, ACES).

## mjlab

raywarp is an RGB backend of mjlab's camera sensors (the `mjlab.rgb_renderers` entry point,
registered on install): configure a camera sensor with `RaywarpCameraSensorCfg`, or give a plain
`CameraSensorCfg` `rgb_backend="raywarp"`. Install it beside mjlab (`pip install "raywarp[mujoco]"
mjlab`).

```python
# docs: requires mjlab cuda
import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg

from raywarp.integrations.mjlab import RaywarpCameraSensorCfg

ROOM = """
<mujoco><worldbody>
  <light pos="0 0 3" dir="0 0 -1"/>
  <geom type="plane" size="2 2 0.1"/><geom type="box" size="0.2 0.2 0.2" pos="0 0 0.2" rgba="0.9 0.1 0.1 1"/>
  <camera name="front" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
</worldbody></mujoco>"""
front = RaywarpCameraSensorCfg(
  name="front",
  camera_name="room/front",
  width=64,
  height=64,
  raywarp_settings=rw.RenderSettings().replace(rw.FAST),  # max throughput (None: the default)
)
nenv = 4
lab = Scene(
  SceneCfg(num_envs=nenv, entities={"room": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROOM))}, sensors=(front,)),
  "cuda:0",
)
sim = Simulation(nenv, SimulationCfg(), lab.compile(), "cuda:0")
lab.initialize(sim.mj_model, sim.model, sim.data)
sim.set_sensor_context(lab.sensor_context)
sim.forward()
lab.sensor_context.render_rgb()
rgb = lab.sensor_context.get_rgb(mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "room/front"))
assert tuple(rgb.shape) == (nenv, 64, 64, 3)  # a uint8 torch tensor, written in place by the renderer
```

The sensors of one resolution and configuration share one Renderer, which writes each camera's
RGB straight into the tensor mjlab reads (no copy). `raywarp_settings` must agree with the
sensor's `use_shadows` / `use_textures` (it raises otherwise); `raywarp_per_world=rw.PerWorld(pose=["table/*"])`
declares the mocap entities reset events pose per world. mjlab's model-field randomisation events
work unchanged ([domain randomisation](domain_randomisation.md#mjlab)).

## Isaac Lab

raywarp is an Isaac Lab 3.0 camera renderer: give a camera `RaywarpRendererCfg` as its
`renderer_cfg`, the way you would pick Isaac's Newton Warp or RTX renderer.

```python
# docs: requires isaaclab
from isaaclab.sensors.camera import TiledCameraCfg

import raywarp as rw
from raywarp.integrations.isaaclab.cfg import RaywarpRendererCfg

camera = TiledCameraCfg(
  prim_path="{ENV_REGEX_NS}/Camera",
  width=128,
  height=128,
  data_types=["rgb", "depth"],
  renderer_cfg=RaywarpRendererCfg(settings={**rw.FAST, "shadows.enabled": True}),
)
```

How it fits in:

- **Scene**: the Newton model Isaac Lab builds for its renderers (its Newton cloner, the same one
  the Newton Warp renderer uses), with PhysX or Newton physics. Body poses come from Isaac's scene
  data provider every frame, on the GPU (no host copy).
- **Outputs**: Isaac's (N, H, W, C) arrays, written in place: `rgb` / `rgba` (uint8 sRGB),
  `rgb_hdr`, `albedo` (uint8 sRGB), `depth` / `distance_to_image_plane`, `distance_to_camera`,
  `normals` (world space), `instance_segmentation` and `semantic_segmentation` (int32 ids with
  Isaac's Newton Warp grouping: per nearest labelled ancestor prim / per label payload, filtered
  by `semantic_filter`; 0 background, 1 unlabelled; `camera.data.info[...]["idToLabels"]` with int
  keys). Misses follow
  `depth_clipping_behavior` ("none": inf, like Isaac RTX; "zero"; "max": the far clip). Not
  written: `motion_vectors`, colorized segmentation, the simple-shading modes, lens distortion.
- **Per environment**: each environment's camera pose and intrinsics (`set_intrinsic_matrices`
  may differ per environment) and Isaac's visual-material color randomisation
  (`randomize_visual_color` events) are per-world tables written on the device: no recompile.
- **Look**: `settings` is dotted overrides of the default (`{**rw.FAST}` for maximum throughput); `lighting` (a
  `rw.Lighting`; default a key and a fill light: Newton models carry none) lights the scene.

Installing next to Isaac Lab (the adapter is not in raywarp 0.3.0: install raywarp from source
until the next release). Isaac Lab 3.0.0 pins `warp-lang==1.17.0` and `newton==1.6.1`; raywarp
needs Warp 1.18, so install Isaac Lab first, then raywarp without its dependencies, then
override Warp. Keep Isaac's Newton 1.6.1 (Isaac Lab 3.0.0 breaks with Newton 1.7). These exact
commands were verified on an NVIDIA L4 (Python 3.12, kit-less); Warp 1.18 under Isaac Lab was
validated on the Newton-physics path only (PhysX physics needs Isaac Sim, not checked):

```bash
uv venv -p 3.12 && . .venv/bin/activate
uv pip install --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match "isaaclab==3.0.0"
git clone https://github.com/julien-blanchon/raywarp && uv pip install --no-deps -e raywarp
uv pip install "warp-lang>=1.18,<1.19"
```

## First frame and warm

The kernels are prebuilt: a renderer's first frame on a machine downloads the build its settings
select (only what it uses, from a public Hugging Face bucket, verified and kept in a local store),
so nothing compiles. A **cold** first frame still loads those modules, creates the OptiX modules
its frames bind (once per machine and driver), uploads the scene and runs its bakes; a **warm**
start finds all of it in the disk caches and starts at once. `Renderer.warm` warms ahead, in
the background, with the constructor's arguments (plus the state the first frame renders). It
returns a `concurrent.futures.Future`; start one per configuration the run will use and they run
together:

```python
setup = {"nworld": model.world_count, "cameras": [rw.CameraSpec(fovy=60)], "resolution": (64, 48)}  # the Newton scene above
fast = rw.RenderSettings().replace(rw.FAST)
warms = [rw.Renderer.warm(scene, **setup, settings=s, sim=state) for s in (rw.RenderSettings(), fast)]
print([str(w.result()) for w in warms])  # per step: seconds, cache entries added
r = rw.Renderer(scene, **setup, settings=fast)  # starts warm, also in a later process
```

Offline machines, mirrors (`RAYWARP_KERNELS_URL`), the store location (`RAYWARP_KERNELS`) and
the caches: [performance](performance.md#first-frame-prebuilt-kernels-caches-and-warm).

## Next

- [Scenes](scenes.md): lights, environment maps, materials and textures, cameras, the load report.
- [Settings](settings.md): what each group costs and when to turn it off.
- [Performance](performance.md): batching, the max-speed checklist, cold start.
- [Robotics recipes](robotics.md): torch, domain randomisation, CUDA graphs.
- [Recording video](recording.md): MP4 files of any batch without slowing the render loop.
- [FAQ](faq.md): black images, slow first frames, settings that did not take effect.
