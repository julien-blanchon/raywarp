# Dataset generation: every knob and ground truth

What a dataset generator (SLAM, 3D reconstruction, 3D learning, robotics) can vary with Raywarp,
at which granularity, and the ground truth it can export. How to declare and write the per-world
tables is on [domain randomisation](domain_randomisation.md).

## Quick start

Render a Newton + USD batch and export depth, segmentation, camera-frame normals, NOCS and the
cameras, all on the device:

```python
import newton
import newton.examples
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.newton

robot = newton.ModelBuilder()
robot.add_usd(newton.examples.get_asset("ant.usda"))
builder = newton.ModelBuilder()
builder.add_ground_plane()
builder.replicate(robot, 4)
model = builder.finalize()
state = model.state()
scene = rw.integrations.newton.scene(model, lighting=rw.Lighting.studio())

outputs = rw.Outputs.of("rgb", "depth", "seg", "normal", motion=rw.settings.MotionOutput(status=True))
r = rw.Renderer(
  scene,
  nworld=model.world_count,
  cameras=[rw.CameraSpec(fovy=60)],
  resolution=(64, 48),
  settings=rw.RenderSettings().replace(rw.FAST).replace(outputs=outputs),
)
r.camera_pose.fill_(wp.transformf((0.0, -3.0, 1.2), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.35)))
out = r.render(state)

cameras = r.camera_info()  # per world and camera: K, lens, OpenCV camera-to-world pose
normals = r.camera_normals()  # (nworld, pixels) normals in each camera's OpenCV frame
nocs = r.nocs(state)  # (nworld, pixels) normalised object coordinates
assert cameras.k.numpy().shape == (4, 1, 3, 3) and nocs.shape == (4, 64 * 48)
depth, seg, flow_status = out.depth.numpy(), out.seg.numpy(), out.flow_status.numpy()
```

Nothing here reads back to the host until you call `.numpy()`. Save frames without stalling the
render loop: [data generation](datagen.md#saving).

<a id="ground-truth"></a>

## Export ground truth for a dataset

> **New in 0.4.0:** `r.camera_info()`, `r.camera_normals()` and `r.nocs(sim)`.

| ground truth | how | notes |
| --- | --- | --- |
| depth | the `depth` output | planar or radial (`DepthOutput(convention=)`) |
| normals | the `normal` output; camera frame: `r.camera_normals()` | world space, shading or geometric; camera frame in OpenCV axes (x right, y down, z forward) |
| NOCS | `r.nocs(sim)` after a render | each pixel's hit in its object's frame, scaled into the unit cube; needs `depth` and `seg` |
| camera parameters | `r.camera_info()` | per world and camera: K in pixels, lens model and coefficients, camera-to-world pose (OpenCV), resolution, frame index |
| instance / semantic segmentation | the `seg` and `semantic` outputs | per-world ids: `r.set_segment_table(table)`, `r.set_semantics(rules, worlds=)` |
| optical flow | the `motion` output, with `MotionOutput(status=True)` for a validity status | backward motion vectors in pixels ([status values](robotics.md#outputs-what-you-can-request-and-what-it-costs)) |
| point clouds | `r.point_cloud(...)` | on the device ([point clouds](robotics.md#point-clouds-and-per-world-labels)) |
| albedo, position, uv, material id, roughness, metallic, emission, direct, indirect, shadow | outputs of the same names | any combination ([outputs](robotics.md#outputs-what-you-can-request-and-what-it-costs)) |
| not yet | | 2D / 3D boxes, visibility fraction, keypoints, forward flow, COCO / BOP / KITTI writers |

`camera_info`, `camera_normals` and `nocs` cost nothing unless called. Each runs one launch with
no host read, and takes `out=` to write an earlier result in place.

<!-- SHOWCASE: randomization/ground_truth.webp (docs/assets/SHOTS.md) -->
![Ground truth: rgb, NOCS, camera-frame normals, motion status, camera frustums](assets/randomization/ground_truth.webp)

## Every knob

**Granularity**: W = per world, C = per camera, R = per renderer (every world), F = may change
every frame. Every W row is a per-world table written between frames: no compile, no host round
trip, and a captured CUDA graph keeps replaying.

### Camera

| quantity | how | gran. | notes |
| --- | --- | --- | --- |
| pose, trajectories (any jump) | `r.camera_pose`; MuJoCo `m.cam_pos` / `m.cam_quat`; mounted cameras | W C F | free |
| multi-camera rigs, stereo pairs | `cameras=[...]`, `CameraSpec(mount=, local=)`; per-world baseline through `camera_pose` | W F | |
| field of view, intrinsics | `PerWorld(camera=["fovy", "intrinsics"])` | W C F | free |
| lens distortion (OpenCV, Kannala-Brandt) | `PerWorld(camera=["distortion"])` | W C F | the lens model is picked at construction |
| camera model (pinhole, fisheye, orthographic) | `CameraSpec.projection` | C | mix them in one rig |
| resolution | `resolution=`, `CameraSpec.resolution` | R / C | a new Renderer |
| exposure | `PerWorld(env=["exposure"])` | W F | free; it does not touch the GI |
| white balance, lift / gamma / gain, saturation | `PerWorld(env=["grading.*"])` | W F | |
| sensor noise, vignette, response, chromatic aberration | `PerWorld(env=["sensor.noise", "sensor.vignette", "sensor.response", "sensor.aberration"])` | W F | |
| tone mapper, motion blur, rolling shutter, quantisation | `r.update_settings(...)` | R | a host step per change; a captured graph is captured again |
| depth sensors (active stereo, ToF noise) | `sensor.depth`, `sensor.*` | R | |
| clip planes | `r.set_clip` | C | |
| depth of field, bloom | not available | | |

### Objects

| quantity | how | gran. | notes |
| --- | --- | --- | --- |
| simulated poses, articulated robots | the simulator's state | W F | |
| static prop poses | `PerWorld(pose=[...])`, then `per_world.columns("instance.pos" / "instance.rot", ...)` | W F | the props get one copy per world |
| primitive sizes | `PerWorld(size=[...])` | W F | primitives only |
| mesh scale (uniform) | `PerWorld(pose=[...])`, then write `instance.rot` as s R (s > 0) | W F | no mesh copy; non-uniform scale is a different mesh |
| show / hide (object pools, distractors) | `PerWorld(visible=[...])` | W F | free |
| mesh swap among the scene's meshes | `PerWorld(mesh=[...])` | W F | |
| different objects per world | Newton models with different shapes per world, Isaac Lab prototypes, a pool + `visible` | W | |
| albedo, roughness, metallic, specular weight / IOR, coat, emission | `PerWorld(material=[...])` (OpenPBR names) | W F | with GI on, a change re-lights that world's GI |
| instance colour, material assignment, glass per world | `material=["color", "instance.material"]` | W F | glass per world is slower in large scenes |
| texture swap, UV offset / rotation, hue, contrast, repeat | `material=["texture", "tex_variation", "tex_affine", "tex_repeat"]`, `rw.randomize` | W F | among the scene's textures; no texture copy |

### Lighting and scene

| quantity | how | gran. | notes |
| --- | --- | --- | --- |
| environment map, rotation, intensity | `PerWorld(env=["id", "yaw", "intensity"])` over a bank of maps | W F | sun direction: a bank of `rw.scene.sun_sky` maps + yaw, or a distant light's `light.dir` |
| light on / off (count), colour, strength, position, direction, radius, cone, range | `PerWorld(light=[...])` | W F | add every light a world may use, switch with `active` |
| light type, area shape and size | `light=["type", "shape", "size", "axis"]` | W F | the extra light code is picked at construction |
| shadows per light | `light=["cast_shadow"]` | W F | shadow quality: R |
| fog | `PerWorld(env=["fog.color", "fog.range"])` | W F | |
| GI | automatic | W | follows each world's lights, materials and environment ([below](#gi-under-randomisation-per-frame-or-per-episode)) |
| another scene | a new Renderer: nothing compiles | | 0.25-0.65 s warm switch ([data generation](datagen.md)) |
| several scenes in one batch | not available: one scene per Renderer | | pool the objects and switch them with `visible` |

## Prove a generator never compiles

Turn compiling off for the process and run the generator's loop. A configuration whose GPU code
is not prebuilt then raises `rw.kernels.KernelsNotPrebuilt`, naming it, instead of compiling:

```python
# docs: skip (a fragment: your generator's loop)
rw.kernels.configure(compile=False)  # or RAYWARP_KERNELS_COMPILE=0
r = rw.Renderer(scene, nworld=64, resolution=(256, 256), per_world=per_world)
for frame in range(100):
  randomise(r, rng)  # writes r.per_world[...] tables every frame
  r.render(sim)
```

What is prebuilt: [how the kernels work](performance.md#how-the-kernels-work).

## GI under randomisation: per frame or per episode

With GI on, a world whose materials, light positions or environment map change is re-lit by rays
on that frame. That takes a few ms per changed world on an RTX GPU, and about a second per frame
for 4 worlds on the CPU. A generator that does not need the GI to follow every frame can re-light once per
episode:

```python
# docs: skip (a fragment of a generator's loop)
r.update_settings(**{"gi.relight": "on_request"})
...
r.request_gi_relight(reset_mask)  # (nworld,) bool device array: the worlds reset this step
r.render(sim)
```

Worlds not requested keep their GI. Their light colours, strengths, on / off and environment
intensity still follow every frame, without rays. What each change costs:
[GI under randomisation](domain_randomisation.md#gi-under-randomisation).

## See also

- [Domain randomisation](domain_randomisation.md): how to declare and write the tables, per simulator.
- [Data generation](datagen.md): many scenes, warm-up, saving to disk.
- [Robotics recipes: outputs](robotics.md#outputs-what-you-can-request-and-what-it-costs): every output.
