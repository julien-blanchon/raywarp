# Randomising everything for dataset generation (draft)

What a dataset generator (SLAM, 3D reconstruction, 3D learning, robotics) can randomise with
raywarp, at which granularity, and what a change costs. Every row marked **run time** is written
into a device table between frames: no compile, no new kernel, no host round trip, the same
captured CUDA graph keeps replaying. How to declare and write the tables:
[domain_randomisation.md](domain_randomisation.md).

**Gran.**: W = per world (each world of the batch its own value), C = per camera, R = per
renderer (every world), F = may change every frame. **Build** = chosen once when the Renderer is
constructed (a declaration or a setting), then run time.

## Camera

| quantity | how | gran. | notes |
| --- | --- | --- | --- |
| pose, trajectories (any jump) | `r.camera_pose`; MuJoCo `m.cam_pos` / `m.cam_quat`; mounts | W C F | free |
| multi-camera rigs, stereo pairs | `cameras=[...]`, `CameraSpec(mount=, local=)`; baseline per world through `camera_pose` | W F | |
| field of view, intrinsics | `PerWorld(camera=["fovy", "intrinsics"])` | W C F | free |
| lens distortion (OpenCV, Kannala-Brandt) | `PerWorld(camera=["distortion"])` | W C F | build: the lens model |
| camera model (pinhole, fisheye, orthographic) | `CameraSpec.projection` | C | mix them in one rig |
| resolution | `resolution=`, `CameraSpec.resolution` | R / C | a new Renderer |
| exposure | `PerWorld(env=["exposure"])` | W F | free: no GI input (it scales the image after the GI) |
| white balance, lift / gamma / gain, saturation | `PerWorld(env=["grading.*"])` | W F | |
| sensor noise, vignette, response, chromatic aberration | `PerWorld(env=["sensor.noise", "sensor.vignette", "sensor.response", "sensor.aberration"])` | W F | |
| tone mapper, motion blur, rolling shutter, quantisation bits | `r.update_settings(...)` | R | a host step per change (re-captures a graph); no host read but for a background / fog colour or tone mapper not seen before |
| depth sensors (active stereo, ToF noise) | `sensor.depth`, `sensor.*` | R | |
| clip planes | `r.set_clip` | C | |
| depth of field, bloom | not available | | |

## Objects

| quantity | how | gran. | notes |
| --- | --- | --- | --- |
| simulated poses, articulated robots | the simulator's state | W F | |
| static prop poses | `PerWorld(pose=[...])` + `per_world.columns("instance.pos"/"instance.rot", ...)` | W F | the props leave the shared tree (one refit per frame) |
| primitive sizes | `PerWorld(size=[...])` | W F | primitives only; meshes: the next row |
| mesh scale (uniform) | `PerWorld(pose=[...])` + `per_world.columns("instance.rot", ...)` written s R (s > 0) | W F | pose data: no recompile, no mesh copy (the rays are mapped by s, its normals stay unit); one refit per frame as any pose. Uniform only: a non-uniform scale is a different mesh (author it, or `PerWorld(mesh=[...])`). A primitive's rotation stays a rotation (its size scales it) |
| show / hide (object pools, distractors) | `PerWorld(visible=[...])` | W F | free |
| mesh swap among the scene's meshes | `PerWorld(mesh=[...])` | W F | |
| different objects per world | Newton `build.pooled`, Isaac Lab prototypes, a pool + `visible` | W | |
| albedo, roughness, metallic, specular weight / IOR, emission | `PerWorld(material=[...])` | W F | with GI on, a change re-lights that world's GI by rays (`gi.relight` below) |
| instance colour, material assignment, glass per world | `material=["color", "instance.material"]` | W F | glass: build (`world_optics`), slower traversal |
| texture swap, UV offset / rotation, hue, contrast, repeat | `material=["texture", "tex_variation", "tex_affine", "tex_repeat"]`, `rw.randomize` | W F | among the scene's textures (a procedural bank built with the scene) |

## Lighting and scene

| quantity | how | gran. | notes |
| --- | --- | --- | --- |
| environment map, rotation, intensity | `PerWorld(env=["id", "yaw", "intensity"])` over a bank | W F | sun direction: a bank of `rw.scene.sun_sky` maps + yaw, or a distant light's `light.dir` |
| light on / off (count), colour, strength, position, direction, radius, cone, range | `PerWorld(light=[...])` | W F | add every light a world may use, switch with `active`; with GI on, on / off, colour and strength recombine the GI without rays (also after a world's materials changed), the others re-light it by rays |
| light type, area shape and size | `light=["type", "shape", "size", "axis"]` | W F | build: the profiles / area lights |
| shadows per light | `light=["cast_shadow"]` | W F | quality: R |
| fog | `PerWorld(env=["fog.color", "fog.range"])` | W F | |
| GI | automatic | W | each world's GI follows its lights, materials and environment: colour / strength / on-off / environment-intensity changes recombine without rays, other changes re-light the world by rays on that frame; `gi.relight="on_request"` holds those until `r.request_gi_relight(worlds)` (e.g. once per episode at reset) |
| another scene | a new Renderer (same kernels: nothing compiles) | | 0.25-0.65 s warm switch ([datagen.md](datagen.md)) |
| several scenes in one batch | not available: one scene per Renderer (pool the objects and switch them with `visible`) | | |

## Ground truth

| output | notes |
| --- | --- |
| depth | planar or radial (`DepthOutput(convention=)`) |
| normals | world space, shading or geometric; camera frame (OpenCV): `r.camera_normals()` after a render (a post pass, nothing when not called) |
| NOCS | `r.nocs(sim)` after a render: each pixel's hit in its instance's frame, centred on the instance's box over its diagonal, + 0.5 (from depth, seg and the instance poses; a post pass) |
| camera parameters | `r.camera_info()`: per world and camera K (pixels), lens model + coefficients, camera-to-world pose (OpenCV frame), resolution, frame index; device arrays, no host read |
| instance / semantic segmentation | per-world ids: `set_segment_table`, `set_semantics(rules, worlds=)` (no host read; the first per-world call makes captured graphs stale) |
| albedo, position, uv, material id, roughness, metallic, emission, direct, indirect, shadow | output words, any combination |
| optical flow | backward motion vectors + a validity status (`MotionOutput(status=True)`) |
| point clouds | `r.point_cloud(...)`, on the device |
| saving | your writer: [data generation](datagen.md#saving) (rgb, depth, seg; the cameras of every frame from `r.camera_info()`) |
| not yet | 2D / 3D boxes, visibility fraction, keypoints, forward flow, COCO / BOP / KITTI writers |

<!-- SHOWCASE: randomization/ground_truth.webp (docs/assets/SHOTS.md) -->
![Ground truth: rgb, NOCS, camera-frame normals, motion status, camera frustums](assets/randomization/ground_truth.webp)

## Check it on your setup

To prove a generator never compiles, turn compiling off for the process and run its loop: a
configuration whose kernels are not prebuilt then raises `rw.kernels.KernelsNotPrebuilt`, naming
it, instead of compiling ([install](install.md#prebuilt-kernels)).

```python
# docs: skip (a fragment: your generator's loop)
rw.kernels.configure(compile=False)  # or RAYWARP_KERNELS_COMPILE=0
r = rw.Renderer(scene, nworld=64, resolution=(256, 256), per_world=per_world)
for frame in range(100):
  randomise(r, rng)  # writes r.per_world[...] tables every frame
  r.render(sim)
```

## GI under randomisation: per frame or per episode

With GI on, a world whose materials, light positions / shapes or environment map change is re-lit
by rays on that frame (a few ms per changed world on RTX; on the CPU about a second per frame
for 4 worlds). A generator that randomises every frame and does not need the GI to follow each
frame exactly can re-light once per episode:

```python
# docs: skip (a fragment of a generator's loop)
r.update_settings(**{"gi.relight": "on_request"})
...
r.request_gi_relight(reset_mask)  # (nworld,) bool device array: the worlds reset this step
r.render(sim)
```

Worlds not requested keep their GI until requested (their lights' colours, strengths, on / off
and the environment intensity still follow every frame, recombined without rays).
