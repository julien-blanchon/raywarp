# Features

Raywarp renders [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
[mjlab](https://github.com/mujocolab/mjlab), [Newton](https://github.com/newton-physics/newton),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), USD and glTF scenes into device arrays that a
policy reads with no copy ([quickstart](quickstart.md) for each integration).

## Why Raywarp

- **Batched.** One launch renders every world and camera. Static scenery is stored and traced
  once for all worlds; only what moves or is randomised has a copy per world.
- **Fast.** OptiX on RT-core GPUs, a Warp BVH elsewhere, a CUDA graph for the simulator step and
  the render together, and no host round trip per frame.
- **Physically based.** One material model (OpenPBR Surface) for every source: MJCF,
  UsdPreviewSurface, MDL, MaterialX, glTF. Area lights, soft shadows, image-based lighting,
  glossy reflections, glass and probe GI, with light units and automatic exposure.
- **Zero recompiles for randomisation.** Materials, textures, lights, environment, exposure,
  cameras and visibility vary per world through device tables. Changing a value, a scene or a
  settings recipe compiles nothing.
- **Prebuilt kernels.** The kernels are compiled ahead for each GPU architecture. The first
  renderer on a machine downloads only the modules it uses, so a new machine does not spend
  minutes compiling the renderer ([install](install.md#prebuilt-kernels)).

## One launch, many worlds

Every world and camera of a batch renders in one launch, into `(nworld, ncam, H, W)` arrays.
`r.render(d, active=ids)` renders only the listed worlds; the frame time follows their number
([selected worlds](robotics.md#selected-worlds-only)).

<!-- SHOWCASE: features/batch_mosaic.webp (docs/assets/SHOTS.md) -->
[![64 worlds of one simulation, one launch](assets/features/batch_mosaic.webp)](assets/features/batch_mosaic.mp4)

## Outputs

Any combination, in one launch: rgb, depth (planar or radial), instance and semantic
segmentation, normals (world or camera frame), albedo, HDR radiance, position, UV, material id,
roughness / metallic, direct / indirect light, motion vectors, NOCS, point clouds and camera
export (intrinsics, lens and pose in OpenCV conventions). Channels-first and float16 layouts for
training are written by the kernel ([robotics recipes](robotics.md#outputs-what-you-can-request-and-what-it-costs),
[ground truth](randomization.md#ground-truth)).

<!-- SHOWCASE: features/outputs.webp (docs/assets/SHOTS.md) -->
![One frame's outputs: rgb, depth, segmentation, normals, NOCS, motion](assets/features/outputs.webp)

## Per-world randomisation

Materials, textures (offset, rotation, hue), lights, environment maps, exposure, fog, cameras,
lens distortion, visibility and static object poses vary per world, with no recompile and inside
a captured CUDA graph ([domain randomisation](domain_randomisation.md)).

<!-- SHOWCASE: features/randomisation.webp (docs/assets/SHOTS.md) -->
![One scene, eight worlds, eight looks](assets/features/randomisation.webp)

## Cameras, lenses, sensors

Pinhole, orthographic, fisheye, OpenCV and Kannala-Brandt distortion, per-pixel ray bundles,
cameras that follow scene cameras ([cameras](scenes.md#cameras)); sensor noise, vignette, camera
response, rolling shutter, motion blur and stereo depth artefacts
([sensor effects](robotics.md#sensor-effects-noise-vignette-response-stereo-depth-rolling-shutter-motion-blur)).

## Lighting and materials

Point, spot, area (LTC), sun and IES lights, HDR environment maps, probe global illumination,
fog, many-light scenes through a light grid ([scenes](scenes.md)). OpenPBR Surface from every
format, textures with ray-cone mip selection and anisotropic filtering, clear coat, glass that
refracts camera rays, translucent layers ([materials](scenes.md#materials-and-textures)).
Anti-aliasing by 2 x 2 supersampling (`aa.spp=2`) and FXAA (`aa.post`); every frame is
deterministic, with no temporal accumulation ([settings](settings.md)).

## Recording and tools

MP4 files of one view or a whole batch, NVENC-encoded from device memory
([recording](recording.md)). A web viewer (`raywarp view`) and recording, conversion and kernel
management commands ([command line](cli.md)).
