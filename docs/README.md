# Raywarp documentation

Raywarp renders thousands of simulated worlds in one call, physically based, on NVIDIA GPUs or the CPU.

## Quick start

```bash
pip install "raywarp[newton,usd]"   # or [mujoco], [gltf], [all]
```

Pick your simulator or format:

| integration | what you get | minimal example |
| --- | --- | --- |
| **Newton + USD** | a USD stage loaded into Newton, every world rendered with its materials and lights | [quickstart](quickstart.md#quick-start), [stage materials](quickstart.md#newton-with-the-stages-own-materials-and-lights) |
| Newton drop-ins | `SensorCamera` and a headless viewer with Newton's own API, PBR shaded | [SensorCamera](quickstart.md#newtons-camera-sensor-sensorcamera), [examples/newton_sensor.py](../examples/newton_sensor.py) |
| MuJoCo Warp | an MJCF model and its batched `mjw.Data`, or a drop-in for its renderer | [quickstart](quickstart.md#mujoco-warp), [examples/mujoco_render.py](../examples/mujoco_render.py) |
| Isaac Lab | a camera renderer next to Isaac's Newton Warp and RTX renderers | [quickstart](quickstart.md#isaac-lab) |
| mjlab | an RGB backend of mjlab's camera sensors | [quickstart](quickstart.md#mjlab) |
| USD / glTF files | a static stage or asset, rendered as authored | [quickstart](quickstart.md#usd-and-gltf-without-a-simulator), [examples/usd_scene.py](../examples/usd_scene.py) |

See it first: the [Demo gallery](gallery.md) shows looping clips of runnable demos.

## Concepts

| page | read it for |
| --- | --- |
| [Concepts](concepts.md) | the mental model: scene, renderer, worlds, cameras, settings, per-world randomisation |
| [Features](features.md) | a tour with pictures of what Raywarp does |

## How-to

| page | read it for |
| --- | --- |
| [Quickstart](quickstart.md) | render a first batch with Newton + USD, MuJoCo Warp, mjlab, Isaac Lab or a USD / glTF file |
| [Robotics recipes](robotics.md) | feed torch with no copy, pick outputs, point clouds, sensor effects, CUDA graphs, skip finished episodes |
| [Domain randomisation](domain_randomisation.md) | give every world its own look, per simulator, with a cookbook |
| [Data generation](datagen.md) | switch between many scenes and save frames to disk without stalling |
| [Dataset generation: every knob and ground truth](randomization.md) | every knob with its granularity; export NOCS, normals, cameras and flow |
| [Recording video](recording.md) | record one view or a whole batch to MP4 at render speed |
| [Scenes](scenes.md) | light a scene, use environment maps, materials, textures and cameras |

## Reference

| page | read it for |
| --- | --- |
| [Settings](settings.md) | every setting, the recipes and what each group costs |
| [Outputs](robotics.md#outputs-what-you-can-request-and-what-it-costs) | every output: layout, dtype, meaning, cost |
| [Cameras](scenes.md#cameras) | projections, lenses, mounts, intrinsics |
| [Performance and kernels](performance.md) | batching, the max-speed checklist, prebuilt kernels, the first frame, multi-GPU |
| [Install](install.md) | extras, supported GPUs and drivers, the CPU, offline machines |
| [Command line](cli.md) | `raywarp`: view, record, convert, prepare, kernels, cache, info |
| [Benchmark](benchmark.md) | `python -m benchmarks.run`: ms per frame, images/s, first frame |
| [Troubleshooting](faq.md) | dark images, missing textures, slow first frames, changes that did not take effect |
| [About](about.md) | citation, licence, acknowledgements, asset credits |

Contributors: the [developer guide](DEVELOPMENT.md), the [changelog](../CHANGELOG.md) and the
[shot list](assets/SHOTS.md) of every image on these pages.

## See also

- Every Python block of these pages runs in the test suite on the CPU, with tiny scenes. Blocks
  marked `# docs: requires ...` run where their dependencies are installed.
- Performance numbers name the GPU and settings they were measured with. They are records, not a
  measurement of your machine: `python -m benchmarks.run` measures yours.
