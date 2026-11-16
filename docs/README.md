# Raywarp documentation

Raywarp is a batched, physically based ray tracer for robot learning and dataset generation: one
launch renders thousands of worlds x cameras of a MuJoCo Warp, mjlab, Newton, Isaac Lab, USD or
glTF scene, on NVIDIA GPUs (CUDA, OptiX RT cores) or on the CPU.

## Getting started

| page | read it for |
| --- | --- |
| [Features](features.md) | what Raywarp does and why, with pictures: batching, outputs, randomisation, cameras, lighting, recording |
| [Install](install.md) | the extras, supported GPUs and drivers, the CPU device, the prebuilt kernels downloaded on first use, offline machines |
| [Quickstart](quickstart.md) | a first render with MuJoCo Warp, USD / glTF, Newton, mjlab and Isaac Lab, a few lines each; the first frame and warm start |
| [Concepts](concepts.md) | scene, renderer, worlds, cameras, settings recipes, per-world randomisation, kernels and first use: the model behind the API |

## Integrations

| page | read it for |
| --- | --- |
| [Quickstart](quickstart.md) | MuJoCo Warp (aliased model arrays, a drop-in for its renderer), Newton (`SensorCamera`, the headless viewer), mjlab camera sensors, the Isaac Lab renderer |
| [Robotics recipes](robotics.md) | zero-copy torch observations, step + render in one CUDA graph, selected worlds, sensor effects, CPU workers, the Isaac Lab lifecycle |
| [Scenes](scenes.md) | lights, environment maps, light units and exposure, materials (OpenPBR) and textures, cameras and lenses, segmentation, the load report |

## Outputs

| page | read it for |
| --- | --- |
| [Robotics recipes: outputs](robotics.md#outputs-what-you-can-request-and-what-it-costs) | rgb, depth, segmentation, normals, albedo, motion vectors and the rest: layouts, dtypes, what each costs |
| [Robotics recipes: point clouds](robotics.md#point-clouds-and-per-world-labels) | point clouds and per-world labels on the device |
| [Randomising everything: ground truth](randomization.md#ground-truth) | NOCS, camera-frame normals, camera export (intrinsics, lens, pose), optical flow |
| [Recording video](recording.md) | MP4 files of one view or a whole batch at render speed: NVENC from device memory or pipelined CPU encoding |

## Randomisation and datasets

| page | read it for |
| --- | --- |
| [Domain randomisation](domain_randomisation.md) | every randomisable quantity per world, per adapter, GI under randomisation, costs, a cookbook |
| [Randomising everything](randomization.md) | the dataset-generator view: each camera, object, lighting and scene knob, its granularity and what a change costs |
| [Data generation](datagen.md) | many scenes one after another: warm-up, fast switching, saving frames without stalling the render loop, measured costs |

## Performance and kernels

| page | read it for |
| --- | --- |
| [Settings](settings.md) | the default, `rw.FAST` and the recipes; each settings group, its cost and when to turn it off; compiled-in vs runtime fields |
| [Performance](performance.md) | batching, the max-speed checklist, OptiX vs the software BVH, the first frame (prebuilt kernels, caches, warm), zero compiles, CPU, multi-GPU |
| [FAQ](faq.md) | black or dark images, missing textures, slow first frames, changes that did not take effect, `StaleBinding`, OptiX errors |

## Tools

| page | read it for |
| --- | --- |
| [Command line](cli.md) | `raywarp`: view, record, convert, prepare, kernels, cache, info |
| [Benchmark](benchmark.md) | `python -m benchmarks.run`: ms/frame, images/s, first frame and compiles on three scenes |
| [Examples](../examples/) | small runnable scripts: MuJoCo Warp, USD, Newton, outputs, domain randomisation |

## Development

| page | read it for |
| --- | --- |
| [Developer guide](DEVELOPMENT.md) | architecture with diagrams, standards, testing, adding a feature or an adapter |
| [Changelog](../CHANGELOG.md) | what changed in each release |
| [About](about.md) | citation, licence, acknowledgements, scene and asset credits |
| [Shot list](assets/SHOTS.md) | every image and clip of these pages: scene, settings, camera, size |

## Notes on these pages

Every Python block of these pages runs in the test suite (`tests/tooling/test_docs.py`) on the
CPU device with tiny scenes; blocks marked `# docs: requires ...` run only where their
dependencies are available, and `# docs: skip` blocks are fragments. Performance numbers name the
GPU and configuration they were measured on; they are records, not a fresh measurement of your
machine (`python -m benchmarks.run` measures it). The reference recipe is the renderer's own
approximations at more samples, not physical ground truth.
