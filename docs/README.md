# raywarp user guide

A batched, physically based ray tracer for robotics training: one launch renders thousands of
worlds x cameras of a MuJoCo Warp, mjlab, Newton, Isaac Lab, USD or glTF scene, on NVIDIA GPUs (CUDA, OptiX RT cores) or
the CPU.

| page | read it for |
| --- | --- |
| [Install](install.md) | the extras, the prebuilt kernels downloaded on first use, the download / compile switches (`rw.kernels`) |
| [Quickstart](quickstart.md) | `X.scene(...)` for MuJoCo Warp, USD / glTF, Newton (`SensorCamera`), mjlab, Isaac Lab in a few lines each; the first frame and warm |
| [Scenes](scenes.md) | lights, environment maps, light units and exposure, materials and textures, cameras (intrinsics, OpenCV, ray bundles), segmentation, the load report |
| [Settings](settings.md) | the default, `rw.FAST` and the recipes; each settings group, its cost and when to turn it off; anti-aliasing; compiled-in vs runtime fields (generated table) |
| [Performance](performance.md) | batching, the max-speed checklist, OptiX, the first frame (prebuilt kernels, caches, warm), zero compiles, CPU, multi-GPU, how to measure, recorded numbers |
| [Robotics recipes](robotics.md) | zero-copy torch observations, domain randomisation per world, step + render in one CUDA graph, CPU workers |
| [Domain randomisation](domain_randomisation.md) | every randomisable quantity per world, per adapter (MuJoCo Warp model tables, mjlab, Newton, Isaac Lab, USD), GI under randomisation, costs, gaps; a cookbook of recipes |
| [Randomising everything (draft)](randomization.md) | the dataset-generator view: every camera, object, lighting, scene knob and ground-truth output, its granularity (world / camera / renderer / frame) and what a change costs; the zero-compile proof script |
| [Recording video](recording.md) | MP4 files of one view or a whole batch at render speed: NVENC from device memory or pipelined CPU encoding, colour tags, pitfalls |
| [Data generation](datagen.md) | many scenes one after another: warm-up, fast switching, saving frames without stalling the render loop, measured costs |
| [FAQ](faq.md) | black or dark images, missing textures, slow first frames, changes that did not take effect, `StaleBinding`, OptiX errors |
| [Command line](cli.md) | `raywarp`: view, record, convert, prepare, kernels, cache, info |
| [Benchmark](benchmark.md) | `python -m benchmarks.run`: ms/frame, images/s, first frame and compiles on three scenes |
| [Examples](../examples/) | small runnable scripts: MuJoCo Warp, USD, Newton, outputs, domain randomisation |
| [Developer guide](DEVELOPMENT.md) | architecture with diagrams, standards and gates, testing, adding a feature or an adapter |

The reference recipe is the renderer's own approximations at more samples; it is not ground truth.
Performance tables name the commit and configuration they were recorded on; they are not a fresh
measurement of the current tree.

Python examples are exercised by `tests/tooling/test_docs.py` with tiny scenes. Blocks marked
`# docs: requires ...` run only when their dependencies are available; `# docs: skip` blocks
are not executed. These checks do not validate every prose claim.

**Evidence paths.** Measurements cite their evidence as `internal/results/...` or
`internal/research/...` paths: the development repository's, not part of the public tree; the
older trees are archived, with a sha256 manifest, in the HF bucket at
`hf://buckets/blanchon/mujoco-warp-pbr/archive/repo-results-2026-11-02/` and
`.../archive/research-plan-2026-11-02/`.
