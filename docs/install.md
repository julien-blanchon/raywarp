# Install

```bash
pip install raywarp               # Python >= 3.12: the core (warp-lang 1.18 and numpy)
pip install "raywarp[mujoco]"     # MuJoCo Warp scenes (and mjlab)
pip install "raywarp[newton]"     # Newton scenes and the SensorCamera drop-in
pip install "raywarp[usd]"        # USD stages (usd-core, pillow, materialx, openexr)
pip install "raywarp[gltf]"       # glTF scenes (the usd extra and pygltflib)
pip install "raywarp[viewer]"     # the web viewer's image encoding (pillow)
pip install "raywarp[video]"      # NVENC recording from device memory (PyNvVideoCodec; Linux, Windows)
pip install "raywarp[all]"        # all of the above
```

The wheel bundles the OptiX SDK headers and the MDL SDK (native code, built for the stable ABI:
one cp312-abi3 wheel per platform). The GPU kernels are not in it: see below. Recording also needs
the ffmpeg CLI on `PATH` ([recording](recording.md)). Check a machine with `raywarp info`
([CLI](cli.md)): the GPU, its compute capability and RT cores, the driver, OptiX, Warp, the
traversal `traversal="auto"` picks and whether the kernels' remote answers.

## Supported GPUs

An NVIDIA GPU with a driver that serves OptiX 9.0 (R580 or newer) runs the RT-core path. Prebuilt
kernels exist for these architectures; others compile their kernels once on first use.

| architecture | GPUs | traversal |
| --- | --- | --- |
| sm_80 | A100 | Warp BVH (no RT cores) |
| sm_86 | A10, A40, RTX 30 | OptiX (RT cores) |
| sm_89 | L4, L40S, RTX 40 | OptiX (RT cores) |
| sm_90 | H100 | Warp BVH (no RT cores) |
| sm_120 | RTX PRO 6000 Blackwell, RTX 50 | OptiX (RT cores) |

`performance.traversal="auto"` (the default) picks OptiX on GPUs with RT cores and the Warp BVH
on GPUs without them and on the CPU. Both are first-class paths with the same features. There is
no silent fallback: where OptiX cannot run on an RT-core GPU, the renderer raises and names the
cause; `performance.traversal="software"` selects the Warp BVH explicitly. OptiX comes with the
NVIDIA driver; Raywarp calls it through its own binding and bundles the SDK headers, so there is
nothing else to install.

## CPU

Every feature runs unchanged on Warp's CPU device (`wp.ScopedDevice("cpu")`), with images within a
few 1/255 of the GPU's. It is meant for debugging and tests, not throughput
([performance](performance.md#running-on-the-cpu)).

## Prebuilt kernels

The kernels are compiled ahead, one bundle per GPU architecture (sm_80: A100; sm_86: A10, RTX 30;
sm_89: L4, L40S, RTX 40; sm_90: H100; sm_120: RTX PRO 6000, RTX 50), plus per NVIDIA driver release
a pack of ready OptiX modules. Nothing to install: the first renderer on a GPU detects its compute
capability and downloads, from the public Hugging Face bucket `blanchon/raywarp-kernels`, only what
it uses of that architecture's bundle (megabytes for a scene's first frame), each file checked
against the hashes pinned in the installed package, into a local store (`$RAYWARP_KERNELS`,
default `~/.cache/raywarp/kernels`). Another architecture (or the CPU) compiles its kernels on
first use, once (cached). [Performance](performance.md#first-frame-prebuilt-kernels-caches-and-warm):
what a first frame costs, warm starts, the caches.

Two switches, for the process (`None` keeps the current value) or by environment (`1` / `0`, on
by default):

```python
import raywarp as rw

rw.kernels.configure(download=False)  # RAYWARP_KERNELS_DOWNLOAD=0: never fetch, the local store only
rw.kernels.configure(compile=False)  # RAYWARP_KERNELS_COMPILE=0: what is not prebuilt raises
print(rw.kernels.current())
rw.kernels.configure(download=True, compile=True)
```

- **download off**: nothing is fetched; a store filled ahead (`rw.kernels.fetch(arch)` or
  `raywarp kernels fetch --arch 86` on a machine with network, then copied, `RAYWARP_KERNELS`
  pointing at it) is read as it is. What it lacks compiles (or raises, with compile off).
- **compile off**: a configuration whose kernels are not prebuilt raises
  `rw.kernels.KernelsNotPrebuilt` instead of compiling, naming it and how to get its kernels:
  settings outside every prebuilt tier (the error lists the fields that differ from the nearest
  tier), `performance.kernels="scene"`, the CPU device, or a kernel the bundle lacks.

`rw.kernels.info()` (`raywarp kernels info`) says what serves this machine: the architecture,
its pinned bundle, the local store, the remote, the driver, the switches. `RAYWARP_KERNELS_URL`
points at a mirror of the remote (`file://` works); `HF_TOKEN` raises the bucket's rate limit.

## From source

```bash
git clone https://github.com/julien-blanchon/raywarp && cd raywarp
uv sync --all-extras                # the native build hook fetches the pinned OptiX headers and MDL SDK
```

The [developer guide](DEVELOPMENT.md) covers the layout, the tests and adding a feature.
