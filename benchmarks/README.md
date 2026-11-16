# Benchmark

One script, three scenes: frame time and throughput of a batch of worlds, the first-frame time
and how many kernels compiled.

```bash
pip install "raywarp[mujoco]"           # franka needs MuJoCo; showcase and sponza need nothing more
python -m benchmarks.run --scene franka --worlds 64 --res 512
python -m benchmarks.run --scene sponza --worlds 16 --res 1024 --settings fast
```

```
scene franka: 64 worlds x 512^2, settings default, 100 frames
  ... ms/frame, ... images/s
  first frame ... s, compiles {...}
  image: benchmark-franka.png
```

Options: `--worlds N` (default 64), `--res R` (square images, default 512), `--settings default|fast`
(`RenderSettings()` or `rw.FAST` over it), `--frames N` (timed frames, default 100),
`--device cuda:N`, `--png PATH`. Run from a checkout's root (the script is `benchmarks/run.py`).

- **ms/frame**: the mean of the timed `render` calls of all worlds (one camera each), after one
  frame of warm-up; **images/s**: worlds x frames / seconds.
- **first frame**: from the `Renderer`'s construction to its first image on the device. It
  includes loading the kernels: on a machine's first run the prebuilt kernels of the GPU's
  architecture download (run twice for the warm number).
- **compiles**: Warp modules and OptiX programs compiled here, OptiX modules created. With the
  prebuilt kernels and the driver's OptiX module pack this is 0.

## Scenes

The scene files download on first use from the public bucket
[`blanchon/raywarp-assets`](https://huggingface.co/buckets/blanchon/raywarp-assets) (`benchmarks/v<format>-c<converter>/`,
beside the USD sources and their licences) into the cache directory (`RAYWARP_CACHE_DIR`, else
`~/.cache/raywarp`, under `benchmarks/<version>/`), each checked against its sha256. They are
converted ahead per `.rayw` format and USD converter version; a mirror: `RAYWARP_BENCHMARKS_URL`.

| Scene | What | Source and licence |
|---|---|---|
| `franka` | Menagerie's MJX pick scene: the Franka Emika Panda over a checker floor, three cubes, a key spot and a fill sun (MJCF, through the MuJoCo adapter) | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `franka_emika_panda` (`mjx_single_cube.xml`), Apache-2.0 |
| `showcase` | raywarp's showcase room: a furnished room with a window, area and punctual lights, a robot arm (`.rayw`) | raywarp, Apache-2.0 |
| `sponza` | Intel's New Sponza, the full scene: curtains, ivy, a cypress (`.rayw`) | [Intel Graphics Research Samples](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html), "Sponza" by Intel Corporation, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (converted to raywarp's scene format) |
