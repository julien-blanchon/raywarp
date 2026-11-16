# Benchmark

`python -m benchmarks.run --scene franka|showcase|sponza [--worlds N --res R --settings fast|default]`,
from a checkout's root, prints the frame time, the images per second, the first-frame time and the
compile count of one scene, and saves one PNG. The scenes (MuJoCo Menagerie's Franka, raywarp's
showcase room, Intel's New Sponza) download on first use. [benchmarks/README.md](../benchmarks/README.md):
the options, what each number measures, the scenes' sources and licences.

[Performance](performance.md) explains what makes a frame fast (batching, OptiX, the max-speed
checklist) and lists recorded numbers.
