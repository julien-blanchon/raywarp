"""Hardware ray-tracing (OptiX) trace backend.

Requirements: an NVIDIA driver serving the OptiX 9.0 ABI (OptiX ships in the driver); the OptiX SDK
headers (github.com/NVIDIA/optix-dev) are bundled with raywarp by its build.

- `launch.py`: `OptixRenderer`, one build of a Renderer and its frame (structures, programs,
  launch); its `TraceBackend` is `core/trace/optix_backend.py`'s `OptixBackend`.
- `accel.py`: the structures of a build (`OptixScene`): static GASes and the per-world and shared
  instance groups (`group.py`: one IAS each), refreshed every frame.
- `gas.py`: geometry acceleration structures: static mesh / cube GASes, curve and custom inputs;
  `dynamic.py`: the height-field and deformable GASes, refitted.
- `instances.py`: Warp kernels writing the build inputs (cell bounds, instances, curves, boxes);
  `mesh_inputs.py`: the static GASes' (packed face normals, the unit cube).
- `device/`: the device code: `trace.py` (`trace_closest` / `trace_any` over OptiX and the world
  cells), `hits.py` (what the hit and intersection programs compute), `programs.py` (the
  programs of a build, one Warp module), `params.py` (their launch parameters), `intrinsics.py`
  (the OptiX device intrinsics' C++ preamble, registered as builtins by `core/kernels/native.py`).
- `pipeline.py`: PTX compile, pipelines, SBTs and (chunked) launches; `modules.py`: the OptiX
  modules, created once per machine and driver; `strides.py`: the launch parameters' stride check.
- `supersample.py`: spp > 1 as a supersampled image, downsampled.
- `api.py`: the OptiX host API through ctypes (the driver's function table and its structs).
- `context.py`: the OptiX device context (one per device), its objects, parallel module compile
  and the PTX target.
- `graph.py`: CUDA graphs of OptiX renders (Warp segments around eager OptiX calls).
- `omm/`: opacity micromaps of the alpha cut-outs (`micromaps.py`; `bake.py` / `kernels.py`: the
  bake; `core/trace/omm_sources.py`: which instances get them).

Nothing here loads the driver's OptiX library at module import time.
"""

from __future__ import annotations
