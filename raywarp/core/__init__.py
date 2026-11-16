"""The GPU core, by pipeline stage (packages only). Device code (the kernel-code packages,
`build.kernel_cache.kernel_code`): `trace.rays`, `trace.accel`, `trace.optix.device` (getting
hits), `shade` (lighting a hit: materials, lights, shadows, GI, AO, environment), `post.exposure`,
`post.grading`, `post.display`, `post.temporal` (the image after the hit), `kernels` (the
megakernel assembly and its ABI). Host code: `trace` (the trace backends, OptiX's host side),
`post.aa`, `post.sensor` (plain Warp passes), `outputs` (render outputs, ground truth, point
clouds, camera export), `pipeline` (`Renderer`, frames, bakes, the GPU scene and its upload,
textures, per-world tables, Warp platform), `build` (kernel variants, the prebuilt kernel cache).
Modules are imported by path; the public names are re-exported by `raywarp`.
"""

from __future__ import annotations

from raywarp.core.pipeline.platform import module_lock as _module_lock

# Before any kernel of the core loads: Warp's module loads are made safe across threads.
_module_lock.install()
