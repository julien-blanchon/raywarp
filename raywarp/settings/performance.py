"""The `performance` settings group: which kernel build serves the scene, which ray traversal
backend runs, and when the acceleration structures rebuild; none of it changes the pixels. Pure
launch tuning (block size, register caps, BVH builder, micromaps, validation cadence) and the
debug counters build are not settings: `core/pipeline/platform/tuning.py`."""

from __future__ import annotations

import dataclasses
from typing import Literal, final

from raywarp.settings.fields import RenderOrderName, runtime_field, validate


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Performance:
  """How the frame is computed; never changes pixels.

  Attributes:
    kernels: which build of the render kernels serves the scene (C). "prebuilt": the tier's
      build for the scene's feature classes (`core/build/variants/tiers.py`), resolved without the
      scene, so that a new scene, a scene change or a run-time setting compiles nothing and the
      release's bundle (`core/build/kernel_cache/prebuilt.py`, downloaded on first use) serves it
      with zero compiles; a build the bundle lacks warns and compiles once (cached on disk).
      "scene": the settings' own build, specialised on this scene (its light count, kinds,
      materials, shapes), compiled on first use and cached on disk: a few percent faster than
      "prebuilt" on most scenes (RTX, 512 worlds, the `FAST` field set: +7-13 % for the prebuilt
      build), at the cost of a compile (about tens of seconds to a few minutes per build) for every
      new scene or compile-time change.
    traversal: "software" (Warp BVH), "optix" (hardware ray tracing) or "auto": OptiX on GPUs
      with RT cores, the software traversal on the CPU, on GPUs without RT cores (A100, H100) and
      for scenes with geometry OptiX does not trace (dim-1 deformables: segments). No fallback: a
      named backend that cannot render the scene, or "auto" on an RT-core GPU without driver OptiX,
      raises. The backend that runs: `Renderer.features.traversal` (C).
    render_order: the thread-to-pixel assignment (`types.RenderOrder`): "auto" (per frame,
      from the cameras), "tiled", "world" or "mixed"; never changes pixels (R).
    rebuild_every: rebuild the scene acceleration structures from the current poses every this
      many frames (0: refit only; topology from the first frame's poses): the software scene BVH
      (in place, on the device's schedule: graph-capturable) and the per-world OptiX structures
      (their world cells restarting from the current geometry). Shared scenery is static and
      never rebuilt; a scene without per-world instances rebuilds nothing. Refit only degrades as
      instances travel from where the topology was built: 1000 drifting boxes per world, 64
      worlds, L4, frames 100-200 -> 1900-2000: OptiX 4.1 -> 12.8 ms refit only vs 3.9 -> 4.4 ms
      at 64; software 14.2 -> 97.1 ms vs 9.5 -> 12.1 ms. One rebuild frame costs +3.4 ms on a
      28 ms frame (an interior of 3784 instances, all per world) and +3.1 ms on 8.5 ms (a room
      of 113; CPU, 2 worlds, 32^2): at 64, 0.2-0.6 % per frame amortised. Hits are the same up
      to float rounding at box and triangle edges (R).
  """

  kernels: Literal["prebuilt", "scene"] = "prebuilt"
  traversal: Literal["auto", "software", "optix"] = "auto"
  render_order: RenderOrderName = runtime_field("auto")  # noqa: RUF009  # a dataclasses.field
  rebuild_every: int = runtime_field(64, ge=0)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "performance")
