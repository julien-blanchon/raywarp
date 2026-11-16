"""Trace backends: what traces a Renderer's camera rays (the software BVH or OptiX), behind one
Protocol, and the registry that picks one for a scene and device.

A `TraceBackend` owns the acceleration structures of one Renderer and the launch of its frame.
The Renderer never tests which backend it has: every difference goes through the object it
holds. A backend reads the renderer's state as a `context.FrameContext` (and writes the software
traversal's structures into its `SoftwareScene`). The contract, in frame order:

- class level (no instance): `name` (the `traversal` setting, `Features.traversal`),
  `device_reason(caps)` / `scene_reason(scene)` (None when it can run on that device / trace
  `scene`, else why not), `preferred(caps)` (traversal="auto" picks it on that device: the
  device's capability decides, never the scene), what the Renderer resolves differently for it
  (`mesh_frames`, `supersamples`) and the Warp kernels of its own that a frame
  launches (`first_frame_kernels`, compiled ahead by `core/pipeline/warmup.py`; without
  the software probe-ray kernel where its hardware traces the probes, `hardware_probes`);
- `tracer(f, cull)`: the ray queries (`trace.Tracer`) kernel bodies call, specialised on `f`;
- `configure(ctx)` after the Features changed, `prepare(ctx, wait=)` (compiles, structure builds;
  `wait=False` from the constructor), then per frame `refit(ctx, frame)` and
  `launch(ctx, frame, seed)` (the Inputs of a software launch, for the post passes, or None);
- the renderer's software ray kernels (GI probes, light grid) always use the software traversal
  (world 0's scene BVH, every world's with GI): `software_bvh(ctx, frame)` /
  `software_meshes(ctx)` make its BVH and mesh BLASes current where the backend does not trace
  with them itself; `grid_tracer(ctx, frame, see_through)` and `gi_tracer(ctx, frame)` may run
  the light grid's visibility test and the GI bake's rays on the backend's own hardware instead;
- `area_shadows(ctx, frame, launch)` traces the soft area shadows' pass after the frame's launch;
- `close()` releases what the backend holds.

`BACKENDS` lists the registered backends in traversal="auto" order and `select` resolves the
setting, with no fallback: "auto" takes the first backend the device prefers that can trace the
scene (OptiX on a GPU with RT cores; the software traversal on the CPU, on GPUs without RT cores
(A100, H100) and for the geometry OptiX does not trace: dim-1 deformables), and raises when
the device's preferred backend cannot run there (an RT-core GPU without driver OptiX); a named
backend that cannot render the scene raises. The backend that runs is `Features.traversal`. A new
backend implements the Protocol and is added to `BACKENDS`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

from raywarp.core.trace.optix_backend import OptixBackend
from raywarp.core.trace.software_backend import SoftwareBackend

if TYPE_CHECKING:
  import warp as wp

  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.kernels.abi.rays import TraceFeatures, Tracer
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.platform.device import DeviceCaps
  from raywarp.core.pipeline.warmup import FirstFrame, Warmup
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.shade.gi.volume import GiRays
  from raywarp.core.shade.lights.flood import Pass
  from raywarp.core.shade.lights.punctual import Lights
  from raywarp.core.shade.shadows.area import AreaLaunch
  from raywarp.core.trace.context import FrameContext
  from raywarp.core.trace.frame_structs import Inputs
  from raywarp.core.trace.rays.primary import CameraModel

# tracer(lit, g, npair): the light grid's visibility test of the `npair` (lattice point, light)
# pairs of one round of its flood, run on a backend's hardware (`light_grid.build`).
type GridTracer = Callable[[Lights, Pass, int], None]


class TraceBackend(Protocol):
  """Traces one Renderer's rays (see the module doc for the contract)."""

  name: ClassVar[str]
  # The meshes are traced through the reframed software mesh BVHs (`Features.mesh_frames`).
  mesh_frames: ClassVar[bool]
  # spp > 1 renders one supersampled image (the temporal features then need spp 1).
  supersamples: ClassVar[bool]
  # The instance shapes it traces (besides deformables): what a scene-free build compiles
  # (`core/build/variants/tiers.py` SCENE_FREE "shape_types").
  shape_types: ClassVar[tuple[int, ...]]
  # The background compile of the current kernels (`warmup.start`; None before `prepare`).
  warm: Warmup | None

  @staticmethod
  def device_reason(caps: DeviceCaps) -> str | None:
    """None if this backend can run on the device of `caps`, else why not."""
    ...

  @staticmethod
  def scene_reason(scene: GpuScene) -> str | None:
    """None if this backend can trace every geometry of `scene`, else what it cannot."""
    ...

  @staticmethod
  def analytic_planes(scene: GpuScene, layout: Layout) -> int:
    """How many planes its queries intersect analytically beside its structures (their loop
    form: `Features.planes`)."""
    ...

  @staticmethod
  def preferred(caps: DeviceCaps) -> bool:
    """Whether traversal="auto" picks this backend on the device of `caps` (for the scenes it can
    trace)."""
    ...

  @staticmethod
  def first_frame_kernels(f: Features, model: CameraModel, caps: DeviceCaps) -> FirstFrame:
    """The backend's own Warp kernels that a frame of `f` launches with cameras of model `model`
    on the device of `caps` (`warmup.first_frame`)."""
    ...

  @staticmethod
  def hardware_probes(f: Features) -> bool:
    """Whether every GI probe ray of build `f` (bake and updates, also inside a capture) is traced
    by this backend's hardware (`gi_tracer`): no software probe-ray kernel loads."""
    ...

  def tracer(self, f: TraceFeatures, cull: bool) -> Tracer:
    """The ray queries of `f`; `cull`: the closest / any queries cull back faces."""
    ...

  def configure(self, ctx: FrameContext) -> None:
    """(Re)allocates the structures after `ctx.features` changed (host only)."""
    ...

  def prepare(self, ctx: FrameContext, *, wait: bool = True) -> None:
    """Builds / compiles what `ctx.features` needs; `wait`: until the first frame can run."""
    ...

  def refit(self, ctx: FrameContext, frame: Frame) -> None:
    """Fits the structures to the frame's poses before the frame's other work."""
    ...

  def software_bvh(self, ctx: FrameContext, frame: Frame) -> None:
    """Makes the software scene BVH (`ctx.software`: world 0's, every world's with GI) current for
    the software ray kernels."""
    ...

  def software_meshes(self, ctx: FrameContext) -> None:
    """Builds the mesh BLASes the software ray kernels read (`ctx.software`)."""
    ...

  def grid_tracer(self, ctx: FrameContext, frame: Frame, see_through: bool) -> GridTracer | None:  # positional: baked.py's call
    """The light grid's visibility test on this backend's hardware, or None (software kernel).

    `frame`: world 0's frame for the grid's rays; `see_through`: it hides the instances that move
    or change per world (`baked._see_through_dynamic`), so it is not the render's frame.
    """
    ...

  def gi_tracer(self, ctx: FrameContext, frame: Frame) -> GiRays | None:
    """The GI bake's probe and static rays on this backend's hardware over `frame`'s structures,
    or None (the software kernels)."""
    ...

  def area_shadows(self, ctx: FrameContext, frame: Frame, launch: AreaLaunch) -> None:
    """Traces the soft area shadows' pass of the frame just rendered (`shade/shadows/area.py`:
    one unit per pixel of every world) with the render's shadow rays."""
    ...

  def launch(self, ctx: FrameContext, frame: Frame, seed: wp.array[wp.int32, [1]]) -> Inputs | None:
    """Renders the frame into the renderer's outputs; the launch's Inputs (software) or None."""
    ...

  def join_background(self) -> None:
    """Waits, without device work, for what `prepare(wait=False)` started in the background (the
    first frame's Warp modules, OptiX modules created ahead); failures are left to the render."""
    ...

  def close(self) -> None:
    """Releases the backend's structures (captured graphs using them then raise)."""
    ...


# Registered backends, in traversal="auto" order (the last one runs everywhere and traces
# everything).
BACKENDS: Final[tuple[type[TraceBackend], ...]] = (OptixBackend, SoftwareBackend)


def named(traversal: str) -> type[TraceBackend]:
  """The registered backend called `traversal` (a `traversal` setting or `Features.traversal`).

  Raises:
    ValueError: for a name no registered backend has.
  """
  for backend in BACKENDS:
    if backend.name == traversal:
      return backend
  raise ValueError(f"no trace backend {traversal!r}; registered: {', '.join(repr(b.name) for b in BACKENDS)}")


def select(traversal: str, scene: GpuScene, caps: DeviceCaps) -> type[TraceBackend]:
  """The backend of the `traversal` setting for `scene` on the device of `caps` (no fallback).

  "auto": the first registered backend the device prefers (`preferred`) that can trace the scene
  (a scene the preferred backend cannot trace goes to the next one: the software traversal traces
  everything); it must then run on the device. A named backend renders the scene or raises.

  Raises:
    ValueError: a named backend that cannot run on the device or trace the scene; "auto" whose
      pick for the scene cannot run on the device (an RT-core GPU without driver OptiX or the OptiX
      SDK headers: install them, or set performance.traversal="software").
  """
  if traversal != "auto":
    backend = named(traversal)
    if (reason := backend.device_reason(caps)) is not None:
      raise ValueError(f'traversal="{traversal}" cannot run on {caps.alias}: {reason}')
    if (reason := backend.scene_reason(scene)) is not None:
      raise ValueError(f'traversal="{traversal}" cannot trace this scene: {reason}')
    return backend
  for backend in BACKENDS:
    if not backend.preferred(caps) or backend.scene_reason(scene) is not None:
      continue
    reason = backend.device_reason(caps)
    if reason is not None:
      raise ValueError(
        f'traversal="auto" picks "{backend.name}" on {caps.alias} (by its capabilities), but it cannot run: '
        f'{reason}. Install what it needs, or set performance.traversal="software".'
      )
    return backend
  raise RuntimeError(f"no registered trace backend renders this scene on {caps.alias}")  # the last one traces everything
