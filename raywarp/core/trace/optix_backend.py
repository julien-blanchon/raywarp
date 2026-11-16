"""The OptiX trace backend: traces the camera rays on the GPU's RT cores through one Renderer's
OptiX build, rebuilt when its spec changes (`backend.TraceBackend`).

A thin owner of `launch.OptixRenderer` (acceleration structures, programs, pipelines): `prepare`
builds it for the renderer's current spec (`launch.spec_key`) and starts a frame, `launch` traces
the frame through it. The renderer's software ray kernels (GI probes, a light grid built inside a
CUDA graph capture) trace a one-world software scene BVH built on first use (`software_bvh`,
`software_meshes`).

Nothing here loads driver OptiX or the programs' modules at import time: `device_reason` reads the
device's `DeviceCaps.optix` and `scene_reason` the scene; the rest runs only for a Renderer that
selected this backend.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar, final

import numpy as np
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.pipeline import warmup
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import device_caps, host_copy
from raywarp.core.trace.accel.bvh import SceneBvh

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.kernels.abi.rays import TraceFeatures, Tracer
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.platform.device import DeviceCaps
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.shade.gi.volume import GiRays
  from raywarp.core.shade.shadows.area import AreaLaunch
  from raywarp.core.trace.backend import GridTracer
  from raywarp.core.trace.context import FrameContext
  from raywarp.core.trace.frame_structs import Inputs
  from raywarp.core.trace.optix.launch import OptixRenderer
  from raywarp.core.trace.rays.primary import CameraModel


@final
class OptixBackend:
  """Hardware ray tracing through OptiX (see the module doc); one per Renderer.

  Attributes:
    optix: the current build (None before the first `prepare` or after `close`).
    warm: the first frame's Warp modules compiling beside a new build, None once waited for.
  """

  name: ClassVar[str] = "optix"
  mesh_frames: ClassVar[bool] = False
  supersamples: ClassVar[bool] = True
  # Every shape (height fields, SDFs as triangles); deformables are added by the scene (`resolve`).
  shape_types: ClassVar[tuple[int, ...]] = tuple(int(t) for t in ShapeType if t != ShapeType.FLEX)

  def __init__(self) -> None:
    """No build yet (`prepare` makes it)."""
    self.optix: OptixRenderer | None = None
    self.warm: warmup.Warmup | None = None

  @staticmethod
  def device_reason(caps: DeviceCaps) -> str | None:
    """None if OptiX runs on the device of `caps` (`DeviceCaps.optix`), else why not."""
    return caps.optix.reason if not caps.optix.available else None

  @staticmethod
  def scene_reason(scene: GpuScene) -> str | None:
    """None if OptiX traces every geometry of `scene`, else what it cannot (`_unsupported`)."""
    reason = _unsupported(scene)
    return None if reason is None else f"the scene has {reason}"

  @staticmethod
  def analytic_planes(scene: GpuScene, layout: Layout) -> int:  # noqa: ARG004 (TraceBackend)
    """Every rendered plane (`accel.rendered_planes`: planes are never instanced)."""
    from raywarp.core.trace.optix.accel import rendered_planes  # noqa: PLC0415 (OptiX host code: on use)

    return len(rendered_planes(scene.rendered, np.asarray(host_copy(scene.instance_type), np.int32)))

  @staticmethod
  def preferred(caps: DeviceCaps) -> bool:
    """traversal="auto" takes OptiX on GPUs with RT cores (its gain is the traversal hardware)."""
    return caps.rt_cores

  @staticmethod
  def first_frame_kernels(f: Features, model: CameraModel, caps: DeviceCaps) -> warmup.FirstFrame:  # noqa: ARG004 (TraceBackend)
    """The Warp kernels beside the frame's programs (those `pipeline.PtxCompile` compiles: the
    build's, and a lens model's module, `programs.build_lens`): the structures' instance, curve
    and primitive writers (per-world and shared), their module-level kernels (instances, mesh
    inputs, dynamic bodies, micromaps), the launch-parameter store and, supersampled, the
    downsample: a bundle holds them, so a scene's first frame compiles none."""
    from raywarp.core.trace.optix import accel, dynamic, instances, mesh_inputs, supersample
    from raywarp.core.trace.optix.device import params  # noqa: PLC0415
    from raywarp.core.trace.optix.omm import kernels as omm_kernels

    default = [instances.write_instances(shared=shared, world_optics=f.world_optics) for shared in (False, True)]
    writers = (instances.write_curves, instances.write_primitive_bounds)
    default += [w(shared=shared) for w in writers for shared in (False, True)]
    if f.world_optics:
      default.append(warmup.kernel(accel.write_world_flags))  # the per-world instance flags
    default += [k for m in (instances, mesh_inputs, dynamic, omm_kernels) for k in warmup.module_kernels(m)]
    default.append(warmup.kernel(params.store_words))
    if f.spp > 1 and f.any_rgb:
      default.append(supersample.downsample_kernel(f))
    return warmup.FirstFrame(default=tuple(default))

  @staticmethod
  def hardware_probes(f: Features) -> bool:
    """Whether the build has the GI programs (`programs.gi_programs`): they trace every probe ray,
    a captured update's too (`launch.OptixRenderer.gi_tracer`)."""
    from raywarp.core.trace.optix.device import programs  # noqa: PLC0415

    return programs.gi_programs(f)

  def tracer(self, f: TraceFeatures, cull: bool) -> Tracer:  # positional `cull`: TraceBackend's
    """The OptiX queries of `f` over the current build's structures.

    Raises:
      RuntimeError: before the first `prepare` (no build).
    """
    if self.optix is None:
      raise RuntimeError("OptixBackend.tracer needs a build (`prepare`)")
    return self.optix.tracer(f, cull)

  def configure(self, ctx: FrameContext) -> None:
    """No software mesh BLASes until a software ray kernel needs them (`software_meshes`)."""
    ctx.software.mesh_bvh_id = wp.zeros(max(ctx.scene.meshes.nmesh, 1), dtype=wp.uint64, device=ctx.scene.device)
    ctx.software.rot_trace = None

  def prepare(self, ctx: FrameContext, *, wait: bool = True) -> None:
    """Builds the OptiX renderer of the context's spec if needed (structures; programs compiling).

    `Renderer()` calls it with `wait=False`: the programs and the first frame's Warp modules then
    compile while the caller goes on. `Renderer.render` calls it before the frame's other work (GI
    probes): it waits for those Warp modules (the programs' compile overlaps the GI step) and
    starts the frame (its first `OptixRenderer.refresh` writes and builds the structures).

    Raises:
      RuntimeError: when called inside a capture other than `Renderer.capture`'s, or before the
        first eager render (or after such a change).
    """
    from raywarp.core.trace.optix import graph, launch  # the OptiX stack: only for a renderer that selected OptiX

    key = launch.spec_key(ctx)
    if device_caps(ctx.scene.device).capturing():
      if graph.recording(wp.get_stream(ctx.scene.device)) is None:
        raise RuntimeError(
          'traversal="optix" can only be captured by Renderer.capture(step, sim) (put the simulation step '
          "in `step`): stream capture cannot record OptiX launches, so the graph is cut around them"
        )
      if self.optix is None or self.optix.key != key:
        raise RuntimeError(
          'traversal="optix": render once outside the CUDA graph capture first (it compiles the OptiX '
          "programs and builds the acceleration structures; again after changing a setting)"
        )
    if self.optix is not None and self.optix.key != key:
      if self.optix.rendered:  # built by Renderer() and never used: no warning
        warnings.warn("a compiled-in setting changed since the OptiX structures were built; rebuilding them", stacklevel=3)
      self.optix.close()
      self.optix = None
    if self.optix is None:
      # The first frame's other Warp modules compile in parallel meanwhile (the programs' own
      # compile goes on until the first trace, see `pipeline.PtxCompile`).
      self.warm = warmup.start(ctx, self, after=self.warm)
      self.optix = launch.OptixRenderer(ctx)
    self.optix.begin_frame()
    if wait and self.warm is not None:
      self.warm.wait()
      self.warm = None

  def refit(self, ctx: FrameContext, frame: Frame) -> None:
    """Nothing ahead of the frame: `launch` writes the instances and refits on the stream."""

  def software_bvh(self, ctx: FrameContext, frame: Frame) -> None:
    """Builds (on first use) and refits the software scene BVH of the software ray kernels: world
    0's for the light grid, every world's with GI (a probe group's shadow rays trace its own world:
    `shade.gi.groups`)."""
    sw = ctx.software
    nworld = ctx.nworld if ctx.effective.gi else 1
    if sw.bvh is None or sw.bvh.nworld != nworld:
      sw.bvh = SceneBvh(ctx.scene, nworld, ctx.layout)
    sw.bvh.refit(frame, ctx.settings.performance.rebuild_every)

  def software_meshes(self, ctx: FrameContext) -> None:
    """Builds the mesh BLASes of the software ray kernels (on first use: cached by the scene)."""
    ctx.software.mesh_bvh_id = ctx.scene.meshes.bvhs(tuning.MESH_BVH, used=ctx.tables.mesh_used)

  def grid_tracer(
    self, ctx: FrameContext, frame: Frame, see_through: bool
  ) -> GridTracer | None:  # positional `see_through`: TraceBackend's
    """The build's light-grid visibility program, when compiled (`OptixRenderer.grid_tracer`)."""
    return self.optix.grid_tracer(ctx, frame, see_through) if self.optix is not None else None

  def gi_tracer(self, ctx: FrameContext, frame: Frame) -> GiRays | None:
    """The GI bake's rays through the build's programs (`OptixRenderer.gi_tracer`)."""
    return self.optix.gi_tracer(ctx, frame) if self.optix is not None else None

  def launch(self, ctx: FrameContext, frame: Frame, seed: wp.array[wp.int32, [1]]) -> Inputs | None:
    """Traces a frame (see `Renderer.render`); None: the post passes read no software Inputs.

    Inside a CUDA graph capture this needs `Renderer.capture` (see `graph.py`) and a first render
    outside the capture, which compiles the programs and builds the acceleration structures. A
    change of the Features, nworld, tuning.optix_max_registers or of which instances may be cut
    out or get micromaps (`launch.spec_key`) rebuilds them at the next render;
    performance.rebuild_every is read every frame. `Renderer.render` calls `prepare` first,
    which starts the frame: not starting another one here keeps one structure build per frame,
    also when the programs traced the light grid.
    """
    if self.optix is None:
      raise RuntimeError("OptixBackend.launch needs a build (`prepare`)")
    self.optix.render(ctx, frame, seed)
    return None

  def area_shadows(self, ctx: FrameContext, frame: Frame, launch: AreaLaunch) -> None:  # noqa: ARG002 (TraceBackend)
    """The soft area shadows' program over every pixel of the frame just rendered, with its launch
    parameters (`OptixRenderer.area_shadows`)."""
    if self.optix is None:
      raise RuntimeError("OptixBackend.area_shadows needs a build (`prepare`)")
    self.optix.area_shadows(ctx, launch)

  def join_background(self) -> None:
    """Waits for the first frame's Warp modules and the build's OptiX modules created ahead
    (`TraceBackend.join_background`)."""
    if self.warm is not None:
      self.warm.join()
    if self.optix is not None:
      self.optix.join_ahead()

  def close(self) -> None:
    """Waits for the first frame's compiles and releases the build (waiting for the device).

    Graphs captured with it then raise. The first frame's compile failures are raised by
    `prepare(wait=True)`: this only joins them.
    """
    if self.warm is not None:
      self.warm.join()
      self.warm = None
    if self.optix is not None:
      self.optix.close()
      self.optix = None


def _unsupported(scene: GpuScene) -> str | None:
  """None if the OptiX backend can trace the scene's rendered geometry, else what it cannot: dim-1
  deformables (segments; surfaces are triangle GASes, `dynamic.py`)."""
  if any(int(d) == 1 for d in scene.deformables.dim):
    return "dim-1 flex geometry"
  return None
