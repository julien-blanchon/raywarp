"""The software trace backend: traces the camera rays through Warp BVHs (on every device) and
launches the megakernel (`backend.TraceBackend`).

Available on every device and for every scene. Its structures are the renderer's software scene
BVH (`SceneBvh`, `context.SoftwareScene.bvh`: the per-world tree refit every frame, the shared one)
and the mesh BLASes, optionally built in better-aligned frames (`mesh_frames`,
`trace.mesh_trace_frames`). Its queries (`tracer`) are also those of every software ray kernel (GI
probes, light grid), whatever backend renders the camera rays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, final

import numpy as np
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels import pixel
from raywarp.core.kernels.abi import rays as trace
from raywarp.core.pipeline import warmup
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import device_caps, host_copy, launch_bands
from raywarp.core.pipeline.platform.launches import BandedLaunch, CachedLaunch
from raywarp.core.shade.shadows import area as area_shadows
from raywarp.core.trace import frame_structs
from raywarp.core.trace.accel import bounds, query
from raywarp.core.trace.accel.bvh import SceneBvh

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.platform.device import DeviceCaps
  from raywarp.core.pipeline.warmup import FirstFrame, Warmup
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.shade.gi.volume import GiRays
  from raywarp.core.shade.shadows.area import AreaLaunch
  from raywarp.core.trace.backend import GridTracer
  from raywarp.core.trace.context import FrameContext
  from raywarp.core.trace.frame_structs import Inputs
  from raywarp.core.trace.rays.primary import CameraModel

# The CPU megakernel's bands (`launches.BandedLaunch`): at least this many rays each (a launch's
# fixed cost is microseconds, a ray's ~2 us per core), and this many per core: image rows differ
# in cost (sky vs. robot), so a core that ends its band early takes another.
_CPU_BAND: Final = 1024
_CPU_BANDS_PER_CORE: Final = 4


def _banded(f: Features, caps: DeviceCaps) -> bool:
  """Whether the megakernel of `f` runs in bands of rays over the cores (`pixel.megakernel`'s
  banded form): where launches are serial (the CPU), except in a counters build, whose per-world
  atomic sums are plain adds on the CPU (bands on other threads would race)."""
  return not caps.parallel_launch and not f.counters


@final
class SoftwareBackend:
  """Warp BVH traversal (see the module doc); one per Renderer.

  Attributes:
    warmed: the Features whose first-frame modules `warm` compiled (or compiles).
    warm: that compile while it runs (`warmup.Warmup`), None once waited for.
  """

  name: ClassVar[str] = "software"
  mesh_frames: ClassVar[bool] = True
  supersamples: ClassVar[bool] = False
  shape_types: ClassVar[tuple[int, ...]] = tuple(int(t) for t in ShapeType if t != ShapeType.FLEX)

  def __init__(self) -> None:
    """Nothing compiled yet."""
    self.warmed: Features | None = None
    self.warm: Warmup | None = None
    # The last frame's structs: (frame, context generation, structs).
    self._cache: tuple[Frame, tuple[int, int], Inputs] | None = None
    self._launch = CachedLaunch()  # the megakernel's eager launch (CUDA)
    self._bands = BandedLaunch()  # its launch in bands over the cores (CPU)

  @staticmethod
  def device_reason(caps: DeviceCaps) -> str | None:  # noqa: ARG004 (TraceBackend)
    """Always None: the software traversal runs on every device."""
    return None

  @staticmethod
  def scene_reason(scene: GpuScene) -> str | None:  # noqa: ARG004 (TraceBackend)
    """Always None: the software traversal traces every geometry."""
    return None

  @staticmethod
  def analytic_planes(scene: GpuScene, layout: Layout) -> int:  # noqa: ARG004 (TraceBackend)
    """The per-world planes kept out of the worlds' BVHs (`sharing.Layout.planes`: with shared
    scenery; without, planes are BVH members)."""
    return len(layout.planes)

  @staticmethod
  def preferred(caps: DeviceCaps) -> bool:  # noqa: ARG004 (TraceBackend)
    """Always True: traversal="auto" takes it wherever OptiX is not preferred, and for the
    geometry OptiX does not trace."""
    return True

  @staticmethod
  def tracer(f: trace.TraceFeatures, cull: bool) -> trace.Tracer:
    """The software queries of `f` (`query.software_tracer`)."""
    return query.software_tracer(f, cull=cull)

  @staticmethod
  def first_frame_kernels(f: Features, model: CameraModel, caps: DeviceCaps) -> FirstFrame:
    """The scene BVH refit's bounds, the meshes' trace frames (`mesh_frames`) and the megakernel
    of camera model `model` (the renderer's block size; `_banded`: its banded form); the
    traversal itself is funcs inlined by the megakernel."""
    frames = (warmup.kernel(trace.mesh_trace_frames),) if f.mesh_frames else ()
    default = (warmup.kernel(bounds.instance_bounds), *frames)
    return warmup.FirstFrame(default=default, render=(pixel.megakernel(f, model, banded=_banded(f, caps)),))

  @staticmethod
  def hardware_probes(f: Features) -> bool:  # noqa: ARG004 (TraceBackend)
    """False: the GI's software kernels are this backend's traversal."""
    return False

  def configure(self, ctx: FrameContext) -> None:
    """(Re)builds the scene BVH for the renderer's worlds, the mesh BLASes and the mesh-frame
    rotations."""
    f = ctx.features
    scene = ctx.scene
    sw = ctx.software
    if f.mesh_frames and ctx.layout.compact:
      raise ValueError("mesh BVH frames need per-world poses of every instance (MuJoCo's layout)")
    if sw.bvh is None or sw.bvh.nworld != ctx.nworld:  # (an OptiX renderer's is world 0's only)
      sw.bvh = SceneBvh(scene, ctx.nworld, ctx.layout)
    sw.mesh_bvh_id = scene.meshes.bvhs(tuning.MESH_BVH, used=ctx.tables.mesh_used, framed=f.mesh_frames)
    if f.mesh_frames and sw.rot_trace is None:
      sw.mesh_frame = wp.array(scene.meshes.frames, dtype=wp.mat33, device=scene.device)
      sw.rot_trace = wp.zeros((ctx.nworld, ctx.layout.ncolumn), dtype=wp.mat33, device=scene.device)
      types = host_copy(scene.instance_type)[: ctx.layout.ncolumn]
      mesh = (types == int(ShapeType.MESH)) | (types == int(ShapeType.SDF))
      # Per-world columns first: the shared ones are read from world 0's row only (`tree_pose`).
      per_world = ctx.layout.per_world[: ctx.layout.ncolumn]
      own = np.flatnonzero(mesh & per_world)
      cols = np.concatenate([own, np.flatnonzero(mesh & ~per_world)]).astype(np.int32)
      sw.mesh_columns = wp.array(cols, dtype=int, device=scene.device)
      sw.mesh_per_world = len(own)
    elif not f.mesh_frames:
      sw.rot_trace = None
      sw.mesh_columns = None

  def prepare(self, ctx: FrameContext, *, wait: bool = True) -> None:
    """Starts compiling the first frame's modules of `ctx.features` (once per Features); `wait` for
    them (and for an earlier warm-up of other Features still running)."""
    if self.warmed != ctx.features:
      self.warmed = ctx.features
      self.warm = warmup.start(ctx, self, after=self.warm)  # the first frame's modules, compiled in parallel
    if wait and self.warm is not None:
      self.warm.wait()
      self.warm = None

  def refit(self, ctx: FrameContext, frame: Frame) -> None:
    """Refits the scene BVH to the frame's poses; the meshes' trace frames follow them."""
    sw = ctx.software
    if sw.bvh is None:
      raise RuntimeError("SoftwareBackend.refit before configure (no scene BVH)")
    sw.bvh.refit(frame, ctx.settings.performance.rebuild_every)
    if sw.rot_trace is not None and sw.mesh_frame is not None and sw.mesh_columns is not None and sw.mesh_columns.shape[0]:
      wp.launch(
        trace.mesh_trace_frames,
        dim=(ctx.nworld, sw.mesh_columns.shape[0]),
        inputs=[frame.instance_mesh, frame.instance_rot, sw.mesh_columns, sw.mesh_per_world, sw.mesh_frame],
        outputs=[sw.rot_trace],
        device=ctx.scene.device,
      )

  def software_bvh(self, ctx: FrameContext, frame: Frame) -> None:
    """Nothing: the scene BVH the software ray kernels read is this backend's, refit every frame."""

  def software_meshes(self, ctx: FrameContext) -> None:
    """Nothing: the mesh BLASes are built by `configure`."""

  def grid_tracer(self, ctx: FrameContext, frame: Frame, see_through: bool) -> GridTracer | None:  # noqa: ARG002 (TraceBackend)
    """None: the light grid's software kernel is this backend's traversal."""
    return None

  def gi_tracer(self, ctx: FrameContext, frame: Frame) -> GiRays | None:  # noqa: ARG002 (TraceBackend)
    """None: the GI's software kernels are this backend's traversal."""
    return None

  def launch(self, ctx: FrameContext, frame: Frame, seed: wp.array[wp.int32, [1]]) -> Inputs:
    """Launches the megakernel over every (unit, ray) of the frame, a unit per world or per
    active-list entry (`FrameContext.nunit`; `_banded`: in bands of rays over the CPU's cores);
    returns its inputs."""
    inputs = self._structs(ctx, frame)
    f = ctx.features
    history = ctx.history
    outputs: list[object] = [inputs.out] if history is None else [inputs.out, history.struct]
    # The geometry-only kernel reads no material, light or environment struct and no seed.
    leading = [*inputs.structs, seed] if f.any_rgb else [inputs.geo, inputs.cams]
    device = ctx.scene.device
    dim = (ctx.nunit, ctx.nray)
    if not _banded(f, device_caps(device)):
      kernel = pixel.megakernel(f, ctx.camera_model)
      self._launch.launch(kernel, dim, inputs=leading, outputs=outputs, block_dim=f.block_dim, device=device)
    else:
      kernel = pixel.megakernel(f, ctx.camera_model, banded=True)
      self._bands.launch(
        kernel, dim, inputs=leading, outputs=outputs, device=device, min_band=_CPU_BAND, per_core=_CPU_BANDS_PER_CORE
      )
    return inputs

  def area_shadows(self, ctx: FrameContext, frame: Frame, launch: AreaLaunch) -> None:
    """The soft area shadows' tracing kernel over every pixel of the frame just rendered (its
    structs; on the CPU in bands of pixels over the cores)."""
    inputs = self._structs(ctx, frame)
    kernel = area_shadows.software_kernel(ctx.features)
    device = ctx.scene.device

    def band(a: int, b: int) -> None:
      """Pixels [a, b) of every world."""
      inputs_ab = [*inputs.structs, inputs.out, launch, a]
      wp.launch(kernel, dim=(ctx.nworld, b - a), inputs=inputs_ab, device=device)

    launch_bands(kernel.module, device=device, n=launch.npix, launch=band, min_band=area_shadows.BAND)

  def _structs(self, ctx: FrameContext, frame: Frame) -> Inputs:
    """The frame's structs, rebuilt only when an array identity or the context's generation
    changed (~0.1 ms otherwise)."""
    cached = self._cache
    if cached is not None and cached[1] == ctx.generation and frame.same(cached[0]):
      return cached[2]
    inputs = frame_structs.build(ctx, frame)
    self._cache = (frame, ctx.generation, inputs)
    return inputs

  def join_background(self) -> None:
    """Waits for the first frame's Warp modules (`TraceBackend.join_background`)."""
    if self.warm is not None:
      self.warm.join()

  def close(self) -> None:
    """Waits for a running first-frame compile (`warm`) and drops it; its failures are raised by
    `prepare(wait=True)`, not here."""
    if self.warm is not None:
      self.warm.join()
      self.warm = None
