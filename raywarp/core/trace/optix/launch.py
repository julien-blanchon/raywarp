"""One OptiX build of a Renderer (`OptixRenderer`): structures, programs, and the frame's launch.

A frame, enqueued on Warp's stream without host synchronisation: the structures' writes and
refits (`accel.OptixScene`, once per frame: the light grid's hardware build and the render share
them), a small Warp kernel copying the launch parameters into the device buffer (one copy per
launch chunk), then `optixLaunch` (`pipeline.Pipeline`). `OptixBackend` (`backend.py`) owns the
build for a Renderer and rebuilds it when its spec (`spec_key`) changes.

Bound values: a build's module compiles in the innermost stride of every launch-parameter array
(`params.bound_words`: element sizes, the same for every scene and frame of the build's layout),
the build word's copies (the classes, shapes, scene properties, settings and structures the scene
and configuration use) and the output word: one module and pipeline per value, created on its first
trace (`trace/optix/modules.py`: OptiX's disk cache, which the module pack seeds, or OptiX's
compile; never inside a capture). Nothing sized by the scene or the batch is bound.

CUDA graphs: stream capture cannot record OptiX calls (see `graph.py`). Inside
`Renderer.capture` the Warp kernels of a frame are captured as usual and the capture is cut at the
OptiX calls (refits, optixLaunch), which `RenderGraph.launch` enqueues between the graph segments.
"""

from __future__ import annotations

import dataclasses
import functools
import weakref
from typing import TYPE_CHECKING, Final, NamedTuple, final

import numpy as np
import warp as wp

from raywarp._base.warp_structs import warp_struct
from raywarp.core.build.variants import registers, tiers
from raywarp.core.kernels import probe as probe_lib
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.output_layout import layout_of
from raywarp.core.pipeline.platform.device import device_caps, host_copy
from raywarp.core.pipeline.platform.launches import struct_bytes
from raywarp.core.post.temporal.state import Temporal
from raywarp.core.shade.gi import kernels as gi_kernels
from raywarp.core.shade.shadows.area import AreaLaunch
from raywarp.core.trace import frame_structs, omm_sources
from raywarp.core.trace.optix import accel, graph, modules
from raywarp.core.trace.optix.context import address, ptx_arch
from raywarp.core.trace.optix.device import params as params_lib
from raywarp.core.trace.optix.device import programs
from raywarp.core.trace.optix.device.params import LaunchParams
from raywarp.core.trace.optix.pipeline import Pipeline, PtxCompile, forget, module_record, read_ptx
from raywarp.core.trace.optix.supersample import Supersampling, factor, module_outputs
from raywarp.core.trace.optix.supersample import program_features as supersample_program
from raywarp.core.trace.rays.primary import PINHOLE, CameraModel

if TYPE_CHECKING:
  from collections.abc import Callable

  from raywarp.core.build.kernel_cache.prebuilt import ModuleRecord
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.kernels.abi.output_layout import OutputLayout
  from raywarp.core.kernels.abi.rays import OptixFields, TraceFeatures, Tracer
  from raywarp.core.kernels.abi.structs import Cameras, EnvironmentArrays, OutputArrays
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.shade.lights import flood
  from raywarp.core.shade.lights.punctual import Lights
  from raywarp.core.trace.context import FrameContext
  from raywarp.core.trace.optix.context import OptixObject
  from raywarp.core.trace.optix.device.programs import SceneFlags
  from raywarp.settings import RenderSettings


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SpecKey:
  """What an OptixRenderer is built for (`spec_key`; it is rebuilt when this changes)."""

  nworld: int
  features: Features
  # The cameras' model: the render program the build launches (`programs.raygen_name`).
  camera: CameraModel
  # The build's OptiX register caps (shaded, light: `registers.register_caps`): each module's cap
  # follows from them and its word (`registers.optix_registers`).
  register_caps: tuple[int, int]
  cut_all: bool  # every instance with a material may be cut out (`_cut_all`)
  micromaps: bool  # cut-out meshes may get opacity micromaps (`omm_sources.allowed`)
  # The supersampling factor (`supersample.factor`: its image's buffers; 1: none).
  supersampling: int


# The tables that decide which instances are cut out (`accel.cut_instances`).
_CUT_TABLES: Final = frozenset({"material.alpha_cutoff", "instance.material"})


def _cut_all(ctx: FrameContext) -> bool:
  """Whether every instance with a material may be cut out: its cutoffs or material ids are
  per-world declarations, or live binding slots (MuJoCo's `geom_matid`: a write moves a geom onto
  a cut-out material without a rebuild), so the build cannot know the cut-out set. Moving material
  ids alone cut out nothing when no material has a cutoff (the static cutoffs all 0): a tier build
  with the alpha test (the surface class) then keeps the any-hit program off every instance (once
  enforced on all of a MuJoCo scene's instances, its prebuilt build ran 15x its own)."""
  varying = _CUT_TABLES & (ctx.per_world_fields | set(ctx.scene.binding.slots(None)))
  if "material.alpha_cutoff" in varying:
    return True
  if not varying:
    return False
  cutoff = ctx.scene.tables.get("material.alpha_cutoff")
  return cutoff is not None and cutoff.shape[-1] > 0 and _any_cutoff(cutoff)


# Whether a static cutoff table holds a cutoff > 0, per table (read back once: `_cut_all` runs at
# every frame's `spec_key`).
_CUTOFFS: Final[weakref.WeakKeyDictionary[wp.array, bool]] = weakref.WeakKeyDictionary()


def _any_cutoff(cutoff: wp.array) -> bool:
  """Whether the static table `cutoff` has a cutoff > 0 (cached per table)."""
  if cutoff not in _CUTOFFS:
    _CUTOFFS[cutoff] = bool(np.any(host_copy(cutoff) > 0.0))
  return _CUTOFFS[cutoff]


@final
class _FrameStructs(NamedTuple):
  """The structs of the last frame's launch parameters (`OptixRenderer._set_params`).

  Attributes:
    frame: the frame they were built from.
    key: (context generation, IAS handle, shared IAS handle) they were built with.
    cams: the cameras.
    hi: the rendered outputs (the supersampled image's with spp > 1, else `out`).
    out: the renderer's outputs.
  """

  frame: Frame
  key: tuple[tuple[int, int], int, int]
  cams: Cameras
  hi: OutputArrays
  out: OutputArrays


def spec_key(ctx: FrameContext) -> SpecKey:
  """What an OptixRenderer is built for (it is rebuilt when this changes; runtime settings are read
  per frame)."""
  return SpecKey(
    nworld=ctx.nworld,
    features=ctx.features,
    camera=ctx.camera_model,
    register_caps=ctx.register_caps,
    cut_all=_cut_all(ctx),
    micromaps=omm_sources.allowed(ctx),
    supersampling=factor(ctx),
  )


# Lab switches (benchmarks only). GENERIC_MODULES: modules created with fewer bound values, read
# from the launch parameters at run time instead (the same PTX and parameters, unfolded): "all"
# nothing bound, "words" the strides only, "build" the strides and the output word. FORCE_BITS:
# build-word bits (`toggles`, both halves) bound on in every copy that holds them, whatever the
# frame's word: the code of a feature the scene does not use stays live, as a run-time branch
# would leave it. Both measure what each bound value saves.
GENERIC_MODULES: str = ""
FORCE_BITS: int = 0


def _unbound(bound: tuple[tuple[int, str], ...], layout: params_lib.ParamsLayout) -> tuple[tuple[int, str], ...]:
  """A record's bound values without those `GENERIC_MODULES` leaves to run time."""
  if GENERIC_MODULES == "all":
    return ()
  struct = params_lib.params_type(layout)
  words = set(params_lib.word_offsets(struct).values())
  if GENERIC_MODULES == "words":
    words |= set(params_lib.output_offsets(struct).values())
  return tuple((o, v) for o, v in bound if o not in words)


def _forced(words: params_lib.BoundWords, layout: params_lib.ParamsLayout) -> params_lib.BoundWords:
  """`words` with `FORCE_BITS` set in every copy of the build word (`params.word_offsets`)."""
  offsets = {o: (path, half) for (path, half), o in params_lib.word_offsets(params_lib.params_type(layout)).items()}
  out = []
  for offset, value in words:
    if offset in offsets:
      path, half = offsets[offset]
      bits = FORCE_BITS & params_lib.copy_mask(path)
      bits = bits & toggles.HALF if half == "bits" else bits >> toggles.HIGH
      value = (int.from_bytes(value, "little", signed=True) | bits).to_bytes(4, "little", signed=True)
    out.append((offset, value))
  return tuple(out)


def record_of(
  f: Features, flags: SceneFlags, layout: params_lib.ParamsLayout, words: params_lib.BoundWords, caps: tuple[int, int]
) -> ModuleRecord:
  """The module record of build `f`'s programs (program flags `flags`, launch-parameter layout
  `layout`) binding `words` (`params.bound_words`): the register cap of its build word under the
  build's caps `caps` (`registers.optix_registers`) and the shapes it traces."""
  word = params_lib.geometry_word(words, params_lib.params_type(layout))
  cap = registers.optix_registers(caps, f, word)
  return module_record(flags, cap, words, shapes=programs.traced_shapes(f, word))


def expected_record(
  f: Features, flags: SceneFlags, layout: params_lib.ParamsLayout, word: int, outputs: OutputLayout, caps: tuple[int, int]
) -> ModuleRecord:
  """The record of the module a frame of build `f` binds over `layout` for build word `word`
  (`programs.build_word`) and output word `outputs` (a GI layout holds none): what its launch
  parameters' words will be (`params.bound_words`: strides from the types, the words' copies).
  One rule for a build's modules created ahead (`OptixRenderer`) and the prebuilt bundle's census
  of module records."""
  p = params_lib.params_type(layout)()
  params_lib.set_words(p, word)
  params_lib.set_outputs(p, outputs)
  return record_of(f, flags, layout, params_lib.bound_words(p), caps)


@final
class Unit(NamedTuple):
  """One OptiX module a frame binds.

  Attributes:
    key: its PTX's key (`programs.ptx_key`).
    record: its options and bound values (`expected_record`).
    program: "render" (the render programs', a lens model's), "gi" (`programs.build_gi`) or
      "area" (`programs.build_area`).
  """

  key: str
  record: ModuleRecord
  program: str


def frame_units(
  build: Features,
  effective: Features,
  settings: RenderSettings,
  scene: SceneFlags,
  *,
  model: CameraModel,
  arch: int,
  caps: tuple[int, int],
) -> tuple[Unit, ...]:
  """The modules a frame of build `build` binds (configuration `effective`, `settings`, cameras of
  model `model`, register caps `caps`) over structures `scene` (`arch`: the PTX target,
  `context.ptx_arch`): the render
  programs', a lens model's render program (`programs.build_lens`, the build's record), with
  GI the GI programs', with `shadows.area_rays` on the soft area shadows' program. One rule for
  the modules a build creates ahead (`OptixRenderer`), those `Renderer.warm` creates for settings
  one change away (`OptixRenderer.create_modules`) and the census's records (`expected_record`)."""
  supersampled = build.spp > 1 and build.any_rgb
  f = supersample_program(build) if supersampled else build
  flags = programs.program_flags(f, scene)
  word = programs.build_word(f, effective, scene)
  outputs = layout_of(module_outputs(build, settings.written_outputs()))
  render = programs.ptx_key(f, flags, supersampled=supersampled, arch=arch)
  record = expected_record(f, flags, programs.layout(f, flags), word, outputs, caps)
  units = [Unit(render, record, "render")]
  if model != PINHOLE:
    units.append(Unit(programs.ptx_key(f, flags, supersampled=supersampled, arch=arch, lens=model), record, "render"))
  if programs.gi_programs(f):
    gi = programs.ptx_key(f, flags, supersampled=False, arch=arch, gi=True)
    units.append(Unit(gi, expected_record(f, flags, programs.gi_layout(f, flags), word, outputs, caps), "gi"))
  if settings.area_rays():
    area = programs.ptx_key(f, flags, supersampled=False, arch=arch, area=True)
    units.append(Unit(area, expected_record(f, flags, programs.area_layout(f, flags), word, outputs, caps), "area"))
  return tuple(units)


@final
class OptixRenderer:
  """Acceleration structures and pipeline of a Renderer (built on its first OptiX render).

  With spp n > 1 (and shaded outputs) the program renders the n x n samples as an n x resolution
  spp-1 image that a Warp kernel then downsamples (`supersample.py`).

  Attributes:
    key: the build's spec (`spec_key`).
    scene: the acceleration structures.
    module: the programs' Warp module (None: their PTX was read from raywarp's cache or the
      bundle, the programs never defined).
    raygen: the render program it launches: its cameras' model's (`programs.raygen_name`).
    pipeline: the last frame's OptiX pipeline (`_pipeline`; None until the first trace).
    rendered: a frame was rendered with this build.
    fresh: the structures hold the current frame (written and built once per frame).
    closed: released (`close`); graphs captured with it raise.
  """

  def __init__(self, ctx: FrameContext) -> None:
    """Builds the structures and starts compiling the programs (`pipeline.PtxCompile`)."""
    f = ctx.features
    self.key = spec_key(ctx)
    self._caps = ctx.register_caps  # the spec's
    self.device = ctx.scene.device
    n = factor(ctx)
    self.supersampling = Supersampling(ctx, n) if n > 1 else None
    f_prog = self.supersampling.features if self.supersampling is not None else f
    scene = ctx.scene
    self.scene = accel.OptixScene(
      scene,
      ctx.nworld,
      ctx.layout,
      device=self.device,
      mesh_used=ctx.tables.mesh_used,
      mesh_frame=None if scene.meshes.frames is None else scene.meshes.frames.astype(np.float32),
      alpha_cutout=f.alpha_cutout,
      translucent_shadows=f.translucent_shadows,
      cut_all=self.key.cut_all,
      opacity_micromaps=self.key.micromaps,
      optics=(ctx.tables.shadow_transmit, ctx.tables.twosided) if f.world_optics else None,
    )
    self.features = f_prog
    # What the programs are compiled for: every geometry kind for a scene-free build (its PTX and
    # module are the same for every scene); the pipeline links `self.scene`'s groups only.
    self.flags = programs.program_flags(f_prog, self.scene)
    supersampled = self.supersampling is not None
    # The render's launch: (unit, ray), a unit per world or per active-list entry (`programs` maps
    # them to the frame's worlds and order, `camera.launch_ids`); the rays per unit.
    self.nray = ctx.nray * n * n
    self.visibility = programs.visibility_program(f_prog)
    # The render program of the cameras' model: a lens model's in a module of its own
    # (`programs.build_lens`), compiled beside the build's.
    self.raygen = programs.raygen_name(ctx.camera_model)
    self.layout = programs.layout(f_prog, self.flags)
    self.params = params_lib.params_type(self.layout)()
    self._ptx_key = programs.ptx_key(f_prog, self.flags, supersampled=supersampled, arch=ptx_arch(self.device))
    # Whether the bundle should hold the programs (a miss warns).
    self._prebuilt = tiers.prebuilt(ctx.settings, f)
    ptx = self._compile(functools.partial(programs.build, f_prog, self.flags, supersampled=supersampled), self._ptx_key)
    self._ptx: PtxCompile | None = ptx
    self.module = self._ptx.module
    model = ctx.camera_model
    self._lens_key: str | None = None
    self._lens_ptx: PtxCompile | None = None
    self._lens_bytes: bytes | None = None
    if model != PINHOLE:
      self._lens_key = programs.ptx_key(f_prog, self.flags, supersampled=supersampled, arch=ptx_arch(self.device), lens=model)
      lens = functools.partial(programs.build_lens, f_prog, self.flags, model, supersampled=supersampled)
      self._lens_ptx = self._compile(lens, self._lens_key)
    # The GI bake's programs (`programs.build_gi`): a module and pipeline of their own, so that the
    # render's are a build's without GI; compiled on the first bake launch (`_gi_pipeline`), not
    # here: a bake restored from the cache traces no ray, and a compile running behind the frames
    # would slow them.
    gi = programs.gi_programs(f_prog)
    self.gi_layout = programs.gi_layout(f_prog, self.flags) if gi else None
    self._gi_ptx: PtxCompile | None = None
    # The GI programs' pipelines by bound words (`_gi_pipeline`).
    self._gi: dict[params_lib.BoundWords, Pipeline] = {}
    self.pipeline: Pipeline | None = None
    # The build's pipelines by bound words (`params.bound_words`: one per build and output word).
    self._pipelines: dict[params_lib.BoundWords, Pipeline] = {}
    self._ptx_bytes: bytes | None = None
    self.part_keys: dict[str, bytes] = {}
    # The temporal state's launch slot (`params.write_struct`), made at the first temporal frame.
    self.tmp_slot: wp.array[Temporal, [int]] | None = None
    self._frame: _FrameStructs | None = None
    self.rendered = self.fresh = self.closed = False
    # The structures `refresh` wrote inside a capture await their build (the frame's first cut).
    self._unbuilt = False
    # The frame and probe colors the structures were written for apart (`_bake_structures`: the
    # GI bake's), None while they hold the render's frame.
    self._apart: tuple[int, int] | None = None
    # One launch struct per GI launch kind (`params.GI_LAUNCHES`), written on the device before
    # each launch (`OptixGiRays.launch`: no host copy, also inside a capture), and the empty arrays
    # of the kinds a launch does not use.
    kinds = params_lib.GI_LAUNCHES.items() if gi else ()
    self.gi_slots = {name: wp.zeros(1, dtype=kind, device=self.device) for name, kind in kinds}
    self.gi_empty = {name: wp.empty(0, dtype=kind, device=self.device) for name, kind in kinds}
    # The soft area shadows' program (`programs.build_area`): a module and pipelines of their own,
    # compiled on the first launch of the pass (`area_shadows`); its launch struct's slot.
    self.area_layout = programs.area_layout(f_prog, self.flags)
    self._area_ptx: PtxCompile | None = None
    self._area: dict[params_lib.BoundWords, Pipeline] = {}
    self._area_slot: wp.array[AreaLaunch, [int]] | None = None
    # The modules the first frame binds, created ahead and together (`_start_ahead`).
    self._ahead: dict[tuple[str, ModuleRecord], modules.Ahead] = self._start_ahead(ctx)

  def _start_ahead(self, ctx: FrameContext) -> dict[tuple[str, ModuleRecord], modules.Ahead]:
    """Starts creating, each on its own thread, the modules the first frame binds whose PTX is at
    hand (bundled or cached; a compiling PTX is not waited for): the render's, the lens model's
    and the GI programs' (`frame_units`). One OptiX module is one serial compile task: created
    together, a cold first frame waits for the slowest instead of their sum; `_module` takes them.
    The GI programs' PTX is only read: its compile waits for a bake that traces (`_gi_pipeline`).
    A failure cancels the creations it started (no module outlives a build that never was)."""
    ahead: dict[tuple[str, ModuleRecord], modules.Ahead] = {}
    own = {self._ptx_key: self._ptx, self._lens_key: self._lens_ptx}
    try:
      for unit in self.frame_modules(ctx):
        compiling = own.get(unit.key)
        code = (
          compiling.ready if compiling is not None else read_ptx(unit.key, device_arch=int(self.device.arch), expected=False)
        )
        if code is not None:
          ahead[unit.key, unit.record] = modules.Ahead(self.scene.ctx, code, unit.record)
    except BaseException:
      for started in ahead.values():
        started.cancel()
      raise
    return ahead

  def frame_modules(self, ctx: FrameContext) -> tuple[Unit, ...]:
    """The modules a frame of `ctx` binds over these structures (`frame_units`): this build's, or
    with another configuration's Features in `ctx` (`Renderer.warm`'s settings one change away)
    that build's."""
    return frame_units(
      ctx.features,
      ctx.effective,
      ctx.settings,
      self.scene,
      model=ctx.camera_model,
      arch=ptx_arch(self.device),
      caps=ctx.register_caps,
    )

  def create_modules(self, units: list[Unit], *, workers: int) -> tuple[int, int, float]:
    """Creates the modules `units` whose PTX is at hand (bundled or cached: never compiled) into
    OptiX's disk cache, `workers` at a time (`modules.create_each`); (created, without PTX, wall
    seconds)."""
    arch = int(self.device.arch)
    jobs = [(code, u.record) for u in units if (code := read_ptx(u.key, device_arch=arch, expected=False)) is not None]
    created, seconds = modules.create_each(self.scene.ctx, jobs, workers=workers)
    return created, len(units) - len(jobs), seconds

  def _gi_key(self) -> str:
    """The PTX key of the GI programs (compiled for the render's program flags: every geometry
    kind in a scene-free build, as the render's)."""
    return programs.ptx_key(self.features, self.flags, supersampled=False, arch=ptx_arch(self.device), gi=True)

  def _compile(self, build: Callable[[], wp.Module], key: str) -> PtxCompile:
    """The PTX of programs `build` (key `key`) for this device: cached, bundled or compiled."""
    return PtxCompile(build, ptx_arch(self.device), key, device_arch=int(self.device.arch), expected=self._prebuilt)

  def _launch(self, pipeline: Pipeline, params: LaunchParams, n: int, program: str | None) -> None:
    """Stores `params` and launches `program` of `pipeline` over `n` units (device stream; inside
    `Renderer.capture` the launch is cut out of the graph, `_cut`)."""
    pipeline.store(params, n, 1, program)
    if device_caps(self.device).capturing():
      self._cut(pipeline, (n, 1), program)
    else:
      pipeline.trace(wp.get_stream(self.device), n, 1, program)

  def _cut(self, pipeline: Pipeline, units: tuple[int, int], program: str | None) -> None:
    """Cuts the capture at a launch of `program` of `pipeline` over `units` (`graph_work` runs it
    at each replay, `graph.Segments.cut`); the frame's first cut also builds the structures that
    `refresh` wrote."""
    segments = graph.recording(wp.get_stream(self.device))
    if segments is None:
      raise RuntimeError("an OptiX launch is captured only inside Renderer.capture (OptixBackend.prepare)")
    handle = self.scene.worlds.handle
    build, self._unbuilt = self._unbuilt, False
    apart = build and self._apart is not None  # written apart (`_bake_structures`): rebuilt apart
    work = functools.partial(
      self.graph_work, pipeline=pipeline, handle=handle, units=units, build=build, apart=apart, program=program
    )
    segments.cut(work, check=functools.partial(self._check_graph, pipeline, handle))

  def tracer(self, f: TraceFeatures, cull: bool) -> Tracer:
    """The OptiX queries of `f` over this build's structures (`programs.scene_tracer`)."""
    return programs.scene_tracer(self.flags, f, cull=cull)

  def _fill(self, ctx: FrameContext, p: LaunchParams, geo: OptixFields) -> None:
    """Sets the structures' fields of launch parameters `p` and of their geometry `geo` (handles,
    cell, tables, the build word's geometry copy for `ctx`'s configuration)."""
    sc, layout = self.scene, self.layout
    toggles.store(geo, programs.build_word(self.features, ctx.effective, sc))
    geo.rt_handle = wp.uint64(sc.worlds.handle)
    geo.rt_cell = sc.tables.cell
    geo.rt_grid = sc.tables.grid
    geo.rt_shared = wp.uint64(sc.shared.handle)
    geo.plane_ids = sc.plane_table
    geo.flex_part = sc.flex_part
    # A scene-free build's layout holds every table; those of kinds the scene lacks stay empty
    # (their programs are not linked: `pipeline.Pipeline`).
    if layout.world_curves and sc.has_world_curves:
      p.curve_instance = sc.worlds.curve_instance
    if layout.world_primitives and sc.primitive_types:
      p.prim_code = sc.worlds.prim_code
    if layout.shared_curves and sc.has_shared_curves:
      p.shared_curve_instance = sc.shared.curve_instance
    if layout.shared_primitives and sc.shared_primitive_types:
      p.shared_prim_code = sc.shared.prim_code
    if layout.mesh and sc.has_mesh:
      p.face_normal = sc.tables.meshes.face_normal

  def _pipeline(self, params: LaunchParams) -> Pipeline:
    """The build's pipeline (made on its first use, outside a capture: its shader-binding-table
    uploads would be recorded, not run; `Renderer.capture`'s eager frame makes it, see
    `trace/optix/modules.py`), the arrays of launch parameters `params` checked against the strides
    it binds (`params.bound_words`, at every launch: a vectorised read of the struct's bytes); one
    pipeline per value of the bound words.

    Raises:
      RuntimeError: a first pipeline needed inside a capture.
      ValueError: an array of `params` the bound strides would misread (`params.bound_words`).
    """
    words = params_lib.bound_words(params)
    pipeline = self._pipelines.get(words)
    if pipeline is None:
      if device_caps(self.device).capturing():
        raise RuntimeError("an OptiX pipeline is needed inside a capture: render once outside it first")
      pipeline = self._pipelines[words] = self._new_pipeline(words)
    return pipeline

  def _ptx_result(self) -> bytes:
    """The programs' PTX (waits for its compile)."""
    if self._ptx_bytes is None:
      if self._ptx is None:
        raise RuntimeError("the OptiX renderer was closed: its programs are gone")
      self._ptx_bytes = self._ptx.result()
      self._ptx = None
    return self._ptx_bytes

  def _lens_result(self) -> bytes:
    """The lens model's render program's PTX (waits for its compile)."""
    if self._lens_bytes is None:
      if self._lens_ptx is None:
        raise RuntimeError("the OptiX renderer was closed (or has no lens camera): its lens program is gone")
      self._lens_bytes = self._lens_ptx.result()
    return self._lens_bytes

  def _module(
    self, key: str, ptx: bytes, words: params_lib.BoundWords, layout: params_lib.ParamsLayout
  ) -> tuple[ModuleRecord, OptixObject]:
    """The record and OptiX module of PTX `ptx` (key `key`, launch-parameter layout `layout`)
    binding `words` at the register cap of its word (`registers.optix_registers`;
    `trace/optix/modules.py`; an expected build's module outside its bundle entry warns)."""
    if FORCE_BITS:
      words = _forced(words, layout)
    record = record_of(self.features, self.flags, layout, words, self._caps)
    if GENERIC_MODULES:
      record = dataclasses.replace(record, bound=_unbound(record.bound, layout))
    # A lens model's module is the census's choice (its scenes' cameras): not checked.
    modules.check(int(self.device.arch), key, record, expected=self._prebuilt and key == self._ptx_key)
    ahead = self._ahead.pop((key, record), None)
    return record, ahead.result() if ahead is not None else modules.create(self.scene.ctx, ptx, record)

  def join_ahead(self) -> None:
    """Waits for the modules created ahead (`_start_ahead`; each stays for its first use)."""
    for ahead in list(self._ahead.values()):  # the render thread may take one meanwhile
      ahead.join()

  def _new_pipeline(self, words: params_lib.BoundWords) -> Pipeline:
    """The build's pipeline binding `words`."""
    extras = (programs.VISIBILITY,) if self.visibility else ()
    anyhit = programs.anyhit_programs(self.features, self.scene)
    ptx = self._ptx_result()
    try:
      record, module = self._module(self._ptx_key, ptx, words, self.layout)
      lens = None
      if self._lens_key is not None:
        code = self._lens_result()
        try:
          lens = (code, self._module(self._lens_key, code, words, self.layout)[1])
        except Exception:
          module.destroy()
          forget(self._lens_key)
          raise
      return Pipeline(
        ctx=self.scene.ctx,
        device=self.device,
        ptx=ptx,
        scene=self.scene,
        record=record,
        module=module,
        layout=self.layout,
        raygen=self.raygen,
        lens=lens,
        extras=extras,
        anyhit=anyhit,
      )
    except Exception:
      forget(self._ptx_key)  # a damaged cached PTX would fail every start: compiled again next time
      raise

  def begin_frame(self) -> None:
    """A new frame: its first `refresh` writes and builds the structures."""
    self.fresh = False
    self._apart = None

  def refresh(self, ctx: FrameContext, frame: Frame) -> None:
    """Writes and builds the structures for this frame, once (the build is left to the frame's
    first cut in a capture, `_cut`)."""
    if self.fresh:
      return
    self.scene.rebuild_every = ctx.settings.performance.rebuild_every
    self.scene.write(frame, visibility=ctx.features.visibility)
    self._apart = None
    self._unbuilt = device_caps(self.device).capturing()
    if not self._unbuilt:
      self.scene.build(wp.get_stream(self.device))
    self.fresh = True

  def _check_graph(self, pipeline: Pipeline, handle: int) -> str | None:
    """Why a graph captured with `pipeline` over the IAS `handle` cannot replay, or None (run by
    `graph.Segments` before the graph's first segment: a refused replay writes nothing)."""
    if self.closed:
      return "the OptiX renderer of this graph was rebuilt or closed since the capture: capture again"
    if pipeline.destroyed:
      return "the OptiX pipeline of this graph was released: capture again"
    if self.scene.worlds.handle != handle:
      return "the OptiX IAS handle changed after the graph was captured"
    return None

  def graph_work(
    self,
    stream: wp.Stream,
    *,
    pipeline: Pipeline,
    handle: int,
    units: tuple[int, int],
    build: bool,
    apart: bool,
    program: str | None,
  ) -> None:
    """The OptiX calls of a captured launch (refits, optixLaunch) on `stream`: `program` (None: the
    render) of the captured `pipeline` (whose launch-parameter buffer the graph's segment wrote),
    after building the structures the segment wrote (`build`; `apart`: a frame apart, the GI
    bake's, `_bake_structures`); run by `graph.Segments.launch`, after its checks
    (`_check_graph`)."""
    if apart:
      self.scene.rebuild_apart(stream)
    elif build:
      self.scene.build(stream)
    if build and self.scene.worlds.handle != handle:
      raise RuntimeError("the OptiX IAS handle changed after the graph was captured")
    pipeline.trace(stream, *units, program)

  def grid_tracer(self, ctx: FrameContext, frame: Frame, see_through: bool):
    """The light grid's visibility test traced by this build's programs, over world 0 of `frame`.

    `light_grid.build`'s `tracer`, or None: no visibility program, or inside a capture (the
    software kernel then runs). A cold build waits for its programs: the software kernel's
    grid differs from the programs' in a few (cell, light) pairs, so the image would depend on
    the kernel cache, and on big scenes it takes longer than the compile (a large interior: 6.2 s on
    L4 vs 0.1 s traced by the programs).
    `see_through`: `frame` hides the instances that move or change per world (not the render's
    frame): the structures are built for it apart (`accel.OptixScene.build_apart`) and rebuilt
    from the render's frame after; otherwise the grid shares the frame's structures.
    """
    if not self.visibility or device_caps(self.device).capturing():
      return None
    prepared: list[bool] = []

    def tracer(lit: Lights, g: flood.Pass, npair: int) -> None:
      """Traces the `npair` (lattice point, light) pairs of `g` (the structures, written for
      `frame` before the build's first round)."""
      if not prepared:
        if see_through:
          self.scene.build_apart(frame, wp.get_stream(self.device), visibility=ctx.features.visibility)
          self.fresh, self._apart = False, None  # the render writes its own frame again
        else:
          self.refresh(ctx, frame)
        prepared.append(True)
      p = params_lib.params_type(self.layout)()
      geo = frame_structs.optix_geometry(ctx, frame, shaded=self.layout.shaded)
      self._fill(ctx, p, geo)
      set_grid(p, geo, lit, self._outputs(ctx)[0])
      p.nworld, p.grid = ctx.nworld, g
      self._launch(self._pipeline(p), p, npair, programs.VISIBILITY)

    return tracer

  def gi_tracer(self, ctx: FrameContext, frame: Frame) -> OptixGiRays | None:
    """The GI bake's rays (probe rays and the placement's: `programs.GI_PROGRAMS`) traced by this
    build's GI pipeline (`_gi_pipeline`) over the structures of `frame` written apart with the
    probe geometry's colors (`_bake_structures`: the moving instances hidden), or None: no GI
    programs. Inside a capture its launches are cut out of the graph (`_launch`): a captured GI
    update traces here too, never by a software kernel."""
    return OptixGiRays(self, ctx, frame) if self.gi_layout is not None else None

  def gi_launch(self, program: str, params: LaunchParams, n: int) -> None:
    """Launches GI program `program` over `n` units with `params` (`OptixGiRays`), on the GI
    pipeline (`programs.PROBE` its main program)."""
    self._launch(self._gi_pipeline(params), params, n, None if program == programs.PROBE else program)

  def _gi_pipeline(self, params: LaunchParams) -> Pipeline:
    """The GI programs' pipeline (`programs.build_gi`; made on first use) binding the words of GI
    launch parameters `params` (`params.bound_words`: one pipeline per word value, as the
    render's), so that OptiX folds the parts of the scene and configuration it lacks."""
    if self.gi_layout is None:
      raise RuntimeError("a GI launch on an OptiX build without GI programs")
    words = params_lib.bound_words(params)
    pipeline = self._gi.get(words)
    if pipeline is not None:
      return pipeline
    if device_caps(self.device).capturing():
      raise RuntimeError("an OptiX GI pipeline is needed inside a capture: render once outside it first")
    key = self._gi_key()
    if self._gi_ptx is None:
      self._gi_ptx = self._compile(functools.partial(programs.build_gi, self.features, self.flags), key)
    ptx = self._gi_ptx.result()
    record, module = self._module(key, ptx, words, self.gi_layout)
    pipeline = self._gi[words] = Pipeline(
      ctx=self.scene.ctx,
      device=self.device,
      ptx=ptx,
      scene=self.scene,
      record=record,
      module=module,
      layout=self.gi_layout,
      raygen=programs.PROBE,
      extras=programs.GI_PROGRAMS[1:],
      anyhit=programs.anyhit_programs(probe_lib.probe_features(self.features), self.scene),
    )
    return pipeline

  def area_shadows(self, ctx: FrameContext, launch: AreaLaunch) -> None:
    """Launches the soft area shadows' program over every pixel of the frame just rendered
    (`ctx.nworld` x `launch.npix` units) with that frame's structs (`programs.build_area`)."""
    if self.supersampling is not None:
      raise RuntimeError("the soft area shadows' pass needs aa.spp=1 (RenderSettings checks it)")
    p = params_lib.params_type(self.area_layout)()
    # Every field of the render's parameters (its structs and the hit programs' tables), but the
    # chunk offset (`Pipeline.store`'s).
    for name in warp_struct(params_lib.params_type(self.layout)).vars:
      if name != "offset":
        setattr(p, name, getattr(self.params, name))
    if self._area_slot is None:
      self._area_slot = wp.zeros(1, dtype=AreaLaunch, device=self.device)
    params_lib.write_struct(launch, self._area_slot)
    p.area = self._area_slot
    p.nworld = ctx.nworld
    self._launch(self._area_pipeline(p), p, ctx.nworld * launch.npix, None)

  def _area_pipeline(self, params: LaunchParams) -> Pipeline:
    """The soft area shadows' pipeline (`programs.build_area`; made on first use) binding the
    words of launch parameters `params` (one pipeline per word value, as the render's)."""
    words = params_lib.bound_words(params)
    pipeline = self._area.get(words)
    if pipeline is not None:
      return pipeline
    if device_caps(self.device).capturing():
      raise RuntimeError("an OptiX area-shadow pipeline is needed inside a capture: render once outside it first")
    key = programs.ptx_key(self.features, self.flags, supersampled=False, arch=ptx_arch(self.device), area=True)
    if self._area_ptx is None:
      # Not in the prebuilt bundle: compiled on the first launch of the pass, then cached.
      build = functools.partial(programs.build_area, self.features, self.flags)
      self._area_ptx = PtxCompile(build, ptx_arch(self.device), key, device_arch=int(self.device.arch), expected=False)
    ptx = self._area_ptx.result()
    record, module = self._module(key, ptx, words, self.area_layout)
    pipeline = self._area[words] = Pipeline(
      ctx=self.scene.ctx,
      device=self.device,
      ptx=ptx,
      scene=self.scene,
      record=record,
      module=module,
      layout=self.area_layout,
      raygen=programs.AREA,
      anyhit=programs.anyhit_programs(self.features, self.scene),
    )
    return pipeline

  def gi_params(self, ctx: FrameContext, frame: Frame, colors: wp.array[wp.vec4, [int, int]]) -> LaunchParams:
    """Launch parameters of a GI launch over the structures of `frame` (written and built for it
    first) and the geometry's instance colors `colors` (the probe geometry's,
    `baked.Baked._probe_geometry`). The GI programs' rays leave the deformables out by their
    visibility mask (`trace.optix.device.trace.BAKE_RAY_MASK`), as the software bake hides them."""
    if self.gi_layout is None:
      raise RuntimeError("GI launch parameters of an OptiX build without GI programs")
    self._bake_structures(ctx, frame, colors)
    p = params_lib.params_type(self.gi_layout)()
    geo = frame_structs.optix_geometry(ctx, frame, shaded=True)
    geo.instance_color = colors
    self._fill(ctx, p, geo)
    p.geo = geo
    # Every word copy as a frame's (the placement's light and environment structs are unset: their
    # copies would bind 0, a module of another word than the census records).
    params_lib.set_words(p, toggles.load(geo))
    p.nworld = ctx.nworld
    return p

  def _bake_structures(self, ctx: FrameContext, frame: Frame, colors: wp.array[wp.vec4, [int, int]]) -> None:
    """Writes and builds the structures of `frame` with the probe geometry's instance colors
    `colors` (the instances it hides at alpha 0: the instance writers leave them out, as the
    software traversal skips them), once per frame and colors; inside a capture the build is
    left to the launch's cut (`graph_work`). The render's next `refresh` writes its own frame
    again. The colors only in the launch parameters would hide nothing in an opaque build: the
    camera code reads them, not the traversal."""
    key = (id(frame), address(colors))
    if self._apart == key:
      return
    apart = dataclasses.replace(frame, instance_color=colors)
    self.scene.rebuild_every = ctx.settings.performance.rebuild_every
    if device_caps(self.device).capturing():
      self.scene.write(apart, visibility=ctx.features.visibility)
      self._unbuilt = True
    else:
      self.scene.build_apart(apart, wp.get_stream(self.device), visibility=ctx.features.visibility)
    self._apart = key
    self.fresh = False

  def _set_params(
    self, ctx: FrameContext, frame: Frame, seed: wp.array[wp.int32, [1]]
  ) -> tuple[Cameras, OutputArrays, OutputArrays]:
    """Sets the frame's launch parameters; returns (cameras, rendered outputs, renderer outputs).

    The rendered outputs are the supersampled image's with spp > 1, else the renderer's. The
    frame's structs are rebuilt only when an array identity, the context's generation or a
    structure handle changed (as the software backend's `_structs`); the temporal state's struct
    (its fields change between frames) is checked every frame.
    """
    sc = self.scene
    key = (ctx.generation, sc.worlds.handle, sc.shared.handle)
    cached = self._frame
    if cached is None or cached.key != key or not frame.same(cached.frame):
      cached = self._frame = _FrameStructs(frame, key, *self._set_parts(ctx, frame))
    history = ctx.history
    if history is not None:
      self._set_tmp(history.struct)
    if self.layout.shaded:
      self.params.seed = seed
    self.params.nworld = ctx.nunit
    return cached.cams, cached.hi, cached.out

  def _set_parts(self, ctx: FrameContext, frame: Frame) -> tuple[Cameras, OutputArrays, OutputArrays]:
    """Builds the frame's structs into the launch parameters; returns `_set_params`' triple."""
    geo = frame_structs.optix_geometry(ctx, frame, shaded=self.layout.shaded)
    self._fill(ctx, self.params, geo)  # refits keep the IAS handle; a captured graph checks it (graph_work)
    cams = ctx.camera_struct(frame)
    ss = self.supersampling
    hi, out = self._outputs(ctx)
    parts: dict[str, object] = {"geo": geo, "cams": ss.cameras(cams) if ss is not None else cams, "out": hi}
    if self.layout.shaded:  # a geometry-only build's parameters hold no shading struct
      parts.update(
        lit=frame_structs.lights_struct(ctx, ctx.light_grid),
        env=frame_structs.environment(ctx, frame),
      )
    for name, part in parts.items():
      self._set_part(name, part)
    return cams, hi, out

  def _outputs(self, ctx: FrameContext) -> tuple[OutputArrays, OutputArrays]:
    """(the outputs the programs store: the supersampled image's with spp > 1, else the renderer's;
    the renderer's): their structs, each with its output word."""
    out = ctx.buffers.struct(ctx.output_tables())
    ss = self.supersampling
    return (ss.buffers(ctx).struct(ctx.output_tables()) if ss is not None else out), out

  def _set_tmp(self, tmp: Temporal) -> None:
    """Writes the temporal state `tmp` into its launch slot on the device when it changed."""
    key = struct_bytes(tmp)
    if self.part_keys.get("tmp") != key:
      if self.tmp_slot is None:
        self.tmp_slot = self.params.tmp = wp.zeros(1, dtype=Temporal, device=self.device)
      params_lib.write_struct(tmp, self.tmp_slot)
      self.part_keys["tmp"] = key

  def _set_part(self, name: str, part: object) -> None:
    """Sets launch-parameter struct `name` when its content changed (assigning a struct field
    deep-copies it)."""
    key = struct_bytes(part)
    if self.part_keys.get(name) != key:
      setattr(self.params, name, part)
      self.part_keys[name] = key

  def render(self, ctx: FrameContext, frame: Frame, seed: wp.array[wp.int32, [1]]) -> None:
    """Updates the acceleration structures and traces a frame."""
    stream = wp.get_stream(self.device)
    self.refresh(ctx, frame)
    cams, hi, out = self._set_params(ctx, frame, seed)
    self.rendered = True
    pipeline = self.pipeline = self._pipeline(self.params)
    units = (ctx.nunit, self.nray)
    pipeline.store(self.params, *units)
    if device_caps(self.device).capturing():
      self._cut(pipeline, units, None)
    else:
      pipeline.trace(stream, *units)
    if self.supersampling is not None:
      self.supersampling.downsample(ctx, cams, self.params.env, hi, out)

  def close(self) -> None:
    """Releases the OptiX objects and the programs' Warp module (waits for the device).

    Launching a CUDA graph captured with this renderer then raises.

    Raises:
      Exception: a programs' compile failure no render raised, once everything is released.
    """
    if self.closed:
      return
    self.closed = True
    for ahead in self._ahead.values():
      ahead.cancel()
    self._ahead.clear()
    lens_module = self._lens_ptx.module if self._lens_ptx is not None else None
    compiles = [c for c in (self._ptx, self._gi_ptx, self._lens_ptx, self._area_ptx) if c is not None]
    self._ptx = self._gi_ptx = self._ptx_bytes = self._lens_ptx = self._lens_bytes = self._area_ptx = None
    # Every compile joined and every object released, then the first compile failure no render
    # raised.
    failures: list[Exception] = []
    for compile_ in compiles:
      try:
        compile_.join()
      except Exception as error:  # noqa: BLE001  # raised once everything is released
        failures.append(error)
    wp.synchronize_device(self.device)
    # The pipeline stays referenced, destroyed: a graph captured with it refuses to replay.
    for pipeline in (*self._pipelines.values(), *self._gi.values(), *self._area.values()):
      pipeline.destroy()
    self._gi.clear()
    for module in (self.module, lens_module):
      if module is not None:
        module.unload()
        # A module keeps its references' dependents sets pointing at it: drop them (Warp private
        # API).
        module._detach_references()
    if failures:
      raise failures[0]


def set_grid(p: LaunchParams, geo: OptixFields, lit: Lights, out: OutputArrays) -> None:
  """Sets the light-grid visibility launch's geometry `geo` (its build word set: `_fill`), lights
  `lit` and stored outputs `out` (the render's, `OptixRenderer._outputs`) in launch parameters `p`,
  every word copy the render's (`params.set_words`): it binds the render's module."""
  p.geo, p.lit, p.out = geo, lit, out
  params_lib.set_words(p, toggles.load(geo))


@final
class OptixGiRays:
  """The GI bake's rays on an OptiX build (`OptixRenderer.gi_tracer`, a `shade.gi.volume.GiRays`):
  the build's GI programs (`programs.GI_PROGRAMS`), each launch with its one-element launch array
  (`params.GI_LAUNCHES`; the others empty)."""

  def __init__(self, renderer: OptixRenderer, ctx: FrameContext, frame: Frame) -> None:
    """Over the structures of `frame` (`OptixRenderer.gi_params`)."""
    self.renderer, self.ctx, self.frame = renderer, ctx, frame

  def launch(self, program: str, field: str, launch: object, n: int, p: LaunchParams) -> None:
    """`n` units of GI program `program` with launch-parameter field `field` holding `launch`
    (written into the renderer's slot of its kind on the device: `params.write_struct`)."""
    r = self.renderer
    for name in params_lib.GI_LAUNCHES:
      setattr(p, name, r.gi_slots[name] if name == field else r.gi_empty[name])
    params_lib.write_struct(launch, r.gi_slots[field])
    r.gi_launch(program, p, n)

  def probe(
    self,
    launch: gi_kernels.ProbeLaunch,
    n: int,
    *,
    source: Frame,
    colors: wp.array[wp.vec4, [int, int]],
    lit: Lights,
    env: EnvironmentArrays,
  ) -> None:
    """Traces `n` units of a probe-ray launch (`shade.gi.kernels.make_probe_ray`) with the materials
    and colors of frame `source` (the authored frame of a relit volume), lights `lit` and the
    environment `env` (whose `gi` is the volume being baked)."""
    p = self.renderer.gi_params(self.ctx, source, colors)
    p.lit, p.env = lit, env
    params_lib.set_words(p, toggles.load(p.geo))  # the bake's structs: every copy the frame's word
    self.launch(programs.PROBE, "gi_probe", launch, n, p)

  def placement(self, colors: wp.array[wp.vec4, [int, int]]) -> OptixPlacement:
    """The placement's rays over the probe geometry of these colors."""
    return OptixPlacement(self, self.renderer.gi_params(self.ctx, self.frame, colors))


@final
class OptixPlacement:
  """The GI placement's geometry-only rays on an OptiX build (a `shade.gi.volume.PlacementRays`)."""

  def __init__(self, rays: OptixGiRays, params: LaunchParams) -> None:
    """With these launch parameters (`OptixRenderer.gi_params`)."""
    self.rays, self.params = rays, params

  def static(self, launch: gi_kernels.StaticLaunch, n: int) -> None:
    """`n` (ray, probe) units of the static bake (`shade.gi.kernels.make_static_ray`)."""
    self.rays.launch(programs.PROBE_STATIC, "gi_static", launch, n, self.params)

  def survey(self, launch: gi_kernels.SurveyLaunch, n: int) -> None:
    """`n` (ray, point) units of a survey (`shade.gi.kernels.make_survey_ray`)."""
    self.rays.launch(programs.PROBE_SURVEY, "gi_survey", launch, n, self.params)

  def walls(self, launch: gi_kernels.WallsLaunch, n: int) -> None:
    """`n` texels of the walls (`shade.gi.kernels.make_walls_texel`)."""
    self.rays.launch(programs.PROBE_WALLS, "gi_walls", launch, n, self.params)

  def ao(self, launch: gi_kernels.AoLaunch, n: int) -> None:
    """`n` listed texels of the contact occlusion (`shade.gi.kernels.make_ao_texel`)."""
    self.rays.launch(programs.PROBE_AO, "gi_ao", launch, n, self.params)
