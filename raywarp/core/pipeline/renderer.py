"""`Renderer`: the public entry point that renders nworld worlds x ncam cameras of a GpuScene.

`render(sim)` resolves the scene's slots for the call (binding + per-world overrides), poses the
cameras, refits the scene BVH (or the OptiX structures) and launches the megakernel specialised
on the resolved `Features` (process-global kernel cache). It never synchronises the host. Its
parts: `cameras.CameraRig`, `tables.PerWorldTables`, `baked.Baked` (light grid, GI, sky),
`validator.Validator` (the per-frame checks), `graph` (CUDA graphs) and the trace backend
(`trace_backend.TraceBackend`), which reads the renderer's state as a `context.FrameContext`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Self

import numpy as np
import warp as wp

from raywarp._base import arguments
from raywarp._base.types import RenderOrder
from raywarp.core.build.kernel_cache import module_hash
from raywarp.core.build.variants import nearby, registers
from raywarp.core.build.variants.resolve import RenderCaps, refuse_unprebuilt, resolve_features
from raywarp.core.outputs import first_image
from raywarp.core.outputs.render_outputs import Buffers, RenderOutputs, shadow_light
from raywarp.core.pipeline import bake_keys, graph, warmup
from raywarp.core.pipeline import warm as warming
from raywarp.core.pipeline.baked import Baked
from raywarp.core.pipeline.cameras import CameraRig, camera_specs
from raywarp.core.pipeline.controls import RendererControls
from raywarp.core.pipeline.debug import RendererDebug
from raywarp.core.pipeline.frame import Frame
from raywarp.core.pipeline.gpu_scene import binding as bind
from raywarp.core.pipeline.gpu_scene.binding import SLOTS, Slots
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, GpuScene, shadow_casters
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.pipeline.warmup import CompileStatus
from raywarp.core.pipeline.worlds import sharing
from raywarp.core.pipeline.worlds.active import ActiveWorlds
from raywarp.core.pipeline.worlds.optics import switched_instances
from raywarp.core.pipeline.worlds.tables import Defaults, PerWorldTables, absent_slots
from raywarp.core.pipeline.worlds.watch import Watch, watched
from raywarp.core.post.display import deferred
from raywarp.core.post.grading import color, transform
from raywarp.core.post.sensor import effects as sensor_pass
from raywarp.core.post.temporal import history as history_lib
from raywarp.core.shade.ao import contact
from raywarp.core.shade.shadows import area as area_shadows
from raywarp.core.trace import backend as trace_backend
from raywarp.core.trace import frame_structs
from raywarp.core.trace.context import FrameContext, RendererTables, SoftwareScene
from raywarp.core.trace.rays import camera as camera_lib
from raywarp.core.trace.software_backend import SoftwareBackend
from raywarp.scene.model.scene import Scene
from raywarp.settings import Outputs, RenderSettings
from raywarp.settings.fields import RENDER_ORDERS, exposure_from_ev100

if TYPE_CHECKING:
  import concurrent.futures

  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.build.variants.resolve import Resolution
  from raywarp.core.build.variants.tiers import Axes
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.trace.backend import TraceBackend
  from raywarp.core.trace.optix import launch

# Display-relative colours kept per renderer (`_display_linear`: background and fog colours by
# tone mapper): a dataset generator cycling a few dozen keeps every one; 12 B each.
DISPLAY_COLORS: Final = 256
# Every module the renderers load hashes each shared function once per process.
module_hash.install()


class Renderer[S](RendererControls[S]):
  """Renders nworld worlds x ncam cameras of a GpuScene.

  Args:
    scene: the GpuScene (shared by any number of renderers).
    nworld: number of worlds.
    cameras: CameraSpecs or scene camera names / indices (one of them: that camera alone); None
      renders every scene camera.
    resolution: (width, height) of the cameras in pixels (a `CameraSpec.resolution` overrides it).
    settings: RenderSettings (None: the defaults, `RenderSettings()`).
    output_arrays: caller arrays the kernels write instead of allocated buffers (`bind_outputs`).
    per_world: what is randomised per world (`sharing.PerWorld`): geometry fields make the matched
      instances per world (the rest of the static scenery is shared by every world, stored and
      traced once), appearance fields get per-world tables (`per_world[field]`). Only declared
      fields can be written per world, and the declaration (not the writes) decides what is
      compiled in (`resolve.resolve_features`).

  Attributes:
    outputs: the requested outputs, dense (nworld, ncam, H, W) arrays written by every render.
    camera_pose: (nworld, ncam) transforms of the free cameras (mount None), written in place.
    cameras: the CameraSpecs, one per camera.
    layout: which instances are shared and where poses live (`sharing.Layout`).
    per_world_fields: the per-world appearance fields declared (qualified names).
    features: what the settings resolve to (`resolve.Resolution.effective`; the kernels compile
      their tier's build, `context().features`: `explain()` names the tier).

  A context manager: `with Renderer(...) as r:` closes it on exit (`close`). One thread at a time:
  a Renderer is not thread-safe (its own background compiles are). Several Renderers on several
  threads are: each is built, rendered and closed on any thread, beside the others and the
  caller's own Warp work. Its device work goes to the device's current stream (Warp's), which
  raywarp never changes (no `wp.ScopedStream`, no `array.numpy()`: `device.host_copy`), and a
  Warp module loads on one thread at a time (`module_lock`).
  """

  def __init__(  # noqa: PLR0913  # keyword-only: the public constructor
    self,
    scene: GpuScene[S],
    *,
    nworld: int,
    cameras: Sequence[CameraSpec | str | int] | CameraSpec | str | int | None = None,
    resolution: tuple[int, int],
    settings: RenderSettings | None = None,
    output_arrays: Mapping[str, AnyDeviceArray] | None = None,
    per_world: sharing.PerWorld | None = None,
  ) -> None:
    """Builds the renderer's tables, resolves its Features once and starts compiling its kernels.

    Raises:
      ValueError: for nworld or a resolution that is not an integer >= 1 (a bool, a float), a
        deformable material outside the scene's, invalid cameras, or settings the scene or device
        cannot render (`resolve.resolve_features`).
      TypeError: for a scene that is not a GpuScene (a host `Scene`: `rw.upload` it first),
        settings that are not a RenderSettings or per_world not a PerWorld.
    """
    if not isinstance(scene, GpuScene):
      hint = ": upload it first (rw.upload(scene))" if isinstance(scene, Scene) else ""
      raise TypeError(f"scene must be a GpuScene, got {type(scene).__name__}{hint}")
    nworld = arguments.check_int("nworld", nworld, low=1)
    width, height = arguments.check_resolution("resolution", resolution)
    if settings is not None and not isinstance(settings, RenderSettings):
      raise TypeError(f"settings must be a RenderSettings, got {type(settings).__name__}")
    if per_world is not None and not isinstance(per_world, sharing.PerWorld):
      raise TypeError(f"per_world must be a PerWorld, got {type(per_world).__name__}")
    self.scene = scene
    self.device = wp.get_device(scene.device)
    self.nworld = nworld
    self.width, self.height = width, height
    self.settings = settings if settings is not None else RenderSettings()
    self.per_world_fields = frozenset(per_world.fields()) if per_world is not None else frozenset()
    self._dynamic_lights = per_world.light_ids(scene.source_lights) if per_world is not None else ()
    self._per_world_decl = per_world
    self._backend: TraceBackend = SoftwareBackend()
    self._closed = False
    self._first_image_checked = False  # `first_image.check` ran (the first render outside a capture)
    self._baked = Baked()
    self._history: history_lib.History | None = None
    self._imaging: deferred.Imaging | None = None
    self._area: area_shadows.AreaShadows | None = None
    self._proxies: contact.ProxySet | None = None
    self._sensor: sensor_pass.Sensor | None = None
    # `render`'s CUDA graphs (none on a device without graph capture: every frame eager).
    self._graphs: graph.FrameGraphs[S] | None = graph.FrameGraphs() if device_caps(self.device).graphs else None
    self.cameras = camera_specs(scene, cameras)
    self.ncam = len(self.cameras)
    self._setup(output_arrays, per_world)
    rig = self._rig
    self._tier_builds: dict[tuple[RenderSettings, int, tuning.Tuning, Axes], Features] = {}  # `resolve_features`' memo
    # `_configure`'s host-read results kept across `update_settings` (no read, no allocation when
    # what they depend on is unchanged): the potential shadow casters, the display-relative
    # colours by (sRGB, tone mapper), the shadow transmittance's mode.
    self._casters: np.ndarray[[int], np.dtype[np.bool_]] | None = None
    self._display: dict[tuple[tuple[float, float, float], int], wp.vec3] = {}
    self._transmit_mode: str | None = None
    self._caps = RenderCaps(
      scene=scene,
      layout=self.layout,
      ncam=self.ncam,
      bundle=rig.rays is not None,
      distortion=rig.distortion,
      uniform=rig.uniform,
    )
    try:
      self._configure()
      self._declare_per_world()
      self._start_compiles()
    except BaseException as error:
      # Nothing else can release what was made (a backend's compiles): no Renderer
      # is returned, and the garbage collector releases no native OptiX object.
      try:
        self.close()
      except Exception as cleanup:  # noqa: BLE001  # the construction's error is the one raised
        error.add_note(f"releasing the partly built renderer failed too: {cleanup!r}")
      raise

  def _declare_per_world(self) -> None:
    """The declared per-world tables (their first values need the resolved ambient source) and
    the validation of their ranges, the camera rays and poses."""
    geometry = [f"instance.{g}" for g in ("size", "mesh") if self.layout.groups[g].any()]
    scene_slots = self._scene_slots(None)
    for field in sorted(self.per_world_fields) + geometry:
      self._per_world.declare(field, scene_slots=scene_slots, env_lit=self.features.env_lit)
    self._per_world.declare_live(scene_slots)
    self._watch.validator.ranges = self._per_world.ranges()
    if self._rig.rays is not None:
      self._watch.validator.invalid_rays = self._rig.invalid_rays
    self._watch.validator.camera_poses = (self._rig.pos, self._rig.rot)
    self._watch.validator.prepare()

  @staticmethod
  def warm(  # noqa: PLR0913  # the constructor's keyword-only arguments
    scene: GpuScene[S],
    *,
    nworld: int,
    cameras: Sequence[CameraSpec | str | int] | None = None,
    resolution: tuple[int, int],
    settings: RenderSettings | None = None,
    per_world: sharing.PerWorld | None = None,
    sim: S | None = None,
    nearby: int = 0,
    nearby_outputs: Sequence[Outputs] = (),
  ) -> concurrent.futures.Future[warming.Warmed]:
    """Prepares the renderer `Renderer(scene, ...)` of the same arguments ahead, in the
    background: its Warp and OptiX modules (compiled, created and loaded), its bakes and
    structures, one frame of `sim` (the state the run's first frame renders; None: a host
    scene). Everything lands in the caches: a renderer of this configuration then starts at once,
    in this process or a later one. Returns at once (`raywarp/core/pipeline/warm.py`); many warms
    run together, an identical one still running is shared. `.result()`: the steps' seconds and the
    cache entries they added (`Warmed`); it raises what the constructor or the frame raised.
    Warms take turns on the device with each other only: start them before the run's own
    device work, or wait for them. `nearby`: also create the modules of the settings one
    change away, at most so many (`nearby_modules`; `nearby.LIMIT` is a typical bound, each module
    2-12 s of OptiX CPU work, or on the software traversal a Warp module compile of ~10 s, several
    at once; `Warmed.nearby` holds the plan and its estimate);
    `nearby_outputs`: the output sets the run switches to (a viewer's output views), planned first
    as changes one step away.
    """
    config = warming.Config(
      renderer=Renderer,
      nworld=nworld,
      resolution=resolution,
      cameras=cameras,
      settings=settings,
      per_world=per_world,
      sim=sim,
      nearby=nearby,
      nearby_outputs=tuple(nearby_outputs),
    )
    return warming.warm(scene, config)

  def __enter__(self) -> Self:
    """The renderer (`close` on exit)."""
    return self

  def __exit__(self, *_: object) -> None:
    """Closes the renderer (`close`)."""
    self.close()

  def _setup(self, output_arrays: Mapping[str, AnyDeviceArray] | None, per_world: sharing.PerWorld | None) -> None:
    """The device tables of the constructor: cameras, outputs, layout, pose buffers, validation."""
    scene = self.scene
    binding = scene.binding
    self._rig = CameraRig(scene, self.cameras, nworld=self.nworld, width=self.width, height=self.height)
    bound = dict(output_arrays) if output_arrays is not None else None
    self._buffers = self._new_buffers(self.settings.written_outputs(), bound)
    scene.deformables.check_materials(scene.nmaterial)
    self._names = sharing.instance_names(scene)
    binding_slots = binding.slots(None)
    self.layout = layout = sharing.decide(
      scene,
      binding,
      spec=per_world,
      slots={**scene.tables, **binding_slots},
      optics=switched_instances(scene, self.per_world_fields),
    )
    if per_world is not None and per_world.mesh:
      scene.meshes.use_all()  # per-world meshes may show any mesh: every mesh gets its BLAS
    self._tables = RendererTables.build(scene, layout, self.nworld)  # (its `mesh_used` after use_all)
    self._active = ActiveWorlds(self.nworld, self._tables.world_skip)
    self._emission_varies = bake_keys.emission_varies(scene, self.per_world_fields)
    self._seed = wp.zeros(1, dtype=wp.int32, device=self.device)
    self._seed_value = 0
    mesh_bvh_id = wp.zeros(max(scene.meshes.nmesh, 1), dtype=wp.uint64, device=self.device)
    self._software = SoftwareScene(bvh=None, mesh_bvh_id=mesh_bvh_id)
    defaults = Defaults.of(scene, self._rig, ev100=self.settings.exposure.render_ev100())
    self._per_world = PerWorldTables.build(
      scene,
      nworld=self.nworld,
      layout=layout,
      names=self._names,
      declared=self.per_world_fields,
      defaults=defaults,
      mount=self._live_mounts(),
    )
    self._absent = absent_slots(
      nworld=self.nworld, nlight=scene.source_lights, ninstance=scene.ninstance, nmaterial=scene.nmaterial, device=self.device
    )
    self._watch = Watch(scene, layout, self._names, watched(scene, layout, binding_slots, self.per_world_fields))

  def _live_mounts(self) -> np.ndarray[[int], np.dtype[np.int32]]:
    """(ncam,) the scene camera whose live lens tables (`Binding.cameras`) each rig camera reads:
    a camera that is its scene camera as authored (by name); -1 for free cameras and for specs of
    their own (another projection, field of view or lens on a mounted pose keeps its values; ray
    bundles and mount offsets, which no authored camera has, are compared first: arrays have no
    truth value)."""
    names, authored = self.scene.camera_names, self.scene.cameras

    def live_mount(c: CameraSpec) -> int:
      """The index of `c`'s scene camera when `c` is it as authored, else -1."""
      if c.mount is None or c.rays is not None or c.local is not None or authored.get(c.mount) != c:
        return -1
      return names.index(c.mount)

    return np.array([live_mount(c) for c in self.cameras], np.int32)

  def _new_buffers(self, outputs: Outputs, arrays: Mapping[str, AnyDeviceArray] | None) -> Buffers:
    """The output buffers of `outputs` (bound to `arrays` where given; `Buffers`); sets the shadow
    light."""
    self._shadow_light = shadow_light(self.scene, outputs.shadow)
    self._rig.set_outputs(self.settings.outputs, depth_with_rgb=self.settings.depth_with_rgb())
    return Buffers(outputs, nworld=self.nworld, sizes=self._rig.sizes, device=self.device, arrays=arrays)

  def _start_compiles(self) -> None:
    """Starts compiling the first frame's kernels in the background (the OptiX programs too).

    The first render waits for them; until then the caller's own work (scene setup, simulation)
    overlaps the compile.
    """
    self._backend.prepare(self.context(), wait=False)

  # Settings --------------------------------------------------------------------------------------

  def update_settings(self, **changes: object) -> None:
    """Changes settings (`RenderSettings.replace`: whole groups or "group.field" keys, validated).

    Compile-time fields switch to the matching cached kernel; the baked structures rebuild on the
    next render when what they were built for changed (`bake_keys.grid_key` / `probe_key`: their
    Features and the fields whose metadata names them). Changing `outputs` reallocates the output
    buffers (a new `outputs` object); `exposure.mode` / `exposure.ev100` reset the default exposure
    (not a per-world override). Captured graphs become stale (a call that changes nothing does
    nothing). No device read but for a background / fog colour or tone mapper not seen before (one
    solve each, kept): the potential shadow casters are counted again only when
    `lighting.emissive_lights` changes (the validation catches a new one). A change that fails to
    configure (e.g. a feature the scene cannot render) raises and leaves the renderer on its
    previous settings.

    Raises:
      ValueError: for an unknown or invalid setting, or one the scene or device cannot render.
      TypeError: for a group that is not an instance of its class.
      RuntimeError: after `close`.
    """
    self._check_open("update_settings")
    old, buffers = self.settings, self._buffers
    new = old.replace(**changes)
    if new == old:
      return  # nothing changed: no reconfiguration (captured graphs stay valid)
    if new.lighting.emissive_lights != old.lighting.emissive_lights:
      self._casters = None  # the scene's light slots change (its emissive lights): counted again
    try:
      self._apply_settings(old, new)
    except Exception:
      # Back to the settings that configured (and their buffers): no half-applied state.
      self._apply_settings(new, old, buffers=buffers)
      raise

  def _apply_settings(self, old: RenderSettings, new: RenderSettings, *, buffers: Buffers | None = None) -> None:
    """Switches from settings `old` to `new`: their output buffers (`buffers` when given, else new
    ones keeping the bindings of the buffers that stay on), the default exposure, the Features."""
    self.settings = new
    kinds = new.written_outputs()
    if new.outputs != old.outputs or kinds != old.written_outputs():
      if buffers is None:
        on = Buffers.bindable(kinds)
        buffers = self._new_buffers(kinds, {k: a for k, a in self._buffers.bound.items() if k in on})
      else:
        self._shadow_light = shadow_light(self.scene, kinds.shadow)
      self._buffers = buffers
    # Also when only the contact AO or aa.post changed: their depth rides on every RGB camera.
    self._rig.set_outputs(new.outputs, depth_with_rgb=new.depth_with_rgb())
    if new.exposure.render_ev100() != old.exposure.render_ev100():
      # Written in place (the default exposure row keeps its identity: no new Frame field).
      self._per_world.defaults.exposure.assign([exposure_from_ev100(new.exposure.render_ev100())])
    self._configure()

  def compile_status(self) -> CompileStatus:
    """What the renderer's kernels are doing: the modules still compiling in the background (the
    next render waits for them), how many its kernels need and the seconds since they started.
    A host read: no device work."""
    warm = self._backend.warm
    pending, modules, elapsed = warm.status() if warm is not None else ((), 0, 0.0)
    return CompileStatus(
      pending=pending,
      modules=modules,
      elapsed_s=elapsed,
    )

  def wait_compiles(self) -> None:
    """Waits, without device work, for what the constructor started in the background: the
    first frame's Warp modules and the OptiX modules its first frame binds (`compile_status`).
    The first `render` then starts at once; compile failures are raised by it. Another thread
    may use the device meanwhile (`raywarp.warm` waits so between its device steps)."""
    self._check_open("wait_compiles")
    self._backend.join_background()

  def nearby_modules(
    self, limit: int = nearby.LIMIT, outputs: Sequence[Outputs] = ()
  ) -> nearby.Plan[launch.Unit] | nearby.Plan[warmup.WarpModule]:
    """The modules of the settings one change from this renderer's (`nearby.changes`: the output
    sets `outputs`, then every compile-time field over its domain) that its own frames do not
    bind, at most `limit`, with their estimated creation time (`nearby.Plan`;
    `Renderer.warm(nearby=...)` creates them): its OptiX modules, or on the software traversal its
    first frames' Warp modules (each output set's megakernel among them). Host work only; none on
    OptiX before the first frame (which makes the build)."""
    self._check_open("nearby_modules")
    ctx, dev = self.context(), device_caps(self.device)
    casters = int(shadow_casters(self._scene_slots(None), self.per_world_fields).sum())

    def resolve(s: RenderSettings) -> Resolution:
      """`s` resolved on this renderer's scene and device (its tier memo)."""
      return resolve_features(s, self._caps, self._per_world_decl, dev, casters=casters, tier_builds=self._tier_builds)

    def changed(s: RenderSettings, r: Resolution) -> FrameContext[S]:
      """This renderer's context at `s` resolved to `r`."""
      return dataclasses.replace(
        ctx,
        settings=s,
        features=r.features,
        effective=r.effective,
        camera_model=r.camera,
        register_caps=registers.register_caps(s),
      )

    backend, device = self._backend, self.device
    if isinstance(backend, SoftwareBackend):
      return nearby.plan(
        self.settings,
        resolve,
        lambda s, r: warmup.frame_modules(changed(s, r), backend),
        own=warmup.frame_modules(ctx, backend),
        limit=limit,
        create=lambda todo, workers: warmup.load(device, todo, workers),
        outputs=outputs,
        kind="warp",
      )
    optix = getattr(backend, "optix", None)
    if optix is None:
      return nearby.Plan((), (), 0, create=lambda _units, _workers: (0, 0, 0.0))
    return nearby.plan(
      self.settings,
      resolve,
      lambda s, r: optix.frame_modules(changed(s, r)),
      own=optix.frame_modules(ctx),
      limit=limit,
      create=lambda todo, workers: optix.create_modules(todo, workers=workers),
      outputs=outputs,
    )

  def update_gi(self, sim: S | None = None, *, updates: int = 1) -> None:
    """Runs `updates` GI probe updates now (periodic hysteresis), e.g. after changing the lights.

    A no-op without GI (gi.enabled). The volume is baked first if it does not exist yet. The
    probes trace `sim`'s poses: the scene structures are refit to them first, as a render does
    (it counts as a frame of `performance.rebuild_every`).

    Raises:
      ValueError: for `updates` that is not an integer >= 1, or sim's world count not nworld.
      RuntimeError: after `close`.
    """
    self._check_open("update_gi")
    updates = arguments.check_int("updates", updates, low=1)
    if not self.settings.gi.enabled:
      return
    self._check_sim(sim)
    self._backend.prepare(self.context())  # after the background compiles
    if not self.features.gi:
      return
    self._device_writes(sim)
    frame = self._frame(sim)
    self._pack_lights(frame)
    ctx = self.context()
    self._backend.refit(ctx, frame)  # the probes' rays meet the frame's poses
    baked = self._baked.generation
    self._baked.probes(ctx, frame, self._backend, force=updates)
    if self._baked.generation != baked:
      self._changed()  # a new volume: captured graphs read the old one

  def _device_writes(self, sim: S | None) -> None:
    """The device writes before a frame's slots are read: the binding's tables (`write_tables`)
    and poses, its live camera tables (`LiveCameras`), the per-world colors with its hiding
    (`write_colors`), the derived per-world fields (`derived_fields.py`) and the per-world
    optics."""
    t = self._per_world
    self.scene.binding.write_tables(sim)
    if t.poses is not None:
      self.scene.binding.write(t.poses, sim)
    if t.live_cameras is not None:
      t.live_cameras.write(self.scene.binding.cameras(sim))
    t.write_colors(self.scene, sim)
    t.derived.derive(lambda: t.instance_material(self.scene, sim))
    t.write_optics(self.scene, sim, self.settings.shadows.translucent)

  def _check_sim(self, sim: S | None) -> None:
    """Raises ValueError when `sim` has another world count than the renderer."""
    nworld = self.scene.binding.nworld(sim)
    if nworld is not None and nworld != self.nworld:
      raise ValueError(f"the simulator state has {nworld} worlds, the renderer {self.nworld}")

  def _display_linear(self, srgb: tuple[float, float, float]) -> wp.vec3:
    """Linear radiance x exposure that the display transform shows as the sRGB color `srgb` (one
    device solve and host read per new (color, tone mapper); kept for the renderer's life, at most
    DISPLAY_COLORS of them)."""
    key = (srgb, self._params.tone_mapper)
    found = self._display.get(key)
    if found is None:
      linear = color.inverse_display_transform(
        color.srgb_to_linear(np.asarray(srgb, dtype=np.float64)), self._params.tone_mapper, self.device
      )
      if len(self._display) >= DISPLAY_COLORS:
        self._display.pop(next(iter(self._display)))
      found = self._display[key] = wp.vec3(*linear)
    return found

  def _configure(self) -> None:
    """Resolves Features and Params from the settings (`resolve_features`, host only) and
    reconfigures the backend, the history and the shadow transmittance."""
    self._changed()
    dev = device_caps(self.device)
    if self._casters is None:
      self._casters = shadow_casters(self._scene_slots(None), self.per_world_fields)
    casters = self._casters
    resolved = resolve_features(
      self.settings,
      self._caps,
      self._per_world_decl,
      dev,
      casters=int(casters.sum()) + len(self.scene.environment.suns),  # the suns cast shadows
      tier_builds=self._tier_builds,
    )
    refuse_unprebuilt(self.settings, resolved.choice, dev)
    self.features, self._params, self._build = resolved.effective, resolved.params, resolved.features
    self._choice = resolved.choice
    self._relight = resolved.relight
    self._camera_model = resolved.camera
    self._arm_casters(casters)
    self._configure_order()
    # The tables the kernels read: the build's.
    # The relight build's carrier (`resolve.relight_build`: every material lobe) reads them too.
    carrier = self._build if self._relight is None else self._relight
    self._tables.prepare_diffuse(self.scene, enabled=self._build.diffuse_roughness or carrier.diffuse_roughness)
    self._tables.prepare_layers(self.scene, enabled=self._build.layered or carrier.layered)
    self._per_world.derived.specular_weight = self._build.specular_weight
    self._tables.prepare_specular(self.scene, enabled=self._build.specular_weight or carrier.specular_weight)
    self._report_material_limits()
    backend = trace_backend.named(self.features.traversal)
    if type(self._backend) is not backend:
      self._backend.close()
      self._backend = backend()
    self._background = self._display_linear(self.settings.environment.background_color)
    self._configure_imaging()
    self._configure_area()
    self._configure_sensor()
    optics = self._per_world.optics
    if optics is None:
      if self._transmit_mode != self.settings.shadows.translucent:
        self._tables.shadow_transmit = RendererTables.transmittance(self.scene, self.settings.shadows.translucent)
        self._transmit_mode = self.settings.shadows.translucent
    else:
      # Per-world optics (`worlds/optics.py`): written every frame, `shadows.translucent` applied.
      self._tables.shadow_transmit, self._tables.optics, self._tables.twosided = optics.shadow, optics.optics, optics.twosided
    self._configure_history()
    self._backend.configure(self.context())

  def _report_material_limits(self) -> None:
    """Records in the scene's load report (once per entry) what the resolved configuration drops
    or approximates of the scene's materials."""
    f, scene, report = self.features, self.scene, self.scene.report
    if f.live_material_uv and scene.material_uv is not None:
      missing = scene.material_uv.missing_second
      key = "gpu.material.swap_uv_set1_missing(set 0 used)"
      if missing and key not in report.dropped:
        report.drop(key, f"core mesh {missing[0]}", len(missing))
    if f.diffuse_roughness and "material.base_diffuse_roughness" in self.per_world_fields:
      mixed = np.flatnonzero(scene.derive_inputs.diffuse_source[:, 1] == 0.0)
      key = "gpu.material.base_diffuse_roughness(declared on subsurface/transmission fallback)"
      if len(mixed) and key not in report.dropped:
        report.drop(key, f"material row {int(mixed[0])}", len(mixed))
    key = "gpu.material.base_diffuse_roughness(probe GI retains Lambert convolution)"
    if f.diffuse_roughness and f.gi and key not in report.approximated:
      report.approx(key)
    key = "gpu.material.base_diffuse_roughness(disabled by settings)"
    if scene.capabilities.diffuse_roughness and self.settings.openpbr.diffuse_roughness is False and key not in report.dropped:
      report.drop(key)

  def _configure_imaging(self) -> None:
    """Writes the fog and look defaults (per-world tables start from them), turns the look rows
    on or off, and (re)builds the deferred display when the settings need it
    (`core/post/display/deferred.py`)."""
    fog, d = self.settings.atmosphere, self._per_world.defaults
    d.fog_color.assign([self._display_linear(fog.color)])
    d.fog_range.assign([wp.vec3(fog.start, fog.end, fog.haze)])
    for name, value in transform.defaults(self.settings).items():
      self._per_world.renderer_default(name).assign([value])
    self._tables.grading.configure(self.settings)
    if not self.settings.deferred_display():
      self._imaging = self._proxies = None
      return
    if self._imaging is None:
      self._imaging = deferred.Imaging(self.device, self.nworld, list(self._rig.sizes))
    self._imaging.configure(self.settings)
    if self.settings.contact_ao() <= 0.0:
      self._proxies = None
    elif self._proxies is None:
      self._proxies = contact.ProxySet(self.scene, self.layout, self.nworld, list(self._rig.sizes))

  def _configure_area(self) -> None:
    """(Re)builds the soft area shadows' pass while `shadows.area_rays` > 0
    (`core/shade/shadows/area.py`; nothing otherwise)."""
    if not self.settings.area_rays():
      self._area = None
    elif self._area is None:
      self._area = area_shadows.AreaShadows(self.device, self.nworld, self._rig.sizes, self._rig.adr)

  def _configure_sensor(self) -> None:
    """Writes the sensor's per-world defaults and (re)configures its pass while an effect or the
    post anti-aliasing (`aa.post`) is on (`core/post/sensor/effects.py`; none otherwise)."""
    s = self.settings
    # A rectified pinhole pair: no ray bundle, orthographic, fisheye or distorted camera.
    pinhole = self._rig.rays is None and not self._rig.distortion and "camera.distortion" not in self.per_world_fields
    pinhole = pinhole and all(c.projection == "perspective" for c in self.cameras)
    sensor_pass.check_cameras(s, perspective=pinhole)
    declared = bool(self.per_world_fields & sensor_pass.PER_WORLD_FIELDS.keys())
    if declared and s.outputs.rgb is None and s.outputs.luminance is None:
      raise ValueError("per-world sensor fields act on the color: request outputs.rgb or outputs.luminance")
    for name, value in sensor_pass.defaults(s).items():
      self._per_world.renderer_default(name).assign([value])
    if not (s.sensor.color() or s.aa.post or declared or s.sensor.depth != "off"):
      self._sensor = None
      return
    if self._sensor is None:
      self._sensor = sensor_pass.Sensor(self.device, self.nworld, self._rig.sizes)
    self._sensor.configure(s, per_world=declared)

  def _configure_history(self) -> None:
    """(Re)allocates the motion output's state when its layout changed (a new one starts empty)."""
    if not (f := self._build).history:
      self._history = None
      return
    # One image size for every camera with history (`resolve._temporal`); a kept history has it.
    size, motion, h = self._rig.sizes[0], self._buffers.motion, self._history
    status = self._buffers.layout.has("flow_status")
    if h is not None and h.adopt(f, motion, status=status, resolution=size):
      return
    self._history = history_lib.History(
      f, nworld=self.nworld, ncam=self.ncam, resolution=size, motion=motion, status=status, device=self.device
    )

  def reset_history(self, worlds: wp.array[bool, [int]] | None = None) -> None:
    """Restarts the temporal history of the masked worlds (bool (nworld,) array; None: all).

    The previous poses and cameras of the motion output, the metered exposure and the sensor
    noise sequence restart, so a reset world renders as on its first frame (no motion). Call it
    on episode resets, camera cuts and teleports. Graph-capturable (device work only).

    Raises:
      ValueError: for a mask that is not a bool (nworld,) array on the renderer's device.
      RuntimeError: after `close`.
    """
    self._check_open("reset_history")
    if worlds is not None and (worlds.dtype != wp.bool or worlds.shape != (self.nworld,) or worlds.device != self.device):
      raise ValueError(
        f"reset_history(worlds): a bool ({self.nworld},) array on {self.device}, "
        f"not {wp.types.type_repr(worlds.dtype)} {worlds.shape} on {worlds.device}"
      )
    if self._history is not None:
      self._backend.prepare(self.context())  # after the background compiles
    self._reset_worlds(worlds)

  def _reset_worlds(self, worlds: wp.array[bool, [int]] | None) -> None:
    """Restarts the masked worlds' temporal state (None: all): history, metered exposure, sensor
    noise sequence (device work; the history after the background compiles)."""
    if self._history is not None:
      self._history.reset(worlds)
    if self._imaging is not None:
      self._imaging.reset(worlds)
    if self._sensor is not None:
      self._sensor.reset(worlds)

  # Frame -----------------------------------------------------------------------------------------

  @property
  def debug(self) -> RendererDebug[S]:
    """The read-only views of the renderer's internal state (`debug.RendererDebug`: one
    frame's kernel-field arrays, history length, GI volume and re-bake log, graph version)."""
    return RendererDebug(self)

  def _frame(self, sim: S | None = None) -> Frame:
    """The kernel-field arrays of a call for `sim` (what `render(sim)` reads), without rendering
    (`RendererDebug.frame`)."""
    return self._per_world.frame(self._scene_slots(sim), absent=self._absent, env_lit=self.features.env_lit)

  def _scene_slots(self, sim: S | None) -> Slots:
    """The slots of the call without the per-world overrides (`PerWorldTables.scene_slots`)."""
    return self._per_world.scene_slots(self.scene, sim, emissive_lights=self.settings.lighting.emissive_lights)

  def _pack_lights(self, frame: Frame) -> None:
    """Packs the frame's lights (`shade/lights/pack.py`) before anything reads them."""
    if self._tables.light_pack.pack(frame, len(self.scene.environment.maps.scale), env_lit=self.features.env_lit):
      self._changed()  # a new packed table (rows changed): kernel structs and graphs are stale

  def _configure_order(self) -> None:
    """The render order and TILED's band height of the resolved Features."""
    f = self.features
    order = RenderOrder(RENDER_ORDERS[self.settings.performance.render_order])
    self._rig.set_order(order, camera_lib.tile_rows(store_bound=not f.any_rgb or f.unlit))

  def _arm_casters(self, casters: np.ndarray[[int], np.dtype[np.bool_]]) -> None:
    """Checks the frames' lights against the potential `casters` while one more would turn the
    shadow budget on (`validator_kernels.Word.CASTERS`), else not."""
    cap, f = self.settings.shadows.max_casters or 0, self.features
    armed = f.shadows and f.budget == 0 and 0 < cap < len(casters) and not casters.all()
    v = self._watch.validator
    v.casters = wp.array(casters, dtype=bool, device=self.device) if armed else None

  def _poll_validation(self, sim: S | None) -> None:
    """Acts on an earlier frame's validation word: re-resolves the Features for a new shadow
    caster (`watch.Watch.poll` raises for the errors)."""
    self._watch.poll(sim, lambda: self._frame(sim), self.features)
    if self._watch.new_casters:
      self._watch.new_casters = False
      self._casters = None  # counted again from the frame's tables
      self._configure()  # the budget follows the casters (graphs are stale if it changed)
      self._backend.prepare(self.context())

  def context(self) -> FrameContext[S]:
    """The renderer's state as its trace backend and the struct builders read it."""
    b = self._baked
    return FrameContext(
      scene=self.scene,
      layout=self.layout,
      settings=self.settings,
      features=self._build,
      params=self._params,
      effective=self.features,
      relight=self._relight,
      camera_model=self._camera_model,
      nworld=self.nworld,
      cameras=self._rig,
      buffers=self._buffers,
      history=self._history,
      software=self._software,
      tables=self._tables,
      light_grid=b.grid,
      gi=b.gi,
      background=self._background,
      per_world_fields=self.per_world_fields,
      dynamic_lights=self._dynamic_lights,
      generation=(self._generation, b.generation),
      shadow_light=self._shadow_light,
      register_caps=registers.register_caps(self.settings),
      reference_exposure=self._per_world.defaults.exposure,
      emission_varies=self._emission_varies,
      view_exposure=self._imaging.auto.view if self._imaging is not None and self._imaging.auto is not None else None,
    )

  def render(
    self,
    sim: S | None = None,
    *,
    seed: int | wp.array[wp.int32, [1]] = 0,
    active: wp.array[wp.int32, [int]] | None = None,
  ) -> RenderOutputs:
    """Renders the current frame of `sim` (e.g. mujoco_warp.Data) into `outputs`.

    On a CUDA device the frame's GPU work is a CUDA graph (`graph.FrameGraphs`): the second frame
    whose inputs repeat (the arrays it reads, `_graph_arrays`: the simulator's and the per-world
    tables; the seed array; no setting or table replaced since) is recorded, and every later frame
    with those inputs replays it. Other frames run eagerly: the first one, and after a change.
    Arrays written in place (simulator state, `camera_pose`, per-world tables) keep the graph;
    swapping a few simulator buffers keeps one graph per buffer. `capture` records a whole loop
    step instead (simulation + render + policy).

    The first render of every world outside a graph capture reads back up to 16 worlds of the
    display color once and warns for a camera whose image is blown out or black (`first_image`);
    no other render synchronizes with the host.

    Args:
      sim: the simulator state the scene's binding reads (MuJoCo: the mjw.Data).
      seed: random seed of stochastic effects: a signed int32 or a one-element int32 array
        on this renderer's device. Array contents are read by the rendering kernels and can be
        updated by captured device work. The caller owns ordering of those writes.
      active: the worlds to render, a (k,) int32 device array of distinct world ids (1 <= k <=
        nworld; -1 entries are holes); None renders every world. The launch covers k units, the
        listed worlds packed first: a short list launches fewer threads, and a fixed (nworld,)
        array written in place (e.g. `where(done, -1, arange(nworld))`, also inside a captured
        graph) renders only the listed worlds with no host round trip or recompile. Worlds not
        listed are skipped: their outputs keep their values (a dense (nworld, ...) output, its
        rows of skipped worlds untouched) and their per-world post passes and temporal state
        pause; a world listed again after a skip restarts its temporal state (`reset_history`).
        An id out of [0, nworld) is not rendered and a repeated id is rendered once; either makes
        the next validated render raise (`tuning.validate_every`).

    Returns:
      `outputs`.

    Raises:
      ValueError: when sim's world count differs from the renderer's, for a seed outside the
        32-bit range (it would wrap: two seeds, one image), an `active` array of another dtype,
        shape or device, or for an error the previous frame's validation found
        (`worlds/validator.py`).
      TypeError: for an `active` that is not a Warp array.
      RuntimeError: after `close`.
    """
    self._check_open("render")
    seed_array = self._seed_array(seed)
    self._check_sim(sim)
    self._use_active(active)
    graphs = self._graphs
    capturing = device_caps(self.device).capturing()
    if graphs is None or capturing:
      self.eager_frame(sim, seed_array, polled=False)
    else:
      graphs.render(self, sim, seed_array)
    if not capturing:
      self._renders += 1  # a recorded frame counts at its replays (`graph.RenderGraph.launch`)
    if not self._first_image_checked and not capturing and active is None:
      self._first_image_checked = True
      first_image.check(self.outputs, self.nworld)
    return self.outputs

  def _replayed(self) -> None:
    """Counts a frame of a `capture` graph's replay (`graph.Replayable`; `camera_info`'s frame
    index)."""
    self._renders += 1

  def _use_active(self, active: wp.array[wp.int32, [int]] | None) -> None:
    """Takes the render's active worlds (`ActiveWorlds.use`): a new list length or skip table
    rebuilds the kernel structs (and graphs); back to every world, the skipped ones restart their
    temporal state."""
    a = self._active
    before = a.skip
    if not a.use(active):
      return
    self._tables.world_skip, self._tables.launch_worlds = a.skip, a.units
    self._watch.validator.set_world_ids(a.invalid if active is not None else None)
    self._changed()
    if active is None:
      self._reset_worlds(before)

  def eager_frame(self, sim: S | None, seed: wp.array[wp.int32, [1]], *, polled: bool) -> None:
    """Enqueues the frame's work (recorded when the device's stream is capturing); `polled`: the
    host checks of `_before_replay` already ran for it (`graph.FrameGraphs`)."""
    backend = self._backend
    # Builds / compiles what the frame needs, after the constructor's background compiles: no
    # launch before them (a launch would wait for its module's load anyway: `module_lock`).
    backend.prepare(self.context())
    if not polled and not device_caps(self.device).capturing():
      self._before_replay(sim)
    if self._active.ids is not None:
      self._active.update()
      self._reset_worlds(self._active.resumed)
    self._device_writes(sim)
    slots = self._scene_slots(sim)
    frame = self._per_world.frame(slots, absent=self._absent, env_lit=self.features.env_lit)
    self._pack_lights(frame)
    look = {name: self._per_world.renderer_field(name) for name in transform.PER_WORLD_FIELDS}
    self._tables.grading.prepare(look)
    self._rig.pose(
      slots.get("camera.pos"),
      slots.get("camera.rot"),
      frame.camera_fovy,
      frame.camera_intrinsics,
      frame.camera_distortion,
      self._tables.launch_worlds,
    )
    history = self._history
    if history is not None:
      history.begin(frame, self._rig.pos, self._rig.rot)
    ctx = self.context()  # after the validation's refits
    backend.refit(ctx, frame)
    ctx = self._bake(ctx, frame)
    if self.features.counters:
      self._tables.counters.zero_()
    backend.launch(ctx, frame, seed)
    self._run_area(ctx, frame)
    self._end_frame(frame)
    self._run_imaging(frame)
    self._run_sensor(frame, seed)
    capturing = device_caps(self.device).capturing()
    every = tuning.current().validate_every
    self._watch.end(sim, frame, self.features, every=every, capturing=capturing)

  def _bake(self, ctx: FrameContext[S], frame: Frame) -> FrameContext[S]:
    """The frame's probe work and light grid (`baked.Baked`); the context with the structures
    rebuilt this frame (a replacement makes captured graphs stale)."""
    baked = self._baked.generation
    self._baked.probes(ctx, frame, self._backend)
    self._baked.grid_for(self._baked.current(ctx), frame, self._backend)
    if self._baked.generation == baked:
      return ctx
    self._changed()
    return self.context()

  def _seed_array(self, seed: int | wp.array[wp.int32, [1]]) -> wp.array[wp.int32, [1]]:
    """The seed array of a render: a validated device seed, or the renderer's reusable one-element
    array holding integer `seed`."""
    if isinstance(seed, wp.array):
      if seed.dtype != wp.int32 or seed.shape != (1,) or seed.device != self.device:
        raise ValueError(f"seed must be an int32 array of shape (1,) on {self.device}")
      return seed
    value = arguments.check_int("seed", seed, low=arguments.SEED_MIN, high=arguments.SEED_MAX)
    if value != self._seed_value:
      self._seed.fill_(value)
      self._seed_value = value
    return self._seed

  def _end_frame(self, frame: Frame) -> None:
    """History bookkeeping of the motion output (poses for the next frame, frame index)."""
    if self._history is not None:
      self._history.end(frame, self._rig.pos, self._rig.rot, world_skip=self._tables.world_skip)

  def _run_area(self, ctx: FrameContext[S], frame: Frame) -> None:
    """The soft area shadows' pass over the frame's HDR image (`shade/shadows/area.py`; none
    when off)."""
    area = self._area
    if area is None:
      return
    out = self._buffers.struct(ctx.output_tables())
    casters = self.settings.shadows.max_casters or 0
    area.run(self.settings.area_rays(), casters, functools.partial(self._backend.area_shadows, ctx, frame), out)

  def _run_imaging(self, frame: Frame) -> None:
    """The deferred display over the frame's HDR image (`core/post/display/deferred.py`; none
    when off)."""
    im = self._imaging
    if im is None:
      return
    buffers = self._buffers
    ctx = self.context()
    out = buffers.struct(ctx.output_tables())
    kind = deferred.source_kind(buffers.layout)
    src = deferred.source(
      kind=kind,
      flat=buffers.flat,
      exposure=frame.exposure,
      lens_table=self._rig.lens,
      worlds=self._tables.launch_worlds,
      placeholders=im.placeholders,
    )
    env = frame_structs.environment(ctx, frame)
    im.run(src, kind, env, out, frame.camera_fovy, self._contact_ao(frame))

  def _contact_ao(self, frame: Frame) -> tuple[float, contact.Proxies] | None:
    """The contact AO's (strength, the frame's proxies in each camera's view) (`core/shade/ao/
    contact.py`; None off: no host work, no launch)."""
    strength, proxies = self.settings.contact_ao(), self._proxies
    if strength <= 0.0 or proxies is None or not proxies.count:
      return None
    return float(strength), proxies.gather(frame, self._rig.pos, self._rig.rot)

  def _run_sensor(self, frame: Frame, seed: wp.array[wp.int32, [1]]) -> None:
    """The sensor pass over the frame's final outputs (`core/post/sensor/effects.py`; none when
    off)."""
    sn = self._sensor
    if sn is None:
      return
    ctx = self.context()
    sn.run(
      self._buffers.struct(ctx.output_tables()),
      ctx.camera_struct(frame),
      seed=seed,
      motion=self._buffers.motion,
      world={name: self._per_world.renderer_field(name) for name in sensor_pass.PER_WORLD_FIELDS},
      skip=self._tables.world_skip,
    )

  # Graphs ----------------------------------------------------------------------------------------

  def _graph_arrays(self, sim: S | None) -> dict[str, AnyDeviceArray]:
    """Slot / binding source name -> array of a call: what a captured graph must find unchanged."""
    slots = bind.merged(self._scene_slots(sim), self._per_world.overrides)
    out = {name: a for name in SLOTS if (a := bind.slot(slots, name)) is not None}
    out.update(self.scene.binding.sources(sim))
    if self._active.ids is not None:
      out["render.active"] = self._active.ids
    return out

  def _before_replay(self, sim: S | None) -> None:
    """The host work of a frame a graph cannot replay: the validation word's poll, the binding's
    host staging (`Binding.stage`) and the GI groups' overflow (a world's inputs matching no baked
    group: graphs become stale, the next eager frame re-bakes)."""
    self._poll_validation(sim)
    self.scene.binding.stage(sim)
    if self._baked.overflowed():
      self._changed()

  def capture(
    self,
    step: Callable[[], None] | None = None,
    sim: S | None = None,
    *,
    seed: int | wp.array[wp.int32, [1]] = 0,
    active: wp.array[wp.int32, [int]] | None = None,
    after: Callable[[], None] | None = None,
    arrays: graph.ArraySource | None = None,
  ) -> graph.RenderGraph[S]:
    """Captures `step()` + `render(sim)` + optional `after()` after one eager run.

    `render` alone already replays its own graph (see `render`): capture fuses the caller's
    simulation and policy work into the same graph, one launch per loop step.

    With traversal="optix" the graph is cut around the OptiX calls, which stream capture cannot
    record (`core/trace/optix/graph.py`); `step` may render other renderers too.
    A device `seed` is read on every replay; `step` may write it. An integer initializes a
    graph-owned scalar, independent of subsequent eager renders. The default remains fixed 0;
    no hidden random-number advance occurs. `graph.seed` exposes the captured scalar.
    `active` (`render`'s) is read on every replay: `step` or the caller may write its ids in place.

    `after` can consume this iteration's image and write actions for the next simulator step.
    Use the application's shared Warp/Torch CUDA stream and preallocated capture-safe operations.
    `arrays` returns named external arrays used by the callbacks (policy inputs, weights, actions,
    simulator scratch). The graph retains their owners and checks names, addresses, shapes,
    strides, dtype and device on replay. Return current bindings, not a permanently cached map
    after replacing storage. Unlisted external resources remain the caller's responsibility.
    Capture preparation executes work; restore simulation, controls and history before comparing
    eager and replayed trajectories. The graph retains both callback closures until released.

    Raises:
      RuntimeError: on a device without graph capture (CPU), or after `close`.
    """
    self._check_open("capture")
    seed_array = self._seed_array(seed)
    if not isinstance(seed, wp.array):
      seed_array = wp.clone(seed_array)

    def work() -> None:
      """One frame: simulation, current observation, then policy/actions."""
      if step is not None:
        step()
      self.render(sim, seed=seed_array, active=active)
      if after is not None:
        after()

    owners: dict[str, AnyDeviceArray] | None = None
    recorded: dict[str, graph.Fingerprint] | None = None

    def before_record() -> None:
      """External storage owned and fingerprinted after warmup, before recording can refer to it."""
      nonlocal owners, recorded
      if arrays is not None:
        owners = dict(arrays())
        recorded = {name: graph.fingerprint(array) for name, array in owners.items()}

    segments = graph.capture(self.device, work, before_record=before_record)
    if arrays is not None and recorded is not None:
      graph.check_arrays(arrays(), recorded, "caller during capture")
    return graph.RenderGraph(self, segments, sim, seed=seed_array, arrays=arrays, work=work, owners=owners)

  def close(self) -> None:
    """Invalidates captured graphs and releases the backend; the renderer cannot render
    afterwards. Idempotent."""
    self._closed = True
    self._changed()
    if self._graphs is not None:
      self._graphs.clear()
    self._backend.close()
