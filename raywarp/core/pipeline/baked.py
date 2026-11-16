"""The lighting structures a Renderer precomputes once instead of every frame (`Baked`: light
grid, GI probes), built on first use from static scenery and cached on disk.

* The light visibility grid (`lighting.light_grid`): which lights each cell sees, built on the
  backend's hardware when it can (`TraceBackend.grid_tracer`), else by the software kernel;
  rebuilt when what it was built for changes (`grid_key`) or on request.
* The GI probe volume (`gi.enabled`): baked on the first render, then the periodic updates on the
  device's schedule (`gi.Schedule`: a captured graph replays them). Worlds whose radiance inputs
  differ get their own irradiance maps (`shade.gi.groups`: one shared volume when every world's
  inputs are equal, K stacked ones for a few variants, else one shared volume tinted per world:
  `gi.Relight`), tracked every frame.

Probe and grid rays use the software traversal of world 0 (`TraceBackend.software_bvh`; a GI
group shades its hits with its own world's materials, lights and environment). Every
replacement bumps `generation` (the kernel structs then rebuild). The first build of each comes
from the disk cache when it holds one for the same inputs (`bake_cache`: bit-identical, keyed by
the content of every struct the bake reads), and is stored there otherwise. The keys hash the
inputs of the software BVHs (instance frames, meshes), not the BVHs: the lookup comes first, and a
hit builds no BVH (a frozen GI bake and the light grid of a warm start).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
import weakref
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.records import record_field
from raywarp._base.types import ShapeType
from raywarp.core.build.variants import tiers
from raywarp.core.kernels import probe
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import software_view
from raywarp.core.pipeline import bake_keys
from raywarp.core.pipeline.frame import Frame
from raywarp.core.pipeline.gpu_scene import binding as bind
from raywarp.core.pipeline.gpu_scene.gpu_scene import shadow_transmit
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.shade import gi as gi_lib
from raywarp.core.shade.gi import cube as gi_cube
from raywarp.core.shade.gi import groups as gi_groups
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.shade.lights import pack
from raywarp.core.trace import frame_structs
from raywarp.core.trace.accel import deformable_bvh
from raywarp.scene.images.colorspace import REC709_Y
from raywarp.settings.fields import exposure_from_ev100

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyDeviceArray, AnyHostArray
  from raywarp.core.kernels.abi import rays as trace
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.kernels.abi.structs import EnvironmentArrays
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.shade.gi.volume import Point
  from raywarp.core.shade.lights.punctual import Light, Lights
  from raywarp.core.trace.backend import GridTracer, TraceBackend
  from raywarp.core.trace.context import FrameContext


# Light fields that change where a light reaches: declared per world, the grid cannot cull.
GRID_FIELDS: Final = frozenset(
  {
    "light.pos",
    "light.dir",
    "light.range",
    "light.type",
    "light.cutoff",
    "light.cone_inner",
    "light.shape",
    "light.axis",
    "light.size",
  }
)


def _every_light_on(packed: wp.array[Light, [int, int]]) -> wp.array[Light, [int, int]]:
  """A copy of packed lights (`lights.Light` rows) with every light active (a host round trip, once
  per grid build)."""
  rows = host_copy(packed[:1]).copy()  # host_copy aliases CPU arrays
  record_field(rows, "active")[...] = True
  return wp.array(rows, dtype=packed.dtype, device=packed.device)


@final
class Baked:
  """The light grid and GI probe volume of one Renderer (see the module doc).

  Attributes:
    grid: the light visibility grid (None: off or not built); `grid_built_for` its `grid_key`.
    gi: the GI probe volume (None before its bake); `schedule` its bake / update policy.
    probes_built_for: the `probe_key` of `gi` and whether GI tracks per-world inputs.
    generation: bumped whenever one of them is replaced.
  """

  def __init__(self) -> None:
    """Nothing built."""
    self.grid: light_grid.Grid | None = None
    self.grid_built_for: tuple[object, ...] = ()
    self.gi: gi_lib.ProbeVolume | None = None
    self.schedule: gi_lib.Schedule | None = None
    self.probes_built_for: tuple[object, ...] = ()
    self.generation = 0
    self._static_colors: wp.array[wp.vec4, [int, int]] | None = None
    self._static_mask: wp.array[wp.int32, [int]] | None = None
    # What the GI digests skip (the hidden instances, the dynamic lights) and the probes' lights
    # (`gi.StaticLights`: the dynamic ones off); built with the static mask.
    self._skips: gi_groups.Skips[int] | None = None
    self._static_lights: gi_lib.StaticLights | None = None
    # The relight kernel's inputs of the last frames (`_relit_inputs`: the authored and the worlds'
    # frame), each with its frame, generation and light grid.
    self._relit: list[tuple[Frame, tuple[int, int], light_grid.Grid | None, gi_lib.ProbeInputs]] = []
    # Moving lights already warned about (`_warn_moving_lights`).
    self._warned: frozenset[int] = frozenset()
    # The GI's re-bake log (`Renderer.debug.gi_rebakes`) and the reason of the next bake.
    self.log = gi_lib.RebakeLog()
    self._reason = gi_lib.Reason.FIRST
    self._hidden_host: np.ndarray[[int], np.dtype[np.bool_]] | None = None
    self._hidden_flex: wp.array[trace.FlexPart, [int]] | None = None
    # New per-world GI inputs appeared after a bake (an overflow): the worlds are randomised
    # continuously, every later bake is relit (`gi.Relight`) instead of re-baking per new variant.
    self._continuous = False
    # World 0's per-world GI input rows at the first GI bake (one-row copies by Frame field): the
    # inputs a relit volume is baked with, whatever world 0 holds when it is (re-)baked, so a
    # world reset to earlier inputs renders as before (no state of past episodes).
    self._authored: dict[str, AnyDeviceArray] = {}
    self._authored_colors: wp.array[wp.vec4, [int, int]] | None = None

  def drop_grid(self) -> None:
    """Rebuilds the light grid on its next use."""
    self.grid = None

  def drop_probes(self, reason: gi_lib.Reason = gi_lib.Reason.REQUESTED) -> None:
    """Re-bakes the GI volume on the next frame (`reason`: for the re-bake log)."""
    if self.gi is not None:
      self._reason = reason
    self.gi = None
    self._static_colors = None
    self._static_mask = None
    self._skips = None
    self._hidden_host = None

  def overflowed(self) -> bool:
    """Whether an earlier frame met GI inputs no baked group holds (`shade.gi.groups`; a pinned
    read, no synchronisation): the next `probes` re-bakes."""
    return self.gi is not None and self.gi.groups.overflow_code() != 0

  def current(self, ctx: FrameContext) -> FrameContext:
    """`ctx` with this object's structures."""
    return dataclasses.replace(ctx, light_grid=self.grid, gi=self.gi, generation=(ctx.generation[0], self.generation))

  def grid_for(self, ctx: FrameContext, frame: Frame, backend: TraceBackend) -> light_grid.Grid | None:
    """The frame's light grid, built when needed; None without `lighting.light_grid`, or when
    footprint fields are declared per world without a movable-light hint. Declared movable
    lights bypass the static grid; undeclared motion is detected by the device reach check."""
    if not ctx.effective.culling or (ctx.per_world_fields & GRID_FIELDS and not ctx.dynamic_lights):
      return None
    key = (bake_keys.grid_key(ctx), _variable_opacity(ctx, frame))
    if self.grid is None or self.grid_built_for != key:
      self.grid = self._build_grid(ctx, frame, backend)
      self.grid_built_for = key
      self.generation += 1
    checked = light_grid.checked(self.grid, ctx.tables.light_pack.lights, ctx.scene.nlight)
    if checked is not self.grid:
      self.grid = checked
      self.generation += 1
    return self.grid

  def _build_grid(self, ctx: FrameContext, frame: Frame, backend: TraceBackend) -> light_grid.Grid:
    """The light visibility grid of world 0 (`light_grid.build`), on the backend's hardware when
    it can; from the bake cache when it holds it (`bake_cache`, looked up before any BVH is
    built)."""
    # World 0's frame with the instances that move or change per world see-through (world 0's
    # occlusion would cull lights in the other worlds), for both tracers.
    dynamic = _bake_dynamic(ctx, frame)
    grid_frame = _see_through_dynamic(frame, dynamic)
    # The backend's hardware rays, when compiled; else the software kernel.
    tracer = backend.grid_tracer(ctx, grid_frame, grid_frame is not frame)
    lit = frame_structs.lights_struct(ctx, None)
    lit.lights = _every_light_on(lit.lights)  # simulator-driven activation also needs a baked entry
    lit.hot = pack.hot_records(lit.lights)
    layout = ctx.layout
    scene = ctx.scene
    # World 0's instance centres (a host read: the grid is built once).
    pos = np.zeros((scene.ninstance, 3), np.float32)
    if frame.instance_pos.size:
      pos[layout.columns] = host_copy(frame.instance_pos[:1]).reshape(-1, 3)[: len(layout.columns)]
    if layout.compact and len(layout.shared):
      pos[layout.shared] = host_copy(frame.shared_pos).reshape(-1, 3)[: len(layout.shared)]
    f = software_view(ctx.features, ctx.effective)
    radius = np.asarray(scene.bound_radius, dtype=np.float32).copy()
    unbounded = host_copy(scene.instance_type)[: scene.ninstance] == int(ShapeType.PLANE)
    radius[unbounded | dynamic] = np.inf
    host = (bake_keys.grid_key(ctx), f.traversal, tracer is not None, pos, radius, *bake_keys.shapes(ctx))
    key = bake_cache.key("light_grid", scene.device, bake_keys.trace_inputs(ctx, grid_frame), bake_keys.world0(lit), host=host)

    def trace_grid() -> tuple[light_grid.Grid, bake_cache.Bake]:
      backend.software_bvh(ctx, frame)  # the software rays' BVH, when the backend has none of its own
      if tracer is None:
        backend.software_meshes(ctx)
      geo = frame_structs.geometry(ctx, grid_frame)
      self._hide_flex(ctx, geo)
      grid = self._trace_grid(ctx, geo, lit, centres=(pos, radius), tracer=tracer)
      return grid, light_grid.to_arrays(grid)

    return bake_cache.cached(key, trace_grid, lambda stored: light_grid.from_arrays(stored, device=scene.device))

  @staticmethod
  def _trace_grid[NInstance: IntVar](
    ctx: FrameContext,
    geo: trace.SoftwareGeometry,
    lit: Lights,
    *,
    centres: tuple[np.ndarray[[NInstance, 3], np.dtype[np.float32]], np.ndarray[[NInstance], np.dtype[np.float32]]],
    tracer: GridTracer | None,
  ) -> light_grid.Grid:
    """`light_grid.build` of world 0 (`centres`: its instance centres and bounding radii)."""
    f = software_view(ctx.features, ctx.effective)
    pos, radius = centres
    return light_grid.build(
      geo=geo,
      lit=lit,
      shape_types=f.shape_types,
      pos=pos,
      radius=radius,
      cell=ctx.settings.lighting.light_grid_cell,
      area=f.area_lights,
      trees=f.trees,
      tracer=tracer,
      alpha_cutout=f.alpha_cutout,
      live_uv=f.live_material_uv,
      world_uv=f.world_uv,
      dynamic_lights=ctx.dynamic_lights,
      flood_dark=ctx.emission_varies,
    )

  def probes(self, ctx: FrameContext, frame: Frame, backend: TraceBackend, force: int = 0) -> None:
    """The frame's probe work: the GI bake on the first frame, then GI's periodic updates
    (`force`: that many updates now, none on the bake's frame). A change of `probe_key` drops the
    volume first (re-baked)."""
    f = ctx.effective  # what to bake: the settings' (the kernels: `ctx.features`)
    if not f.gi and self.gi is None:
      return  # nothing baked, nothing to bake: no key to build (a per-frame host cost)
    tracked = gi_tracked(ctx, frame)
    key = (bake_keys.probe_key(ctx), _variable_opacity(ctx, frame), tracked)
    # A world's inputs that match no group re-classify the worlds (`shade.gi.groups`): one re-bake.
    code = self.gi.groups.overflow_code() if self.gi is not None else 0
    overflowed = code != 0
    self._continuous = self._continuous or code == gi_groups.NEW_VARIANT
    if key != self.probes_built_for or overflowed:
      if self.gi is not None:
        self.drop_probes(_drop_reason(key == self.probes_built_for, code))
        self.generation += 1
      self.probes_built_for = key
    if not f.gi:
      return
    schedule, vol = self.schedule, self.gi
    periodic = ctx.settings.gi.update_every > 0
    frozen = vol is not None and schedule is not None and not force and not periodic
    if frozen and not tracked:
      # A frozen bake: only the check of the shared inputs (a light switched on re-bakes).
      if vol is not None and self._skips is not None:
        vol.groups.track(frame, self._skips, shared=True)
      return
    if vol is not None and vol.released and periodic and not force:
      # Periodic updates switched on after a frozen bake released the cube's staging arrays:
      # placed again (from the bake cache), keeping them.
      self.drop_probes(gi_lib.Reason.PERIODIC)
      self.generation += 1
      vol = None
    if frozen and vol is not None and vol.relight is None and self._skips is not None:
      # Tracked but frozen with shared groups: a world only moves between the baked groups (an
      # unseen input overflows and re-bakes): no probe work this frame.
      vol.groups.track(frame, self._skips, shared=True)
      return
    if vol is None or schedule is None:
      self._bake(ctx, frame, backend, tracked=tracked)
      return
    kernels = probe_kernels(ctx, relit=vol.relight is not None)
    # The probes see the static scenery only (`_probe_geometry`): after the bake the trees need no
    # refit for them (the per-world instances they hide are all that moves).
    traverse = vol.relight is not None or backend.gi_tracer(ctx, frame) is None
    geo = self._probe_geometry(ctx, frame, backend, traverse=traverse, refit=False)
    if self._skips is not None:
      vol.groups.track(frame, self._skips, shared=not periodic)
      if vol.relight is not None:
        vol.relight.set_policy(on_request=ctx.settings.gi.relight == "on_request")
        vol.relight.track(frame, self._skips)
    schedule.settings = ctx.settings.gi
    self.grid_for(ctx, frame, backend)
    inputs = self._probe_inputs(ctx, frame, geo, backend)
    if force:
      self.log.updates += force
      schedule.update(kernels, inputs, force)
    else:
      schedule.step(kernels, inputs)

  def _bake(self, ctx: FrameContext, frame: Frame, backend: TraceBackend, *, tracked: bool) -> None:
    """Places a new GI volume over the worlds' groups and bakes it (the light grid first: the
    probes' lights read it); the placed, baked volume comes from the bake cache when it holds it
    (looked up before any BVH is built: a frozen bake then never builds one).

    `tracked`: the radiance inputs can differ per world: the groups come from the frame's
    digests (`shade.gi.groups`, one host read here); mostly distinct worlds (`groups.relit`), or
    worlds that met new inputs after an earlier bake, get one shared volume of world 0's inputs and
    their tint (`gi.Relight`, computed here).
    """
    hidden = self._hidden(ctx, frame)
    self._warn_moving_lights(ctx)
    self._authored_frame(ctx, frame)  # the first bake's rows (kept for every later relit bake)
    skips = self._skips
    if skips is None:
      raise RuntimeError("the digests' skips are prepared with the hidden instances (_hidden)")
    digests = _digests(ctx, frame, skips) if tracked else None
    relit = digests is not None and (self._continuous or gi_groups.relit(digests))
    groups = gi_lib.Groups(ctx.nworld, digests=None if relit else digests, device=wp.get_device(ctx.scene.device))
    # The shared tables written in place: the source's live slots and the declared per-world ones
    # (one row with one world: `Renderer.per_world`).
    live = frozenset(name.replace(".", "_") for name in (*ctx.scene.binding.slots(None), *ctx.per_world_fields))
    see_host, see = _see_through(ctx)
    # A relit volume is baked with the authored inputs, the groups with their worlds' own.
    source = self._authored_frame(ctx, frame) if relit else frame
    key = bake_keys.gi_key(ctx, source, groups, see_host=see_host, hidden=hidden)
    stored = bake_cache.load(key)
    # The bake's probe and static rays on the backend's hardware (None: the software kernels).
    rays = backend.gi_tracer(ctx, frame)
    geo = None
    if stored is not None:
      vol = _new_volume(ctx, (stored["box"][0], stored["box"][1], float(stored["grid"][0]), float(stored["grid"][1])), groups)
    else:
      geo = self._probe_geometry(ctx, frame, backend, traverse=rays is None)
      placement = rays.placement(geo.instance_color) if rays is not None else None
      vol = _new_volume(ctx, _probe_grid(ctx, hidden, geo, see, placement), groups)
      f = software_view(ctx.features, ctx.effective)
      vol.place(
        geo,
        see,
        _emitters(frame, see_host),
        trees=f.trees,
        shape_types=f.shape_types,
        alpha_cutout=f.alpha_cutout,
        live_uv=f.live_material_uv,
        world_uv=f.world_uv,
        rays=placement,
      )
    self.log.bake(self._reason, cached=stored is not None)
    # A later bake without a drop: GI switched on again by a setting.
    self._reason = gi_lib.Reason.SETTINGS
    # Light positions checked in cells of half the probe spacing
    # (`shade.gi.groups.POSITION_QUANTUM`).
    groups.bake_shared(frame, live, skips, quantum=gi_groups.POSITION_QUANTUM * float(np.min(vol.spacing)))
    self.gi = vol
    self.generation += 1
    self.grid_for(ctx, frame, backend)
    self.schedule = gi_lib.Schedule(vol, ctx.settings.gi)
    kernels = probe_kernels(ctx, relit=relit)
    if stored is not None:
      vol.restore(stored)
    elif geo is not None:  # always: probed above without a stored bake
      shared = self._authored_geometry(ctx, source) if relit else geo
      self.schedule.bake(kernels.trace, self._inputs_of(self.current(ctx), source, shared, rays))
      bake_cache.save(key, vol.state())
    if relit and digests is not None:
      # The relight kernel traverses the software trees (`traverse`: the mesh BLASes built now).
      geo = self._probe_geometry(ctx, frame, backend, traverse=True, refit=geo is None)
      slot_of, nslot = gi_lib.assign_slots(_light_power(ctx), ctx.dynamic_lights)
      vol.relight = gi_lib.Relight(vol, digests, self.log.counter(ctx.nworld, wp.get_device(ctx.scene.device)), slot_of, nslot)
      vol.relight.adopt(frame, skips, source)
      vol.relight.bake(gi_lib.relight_kernel_of(kernels), gi_lib.relit_inputs(self._probe_inputs(ctx, frame, geo, backend)))
    if ctx.settings.gi.update_every <= 0:
      vol.release()

  def _probe_inputs(
    self, ctx: FrameContext, frame: Frame, geo: trace.SoftwareGeometry, backend: TraceBackend
  ) -> gi_lib.ProbeInputs:
    """The frame for the probe rays (a new light struct per `lights()` call): its own, or for a
    relit volume the authored inputs (`_authored_frame`) with the frame's own as `worlds` (the
    worlds' tints), and the relight kernel's view of both as `relit` (`_relit_inputs`)."""
    now, rays = self.current(ctx), backend.gi_tracer(ctx, frame)
    worlds = self._inputs_of(now, frame, geo, rays)
    if self.gi is None or self.gi.relight is None:
      return worlds
    authored = self._authored_frame(ctx, frame)
    shared = self._authored_geometry(ctx, authored)
    relit = self._relit_inputs(now, authored, shared)._replace(worlds=self._relit_inputs(now, frame, geo))
    return self._inputs_of(now, authored, shared, rays)._replace(worlds=worlds, relit=relit)

  def _relit_inputs(self, now: FrameContext, frame: Frame, geo: trace.SoftwareGeometry) -> gi_lib.ProbeInputs:
    """`ProbeInputs` of `frame` as the relight kernel reads them (`_relit_structs`), kept while the
    frame holds the same arrays at the same generation and light grid: built every frame, they
    cost a relit volume's steady frames 0.5-1.3 ms of host time (RTX PRO 6000, 512 worlds)."""
    for held, generation, grid, inputs in self._relit:
      if generation == now.generation and grid is self.grid and frame.same(held):
        return inputs
    inputs = self._relit_structs(now, frame, geo)
    self._relit = [*self._relit[-1:], (frame, now.generation, self.grid, inputs)]
    return inputs

  def _relit_structs(self, now: FrameContext, frame: Frame, geo: trace.SoftwareGeometry) -> gi_lib.ProbeInputs:
    """`ProbeInputs` of `frame` for the relight kernel (`ctx.relight`, its build): its material
    carrier, the relight word (`tiers.relight_word`) in every copy (`geo` copied)."""
    build = _relight_build(now)
    word = tiers.relight_word(build, now.effective)
    own = dataclasses.replace(now, features=build)
    worded = type(geo)()
    for name, value in bake_cache.fields(geo).items():
      setattr(worded, name, value)
    toggles.store(worded, word)
    mats = frame_structs.materials(own, frame)
    mats.bits = word & toggles.HALF  # the first half (`GeometryFields.bits2` the second)
    current = self._current_lights(now)

    def lights() -> Lights:
      """A new relight light struct of the frame."""
      lit = current()
      toggles.store(lit, word)
      return lit

    def environment() -> EnvironmentArrays:
      """A new relight environment struct of the frame."""
      env = frame_structs.probe_environment(now, frame)
      toggles.store(env, word)
      return env

    return gi_lib.ProbeInputs(worded, mats, environment(), lights, environment)

  def _inputs_of(
    self, now: FrameContext, frame: Frame, geo: trace.SoftwareGeometry, rays: gi_lib.GiRays | None
  ) -> gi_lib.ProbeInputs:
    """`ProbeInputs` of `frame` (`rays`: the backend's hardware probe rays, None: the software
    kernel; with the probe geometry `geo`'s colors)."""
    current = self._current_lights(now)
    hardware = None
    if rays is not None:

      def hardware(q: gi_lib.kernels.ProbeLaunch, n: int, lit: Lights, env: EnvironmentArrays) -> None:
        """`n` units of probe rays on the backend's hardware (`gi.ProbeInputs.rays`)."""
        rays.probe(q, n, source=frame, colors=geo.instance_color, lit=lit, env=env)

    return gi_lib.ProbeInputs(
      geo,
      frame_structs.materials(now, frame),
      frame_structs.probe_environment(now, frame),
      current,
      lambda: frame_structs.probe_environment(now, frame),
      rays=hardware,
    )

  def _current_lights(self, now: FrameContext):
    """A factory of the probes' light struct of `now` read when called: the light grid and the
    static lights as they are then (a grid rebuilt after the factory was made is the one used).
    It holds a weak reference to this `Baked`, not `self`: the probe inputs that keep it are kept
    by `self` (no reference cycle)."""
    owner = weakref.ref(self)

    def lights() -> Lights:
      """The probes' light struct (the owner's current grid and static lights)."""
      baked = owner()
      if baked is None:
        raise RuntimeError("the renderer's bakes were released: its probe inputs are no longer valid")
      return baked._probe_lights(now).of(frame_structs.lights_struct(now, baked.grid))

    return lights

  def _probe_lights(self, ctx: FrameContext) -> gi_lib.StaticLights:
    """The probes' view of the lights: the dynamic ones (`PerWorld.dynamic_lights`) off."""
    lights = self._static_lights
    if lights is None or lights.ids != ctx.dynamic_lights:
      lights = self._static_lights = gi_lib.StaticLights(ctx.scene.nlight, ctx.dynamic_lights, wp.get_device(ctx.scene.device))
    return lights

  def _warn_moving_lights(self, ctx: FrameContext) -> None:
    """Warns once per light about the lights the simulator moves (`Binding.moving_lights`) that
    are not dynamic: each move re-lights the GI of every world it changes."""
    moving = frozenset(ctx.scene.binding.moving_lights()) - frozenset(ctx.dynamic_lights) - self._warned
    if moving:
      self._warned |= moving
      ids = sorted(moving)
      warnings.warn(
        f"lights {ids} move with the simulation but are not dynamic: GI re-lights the worlds whenever "
        f"they move (about a probe update per move); declare them PerWorld(dynamic_lights={ids}) for "
        "direct light and shadows only, with no GI and no re-light",
        RuntimeWarning,
        stacklevel=4,
      )

  def _authored_frame(self, ctx: FrameContext, frame: Frame) -> Frame:
    """`frame` with its per-world GI input tables replaced by the authored ones: the scene's
    own one-row table of that slot (`GpuScene.tables`: what the worlds start from, whatever
    their history or world 0's randomisation), else world 0's rows at the first GI bake
    (`_authored`: a table first seen now is taken from this frame)."""
    rows: dict[str, AnyDeviceArray] = {}
    for name, table in gi_lib.per_world_inputs(frame):
      kept = self._authored.get(name)
      if kept is None or kept.shape[1:] != table.shape[1:] or kept.dtype != table.dtype:
        scene_table = bind.slot(ctx.scene.tables, _SLOT_OF.get(name, ""))
        if scene_table is not None and scene_table.shape == (1, *table.shape[1:]) and scene_table.dtype == table.dtype:
          kept = scene_table[:1]
        else:
          kept = wp.clone(table[:1])
        self._authored[name] = kept
      rows[name] = kept
    return dataclasses.replace(frame, **rows)

  def _hidden(self, ctx: FrameContext, frame: Frame) -> np.ndarray[[int], np.dtype[np.bool_]]:
    """(ninstance,) the instances the probes do not see (moving / per-world: `_dynamic`), with their
    device mask (`_static_mask`) and the buffer of the colors that hide them allocated."""
    hidden = _dynamic(ctx.layout)
    if self._static_colors is None or self._static_colors.shape != frame.instance_color.shape:
      self._static_colors = wp.empty(frame.instance_color.shape, dtype=wp.vec4, device=ctx.scene.device)
      self._hidden_host = None
    if self._hidden_host is None or not np.array_equal(self._hidden_host, hidden) or self._skips is None:
      dev = ctx.scene.device
      self._static_mask = wp.array(hidden.astype(np.int32), dtype=wp.int32, device=dev)
      self._hidden_host = hidden
      self._skips = gi_groups.Skips(self._static_mask, self._probe_lights(ctx).mask, wp.zeros(0, dtype=wp.int32, device=dev))
    return hidden

  def _probe_geometry(
    self, ctx: FrameContext, frame: Frame, backend: TraceBackend, *, traverse: bool, refit: bool = True
  ) -> trace.SoftwareGeometry:
    """Static geometry for every probe ray and relocation survey, with dynamic hits hidden
    (`refit` False: the software trees as they are, built on first use). `traverse`: a software
    kernel traces it (no hardware GI rays, or the relight kernel), so the mesh BLASes are built;
    the hardware rays read its colors only and the volume's placement the scene BVH's boxes."""
    if refit or ctx.software.bvh is None:
      backend.software_bvh(ctx, frame)
    if traverse:
      backend.software_meshes(ctx)
    geo = frame_structs.geometry(ctx, frame)
    self._hidden(ctx, frame)
    if self._static_mask is None or self._static_colors is None:
      raise RuntimeError("static bake mask must be allocated with its color buffer")
    wp.launch(
      _mask_colors,
      dim=self._static_colors.shape,
      inputs=[frame.instance_color, self._static_mask],
      outputs=[self._static_colors],
      device=ctx.scene.device,
    )
    geo.instance_color = self._static_colors
    self._hide_flex(ctx, geo)
    return geo

  def _authored_geometry(self, ctx: FrameContext, authored: Frame) -> trace.SoftwareGeometry:
    """The probe geometry (`_probe_geometry`'s trees, built first) of the authored frame
    (`_authored_frame`): its colors (`geo.material_base_color`, `geo.instance_color`) are GI
    inputs too."""
    geo = frame_structs.geometry(ctx, authored)
    colors = authored.instance_color
    if self._static_mask is None:
      raise RuntimeError("the static mask is prepared with the hidden instances (_hidden)")
    if self._authored_colors is None or self._authored_colors.shape != colors.shape:
      self._authored_colors = wp.empty(colors.shape, dtype=wp.vec4, device=ctx.scene.device)
    wp.launch(
      _mask_colors,
      dim=colors.shape,
      inputs=[colors, self._static_mask],
      outputs=[self._authored_colors],
      device=ctx.scene.device,
    )
    geo.instance_color = self._authored_colors
    self._hide_flex(ctx, geo)
    return geo

  def _hide_flex(self, ctx: FrameContext, geo: trace.SoftwareGeometry) -> None:
    """Deformables never occlude shared bakes (the OptiX bakes' rays leave them out by their
    visibility mask, `trace.optix.device.trace.BAKE_RAY_MASK`); retain BVH indexing while hiding
    their hits. A scene without deformables keeps its (placeholder) part table: no deformables
    module loads (a GI bake switched on at run time compiled it)."""
    if not ctx.scene.deformables.count:
      return
    if self._hidden_flex is None:
      self._hidden_flex = deformable_bvh.hidden(geo.flex_part)
    geo.flex_part = self._hidden_flex


def _drop_reason(same_key: bool, code: int) -> gi_lib.Reason:
  """Why the probes re-bake: their key changed (a setting), else the groups' overflow `code`."""
  if not same_key:
    return gi_lib.Reason.SETTINGS
  return gi_lib.Reason.NEW_VARIANT if code == gi_groups.NEW_VARIANT else gi_lib.Reason.SHARED


# Frame field -> its slot ("material_base_color" -> "material.base_color", `binding.SLOTS`).
_SLOT_OF: Final = {field: slot for slot, field in bind.SLOTS.items()}


@wp.kernel(enable_backward=False)
def _mask_colors(
  # In:
  colors: wp.array2d[wp.vec4],
  dynamic: wp.array[wp.int32],
  # Out:
  static_colors: wp.array2d[wp.vec4],
):
  """Every world's colors with the `dynamic` instances hidden (alpha 0), refreshed on device
  (captured periodic GI updates too)."""
  w, i = wp.tid()
  color = colors[w, i]
  if i < dynamic.shape[0] and dynamic[i] != 0:
    color[3] = 0.0
  static_colors[w, i] = color


def _bake_dynamic(ctx: FrameContext, frame: Frame) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """Moving/variable geometry cannot be an occluder of any shared lighting bake.

  Appearance DR may change alpha-cutout textures, their cutoff or their projection, or swap
  materials, without changing the sharing layout: every instance is then excluded rather than
  baking world 0's visibility. Per-world colors (`instance.color`, `material.base_color`, the
  most common randomization) do not: the bakes see world 0's opacity (hide an instance per world
  with `instance.visible`, which makes it dynamic).
  """
  dynamic = _dynamic(ctx.layout)
  if _variable_opacity(ctx, frame):
    dynamic[:] = True
  return dynamic


def _variable_opacity[S, NWorld: IntVar, NColumn: IntVar](ctx: FrameContext[S], frame: Frame[NWorld, NColumn]) -> bool:
  """Whether appearance tables may change shared occluders' cut-outs independently per world
  (colors do not count: `_bake_dynamic`)."""
  fields = (
    ("instance.material", frame.instance_material),
    ("material.texture", frame.material_texture),
    ("material.alpha_cutoff", frame.material_alpha_cutoff),
    ("material.tex_affine", frame.material_tex_affine),
    ("material.tex_variation", frame.material_tex_variation),
    ("material.tex_repeat", frame.material_tex_repeat),
    ("material.tex_uniform", frame.material_tex_uniform),
    ("material.uv_world", frame.material_uv_world),
  )
  return any(name in ctx.per_world_fields or values.shape[0] > 1 for name, values in fields)


def _see_through(
  ctx: FrameContext,
) -> tuple[np.ndarray[[int], np.dtype[np.int32]], wp.array[wp.int32, [int]]]:
  """The (ninstance,) int32 flags of the instances probe rays pass, on the host (for the bakes'
  cache keys) and the scene's device: window glass, and the instances that move or change per
  world (a robot baked into the probe placement would leave its occlusion
  behind when it moves)."""
  see = np.any(shadow_transmit(ctx.scene) > 0.0, axis=1)
  see[: ctx.scene.ninstance] |= _dynamic(ctx.layout)
  flags = see.astype(np.int32)
  return flags, wp.array(flags, dtype=wp.int32, device=ctx.scene.device)


def _light_power(ctx: FrameContext) -> np.ndarray[[int], np.dtype[np.float32]]:
  """(nlight,) world 0's packed lights' luminous power (luminance of colour x intensity, 0 when
  off): the linear re-light's slot ranking (`shade.gi.slots.assign`; a host read, once per bake)."""
  rows = host_copy(ctx.tables.light_pack.lights[:1])[0][: ctx.scene.nlight]
  color = np.asarray(record_field(rows, "color"), np.float32).reshape(-1, 3)
  intensity = np.asarray(record_field(rows, "intensity"), np.float32)
  active = np.asarray(record_field(rows, "active"), np.float32)
  lum = color @ np.asarray(REC709_Y, np.float32)
  return (lum * intensity * active).astype(np.float32)


def _emitters(frame: Frame, see: np.ndarray[[int], np.dtype[np.int32]]) -> wp.array[wp.int32, [int]]:
  """(ninstance,) int32 1 for the instances the contact occlusion's rays pass (`gi.ao_kernel`):
  `see` (`_see_through`) and the instances of emissive materials in world 0 (an emitter next to a
  surface lights it; it does not occlude it). A host read of world 0's material rows (once per
  placement)."""
  out = see.copy()
  device = frame.instance_material.device
  if frame.instance_material.size == 0 or frame.material_emission.size == 0:
    return wp.array(out, dtype=wp.int32, device=device)
  material = host_copy(frame.instance_material[:1]).reshape(-1)
  emission = host_copy(frame.material_emission[:1]).reshape(-1)
  n = min(len(out), len(material))
  ids = material[:n]
  emits = np.zeros(n, bool)
  valid = (ids >= 0) & (ids < len(emission))
  emits[valid] = emission[ids[valid]] > 0.0
  out[:n] |= emits.astype(np.int32)
  return wp.array(out, dtype=wp.int32, device=device)


def _see_through_dynamic(frame: Frame, dynamic: np.ndarray[[int], np.dtype[np.bool_]]) -> Frame:
  """`frame` with the `dynamic` instances hidden in world 0 (alpha 0 in a one-row copy of its
  colors: the software traversal skips them, the OptiX instance writers hide them), for the grid's
  rays; `frame` itself without any."""
  if not np.any(dynamic):
    return frame
  color = host_copy(frame.instance_color[:1]).reshape(1, -1, 4).copy()  # host_copy aliases CPU arrays
  n = min(color.shape[1], len(dynamic))
  color[0, :n, 3] = np.where(dynamic[:n], 0.0, color[0, :n, 3])
  row = wp.array(color, dtype=wp.vec4, device=frame.instance_color.device)
  return dataclasses.replace(frame, instance_color=row)


def _dynamic(layout: Layout) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(ninstance,) the instances the binding moves or `PerWorld` declares (pose, size, mesh,
  visibility)."""
  out = np.asarray(layout.dynamic, bool).copy()
  for mask in layout.groups.values():
    out |= np.asarray(mask, bool)
  return out


def _first_rows[D, N: IntVar](a: wp.array[D, [N]], n: int) -> AnyHostArray:
  """Host copy of the first `n` entries of device array `a` (a device -> host copy of those
  only, not of every world's; Warp cannot slice an empty array)."""
  if a.size == 0:
    return host_copy(a)
  head = host_copy(a[: max(n, 1)])
  return head[:n]


# Probe cells each axis of a default probe box spans at least (`_probe_box`).
MIN_PROBE_CELLS: Final = 2


def _probe_box(lo: Point, hi: Point, spacing: float) -> tuple[Point, Point]:
  """[lo, hi] grown about its centre until every axis spans MIN_PROBE_CELLS probe spacings.

  Probes sit at cell centres: an axis shorter than the spacing gets one probe in its middle, i.e.
  inside a lone object (a sphere: the probe is inactive and the scene unlit) or under a table top
  (a 1 m pick table got 2 probes, both under its top). Two cells put
  the probes on either side: around the object, above the table (the ones below a floor plane
  see its back faces and are disabled). Axes already that long keep their bounds exactly (not
  recomputed from the centre: that would move every probe by a rounding error).
  """
  centre = 0.5 * (lo + hi)
  half = 0.5 * MIN_PROBE_CELLS * spacing
  short = hi - lo < MIN_PROBE_CELLS * spacing
  return np.where(short, centre - half, lo), np.where(short, centre + half, hi)


def gi_tracked(ctx: FrameContext, frame: Frame) -> bool:
  """Whether the worlds' GI radiance inputs can differ (`shade.gi.groups`): declared per world, or a
  table of the frame with more than one row (metadata only)."""
  declared = any(gi_lib.gi_slot(name) for name in ctx.per_world_fields)
  return ctx.nworld > 1 and (declared or bool(gi_lib.per_world_inputs(frame)))


def _digests(ctx: FrameContext, frame: Frame, skips: gi_groups.Skips[int]) -> np.ndarray[[int, 2], np.dtype[np.uint32]]:
  """The worlds' GI input digests (`shade.gi.groups.digest`), read to the host (once per bake)."""
  out = wp.zeros((ctx.nworld, 2), dtype=wp.uint32, device=wp.get_device(ctx.scene.device))
  return host_copy(gi_lib.digest(frame, ctx.nworld, skips, out))


def probe_kernels(ctx: FrameContext, *, relit: bool) -> gi_lib.ProbeKernels:
  """The probe kernels a bake of `ctx` launches: its software view's (`software_view`), the
  relight kernel only for a relit volume."""
  f = software_view(ctx.features, ctx.effective)
  return gi_lib.ProbeKernels(probe.probe_kernel(f), relight_kernel(ctx) if relit else None)


def relight_kernel(ctx: FrameContext) -> wp.Kernel:
  """The relight kernel of a relit bake of `ctx` (its relight build's, `resolve.relight_build`:
  one per probe-relevant settings; the warm-up and the prebuilt fill list it)."""
  return probe.relight_kernel(_relight_build(ctx))


def _relight_build(ctx: FrameContext) -> Features:
  """`ctx.relight`, resolved with GI on.

  Raises:
    RuntimeError: GI is off (no relight build).
  """
  if ctx.relight is None:
    raise RuntimeError("the relight kernel needs GI on (FrameContext.relight)")
  return ctx.relight


def _probe_grid(
  ctx: FrameContext,
  hidden: np.ndarray[[int], np.dtype[np.bool_]],
  geo: trace.SoftwareGeometry,
  see: wp.array[wp.int32, [int]],
  rays: gi_lib.PlacementRays | None = None,
) -> tuple[Point, Point, float, float]:
  """(box corners, target spacing [m], largest cube texel [m]) of a probe volume. The derived
  spacing: from the scene's free distance (`gi.free_path` over the static instances' bounds,
  `gi.target_spacing`); the spacing: gi.spacing, else the derived one; the texel: the derived
  spacing / `cube.SUBDIV` (a coarser gi.spacing keeps the scene's texel size; the cube's
  budget coarsens it: `cube.subdivision`). The box: gi.bounds, else the bounds of world 0's static
  instances (the software BVH's boxes: it is built first; `hidden` instances left out; all of
  world 0's instances when only planes are static: a ground plane under moving objects), padded by
  one spacing (the boundary probes and texels lie in the open air around the scenery, which is
  what points outside the box read) and at least MIN_PROBE_CELLS spacings per axis
  (`_probe_box`). `rays`: the free-path survey on the backend's hardware."""
  s = ctx.settings
  bvh = ctx.software.bvh
  if bvh is None:
    raise RuntimeError("the GI volume is placed after the software BVH is built (Baked._probe_geometry)")
  # World 0's boxes (the first `ninstance` entries) and the shared ones: host reads of those only.
  n = bvh.ninstance
  all_types = host_copy(ctx.scene.instance_type)
  ids = np.concatenate([_first_rows(bvh.enabled_ids, n), host_copy(bvh.shared_ids)[: bvh.nshared]])
  types = all_types[ids]
  lower = np.concatenate([_first_rows(bvh.lower, n), host_copy(bvh.shared_lower)]).reshape(-1, 3)
  upper = np.concatenate([_first_rows(bvh.upper, n), host_copy(bvh.shared_upper)]).reshape(-1, 3)
  use = ~hidden[ids]
  if not np.any(use & (types != int(ShapeType.PLANE))):
    use = np.ones_like(use)
  lower, upper, types = lower[use], upper[use], types[use]
  derived = _derived_spacing(ctx, geo, see, lower, upper, types, rays)
  spacing = derived if s.gi.spacing is None else s.gi.spacing
  cell = derived / gi_cube.SUBDIV
  if s.gi.bounds is not None:
    return np.asarray(s.gi.bounds[0]), np.asarray(s.gi.bounds[1]), spacing, cell
  lo, hi = gi_lib.scene_bounds(lower, upper, types, spacing=spacing, max_probes=s.gi.max_probes)
  lo, hi = _probe_box(lo - spacing, hi + spacing, spacing)
  return lo, hi, spacing, cell


def _derived_spacing[N: IntVar](
  ctx: FrameContext,
  geo: trace.SoftwareGeometry,
  see: wp.array[wp.int32, [int]],
  lower: np.ndarray[[N, 3], np.dtype[np.float32]],
  upper: np.ndarray[[N, 3], np.dtype[np.float32]],
  types: np.ndarray[[N], np.dtype[np.int32]],
  rays: gi_lib.PlacementRays | None,
) -> float:
  """The scene's derived probe spacing [m] (`gi.target_spacing` of its free distance,
  `gi.free_path`), surveyed over the instances' main cluster (`gi.cluster_bounds`, boxes more
  than two median instance diagonals apart: far scenery out in empty space is left out)."""
  s, f = ctx.settings, software_view(ctx.features, ctx.effective)
  lo, hi = gi_lib.scene_bounds(lower, upper, types, spacing=math.inf, max_probes=s.gi.max_probes)
  if len(lower):
    diagonal = float(np.median(np.linalg.norm(upper - lower, axis=1)))
  else:
    diagonal = np.linalg.norm(hi - lo).astype(np.float64).item()
  lo, hi = gi_lib.cluster_bounds(lower, upper, types, gap=2.0 * diagonal)
  free = gi_lib.free_path(
    geo,
    see,
    lo,
    hi,
    device=wp.get_device(ctx.scene.device),
    shape_types=f.shape_types,
    trees=f.trees,
    alpha_cutout=f.alpha_cutout,
    live_uv=f.live_material_uv,
    world_uv=f.world_uv,
    rays=rays,
  )
  return gi_lib.target_spacing(free, lo, hi, max_probes=s.gi.max_probes, diagonal=diagonal)


def _new_volume(
  ctx: FrameContext,
  grid: tuple[Point, Point, float, float],
  groups: gi_lib.Groups | None = None,
) -> gi_lib.ProbeVolume:
  """A probe volume over `grid` (box corners, target spacing, largest texel: `_probe_grid`) and
  `groups` (None: one)."""
  s = ctx.settings
  return gi_lib.ProbeVolume(
    grid[0],
    grid[1],
    spacing=grid[2],
    cell=grid[3],
    settings=s.gi,
    scale=exposure_from_ev100(s.exposure.render_ev100()),
    device=wp.get_device(ctx.scene.device),
    groups=groups,
  )
