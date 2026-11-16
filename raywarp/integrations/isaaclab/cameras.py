"""`Cameras`: Isaac Lab's renderer interface (`BaseRenderer`) implemented on raywarp, without
importing Isaac Lab (`renderer.RaywarpRenderer` binds it to Isaac's classes).

One GpuScene of the authored USD visuals (or an explicit model-only source), bound to the Newton
model through `NewtonBinding`, is shared by every camera; each camera (`create_render_data`) is a
`CameraSession`, one camera of a `CameraGroup`: a Renderer with a render world per environment and a
camera per Isaac camera rendering alike (same environments, size, outputs, clips and settings), so
`render_batch` renders each group once. Body poses come from Isaac's scene data provider (PhysX
tensor views or Newton state alike), once per frame (`update_transforms`), copied into the
`newton.State` the sessions render (whose arrays never change: captured graphs replay).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, final

import newton
import numpy as np
import warp as wp

from raywarp.core.pipeline.upload.build import upload
from raywarp.core.pipeline.upload.options import SceneOptions
from raywarp.formats.usd import options as usd_options
from raywarp.integrations.isaaclab import contract
from raywarp.integrations.isaaclab.materials import MaterialLayout, MaterialWriter, PreparedColors
from raywarp.integrations.isaaclab.scalars import ScalarChannel, ScalarTable, ScalarWriter, scalar_table
from raywarp.integrations.isaaclab.segments import SegmentTables, segments, world_shapes
from raywarp.integrations.isaaclab.session import Appearance, CameraConfig, CameraGroup, CameraSession
from raywarp.integrations.newton.api import binding as newton_binding
from raywarp.integrations.newton.build import load as newton_load
from raywarp.integrations.newton.from_usd import load_usd
from raywarp.integrations.newton.types import Options, default_lights
from raywarp.scene.lighting.rig import Lighting
from raywarp.settings import RenderSettings
from raywarp.settings.render import mapped

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.integrations.isaaclab.materials import WorldMask
  from raywarp.integrations.isaaclab.types import Buffer, CameraData, CameraRenderSpec, Config, ProxyArray, Simulation
  from raywarp.integrations.isaaclab.usd_source import UsdSource
  from raywarp.integrations.newton.binding import IntTable, NewtonBinding
  from raywarp.integrations.newton.types import NewtonState
  from raywarp.scene.model.scene import Scene


# The model's rigid shapes only (`initialize`: particle meshes are not read from Isaac).
_RIGID: Final = Options(particle_meshes=False)


def render_settings(config: Config) -> RenderSettings:
  """The default settings with the config's dotted overrides; a `lighting`'s `ev100` is the
  manual exposure (`exposure.ev100`) unless the overrides set it, as in the Newton front ends.

  Raises:
    ValueError: for an unknown setting.
  """
  lit = config.lighting
  exposure = {} if lit is None else {"exposure.mode": "manual", "exposure.ev100": lit.ev100}
  return mapped(RenderSettings(), exposure, config.settings)


@final
class Cameras:
  """raywarp's renderer of Isaac Lab cameras (see the module doc); methods follow `BaseRenderer`.

  Attributes:
    config: the renderer config (`RaywarpRendererCfg`).
    scene: the GpuScene (after `initialize`).
    sessions: the live cameras.
    groups: the cameras' groups (one Renderer each).
  """

  def __init__(self, config: Config) -> None:
    """Pre-physics: nothing is built before `initialize` (the Newton model does not exist yet)."""
    self.config = config
    self.sessions: list[CameraSession] = []
    self.groups: list[CameraGroup] = []
    self._sim: Simulation | None = None
    self._scene: GpuScene[NewtonState] | None = None
    self._binding: NewtonBinding | None = None
    self._segments: dict[int, SegmentTables] = {}
    self._material_layout: MaterialLayout | None = None
    self._material_colors: PreparedColors | None = None
    self._material_scalars: dict[str, ScalarTable] = {}
    self._state: newton.State | None = None
    self._usd_shapes: IntTable | None = None
    # (ninstance,) the USD-only visuals (no Newton shape in any world: `host.instances.source` < 0).
    self._usd_only: np.ndarray[[int], np.dtype[np.bool_]] | None = None
    self._usd_materials: tuple[tuple[str, ...], ...] | None = None
    self._visibility: np.ndarray[[int, int], np.dtype[np.bool_]] | None = None

  def initialize(self, sim: Simulation) -> None:
    """Post-physics: upload authored USD visuals, or explicit model-only shapes, bound to bodies."""
    model = sim.model
    c = self.config
    source = sim.usd_source()
    if source is None:
      lighting = Lighting(default_lights(model.up_axis)) if c.lighting is None else c.lighting
      # Isaac's own per-world material tables color the shapes (not `world_colors`).
      adapter = Options(load_textures=c.load_textures, particle_meshes=False, world_colors=False)
      host = newton_load(model, adapter=adapter, lighting=lighting)
      binding = newton_binding(host, model, adapter=adapter)
    else:
      host, binding = self._ingest_usd(model, source)
    self._binding = binding
    if model.tri_count:
      # Isaac publishes particle positions through its geometry update, which this adapter does not
      # read (`update_geometries`): rendered at their initial pose they would be wrong.
      host.report.drop("isaac.particle_meshes(unsupported)", n=model.tri_count)
    if self._usd_materials is None:
      shapes = world_shapes(binding, max(model.world_count, 1))
      shape_materials = list(sim.shape_materials()) or [""] * model.shape_count
      paths = tuple(tuple(shape_materials[int(s)] if s >= 0 else "" for s in row) for row in shapes)
    else:
      paths = self._usd_materials
    self._material_layout = MaterialLayout.prepare(host, paths)
    # Every session renders with `render_settings(c)`: without emissive lights the upload skips
    # the panels' fit.
    emissive = render_settings(c).lighting.emissive_lights
    scene = self._scene = upload(host, model.device, binding=binding, options=SceneOptions(emissive_lights=emissive))
    if source is not None:
      source.place_lights(host, scene, model)
    self._sim = sim
    self._state = model.state()
    self.update_transforms()

  def _ingest_usd(self, model: newton.Model, source: UsdSource) -> tuple[Scene, NewtonBinding]:
    """The host scene of the authored USD visuals of `source` and its binding; keeps their shape
    rows, material addresses and per-world visibility."""
    paths = {"body_paths": source.body_paths, "world_local_roots": source.roots, "world_homes": source.homes}
    usd = usd_options.Options(lux_dialect=source.lux_dialect)
    host = load_usd(model, source.stage, lighting=self.config.lighting, usd=usd, **paths)
    binding = newton_binding(host, model, adapter=_RIGID, stage=source.stage, **paths)
    self._usd_shapes = source.shape_rows(host, model)
    native = host.instances.source >= 0
    self._usd_shapes[:, native] = world_shapes(binding, len(source.roots))[:, native]
    self._usd_only = ~np.asarray(native, bool)
    self._usd_materials = source.material_paths(host)
    visible = source.visibility(host)
    self._visibility = visible if not visible.all() else None
    if not self.config.load_textures:
      host.report.drop("isaac.load_textures(USD override unsupported)")
    return host, binding

  @property
  def scene(self) -> GpuScene[NewtonState]:
    """The GpuScene (raises before `initialize`)."""
    if self._scene is None:
      raise RuntimeError("raywarp's Isaac Lab renderer is not initialized (initialize() runs after physics is ready)")
    return self._scene

  def _simulation(self) -> Simulation:
    """The simulation (raises before `initialize`)."""
    if self._sim is None:
      raise RuntimeError("raywarp's Isaac Lab renderer is not initialized")
    return self._sim

  def create_render_data(self, spec: CameraRenderSpec) -> CameraSession:
    """A session rendering camera `spec` in every environment.

    Raises:
      ValueError: for an output this renderer does not write.
      NotImplementedError: for a lens-distortion camera.
    """
    cfg, c = spec.cfg, self.config
    names = tuple(cfg.data_types)
    contract.check_names(names)
    spawn = cfg.spawn
    if spawn is not None and spawn.distortion is not None:
      raise NotImplementedError("raywarp's Isaac Lab renderer does not render lens distortion (spawn.distortion) yet")
    near, far = (float(v) for v in spawn.clipping_range) if spawn is not None else (None, None)
    n = spec.num_instances
    config = CameraConfig(
      nworld=n,
      width=cfg.width,
      height=cfg.height,
      names=names,
      clipping=c.depth_clipping_behavior,
      znear=near,
      zfar=far,
      graph=c.use_cuda_graph,
      period=float(cfg.update_period),
    )
    settings = render_settings(c)
    if cfg.background_color is not None:
      # Isaac's contract: misses show this solid color, also in front of the environment map.
      color = tuple(float(v) for v in cfg.background_color)
      settings = settings.replace(**{"environment.background_color": color, "environment.background": False})
    session = CameraSession(config, self._group(config, settings))
    self.sessions.append(session)
    return session

  def _group(self, config: CameraConfig, settings: RenderSettings) -> CameraGroup:
    """A new group for cameras rendering like `config` with `settings`."""
    wants_seg = bool({contract.INSTANCE, contract.SEMANTIC} & set(config.names))
    appearance = Appearance(
      segments=self._segments_of(config.nworld) if wants_seg else None,
      visibility=self._visibility,
      material_colors=None if self._material_colors is None else self._material_colors.table,
      material_scalars=dict(self._material_scalars),
    )
    group = CameraGroup(self.scene, config, settings, appearance)
    self.groups.append(group)
    return group

  def _regroup(self, sessions: Sequence[CameraSession]) -> None:
    """Makes the alike cameras of a call one group each (exactly them): cameras Isaac renders
    together share a renderer; one rendered alone (Isaac's lazy per-camera reads) gets its own.
    Stable call patterns regroup once."""
    alike: list[list[CameraSession]] = []
    for session in sessions:
      for members in alike:
        first = members[0]
        if first.config == session.config and first.group.settings == session.group.settings:
          members.append(session)
          break
      else:
        alike.append([session])
    for members in alike:
      group = members[0].group
      if all(m.group is group for m in members) and len(group.sessions) == len(members):
        continue
      target = self._group(members[0].config, group.settings)
      for member in members:
        member.move(target)
    self.groups = [g for g in self.groups if g.sessions]

  def _segments_of(self, nworld: int) -> SegmentTables:
    """The scene's segment ids for `nworld` environments (built once)."""
    if nworld not in self._segments:
      sim, binding = self._simulation(), self._binding
      if binding is None:
        raise RuntimeError("Cameras: the scene is not initialized (initialize(sim) first)")
      if self._usd_shapes is not None and self._usd_only is not None:
        # Shown and without a shape: a native instance's -1 is a world that hides it
        # (`world_shapes`), a USD visual's a world that does not show it (`visibility`).
        shown = self._visibility if self._visibility is not None else True
        unlabelled = int(np.sum((self._usd_shapes < 0) & self._usd_only[None] & shown))
        if unlabelled:
          self.scene.report.drop("isaac.segmentation.USD_only_visual(unlabelled)", n=unlabelled)
      self._segments[nworld] = segments(binding, nworld, sim, self.scene.device, shapes=self._usd_shapes)
    return self._segments[nworld]

  def set_outputs(self, session: CameraSession, output_data: Mapping[str, ProxyArray[Buffer]]) -> None:
    """Renders into Isaac's output buffers (`"rgb"` is Isaac's view of `"rgba"`: skipped)."""
    session.bind({name: proxy.warp for name, proxy in output_data.items() if name != "rgb"})

  def update_transforms(self) -> None:
    """Copies the body poses of this frame from the simulation into the state the sessions
    render (a device copy of the bodies' transforms; none when the simulation publishes into that
    state's own storage). The state's array never changes, so a captured graph replays even when
    the provider hands out another buffer (a double-buffered physics state, a converted copy)."""
    poses = self._simulation().body_poses()
    state = self._state
    if poses is None or state is None or state.body_q is None:
      return
    if poses.shape != state.body_q.shape:
      raise ValueError(f"the simulation published {poses.shape[0]} body poses, the model has {state.body_q.shape[0]}")
    if poses.ptr != state.body_q.ptr:
      wp.copy(state.body_q, poses)

  def update_camera(
    self,
    session: CameraSession,
    positions: ProxyArray[wp.array[wp.vec3, [int]]],
    orientations: ProxyArray[wp.array[wp.quatf, [int]]],
    intrinsics: ProxyArray[wp.array[wp.mat33, [int]]],
  ) -> None:
    """Poses camera `session` in every environment (world convention, quaternion x y z w)."""
    session.update(positions.warp, orientations.warp, intrinsics.warp)

  def update_camera_intrinsics(self, session: CameraSession, intrinsics: wp.array[wp.mat33, [int]]) -> None:
    """New per-environment pixel intrinsics (any environment apart: no shared ray field)."""
    session.set_intrinsics(intrinsics)

  def render(self, session: CameraSession) -> None:
    """Renders camera `session` into its bound outputs (`render_batch` of one)."""
    self.render_batch([session])

  def render_batch(self, sessions: Sequence[CameraSession]) -> None:
    """Renders cameras whose poses and shared simulator state have already been updated: their
    alike cameras as one group (`_regroup`), each group once, then the cameras' outputs."""
    state = self._state
    if state is None:
      raise RuntimeError("raywarp's Isaac Lab renderer is not initialized")
    self._regroup(sessions)
    requested: dict[int, tuple[CameraGroup, list[CameraSession]]] = {}
    for session in sessions:
      requested.setdefault(id(session.group), (session.group, []))[1].append(session)
    for group, members in requested.values():
      group.render(state, members)

  def reset(self, session: CameraSession, env_ids: Sequence[int] | wp.array[wp.int32, [int]] | None = None) -> None:
    """Reset only selected environments of a camera, preserving other environments' history."""
    session.reset(env_ids)

  def read_output(self, session: CameraSession, camera_data: CameraData) -> None:
    """Publishes the segmentation id -> label maps (the pixels are already in the bound buffers)."""
    for name, info in session.info().items():
      camera_data.info[name] = info

  def cleanup(self, session: CameraSession | None) -> None:
    """Releases one camera."""
    if session is not None and session in self.sessions:
      self.sessions.remove(session)
      session.close()
      if not session.group.sessions:
        self.groups.remove(session.group)

  def material_writer(
    self,
    material_paths: Sequence[str],
    colors: wp.array[wp.vec3, [int]],
    *,
    before_write: Callable[[], None] | None = None,
    input_names: Sequence[str] | None = None,
  ) -> MaterialWriter:
    """Prepare shared live color storage before cameras (Isaac finalize_consumers order)."""
    layout = self._material_layout
    if layout is None:
      raise RuntimeError("Cameras: the scene is not initialized (initialize(sim) first)")
    prepared = self._material_colors
    if material_paths and prepared is None:
      if self.sessions:
        raise RuntimeError("prepare material writers before creating cameras")
      prepared = PreparedColors.create(self.scene, layout)

    writer = MaterialWriter(
      self.scene,
      layout=layout,
      material_paths=material_paths,
      colors=colors,
      prepared=prepared,
      reset=self._reset_material_worlds,
      before_write=before_write,
      input_names=input_names,
    )
    self._material_colors = prepared
    return writer

  def _reset_material_worlds(self, changed: WorldMask) -> None:
    """Restarts the history of the worlds a material write changed, in every group's renderer
    (they all bind the written table)."""
    for group in self.groups:
      group.reset_worlds(changed)

  def scalar_writer(
    self,
    spec: ScalarChannel,
    material_paths: Sequence[str],
    input_names: Sequence[str],
    values: wp.array[float, [int]],
    *,
    before_write: Callable[[], None] | None = None,
  ) -> ScalarWriter:
    """Prepare one explicitly supported scalar channel before camera creation."""
    layout = self._material_layout
    if layout is None:
      raise RuntimeError("Cameras: initialize the scene before material preparation")
    table = self._material_scalars.get(spec.field)
    if table is None:
      if self.sessions:
        raise RuntimeError("prepare material writers before creating cameras")
      table = scalar_table(self.scene, len(layout.paths), spec)
    writer = ScalarWriter(
      self.scene,
      layout=layout,
      spec=spec,
      material_paths=material_paths,
      input_names=input_names,
      values=values,
      table=table,
      reset=self._reset_material_worlds,
      before_write=before_write,
    )
    self._material_scalars[spec.field] = table
    return writer

  def close(self) -> None:
    """Releases every camera and the scene (idempotent): every one, also after a failed release,
    then the failures raised together."""
    errors: list[Exception] = []
    for session in list(self.sessions):
      try:
        self.cleanup(session)
      except Exception as error:  # noqa: BLE001  # every camera released first
        errors.append(error)
    self.groups.clear()
    self._segments.clear()
    self._material_layout = self._material_colors = None
    self._material_scalars.clear()
    self._scene = self._binding = self._sim = self._state = None
    self._usd_shapes = self._usd_materials = self._visibility = None
    if errors:
      raise ExceptionGroup("closing the Isaac Lab cameras failed", errors)
