"""`RaywarpRenderer`: raywarp as an Isaac Lab renderer (`isaaclab.renderers.BaseRenderer`).

Targets the renderer interface at Isaac Lab commit 647e39c5a108e6b217014e4f57ab86efeef7cdaa.
`prepare_capture` inherits the synchronous no-op: observations are delivered in the same call.
Created by Isaac Lab from `RaywarpRendererCfg` (`get_or_create_backend`) like its Newton Warp
renderer: it asks Isaac for the Newton model of the scene (built by Isaac's Newton cloner, with
PhysX or Newton physics) and reads body poses from the scene data provider. The work is
`cameras.Cameras`'; this module only connects it to Isaac Lab's objects.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, final

import numpy as np
import torch
import warp as wp
from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.scene_data import REQUIRES_STAGE_AND_MODEL, SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.utils.version import has_kit
from isaaclab_newton.physics import NewtonBackendCfg
from isaaclab_newton.renderers.segmentation import NewtonSegmentationMapper

from raywarp.core.pipeline.platform.device import host_copy
from raywarp.formats.usd.prim_paths import under
from raywarp.integrations.isaaclab.cameras import Cameras
from raywarp.integrations.isaaclab.scalars import SCALAR_CHANNELS, MaterialWriters, ScalarWriter
from raywarp.integrations.isaaclab.usd_source import UsdSource, clone_assets, kit_lux_dialect, world_homes

if TYPE_CHECKING:
  import newton
  from isaaclab.renderers.base_renderer import VisualMaterialBatch

  from raywarp.integrations.isaaclab.cfg import RaywarpRendererCfg
  from raywarp.integrations.isaaclab.materials import MaterialWriter
  from raywarp.integrations.isaaclab.session import CameraSession
  from raywarp.integrations.isaaclab.types import Buffer, CameraData, CameraRenderSpec, ProxyArray, SegmentInfo, ShapeIds


def consume_torch(device: wp.Device) -> wp.Stream | None:
  """Orders the device's Warp stream after Torch's current stream (Isaac's producer of body
  poses, camera states and material values); returns that Warp stream (None on CPU)."""
  if not device.is_cuda:
    return None
  consumer = wp.get_stream(device)
  producer = torch.cuda.current_stream(torch.device(str(device)))
  if producer.cuda_stream != consumer.cuda_stream:
    consumer.wait_stream(wp.stream_from_torch(producer))
  return consumer


def publish_torch(device: wp.Device) -> None:
  """Orders Torch's current stream (Isaac's consumer of the camera outputs) after the device's Warp
  stream: the rendered images are complete when Isaac's next kernels read them."""
  if not device.is_cuda:
    return
  producer = wp.get_stream(device)
  consumer = torch.cuda.current_stream(torch.device(str(device)))
  if producer.cuda_stream != consumer.cuda_stream:
    consumer.wait_stream(wp.stream_to_torch(producer))


def _warp_view[D](tensor: torch.Tensor, dtype: type[D], what: str) -> wp.array[D, [int]]:
  """`tensor`'s memory as a dense Warp array of `dtype` (no copy).

  Raises:
    TypeError: when Warp cannot view it as one array (`what` names it).
  """
  view = wp.from_torch(tensor, dtype=dtype)
  if not isinstance(view, wp.array):
    raise TypeError(f"{what} must be a Warp array view")
  return view


def _kit_version() -> str | None:
  """The running Kit's version ("107.3.1+..."), None in kit-less Isaac Lab (Newton without Kit)."""
  if not has_kit():
    return None
  import omni.kit.app  # noqa: PLC0415  # Kit only, present whenever has_kit()

  return str(omni.kit.app.get_app().get_kit_version())


@final
class IsaacSimulation:
  """`types.Simulation` over Isaac Lab: its renderer's Newton backend, scene data provider and
  stage."""

  def __init__(self, sim: SimulationContext, newton_cfg: NewtonBackendCfg, cfg: RaywarpRendererCfg) -> None:
    """Acquires the Newton backend built by Isaac's cloner and maps its bodies to the provider."""
    self._sim = sim
    self._cfg = cfg
    self._backend = sim.get_or_create_backend(newton_cfg)
    self._provider = sim.get_scene_data_provider()
    self._mapping = self._provider.create_mapping(list(self._backend.model.body_label))
    self._segmentation: NewtonSegmentationMapper | None = None
    self._warned = False
    # One output for every frame: a converted (not passed-through) layout reuses its storage.
    self._poses = SceneDataFormat.Transform()

  @property
  def model(self) -> newton.Model:
    """The Newton model of every environment."""
    return self._backend.model

  def usd_source(self) -> UsdSource | None:
    """One prototype stage view addressed through Isaac's explicit clone plan."""
    from isaaclab.cloner.clone_plan import path as clone_path  # noqa: PLC0415

    stage = self._sim.stage
    if stage is None:
      return None
    lux_dialect = kit_lux_dialect(_kit_version())
    plan = self._sim.get_clone_plan()
    if plan is None:
      return UsdSource.create(stage, self.model, ["/"], lux_dialect=lux_dialect)
    layout = np.asarray(plan.topology.world_prototype_layout, np.int32)
    roots = tuple(plan.env_template.format(world) for world in range(len(layout)))
    sources = clone_path.get_asset_prototype_paths(plan)
    templates, starts = clone_path.get_world_prototype_asset_templates(plan)
    homes = world_homes(layout)
    assets = clone_assets(layout, starts, np.asarray(plan.topology.world_prototypes), sources, templates, homes)
    targets = [target for _, target in (*assets.shared, *(a for pairs in assets.local.values() for a in pairs))]
    excluded = [source for source in sources if source is not None and not any(under(target, source) for target in targets)]
    origins = None if plan.positions is None else np.asarray(plan.positions, np.float32)
    return UsdSource.create(
      stage,
      self.model,
      roots,
      homes=homes,
      assets=assets,
      origins=origins,
      excluded_sources=excluded,
      lux_dialect=lux_dialect,
    )

  def body_poses(self) -> wp.array[wp.transformf, [int]] | None:
    """The bodies' world poses published by the physics (borrowed when already in Newton's
    layout; else converted into this simulation's one output), ready on the device's Warp stream:
    it waits for Torch's work, the physics' and the provider's conversion, after the conversion."""
    poses = self._poses
    published = self._provider.get_transforms(poses, mapping=self._mapping, count=self.model.body_count)
    consume_torch(wp.get_device(self.model.device))
    return poses.transforms if published else None

  def segmentation(self, kind: str) -> tuple[ShapeIds, SegmentInfo]:
    """Isaac's own per-shape ids and info of `kind` (its Newton Warp mapper: nearest labelled
    ancestor groups, `semantic_filter`, prototype fallback for replicated envs; ids only)."""
    if self._segmentation is None:
      plan = self._sim.get_clone_plan()
      self._segmentation = NewtonSegmentationMapper(self.model, self._sim.stage, self._cfg, plan)
    # Isaac's API takes its `force_rebuild` flag positionally.
    self._segmentation.build_mapping(kind, False)  # noqa: FBT003
    mapping = self._segmentation.get_mapping(kind, False)  # noqa: FBT003
    return host_copy(mapping.shape_to_id).astype(np.int32), mapping.info

  def shape_materials(self) -> Sequence[str]:
    """Each shape's material path (Isaac's `isaaclab:visual_material_path` shape attribute)."""
    # Newton exposes custom attribute namespaces only as instance attributes (none without it).
    namespace = vars(self.model).get("isaaclab")
    if namespace is None:
      if not self._warned:
        warnings.warn(
          "raywarp: the Newton model has no isaaclab:visual_material_path attribute: visual-material writes "
          "do not reach the render",
          stacklevel=2,
        )
        self._warned = True
      return []
    return list(namespace.visual_material_path)


@final
class RaywarpRenderer(BaseRenderer):
  """raywarp's physically based renderer behind Isaac Lab's camera sensors (see the module doc).

  Every Isaac Lab call maps to `cameras.Cameras`: `create_render_data` -> a `CameraSession` (a
  camera of the Renderer of the cameras rendering alike, a world per environment), `set_outputs`
  -> output binding (zero copy for a camera alone), `render_batch` -> one frame per such Renderer,
  `update_transforms` -> body poses from the scene data provider, `update_camera` /
  `update_camera_intrinsics` -> per-world camera pose / intrinsics kernels, `render` -> a batch
  of one (frames are CUDA graph replays on GPUs), `read_output` -> segmentation labels,
  `visual_material_writer` -> prepared per-world material tables.
  """

  def __init__(self, cfg: RaywarpRendererCfg, *, sim: SimulationContext) -> None:
    """Pre-physics: asks Isaac Lab for the Newton model its cloner builds (PhysX or Newton
    physics)."""
    super().__init__(cfg, sim=sim)
    self.cfg: RaywarpRendererCfg = cfg
    self.cameras = Cameras(cfg)
    self._newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
    _, requires_model = REQUIRES_STAGE_AND_MODEL["newton_warp"]
    sim.requires_newton_model |= requires_model
    sim.requires_usd_stage = True  # segmentation / material provenance and stage ingestion read the stage

  def initialize(self) -> None:
    """Post-physics: prepare the stage's visuals and their Newton body bindings."""
    self.cameras.initialize(IsaacSimulation(self._sim, self._newton_cfg, self.cfg))

  @property
  def visual_material_writer(self) -> Callable[[tuple[VisualMaterialBatch, ...]], MaterialWriters]:
    """The factory of writers for Isaac's composed material channels."""
    return self._material_writer

  def _material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> MaterialWriters:
    """Prepare supported authored channels; report remaining inputs without approximating them."""
    channels: list[MaterialWriter | ScalarWriter] = []
    for batch in batches:
      wait = self._material_producer(batch)
      if batch.channel == "color":
        colors = _warp_view(batch.values, wp.vec3, "material color buffer")
        channels.append(
          self.cameras.material_writer(batch.material_paths, colors, before_write=wait, input_names=batch.input_names)
        )
      elif (spec := SCALAR_CHANNELS.get(batch.channel)) is not None:
        values = _warp_view(batch.values, wp.float32, "scalar material buffer")
        channels.append(self.cameras.scalar_writer(spec, batch.material_paths, batch.input_names, values, before_write=wait))
      else:
        self.cameras.scene.report.drop("isaac.material.channel(unsupported)", batch.channel)
    return MaterialWriters(channels=tuple(channels))

  def _material_producer(self, batch: VisualMaterialBatch):
    """Retain each composed Torch input and order its producer before current Warp work."""
    device = wp.get_device(self.cameras.scene.device)

    def wait() -> None:
      """Marks the composed Torch input as used on the consuming Warp stream (Torch keeps its
      memory until that stream has run past it)."""
      consumer = consume_torch(device)
      if consumer is not None:
        batch.values.record_stream(wp.stream_to_torch(consumer))

    return wait

  def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
    """The outputs written (`contract.OUTPUTS`, through the config)."""
    return self.cfg.supported_output_types()

  def prepare_stage(self, stage: object, num_envs: int) -> None:  # BaseRenderer's signature
    """Nothing to export: the scene comes from the Newton model."""

  def create_render_data(self, spec: CameraRenderSpec) -> CameraSession:
    """A session rendering the camera in every environment."""
    return self.cameras.create_render_data(spec)

  def set_outputs(self, render_data: CameraSession, output_data: dict[str, ProxyArray[Buffer]]) -> None:
    """Renders into Isaac's output buffers (zero copy where the layouts match)."""
    self.cameras.set_outputs(render_data, output_data)

  def update_transforms(self) -> None:
    """Body poses of this frame from the scene data provider (`IsaacSimulation.body_poses` orders
    the copy after Torch's work producing them)."""
    self.cameras.update_transforms()

  def update_geometries(self) -> None:
    """Nothing: deformable geometry is not rendered by this adapter."""

  def update_camera(
    self,
    render_data: CameraSession,
    positions: ProxyArray[wp.array[wp.vec3, [int]]],
    orientations: ProxyArray[wp.array[wp.quatf, [int]]],
    intrinsics: ProxyArray[wp.array[wp.mat33, [int]]],
  ) -> None:
    """Camera poses (and first intrinsics) of every environment (after Torch's producers)."""
    consume_torch(render_data.renderer.device)
    self.cameras.update_camera(render_data, positions, orientations, intrinsics)

  def update_camera_intrinsics(
    self,
    render_data: CameraSession,
    intrinsics: wp.array[wp.mat33, [int]],
    parameters: wp.array[float, [int, int]],  # noqa: ARG002  # BaseRenderer's signature
  ) -> None:
    """Per-environment intrinsics (the physical `parameters` are not needed: the matrices are)."""
    consume_torch(render_data.renderer.device)
    self.cameras.update_camera_intrinsics(render_data, intrinsics)

  def render(self, render_data: CameraSession) -> None:
    """Renders one camera into its outputs (Torch's later work waits for them)."""
    self.cameras.render(render_data)
    publish_torch(render_data.renderer.device)

  def render_batch(self, render_data: Sequence[CameraSession]) -> None:
    """Submit prepared cameras (empty sequences are a no-op); Torch's later work waits for them."""
    self.cameras.render_batch(render_data)
    for alias in {str(session.renderer.device) for session in render_data}:
      publish_torch(wp.get_device(alias))

  def reset(self, render_data: CameraSession, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Discard selected histories; Isaac's CUDA tensor ids stay on their device (CPU tensor ids
    take the host path, as a list)."""
    if isinstance(env_ids, torch.Tensor) and not env_ids.is_cuda:
      env_ids = [int(i) for i in env_ids.reshape(-1).tolist()]
    if isinstance(env_ids, torch.Tensor):
      stream = None
      consumer = consume_torch(wp.get_device(str(env_ids.device)))
      if consumer is not None:
        stream = wp.stream_to_torch(consumer)
        env_ids.record_stream(stream)
      # Allocate/convert on the consumer stream: temporary tensor storage remains valid
      # until the reset kernel completes, even when Isaac and Warp use different streams.
      with torch.cuda.stream(stream) if stream is not None else nullcontext():
        ids = _warp_view(env_ids.to(dtype=torch.int32).contiguous(), wp.int32, "Isaac reset ids")
        self.cameras.reset(render_data, ids)
    else:
      self.cameras.reset(render_data, env_ids)

  def read_output(self, render_data: CameraSession, camera_data: CameraData) -> None:
    """Publishes the segmentation labels (pixels are already in the camera's buffers)."""
    self.cameras.read_output(render_data, camera_data)

  def cleanup(self, render_data: CameraSession | None) -> None:
    """Releases one camera."""
    self.cameras.cleanup(render_data)

  def close(self) -> None:
    """Releases every camera and the scene (idempotent)."""
    self.cameras.close()
