"""The records a Renderer hands its trace backend and kernel-struct builders, so that they read
typed fields instead of the Renderer's private attributes (`r._*`).

* `SoftwareScene`: the software traversal's structures (scene BVH, mesh BLAS ids, mesh trace
  frames), owned by the Renderer and (re)built by the backends (`TraceBackend.configure`,
  `software_bvh`, `software_meshes`).
* `RendererTables`: the Renderer's own kernel tables and placeholders (built at
  construction; the shadow transmittance per configuration).
* `FrameContext`: everything a frame's backend work and struct builders read besides the
  `Frame` (`Renderer.context()`, rebuilt per call: references only).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, final

import numpy as np
import warp as wp

from raywarp._base.types import OPTICS_ROWS, UV_IDENTITY, uv_xform
from raywarp.core.kernels.abi import counters
from raywarp.core.kernels.abi.output_layout import OUT_SEG_SEMANTIC
from raywarp.core.kernels.abi.rays import ray_floors, scene_scale
from raywarp.core.outputs.render_outputs import OutputTables
from raywarp.core.outputs.semantic_classes import UNLABELLED, Semantics
from raywarp.core.pipeline.gpu_scene.gpu_scene import shadow_transmit
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.worlds.sharing import instance_names
from raywarp.core.post.grading.transform import Grading
from raywarp.core.shade import gi as gi_lib
from raywarp.core.shade.environment import ibl
from raywarp.core.shade.lights.pack import LightPack, sun_discs
from raywarp.core.shade.lights.punctual import falloff_floor
from raywarp.core.shade.materials.material import LayerConstants
from raywarp.scene.model.types import LightId

if TYPE_CHECKING:
  from raywarp._base.dims import AnyHostArray
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.kernels.abi.structs import Cameras, Params
  from raywarp.core.outputs.render_outputs import Buffers
  from raywarp.core.pipeline.cameras import CameraRig
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.post.temporal.history import History
  from raywarp.core.shade.lights.grid import Grid
  from raywarp.core.trace.accel.bvh import SceneBvh
  from raywarp.core.trace.rays.primary import CameraModel
  from raywarp.settings import RenderSettings


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class SoftwareScene:
  """The software traversal's structures of one Renderer.

  Attributes:
    bvh: the scene BVH: the software backend's (every world), else the software ray kernels' (GI
      probes, light grid: world 0's, every world's with GI), built on first use; None before.
    mesh_bvh_id: (nmesh,) the mesh BLAS ids the software queries read (zeros until built).
    mesh_frame: (nmesh,) rotations of meshes whose BLAS is built in a better-aligned frame
      (`Features.mesh_frames`); `rot_trace` (nworld, ncolumn) the mesh instances' frames of the
      ray queries (instance rotation x frame^T) at their `mesh_columns`, None without mesh frames;
      the first `mesh_per_world` of those columns are per world, the shared ones after them are
      written in world 0's row only.
  """

  bvh: SceneBvh | None
  mesh_bvh_id: wp.array[wp.uint64, [int]]
  mesh_frame: wp.array[wp.mat33, [int]] | None = None
  rot_trace: wp.array[wp.mat33, [int, int]] | None = None
  mesh_columns: wp.array[int, [int]] | None = None
  mesh_per_world: int = 0


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class RendererTables:
  """A Renderer's own kernel tables (built at construction; the Renderer replaces some: shadow
  transmittance, segment table, world skip, launch worlds).

  Attributes:
    shadow_transmit: (max(ninstance, 1),) RGB shadow transmittance under `shadows.translucent`
      (rebuilt by each configuration; per world, world-major: `world_optics.WorldOptics`).
    optics: (max(ninstance, 1), `types.OPTICS_ROWS`) the instances' optics rows
      (`GpuScene.instance_optics`; per world as `shadow_transmit`).
    twosided: (max(ninstance, 1),) instances back-face culling skips, closed panes (per world as
      `shadow_transmit`).
    planes: (nplane,) the per-world analytic planes of the software traversal
      (`sharing.Layout.planes`; empty: none).
    pose_slot: (max(ninstance, 1),) per-world pose column (>= 0) or -1 - shared slot.
    visible_slot: (max(ninstance, 1),) column in the per-world visibility table (-1: always shown).
    segment_id: (1 or nworld, ninstance) segment id per (world, instance) (without a table: the
      instance index; `Renderer.set_segment_table`).
    semantics: the instances' semantic class ids (the semantic output).
    emitter_light: (max(ninstance, 1),) light id standing for each instance's emission (-1: none).
    all_lights: (start, ids) of the one-cell placeholder light grid (every light).
    dfg: the split-sum DFG texture.
    no_hfields: the empty height-field BLAS table of scenes without height fields.
    ray_floor: (ninstance + 1,) the secondary rays' origin offset floor per instance, then the
      scene's (`kernels.abi.rays.ray_floors`).
    falloff_floor: the lights' squared falloff distance floor (`punctual.falloff_floor`).
    no_gi: the GI volume placeholder (lookups compiled out).
    no_view: the empty metered-exposure table (`Environment.view_exposure`) without automatic
      exposure.
    world_skip: (nworld,) or (1,) skipped worlds (the worlds `Renderer.render(active=...)` does
      not list, `active.ActiveWorlds.skip`).
    launch_worlds: (k,) the render launch's worlds (`active.ActiveWorlds.units`; empty: every
      world, one unit each).
    light_pack: the packed lights the light structs read (`shade/lights/pack.py`; packed by
      the Renderer at the start of each frame).
    sun_discs: per environment map, its sun's disc as the background draws it
      (`light_pack.sun_discs`).
    counters: (nworld, NCOUNTER) light-cost counters (`kernels/abi/counters.py`; written by
      Features.counters builds only).
    mesh_used: the meshes with their own BLAS / GAS in this Renderer's structures
      (`Meshes.used_now` at construction; None: every mesh); its validation checks the same mask.
    grading: the display transform's per-world look rows (`post/grading/transform.py`).
    diffuse_source, no_diffuse_source: the materials' (base weight, eligible substrate) rows of
      the EON diffuse lobe (`prepare_diffuse`; None without it) and their one-row placeholder.
    specular_models, no_models: the materials' authored specular model bits of weighted shading
      (`prepare_specular`; None without it) and their one-row placeholder.
  """

  shadow_transmit: wp.array[wp.vec3, [int]]
  optics: wp.array[wp.vec4, [int, int]]
  twosided: wp.array[int, [int]]
  planes: wp.array[int, [int]]
  pose_slot: wp.array[int, [int]]
  visible_slot: wp.array[int, [int]]
  segment_id: wp.array[int, [int, int]]
  semantics: Semantics
  emitter_light: wp.array[int, [int]]
  all_lights: tuple[wp.array[int, [int]], wp.array[int, [int]]]
  dfg: wp.Texture2D
  no_hfields: wp.array[wp.uint64, [int]]
  ray_floor: wp.array[float, [int]]
  falloff_floor: float
  no_gi: gi_lib.Volume
  no_view: wp.array[float, [int]]
  world_skip: wp.array[bool, [int]]
  launch_worlds: wp.array[int, [int]]
  light_pack: LightPack
  sun_discs: tuple[wp.array[wp.vec4, [int]], wp.array[wp.vec3, [int]]]
  counters: wp.array[wp.int64, [int, int]]
  no_diffuse_source: wp.array[wp.vec2, [int]]
  no_models: wp.array[int, [int]]
  mesh_used: np.ndarray[[int], np.dtype[np.bool_]] | None
  grading: Grading
  diffuse_source: wp.array[wp.vec2, [int]] | None = None
  specular_models: wp.array[int, [int]] | None = None
  # The layered carrier's static arrays (`prepare_layers`): per-material constants, the meshes'
  # vertex tangents, the cloth albedo table.
  layers: tuple[wp.array[LayerConstants, [int]], wp.array[wp.vec4, [int]], wp.Texture2D] | None = None

  def prepare_layers[S](self, scene: GpuScene[S], *, enabled: bool) -> None:
    """Uploads the layered lobes' static tables once while `enabled` (Features.layered of the
    build or its relight carrier); drops them otherwise."""
    if not enabled:
      self.layers = None
      return
    if self.layers is not None:
      return
    device = scene.device
    m = max(scene.nmaterial, 1)
    rows = [LayerConstants() for _ in range(m)]
    host = scene.layers
    for i, row in enumerate(rows):
      row.detail_uv = uv_xform(*UV_IDENTITY.reshape(-1))
      if host is not None and i < len(host.flags):
        row.coat_darkening = float(host.coat_darkening[i])
        row.wrap = float(host.wrap[i])
        row.wrap_tint = wp.vec3(*host.wrap_tint[i].tolist())
        row.detail_uv = uv_xform(*host.detail_uv[i].reshape(-1).tolist())
        row.flags = int(host.flags[i])
    tangent = host.mesh_tangent if host is not None and host.mesh_tangent is not None else np.zeros((0, 4), np.float32)
    self.layers = (
      wp.array(rows, dtype=LayerConstants, device=device),
      wp.array(tangent, dtype=wp.vec4, device=device),
      ibl.cloth_texture(device),
    )

  def prepare_specular[S](self, scene: GpuScene[S], *, enabled: bool) -> None:
    """Uploads the materials' authored specular model bits (flex materials included) once while
    `enabled` (Features.specular_weight: weighted shading); drops them otherwise."""
    if enabled:
      if self.specular_models is None:
        self.specular_models = wp.array(scene.derive_inputs.shading, dtype=int, device=scene.device)
    else:
      self.specular_models = None

  def prepare_diffuse[S](self, scene: GpuScene[S], *, enabled: bool) -> None:
    """Uploads the materials' (base weight, eligible substrate) rows once while `enabled`
    (Features.diffuse_roughness: the EON lobe); drops them otherwise."""
    if enabled:
      if self.diffuse_source is None:
        self.diffuse_source = wp.array(scene.derive_inputs.diffuse_source, dtype=wp.vec2, device=scene.device)
    else:
      self.diffuse_source = None

  @staticmethod
  def transmittance[S](scene: GpuScene[S], translucent: Literal["off", "transmittance", "clear"]) -> wp.array[wp.vec3, [int]]:
    """The instances' shadow transmittance under `shadows.translucent` ("off": 0, "clear": 1
    where any), one row on the scene's device."""
    transmit = shadow_transmit(scene)
    if translucent == "off":
      # Opaque casters (a build tracing translucent casters reads them per this table).
      transmit = np.zeros_like(transmit)
    elif translucent == "clear":
      transmit = np.repeat(np.any(transmit > 0.0, axis=1, keepdims=True), 3, axis=1).astype(np.float32)
    return wp.array(transmit, dtype=wp.vec3, device=scene.device)

  @staticmethod
  def build[S](scene: GpuScene[S], layout: Layout, nworld: int) -> RendererTables:
    """The tables of `scene` under `layout` for `nworld` worlds (all rendered), all on the scene's
    device."""
    device = scene.device
    n = max(scene.ninstance, 1)
    types = host_copy(scene.instance_type)[: len(scene.bound_radius)]
    twosided = scene.instance_twosided
    emitter = np.full(n, -1, np.int32)
    for light, inst in enumerate(scene.light_instance):
      if 0 <= inst < n:
        emitter[inst] = light
    nlight = scene.nlight
    return RendererTables(
      shadow_transmit=wp.zeros(n, dtype=wp.vec3, device=device),
      optics=wp.array(_optics_rows(scene.instance_optics, n), dtype=wp.vec4, device=device),
      twosided=_ints(device, twosided if len(twosided) else [0]),
      planes=_ints(device, layout.planes),
      pose_slot=_ints(device, layout.pose_slot if len(layout.pose_slot) else [0]),
      visible_slot=_ints(device, layout.visible_slot if len(layout.visible_slot) else [-1]),
      # Without a segment table, the instance index (what a build compiling no table writes).
      segment_id=scene.segment_id
      if scene.segment_id is not None
      else wp.array(np.arange(max(scene.ninstance, 1)).astype(np.int32)[None], dtype=int, device=device),
      semantics=Semantics(instance_names(scene), scene.instance_labels, device=device),
      emitter_light=_ints(device, emitter),
      all_lights=(_ints(device, [0, nlight]), _ints(device, np.arange(max(nlight, 1)))),
      dfg=ibl.dfg_texture(device),
      no_diffuse_source=wp.zeros(1, dtype=wp.vec2, device=device),
      no_models=wp.zeros(1, dtype=int, device=device),
      no_hfields=wp.zeros(0, dtype=wp.uint64, device=device),
      ray_floor=wp.array(ray_floors(scene.bound_radius, scene.init_pose.pos, types), dtype=float, device=device),
      falloff_floor=falloff_floor(scene_scale(scene.bound_radius, scene.init_pose.pos, types)),
      no_gi=gi_lib.empty_volume(device),
      no_view=wp.zeros(0, dtype=float, device=device),
      world_skip=wp.zeros(1, dtype=bool, device=device),
      launch_worlds=wp.zeros(0, dtype=int, device=device),
      light_pack=LightPack(nlight, scene.environment.suns, ies=scene.ies, visible=scene.light_visible, device=device),
      sun_discs=sun_discs(scene.environment.suns, scene.environment.maps.scale.shape[0], device),
      counters=wp.zeros((nworld, counters.NCOUNTER), dtype=wp.int64, device=device),
      mesh_used=scene.meshes.used_now(),
      grading=Grading(device, nworld),
    )


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class FrameContext[S]:
  """A Renderer's state as its backend and the struct builders read it (see the module doc).

  Attributes:
    scene, layout: the GpuScene and the shared-scenery layout.
    settings, features, params, effective: the settings and their resolution
      (`resolve.Resolution`: `features` what the kernels compile, `effective` what the settings
      resolve to, which the host's decisions read); `relight` the GI relight kernel's build
      (`resolve.relight_build`; None without GI); `camera_model` the cameras' model (the program
      the render launches, `resolve.camera_model`).
    nworld: worlds; `cameras` the camera rig (resolution, poses, bundles).
    buffers: the output buffers; `history` the temporal state (None without one).
    software: the software traversal's structures (mutable, shared with the Renderer).
    tables: the Renderer's own tables.
    light_grid: the light visibility grid (None: off or not built).
    gi: the GI probe volume (None before its bake).
    background: linear radiance of rays that hit nothing without a background map.
    per_world_fields: the declared per-world appearance fields.
    shadow_light: the shadow output's light index (`Renderer` resolves `ShadowOutput.light`).
    generation: (renderer, baked structures) counters that change whenever the kernel structs
      must be rebuilt (struct caches compare it).
    dynamic_lights: the lights whose footprint may change (`PerWorld.dynamic_lights`: they bypass
      the light grid).
    view_exposure: camera 0's metered exposure per world (`exposure.AutoExposure.view`; None: no
      metering).
    reference_exposure: (1,) the renderer's exposure (`Exposure.render_ev100()`, written in place
      when it changes): the exposure the GI probes light display-relative emission with
      (`frame_structs.probe_environment`), so a per-world exposure re-lights no probe.
    register_caps: the build's OptiX register caps (shaded, light: `registers.register_caps`).
    emission_varies: the lights' emission is written after the bakes (`bake_keys.emission_varies`,
      fixed for a renderer).
  """

  scene: GpuScene[S]
  layout: Layout
  settings: RenderSettings
  features: Features
  params: Params
  effective: Features
  relight: Features | None
  camera_model: CameraModel
  nworld: int
  cameras: CameraRig
  buffers: Buffers
  history: History | None
  software: SoftwareScene
  tables: RendererTables
  light_grid: Grid | None
  gi: gi_lib.ProbeVolume | None
  background: wp.vec3
  per_world_fields: frozenset[str]
  generation: tuple[int, int]
  shadow_light: LightId
  register_caps: tuple[int, int]
  reference_exposure: wp.array[float, [int]]
  emission_varies: bool
  dynamic_lights: tuple[int, ...] = ()
  view_exposure: wp.array[float, [int]] | None = None

  def camera_struct(self, frame: Frame) -> Cameras:
    """The cameras' kernel struct for `frame` and the frame's launch worlds (`CameraRig.struct`)."""
    return self.cameras.struct(frame, self.tables.launch_worlds)

  @property
  def nunit(self) -> int:
    """Units of the render launch's first dimension: the active list's length, else nworld."""
    return self.tables.launch_worlds.shape[0] or self.nworld

  def output_tables(self) -> OutputTables:
    """The tables of the output struct: the "seg" buffer's per-instance ids and deformable 0's id
    (segment ids, or semantic class ids, every deformable unlabelled, when they are written
    alone), the semantic buffer's, the skipped worlds and the shadow output's light."""
    semantic = self.tables.semantics.ids
    alone = self.buffers.layout.bits & OUT_SEG_SEMANTIC
    seg, flex = (semantic, UNLABELLED) if alone else (self.tables.segment_id, self.scene.flex_segment)
    return OutputTables(
      segment_id=seg,
      flex_segment=flex,
      world_skip=self.tables.world_skip,
      semantic_id=semantic,
      semantic_flex=UNLABELLED,
      shadow_light=self.shadow_light,
    )

  @property
  def nray(self) -> int:
    """Camera rays per world (every camera's pixels)."""
    return self.cameras.nray


def _ints(device: wp.DeviceLike, values: AnyHostArray | list[int]) -> wp.array[int, [int]]:
  """An int32 table of host `values` on `device`."""
  return wp.array(np.asarray(values, np.int32), dtype=int, device=device)


def _optics_rows(
  optics: np.ndarray[[int, int, int], np.dtype[np.float32]], n: int
) -> np.ndarray[[int, int, int], np.dtype[np.float32]]:
  """(n, OPTICS_ROWS, 4) float32 optics rows of the instances (zero rows past them: an empty scene
  holds one)."""
  out = np.zeros((n, OPTICS_ROWS, 4), np.float32)
  out[: len(optics)] = optics
  return out
