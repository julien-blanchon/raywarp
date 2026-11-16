"""Host scene (`raywarp.scene.Scene`) -> GpuScene: `upload` (`rw.upload`).

Field mapping (everything the kernels support today; the rest is counted in the scene's
`LoadReport` under `gpu.*`):

* Instances -> the instance tables, index i = host instance i (`rendered`: the VISIBLE ones);
  shapes, meshes and stretches in `meshes.py`. `segment` -> `segment_id`; prim paths ->
  `instance_names`. The poses (rotations; a mesh instance's carries its uniform scale, s R) are
  the scene's one pose store (`GpuScene.init_pose`) and,
  unless the caller passes its own binding (Newton), the binding (`StaticPoses`: nothing moves
  them, so a Renderer shares every instance it is not told to randomise, see
  `core/pipeline/worlds/sharing.py`).
* Materials -> "material.*", textures and the per-instance shading tables (`materials.py`): the
  device values derived from the OpenPBR inputs (`derive.py`).
* Lights -> "light.*" (`lights.py`): the punctual and the area-light representations; emissive
  panels become rect lights here (`scene.lighting.emissive`, active with
  `lighting.emissive_lights`); their poses are (1, nlight) tables. The environment's suns
  (`SceneOptions.env_sun`) are shaded after them, outside the tables (`GpuEnvironment.suns`).
  CAMERA / AMBIENT rows are the headlight (`GpuScene.headlight`) and the constant ambient
  (`GpuEnvironment.ambient`).
* Environments -> environment maps; cameras -> `cameras` (`environment.py`).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import importlib
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, overload

import numpy as np
import warp as wp

from raywarp._base.types import ShapeType, TexRole, tex_affine, uv_xform
from raywarp.core.pipeline.derive.fields import DeriveInputs
from raywarp.core.pipeline.gpu_scene.binding import Binding, InstancePoses, Slots, StaticPoses, merged
from raywarp.core.pipeline.gpu_scene.gpu_scene import (
  Capabilities,
  GpuEnvironment,
  GpuHeadlight,
  GpuScene,
  LayerTables,
  Meshes,
  MeshInstances,
)
from raywarp.core.pipeline.upload import environment as env_upload
from raywarp.core.pipeline.upload import lights as light_upload
from raywarp.core.pipeline.upload import meshes as mesh_upload
from raywarp.core.pipeline.upload.capabilities import capabilities
from raywarp.core.pipeline.upload.materials import InstanceMaterials, PreparedMaterials, instance_materials, material_tables
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.core.pipeline.upload.reporting import report_mask
from raywarp.core.pipeline.upload.uv_plans import UvPlans, uv_plans
from raywarp.core.shade.lights.ies import IesLights
from raywarp.scene.lighting import emissive
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import Column, InstanceFlags, Rows

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray


@overload
def upload(
  scene: Scene, device: wp.DeviceLike = None, *, binding: None = None, options: SceneOptions = ...
) -> GpuScene[None]: ...


@overload
def upload[S](
  scene: Scene, device: wp.DeviceLike = None, *, binding: Binding[S], options: SceneOptions = ...
) -> GpuScene[S]: ...


def upload[S](
  scene: Scene,
  device: wp.DeviceLike = None,
  *,
  binding: Binding[S] | None = None,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
) -> GpuScene[S] | GpuScene[None]:
  """GpuScene of a host scene (see the module docstring), on `device` (default: the current one).

  Field mapping and approximations: the module docstring (counted in `scene.model.report` under
  `gpu.*`).

  Args:
    scene: the host scene.
    device: Warp device.
    binding: what poses the instances (default: `StaticPoses`, the host scene's poses; the
      Newton adapter passes its `NewtonBinding`).
    options: texture storage, emissive panel lights and the environment sun (`SceneOptions`).
  """
  report = scene.report
  dev = wp.get_device(device)
  _import_renderer()
  if options.fit_emissive:
    with report.timer("gpu.emissive_lights"):
      scene = emissive.add_panel_lights(scene)
  with report.timer("gpu.upload"):
    if binding is None:
      return _upload(scene, report, dev, options=options, binding_of=StaticPoses, scaled=StaticPoses.scaled_poses)
    bound = binding
    return _upload(scene, report, dev, options=options, binding_of=lambda _poses: bound, scaled=bound.scaled_poses)


# The renderer's module: imported on first use (`raywarp.__getattr__`), here on a thread while the
# scene uploads (its Warp kernel definitions are most of the package's import time).
_RENDERER: Final = "raywarp.core.pipeline.renderer"


def _import_renderer() -> None:
  """Starts importing the renderer on a thread unless it is imported (an import error is raised
  again where the Renderer is used)."""
  if _RENDERER not in sys.modules:
    threading.Thread(target=importlib.import_module, args=(_RENDERER,), name="raywarp-import").start()


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Shapes:
  """The instances' shapes on the GPU side (`_shapes`).

  Attributes:
    poses: the poses (the one pose store; s R for a scaled mesh instance).
    types: (n,) core ShapeType.
    sizes: (n, 3) core sizes.
    mesh: (n,) core mesh id (-1: none).
    radius: (n,) bound radius.
    meshes: the core meshes.
    instances: how each instance reads its mesh.
    plan: the host-side plan of `instances`.
    uploaded: the meshes' (host, device) tables (`meshes.register_digests`).
  """

  poses: InstancePoses
  types: Column[np.int32]
  sizes: Rows[np.float32, 3]
  mesh: Column[np.int32]
  radius: Column[np.float32]
  meshes: Meshes
  instances: MeshInstances
  plan: mesh_upload.MeshPlan
  uploaded: mesh_upload.Uploaded


def _names(scene: Scene) -> list[str]:
  """Each instance's prim path ("" without one): `GpuScene.instance_names`."""
  inst = scene.instances
  paths = inst.paths.tolist()
  return [paths[p] if 0 <= p < len(paths) else "" for p in np.asarray(inst.prim).tolist()]


def _shapes(scene: Scene, report: LoadReport, plans: UvPlans, *, paths: list[str], device: wp.Device) -> _Shapes:
  """Poses, shape types, sizes and meshes of the instances (`meshes.py`; mesh stretches but their
  uniform scale are baked into meshes before, `bake_stretches`)."""
  inst = scene.instances
  xf = mesh_upload.split_transforms(inst.xform)
  poses = mesh_upload.instance_poses(scene, xf)
  plan = mesh_upload.mesh_plan(scene, plans, report=report)
  sizes = mesh_upload.primitive_sizes(inst.kind, inst.size, xf.stretch, report=report, paths=paths)
  meshes, uploaded = mesh_upload.core_meshes(
    scene,
    plan.used,
    second_set=bool(np.any(plans.uv_set == 1) and np.any(np.asarray(scene.meshes.uv1_num)[plan.used] > 0)),
    report=report,
    device=device,
  )
  is_mesh = plan.core >= 0
  radius = np.where(is_mesh, mesh_upload.mesh_radii(scene, plan, mesh_upload.pose_scales(scene, xf)), sizes.radius)
  types = np.where(is_mesh, int(ShapeType.MESH), sizes.types).astype(np.int32)
  instances = mesh_upload.mesh_instances(scene, plan, plans=plans, report=report, device=device)
  return _Shapes(
    poses=poses,
    types=types,
    sizes=sizes.sizes,
    mesh=plan.core,
    radius=radius.astype(np.float32),
    meshes=meshes,
    instances=instances,
    plan=plan,
    uploaded=uploaded,
  )


def _report_instances(scene: Scene, report: LoadReport, paths: list[str]) -> Column[np.bool_]:
  """(n,) the VISIBLE instances; reports the instance flags the core does not render."""
  flags = np.asarray(scene.instances.flags, np.int64)
  visible = (flags & int(InstanceFlags.VISIBLE)) != 0
  for flag, name in ((InstanceFlags.CAST_SHADOW, "no_cast_shadow"), (InstanceFlags.PRIMARY_VISIBLE, "not_primary_visible")):
    report_mask(report.drop, f"gpu.instance.{name}", visible & ((flags & int(flag)) == 0), example=paths.__getitem__)
  return visible


def _row[T: np.generic, D](
  values: Column[T] | np.ndarray[[int, int], np.dtype[T]] | np.ndarray[[int, int, int], np.dtype[T]],
  dtype: type[D],
  shape: tuple[int, ...] = (),
  *,
  device: wp.Device,
) -> wp.array[D, [int, int]]:
  """The (1, n, ...) table of per-row `values` (one row shared by every world)."""
  return wp.array(np.ascontiguousarray(np.asarray(values).reshape(1, -1, *shape)), dtype=dtype, device=device)


def _tables(
  shapes: _Shapes,
  tables: PreparedMaterials,
  inst: InstanceMaterials,
  *,
  plans: UvPlans,
  device: wp.Device,
) -> Slots:
  """The instance and material slots ("instance.*", "material.*"; core texture ids, -1 none)."""
  row = functools.partial(_row, device=device)
  derived = tables.derived
  m = len(tables.texid)
  return {
    "instance.mesh": row(shapes.mesh, int),
    "instance.material": row(inst.material, int),
    "instance.size": row(shapes.sizes, wp.vec3, (3,)),
    "instance.color": row(inst.rgba, wp.vec4, (4,)),
    "instance.shading": row(inst.shading, int),
    "material.base_color": row(derived.base, wp.vec4, (4,)),
    "material.emission": row(derived.emission, float),
    "material.mirror": row(derived.mirror, float),
    "material.texture": wp.array(
      np.ascontiguousarray(tables.texid.reshape(1, m, tables.texid.shape[1])), dtype=int, device=device
    ),
    "material.tex_affine": wp.array(
      np.ascontiguousarray(tables.tex_affine.reshape(1, m, tables.texid.shape[1], 2, 4)), dtype=tex_affine, device=device
    ),
    "material.tex_variation": row(np.zeros((m, 4), np.float32), wp.vec4, (4,)),
    "material.tex_repeat": row(plans.repeat, wp.vec2, (2,)),
    "material.uv_world": row(plans.world, uv_xform, (2, 3)),
    "material.tex_uniform": row(tables.uniform, bool),
    "material.alpha_cutoff": row(tables.cutoff, float),
    "material.base_diffuse_roughness": (
      row(derived.base_diffuse_roughness, float)
      if np.any(derived.base_diffuse_roughness > 0.0)
      else wp.empty((1, 0), dtype=float, device=device)
    ),
    "material.base_metalness": row(derived.base_metalness, float),
    "material.specular_roughness": row(derived.specular_roughness, float),
    "material.reflectance": row(derived.reflectance, float),
    "material.specular_anisotropy": row(derived.layers.anisotropy, wp.vec2, (2,)),
    "material.fuzz": row(derived.layers.fuzz, wp.vec2, (2,)),
    "material.fuzz_color": row(derived.layers.fuzz_color, wp.vec3, (3,)),
    "material.coat_color": row(derived.layers.coat_color, wp.vec3, (3,)),
    "material.coat_eta": row(derived.layers.coat_ior, float),
  }


def _layer_tables(scene: Scene, tables: PreparedMaterials, shapes: _Shapes, caps: Capabilities) -> LayerTables:
  """The layered lobes' static tables; the meshes' vertex tangents only for anisotropy."""
  layers = tables.derived.layers
  anisotropic = caps.layers.anisotropy or (caps.prepared_materials is not None and caps.prepared_materials.layers.anisotropy)
  return LayerTables(
    coat_darkening=layers.coat_darkening,
    wrap=layers.wrap,
    wrap_tint=layers.wrap_tint,
    detail_uv=tables.detail_uv,
    flags=tables.layer_flags,
    mesh_tangent=mesh_upload.vertex_tangents(scene, shapes.plan) if anisotropic else None,
  )


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _SceneLights:
  """The lights of a host scene on the GPU side, and its environment (`_scene_lights`).

  Attributes:
    environment: the environment maps and the constant ambient.
    headlight: the camera headlight.
    nlight: the lights shaded: the emitters (the tables' columns), then the environment suns.
    slots: the "light.*" tables.
    info: what the lights use (`Capabilities` fields).
    panel_of: (nemitter,) instance of each emitter's panel (-1: none).
    visible: (k, 2) (light, LightShape) of the lights camera rays see
      (`light_upload.visible_lights`).
    is_panel: (ninstance,) the instances some light stands for.
    ies: the lights' IES profiles.
  """

  environment: GpuEnvironment
  headlight: GpuHeadlight
  nlight: int
  slots: dict[str, AnyDeviceArray]
  info: light_upload.LightInfo
  panel_of: Column[np.int32]
  visible: Rows[np.int32, 2]
  is_panel: Column[np.bool_]
  ies: IesLights


def _scene_lights(
  scene: Scene, report: LoadReport, *, ninstance: int, options: SceneOptions, device: wp.Device
) -> _SceneLights:
  """The environment (with its suns, shaded after the emitters), the display lights and the
  light tables of `scene` (`lights.py`, `environment.py`)."""
  environment = env_upload.environment(scene, report, env_sun=options.env_sun, device=device)
  emitters, display = light_upload.split_display(light_upload.derive_lights(scene.lights))
  environment = dataclasses.replace(environment, ambient=display.ambient)
  arrays, info = light_upload.light_arrays(emitters)
  light_upload.report_lights(emitters, report)
  nlight = len(emitters) + len(environment.suns)
  own = np.asarray(emitters.instance)
  panel_of = np.where((own >= 0) & (own < ninstance), own, -1).astype(np.int32)
  is_panel = np.zeros(ninstance, bool)
  is_panel[panel_of[panel_of >= 0]] = True
  slots = {k: wp.array(np.ascontiguousarray(v), dtype=light_upload.DTYPES[k], device=device) for k, v in arrays.items()}
  return _SceneLights(
    environment=environment,
    headlight=display.headlight,
    nlight=nlight,
    slots=slots,
    info=light_upload.with_suns(info, environment.suns),
    panel_of=panel_of,
    visible=light_upload.visible_lights(emitters),
    is_panel=is_panel,
    ies=light_upload.ies_lights(emitters),
  )


def _camera_slots(cams: env_upload.SceneCameras, device: wp.Device) -> dict[str, AnyDeviceArray]:
  """The (1, ncamera) camera pose tables ("camera.pos", "camera.rot")."""
  return {
    "camera.pos": _row(cams.pos, wp.vec3, (3,), device=device),
    "camera.rot": _row(cams.rot, wp.mat33, (3, 3), device=device),
  }


def _optix_context(device: wp.Device) -> None:
  """Creates `device`'s OptiX context (`trace.optix.context.device_context`, ~0.1 s) before the
  upload's threads start, so an OptiX Renderer finds it made: `optixDeviceContextCreate` on a worker
  thread beside the texture creation and allocations of the others crashed the process (3 of ~220
  scene loads, RTX PRO 6000). Nothing on the CPU or where OptiX cannot load
  (`trace.optix.api.unavailable_reason`: the Renderer reports it when OptiX is asked for)."""
  from raywarp.core.trace.optix import api as optix_api  # noqa: PLC0415 (only loaded with CUDA)
  from raywarp.core.trace.optix.context import device_context  # noqa: PLC0415

  if device.is_cuda and optix_api.unavailable_reason() is None:
    device_context(device)


def _material_tables(scene: Scene, report: LoadReport, *, options: SceneOptions, device: wp.Device) -> PreparedMaterials:
  """The material tables and textures (`material_tables`), timed as "gpu.materials" in `report`.

  Runs on its own thread (`_upload_parts`) with its own `report`: the texture copies go on the
  device's stream while the calling thread uploads the rest, and every later use of the textures
  is ordered after them on that stream.
  """
  with report.timer("gpu.materials"):
    return material_tables(scene, report, device=device, storage=options.textures)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Parts:
  """The parts of a scene's upload made side by side (`_upload_parts`).

  Attributes:
    plans: the materials' UV plans.
    shapes: the instances' shapes.
    rendered: (k,) the VISIBLE instances.
    lights: the lights and the environment.
    tables: the material tables and textures.
  """

  plans: UvPlans
  shapes: _Shapes
  rendered: Column[np.int32]
  lights: _SceneLights
  tables: PreparedMaterials


def _upload_parts(scene: Scene, report: LoadReport, *, paths: list[str], options: SceneOptions, device: wp.Device) -> _Parts:
  """The meshes, lights and materials of `scene` (its stretches baked but the mesh instances'
  uniform scales: `bake_stretches`).

  On threads of their own while this one uploads the meshes and lights: the materials and their
  textures (file reads and texel copies), the mesh tables' digests (the OptiX context is made
  first, on this thread: `_optix_context`). Each part is timed on its own
  (`report.timings["gpu.<part>"]`; "gpu.materials" overlaps the others, "gpu.materials_wait" is
  what this thread waited for the threads).
  """
  textured = LoadReport(source=report.source, describe=report.describe)
  with report.timer("gpu.optix_context"):
    _optix_context(device)
  with concurrent.futures.ThreadPoolExecutor(2) as worker:
    pending = worker.submit(_material_tables, scene, textured, options=options, device=device)
    with report.timer("gpu.meshes"):
      plans = uv_plans(scene, report)
      shapes = _shapes(scene, report, plans, paths=paths, device=device)
      rendered = np.flatnonzero(_report_instances(scene, report, paths)).astype(np.int32)
    digests = worker.submit(mesh_upload.register_digests, shapes.uploaded)
    with report.timer("gpu.lights"):
      lights = _scene_lights(scene, report, ninstance=len(scene.instances), options=options, device=device)
    with report.timer("gpu.materials_wait"):
      tables = pending.result()
      digests.result()
  report.merge(textured)
  report.timings["gpu.materials"] = report.timings.get("gpu.materials", 0.0) + textured.timings["gpu.materials"]
  return _Parts(plans=plans, shapes=shapes, rendered=rendered, lights=lights, tables=tables)


def _upload[S](
  scene: Scene,
  report: LoadReport,
  device: wp.Device,
  *,
  options: SceneOptions,
  binding_of: Callable[[InstancePoses], Binding[S]],
  scaled: bool,
) -> GpuScene[S]:
  """The upload of `upload` on `device` with its `options`; `binding_of(poses)` makes the binding of
  the poses, which carry a mesh instance's uniform scale with `scaled` (`Binding.scaled_poses`)."""
  names = _names(scene)
  paths = [name or f"instance {i}" for i, name in enumerate(names)]  # report examples
  scene = mesh_upload.bake_stretches(scene, scaled=scaled, report=report, paths=paths)
  inst = scene.instances
  n, m = len(inst), len(scene.materials)
  parts = _upload_parts(scene, report, paths=paths, options=options, device=device)
  shapes, tables, lit, plans, rendered = parts.shapes, parts.tables, parts.lights, parts.plans, parts.rendered
  inst_mats = instance_materials(scene, tables)
  cams = env_upload.cameras(scene, report)
  caps = capabilities(
    scene,
    tables,
    types=shapes.types,
    plan=shapes.plan,
    plans=plans,
    inst=inst_mats,
    rendered=rendered,
    info=lit.info,
    panel=lit.is_panel,
  )
  cameras = _camera_slots(cams, device)
  slots = merged(_tables(shapes, tables, inst_mats, plans=plans, device=device), {**lit.slots, **cameras})
  pool = tables.pool
  report.count("gpu.textures", len({id(t) for t in pool.textures[1:]}))  # distinct GPU textures
  binding = binding_of(shapes.poses)
  segments = np.asarray(inst.segment, np.int32)[None]
  flex = binding.deformables(device)
  if flex.nprim:
    caps = dataclasses.replace(caps, shape_types=tuple(sorted({*caps.shape_types, int(ShapeType.FLEX)})))
  return GpuScene(
    device=device,
    deformables=flex,
    flex_segment=n,
    ninstance=n,
    nmaterial=m,
    nlight=lit.nlight,
    tables=slots,
    derive_inputs=DeriveInputs.of(scene.materials, tables.derived, tables.texid[:, TexRole.EMISSIVE]),
    instance_type=wp.array(shapes.types, dtype=int, device=device),
    rendered=rendered,
    init_pose=shapes.poses,
    bound_radius=shapes.radius,
    meshes=shapes.meshes,
    mesh_instances=shapes.instances,
    material_uv=mesh_upload.material_uv_tables(scene, shapes.plan, plans, device=device),
    textures=pool.textures,
    texture_array=wp.array(pool.textures, dtype=wp.Texture2D, device=device),
    tex_size=wp.array(np.asarray(pool.sizes, np.float32), dtype=wp.vec2, device=device),
    tex_layout=wp.array(np.asarray(pool.layouts, np.int32), dtype=int, device=device),
    environment=lit.environment,
    headlight=lit.headlight,
    cameras=cams.specs,
    clip=cams.clip,
    capabilities=caps,
    binding=binding,
    color_rule=binding.color_rule,
    segment_id=None if binding.instance_segments else wp.array(segments, dtype=int, device=device),
    light_instance=lit.panel_of,
    light_visible=lit.visible,
    ies=lit.ies,
    instance_shadow_transmit=inst_mats.optics.shadow,
    instance_optics=inst_mats.optics.rows,
    instance_twosided=inst_mats.optics.twosided,
    optics_inputs=inst_mats.inputs if inst_mats.inputs.switchable else None,
    instance_names=names,
    instance_labels=inst.label_of() if np.any(np.asarray(inst.label) >= 0) else [],
    report=report,
    layers=_layer_tables(scene, tables, shapes, caps),
  )
