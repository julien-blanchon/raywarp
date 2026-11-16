"""What a GpuScene's content uses (`Capabilities`), from its upload's parts (`gpu_scene.upload`):
the Features a Renderer compiles for the rendered instances, and those a material edit can turn
on (`PreparedMaterialCapabilities`)."""

from __future__ import annotations

import numpy as np

from raywarp._base.types import LayerFlag, ShadingFlag, TexRole
from raywarp.core.pipeline.gpu_scene.gpu_scene import Capabilities, LayerUse, PreparedMaterialCapabilities, TextureSlots
from raywarp.core.pipeline.upload.lights import LightInfo
from raywarp.core.pipeline.upload.materials import InstanceMaterials, PreparedMaterials
from raywarp.core.pipeline.upload.meshes import MeshPlan
from raywarp.core.pipeline.upload.uv_plans import UvPlans
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import Column, MaterialFlags


def capabilities(  # noqa: PLR0913  # the upload's parts, keyword-only
  scene: Scene,
  tables: PreparedMaterials,
  *,
  types: Column[np.int32],
  plan: MeshPlan,
  plans: UvPlans,
  inst: InstanceMaterials,
  rendered: Column[np.int32],
  info: LightInfo,
  panel: Column[np.bool_],
) -> Capabilities:
  """What the `rendered` instances, their materials and the lights use (`Capabilities`).

  Args:
    scene: the host scene (stretches baked).
    tables: its material tables.
    types: (n,) each instance's core ShapeType.
    plan: how the instances read their meshes.
    plans: the materials' UV plans.
    inst: the instances' material tables.
    rendered: (k,) the rendered instances.
    info: what the lights use.
    panel: (n,) the instances some light stands for.
  """
  matid = inst.material[rendered]
  rendered_mats = np.unique(matid[matid >= 0]).astype(np.int64)
  colored_instances = rendered[plan.vertex_colors[rendered]]
  colored = bool(len(colored_instances))
  vertex_alpha = colored and _translucent_colors(scene, plan, colored_instances)
  alpha = inst.alpha[rendered]
  return Capabilities(
    shape_types=tuple(sorted({int(t) for t in types[rendered]})),
    textures=_texture_slots(tables, rendered_mats),
    prepared_materials=_prepared_capabilities(scene, tables, plan=plan, plans=plans, rendered=rendered),
    has_alpha=bool(np.any((alpha > 0.0) & (alpha < 1.0))) or vertex_alpha,
    alpha_map=bool(np.any(tables.alpha_map[rendered_mats])),
    has_spot=info.has_spot,
    has_area_lights=info.has_area_lights,
    visible_lights=info.visible_lights,
    uv_maps=bool(np.any(plan.own_uvs[rendered])),
    world_uv=bool(np.any(plans.world[rendered_mats])),
    physical_emission=True,
    has_mirror=bool(np.any(tables.derived.mirror[rendered_mats] > 0.0)),
    specular_curve=bool(np.any(inst.curve[rendered])),
    transmissive=bool(np.any(inst.transmissive[rendered])),
    diffuse_roughness=bool(np.any(tables.derived.base_diffuse_roughness[rendered_mats] > 0.0)),
    clearcoat=bool(np.any(inst.coat[rendered] > 0.0)),
    sun_disc=info.sun_disc,
    has_ies=info.has_ies,
    light_panels=bool(np.any(panel[rendered])),
    alpha_cutout=bool(np.any(tables.cutoff[rendered_mats] > 0.0)),
    vertex_colors=colored,
    unlit_emission=bool(np.any(inst.shading[rendered] & int(ShadingFlag.UNLIT | ShadingFlag.DISPLAY_EMISSION))),
    layers=_layer_use(tables, rendered_mats),
  )


def _prepared_capabilities(
  scene: Scene, tables: PreparedMaterials, *, plan: MeshPlan, plans: UvPlans, rendered: Column[np.int32]
) -> PreparedMaterialCapabilities:
  """What every material of the scene uses (`PreparedMaterialCapabilities`: what a material edit
  can turn on for the rendered instances)."""
  every_mat = np.arange(len(scene.materials)).astype(np.int64)
  derived = tables.derived
  colored, vertex_alpha = _prepared_vertex_colors(scene, plan, rendered)
  return PreparedMaterialCapabilities(
    textures=_texture_slots(tables, every_mat),
    world_uv=bool(np.any(plans.world)),
    alpha_cutout=bool(np.any(tables.cutoff > 0.0)),
    alpha_map=bool(np.any(tables.alpha_map)),
    has_alpha=bool(np.any((derived.base[:, 3] > 0.0) & (derived.base[:, 3] < 1.0))) or vertex_alpha,
    has_mirror=bool(np.any(derived.mirror > 0.0)),
    specular_curve=bool(np.any(derived.curve)),
    clearcoat=bool(np.any(derived.coat > 0.0)),
    transmissive=bool(np.any(derived.transmissive)),
    vertex_colors=colored,
    layers=_layer_use(tables, every_mat),
    unlit_emission=bool(
      np.any(np.asarray(scene.materials.ext_flags) & int(MaterialFlags.UNLIT | MaterialFlags.DISPLAY_EMISSION))
    ),
  )


def _prepared_vertex_colors(scene: Scene, plan: MeshPlan, rendered: Column[np.int32]) -> tuple[bool, bool]:
  """(the rendered meshes have vertex colors and a material reads vertex colors, one of those
  meshes' vertex colors has an alpha below 1): what a material edit can turn on."""
  color_counts = np.asarray(scene.meshes.color_num)[_host_meshes(plan, rendered)]
  colored = bool(np.any(color_counts > 0)) and bool(
    np.any(np.asarray(scene.materials.ext_flags) & int(MaterialFlags.VERTEX_COLOR))
  )
  return colored, colored and _translucent_colors(scene, plan, rendered)


def _host_meshes(plan: MeshPlan, instances: Column[np.int32]) -> Column[np.int64]:
  """The distinct host meshes the mesh instances among `instances` read."""
  core = plan.core[instances]
  return plan.used[np.unique(core[core >= 0])]


def _translucent_colors(scene: Scene, plan: MeshPlan, instances: Column[np.int32]) -> bool:
  """Whether a vertex color of the meshes of `instances` has an alpha below 1."""
  meshes = _host_meshes(plan, instances)
  colors = scene.meshes.colors
  return any(
    bool(np.any(colors[adr : adr + count, 3] < 1.0))
    for adr, count in zip(
      np.asarray(scene.meshes.color_adr)[meshes].tolist(), np.asarray(scene.meshes.color_num)[meshes].tolist(), strict=True
    )
  )


def _texture_slots(tables: PreparedMaterials, ids: Column[np.int64]) -> TextureSlots:
  """Texture roles of exactly the prepared material rows `ids`."""
  used = tables.texid[ids]

  def has(*roles: TexRole) -> bool:
    """Whether one of these rows binds a texture to one of `roles`."""
    return bool(np.any(used[:, [int(r) for r in roles]] >= 0))

  return TextureSlots(
    base_color=has(TexRole.RGB, TexRole.RGBA),
    normal=has(TexRole.NORMAL),
    orm=has(TexRole.ORM),
    emissive=has(TexRole.EMISSIVE),
    opacity=has(TexRole.OPACITY),
    diffuse_roughness=used.shape[1] > int(TexRole.DIFFUSE_ROUGHNESS) and has(TexRole.DIFFUSE_ROUGHNESS),
    layered=used.shape[1] > int(TexRole.FUZZ) and has(TexRole.ANISOTROPY, TexRole.FUZZ),
    coat_normal=used.shape[1] > int(TexRole.COAT_NORMAL) and has(TexRole.COAT_NORMAL),
    detail_normal=used.shape[1] > int(TexRole.DETAIL_NORMAL) and has(TexRole.DETAIL_NORMAL),
    scaled_normal=bool(np.any(tables.scaled_normal_maps()[ids])),
  )


def _layer_use(tables: PreparedMaterials, ids: Column[np.int64]) -> LayerUse:
  """The layered lobes of exactly the material rows `ids`."""
  layers = tables.derived.layers
  flags = tables.layer_flags[ids]
  coated = tables.derived.coat[ids] > 0.0
  mapped = tables.texid.shape[1] > int(TexRole.ANISOTROPY) and bool(np.any(tables.texid[ids, TexRole.ANISOTROPY] >= 0))
  return LayerUse(
    anisotropy=bool(np.any(layers.anisotropy[ids, 0] != 0.0)) or mapped,
    fuzz=bool(np.any(layers.fuzz[ids, 0] > 0.0)),
    coat=bool(np.any(coated & ((flags & int(LayerFlag.COAT_OPENPBR)) != 0))),
    coat_own_normal=bool(np.any(coated & ((flags & int(LayerFlag.COAT_BASE_NORMAL)) == 0))),
  )
