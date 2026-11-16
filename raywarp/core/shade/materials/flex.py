"""What a deformable (MuJoCo flex: cloth, soft bodies) looks like where a ray hit it: its color,
and with Features.flex_materials its material and textures.

MuJoCo's flex material semantics (`engine_vis_visualize.c`: `setMaterial`, `mjv_updateActiveFlex`):
the flex rgba unless it is the default (0.5, 0.5, 0.5, 1), then the material's (the scene's
`ColorRule`, like geoms); the material's metalness / roughness / reflectance /
emission; its textures only on a surface with texture coordinates (`Deformables.corner_uv`,
`texrepeat` scaling them, no `texuniform`). The maps are the geoms' (`shade/materials/maps.py`:
every role, the normal map in the triangle's UV frame: `uv_maps.make_uv_maps`). Both backends report
a flex hit alike (part, face in its every-world mesh, barycentrics, geometric normal), and this
reads the meshes' world-space triangles (`FlexPart.mesh`).
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import DEFAULT_MATERIAL_PARAMS, ShadingFlag
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields, Hit, offset_floor, ray_offset
from raywarp.core.kernels.abi.tables import batched, material_rgba, material_value
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.materials import brdf, texture
from raywarp.core.shade.materials.material import MaterialTables, Surface, diffuse_parameters, specular_weight_parameter
from raywarp.core.shade.materials.uv_maps import make_taps, make_uv_maps

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_NO_METALLIC: Final = DEFAULT_MATERIAL_PARAMS[0]
_NO_ROUGHNESS: Final = DEFAULT_MATERIAL_PARAMS[1]
_NO_REFLECTANCE: Final = DEFAULT_MATERIAL_PARAMS[2]


@wp.func
@device_generic
def flex_uv_frame[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  hit: Hit,
  adr: wp.int32,
  repeat: wp.vec2,
) -> tuple[wp.vec2, wp.vec3, wp.vec3, wp.float32]:
  """(uv, tangent, bitangent, uv_per_length) of a flex triangle hit, world frame; `adr`: the flex's
  first corner row of `flex_uv`."""
  part = geo.flex_part[hit.mesh_id]
  mesh = part.mesh
  corner = adr + (hit.face_id % part.nface) * 3  # the face in its world's group
  p0 = wp.mesh_get_point(mesh, wp.mesh_get_index(mesh, hit.face_id * 3 + 0))
  e1 = wp.mesh_get_point(mesh, wp.mesh_get_index(mesh, hit.face_id * 3 + 1)) - p0
  e2 = wp.mesh_get_point(mesh, wp.mesh_get_index(mesh, hit.face_id * 3 + 2)) - p0
  uv0 = geo.flex_uv[corner]
  uv1 = geo.flex_uv[corner + 1]
  uv2 = geo.flex_uv[corner + 2]
  return texture.triangle_frame(e1, e2, hit.bary_u, hit.bary_v, uv0, uv1, uv2, repeat)


@wp.func
@device_generic
def flex_uv[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: wp.int32,
  hit: Hit,
  mat_id: wp.int32,
) -> wp.vec2:
  """The texture coordinate the maps of material `mat_id` read at a flex hit (the uv output; 0
  without UVs or a material)."""
  adr = geo.flex_part[hit.mesh_id].uvadr
  if adr < 0 or hit.face_id < 0 or mat_id < 0:
    return wp.vec2(0.0)
  uv, _tangent, _bitangent, _per_length = flex_uv_frame(geo, hit, adr, batched(geo.cut.material_tex_repeat, worldid, mat_id))
  return uv


@specialize
def make_flex_maps(f: Features):
  """flex_maps(geo, mats, worldid, hit, mat_id, direction, normal, side, width, surface,
  emission) -> (surface, shading normal, emission) with the material's maps
  (`uv_maps.make_uv_maps`)."""
  uv_maps = make_uv_maps(f)
  taps = make_taps(f)

  @wp.func
  @device_generic
  def flex_maps[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    # In:
    worldid: wp.int32,
    hit: Hit,
    mat_id: wp.int32,
    direction: wp.vec3,
    normal: wp.vec3,
    side: wp.float32,
    width: wp.float32,
    surface: Surface,
    emission: wp.float32,
  ) -> tuple[Surface, wp.vec3, wp.float32]:
    """The maps at a flex triangle hit with UVs (`normal`: viewer-facing, world frame; `width`: the
    ray cone's width at the hit); unchanged without UVs (MuJoCo: no texture)."""
    adr = geo.flex_part[hit.mesh_id].uvadr
    if adr < 0 or hit.face_id < 0:
      return surface, normal, emission
    repeat = batched(mats.material_tex_repeat, worldid, mat_id)
    uv, tangent, bitangent, uv_per_length = flex_uv_frame(geo, hit, adr, repeat)
    cone = texture.surface_cone(width, normal, direction)
    return uv_maps(
      mats,
      worldid,
      mat_id,
      True,
      uv,
      tangent,
      bitangent,
      uv_per_length,
      cone,
      width,
      direction,
      normal,
      side,
      normal,
      True,
      taps(geo, mats),
      surface,
      emission,
    )

  return flex_maps


@specialize
def make_flex_surface(f: Features):  # noqa: C901, PLR0915 (one static branch and class gate per lobe / role)
  """flex_surface(geo, mats, worldid, hit, point, direction, footprint) -> (surface, viewer-facing
  normal) of a flex hit: its color over the default material, or (Features.flex_materials) its
  material's parameters and maps (see the module doc)."""
  materials = f.flex_materials
  textures = materials and f.textures
  flex_maps = make_flex_maps(f)
  diffuse = diffuse_parameters(enabled=f.diffuse_roughness)
  weight_of = specular_weight_parameter(enabled=f.specular_weight)
  gate = toggles.bound_on(f.bound)  # the materials, map class, deformable-materials and specular-curve bits
  any_map = toggles.CLASS_BASE_MAP | toggles.CLASS_MAPS | toggles.CLASS_LAYER_MAPS

  @wp.func
  @device_generic
  def flex_surface[G: GeometryFields, M: MaterialTables](  # noqa: C901, PLR0915 (a gate per lobe / role)
    # Model:
    geo: G,
    mats: M,
    # In:
    worldid: wp.int32,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
    footprint: wp.float32,
  ) -> tuple[Surface, wp.vec3]:
    """(surface, viewer-facing shading normal) of flex `hit.mesh_id`'s hit at `point`."""
    surface = Surface()
    surface.bits = mats.bits  # the build word (`evaluate.instance_surface`)
    surface.bits2 = geo.bits2
    surface.occlusion = 1.0
    surface.emissive = wp.vec3(0.0)
    surface.f90 = wp.vec3(1.0)
    materials_on = False
    if wp.static(f.specular_weight or f.diffuse_roughness):
      materials_on = gate(surface.bits, wp.static(toggles.CLASS_MATERIALS))
    if wp.static(f.specular_weight):
      if materials_on:
        surface.specular_weight = 1.0
    part = geo.flex_part[hit.mesh_id]
    rgba = part.rgba
    normal = hit.normal
    side = wp.float32(1.0)
    if wp.dot(normal, direction) > 0.0:
      normal = -normal
      side = -1.0
    shading = normal
    model = wp.int32(0)
    with_materials = False
    if wp.static(materials):
      with_materials = gate(surface.bits, wp.static(toggles.FLEX_MATERIALS))
    if with_materials:
      mat_id = part.material
      if wp.static(f.specular_weight):
        if materials_on:
          weight, model = weight_of(mats, worldid, mat_id)
          surface.specular_weight = weight
          surface.preserve_specular_coefficients = (model & wp.static(int(ShadingFlag.DIELECTRIC_SPECULAR_LEVEL))) != 0
      if wp.static(f.diffuse_roughness):
        if materials_on:
          diffuse_values = diffuse(mats, worldid, mat_id)
          surface.diffuse_roughness = diffuse_values[0]
          surface.diffuse_base_weight = diffuse_values[1]
      rgba = material_rgba(geo.color_rule, geo.material_base_color, worldid, mat_id, rgba)
      surface.metallic = material_value(mats.material_base_metalness, worldid, mat_id, wp.static(_NO_METALLIC))
      surface.perceptual_roughness = material_value(mats.material_specular_roughness, worldid, mat_id, wp.static(_NO_ROUGHNESS))
      surface.reflectance = material_value(mats.material_reflectance, worldid, mat_id, wp.static(_NO_REFLECTANCE))
      surface.base_color = wp.vec3(rgba[0], rgba[1], rgba[2])
      surface.alpha = rgba[3]
      emission = material_value(mats.material_emission, worldid, mat_id, 0.0)
      if wp.static(textures):
        if gate(surface.bits, wp.static(any_map)) and mat_id >= 0:
          surface, shading, emission = flex_maps(
            geo, mats, worldid, hit, mat_id, direction, normal, side, footprint, surface, emission
          )
      if wp.static(f.specular_weight and f.specular_curve):
        curve = gate(surface.bits, wp.static(toggles.SCENE_SPECULAR_CURVE)) and materials_on
        if curve and (model & wp.static(int(ShadingFlag.SPECULAR_CURVE))) != 0:
          surface.preserve_specular_coefficients = True
          surface.f90 = brdf.f90_curve(surface.base_color, surface.metallic, surface.reflectance, surface.perceptual_roughness)
      if emission > 0.0:
        surface.emissive = surface.base_color * emission
    else:
      surface.base_color = wp.vec3(rgba[0], rgba[1], rgba[2])
      surface.alpha = rgba[3]
      surface.metallic = wp.static(_NO_METALLIC)
      surface.perceptual_roughness = wp.static(_NO_ROUGHNESS)
      surface.reflectance = wp.static(_NO_REFLECTANCE)
    surface.origin = point + normal * ray_offset(point, offset_floor(geo, -1))
    surface.outward = normal
    return surface, shading

  return flex_surface
