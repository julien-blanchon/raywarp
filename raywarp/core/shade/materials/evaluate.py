"""What the surface looks like where a ray hit it: its material parameters, textures and shading
normal, assembled from the instance, material and mesh tables.

`make_eval_surface(f)` returns eval_surface(geo, mats, worldid, hit, point, direction, footprint)
-> (Surface, world shading normal), composed of named `@wp.func` pieces (not forced inline:
measured on the CPU, `inline=True` on the shading pieces cost 11 % on primitives, 4 % on a
textured glTF asset, by code growth; the compiler inlines what pays):

  flex_surface                      a flex hit's whole surface (`shade/materials/flex.py`)
  instance_surface                  base parameters (instance + material tables)
  layer_inputs                      the layered lobes' material values (`shade/materials/layers.py`)
  shape_normal                      vertex colors and the shading normal
                                    (`shade/materials/shape_normal.py`)
  apply_maps                        the material's texture maps (`uv_frame.py`, `maps.py`)
  finish_surface                    emission, alpha cut-out, clearcoat bits, specular AA, F90,
                                    a transmissive layer's specular-only flag
  coat_frame / layer_frame          the coat's normal, the anisotropy frame and fuzz albedo
                                    (`shade/materials/layers.py`)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import warp as wp

from raywarp._base.types import (
  COAT_ROUGHNESS_SHIFT,
  COAT_WEIGHT_SHIFT,
  DEFAULT_MATERIAL_PARAMS,
  TWO_SIDED,
  ShadingFlag,
  ShapeType,
)
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import (
  FLEX_HIT,
  GeometryFields,
  Hit,
  frame_scale,
  instance_pose,
  offset_floor,
  ray_offset,
  world_entry,
)
from raywarp.core.kernels.abi.tables import batched, instance_rgba, material_value
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.materials import brdf, maps, texture
from raywarp.core.shade.materials.flex import make_flex_surface
from raywarp.core.shade.materials.layers import make_coat_frame, make_layer_frame, make_layer_inputs
from raywarp.core.shade.materials.material import (
  OPAQUE,
  MaterialTables,
  Surface,
  diffuse_parameters,
  specular_weight_parameter,
)
from raywarp.core.shade.materials.shape_normal import face_viewer, make_shape_normal
from raywarp.core.shade.materials.uv_frame import make_apply_maps

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_SPECULAR_CURVE: Final = int(ShadingFlag.SPECULAR_CURVE)
_TRANSMISSIVE: Final = int(ShadingFlag.TRANSMISSIVE)
_UNLIT: Final = int(ShadingFlag.UNLIT)
_DISPLAY_EMISSION: Final = int(ShadingFlag.DISPLAY_EMISSION)
_MESH: Final = int(ShapeType.MESH)
_SDF: Final = int(ShapeType.SDF)
_FLEX: Final = int(ShapeType.FLEX)

# Filament's geometric specular anti-aliasing constants (screen-space variance, threshold).
_SPECULAR_AA_VARIANCE: Final = 0.15
_SPECULAR_AA_THRESHOLD: Final = 0.2


# Instances without a material: MuJoCo's default material (shininess 0.5 ->
# perceptual roughness 0.5, specular 0.5 -> reflectance 0.5), a dielectric.
_NO_METALLIC: Final = DEFAULT_MATERIAL_PARAMS[0]
_NO_ROUGHNESS: Final = DEFAULT_MATERIAL_PARAMS[1]
_NO_REFLECTANCE: Final = DEFAULT_MATERIAL_PARAMS[2]


@wp.func
def facing(
  # In:
  gmat_t: wp.mat33,
  direction: wp.vec3,
  normal: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.float32]:
  """(ray direction, geometric normal turned to the viewer, side seen) in the shape frame.

  Side +1: the viewer sees the front (the outward / winding normal's side); -1 the back.
  """
  local_dir = gmat_t @ direction
  local_normal = gmat_t @ normal
  side = wp.float32(1.0)
  if wp.dot(local_normal, local_dir) > 0.0:
    local_normal = -local_normal
    side = -1.0
  return local_dir, local_normal, side


@wp.func
@device_generic
def instance_surface[G: GeometryFields, M: MaterialTables](
  # Model:
  geo: G,
  mats: M,
  # In:
  worldid: wp.int32,
  instance_id: wp.int32,
) -> tuple[Surface, wp.int32, wp.float32]:
  """(surface, material id, emission strength) of an instance from its and its material's tables.

  Metallic, roughness and reflectance are the material's (looked up through the live
  `instance_material`), MuJoCo's default material without one.
  """
  surface = Surface()
  # The build word, for the gates past the carrier (`toggles.bound_on`).
  surface.bits = mats.bits
  surface.bits2 = geo.bits2
  surface.occlusion = 1.0
  surface.emissive = wp.vec3(0.0)
  surface.f90 = wp.vec3(1.0)
  rgba, mat_id = instance_rgba(
    geo.color_rule, geo.instance_material, geo.instance_color, geo.material_base_color, worldid, instance_id
  )
  surface.base_color = wp.vec3(rgba[0], rgba[1], rgba[2])
  surface.alpha = rgba[3]
  surface.metallic = material_value(mats.material_base_metalness, worldid, mat_id, wp.static(_NO_METALLIC))
  surface.perceptual_roughness = material_value(mats.material_specular_roughness, worldid, mat_id, wp.static(_NO_ROUGHNESS))
  surface.reflectance = material_value(mats.material_reflectance, worldid, mat_id, wp.static(_NO_REFLECTANCE))
  emission = wp.float32(0.0)
  if mat_id >= 0:
    emission = batched(mats.material_emission, worldid, mat_id)
  return surface, mat_id, emission


# eval_surface(geo, mats, worldid, hit, point, direction, footprint) -> (surface, shading normal).
type EvalSurface = Callable[[GeometryFields, MaterialTables, int, Hit, wp.vec3, wp.vec3, float], tuple[Surface, wp.vec3]]


@wp.func
@device_generic
def mirror_reflectance[M: MaterialTables](
  # Model:
  mats: M,
  # In:
  worldid: wp.int32,
  mat_id: wp.int32,
  reflectance: wp.float32,
) -> wp.float32:
  """The dielectric reflectance of a material with a MuJoCo `reflectance` r
  (`Materials.material_mirror`): r is the specular lobe's F0, at least r (the reflectance of
  F0 = 0.16 x reflectance^2 = r). Its reflection, roughness, Fresnel and energy coupling are every
  other specular lobe's (the glossy reflection rays: `kernels/reflections.py`)."""
  if mat_id >= 0:
    r = batched(mats.material_mirror, worldid, mat_id)
    if r > 0.0:
      return wp.max(reflectance, wp.sqrt(r * wp.static(1.0 / brdf.DIELECTRIC_F0_SCALE)))
  return reflectance


@wp.func
def _specular_aa(
  # In:
  surface: Surface,
  dn: float,
  coated: bool,
) -> Surface:
  """`surface` with its roughness widened for a normal change `dn` across the footprint (Filament:
  α² += min(2 σ², threshold)), the coat's too when `coated`."""
  kernel = wp.min(2.0 * wp.static(_SPECULAR_AA_VARIANCE) * dn * dn, wp.static(_SPECULAR_AA_THRESHOLD))
  surface.perceptual_roughness = maps.widen_roughness(surface.perceptual_roughness, kernel)
  if coated:
    surface.clearcoat_roughness = maps.widen_roughness(surface.clearcoat_roughness, kernel)
  return surface


@wp.func
def unlit_surface(
  # In:
  surface: Surface,
) -> Surface:
  """`surface` of an unlit material (`ShadingFlag.UNLIT`): its base color moved into its emission
  (shown display-relative at the emission sites: `Surface.unlit`), no lobe left for light to
  reflect (black base, no specular, no coat)."""
  surface.emissive = surface.base_color + surface.emissive
  surface.base_color = wp.vec3(0.0)
  surface.metallic = 0.0
  surface.reflectance = 0.0
  surface.clearcoat = 0.0
  surface.unlit = True
  surface.display_emission = True
  return surface


def _make_unlit(f: Features):
  """unlit(mats, worldid, instance_id, surface) -> `surface`, as `unlit_surface` where its
  instance's material is unlit (`ShadingFlag.UNLIT`), with its emission display-relative where
  the material's is (`ShadingFlag.DISPLAY_EMISSION`, `Surface.display_emission`; both
  Features.unlit_emission, read where the build word has the materials class and its field bit,
  `toggles.MAT_UNLIT`)."""
  compiled = f.unlit_emission
  gate = toggles.bound_on(f.bound)
  field = toggles.bound_on(f.bound, high=True)

  @wp.func(inline=True)
  @device_generic
  def unlit[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    worldid: wp.int32,
    instance_id: wp.int32,
    surface: Surface,
  ) -> Surface:
    """`surface`, unlit where its material is, its emission display-relative where the
    material's is."""
    if wp.static(compiled):
      if gate(surface.bits, wp.static(toggles.CLASS_MATERIALS)) and field(surface.bits2, wp.static(toggles.MAT_UNLIT)):
        shading = batched(mats.instance_shading, worldid, instance_id)
        if (shading & wp.static(_UNLIT)) != 0:
          return unlit_surface(surface)
        surface.display_emission = (shading & wp.static(_DISPLAY_EMISSION)) != 0
    return surface

  return unlit


@specialize
def make_finish_surface(f: Features):
  """finish_surface(...) -> the surface after emission, cut-out, clearcoat, specular AA, F90 and the
  unlit materials (`unlit_surface`)."""
  alpha_cutout = f.alpha_cutout
  clearcoat = f.clearcoat
  specular_aa = f.specular_aa
  specular_curve = f.specular_curve
  transmissive = f.transmissive
  unlit = _make_unlit(f)
  gate = toggles.bound_on(f.bound)  # the coat class bit
  field = toggles.bound_on(f.bound, high=True)  # the transmissive field bit
  setting = toggles.setting_on(f)  # translucent layers: a tier build reads them at run time

  @wp.func
  @device_generic
  def finish_surface[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    # In:
    worldid: wp.int32,
    instance_id: wp.int32,
    mat_id: wp.int32,
    emission: wp.float32,
    cone: wp.float32,
    curvature: wp.float32,
    surface: Surface,
  ) -> Surface:
    """The parameters that depend on the textured ones (emission: 0 without a material)."""
    if emission > 0.0:
      surface.emissive = surface.base_color * emission
    bits = wp.int32(0)
    if wp.static(clearcoat or specular_curve):
      # The instance's shading bits (the coat's weight and roughness bytes, ShadingFlag), read once.
      bits = batched(mats.instance_shading, worldid, instance_id)

    if wp.static(alpha_cutout):
      # A cut-out surface that passed the alpha test (cutout.py) is opaque.
      if mat_id >= 0:
        if batched(geo.cut.material_alpha_cutoff, worldid, mat_id) > 0.0:
          surface.alpha = 1.0

    coated = False
    if wp.static(clearcoat):
      coated = gate(surface.bits, wp.static(toggles.CLASS_COAT))
    if coated:
      surface.clearcoat = wp.float32((bits >> wp.static(COAT_WEIGHT_SHIFT)) & 255) * wp.static(1.0 / 255.0)
      surface.clearcoat_roughness = wp.float32((bits >> wp.static(COAT_ROUGHNESS_SHIFT)) & 255) * wp.static(1.0 / 255.0)

    if wp.static(specular_aa):  # the normal maps' Toksvig term: `uv_maps`
      surface = _specular_aa(surface, cone * curvature, coated)

    if wp.static(specular_curve):
      if gate(surface.bits, wp.static(toggles.SCENE_SPECULAR_CURVE)) and (bits & wp.static(_SPECULAR_CURVE)) != 0:
        surface.preserve_specular_coefficients = True
        surface.f90 = brdf.f90_curve(surface.base_color, surface.metallic, surface.reflectance, surface.perceptual_roughness)
    if wp.static(transmissive):
      # A translucent layer of glass reflects and transmits, no diffuse lobe (translucent layers
      # on: with them off every surface is opaque, shaded whole; the transmissive field bit on).
      if (
        field(surface.bits2, wp.static(toggles.SURF_TRANSMISSIVE))
        and surface.alpha < wp.static(OPAQUE)
        and setting(geo.rt.flags, geo.bits2, wp.static(toggles.TRANSPARENCY))
      ):
        if wp.static(not (clearcoat or specular_curve)):
          bits = batched(mats.instance_shading, worldid, instance_id)  # translucent hits only
        surface.specular_only = (bits & wp.static(_TRANSMISSIVE)) != 0
    return unlit(mats, worldid, instance_id, surface)

  return finish_surface


@wp.func(inline=True)
@device_generic
def front_emission[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  instance_id: int,
  side: float,
  emissive: wp.vec3,
) -> wp.vec3:
  """`emissive` of a surface seen from `side` (< 0: its back): a single-sided emitter's back emits
  nothing. Nested: only an emitting back face reads the two-sided table."""
  if side < 0.0:
    if emissive[0] + emissive[1] + emissive[2] > 0.0:
      if (geo.instance_twosided[instance_id] & wp.static(TWO_SIDED)) == 0:
        return wp.vec3(0.0)
  return emissive


@specialize
def make_eval_surface(f: Features):  # noqa: C901, PLR0915 (one static branch and class gate per lobe / role)
  """eval_surface(geo, mats, worldid, hit, point, direction, footprint) -> (Surface, normal).

  Specialised on the shape types and material features of `f` (see the module docstring).
  """
  instance_types = f.shape_types
  pose = instance_pose(compact=f.compact_poses, bound=f.bound)
  has_mesh = _MESH in instance_types or _SDF in instance_types
  has_primitive = any(t not in (_MESH, _SDF, _FLEX) for t in instance_types)
  flex = _FLEX in instance_types  # deformables (their hits: `FLEX_HIT`)
  textures = f.textures
  shape_normal = make_shape_normal(f, has_mesh=has_mesh, has_primitive=has_primitive)
  apply_maps = make_apply_maps(f, has_mesh=has_mesh, has_primitive=has_primitive)
  finish_surface = make_finish_surface(f)
  diffuse = diffuse_parameters(enabled=f.diffuse_roughness)
  weight_of = specular_weight_parameter(enabled=f.specular_weight)
  flex_surface = make_flex_surface(f)
  layered = f.layered
  coat_frame = f.coat_normal or f.coat_layer
  layer_inputs = make_layer_inputs(f)
  coat_frame_of = make_coat_frame(f)
  world_optics = f.world_optics  # the emitter's two-sided bit: its world's (`trace.world_entry`)
  layer_frame = make_layer_frame(f, has_mesh=has_mesh, has_primitive=has_primitive)
  gate = toggles.bound_on(f.bound)  # the material, layers, map class and deformables bits
  field = toggles.bound_on(f.bound, high=True)  # the materials class's fields (`toggles.MAT_*`)
  # Any map class: `apply_maps` gates each slot on its own class.
  any_map = toggles.CLASS_BASE_MAP | toggles.CLASS_MAPS | toggles.CLASS_LAYER_MAPS

  @wp.func
  @device_generic
  def eval_surface[G: GeometryFields, M: MaterialTables](  # noqa: C901 (one static branch and class gate per lobe / role)
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
    """Surface parameters and the viewer-facing shading normal at a hit.

    `footprint` is the ray-cone width at the hit point (before projection on the surface).
    """
    instance_id = hit.instance_id
    if wp.static(flex):
      if gate(geo.bits, wp.static(toggles.FLEX)) and instance_id == wp.static(FLEX_HIT):
        flex_hit, flex_normal = flex_surface(geo, mats, worldid, hit, point, direction, footprint)
        return flex_hit, flex_normal
    surface, mat_id, emission = instance_surface(geo, mats, worldid, instance_id)
    materials_on = False
    if wp.static(f.reflection or f.specular_weight or f.diffuse_roughness):
      materials_on = gate(surface.bits, wp.static(toggles.CLASS_MATERIALS))
    if wp.static(f.reflection):
      if materials_on and field(surface.bits2, wp.static(toggles.MAT_REFLECTION)):
        surface.reflectance = mirror_reflectance(mats, worldid, mat_id, surface.reflectance)
    if wp.static(f.specular_weight):
      if materials_on and field(surface.bits2, wp.static(toggles.MAT_SPECULAR_WEIGHT)):
        weight, model = weight_of(mats, worldid, mat_id)
        surface.specular_weight = weight
        surface.preserve_specular_coefficients = (model & wp.static(int(ShadingFlag.DIELECTRIC_SPECULAR_LEVEL))) != 0
    if wp.static(f.diffuse_roughness):
      if materials_on and field(surface.bits2, wp.static(toggles.MAT_DIFFUSE_ROUGHNESS)):
        diffuse_values = diffuse(mats, worldid, mat_id)
        surface.diffuse_roughness = diffuse_values[0]
        surface.diffuse_base_weight = diffuse_values[1]
    layered_on = False
    if wp.static(layered or coat_frame):
      layered_on = gate(surface.bits, wp.static(toggles.CLASS_LAYERS))
    if wp.static(layered):
      if layered_on:
        surface = layer_inputs(mats, worldid, mat_id, surface)

    gtype = geo.instance_type[instance_id]
    # The shape frame: the instance's rotation, in world units (a mesh instance's frame s R:
    # what is read from its mesh vertices is scaled by s).
    gpos, frame = pose(geo, worldid, instance_id)
    scale = frame_scale(frame)
    gmat = frame * (1.0 / scale)
    gmat_t = wp.transpose(gmat)
    local_dir, local_normal, side = facing(gmat_t, direction, hit.normal)
    is_mesh = gtype == _MESH or gtype == _SDF
    surface, shading, side, lift, curvature = shape_normal(
      geo, mats, worldid, hit, gtype, is_mesh, scale, local_normal, side, surface
    )

    cone = texture.surface_cone(footprint, local_normal, local_dir)
    unmapped = shading  # the coat's normal without its own map (OpenPBR / glTF default)
    local = gmat_t @ (point - gpos)

    if wp.static(textures):
      if gate(surface.bits, wp.static(any_map)) and mat_id >= 0:
        surface, shading, emission = apply_maps(
          geo,
          mats,
          worldid,
          hit,
          gtype,
          is_mesh,
          mat_id,
          point,
          gmat_t,
          scale,
          local,
          local_dir,
          local_normal,
          cone,
          footprint,
          side,
          shading,
          surface,
          emission,
        )

    # Interpolated and mapped normals can face away from the viewer near silhouettes.
    shading = face_viewer(shading, -local_dir)
    surface = finish_surface(geo, mats, worldid, instance_id, mat_id, emission, cone, curvature, surface)
    # A single-sided emitter emits from its front face only (USD / Mitsuba): its back is dark for
    # camera, reflection and probe rays alike, as its panel light (`scene/lighting/emissive.py`)
    # lights only the front side. Gated on the finished emissive color (an emissive map or tint sets
    # it).
    if wp.static(world_optics):
      entry = world_entry(geo.instance_type.shape[0], worldid, instance_id)
      surface.emissive = front_emission(geo, entry, side, surface.emissive)
    else:
      surface.emissive = front_emission(geo, instance_id, side, surface.emissive)
    # Secondary-ray origin: along the geometric normal, from the lifted point.
    surface.outward = wp.normalize(gmat @ local_normal)
    surface.origin = point + gmat @ lift + surface.outward * ray_offset(point, offset_floor(geo, instance_id))
    normal = wp.normalize(gmat @ shading)
    if wp.static(coat_frame):
      if layered_on:
        surface = coat_frame_of(mats, mat_id, unmapped, shading, local_dir, gmat, normal, direction, surface)
    if wp.static(layered):
      if layered_on:
        nov = wp.max(wp.dot(normal, -direction), 1.0e-4)
        surface = layer_frame(geo, mats, worldid, hit, gtype, is_mesh, local, local_normal, gmat, normal, nov, surface)
    return surface, normal

  return eval_surface
