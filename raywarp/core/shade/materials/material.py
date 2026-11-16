"""The description of a surface at a hit (color, metalness, roughness, ...), the material tables it
is read from, and the BRDF constants derived from it once per hit.

The surface is the core's subset of OpenPBR (the material tables hold the values derived from
the canonical `rw.Material` at upload, `pipeline.derive.materials`): a metal / dielectric mix of
`base_metalness`, `specular_roughness`, the reflectance of `specular_weight` / `specular_ior`,
optionally under a coat (Features.clearcoat): a dielectric GGX lobe of F0 0.04 (Features.coat_layer:
of the coat's own IOR) over the base, which receives what the coat's Fresnel transmits. Surfaces
are evaluated at hits by `shade.materials.evaluate`; `Lobes` are the per-hit BRDF constants every
light and the ambient term reuse.

The material carriers (`Materials`, `RoughDiffuseMaterials`, `WeightedMaterials`,
`RoughWeightedMaterials`, `LayeredMaterials`) share their leading fields; each adds the tables of
the optional lobes its Features compile in (`material_type`), so a build without them binds no
extra array descriptor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import warp as wp

from raywarp._base.types import MaterialUvMapping, MeshAdr, tex_affine, uv_xform
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.tables import material_value
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.environment import ibl
from raywarp.core.shade.materials import brdf, openpbr_lobes

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Surfaces with alpha below this are translucent layers (composited front to back by the camera
# and probe rays; shaded specular-only, `make_shading_lobes`).
OPAQUE: Final = 0.99

# F0 of the clearcoat layer (a dielectric of index 1.5, glTF KHR_materials_clearcoat).
COAT_F0: Final = 0.04


@wp.struct
class Surface:
  """Shading inputs at a hit point, in linear color.

  Attributes:
    base_color: base color (albedo for dielectrics, F0 for metals).
    alpha: opacity.
    metallic: metalness in [0, 1].
    perceptual_roughness: roughness before the α = r² remap.
    reflectance: dielectric reflectance, F0 = 0.16 · reflectance².
    specular_weight: OpenPBR `specular_weight` (Features.specular_weight with its field bit; else
      0 and unread by `make_shading_lobes`).
    preserve_specular_coefficients: an authored native specular level / custom curve model
      (ShadingFlag.DIELECTRIC_SPECULAR_LEVEL / SPECULAR_CURVE): the weight leaves its Fresnel
      coefficients as they are.
    emissive: emitted radiance before exposure compensation.
    occlusion: ambient occlusion in [0, 1].
    f90: specular reflectance at grazing incidence (1, or the Unreal specular curve's).
    clearcoat: clearcoat layer weight in [0, 1] (Features.clearcoat; else 0).
    clearcoat_roughness: the coat's perceptual roughness.
    specular_only: a translucent layer of a transmissive material (Features.transmissive).
    origin: world origin of the secondary rays (shadow, reflection) leaving the hit: the hit
      point moved onto the vertex tangent planes of a smooth mesh (Hanika 2021: no shadow-
      terminator notches on low-poly meshes), then offset by `ray_offset` along the geometric
      normal on the viewer's side (never along the shading normal: a bent or mapped normal
      would start rays under the surface).
    outward: the unit geometric normal on the viewer's side. Reflection rays under it are not
      traced (a mapped or bent normal reflects grazing views under the surface, where they
      hit the surface itself and show a displaced copy of its texture).
    bits, bits2: the build word's copy, both halves (the carrier's first, the geometry's second:
      `GeometryFields.bits2`), read by the bound gates past the carrier (`toggles.bound_on`; a
      constant once the OptiX module binds it).
  """

  bits: wp.int32
  bits2: wp.int32
  base_color: wp.vec3
  alpha: wp.float32
  metallic: wp.float32
  perceptual_roughness: wp.float32
  reflectance: wp.float32
  specular_weight: wp.float32
  preserve_specular_coefficients: bool
  emissive: wp.vec3
  occlusion: wp.float32
  f90: wp.vec3
  clearcoat: wp.float32
  clearcoat_roughness: wp.float32
  # A translucent layer of a transmissive material: its specular, its diffuse weighted by its
  # opacity (`make_shading_lobes`).
  specular_only: bool
  # An unlit material (`ShadingFlag.UNLIT`, Features.unlit_emission): no lobe, its color in
  # `emissive`, shown display-relative (the emission sites).
  unlit: bool
  # Display-relative emission (an unlit material, or `ShadingFlag.DISPLAY_EMISSION`: glTF's
  # emissive factor on a lit surface), read by the emission sites (`make_relative_emission`).
  display_emission: bool
  origin: wp.vec3
  outward: wp.vec3
  diffuse_roughness: float
  diffuse_base_weight: float
  # Layered lobes (Features.layered; zero / unread otherwise). Anisotropy a in [0, 1] stretching
  # the lobe along `tangent` (world, unit, orthogonal to the shading normal; `bitangent` = n x t;
  # `layers.layer_inputs` holds the material's rotation in tangent[0] until `layer_frame`).
  anisotropy: float
  tangent: wp.vec3
  bitangent: wp.vec3
  # Fuzz: weight F, Charlie alpha, color, and its directional albedo E_fuzz(NoV) (cloth table).
  fuzz_weight: float
  fuzz_alpha: float
  fuzz_color: wp.vec3
  fuzz_albedo: float
  # Cloth wrapped diffuse width (0: Lambert) and subsurface tint.
  wrap: float
  wrap_tint: wp.vec3
  # The OpenPBR coat (Features.coat_layer): its law (False: glTF's), color, IOR, darkening.
  coat_openpbr: bool
  coat_color: wp.vec3
  coat_ior: float
  coat_darkening: float
  # The coat's own shading normal (world, viewer-facing) and its NoV (Features.coat_layer /
  # coat_normal; the base's without Features.coat_normal).
  coat_normal: wp.vec3
  coat_nov: float
  # The coat normal map's normal in the shape frame (zero: no coat map), until `eval_surface`.
  coat_normal_local: wp.vec3


@wp.struct
class LayerConstants:
  """A material's static layer constants (Features.layered;
  `core.pipeline.gpu_scene.gpu_scene.LayerTables`, uploaded by
  `context.RendererTables.prepare_layers`).

  Attributes:
    coat_darkening: OpenPBR `coat_darkening` d in [0, 1] (0: an untinted glTF-law coat).
    wrap: cloth wrapped-diffuse width w (0: Lambert; `RaywarpExt.wrap`).
    wrap_tint: the cloth's subsurface tint saturate(tint + NoL) (Filament's cloth model).
    detail_uv: the detail normal map's UV transform applied to the material's base UVs.
    flags: LayerFlag bits.
  """

  coat_darkening: wp.float32
  wrap: wp.float32
  wrap_tint: wp.vec3
  detail_uv: uv_xform
  flags: wp.int32


@wp.struct
class Materials:
  """Material, texture and mesh-attribute arrays read by surface evaluation."""

  # Model:
  mesh_adr: wp.array1d[MeshAdr]
  mesh_vert: wp.array1d[wp.vec3]
  mesh_normal: wp.array1d[wp.vec3]
  mesh_face: wp.array1d[wp.vec3i]
  material_texture: wp.array3d[wp.int32]
  # Texel map per material and TexRole (scale * texel + bias, `types.tex_affine`).
  material_tex_affine: wp.array3d[tex_affine]
  # Texture variation per material (u, v offset, UV rotation, hue; `material.tex_variation`).
  material_tex_variation: wp.array2d[wp.vec4]
  material_tex_uniform: wp.array2d[bool]
  # The run-time texture taps (`Toggles.anisotropy`; read by a build reading them at run time).
  texture_taps: int
  # The build word's copy (`kernels/abi/toggles.py`, `tiers.word`): every bound bit but the
  # geometry's (`GEO_*`), which an OptiX module binds (`trace/optix/device/params.bound_words`);
  # unread without bound bits. The first half (the second: `GeometryFields.bits2`, copied into
  # `Surface.bits2`).
  bits: wp.int32
  material_tex_repeat: wp.array2d[wp.vec2]
  # World-projected UV map per material (Features.world_uv; zeros: mesh UVs).
  material_uv_world: wp.array2d[uv_xform]
  material_emission: wp.array2d[wp.float32]
  # Render context:
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_facenormal: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  material_base_metalness: wp.array2d[wp.float32]
  material_specular_roughness: wp.array2d[wp.float32]
  material_reflectance: wp.array2d[wp.float32]
  instance_shading: wp.array2d[wp.int32]
  # Light standing for each instance's emission (emissive panels turned into lights), -1 none:
  # GI probe rays and reflection rays skip that emission while the light is active.
  instance_emitter_light: wp.array1d[wp.int32]
  material_mirror: wp.array2d[wp.float32]
  textures: wp.array1d[wp.Texture2D]
  tex_size: wp.array1d[wp.vec2]
  # Texel layout per texture (`types.TEX_LAYOUT_*`, `texture.apply_layout`).
  tex_layout: wp.array1d[wp.int32]
  # Vertex colors (read only with Features.vertex_colors; per-mesh address: `MeshAdr.color`)
  # and per-face corner indices (aligned with `mesh_face`); the instances whose
  # material reads them: `ShadingFlag.VERTEX_COLOR` of `instance_shading`.
  mesh_facecolor: wp.array1d[wp.vec3i]
  mesh_color: wp.array1d[wp.vec4]
  # The alpha test's own tables (`trace/accel/cutout.py`): an OptiX shaded build's geometry holds
  # its carrier as `GeometryFields.cut` (one copy of the shared texture / UV tables in its launch
  # parameters, `trace/optix/device/params.py`). Cutoff per material (0: no test); the live-material
  # UV mappings (empty without Features.live_material_uv).
  material_alpha_cutoff: wp.array2d[wp.float32]
  material_uv_mapping: wp.array1d[MaterialUvMapping]
  mesh_uv_second: wp.array1d[wp.vec2i]
  # Per (world, instance), its material's optics rows (`pipeline/upload/optics.py`, (1 or nworld,
  # max(ninstance, 1), OPTICS_ROWS)):
  # Beer-Lambert extinction and refraction index, read at transmissive layers; the rows a
  # surface's ShadingFlag.OPTICS bit selects (run-time data, no kernel key).
  instance_optics: wp.array2d[wp.vec4]


@wp.struct
class RoughDiffuseMaterials:
  """`Materials` plus the EON diffuse-roughness tables (Features.diffuse_roughness)."""

  mesh_adr: wp.array1d[MeshAdr]
  mesh_vert: wp.array1d[wp.vec3]
  mesh_normal: wp.array1d[wp.vec3]
  mesh_face: wp.array1d[wp.vec3i]
  material_texture: wp.array3d[wp.int32]
  material_tex_affine: wp.array3d[tex_affine]
  material_tex_variation: wp.array2d[wp.vec4]
  material_tex_uniform: wp.array2d[bool]
  # The run-time texture taps (`Toggles.anisotropy`; read by a build reading them at run time).
  texture_taps: int
  # The build word's copy (`kernels/abi/toggles.py`, `tiers.word`): every bound bit but the
  # geometry's (`GEO_*`), which an OptiX module binds (`trace/optix/device/params.bound_words`);
  # unread without bound bits. The first half (the second: `GeometryFields.bits2`, copied into
  # `Surface.bits2`).
  bits: wp.int32
  material_tex_repeat: wp.array2d[wp.vec2]
  material_uv_world: wp.array2d[uv_xform]
  material_emission: wp.array2d[wp.float32]
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_facenormal: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  material_base_metalness: wp.array2d[wp.float32]
  material_specular_roughness: wp.array2d[wp.float32]
  material_reflectance: wp.array2d[wp.float32]
  instance_shading: wp.array2d[wp.int32]
  instance_emitter_light: wp.array1d[wp.int32]
  material_mirror: wp.array2d[wp.float32]
  textures: wp.array1d[wp.Texture2D]
  tex_size: wp.array1d[wp.vec2]
  tex_layout: wp.array1d[wp.int32]
  mesh_facecolor: wp.array1d[wp.vec3i]
  mesh_color: wp.array1d[wp.vec4]
  material_alpha_cutoff: wp.array2d[wp.float32]
  material_uv_mapping: wp.array1d[MaterialUvMapping]
  mesh_uv_second: wp.array1d[wp.vec2i]
  # Per (world, instance), its material's optics rows (`pipeline/upload/optics.py`, (1 or nworld,
  # max(ninstance, 1), OPTICS_ROWS)):
  # Beer-Lambert extinction and refraction index, read at transmissive layers; the rows a
  # surface's ShadingFlag.OPTICS bit selects (run-time data, no kernel key).
  instance_optics: wp.array2d[wp.vec4]
  material_diffuse_roughness: wp.array2d[wp.float32]
  material_diffuse_source: wp.array1d[wp.vec2]


@wp.struct
class WeightedMaterials:
  """`Materials` plus the specular-weight tables (Features.specular_weight)."""

  # Model:
  mesh_adr: wp.array1d[MeshAdr]
  mesh_vert: wp.array1d[wp.vec3]
  mesh_normal: wp.array1d[wp.vec3]
  mesh_face: wp.array1d[wp.vec3i]
  material_texture: wp.array3d[wp.int32]
  # Texel map per material and TexRole (scale * texel + bias, `types.tex_affine`).
  material_tex_affine: wp.array3d[tex_affine]
  # Texture variation per material (u, v offset, UV rotation, hue; `material.tex_variation`).
  material_tex_variation: wp.array2d[wp.vec4]
  material_tex_uniform: wp.array2d[bool]
  # The run-time texture taps (`Toggles.anisotropy`; read by a build reading them at run time).
  texture_taps: int
  # The build word's copy (`kernels/abi/toggles.py`, `tiers.word`): every bound bit but the
  # geometry's (`GEO_*`), which an OptiX module binds (`trace/optix/device/params.bound_words`);
  # unread without bound bits. The first half (the second: `GeometryFields.bits2`, copied into
  # `Surface.bits2`).
  bits: wp.int32
  material_tex_repeat: wp.array2d[wp.vec2]
  # World-projected UV map per material (Features.world_uv; zeros: mesh UVs).
  material_uv_world: wp.array2d[uv_xform]
  material_emission: wp.array2d[wp.float32]
  # Render context:
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_facenormal: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  material_base_metalness: wp.array2d[wp.float32]
  material_specular_roughness: wp.array2d[wp.float32]
  material_reflectance: wp.array2d[wp.float32]
  instance_shading: wp.array2d[wp.int32]
  # Light standing for each instance's emission (emissive panels turned into lights), -1 none:
  # GI probe rays and reflection rays skip that emission while the light is active.
  instance_emitter_light: wp.array1d[wp.int32]
  material_mirror: wp.array2d[wp.float32]
  textures: wp.array1d[wp.Texture2D]
  tex_size: wp.array1d[wp.vec2]
  # Texel layout per texture (`types.TEX_LAYOUT_*`, `texture.apply_layout`).
  tex_layout: wp.array1d[wp.int32]
  # Vertex colors (read only with Features.vertex_colors; per-mesh address: `MeshAdr.color`)
  # and per-face corner indices (aligned with `mesh_face`); the instances whose
  # material reads them: `ShadingFlag.VERTEX_COLOR` of `instance_shading`.
  mesh_facecolor: wp.array1d[wp.vec3i]
  mesh_color: wp.array1d[wp.vec4]
  material_alpha_cutoff: wp.array2d[wp.float32]
  material_uv_mapping: wp.array1d[MaterialUvMapping]
  mesh_uv_second: wp.array1d[wp.vec2i]
  # Per (world, instance), its material's optics rows (`pipeline/upload/optics.py`, (1 or nworld,
  # max(ninstance, 1), OPTICS_ROWS)):
  # Beer-Lambert extinction and refraction index, read at transmissive layers; the rows a
  # surface's ShadingFlag.OPTICS bit selects (run-time data, no kernel key).
  instance_optics: wp.array2d[wp.vec4]
  material_specular_weight: wp.array2d[wp.float32]
  material_specular_models: wp.array1d[wp.int32]


@wp.struct
class RoughWeightedMaterials:
  """`Materials` plus the EON and the specular-weight tables (both Features flags)."""

  mesh_adr: wp.array1d[MeshAdr]
  mesh_vert: wp.array1d[wp.vec3]
  mesh_normal: wp.array1d[wp.vec3]
  mesh_face: wp.array1d[wp.vec3i]
  material_texture: wp.array3d[wp.int32]
  material_tex_affine: wp.array3d[tex_affine]
  material_tex_variation: wp.array2d[wp.vec4]
  material_tex_uniform: wp.array2d[bool]
  # The run-time texture taps (`Toggles.anisotropy`; read by a build reading them at run time).
  texture_taps: int
  # The build word's copy (`kernels/abi/toggles.py`, `tiers.word`): every bound bit but the
  # geometry's (`GEO_*`), which an OptiX module binds (`trace/optix/device/params.bound_words`);
  # unread without bound bits. The first half (the second: `GeometryFields.bits2`, copied into
  # `Surface.bits2`).
  bits: wp.int32
  material_tex_repeat: wp.array2d[wp.vec2]
  material_uv_world: wp.array2d[uv_xform]
  material_emission: wp.array2d[wp.float32]
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_facenormal: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  material_base_metalness: wp.array2d[wp.float32]
  material_specular_roughness: wp.array2d[wp.float32]
  material_reflectance: wp.array2d[wp.float32]
  instance_shading: wp.array2d[wp.int32]
  instance_emitter_light: wp.array1d[wp.int32]
  material_mirror: wp.array2d[wp.float32]
  textures: wp.array1d[wp.Texture2D]
  tex_size: wp.array1d[wp.vec2]
  tex_layout: wp.array1d[wp.int32]
  mesh_facecolor: wp.array1d[wp.vec3i]
  mesh_color: wp.array1d[wp.vec4]
  material_alpha_cutoff: wp.array2d[wp.float32]
  material_uv_mapping: wp.array1d[MaterialUvMapping]
  mesh_uv_second: wp.array1d[wp.vec2i]
  # Per (world, instance), its material's optics rows (`pipeline/upload/optics.py`, (1 or nworld,
  # max(ninstance, 1), OPTICS_ROWS)):
  # Beer-Lambert extinction and refraction index, read at transmissive layers; the rows a
  # surface's ShadingFlag.OPTICS bit selects (run-time data, no kernel key).
  instance_optics: wp.array2d[wp.vec4]
  material_specular_weight: wp.array2d[wp.float32]
  material_specular_models: wp.array1d[wp.int32]
  material_diffuse_roughness: wp.array2d[wp.float32]
  material_diffuse_source: wp.array1d[wp.vec2]


@wp.struct
class LayeredMaterials:
  """The carrier of the layered lobes (Features.layered: anisotropy, fuzz, the OpenPBR coat,
  detail normals), a superset of `RoughWeightedMaterials` whose EON / specular-weight arrays are
  placeholders unless their own Features flags are on (their accessors are compiled out)."""

  mesh_adr: wp.array1d[MeshAdr]
  mesh_vert: wp.array1d[wp.vec3]
  mesh_normal: wp.array1d[wp.vec3]
  mesh_face: wp.array1d[wp.vec3i]
  material_texture: wp.array3d[wp.int32]
  material_tex_affine: wp.array3d[tex_affine]
  material_tex_variation: wp.array2d[wp.vec4]
  material_tex_uniform: wp.array2d[bool]
  # The run-time texture taps (`Toggles.anisotropy`; read by a build reading them at run time).
  texture_taps: int
  # The build word's copy (`kernels/abi/toggles.py`, `tiers.word`): every bound bit but the
  # geometry's (`GEO_*`), which an OptiX module binds (`trace/optix/device/params.bound_words`);
  # unread without bound bits. The first half (the second: `GeometryFields.bits2`, copied into
  # `Surface.bits2`).
  bits: wp.int32
  material_tex_repeat: wp.array2d[wp.vec2]
  material_uv_world: wp.array2d[uv_xform]
  material_emission: wp.array2d[wp.float32]
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_facenormal: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  material_base_metalness: wp.array2d[wp.float32]
  material_specular_roughness: wp.array2d[wp.float32]
  material_reflectance: wp.array2d[wp.float32]
  instance_shading: wp.array2d[wp.int32]
  instance_emitter_light: wp.array1d[wp.int32]
  material_mirror: wp.array2d[wp.float32]
  textures: wp.array1d[wp.Texture2D]
  tex_size: wp.array1d[wp.vec2]
  tex_layout: wp.array1d[wp.int32]
  mesh_facecolor: wp.array1d[wp.vec3i]
  mesh_color: wp.array1d[wp.vec4]
  material_alpha_cutoff: wp.array2d[wp.float32]
  material_uv_mapping: wp.array1d[MaterialUvMapping]
  mesh_uv_second: wp.array1d[wp.vec2i]
  # Per (world, instance), its material's optics rows (`pipeline/upload/optics.py`, (1 or nworld,
  # max(ninstance, 1), OPTICS_ROWS)):
  # Beer-Lambert extinction and refraction index, read at transmissive layers; the rows a
  # surface's ShadingFlag.OPTICS bit selects (run-time data, no kernel key).
  instance_optics: wp.array2d[wp.vec4]
  material_specular_weight: wp.array2d[wp.float32]
  material_specular_models: wp.array1d[wp.int32]
  material_diffuse_roughness: wp.array2d[wp.float32]
  material_diffuse_source: wp.array1d[wp.vec2]
  # (world, material) OpenPBR anisotropy a in [-1, 1] (a < 0: the stretch turned a quarter turn,
  # Filament's signed convention) and the stretch direction's rotation [rad] from dp/du.
  material_specular_anisotropy: wp.array2d[wp.vec2]
  # (world, material) fuzz (weight F, Charlie roughness α) and its color.
  material_fuzz: wp.array2d[wp.vec2]
  material_fuzz_color: wp.array2d[wp.vec3]
  # (world, material) OpenPBR coat color (transmittance² at normal incidence) and coat IOR.
  material_coat_color: wp.array2d[wp.vec3]
  material_coat_ior: wp.array2d[wp.float32]
  # Static per-material layer constants (`LayerConstants`).
  material_layer: wp.array1d[LayerConstants]
  # The fuzz lobe's directional albedo table (`ibl.cloth_texture`).
  cloth_lut: wp.Texture2D
  # Per-vertex UV tangents (xyz, w: bitangent handedness), addressed like `mesh_normal`
  # (`MeshAdr.normal`, `mesh_facenormal`); empty without Features.specular_anisotropy.
  mesh_tangent: wp.array1d[wp.vec4]


# A closed set: every callable receives the carrier selected by the same Features.
type MaterialTables = Materials | RoughDiffuseMaterials | WeightedMaterials | RoughWeightedMaterials | LayeredMaterials


@specialize
def material_type(*, diffuse_roughness: bool, specular_weight: bool = False, layered: bool = False) -> type[MaterialTables]:
  """Concrete material layout of the compiled shading feature set (`layered`: Features.layered,
  the one carrier holding every optional array)."""
  if layered:
    return LayeredMaterials
  if specular_weight:
    return RoughWeightedMaterials if diffuse_roughness else WeightedMaterials
  return RoughDiffuseMaterials if diffuse_roughness else Materials


# diffuse_parameters(mats, worldid, material_id) -> (EON roughness, base weight); the carrier is
# the one `material_type` selects for the same Features.
type DiffuseParameters = Callable[[MaterialTables, int, int], wp.vec2]


@wp.func(inline=True)
@device_generic
def _diffuse_parameters[M: RoughDiffuseMaterials | RoughWeightedMaterials | LayeredMaterials](
  # Model:
  mats: M,
  # In:
  worldid: int,
  material_id: int,
) -> wp.vec2:
  """(EON roughness, base weight) of material `material_id`: (0, 1), the Lambert lobe, without a
  rough-diffuse table, an eligible substrate or a valid roughness."""
  if material_id < 0 or mats.material_diffuse_roughness.shape[1] == 0:
    return wp.vec2(0.0, 1.0)
  source = mats.material_diffuse_source[material_id]
  if source[1] == 0.0 or source[0] == 0.0:
    return wp.vec2(0.0, 1.0)
  roughness = material_value(mats.material_diffuse_roughness, worldid, material_id, 0.0)
  if not wp.isfinite(roughness) or roughness < 0.0 or roughness > 1.0:
    roughness = 0.0  # Lambert; the asynchronous range validator reports the invalid input.
  return wp.vec2(roughness, source[0])


@wp.func(inline=True)
@device_generic
def _no_diffuse_parameters[M: MaterialTables](
  # Model:
  _mats: M,
  # In:
  _worldid: int,
  _material_id: int,
) -> wp.vec2:
  """(0, 1): the Lambert lobe (builds without Features.diffuse_roughness)."""
  return wp.vec2(0.0, 1.0)


@specialize
def diffuse_parameters(*, enabled: bool) -> DiffuseParameters:
  """The EON parameters' accessor of Features.diffuse_roughness `enabled` (its carrier, from
  `material_type`, holds the tables), else the Lambert lobe's constant."""
  if enabled:
    # Same compile-time condition as material_type and the launch-param factory.
    return cast(DiffuseParameters, _diffuse_parameters)
  return _no_diffuse_parameters


# specular_weight(mats, worldid, mat_id) -> (weight, ShadingFlag model bits).
type SpecularWeight = Callable[[MaterialTables, int, int], tuple[float, int]]


@wp.func(inline=True)
@device_generic
def _specular_weight[M: WeightedMaterials | RoughWeightedMaterials | LayeredMaterials](
  # Model:
  mats: M,
  # In:
  worldid: int,
  mat_id: int,
) -> tuple[float, int]:
  """(canonical specular weight, ShadingFlag model bits) of material `mat_id` (weight 1 without a
  material; an invalid weight reads 0)."""
  flags = wp.int32(0)
  if mat_id >= 0 and mat_id < mats.material_specular_models.shape[0]:
    flags = mats.material_specular_models[mat_id]
  value = material_value(mats.material_specular_weight, worldid, mat_id, 1.0)
  if not wp.isfinite(value) or value < 0.0:
    return 0.0, flags  # The asynchronous range validator reports the invalid input.
  return value, flags


@wp.func(inline=True)
@device_generic
def _unit_specular_weight[M: MaterialTables](
  # Model:
  _mats: M,
  # In:
  _worldid: int,
  _mat_id: int,
) -> tuple[float, int]:
  """(1, no model bits) (builds without Features.specular_weight)."""
  return 1.0, 0


@specialize
def specular_weight_parameter(*, enabled: bool) -> SpecularWeight:
  """The specular weight's accessor of Features.specular_weight `enabled` (its carrier, from
  `material_type`, holds the tables), else the unit weight."""
  if enabled:
    return cast(SpecularWeight, _specular_weight)
  return _unit_specular_weight


@wp.struct
class Lobes:
  """BRDF inputs of a surface seen at NoV (Filament's split-sum terms).

  Attributes:
    perceptual_roughness: clamped perceptual roughness.
    alpha: GGX roughness (perceptual_roughness²).
    f0: specular reflectance at normal incidence.
    f90: specular reflectance at grazing incidence.
    energy_comp: multiscatter energy compensation of the specular lobe.
    spec_albedo: directional albedo of the specular lobe.
    diffuse_color: diffuse albedo weighted by (1 - metallic)(1 - spec_albedo): Lambert's, or EON's
      directional albedo at NoV for a rough diffuse lobe (`diffuse_roughness` > 0).
    coat: clearcoat weight (0: none; Features.clearcoat).
    coat_alpha: the coat's GGX roughness.
    coat_roughness: the coat's clamped perceptual roughness.
    coat_albedo: coat weight x the coat lobe's directional albedo at NoV: the base's ambient
      light is scaled by 1 - coat_albedo.
    diffuse_roughness: the EON lobe's roughness (0: Lambert's; Features.diffuse_roughness).
    diffuse_single, diffuse_multi: its view factors (`openpbr_lobes.eon_split`), weighted.
    bits, bits2: the build word's copy (`Surface`'s), read by the bound gates of the lobes' users.
  """

  bits: wp.int32
  bits2: wp.int32
  perceptual_roughness: wp.float32
  alpha: wp.float32
  f0: wp.vec3
  f90: wp.vec3
  energy_comp: wp.vec3
  spec_albedo: wp.vec3
  diffuse_color: wp.vec3
  coat: wp.float32
  coat_alpha: wp.float32
  coat_roughness: wp.float32
  coat_albedo: wp.float32
  diffuse_roughness: float
  diffuse_single: wp.vec3
  diffuse_multi: wp.vec3
  # Anisotropic GGX (Features.specular_anisotropy): a in [0, 1] (0: isotropic), (α_t, α_b), the
  # world frame (tangent, bitangent) the lobe is stretched in.
  anisotropy: float
  alpha_t: float
  alpha_b: float
  tangent: wp.vec3
  bitangent: wp.vec3
  # Fuzz (Features.fuzz): weight F, Charlie α, color, albedo E = min(E_fuzz(NoV), 1), the direct
  # lobe's normalisation min(1, 1 / E_fuzz(NoV)) and the layers below's scale 1 - F E.
  fuzz_weight: float
  fuzz_alpha: float
  fuzz_color: wp.vec3
  fuzz_albedo: float
  fuzz_norm: float
  under_fuzz: float
  # Cloth wrapped diffuse width (0: Lambert) and tint.
  wrap: float
  wrap_tint: wp.vec3
  # Coat F0 (COAT_F0 without Features.coat_layer), its law, color (T² at normal incidence),
  # IOR, and the base's darkening lerp(1, Δ, C d) (vec3(1): none).
  coat_f0: float
  coat_openpbr: bool
  coat_color: wp.vec3
  coat_ior: float
  coat_darkening: wp.vec3
  # The coat's shading normal and its NoV (Features.coat_normal; else the base's).
  coat_normal: wp.vec3
  coat_nov: float


# shading_lobes(dfg_lut, surface, nov) -> Lobes (`make_shading_lobes`).
type ShadingLobes = Callable[[wp.Texture2D, Surface, float], Lobes]


@specialize
def make_shading_lobes(  # noqa: C901, PLR0915 (one static branch and class gate per lobe / role)
  *,
  specular_curve: bool,
  clearcoat: bool,
  specular_layers: bool,
  diffuse_roughness: bool = False,
  specular_weight: bool = False,
  specular_anisotropy: bool = False,
  fuzz: bool = False,
  coat_layer: bool = False,
  coat_normal: bool = False,
  unlit_emission: bool = False,
  bound: int = 0,
):
  """shading_lobes(dfg_lut, surface, nov) -> Lobes.

  `specular_curve` reads the surface's F90 (Features.specular_curve), else F90 = 1; `clearcoat`
  the coat (Features.clearcoat), else no coat; `specular_layers` (Features.transmissive): a
  surface flagged `specular_only` (a translucent layer of a transmissive material) keeps its
  Fresnel-weighted specular, its diffuse lobe (direct, ambient, GI) weighted by its opacity
  1 - t, and the transmission to the next layer. `specular_weight` scales the metal coefficients
  after computing the existing compensation, and removes the index-matched dielectric; native
  authored models retain their coefficients. Mixed-metal compensation and nonzero dielectric
  Fresnel remain approximate.
  The layered lobes (`specular_anisotropy`, `fuzz`, `coat_layer`, `coat_normal`: their Features
  flags) are `_anisotropic_lobe`, `_fuzz_lobe` and `_coat_lobe`'s. `unlit_emission` (its
  Features flag): an unlit surface (`Surface.unlit`) has no specular lobe (F0 = F90 = 0) and a
  black base, so no light reaches it. `bound` (Features.bound): the lobes of the bound classes and
  the specular curve read the build word (`Surface.bits`).
  """
  gate = toggles.bound_on(bound)
  field = toggles.bound_on(bound, high=True)  # the specular weight's field bit (`toggles.MAT_SPECULAR_WEIGHT`)

  @wp.func
  def shading_lobes(  # noqa: C901, PLR0915 (one static branch and class gate per lobe / role)
    # Model:
    dfg_lut: wp.Texture2D,
    # In:
    surface: Surface,
    nov: wp.float32,
  ) -> Lobes:
    """The BRDF lobes of `surface` seen at NoV `nov` (split-sum DFG terms from `dfg_lut`)."""
    lobes = Lobes()
    lobes.bits = surface.bits
    lobes.bits2 = surface.bits2
    materials_on = False
    if wp.static(specular_weight or diffuse_roughness):
      materials_on = gate(surface.bits, wp.static(toggles.CLASS_MATERIALS))
    layered_on = False
    if wp.static(specular_anisotropy or fuzz or coat_layer or coat_normal):
      layered_on = gate(surface.bits, wp.static(toggles.CLASS_LAYERS))
    pr, alpha = brdf.roughness_to_alpha(surface.perceptual_roughness)
    lobes.perceptual_roughness = pr
    lobes.alpha = alpha
    lobes.f0 = brdf.surface_f0(surface.base_color, surface.metallic, surface.reflectance)
    dfg = ibl.sample_dfg(dfg_lut, nov, pr)
    lobes.energy_comp = brdf.energy_compensation(lobes.f0, dfg)
    curve = False
    if wp.static(specular_curve):
      curve = gate(surface.bits, wp.static(toggles.SCENE_SPECULAR_CURVE))
    if curve:
      lobes.f90 = surface.f90
      lobes.spec_albedo = brdf.specular_albedo_rgb(lobes.f0, lobes.f90, dfg)
    else:
      lobes.f90 = wp.vec3(1.0)
      lobes.spec_albedo = brdf.specular_albedo(lobes.f0, dfg)
    if wp.static(specular_weight):
      # Off in the word, the weight was never read (`evaluate.eval_surface`): the scene build's
      # unweighted lobes.
      weighted = materials_on and field(surface.bits2, wp.static(toggles.MAT_SPECULAR_WEIGHT))
      if weighted and not surface.preserve_specular_coefficients:
        weighted_f0, weighted_f90 = brdf.weighted_fresnel(
          surface.base_color, surface.metallic, surface.reflectance, surface.specular_weight
        )
        lobes.f0 = weighted_f0
        lobes.f90 = weighted_f90
        lobes.spec_albedo = brdf.specular_albedo_rgb(lobes.f0, lobes.f90, dfg)
    if wp.static(unlit_emission):
      if surface.unlit:
        lobes.f0 = wp.vec3(0.0)
        lobes.f90 = wp.vec3(0.0)
        lobes.spec_albedo = wp.vec3(0.0)
    # Energy-conserving coupling: the diffuse lobe gets what the dielectric's compensated
    # (multiscatter) specular leaves, E_ss x the compensation (a white non-absorbing dielectric's
    # furnace albedo is exactly 1); a partly metallic surface blends its two specular albedos
    # (`_metal_mix`).
    reflected, comp = _metal_mix(surface, lobes, dfg)
    lobes.energy_comp = comp
    lobes.diffuse_color = brdf.coupled_diffuse(surface.base_color, surface.metallic, reflected)
    if wp.static(diffuse_roughness):
      if materials_on and surface.diffuse_roughness > 0.0 and surface.metallic < 1.0:
        lobes.diffuse_roughness = surface.diffuse_roughness
        # EON's substrate albedo is physical [0,1], including texture/instance modulation.
        rho = surface.base_color / surface.diffuse_base_weight
        rho = wp.vec3(wp.clamp(rho[0], 0.0, 1.0), wp.clamp(rho[1], 0.0, 1.0), wp.clamp(rho[2], 0.0, 1.0))
        weight = (wp.vec3(1.0) - reflected) * ((1.0 - surface.metallic) * surface.diffuse_base_weight)
        # The diffuse albedo every integrated light (image-based, GI probes, LTC area lights)
        # scales: EON's exact directional albedo at NoV (exact under uniform radiance; the
        # analytic albedo ratio over Lambert's integral elsewhere). Punctual lights evaluate the
        # lobe itself from its view factors (`openpbr_lobes.eon_eval_split`: per light only what
        # depends on it).
        lobes.diffuse_color = wp.cw_mul(openpbr_lobes.eon_albedo(rho, lobes.diffuse_roughness, nov), weight)
        single, multi = openpbr_lobes.eon_split(rho, lobes.diffuse_roughness, nov)
        lobes.diffuse_single = wp.cw_mul(single, weight)
        lobes.diffuse_multi = wp.cw_mul(multi, weight)
    if wp.static(specular_layers):
      if surface.specular_only:
        # OpenPBR's base mixes diffuse and transmission by t; the layer's opacity is 1 - t (the
        # transmitted share passes to the next layer: `layer.composite_transmissive`).
        lobes.diffuse_color = lobes.diffuse_color * surface.alpha
        if wp.static(diffuse_roughness):  # zero without the materials class
          lobes.diffuse_single = lobes.diffuse_single * surface.alpha
          lobes.diffuse_multi = lobes.diffuse_multi * surface.alpha
    if wp.static(specular_anisotropy):
      if layered_on:
        lobes = _anisotropic_lobe(surface, lobes)
    if wp.static(fuzz):
      if layered_on:
        lobes = _fuzz_lobe(surface, lobes)
    coated = False
    if wp.static(clearcoat):
      coated = gate(surface.bits, wp.static(toggles.CLASS_COAT))
    if coated:
      if surface.clearcoat > 0.0:
        openpbr = False
        if wp.static(coat_layer or coat_normal):
          openpbr = layered_on
        if openpbr:
          lobes = _coat_lobe(dfg_lut, surface, nov, lobes)
        else:
          pr_c, alpha_c = brdf.roughness_to_alpha(surface.clearcoat_roughness)
          dfg_c = ibl.sample_dfg(dfg_lut, nov, pr_c)
          lobes.coat = surface.clearcoat
          lobes.coat_alpha = alpha_c
          lobes.coat_roughness = pr_c
          lobes.coat_albedo = surface.clearcoat * (dfg_c[0] * wp.static(1.0 - COAT_F0) + dfg_c[1] * wp.static(COAT_F0))
    return lobes

  return shading_lobes


@wp.func
def _metal_mix(
  # In:
  surface: Surface,
  lobes: Lobes,
  dfg: wp.vec2,
) -> tuple[wp.vec3, wp.vec3]:
  """(the dielectric lobe's compensated albedo E_d ec_d: what the diffuse lobe is coupled to, the
  specular lobe's compensation) of a surface of metalness m. A metal / dielectric mix is two lobes:
  its specular albedo is (1 - m) E_d ec_d + m E_m ec_m, its diffuse (1 - m)(1 - E_d ec_d). One
  GGX lobe of the mixed F0 has the albedo E(F0) (linear in F0) but a compensation ec(F0) and a
  coupling 1 - E(F0) ec(F0) that are not: a white half-metal kept 69-76 % of a white furnace.
  The mixed lobe keeps its shape and is compensated to the blended albedo; m = 0 and m = 1 are the
  single lobes, unchanged."""
  m = surface.metallic
  f0_d = wp.vec3(wp.static(brdf.DIELECTRIC_F0_SCALE) * surface.reflectance * surface.reflectance)
  reflected = wp.cw_mul(brdf.specular_albedo_rgb(f0_d, lobes.f90, dfg), brdf.energy_compensation(f0_d, dfg))
  comp = lobes.energy_comp
  if m > 0.0 and m < 1.0:
    # The metal lobe's F0: the mixed F0 less the dielectric's share (any specular weight kept).
    f0_m = (lobes.f0 - f0_d * (1.0 - m)) / m
    e_m = wp.cw_mul(brdf.specular_albedo_rgb(f0_m, wp.vec3(1.0), dfg), brdf.energy_compensation(f0_m, dfg))
    target = reflected * (1.0 - m) + e_m * m
    albedo = lobes.spec_albedo
    for c in range(3):
      if albedo[c] > 1.0e-6:
        comp[c] = target[c] / albedo[c]
  return reflected, comp


def make_relative_emission(f: Features):
  """relative_emission(relative, surface) -> the emission mode of `surface`
  (`tables.emission_radiance`): `relative` (`Params.relative_emission`), display-relative for an
  unlit material or a display-relative emission (`Surface.display_emission`,
  Features.unlit_emission)."""
  unlit = f.unlit_emission

  @wp.func(inline=True)
  def relative_emission(
    # In:
    relative: int,
    surface: Surface,
  ) -> int:
    """1: display-relative emission, 0: radiance."""
    if wp.static(unlit):
      if surface.display_emission:
        return 1
    return relative

  return relative_emission


def lobes_of(f: Features) -> ShadingLobes:
  """`make_shading_lobes` of every lobe Features `f` compiles in (the camera, reflection-hit and
  probe shading share one lobe model)."""
  return make_shading_lobes(
    specular_weight=f.specular_weight,
    diffuse_roughness=f.diffuse_roughness,
    specular_curve=f.specular_curve,
    clearcoat=f.clearcoat,
    specular_layers=f.transmissive,
    specular_anisotropy=f.specular_anisotropy,
    fuzz=f.fuzz,
    coat_layer=f.coat_layer,
    coat_normal=f.coat_normal,
    unlit_emission=f.unlit_emission,
    bound=f.bound,
  )


@wp.func
def bent_normal(
  # In:
  lobes: Lobes,
  normal: wp.vec3,
  view: wp.vec3,
) -> wp.vec3:
  """The normal the base specular's image-based lookups reflect about: the anisotropic bent
  normal where `Lobes.anisotropy` > 0, else `normal`."""
  if lobes.anisotropy > 0.0:
    return brdf.anisotropic_bent_normal(
      view, normal, lobes.tangent, lobes.bitangent, lobes.anisotropy, lobes.perceptual_roughness
    )
  return normal


@wp.func
def _anisotropic_lobe(
  # In:
  surface: Surface,
  lobes: Lobes,
) -> Lobes:
  """`lobes` with the anisotropic GGX roughness (α_t, α_b) of `surface.anisotropy` (OpenPBR's
  mapping, `brdf.anisotropic_alpha`) in the surface's tangent frame; isotropic at 0."""
  if surface.anisotropy > 0.0:
    ab = brdf.anisotropic_alpha(lobes.alpha, surface.anisotropy)
    lobes.anisotropy = surface.anisotropy
    lobes.alpha_t = ab[0]
    lobes.alpha_b = ab[1]
    lobes.tangent = surface.tangent
    lobes.bitangent = surface.bitangent
  return lobes


@wp.func
def _fuzz_lobe(
  # In:
  surface: Surface,
  lobes: Lobes,
) -> Lobes:
  """`lobes` with the fuzz layer (OpenPBR: F f_fuzz + (1 - F E_fuzz) f_base): the Charlie lobe's
  albedo E = min(E_fuzz, 1), the direct lobe's normalisation min(1, 1 / E_fuzz) (Charlie with
  Neubelt's visibility reflects more than it receives at grazing low roughness: its albedo is
  held at 1) and the scale 1 - F E of every layer below. Also the cloth wrap of the diffuse."""
  lobes.under_fuzz = 1.0
  lobes.fuzz_norm = 1.0
  if surface.fuzz_weight > 0.0:
    e = surface.fuzz_albedo
    lobes.fuzz_weight = surface.fuzz_weight
    lobes.fuzz_alpha = wp.max(surface.fuzz_alpha, wp.static(brdf.MIN_CHARLIE_ALPHA))
    lobes.fuzz_color = surface.fuzz_color
    lobes.fuzz_albedo = wp.min(e, 1.0)
    if e > 1.0:
      lobes.fuzz_norm = 1.0 / e
    lobes.under_fuzz = 1.0 - surface.fuzz_weight * lobes.fuzz_albedo
    lobes.wrap = surface.wrap
    lobes.wrap_tint = surface.wrap_tint
  return lobes


@wp.func
def coat_transmittance(
  # In:
  coat_color: wp.vec3,
  coat_ior: float,
  mu_i: float,
  mu_o: float,
) -> wp.vec3:
  """OpenPBR's coat absorption T^(1 / mu_i' + 1 / mu_o') with T² = `coat_color` (the
  transmittance at normal incidence), mu' the refracted cosines sqrt(1 - (1 - mu²) / η²)."""
  inv_eta2 = 1.0 / (coat_ior * coat_ior)
  ti = wp.sqrt(wp.max(1.0 - (1.0 - mu_i * mu_i) * inv_eta2, 1.0e-4))
  to = wp.sqrt(wp.max(1.0 - (1.0 - mu_o * mu_o) * inv_eta2, 1.0e-4))
  k = 0.5 * (1.0 / ti + 1.0 / to)
  return wp.vec3(
    wp.pow(wp.max(coat_color[0], 0.0), k), wp.pow(wp.max(coat_color[1], 0.0), k), wp.pow(wp.max(coat_color[2], 0.0), k)
  )


@wp.func
def _coat_lobe(
  # Model:
  dfg_lut: wp.Texture2D,
  # In:
  surface: Surface,
  nov: float,
  lobes: Lobes,
) -> Lobes:
  """`lobes` with the coat of its own IOR (Schlick F0((n - 1) / (n + 1))²), normal and, for an
  OpenPBR coat, its darkening of the base lerp(1, (1 - K) / (1 - E_b K), C d) (OpenPBR 1.1:
  K = lerp(F(mu_o), 1 - (1 - E_F(η)) / η², base roughness); E_b the base's albedo at NoV, its
  multiple scattering included)."""
  pr_c, alpha_c = brdf.roughness_to_alpha(surface.clearcoat_roughness)
  lobes.coat_normal = surface.coat_normal
  lobes.coat_nov = surface.coat_nov
  f0 = wp.static(COAT_F0)
  if surface.coat_openpbr:
    f0 = brdf.dielectric_f0(surface.coat_ior)
  dfg_c = ibl.sample_dfg(dfg_lut, surface.coat_nov, pr_c)
  lobes.coat = surface.clearcoat
  lobes.coat_alpha = alpha_c
  lobes.coat_roughness = pr_c
  lobes.coat_f0 = f0
  lobes.coat_albedo = surface.clearcoat * (dfg_c[0] * (1.0 - f0) + dfg_c[1] * f0)
  lobes.coat_openpbr = surface.coat_openpbr
  lobes.coat_color = surface.coat_color
  lobes.coat_ior = surface.coat_ior
  lobes.coat_darkening = wp.vec3(1.0)
  if surface.coat_openpbr and surface.coat_darkening > 0.0:
    x = 1.0 - wp.clamp(nov, 0.0, 1.0)
    x2 = x * x
    k_s = f0 + (1.0 - f0) * (x2 * x2 * x)
    eta = wp.max(surface.coat_ior, 1.0e-3)
    k_r = 1.0 - (1.0 - brdf.fresnel_average(eta)) / (eta * eta)
    k = k_s + (k_r - k_s) * lobes.perceptual_roughness
    # The base's albedo with its multiple scattering: a white non-absorbing base (E_b = 1) is
    # not darkened (with the single-scatter albedo a white rough metal under a clear coat lost up
    # to 55 % in a white furnace).
    e_b = wp.cw_mul(lobes.spec_albedo, lobes.energy_comp) + lobes.diffuse_color
    amount = surface.clearcoat * surface.coat_darkening
    for c in range(3):
      delta = (1.0 - k) / (1.0 - wp.min(e_b[c], 1.0) * k)
      lobes.coat_darkening[c] = 1.0 + (delta - 1.0) * amount
  return lobes


@wp.func
def coat_base_scale(
  # In:
  lobes: Lobes,
  nol: float,
) -> wp.vec3:
  """The factor of the layers under the coat for light arriving at NoL `nol` (albedo scaling, the
  same law for direct, area and image-based light): an OpenPBR coat lerp(1, T (1 - E_coat(NoV)), C)
  x its darkening, T its transmittance along both directions; a glTF coat 1 - C E_coat(NoV)."""
  if lobes.coat_openpbr:
    t = coat_transmittance(lobes.coat_color, lobes.coat_ior, nol, lobes.coat_nov)
    e = lobes.coat_albedo / lobes.coat
    return wp.cw_mul(wp.vec3(1.0 - lobes.coat) + t * (lobes.coat * (1.0 - e)), lobes.coat_darkening)
  return wp.vec3(1.0 - lobes.coat_albedo)


# layers(lobes, base, coat, coat_nol, fuzz_term) -> radiance (`make_layers`).
type Layers = Callable[[Lobes, wp.vec3, wp.vec3, float, wp.vec3], wp.vec3]


@specialize
def make_layers(*, clearcoat: bool, layered_coat: bool, fuzz: bool, bound: int):
  """layers(lobes, base, coat, coat_nol, fuzz_term) -> the radiance of the layer stack: `base` (the
  base lobes' radiance) under the coat plus the coat lobe's `coat`, under the fuzz layer plus the
  fuzz lobe's `fuzz_term` (OpenPBR's albedo-scaling layering, one law for punctual, area and
  image-based light: only the lobe integrals differ).

  The coat passes `coat_base_scale(lobes, coat_nol)` of the base (`layered_coat`: Features
  coat_layer / coat_normal, the OpenPBR law where `Lobes.coat_openpbr`), else the glTF law
  1 - C E_coat(NoV); the fuzz 1 - F E_fuzz. `clearcoat` / `fuzz`: their Features (each layer
  compiled in only with its Features flag). With `coat` and `fuzz_term` zero it is the attenuation
  the layers apply to anything under them. `bound` (Features.bound): the coat and layers classes
  read the build word (`Lobes.bits`).
  """
  gate = toggles.bound_on(bound)

  @wp.func
  def layers(
    # In:
    lobes: Lobes,
    base: wp.vec3,
    coat: wp.vec3,
    coat_nol: float,
    fuzz_term: wp.vec3,
  ) -> wp.vec3:
    """`base` under the coat and fuzz layers plus their lobes (`make_layers`)."""
    out = base
    layered_on = False
    if wp.static(layered_coat or fuzz):
      layered_on = gate(lobes.bits, wp.static(toggles.CLASS_LAYERS))
    coated = False
    if wp.static(clearcoat):
      coated = gate(lobes.bits, wp.static(toggles.CLASS_COAT))
    if coated:
      if lobes.coat > 0.0:
        openpbr = False
        if wp.static(layered_coat):
          openpbr = layered_on
        if openpbr:
          out = wp.cw_mul(out, coat_base_scale(lobes, coat_nol)) + coat
        else:
          out = out * (1.0 - lobes.coat_albedo) + coat
    if wp.static(fuzz):
      if layered_on and lobes.fuzz_weight > 0.0:
        out = out * lobes.under_fuzz + fuzz_term
    return out

  return layers


def layers_of(f: Features) -> Layers:
  """`make_layers` of the layers Features `f` compiles in."""
  return make_layers(clearcoat=f.clearcoat, layered_coat=f.coat_layer or f.coat_normal, fuzz=f.fuzz, bound=f.bound)
