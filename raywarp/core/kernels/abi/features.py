"""The list of compile-time choices one render kernel is built for (which effects, outputs and
scene contents it handles): the key under which compiled kernels are cached.

`Features`: everything a render kernel is specialised on (the process-global kernel cache key).
Built only by `core/build/variants/resolve.py` `resolve_features` (every field set there;
`light_forms` from the others, `bound` and `word` by the tier routing,
`core/build/variants/tiers.py`: 0 by default).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, TypedDict

from raywarp.core.kernels.abi import toggles

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.rays import Trees

# Features.nlight of a scene-free build (`core/build/variants/tiers.py` SCENE_FREE): the light
# loops' bound is the light table's count at run time (`Lights.nlight`), not a constant of the code.
RUNTIME_LIGHTS: Final = -1
# `Features.light_forms` values: 1..SMALL_LIGHTS_MAX the camera rays' light loop over a small table
# (every light exact and traced, that many straight-line entries, whatever the budget and the cap:
# ranking a table this small costs more than it saves), LIGHT_FORM_GENERAL the loop over the light
# source (grid, cap, budget).
SMALL_LIGHTS_MAX: Final = 4
LIGHT_FORM_GENERAL: Final = SMALL_LIGHTS_MAX + 1


# `Features.planes` values (`trace/accel/query.make_plane_queries`; the plane ids are run-time data,
# `GeometryFields.plane_ids`): no plane code (the scene's queries intersect no analytic plane), or
# a loop over the table. Compiled in unused, the plane code at every trace site cost the bundle
# two large plane-free interiors 15-24 % (RTX, fast). One form for every count: a
# few-plane form beside it would add a third of the bundle's builds.
PLANES_NONE: Final = 0
PLANES_SOME: Final = 1


def plane_form(nplane: int) -> int:
  """The `Features.planes` value of a scene whose queries intersect `nplane` analytic planes."""
  return PLANES_SOME if nplane > 0 else PLANES_NONE


def small_form(light_forms: int) -> int:
  """The entry count of light-loop form `light_forms` (`Features.light_forms`): 0 unless the small
  form."""
  return light_forms if 0 < light_forms <= SMALL_LIGHTS_MAX else 0


# Not slots: `_base/named.py` reads fields through `vars()` (`canonical.canonical`).
@dataclasses.dataclass(frozen=True, kw_only=True)
class Features:
  """Everything a render kernel is specialised on, resolved from the compile-time settings against
  the scene's capabilities, the cameras, the per-world declaration and the device."""

  shape_types: tuple[int, ...]
  traversal: str
  ordered_traversal: bool
  max_registers: int
  # Threads per block of the render kernels (launch and CUDA module option).
  block_dim: int
  backface_culling: bool
  # Samples per pixel side (aa.spp); on a supersampling backend (OptiX) 1 or 2: supersampled, the
  # factor run-time data (`trace/optix/supersample.factor`, `Lens.scale`).
  spp: int
  # A shaded output (`Outputs.shaded`): the shaded body, else the geometry-only one. Which outputs
  # and in which precision is run-time data, the output word (`kernels/abi/output_layout.py`).
  any_rgb: bool
  # surfaces.shading="unlit": the base color as is (kernels/surface.py shade_unlit).
  unlit: bool
  # Distance fog and haze of the camera rays (settings.Atmosphere; shade/environment/fog.py).
  fog: bool
  # The scene's light count (a constant loop bound), or RUNTIME_LIGHTS (a scene-free tier build).
  nlight: int
  has_spot: bool
  # IES photometric profiles (shade/lights/ies.py): some light has one (profiled area lights are
  # shaded in their point form).
  ies: bool
  shadows: bool
  shadow_samples: int
  budget: int
  culling: bool
  # Lights shaded exactly per point, the rest diffusely (lighting.max_exact; 0: all); at glossy
  # reflection hits (reflections.max_exact).
  light_cap: int
  hit_light_cap: int
  # The camera rays' light-loop form (0: no shadowed loop): the small form's entry count when the
  # scene's table fits it (no grid), else LIGHT_FORM_GENERAL (`tiers.LOOP_FORM`).
  light_forms: int = 0
  area_lights: bool
  headlight: bool
  env_lit: bool
  env_background: bool
  textures: bool
  # Texture taps along the footprint's major axis at grazing angles (1: isotropic;
  # surfaces.anisotropy).
  anisotropy: int
  base_color_map: bool
  normal_map: bool
  orm_map: bool
  emissive_map: bool
  opacity_map: bool
  diffuse_roughness_map: bool
  # A normal map has a per-axis strength / offset (`pipeline.upload.materials.scaled_normal_maps`):
  # Toksvig on the fetched texel, the texel map unclamped (`shade/materials/uv_maps.py`).
  scaled_normal_map: bool
  # Specular anti-aliasing and the meshes' interpolated vertex normals: always on (no setting),
  # off only where a build does not read them (`canonical`: unlit and geometry-only builds).
  specular_aa: bool
  vertex_normals: bool
  # Materials carry a MuJoCo mirror reflectance (`Materials.material_mirror`): their specular F0
  # (`evaluate.mirror_reflectance`), reflected by the glossy reflection rays like every lobe.
  reflection: bool
  transparency: bool
  mesh_frames: bool
  # Per-instance UV transforms and UV sets (GeometryFields.mesh_instance).
  uv_maps: bool
  live_material_uv: bool
  # A material projects its UVs from the world position (`Materials.material_uv_world`,
  # `uv_frame.world_uv_frame`).
  world_uv: bool
  # The motion output's history (core/post/temporal/state.py); off: no temporal state (`history`).
  motion: bool
  # Irradiance probe GI (core/shade/gi/).
  gi: bool
  # Traced glossy reflections (core/kernels/reflections.py); 0: off (the megakernel without them).
  glossy: int
  # Shadow rays at reflection hits: 0 none, 1 directional lights (settings.REFLECTION_SHADOWS).
  reflection_shadows: int
  # Shadow rays of directional lights pass translucent instances (instance_shadow_transmit); tinted:
  # their transmittance is colored (RGB visibilities).
  translucent_shadows: bool
  tinted_shadows: bool
  # Directional lights with an angular radius (`light.radius`): shadow rays sample their disc
  # (the area shadows' fixed pattern).
  sun_disc: bool
  # The EON rough diffuse lobe (OpenPBR base_diffuse_roughness).
  diffuse_roughness: bool
  # Clearcoat layer (instance.shading coat bits): a second GGX lobe over the base.
  clearcoat: bool
  # Alpha test of cut-out materials in every ray query (core/trace/accel/cutout.py).
  alpha_cutout: bool
  # Back-face culling skips the instances of `GpuScene.instance_twosided`
  # (GeometryFields.instance_twosided).
  two_sided: bool
  # Per-world instance optics (`core/pipeline/worlds/optics.py`): every reader of the shadow
  # transmittance, the optics rows and the two-sided / closed-pane bits takes world w's entry of
  # world-major tables (`trace.world_entry`); off, instance gi's of the one-row tables. On only
  # where a renderer declares `instance.material` or `material.specular_ior` per world in a scene
  # where that choice can change them (`world_optics.switched`).
  world_optics: bool
  # Material model (core ShadingFlag): Unreal specular curve. Reflection rays skip the emission of
  # emissive panels that are also (area) lights.
  specular_curve: bool
  # The material carrier holds the OpenPBR specular weight and the specular models
  # (`material.material_type`), for materials whose reflectance derives from them (`resolve.py`).
  specular_weight: bool
  light_panels: bool
  # Camera rays see the scene's area lights without geometry (`shade/lights/visible.py`;
  # compiled in only for scenes that have one: `Capabilities.visible_lights`).
  visible_lights: bool
  # Per-vertex colors multiply the base color (Materials.mesh_color; host scenes).
  vertex_colors: bool
  # Unlit materials (`ShadingFlag.UNLIT`): no light, base color plus emission display-relative
  # (`Surface.unlit`); display-relative emission on lit materials (`ShadingFlag.DISPLAY_EMISSION`,
  # glTF). The emission sites read `Surface.display_emission` beside `Params.relative_emission`.
  unlit_emission: bool
  # A deformable has a material (`Deformables.material`, `core/shade/materials/flex.py`): flex hits
  # read its color, metallic / roughness / specular / emission and (with UVs) its texture maps.
  flex_materials: bool
  # Shared scenery (core/pipeline/worlds/sharing.py): some rendered instances are shared by every
  # world, in their own BVH / IAS traced by every world; world_tree: the per-world BVH / IAS holds
  # some. compact_poses: the per-world pose arrays hold only the per-world instances (column
  # `GeometryFields.pose_slot[i]`; a shared instance reads the shared pose table at -1 - slot); off,
  # instance i is column i (MuJoCo's aliased d.geom_xpos) and a shared one reads world 0's row.
  # visibility: the per-world "instance.visible" table is read (shared and per-world instances,
  # `GeometryFields.visible_slot`).
  # planes: whether the analytic planes' code is compiled in (`plane_form`, `tiers.PLANE_FORM`):
  # the planes the backend intersects beside its structures (`TraceBackend.analytic_planes`: OptiX
  # every rendered plane; software the per-world planes with shared scenery, else they are in the
  # worlds' BVH).
  shared: bool
  world_tree: bool
  compact_poses: bool
  visibility: bool
  planes: int
  # Transmissive materials' translucent layers are shaded specular-only (ShadingFlag.TRANSMISSIVE).
  transmissive: bool
  # The solid-glass feature class (`tiers.CLASSES`; taken from the scene like the other classes'
  # fields): solid transmissive materials (an optics row: refraction index or Beer-Lambert
  # extinction, `pipeline/upload/optics.py`) bend and absorb camera rays in the layer loop
  # (`layer.transmit_step`, `layer.absorb_exit`). Its own class: compiled into the shared layer
  # loop, this code cost a feature-dense room (no solid glass) 28 % in fast, 512 worlds x
  # 512^2 on an RTX PRO 6000 (the OptiX raygen's continuation stack, 704 -> 736 B).
  refraction: bool
  # Solid glass on analytic primitives (spheres, boxes, capsules, ellipsoids, cylinders; the glass
  # class; an OptiX build's word bit, `toggles.GLASS_SHAPES`): an OptiX camera ray inside a
  # sphere, capsule or box leaves it through its analytic exit face (`Tracer.leave`: OptiX's
  # curves report no hit from inside, and one trace orders a box's exit face flush with another
  # surface arbitrarily), and past the layer budget inside one sees the ambient light ahead
  # (`pixel._make_through_ambient`), not the background (both backends; the software traversal
  # reports the exit crossings itself; its tier modules hold the bit as a literal).
  glass_shapes: bool
  # The layered lobes (`LayeredMaterials`, `shade/materials/layers.py`): anisotropic GGX, the fuzz
  # (Charlie) layer with the cloth wrap, the OpenPBR coat (own IOR, color, darkening, albedo-scaled
  # base), the coat's own normal (map, unmapped or the base's) and its map, the detail normal map
  # (RNM), and the layered texture roles (anisotropy strength, fuzz color / roughness).
  specular_anisotropy: bool
  fuzz: bool
  coat_layer: bool
  coat_normal: bool
  coat_normal_map: bool
  detail_normal: bool
  layered_maps: bool
  # Normal-mapped specular fades below the geometric horizon (Filament listing 39).
  horizon_fade: bool
  # Light-cost counters (performance.counters, kernels/abi/counters.py): a debug build.
  counters: bool
  # A tier variant reading the shadows' sampling at run time (`tiers.SAMPLED`,
  # `toggles.SOFT_SHADOWS`: the samples, the sun disc's pattern and the area-light targets the
  # launch's where they differ).
  run_sampling: bool = False
  # A tier variant reading the shadow budget (its K and remainder) and the light caps at run time
  # (`tiers.COUNTED`): `budget`, `light_cap` and `hit_light_cap` are the capacities
  # (`toggles.COUNTS_CAPACITY`), the launch's values (`Toggles`) select below them.
  run_counts: bool = False
  # A tier build (`core/build/variants/tiers.py`): the `Toggles.flags` bits of the effects it
  # compiles in and each launch switches (`toggles.on`); 0: every compiled effect is on.
  switches: int
  # The bound bits (`kernels/abi/toggles.py`): what the build compiles in and reads from its build
  # word, which an OptiX module binds (`trace/optix/device/params.bound_words`) and a software
  # megakernel module defines (`toggles.DEFINED`); 0: none.
  bound: int = 0
  # A software build's word (`tiers.word`): the bound classes its megakernel modules compile on
  # (`toggles.preamble`; the served class set, `tiers.served`). 0 on OptiX (a value per module).
  word: int = 0
  # A software build's output word (`output_layout.OutputLayout.word`), which its megakernel
  # modules define (`toggles.preamble`, `out_on`). 0 on OptiX (a value per module).
  out_word: int = 0
  # A tier build resolved without the scene (`core/build/variants/tiers.py` SCENE_FREE): its OptiX
  # programs are compiled for the geometry kinds of its shape types
  # (`trace/optix/device/programs.scene_free_flags`).
  scene_free: bool

  @property
  def trees(self) -> Trees:
    """(world_tree, shared, compact_poses, visibility, planes): what ray queries are built for
    (the plane ids are run-time data, `GeometryFields.plane_ids`; `planes` their loop form)."""
    return (self.world_tree, self.shared, self.compact_poses, self.visibility, self.planes)

  @property
  def layered(self) -> bool:
    """Whether the material carrier is `LayeredMaterials` (a layered lobe is compiled in)."""
    return self.specular_anisotropy or self.fuzz or self.coat_layer or self.coat_normal or self.detail_normal

  @property
  def history(self) -> bool:
    """Whether the frame needs the temporal inputs (frame index, history, previous poses)."""
    return self.motion


# The ray-query layout fields (`Features.trees`), the software traversal's order and the live
# material UV mappings (the software geometry struct's layout, `frame_structs.geometry`) an OptiX
# tier build resolves for every scene (`tiers.BOUND_FREE`: the deformables' order is the scene's).
_TREE_FIELDS: Final = (
  "world_tree",
  "shared",
  "compact_poses",
  "visibility",
  "planes",
  "ordered_traversal",
  "flex_materials",
  "live_material_uv",
)
# Their bits, and the deformables' (traced in every OptiX tier build: a flex hit is tested per hit
# in the software kernels).
_SCENE_BITS: Final = toggles.GEO_ALL | toggles.FLEX | toggles.FLEX_MATERIALS


def software_view(build: Features, scene: Features) -> Features:
  """Build `build` as the software kernels beside it are compiled (bakes, light grid, sky
  visibility, warm-up): with the ray-query layout of `scene` (Features specialised on the scene).
  An OptiX tier build compiles every structure, plane loop, pose indexing and visibility table
  (`tiers.BOUND_FREE`, folded per module); the software traversal has no bound values, so its
  helper kernels keep the scene's trees (a BVH it lacks is never traversed), traversal order,
  deformable materials and live material UV mappings, and read the other bits of the word at run
  time (`Lights.bits`, the
  material carrier's, `Environment.bits`)."""
  trees = {name: getattr(scene, name) for name in _TREE_FIELDS}
  return dataclasses.replace(build, **trees, bound=build.bound & ~_SCENE_BITS)


# The groups of `Features` fields core/build/variants/resolve.py resolves together (one function
# each).
class TuningFeatures(TypedDict):
  """`Features` fields of `Performance` (`_tuning`)."""

  block_dim: int
  max_registers: int
  ordered_traversal: bool


class LightFeatures(TypedDict):
  """`Features` fields of the light loop and the environment (`_lights`)."""

  nlight: int
  has_spot: bool
  ies: bool
  culling: bool
  light_cap: int
  hit_light_cap: int
  area_lights: bool
  headlight: bool
  env_lit: bool
  env_background: bool


class ShadowFeatures(TypedDict):
  """`Features` fields of the shadow rays (`_shadows`)."""

  shadows: bool
  shadow_samples: int
  budget: int
  reflection_shadows: int
  translucent_shadows: bool
  tinted_shadows: bool
  sun_disc: bool


class SurfaceFeatures(TypedDict):
  """`Features` fields of culling, normals, textures, translucency and cut-outs (`_surfaces`)."""

  backface_culling: bool
  two_sided: bool
  world_optics: bool
  vertex_normals: bool
  textures: bool
  anisotropy: int
  base_color_map: bool
  normal_map: bool
  orm_map: bool
  emissive_map: bool
  opacity_map: bool
  diffuse_roughness_map: bool
  transparency: bool
  alpha_cutout: bool


class LayerFeatures(TypedDict):
  """`Features` fields of the layered lobes (`_layers`)."""

  specular_anisotropy: bool
  fuzz: bool
  coat_layer: bool
  coat_normal: bool
  coat_normal_map: bool
  detail_normal: bool
  layered_maps: bool
  horizon_fade: bool


class MaterialFeatures(TypedDict):
  """`Features` fields of the material capabilities (`resolve._materials`)."""

  reflection: bool
  uv_maps: bool
  live_material_uv: bool
  world_uv: bool
  scaled_normal_map: bool
  diffuse_roughness: bool
  clearcoat: bool
  specular_weight: bool
  specular_curve: bool
  transmissive: bool
  refraction: bool
  glass_shapes: bool
  light_panels: bool
  vertex_colors: bool
  unlit_emission: bool
  flex_materials: bool
