"""Fills, on the host, the structs a render kernel reads for one frame (geometry, materials,
lights, environment) from the frame's arrays and the renderer's own tables.

The structs are filled from a `frame.Frame` (the kernel-field arrays of the call) and the
Renderer's `context.FrameContext` (its own tables, cameras, outputs, software structures). They
only store array descriptors (pointers, shapes): building them is a host-side copy of ~100
descriptors, cached by the software backend by the identity of the frame's arrays (`Frame.same`)
and the context's generation.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, final

import warp as wp

from raywarp._base.types import MaterialUvMapping
from raywarp.core.build.variants import tiers
from raywarp.core.kernels.abi import rays, toggles
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.shade.lights import ltc, punctual
from raywarp.core.shade.materials import material
from raywarp.core.trace.accel import packed

if TYPE_CHECKING:
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.shade.lights.grid import Grid
  from raywarp.core.trace.context import FrameContext


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Inputs:
  """Structs of one megakernel launch (cached per frame identity by the trace backends)."""

  geo: rays.SoftwareFields
  mats: material.MaterialTables
  lit: punctual.Lights
  cams: kernel_structs.Cameras
  env: kernel_structs.EnvironmentArrays
  out: kernel_structs.OutputArrays

  @property
  def structs(self) -> list[object]:
    """The leading kernel arguments (geo, mats, lit, cams, env)."""
    return [self.geo, self.mats, self.lit, self.cams, self.env]


def _common[G: rays.GeometryFields](
  geo: G, ctx: FrameContext, f: Frame, *, live_uv: bool = False, cut: material.MaterialTables | None = None
) -> G:
  """`geo` with the fields every backend reads (`rays.GeometryFields`; its build word copy
  `word`, the OptiX launch adds the structures' bits); `cut`: the alpha test's tables when they
  are a material carrier (else the frame's `cutout_struct`, of the layout `live_uv` selects:
  `geo`'s)."""
  scene = ctx.scene
  toggles.store(geo, word(ctx))
  geo.instance_type = scene.instance_type
  geo.instance_mesh = f.instance_mesh
  geo.instance_material = f.instance_material
  geo.instance_size = f.instance_size
  geo.instance_color = f.instance_color
  geo.material_base_color = f.material_base_color
  geo.color_rule = int(scene.color_rule)
  geo.rt = ctx.params.rt
  geo.instance_pos = f.instance_pos
  geo.instance_rot = f.instance_rot
  geo.instance_shadow_transmit = ctx.tables.shadow_transmit
  geo.instance_twosided = ctx.tables.twosided
  geo.mesh_instance = scene.mesh_instances.table
  geo.pose_slot = ctx.tables.pose_slot
  geo.shared_pos, geo.shared_rot = f.shared_poses(ctx.layout)
  geo.instance_visible = f.instance_visible
  geo.visible_slot = ctx.tables.visible_slot
  geo.nmesh = ctx.software.mesh_bvh_id.shape[0]
  geo.flex_uv = scene.deformables.corner_uv
  geo.cut = cutout_struct(ctx, f, live_uv=live_uv) if cut is None else cut
  geo.counters = ctx.tables.counters
  geo.ray_floor = ctx.tables.ray_floor
  return geo


def geometry(ctx: FrameContext, f: Frame) -> rays.SoftwareFields:
  """The software kernels' geometry struct of the frame (traversal fields from `ctx.software`)."""
  scene = ctx.scene
  flex = scene.deformables
  sw = ctx.software
  rot_trace = sw.rot_trace if sw.rot_trace is not None else f.instance_rot
  # The software kernels' layout: the scene's live UV mappings (an OptiX tier build compiles them
  # in for every scene, its software helper kernels the scene's: `features.software_view`).
  live = ctx.effective.live_material_uv
  geo = _common(rays.software_geometry_type(live_uv=live)(), ctx, f, live_uv=live)
  geo.plane_ids = ctx.tables.planes
  geo.flex_vertadr = flex.vertadr
  geo.flex_edge = flex.edge
  geo.flex_radius = flex.radius
  geo.flex_vert = f.flex_vert
  geo.instance_rot_trace = rot_trace
  if sw.bvh is not None:
    sw.bvh.fill_geometry(geo)
  else:
    geo.bvh_ninstance = len(scene.rendered)
    geo.bvh_nflexprim = 0
  meshes = scene.meshes
  if meshes.hfield_bvh_id is not None and meshes.hfield_nodes is not None:
    geo.hfield_bvh_id, geo.hfield_nodes = meshes.hfield_bvh_id, meshes.hfield_nodes
  else:
    geo.hfield_bvh_id = geo.hfield_nodes = ctx.tables.no_hfields  # empty: no height field
  geo.mesh_bvh_id = ctx.software.mesh_bvh_id
  geo.mesh_nodes = packed.packed_nodes(ctx.software.mesh_bvh_id)
  geo.flex_prim_flexid = flex.prim_flexid
  geo.flex_prim_edgeid = flex.prim_edgeid
  return geo


def optix_geometry(ctx: FrameContext, f: Frame, *, shaded: bool = False) -> rays.OptixFields:
  """The OptiX programs' geometry struct of the frame, without the build's handles
  (`OptixRenderer._fill`). A shaded build's alpha-test tables are its material carrier
  (`params.ParamsLayout.shaded`: the programs shade with `geo.cut`)."""
  features = ctx.features
  if not shaded:
    live = features.live_material_uv
    return _common(rays.optix_geometry_type(live_uv=live)(), ctx, f, live_uv=live)
  # The carrier's Warp struct as the launch parameters' (`params._fields`): `materials` builds it.
  carrier = material.material_type(
    diffuse_roughness=features.diffuse_roughness, specular_weight=features.specular_weight, layered=features.layered
  )
  geometry = rays.optix_geometry_type(live_uv=features.live_material_uv, carrier=carrier)
  return _common(geometry(), ctx, f, cut=materials(ctx, f))


def cutout_struct(ctx: FrameContext, f: Frame, *, live_uv: bool) -> rays.Cutout | rays.LiveUvCutout:
  """Alpha-test/UV-observation arrays and, in the live-UV layout (`live_uv`), the live-material
  UV tables (`_uv_mappings`: empty where the configuration has none)."""
  meshes = ctx.scene.meshes
  cut: rays.Cutout | rays.LiveUvCutout
  if live_uv:
    active = rays.LiveUvCutout()
    active.material_uv_mapping, active.mesh_uv_second = _uv_mappings(ctx)
    cut = active
  else:
    cut = rays.Cutout()
  cut.material_alpha_cutoff = alpha_cutoffs(ctx, f)
  cut.material_texture = f.material_texture
  cut.material_tex_repeat = f.material_tex_repeat
  cut.material_tex_uniform = f.material_tex_uniform
  cut.material_uv_world = f.material_uv_world
  cut.mesh_adr = meshes.adr
  cut.mesh_facetexcoord = meshes.facetexcoord
  cut.mesh_texcoord = meshes.texcoord
  cut.textures = ctx.scene.texture_array
  cut.tex_layout = ctx.scene.tex_layout
  return cut


def _carrier(ctx: FrameContext, f: Frame) -> material.MaterialTables:
  """The material struct of the build's material model, with that model's own tables set."""
  mats: material.MaterialTables
  if ctx.features.layered:
    mats = _layered(ctx, f)
  elif ctx.features.specular_weight:
    if ctx.tables.specular_models is None:
      raise RuntimeError("specular material models were not prepared")
    if ctx.features.diffuse_roughness:
      both = material.RoughWeightedMaterials()
      both.material_specular_weight = f.material_specular_weight
      both.material_specular_models = ctx.tables.specular_models
      both.material_diffuse_roughness = f.material_base_diffuse_roughness
      if ctx.tables.diffuse_source is None:
        raise RuntimeError("EON material source was not prepared")
      both.material_diffuse_source = ctx.tables.diffuse_source
      mats = both
    else:
      weighted = material.WeightedMaterials()
      weighted.material_specular_weight = f.material_specular_weight
      weighted.material_specular_models = ctx.tables.specular_models
      mats = weighted
  elif ctx.features.diffuse_roughness:
    active = material.RoughDiffuseMaterials()
    active.material_diffuse_roughness = f.material_base_diffuse_roughness
    if ctx.tables.diffuse_source is None:
      raise RuntimeError("EON material source was not prepared")
    active.material_diffuse_source = ctx.tables.diffuse_source
    mats = active
  else:
    mats = material.Materials()
  return mats


def materials(ctx: FrameContext, f: Frame) -> material.MaterialTables:
  """The material struct of the frame (mesh shading data, textures, per-world material tables)."""
  meshes = ctx.scene.meshes
  mats = _carrier(ctx, f)
  mats.bits = word(ctx) & toggles.HALF  # the first half (`GeometryFields.bits2` the second)
  mats.mesh_adr = meshes.adr
  mats.mesh_vert = meshes.vert
  mats.mesh_normal = meshes.normal
  mats.mesh_face = meshes.face
  mats.material_texture = f.material_texture
  mats.material_tex_affine = f.material_tex_affine
  mats.material_tex_variation = f.material_tex_variation
  mats.material_tex_uniform = f.material_tex_uniform
  mats.texture_taps = ctx.params.rt.anisotropy
  mats.material_tex_repeat = f.material_tex_repeat
  mats.material_uv_world = f.material_uv_world
  mats.material_emission = f.material_emission
  mats.mesh_facetexcoord = meshes.facetexcoord
  mats.mesh_facenormal = meshes.facenormal
  mats.mesh_texcoord = meshes.texcoord
  mats.instance_shading = f.instance_shading  # read only with specular_curve / clearcoat
  mats.material_base_metalness, mats.material_specular_roughness, mats.material_reflectance = (
    f.material_base_metalness,
    f.material_specular_roughness,
    f.material_reflectance,
  )
  mats.material_mirror = f.material_mirror
  mats.instance_emitter_light = ctx.tables.emitter_light
  mats.textures = ctx.scene.texture_array
  mats.tex_size = ctx.scene.tex_size
  mats.tex_layout = ctx.scene.tex_layout
  # Read only by kernels compiled with vertex_colors.
  mats.mesh_facecolor, mats.mesh_color = meshes.facecolor, meshes.color
  # The alpha test's tables (read through the carrier as an OptiX shaded build's `geo.cut`).
  mats.material_alpha_cutoff = alpha_cutoffs(ctx, f)
  mats.material_uv_mapping, mats.mesh_uv_second = _uv_mappings(ctx)
  mats.instance_optics = ctx.tables.optics
  return mats


def alpha_cutoffs(ctx: FrameContext, f: Frame) -> wp.array2d[float]:
  """The alpha test's cutoffs (1 or nworld, nmat): the frame's, or zeros (no material tested) where
  the build compiles the alpha test and the configuration has none (`Features.alpha_cutout` off:
  textures off on a cut-out scene, `tiers.differences`), so that the build renders as one without
  it."""
  if not ctx.features.alpha_cutout or ctx.effective.alpha_cutout:
    return f.material_alpha_cutoff
  return _no_cutoffs(str(wp.get_device(ctx.scene.device)), f.material_alpha_cutoff.shape[1])


@functools.cache
def _no_cutoffs(device: str, nmat: int) -> wp.array2d[float]:
  """(1, nmat) zero cutoffs on `device` (`alpha_cutoffs`)."""
  return wp.zeros((1, nmat), dtype=float, device=device)


def _uv_mappings(ctx: FrameContext) -> tuple[wp.array[MaterialUvMapping, [int]], wp.array[wp.vec2i, [int]]]:
  """(material UV mappings, second mesh UV sets) of the configuration's Features.live_material_uv,
  else empty (an OptiX tier build compiles them in, read where its word has them:
  `toggles.MAP_LIVE_UV`)."""
  if not ctx.effective.live_material_uv:
    return _no_uv_mappings(str(wp.get_device(ctx.scene.device)))
  tables = ctx.scene.material_uv
  if tables is None:
    raise RuntimeError("Live material UV mappings were not prepared")
  return tables.material, tables.mesh_second


@functools.cache
def _no_uv_mappings(device: str) -> tuple[wp.array[MaterialUvMapping, [int]], wp.array[wp.vec2i, [int]]]:
  """Empty UV mapping tables on `device` (unread without Features.live_material_uv)."""
  return wp.empty(0, dtype=MaterialUvMapping, device=device), wp.empty(0, dtype=wp.vec2i, device=device)


def _layered(ctx: FrameContext, f: Frame) -> material.LayeredMaterials:
  """The layered carrier's own arrays (`material.LayeredMaterials`): the layer tables, the static
  constants and tangents (`RendererTables.prepare_layers`), and the EON / specular-weight arrays
  when their Features are on (else placeholders their compiled-out accessors never read)."""
  tables = ctx.tables
  if tables.layers is None:
    raise RuntimeError("layered material tables were not prepared")
  mats = material.LayeredMaterials()
  mats.material_specular_anisotropy = f.material_specular_anisotropy
  mats.material_fuzz = f.material_fuzz
  mats.material_fuzz_color = f.material_fuzz_color
  mats.material_coat_color = f.material_coat_color
  mats.material_coat_ior = f.material_coat_eta
  mats.material_layer, mats.mesh_tangent, mats.cloth_lut = tables.layers
  mats.material_diffuse_roughness = f.material_base_diffuse_roughness
  mats.material_specular_weight = f.material_specular_weight
  mats.material_diffuse_source = tables.diffuse_source if tables.diffuse_source is not None else tables.no_diffuse_source
  mats.material_specular_models = tables.specular_models if tables.specular_models is not None else tables.no_models
  return mats


def word(ctx: FrameContext) -> int:
  """The frame's build word in its copies (`GeometryFields.bits`, `Lights.bits`, the material
  carrier's `bits`, `Environment.bits`): `tiers.word`; the OptiX geometry copy adds the structures'
  bits (`trace/optix/device/programs.build_word`)."""
  return tiers.word(ctx.features, ctx.effective)


def lights_struct(ctx: FrameContext, grid: Grid | None) -> punctual.Lights:
  """The light struct of the frame: the packed lights (`shade/lights/pack.py`, packed by the
  Renderer before any reader) over the light grid `grid` (None: one cell holding every light)."""
  lit = punctual.Lights()
  lit.lights, lit.hot = ctx.tables.light_pack.lights, ctx.tables.light_pack.hot
  lit.nlight = ctx.scene.nlight
  lit.falloff_floor = ctx.tables.falloff_floor
  lit.ltc_minv, lit.ltc_mag = ltc.ltc_textures(wp.get_device(ctx.scene.device))
  lit.ies = ctx.tables.light_pack.ies
  lit.visible = ctx.tables.light_pack.visible
  # `lighting.visible_lights` off: the camera rays test none (a run-time switch).
  lit.nvisible = ctx.tables.light_pack.nvisible if ctx.settings.lighting.visible_lights else 0
  toggles.store(lit, word(ctx))
  if grid is not None:
    light_grid.wire(lit, grid)
  else:
    lit.grid_origin, lit.grid_inv_cell, lit.grid_dims = wp.vec3(0.0), 1.0, wp.vec3i(1, 1, 1)
    lit.grid_start, lit.grid_lights = ctx.tables.all_lights
  return lit


def environment(ctx: FrameContext, f: Frame) -> kernel_structs.EnvironmentArrays:
  """The environment struct of the frame (exposure, ambient, maps, GI volume, sky grid, Params)."""
  maps = ctx.scene.environment.maps
  env = kernel_structs.EnvironmentArrays()
  toggles.store(env, word(ctx))
  env.exposure = f.exposure
  env.ambient_radiance = f.ambient
  env.background = ctx.background
  env.dfg_lut = ctx.tables.dfg
  env.env_radiance = maps.radiance
  env.env_prefiltered = maps.prefiltered
  env.env_sh = maps.sh
  env.env_scale = maps.scale
  env.env_id = f.env_id
  env.env_intensity = f.env_intensity
  env.env_yaw = f.env_yaw
  env.env_sun, env.env_sun_radiance = ctx.tables.sun_discs
  env.fog_color = f.fog_color
  env.fog_range = f.fog_range
  env.look = ctx.tables.grading.look
  env.view_exposure = ctx.view_exposure if ctx.view_exposure is not None else ctx.tables.no_view
  env.gi = ctx.gi.struct if ctx.gi is not None else ctx.tables.no_gi
  env.params = ctx.params
  return env


def probe_environment(ctx: FrameContext, f: Frame) -> kernel_structs.EnvironmentArrays:
  """The environment struct of the frame as the GI probe rays read it: `environment` with the
  renderer's exposure (`FrameContext.reference_exposure`) in place of the worlds': the probes'
  display-relative emission (`tables.emission_radiance`) is lit at it, so a per-world exposure is
  no GI input (it scales the image after the probes: a change re-lights nothing)."""
  env = environment(ctx, f)
  env.exposure = ctx.reference_exposure
  return env


def build(ctx: FrameContext, f: Frame) -> Inputs:
  """Every struct of the frame `f` (the software backend caches the result)."""
  return Inputs(
    geo=geometry(ctx, f),
    mats=materials(ctx, f),
    lit=lights_struct(ctx, ctx.light_grid),
    cams=ctx.camera_struct(f),
    env=environment(ctx, f),
    out=ctx.buffers.struct(ctx.output_tables()),
  )
