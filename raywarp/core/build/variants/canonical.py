"""`canonical`: a resolved `Features` with the values its kernels cannot see collapsed, so that
equal generated code means one cached kernel (`resolve.resolve_features` returns canonical
Features)."""

from __future__ import annotations

import dataclasses
from typing import Final

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.features import RUNTIME_LIGHTS, Features, LayerFeatures, small_form


def canonical(f: Features) -> Features:
  """`f` with the values its kernels cannot see collapsed (one kernel per distinct codegen).

  The lighting fields of an unlit render (`_unlit`), shading fields of a render without shaded
  outputs (`_unshaded`; textures only for alpha cut-outs), the shadow-ray fields without shadows, a
  shadow budget covering every light shaded exactly (the light count, or the cap), the layer count
  without transparency or GI (camera rays read it from `Params`), the reflection-hit cap without
  reflection rays, the area-light shadow targets without area lights, refraction without
  transmissive layers, the EON roughness map without EON, the light grid without lights (nothing
  to cull, no visibility test to trace), the normal-map strength decode and the horizon
  fade without normal maps, the coat and layered-map fields without a coat, textures or a lobe
  reading them (`_canonical_layers`) and the mesh fields without meshes. Each rule reads the
  fields the earlier ones collapsed, so the result is canonical itself. Warp's module hash sees
  every remaining field (it reaches the kernels through `wp.static` or an argument type:
  tests/unit/test_canonical_features.py).
  """
  if f.unlit:
    f = _unlit(f)
  if not f.any_rgb:
    f = _unshaded(f)
  if not f.diffuse_roughness:
    f = dataclasses.replace(f, diffuse_roughness_map=False)
  if not f.shadows:
    f = dataclasses.replace(f, shadow_samples=1, budget=0)
  elif not 0 < f.budget < (f.light_cap or (f.nlight if f.nlight != RUNTIME_LIGHTS else f.budget + 1)):
    f = dataclasses.replace(f, budget=0)
  if small_form(f.light_forms):
    # The small form shades and traces every light: the budget and the cap select nothing.
    f = dataclasses.replace(f, budget=0, light_cap=0)
  f = _canonical_lit_surface(f)
  if not f.glossy:
    f = dataclasses.replace(f, hit_light_cap=0)
  if not f.nlight:
    f = dataclasses.replace(f, culling=False)
  if not f.normal_map:
    f = dataclasses.replace(f, scaled_normal_map=False)
  if not (f.normal_map or f.detail_normal):
    f = dataclasses.replace(f, horizon_fade=False)
  f = _canonical_layers(f)
  if not {int(ShapeType.MESH), int(ShapeType.SDF)} & set(f.shape_types):
    f = dataclasses.replace(f, mesh_frames=False, vertex_normals=False, vertex_colors=False)
  return f


def _canonical_lit_surface(f: Features) -> Features:
  """`canonical`'s refraction rules: transmission (a translucent layer of glass shaded
  specular-only) only with translucent layers (without them every surface is opaque, shaded
  whole); refraction only through transmissive layers (`pixel._TraceFlags.refraction`); solid
  glass primitives only with refraction."""
  if not f.transparency:
    f = dataclasses.replace(f, transmissive=False)
  if not (f.transparency and f.transmissive):
    f = dataclasses.replace(f, refraction=False)
  if not f.refraction:
    f = dataclasses.replace(f, glass_shapes=False)
  return f


# The layered lobes off (unlit and unshaded renders read none of them).
NO_LAYERS: Final[LayerFeatures] = {
  "specular_anisotropy": False,
  "fuzz": False,
  "coat_layer": False,
  "coat_normal": False,
  "coat_normal_map": False,
  "detail_normal": False,
  "layered_maps": False,
  "horizon_fade": False,
}


def _canonical_layers(f: Features) -> Features:
  """`canonical`'s layered-lobe rules: no coat features without a coat, no layered maps without
  textures or a lobe that reads them."""
  if not f.clearcoat:
    f = dataclasses.replace(f, coat_layer=False, coat_normal=False, coat_normal_map=False)
  if not f.textures:
    f = dataclasses.replace(f, coat_normal_map=False, detail_normal=False, layered_maps=False)
  if not (f.specular_anisotropy or f.fuzz):
    f = dataclasses.replace(f, layered_maps=False)
  return f


def _no_lights(f: Features) -> Features:
  """`f` without the light loop: no light, shadow ray, light cap, grid or headlight."""
  return dataclasses.replace(
    f,
    nlight=0,
    has_spot=False,
    ies=False,
    shadows=False,
    culling=False,
    light_cap=0,
    hit_light_cap=0,
    area_lights=False,
    headlight=False,
    visible_lights=False,
  )


def _unlit(f: Features) -> Features:
  """`f` without what an unlit render (surfaces.shading="unlit": the base color as is, plus the
  emission) never reads; the background, translucency, cut-outs, the base-color and emissive
  textures stay."""
  return dataclasses.replace(
    _no_lights(f),
    env_lit=False,
    normal_map=False,
    orm_map=False,
    specular_aa=False,
    vertex_normals=False,
    reflection=False,
    light_panels=False,
    gi=False,
    glossy=0,
    reflection_shadows=0,
    translucent_shadows=False,
    tinted_shadows=False,
    sun_disc=False,
    diffuse_roughness=False,
    clearcoat=False,
    specular_curve=False,
    specular_weight=False,
    unlit_emission=False,
    **NO_LAYERS,
  )


def _unshaded(f: Features) -> Features:
  """`f` without what a depth / segmentation-only render never reads (fixed values, so that such
  renders share one kernel whatever the shading settings); the textures stay for alpha cut-outs."""
  textures = f.alpha_cutout
  return dataclasses.replace(
    _no_lights(f),
    env_lit=False,
    env_background=False,
    textures=f.textures and textures,
    base_color_map=False,
    normal_map=False,
    orm_map=False,
    emissive_map=False,
    opacity_map=f.opacity_map and textures,
    diffuse_roughness_map=False,
    specular_weight=False,
    **NO_LAYERS,
    anisotropy=1,
    specular_aa=False,
    vertex_normals=False,
    reflection=False,
    transparency=False,
    light_panels=False,
    transmissive=False,
    refraction=False,
    glass_shapes=False,
    flex_materials=False,
    # The color's anti-aliasing: the geometry outputs come from the pixel center
    # (`kernels/geometry.py`); motion keeps the temporal state.
    spp=1,
  )
