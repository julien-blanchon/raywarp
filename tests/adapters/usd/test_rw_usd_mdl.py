"""MDL through the MDL SDK: library and variant materials, distilled lobes, bakes and their
domains, the disk cache, the module search, the fallback without the SDK."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rw_helpers import bind, new_stage, quad, write_png
from rw_helpers import material_named as _material
from rw_helpers import mdl_material as _shader

pytest.importorskip("pxr")

from pxr import Sdf

import raywarp as rw
import raywarp.formats.usd
from raywarp.formats.usd.materials.mdl.baker import ERROR

S = rw.scene.Slot
F = Sdf.ValueTypeNames.Float


def _read(stage):
  return rw.formats.usd.read(stage)


def test_library_material_distills_to_its_constants(tmp_path):
  """OmniPBR from raywarp's library: authored constants, the level 0.5 as F0 0.04, the
  emission as exitance / pi; the library origin is reported."""
  stage = new_stage(tmp_path)
  inputs = {
    "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, (0.6, 0.2, 0.1)),
    "metallic_constant": (F, 0.25),
    "reflection_roughness_constant": (F, 0.4),
    "enable_emission": (Sdf.ValueTypeNames.Bool, True),
    "emissive_color": (Sdf.ValueTypeNames.Color3f, (1.0, 0.5, 0.25)),
    "emissive_intensity": (F, 100.0),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "OmniPBR.mdl", "OmniPBR", **inputs))
  builder = _read(stage)
  m = _material(builder, "/Looks/P")
  assert m.base_color == pytest.approx((0.6, 0.2, 0.1), abs=1e-6)
  assert m.base_metalness == pytest.approx(0.25) and m.specular_roughness == pytest.approx(0.4)
  assert m.specular_weight * 0.04 == pytest.approx(0.04, rel=1e-5) and m.specular_ior == 1.5
  assert m.ext.flags & rw.scene.MaterialFlags.DIELECTRIC_SPECULAR_LEVEL
  assert builder.report.approximated["mdl.specular_grazing(fixed at 1)"] == 1
  assert m.emission_color == pytest.approx((1.0, 0.5, 0.25)) and m.emission_luminance == pytest.approx(100.0 / np.pi)
  assert builder.report.counts["mdl.distilled"] == 1
  assert builder.report.approximated["mdl.library_module:OmniPBR.mdl(raywarp's copy)"] == 1


@pytest.mark.parametrize("metal", [False, True], ids=["diffuse-only", "independent-metal"])
def test_absent_dielectric_layer_does_not_remove_independent_metal(tmp_path: Path, metal: bool) -> None:
  """A missing dielectric layer has zero weight; an independent metal keeps its native model."""
  stage = new_stage(tmp_path)
  lobe = (
    "df::microfacet_ggx_vcavities_bsdf(roughness_u: 0.2, roughness_v: 0.2, tint: color(0.6, 0.3, 0.1))"
    if metal
    else "df::diffuse_reflection_bsdf(tint: color(0.6, 0.3, 0.1))"
  )
  (tmp_path / "Independent.mdl").write_text(
    f"mdl 1.6;\nimport ::df::*;\nexport material Independent() = material(surface: material_surface(scattering: {lobe}));\n"
  )
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "Independent.mdl", "Independent"))
  builder = _read(stage)
  material = _material(builder, "/Looks/P")
  assert material.specular_weight == 0.0
  assert material.base_metalness == float(metal)
  assert bool(material.ext.flags & rw.scene.MaterialFlags.DIELECTRIC_SPECULAR_LEVEL) == metal
  assert ("mdl.metal_fresnel(Schlick instead of constant tint)" in builder.report.approximated) == metal


def test_variant_and_called_material(tmp_path):
  """`V(*) = OmniPBR(...)` keeps its defaults under authored inputs; a material calling another
  module's material takes the call's arguments."""
  stage = new_stage(tmp_path)
  (tmp_path / "Red.mdl").write_text(
    "mdl 1.6;\nimport ::OmniPBR::*;\nexport material Red(*) = ::OmniPBR::OmniPBR(diffuse_color_constant: color(0.8, 0.1, 0.1), metallic_constant: 0.25);\n"
  )
  (tmp_path / "Wrap.mdl").write_text(
    "mdl 1.6;\nimport ::Red::*;\nexport material Wrap(float K = 0.5) = ::Red::Red(reflection_roughness_constant: K * 0.5);\n"
  )
  bind(quad(stage, "/A"), _shader(stage, "/Looks/R", "Red.mdl", "Red", reflection_roughness_constant=(F, 0.2)))
  bind(quad(stage, "/B"), _shader(stage, "/Looks/W", "Wrap.mdl", "Wrap", K=(F, 0.6)))
  builder = _read(stage)
  red, wrap = _material(builder, "/Looks/R"), _material(builder, "/Looks/W")
  assert red.base_color == pytest.approx((0.8, 0.1, 0.1), abs=1e-6) and red.specular_roughness == pytest.approx(0.2)
  assert wrap.base_color == pytest.approx((0.8, 0.1, 0.1), abs=1e-6) and wrap.specular_roughness == pytest.approx(0.3)


def test_textured_base_color_binds_the_file_through_its_tiling(tmp_path):
  """An OmniPBR diffuse texture is the file itself, read through texture_scale (no bake)."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.full((4, 4, 3), 200, np.uint8))
  inputs = {
    "diffuse_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("t.png")),
    "texture_scale": (Sdf.ValueTypeNames.Float2, (4.0, 2.0)),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/T", "OmniPBR.mdl", "OmniPBR", **inputs))
  builder = _read(stage)
  ref = _material(builder, "/Looks/T").texture(S.BASE_COLOR)
  assert ref is not None and ref.uv_xform == ((4.0, 0.0, 0.0), (0.0, 2.0, 0.0))
  assert "mdl.baked" not in builder.report.counts


def test_glass_and_coat_lobes(tmp_path):
  """OmniGlass: full transmission at its IOR; OmniPBR_ClearCoat: a glTF-style coat."""
  stage = new_stage(tmp_path)
  bind(quad(stage, "/G"), _shader(stage, "/Looks/G", "OmniGlass.mdl", "OmniGlass", glass_ior=(F, 1.2)))
  coat = {"enable_clearcoat": (Sdf.ValueTypeNames.Bool, True), "clearcoat_weight": (F, 0.5)}
  bind(quad(stage, "/C"), _shader(stage, "/Looks/C", "OmniPBR_ClearCoat.mdl", "OmniPBR_ClearCoat", **coat))
  builder = _read(stage)
  glass, clear = _material(builder, "/Looks/G"), _material(builder, "/Looks/C")
  assert glass.transmission_weight == 1.0 and glass.specular_ior == pytest.approx(1.2)
  assert clear.coat_weight > 0.0 and clear.ext.flags & rw.scene.MaterialFlags.COAT_GLTF


UV_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::tex::*;
export material {name}(uniform texture_2d T = texture_2d("./t.png", ::tex::{gamma}))
 = let {{
  float2 UV = float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y);
  float4 A = tex::lookup_float4(T, UV, tex::wrap_repeat, tex::wrap_repeat);
  float4 B = tex::lookup_float4(T, UV * 2.0, tex::wrap_repeat, tex::wrap_repeat);
  float R = {expr};
}} in material(surface: material_surface(scattering: df::custom_curve_layer(
  normal_reflectivity: 0.04, layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: R * R, roughness_v: R * R),
  base: df::diffuse_reflection_bsdf())));
"""


def test_tiled_lookups_bake_exactly_and_hit_the_disk_cache(tmp_path):
  """Two lookups of one texture at integer tilings bake exactly over the coarser map's window;
  the second read takes the bake from the disk cache."""
  stage = new_stage(tmp_path)
  img = np.zeros((4, 4), np.uint8)
  img[:, :2] = 255
  write_png(tmp_path / "t.png", img)
  (tmp_path / "Tiled.mdl").write_text(UV_BODY.format(name="Tiled", gamma="gamma_linear", expr="A.x * B.x"))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/T", "Tiled.mdl", "Tiled"))
  stage.Save()
  cache = tmp_path / "cache"
  scenes = [rw.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache).build() for _ in range(2)]
  assert len(list((cache / "mdl").glob("*.npy"))) == 1
  first, second = (s.images.level(int(s.textures.image[s.materials.tex[0, S.SPECULAR_ROUGHNESS]]))[..., 0] for s in scenes)
  np.testing.assert_array_equal(first, second)
  # Columns 0-1 white, 2-3 black: A x B is 1 on the first quarter, 0 on the rest (B tiles
  # twice), baked at the finer lookup's density (8 texels).
  assert first.shape[1] == 8
  np.testing.assert_array_equal(first[0, [1, 2, 3, 5, 6, 7]], [255, 0, 0, 0, 0, 0])


def test_one_file_in_two_color_spaces_stays_two_textures(tmp_path):
  """One body over one file read as linear and as sRGB binds two textures, one per space (the
  file is read directly: roughness alpha = texel^2)."""
  from raywarp.scene.images import colorspace

  stage = new_stage(tmp_path)
  img = np.full((4, 4), 128, np.uint8)
  img[:, 2:] = 0
  write_png(tmp_path / "t.png", img)
  for k, gamma in enumerate(("gamma_linear", "gamma_srgb")):
    (tmp_path / f"T{k}.mdl").write_text(UV_BODY.format(name=f"T{k}", gamma=gamma, expr="A.x"))
    bind(quad(stage, f"/Q{k}"), _shader(stage, f"/Looks/T{k}", f"T{k}.mdl", f"T{k}"))
  builder = _read(stage)
  spaces = [
    colorspace.from_code(builder._textures[_material(builder, "/Looks/" + f"T{k}").texture(S.SPECULAR_ROUGHNESS).texture][1])
    for k in (0, 1)
  ]
  assert spaces == [rw.scene.ColorSpace.LIN_REC709, rw.scene.ColorSpace.SRGB_REC709]


def test_a_shared_bake_is_one_image(tmp_path):
  """Two shaders sharing a bake get one image, cold and from the disk cache."""
  stage = new_stage(tmp_path)
  img = np.full((4, 4), 128, np.uint8)
  img[:, 2:] = 0
  write_png(tmp_path / "t.png", img)
  (tmp_path / "Tiled.mdl").write_text(UV_BODY.format(name="Tiled", gamma="gamma_linear", expr="A.x * B.x"))
  for k in range(2):  # an unknown input makes the two shaders two conversions
    bind(quad(stage, f"/Q{k}"), _shader(stage, f"/Looks/T{k}", "Tiled.mdl", "Tiled", Extra=(F, k)))
  stage.Save()
  for _ in ("cold", "warm"):
    builder = rw.formats.usd.read(tmp_path / "scene.usda", mdl_cache=tmp_path / "cache")
    assert builder.report.dropped["mdl.input:Extra"] == 2
    s = builder.build()
    assert s.nimage == 1 and s.ntexture == 1


def test_thresholded_opacity_is_a_cutout(tmp_path):
  """A thresholded opacity bakes to 0 / 1 texels: an alpha cut-out."""
  stage = new_stage(tmp_path)
  img = np.zeros((4, 4), np.uint8)
  img[:, 2:] = 255
  write_png(tmp_path / "m.png", img)
  (tmp_path / "Cut.mdl").write_text(
    "mdl 1.6;\nimport ::df::*;\nimport ::state::*;\nimport ::tex::*;\n"
    'export material Cut(uniform texture_2d M = texture_2d("./m.png", ::tex::gamma_linear)) = let {\n'
    "  float V = tex::lookup_float4(M, float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y)).x;\n"
    "} in material(surface: material_surface(scattering: df::diffuse_reflection_bsdf()),\n"
    "  geometry: material_geometry(cutout_opacity: V < 0.5 ? 0.0 : 1.0));\n"
  )
  bind(quad(stage, "/A"), _shader(stage, "/Looks/C", "Cut.mdl", "Cut"))
  m = _material(_read(stage), "/Looks/C")
  assert m.texture(S.GEOMETRY_OPACITY) is not None
  assert m.ext.alpha_mode == rw.scene.AlphaMode.MASK and m.ext.alpha_cutoff == pytest.approx(0.5)


# MI_Hospital_Floor's projection: world position [cm] / -TextureSize, three planar lookups blended
# by the normal (the floor's lookup at the baker's +Z normal).
WORLD_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material World(uniform texture_2d T = texture_2d("./t.png", ::tex::gamma_linear), float S = 250.0)
 = let {
  float3 P = (state::transform_point(state::coordinate_internal, state::coordinate_world, state::position())
    * state::meters_per_scene_unit() * 100.0) / (math::abs(S) * -1.0);
  float4 XZ = tex::lookup_float4(T, float2(P.x, 1.0 - P.z), tex::wrap_repeat, tex::wrap_repeat);
  float4 XY = tex::lookup_float4(T, float2(P.x, 1.0 - P.y), tex::wrap_repeat, tex::wrap_repeat);
  float R = math::lerp(XZ.x, XY.x, math::abs(state::normal().z));
} in material(surface: material_surface(scattering: df::custom_curve_layer(
  normal_reflectivity: 0.04, layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: R, roughness_v: R),
  base: df::diffuse_reflection_bsdf())));
"""


@pytest.mark.parametrize("mpu", [1.0, 0.01])
def test_world_space_lookup_becomes_a_world_projection(tmp_path, mpu):
  """A world-position lookup is baked in the map's own window and read as a UV_SET_WORLD map in
  scene metres, u = -100 x / S, v = 1 + 100 y / S, whatever the stage units."""
  stage = new_stage(tmp_path, mpu=mpu)
  write_png(tmp_path / "t.png", np.tile(np.array([[0, 255]], np.uint8), (2, 1)))
  (tmp_path / "World.mdl").write_text(WORLD_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/W", "World.mdl", "World", S=(F, 200.0)))
  ref = _material(_read(stage), "/Looks/W").texture(S.SPECULAR_ROUGHNESS)
  assert ref is not None and ref.uv_set == rw.scene.model.types.UV_SET_WORLD
  np.testing.assert_allclose(ref.uv_xform, ((-0.5, 0.0, 0.0), (0.0, 0.5, 1.0)), atol=1e-6)


# A road's world-projected detail (period 0.5 m) under a stain noise of an incommensurate period
# (1 / 0.3 m), as SimPBR_Road's asphalt sets and stains.
ROAD_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material Road(uniform texture_2d T = texture_2d("./t.png", ::tex::gamma_linear),
  uniform texture_2d N = texture_2d("./n.png", ::tex::gamma_linear))
 = let {
  float3 P = state::transform_point(state::coordinate_internal, state::coordinate_world, state::position())
    * state::meters_per_scene_unit();
  float fine = tex::lookup_float4(T, float2(P.x * 2.0, 1.0 - P.y * 2.0), tex::wrap_repeat, tex::wrap_repeat).x;
  float stain = tex::lookup_float4(N, float2(P.x * 0.3, 1.0 - P.y * 0.3), tex::wrap_repeat, tex::wrap_repeat).x;
} in material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(
  tint: color(math::lerp(fine, 0.5 * fine, stain)))));
"""


def test_incommensurate_world_lookups_bake_the_finest_world_window(tmp_path):
  """World lookups of incommensurate periods are baked over the finest one's window and read
  through the world projection (its detail at its own density; reported), not over the mesh's UV
  square with the position read as (u, v, 0) (the detail stretched over the whole mesh)."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.tile(np.array([[0, 255]], np.uint8), (2, 1)))
  write_png(tmp_path / "n.png", np.array([[0, 255], [255, 0]], np.uint8))
  (tmp_path / "Road.mdl").write_text(ROAD_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/R", "Road.mdl", "Road"))
  builder = _read(stage)
  ref = _material(builder, "/Looks/R").texture(S.BASE_COLOR)
  assert ref is not None and ref.uv_set == rw.scene.model.types.UV_SET_WORLD
  np.testing.assert_allclose(ref.uv_xform, ((2.0, 0.0, 0.0), (0.0, -2.0, 1.0)), atol=1e-6)
  assert any(k.startswith("mdl.baked_world_window.base_color") for k in builder.report.approximated)
  assert not any(k.startswith("mdl.baked_position") for k in builder.report.approximated)


def test_unresolved_module_and_unknown_material_are_errors(tmp_path):
  """A module the search paths do not have, or a material it does not export: ERROR, reported."""
  stage = new_stage(tmp_path)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/U", "Nowhere.mdl", "Nowhere"))
  bind(quad(stage, "/B"), _shader(stage, "/Looks/N", "OmniPBR.mdl", "NoSuchMaterial"))
  builder = _read(stage)
  assert _material(builder, "/Looks/U").base_color == ERROR.base_color
  assert any(k.startswith("mdl.load_failed:Nowhere") for k in builder.report.dropped)
  assert builder.report.dropped["mdl.unknown_material:NoSuchMaterial(error)"] == 1


def test_a_parameter_without_default_takes_its_zero_value(tmp_path):
  """BUG-746: SimPBR_RoadPaint's `TileOrigin` / `TileSize` have no default and no stage authors
  them: the SDK refused the call (the road markings were the magenta error material). Such a
  parameter takes its type's default-constructed value (as Omniverse RTX renders it), reported;
  the authored inputs keep their values."""
  stage = new_stage(tmp_path)
  (tmp_path / "Paint.mdl").write_text(
    "mdl 1.6;\nimport ::df::*;\n"
    "export material Paint(float2 TileSize, color tint = color(0.2), float k = 1.0)"
    " = material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(tint: tint * k * (TileSize.x + 1.0))));\n"
  )
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "Paint.mdl", "Paint", tint=(Sdf.ValueTypeNames.Color3f, (0.9, 0.8, 0.7))))
  builder = _read(stage)
  assert _material(builder, "/Looks/P").base_color == pytest.approx((0.9, 0.8, 0.7), abs=1e-6)
  assert builder.report.approximated["mdl.parameter_without_default:TileSize(default-constructed)"] == 1
  assert not any("error" in k for k in builder.report.dropped)


def test_stage_module_wins_over_the_library(tmp_path):
  """An `OmniPBR.mdl` next to the stage is the one loaded (not raywarp's copy)."""
  stage = new_stage(tmp_path)
  (tmp_path / "OmniPBR.mdl").write_text(
    "mdl 1.6;\nimport ::df::*;\nexport material OmniPBR() = material(surface: material_surface("
    "scattering: df::diffuse_reflection_bsdf(tint: color(0.1, 0.7, 0.2))));\n"
  )
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "OmniPBR.mdl", "OmniPBR"))
  stage.Save()
  builder = rw.formats.usd.read(tmp_path / "scene.usda")
  assert _material(builder, "/Looks/P").base_color == pytest.approx((0.1, 0.7, 0.2))
  assert not builder.report.approximated
  assert str(tmp_path / "OmniPBR.mdl") in builder.dependencies


EDGE_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material Edge(uniform texture_2d N = texture_2d("./n.png", ::tex::gamma_linear))
 = let {
  float2 UV = float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y);
  float2 XY = float2(tex::lookup_float4(N, UV).x, tex::lookup_float4(N, UV).y) * 2.0 - 1.0;
  float3 T = float3(XY.x, XY.y, math::sqrt(math::saturate(1.0 - math::dot(XY, XY))));
  float E = 100.0 * math::sin(UV.x * 3.0) + 50.0;
} in material(
  surface: material_surface(
    scattering: df::weighted_layer(weight: 1.0, layer: df::diffuse_reflection_bsdf(tint: color(8.0, -0.5, 0.5)),
      normal: state::texture_tangent_u(0) * T.x + state::texture_tangent_v(0) * T.y + state::normal() * T.z),
    emission: material_emission(emission: df::diffuse_edf(), intensity: color(E))));
"""


def test_edge_values_are_clamped_unpacked_normals_bound_and_emission_peaks_lift(tmp_path):
  """Out-of-range colors are clamped and reported (never a stage error); a normal map's z
  rebuilt from x, y binds the map; a baked emission is normalised, its peak in the luminance."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "n.png", np.full((2, 2, 3), (128, 128, 255), np.uint8))
  (tmp_path / "Edge.mdl").write_text(EDGE_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/E", "Edge.mdl", "Edge"))
  builder = _read(stage)
  m = _material(builder, "/Looks/E")
  assert m.base_color == pytest.approx((1.0, 0.0, 0.5))
  assert any(k.startswith("mdl.clamped.base_color") for k in builder.report.approximated)
  normal = m.texture(S.GEOMETRY_NORMAL)
  assert normal is not None and normal.channel == rw.scene.CHANNEL_RGB and builder.report.counts.get("mdl.baked", 0) == 1
  assert m.texture(S.EMISSION_COLOR) is not None and m.emission_luminance > 100.0 / np.pi


def test_a_webp_texture_input_is_baked(tmp_path):
  """A WebP input (a Pillow format the SDK's image plugins may not read: raywarp then decodes it
  for the SDK's bake) is baked with its texels."""
  from PIL import Image

  texels = np.zeros((4, 8, 4), np.uint8)
  texels[:, :4] = (200, 100, 50, 254)
  texels[:, 4:] = (10, 220, 130, 128)
  Image.fromarray(texels, "RGBA").save(tmp_path / "t.webp", lossless=True)
  stage = new_stage(tmp_path)
  (tmp_path / "Prod.mdl").write_text(UV_BODY.format(name="Prod", gamma="gamma_linear", expr="A.x * A.y"))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "Prod.mdl", "Prod", T=(Sdf.ValueTypeNames.Asset, Sdf.AssetPath("t.webp"))))
  scene = _read(stage).build()
  baked = scene.images.level(int(scene.textures.image[scene.materials.tex[0, S.SPECULAR_ROUGHNESS]]))[..., 0]
  assert baked[:, : baked.shape[1] // 2].min() > baked[:, baked.shape[1] // 2 :].max()  # 201 x 101 > 11 x 221


TINT_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::tex::*;
export material {name}(uniform texture_2d T = texture_2d())
 = material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(
   tint: tex::lookup_color(T, float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y)))));
"""


def test_one_texture_input_in_modules_of_two_roots(tmp_path):
  """A texture input shared by modules under two search roots is made in each root's scope."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.full((2, 2, 3), 200, np.uint8))
  for d in ("a", "b"):
    (tmp_path / d).mkdir()
    (tmp_path / d / f"M{d}.mdl").write_text(TINT_BODY.format(name=f"M{d}"))
    shader = _shader(stage, f"/Looks/{d}", f"./{d}/M{d}.mdl", f"M{d}", T=(Sdf.ValueTypeNames.Asset, Sdf.AssetPath("./t.png")))
    bind(quad(stage, f"/Q{d}"), shader)
  builder = _read(stage)
  for d in "ab":
    assert _material(builder, "/Looks/" + d).texture(S.BASE_COLOR) is not None
  assert not builder.report.dropped


PALETTE_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::tex::*;
export material {name}(uniform texture_2d T = texture_2d("./m.png", ::tex::gamma_linear))
 = let {{
  float x = tex::lookup_float(T, float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y));
  color[2] pal = color[]({c0}, {c1});
}} in material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(tint: pal[int(x * 1.5)])));
"""


def test_bake_keys_tell_arrays_apart(tmp_path):
  """Two palettes of other colors over one texture are two bakes (keys read arrays)."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "m.png", np.array([[0, 255], [255, 0]], np.uint8))
  (tmp_path / "A1.mdl").write_text(PALETTE_BODY.format(name="A1", c0="color(1,0,0)", c1="color(0,1,0)"))
  (tmp_path / "A2.mdl").write_text(PALETTE_BODY.format(name="A2", c0="color(0,0,1)", c1="color(1,1,0)"))
  for n in ("A1", "A2"):
    bind(quad(stage, f"/Q{n}"), _shader(stage, f"/Looks/{n}", f"./{n}.mdl", n))
  scene = _read(stage).build()
  levels = [scene.images.level(int(scene.textures.image[scene.materials.tex[i, S.BASE_COLOR]])) for i in range(2)]
  assert not np.array_equal(levels[0], levels[1])


OPACITY_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::tex::*;
export material Opa(uniform texture_2d T = texture_2d("./m.png", ::tex::gamma_linear),
  uniform texture_2d O = texture_2d("./o.png", ::tex::gamma_linear))
 = let {
  float2 uv = float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y);
  float x = tex::lookup_float(T, uv);
  float w = x > 0.5 ? 1.0 : 0.0;
} in material(surface: material_surface(scattering: df::normalized_mix(df::bsdf_component[](
   df::bsdf_component(w, df::microfacet_ggx_vcavities_bsdf(roughness_u: 0.1, tint: color(0.9))),
   df::bsdf_component(1.0 - w, df::diffuse_reflection_bsdf(tint: color(0.5)))))),
   geometry: material_geometry(cutout_opacity: tex::lookup_float(O, uv) * tex::lookup_float(O, uv)));
"""


def test_a_binary_metal_bake_does_not_cut_out_a_graded_opacity(tmp_path):
  """Only an opacity bake of 0 / 1 texels makes a cut-out: a binary metalness does not."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "m.png", np.array([[0, 255], [255, 0]], np.uint8))
  write_png(tmp_path / "o.png", np.array([[64, 128], [192, 255]], np.uint8))
  (tmp_path / "Opa.mdl").write_text(OPACITY_BODY)
  bind(quad(stage, "/Q"), _shader(stage, "/Looks/Opa", "./Opa.mdl", "Opa"))
  m = _material(_read(stage), "/Looks/Opa")
  assert m.texture(S.GEOMETRY_OPACITY) is not None and m.ext.alpha_mode == rw.scene.AlphaMode.BLEND


def test_failed_bakes_keep_neutral_values(tmp_path, monkeypatch):
  """When the SDK cannot bake, a field keeps its default (an F0 mean of 0.04, not 1): reported."""
  from raywarp.formats.usd.materials.mdl import bake, writer

  monkeypatch.setattr(bake, "_texels", lambda *_: None)
  monkeypatch.setattr(writer, "bake", bake.bake)
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.array([[0, 255], [128, 64]], np.uint8))
  (tmp_path / "Sin.mdl").write_text(UV_BODY.format(name="Sin", gamma="gamma_linear", expr="A.x * A.x"))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/S", "Sin.mdl", "Sin"))
  builder = _read(stage)
  m = _material(builder, "/Looks/S")
  assert m.texture(S.SPECULAR_ROUGHNESS) is None and m.specular_weight == pytest.approx(1.0)
  assert any(k.startswith("mdl.bake_failed") for k in builder.report.dropped)


DEPS_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material L(uniform texture_2d T = texture_2d("./m.png", ::tex::gamma_linear),
  uniform texture_2d C = texture_2d("./c.png", ::tex::gamma_linear))
 = let {
  float2 uv = float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y);
  float x = tex::lookup_float(T, uv);
} in material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(
   tint: math::lerp(tex::lookup_color(C, uv), color(0.0), float(int(x * 1.5))))));
"""


def test_a_bake_depends_on_every_texture_below_it(tmp_path):
  """The files of a baked expression are dependencies and bake-key inputs, read by the
  expression reader or not: an edited texture re-bakes."""
  cache = tmp_path / "cache"
  write_png(tmp_path / "m.png", np.array([[0, 255], [255, 0]], np.uint8))
  write_png(tmp_path / "c.png", np.full((2, 2, 3), (255, 0, 0), np.uint8))
  (tmp_path / "L.mdl").write_text(DEPS_BODY)
  stage = new_stage(tmp_path)
  bind(quad(stage, "/Q"), _shader(stage, "/Looks/L", "./L.mdl", "L"))
  stage.Save()
  first = rw.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)
  assert str(tmp_path / "c.png") in first.dependencies
  img = [first.build()]
  write_png(tmp_path / "c.png", np.full((2, 2, 3), (0, 0, 255), np.uint8))
  img.append(rw.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache).build())
  red, blue = (
    s.images.level(int(s.textures.image[s.materials.tex[0, S.BASE_COLOR]])).reshape(-1, 4)[:, :3].max(0) for s in img
  )
  assert red[0] > red[2] and blue[2] > blue[0]


def test_a_bake_of_a_small_texture_is_its_size(tmp_path):
  """A bake of an expression the reader cannot read takes the domain of its lookups: a 4 x 4
  texture's own size, not the procedural size."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", (np.arange(16).astype(np.uint8).reshape(4, 4) * 16))
  (tmp_path / "Sin.mdl").write_text(UV_BODY.format(name="Sin", gamma="gamma_linear", expr="float(int(A.x * 1.5)) * 0.5 + 0.25"))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/S", "Sin.mdl", "Sin"))
  scene = _read(stage).build()
  baked = scene.images.level(int(scene.textures.image[scene.materials.tex[0, S.SPECULAR_ROUGHNESS]]))
  assert baked.shape[:2] == (4, 4)


GUARD_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::tex::*;
export material G(uniform texture_2d T = texture_2d("./t.png", ::tex::gamma_linear))
 = let {
  float a = tex::lookup_float(T, float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y));
  float w = 0.0 < a * -1.0 ? a : 0.0;
} in material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(tint: color(w))));
"""


def test_a_guard_on_a_negative_factor_is_not_its_value(tmp_path):
  """`0 < a x ? a : 0` is a only when x > 0: with x = -1 it is 0, not the texture."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.full((2, 2), 200, np.uint8))
  (tmp_path / "G.mdl").write_text(GUARD_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/G", "G.mdl", "G"))
  builder = _read(stage)
  ref = _material(builder, "/Looks/G").texture(S.BASE_COLOR)
  assert ref is not None and builder.report.counts["mdl.baked"] == 1  # baked (the texture lookup varies)
  s = builder.build()
  assert not np.asarray(s.images.level(int(s.textures.image[ref.texture])))[..., :3].any()  # 0, not the texture


LOBES_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
export material Lobes(double gloss = 0.3)
 = material(surface: material_surface(scattering: df::custom_curve_layer(
   normal_reflectivity: 0.04, grazing_reflectivity: 1.0, weight: 1.0,
   layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: 0.1, roughness_v: 0.4, tint: color(1.0, 0.5, 0.5)),
   base: df::microfacet_ggx_vcavities_bsdf(roughness_u: float(gloss), roughness_v: float(gloss) * 2.0, tint: color(0.8)))));
"""


def test_coat_tint_anisotropy_and_double_inputs(tmp_path):
  """A coat's tint is coat_color; its anisotropy is reported (the core's coat is isotropic), the
  base's becomes OpenPBR's (alphas 0.6 along, 1.2 across tangent_u: stretched a quarter turn
  from it); a double parameter takes its authored value."""
  stage = new_stage(tmp_path)
  (tmp_path / "Lobes.mdl").write_text(LOBES_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/L", "Lobes.mdl", "Lobes", gloss=(Sdf.ValueTypeNames.Double, 0.6)))
  builder = _read(stage)
  m = _material(builder, "/Looks/L")
  assert m.extras is not None and m.extras.coat_color[1] == pytest.approx(0.5, abs=1e-3)
  assert "mdl.coat_anisotropy(as isotropic: roughness_u)" in builder.report.approximated
  assert _openpbr_alphas(m.specular_roughness, m.extras.specular_roughness_anisotropy) == pytest.approx((1.2, 0.6), abs=1e-5)
  assert m.ext.anisotropy_rotation == pytest.approx(0.5 * np.pi)
  assert "mdl.input:gloss" not in builder.report.dropped


def _openpbr_alphas(r: float, a: float) -> tuple[float, float]:
  """OpenPBR's (alpha_t, alpha_b) of roughness r and anisotropy a (core `anisotropic_alpha`)."""
  at = r * r * np.sqrt(2.0 / (1.0 + (1.0 - a) ** 2))
  return float(at), float((1.0 - a) * at)


ANISO_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::math::*;
export material Aniso()
 = material(surface: material_surface(scattering: df::custom_curve_layer(
   normal_reflectivity: 0.04, grazing_reflectivity: 1.0, weight: 1.0,
   layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: {u}, roughness_v: {v}, tangent_u: {t}),
   base: df::diffuse_reflection_bsdf(tint: color(0.5)))));
"""
_TURNED = "math::cos(0.5) * state::texture_tangent_u(0) + math::sin(0.5) * state::texture_tangent_v(0)"


@pytest.mark.parametrize(
  ("u", "v", "tangent", "rotation"),
  [
    ("0.3", "0.1", _TURNED, 0.5),  # stretched along the turned tangent_u
    ("0.1", "0.3", "state::texture_tangent_u(0)", 0.5 * np.pi),  # along tangent_v: a quarter turn
    ("0.2", "0.0", "state::texture_tangent_v(0)", 0.5 * np.pi),  # fully anisotropic
  ],
)
def test_anisotropic_ggx_becomes_openpbr_anisotropy(tmp_path, u, v, tangent, rotation):
  """Constant GGX alphas (along, across tangent_u) map exactly to OpenPBR's (alpha_t, alpha_b),
  the stretch direction's angle from the UV tangent to ext.anisotropy_rotation."""
  stage = new_stage(tmp_path)
  (tmp_path / "Aniso.mdl").write_text(ANISO_BODY.format(u=u, v=v, t=tangent))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/A", "Aniso.mdl", "Aniso"))
  builder = _read(stage)
  m = _material(builder, "/Looks/A")
  assert m.extras is not None
  alphas = sorted((float(u), float(v)), reverse=True)
  assert _openpbr_alphas(m.specular_roughness, m.extras.specular_roughness_anisotropy) == pytest.approx(alphas, abs=1e-6)
  assert m.ext.anisotropy_rotation == pytest.approx(rotation, abs=1e-6)
  assert not any("anisotropy" in k for k in builder.report.approximated)


def test_anisotropy_with_a_varying_tangent_is_reported(tmp_path):
  """A tangent_u that varies over the surface: the alphas still map, the UV tangent is used
  (reported)."""
  stage = new_stage(tmp_path)
  t = "math::normalize(float3(1.0, state::texture_coordinate(0).x, 0.0))"
  (tmp_path / "Aniso.mdl").write_text(ANISO_BODY.format(u="0.3", v="0.1", t=t))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/A", "Aniso.mdl", "Aniso"))
  builder = _read(stage)
  m = _material(builder, "/Looks/A")
  assert m.extras is not None and m.extras.specular_roughness_anisotropy == pytest.approx(2.0 / 3.0, abs=1e-6)
  assert m.ext.anisotropy_rotation == 0.0
  assert builder.report.approximated["mdl.anisotropy.tangent_u(varying: the UV tangent)"] == 1


def test_saturate_is_kept_around_float_texels():
  """saturate / min / max of an affine texel sum are dropped for 8-bit texels only: float
  texels (EXR, HDR) exceed [0, 1]."""
  from raywarp.formats.usd.materials.mdl import ops
  from raywarp.formats.usd.materials.mdl.values import Lin, Node, TexDecl, TexTerm, UvMap

  decl = TexDecl("x.exr")
  texel = Lin({TexTerm(decl, 0, UvMap(0, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)), (0, 0)): 1.0})
  assert ops.apply("math::saturate", [texel]) is texel
  assert isinstance(ops.apply("math::saturate", [texel], frozenset({decl})), Node)
  assert isinstance(ops.apply("math::min", [texel, Lin(c=1.0)], frozenset({decl})), Node)


def test_a_module_not_found_depends_on_where_it_would_be(tmp_path):
  """A read with a module that does not resolve depends on the files it would be found at: one
  added later makes the read stale."""
  stage = new_stage(tmp_path)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/U", "Later.mdl", "Later"))
  stage.Save()
  builder = rw.formats.usd.read(tmp_path / "scene.usda")
  assert str(tmp_path / "Later.mdl") in builder.dependencies


SKIN_BODY = """mdl 1.6;
import ::df::*;
export material Skin() = material(
  surface: material_surface(scattering: df::diffuse_transmission_bsdf(tint: color(1.0))),
  volume: material_volume(scattering_coefficient: color(25.6, 56.1, 81.7), absorption_coefficient: color(1.6, 16.9, 65.3)));
"""


def test_a_subsurface_takes_its_volume_albedo(tmp_path):
  """A diffuse transmission over a scattering volume (OmniSurface skin) is a subsurface whose
  colour is the volume's multiple-scattering albedo (van de Hulst), not the white tint."""
  stage = new_stage(tmp_path)
  (tmp_path / "Skin.mdl").write_text(SKIN_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/S", "Skin.mdl", "Skin"))
  m = _material(_read(stage), "/Looks/S")
  assert m.extras is not None and m.extras.subsurface_weight == pytest.approx(1.0)
  assert m.extras.subsurface_color == pytest.approx((0.598, 0.310, 0.170), abs=0.01)


def test_a_conversion_is_replayed_from_the_disk_cache_until_its_files_change(tmp_path):
  """A second read with the disk cache replays each MDL conversion (no SDK load or distill):
  the same material, textures and reports; a changed texture file makes it a miss."""
  stage = new_stage(tmp_path)
  img = np.zeros((4, 4), np.uint8)
  img[:, :2] = 255
  write_png(tmp_path / "t.png", img)
  (tmp_path / "Tiled.mdl").write_text(UV_BODY.format(name="Tiled", gamma="gamma_linear", expr="A.x * B.x"))
  bind(quad(stage, "/A"), _shader(stage, "/Looks/T", "Tiled.mdl", "Tiled"))
  stage.Save()
  cache = tmp_path / "cache"

  def read():
    return rw.formats.usd.read(tmp_path / "scene.usda", mdl_cache=cache)

  first, second = read(), read()
  assert "mdl.disk_cached" not in first.report.counts and second.report.counts["mdl.disk_cached"] == 1
  assert _material(first, "/Looks/T") == _material(second, "/Looks/T")
  assert first.report.approximated == second.report.approximated
  img[:, :] = 128
  write_png(tmp_path / "t.png", img)
  assert "mdl.disk_cached" not in read().report.counts


@pytest.mark.parametrize("flip", [True, False])
def test_omnipbr_normal_maps_take_the_compiled_tangent_signs(tmp_path, flip):
  """OmniPBR's normal map binds its compiled affine as is: the default `flip_tangent_v = true`
  negates y (a DirectX map), `false` keeps it; `bump_factor` scales x and y
  (scene.model.normal_map)."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "n.png", np.full((2, 2, 3), (128, 128, 255), np.uint8))
  inputs = {
    "normalmap_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("n.png")),
    "flip_tangent_v": (Sdf.ValueTypeNames.Bool, flip),
    "bump_factor": (F, 0.5),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "OmniPBR.mdl", "OmniPBR", **inputs))
  builder = _read(stage)
  ref = _material(builder, "/Looks/P").texture(S.GEOMETRY_NORMAL)
  sy = -0.5 if flip else 0.5
  assert ref is not None and ref.scale[:3] == pytest.approx((1.0, 2.0 * sy, 2.0))
  assert ref.bias[:3] == pytest.approx((-0.5, -sy, -1.0))
  assert not any("normal_map.flip" in k for k in builder.report.approximated)


def _normal_pngs(tmp_path: Path) -> None:
  """n.png (a tilt along +x) and d.png (a tilt along -y): two distinct normal maps."""
  write_png(tmp_path / "n.png", np.full((2, 2, 3), (150, 128, 255), np.uint8))
  write_png(tmp_path / "d.png", np.full((2, 2, 3), (128, 100, 255), np.uint8))


@pytest.mark.parametrize("flatten", [1.0, 0.0])
def test_omnipbr_clearcoat_normals(tmp_path, flatten):
  """OmniPBR_ClearCoat over a normal map: the base binds its own map (the distiller's blend
  toward the coat's normal undone, no bake); a flattened coat (the default 1) has the unmapped
  normal (no flag), an unflattened one the base's (COAT_BASE_NORMAL)."""
  stage = new_stage(tmp_path)
  _normal_pngs(tmp_path)
  inputs = {
    "normalmap_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("n.png")),
    "enable_clearcoat": (Sdf.ValueTypeNames.Bool, True),
    "clearcoat_weight": (F, 0.5),
    "clearcoat_flatten": (F, flatten),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/C", "OmniPBR_ClearCoat.mdl", "OmniPBR_ClearCoat", **inputs))
  builder = _read(stage)
  m = _material(builder, "/Looks/C")
  ref = m.texture(S.GEOMETRY_NORMAL)
  assert ref is not None and ref.scale[:3] == pytest.approx((2.0, -2.0, 2.0)) and "mdl.baked" not in builder.report.counts
  assert m.ext.flags & rw.scene.MaterialFlags.COAT_GLTF
  assert bool(m.ext.flags & rw.scene.MaterialFlags.COAT_BASE_NORMAL) == (flatten == 0.0)
  assert m.extras is None or not m.extras.textures


def test_omnisurface_coat_normal_is_its_own_map(tmp_path):
  """OmniSurface's coat_normal_image: the coat's own normal map (`geometry_coat_normal`), the
  base keeps geometry_normal_image; no COAT_BASE_NORMAL."""
  stage = new_stage(tmp_path)
  _normal_pngs(tmp_path)
  inputs = {
    "geometry_normal_image": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("n.png")),
    "coat_normal_image": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("d.png")),
    "coat_weight": (F, 0.5),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/S", "OmniSurface.mdl", "OmniSurface", **inputs))
  builder = _read(stage)
  m = _material(builder, "/Looks/S")
  assert m.extras is not None and m.extras.coat_darkening == 0.0
  textures = dict(m.extras.textures)
  base, coat = m.texture(S.GEOMETRY_NORMAL), textures.get("geometry_coat_normal")
  assert base is not None and coat is not None and base.texture != coat.texture
  assert not m.ext.flags & rw.scene.MaterialFlags.COAT_BASE_NORMAL


@pytest.mark.parametrize("rotate", [0.0, 30.0])
def test_omnipbr_detail_normal_map_is_its_own_binding(tmp_path, rotate):
  """OmniPBR's detail_normalmap_texture over normalmap_texture: geometry_normal binds the base
  map as alone, geometry_detail_normal the detail map with its strength (detail_bump_factor),
  the same tangent flip and its own tiling (detail_texture_scale / rotate); nothing is baked."""
  stage = new_stage(tmp_path)
  _normal_pngs(tmp_path)
  inputs = {
    "normalmap_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("n.png")),
    "detail_normalmap_texture": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath("d.png")),
    "detail_bump_factor": (F, 0.3),
    "detail_texture_scale": (Sdf.ValueTypeNames.Float2, (4.0, 4.0)),
    "detail_texture_rotate": (F, rotate),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "OmniPBR.mdl", "OmniPBR", **inputs))
  builder = _read(stage)
  m = _material(builder, "/Looks/P")
  base = m.texture(S.GEOMETRY_NORMAL)
  assert (
    base is not None and base.scale[:3] == pytest.approx((2.0, -2.0, 2.0)) and base.bias[:3] == pytest.approx((-1.0, 1.0, -1.0))
  )
  assert np.asarray(base.uv_xform) == pytest.approx(np.eye(3)[:2])
  assert m.extras is not None
  textures = dict(m.extras.textures)
  d = textures["geometry_detail_normal"]
  assert d.scale[:3] == pytest.approx((0.6, -0.6, 2.0)) and d.bias[:3] == pytest.approx((-0.3, 0.3, -1.0))
  assert d.texture != base.texture
  (a, b, _), (c, e, _) = d.uv_xform
  assert np.hypot(a, c) == pytest.approx(4.0, rel=1e-5) and np.hypot(b, e) == pytest.approx(4.0, rel=1e-5)
  assert (abs(b) > 1e-3) == (rotate != 0.0)
  assert "mdl.baked" not in builder.report.counts


DETAIL_BODY = """mdl 1.6;
import ::df::*;
import ::state::*;
import ::tex::*;
import ::nvidia::distilling_support::*;
float3 unpacked(uniform texture_2d t, float2 uv, float k) {
  float3 c = tex::lookup_float3(t, uv, tex::wrap_repeat, tex::wrap_repeat);
  return state::texture_tangent_u(0) * ((c.x * 2.0 - 1.0) * k) + state::texture_tangent_v(0) * ((c.y * 2.0 - 1.0) * k)
    + state::normal() * (c.z * 2.0 - 1.0);
}
export material Detail(
  uniform texture_2d n = texture_2d("./n.png", ::tex::gamma_linear),
  uniform texture_2d d = texture_2d("./d.png", ::tex::gamma_linear))
 = let {
  float2 uv = float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y);
  float3 nd = unpacked(d, uv * 3.0, 0.5);
  float3 nb = unpacked(n, uv, 1.0);
} in material(surface: material_surface(scattering: df::weighted_layer(
  weight: 1.0, layer: df::diffuse_reflection_bsdf(tint: color(0.5)),
  normal: nvidia::distilling_support::add_detail_normal(nd, nb))));
"""


def test_add_detail_normal_of_two_maps_splits(tmp_path):
  """`nvidia::distilling_support::add_detail_normal(nd, n)` of two normal maps: n binds
  geometry_normal, nd geometry_detail_normal (its strength 0.5 and tiling 3), not a bake."""
  stage = new_stage(tmp_path)
  _normal_pngs(tmp_path)
  (tmp_path / "Detail.mdl").write_text(DETAIL_BODY)
  bind(quad(stage, "/A"), _shader(stage, "/Looks/D", "Detail.mdl", "Detail"))
  builder = _read(stage)
  m = _material(builder, "/Looks/D")
  base = m.texture(S.GEOMETRY_NORMAL)
  assert (
    base is not None and base.scale[:3] == pytest.approx((2.0, 2.0, 2.0)) and base.bias[:3] == pytest.approx((-1.0, -1.0, -1.0))
  )
  assert m.extras is not None
  d = dict(m.extras.textures)["geometry_detail_normal"]
  assert d.scale[:3] == pytest.approx((1.0, 1.0, 2.0)) and d.bias[:3] == pytest.approx((-0.5, -0.5, -1.0))
  assert np.asarray(d.uv_xform) == pytest.approx(3.0 * np.eye(3)[:2])
  assert "mdl.baked" not in builder.report.counts


def test_rnm_matches_the_reference_formula():
  """`detail.rnm`: a flat detail keeps the base, a flat base gives the detail (up to length), and
  a tilted pair follows Barré-Brisebois and Hill's formula."""
  from raywarp.formats.usd.materials.mdl import detail

  base = np.array([[0.3], [-0.2], [0.9]])
  flat = np.array([[0.0], [0.0], [1.0]])
  unit = lambda v: v / np.linalg.norm(v, axis=0)  # noqa: E731
  assert unit(detail.rnm(base, flat)) == pytest.approx(unit(base))
  assert unit(detail.rnm(flat, base)) == pytest.approx(unit(base))
  t, u = base[:, 0] + np.array([0.0, 0.0, 1.0]), np.array([0.1, 0.4, 0.8]) * np.array([-1.0, -1.0, 1.0])
  expected = t * np.dot(t, u) / t[2] - u
  assert detail.rnm(base, np.array([[0.1], [0.4], [0.8]]))[:, 0] == pytest.approx(expected)


def test_symbolic_rnm_matches_detail_rnm():
  """`ops.rnm` over texel terms (the distiller's add_detail_normal of two maps) evaluates to
  `detail.rnm` of the decoded maps."""
  from raywarp.formats.usd.materials.mdl import detail, ops
  from raywarp.formats.usd.materials.mdl.bake_eval import Numeric, TexelPlanes
  from raywarp.formats.usd.materials.mdl.values import Lin, TexDecl, TexTerm, UvMap

  terms = [TexTerm(TexDecl(name), k, UvMap(0, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)), (0, 0)) for name in ("n", "d") for k in range(3)]
  maps = [tuple(Lin({t: 2.0}, c=-1.0) for t in terms[i : i + 3]) for i in (0, 3)]
  rng = np.random.default_rng(1)
  planes: TexelPlanes = {t: rng.uniform(0.3, 0.7, (1, 8)) if t.channel < 2 else rng.uniform(0.85, 1.0, (1, 8)) for t in terms}
  value = ops.rnm(maps[1], maps[0])
  assert isinstance(value, tuple)
  got = np.stack([Numeric(planes, (1, 8))(c)[0] for c in value])
  decode = lambda ts: np.stack([2.0 * planes[t][0] - 1.0 for t in ts])  # noqa: E731
  assert got == pytest.approx(detail.rnm(decode(terms[:3]), decode(terms[3:])))


def test_a_process_exits_cleanly_after_an_mdl_read(tmp_path: Path) -> None:
  """A process that read MDL materials (the SDK started, the baker's session and its committed
  transactions alive) exits with status 0: the SDK is shut down at exit, before its own teardown
  (which crashed with SIGSEGV)."""
  script = """
import sys
from pathlib import Path
from pxr import Sdf
from rw_helpers import bind, mdl_material, new_stage, quad
import raywarp.formats.usd
stage = new_stage(Path(sys.argv[1]))
inputs = {"metallic_constant": (Sdf.ValueTypeNames.Float, 0.5)}
bind(quad(stage, "/A"), mdl_material(stage, "/Looks/P", "OmniPBR.mdl", "OmniPBR", **inputs))
kept = raywarp.formats.usd.read(stage)
print("distilled", kept.report.counts["mdl.distilled"])
"""
  path = os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")])
  done = subprocess.run(
    [sys.executable, "-c", script, str(tmp_path)],
    env={**os.environ, "PYTHONPATH": path},
    capture_output=True,
    text=True,
    timeout=300,
    check=False,
  )
  assert done.returncode == 0, (done.returncode, done.stderr[-2000:])
  assert "distilled 1" in done.stdout


def test_empty_texture_inputs_are_no_texture_and_unresolved_ones_are_named(tmp_path):
  """OmniPBR slots authored `@@` (SimReady writes every unused slot so) are "no texture", the
  parameter default, and not reported; a file that does not resolve is reported by input."""
  stage = new_stage(tmp_path)
  asset = Sdf.ValueTypeNames.Asset
  inputs = {
    "diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, (0.6, 0.2, 0.1)),
    "ao_texture": (asset, Sdf.AssetPath("")),
    "metallic_texture": (asset, Sdf.AssetPath("")),
    "reflectionroughness_texture": (asset, Sdf.AssetPath("textures/missing_rough.png")),
  }
  bind(quad(stage, "/A"), _shader(stage, "/Looks/P", "OmniPBR.mdl", "OmniPBR", **inputs))
  builder = _read(stage)
  dropped = builder.report.dropped
  assert not any(k.startswith("mdl.input:") for k in dropped)
  assert dropped["mdl.texture.unresolved:reflectionroughness_texture(parameter default)"] == 1
  assert _material(builder, "/Looks/P").base_color == pytest.approx((0.6, 0.2, 0.1), abs=1e-6)


def test_node_key_tells_a_files_colour_spaces_apart():
  """`node_key` (the identity the MDL conversion compares lookups by) distinguishes the same
  file read in two colour spaces, or as a cube map: they hold different values."""
  from raywarp.formats.usd.materials.mdl.bake_eval import node_key
  from raywarp.formats.usd.materials.mdl.values import Lin, TexDecl, TexTerm, UvMap
  from raywarp.scene.images.colorspace import ColorSpace

  def lookup(decl: TexDecl) -> Lin:
    return Lin({TexTerm(decl, 0, UvMap(0, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)), (0, 0)): 1.0})

  plain, srgb = TexDecl("a.png"), TexDecl("a.png", ColorSpace.SRGB_REC709)
  assert node_key(lookup(plain)) == node_key(lookup(TexDecl("a.png")))
  assert node_key(lookup(plain)) != node_key(lookup(srgb))
  assert node_key(lookup(plain)) != node_key(lookup(TexDecl("a.png", cube=True)))


def test_uv_map_rejects_a_world_z_term():
  """A texture coordinate reading the world z axis has no UV-map column (row[2] is the constant):
  uv_map raises instead of overwriting the offset."""
  from raywarp.formats.usd.materials.mdl.values import WORLD_SET, Lin, UnsupportedError, UvTerm, uv_map

  u = Lin({UvTerm(WORLD_SET, 2): 1.0}, c=0.5)
  v = Lin({UvTerm(WORLD_SET, 1): 1.0})
  with pytest.raises(UnsupportedError, match="world z"):
    uv_map((u, v))
  assert uv_map((Lin({UvTerm(WORLD_SET, 0): 2.0}, c=0.5), v)).rows == (2.0, 0.0, 0.5, 0.0, 1.0, 0.0)
