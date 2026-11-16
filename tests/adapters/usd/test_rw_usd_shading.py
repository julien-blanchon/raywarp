"""USD adapter materials: UsdPreviewSurface networks, and MDL through the MDL SDK (the Omniverse
library from raywarp's copy, Unreal exports with the OmniUe4 modules the Isaac stages ship)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from rw_helpers import bind, new_stage, preview_material, quad, write_png
from rw_helpers import material_named as material_of
from rw_helpers import mdl_material as _mdl_shader

pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdShade

import raywarp as rw
import raywarp.formats.usd
from raywarp.formats.usd.materials.mdl.baker import ERROR
from raywarp.formats.usd.materials.mdl.load import LIBRARY
from raywarp.scene.images import colorspace

S = rw.scene.Slot
# The OmniUe4 modules of Unreal exports: raywarp's library copy (Omniverse Core Materials).
UE4_DIR = LIBRARY / "Ue4"
UE4_MODULES = ("OmniUe4Base.mdl", "OmniUe4Function.mdl", "OmniUe4Translucent.mdl")


@pytest.fixture
def sdk():
  """MDL tests need the MDL SDK wheel."""


@pytest.fixture
def ue4(tmp_path, sdk):
  """The OmniUe4 modules next to the test stage (the stage's own copy, as Unreal exports ship)."""
  for m in UE4_MODULES:
    shutil.copy(UE4_DIR / m, tmp_path / m)


def image_of(builder: rw.SceneBuilder, tex: int) -> tuple[str, rw.scene.ColorSpace, tuple[int, int]]:
  image, cs, ws, wt = builder._textures[tex]
  return builder._images.sources[image].path.rsplit("/", 1)[-1], colorspace.from_code(cs), (ws, wt)


def uv_texture(stage, path: str, file: str, **inputs):
  tex = UsdShade.Shader.Define(stage, path)
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
  for name, (type_name, value) in inputs.items():
    tex.CreateInput(name, type_name).Set(value)
  return tex


def test_preview_surface_constants(tmp_path):
  stage = new_stage(tmp_path)
  mat, shader = preview_material(stage, "/Looks/M", (0.1, 0.2, 0.3), metallic=0.7, roughness=0.3, ior=1.33, opacity=0.5)
  shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((2, 0, 0))
  bind(quad(stage, "/Q"), mat)
  m = material_of(rw.formats.usd.read(stage), "/Looks/M")
  assert m.base_color == pytest.approx((0.1, 0.2, 0.3)) and m.base_metalness == pytest.approx(0.7)
  assert (
    m.specular_roughness == pytest.approx(0.3) and m.specular_ior == pytest.approx(1.33) and m.emission_color == (2.0, 0.0, 0.0)
  )
  assert m.geometry_opacity == 0.5 and m.ext.alpha_mode == rw.scene.AlphaMode.BLEND


def test_preview_surface_out_of_range_constants_are_clamped_and_reported(tmp_path):
  """An out-of-range UsdPreviewSurface constant loads clamped and reported, as on the MaterialX
  path (it used to fail the whole stage in `checks.material`)."""
  stage = new_stage(tmp_path)
  mat, _ = preview_material(stage, "/Looks/M", (-1.0, 0.5, 0.5), roughness=2.0, ior=0.0, opacityThreshold=2.0)
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  m = material_of(builder, "/Looks/M")
  assert m.base_color == (0.0, 0.5, 0.5) and m.specular_roughness == 1.0 and m.specular_ior == 1.5
  assert m.ext.alpha_mode == rw.scene.AlphaMode.MASK and m.ext.alpha_cutoff == 1.0
  approximated = builder.report.approximated
  for key in (
    "openpbr.base_color(clamped to [0.0, 1.0])",
    "openpbr.specular_roughness(clamped to [0.0, 1.0])",
    "openpbr.specular_ior(not positive: the default)",
    f"openpbr.ext.alpha_cutoff(clamped to [{1 / 512}, 1.0])",
  ):
    assert approximated[key] == 1, (key, dict(approximated))


def test_preview_surface_textures(tmp_path):
  stage = new_stage(tmp_path)
  write_png(tmp_path / "base.png", np.zeros((2, 2, 4), np.uint8))
  write_png(tmp_path / "orm.png", np.zeros((2, 2, 3), np.uint8))
  write_png(tmp_path / "n.png", np.zeros((2, 2, 3), np.uint8))
  mat, shader = preview_material(stage, "/Looks/M", opacityThreshold=0.4)
  reader = UsdShade.Shader.Define(stage, "/Looks/M/Reader")
  reader.CreateIdAttr("UsdPrimvarReader_float2")
  reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("uvmap")
  xf = UsdShade.Shader.Define(stage, "/Looks/M/Xf")
  xf.CreateIdAttr("UsdTransform2d")
  xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
  xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((2, 3))
  xf.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set((0.5, 0.25))
  base = uv_texture(stage, "/Looks/M/Base", "base.png", wrapS=(Sdf.ValueTypeNames.Token, "clamp"))
  base.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
  orm = uv_texture(stage, "/Looks/M/ORM", "orm.png", sourceColorSpace=(Sdf.ValueTypeNames.Token, "raw"))
  normal = uv_texture(
    stage,
    "/Looks/M/N",
    "n.png",
    scale=(Sdf.ValueTypeNames.Float4, (2, 2, 2, 1)),
    bias=(Sdf.ValueTypeNames.Float4, (-1, -1, -1, 0)),
  )
  shader.GetInput("diffuseColor").ConnectToSource(base.ConnectableAPI(), "rgb")
  shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(base.ConnectableAPI(), "a")
  shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(orm.ConnectableAPI(), "g")
  shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(orm.ConnectableAPI(), "b")
  shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(), "rgb")
  mesh = quad(stage, "/Q")
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("uvmap", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
    [(0, 0), (1, 0), (1, 1), (0, 1)]
  )
  bind(mesh, mat)
  builder = rw.formats.usd.read(stage)
  m = material_of(builder, "/Looks/M")
  b = m.texture(S.BASE_COLOR)
  assert image_of(builder, b.texture) == (
    "base.png",
    rw.scene.ColorSpace.SRGB_REC709,
    (rw.scene.Wrap.CLAMP, rw.scene.Wrap.REPEAT),
  )
  np.testing.assert_allclose(b.uv_xform, [[2, 0, 0.5], [0, 3, 0.25]])
  assert (
    m.texture(S.GEOMETRY_OPACITY).channel == 3
    and m.ext.alpha_mode == rw.scene.AlphaMode.MASK
    and m.ext.alpha_cutoff == pytest.approx(0.4)
  )
  assert m.texture(S.SPECULAR_ROUGHNESS).channel == 1 and m.texture(S.BASE_METALNESS).channel == 2
  assert image_of(builder, m.texture(S.SPECULAR_ROUGHNESS).texture)[1] == rw.scene.ColorSpace.DATA  # "raw" (§7.1)
  assert m.texture(S.GEOMETRY_NORMAL).bias == (-1.0, -1.0, -1.0, 0.0) and m.base_metalness == 1.0 and m.base_color == (1, 1, 1)
  assert builder.build().meshes.uv_num[0] == 4  # the UV set named by the primvar reader was read


def test_omnipbr_field_map_with_vendored_defaults(tmp_path, sdk):
  stage = new_stage(tmp_path)
  write_png(tmp_path / "d.png", np.zeros((2, 2, 3), np.uint8))
  write_png(tmp_path / "r.png", np.array([[0, 255], [128, 64]], np.uint8))
  write_png(tmp_path / "n.png", np.full((2, 2, 3), (128, 128, 255), np.uint8))
  mat = _mdl_shader(
    stage,
    "/Looks/Pbr",
    "OmniPBR.mdl",  # not next to the stage: the vendored library source's defaults
    "OmniPBR",
    diffuse_texture=(Sdf.ValueTypeNames.Asset, "d.png"),
    diffuse_tint=(Sdf.ValueTypeNames.Color3f, (0.5, 0.5, 0.5)),
    reflectionroughness_texture=(Sdf.ValueTypeNames.Asset, "r.png"),
    reflection_roughness_texture_influence=(Sdf.ValueTypeNames.Float, 1.0),
    enable_emission=(Sdf.ValueTypeNames.Bool, True),
    texture_scale=(Sdf.ValueTypeNames.Float2, (4, 4)),
    normalmap_texture=(Sdf.ValueTypeNames.Asset, "n.png"),
    bump_factor=(Sdf.ValueTypeNames.Float, 0.5),
    some_unknown_input=(Sdf.ValueTypeNames.Float, 3.0),
  )
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  m = material_of(builder, "/Looks/Pbr")
  base = m.texture(S.BASE_COLOR)
  assert m.base_color == (1.0, 1.0, 1.0) and base is not None and base.scale[:3] == pytest.approx((0.5, 0.5, 0.5))
  np.testing.assert_allclose(base.uv_xform, [[4, 0, 0], [0, 4, 0]])
  assert m.texture(S.SPECULAR_ROUGHNESS) is not None and m.specular_roughness == 1.0
  assert m.specular_weight == pytest.approx(1.0)
  # Library defaults; emissive_intensity is a radiant exitance (diffuse EDF: radiance = I / pi).
  assert m.emission_color == pytest.approx((1.0, 0.1, 0.1)) and m.emission_luminance == pytest.approx(40.0 / np.pi)
  assert builder.report.dropped["mdl.input:some_unknown_input"] == 1
  # The normal map is the file itself through the same tiling, its compiled affine as is: unpacked
  # 2 t - 1, bump_factor 0.5 on x and y, y negated by OmniPBR's default flip_tangent_v (DirectX).
  normal = m.texture(S.GEOMETRY_NORMAL)
  assert normal is not None and normal.scale[:3] == (1.0, -1.0, 2.0) and normal.bias[:3] == (-0.5, 0.5, -1.0)
  assert "mdl.baked" not in builder.report.counts and not any("normal_map.flip" in k for k in builder.report.approximated)
  np.testing.assert_allclose(normal.uv_xform, base.uv_xform)


def test_omniglass(tmp_path, sdk):
  stage = new_stage(tmp_path)
  mat = _mdl_shader(stage, "/Looks/Glass", "OmniGlass.mdl", "OmniGlass", glass_ior=(Sdf.ValueTypeNames.Float, 1.2))
  bind(quad(stage, "/Q"), mat)
  m = material_of(rw.formats.usd.read(stage), "/Looks/Glass")
  assert (
    m.transmission_weight == 1.0
    and m.specular_ior == pytest.approx(1.2)
    and m.specular_roughness == 0.0
    and m.base_metalness == 0.0
  )


UE_MDL = """mdl 1.3;
import ::anno::*;
import ::math::*;
import ::state::*;
import ::tex::*;
using .::OmniUe4Function import *;
using .::OmniUe4Base import *;
export material MI_Test(
\tuniform texture_2d BaseColor = texture_2d("./base.png",::tex::gamma_srgb)
\t[[sampler_color()]],
\tfloat4 Color = float4(0.5,0.25,1.0,1.0),
\tuniform texture_2d AO_R_Rough_G_Metallic_B_ = texture_2d("./orm.png",::tex::gamma_linear),
\tfloat EmissionIntensity = 2.0,
\tfloat RoughnessMultiplier = 0.5)
 =
\tlet {
\tfloat2 CustomizedUV0_mdl = float2(state::texture_coordinate(0).x,1.0-state::texture_coordinate(0).y);
\tfloat4 Local0 = tex::lookup_float4(BaseColor,float2(CustomizedUV0_mdl.x,1.0-CustomizedUV0_mdl.y),
		tex::wrap_repeat,tex::wrap_clamp);
\tfloat3 Local1 = (float3(Local0.x,Local0.y,Local0.z) * float3(Color.x,Color.y,Color.z));
\tfloat4 Local2 = tex::lookup_float4(AO_R_Rough_G_Metallic_B_,float2(CustomizedUV0_mdl.x,1.0-CustomizedUV0_mdl.y),
		tex::wrap_repeat,tex::wrap_repeat);
\tfloat Local3 = (Local2.w * EmissionIntensity);
\tfloat3 EmissiveColor_mdl = (Local3 * float3(1.0,0.5,0.0));
\tfloat OpacityMask_mdl = (Local0.w - 0.3333) < 0.0f ? 0.0f : 1.0f;
\tfloat Roughness_mdl = (Local2.y * RoughnessMultiplier);
\t} in
\t\t::OmniUe4Base(
\t\t\tbase_color: Local1,
\t\t\tmetallic: Local2.z,
\t\t\troughness: Roughness_mdl,
\t\t\tspecular: 0.5,
\t\t\topacity: OpacityMask_mdl,
\t\t\temissive_color: EmissiveColor_mdl,
\t\t\ttwo_sided: true);
"""


def test_unreal_mdl_body_is_evaluated(tmp_path, ue4):
  stage = new_stage(tmp_path)
  (tmp_path / "MI_Test.mdl").write_text(UE_MDL)
  rng = np.random.default_rng(0)
  write_png(tmp_path / "base.png", rng.integers(0, 256, (2, 2, 4), np.uint8))
  write_png(tmp_path / "orm.png", rng.integers(0, 256, (2, 2, 4), np.uint8))
  mat = _mdl_shader(stage, "/Looks/UE", "MI_Test.mdl", "MI_Test", RoughnessMultiplier=(Sdf.ValueTypeNames.Float, 0.8))
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  m = material_of(builder, "/Looks/UE")
  base = m.texture(S.BASE_COLOR)
  assert base.channel == rw.scene.CHANNEL_RGB and base.scale[:3] == pytest.approx((0.5, 0.25, 1.0))  # x Color (default)
  assert image_of(builder, base.texture) == (
    "base.png",
    rw.scene.ColorSpace.SRGB_REC709,
    (rw.scene.Wrap.REPEAT, rw.scene.Wrap.CLAMP),
  )
  np.testing.assert_allclose(base.uv_xform, [[1, 0, 0], [0, 1, 0]])  # the two v flips cancel
  rough = m.texture(S.SPECULAR_ROUGHNESS)
  assert rough.channel == 1 and rough.scale[0] == pytest.approx(0.8)  # authored multiplier wins
  assert m.texture(S.BASE_METALNESS).channel == 2
  # Packed emission alpha x EmissionIntensity x (1, 0.5, 0) x emissive_multiplier() (2560), as
  # radiant exitance: radiance = I / pi; the largest channel scale is the luminance.
  emission = m.texture(S.EMISSION_COLOR)
  assert emission.scale[:3] == pytest.approx((1.0, 0.5, 0.0)) and m.emission_luminance == pytest.approx(5120.0 / np.pi)
  # The thresholded opacity mask bakes to 0 / 1 texels: a cut-out; two_sided is thin_walled.
  assert m.ext.alpha_mode == rw.scene.AlphaMode.MASK and m.texture(S.GEOMETRY_OPACITY) is not None
  assert m.geometry_thin_walled
  assert m.ext.flags & rw.scene.MaterialFlags.SPECULAR_CURVE  # custom_curve_layer specular (F90 = s (1 - r))
  assert builder.report.counts["mdl.distilled"] == 1


UE_SPECULAR = """mdl 1.3;
import ::anno::*;
import ::math::*;
import ::state::*;
import ::tex::*;
using .::OmniUe4Function import *;
using .::OmniUe4Base import *;
export material M_Spec(uniform texture_2d Mask = texture_2d("./{file}",::tex::gamma_linear))
 =
\tlet {{
\tfloat4 L = tex::lookup_float4(Mask,float2(state::texture_coordinate(0).x,state::texture_coordinate(0).y),
\t\ttex::wrap_repeat,tex::wrap_repeat);
\t}} in
\t\t::OmniUe4Base(specular: L.x * 0.5 + 0.1);
"""


@pytest.mark.parametrize("file", ["mask.png", "missing.png"])
def test_unreal_textured_specular_is_twice_the_level(tmp_path, file, ue4):
  """w = 2 s (F0 = 0.08 s against 0.04), a texture's mean (the core renders no specular_weight
  texture: none is bound); the black mask and a texture that cannot be read (0) both leave
  2 x 0.1."""
  stage = new_stage(tmp_path)
  (tmp_path / "M_Spec.mdl").write_text(UE_SPECULAR.format(file=file))
  write_png(tmp_path / "mask.png", np.zeros((2, 2, 4), np.uint8))
  bind(quad(stage, "/Q"), _mdl_shader(stage, "/Looks/Spec", "M_Spec.mdl", "M_Spec"))
  builder = rw.formats.usd.read(stage)
  m = material_of(builder, "/Looks/Spec")
  assert m.texture(S.SPECULAR_WEIGHT) is None and m.specular_weight == pytest.approx(2.0 * 0.1, rel=1e-5)


UE_EMITTER = """mdl 1.3;
import ::anno::*;
import ::math::*;
import ::state::*;
import ::tex::*;
using .::OmniUe4Function import *;
using .::OmniUe4Base import *;
export material M_Emit(float B = 1.0)
 =
\tlet {
\tfloat3 EmissiveColor_mdl = float3(B,B,B);
\t} in
\t\t::OmniUe4Base(base_color: float3(0.0,0.0,0.0), emissive_color: EmissiveColor_mdl);
"""


def test_omnipbr_and_unreal_emission_agree(tmp_path, ue4):
  """OmniPBR and OmniUe4Base both emit `material_emission(diffuse_edf(), intensity: I)` (radiant
  exitance): the same I gives the same radiance, I / pi (Omniverse RTX shows them equal).
  OmniPBR's I = color x emissive_intensity; OmniUe4Base's I = emissive color x
  emissive_multiplier() (2560)."""
  stage = new_stage(tmp_path)
  (tmp_path / "M_Emit.mdl").write_text(UE_EMITTER)
  pbr = _mdl_shader(
    stage,
    "/Looks/Pbr",
    "OmniPBR.mdl",
    "OmniPBR",
    enable_emission=(Sdf.ValueTypeNames.Bool, True),
    emissive_color=(Sdf.ValueTypeNames.Color3f, (1.0, 1.0, 1.0)),
    emissive_intensity=(Sdf.ValueTypeNames.Float, 2560.0),
  )
  ue = _mdl_shader(stage, "/Looks/UE", "M_Emit.mdl", "M_Emit")
  bind(quad(stage, "/A"), pbr)
  bind(quad(stage, "/B"), ue)
  builder = rw.formats.usd.read(stage)
  radiance = {}
  for name in ("Pbr", "UE"):
    m = material_of(builder, "/Looks/" + name)
    radiance[name] = np.asarray(m.emission_color) * m.emission_luminance
  np.testing.assert_allclose(radiance["Pbr"], radiance["UE"], rtol=1e-6)
  np.testing.assert_allclose(radiance["Pbr"], [2560.0 / np.pi] * 3, rtol=1e-6)


def test_omnisurface_emission_reads_its_emission_mode(tmp_path, sdk):
  """OmniSurfaceBase.mdl: I = weight x intensity x color x (emission_lx ? 1 : pi) as radiant
  exitance. The default emission_lx gives OmniPBR's radiance I / pi for the same intensity (RTX
  shows the two quads equal); emission_nt (1, or its
  token) makes the intensity the radiance [nits]."""
  stage = new_stage(tmp_path)
  t = Sdf.ValueTypeNames
  pbr = _mdl_shader(
    stage,
    "/Looks/Pbr",
    "OmniPBR.mdl",
    "OmniPBR",
    enable_emission=(t.Bool, True),
    emissive_color=(t.Color3f, (1.0, 1.0, 1.0)),
    emissive_intensity=(t.Float, 1000.0),
  )
  emit = {"emission_weight": (t.Float, 0.5), "emission_intensity": (t.Float, 2000.0)}
  lux = _mdl_shader(stage, "/Looks/Lux", "OmniSurface.mdl", "OmniSurface", **emit)
  nit = _mdl_shader(stage, "/Looks/Nit", "OmniSurface.mdl", "OmniSurface", emission_mode=(t.Int, 1), **emit)
  tok = _mdl_shader(stage, "/Looks/Tok", "OmniSurface.mdl", "OmniSurface", emission_mode=(t.Token, "emission_nt"), **emit)
  for i, mat in enumerate((pbr, lux, nit, tok)):
    bind(quad(stage, f"/Q{i}"), mat)
  builder = rw.formats.usd.read(stage)
  radiance = {
    n: np.asarray(material_of(builder, "/Looks/" + n).emission_color) * material_of(builder, "/Looks/" + n).emission_luminance
    for n in ("Pbr", "Lux", "Nit", "Tok")
  }
  np.testing.assert_allclose(radiance["Lux"], radiance["Pbr"], rtol=1e-6)
  np.testing.assert_allclose(radiance["Lux"], [1000.0 / np.pi] * 3, rtol=1e-6)
  np.testing.assert_allclose(radiance["Nit"], [1000.0] * 3, rtol=1e-6)
  np.testing.assert_allclose(radiance["Tok"], [1000.0] * 3, rtol=1e-6)


def test_omnisurface_transmission_and_thin_walled_follow_its_inputs(tmp_path, sdk):
  """OmniSurfaceBase.mdl: specular_transmission_weight counts only with
  enable_specular_transmission (default false); the thin-walled input is `thin_walled`."""
  stage = new_stage(tmp_path)
  t = Sdf.ValueTypeNames
  off = _mdl_shader(stage, "/Looks/Off", "OmniSurface.mdl", "OmniSurface", specular_transmission_weight=(t.Float, 1.0))
  on = _mdl_shader(
    stage,
    "/Looks/On",
    "OmniSurface.mdl",
    "OmniSurface",
    enable_specular_transmission=(t.Bool, True),
    specular_transmission_weight=(t.Float, 1.0),
    specular_transmission_color=(t.Color3f, (0.5, 0.25, 1.0)),
    specular_reflection_weight=(t.Float, 0.5),
    diffuse_reflection_weight=(t.Float, 0.5),
    diffuse_reflection_roughness=(t.Float, 0.75),
    coat_weight=(t.Float, 1.0),
    thin_walled=(t.Bool, True),
  )
  bind(quad(stage, "/A"), off)
  bind(quad(stage, "/B"), on)
  builder = rw.formats.usd.read(stage)
  m_off, m_on = material_of(builder, "/Looks/Off"), material_of(builder, "/Looks/On")
  assert m_off.transmission_weight == 0.0 and not m_off.geometry_thin_walled
  assert m_on.transmission_weight == 1.0 and m_on.geometry_thin_walled
  # OpenPBR terms (canonical_design.md §3.4): no 0.5 sqrt(w); the base weight stays folded in
  # (today's bits, Q7); an unauthored transmission color is the base's (the derived base unchanged).
  # The specular weight is the specular layer's tint in the distilled material.
  assert m_on.transmission_color == pytest.approx((0.5, 0.25, 1.0)) and m_on.specular_weight == pytest.approx(0.5)
  assert m_off.transmission_color == (1.0, 1.0, 1.0)  # no transmission: left at the default
  assert m_on.coat_weight == pytest.approx(1.0) and m_on.ext.flags & rw.scene.MaterialFlags.COAT_GLTF


def test_omnisurface_partial_transmission_keeps_the_base(tmp_path, sdk):
  """t < 1 with no authored transmission color: the color is the base's, so derive's
  lerp(base, transmission_color, t) leaves today's base unchanged."""
  stage = new_stage(tmp_path)
  t = Sdf.ValueTypeNames
  mat = _mdl_shader(
    stage,
    "/Looks/Half",
    "OmniSurface.mdl",
    "OmniSurface",
    enable_specular_transmission=(t.Bool, True),
    specular_transmission_weight=(t.Float, 0.5),
    diffuse_reflection_color=(t.Color3f, (0.2, 0.4, 0.6)),
  )
  bind(quad(stage, "/A"), mat)
  m = material_of(rw.formats.usd.read(stage), "/Looks/Half")
  # OmniSurface's default transmission color is white (the distilled transmission's tint).
  assert m.transmission_weight == pytest.approx(0.5) and m.transmission_color == pytest.approx((1.0, 1.0, 1.0))


def test_mdl_specular_levels_become_openpbr_weights(tmp_path, ue4):
  """OmniPBR's specular_level s and Unreal's specular s are both linear in F0 = 0.08 s (their
  custom_curve_layer's weight): w = 2 s."""
  stage = new_stage(tmp_path)
  t = Sdf.ValueTypeNames
  pbr = _mdl_shader(stage, "/Looks/Pbr", "OmniPBR.mdl", "OmniPBR", specular_level=(t.Float, 0.25))
  bind(quad(stage, "/A"), pbr)
  (tmp_path / "Ue.mdl").write_text(
    "mdl 1.6;\nusing .::OmniUe4Base import *;\nexport material Ue() = OmniUe4Base(specular: 0.25);\n"
  )
  ue = _mdl_shader(stage, "/Looks/Ue", "Ue.mdl", "Ue")
  bind(quad(stage, "/B"), ue)
  builder = rw.formats.usd.read(stage)
  assert material_of(builder, "/Looks/Pbr").specular_weight == pytest.approx(0.5)
  assert material_of(builder, "/Looks/Ue").specular_weight == pytest.approx(0.5)


UE_TRANSLUCENT = """mdl 1.3;
import ::anno::*;
import ::math::*;
import ::state::*;
import ::tex::*;
using .::OmniUe4Function import *;
using .::OmniUe4Translucent import *;
export material MI_Glass(
\tuniform texture_2d BaseColor = texture_2d("./base.png",::tex::gamma_srgb),
\tfloat Opacity_Add = 0.1,
\tfloat Opacity_Multiply = 0.5)
 =
\tlet {
\tfloat4 Local1 = tex::lookup_float4(BaseColor,float2(state::texture_coordinate(0).x,state::texture_coordinate(0).y),
		tex::wrap_repeat,tex::wrap_repeat);
\tfloat Local5 = ((Local1.w + Opacity_Add) * Opacity_Multiply);
\tfloat OpacityMask_mdl = (math::saturate(Local5) - 1.0f / 255.0f) < 0.0f ? 0.0f : 1.0f;
\t} in
\t\t::OmniUe4Translucent(
\t\t\tbase_color: float3(Local1.x,Local1.y,Local1.z),
\t\t\topacity: Local5,
\t\t\topacity_mask: OpacityMask_mdl,
\t\t\tsomething_new: lerp_fancy(Local1.x));
"""


def test_unreal_translucent_and_an_unknown_argument(tmp_path, ue4):
  stage = new_stage(tmp_path)
  (tmp_path / "MI_Glass.mdl").write_text(
    UE_TRANSLUCENT.replace("\t\t\tsomething_new: lerp_fancy(Local1.x));", "\t\t\tdisplacement: float3(0.0));")
  )
  (tmp_path / "MI_Odd.mdl").write_text(UE_TRANSLUCENT.replace("MI_Glass", "MI_Odd"))
  write_png(tmp_path / "base.png", np.zeros((2, 2, 4), np.uint8))
  bind(quad(stage, "/A"), _mdl_shader(stage, "/Looks/Glass", "MI_Glass.mdl", "MI_Glass"))
  bind(quad(stage, "/B", 2.0), _mdl_shader(stage, "/Looks/Odd", "MI_Odd.mdl", "MI_Odd"))
  builder = rw.formats.usd.read(stage)
  glass = material_of(builder, "/Looks/Glass")
  op = glass.texture(S.GEOMETRY_OPACITY)
  assert glass.ext.alpha_mode == rw.scene.AlphaMode.BLEND and op.channel == 3
  assert op.scale[0] == pytest.approx(0.5) and op.bias[0] == pytest.approx(0.05)  # (a + add) * mul
  # An unknown function does not compile: the material is ERROR, reported.
  assert material_of(builder, "/Looks/Odd").base_color == ERROR.base_color
  assert any(k.startswith("mdl.load_failed:MI_Odd") for k in builder.report.dropped)
  assert glass.ext.flags & rw.scene.MaterialFlags.SPECULAR_CURVE


def test_unknown_mdl_material_is_an_error(tmp_path, sdk):
  """Any MDL material is compiled as written; one that does not load is ERROR, reported."""
  stage = new_stage(tmp_path)
  (tmp_path / "Custom.mdl").write_text(
    "mdl 1.6;\nexport material Custom(uniform texture_2d BaseColor = texture_2d()) = material(surface: material_surface());\n"
  )
  bind(quad(stage, "/A"), _mdl_shader(stage, "/Looks/C", "Custom.mdl", "Custom"))
  bind(quad(stage, "/B", 2.0), _mdl_shader(stage, "/Looks/M", "Missing.mdl", "Missing"))
  builder = rw.formats.usd.read(stage)
  # `material(surface: material_surface())` is a valid black material; a missing module is ERROR.
  assert material_of(builder, "/Looks/C").base_color == (0.0, 0.0, 0.0)
  assert material_of(builder, "/Looks/M").base_color == ERROR.base_color
  assert any(k.startswith("mdl.load_failed:Missing") for k in builder.report.dropped)


def test_unbound_gprim_uses_display_color(tmp_path):
  stage = new_stage(tmp_path)
  mesh = quad(stage, "/Q")
  mesh.CreateDisplayColorAttr([(0.9, 0.1, 0.1)])
  s = rw.formats.usd.read(stage).build()
  np.testing.assert_allclose(s.materials.base_color[s.instances.material[0], :3], [0.9, 0.1, 0.1], rtol=1e-6)


def test_vertex_display_opacity_blends(tmp_path):
  """An unbound mesh with per-vertex displayColor and displayOpacity below 1 gets a blended
  vertex-color material (the vertex alpha is its opacity); opaque vertices stay opaque."""
  for opacity, mode in ((0.5, rw.scene.AlphaMode.BLEND), (1.0, rw.scene.AlphaMode.OPAQUE)):
    (tmp_path / str(opacity)).mkdir()
    stage = new_stage(tmp_path / str(opacity))
    mesh = quad(stage, "/Q")
    mesh.CreateDisplayColorAttr([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]).SetMetadata("interpolation", "vertex")
    mesh.CreateDisplayOpacityAttr([opacity] * 4).SetMetadata("interpolation", "vertex")
    s = rw.formats.usd.read(stage).build()
    material = int(s.instances.material[0])
    assert int(s.materials.ext_flags[material]) & int(rw.scene.MaterialFlags.VERTEX_COLOR)
    assert int(s.materials.ext_alpha_mode[material]) == int(mode)


def test_a_uv_texture_inside_a_usdz_decodes_from_the_package(tmp_path):
  """A UsdUVTexture of a `.usdz` stage resolves to `<package>[member]` and decodes from the
  package (it was reported missing: the decoders read files only)."""
  src = tmp_path / "src"
  (src / "tex").mkdir(parents=True)
  write_png(src / "tex" / "c.png", np.full((2, 2, 3), (0, 255, 0), np.uint8))
  stage = new_stage(src)
  mat, shader = preview_material(stage, "/Looks/M")
  tex = uv_texture(stage, "/Looks/M/Tex", "./tex/c.png")
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
  bind(quad(stage, "/Q"), mat)
  stage.GetRootLayer().Save()
  package = tmp_path / "scene.usdz"
  writer = Sdf.ZipFileWriter.CreateNew(str(package))
  for name in ("scene.usda", "tex/c.png"):
    writer.AddFile(str(src / name), name)
  writer.Save()
  builder = rw.formats.usd.read(Usd.Stage.Open(str(package)))
  assert "texture.missing_file" not in builder.report.dropped
  images = builder.build().images
  assert list(images.paths) == [f"{package}[tex/c.png]"]
  assert images.data[int(images.offset[0]) : int(images.offset[0]) + 4].tolist() == [0, 255, 0, 255]
  assert str(package) in builder.dependencies


def test_udim_set_is_one_atlas_texture(tmp_path):
  """A UsdUVTexture whose file is a UDIM template (`<UDIM>`) binds one atlas of its tiles: tile
  1001 + u + 10 v at column u, row v from the bottom, the UV scaled by 1 / (cols, rows) after the
  UsdTransform2d chain, missing grid cells at the fallback (the SimReady plywood crate rendered
  grey: UDIM sets were dropped)."""
  stage = new_stage(tmp_path)
  colors = {1001: (255, 0, 0), 1002: (0, 255, 0), 1011: (0, 0, 255)}
  for tile, rgb in colors.items():
    write_png(tmp_path / f"wood.{tile}.png", np.full((4, 4, 3), rgb, np.uint8))
  mat, shader = preview_material(stage, "/Looks/M")
  tex = uv_texture(stage, "/Looks/M/Tex", "./wood.<UDIM>.png", fallback=(Sdf.ValueTypeNames.Float4, (1, 1, 1, 1)))
  shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  assert not any("udim" in k for k in builder.report.dropped)
  b = material_of(builder, "/Looks/M").texture(S.BASE_COLOR)
  np.testing.assert_allclose(b.uv_xform, [[0.5, 0, 0], [0, 0.5, 0]])
  images = builder.build().images
  start = int(images.offset[0])
  atlas = images.data[start : start + 8 * 8 * 4].reshape(8, 8, 4)
  # Rows from the top: v = 1 (tile 1011 blue, then the missing 1012: fallback white), then v = 0.
  np.testing.assert_array_equal(atlas[0, 0, :3], colors[1011])
  np.testing.assert_array_equal(atlas[0, 7], (255, 255, 255, 255))
  np.testing.assert_array_equal(atlas[7, 0, :3], colors[1001])
  np.testing.assert_array_equal(atlas[7, 7, :3], colors[1002])


def test_unresolved_uv_texture_file_is_reported_and_takes_its_fallback(tmp_path):
  """A UsdUVTexture whose file does not resolve shades with its `fallback` (the spec's rule) and
  is reported: a typo'd or unpackaged texture never turns a surface black silently."""
  stage = new_stage(tmp_path)
  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Tex")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("textures/missing.png")
  tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set((0.25, 0.5, 0.75, 1.0))
  shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  m = material_of(builder, "/Looks/M")
  assert m.texture(S.BASE_COLOR) is None
  assert m.base_color == pytest.approx((0.25, 0.5, 0.75))
  assert builder.report.dropped["UsdUVTexture.file.unresolved(fallback used)"] == 1


def test_surface_wrapped_in_a_node_graph_is_its_shader(tmp_path):
  """A material whose surface output connects to a NodeGraph output that the UsdPreviewSurface
  feeds (the usd-wg Teapot's layout) converts that shader, as USD resolves it."""
  stage = new_stage(tmp_path)
  mat = UsdShade.Material.Define(stage, "/Looks/M")
  graph = UsdShade.NodeGraph.Define(stage, "/Looks/M/Graph")
  shader = UsdShade.Shader.Define(stage, "/Looks/M/Graph/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.7, 0.2))
  graph.CreateOutput("surface", Sdf.ValueTypeNames.Token).ConnectToSource(shader.ConnectableAPI(), "surface")
  mat.CreateSurfaceOutput().ConnectToSource(graph.ConnectableAPI(), "surface")
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  assert material_of(builder, "/Looks/M").base_color == pytest.approx((0.1, 0.7, 0.2))
  assert not any(k.startswith("material.") for k in builder.report.dropped)
