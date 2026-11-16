"""The color-space registry (`scene/images/colorspace.py`), its use at upload (`texture_pool.py`)
and by the importers and outputs (canonical_design.md §7)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import warp as wp
from hypothesis import given
from hypothesis import strategies as st

import raywarp as rw
from raywarp.core.outputs.render_outputs import OUTPUT_COLOR_SPACES, VIEWS, RenderOutputs
from raywarp.core.pipeline.textures.pool import ONE, Channel, TexturePool, converted_view
from raywarp.scene.images import colorspace
from raywarp.scene.images.colorspace import ALIASES, COLOR_SPACES, ColorSpace, auto_colorspace
from raywarp.scene.images.mips import linear_to_srgb_u8
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import TextureId

# Chromaticities (x, y) of the primaries (R, G, B) and white of each space (module docstring).
D65 = (0.3127, 0.3290)
ACES_WHITE = (0.32168, 0.33767)
REC709 = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
PRIMARIES = {
  "ACESCG_TO_REC709": (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), ACES_WHITE),
  "ACES2065_TO_REC709": (((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770)), ACES_WHITE),
  "P3D65_TO_REC709": (((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), D65),
  "REC2020_TO_REC709": (((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), D65),
  "ADOBERGB_TO_REC709": (((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), D65),
}
# The Bradford cone response matrix (Lam 1985, as OCIO and colour-science use it).
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
# OCIO's published ACEScg -> linear Rec.709 matrix (ACES-AP1_to_LINEAR-REC709_BFD).
OCIO_ACESCG_TO_REC709 = (
  (1.70505099, -0.62179212, -0.08325887),
  (-0.13025642, 1.14080474, -0.01054832),
  (-0.02400336, -0.12896898, 1.15297233),
)


def _xyz(xy: tuple[float, float]) -> np.ndarray:
  x, y = xy
  return np.array([x / y, 1.0, (1.0 - x - y) / y])


def _rgb_to_xyz(primaries: tuple[tuple[float, float], ...], white: tuple[float, float]) -> np.ndarray:
  """The normalized primary matrix (SMPTE RP 177)."""
  p = np.stack([_xyz(c) for c in primaries], axis=1)
  return p * np.linalg.solve(p, _xyz(white))


def _bradford(src: tuple[float, float], dst: tuple[float, float]) -> np.ndarray:
  a, b = BRADFORD @ _xyz(src), BRADFORD @ _xyz(dst)
  return np.linalg.inv(BRADFORD) @ np.diag(b / a) @ BRADFORD


# Registry ------------------------------------------------------------------------------------


def test_every_alias_round_trips():
  """Every alias parses to its space whatever its case and surrounding blanks; every interop ID
  parses to its member; codes round-trip in the frozen order."""
  for alias, space in ALIASES.items():
    assert colorspace.parse(alias) is space
    assert colorspace.parse(f"  {alias.upper()} ") is space
  for space in ColorSpace:
    assert colorspace.parse(space.value) is space
    assert colorspace.from_code(colorspace.code(space)) is space
  assert set(COLOR_SPACES) == set(ColorSpace)
  assert COLOR_SPACES[:2] == (ColorSpace.SRGB_REC709, ColorSpace.LIN_REC709)  # the former SRGB / LINEAR codes


def test_design_aliases():
  """The OCIO / MaterialX / MDL names of canonical_design.md §7.1."""
  expected = {
    ColorSpace.LIN_REC709: ("lin_rec709", "linear", "lin_srgb", "Linear Rec.709 (sRGB)"),
    ColorSpace.SRGB_REC709: ("srgb_texture", "srgb", "sRGB - Texture", "srgb_tx", "gamma_srgb"),
    ColorSpace.LIN_AP1: ("acescg", "ACEScg", "lin_ap1"),
    ColorSpace.LIN_AP0: ("aces2065-1", "lin_ap0"),
    ColorSpace.G22_REC709: ("g22_rec709",),
    ColorSpace.G18_REC709: ("g18_rec709",),
    ColorSpace.G24_REC709: ("g24_rec709",),
    ColorSpace.DATA: ("raw", "Raw", "none", "data", "Utility - Raw", "non-color"),
  }
  for space, names in expected.items():
    for name in names:
      assert colorspace.parse(name) is space, name


def test_unknown_and_context_names_parse_as_none():
  for name in ("", "auto", "sRGB - Display P3 (made up)", "gamma_default"):
    assert colorspace.parse(name) is None
  with pytest.raises(ValueError, match="not a known name"):
    colorspace.texture_space("bogus", where="t")
  with pytest.raises(ValueError, match="color space code"):
    colorspace.from_code(len(COLOR_SPACES))


def test_texture_space_rejects_output_spaces_and_accepts_wide_gamut():
  with pytest.raises(ValueError, match="outputs only"):
    colorspace.texture_space("srgb_rec709_display", where="add_texture")
  assert colorspace.texture_space("acescg", where="add_texture") is ColorSpace.LIN_AP1
  b = rw.SceneBuilder()
  with pytest.raises(ValueError, match="outputs only"):
    b.add_texture(np.zeros((2, 2, 4), np.uint8), colorspace=ColorSpace.SRGB_REC709_DISPLAY)
  assert b.add_texture(np.zeros((2, 2, 4), np.uint8), colorspace="ACEScg") >= 0


@pytest.mark.parametrize("name", sorted(PRIMARIES))
def test_matrices_match_their_chromaticities(name):
  """Each matrix is the primaries' NPM, Bradford-adapted to D65, within 1e-6."""
  primaries, white = PRIMARIES[name]
  derived = np.linalg.inv(_rgb_to_xyz(REC709, D65)) @ _bradford(white, D65) @ _rgb_to_xyz(primaries, white)
  np.testing.assert_allclose(np.asarray(getattr(colorspace, name)), derived, atol=1e-6)


def test_acescg_matrix_is_ocios():
  np.testing.assert_allclose(np.asarray(colorspace.ACESCG_TO_REC709), OCIO_ACESCG_TO_REC709, atol=1e-6)


@pytest.mark.parametrize("space", [s for s in ColorSpace if s not in colorspace.OUTPUT_ONLY])
def test_white_stays_white(space):
  np.testing.assert_allclose(colorspace.to_lin_rec709((1.0, 1.0, 1.0), space), (1.0, 1.0, 1.0), atol=1e-6)


def test_constants_transfers():
  """sRGB and pure-gamma transfers; linear and data unchanged; outputs rejected."""
  assert colorspace.to_lin_rec709((0.5, 0.2, 0.1), ColorSpace.LIN_REC709) == (0.5, 0.2, 0.1)
  assert colorspace.to_lin_rec709((0.5, 0.2, 0.1), ColorSpace.DATA) == (0.5, 0.2, 0.1)
  assert colorspace.to_lin_rec709((0.5, 0.5, 0.5), ColorSpace.G22_REC709) == pytest.approx((0.5**2.2,) * 3)
  assert colorspace.to_lin_rec709((0.5, 0.5, 0.5), ColorSpace.SRGB_REC709) == pytest.approx((0.21404114,) * 3)
  with pytest.raises(ValueError, match="outputs only"):
    colorspace.to_lin_rec709((0.5, 0.5, 0.5), ColorSpace.SRGB_REC709_DISPLAY)


@given(
  st.sampled_from([s for s in ColorSpace if s not in colorspace.OUTPUT_ONLY | {ColorSpace.DATA}]),
  st.tuples(*(st.floats(0.0, 4.0, width=32),) * 3),
)
def test_texels_agree_with_constants(space, rgb):
  """The per-texel conversion (float texels, unclamped) is the constants' conversion."""
  texels = np.asarray(rgb, np.float32).reshape(1, 1, 3)
  got = colorspace.linear_rec709(texels, space)[0, 0]
  r, g, b = (float(c) for c in texels[0, 0])
  np.testing.assert_allclose(got[:3], colorspace.to_lin_rec709((r, g, b), space), rtol=1e-5, atol=1e-6)
  assert got[3] == 1.0


def test_decode_clamps_and_rejects_data():
  saturated = np.array([[[0.0, 0.0, 1.0]]], np.float32)  # pure AP1 blue: negative red / green in Rec.709
  assert (colorspace.linear_rec709(saturated, ColorSpace.LIN_AP1)[..., :2] < 0.0).all()
  assert (colorspace.decode(saturated, ColorSpace.LIN_AP1) >= 0.0).all()
  with pytest.raises(ValueError, match="never converted"):
    colorspace.decode(saturated, ColorSpace.DATA)
  with pytest.raises(ValueError, match="outputs only"):
    colorspace.decode(saturated, ColorSpace.SRGB_REC709_DISPLAY)


# Upload --------------------------------------------------------------------------------------


def test_8bit_acescg_reencoded_within_one_lsb():
  """An 8-bit ACEScg texture becomes 8-bit sRGB within 1 LSB of a float64 reference; alpha kept."""
  rng = np.random.default_rng(7)
  texels = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
  done = converted_view(texels, ColorSpace.LIN_AP1)
  assert done.space is ColorSpace.SRGB_REC709
  assert done.texels.dtype == np.uint8
  linear = np.clip(texels[..., :3] / 255.0 @ np.asarray(OCIO_ACESCG_TO_REC709).T, 0.0, 1.0)
  encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1.0 / 2.4) - 0.055) * 255.0
  assert np.abs(done.texels[..., :3].astype(np.float64) - encoded).max() <= 1.0
  np.testing.assert_array_equal(done.texels[..., 3], texels[..., 3])
  assert done.negative and done.clipped  # random AP1 colors leave the Rec.709 gamut


def test_float_sources_stay_linear():
  texels = np.array([[[0.25, 0.5, 2.0, 1.0]]], np.float32)
  done = converted_view(texels, ColorSpace.LIN_REC2020)
  assert done.space is ColorSpace.LIN_REC709
  assert done.texels.dtype == np.float32
  np.testing.assert_allclose(done.texels[0, 0], colorspace.decode(texels, ColorSpace.LIN_REC2020)[0, 0])


def _rgb_recipe(texture: TextureId) -> tuple[Channel, Channel, Channel, Channel]:
  return (Channel(texture, 0), Channel(texture, 1), Channel(texture, 2), ONE)


def test_pool_converts_color_textures_like_their_srgb_equivalent():
  """A color recipe of an ACEScg texture stores the same texels as one of its sRGB re-encoding
  (one shared GPU texture); a data recipe of it reads the raw channels; the conversion is
  reported."""
  rng = np.random.default_rng(3)
  texels = rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)
  b = rw.SceneBuilder()
  aces = b.add_texture(texels, colorspace="acescg")
  srgb = b.add_texture(converted_view(texels, ColorSpace.LIN_AP1).texels, colorspace="srgb_texture")
  raw = b.add_texture(texels, colorspace="raw")
  report = LoadReport()
  with wp.ScopedDevice("cpu"):
    pool = TexturePool(b.build(), report, device=wp.get_device("cpu"))
    a, s = pool.get(_rgb_recipe(aces), color=True), pool.get(_rgb_recipe(srgb), color=True)
    data_aces, data_raw = pool.get(_rgb_recipe(aces), color=False), pool.get(_rgb_recipe(raw), color=False)
    pool.finish()
  assert pool.textures[a] is pool.textures[s]
  assert pool.layouts[a] == pool.layouts[s]
  assert pool.textures[data_aces] is pool.textures[data_raw]  # data roles ignore the space
  keys = set(report.approximated)
  assert any(k.startswith("gpu.texture.colorspace:lin_ap1_scene") for k in keys), keys
  assert "gpu.texture.colorspace.gamut_clamped" in keys


def test_pool_shares_linear_and_data_sources():
  """One image registered as LIN_REC709 and as DATA (stored alike) is one core texture."""
  b = rw.SceneBuilder()
  img = np.full((4, 4, 4), 128, np.uint8)
  lin, data = b.add_texture(img, colorspace="linear"), b.add_texture(img, colorspace="raw")
  assert lin != data
  with wp.ScopedDevice("cpu"):
    pool = TexturePool(b.build(), LoadReport(), device=wp.get_device("cpu"))
    assert pool.get(_rgb_recipe(lin), color=False) == pool.get(_rgb_recipe(data), color=False)
    assert pool.get(_rgb_recipe(lin), color=True) == pool.get(_rgb_recipe(data), color=True)


def test_stored_as_is_textures_are_not_converted():
  b = rw.SceneBuilder()
  b.add_texture(np.full((4, 4, 4), 200, np.uint8), colorspace="srgb")
  report = LoadReport()
  with wp.ScopedDevice("cpu"):
    pool = TexturePool(b.build(), report, device=wp.get_device("cpu"))
    pool.get(_rgb_recipe(TextureId(0)), color=True)
    pool.finish()
  assert not any("colorspace" in k for k in report.approximated)


# Importers and outputs -----------------------------------------------------------------------


def test_auto_colorspace():
  assert auto_colorspace("albedo.png", color=True) is ColorSpace.SRGB_REC709
  assert auto_colorspace("sky.exr", color=True) is ColorSpace.LIN_REC709
  assert auto_colorspace("normal.png", color=False) is ColorSpace.DATA


def test_usd_colorspace_names():
  """USD sourceColorSpace tokens / colorSpace metadata go through the registry; an unknown name is
  reported and warned, and reads as before the registry ("lin_*" linear, others unauthored)."""
  pytest.importorskip("pxr")
  from raywarp.formats.usd import util  # pxr is optional

  report = LoadReport()

  def of(token: str) -> ColorSpace | None:
    return util.colorspace_of(None, token, report=report, where="/M/Tex")

  assert of("auto") is None
  assert of("") is None
  assert of("sRGB") is ColorSpace.SRGB_REC709
  assert of("raw") is ColorSpace.DATA
  assert of("lin_ap1_scene") is ColorSpace.LIN_AP1
  assert not report.approximated
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert of("Made Up Space") is None  # unauthored: sRGB for 8-bit color images, as before
    assert of("lin_made_up") is ColorSpace.LIN_REC709  # "lin_*" read linear, as before
  assert any("Made Up Space" in str(w.message) for w in caught)
  assert set(report.approximated) == {
    "texture.colorspace.unknown:Made Up Space(read as unauthored)",
    "texture.colorspace.unknown:lin_made_up(read linear Rec.709)",
  }


def test_usd_preview_reports_unknown_names(tmp_path):
  """A UsdUVTexture with an unknown sourceColorSpace is reported where it is read."""
  pytest.importorskip("pxr")
  from pxr import Sdf, UsdShade  # pxr is optional
  from rw_helpers import bind, new_stage, quad, write_png  # USD helpers

  import raywarp.formats.usd  # pxr is optional

  stage = new_stage(tmp_path)
  write_png(tmp_path / "base.png", np.full((2, 2, 3), 128, np.uint8))
  mat = UsdShade.Material.Define(stage, "/Looks/M")
  shader = UsdShade.Shader.Define(stage, "/Looks/M/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Base")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("base.png")
  tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("Studio Weird")
  tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
  mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
  bind(quad(stage, "/A"), mat)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    builder = raywarp.formats.usd.read(stage)
  assert "texture.colorspace.unknown:Studio Weird(read as unauthored)" in builder.report.approximated
  s = builder.build()
  assert colorspace.from_code(int(s.textures.colorspace[0])) is ColorSpace.SRGB_REC709  # the old "auto"


# The color space names of OCIO's ACES configs and MaterialX this registry must read (review).
OCIO_NAMES = {
  "ACES - ACEScg": ColorSpace.LIN_AP1,
  "ACES - ACES2065-1": ColorSpace.LIN_AP0,
  "Utility - sRGB - Texture": ColorSpace.SRGB_REC709,
  "Input - Generic - sRGB - Texture": ColorSpace.SRGB_REC709,
  "sRGB Encoded Rec.709 (sRGB)": ColorSpace.SRGB_REC709,
  "Utility - Linear - sRGB": ColorSpace.LIN_REC709,
  "Utility - Linear - Rec.709": ColorSpace.LIN_REC709,
  "Linear Rec.709 (sRGB)": ColorSpace.LIN_REC709,
  "Linear Rec.2020": ColorSpace.LIN_REC2020,
  "Utility - Linear - Rec.2020": ColorSpace.LIN_REC2020,
  "Linear P3-D65": ColorSpace.LIN_P3D65,
  "Utility - Linear - P3-D65": ColorSpace.LIN_P3D65,
  "sRGB Encoded P3-D65": ColorSpace.SRGB_P3D65,
  "Gamma 2.2 Rec.709 - Texture": ColorSpace.G22_REC709,
  "Utility - Gamma 2.2 - Rec.709 - Texture": ColorSpace.G22_REC709,
  "Gamma 2.2 Encoded Rec.709": ColorSpace.G22_REC709,
  "Gamma 1.8 Encoded Rec.709": ColorSpace.G18_REC709,
  "Gamma 2.4 Encoded Rec.709": ColorSpace.G24_REC709,
  "lin_displayp3": ColorSpace.LIN_P3D65,
  "srgb_displayp3": ColorSpace.SRGB_P3D65,
  "Utility - Raw": ColorSpace.DATA,
}


@pytest.mark.parametrize(("name", "space"), sorted(OCIO_NAMES.items()))
def test_ocio_config_names(name, space):
  assert colorspace.parse(name) is space


def test_mdl_bake_decodes_wide_gamut_sources(tmp_path):
  """An MDL bake reads a texture authored in g22_rec709 (USD colorSpace metadata) through its
  transfer, not as raw linear values."""
  pytest.importorskip("pxr")
  from pxr import Sdf, UsdShade  # pxr is optional
  from rw_helpers import bind, mdl_material, new_stage, quad, write_png  # USD helpers

  import raywarp.formats.usd  # pxr is optional

  stage = new_stage(tmp_path)
  img = np.full((4, 4), 128, np.uint8)
  img[:, 2:] = 0
  write_png(tmp_path / "t.png", img)
  (tmp_path / "G.mdl").write_text(
    "mdl 1.6;\nimport ::df::*;\nimport ::math::*;\nimport ::state::*;\nimport ::tex::*;\n"
    "export material G(uniform texture_2d T = texture_2d())\n = let {\n"
    "  float R = math::sin(tex::lookup_float4(T, float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y)).x);\n"
    "} in material(surface: material_surface(scattering: df::custom_curve_layer(normal_reflectivity: 0.04,\n"
    "  layer: df::microfacet_ggx_vcavities_bsdf(roughness_u: R * R, roughness_v: R * R), base: df::diffuse_reflection_bsdf())));\n"
  )
  mat = mdl_material(stage, "/Looks/G", "G.mdl", "G", T=(Sdf.ValueTypeNames.Asset, Sdf.AssetPath("t.png")))
  UsdShade.Shader(stage.GetPrimAtPath("/Looks/G/Shader")).GetInput("T").GetAttr().SetColorSpace("g22_rec709")
  bind(quad(stage, "/A"), mat)
  stage.Save()
  s = raywarp.formats.usd.read(tmp_path / "scene.usda").build()
  value = s.images.level(int(s.textures.image[s.materials.tex[0, rw.scene.Slot.SPECULAR_ROUGHNESS]]))[0, 0, 0]
  # Baked (sin is not affine): the SDK reads the texel through g22's transfer, not as raw linear.
  assert abs(int(value) - round(255 * np.sin((128 / 255) ** 2.2))) <= 1


def test_wide_gamut_above_one_clipped_for_color_kept_for_emission():
  """A saturated 8-bit ACEScg red converts above 1: color recipes clip it (reported apart),
  emission recipes keep it as linear float."""
  b = rw.SceneBuilder()
  red = np.zeros((2, 2, 4), np.uint8)
  red[..., 0], red[..., 3] = 255, 255
  aces = b.add_texture(red, colorspace="acescg")
  srgb = b.add_texture(red, colorspace="srgb")
  report = LoadReport()
  with wp.ScopedDevice("cpu"):
    pool = TexturePool(b.build(), report, device=wp.get_device("cpu"))
    base = pool.get(_rgb_recipe(aces), color=True)
    emission = pool.get(_rgb_recipe(aces), color=True, hdr=True)
    assert base != emission
    assert pool.get(_rgb_recipe(srgb), color=True) == pool.get(_rgb_recipe(srgb), color=True, hdr=True)
    pool.finish()
  assert "gpu.texture.colorspace.gamut_clamped(above 1 clipped)" in report.approximated
  kept = converted_view(red, ColorSpace.LIN_AP1, hdr=True)
  assert kept.space is ColorSpace.LIN_REC709 and not kept.clipped
  assert kept.texels[0, 0, 0] == pytest.approx(colorspace.ACESCG_TO_REC709[0][0], rel=1e-6)


def test_copied_channels_never_alias_the_source():
  """A 1-row image (its flip is contiguous) with an inverted channel leaves the source intact."""
  from raywarp.core.pipeline.textures.pool import _copied_channels  # a private helper

  img = np.full((1, 4, 4), 200, np.uint8)
  t = TextureId(0)
  recipe = (Channel(t, 0, invert=True), Channel(t, 1), Channel(t, 2), ONE)
  copied = _copied_channels(recipe, img, ColorSpace.LIN_REC709, color=False)
  assert copied is not None
  out, _ = copied
  assert (out[..., 0] == 55).all()
  assert (img == 200).all()


def test_stored_output_space_code_is_rejected():
  """A Textures row holding srgb_rec709_display (a corrupt .wpbr) fails at upload."""
  b = rw.SceneBuilder()
  b.add_texture(np.zeros((2, 2, 4), np.uint8))
  scene = b.build()
  scene.textures.colorspace[0] = colorspace.code(ColorSpace.SRGB_REC709_DISPLAY)
  with pytest.raises(ValueError, match="outputs only"):
    TexturePool(scene, LoadReport(), device=wp.get_device("cpu"))


def test_environment_and_emissive_decode_converted_spaces():
  """Environment radiance and the emissive lights' texels read a g22 texture through its transfer
  (reported)."""
  from raywarp.core.pipeline.upload import environment  # private helpers under test
  from raywarp.scene.lighting import emissive

  b = rw.SceneBuilder()
  tex = b.add_texture(np.full((2, 4, 3), 128, np.uint8), colorspace="g22_rec709")
  scene = b.build()
  report = LoadReport()
  rgb = environment._map_radiance(scene, tex, report)
  np.testing.assert_allclose(rgb, (128 / 255) ** 2.2, rtol=1e-5)
  assert "gpu.environment.colorspace:g22_rec709_scene(converted to Rec.709 at upload)" in report.approximated
  texels = emissive._texels(scene, tex, {})
  np.testing.assert_allclose(texels[..., :3], (128 / 255) ** 2.2, rtol=1e-5)
  assert "emissive_light.texture.colorspace:g22_rec709_scene(converted to Rec.709)" in scene.report.approximated


def test_output_color_spaces():
  rgba = wp.empty((1, 1, 2, 2), dtype=wp.vec4ub, device="cpu")
  depth = wp.empty((1, 1, 2, 2), dtype=wp.float32, device="cpu")
  hdr = wp.empty((1, 1, 2, 2), dtype=wp.vec3, device="cpu")
  outputs = RenderOutputs(rgb=rgba, depth=depth, hdr=hdr)
  assert outputs.color_spaces == {
    "rgb": ColorSpace.SRGB_REC709_DISPLAY,
    "depth": ColorSpace.DATA,
    "hdr": ColorSpace.LIN_REC709,
  }
  assert set(OUTPUT_COLOR_SPACES) == set(VIEWS)
  assert OUTPUT_COLOR_SPACES["luminance"] == ColorSpace.SRGB_REC709_DISPLAY
  assert OUTPUT_COLOR_SPACES["direct"] == ColorSpace.LIN_REC709 and OUTPUT_COLOR_SPACES["uv"] == ColorSpace.DATA


def test_linear_to_srgb_inverts_the_decode():
  """Re-encoding a decoded 8-bit sRGB texture is the identity (no drift on re-upload)."""
  codes = np.arange(256).astype(np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
  np.testing.assert_array_equal(linear_to_srgb_u8(colorspace.decode(codes, ColorSpace.SRGB_REC709)[..., :3]), codes)


def test_float_texels_tagged_srgb_are_decoded_like_other_spaces():
  """One rule for a space's transfer, whatever the primaries: float texels explicitly tagged
  SRGB_REC709 are decoded (they were read as linear while P3 / gamma-tagged floats were decoded);
  8-bit SRGB_REC709 and LIN_REC709 floats stay stored as is."""
  from raywarp.core.pipeline.upload.environment import _map_radiance

  assert not colorspace.stored_as_is(ColorSpace.SRGB_REC709, float_texels=True)
  assert colorspace.stored_as_is(ColorSpace.SRGB_REC709, float_texels=False)
  assert colorspace.stored_as_is(ColorSpace.LIN_REC709, float_texels=True)
  b = rw.SceneBuilder()
  half = np.full((4, 8, 3), 0.5, np.float32)
  srgb = b.add_texture(half, colorspace=ColorSpace.SRGB_REC709)
  p3 = b.add_texture(half, colorspace=ColorSpace.SRGB_P3D65)
  scene = b.build()
  decoded = _map_radiance(scene, srgb, LoadReport())
  np.testing.assert_allclose(decoded, 0.21404114, rtol=1e-5)  # the sRGB curve at 0.5
  grey = _map_radiance(scene, p3, LoadReport())  # neutral grey: P3 primaries keep it grey
  np.testing.assert_allclose(grey, decoded, rtol=1e-4)
