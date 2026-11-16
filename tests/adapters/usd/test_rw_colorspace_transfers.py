"""8-bit transfers (`scene/images/mips.py`, `scene/images/colorspace.py`): the Adobe RGB / gamma AP1
spaces, host mips filtered in linear for every transfer, the one decode table, the transfer
argument."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import raywarp as rw
from raywarp.scene.images import colorspace, mips
from raywarp.scene.images.colorspace import ColorSpace


def test_adobe_rgb_and_gamma_ap1_spaces():
  """Adobe RGB (1998): gamma 563/256 and its primaries; linear Adobe RGB: the primaries only;
  g22_ap1: gamma 2.2 then ACEScg's matrix. The MaterialX names parse to them, and their codes
  come after the existing ones (stored codes unchanged)."""
  assert colorspace.parse("adobergb") is ColorSpace.G22_ADOBERGB
  assert colorspace.parse("lin_adobergb") is ColorSpace.LIN_ADOBERGB
  assert colorspace.parse("g22_ap1") is ColorSpace.G22_AP1
  assert [colorspace.code(s) for s in (ColorSpace.G22_ADOBERGB, ColorSpace.LIN_ADOBERGB, ColorSpace.G22_AP1)] == [12, 13, 14]
  rgb = (0.25, 0.5, 0.75)
  adobe = np.asarray(colorspace.ADOBERGB_TO_REC709)
  gamma = np.asarray(rgb) ** (563.0 / 256.0)
  np.testing.assert_allclose(colorspace.to_lin_rec709(rgb, ColorSpace.G22_ADOBERGB), adobe @ gamma, rtol=1e-12)
  np.testing.assert_allclose(colorspace.to_lin_rec709(rgb, ColorSpace.LIN_ADOBERGB), adobe @ np.asarray(rgb), rtol=1e-12)
  ap1 = np.asarray(colorspace.ACESCG_TO_REC709) @ (np.asarray(rgb) ** 2.2)
  np.testing.assert_allclose(colorspace.to_lin_rec709(rgb, ColorSpace.G22_AP1), ap1, rtol=1e-12)
  # Adobe RGB green (0, 1, 0) is outside Rec.709: negative red once converted.
  assert colorspace.to_lin_rec709((0.0, 1.0, 0.0), ColorSpace.LIN_ADOBERGB)[0] < 0.0


@pytest.mark.parametrize("space", [ColorSpace.SRGB_REC709, ColorSpace.SRGB_P3D65])
def test_host_mips_of_srgb_transfer_spaces_filter_in_linear(space):
  """Host mips of an 8-bit texture in an sRGB-transfer space average decoded values (LCS open
  item 4): black and white average to sRGB 188 (linear 0.5), not the encoded mean 128."""
  b = rw.SceneBuilder()
  pixels = np.zeros((2, 2, 4), np.uint8)
  pixels[:, 1] = 255
  pixels[..., 3] = 255
  b.add_texture(pixels, colorspace=space)
  images = b.build(mips=True).images
  assert int(images.levels[0]) == 2
  np.testing.assert_array_equal(images.level(0, 1)[0, 0], [188, 188, 188, 255])


@pytest.mark.parametrize("space", [ColorSpace.G22_REC709, ColorSpace.G18_REC709, ColorSpace.G22_ADOBERGB, ColorSpace.G22_AP1])
def test_host_mips_of_pure_gamma_spaces_filter_in_linear(space):
  """Pure-gamma 8-bit textures too: black and white average to the gamma encoding of 0.5 (they
  were averaged encoded, 128, so emissive light fitting read a darker mip)."""
  b = rw.SceneBuilder()
  pixels = np.zeros((2, 2, 4), np.uint8)
  pixels[:, 1] = 255
  pixels[..., 3] = 255
  b.add_texture(pixels, colorspace=space)
  images = b.build(mips=True).images
  gamma = colorspace.texel_transfer(space)
  assert isinstance(gamma, float)
  expected = int(0.5 ** (1.0 / gamma) * 255.0 + 0.5)
  np.testing.assert_array_equal(images.level(0, 1)[0, 0], [expected, expected, expected, 255])


def test_every_8bit_srgb_decode_is_the_one_exact_table():
  """The conversion (`colorspace`), the host decode / mips (`images.linear_rgba`) and the core's
  texture formats decode an 8-bit sRGB code to the same float32: the float64 transfer rounded
  once (the conversion's float32 arithmetic differed by up to 1.8e-7 on 214 codes)."""
  from raywarp.core.pipeline.textures import formats
  from raywarp.scene.images import decode as images

  codes = np.arange(256).astype(np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
  x = np.arange(256) / 255.0
  exact = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32).reshape(16, 16)
  np.testing.assert_array_equal(colorspace.linear_rec709(codes, ColorSpace.SRGB_REC709)[..., 0], exact)
  np.testing.assert_array_equal(images.linear_rgba(codes, transfer="srgb")[..., 0], exact)
  np.testing.assert_array_equal(formats.decode_linear(codes, lambda c: True)[..., 0], exact)
  gamma = colorspace.linear_rec709(codes, ColorSpace.G22_REC709)[..., 0]
  np.testing.assert_array_equal(gamma, (x**2.2).astype(np.float32).reshape(16, 16))


def test_the_shared_decode_tables_are_read_only():
  """`SRGB_TO_LINEAR`, the gamma tables and the core's float16 table are shared: no caller can
  write them; the float16 one is the same half per code as the float64 transfer's."""
  from raywarp.core.pipeline.textures import formats

  for table in (mips.SRGB_TO_LINEAR, mips.gamma_lut(2.2), formats.SRGB_F16):
    with pytest.raises(ValueError, match="read-only"):
      table[0] = 1.0
  x = np.arange(256) / 255.0
  exact16 = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float16)
  np.testing.assert_array_equal(formats.SRGB_F16.view(np.uint16), exact16.view(np.uint16))


@pytest.mark.parametrize("transfer", ["gamma", 0.0, -2.2, float("nan"), float("inf"), True])
def test_add_image_rejects_a_bad_transfer(transfer):
  with pytest.raises(ValueError, match="transfer"):
    rw.SceneBuilder().add_image(np.zeros((2, 2, 4), np.uint8), transfer=transfer)


@pytest.mark.parametrize(("transfer", "expected"), [("linear", 128), ("srgb", 188), (2.2, 186)])
def test_downsample_filters_each_transfer_in_linear(transfer, expected):
  """Black and white average to the transfer's encoding of 0.5; alpha stays linear."""
  img = np.zeros((1, 2, 2), np.uint8)
  img[0, 1] = 255
  assert mips.downsample(img, transfer=transfer)[0, 0].tolist() == [expected, 128]


def test_float_srgb_decode_of_negative_values_warns_nothing():
  """Float texels tagged sRGB below the curve's domain (< -0.055) follow the linear segment without
  a NaN power warning; the result matches the piecewise curve."""
  x = np.array([-1.0, -0.06, 0.0, 0.04045, 0.5, 2.0], np.float32)
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    out = mips.srgb_decode(x)
  assert out.dtype == np.float32
  np.testing.assert_array_equal(out[:4], x[:4] / 12.92)
  np.testing.assert_allclose(out[4:], ((x[4:] + 0.055) / 1.055) ** 2.4, rtol=1e-6)
