"""OpenEXR images (`scene/images/exr.py`, the OpenEXR package) against files written by an
independent writer (`rw_helpers.write_exr`) and by the package itself (PIZ), and Radiance header
checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rw_helpers import bind, new_stage, preview_material, quad, write_exr

import raywarp as rw
import raywarp.formats.usd
from raywarp.scene.images import colorspace, exr, hdr
from raywarp.scene.images import decode as imaging
from raywarp.scene.model.types import PixelType

pytest.importorskip("OpenEXR")


@pytest.mark.parametrize(("compression", "half"), [(0, True), (1, False), (2, False), (3, True)])
def test_exr_decodes_linear_hdr(tmp_path, compression, half):
  rng = np.random.default_rng(0)
  px = rng.uniform(0.0, 40.0, (37, 21, 4)).astype(np.float32)  # HDR values, odd sizes
  if half:
    px = px.astype(np.float16).astype(np.float32)
  path = write_exr(tmp_path / "t.exr", px, compression=compression, half=half)
  np.testing.assert_allclose(exr.decode(str(path)), px, rtol=1e-6)
  info = imaging.probe(imaging.ImageSource(path=str(path)))
  assert (info.width, info.height, info.channels) == (21, 37, 4)
  assert info.pixel == (PixelType.F16 if half else PixelType.F32)
  # Through the builder: an emissive EXR is linear ("auto" color space) and keeps its range.
  pytest.importorskip("pxr")
  stage = new_stage(tmp_path)
  mat, shader = preview_material(stage, "/Looks/M")
  from pxr import Sdf, UsdShade

  tex = UsdShade.Shader.Define(stage, "/Looks/M/E")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("t.exr")
  shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  s = builder.build()
  assert colorspace.from_code(int(s.textures.colorspace[0])) == rw.scene.ColorSpace.LIN_REC709
  np.testing.assert_allclose(s.images.level(0).astype(np.float32), px, rtol=1e-3)
  assert not builder.report.dropped


def test_exr_grey_channel(tmp_path):
  y = np.linspace(0, 5, 12, dtype=np.float32).reshape(3, 4)
  path = write_exr(tmp_path / "y.exr", y, compression=3, names="Y")
  assert exr.read_header(str(path)).channels == ("Y",)
  np.testing.assert_allclose(exr.decode(str(path))[..., 0], y.astype(np.float16).astype(np.float32))


def test_exr_without_the_openexr_package_is_rejected_at_probe(tmp_path, monkeypatch):
  """Without OpenEXR: the probe fails (no decode scheduled) with its own report key, naming the
  package."""
  path = write_exr(tmp_path / "y.exr", np.ones((2, 2), np.float32), compression=3, names="Y")
  find_spec = importlib.util.find_spec
  monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "OpenEXR" else find_spec(name, *a))
  info = imaging.probe(imaging.ImageSource(path=str(path)))
  assert imaging.report_key(info.error) == "texture.exr.openexr_missing"
  assert "pip install openexr" in info.error


def _openexr_piz(path, pixels: np.ndarray) -> None:
  """A PIZ-compressed RGBA half file written by the OpenEXR package."""
  import OpenEXR

  header = {"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.scanlineimage}
  with OpenEXR.File(header, {"RGBA": pixels.astype(np.float16)}) as f:
    f.write(str(path))


def test_exr_piz_from_a_file_and_a_usdz_package(tmp_path):
  """PIZ (the HDRIs UF found), from a file and from inside a `.usdz` package (its bytes); the
  texels are the file's."""
  import zipfile

  px = np.random.default_rng(1).uniform(0.0, 20.0, (9, 13, 4)).astype(np.float16).astype(np.float32)
  path = tmp_path / "piz.exr"
  _openexr_piz(path, px)
  np.testing.assert_array_equal(exr.decode(str(path)), px)
  with zipfile.ZipFile(tmp_path / "p.usdz", "w", zipfile.ZIP_STORED) as z:
    z.write(path, "env/piz.exr")
  source = imaging.ImageSource(path=f"{tmp_path / 'p.usdz'}[env/piz.exr]", transfer="linear")
  info = imaging.probe(source)
  assert info.pixel == PixelType.F16
  base, error = imaging.decode_base(source, info)
  assert not error, error
  np.testing.assert_array_equal(base.astype(np.float32), px)


def test_a_truncated_exr_is_a_reported_failure(tmp_path: Path) -> None:
  """A file cut after its header probes or decodes to a reported failure (mid-grey), never to
  uninitialised texels."""
  path = write_exr(tmp_path / "cut.exr", np.ones((32, 8, 3), np.float32), compression=0, names="RGB")
  path.write_bytes(path.read_bytes()[:-64])
  source = imaging.ImageSource(path=str(path), transfer="linear")
  info = imaging.probe(source)
  base, error = imaging.decode_base(source, info)
  assert error or info.error
  assert np.all(base.astype(np.float32) == 0.5)


# ---------------------------------------------------------------------------------------------
# Reject unsupported Radiance color encodings before probing or allocating pixels.


@pytest.mark.parametrize(
  ("header", "message"),
  [
    (b"#?RADIANCE\n\n-Y 1 +X 1\n", "FORMAT"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n", "FORMAT"),
    (b"#?RADIANCE\nFORMAT=unknown\n\n-Y 1 +X 1\n", "FORMAT"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n", "FORMAT"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 0 +X 1\n", "positive"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 0\n", "positive"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1garbage\n", "oriented"),
  ],
)
def test_probe_and_decode_reject_invalid_header(tmp_path: Path, header: bytes, message: str) -> None:
  """Metadata and pixel decoding share validation; XYZE must never silently become RGB."""
  path = tmp_path / "bad.hdr"
  path.write_bytes(header + bytes([128, 64, 32, 129]))
  with pytest.raises(ValueError, match=message):
    hdr.size(str(path))
  with pytest.raises(ValueError, match=message):
    hdr.read_hdr(path)


def test_layered_exr_channels_come_from_one_layer():
  """Layered channel names pick one layer holding every wanted component, never a mix (R and G
  of one layer, B of another)."""
  names = ["a.R", "a.G", "a.B", "b.R", "b.G"]
  assert exr._selected(names) == ["a.R", "a.G", "a.B"]
  assert exr._selected(["diffuse.R", "diffuse.G", "diffuse.B", "diffuse.A"]) == [f"diffuse.{c}" for c in "RGBA"]


def test_derived_image_rejects_a_source_that_is_not_a_file():
  """A derived image is computed from image files: an array source is rejected by name, not
  reported as a missing file "" (and dropped)."""
  array = imaging.ImageSource(array=np.zeros((2, 2, 4), np.uint8))
  derived = imaging.Derived(sources=(array,), fn=lambda inputs: inputs[0], channels=4, encoding="srgb", key=("k",))
  b = rw.SceneBuilder()
  with pytest.raises(ValueError, match="every source must be an image file"):
    b.add_derived_image(derived)
  assert not b.report.dropped


def test_decode_linear_resamples_by_box_or_nearest():
  """`decode_linear`: an integer factor on both axes box-filters, any other size takes the nearest
  texels."""
  values = np.arange(16).astype(np.float32).reshape(4, 4) / 16.0
  source = imaging.ImageSource(array=values, transfer="linear")
  box = imaging.decode_linear(source, 2, 2)
  np.testing.assert_allclose(box[..., 0], values.reshape(2, 2, 2, 2).mean(axis=(1, 3)), atol=1e-3)
  nearest = imaging.decode_linear(source, 3, 2)
  np.testing.assert_allclose(nearest[..., 0], values[np.ix_([0, 2], [0, 1, 2])], atol=1e-3)
