"""Helpers for the raywarp scene / USD adapter tests: tiny image writers (no Pillow needed) and
stage builders (quads, UsdPreviewSurface / MDL materials)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, pixels: np.ndarray) -> Path:
  """Writes an 8-bit grey / RGB / RGBA PNG (filter 0) with the standard library only."""
  pixels = np.asarray(pixels, dtype=np.uint8)
  if pixels.ndim == 2:
    pixels = pixels[..., None]
  h, w, c = pixels.shape
  color_type = {1: 0, 2: 4, 3: 2, 4: 6}[c]
  raw = b"".join(b"\0" + pixels[y].tobytes() for y in range(h))

  def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(data, zlib.crc32(kind)))

  # Several IDAT chunks, as real encoders write.
  stream = zlib.compress(raw)
  idats = b"".join(chunk(b"IDAT", stream[i : i + 64]) for i in range(0, len(stream), 64))
  ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
  path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + idats + chunk(b"IEND", b""))
  return path


def write_hdr(path: Path, rgb: np.ndarray) -> Path:
  """Writes a flat (non run-length) Radiance RGBE file."""
  rgb = np.asarray(rgb, dtype=np.float64)
  h, w, _ = rgb.shape
  peak = rgb.max(axis=2)
  mantissa, exponent = np.frexp(peak)
  scale = np.where(peak > 1e-32, mantissa * 256.0 / np.maximum(peak, 1e-32), 0.0)
  rgbe = np.zeros((h, w, 4), np.uint8)
  rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
  rgbe[..., 3] = np.where(peak > 1e-32, exponent + 128, 0).astype(np.uint8)
  path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode() + rgbe.tobytes())
  return path


def new_stage(tmp_path: Path, name: str = "scene.usda", up: str = "Z", mpu: float = 1.0):
  """An empty stage saved under tmp_path (Z-up, metres by default)."""
  from pxr import Usd, UsdGeom

  stage = Usd.Stage.CreateNew(str(tmp_path / name))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
  UsdGeom.SetStageMetersPerUnit(stage, mpu)
  return stage


def quad(stage, path: str, size: float = 1.0):
  """A unit quad (one 4-gon) in the XY plane, counter-clockwise seen from +Z."""
  from pxr import UsdGeom

  mesh = UsdGeom.Mesh.Define(stage, path)
  s = size
  mesh.CreatePointsAttr([(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0)])
  mesh.CreateFaceVertexCountsAttr([4])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
  mesh.CreateSubdivisionSchemeAttr("none")
  return mesh


def preview_material(stage, path: str, color=(0.2, 0.4, 0.6), **inputs):
  """A UsdPreviewSurface material with a constant diffuse color (plus extra float inputs)."""
  from pxr import Sdf, UsdShade

  mat = UsdShade.Material.Define(stage, path)
  shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
  for name, value in inputs.items():
    shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
  mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
  return mat, shader


def mdl_material(stage, path: str, asset: str, name: str, **inputs):
  """A material whose MDL shader is `name` of module `asset`, with typed inputs
  (`key=(Sdf type, value)`)."""
  from pxr import Sdf, UsdShade

  mat = UsdShade.Material.Define(stage, path)
  shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
  shader.SetSourceAsset(Sdf.AssetPath(asset), "mdl")
  shader.SetSourceAssetSubIdentifier(name, "mdl")
  for key, (type_name, value) in inputs.items():
    shader.CreateInput(key, type_name).Set(value)
  mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
  return mat


def bind(prim, material) -> None:
  from pxr import UsdShade

  UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(material)


def material_named(builder, name: str):
  """The builder's material converted from the USD material `name` (its full prim path)."""
  return next(m for m in builder.materials if m.name == name)


def quadrant_png(path: Path) -> Path:
  """An 8 x 8 PNG of four colored quadrants: top-left red, top-right green, bottom-left blue,
  bottom-right white (image rows top down)."""
  img = np.zeros((8, 8, 3), np.uint8)
  img[:4, :4] = (255, 0, 0)
  img[:4, 4:] = (0, 255, 0)
  img[4:, :4] = (0, 0, 255)
  img[4:, 4:] = 255
  return write_png(path, img)


def write_exr(path: Path, pixels: np.ndarray, compression: int = 3, half: bool = True, names: str = "RGBA") -> Path:
  """Writes a scan-line OpenEXR file (compression 0 NONE, 2 ZIPS or 3 ZIP; HALF or FLOAT)."""
  pixels = np.asarray(pixels, np.float32)
  pixels = pixels[..., None] if pixels.ndim == 2 else pixels
  h, w, c = pixels.shape
  chans = sorted(zip(names[:c], range(c), strict=True))  # EXR stores channels alphabetically
  ptype, dtype = (1, "<f2") if half else (2, "<f4")

  def attr(name: str, kind: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(value)) + value

  chlist = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", ptype, 0, 1, 1) for n, _ in chans) + b"\0"
  box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
  header = (
    b"\x76\x2f\x31\x01"
    + struct.pack("<I", 2)
    + attr("channels", "chlist", chlist)
    + attr("compression", "compression", bytes([compression]))
    + attr("dataWindow", "box2i", box)
    + attr("displayWindow", "box2i", box)
    + attr("lineOrder", "lineOrder", b"\0")
    + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    + attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
    + attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    + b"\0"
  )
  lines = 16 if compression == 3 else 1
  chunks = []
  for y0 in range(0, h, lines):
    raw = b"".join(pixels[y, :, i].astype(dtype).tobytes() for y in range(y0, min(y0 + lines, h)) for _, i in chans)
    data = raw
    if compression in (2, 3):
      b = np.frombuffer(raw, np.uint8)
      t = np.concatenate([b[0::2], b[1::2]]).astype(np.int64)  # interleave, then byte deltas
      d = t.copy()
      d[1:] = (t[1:] - t[:-1] + 128) & 0xFF
      data = zlib.compress(d.astype(np.uint8).tobytes())
      if len(data) >= len(raw):
        data = raw
    chunks.append(struct.pack("<ii", y0, len(data)) + data)
  table_end = len(header) + 8 * len(chunks)
  offsets, pos = [], table_end
  for ch in chunks:
    offsets.append(pos)
    pos += len(ch)
  path.write_bytes(header + struct.pack(f"<{len(offsets)}Q", *offsets) + b"".join(chunks))
  return path
