"""Format-neutral scene: SceneBuilder dedupe / tables, .wpbr round trip, images, geometry."""

from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
from typing import ClassVar

import numpy as np
import pytest
from rw_helpers import write_hdr, write_png

import raywarp as rw
from raywarp.core.pipeline.textures import formats
from raywarp.core.pipeline.upload.lights import derive_lights
from raywarp.scene.images import decode as imaging
from raywarp.scene.images import mips
from raywarp.scene.model import geometry, wpbr, xform
from raywarp.scene.model.table import Table
from raywarp.scene.model.types import NO_MESH, NO_TEXTURE
from raywarp.scene.model.wpbr import read_header

TRI = np.array([[0, 1, 2]])
POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)


def test_import_boundary():
  code = "import sys, raywarp; bad = {'pxr', 'mujoco', 'newton'} & set(sys.modules); assert not bad, bad"
  subprocess.run([sys.executable, "-c", code], check=True)


def test_mesh_dedupe_and_parts():
  b = rw.SceneBuilder()
  a = b.add_mesh(POINTS, TRI)
  assert b.add_mesh(POINTS.copy(), TRI.copy()) == a
  assert b.add_mesh(POINTS * 2, TRI) != a
  quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
  parts = b.add_mesh_parts(quad, [[0, 1, 2], [0, 2, 3], [0, 1, 3]], tri_part=[1, 0, 1], nparts=3)
  assert parts[2] == -1 and parts[0] >= 0 and parts[1] >= 0
  s = b.build()
  m0, m1 = s.meshes.get(parts[0]), s.meshes.get(parts[1])
  assert s.meshes.vert_adr[parts[0]] == s.meshes.vert_adr[parts[1]]  # shared vertices
  assert len(m0.tris) == 1 and len(m1.tris) == 2
  np.testing.assert_array_equal(m0.tris, [[0, 2, 3]])
  assert s.meshes.normal_num[a] == 0 and np.all(s.meshes.normal_idx == -1)


def test_corner_indexed_attributes():
  b = rw.SceneBuilder()
  normals = np.array([[0, 0, 1], [0, 0, -1]], np.float32)
  uvs = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]], np.float32)
  mid = b.add_mesh(POINTS, TRI, normals=normals, normal_idx=[[1, 1, 1]], uvs=uvs, uv_idx=[[3, 1, 2]])
  m = b.build().meshes.get(mid)
  np.testing.assert_array_equal(m.normals[m.normal_idx[0]], [[0, 0, -1]] * 3)
  np.testing.assert_array_equal(m.uvs[m.uv_idx[0, 0]], [0.5, 0.5])
  with pytest.raises(ValueError):
    b.add_mesh(POINTS, TRI, normals=normals)  # 2 normals for 3 vertices need normal_idx


def test_material_and_texture_dedupe(tmp_path):
  b = rw.SceneBuilder()
  png = write_png(tmp_path / "a.png", np.zeros((4, 4, 3), np.uint8))
  t0 = b.add_texture(png)
  assert b.add_texture(str(png)) == t0
  assert b.add_texture(png, colorspace="linear") != t0  # same file, other image (color space)
  assert b.add_texture(tmp_path / "missing.png") == -1
  assert b.report.dropped["texture.missing_file"] == 1
  ref = rw.scene.TextureRef(t0, channel=3, scale=(2, 2, 2, 1), bias=(-1, -1, -1, 0))
  red = rw.Material(base_color=(1.0, 0.0, 0.0), ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND))
  m = red.with_texture(rw.scene.Slot.GEOMETRY_OPACITY, ref)
  assert b.add_material(m) == b.add_material(red.with_texture(rw.scene.Slot.GEOMETRY_OPACITY, ref))
  s = b.build()
  assert s.nimage == 2 and s.ntexture == 2
  mats = s.materials
  assert mats.tex[0, rw.scene.Slot.GEOMETRY_OPACITY] == t0 and mats.tex_channel[0, rw.scene.Slot.GEOMETRY_OPACITY] == 3
  assert mats.tex[0, rw.scene.Slot.BASE_COLOR] == -1
  np.testing.assert_allclose(mats.tex_scale[0, rw.scene.Slot.GEOMETRY_OPACITY], [2, 2, 2, 1])


def test_instance_chunks_expand_to_columns():
  b = rw.SceneBuilder()
  mesh = b.add_mesh(np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.int32))
  mat = b.add_material(rw.Material())
  eye = np.eye(4)[None]
  b.add_instances("mesh", xforms=np.repeat(eye, 3, 0), mesh=mesh, material=mat, paths=["/a", "/b", "/c"], body="/body")
  b.add_instances("box", xforms=eye, material=mat, size=(1.0, 2.0, 3.0), segment=7, labels=["chair"])
  b.add_instances("sphere", xforms=np.repeat(eye, 2, 0), material=mat, color=(0.5, 0.5, 0.5, 1.0), source=4)
  inst = b.build().instances
  np.testing.assert_array_equal(
    inst.kind,
    [int(rw.scene.InstanceKind.MESH)] * 3 + [int(rw.scene.InstanceKind.BOX)] + [int(rw.scene.InstanceKind.SPHERE)] * 2,
  )
  np.testing.assert_array_equal(inst.mesh, [mesh] * 3 + [-1] * 3)
  np.testing.assert_array_equal(inst.segment, [0, 1, 2, 7, 4, 5])
  np.testing.assert_array_equal(inst.source, [-1, -1, -1, -1, 4, 4])
  np.testing.assert_array_equal(inst.size[3], [1.0, 2.0, 3.0])
  np.testing.assert_array_equal(inst.color[4], [0.5, 0.5, 0.5, 1.0])
  assert [inst.paths[p] for p in inst.prim[:3]] == ["/a", "/b", "/c"] and np.all(inst.prim[3:] == -1)
  assert inst.bodies[int(inst.body[0])] == "/body" and np.all(inst.body[3:] == inst.body[3])
  assert inst.labels[int(inst.label[3])] == "chair" and np.all(np.delete(inst.label, 3) == -1)
  assert inst.xform.dtype == np.float32 and inst.xform.shape == (6, 3, 4)


def test_instances_and_lights():
  b = rw.SceneBuilder()
  mesh = b.add_mesh(POINTS, TRI)
  xforms = np.tile(np.eye(4), (3, 1, 1))
  xforms[:, 0, 3] = [1, 2, 3]
  r = b.add_instances("mesh", xforms=xforms, mesh=mesh, paths=["/a", "/b", "/c"], body="/robot/link")
  i = b.add_instance("sphere", size=(0.5, 0, 0), path="/s")
  sphere = b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=10, radius=0.1))
  rect = b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.RECT, color=(1, 2, 3), width=2, height=3, shaping=rw.scene.Shaping(cone_angle=math.degrees(0.5))
    )
  )
  s = b.build()
  inst = s.instances
  assert list(r) == [0, 1, 2] and i == 3
  np.testing.assert_allclose(inst.xform[:3, 0, 3], [1, 2, 3])
  assert inst.paths[int(inst.prim[1])] == "/b" and inst.bodies[int(inst.body[0])] == "/robot/link" and inst.body[3] == -1
  assert inst.kind[3] == rw.scene.InstanceKind.SPHERE and list(inst.segment) == [0, 1, 2, 3]
  derived = derive_lights(s.lights)
  np.testing.assert_allclose(derived.intensity[sphere], [10 * math.pi * 0.01] * 3, rtol=1e-6)
  np.testing.assert_allclose(derived.intensity[rect], [6, 12, 18])
  assert derived.flags[rect] & rw.scene.LightFlags.SHAPED and not derived.flags[sphere] & rw.scene.LightFlags.SHAPED


def test_wpbr_round_trip(tmp_path):
  b = rw.SceneBuilder()
  mesh = b.add_mesh(POINTS, TRI, uvs=POINTS[:, :2])
  tex = b.add_texture(np.arange(4 * 4 * 4).astype(np.uint8).reshape(4, 4, 4))
  mat = b.add_material(
    rw.Material(base_color=(0.1, 0.2, 0.3), textures=((rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(tex)),))
  )
  # A material with OpenPBR extras and a texture on an extra input (the MaterialExtras table).
  extras = rw.scene.OpenPbrExtras(
    fuzz_weight=0.5, coat_color=(0.2, 0.4, 0.6), textures=(("geometry_coat_normal", rw.scene.TextureRef(tex)),)
  )
  fuzzy = b.add_material(rw.Material(specular_weight=0.3, extras=extras))
  b.add_instance("mesh", mesh=mesh, material=mat, path="/m")
  b.add_instance("mesh", mesh=mesh, material=fuzzy, path="/f")
  b.add_camera("/cam", rw.scene.Camera(xform=np.eye(4)[:3], focal_length=10, vertical_aperture=10))
  b.add_environment(rw.scene.EnvironmentLight(texture=NO_TEXTURE, color=(1, 2, 3)))
  b.report.drop("x.y", "/p")
  b.meta["k"] = "v"
  mem = b.build()
  mem.save(tmp_path / "a.wpbr")
  loaded = rw.Scene.load(tmp_path / "a.wpbr")
  for name, array in mem.arrays().items():
    np.testing.assert_array_equal(loaded.arrays()[name], array, err_msg=name)
  assert isinstance(loaded.images.data, np.memmap)
  assert list(loaded.materials.extras) == [-1, 0] and loaded.materials.specular_weight.dtype == np.float64
  assert loaded.material_extras.fuzz_weight[0] == 0.5 and list(loaded.material_extras.tex_material) == [fuzzy]
  assert rw.scene.model.material.INPUTS[int(loaded.material_extras.tex_input[0])] == "geometry_coat_normal"
  assert loaded.report.dropped["x.y"] == 1 and loaded.meta["k"] == "v"
  assert loaded.camera("/cam").fovy == pytest.approx(2 * math.degrees(math.atan(0.5)))
  # build(path) writes the same file directly.
  direct = b.build(tmp_path / "b.wpbr")
  np.testing.assert_array_equal(direct.images.level(0), loaded.images.level(0))
  arrays = read_header(tmp_path / "b.wpbr")["arrays"]
  assert isinstance(arrays, dict) and "images.data" in arrays


@pytest.mark.parametrize("to_file", [False, True])
def test_images_of_equal_content_share_their_bytes(tmp_path, to_file):
  """Images of equal content (two arrays, two files of one image) are stored once: one offset,
  the blob of the distinct contents only (in memory and in a `.wpbr`, laid out last), every level
  read back as decoded; the build counts them."""
  rng = np.random.default_rng(7)
  a, c = (rng.integers(0, 256, (8, 16, 4), dtype=np.uint8) for _ in range(2))
  b = rw.SceneBuilder()
  ids = [b.add_image(img, max_size=None, transfer="srgb") for img in (a, c, a.copy(), a.copy(), c.copy())]
  s = b.build(tmp_path / "s.wpbr" if to_file else None, mips=True)
  img = s.images
  offsets = [int(img.offset[i]) for i in ids]
  assert offsets[0] == offsets[2] == offsets[3] and offsets[1] == offsets[4] and offsets[0] != offsets[1]
  assert img.blob_size == 2 * (offsets[1] - offsets[0])
  for i, src in zip(ids, (a, c, a, a, c), strict=True):
    np.testing.assert_array_equal(img.level(i), src)
    assert img.level(i, 4).shape == (1, 1, 4)
  assert s.report.counts["images.shared_content"] == 3
  if to_file:
    layout = wpbr._layout(read_header(tmp_path / "s.wpbr"))
    assert max(layout, key=lambda n: layout[n][2]) == "images.data"
    assert layout["images.data"][1] == [img.blob_size]


def test_wpbr_keeps_the_optional_corner_rows_of_their_meshes_only(tmp_path):
  """A `.wpbr` stores the second UV set's and the vertex colors' corner indices of the meshes
  with them only; the load restores one row per triangle (-1 for the others)."""
  b = rw.SceneBuilder()
  quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
  tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
  b.add_mesh(quad * 2, tris)
  b.add_mesh(quad, tris, uvs1=quad[:, :2], colors=np.ones((4, 3), np.float32))
  b.add_mesh(quad * 3, tris, colors=np.full((4, 4), 0.5, np.float32))
  mem = b.build()
  mem.save(tmp_path / "m.wpbr")
  layout = wpbr._layout(read_header(tmp_path / "m.wpbr"))
  assert layout["meshes.uv1_idx"][1] == [2, 3] and layout["meshes.color_idx"][1] == [4, 3]
  loaded = rw.Scene.load(tmp_path / "m.wpbr")
  for name in ("uv1_idx", "color_idx", "tris"):
    np.testing.assert_array_equal(getattr(loaded.meshes, name), getattr(mem.meshes, name), err_msg=name)
  assert (mem.meshes.uv1_idx[:2] == -1).all() and (mem.meshes.color_idx[2:4] >= 0).all()


def test_images_layout_and_mips(tmp_path):
  b = rw.SceneBuilder()
  rgb = np.full((8, 4, 3), 200, np.uint8)
  b.add_texture(rgb)  # RGB is padded to RGBA
  b.add_texture(np.full((4, 4), 7, np.uint8), colorspace="linear")
  hdr = write_hdr(tmp_path / "sky.hdr", np.full((2, 16, 3), 3.0))
  b.add_texture(hdr, colorspace="linear")
  s = b.build(mips=True)
  img = s.images
  assert img.level(0).shape == (8, 4, 4) and np.all(img.level(0)[..., 3] == 255)
  assert int(img.levels[0]) == 4 and img.level(0, 3).shape == (1, 1, 4)
  np.testing.assert_array_equal(img.level(0, 2)[..., :3], 200)  # constant image: every mip identical
  assert img.level(1).shape == (4, 4, 1)
  assert img.pixel[2] == rw.scene.PixelType.F32
  np.testing.assert_allclose(img.level(2)[..., :3], 3.0, rtol=1e-2)


def test_max_texture_size_downscales():
  b = rw.SceneBuilder()
  b.add_texture(np.zeros((64, 32, 4), np.uint8), max_size=16)
  s = b.build()
  assert s.images.level(0).shape == (16, 8, 4)
  assert s.report.approximated["texture.downscaled"] == 1


def _srgb_box2(pixels: np.ndarray) -> np.ndarray:
  """2 x 2 box means of 8-bit sRGB `pixels` in linear light (the IEC 61966-2-1 curves)."""
  c = pixels.astype(np.float64) / 255.0
  lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
  h, w = lin.shape[:2]
  mean = lin.reshape(h // 2, 2, w // 2, 2, -1).mean(axis=(1, 3))
  enc = np.where(mean <= 0.0031308, mean * 12.92, 1.055 * mean ** (1.0 / 2.4) - 0.055)
  return (enc * 255.0 + 0.5).astype(np.uint8)


def test_default_texture_cap_is_1024_with_box_decode(tmp_path):
  pytest.importorskip("PIL")
  assert rw.SceneOptions().max_texture_size == 1024 == formats.MAX_TEXTURE_SIZE
  rng = np.random.default_rng(1)
  pixels = rng.integers(0, 255, (8, 2048, 3), dtype=np.uint8)
  b = rw.SceneBuilder()
  b.add_texture(write_png(tmp_path / "wide.png", pixels), max_size=rw.SceneOptions().max_texture_size)
  level = b.build(threads=1).images.level(0)
  # Decode-time downscale of a color (sRGB) image: 2 x 2 means in linear light, as the mips.
  np.testing.assert_allclose(level[..., :3].astype(int), _srgb_box2(pixels).astype(int), atol=1)


def test_srgb_downscale_averages_in_linear_light(tmp_path):
  """A black / white checker downscaled 2x is linear mid grey (sRGB 188), not code 128, for files
  and arrays; a data (linear) image averages its codes."""
  pytest.importorskip("PIL")
  checker = np.zeros((4, 4, 3), np.uint8)
  checker[::2, ::2] = checker[1::2, 1::2] = 255
  b = rw.SceneBuilder()
  b.add_texture(write_png(tmp_path / "c.png", checker), max_size=2)
  b.add_texture(checker, max_size=2)
  b.add_texture(write_png(tmp_path / "d.png", checker), colorspace="linear", max_size=2)
  s = b.build(threads=1)
  for i, code in ((0, 188), (1, 188), (2, 128)):
    np.testing.assert_array_equal(s.images.level(i)[..., :3], code)


def test_packed_data_texture_downscale_ignores_alpha(tmp_path):
  """A channel-packed data texture whose alpha is another map (Unreal's ao / roughness / metallic /
  emissive mask, mask 0) keeps its channels when downscaled: Pillow's RGBA resampling premultiplies
  by alpha and zeroed them (an Office stanchion's metalness and roughness read as 0)."""
  pytest.importorskip("PIL")
  packed = np.zeros((4, 4, 4), np.uint8)
  packed[..., 0], packed[..., 1], packed[..., 2] = 250, 86, 255  # ao, roughness, metallic; mask 0
  grey_alpha = np.zeros((4, 4, 2), np.uint8)
  grey_alpha[..., 0] = 120
  b = rw.SceneBuilder()
  b.add_texture(write_png(tmp_path / "orm_e.png", packed), colorspace="linear", max_size=2)
  b.add_texture(write_png(tmp_path / "la.png", grey_alpha), colorspace="linear", max_size=2)
  s = b.build(threads=1)
  np.testing.assert_array_equal(s.images.level(0), np.broadcast_to(packed[:2, :2], (2, 2, 4)))
  np.testing.assert_array_equal(s.images.level(1)[..., 0], 120)


def test_png_decode(tmp_path):
  pytest.importorskip("PIL")
  rng = np.random.default_rng(0)
  pixels = rng.integers(0, 255, (5, 7, 4), dtype=np.uint8)
  b = rw.SceneBuilder()
  b.add_texture(write_png(tmp_path / "rgba.png", pixels))
  b.add_texture(write_png(tmp_path / "rgb.png", pixels[..., :3]))
  b.add_texture(write_png(tmp_path / "big.png", np.zeros((32, 32, 3), np.uint8)), max_size=8)
  s = b.build(threads=2)
  np.testing.assert_array_equal(s.images.level(0), pixels)  # single-IDAT fast path is exact
  np.testing.assert_array_equal(s.images.level(1)[..., :3], pixels[..., :3])
  assert s.images.level(2).shape == (8, 8, 4)


def test_triangulate_fan_holes_winding():
  counts = np.array([3, 4, 5, 2])
  indices = np.arange(14)
  tris, _, face = geometry.triangulate(counts, indices, holes=np.array([0]))
  assert len(tris) == 2 + 3  # hole face 0 and the 2-gon are skipped
  np.testing.assert_array_equal(face, [1, 1, 2, 2, 2])
  np.testing.assert_array_equal(tris[0], [3, 4, 5])
  left, _, _ = geometry.triangulate(counts, indices, left_handed=True)
  np.testing.assert_array_equal(left[0], [0, 2, 1])


def test_triangulate_ignores_hole_indices_naming_no_face():
  """A hole index outside [0, F) skips nothing (as Hydra): -1 does not wrap to the last face, and
  an index past the faces does not fail the mesh."""
  counts, indices = np.array([3, 3]), np.array([0, 1, 2, 0, 2, 3])
  for holes in ([-1], [2], [-1, 7, 1]):
    _, _, face = geometry.triangulate(counts, indices, holes=np.array(holes))
    np.testing.assert_array_equal(face, [0] if 1 in holes else [0, 1])


def test_cone_mesh_is_closed():
  points, tris, normals, normal_idx = geometry.cone_mesh(1.0, 2.0, 16)
  assert points[:, 2].max() == pytest.approx(1.0) and points[:, 2].min() == pytest.approx(-1.0)
  assert normal_idx.max() < len(normals)
  edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
  _, counts = np.unique(edges, axis=0, return_counts=True)
  assert np.all(counts == 2)  # watertight


def test_report_summary():
  r = rw.scene.LoadReport(source="x.usd")
  r.drop("Mesh.primvars:st1", "/a")
  r.approx("Cone.tessellated", "/b", 2)
  with r.timer("phase"):
    pass
  text = str(r)
  assert "Mesh.primvars:st1" in text and "(e.g. /a)" in text and "phase=" in text
  assert rw.scene.LoadReport.from_dict(r.to_dict()).approximated["Cone.tessellated"] == 2


def test_downsample_srgb_is_linear_average():
  img = np.zeros((2, 2, 1), np.uint8)
  img[0, 0] = img[1, 1] = 255
  assert mips.downsample(img, transfer="srgb")[0, 0, 0] == 188  # sRGB(0.5)
  assert mips.downsample(img, transfer="linear")[0, 0, 0] == 128


def test_image_dedup_key_includes_color_space(tmp_path):
  """An image bound as sRGB color and as linear data is two images: their mips are filtered in
  different spaces (an image keyed by path and size only was shared)."""
  pixels = np.zeros((2, 2, 3), np.uint8)
  pixels[0, 0] = 255
  path = write_png(tmp_path / "a.png", pixels)
  b = rw.SceneBuilder()
  color = b.add_texture(path, colorspace="srgb")
  data = b.add_texture(path, colorspace="linear")
  s = b.build(mips=True)
  assert s.nimage == 2 and s.textures.image[color] != s.textures.image[data]
  srgb_mip = s.images.level(int(s.textures.image[color]), 1)[0, 0, 0]
  linear_mip = s.images.level(int(s.textures.image[data]), 1)[0, 0, 0]
  assert linear_mip == 64 and srgb_mip == 137  # mean 0.25 in linear space, encoded as sRGB


def test_default_material_is_a_dielectric():
  """`Material.metallic` once defaulted to glTF's 1, so `add_material(base_color=...)` made a
  metal; the canonical material has OpenPBR's defaults (a dielectric)."""
  b = rw.SceneBuilder()
  b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5)))
  assert rw.Material().base_metalness == 0.0 and b.build().materials.base_metalness[0] == 0.0


def test_cache_file_name_survives_code_changes():
  """The code digest is not part of the `.wpbr` file name (a code change rebuilds the same
  file: `fresh` still compares the whole key), the source and options are."""
  from raywarp.scene.model.wpbr import CacheKey

  key = CacheKey(adapter="usd", source="/scenes/office.usda", options={"max_texture_size": 2048}, code="code-a")
  assert (
    key.file_name()
    == CacheKey(adapter="usd", source="/scenes/office.usda", options={"max_texture_size": 2048}, code="code-b").file_name()
  )
  assert (
    key.file_name()
    != CacheKey(adapter="usd", source="/scenes/office.usda", options={"max_texture_size": 1024}, code="code-a").file_name()
  )
  assert key.file_name().startswith("office-")


def test_light_frame_rejects_a_zero_direction():
  """A zero or non-finite light direction raises instead of giving a NaN frame."""
  frame = xform.light_frame((0.0, 0.0, -2.0), (1.0, 2.0, 3.0))
  np.testing.assert_allclose(frame[:, 2], (0.0, 0.0, 1.0))
  np.testing.assert_allclose(frame[:, 3], (1.0, 2.0, 3.0))
  for bad in ((0.0, 0.0, 0.0), (math.nan, 0.0, 1.0)):
    with pytest.raises(ValueError, match="non-zero and finite"):
      xform.light_frame(bad, (0.0, 0.0, 0.0))


def test_table_take_keeps_strings_in_row_order():
  @dataclasses.dataclass
  class Named(Table):
    SCHEMA: ClassVar = {"value": (np.dtype(np.int32), ())}
    STRINGS: ClassVar = ("names",)
    value: np.ndarray
    names_blob: np.ndarray
    names_offsets: np.ndarray

  t = Named.from_columns({"value": [10, 11, 12]}, {"names": ["a", "bé", ""]})
  picked = t.take(np.array([2, 0, 0, 1]))
  assert picked.value.tolist() == [12, 10, 10, 11]
  assert picked.strings("names").tolist() == ["", "a", "a", "bé"]


@pytest.mark.parametrize("transparency", [None, 3])
def test_palette_png_decodes_as_rgba(tmp_path, transparency):
  """Palette PNGs, with or without a transparent index, store Pillow's RGBA conversion."""
  from PIL import Image

  rgb = (np.random.default_rng(0).random((8, 8, 3)) * 255).astype(np.uint8)
  path = tmp_path / "palette.png"
  extra = {} if transparency is None else {"transparency": transparency}
  Image.fromarray(rgb).convert("P").save(path, **extra)
  source = imaging.ImageSource(path=str(path))
  info = imaging.probe(source)
  base, error = imaging.decode_base(source, info)
  levels = mips.mip_chain(base, 1, transfer=source.transfer)
  assert not error and info.channels == 4
  with Image.open(path) as im:
    np.testing.assert_array_equal(levels[0], np.asarray(im.convert("RGBA")))


def test_mesh_part_ids_are_the_callers_own_list():
  """A deduplicated mesh's ids come back as a new list: mutating one result cannot change what a
  later identical mesh gets."""
  b = rw.SceneBuilder()
  points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
  tris = np.array([[0, 1, 2], [1, 3, 2]])
  first = b.add_mesh_parts(points, tris, tri_part=[1, 0], nparts=2)
  first[0] = NO_MESH
  assert b.add_mesh_parts(points, tris, tri_part=[1, 0], nparts=2) == [0, 1]
  again = b.add_mesh_parts(points, tris, tri_part=[1, 0], nparts=2)
  again.append(NO_MESH)
  assert b.add_mesh_parts(points, tris, tri_part=[1, 0], nparts=2) == [0, 1]


def test_builder_rejects_misaligned_per_row_inputs():
  """`tri_part` / `paths` / `labels` of the wrong length raise instead of shifting rows."""
  b = rw.SceneBuilder()
  points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
  tris = np.array([[0, 1, 2], [1, 3, 2]])
  for bad in ([0], [0, 2], [0, -1]):
    with pytest.raises(ValueError, match="tri_part"):
      b.add_mesh_parts(points, tris, tri_part=bad, nparts=2)
  assert b.add_mesh_parts(points, tris, tri_part=[1, 0], nparts=2) == [0, 1]
  xforms = np.tile(np.eye(4), (2, 1, 1))
  with pytest.raises(ValueError, match="1 paths for 2 instances"):
    b.add_instances("sphere", xforms=xforms, size=(0.1, 0, 0), paths=["/a"])
  with pytest.raises(ValueError, match="3 labels for 2 instances"):
    b.add_instances("sphere", xforms=xforms, size=(0.1, 0, 0), labels=["a", "b", "c"])


def test_point_light_has_no_direction():
  """A `Light.point` takes no direction (identity axes); a spot emits along its direction."""
  from raywarp.scene.lighting.rig import add_lights

  b = rw.SceneBuilder()
  lights = [
    rw.Light.point((1.0, 2.0, 3.0)),
    rw.Light.spot((0.0, 0.0, 3.0), (0.0, 0.0, -2.0)),
  ]
  add_lights(b, lights, center=np.zeros(3), extent=1.0, exposure=1.0)
  frames = b.build().lights.xform
  np.testing.assert_allclose(frames[0], np.c_[np.eye(3), [1.0, 2.0, 3.0]])
  np.testing.assert_allclose(-frames[1][:, 2], [0.0, 0.0, -1.0], atol=1e-6)


@pytest.mark.parametrize("budget", [0, 1 << 30])
def test_images_read_ahead_match_a_build_decode(tmp_path, monkeypatch, budget):
  """Images decoded ahead (during the walk) or at build (over the budget) give the same blob,
  with mips, and the read file states are the files' stat."""
  from PIL import Image

  from raywarp.scene.images import blob
  from raywarp.scene.model import wpbr

  rng = np.random.default_rng(1)
  files = [write_png(tmp_path / "a.png", (rng.random((16, 8, 3)) * 255).astype(np.uint8))]
  files.append(tmp_path / "b.jpg")
  Image.fromarray((rng.random((8, 8, 3)) * 255).astype(np.uint8)).save(files[1])
  files.append(write_png(tmp_path / "c.png", (rng.random((4, 4, 4)) * 255).astype(np.uint8)))

  def build(ahead_bytes: int) -> rw.Scene:
    monkeypatch.setattr(blob, "AHEAD_BYTES", ahead_bytes)
    b = rw.SceneBuilder()
    for f in files:
      b.add_texture(f)
      b.add_texture(f, colorspace="linear", max_size=4)
    assert b.read_states() == {str(f): wpbr.stat_files([str(f)])[0] for f in files}
    return b.build(mips=True)

  ahead, at_build = build(budget), build(0)
  np.testing.assert_array_equal(ahead.images.data, at_build.images.data)
  np.testing.assert_array_equal(ahead.images.offset, at_build.images.offset)


def test_images_decoded_ahead_are_released_after_build(tmp_path, monkeypatch):
  """The decoded top levels held for the build are freed once written (not kept by the builder)."""
  import gc
  import weakref

  from raywarp.scene.images import blob

  refs: list[weakref.ref] = []
  decode = blob.decode_base

  def tracked(*args, **kwargs):
    """`decode_base`, remembering its arrays."""
    base, error = decode(*args, **kwargs)
    if args[0].path:  # an array source's base is the caller's array itself
      refs.append(weakref.ref(base))
    return base, error

  monkeypatch.setattr(blob, "decode_base", tracked)
  b = rw.SceneBuilder()
  rng = np.random.default_rng(2)
  for name in ("a", "b", "c"):
    b.add_texture(write_png(tmp_path / f"{name}.png", (rng.random((8, 8, 3)) * 255).astype(np.uint8)))
  b.add_texture(np.zeros((4, 4, 4), np.uint8))  # an array: probed and decoded at build
  scene = b.build(tmp_path / "s.wpbr")
  gc.collect()
  assert len(refs) == 3 and all(r() is None for r in refs)
  assert scene.nimage == 4 and all(p.base is None for p in b._images._ahead.results())
