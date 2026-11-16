"""Loader fidelity: image formats, EXR, second UV sets, vertex colors, UV transforms, baked
specular-glossiness and MDL outputs, opacityMode, the .rayw writer on rename-copying mounts."""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import required
from rw_helpers import bind, mdl_material, new_stage, preview_material, quad, quadrant_png, write_png

import raywarp as rw
import raywarp.formats.usd
from raywarp.cache import paths
from raywarp.scene.images import decode as imaging
from raywarp.scene.images import mips
from raywarp.scene.model import rayw

S = rw.scene.Slot
_RES = 32


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
  """sRGB bytes (0-255) to linear values."""
  c = c / 255.0
  return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# Images ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["P", "PA", "LA", "CMYK", "I;16", "1"])
def test_pillow_modes_decode(tmp_path, mode):
  """Palette (with / without transparency), grey + alpha, CMYK, 16-bit grey and 1-bit images
  decode to their probed layout (no 1x1 mid-grey fallback)."""
  Image = pytest.importorskip("PIL.Image")
  rgb = np.zeros((4, 4, 3), np.uint8)
  rgb[:, 2:] = (255, 0, 0)
  im = Image.fromarray(rgb)
  if mode in ("P", "PA"):
    im = im.convert("P")
    if mode == "PA":
      im.info["transparency"] = 0
  elif mode == "I;16":
    im = Image.fromarray(np.full((4, 4), 40000, np.uint16))
  else:
    im = im.convert(mode)
  path = tmp_path / ("img.jpg" if mode == "CMYK" else "img.png")
  im.save(path)
  src = imaging.ImageSource(path=str(path))
  info = imaging.probe(src)
  base, error = imaging.decode_base(src, info)
  levels = mips.mip_chain(base, 1, transfer=src.transfer)
  assert error == "" and levels[0].shape == (4, 4, info.channels)
  if mode in ("P", "PA"):  # the palette is resolved: black left half, red right half
    np.testing.assert_array_equal(levels[0][:, 0, :3], 0)
    np.testing.assert_array_equal(levels[0][:, 3, :3], [(255, 0, 0)] * 4)
  if mode == "I;16":
    assert int(levels[0][0, 0, 0]) == round(40000 / 65535 * 255)


# Second UV set, UV transforms, vertex colors ----------------------------------------------------


def _textured_quad(stage, path: str, rotation: float | None = None, x: float = 0.0, uv_space_index: int | None = None):
  from pxr import Sdf, UsdGeom, UsdShade

  mesh = UsdGeom.Mesh.Define(stage, path)
  mesh.CreatePointsAttr([(x - 0.5, -0.5, 0), (x + 0.5, -0.5, 0), (x + 0.5, 0.5, 0), (x - 0.5, 0.5, 0)])
  mesh.CreateFaceVertexCountsAttr([4])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
  api = UsdGeom.PrimvarsAPI(mesh)
  api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
  # st1: mirrored left-right, on its own (faceVarying) indexing.
  st1 = api.CreatePrimvar("st1", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
  st1.Set([(1, 0), (0, 0), (0, 1), (1, 1)])
  if uv_space_index is not None:
    # OmniPBR (built-in defaults) reading texture_coordinate(uv_space_index): primvars:st1.
    inputs = {
      "diffuse_texture": (Sdf.ValueTypeNames.Asset, "quad.png"),
      "uv_space_index": (Sdf.ValueTypeNames.Int, uv_space_index),
    }
    bind(mesh, mdl_material(stage, f"{path}_Looks", "OmniPBR.mdl", "OmniPBR", **inputs))
    return mesh
  mat, surface = preview_material(stage, f"{path}_Looks")
  reader = UsdShade.Shader.Define(stage, f"{path}_Looks/Reader")
  reader.CreateIdAttr("UsdPrimvarReader_float2")
  reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
  st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
  if rotation is not None:
    xf = UsdShade.Shader.Define(stage, f"{path}_Looks/Xf")
    xf.CreateIdAttr("UsdTransform2d")
    xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
    xf.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(rotation)
    xf.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set((1.0, 1.0))
    st = xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)
  tex = UsdShade.Shader.Define(stage, f"{path}_Looks/Image")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("quad.png")
  tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
  surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
  )
  bind(mesh, mat)
  return mesh


def _camera(stage):
  from pxr import UsdGeom, UsdLux

  camera = UsdGeom.Camera.Define(stage, "/World/Camera")
  UsdGeom.XformCommonAPI(camera).SetTranslate((0.0, 0.0, 6.0))  # sees x, y in [-1.25, 1.25]
  sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
  sun.CreateAngleAttr(0.0)


def _albedo(host) -> np.ndarray:
  scene = rw.upload(host, device="cpu")
  settings = rw.RenderSettings(outputs=rw.Outputs.of("rgb", "albedo"))
  out = rw.Renderer(scene, nworld=1, cameras=None, resolution=(_RES, _RES), settings=settings).render()
  return required(out.albedo).numpy().reshape(1, _RES, _RES, 3)[0]


def test_second_uv_set_and_rotated_uv_transform(tmp_path):
  """A texture reading st1 (OmniPBR uv_space_index 1) samples the mesh's second UV set (its own
  corner indices: the core mesh gets its own face rows); a UsdTransform2d rotation and
  translation (180 degrees about the origin, then (1, 1)) is applied in full (baked into the
  core mesh's texcoords): both quads show the image mirrored / rotated."""
  pytest.importorskip("pxr")
  stage = new_stage(tmp_path)
  quadrant_png(tmp_path / "quad.png")
  _textured_quad(stage, "/World/A", x=-0.6, uv_space_index=1)
  _textured_quad(stage, "/World/B", rotation=180.0, x=0.6)
  _camera(stage)
  stage.Save()
  builder = rw.formats.usd.read(stage)
  host = builder.build()
  a = host.instances.material[0]
  assert host.materials.uv_set[a, S.BASE_COLOR] == 1
  m = host.meshes.get(int(host.instances.mesh[0]))
  np.testing.assert_allclose(m.uvs1[m.uv1_idx[0]], [(1, 0), (0, 0), (0, 1)])
  assert builder.report.dropped["Mesh.primvars:st1"] == 1  # quad B does not read it
  albedo = _albedo(host)
  assert not [k for k in host.report.dropped if k.startswith("gpu.")]
  # Quad A (left half of the view) mirrored: its top-left quadrant is the image's green one.
  # Top quadrants (row 10): quad A at columns 3 / 10, quad B at columns 21 / 28.
  # Quad A mirrored: green | red (authored: red | green).
  np.testing.assert_allclose(albedo[10, 3], [0, 1, 0], atol=0.05)
  np.testing.assert_allclose(albedo[10, 10], [1, 0, 0], atol=0.05)
  # Quad B rotated by 180 degrees: white | blue.
  np.testing.assert_allclose(albedo[10, 21], [1, 1, 1], atol=0.05)
  np.testing.assert_allclose(albedo[10, 28], [0, 0, 1], atol=0.05)


def test_vertex_colors_multiply_the_base_color(tmp_path):
  """An unbound mesh's per-vertex displayColor becomes vertex colors (kernel compiled with
  Features.vertex_colors); a constant one stays the fallback material's color."""
  pytest.importorskip("pxr")
  from pxr import Sdf, UsdGeom

  stage = new_stage(tmp_path)
  mesh = UsdGeom.Mesh.Define(stage, "/World/Q")
  mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
  mesh.CreateFaceVertexCountsAttr([4])
  mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
  pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
  pv.Set([(1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 1)])  # bottom red, top blue
  flat = quad(stage, "/World/Flat", 0.1)
  flat.CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
  UsdGeom.XformCommonAPI(flat).SetTranslate((5.0, 5.0, 0.0))
  _camera(stage)
  host = rw.formats.usd.read(stage).build()
  mid = int(host.instances.material[0])
  assert host.materials.ext_flags[mid] & rw.scene.MaterialFlags.VERTEX_COLOR
  assert host.meshes.color_num[int(host.instances.mesh[0])] > 0
  assert host.meshes.color_num[int(host.instances.mesh[1])] == 0  # constant: material color
  scene = rw.upload(host, device="cpu")
  assert scene.capabilities.vertex_colors
  albedo = _albedo(host)
  # Rows 6 / RES - 7: y = +-0.75 on the quad (the fitted aperture spans +-1.26 both ways).
  top, bottom, centre = albedo[6, _RES // 2], albedo[_RES - 7, _RES // 2], albedo[_RES // 2, _RES // 2]
  assert top[2] > 0.6 and top[0] < 0.4 and bottom[0] > 0.6 and bottom[2] < 0.4
  np.testing.assert_allclose(centre, [0.5, 0.0, 0.5], atol=0.08)


def test_material_vertex_color_attribute_and_primvar_reader(tmp_path):
  pytest.importorskip("pxr")
  from pxr import Sdf, UsdGeom, UsdShade

  stage = new_stage(tmp_path)
  mesh = quad(stage, "/A")
  UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("Col", Sdf.ValueTypeNames.Color4fArray, UsdGeom.Tokens.vertex).Set(
    [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 1, 0.5)]
  )
  mat, _ = preview_material(stage, "/Looks/Tinted", color=(0.5, 0.5, 0.5))
  mat.GetPrim().CreateAttribute("raywarp:vertexColor", Sdf.ValueTypeNames.Token, custom=True).Set("Col")
  bind(mesh, mat)
  other = quad(stage, "/B")
  UsdGeom.PrimvarsAPI(other).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
    [(1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0)]
  )
  mat2, surface = preview_material(stage, "/Looks/Reader")
  reader = UsdShade.Shader.Define(stage, "/Looks/Reader/Col")
  reader.CreateIdAttr("UsdPrimvarReader_float3")
  reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
  surface.GetInput("diffuseColor").ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float3))
  bind(other, mat2)
  builder = rw.formats.usd.read(stage)
  s = builder.build()
  for i, base in ((0, (0.5, 0.5, 0.5)), (1, (1.0, 1.0, 1.0))):
    mid = int(s.instances.material[i])
    assert s.materials.ext_flags[mid] & rw.scene.MaterialFlags.VERTEX_COLOR
    np.testing.assert_allclose(s.materials.base_color[mid, :3], base)
  m = s.meshes.get(int(s.instances.mesh[0]))
  np.testing.assert_allclose(m.colors[m.color_idx[0]], [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
  assert not [k for k in builder.report.dropped if "Col" in k or "preview.connected_node" in k]


# Specular-glossiness, opacityMode ---------------------------------------------------------------


def test_specular_workflow_textures_are_baked_to_metallic(tmp_path):
  """useSpecularWorkflow with diffuse and specularColor textures: base color and metallic
  textures computed per texel with the KHR_materials_pbrSpecularGlossiness conversion."""
  pytest.importorskip("pxr")
  from pxr import Sdf, UsdShade

  from raywarp.formats.usd.materials.preview import specular_to_metallic_arrays

  stage = new_stage(tmp_path)
  diffuse = np.zeros((2, 2, 3), np.uint8)
  diffuse[0] = (200, 30, 30)
  diffuse[1] = (10, 10, 10)
  spec = np.zeros((2, 2, 3), np.uint8)
  spec[0] = (10, 10, 10)  # dielectric
  spec[1] = (250, 200, 60)  # gold-like metal
  write_png(tmp_path / "d.png", diffuse)
  write_png(tmp_path / "s.png", spec)
  mat, surface = preview_material(stage, "/Looks/SG")
  surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
  for name, file, inp, kind in (
    ("D", "d.png", "diffuseColor", Sdf.ValueTypeNames.Color3f),
    ("S", "s.png", "specularColor", Sdf.ValueTypeNames.Color3f),
  ):
    tex = UsdShade.Shader.Define(stage, f"/Looks/SG/{name}")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
    surface.CreateInput(inp, kind).ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  s = builder.build()
  mid = int(s.instances.material[0])
  assert s.materials.base_metalness[mid] == 1.0 and builder.report.counts["images.derived"] == 2
  assert "UsdPreviewSurface.specularColor.texture" not in builder.report.dropped
  base_img = int(s.textures.image[s.materials.tex[mid, S.BASE_COLOR]])
  metal_img = int(s.textures.image[s.materials.tex[mid, S.BASE_METALNESS]])
  lin = _srgb_to_linear
  metallic, base = specular_to_metallic_arrays(lin(diffuse.astype(np.float64)), lin(spec.astype(np.float64)))
  np.testing.assert_allclose(s.images.level(metal_img)[..., 0] / 255.0, metallic, atol=1 / 255)
  assert metallic[0, 0] == 0.0 and metallic[1, 0] > 0.9
  got = lin(s.images.level(base_img)[..., :3].astype(np.float64))
  np.testing.assert_allclose(got, base, atol=0.01)


@pytest.mark.parametrize(("mode", "approx"), [(None, True), ("transparent", True), ("presence", False)])
def test_opacity_mode(tmp_path, mode, approx):
  pytest.importorskip("pxr")
  from pxr import Sdf

  stage = new_stage(tmp_path)
  mat, surface = preview_material(stage, "/Looks/M", opacity=0.4)
  if mode is not None:
    surface.CreateInput("opacityMode", Sdf.ValueTypeNames.Token).Set(mode)
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  m = builder.materials[0]
  assert m.ext.alpha_mode == rw.scene.AlphaMode.BLEND and m.geometry_opacity == pytest.approx(0.4)
  key = "UsdPreviewSurface.opacityMode.transparent(as presence: specular attenuated too)"
  assert (key in builder.report.approximated) == approx


# MDL bakes --------------------------------------------------------------------------------------

BAKE_BODY = """mdl 1.6;
import ::df::*;
import ::math::*;
import ::state::*;
import ::tex::*;
export material MI_Bake(
\tuniform texture_2d Albedo = texture_2d("./albedo.png",::tex::gamma_srgb),
\tuniform texture_2d Mask = texture_2d("./mask.png",::tex::gamma_linear),
\tfloat Desaturation = DESAT,
\tfloat4 Tint = float4(1.0,0.5,0.5,1.0))
 =
\tlet {
\tfloat2 UV = float2(state::texture_coordinate(0).x,1.0-state::texture_coordinate(0).y);
\tfloat4 Local0 = tex::lookup_float4(Albedo,float2(UV.x,1.0-UV.y),tex::wrap_repeat,tex::wrap_repeat);
\tfloat Local1 = math::dot(float3(Local0.x,Local0.y,Local0.z), float3(0.3,0.59,0.11));
\tfloat3 Local2 = math::lerp(float3(Local0.x,Local0.y,Local0.z),float3(Local1,Local1,Local1),Desaturation);
\tfloat4 Local3 = tex::lookup_float4(Mask,float2(UV.x,1.0-UV.y),tex::wrap_repeat,tex::wrap_repeat);
\tfloat3 Local4 = math::lerp(float3(Tint.x,Tint.y,Tint.z),Local2,float3(Local3.x,Local3.x,Local3.x));
\tfloat Local5 = math::min(math::max(math::pow(Local3.y, 2.0), 0.1), 0.9);
\t} in material(surface: material_surface(scattering: df::custom_curve_layer(normal_reflectivity: 0.04,
\t\tlayer: df::microfacet_ggx_vcavities_bsdf(roughness_u: Local5 * Local5, roughness_v: Local5 * Local5),
\t\tbase: df::diffuse_reflection_bsdf(tint: color(Local4)))));
"""


@pytest.mark.parametrize("desaturation", [0.0, 0.5])
def test_mdl_non_affine_outputs_are_baked(tmp_path, desaturation):
  """math::dot (affine), lerp by a texture (a product of two textures), pow / min / max: outputs
  that are not one channel's affine function are baked per texel by the MDL SDK."""
  pytest.importorskip("pxr")
  from pxr import Sdf, UsdShade

  stage = new_stage(tmp_path)
  (tmp_path / "MI_Bake.mdl").write_text(BAKE_BODY.replace("DESAT", f"{desaturation:.1f}"))
  albedo = np.zeros((2, 2, 3), np.uint8)
  albedo[:, 0] = (255, 128, 0)
  albedo[:, 1] = (0, 64, 255)
  mask = np.zeros((2, 2, 3), np.uint8)
  mask[0] = (255, 255, 0)
  mask[1] = (0, 128, 0)
  write_png(tmp_path / "albedo.png", albedo)
  write_png(tmp_path / "mask.png", mask)
  mat = UsdShade.Material.Define(stage, "/Looks/UE")
  shader = UsdShade.Shader.Define(stage, "/Looks/UE/Shader")
  shader.SetSourceAsset(Sdf.AssetPath("MI_Bake.mdl"), "mdl")
  shader.SetSourceAssetSubIdentifier("MI_Bake", "mdl")
  mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  assert builder.report.counts["mdl.baked"] == 2
  s = builder.build()
  mid = int(s.instances.material[0])
  lin = _srgb_to_linear
  a = lin(albedo.astype(np.float64))
  grey = a @ np.array([0.3, 0.59, 0.11])
  a = a * (1.0 - desaturation) + grey[..., None] * desaturation
  t = mask[..., :1] / 255.0
  expected = np.array([1.0, 0.5, 0.5]) * (1.0 - t) + a * t
  base_img = int(s.textures.image[s.materials.tex[mid, S.BASE_COLOR]])
  np.testing.assert_allclose(lin(s.images.level(base_img)[..., :3].astype(np.float64)), expected, atol=0.01)
  rough_img = int(s.textures.image[s.materials.tex[mid, S.SPECULAR_ROUGHNESS]])
  g = mask[..., 1] / 255.0
  np.testing.assert_allclose(s.images.level(rough_img)[..., 0] / 255.0, np.clip(g * g, 0.1, 0.9), atol=1 / 255)
  assert s.materials.specular_roughness[mid] == 1.0 and s.materials.base_metalness[mid] == 0.0


# The .rayw writer --------------------------------------------------------------------------------


def _small_scene() -> rw.SceneBuilder:
  b = rw.SceneBuilder()
  m = b.add_mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
  b.add_instance("mesh", mesh=m, material=b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5))))
  return b


def test_rayw_writer_leaves_no_temporary_file_when_rename_copies(tmp_path, monkeypatch):
  """A rename that copies and leaves its source behind (some network mounts): the writer still
  removes its temporary file."""
  real_replace = os.replace

  def copying_replace(src, dst):
    import shutil

    shutil.copyfile(src, dst)  # the source stays

  monkeypatch.setattr(os, "replace", copying_replace)
  path = tmp_path / "s.rayw"
  scene = _small_scene().build(path)
  monkeypatch.setattr(os, "replace", real_replace)
  assert len(scene.instances) == 1
  assert sorted(p.name for p in tmp_path.iterdir()) == ["s.rayw"]
  rayw.read_header(path)


def test_failed_rename_does_not_close_a_reused_file_descriptor(tmp_path, monkeypatch):
  """The writer's descriptor is closed before the rename; when the rename fails, the cleanup must
  not close that number again: another file opened meanwhile reuses it."""
  reused: list[int] = []

  def failing_replace(src, dst):
    reused.append(os.open(src, os.O_RDONLY))  # the lowest free number: the writer's old one
    raise OSError("rename failed")

  monkeypatch.setattr(os, "replace", failing_replace)
  with pytest.raises(OSError, match="rename failed"):
    rayw.save(tmp_path / "s.rayw", {"a": np.zeros(4, np.uint8)}, dict)
  os.fstat(reused[0])  # still open (the writer did not close it twice)
  os.close(reused[0])
  assert list(tmp_path.iterdir()) == []


def test_stale_temporary_files_are_removed(tmp_path):
  path = tmp_path / "s.rayw"
  dead = tmp_path / f"s.rayw.tmp-{paths._host()}-999999999-a1"
  live = tmp_path / f"s.rayw.tmp-{paths._host()}-{os.getpid()}-a2"
  old = tmp_path / "s.rayw.tmp-otherhost-12-a3"
  fresh = tmp_path / "s.rayw.tmp-otherhost-13-a4"
  for p in (dead, live, old, fresh):
    p.write_bytes(b"x")
  os.utime(old, (0, 0))
  removed = paths.remove_stale(path)
  assert sorted(p.name for p in removed) == sorted([dead.name, old.name])
  assert live.exists() and fresh.exists()


def test_old_temporary_file_of_a_reused_pid_is_removed(tmp_path):
  """A temporary file of this pid space whose pid now names another, live process (pid reuse)
  is stale once it was last written STALE_SECONDS ago: it is not kept forever."""
  path = tmp_path / "s.rayw"
  reused = tmp_path / f"s.rayw.tmp-{paths._host()}-{os.getpid()}-a1"
  reused.write_bytes(b"x")
  os.utime(reused, (0, 0))
  assert paths.remove_stale(path) == [reused]


def test_concurrent_writers_of_one_file_use_their_own_temporary_file(tmp_path):
  """Two writers of the same `.rayw` in one process (threads) never share a temporary file."""
  import concurrent.futures

  path = tmp_path / "s.rayw"
  arrays = {"a": np.zeros(4, np.uint8)}
  first, second = rayw.Writer(path, arrays), rayw.Writer(path, arrays)
  assert first.tmp != second.tmp
  first.abort()
  second.abort()
  with concurrent.futures.ThreadPoolExecutor(4) as pool:
    scenes = list(pool.map(lambda _: _small_scene().build(path), range(8)))
  assert all(len(s.instances) == 1 for s in scenes)
  assert sorted(p.name for p in tmp_path.iterdir()) == ["s.rayw"]
