"""MaterialX materials through the official MaterialX library and its MDL generator
(`formats/usd/materials/materialx.py`), converted by the MDL SDK baker.

The generator tests need MaterialX only; the conversion tests need the MDL SDK wheel too. Example
materials: the OpenPBR repository's `examples/open_pbr_{gold,carpaint,glass,brick}.mtlx` (values in
ACEScg as authored there) and MaterialX's Standard Surface / glTF PBR examples (tests/data/mtlx).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import DATA, required
from rw_helpers import bind, mdl_material, new_stage, preview_material, quad, write_png

import raywarp as rw
import raywarp.formats.usd
from raywarp.scene.images import colorspace
from raywarp.settings.fields import DEFAULT_EV100

pytest.importorskip("pxr")
mx = pytest.importorskip("MaterialX")

from pxr import Sdf, UsdShade  # after the skips

from raywarp.formats.usd.materials import materialx
from raywarp.scene.model.report import LoadReport

S = rw.scene.Slot


def _sdk() -> None:
  """Skips without the MDL SDK wheel (MaterialX converts through it)."""


def _material(builder: rw.SceneBuilder, name: str) -> rw.Material:
  return next(m for m in builder.materials if m.name == name)


def _network_stage(tmp_path: Path, *, texture: str | None = None, uv: str | None = None):
  """A quad bound to an OpenPBR network: a red rough dielectric, or `texture` (sRGB) on its base
  color read through `uv` ("st" / "st1" by a geompropvalue, None: the default UVs)."""
  stage = new_stage(tmp_path)
  mat = UsdShade.Material.Define(stage, "/Looks/M")
  surface = UsdShade.Shader.Define(stage, "/Looks/M/S")
  surface.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
  surface.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(0.6)
  color = surface.CreateInput("base_color", Sdf.ValueTypeNames.Color3f)
  if texture is None:
    color.Set((0.8, 0.1, 0.1))
  else:
    image = UsdShade.Shader.Define(stage, "/Looks/M/Image")
    image.CreateIdAttr("ND_image_color3")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    image.GetInput("file").GetAttr().SetColorSpace("srgb_texture")
    if uv is not None:
      reader = UsdShade.Shader.Define(stage, "/Looks/M/UV")
      reader.CreateIdAttr("ND_geompropvalue_vector2")
      reader.CreateInput("geomprop", Sdf.ValueTypeNames.String).Set(uv)
      out = reader.CreateOutput("out", Sdf.ValueTypeNames.Float2)
      image.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(out)
    color.ConnectToSource(image.CreateOutput("out", Sdf.ValueTypeNames.Color3f))
  mat.CreateSurfaceOutput("mtlx").ConnectToSource(surface.CreateOutput("out", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q"), mat)
  return stage, surface


# -- generation (MaterialX only) ---------------------------------------------------------------


def test_a_network_becomes_a_module_with_its_files_as_arguments(tmp_path):
  write_png(tmp_path / "t.png", np.full((2, 2, 3), 200, np.uint8))
  _, surface = _network_stage(tmp_path, texture="t.png")
  report = LoadReport()
  out = materialx.generate(materialx.from_shader(surface, report), report, surface.GetPrim())
  assert f"export material {materialx.MATERIAL}" in out.source
  assert "t.png" not in out.source  # the file is an argument, not baked into the module
  (arg,) = out.inputs
  assert isinstance(arg.value, Sdf.AssetPath)
  assert arg.value.resolvedPath == str(tmp_path / "t.png") and arg.space == rw.scene.ColorSpace.SRGB_REC709
  assert not report.dropped


def test_color_constants_are_converted_to_linear_rec709(tmp_path):
  _, surface = _network_stage(tmp_path)
  surface.GetInput("base_color").GetAttr().SetColorSpace("acescg")
  network = materialx.from_shader(surface, LoadReport())
  materialx.generate(network, LoadReport(), surface.GetPrim())
  value = network.document.getNode("S").getInput("base_color").getValue()
  expected = colorspace.to_lin_rec709((0.8, 0.1, 0.1), rw.scene.ColorSpace.LIN_AP1)
  assert tuple(value) == pytest.approx(expected, abs=1e-6)


def test_uv_primvar_reads_become_texcoord_reads(tmp_path):
  write_png(tmp_path / "t.png", np.full((2, 2, 3), 200, np.uint8))
  _, surface = _network_stage(tmp_path, texture="t.png", uv="st1")
  network = materialx.from_shader(surface, LoadReport())
  materialx.generate(network, LoadReport(), surface.GetPrim())
  node = network.document.getNode("UV")
  assert node.getCategory() == "texcoord" and node.getInputValue("index") == 1


def test_a_connection_to_one_output_of_a_multi_output_node_names_it(tmp_path):
  """Exporters author only the connected output of a multi-output node (`outg` of separate3):
  the MaterialX input still names it (else it connects to a missing default `out`)."""
  stage, surface = _network_stage(tmp_path)
  split = UsdShade.Shader.Define(stage, "/Looks/M/Split")
  split.CreateIdAttr("ND_separate3_color3")
  split.CreateInput("in", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.5, 0.8))
  outg = split.CreateOutput("outg", Sdf.ValueTypeNames.Float)
  surface.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).ConnectToSource(outg)
  network = materialx.from_shader(surface, LoadReport())
  roughness = network.document.getNode("S").getInput("specular_roughness")
  assert roughness.getNodeName() == "Split" and roughness.getOutputString() == "outg"


def test_legacy_ids_resolve_to_the_fitting_nodedef(tmp_path):
  """1.38's `ND_normalmap` (no type suffix) is the 1.39 nodedef its inputs fit (reported)."""
  stage = new_stage(tmp_path)
  shader = UsdShade.Shader.Define(stage, "/N")
  shader.CreateIdAttr("ND_normalmap")
  shader.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((1.0, -1.0))
  shader.CreateOutput("out", Sdf.ValueTypeNames.Float3)
  report = LoadReport()
  assert materialx._nodedef(shader, report).getName() == "ND_normalmap_vector2"
  assert report.approximated["mtlx.nodedef:ND_normalmap->ND_normalmap_vector2"] == 1


def test_unknown_nodes_raise_and_unknown_inputs_are_reported(tmp_path):
  _, surface = _network_stage(tmp_path)
  surface.CreateInput("not_an_input", Sdf.ValueTypeNames.Float).Set(1.0)
  report = LoadReport()
  materialx.from_shader(surface, report)
  assert report.dropped["mtlx.input:open_pbr_surface.not_an_input"] == 1
  surface.CreateIdAttr("ND_no_such_surface_surfaceshader")
  with pytest.raises(materialx.MaterialXError, match="unknown_node"):
    materialx.from_shader(surface, LoadReport())


def test_modules_are_named_by_their_content(tmp_path):
  a = materialx.module_file(tmp_path, "mdl 1.10;\n")
  assert a == materialx.module_file(tmp_path, "mdl 1.10;\n") and a.read_text() == "mdl 1.10;\n"
  assert a != materialx.module_file(tmp_path, "mdl 1.9;\n")


def test_surface_shaders_are_recognised_by_their_nodedef(tmp_path):
  stage = new_stage(tmp_path)
  for i, node_id in enumerate(("ND_standard_surface_surfaceshader", "ND_gltf_pbr_surfaceshader", "ND_surface_unlit")):
    shader = UsdShade.Shader.Define(stage, f"/S{i}")
    shader.CreateIdAttr(node_id)
    assert materialx.is_surface(shader), node_id
  for i, node_id in enumerate(("ND_image_color3", "UsdPreviewSurface", "")):
    shader = UsdShade.Shader.Define(stage, f"/T{i}")
    shader.CreateIdAttr(node_id)
    assert not materialx.is_surface(shader), node_id


_DATA_IMAGES = """<?xml version="1.0"?>
<materialx version="1.39" colorspace="srgb_texture">
  <image name="base" type="color3"><input name="file" type="filename" value="c.png" /></image>
  <image name="rough" type="float"><input name="file" type="filename" value="r.png" /></image>
  <image name="nimg" type="vector3"><input name="file" type="filename" value="n.png" /></image>
  <normalmap name="nrm" type="vector3"><input name="in" type="vector3" nodename="nimg" /></normalmap>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodename="base" />
    <input name="specular_roughness" type="float" nodename="rough" />
    <input name="geometry_normal" type="vector3" nodename="nrm" />
  </open_pbr_surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S" /></surfacematerial>
</materialx>
"""


def test_only_color_images_are_color_managed(tmp_path):
  """A document in srgb_texture: its color image is read as sRGB, its roughness and normal maps
  (float / vector3 images: data) as linear, as MaterialX color-manages only color3 / color4."""
  for name in ("c.png", "r.png", "n.png"):  # a document image that names no file is left out
    write_png(tmp_path / name, np.zeros((2, 2, 3), np.uint8))
  doc = materialx.new_document()
  text = _DATA_IMAGES
  for name in ("c.png", "r.png", "n.png"):  # resolved paths, as `from_shader` writes them
    text = text.replace(f'value="{name}"', f'value="{tmp_path / name}"')
  mx.readFromXmlString(doc, text)
  out = materialx.generate(materialx.Network(document=doc, material="M"), LoadReport(), "t")
  spaces = {Path(_asset(i.value)).name: i.space for i in out.inputs}
  assert spaces == {
    "c.png": rw.scene.ColorSpace.SRGB_REC709,
    "r.png": rw.scene.ColorSpace.LIN_REC709,
    "n.png": rw.scene.ColorSpace.LIN_REC709,
  }


def test_glTF_pbr_in_a_node_graph_gets_its_sheen_guard():
  doc = materialx.new_document()
  graph = doc.addNodeGraph("G")
  node = graph.addNodeInstance(materialx.library().getNodeDef("ND_gltf_pbr_surfaceshader"), "P")
  materialx._gltf_sheen(doc)  # the guard alone
  assert max(node.getInputValue("sheen_color")) > 0.0


def test_a_missing_material_raises():
  with pytest.raises(materialx.MaterialXError, match="no_material:Nope"):
    materialx.generate(materialx.Network(document=materialx.new_document(), material="Nope"), LoadReport(), "t")


def test_udim_files_of_documents_are_reported_unbound(tmp_path):
  doc = materialx.new_document()
  mx.readFromXmlString(doc, _DATA_IMAGES.replace('"c.png"', '"c.<UDIM>.png"'))
  report = LoadReport()
  out = materialx.generate(materialx.Network(document=doc, material="M"), report, "t")
  assert report.dropped["mtlx.image.udim"] == 1 and "c.<UDIM>.png" not in {Path(_asset(i.value)).name for i in out.inputs}


def test_a_preview_stage_never_imports_materialx(tmp_path):
  """MaterialX is imported only when a stage has a MaterialX network."""
  import subprocess
  import sys

  stage = new_stage(tmp_path)
  mat, shader = preview_material(stage, "/Looks/Preview")  # a UsdPreviewSurface with a texture
  texture = UsdShade.Shader.Define(stage, "/Looks/Preview/Texture")
  texture.CreateIdAttr("UsdUVTexture")
  png = write_png(tmp_path / "c.png", np.full((4, 4, 3), 128, np.uint8))
  texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
  rgb = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
  shader.GetInput("diffuseColor").ConnectToSource(rgb)
  bind(quad(stage, "/Q"), mat)
  stage.GetRootLayer().Save()
  code = (
    "import sys, raywarp.formats.usd; s = raywarp.formats.usd.read(sys.argv[1]).build();"
    "assert len(s.textures) == 1, 'the preview network was not read';"
    "assert 'MaterialX' not in sys.modules, 'MaterialX imported for a preview-only stage'"
  )
  subprocess.run([sys.executable, "-c", code, str(tmp_path / "scene.usda")], check=True, timeout=120)


# -- conversion (MaterialX + MDL SDK) ----------------------------------------------------------


def test_extension_inputs_survive_the_generator():
  """raywarp's extensions authored on open_pbr_surface (not MaterialX inputs): the detail normal
  map and the wrapped diffuse are carried past the MDL conversion, not reported."""
  _sdk()
  scenes = DATA.parents[1] / "tests" / "data" / "scenes"
  detail = rw.formats.usd.read(str(scenes / "layered_detail.usda"))
  on = _material(detail, "/World/Looks/PlaneDetailOn")
  assert on.extras is not None and "geometry_detail_normal" in dict(on.extras.textures)
  ref = dict(on.extras.textures)["geometry_detail_normal"]
  assert ref.uv_xform[0][0] == pytest.approx(8.0)
  fuzz = rw.formats.usd.read(str(scenes / "layered_fuzz.usda"))
  assert _material(fuzz, "/World/Looks/WrapR05").ext.wrap == pytest.approx(0.5)
  for b in (detail, fuzz):
    assert not [k for k in b.report.dropped if k.startswith("mtlx.")], b.report.dropped


def test_unread_detail_normal_inputs_are_reported():
  """The detail normal map's image address modes / texcoord and the normalmap's space are not
  read (REPEAT, UV set 0, tangent space): authored, they are reported, not silently dropped."""
  _sdk()
  from pxr import Usd

  stage = Usd.Stage.Open(str(DATA.parents[1] / "tests" / "data" / "scenes" / "layered_detail.usda"))
  stage.SetEditTarget(stage.GetSessionLayer())
  looks = "/World/Looks/PlaneDetailOn"
  tex = UsdShade.Shader(stage.GetPrimAtPath(f"{looks}/Tex_geometry_detail_normal"))
  tex.CreateInput("uaddressmode", Sdf.ValueTypeNames.String).Set("clamp")
  UsdShade.Shader(stage.GetPrimAtPath(f"{looks}/Normal_geometry_detail_normal")).CreateInput(
    "space", Sdf.ValueTypeNames.String
  ).Set("object")
  dropped = rw.formats.usd.read(stage).report.dropped
  assert any(k.startswith("mtlx.detail_normal.") and k.endswith(".uaddressmode") for k in dropped), dict(dropped)
  assert any(k.startswith("mtlx.detail_normal.") and k.endswith(".space") for k in dropped), dict(dropped)


def test_constants_convert_through_the_mdl_sdk(tmp_path):
  _sdk()
  stage, _ = _network_stage(tmp_path)
  builder = rw.formats.usd.read(stage)
  m = _material(builder, "/Looks/M")
  assert m.base_color == pytest.approx((0.8, 0.1, 0.1), abs=0.01)
  assert m.specular_roughness == pytest.approx(0.6, abs=0.02) and m.base_metalness == 0.0
  assert builder.report.counts["mtlx.generated"] == 1 and builder.report.counts["mdl.distilled"] == 1


@pytest.mark.parametrize("uv", [None, "st", "st1"])
def test_textures_bind_slots_and_their_uv_set(tmp_path, uv):
  _sdk()
  write_png(tmp_path / "t.png", np.full((2, 2, 3), 200, np.uint8))
  stage, _ = _network_stage(tmp_path, texture="t.png", uv=uv)
  builder = rw.formats.usd.read(stage)
  m = _material(builder, "/Looks/M")
  ref = m.texture(S.BASE_COLOR)
  assert ref is not None and ref.uv_set == (1 if uv == "st1" else 0)
  assert m.base_color == pytest.approx((1.0, 1.0, 1.0))


def test_a_procedural_graph_is_baked(tmp_path):
  _sdk()
  stage, surface = _network_stage(tmp_path)
  noise = UsdShade.Shader.Define(stage, "/Looks/M/Noise")
  noise.CreateIdAttr("ND_noise2d_color3")
  noise.CreateInput("amplitude", Sdf.ValueTypeNames.Float3).Set((0.5, 0.5, 0.5))
  surface.GetInput("base_color").ConnectToSource(noise.CreateOutput("out", Sdf.ValueTypeNames.Color3f))
  builder = rw.formats.usd.read(stage)
  assert _material(builder, "/Looks/M").texture(S.BASE_COLOR) is not None
  assert builder.report.counts["mdl.baked"] >= 1


@pytest.mark.parametrize("node", ["ND_noise3d_color3", "ND_triplanarprojection_color3"])
def test_a_bake_that_reads_the_position_is_reported_as_such(tmp_path, node):
  """A 3D noise of the position or a triplanar projection is baked over the mesh UV square, the
  position read as (u, v, 0) (the SDK's bake state): reported as that, never as an exact UV bake."""
  _sdk()
  write_png(tmp_path / "t.png", np.full((2, 2, 3), 200, np.uint8))
  stage, surface = _network_stage(tmp_path)
  shader = UsdShade.Shader.Define(stage, "/Looks/M/P")
  shader.CreateIdAttr(node)
  for name in ("filex", "filey", "filez") if "triplanar" in node else ():
    shader.CreateInput(name, Sdf.ValueTypeNames.Asset).Set("t.png")
  surface.GetInput("base_color").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Color3f))
  approx = rw.formats.usd.read(stage).report.approximated
  assert [k for k in approx if k.startswith("mdl.baked_position.base_color")], approx
  assert not [k for k in approx if k.startswith("mdl.baked_uv_grid")], approx


def test_a_network_the_library_rejects_is_the_error_material(tmp_path):
  _sdk()
  from raywarp.formats.usd.materials.mdl.baker import ERROR

  stage, surface = _network_stage(tmp_path)
  image = UsdShade.Shader.Define(stage, "/Looks/M/Bad")
  image.CreateIdAttr("ND_no_such_node_color3")
  surface.GetInput("base_color").ConnectToSource(image.CreateOutput("out", Sdf.ValueTypeNames.Color3f))
  builder = rw.formats.usd.read(stage)
  m = _material(builder, "/Looks/M")
  assert m.base_color == ERROR.base_color
  assert any(k.startswith("mtlx.unknown_node:ND_no_such_node_color3") for k in builder.report.dropped)


def test_without_the_mdl_sdk_a_materialx_stage_raises(tmp_path, monkeypatch):
  from raywarp.formats.usd.materials.mdl import sdk

  def missing() -> sdk.Api:
    raise sdk.MdlSdkError("no SDK (test)")

  monkeypatch.setattr(sdk, "api", missing)
  stage, _ = _network_stage(tmp_path)
  with pytest.raises(sdk.MdlSdkError):
    rw.formats.usd.read(stage)


@pytest.mark.parametrize("path", sorted((DATA / "mtlx").glob("*/*.mtlx")), ids=lambda p: p.stem)
def test_every_example_document_generates(path):
  """MaterialX's and OpenPBR's example materials: each generates an MDL module, nothing dropped by
  the generator (a stage reaches them through a MaterialX network inside it; a `.mtlx` layer
  itself is reported as dropped)."""
  doc = materialx.new_document()
  mx.readFromXmlFile(doc, str(path))
  name = doc.getMaterialNodes()[0].getName()
  report = LoadReport()
  out = materialx.generate(materialx.Network(document=doc, material=name), report, path.stem)
  assert materialx.MATERIAL in out.source
  assert not [k for k in report.dropped if k.startswith(("mtlx.", "mdl."))], report.dropped


# name -> OpenPBR inputs (ACEScg colors).
EXAMPLES: dict[str, dict[str, float | bool | tuple[float, float, float]]] = {
  "Gold": {
    "base_color": (0.929, 0.788, 0.374),
    "base_metalness": 1.0,
    "specular_color": (0.987, 1.013, 0.997),
    "specular_roughness": 0.02,
  },
  "Car_Paint": {
    "base_color": (0.1, 0.6, 0.9),
    "specular_ior": 1.6,
    "specular_roughness": 0.3,
    "coat_weight": 1.0,
    "coat_roughness": 0.02,
    "coat_ior": 1.6,
  },
  "Glass": {
    "specular_roughness": 0.0,
    "specular_ior": 1.52,
    "transmission_weight": 1.0,
    "transmission_dispersion_scale": 1.0,
    "transmission_dispersion_abbe_number": 64.0,
  },
  "Brick": {"base_color": (0.196, 0.106, 0.069), "base_diffuse_roughness": 1.0, "specular_roughness": 1.0},
}


type _Inputs = dict[str, float | bool | tuple[float, float, float]]


def _color(inputs: _Inputs, key: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
  """Color input `key` of example `inputs` (`default` when unset)."""
  value = inputs.get(key, default)
  if not isinstance(value, tuple):
    raise TypeError(f"{key} is {value!r}, not a color")
  return value


def _number(inputs: _Inputs, key: str, default: float) -> float:
  """Float input `key` of example `inputs` (`default` when unset)."""
  value = inputs.get(key, default)
  if isinstance(value, tuple):
    raise TypeError(f"{key} is {value!r}, not a number")
  return float(value)


def _asset(value: object) -> str:
  """The authored path of a generated module's texture argument."""
  assert isinstance(value, Sdf.AssetPath)
  return value.path


def _rec709(c: float | bool | tuple[float, float, float]) -> tuple[float, float, float]:
  """ACEScg -> linear Rec.709 (Bradford), clamped to [0, 1], of an `EXAMPLES` colour."""
  assert isinstance(c, tuple)
  m = np.array(
    [[1.70505099, -0.62179212, -0.08325887], [-0.13025642, 1.14080474, -0.01054832], [-0.02400336, -0.12896898, 1.15297233]]
  )
  r, g, b = np.clip(m @ np.asarray(c), 0.0, 1.0)
  return float(r), float(g), float(b)


def _add_mtlx_surface(stage, mat, shader_id: str, color=(0.1, 0.9, 0.1)):
  """Adds a MaterialX surface `shader_id` on the `mtlx` context of material `mat`."""
  from pxr import Sdf, UsdShade

  shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Mtlx")
  shader.CreateIdAttr(shader_id)
  shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(color)
  mat.CreateSurfaceOutput("mtlx").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
  return shader


def _mtlx_and_mdl_stage(tmp_path, shader_id: str):
  from pxr import Sdf

  stage = new_stage(tmp_path)
  inputs = {"diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, (0.9, 0.1, 0.1))}
  mat = mdl_material(stage, "/Looks/Both", "OmniPBR.mdl", "OmniPBR", **inputs)
  _add_mtlx_surface(stage, mat, shader_id)
  bind(quad(stage, "/Q"), mat)
  return stage


@pytest.mark.parametrize(
  ("contexts", "green"),
  [(None, True), (("mtlx", "mdl"), True), (("mdl", "mtlx"), False)],
)
def test_materialx_and_mdl_order_follows_the_option(tmp_path, contexts, green):
  _sdk()
  adapter = rw.formats.usd.Options() if contexts is None else rw.formats.usd.Options(material_contexts=contexts)
  builder = rw.formats.usd.read(_mtlx_and_mdl_stage(tmp_path, "ND_open_pbr_surface_surfaceshader"), adapter=adapter)
  m = _material(builder, "/Looks/Both")
  assert (m.base_color[1] > 0.5) == green and (m.base_color[0] > 0.5) != green


def test_every_materialx_surface_is_taken(tmp_path):
  """Any surfaceshader nodedef converts (`surface_unlit` too): the MaterialX surface wins in the
  default order."""
  _sdk()
  builder = rw.formats.usd.read(_mtlx_and_mdl_stage(tmp_path, "ND_surface_unlit"))
  assert builder.report.counts["mtlx.generated"] == 1
  m = _material(builder, "/Looks/Both")  # unlit: its emission is its color, no base to light
  assert m.ext.flags & rw.scene.MaterialFlags.UNLIT and m.base_color == (0.0, 0.0, 0.0)


def test_universal_openpbr_surface(tmp_path):
  _sdk()
  stage = new_stage(tmp_path)
  mat = UsdShade.Material.Define(stage, "/Looks/U")
  shader = UsdShade.Shader.Define(stage, "/Looks/U/S")
  shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
  shader.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(1.0)
  mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  assert _material(builder, "/Looks/U").base_metalness == pytest.approx(1.0) and builder.report.counts["mtlx.generated"] == 1


# -- UsdShade OpenPBR networks against UsdPreviewSurface -------------------------------------------

BRICK_TEXTURED = "Brick_Textured"
_RES = (320, 64)


def _brick_png(path: Path) -> Path:
  """A 16 x 16 sRGB brick pattern (rows of red bricks, grey mortar)."""
  y, x = np.mgrid[0:16, 0:16]
  mortar = (y % 4 == 0) | (((x + 4 * ((y // 4) % 2)) % 8) == 0)
  rgb = np.where(mortar[..., None], [150, 150, 140], [140, 60, 40]).astype(np.uint8)
  return write_png(path, rgb)


def _openpbr_shader(stage, path: str, inputs, texture: str | None = None):
  from pxr import Sdf, UsdShade

  mat = UsdShade.Material.Define(stage, path)
  shader = UsdShade.Shader.Define(stage, f"{path}/OpenPBR")
  shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
  for name, value in inputs.items():
    if isinstance(value, tuple):
      inp = shader.CreateInput(name, Sdf.ValueTypeNames.Color3f)
      inp.Set(value)
      inp.GetAttr().SetColorSpace("acescg")
    elif isinstance(value, bool):
      shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)
    else:
      shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
  if texture is not None:
    uv = UsdShade.Shader.Define(stage, f"{path}/UV")
    uv.CreateIdAttr("ND_texcoord_vector2")
    image = UsdShade.Shader.Define(stage, f"{path}/Image")
    image.CreateIdAttr("ND_image_color3")
    image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    image.GetInput("file").GetAttr().SetColorSpace("srgb_texture")
    image.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.CreateOutput("out", Sdf.ValueTypeNames.Float2))
    shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(
      image.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
    )
  mat.CreateSurfaceOutput("mtlx").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
  return mat


def _preview_shader(stage, path: str, name: str, texture: str | None = None):
  """The UsdPreviewSurface an author would write for the same example."""
  from pxr import Sdf, UsdShade

  inputs = EXAMPLES["Brick" if name == BRICK_TEXTURED else name]
  mat = UsdShade.Material.Define(stage, path)
  shader = UsdShade.Shader.Define(stage, f"{path}/Preview")
  shader.CreateIdAttr("UsdPreviewSurface")
  glass = name == "Glass"
  color = (1.0, 1.0, 1.0) if glass else _rec709(_color(inputs, "base_color", (0.8, 0.8, 0.8)))
  floats = {
    "metallic": _number(inputs, "base_metalness", 0.0),
    "roughness": _number(inputs, "specular_roughness", 0.3),
    "ior": _number(inputs, "specular_ior", 1.5),
    "clearcoat": _number(inputs, "coat_weight", 0.0),
    "clearcoatRoughness": _number(inputs, "coat_roughness", 0.0),
    "opacity": 0.15 if glass else 1.0,  # the core's alpha of a transmissive material
  }
  for key, value in floats.items():
    shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(value)
  diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
  if texture is None:
    diffuse.Set(color)
  else:
    tex = UsdShade.Shader.Define(stage, f"{path}/Image")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    diffuse.ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
  mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
  return mat


def _stage(tmp_path: Path, kind: str):
  """Five spheres in a row (gold, car paint, glass, brick, textured brick) under one camera."""
  from pxr import UsdGeom

  stage = new_stage(tmp_path, f"{kind}.usda")
  _brick_png(tmp_path / "brick.png")
  names = [*EXAMPLES, BRICK_TEXTURED]
  for i, name in enumerate(names):
    sphere = UsdGeom.Sphere.Define(stage, f"/World/{name}")
    sphere.CreateRadiusAttr(0.45)
    UsdGeom.XformCommonAPI(sphere).SetTranslate((i - 2.0, 0.0, 0.0))
    texture = "brick.png" if name == BRICK_TEXTURED else None
    if kind == "openpbr":
      mat = _openpbr_shader(stage, f"/Looks/{name}", EXAMPLES["Brick" if texture else name], texture)
    else:
      mat = _preview_shader(stage, f"/Looks/{name}", name, texture)
    bind(sphere, mat)
  camera = UsdGeom.Camera.Define(stage, "/World/Camera")
  camera.CreateFocalLengthAttr(20.0)
  camera.CreateHorizontalApertureAttr(20.955)
  camera.CreateVerticalApertureAttr(20.955 * _RES[1] / _RES[0])
  UsdGeom.XformCommonAPI(camera).SetTranslate((0.0, -6.0, 0.0))
  UsdGeom.XformCommonAPI(camera).SetRotate((90.0, 0.0, 0.0))
  stage.Save()
  return stage


def _render(host, settings: rw.RenderSettings | None = None) -> np.ndarray:
  """The stage at the fixed reference exposure (both stages compared at the same exposure, not
  each metered on its own)."""
  lit = host.with_lighting(rw.Lighting.studio())
  fixed = (settings or rw.RenderSettings()).replace(**{"exposure.mode": "manual", "exposure.ev100": DEFAULT_EV100})
  r = rw.Renderer(rw.upload(lit, device="cpu"), nworld=1, cameras=None, resolution=_RES, settings=fixed)
  return required(r.render().rgb).numpy().reshape(_RES[1], _RES[0], 4)[..., :3].astype(np.float32) / 255.0


def test_usd_openpbr_networks_convert_to_their_examples(tmp_path):
  """The OpenPBR examples' main inputs survive the MDL distillation: gold is a metal of its color,
  the brick a rough dielectric of its color, the glass transmissive, the car paint coated; the
  textured brick binds its base color."""
  _sdk()
  builder = rw.formats.usd.read(_stage(tmp_path, "openpbr"))
  mats = {m.name: m for m in builder.materials}
  gold, brick, glass, paint = (mats[f"/Looks/{n}"] for n in ("Gold", "Brick", "Glass", "Car_Paint"))
  assert gold.base_metalness == pytest.approx(1.0) and gold.base_color == pytest.approx(
    _rec709(EXAMPLES["Gold"]["base_color"]), abs=0.03
  )
  assert brick.base_metalness == 0.0 and brick.base_color == pytest.approx(_rec709(EXAMPLES["Brick"]["base_color"]), abs=0.02)
  assert brick.specular_roughness > 0.9 and glass.transmission_weight > 0.9 and paint.coat_weight > 0.9
  textured = mats[f"/Looks/{BRICK_TEXTURED}"]
  assert textured.texture(S.BASE_COLOR) is not None
  assert builder.report.counts["mtlx.generated"] == 5
  assert not [k for k in builder.report.dropped if k.startswith("material.")]


def test_usd_openpbr_renders_close_to_preview_surface(tmp_path):
  """ꟻLIP per sphere of the OpenPBR stage against its UsdPreviewSurface twin: the same where the
  mapping is exact (gold; brick and textured brick with Lambert diffuse: their
  base_diffuse_roughness 1 is EON's rough diffuse, which preview surfaces cannot express); the
  car paint differs by OpenPBR's coat semantics (no base specular under an equal-IOR coat, a
  stronger coat at IOR 1.6, the coat's darkening of the base) and the glass by transmission (a
  specular-only alpha layer) against preview opacity."""
  _sdk()
  flip_evaluator = pytest.importorskip("flip_evaluator")
  openpbr = rw.formats.usd.read(_stage(tmp_path, "openpbr")).build()
  ref = _render(rw.formats.usd.read(_stage(tmp_path, "preview")).build())
  width = _RES[0] // 5

  def flips_of(ours: np.ndarray) -> dict[str, float]:
    flips = {}
    for i, name in enumerate([*EXAMPLES, BRICK_TEXTURED]):
      a, b = ours[:, i * width : (i + 1) * width], ref[:, i * width : (i + 1) * width]
      _, mean, _ = flip_evaluator.evaluate(b, a, "LDR")
      flips[name] = float(mean)
    return flips

  flips = flips_of(_render(openpbr))
  lambert = flips_of(_render(openpbr, rw.RenderSettings(openpbr=rw.settings.OpenPBR(diffuse_roughness=False))))
  print(
    "FLIP per sphere:", {k: round(v, 4) for k, v in flips.items()}, "Lambert:", {k: round(v, 4) for k, v in lambert.items()}
  )
  assert flips["Gold"] < 0.01 and lambert["Brick"] < 0.01 and lambert[BRICK_TEXTURED] < 0.01
  # EON's rough diffuse (flatter, retro-reflective) against Lambert; the glass differs by design
  # (alpha layer vs preview opacity): no bound on it.
  assert flips["Brick"] < 0.1 and flips[BRICK_TEXTURED] < 0.1
  # Two material models: OpenPBR darkens the base under the coat (coat_darkening, kept by the MDL
  # distillation), UsdPreviewSurface does not.
  assert flips["Car_Paint"] < 0.2


def test_fuzz_over_a_constant_base_keeps_the_base_color() -> None:
  """OpenPBR fuzz over a constant base (the feature scene's cloth): the base is read under the
  sheen, not the distiller's diffuse with the fuzz tint folded in (it read pink, (0.51, 0.21,
  0.33), for a pink fuzz over a dark red base); the fuzz keeps its color and roughness."""
  path = Path(__file__).resolve().parents[3] / "tests" / "data" / "scenes" / "layered_fuzz.usda"
  builder = rw.formats.usd.read(str(path))
  for name in ("/World/Looks/PinkR03", "/World/Looks/BlueR06", "/World/Looks/WhiteR03"):
    m = _material(builder, name)
    assert m.base_color == pytest.approx((0.3, 0.02, 0.03), abs=2e-3), name
    assert m.extras is not None and m.extras.fuzz_weight > 0.5, name
  pink = _material(builder, "/World/Looks/PinkR03").extras
  assert pink is not None and pink.fuzz_color == pytest.approx((1.0, 0.45, 0.7), abs=1e-3)
  assert pink.fuzz_roughness == pytest.approx(0.3, abs=1e-3)
