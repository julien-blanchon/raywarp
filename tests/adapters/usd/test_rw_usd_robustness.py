"""USD import robustness: out-of-range display values, instanced mesh attributes, material routing
by render context."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rw_helpers import bind, mdl_material, new_stage, preview_material, quad, write_png
from rw_helpers import material_named as _material

pytest.importorskip("pxr")

from pxr import Sdf, UsdGeom, UsdLux, UsdShade

import raywarp as rw
import raywarp.formats.usd

# -- displayColor / displayOpacity / mesh-light radiance -------------------------------------------


@pytest.mark.parametrize("primitive", [False, True])
def test_hdr_display_color_is_clamped_and_reported(tmp_path, primitive):
  """displayColor (1.5, 0.4, 0.4) (HDR authoring) loads clamped to [0, 1], reported; the stage
  is not aborted."""
  stage = new_stage(tmp_path)
  prim = UsdGeom.Sphere.Define(stage, "/S") if primitive else quad(stage, "/Q")
  prim.CreateDisplayColorAttr([(1.5, 0.4, -0.2)])
  b = rw.formats.usd.read(stage)
  m = _material(b, "<displayColor>")
  assert m.base_color == pytest.approx((1.0, 0.4, 0.0))
  assert b.report.approximated["Mesh.primvars:displayColor.clamped"] == 1


def test_non_finite_display_values_are_reported_and_unset(tmp_path):
  """A NaN displayColor / displayOpacity is reported and left unset: the grey, opaque fallback."""
  stage = new_stage(tmp_path)
  q = quad(stage, "/Q")
  q.CreateDisplayColorAttr([(math.nan, 0.4, 0.4)])
  q.CreateDisplayOpacityAttr([math.nan])
  b = rw.formats.usd.read(stage)
  (m,) = b.materials
  assert m.base_color == (0.5, 0.5, 0.5) and m.geometry_opacity == 1.0
  assert b.report.dropped["Mesh.primvars:displayColor.non_finite"] == 1
  assert b.report.dropped["Mesh.primvars:displayOpacity.non_finite"] == 1


@pytest.mark.parametrize(("authored", "expected"), [(1.5, None), (-0.5, 0.0)])
def test_display_opacity_outside_unit_range_is_clamped(tmp_path, authored, expected):
  """displayOpacity is clamped to [0, 1] (reported): 1.5 is opaque, -0.5 fully transparent."""
  stage = new_stage(tmp_path)
  quad(stage, "/Q").CreateDisplayOpacityAttr([authored])
  b = rw.formats.usd.read(stage)
  (m,) = b.materials
  assert m.geometry_opacity == (1.0 if expected is None else expected)
  assert b.report.approximated["Mesh.primvars:displayOpacity.clamped"] == 1


def test_nan_mesh_light_radiance_drops_the_light_not_the_stage(tmp_path):
  """A MeshLightAPI mesh with a NaN intensity keeps its surface, does not emit, and is reported."""
  stage = new_stage(tmp_path)
  q = quad(stage, "/Q")
  UsdLux.MeshLightAPI.Apply(q.GetPrim())
  q.GetPrim().CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(math.nan)
  quad(stage, "/Other")
  b = rw.formats.usd.read(stage)
  assert all(m.emission_luminance == 0.0 for m in b.materials)
  assert b.report.dropped["MeshLightAPI.invalid_radiance"] == 1
  assert b.build().ninstance == 2


# -- instanced assets read what their instance's material reads ------------------------------------


def _instanced_stage(tmp_path, *, instanced: bool):
  """A quad with st, a mirrored st1 and a COLOR_0 vertex-color primvar, under /Proto; bound
  (instanceable references /I0, /I1 when `instanced`, else directly) to a material multiplying
  its base color by COLOR_0 and reading roughness through st1."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "t.png", np.full((2, 2, 3), 200, np.uint8))
  UsdGeom.Xform.Define(stage, "/Proto")
  mesh = quad(stage, "/Proto/Mesh")
  api = UsdGeom.PrimvarsAPI(mesh)
  api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set([(0, 0), (1, 0), (1, 1), (0, 1)])
  api.CreatePrimvar("st1", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set([(1, 0), (0, 0), (0, 1), (1, 1)])
  colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
  api.CreatePrimvar("COLOR_0", Sdf.ValueTypeNames.Color3fArray, "vertex").Set(colors)
  mat, surface = preview_material(stage, "/Looks/M")
  mat.GetPrim().CreateAttribute("raywarp:vertexColor", Sdf.ValueTypeNames.String).Set("COLOR_0")
  reader = UsdShade.Shader.Define(stage, "/Looks/M/Reader")
  reader.CreateIdAttr("UsdPrimvarReader_float2")
  reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st1")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Tex")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("t.png")
  tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
  surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(tex.CreateOutput("r", Sdf.ValueTypeNames.Float))
  if instanced:
    stage.GetPrimAtPath("/Proto").SetSpecifier(Sdf.SpecifierClass)
    for i in range(2):
      prim = stage.DefinePrim(f"/I{i}", "Xform")
      prim.GetReferences().AddInternalReference("/Proto")
      prim.SetInstanceable(True)
      bind(prim, mat)
  else:
    bind(mesh, mat)
  return stage


def test_instances_read_the_vertex_colors_and_uv_set_their_material_reads(tmp_path):
  """Prototype meshes inheriting the instance's binding read the vertex colors and the UV set
  that material reads, as the same mesh bound directly does (no white base color, no unread
  st1)."""
  (tmp_path / "a").mkdir()
  (tmp_path / "b").mkdir()
  direct = rw.formats.usd.read(_instanced_stage(tmp_path / "a", instanced=False)).build()
  b = rw.formats.usd.read(_instanced_stage(tmp_path / "b", instanced=True))
  assert "Mesh.primvars:COLOR_0" not in b.report.dropped and "Mesh.primvars:st1" not in b.report.dropped
  inst = b.build()
  assert inst.ninstance == 2 and inst.nmesh == 1
  np.testing.assert_array_equal(inst.meshes.colors, direct.meshes.colors)
  np.testing.assert_array_equal(inst.meshes.uvs, direct.meshes.uvs)
  assert len(inst.meshes.colors) and (inst.meshes.color_idx >= 0).all()


def test_instances_binding_different_materials_each_get_their_attributes(tmp_path):
  """Two instances of one prototype bound to a vertex-color material and a plain one: the
  prototype's mesh is read once per distinct need (colors for the first only)."""
  stage = _instanced_stage(tmp_path, instanced=True)
  plain, _ = preview_material(stage, "/Looks/Plain")
  bind(stage.GetPrimAtPath("/I1"), plain)
  scene = rw.formats.usd.read(stage).build()
  assert scene.nmesh == 2
  meshes = scene.meshes
  colored = [
    bool((meshes.color_idx[int(a) : int(a) + int(n)] >= 0).all()) for a, n in zip(meshes.tri_adr, meshes.tri_num, strict=True)
  ]
  assert sorted(colored) == [False, True]


# -- material routing by render context ------------------------------------------------------------


def _preview_and_mtlx(tmp_path):
  """A material with a universal UsdPreviewSurface (red) and an OpenPBR surface on the `mtlx`
  context (green)."""
  stage = new_stage(tmp_path)
  mat, _ = preview_material(stage, "/Looks/M", color=(0.9, 0.1, 0.1))
  shader = UsdShade.Shader.Define(stage, "/Looks/M/Mtlx")
  shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
  shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.9, 0.1))
  mat.CreateSurfaceOutput("mtlx").ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q"), mat)
  return stage


@pytest.mark.parametrize(("contexts", "green"), [(None, True), (("mdl", "mtlx"), True), (("mdl",), False)])
def test_requested_render_context_wins_over_the_universal_surface(tmp_path, contexts, green):
  """USD's rule: a render context of `material_contexts` with a surface it maps wins over the
  universal output (the preview approximation); a context not requested is never read."""
  pytest.importorskip("MaterialX")
  adapter = rw.formats.usd.Options() if contexts is None else rw.formats.usd.Options(material_contexts=contexts)
  m = _material(rw.formats.usd.read(_preview_and_mtlx(tmp_path), adapter=adapter), "/Looks/M")
  assert (m.base_color[1] > 0.5) == green and (m.base_color[0] > 0.5) != green


def test_last_resort_shader_child_is_a_surface_never_a_texture(tmp_path):
  """A material without connected surface outputs takes its first child that is a surface shader
  (a UsdUVTexture before it is skipped); with only a texture child it has no surface."""
  stage = new_stage(tmp_path)
  for path, with_surface in (("/Looks/A", True), ("/Looks/B", False)):
    mat = UsdShade.Material.Define(stage, path)
    tex = UsdShade.Shader.Define(stage, f"{path}/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    if with_surface:
      surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
      surface.CreateIdAttr("UsdPreviewSurface")
      surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.2, 0.9))
    bind(quad(stage, f"/Q{path[-1]}"), mat)
  b = rw.formats.usd.read(stage)
  assert _material(b, "/Looks/A").base_color == pytest.approx((0.1, 0.2, 0.9))
  assert _material(b, "/Looks/B").base_color == (0.5, 0.5, 0.5)
  assert b.report.dropped["material.no_surface"] == 1
  assert not any(k.startswith("material.unknown_shader") for k in b.report.dropped)


def test_osl_source_asset_is_reported_not_sent_to_the_mdl_baker(tmp_path):
  """A surface implemented by an OSL source asset is an unknown shader (reported, grey): only an
  `mdl` source asset reaches the MDL baker."""
  stage = new_stage(tmp_path)
  mat = UsdShade.Material.Define(stage, "/Looks/Osl")
  shader = UsdShade.Shader.Define(stage, "/Looks/Osl/Shader")
  shader.SetSourceAsset(Sdf.AssetPath("plastic.osl"), "osl")
  mat.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q"), mat)
  # A universal source asset (USD returns it for every source type, "mdl" included).
  universal = UsdShade.Material.Define(stage, "/Looks/Universal")
  shader = UsdShade.Shader.Define(stage, "/Looks/Universal/Shader")
  shader.SetSourceAsset(Sdf.AssetPath("plastic.osl"))
  universal.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("out", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q2"), universal)
  b = rw.formats.usd.read(stage)
  assert _material(b, "/Looks/Osl").base_color == (0.5, 0.5, 0.5)
  assert _material(b, "/Looks/Universal").base_color == (0.5, 0.5, 0.5)
  assert b.report.dropped["material.unknown_shader:osl"] == 1
  assert sum(n for k, n in b.report.dropped.items() if k.startswith("material.unknown_shader")) == 2
  assert not any(k.startswith("mdl.") for k in (*b.report.dropped, *b.report.counts))


def test_mdl_context_wins_over_a_preview_fallback(tmp_path):
  """An asset shipping MDL with a preview approximation renders its MDL material by default;
  `material_contexts=("mtlx",)` (no MDL) takes the preview."""
  stage = new_stage(tmp_path)
  inputs = {"diffuse_color_constant": (Sdf.ValueTypeNames.Color3f, (0.9, 0.1, 0.1))}
  mat = mdl_material(stage, "/Looks/Both", "OmniPBR.mdl", "OmniPBR", **inputs)
  preview = UsdShade.Shader.Define(stage, "/Looks/Both/Preview")
  preview.CreateIdAttr("UsdPreviewSurface")
  preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.9, 0.1))
  mat.CreateSurfaceOutput().ConnectToSource(preview.CreateOutput("surface", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q"), mat)
  mdl = _material(rw.formats.usd.read(stage), "/Looks/Both")
  assert mdl.base_color[0] > 0.5 and mdl.base_color[1] < 0.5
  approx = _material(rw.formats.usd.read(stage, adapter=rw.formats.usd.Options(material_contexts=("mtlx",))), "/Looks/Both")
  assert approx.base_color[1] > 0.5 and approx.base_color[0] < 0.5


@pytest.mark.parametrize(
  "bad",
  [
    {"displacement": "off"},
    {"double_sided": 1},
    {"purposes": "render"},
    {"material_contexts": "mtlx"},
    {"material_contexts": ("mtlx", "glslfx")},
  ],
)
def test_usd_options_reject_wrong_types_at_construction(bad):
  """A wrongly typed option raises when built (a truthy string must not turn displacement on, a
  string of purposes must not become a substring test)."""
  with pytest.raises(ValueError, match=f"usd.Options.{next(iter(bad))}"):
    rw.formats.usd.Options(**bad)
