"""Host scene -> rw.upload -> render (Warp CPU device): a USD stage end to end."""

import numpy as np
import pytest
from conftest import required

pytest.importorskip("pxr")
from pxr import Sdf, UsdGeom, UsdLux, UsdShade
from rw_helpers import bind, new_stage, preview_material, quadrant_png

import raywarp as rw
import raywarp.formats.usd

_RES = 32


def _stage(tmp_path):
  """A textured quad (scaled x2, four colored quadrants) under a top-down camera, and a cube."""
  stage = new_stage(tmp_path)
  quadrant_png(tmp_path / "quad.png")
  quad = UsdGeom.Mesh.Define(stage, "/World/Quad")
  quad.CreatePointsAttr([(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)])
  quad.CreateFaceVertexCountsAttr([4])
  quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
  st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
  st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
  UsdGeom.XformCommonAPI(quad).SetScale((2.0, 2.0, 2.0))
  mat, surface = preview_material(stage, "/World/Looks/Tex")
  tex = UsdShade.Shader.Define(stage, "/World/Looks/Tex/Image")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("quad.png")
  surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
  )
  bind(quad, mat)
  cube = UsdGeom.Cube.Define(stage, "/World/Cube")
  cube.CreateSizeAttr(0.2)
  UsdGeom.XformCommonAPI(cube).SetTranslate((0.9, 0.9, 0.2))
  red, _ = preview_material(stage, "/World/Looks/Red", color=(0.8, 0.1, 0.1))
  bind(cube, red)
  sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
  sun.CreateIntensityAttr(3.0)
  sun.CreateAngleAttr(0.0)  # the core's directional lights are hard
  camera = UsdGeom.Camera.Define(stage, "/World/Camera")
  UsdGeom.XformCommonAPI(camera).SetTranslate((0.0, 0.0, 8.0))
  stage.Save()
  return tmp_path / "scene.usda"


def test_usd_stage_renders_through_the_core(tmp_path):
  host = rw.formats.usd.load(_stage(tmp_path))
  scene = rw.upload(host, device="cpu")
  assert scene.ninstance == 2 and list(scene.cameras) == ["/World/Camera"]
  assert scene.meshes.nmesh == 1  # the quad as authored: its uniform x2 scale is in the instance transform
  np.testing.assert_allclose(scene.meshes.points[0].max(axis=0), [0.5, 0.5, 0.0])
  assert not [k for k in host.report.dropped if k.startswith("gpu.")]
  settings = rw.RenderSettings(
    outputs=rw.Outputs.of("rgb", "seg", "albedo"), exposure=rw.settings.Exposure(mode="manual", ev100=3.0)
  )
  r = rw.Renderer(scene, nworld=2, cameras=None, resolution=(_RES, _RES), settings=settings)
  out = r.render()
  seg = required(out.seg).numpy()
  assert np.array_equal(seg[0], seg[1])
  assert set(np.unique(seg[0])) == {-1, 0, 1}  # quad, cube, background
  albedo = required(out.albedo).numpy().reshape(2, _RES, _RES, 3)[0]
  # USD st has its origin at the image's bottom-left: the image is seen as authored from above.
  lo, hi = _RES * 3 // 8, _RES * 5 // 8
  quadrants = {"red": albedo[lo, lo], "green": albedo[lo, hi], "blue": albedo[hi, lo], "white": albedo[hi, hi]}
  np.testing.assert_allclose(quadrants["red"], [1, 0, 0], atol=0.02)
  np.testing.assert_allclose(quadrants["green"], [0, 1, 0], atol=0.02)
  np.testing.assert_allclose(quadrants["blue"], [0, 0, 1], atol=0.02)
  np.testing.assert_allclose(quadrants["white"], [1, 1, 1], atol=0.02)
  red = required(out.rgb).numpy().reshape(2, _RES, _RES, 4)[0, lo, lo]
  assert red[0] > 60 and red[1] < 30 and red[2] < 30  # lit by the sun (3 lux at ev100 3)
