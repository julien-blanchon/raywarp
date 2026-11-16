"""Regression tests of USD adapter bugs (plane axes, channel scale / bias, light normalize,
binding purpose, texture fallbacks, render visibility, mesh lights, camera exposure, dome poles)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rw_helpers import bind, new_stage, preview_material, quad, write_hdr, write_png
from rw_helpers import material_named as _material

pytest.importorskip("pxr")

from pxr import Sdf, UsdGeom, UsdLux, UsdShade

import raywarp as rw
import raywarp.formats.usd
from raywarp.core.pipeline.upload.lights import derive_lights

S = rw.scene.Slot


def _instance(scene: rw.Scene, path: str) -> int:
  inst = scene.instances
  return next(i for i in range(scene.ninstance) if inst.paths[int(inst.prim[i])] == path)


@pytest.mark.parametrize(("axis", "width_axis", "length_axis"), [("X", 2, 1), ("Y", 0, 2), ("Z", 0, 1)])
def test_b31_plane_width_and_length_follow_the_axis(tmp_path, axis, width_axis, length_axis):
  """UsdGeomPlane: width along X (axis Z / Y) or Z (axis X); length along Y (axis Z / X) or Z (axis
  Y)."""
  stage = new_stage(tmp_path)
  plane = UsdGeom.Plane.Define(stage, "/Plane")
  plane.CreateAxisAttr(axis)
  plane.CreateWidthAttr(10.0)
  plane.CreateLengthAttr(4.0)
  s = rw.formats.usd.read(stage).build()
  i = _instance(s, "/Plane")
  inst = s.instances
  # World extent of the plane's two half sizes along their local axes (X: size 0, Y: size 1).
  half_x = np.abs(inst.xform[i, :, 0]) * inst.size[i, 0]
  half_y = np.abs(inst.xform[i, :, 1]) * inst.size[i, 1]
  extent = half_x + half_y
  assert extent[width_axis] == pytest.approx(5.0) and extent[length_axis] == pytest.approx(2.0)
  np.testing.assert_allclose(np.abs(inst.xform[i, :, 2]), np.eye(3)["XYZ".index(axis)], atol=1e-6)


def test_b32_single_channel_scale_and_bias_come_from_that_channel(tmp_path):
  """UsdUVTexture outputs:g = texel.g * scale.g + bias.g: the slot stores that channel's factors."""
  stage = new_stage(tmp_path)
  write_png(tmp_path / "orm.png", np.zeros((2, 2, 3), np.uint8))
  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/ORM")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("orm.png")
  tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((0.1, 0.5, 0.25, 1.0))
  tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((0.01, 0.2, 0.3, 0.0))
  shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "g")
  shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "b")
  bind(quad(stage, "/Q"), mat)
  m = _material(rw.formats.usd.read(stage), "/Looks/M")
  rough, metal = m.texture(S.SPECULAR_ROUGHNESS), m.texture(S.BASE_METALNESS)
  assert rough.channel == 1 and rough.scale[0] == pytest.approx(0.5) and rough.bias[0] == pytest.approx(0.2)
  assert metal.channel == 2 and metal.scale[0] == pytest.approx(0.25) and metal.bias[0] == pytest.approx(0.3)


def test_b39_normalize_divides_by_the_surface_area(tmp_path):
  """UsdLux normalize: luminance / world surface area (sphere 4 pi r^2, cylinder 2 pi r l)."""
  stage = new_stage(tmp_path)
  sphere = UsdLux.SphereLight.Define(stage, "/Sphere")
  sphere.CreateRadiusAttr(0.5)
  sphere.CreateIntensityAttr(100.0)
  sphere.CreateNormalizeAttr(True)
  cyl = UsdLux.CylinderLight.Define(stage, "/Cyl")
  cyl.CreateRadiusAttr(0.1)
  cyl.CreateLengthAttr(2.0)
  cyl.CreateIntensityAttr(100.0)
  cyl.CreateNormalizeAttr(True)
  disk = UsdLux.DiskLight.Define(stage, "/Disk")
  disk.CreateRadiusAttr(0.5)
  disk.CreateIntensityAttr(100.0)
  disk.CreateNormalizeAttr(True)
  sun = UsdLux.DistantLight.Define(stage, "/Sun")
  sun.CreateIntensityAttr(100.0)
  sun.CreateNormalizeAttr(True)
  s = rw.formats.usd.read(stage).build()
  li = derive_lights(s.lights)
  by = {li.paths[i]: i for i in range(s.nlight)}
  np.testing.assert_allclose(li.radiance[by["/Sphere"]], 100.0 / (4.0 * math.pi * 0.25), rtol=1e-6)
  np.testing.assert_allclose(li.radiance[by["/Cyl"]], 100.0 / (2.0 * math.pi * 0.1 * 2.0), rtol=1e-6)
  np.testing.assert_allclose(li.radiance[by["/Disk"]], 100.0 / (math.pi * 0.25), rtol=1e-6)
  np.testing.assert_allclose(li.radiance[by["/Sun"]], 100.0)  # illuminance [lux]
  assert not any("normalize" in k for k in s.report.approximated)


def test_b40_full_purpose_binding_wins(tmp_path):
  """A `material:binding:full` binding is the one a final-quality renderer uses."""
  stage = new_stage(tmp_path)
  preview, _ = preview_material(stage, "/Looks/Preview", (0.0, 0.0, 1.0))
  full, _ = preview_material(stage, "/Looks/Full", (1.0, 0.0, 0.0))
  mesh = quad(stage, "/Q")
  api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
  api.Bind(preview)
  api.Bind(full, materialPurpose=UsdShade.Tokens.full)
  s = rw.formats.usd.read(stage).build()
  assert s.materials.names[int(s.instances.material[0])] == "/Looks/Full"


def test_b40_missing_texture_uses_the_fallback(tmp_path):
  """UsdUVTexture `fallback` (default (0, 0, 0, 1)) when its file does not resolve, not white."""
  stage = new_stage(tmp_path)
  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/Base")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("missing.png")
  tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set((0.2, 0.4, 0.6, 1.0))
  shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
  rough = UsdShade.Shader.Define(stage, "/Looks/M/Rough")
  rough.CreateIdAttr("UsdUVTexture")
  rough.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("missing_too.png")
  shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(rough.ConnectableAPI(), "r")
  bind(quad(stage, "/Q"), mat)
  m = _material(rw.formats.usd.read(stage), "/Looks/M")
  assert m.base_color == pytest.approx((0.2, 0.4, 0.6)) and m.texture(S.BASE_COLOR) is None
  assert m.specular_roughness == 0.0 and m.texture(S.SPECULAR_ROUGHNESS) is None  # the default fallback's red


def test_b40_clearcoat_roughness_default(tmp_path):
  """UsdPreviewSurface clearcoatRoughness defaults to 0.01."""
  stage = new_stage(tmp_path)
  mat, _ = preview_material(stage, "/Looks/M", clearcoat=1.0)
  bind(quad(stage, "/Q"), mat)
  m = _material(rw.formats.usd.read(stage), "/Looks/M")
  assert m.coat_weight == 1.0 and m.coat_roughness == pytest.approx(0.01)
  # glTF's coat: IOR 1.5, the base unchanged under it (canonical_design.md §1.4), over the base's
  # mapped normal (UsdPreviewSurface's one `normal` input shades every lobe).
  assert m.coat_ior == 1.5 and m.ext.flags & rw.scene.MaterialFlags.COAT_GLTF
  assert m.ext.flags & rw.scene.MaterialFlags.COAT_BASE_NORMAL


def test_preview_without_a_coat_has_no_coat_flags(tmp_path):
  stage = new_stage(tmp_path)
  mat, _ = preview_material(stage, "/Looks/M")
  bind(quad(stage, "/Q"), mat)
  m = _material(rw.formats.usd.read(stage), "/Looks/M")
  assert not m.ext.flags & (rw.scene.MaterialFlags.COAT_GLTF | rw.scene.MaterialFlags.COAT_BASE_NORMAL)


def test_preview_specular_workflow_dielectric_is_an_openpbr_weight(tmp_path):
  """A constant dielectric specularColor: F0 = specular_weight x 0.04 x specular_color."""
  from pxr import Sdf

  stage = new_stage(tmp_path)
  mat, shader = preview_material(stage, "/Looks/M", color=(0.3, 0.3, 0.3))
  shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
  shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.02, 0.01, 0.02))
  bind(quad(stage, "/Q"), mat)
  m = _material(rw.formats.usd.read(stage), "/Looks/M")
  assert m.base_metalness == 0.0 and m.base_color == pytest.approx((0.3, 0.3, 0.3))
  assert m.specular_weight == pytest.approx(0.5) and m.specular_color == pytest.approx((1.0, 0.5, 1.0))


def test_b40_render_visibility_prunes_render_purpose(tmp_path):
  """UsdGeomVisibilityAPI renderVisibility = invisible hides the render-purpose subtree."""
  stage = new_stage(tmp_path)
  group = UsdGeom.Xform.Define(stage, "/Render")
  group.CreatePurposeAttr(UsdGeom.Tokens.render)
  UsdGeom.VisibilityAPI.Apply(group.GetPrim()).CreateRenderVisibilityAttr(UsdGeom.Tokens.invisible)
  quad(stage, "/Render/Q")
  quad(stage, "/Shown")
  s = rw.formats.usd.read(stage).build()
  assert s.ninstance == 1 and s.report.counts["pruned.render_visibility"] == 1


def test_b40_mesh_light_api_makes_an_emissive_surface(tmp_path):
  """MeshLightAPI: the mesh emits intensity * 2^exposure * color [nits] (normalize: / area)."""
  stage = new_stage(tmp_path)
  mat, _ = preview_material(stage, "/Looks/M", (0.5, 0.5, 0.5))
  mesh = quad(stage, "/Panel", 2.0)
  bind(mesh, mat)
  UsdLux.MeshLightAPI.Apply(mesh.GetPrim())
  light = UsdLux.LightAPI(mesh.GetPrim())
  light.CreateIntensityAttr(10.0)
  light.CreateExposureAttr(1.0)
  light.CreateColorAttr((1.0, 0.5, 0.25))
  builder = rw.formats.usd.read(stage)
  s = builder.build()
  m = builder.materials[int(s.instances.material[0])]
  np.testing.assert_allclose(np.asarray(m.emission_color) * m.emission_luminance, [20.0, 10.0, 5.0], rtol=1e-6)
  assert m.base_color == pytest.approx((0.5, 0.5, 0.5))


@pytest.mark.parametrize("mpu", [1.0, 0.01])
def test_normalize_divides_by_the_area_in_stage_units(tmp_path, mpu):
  """One `normalize` rule for area lights and MeshLightAPI meshes: the world-space area in stage
  units (UsdLux). A 2 m RectLight and a 2 m MeshLightAPI quad of intensity 40 emit 10 nits on a
  metre stage, 40 / 200^2 on a centimetre stage (Omniverse RTX: a normalized 50 cm RectLight on a
  cm stage is 1e4 x dimmer than on a metre stage,
  internal/results/history/post_release/usd_look)."""
  stage = new_stage(tmp_path, mpu=mpu)
  size = 2.0 / mpu
  mesh = quad(stage, "/Panel", size)
  UsdLux.MeshLightAPI.Apply(mesh.GetPrim())
  light = UsdLux.LightAPI(mesh.GetPrim())
  light.CreateIntensityAttr(40.0)
  light.CreateNormalizeAttr(True)
  rect = UsdLux.RectLight.Define(stage, "/Rect")
  rect.CreateWidthAttr(size)
  rect.CreateHeightAttr(size)
  rect.CreateIntensityAttr(40.0)
  rect.CreateNormalizeAttr(True)
  builder = rw.formats.usd.read(stage)
  s = builder.build()
  m = builder.materials[int(s.instances.material[0])]
  expected = 40.0 / size**2
  np.testing.assert_allclose(np.asarray(m.emission_color) * m.emission_luminance, [expected] * 3, rtol=1e-6)
  radiance = derive_lights(s.lights).radiance
  np.testing.assert_allclose(radiance[list(s.lights.paths).index("/Rect")], [expected] * 3, rtol=1e-6)


def test_camera_exposure_is_read_into_the_camera_lens(tmp_path):
  """A camera's exposure (exposure + exposure:fStop / time / iso / responsivity, as stops) is a
  host column (kept by the `.wpbr` file) and the scene camera's `CameraSpec.exposure`."""
  import warp as wp

  stage = new_stage(tmp_path)
  cam = UsdGeom.Camera.Define(stage, "/Cam")
  cam.CreateExposureAttr(2.0)
  plain = UsdGeom.Camera.Define(stage, "/Plain")
  s = rw.formats.usd.read(stage).build()
  assert not s.report.dropped
  path = tmp_path / "scene.wpbr"
  s.save(path)
  loaded = rw.Scene.load(path)
  names = loaded.cameras.names.tolist()
  expected = float(np.log2(cam.ComputeLinearExposureScale()))
  assert loaded.cameras.exposure[names.index("/Cam")] == pytest.approx(expected)
  assert loaded.cameras.exposure[names.index("/Plain")] == pytest.approx(float(np.log2(plain.ComputeLinearExposureScale())))
  with wp.ScopedDevice("cpu"):
    gpu = rw.upload(loaded)
  assert gpu.cameras["/Cam"].exposure == pytest.approx(expected)


@pytest.mark.parametrize("schema", ["DomeLight_1", "DomeLight"])
@pytest.mark.parametrize(("pole", "expected"), [("Y", [0, 1, 0]), ("Z", [0, 0, 1]), ("scene", [0, 0, 1])])
def test_b40_dome_light_1_pole_axis(tmp_path, schema, pole, expected):
  """DomeLight_1 poleAxis: Y / Z, or "scene" (the stage's up axis, Z here); Houdini authors it on
  a DomeLight too (Intel Jungle Ruins, BUG-801)."""
  stage = new_stage(tmp_path)
  write_hdr(tmp_path / "sky.hdr", np.full((4, 8, 3), 0.5))
  dome = stage.DefinePrim("/Dome", schema)
  dome.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set("sky.hdr")
  dome.CreateAttribute("poleAxis", Sdf.ValueTypeNames.Token).Set(pole)
  s = rw.formats.usd.read(stage).build()
  rot = s.environments.rotation[0]
  # The core's environment pole is its local +Z.
  np.testing.assert_allclose(rot @ np.array([0.0, 0.0, 1.0]), expected, atol=1e-6)
  assert not s.report.approximated


@pytest.mark.parametrize(
  ("kit", "rotate_x", "expected"),
  [(True, None, [0, -1, 0]), (True, 270.0, [0, 0, 1]), (False, None, [0, 0, 1]), (False, 270.0, [0, 1, 0])],
)
def test_b60_y_up_dome_pole(tmp_path, kit, rotate_x, expected):
  """A DomeLight's pole in a Y-up stage. A Kit layer's (RTX legacy) is its local +Z: unrotated it
  lies along stage +Z (world -Y); Marbles' 270 degree turn about X stands it upright (world +Z).
  The UsdLux spec's is its local +Y: upright unrotated."""
  stage = new_stage(tmp_path, up="Y")
  if kit:
    stage.GetRootLayer().customLayerData = {"omni_layer": {}}
  write_hdr(tmp_path / "sky.hdr", np.full((4, 8, 3), 0.5))
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.CreateTextureFileAttr(Sdf.AssetPath("sky.hdr"))
  if rotate_x is not None:
    dome.AddRotateXOp().Set(rotate_x)
  s = rw.formats.usd.read(stage).build()
  np.testing.assert_allclose(s.environments.rotation[0] @ np.array([0.0, 0.0, 1.0]), expected, atol=1e-6)


def test_camera_negative_fstop_loads_without_depth_of_field(tmp_path):
  """fStop -1 (the Automotive Material Library's camera): UsdGeom reads fStop <= 0 as no depth of
  field. The stage loads, the camera is a pinhole, the authored value is reported."""
  stage = new_stage(tmp_path)
  UsdGeom.Camera.Define(stage, "/Cam").CreateFStopAttr(-1.0)
  s = rw.formats.usd.read(stage).build()
  assert s.ncamera == 1 and s.cameras.fstop[0] == 0.0
  assert s.report.approximated["Camera.fStop.clamped"] == 1
  assert "/Cam: -1.0" in s.report.examples["Camera.fStop.clamped"]
  assert "Camera.fStop(pinhole)" not in s.report.approximated


def test_camera_invalid_values_default_and_report(tmp_path):
  """Authored nonsense in a camera (zero focal length, a NaN aperture, a negative focus distance,
  far before near) is replaced by UsdGeom's defaults / unbounded far and reported per attribute."""
  stage = new_stage(tmp_path)
  cam = UsdGeom.Camera.Define(stage, "/Cam")
  cam.CreateFocalLengthAttr(0.0)
  cam.CreateHorizontalApertureAttr(math.nan)
  cam.CreateFocusDistanceAttr(-3.0)
  cam.CreateClippingRangeAttr((5.0, 2.0))
  s = rw.formats.usd.read(stage).build()
  c = s.cameras
  assert c.focal_length[0] == pytest.approx(50.0) and c.horizontal_aperture[0] == pytest.approx(20.955)
  assert c.focus_distance[0] == 0.0 and c.near[0] == pytest.approx(5.0) and c.far[0] == 0.0
  for attr in ("focalLength", "horizontalAperture", "focusDistance", "clippingRange.far"):
    assert s.report.approximated[f"Camera.{attr}.clamped"] == 1, attr


def test_camera_with_a_non_finite_transform_is_dropped_and_reported(tmp_path):
  """A camera the builder rejects (a NaN transform) is dropped and reported; the stage loads."""
  stage = new_stage(tmp_path)
  UsdGeom.Camera.Define(stage, "/Bad").AddTranslateOp().Set((math.nan, 0.0, 0.0))
  UsdGeom.Camera.Define(stage, "/Good")
  s = rw.formats.usd.read(stage).build()
  assert s.cameras.names.tolist() == ["/Good"]
  assert s.report.dropped["camera.invalid"] == 1 and s.report.examples["camera.invalid"].startswith("/Bad")


@pytest.mark.parametrize("near", [-1.0, math.nan])
def test_camera_invalid_near_reads_as_the_default(tmp_path, near):
  """A negative or NaN near clipping distance becomes UsdGeom's default (1), reported."""
  stage = new_stage(tmp_path)
  UsdGeom.Camera.Define(stage, "/Cam").CreateClippingRangeAttr((near, 100.0))
  s = rw.formats.usd.read(stage).build()
  assert s.cameras.near[0] == pytest.approx(1.0) and s.cameras.far[0] == pytest.approx(100.0)
  assert s.report.approximated["Camera.clippingRange.near.clamped"] == 1


def test_camera_non_finite_aperture_offset_reads_as_zero(tmp_path):
  stage = new_stage(tmp_path)
  UsdGeom.Camera.Define(stage, "/Cam").CreateVerticalApertureOffsetAttr(math.inf)
  s = rw.formats.usd.read(stage).build()
  assert s.cameras.vertical_offset[0] == 0.0
  assert s.report.approximated["Camera.verticalApertureOffset.clamped"] == 1


def test_camera_invalid_exposure_scale_reads_as_zero_stops(tmp_path):
  """exposure:fStop 0 makes the linear exposure scale infinite: read as 0 stops, reported."""
  stage = new_stage(tmp_path)
  cam = UsdGeom.Camera.Define(stage, "/Cam")
  cam.GetPrim().CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(0.0)
  scale = cam.ComputeLinearExposureScale()
  assert not math.isfinite(scale) or scale <= 0.0
  s = rw.formats.usd.read(stage).build()
  assert s.cameras.exposure[0] == 0.0
  assert s.report.approximated["Camera.exposure.clamped"] == 1


def test_orthographic_camera_focal_length_is_not_validated(tmp_path):
  """An orthographic camera does not use its focal length: 0 loads silently."""
  stage = new_stage(tmp_path)
  cam = UsdGeom.Camera.Define(stage, "/Cam")
  cam.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
  cam.CreateFocalLengthAttr(0.0)
  s = rw.formats.usd.read(stage).build()
  assert s.ncamera == 1 and not s.report.approximated and not s.report.dropped
