"""USD adapter lights (UsdLux, physical units) and cameras."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from rw_helpers import new_stage, write_hdr

pytest.importorskip("pxr")

from pxr import Gf, Sdf, UsdGeom, UsdLux

import raywarp as rw
import raywarp.formats.usd
from raywarp.core.pipeline.upload.lights import derive_environments, derive_lights
from raywarp.scene.images import colorspace

K = rw.scene.LightKind


def lights(stage, **options) -> rw.Scene:
  return rw.formats.usd.read(stage, adapter=rw.formats.usd.Options(**options)).build()


def test_light_kinds_sizes_and_units(tmp_path):
  stage = new_stage(tmp_path, mpu=0.01)  # centimetres: sizes scale, radiance does not
  sphere = UsdLux.SphereLight.Define(stage, "/Sphere")
  sphere.CreateRadiusAttr(10.0)
  sphere.CreateIntensityAttr(100.0)
  sphere.CreateExposureAttr(1.0)
  rect = UsdLux.RectLight.Define(stage, "/Rect")
  rect.CreateWidthAttr(200.0)
  rect.CreateHeightAttr(50.0)
  rect.CreateNormalizeAttr(True)
  rect.CreateIntensityAttr(10.0)
  UsdLux.DiskLight.Define(stage, "/Disk").CreateRadiusAttr(100.0)
  cyl = UsdLux.CylinderLight.Define(stage, "/Cyl")
  cyl.CreateLengthAttr(100.0)
  cyl.CreateTreatAsLineAttr(True)
  sun = UsdLux.DistantLight.Define(stage, "/Sun")
  sun.CreateIntensityAttr(3000.0)
  sun.CreateAngleAttr(1.0)
  sun.CreateNormalizeAttr(True)  # intensity: lux
  UsdGeom.Xformable(sun).AddRotateXOp().Set(90.0)
  s = lights(stage)
  li = derive_lights(s.lights)
  by = {li.paths[i]: i for i in range(s.nlight)}
  i = by["/Sphere"]
  assert li.kind[i] == K.SPHERE and li.radius[i] == pytest.approx(0.1)
  np.testing.assert_allclose(li.radiance[i], 200.0)  # 100 * 2^1
  np.testing.assert_allclose(li.intensity[i], 200.0 * math.pi * 0.01, rtol=1e-6)
  i = by["/Rect"]
  assert (li.width[i], li.height[i]) == pytest.approx((2.0, 0.5))
  # normalize: / the area in stage units (cm^2, as RTX), so the intensity is 10 x (0.01 m)^2
  np.testing.assert_allclose(li.intensity[i], 10.0 * 0.01**2, rtol=1e-6)
  assert li.radius[by["/Disk"]] == pytest.approx(1.0)
  assert li.flags[by["/Cyl"]] & rw.scene.LightFlags.TREAT_AS_LINE and li.length[by["/Cyl"]] == pytest.approx(1.0)
  i = by["/Sun"]
  assert li.kind[i] == K.DISTANT and li.angle[i] == pytest.approx(math.radians(1.0))
  np.testing.assert_allclose(li.radiance[i], 3000.0)
  np.testing.assert_allclose(li.xform[i] @ np.array([0.0, 0.0, -1.0, 0.0]), [0, 1, 0], atol=1e-6)  # emits along local -Z


def test_primary_ray_visibility_follows_omniverse(tmp_path):
  """`visibleInPrimaryRay` (Omniverse RTX, un-namespaced) flags a light seen by camera rays; unset,
  an area light is unseen (RTX's default), and the attribute is consumed, not reported dropped."""
  stage = new_stage(tmp_path)
  UsdLux.SphereLight.Define(stage, "/Plain").CreateRadiusAttr(0.2)
  shown = UsdLux.RectLight.Define(stage, "/Shown").GetPrim()
  shown.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
  s = lights(stage)
  li = derive_lights(s.lights)
  flags = {li.paths[i]: int(li.flags[i]) & rw.scene.LightFlags.VISIBLE for i in range(s.nlight)}
  assert flags == {"/Plain": 0, "/Shown": rw.scene.LightFlags.VISIBLE}
  assert not any("visibleInPrimaryRay" in key for key in s.report.dropped)


def test_color_temperature_and_shaping(tmp_path):
  stage = new_stage(tmp_path)
  spot = UsdLux.SphereLight.Define(stage, "/Spot")
  spot.CreateRadiusAttr(0.0)
  spot.CreateEnableColorTemperatureAttr(True)
  spot.CreateColorTemperatureAttr(3000.0)
  shaping = UsdLux.ShapingAPI.Apply(spot.GetPrim())
  shaping.CreateShapingConeAngleAttr(30.0)
  shaping.CreateShapingConeSoftnessAttr(0.2)
  wide = UsdLux.RectLight.Define(stage, "/Wide")
  UsdLux.ShapingAPI.Apply(wide.GetPrim()).CreateShapingConeAngleAttr(90.0)  # no restriction for a rect
  s = lights(stage)
  li = derive_lights(s.lights)
  rgb = li.radiance[0]
  assert rgb[0] > rgb[1] > rgb[2]  # warm
  assert li.cone_angle[0] == pytest.approx(math.radians(30.0)) and li.cone_softness[0] == pytest.approx(0.2)
  assert li.flags[0] & rw.scene.LightFlags.SHAPED and not li.flags[1] & rw.scene.LightFlags.SHAPED


@pytest.mark.parametrize("version", [None, 2411], ids=["usdlux", "rtx-legacy"])
def test_dome_light_environment(tmp_path, version):
  stage = new_stage(tmp_path)
  write_hdr(tmp_path / "sky.hdr", np.full((4, 8, 3), 0.5))
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.CreateTextureFileAttr(Sdf.AssetPath("sky.hdr"))
  dome.CreateIntensityAttr(1000.0)
  if version is not None:
    dome.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(version)
  UsdGeom.Xformable(dome).AddRotateZOp().Set(90.0)
  s = lights(stage)
  assert s.nlight == 0 and s.nenv == 1
  env = s.environments
  np.testing.assert_allclose(derive_environments(env)[0], 1000.0)
  rot = env.rotation[0]
  if version is None:
    # UsdLux (OpenEXR's latlong): pole local +Y (-> -x with the dome's 90 degrees), centre
    # column local +Z, u = 0.25 local +X (-> +y).
    np.testing.assert_allclose(rot @ np.array([0.0, 0.0, 1.0]), [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(rot @ np.array([0.0, 1.0, 0.0]), [0, 1, 0], atol=1e-6)
  else:
    # Omniverse's latlong is the core's turned by a half turn: the map's -x (its centre column)
    # turns to +y with the dome's 90 degrees; no mirror (B10).
    np.testing.assert_allclose(rot @ np.array([-1.0, 0.0, 0.0]), [0, 1, 0], atol=1e-6)
  assert np.linalg.det(rot) == pytest.approx(1.0)
  assert colorspace.from_code(int(s.textures.colorspace[env.texture[0]])) == rw.scene.ColorSpace.LIN_REC709
  np.testing.assert_allclose(s.images.level(int(s.textures.image[env.texture[0]]))[..., :3], 0.5, rtol=1e-2)


def test_dome_inputs_an_environment_cannot_hold_are_reported(tmp_path):
  """A DomeLight's diffuse / specular scales, shadow switch and portals are reported, not
  silently dropped (the environment lights both lobes of every surface); its read inputs are not."""
  stage = new_stage(tmp_path)
  write_hdr(tmp_path / "sky.hdr", np.full((4, 8, 3), 0.5))
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.CreateTextureFileAttr(Sdf.AssetPath("sky.hdr"))
  dome.CreateIntensityAttr(2.0)
  dome.CreateSpecularAttr(0.0)
  dome.CreateDiffuseAttr(0.5)
  dome.GetPrim().CreateAttribute("inputs:shadow:enable", Sdf.ValueTypeNames.Bool).Set(False)
  dome.CreatePortalsRel().SetTargets(["/Portal"])
  dropped = rw.formats.usd.read(stage).build().report.dropped
  for name in ("inputs:specular", "inputs:diffuse", "inputs:shadow:enable", "portals"):
    assert dropped[f"DomeLight.{name}"] == 1, (name, dict(dropped))
  assert not any(k.startswith("DomeLight.inputs:intensity") or "texture" in k for k in dropped)


def test_pre_2102_shaping_names_shape_the_light(tmp_path):
  """A cone authored with the pre-21.02 input names (`shaping:cone:angle`, no `inputs:` prefix and
  no ShapingAPI applied) shapes the light like the current names."""
  stage = new_stage(tmp_path)
  light = UsdLux.SphereLight.Define(stage, "/Spot")
  light.GetPrim().CreateAttribute("shaping:cone:angle", Sdf.ValueTypeNames.Float).Set(30.0)
  s = lights(stage)
  assert s.lights.cone_angle[0] == pytest.approx(30.0)


def test_dome_texture_follows_its_authored_color_space(tmp_path):
  """A dome texture's authored colorSpace decides its decoding (unauthored: by file type)."""
  stage = new_stage(tmp_path)
  write_hdr(tmp_path / "sky.hdr", np.full((4, 8, 3), 0.5))
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.CreateTextureFileAttr(Sdf.AssetPath("sky.hdr")).SetColorSpace("srgb_rec709_scene")
  s = lights(stage)
  env = s.environments
  assert colorspace.from_code(int(s.textures.colorspace[env.texture[0]])) == rw.scene.ColorSpace.SRGB_REC709


@pytest.mark.parametrize(
  ("layer_data", "names", "expected", "ambiguous"),
  [
    ({}, ("inputs:intensity",), 1000.0 * math.pi * math.sin(math.radians(0.53) / 2.0) ** 2, False),
    ({"omni_layer": {}}, ("inputs:intensity",), 1000.0, False),
    ({"renderSettings": {}, "cameraSettings": {}}, ("inputs:intensity",), 1000.0, False),
    ({}, ("intensity", "inputs:intensity"), 1000.0, True),  # Kit's dual naming, no Kit metadata
    ({}, ("intensity",), 1000.0 * math.pi * math.sin(math.radians(0.53) / 2.0) ** 2, True),  # pre-21.02 only
  ],
  ids=["usdlux", "kit-omni-layer", "kit-settings", "kit-naming-only", "old-naming-only"],
)
def test_distant_light_dialect_is_detected(tmp_path, layer_data, names, expected, ambiguous):
  """Without an authored `omni:rtx:usdluxVersion`, a distant light follows UsdLux unless its layer
  was saved by Kit (RTX legacy: lux); attribute naming alone is ambiguous: reported and warned."""
  stage = new_stage(tmp_path)
  stage.GetRootLayer().customLayerData = layer_data
  sun = stage.DefinePrim("/Sun", "DistantLight")
  for name in names:
    sun.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(1000.0)
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s = lights(stage)
  assert s.lights.intensity[0] == pytest.approx(expected, rel=1e-5)
  flagged = [k for k in s.report.approximated if k.startswith("UsdLux.dialect.ambiguous")]
  assert bool(flagged) == ambiguous and bool([w for w in caught if "ambiguous" in str(w.message)]) == ambiguous


def test_a_referenced_kit_layer_keeps_its_dialect(tmp_path):
  """A Kit asset referenced into a UsdLux stage: its lights keep the Kit layer's dialect. An
  override authored in the UsdLux stage, or a light of a non-Kit sublayer of a Kit stage, has
  mixed evidence: RTX legacy, reported as ambiguous."""
  kit = new_stage(tmp_path, "kit.usda")
  kit.GetRootLayer().customLayerData = {"omni_layer": {}}
  kit.SetDefaultPrim(kit.DefinePrim("/Asset", "Xform"))
  UsdLux.DistantLight.Define(kit, "/Asset/Sun").CreateIntensityAttr(1000.0)
  UsdLux.DistantLight.Define(kit, "/Asset/Over").CreateIntensityAttr(1000.0)
  kit.Save()
  stage = new_stage(tmp_path)
  stage.DefinePrim("/World", "Xform").GetReferences().AddReference("kit.usda")
  UsdLux.DistantLight(stage.GetPrimAtPath("/World/Over")).CreateIntensityAttr(2000.0)
  with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    s = lights(stage)
  by_path = dict(zip(s.lights.paths.tolist(), s.lights.intensity.tolist(), strict=True))
  assert by_path["/World/Sun"] == pytest.approx(1000.0)
  assert by_path["/World/Over"] == pytest.approx(2000.0)
  assert [k for k in s.report.approximated if k.startswith("UsdLux.dialect.ambiguous")]


def test_a_kit_stage_with_lights_in_a_plain_sublayer_is_ambiguous(tmp_path):
  """A Kit stage (root customLayerData) whose light is defined in a sublayer without Kit metadata
  renders in RTX legacy: read so, reported and warned (never silently as the spec)."""
  sub = new_stage(tmp_path, "lights.usda")
  UsdLux.DistantLight.Define(sub, "/Sun").CreateIntensityAttr(1000.0)
  sub.Save()
  stage = new_stage(tmp_path)
  stage.GetRootLayer().customLayerData = {"renderSettings": {}, "cameraSettings": {}}
  stage.GetRootLayer().subLayerPaths.append("lights.usda")
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s = lights(stage)
  assert s.lights.intensity[0] == pytest.approx(1000.0)
  assert [k for k in s.report.approximated if k.startswith("UsdLux.dialect.ambiguous")]
  assert [w for w in caught if "ambiguous" in str(w.message)]


@pytest.mark.parametrize("value", [(Sdf.ValueTypeNames.String, "2505"), (Sdf.ValueTypeNames.Float, float("nan"))])
def test_an_invalid_rtx_version_is_reported_not_fatal(tmp_path, value):
  """A non-numeric or NaN `omni:rtx:usdluxVersion` is reported and ignored (detection decides)."""
  stage = new_stage(tmp_path)
  sun = UsdLux.DistantLight.Define(stage, "/Sun")
  sun.CreateIntensityAttr(1000.0)
  sun.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", value[0]).Set(value[1])
  s = lights(stage)
  # Ignored: no Kit layer, so the spec's unnormalized radiance (the `omni:` attribute itself is
  # Kit naming: ambiguous, RTX legacy, lux).
  assert s.lights.intensity[0] == pytest.approx(1000.0)
  assert s.report.approximated["UsdLux.usdluxVersion.invalid(ignored)"] == 1


def test_lux_dialect_option_reads_unmarked_lights(tmp_path):
  """`usd.Options(lux_dialect=RTX_LEGACY)`: a stage built live in Kit (no layer marks) reads as
  RTX legacy; an authored version still wins."""
  stage = new_stage(tmp_path)
  UsdLux.DistantLight.Define(stage, "/Sun").CreateIntensityAttr(1000.0)
  marked = UsdLux.DistantLight.Define(stage, "/Marked")
  marked.CreateIntensityAttr(1000.0)
  marked.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
  s = rw.formats.usd.read(stage, adapter=rw.formats.usd.Options(lux_dialect=rw.formats.usd.Dialect.RTX_LEGACY)).build()
  by_path = dict(zip(s.lights.paths.tolist(), s.lights.intensity.tolist(), strict=True))
  assert by_path["/Sun"] == pytest.approx(1000.0)
  assert by_path["/Marked"] == pytest.approx(1000.0 * math.pi * math.sin(math.radians(0.53) / 2.0) ** 2, rel=1e-5)


@pytest.mark.parametrize("invisible_lights", [False, True])
def test_lights_inside_instances_under_an_invisible_prim(tmp_path, invisible_lights):
  """A light inside an instanceable asset under an invisible prim is found (instance proxies):
  kept with `invisible_lights`, else reported off; never silently lost."""
  asset = new_stage(tmp_path, "lamp.usda")
  asset.SetDefaultPrim(asset.DefinePrim("/Lamp", "Xform"))
  UsdLux.SphereLight.Define(asset, "/Lamp/Bulb").CreateIntensityAttr(10.0)
  asset.Save()
  stage = new_stage(tmp_path)
  hidden = UsdGeom.Xform.Define(stage, "/Hidden")
  hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
  lamp = stage.DefinePrim("/Hidden/Lamp", "Xform")
  lamp.GetReferences().AddReference("lamp.usda")
  lamp.SetInstanceable(True)
  s = lights(stage, invisible_lights=invisible_lights)
  assert s.nlight == int(invisible_lights)
  assert s.report.dropped["UsdLux.invisible_light(off)"] == int(not invisible_lights)


def test_invisible_lights_are_off_unless_requested(tmp_path):
  stage = new_stage(tmp_path)
  group = UsdGeom.Xform.Define(stage, "/Off")
  group.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
  UsdLux.SphereLight.Define(stage, "/Off/L")
  UsdLux.SphereLight.Define(stage, "/On")
  s = lights(stage)
  assert s.nlight == 1 and s.report.dropped["UsdLux.invisible_light(off)"] == 1
  assert lights(stage, invisible_lights=True).nlight == 2


def test_light_inside_prototype_is_instanced(tmp_path):
  stage = new_stage(tmp_path)
  stage.CreateClassPrim("/Lamp")
  UsdLux.SphereLight.Define(stage, "/Lamp/Bulb")
  for i in range(3):
    prim = UsdGeom.Xform.Define(stage, f"/L{i}")
    prim.GetPrim().GetReferences().AddInternalReference("/Lamp")
    prim.GetPrim().SetInstanceable(True)
    prim.AddTranslateOp().Set(Gf.Vec3d(i, 0, 0))
  s = lights(stage)
  np.testing.assert_allclose(sorted(s.lights.xform[:, 0, 3]), [0, 1, 2])


def test_cone_softness_beyond_one_is_clamped(tmp_path):
  """Content authors cone softness past UsdLux's [0, 1] (the Office's spots: 80): the stage
  loads, the softness is clamped to 1 and the clamp is reported (it used to fail at add_light)."""
  stage = new_stage(tmp_path)
  spot = UsdLux.SphereLight.Define(stage, "/Spot")
  shaping = UsdLux.ShapingAPI.Apply(spot.GetPrim())
  shaping.CreateShapingConeAngleAttr(40.0)
  shaping.CreateShapingConeSoftnessAttr(80.0)
  s = lights(stage)
  assert s.lights.cone_softness[0] == pytest.approx(1.0)
  assert s.report.approximated["ShapingAPI.coneSoftness.clamped"] == 1


def test_distant_light_angle_beyond_180_is_clamped(tmp_path):
  """SimReady thumbnail rigs author a DistantLight `angle` of 260 deg (past UsdLux's angular
  diameter range): the stage loads, the angle is clamped to 180 deg and the clamp is reported;
  `check_light` still rejects such a record built by hand."""
  stage = new_stage(tmp_path)
  UsdLux.DistantLight.Define(stage, "/Sun").CreateAngleAttr(260.0)
  s = lights(stage)
  assert s.lights.angle[0] == 180.0
  assert s.report.approximated["DistantLight.angle.clamped"] == 1
  with pytest.raises(ValueError, match="angle"):
    rw.SceneBuilder().add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, angle=260.0))


def test_nan_cone_and_softness_are_no_cone(tmp_path):
  """A NaN ShapingAPI cone angle / softness loads as no cone / no softness (reported), not an
  abort."""
  stage = new_stage(tmp_path)
  shaping = UsdLux.ShapingAPI.Apply(UsdLux.SphereLight.Define(stage, "/Spot").GetPrim())
  shaping.CreateShapingConeAngleAttr(math.nan)
  shaping.CreateShapingConeSoftnessAttr(math.nan)
  s = lights(stage)
  assert s.lights.cone_angle[0] == 180.0 and s.lights.cone_softness[0] == 0.0
  assert not derive_lights(s.lights).flags[0] & rw.scene.LightFlags.SHAPED
  assert s.report.approximated["ShapingAPI.coneAngle.clamped"] == 1
  assert s.report.approximated["ShapingAPI.coneSoftness.clamped"] == 1


def test_an_invalid_light_is_dropped_and_the_stage_loads(tmp_path):
  """One bad authored light (a negative intensity, a NaN color, a negative focus, a negative
  rect size) is dropped and reported; the stage's other lights load."""
  stage = new_stage(tmp_path)
  UsdLux.SphereLight.Define(stage, "/Bad").CreateIntensityAttr(-5.0)
  UsdLux.DistantLight.Define(stage, "/Nan").CreateColorAttr(Gf.Vec3f(math.nan, 1.0, 1.0))
  UsdLux.ShapingAPI.Apply(UsdLux.SphereLight.Define(stage, "/Focus").GetPrim()).CreateShapingFocusAttr(-1.0)
  UsdLux.RectLight.Define(stage, "/Rect").CreateWidthAttr(-1.0)
  UsdLux.SphereLight.Define(stage, "/Good")
  s = lights(stage)
  assert list(s.lights.paths) == ["/Good"]
  assert s.report.dropped["light.invalid"] == 4


def test_infinite_far_clip_is_unbounded(tmp_path):
  """A camera clipping range with an infinite far plane loads as far 0 (unbounded): the builder
  rejects non-finite values."""
  stage = new_stage(tmp_path)
  UsdGeom.Camera.Define(stage, "/Cam").CreateClippingRangeAttr(Gf.Vec2f(0.1, math.inf))
  assert lights(stage).camera("/Cam").far == 0.0


def test_camera(tmp_path):
  stage = new_stage(tmp_path, mpu=0.01)
  cam = UsdGeom.Camera.Define(stage, "/Cam")
  cam.CreateFocalLengthAttr(18.0)
  cam.CreateVerticalApertureAttr(15.0)
  cam.CreateClippingRangeAttr(Gf.Vec2f(10.0, 1000.0))
  cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, 200))
  s = lights(stage)
  c = s.camera("/Cam")
  assert c.fovy == pytest.approx(math.degrees(2 * math.atan(7.5 / 18.0)))
  assert (c.near, c.far) == pytest.approx((0.1, 10.0))
  np.testing.assert_allclose(c.xform[:, 3], [0, 0, 2])
  np.testing.assert_allclose(c.xform[:, :3], np.eye(3), atol=1e-6)  # scale removed


@pytest.mark.parametrize("name", ["inputs:texture:file", "texture:file"])
def test_an_unresolved_dome_texture_is_reported_under_either_input_name(tmp_path, name):
  """A DomeLight whose texture file does not resolve is reported, whether it uses the current
  input name or the pre-21.02 one (both are read)."""
  stage = new_stage(tmp_path)
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("missing.hdr"))
  assert rw.formats.usd.read(stage).build().report.dropped["DomeLight.texture:file.unresolved"] == 1


def test_declared_inputs_without_values_are_not_reported(tmp_path):
  """Kit declares every input of an applied schema without a value (McUsd's DomeLight with
  ShapingAPI): nothing is authored, nothing is reported; an authored value still is."""
  stage = new_stage(tmp_path)
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  UsdLux.ShapingAPI.Apply(dome.GetPrim())
  dome.GetPrim().CreateAttribute("inputs:shaping:focus", Sdf.ValueTypeNames.Float)
  dome.GetPrim().CreateAttribute("inputs:shaping:cone:angle", Sdf.ValueTypeNames.Float).Set(30.0)
  dropped = rw.formats.usd.read(stage).report.dropped
  assert "DomeLight.inputs:shaping:focus" not in dropped
  assert dropped["DomeLight.inputs:shaping:cone:angle"] == 1
