"""UsdLux ShapingAPI IES profiles and Omniverse's physical-lighting schemas through the USD
adapter."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
from conftest import DATA, required
from rw_helpers import new_stage, write_hdr

pytest.importorskip("pxr")

from pxr import Sdf, UsdLux

import raywarp as rw
import raywarp.formats.usd
from raywarp.core.pipeline.upload.lights import derive_lights, ies_lights
from raywarp.scene.lighting import ies_profile, photometry

IES = DATA / "ies"


def _sphere(stage, path: str, ies_file: str | None, **attrs: object) -> UsdLux.SphereLight:
  """A treatAsPoint sphere light with ShapingAPI (and its IES inputs)."""
  light = UsdLux.SphereLight.Define(stage, path)
  light.CreateTreatAsPointAttr(True)
  light.CreateRadiusAttr(0.0)
  light.CreateIntensityAttr(3.0)
  shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
  shaping.CreateShapingConeAngleAttr(180.0)
  if ies_file is not None:
    shaping.CreateShapingIesFileAttr(Sdf.AssetPath(ies_file))
  if "scale" in attrs:
    shaping.CreateShapingIesAngleScaleAttr(attrs["scale"])
  if "normalize" in attrs:
    shaping.CreateShapingIesNormalizeAttr(attrs["normalize"])
  return light


def test_ies_inputs_are_read(tmp_path):
  """ies:file (relative to the layer), angleScale and normalize; the file is a dependency; one
  grid for two lights; the "ies dropped" report is gone."""
  shutil.copy(IES / "asym_2002.ies", tmp_path / "asym.ies")
  stage = new_stage(tmp_path)
  _sphere(stage, "/A", "./asym.ies", scale=0.5, normalize=True)
  _sphere(stage, "/B", "./asym.ies")
  stage.Save()
  builder = rw.formats.usd.read(stage)
  assert any(d.endswith("asym.ies") for d in builder.dependencies)
  s = builder.build()
  assert "ShapingAPI.ies" not in s.report.dropped
  assert [bool(t) for t in s.lights.ies.tolist()] == [True, True]
  assert s.lights.ies_angle_scale.tolist() == [0.5, 0.0] and s.lights.ies_normalize.tolist() == [1, 0]
  derived = derive_lights(s.lights)
  table = ies_lights(derived)
  # One grid, two instances (angle scales 0.5 and 0).
  assert table.light_profile.tolist() == [0, 1] and len(table.grids) == 1
  g = table.grids[0]
  assert float(derived.intensity[0, 0]) == pytest.approx(3.0 * g.peak / g.mean, rel=1e-6)
  assert float(derived.intensity[1, 0]) == pytest.approx(3.0, rel=1e-6)


def test_ies_errors_and_distant_lights_are_reported(tmp_path):
  """An unresolved or malformed file, and a profile on a distant light, are reported and the
  light kept without a profile."""
  (tmp_path / "bad.ies").write_text("IESNA:LM-63-2002\nTILT=NONE\n1 2 3\n")
  stage = new_stage(tmp_path)
  _sphere(stage, "/Missing", "./nope.ies")
  _sphere(stage, "/Bad", "./bad.ies")
  sun = UsdLux.DistantLight.Define(stage, "/Sun")
  UsdLux.ShapingAPI.Apply(sun.GetPrim()).CreateShapingIesFileAttr(Sdf.AssetPath(str(IES / "downlight_1995.ies")))
  stage.Save()
  s = rw.formats.usd.read(stage).build()
  assert s.nlight == 3 and not any(s.lights.ies.tolist())
  dropped = s.report.dropped
  assert dropped["ShapingAPI.ies.unresolved"] == 1 and dropped["ShapingAPI.ies.invalid"] == 1
  assert dropped["DistantLight.ies(no angular profile)"] == 1


def _apply(prim, *schemas: str) -> None:
  """Applies codeless API schemas by name (the apiSchemas list op)."""
  prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=list(schemas)))


def test_physical_illuminant_measures_the_profile(tmp_path):
  """PhysicalLightIlluminantAPI without ies:normalize: the file's candela values (intensity 1)."""
  stage = new_stage(tmp_path)
  light = _sphere(stage, "/L", str(IES / "downlight_1995.ies"))
  light.GetIntensityAttr().Set(1.0)
  _apply(light.GetPrim(), "ShapingAPI", "PhysicalLightIlluminantAPI")
  stage.Save()
  s = rw.formats.usd.read(stage).build()
  assert s.lights.ies_mode.tolist() == [1]
  g = ies_profile.profile(s.lights.ies.tolist()[0])
  assert float(derive_lights(s.lights).intensity[0, 0]) == pytest.approx(g.peak, rel=1e-6)


def test_photometric_area_power_illuminance_and_illuminants(tmp_path):
  """PhotometricAreaLightAPI power (default 1600 lm) and illuminance at a distance; the distant
  API's lux; illuminant D / blackbody / custom colours."""
  stage = new_stage(tmp_path)
  rect = UsdLux.RectLight.Define(stage, "/Panel")
  rect.CreateWidthAttr(0.5)
  rect.CreateHeightAttr(0.2)
  _apply(rect.GetPrim(), "PhotometricAreaLightAPI")
  sphere = UsdLux.SphereLight.Define(stage, "/Bulb")
  sphere.CreateRadiusAttr(0.05)
  _apply(sphere.GetPrim(), "PhotometricAreaLightAPI")
  sphere.GetPrim().CreateAttribute("photometric:power", Sdf.ValueTypeNames.Float).Set(0.0)
  sphere.GetPrim().CreateAttribute("photometric:illuminance", Sdf.ValueTypeNames.Float).Set(500.0)
  sphere.GetPrim().CreateAttribute("photometric:illuminance:distance", Sdf.ValueTypeNames.Float).Set(2.0)
  sun = UsdLux.DistantLight.Define(stage, "/Sun")
  _apply(sun.GetPrim(), "PhotometricDistantLightAPI")
  sun.GetPrim().CreateAttribute("physical:illuminant", Sdf.ValueTypeNames.Token).Set("illuminant D")
  sun.CreateColorTemperatureAttr(6504.0)
  warm = UsdLux.DiskLight.Define(stage, "/Warm")
  _apply(warm.GetPrim(), "PhysicalLightIlluminantAPI")
  warm.GetPrim().CreateAttribute("physical:illuminant", Sdf.ValueTypeNames.Token).Set("blackbody")
  warm.CreateColorTemperatureAttr(3000.0)
  custom = UsdLux.DiskLight.Define(stage, "/Custom")
  _apply(custom.GetPrim(), "PhysicalLightIlluminantAPI")
  custom.GetPrim().CreateAttribute("physical:illuminant", Sdf.ValueTypeNames.Token).Set("custom")
  custom.GetPrim().CreateAttribute("physical:customIlluminant", Sdf.ValueTypeNames.Float2Array).Set(
    [(600.0, 1.0), (700.0, 1.0)]
  )
  stage.Save()
  s = rw.formats.usd.read(stage).build()
  assert not [k for k in s.report.dropped if "photometric" in k or "physical" in k]
  by_path = {p: i for i, p in enumerate(s.lights.paths.tolist())}
  record = lambda i: next(r for j, r in enumerate(_records(s)) if j == i)  # noqa: E731
  assert photometry.luminous_flux(record(by_path["/Panel"])) == pytest.approx(1600.0, rel=1e-3)
  derived = derive_lights(s.lights)
  bulb = derived.intensity[by_path["/Bulb"]]
  assert float(np.dot(bulb, photometry.REC709_Y)) / 4.0 == pytest.approx(500.0, rel=1e-5)
  sun_rad = derived.radiance[by_path["/Sun"]]
  assert float(np.dot(sun_rad, photometry.REC709_Y)) == pytest.approx(10000.0, rel=1e-5)
  assert s.lights.enable_color_temperature[by_path["/Warm"]] == 1
  red = np.asarray(s.lights.color[by_path["/Custom"]])
  assert red[0] > 2.0 * red[2]  # a 600-700 nm spectrum is red


def _records(s: rw.Scene) -> list[rw.Light]:
  """The scene's lights as records (the fields `luminous_flux` reads)."""
  t = s.lights
  out = []
  for i in range(len(t)):
    out.append(
      rw.Light(
        kind=rw.scene.LightKind(int(t.kind[i])),
        intensity=float(t.intensity[i]),
        exposure=float(t.exposure[i]),
        color=(float(t.color[i][0]), float(t.color[i][1]), float(t.color[i][2])),
        width=float(t.width[i]),
        height=float(t.height[i]),
        radius=float(t.radius[i]),
        xform=np.asarray(t.xform[i]),
      )
    )
  return out


def test_lumens_light_renders_its_flux():
  """A `Light.spot` given in lumens: its record emits that flux (reflector vs absorber cones)."""
  spec = rw.Light.spot((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), cone_angle=30.0, power=1000.0)
  b = rw.SceneBuilder()
  rw.scene.lighting.rig.add_lights(b, [spec], center=np.zeros(3), extent=1.0, exposure=1.0)
  s = b.build()
  cd = float(derive_lights(s.lights).intensity[0].max())
  assert cd == pytest.approx(1000.0 / (2.0 * math.pi * (1.0 - math.cos(math.radians(30.0)))), rel=2e-3)


def test_photometric_dome_illuminance(tmp_path):
  """PhotometricDomeLightAPI: the dome is scaled so that an upward-facing surface receives
  `photometric:illuminance` lux: a 50 % Lambertian floor under a sky (bright zenith, dark
  ground) shows 0.5 E / pi."""
  stage = new_stage(tmp_path)
  h = 64
  sky = np.ones((h, 2 * h, 3)) * np.linspace(2.0, 0.1, h)[:, None, None]
  write_hdr(tmp_path / "sky.hdr", sky)
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.CreateIntensityAttr(3.0)
  dome.CreateTextureFileAttr(Sdf.AssetPath("./sky.hdr"))
  _apply(dome.GetPrim(), "PhotometricDomeLightAPI")
  dome.GetPrim().CreateAttribute("photometric:illuminance", Sdf.ValueTypeNames.Float).Set(2000.0)
  stage.Save()
  host = rw.formats.usd.read(stage)
  m = host.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), specular_weight=0.0, specular_roughness=1.0))
  host.add_instance("plane", material=m, size=(20.0, 20.0, 0.0))
  s = host.build()
  assert s.environments.illuminance.tolist() == [2000.0]
  settings = rw.RenderSettings().replace(
    **{
      "outputs": rw.Outputs.of("rgb", "hdr"),
      "performance.traversal": "software",
      "environment.background": False,
      "gi.enabled": False,
      "reflections.enabled": False,
    }
  )
  cams = [rw.CameraSpec(fovy=20.0)]
  with rw.Renderer(rw.upload(s, device="cpu"), nworld=1, cameras=cams, resolution=(4, 4), settings=settings) as r:
    r.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    hdr = required(r.render().hdr).numpy()[0, 0]
  np.testing.assert_allclose(hdr, 0.5 * 2000.0 / math.pi, rtol=0.02)
