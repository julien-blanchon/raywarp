"""UsdLux as the canonical host light model (canonical_design.md §6): the blackbody port, the
record validation, the upload derivation and the conversions (Lighting, emissive panels,
MuJoCo)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import raywarp as rw
from raywarp._base.named import replace_named
from raywarp._base.types import SPOT_INNER_FRACTION, LightShape, LightType
from raywarp.core.pipeline.upload.lights import DerivedLights, derive_environments, derive_lights, light_arrays
from raywarp.scene import EnvironmentLight, Light, LightExt, Shaping
from raywarp.scene.lighting import emissive, rig
from raywarp.scene.lighting.blackbody import blackbody_rgb
from raywarp.scene.lighting.checks import MAX_EXPOSURE, check_light
from raywarp.scene.model.lights import Lights

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given  # after the optional-dependency skip
from hypothesis import strategies as st


def _derive(*lights: Light) -> DerivedLights:
  """The derived table of `lights` built in a scene."""
  b = rw.SceneBuilder()
  for light in lights:
    b.add_light(light)
  return derive_lights(b.build().lights)


# Blackbody ---------------------------------------------------------------------------------------


def test_blackbody_port_is_bit_exact_against_pxr():
  """`scene.lighting.blackbody` equals pxr's UsdLuxBlackbodyTemperatureAsRgb bit for bit (float32)
  on a 1000-40000 K grid and outside the table (below 1000 K, above 10000 K)."""
  usdlux = pytest.importorskip("pxr.UsdLux")
  kelvins = [*np.linspace(1000.0, 40000.0, 3901), 0.0, 500.0, 999.5, 1.0e6]
  for k in kelvins:
    ours = np.array(blackbody_rgb(float(k)), np.float32)
    theirs = np.array(tuple(usdlux.BlackbodyTemperatureAsRgb(float(k))), np.float32)
    assert ours.tobytes() == theirs.tobytes(), (k, ours, theirs)


def test_blackbody_has_unit_luma_and_clamps_to_its_table():
  """A color has the luma of white (Rec.709; below about 1300 K the spline's negative blue is
  clamped) and the table's ends hold outside it."""
  for k in (2000.0, 3000.0, 6500.0, 9000.0):
    r, g, b = blackbody_rgb(k)
    assert 0.2126 * r + 0.7152 * g + 0.0722 * b == pytest.approx(1.0, abs=1e-6)
  assert blackbody_rgb(100.0) == blackbody_rgb(1000.0)
  assert blackbody_rgb(40000.0) == blackbody_rgb(10000.0)


# Derivation -------------------------------------------------------------------------------------


def test_radiance_is_color_times_intensity_times_two_to_the_exposure():
  """Formulas 1-4: L = color x intensity x 2^exposure; a sphere's point intensity is L pi r^2."""
  d = _derive(Light(kind=rw.scene.LightKind.SPHERE, intensity=3.0, exposure=2.0, color=(0.5, 1.0, 0.25), radius=0.1))
  np.testing.assert_array_equal(d.radiance[0], np.asarray([6.0, 12.0, 3.0], np.float32))
  np.testing.assert_array_equal(d.intensity[0], (np.array([6.0, 12.0, 3.0]) * (math.pi * 0.1 * 0.1)).astype(np.float32))


def test_normalize_divides_by_the_world_area_in_stage_units():
  """Formula 3: `normalize` divides by the surface area in stage units (a centimetre stage: the
  area in cm^2); a distant light ignores it."""
  rect = Light(kind=rw.scene.LightKind.RECT, intensity=8.0, width=2.0, height=0.5, normalize=True, meters_per_unit=0.01)
  sun = Light(kind=rw.scene.LightKind.DISTANT, intensity=8.0, normalize=True, angle=0.0)
  d = _derive(rect, sun)
  assert d.radiance[0, 0] == np.asarray(8.0 * (0.01**2 / 1.0), np.float32)
  assert d.radiance[1, 0] == np.asarray(8.0, np.float32)


def test_color_temperature_multiplies_the_blackbody_color():
  """Formula 1: the color is x blackbody(T) only with `enable_color_temperature`."""
  warm = Light(kind=rw.scene.LightKind.DISTANT, angle=0.0, enable_color_temperature=True, color_temperature=3000.0)
  plain = dataclasses.replace(warm, enable_color_temperature=False)
  d = _derive(warm, plain)
  np.testing.assert_array_equal(d.radiance[0], np.asarray(blackbody_rgb(3000.0), np.float32))
  np.testing.assert_array_equal(d.radiance[1], np.ones(3, np.float32))


def test_environment_scale_follows_the_same_formula():
  """Formula 8: an environment's radiance scale is color x blackbody x intensity x 2^exposure."""
  b = rw.SceneBuilder()
  b.add_environment(EnvironmentLight(intensity=2.0, exposure=-1.0, color=(1.0, 0.5, 0.25)))
  b.add_environment(EnvironmentLight(enable_color_temperature=True, color_temperature=5000.0))
  scales = derive_environments(b.build().environments)
  np.testing.assert_array_equal(scales[0], np.asarray([1.0, 0.5, 0.25], np.float32))
  np.testing.assert_array_equal(scales[1], np.asarray(blackbody_rgb(5000.0), np.float32))


def test_angles_cones_and_shaped_flags():
  """Formulas 5-7: degrees -> radians; a cone narrower than the emitter's spread makes a spot
  (SHAPED); a 90 deg cone on a rect, or none, is no cone."""
  spot = Light(kind=rw.scene.LightKind.SPHERE, radius=0.0, shaping=Shaping(cone_angle=30.0, cone_softness=0.2))
  rect = Light(kind=rw.scene.LightKind.RECT, shaping=Shaping(cone_angle=90.0))
  sun = Light(kind=rw.scene.LightKind.DISTANT, angle=2.0)
  d = _derive(spot, rect, sun)
  assert d.cone_angle[0] == np.asarray(math.radians(30.0), np.float32) and d.cone_softness[0] == np.asarray(0.2, np.float32)
  assert d.flags[0] & rw.scene.LightFlags.SHAPED and not d.flags[1] & rw.scene.LightFlags.SHAPED
  assert d.cone_angle[1] == np.asarray(math.pi, np.float32)
  assert d.angle[2] == np.asarray(math.radians(2.0), np.float32)


def test_sizes_a_kind_does_not_have_are_zero():
  """The UsdLux defaults of sizes a kind does not have (a distant light's radius 0.5) do not
  reach the device: only a kind's own sizes (and a distant light's angle) are kept."""
  d = _derive(Light(kind=rw.scene.LightKind.DISTANT, angle=0.0), Light(kind=rw.scene.LightKind.RECT, width=2.0, height=3.0))
  assert (d.radius[0], d.width[0], d.length[0], d.radius[1], d.length[1], d.angle[1]) == (0.0,) * 6
  assert (d.width[1], d.height[1]) == (2.0, 3.0)


def test_light_ext_range_and_no_falloff_reach_light_range():
  """Formula 9: `ext.range` is the falloff window, `falloff="none"` the core's no-falloff 0;
  otherwise no window (inf: the inverse square law at any distance)."""
  base = Light(kind=rw.scene.LightKind.SPHERE, intensity=100.0, radius=0.0)
  d = _derive(base, dataclasses.replace(base, ext=LightExt(range=4.0)), dataclasses.replace(base, ext=LightExt(falloff="none")))
  slots, _ = light_arrays(d)
  np.testing.assert_array_equal(slots["light.range"][0], np.asarray([np.inf, 4.0, 0.0], np.float32))


def test_records_round_trip_through_rayw(tmp_path):
  """The v7 light / environment columns survive a `.rayw` round trip."""
  b = rw.SceneBuilder()
  b.add_light(Light(kind=rw.scene.LightKind.SPHERE, intensity=2.5, exposure=1.5, shaping=Shaping(cone_angle=40.0), path="/L"))
  b.add_light(Light(kind=rw.scene.LightKind.DISTANT, ext=LightExt(range=3.0), body="arm", path="/S"))
  b.add_environment(EnvironmentLight(intensity=0.25, enable_color_temperature=True, color_temperature=4000.0, path="/D"))
  scene = b.build(tmp_path / "s.rayw")
  again = rw.Scene.load(tmp_path / "s.rayw")
  for name, value in scene.lights.arrays().items():
    np.testing.assert_array_equal(again.lights.arrays()[name], value, err_msg=name)
  assert list(again.lights.paths) == ["/L", "/S"] and int(again.lights.body[1]) == 0
  assert again.environments.color_temperature[0] == 4000.0


# Validation --------------------------------------------------------------------------------------

_FIELDS = {
  "intensity": -1.0,
  "diffuse": math.nan,
  "specular": -0.5,
  "exposure": math.inf,
  "color": (1.0, -1.0, 0.0),
  "color_temperature": 0.0,
  "meters_per_unit": 0.0,
  "radius": -0.1,
  "angle": 181.0,
  "texture": 3,
  "instance": 0,
  "shaping": Shaping(cone_angle=200.0),
  "ext": LightExt(range=0.0),
  "xform": np.c_[np.eye(3) * 2.0, np.zeros(3)],
}


@pytest.mark.parametrize("field", sorted(_FIELDS))
def test_check_light_rejects_each_bad_field(field):
  """`check_light` raises naming the field for each out-of-range value (design §6.5)."""
  light = replace_named(Light(kind=rw.scene.LightKind.SPHERE), {field: _FIELDS[field]})
  with pytest.raises(ValueError, match=field.split("_")[0]):
    check_light(light, ntexture=0, ninstance=0, where="test")


def test_a_rect_needs_area():
  """A rect without area is rejected."""
  with pytest.raises(ValueError, match="positive width and height"):
    check_light(Light(kind=rw.scene.LightKind.RECT, width=0.0), ntexture=0, ninstance=0, where="test")


@given(
  intensity=st.floats(0.0, 1e38),
  exposure=st.floats(-MAX_EXPOSURE, MAX_EXPOSURE),
  color=st.tuples(*[st.floats(0.0, 1e38)] * 3),
  kelvin=st.floats(1e-300, 1e300),
  radius=st.floats(0.0, 1e30),
  cone=st.floats(0.0, 180.0),
  softness=st.floats(0.0, 1.0),
)
def test_valid_lights_pass_and_derive_finite(intensity, exposure, color, kelvin, radius, cone, softness):
  """Over the validated ranges, a record either validates and derives to finite, non-negative
  float32 radiance, or is rejected for a radiance scale beyond float32."""
  light = Light(
    kind=rw.scene.LightKind.SPHERE,
    intensity=intensity,
    exposure=exposure,
    color=color,
    enable_color_temperature=True,
    color_temperature=kelvin,
    radius=radius,
    shaping=Shaping(cone_angle=cone, cone_softness=softness),
  )
  try:
    check_light(light, ntexture=0, ninstance=0, where="test")
  except ValueError as e:
    assert "exceeds float32" in str(e)
    return
  d = derive_lights(Lights.from_records([light], bodies=[-1]))
  assert np.all(np.isfinite(d.radiance)) and np.all(d.radiance >= 0.0)


@pytest.mark.parametrize(
  "light",
  [
    Light(kind=rw.scene.LightKind.SPHERE, exposure=1100.0),
    Light(kind=rw.scene.LightKind.SPHERE, intensity=3.0e38, exposure=1.0),
    Light(kind=rw.scene.LightKind.SPHERE, intensity=1.0, color=(1.0e38, 0.0, 0.0), exposure=8.0),
  ],
)
def test_check_light_rejects_underivable_records(light):
  """An exposure beyond MAX_EXPOSURE stops, a radiance scale beyond float32 (it would derive to
  inf / NaN on the device) are rejected."""
  with pytest.raises(ValueError, match=r"exposure|float32"):
    check_light(light, ntexture=0, ninstance=0, where="test")


def test_environment_exposure_is_bounded():
  """An environment's exposure is bounded as a light's."""
  with pytest.raises(ValueError, match="exposure"):
    rw.SceneBuilder().add_environment(EnvironmentLight(exposure=-200.0))


def test_a_180_degree_spot_is_a_point_or_sphere_light():
  """A `Light.spot` of cone angle 180 deg has no cone (UsdLux): a point light, and with a
  bulb radius a sphere area light (LTC)."""
  b = rw.SceneBuilder()
  spots = [
    rw.Light.spot((0.0, 0.0, 3.0), (-0.3, 0.4, -1.0), intensity=10.0, cone_angle=180.0),
    rw.Light.spot((0.0, 0.0, 3.0), (-0.3, 0.4, -1.0), intensity=10.0, cone_angle=180.0, radius=0.1),
  ]
  rig.add_lights(b, spots, center=np.zeros(3), extent=1.0, exposure=1.0)
  d = derive_lights(b.build().lights)
  assert not np.any(d.flags & rw.scene.LightFlags.SHAPED)
  slots, info = light_arrays(d)
  assert slots["light.type"][0].tolist() == [LightType.POINT, LightType.POINT]
  assert slots["light.shape"][0].tolist() == [LightShape.NONE, LightShape.SPHERE] and info.has_area_lights


@given(value=st.floats(allow_nan=True, allow_infinity=True))
def test_cone_angle_is_accepted_exactly_in_its_range(value):
  """A ShapingAPI cone half angle is valid iff it is finite and in [0, 180] deg."""
  light = Light(kind=rw.scene.LightKind.SPHERE, shaping=Shaping(cone_angle=value))
  if math.isfinite(value) and 0.0 <= value <= 180.0:
    check_light(light, ntexture=0, ninstance=0, where="test")
  else:
    with pytest.raises(ValueError, match="cone_angle"):
      check_light(light, ntexture=0, ninstance=0, where="test")


# Conversions -------------------------------------------------------------------------------------


def test_scene_lighting_converts_to_usdlux_exactly():
  """`Light.sun` is a DistantLight of the given illuminance, `Light.point` / `spot` SphereLights of
  radiance intensity / pi r^2 (a point light of `intensity` cd at r = 0): the derived peak is the
  given intensity."""
  lights = [
    rw.Light.sun((-0.3, 0.4, -1.0), illuminance=7.0, color=(1.0, 0.5, 0.5)),
    rw.Light.point((0.0, 0.0, 2.0), intensity=40.0),
    rw.Light.spot((0.0, 0.0, 3.0), (-0.3, 0.4, -1.0), intensity=40.0, cone_angle=30.0, radius=0.1),
  ]
  b = rw.SceneBuilder()
  rig.add_lights(b, lights, center=np.zeros(3), extent=1.0, exposure=1.0)
  slots, info = light_arrays(derive_lights(b.build().lights))
  np.testing.assert_allclose(slots["light.strength"][0], [7.0, 40.0, 40.0], rtol=1e-6)
  assert info.has_spot and slots["light.cutoff"][0, 2] == np.asarray(30.0, np.float32)


def test_emissive_panel_light_radiance_is_the_panel_color():
  """An emissive panel's RECT light has intensity 1, exposure 0: its derived radiance is the
  panel's radiance exactly."""
  b = rw.SceneBuilder()
  glow = b.add_material(rw.Material(base_color=(1.0, 1.0, 1.0), emission_color=(1.0, 0.5, 0.25), emission_luminance=8.0))
  b.add_instance("box", size=(0.5, 0.5, 0.002), xform=np.c_[np.eye(3), [0.0, 0.0, 2.0]], material=glow)
  lit = emissive.add_panel_lights(b.build())
  d = derive_lights(lit.lights)
  assert len(d) == 2 and int(d.kind[0]) == rw.scene.LightKind.RECT
  np.testing.assert_array_equal(d.radiance[0], np.asarray([8.0, 4.0, 2.0], np.float32))


_MJCF = """
<mujoco>
  <worldbody>
    <light name="sun" type="directional" dir="0 0 -1" diffuse="0.8 0.7 0.6" intensity="{i}"/>
    <light name="bulb" pos="0 0 2" diffuse="1 1 1" intensity="{i}" range="5" bulbradius="0.05"/>
    <light name="tiny" pos="1 0 2" diffuse="0.5 0.5 1" intensity="{i}"/>
    <light name="spot" type="spot" pos="0 1 2" dir="0 0 -1" cutoff="30" softness="0.3" intensity="{i}" castshadow="false"/>
    <geom type="plane" size="2 2 0.1"/>
  </worldbody>
</mujoco>
"""


def _usdlux(mjm, i: int, *, classic: float | None) -> Light:
  """MuJoCo light `i` as the UsdLux record of design §6.3 (`classic`: a classic model's
  display-relative intensity, None: photometric): directional -> DistantLight (lux); point ->
  SphereLight of radius `bulbradius` (a point light without one or without falloff), intensity
  cd / pi r^2; spot -> + ShapingAPI (`cutoff`, `softness`; classic: the core's inner 0.75);
  `range` -> `LightExt.range` (0: no falloff)."""
  from raywarp.scene.model import xform

  light_range = float(mjm.light_range[i]) if classic is None else 0.0
  red, green, blue = (float(c) for c in mjm.light_diffuse[i])
  common = Light(
    kind=rw.scene.LightKind.DISTANT, intensity=float(mjm.light_intensity[i]) if classic is None else classic,
    color=(red, green, blue), angle=0.0, shadow_enable=bool(mjm.light_castshadow[i]),
    xform=xform.light_frame(mjm.light_dir[i], mjm.light_pos[i]),
    ext=LightExt(range=light_range) if light_range > 0.0 else LightExt(falloff="none"),
  )  # fmt: skip
  if int(mjm.light_type[i]) == int(LightType.DIRECTIONAL):
    return common
  r = float(mjm.light_bulbradius[i])
  shaping = None
  if int(mjm.light_type[i]) == int(LightType.SPOT):
    softness = float(mjm.light_softness[i]) if classic is None else 1.0 - SPOT_INNER_FRACTION
    shaping = Shaping(cone_angle=float(mjm.light_cutoff[i]), cone_softness=softness)
  return dataclasses.replace(
    common, kind=rw.scene.LightKind.SPHERE, intensity=common.intensity / (math.pi * r * r if r > 0.0 else 1.0), radius=r,
    treat_as_point=r == 0.0 or light_range == 0.0, shaping=shaping,
  )  # fmt: skip


@pytest.mark.parametrize("photometric", [True, False])
def test_mujoco_usdlux_lights_match_light_tables(photometric):
  """The UsdLux records of MuJoCo's lights derive what `light_tables` aliases: range and shape
  exact, color x strength and radius within an ulp (photometric and classic models)."""
  mujoco = pytest.importorskip("mujoco")
  from raywarp.integrations.mujoco import lights as mj_lights

  mjm = mujoco.MjModel.from_xml_string(_MJCF.format(i=100.0 if photometric else 0.0))
  exposure = 0.25
  tables = mj_lights.light_tables(mjm, exposure)
  classic = None if mj_lights.photometric(mjm) else math.pi / exposure
  records = [_usdlux(mjm, i, classic=classic) for i in range(mjm.nlight)]
  slots, _ = light_arrays(derive_lights(Lights.from_records(records, bodies=[-1] * len(records))))
  local = np.asarray(mjm.light_type) != int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)  # distant: no window
  np.testing.assert_array_equal(slots["light.range"][0][local], tables.range[local])
  np.testing.assert_array_equal(slots["light.shape"][0], tables.shape)
  np.testing.assert_allclose(slots["light.radius"][0], tables.radius, rtol=1e-6)
  ours = slots["light.color"][0] * slots["light.strength"][0][:, None]
  np.testing.assert_allclose(ours, np.asarray(mjm.light_diffuse) * tables.intensity[:, None], rtol=1e-6)
  assert [bool(c) for c in slots["light.cast_shadow"][0]] == [bool(c) for c in mjm.light_castshadow]
