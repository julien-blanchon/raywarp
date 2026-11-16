"""The user's fill lighting: `lighting.fill` (a runtime, display-relative diffuse term added on
top of everything else) and `rw.Lighting.fill_lights` (display-relative directional fills)."""

import math

import numpy as np
import pytest
from conftest import required
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp.settings.fields import DEFAULT_EV100, exposure_from_ev100

# No light, no scene ambient, no environment: only the user's ambient reaches the camera. One black
# light: a model without any light shows MuJoCo's no-light ambient (`NO_LIGHT_AMBIENT`).
_DARK = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0" diffuse="0 0 0"/></visual>
  <asset>
    <material name="paint" rgba="{r} {g} {b} 1" metallic="{metal}" roughness="1" specular="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0 0 0" specular="0 0 0" castshadow="false"/>
    <geom type="plane" size="5 5 0.1" material="paint"/>
    <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
  </worldbody>
</mujoco>
"""


def _dark(rgb: tuple[float, float, float] = (0.5, 0.5, 0.5), metal: float = 0.0) -> str:
  return _DARK.format(r=rgb[0], g=rgb[1], b=rgb[2], metal=metal)


def test_ambient_is_base_color_times_ambient_display_relative(scene):
  """A white-lit diffuse plane shows base x ambient x color of display white at the exposure:
  hdr radiance = base x ambient x color / exposure; 0 adds nothing; linear in the ambient."""
  s = scene(_dark((0.5, 0.25, 0.125)), outputs=rw.Outputs.of("hdr"), **{"lighting.fill": 0.0})
  assert np.all(s.render().hdr()[0] == 0.0)
  exposure = exposure_from_ev100(DEFAULT_EV100)
  s.r.update_settings(**{"lighting.fill": 0.05, "lighting.fill_color": (1.0, 0.5, 1.0)})
  hdr = s.render().hdr()[0, 16, 16].copy()
  want = np.asarray([0.5, 0.25 * 0.5, 0.125]) * 0.05 / exposure
  np.testing.assert_allclose(hdr, want, rtol=1e-4)
  s.r.update_settings(**{"lighting.fill": 0.1})
  np.testing.assert_allclose(s.render().hdr()[0, 16, 16], 2.0 * hdr, rtol=1e-5)


def test_ambient_change_recompiles_nothing(scene):
  """`lighting.fill` is a runtime field: the Features (the compiled kernels' key) stay."""
  s = scene(_dark(), outputs=rw.Outputs.of("hdr"))
  before = s.render().hdr()
  features = s.r.features
  with module_loads() as loads:
    s.r.update_settings(**{"lighting.fill": 0.08, "lighting.fill_color": (0.2, 0.4, 0.6)})
    after = s.render().hdr()
  assert s.r.features == features and not loads, loads
  assert after.max() > before.max()  # the change took effect


def test_ambient_skips_metals(scene):
  """Diffuse only (base x (1 - metalness)): a metal plane gets nothing from it."""
  s = scene(_dark(metal=1.0), outputs=rw.Outputs.of("hdr"), **{"lighting.fill": 0.1})
  assert np.all(s.render().hdr()[0] == 0.0)


def test_ambient_adds_on_top_of_gi_without_rebake(scene):
  """With GI the ambient is added to the probes' light, not replaced by it, and changing it
  leaves the baked volume and its maps alone (the probes do not see it)."""
  xml = _dark().replace("<worldbody>", '<worldbody><light pos="0 0 3" dir="0 0 -1" castshadow="false"/>')
  gi = rw.settings.GI(enabled=True, spacing=1.0, rays=32, bake_updates=2, max_probes=64)
  s = scene(xml, outputs=rw.Outputs.of("hdr"), gi=gi)
  base = s.render().hdr()[0, 16, 16].copy()
  vol = required(s.r.debug.gi_volume)
  maps = vol.struct.irradiance.numpy().copy() if not vol.released else None
  s.r.update_settings(**{"lighting.fill": 0.1})
  hdr = s.render().hdr()[0, 16, 16]
  assert s.r.debug.gi_volume is vol
  if maps is not None:
    np.testing.assert_array_equal(vol.struct.irradiance.numpy(), maps)
  added = 0.5 * 0.1 / exposure_from_ev100(DEFAULT_EV100)
  np.testing.assert_allclose(hdr - base, added, rtol=1e-3)


# A lit room corner: a white floor and a grey wall under one shadowing light, with a box casting a
# shadow on the floor (shading contrast to keep); `{k}` scales the light (physical units).
_LIT = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0" diffuse="0 0 0"/></visual>
  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" diffuse="{k} {k} {k}" castshadow="true"/>
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="box" pos="0 0 0.3" size="0.3 0.3 0.3" rgba="0.5 0.3 0.2 1"/>
    <camera name="cam" pos="0 -2.5 2" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("k", [1.0, 0.01])
def test_ambient_under_auto_exposure_is_a_small_fill(scene, k):
  """Under automatic exposure (the default) the fill is relative to the metered exposure, in any
  light units (the light 100x dimmer: the same images): ambient 0.05 lifts the image's mean by a
  small bounded amount, keeps the shading contrast (lit floor vs its shadow) and the metered EV
  within a fraction of a stop; the first frame (not metered yet) has no fill. It used to be
  relative to the reference exposure: a scene in physical units became a flat, unshaded image."""
  auto = {"exposure.mode": "auto", "exposure.adaptation": "locked", "outputs": rw.Outputs.of("rgb")}
  plain = scene(_LIT.format(k=k), cameras=["cam"], cam_res=(48, 48), **auto)
  base = plain.render().render().rgb()[0].copy()
  s = scene(_LIT.format(k=k), cameras=["cam"], cam_res=(48, 48), **auto, **{"lighting.fill": 0.05})
  first = s.render().rgb()[0].copy()
  np.testing.assert_array_equal(first, base)  # no fill before the world is metered
  filled = s.render().rgb()[0]
  lift = filled.mean() - base.mean()
  assert 0.005 < lift < 0.08, lift
  lit, shadow = (slice(32, 40), slice(30, 40)), (slice(20, 27), slice(16, 18))
  contrast = lambda img: img[lit].mean() - img[shadow].mean()  # noqa: E731
  assert contrast(base) > 0.3
  assert contrast(filled) > 0.8 * contrast(base)
  assert filled[shadow].mean() > base[shadow].mean() + 0.01  # the shadow is lifted


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("lighting.fill", -0.1),
    ("lighting.fill", 0.2),
    ("lighting.fill", math.inf),
    ("lighting.fill_color", (1.0, -0.5, 1.0)),
  ],
)
def test_ambient_is_validated(field: str, value: object):
  with pytest.raises(ValueError, match=field.replace(".", r"\.")):
    rw.RenderSettings().replace(**{field: value})


@pytest.mark.parametrize("up", ["x", "y", "z"])
def test_fill_lights_surround_the_scene_from_above(up: str):
  """n unshadowed directional lights, unit directions pointing down along the up axis at the
  elevation, evenly spread in azimuth (they cancel horizontally), colors summing to brightness."""
  lights = rw.Lighting.fill_lights(4, brightness=0.4, elevation=30.0, up=up)  # pyrefly: ignore[bad-argument-type]
  axis = "xyz".index(up)
  # A distant light's frame: -Z travels along its direction.
  dirs = np.asarray([-np.asarray(light.xform)[:, 2] for light in lights])
  assert len(lights) == 4 and all(light.kind == rw.scene.LightKind.DISTANT and not light.shadow_enable for light in lights)
  np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
  np.testing.assert_allclose(dirs[:, axis], -0.5, atol=1e-12)  # sin(30 deg) downwards
  np.testing.assert_allclose(dirs.sum(axis=0)[[i for i in range(3) if i != axis]], 0.0, atol=1e-12)
  np.testing.assert_allclose(np.asarray([light.color for light in lights]).sum(axis=0), 0.4, atol=1e-12)


@pytest.mark.parametrize("kw", [{"n": 0}, {"brightness": -1.0}, {"elevation": 0.0}, {"elevation": 95.0}])
def test_fill_lights_are_validated(kw: dict[str, float]):
  with pytest.raises(ValueError, match="fill_lights"):
    rw.Lighting.fill_lights(**kw)  # pyrefly: ignore[bad-argument-type]


def test_unknown_up_axis_is_named():
  """An up axis other than x / y / z fails with a message naming it (not str.index's)."""
  with pytest.raises(ValueError, match="up must be one of"):
    rw.Lighting.studio(up="w")  # pyrefly: ignore[bad-argument-type]
  with pytest.raises(ValueError, match="up must be one of"):
    rw.Lighting.fill_lights(up="-z")  # pyrefly: ignore[bad-argument-type]


def test_fill_lights_light_a_scene_without_lights(scene):
  """Added through `Lighting`, they light an unlit scene's plane by about brightness x
  sin(elevation) of display white (diffuse, display-relative)."""
  lighting = rw.Lighting(lights=rw.Lighting.fill_lights(3, brightness=0.6, elevation=90.0))
  # MuJoCo lights live in the model: added to the spec first (`rw.integrations.mujoco.add_lights`).
  import mujoco

  import raywarp.integrations.mujoco  # noqa: F401

  spec = mujoco.MjSpec.from_string(_dark((1.0, 1.0, 1.0)))
  rw.integrations.mujoco.add_lights(spec, lighting)
  s = scene(spec.to_xml(), outputs=rw.Outputs.of("hdr"))
  exposure = exposure_from_ev100(DEFAULT_EV100)
  white = s.render().hdr()[0, 16, 16] * exposure
  np.testing.assert_allclose(white, 0.6, rtol=0.05)
