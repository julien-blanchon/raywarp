"""The procedural sun-sky (`scene/lighting/sky.py`): the Hosek-Wilkie sky and its sun disc baked to
an equirectangular radiance map, cut into a directional light at upload."""

import math

import numpy as np
import pytest

from raywarp.scene.images.colorspace import REC709_Y
from raywarp.scene.images.latlong import extract_sun, latlong_directions, texel_solid_angles
from raywarp.scene.lighting.sky import SUN_RADIUS, _disc_coverage, sun_direction, sun_radiance, sun_sky

_Y = np.asarray(REC709_Y)


def _upper_mask(h: int) -> np.ndarray:
  return latlong_directions(h, 2 * h)[..., 2] > 0.0


def test_sun_is_cut_into_a_directional_light_at_its_direction_and_power():
  """The upload's sun extraction finds the disc where it was drawn, with the disc's power (the
  ring fill subtracted: within a few %), and the disc's own size as the map resolves it."""
  h = 256
  image = sun_sky(40.0, 75.0, turbidity=3.0, height=h)
  cut = extract_sun(image)
  assert cut is not None
  _, sun = cut
  assert float(sun.direction @ sun_direction(40.0, 75.0)) > math.cos(math.radians(0.5))
  power = 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS)) * sun_radiance(40.0, 3.0)
  np.testing.assert_allclose(sun.irradiance, power, rtol=0.05)


def test_disc_power_is_drawn_exactly():
  """The map's excess over the sky near the sun integrates to the disc's radiance x solid angle."""
  h = 512
  with_sun = sun_sky(30.0, 10.0, height=h).astype(np.float64)
  omega = texel_solid_angles(h, 2 * h)
  d = latlong_directions(h, 2 * h) @ sun_direction(30.0, 10.0)
  near = d > math.cos(SUN_RADIUS + 4.0 * math.pi / h)
  sky = np.median(with_sun[near & (d < math.cos(SUN_RADIUS + 2.0 * math.pi / h))], axis=0)
  excess = ((with_sun[near] - sky) * omega[near][:, None]).sum(axis=0)
  power = 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS)) * sun_radiance(30.0, 3.0)
  np.testing.assert_allclose(excess, power, rtol=0.02)


def test_sky_is_physically_plausible():
  """A clear noon sky: blue zenith of a few thousand nits, brighter towards the sun, a ground of
  albedo x horizontal illuminance / pi, a direct sun of about 100 klux; a hazy sky's horizon is
  whiter than a clear one's."""
  h = 128
  clear = sun_sky(60.0, 0.0, turbidity=2.0, ground_albedo=0.2, height=h).astype(np.float64)
  up = _upper_mask(h)
  zenith = clear[0].mean(axis=0)
  assert zenith[2] > zenith[0] and 1.0e3 < float(zenith @ _Y) < 2.0e4
  # Along the sun's elevation row, brighter towards the sun (the circumsolar aureole).
  row = int(30.0 / 180.0 * h)
  d = latlong_directions(h, 2 * h)[row] @ sun_direction(60.0, 0.0)
  lum = clear[row] @ _Y
  assert float(lum[(d > 0.98) & (d < 0.9995)].mean()) > 1.5 * float(lum[d < 0.6].mean())
  sun_e = float(sun_radiance(60.0, 2.0) @ _Y) * 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS))
  assert 8.0e4 < sun_e < 1.3e5
  ground = clear[~up]
  np.testing.assert_allclose(ground, ground[:1].repeat(len(ground), axis=0))
  hazy = sun_sky(60.0, 0.0, turbidity=8.0, height=h).astype(np.float64)
  low = h // 2 - 2  # just above the horizon
  ratio = lambda im: im[low, :, 2].mean() / im[low, :, 0].mean()  # noqa: E731
  assert ratio(hazy) < ratio(clear)


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"elevation": -1.0}, "elevation"),
    ({"elevation": 30.0, "turbidity": 11.0}, "turbidity"),
    ({"elevation": 30.0, "ground_albedo": 1.5}, "ground_albedo"),
    ({"elevation": 30.0, "height": 8}, "height"),
  ],
)
def test_out_of_range_arguments_raise(kwargs, match):
  with pytest.raises(ValueError, match=match):
    sun_sky(**kwargs)


@pytest.mark.parametrize(("elevation", "height"), [(90.0, 256), (90.0, 512), (45.0, 64), (0.0, 128)])
def test_disc_covers_its_solid_angle_exactly_at_any_elevation_and_size(elevation, height):
  """The disc's texel coverage holds its solid angle exactly (near the zenith too), the part above
  the horizon only."""
  dirs = latlong_directions(height, 2 * height)
  window, coverage = _disc_coverage(dirs, sun_direction(elevation, 20.0), height)
  covered = float((coverage * texel_solid_angles(height, 2 * height)[window]).sum())
  disc = 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS))
  if elevation == 0.0:
    assert 0.3 * disc < covered < 0.7 * disc
  else:
    assert covered == pytest.approx(disc, rel=1e-9)


@pytest.mark.parametrize("height", [32, 48])
def test_small_maps_hold_the_disc_between_their_sub_samples(height):
  """At the smallest heights the 0.27 deg disc can fall between every sub-sample of its texels:
  the texel nearest the sun then holds its whole solid angle (no division by zero, the disc's
  power kept) for every sun position."""
  dirs = latlong_directions(height, 2 * height)
  omega = texel_solid_angles(height, 2 * height)
  disc = 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS))
  for elevation in np.linspace(5.0, 85.0, 9):
    for azimuth in np.linspace(0.0, 340.0, 14):
      sun = sun_direction(float(elevation), float(azimuth))
      window, coverage = _disc_coverage(dirs, sun, height)
      assert float((coverage * omega[window]).sum()) == pytest.approx(disc, rel=1e-9)
      assert np.isfinite(sun_sky(float(elevation), float(azimuth), height=height)).all()
