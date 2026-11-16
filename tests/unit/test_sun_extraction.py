"""The sun rule of `latlong.extract_sun` on synthetic skies: a sun disc is cut out (direction,
size, irradiance), an overcast sky and studio softboxes are not, at most one source per map, and
the map's light is conserved (the remainder plus the sun)."""

import math

import numpy as np
import pytest

from raywarp.scene.images import latlong
from raywarp.scene.images.colorspace import REC709_Y

H, W = 256, 512
# A sun 50 degrees up, 30 degrees round from +x; its irradiance at normal incidence [lux-like].
SUN = np.array(
  [
    math.cos(math.radians(50)) * math.cos(math.radians(30)),
    math.cos(math.radians(50)) * math.sin(math.radians(30)),
    math.sin(math.radians(50)),
  ]
)
SUN_E = np.array([60.0, 55.0, 45.0])


def _sky(h: int = H, w: int = W) -> np.ndarray:
  """A clear sky (brighter towards the zenith, blue) over a grey ground: radiance (h, w, 3)."""
  z = latlong.latlong_directions(h, w)[..., 2]
  sky = (0.6 + 0.4 * z)[..., None] * np.array([0.55, 0.75, 1.0])
  return np.where((z > 0.0)[..., None], sky, 0.25).astype(np.float32)


def _add_disc(image: np.ndarray, direction: np.ndarray, radius: float, irradiance: np.ndarray) -> None:
  """Adds a uniform disc of angular `radius` [rad] (at least the texel under `direction`) holding
  `irradiance` (3,) at normal incidence to `image` (h, w, 3), in place."""
  h, w = image.shape[:2]
  dirs = latlong.latlong_directions(h, w)
  cos = dirs @ direction
  disc = cos > math.cos(radius)
  disc[np.unravel_index(np.argmax(cos), cos.shape)] = True
  omega = latlong.texel_solid_angles(h, w)
  image[disc] += (irradiance / omega[disc].sum()).astype(np.float32)


def _power(image: np.ndarray) -> np.ndarray:
  """(3,) the map's radiance x solid angle summed over the sphere."""
  return (image.astype(np.float64) * latlong.texel_solid_angles(*image.shape[:2])[..., None]).sum((0, 1))


def _irradiance(image: np.ndarray, normal: np.ndarray) -> float:
  """Luminance irradiance of a surface of `normal` under the map."""
  h, w = image.shape[:2]
  cos = np.maximum(latlong.latlong_directions(h, w) @ normal, 0.0)
  return float(((image.astype(np.float64) @ np.asarray(REC709_Y)) * latlong.texel_solid_angles(h, w) * cos).sum())


def test_a_sun_disc_is_cut_out():
  """A 0.27 degree sun (the real one) over a clear sky: one sun towards it (within half a texel),
  a disc the size of the texel it covers, its irradiance; the remainder has no peak left."""
  sky = _sky()
  _add_disc(sky, SUN, math.radians(0.27), SUN_E)
  cut = latlong.extract_sun(sky)
  assert cut is not None
  rest, sun = cut
  texel = math.pi / H
  assert math.acos(min(1.0, float(sun.direction @ SUN))) < 0.5 * texel
  assert 0.2 * texel < sun.angle < texel
  np.testing.assert_allclose(sun.irradiance, SUN_E, rtol=0.02)
  assert sun.share > 0.5
  assert rest.max() < 2.0


@pytest.mark.parametrize("degrees", [1.4, 2.0, 3.0])
def test_a_sun_at_the_pole_has_its_size(degrees):
  """A resolved sun (2-4 texels round) straight overhead has the size it has at mid elevation:
  the row-0 texels are thin wedges around the pole, not discs of their tiny solid angle (which
  gave 0.38 degrees for a 1 degree sun)."""
  sizes = []
  for direction in (SUN, np.array([0.0, 0.0, 1.0])):
    sky = _sky()
    _add_disc(sky, direction, math.radians(degrees), SUN_E)
    cut = latlong.extract_sun(sky)
    assert cut is not None
    sizes.append(cut[1].angle)
  np.testing.assert_allclose(sizes[1], sizes[0], rtol=0.1)
  np.testing.assert_allclose(sizes[1], math.radians(degrees), rtol=0.15)


def test_the_light_is_conserved():
  """The map's power is the remainder's plus the sun's irradiance, per channel; an upward and a
  sun-facing surface get the same irradiance (the sun as a directional light: E cos)."""
  sky = _sky()
  _add_disc(sky, SUN, math.radians(0.27), SUN_E)
  cut = latlong.extract_sun(sky)
  assert cut is not None
  rest, sun = cut
  np.testing.assert_allclose(_power(rest) + sun.irradiance, _power(sky), rtol=1e-9)
  y = float(sun.irradiance @ np.asarray(REC709_Y))
  for normal in (np.array([0.0, 0.0, 1.0]), SUN, np.array([-SUN[0], -SUN[1], SUN[2]])):
    split = _irradiance(rest, normal) + y * max(float(sun.direction @ normal), 0.0)
    np.testing.assert_allclose(split, _irradiance(sky, normal), rtol=2e-3)


def test_an_overcast_sky_has_no_sun():
  """A bright, smooth overcast sky (the zenith 3 x the horizon): no texel stands out, no sun."""
  z = latlong.latlong_directions(H, W)[..., 2]
  overcast = np.repeat((np.where(z > 0.0, 1.0 + 2.0 * z, 0.3))[..., None], 3, axis=-1).astype(np.float32)
  assert latlong.extract_sun(overcast) is None


def test_studio_softboxes_have_no_sun():
  """Two large softboxes (12 degrees round, 300 x the room), side by side up front: extended
  sources, not a sun (nothing cut out, so no softbox is lit as a hard-shadowed point)."""
  room = np.full((H, W, 3), 0.5, np.float32)
  dirs = latlong.latlong_directions(H, W)
  for azimuth in (-40.0, 40.0):
    a = math.radians(azimuth)
    centre = np.array([math.cos(a) * math.cos(0.5), math.sin(a) * math.cos(0.5), math.sin(0.5)])
    room[dirs @ centre > math.cos(math.radians(12.0))] = 150.0
  assert latlong.extract_sun(room) is None


def test_two_compact_lamps_give_at_most_one_sun():
  """Two equal compact lamps 90 degrees apart: one is cut out (the map's peak), the other stays
  in the map; the light is conserved."""
  room = np.full((H, W, 3), 0.2, np.float32)
  a, b = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0), np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0)
  _add_disc(room, a, math.radians(1.0), np.full(3, 20.0))
  _add_disc(room, b, math.radians(1.0), np.full(3, 20.0))
  cut = latlong.extract_sun(room)
  assert cut is not None
  rest, sun = cut
  assert max(float(sun.direction @ a), float(sun.direction @ b)) > 0.999
  assert rest.max() > 100.0  # the other lamp
  np.testing.assert_allclose(_power(rest) + sun.irradiance, _power(room), rtol=1e-9)


def test_a_dim_compact_lamp_is_no_sun():
  """A compact lamp holding under SUN_MIN_SHARE of the light of a surface facing it: no sun (its
  shadow would be faint, its ray not worth it)."""
  room = np.full((H, W, 3), 1.0, np.float32)
  _add_disc(room, SUN, math.radians(1.0), np.full(3, 0.5))  # pi of sky irradiance vs 0.5
  assert latlong.extract_sun(room) is None


@pytest.mark.parametrize("width", [64, 128])
def test_a_low_resolution_sun_is_found(width):
  """A sun in a coarse map spreads over a texel of 2.8-5.6 degrees: still compact (the bound adds
  one texel), cut out towards it; a map coarser than SUN_MIN_ROWS has none."""
  h = width // 2
  sky = _sky(h, width)
  _add_disc(sky, SUN, math.radians(0.27), SUN_E)
  cut = latlong.extract_sun(sky)
  assert cut is not None
  assert math.acos(min(1.0, float(cut[1].direction @ SUN))) < math.pi / h
  coarse = _sky(8, 16)
  _add_disc(coarse, SUN, math.radians(0.27), SUN_E)
  assert latlong.extract_sun(coarse) is None


def test_the_cut_is_cached_by_the_maps_content(monkeypatch, tmp_path):
  """`cut_suns` stores the cut maps, suns and report by the maps' content: the same maps load
  them without searching again (identical arrays and report), another map searches."""
  import warp as wp

  from raywarp.core.pipeline.upload import environment
  from raywarp.scene.model.report import LoadReport

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  sunny = _sky()
  _add_disc(sunny, SUN, math.radians(0.27), SUN_E)
  maps, lit, paths, cpu = [sunny, _sky()], [True, True], ["sunny", "clear"], wp.get_device("cpu")
  first = LoadReport()
  out, suns = environment.cut_suns(maps, lit, paths, first, device=cpu)

  def boom(*_args: object) -> None:
    raise AssertionError("searched again")

  with monkeypatch.context() as m:
    m.setattr(latlong, "extract_sun", boom)
    again = LoadReport()
    out2, suns2 = environment.cut_suns(maps, lit, paths, again, device=cpu)
  for a, b in zip(
    [*out, suns.direction, suns.irradiance, suns.angle, suns.env],
    [*out2, suns2.direction, suns2.irradiance, suns2.angle, suns2.env],
    strict=True,
  ):
    np.testing.assert_array_equal(a, b)
  assert again.to_dict() == first.to_dict()
  assert len(suns) == 1
  with monkeypatch.context() as m:
    m.setattr(latlong, "extract_sun", boom)
    with pytest.raises(AssertionError, match="searched again"):
      environment.cut_suns([sunny * 2.0], [True], ["brighter"], LoadReport(), device=cpu)
