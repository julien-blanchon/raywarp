"""The grading fused into every pixel writer (`settings.Grading`; `core/post/grading/transform.py`):
off by default (the plain tone mapper), white balance against CIE illuminants, grading identities,
per-world values without recompiling, and the HDR spatial resolve (samples averaged before the
display transform)."""

import numpy as np
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp.core.post.grading import colorimetry

# A large unlit card filling the view (one color), a small sphere on it.
CARD = """
<mujoco>
  <worldbody>
    <geom name="card" type="box" pos="0 0 0" size="5 5 0.01" rgba="{rgba}"/>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="40"/>
  </worldbody>
</mujoco>
"""
RES = (24, 16)
UNLIT = {"surfaces.shading": "unlit", "post.tone_mapper": "clamp"}


def _card(scene, rgba, **kw):
  """The unlit card scene rendered with rgb and hdr."""
  xml = CARD.format(rgba=" ".join(str(v) for v in rgba))
  return scene(xml, cam_res=RES, outputs=rw.Outputs.of("rgb", "hdr"), **{**UNLIT, **kw}).render()


def _linear(rgb):
  """sRGB display values -> display-linear."""
  rgb = np.asarray(rgb, np.float64)
  return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def test_off_by_default_compiles_nothing(scene):
  """The default settings grade nothing: the plain display transform, no deferred display."""
  s = _card(scene, (0.4, 0.3, 0.2, 1))
  assert not s.r.settings.deferred_display()


def test_white_points_match_cie_illuminants():
  """6504 K is D65 on the daylight locus, 5003 K D50, 2856 K illuminant A (Planckian, Kim et al.
  within 1e-3); the white balance at 6504 K / tint 0 is the identity."""
  xy, _ = colorimetry.white_balance([[6504.0, 0.0], [5003.0, 0.0], [2856.0, 0.0]])
  np.testing.assert_allclose(xy[0], [0.3127, 0.3290], atol=3e-4)
  np.testing.assert_allclose(xy[1], [0.3457, 0.3585], atol=3e-4)
  np.testing.assert_allclose(xy[2], [0.4476, 0.4074], atol=1e-3)
  _, m = colorimetry.white_balance([[6504.0, 0.0]])
  np.testing.assert_allclose(m[0], np.eye(3), atol=1e-5)


def test_tint_moves_towards_green():
  """Positive tint raises the white's y (greener), negative lowers it."""
  xy, _ = colorimetry.white_balance([[5000.0, 0.0], [5000.0, 0.5], [5000.0, -0.5]])
  assert xy[1, 1] > xy[0, 1] > xy[2, 1]


def test_tungsten_grey_card_balances_neutral(scene):
  """A grey card of illuminant A's color renders neutral with the white balance at 2856 K (and
  stays orange without it): the camera's setting, not the light's."""
  xy, _ = colorimetry.white_balance([[2856.0, 0.0]])
  x, y = xy[0]
  xyz = np.array([x / y, 1.0, (1 - x - y) / y])
  rgb = np.linalg.solve(np.asarray(colorimetry.RGB_TO_XYZ, np.float64).reshape(3, 3), xyz)
  card = (*tuple(0.18 * rgb / rgb.max()), 1.0)
  plain = _linear(_card(scene, card).rgb()[0]).mean(axis=(0, 1))
  balanced = _linear(_card(scene, card, **{"grading.enabled": True, "grading.kelvin": 2856.0}).rgb()[0]).mean(axis=(0, 1))
  assert plain[0] > 2 * plain[2]
  assert np.ptp(balanced) / balanced.mean() < 0.03, balanced


def test_identity_grading_keeps_pixels(scene):
  """Grading compiled in at its identity values (and an identity LUT) keeps every pixel."""
  plain = _card(scene, (0.6, 0.3, 0.1, 1)).rgb()
  graded = _card(scene, (0.6, 0.3, 0.1, 1), **{"grading.enabled": True}).rgb()
  np.testing.assert_allclose(graded, plain, atol=1.01 / 255)


def test_saturation_zero_is_grey(scene):
  """Saturation 0 leaves the Rec. 709 luminance in every channel."""
  rgb = _card(scene, (0.6, 0.3, 0.1, 1), **{"grading.enabled": True, "grading.saturation": 0.0}).rgb()[0]
  assert np.abs(rgb - rgb[..., :1]).max() <= 1.01 / 255


def test_per_world_grading_never_recompiles(scene):
  """Per-world white balance / tone: world 0 at identity renders as without grading, world 1
  desaturated; writing the tables keeps the Features."""
  decl = rw.PerWorld(env=["grading.white_balance", "grading.saturation"])
  xml = CARD.format(rgba="0.6 0.3 0.1 1")
  s = scene(xml, nworld=2, cam_res=RES, per_world=decl, outputs=rw.Outputs.of("rgb"), **{**UNLIT, "grading.enabled": True})
  features = s.render().r.features
  with module_loads() as loads:
    s.r.per_world["grading.saturation"].assign(np.array([1.0, 0.0], np.float32))
    rgb = s.render().rgb()
  assert s.r.features is features and not loads, loads
  plain = _card(scene, (0.6, 0.3, 0.1, 1)).rgb()[0]
  np.testing.assert_allclose(rgb[0], plain, atol=1.01 / 255)
  assert np.abs(rgb[1] - rgb[1][..., :1]).max() <= 1.01 / 255
  s.r.per_world["grading.white_balance"].assign(np.array([[6504.0, 0.0], [3000.0, 0.0]], np.float32))
  s.r.per_world["grading.saturation"].assign(np.array([1.0, 1.0], np.float32))
  rgb = s.render().rgb()
  assert rgb[1][..., 2].mean() > rgb[0][..., 2].mean() + 0.05  # balanced for tungsten: bluer


STRIP = """
<mujoco>
  <asset><material name="lamp" rgba="1 1 1 1" emission="{emission}"/></asset>
  <worldbody>
    <geom type="box" pos="0 0 0" size="5 5 0.01" rgba="0 0 0 1"/>
    <geom type="box" pos="0 0 0.02" size="0.03 5 0.01" material="lamp"/>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="40"/>
  </worldbody>
</mujoco>
"""


def test_supersampling_averages_hdr_before_the_tone_mapper(scene):
  """A sub-pixel strip 50x brighter than white: spp 4 averages its HDR before the display
  transform, so a partly covered pixel saturates (the old tone-mapped mean showed its coverage
  fraction instead); the hdr output is the same mean."""
  xml = STRIP.format(emission=50.0)
  kw = {"surfaces.shading": "unlit", "aa.spp": 4, "post.tone_mapper": "clamp"}
  s = scene(xml, cam_res=(64, 8), outputs=rw.Outputs.of("rgb", "hdr"), **kw).render()
  rgb, hdr = s.rgb()[0][4], s.hdr()[0][4] * s.exposure
  # The strip (exposed radiance 50) covers a fraction of each pixel it touches.
  lit = hdr[:, 0] > 0.0
  assert lit.any()
  assert (hdr[lit, 0] < 0.5 * 50.0).all()
  # The mean's display: every touched pixel reaches white, as the tone-mapped mean (coverage < 1/2)
  # would not.
  np.testing.assert_allclose(rgb[lit], 1.0, atol=1.01 / 255)
  np.testing.assert_allclose(rgb[~lit], 0.0, atol=1.01 / 255)


def test_one_planckian_and_daylight_locus():
  """The white balance's device loci are `scene.lighting.photometry`'s (Krystek's Planckian locus,
  the CIE daylight series): one set of coefficients, equal values."""
  import warp as wp

  from raywarp.core.post.grading import colorimetry
  from raywarp.scene.lighting import photometry

  @wp.kernel
  def loci(kelvin: wp.array[float], planck: wp.array[wp.vec2], day: wp.array[wp.vec2]):
    i = wp.tid()
    planck[i] = colorimetry.planckian_xy(kelvin[i])
    day[i] = colorimetry.daylight_xy(kelvin[i])

  temps = np.array([1500.0, 2856.0, 4000.0, 5003.0, 6504.0, 7500.0, 12000.0], np.float32)
  planck = wp.zeros(len(temps), dtype=wp.vec2, device="cpu")
  day = wp.zeros(len(temps), dtype=wp.vec2, device="cpu")
  wp.launch(loci, dim=len(temps), inputs=[wp.array(temps, dtype=float, device="cpu")], outputs=[planck, day], device="cpu")
  np.testing.assert_allclose(planck.numpy(), [photometry.planckian_xy(float(t)) for t in temps], atol=2e-6)
  day_temps = temps >= photometry.DAYLIGHT_RANGE[0]
  want = [photometry.daylight_xy(float(t)) for t in temps[day_temps]]
  np.testing.assert_allclose(day.numpy()[day_temps], want, atol=2e-6)
