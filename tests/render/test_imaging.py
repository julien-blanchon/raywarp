"""The deferred display (`core/post/display/deferred.py`): automatic exposure (metering modes,
percentile rejection, highlight protection and its continuity, compensation, the locked /
continuous / per-frame adaptation, resets, skipped worlds, determinism) and chromatic
aberration."""

import numpy as np
import pytest
import warp as wp
from conftest import recipe

import raywarp as rw
from raywarp.core.post.exposure import auto as exposure

# An unlit card (one color) with an optional small emissive square at `spot`.
CARD = """
<mujoco>
  <asset><material name="lamp" rgba="1 1 1 1" emission="{emission}"/></asset>
  <worldbody>
    <geom name="card" type="box" pos="0 0 0" size="5 5 0.01" rgba="{rgba}"/>
    <geom name="lamp" type="box" pos="{spot} 0.02" size="{half} {half} 0.01" material="lamp"/>
    <camera name="cam" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="40"/>
    <camera name="side" pos="0.3 0 2" xyaxes="1 0 0 0 1 0" fovy="40"/>
  </worldbody>
</mujoco>
"""
RES = (32, 32)
BASE = {"surfaces.shading": "unlit", "post.tone_mapper": "clamp"}


def _scene(scene, *, rgba=(0.5, 0.5, 0.5, 1), emission=0.0, spot=(0.0, 0.0), half=0.0001, nworld=1, cameras=None, **kw):
  """The card scene and its renderer."""
  xml = CARD.format(rgba=" ".join(map(str, rgba)), emission=emission, spot=f"{spot[0]} {spot[1]}", half=half)
  kw.setdefault("cam_res", RES)
  return scene(xml, nworld=nworld, cameras=cameras or ["cam"], outputs=rw.Outputs.of("rgb", "hdr"), **{**BASE, **kw})


def _linear(rgb):
  """sRGB display values -> display-linear."""
  rgb = np.asarray(rgb, np.float64)
  return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


# Automatic exposure: `exposure.mode` "auto", lit shading (an unlit render shows its base colors as
# they are), stateless per frame unless a test sets another adaptation.
AUTO = {"exposure.mode": "auto", "surfaces.shading": "pbr", "exposure.adaptation": "per_frame"}
# The default key: the metered mean's exposed luminance.
KEY = 0.18


def test_off_allocates_nothing(scene):
  """Without any deferred stage the renderer has no deferred display and writes no hdr."""
  s = _scene(scene)
  assert s.r._imaging is None
  assert not s.r.settings.deferred_display()


def test_default_is_automatic():
  """The default settings meter every image (`exposure.mode` "auto"); a manual EV, an unlit
  render or a render without a shaded output does not."""
  s = rw.RenderSettings()
  assert s.exposure.mode == "auto" and s.auto_exposure() and s.deferred_display()
  assert not s.replace(**{"exposure.mode": "manual", "exposure.ev100": 12.0}).auto_exposure()
  assert not s.replace(**{"surfaces.shading": "unlit"}).auto_exposure()
  assert not rw.RenderSettings(outputs=rw.Outputs.of("depth", "seg")).auto_exposure()
  for name in ("fast", "default", "quality", "rtx_look", "reference"):
    assert recipe(name).auto_exposure(), name


@pytest.mark.parametrize("grey", [0.1, 0.8])
def test_auto_exposure_meters_middle(scene, grey):
  """Whatever the card's albedo, a uniform card is exposed to the key (middle grey 0.18),
  within the histogram's bin (1/8 EV); compensation +1 doubles it."""
  for comp, want in ((0.0, KEY), (1.0, 2 * KEY)):
    s = _scene(scene, rgba=(grey, grey, grey, 1), **AUTO, **{"exposure.compensation": comp}).render()
    got = _linear(s.rgb()[0]).mean()
    assert abs(np.log2(got / want)) < 0.07, (grey, comp, got)


def test_histogram_rejects_a_hot_spot(scene, monkeypatch):
  """A small very bright square (9 % of the frame): the histogram band (bright 10 % ignored)
  keeps the card's exposure (highlight protection off: it would darken it,
  `test_highlight_protection`)."""
  monkeypatch.setattr(exposure, "_HIGHLIGHT_STOPS", 0.0)
  plain = _linear(_scene(scene, **AUTO).render().rgb()[0])
  hot = {"emission": 1.0e4, "spot": (0.0, 0.0), "half": 0.22}
  hist = _linear(_scene(scene, **hot, **AUTO).render().rgb()[0])
  corner = (slice(0, 4), slice(0, 4))
  np.testing.assert_allclose(hist[corner], plain[corner], rtol=0.03)


def _ev(s):
  """The metered EV of world 0, camera 0 after a render."""
  return float(s.r._imaging.auto.ev.numpy()[0, 0])


def _exposed(s, y, x):
  """The exposed luminance of pixel (y, x) of world 0, camera 0: its hdr radiance x 2^-EV."""
  rgb = s.hdr()[0, y, x]
  return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) * 2.0 ** -_ev(s)


def test_highlight_protection(scene, monkeypatch):
  """A very bright square on a dark card (13 stops brighter: it clips at the band mean's
  exposure): a small square (2 % of the frame) is allowed to clip; a large one (a third) darkens
  the exposure by `_HIGHLIGHT_STOPS` (1 stop), never more; off (0 stops) the card shows at the
  key; with a larger cap the darkening is the square's overshoot, which shows it at `_WHITE`."""
  hot = {"rgba": (0.02, 0.02, 0.02, 1), "emission": 50.0}
  c = RES[0] // 2

  def square(half, stops=exposure._HIGHLIGHT_STOPS):
    monkeypatch.setattr(exposure, "_HIGHLIGHT_STOPS", stops)
    return _scene(scene, **hot, half=half, **AUTO).render()

  off = square(0.42, stops=0.0)
  on = square(0.42)
  free = square(0.42, stops=16.0)
  assert _exposed(off, c, c) > 2.0 * exposure._WHITE  # clipped without protection
  assert abs(_ev(on) - _ev(off) - 1.0) < 1e-3  # capped at 1 stop
  assert _ev(free) > _ev(on) + 0.5  # the overshoot is more than a stop
  assert abs(np.log2(_exposed(free, c, c) / exposure._WHITE)) < 0.05  # the square at its white
  small_off, small_on = square(0.08, stops=0.0), square(0.08)
  assert abs(_ev(small_on) - _ev(small_off)) < 1e-3  # 2 % of the frame: no protection


def test_highlight_protection_is_continuous(scene):
  """A window entering the view (a bright square growing in small steps from nothing to half the
  frame) moves the stateless exposure in small steps: no jump where the percentile of a two-level
  histogram used to cross the gap (the bedroom tour flicker)."""
  hot = {"rgba": (0.05, 0.05, 0.05, 1), "emission": 20.0}
  evs = []
  for half in np.linspace(0.0, 0.5, 26):
    s = _scene(scene, **hot, half=float(max(half, 1e-4)), **AUTO, cam_res=(96, 96)).render()
    evs.append(_ev(s))
  steps = np.abs(np.diff(evs))
  # 25 steps (the area grows quadratically, the band mean takes the window in from 10 % of the
  # frame): the change spreads over the steps; a percentile meter put stops into one step.
  assert evs[-1] > evs[0] + 2.0
  assert steps.max() < 0.2 * (evs[-1] - evs[0]), [round(e, 3) for e in evs]


def test_spot_metering_reads_the_centre(scene):
  """Spot metering of a bright centre square exposes the square to the key; average metering
  exposes the mostly dark frame brighter."""
  hot = {"rgba": (0.05, 0.05, 0.05, 1), "emission": 2.0, "half": 0.15}
  spot = _linear(_scene(scene, **hot, **AUTO, **{"exposure.metering": "spot"}).render().rgb()[0])
  avg = _linear(_scene(scene, **hot, **AUTO).render().rgb()[0])
  c = RES[0] // 2
  assert abs(np.log2(spot[c, c].mean() / KEY)) < 0.1
  assert avg[c, c].mean() > 2 * spot[c, c].mean()


def _brighten(s, factor):
  """Scales every instance color of the per-world color table by `factor` (a scene change)."""
  colors = s.r.per_world["instance.color"]
  rows = colors.numpy()
  rows[..., :3] *= factor
  colors.assign(rows)


def test_adaptation_reset_and_skip(scene, monkeypatch):
  """Continuous adaptation: a brighter scene moves the EV by (1 - exp(-dt speed_up)) of the step
  per frame; a reset takes the metered EV at once; skipped worlds keep their state."""
  ref = _scene(scene, rgba=(0.8, 0.8, 0.8, 1), **AUTO)
  ref.render()
  target = _ev(ref)  # the bright card's stateless metering
  decl = rw.PerWorld(material=["color"])
  dt, up = 0.1, 2.0
  monkeypatch.setattr(exposure, "_SPEED_UP", up)
  kw = {**AUTO, "exposure.adaptation": "continuous", "exposure.frame_time": dt}
  s = _scene(scene, nworld=2, per_world=decl, rgba=(0.2, 0.2, 0.2, 1), **kw)
  s.render()
  ev0 = s.r._imaging.auto.ev.numpy().copy()
  _brighten(s, 4.0)  # the card at 0.8
  s.render()
  ev1 = s.r._imaging.auto.ev.numpy().copy()  # a copy: on the CPU numpy() is a view of the state
  rate = 1.0 - np.exp(-dt * up)
  assert target > ev0[0, 0] + 1.0
  np.testing.assert_allclose(ev1 - ev0, (target - ev0) * rate, atol=0.01)
  first = s.rgb().copy()
  s.r.reset_history(rw_array([True, True], s))
  s.r.render(s.d, active=wp.array([1], dtype=wp.int32, device=s.r.device))
  ev2 = s.r._imaging.auto.ev.numpy()
  np.testing.assert_allclose(ev2[0], ev1[0])  # skipped: kept
  np.testing.assert_allclose(ev2[1], target, atol=0.01)  # reset: the metered EV at once
  np.testing.assert_array_equal(s.rgb()[0], first[0])


def test_continuous_steps_are_smooth(scene):
  """Continuous adaptation over a scene that alternates dark / bright every 10 frames (30 fps):
  every per-frame EV step is a fraction (1 - exp(-dt speed)) of the remaining distance, at most
  `_SPEED_UP` x dt x the scene change, and never overshoots the target."""
  decl = rw.PerWorld(material=["color"])
  kw = {**AUTO, "exposure.adaptation": "continuous"}
  s = _scene(scene, per_world=decl, rgba=(0.2, 0.2, 0.2, 1), **kw)
  s.render()
  dark = _ev(s)
  evs = [dark]
  for frame in range(40):
    if frame % 10 == 0:
      _brighten(s, 4.0 if (frame // 10) % 2 == 0 else 0.25)
    s.render()
    evs.append(_ev(s))
  steps = np.diff(evs)
  dt = 1.0 / 30.0
  # The scene changes by 2 stops; the fastest rate moves 1 - exp(-dt 3) ~ 9.5 % of it per frame.
  assert np.abs(steps).max() <= 2.0 * (1.0 - np.exp(-dt * exposure._SPEED_UP)) + 1e-3
  assert min(evs) >= dark - 1e-3 and max(evs) <= dark + 2.0 + 1e-3  # no overshoot
  # Within each 10-frame segment the steps keep one sign (monotone approach).
  for k in range(4):
    seg = steps[10 * k : 10 * k + 10]
    assert (seg >= -1e-5).all() or (seg <= 1e-5).all(), seg.tolist()


def test_locked_exposure_meters_once(scene):
  """The default ("locked"): the first frame is metered and shown at its own EV (the same image
  as a per-frame meter of that frame), a scene change keeps the EV, a reset world meters its next
  frame and the other keeps its lock."""
  assert rw.RenderSettings().exposure.adaptation == "locked"
  decl = rw.PerWorld(material=["color"])
  kw = {**AUTO, "exposure.adaptation": "locked"}
  s = _scene(scene, nworld=2, per_world=decl, rgba=(0.2, 0.2, 0.2, 1), **kw)
  first = s.render().rgb().copy()
  stateless = _scene(scene, rgba=(0.2, 0.2, 0.2, 1), **AUTO).render().rgb()
  np.testing.assert_array_equal(first[0], stateless[0])  # the first frame at its own EV
  ev0 = s.r._imaging.auto.ev.numpy().copy()
  _brighten(s, 4.0)
  s.render()
  np.testing.assert_array_equal(s.r._imaging.auto.ev.numpy(), ev0)  # locked
  ref = _scene(scene, rgba=(0.8, 0.8, 0.8, 1), **AUTO)
  ref.render()
  s.r.reset_history(rw_array([False, True], s))
  s.render()
  ev = s.r._imaging.auto.ev.numpy()
  np.testing.assert_allclose(ev[0], ev0[0])
  np.testing.assert_allclose(ev[1, 0], _ev(ref), atol=0.01)


def rw_array(values, s):
  """A bool device array of the renderer's device."""
  import warp as wp

  return wp.array(np.asarray(values, bool), dtype=bool, device=s.r.device)


def test_metering_is_deterministic(scene):
  """Without adaptation the same frame always gives the same bytes."""
  s = _scene(scene, emission=50.0, half=0.1, spot=(0.3, 0.2), **AUTO)
  a = s.render().rgb().copy()
  b = s.render().rgb()
  np.testing.assert_array_equal(a, b)


def test_ev_bins():
  """EV100 = log2(L x 8): L = 1 nit sits in the bin of EV 3; non-positive luminance in bin 0."""
  import warp as wp

  @wp.kernel
  def bins(lum: wp.array[float], out: wp.array[int]):
    i = wp.tid()
    out[i] = exposure.ev_bin(lum[i])

  lum = wp.array([1.0, 0.0, -1.0, 1.0e30], dtype=float)
  out = wp.zeros(4, dtype=int)
  wp.launch(bins, dim=4, inputs=[lum], outputs=[out])
  b = out.numpy()
  assert b[0] == int((3.0 - exposure.EV_LO) * exposure.BINS / (exposure.EV_HI - exposure.EV_LO))
  assert b[1] == 0 and b[2] == 0 and b[3] == exposure.BINS - 1


def test_histogram_mean_of_a_lone_pixel():
  """One metered pixel of weight 1 (the band holds 0.8 of it): the mean is its bin's EV."""
  import warp as wp

  @wp.kernel
  def mean(m: exposure.Metering, histogram: wp.array2d[wp.int32], out: wp.array[wp.vec2]):
    out[0] = exposure.histogram_ev(m, histogram, 0)

  m = exposure.Metering()
  m.low, m.high = 0.1, 0.9
  rows = np.zeros((1, exposure.BINS), np.int32)
  rows[0, 100] = 1
  out = wp.zeros(1, dtype=wp.vec2)
  wp.launch(mean, dim=1, inputs=[m, wp.array(rows, dtype=wp.int32)], outputs=[out])
  ev, total = out.numpy()[0]
  width = (exposure.EV_HI - exposure.EV_LO) / exposure.BINS
  assert total == 1.0
  np.testing.assert_allclose(ev, exposure.EV_LO + 100.5 * width, rtol=1e-6)


def test_aberration_fringes_outward(scene):
  """Lateral chromatic aberration: on the outer edge of an off-centre white square the red image
  (magnified) extends past the blue one; at strength 0 the channels coincide."""
  hot = {"rgba": (0, 0, 0, 1), "emission": 1.0, "spot": (0.45, 0.0), "half": 0.15}
  s = _scene(scene, **hot, **{"sensor.aberration": 0.08}).render().rgb()[0]
  row = s[RES[1] // 2]
  red, blue = np.flatnonzero(row[:, 0] > 0.5), np.flatnonzero(row[:, 2] > 0.5)
  assert red.max() > blue.max()
  same = _scene(scene, **hot).render().rgb()[0]
  np.testing.assert_array_equal(same[..., 0], same[..., 2])


@pytest.mark.parametrize("recipe_name", ["fast", "quality", "rtx_look", "reference"])
def test_recipes_auto_exposure_and_grading(scene, recipe_name):
  """Every lit recipe meters a uniform card to the key and grades it (saturation 0: grey), whatever
  its sampling (spp 2, more shadow samples)."""
  target = recipe(recipe_name)
  over = {k: v for k, v in rw.RenderSettings().diff(target).items() if not k.startswith(("gi.", "outputs"))}
  kw = {**over, **AUTO, "grading.enabled": True, "grading.saturation": 0.0, "post.tone_mapper": "clamp"}
  s = _scene(scene, rgba=(0.6, 0.3, 0.1, 1), **kw)
  for _ in range(2):
    s.render()
  rgb = _linear(s.rgb()[0])
  assert abs(np.log2(rgb.mean() / KEY)) < 0.1, rgb.mean()
  assert np.abs(rgb - rgb[..., :1]).max() < 0.01


def test_black_background_is_not_metered(scene):
  """Misses without an environment (zero luminance) carry no exposure information: a card filling
  a quarter of a black frame is exposed as if it filled the frame."""
  full = _linear(_scene(scene, **AUTO).render().rgb()[0])
  xml = CARD.replace('size="5 5 0.01"', 'size="0.3 0.3 0.01"')
  s = scene(
    xml.format(rgba="0.5 0.5 0.5 1", emission=0.0, spot="0 0", half=0.0001),
    cam_res=RES,
    cameras=["cam"],
    outputs=rw.Outputs.of("rgb", "hdr"),
    **{**BASE, **AUTO, "environment.enabled": False, "environment.background": False},
  )
  part = _linear(s.render().rgb()[0])
  c = RES[0] // 2
  np.testing.assert_allclose(part[c, c], full[c, c], rtol=0.05)


def test_internal_hdr_is_float16_square_root(scene):
  """The deferred display's own HDR image (no "hdr" output requested) is float16 of
  sqrt(radiance): half of float32's bytes, no user view, and the display within one 8-bit level of
  the float32 path (a card with a bright square: 1e4 x its luminance)."""
  hot = {"rgba": (0.3, 0.2, 0.1, 1), "emission": 1.0e4, "half": 0.2, **AUTO}
  full = _scene(scene, **hot).render()
  xml = CARD.format(rgba="0.3 0.2 0.1 1", emission=1.0e4, spot="0.0 0.0", half=0.2)
  compact = scene(xml, nworld=1, cam_res=RES, cameras=["cam"], outputs=rw.Outputs.of("rgb"), **{**BASE, **AUTO})
  compact.render()
  assert compact.r.settings.written_outputs().hdr == rw.settings.FloatOutput(dtype="float16", sqrt=True)
  assert compact.r.outputs.hdr is None
  a = np.asarray(full.rgb()[0], np.int32)
  b = np.asarray(compact.rgb()[0], np.int32)
  assert np.abs(a - b).max() <= 1


@pytest.mark.parametrize("value", [1.0e-6, 1.0, 1.0e9])
def test_sqrt16_round_trips(value):
  """sqrt16 keeps 1e-6 .. 1e9 within 2^-10 relative (two float16 roundings of the root)."""
  import warp as wp

  from raywarp.core.kernels.display import sqrt16

  @wp.kernel
  def k(v: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    r = wp.vec3(sqrt16(v[0]))
    out[0] = wp.cw_mul(r, r)

  v = wp.array([wp.vec3(value)], dtype=wp.vec3, device="cpu")
  out = wp.zeros(1, dtype=wp.vec3, device="cpu")
  wp.launch(k, dim=1, inputs=[v], outputs=[out], device="cpu")
  np.testing.assert_allclose(out.numpy()[0], value, rtol=2.0**-10)


def test_internal_hdr_square_root_through_the_motion_body(scene):
  """The motion body's store writes the internal sqrt16 image too: its display matches the
  float32 path within one level."""
  kw = {**BASE, **AUTO}
  xml = CARD.format(rgba="0.3 0.2 0.1 1", emission=1.0e4, spot="0.0 0.0", half=0.2)
  full = scene(xml, nworld=1, cam_res=RES, cameras=["cam"], outputs=rw.Outputs.of("rgb", "hdr", "motion"), **kw)
  compact = scene(xml, nworld=1, cam_res=RES, cameras=["cam"], outputs=rw.Outputs.of("rgb", "motion"), **kw)
  for _ in range(2):
    full.render()
    compact.render()
  assert compact.r.settings.written_outputs().hdr == rw.settings.FloatOutput(dtype="float16", sqrt=True)
  assert np.abs(np.asarray(full.rgb()[0], np.int32) - np.asarray(compact.rgb()[0], np.int32)).max() <= 1
