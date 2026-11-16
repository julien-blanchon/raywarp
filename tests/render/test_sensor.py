"""The sensor pass (`settings.Sensor`, `core/post/sensor/effects.py`): shot / read noise statistics,
vignette, ADC quantisation, response curve, stereo depth noise and occlusion, range depth noise,
edge dropout, rolling shutter and motion blur,
against their closed forms; off means no pass; runtime toggles never recompile."""

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import DOWN, MakeScene
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp._base.named import construct_named
from raywarp.settings.sensor import IDENTITY_RESPONSE

# A camera 1 m above a grey floor (left half dark, right half bright: one vertical edge), on a
# slide joint along x; a thin plate 0.5 m below the camera, right of the image centre (`_PLATE`).
XML = """
<mujoco>
  <worldbody>
    <geom type="box" pos="-5 0 -0.01" size="5 5 0.01" rgba="0.2 0.2 0.2 1"/>
    <geom type="box" pos="5 0 -0.01" size="5 5 0.01" rgba="0.6 0.6 0.6 1"/>
    <body name="rig" pos="0 0 1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <camera name="top" pos="0 0 0" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""
_PLATE = '<geom type="box" pos="0.15 0 0.5" size="0.05 0.6 0.001" rgba="0.6 0.6 0.6 1"/>'
_UNLIT = {"surfaces": rw.settings.Surfaces(shading="unlit"), "aa": rw.settings.AntiAliasing()}
_PLANAR = rw.settings.ColorOutput(channels=3, planar=True, dtype="float16")


def _linear(planes: np.ndarray) -> np.ndarray:
  """sRGB-encoded planar float16 values -> display-linear float64."""
  x = planes.astype(np.float64)
  return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _array(a: wp.array | None) -> np.ndarray:
  """An output as NumPy (it must be requested)."""
  assert a is not None
  return a.numpy()


def _color(s, world: int = 0) -> np.ndarray:
  """World `world`'s (3, H, W) display-linear color of a planar float16 render."""
  return _linear(_array(s.out.rgb)[world])


def _sensor(**fields: object) -> rw.settings.Sensor:
  return construct_named(rw.settings.Sensor, fields)


def test_off_runs_no_pass_and_on_off_never_recompiles(scene: MakeScene):
  s = scene(XML, cam_res=(16, 12), outputs=rw.Outputs.of(rgb=_PLANAR), **_UNLIT)
  assert s.r._sensor is None
  clean = _color(s.render())
  key = s.r.features
  s.r.update_settings(**{"sensor.noise": True})
  assert s.r.features == key and s.r._sensor is not None
  assert not np.array_equal(_color(s.render()), clean)
  with module_loads() as loads:  # the sensor pass is loaded: switching it off and on loads nothing
    s.r.update_settings(**{"sensor.noise": False})
    assert s.r._sensor is None
    np.testing.assert_array_equal(_color(s.render()), clean)
    s.r.update_settings(**{"sensor.noise": True})
    assert not np.array_equal(_color(s.render()), clean)
  assert s.r.features == key and not loads, loads


def test_shot_and_read_noise_statistics(scene: MakeScene):
  """Mean preserved, variance v / full_well + (read_noise / full_well)^2 (display-linear)."""
  well, read = 200.0, 2.0
  s = scene(XML, cam_res=(64, 48), outputs=rw.Outputs.of(rgb=_PLANAR), **_UNLIT)
  clean = _color(s.render())
  s.r.update_settings(sensor=_sensor(noise=True, full_well=well, read_noise=read))
  noisy = _color(s.render(seed=3))
  for half in (slice(0, 30), slice(34, 64)):  # the dark and the bright floor
    v = clean[:, :, half]
    assert np.ptp(v) < 1e-3
    n = noisy[:, :, half]
    want = v.mean() / well + (read / well) ** 2
    assert abs(n.mean() / v.mean() - 1.0) < 0.03
    assert abs(n.var() / want - 1.0) < 0.1


def test_noise_is_per_world_seeded_temporal_and_restarts(scene: MakeScene):
  """World 0's noise is the same in a batch of 1 and of 3; it changes per frame and per seed;
  reset_history restarts its sequence."""
  kw = {"cam_res": (16, 12), "outputs": rw.Outputs.of(rgb=_PLANAR), "sensor": _sensor(noise=True, full_well=100.0)}
  one = scene(XML, **kw, **_UNLIT)
  three = scene(XML, nworld=3, **kw, **_UNLIT)
  first = _color(one.render(seed=7))
  np.testing.assert_array_equal(_color(three.render(seed=7)), first)
  assert not np.array_equal(_color(three.render(seed=7), 1), first)  # worlds differ
  second = _color(one.render(seed=7))
  assert not np.array_equal(second, first)
  one.r.reset_history()
  np.testing.assert_array_equal(_color(one.render(seed=7)), first)


def test_vignette_follows_its_polynomial_and_is_per_world(scene: MakeScene):
  a1 = -0.8
  s = scene(
    XML,
    nworld=2,
    cam_res=(32, 32),
    outputs=rw.Outputs.of(rgb=_PLANAR),
    per_world=rw.PerWorld(env=["sensor.vignette"]),
    **_UNLIT,
  )
  clean = _color(s.render(), 1)
  s.r.per_world["sensor.vignette"].assign(np.array([[0, 0, 0], [a1, 0, 0]], np.float32))
  out = s.render()
  np.testing.assert_allclose(_color(out, 0), clean, atol=1e-3)  # world 0: no falloff
  x = (np.arange(32) + 0.5 - 16) / 32
  r2 = x[None, :] ** 2 + x[:, None] ** 2
  np.testing.assert_allclose(_color(out, 1), clean * np.clip(1 + a1 * r2, 0, 1), rtol=2e-3, atol=1e-3)


def test_response_curve_maps_the_encoded_value(scene: MakeScene):
  curve = (2.0, 0.5, 0.4, 1.2)
  s = scene(XML, cam_res=(16, 12), outputs=rw.Outputs.of(rgb=_PLANAR), **_UNLIT)
  enc = _array(s.render().out.rgb)[0].astype(np.float64)
  s.r.update_settings(**{"sensor.response": curve})
  got = _array(s.render().out.rgb)[0].astype(np.float64)
  toe, shoulder, center, gamma = curve
  a = shoulder * center / ((shoulder - toe) * center + toe)
  y = np.where(enc <= center, a * (enc / center) ** toe, 1 - (1 - a) * ((1 - enc) / (1 - center)) ** shoulder) ** gamma
  np.testing.assert_allclose(got, y, atol=2e-3)
  assert rw.settings.Sensor().response == IDENTITY_RESPONSE


def _depth_scene(scene: MakeScene, depth: str = "stereo", **sensor: object):
  return scene(
    XML.replace("<body", _PLATE + "<body"),
    cam_res=(64, 48),
    outputs=rw.Outputs.of("depth"),
    sensor=_sensor(depth=depth, **sensor),
  )


def test_stereo_depth_noise_quantisation_and_occlusion_band(scene: MakeScene):
  """Disparity noise gives depth rms z^2 sigma / (f b); the quantum bounds the error; the band
  left of the plate (seen by the left imager only) is fb / z_plate - fb / z_floor pixels wide."""
  f = 24 / np.tan(np.radians(30))  # focal length [px]: 48 rows, 60 degrees
  b = 0.05
  exact = _depth_scene(scene, disparity_noise=0.0, disparity_step=0.0).render().depth()[0]
  floor = exact[:, :8]
  np.testing.assert_allclose(floor, 1.0, rtol=1e-5)
  band = int(np.sum(exact[24] == 0.0))
  assert abs(band - (f * b / 0.5 - f * b / 1.0)) <= 1.0
  hit = exact[24] > 0
  plate = np.flatnonzero(hit & (exact[24] < 0.75))
  left = int(plate[0])
  assert np.all(exact[24, left - band : left] == 0.0)  # the band is left of the plate
  stepped = _depth_scene(scene, disparity_noise=0.0, disparity_step=0.25).render().depth()[0]
  assert np.abs(stepped[:, :8] - 1.0).max() <= 1.0**2 * 0.125 / (f * b) * 1.001
  # The floor's disparity f b / z (2.08 px) on the quantum's lattice (2.0 px): fb / 2.0.
  np.testing.assert_allclose(stepped[:, :8], f * b / (np.round(f * b / 0.25) * 0.25), rtol=1e-4)
  noisy = _depth_scene(scene, disparity_noise=0.2, disparity_step=0.0).render(seed=1).depth()[0]
  rms = np.sqrt(np.mean((noisy[:, :8] - 1.0) ** 2))
  assert abs(rms / (0.2 / (f * b)) - 1.0) < 0.15


def test_stereo_depth_range_and_dropout(scene: MakeScene):
  near = _depth_scene(scene, disparity_noise=0.0, disparity_step=0.0, max_depth=0.8).render().depth()[0]
  assert np.all(near[:, :8] == 0.0)  # the floor at 1 m is beyond the range
  dropped = _depth_scene(scene, disparity_noise=0.0, disparity_step=0.0, dropout=0.3).render().depth()[0]
  assert abs(np.mean(dropped[:, :8] == 0.0) - 0.3) < 0.08


def test_adc_quantises_the_linear_signal(scene: MakeScene):
  """bits = 3: every display-linear value is on the 1/7 lattice (no noise, identity response)."""
  s = scene(XML, cam_res=(16, 12), outputs=rw.Outputs.of(rgb=_PLANAR), sensor=_sensor(bits=3), **_UNLIT)
  got = _color(s.render()) * 7.0
  np.testing.assert_allclose(got, np.round(got), atol=7.0 * 2e-3)  # float16 sRGB encoding
  assert len(np.unique(np.round(got))) == 2  # the two floor halves, on their own steps


def test_range_depth_noise_grows_with_the_distance(scene: MakeScene):
  """sigma0 + k (z - z0)^2 metres rms: 0.03 at the floor (1 m), 0.015 on the plate (0.5 m)."""
  coeff = (0.01, 0.0, 0.02)
  exact = _depth_scene(scene, "range", range_noise=(0.0, 0.0, 0.0)).render().depth()[0]
  noisy = _depth_scene(scene, "range", range_noise=coeff).render(seed=2).depth()[0]
  for z, sigma in ((1.0, 0.03), (0.5, 0.015)):
    mask = np.abs(exact - z) < 1e-3
    assert mask.sum() > 200
    rms = np.sqrt(np.mean((noisy[mask] - exact[mask]) ** 2))
    assert abs(rms / sigma - 1.0) < 0.15


def test_edge_dropout_drops_the_pixels_at_depth_edges(scene: MakeScene):
  """With probability 1 exactly the pixels next to a jump of more than 5 % (the plate's border)
  read the miss value; the others keep their depth."""
  exact = _depth_scene(scene, "range", range_noise=(0.0, 0.0, 0.0)).render().depth()[0]
  dropped = _depth_scene(scene, "range", range_noise=(0.0, 0.0, 0.0), edge_dropout=1.0, edge_threshold=0.05)
  got = dropped.render().depth()[0]
  pad = np.pad(exact, 1, mode="edge")
  jumps = [
    np.abs(pad[1 + dy : 1 + dy + 48, 1 + dx : 1 + dx + 64] - exact) > 0.05 * exact
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
  ]
  edge = np.logical_or.reduce(jumps)
  assert 50 < edge.sum() < 500
  np.testing.assert_array_equal(got[edge], 0.0)
  np.testing.assert_array_equal(got[~edge], exact[~edge])


def test_range_depth_model_takes_any_camera(scene: MakeScene):
  s = scene(
    XML,
    cam_res=(16, 12),
    cameras=[rw.CameraSpec(projection="orthographic", fovy=1.0)],
    outputs=rw.Outputs.of("depth"),
    sensor=_sensor(depth="range", range_noise=(0.0, 0.0, 0.0)),
  )
  s.r.camera_pose.assign(np.array([[[0.0, 0.0, 1.0, *DOWN]]], np.float32))  # 1 m above the floor
  np.testing.assert_allclose(s.render().depth()[0], 1.0, rtol=1e-5)


def _move(s, x: float) -> None:
  qpos = s.mjd.qpos.copy()
  qpos[0] = x
  s.d.qpos = wp.array(np.tile(qpos, (s.d.nworld, 1)).astype(np.float32), dtype=float)
  mjw.forward(s.m, s.d)


def _edge(row: np.ndarray) -> float:
  """Sub-pixel column of the dark -> bright step of a row (its normalised integral)."""
  lo, hi = row.min(), row.max()
  return float(len(row) - np.sum((row - lo) / (hi - lo)))


def _moving(scene: MakeScene, sensor: rw.settings.Sensor):
  s = scene(XML, cam_res=(64, 48), outputs=rw.Outputs.of("motion", rgb=_PLANAR), sensor=sensor, **_UNLIT)
  s.render()
  _move(s, 0.05)
  s.render()
  return s


def test_rolling_shutter_slants_a_moving_edge(scene: MakeScene):
  rs = 0.8
  s = _moving(scene, _sensor(rolling_shutter=rs))
  m = _array(s.out.motion)[0, 0]
  mx = float(m[24, 8, 0])
  assert abs(mx) > 1.0 and np.ptp(m[..., 0]) < 1e-3  # a uniform motion
  img = _color(s)[1]
  top, bottom = _edge(img[0]), _edge(img[-1])
  # Row y shows the image at p - s_y m: the edge moves by s_y m_x, s_y = rs (1 - (y + 0.5) / H).
  assert abs((top - bottom) - rs * mx * (1 - 1 / 48)) < 0.3


def test_motion_blur_spreads_a_moving_edge_over_its_streak(scene: MakeScene):
  blur = 1.0
  sharp = _color(_moving(scene, _sensor(rolling_shutter=0.0, motion_blur=0.0, noise=False, vignette=(0.0, 0.0, 0.0))))[1]
  s = _moving(scene, _sensor(motion_blur=blur))
  mx = float(_array(s.out.motion)[0, 0, 24, 8, 0])
  row = _color(s)[1, 24]
  lo, hi = row.min(), row.max()
  ramp = int(np.sum((row > lo + 0.05 * (hi - lo)) & (row < hi - 0.05 * (hi - lo))))
  assert abs(ramp - 0.9 * blur * abs(mx)) <= 2.0
  assert abs(float(row.mean()) - float(sharp[24].mean())) < 0.02  # energy kept (bar the border)


# A static camera 1 m above a dark floor and a bright bar 0.5 m below it, on a slide joint along x.
MOVING_BAR = """
<mujoco>
  <worldbody>
    <geom type="box" pos="0 0 -0.01" size="5 5 0.01" rgba="0.1 0.1 0.1 1"/>
    <body name="bar" pos="0 0 0.5">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.08 0.6 0.001" rgba="0.9 0.9 0.9 1"/>
    </body>
    <camera name="top" pos="0 0 1" fovy="60"/>
  </worldbody>
</mujoco>
"""


def _ramps(row: np.ndarray) -> int:
  """Pixels of a row strictly between its darkest and brightest 5 %."""
  lo, hi = row.min(), row.max()
  return int(np.sum((row > lo + 0.05 * (hi - lo)) & (row < hi - 0.05 * (hi - lo))))


def test_motion_blur_streaks_a_moving_object_over_the_static_background(scene: MakeScene):
  """A bar moving under a static camera: over the exposure it sweeps the background behind it as
  well as its own leading side, so both its edges ramp over the motion (the tile maximum's
  dominant motion; the background's own zero motion alone would keep the trailing edge sharp)."""
  blur = 1.0
  sensor = _sensor(motion_blur=blur)
  s = scene(MOVING_BAR, cam_res=(64, 48), outputs=rw.Outputs.of("motion", rgb=_PLANAR), sensor=sensor, **_UNLIT)
  s.render()
  _move(s, 0.06)
  s.render()
  motion = _array(s.out.motion)[0, 0, 24]
  dx = float(np.abs(motion[:, 0]).max())
  assert dx > 4.0 and float(np.abs(motion[:4, 0]).max()) == 0.0  # the bar moves, the floor not
  row = _color(s)[1, 24]
  assert abs(_ramps(row) - 2 * 0.9 * blur * dx) <= 3.0
  s.r.update_settings(**{"sensor.motion_blur": 0.0})
  sharp = _color(s.render())[1, 24]  # the same frame unblurred: the exposure keeps the energy
  assert _ramps(sharp) <= 2 and abs(float(row.mean()) - float(sharp.mean())) < 0.01


def test_motion_blur_of_a_second_camera_matches_it_alone(scene: MakeScene):
  """Two cameras: the second one's blurred image (its own motion tiles, after the first's) is the
  image the same camera renders alone."""
  xml = MOVING_BAR.replace(
    '<camera name="top" pos="0 0 1" fovy="60"/>',
    """<camera name="side" pos="0.3 0 1" fovy="70"/>
    <camera name="top" pos="0 0 1" fovy="60"/>""",
  )
  images = []
  views: tuple[list[rw.CameraSpec | int | str], ...] = (["side", "top"], ["top"])
  for cameras in views:
    s = scene(
      xml,
      cam_res=(80, 48),
      cameras=cameras,
      outputs=rw.Outputs.of("rgb", "motion"),
      sensor=_sensor(motion_blur=1.0),
      **_UNLIT,
    )
    s.render()
    _move(s, 0.06)
    s.render()
    images.append(_array(s.out.rgb)[0, -1])  # (H, W, 4) bytes of the last camera
  assert _ramps(images[1][24, :, 0].astype(np.float64)) > 6
  np.testing.assert_array_equal(images[0], images[1])


_RGB_DEPTH = rw.Outputs.of(rgb=_PLANAR, depth=rw.settings.DepthOutput())


def _plate(slope: float, phase: float) -> str:
  """A bright plate 0.5 m above a dark floor under a downward camera: in its image of `_ortho`'s
  rays (n pixels across 1 m), bright where row + 0.5 > slope x (column + 0.5 - n / 2) + n / 2 +
  phase x n."""
  normal = -np.array([slope, 1.0]) / np.hypot(slope, 1.0)
  x, y = np.array([0.0, -phase]) + 2.0 * normal
  angle = np.degrees(np.arctan2(normal[1], normal[0]))
  return f"""
<mujoco>
  <worldbody>
    <geom type="box" pos="0 0 -0.01" size="5 5 0.01" rgba="0.1 0.1 0.1 1"/>
    <geom type="box" pos="{x} {y} 0.5" euler="0 0 {angle}" size="2 4 0.001" rgba="0.9 0.9 0.9 1"/>
    <camera name="top" pos="0 0 1" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def _ortho(n: int) -> np.ndarray:
  """Parallel rays down through the pixel centres of an n x n image 1 m wide."""
  c = (np.arange(n) + 0.5) / n - 0.5
  origin = np.stack(np.broadcast_arrays(c[None, :], -c[:, None], np.zeros((1, 1))), axis=-1)
  direction = np.broadcast_to([0.0, 0.0, -1.0], origin.shape)
  return np.stack([origin, direction], axis=2).astype(np.float32)


@pytest.mark.parametrize("slope", [1 / 8, 1 / 3, 5.0])
def test_post_aa_ramps_a_stair_stepped_silhouette_toward_its_coverage(scene: MakeScene, slope: float):
  """`aa.post` on 1-spp images of a straight silhouette: the pixels' error to their exact coverage
  (16 x 16 supersampled), summed over two sub-pixel phases of the edge, at least halves (stair
  steps become ramps; phases off the pixel centres: a centre on the edge is a tie)."""
  n, ss = 64, 16
  y, x = np.mgrid[0 : n * ss, 0 : n * ss]
  kw = {"cam_res": (n, n), "outputs": _RGB_DEPTH, "rays": [_ortho(n)], **_UNLIT}
  inner = (slice(8, n - 8), slice(8, n - 8))
  before = after = 0.0
  for phase in (0.13, 0.61):
    xml = _plate(slope, phase / n)
    aliased = _color(scene(xml, **kw).render())[1]
    smooth = _color(scene(xml, **{**kw, "aa": rw.settings.AntiAliasing(post=True)}).render())[1]
    inside = (y + 0.5) / ss > slope * ((x + 0.5) / ss - n / 2) + n / 2 + phase
    lo, hi = aliased.min(), aliased.max()
    exact = lo + (hi - lo) * inside.reshape(n, ss, n, ss).mean((1, 3))
    before += np.abs(aliased - exact)[inner].sum()
    after += np.abs(smooth - exact)[inner].sum()
  assert after < 0.5 * before, (before, after)


CHECKER = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.9 0.9 0.9" width="64" height="64"/>
    <material name="grid" texture="grid" texrepeat="5 5"/>
  </asset>
  <worldbody>
    <geom type="plane" pos="0 0 0" euler="0 0 17" size="2 2 0.1" material="grid"/>
    <geom type="box" pos="0.1 0 0.3" euler="0 0 30" size="0.12 0.12 0.001" rgba="0.5 0.5 0.5 1"/>
    <camera name="top" pos="0 0 1" fovy="60"/>
  </worldbody>
</mujoco>
"""


def test_post_aa_leaves_texture_edges_as_rendered(scene: MakeScene):
  """A tilted checker floor (stair-stepped texture edges of full contrast) under a plate: `aa.post`
  changes pixels only next to the plate's silhouette (its depth jump), every texture edge's
  pixel is as rendered."""
  kw = {"cam_res": (64, 64), "outputs": _RGB_DEPTH, **_UNLIT}
  s = scene(CHECKER, **kw)
  s.render()
  plain = _color(s)
  depth = s.depth()[0]
  s.r.update_settings(**{"aa.post": True})
  smooth = _color(s.render())
  jump = np.zeros(depth.shape, bool)
  for axis in (0, 1):
    step = np.abs(np.diff(depth, axis=axis)) > 0.1
    lo = [slice(None)] * 2
    hi = [slice(None)] * 2
    lo[axis], hi[axis] = slice(None, -1), slice(1, None)
    jump[tuple(lo)] |= step
    jump[tuple(hi)] |= step
  changed = np.abs(smooth - plain).max(0) > 0
  assert changed.any() and not (changed & ~jump).any(), np.argwhere(changed & ~jump)[:8]


def test_post_aa_is_a_runtime_switch_without_a_pass_when_off(scene: MakeScene):
  """`aa.post` on smooths a tilted box's silhouette (some pixels change, none far from the 1-spp
  image's range); off again the pass is gone and the image is the plain one; no recompile."""
  xml = MOVING_BAR.replace('size="0.08 0.6 0.001"', 'size="0.08 0.6 0.001" euler="0 0 17"')
  s = scene(xml, cam_res=(64, 48), outputs=_RGB_DEPTH, **_UNLIT)
  plain = _color(s.render())
  assert s.r._sensor is None
  s.r.update_settings(**{"aa.post": True})
  key = s.r.features
  smooth = _color(s.render())
  changed = np.abs(smooth - plain).max(0) > 0.01
  assert 0 < changed.mean() < 0.2 and smooth.min() >= plain.min() - 1e-3 and smooth.max() <= plain.max() + 1e-3
  with module_loads() as loads:
    s.r.update_settings(**{"aa.post": False})
    assert s.r._sensor is None
    np.testing.assert_array_equal(_color(s.render()), plain)
    s.r.update_settings(**{"aa.post": True})
    np.testing.assert_array_equal(_color(s.render()), smooth)
  assert s.r.features == key and not loads, loads


@pytest.mark.parametrize(
  ("change", "message"),
  [
    ({"sensor.motion_blur": 0.5}, "outputs.motion"),
    ({"sensor.depth": "stereo"}, "outputs.depth"),
    ({"aa.post": True, "outputs": rw.Outputs.of("depth")}, "outputs.rgb"),
    ({"aa.post": True}, "outputs.depth"),
    ({"sensor.response": (1.0, 1.0, 1.5, 1.0)}, "center"),
    ({"sensor.response": (1.0, float("nan"), 0.5, 1.0)}, "finite"),
    ({"sensor.response": (1.0, float("inf"), 0.5, 1.0)}, "finite"),
    ({"sensor.vignette": (float("nan"), 0.0, 0.0)}, "finite"),
    ({"sensor.vignette": (0.6, 0.6, 0.6)}, "brightens"),  # clamped to 1 everywhere: no effect
    ({"sensor.vignette": (-0.5, 2.0, 0.0)}, "brightens"),  # brighter than the centre at the corners
    ({"sensor.vignette": (0.0, 1.0, -8.0)}, "brightens"),  # a bump inside (the concave quadratic's vertex)
    ({"sensor.range_noise": (-0.01, 0.0, 0.0)}, "sigma0, k >= 0"),
    ({"sensor.range_noise": (0.01, 0.0)}, "3 finite numbers"),
    ({"sensor.bits": 17}, "sensor.bits"),
    ({"sensor.edge_threshold": 0.0}, "sensor.edge_threshold"),
    ({"atmosphere.haze": float("inf")}, "atmosphere.haze"),
    ({"gi.normal_bias": float("inf")}, "gi.normal_bias"),
    ({"gi.bounds": ((0.0, 0.0, 0.0), (float("inf"), 1.0, 1.0))}, "finite"),
  ],
)
def test_invalid_sensor_settings_raise(change, message):
  with pytest.raises(ValueError, match=message):
    rw.RenderSettings().replace(change)


def test_depth_model_needs_perspective_cameras(scene: MakeScene):
  with pytest.raises(ValueError, match="perspective"):
    scene(
      XML,
      cameras=[rw.CameraSpec(projection="orthographic", fovy=1.0)],
      outputs=rw.Outputs.of("depth"),
      sensor=_sensor(depth="stereo"),
    )


def test_depth_model_rejects_distorted_cameras(scene: MakeScene):
  with pytest.raises(ValueError, match="undistorted"):
    scene(
      XML,
      cameras=[rw.CameraSpec.from_opencv([[20, 0, 16], [0, 20, 12], [0, 0, 1]], (32, 24), distortion=(0.1, 0, 0, 0))],
      outputs=rw.Outputs.of("depth"),
      sensor=_sensor(depth="stereo"),
    )
