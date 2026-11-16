"""`Renderer.point_cloud`: the depth output of two cameras (pinhole and orthographic) unprojected
to world space lies on the scene's analytic surfaces (a floor plane, a sphere) and equals the
position output, for every depth convention and miss value; resampling, id filters, colors and
in-place reuse."""

from __future__ import annotations

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, MjScene

import raywarp as rw

_XML = """
<mujoco>
  <visual><map znear="0.05" zfar="6"/></visual>
  <statistic extent="1"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="ball" pos="0 0 0.4">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.3" rgba="0.9 0.2 0.1 1"/>
    </body>
    <camera name="front" pos="0 -2.5 0.8" xyaxes="1 0 0 0 0.2 1" fovy="60"/>
    <camera name="side" pos="2.2 0.3 1.6" xyaxes="0 1 0 -0.6 0 0.8" fovy="50"/>
  </worldbody>
</mujoco>
"""
BALL_CENTER, BALL_RADIUS = np.array([0.0, 0.0, 0.4]), 0.3
FLOOR, BALL = 0, 1
RES = (24, 18)
NWORLD = 2


def _host(a: wp.array | None) -> np.ndarray:
  """A requested output's values (it must be requested)."""
  assert a is not None
  return a.numpy()


def _scene(scene: MakeScene, depth: rw.settings.DepthOutput, *, projection: str = "perspective", **outputs: object) -> MjScene:
  """`_XML` with two worlds (world 1's ball 0.2 m to the side), its "side" camera in `projection`,
  rendered once with `depth` and the position and seg outputs."""
  xml = _XML.replace('fovy="50"', 'fovy="2.5" projection="orthographic"') if projection == "orthographic" else _XML
  specs = {"depth": depth, "position": rw.settings.FloatOutput(), "seg": rw.settings.IdOutput(), **outputs}
  s = scene(xml, nworld=NWORLD, cameras=["front", "side"], cam_res=RES, outputs=rw.Outputs(**specs))  # pyrefly: ignore[bad-argument-type]
  qpos = np.tile(s.mjd.qpos, (NWORLD, 1)).astype(np.float32)
  qpos[1, 0] += 0.2
  s.d.qpos = wp.array(qpos, dtype=float, device=s.d.qpos.device)
  mjw.forward(s.m, s.d)
  return s.render()


def _hit_positions(s: MjScene, world: int) -> np.ndarray:
  """(n, 3) the position output of world `world`'s hit pixels, camera by camera in row order."""
  pos, seg = _host(s.out.position)[world], _host(s.out.seg)[world]
  return pos[seg >= 0]


def _on_surfaces(points: np.ndarray, world: int, tol: float) -> np.ndarray:
  """Whether each point lies on the floor (z = 0) or on world `world`'s ball, within `tol` [m]."""
  center = BALL_CENTER + np.array([0.2 * world, 0.0, 0.0])
  on_ball = np.abs(np.linalg.norm(points - center, axis=-1) - BALL_RADIUS) < tol
  return (np.abs(points[:, 2]) < tol) | on_ball


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
@pytest.mark.parametrize(
  ("convention", "miss", "dtype"),
  [("planar", "zero", "float32"), ("radial", "inf", "float32"), ("planar", "far", "float32"), ("radial", "far", "float16")],
)
def test_points_are_the_hit_points(scene: MakeScene, projection: str, convention: str, miss: str, dtype: str) -> None:
  """Every hit pixel of both cameras, fused in camera / row order, equals the position output and
  lies on the floor or the ball; misses (0, inf or zfar) are dropped; zeros past `count`."""
  depth = rw.settings.DepthOutput(convention=convention, miss=miss, dtype=dtype)  # pyrefly: ignore[bad-argument-type]
  s = _scene(scene, depth, projection=projection)
  cloud = s.r.point_cloud()
  npix = 2 * RES[0] * RES[1]
  assert cloud.points.shape == (NWORLD, npix) and cloud.colors is None and cloud.ids is None
  tol = 2e-3 * 6.0 if dtype == "float16" else 1e-4 * 6.0  # depth precision x the farthest hit
  for w in range(NWORLD):
    ref = _hit_positions(s, w)
    n = int(cloud.count.numpy()[w])
    assert n == len(ref) and 0 < n < npix
    points = cloud.points.numpy()[w]
    np.testing.assert_allclose(points[:n], ref, atol=tol)
    assert _on_surfaces(points[:n], w, 2 * tol).all()
    assert (points[n:] == 0.0).all()


def test_resampled_cloud_is_a_deterministic_subset(scene: MakeScene) -> None:
  """N points per world, each a hit point, no repeats when a world keeps more than N, the same
  for the same seed, another subset for another; a world keeping fewer repeats them."""
  s = _scene(scene, rw.settings.DepthOutput())
  a = s.r.point_cloud(64, seed=3)
  b = s.r.point_cloud(64, seed=3)
  c = s.r.point_cloud(64, seed=4)
  dense = s.r.point_cloud()
  assert a.points.shape == (NWORLD, 64)
  np.testing.assert_array_equal(a.points.numpy(), b.points.numpy())
  assert not np.array_equal(a.points.numpy(), c.points.numpy())
  for w in range(NWORLD):
    ref = {p.tobytes() for p in dense.points.numpy()[w, : dense.count.numpy()[w]]}  # same computation: exact
    points = a.points.numpy()[w]
    assert {p.tobytes() for p in points} <= ref
    assert len(np.unique(points, axis=0)) == 64
  ball = s.r.point_cloud(1000, cameras=[0], labels="seg", keep=[BALL])
  n = ball.count.numpy()
  assert (n < 1000).all()
  for w in range(NWORLD):
    points = ball.points.numpy()[w]
    np.testing.assert_array_equal(points, points[np.arange(1000) % n[w]])  # repeated cyclically


def test_id_filter_colors_and_reuse(scene: MakeScene) -> None:
  """keep=[ball] keeps the ball's points only (on the sphere, id = ball, red); the colors are the
  rgb output's bytes at each pixel; out= writes the same arrays in place."""
  s = _scene(scene, rw.settings.DepthOutput(), rgb=rw.settings.ColorOutput(channels=3))
  cloud = s.r.point_cloud(labels="seg", keep=[BALL], colors=True)
  assert cloud.ids is not None and cloud.colors is not None
  rgb, seg = _host(s.out.rgb), _host(s.out.seg)
  for w in range(NWORLD):
    n = int(cloud.count.numpy()[w])
    assert n == int((seg[w] == BALL).sum()) > 0
    points = cloud.points.numpy()[w, :n]
    center = BALL_CENTER + np.array([0.2 * w, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(points - center, axis=-1), BALL_RADIUS, atol=1e-3)
    assert (cloud.ids.numpy()[w, :n] == BALL).all() and (cloud.ids.numpy()[w, n:] == -1).all()
    np.testing.assert_array_equal(cloud.colors.numpy()[w, :n], rgb[w][seg[w] == BALL])
  tables = cloud._cache[0].src  # the device tables of the first call
  again = s.r.point_cloud(labels="seg", keep=[BALL], colors=True, out=cloud)
  assert again is cloud
  assert cloud._cache[0].src is tables  # reused: no allocation, capture-safe
  with pytest.raises(ValueError, match="same request"):
    s.r.point_cloud(labels="seg", keep=[FLOOR], colors=True, out=cloud)


def test_reuse_follows_clip_changes_and_takes_array_arguments(scene: MakeScene) -> None:
  """`out=` reuse rebuilds the tables a change of the clip planes stales (the "far" miss depth:
  every miss was kept as a point at the old zfar), and NumPy `cameras` / `keep` compare by value
  (they raised "truth value of an array is ambiguous")."""
  s = _scene(scene, rw.settings.DepthOutput(miss="far"))
  cloud = s.r.point_cloud()
  s.r.set_clip(zfar=5.0)
  s.render()
  again = s.r.point_cloud(out=cloud)
  np.testing.assert_array_equal(again.count.numpy(), s.r.point_cloud().count.numpy())
  ids = {"cameras": np.array([0]), "labels": "seg", "keep": np.array([BALL])}
  picked = s.r.point_cloud(**ids)  # pyrefly: ignore[bad-argument-type]  # NumPy ids on purpose
  assert s.r.point_cloud(**ids, out=picked) is picked  # pyrefly: ignore[bad-argument-type]  # NumPy ids on purpose
  with pytest.raises(TypeError):
    s.r.point_cloud(cameras=[0.5])  # pyrefly: ignore[bad-argument-type]  # a float camera on purpose


def test_point_cloud_errors(scene: MakeScene) -> None:
  """Missing outputs, keep without labels and unknown cameras raise."""
  s = _scene(scene, rw.settings.DepthOutput())
  for kwargs, match in (
    ({"keep": [1]}, "labels"),
    ({"cameras": [2]}, "cameras"),
    ({"labels": "semantic"}, "outputs.semantic"),
    ({"colors": True}, "outputs.rgb"),
    ({"npoints": 0}, "npoints"),
  ):
    with pytest.raises(ValueError, match=match):
      s.r.point_cloud(**kwargs)  # pyrefly: ignore[bad-argument-type]
  no_depth = scene(_XML, cam_res=RES, outputs=rw.Outputs.of("seg")).render()
  with pytest.raises(ValueError, match="depth output"):
    no_depth.r.point_cloud()
