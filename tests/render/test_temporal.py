"""The motion output's temporal state (M5): history, reprojection, resets."""

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, required

import raywarp as rw
from raywarp.core.post.temporal.history import FRAME_WRAP

XML = """
<mujoco>
  <worldbody>
    <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
    <light type="point" pos="0.5 -0.5 1.5" bulbradius="0.2" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.8 0.3 0.2 1"/>
    <geom type="capsule" fromto="-1 0.5 0.05 1 0.8 0.9" size="0.01" rgba="0.9 0.9 0.2 1"/>
    <body name="arm" pos="0.5 -0.3 0.4">
      <freejoint/>
      <geom type="box" size="0.25 0.05 0.05" rgba="0.2 0.4 0.9 1"/>
    </body>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
  </worldbody>
</mujoco>
"""
_GEOM = rw.Outputs.of("rgb", "depth", "seg")


def _move_body(s, dx: float) -> None:
  qpos = s.mjd.qpos.copy()
  qpos[s.mjm.jnt_qposadr[0]] += dx
  s.d.qpos = wp.array(np.tile(qpos, (s.d.nworld, 1)).astype(np.float32), dtype=float)
  mjw.forward(s.m, s.d)


def test_everything_off_compiles_the_plain_megakernel(scene):
  s = scene(XML, outputs=_GEOM, aa=rw.settings.AntiAliasing())
  assert not s.r.features.history and s.r.debug.history_length is None
  s.r.reset_history()  # a no-op


@pytest.mark.parametrize("moving", ["body", "projection"])
def test_reset_discards_prior_pose_motion_only_in_selected_world(scene: MakeScene, moving: str) -> None:
  """A moved view reset to its original pose must reproduce its first frame, including motion."""
  s = scene(
    XML,
    nworld=2,
    cam_res=(48, 36),
    outputs=rw.Outputs.of("rgb", "motion"),
    per_world=rw.PerWorld(camera=["fovy"]),
  )
  _move_body(s, 0.0)  # Use the same MJW forward path for the initial and restored pose.
  s.render(seed=5)
  original = required(s.out.rgb).numpy().copy()
  fovy = s.r.per_world["camera.fovy"]
  projection = fovy.numpy().copy()
  if moving == "body":
    _move_body(s, 0.25)
  else:
    fovy.assign(projection * 1.2)
  s.render(seed=5)
  assert np.max(np.abs(required(s.out.motion).numpy())) > 0.5
  if moving == "body":
    _move_body(s, 0.0)
  else:
    fovy.assign(projection)
  s.r.reset_history(wp.array([False, True], dtype=bool, device=s.r.device))
  s.render(seed=5)
  np.testing.assert_array_equal(required(s.out.rgb).numpy()[1], original[1])
  motion = required(s.out.motion).numpy()
  np.testing.assert_array_equal(motion[1], 0.0)
  assert np.max(np.abs(motion[0])) > 0.5  # The unreset world still reports actual motion.
  assert required(s.r.debug.history_length).numpy().tolist() == [[3], [1]]


def test_motion_is_zero_when_still_and_reprojects_moving_geometry(scene):
  s = scene(XML, cam_res=(48, 36), outputs=rw.Outputs.of("depth", "seg", "motion"))
  s.render().render()
  assert s.out.motion.shape == (1, 1, 36, 48) and s.out.motion.dtype == wp.vec2
  assert np.abs(s.out.motion.numpy()).max() == 0.0
  prev = s.seg()[0].copy()
  _move_body(s, 0.1)
  s.render()
  mv = s.out.motion.numpy()[0, 0]
  seg = s.seg()[0]
  arm = seg == s.mjm.ngeom - 1
  assert np.abs(mv[arm]).max() > 0.5 and np.abs(mv[seg == 0]).max() < 1e-3  # only the body moves
  ys, xs = np.mgrid[0:36, 0:48]
  qx = np.clip((xs + 0.5 + mv[..., 0]).astype(int), 0, 47)
  qy = np.clip((ys + 0.5 + mv[..., 1]).astype(int), 0, 35)
  assert (prev[qy, qx] == seg)[arm].mean() > 0.8


def test_motion_follows_a_change_of_projection(scene):
  """W23: the reprojection uses the previous frame's projection: a zoom (fovy change) of a still
  camera moves every pixel away from the centre by the ratio of the half-angle tangents."""
  s = scene(XML, cam_res=(48, 36), outputs=rw.Outputs.of("seg", "motion"), per_world=rw.PerWorld(camera=["fovy"]))
  s.render().render()
  assert np.abs(s.out.motion.numpy()).max() == 0.0
  fovy = s.r.per_world["camera.fovy"]
  values = fovy.numpy()
  old = float(values[0, 0])
  values[0, 0] = old * 1.1
  fovy.assign(values)
  s.render()
  mv = s.out.motion.numpy()[0, 0]
  k = np.tan(np.radians(old * 1.1) / 2.0) / np.tan(np.radians(old) / 2.0)
  ys, xs = np.mgrid[0:36, 0:48]
  np.testing.assert_allclose(mv[..., 0], (xs + 0.5 - 24.0) * (k - 1.0), atol=2e-3)
  np.testing.assert_allclose(mv[..., 1], (ys + 0.5 - 18.0) * (k - 1.0), atol=2e-3)


def test_motion_with_area_lights(scene):
  # Sphere lights are physical (photometric model): MuJoCo's classic GL lights stay punctual.
  xml = XML.replace(
    '<light type="directional" dir="0.3 0.2 -1" castshadow="true"/>',
    '<light type="directional" dir="0.3 0.2 -1" intensity="40000" castshadow="true"/>',
  ).replace('bulbradius="0.2" castshadow="true"/>', 'bulbradius="0.2" intensity="5000" castshadow="true"/>')
  s = scene(
    xml,
    cam_res=(24, 16),
    outputs=rw.Outputs.of("rgb", "motion"),
    lighting=rw.settings.Lights(area_lights="ltc"),
    shadows=rw.settings.Shadows(samples=2),
  )
  assert s.r.features.area_lights and s.r.features.history
  s.render().render()
  assert np.isfinite(s.rgb()).all() and s.rgb().mean() > 0.05


def _zoom(s, ratio: float) -> None:
  fovy = s.r.per_world["camera.fovy"]
  fovy.assign(fovy.numpy() * ratio)


@pytest.mark.parametrize("move", ["body", "zoom"])
def test_motion_of_a_world_resumed_after_a_skip_restarts(scene, move):
  """A world rendered again after a skip restarts its history: its motion is that of a first
  frame of the moved scene, not one measured against poses or cameras from before the skip."""
  kw = {
    "nworld": 2,
    "cam_res": (24, 16),
    "outputs": rw.Outputs.of("rgb", "seg", "motion"),
    "per_world": rw.PerWorld(camera=["fovy"]),
  }

  def moved(s) -> None:
    if move == "body":
      _move_body(s, 0.1)
    else:
      _zoom(s, 1.1)

  s = scene(XML, **kw)
  s.render().render()
  active = wp.array([0], dtype=wp.int32)
  moved(s)
  s.r.render(s.d, active=active)
  assert np.abs(s.out.motion.numpy()[0]).max() > 0.0  # world 0's first moved frame has motion
  s.r.render(s.d, active=active)
  s.render()
  fresh = scene(XML, **kw)
  moved(fresh)
  fresh.render()
  np.testing.assert_array_equal(s.out.motion.numpy()[1], fresh.out.motion.numpy()[1])


def test_frame_index_wraps_before_int32_overflow(scene):
  """The per-world frame index (the previous-depth slot frame % 2) is an int32: past 2^31 frames it
  turned negative and indexed slot -1. It wraps to 0 after FRAME_WRAP frames."""
  s = scene(XML, cam_res=(24, 16), outputs=rw.Outputs.of("rgb", "motion"))
  s.render()
  frame = s.r.context().history.struct.frame
  frame.fill_(FRAME_WRAP - 1)
  s.render()
  assert frame.numpy()[0] == 0


@pytest.mark.parametrize("pose", ["nan_position", "zero_quaternion"])
def test_invalid_camera_pose_raises(scene, pose):
  """A NaN camera position or a zero quaternion rendered a silent black view."""
  s = scene(XML, cam_res=(8, 6), outputs=rw.Outputs.of("rgb"), cameras=[rw.CameraSpec(fovy=50.0)])
  p = np.array([[[0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 1.0]]], np.float32)
  s.r.camera_pose.assign(p)
  s.render()
  if pose == "nan_position":
    p[..., 0] = np.nan
  else:
    p[..., 3:] = 0.0
  s.r.camera_pose.assign(p)
  with pytest.raises(ValueError, match="camera poses must be finite"):
    for _ in range(3):
      s.render()
