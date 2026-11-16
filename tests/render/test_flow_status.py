"""The motion output's status (`MotionOutput(status=True)`: `RenderOutputs.flow_status`), checked
against closed forms: an orthographic view of a box sliding over a floor (the floor it uncovers
was OCCLUDED), and a pinhole camera moving back over a plane (the previous camera's projection of
every hit point, from the position output, says BEHIND / OUT_OF_FRAME / VALID)."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, MjScene

import raywarp as rw

F = rw.RenderOutputs.FlowStatus

# Orthographic camera 3 m above a 2 x 2 m floor, 2.4 m view height at 48 px: 20 px per metre; the
# box's top (0.4 x 0.4 m, 8 px) starts at x = -0.3 m. Every edge falls between pixel centres.
_SLIDE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="box" size="1 1 0.01" pos="0 0 -0.01" rgba="0.7 0.7 0.7 1"/>
    <body name="box" pos="-0.3 0 0.2">
      <freejoint/>
      <geom name="box" type="box" size="0.2 0.2 0.2" rgba="0.2 0.4 0.9 1"/>
    </body>
    <camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="2.4"/>
  </worldbody>
</mujoco>
"""
RES = 48
PX_PER_M = RES / 2.4
_STATUS = rw.settings.MotionOutput(status=True)


def _host(a: wp.array | None) -> np.ndarray:
  """A requested output's values (it must be requested)."""
  assert a is not None
  return a.numpy()


def _move(s: MjScene, dx: float, joint: int = 0) -> None:
  """Moves free joint `joint` by `dx` along x (y with a tuple index) in every world."""
  qpos = s.mjd.qpos.copy()
  qpos[s.mjm.jnt_qposadr[joint]] += dx
  s.d.qpos = wp.array(np.tile(qpos, (s.d.nworld, 1)).astype(np.float32), dtype=float)
  mjw.forward(s.m, s.d)


def _rect(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
  """(RES, RES) bool: the pixel centres of the orthographic view inside [x0, x1] x [y0, y1] [m]."""
  c = (np.arange(RES) + 0.5 - RES / 2) / PX_PER_M
  x, y = np.meshgrid(c, -c)  # image rows go down, world y up
  return (x > x0) & (x < x1) & (y > y0) & (y < y1)


def _slide_expected(dx: float) -> np.ndarray:
  """The statuses after the box slid by `dx` [m]: MISS off the floor, OCCLUDED where the floor was
  under the box and is not any more, VALID elsewhere."""
  floor = _rect(-1.0, 1.0, -1.0, 1.0)
  before = _rect(-0.5, -0.1, -0.2, 0.2)
  after = _rect(-0.5 + dx, -0.1 + dx, -0.2, 0.2)
  expected = np.where(floor, int(F.VALID), int(F.MISS))
  expected[before & ~after] = int(F.OCCLUDED)
  return expected


@pytest.mark.parametrize("shaded", [False, True])
def test_uncovered_floor_is_occluded(scene: MakeScene, shaded: bool) -> None:
  """UNKNOWN on the first frame, VALID / MISS when still; after the box slides 0.3 m (6 px), the
  floor it uncovers is OCCLUDED, everything else VALID or MISS, and the box's motion is -6 px
  (geometry-only and shaded bodies)."""
  outputs = rw.Outputs(rgb=rw.settings.ColorOutput(), motion=_STATUS) if shaded else rw.Outputs(motion=_STATUS)
  s = scene(_SLIDE, cam_res=(RES, RES), outputs=outputs).render()
  status = s.out.flow_status
  assert status is not None and status.shape == (1, 1, RES, RES) and status.dtype == wp.uint8
  assert (status.numpy() == F.UNKNOWN).all()
  s.render()
  np.testing.assert_array_equal(status.numpy()[0, 0], _slide_expected(0.0))
  _move(s, 0.3)
  s.render()
  np.testing.assert_array_equal(status.numpy()[0, 0], _slide_expected(0.3))
  mv = _host(s.out.motion)[0, 0]
  box = _rect(-0.2, 0.2, -0.2, 0.2)
  np.testing.assert_allclose(mv[box][:, 0], -6.0, atol=1e-3)
  np.testing.assert_allclose(mv[box][:, 1], 0.0, atol=1e-3)
  s.r.reset_history()
  s.render()
  assert (status.numpy() == F.UNKNOWN).all()


def test_status_costs_nothing_unless_requested(scene: MakeScene) -> None:
  """Motion without status: no status buffer, no previous-depth slots; with it: two float16
  slots per pixel."""
  plain = scene(_SLIDE, cam_res=(RES, RES), outputs=rw.Outputs.of("motion"))
  assert plain.out.flow_status is None and not plain.r._buffers.layout.has("flow_status")
  history = plain.r._history  # the allocation under test
  assert history is not None and history.struct.gz.shape == (1, 1, 1)
  status = scene(_SLIDE, cam_res=(RES, RES), outputs=rw.Outputs(motion=_STATUS))
  history = status.r._history
  assert history is not None and history.struct.gz.shape == (1, 2, RES * RES)


# A pinhole camera on a free body looking 15 degrees down at a large plane (the sky above it).
_DOLLY = """
<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="40 40 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="rig" pos="0 -2 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <camera name="eye" xyaxes="1 0 0 0 0.258819 0.965926" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""
_W, _H = 40, 30


def _expected_status(s: MjScene, position: np.ndarray, seg: np.ndarray, cam: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
  """The statuses of the hit points `position` (seg -1: miss) through the previous camera `cam`
  (position, rotation): BEHIND its near plane, OUT_OF_FRAME of its image or far plane, VALID."""
  pos, rot = cam
  znear, zfar = s.r.clips[0]
  pc = (position - pos) @ rot  # camera frame: R^T (p - pos)
  z = -pc[..., 2]
  f = (_H / 2) / np.tan(np.radians(float(s.mjm.cam_fovy[0])) / 2)
  with np.errstate(divide="ignore", invalid="ignore"):
    u = _W / 2 + f * pc[..., 0] / z
    v = _H / 2 - f * pc[..., 1] / z
  out = np.full(seg.shape, int(F.VALID))
  out[(u < 0) | (v < 0) | (u >= _W) | (v >= _H) | (zfar > 0) & (z > zfar)] = int(F.OUT_OF_FRAME)
  out[z <= znear] = int(F.BEHIND)
  out[seg == -1] = int(F.MISS)
  return out


def _camera_pose(s: MjScene) -> tuple[np.ndarray, np.ndarray]:
  """(position, rotation) of the scene camera at the host data's qpos."""
  mujoco.mj_forward(s.mjm, s.mjd)
  return s.mjd.cam_xpos[0].copy(), s.mjd.cam_xmat[0].reshape(3, 3).copy()


def test_camera_dolly_behind_and_out_of_frame(scene: MakeScene) -> None:
  """The camera moves 2 m back and 0.5 m up: the ground near it was behind the previous camera
  (BEHIND, no motion), the sides outside its image (OUT_OF_FRAME), the middle VALID, the sky MISS;
  each pixel as the previous camera's pinhole projection of its hit point says (a static plane
  occludes nothing)."""
  outputs = rw.Outputs(seg=rw.settings.IdOutput(), position=rw.settings.FloatOutput(), motion=_STATUS)
  s = scene(_DOLLY, cam_res=(_W, _H), outputs=outputs).render()
  prev = _camera_pose(s)
  qpos = s.mjd.qpos.copy()
  qpos[1] -= 2.0
  qpos[2] += 0.5
  s.mjd.qpos[:] = qpos
  s.d.qpos = wp.array(qpos[None].astype(np.float32), dtype=float)
  mjw.forward(s.m, s.d)
  s.render()
  status = _host(s.out.flow_status)[0, 0]
  expected = _expected_status(s, _host(s.out.position)[0, 0], _host(s.out.seg)[0, 0], prev)
  counts = {k: int((expected == k).sum()) for k in (F.VALID, F.BEHIND, F.OUT_OF_FRAME, F.MISS)}
  assert min(counts.values()) > 0, counts  # every case is on screen
  np.testing.assert_array_equal(status, expected)
  mv = _host(s.out.motion)[0, 0]
  assert (mv[status == F.BEHIND] == 0.0).all() and (np.abs(mv[status == F.VALID]).max(axis=-1) > 0.0).all()


def test_motion_status_validation() -> None:
  """The status option is a bool (ValueError otherwise) and adds the status buffer to the layout."""
  with pytest.raises(ValueError, match=r"outputs\.motion\.status"):
    rw.settings.MotionOutput(status=1)  # pyrefly: ignore[bad-argument-type]  # wrong type on purpose
  assert rw.Outputs.parse({"motion": {"status": True}}).motion == _STATUS
