"""MuJoCo Warp robot, four cameras per world: the Franka Panda scene of MuJoCo Menagerie
(Apache-2.0, the `franka` benchmark scene), every hinge joint driven by a sine (each world its own
phase), seen from the front, the side, above and the wrist (a free camera following the last link).

  python examples/demos/robot_multicam.py --worlds 4 --size 1280x720 --out robot_multicam.mp4
"""

from __future__ import annotations

import math

import _common
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import raywarp as rw
from raywarp.cli.record import orbit


def hinge_targets(mjm: mujoco.MjModel, t: float, nworld: int) -> tuple[np.ndarray, np.ndarray]:
  """(qpos addresses, (nworld, n) angles) of the hinge joints at time `t`: a sine around the middle
  of each joint's range, world w a quarter period later than world w - 1."""
  hinges = np.flatnonzero(mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
  lo, hi = mjm.jnt_range[hinges].T
  limited = mjm.jnt_limited[hinges].astype(bool)
  mid = np.where(limited, (lo + hi) / 2, 0.0)
  amp = np.where(limited, 0.35 * (hi - lo), 0.6)
  phase = 2 * math.pi * (0.25 * np.arange(nworld)[:, None] + t) + np.arange(hinges.size)
  return mjm.jnt_qposadr[hinges], mid + amp * np.sin(phase)


def wrist_poses(stage: _common.Stage, body: int) -> np.ndarray:
  """(nworld, 7) poses of an over-the-wrist camera on `body`: behind and beside it, looking past it
  along its z axis (a host read of the poses)."""
  assert stage.sim is not None
  pos = stage.sim.xpos.numpy()[:, body]
  axes = stage.sim.xmat.numpy()[:, body].reshape(-1, 3, 3)
  x, z = axes[:, :, 0], axes[:, :, 2]
  return np.stack([_common.look_at(p - 0.25 * c + 0.12 * a, p + 0.3 * c) for p, a, c in zip(pos, x, z, strict=True)])


def fixed_poses(view: _common.View) -> list[np.ndarray]:
  """Front (the scene's view), side (turned 90 degrees about the target) and top poses."""
  target = np.asarray(view[1])
  reach = float(np.linalg.norm(np.subtract(view[0], target)))
  top = _common.look_at(target + np.array([0.0, -0.05 * reach, reach]), target)
  return [_common.look_at(view[0], target), orbit(view, 90.0), top]


def main(
  device: str = "cuda",
  worlds: int = 4,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "robot_multicam.mp4",
  scene: str = "franka",
) -> None:
  """Records every world's four cameras, tiled world by world."""
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    assert stage.mjm is not None and stage.m is not None and stage.sim is not None, "an MJCF scene"
    resolution, cols = _common.tile(size, 4 * worlds, cols=4 if worlds > 1 else 2)  # a row per world
    cams = [rw.CameraSpec(fovy=stage.view[2])] * 3 + [rw.CameraSpec(fovy=90.0)]
    r = _common.renderer(stage, worlds, resolution, cameras=cams)
    hinges = np.flatnonzero(stage.mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    wrist = int(stage.mjm.jnt_bodyid[hinges[-1]])
    poses = np.zeros((worlds, 4, 7), np.float32)
    poses[:, :3] = fixed_poses(stage.view)
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        adr, angles = hinge_targets(stage.mjm, f / frames, worlds)
        qpos = stage.sim.qpos.numpy()
        qpos[:, adr] = angles
        stage.sim.qpos.assign(qpos)
        mjw.forward(stage.m, stage.sim)  # kinematics: the joints follow the sines exactly
        poses[:, 3] = wrist_poses(stage, wrist)
        r.camera_pose.assign(poses)
        r.render(stage.sim)
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
