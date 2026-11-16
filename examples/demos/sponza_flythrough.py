"""Big-scene fly-through: Intel's New Sponza (CC BY 4.0, the `sponza` benchmark scene) along its
long axis, the camera drifting and looking around. Quality settings: 2 x 2 samples per pixel,
soft shadows, probe GI, glossy reflections.

  python examples/demos/sponza_flythrough.py --worlds 1 --size 1280x720 --out sponza_flythrough.mp4
"""

from __future__ import annotations

import math

import _common
import numpy as np
import warp as wp

import raywarp as rw
from raywarp.viewer.camera import bounds

QUALITY = {"aa.spp": 2, "shadows.samples": 2, "shadows.max_casters": 16}


def path(lo: np.ndarray, hi: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
  """(eye, target) at `t` in [0, 1]: along the bounds' longer horizontal axis at a quarter of its
  height, looking ahead and swaying left and right."""
  axis = 0 if hi[0] - lo[0] >= hi[1] - lo[1] else 1
  side = 1 - axis
  centre = (lo + hi) / 2
  eye = centre.copy()
  eye[axis] = lo[axis] + (0.15 + 0.7 * t) * (hi[axis] - lo[axis])
  eye[side] += 0.15 * (hi[side] - lo[side]) * math.sin(2 * math.pi * t)
  eye[2] = lo[2] + 0.22 * (hi[2] - lo[2])
  ahead = np.zeros(3)
  ahead[axis] = math.cos(0.5 * math.sin(3 * math.pi * t))
  ahead[side] = math.sin(0.5 * math.sin(3 * math.pi * t))
  ahead[2] = 0.15 + 0.1 * math.sin(2 * math.pi * t)
  return eye, eye + ahead


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "sponza_flythrough.mp4",
  scene: str = "sponza",
) -> None:
  """Records the fly-through (every world on the same path, shifted in time)."""
  file, _ = _common.scene_path(scene)
  host = _common.host_scene(file)
  box = bounds(host)
  if box is None:
    raise ValueError(f"{scene}: an empty scene")
  with wp.ScopedDevice(device):
    gpu = rw.upload(host)
    resolution, cols = _common.tile(size, worlds)
    r = rw.Renderer(
      gpu,
      nworld=worlds,
      cameras=[rw.CameraSpec(fovy=65.0)],
      resolution=resolution,
      settings=rw.RenderSettings().replace(QUALITY),
    )
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        ts = [(f / frames + w / worlds) % 1.0 for w in range(worlds)]
        poses = [_common.look_at(*path(box[0], box[1], t)) for t in ts]
        r.camera_pose.assign(np.stack(poses)[:, None, :])
        r.render()
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
