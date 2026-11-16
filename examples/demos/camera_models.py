"""Camera models and lenses, four cameras on one pose: pinhole, orthographic, a 200 degree fisheye
and an OpenCV-calibrated lens whose radial distortion k1 sweeps from barrel to pincushion
(`PerWorld(camera=["distortion"])`: a per-world table, no recompile).

  python examples/demos/camera_models.py --size 1280x720 --out camera_models.mp4
"""

from __future__ import annotations

import math

import _common
import warp as wp

import raywarp as rw


def cameras(view: _common.View, extent: float, resolution: tuple[int, int]) -> list[rw.CameraSpec]:
  """Pinhole, orthographic (a view height of the scene's radius), fisheye, OpenCV lens."""
  w, h = resolution
  f = 0.5 * h / math.tan(math.radians(view[2]) / 2)
  k = [[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]]
  return [
    rw.CameraSpec(fovy=view[2]),
    rw.CameraSpec(projection="orthographic", fovy=extent),
    rw.CameraSpec(projection="fisheye", fovy=200.0),
    rw.CameraSpec.from_opencv(k, (w, h), distortion=(-0.3, 0.0, 0.0, 0.0)),
  ]


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "camera_models.mp4",
  scene: str = "showcase",
) -> None:
  """Records the four cameras of every world, tiled (2 x 2 for one world)."""
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    resolution, cols = _common.tile(size, worlds * 4)
    r = _common.renderer(
      stage,
      worlds,
      resolution,
      per_world=rw.PerWorld(camera=["distortion"]),
      cameras=cameras(stage.view, stage.extent, resolution),
    )
    distortion = r.per_world["camera.distortion"]  # (nworld, ncam, 12): OpenCV's coefficients
    coefficients = distortion.numpy()
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        coefficients[:, 3, 0] = 0.35 * math.sin(2 * math.pi * f / frames)  # camera 3's k1
        distortion.assign(coefficients)
        _common.pose_cameras(r, _common.orbit_pose(stage.view, f, frames, 40.0))
        stage.step()
        r.render(stage.sim)
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
