"""Parallel rendering: 64 to 1024 worlds of one scene in one launch, tiled on one video frame.

Each world's camera swings at its own phase, so every tile is a different view of the same scene.

  python examples/demos/parallel_worlds.py --worlds 256 --size 1280x720 --out parallel_worlds.mp4
"""

from __future__ import annotations

import _common
import numpy as np
import warp as wp


def main(
  device: str = "cuda",
  worlds: int = 256,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "parallel_worlds.mp4",
  scene: str = "franka",
) -> None:
  """Renders `worlds` worlds per frame, each world's camera on its own orbit phase."""
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    resolution, cols = _common.tile(size, worlds)
    r = _common.renderer(stage, worlds, resolution)
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        phases = [f + i * frames / worlds for i in range(worlds)]
        poses = np.stack([_common.orbit_pose(stage.view, round(p), frames, 90.0) for p in phases])
        r.camera_pose.assign(poses[:, None, :])
        stage.step()
        r.render(stage.sim)
        rec.write(_common.rgb_of(r))
    print(f"{out}: {worlds} worlds of {resolution[0]}x{resolution[1]}, {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
