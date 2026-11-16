"""Batch mask: `render(sim, active=ids)` renders only the listed worlds; the others keep their last
image (here: they freeze while the camera keeps moving). First every world, then the even worlds,
then a band of worlds sweeping across the batch. The frame time follows the active count.

  python examples/demos/batch_mask.py --worlds 16 --size 1280x720 --out batch_mask.mp4
"""

from __future__ import annotations

import _common
import numpy as np
import warp as wp


def active_worlds(f: int, frames: int, nworld: int) -> np.ndarray:
  """(nworld,) int32 world ids rendered at frame `f`, -1 for the skipped ones."""
  ids = np.arange(nworld, dtype=np.int32)
  phase = 3 * f // max(frames, 1)
  if phase == 0:
    return ids
  if phase == 1:
    return np.where(ids % 2 == 0, ids, -1).astype(np.int32)
  band = (f * nworld // max(frames // 3, 1)) % nworld
  return np.where(np.abs(ids - band) <= nworld // 8, ids, -1).astype(np.int32)


def main(
  device: str = "cuda",
  worlds: int = 16,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "batch_mask.mp4",
  scene: str = "franka",
) -> None:
  """Records the batch while the active set changes."""
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    resolution, cols = _common.tile(size, worlds)
    r = _common.renderer(stage, worlds, resolution)
    active = wp.zeros(worlds, dtype=wp.int32)  # one device array, written in place
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        active.assign(active_worlds(f, frames, worlds))
        _common.pose_cameras(r, _common.orbit_pose(stage.view, f, frames, 60.0))
        stage.step()
        r.render(stage.sim, active=active)
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
