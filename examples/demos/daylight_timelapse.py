"""Lighting time-lapse: a day in a few seconds. A bank of physical sun-and-sky maps (sunrise to
sunset) is uploaded once; each frame picks one per world (`env.id`). Every map's sun is a shadowed
directional light, so the shadows sweep with it, and probe GI follows.

  python examples/demos/daylight_timelapse.py --size 1280x720 --out daylight_timelapse.mp4
"""

from __future__ import annotations

import math

import _common
import numpy as np
import warp as wp

import raywarp as rw

MAPS = 36


def day(n: int, height: int) -> list[np.ndarray]:
  """`n` sun-and-sky maps from sunrise (east, low) over noon (south, high) to sunset (west)."""
  maps = []
  for k in range(n):
    t = k / max(n - 1, 1)
    elevation = 3.0 + 57.0 * math.sin(math.pi * t)
    maps.append(rw.scene.sun_sky(elevation, azimuth=-60.0 + 300.0 * t, height=height))
  return maps


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "daylight_timelapse.mp4",
  scene: str = "sponza",
) -> None:
  """Records the day, each world shifted in time."""
  n = min(MAPS, frames)
  with wp.ScopedDevice(device):
    sky = rw.Lighting(envmap=day(n, 256), envmap_intensity=1.0, replace=True)  # nits; the scene's lights dropped
    stage = _common.load_scene(scene, worlds, lighting=sky)
    resolution, cols = _common.tile(size, worlds)
    settings = rw.RenderSettings().replace({"exposure.adaptation": "continuous"})  # the eye adapts over the day
    r = _common.renderer(stage, worlds, resolution, settings, rw.PerWorld(env=["id"]))
    env = r.per_world["env.id"]  # (nworld,) int32: the map each world shows
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        t = (f / frames + np.arange(worlds) / worlds) % 1.0
        env.assign(np.minimum((t * n).astype(np.int32), n - 1))
        _common.pose_cameras(r, _common.orbit_pose(stage.view, f, frames, 20.0))
        r.render(stage.sim)
        rec.write(_common.rgb_of(r))
    print(f"{out}: {n} sky maps, {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
