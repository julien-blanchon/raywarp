"""Sensor effects, per world: clean, shot + read noise (a small, dim sensor), vignette + a contrasty
response curve, and all of them; halfway through, rolling shutter and motion blur switch on while
the camera whips around (run-time settings: nothing recompiles).

  python examples/demos/sensor_effects.py --worlds 4 --size 1280x720 --out sensor_effects.mp4
"""

from __future__ import annotations

import _common
import numpy as np
import warp as wp

import raywarp as rw

# Per variant: noise (full_well, read_noise [e-]), vignette (a1, a2, a3), response (toe, shoulder,
# centre, gamma).
CLEAN = ((1e9, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.5, 1.0))
NOISY = ((300.0, 6.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.5, 1.0))
LOOK = ((1e9, 0.0), (-0.5, 0.1, 0.0), (1.6, 1.6, 0.5, 1.0))
VARIANTS = (CLEAN, NOISY, LOOK, (NOISY[0], LOOK[1], LOOK[2]))


def main(
  device: str = "cuda",
  worlds: int = 4,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "sensor_effects.mp4",
  scene: str = "showcase",
) -> None:
  """Records world w with variant w % 4; rolling shutter and motion blur in the second half."""
  per_world = rw.PerWorld(env=["sensor.noise", "sensor.vignette", "sensor.response"])
  # Rolling shutter and motion blur resample along the motion vectors: request them from the start.
  settings = rw.RenderSettings(outputs=rw.Outputs.of("rgb", "motion")).replace({"sensor.noise": True})
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    resolution, cols = _common.tile(size, worlds)
    r = _common.renderer(stage, worlds, resolution, settings, per_world)
    for k, field in enumerate(("sensor.noise", "sensor.vignette", "sensor.response")):
      r.per_world[field].assign(np.array([VARIANTS[w % 4][k] for w in range(worlds)], np.float32))
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        if f == frames // 2:
          r.update_settings(**{"sensor.rolling_shutter": 0.8, "sensor.motion_blur": 0.5})
        fast = f >= frames // 2
        swing = _common.orbit_pose(stage.view, 4 * f if fast else f, frames, 60.0 if fast else 20.0)
        _common.pose_cameras(r, swing)
        stage.step()
        r.render(stage.sim, seed=f)  # new noise every frame
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
