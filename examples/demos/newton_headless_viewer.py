"""Newton's own examples, ray traced offscreen: `ViewerRaywarpHeadless` is a Newton viewer (the
`ViewerNull` API) that renders each frame with raywarp, with no window and no OpenGL. Here Newton's
`basic_shapes` example (XPBD: a sphere, ellipsoid, capsule, cylinder, box, the bunny mesh and a
cone dropped on the ground) runs unchanged; `get_frame()` feeds the recorder.

  python examples/demos/newton_headless_viewer.py --size 1280x720 --out newton_headless_viewer.mp4
"""

from __future__ import annotations

import argparse
import importlib

import _common
import warp as wp

import raywarp as rw
from raywarp.integrations.newton.headless import ViewerRaywarpHeadless


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "newton_headless_viewer.mp4",
  example: str = "basic.example_basic_shapes",
) -> None:
  """Runs a Newton example (`newton.examples.<example>`) for `frames` frames and records them
  (`worlds` is the example's world count where it has one)."""
  width, height = _common.parse_size(size)
  with wp.ScopedDevice(device):
    sky = rw.Lighting(envmap=rw.scene.sun_sky(35.0, 200.0, height=256), envmap_intensity=1.0)
    viewer = ViewerRaywarpHeadless(num_frames=frames, width=width, height=height, lighting=sky, **{"aa.spp": 2})
    module = importlib.import_module(f"newton.examples.{example}")
    demo = module.Example(viewer, argparse.Namespace(world_count=worlds, solver=None))  # Newton's example class
    canvas = _common.Canvas(width, height)
    with _common.record(out, canvas.rgb) as rec:
      while viewer.is_running():
        demo.step()
        demo.render()  # begin_frame / log_state / end_frame: the ray-traced frame
        canvas.show(viewer.get_frame().numpy())
        rec.write(canvas.rgb)
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
