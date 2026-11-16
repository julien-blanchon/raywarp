"""Scene tour: several scenes one after another, the camera swinging around each one's view. One
Renderer per scene (a scene switch is a new Renderer: the kernels are shared, nothing compiles).
Any mix of benchmark scenes, `tests/data/scenes/` fixtures and USD / glTF / `.rayw` / MJCF files.

  python examples/demos/scene_tour.py --scenes sponza,showcase,open_plan --out scene_tour.mp4
  python examples/demos/scene_tour.py --scenes lights_area,ies_fixtures,many_lights --out lights.mp4
"""

from __future__ import annotations

import _common
import warp as wp

import raywarp as rw

QUALITY = {"aa.spp": 2, "shadows.samples": 2, "shadows.max_casters": 16}


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "scene_tour.mp4",
  scenes: str = "sponza,showcase,open_plan,cornell_rect",
) -> None:
  """Records `frames` frames split evenly between `scenes` (comma separated)."""
  names = scenes.split(",")
  resolution, cols = _common.tile(size, worlds)
  settings = rw.RenderSettings().replace(QUALITY)  # the quality recipe (docs/settings.md)
  with wp.ScopedDevice(device):
    rec = None
    for k, name in enumerate(names):
      stage = _common.load_scene(name, worlds)
      with _common.renderer(stage, worlds, resolution, settings) as r:
        rgb = _common.rgb_of(r)
        rec = rec or _common.record(out, rgb, cols)
        span = range(k * frames // len(names), (k + 1) * frames // len(names))
        for f in span:
          _common.pose_cameras(r, _common.orbit_pose(stage.view, f - span.start, len(span), 50.0))
          stage.step()
          r.render(stage.sim)
          rec.write(rgb)
    if rec is not None:
      print(f"{out}: {', '.join(names)}, {rec.close()}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
