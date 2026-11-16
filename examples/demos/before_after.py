"""Before / after: the same shot rendered with one setting off and on, side by side, plus one raw
clip of each (`<out>_before.mp4`, `<out>_after.mp4`) for a wipe in an editor. `--setting` picks a
preset (fxaa, gi, contact_ao, glass, noise) with its scene, or any `group.field` with `--before`
and `--after` values (Python literals).

  python examples/demos/before_after.py --setting gi --size 1280x360 --out before_after_gi.mp4
  python examples/demos/before_after.py --setting reflections.enabled --before False --after True
"""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path

import _common
import warp as wp

import raywarp as rw

# preset -> (field, before, after, scene)
PRESETS: dict[str, tuple[str, object, object, str]] = {
  "fxaa": ("aa.post", False, True, "franka"),
  "gi": ("gi.enabled", False, True, "gi_window"),
  "contact_ao": ("lighting.contact_ao", 0.0, 1.0, "franka"),
  "glass": ("shadows.translucent", "off", "transmittance", "glass"),
  "noise": ("sensor.noise", False, True, "showcase"),
}


def resolve(setting: str, before: str, after: str, scene: str) -> tuple[str, object, object, str]:
  """(field, before, after, scene) of a preset or of a `group.field` (`scene` "": the default)."""
  if setting in PRESETS:
    field, a, b, default = PRESETS[setting]
    return field, a, b, scene or default
  return setting, ast.literal_eval(before), ast.literal_eval(after), scene or "showcase"


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x360",
  frames: int = 240,
  out: str = "before_after.mp4",
  setting: str = "gi",
  before: str = "False",
  after: str = "True",
  scene: str = "",
) -> None:
  """Records world 0 with `field` = before (left) and after (right)."""
  field, a, b, scene = resolve(setting, before, after, scene)
  width, height = _common.parse_size(size)
  tile = (width // 2 // 2 * 2 - 2, height // 2 * 2)
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    pair = [_common.renderer(stage, worlds, tile, rw.RenderSettings().replace({field: v})) for v in (a, b)]
    side = _common.Canvas(width, height)
    stem = Path(out).with_suffix("")
    with contextlib.ExitStack() as stack:
      raw = [
        stack.enter_context(_common.record(f"{stem}_{k}.mp4", _common.rgb_of(r)))
        for k, r in zip(("before", "after"), pair, strict=True)
      ]
      both = stack.enter_context(_common.record(out, side.rgb))
      for f in range(frames):
        stage.step()
        images = []
        for r, rec in zip(pair, raw, strict=True):
          _common.pose_cameras(r, _common.orbit_pose(stage.view, f, frames, 30.0))
          r.render(stage.sim, seed=f)
          rec.write(_common.rgb_of(r))
          images.append(_common.rgb_of(r).numpy()[0, 0])
        side.show(_common.row(images))
        both.write(side.rgb)
    print(f"{out}: {field} = {a!r} | {b!r} on {scene}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
