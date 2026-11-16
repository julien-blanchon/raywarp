"""Material studio: OpenPBR spheres under a studio HDRI. Rows, bottom to top: dielectric roughness
0 -> 1, metal roughness 0 -> 1, a clear coat 0 -> 1 over a rough base, glass roughness 0 -> 0.5.
The environment turns (a per-world yaw) while the camera swings.

  python examples/demos/material_studio.py --worlds 1 --size 1280x720 --out material_studio.mp4
"""

from __future__ import annotations

import _common
import numpy as np
import warp as wp

import raywarp as rw

COLUMNS = 6
SPACING = 0.75
RADIUS = 0.3
HDRI = _common.FIXTURES / "textures" / "studio.hdr"


def material(row: int, t: float) -> rw.Material:
  """The material of row `row` at sweep position `t` in [0, 1]."""
  if row == 0:
    return rw.Material(base_color=(0.8, 0.35, 0.1), specular_roughness=t)
  if row == 1:
    return rw.Material(base_color=(0.95, 0.75, 0.4), base_metalness=1.0, specular_roughness=t)
  if row == 2:
    return rw.Material(base_color=(0.6, 0.05, 0.05), specular_roughness=0.6, coat_weight=t, coat_roughness=0.03)
  return rw.Material(base_color=(1.0, 1.0, 1.0), transmission_weight=1.0, specular_roughness=0.5 * t)


def build() -> rw.Scene:
  """The 4 x COLUMNS sphere wall on a floor, lit by the studio HDRI only."""
  b = rw.SceneBuilder()
  for row in range(4):
    for col in range(COLUMNS):
      mat = b.add_material(material(row, col / (COLUMNS - 1)))
      pos = [(col - (COLUMNS - 1) / 2) * SPACING, 0.0, RADIUS + 0.1 + row * SPACING]
      b.add_instance("sphere", material=mat, size=(RADIUS, 0, 0), xform=np.c_[np.eye(3), pos])
  floor = b.add_material(rw.Material(base_color=(0.35, 0.35, 0.38), specular_roughness=0.4))
  b.add_instance("box", material=floor, size=(8, 8, 0.05), xform=np.c_[np.eye(3), [0, 0, -0.05]])
  return b.build().with_lighting(rw.Lighting(envmap=str(HDRI), envmap_intensity=1.0, replace=True))


def main(
  device: str = "cuda",
  worlds: int = 1,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "material_studio.mp4",
) -> None:
  """Records the sphere wall, the environment turning once over the clip."""
  with wp.ScopedDevice(device):
    stage = _common.Stage(rw.upload(build()), ((0.0, -6.0, 2.0), (0.0, 0.0, 1.45), 40.0), 3.0)
    resolution, cols = _common.tile(size, worlds)
    r = _common.renderer(stage, worlds, resolution, per_world=rw.PerWorld(env=["yaw"]))
    yaw = r.per_world["env.yaw"]  # (nworld,) radians about +z
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        yaw.assign((2.0 * np.pi * (f / frames + np.arange(worlds) / worlds)).astype(np.float32))
        _common.pose_cameras(r, _common.orbit_pose(stage.view, f, frames, 40.0))
        r.render()
        rec.write(_common.rgb_of(r))
    print(f"{out}: {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
