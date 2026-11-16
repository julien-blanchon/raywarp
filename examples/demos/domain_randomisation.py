"""Domain randomisation: every world its own material colours and roughness, light strength and
colour, camera position and field of view, re-drawn every `every` frames. The writes go to
per-world tables in place: nothing recompiles, nothing is rebuilt.

  python examples/demos/domain_randomisation.py --worlds 16 --out domain_randomisation.mp4
"""

from __future__ import annotations

import _common
import numpy as np
import warp as wp

import raywarp as rw

PER_WORLD = rw.PerWorld(
  material=["base_color", "specular_roughness", "base_metalness"],
  light=["strength", "color"],
  camera=["fovy"],
)


def randomise(r: rw.Renderer, rng: np.random.Generator, view: _common.View) -> None:
  """New looks and camera poses in every world."""
  color = r.per_world["material.base_color"]  # (nworld, nmaterial, 4): RGB tints
  tint = color.numpy()
  tint[..., :3] = rng.uniform(0.05, 0.95, tint[..., :3].shape)
  color.assign(tint)
  for field, lo, hi in (("material.specular_roughness", 0.05, 1.0), ("material.base_metalness", 0.0, 0.6)):
    table = r.per_world[field]
    table.assign(rng.uniform(lo, hi, table.shape).astype(np.float32))
  strength = r.per_world["light.strength"]
  strength.assign((strength.numpy() * rng.uniform(0.4, 1.6, strength.shape)).astype(np.float32))
  r.per_world["light.color"].assign(rng.uniform(0.6, 1.0, (*strength.shape, 3)).astype(np.float32))
  fovy = r.per_world["camera.fovy"]
  fovy.assign((view[2] * rng.uniform(0.8, 1.2, fovy.shape)).astype(np.float32))
  eye, target = np.asarray(view[0]), np.asarray(view[1])
  shake = rng.normal(0.0, 0.05 * np.linalg.norm(eye - target), (r.nworld, 3))
  poses = [_common.look_at(eye + s, target) for s in shake]
  r.camera_pose.assign(np.stack(poses)[:, None, :])


def main(
  device: str = "cuda",
  worlds: int = 16,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "domain_randomisation.mp4",
  scene: str = "franka",
  every: int = 15,
) -> None:
  """Records `worlds` worlds, re-randomised every `every` frames."""
  rng = np.random.default_rng(0)
  with wp.ScopedDevice(device):
    stage = _common.load_scene(scene, worlds)
    resolution, cols = _common.tile(size, worlds)
    r = _common.renderer(stage, worlds, resolution, per_world=PER_WORLD)
    features = r.features
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for f in range(frames):
        if f % every == 0:
          randomise(r, rng, stage.view)
        stage.step()
        r.render(stage.sim)
        rec.write(_common.rgb_of(r))
    assert r.features is features  # nothing recompiled
    print(f"{out}: {worlds} worlds, {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
