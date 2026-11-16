"""Domain randomisation: every world its own colours, roughness, light and camera field of view,
written into per-world tables between frames (no recompile, no rebuild).

  python examples/domain_randomisation.py [--device cpu] [--worlds 16] [--size 64] [--steps 5]
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco

XML = """
<mujoco>
  <asset>
    <material name="table" rgba="0.6 0.5 0.4 1"/>
    <material name="cube" rgba="0.8 0.1 0.1 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.5 -0.5 2" dir="-0.2 0.2 -1" castshadow="true"/>
    <geom name="table" type="box" size="0.6 0.4 0.02" pos="0 0 0.5" material="table"/>
    <body name="cube" pos="0.1 0 0.56"><freejoint/><geom type="box" size="0.04 0.04 0.04" material="cube"/></body>
    <geom name="distractor" type="sphere" size="0.05" pos="0.2 0.2 0.57" rgba="0.9 0.9 0.1 1"/>
    <camera name="cam" pos="0 -1.0 1.1" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
  </worldbody>
</mujoco>
"""

# What varies per world, declared once: the tables are allocated at construction.
PER_WORLD = rw.PerWorld(
  material=["base_color", "specular_roughness"],
  light=["strength", "color"],
  camera=["fovy"],
  visible=["distractor"],
)


def randomise(r: rw.Renderer, rng: np.random.Generator) -> None:
  """New values in every world (on reset, or every step)."""
  color = r.per_world["material.base_color"]  # (nworld, nmaterial) RGBA: RGB tints
  tint = color.numpy()
  tint[..., :3] = rng.uniform(0.05, 0.95, tint[..., :3].shape)
  color.assign(tint)
  roughness = r.per_world["material.specular_roughness"]
  roughness.assign(rng.uniform(0.1, 1.0, roughness.shape).astype(np.float32))
  strength = r.per_world["light.strength"]
  strength.assign(strength.numpy() * rng.uniform(0.5, 1.5, strength.shape).astype(np.float32))
  r.per_world["light.color"].assign(rng.uniform(0.7, 1.0, (*strength.shape, 3)).astype(np.float32))
  fovy = r.per_world["camera.fovy"]
  fovy.assign(rng.uniform(40.0, 60.0, fovy.shape).astype(np.float32))
  shown = r.per_world.columns("instance.visible", "distractor")
  shown.assign(rng.integers(0, 2, shown.shape).astype(np.int32))


def main(device: str | None, nworld: int, size: int, steps: int) -> None:
  """Randomises and renders `steps` frames; the kernels never change."""
  with wp.ScopedDevice(device):
    mjm = mujoco.MjModel.from_xml_string(XML)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    settings = rw.RenderSettings().replace(rw.FAST).replace(outputs=rw.Outputs.of("rgb"))
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m, d),
      nworld=nworld,
      cameras="cam",
      resolution=(size, size),
      settings=settings,
      per_world=PER_WORLD,
    )
    rng = np.random.default_rng(0)
    features = r.features
    for _ in range(steps):
      randomise(r, rng)
      mjw.step(m, d)
      r.render(d)
    assert r.features is features  # nothing recompiled
    assert r.outputs.rgb is not None
    rgb = r.outputs.rgb.numpy()  # (nworld, 1, H, W, 4): each world its own look
    means = rgb[:, 0, ..., :3].reshape(nworld, -1).mean(axis=1)
    print(f"{nworld} worlds, mean intensity per world: {means.round(1)}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--device", help="Warp device (default: Warp's)")
  p.add_argument("--worlds", type=int, default=16)
  p.add_argument("--size", type=int, default=64, help="image width and height [px]")
  p.add_argument("--steps", type=int, default=5)
  a = p.parse_args()
  main(a.device, a.worlds, a.size, a.steps)
