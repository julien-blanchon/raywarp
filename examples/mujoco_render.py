"""Render a MuJoCo Warp simulation: 8 worlds x 2 cameras of RGB, depth and segmentation per call.

python examples/mujoco_render.py [--device cpu] [--worlds 8] [--size 128]
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
  <worldbody>
    <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="ball" pos="0 0 0.6"><freejoint/><geom type="sphere" size="0.2" rgba="0.8 0.3 0.2 1"/></body>
    <geom name="crate" type="box" pos="0.6 0.2 0.2" size="0.2 0.2 0.2" rgba="0.2 0.4 0.9 1"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="side" pos="2 0 1" xyaxes="0 1 0 -0.4 0 0.9"/>
  </worldbody>
</mujoco>
"""


def main(device: str | None, nworld: int, size: int) -> None:
  """Steps the simulation and renders every world from both cameras."""
  with wp.ScopedDevice(device):
    mjm = mujoco.MjModel.from_xml_string(XML)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m),  # model tables aliased: no copy
      nworld=nworld,
      cameras=["front", "side"],
      resolution=(size, size),
      settings=rw.RenderSettings(outputs=rw.Outputs.of("rgb", "depth", "seg")),
    )
    print(r.explain())  # which prebuilt build serves these settings, and the frame's cost class
    for _ in range(3):
      mjw.step(m, d)
      r.render(d)  # poses from d; never syncs the host
    out = r.outputs  # the same arrays every frame (None: an output not requested)
    assert out.rgb is not None and out.depth is not None and out.seg is not None
    rgb, depth, seg = out.rgb.numpy(), out.depth.numpy(), out.seg.numpy()  # host copies, to look at
    print(f"rgb {rgb.shape} mean {rgb[..., :3].mean():.1f}, depth max {depth.max():.2f} m, geoms seen {np.unique(seg)}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--device", help="Warp device (default: Warp's, cuda:0 on a GPU machine)")
  p.add_argument("--worlds", type=int, default=8)
  p.add_argument("--size", type=int, default=128, help="image width and height [px]")
  a = p.parse_args()
  main(a.device, a.worlds, a.size)
