"""Ask for exactly the images a policy reads: channels-first color, radial depth, segmentation,
normals and albedo, each with its conventions, written in place every frame.

  python examples/outputs.py [--device cpu] [--worlds 4] [--size 96]
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.settings import ColorOutput, DepthOutput, NormalOutput

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" castshadow="true"/>
    <geom type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <geom name="cube" type="box" pos="0 0 0.15" size="0.15 0.15 0.15" rgba="0.2 0.5 0.9 1"/>
    <camera name="cam" pos="0 -1.5 1" xyaxes="1 0 0 0 0.55 0.83"/>
  </worldbody>
</mujoco>
"""

OUTPUTS = rw.Outputs.of(
  "seg",  # int32 geom id, -1 where nothing is hit
  "albedo",
  rgb=ColorOutput(channels=3, planar=True, dtype="float16"),  # (nworld * ncam, 3, H, W): a policy's input
  depth=DepthOutput(convention="radial", miss="inf"),  # distance along the ray [m]
  normal=NormalOutput(frame="geometric"),
)


def main(device: str | None, nworld: int, size: int) -> None:
  """Renders one frame with `OUTPUTS` and prints each array's shape and dtype."""
  with wp.ScopedDevice(device):
    mjm = mujoco.MjModel.from_xml_string(XML)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld)
    mjw.forward(m, d)
    settings = rw.RenderSettings().replace(rw.FAST).replace(outputs=OUTPUTS)
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m), nworld=nworld, cameras="cam", resolution=(size, size), settings=settings
    )
    out = r.render(d)  # the same arrays every frame; out.torch() gives zero-copy torch views
    for name in ("rgb", "depth", "seg", "normal", "albedo"):
      array = getattr(out, name)
      print(f"{name:7s} {array.dtype.__name__:10s} {tuple(array.shape)}")
    assert out.depth is not None  # requested (an output not requested is None)
    depth = out.depth.numpy()
    print(f"centre depth {depth[0, 0, size // 2, size // 2]:.3f} m, misses {np.isinf(depth).mean():.0%}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--device", help="Warp device (default: Warp's)")
  p.add_argument("--worlds", type=int, default=4)
  p.add_argument("--size", type=int, default=96, help="image width and height [px]")
  a = p.parse_args()
  main(a.device, a.worlds, a.size)
