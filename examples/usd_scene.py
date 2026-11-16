"""Render a USD stage: a tiny stage written here, read once into a cached `.rayw`, then uploaded.

python examples/usd_scene.py [--device cpu] [--size 128] [--stage office.usd]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import warp as wp

import raywarp as rw
import raywarp.formats.usd

STAGE = """#usda 1.0
(
  upAxis = "Z"
  metersPerUnit = 1
)
def Xform "World"
{
  def Cube "floor"
  {
    double size = 2
    float3 xformOp:scale = (3, 3, 0.02)
    uniform token[] xformOpOrder = ["xformOp:scale"]
    color3f[] primvars:displayColor = [(0.7, 0.7, 0.7)]
  }
  def Sphere "ball"
  {
    double radius = 0.3
    double3 xformOp:translate = (0, 0, 0.3)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    color3f[] primvars:displayColor = [(0.8, 0.3, 0.2)]
  }
  def DomeLight "sky"
  {
    float inputs:intensity = 1000
  }
  def Camera "camera"
  {
    double3 xformOp:translate = (0, -3, 1.2)
    float3 xformOp:rotateXYZ = (70, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
  }
}
"""


def main(device: str | None, size: int, stage: Path | None) -> None:
  """Reads the stage (the demo one by default) and renders it from its own cameras."""
  with tempfile.TemporaryDirectory() as tmp, wp.ScopedDevice(device):
    if stage is None:
      stage = Path(tmp) / "demo.usda"
      stage.write_text(STAGE)
    scene = rw.formats.usd.scene(stage, cache=Path(tmp) / "cache")  # a second run of a real stage maps its .rayw
    print(scene.report)  # what was dropped or approximated, counts, timings
    r = rw.Renderer(scene, nworld=1, cameras=None, resolution=(size, size))  # None: the stage's cameras
    out = r.render()  # a static stage: no simulator state
    assert out.rgb is not None
    rgb = out.rgb.numpy()
    print(f"rgb {rgb.shape} mean {rgb[..., :3].mean():.1f}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--device", help="Warp device (default: Warp's)")
  p.add_argument("--size", type=int, default=128, help="image width and height [px]")
  p.add_argument("--stage", type=Path, help="a USD file (default: the demo stage)")
  a = p.parse_args()
  main(a.device, a.size, a.stage)
