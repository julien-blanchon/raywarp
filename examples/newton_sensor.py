"""Newton's camera sensor, PBR shaded: `raywarp.integrations.newton.sensor.SensorCamera` is a
drop-in for `newton.sensors.SensorCamera` (same constructor and `update()`).

  python examples/newton_sensor.py [--device cpu] [--worlds 4] [--size 96]
"""

from __future__ import annotations

import argparse

import newton
import warp as wp

import raywarp as rw
from raywarp.integrations.newton.sensor import SensorCamera  # instead of: from newton.sensors import SensorCamera


def main(device: str | None, nworld: int, size: int) -> None:
  """Builds a replicated Newton model and renders color and depth for every world."""
  with wp.ScopedDevice(device):
    world = newton.ModelBuilder()
    body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(world, nworld)
    model = builder.finalize()
    state = model.state()

    camera = SensorCamera(model, lighting=rw.Lighting.studio())  # PBR extras: lighting, settings fields
    rays = SensorCamera.compute_camera_rays_pinhole(size, size, camera_fov=1.0)
    eye = wp.transformf((0.0, -3.0, 1.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.25))
    views = wp.array([eye] * model.world_count, dtype=wp.transformf)  # one camera per world
    color = camera.allocate_image_color(model.world_count, size, size)
    depth = camera.allocate_image_depth(model.world_count, size, size)
    camera.update(state, views, rays, color_image=color, depth_image=depth)  # Newton's update()
    print(f"color {tuple(color.shape)}, depth {tuple(depth.shape)}, max depth {depth.numpy().max():.2f}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--device", help="Warp device (default: Warp's)")
  p.add_argument("--worlds", type=int, default=4)
  p.add_argument("--size", type=int, default=96, help="image width and height [px]")
  a = p.parse_args()
  main(a.device, a.worlds, a.size)
