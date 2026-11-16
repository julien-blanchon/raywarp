"""Newton rigid bodies: boxes, spheres, capsules, cylinders and Newton's bunny (a USD mesh) dropped
on the ground in every world, each world its own drop layout and colours (Newton's `shape_color`,
re-read per world every frame). Newton's XPBD solver; raywarp reads `state.body_q` on the device.

  python examples/demos/newton_falling_objects.py --worlds 16 --out newton_falling_objects.mp4
"""

from __future__ import annotations

import _common
import newton
import newton.examples
import newton.usd
import numpy as np
import warp as wp
from pxr import Usd

import raywarp as rw
import raywarp.integrations.newton

OBJECTS = 10
SUBSTEPS = 8


def pile() -> newton.ModelBuilder:
  """One world: OBJECTS bodies of mixed shapes (posed later, per world)."""
  stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
  bunny = newton.usd.get_mesh(stage.GetPrimAtPath("/root/bunny"))
  b = newton.ModelBuilder()
  for k in range(OBJECTS):
    body = b.add_body(xform=wp.transform((0.0, 0.0, 1.0 + k), wp.quat_identity()))
    kind = k % 5
    if kind == 0:
      b.add_shape_box(body, hx=0.15, hy=0.1, hz=0.08)
    elif kind == 1:
      b.add_shape_sphere(body, radius=0.12)
    elif kind == 2:
      b.add_shape_capsule(body, radius=0.06, half_height=0.12)
    elif kind == 3:
      b.add_shape_cylinder(body, radius=0.1, half_height=0.1)
    else:
      b.add_shape_mesh(body, mesh=bunny, scale=(0.25, 0.25, 0.25))
  return b


def scatter(state: newton.State, nworld: int, rng: np.random.Generator) -> None:
  """Random drop positions and orientations in every world."""
  q = state.body_q.numpy().reshape(nworld, OBJECTS, 7)
  q[..., :2] = rng.uniform(-0.35, 0.35, (nworld, OBJECTS, 2))
  q[..., 2] = 0.3 + 0.35 * np.arange(OBJECTS) + rng.uniform(0.0, 0.1, (nworld, OBJECTS))
  quat = rng.normal(size=(nworld, OBJECTS, 4))
  q[..., 3:] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
  state.body_q.assign(q.reshape(-1, 7))


def main(
  device: str = "cuda",
  worlds: int = 16,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "newton_falling_objects.mp4",
) -> None:
  """Simulates and records the drop in every world."""
  rng = np.random.default_rng(1)
  with wp.ScopedDevice(device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(pile(), worlds, spacing=(0.0, 0.0, 0.0))
    model = builder.finalize()
    colors = model.shape_color.numpy()
    colors[1:] = rng.uniform(0.1, 0.95, colors[1:].shape)  # shape 0: the ground
    model.shape_color.assign(colors)
    sky = rw.Lighting(envmap=rw.scene.sun_sky(40.0, 150.0, height=256), envmap_intensity=1.0)
    scene = rw.integrations.newton.scene(model, lighting=sky)
    resolution, cols = _common.tile(size, worlds)
    r = rw.Renderer(scene, nworld=worlds, cameras=[rw.CameraSpec(fovy=45.0)], resolution=resolution)
    _common.pose_cameras(r, _common.look_at((1.6, -1.6, 1.2), (0.0, 0.0, 0.25)))
    solver = newton.solvers.SolverXPBD(model, iterations=5)
    collide = newton.CollisionPipeline(model)
    contacts = collide.contacts()
    s0, s1, control = model.state(), model.state(), model.control()
    scatter(s0, worlds, rng)
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for _ in range(frames):
        for _ in range(SUBSTEPS):
          s0.clear_forces()
          collide.collide(s0, contacts)
          solver.step(s0, s1, control, contacts, 1.0 / (_common.FPS * SUBSTEPS))
          s0, s1 = s1, s0
        r.render(s0)
        rec.write(_common.rgb_of(r))
    print(f"{out}: {worlds} worlds x {OBJECTS} bodies, {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
