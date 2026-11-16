"""Newton + USD: a USD robot (Newton's cartpole asset) replicated into many worlds, simulated by
Newton and rendered with the stage's own materials and lights; every world starts from its own
pose and gets its own colours (per-world tables, no recompile).

  python examples/demos/newton_usd_robots.py --worlds 64 --size 1280x720 --out newton_usd_robots.mp4
"""

from __future__ import annotations

import _common
import newton
import newton.examples
import numpy as np
import warp as wp

import raywarp as rw
import raywarp.integrations.newton

SUBSTEPS = 20
ASSET = "cartpole.usda"
# eye, target, fovy: from the side of the 8 m rail (along y), the poles 1 m above it.
VIEW = ((2.0, -0.35, 1.3), (0.0, 0.0, 0.7), 50.0)


def build(nworld: int, rng: np.random.Generator) -> newton.Model:
  """`nworld` copies of the USD articulation on a ground plane, each from a random pole angle."""
  robot = newton.ModelBuilder()
  robot.default_joint_cfg.armature = 0.1
  robot.add_usd(newton.examples.get_asset(ASSET), enable_self_collisions=False, collapse_fixed_joints=True)
  builder = newton.ModelBuilder()
  builder.replicate(robot, nworld, spacing=(0.0, 0.0, 0.0))  # every world at the origin: one camera pose
  builder.add_ground_plane()
  model = builder.finalize()
  q = model.joint_q.numpy().reshape(nworld, -1)
  q[:, 0] = rng.uniform(-1.0, 1.0, nworld)  # the cart on its rail
  q[:, 1:] = rng.uniform(-0.6, 0.6, q[:, 1:].shape)  # the poles' angles
  model.joint_q.assign(q.reshape(-1))
  return model


def main(
  device: str = "cuda",
  worlds: int = 64,
  size: str = "1280x720",
  frames: int = 240,
  out: str = "newton_usd_robots.mp4",
) -> None:
  """Simulates and records `worlds` cartpoles, tiled."""
  rng = np.random.default_rng(0)
  with wp.ScopedDevice(device):
    model = build(worlds, rng)
    stage = newton.examples.get_asset(ASSET)
    sky = rw.Lighting(envmap=rw.scene.sun_sky(35.0, 120.0, height=256), envmap_intensity=1.0)  # nits
    host = rw.integrations.newton.load_usd(model, stage, lighting=sky)  # the stage's materials and lights
    scene = rw.upload(host, model.device, binding=rw.integrations.newton.binding(host, model, stage=stage))
    resolution, cols = _common.tile(size, worlds)
    r = rw.Renderer(
      scene,
      nworld=worlds,
      cameras=[rw.CameraSpec(fovy=VIEW[2])],
      resolution=resolution,
      per_world=rw.PerWorld(material=["base_color"]),
    )
    _common.pose_cameras(r, _common.look_at(VIEW[0], VIEW[1]))
    color = r.per_world["material.base_color"]  # (nworld, nmaterial, 4)
    tint = color.numpy()
    robot = np.arange(tint.shape[1]) != host.instances.material[-1]  # the ground plane, added last, stays
    tint[:, robot, :3] = rng.uniform(0.1, 0.9, tint[:, robot, :3].shape)
    color.assign(tint)

    solver = newton.solvers.SolverFeatherstone(model)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    with _common.record(out, _common.rgb_of(r), cols) as rec:
      for _ in range(frames):
        for _ in range(SUBSTEPS):
          solver.step(s0, s1, control, None, 1.0 / (_common.FPS * SUBSTEPS))
          s0, s1 = s1, s0
        r.render(s0)  # body_q -> instance poses on the device
        rec.write(_common.rgb_of(r))
    print(f"{out}: {worlds} worlds, {rec.stats}")


if __name__ == "__main__":
  _common.cli(main, __doc__)
