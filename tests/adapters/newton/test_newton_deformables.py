"""rw.integrations.newton height fields and particle meshes (cloth, soft bodies): geometry, live
updates."""

import numpy as np
import pytest
import warp as wp
from conftest import required

newton = pytest.importorskip("newton")

import raywarp as rw
import raywarp.integrations.newton as newton_pbr
from raywarp.core.pipeline.gpu_scene.deformables import CLOSED_SURFACE, THICK_SURFACE
from raywarp.integrations.newton.deformables import particle_meshes
from raywarp.integrations.newton.types import NewtonState
from raywarp.scene.model.report import LoadReport

HEIGHT = 3.0  # the camera's height, looking straight down (-Z)


def _depth(model: newton.Model, state: NewtonState, nworld: int = 1) -> np.ndarray:
  """(nworld, 9, 9) planar depth of a camera at (0, 0, HEIGHT) looking down."""
  scene = newton_pbr.scene(model)
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"), performance=rw.settings.Performance(traversal="software"))
  r = rw.Renderer(scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(9, 9), settings=settings)
  r.camera_pose.assign(np.tile(np.array([0.0, 0.0, HEIGHT, 0.0, 0.0, 0.0, 1.0], np.float32), (nworld, 1, 1)))
  r.render(state)
  return required(r.outputs.depth).numpy().reshape(nworld, 9, 9)


def test_height_field_renders_its_surface():
  builder = newton.ModelBuilder()
  x = np.linspace(-1.0, 1.0, 21)
  data = np.tile(0.5 * (x + 1.0), (21, 1))  # rises along +x: z = 0.5 (x + 1) on [-1, 1]
  field = newton.Heightfield(data, nrow=21, ncol=21, hx=1.0, hy=1.0, min_z=0.0, max_z=1.0)
  builder.add_shape_heightfield(heightfield=field)
  model = builder.finalize(device="cpu")
  depth = _depth(model, model.state())[0]
  assert depth[4, 4] == pytest.approx(HEIGHT - 0.5, abs=1e-3)
  assert depth[4, 6] < depth[4, 4] < depth[4, 2]  # image +x is world +x: higher ground, nearer


def test_worlds_of_different_height_fields_each_show_their_own():
  """Each world's own terrain (a ramp over another height range per world): the fields are pooled
  (one instance each, hidden in the other worlds), equal fields share one."""
  ramp = np.tile(np.linspace(0.0, 1.0, 5), (5, 1))
  scene = newton.ModelBuilder()
  for low in (0.0, 0.4, 0.0):
    world = newton.ModelBuilder()
    field = newton.Heightfield(ramp, nrow=5, ncol=5, hx=1.0, hy=1.0, min_z=low, max_z=low + 1.0)
    world.add_shape_heightfield(heightfield=field)
    scene.add_world(world)
  model = scene.finalize(device="cpu")
  assert newton_pbr.scene(model).ninstance == 2
  depth = _depth(model, model.state(), nworld=3)
  np.testing.assert_allclose(depth[:, 4, 4], [HEIGHT - 0.5, HEIGHT - 0.9, HEIGHT - 0.5], atol=1e-3)


def _cloth(builder: newton.ModelBuilder, z: float) -> None:
  """A 1 m horizontal cloth at height z, particle radius 0.01 m."""
  builder.add_cloth_grid(
    pos=wp.vec3(-0.5, -0.5, z),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=4,
    dim_y=4,
    cell_x=0.25,
    cell_y=0.25,
    mass=0.1,
    particle_radius=0.01,
  )


def test_cloth_is_a_thick_surface_following_the_particles():
  builder = newton.ModelBuilder()
  _cloth(builder, 1.0)
  model = builder.finalize(device="cpu")
  meshes = required(particle_meshes(model, LoadReport(), merge_worlds=False))
  assert [p.dim for p in meshes.parts] == [THICK_SURFACE]
  assert meshes.parts[0].radius == pytest.approx(0.01)
  state = model.state()
  assert _depth(model, state)[0, 4, 4] == pytest.approx(HEIGHT - 1.0 - 0.01, abs=1e-3)
  q = state.particle_q.numpy()
  q[:, 2] += 0.5  # a simulation step lifting the cloth
  state.particle_q.assign(q)
  assert _depth(model, state)[0, 4, 4] == pytest.approx(HEIGHT - 1.5 - 0.01, abs=1e-3)


def _two_worlds() -> tuple[newton.Model, newton.State]:
  """Two worlds of one cloth each, world 1's lifted by 0.5 m."""
  item = newton.ModelBuilder()
  _cloth(item, 1.0)
  builder = newton.ModelBuilder()
  builder.replicate(item, 2, spacing=(0.0, 0.0, 0.0))
  model = builder.finalize(device="cpu")
  state = model.state()
  q = state.particle_q.numpy()
  q[model.particle_world.numpy() == 1, 2] += 0.5
  state.particle_q.assign(q)
  return model, state


def test_each_render_world_shows_its_newton_worlds_particles():
  """Render world v shows Newton world v, world 0 beyond them (the rigid instances' rule), or the
  world StateView.world_indices names."""
  model, state = _two_worlds()
  depth = _depth(model, state, nworld=3)[:, 4, 4]
  np.testing.assert_allclose(depth, [HEIGHT - 1.01, HEIGHT - 1.51, HEIGHT - 1.01], atol=1e-3)
  view = newton_pbr.StateView(state, world_indices=wp.array([1, 1, 0], dtype=wp.int32, device="cpu"))
  depth = _depth(model, view, nworld=3)[:, 4, 4]
  np.testing.assert_allclose(depth, [HEIGHT - 1.51, HEIGHT - 1.51, HEIGHT - 1.01], atol=1e-3)


def test_worlds_with_other_cloth_triangles_raise():
  builder = newton.ModelBuilder()
  for flip in (False, True):
    builder.begin_world()
    builder.add_cloth_grid(
      pos=wp.vec3(0.0, 0.0, 1.0),
      rot=wp.quat_identity(),
      vel=wp.vec3(0.0, 0.0, 0.0),
      dim_x=2,
      dim_y=2,
      cell_x=0.5,
      cell_y=0.5,
      mass=0.1,
      reverse_winding=flip,
    )
    builder.end_world()
  model = builder.finalize(device="cpu")
  with pytest.raises(ValueError, match="differ from world 0"):
    particle_meshes(model, LoadReport(), merge_worlds=False)


def test_a_soft_body_skin_is_a_closed_surface_and_loose_particles_are_reported():
  builder = newton.ModelBuilder()
  builder.add_soft_grid(
    pos=wp.vec3(-0.2, -0.2, 0.5),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=2,
    dim_y=2,
    dim_z=2,
    cell_x=0.2,
    cell_y=0.2,
    cell_z=0.2,
    density=100.0,
    k_mu=1000.0,
    k_lambda=1000.0,
    k_damp=0.0,
  )
  builder.add_particle(wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
  model = builder.finalize(device="cpu")
  meshes = required(particle_meshes(model, LoadReport(), merge_worlds=False))
  assert [p.dim for p in meshes.parts] == [CLOSED_SURFACE]
  report = newton_pbr.load(model).report
  assert report.dropped["newton.particles(unrendered)"] == 2  # the grid's centre particle and the loose one
  assert _depth(model, model.state())[0, 4, 4] == pytest.approx(HEIGHT - 0.9, abs=1e-3)  # the top at 0.5 + 0.4
