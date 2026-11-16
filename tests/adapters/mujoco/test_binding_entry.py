"""The adapters' Bind entry (W28): a binding rebuilt from a host scene's source-id column."""

import dataclasses

import numpy as np
import pytest
import warp as wp
from conftest import required

import raywarp as rw
import raywarp.formats.usd

newton = pytest.importorskip("newton")

import raywarp.integrations.newton  # after the skip
from raywarp.integrations.newton.binding import NewtonBinding


def _model(worlds: int) -> "newton.Model":
  """A Newton model of `worlds` replicated worlds: a ground plane, a free box and a free sphere."""
  robot = newton.ModelBuilder()
  body = robot.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
  robot.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
  other = robot.add_body(xform=wp.transform((0.5, 0.0, 0.5), wp.quat_identity()))
  robot.add_shape_sphere(other, radius=0.2)
  builder = newton.ModelBuilder()
  builder.add_ground_plane()
  for _ in range(worlds):
    builder.add_world(robot)
  return builder.finalize(device="cpu")


def test_newton_binding_from_a_cached_scene(tmp_path):
  """`rw.integrations.newton.binding(scene, model)` of a scene saved to `.wpbr` and loaded back
  renders as the binding of `rw.integrations.newton.load(model)` before the cache (the source
  column holds each instance's shape)."""
  model = _model(3)
  host = rw.integrations.newton.load(model)
  gpu = rw.upload(host, model.device, binding=rw.integrations.newton.binding(host, model))
  path = tmp_path / "s.wpbr"
  host.save(path)
  loaded = rw.Scene.load(path)
  assert np.array_equal(loaded.instances.source, host.instances.source) and np.all(loaded.instances.source >= 0)
  rebound = rw.integrations.newton.binding(loaded, model)
  ref = gpu.binding
  assert isinstance(ref, NewtonBinding)
  np.testing.assert_array_equal(rebound.shape_table().shape0, ref.shape_table().shape0)
  np.testing.assert_array_equal(rebound.shape_table().rows, ref.shape_table().rows)
  state = model.state()
  outputs = rw.Outputs.of("depth", "seg")
  renders = []
  for scene in (gpu, rw.upload(loaded, model.device, binding=rebound)):
    r = rw.Renderer(
      scene, nworld=3, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(24, 16), settings=rw.RenderSettings(outputs=outputs)
    )
    pose = r.camera_pose.numpy()
    pose[:] = np.array([0.0, -3.0, 1.0, 0.7071068, 0.0, 0.0, 0.7071068], np.float32)
    r.camera_pose.assign(pose)
    r.render(state)
    renders.append((required(r.outputs.depth).numpy(), required(r.outputs.seg).numpy()))
  np.testing.assert_array_equal(renders[0][0], renders[1][0])
  np.testing.assert_array_equal(renders[0][1], renders[1][1])
  assert (renders[0][1] >= 0).mean() > 0.3  # the shapes are seen


def test_newton_binding_needs_shapes():
  """Instances without a Newton shape (source -1) cannot be rebound by shape."""
  builder = rw.SceneBuilder()
  builder.add_instance("sphere", size=(0.1, 0.1, 0.1))
  with pytest.raises(ValueError, match="without a Newton shape"):
    rw.integrations.newton.binding(builder.build(), _model(1))


def test_usd_binding_is_the_static_poses():
  """`rw.formats.usd.binding`: a stage is static; the binding serves the scene's poses."""
  builder = rw.SceneBuilder()
  builder.add_instance("sphere", size=(0.1, 0.1, 0.1), xform=np.array([[1, 0, 0, 1.0], [0, 1, 0, 2.0], [0, 0, 1, 3.0]]))
  b = rw.formats.usd.binding(builder.build())
  np.testing.assert_allclose(b.poses.pos, [[1.0, 2.0, 3.0]])
  assert dataclasses.is_dataclass(b.poses)
