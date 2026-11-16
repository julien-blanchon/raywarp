"""raywarp.integrations.mujoco.mjx: a JAX state rendered by a MuJoCo Warp scene, the outputs as JAX
arrays."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

jax = pytest.importorskip("jax")

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.integrations.mujoco import mjx

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1"/>
    <body pos="0 0 0.5"><freejoint/><geom type="sphere" size="0.3"/></body>
  </worldbody>
</mujoco>
"""


class _State:
  """An MJX-like batched state: JAX arrays of the pose fields."""

  def __init__(self, d: mjw.Data) -> None:
    self.geom_xpos = jax.numpy.asarray(d.geom_xpos.numpy())
    self.geom_xmat = jax.numpy.asarray(d.geom_xmat.numpy())  # (nworld, ngeom, 3, 3)


def test_a_jax_state_renders_and_the_outputs_are_jax_views():
  """The sphere moved in a JAX state (world 1 only) shows in world 1's depth; the outputs come
  back as JAX arrays of the renderer's buffers; the renderer keeps reading the same arrays."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd, nworld=2)
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth", "rgb"))
  r = rw.Renderer(
    rw.integrations.mujoco.scene(mjm, m), nworld=2, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(9, 9), settings=settings
  )
  r.camera_pose.assign(np.tile(np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], np.float32), (2, 1, 1)))
  state = _State(d)
  state.geom_xpos = state.geom_xpos.at[1, 1, 2].set(1.5)  # world 1's sphere rises by 1 m
  pointer = d.geom_xpos.ptr
  mjx.feed(d, state)
  assert d.geom_xpos.ptr == pointer
  r.render(d)
  out = mjx.to_jax(r.outputs)
  assert set(out) == {"depth", "rgb"}
  depth = np.asarray(out["depth"]).reshape(2, 9, 9)
  np.testing.assert_allclose(depth[:, 4, 4], [3.0 - 0.8, 3.0 - 1.8], atol=1e-3)
  assert out["rgb"].dtype == jax.numpy.uint8


def test_a_state_without_geom_poses_or_of_another_size_raises():
  mjm = mujoco.MjModel.from_xml_string(XML)
  d = mjw.make_data(mjm, nworld=2)
  with pytest.raises(ValueError, match="no geom_xpos"):
    mjx.feed(d, object())
  state = _State(d)
  state.geom_xpos = state.geom_xpos[:1]
  with pytest.raises(ValueError, match="geom_xpos is"):
    mjx.feed(d, state)
  assert isinstance(d.geom_xpos, wp.array)
