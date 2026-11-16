"""`Renderer.per_world` is a Mapping of the declared tables: keys, item access, printing, the
columns of matched instances, and KeyError for an undeclared field."""

from __future__ import annotations

import collections.abc

import numpy as np
import pytest
from conftest import MakeScene

import raywarp as rw

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="box" type="box" size="0.3 0.3 0.3" pos="0 0 0.3" rgba="1 0 0 1"/>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""


def test_per_world_is_a_mapping_of_the_declared_tables(scene: MakeScene) -> None:
  decl = rw.PerWorld(light=["strength"], visible=["box"])
  s = scene(XML, nworld=2, cam_res=(8, 8), per_world=decl)
  view = s.r.per_world
  assert isinstance(view, collections.abc.Mapping)
  assert list(view) == ["instance.visible", "light.strength"] and len(view) == 2
  assert "light.strength" in view and "material.base_color" not in view
  strength = view["light.strength"]
  assert strength.shape == (2, 1)
  strength.assign(np.array([[0.5], [2.0]], np.float32))
  np.testing.assert_allclose(s.r.per_world["light.strength"].numpy(), [[0.5], [2.0]])
  assert "'light.strength': (2, 1)" in repr(view)
  box = view.columns("instance.visible", "box")
  assert box.shape == (2, 1)
  box.assign(np.array([[1], [0]], np.int32))
  s.render()
  with pytest.raises(KeyError, match="not declared"):
    view["material.base_color"]
  with pytest.raises(ValueError, match="pass instance patterns"):
    view.columns("light.strength")


def test_listing_with_per_world_sizes(scene: MakeScene) -> None:
  """A size declared for one instance: the view lists and prints its whole table."""
  s = scene(XML, nworld=2, cam_res=(8, 8), per_world=rw.PerWorld(size=["box"]))
  view = s.r.per_world
  assert "instance.size" in list(view)
  assert all(t.shape[0] == 2 for t in view.values())
  assert "instance.size" in repr(view)


def test_columns_take_every_world_values_without_the_world_axis(scene: MakeScene) -> None:
  """(ncolumn, 3) vec3 and (ncolumn, 3, 3) / (ncolumn, 9) mat33 values are every world's (NumPy
  broadcasting; the vector / matrix axes were taken for the world axes: a reshape error)."""
  s = scene(XML, nworld=2, cam_res=(8, 8), per_world=rw.PerWorld(size=["box"]))
  size = s.r.per_world.columns("instance.size", "box")
  size.assign(np.array([[0.1, 0.2, 0.3]], np.float32))
  np.testing.assert_allclose(size.numpy(), [[[0.1, 0.2, 0.3]]] * 2)
  b = rw.SceneBuilder()
  b.add_instance("box", size=(0.3, 0.3, 0.3), path="box")
  host = rw.upload(b.build(), device="cpu")
  r = rw.Renderer(host, nworld=2, cameras=[rw.CameraSpec()], resolution=(8, 8), per_world=rw.PerWorld(pose=["box"]))
  rot = r.per_world.columns("instance.rot", "box")
  turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
  for values in (turn[None], turn.reshape(1, 9), np.stack([turn, turn])[:, None]):
    rot.assign(np.eye(3, dtype=np.float32)[None, None].repeat(2, 0))
    rot.assign(values)
    np.testing.assert_allclose(rot.numpy(), np.broadcast_to(turn, (2, 1, 3, 3)))
  r.close()
