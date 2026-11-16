"""Contact AO of the dynamic objects (`shade/ao/contact.py`): box / sphere proxies of the
per-world instances darken the static scenery's indirect light around them, deterministically; the
dynamic objects themselves and a scene without per-world instances are left as they are."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco

_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    {body}
    <camera name="cam" pos="0 -0.9 0.7" xyaxes="1 0 0 0 0.6 0.8" fovy="40"/>
  </worldbody>
</mujoco>
"""
_CUBE = (
  '<body name="cube" pos="0 0 0.1"><freejoint/><geom name="cube" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/></body>'
)
_STATIC = '<geom name="cube" type="box" size="0.1 0.1 0.1" pos="0 0 0.1" rgba="0.8 0.2 0.2 1"/>'
# A capsule lying on the floor: its own surface is inside its round proxy, never darkened by it.
_CAPSULE = '<body name="cube" pos="0 0 0.08" euler="0 90 20"><freejoint/><geom type="capsule" size="0.08 0.15"/></body>'
_SIZE = 64


def _render(body: str, ao: float, *, hidden: bool = False) -> tuple[np.ndarray, np.ndarray]:
  """(rgb, segmentation) of the floor and a cube under a uniform grey environment (indirect
  light only), at contact AO strength `ao`; `hidden`: the cube hidden in the world
  (`PerWorld.visible`)."""
  mjm = mujoco.MjModel.from_xml_string(_XML.format(body=body))
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=1)
  mjw.forward(m, d)
  sky = np.full((32, 64, 3), 0.18, np.float32)
  scene = rw.integrations.mujoco.scene(mjm, m, d, lighting=rw.Lighting(envmap=sky, replace=True))
  settings = rw.RenderSettings().replace(outputs=rw.Outputs.of("rgb", "seg"), **{"lighting.contact_ao": ao})
  per_world = rw.PerWorld(visible=["cube"]) if hidden else None
  with rw.Renderer(scene, nworld=1, cameras="cam", resolution=(_SIZE, _SIZE), settings=settings, per_world=per_world) as r:
    if hidden:
      r.per_world.columns("instance.visible", "cube").assign(np.zeros((1, 1), np.int32))
    r.render(d)
    assert r.outputs.rgb is not None and r.outputs.seg is not None
    return r.outputs.rgb.numpy()[0, 0, ..., :3].astype(np.float64), r.outputs.seg.numpy()[0, 0]


@pytest.fixture(scope="module")
def cube() -> dict[float, tuple[np.ndarray, np.ndarray]]:
  with wp.ScopedDevice("cpu"):
    return {ao: _render(_CUBE, ao) for ao in (0.0, 1.0)}


def test_floor_darkens_around_a_dynamic_cube_and_the_cube_is_left_as_it_is(cube):
  (off, seg), (on, _) = cube[0.0], cube[1.0]
  cube_id = int(seg[_SIZE // 2, _SIZE // 2])
  is_cube = seg == cube_id
  np.testing.assert_array_equal(on[is_cube], off[is_cube])
  floor = ~is_cube
  ratio = on.sum(axis=2)[floor] / np.maximum(off.sum(axis=2)[floor], 1e-6)
  assert ratio.max() <= 1.0 + 1e-9
  # Next to the cube the floor loses a good part of its indirect light, far away nothing.
  rows, cols = np.nonzero(is_cube)
  below = (rows.max() + 2, int(cols.mean()))
  assert on[below].sum() < 0.85 * off[below].sum()
  np.testing.assert_allclose(on[0, 0], off[0, 0], atol=1.0)


def test_contact_ao_is_deterministic(cube):
  with wp.ScopedDevice("cpu"):
    again, _ = _render(_CUBE, 1.0)
  np.testing.assert_array_equal(again, cube[1.0][0])


def test_static_scenery_gets_no_proxy_occlusion():
  """A cube that is not per world (shared, static) has no proxy: the GI's occlusion is its."""
  with wp.ScopedDevice("cpu"):
    off, _ = _render(_STATIC, 0.0)
    on, _ = _render(_STATIC, 1.0)
  np.testing.assert_array_equal(on, off)


def test_a_round_proxy_leaves_its_own_surface_alone():
  """A capsule's surface lies inside its round proxy (the true shape, not the area-equal box the
  form factor uses): it is not darkened, the floor next to it is."""
  with wp.ScopedDevice("cpu"):
    off, seg = _render(_CAPSULE, 0.0)
    on, _ = _render(_CAPSULE, 1.0)
  capsule = seg == int(seg[_SIZE // 2, _SIZE // 2])
  np.testing.assert_array_equal(on[capsule], off[capsule])
  assert on[~capsule].sum() < off[~capsule].sum()


def test_a_hidden_instance_casts_no_occlusion():
  with wp.ScopedDevice("cpu"):
    off, _ = _render(_CUBE, 0.0, hidden=True)
    on, _ = _render(_CUBE, 1.0, hidden=True)
  np.testing.assert_array_equal(on, off)
