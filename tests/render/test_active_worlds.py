"""`Renderer.render(active=...)`: only the listed worlds render, the others keep their outputs,
toggling never compiles, and a world rendered again restarts its temporal state."""

import numpy as np
import pytest
import warp as wp
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp._base.types import RenderOrder

XML = """
<mujoco>
  <worldbody>
    <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="ball" pos="0 0 0.4">
      <freejoint/>
      <geom type="sphere" size="0.3" rgba="0.8 0.3 0.2 1"/>
    </body>
    <geom type="box" pos="0.6 0.2 0.2" size="0.2 0.2 0.2" rgba="0.2 0.4 0.9 1"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
  </worldbody>
</mujoco>
"""
NWORLD = 5
RES = (16, 12)
OUTPUTS = rw.Outputs.of("rgb", "depth", "seg")


def _ids(values: list[int]) -> wp.array:
  return wp.array(np.asarray(values, np.int32), dtype=wp.int32)


def _per_world_scene(scene, **kw):
  """NWORLD worlds whose ball colors differ (every world's image is its own)."""
  s = scene(XML, nworld=NWORLD, cam_res=RES, outputs=OUTPUTS, per_world=rw.PerWorld(material=["color"]), **kw)
  color = s.r.per_world["instance.color"]
  rgba = color.numpy()
  rgba[:, 1, :3] = np.linspace(0.1, 0.9, NWORLD)[:, None]
  color.assign(rgba)
  return s


def _outputs(s) -> dict[str, np.ndarray]:
  return {"rgb": s.out.rgb.numpy().copy(), "depth": s.out.depth.numpy().copy(), "seg": s.out.seg.numpy().copy()}


def _poison(s) -> None:
  s.out.rgb.fill_(wp.vec4ub(1, 2, 3, 4))
  s.out.depth.fill_(-7.0)
  s.out.seg.fill_(-9)


@pytest.mark.parametrize("exposure", ["manual", "auto"])
@pytest.mark.parametrize("ids", [[3, 1], [-1, 4, -1, 0, 2], [2]])
def test_listed_worlds_match_a_full_render_and_others_keep_their_outputs(scene, ids, exposure):
  """Also with automatic exposure: the deferred display and the metering launch over the listed
  worlds only (the active list's units)."""
  s = _per_world_scene(scene, **{"exposure.mode": exposure})
  full = _outputs(s.render())
  _poison(s)
  poisoned = _outputs(s)
  s.r.render(s.d, active=_ids(ids))
  got = _outputs(s)
  listed = sorted(w for w in ids if w >= 0)
  others = [w for w in range(NWORLD) if w not in listed]
  for name in full:
    np.testing.assert_array_equal(got[name][listed], full[name][listed], err_msg=name)
    np.testing.assert_array_equal(got[name][others], poisoned[name][others], err_msg=name)


def test_random_toggling_compiles_nothing_and_matches_full_renders(scene):
  """The ids array is written in place every frame (as inside a graph): no compile, every frame's
  listed worlds equal a full render of that frame."""
  s = _per_world_scene(scene)
  full = _outputs(s.render())
  ids = _ids([-1] * NWORLD)
  s.r.render(s.d, active=ids)
  rng = np.random.default_rng(7)
  with module_loads() as loads:
    for _ in range(12):
      on = rng.random(NWORLD) < 0.5
      ids.assign(np.where(on, np.arange(NWORLD), -1).astype(np.int32))
      _poison(s)
      poisoned = _outputs(s)
      s.r.render(s.d, active=ids)
      got = _outputs(s)
      for name in full:
        np.testing.assert_array_equal(got[name][on], full[name][on], err_msg=name)
        np.testing.assert_array_equal(got[name][~on], poisoned[name][~on], err_msg=name)
    short = _ids([4, 0])  # another length: kernel structs rebuilt, nothing compiled
    s.r.render(s.d, active=short)
    s.r.render(s.d)
  assert not loads, loads


def test_invalid_ids_raise_on_the_next_render(scene):
  s = _per_world_scene(scene)
  s.r.render(s.d, active=_ids([1, 1]))
  with pytest.raises(ValueError, match="distinct world ids"):
    s.r.render(s.d, active=_ids([0]))
  s.r.render(s.d, active=_ids([NWORLD]))
  with pytest.raises(ValueError, match="distinct world ids"):
    s.r.render(s.d)
  s.r.render(s.d)  # a valid frame: nothing raised


def test_bad_active_arrays_raise(scene):
  s = _per_world_scene(scene)
  with pytest.raises(TypeError, match="warp int32 array"):
    s.r.render(s.d, active=np.zeros(2, np.int32))
  with pytest.raises(ValueError, match="1-D int32"):
    s.r.render(s.d, active=wp.zeros(2, dtype=bool))
  with pytest.raises(ValueError, match="1-D int32"):
    s.r.render(s.d, active=wp.zeros(NWORLD + 1, dtype=wp.int32))


def test_a_world_rendered_again_restarts_its_history(scene):
  """The motion output's history: a skipped world keeps its history length while skipped, and
  restarts at 1 when it is listed again (its next frame renders as a first frame); the others go
  on."""
  s = scene(XML, nworld=2, cam_res=RES, outputs=rw.Outputs.of("rgb", "motion"))
  for _ in range(3):
    s.render()
  first = s.r.debug.history_length.numpy()[:, 0].tolist()
  assert first == [3, 3]
  ids = _ids([0, -1])
  s.r.render(s.d, active=ids)
  s.r.render(s.d, active=ids)
  assert s.r.debug.history_length.numpy()[:, 0].tolist() == [5, 3]
  ids.assign(np.array([0, 1], np.int32))
  s.r.render(s.d, active=ids)
  assert s.r.debug.history_length.numpy()[:, 0].tolist() == [6, 1]
  ids.assign(np.array([-1, 1], np.int32))
  s.r.render(s.d, active=ids)
  s.r.render(s.d)  # every world again: world 0 restarts
  assert s.r.debug.history_length.numpy()[:, 0].tolist() == [1, 3]


def test_auto_order_compares_the_listed_worlds_cameras_only(scene):
  """RenderOrder.AUTO: an unlisted world whose camera differs does not turn MIXED off for the
  listed ones; a listed one does."""
  s = scene(XML, nworld=NWORLD, cam_res=RES, outputs=rw.Outputs.of("rgb"), per_world=rw.PerWorld(camera=["fovy"]))
  fovy = s.r.per_world["camera.fovy"]
  wide = fovy.numpy()
  wide[0, 0] += 5.0  # world 0's camera
  fovy.assign(wide)
  s.r.render(s.d, active=_ids([4, -1, 2, 3, 1]))
  assert int(s.r.context().cameras.order.numpy()[0][0]) == int(RenderOrder.MIXED)
  s.r.render(s.d, active=_ids([4, 0, 2, 3, 1]))
  assert int(s.r.context().cameras.order.numpy()[0][0]) == int(RenderOrder.TILED)
