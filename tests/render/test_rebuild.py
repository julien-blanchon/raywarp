"""`performance.rebuild_every` on the software scene BVH (P9): periodic rebuilds from the current
poses, gated on the device, with correct per-world group roots."""

from typing import Final

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.trace.accel import reach

_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]

# 12 boxes on mocap bodies, clustered at the origin (the BVH's initial topology), a camera above.
# The tests pose them per world (`_scatter`): declared per world (`_MOVING`), not shared scenery.
_BOXES = "\n".join(
  f'<body name="b{i}" mocap="true" pos="{0.01 * i} 0 0"><geom type="box" size="0.1 0.1 0.1" rgba="{i / 12} 0.5 0.5 1"/></body>'
  for i in range(12)
)
_XML = f"""
  <mujoco>
    <worldbody>
      <light type="directional" dir="0 0 -1"/>
      {_BOXES}
      <camera name="cam" pos="0 0 6" xyaxes="1 0 0 0 1 0" fovy="70"/>
    </worldbody>
  </mujoco>
"""


_MOVING: Final = rw.PerWorld(pose=["*"])


def _scatter(s, rng: np.random.Generator) -> None:
  """New per-world box positions over a 6 m square (written into the bound geom poses)."""
  pos = s.d.geom_xpos.numpy()
  pos[:, :, :2] = rng.uniform(-3.0, 3.0, size=pos[:, :, :2].shape)
  s.d.geom_xpos.assign(pos)


def _due(s) -> int:
  """The rebuild gate of the last refit (1: that refit rebuilt the world BVH)."""
  bvh = s.r._software.bvh
  assert bvh is not None
  return int(bvh._cadence.due.numpy()[0])


@pytest.mark.parametrize("device", _DEVICES)
def test_periodic_rebuild_matches_a_fresh_renderer(scene, device):
  kinds = {"outputs": rw.Outputs.of("depth", "seg"), "performance.traversal": "software"}
  with wp.ScopedDevice(device):
    s = scene(_XML, nworld=2, cam_res=(48, 48), per_world=_MOVING, **kinds, **{"performance.rebuild_every": 2})
    rng = np.random.default_rng(0)
    for frame in range(1, 5):
      _scatter(s, rng)
      s.render()
      assert _due(s) == (frame % 2 == 0)  # rebuilt on refits 2 and 4
    # The same poses through a fresh renderer (topology built from them): same hits in both worlds.
    fresh = scene(_XML, nworld=2, cam_res=(48, 48), per_world=_MOVING, **kinds)
    fresh.d.geom_xpos.assign(s.d.geom_xpos.numpy())
    fresh.render()
    np.testing.assert_array_equal(s.seg(), fresh.seg())
    np.testing.assert_allclose(s.depth(), fresh.depth(), atol=1e-6)
    assert len(np.unique(s.seg()[1])) == 13  # every box (and the background) in view


@pytest.mark.parametrize("device", _DEVICES)
def test_every_frame_rebuild_keeps_hits(scene, device):
  """Rebuilt on every refit while the boxes scatter: the hits equal a never-rebuilt renderer's
  (the new topology is a valid tree with one leaf per box and correct world roots)."""
  kinds = {"outputs": rw.Outputs.of("depth", "seg"), "performance.traversal": "software"}
  with wp.ScopedDevice(device):
    s = scene(_XML, nworld=3, cam_res=(40, 40), per_world=_MOVING, **kinds, **{"performance.rebuild_every": 1})
    ref = scene(_XML, nworld=3, cam_res=(40, 40), per_world=_MOVING, **kinds, **{"performance.rebuild_every": 0})
    rng = np.random.default_rng(3)
    for _ in range(3):
      _scatter(s, rng)
      ref.d.geom_xpos.assign(s.d.geom_xpos.numpy())
      s.render()
      ref.render()
      assert _due(s) == 1
      np.testing.assert_array_equal(s.seg(), ref.seg())
      np.testing.assert_array_equal(s.depth(), ref.depth())


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
@pytest.mark.parametrize("rebuild_every", [64, 1])
def test_rebuild_captures_in_a_graph(scene, rebuild_every):
  """A CUDA graph of a software frame with periodic rebuilds captures (no allocation in the
  rebuild) and its replays write the eager frames of the same poses."""
  kinds = {"outputs": rw.Outputs.of("depth", "seg"), "performance.traversal": "software"}
  with wp.ScopedDevice("cuda:0"):
    s = scene(_XML, nworld=2, cam_res=(32, 32), per_world=_MOVING, **kinds, **{"performance.rebuild_every": rebuild_every})
    ref = scene(_XML, nworld=2, cam_res=(32, 32), per_world=_MOVING, **kinds, **{"performance.rebuild_every": 0})
    graph = s.r.capture(sim=s.d)
    rng = np.random.default_rng(4)
    for _ in range(3):
      _scatter(s, rng)
      ref.d.geom_xpos.assign(s.d.geom_xpos.numpy())
      graph.launch()
      ref.render()
      np.testing.assert_array_equal(s.seg(), ref.seg())
      np.testing.assert_array_equal(s.depth(), ref.depth())


def test_default_rebuilds_periodically(scene):
  """The default schedule rebuilds the world BVH on every `rebuild_every`-th refit (periodic, not
  refit forever), none before."""
  every = rw.settings.Performance().rebuild_every
  assert every > 0
  s = scene(_XML, cam_res=(8, 8), per_world=_MOVING, **{"performance.traversal": "software"})
  for _ in range(every - 1):
    s.render()
    assert _due(s) == 0
  s.render()
  assert _due(s) == 1


def test_no_rebuild_with_rebuild_every_0(scene):
  s = scene(_XML, cam_res=(16, 16), per_world=_MOVING, **{"performance.traversal": "software", "performance.rebuild_every": 0})
  for _ in range(3):
    s.render()
    assert _due(s) == 0  # the cadence never ticks: no rebuild launch


@pytest.mark.parametrize("rebuild_every", [0, 1])
def test_hidden_instances_are_points_in_the_world_bvh(scene, rebuild_every):
  """An alpha-0 instance has a point box after the refit (rays skip it unread), also on rebuild
  frames (a finite centroid); the other boxes keep theirs and the hits are unchanged."""
  kinds = {"outputs": rw.Outputs.of("seg"), "performance.traversal": "software"}
  s = scene(_XML, nworld=2, cam_res=(48, 48), per_world=_MOVING, **kinds, **{"performance.rebuild_every": rebuild_every})
  _scatter(s, np.random.default_rng(1))
  rgba = s.m.geom_rgba.numpy()
  rgba[..., 3, 3] = 0.0  # box 3 hidden in every world
  s.m.geom_rgba.assign(rgba)
  s.render()
  bvh = s.r._software.bvh
  assert bvh is not None
  lower, upper = bvh.lower.numpy(), bvh.upper.numpy()
  group = bvh.ninstance + bvh.nflexprim
  for w in range(2):
    for k, gi in enumerate(bvh.enabled_ids.numpy()):
      assert np.array_equal(lower[w * group + k], upper[w * group + k]) == (gi == 3)
  seg = s.seg()
  assert not np.any(seg == 3) and len(np.unique(seg[1])) == 12  # 11 boxes and the background


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("rebuild_every", [0, 1])
def test_hidden_then_shown_instance_gets_its_box_back(scene, device, rebuild_every):
  """Hide a box (alpha 0: a point box in the world BVH), then show it again: its box and hits
  come back (the frame after equals the frame before), in the refit and the rebuilt tree."""
  kinds = {"outputs": rw.Outputs.of("depth", "seg"), "performance.traversal": "software"}
  with wp.ScopedDevice(device):
    s = scene(_XML, nworld=2, cam_res=(48, 48), per_world=_MOVING, **kinds, **{"performance.rebuild_every": rebuild_every})
    _scatter(s, np.random.default_rng(2))
    s.render()
    seg, depth = s.seg().copy(), s.depth().copy()
    assert np.any(seg == 3)
    rgba = s.m.geom_rgba.numpy().copy()  # (a CPU array's numpy() is a view)
    hidden = rgba.copy()
    hidden[..., 3, 3] = 0.0
    s.m.geom_rgba.assign(hidden)
    s.render()
    assert not np.any(s.seg() == 3)
    s.m.geom_rgba.assign(rgba)
    s.render()
    np.testing.assert_array_equal(s.seg(), seg)
    np.testing.assert_array_equal(s.depth(), depth)


@pytest.mark.parametrize("distance", [0.5 * reach.NEAR_RADIUS, 1.5 * reach.NEAR_RADIUS])
def test_a_box_beyond_near_radius_is_hidden_on_software(scene, distance):
  """The software world tree applies the OptiX world cell's rule (`trace.accel.reach`): world 1's
  box 3 moved `distance` along x, with world 1's camera above it. Within `reach.NEAR_RADIUS` of the
  world's reference the box keeps its tree box and is seen; beyond, its box is a point (the
  reference) and the camera sees the background. World 0 is unchanged."""
  kinds = {"outputs": rw.Outputs.of("seg"), "performance.traversal": "software"}
  s = scene(_XML, nworld=2, cam_res=(32, 32), per_world=_MOVING, **kinds)
  _scatter(s, np.random.default_rng(3))
  s.render()
  before = s.seg().copy()
  pos = s.d.geom_xpos.numpy()
  pos[1, 3] = (distance, 0.0, 0.0)
  s.d.geom_xpos.assign(pos)
  cam = s.d.cam_xpos.numpy()
  cam[1, 0] = (distance, 0.0, 1.0)  # close: the 0.2 m box covers pixels
  s.d.cam_xpos.assign(cam)
  s.render()
  seg = s.seg()
  np.testing.assert_array_equal(seg[0], before[0])
  bvh = s.r._software.bvh
  assert bvh is not None
  k = int(np.flatnonzero(bvh.enabled_ids.numpy() == 3)[0])
  box = bvh.ninstance + bvh.nflexprim + k  # world 1's primitive
  lower, upper = bvh.lower.numpy()[box], bvh.upper.numpy()[box]
  if distance <= reach.NEAR_RADIUS:
    assert np.any(seg[1] == 3)
    assert upper[0] > distance > lower[0]
  else:
    assert not np.any(seg[1] == 3)
    np.testing.assert_array_equal(lower, upper)
    assert np.abs(lower).max() < 10.0  # the world's reference, among its other boxes
