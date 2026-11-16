"""Mesh BLAS builders and frames: one BLAS per mesh, same images."""

import mujoco_warp as mjw
import numpy as np
import pytest

import raywarp as rw
from raywarp.core.pipeline.platform import tuning


def _l_shape() -> tuple[str, str]:
  """(vertex, face) of an axis-aligned L of two boxes: its principal axes are diagonal."""
  boxes = [((0.0, 0.0, 0.0), (1.0, 0.1, 0.1)), ((0.0, 0.1, 0.0), (0.1, 0.5, 0.1))]
  quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
  verts, faces = [], []
  for lo, hi in boxes:
    base = len(verts)
    verts += [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    for a, b, c, d in quads:
      faces += [(base + a, base + b, base + c), (base + a, base + c, base + d)]
  return " ".join(f"{x} {y} {z}" for x, y, z in verts), " ".join(f"{a} {b} {c}" for a, b, c in faces)


_VERTEX, _FACE = _l_shape()
XML = f"""
<mujoco>
  <asset>
    <mesh name="plank" vertex="{_VERTEX}" face="{_FACE}"/>
    <mesh name="tet" vertex="0 0 0  0.3 0 0  0 0.3 0  0 0 0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -2 1.2" xyaxes="1 0 0 0 0.5 1"/>
    <geom type="plane" size="2 2 0.1"/>
    <geom type="mesh" mesh="plank" pos="-0.4 0 0.2" rgba="0.8 0.5 0.2 1"/>
    <geom type="mesh" mesh="tet" pos="0.5 0.4 0" rgba="0.2 0.5 0.8 1"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("constructor", ["sah", "lbvh"])
def test_mesh_bvh_constructor_renders_the_same(scene, constructor, monkeypatch):
  reference = scene(XML, cam_res=(32, 24)).render()
  monkeypatch.setattr(tuning, "MESH_BVH", constructor)
  s = scene(XML, cam_res=(32, 24)).render()
  meshes = s.scene.meshes
  framed, plain = meshes.bvhs(constructor, used=None, framed=True).numpy(), meshes.bvhs(constructor, used=None).numpy()
  assert len(framed) == s.mjm.nmesh and np.all(framed != 0)
  assert s.scene.capabilities.mesh_frames  # the L plank is re-framed: its own BLAS, the others shared
  assert [framed[i] != plain[i] for i in range(s.mjm.nmesh)] == [i in meshes.framed_points for i in range(s.mjm.nmesh)]
  np.testing.assert_allclose(s.rgb(), reference.rgb(), atol=1.5 / 255)


# A framed plank shared by every world and one on a free body posed per world.
MOVING = XML.replace(
  '<geom type="mesh" mesh="tet"',
  '<body name="free" pos="0.4 0.3 0.3"><freejoint/><geom type="mesh" mesh="plank" euler="0 0 40" rgba="0.3 0.8 0.3 1"/></body>'
  '<geom type="mesh" mesh="tet"',
)


def test_shared_mesh_frames_are_written_in_world_0_only(scene):
  """The shared tree reads world 0's trace frames (`instances.make_tree_pose`): the shared mesh
  columns are written there only, the images equal those of per-world copies, byte for byte."""
  kinds = {"cam_res": (32, 24), "nworld": 3, "performance.traversal": "software"}
  s = scene(MOVING, **kinds)
  copies = scene(MOVING, per_world=rw.PerWorld(pose=["*"]), **kinds)
  for r in (s, copies):
    qpos = r.d.qpos.numpy()
    qpos[1, 0] += 0.1  # the free plank moves in worlds 1 and 2
    qpos[2, 1] -= 0.2
    r.d.qpos.assign(qpos)
    mjw.forward(r.m, r.d)
    r.render()
  sw = s.r._software
  assert sw.rot_trace is not None and sw.mesh_columns is not None
  shared = sw.mesh_columns.numpy()[sw.mesh_per_world :]
  assert sw.mesh_per_world == 1 and len(shared) == 2 and set(shared.tolist()) <= set(s.r.layout.shared_ids.tolist())
  trace = sw.rot_trace.numpy()
  assert np.all(trace[1:, shared] == 0) and np.any(trace[0, shared] != 0)
  np.testing.assert_array_equal(s.out.rgb.numpy(), copies.out.rgb.numpy())
  assert np.any(s.rgb()[1] != s.rgb()[0])
