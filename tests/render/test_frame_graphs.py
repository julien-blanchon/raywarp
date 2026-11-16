"""`Renderer.render` on CUDA replays its own graphs (`graph.FrameGraphs`): every frame equals the
eager frame bit for bit (temporal state, moving geometry, a settings change in between), and the
periodic BVH rebuild recorded as a conditional body (`trace.accel.lbvh.Rebuild` under capture)
builds the same tree as eagerly."""

from typing import Literal, TypedDict

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, MjScene, required

import raywarp as rw
from raywarp.core.trace.accel import lbvh

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs need a GPU")

XML = """
<mujoco><worldbody>
  <light type="directional" dir="0.2 0.4 -1"/>
  <geom type="plane" size="3 3 .1" rgba=".6 .6 .6 1"/>
  <body name="ball" pos="0 0 .4"><freejoint/><geom type="sphere" size=".3" rgba=".8 .3 .1 1"/></body>
  <geom type="box" pos=".8 .3 .2" size=".2 .2 .2" rgba=".2 .5 .8 1"/>
  <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 .45 .9"/>
</worldbody></mujoco>
"""


@wp.kernel
def _shift(
  # In:
  dx: float,
  # Out:
  geom_xpos_out: wp.array2d[wp.vec3],
) -> None:
  """Moves geom 2 (the ball: MuJoCo lists the world body's plane and box first) of every world by
  dx along x (the box is static scenery, which may not move)."""
  w = wp.tid()
  geom_xpos_out[w, 2] = geom_xpos_out[w, 2] + wp.vec3(dx, 0.0, 0.0)


# `_scenes`' settings keywords (dotted: `RenderSettings.replace` paths).
_Settings = TypedDict(
  "_Settings",
  {
    "outputs": rw.Outputs,
    "performance.traversal": str,
    "performance.rebuild_every": int,
  },
)


def _scenes(scene: MakeScene, traversal: str) -> tuple[MjScene, MjScene]:
  """Two identical renderers with the motion output (temporal state: frame index, history) of 2
  worlds."""
  kw: _Settings = {
    "outputs": rw.Outputs(rgb=rw.settings.ColorOutput(), motion=rw.settings.MotionOutput()),
    "performance.traversal": traversal,
    "performance.rebuild_every": 2,
  }
  return (
    scene(XML, nworld=2, cam_res=(24, 16), cameras=["front"], **kw),
    scene(XML, nworld=2, cam_res=(24, 16), cameras=["front"], **kw),
  )


@pytest.mark.parametrize("traversal", ["software", "optix"])
def test_render_graph_frames_equal_eager_frames(scene: MakeScene, traversal: Literal["software", "optix"]) -> None:
  """render() (recorded on its second frame, then replayed) against forced eager frames: equal
  outputs every frame; a settings change drops the graph, the next frames record a new one."""
  with wp.ScopedDevice("cuda:0"):
    graphed, eager = _scenes(scene, traversal)
    seed = eager.r._seed_array(0)
    for frame in range(8):
      if frame == 5:
        for s in (graphed, eager):
          s.r.update_settings(**{"exposure.mode": "manual", "exposure.ev100": 9.0})
      for s in (graphed, eager):
        wp.launch(_shift, dim=2, inputs=[0.01 * (frame % 3)], outputs=[s.d.geom_xpos])
      graphed.r.render(graphed.d)
      eager.r.eager_frame(eager.d, seed, polled=False)
      np.testing.assert_array_equal(
        required(graphed.out.rgb).numpy(), required(eager.out.rgb).numpy(), err_msg=f"frame {frame}"
      )
      np.testing.assert_array_equal(
        required(graphed.out.motion).numpy(), required(eager.out.motion).numpy(), err_msg=f"frame {frame}"
      )
    graphs = required(graphed.r._graphs)._graphs
    assert len(graphs) == 1 and graphs[0][0] == graphed.r.debug.graph_version()


def test_render_records_again_for_a_new_simulator_state(scene: MakeScene) -> None:
  """Another mjw.Data (other arrays) runs eagerly, then records its own graph; the first one is
  kept for the original state."""
  with wp.ScopedDevice("cuda:0"):
    s, _ = _scenes(scene, "optix")
    other = mjw.put_data(s.mjm, s.mjd, nworld=2)
    for d in (s.d, s.d, s.d, other, other, other, s.d):
      s.r.render(d)
    assert len(required(s.r._graphs)._graphs) == 2


@pytest.mark.parametrize("due", [0, 1])
def test_captured_rebuild_equals_eager_rebuild(due: int) -> None:
  """The LBVH rebuild recorded under `wp.capture_if` gives the eager rebuild's topology (open gate)
  or leaves the tree (closed gate)."""
  rng = np.random.default_rng(3)
  nworld, group = 3, 16
  centres = rng.uniform(-2, 2, (nworld * group, 3)).astype(np.float32)
  trees = []
  for captured in (False, True):
    with wp.ScopedDevice("cuda:0"):
      lower = wp.array(centres - 0.1, dtype=wp.vec3)
      upper = wp.array(centres + 0.1, dtype=wp.vec3)
      groups = wp.array(np.repeat(np.arange(nworld), group).astype(np.int32), dtype=int)
      bvh = wp.Bvh(lower, upper, groups=groups, constructor="sah")
      rebuild = lbvh.Rebuild(bvh, nworld, group)
      roots = wp.full(nworld, -7, dtype=int)
      gate = wp.array([due], dtype=int)
      shifted_lower = wp.array(centres[::-1] - 0.1, dtype=wp.vec3)
      shifted_upper = wp.array(centres[::-1] + 0.1, dtype=wp.vec3)
      if captured:
        with wp.ScopedCapture() as capture:
          rebuild.rebuild(gate, shifted_lower, shifted_upper, roots)
        wp.capture_launch(required(capture.graph))
      else:
        rebuild.rebuild(gate, shifted_lower, shifted_upper, roots)
      wp.synchronize()
      trees.append((roots.numpy(), rebuild._children.numpy(), rebuild._left.numpy(), rebuild._right.numpy()))
  for eager_part, captured_part in zip(*trees, strict=True):
    np.testing.assert_array_equal(eager_part, captured_part)
  if not due:
    assert (trees[0][0] == -7).all()
