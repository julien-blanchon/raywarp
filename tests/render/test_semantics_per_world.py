"""Per-world semantic labels (`Renderer.set_semantics(rules, worlds)`): one label list for every
world, each world's class table its own; and the body-level rule helper
(`semantics.body_instances`)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, MjScene

import raywarp as rw
from raywarp.core.outputs.semantic_classes import body_instances
from raywarp.settings.fields import OutputKind

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="robot" pos="0 0 0.4">
      <geom name="ball" type="sphere" size="0.3"/>
      <body name="hand" pos="0.6 0.2 -0.1">
        <geom name="crate" type="box" size="0.2 0.2 0.2"/>
      </body>
    </body>
    <camera name="cam" pos="0 -2.5 0.8" xyaxes="1 0 0 0 0.2 1" fovy="60"/>
  </worldbody>
</mujoco>
"""
RES = (32, 24)
FLOOR, BALL, CRATE = 0, 1, 2


def _host(a: wp.array | None) -> np.ndarray:
  """A requested output's values (it must be requested)."""
  assert a is not None
  return a.numpy()


def _scene(scene: MakeScene, *kinds: OutputKind) -> MjScene:
  """Two worlds of `_XML` writing `kinds`."""
  return scene(_XML, nworld=2, cam_res=RES, outputs=rw.Outputs.of(*kinds))


def _expected(instance: np.ndarray, classes: dict[int, int]) -> np.ndarray:
  """The semantic ids of the instance ids `instance` (-1 miss -> 0, unlisted -> 1 unlabelled)."""
  out = np.where(instance == -1, 0, 1)
  for geom, cls in classes.items():
    out[instance == geom] = cls
  return out


@pytest.mark.parametrize("kinds", [("seg", "semantic"), ("semantic",)])
def test_worlds_get_their_own_classes_with_one_label_list(scene: MakeScene, kinds: tuple[OutputKind, ...]) -> None:
  """World 0 labels the ball "object", world 1 the crate "object" and the floor "ground": the
  same id means the same label in both worlds; the first per-world call replaces the table once
  (graphs stale), later ones write it in place; an all-world call restarts the numbering (with
  the semantic ids in their own buffer, and written alone into the seg buffer)."""
  instance = _host(_scene(scene, "seg").render().out.seg)[:, 0]
  s = _scene(scene, *kinds)
  version = s.r.debug.graph_version()
  assert s.r.set_semantics({"object": "ball"}, worlds=[0]) == ("background", "unlabelled", "object")
  assert s.r.debug.graph_version() != version  # the (nworld, ninstance) table is new
  table, version = s.r._tables.semantics.ids, s.r.debug.graph_version()  # the table under test
  labels = s.r.set_semantics({"ground": "floor", "object": "crate"}, worlds=np.array([False, True]))
  assert labels == ("background", "unlabelled", "object", "ground")
  assert s.r._tables.semantics.ids is table and table.shape == (2, 3)
  assert s.r.debug.graph_version() == version  # written in place
  semantic = _host(s.render().out.semantic)[:, 0]
  np.testing.assert_array_equal(semantic[0], _expected(instance[0], {BALL: 2}))
  np.testing.assert_array_equal(semantic[1], _expected(instance[1], {CRATE: 2, FLOOR: 3}))
  assert s.r.set_semantics({"box": "crate"}) == ("background", "unlabelled", "box")
  assert s.r._tables.semantics.ids is table  # the per-world table stays
  semantic = _host(s.render().out.semantic)[:, 0]
  for w in range(2):
    np.testing.assert_array_equal(semantic[w], _expected(instance[w], {CRATE: 2}))


def test_world_selection_errors(scene: MakeScene) -> None:
  """World indices out of range, a mask of another length and an empty selection raise."""
  s = _scene(scene, "semantic")
  for worlds, match in (([2], "world indices"), (np.ones(3, bool), r"\(2,\) bool"), ([], "no world")):
    with pytest.raises(ValueError, match=match):
      s.r.set_semantics({"object": "ball"}, worlds=worlds)


def test_body_instances_label_a_body_subtree() -> None:
  """A body pattern selects its geoms and its descendants' (MuJoCo: instance = geom)."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  assert body_instances(mjm, "robot") == [BALL, CRATE]
  assert body_instances(mjm, "robot", subtree=False) == [BALL]
  assert body_instances(mjm, ["hand", "world"], subtree=False) == [FLOOR, CRATE]
  with pytest.raises(ValueError, match="matches no"):
    body_instances(mjm, "arm*")
