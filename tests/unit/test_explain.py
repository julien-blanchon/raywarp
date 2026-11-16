"""`Renderer.explain()`: the cost class of every recipe and the reasons of "auto" choices, read
from the resolved Features."""

from __future__ import annotations

import pytest
from conftest import MakeScene, recipe

import raywarp as rw
from raywarp.core.build.variants.explain import Explanation

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom type="box" size="0.3 0.3 0.3" pos="0 0 0.3" rgba="1 0 0 1"/>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""

# Recipe (`conftest.recipe`) -> its cost class on a scene with one point light (no area light: the
# default's class comes from its reflection rays).
_COST = {
  "unlit_fast": "unlit",
  "fast": "fast",
  "default": "default",
  "quality": "quality",
  "reference": "quality",
  "rtx_look": "default",
}


def _explain(scene: MakeScene, settings: rw.RenderSettings) -> Explanation:
  s = scene(XML, cam_res=(8, 8))
  r = rw.Renderer(s.scene, nworld=1, cameras=[s.scene.cameras["cam"]], resolution=(8, 8), settings=settings)
  try:
    return r.explain()
  finally:
    r.close()


@pytest.mark.parametrize("name", list(_COST))
def test_every_recipe_has_its_cost_class(scene: MakeScene, name: str) -> None:
  e = _explain(scene, recipe(name).replace(gi={"enabled": False}))
  assert e.cost == _COST[name], str(e)
  assert e.cost_why and str(e).startswith(f"cost: {e.cost}")


def test_geometry_outputs_compile_no_shading(scene: MakeScene) -> None:
  e = _explain(scene, recipe("default").replace(outputs=rw.Outputs.of("depth", "seg"), gi={"enabled": False}))
  assert e.cost == "geometry" and e.row("shading").resolved == "off"


def test_auto_choices_say_why(scene: MakeScene) -> None:
  """The scene has no texture, no area light and one light: textures "auto" is off for that
  reason, the light grid stays off below 8 lights, the cap does not bind."""
  e = _explain(scene, recipe("default").replace(gi={"enabled": False}))
  textures = e.row("textures")
  assert textures.resolved == "off" and textures.why.startswith("not:")
  assert e.row("light grid").resolved == "off"
  assert e.row("lights").resolved == "1 exact"
  assert e.row("area lights").resolved == "off" and e.row("area lights").why == "no area light"
  assert e.row("shadows").resolved == "1 ray(s), every caster"
  assert e.row("display").resolved == "deferred"


def test_unlit_and_lightless_rows_name_their_cause(scene: MakeScene) -> None:
  """Unlit shading turns shadows and reflections off for that reason (the rows said "no light" and
  "no shaded output"); a lit scene without lights traces no shadow ray (the row said "on")."""
  e = _explain(scene, recipe("default").replace(surfaces={"shading": "unlit"}, gi={"enabled": False}))
  assert e.row("shadows").why == e.row("reflections").why == 'surfaces.shading="unlit"'
  dark = scene(XML.replace('<light pos="0 0 3" dir="0 0 -1"/>', ""), cam_res=(8, 8))
  r = rw.Renderer(dark.scene, nworld=1, cameras=[dark.scene.cameras["cam"]], resolution=(8, 8), settings=recipe("default"))
  try:
    row = r.explain().row("shadows")
  finally:
    r.close()
  assert (row.resolved, row.why) == ("off", "no light")
