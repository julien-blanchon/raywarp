"""The shadow budget compares its cap with the lights that can cast, not with the light count.

A budget covering every potential caster (active and castshadow in some world) traces each of them
and only adds its selection: resolve turns it off. Per-world declared tables count as on.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.core.pipeline.gpu_scene.gpu_scene import shadow_casters
from raywarp.integrations.mujoco.binding import MjwBinding
from raywarp.settings.fields import DEFAULT_EV100

_LIGHTS = {
  "three": ('castshadow="true"', 'castshadow="true"', 'castshadow="true"'),
  "inactive": ('castshadow="true"', 'castshadow="true"', 'castshadow="true" active="false"'),
  "no_cast": ('castshadow="true"', 'castshadow="true"', 'castshadow="false"'),
}


def _xml(lights: tuple[str, str, str]) -> str:
  """A plane, a sphere, three point lights with the given attributes and two non-casting ones
  (five lights: more than the small light form takes, whose budget selects nothing)."""
  rows = "\n".join(f'<light pos="{i - 1} 0 3" dir="0 0 -1" {a}/>' for i, a in enumerate(lights))
  rows += '\n<light pos="-3 2 3" dir="0 0 -1" castshadow="false"/>\n<light pos="3 2 3" dir="0 0 -1" castshadow="false"/>'
  return f"""
<mujoco>
  <worldbody>
    {rows}
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
  </worldbody>
</mujoco>
"""


def _renderer(
  kind: str, max_casters: int | None, per_world: rw.PerWorld | None = None, *, nworld: int = 2, gi: bool = False
) -> tuple[rw.Renderer, mujoco.MjModel]:
  """A CPU renderer of the three-light scene (`kind`: _LIGHTS) with a shadow cap (`gi`: probe GI
  on, the coarse test grid)."""
  mjm = mujoco.MjModel.from_xml_string(_xml(_LIGHTS[kind]))
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
  settings = rw.RenderSettings().replace(
    **{
      "performance.traversal": "software",
      "lighting.light_grid": False,
      "lighting.max_exact": None,
      "shadows.max_casters": max_casters,
      "exposure.mode": "manual",
      "exposure.ev100": DEFAULT_EV100,  # renders are compared across configurations
      "gi.enabled": gi,
    }
  )
  r = rw.Renderer(scene, nworld=nworld, cameras=["cam"], resolution=(32, 24), settings=settings, per_world=per_world)
  return r, mjm


@pytest.mark.parametrize(("kind", "budget"), [("three", 2), ("inactive", 2), ("no_cast", 0)])
def test_budget_counts_potential_casters(kind: str, budget: int, monkeypatch: pytest.MonkeyPatch) -> None:
  """Cap 2: a budget with three casting lights, also when one is inactive (it can be turned on
  without a host call); none with a non-casting third light."""
  monkeypatch.setattr(rw.Renderer, "_start_compiles", lambda self: None)  # Features only
  with wp.ScopedDevice("cpu"):
    r, _ = _renderer(kind, 2)
  assert r.scene.nlight == 5
  assert r.features.budget == budget


def test_declared_cast_shadow_counts_as_on(monkeypatch: pytest.MonkeyPatch) -> None:
  """A per-world light.cast_shadow table can turn the third light's shadow on in any world: budget
  kept."""
  monkeypatch.setattr(rw.Renderer, "_start_compiles", lambda self: None)  # Features only
  with wp.ScopedDevice("cpu"):
    r, _ = _renderer("no_cast", 2, per_world=rw.PerWorld(light=("cast_shadow",)))
  assert r.features.budget == 2


def test_shadow_casters_any_world() -> None:
  """A light casting in one world only is a potential caster, active or not."""
  with wp.ScopedDevice("cpu"):
    cast = wp.array(np.array([[True, True, False], [True, False, False]]), dtype=bool)
    slots = {"light.cast_shadow": cast}
    assert shadow_casters(slots, frozenset()).tolist() == [True, True, False]
    assert shadow_casters(slots, frozenset({"light.cast_shadow"})).all()


def _model(r: rw.Renderer) -> mjw.Model:
  """The device model the renderer's scene aliases."""
  binding = r.scene.binding
  assert isinstance(binding, MjwBinding)
  return binding.m


def _rgba(r: rw.Renderer, d: mjw.Data) -> np.ndarray:
  """The 8-bit image of one render."""
  r.render(d)
  wp.synchronize()
  rgba = r.outputs.rgb
  assert rgba is not None
  return rgba.numpy().copy()


def _data(mjm: mujoco.MjModel, nworld: int = 2) -> mjw.Data:
  """`nworld` worlds of `mjm` at its initial state."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  return mjw.put_data(mjm, mjd, nworld=nworld)


def test_non_caster_render_matches_uncapped() -> None:
  """With the third light not casting, cap 2 renders exactly the all-casters image."""
  with wp.ScopedDevice("cpu"):
    images = []
    for cap in (2, None):
      r, mjm = _renderer("no_cast", cap)
      images.append(_rgba(r, _data(mjm)))
      r.close()
  np.testing.assert_array_equal(images[0], images[1])


def test_light_turned_on_later_keeps_the_budget() -> None:
  """An inactive caster turned on after the Renderer was made: the budget built for it applies
  (the image of a Renderer made with the light on)."""
  with wp.ScopedDevice("cpu"):
    r, mjm = _renderer("inactive", 2)
    m = _model(r)
    d = _data(mjm)
    off = _rgba(r, d)
    m.light_active.assign(np.ones(3, bool))
    late = _rgba(r, d)
    assert not np.array_equal(late, off)  # the light read live from the model
    on, mjm_on = _renderer("three", 2)
    np.testing.assert_array_equal(late, _rgba(on, _data(mjm_on)))


@pytest.mark.parametrize("nworld", [1, 2])
def test_shared_light_turned_on_later_re_bakes_gi(nworld: int) -> None:
  """GI on: a light of the model's one shared m.light_active row turned on after the bake; a frame
  later the image is that of a Renderer made with the light on (its bounce light re-baked): the
  inputs every world shares are checked against the bake's every frame (`gi.Groups.track`)."""
  with wp.ScopedDevice("cpu"):
    r, mjm = _renderer("inactive", 2, nworld=nworld, gi=True)
    m = _model(r)
    d = _data(mjm, nworld)
    _rgba(r, d)
    m.light_active.assign(np.ones(3, bool))
    _rgba(r, d)  # the change may be seen one frame late (a pinned read, no synchronisation)
    late = _rgba(r, d)
    on, mjm_on = _renderer("three", 2, nworld=nworld, gi=True)
    np.testing.assert_array_equal(late, _rgba(on, _data(mjm_on, nworld)))


def test_shadow_turned_on_later_re_resolves_the_budget() -> None:
  """m.light_castshadow written on for the third light: the next frame's validation flags it, the
  frame after re-resolves the Features with the budget (the image of a Renderer made with it)."""
  with wp.ScopedDevice("cpu"):
    r, mjm = _renderer("no_cast", 2)
    assert r.features.budget == 0 and r._watch.validator.casters is not None
    m = _model(r)
    d = _data(mjm)
    _rgba(r, d)
    m.light_castshadow.assign(np.ones(3, bool))
    _rgba(r, d)  # checked: the third light casts outside the resolved casters
    late = _rgba(r, d)  # re-resolved before this frame
    assert r.features.budget == 2 and r._watch.validator.casters is None
    on, mjm_on = _renderer("three", 2)
    np.testing.assert_array_equal(late, _rgba(on, _data(mjm_on)))
