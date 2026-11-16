"""`surfaces.shading="unlit"` and `conftest.UNLIT_FAST` (the old `fastest` preset, W26)."""

import dataclasses

import numpy as np
from conftest import recipe

import raywarp as rw

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <light pos="1 1 3" dir="0 0 -1" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.6 0.4 1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.2 0.5 0.9 1"/>
    <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2" rgba="0.9 0.2 0.2 1"/>
    <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""


def test_unlit_radiance_is_the_base_color(scene):
  """Unlit: radiance x exposure is the first hit's base color (the albedo output) everywhere
  something is hit; lights and shadows play no part."""
  kinds = rw.Outputs.of("rgb", "hdr", "albedo")
  s = scene(XML, cam_res=(32, 24), outputs=kinds, **{"surfaces.shading": "unlit"}).render()
  hdr, albedo = s.hdr()[0], s.albedo()[0]
  hit = albedo.sum(axis=-1) > 0.0
  assert hit.mean() > 0.5
  np.testing.assert_allclose(hdr[hit] * s.exposure, albedo[hit], rtol=1e-5, atol=1e-6)


def test_unlit_fast_compiles_no_lighting(scene):
  """`UNLIT_FAST` is unlit, opaque, without shadows, reflections, light loops or GI."""
  s = scene(
    XML,
    cam_res=(16, 12),
    **{f"{g.name}": getattr(recipe("unlit_fast"), g.name) for g in dataclasses.fields(rw.RenderSettings)},
  )
  f = s.r.features
  assert f.unlit and not f.shadows and f.nlight == 0 and not f.reflection and not f.gi
  assert not f.transparency and not f.env_lit
  s.render()
  assert s.rgb()[0].max() > 0.2


EMISSIVE_XML = """
<mujoco>
  <asset><material name="lamp" rgba="0.2 0.5 0.9 1" emission="1"/></asset>
  <worldbody>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.6 0.4 1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3" material="lamp"/>
    <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""


def test_unlit_shows_emission(scene):
  """Sweep S6: unlit adds the emission as a lit render shows it (lamps stay visible to a policy;
  they were drawn as their base color only): the lamp reads brighter than its albedo, the floor
  stays its albedo."""
  kinds = rw.Outputs.of("hdr", "albedo")
  s = scene(EMISSIVE_XML, cam_res=(32, 24), outputs=kinds, **{"surfaces.shading": "unlit"}).render()
  hdr, albedo = s.hdr()[0] * s.exposure, s.albedo()[0]
  hit = albedo.sum(axis=-1) > 0.0
  lamp = hit & (hdr.sum(axis=-1) > 1.2 * albedo.sum(axis=-1))
  floor = hit & ~lamp
  assert lamp.sum() > 10 and floor.sum() > 10
  np.testing.assert_allclose(hdr[lamp], 2.0 * albedo[lamp], rtol=1e-4)  # emission 1: + its base color
  np.testing.assert_allclose(hdr[floor], albedo[floor], rtol=1e-5, atol=1e-6)
