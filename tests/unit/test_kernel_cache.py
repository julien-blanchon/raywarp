"""Settings changed after the first render, and the process-global kernel cache."""

import numpy as np
import pytest
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp.core.kernels import pixel
from raywarp.core.trace.rays.primary import PINHOLE

XML = """
<mujoco>
  <asset><material name="plastic" roughness="0.6"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1" material="plastic"/>
    <geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>
  </worldbody>
</mujoco>
"""

# (setting, new value, compile-time).
CHANGES = [
  ("performance.render_order", "world", False),
  ("post.tone_mapper", "aces", False),
  ("shadows.enabled", False, True),
  ("aa.spp", 2, True),
  ("exposure.ev100", 14.0, False),
  ("reflections.max_roughness", 0.3, False),
  # Run-time switches of the deferred display: on under manual exposure they add its HDR / depth
  # output (the software build's output word, so its Features change).
  ("grading.enabled", True, True),
  ("lighting.contact_ao", 1.0, True),
]


@pytest.mark.parametrize("field,value,compiled", CHANGES, ids=[f"{f}={v}" for f, v, _ in CHANGES])
def test_change_after_first_render(scene, field, value, compiled):
  """A setting changed after the first render renders as a renderer created with it."""
  opts = {"nworld": 2, "cam_res": (24, 16), "outputs": rw.Outputs.of("rgb", "seg")}
  s = scene(XML, **opts).render()
  features = s.r.features
  kernel = pixel.megakernel(features, PINHOLE)
  s.r.update_settings(**{field: value})
  s.render()
  expected = scene(XML, **opts, **{field: value}).render()
  np.testing.assert_allclose(s.rgb(), expected.rgb(), atol=1.5 / 255)
  np.testing.assert_array_equal(s.seg(), expected.seg())
  # Compile-time settings switch kernels; runtime ones keep the same Features and kernel.
  assert (s.r.features != features) == compiled
  assert (pixel.megakernel(s.r.features, PINHOLE) is not kernel) == compiled


def test_same_features_share_a_kernel(scene):
  """Renderers with equal Features share the kernels of the process: a second renderer of another
  run-time setting loads no Warp module (none compiles)."""
  a = scene(XML, cam_res=(24, 16)).render()
  with module_loads() as loads:
    b = scene(XML, cam_res=(24, 16), exposure=rw.settings.Exposure(mode="manual", ev100=14.0)).render()
  assert a.r.features == b.r.features
  assert not loads, loads


def test_segmentation_background_without_fill(scene):
  """Pixels that hit nothing are -1 in every frame (written by the kernel itself)."""
  s = scene(XML, cam_res=(24, 16), outputs=rw.Outputs.of("rgb", "seg")).render()
  seg = s.seg().copy()
  assert np.all(seg[0, 0] == -1)  # top row: sky
  s.out.seg.fill_(7)
  s.render()
  np.testing.assert_array_equal(s.seg(), seg)


def test_per_world_override_is_used(scene):
  """Input structs are cached between frames, but writing a per-world array takes effect."""
  s = scene(XML, cam_res=(24, 16), per_world=rw.PerWorld(material=["specular_roughness", "base_metalness"])).render()
  before = s.rgb()
  s.render()
  np.testing.assert_array_equal(s.rgb(), before)
  s.r.per_world["material.specular_roughness"].fill_(0.05)
  s.r.per_world["material.base_metalness"].fill_(1.0)
  s.render()
  assert np.abs(s.rgb() - before).max() > 0.05


def test_replaced_model_array_is_used(scene):
  """The MuJoCo binding re-reads m every render: replacing m.geom_rgba takes effect."""
  s = scene(XML, cam_res=(24, 16)).render()
  before = s.rgb()
  rgba = s.m.geom_rgba.numpy()
  rgba[..., :3] = 0.1
  import warp as wp

  s.m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  s.render()
  assert np.abs(s.rgb() - before).max() > 0.05


def test_translucent_override_compiles_transparency(scene):
  """W41 (b): translucency is compiled in on an opaque scene when instance colors are declared per
  world with their alpha (`PerWorld.alpha`); RGB tints alone compile nothing (the plain kernel)."""
  plain = scene(XML, cam_res=(24, 16)).render()
  assert not plain.r.features.transparency
  tint = scene(XML, cam_res=(24, 16), per_world=rw.PerWorld(material=["color", "base_color"])).render()
  assert tint.r.features == plain.r.features
  np.testing.assert_array_equal(tint.rgb(), plain.rgb())
  with pytest.raises(ValueError, match="alpha of a declared color table"):
    rw.PerWorld(material=["specular_roughness"], alpha=True)
  s = scene(XML, cam_res=(24, 16), per_world=rw.PerWorld(material=["color"], alpha=True)).render()
  assert s.r.features.transparency
  opaque = s.rgb()
  np.testing.assert_array_equal(opaque, plain.rgb())  # opaque colors: the same image
  color = s.r.per_world["instance.color"]
  rgba = color.numpy()
  rgba[..., 3] = 0.5
  color.assign(rgba)
  s.render()
  assert np.abs(s.rgb() - opaque).max() > 0.05
