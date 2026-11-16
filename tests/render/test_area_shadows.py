"""The soft area shadows' pass (`shadows.area_rays`, `shade/shadows/area.py`): off (the default)
it adds no output and no work; on, it softens the hard umbra of a large near light and leaves the
open floor as it is, deterministically, and it needs the settings it is defined for."""

import numpy as np
import pytest

import raywarp as rw

# A large bulb (radius 0.25 m, a sphere area light with LTC lights) 0.5 m above a small box on a
# matte floor: the centre ray's umbra is a sharp, magnified square.
XML = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <light type="point" pos="0 0 0.55" intensity="4" range="10" bulbradius="0.25" castshadow="true"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="box" pos="0 0 0.08" size="0.06 0.06 0.08" rgba="0.8 0.8 0.8 1"/>
    <camera name="cam" pos="0 -0.9 0.9" xyaxes="1 0 0 0 0.7 0.7"/>
  </worldbody>
</mujoco>
"""
_OPTS = {"cam_res": (64, 64), "outputs": rw.Outputs.of("hdr", "seg"), "lighting.area_lights": "ltc"}


def test_off_adds_no_output_and_no_pass(scene) -> None:
  """Off (the default): the written outputs are the requested ones and the renderer has no pass."""
  s = scene(XML, **_OPTS)
  assert s.r.settings.area_rays() == 0
  assert s.r.settings.written_outputs() == s.r.settings.outputs
  assert s.r._area is None  # the pass' state is private


@pytest.mark.parametrize(
  ("changes", "needs"),
  [
    ({"lighting.area_lights": "point"}, "area_lights"),
    ({"aa.spp": 2}, "aa.spp"),
    ({"outputs": rw.Outputs(rgb=rw.settings.ColorOutput(), position=rw.settings.FloatOutput(dtype="float16"))}, "position"),
  ],
)
def test_needs_its_settings(changes: dict[str, object], needs: str) -> None:
  """On, it raises for the settings it is not defined for, naming the one missing."""
  with pytest.raises(ValueError, match=needs):
    rw.RenderSettings().replace(**{"shadows.area_rays": 8, **changes})


def test_on_writes_its_inputs() -> None:
  """On, the render also writes the position (float32) and normal it reads (the requested ones
  kept as requested), through the deferred display."""
  s = rw.RenderSettings().replace(**{"shadows.area_rays": 8, "exposure.mode": "manual"})
  written = s.written_outputs()
  assert {"position", "normal", "hdr"} <= written.kinds
  assert "albedo" not in written.kinds
  assert written.position is not None and written.position.dtype == "float32"
  assert s.deferred_display()
  geometric = rw.Outputs(rgb=rw.settings.ColorOutput(), normal=rw.settings.NormalOutput(frame="shading"))
  assert s.replace(outputs=geometric).written_outputs().normal == geometric.normal


def test_softens_the_umbra_only(scene) -> None:
  """On, the floor under the box (the centre ray's umbra) gets the light it partly sees; the open
  floor and the box's top are as before; two frames are identical; turning it off again renders
  the image without it."""
  s = scene(XML, **_OPTS)
  base = s.render().hdr().copy()
  seg = s.seg().copy()
  floor = seg == seg[0, 4, 32]  # the floor's segment id (a pixel of the image's top edge)
  umbra = floor & (base.max(axis=-1) < 0.02 * base[floor].max())
  assert umbra.sum() > 20
  s.r.update_settings(**{"shadows.area_rays": 8})
  soft = s.render().hdr().copy()
  np.testing.assert_array_equal(soft, s.render().hdr())
  assert np.isfinite(soft).all()
  # The umbra gets a large share of the light the open floor beside it gets.
  assert soft[umbra].mean() > 0.2 * base[floor & ~umbra].mean()
  # The open floor far from the box: as before (within the half-float images' rounding).
  far = floor.copy()
  far[..., 16:48, 16:48] = False
  np.testing.assert_allclose(soft[far], base[far], rtol=0.02, atol=1e-6)
  s.r.update_settings(**{"shadows.area_rays": 0})
  np.testing.assert_array_equal(s.render().hdr(), base)
