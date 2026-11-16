"""Emission calibration, physical and display-relative separately (docs/scenes.md, light units):

* physical (host scenes: UsdPreviewSurface / OpenPBR emission in nits): the HDR radiance is the
  authored luminance x colour at any exposure;
* display-relative (MuJoCo's emission): the HDR is emission x colour / exposure, so the exposed
  value (what the display transform receives) is emission x colour at any `exposure.ev100`. That is
  Filament's exposure-compensated emissive with a fixed compensation of 3 + log2(1.2) EV.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_scene, required

import raywarp as rw

_SETTINGS = {
  "outputs": rw.Outputs.of("rgb", "hdr"),
  "performance.traversal": "software",
  "environment.enabled": False,
  "gi.enabled": False,
  "shadows.enabled": False,
  "reflections.enabled": False,
}


def _host(ev100: float) -> np.ndarray:
  """HDR of an emissive host plane (60 nits, orange) filling the view, no lights."""
  b = rw.SceneBuilder()
  m = b.add_material(
    rw.Material(base_color=(0.0, 0.0, 0.0), emission_luminance=60.0, emission_color=(1.0, 0.5, 0.25), specular_weight=0.0)
  )
  b.add_instance("plane", material=m, size=(20.0, 20.0, 0.0))
  settings = rw.RenderSettings().replace(**_SETTINGS, **{"exposure.mode": "manual", "exposure.ev100": ev100})
  with rw.Renderer(
    rw.upload(b.build(), device="cpu"), nworld=1, cameras=[rw.CameraSpec(fovy=20.0)], resolution=(4, 4), settings=settings
  ) as r:
    r.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    return required(r.render().hdr).numpy()[0, 0]


@pytest.mark.parametrize("ev100", [6.0, 12.0])
def test_physical_emission_is_nits(ev100):
  """Physical emission: luminance x colour [nits], whatever the exposure."""
  np.testing.assert_allclose(_host(ev100), np.broadcast_to([60.0, 30.0, 15.0], (4, 4, 3)), rtol=1e-5)


_XML = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0" diffuse="0 0 0"/></visual>
  <asset><material name="glow" rgba="1 0.5 0.25 1" emission="0.8" specular="0"/></asset>
  <worldbody>
    <camera name="cam" pos="0 0 2" fovy="20"/>
    <!-- MuJoCo lights a model without lights by a 0.3 ambient: one black light keeps emission only. -->
    <light pos="0 0 3" diffuse="0 0 0" specular="0 0 0" ambient="0 0 0"/>
    <geom type="plane" size="10 10 0.1" material="glow"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("ev100", [0.0, 3.0])
def test_display_relative_emission_follows_exposure(ev100):
  """MuJoCo emission: HDR x exposure = emission x colour (the display value is fixed)."""
  s = make_scene(_XML, cam_res=(4, 4), cameras=["cam"], **{**_SETTINGS, "exposure.mode": "manual", "exposure.ev100": ev100})
  hdr = s.render().hdr()[0]
  np.testing.assert_allclose(hdr * 2.0**-ev100, np.broadcast_to([0.8, 0.4, 0.2], hdr.shape), rtol=1e-4)
