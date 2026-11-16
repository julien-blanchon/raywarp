"""Smoke test of the real-time tier on Warp's CPU device (tiny scene, no environment map)."""

import numpy as np
import warp as wp
from conftest import required

import raywarp as rw

_XML = """
  <mujoco>
    <asset><material name="m" rgba="0.8 0.3 0.2 1" metallic="0.5" roughness="0.4"/></asset>
    <worldbody>
      <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
      <geom type="plane" size="2 2 0.1"/>
      <geom type="sphere" pos="0 0 0.3" size="0.3" material="m"/>
      <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
  </mujoco>
"""


def _render(scene, device: str, nworld: int) -> np.ndarray:
  with wp.ScopedDevice(device):
    return scene(_XML, nworld=nworld, cam_res=(16, 12)).render().rgb()


def test_cpu_device_renders(scene):
  rgb = _render(scene, "cpu", nworld=2)
  assert rgb.shape == (2, 12, 16, 3)
  assert np.isfinite(rgb).all() and rgb.min() >= 0.0 and rgb.max() <= 1.0
  assert rgb.std() > 0.01  # sphere, floor and shadow, not a flat image
  np.testing.assert_array_equal(rgb[0], rgb[1])  # identical worlds
  if wp.is_cuda_available():  # the README promises CPU images within 1/255 of the GPU's
    gpu = _render(scene, "cuda:0", nworld=1)
    assert np.abs(np.round(rgb[0] * 255) - np.round(gpu[0] * 255)).max() <= 1


def _panel_room() -> rw.Scene:
  """A floor lit only by two emissive ceiling panels (converted to rect lights), a box between."""
  b = rw.SceneBuilder()
  glow = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=50.0, emission_color=(1, 1, 1))
  )
  b.add_instance("plane", size=(4, 4, 0))
  for x in (-1.0, 1.0):
    b.add_instance("box", size=(0.5, 0.5, 0.01), xform=np.c_[np.eye(3), [x, 0, 2.0]], material=glow)
  b.add_instance("box", size=(0.2, 0.2, 0.2), xform=np.c_[np.eye(3), [0, 0, 0.2]])
  return b.build()


def _panel_render(host: rw.Scene, device: str) -> tuple[np.ndarray, np.ndarray, int]:
  """(hdr (H, W, 3), rgb (H, W, 3) uint8, nlight) of the panel room seen from above on `device`."""
  gpu = rw.upload(host, device=device)
  outs = rw.Outputs(rgb=rw.settings.ColorOutput(), hdr=rw.settings.FloatOutput())
  settings = rw.RenderSettings().replace(
    **{"exposure.mode": "manual", "exposure.ev100": 6.0}, outputs=outs
  )  # GI off (session default)
  r = rw.Renderer(gpu, nworld=1, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(24, 24), settings=settings)
  r.camera_pose.assign(np.array([[[0, 0, 1.8, 0, 0, 0, 1]]], np.float32))  # under the panels, looking down -z
  for _ in range(2):
    out = r.render(None)
  return required(out.hdr).numpy()[0, 0], required(out.rgb).numpy()[0, 0][..., :3], gpu.nlight


def test_cpu_emissive_panels_light_the_floor():
  """Emissive panels as rect lights on the CPU: every floor / box pixel in view is lit (the Isaac
  Hospital once looked dark on the CPU: its MDL emission was missing, not the CPU path) and the
  image matches the GPU's."""
  host = _panel_room()
  hdr, rgb, nlight = _panel_render(host, "cpu")
  assert nlight >= 2  # both panels became lights
  assert (np.abs(hdr).max(-1) > 0).all()
  assert hdr.mean() > 1.0
  if wp.is_cuda_available():
    _, gpu_rgb, _ = _panel_render(host, "cuda:0")
    diff = np.abs(rgb.astype(np.int32) - gpu_rgb.astype(np.int32))
    assert diff.max() <= 2 and diff.mean() < 0.5
