"""Image-based lighting tests: furnace, orientation, per-world maps, skyboxes, latlong layout."""

import numpy as np
import pytest
import warp as wp
from fixtures.display_helpers import apply_display_transform

import raywarp as rw
from raywarp.core.shade.environment import ibl


def _sphere_xml(metallic: float, roughness: float) -> str:
  return f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" metallic="{metallic}" roughness="{roughness}"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="mat"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """


@pytest.mark.parametrize("metallic", [0.0, 1.0])
@pytest.mark.parametrize("roughness", [0.0, 0.5, 1.0])
def test_white_furnace(scene, metallic, roughness):
  """A white surface in a uniform white environment reflects (nearly) exactly the environment."""
  white = np.ones((64, 128, 3), dtype=np.float32)
  s = scene(_sphere_xml(metallic, roughness), envmap=white, envmap_intensity=1.0, outputs=rw.Outputs.of("rgb", "hdr")).render()
  np.testing.assert_allclose(s.hdr()[0, 16, 16], 1.0, rtol=0.03)


def test_environment_orientation(scene):
  """Row 0 of the map is the zenith: a sky-red / ground-blue map lights an upward plane red."""
  image = np.zeros((64, 128, 3), dtype=np.float32)
  image[:32, :, 0] = 1.0
  image[32:, :, 2] = 1.0
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" roughness="1" specular="0"/></asset>
      <worldbody>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  c = scene(xml, envmap=image, envmap_intensity=1.0, outputs=rw.Outputs.of("rgb", "hdr")).render().hdr()[0, 16, 16]
  assert c[0] > 0.9 and c[2] < 0.05


def test_background_shows_environment_and_yaw_rotates_it(scene):
  image = np.zeros((64, 128, 3), dtype=np.float32)
  image[:, :64, 1] = 1.0  # left half (u < 0.5): the +y half of the horizon (Blender's layout, B10)
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <worldbody>
        <geom type="sphere" pos="0 -5 0" size="0.1"/>
        <camera name="look_y" pos="0 0 0" xyaxes="1 0 0 0 0 1" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  green = scene(xml, envmap=image, envmap_intensity=1.0, outputs=rw.Outputs.of("rgb", "hdr")).render().hdr()[0, 16, 16]
  rotated = (
    scene(xml, envmap=image, envmap_intensity=1.0, envmap_rotation=180.0, outputs=rw.Outputs.of("rgb", "hdr"))
    .render()
    .hdr()[0, 16, 16]
  )
  assert green[1] > 0.9
  assert rotated[1] < 0.05


@pytest.mark.parametrize("nworld", [1, 2])
def test_per_world_environment(scene, nworld):
  maps = [np.full((32, 64, 3), 0.25, np.float32), np.full((32, 64, 3), 1.0, np.float32)]
  s = scene(
    _sphere_xml(0.0, 1.0),
    nworld=nworld,
    envmap=maps,
    envmap_intensity=1.0,
    outputs=rw.Outputs.of("rgb", "hdr"),
    per_world=rw.PerWorld(env=["id"]),
  )
  s.r.per_world["env.id"].assign((np.arange(nworld) % 2).astype(np.int32))
  hdr = s.render().hdr()
  np.testing.assert_allclose(hdr[0, 16, 16], 0.25, rtol=0.03)
  if nworld == 2:
    np.testing.assert_allclose(hdr[1, 16, 16], 1.0, rtol=0.03)


_SKY = """
  <mujoco>
    <visual><headlight active="0"/></visual>
    <asset>
      <texture name="sky" type="skybox" builtin="flat" width="16" height="16" rgb1="1 1 1" rgb2="1 1 1"/>
      <material name="white" rgba="1 1 1 1" metallic="0" roughness="1"/>
    </asset>
    <worldbody>
      {light}
      <geom type="sphere" size="0.5" material="white"/>
      <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="{fovy}"/>
    </worldbody>
  </mujoco>
"""


def test_b56_skybox_is_background_only(scene):
  """B56: MuJoCo's skybox lights nothing (its renderers draw it as the background only): a white
  sphere under a white skybox and a black light stays black, the background is the skybox's white
  (a black light: a model without any light shows MuJoCo's no-light ambient)."""
  black = '<light pos="0 -3 0" dir="0 1 0" diffuse="0 0 0" specular="0 0 0" castshadow="false"/>'
  s = scene(_SKY.format(light=black, fovy=25), post=rw.settings.Post(tone_mapper="clamp")).render()
  rgb = s.rgb()[0]
  assert rgb[16, 16].max() < 0.02  # the sphere
  assert rgb[1, 1].min() > 0.98  # the skybox behind it
  assert not s.r.features.env_lit


def test_b38_image_light_lights_the_scene_scaled_by_its_intensity(scene):
  """B38: <light type="image" texture=...> lights the scene with its texture, radiance = texel x
  intensity [nits] (Filament's IndirectLight): a white sphere in a uniform white environment of
  radiance L reflects L (the white furnace)."""
  intensity = 2000.0
  light = f'<light type="image" texture="sky" intensity="{intensity}"/>'
  s = scene(_SKY.format(light=light, fovy=5), outputs=rw.Outputs.of("rgb", "hdr")).render()
  assert s.r.features.env_lit
  np.testing.assert_allclose(s.hdr()[0, 16, 16, :3], intensity, rtol=0.03)


def test_skybox_beside_an_image_light_is_the_background_only(scene):
  """A skybox beside an image light of another texture (the MuJoCo gallery's house_of_cards,
  arch/*, flex/mechanisms): the image light's texture lights the scene, the skybox shows behind
  it (two maps; the skybox one display-relative and not prefiltered)."""
  intensity = 2000.0
  xml = _SKY.format(light=f'<light type="image" texture="env" intensity="{intensity}"/>', fovy=5)
  xml = xml.replace(
    'rgb1="1 1 1" rgb2="1 1 1"/>',
    'rgb1="1 0 0" rgb2="1 0 0"/><texture name="env" type="cube" builtin="flat" width="16" height="16" rgb1="1 1 1" rgb2="1 1 1"/>',
    1,
  )
  s = scene(xml, outputs=rw.Outputs.of("rgb", "hdr"), post=rw.settings.Post(tone_mapper="clamp")).render()
  assert s.scene.environment.background == 1 and len(s.scene.environment.maps.maps) == 2
  np.testing.assert_allclose(s.hdr()[0, 16, 16, :3], intensity, rtol=0.03)  # lit by the white map
  wide = scene(xml.replace('fovy="5"', 'fovy="25"'), post=rw.settings.Post(tone_mapper="clamp")).render()
  np.testing.assert_allclose(wide.rgb()[0, 1, 1], (1.0, 0.0, 0.0), atol=0.02)  # the red skybox behind


def test_latlong_layout_is_blenders():
  """B10: u = 1/2 - atan2(y, x) / 2 pi (centre column +x, u = 0.25 is +y, row 0 the zenith).

  The old layout (u = 1/2 + atan2(y, x) / 2 pi) showed every panorama mirrored compared with
  Blender, three.js and Omniverse (once its half turn is applied).
  """
  from raywarp.core.shade.environment import latlong as device_latlong
  from raywarp.scene.images import latlong

  @wp.kernel
  def lookup(d: wp.array1d[wp.vec3], yaw: float, uv: wp.array1d[wp.vec2], back: wp.array1d[wp.vec3]):
    i = wp.tid()
    uv[i] = device_latlong.direction_to_uv(d[i], yaw)
    back[i] = device_latlong.uv_to_direction(device_latlong.direction_to_uv(d[i], 0.0))

  dirs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32)
  d = wp.array(dirs, dtype=wp.vec3)
  uv, back = wp.zeros(5, dtype=wp.vec2), wp.zeros(5, dtype=wp.vec3)
  wp.launch(lookup, dim=5, inputs=[d, 0.0], outputs=[uv, back])
  np.testing.assert_allclose(uv.numpy()[:4], [[0.5, 0.5], [0.25, 0.5], [0.0, 0.5], [0.75, 0.5]], atol=1e-6)
  assert uv.numpy()[4, 1] == pytest.approx(0.0, abs=1e-6)
  np.testing.assert_allclose(back.numpy(), dirs, atol=1e-6)
  # A yaw turns the map counter-clockwise about +z: its +x column then looks along +y.
  wp.launch(lookup, dim=5, inputs=[d, float(np.pi / 2)], outputs=[uv, back])
  assert uv.numpy()[1, 0] == pytest.approx(0.5, abs=1e-6)
  # The NumPy forms agree with the device ones.
  u, v = latlong.latlong_uv(latlong.latlong_directions(4, 8))
  np.testing.assert_allclose(u, np.broadcast_to((np.arange(8) + 0.5) / 8, (4, 8)), atol=1e-12)
  np.testing.assert_allclose(v, np.broadcast_to(((np.arange(4) + 0.5) / 4)[:, None], (4, 8)), atol=1e-12)


def test_rotate_latlong_and_extract_sun():
  from raywarp.core.shade.environment import latlong as device_latlong
  from raywarp.scene.images import latlong

  h, w = 64, 128
  img = np.ones((h, w, 3), np.float32)
  img[:, : w // 2] *= 2.0  # the +y half (u < 0.5) twice as bright
  # A yaw of pi swaps the halves.
  c = -1.0
  rot = np.array([[c, 0.0, 0.0], [0.0, c, 0.0], [0.0, 0.0, 1.0]])
  out = device_latlong.rotate_latlong(img, rot, device=wp.get_device())
  assert out[:, w // 2 + 4 :].mean() > 1.9 and out[:, : w // 2 - 4].mean() < 1.1
  # A 4-texel sun 30 degrees above the +x horizon holding most of the power.
  sky = np.ones((h, w, 3), np.float32)
  row, col = int(h * (60.0 / 180.0)), w // 2
  sky[row - 1 : row + 1, col - 1 : col + 1] = 2.0e4
  cut = latlong.extract_sun(sky)
  assert cut is not None
  rest, sun = cut
  assert rest.max() < 2.0 and sun.share > 0.5
  assert sun.direction @ np.array([np.cos(np.radians(30.0)), 0.0, np.sin(np.radians(30.0))]) > 0.99
  assert latlong.extract_sun(np.ones((h, w, 3), np.float32)) is None


def test_aces_rtx_is_narkowicz_at_rtx_exposure():
  from raywarp._base.types import ToneMapper
  from raywarp.core.post.grading import color

  # 800 nits at EV100 10: every curve receives 800 / 2^10, including RTX's Narkowicz fit.
  x = 800.0 / 1024.0
  want = x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)
  got = apply_display_transform(np.full(3, 800.0 / 1024.0), int(ToneMapper.ACES_RTX))
  np.testing.assert_allclose(got, want, rtol=1e-6)
  back = color.inverse_display_transform(got, int(ToneMapper.ACES_RTX), "cpu")
  np.testing.assert_allclose(back, 800.0 / 1024.0, rtol=1e-4)


def test_shipped_dfg_table_is_the_generator_output():
  """raywarp/data/dfg_lut.npz is what tools/make_dfg_lut.py computes (rerun it after a
  BRDF change).

  Equal to a few ulps, not bit-exact: Warp's CPU kernels call the platform's libm (pow, sin, cos,
  sqrt round differently on macOS arm64 and Linux x86-64), measured <= 2 ulps (1.2e-7) on 1.2 %
  of the texels. A stale table differs by far more: half the samples moves it by 1.5e-2, a BRDF
  change by more still. Every 16th roughness row (from the smoothest, the mirror-like corner, to
  the roughest): a stale table moves them all, and the whole table is 1e9 samples on one core.
  """
  import make_dfg_lut

  rows = [*range(0, ibl.DFG_LUT_SIZE, 16), ibl.DFG_LUT_SIZE - 1]
  np.testing.assert_allclose(ibl.dfg_table()[rows], make_dfg_lut.dfg_table(rows=rows), rtol=1e-6, atol=1e-9)
