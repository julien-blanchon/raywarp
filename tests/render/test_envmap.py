"""Environment maps beyond float16's range (B42): physical skies whose sun reaches ~1e8.

Every adapter goes through `EnvironmentMap`, which stores such maps divided by a scale and
multiplies it back at lookup; before, MuJoCo / Newton cast them to float16 directly (inf texels,
NaN pixels: black renders).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import warp as wp
from conftest import NO_GI, required

import raywarp as rw
from raywarp.core.shade.environment import envmap
from raywarp.core.shade.environment.latlong import uv_to_direction
from raywarp.scene.images import latlong

SUN = 2.0e8


def _sun_sky(sky: float = 1000.0) -> np.ndarray[[32, 64, 3], np.dtype[np.float32]]:
  """A uniform sky of `sky` radiance with a 2x2-texel sun of SUN (a physical sky's dynamic range).

  The sun is 40 degrees up towards -y (u ~ 0.67: longitude -60 degrees), on the cameras' side.
  """
  image = np.full((32, 64, 3), sky, np.float32)
  image[8:10, 42:44] = SUN
  return image


_SPHERE = """
  <mujoco>
    <visual><headlight active="0"/></visual>
    <asset><material name="mat" rgba="1 1 1 1" metallic="0" roughness="1"/></asset>
    <worldbody>
      <geom type="sphere" size="0.5" material="mat"/>
      <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="{fovy}"/>
    </worldbody>
  </mujoco>
"""


def test_maps_beyond_float16_are_stored_scaled():
  """The stored maps peak at MAX_STORED_RADIANCE; their SH times the scale is the map's SH."""
  image = _sun_sky()
  envs = envmap.build_environments([image, np.full((32, 64, 3), 0.5, np.float32)], device=wp.get_device("cpu"))
  scale = envs.scale.numpy()
  assert scale[0] == pytest.approx(SUN / envmap.MAX_STORED_RADIANCE, rel=1e-6)
  assert scale[1] == 1.0
  sh = envs.sh.numpy()
  assert np.all(np.isfinite(sh))
  np.testing.assert_allclose(sh[0] * scale[0], envmap.fold_sh9(envmap._sh9_irradiance(image)), rtol=1e-4, atol=1e-2)


def test_white_furnace_beyond_float16(scene):
  """A white sphere in a uniform 1e6 sky (beyond float16) reflects 1e6, as at radiance 1."""
  white = np.full((64, 128, 3), 1.0e6, np.float32)
  s = scene(_SPHERE.format(fovy=5), envmap=white, envmap_intensity=1.0, outputs=rw.Outputs.of("rgb", "hdr")).render()
  np.testing.assert_allclose(s.hdr()[0, 16, 16], 1.0e6, rtol=0.03)


def test_physical_sky_renders_finite(scene):
  """A sky with a 2e8 sun disc gives finite pixels (it was 78% NaN) and a lit, sky-colored
  background; the sphere is lit by the sun's share of the irradiance on top of the sky's."""
  s = scene(_SPHERE.format(fovy=40), envmap=_sun_sky(), envmap_intensity=1.0, outputs=rw.Outputs.of("rgb", "hdr")).render()
  hdr = s.hdr()[0]
  assert np.all(np.isfinite(hdr))
  np.testing.assert_allclose(hdr[0, 0], 1000.0, rtol=0.03)  # background corner: the sky
  assert hdr[16, 16].min() > 1000.0


def test_host_array_environment_beyond_float16():
  """The SceneBuilder path (USD / Newton arrays) keeps the authored intensity: a floor of albedo
  0.8 under a uniform 1e6 sky reflects 0.8e6. A 4 m floor: its centre sees the whole sky (the
  sky-occlusion grid, the CPU test's cost, grows with the floor's area)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_weight=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(2.0, 2.0, 0.05), material=m, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
  sky = np.full((32, 64, 4), 1.0e6, np.float32)
  b.add_environment(rw.scene.EnvironmentLight(texture=b.add_texture(sky, colorspace="linear", wrap=("repeat", "clamp"))))
  with wp.ScopedDevice("cpu"):
    gpu = rw.upload(b.build())
    assert gpu.environment.intensity == 1.0
    assert gpu.environment.maps.scale.numpy()[0] > 1.0
    settings = rw.RenderSettings(
      gi=NO_GI,
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal="software"),
      environment=rw.settings.Environment(enabled=True),
    )
    r = rw.Renderer(gpu, nworld=1, cameras=[rw.CameraSpec(fovy=1.0)], resolution=(2, 2), settings=settings)
    r.camera_pose.assign(np.array([[[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]]], np.float32))
    r.render()
    hdr = required(r.outputs.hdr).numpy()[0, 0, ..., :3]
  np.testing.assert_allclose(hdr.reshape(-1, 3).mean(0), 0.8e6, rtol=0.03)


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_non_finite_map_raises(bad):
  """An overflowed (inf) or NaN texel is refused at upload instead of rendering black."""
  image = np.ones((8, 16, 3), np.float32)
  image[2, 3, 1] = bad
  with pytest.raises(ValueError, match="NaN or infinite"):
    envmap.EnvironmentMap(image, lighting=True, device=wp.get_device("cpu"))


def test_sh_irradiance_facing_away_from_a_sun_keeps_the_sky():
  """B62: a uniform sky of 1000 with a 2e8 sun. Facing away from the sun only the sky lights a
  surface (Lambertian radiance 1000); the SH9 kernel's negative lobe times the sun drove it below
  zero (clamped black). The windowed SH keeps it at the sky's level, and the sun side bright."""

  image = _sun_sky()
  sh = envmap._sh9_irradiance(image).astype(np.float64)
  d = latlong.latlong_directions(32, 64)
  sun = d[8:10, 42:44].reshape(-1, 3).mean(0)
  sun /= np.linalg.norm(sun, keepdims=True)
  # Directions 90-180 degrees from the sun: the hemisphere that never sees it.
  dirs = d.reshape(-1, 3)
  away = dirs[dirs @ sun < -0.05]
  radiance = np.maximum(envmap._sh9_basis(away).T @ sh, 0.0)
  assert radiance.min() > 0.9 * 1000.0, radiance.min()
  assert (envmap._sh9_basis(sun).T @ sh).min() > 1.0e5


def test_negative_radiance_is_clamped_with_a_warning():
  """Negative texels (out-of-gamut colors) were uploaded as is: the SH and the prefilter subtracted
  light. They are clamped to 0, with a warning, and the caller's image is left as it was."""
  image = np.full((4, 8, 3), 0.5, np.float32)
  image[1, 2] = (-2.0, 0.5, 0.5)
  with pytest.warns(RuntimeWarning, match="1 negative radiance components clamped"):
    clamped = envmap.EnvironmentMap(image, lighting=True, device=wp.get_device("cpu"))
  reference = image.copy()
  reference[1, 2, 0] = 0.0
  expected = envmap.EnvironmentMap(reference, lighting=True, device=wp.get_device("cpu"))
  np.testing.assert_array_equal(clamped.sh, expected.sh)
  assert image[1, 2, 0] == -2.0


# ---------------------------------------------------------------------------------------------
# The GGX prefilter of an environment map under Warp debug mode.


_DEBUG_PREFILTER = textwrap.dedent(
  """
  import os
  import numpy as np
  import warp as wp

  # One band: Warp debug-mode CPU launches from several threads crash (post_release/furnace).
  os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
  wp.config.mode = "debug"
  wp.config.verify_fp = True
  # verify_fp checks a store's destination too: zero-filled `wp.empty` (as conftest's
  # --warp-verify-fp), or uninitialised memory aborts at random.
  wp.empty = wp.zeros
  wp.empty_like = wp.zeros_like
  wp.set_device("cpu")
  from raywarp.core.shade.environment import envmap

  stack = envmap.EnvironmentMap(np.ones((8, 16, 3), np.float32), lighting=True, device=wp.get_device("cpu")).prefiltered
  print("ok", stack.depth)
  """
)


@pytest.mark.skipif(sys.platform != "linux", reason="pins the subprocess to one core (sched_setaffinity)")
def test_prefilter_runs_clean_under_warp_debug_mode():
  """The GGX prefilter of an environment map passes Warp's debug checks (its mirror level
  divided by a zero D(h) denominator: a div(0, 0) assertion)."""
  env = {**os.environ, "RAYWARP_CACHE_DIR": ""}  # no bake cache: the prefilter runs
  out = subprocess.run(
    [sys.executable, "-c", _DEBUG_PREFILTER], check=False, env=env, capture_output=True, text=True, timeout=900
  )
  assert out.returncode == 0 and "div(" not in out.stdout + out.stderr, out.stdout[-2000:] + out.stderr[-2000:]
  assert "ok" in out.stdout


# ---------------------------------------------------------------------------------------------
# Sharp environment angular footprints, without a scene or expensive bake.


@pytest.fixture(params=["cpu", "cuda:0"])
def sampling_device(request: pytest.FixtureRequest) -> wp.Device:
  if request.param == "cuda:0" and not wp.is_cuda_available():
    pytest.skip("CUDA sampling parity requires a GPU")
  return wp.get_device(request.param)


@wp.kernel
def _specular(
  # Model:
  maps: wp.array1d[wp.Texture2D],
  stack: wp.Texture3D,
  # Out:
  sharp: wp.array3d[wp.vec3],
  filtered: wp.array3d[wp.vec3],
  reference: wp.array3d[wp.vec3],
) -> None:
  world, rough, index = wp.tid()
  radiance = maps[world]
  direction = uv_to_direction(wp.vec2((float(index) + 0.5) / float(radiance.width), 0.5))
  pr = wp.where(rough == 0, 0.0, wp.where(rough == 1, 0.07, 0.5))
  sharp[world, rough, index] = envmap.sample_specular(radiance, stack, direction, 0.0, pr, 0.0)
  # Eight texels of the 128-wide map, two of the 32-wide map: the same angular cone.
  filtered[world, rough, index] = envmap.sample_specular(radiance, stack, direction, 0.0, pr, wp.pi / 8.0)
  lod = wp.where(world == 0, 1.0, 3.0)
  reference[world, rough, index] = envmap.sample_radiance(radiance, direction, 0.0, lod)


def test_sharp_specular_filters_cone_using_selected_map_resolution(sampling_device: wp.Device) -> None:
  textures: list[wp.Texture2D] = []
  widths: tuple[int, ...] = (32, 128)
  for width in widths:
    image = np.ones((16, width, 4), dtype=np.float32)
    image[:, :, :3] = (np.arange(width) // 4 % 2)[None, :, None]
    textures.append(
      wp.Texture2D(
        image,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode_u=wp.TextureAddressMode.WRAP,
        num_mip_levels=0,
        mip_filter_mode=wp.TextureFilterMode.LINEAR,
        device=sampling_device,
      )
    )
  stack = wp.Texture3D(np.full((8, 1, 1, 4), 0.25, np.float32), device=sampling_device)
  maps = wp.array(textures, dtype=wp.Texture2D, device=sampling_device)
  sharp = wp.empty((2, 3, 8), dtype=wp.vec3, device=sampling_device)
  filtered, reference = wp.empty_like(sharp), wp.empty_like(sharp)
  wp.launch(_specular, dim=(2, 3, 8), inputs=[maps, stack], outputs=[sharp, filtered, reference], device=sampling_device)
  raw, actual, expected = sharp.numpy(), filtered.numpy(), reference.numpy()
  np.testing.assert_allclose(actual[:, 0], expected[:, 0], atol=2e-6)
  np.testing.assert_allclose(actual[:, 1], expected[:, 1] * 0.51 + 0.25 * 0.49, atol=2e-6)
  np.testing.assert_allclose(actual[:, 2], 0.25, atol=2e-6)  # rough lobe is unchanged
  np.testing.assert_allclose(actual[1, 0], 0.5, atol=2e-6)
  assert actual[0, 0].std() > 0.2  # smaller map must not inherit the larger map's mip level
  # Direction -> UV roundtrip can move a 128-wide bilinear sample a few float32 ulps.
  np.testing.assert_allclose(raw[:, 0, :, 0], np.broadcast_to([0, 0, 0, 0, 1, 1, 1, 1], (2, 8)), atol=2e-5)


def test_sh_irradiance_of_a_sky_over_a_black_ground_is_not_windowed():
  """A sky brightening towards the zenith over a black ground rings slightly below zero under the
  horizon, where nothing is lit; windowing the whole map lowered the upward irradiance 7.9 %.
  Only peaks are windowed: the upward irradiance matches the cosine integral."""

  h, w = 128, 256
  image = np.zeros((h, w, 3), np.float32)
  image[: h // 2] = 4000.0
  image *= np.linspace(2.0, 0.0, h, dtype=np.float32)[:, None, None]
  d = latlong.latlong_directions(h, w)
  exact = (image[..., 0] * np.maximum(d[..., 2], 0.0) * latlong.texel_solid_angles(h, w)).sum() / np.pi
  sh = envmap._sh9_irradiance(image).astype(np.float64)
  up = (envmap._sh9_basis(np.array([0.0, 0.0, 1.0])).T @ sh)[0]
  assert abs(up / exact - 1.0) < 0.01, up / exact
