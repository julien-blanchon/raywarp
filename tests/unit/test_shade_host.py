"""Host-side helpers of `core/shade` (wave C lane C2 fixes): explicit devices, environment-map
validation at the boundary, the OpenCV undistortion past the lens model's fold."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp
from conftest import no_default_device

from raywarp._base.types import ToneMapper
from raywarp.core.kernels.abi.structs import lens_coefficients
from raywarp.core.post.grading import color
from raywarp.core.shade.environment import envmap
from raywarp.core.trace.rays import bundle, camera
from raywarp.scene.images import latlong


@pytest.fixture
def no_scoped_device(monkeypatch):
  """`wp.ScopedDevice` raises and Warp's default device fails when used (`no_default_device`):
  library code must name its devices (Warp's default device is process-wide, a scoped switch races
  with other threads)."""

  def scoped(*_args, **_kwargs):
    raise AssertionError("library code entered wp.ScopedDevice")

  wp.init()
  monkeypatch.setattr(wp, "ScopedDevice", scoped)
  with no_default_device():
    yield


def test_inverse_display_transform_names_its_device(no_scoped_device):
  """The inverse tone mapping runs on the device it names (it entered no `wp.ScopedDevice`)."""
  x = color.inverse_display_transform(np.full((2, 3), 0.5), int(ToneMapper.PBR_NEUTRAL), "cpu")
  assert x.shape == (2, 3) and np.all(np.isfinite(x))


def test_load_camera_rays_names_its_device(no_scoped_device):
  """The bundle buffers are allocated on the given device (they entered `wp.ScopedDevice`)."""
  image = np.zeros((2, 3, 2, 3), np.float32)
  image[..., 1, 2] = -1.0
  cpu = wp.get_device("cpu")
  invalid = wp.zeros(1, dtype=wp.int32, device=cpu)
  buffers = bundle.load_camera_rays([image, image], width=3, height=2, device=cpu, invalid=invalid)
  assert all(b.device == wp.get_device("cpu") for b in buffers)
  np.testing.assert_array_equal(buffers[2].numpy(), [0, 6])
  np.testing.assert_array_equal(buffers[1].numpy()[:, 2], -1.0)


@pytest.mark.parametrize(
  ("shape", "dtype"),
  [((1, 1, 3), np.float32), ((0, 8, 3), np.float32), ((4, 8), np.float32), ((4, 8, 2), np.float32), ((4, 8, 3), np.uint8)],
)
def test_invalid_environment_maps_fail_at_the_boundary(shape, dtype):
  """A map that cannot be uploaded fails with the expected shape / dtype, not deep in Warp or
  NumPy (a 1 x 1 map raised "Texture height must be a positive integer", an 8-bit map was taken
  as radiance 0..255)."""
  with pytest.raises(ValueError, match="environment map: expected"):
    envmap.EnvironmentMap(np.ones(shape, dtype), lighting=True, device=wp.get_device("cpu"))


def test_single_row_maps_beyond_the_width_cap_are_downsampled():
  """`downsample` halves only the width of a single row (a 2 x 8192 map reached a 1-row image,
  then failed to reshape it)."""
  image = np.ones((2, 8192, 3), np.float32)
  e = envmap.EnvironmentMap(image, lighting=False, device=wp.get_device("cpu"))
  assert (e.radiance.width, e.radiance.height) == (envmap.MAX_RADIANCE_WIDTH, 1)


def test_build_environments_checks_its_lists():
  """No map and mismatched lighting flags fail with a message (were max() of an empty sequence and
  zip()'s length error)."""
  with pytest.raises(ValueError, match="at least one map"):
    envmap.build_environments([], device=wp.get_device("cpu"))
  with pytest.raises(ValueError, match="2 lighting flags for 1 maps"):
    envmap.build_environments([np.ones((4, 8, 3), np.float32)], [True, False], device=wp.get_device("cpu"))


def test_environment_textures_live_on_the_named_device(no_scoped_device):
  """Every texture and table of the maps is on the device given (was Warp's current device)."""
  envs = envmap.build_environments([np.ones((4, 8, 3), np.float32)], [False], device=wp.get_device("cpu"))
  assert all(a.device == wp.get_device("cpu") for a in (envs.radiance, envs.prefiltered, envs.sh, envs.scale))
  assert envmap.empty_environments(device=wp.get_device("cpu")).sh.device == wp.get_device("cpu")


def test_sh9_of_a_narrow_map_is_its_piecewise_constant_texels():
  """A map narrower than 2:1 projects as its texels resampled to the quadrature's 512 columns,
  like the same texels repeated by hand (a 256 x 32 map read its texel centres only: 0.25 % off,
  and `_blackens` pooled it into an empty probe grid)."""
  h, w = 256, 32
  image = np.full((h, w, 3), 0.01, np.float32)
  image[latlong.latlong_directions(h, w) @ np.array([1.0, 0.0, 0.0]) > np.cos(0.35)] = 20.0
  narrow = envmap._sh9_irradiance(image)
  wide = envmap._sh9_irradiance(np.repeat(image, 16, axis=1))
  np.testing.assert_allclose(narrow, wide, rtol=0.0, atol=1.0e-6 * float(np.abs(wide).max()))


def test_map_sun_is_a_frozen_value():
  """`latlong.extract_sun`'s record is a value."""
  h, w = 32, 64
  sky = np.ones((h, w, 3), np.float32)
  sky[4, 10] = 1.0e6
  cut = latlong.extract_sun(sky)
  assert cut is not None
  with pytest.raises(dataclasses.FrozenInstanceError):
    cut[1].angle = 0.0  # pyrefly: ignore[read-only]  # the assignment under test


@wp.kernel
def _undistort(k: lens_coefficients, points: wp.array[wp.vec2], out: wp.array[wp.vec2]):
  i = wp.tid()
  out[i] = camera.undistort(k, points[i][0], points[i][1])


def _undistorted(k1: float, points: list[tuple[float, float]]) -> np.ndarray:
  k = lens_coefficients(k1, *([0.0] * 11))
  p = wp.array(np.asarray(points, np.float32), dtype=wp.vec2, device="cpu")
  out = wp.empty_like(p)
  wp.launch(_undistort, dim=len(points), inputs=[k, p], outputs=[out], device="cpu")
  return out.numpy()


def test_undistort_inverts_distort_inside_the_fold():
  """Points with a preimage are undistorted to it (unchanged by the fold guard)."""
  u = _undistorted(-0.3, [(0.2, 0.1), (0.5, 0.4)])
  r2 = (u * u).sum(1, keepdims=True)
  np.testing.assert_allclose(u * (1.0 - 0.3 * r2), [(0.2, 0.1), (0.5, 0.4)], atol=1e-4)


def test_undistort_stops_at_the_lens_fold():
  """Past the fold of a strong barrel distortion (no preimage), OpenCV's guard returns the
  distorted point (the iteration flipped it through the centre: (-0.214, -0.171), a mirrored
  ray)."""
  np.testing.assert_allclose(_undistorted(-0.5, [(0.5, 0.4)]), [(0.5, 0.4)])
