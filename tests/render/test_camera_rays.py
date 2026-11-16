"""Per-pixel camera rays (set_camera_rays) and the first-hit guides (render_guides)."""

import dataclasses

import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, MjScene

import raywarp as rw
from raywarp.core.kernels.abi.output_layout import OUT_GUIDES

_XML = """
  <mujoco>
    <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".8 .8 .8" width="64" height="64"/>
      <material name="floor" texture="grid" texrepeat="4 4"/>
      <material name="m" rgba="0.8 0.3 0.2 1" metallic="0.5" roughness="0.4"/>
    </asset>
    <worldbody>
      <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
      <geom type="plane" size="2 2 0.1" material="floor"/>
      <geom type="sphere" pos="0 0 0.3" size="0.3" material="m"/>
      <geom type="box" pos="0.6 0.2 0.2" size="0.15 0.2 0.2" rgba="0.2 0.8 0.3 1"/>
      <camera name="front" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
      <camera name="side" pos="1.5 -1.5 1.2" xyaxes="0.7 0.7 0 -0.3 0.3 0.9" fovy="70"/>
    </worldbody>
  </mujoco>
"""
_RES = (24, 16)
_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


def pinhole_rays(width: int, height: int, fovy_deg: float, offset: float = 0.5) -> np.ndarray:
  """(height, width, 2, 3) camera-frame rays of a pinhole camera through (px, py) + offset."""
  h = np.tan(np.radians(fovy_deg) / 2.0)
  u = (np.arange(width) + offset) / width - 0.5
  v = (np.arange(height) + offset) / height - 0.5
  d = np.stack(np.broadcast_arrays(u[None, :] * 2.0 * h * width / height, -v[:, None] * 2.0 * h, -np.ones((1, 1))), axis=-1)
  d /= np.linalg.norm(d, axis=-1, keepdims=True)
  return np.stack([np.zeros_like(d), d], axis=2).astype(np.float32)


def _pinhole_bundle(fovys: tuple[float, ...] = (50.0, 70.0)) -> np.ndarray:
  """(ncam, H, W, 2, 3) pinhole rays of the two cameras."""
  return np.stack([pinhole_rays(*_RES, fovy) for fovy in fovys])


def _render(
  scene: MakeScene, rays: list[np.ndarray] | None = None, nworld: int = 2, aa: rw.settings.AntiAliasing | None = None
) -> tuple[MjScene, np.ndarray, np.ndarray, np.ndarray]:
  """(scene, *`_flat`) of a render of rgb, depth and seg (`aa`: None, the default)."""
  s = scene(
    _XML,
    nworld=nworld,
    cam_res=_RES,
    outputs=rw.Outputs.of("rgb", "depth", "seg"),
    rays=rays,
    aa=aa or rw.settings.AntiAliasing(),
  )
  s.render()
  return (s, *_flat(s))


def _flat(s: MjScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(rgb channels, depth, seg) of every camera, flattened to (nworld, ncam * h * w, ...)."""
  out = s.out
  nworld = s.r.nworld
  rgba = out.rgb
  if rgba is None or out.depth is None or out.seg is None:
    raise AssertionError("rgb, depth and seg must be on")
  # .copy(): CPU arrays share their memory with .numpy().
  rgb8 = rgba.numpy().reshape(nworld, -1, 4)[..., :3].astype(np.int32).copy()
  return rgb8, out.depth.numpy().reshape(nworld, -1).copy(), out.seg.numpy().reshape(nworld, -1).copy()


def _ray_scale(offset: float, fovys=(50.0, 70.0)) -> np.ndarray:
  """Distance along the ray per unit of planar depth, both cameras concatenated."""
  return np.concatenate([1.0 / -pinhole_rays(*_RES, f, offset)[..., 1, 2].ravel() for f in fovys])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("aa", [rw.settings.AntiAliasing(), rw.settings.AntiAliasing(spp=2)], ids=["realtime", "supersampled"])
def test_pinhole_rays_match_pinhole_cameras(scene: MakeScene, aa: rw.settings.AntiAliasing, device: str) -> None:
  with wp.ScopedDevice(device):
    _, rgb, depth, seg = _render(scene, aa=aa)
    _, rgb_rays, dist_rays, seg_rays = _render(scene, rays=list(_pinhole_bundle()), aa=aa)
  diff = np.abs(rgb - rgb_rays).max(axis=-1)
  # Within 1/255; with CUDA fast math a few texels at checker edges may round 2/255 apart.
  assert diff.max() <= 2 and np.mean(diff <= 1) > 0.99
  np.testing.assert_array_equal(seg, seg_rays)
  # Geometry observations use the unjittered center even when color is supersampled. Bundle
  # depth is radial and pinhole depth is planar, so convert using that same center direction.
  np.testing.assert_allclose(dist_rays, depth * _ray_scale(0.5), rtol=1e-4, atol=1e-6)


def test_rays_are_updated(scene):
  w, h = _RES

  def first_camera(array: np.ndarray) -> np.ndarray:
    return array[0, : w * h].reshape(h, w, *array.shape[2:])

  with wp.ScopedDevice("cpu"):
    _, rgb, depth, seg = _render(scene, nworld=1)
    rays = _pinhole_bundle()
    s, rgb_rays, _, _ = _render(scene, rays=list(rays), nworld=1)
    assert np.abs(rgb_rays - rgb).max() <= 1
    # Mirrored rays (x -> -x) mirror the image.
    flipped = rays.copy()
    flipped[..., 0] *= -1.0
    s.r.set_camera_rays([flipped[0], flipped[1]])
    s.render()
    _, depth_f, seg_f = _flat(s)
    np.testing.assert_array_equal(first_camera(seg_f)[:, ::-1], first_camera(seg))
    scale = first_camera(_ray_scale(0.5)[None])
    np.testing.assert_allclose(first_camera(depth_f)[:, ::-1], first_camera(depth) * scale, rtol=1e-4)


def test_rays_without_direction_miss(scene):
  rays = _pinhole_bundle()
  rays[:, :6, :, 1] = 0.0  # the top 6 rows of both cameras have no ray
  with wp.ScopedDevice("cpu"):
    _, _, _, reference = _render(scene, nworld=1)
    _, _, depth, seg = _render(scene, rays=list(rays), nworld=1)
  w, h = _RES
  for cam in range(2):
    pixels = slice(cam * w * h, (cam + 1) * w * h)
    image_seg = seg[0, pixels].reshape(h, w)
    assert np.all(image_seg[:6] == -1) and np.all(depth[0, pixels].reshape(h, w)[:6] == 0.0)
    np.testing.assert_array_equal(image_seg[6:], reference[0, pixels].reshape(h, w)[6:])


def test_parallel_rays_measure_distance_from_their_origin(scene):
  # Orthographic bundle: origins on a 1 m grid in front of the camera, all looking along -z.
  w, h = _RES
  x = ((np.arange(w) + 0.5) / w - 0.5) * 1.5
  y = (0.5 - (np.arange(h) + 0.5) / h) * 1.0
  origin = np.stack(np.broadcast_arrays(x[None, :], y[:, None], np.full((1, 1), -0.25)), axis=-1)
  direction = np.broadcast_to([0.0, 0.0, -1.0], origin.shape)
  ortho = np.stack([origin, direction], axis=2).astype(np.float32)
  xml = _XML.replace('<camera name="side" pos="1.5 -1.5 1.2" xyaxes="0.7 0.7 0 -0.3 0.3 0.9" fovy="70"/>', "")
  xml = xml.replace('pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9" fovy="50"', 'pos="0 0 3" xyaxes="1 0 0 0 1 0"')
  with wp.ScopedDevice("cpu"):
    s = scene(xml, nworld=1, cam_res=_RES, outputs=rw.Outputs.of("depth", "seg"), rays=[ortho])
    s.render()
  depth = s.depth()[0]
  seg = s.seg()[0]
  floor = seg == 0
  assert floor.any()
  # Camera at z = 3, origins 0.25 m in front of it: the floor (z = 0) is 2.75 m along every ray.
  np.testing.assert_allclose(depth[floor], 2.75, atol=1e-4)
  sphere = seg == 1
  np.testing.assert_allclose(depth[sphere].min(), 3.0 - 0.25 - 0.6, atol=0.02)


@pytest.mark.parametrize("device", _DEVICES)
def test_guides(scene, device):
  with wp.ScopedDevice(device):
    s = scene(_XML, nworld=1, cam_res=_RES, outputs=rw.Outputs.of("rgb", "seg", "albedo", "normal")).render()
  seg = s.seg()[0]
  albedo_np = s.albedo()[0]
  normal_np = s.normal()[0]
  box = seg == 2
  assert box.any()
  np.testing.assert_allclose(albedo_np[box], np.broadcast_to([0.2, 0.8, 0.3], albedo_np[box].shape), atol=1e-5)
  floor = seg == 0
  np.testing.assert_allclose(normal_np[floor], np.broadcast_to([0.0, 0.0, 1.0], normal_np[floor].shape), atol=1e-5)
  sphere = seg == 1
  np.testing.assert_allclose(np.linalg.norm(normal_np[sphere], axis=-1), 1.0, atol=1e-4)


def test_guides_only_when_requested(scene):
  with wp.ScopedDevice("cpu"):
    s = scene(_XML, cam_res=_RES)
    assert s.out.albedo is None and s.out.normal is None
    assert not s.r._buffers.layout.bits & OUT_GUIDES


def test_ray_validation(scene):
  with wp.ScopedDevice("cpu"):
    s = scene(_XML, cam_res=_RES, rays=_pinhole_bundle())
    with pytest.raises(ValueError, match="each of the 2 cameras"):
      s.r.set_camera_rays(_pinhole_bundle()[:1])
    with pytest.raises(ValueError, match=r"shape \(16, 24, 2\)"):
      s.r.set_camera_rays([pinhole_rays(24, 16, 50.0), pinhole_rays(16, 16, 50.0)])
    for bad in (np.nan, np.inf):
      rays = _pinhole_bundle()
      rays[1, 3, 5, 1, 2] = bad
      with pytest.raises(ValueError, match=r"finite: NaN / inf at 1 pixels, first \(row, column\) \(3, 5\)"):
        s.r.set_camera_rays(list(rays))
    with pytest.raises(ValueError, match="dtype vec3"):
      wrong = wp.zeros((16, 24, 2), dtype=wp.vec4)
      s.r.set_camera_rays([wrong, wrong])  # vec4 rays
    pinhole = scene(_XML, cam_res=_RES)
    assert pinhole.r.camera_rays is None
    with pytest.raises(ValueError, match="CameraSpec"):
      pinhole.r.set_camera_rays(_pinhole_bundle())
    with pytest.raises(ValueError, match="every camera"):
      cams = [dataclasses.replace(pinhole.scene.cameras["front"], rays=_pinhole_bundle()[0]), "side"]
      rw.Renderer(pinhole.scene, nworld=1, cameras=cams, resolution=_RES)


@pytest.mark.parametrize("bad", ["origin", "direction"])
def test_warp_rays_with_nan_are_rejected_and_rendered_as_misses(scene, bad):
  """Warp-array rays were not checked (a host read): a NaN origin with a finite direction reached
  the traversal. `set_camera_rays` now counts them on the device (no host read) and writes them
  as pixels without a ray (a miss: seg -1, depth 0); the validation word reports them one frame
  late; every other pixel renders as before."""
  with wp.ScopedDevice("cpu"):
    s, _, depth0, seg0 = _render(scene, rays=list(_pinhole_bundle()), nworld=1)
    rays = _pinhole_bundle()
    rays[1, 12, 12, 0 if bad == "origin" else 1, 1] = np.nan
    images = [wp.array(rays[c], dtype=wp.vec3) for c in range(len(rays))]
    s.r.set_camera_rays(images)
    s.render()
    _, depth, seg = _flat(s)
    with pytest.raises(ValueError, match="camera rays must be finite"):
      s.render()
  w, h = _RES
  pixel = w * h + 12 * w + 12  # camera 1, row 12, column 12 (the floor)
  assert seg[0, pixel] == -1 and depth[0, pixel] == 0.0 and np.isfinite(depth).all()
  others = np.arange(depth.shape[1]) != pixel
  np.testing.assert_array_equal(seg[0, others], seg0[0, others])
  np.testing.assert_array_equal(depth[0, others], depth0[0, others])
