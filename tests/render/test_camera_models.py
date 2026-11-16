"""Camera lens models against their closed forms: fisheye (equidistant f-theta and OpenCV's
Kannala-Brandt, fields beyond 180 degrees), OpenCV distortion, per-world lens coefficients, and
per-camera output masks.

The kernel tests cast each pixel's ray (`camera.make_camera_ray`) and project it back
(`camera.project`, the inverse used by motion vectors); the render tests read
each pixel's ray back from the "position" output (the hit point seen from the eye).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp
from conftest import required
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp._base.types import ProjectionType
from raywarp.core.kernels.abi.structs import Lens, lens_coefficients
from raywarp.core.trace.rays import camera
from raywarp.core.trace.rays.primary import DISTORTION, PINHOLE

# A closed room 4 m wide around the camera (six thin slabs: every direction hits a wall).
ROOM = """
<mujoco>
  <visual><map znear="0.01" zfar="50"/></visual>
  <statistic extent="1"/>
  <worldbody>
    <light pos="0.5 0.5 1.5" dir="0 0 -1"/>
    <geom name="px" type="box" pos="2 0 0" size="0.01 2 2"/>
    <geom name="nx" type="box" pos="-2 0 0" size="0.01 2 2"/>
    <geom name="py" type="box" pos="0 2 0" size="2 0.01 2"/>
    <geom name="ny" type="box" pos="0 -2 0" size="2 0.01 2"/>
    <geom name="pz" type="box" pos="0 0 2" size="2 2 0.01"/>
    <geom name="nz" type="box" pos="0 0 -2" size="2 2 0.01"/>
    <geom name="ball" type="sphere" pos="0.4 1.2 0.2" size="0.25"/>
    <camera name="cam" pos="0.1 -0.2 0.05" xyaxes="1 0 0 0 0 1" fovy="60"/>
  </worldbody>
</mujoco>
"""
RES = (48, 40)
# OpenCV fisheye calibration (pixels) of a RES image: a wide lens with mild Kannala-Brandt terms.
K_FISHEYE = np.array([[14.0, 0.0, 23.2], [0.0, 13.5, 19.6], [0.0, 0.0, 1.0]])
D_FISHEYE = (0.05, -0.01, 0.002, -0.0003)
# OpenCV's rational + thin prism model (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4), of the
# size of a wide-angle depth camera's calibration (an Azure-Kinect-like rational fit).
D_RATIONAL = (0.2, -0.02, 0.001, -0.0005, -0.001, 0.3, 0.01, -0.002, 0.002, -0.0005, 0.001, 0.0003)


def kb_theta_d(theta: np.ndarray, k: tuple[float, ...]) -> np.ndarray:
  """Kannala-Brandt distorted angle (float64)."""
  t2 = theta * theta
  return theta * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


def kb_theta(theta_d: np.ndarray, k: tuple[float, ...]) -> np.ndarray:
  """The angle of distorted angle `theta_d` (float64 Newton, converged)."""
  theta = theta_d.copy()
  for _ in range(60):
    t2 = theta * theta
    slope = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
    theta = theta - (kb_theta_d(theta, k) - theta_d) / slope
  return theta


def fisheye_rays(k_matrix: np.ndarray, k: tuple[float, ...], res: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
  """(theta, azimuth) of each pixel centre (OpenCV pixel (u, v) = (px, py)) of a fisheye (float64,
  azimuth in OpenCV's image axes: x right, y down)."""
  u, v = np.meshgrid(np.arange(res[0]).astype(np.float64), np.arange(res[1]).astype(np.float64))
  xd, yd = (u - k_matrix[0, 2]) / k_matrix[0, 0], (v - k_matrix[1, 2]) / k_matrix[1, 1]
  return kb_theta(np.hypot(xd, yd), k), np.arctan2(yd, xd)


# --- kernel-level: ray -> project round trip ------------------------------------------------------


@wp.kernel
def _round_trip(
  # In:
  lens: wp.array1d[Lens],
  coefficients: wp.array1d[lens_coefficients],
  fovy: float,
  intrinsics: wp.vec4,
  width: int,
  height: int,
  subdiv: int,
  # Out:
  direction: wp.array1d[wp.vec3],
  max_dist: wp.array1d[float],
  image: wp.array1d[wp.vec2],
  valid: wp.array1d[bool],
):
  """Casts the ray of sub-pixel cell i (pixel i // subdiv² ...) and projects a point on it back."""
  i = wp.tid()
  n = width * subdiv
  pix = camera.Pixel()
  pix.lens = lens[0]
  pix.width = width
  pix.height = height
  pix.fovy = fovy
  pix.intrinsics = intrinsics
  pix.distortion = coefficients[0]
  pix.rot = wp.identity(3, dtype=float)
  pix.pos = wp.vec3(0.0)
  x = i % n
  y = i // n
  pix.px = x // subdiv
  pix.py = y // subdiv
  _origin, d, far, _axis = wp.static(camera.make_camera_ray(distortion=True))(pix, subdiv, x % subdiv, y % subdiv)
  direction[i] = d
  max_dist[i] = far
  uv, ok = camera.project(pix, camera.frustum(pix), d * 3.7)
  image[i] = uv
  valid[i] = ok


def _cast(spec: rw.CameraSpec, res: tuple[int, int], subdiv: int = 1) -> dict[str, np.ndarray]:
  """Every sub-pixel ray of `spec` (as `_round_trip`) on the CPU."""
  lens = Lens()
  lens.projection = int(ProjectionType[spec.projection.upper()])
  lens.sensorsize = wp.vec2(*(spec.sensor_size or (0.0, 0.0)))
  lens.znear, lens.zfar = 0.01, 0.0
  k = (*(spec.distortion or ()), *([0.0] * 12))[:12]
  n = res[0] * res[1] * subdiv * subdiv
  out = {
    "direction": wp.zeros(n, dtype=wp.vec3, device="cpu"),
    "max_dist": wp.zeros(n, dtype=float, device="cpu"),
    "image": wp.zeros(n, dtype=wp.vec2, device="cpu"),
    "valid": wp.zeros(n, dtype=bool, device="cpu"),
  }
  wp.launch(
    _round_trip,
    dim=n,
    inputs=[
      wp.array([lens], dtype=Lens, device="cpu"),
      wp.array([k], dtype=lens_coefficients, device="cpu"),
      spec.fovy,
      wp.vec4(*(spec.intrinsics or (0.0, 0.0, 0.0, 0.0))),
      res[0],
      res[1],
      subdiv,
    ],
    outputs=list(out.values()),
    device="cpu",
  )
  h, w = res[1] * subdiv, res[0] * subdiv
  vectors = {name: out[name].numpy().reshape(h, w, -1) for name in ("direction", "image")}
  return {**vectors, **{name: out[name].numpy().reshape(h, w) for name in ("max_dist", "valid")}}


def _cell_centres(res: tuple[int, int], subdiv: int) -> np.ndarray:
  """(H * subdiv, W * subdiv, 2) image positions [pixels] of the sub-pixel cell centres."""
  x = (np.arange(res[0] * subdiv) + 0.5) / subdiv
  y = (np.arange(res[1] * subdiv) + 0.5) / subdiv
  return np.stack(np.meshgrid(x, y), axis=-1)


@pytest.mark.parametrize("fovy", [120.0, 200.0, 360.0])
def test_equidistant_fisheye_angle_is_radius_over_focal_length(fovy):
  """f-theta lens of `fovy` degrees across the image height: the ray's angle from the axis is
  r / f (f = (H / 2) / (fovy / 2) in pixels), its azimuth the pixel's; beyond 180 degrees from
  the axis (fovy 360: the corners) a pixel has no ray (max distance 0)."""
  res = RES
  spec = rw.CameraSpec(projection="fisheye", fovy=fovy)
  out = _cast(spec, res)
  f = (res[1] / 2) / np.radians(fovy / 2)
  centre = np.array(res, np.float64) / 2
  xy = _cell_centres(res, 1) - centre
  r = np.hypot(xy[..., 0], xy[..., 1])
  theta = r / f
  inside = theta <= np.pi - 1e-3
  d = out["direction"].astype(np.float64)
  measured = np.arccos(np.clip(-d[..., 2], -1, 1))
  np.testing.assert_allclose(measured[inside], theta[inside], atol=2e-5)
  az = np.arctan2(-d[..., 1], d[..., 0])  # image y down = camera -y
  on_axis = r > 0.5
  _angle_close(az[inside & on_axis], np.arctan2(xy[..., 1], xy[..., 0])[inside & on_axis], 2e-5)
  outside = theta > np.pi + 1e-3
  assert np.all(out["max_dist"][outside] == 0.0) and np.all(out["max_dist"][inside] > 1e6)
  assert (fovy == 360.0) == bool(outside.any())


@pytest.mark.parametrize(
  "spec",
  [
    rw.CameraSpec(projection="fisheye", fovy=200.0),
    rw.CameraSpec.from_opencv_fisheye(K_FISHEYE, RES, D_FISHEYE),
    rw.CameraSpec.from_opencv_fisheye(K_FISHEYE, RES, (-0.02, 0.003, 0.0, 0.0)),
    rw.CameraSpec.from_opencv(
      np.array([[30.0, 0, 23.5], [0, 28.0, 19.0], [0, 0, 1]]), RES, distortion=(-0.25, 0.08, 0.002, -0.003, 0.01)
    ),
    rw.CameraSpec.from_opencv(np.array([[30.0, 0, 23.5], [0, 28.0, 19.0], [0, 0, 1]]), RES, distortion=D_RATIONAL),
    rw.CameraSpec(fovy=70.0),
  ],
  ids=["f-theta-200", "kannala-brandt", "kannala-brandt-barrel", "opencv", "opencv-rational-prism", "pinhole"],
)
def test_projection_inverts_the_camera_rays(spec):
  """`camera.project` of a point on any sub-pixel ray lands on that cell's centre to 1e-4 px
  (fisheye: Newton inversion exact past 90 degrees; OpenCV: forward distortion of the iterated
  undistortion; 1e-3 px for the rational model, whose fixed-point iteration (OpenCV's) contracts
  more slowly at the image corners), and is valid wherever the ray is."""
  out = _cast(spec, RES, subdiv=3)
  ray = out["max_dist"] > 0.0
  assert ray.mean() > 0.9
  atol = 1e-3 if len(spec.distortion or ()) > 5 else 1e-4
  np.testing.assert_allclose(out["image"][ray], _cell_centres(RES, 3)[ray], atol=atol)
  assert np.all(out["valid"][ray])


def _opencv_project_points(xn: np.ndarray, yn: np.ndarray, d: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
  """OpenCV's `projectPoints` distortion of normalised coordinates (its documented formula, the
  12-coefficient model without the tilt)."""
  k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = d
  r2 = xn * xn + yn * yn
  radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
  xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn) + s1 * r2 + s2 * r2**2
  yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn + s3 * r2 + s4 * r2**2
  return xd, yd


def test_rational_thin_prism_rays_match_opencv():
  """`from_opencv` with OpenCV's 12 coefficients: the ray of pixel (u, v) (OpenCV centres at
  integers) projects through OpenCV's distortion formula back onto (u, v)."""
  k = np.array([[30.0, 0, 23.5], [0, 28.0, 19.0], [0, 0, 1]])
  out = _cast(rw.CameraSpec.from_opencv(k, RES, distortion=D_RATIONAL), RES)
  d = out["direction"].astype(np.float64)
  xd, yd = _opencv_project_points(d[..., 0] / -d[..., 2], -d[..., 1] / -d[..., 2], D_RATIONAL)
  u, v = np.meshgrid(np.arange(RES[0]), np.arange(RES[1]))
  np.testing.assert_allclose(xd * k[0, 0] + k[0, 2], u, atol=2e-3)
  np.testing.assert_allclose(yd * k[1, 1] + k[1, 2], v, atol=2e-3)


def test_kannala_brandt_rays_match_opencv():
  """`from_opencv_fisheye`: pixel (u, v) (OpenCV centres at integers) casts the ray at the angle
  OpenCV's model inverts to from theta_d = |((u - cx) / fx, (v - cy) / fy)|, with the pixel's
  azimuth; its corners see more than 90 degrees from the axis."""
  spec = rw.CameraSpec.from_opencv_fisheye(K_FISHEYE, RES, D_FISHEYE)
  out = _cast(spec, RES)
  theta, azimuth = fisheye_rays(K_FISHEYE, D_FISHEYE, RES)
  d = out["direction"].astype(np.float64)
  np.testing.assert_allclose(np.arccos(np.clip(-d[..., 2], -1, 1)), theta, atol=3e-5)
  az = np.arctan2(-d[..., 1], d[..., 0])
  off = np.hypot(*np.meshgrid(np.arange(RES[0]) - K_FISHEYE[0, 2], np.arange(RES[1]) - K_FISHEYE[1, 2])) > 0.5
  _angle_close(az[off], azimuth[off], 3e-5)
  assert theta.max() > np.pi / 2


def test_fisheye_model_fold_is_a_miss():
  """A Kannala-Brandt model that folds (theta_d stops increasing at 74 degrees for k1 = -0.2):
  the image points past its largest theta_d have no ray, and points past the fold project
  invalid."""
  k = (-0.2, 0.0, 0.0, 0.0)
  spec = rw.CameraSpec(projection="fisheye", fovy=180.0, distortion=k)
  out = _cast(spec, RES)
  fold = np.sqrt(1 / (3 * 0.2))
  limit = kb_theta_d(np.array(fold), k)
  centre = np.array(RES, np.float64) / 2
  xy = _cell_centres(RES, 1) - centre
  theta_d = np.hypot(xy[..., 0], xy[..., 1]) * np.radians(90.0) / (RES[1] / 2)
  assert np.all(out["max_dist"][theta_d > limit * 1.001] == 0.0)
  assert np.all(out["max_dist"][theta_d < limit * 0.95] > 0.0)
  ray = out["max_dist"] > 0.0
  measured = np.arccos(np.clip(-out["direction"][ray][..., 2].astype(np.float64), -1, 1))
  assert np.all(measured < fold + 1e-3)


# --- renders --------------------------------------------------------------------------------------


def _eye(s) -> tuple[np.ndarray, np.ndarray]:
  """(position, rotation) of the scene camera (float64)."""
  return s.mjd.cam_xpos[0].astype(np.float64), s.mjd.cam_xmat[0].reshape(3, 3).astype(np.float64)


def _ray_angles(position: np.ndarray, eye: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(theta from the optical axis, azimuth in image axes, distance) of hit points seen from the
  eye."""
  local = (position.astype(np.float64) - eye) @ rot  # camera frame: x right, y up, -z forward
  dist = np.linalg.norm(local, axis=-1)
  theta = np.arccos(np.clip(-local[..., 2] / np.maximum(dist, 1e-12), -1, 1))
  return theta, np.arctan2(-local[..., 1], local[..., 0]), dist


def _angle_close(a: np.ndarray, b: np.ndarray, atol: float) -> None:
  """Angles equal modulo 2 pi."""
  np.testing.assert_allclose((a - b + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=atol)


def test_fisheye_and_pinhole_cameras_mix(scene):
  """One Renderer with a pinhole, a 360-degree f-theta and an OpenCV fisheye camera: each fisheye
  pixel sees the room along its model's ray (read back from the hit position), its depth is the
  distance along the ray whatever the depth convention, the pixels outside the image circle are
  misses, and the pinhole camera's outputs are bit-identical to a Renderer without fisheye."""
  outputs = rw.Outputs.of("rgb", "seg", "position", depth=rw.settings.DepthOutput(convention="planar"))
  s = scene(ROOM, cam_res=RES, outputs=outputs, **{"gi.enabled": False})
  base = s.scene.cameras["cam"]
  ftheta = dataclasses.replace(base, projection="fisheye", fovy=360.0)
  kb = rw.CameraSpec.from_opencv_fisheye(K_FISHEYE, RES, D_FISHEYE, mount="cam", local=base.local)
  r = rw.Renderer(s.scene, nworld=1, cameras=[base, ftheta, kb], resolution=RES, settings=s.r.settings)
  r.render(s.d)
  assert r.context().camera_model == DISTORTION
  eye, rot = _eye(s)
  position, depth, seg = (required(a).numpy()[0] for a in (r.outputs.position, r.outputs.depth, r.outputs.seg))
  rgb = required(r.outputs.rgb).numpy()[0]
  # f-theta, 360 degrees across the height: theta = r / f, misses beyond 180 degrees.
  f = (RES[1] / 2) / np.pi
  xy = _cell_centres(RES, 1) - np.array(RES, np.float64) / 2
  theta = np.hypot(xy[..., 0], xy[..., 1]) / f
  inside, outside = theta < np.pi - 0.02, theta > np.pi + 0.02
  measured, azimuth, dist = _ray_angles(position[1], eye, rot)
  np.testing.assert_allclose(measured[inside], theta[inside], atol=2e-4)
  off_axis = inside & (theta > 0.05)
  _angle_close(azimuth[off_axis], np.arctan2(xy[..., 1], xy[..., 0])[off_axis], 2e-4)
  np.testing.assert_allclose(depth[1][inside], dist[inside], rtol=1e-5)
  assert np.all(depth[1][outside] == 0.0) and np.all(seg[1][outside] == -1) and np.all(position[1][outside] == 0.0)
  assert np.all(seg[1][inside] >= 0)
  # OpenCV fisheye: OpenCV's inverse model, beyond 90 degrees in the corners.
  theta, azimuth_cv = fisheye_rays(K_FISHEYE, D_FISHEYE, RES)
  measured, azimuth, dist = _ray_angles(position[2], eye, rot)
  np.testing.assert_allclose(measured, theta, atol=2e-4)
  _angle_close(azimuth[theta > 0.05], azimuth_cv[theta > 0.05], 2e-4)
  np.testing.assert_allclose(depth[2], dist, rtol=1e-5)
  assert theta.max() > np.pi / 2 and np.all(seg[2] >= 0)
  # The pinhole camera: what a Renderer without the fisheye cameras renders, bit for bit.
  alone = rw.Renderer(s.scene, nworld=1, cameras=[base], resolution=RES, settings=s.r.settings)
  alone.render(s.d)
  assert alone.context().camera_model == PINHOLE
  for a, b in (
    (r.outputs.rgb, alone.outputs.rgb),
    (r.outputs.depth, alone.outputs.depth),
    (r.outputs.position, alone.outputs.position),
  ):
    np.testing.assert_array_equal(required(a).numpy()[0, 0], required(b).numpy()[0, 0])
  assert rgb[1][inside].mean() > 0 and np.all(rgb[2][..., 3] > 0)


def test_per_world_distortion(scene):
  """Per-world "camera.distortion": each world's pinhole (OpenCV) and fisheye (Kannala-Brandt)
  camera casts the rays of its own coefficients, written in place without recompiling; the
  coefficients start as the CameraSpecs'."""
  k_cv = np.array([[30.0, 0.0, 23.5], [0.0, 28.0, 19.0], [0.0, 0.0, 1.0]])
  nworld = 3
  s = scene(ROOM, nworld=nworld, cam_res=RES, outputs=rw.Outputs.of("seg", "position"))
  base = s.scene.cameras["cam"]
  cameras = [
    rw.CameraSpec.from_opencv(k_cv, RES, distortion=(-0.1, 0.0, 0.0, 0.0), mount="cam", local=base.local),
    rw.CameraSpec.from_opencv_fisheye(K_FISHEYE, RES, D_FISHEYE, mount="cam", local=base.local),
  ]
  r = rw.Renderer(
    s.scene,
    nworld=nworld,
    cameras=cameras,
    resolution=RES,
    settings=s.r.settings,
    per_world=rw.PerWorld(camera=["distortion"]),
  )
  table = r.per_world["camera.distortion"]
  assert table.shape == (nworld, 2)
  np.testing.assert_allclose(table.numpy()[:, 0], [[-0.1, *([0.0] * 11)]] * nworld)
  np.testing.assert_allclose(table.numpy()[:, 1], [[*D_FISHEYE, *([0.0] * 8)]] * nworld, rtol=1e-6)
  r.render(s.d)
  features = r.features
  coefficients = np.zeros((nworld, 2, 12), np.float32)
  opencv = [(0.0,) * 5, (-0.3, 0.09, 0.001, -0.002, 0.0), (0.15, -0.05, 0.0, 0.0, 0.01)]
  fisheye = [(0.0,) * 4, D_FISHEYE, (-0.03, 0.004, 0.0, 0.0)]
  coefficients[:, 0, :5] = opencv
  coefficients[:, 1, :4] = fisheye
  with module_loads() as loads:
    table.assign(coefficients)
    r.render(s.d)
  assert r.features is features and not loads, loads
  eye, rot = _eye(s)
  position = required(r.outputs.position).numpy()
  u, v = np.meshgrid(np.arange(RES[0]).astype(np.float64), np.arange(RES[1]).astype(np.float64))
  for w in range(nworld):
    # OpenCV: the hit's normalised coordinates are the undistorted pixel's.
    local = (position[w, 0].astype(np.float64) - eye) @ rot
    xn, yn = local[..., 0] / -local[..., 2], -local[..., 1] / -local[..., 2]
    k1, k2, p1, p2, k3 = opencv[w]
    r2 = xn * xn + yn * yn
    radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    np.testing.assert_allclose(xd * k_cv[0, 0] + k_cv[0, 2], u, atol=2e-3)
    np.testing.assert_allclose(yd * k_cv[1, 1] + k_cv[1, 2], v, atol=2e-3)
    theta, _ = fisheye_rays(K_FISHEYE, fisheye[w], RES)
    measured, _, _ = _ray_angles(position[w, 1], eye, rot)
    np.testing.assert_allclose(measured, theta, atol=2e-4)


def test_per_camera_outputs(scene):
  """A depth-only wrist camera next to an all-output head camera at the same pose: the same depth
  bit for bit, no RGB / segmentation written (the buffers' zeros); camera outputs are checked
  against the Renderer's, also when the Renderer's outputs change."""
  outputs = rw.Outputs.of("rgb", "depth", "seg")
  s = scene(ROOM, nworld=2, cam_res=RES, outputs=outputs, **{"gi.enabled": False})
  base = s.scene.cameras["cam"]
  wrist = dataclasses.replace(base, outputs=("depth",))
  r = rw.Renderer(s.scene, nworld=2, cameras=[base, wrist], resolution=RES, settings=s.r.settings)
  r.render(s.d)
  rgb, depth, seg = (required(a).numpy() for a in (r.outputs.rgb, r.outputs.depth, r.outputs.seg))
  np.testing.assert_array_equal(depth[:, 1], depth[:, 0])
  assert np.all(depth[:, 0] > 0.0)
  assert np.all(rgb[:, 1] == 0) and np.any(rgb[:, 0] != 0)
  assert np.all(seg[:, 1] == 0) and np.any(seg[:, 0] != 0)
  with pytest.raises(ValueError, match="not among the Renderer's"):
    r.update_settings(outputs=rw.Outputs.of("rgb", "seg"))
  with pytest.raises(ValueError, match="whole groups"):
    rw.Renderer(
      s.scene,
      nworld=1,
      cameras=[base, dataclasses.replace(base, outputs=("rgb",))],
      resolution=RES,
      settings=s.r.settings.replace(outputs=rw.Outputs.of("rgb", "hdr")),
    )
  with pytest.raises(ValueError, match="written for every camera"):
    rw.Renderer(
      s.scene,
      nworld=1,
      cameras=[wrist],
      resolution=RES,
      settings=s.r.settings.replace(outputs=rw.Outputs.of("depth", "position")),
    )
  # A geometry-only Renderer: the wrist's seg left out.
  r = rw.Renderer(
    s.scene,
    nworld=2,
    cameras=[base, wrist],
    resolution=RES,
    settings=s.r.settings.replace(outputs=rw.Outputs.of("depth", "seg")),
  )
  r.render(s.d)
  seg = required(r.outputs.seg).numpy()[0]
  assert np.all(seg[1] == 0) and np.any(seg[0] != 0)
  np.testing.assert_array_equal(required(r.outputs.depth).numpy()[0, 1], depth[0, 0])


def _project_fisheye(
  points: np.ndarray, eye: np.ndarray, rot: np.ndarray, k: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
  """(image positions [pixels, centres at i + 0.5], angle past the model's fold [rad]) of world
  points through the `K_FISHEYE` lens with coefficients `k` at pose (eye, rot) (OpenCV's
  fisheye::projectPoints, float64)."""
  local = (points - eye) @ rot
  x, y_down = local[..., 0], -local[..., 1]
  r = np.hypot(x, y_down)
  theta = np.arctan2(r, -local[..., 2])
  scale = kb_theta_d(theta, k) / np.maximum(r, 1e-12)
  u = K_FISHEYE[0, 0] * x * scale + K_FISHEYE[0, 2] + 0.5
  v = K_FISHEYE[1, 1] * y_down * scale + K_FISHEYE[1, 2] + 0.5
  grid = np.linspace(0.0, np.pi, 2000)
  t2 = grid * grid
  slope = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])))
  fold = grid[np.argmax(slope <= 0)] if np.any(slope <= 0) else np.pi
  return np.stack([u, v], axis=-1), theta - fold


def test_fisheye_motion_vectors_reproject_through_the_previous_lens(scene):
  """Motion of a free fisheye camera: after a 5-degree yaw (world 0) or a change of its per-world
  lens coefficients alone (world 1), each pixel's motion is the hit point's position through the
  previous frame's pose and coefficients (OpenCV's model) minus the pixel centre."""
  nworld = 2
  s = scene(ROOM, nworld=nworld, cam_res=RES, outputs=rw.Outputs.of("position", "motion"))
  spec = rw.CameraSpec.from_opencv_fisheye(K_FISHEYE, RES, D_FISHEYE)
  r = rw.Renderer(
    s.scene, nworld=nworld, cameras=[spec], resolution=RES, settings=s.r.settings, per_world=rw.PerWorld(camera=["distortion"])
  )
  # Looking along +y from (0.1, -0.2, 0.05): camera x = world x, camera y = world z.
  rot0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
  yaw = np.radians(5.0)
  rot1 = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]) @ rot0
  eye = np.array([0.1, -0.2, 0.05])

  def pose(rots: list[np.ndarray]) -> None:
    quats = [wp.quat_from_matrix(wp.mat33(*m.astype(np.float32).ravel())) for m in rots]
    r.camera_pose.assign(np.array([[[*eye, *q]] for q in quats], np.float32))

  k_new = (0.08, -0.02, 0.0, 0.0)
  pose([rot0, rot0])
  r.render(s.d)
  pose([rot1, rot0])
  table = r.per_world["camera.distortion"].numpy()
  table[1, 0, :4] = k_new
  r.per_world["camera.distortion"].assign(table)
  r.render(s.d)
  position = required(r.outputs.position).numpy()[:, 0].astype(np.float64)
  motion = required(r.outputs.motion).numpy()[:, 0].astype(np.float64)
  centres = _cell_centres(RES, 1)
  for w in range(nworld):
    # Both worlds' previous frame: pose rot0, the spec's coefficients.
    previous, past_fold = _project_fisheye(position[w], eye, rot0, D_FISHEYE)
    # Past the previous lens's fold (the corners beyond ~137 degrees): no reprojection. World 1's
    # new lens folds at 123 degrees: its corners are misses (no hit point).
    hit = np.any(position[w] != 0.0, axis=-1)
    seen = (past_fold < -0.01) & hit
    np.testing.assert_allclose(motion[w][seen], (previous - centres)[seen], atol=2e-3)
    assert np.all(motion[w][(past_fold > 0.01) & hit] == 0.0) and seen.mean() > 0.85
  assert np.abs(motion[0]).max() > 1.0 and np.abs(motion[1]).max() > 0.5


@pytest.mark.parametrize(
  "post",
  [{}, {"grading.enabled": True, "grading.lift": (0.2, 0.2, 0.2)}],
  ids=["deferred", "deferred-grading"],
)
def test_masked_camera_outputs_survive_the_post_passes(scene, post):
  """The display passes (the render's, the deferred display of automatic exposure; also with a lift
  that maps black to grey) leave a depth-only camera's display untouched (zeros) and its depth
  exact, while the RGB camera is displayed."""
  outputs = rw.Outputs.of("rgb", "depth")
  s = scene(ROOM, cam_res=RES, outputs=outputs, **{"gi.enabled": False, **post})
  base = s.scene.cameras["cam"]
  r = rw.Renderer(
    s.scene, nworld=1, cameras=[base, dataclasses.replace(base, outputs=("depth",))], resolution=RES, settings=s.r.settings
  )
  for _ in range(3):
    r.render(s.d)
  rgb, depth = required(r.outputs.rgb).numpy()[0], required(r.outputs.depth).numpy()[0]
  assert np.all(rgb[1] == 0) and np.all(rgb[0][..., :3].mean(axis=-1) > 0)
  # The shaded and the geometry-only branch compute the depth in two code paths: CUDA contracts
  # them into FMAs differently (L40S: <= 4.8e-7 relative); bit-exact on the CPU.
  np.testing.assert_allclose(depth[1], depth[0], rtol=1e-6)


def test_perspective_camera_without_near_plane(scene):
  """znear 0 (valid: no near clip) casts the same rays as a near plane: the image window is on a
  unit plane instead of a zero-size one (which cast NaN rays: every pixel a miss)."""
  outputs = rw.Outputs.of("seg", depth=rw.settings.DepthOutput(convention="radial"))
  s = scene(ROOM, cam_res=RES, outputs=outputs)
  base = s.scene.cameras["cam"]
  r = rw.Renderer(
    s.scene, nworld=1, cameras=[dataclasses.replace(base, znear=0.0), base], resolution=RES, settings=s.r.settings
  )
  r.render(s.d)
  depth = required(r.outputs.depth).numpy()[0]
  assert np.all(depth[0] > 0.0)
  np.testing.assert_allclose(depth[0], depth[1], rtol=1e-5)


def _pose_scene() -> rw.GpuScene:
  """A floor, a box at the origin and a sphere off-centre under a sun (any recipe lights it)."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), specular_roughness=0.8))
  red = b.add_material(rw.Material(base_color=(0.8, 0.1, 0.1), specular_roughness=0.5))
  b.add_instance("plane", material=grey, size=(4.0, 4.0, 0.1))
  b.add_instance("box", material=red, size=(0.2, 0.2, 0.2), xform=np.c_[np.eye(3), [0.0, 0.0, 0.2]])
  b.add_instance("sphere", material=grey, size=(0.15, 0.0, 0.0), xform=np.c_[np.eye(3), [0.6, 0.3, 0.15]])
  return rw.upload(b.build().with_lighting(rw.Lighting([rw.Light.sun((-0.3, 0.4, -1.0))])))


def test_a_free_pose_written_before_the_first_render_frames_every_recipe_alike():
  """The first renderer of a process (FAST, the warp-optix comparison's order), then a default and
  a FAST one: a free camera pose assigned before each first render frames them alike (the same
  depth and segmentation everywhere), the look-at target projects to the image centre, and the
  colour image agrees with them (the red box where the segmentation has it: one camera for every
  output)."""
  res = 32
  eye, target = np.array([1.2, -1.6, 1.0]), np.array([0.0, 0.0, 0.4])
  f = (target - eye) / np.sqrt(np.sum((target - eye) ** 2))
  x = np.cross(f, [0.0, 0.0, 1.0])
  x = x / np.sqrt(np.sum(x**2))
  m = np.stack([x, np.cross(x, f), -f], axis=1)  # camera x, y, z axes (looking down -z)
  w = np.sqrt(1.0 + np.trace(m)) / 2.0
  q = [(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w]
  scene = _pose_scene()
  depths, rgbs, segs = [], [], []
  for recipe_fields in (rw.FAST, {}, rw.FAST):
    s = rw.RenderSettings().replace(recipe_fields).replace(outputs=rw.Outputs.of("rgb", "depth", "hdr", "seg"))
    with rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=50.0)], resolution=(res, res), settings=s) as r:
      r.camera_pose.assign(np.array([[[*eye, *q]]], np.float32))
      r.render()
      out = r.render()
      depths.append(np.asarray(required(out.depth).numpy()).reshape(res, res).copy())
      rgbs.append(np.asarray(required(out.rgb).numpy()).view(np.uint8).reshape(res, res, -1)[..., :3].astype(float))
      segs.append(np.asarray(required(out.seg).numpy()).reshape(res, res, -1)[..., 0].copy())
  for d, seg in zip(depths[1:], segs[1:], strict=True):
    np.testing.assert_allclose(d, depths[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(seg, segs[0])
  # The target (the centre of the box's top face) at the image centre: its depth along the view axis
  # (the mean of the four centre pixels: depth is linear across a plane in the image).
  centre = depths[0][res // 2 - 1 : res // 2 + 1, res // 2 - 1 : res // 2 + 1].mean()
  np.testing.assert_allclose(centre, np.dot(target - eye, f), rtol=0.01)
  # Every recipe's colour shows the red box exactly where the segmentation has it.
  box = segs[0] == segs[0][res // 2, res // 2]
  for rgb in rgbs:
    red = (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 0] > 20.0)
    assert (red & box).sum() >= 0.8 * box.sum() and (red & ~box).sum() <= 0.05 * box.sum(), (red.sum(), box.sum())
