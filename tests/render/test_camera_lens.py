"""Per-camera lens constants (W16, B33): resolution, clip planes, exposure, OpenCV K +
distortion."""

import dataclasses

import numpy as np
import pytest
import warp as wp
from conftest import required
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp._base.types import CameraSlot
from raywarp.core.kernels.abi.structs import Lens
from raywarp.core.trace.rays.projection import camera_window

# A wall 3 m in front of the camera (fronto-parallel, larger than the view) with a sphere in front.
XML = """
<mujoco>
  <visual><map znear="0.01" zfar="50"/></visual>
  <statistic extent="1"/>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.3 -1"/>
    <geom name="wall" type="box" pos="0 3 0" size="20 0.01 20"/>
    <geom name="ball" type="sphere" pos="0.3 1.5 0.2" size="0.3"/>
    <camera name="cam" pos="0 0 0" xyaxes="1 0 0 0 0 1" fovy="60"/>
  </worldbody>
</mujoco>
"""
RES = (32, 24)


def _spec(s, **fields):
  """The scene camera's spec with `fields` changed."""
  return dataclasses.replace(s.scene.cameras["cam"], **fields)


def _two(scene, a: dict, b: dict, **settings):
  """A Renderer of the scene camera twice, with spec changes `a` and `b`, rendered once."""
  s = scene(XML, cam_res=RES, **settings)
  r = rw.Renderer(s.scene, nworld=1, cameras=[_spec(s, **a), _spec(s, **b)], resolution=RES, settings=s.r.settings)
  r.render(s.d)
  return r


def test_per_camera_clip_planes(scene):
  """Each camera clips at its own zfar (B33: camera 0's planes were used for every camera)."""
  r = _two(scene, {"zfar": 2.0}, {}, outputs=rw.Outputs.of("depth"))
  depth = required(r.outputs.depth).numpy()[0]
  assert np.all(depth[0] < 2.0) and np.any(depth[0] == 0.0)  # the wall at 3 m is clipped
  assert np.all(depth[1] > 0.0)  # every pixel sees the wall or the ball
  features = r.features
  d = scene(XML, cam_res=RES).render().d  # rendered: its renderer's warm-up loads end here
  with module_loads() as loads:
    r.set_clip(zfar=0.0, camera=CameraSlot(0))
    r.render(d)
  np.testing.assert_array_equal(required(r.outputs.depth).numpy()[0, 0], required(r.outputs.depth).numpy()[0, 1])
  assert r.features is features and not loads, loads
  # A camera index out of range raises (-1 silently changed the last camera, 2 an IndexError).
  for camera in (-1, 2):
    with pytest.raises(ValueError, match=r"camera must be in \[0, 1\]"):
      r.set_clip(zfar=1.0, camera=CameraSlot(camera))


@pytest.mark.parametrize("depth", ["planar", "radial"])
def test_near_clip_hides_what_is_nearer_than_znear(scene, depth):
  """S2: znear is a real near clip. Camera 0's near plane (2 m) lies behind the ball (1.2-1.8 m):
  it sees only the wall, at the depth (from the eye) camera 1 (znear 0.01) gives the wall, with
  the wall's segment id and color everywhere. A plane 1.5 m away cuts the ball (the camera is
  through it: nothing nearer than 1.5 m is seen; the camera is inside the ball, whose back faces
  are culled)."""
  outputs = rw.Outputs.of("rgb", "seg", depth=rw.settings.DepthOutput(convention=depth))
  r = _two(scene, {"znear": 2.0}, {}, outputs=outputs, **{"gi.enabled": False})
  d, seg, rgb = (required(a).numpy()[0] for a in (r.outputs.depth, r.outputs.seg, r.outputs.rgb))
  ball = d[1] < 2.5
  assert ball.sum() > 20 and np.all(d[0] > 2.9)
  np.testing.assert_allclose(d[0][~ball], d[1][~ball], rtol=1e-5)
  wall_id = seg[1][~ball][0]
  assert np.all(seg[0] == wall_id) and not np.any(seg[1][ball] == wall_id)
  # The wall's color where camera 1 saw the ball: as bright as the wall around it (the ball's
  # shadow aside: the light is behind the camera, the wall pixels next to the ball are lit too).
  assert np.abs(rgb[0][ball].astype(float).mean() - rgb[1][~ball].astype(float).mean()) < 40
  r.set_clip(znear=1.5, camera=CameraSlot(0))
  r.render(scene(XML, cam_res=RES).d)
  d = required(r.outputs.depth).numpy()[0, 0]
  assert np.all(d >= 1.5 - 1e-4)


def test_per_camera_exposure(scene):
  """CameraSpec.exposure (stops) brightens the display image; the HDR output is unchanged."""
  r = _two(scene, {}, {"exposure": 1.0}, outputs=rw.Outputs.of("rgb", "hdr"))
  hdr = required(r.outputs.hdr).numpy()[0]
  np.testing.assert_array_equal(hdr[0], hdr[1])
  rgb = required(r.outputs.rgb).numpy()[0, ..., :3].astype(np.float32)
  assert rgb[1].mean() > rgb[0].mean() * 1.2


def test_per_camera_resolution(scene):
  """A camera with its own resolution renders what a Renderer at that resolution renders;
  outputs are per camera (`RenderOutputs.cameras`), the dense views None."""
  kinds = rw.Outputs.of("rgb", "depth", "seg")
  r = _two(scene, {}, {"resolution": (16, 12)}, outputs=kinds)
  assert r.outputs.rgb is None and len(r.outputs.cameras) == 2
  small = scene(XML, cam_res=(16, 12), outputs=kinds).render()
  cam1 = r.outputs.cameras[1]
  assert required(cam1.rgb).shape == (1, 1, 12, 16)
  np.testing.assert_array_equal(required(cam1.rgb).numpy()[:, 0], small.out.rgb.numpy()[:, 0])
  np.testing.assert_array_equal(required(cam1.depth).numpy()[:, 0], small.out.depth.numpy()[:, 0])
  big = scene(XML, cam_res=RES, outputs=kinds).render()
  np.testing.assert_array_equal(required(r.outputs.cameras[0].seg).numpy()[:, 0], big.out.seg.numpy()[:, 0])
  with pytest.raises(ValueError, match="same resolution"):
    r.update_settings(outputs=rw.Outputs.of("rgb", "motion"))


def _undistort(xd: np.ndarray, yd: np.ndarray, k: tuple) -> tuple[np.ndarray, np.ndarray]:
  """OpenCV's undistortPoints iteration in float64 (k1, k2, p1, p2, k3)."""
  k1, k2, p1, p2, k3 = k
  x, y = xd.copy(), yd.copy()
  for _ in range(50):
    r2 = x * x + y * y
    radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
    dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    x, y = (xd - dx) / radial, (yd - dy) / radial
  return x, y


@pytest.mark.parametrize("distortion", [None, (-0.25, 0.08, 0.002, -0.003, 0.0)])
def test_opencv_camera_matrix_and_distortion(scene, distortion):
  """`CameraSpec.from_opencv`: the ray through OpenCV pixel (u, v) has normalised coordinates
  undistort(((u - cx) / fx, (v - cy) / fy)), read back from the wall's planar / radial depth."""
  w, h = RES
  k = np.array([[30.0, 0.0, 17.5], [0.0, 28.0, 10.0], [0.0, 0.0, 1.0]])
  s = scene(XML, cam_res=RES)
  base = s.scene.cameras["cam"]
  depths = []
  for kind in ("planar", "radial"):
    spec = rw.CameraSpec.from_opencv(k, (w, h), distortion=distortion, mount="cam", local=base.local)
    r = rw.Renderer(
      s.scene,
      nworld=1,
      cameras=[spec],
      resolution=(w, h),
      settings=rw.RenderSettings(outputs=rw.Outputs.of("seg", depth=rw.settings.DepthOutput(convention=kind))),
    )
    r.render(s.d)
    depths.append(required(r.outputs.depth).numpy()[0, 0].astype(np.float64))
    seg = required(r.outputs.seg).numpy()[0, 0]
  u, v = np.meshgrid(np.arange(w).astype(np.float64), np.arange(h).astype(np.float64))
  xd, yd = (u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1]
  x, y = _undistort(xd, yd, distortion or (0.0,) * 5)
  wall = seg == 0
  assert wall.sum() > 0.5 * w * h
  measured = (depths[1] / depths[0]) ** 2 - 1.0  # x^2 + y^2 on the wall's plane
  np.testing.assert_allclose(measured[wall], (x * x + y * y)[wall], atol=2e-4, rtol=2e-3)
  np.testing.assert_allclose(depths[0][wall], 3.0 - 0.01, rtol=1e-5)


def test_host_scene_cameras_keep_their_clip_and_aperture_offset():
  """B33: a host (USD) scene's cameras each keep their clipping range, and a perspective aperture
  offset shifts the image window (objects move the other way)."""
  b = rw.SceneBuilder()
  b.add_instance("box", size=(20.0, 20.0, 0.01), xform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -3.0]]))
  b.add_instance("sphere", size=(0.3, 0.3, 0.3), xform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.5]]))
  pose = np.eye(4)[:3]
  lens = {"focal_length": 20.0, "vertical_aperture": 15.0, "horizontal_aperture": 20.0}
  b.add_camera("near", rw.scene.Camera(xform=pose, near=0.05, far=2.0, **lens))
  b.add_camera("plain", rw.scene.Camera(xform=pose, near=0.01, far=0.0, **lens))
  b.add_camera("shifted", rw.scene.Camera(xform=pose, horizontal_offset=4.0, **lens))
  gpu = rw.upload(b.build())
  specs = gpu.cameras
  assert (specs["near"].znear, specs["near"].zfar) == (pytest.approx(0.05), 2.0) and specs["plain"].zfar == 0.0
  assert specs["shifted"].intrinsics == (20.0, 20.0, 4.0, -0.0) and specs["plain"].intrinsics is None
  kinds = rw.Outputs.of("depth", "seg")
  r = rw.Renderer(gpu, nworld=1, resolution=(40, 30), settings=rw.RenderSettings(outputs=kinds))
  r.render()
  depth, seg = required(r.outputs.depth).numpy()[0], required(r.outputs.seg).numpy()[0]
  assert np.all(depth[0] < 2.0) and np.all(depth[1] > 0.0)  # "near" clips the wall at 3 m
  sphere_x = [np.argwhere(seg[c] == 1)[:, 1].mean() for c in (1, 2)]
  assert sphere_x[1] < sphere_x[0] - 5  # the window moved right: the sphere moved left


def test_host_cameras_with_one_name_keep_their_own_pose():
  """Two host cameras named alike are two scene cameras, each mounted on its own pose row (one
  dict entry used to shift every later camera onto the wrong row)."""
  b = rw.SceneBuilder()
  b.add_instance("sphere", size=(0.3, 0.3, 0.3), xform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.5]]))
  lens = {"focal_length": 20.0, "vertical_aperture": 15.0, "horizontal_aperture": 20.0}
  away = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])  # looks along +z, away from it
  b.add_camera("cam", rw.scene.Camera(xform=away, **lens))
  b.add_camera("cam", rw.scene.Camera(xform=np.eye(4)[:3], **lens))
  b.add_camera("last", rw.scene.Camera(xform=np.eye(4)[:3], **lens))
  gpu = rw.upload(b.build())
  assert gpu.camera_names == ["cam", "cam#1", "last"]
  r = rw.Renderer(gpu, nworld=1, resolution=(16, 12), settings=rw.RenderSettings(outputs=rw.Outputs.of("seg")))
  r.render()
  seg = required(r.outputs.seg).numpy()[0]
  # The sphere (instance 0) is behind the first camera, in front of the other two.
  assert not np.any(seg[0] == 0) and np.any(seg[1] == 0) and np.any(seg[2] == 0)


@pytest.mark.parametrize(
  ("fields", "match"),
  [
    ({"fovy": float("nan")}, "fovy"),
    ({"fovy": 0.0}, "fovy"),
    ({"fovy": 180.0}, "fovy"),
    ({"intrinsics": (10.0, 10.0, 0.0, 0.0)}, "sensor_size"),  # was silently ignored (fovy used)
    ({"sensor_size": (4.0, 3.0)}, "conform='fit'"),  # a sensor alone is a fit window only
    ({"sensor_size": (4.0, 0.0), "conform": "fit"}, "conform='fit'"),
    ({"conform": "fill"}, "conform"),
    ({"intrinsics": (-1.0, 10.0, 0.0, 0.0), "sensor_size": (4.0, 3.0)}, "fx, fy > 0"),
    ({"znear": -0.1}, "znear"),
    ({"znear": 2.0, "zfar": 1.0}, "must be below zfar"),
    ({"zfar": float("inf")}, "zfar"),
    ({"resolution": (32.5, 24)}, "resolution"),  # was accepted, then failed in the buffers
    ({"resolution": (True, 24)}, "resolution"),
    ({"resolution": (0, 24)}, "resolution"),
    ({"exposure": float("nan")}, "exposure"),
    ({"exposure": 200.0}, r"\[-126, 126\] stops"),  # 2^200 was inf in float32: NaN images
    ({"exposure": -200.0}, r"\[-126, 126\] stops"),  # 2^-200 was 0
    ({"projection": "spherical"}, "projection"),
    ({"projection": "fisheye", "fovy": 361.0}, "fovy"),
    ({"distortion": (0.1, 0.0, 0.0)}, "distortion"),
    ({"distortion": (0.1, 0.0, 0.0, 0.0), "projection": "orthographic"}, "distortion"),
    ({"distortion": (0.1, 0.0, 0.0, 0.0, 0.0), "projection": "fisheye"}, "distortion"),
    ({"projection": "fisheye", "rays": np.zeros((2, 2, 2, 3))}, "rays"),
    ({"outputs": ()}, "outputs"),
    ({"outputs": ("rgb", "rgb")}, "outputs"),
    ({"outputs": ("colour",)}, "outputs"),
    ({"outputs": ["depth"]}, "outputs"),
    ({"mount": ""}, "mount"),
    ({"local": (np.zeros(2), np.eye(3))}, "local"),
    ({"exposure": "0"}, "exposure"),  # a string was split into its characters
    ({"fovy": "12"}, "fovy"),
    ({"fovy": True}, "fovy"),  # was fovy 1.0
    ({"fovy": None}, "fovy"),
    ({"fovy": [60.0]}, "fovy"),  # one-element sequences passed and failed later, far from the field
    ({"exposure": [1.0]}, "exposure"),
    ({"znear": [0.1]}, "znear"),
  ],
)
def test_camera_spec_rejects_invalid_fields(fields, match):
  """A CameraSpec validates its own fields at construction (non-finite numbers, ranges, a
  resolution of two integers, intrinsics only with a sensor), naming the field."""
  with pytest.raises(ValueError, match=match):
    rw.CameraSpec(**fields)


def test_camera_spec_accepts_the_adapters_conventions():
  """MuJoCo's "no intrinsics" (zeros on a (0, 0) sensor), an orthographic view height, an
  unbounded zfar of 0 and NumPy integer resolutions are valid."""
  rw.CameraSpec(fovy=45.0, intrinsics=(0.0, 0.0, 0.0, 0.0), sensor_size=(0.0, 0.0))
  rw.CameraSpec(fovy=250.0, projection="orthographic", znear=0.0, zfar=0.0)
  rw.CameraSpec(resolution=(np.int64(32), np.int32(24)))  # pyrefly: ignore[bad-argument-type]  # NumPy scalars on purpose
  rw.CameraSpec(fovy=np.float32(45.0), exposure=np.float32(0.5), znear=np.float32(0.1), zfar=np.float64(9.0))  # pyrefly: ignore[bad-argument-count, bad-argument-type]  # NumPy scalars on purpose (np.float32(x): untyped)


def test_set_clip_validates_the_distances(scene):
  """`set_clip` rejects negative / non-finite distances, a zfar not beyond the camera's znear and
  a camera index that is not an integer (True was camera 1; 1.5 failed deep in the loop)."""
  s = scene(XML, cam_res=RES)
  for kwargs, match in (
    ({"znear": -1.0}, "znear must be a finite distance"),
    ({"zfar": float("nan")}, "zfar must be a finite distance"),
    ({"zfar": 0.001}, "must be below zfar"),
    ({"znear": 0.5, "camera": True}, "camera must be in"),
    ({"znear": 0.5, "camera": 1.5}, "camera must be in"),
  ):
    with pytest.raises(ValueError, match=match):
      s.r.set_clip(**kwargs)
  s.r.set_clip(znear=0.5)
  assert s.r._rig.clips[0][0] == 0.5
  s.r.set_clip(zfar=np.float32(50.0), camera=np.int64(0))  # pyrefly: ignore[bad-argument-count]  # np.float32(x): untyped (NumPy scalars on purpose)
  assert s.r._rig.clips[0][1] == 50.0


def test_a_spec_clip_plane_beyond_the_scene_other_plane_raises(scene):
  """A camera's own znear at or beyond the scene's zfar (or its own zfar before the scene's znear)
  left znear >= zfar: every ray missed, silently. The Renderer rejects it, as `set_clip` does."""
  with pytest.raises(ValueError, match="must be below zfar"):
    scene(XML, cam_res=RES, cameras=[rw.CameraSpec(znear=60.0)])
  with pytest.raises(ValueError, match="must be below zfar"):
    scene(XML, cam_res=RES, cameras=[rw.CameraSpec(zfar=0.005)])
  scene(XML, cam_res=RES, cameras=[rw.CameraSpec(znear=60.0, zfar=0.0)])  # unbounded: valid


@pytest.mark.parametrize(("scale", "near"), [(1.0e-3, 1.0e-3 * 0.01 * np.sqrt(3.0) * 2.0), (1.0, 0.01), (100.0, 0.01)])
def test_default_near_plane_follows_a_small_scene(scale: float, near: float) -> None:
  """A scene without cameras had a fixed 1 cm near plane: a millimetre-scale scene rendered
  black. Its default is 0.01 of the scene's extent, at most 1 cm."""
  b = rw.SceneBuilder()
  b.add_instance("box", size=(scale, scale, scale))
  scene = rw.upload(b.build(), device="cpu")
  assert scene.clip == pytest.approx((min(near, 0.01), 0.0), rel=1e-6)


def _hydra_window(aperture, offset, focal, aspect):
  """(left, right, top, bottom) on the plane at unit distance of a USD camera's window conformed to
  an image of `aspect` by Hydra's default policy (OpenUSD CameraUtilConformedWindow with
  CameraUtilFit: a window wider than the image matches horizontally, else vertically; the centre
  stays)."""
  (w, h), (ox, oy) = aperture, offset
  if w / h > aspect:  # _ResolveConformWindowPolicy: Fit and aspect > target -> MatchHorizontally
    h = w / aspect
  else:
    w = h * aspect
  return np.array([ox - 0.5 * w, ox + 0.5 * w, oy + 0.5 * h, oy - 0.5 * h]) / focal


@wp.kernel
def _window(lens: wp.array[Lens], fovy: wp.array[float], intrinsics: wp.array[wp.vec4], size: wp.vec2i, out: wp.array[wp.vec4]):
  """The camera window (`projection.camera_window`) of lens 0 at unit distance."""
  out[0] = camera_window(lens[0].projection, fovy[0], lens[0].sensorsize, intrinsics[0], size[0], size[1], 1.0)


@pytest.mark.parametrize("offset", [(0.0, 0.0), (2.0, -1.5)])
@pytest.mark.parametrize("size", [(512, 512), (1024, 512), (512, 1024), (640, 466)])
@pytest.mark.parametrize("ortho", [False, True])
def test_usd_cameras_conform_like_hydra(size, offset, ortho):
  """A host (USD) camera's window on an image of another aspect is Hydra's (CameraUtilFit): the
  default 20.955 x 15.2908 aperture at 512^2 shows its whole width (the vertical aperture was
  kept: 27 % narrower than Hydra / RTX), an aperture offset keeps the same framing (it switched
  to cropping the sensor: a 31 % jump at 2:1), orthographic windows alike (no offsets there)."""
  from raywarp.core.pipeline.cameras import CameraRig, conformed

  if ortho and offset != (0.0, 0.0):
    pytest.skip("orthographic aperture offsets are dropped (reported)")
  aperture, focal = (20.955, 15.2908), 50.0
  b = rw.SceneBuilder()
  b.add_instance("sphere", size=(0.3, 0.3, 0.3), xform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.5]]))
  projection = rw.scene.Projection.ORTHOGRAPHIC if ortho else rw.scene.Projection.PERSPECTIVE
  cam = rw.scene.Camera(
    xform=np.eye(4)[:3], projection=projection, focal_length=focal, horizontal_aperture=aperture[0],
    vertical_aperture=aperture[1], horizontal_offset=offset[0], vertical_offset=offset[1],
  )  # fmt: skip
  b.add_camera("cam", cam)
  gpu = rw.upload(b.build(), "cpu")
  rig = CameraRig(gpu, [gpu.cameras["cam"]], nworld=1, width=size[0], height=size[1])
  out = wp.zeros(1, dtype=wp.vec4, device="cpu")
  inputs = [rig.lens, rig.fovy.flatten(), rig.intrinsics.flatten(), wp.vec2i(*size), out]
  wp.launch(_window, dim=1, inputs=inputs, device="cpu")
  got = out.numpy()[0]
  expected = _hydra_window(aperture, offset, focal, size[0] / size[1])
  if ortho:  # the view window itself [m]: apertures are tenths of a unit, no focal length
    expected = expected * focal
  np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-7)
  assert conformed(gpu.cameras["cam"], size)[0] == pytest.approx(float(rig.fovy.numpy()[0, 0]), rel=1e-6)


def test_an_authored_camera_range_does_not_clip_the_free_cameras() -> None:
  """A scene camera's clipping range is its own: free cameras keep the scene default (near from
  the extent, unbounded far). The first camera's range was every free camera's: a glTF / USD
  camera with a 3 m far plane hid a whole asset from a free camera 4 m away."""
  b = rw.SceneBuilder()
  b.add_instance("box", size=(1.0, 1.0, 1.0))
  b.add_camera("authored", rw.scene.Camera(xform=np.eye(4)[:3], near=0.001, far=3.0))
  scene = rw.upload(b.build(), device="cpu")
  assert scene.clip == pytest.approx((0.01, 0.0))
  assert (scene.cameras["authored"].znear, scene.cameras["authored"].zfar) == pytest.approx((0.001, 3.0))
