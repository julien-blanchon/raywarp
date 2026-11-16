"""Ground truth computed from a render's outputs (`core/outputs/ground_truth.py`) and the camera
export (`core/outputs/camera_export.py`): checked against the renderer's own world-space outputs
(position, normal) through the exported intrinsics and poses, on a small host scene (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp
from conftest import NO_GI

import raywarp as rw

NWORLD = 2
SIZE = (24, 16)
# The box: centre, half extents; world 1 turns it about z and moves it.
BOX = ((0.1, 0.2, 0.3), (0.3, 0.2, 0.3))
_TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)


def _scene() -> rw.Scene:
  """A floor, a box and a mesh tetrahedron off the origin of its own frame, a light."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.6, 0.6, 0.6)))
  b.add_instance("box", material=grey, size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], path="floor")
  b.add_instance("box", material=grey, size=BOX[1], xform=np.c_[np.eye(3), BOX[0]], path="box")
  tet = np.array([[1.0, 1.0, 1.3], [1.2, 1.0, 1.0], [0.9, 1.17, 1.0], [0.9, 0.83, 1.0]], np.float32)
  mesh = b.add_mesh(tet, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
  b.add_instance("mesh", mesh=mesh, material=grey, xform=np.c_[np.eye(3), [-1.4, -0.6, -1.0]], path="tet")
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=10.0, radius=0.05, xform=np.c_[np.eye(3), [0, 0, 2.0]]))
  return b.build()


def _renderer(cameras: list[rw.CameraSpec], *, half: bool = False) -> rw.Renderer[None]:
  """Two worlds (world 1's box turned and moved), the outputs the checks read."""
  depth = rw.settings.DepthOutput(dtype="float16" if half else "float32")
  outputs = rw.Outputs.of("rgb", "seg", "position", "normal", depth=depth)
  r = rw.Renderer(
    rw.upload(_scene(), device="cpu"),
    nworld=NWORLD,
    cameras=cameras,
    resolution=SIZE,
    settings=rw.RenderSettings(gi=NO_GI, outputs=outputs, performance=rw.settings.Performance(traversal="software")),
    per_world=rw.PerWorld(pose=["box"]),
  )
  pos, rot = r.per_world.columns("instance.pos", "box"), r.per_world.columns("instance.rot", "box")
  p, q = pos.numpy(), rot.numpy()
  p[1, 0] = (0.4, -0.1, 0.3)
  q[1, 0] = _TURN
  pos.assign(p)
  rot.assign(q)
  eye = np.array([0.3, -3.0, 1.6])
  poses = np.zeros((NWORLD, len(cameras), 7), np.float32)
  for c in range(len(cameras)):
    poses[:, c] = _look_at(eye + np.array([0.2 * c, 0.0, 0.0]), np.array([0.0, 0.0, 0.3]))
  r.camera_pose.assign(poses)
  return r


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
  """(x, y, z, qx, qy, qz, qw) of a camera at `eye` looking at `target` (-z forward, +z up)."""
  f = (target - eye) / math.hypot(*(target - eye).tolist())
  x = np.cross(f, [0.0, 0.0, 1.0])
  x /= math.hypot(*x.tolist())
  m = np.stack([x, np.cross(x, f), -f], axis=1)
  w = np.sqrt(max(0.0, 1.0 + np.trace(m))) / 2.0
  q = np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])
  return np.concatenate([eye, q])


def _np(a: wp.array | None) -> np.ndarray:
  """The host copy of an output that was requested."""
  assert a is not None
  return a.numpy()


def _flat(a: wp.array, r: rw.Renderer[None]) -> np.ndarray:
  """(nworld, ncam, H, W, ...) of a flat (nworld, pixels) array."""
  w, h = SIZE
  v = a.numpy()
  return v.reshape(r.nworld, r.ncam, h, w, *v.shape[2:])


@pytest.mark.parametrize(
  "camera",
  [
    rw.CameraSpec(fovy=50.0),
    rw.CameraSpec(intrinsics=(30.0, 28.0, 1.5, -0.8), sensor_size=(24.0, 16.0), fovy=50.0),
    rw.CameraSpec(projection="orthographic", fovy=3.0),
  ],
  ids=["fovy", "intrinsics", "orthographic"],
)
def test_exported_cameras_project_the_rendered_points_onto_their_pixels(camera: rw.CameraSpec) -> None:
  """Each hit's world-space position (the position output) projected through the exported
  camera-to-world pose and K lands on the centre of its own pixel (OpenCV conventions: pixel
  centres at integer coordinates, y down, z forward)."""
  with _renderer([camera, camera]) as r:
    out = r.render()
    info = r.camera_info()
    assert info.frame == 1 and info.resolution == (SIZE, SIZE)
    assert info.models == (("orthographic",) * 2 if camera.projection == "orthographic" else ("pinhole",) * 2)
    position = _np(out.position)
    seg = _np(out.seg)
    k, pose = info.k.numpy(), info.cam_to_world.numpy()
  h, w = SIZE[1], SIZE[0]
  v, u = np.mgrid[0:h, 0:w].astype(np.float64)
  for world in range(NWORLD):
    for c in range(2):
      hit = seg[world, c] >= 0
      p = position[world, c][hit].astype(np.float64)
      world_to_cam = np.linalg.inv(pose[world, c].astype(np.float64))
      x = p @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
      assert (x[:, 2] > 0).all()  # in front: z forward
      plane = x[:, :2] if camera.projection == "orthographic" else x[:, :2] / x[:, 2:3]
      uv = plane @ k[world, c][:2, :2].T + k[world, c][:2, 2]
      np.testing.assert_allclose(uv[:, 0], u[hit], atol=2e-3)
      np.testing.assert_allclose(uv[:, 1], v[hit], atol=2e-3)


def test_camera_info_is_written_in_place() -> None:
  """`out=` writes the same arrays (no allocation) and follows the new poses and frame index."""
  with _renderer([rw.CameraSpec(fovy=50.0)]) as r:
    r.render()
    info = r.camera_info()
    before = info.cam_to_world.numpy().copy()
    poses = r.camera_pose.numpy()
    poses[:, 0, 0] += 0.5
    r.camera_pose.assign(poses)
    r.render()
    again = r.camera_info(out=info)
  assert again.k is info.k and again.cam_to_world is info.cam_to_world and again.frame == 2
  np.testing.assert_allclose(again.cam_to_world.numpy()[:, 0, 0, 3], before[:, 0, 0, 3] + 0.5, atol=1e-6)


def test_skipped_worlds_keep_the_ground_truth_of_their_kept_image() -> None:
  """With `render(active=...)`, a skipped world keeps its last image: its camera export, camera
  normals and NOCS written in place keep the values of that image (the cameras moved since), the
  rendered world's follow its new camera."""
  with _renderer([rw.CameraSpec(fovy=50.0)]) as r:
    r.render()
    info, normals, nocs = r.camera_info(), r.camera_normals(), r.nocs()
    before = [info.cam_to_world.numpy().copy(), normals.numpy().copy(), nocs.numpy().copy()]
    poses = r.camera_pose.numpy()
    poses[:, 0, 0] += 0.5
    r.camera_pose.assign(poses)
    r.render(active=wp.array([0], dtype=wp.int32, device=r.device))
    after = [r.camera_info(out=info).cam_to_world.numpy(), r.camera_normals(out=normals).numpy(), r.nocs(out=nocs).numpy()]
  for old, new in zip(before, after, strict=True):
    np.testing.assert_array_equal(new[1], old[1])
  np.testing.assert_allclose(after[0][0, 0, 0, 3], before[0][0, 0, 0, 3] + 0.5, atol=1e-6)
  assert not np.array_equal(after[2][0], before[2][0])


def test_skipped_worlds_keep_their_point_cloud() -> None:
  """`point_cloud(out=...)` after `render(active=...)`: a skipped world keeps the cloud of its
  kept depth (BUG-768: it was unprojected through this frame's moved cameras), the rendered
  world's follows its new camera (its points are its position output's hits)."""
  with _renderer([rw.CameraSpec(fovy=50.0)]) as r:
    r.render()
    cloud = r.point_cloud()
    before = (cloud.points.numpy().copy(), cloud.count.numpy().copy())
    poses = r.camera_pose.numpy()
    poses[:, 0, 0] += 0.5
    r.camera_pose.assign(poses)
    out = r.render(active=wp.array([0], dtype=wp.int32, device=r.device))
    position, seg = _np(out.position), _np(out.seg)
    r.point_cloud(out=cloud)
    points, count = cloud.points.numpy(), cloud.count.numpy()
  np.testing.assert_array_equal(points[1], before[0][1])
  assert count[1] == before[1][1]
  hit = seg[0] >= 0
  assert count[0] == hit.sum()
  np.testing.assert_allclose(points[0, : count[0]], position[0][hit], atol=1e-4)


def test_pinhole_and_lens_cameras_share_one_point_cloud_kernel() -> None:
  """One gather kernel serves pinhole, OpenCV-distorted and fisheye cameras (BUG-650: a lens
  camera's cloud compiled a second, unbundled kernel): no Warp module is created by a lens
  camera's cloud after a pinhole one's, and each camera's points are its position output's hits."""
  from warp._src import context as wcontext

  cameras = [
    rw.CameraSpec(fovy=50.0),
    rw.CameraSpec(fovy=50.0, distortion=(-0.05, 0.01, 0.0, 0.0, 0.0)),
    rw.CameraSpec(projection="fisheye", fovy=160.0, distortion=(0.02, -0.01, 0.0, 0.0)),
  ]
  with _renderer(cameras[:1]) as r:
    r.render()
    r.point_cloud()
  modules = set(wcontext.user_modules)
  with _renderer(cameras) as r:
    out = r.render()
    position, seg = _np(out.position), _np(out.seg)
    clouds = [r.point_cloud(cameras=[c]) for c in range(len(cameras))]
    for c, cloud in enumerate(clouds):
      points, count = cloud.points.numpy(), cloud.count.numpy()
      for world in range(NWORLD):
        hit = seg[world, c] >= 0
        assert count[world] == hit.sum() > 10
        expected = position[world, c][hit]
        np.testing.assert_allclose(points[world, : count[world]], expected, atol=2e-3)
  assert not {m for m in set(wcontext.user_modules) - modules if "pointcloud" in m or "gather" in m}


def test_the_frame_index_counts_rendered_frames() -> None:
  """`CameraInfo.frame` counts the renders that ran, not a call that raised before rendering."""
  with _renderer([rw.CameraSpec(fovy=50.0)]) as r:
    r.render()
    with pytest.raises(ValueError, match="active"):
      r.render(active=wp.array([0.0], dtype=float, device=r.device))
    assert r.camera_info().frame == 1
    r.render()
    assert r.camera_info().frame == 2


def test_camera_normals_are_the_world_normals_in_the_opencv_camera_frame() -> None:
  """The camera-frame normals are the normal output rotated by the exported pose; the floor (+z)
  seen from above points up the image (-y) and towards the camera (-z); misses are zero."""
  with _renderer([rw.CameraSpec(fovy=50.0)]) as r:
    out = r.render()
    normals = r.camera_normals()
    again = r.camera_normals(out=normals)
    world, seg = _np(out.normal), _np(out.seg)
    pose = r.camera_info().cam_to_world.numpy()
    got = _flat(normals, r)
  assert again is normals
  expected = np.einsum("ncji,nchwj->nchwi", pose[:, :, :3, :3], world)
  np.testing.assert_allclose(got, expected, atol=1e-5)
  floor = got[0, 0][seg[0, 0] == 0]
  assert len(floor) and (floor[:, 1] < -0.3).all() and (floor[:, 2] < 0.0).all()
  assert (got[seg < 0] == 0.0).all()


@pytest.mark.parametrize("half", [False, True], ids=["float32", "float16"])
def test_nocs_are_the_hits_in_their_instances_unit_box(half: bool) -> None:
  """NOCS: each hit's position output in its instance's frame (world 1's box turned and moved,
  the mesh's box off its origin), centred on its local box over the box's diagonal, + 0.5; inside
  the unit cube for the box and the mesh; zero on misses."""
  with _renderer([rw.CameraSpec(fovy=50.0, distortion=(-0.05, 0.01, 0.0, 0.0, 0.0))], half=half) as r:
    out = r.render()
    nocs = _flat(r.nocs(), r)
    position, seg = _np(out.position).astype(np.float64), _np(out.seg)
  centre = np.asarray(BOX[0])
  diagonal = 2.0 * math.hypot(*BOX[1])
  box = 1  # floor, box, tet
  tolerance = 2e-2 if half else 1e-3
  for world, (at, turn) in enumerate([(centre, np.eye(3)), (np.array([0.4, -0.1, 0.3]), _TURN)]):
    hit = seg[world, 0] == box
    assert hit.sum() > 3
    local = (position[world, 0][hit] - at) @ turn  # R^T (p - t), row vectors
    expected = local / diagonal + 0.5
    np.testing.assert_allclose(nocs[world, 0][hit], expected, atol=tolerance)
  inside = nocs[:, 0][(seg[:, 0] >= 0)]
  assert ((inside >= -tolerance) & (inside <= 1.0 + tolerance)).all()
  assert (nocs[seg < 0] == 0.0).all()


def test_exported_fisheye_projects_through_kannala_brandt() -> None:
  """A fisheye camera exports "kannala_brandt" with its coefficients: the hits projected through
  theta_d = theta (1 + k1 theta^2 + k2 theta^4 + ...) and K land on their pixels."""
  camera = rw.CameraSpec(projection="fisheye", fovy=160.0, distortion=(0.02, -0.01, 0.0, 0.0))
  with _renderer([camera]) as r:
    out = r.render()
    info = r.camera_info()
    position, seg = _np(out.position), _np(out.seg)
    k, pose, coef = info.k.numpy(), info.cam_to_world.numpy(), info.distortion.numpy()
  assert info.models == ("kannala_brandt",)
  h, w = SIZE[1], SIZE[0]
  v, u = np.mgrid[0:h, 0:w].astype(np.float64)
  for world in range(NWORLD):
    hit = seg[world, 0] >= 0
    world_to_cam = np.linalg.inv(pose[world, 0].astype(np.float64))
    x = position[world, 0][hit].astype(np.float64) @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    r_xy = np.linalg.norm(x[:, :2], axis=1)
    theta = np.arctan2(r_xy, x[:, 2])
    k1, k2, k3, k4 = coef[world, 0][:4]
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    plane = x[:, :2] * (theta_d / np.maximum(r_xy, 1e-12))[:, None]
    uv = plane @ k[world, 0][:2, :2].T + k[world, 0][:2, 2]
    np.testing.assert_allclose(uv[:, 0], u[hit], atol=5e-3)
    np.testing.assert_allclose(uv[:, 1], v[hit], atol=5e-3)


def test_camera_log_writes_the_episode_cameras(tmp_path) -> None:  # pytest fixture
  """The datagen camera log (`datagen.sinks.CameraLog`, development repository only) writes every
  frame's exported cameras, frame indices and timestamps beside the episode's outputs."""
  import json

  from conftest import internal_module

  sinks = internal_module("datagen.sinks")

  with _renderer([rw.CameraSpec(fovy=50.0), rw.CameraSpec(projection="fisheye", fovy=180.0)]) as r:
    log = sinks.CameraLog("cpu", steps=3)
    info = None
    for step in range(3):
      r.render()
      info = r.camera_info(out=info)
      log.add(info, time_s=np.full(NWORLD, 0.01 * step))
    log.write(tmp_path)
    assert info is not None
    expected = info.k.numpy()
  saved = np.load(tmp_path / "cameras.npz")
  assert saved["k"].shape == (3, NWORLD, 2, 3, 3) and saved["cam_to_world"].shape == (3, NWORLD, 2, 4, 4)
  assert saved["distortion"].shape == (3, NWORLD, 2, 12)
  assert saved["frame"].tolist() == [1, 2, 3] and np.allclose(saved["time"][:, 0], [0.0, 0.01, 0.02])
  np.testing.assert_array_equal(saved["k"][-1], expected)
  meta = json.loads((tmp_path / "cameras.json").read_text())
  assert meta["models"] == ["pinhole", "kannala_brandt"] and meta["resolution"] == [list(SIZE)] * 2
