"""Mesh instances scaled by their pose (s R, `pipeline/upload/meshes.py`): per-world scales written
at run time render like mesh copies scaled by hand and compile nothing; their NOCS stay in the mesh
frame and their motion vectors follow the scale (CPU, software traversal)."""

from __future__ import annotations

import math

import numpy as np
from conftest import DOWN, NO_GI, required
from fixtures.conformance import module_loads

import raywarp as rw

RES = 40
FOVY = 25.0
EYE = 6.0
RADIUS = 0.6
SCALE = 1.8
# The ball's per-world scales: world 0 as authored, world 1 scaled.
SCALES = np.array([1.0, SCALE], np.float32)


def _ball(n: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(points, triangles, uvs) of a UV sphere of `RADIUS` about the origin (outward faces)."""
  th, ph = np.meshgrid(np.linspace(0.05, np.pi - 0.05, n), np.linspace(0.0, 2.0 * np.pi, 2 * n), indexing="ij")
  unit = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
  uv = np.stack([ph / (2.0 * np.pi), 1.0 - th / np.pi], -1).reshape(-1, 2)
  w = 2 * n
  tris = []
  for i in range(n - 1):
    for j in range(w - 1):
      a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
      tris += [(c, b, a), (c, d, b)]
  return RADIUS * unit, np.array(tris), uv


def _scene(points_scale: float) -> rw.Scene:
  """A floor and a checker-textured ball (its points scaled by `points_scale`: a copy baked by
  hand) under a sun."""
  pts, tris, uv = _ball()
  b = rw.SceneBuilder()
  img = np.zeros((8, 8, 4), np.uint8)
  img[..., 3] = 255
  img[::2, :, 0] = 255
  img[:, ::2, 1] = 255
  tex = b.add_texture(img)
  checker = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=0.4).with_texture(
      rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(tex)
    )
  )
  grey = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5)))
  b.add_instance("box", material=grey, size=(4.0, 4.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -1.3]], path="floor")
  mesh = b.add_mesh(points_scale * pts, tris, normals=pts / RADIUS, uvs=uv)
  b.add_instance("mesh", mesh=mesh, material=checker, xform=np.c_[np.eye(3), np.zeros(3)], path="ball")
  sun = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=5.0, angle=0.0, xform=np.c_[sun, np.zeros(3)]))
  return b.build()


def _renderer(host: rw.Scene, nworld: int, *, per_world: rw.PerWorld | None = None) -> rw.Renderer[None]:
  """A software-traced CPU renderer of `host` looking down at the ball from z = EYE."""
  settings = rw.RenderSettings(
    gi=NO_GI,
    outputs=rw.Outputs.of("hdr", "seg", "position", "motion", "depth"),
    performance=rw.settings.Performance(traversal="software"),
    environment=rw.settings.Environment(enabled=False),
  )
  r = rw.Renderer(
    rw.upload(host, "cpu"),
    nworld=nworld,
    cameras=[rw.CameraSpec(fovy=FOVY)],
    resolution=(RES, RES),
    settings=settings,
    per_world=per_world,
  )
  r.camera_pose.assign(np.tile(np.array([[[0.0, 0.0, EYE, *DOWN]]], np.float32), (nworld, 1, 1)))
  return r


def _scaled(r: rw.Renderer[None], scales: np.ndarray) -> None:
  """Writes the ball's per-world scale into its rotation column (s R, R the identity)."""
  rot = r.per_world.columns("instance.rot", "ball")
  rot.assign((scales[:, None, None, None] * np.eye(3, dtype=np.float32)).astype(np.float32))


def _images(r: rw.Renderer[None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(hdr, seg, position) of the last render, (nworld, RES * RES, ...)."""
  out = r.outputs
  hdr = required(out.hdr).numpy().reshape(r.nworld, RES * RES, 3)
  seg = required(out.seg).numpy().reshape(r.nworld, RES * RES)
  return hdr, seg, required(out.position).numpy().reshape(r.nworld, RES * RES, 3)


def test_per_world_mesh_scale_renders_like_scaled_copies_and_compiles_nothing() -> None:
  """World w's ball scaled by its pose (written per world, then rewritten) renders like the ball's
  points scaled by hand: same silhouette, colours, positions; no Warp module is built for a new
  scale (the scale is pose data)."""
  r = _renderer(_scene(1.0), 2, per_world=rw.PerWorld(pose=["ball"]))
  _scaled(r, SCALES)
  r.render()
  features = r.features
  with module_loads() as loads:
    for scales in (np.array([0.7, 1.3], np.float32), SCALES):
      _scaled(r, scales)
      r.render()
  assert r.features is features and not loads, loads
  hdr, seg, pos = _images(r)
  ball = 1
  for world, s in enumerate(SCALES.tolist()):
    with _renderer(_scene(s), 1) as ref:
      ref.render()
      rhdr, rseg, rpos = _images(ref)
    assert (seg[world] == ball).sum() > 100
    assert np.mean(seg[world] != rseg[0]) < 0.01  # silhouettes up to float rounding
    same = seg[world] == rseg[0]
    np.testing.assert_allclose(hdr[world][same], rhdr[0][same], atol=2e-3)
    np.testing.assert_allclose(pos[world][same], rpos[0][same], atol=1e-4)
  r.close()


def test_scaled_mesh_nocs_and_motion_follow_the_scale() -> None:
  """NOCS of a scaled ball are its hits in the mesh frame (p / s over the mesh box); a scale
  change between frames moves each hit radially in the image: the previous point p s0 / s1."""
  r = _renderer(_scene(1.0), 2, per_world=rw.PerWorld(pose=["ball"]))
  _scaled(r, np.ones(2, np.float32))
  r.render()
  _scaled(r, SCALES)
  r.render()
  _hdr, seg, pos = _images(r)
  nocs = r.nocs().numpy().reshape(2, RES * RES, 3)
  motion = required(r.outputs.motion).numpy().reshape(2, RES * RES, 2)
  r.close()
  pts = _ball()[0]
  lo, hi = pts.min(axis=0), pts.max(axis=0)
  diagonal = math.dist(hi.tolist(), lo.tolist())
  pix = (np.stack(np.meshgrid(np.arange(RES), np.arange(RES)), -1).reshape(-1, 2) + 0.5) - RES / 2
  for world, s in enumerate(SCALES.tolist()):
    hit = seg[world] == 1
    p = pos[world][hit].astype(np.float64)
    np.testing.assert_allclose(nocs[world][hit], (p / s - 0.5 * (lo + hi)) / diagonal + 0.5, atol=2e-3)
    # Looking straight down from z = EYE, a point's image offset from the centre is f x / (EYE - z).
    k = 1.0 / s
    ratio = k * (EYE - p[:, 2]) / (EYE - k * p[:, 2])
    expected = pix[hit] * (ratio - 1.0)[:, None]
    got = motion[world][hit]
    assert math.isclose(float(np.abs(got).max()), 0.0, abs_tol=1e-4) if world == 0 else np.abs(got).max() > 1.0
    np.testing.assert_allclose(np.abs(got), np.abs(expected), atol=0.05 + 0.02 * np.abs(expected).max())
