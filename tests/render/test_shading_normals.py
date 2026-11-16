"""Shading-normal regressions: smooth-mesh normals at silhouettes (B9) and Toksvig specular
anti-aliasing on 8-bit normal maps (B11)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import render_host, required

import raywarp as rw
from raywarp.settings.fields import OutputKind


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
  """Unit icosphere (points, counter-clockwise triangles); its exact normals are its points."""
  t = (1.0 + 5.0**0.5) / 2.0
  corners = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t)]
  corners += [(t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
  points = [(q := np.asarray(p, float)) / np.linalg.norm(q, keepdims=True) for p in corners]
  faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8)]
  faces += [(3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
  for _ in range(subdivisions):
    midpoints: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int, midpoints: dict[tuple[int, int], int] = midpoints) -> int:
      key = (min(a, b), max(a, b))
      if key not in midpoints:
        m = points[a] + points[b]
        points.append(m / np.linalg.norm(m, keepdims=True))
        midpoints[key] = len(points) - 1
      return midpoints[key]

    split = []
    for a, b, c in faces:
      ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
      split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    faces = split
  return np.asarray(points), np.asarray(faces)


def test_smooth_mesh_silhouette_normals_face_the_light_behind():
  """B9: an 80-face smooth icosphere seen from +z, a point light behind and above it. Near the
  silhouette the interpolated normals face away from the eye; they must keep pointing outward
  (the top rim catches the light), not be mirrored into the sphere (which lit the bottom rim)."""
  b = rw.SceneBuilder()
  points, tris = _icosphere(1)
  mesh = b.add_mesh(points, tris, normals=points)
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=0.4))
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[np.eye(3), np.zeros(3)])
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.SPHERE,
      intensity=200.0,
      radius=0.0,
      treat_as_point=True,
      shadow_enable=False,
      xform=np.c_[np.eye(3), [0.0, 2.0, -3.0]],
    )
  )
  res = 96
  r = render_host(b.build(), (0.0, 0.0, 4.0), fovy=32.0, res=res)
  radiance = required(r.outputs.hdr).numpy()[0, 0, ..., :3].sum(-1)
  # Image row 0 is the top. No outward normal of the lower half sees a light above the sphere.
  top, bottom = radiance[: res // 2], radiance[res // 2 :]
  assert bottom.max() == 0.0, f"{int((bottom > 0).sum())} lower-rim pixels lit by mirrored normals"
  assert (top > 1.0).sum() >= 4


def _normal_map_pixel(texels: np.ndarray | None, uv_extent: float, tilt: np.ndarray | None = None) -> np.ndarray:
  """Mean radiance of a 1 m metal quad (roughness 0.2) under a normal map `texels` ((H, W, 3)
  uint8, UVs 0..uv_extent; None: no map), its plane turned so that its normal is `tilt` (None:
  +z), seen straight down from 1 m with a point light just behind the camera."""
  b = rw.SceneBuilder()
  points = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
  uvs = np.array([[0.0, uv_extent], [uv_extent, uv_extent], [uv_extent, 0.0], [0.0, 0.0]])
  mesh = b.add_mesh(points, np.array([[0, 1, 2], [0, 2, 3]]), uvs=uvs)
  normal = rw.scene.TextureRef(
    b.add_texture(texels if texels is not None else np.zeros((1, 1, 3), np.uint8), colorspace="linear"),
    scale=(2.0, 2.0, 2.0, 1.0),
    bias=(-1.0, -1.0, -1.0, 0.0),
  )
  m = b.add_material(
    rw.Material(
      base_color=(0.9, 0.7, 0.3),
      base_metalness=1.0,
      specular_roughness=0.2,
      textures=((rw.scene.Slot.GEOMETRY_NORMAL, normal),) if texels is not None else (),
    )
  )
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[_turn_to(tilt), np.zeros(3)])
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.SPHERE,
      intensity=100.0,
      radius=0.0,
      treat_as_point=True,
      shadow_enable=False,
      xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]],
    )
  )
  r = render_host(b.build(), (0.0, 0.0, 1.0), fovy=1.0, res=2)
  assert r.features.normal_map == (texels is not None) and r.features.specular_aa
  return required(r.outputs.hdr).numpy()[0, 0, ..., :3].reshape(-1, 3).mean(0)


def _turn_to(normal: np.ndarray | None) -> np.ndarray:
  """The rotation taking +z to the unit vector `normal` about their common perpendicular (None:
  the identity)."""
  if normal is None:
    return np.eye(3)
  n = normal / math.hypot(*(float(x) for x in normal))
  axis = np.cross([0.0, 0.0, 1.0], n)
  s, c = math.hypot(*(float(x) for x in axis)), float(n[2])
  k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]) / max(s, 1e-12)
  return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def test_toksvig_ignores_normal_map_quantisation():
  """B11: a uniform normal map tilted 34 degrees whose 8-bit texel decodes 0.5 % short (the
  worst rounding of that tilt), magnified (one texel spans the view): the filtered normal has no
  spread, so specular AA must not widen the roughness: it shades as the quad turned to the same
  normal without a map. Rounding alone used to add ~0.005 to α² (0.0016 at roughness 0.2) and
  speckled glossy normal-mapped metal."""
  texel = np.array([191, 158, 233], np.uint8)
  decoded = texel / 255.0 * 2.0 - 1.0
  assert math.hypot(*(float(x) for x in decoded)) < 0.996
  flat = np.broadcast_to(texel, (4, 4, 3)).copy()
  np.testing.assert_allclose(_normal_map_pixel(flat, 1.0), _normal_map_pixel(None, 1.0, decoded), rtol=1e-2)


def test_toksvig_widens_minified_normal_spread():
  """Specular AA keeps its purpose: a 1-texel checker of normals tilted +-30 degrees, minified
  (~18 texels per pixel), averages to a normal of length cos 30: Toksvig widens the lobe, and
  the highlight straight back at the light drops well below the unfiltered (mirror-like) peak."""
  tilt = np.array([[np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)], [-np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)]])
  texels = np.round((tilt + 1.0) / 2.0 * 255.0).astype(np.uint8)
  checker = texels[(np.arange(64)[:, None] + np.arange(64)[None, :]) % 2]
  wide = _normal_map_pixel(checker, 32.0)
  sharp = _normal_map_pixel(None, 1.0)  # the mean normal (+z), unfiltered
  assert wide.mean() < 0.5 * sharp.mean()


def _render_mesh(
  b: rw.SceneBuilder, pose: list[float], kind: OutputKind, fovy: float, res: int, **settings_kw: object
) -> np.ndarray:
  """Output `kind` (camera 0, world 0) of the built scene seen from `pose` (x y z qx qy qz qw)."""
  settings = {"surfaces.backface_culling": False, **settings_kw}
  x, y, z, qx, qy, qz, qw = pose
  r = render_host(b.build(), (x, y, z), (qx, qy, qz, qw), fovy=fovy, res=res, kinds=frozenset({kind}), settings=settings)
  return getattr(r.outputs, kind).numpy()[0, 0]


# Camera poses (x, y, z, qx, qy, qz, qw): looking down -z from +z, and up +z from -z.
_ABOVE = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
_BELOW = [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]


def test_back_face_normal_map_is_the_front_one_turned():
  """B14: a quad's normal map (tilted 30 degrees towards +u = +x) seen from below. The back's
  normal is minus the front's (the relief is the same surface); it used to keep the front's +x
  slope (mirrored relief)."""
  b = rw.SceneBuilder()
  points = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
  uvs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
  mesh = b.add_mesh(points, np.array([[0, 1, 2], [0, 2, 3]]), uvs=uvs)
  tilt = np.array([np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)])
  texels = np.broadcast_to(np.round((tilt + 1.0) / 2.0 * 255.0).astype(np.uint8), (4, 4, 3)).copy()
  normal = rw.scene.TextureRef(
    b.add_texture(texels if texels is not None else np.zeros((1, 1, 3), np.uint8), colorspace="linear"),
    scale=(2.0, 2.0, 2.0, 1.0),
    bias=(-1.0, -1.0, -1.0, 0.0),
  )
  m = b.add_material(
    rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=0.5, textures=((rw.scene.Slot.GEOMETRY_NORMAL, normal),))
  )
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[np.eye(3), np.zeros(3)])
  front = _render_mesh(b, _ABOVE, "normal", 10.0, 4).reshape(-1, 3)
  back = _render_mesh(b, _BELOW, "normal", 10.0, 4).reshape(-1, 3)
  assert front[:, 0].mean() > 0.4  # the tilt towards +x
  np.testing.assert_allclose(back, -front, atol=2e-3)


def test_back_faces_of_smooth_meshes_are_smooth():
  """B14: the inside of a smooth 320-face icosphere, seen from its centre: the interpolated
  normals (turned to the viewer) vary across every triangle; back faces used to fall back to
  the flat face normal (a handful of distinct normals)."""
  b = rw.SceneBuilder()
  points, tris = _icosphere(2)
  mesh = b.add_mesh(points, tris, normals=points)
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=0.5))
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[np.eye(3), np.zeros(3)])
  n = _render_mesh(b, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "normal", 40.0, 16).reshape(-1, 3)
  assert np.all(n[:, 2] > 0.5)  # facing the viewer (looking down -z: the normals point to +z)
  distinct = len(np.unique(n.round(3), axis=0))
  assert distinct > 128, f"{distinct} distinct normals over 256 pixels: faceted back faces"


def test_low_poly_smooth_sphere_has_no_shadow_terminator_notches():
  """B46: an 80-face smooth icosphere lit from the side by a directional light. Its only
  shadow is the terminator; rays leaving the flat triangles from the hit point were blocked by
  the triangle's own neighbours where the smooth normal still faces the light (dark facet
  notches). From Hanika's lifted point they escape: shadowed and unshadowed renders agree."""
  b = rw.SceneBuilder()
  points, tris = _icosphere(1)
  mesh = b.add_mesh(points, tris, normals=points)
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[np.eye(3), np.zeros(3)])
  # Light travelling along -x (from +x), slightly towards the camera's side.
  z = np.array([1.0, 0.0, 0.25]) / np.linalg.norm(np.array([1.0, 0.0, 0.25]), keepdims=True)
  x = np.cross([0.0, 1.0, 0.0], z)
  x /= np.linalg.norm(x, keepdims=True)
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.DISTANT,
      intensity=5.0,
      angle=0.0,
      shadow_enable=True,
      xform=np.c_[x, np.cross(z, x), z, np.zeros(3)],
    )
  )
  pose = [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0]
  lit = _render_mesh(b, pose, "hdr", 32.0, 64)[..., :3].sum(-1)
  free = _render_mesh(b, pose, "hdr", 32.0, 64, **{"shadows.enabled": False})[..., :3].sum(-1)
  notches = int(((free > 0.05 * free.max()) & (lit < 0.5 * free)).sum())
  assert notches <= 4, f"{notches} pixels self-shadowed where the smooth normal faces the light"


@pytest.mark.parametrize("scale", [100.0, 1000.0])
def test_shadow_rays_leave_along_the_geometric_normal(scale):
  """B17: a 200 m quad seen from 100 m under a normal map tilted 85 degrees, the sun 30 degrees
  up on the tilt's side. Offset along the mapped normal, the shadow origin rose only 0.09 x the
  ray offset above the plane, below the hit point's float error at that distance: acne. Offset
  along the geometric normal, every pixel sees the sun. B61: the same at 1 km, where a fixed
  1e-4 offset is below the float error of the hit point (83 / 1024 pixels of acne): the offset
  grows with the point's magnitude."""
  b = rw.SceneBuilder()
  points = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]) * scale
  uvs = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
  mesh = b.add_mesh(points, np.array([[0, 1, 2], [0, 2, 3]]), uvs=uvs)
  tilt = np.array([np.sin(np.radians(85.0)), 0.0, np.cos(np.radians(85.0))])
  texels = np.broadcast_to(np.round((tilt + 1.0) / 2.0 * 255.0).astype(np.uint8), (4, 4, 3)).copy()
  normal = rw.scene.TextureRef(
    b.add_texture(texels if texels is not None else np.zeros((1, 1, 3), np.uint8), colorspace="linear"),
    scale=(2.0, 2.0, 2.0, 1.0),
    bias=(-1.0, -1.0, -1.0, 0.0),
  )
  m = b.add_material(
    rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0, textures=((rw.scene.Slot.GEOMETRY_NORMAL, normal),))
  )
  b.add_instance("mesh", mesh=mesh, material=m, xform=np.c_[np.eye(3), np.zeros(3)])
  z = np.array([np.cos(np.radians(30.0)), 0.0, np.sin(np.radians(30.0))])
  x = np.cross([0.0, 1.0, 0.0], z)
  x /= np.linalg.norm(x, keepdims=True)
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.DISTANT,
      intensity=5.0,
      angle=0.0,
      shadow_enable=True,
      xform=np.c_[x, np.cross(z, x), z, np.zeros(3)],
    )
  )
  pose = [0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0]
  lit = _render_mesh(b, pose, "hdr", 60.0, 32)[..., :3].sum(-1)
  free = _render_mesh(b, pose, "hdr", 60.0, 32, **{"shadows.enabled": False})[..., :3].sum(-1)
  assert free.min() > 0.0
  acne = int((lit < 0.5 * free).sum())
  assert acne == 0, f"{acne} of {lit.size} pixels self-shadowed (acne)"
