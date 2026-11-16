"""Where a texture lands on primitive shapes (spheres, boxes, ...: MuJoCo's texrepeat / texuniform
mapping), and the vertex-normal acceptance rule of mesh shading.

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0), unchanged except for
names and imports.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def texture_plane(
  # In:
  local: wp.vec3,
  size: wp.vec3,
  repeat: wp.vec2,
  tex_uniform: bool,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """(uv, offset, repeat) of a plane hit (MuJoCo's mapping; infinite axes tile per 2 m)."""
  # Evaluate X and Y independently to support mixed finite/infinite planes
  # (render_context.c lines 211 & 219)
  u = 0.5 * local[0]
  rep_u = repeat[0]
  if size[0] > 0.0:
    s0 = wp.max(size[0], 1e-6)
    u = (local[0] + size[0]) / (2.0 * s0)
    if tex_uniform:
      rep_u = repeat[0] * size[0]

  v = -0.5 * local[1]
  rep_v = repeat[1]
  if size[1] > 0.0:
    s1 = wp.max(size[1], 1e-6)
    v = (size[1] - local[1]) / (2.0 * s1)
    if tex_uniform:
      rep_v = repeat[1] * size[1]

  off = -0.5 if (size[0] <= 0.0 or size[1] <= 0.0) else 0.0
  return wp.vec2(u, v), wp.vec2(off, off), wp.vec2(rep_u, rep_v)


@wp.func
def texture_sphere_ellipsoid(
  # In:
  local: wp.vec3,
  size: wp.vec3,
  repeat: wp.vec2,
  tex_uniform: bool,
  is_sphere: bool,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """(uv, offset, repeat) of a sphere / ellipsoid hit: azimuth, elevation."""
  radii = size
  if is_sphere:
    radii = wp.vec3(size[0], size[0], size[0])
  rx = wp.max(radii[0], 1e-6)
  ry = wp.max(radii[1], 1e-6)
  rz = wp.max(radii[2], 1e-6)
  pt = wp.vec3(local[0] / rx, local[1] / ry, local[2] / rz)
  inv_len = 1.0 / wp.max(wp.length(pt), 1e-6)
  d = pt * inv_len

  azimuth = wp.atan2(d[1], d[0])
  if azimuth < 0.0:
    azimuth += wp.static(2.0 * wp.pi)
  elevation = wp.asin(wp.clamp(d[2], -1.0, 1.0))
  u = azimuth * wp.static(0.5 / wp.pi)
  v = 0.5 - elevation * wp.static(1.0 / wp.pi)
  rep = repeat
  if tex_uniform:
    rep = wp.vec2(repeat[0] * radii[0], repeat[1] * radii[1])
  return wp.vec2(u, v), wp.vec2(0.0, 0.0), rep


@wp.func
def texture_cylinder(
  # In:
  local: wp.vec3,
  local_normal: wp.vec3,
  size: wp.vec3,
  repeat: wp.vec2,
  tex_uniform: bool,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """(uv, offset, repeat) of a cylinder hit: caps as disks, the body by azimuth and height."""
  r = wp.max(size[0], 1e-6)
  h = wp.max(size[1], 1e-6)
  if wp.abs(local_normal[2]) > 0.7071:
    # Top and bottom caps match MuJoCo disk(+1) and disk(-1)
    uv = wp.vec2(0.5 + 0.5 * (local[0] / r), 0.5 + 0.5 * (local[1] / r))
  else:
    # Cylindrical body
    azimuth = wp.atan2(local[1], local[0])
    if azimuth < 0.0:
      azimuth += wp.static(2.0 * wp.pi)
    u = azimuth * wp.static(0.5 / wp.pi)
    v = (h - local[2]) / (2.0 * h)
    uv = wp.vec2(u, v)
  rep = repeat
  if tex_uniform:
    rep = wp.vec2(repeat[0] * size[0], repeat[1] * size[0])
  return uv, wp.vec2(0.0, 0.0), rep


@wp.func
def texture_capsule(
  # In:
  local: wp.vec3,
  size: wp.vec3,
  repeat: wp.vec2,
  tex_uniform: bool,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """(uv, offset, repeat) of a capsule hit: azimuth; height on the body, arc on the caps."""
  r = wp.max(size[0], 1e-6)
  h = wp.max(size[1], 1e-6)
  z = local[2]
  azimuth = wp.atan2(local[1], local[0])
  if azimuth < 0.0:
    azimuth += wp.static(2.0 * wp.pi)
  u = azimuth * wp.static(0.5 / wp.pi)

  if z > h:
    dz = wp.clamp((z - h) / r, 0.0, 1.0)
    v = 1.0 - wp.asin(dz) * wp.static(2.0 / wp.pi)
  elif z < -h:
    dz = wp.clamp((-h - z) / r, 0.0, 1.0)
    v = wp.asin(dz) * wp.static(2.0 / wp.pi)
  else:
    v = (h - z) / (2.0 * h)
  rep = repeat
  if tex_uniform:
    rep = wp.vec2(repeat[0] * size[0], repeat[1] * size[0])
  return wp.vec2(u, v), wp.vec2(0.0, 0.0), rep


@wp.func
def texture_box(
  # In:
  local: wp.vec3,
  local_normal: wp.vec3,
  size: wp.vec3,
  repeat: wp.vec2,
  tex_uniform: bool,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """(uv, offset, repeat) of a box hit: the hit face's planar mapping."""
  sx = wp.max(size[0], 1e-6)
  sy = wp.max(size[1], 1e-6)
  sz = wp.max(size[2], 1e-6)
  ax = wp.abs(local_normal[0])
  ay = wp.abs(local_normal[1])
  az = wp.abs(local_normal[2])

  if az >= ax and az >= ay:
    if local_normal[2] > 0.0:
      uv = wp.vec2((local[0] + sx) / (2.0 * sx), (sy - local[1]) / (2.0 * sy))
    else:
      uv = wp.vec2((local[0] + sx) / (2.0 * sx), (local[1] + sy) / (2.0 * sy))
  elif ax >= ay and ax >= az:
    if local_normal[0] > 0.0:
      uv = wp.vec2((local[1] + sy) / (2.0 * sy), (sz - local[2]) / (2.0 * sz))
    else:
      uv = wp.vec2((sy - local[1]) / (2.0 * sy), (sz - local[2]) / (2.0 * sz))
  else:
    if local_normal[1] < 0.0:
      uv = wp.vec2((local[0] + sx) / (2.0 * sx), (sz - local[2]) / (2.0 * sz))
    else:
      uv = wp.vec2((sx - local[0]) / (2.0 * sx), (sz - local[2]) / (2.0 * sz))

  rep = repeat
  if tex_uniform:
    rep = wp.vec2(repeat[0] * size[0], repeat[1] * size[1])
  return uv, wp.vec2(0.0, 0.0), rep


@wp.func
def texture_hfield(
  # In:
  local: wp.vec3,
  size: wp.vec3,
  repeat: wp.vec2,
  tex_uniform: bool,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """(uv, offset, repeat) of a height-field hit: planar over its extent."""
  # Heightfield horizontal extent is [-size[0], size[0]] x [-size[1], size[1]]
  # render_context.c lines 465-500
  s0 = wp.max(size[0], 1e-6)
  s1 = wp.max(size[1], 1e-6)
  uv = wp.vec2(
    (local[0] + size[0]) / (2.0 * s0),
    (size[1] - local[1]) / (2.0 * s1),
  )
  rep = repeat
  if tex_uniform:
    rep = wp.vec2(repeat[0] * size[0], repeat[1] * size[1])
  return uv, wp.vec2(0.0, 0.0), rep


@wp.func
def vertex_normal(
  # In:
  n: wp.vec3,
  face: wp.vec3,
) -> wp.vec3:
  """Vertex normal `n`, or the face normal when it is more than ~37 degrees off it."""
  if wp.dot(n, face) < 0.8:
    return face
  return n
