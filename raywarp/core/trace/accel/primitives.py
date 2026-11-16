"""Ray intersections of the analytic primitives (plane, sphere, capsule, ellipsoid, cylinder, box).

Each returns (distance along the ray, world normal); distance -1 means no hit, and a hit is at a
distance > 0: a ray starting on a surface does not hit it at 0 (an eye on a plane would see it at
depth -0.0, which the depth output reports as a miss). The ray direction need not be unit length
(distances are in its units).

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0, `_src/ray.py`); the
arithmetic is unchanged (hits are bit-identical to MuJoCo Warp's). Changed: names and imports,
docstrings and argument groups, `safe_div` typed for floats (was generic over `Any`), `ray_box`
no longer returns the per-face distances nobody read, the mesh / flex queries moved to
`meshes.py`.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import MINVAL as MJ_MINVAL

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Relative / absolute tolerance of distance comparisons (tie-break window, widened cut-offs).
RAY_TOL_REL: Final = 1.0e-6
RAY_TOL_ABS: Final = 1.0e-9


@wp.func
def safe_div(
  # In:
  x: float,
  y: float,
) -> float:
  """Quotient x / y, with y = 0 replaced by MuJoCo's MINVAL."""
  return x / wp.where(y != 0.0, y, MJ_MINVAL)


@wp.func
def ray_map(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(origin, direction) of the world ray (pnt, vec) in the frame (pos, mat)."""
  matT = wp.transpose(mat)
  lpnt = matT @ (pnt - pos)
  lvec = matT @ vec

  return lpnt, lvec


@wp.func
def _ray_quad(
  # In:
  a: float,
  b: float,
  c: float,
) -> tuple[float, wp.vec2]:
  """(smallest positive root or -1, both roots) of a * x^2 + 2 * b * x + c = 0."""
  det = b * b - a * c
  if det < MJ_MINVAL:
    return -1.0, wp.vec2(-1.0, -1.0)
  det = wp.sqrt(det)

  # compute the two solutions
  den = safe_div(1.0, a)
  x0 = (-b - det) * den
  x1 = (-b + det) * den
  x = wp.vec2(x0, x1)

  # finalize result
  if x0 > 0.0:
    return x0, x
  elif x1 > 0.0:
    return x1, x
  else:
    return -1.0, x


@wp.func
def ray_plane(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]:
  """Distance and normal (the front face's, +z) of the ray's hit on the plane (size <= 0:
  infinite). Both faces are hit (a probe under the ground must see its back); queries that
  cull back faces drop the back-face hits (`instances.culled`)."""
  # map to local frame
  lpnt, lvec = ray_map(pos, mat, pnt, vec)

  # z-vec parallel to the plane: reject
  if wp.abs(lvec[2]) <= MJ_MINVAL:
    return -1.0, wp.vec3()

  # intersection with plane
  x = -lpnt[2] / lvec[2]
  if x <= 0.0:  # also -0.0 (the ray starts on the plane)
    return -1.0, wp.vec3()

  p = wp.vec2(lpnt[0] + x * lvec[0], lpnt[1] + x * lvec[1])

  # accept only within rendered rectangle
  if (size[0] <= 0.0 or wp.abs(p[0]) <= size[0]) and (size[1] <= 0.0 or wp.abs(p[1]) <= size[1]):
    return x, wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
  else:
    return -1.0, wp.vec3()


@wp.func
def ray_sphere(
  # In:
  pos: wp.vec3,
  dist_sqr: float,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]:
  """Distance and normal of the ray's hit on the sphere at `pos` of squared radius `dist_sqr`."""
  dif = pnt - pos

  a = wp.dot(vec, vec)
  b = wp.dot(vec, dif)
  c = wp.dot(dif, dif) - dist_sqr

  sol, _ = _ray_quad(a, b, c)
  normal = wp.vec3()
  if sol >= 0:
    s = pnt + vec * sol
    normal = wp.normalize(s - pos)
  return sol, normal


@wp.func
def ray_capsule(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]:
  """Distance and normal of the ray's hit on the capsule (radius size[0], half length size[1] on
  z)."""
  # bounding sphere test
  ssz = size[0] + size[1]
  dist_sphere, _sphere_normal = ray_sphere(pos, ssz * ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, wp.vec3()

  # map to local frame
  lpnt, lvec = ray_map(pos, mat, pnt, vec)

  # init solution
  x = -1.0

  # cylinder round side: (x * lvec + lpnt)' * (x * lvec + lpnt) = size[0] * size[0]
  sq_size0 = size[0] * size[0]
  a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
  b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
  c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - sq_size0

  # solve a * x^2 + 2 * b * x + c = 0
  sol, xx = _ray_quad(a, b, c)
  part = 0  # -1: bottom, 0: cylinder, 1: top

  # make sure round solution is between flat sides (must use local z component)
  if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
    if x < 0.0 or sol < x:
      x = sol

  # top cap
  ldif = wp.vec3(lpnt[0], lpnt[1], lpnt[2] - size[1])
  a += lvec[2] * lvec[2]
  b = wp.dot(lvec, ldif)
  c = wp.dot(ldif, ldif) - sq_size0
  _, xx = _ray_quad(a, b, c)

  # accept only top half of sphere (use local z component)
  for i in range(2):
    if xx[i] > 0.0 and lpnt[2] + xx[i] * lvec[2] >= size[1]:
      if x < 0.0 or xx[i] < x:
        x = xx[i]
        part = 1

  # bottom cap
  ldif = wp.vec3(ldif[0], ldif[1], lpnt[2] + size[1])
  b = wp.dot(lvec, ldif)
  c = wp.dot(ldif, ldif) - sq_size0
  _, xx = _ray_quad(a, b, c)

  # accept only bottom half of sphere (use local z component)
  for i in range(2):
    if xx[i] > 0.0 and lpnt[2] + xx[i] * lvec[2] <= -size[1]:
      if x < 0.0 or xx[i] < x:
        x = xx[i]
        part = -1

  normal = wp.vec3()
  if x >= 0:
    normal[0] = lpnt[0] + lvec[0] * x
    normal[1] = lpnt[1] + lvec[1] * x
    if part == 0:
      normal[2] = 0.0
    else:
      normal[2] = lpnt[2] + lvec[2] * x - size[1] * float(part)

    # normalize, rotate into global frame
    normal = wp.normalize(normal)
    normal = mat @ normal

  return x, normal


@wp.func
def ray_ellipsoid(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]:
  """Distance and normal of the ray's hit on the ellipsoid of semi-axes `size`."""
  # map to local frame
  lpnt, lvec = ray_map(pos, mat, pnt, vec)

  # invert size^2
  s = wp.vec3(safe_div(1.0, size[0] * size[0]), safe_div(1.0, size[1] * size[1]), safe_div(1.0, size[2] * size[2]))

  # (x * lvec + lpnt)' * diag(1 / size^2) * (x * lvec + lpnt) = 1
  slvec = wp.cw_mul(s, lvec)
  a = wp.dot(slvec, lvec)
  b = wp.dot(slvec, lpnt)
  c = wp.dot(wp.cw_mul(s, lpnt), lpnt) - 1.0

  # solve a * x^2 + 2 * b * x + c = 0
  sol, _ = _ray_quad(a, b, c)

  normal = wp.vec3()
  if sol >= 0:
    # surface intersection (local frame)
    l = lpnt + lvec * sol

    # gradient of ellipsoid function
    normal = wp.cw_mul(s, l)
    normal = wp.normalize(normal)
    normal = mat @ normal

  return sol, normal


@wp.func
def ray_cylinder(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]:
  """Distance and normal of the ray's hit on the cylinder (radius size[0], half height size[1] on
  z)."""
  # bounding sphere test
  ssz = size[0] * size[0] + size[1] * size[1]
  dist_sphere, _sphere_normal = ray_sphere(pos, ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, wp.vec3()

  # map to local frame
  lpnt, lvec = ray_map(pos, mat, pnt, vec)

  # init solution
  x = -1.0
  part = 0  # -1: bottom, 0: cylinder, 1: top

  # flat sides
  if wp.abs(lvec[2]) > MJ_MINVAL:
    for side in range(-1, 2, 2):
      # solution of: lpnt[2] + x * lvec[2] = side * height_size
      sol = (float(side) * size[1] - lpnt[2]) / lvec[2]

      # process if positive (a hit at distance 0 is no hit)
      if sol > 0.0:
        # intersection with horizontal face
        p = wp.vec2(lpnt[0] + sol * lvec[0], lpnt[1] + sol * lvec[1])

        # accept within radius
        if wp.dot(p, p) <= size[0] * size[0]:
          if x < 0.0 or sol < x:
            x = sol
            part = side

  # (x * lvec + lpnt)' * (x * lvec + lpnt) = size[0] * size[0]
  a = lvec[0] * lvec[0] + lvec[1] * lvec[1]
  b = lvec[0] * lpnt[0] + lvec[1] * lpnt[1]
  c = lpnt[0] * lpnt[0] + lpnt[1] * lpnt[1] - size[0] * size[0]

  # solve a * x^2 + 2 * b * x + c = 0
  sol, _ = _ray_quad(a, b, c)

  # make sure round solution is between flat sides
  if sol >= 0.0 and wp.abs(lpnt[2] + sol * lvec[2]) <= size[1]:
    if x < 0.0 or sol < x:
      x = sol
      part = 0

  normal = wp.vec3()
  if x >= 0:
    if part == 0:
      normal = lpnt + lvec * x
      normal[2] = 0.0
      normal = wp.normalize(normal)
    else:
      normal = wp.vec3(0.0, 0.0, float(part))

    normal = mat @ normal

  return x, normal


@wp.func
def ray_box(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]:
  """Distance and normal of the ray's hit on the box of half extents `size`."""
  # bounding sphere test
  ssz = wp.dot(size, size)
  dist_sphere, _sphere_normal = ray_sphere(pos, ssz, pnt, vec)
  if dist_sphere < 0:
    return -1.0, wp.vec3()

  # map to local frame
  lpnt, lvec = ray_map(pos, mat, pnt, vec)

  # init solution
  x = float(-1.0)
  face_side = -1
  face_axis = -1

  # loop over axes with non-zero vec
  for i in range(3):
    if wp.abs(lvec[i]) > MJ_MINVAL:
      for side in range(-1, 2, 2):
        # solution of: lpnt[i] + x * lvec[i] = side * size[i]
        sol = (float(side) * size[i] - lpnt[i]) / lvec[i]

        # process if positive (a hit at distance 0 is no hit)
        if sol > 0.0:
          id0 = 1 if i == 0 else 0
          id1 = 1 if i == 2 else 2

          # intersection with face
          p0 = lpnt[id0] + sol * lvec[id0]
          p1 = lpnt[id1] + sol * lvec[id1]

          # accept within rectangle
          if (wp.abs(p0) <= size[id0]) and (wp.abs(p1) <= size[id1]):
            if x < 0.0 or sol < x:
              x = sol
              face_axis = i
              face_side = side

  normal = wp.vec3()
  if x >= 0:
    # The face's axis by selects (a dynamic component write puts the vector in local memory).
    sign = float(face_side)
    normal = wp.vec3(
      wp.where(face_axis == 0, sign, 0.0), wp.where(face_axis == 1, sign, 0.0), wp.where(face_axis == 2, sign, 0.0)
    )
    normal = mat @ normal

  return x, normal
