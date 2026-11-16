"""Ray queries of triangle meshes (and height fields) and deformables through their Warp BVHs.

Meshes are queried in the instance frame (`wp.Mesh` per mesh, `mesh_bvh_id[mesh]`; a frame s R
with a uniform scale s, `mesh_ray`); deformable
(flex) meshes are already in world space, one `wp.Mesh` per flex whose BVH groups the worlds;
deformable edges (1D flex) are capsules along the edge (`edge_frame`).

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0, `_src/ray.py`,
`_src/math.py`); the arithmetic is unchanged. Changed: names and imports, docstrings and argument
groups, back faces culled inside the mesh query (skipped like a rasterizer, not the whole mesh),
the edge capsule frame as one func (`edge_frame`), scaled instance frames (`mesh_ray`), our mesh
BLAS queries instead of Warp's `mesh_query_ray(_anyhit)` (`native.mesh_closest` / `mesh_any`, C++
in `nodes.py`: Warp's traversal and watertight test with the per-ray shear set up once).
"""

from __future__ import annotations

import warp as wp

from raywarp._base.types import MINVAL as MJ_MINVAL
from raywarp.core.kernels import native
from raywarp.core.kernels.abi.rays import frame_inverse

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def quat_to_mat(
  # In:
  quat: wp.quat,
) -> wp.mat33:
  """Rotation matrix of a MuJoCo quaternion (w, x, y, z)."""
  q00 = quat[0] * quat[0]
  q01 = quat[0] * quat[1]
  q02 = quat[0] * quat[2]
  q03 = quat[0] * quat[3]
  q11 = quat[1] * quat[1]
  q12 = quat[1] * quat[2]
  q13 = quat[1] * quat[3]
  q22 = quat[2] * quat[2]
  q23 = quat[2] * quat[3]
  q33 = quat[3] * quat[3]

  return wp.mat33(
    q00 + q11 - q22 - q33,
    2.0 * (q12 - q03),
    2.0 * (q13 + q02),
    2.0 * (q12 + q03),
    q00 - q11 + q22 - q33,
    2.0 * (q23 - q01),
    2.0 * (q13 - q02),
    2.0 * (q23 + q01),
    q00 - q11 - q22 + q33,
  )


@wp.func
def axis_angle_to_quat(
  # In:
  axis: wp.vec3,
  angle: float,
) -> wp.quat:
  """MuJoCo quaternion (w, x, y, z) of the rotation by `angle` about the unit `axis`."""
  s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)
  axis = axis * s
  return wp.quat(c, axis[0], axis[1], axis[2])


@wp.func
def quat_z2vec(
  # In:
  vec: wp.vec3,
) -> wp.quat:
  """MuJoCo quaternion of the shortest rotation taking the z axis to `vec`."""
  quat = wp.quat(0.0, 0.0, 0.0, 1.0)

  # normalize vector; if too small, no rotation
  norm = wp.length(vec)
  if norm < MJ_MINVAL:
    return quat
  vec = vec / norm

  axis = wp.vec3(-vec[1], vec[0], 0.0)
  a = wp.length(axis)

  # almost parallel
  if a < MJ_MINVAL:
    # opposite: 180 deg rotation around x axis
    if vec[2] < 0.0:
      quat = wp.quat(1.0, 0.0, 0.0, 0.0)
    return quat

  # make quaternion from angle and axis
  axis = axis / a
  angle = wp.atan2(a, vec[2])
  quat = axis_angle_to_quat(axis, angle)

  return quat


@wp.func
def edge_frame(
  # In:
  v0: wp.vec3,
  v1: wp.vec3,
  radius: float,
) -> tuple[wp.vec3, wp.mat33, wp.vec3]:
  """(centre, rotation, capsule size) of the capsule of `radius` along the edge v0 -> v1."""
  vec = v1 - v0
  size = wp.vec3(radius, 0.5 * wp.length(vec), 0.0)
  return 0.5 * (v0 + v1), quat_to_mat(quat_z2vec(vec)), size


@wp.func
def mesh_ray(
  # In:
  pos: wp.vec3,
  frame: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(origin, direction) of the world ray (pnt, vec) in the mesh frame of an instance at (pos,
  frame s R): through `trace.frame_inverse`, so the direction has length 1 / s and a distance along
  it is the world distance."""
  inv = frame_inverse(frame)
  return inv @ (pnt - pos), inv @ vec


@wp.func
def ray_mesh_with_bvh(
  # In:
  mesh_bvh_id: wp.array[wp.uint64],
  mesh_nodes: wp.array[wp.uint64],
  mesh_instance_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
  cull_backfaces: bool,
  counts: wp.uint64,
) -> tuple[float, wp.vec3, float, float, int, int]:
  """(distance, world normal, u, v, face, mesh id) of the ray's closest hit on mesh
  `mesh_instance_id` at (pos, mat) (`mat`: the instance frame s R, `mesh_ray`).

  Distance -1: no hit before `max_t`. With `cull_backfaces`, back-facing triangles are skipped
  (the rule of MuJoCo's OpenGL rendering): the nearest front-facing triangle is the hit.
  `counts`: the traversal counters' address (`nodes.counter_address`; 0: not counted).
  """
  lpnt, lvec = mesh_ray(pos, mat, pnt, vec)
  # Culling drops back-facing triangles, not the mesh: like a rasterizer, the query skips them
  # inside its leaves and returns the nearest front face.
  q = native.mesh_closest(
    mesh_bvh_id[mesh_instance_id], mesh_nodes[mesh_instance_id], lpnt, lvec, max_t, -1, cull_backfaces, counts
  )
  if q.t < 0.0:
    return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
  normal = wp.normalize(mat @ q.normal)
  return q.t, normal, q.u, q.v, q.face, mesh_instance_id


@wp.func
def ray_mesh_with_bvh_anyhit(
  # In:
  mesh_bvh_id: wp.array[wp.uint64],
  mesh_nodes: wp.array[wp.uint64],
  mesh_instance_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
  counts: wp.uint64,
) -> bool:
  """Whether the ray hits mesh `mesh_instance_id` at (pos, mat) before `max_t` (shadow rays);
  `counts` as `ray_mesh_with_bvh`."""
  lpnt, lvec = mesh_ray(pos, mat, pnt, vec)
  return native.mesh_any(mesh_bvh_id[mesh_instance_id], mesh_nodes[mesh_instance_id], lpnt, lvec, max_t, -1, counts)


@wp.func
def ray_flex_with_bvh(
  # In:
  flex_bvh_id: wp.array[wp.uint64],
  flex_nodes: wp.array[wp.uint64],
  flexid: int,
  group_root: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
  cull_backfaces: bool,
  counts: wp.uint64,
) -> tuple[float, wp.vec3, float, float, int]:
  """(distance, normal, u, v, face) of the world ray's closest hit on flex `flexid` (one world's
  group); `cull_backfaces` and `counts` as `ray_mesh_with_bvh` (the nearest front face is the
  hit, not a back face hiding the rest of the flex)."""
  q = native.mesh_closest(flex_bvh_id[flexid], flex_nodes[flexid], pnt, vec, max_t, group_root, cull_backfaces, counts)
  if q.t >= 0.0:
    return q.t, q.normal, q.u, q.v, q.face

  return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1


@wp.func
def ray_flex_with_bvh_anyhit(
  # In:
  flex_bvh_id: wp.array[wp.uint64],
  flex_nodes: wp.array[wp.uint64],
  flexid: int,
  group_root: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
  counts: wp.uint64,
) -> bool:
  """Whether the world ray hits flex `flexid` (one world's group) before `max_t`; `counts` as
  `ray_mesh_with_bvh`."""
  return native.mesh_any(flex_bvh_id[flexid], flex_nodes[flexid], pnt, vec, max_t, group_root, counts)
