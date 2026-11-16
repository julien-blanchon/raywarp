"""World-space bounding boxes of the scene BVHs' primitives (instances and deformable elements).

`world_aabb` is the one closed form of an instance's world box (the OptiX build pads it:
`trace/optix/instances.py`). The kernels write the world tree's boxes (`instance_bounds`,
`flex_bounds`; world w's group is its instances, then its deformable primitives) and read the root
of each group (`group_roots`).

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0, `_src/bvh.py`). Changed:
closed-form boxes (box, plane, mesh and height field as |R| @ half extents instead of the min / max
of 8 or 4 transformed corners: equal up to float rounding), one `world_aabb` instead of one func
per type, the pose column of each primitive (`pose_cols`: compact per-world columns or shared
pose slots, `core/pipeline/worlds/sharing.py`), poses / vertices read at row `worldid % rows` (a
single row of initial poses builds every world's tree), names, imports, docstrings.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import MAXVAL as MJ_MAXVAL
from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.tables import batched, in_table, instance_rgba, instance_shown
from raywarp.core.trace.accel import reach

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Half size of the square standing in for an infinite plane (size <= 0), and the slab
# thickness added around every plane box.
INFINITE_PLANE: Final = 1000.0
PLANE_PAD: Final = 0.01
# Relative widening of a mesh instance's tight world box: 8 float32 ulps (2^-23) of the magnitude
# of its terms (|pos| + |rot| |centre| + |rot| half), above the rounding of pos + rot @ centre and
# |rot| @ half (a few operations, each within half an ulp) and of the float32 centre / half.
MESH_BOX_ULPS: Final = 8.0 / 8388608.0


@wp.func
def abs_vec(
  # In:
  v: wp.vec3,
) -> wp.vec3:
  """Componentwise |v|."""
  return wp.vec3(wp.abs(v[0]), wp.abs(v[1]), wp.abs(v[2]))


@wp.func
def max_abs(
  # In:
  v: wp.vec3,
) -> float:
  """The largest |component| of v."""
  return wp.max(wp.max(wp.abs(v[0]), wp.abs(v[1])), wp.abs(v[2]))


@wp.func
def abs_rot(
  # In:
  rot: wp.mat33,
  half: wp.vec3,
) -> wp.vec3:
  """Half extents of the world box of the box of half extents `half` rotated by `rot` (|R| @
  half)."""
  return wp.vec3(
    wp.abs(rot[0, 0]) * half[0] + wp.abs(rot[0, 1]) * half[1] + wp.abs(rot[0, 2]) * half[2],
    wp.abs(rot[1, 0]) * half[0] + wp.abs(rot[1, 1]) * half[1] + wp.abs(rot[1, 2]) * half[2],
    wp.abs(rot[2, 0]) * half[0] + wp.abs(rot[2, 1]) * half[1] + wp.abs(rot[2, 2]) * half[2],
  )


@wp.func
def world_aabb(
  # In:
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(lower, upper) world box of a shape of type `gtype` at (pos, rot).

  `size`: the shape's size; for meshes (and SDFs) their box's half extents (`pos`: its centre),
  for height fields the height field's (x, y half sizes, half height; the box is centred half its
  height above `pos`).
  """
  centre = pos
  half = wp.vec3(0.0)
  if gtype == ShapeType.SPHERE:
    half = wp.vec3(size[0], size[0], size[0])
  elif gtype == ShapeType.CAPSULE:
    # The axis segment (local z, half length size[1]) inflated by the radius.
    r = size[0]
    half = wp.vec3(wp.abs(rot[0, 2]) * size[1] + r, wp.abs(rot[1, 2]) * size[1] + r, wp.abs(rot[2, 2]) * size[1] + r)
  elif gtype == ShapeType.PLANE:
    # A finite square of twice the larger half size (infinite planes: INFINITE_PLANE) in local xy.
    s = wp.max(size[0], size[1]) * 2.0
    if size[0] <= 0.0 or size[1] <= 0.0:
      s = INFINITE_PLANE
    half = abs_rot(rot, wp.vec3(s, s, 0.0)) + wp.vec3(PLANE_PAD)
  elif gtype == ShapeType.ELLIPSOID:
    # Tight: the norm of each row of rot @ diag(size).
    half = wp.vec3(
      wp.length(wp.vec3(rot[0, 0] * size[0], rot[0, 1] * size[1], rot[0, 2] * size[2])),
      wp.length(wp.vec3(rot[1, 0] * size[0], rot[1, 1] * size[1], rot[1, 2] * size[2])),
      wp.length(wp.vec3(rot[2, 0] * size[0], rot[2, 1] * size[1], rot[2, 2] * size[2])),
    )
  elif gtype == ShapeType.CYLINDER:
    # Tight: the cap discs (radius size[0] across local x, y) swept along the axis.
    r = size[0]
    h = size[1]
    half = wp.vec3(
      r * wp.sqrt(rot[0, 0] * rot[0, 0] + rot[0, 1] * rot[0, 1]) + h * wp.abs(rot[0, 2]),
      r * wp.sqrt(rot[1, 0] * rot[1, 0] + rot[1, 1] * rot[1, 1]) + h * wp.abs(rot[1, 2]),
      r * wp.sqrt(rot[2, 0] * rot[2, 0] + rot[2, 1] * rot[2, 1]) + h * wp.abs(rot[2, 2]),
    )
  elif gtype == ShapeType.HFIELD:
    centre = pos + wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2]) * size[2]
    half = abs_rot(rot, size)
  else:
    # Box, mesh, SDF.
    half = abs_rot(rot, size)
  return centre - half, centre + half


@wp.kernel
def instance_bounds(
  # Model:
  instance_type: wp.array[int],
  instance_mesh: wp.array2d[int],
  instance_size: wp.array2d[wp.vec3],
  # Data in:
  instance_pos_in: wp.array2d[wp.vec3],
  instance_rot_in: wp.array2d[wp.mat33],
  # In:
  group_size: int,
  enabled_instance_ids: wp.array[int],
  mesh_centre: wp.array[wp.vec3],
  mesh_half: wp.array[wp.vec3],
  hfield_half: wp.array[wp.vec3],
  pose_cols: wp.array[int],
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
  group_out: wp.array[int],
):
  """World box of rendered instance k (`enabled_instance_ids[k]`) in world w, primitive w *
  group_size + k."""
  worldid, k = wp.tid()
  gi = enabled_instance_ids[k]
  col = pose_cols[k]
  # 1 or nworld rows of poses (the BVH build reads the initial poses once).
  pos = instance_pos_in[worldid % instance_pos_in.shape[0], col]
  rot = instance_rot_in[worldid % instance_rot_in.shape[0], col]
  size = instance_size[worldid % instance_size.shape[0], gi]
  gtype = instance_type[gi]
  out = worldid * group_size + k
  group_out[out] = worldid

  # Meshes are bounded by their own box (`Meshes.local_boxes`), height fields by their half
  # extents; without one, by a point.
  if gtype == ShapeType.MESH or gtype == ShapeType.SDF or gtype == ShapeType.HFIELD:
    mesh = instance_mesh[worldid % instance_mesh.shape[0], gi]
    if mesh < 0:
      lower_out[out] = pos
      upper_out[out] = pos
      return
    if gtype == ShapeType.HFIELD:
      size = hfield_half[in_table(mesh, hfield_half.shape[0])]
    else:
      # The mesh's box.
      m = in_table(mesh, mesh_half.shape[0])
      size = mesh_half[m]
      offset = rot @ mesh_centre[m]
      # The box is tight: widen it by the float32 rounding of pos + rot @ centre and of
      # |rot| @ half, a few ulps of the magnitudes of their terms (not of the result: a mesh
      # authored far from its origin placed back near the world origin cancels them).
      magnitude = max_abs(pos) + max_abs(abs_rot(rot, abs_vec(mesh_centre[m]))) + max_abs(abs_rot(rot, size))
      pad = wp.vec3(MESH_BOX_ULPS * magnitude)
      lower, upper = world_aabb(gtype, pos + offset, rot, size)
      lower_out[out] = lower - pad
      upper_out[out] = upper + pad
      return
  lower, upper = world_aabb(gtype, pos, rot, size)
  lower_out[out] = lower
  upper_out[out] = upper


@wp.kernel
def world_references(
  # Model:
  instance_size: wp.array2d[wp.vec3],
  # Data in:
  instance_pos_in: wp.array2d[wp.vec3],
  instance_rot_in: wp.array2d[wp.mat33],
  # In:
  enabled_instance_ids: wp.array[int],
  pose_cols: wp.array[int],
  # Out:
  reference_out: wp.array[wp.vec3],
):
  """World w's reference position (`reach.reference`) over its world-tree instances."""
  worldid = wp.tid()
  reference_out[worldid] = reach.reference(
    instance_pos_in, instance_rot_in, instance_size, enabled_instance_ids, pose_cols, worldid
  )


@wp.kernel
def hide_bounds(
  # Model:
  color_rule: int,
  instance_material: wp.array2d[int],
  instance_color: wp.array2d[wp.vec4],
  material_base_color: wp.array2d[wp.vec4],
  visible_slot: wp.array[int],
  instance_size: wp.array2d[wp.vec3],
  # Data in:
  instance_visible_in: wp.array2d[int],
  instance_pos_in: wp.array2d[wp.vec3],
  instance_rot_in: wp.array2d[wp.mat33],
  # In:
  use_visible: int,
  group_size: int,
  enabled_instance_ids: wp.array[int],
  pose_cols: wp.array[int],
  reference: wp.array[wp.vec3],
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
):
  """Shrinks the world box of a hidden instance to a point, so the refit tree has no box around
  it and rays skip it unread.

  An instance hidden by its colour (alpha 0) or its world's visibility shrinks to its centre: a
  ray through that point exactly is still rejected by the traversal's own test
  (`instances.make_resolve`). One the world does not draw (`reach.drawn`: diverged, or beyond
  `reach.NEAR_RADIUS` of the world's reference) shrinks to the reference position, beyond
  `reach.NEAR_RADIUS` of its position: only a ray through that point exactly (measure zero) still
  tests it, at its own pose (the OptiX traversal leaves it out of the world cell: one image on
  both backends). Either point keeps the primitive's centroid finite and inside the world's box
  for an LBVH rebuild.
  """
  worldid, k = wp.tid()
  gi = enabled_instance_ids[k]
  out = worldid * group_size + k
  col = pose_cols[k]
  ref = reference[worldid]
  if not reach.drawn(ref, instance_pos_in[worldid, col], instance_rot_in[worldid, col], batched(instance_size, worldid, gi)):
    lower_out[out] = ref
    upper_out[out] = ref
    return
  rgba, _ = instance_rgba(color_rule, instance_material, instance_color, material_base_color, worldid, gi)
  if not instance_shown(visible_slot, instance_visible_in, rgba, use_visible != 0, worldid, gi):
    centre = 0.5 * (lower_out[out] + upper_out[out])
    lower_out[out] = centre
    upper_out[out] = centre


@wp.kernel
def group_roots(
  # In:
  bvh_id: wp.uint64,
  # Out:
  group_root_out: wp.array[int],
):
  """Root node of each group (world) of the BVH."""
  tid = wp.tid()
  group_root_out[tid] = wp.bvh_get_group_root(bvh_id, tid)


@wp.kernel
def flex_bounds(
  # Model:
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flex_radius: wp.array[float],
  # Data in:
  flex_vert_in: wp.array2d[wp.vec3],
  # In:
  flex_prim_flexid: wp.array[int],
  flex_prim_edgeid: wp.array[int],
  ninstance: int,
  group_size: int,
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
  group_out: wp.array[int],
):
  """World box of deformable primitive k in world w (after the world's `ninstance` instances)."""
  worldid, k = wp.tid()
  flex_id = flex_prim_flexid[k]
  edge_id = flex_prim_edgeid[k]
  out = worldid * group_size + ninstance + k
  row = worldid % flex_vert_in.shape[0]
  adr = flex_vertadr[flex_id]
  inflate = wp.vec3(flex_radius[flex_id])

  if edge_id >= 0:
    # 1D flex: the edge's capsule.
    edge = flex_edge[edge_id]
    v0 = flex_vert_in[row, adr + edge[0]]
    v1 = flex_vert_in[row, adr + edge[1]]
    lower_out[out] = wp.min(v0, v1) - inflate
    upper_out[out] = wp.max(v0, v1) + inflate
  else:
    # 2D / 3D flex: the box of all its vertices.
    lo = wp.vec3(MJ_MAXVAL)
    hi = wp.vec3(-MJ_MAXVAL)
    for i in range(flex_vertnum[flex_id]):
      v = flex_vert_in[row, adr + i]
      lo = wp.min(lo, v)
      hi = wp.max(hi, v)
    lower_out[out] = lo - inflate
    upper_out[out] = hi + inflate

  group_out[out] = worldid
