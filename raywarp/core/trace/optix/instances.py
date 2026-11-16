"""Warp kernels writing the OptiX build inputs of a frame: cell bounds, instances, curves, boxes.

The OptiX structures (`accel.py`) hold two groups: per-world instances in their normalized cell
(`trace.to_cell`), and shared scenery, untranslated, posed by the shared pose table and world 0's
sizes, alpha and meshes. Each writer is one kernel factory over a compile-time `shared` flag
(launch dim (nworld, n) per world, (1, n) shared), so both groups apply one rule:

* `write_instances`: triangle instances (meshes and boxes) of the instance acceleration structure
  (IAS); hidden instances (alpha 0, per-world visibility table, no mesh, not drawn:
  `trace.accel.reach`) get visibility mask 0 (the shared group's records are written once for every
  world: its per-world visibility is the any-hit / intersection programs' test, `programs.py`).
* `write_curves`: the linear-curve segment of a sphere / capsule (radius 0 when hidden).
* `write_primitive_bounds`: the world box of an ellipsoid / cylinder (`bounds.world_aabb`, padded;
  a point when hidden) and its code (the instance, -1 - instance when hidden), read by the
  intersection programs.

Instance ids (OptiX keeps 28 bits): a triangle instance's id is its instance index in both groups
(the ray's world is the payload's, `hits.instance_of`).
Per-world kernels compute each world's bounds from the frame's poses alone (`world_bounds`: its
instances drawn, `trace.accel.reach`) and derive its cell's uniform scale (`update_cell`): a world's
cell transform and instance records depend on its current geometry only, never on earlier frames (a
refitted tree keeps its build's topology: that changes traversal order only, not the transforms).
Per-world instances that `trace.accel.reach` hides (a diverged simulation, a body parked or
teleported beyond `reach.NEAR_RADIUS`) are left out and hidden.

The static GASes' own inputs (packed face normals, the unit cube) are in `mesh_inputs.py`.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.rays import world_entry
from raywarp.core.kernels.abi.tables import batched, instance_rgba, instance_shown
from raywarp.core.kernels.generic import specialize
from raywarp.core.trace.accel import reach
from raywarp.core.trace.accel.bounds import world_aabb
from raywarp.core.trace.optix.device import trace as rt_trace

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Empty bounds (`world_bounds`' starting value): lower xyz +BIG, upper xyz -BIG.
BIG: Final = 1.0e30
# Bounds entries per world (`world_bounds`): lower xyz, upper xyz of its members in reach, its
# reference position.
BOUNDS: Final = 9
# Custom-primitive AABB padding in acceleration coordinates, for rounding at grazing hits: relative
# to the box extent, plus an absolute floor (cell units per world, metres for shared geometry), plus
# 8 float spacings of the box's largest coordinate (2^-20 of it: world cells sit at |x| in [b, 2b),
# where a fixed floor below the spacing would add nothing).
_BOUND_PAD_REL: Final = 1.0e-4
_BOUND_PAD_ABS: Final = 1.0e-6
_BOUND_PAD_ULP: Final = 2.0**-20


@wp.struct
class OptixInstance:
  """Layout of OptixInstance (80 bytes): 3×4 row-major transform, ids, flags, child handle."""

  t0: wp.vec4
  t1: wp.vec4
  t2: wp.vec4
  instance_id: wp.uint32
  sbt_offset: wp.uint32
  visibility_mask: wp.uint32
  flags: wp.uint32
  handle: wp.uint64
  pad0: wp.uint32
  pad1: wp.uint32


# A custom primitive's box: lower xyz, upper xyz (OptixAabb).
Aabb = wp.types.vector(length=6, dtype=float)


@wp.func
def gas_frame(
  # Model:
  mesh_frame_t: wp.array1d[wp.mat33],
  # In:
  gtype: int,
  meshid: int,
  rot: wp.mat33,
) -> wp.mat33:
  """Linear part of an instance's GAS transform: `rot`, times R^T for a mesh (or SDF: traced as
  its mesh) GAS built in frame R."""
  if (gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF)) and meshid >= 0:
    return rot @ mesh_frame_t[meshid]
  return rot


@wp.func
def transform_rows(
  # In:
  rot: wp.mat33,
  pos: wp.vec3,
  inst: OptixInstance,
) -> OptixInstance:
  """`inst` with the 3×4 row-major transform [rot | pos]."""
  inst.t0 = wp.vec4(rot[0, 0], rot[0, 1], rot[0, 2], pos[0])
  inst.t1 = wp.vec4(rot[1, 0], rot[1, 1], rot[1, 2], pos[1])
  inst.t2 = wp.vec4(rot[2, 0], rot[2, 1], rot[2, 2], pos[2])
  return inst


@specialize
def _make_pose(*, shared: bool):
  """pose(...) of instance column / slot `col` in world `worldid`: position on its cell (per world)
  or in scene coordinates (shared, slot of the shared pose table), rotation, and whether a
  per-world instance is placed in the cell (`reach.drawn`; hidden when not)."""

  @wp.func
  def pose(
    # Data in:
    instance_pos_in: wp.array2d[wp.vec3],
    instance_rot_in: wp.array2d[wp.mat33],
    shared_pos_in: wp.array1d[wp.vec3],
    shared_rot_in: wp.array1d[wp.mat33],
    # In:
    cell: wp.array1d[wp.vec3],
    worldid: int,
    col: int,
    size: wp.vec3,
  ) -> tuple[wp.vec3, wp.mat33, bool]:
    """(position, rotation, placed) of the instance at column / slot `col` of size `size` (see
    `_make_pose`). A per-world instance outside the cell collapses to the cell's centre with a
    zero rotation, so that its build inputs stay finite and inside its world's cell."""
    if wp.static(shared):
      return shared_pos_in[col], shared_rot_in[col], True
    pos = wp.vec3(instance_pos_in[worldid, col])
    rot = wp.mat33(instance_rot_in[worldid, col])
    if reach.drawn(rt_trace.world_reference(cell, worldid), pos, rot, size):
      return rt_trace.to_cell(cell, worldid, pos), rot, True
    return rt_trace.cell_offset(cell, worldid), wp.mat33(0.0), False

  return pose


@specialize
def write_instances(*, shared: bool, world_optics: bool = False):
  """Kernel writing the triangle instance of (world, k): mesh or box instance `ids[k]` (see the
  module doc); `world_optics` (Features.world_optics): the instance flags are the world's
  (`group.InstanceTables.flags`).

  A mesh instance maps its GAS frame to the world (rot @ R^T) and references its mesh's GAS (or its
  own, `instance_handle`); a box instance scales the unit cube to the box's half sizes.
  """
  pose = _make_pose(shared=shared)

  @wp.kernel(module="unique", enable_backward=False)
  def triangle_instances(
    # Model:
    instance_type: wp.array1d[int],
    instance_mesh: wp.array2d[int],
    color_rule: int,
    instance_material: wp.array2d[int],
    instance_size: wp.array2d[wp.vec3],
    instance_color: wp.array2d[wp.vec4],
    material_base_color: wp.array2d[wp.vec4],
    mesh_handle: wp.array1d[wp.uint64],
    mesh_frame_t: wp.array1d[wp.mat33],
    cube_handle: wp.uint64,
    hfield_handle: wp.array1d[wp.uint64],
    instance_flags: wp.array1d[wp.uint32],
    instance_handle: wp.array1d[wp.uint64],
    instance_mask: wp.array1d[wp.uint32],
    # Data in:
    instance_pos_in: wp.array2d[wp.vec3],
    instance_rot_in: wp.array2d[wp.mat33],
    shared_pos_in: wp.array1d[wp.vec3],
    shared_rot_in: wp.array1d[wp.mat33],
    instance_visible_in: wp.array2d[int],
    visible_slot: wp.array1d[int],
    # In:
    use_visible: bool,
    ids: wp.array1d[int],
    cols: wp.array1d[int],
    cell: wp.array1d[wp.vec3],
    # Out:
    instances_out: wp.array1d[OptixInstance],
  ):
    """Triangle instance (world, k) of the group (see `write_instances`)."""
    worldid, k = wp.tid()
    gi = ids[k]
    col = cols[k]
    rgba, _ = instance_rgba(color_rule, instance_material, instance_color, material_base_color, worldid, gi)
    size = batched(instance_size, worldid, gi)
    pos, rot, ok = pose(instance_pos_in, instance_rot_in, shared_pos_in, shared_rot_in, cell, worldid, col, size)
    visible = ok and instance_shown(visible_slot, instance_visible_in, rgba, use_visible, worldid, gi)
    handle = cube_handle
    gtype = instance_type[gi]
    if gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF):
      meshid = batched(instance_mesh, worldid, gi)
      visible = visible and meshid >= 0
      handle = mesh_handle[wp.max(meshid, 0)]
      # A cut-out mesh with opacity micromaps has a GAS of its own (`omm/micromaps.py`; the same
      # mesh in every world).
      if instance_handle[gi] != wp.uint64(0):
        handle = instance_handle[gi]
      rot = gas_frame(mesh_frame_t, gtype, meshid, rot)
    elif gtype == int(ShapeType.HFIELD):
      # Its GAS holds the grid in the height field's frame (`dynamic.HeightFieldGases`).
      hid = batched(instance_mesh, worldid, gi)
      handle = hfield_handle[wp.clamp(hid, 0, hfield_handle.shape[0] - 1)]
      visible = visible and hid >= 0 and hid < hfield_handle.shape[0] and handle != wp.uint64(0)
    elif ok:
      rot = rot @ wp.diag(size)
    if wp.static(not shared):
      rot = rot * rt_trace.world_scale(cell, worldid)
    inst = transform_rows(rot, pos, OptixInstance())
    inst.instance_id = wp.uint32(gi)
    inst.sbt_offset = wp.uint32(0)
    inst.visibility_mask = wp.where(visible, instance_mask[gi], wp.uint32(0))
    if wp.static(world_optics):
      inst.flags = instance_flags[world_entry(instance_type.shape[0], worldid, gi)]
    else:
      inst.flags = instance_flags[gi]
    inst.handle = handle
    instances_out[worldid * ids.shape[0] + k] = inst

  return triangle_instances


@specialize
def write_curves(*, shared: bool):
  """Kernel writing the linear-curve segment (world, k) of sphere / capsule `ids[k]`: its axis, of
  its radius.

  Hidden instances get radius 0.
  """
  pose = _make_pose(shared=shared)

  @wp.kernel(module="unique", enable_backward=False)
  def curves(
    # Model:
    instance_type: wp.array1d[int],
    color_rule: int,
    instance_material: wp.array2d[int],
    instance_size: wp.array2d[wp.vec3],
    instance_color: wp.array2d[wp.vec4],
    material_base_color: wp.array2d[wp.vec4],
    # Data in:
    instance_pos_in: wp.array2d[wp.vec3],
    instance_rot_in: wp.array2d[wp.mat33],
    shared_pos_in: wp.array1d[wp.vec3],
    shared_rot_in: wp.array1d[wp.mat33],
    instance_visible_in: wp.array2d[int],
    visible_slot: wp.array1d[int],
    # In:
    use_visible: bool,
    ids: wp.array1d[int],
    cols: wp.array1d[int],
    cell: wp.array1d[wp.vec3],
    # Out:
    vertices_out: wp.array1d[wp.vec3],
    widths_out: wp.array1d[float],
  ):
    """Segment (world, k) of the group's curves (see `write_curves`)."""
    worldid, k = wp.tid()
    gi = ids[k]
    col = cols[k]
    rgba, _ = instance_rgba(color_rule, instance_material, instance_color, material_base_color, worldid, gi)
    size = batched(instance_size, worldid, gi)
    center, rot, ok = pose(instance_pos_in, instance_rot_in, shared_pos_in, shared_rot_in, cell, worldid, col, size)
    if wp.static(not shared):
      size = size * rt_trace.world_scale(cell, worldid)
    axis = wp.vec3(0.0)
    if instance_type[gi] == int(ShapeType.CAPSULE) and ok:
      axis = size[1] * wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    i = worldid * ids.shape[0] + k
    vertices_out[2 * i] = center - axis
    vertices_out[2 * i + 1] = center + axis
    visible = ok and instance_shown(visible_slot, instance_visible_in, rgba, use_visible, worldid, gi)
    radius = wp.where(visible, size[0], 0.0)
    widths_out[2 * i] = radius
    widths_out[2 * i + 1] = radius

  return curves


@wp.func
def padded_aabb(
  # In:
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Aabb:
  """The world box of a primitive (`bounds.world_aabb`), padded so that grazing hits survive
  rounding."""
  lower, upper = world_aabb(gtype, pos, rot, size)
  largest = wp.max(wp.max(wp.abs(lower), wp.abs(upper)))
  pad = (upper - lower) * wp.static(0.5 * _BOUND_PAD_REL) + wp.vec3(
    wp.static(_BOUND_PAD_ABS) + largest * wp.static(_BOUND_PAD_ULP)
  )
  lower = lower - pad
  upper = upper + pad
  return Aabb(lower[0], lower[1], lower[2], upper[0], upper[1], upper[2])


@specialize
def write_primitive_bounds(*, shared: bool):
  """Kernel writing the box of custom primitive (world, k), ellipsoid / cylinder `ids[k]`, and its
  code.

  The code is the instance, -1 - instance when hidden: hidden instances also shrink to a point.
  """
  pose = _make_pose(shared=shared)

  @wp.kernel(module="unique", enable_backward=False)
  def primitive_bounds(
    # Model:
    instance_type: wp.array1d[int],
    color_rule: int,
    instance_material: wp.array2d[int],
    instance_size: wp.array2d[wp.vec3],
    instance_color: wp.array2d[wp.vec4],
    material_base_color: wp.array2d[wp.vec4],
    # Data in:
    instance_pos_in: wp.array2d[wp.vec3],
    instance_rot_in: wp.array2d[wp.mat33],
    shared_pos_in: wp.array1d[wp.vec3],
    shared_rot_in: wp.array1d[wp.mat33],
    instance_visible_in: wp.array2d[int],
    visible_slot: wp.array1d[int],
    # In:
    use_visible: bool,
    ids: wp.array1d[int],
    cols: wp.array1d[int],
    cell: wp.array1d[wp.vec3],
    # Out:
    aabbs_out: wp.array1d[Aabb],
    code_out: wp.array2d[int],
  ):
    """Box and code of primitive (world, k) of the group (see `write_primitive_bounds`)."""
    worldid, k = wp.tid()
    gi = ids[k]
    col = cols[k]
    rgba, _ = instance_rgba(color_rule, instance_material, instance_color, material_base_color, worldid, gi)
    size = batched(instance_size, worldid, gi)
    center, rot, ok = pose(instance_pos_in, instance_rot_in, shared_pos_in, shared_rot_in, cell, worldid, col, size)
    if wp.static(not shared):
      size = size * rt_trace.world_scale(cell, worldid)
    visible = ok and instance_shown(visible_slot, instance_visible_in, rgba, use_visible, worldid, gi)
    i = worldid * ids.shape[0] + k
    aabbs_out[i] = Aabb(center[0], center[1], center[2], center[0], center[1], center[2])
    if visible:
      aabbs_out[i] = padded_aabb(instance_type[gi], center, rot, size)
    code_out[worldid, k] = wp.where(visible, gi, -1 - gi)

  return primitive_bounds


@wp.kernel
def write_gas_instance(
  # In:
  handle: wp.uint64,
  sbt_offset: wp.uint32,
  flags: wp.uint32,
  instance_id: wp.uint32,
  mask: wp.uint32,
  index: int,
  # Out:
  instances_out: wp.array1d[OptixInstance],
):
  """Identity instance `index` of a world-space GAS (curves, custom primitives, a deformable),
  with visibility mask `mask` (`trace.GAS_BITS`, a deformable's `trace.FLEX_BITS`)."""
  inst = transform_rows(wp.identity(3, dtype=float), wp.vec3(0.0), OptixInstance())
  inst.instance_id = instance_id
  inst.sbt_offset = sbt_offset
  inst.visibility_mask = mask
  inst.flags = flags
  inst.handle = handle
  instances_out[index] = inst


@wp.kernel
def world_bounds(
  # Model:
  instance_type: wp.array1d[int],
  instance_mesh: wp.array2d[int],
  instance_size: wp.array2d[wp.vec3],
  mesh_lower: wp.array1d[wp.vec3],
  mesh_upper: wp.array1d[wp.vec3],
  mesh_frame_t: wp.array1d[wp.mat33],
  hfield_half: wp.array1d[wp.vec3],
  # Data in:
  instance_pos_in: wp.array2d[wp.vec3],
  instance_rot_in: wp.array2d[wp.mat33],
  # In:
  ids: wp.array1d[int],
  cols: wp.array1d[int],
  # Out:
  bounds_out: wp.array1d[float],
):
  """This world's `BOUNDS` entries in this frame (one thread per world: no cross-world reduction;
  empty: lower > upper): the box of its drawn instances (`reach.drawn` at its
  `reach.reference`; the others are hidden), the reference."""
  worldid = wp.tid()
  ref = reach.reference(instance_pos_in, instance_rot_in, instance_size, ids, cols, worldid)
  lo = wp.vec3(BIG)
  hi = wp.vec3(-BIG)
  for k in range(ids.shape[0]):
    gi = ids[k]
    gtype = instance_type[gi]
    col = cols[k]
    pos = wp.vec3(instance_pos_in[worldid, col])
    rot = wp.mat33(instance_rot_in[worldid, col])
    size = batched(instance_size, worldid, gi)
    if not reach.drawn(ref, pos, rot, size):
      continue
    center = pos
    if gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF):
      # The mesh's box in its GAS frame (a point without a mesh).
      meshid = batched(instance_mesh, worldid, gi)
      size = wp.vec3(0.0)
      if meshid >= 0:
        rot = gas_frame(mesh_frame_t, gtype, meshid, rot)
        center = pos + rot @ (0.5 * (mesh_lower[meshid] + mesh_upper[meshid]))
        size = 0.5 * (mesh_upper[meshid] - mesh_lower[meshid])
    elif gtype == int(ShapeType.HFIELD):
      # Its `hfield_size` box (a point without a height field; `bounds.world_aabb`).
      hid = batched(instance_mesh, worldid, gi)
      size = wp.vec3(0.0)
      if hid >= 0 and hid < hfield_half.shape[0]:
        size = hfield_half[hid]
    lower, upper = world_aabb(gtype, center, rot, size)
    lo = wp.min(lo, lower)
    hi = wp.max(hi, upper)
  base = worldid * BOUNDS
  for i in range(3):
    bounds_out[base + i] = lo[i]
    bounds_out[base + 3 + i] = hi[i]
    bounds_out[base + 6 + i] = ref[i]


@wp.kernel
def no_bounds(
  # Out:
  bounds_out: wp.array1d[float],
):
  """This world's `BOUNDS` entries without a per-world instance: an empty box, reference 0
  (deformables widen the box, `dynamic.flex_bounds`)."""
  worldid = wp.tid()
  base = worldid * BOUNDS
  for i in range(3):
    bounds_out[base + i] = BIG
    bounds_out[base + 3 + i] = -BIG
    bounds_out[base + 6 + i] = 0.0


@wp.kernel
def update_cell(
  # In:
  bounds: wp.array1d[float],
  grid: wp.vec3i,
  # Out:
  cell_out: wp.array1d[wp.vec3],
):
  """This world's padded cell bounds, uniform normalization scale, scene-coordinate center, grid
  offset and reference position from its `world_bounds` entries (the padded geometry fits inside a
  unit cube; the scale is a power of two: exact products and quotients, `trace.py`'s module doc).
  An empty world has finite padding and scale."""
  worldid = wp.tid()
  base = worldid * BOUNDS
  lo = wp.vec3(bounds[base], bounds[base + 1], bounds[base + 2])
  hi = wp.vec3(bounds[base + 3], bounds[base + 4], bounds[base + 5])
  slot = worldid * rt_trace.CELL_SIZE
  if lo[0] > hi[0]:
    lo = wp.vec3(0.0)
    hi = wp.vec3(0.0)
  extent = hi - lo
  largest = wp.max(wp.max(extent[0], extent[1]), extent[2])
  pad = wp.static(rt_trace.CELL_PAD_REL) * largest + wp.static(rt_trace.CELL_PAD_ABS)
  cell_out[slot + rt_trace.CELL_LOWER] = lo - wp.vec3(pad)
  cell_out[slot + rt_trace.CELL_UPPER] = hi + wp.vec3(pad)
  span = largest + 4.0 * pad
  scale = float(1.0)
  while scale * span > 1.0:
    scale = 0.5 * scale
  while 2.0 * scale * span <= 1.0:
    scale = 2.0 * scale
  cell_out[slot + rt_trace.CELL_SCALE] = wp.vec3(scale)
  cell_out[slot + rt_trace.CELL_CENTER] = 0.5 * (lo + hi)
  cell_out[slot + rt_trace.CELL_OFFSET] = rt_trace.world_offset(grid, worldid)
  cell_out[slot + rt_trace.CELL_REF] = wp.vec3(bounds[base + 6], bounds[base + 7], bounds[base + 8])
