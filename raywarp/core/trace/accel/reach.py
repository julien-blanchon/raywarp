"""Which per-world instances a world draws: the one rule of both traversals.

A per-world instance is drawn when it is `placed` (finite and within `CELL_LIMIT`: not a diverged
simulation) and lies within `NEAR_RADIUS` of its world's reference position (`reference`: a body
parked or teleported far away is hidden). The OptiX traversal leaves the others out of its world
cell (`trace/optix/instances.py`), so a far body cannot stretch the cell and cost the world's other
geometry its precision; the software traversal shrinks their boxes out of its world tree
(`bounds.hide_bounds`), so both backends show the same image. Shared instances are always drawn.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.kernels.abi.tables import batched

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Per-world instances placed farther than this [m] from the origin on any axis (or of a larger
# size, or not finite: a diverged simulation) are hidden: they cannot blow up their world's
# OptiX cell (each world's bounds affect only that world).
CELL_LIMIT: Final = 1.0e5
# Per-world instances farther than this from their world's reference position on any axis [m]
# are hidden. An OptiX world cell then spans at most ~2 x this plus the instances' sizes: a
# power-of-two scale of 2^-11 at worst, coordinates within ~1e-3 m in [b, 2b) cells; parked
# bodies (moved out of view, typically km away) and teleported ones leave it.
NEAR_RADIUS: Final = 1000.0
# Resolution of the per-world median's bisection [m] (`reference`): the reference only decides
# which instances are within `NEAR_RADIUS` of it (at most ~18 halvings of a 2 x CELL_LIMIT range).
MEDIAN_RESOLUTION: Final = 1.0
# Empty position box (`reference`'s starting value).
_BIG: Final = 1.0e30


@wp.func
def placed(
  # In:
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> bool:
  """Whether a per-world instance at `pos` of rotation `rot` and size `size` is finite and within
  `CELL_LIMIT` (NaN fails every comparison; a rotation entry is at most 1 in magnitude)."""
  ok = True
  for i in range(3):
    ok = ok and wp.abs(pos[i]) <= wp.static(CELL_LIMIT) and wp.abs(size[i]) <= wp.static(CELL_LIMIT)
    for j in range(3):
      ok = ok and wp.abs(rot[i, j]) <= wp.static(CELL_LIMIT)
  return ok


@wp.func
def within_reach(
  # In:
  ref: wp.vec3,
  pos: wp.vec3,
) -> bool:
  """Whether scene position `pos` lies within `NEAR_RADIUS` of reference position `ref` on every
  axis."""
  d = pos - ref
  return wp.max(wp.max(wp.abs(d[0]), wp.abs(d[1])), wp.abs(d[2])) <= wp.static(NEAR_RADIUS)


@wp.func
def drawn(
  # In:
  ref: wp.vec3,
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> bool:
  """Whether a per-world instance at `pos` (rotation `rot`, size `size`) of a world of reference
  position `ref` is drawn (see the module doc)."""
  return placed(pos, rot, size) and within_reach(ref, pos)


@wp.func
def reference(
  # Data in:
  instance_pos_in: wp.array2d[wp.vec3],
  instance_rot_in: wp.array2d[wp.mat33],
  # In:
  instance_size: wp.array2d[wp.vec3],
  ids: wp.array1d[int],
  cols: wp.array1d[int],
  worldid: int,
) -> wp.vec3:
  """World `worldid`'s reference position (0 without a placed per-world instance; instance
  `ids[k]` at pose column `cols[k]`): the centre of its placed instance positions' box when the
  box spans at most `NEAR_RADIUS` per axis (every instance then lies within it of any point of the
  box, as of the median: the same instances in reach), else per axis the lower median, by
  bisection to `MEDIAN_RESOLUTION` (deterministic, no sort, one thread per world)."""
  lo = wp.vec3(_BIG)
  hi = wp.vec3(-_BIG)
  n = int(0)
  for k in range(ids.shape[0]):
    col = cols[k]
    pos = wp.vec3(instance_pos_in[worldid, col])
    if placed(pos, wp.mat33(instance_rot_in[worldid, col]), batched(instance_size, worldid, ids[k])):
      lo = wp.min(lo, pos)
      hi = wp.max(hi, pos)
      n += 1
  if n == 0:
    return wp.vec3(0.0)
  extent = hi - lo
  if wp.max(wp.max(extent[0], extent[1]), extent[2]) <= wp.static(NEAR_RADIUS):
    return 0.5 * (lo + hi)
  out = wp.vec3(0.0)
  half = (n + 1) // 2
  for axis in range(3):
    a = lo[axis]
    b = hi[axis]
    while b - a > wp.static(MEDIAN_RESOLUTION):
      mid = 0.5 * (a + b)
      count = int(0)
      for k in range(ids.shape[0]):
        col = cols[k]
        pos = wp.vec3(instance_pos_in[worldid, col])
        if placed(pos, wp.mat33(instance_rot_in[worldid, col]), batched(instance_size, worldid, ids[k])):
          if pos[axis] <= mid:
            count += 1
      if count >= half:
        b = mid
      else:
        a = mid
    out[axis] = b
  return out
