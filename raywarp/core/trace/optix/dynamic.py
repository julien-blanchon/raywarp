"""Triangle GASes whose vertices move: height fields and deformables.

- Height fields: one GAS per rendered height field over its triangle mesh (`Meshes.hfield_registry`:
  MuJoCo's grid, two triangles per cell, the software BLAS's own vertex and index buffers), in the
  height field's frame like a mesh GAS (its instances are triangle instances, `instances.py`). Built
  with the static structures; refitted when the terrain changes (`Meshes.hfield_revision`, bumped by
  `MjwBinding.refit_hfields`), which keeps the handles the instance records hold.
- Deformables: one GAS per surface part (dim 2 / 3) over its triangles of every world, each world's
  in its cell (`trace.to_cell`), refitted every frame with the per-world group (rebuilt on its
  schedule). The world-space triangles (`trace/accel/deformable_bvh.DeformableBvhs`, the software
  BVHs' writer; the shading reads them, `GeometryFields.flex_bvh_id`) are written first, then their
  bounds widen each world's cell (`flex_bounds`), then the cell-space copies are written
  (`flex_cell_points`). Part k's identity instance has instance id `trace.FLEX_INSTANCE` + k: its
  hits are reported as the software traversal's flex hits (`FLEX_HIT`,
  `trace/optix/device/trace.py`).

The hit programs read a height field's or deformable's face normal from the GAS's own vertices
(`optixGetTriangleVertexData`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.rays import FlexPart
from raywarp.core.kernels.abi.tables import material_rgba
from raywarp.core.pipeline.platform.device import stream_handle
from raywarp.core.trace.accel.deformable_bvh import DeformableBvhs
from raywarp.core.trace.optix import api, gas
from raywarp.core.trace.optix import instances as ops
from raywarp.core.trace.optix.device import trace as rt_trace

if TYPE_CHECKING:
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.deformables import Deformables
  from raywarp.core.pipeline.gpu_scene.gpu_scene import Meshes
  from raywarp.core.trace.optix.context import DeviceContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


# Threads per world of `flex_bounds` (a block: a strided min / max each, then the block's).
FLEX_BLOCK: Final = 128


@wp.kernel
def flex_bounds(
  # In:
  points: wp.array1d[wp.vec3],
  per_world: int,
  # Out:
  bounds_out: wp.array1d[float],
):
  """Widens world w's cell box (`instances.world_bounds`' first six entries) by its points of a
  deformable part's triangles (`per_world` per world, world-major). One block per world: each
  thread's strided min / max, then the block's per axis (min / max are exact: the box does not
  depend on the order; the per-point float atomics were CAS loops contending on 6 words)."""
  worldid, lane = wp.tid()
  lanes = wp.block_dim()
  first = worldid * per_world
  lo = wp.vec3(wp.static(ops.BIG))
  hi = wp.vec3(-wp.static(ops.BIG))
  for i in range(lane, per_world, lanes):
    p = points[first + i]
    lo = wp.min(lo, p)
    hi = wp.max(hi, p)
  base = worldid * wp.static(ops.BOUNDS)
  for k in range(3):
    low = wp.tile_min(wp.tile(lo[k]))
    high = wp.tile_max(wp.tile(hi[k]))
    if lane == 0:
      bounds_out[base + k] = wp.min(bounds_out[base + k], low[0])
      bounds_out[base + 3 + k] = wp.max(bounds_out[base + 3 + k], high[0])


@wp.kernel
def flex_cell_points(
  # Model:
  color_rule: int,
  material_base_color: wp.array2d[wp.vec4],
  flex_part: wp.array1d[FlexPart],
  # In:
  part: int,
  points: wp.array1d[wp.vec3],
  per_world: int,
  cell: wp.array1d[wp.vec3],
  # Out:
  points_out: wp.array1d[wp.vec3],
):
  """Point i of deformable part `part`'s triangles in its world's cell (world i //
  `per_world`); the cell's offset (degenerate triangles: no hit) where the part is invisible in
  that world (alpha 0, its material's for the default color, as the software traversal skips it:
  `trace.accel.instances.intersect_flex`)."""
  i = wp.tid()
  worldid = i // per_world
  fp = flex_part[part]
  if material_rgba(color_rule, material_base_color, worldid, fp.material, fp.rgba)[3] == 0.0:
    points_out[i] = rt_trace.cell_offset(cell, worldid)
  else:
    points_out[i] = rt_trace.to_cell(cell, worldid, points[i])


def _triangles(
  ctx: DeviceContext, points: wp.array[wp.vec3, [int]], indices: wp.array[wp.int32, [int]], *, device: wp.Device
) -> gas.Accel:
  """Refittable triangle GAS (no any-hit) over `points` and the flat triangle `indices`."""
  tri = wp.array(ptr=indices.ptr, dtype=wp.vec3i, shape=(indices.shape[0] // 3,), device=device)
  build_input = gas.triangles(points, tri, api.GEOMETRY_FLAG_DISABLE_ANYHIT)
  return gas.Accel(ctx, build_input, gas.dynamic_build_flags(), device=device, keep=(points, indices, tri))


@final
class HeightFieldGases:
  """The height fields' GASes (module doc).

  Attributes:
    handle: (max(nhfield, 1),) traversable handle per height field (0: not rendered).
    revision: the `Meshes.hfield_revision` their vertices were last built or refitted at.
  """

  def __init__(self, ctx: DeviceContext, meshes: Meshes, *, device: wp.Device) -> None:
    """Builds the GASes of `meshes`' height fields on `device`'s current stream."""
    self.meshes = meshes
    self.accels = {h: _triangles(ctx, m.points, m.indices, device=device) for h, m in meshes.hfield_registry.items()}
    nhfield = 0 if meshes.hfield_bvh_id is None else meshes.hfield_bvh_id.shape[0]
    handles = np.zeros(max(nhfield, 1), np.uint64)
    stream = stream_handle(wp.get_stream(device))
    for h, accel in self.accels.items():
      handles[h] = accel.build(stream)
    self.handle = wp.array(handles, dtype=wp.uint64, device=device)
    self.revision = meshes.hfield_revision

  def refit(self, stream: int) -> None:
    """Refits the GASes on `stream` when the terrain changed since their last build (same handles:
    the instance records stay valid)."""
    if self.revision == self.meshes.hfield_revision:
      return
    self.revision = self.meshes.hfield_revision
    for accel in self.accels.values():
      accel.build(stream, update=True)


@final
class DeformableGases:
  """The deformables' GASes over every world (module doc).

  Attributes:
    bvhs: the world-space triangle meshes (`DeformableBvhs`; their BVHs are not refitted here).
    parts: (part, world-space points, cell-space points, triangles per world x 3) per surface part.
    gases: (GAS, instance id) per surface part (`group.InstanceGroup`'s identity instances).
  """

  def __init__(self, ctx: DeviceContext, d: Deformables, nworld: int, *, device: wp.Device) -> None:
    """The GASes of `d`'s surface parts over `nworld` worlds on `device` (built by the group)."""
    self.device = device
    self.bvhs = DeformableBvhs(d, nworld, device=device, constructor="lbvh")
    self.parts: list[tuple[int, wp.array[wp.vec3, [int]], wp.array[wp.vec3, [int]], int]] = []
    self.gases: list[tuple[gas.Accel, int]] = []
    for part, mesh in self.bvhs.meshes:
      cell_points = wp.zeros_like(mesh.points)
      self.parts.append((part, mesh.points, cell_points, mesh.points.shape[0] // nworld))
      self.gases.append((_triangles(ctx, cell_points, mesh.indices, device=device), rt_trace.FLEX_INSTANCE + part))

  def write_world(self, f: Frame, bounds: wp.array[float, [int]]) -> None:
    """Writes the frame's world-space triangles and widens each world's cell box in `bounds`
    (`instances.world_bounds`' output) by them."""
    self.bvhs.write(f.flex_vert)
    for _part, points, _cell_points, per_world in self.parts:
      wp.launch_tiled(
        flex_bounds,
        dim=(points.shape[0] // per_world,),
        inputs=[points, per_world],
        outputs=[bounds],
        block_dim=FLEX_BLOCK,
        device=self.device,
      )

  def write_cells(self, f: Frame, color_rule: int, cell: wp.array[wp.vec3, [int]]) -> None:
    """Writes the GASes' cell-space points from the world-space ones (after the cells' update;
    `color_rule`: the scene's, with `f`'s material colors: hidden parts)."""
    for part, points, cell_points, per_world in self.parts:
      wp.launch(
        flex_cell_points,
        dim=points.shape[0],
        inputs=[color_rule, f.material_base_color, self.bvhs.parts, part, points, per_world, cell],
        outputs=[cell_points],
        device=self.device,
      )
