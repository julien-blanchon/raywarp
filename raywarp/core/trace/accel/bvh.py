"""Scene BVHs: the per-world instances (and deformables) of every world, and the shared ones.

* World BVH: one `wp.Bvh` (SAH) over nworld x (nrendered per-world instances + ndeformable) boxes
  grouped by world; its topology comes from the scene's initial poses (as MuJoCo Warp builds it;
  one row of poses read by every world, not a host copy per world), its bounds are refit from the
  current poses every render. Refit-only trees degrade when instances travel far from where the
  topology was built (mobile bases, scattered objects): `performance.rebuild_every` = N rebuilds
  the topology from the current poses every N-th refit, in place, gated on the device
  (`cadence.Cadence` gating `lbvh.Rebuild`'s launches: a captured CUDA graph replays the cadence,
  and the rebuild allocates nothing, so it captures). The deformables' triangle BVHs are refit
  only (Warp has no in-place `Mesh` rebuild).
* Shared BVH: one `wp.Bvh` over the rendered shared instances (`sharing.Layout.shared_ids`), built
  on the first refit from the poses that frame renders and traced by every world after its own
  group of the world BVH. Shared scenery is static: the per-frame validation raises when it moves
  (`core/pipeline/worlds/validator.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.cadence import Cadence
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.trace.accel import bounds, lbvh
from raywarp.core.trace.accel.deformable_bvh import DeformableBvhs
from raywarp.core.trace.accel.packed import PackedNodes

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import Ids, Table
  from raywarp.core.kernels.abi.rays import SoftwareGeometry
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.binding import Slots
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.worlds.sharing import Layout


def _shared_slots(layout: Layout) -> Ids[int]:
  """The pose slot of each rendered shared instance in `Frame.shared_poses`: its row of the shared
  pose table (compact layout), else its instance id (column of world 0's row)."""
  ids = np.asarray(layout.shared_ids, np.int32)
  if layout.compact:
    return (-1 - np.asarray(layout.pose_slot)[ids]).astype(np.int32)
  return ids


def _ids_array(ids: Ids[int], device: wp.DeviceLike) -> wp.array[int, [int]]:
  """A device array of `ids` (one dummy 0 when empty: kernels index it only below its length)."""
  return wp.array(ids if len(ids) else np.zeros(1, np.int32), dtype=int, device=device)


@final
class SceneBvh:
  """Scene BVHs of `nworld` worlds of a GpuScene under a Renderer's `sharing.Layout`.

  Attributes:
    nworld, scene, layout: what the BVHs were built for (fixed per Renderer).
    ninstance: rendered per-world instances per world (`enabled_ids`).
    nflexprim: deformable primitives per world.
    enabled_ids: (ninstance,) rendered per-world instance ids; `enabled_cols` their pose columns.
    bvh: the world wp.Bvh (None when it would be empty); `bvh_id` its id (0: none).
    lower, upper, group: the world BVH's primitive boxes and groups.
    group_root: (nworld,) root node per world.
    deformables: the deformables' triangle BVHs (`DeformableBvhs`).
    nshared: rendered shared instances; `shared_ids` (nshared,) their ids (primitives of the
      shared BVH).
    shared_bvh: the shared wp.Bvh (None before the first refit or without shared instances);
      `shared_bvh_id`, `shared_root`.
    shared_lower, shared_upper: (nshared,) world bounds of the shared instances.
  """

  def __init__(self, scene: GpuScene, nworld: int, layout: Layout) -> None:
    """Builds the world BVH from the scene's initial poses (the shared one waits for the first
    refit)."""
    self.nworld = nworld
    self.scene = scene
    self.layout = layout
    world_ids = self._world_instances(scene, layout)
    cols = np.asarray(layout.pose_slot, np.int32)[world_ids] if layout.compact else world_ids
    device = scene.device
    self.enabled_ids = _ids_array(world_ids, device)
    self.enabled_cols = _ids_array(cols, device)
    self.ninstance = len(world_ids)
    self.nflexprim = scene.deformables.nprim
    self.deformables = DeformableBvhs(scene.deformables, nworld, device=device)
    self._mesh_centre, self._mesh_half = scene.meshes.local_boxes()
    self._hfield_half = scene.meshes.hfield_bounds_size
    self.group_root = wp.zeros(nworld, dtype=int, device=device)
    self._cadence = Cadence(device)  # periodic rebuilds (performance.rebuild_every)
    self._rebuild: lbvh.Rebuild | None = None  # their scratch (made with the world BVH)
    self.bvh: wp.Bvh | None = None
    self.bvh_id = wp.uint64(0)
    self._world_nodes: PackedNodes | None = None  # the world BVH's packed children, repacked per refit
    total = nworld * (self.ninstance + self.nflexprim)
    self.lower = wp.zeros(total, dtype=wp.vec3, device=device)
    self.upper = wp.zeros(total, dtype=wp.vec3, device=device)
    self.group = wp.zeros(total, dtype=int, device=device)
    if total:
      self._build_world()
    # Shared instances: one static BVH, built on the first refit.
    shared_ids = np.asarray(layout.shared_ids, np.int32)
    self.nshared = len(shared_ids)
    self.shared_ids = _ids_array(shared_ids, device)
    self.shared_bvh: wp.Bvh | None = None
    self.shared_bvh_id = wp.uint64(0)
    self._shared_nodes: PackedNodes | None = None
    self.shared_root = 0
    self.shared_lower = wp.zeros(self.nshared, dtype=wp.vec3, device=device)
    self.shared_upper = wp.zeros(self.nshared, dtype=wp.vec3, device=device)
    self._shared_group = wp.zeros(max(self.nshared, 1), dtype=int, device=device)
    self._shared_slots = wp.array(_shared_slots(layout), dtype=int, device=device)
    self._visible_slot = _ids_array(np.asarray(layout.visible_slot, np.int32), device)
    self._reference = wp.zeros(nworld, dtype=wp.vec3, device=device)  # per world, `_hide`

  @staticmethod
  def _world_instances(scene: GpuScene, layout: Layout) -> Ids[int]:
    """Rendered per-world instances held by the world BVH (analytic planes are intersected
    apart)."""
    rendered = np.asarray(scene.rendered, dtype=np.int32)
    per_world = np.asarray(layout.per_world, bool).copy()
    per_world[np.asarray(layout.planes, np.int64)] = False
    return rendered[per_world[rendered]]

  def _build_world(self) -> None:
    """Builds the world BVH from the scene's initial poses (one row, read by every world) and its
    group roots.

    Sizes and meshes: the scene's tables, or its binding's model arrays.
    """
    scene = self.scene
    device = scene.device
    columns = np.asarray(self.layout.columns, np.int64)
    flex = scene.deformables
    model: Slots = {**scene.tables, **scene.binding.slots(None)}
    self._bounds(
      model["instance.mesh"],
      model["instance.size"],
      wp.array(scene.init_pose.pos[None, columns], dtype=wp.vec3, device=device),
      wp.array(scene.init_pose.rot[None, columns], dtype=wp.mat33, device=device),
      wp.array(flex.init_vert[np.newaxis], dtype=wp.vec3, device=device),
    )
    self.bvh = wp.Bvh(self.lower, self.upper, groups=self.group, constructor="sah")
    self.bvh_id = self.bvh.id
    # Built now, outside any capture (rebuild_every is a runtime setting: any refit may rebuild).
    self._rebuild = lbvh.Rebuild(self.bvh, self.nworld, self.ninstance + self.nflexprim)
    self._world_nodes = PackedNodes(wp.array([self.bvh.id], dtype=wp.uint64, device=device), meshes=False)
    wp.launch(bounds.group_roots, dim=self.nworld, inputs=[self.bvh.id], outputs=[self.group_root], device=device)

  def fill_geometry(self, geo: SoftwareGeometry) -> None:
    """Writes the software traversal's fields of a frame's SoftwareGeometry (both trees,
    deformables)."""
    geo.bvh_id = self.bvh_id
    geo.bvh_nodes = wp.uint64(0 if self._world_nodes is None else int(self._world_nodes.host_addresses[0]))
    geo.shared_nodes = wp.uint64(0 if self._shared_nodes is None else int(self._shared_nodes.host_addresses[0]))
    geo.group_root = self.group_root
    geo.bvh_ninstance = self.ninstance
    geo.bvh_nflexprim = self.nflexprim
    geo.enabled_instance_ids = self.enabled_ids
    geo.flex_bvh_id = self.deformables.ids
    geo.flex_part = self.deformables.parts
    geo.flex_nodes = self.deformables.nodes.addresses
    geo.flex_group_root = self.deformables.group_root
    geo.shared_bvh_id = self.shared_bvh_id
    geo.shared_root = self.shared_root
    geo.shared_ids = self.shared_ids

  def _shared_bounds(self, frame: Frame) -> None:
    """World bounds of the shared instances at the frame's shared poses."""
    shared_pos, shared_rot = frame.shared_poses(self.layout)
    wp.launch(
      bounds.instance_bounds,
      dim=(1, self.nshared),
      inputs=[
        self.scene.instance_type,
        frame.instance_mesh,
        frame.instance_size,
        shared_pos.reshape((1, -1)),
        shared_rot.reshape((1, -1)),
        self.nshared,
        self.shared_ids,
        self._mesh_centre,
        self._mesh_half,
        self._hfield_half,
        self._shared_slots,
      ],
      outputs=[self.shared_lower, self.shared_upper, self._shared_group],
      device=self.scene.device,
    )

  def _build_shared(self, frame: Frame) -> None:
    """Builds the shared BVH from the frame's shared poses (world 0's sizes and meshes)."""
    self._shared_bounds(frame)
    self.shared_bvh = wp.Bvh(self.shared_lower, self.shared_upper, groups=self._shared_group, constructor="sah")
    self.shared_bvh_id = self.shared_bvh.id
    ids = wp.array([self.shared_bvh.id], dtype=wp.uint64, device=self.scene.device)
    self._shared_nodes = PackedNodes(ids, meshes=False)
    root = wp.zeros(1, dtype=int, device=self.scene.device)
    wp.launch(bounds.group_roots, dim=1, inputs=[self.shared_bvh.id], outputs=[root], device=self.scene.device)
    self.shared_root = int(host_copy(root)[0])

  def _bounds[NRow: IntVar, NColumn: IntVar, NVert: IntVar](
    self,
    instance_mesh: Table[int, NRow, int],
    instance_size: Table[wp.vec3, NRow, int],
    instance_pos: Table[wp.vec3, NRow, NColumn],
    instance_rot: Table[wp.mat33, NRow, NColumn],
    flex_vert: Table[wp.vec3, NRow, NVert],
  ) -> None:
    """World-BVH primitive bounds of every world from these instance and deformable tables (1 or
    nworld rows)."""
    group_size = self.ninstance + self.nflexprim
    if self.ninstance:
      wp.launch(
        bounds.instance_bounds,
        dim=(self.nworld, self.ninstance),
        inputs=[
          self.scene.instance_type,
          instance_mesh,
          instance_size,
          instance_pos,
          instance_rot,
          group_size,
          self.enabled_ids,
          self._mesh_centre,
          self._mesh_half,
          self._hfield_half,
          self.enabled_cols,
        ],
        outputs=[self.lower, self.upper, self.group],
        device=self.scene.device,
      )
    flex = self.scene.deformables
    if self.nflexprim:
      wp.launch(
        bounds.flex_bounds,
        dim=(self.nworld, self.nflexprim),
        inputs=[
          flex.vertadr,
          flex.vertnum,
          flex.edge,
          flex.radius,
          flex_vert,
          flex.prim_flexid,
          flex.prim_edgeid,
          self.ninstance,
          group_size,
        ],
        outputs=[self.lower, self.upper, self.group],
        device=self.scene.device,
      )

  def refit(self, frame: Frame, rebuild_every: int) -> None:
    """Refits the world BVH to the frame's poses, or rebuilds it from them on every
    `rebuild_every`-th call (0: never); the first call builds the shared BVH. Hidden instances
    are points in either (`_hide`)."""
    if self.nshared and self.shared_bvh is None:
      self._build_shared(frame)
    if self.bvh is None or self._world_nodes is None:
      return
    self._bounds(frame.instance_mesh, frame.instance_size, frame.instance_pos, frame.instance_rot, frame.flex_vert)
    self._hide(frame)
    if rebuild_every > 0 and self._rebuild is not None:
      self._rebuild.rebuild(self._cadence.tick(rebuild_every), self.lower, self.upper, self.group_root)
    self.bvh.refit()  # the bounds of the current topology (a rebuilt one's first)
    self._world_nodes.refresh()  # the packed children follow the refit or rebuilt tree
    self.deformables.refit(frame.flex_vert)

  def _hide(self, frame: Frame) -> None:
    """Shrinks the boxes of the instances hidden in their world (alpha 0, per-world visibility,
    not drawn: `reach.drawn` at the world's `reach.reference`) to points: traversals skip them
    unread. The shared tree, built once for every world, keeps its boxes (its instances are hidden
    per world: the traversal tests them)."""
    if not self.ninstance:
      return
    device = self.scene.device
    wp.launch(
      bounds.world_references,
      dim=self.nworld,
      inputs=[frame.instance_size, frame.instance_pos, frame.instance_rot, self.enabled_ids, self.enabled_cols],
      outputs=[self._reference],
      device=device,
    )
    wp.launch(
      bounds.hide_bounds,
      dim=(self.nworld, self.ninstance),
      inputs=[
        int(self.scene.color_rule),
        frame.instance_material,
        frame.instance_color,
        frame.material_base_color,
        self._visible_slot,
        frame.instance_size,
        frame.instance_visible,
        frame.instance_pos,
        frame.instance_rot,
        int(self.layout.nvisible > 0),
        self.ninstance + self.nflexprim,
        self.enabled_ids,
        self.enabled_cols,
        self._reference,
      ],
      outputs=[self.lower, self.upper],
      device=device,
    )
