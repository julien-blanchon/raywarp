"""One OptiX instance group (`InstanceGroup`: an IAS over mesh / box instances plus the identity
instances of its curve and custom-primitive GASes) and the per-instance tables its records read
(`InstanceTables`); `accel.OptixScene` holds two (module doc of `trace/optix/accel.py`)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import stream_handle
from raywarp.core.trace.optix import api, gas
from raywarp.core.trace.optix import instances as ops
from raywarp.core.trace.optix.context import DeviceContext
from raywarp.core.trace.optix.device import trace as rt_trace
from raywarp.core.trace.optix.device.intrinsics import RAY_TYPES

if TYPE_CHECKING:
  from collections.abc import Sequence

  from raywarp._base.dims import Ids
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.trace.optix.device.programs import SceneFlags
  from raywarp.scene.model.types import InstanceId


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class _GasSlot:
  """Identity instance of a world-space GAS in its group's IAS.

  Attributes:
    accel: the curve, custom-primitive or deformable-triangle GAS.
    sbt_offset: SBT record of its hit group.
    index: its record in `InstanceGroup.instances`.
    flags: OptixInstance flags of that record (any-hit disabled, or enforced: the shared curves'
      per-world visibility, `OptixScene.visible_curves`, or translucent curves,
      `OptixScene.translucent_curves`).
    instance_id: OptiX instance id of that record (0; a deformable part's: `trace.FLEX_INSTANCE`
      + part).
    mask: visibility mask of that record (`trace.GAS_BITS`; a deformable's `trace.FLEX_BITS`).
    written: GAS handle last written into that record (a rebuild may change it).
  """

  accel: gas.Accel
  sbt_offset: int
  index: int
  flags: int
  instance_id: int = 0
  mask: int = rt_trace.GAS_BITS
  written: int = 0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InstanceTables:
  """Per-instance tables every triangle-instance record reads (`instances.write_instances`).

  Attributes:
    instance_type: (ninstance,) shape type of every instance.
    color_rule: the scene's `ColorRule` (`GpuScene.color_rule`).
    meshes: the static GASes (mesh handles, frames, cube handle).
    hfield_handle: (max(nhfield, 1),) GAS of each height field (0: not rendered,
      `dynamic.HeightFieldGases`).
    hfield_half: (max(nhfield, 1),) each height field's box half extents
      (`Meshes.hfield_bounds_size`).
    flags: (max(ninstance, 1),) OptixInstance flags (any-hit on cut-out instances and shared ones
      with a per-world visibility, face culling); with per-world optics (`world_flags`) each
      world's, world-major (nworld x max(ninstance, 1): `accel.write_world_flags`).
    handle: (max(ninstance, 1),) GAS of an instance with its own (a cut-out mesh with opacity
      micromaps, `omm/micromaps.py`), 0: its mesh's.
    mask: (max(ninstance, 1),) visibility mask bits (`trace.WORLD_BITS` / `SHARED_BITS`).
    visible_slot: (max(ninstance, 1),) column in the per-world visibility table
      (`Layout.visible_slot`).
    cell: (nworld * `trace.CELL_SIZE` + `trace.SHARED_BOX`,) each world's cell (lower, upper,
      scale, centre, offset, the world's reference position; `trace.CELL_*`), then the shared
      IAS's box (`trace.shared_span`).
    grid: world cells per axis.
    world_flags: `flags` are per world (Features.world_optics).
  """

  instance_type: wp.array[int, [int]]
  color_rule: int
  meshes: gas.MeshGases
  hfield_handle: wp.array[wp.uint64, [int]]
  hfield_half: wp.array[wp.vec3, [int]]
  flags: wp.array[wp.uint32, [int]]
  handle: wp.array[wp.uint64, [int]]
  mask: wp.array[wp.uint32, [int]]
  visible_slot: wp.array[int, [int]]
  cell: wp.array[wp.vec3, [int]]
  grid: wp.vec3i
  world_flags: bool


# The poses the instance writers read: per-world (nworld, ncolumn) positions and rotations, then
# the shared pose table's (or world 0's row: `accel.OptixScene._write_shared`).
type Poses = tuple[
  wp.array[wp.vec3, [int, int]], wp.array[wp.mat33, [int, int]], wp.array[wp.vec3, [int]], wp.array[wp.mat33, [int]]
]


# An instance group's triangle (mesh, SDF, height field, then box), sphere / capsule and ellipsoid /
# cylinder members.
type Members = tuple[list[InstanceId], list[InstanceId], list[InstanceId]]


# `HitGroups.members` kind of a structure's curves (the others: their custom-primitive type).
CURVES: Final = -1


def custom_types(ids: Sequence[InstanceId], instance_type: Ids[int]) -> tuple[int, ...]:
  """The custom-primitive types among instances `ids` (`trace.CUSTOM_TYPES` order: their hit
  groups' order)."""
  return tuple(t for t in rt_trace.CUSTOM_TYPES if any(instance_type[g] == t for g in ids))


@final
@dataclasses.dataclass(frozen=True, slots=True)
class HitGroups:
  """The shader binding table's hit groups over a scene's structures, in record order: the
  triangles' (meshes, boxes, height fields, deformables), then per structure (per-world, shared)
  its curves' and one per custom-primitive type (`members`). Each holds `RAY_TYPES` records in a
  row, one per ray type (the traces' SBT stride). `pipeline.Pipeline` writes the records in this
  order; an instance record points at its hit group's first one (`offset`).

  Attributes:
    world_curves, shared_curves: the structure has curves (spheres, capsules).
    world_types, shared_types: its custom-primitive types (`custom_types`).
  """

  world_curves: bool
  world_types: tuple[int, ...]
  shared_curves: bool
  shared_types: tuple[int, ...]

  @staticmethod
  def of(scene: SceneFlags) -> HitGroups:
    """The hit groups of `scene`'s structures."""
    return HitGroups(scene.has_world_curves, scene.primitive_types, scene.has_shared_curves, scene.shared_primitive_types)

  def members(self) -> list[tuple[bool, int]]:
    """(shared, `CURVES` or custom type) of every hit group after the triangles', in order."""
    out: list[tuple[bool, int]] = []
    for shared, curves, types in ((False, self.world_curves, self.world_types), (True, self.shared_curves, self.shared_types)):
      out += [(shared, CURVES)] * int(curves) + [(shared, t) for t in types]
    return out

  def offset(self, *, shared: bool, kind: int) -> int:
    """The SBT offset of an instance record of hit group (`shared`, `kind`) (`members`)."""
    return (1 + self.members().index((shared, kind))) * RAY_TYPES


def device_ids(values: Sequence[int], device: wp.Device) -> wp.array[int, [int]]:
  """`values` on `device` (one 0 when empty: kernels index it only below its length)."""
  return wp.array(list(values) or [0], dtype=int, device=device)


@final
class InstanceGroup:
  """One IAS over a group of instances: the per-world ones (`rows` = nworld) or the shared scenery
  (1).

  Attributes:
    shared: the shared scenery (instances in scene coordinates, world 0's sizes and alpha).
    rows: worlds the records repeat over.
    tri_ids, curve_ids, prim_ids: the mesh / box, sphere / capsule and ellipsoid / cylinder members.
    primitive_types: custom-primitive types present, in SBT order.
    tri_instance, curve_instance, prim_instance: device copies of the member lists.
    prim_code: (rows, nprim) the instance of each custom primitive, -1 - instance when hidden (its
      intersection program's input: `hits.make_intersection`).
    built: built once (later frames may refit).
    handle: the IAS's traversable handle (0: no member).
  """

  def __init__(
    self,
    *,
    ctx: DeviceContext,
    device: wp.Device,
    shared: bool,
    rows: int,
    members: Members,
    cols: Ids[int],
    instance_type: Ids[int],
    hit_groups: HitGroups,
    gas_flags: tuple[int, int],
    triangle_gases: Sequence[tuple[gas.Accel, int]] = (),
    bounds: int = 0,
  ) -> None:
    """Allocates the group's build inputs on `device` (`ctx`'s); `cols`: each instance's pose
    column or shared slot; `hit_groups`: the scene's, holding the group's custom-primitive types and
    its curve and custom-primitive GASes' records; `gas_flags`: those GASes' instance flags;
    `triangle_gases`: (GAS, instance id) of the world-space triangle
    GASes it also holds (deformables, `dynamic.py`: the triangle hit group, any-hit disabled, on
    `trace.FLEX_BITS`); `bounds`: the device address each IAS build and refit writes the IAS's
    AABB to (OptiX's emitted `PROPERTY_TYPE_AABBS`, 8-byte aligned; 0: none)."""
    self.shared, self.rows, self.device = shared, rows, device
    self._emit = (api.AccelEmitDesc * 1)(api.AccelEmitDesc(result=bounds, type=api.PROPERTY_TYPE_AABBS)) if bounds else None
    self.tri_ids, self.curve_ids, self.prim_ids = members
    self.primitive_types = hit_groups.shared_types if shared else hit_groups.world_types
    self.tri_instance, self.curve_instance, self.prim_instance = (device_ids(ids, device) for ids in members)
    self.tri_cols, self.curve_cols, self.prim_cols = (device_ids([int(cols[g]) for g in ids], device) for ids in members)
    ncurve, nprim = rows * len(self.curve_ids), rows * len(self.prim_ids)
    self.curve_vertices = wp.zeros(max(2 * ncurve, 1), dtype=wp.vec3, device=device)
    self.curve_widths = wp.zeros(max(2 * ncurve, 1), dtype=float, device=device)
    self.prim_aabbs = wp.zeros(max(nprim, 1), dtype=ops.Aabb, device=device)
    self.prim_code = wp.zeros((rows, max(len(self.prim_ids), 1)), dtype=int, device=device)
    # (GAS, which of `gas_flags`, its hit group): the curve GAS, then the custom-primitive one.
    world_gases: list[tuple[gas.Accel, int, int]] = []
    if ncurve:
      segments = wp.array(np.arange(ncurve).astype(np.uint32) * 2, dtype=wp.uint32, device=device)
      world_gases.append((gas.curves(ctx, self.curve_vertices, self.curve_widths, segments, device=device), 0, CURVES))
    if nprim:
      types = [self.primitive_types.index(int(instance_type[g])) for g in self.prim_ids]
      prim_sbt = wp.array(np.tile(np.asarray(types, np.uint32), rows), dtype=wp.uint32, device=device)
      ntypes = len(self.primitive_types)
      prims = gas.custom_primitives(ctx, self.prim_aabbs, prim_sbt, ntypes, device=device)
      world_gases.append((prims, 1, self.primitive_types[0]))
    # Their identity instances follow the triangle instances.
    first = rows * len(self.tri_ids)
    self.slots = [
      _GasSlot(accel=accel, sbt_offset=hit_groups.offset(shared=shared, kind=kind), index=first + j, flags=gas_flags[flag])
      for j, (accel, flag, kind) in enumerate(world_gases)
    ]
    self.slots += [
      _GasSlot(
        accel=accel,
        sbt_offset=0,
        index=first + len(world_gases) + j,
        flags=api.INSTANCE_FLAG_DISABLE_ANYHIT,
        instance_id=iid,
        mask=rt_trace.FLEX_BITS,
      )
      for j, (accel, iid) in enumerate(triangle_gases)
    ]
    index = first + len(self.slots)
    self.instances = wp.zeros(max(index, 1), dtype=ops.OptixInstance, device=device)
    self.ias = gas.instance_array(ctx, self.instances, device=device) if index else None
    self.built = False
    self.handle = 0

  @property
  def empty(self) -> bool:
    """Whether the group has no member (no IAS: handle 0)."""
    return self.ias is None

  def write(self, f: Frame, poses: Poses, t: InstanceTables, *, visibility: bool) -> None:
    """Writes the members' build inputs from frame `f` (Warp kernels; `instances.py`).

    `poses`: (per-world positions, rotations, shared positions, rotations) of the frame;
    `visibility`: the per-world visibility table hides per-world instances (`Features.visibility`;
    shared ones: the programs' test).
    """
    shading = [t.color_rule, f.instance_material, f.instance_size, f.instance_color, f.material_base_color]
    use_visible = visibility and not self.shared

    def data(ids: wp.array[int, [int]], cols: wp.array[int, [int]]) -> list[object]:
      """The writers' data and member inputs (poses, visibility, members `ids` at `cols`, cell)."""
      return [*poses, f.instance_visible, t.visible_slot, use_visible, ids, cols, t.cell]

    if self.curve_ids:
      wp.launch(
        ops.write_curves(shared=self.shared),
        dim=(self.rows, len(self.curve_ids)),
        inputs=[t.instance_type, *shading, *data(self.curve_instance, self.curve_cols)],
        outputs=[self.curve_vertices, self.curve_widths],
        device=self.device,
      )
    if self.prim_ids:
      wp.launch(
        ops.write_primitive_bounds(shared=self.shared),
        dim=(self.rows, len(self.prim_ids)),
        inputs=[t.instance_type, *shading, *data(self.prim_instance, self.prim_cols)],
        outputs=[self.prim_aabbs, self.prim_code],
        device=self.device,
      )
    if self.tri_ids:
      m = t.meshes
      model = [
        t.instance_type,
        f.instance_mesh,
        *shading,
        m.handle,
        m.frame_t,
        wp.uint64(m.cube),
        t.hfield_handle,
        t.flags,
        t.handle,
        t.mask,
      ]
      wp.launch(
        ops.write_instances(shared=self.shared, world_optics=t.world_flags),
        dim=(self.rows, len(self.tri_ids)),
        inputs=[*model, *data(self.tri_instance, self.tri_cols)],
        outputs=[self.instances],
        device=self.device,
      )

  def build(self, stream: wp.Stream, *, rebuild: bool) -> None:
    """Refits (or rebuilds) the group's GASes and IAS on `stream` (OptiX calls).

    A GAS whose handle changed gets its identity instance rewritten.
    """
    if self.ias is None:
      return
    handle = stream_handle(stream)
    update = self.built and not rebuild
    for slot in self.slots:
      slot.accel.build(handle, update=update)
      if slot.accel.handle != slot.written:
        wp.launch(
          ops.write_gas_instance,
          dim=1,
          inputs=[
            wp.uint64(slot.accel.handle),
            wp.uint32(slot.sbt_offset),
            wp.uint32(slot.flags),
            wp.uint32(slot.instance_id),
            wp.uint32(slot.mask),
            slot.index,
          ],
          outputs=[self.instances],
          stream=stream,
        )
        slot.written = slot.accel.handle
    self.handle = self.ias.build(handle, update=update, emit=self._emit)
    self.built = True
