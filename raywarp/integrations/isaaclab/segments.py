"""Isaac Lab's segmentation ids of a Newton-bound scene, in the device tables the renderer reads.

The ids and labels are Isaac Lab's own (its Newton Warp contract,
`isaaclab_newton.renderers.segmentation.NewtonSegmentationMapper`, read through
`types.Simulation.segmentation`): per Newton shape, `instance_segmentation` groups shapes by their
nearest labelled ancestor prim (UsdSemantics labels passing the renderer's `semantic_filter`),
`semantic_segmentation` by their label payload ({type: "a,b"}); 0 background, 1 unlabelled, real
groups from 2; `idToLabels` / `idToSemantics` keyed by int id. Here they become the renderer's
segment table (the instance id each instance writes in each world: the megakernel writes it) and
the instance id -> semantic id table one lookup kernel reads (a labelled ancestor has one payload).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, final

import numpy as np
import warp as wp

from raywarp.integrations.isaaclab import contract
from raywarp.integrations.isaaclab.kernels import BACKGROUND_ID, UNLABELLED_ID
from raywarp.integrations.newton.binding import IntTable

if TYPE_CHECKING:
  from raywarp.integrations.isaaclab.types import SegmentInfo, Simulation
  from raywarp.integrations.newton.binding import NewtonBinding


@final
@dataclasses.dataclass(frozen=True)
class SegmentTables:
  """A scene's segmentation tables and Isaac's info.

  Attributes:
    ids: (nworld, ninstance) int32 instance-segmentation id each instance writes in each world.
    classes: (max instance id + 1,) int32 semantic-segmentation id of each instance id.
    info: Isaac's `CameraData.info` entry per output name (`idToLabels`, `idToSemantics`).
  """

  ids: wp.array[wp.int32, [int, int]]
  classes: wp.array[wp.int32, [int]]
  info: dict[str, SegmentInfo]


def world_shapes(binding: NewtonBinding, nworld: int) -> IntTable:
  """(nworld, ninstance) Newton shape of each instance in each render world."""
  table = binding.shape_table()
  shapes = np.repeat(table.shape0[None], nworld, axis=0).astype(np.int32)
  moving = np.flatnonzero(table.column >= 0)
  if len(moving):
    rows = table.rows[np.arange(nworld) % table.rows.shape[0]]
    shapes[:, moving] = rows[:, table.column[moving]]
  return shapes


def segments(
  binding: NewtonBinding, nworld: int, sim: Simulation, device: wp.DeviceLike, *, shapes: IntTable | None = None
) -> SegmentTables:
  """The segmentation tables of a scene bound by `binding` from Isaac's per-shape ids.

  Raises:
    ValueError: when Isaac's grouping gives one instance id two semantic ids (not a nearest
      labelled ancestor grouping).
  """
  instance, instance_info = sim.segmentation(contract.INSTANCE)
  semantic, semantic_info = sim.segmentation(contract.SEMANTIC)
  top = int(instance.max()) if len(instance) else int(UNLABELLED_ID)
  classes = np.full(top + 1, int(UNLABELLED_ID), np.int32)
  classes[int(BACKGROUND_ID)] = int(BACKGROUND_ID)
  classes[instance] = semantic
  if np.any(classes[instance] != semantic):
    raise ValueError("Isaac's instance groups do not have one semantic class each")
  shapes = world_shapes(binding, nworld) if shapes is None else shapes
  ids = np.full(shapes.shape, int(UNLABELLED_ID), np.int32)
  valid = (shapes >= 0) & (shapes < len(instance))
  ids[valid] = instance[shapes[valid]]
  return SegmentTables(
    ids=wp.array(ids, dtype=wp.int32, device=device),
    classes=wp.array(classes, dtype=wp.int32, device=device),
    info={contract.INSTANCE: instance_info, contract.SEMANTIC: semantic_info},
  )
