"""The parts of Isaac Lab's renderer interface the adapter reads, as Protocols.

Isaac Lab's objects satisfy them structurally (`CameraRenderSpec`, `CameraCfg`, `ProxyArray`,
`CameraData`, `RaywarpRendererCfg`); `Simulation` is what the adapter needs from Isaac's
simulation (the Newton model it renders, body poses from the scene data provider, stage labels),
implemented over Isaac Lab in `renderer.py` and by plain objects in tests (no Isaac Lab needed).
Isaac's `ProxyArray` wraps either a camera output `Buffer` or a camera-state array.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
import warp as wp

from raywarp.scene.lighting.rig import Lighting

if TYPE_CHECKING:
  import newton

  from raywarp.integrations.isaaclab.usd_source import UsdSource


# An Isaac Lab camera output buffer: (N, H, W, C) of a Warp scalar (`CameraData.output`).
type Buffer = wp.array[object, [int, int, int, int]]
# Isaac Lab's segmentation info of one output: {"idToLabels": {id: label}, "idToSemantics": ...}.
type SegmentInfo = dict[str, dict[int, object]]
# Isaac Lab's per-shape segmentation ids of one output: (shape_count,) int32.
type ShapeIds = np.ndarray[[int], np.dtype[np.int32]]


class ProxyArray[A](Protocol):
  """Isaac Lab's `ProxyArray`: a Warp array `A` with a torch view (only the Warp side is read)."""

  @property
  def warp(self) -> A:
    """The Warp array."""
    ...


class Spawn(Protocol):
  """The camera's spawn config fields read (`PinholeCameraCfg`)."""

  clipping_range: tuple[float, float]
  distortion: object | None


class CameraCfg(Protocol):
  """The `CameraCfg` fields read."""

  width: int
  height: int
  data_types: Sequence[str]
  spawn: Spawn | None
  background_color: tuple[float, float, float] | None
  update_period: float


class CameraRenderSpec(Protocol):
  """Isaac Lab's `CameraRenderSpec`: one tiled camera (a view per environment)."""

  cfg: CameraCfg
  num_instances: int


class CameraData(Protocol):
  """Isaac Lab's `CameraData`: only `info` is written (the pixel outputs are bound)."""

  info: MutableMapping[str, object]


class Config(Protocol):
  """The renderer config fields read (`RaywarpRendererCfg`; plain values: Isaac's configclass
  rewrites nested dataclass fields, so settings are dotted overrides of the default)."""

  settings: Mapping[str, object]
  lighting: Lighting | None
  load_textures: bool
  depth_clipping_behavior: str
  use_cuda_graph: bool


class Simulation(Protocol):
  """What the adapter reads from the simulation (Isaac Lab: `renderer.IsaacSimulation`)."""

  @property
  def model(self) -> newton.Model:
    """The Newton model of every environment (Isaac's renderer model, PhysX or Newton physics)."""
    ...

  def usd_source(self) -> UsdSource | None:
    """An isolated prototype stage view; None for an explicit model-only source."""
    ...

  def body_poses(self) -> wp.array[wp.transformf, [int]] | None:
    """(body_count,) world poses of the model's bodies in body order (None: none published),
    complete for work queued next on the device's Warp stream."""
    ...

  def segmentation(self, kind: str) -> tuple[ShapeIds, SegmentInfo]:
    """Isaac's ids of each shape and their info for "instance_segmentation" or
    "semantic_segmentation" (Isaac Lab: its Newton Warp segmentation mapper)."""
    ...

  def shape_materials(self) -> Sequence[str]:
    """Each shape's bound material prim path ("" for none; empty: unknown)."""
    ...
