"""The instances of a `SceneBuilder`: chunks of validated instances, expanded at build.

`SceneBuilder.add_instances` checks the values; this pool keeps one chunk per call (a
PointInstancer adds thousands at once, a stage walk one per shape) with its shared values once,
and expands them into the `Instances` columns in one pass per column (`table`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import final

import numpy as np

from raywarp.scene.model.table import StringPool
from raywarp.scene.model.tables import Instances
from raywarp.scene.model.types import Array, Column, InstanceKind, MaterialId, MeshId


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class InstanceChunk:
  """Validated columns shared by N instances added at once (`SceneBuilder.add_instances`).

  Attributes:
    kind: their shape.
    xforms: (N, 3, 4) float32 world transforms.
    mesh: their mesh (NO_MESH for primitives).
    material: their material (NO_MATERIAL: none).
    size: (3,) primitive sizes.
    color: (4,) linear RGBA tint.
    flags: InstanceFlags.
    segment: their segment id (None: each its own instance index).
    source: their id in what the adapter binds (-1: none).
  """

  kind: InstanceKind
  xforms: np.ndarray[[int, 3, 4], np.dtype[np.float32]]
  mesh: MeshId
  material: MaterialId
  size: np.ndarray[[3], np.dtype[np.float64]]
  color: np.ndarray[[4], np.dtype[np.float64]]
  flags: int
  segment: int | None
  source: int


@final
class InstancePool:
  """Instances added so far: the chunks themselves (their shared values once each) and their
  per-instance prim / label ids, expanded into columns by `table`.

  Attributes:
    bodies: rigid-body paths (instances and lights index them: `Instances.bodies`).
  """

  def __init__(self) -> None:
    """An empty pool."""
    self._added: list[InstanceChunk] = []
    self._bodies: list[int] = []  # per chunk: its body id
    self._prims: list[Column[np.int32]] = []
    self._label_ids: list[Column[np.int32]] = []
    self._count = 0
    self._paths = StringPool()
    self._labels = StringPool()
    self.bodies = StringPool()

  def __len__(self) -> int:
    """Number of instances."""
    return self._count

  def add(self, chunk: InstanceChunk, *, paths: Sequence[str] | None, body: str | None, labels: Sequence[str] | None) -> range:
    """Appends the instances of `chunk` with their prim `paths`, rigid `body` and semantic `labels`
    ("" or None: none); their indices.

    Raises:
      TypeError: for one string as `paths` / `labels`.
      ValueError: when `paths` / `labels` do not hold one entry per instance.
    """
    count, start = len(chunk.xforms), self._count
    if isinstance(labels, str):
      raise TypeError(f"labels is a sequence of one string for each of the instances, got the string {labels!r}")
    # Both checked before either pool row is added (a failed call leaves the pool unchanged).
    prims = self._paths.ids(paths, count, "paths", rows="instances")
    label_ids = self._labels.ids(None if labels is None else [s or None for s in labels], count, "labels", rows="instances")
    self._prims.append(prims)
    self._label_ids.append(label_ids)
    self._bodies.append(self.bodies.add(body or None))
    self._added.append(chunk)
    self._count += count
    return range(start, start + count)

  def xforms(self) -> np.ndarray[[int, 3, 4], np.dtype[np.float32]]:
    """The (N, 3, 4) transforms of the instances added so far (float32, as stored)."""
    return np.concatenate([c.xforms for c in self._added]) if self._added else np.zeros((0, 3, 4), np.float32)

  def table(self) -> Instances:
    """The `Instances` table of every instance added."""
    names = {"paths": self._paths.items, "bodies": self.bodies.items, "labels": self._labels.items}
    if not self._added:
      return Instances.from_columns(dict.fromkeys(Instances.SCHEMA), names)
    added = self._added
    counts = np.array([len(c.xforms) for c in added], np.int64)

    def each[T: np.generic](values: list[int], dtype: type[T]) -> Column[T]:
      """A per-chunk value repeated for each of its instances."""
      return np.repeat(np.array(values, dtype), counts)

    # Segments: each instance its own index unless its chunk shares one.
    fixed = [c.segment is not None for c in added]
    segment = np.arange(self._count).astype(np.int32)
    if any(fixed):
      shared = np.repeat(np.array(fixed, bool), counts)
      segment[shared] = each([-1 if c.segment is None else c.segment for c in added], np.int32)[shared]
    columns: dict[str, Array | None] = {
      "kind": each([int(c.kind) for c in added], np.uint8),
      "mesh": each([c.mesh for c in added], np.int32),
      "material": each([c.material for c in added], np.int32),
      "size": np.repeat(np.array([c.size for c in added], np.float64).astype(np.float32), counts, axis=0),
      "xform": self.xforms(),
      "color": np.repeat(np.array([c.color for c in added], np.float64).astype(np.float32), counts, axis=0),
      "flags": each([c.flags for c in added], np.uint32),
      "segment": segment,
      "source": each([c.source for c in added], np.int32),
      "prim": np.concatenate(self._prims),
      "body": each(self._bodies, np.int32),
      "label": np.concatenate(self._label_ids),
    }
    return Instances.from_columns(columns, names)
