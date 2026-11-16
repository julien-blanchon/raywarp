"""Canonical material colors shared by Isaac cameras; ordinary writes stay on device."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.derive.materials import base_color_affine
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.upload.optics import material_optics
from raywarp.integrations.isaaclab import kernels
from raywarp.integrations.isaaclab.session import check_device

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.scene.model.report import LoadReport
  from raywarp.scene.model.scene import Scene


type ColorTable = wp.array[wp.vec4, [int, int]]
type WorldMask = wp.array[bool, [int]]
# Isaac's per-channel material row offsets of one update ("color", "roughness", ...).
type Offsets = dict[str, wp.array[wp.int32, [int]]]
type Rows = wp.array[wp.int32, [int]]
_BASE_INPUTS: Final = frozenset({"diffuseColor", "diffuse_color_constant", "diffuse_tint"})


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MaterialTargets:
  """Prepared CSR targets of a composed channel, shared by color and scalar input writers."""

  start: np.ndarray[[int], np.dtype[np.int32]]
  pairs: np.ndarray[[int, 2], np.dtype[np.int32]]
  sources: np.ndarray[[int], np.dtype[np.int32]]
  columns: np.ndarray[[int], np.dtype[np.int32]]
  width: int

  @classmethod
  def create(
    cls,
    *,
    layout: MaterialLayout,
    paths: Sequence[str],
    names: Sequence[str],
    accepted: frozenset[str],
    report: LoadReport,
    blocked: np.ndarray[[int], np.dtype[np.bool_]] | None = None,
  ) -> MaterialTargets:
    """Resolve exact addresses and authored input names, reporting omitted unsupported inputs."""
    if len(set(paths)) != len(paths):
      raise ValueError("composed material paths must be unique")
    if len(names) != len(paths):
      raise ValueError("material input names must match the composed material paths")
    row_of: dict[str, int] = {}
    for r, (path, name) in enumerate(zip(paths, names, strict=True)):
      if name in accepted:
        row_of[path] = r
      else:
        report.drop("isaac.material.input(unsupported)", f"{path}:{name}")
    rows = np.asarray([[row_of.get(p, -1) for p in world] for world in layout.paths], np.int32)
    if blocked is not None:
      for world_paths in layout.paths:
        for column, path in enumerate(world_paths):
          if blocked[column] and path in row_of:
            report.drop("isaac.material.mapped_input(unsupported)", f"{path}:{names[row_of[path]]}")
      rows[:, blocked] = -1
    world, column = np.nonzero(rows >= 0)
    order = np.argsort(rows[world, column], kind="stable")
    counts = np.bincount(rows[world, column], minlength=len(paths))
    return cls(
      start=np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
      pairs=np.stack([world[order], column[order]], axis=1).astype(np.int32) if len(order) else np.zeros((1, 2), np.int32),
      sources=rows[world[order], column[order]],
      columns=column[order].astype(np.int32),
      width=int(counts.max()) if len(counts) else 0,
    )


@final
class ScatterTargets:
  """The device side of one channel writer: its `MaterialTargets` CSR, the rows of a full sync,
  and the per-world change flags that reset only the worlds a write changed.

  Attributes:
    start, pairs: the CSR targets (`MaterialTargets.start` / `pairs`).
    width: the most targets of one row (the scatter's third launch dimension; 0: closed).
  """

  def __init__(
    self,
    targets: MaterialTargets,
    *,
    nrow: int,
    nworld: int,
    device: wp.Device,
    reset: Callable[[WorldMask], None],
    before_write: Callable[[], None] | None,
  ) -> None:
    """Uploads the targets of `nrow` composed rows over `nworld` worlds to `device`."""
    self.device = device
    self.width = targets.width
    self.start = wp.array(targets.start, dtype=wp.int32, device=device)
    self.pairs = wp.array(targets.pairs, dtype=wp.vec2i, device=device)
    self._all_rows = wp.array(np.arange(nrow).astype(np.int32), dtype=wp.int32, device=device)
    self._env0 = wp.zeros(1, dtype=wp.int32, device=device)
    self.dirty = wp.zeros(nworld, dtype=wp.int32, device=device)
    self._changed = wp.zeros(nworld, dtype=bool, device=device)
    self._reset = reset
    self._before_write = before_write

  def begin(self, material_offsets: Offsets | None, channel: str, env_ids: Rows | None) -> tuple[Rows, Rows] | None:
    """(rows, environments) of a write of `channel`: its rows of `material_offsets` (None: a full
    sync, every row) in environments `env_ids` (None: environment 0); None when there is nothing
    to write. Runs the `before_write` hook and clears the change flags."""
    rows = self._all_rows if material_offsets is None else material_offsets.get(channel)
    envs = self._env0 if env_ids is None else env_ids
    if rows is None or not self.width or not len(rows) or not len(envs):
      return None
    check_device(self.device, material_offsets=rows, env_ids=envs)
    if self._before_write is not None:
      self._before_write()
    self.dirty.zero_()
    return rows, envs

  def finish(self) -> None:
    """Resets the worlds the write changed (`dirty`)."""
    wp.launch(kernels.changed_worlds, len(self._changed), inputs=[self.dirty], outputs=[self._changed], device=self.device)
    self._reset(self._changed)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MaterialLayout:
  """Exact per-world material addresses and canonical color derivation, prepared before upload."""

  paths: tuple[tuple[str, ...], ...]
  scale: np.ndarray[[int], np.dtype[np.float32]]
  bias: np.ndarray[[int, 3], np.dtype[np.float32]]
  transmission_scale: np.ndarray[[int], np.dtype[np.float32]]
  transmission_bias: np.ndarray[[int, 3], np.dtype[np.float32]]
  # Volume glass (`pipeline/upload/optics.py`): its color is a static extinction a live glass color
  # does not reach (per-instance optics rows, not per-world material rows).
  volume: np.ndarray[[int], np.dtype[np.bool_]]

  @classmethod
  def prepare(cls, host: Scene, instance_paths: Sequence[Sequence[str]]) -> MaterialLayout:
    """Split material rows with distinct live addresses; keep textures and geometry shared.

    Mutates `host` in place: `host.materials` gains the split rows and `host.instances.material`
    points at them.
    """
    if any(len(row) != host.ninstance for row in instance_paths):
      raise ValueError("material addresses need one path per world and instance")
    indices = list(range(len(host.materials)))
    paths = [[""] * len(indices) for _ in instance_paths]
    groups: dict[tuple[int, tuple[str, ...]], int] = {}
    claimed: set[int] = set()
    ids = host.instances.material.copy()
    for instance, material in enumerate(ids):
      mid = int(material)
      if mid < 0:
        continue
      addresses = tuple(row[instance] for row in instance_paths)
      key = (mid, addresses)
      column = groups.get(key)
      if column is None:
        column = mid if mid not in claimed else len(indices)
        if column == len(indices):
          indices.append(mid)
          for row in paths:
            row.append("")
        claimed.add(mid)
        groups[key] = column
        for row, address in zip(paths, addresses, strict=True):
          row[column] = address
      ids[instance] = column
    if len(indices) != len(host.materials):
      host.materials = host.materials.take(np.asarray(indices, np.intp))
      host.instances.material[:] = ids
    volume = material_optics(host).volume
    scale, bias = base_color_affine(host.materials, host.material_extras, volume)
    ts, tb = base_color_affine(host.materials, host.material_extras, volume, field="transmission_color")
    return cls(
      paths=tuple(tuple(row) for row in paths),
      scale=scale.astype(np.float32),
      bias=bias.astype(np.float32),
      transmission_scale=ts.astype(np.float32),
      transmission_bias=tb.astype(np.float32),
      volume=volume,
    )


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PreparedColors:
  """One device table for every camera, allocated only when a live color channel is prepared."""

  table: ColorTable

  @classmethod
  def create[S](cls, scene: GpuScene[S], layout: MaterialLayout) -> PreparedColors:
    """Prepare canonical color storage once; each camera borrows the same retained arrays."""
    rows = host_copy(scene.tables["material.base_color"])
    values = np.broadcast_to(rows, (len(layout.paths), *rows.shape[1:]))
    return cls(table=wp.array(np.ascontiguousarray(values), dtype=wp.vec4, device=scene.device))


@final
class MaterialWriter:
  """Scatter composed Isaac color rows into canonical material storage and reset changed worlds."""

  def __init__[S](
    self,
    scene: GpuScene[S],
    *,
    layout: MaterialLayout,
    material_paths: Sequence[str],
    colors: wp.array[wp.vec3, [int]],
    prepared: PreparedColors | None,
    reset: Callable[[WorldMask], None],
    before_write: Callable[[], None] | None = None,
    input_names: Sequence[str] | None = None,
  ) -> None:
    """Map each composed material row to its exact world/material columns during preparation."""
    if colors.shape != (len(material_paths),) or colors.dtype != wp.vec3:
      raise ValueError("material colors must have one vec3 row per material path")
    names = ("diffuseColor",) * len(material_paths) if input_names is None else tuple(input_names)
    device = wp.get_device(scene.device)
    check_device(device, colors=colors)
    targets = MaterialTargets.create(
      layout=layout, paths=material_paths, names=names, accepted=_BASE_INPUTS | {"glass_color"}, report=scene.report
    )
    self._prepared = prepared
    self._colors = colors
    self._targets = ScatterTargets(
      targets, nrow=len(material_paths), nworld=len(layout.paths), device=device, reset=reset, before_write=before_write
    )
    target_columns = targets.columns
    glass = np.asarray([names[int(row)] == "glass_color" for row in targets.sources], bool)
    for row in np.unique(targets.sources[glass & layout.volume[target_columns]]):
      scene.report.approx("isaac.material.glass_color(volume glass: static extinction)", material_paths[int(row)])
    scale = np.where(glass, layout.transmission_scale[target_columns], layout.scale[target_columns])
    bias = np.where(glass[:, None], layout.transmission_bias[target_columns], layout.bias[target_columns])
    self._scale = wp.array(scale, dtype=float, device=device)
    self._bias = wp.array(bias, dtype=wp.vec3, device=device)

  def __call__(self, material_offsets: Offsets | None = None, env_ids: Rows | None = None) -> None:
    """Write offset+environment rows, or all rows, without readback, copies or pointer changes."""
    prepared, t = self._prepared, self._targets
    if prepared is None:
      return
    write = t.begin(material_offsets, "color", env_ids)
    if write is None:
      return
    rows, envs = write
    wp.launch(
      kernels.scatter_colors,
      (len(rows), len(envs), t.width),
      inputs=[self._colors, rows, envs, t.start, t.pairs, self._scale, self._bias],
      outputs=[prepared.table, t.dirty],
      device=t.device,
    )
    t.finish()

  def close(self) -> None:
    """Stop writing (idempotent)."""
    self._targets.width = 0
