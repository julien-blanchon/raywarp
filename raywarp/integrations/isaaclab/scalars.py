"""Prepared scalar material channels with explicit authored-input semantics."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, Literal, final

import numpy as np
import warp as wp

from raywarp._base.types import TexRole
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.integrations.isaaclab import kernels
from raywarp.integrations.isaaclab.materials import (
  MaterialLayout,
  MaterialTargets,
  MaterialWriter,
  Offsets,
  Rows,
  ScatterTargets,
  WorldMask,
)
from raywarp.integrations.isaaclab.session import check_device

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene


type ScalarTable = wp.array[float, [int, int]]
type ScalarField = Literal["material.specular_roughness", "material.base_metalness", "material.specular_weight"]


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ScalarChannel:
  """One supported Isaac scalar channel and the canonical renderer field it controls."""

  channel: str
  field: ScalarField
  inputs: frozenset[str]
  multiplier: float = 1.0
  roughness: bool = False


SCALAR_CHANNELS: Final = {
  "roughness": ScalarChannel(
    channel="roughness",
    field="material.specular_roughness",
    inputs=frozenset({"roughness", "reflection_roughness_constant", "frosting_roughness"}),
    roughness=True,
  ),
  "metallic": ScalarChannel(
    channel="metallic", field="material.base_metalness", inputs=frozenset({"metallic", "metallic_constant"})
  ),
  "specular": ScalarChannel(
    channel="specular", field="material.specular_weight", inputs=frozenset({"specular_level"}), multiplier=2.0
  ),
}


def scalar_table[S](scene: GpuScene[S], nworld: int, spec: ScalarChannel) -> ScalarTable:
  """Allocate one shared declared table, starting at canonical user values for derived fields."""
  if spec.field == "material.specular_weight":
    rows = scene.derive_inputs.values[spec.field][None]
  else:
    rows = host_copy(scene.tables[spec.field])
  values = np.broadcast_to(rows, (nworld, rows.shape[1]))
  return wp.array(np.ascontiguousarray(values), dtype=float, device=scene.device)


@final
class ScalarWriter:
  """Scatter one scalar channel into shared material storage, resetting changed worlds only."""

  def __init__[S](
    self,
    scene: GpuScene[S],
    *,
    layout: MaterialLayout,
    spec: ScalarChannel,
    material_paths: Sequence[str],
    input_names: Sequence[str],
    values: wp.array[float, [int]],
    table: ScalarTable,
    reset: Callable[[WorldMask], None],
    before_write: Callable[[], None] | None = None,
  ) -> None:
    """Retain composed input storage and build exact per-world targets once."""
    if values.shape != (len(material_paths),) or values.dtype != wp.float32:
      raise ValueError("scalar material values need one float32 row per material path")
    self._device = wp.get_device(scene.device)
    check_device(self._device, values=values, table=table)
    blocked = None
    if spec.channel in ("roughness", "metallic"):
      # Authored constants were blended into prepared maps; assigning a runtime factor would
      # change even an unchanged input. Omit until the original affine map provenance exists.
      textures = host_copy(scene.tables["material.texture"])
      role = TexRole.ROUGHNESS if spec.roughness else TexRole.METALLIC
      blocked = np.any((textures[:, :, role] >= 0) | (textures[:, :, TexRole.ORM] >= 0), axis=0)
    targets = MaterialTargets.create(
      layout=layout, paths=material_paths, names=input_names, accepted=spec.inputs, report=scene.report, blocked=blocked
    )
    self._spec, self._table, self._values = spec, table, values
    self._targets = ScatterTargets(
      targets, nrow=len(material_paths), nworld=len(layout.paths), device=self._device, reset=reset, before_write=before_write
    )
    inputs = scene.derive_inputs
    # Static coat inputs remain authored; no live coat channel is claimed by this adapter.
    weights = inputs.values["material.coat_weight"] if spec.roughness else np.empty(0, np.float32)
    roughness = inputs.values["material.coat_roughness"] if spec.roughness else np.empty(0, np.float32)
    enabled = inputs.roughenable & ~inputs.gltf_coat if spec.roughness else np.empty(0, bool)
    self._coat_weight = wp.array(weights, dtype=float, device=self._device)
    self._coat_roughness = wp.array(roughness, dtype=float, device=self._device)
    self._roughenable = wp.array(enabled, dtype=bool, device=self._device)

  def __call__(self, material_offsets: Offsets | None = None, env_ids: Rows | None = None) -> None:
    """Write selected rows without readback or replacement storage."""
    t = self._targets
    write = t.begin(material_offsets, self._spec.channel, env_ids)
    if write is None:
      return
    rows, envs = write
    wp.launch(
      kernels.scatter_scalars,
      (len(rows), len(envs), t.width),
      inputs=[
        self._values,
        rows,
        envs,
        t.start,
        t.pairs,
        self._spec.multiplier,
        self._spec.roughness,
        self._coat_weight,
        self._coat_roughness,
        self._roughenable,
      ],
      outputs=[self._table, t.dirty],
      device=self._device,
    )
    t.finish()

  def close(self) -> None:
    """Stop writes (idempotent)."""
    self._targets.width = 0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MaterialWriters:
  """Isaac's composed channel writers; each consumes only its supplied offset rows."""

  channels: tuple[MaterialWriter | ScalarWriter, ...]

  def __call__(self, material_offsets: Offsets | None = None, env_ids: Rows | None = None) -> None:
    """Dispatch only the channels named in this update, or all for a full sync."""
    for channel in self.channels:
      channel(material_offsets, env_ids)

  def close(self) -> None:
    """Stop all channel writers (idempotent)."""
    for channel in self.channels:
      channel.close()
