"""The GI's re-bake log (`Renderer.debug.gi_rebakes`): every bake of the volume with its reason, the
probe updates `Renderer.update_gi` requested, and per world the re-lights of its tint
(`relight.Relight`: a world whose GI inputs changed). A frame that changes nothing GI reads
(poses of moving instances, dynamic lights, cameras) adds nothing to it.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy

wp.set_module_options({"enable_backward": False})


class Reason(enum.StrEnum):
  """Why the volume was (re-)baked."""

  FIRST = "first bake"
  SETTINGS = "a gi.* setting or the probes' Features changed"
  REQUESTED = "rebuild_gi() was called"
  NEW_VARIANT = "a world's GI inputs match no baked group (new per-world variant: re-baked relit)"
  SHARED = "a GI input every world shares changed (a light, material or environment of the scene)"
  PERIODIC = "periodic updates switched on after a frozen bake (re-placed)"


@final
@dataclasses.dataclass(frozen=True, slots=True)
class GIRebakes:
  """A snapshot of the re-bake log (`Renderer.debug.gi_rebakes`).

  Attributes:
    bakes: the reason of each bake of the volume, in order (`Reason` values; " (cache)" when it
      was restored from the bake cache).
    updates: probe updates `Renderer.update_gi` requested after the bakes (the periodic updates
      of gi.update_every are decided on the device and not counted).
    relights: (nworld,) per world, the re-lights of its tint (a relit volume: its GI inputs
      changed); zeros otherwise.
  """

  bakes: tuple[str, ...]
  updates: int
  relights: np.ndarray[[int], np.dtype[np.int64]]


@wp.kernel
def count_relights(
  # In:
  changed: wp.array1d[wp.int32],
  # Out:
  counts: wp.array1d[wp.int32],
):
  """Counts a re-light of world w's tint when its inputs changed."""
  w = wp.tid()
  if changed[w] != 0:
    counts[w] = counts[w] + 1


@final
class RebakeLog:
  """The re-bake log of a Renderer's GI (see the module doc)."""

  def __init__(self) -> None:
    """Nothing baked yet."""
    self.bakes: list[str] = []
    self.updates = 0
    self.relights: wp.array[wp.int32, [int]] | None = None

  def bake(self, reason: Reason, *, cached: bool) -> None:
    """Records a bake of the volume."""
    self.bakes.append(str(reason) + (" (cache)" if cached else ""))

  def counter(self, nworld: int, device: wp.Device) -> wp.array[wp.int32, [int]]:
    """The (nworld,) device counter of the worlds' re-lights (kept across re-bakes)."""
    if self.relights is None or self.relights.shape[0] != nworld:
      self.relights = wp.zeros(nworld, dtype=wp.int32, device=device)
    return self.relights

  def snapshot(self, nworld: int) -> GIRebakes:
    """The log now (a host read of the re-light counter)."""
    counts = host_copy(self.relights) if self.relights is not None else np.zeros(nworld, np.int32)
    relights: np.ndarray[[int], np.dtype[np.int64]] = np.zeros(len(counts), np.int64)
    relights[:] = counts
    return GIRebakes(tuple(self.bakes), self.updates, relights)
