"""A stand-in for the driver's OptiX: runs the OptiX backend's host logic on the CPU.

`install(monkeypatch)` puts a fake device context in place of OptiX's and a fake function table
in place of the driver's (`api.table`): every Warp kernel of the backend runs for real (on the
CPU device) and the build inputs and options are the binding's real ctypes structs, while the
OptiX calls (`accel_build`, `accel_compact`, `optixLaunch`) only record what they were asked to do
(`FakeContext.builds`, `FakeTable.launches`).
"""

from __future__ import annotations

import ctypes
import itertools
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pytest
import warp as wp

from raywarp.core.trace.optix import api
from raywarp.core.trace.optix.context import OptixObject


@dataclass
class Build:
  """One recorded `accel_build`: whether it updated (refit) and which build inputs it read."""

  update: bool
  inputs: ctypes.Array[api.BuildInput]
  handle: int


@dataclass
class FakeContext:
  """A device context recording acceleration-structure builds (handles count up from 1)."""

  builds: list[Build] = field(default_factory=list)
  _handles: itertools.count = field(default_factory=lambda: itertools.count(1))

  def accel_sizes(self, options: api.AccelBuildOptions, inputs: ctypes.Array[api.BuildInput]) -> api.AccelBufferSizes:
    """Sizes of a build: a few bytes each."""
    return api.AccelBufferSizes(outputSizeInBytes=64, tempSizeInBytes=8, tempUpdateSizeInBytes=8)

  def accel_build(
    self, stream: int, options: api.AccelBuildOptions, inputs: ctypes.Array[api.BuildInput], *args: object
  ) -> int:
    """Records the build; a rebuild gets a new handle, a refit keeps the structure's."""
    update = options.operation == api.BUILD_OPERATION_UPDATE
    handle = next(self._handles)
    self.builds.append(Build(update, inputs, handle))
    return handle

  def accel_compact(self, stream: int, handle: int, output: int, size: int) -> int:
    """No compaction: the same handle."""
    return handle

  def relocation_info(self, handle: int) -> np.ndarray:
    """Relocation information of a recorded build (zeros: never relocatable)."""
    return np.zeros(4, np.uint64)

  def relocation_compatible(self, info: np.ndarray) -> bool:
    """Recorded builds hold no bytes to relocate: every GAS cache entry is a miss."""
    return False


class FakeTable:
  """The driver's function table: a recording `optixLaunch` (success, nothing launched)."""

  def __init__(self) -> None:
    """An empty launch log."""
    self.launches: list[int] = []
    self.launch_params: list[int] = []

  def optixLaunch(  # OptiX's name
    self, pipeline: int, stream: int, params: int, size: int, sbt: object, width: int, height: int, depth: int
  ) -> int:
    """Records the launch width (`optixLaunch` takes at most 2^30 indices) and the device address
    of its launch parameters."""
    self.launches.append(width)
    self.launch_params.append(params)
    return api.SUCCESS


def pipeline_object() -> OptixObject:
  """A native pipeline stand-in: handle 0, nothing to release."""
  return OptixObject(0, "optixPipelineDestroy")


class FakeStream(wp.Stream):
  """Stream handle 0 (the CPU device has no streams): what the OptiX calls get. Its device is
  set by `install`."""

  def __init__(self) -> None:
    """Handle 0: no CUDA stream (`wp.Stream.__init__` would create one; `__del__` skips 0)."""
    self.cuda_stream = 0


STREAM: Final = FakeStream()


def install(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeTable, FakeContext]:
  """Installs the fake function table, device context and stream (synchronised: no-op); returns
  the first two."""
  from raywarp.core.trace.optix import accel

  table, ctx = FakeTable(), FakeContext()
  STREAM.device = wp.get_device("cpu")
  monkeypatch.setattr(api, "table", lambda: table)
  monkeypatch.setattr(accel, "device_context", lambda device: ctx)
  monkeypatch.setattr(wp, "get_stream", lambda device=None: STREAM)
  monkeypatch.setattr(wp, "synchronize_stream", lambda stream=None: None)
  return table, ctx
