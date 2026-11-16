"""Newton arrays copied to the render device when the model lives on another one.

`Staging.get(name, array)` returns `array` itself on the render device, else a reused copy made
at most once per frame (`next_frame`; the pose kernel and the moved-scenery check read the same
copy). A CPU array bound for CUDA is uploaded by `upload` (the binding's `Binding.stage`: an
eager render, and before each graph replay, outside the graph): it waits for the previous
upload's event, copies the host array into a pinned buffer, enqueues the pinned -> device copy
on the render stream and records the event right after it. The next frame's host step waits for
that copy only, not for the frame's whole work, and a CUDA graph needs no copy of its own: it
reads the device copy, which each replay's upload refreshes before the launch (same stream).
`get` uploads itself when no `upload` ran this frame (e.g. a direct write, not under a capture).
The staged buffers are part of what a captured graph checks (`buffers`): a reallocation makes
it stale.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast, final

import warp as wp

if TYPE_CHECKING:
  from shape_extensions import IntVar


@final
@dataclasses.dataclass
class _Copy:
  """A staged array: the device copy; for a CPU source bound for CUDA, the pinned host buffer and
  the reused event of its last upload (`recorded`: once one was); the frame the device copy
  is for; whether an upload ran since the last `get`."""

  device: wp.array[object, [int]]
  pinned: wp.array[object, [int]] | None
  event: wp.Event | None
  recorded: bool = False
  frame: int = -1
  fresh: bool = False


@final
class Staging:
  """Copies of foreign-device arrays on `device`, one per name, refreshed once per frame."""

  def __init__(self, device: wp.Device) -> None:
    """No copy yet."""
    self.device = device
    self._copies: dict[str, _Copy] = {}
    self._frame = 0

  def next_frame(self) -> None:
    """Starts a frame: the next `get` of each name copies again (unless `upload` just did)."""
    self._frame += 1

  def upload[D, N: IntVar](self, name: str, array: wp.array[D, [N]] | None) -> None:
    """Uploads `array` (a CPU array bound for CUDA; nothing otherwise) through its pinned buffer,
    the event recorded right after the copy. Never under a capture."""
    if array is None or array.device == self.device:
      return
    staged = self._buffer(name, array)
    if staged.pinned is None or staged.event is None:
      return
    if staged.recorded:
      wp.synchronize_event(staged.event)  # the previous copy still reads the pinned buffer
    wp.copy(staged.pinned, array)
    wp.copy(staged.device, staged.pinned)
    wp.get_stream(self.device).record_event(staged.event)
    staged.recorded = staged.fresh = True

  def get[D, N: IntVar](self, name: str, array: wp.array[D, [N]] | None) -> wp.array[D, [N]] | None:
    """`array` on the render device (None stays None): the device copy, made once per frame. Under
    a capture a pinned upload is not recorded: each replay's `upload` refreshes the copy."""
    if array is None or array.device == self.device:
      return array
    staged = self._buffer(name, array)
    if staged.frame != self._frame:
      if staged.pinned is None:
        wp.copy(staged.device, array)
      elif not staged.fresh and not self.device.is_capturing:
        self.upload(name, array)  # a render without the binding's stage hook (a direct write)
      staged.fresh = False
      staged.frame = self._frame
    return cast("wp.array[D, [N]]", staged.device)  # a buffer like `array`

  def buffers(self) -> dict[str, wp.array[object, [int]]]:
    """The staged buffers by name (`staging.<name>.device`, `staging.<name>.pinned`): what a
    captured graph reads or each replay's upload writes, so it must find them unchanged."""
    out: dict[str, wp.array[object, [int]]] = {}
    for name, staged in self._copies.items():
      out[f"staging.{name}.device"] = staged.device
      if staged.pinned is not None:
        out[f"staging.{name}.pinned"] = staged.pinned
    return out

  def _buffer[D, N: IntVar](self, name: str, array: wp.array[D, [N]]) -> _Copy:
    """The reused buffers of `name` (reallocated when `array`'s shape or dtype changed)."""
    staged = self._copies.get(name)
    if staged is None or staged.device.shape != array.shape or staged.device.dtype != array.dtype:
      pinned = event = None
      if array.device is not None and array.device.is_cpu and self.device.is_cuda:
        pinned = wp.empty(array.shape, dtype=array.dtype, device="cpu", pinned=True)
        event = wp.Event(device=self.device)
      device = wp.empty(array.shape, dtype=array.dtype, device=self.device)
      staged = self._copies[name] = _Copy(device=device, pinned=pinned, event=event)
    return staged
