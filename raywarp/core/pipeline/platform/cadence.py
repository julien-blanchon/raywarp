"""Periodic work decided on the device: "every N frames" without a host counter.

It lets a captured CUDA graph replay periodic work (GI probe updates, BVH rebuilds) on schedule.
A `Cadence` counts frames in a device array and opens a gate (`due`, 1 on every N-th frame).
The gated work reads the gate on the device: a kernel tests it (GI probe updates,
`gi.update_every`; the software BVH rebuild, `trace/accel/bvh.py`,
`performance.rebuild_every`). A CUDA graph capturing a frame therefore replays the cadence: a
host counter would freeze at its captured value.
"""

from __future__ import annotations

from typing import final

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def tick(
  # In:
  every: int,
  # Out:
  frames: wp.array1d[int],
  gate: wp.array1d[int],
) -> None:
  """One more frame: the gate opens on every `every`-th one."""
  frames[0] = frames[0] + 1
  gate[0] = wp.where(frames[0] % every == 0, 1, 0)


@final
class Cadence:
  """A device frame counter and its gate.

  Attributes:
    frames: (1,) frames counted so far.
    due: (1,) 1 on the frames the periodic work runs, else 0.
  """

  def __init__(self, device: wp.Device) -> None:
    """No frame counted, gate closed (on `device`)."""
    self.frames: wp.array[int, [int]] = wp.zeros(1, dtype=int, device=device)
    self.due: wp.array[int, [int]] = wp.zeros(1, dtype=int, device=device)

  def tick(self, every: int) -> wp.array[int, [int]]:
    """Counts a frame; returns the gate (open on every `every`-th frame).

    Raises:
      ValueError: for `every` < 1 (the kernel would take the frame count modulo it).
    """
    if every < 1:
      raise ValueError(f"Cadence.tick: every must be at least 1 frame, got {every}")
    wp.launch(tick, dim=1, inputs=[every], outputs=[self.frames, self.due], device=self.frames.device)
    return self.due
