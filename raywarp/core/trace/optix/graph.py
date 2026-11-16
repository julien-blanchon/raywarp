"""CUDA graphs of OptiX renders: captured Warp segments around eagerly enqueued OptiX calls.

`optixLaunch` cannot be stream-captured (OptiX 9.0 on RTX, 9.1 on GH200). Traced with Nsight
Systems, an eager launch is one `cuMemcpyDtoDAsync` (the launch parameters) plus three
`cuEventRecord`s, and the ray-tracing kernel is submitted by the driver outside the CUDA API, under
the memcpy's correlation id. Inside a capture the memcpy and the event records are captured, but the
kernel still runs immediately, at capture time, reading a parameter slot that the captured copy
never filled; the graph holds only the memcpy node. Stream capture mode (global, thread-local,
relaxed) makes no difference. Acceleration-structure builds are captured, but OptiX then fails on
its next eager build ("Failed to query command list event"), so they cannot be captured either.

No graph can therefore make the GPU wait for OptiX work issued by the host: a host node that
blocks the graph until another thread has enqueued it (the former design: `sem_post` /
`sem_wait` host nodes and a worker thread) deadlocks as soon as the thread that launches graphs
blocks inside the driver, which holds the context lock the OptiX calls need. `cuGraphLaunch`
blocks that way once about 30 launches of the graph are pending (back-to-back replays without a
synchronisation: the worker waits in `optixAccelBuild` for the lock, `cuGraphLaunch` for the GPU,
the GPU for the worker), as does a synchronous copy to pageable host memory (`array.numpy()`).
Gating pre-enqueued OptiX work on the GPU instead (`cuStreamWaitValue32`) makes
`cuCtxSynchronize` wait forever for the pending work.

`Renderer.capture` instead cuts the capture at every OptiX point of the captured work
(`Segments.cut`): the capture is ended there and begun again, and the OptiX calls (refits,
`optixLaunch`) are kept as a host callable. Launching replays the graph
segments and enqueues the callables between them, on the launching stream, in order: nothing
waits for the GPU, the GPU never waits for the host, no thread or host node is involved. The
callables run on the launching thread, so their errors raise from `RenderGraph.launch`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from types import TracebackType
from typing import Final, Self, final

import warp as wp

# Segments being recorded, by CUDA stream handle (`Renderer.capture`).
_recording: Final[dict[int, Segments]] = {}


def recording(stream: wp.Stream) -> Segments | None:
  """The `Segments` recorded on `stream`, or None if its capture is not a `Renderer.capture`."""
  return _recording.get(stream.cuda_stream)


@final
class Segments:
  """A CUDA graph cut into captured Warp segments and eager OptiX calls between them."""

  def __init__(self, stream: wp.Stream) -> None:
    """Segments captured on `stream` (none yet: the capture starts on entering the context)."""
    self.stream = stream
    self.parts: list[wp.Graph | Callable[[wp.Stream], None]] = []
    # Run before a replay's first segment (`cut`): why the replay is refused (an OptiX object
    # released since the capture), or None.
    self.checks: list[Callable[[], str | None]] = []

  def __enter__(self) -> Self:
    """Begins the first segment's capture."""
    wp.capture_begin(stream=self.stream)
    _recording[self.stream.cuda_stream] = self
    return self

  def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
    """Ends the last segment's capture (on an error, ends the capture and drops it)."""
    del _recording[self.stream.cuda_stream]
    if exc_type is None:
      self.parts.append(wp.capture_end(stream=self.stream))
    elif self.stream.is_capturing:
      # The capture already failed: its error propagates, not a second one from ending it.
      with contextlib.suppress(Exception):
        wp.capture_end(stream=self.stream)

  def cut(self, work: Callable[[wp.Stream], None], check: Callable[[], str | None] | None = None) -> None:
    """Ends the current segment; `work(stream)` runs between it and the next one at launch, and
    `check()` (why the replay is refused, or None) before the first segment."""
    if check is not None:
      self.checks.append(check)
    self.parts.append(wp.capture_end(stream=self.stream))
    self.parts.append(work)
    wp.capture_begin(stream=self.stream)

  @property
  def graphs(self) -> list[wp.Graph]:
    """The captured segments, in order."""
    return [p for p in self.parts if isinstance(p, wp.Graph)]

  def stale(self) -> str | None:
    """Why a replay would be refused (an OptiX object of a cut released or replaced since the
    capture), or None: it replays."""
    for check in self.checks:
      reason = check()
      if reason is not None:
        return reason
    return None

  def launch(self, stream: wp.Stream | None = None) -> None:
    """Replays the segments and the OptiX calls between them on `stream`; never synchronises.

    `stream` defaults to the current stream of the capture's device. Another stream is made the
    device's current one meanwhile (the OptiX calls' Warp work goes to the current stream): a
    process-wide change, so replay on the current stream when other threads use the device.

    Raises:
      RuntimeError: when the replay is refused (`stale`), before anything is enqueued.
    """
    current = wp.get_stream(self.stream.device)
    stream = stream if stream is not None else current
    reason = self.stale()
    if reason is not None:
      raise RuntimeError(reason)
    with contextlib.nullcontext() if stream is current else wp.ScopedStream(stream):
      for part in self.parts:
        if isinstance(part, wp.Graph):
          wp.capture_launch(part, stream=stream)
        else:
          part(stream)
