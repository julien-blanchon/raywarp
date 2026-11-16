"""`recorder`: render-loop-friendly MP4 recording of raywarp's `rgb` output (docs/recording.md).

The render thread only replays one small recorded kernel launch per frame (RGBA -> NV12 into a
ring slot, on the render stream), records an event and sends the slot number to the encoder
processes (`raywarp/video/worker.py`: they map the ring and the events through CUDA IPC, so
encoding never takes the render process's GIL). The render thread blocks only when every ring
slot is still being encoded (the encoders are slower than the renderer): `RecordStats.blocked_s`
says how long.

Pipeline per `write`:
  render stream:   ... render -> rgba_to_nv12 (ring[slot]) -> event ready[slot] -> next render ...
  encoder process: wait ready[slot] -> NVENC sessions read ring[slot] (or: copy stream -> pinned
                   host -> ffmpeg libx264 pipes) -> "slot done" back to the recorder
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import warp as wp

from raywarp.video import colour, protocol
from raywarp.video.encoders import Codec, Encoding, Ring, nvenc_status

if TYPE_CHECKING:
  from shape_extensions import IntVar

# Longest wait for an encoder process's message (start-up imports included) before it is killed.
TIMEOUT_S: Final = 120.0
type Layout = Literal["streams", "mosaic"]
type Backend = Literal["auto", "nvenc", "x264"]
# raywarp's default `rgb` output: (nworld, ncam, H, W) RGBA8, sRGB.
type Rgb[NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar] = wp.array[wp.vec4ub, [NWorld, NCam, H, W]]


@dataclasses.dataclass
class RecordStats:
  """What a recording cost.

  Attributes:
    frames: writes.
    blocked_s: render-thread seconds spent waiting for a free ring slot (0: never slowed down).
    write_s: render-thread seconds inside `write` (blocked_s included).
    close_s: seconds `close` waited for the encoders to drain and the files to finish.
    backend: "nvenc" or "x264".
    reason: why NVENC was not used ("" when it was).
  """

  frames: int = 0
  blocked_s: float = 0.0
  write_s: float = 0.0
  close_s: float = 0.0
  backend: str = ""
  reason: str = ""


class _Worker:
  """One encoder process (`raywarp/video/worker.py`) and its pipes."""

  def __init__(self, job: protocol.Job) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    self.proc = subprocess.Popen(
      [sys.executable, "-m", "raywarp.video.worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env
    )
    assert self.proc.stdin is not None and self.proc.stdout is not None
    self.fd_in, self.fd_out = self.proc.stdin.fileno(), self.proc.stdout.fileno()
    data = pickle.dumps(job)
    os.write(self.fd_in, protocol.INT.pack(len(data)) + data)

  def send(self, value: int) -> None:
    """A slot to encode, or FINISH."""
    protocol.send_int(self.fd_in, value)

  def receive(self) -> int:
    """The next message (blocking, at most TIMEOUT_S); raises the process's error."""
    if not select.select([self.fd_out], [], [], TIMEOUT_S)[0]:
      self.proc.kill()
      raise RuntimeError(f"video encoder process sent nothing for {TIMEOUT_S} s (killed)")
    try:
      value = protocol.read_int(self.fd_out)
    except EOFError:
      raise RuntimeError(f"video encoder process exited with {self.proc.wait()}") from None
    if value == protocol.ERROR:
      text = protocol.read_exact(self.fd_out, protocol.read_int(self.fd_out)).decode()
      raise RuntimeError(f"video encoder process failed:\n{text}")
    return value

  def close(self) -> None:
    """Closes the pipes and waits for the process."""
    for f in (self.proc.stdin, self.proc.stdout):
      if f is not None:
        f.close()
    self.proc.wait()


class Recorder:
  """MP4 files from successive `rgb` outputs (create with `recorder`)."""

  def __init__(
    self,
    paths: list[Path],
    tiling: colour.Tiling,
    enc: Encoding,
    device: wp.Device,
    backend: Backend,
    depth: int,
    workers: int,
  ) -> None:
    """Allocates the ring and starts `workers` encoder processes (`recorder`'s arguments; at most
    one per output frame)."""
    h, w = tiling.frame_size
    self.tiling, self.device, self.stats = tiling, device, RecordStats()
    self.stats.backend, self.stats.reason = choose(device, enc.codec, backend)
    shape = (depth, tiling.frames, h * 3 // 2, w)
    self.ring, handle, self.mapped = _ring(shape, device)
    with wp.ScopedDevice(device):
      self.ready = [wp.Event(device, interprocess=True) for _ in range(depth)] if device.is_cuda else []
      events = [e.ipc_handle() for e in self.ready]
    self.launches: dict[tuple[int, str], list[wp.Launch]] = {}  # (address, strides) -> slots' launches
    self.free = list(range(depth))
    self.pending = [0] * depth
    threads = max(1, (os.cpu_count() or 1) // len(paths))
    self.workers: list[_Worker] = []
    for k in range(workers):
      frames = list(range(k, tiling.frames, workers))
      job = protocol.Job(
        str(device), self.stats.backend, enc, frames, [paths[f] for f in frames], shape, handle,
        events, threads,
      )  # fmt: skip
      self.workers.append(_Worker(job))
    for wk in self.workers:
      if wk.receive() != protocol.READY:
        raise RuntimeError("video encoder process did not start")

  def write[NWorld: IntVar, NCam: IntVar, H: IntVar, W: IntVar](self, rgb: Rgb[NWorld, NCam, H, W]) -> None:
    """Queues one frame of every world and camera (call after the render, on its stream; returns
    at once unless every ring slot is still being encoded)."""
    t0 = time.perf_counter()
    nworld, ncam, h, w = rgb.shape
    if (nworld * ncam, h, w) != (self.tiling.count, self.tiling.height, self.tiling.width):
      raise ValueError(f"rgb is {tuple(rgb.shape)}; the recorder was made for {self.tiling}")
    self._collect(block=False)
    while not self.free:
      self._collect(block=True)
    slot = self.free.pop()
    self.stats.blocked_s += time.perf_counter() - t0
    stream = wp.get_stream(self.device) if self.device.is_cuda else None
    self._launches(rgb)[slot].launch(stream)
    if stream is not None:
      stream.record_event(self.ready[slot])
    self.pending[slot] = len(self.workers)
    for wk in self.workers:
      wk.send(slot)
    self.stats.frames += 1
    self.stats.write_s += time.perf_counter() - t0

  def _collect(self, block: bool) -> None:
    """Frees the slots the encoder processes finished (waits for one message when `block`)."""
    fds = {wk.fd_out: wk for wk in self.workers}
    ready, _, _ = select.select(list(fds), [], [], None if block else 0)
    for fd in ready:
      slot = fds[fd].receive()
      self.pending[slot] -= 1
      if self.pending[slot] == 0:
        self.free.append(slot)

  def _launches(self, rgb: Rgb) -> list[wp.Launch]:
    """The recorded conversion of `rgb` into each ring slot (recorded on first sight of its
    address: a renderer writes the same output arrays every frame)."""
    key = (int(rgb.ptr or 0), str(rgb.strides))
    if key not in self.launches:
      images = rgb.reshape((self.tiling.count, self.tiling.height, self.tiling.width))
      self.launches[key] = [colour.record(images, self.tiling, self.ring[k]) for k in range(self.ring.shape[0])]
    return self.launches[key]

  def close(self) -> RecordStats:
    """Lets the encoder processes drain and finish the files; raises when one failed."""
    t0 = time.perf_counter()
    try:
      for wk in self.workers:
        wk.send(protocol.FINISH)
      for wk in self.workers:
        while wk.receive() != protocol.DONE:
          pass
    finally:
      for wk in self.workers:
        wk.close()
      if self.mapped is not None:
        self.mapped.unlink(missing_ok=True)
    self.stats.close_s = time.perf_counter() - t0
    return self.stats

  def __enter__(self) -> Recorder:
    """The recorder itself."""
    return self

  def __exit__(self, *exc: object) -> None:
    """Closes the recorder."""
    self.close()


def choose(device: wp.Device, codec: Codec, backend: Backend) -> tuple[str, str]:
  """(back end, why not NVENC: "" when NVENC is used)."""
  reason = nvenc_status(device, codec) if backend != "x264" else "x264 requested"
  if backend == "nvenc" and reason is not None:
    raise RuntimeError(f"backend='nvenc' but {reason}")
  return ("nvenc", "") if reason is None else ("x264", reason)


def _ring(shape: tuple[int, int, int, int], device: wp.Device) -> tuple[Ring, bytes | str, Path | None]:
  """The NV12 ring, how an encoder process opens it (CUDA IPC handle: a plain allocation, as
  stream-ordered pool memory cannot be shared; or a memory-mapped file), and that file."""
  if device.is_cuda:
    # The device's context must be current: Warp's IPC calls only log a CUDA error otherwise.
    with wp.ScopedDevice(device), wp.ScopedMempool(device, False):
      ring = wp.empty(shape, dtype=wp.uint8, device=device)
      handle = ring.ipc_handle()
    if handle == bytes(len(handle)):
      raise RuntimeError(f"no CUDA IPC handle for the video ring on {device}")
    return ring, handle, None
  tmp = Path("/dev/shm" if Path("/dev/shm").is_dir() else tempfile.gettempdir())
  fd, name = tempfile.mkstemp(prefix="wpbr_video_", dir=tmp)
  os.close(fd)
  host = np.memmap(name, np.uint8, "w+", shape=shape)
  return wp.array(host, dtype=wp.uint8, device="cpu", copy=False), name, Path(name)


def stream_paths(path: str | Path, nworld: int, ncam: int) -> list[Path]:
  """One path per (world, camera), world-major: `path` formatted with `world`, `cam` and `index`."""
  paths = [Path(str(path).format(world=i // ncam, cam=i % ncam, index=i)) for i in range(nworld * ncam)]
  if len(set(paths)) != len(paths):
    raise ValueError(f"{path!r} must name each stream: use {{world}}, {{cam}} or {{index}}")
  return paths


def recorder(
  path: str | Path,
  shape: tuple[int, int, int, int],
  *,
  fps: int = 30,
  layout: Layout = "streams",
  codec: Codec = "h264",
  backend: Backend = "auto",
  cols: int = 0,
  depth: int = 6,
  workers: int = 4,
  device: str | None = None,
  **encoding: float | str,
) -> Recorder:
  """A Recorder of `rgb` outputs of `shape` (nworld, ncam, H, W).

  Args:
    path: the MP4 file; with layout "streams" and several images, a template naming each stream
      (`"out/world{world:03d}_cam{cam}.mp4"`; `{index}` = world * ncam + cam).
    shape: the `rgb` output's (nworld, ncam, H, W).
    fps: frame rate of the files.
    layout: "streams" (one file per world and camera) or "mosaic" (all images tiled on one
      frame).
    cols: a mosaic's images per row (0: a square grid).
    codec: "h264" or "hevc" (mosaics wider than 4096 px need hevc on NVENC).
    backend: "nvenc" (GPU encoder; raises without one), "x264" (CPU encoder processes) or "auto":
      NVENC when the device has one and PyNvVideoCodec imports, else x264 (`RecordStats.reason`).
    depth: ring slots (frames in flight between the renderer and the encoders).
    workers: encoder processes (each owns a fixed subset of the output files).
    device: the device of the `rgb` arrays (None: Warp's default device).
    encoding: further `Encoding` fields (bits_per_pixel, nvenc_preset, x264_preset, gop_seconds).
  """
  nworld, ncam, h, w = shape
  count = nworld * ncam
  if layout == "mosaic":
    tiling, paths = colour.Tiling.mosaic(count, h, w, cols), [Path(path)]
  else:
    tiling, paths = colour.Tiling(count, h, w), stream_paths(path, nworld, ncam)
  for p in paths:
    p.parent.mkdir(parents=True, exist_ok=True)
  enc = Encoding(fps=fps, codec=codec, **encoding)  # pyrefly: ignore[bad-argument-type]
  return Recorder(paths, tiling, enc, wp.get_device(device), backend, depth, max(1, min(workers, tiling.frames)))
