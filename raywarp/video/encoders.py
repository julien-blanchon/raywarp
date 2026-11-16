"""The two encoder back ends of `Recorder`, both fed NV12 frames from the device ring.

- `NvencEncoder`: the GPU's NVENC through PyNvVideoCodec (optional dependency): it reads the NV12
  frames in device memory (CUDA array interface), so no raw frame crosses PCIe; the compressed
  bitstream (a few kB per frame) goes to an `ffmpeg -c:v copy` muxer process per output file,
  which also writes the colour tags into the stream headers (`*_metadata` bitstream filter).
- `X264Encoder`: no NVENC (A100 / H100 / CPU device): the NV12 frames are copied on their own
  stream into pinned host buffers (1.5 bytes per pixel, not RGBA's 4), then piped to one ffmpeg
  libx264 / libx265 process per output file (a process pool: encoding runs on every CPU core,
  outside the Python process).

Both run in the Recorder's encoder processes (`raywarp/video/worker.py`): each process owns a fixed
subset of the output frames and maps the ring and its events through CUDA IPC.
"""

from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

import warp as wp

from raywarp.core.pipeline.platform.device import host_copy

if TYPE_CHECKING:
  from shape_extensions import IntVar

type Codec = Literal["h264", "hevc"]
# NV12 ring: (depth, frames, 3 H / 2, W) uint8.
type Ring[D: IntVar, F: IntVar, H2: IntVar, W: IntVar] = wp.array[wp.uint8, [D, F, H2, W]]

# BT.709 code points (ITU-T H.273): colour_primaries, transfer_characteristics, matrix_coefficients.
BT709: Final = 1
# Chroma sample location type 1: center (the 2x2 mean of `colour.rgba_to_nv12`).
CHROMA_CENTER: Final = 1


@dataclasses.dataclass(frozen=True)
class Encoding:
  """Encoder settings shared by both back ends.

  Attributes:
    fps: frame rate of the files.
    codec: "h264" (plays everywhere; NVENC up to 4096 px a side) or "hevc" (up to 8192: mosaics).
    bits_per_pixel: target bitrate per pixel and frame (bitrate = bpp x W x H x fps); 0.1-0.2
      keeps rendered content visually lossless at 512-1024 px.
    nvenc_preset: NVENC preset, from the fastest ("P1") to the best ("P7").
    x264_preset: libx264 / libx265 preset of the CPU path ("ultrafast" ... "veryslow").
    gop_seconds: key frame interval (seek granularity).
  """

  fps: int = 30
  codec: Codec = "h264"
  bits_per_pixel: float = 0.15
  nvenc_preset: str = "P3"
  x264_preset: str = "veryfast"
  gop_seconds: float = 2.0

  def bitrate(self, height: int, width: int) -> int:
    """Target bits per second of one (height, width) stream."""
    return round(self.bits_per_pixel * height * width * self.fps)

  @property
  def gop(self) -> int:
    """Frames between key frames."""
    return max(1, round(self.gop_seconds * self.fps))


class Encoder(Protocol):
  """A back end inside an encoder process: it owns some output frames of the ring."""

  def encode(self, slot: int, ready: wp.Event | None) -> None:
    """Encodes its frames of `slot` once `ready` fired; returns when the slot may be reused."""
    ...

  def close(self) -> None:
    """Flushes and finalizes its files."""
    ...


def _run(cmd: list[str]) -> subprocess.Popen[bytes]:
  # stdout: the encoder process's own stdout is its reply channel.
  return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)


def _finish(procs: Sequence[subprocess.Popen[bytes]]) -> None:
  for p in procs:
    if p.stdin is not None:
      p.stdin.close()
  failed = [p.args for p in procs if p.wait() != 0]
  if failed:
    raise RuntimeError(f"ffmpeg failed: {failed[0]}")


def mux_command(path: Path, enc: Encoding) -> list[str]:
  """Ffmpeg reading an Annex B elementary stream on stdin, writing `path` (no re-encode) with
  BT.709 limited-range tags in the stream headers."""
  tags = (
    f"colour_primaries={BT709}:transfer_characteristics={BT709}:matrix_coefficients={BT709}"
    f":video_full_range_flag=0:chroma_sample_loc_type={CHROMA_CENTER}"
  )
  cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", enc.codec, "-framerate", str(enc.fps), "-i", "pipe:0"]
  cmd += ["-c:v", "copy", "-bsf:v", f"{enc.codec}_metadata={tags}", "-movflags", "+faststart"]
  return [*cmd, *(["-tag:v", "hvc1"] if enc.codec == "hevc" else []), str(path)]


def x264_command(path: Path, enc: Encoding, size: tuple[int, int], threads: int) -> list[str]:
  """Ffmpeg encoding raw NV12 frames of `size` (height, width) from stdin with libx264 / libx265."""
  h, w = size
  rate = enc.bitrate(h, w)
  cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "nv12", "-s", f"{w}x{h}"]
  cmd += ["-framerate", str(enc.fps), "-i", "pipe:0", "-c:v", "libx264" if enc.codec == "h264" else "libx265"]
  cmd += ["-preset", enc.x264_preset, "-b:v", str(rate), "-maxrate", str(2 * rate), "-bufsize", str(2 * rate)]
  cmd += ["-g", str(enc.gop), "-bf", "0", "-threads", str(threads)]
  cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
  cmd += [f"-{'x264' if enc.codec == 'h264' else 'x265'}-params", f"chromaloc={CHROMA_CENTER}", "-movflags", "+faststart"]
  return [*cmd, *(["-tag:v", "hvc1"] if enc.codec == "hevc" else []), str(path)]


def nvenc_status(device: wp.Device, codec: Codec) -> str | None:
  """None when NVENC can encode `codec` on `device`, else why not."""
  if not device.is_cuda:
    return "not a CUDA device"
  try:
    import PyNvVideoCodec as nvc
  except ImportError as e:
    return f"PyNvVideoCodec not importable ({e})"
  except RuntimeError as e:  # its import raises when the driver's libnvidia-encode is missing
    return f"PyNvVideoCodec: {str(e).splitlines()[0]}"
  try:
    caps = nvc.GetEncoderCaps(codec=codec)  # the first GPU's (multi-GPU: set CUDA_VISIBLE_DEVICES)
  except Exception as e:  # noqa: BLE001  (the binding raises its own types: no NVENC on this GPU)
    return f"no NVENC {codec} on {device.name} ({e})"
  if not caps or int(caps.get("num_encoder_engines", 1)) < 1:
    return f"no NVENC {codec} on {device.name}"
  return None


def _bitstream(packets: bytes | bytearray | memoryview | list[dict[str, bytes | memoryview]]) -> bytes:
  """The bytes of an Encode / EndEncode result: a buffer (PyNvVideoCodec < 2.1) or a list of
  packets with a "data" buffer."""
  if isinstance(packets, list):
    return b"".join(bytes(p["data"]) for p in packets)
  return bytes(packets)


class _Plane:
  """One plane of an NV12 frame in device memory (CUDA array interface v3)."""

  def __init__(self, ptr: int, shape: tuple[int, int, int], strides: tuple[int, int, int]) -> None:
    self.__cuda_array_interface__ = {"shape": shape, "strides": strides, "data": (ptr, False), "typestr": "|u1", "version": 3}


class Nv12Frame:
  """An NV12 frame of the ring as PyNvVideoCodec's device input: `cuda()` lists the Y plane
  (H, W, 1) and the interleaved CbCr plane (H / 2, W / 2, 2), both of pitch W."""

  def __init__(self, ptr: int, height: int, width: int) -> None:
    """The frame at device address `ptr` of an (height, width) image."""
    self.planes = [
      _Plane(ptr, (height, width, 1), (width, 1, 1)),
      _Plane(ptr + height * width, (height // 2, width // 2, 2), (width, 2, 1)),
    ]

  def cuda(self) -> list[_Plane]:
    """The planes."""
    return self.planes


class NvencEncoder:
  """NVENC sessions (one per output frame of `frames`) reading the device ring; `ffmpeg -c copy`
  muxers."""

  def __init__(self, ring: Ring, frames: Sequence[int], paths: Sequence[Path], enc: Encoding) -> None:
    """Opens a session and a muxer per frame of `frames` (written to `paths`)."""
    import PyNvVideoCodec as nvc

    device = ring.device
    h, w = ring.shape[2] * 2 // 3, ring.shape[3]
    rate = enc.bitrate(h, w)
    config = {
      "codec": enc.codec,
      "preset": enc.nvenc_preset,
      "tuning_info": "high_quality",
      "rc": "vbr",
      "bitrate": rate,
      "maxbitrate": 2 * rate,
      "vbvbufsize": 2 * rate,
      "fps": enc.fps,
      "gop": enc.gop,
      "idrperiod": enc.gop,
      "bf": 0,
      "repeatspspps": 1,
    }
    self.stream = wp.Stream(device)
    self.sessions = []
    for path in paths:
      try:
        session = nvc.CreateEncoder(
          w, h, "NV12", False, cudacontext=device.context, cudastream=self.stream.cuda_stream, **config
        )
      except Exception as e:
        raise RuntimeError(
          f"NVENC session for {path} failed ({e}). GeForce drivers allow a few concurrent sessions per "
          "system (8 on recent drivers); record a mosaic (layout='mosaic') or fewer streams."
        ) from e
      self.sessions.append(session)
    self.muxers = [_run(mux_command(p, enc)) for p in paths]
    frame_bytes = ring.shape[2] * w
    base = int(ring.ptr or 0)  # an allocated ring's device address
    self.inputs = [
      [Nv12Frame(base + (k * ring.shape[1] + f) * frame_bytes, h, w) for f in frames] for k in range(ring.shape[0])
    ]

  def encode(self, slot: int, ready: wp.Event | None) -> None:
    """Each session encodes its frame of the slot; the sessions' copy stream is drained before the
    slot is released."""
    if ready is not None:
      wp.synchronize_event(ready)
    for i, session in enumerate(self.sessions):
      self._send(i, _bitstream(session.Encode(self.inputs[slot][i])))
    wp.synchronize_stream(self.stream)

  def _send(self, i: int, data: bytes) -> None:
    stdin = self.muxers[i].stdin
    if data and stdin is not None:
      stdin.write(data)

  def close(self) -> None:
    """Flushes every session (EndEncode) into its muxer and waits for the files."""
    for i, session in enumerate(self.sessions):
      self._send(i, _bitstream(session.EndEncode()))
    self.sessions.clear()
    _finish(self.muxers)


class X264Encoder:
  """Device ring -> pinned host buffers on a copy stream, then one libx264 / libx265 process per
  output frame of `frames` (a CPU-device ring is host memory already: no copy)."""

  def __init__(self, ring: Ring, frames: Sequence[int], paths: Sequence[Path], enc: Encoding, threads: int) -> None:
    """Allocates the pinned buffers and starts an encoder process per frame of `frames` (`threads`
    encoder threads each)."""
    self.ring, self.frames = ring, list(frames)
    cuda = ring.device.is_cuda
    shape = (ring.shape[0], len(self.frames), ring.shape[2], ring.shape[3])
    self.host = wp.empty(shape, dtype=wp.uint8, device="cpu", pinned=True) if cuda else None
    self.host_np = host_copy(self.host) if self.host is not None else host_copy(ring)
    self.copy_stream = wp.Stream(ring.device) if cuda else None
    size = (ring.shape[2] * 2 // 3, ring.shape[3])
    self.procs = [_run(x264_command(p, enc, size, threads)) for p in paths]

  def encode(self, slot: int, ready: wp.Event | None) -> None:
    """Copies the slot's frames to pinned memory after `ready` (copy stream); pipes them to
    ffmpeg."""
    rows = range(len(self.frames))
    if self.host is not None and self.copy_stream is not None and ready is not None:
      self.copy_stream.wait_event(ready)
      for i, f in enumerate(self.frames):
        wp.copy(self.host[slot, i], self.ring[slot, f], stream=self.copy_stream)
      wp.synchronize_stream(self.copy_stream)
    else:
      rows = self.frames
    for i, r in enumerate(rows):
      stdin = self.procs[i].stdin
      assert stdin is not None
      stdin.write(memoryview(self.host_np[slot, r]))

  def close(self) -> None:
    """Closes the pipes and waits for the files."""
    _finish(self.procs)
