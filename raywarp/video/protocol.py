"""The recorder <-> encoder process protocol (`raywarp/video/worker.py`), over the process's stdin
/ stdout:
  in:  a 4-byte length + the pickled `Job`, then one 4-byte slot per frame, FINISH to end;
  out: READY (sessions open) or an error, then each finished slot, then DONE or an error.
A message is a 4-byte signed int: >= 0 a slot, READY / DONE, or ERROR followed by a 4-byte length
and the UTF-8 traceback.
"""

from __future__ import annotations

import dataclasses
import os
import struct
from pathlib import Path
from typing import Final

from raywarp.video.encoders import Encoding

READY: Final = -1
DONE: Final = -2
ERROR: Final = -3
FINISH: Final = -1
INT: Final = struct.Struct("<i")


@dataclasses.dataclass(frozen=True)
class Job:
  """What an encoder process encodes.

  Attributes:
    device: the ring's device ("cuda:0", "cpu").
    backend: "nvenc" or "x264".
    enc: encoder settings.
    frames: the ring's output frames this process owns; paths: their files.
    shape: the ring's shape (depth, frames, 3 H / 2, W).
    ring: the ring's CUDA IPC handle, or the path of its memory-mapped file (CPU device).
    events: the IPC handles of the per-slot ready events (CUDA).
    threads: libx264 threads per file.
  """

  device: str
  backend: str
  enc: Encoding
  frames: list[int]
  paths: list[Path]
  shape: tuple[int, int, int, int]
  ring: bytes | str
  events: list[bytes]
  threads: int


def send_int(fd: int, value: int) -> None:
  """Writes one message int to `fd`."""
  os.write(fd, INT.pack(value))


def read_exact(fd: int, n: int) -> bytes:
  """Exactly n bytes from `fd` (EOFError at the end of the stream)."""
  out = b""
  while len(out) < n:
    chunk = os.read(fd, n - len(out))
    if not chunk:
      raise EOFError
    out += chunk
  return out


def read_int(fd: int) -> int:
  """One message int from `fd`."""
  return INT.unpack(read_exact(fd, 4))[0]


def send_error(fd: int, text: str) -> None:
  """ERROR, then the length-prefixed text."""
  data = text.encode()
  os.write(fd, INT.pack(ERROR) + INT.pack(len(data)) + data)
