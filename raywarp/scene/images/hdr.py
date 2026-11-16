"""Radiance RGBE (`.hdr`) images: header and run-length decoding to linear float32, numpy only."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp.scene.images import files

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar


# Run-length scanlines: a count byte above _RUN is a run of count - _RUN copies; only widths in
# [_RLE_MIN_WIDTH, _RLE_MAX_WIDTH] are run-length encoded (Radiance's rule).
_RUN: Final = 128
_RLE_MIN_WIDTH: Final = 8
_RLE_MAX_WIDTH: Final = 0x7FFF
# Bytes read for the header (64 KiB [bytes]).
_HEADER_BYTES: Final = 65536


def _parse_header(data: bytes, path: str) -> tuple[int, int, int]:
  """(height, width, offset of the pixel data) of the Radiance .hdr bytes `data` (from `path`)."""
  if not data.startswith(b"#?"):
    raise ValueError(f"{path}: not a Radiance .hdr file (missing '#?' signature)")
  try:
    header_end = data.index(b"\n\n") + 2
    resolution_end = data.index(b"\n", header_end)
  except ValueError:
    raise ValueError(f"{path}: truncated .hdr header") from None
  formats = [line.removeprefix(b"FORMAT=") for line in data[:header_end].splitlines() if line.startswith(b"FORMAT=")]
  if not formats or any(value != b"32-bit_rle_rgbe" for value in formats):
    raise ValueError(f"{path}: .hdr FORMAT must be '32-bit_rle_rgbe'; XYZE and other encodings are unsupported")
  match = re.fullmatch(rb"-Y (\d+) \+X (\d+)", data[header_end:resolution_end])
  if match is None:
    raise ValueError(f"{path}: only '-Y H +X W' oriented .hdr files are supported")
  height, width = int(match.group(1)), int(match.group(2))
  if height <= 0 or width <= 0:
    raise ValueError(f"{path}: .hdr dimensions must be positive, got {height} x {width}")
  return height, width, resolution_end + 1


def size(path: str) -> tuple[int, int]:
  """(height, width) of a Radiance .hdr file (reads its first 64 KiB)."""
  height, width, _ = _parse_header(files.read(path, _HEADER_BYTES), path)
  return height, width


def _scanline[W: IntVar](
  data: bytes, pos: int, width: Int[W], *, path: str, y: int
) -> tuple[np.ndarray[[W, 4], np.dtype[np.uint8]], int]:
  """Decodes the adaptive run-length encoded RGBE scanline at `pos`: (rgbe (width, 4), end)."""
  line = bytearray(4 * width)
  n = len(data)
  for c in range(4):
    x = c * width
    end = x + width
    while x < end:
      if pos >= n:
        raise ValueError(f"{path}: truncated run-length data in scanline {y}")
      count = data[pos]
      pos += 1
      if count > _RUN:
        # A run: one byte repeated count - 128 times.
        count -= _RUN
        if count > end - x or pos >= n:
          raise ValueError(f"{path}: bad run-length data in scanline {y}")
        line[x : x + count] = bytes((data[pos],)) * count
        pos += 1
      else:
        # A literal sequence of count bytes.
        if count == 0 or count > end - x or pos + count > n:
          raise ValueError(f"{path}: bad run-length data in scanline {y}")
        line[x : x + count] = data[pos : pos + count]
        pos += count
      x += count
  return np.frombuffer(bytes(line), dtype=np.uint8).reshape(4, width).T, pos


def read_hdr(path: str | os.PathLike[str]) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
  """Reads a Radiance RGBE (.hdr) file as float32 (height, width, 3) linear radiance.

  Raises:
    ValueError: if the file is not a '-Y H +X W' oriented RGBE file or its pixel data is
      truncated or malformed.
  """
  name = os.fspath(path)
  data = files.read(name)
  height, width, pos = _parse_header(data, name)
  rgbe = np.empty((height, width, 4), dtype=np.uint8)
  for y in range(height):
    # A run-length scanline starts with 2, 2 and its width as two bytes (high bit clear).
    rle = (
      _RLE_MIN_WIDTH <= width <= _RLE_MAX_WIDTH
      and data[pos : pos + 2] == b"\x02\x02"
      and pos + 4 <= len(data)
      and not data[pos + 2] & 0x80
    )
    if not rle:
      # Flat (non run-length encoded) scanline.
      if pos + 4 * width > len(data):
        raise ValueError(f"{name}: truncated pixel data in scanline {y}")
      rgbe[y] = np.frombuffer(data, dtype=np.uint8, count=4 * width, offset=pos).reshape(width, 4)
      pos += 4 * width
      continue
    if (data[pos + 2] << 8 | data[pos + 3]) != width:
      raise ValueError(f"{name}: scanline {y} width does not match the header")
    rgbe[y], pos = _scanline(data, pos + 4, width, path=name, y=y)
  exponent = rgbe[..., 3].astype(np.int32)
  scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0).astype(np.float32)
  return rgbe[..., :3].astype(np.float32) * scale[..., None]
