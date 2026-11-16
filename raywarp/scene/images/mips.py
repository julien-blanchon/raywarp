"""Mip levels of stored images: complete-area box filtering, in linear space for sRGB and pure-gamma
8-bit texels.

The 8-bit transfers live here too (`Transfer`, `SRGB_TO_LINEAR`, `gamma_lut`, `encode_u8`):
`decode.py` uses them to decode and encode texels.
"""

from __future__ import annotations

import functools
import math
from typing import Final, Literal

import numpy as np

from raywarp.scene.model.types import Array, Texels

# sRGB transfer function (IEC 61966-2-1): the encoded and the linear value where its linear segment
# ends.
SRGB_KNEE: Final = 0.04045
_LINEAR_KNEE: Final = 0.0031308
# Channel counts whose last channel is not alpha (grey, RGB).
_NO_ALPHA: Final = (1, 3)
# Source texels per row strip of `reduce_u8` (a few MB of float32 at a time).
_REDUCE_STRIP_TEXELS: Final = 1 << 20

# How 8-bit color channels encode linear values: as they are, the sRGB curve, or a pure gamma
# (its exponent). Mips filter in linear (`downsample`); alpha is always linear.
type Transfer = Literal["linear", "srgb"] | float


def ceil_div(a: int, b: int) -> int:
  """ceil(a / b) of positive integers."""
  return -(-a // b)


def box_reduce(image: Texels, factor: int) -> Texels:
  """Box-filters a float image by an integer factor (edges padded by replication)."""
  if factor == 1:
    return image
  h, w = image.shape[:2]
  ph, pw = ceil_div(h, factor) * factor, ceil_div(w, factor) * factor
  if (ph, pw) != (h, w):
    image = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
  return image.reshape(ph // factor, factor, pw // factor, factor, -1).mean(axis=(1, 3))


def area_half_sum[T: (np.float32, np.uint64)](
  image: np.ndarray[[int, int, int], np.dtype[T]], *, width: int, height: int
) -> tuple[np.ndarray[[int, int, int], np.dtype[T]], int]:
  """Integer-weighted area sum and divisor for one floor-halving step.

  An odd axis 2n+1 covers three source texels per destination: weights n-i, n, i+1,
  divided by 2n+1. An even axis covers two equally. Every source texel contributes, and
  destination cells cover equal source area. uint64 preserves exact uint8 rounding; float32
  is shared by linear color/data and coverage. Callers keep their even-size fast path.
  """
  divisor = 1
  if image.shape[1] != width:
    size = image.shape[1]
    if size % 2:
      i = np.arange(width).astype(image.dtype)[None, :, None]
      image = image[:, 0 : 2 * width : 2] * (width - i) + image[:, 1 : 2 * width : 2] * width + image[:, 2::2] * (i + 1)
      divisor *= size
    else:
      image = image[:, 0::2] + image[:, 1::2]
      divisor *= 2
  if image.shape[0] != height:
    size = image.shape[0]
    if size % 2:
      i = np.arange(height).astype(image.dtype)[:, None, None]
      image = image[0 : 2 * height : 2] * (height - i) + image[1 : 2 * height : 2] * height + image[2::2] * (i + 1)
      divisor *= size
    else:
      image = image[0::2] + image[1::2]
      divisor *= 2
  return image, divisor


def mip_chain(base: Texels, levels: int, *, transfer: Transfer) -> list[Texels]:
  """`base` and its next `levels - 1` mip levels (`downsample`)."""
  out = [base]
  for _ in range(levels - 1):
    out.append(downsample(out[-1], transfer=transfer))
  return out


def read_only[T: np.ndarray](table: T) -> T:
  """`table`, made read-only (a shared table no caller may write)."""
  table.setflags(write=False)
  return table


@functools.cache
def gamma_lut(gamma: float) -> np.ndarray[[int], np.dtype[np.float32]]:
  """The read-only (256,) linear values of 8-bit codes encoded with pure gamma `gamma`, computed
  in float64 and rounded once, as `SRGB_TO_LINEAR` (cached per gamma: a scene has a few)."""
  return read_only(((np.arange(256) / 255.0) ** gamma).astype(np.float32))


def srgb_decode(encoded: Array) -> Array:
  """Linear values of sRGB-encoded ones (the transfer function's inverse), in their own float
  dtype; values below the knee, negatives included, follow the linear segment."""
  curve = ((np.maximum(encoded, SRGB_KNEE) + 0.055) / 1.055) ** 2.4  # (no power of a negative base)
  return np.where(encoded <= SRGB_KNEE, encoded / 12.92, curve)


# The one 8-bit sRGB decode table (code -> linear float32, computed in float64 and rounded once,
# read-only): host decodes (`decode.linear_rgba`), mips, color-space conversion (`colorspace`)
# and the core's texture formats (`core.pipeline.textures.formats`) all read it.
SRGB_TO_LINEAR: Final = read_only(srgb_decode(np.arange(256) / 255.0).astype(np.float32))


def check_transfer(transfer: Transfer, where: str) -> None:
  """Raises ValueError for a transfer that is neither "linear", "srgb" nor a finite gamma > 0."""
  if transfer in ("linear", "srgb"):
    return
  if isinstance(transfer, bool) or not isinstance(transfer, (int, float)) or not (math.isfinite(transfer) and transfer > 0):
    raise ValueError(f"{where}: transfer must be 'linear', 'srgb' or a finite gamma > 0, got {transfer!r}")


def decode_lut(transfer: Transfer) -> np.ndarray[[int], np.dtype[np.float32]] | None:
  """The 8-bit decode table of `transfer` (None: linear, codes / 255)."""
  if transfer == "linear":
    return None
  return SRGB_TO_LINEAR if transfer == "srgb" else gamma_lut(float(transfer))


def linear_to_srgb_u8(x: Array) -> Array:
  """8-bit sRGB encoding of linear values (clipped to [0, 1])."""
  x = np.clip(x, 0.0, 1.0)
  s = np.where(x <= _LINEAR_KNEE, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
  return (s * 255.0 + 0.5).astype(np.uint8)


def _encode_color(values: Array, transfer: Transfer) -> Array:
  """8-bit codes of linear color values (clipped to [0, 1]) in `transfer`."""
  if transfer == "srgb":
    return linear_to_srgb_u8(values)
  clipped = np.clip(values, 0.0, 1.0)
  encoded = clipped if transfer == "linear" else clipped ** (1.0 / float(transfer))
  return (encoded * 255.0 + 0.5).astype(np.uint8)


def encode_u8(values: Array, *, transfer: Transfer) -> Texels:
  """8-bit texels of linear (h, w, c) values clipped to [0, 1]: the color channels encoded with
  `transfer` (a last alpha channel of 2 or 4 stays linear)."""
  color = values.shape[2] if values.shape[2] in _NO_ALPHA else values.shape[2] - 1
  out = np.empty(values.shape, np.uint8)
  out[..., :color] = _encode_color(values[..., :color], transfer)
  out[..., color:] = _encode_color(values[..., color:], "linear")
  return out


# 8-bit codes as linear values (code / 255, float32): the table of a linear transfer.
_CODES_LINEAR: Final = read_only((np.arange(256) / 255.0).astype(np.float32))


def reduce_u8_work(width: int, height: int, channels: int) -> int:
  """Bytes `reduce_u8` holds besides its input: the edge pad copy and one strip's float32 copies
  (the decode, the row sums, the cell sums)."""
  return width * height * channels + 4 * _REDUCE_STRIP_TEXELS * channels * 3


def reduce_u8(pixels: Texels, factor: int, *, transfer: Transfer) -> Texels:
  """8-bit (h, w, c) texels box-filtered by `factor` in linear light: the color channels decoded
  with `transfer`, averaged, encoded again (a last alpha channel of 2 or 4 averaged as is);
  edges padded by replication (`box_reduce`)."""
  if factor == 1:
    return pixels
  h, w, c = pixels.shape
  color = c if c in _NO_ALPHA else c - 1
  lut = decode_lut(transfer)
  ph, pw = ceil_div(h, factor) * factor, ceil_div(w, factor) * factor
  if (ph, pw) != (h, w):
    pixels = np.pad(pixels, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
  out = np.empty((ph // factor, pw // factor, c), np.float32)
  # Row strips of whole output rows (a few MB of float32 at a time), summed by strided adds.
  step = max(1, _REDUCE_STRIP_TEXELS // (pw * factor)) * factor
  for y in range(0, ph, step):
    block = pixels[y : y + step]
    linear = np.empty(block.shape, np.float32)
    linear[..., :color] = (_CODES_LINEAR if lut is None else lut)[block[..., :color]]
    linear[..., color:] = _CODES_LINEAR[block[..., color:]]
    rows = linear[0::factor].copy()
    for k in range(1, factor):
      rows += linear[k::factor]
    cells = rows[:, 0::factor].copy()
    for k in range(1, factor):
      cells += rows[:, k::factor]
    out[y // factor : (y + step) // factor] = cells
  out *= 1.0 / (factor * factor)
  return encode_u8(out, transfer=transfer)


def downsample(image: Texels, *, transfer: Transfer) -> Texels:
  """Next mip level of `image` (complete-area box filter): 8-bit color channels are filtered in
  linear space (decoded with `transfer`), alpha as it is; float texels as they are."""
  h, w, c = image.shape
  nh, nw = max(1, h // 2), max(1, w // 2)
  lut = decode_lut(transfer) if image.dtype == np.uint8 else None
  if image.dtype != np.uint8:
    src = image.astype(np.float32)
  elif lut is not None:
    src = lut[image]
    if c in (2, 4):
      src[..., -1] = image[..., -1] / 255.0
  else:
    src = image.astype(np.float32) / 255.0
  if (h > 1 and h % 2) or (w > 1 and w % 2):
    total, divisor = area_half_sum(src, width=nw, height=nh)
    out = total / divisor
  else:
    fh, fw = (2 if h > 1 else 1), (2 if w > 1 else 1)
    out = src[: nh * fh, : nw * fw].reshape(nh, fh, nw, fw, c).mean(axis=(1, 3))
  return out.astype(image.dtype) if image.dtype != np.uint8 else encode_u8(out, transfer=transfer)


def mip_count(width: int, height: int) -> int:
  """Levels of a full mip chain (down to 1 x 1)."""
  return max(width, height).bit_length()
