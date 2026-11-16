"""How a material texture is stored on the GPU: its channel layout and the storage options.

A core texture is read as four channels (`TexRole` reads them through its texel map), but it
stores only the distinct ones that are not the constant 1 (a CUDA texture holds 1, 2 or 4
channels): its texel layout (`types.TEX_LAYOUT_*`, one int per texture, `shade/materials/texture.py`
`apply_layout`) maps each read channel to a stored one or to 1. Packing is lossless: a stored
channel is filtered exactly as the same channel of the 4-channel texture.

`TextureStorage` (`SceneOptions.textures`) picks how sRGB color is stored (8-bit sRGB decoded by
the texture unit, the default on CUDA, or exact linear float16) and caps the resolution; data and
normal maps are stored exact (8-bit or float16 texels).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Final, Literal, NamedTuple, final, overload

import numpy as np

from raywarp._base.types import TEX_LAYOUT_ONE
from raywarp.scene.images.mips import SRGB_TO_LINEAR, area_half_sum, read_only

# Color channels of an RGBA texel (alpha is channel NCOLOR).
NCOLOR: Final = 3

# An image level: (h, w, channels) texels.
type Level[T: np.generic] = np.ndarray[[int, int, int], np.dtype[T]]
# A float32 level being filtered: (h, w, channels) color or (h, w) coverage.
type Filtered = Level[np.float32] | np.ndarray[[int, int], np.dtype[np.float32]]
# (h, w) weights in [0, 1] of a color image's texels.
type Coverage = np.ndarray[[int, int], np.dtype[np.float32]]

# sRGB 8-bit -> linear float32 (the scene's one table), and linear [0, 1] -> the nearest sRGB
# 8-bit code (midpoints).
_DECODE: Final = SRGB_TO_LINEAR
_MIDPOINTS: Final = ((_DECODE[1:].astype(np.float64) + _DECODE[:-1]) * 0.5).astype(np.float32)
# 8-bit code -> linear float16 (sRGB: the one table rounded to float16, every code the same half
# as the float64 transfer's) and unit float16 (alpha, data): the texels of an uncompressed 8-bit
# texture (`linear_f16`).
SRGB_F16: Final = read_only(SRGB_TO_LINEAR.astype(np.float16))
UNIT_F16: Final = read_only((np.arange(256) / 255.0).astype(np.float16))


# How 8-bit sRGB color is stored: "float16" linear RGBA (8 bytes / texel per 4 channels, exact),
# "srgb8" 8-bit sRGB RGBA (4 bytes, or 1 for a single color channel without alpha; decoded by the
# texture unit per texel before filtering: exact at mip 0; CUDA only).
type ColorStorage = Literal["float16", "srgb8"]
_COLOR: Final = ("float16", "srgb8")


# Largest texture side [texels] of every adapter by default (MuJoCo's upload, Newton's and USD's
# decode): the 512-1024 px training target never reads a finer level of a texture spanning the
# frame, and a 4096 texture holds 16x the memory of a 1024 one.
MAX_TEXTURE_SIZE: Final = 1024


@final
@dataclasses.dataclass(frozen=True)
class TextureStorage:
  """How the material textures of `upload` are stored (`DEFAULT_STORAGE`; every texel exact:
  `EXACT_STORAGE`).

  Data textures (ORM, opacity, roughness...) and normal maps are stored exact (8-bit texels as
  uint8, float textures as float16), and so is HDR color (texels outside [0, 1]). A device
  without the texture unit's sRGB decode (the CPU) stores 8-bit color as float16
  (`TextureStorage.on`).

  Attributes:
    color: 8-bit sRGB color (base color, emissive): `ColorStorage`.
    max_size: largest stored side [texels]; larger textures are halved (area box, linear values)
      until they fit, as if their finest mip levels were dropped (default `MAX_TEXTURE_SIZE`, the
      adapters' load-time cap too; 0: no cap; any size >= 1, a power of two not required: halving
      stops at the first size within it).
  """

  color: ColorStorage = "srgb8"
  max_size: int = MAX_TEXTURE_SIZE

  def __post_init__(self) -> None:
    """Rejects an unknown color format and a negative or non-integer size cap."""
    if self.color not in _COLOR:
      raise ValueError(f"TextureStorage.color must be one of {_COLOR}, got {self.color!r}")
    if isinstance(self.max_size, bool) or not isinstance(self.max_size, int) or self.max_size < 0:
      raise ValueError(f"TextureStorage.max_size must be 0 (no cap) or a size >= 1, got {self.max_size!r}")

  def on(self, *, srgb8: bool) -> TextureStorage:
    """The storage used on a device with (`srgb8`) or without the texture unit's sRGB decode:
    float16 color without it (`max_size` kept)."""
    if srgb8:
      return self
    return dataclasses.replace(self, color="float16")


# The default storage: 8-bit sRGB color (on CUDA; float16 elsewhere), exact data and normal maps,
# textures capped at MAX_TEXTURE_SIZE.
DEFAULT_STORAGE: Final = TextureStorage()
# Every stored texel exact (the same size cap).
EXACT_STORAGE: Final = TextureStorage(color="float16")


@final
class Packed(NamedTuple):
  """A texture's stored channels: `kept` source channels (in order) and the texel layout."""

  kept: tuple[int, ...]
  layout: int


def layout_code(fields: tuple[int, int, int, int]) -> int:
  """The texel layout of the stored channel (or TEX_LAYOUT_ONE) of each read channel."""
  return sum(f << (3 * k) for k, f in enumerate(fields))


def pack_channels(texels: Level[np.uint8] | Level[np.float16], *, border: bool = False, alpha_apart: bool = False) -> Packed:
  """The fewest channels storing (h, w, 4) `texels` exactly.

  A channel that is 1 everywhere (255 / 1.0) reads TEX_LAYOUT_ONE, a copy of an earlier channel
  reads that one; the rest are stored, padded to 4 channels when 3 remain (CUDA textures hold 1,
  2 or 4). A texture of ones keeps its channel 0. `border`: the texture reads 0 outside [0, 1]
  (BORDER address mode), its blend with the edge texels included: a channel of ones is stored
  (once, its copies read it), as the constant 1 would not fade to 0 there. `alpha_apart`: the
  color channels (0-2) and alpha (3) are stored differently (sRGB linearisation, coverage-weighted
  mips): alpha never copies a color channel, nor the reverse.
  """
  one = np.array(255 if texels.dtype == np.uint8 else 1.0, texels.dtype)
  kept: list[int] = []
  source: list[int] = []  # the source channel each read channel equals, -1: the constant 1
  for c in range(4):
    ch = texels[..., c]
    twin = next((s for s in kept if _twins(texels, s, c, alpha_apart=alpha_apart)), None)
    if twin is not None:
      source.append(twin)
    elif not border and bool(np.all(ch == one)):
      source.append(-1)
    else:
      source.append(c)
      kept.append(c)
  if not kept:
    kept, source[0] = [0], 0
  if len(kept) == 3:
    kept = sorted([*kept, next(c for c in range(4) if c not in kept)])
  f = [TEX_LAYOUT_ONE if s < 0 else kept.index(s) for s in source]
  return Packed(tuple(kept), layout_code((f[0], f[1], f[2], f[3])))


def _twins(texels: Level[np.uint8] | Level[np.float16], s: int, c: int, *, alpha_apart: bool) -> bool:
  """Whether channel `c` can read stored channel `s`: equal texels, filtered alike."""
  if alpha_apart and (s == NCOLOR) != (c == NCOLOR):
    return False
  return np.array_equal(texels[..., s], texels[..., c])


@overload
def halve_pairs(x: Level[np.float32], w: int, h: int) -> Level[np.float32]: ...


@overload
def halve_pairs(
  x: np.ndarray[[int, int], np.dtype[np.float32]], w: int, h: int
) -> np.ndarray[[int, int], np.dtype[np.float32]]: ...


def halve_pairs(x: Filtered, w: int, h: int) -> Filtered:
  """2x box filter of a float32 (H, W, ...) level (color) or (H, W) level (coverage) to (h, w)
  as pair averages (width, then height); odd axes integrate the complete source extent."""
  if (x.shape[1] != w and x.shape[1] % 2) or (x.shape[0] != h and x.shape[0] % 2):
    total, divisor = area_half_sum(x.reshape(x.shape[0], x.shape[1], -1), width=w, height=h)
    if x.ndim == 2:
      return total[:, :, 0] / divisor
    return total / divisor
  if x.shape[1] != w:
    x = (x[:, 0 : 2 * w : 2] + x[:, 1 : 2 * w : 2]) * 0.5
  if x.shape[0] != h:
    x = (x[0 : 2 * h : 2] + x[1 : 2 * h : 2]) * 0.5
  return x


def weighted_halve(
  color: Level[np.float32], weight: Coverage | None, ncolor: int, w: int, h: int
) -> tuple[Level[np.float32], Coverage | None]:
  """One coverage-weighted mip step of float32 `color` (and its weights) to (h, w).

  The first `ncolor` channels average as sum(c x weight) / sum(weight) where a block is covered
  (premultiplied-alpha filtering), plainly elsewhere; the others and the weights average plainly.
  Without weights: an area box over the complete image extent.
  """
  plain = halve_pairs(color, w, h)
  if weight is None:
    return plain, None
  weighted = halve_pairs(color[..., :ncolor] * weight[..., None], w, h)
  weight = halve_pairs(weight, w, h)
  covered = weight[..., None] > 0.0
  out = plain.copy()
  out[..., :ncolor] = np.where(covered, weighted / np.where(covered, weight[..., None], 1.0), plain[..., :ncolor])
  return out, weight


def level_shapes(width: int, height: int) -> list[tuple[int, int, int]]:
  """Warp's (w, h, 1) of every mip level of a (width, height) texture, down to 1 x 1."""
  return [(max(1, width >> lv), max(1, height >> lv), 1) for lv in range(max(width, height).bit_length())]


def capped_levels(width: int, height: int, max_size: int) -> int:
  """Halvings that bring (width, height) within `max_size` (0: no cap)."""
  k = 0
  while max_size > 0 and max(width >> k, height >> k) > max_size and (width >> k > 1 or height >> k > 1):
    k += 1
  return k


def reduced(
  rgba: Level[np.uint8] | Level[np.float16],
  coverage: Coverage | None,
  k: int,
  *,
  srgb: bool,
  srgb8: bool,
) -> tuple[Level[np.uint8] | Level[np.float16], bool, Coverage | None]:
  """(texels, sRGB, coverage) of (h, w, 4) `rgba` halved `k` times in linear values (color
  coverage-weighted): float16 for color (8-bit sRGB with `srgb8`), uint8 for 8-bit data."""

  def color(c: int) -> bool:
    """Whether channel `c` is sRGB-encoded color."""
    return srgb and c < NCOLOR

  linear = decode_linear(rgba, color)
  weight = coverage
  w, h = rgba.shape[1], rgba.shape[0]
  for _ in range(k):
    w, h = max(1, w // 2), max(1, h // 2)
    linear, weight = weighted_halve(linear, weight, NCOLOR, w, h)
  if rgba.dtype == np.float16 or (srgb and not srgb8):
    return linear.astype(np.float16), False, weight
  return encode_srgb8(linear, color), srgb, weight


def decode_linear(texels: Level[np.uint8] | Level[np.float16], srgb: Callable[[int], bool]) -> Level[np.float32]:
  """Linear float32 values of stored `texels`: sRGB-decoded channels where `srgb(c)`, else unit
  values (uint8 / 255, float16 as they are)."""
  if texels.dtype != np.uint8:
    return texels.astype(np.float32)
  out = texels.astype(np.float32) / 255.0
  for c in range(texels.shape[2]):
    if srgb(c):
      out[..., c] = _DECODE[texels[..., c]]
  return out


def encode_srgb8(linear: Level[np.float32], srgb: Callable[[int], bool]) -> Level[np.uint8]:
  """8-bit texels of linear float32 values: the nearest sRGB code where `srgb(c)`, else rounded
  unit values."""
  out = np.empty(linear.shape, np.uint8)
  for c in range(linear.shape[2]):
    if srgb(c):
      out[..., c] = np.searchsorted(_MIDPOINTS, linear[..., c]).astype(np.uint8)
    else:
      out[..., c] = np.clip(np.rint(linear[..., c] * 255.0), 0, 255).astype(np.uint8)
  return out


def linear_f16(texels: Level[np.uint8], ncolor: int) -> Level[np.float16]:
  """Linear float16 values of 8-bit `texels`: the first `ncolor` channels sRGB-decoded, the rest
  unit values (`SRGB_F16`, `UNIT_F16`)."""
  return np.stack([(SRGB_F16 if c < ncolor else UNIT_F16)[texels[..., c]] for c in range(texels.shape[2])], axis=-1)


def srgb8_mip_chain(
  base: Level[np.uint8], coverage: Coverage | None, shapes: list[tuple[int, int, int]], *, ncolor: int, nsrgb: int
) -> list[Level[np.uint8]]:
  """Mip levels of 8-bit `base` filtered in linear space (decoded, averaged in float32, re-encoded
  per level): its first `nsrgb` channels sRGB, its first `ncolor` channels color (averaged with
  the `coverage` weights), the rest unit values.

  `shapes`: Warp's `(w, h, d)` level shapes; level 0 is `base` unchanged.
  """

  def srgb(c: int) -> bool:
    """Whether channel `c` is sRGB-encoded."""
    return c < nsrgb

  levels = [base]
  linear = decode_linear(base, srgb)
  weight = coverage
  for w, h, _ in shapes[1:]:
    linear, weight = weighted_halve(linear, weight, ncolor, w, h)
    levels.append(encode_srgb8(linear, srgb))
  return levels


def coverage_mip_chain[T: (np.uint8, np.float16)](
  base: Level[T], coverage: Coverage, shapes: list[tuple[int, int, int]], ncolor: int = NCOLOR
) -> list[Level[T]]:
  """Mip levels of an (h, w, c) uint8 / float16 color image whose first `ncolor` channels are
  averaged by `coverage` (the material's opacity).

  Premultiplied-alpha filtering: each level's color is sum(rgb x c) / sum(c) over the texels it
  covers, `c` the (h, w) coverage in [0, 1] (the material's opacity), so the color of cut-away
  or transparent texels (often black) does not bleed into the visible ones as halos at a
  distance; a fully uncovered texel keeps the plain average. The other channels (alpha) are box
  filtered. `shapes`: Warp's `(w, h, d)` level shapes; level 0 is `base` unchanged.
  """
  levels = [base]
  color = base.astype(np.float32)
  weight: Coverage | None = np.asarray(coverage, np.float32)
  for w, h, _ in shapes[1:]:
    color, weight = weighted_halve(color, weight, ncolor, w, h)
    levels.append((np.clip(np.rint(color), 0, 255) if base.dtype == np.uint8 else color).astype(base.dtype))
  return levels
