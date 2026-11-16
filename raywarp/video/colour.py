"""sRGB 8-bit RGBA frames -> NV12 (Y'CbCr 4:2:0, BT.709, limited range), on the frames' device.

raywarp's `rgb` output is gamma-encoded sRGB (R'G'B'), so no transfer function is applied: the
BT.709 matrix converts R'G'B' to Y'CbCr exactly as a player converts it back. Limited ("tv")
range: Y' in [16, 235], Cb / Cr in [16, 240] (BT.709 table 3.2: Y = 16 + 219 E'Y,
C = 128 + 224 E'C). Chroma is the mean of each 2x2 block (center siting: the muxer tags
`chroma_sample_loc_type=1`).

One kernel serves both layouts: `rows` x `cols` tiles of the (H, W) input images per output frame
(1 x 1: one output frame per image). Output frames have even sizes (4:2:0 needs them): an odd
input edge is padded by repeating its last row / column; tiles past the last image are black.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Final

import warp as wp

if TYPE_CHECKING:
  from shape_extensions import IntVar

# BT.709 luma coefficients (ITU-R BT.709-6, item 3.2).
KR: Final = 0.2126
KB: Final = 0.0722
KG: Final = 1.0 - KR - KB
# Limited-range scales per 8-bit code value: 219 / 255 for luma, 224 / 255 for chroma.
Y_SCALE: Final = 219.0 / 255.0
C_SCALE: Final = 224.0 / 255.0


@dataclasses.dataclass(frozen=True)
class Tiling:
  """How (H, W) input images map onto output frames.

  Attributes:
    count: input images (nworld * ncam).
    height, width: one input image.
    rows, cols: tiles per output frame (1, 1: one output frame per image).
  """

  count: int
  height: int
  width: int
  rows: int = 1
  cols: int = 1

  @staticmethod
  def mosaic(count: int, height: int, width: int, cols: int = 0) -> Tiling:
    """All images on one frame, `cols` per row (0: ceil(sqrt(count)))."""
    cols = cols or math.ceil(math.sqrt(count))
    return Tiling(count, height, width, math.ceil(count / cols), cols)

  @property
  def frames(self) -> int:
    """Output frames per write."""
    return math.ceil(self.count / (self.rows * self.cols))

  @property
  def frame_size(self) -> tuple[int, int]:
    """(height, width) of an output frame, rounded up to even."""
    return _even(self.rows * self.height), _even(self.cols * self.width)


def _even(n: int) -> int:
  return n + (n & 1)


@wp.func
def _luma(c: wp.vec3) -> float:
  return KR * c[0] + KG * c[1] + KB * c[2]


@wp.func
def _pixel(
  # In:
  images: wp.array3d[wp.vec4ub],
  frame: int,
  y: int,
  x: int,
  rows: int,
  cols: int,
) -> wp.vec3:
  """The R'G'B' code values (0-255) at output pixel (y, x) of `frame` (black past the last
  image)."""
  h, w = images.shape[1], images.shape[2]
  ty = wp.min(y // h, rows - 1)
  tx = wp.min(x // w, cols - 1)
  i = frame * rows * cols + ty * cols + tx
  if i >= images.shape[0]:
    return wp.vec3(0.0)
  c = images[i, wp.min(y - ty * h, h - 1), wp.min(x - tx * w, w - 1)]
  return wp.vec3(float(c[0]), float(c[1]), float(c[2]))


@wp.func
def _code(v: float) -> wp.uint8:
  return wp.uint8(wp.clamp(v + 0.5, 0.0, 255.0))


@wp.kernel
def rgba_to_nv12(
  # In:
  images: wp.array3d[wp.vec4ub],
  rows: int,
  cols: int,
  # Out:
  nv12: wp.array3d[wp.uint8],
):
  """One thread per 2x2 block of an output frame: 4 luma samples and the block's Cb, Cr pair.

  nv12: (frames, 3 H / 2, W): the Y plane (H rows), then the interleaved CbCr plane (H / 2 rows).
  """
  f, by, bx = wp.tid()
  height = nv12.shape[1] * 2 // 3
  mean = wp.vec3(0.0)
  for dy in range(2):
    for dx in range(2):
      y, x = 2 * by + dy, 2 * bx + dx
      c = _pixel(images, f, y, x, rows, cols)
      nv12[f, y, x] = _code(16.0 + Y_SCALE * _luma(c))
      mean += c
  mean *= 0.25
  luma = _luma(mean)
  nv12[f, height + by, 2 * bx] = _code(128.0 + C_SCALE * (mean[2] - luma) / (2.0 * (1.0 - KB)))
  nv12[f, height + by, 2 * bx + 1] = _code(128.0 + C_SCALE * (mean[0] - luma) / (2.0 * (1.0 - KR)))


def convert[N: IntVar, H: IntVar, W: IntVar, F: IntVar, HC: IntVar, WC: IntVar](
  images: wp.array[wp.vec4ub, [N, H, W]],
  tiling: Tiling,
  nv12: wp.array[wp.uint8, [F, HC, WC]],
  stream: wp.Stream | None = None,
) -> None:
  """Writes `images` as NV12 frames into `nv12`: (tiling.frames, 3 H / 2, W) of the even frame
  size.

  Launched on `stream` (None: the device's current stream).
  """
  record(images, tiling, nv12).launch(stream)


def record[N: IntVar, H: IntVar, W: IntVar, F: IntVar, HC: IntVar, WC: IntVar](
  images: wp.array[wp.vec4ub, [N, H, W]],
  tiling: Tiling,
  nv12: wp.array[wp.uint8, [F, HC, WC]],
) -> wp.Launch:
  """The conversion of `images` into `nv12` as a recorded launch (`convert`): replaying it costs a
  few microseconds of host time, a fresh `wp.launch` tens."""
  h, w = tiling.frame_size
  if tuple(nv12.shape) != (tiling.frames, h * 3 // 2, w):
    raise ValueError(f"nv12 is {tuple(nv12.shape)}, expected {(tiling.frames, h * 3 // 2, w)}")
  dims = (tiling.frames, h // 2, w // 2)
  inputs = [images, tiling.rows, tiling.cols]
  return wp.launch(rgba_to_nv12, dims, inputs=inputs, outputs=[nv12], device=images.device, record_cmd=True)
