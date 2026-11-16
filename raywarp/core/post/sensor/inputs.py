"""What the sensor pass (`sensor.py`) and the post anti-aliasing (`fxaa.py`) share: the pass's
runtime parameters (`SensorParams`) and the display they read (`decode_display`: a pixel's display
color decoded to display-linear, by table for the 8-bit layouts), with the bilinear lookup's corner.

A plain Warp module (`kernel_code.WARP_MODULES`).
"""

from __future__ import annotations

import warp as wp

from raywarp.core.kernels.abi.output_layout import OUT_CHW8, OUT_CHW16, OUT_RGB8, OUT_RGBA
from raywarp.core.kernels.abi.structs import OutputArrays, f16s, rgb8s, u8s, u32s
from raywarp.core.kernels.abi.toggles import out_on
from raywarp.core.post.grading.color import (
  SRGB_ENCODED_CUTOFF,
  SRGB_GAMMA,
  SRGB_LINEAR_CUTOFF,
  SRGB_OFFSET,
  SRGB_SCALE,
  SRGB_SLOPE,
)

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class SensorParams:
  """The runtime parameters of the pass (`settings.Sensor`), the per-world ones apart.

  Attributes:
    rolling_shutter, motion_blur: row readout and exposure time over the frame interval.
    noise: shot and read noise on.
    levels: the ADC's steps (2^bits - 1); 0: no quantisation.
    stereo: the depth model is the stereo one (else the range model).
    baseline, disparity_noise, disparity_step: the stereo model.
    range_noise: (sigma0, z0, k) of the range model.
    min_depth, max_depth, dropout, edge_dropout, edge_threshold: both models.
    miss: `settings.MISS_DEPTHS` id of the depth output's miss value.
    radial: the depth output is radial (else planar).
  """

  rolling_shutter: float
  motion_blur: float
  noise: bool
  levels: float
  stereo: bool
  baseline: float
  disparity_noise: float
  disparity_step: float
  range_noise: wp.vec3
  min_depth: float
  max_depth: float
  dropout: float
  edge_dropout: float
  edge_threshold: float
  miss: int
  radial: bool


@wp.func
def power(
  # In:
  x: float,
  y: float,
) -> float:
  """x^y for x >= 0 (0 at x <= 0) as exp(y ln x): a few float ulps off `pow` (invisible after the
  8- or 16-bit output encoding), several times cheaper than its exact rounding."""
  if x <= 0.0:
    return 0.0
  return wp.exp(y * wp.log(x))


@wp.func
def srgb_encode(
  # In:
  x: float,
) -> float:
  """`color.srgb_oetf` with its power as `power` (exp / log: a few float ulps off, invisible after
  the output encoding)."""
  if x <= wp.static(SRGB_LINEAR_CUTOFF):
    return wp.static(SRGB_SLOPE) * x
  return wp.static(SRGB_SCALE) * power(x, wp.static(1.0 / SRGB_GAMMA)) - wp.static(SRGB_OFFSET)


@wp.func
def srgb_decode(
  # In:
  x: float,
) -> float:
  """`color.srgb_eotf` with its power as `power`."""
  if x <= wp.static(SRGB_ENCODED_CUTOFF):
    return x * wp.static(1.0 / SRGB_SLOPE)
  return power((x + wp.static(SRGB_OFFSET)) * wp.static(1.0 / SRGB_SCALE), wp.static(SRGB_GAMMA))


@wp.func
def decode_display(
  # In:
  out: OutputArrays,
  srgb8: wp.array1d[float],
  worldid: int,
  adr: int,
  cam_adr: int,
  npix: int,
) -> wp.vec4:
  """The pixel's display color, decoded to display-linear, and its alpha (`display.write_display`'s
  inverse to its encoding: the first color layout of the output word; alpha 1 without an alpha
  byte). The 8-bit layouts decode by table (`srgb8`: `Sensor.srgb8`, 256 entries; exactly rounded
  `wp.pow` was the pass's most expensive part), float16 channels by `srgb_decode`."""
  e = wp.vec4(0.0, 0.0, 0.0, 1.0)
  base = 2 * cam_adr + adr
  if out_on(out.bits, wp.static(OUT_RGBA)):
    v = wp.int32(u32s(out, out.table.rgb)[worldid, adr])  # int32 bit operations (the sign bit masked off)
    return wp.vec4(srgb8[v & 255], srgb8[(v >> 8) & 255], srgb8[(v >> 16) & 255], float((v >> 24) & 255) / 255.0)
  elif out_on(out.bits, wp.static(OUT_RGB8)):
    b = rgb8s(out, out.table.rgb_hwc)[worldid, adr]
    return wp.vec4(srgb8[int(b[0])], srgb8[int(b[1])], srgb8[int(b[2])], 1.0)
  elif out_on(out.bits, wp.static(OUT_CHW8)):
    for k in range(3):
      e[k] = srgb8[int(u8s(out, out.table.chw8, 3)[worldid, base + k * npix])]
  elif out_on(out.bits, wp.static(OUT_CHW16)):
    for k in range(3):
      e[k] = srgb_decode(float(f16s(out, out.table.chw16, 3)[worldid, base + k * npix]))
  return e


@wp.func
def bilinear_corner(
  # In:
  p: wp.vec2,
) -> tuple[int, int, float, float]:
  """The texel (x0, y0) at the lower corner of the 2 x 2 a bilinear lookup at position p [pixels]
  blends, and the weights (fx, fy) of the texels after it in x and y."""
  x = p[0] - 0.5
  y = p[1] - 0.5
  fx = wp.floor(x)
  fy = wp.floor(y)
  return int(fx), int(fy), x - fx, y - fy
