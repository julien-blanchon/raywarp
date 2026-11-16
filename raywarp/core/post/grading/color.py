"""Color math of the display path: the tone mappers that turn exposed HDR radiance into display
values, the sRGB transfer functions and the 8-bit RGBA packing of the color output.

Device functions (`display_transform`, `encode_display`, the tone mappers) and their host
counterparts (`srgb_to_linear`, `inverse_display_transform`: the exposed color a display value
comes from, solved on the device function itself).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt
import warp as wp

from raywarp._base.types import ToneMapper
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.scene.images.colorspace import REC709_Y

if TYPE_CHECKING:
  from raywarp._base.dims import AnyRank

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


# The sRGB transfer function (IEC 61966-2-1): linear segment slope and its end (linear / encoded),
# the power segment's exponent, scale and offset (encoded = scale x^(1/gamma) - offset).
SRGB_SLOPE: Final = 12.92
SRGB_LINEAR_CUTOFF: Final = 0.0031308
SRGB_ENCODED_CUTOFF: Final = 0.04045
SRGB_GAMMA: Final = 2.4
SRGB_SCALE: Final = 1.055
SRGB_OFFSET: Final = 0.055
# Largest 8-bit channel value (unorm8 <-> [0, 1]).
_UNORM8: Final = 255.0
# Khronos PBR Neutral: the toe below which the black offset is quadratic, and the offset above it.
_NEUTRAL_TOE: Final = 0.08
_NEUTRAL_OFFSET: Final = 0.04


@wp.func
def tonemap_pbr_neutral(
  # In:
  color: wp.vec3,
) -> wp.vec3:
  """Khronos PBR Neutral tone mapper (keeps base colors under neutral light)."""
  start_compression = wp.static(0.8 - _NEUTRAL_OFFSET)
  desaturation = 0.15

  x = wp.min(color[0], wp.min(color[1], color[2]))
  offset = wp.static(_NEUTRAL_OFFSET)
  if x < wp.static(_NEUTRAL_TOE):
    offset = x - 6.25 * x * x
  c = color - wp.vec3(offset)

  peak = wp.max(c[0], wp.max(c[1], c[2]))
  if peak < start_compression:
    return c

  d = 1.0 - start_compression
  new_peak = 1.0 - d * d / (peak + d - start_compression)
  c = c * (new_peak / peak)
  g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0)
  return c * (1.0 - g) + wp.vec3(new_peak * g)


@wp.func
def tonemap_aces(
  # In:
  color: wp.vec3,
) -> wp.vec3:
  """ACES RRT + ODT fit (Stephen Hill), operating on linear Rec.709."""
  # sRGB -> ACES AP1 with the RRT saturation baked in.
  a = wp.mat33(0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777)
  b = wp.mat33(1.60475, -0.53108, -0.07367, -0.10208, 1.10813, -0.00605, -0.00327, -0.07276, 1.07602)
  v = a @ color
  num = wp.cw_mul(v, v + wp.vec3(0.0245786)) - wp.vec3(0.000090537)
  den = wp.cw_mul(v, 0.983729 * v + wp.vec3(0.4329510)) + wp.vec3(0.238081)
  v = wp.cw_div(num, den)
  return b @ v


@wp.func
def tonemap_aces_rtx(
  # In:
  color: wp.vec3,
) -> wp.vec3:
  """Narkowicz's ACES fit as Omniverse RTX applies it ("acesApproximation").

  Exposure is applied before this curve, identically for every tone mapper.
  """
  x = color
  return wp.cw_div(wp.cw_mul(x, x * 2.51 + wp.vec3(0.03)), wp.cw_mul(x, x * 2.43 + wp.vec3(0.59)) + wp.vec3(0.14))


@wp.func
def srgb_oetf(
  # In:
  x: wp.float32,
) -> wp.float32:
  """Linear -> sRGB-encoded display value (the sRGB transfer function)."""
  if x <= wp.static(SRGB_LINEAR_CUTOFF):
    return wp.static(SRGB_SLOPE) * x
  return wp.static(SRGB_SCALE) * wp.pow(x, wp.static(1.0 / SRGB_GAMMA)) - wp.static(SRGB_OFFSET)


@wp.func
def srgb_eotf(
  # In:
  x: wp.float32,
) -> wp.float32:
  """SRGB-encoded display value -> linear."""
  if x <= wp.static(SRGB_ENCODED_CUTOFF):
    return x * wp.static(1.0 / SRGB_SLOPE)
  return wp.pow((x + wp.static(SRGB_OFFSET)) * wp.static(1.0 / SRGB_SCALE), wp.static(SRGB_GAMMA))


@wp.func
def luminance(
  # In:
  c: wp.vec3,
) -> wp.float32:
  """Relative luminance of linear Rec. 709 color."""
  return wp.dot(c, wp.static(wp.vec3(*REC709_Y)))


def srgb_to_linear(x: npt.ArrayLike) -> AnyRank[np.float64]:
  """SRGB display values in [0, 1] -> linear (host-side `srgb_eotf`)."""
  x = np.asarray(x, dtype=np.float64)
  return np.where(x <= SRGB_ENCODED_CUTOFF, x / SRGB_SLOPE, ((x + SRGB_OFFSET) / SRGB_SCALE) ** SRGB_GAMMA)


@wp.func
def display_transform(
  # In:
  color: wp.vec3,
  tone_mapper: wp.int32,
) -> wp.vec3:
  """Exposed linear HDR -> tone-mapped linear [0, 1].

  The result is still linear: average sub-pixel samples on it, then encode (`encode_display`).
  """
  c = wp.max(color, wp.vec3(0.0))
  if tone_mapper == wp.static(int(ToneMapper.PBR_NEUTRAL)):
    c = tonemap_pbr_neutral(c)
  elif tone_mapper == wp.static(int(ToneMapper.ACES)):
    c = tonemap_aces(c)
  elif tone_mapper == wp.static(int(ToneMapper.ACES_RTX)):
    c = tonemap_aces_rtx(c)
  return wp.min(wp.max(c, wp.vec3(0.0)), wp.vec3(1.0))


# Newton iterations of `inverse_display_transform` and their stopping residual; the Jacobian's
# forward-difference step (relative to the color, with a floor) and the Levenberg-Marquardt damping.
_INVERSE_ITERATIONS: Final = 64
_INVERSE_TOLERANCE: Final = 1.0e-7
_JACOBIAN_STEP: Final = 1.0e-3
_MIN_JACOBIAN_STEP: Final = 1.0e-4
_LM_DAMPING: Final = 1.0e-9


@wp.kernel
def _invert_display_transform(
  # In:
  tone_mapper: wp.int32,
  target_in: wp.array1d[wp.vec3],
  # Out:
  color_out: wp.array1d[wp.vec3],
) -> None:
  """Exposed color whose `display_transform` is the target: damped Newton steps on a
  finite-difference Jacobian of the device tone mapper itself (no host copy to drift from it)."""
  i = wp.tid()
  target = wp.min(wp.max(target_in[i], wp.vec3(0.0)), wp.vec3(1.0))
  x = target
  for _ in range(wp.static(_INVERSE_ITERATIONS)):
    y = display_transform(x, tone_mapper)
    r = y - target
    if wp.max(wp.abs(r[0]), wp.max(wp.abs(r[1]), wp.abs(r[2]))) < wp.static(_INVERSE_TOLERANCE):
      break
    # Columns of the Jacobian: forward differences with a step relative to the color.
    j = wp.mat33(0.0)
    for k in range(3):
      e = wp.vec3(0.0)
      h = wp.max(wp.static(_JACOBIAN_STEP) * x[k], wp.static(_MIN_JACOBIAN_STEP))
      e[k] = h
      d = (display_transform(x + e, tone_mapper) - y) / h
      j[0, k] = d[0]
      j[1, k] = d[1]
      j[2, k] = d[2]
    # Levenberg-Marquardt step (J^T J + mu I)^-1 J^T r: flat regions (a clamped channel, PBR
    # Neutral's shoulder) stay solvable and converge to the closest reachable color.
    jt = wp.transpose(j)
    step = wp.inverse(jt @ j + wp.identity(3, dtype=float) * _LM_DAMPING) @ (jt @ r)
    x = wp.max(x - step, wp.vec3(0.0))
  color_out[i] = x


def inverse_display_transform(display: npt.ArrayLike, tone_mapper: int, device: wp.DeviceLike) -> AnyRank[np.float64]:
  """Exposed linear HDR colors (..., 3) whose `display_transform` is `display` (display-linear).

  Solved on the device function itself (`_invert_display_transform`, on `device`: the renderer's,
  so that its prebuilt kernels hold it and nothing compiles on the host); colors the tone mapper
  cannot reach (e.g. 1.0 under PBR Neutral's shoulder) give the closest one found.
  """
  d = np.asarray(display, dtype=np.float32)
  # On the device by name: Warp's default device is process-wide (no `wp.ScopedDevice` here).
  t = wp.array(d.reshape(-1, 3), dtype=wp.vec3, device=device)
  x = wp.empty_like(t)
  wp.launch(_invert_display_transform, dim=t.shape[0], inputs=[tone_mapper, t], outputs=[x], device=device)
  return host_copy(x).reshape(d.shape).astype(np.float64)


@wp.func
def pack_rgba(
  # In:
  r: wp.float32,
  g: wp.float32,
  b: wp.float32,
  a: wp.float32,
) -> wp.uint32:
  """8-bit channel values (truncated) packed so that the bytes in memory are R, G, B, A."""
  return wp.uint32((wp.int32(a) << wp.int32(24)) | (wp.int32(b) << wp.int32(16)) | (wp.int32(g) << wp.int32(8)) | wp.int32(r))


@wp.func
def unpack_rgba(
  # In:
  c: wp.uint32,
) -> wp.vec4:
  """`pack_rgba`'s bytes (R in the lowest) as floats in [0, 1]."""
  v = wp.int32(c)  # int32 bit operations (the sign bit is masked off)
  return wp.vec4(
    wp.float32(v & 0xFF) / wp.static(_UNORM8),
    wp.float32((v >> 8) & 0xFF) / wp.static(_UNORM8),
    wp.float32((v >> 16) & 0xFF) / wp.static(_UNORM8),
    wp.float32((v >> 24) & 0xFF) / wp.static(_UNORM8),
  )


@wp.func
def encode_display(
  # In:
  c: wp.vec3,
  alpha: wp.float32,
) -> wp.uint32:
  """Display-linear color in [0, 1] and alpha in [0, 1] -> 8-bit sRGB + alpha, RGBA byte order.

  One 32-bit store; alpha 1 (255) unless the coverage alpha is requested (`ColorOutput.alpha`).
  """
  # + 0.5: `pack_rgba` truncates, so the 8-bit values are rounded to nearest.
  return pack_rgba(
    srgb_oetf(c[0]) * wp.static(_UNORM8) + 0.5,
    srgb_oetf(c[1]) * wp.static(_UNORM8) + 0.5,
    srgb_oetf(c[2]) * wp.static(_UNORM8) + 0.5,
    alpha * wp.static(_UNORM8) + 0.5,
  )
