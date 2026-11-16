"""Colour science of the imaging stages (`core/post/grading/transform.py`): the white point of a
colour temperature and tint, von Kries chromatic adaptation (the Bradford cone space) and ALEXA
LogC.

Device functions, with a host launcher (`white_balance`) that runs the same functions on the CPU
for tests and tools (one implementation).
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import numpy.typing as npt
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy
from raywarp.scene.lighting.photometry import (
  DAYLIGHT_RANGE,
  DAYLIGHT_SPLIT,
  DAYLIGHT_X_HIGH,
  DAYLIGHT_X_LOW,
  PLANCKIAN_RANGE,
  PLANCKIAN_U,
  PLANCKIAN_V,
)

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Linear Rec. 709 (sRGB primaries, D65) <-> CIE XYZ (IEC 61966-2-1).
RGB_TO_XYZ: Final = wp.constant(
  wp.mat33(0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041)
)
XYZ_TO_RGB: Final = wp.constant(
  wp.mat33(3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252)
)
# XYZ -> cone responses of the white balance's von Kries adaptation (Bradford, as Filament's).
BRADFORD: Final = wp.constant(wp.mat33(0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296))
# The loci's coefficients (`scene.lighting.photometry`: one definition for the host and the device).
_Coefficients5 = wp.types.vector(length=5, dtype=wp.float32)
_PLANCKIAN_U: Final = wp.constant(_Coefficients5(*PLANCKIAN_U))
_PLANCKIAN_V: Final = wp.constant(_Coefficients5(*PLANCKIAN_V))
_DAYLIGHT_X_LOW: Final = wp.constant(wp.vec4(*DAYLIGHT_X_LOW))
_DAYLIGHT_X_HIGH: Final = wp.constant(wp.vec4(*DAYLIGHT_X_HIGH))
# Below this colour temperature [K] the white is the Planckian locus, from `_DAYLIGHT_FULL` the
# CIE daylight locus (D50, D65 exact), blended linearly in uv between.
_DAYLIGHT_START: Final = 4000.0
_DAYLIGHT_FULL: Final = 5000.0
# Duv of the tint at +1 (offset along the isotherm, towards green).
TINT_DUV: Final = 0.05
# ALEXA LogC (EI 800) as Filament's colour grading uses it.
_LOGC_A: Final = 5.555556
_LOGC_B: Final = 0.047996
_LOGC_C: Final = 0.244161
_LOGC_D: Final = 0.386036


@wp.func
def planckian_xy(
  # In:
  kelvin: float,
) -> wp.vec2:
  """CIE 1931 xy of a black body at `kelvin` (`photometry.planckian_xy`: Krystek's locus, the
  same coefficients; clamped to PLANCKIAN_RANGE)."""
  t = wp.clamp(kelvin, wp.static(PLANCKIAN_RANGE[0]), wp.static(PLANCKIAN_RANGE[1]))
  pu = _PLANCKIAN_U
  pv = _PLANCKIAN_V
  u = (pu[0] + pu[1] * t + pu[2] * t * t) / (1.0 + pu[3] * t + pu[4] * t * t)
  v = (pv[0] + pv[1] * t + pv[2] * t * t) / (1.0 + pv[3] * t + pv[4] * t * t)
  d = 2.0 * u - 8.0 * v + 4.0
  return wp.vec2(3.0 * u / d, 2.0 * v / d)


@wp.func
def daylight_xy(
  # In:
  kelvin: float,
) -> wp.vec2:
  """CIE 1931 xy of the CIE daylight illuminant of correlated temperature `kelvin`
  (`photometry.daylight_xy`, the same coefficients; clamped to DAYLIGHT_RANGE)."""
  t = wp.clamp(kelvin, wp.static(DAYLIGHT_RANGE[0]), wp.static(DAYLIGHT_RANGE[1]))
  c = wp.where(t > wp.static(DAYLIGHT_SPLIT), _DAYLIGHT_X_HIGH, _DAYLIGHT_X_LOW)
  it = 1.0e3 / t
  x = ((c[3] * it + c[2]) * it + c[1]) * it + c[0]
  return wp.vec2(x, (-3.0 * x + 2.87) * x - 0.275)


@wp.func
def xy_to_uv(
  # In:
  xy: wp.vec2,
) -> wp.vec2:
  """Converts CIE 1931 xy to CIE 1960 uv."""
  d = -2.0 * xy[0] + 12.0 * xy[1] + 3.0
  return wp.vec2(4.0 * xy[0] / d, 6.0 * xy[1] / d)


@wp.func
def uv_to_xy(
  # In:
  uv: wp.vec2,
) -> wp.vec2:
  """Converts CIE 1960 uv to CIE 1931 xy."""
  d = 2.0 * uv[0] - 8.0 * uv[1] + 4.0
  return wp.vec2(3.0 * uv[0] / d, 2.0 * uv[1] / d)


@wp.func
def locus_uv(
  # In:
  kelvin: float,
) -> wp.vec2:
  """Uv of the white locus: Planckian below 4000 K, daylight from 5000 K, blended between."""
  planck = xy_to_uv(planckian_xy(kelvin))
  if kelvin <= wp.static(_DAYLIGHT_START):
    return planck
  day = xy_to_uv(daylight_xy(kelvin))
  s = wp.clamp((kelvin - wp.static(_DAYLIGHT_START)) / wp.static(_DAYLIGHT_FULL - _DAYLIGHT_START), 0.0, 1.0)
  return planck + (day - planck) * s


@wp.func
def white_xy(
  # In:
  kelvin: float,
  tint: float,
) -> wp.vec2:
  """Xy of the white of colour temperature `kelvin` moved `TINT_DUV` x tint along its isotherm
  (the locus normal in uv; tint > 0: greener)."""
  uv = locus_uv(kelvin)
  tangent = locus_uv(kelvin * 1.001) - locus_uv(kelvin * 0.999)
  normal = wp.normalize(wp.vec2(tangent[1], -tangent[0]))
  # Rising temperature runs the locus towards lower u: (dv, -du) points to the green side (+v).
  return uv_to_xy(uv + normal * (tint * wp.static(TINT_DUV)))


@wp.func
def xy_to_xyz(
  # In:
  xy: wp.vec2,
) -> wp.vec3:
  """XYZ of luminance 1 at chromaticity xy."""
  return wp.vec3(xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1])


@wp.func
def adaptation_matrix(
  # In:
  kelvin: float,
  tint: float,
) -> wp.mat33:
  """Linear Rec. 709 -> linear Rec. 709 von Kries adaptation from the white of (`kelvin`, `tint`)
  to D65 (6504 K on the daylight locus): what the camera's white balance does."""
  m = BRADFORD
  src = m @ xy_to_xyz(white_xy(kelvin, tint))
  dst = m @ xy_to_xyz(white_xy(6504.0, 0.0))
  gain = wp.mat33(dst[0] / src[0], 0.0, 0.0, 0.0, dst[1] / src[1], 0.0, 0.0, 0.0, dst[2] / src[2])
  return XYZ_TO_RGB @ wp.inverse(m) @ gain @ m @ RGB_TO_XYZ


@wp.func
def linear_to_logc(
  # In:
  x: wp.vec3,
) -> wp.vec3:
  """Scene-linear -> ALEXA LogC (inputs clamped at the log's domain)."""
  out = wp.vec3(0.0)
  for k in range(3):
    v = wp.max(wp.static(_LOGC_A) * x[k] + wp.static(_LOGC_B), 1.0e-10)
    out[k] = wp.static(_LOGC_C) * wp.log10(v) + wp.static(_LOGC_D)
  return out


@wp.func
def logc_to_linear(
  # In:
  x: wp.vec3,
) -> wp.vec3:
  """ALEXA LogC -> scene-linear."""
  out = wp.vec3(0.0)
  for k in range(3):
    out[k] = (wp.pow(10.0, (x[k] - wp.static(_LOGC_D)) / wp.static(_LOGC_C)) - wp.static(_LOGC_B)) / wp.static(_LOGC_A)
  return out


@wp.kernel
def _white_points(
  # In:
  kelvin_tint: wp.array1d[wp.vec2],
  # Out:
  xy_out: wp.array1d[wp.vec2],
  matrix_out: wp.array1d[wp.mat33],
) -> None:
  """White xy and adaptation matrix of each (kelvin, tint)."""
  i = wp.tid()
  k = kelvin_tint[i]
  xy_out[i] = white_xy(k[0], k[1])
  matrix_out[i] = adaptation_matrix(k[0], k[1])


def white_balance(
  kelvin_tint: npt.ArrayLike,
) -> tuple[np.ndarray[[int, 2], np.dtype[np.float32]], np.ndarray[[int, 3, 3], np.dtype[np.float32]]]:
  """((n, 2) white xy, (n, 3, 3) adaptation matrices) of (n, 2) (kelvin, tint) rows: the device
  functions run on the CPU."""
  kt = np.asarray(kelvin_tint, np.float32).reshape(-1, 2)
  src = wp.array(kt, dtype=wp.vec2, device="cpu")
  xy = wp.empty(len(kt), dtype=wp.vec2, device="cpu")
  m = wp.empty(len(kt), dtype=wp.mat33, device="cpu")
  wp.launch(_white_points, dim=len(kt), inputs=[src], outputs=[xy, m], device="cpu")
  # The shapes and dtype are the launch's (Warp's `numpy()` is untyped).
  return cast("np.ndarray[[int, 2], np.dtype[np.float32]]", host_copy(xy)), cast(
    "np.ndarray[[int, 3, 3], np.dtype[np.float32]]", host_copy(m)
  )
