"""UsdLux's blackbody color (a numpy port of `UsdLuxBlackbodyTemperatureAsRgb`) and the radiance
scale of the UsdLux photometric inputs (`scaled_colors`).

Ported from OpenUSD `pxr/usd/usdLux/blackbody.cpp` (Copyright 2016 Pixar, the Tomorrow Open Source
Technology License 1.0 / Apache-2.0 modified, https://openusd.org/license), whose table comes from
John Walker's "Colour Rendering of Spectra" (https://www.fourmilab.ch/documents/specrend/, public
domain). `scene/` imports no pxr, so the host derivation (`core/pipeline/upload/lights.py`) uses
this port.

The arithmetic follows the C++ operation by operation in float32, so the result is bit-identical to
pxr's (tested against `pxr.UsdLux.BlackbodyTemperatureAsRgb` on a 1000-40000 K grid): `GfVec3f`
scales by a double and rounds to float (exact for float x float products), adds in float, and
divides by the luma as a multiply by the double reciprocal.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from raywarp.scene.images.colorspace import REC709_Y
from raywarp.scene.model.types import Column, Rows

# The table [Rec.709 rgb] from 1000 K to 10000 K in 500 K steps, end knots repeated (pxr's layout:
# the last 4 knots are the segment starting at 10000 K).
_KNOTS: Final = np.array(
  [
    (1.000000, 0.027490, 0.000000),  # 1000 K (approximation)
    (1.000000, 0.027490, 0.000000),  # 1000 K (approximation)
    (1.000000, 0.149664, 0.000000),  # 1500 K (approximation)
    (1.000000, 0.256644, 0.008095),  # 2000 K
    (1.000000, 0.372033, 0.067450),  # 2500 K
    (1.000000, 0.476725, 0.153601),  # 3000 K
    (1.000000, 0.570376, 0.259196),  # 3500 K
    (1.000000, 0.653480, 0.377155),  # 4000 K
    (1.000000, 0.726878, 0.501606),  # 4500 K
    (1.000000, 0.791543, 0.628050),  # 5000 K
    (1.000000, 0.848462, 0.753228),  # 5500 K
    (1.000000, 0.898581, 0.874905),  # 6000 K
    (1.000000, 0.942771, 0.991642),  # 6500 K
    (0.906947, 0.890456, 1.000000),  # 7000 K
    (0.828247, 0.841838, 1.000000),  # 7500 K
    (0.765791, 0.801896, 1.000000),  # 8000 K
    (0.715255, 0.768579, 1.000000),  # 8500 K
    (0.673683, 0.740423, 1.000000),  # 9000 K
    (0.638992, 0.716359, 1.000000),  # 9500 K
    (0.609681, 0.695588, 1.000000),  # 10000 K
    (0.609681, 0.695588, 1.000000),  # 10000 K
    (0.609681, 0.695588, 1.000000),  # 10000 K
  ],
  np.float32,
)
# Catmull-Rom basis: rows give the cubic's coefficients a, b, c, d from the knots k0..k3.
_BASIS: Final = ((-0.5, 1.5, -1.5, 0.5), (1.0, -2.5, 2.0, -0.5), (-0.5, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 0.0))
# Rec.709 luma weights (float32 in pxr).
_LUMA: Final = np.asarray(REC709_Y, np.float32)
_T_MIN_K: Final = np.array([1000.0], np.float32)
_T_SPAN_K: Final = np.array([9000.0], np.float32)
F32_MAX: Final = 3.4028234663852886e38  # the largest float32
_NSEG: Final = np.array([len(_KNOTS) - 4], np.float32)

# Scalars are (1,) float32 arrays below: numpy then rounds each operation to float32, as the C++.
type F32 = np.ndarray[[int], np.dtype[np.float32]]
type Rgb32 = np.ndarray[[3], np.dtype[np.float32]]


def _scaled(s: float, v: Rgb32) -> Rgb32:
  """`GfVec3f * double`: each float scaled in double, rounded to float."""
  return (v.astype(np.float64) * s).astype(np.float32)


def blackbody_rgb(kelvin: float) -> tuple[float, float, float]:
  """Pxr's `UsdLuxBlackbodyTemperatureAsRgb(kelvin)` (kelvin rounded to float32 as the C++ takes
  it): Rec.709 rgb with the luma of (1, 1, 1), clamped to the 1000-10000 K table, negatives 0."""
  # A temperature beyond float32 is +inf to the C++ (its float argument): the table's 10000 K end.
  t: F32 = np.array([min(kelvin, F32_MAX)], np.float32)
  u_spline: F32 = np.clip((t - _T_MIN_K) / _T_SPAN_K, 0.0, 1.0)
  x: F32 = u_spline * _NSEG
  seg = math.floor(float(x[0]))
  u_seg = float((x - np.array([seg], np.float32))[0])
  k = _KNOTS[seg : seg + 4]
  a, b, c, d = (_scaled(w0, k[0]) + _scaled(w1, k[1]) + _scaled(w2, k[2]) + _scaled(w3, k[3]) for w0, w1, w2, w3 in _BASIS)
  rgb = _scaled(u_seg, _scaled(u_seg, _scaled(u_seg, a) + b) + c) + d
  # GfDot in float, left to right.
  terms = rgb * _LUMA
  luma = (terms[0:1] + terms[1:2]) + terms[2:3]
  out = np.maximum(_scaled(1.0 / float(luma[0]), rgb), 0.0)
  return float(out[0]), float(out[1]), float(out[2])


def scaled_colors(
  color: Rows[np.float64, 3],
  *,
  intensity: Column[np.float64],
  exposure: Column[np.float64],
  kelvin: Column[np.float64],
  enabled: Column[np.bool_] | Column[np.uint8],
) -> Rows[np.float64, 3]:
  """(n, 3) c x s: c = color, x blackbody(kelvin) where `enabled`; s = intensity x 2^exposure
  (UsdLux). Float64, elementwise in a fixed operation order (IEEE: identical to a scalar loop);
  `2 ** e` per value, as Python's pow (numpy's power differs by an ulp in some cases)."""
  on = np.asarray(enabled, bool)
  c = np.asarray(color, np.float64)
  if np.any(on):
    tint = {t: blackbody_rgb(t) for t in set(np.asarray(kelvin)[on].tolist())}  # one per temperature
    c = np.where(on[:, None], c * np.array([tint.get(t, (1.0, 1.0, 1.0)) for t in np.asarray(kelvin).tolist()]), c)
  power = np.array([2.0**e for e in np.asarray(exposure, np.float64).tolist()])
  return c * (np.asarray(intensity, np.float64) * power)[:, None]
