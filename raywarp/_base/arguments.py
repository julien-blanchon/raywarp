"""Checks of the numeric arguments of the public entry points (`Renderer`, `CameraSpec`):
counts, image sizes, seeds, clip distances and other numbers are rejected at the call when they
are bools, of the wrong kind, not finite or out of range, instead of failing (or silently
wrapping, or tracing garbage) deep in an allocation or a kernel launch.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Final, SupportsFloat, TypeGuard

import numpy as np

# The kernels' int32 parameters (seeds, ids): larger values would wrap (two seeds, one image).
INT32_MIN: Final = -(2**31)
INT32_MAX: Final = 2**31 - 1
SEED_MIN: Final = INT32_MIN
SEED_MAX: Final = INT32_MAX


def is_number(v: object) -> TypeGuard[SupportsFloat]:
  """Whether `v` is a real number (Python or NumPy), not a bool."""
  return isinstance(v, numbers.Real) and not isinstance(v, bool | np.bool_)


def as_int(value: object) -> int | None:
  """`value` as an int when it is an integer (Python or NumPy, not a bool), else None."""
  if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
    return None
  return int(value)


def check_int(name: str, value: object, *, low: int, high: int | None = None) -> int:
  """`value` when it is an integer (not a bool) in [low, high].

  Raises:
    ValueError: for a bool, a non-integer or a value out of range.
  """
  v = as_int(value)
  if v is None or v < low or (high is not None and v > high):
    bound = f"at least {low}" if high is None else f"in [{low}, {high}]"
    raise ValueError(f"{name} must be {bound} (an integer), got {value!r}")
  return v


def check_number(name: str, value: object) -> float:
  """`value` (one number, Python or NumPy) as a float.

  Raises:
    ValueError: for a value that is not one finite number (a bool, a string or a sequence).
  """
  if not is_number(value) or not math.isfinite(float(value)):
    raise ValueError(f"{name} must be a finite number, got {value!r}")
  return float(value)


def check_numbers(name: str, values: Sequence[float] | float) -> tuple[float, ...]:
  """`values` (one number, Python or NumPy, or a sequence of them) as floats.

  Raises:
    ValueError: for a value that is not a finite number (a bool or a string included).
  """
  try:
    items: tuple[object, ...] = tuple(values) if isinstance(values, (Sequence, np.ndarray)) else (values,)
  except TypeError as e:
    raise ValueError(f"{name} must be finite numbers, got {values!r}") from e
  out = tuple(float(v) for v in items if is_number(v))
  if len(out) != len(items) or not all(math.isfinite(v) for v in out):
    raise ValueError(f"{name} must be finite numbers, got {values!r}")
  return out


def check_resolution(name: str, resolution: object) -> tuple[int, int]:
  """(width, height) when `resolution` is a pair of integers >= 1 [pixels].

  Raises:
    ValueError: for another length, a non-integer or a size below 1.
  """
  match resolution:
    case (w, h):  # any sequence of two (not a string)
      width, height = as_int(w), as_int(h)
      if width is not None and height is not None and min(width, height) >= 1:
        return width, height
  raise ValueError(f"{name} must be (width, height) integers >= 1, got {resolution!r}")


def _distance(name: str, value: object) -> float:
  """`value` as a float when it is a finite real number >= 0 [m].

  Raises:
    ValueError: for a bool, a non-number, a NaN / infinity or a negative value.
  """
  v = float(value) if is_number(value) else math.nan
  if not (math.isfinite(v) and v >= 0.0):
    raise ValueError(f"{name} must be a finite distance >= 0 [m] (zfar 0: unbounded), got {value!r}")
  return v


def check_clip(znear: object, zfar: object, where: str) -> tuple[float | None, float | None]:
  """(znear, zfar) [m] as floats (None: unchanged / the default) when each is a finite distance
  >= 0 and znear is before zfar unless zfar is 0 (unbounded).

  Raises:
    ValueError: for an invalid distance or a far plane not beyond the near one (`where` names
      the caller).
  """
  near = None if znear is None else _distance(f"{where}: znear", znear)
  far = None if zfar is None else _distance(f"{where}: zfar", zfar)
  if near is not None and far is not None and 0.0 < far <= near:
    raise ValueError(f"{where}: znear ({near}) must be below zfar ({far}; 0 = unbounded)")
  return near, far
