"""The building blocks every settings group shares: the lists of choices of the enumerated
settings, the field kinds (compile-time or runtime, with their limits) and `validate`.

Each enumerated setting has one `Literal` alias (`ToneMapperName`, ...): it is the field's
annotation, `validate` reads its choices from it, and its id table (`TONE_MAPPERS`, ...: the value
`resolve` hands the kernels) is built from it. `validate` checks every field of a group from its
annotation (bool, int, float, `Literal` choices, `Tri`, `None` when optional) and from the limits
its field helper stored in the field's metadata.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
import numbers
import types
import typing
from collections.abc import Sequence
from typing import Final, Literal, TypedDict, TypeGuard, Unpack, final

import numpy as np

from raywarp._base.named import DataclassInstance, field_value
from raywarp._base.types import RenderOrder, ToneMapper

# Default daylight camera: f/16, 1/125 s, ISO 100.
DEFAULT_EV100: Final = 14.96578428466209
# Channels of a color setting (`is_rgb`).
_RGB: Final = 3

# A switch that can also follow the scene ("auto").
type Tri = bool | Literal["auto"]
# What changing a setting rebuilds besides the kernels: the GI probe volume or the light visibility
# grid.
type Rebuild = Literal["gi", "light_grid"]
# An axis-aligned box: (lower corner, upper corner).
type Box = tuple[tuple[float, float, float], tuple[float, float, float]]

# The choices of the enumerated settings (each Literal is the one list of its names).
type OutputKind = Literal[
  "rgb",
  "depth",
  "seg",
  "semantic",
  "hdr",
  "albedo",
  "normal",
  "position",
  "uv",
  "material",
  "roughness",
  "metallic",
  "emission",
  "direct",
  "indirect",
  "shadow",
  "luminance",
  "motion",
]
type ToneMapperName = Literal["pbr_neutral", "aces", "clamp", "aces_rtx"]
type RenderOrderName = Literal["auto", "tiled", "world", "mixed"]
type ReflectionShadowMode = Literal["off", "sun"]
type MissDepth = Literal["zero", "inf", "far"]
type ColorLayout = Literal["rgba8", "rgb8", "rgb8_planar", "rgb16f_planar"]


def choices(alias: typing.TypeAliasType) -> tuple[str, ...]:
  """The names of a `Literal` alias, in order."""
  return typing.get_args(alias.__value__)


def _ids(alias: typing.TypeAliasType) -> dict[str, int]:
  """Name -> its position in the `Literal` alias (the Features / Params id)."""
  return {name: i for i, name in enumerate(choices(alias))}


def _enum_codes(alias: typing.TypeAliasType, ids: type[enum.IntEnum]) -> dict[str, int]:
  """Name -> the id of the device enum member of that name (ImportError if they disagree)."""
  table = {m.name.lower(): int(m) for m in ids}
  if set(table) != set(choices(alias)):
    raise ImportError(f"{ids.__name__} members {sorted(table)} differ from the settings' {choices(alias)}")
  return {name: table[name] for name in choices(alias)}


# The ids of the discrete settings in `Features` / `Params` (`core/build/variants/resolve.py`).
TONE_MAPPERS: Final = _enum_codes(ToneMapperName, ToneMapper)
RENDER_ORDERS: Final = _enum_codes(RenderOrderName, RenderOrder)
REFLECTION_SHADOWS: Final = _ids(ReflectionShadowMode)
MISS_DEPTHS: Final = _ids(MissDepth)
LAYOUTS: Final = _ids(ColorLayout)


class Limits(TypedDict, total=False):
  """The numeric limits a field helper takes: ge / gt (lower bound, closed / open) and le / lt
  (upper bound; `lt=math.inf`: finite)."""

  ge: float
  gt: float
  le: float
  lt: float


@final
@dataclasses.dataclass(frozen=True)
class ValueRange:
  """The range a numeric field is validated in: lo <= x (lo < x when `lo_open`), x <= hi (x < hi
  when `hi_open`; hi = inf open: finite). NaN is outside every range."""

  lo: float
  hi: float
  lo_open: bool
  hi_open: bool

  @classmethod
  def of(cls, limits: Limits) -> ValueRange | None:
    """The range of a field helper's limits (None: unbounded)."""
    if not limits:
      return None
    lo, hi = limits.get("gt", limits.get("ge", -math.inf)), limits.get("lt", limits.get("le", math.inf))
    return cls(lo, hi, lo_open="gt" in limits, hi_open="lt" in limits)

  def holds(self, v: float) -> bool:
    """Whether `v` is in the range."""
    return (self.lo < v if self.lo_open else self.lo <= v) and (v < self.hi if self.hi_open else v <= self.hi)

  def describe(self) -> str:
    """The range in words ("at least 1", "in [0, 16]", "positive and finite", ...)."""
    if self.hi == math.inf:
      if self.lo == -math.inf:
        return "finite"
      head = f"at least {self.lo}"
      if self.lo_open:
        head = "positive" if self.lo == 0 else f"greater than {self.lo}"
      return f"{head} and finite" if self.hi_open else head
    return f"in {'(' if self.lo_open else '['}{self.lo}, {self.hi}{')' if self.hi_open else ']'}"


def setting[T](default: T, **limits: Unpack[Limits]) -> T:
  """A compile-time (C) numeric field that `validate` checks against `limits`."""
  return dataclasses.field(default=default, metadata={"runtime": False, "rebuild": None, "limits": ValueRange.of(limits)})


def runtime_field[T](default: T, rebuild: Rebuild | None = None, **limits: Unpack[Limits]) -> T:
  """A runtime (R) field: changing it never recompiles (`rebuild`: what it rebuilds instead);
  numeric `limits` as for `setting`."""
  return dataclasses.field(default=default, metadata={"runtime": True, "rebuild": rebuild, "limits": ValueRange.of(limits)})


def rebuild_field[T](default: T, rebuild: Rebuild) -> T:
  """A compile-time (C) field whose change also rebuilds `rebuild`."""
  return dataclasses.field(default=default, metadata={"runtime": False, "rebuild": rebuild, "limits": None})


# How `validate` checks a value: "tri" True / False / "auto", "choice" one of `choices`, "bool",
# "int" (not a bool), "float" (any real but a bool); "other": the group's own checks.
type _Kind = Literal["tri", "choice", "bool", "int", "float", "other"]


@final
@dataclasses.dataclass(frozen=True)
class _FieldCheck:
  """What `validate` checks of one field (from its annotation and metadata)."""

  name: str
  kind: _Kind
  choices: tuple[str, ...]
  optional: bool
  limits: ValueRange | None


def _unalias(hint: object) -> object:
  """`hint` with `type X = ...` aliases replaced by their value."""
  while isinstance(hint, typing.TypeAliasType):
    hint = hint.__value__
  return hint


def _field_check(f: dataclasses.Field[object], hint: object) -> _FieldCheck:
  """The check of field `f` annotated `hint`."""
  hint = _unalias(hint)
  members = typing.get_args(hint) if typing.get_origin(hint) in {typing.Union, types.UnionType} else (hint,)
  members = tuple(_unalias(m) for m in members)
  optional = types.NoneType in members
  rest = tuple(m for m in members if m is not types.NoneType)
  literal = tuple(a for m in rest if typing.get_origin(m) is Literal for a in typing.get_args(m))
  scalar = rest[0] if len(rest) == 1 else None
  kind: _Kind = "bool" if scalar is bool else "int" if scalar is int else "float" if scalar is float else "other"
  if literal:
    kind = "tri" if bool in rest else "choice" if len(rest) == 1 else "other"
  return _FieldCheck(f.name, kind, literal, optional, f.metadata.get("limits"))


@functools.cache
def _plan(cls: type) -> tuple[_FieldCheck, ...]:
  """The checks of every field of settings group class `cls`."""
  hints = typing.get_type_hints(cls)
  return tuple(_field_check(f, hints[f.name]) for f in dataclasses.fields(cls))


def _same_kind(value: object, choices: tuple[object, ...]) -> bool:
  """Whether `value` is of the kind of the `Literal` `choices`: an integer (a NumPy one included,
  never a bool or a float equal to one) for integer choices, a string for string choices."""
  if all(isinstance(v, int) for v in choices):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_)
  return isinstance(value, str) or not all(isinstance(v, str) for v in choices)


def _type_error(c: _FieldCheck, name: str, value: object) -> str | None:
  """The message for a value of the wrong type or choice (None: it fits)."""
  is_bool = isinstance(value, bool | np.bool_)
  if c.kind == "tri" and not (is_bool or value == "auto"):
    return f'{name} must be True, False or "auto", got {value!r}'
  if c.kind == "choice" and not (_same_kind(value, c.choices) and value in c.choices):
    return f"{name} must be one of {list(c.choices)}, got {value!r}"
  if c.kind == "bool" and not is_bool:
    return f"{name} must be True or False, got {value!r}"
  if c.kind == "int" and (is_bool or not isinstance(value, numbers.Integral)):
    return f"{name} must be an integer, got {value!r}"
  if c.kind == "float" and (is_bool or not isinstance(value, numbers.Real)):
    return f"{name} must be a number, got {value!r}"
  return None


def validate(group: DataclassInstance, key: str) -> None:
  """Checks every field of settings group `group` (named `key` in `RenderSettings`) against its
  annotation and limits (ValueError naming "key.field"): the type first, then the range. A numpy
  bool is stored as the Python bool (the resolvers compare with `is True` / `is False`)."""
  for c in _plan(type(group)):
    name, value = f"{key}.{c.name}", field_value(group, c.name)
    if value is None and c.optional:
      continue
    if message := _type_error(c, name, value):
      raise ValueError(message)
    if isinstance(value, np.bool_):
      object.__setattr__(group, c.name, bool(value))
    elif isinstance(value, numbers.Integral) and not isinstance(value, int) and c.kind in ("int", "choice"):
      object.__setattr__(group, c.name, int(value))  # a NumPy integer stored as the Python int
    v = float(value) if isinstance(value, numbers.Real) else math.nan
    if c.limits is not None and not c.limits.holds(v):
      raise ValueError(f"{name} must be {c.limits.describe()}, got {value}")


def is_numbers(value: object) -> TypeGuard[Sequence[object] | np.ndarray]:
  """Whether `value` is a sequence of values (a tuple, list or 1-D NumPy array; not a string, not
  a scalar), whose entries the caller checks."""
  if isinstance(value, np.ndarray):
    return value.ndim == 1
  return isinstance(value, Sequence) and not isinstance(value, str)


def is_rgb(value: object) -> TypeGuard[Sequence[float] | np.ndarray[[int], np.dtype[np.floating]]]:
  """Whether `value` is a color: a sequence (tuple, list, (3,) NumPy array) of three finite
  numbers (a NaN would reach every pixel it colors); `rgb_tuple` stores it."""
  if isinstance(value, np.ndarray):
    value = value.tolist() if value.shape == (_RGB,) else None
  if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != _RGB:
    return False
  return all(isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v) for v in value)


def rgb_tuple(value: Sequence[float] | np.ndarray[[int], np.dtype[np.floating]]) -> tuple[float, float, float]:
  """A validated color (`is_rgb`) stored as a tuple of Python floats: a list or an array would
  make the settings unhashable (the kernel-tier memo keys on them) and compare unequal to its
  tuple."""
  r, g, b = (float(v) for v in value)
  return r, g, b


def is_display_rgb(value: object) -> bool:
  """Whether `value` is a display (sRGB-encoded) color: an rgb triple (`is_rgb`) in [0, 1]."""
  return is_rgb(value) and all(0.0 <= v <= 1.0 for v in rgb_tuple(value))


def exposure_from_ev100(ev100: float) -> float:
  """USD/RTX linear exposure scale for EV100: 2^-EV100, independent of the tone mapper."""
  return math.pow(2.0, -ev100)
