"""Symbolic shading values: affine texel sums, expression trees, normals, textures (the MDL SDK
path's reading of distilled expressions).

`Lin` is affine in the texels (sum(coeff * texel channel) + c: what a texture slot stores), `Node`
an expression tree baked per texel; arithmetic keeps values affine where it can (`binop`,
`nary`, folded on constants).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from types import MappingProxyType
from typing import Final, final

import numpy as np

from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model.types import CHANNEL_RGB

# Components of a texture coordinate.
UV_DIMS: Final = 2


class UnsupportedError(Exception):
  """A construct the evaluators cannot represent: an MDL expression or a MaterialX node."""


@final
@dataclasses.dataclass(frozen=True)
class TexDecl:
  """A texture value: resolved file path ('' = none), color space (None = by slot), cube map."""

  path: str
  colorspace: ColorSpace | None = None
  cube: bool = False


@final
@dataclasses.dataclass(frozen=True)
class UvMap:
  """An affine texture-coordinate map of mesh UV set `uv_set`: rows (a, b, c), (d, e, f)."""

  uv_set: int
  rows: tuple[float, float, float, float, float, float]

  def xform(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The map as TextureRef.uv_xform rows."""
    a, b, c, d, e, f = self.rows
    return (a, b, c), (d, e, f)


@final
@dataclasses.dataclass(frozen=True)
class TexTerm:
  """A texel channel looked up through a UV map."""

  tex: TexDecl
  channel: int
  uv: UvMap
  wrap: tuple[int, int]


# UvTerm / UvMap set of the hit's world position (axis 0-2: scene x, y, z in metres); a slot
# reads it as TextureRef.uv_set UV_SET_WORLD.
WORLD_SET: Final = -1


@final
@dataclasses.dataclass(frozen=True)
class UvTerm:
  """A mesh texture coordinate (uv set, axis 0 / 1), or a world coordinate (WORLD_SET, 0-2)."""

  uv_set: int
  axis: int


type Source = TexTerm | UvTerm


@final
class Lin:
  """An affine scalar sum(coeff * source) + c."""

  __slots__ = ("c", "terms")

  def __init__(self, terms: dict[Source, float] | None = None, c: float = 0.0) -> None:
    """`terms`: source -> coefficient; `c`: the constant."""
    self.terms = terms or {}
    self.c = float(c)

  @property
  def const(self) -> bool:
    """Whether there is no texture / uv term."""
    return not self.terms

  def add(self, other: Lin, sign: float = 1.0) -> Lin:
    """Sum with `other` times `sign`."""
    terms = dict(self.terms)
    for k, v in other.terms.items():
      terms[k] = terms.get(k, 0.0) + sign * v
    return Lin({k: v for k, v in terms.items() if v != 0.0}, self.c + sign * other.c)

  def scale(self, k: float) -> Lin:
    """Product with the constant `k`."""
    return Lin({key: v * k for key, v in self.terms.items()} if k != 0.0 else {}, self.c * k)


@final
class Node:
  """A scalar that is not affine in the texels: `op` over `args` (baked per texel)."""

  __slots__ = ("args", "op")

  def __init__(self, op: str, *args: Scalar | str) -> None:
    """`args`: Lin / Node scalars (and a comparison token first for "select")."""
    self.op = op
    self.args = args


type Scalar = Lin | Node


@final
@dataclasses.dataclass
class Normal:
  """An unpacked normal-map lookup (`::unpack_normal_map`, MaterialX `normalmap`): the
  tangent-space components 2 t - 1 and the XY strength they are scaled by (MDL: 1)."""

  comps: tuple[Scalar, ...]
  scale: tuple[float, float] = (1.0, 1.0)


type Value = Lin | Node | Normal | TexDecl | str | tuple[Scalar, ...]


def comps(v: Value) -> tuple[Scalar, ...]:
  """The scalar components of a number or vector value."""
  if isinstance(v, (Lin, Node)):
    return (v,)
  if isinstance(v, Normal):
    return v.comps
  if isinstance(v, tuple):
    return v
  raise UnsupportedError(f"not a number: {type(v).__name__}")


def pack(cs: list[Scalar] | tuple[Scalar, ...]) -> Value:
  """A scalar for one component, else a vector."""
  return cs[0] if len(cs) == 1 else tuple(cs)


def is_const(x: Scalar) -> bool:
  """Whether `x` is a constant."""
  return isinstance(x, Lin) and x.const


def const_value(x: Scalar) -> float:
  """The value of a constant scalar."""
  if isinstance(x, Lin) and x.const:
    return x.c
  raise UnsupportedError("not a constant")


def lin_range(x: Lin) -> tuple[float, float] | None:
  """Bounds of an affine scalar over texels in [0, 1] (None if it depends on the uv)."""
  lo = hi = x.c
  for key, coeff in x.terms.items():
    if not isinstance(key, TexTerm):
      return None
    lo, hi = lo + min(coeff, 0.0), hi + max(coeff, 0.0)
  return lo, hi


def broadcast(a: Value, b: Value) -> tuple[tuple[Scalar, ...], tuple[Scalar, ...]]:
  """Components of `a` and `b` broadcast to the same size."""
  ca, cb = comps(a), comps(b)
  n = max(len(ca), len(cb))
  if len(ca) not in (1, n) or len(cb) not in (1, n):
    raise UnsupportedError("vector size mismatch")
  return ca * (n // len(ca)), cb * (n // len(cb))


def _binop1(op: str, x: Scalar, y: Scalar) -> Scalar:
  """`x op y` of two scalars, affine when it can be."""
  if op in ("+", "-"):
    if isinstance(x, Lin) and isinstance(y, Lin):
      return x.add(y, 1.0 if op == "+" else -1.0)
    return Node(op, x, y)
  if op == "*":
    if isinstance(x, Lin) and x.const and isinstance(y, Lin):
      return y.scale(x.c)
    if isinstance(y, Lin) and y.const and isinstance(x, Lin):
      return x.scale(y.c)
    return Node("*", x, y)
  if op == "/":
    if isinstance(y, Lin) and y.const:
      if y.c == 0.0:
        raise UnsupportedError("division by zero")
      if isinstance(x, Lin):
        return x.scale(1.0 / y.c)
    return Node("/", x, y)
  raise UnsupportedError(f"operator {op}")


def binop(op: str, a: Value, b: Value) -> Value:
  """Elementwise `a op b` (+, -, *, /) with broadcasting."""
  ca, cb = broadcast(a, b)
  return pack([_binop1(op, x, y) for x, y in zip(ca, cb, strict=True)])


# Per-texel planes (or 0-d arrays when folding constants) of the numeric functions.
type Numbers = np.ndarray[[int, int], np.dtype[np.float64]]


def _smoothstep(a: Numbers, b: Numbers, x: Numbers) -> Numbers:
  """math::smoothstep."""
  t = np.clip((x - a) / np.where(b != a, b - a, 1e-30), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


# The per-texel functions of expression nodes by name (str operands first: integer / enum
# parameters): the MDL `math::` functions.
FUNCS: Final = MappingProxyType[str, Callable[..., Numbers]](
  {
    "math::max": np.maximum,
    "math::min": np.minimum,
    # A negative base is clamped to 0: pow is undefined there in MDL and GLSL, and a real result
    # for any exponent keeps NaN out of the bake.
    "math::pow": lambda a, b: np.power(np.maximum(a, 0.0), b),
    "math::sqrt": lambda a: np.sqrt(np.maximum(a, 0.0)),
    "math::abs": np.abs,
    "math::floor": np.floor,
    "math::ceil": np.ceil,
    "math::frac": lambda a: a - np.floor(a),
    "math::exp": np.exp,
    "math::log": lambda a: np.log(np.maximum(a, 1e-30)),
    "math::sin": np.sin,
    "math::cos": np.cos,
    "math::saturate": lambda a: np.clip(a, 0.0, 1.0),
    "math::smoothstep": _smoothstep,
    "math::step": lambda a, b: (b >= a).astype(np.float64),
  }
)
CMP: Final = MappingProxyType(
  {
    "<": np.less,
    ">": np.greater,
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
  }
)


def nary(fn: str, *vs: Value) -> Value:
  """Elementwise function `fn` (of `FUNCS`) of values: folded on constants, else a Node."""
  cs = [comps(v) for v in vs]
  n = max(len(c) for c in cs)
  if any(len(c) not in (1, n) for c in cs):
    raise UnsupportedError("vector size mismatch")
  impl = FUNCS[fn]
  out: list[Scalar] = []
  for xs in zip(*[c * (n // len(c)) for c in cs], strict=True):
    if all(is_const(x) for x in xs):
      out.append(Lin(c=float(np.asarray(impl(*(np.asarray(const_value(x), np.float64) for x in xs)), np.float64))))
    else:
      out.append(Node(fn, *xs))
  return pack(out)


def uv_map(v: Value) -> UvMap:
  """The affine UV map of a texture-coordinate value (UnsupportedError when it is not one)."""
  cs = comps(v)
  if len(cs) < UV_DIMS:
    raise UnsupportedError("scalar texture coordinate")
  rows: list[float] = []
  uv_set = None
  for comp in cs[:2]:
    if not isinstance(comp, Lin):
      raise UnsupportedError("non-affine texture coordinate")
    row = [0.0, 0.0, comp.c]
    for key, coeff in comp.terms.items():
      if not isinstance(key, UvTerm):
        raise UnsupportedError("texture coordinate from a texture")
      uv_set = key.uv_set if uv_set is None else uv_set
      if key.uv_set != uv_set:
        raise UnsupportedError("mixed uv sets")
      if key.axis > 1:  # row[2] is the constant: a world z term has no column of its own
        raise UnsupportedError("lookup through the world z axis")
      row[key.axis] = coeff
    rows += row
  a, b, c, d, e, f = rows
  return UvMap(uv_set or 0, (a, b, c, d, e, f))


# Slot values: what a texture slot stores ---------------------------------------------------

type Vec4 = tuple[float, float, float, float]
# Components of an RGB value.
RGB: Final = 3


class NeedsBakeError(Exception):
  """An output that is not one texture channel's affine function: baked per texel."""

  def __init__(self, cs: tuple[Scalar, ...]) -> None:
    """`cs`: the output's components."""
    super().__init__("needs a baked texture")
    self.comps = cs


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class SlotValue:
  """What a slot stores: a constant, or `texel * scale + bias` of one texture (channel or RGB)."""

  const: tuple[float, ...] | None = None
  tex: TexDecl | None = None
  uv: UvMap | None = None
  wrap: tuple[int, int] = (0, 0)
  channel: int = CHANNEL_RGB
  scale: tuple[float, ...] = ()
  bias: tuple[float, ...] = ()


def analyse(value: Value, n: int) -> SlotValue:
  """The SlotValue of an `n`-component output (NeedsBakeError / UnsupportedError when it is
  none)."""
  cs = list(comps(value))
  if len(cs) == 1 and n > 1:
    cs = cs * n
  cs = cs[:n]
  lins = [c for c in cs if isinstance(c, Lin)]
  if len(lins) == len(cs) and all(c.const for c in lins):
    return SlotValue(const=tuple(c.c for c in lins))
  if len(lins) != len(cs):
    raise NeedsBakeError(tuple(cs))
  terms = [k for c in lins for k in c.terms]
  if not all(isinstance(k, TexTerm) for k in terms):
    raise UnsupportedError("output depends on uv")
  sources = {(k.tex, k.uv, k.wrap) for k in terms if isinstance(k, TexTerm)}
  if len(sources) != 1 or any(len(c.terms) > 1 for c in lins):
    raise NeedsBakeError(tuple(cs))
  tex, uv, wrap = next(iter(sources))
  channels = [next((k.channel for k in c.terms if isinstance(k, TexTerm)), None) for c in lins]
  used = {ch for ch in channels if ch is not None}
  if len(used) == 1 and (n == 1 or not all(ch == i for i, ch in enumerate(channels) if ch is not None)):
    channel = used.pop()
  elif all(ch is None or ch == i for i, ch in enumerate(channels)):
    channel = CHANNEL_RGB
  else:
    raise NeedsBakeError(tuple(cs))
  scale = tuple(next(iter(c.terms.values())) if c.terms else 0.0 for c in lins)
  return SlotValue(tex=tex, uv=uv, wrap=wrap, channel=channel, scale=scale, bias=tuple(c.c for c in lins))


def pad4(t: tuple[float, ...], fill: float) -> Vec4:
  """A 1-4 component scale / bias as 4 (the last of 1-2 components repeated to 3, `fill` 4th)."""
  t = tuple(t) + (t[-1],) * (RGB - len(t)) if len(t) < RGB else tuple(t)
  return t[0], t[1], t[2], t[3] if len(t) > RGB else fill
