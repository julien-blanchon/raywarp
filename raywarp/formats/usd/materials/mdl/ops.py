"""The arithmetic of `Dag` values: MDL operators, constructors, `math::` functions and the
`base::` / distiller helpers over `values` (affine where they can be; UnsupportedError for a
call outside the subset)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from raywarp.formats.usd.materials.mdl.bake_eval import leaves, node_key
from raywarp.formats.usd.materials.mdl.values import (
  FUNCS,
  WORLD_SET,
  Lin,
  Node,
  Scalar,
  TexDecl,
  TexTerm,
  UnsupportedError,
  UvMap,
  UvTerm,
  Value,
  binop,
  broadcast,
  comps,
  const_value,
  is_const,
  lin_range,
  nary,
  pack,
  uv_map,
)

_SIZES: Final = {"float": 1, "float2": 2, "float3": 3, "float4": 4, "color": 3}
_MATRIX4: Final = 16


def coord_map(coord: Value) -> UvMap:
  """The affine map of lookup coordinates (mesh UVs, or the world x, y; not the world z)."""
  if any(k.axis > 1 for c in comps(coord)[:2] if isinstance(c, Lin) for k in c.terms if isinstance(k, UvTerm)):
    raise UnsupportedError("lookup through the world z axis")
  return uv_map(coord)


def apply(name: str, vs: list[Value], floats: frozenset[TexDecl] = frozenset()) -> Value:
  """`name` applied to argument values (UnsupportedError for a call outside the subset);
  `floats`: textures of float pixels (their texel sums are not bounded by [0, 1])."""
  if name in ("operator*", "operator+") or (name == "operator-" and len(vs) == 2):
    return pack([_fold(name[-1], x, y) for x, y in zip(*broadcast(vs[0], vs[1]), strict=True)])
  if name == "operator/":
    return binop("/", vs[0], vs[1])
  if name == "math::sqrt" and all(square_of(c, floats) is not None for c in comps(vs[0])):
    return pack([r for c in comps(vs[0]) if (r := square_of(c, floats)) is not None])
  if name == "operator-":
    return binop("*", vs[0], Lin(c=-1.0))
  if name == "operator[]":
    return comps(vs[0])[int(const_value(comps(vs[1])[0]))]
  if name in _SIZES:  # constructors and conversions
    cs = [x for v in vs for x in comps(v)]
    cs = cs * _SIZES[name] if len(cs) == 1 else cs
    if len(cs) < _SIZES[name]:
      raise UnsupportedError(f"{name} of {len(cs)} components")
    return pack(cs[: _SIZES[name]])
  if name == "math::saturate":
    inside = all((r := _range(c, floats)) is not None and r[0] >= 0.0 and r[1] <= 1.0 for c in comps(vs[0]))
    return vs[0] if inside else nary(name, vs[0])  # the identity on 8-bit texels' affine sums
  if name in ("math::min", "math::max", "math::clamp"):
    low = _clamp("math::max", vs[:2], floats) if name == "math::clamp" else None
    clamped = _clamp(name, vs, floats) if name != "math::clamp" else _clamp("math::min", [low, vs[2]], floats)
    if clamped is not None:
      return clamped
  if name in FUNCS and all(not isinstance(v, (str, TexDecl)) for v in vs):
    return nary(name, *vs)
  if name in ("math::lerp", "math::dot"):
    return _lerp(*vs) if name == "math::lerp" else _sum(binop("*", vs[0], vs[1]))
  handler = _HELPERS.get(name)
  if handler is None:
    raise UnsupportedError(f"call {name}")
  return handler(vs)


def rnm(nd: tuple[Scalar, ...], n: tuple[Scalar, ...]) -> Value:
  """Reoriented normal mapping of tangent-space detail `nd` over base `n` (MDL's
  `add_detail_normal` before its normalisation): t = n + (0, 0, 1), u = nd x (-1, -1, 1),
  t dot(t, u) / t.z - u."""
  t = (n[0], n[1], comps(binop("+", n[2], Lin(c=1.0)))[0])
  u = (comps(binop("*", nd[0], Lin(c=-1.0)))[0], comps(binop("*", nd[1], Lin(c=-1.0)))[0], nd[2])
  k = binop("/", _sum(binop("*", t, u)), t[2])
  return tuple(comps(binop("-", binop("*", a, k), b))[0] for a, b in zip(t, u, strict=True))


def _fold(op: str, x: Scalar, y: Scalar) -> Scalar:
  """`x op y` (+, -, *) with 0 x a = 0 and a +- 0 = a (the zero entries of tangent frames, a
  regular normal map's roughness 0)."""
  if op == "*" and (is_zero(x) or is_zero(y)):
    return Lin()
  if is_zero(y):
    return x
  if op == "+" and is_zero(x):
    return y
  return comps(binop(op, x, y))[0]


def square_of(x: Scalar, floats: frozenset[TexDecl]) -> Scalar | None:
  """The root a of x = a x a with a >= 0 (an affine sum of 8-bit texels or a constant), else
  None."""
  if not isinstance(x, Node) or x.op != "*" or len(x.args) != 2:  # a binary product
    return None
  a, b = x.args
  if not isinstance(a, Lin) or not isinstance(b, Lin) or node_key(a) != node_key(b):
    return None
  r = _range(a, floats)
  return a if r is not None and r[0] >= 0.0 else None


def stored_z(xy: Value) -> Value:
  """A normal map's z = sqrt(1 - x^2 - y^2) from its x, y channels: its stored z channel (the
  unit normals of 8-bit RGB normal maps); UnsupportedError for any other root."""
  terms = leaves(comps(xy))
  maps = {(t.tex, t.uv, t.wrap) for t in terms if isinstance(t, TexTerm) and t.channel in (0, 1)}
  if len(maps) != 1 or len(terms) != sum(isinstance(t, TexTerm) for t in terms):
    raise UnsupportedError("a square root in a normal")
  tex, uv, wrap = maps.pop()
  return Lin({TexTerm(tex, 2, uv, wrap): 2.0}, c=-1.0)


def is_zero(v: Value) -> bool:
  """Whether `v` is the constant 0."""
  return all(is_const(c) and const_value(c) == 0.0 for c in comps(v))


def _clamp(name: str, vs: Sequence[Value | None], floats: frozenset[TexDecl]) -> Value | None:
  """Min / max of an affine sum of 8-bit texels (or a product of two) and a constant outside its
  range: the sum, else None (also for a None argument)."""
  for x, bound in ((vs[0], vs[1]), (vs[1], vs[0])):
    if x is None or bound is None:
      return None
    bs = comps(bound)
    ranges = [_range(c, floats) for c in comps(x)]
    if len(bs) != 1 or not is_const(bs[0]) or any(r is None for r in ranges):
      continue
    b = const_value(bs[0])
    if all(r is not None and (r[1] <= b if name == "math::min" else r[0] >= b) for r in ranges):
      return x
  return None


def _range(x: Scalar, floats: frozenset[TexDecl]) -> tuple[float, float] | None:
  """Bounds of an affine sum of 8-bit texels (in [0, 1]) or of a product of two (None: unknown;
  float texels are unbounded)."""
  if isinstance(x, Lin):
    return None if any(isinstance(k, TexTerm) and k.tex in floats for k in x.terms) else lin_range(x)
  if not isinstance(x, Node) or x.op != "*" or len(x.args) != 2 or isinstance(x.args[0], str) or isinstance(x.args[1], str):
    return None
  a, b = _range(x.args[0], floats), _range(x.args[1], floats)
  if a is None or b is None:
    return None
  ends = [p * q for p in a for q in b]
  return min(ends), max(ends)


def _sum(v: Value) -> Value:
  """The sum of a value's components."""
  total: Value = Lin()
  for x in comps(v):
    total = binop("+", total, x)
  return total


def _lerp(a: Value, b: Value, t: Value) -> Value:
  """math::lerp(a, b, t) = a + (b - a) t (affine for a constant t)."""
  return binop("+", a, binop("*", binop("-", b, a), t))


def _blend_colors(vs: list[Value]) -> Value:
  """base::blend_colors(top, bottom, weight, mode) at a constant weight (OmniPBR's modes)."""
  top, bottom, weight, mode = vs
  if not is_const(comps(weight)[0]):
    raise UnsupportedError("blend_colors with a varying weight")
  if const_value(comps(weight)[0]) == 0.0:
    return bottom
  blends = {"color_layer_blend": top, "color_layer_multiply": binop("*", bottom, top)}
  if mode in blends:
    return _lerp(bottom, blends[str(mode)], weight)
  if mode == "color_layer_add":
    return binop("+", bottom, binop("*", top, weight))
  raise UnsupportedError(f"blend_colors mode {mode}")


def _average(vs: list[Value]) -> Value:
  """distilling_support::average(w1, r1, w2, r2, ...) = sum(w r) / sum(w) (equal r: `Dag`)."""
  total: Value = Lin()
  for w, r in zip(vs[::2], vs[1::2], strict=True):
    total = binop("+", total, binop("*", r, w))
  return binop("/", total, _sum(pack([comps(w)[0] for w in vs[::2]])))


def _transform(vs: list[Value]) -> Value:
  """base::transform_coordinate(m, info): info's position x m (column vectors, w = 1)."""
  m, info = comps(vs[0]), comps(vs[1])
  if len(m) != _MATRIX4:
    raise UnsupportedError("transform_coordinate matrix")
  out: list[Scalar] = []
  for row in range(3):
    acc: Value = m[12 + row]
    for col in range(3):
      acc = binop("+", acc, binop("*", info[col], m[4 * col + row]))
    out.append(comps(acc)[0])
  return tuple(out)


def _coordinate_source(vs: list[Value]) -> Value:
  """base::coordinate_source(system, index): texture coordinates or the world / object position."""
  system, index = vs
  if system == "texture_coordinate_uvw":
    i = int(const_value(comps(index)[0]))
    return (Lin({UvTerm(i, 0): 1.0}), Lin({UvTerm(i, 1): 1.0}), Lin())
  if system in ("texture_coordinate_world", "texture_coordinate_object"):
    return tuple(Lin({UvTerm(WORLD_SET, k): 1.0}) for k in range(3))
  raise UnsupportedError(f"coordinate_source {system}")


# texture_coordinate_info (position, tangents) is carried as its position, the only member a
# lookup reads.
_HELPERS: Final = {
  "base::blend_colors": _blend_colors,
  "nvidia::distilling_support::average": _average,
  "base::transform_coordinate": _transform,
  "base::texture_coordinate_info": lambda vs: vs[0],
  "base::texture_coordinate_info.position": lambda vs: vs[0],
  "base::coordinate_source": _coordinate_source,
}
