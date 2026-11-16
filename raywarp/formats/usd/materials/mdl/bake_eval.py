"""Walks of `values` expressions for the baker: their sources, per-texel values and identities.

Expressions are DAGs: one `Node` can feed several others (an MDL expression read by two, the
components of one color). Every walk visits each node once (memoised by identity within the
walk), so the cost is linear in the distinct nodes, not in the expanded tree.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, final

import numpy as np

from raywarp.formats.usd.materials.mdl.values import (
  CMP,
  FUNCS,
  WORLD_SET,
  Lin,
  Node,
  Scalar,
  Source,
  TexTerm,
  UnsupportedError,
  UvTerm,
)

type Plane = np.ndarray[[int, int], np.dtype[np.float64]]
# Texture lookup or mesh texture coordinate -> its per-texel values on the bake grid.
type TexelPlanes = dict[Source, Plane]

# Divisors below this are clamped (an MDL division by a texel that is 0).
TINY_DIVISOR: Final = 1e-30
# Expanded expression trees up to this many nodes are keyed by their nested tuple (`node_key`);
# larger ones by their DAG (the tuple would grow exponentially with shared subtrees).
TREE_KEY_NODES: Final = 4096
_ARITHMETIC: Final = frozenset({"+", "-", "*", "/"})


def leaves(cs: Iterable[Scalar | str]) -> set[Source]:
  """Texture lookups and mesh texture coordinates of scalars (UnsupportedError when one reads the
  world position)."""
  out: set[Source] = set()
  seen: set[int] = set()
  todo: list[Scalar | str] = list(cs)
  while todo:
    x = todo.pop()
    if id(x) in seen:
      continue
    seen.add(id(x))
    if isinstance(x, Lin):
      for key in x.terms:
        if isinstance(key, UvTerm) and key.uv_set == WORLD_SET:  # (the MDL reports' wording)
          raise UnsupportedError("baked output depends on uv")
        out.add(key)
    elif isinstance(x, Node):
      todo.extend(x.args)
  return out


@final
class Numeric:
  """Per-texel values of scalars over one grid, each node computed once."""

  def __init__(self, planes: TexelPlanes, shape: tuple[int, int]) -> None:
    """`planes`: the sources' values; `shape`: the grid."""
    self.planes = planes
    self.shape = shape
    self.memo: dict[int, Plane] = {}

  def __call__(self, x: Scalar | str) -> Plane:
    """The value of `x` (memoised by node identity)."""
    if isinstance(x, Node) and id(x) in self.memo:
      return self.memo[id(x)]
    if isinstance(x, Lin):
      v = np.full(self.shape, x.c, np.float64)
      for key, coeff in x.terms.items():
        v = v + coeff * self.planes[key]
      return v
    if not isinstance(x, Node):
      raise UnsupportedError(f"not a number: {type(x).__name__}")
    value = self._node(x)
    self.memo[id(x)] = value
    return value

  def _node(self, x: Node) -> Plane:
    """The value of an expression node."""
    if x.op == "select":
      cmp, d, a, b = x.args
      return np.where(CMP[str(cmp)](self(d), 0.0), self(a), self(b))
    if x.op in _ARITHMETIC:
      a, b = (self(arg) for arg in x.args)
      if x.op == "+":
        return a + b
      if x.op == "-":
        return a - b
      if x.op == "*":
        return a * b
      return a / np.where(np.abs(b) > TINY_DIVISOR, b, TINY_DIVISOR)
    # str operands are a function's integer / enum parameters, passed as is.
    return FUNCS[x.op](*(arg if isinstance(arg, str) else self(arg) for arg in x.args))


def _lin_key(x: Lin) -> tuple[object, ...]:
  """The identity of an affine scalar (with texture coordinates only when it reads them)."""
  terms = sorted(
    ((k.tex.path, str(k.tex.colorspace), k.tex.cube, k.channel, k.uv.uv_set, k.uv.rows, k.wrap), v)
    for k, v in x.terms.items()
    if isinstance(k, TexTerm)
  )
  uvs = sorted(((k.uv_set, k.axis), v) for k, v in x.terms.items() if isinstance(k, UvTerm))
  return ("lin", x.c, tuple(terms), tuple(uvs)) if uvs else ("lin", x.c, tuple(terms))


def _tree_size(x: Scalar | str, memo: dict[int, int]) -> int:
  """Nodes of the expanded expression tree of `x` (computed over the DAG)."""
  if not isinstance(x, Node):
    return 1
  if id(x) not in memo:
    memo[id(x)] = 1 + sum(_tree_size(a, memo) for a in x.args)
  return memo[id(x)]


def _tree_key(x: Scalar | str, memo: dict[int, object]) -> object:
  """The nested-tuple identity of `x` (shared nodes give one shared tuple)."""
  if isinstance(x, Lin):
    return _lin_key(x)
  if not isinstance(x, Node):
    return x
  if id(x) not in memo:
    memo[id(x)] = (x.op, *(a if isinstance(a, str) else _tree_key(a, memo) for a in x.args))
  return memo[id(x)]


def _dag_key(x: Scalar | str, memo: dict[int, int], rows: list[object]) -> int | object:
  """`x` as a row index of `rows` (nodes, in dependency order), or its leaf identity."""
  if isinstance(x, Lin):
    return _lin_key(x)
  if not isinstance(x, Node):
    return x
  if id(x) not in memo:
    args = tuple(a if isinstance(a, str) else _dag_key(a, memo, rows) for a in x.args)
    rows.append((x.op, args))
    memo[id(x)] = len(rows) - 1
  return memo[id(x)]


def node_key(x: Scalar | str) -> object:
  """A hashable, file-independent identity of a scalar's expression: the nested tuple of its
  tree, or ("dag", rows) when the tree expands past TREE_KEY_NODES."""
  if _tree_size(x, {}) <= TREE_KEY_NODES:
    return _tree_key(x, {})
  rows: list[object] = []
  root = _dag_key(x, {}, rows)
  return ("dag", tuple(rows), root)
