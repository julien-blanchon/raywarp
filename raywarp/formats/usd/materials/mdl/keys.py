"""What a bake of a `Dag` expression depends on: its structure as a stable key, the texture
files below it, and its lookups (the bake's domain)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, Final, final

from raywarp.formats.usd.materials.mdl import ops
from raywarp.formats.usd.materials.mdl.bake_eval import leaves
from raywarp.formats.usd.materials.mdl.dag import COMPOUNDS, LOOKUPS, MAX_NODES, RESOURCES, Dag
from raywarp.formats.usd.materials.mdl.values import (
  WORLD_SET,
  Lin,
  Node,
  TexDecl,
  TexTerm,
  UnsupportedError,
  UvMap,
  UvTerm,
  Value,
  comps,
)
from raywarp.scene.images.colorspace import ColorSpace

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.mdl import sdk_types as st


def files(dag: Dag, path: str = "") -> set[str]:
  """The files of every texture constant below `path` (the whole material for ''), read or
  not: a bake's and a conversion's dependencies."""
  e = dag.material.get_body() if not path else dag.expr(path)
  out: set[str] = set()
  for v in [] if e is None else _constants(dag, e):
    out |= _value_files(dag, v)
  return {f for f in out if f}


def _constants(dag: Dag, e: st.IExpression) -> Iterator[st.IValue]:
  """The constants below `e` (temporaries once)."""
  seen: set[int] = set()
  todo = [e]
  while todo:
    x = todo.pop()
    index = dag.temporary(x)
    if index is not None:
      if index in seen:
        continue
      seen.add(index)
    x = dag.resolve(x)
    c = dag.call(x)
    if c is not None:
      todo.extend(c[1].get_expression(i) for i in range(c[1].get_size()))
    elif x.get_kind().name == "EK_CONSTANT":
      yield x.get_interface(dag.api.bindings.IExpression_constant).get_value()


def _value_files(dag: Dag, v: st.IValue) -> set[str]:
  """The texture files of a constant (through arrays and structs)."""
  kind = v.get_kind().name
  if kind == "VK_TEXTURE":
    name = v.get_interface(dag.api.bindings.IValue_texture).get_value()
    return set(dag.tiles(name))
  if kind in COMPOUNDS:
    c = v.get_interface(dag.api.bindings.IValue_compound)
    return {f for i in range(c.get_size()) for f in _value_files(dag, c.get_value(i))}
  return set()


def key(dag: Dag, path: str) -> str:
  """The structure of the expression at `path` as text: call names and constants (textures as
  files), temporaries once; stable across SDK sessions (the SDK hashes DB tags). '' when too
  large."""
  e = dag.expr(path)
  parts: list[str] = []
  numbers: dict[int, int] = {}
  todo: list[st.IExpression | str] = [] if e is None else [e]
  while todo and len(parts) <= MAX_NODES:
    x = todo.pop()
    if isinstance(x, str):
      parts.append(x)
      continue
    index = dag.temporary(x)
    if index is not None and index in numbers:
      parts.append(f"#{numbers[index]}")
      continue
    if index is not None:
      numbers[index] = len(numbers)
    x = dag.resolve(x)
    c = dag.call(x)
    if c is not None:
      _, args = c
      parts.append(f"{x.get_interface(dag.api.bindings.IExpression_direct_call).get_definition()}(")
      todo.append(")")
      todo.extend(args.get_expression(i) for i in reversed(range(args.get_size())))
    elif x.get_kind().name == "EK_CONSTANT":
      parts.append(_key_value(dag, x.get_interface(dag.api.bindings.IExpression_constant).get_value()))
  return " ".join(parts) if len(parts) <= MAX_NODES else ""


def _key_value(dag: Dag, v: st.IValue) -> str:
  """A constant as text: numbers, enum names, strings, compounds element by element, resources
  by their files (a texture: its tiles and color space)."""
  b = dag.api.bindings
  kind = v.get_kind().name
  if kind in COMPOUNDS:
    c = v.get_interface(b.IValue_compound)
    return "[" + ",".join(_key_value(dag, c.get_value(i)) for i in range(c.get_size())) + "]"
  if kind == "VK_TEXTURE":
    name = v.get_interface(b.IValue_texture).get_value()
    decl = dag.declaration(name)[0]
    return f"tex({','.join(dag.tiles(name))};{decl.colorspace})"
  if kind in RESOURCES:
    r = v.get_interface(b.IValue_resource)
    return f"{kind}({r.get_file_path() or ''};{r.get_value() or ''})"
  try:
    value = dag.ivalue(v)
  except UnsupportedError:
    return kind
  return repr([x.c if isinstance(x, Lin) else x for x in comps(value)] if isinstance(value, (Lin, tuple)) else value)


def lookups(dag: Dag, path: str) -> Lookups:
  """The lookups of the expression at `path`: read from its value, else from its calls (maps
  None when a lookup's coordinates are not affine)."""
  try:
    keys = leaves(comps(dag.value(path)))
  except UnsupportedError:
    return _walked(dag, path)
  terms = [k for k in keys if isinstance(k, TexTerm)]
  sizes = tuple(side for decl, side in dag.read() if any(t.tex == decl for t in terms))
  spaces = frozenset(t.tex.colorspace for t in terms if t.tex.colorspace is not None)
  return Lookups(frozenset(t.uv for t in terms), sizes, uv_read=any(isinstance(k, UvTerm) for k in keys), spaces=spaces)


def _reads_position(value: Value) -> bool:
  """Whether a value reads the position: a world coordinate term in it (`leaves` raises on one)."""
  todo: list[object] = list(comps(value))
  seen: set[int] = set()
  while todo:
    x = todo.pop()
    if id(x) in seen:
      continue
    seen.add(id(x))
    if isinstance(x, Lin) and any(isinstance(k, UvTerm) and k.uv_set == WORLD_SET for k in x.terms):
      return True
    if isinstance(x, Node):
      todo.extend(x.args)
  return False


# The `base::` call whose coordinate system argument may select the world or object position.
_COORDINATE_SOURCE: Final = "base::coordinate_source"


def _source_reads_position(dag: Dag, path: str) -> bool:
  """Whether the `base::coordinate_source` call at `path` reads the position (its system is the
  world or the object; an unreadable call: unknown, False)."""
  try:
    return _reads_position(dag.value(path))
  except UnsupportedError:
    return False


def _walked(dag: Dag, path: str) -> Lookups:
  """`lookups` of an expression `value` does not read: its lookup calls' textures and maps."""
  e = dag.expr(path)
  maps: set[UvMap] | None = set()
  sizes: list[int] = []
  spaces: set[ColorSpace] = set()
  uv_read = position = False
  try:
    for name, args, p in [] if e is None else dag.walk(e, path, lambda n: n not in LOOKUPS):
      uv_read = uv_read or name in ("state::texture_coordinate", "state::position")
      position = position or name == "state::position" or (name == _COORDINATE_SOURCE and _source_reads_position(dag, p))
      if name not in LOOKUPS or args.get_size() < 2:
        continue
      tex = dag.value(f"{p}.{args.get_name(0)}")
      sizes += [side for decl, side in dag.read() if decl == tex]
      if isinstance(tex, TexDecl) and tex.colorspace is not None:
        spaces.add(tex.colorspace)
      try:
        coord = dag.value(f"{p}.{args.get_name(1)}")
        position = position or _reads_position(coord)
        maps = None if maps is None else {*maps, ops.coord_map(coord)}
      except UnsupportedError:
        maps = None
  except UnsupportedError:
    return Lookups(None, tuple(sizes), uv_read=True, spaces=frozenset(spaces), position=position)
  found = None if maps is None else frozenset(maps)
  return Lookups(found, tuple(sizes), uv_read=uv_read, spaces=frozenset(spaces), position=position)


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Lookups:
  """What a bake's domain needs: the lookups' coordinate maps (None: not all affine), the
  textures' largest sides, whether texture coordinates are read outside a lookup, the textures'
  color spaces, whether the position is read (`state::position`, a lookup through world
  coordinates)."""

  maps: frozenset[UvMap] | None
  sizes: tuple[int, ...]
  uv_read: bool
  spaces: frozenset[ColorSpace] = frozenset()
  position: bool = False
