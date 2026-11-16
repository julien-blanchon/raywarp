"""Settings controls generated from the `RenderSettings` dataclasses, and validated patches.

`schema` lists every leaf field (dotted path, label, control type, value, choices, description from
the group's docstring, change kind); `patch_settings` applies a browser patch of dotted fields by
rebuilding the changed groups, so every group validator runs before a renderer sees the result.
Values are constructed from the field annotations only (no expression or import from a client).
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
import types
import typing
from collections.abc import Mapping
from typing import Final, Literal

from raywarp._base.named import DataclassInstance, field_value, replace_named
from raywarp.settings import RenderSettings

type ChangeKind = Literal["runtime", "rebake", "recompile"]
type Control = Literal["enum", "boolean", "number", "string"]
type Row = dict[str, object]

# Change kind of each "group.field": "runtime" (next frame), "rebake" (runtime, rebuilds a baked
# structure) or "recompile" (compile-time: another kernel build, a prebuilt tier variant or a
# compile), from the fields' own metadata.
CHANGE_KINDS: Final[dict[str, ChangeKind]] = {
  f.name: "recompile" if not f.runtime else "rebake" if f.rebuild is not None else "runtime"
  for f in RenderSettings.fields_table()
}
# Fields a patch may change at most (a browser sends one or a few fields at once).
MAX_PATCH: Final = 256


def change_kind(path: str) -> ChangeKind:
  """The change kind of dotted field `path` (nested leaves inherit their group field's kind).

  Raises:
    KeyError: for a path that is not a settings field.
  """
  field = path
  while field not in CHANGE_KINDS and "." in field:
    field = field.rsplit(".", 1)[0]
  if field not in CHANGE_KINDS:
    raise KeyError(f"not a settings field: {path!r}")
  return CHANGE_KINDS[field]


def plain(value: object) -> object:
  """`value` as JSON-compatible data (dataclasses as objects, tuples as lists)."""
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
  if isinstance(value, tuple | list):
    return [plain(v) for v in value]
  return value


def annotation(value: object) -> object:
  """`value` with the project's named type aliases (`type X = ...`) unwrapped."""
  while isinstance(value, typing.TypeAliasType):
    value = value.__value__
  return value


def descriptions(cls: type) -> dict[str, str]:
  """Field name -> its description in the `Attributes:` section of `cls`'s docstring."""
  result: dict[str, str] = {}
  name = None
  for line in (inspect.getdoc(cls) or "").splitlines():
    match = re.match(r"  (\w+):\s*(.*)", line)
    if match:
      name = match[1]
      result[name] = match[2]
    elif name and line.startswith("    "):
      result[name] += " " + line.strip()
    else:
      name = None
  return result


def enum_options(kind: object) -> list[object]:
  """The choices of a `Literal` / bool / union-of-those annotation (empty: not an enumeration)."""
  kind = annotation(kind)
  origin, args = typing.get_origin(kind), typing.get_args(kind)
  if origin is typing.Literal:
    return list(args)
  if origin in (typing.Union, types.UnionType):
    choices: list[object] = []
    for arg in args:
      if arg is bool:
        choices.extend([False, True])
        continue
      options = enum_options(arg)
      if not options:
        return []
      choices.extend(options)
    return choices
  return []


def _control(value: object, options: list[object]) -> Control:
  """The control type of a leaf value."""
  if options:
    return "enum"
  if isinstance(value, bool):
    return "boolean"
  return "number" if isinstance(value, int | float) else "string"


def _row(group: DataclassInstance, name: str, path: str, hint: object, doc: str) -> Row:
  """The control row of leaf field `name` of `group` (dotted `path`)."""
  value = field_value(group, name)
  options = enum_options(hint)
  structured = value is None or isinstance(value, tuple | list)
  row: Row = {
    "path": path,
    "group": path.split(".", 1)[0],
    "label": name.replace("_", " ").capitalize(),
    "type": _control(value, options),
    "value": json.dumps(plain(value)) if structured else value,
    "description": doc + (" (JSON value)" if structured else ""),
    "change": change_kind(path),
  }
  if options:
    row["options"] = options
  if structured:
    row["encoding"] = "json"
  return row


def schema(settings: RenderSettings) -> list[Row]:
  """One control row per dataclass leaf of `settings` (nested groups walked; structured and
  nullable values are JSON text controls)."""
  rows: list[Row] = []

  def walk(obj: DataclassInstance, prefix: str) -> None:
    """Visits the fields of `obj` (no hand-maintained field list)."""
    hints = typing.get_type_hints(type(obj))
    docs = descriptions(type(obj))
    for field in dataclasses.fields(obj):
      value = field_value(obj, field.name)
      path = prefix + field.name
      if dataclasses.is_dataclass(value) and not isinstance(value, type):
        walk(value, path + ".")
        continue
      doc = docs.get(field.name, f"{type(obj).__name__}.{field.name}")
      rows.append(_row(obj, field.name, path, hints[field.name], doc))

  walk(settings, "")
  return rows


def _convert_union(value: object, args: tuple[object, ...]) -> object:
  """`value` as the first member of a union annotation it converts to."""
  for option in args:
    try:
      return convert(value, option)
    except (TypeError, ValueError):
      continue
  raise ValueError("value does not match any allowed type")


def _convert_tuple(value: object, args: tuple[object, ...]) -> tuple[object, ...]:
  """`value` (a JSON array) as a tuple annotated `tuple[args]`."""
  if not isinstance(value, list | tuple):
    raise ValueError("expected a JSON array")
  if len(args) == 2 and args[1] is Ellipsis:
    return tuple(convert(v, args[0]) for v in value)
  if len(value) != len(args):
    raise ValueError(f"expected {len(args)} values")
  return tuple(convert(v, k) for v, k in zip(value, args, strict=True))


def _convert_dataclass(value: object, kind: type) -> object:
  """`value` (a JSON object) as dataclass `kind`."""
  if not isinstance(value, dict):
    raise ValueError("expected a settings object")
  hints = typing.get_type_hints(kind)
  if set(value) - set(hints):
    raise ValueError("unknown settings fields")
  return kind(**{k: convert(v, hints[k]) for k, v in value.items()})


def convert(value: object, kind: object) -> object:
  """`value` (decoded JSON) as annotation `kind`: only the annotated types are constructed.

  Raises:
    ValueError: for a value of another type, a non-finite number or an unknown choice.
  """
  kind = annotation(kind)
  origin, args = typing.get_origin(kind), typing.get_args(kind)
  if origin in (typing.Union, types.UnionType):
    return _convert_union(value, args)
  if origin is typing.Literal:
    if not any(type(value) is type(v) and value == v for v in args):
      raise ValueError(f"expected one of {args}")
    return value
  if isinstance(kind, type) and dataclasses.is_dataclass(kind):
    return _convert_dataclass(value, kind)
  if origin is tuple:
    return _convert_tuple(value, args)
  if kind is float and isinstance(value, int | float) and type(value) in (int, float) and math.isfinite(value):
    return float(value)
  if kind in (bool, int, str, type(None)) and type(value) is kind:
    return value
  raise ValueError("invalid setting type or non-finite number")


def _decoded(row: Row, value: object) -> object:
  """A browser value of control `row` decoded: "true" / "false" of a bool enum, JSON text of a
  structured field."""
  if row.get("type") == "enum" and isinstance(value, str):
    options = row.get("options")
    if isinstance(options, list) and value not in options and value in ("true", "false"):
      return json.loads(value)
  if row.get("encoding") == "json" and isinstance(value, str):
    return json.loads(value)
  return value


def patch_tree(settings: RenderSettings, patch: object) -> dict[str, object]:
  """A browser patch {dotted path: value} checked against `settings`' controls, as a nested tree.

  Raises:
    ValueError: for an empty or oversized patch, an unknown path or undecodable JSON text.
  """
  if not isinstance(patch, Mapping) or not patch or len(patch) > MAX_PATCH:
    raise ValueError(f"patch must be a nonempty object of at most {MAX_PATCH} dotted settings")
  controls = {str(r["path"]): r for r in schema(settings)}
  tree: dict[str, object] = {}
  for path, value in patch.items():
    if path not in controls:
      raise ValueError(f"unknown setting: {path}")
    branch = tree
    parts = path.split(".")
    for part in parts[:-1]:
      branch = typing.cast("dict[str, object]", branch.setdefault(part, {}))
    branch[parts[-1]] = _decoded(controls[path], value)
  return tree


def _replace(obj: DataclassInstance, updates: dict[str, object]) -> DataclassInstance:
  """`obj` with `updates` (nested dicts for nested dataclasses), rebuilt bottom-up so that every
  changed dataclass's validator runs."""
  hints = typing.get_type_hints(type(obj))
  changes: dict[str, object] = {}
  for key, value in updates.items():
    old = field_value(obj, key)
    if dataclasses.is_dataclass(old) and not isinstance(old, type) and isinstance(value, dict):
      changes[key] = _replace(old, value)
    else:
      changes[key] = convert(value, hints[key])
  return replace_named(obj, changes)


def patch_settings(settings: RenderSettings, patch: object) -> RenderSettings:
  """`settings` with a browser patch {dotted path: value} applied and validated (the renderer
  only ever sees complete, valid settings).

  Raises:
    ValueError: for a malformed patch or an invalid value (the groups' own validation).
    TypeError: from a group constructor given a value of the wrong type.
  """
  result = _replace(settings, patch_tree(settings, patch))
  if not isinstance(result, RenderSettings):
    raise TypeError("a settings patch must give RenderSettings")
  return result


def changed(a: RenderSettings, b: RenderSettings) -> dict[str, object]:
  """{dotted path: value in `b`} of every leaf where `b` differs from `a`."""
  old = {str(r["path"]): r["value"] for r in schema(a)}
  return {str(r["path"]): r["value"] for r in schema(b) if old.get(str(r["path"])) != r["value"]}
