"""Helpers to build, change and read dataclasses whose field names are only known at runtime
(settings paths such as "lighting.light_grid", slot tables).

A type checker types each keyword of a dataclass constructor or of `dataclasses.replace`, which a
mapping whose keys are known only at runtime cannot match. These helpers are the one place
where such keywords pass unchecked: the values are checked by the dataclass's own validation
(`__post_init__`) or by the code that reads the fields.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import ClassVar, Protocol


class DataclassInstance(Protocol):
  """Any dataclass instance (what `dataclasses.replace` accepts)."""

  __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]


def replace_named[D: DataclassInstance](obj: D, changes: Mapping[str, object]) -> D:
  """`dataclasses.replace(obj, **changes)` with field names known only at runtime."""
  rebuild: Callable[..., D] = dataclasses.replace
  return rebuild(obj, **changes)


def field_value(obj: DataclassInstance, name: str) -> object:
  """The value of `obj`'s dataclass field `name` (a name known only at runtime; `obj` has a
  `__dict__`: not a slotted dataclass).

  Raises:
    KeyError: for a name that is not one of `obj`'s fields.
  """
  if name not in obj.__dataclass_fields__:
    raise KeyError(f"{type(obj).__name__} has no field {name!r}")
  return vars(obj)[name]


def construct_named[D: DataclassInstance](cls: type[D], fields: Mapping[str, object], **typed: object) -> D:
  """`cls(**fields, **typed)` with the names of `fields` known only at runtime."""
  make: Callable[..., D] = cls
  return make(**fields, **typed)
