"""Warp's struct API, typed: a `@wp.struct` class is a `warp.codegen.Struct` at run time (field
table `vars`, ctypes mirror `ctype`, `numpy_dtype()`) and its instances `StructInstance`s
(`__ctype__()`), but type checkers see the plain decorated class. These helpers narrow with
Warp's public predicates (`wp.types.type_is_struct` / `is_struct`, `TypeIs` in the stubs).
"""

from __future__ import annotations

import ctypes

import warp as wp


def warp_struct(cls: object):
  """`cls`, a `@wp.struct` class, as Warp's `Struct` (inferred: its module is private).

  Raises:
    TypeError: when `cls` is not a Warp struct type.
  """
  if not wp.types.type_is_struct(cls):
    raise TypeError(f"{cls!r} is not a Warp struct type")
  return cls


def struct_mirror(value: object) -> ctypes.Structure:
  """The ctypes mirror of Warp struct instance `value` (its kernel-argument bytes).

  Raises:
    TypeError: when `value` is not a Warp struct instance.
  """
  if not wp.types.is_struct(value):
    raise TypeError(f"{type(value).__name__} is not a Warp struct instance")
  return value.__ctype__()
