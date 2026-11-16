"""Typed generic Warp functions: PEP 695 type parameters in `@wp.func` signatures.

Warp's only generics syntax is `typing.Any` in a signature (each call site instantiates a
concrete, type-checked copy, like a C++ template), and Warp rejects a `TypeVar`. A type checker,
however, needs a type parameter to infer exact types at call sites. `device_generic` bridges the
two: the function is written with a PEP 695 type parameter, which pyrefly sees through `@wp.func`,
and the decorator replaces that parameter by `Any` in the annotations Warp reads:

    @wp.func
    @device_generic
    def batched[T](table: wp.array2d[T], worldid: int, i: int) -> T:
      return table[worldid % table.shape[0], i]

    v = batched(vec3_table, w, i)  # pyrefly: Vector[float, Literal[3]]; Warp: a vec3 instance

The generated code is identical to a hand-written `wp.array2d[Any]` function. Warp's own
constrained type variables (`wp.Scalar`, `wp.Float`, `wp.Int`) are kept. Limits: pyrefly does not
see Warp's same-name overloading, so variants get distinct names (`batched`, `batched1d`).

String annotations (`from __future__ import annotations`, S1) are evaluated with the function's
globals, the defining scope's locals (a factory's local types, e.g. a vector sized by a factory
argument, which Warp alone would look up among the globals and the closure's variables only) and
its type parameters; a function without type parameters (a factory's kernel naming its local
types) only gets its annotations evaluated.

When Warp accepts type variables as generic placeholders (upstream request), the decorator is
deleted.
"""

from __future__ import annotations

import functools
import inspect
import threading
import types
import typing
from collections.abc import Callable
from types import FunctionType
from typing import Any, Final, Never, TypeVar

import warp as wp

# Warp's own constrained type variables: real Warp generics, left untouched.
_WARP_TYPE_VARS: Final = (wp.Scalar, wp.Float, wp.Int)


def _is_parameter(annotation: object) -> bool:
  """Whether `annotation` is a type parameter of ours (not one of Warp's constrained ones)."""
  return isinstance(annotation, TypeVar) and annotation not in _WARP_TYPE_VARS


def _erase(annotation: object) -> object:
  """Map a type parameter, bare, as an array's dtype or as a tuple member (a return type), to
  Warp's generic placeholder `Any`."""
  if _is_parameter(annotation):
    return Any
  if typing.get_origin(annotation) is tuple:
    return types.GenericAlias(tuple, tuple(_erase(a) for a in typing.get_args(annotation)))
  # Warp's array annotation class is private: its `dtype` / `ndim` are read untyped.
  if wp.types.is_array(annotation) and _is_parameter(getattr(annotation, "dtype", None)):
    ndim: int = getattr(annotation, "ndim")  # noqa: B009
    return wp.array(dtype=Any, ndim=ndim)
  return annotation


def _hints(f: FunctionType, scope: dict[str, object]) -> dict[str, object]:
  """`f`'s annotations evaluated with its globals, `scope` (the defining frame's locals), its
  closure's variables and its type parameters."""
  params = {p.__name__: p for p in f.__type_params__}
  cells = {n: c.cell_contents for n, c in zip(f.__code__.co_freevars, f.__closure__ or (), strict=True)}
  return inspect.get_annotations(f, eval_str=True, locals={**scope, **cells, **params})


def _defining_scope() -> dict[str, object]:
  """The locals of the frame that applied the calling decorator (the factory defining `f`)."""
  frame = inspect.currentframe()
  decorator = frame.f_back if frame is not None else None
  scope = decorator.f_back if decorator is not None else None
  return dict(scope.f_locals) if scope is not None else {}


def device_generic[F: FunctionType](f: F) -> F:
  """Return `f` with its type parameters replaced by `Any` in the annotations Warp reads (string
  annotations evaluated in the defining scope)."""
  hints = _hints(f, _defining_scope())
  f.__annotations__ = {name: _erase(hint) for name, hint in hints.items()}
  return f


def struct_schema[C: type](cls: C) -> C:
  """Marks `cls` as the field declarations of Warp structs built from its annotations (e.g. a
  Protocol whose fields `kernels.abi.rays` turns into `wp.struct`s): the annotations are Warp's own
  (`wp.array2d[...]`), as in any `@wp.struct`, not shaped host types. Returns `cls` unchanged."""
  return cls


def concrete[F: FunctionType](**annotations: object) -> Callable[[F], F]:
  """Decorator giving parameters (or "return") of a Warp function concrete types (Warp reads these
  annotations).

  For kernels whose struct type is built per specialisation (e.g. `trace.software_geometry_type`):
  the source annotates the parameter with the struct's Protocol, which the type checker reads, and
  the factory names the struct type the kernel takes:

      @wp.kernel(...)
      @concrete(geo=software_geometry_type(live_uv=f.live_material_uv))
      def k(geo: SoftwareGeometry): ...
  """

  def decorator(f: F) -> F:
    """`f` with the parameters named in `annotations` retyped."""
    unknown = set(annotations) - set(inspect.signature(f).parameters) - {"return"}
    if unknown:
      raise TypeError(f"{f.__name__} has no parameters {sorted(unknown)}")
    f.__annotations__ = {**f.__annotations__, **annotations}
    return f

  return decorator


def compiled_out(*_args: object) -> Never:
  """Stand-in for a device function a factory does not build, read only under a false
  `wp.static` flag (`f = built if flag else compiled_out`): Warp never resolves a dead static
  branch, and the type checker sees `built`'s signature (a union with this one: the call checks
  the real arguments, its result is the real return type). Called, it raises.

  Raises:
    AssertionError: always (a dead branch was compiled or called from Python).
  """
  raise AssertionError("a compiled-out device function was called")


def specialize[**P, R](factory: Callable[P, R]) -> Callable[P, R]:
  """A device-code factory (`make_*`: compile-time arguments -> `wp.func` / `wp.kernel` closing
  over them) cached per argument tuple: one Warp function per specialisation and process.

  `functools.cache` with the factory's signature kept (`functools.cache` alone types the
  arguments as `*args: Hashable`, so a wrong flag type goes unseen). Arguments must be hashable.
  The key is the bound arguments, defaults applied, as (name, type, value): positional and keyword
  spellings of one call share a build, `True` and `1` do not. Built under a lock (one build per
  key across threads; factories call only other factories, so the locks never cycle).
  """
  signature = inspect.signature(factory)
  built: dict[tuple[tuple[str, type, object], ...], R] = {}
  lock = threading.RLock()

  @functools.wraps(factory)
  def specialized(*args: P.args, **kwargs: P.kwargs) -> R:
    """`factory(*args, **kwargs)`, built once per bound argument set."""
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    key = tuple((name, type(value), value) for name, value in bound.arguments.items())
    with lock:
      if key not in built:
        built[key] = factory(*args, **kwargs)
      return built[key]

  return specialized
