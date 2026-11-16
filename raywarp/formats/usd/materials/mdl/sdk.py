"""The NVIDIA MDL SDK (`raywarp._native.mdl`'s `pymdlsdk`): the subset raywarp calls, typed,
and the process's one SDK instance.

The bindings are untyped SWIG classes, imported at run time when a stage first needs an MDL
material. The Protocols below type what this package calls; `api()` returns the started SDK and
raises `MdlSdkError`, with the cause, when the bundled SDK is missing or does not start (an MDL
material is never silently made grey). The SDK is started once per process and shut down at its
exit; every use holds `LOCK` (one transaction at a time: search paths are global to the SDK).
"""

from __future__ import annotations

import atexit
import dataclasses
import functools
import gc
import importlib
import threading
from typing import TYPE_CHECKING, Final, NewType, final

from raywarp import _native as native

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.mdl.sdk_types import (
    Bindings,
    IDatabase,
    IImage_api,
    IMdl_configuration,
    IMdl_distiller_api,
    IMdl_execution_context,
    IMdl_factory,
    IMdl_impexp_api,
    INeuray,
    Interface,
    Named,
  )

# Every SDK call holds it (reentrant: a conversion calls helpers that take it too), except the
# compiles and bakes a prefetch runs on its pool while the caller holds it (the SDK releases the
# GIL; each worker uses its own transactions).
LOCK: Final = threading.RLock()
# Set when the SDK was shut down at exit (`_shutdown`): `api()` then raises.
_SHUT_DOWN: Final = threading.Event()
# How long the exit shutdown waits for a thread inside the SDK [s].
_EXIT_LOCK_S: Final = 1.0
# The DB scope privacy level of a stage read's modules (one level below the global scope).
SCOPE_PRIVACY: Final = 1
# A DB element name (a texture, an image, a module).
DbName = NewType("DbName", str)


class MdlSdkError(RuntimeError):
  """The bundled MDL SDK (`raywarp._native.mdl`) is missing or does not start: a stage with MDL
  materials cannot be read."""


def valid(x: Interface | None) -> bool:
  """Whether `x` is a valid interface."""
  return x is not None and x.is_valid_interface()


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Api:
  """The started SDK and its components."""

  bindings: Bindings
  # Kept alive with the process: the components are valid while it is.
  neuray: INeuray
  database: IDatabase
  config: IMdl_configuration
  impexp: IMdl_impexp_api
  factory: IMdl_factory
  distiller: IMdl_distiller_api
  images: IImage_api
  # Library directories every search ends with: the bundled SDK's (`::nvidia::core_definitions`).
  system_paths: tuple[str, ...]
  # `IMaterial_instance::DEFAULT_OPTIONS` (instance compilation) and `Baker_resource::BAKE_ON_CPU`.
  instance_compilation: int
  bake_on_cpu: Named


@final
@dataclasses.dataclass(slots=True)
class SearchPaths:
  """The MDL / resource search paths last applied to the SDK (global to the process)."""

  applied: tuple[str, ...] = ()


SEARCH: Final = SearchPaths()


def api() -> Api:
  """The process's started SDK (started on the first call, shut down at interpreter exit).

  Raises:
    MdlSdkError: when the bundled SDK is missing or fails to load or start (the cause chained: a
      missing build, a build lacking a binding, a plugin that does not load).
  """
  with LOCK:  # outside the memo: one start, never two
    return _api()


def _api() -> Api:
  """The memoized start (a failure too: the SDK is never loaded twice), raised or returned."""
  if _SHUT_DOWN.is_set():
    raise MdlSdkError("the MDL SDK was shut down (interpreter exit)")
  result = _start_once()
  if isinstance(result, MdlSdkError):
    raise result
  return result


@functools.cache
def _start_once() -> Api | MdlSdkError:
  """The started SDK, or the error of the one attempt (cause chained)."""
  try:
    return _start()
  except Exception as e:  # noqa: BLE001  # returned, then raised by every `api()` call
    error = MdlSdkError(f"the bundled MDL SDK is not usable: {type(e).__name__}: {e}")
    error.__cause__ = e
    return error


def _start() -> Api:
  """Loads the bundled SDK (`raywarp._native.mdl`), its plugins, and starts it."""
  importlib.import_module("raywarp._native.mdl.pymdlsdk")
  bundle = importlib.import_module("raywarp._native.mdl")
  bindings: Bindings = bundle.pymdlsdk
  neuray: INeuray = bundle.load_neuray()
  if neuray.start() != 0:
    raise RuntimeError("the MDL SDK did not start")
  atexit.register(_shutdown, neuray)
  raw = bundle.pymdlsdk
  instance = int(raw.IMaterial_instance.Compilation_options.DEFAULT_OPTIONS.value)
  bake_on_cpu: Named = raw.Baker_resource.BAKE_ON_CPU
  return Api(
    bindings=bindings,
    neuray=neuray,
    database=neuray.get_api_component(bindings.IDatabase),
    config=neuray.get_api_component(bindings.IMdl_configuration),
    impexp=neuray.get_api_component(bindings.IMdl_impexp_api),
    factory=neuray.get_api_component(bindings.IMdl_factory),
    distiller=neuray.get_api_component(bindings.IMdl_distiller_api),
    images=neuray.get_api_component(bindings.IImage_api),
    system_paths=(str(bundle.MDL_DIR),),
    instance_compilation=instance,
    bake_on_cpu=bake_on_cpu,
  )


def _shutdown(neuray: INeuray) -> None:
  """Shuts the SDK down at interpreter exit, after the memoized `Api` and the collectable SDK
  handles are released: a process exiting with the SDK started (and a committed transaction)
  crashes in the SDK's own teardown (SIGSEGV with the release binaries). A thread still inside the
  SDK (`LOCK` held past `_EXIT_LOCK_S`) keeps it running: shutting it down under that thread would
  crash it instead."""
  if not LOCK.acquire(timeout=_EXIT_LOCK_S):
    return
  try:
    _SHUT_DOWN.set()
    _start_once.cache_clear()
    gc.collect()
    neuray.shutdown()
  finally:
    LOCK.release()


def version() -> str:
  """The bundled MDL SDK release (raywarp._native's build record): part of cache keys.
  "unbuilt" on a source tree the build hook never ran on: it has no SDK, so its loads convert no
  MDL material (using one raises), and its caches never match a built tree's."""
  try:
    return native.build()["mdl_sdk"]
  except native.NativeBuildError:
    return "unbuilt"


def messages(context: IMdl_execution_context) -> str:
  """The context's first error message, without its file location prefix."""
  if not context.get_error_messages_count():
    return ""
  return context.get_error_message(0).get_string().rpartition("): ")[2]
