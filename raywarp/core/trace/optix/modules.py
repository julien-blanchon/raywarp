"""OptiX modules: one per (PTX, `prebuilt.ModuleRecord`), created once per machine and driver.

A module (`create`) is OptiX's compile of a PTX alone (`optixModuleCreateWithTasks`, every core:
`context.DeviceContext.module`), with OptiX's built-in curve intersection module beside it when
its record traverses curves (`pipeline.curves_module`), never NVRTC: the PTX is the bundle's (one
per code split) or the PTX cache's. Its record (register cap, bound values: array strides, the
build and output words) is opaque here. OptiX keeps each module in its disk cache; the running
driver's module pack holds the rows of every module the bundle index records (the census's
combinations), one file per module (`digest`), fetched and merged into that cache just before
the module is created (`prebuilt.seed_module`): those are cache hits. A module the pack lacks
(another driver, a combination outside the census) is created on first use (seconds to a minute),
once per machine and driver: a lock per module, in the process and across processes, makes
concurrent users wait for one creation and then read OptiX's cache. An expected build's module
outside the index warns (`check`).

CUDA graph capture: a module is no CUDA work (it is created while another thread captures), a
pipeline is (its shader binding table uploads): pipelines are created only outside a capture,
before it. `Renderer.capture` renders one frame eagerly before recording, which creates every
pipeline the frame needs (waiting for an uncovered module); inside a capture a missing pipeline
raises (`launch.OptixRenderer._pipeline`). No superset module stands in during a capture: a module
binding other words than the launch parameters hold is undefined in OptiX, and swapping it
after the capture would be a second path.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import threading
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Final, final

from raywarp.cache.paths import cache_dir
from raywarp.core.build.kernel_cache import prebuilt
from raywarp.core.trace.optix.pipeline import curves_module, module_calls, record_options

if TYPE_CHECKING:
  from raywarp.core.trace.optix.context import DeviceContext, OptixObject

_LOCK: Final = threading.Lock()
# The in-process lock of each module being created (by `digest`).
_CREATING: Final[dict[str, threading.Lock]] = {}
# The modules `check` warned about.
_WARNED: Final[set[str]] = set()
# Seconds between two tries of a held module lock while a creation can be cancelled.
_POLL_S: Final = 0.05


def digest(ptx: bytes, record: prebuilt.ModuleRecord) -> str:
  """The name of the module of `ptx` with options `record` (its lock file)."""
  return hashlib.blake2b(ptx + repr(record).encode(), digest_size=12).hexdigest()


@contextlib.contextmanager
def _exclusive(name: str, stop: threading.Event | None) -> Generator[None]:
  """Holds module `name`'s lock: in this process, and across processes (a lock file in raywarp's
  cache directory; none with caching off). A set `stop` ends the wait for another holder
  (RuntimeError): a cancelled creation does not wait out another process's.

  Raises:
    RuntimeError: `stop` was set while another holder had the lock.
  """
  with _LOCK:
    lock = _CREATING.setdefault(name, threading.Lock())
  while not lock.acquire(timeout=_POLL_S):
    _check(stop)
  try:
    user = cache_dir()
    if user is None:
      yield
      return
    path = user / "prebuilt" / "modules" / f"{name}.lock"
    while True:
      with prebuilt.locked(path, wait=stop is None) as held:
        if held:
          yield
          return
      _check(stop)
      time.sleep(_POLL_S)
  finally:
    lock.release()


def _check(stop: threading.Event | None) -> None:
  """Raises when `stop` is set (a cancelled creation).

  Raises:
    RuntimeError: `stop` is set.
  """
  if stop is not None and stop.is_set():
    raise RuntimeError("the OptiX module creation was cancelled")


def create(ctx: DeviceContext, ptx: bytes, record: prebuilt.ModuleRecord, stop: threading.Event | None = None) -> OptixObject:
  """The OptiX module of `ptx` compiled with `record`: its rows of the running driver's module pack
  merged into OptiX's disk cache first (`prebuilt.seed_module`: a cache hit), else OptiX's compile;
  once across threads and processes (`_exclusive`). No CUDA call: any thread, also during a
  capture. A set `stop` abandons a compile between tasks, or the wait for another creation
  (RuntimeError).

  Raises:
    RuntimeError: OptiX could not create the module (its messages attached).
  """
  name = digest(ptx, record)
  with _exclusive(name, stop):
    db = ctx.cache_db
    if db is not None and prebuilt.seed_module(ctx.arch, name, db):
      ctx.reload_cache()  # OptiX sees the merged rows (`DeviceContext.reload_cache`)
    return make(ctx, ptx, record, stop)


def make(ctx: DeviceContext, ptx: bytes, record: prebuilt.ModuleRecord, stop: threading.Event | None = None) -> OptixObject:
  """`create`'s module made by OptiX alone (its disk cache, or its compile), the curve module its
  record needs first (`pipeline.curves_module`); no pack, no lock: the module pack builder's.

  Raises:
    RuntimeError: OptiX could not create the module (its messages attached).
  """
  options = record_options(record, calls=module_calls(ptx))
  try:
    if record.has_curves:  # its pipeline's curve hit groups' (into OptiX's disk cache, a pack's)
      curves_module(ctx, options)
    return ctx.module(options.module, options.pipeline, ptx, stop)
  except Exception as e:
    raise RuntimeError(f"OptiX could not create the program module: {e}") from e


@final
class Ahead:
  """One module created ahead of its first use on a thread of its own (`create`, every core): a
  build starts the modules its first frame binds together (`launch.OptixRenderer`), so a cold
  first frame waits for the slowest, not their sum. `cancel` returns at once (a closed build does
  not wait for a module it will never bind: the thread stops between OptiX tasks and releases what
  it made); process exit stops it and waits for it (a daemon, joined before the interpreter stops
  daemon threads)."""

  def __init__(self, ctx: DeviceContext, ptx: bytes, record: prebuilt.ModuleRecord) -> None:
    """Starts creating the module of `ptx` with options `record`."""
    self._stop = threading.Event()
    self._guard = threading.Lock()  # `_module` between the thread and `cancel`
    self._module: OptixObject | None = None
    self._error: BaseException | None = None
    self._thread = threading.Thread(target=self._run, args=(ctx, ptx, record), name="rw_optix_ahead", daemon=True)
    atexit.register(self._exit)
    self._thread.start()

  def _run(self, ctx: DeviceContext, ptx: bytes, record: prebuilt.ModuleRecord) -> None:
    """The creation (its module, or its failure kept for `result`; released here when cancelled
    meanwhile)."""
    try:
      module = create(ctx, ptx, record, self._stop)
      with self._guard:
        orphan = module if self._stop.is_set() else None
        if orphan is None:
          self._module = module
      if orphan is not None:
        orphan.destroy()
    except BaseException as e:  # noqa: BLE001  # re-raised by `result` on the caller's thread
      self._error = e
    finally:
      atexit.unregister(self._exit)

  def _exit(self) -> None:
    """At process exit: stops the creation and waits for the thread."""
    self._stop.set()
    self._thread.join()

  def join(self) -> None:
    """Waits for the creation to end (the module, or its failure, stays for `result`)."""
    self._thread.join()

  def result(self) -> OptixObject:
    """The module (waits; owned by the caller from here).

    Raises:
      RuntimeError: OptiX could not create it, or it was cancelled.
    """
    self._thread.join()
    if self._error is not None:
      raise self._error
    with self._guard:
      module, self._module = self._module, None
    if module is None:
      raise RuntimeError("an OptiX module created ahead was cancelled")
    return module

  def cancel(self) -> None:
    """Stops the creation between OptiX tasks without waiting for it, and releases its module
    (now if made, else by the thread when it ends): a build closed before it bound the module."""
    self._stop.set()
    with self._guard:
      module, self._module = self._module, None
    if module is not None:
      module.destroy()


def check(arch: int, key: str, record: prebuilt.ModuleRecord, *, expected: bool) -> None:
  """Warns (once per module) when an expected build (`tiers.prebuilt`) needs a module its bundle
  entry does not record: no module pack holds it, OptiX creates it now (once per machine)."""
  b = prebuilt.bundle(arch) if expected else None
  try:
    entry = None if b is None else b.entry(key)
  except ValueError:
    return  # damaged: `prebuilt.ptx` warned
  name = f"{key}:{record}"
  if entry is None or record in entry.modules or name in _WARNED:
    return
  _WARNED.add(name)
  # What differs from the most used recorded module (bound words: offset, bytes), else all of it.
  near = entry.modules[0] if entry.modules else None
  differs = record if near is None else sorted(set(record.bound) - set(near.bound)) or record
  prebuilt.warn(
    f"the OptiX module of build {key} is not prebuilt ({differs}; no census scene uses this combination): OptiX "
    "creates it now (no NVRTC; seconds to a minute, once per machine and driver; please report the scene)"
  )


def create_each(ctx: DeviceContext, jobs: list[tuple[bytes, prebuilt.ModuleRecord]], *, workers: int) -> tuple[int, float]:
  """Creates the modules of `jobs` ((PTX, record) each), `workers` at a time (each in parallel
  tasks), into OptiX's disk cache (each destroyed: the cache keeps it); (modules created, wall
  seconds). `Renderer.warm`'s modules of the settings one change away.

  Raises:
    RuntimeError: a module OptiX rejected.
  """
  import concurrent.futures  # noqa: PLC0415  # warms only

  def one(job: tuple[bytes, prebuilt.ModuleRecord]) -> int:
    """One module, created and released."""
    create(ctx, *job).destroy()
    return 1

  t0 = time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max(1, workers), thread_name_prefix="rw_optix_modules") as pool:
    done = sum(pool.map(one, jobs))
  return done, time.perf_counter() - t0
