"""The OptiX device context of a CUDA device (one per CUDA context), its objects and its log.

`DeviceContext` wraps an `OptixDeviceContext` of the ctypes binding (`api.py`): acceleration
structures, modules, program groups, pipelines and opacity micromaps. Modules compile
with `optixModuleCreateWithTasks`: OptiX splits the compile into tasks that `_run_tasks` executes on
a pool of Python threads (each foreign call releases the GIL), so a cold compile uses every core.

OptiX's messages go to the `raywarp.optix` logger (fatal -> CRITICAL, error -> ERROR, warning ->
WARNING, print -> INFO): configure Python logging to see more or less of them.

OptiX keeps the modules it compiled in a disk cache: raywarp's cache directory's `optix/`
(`cache.paths.cache_dir()`, `$RAYWARP_CACHE_DIR`; next to the bakes and textures, so that one
directory holds every cache a warm start reads). OptiX's own `OPTIX_CACHE_PATH`, when set, takes
precedence (that directory is used); an empty `RAYWARP_CACHE_DIR` (caching off) turns the module
cache off. A location OptiX cannot use warns and keeps OptiX's default one. The module pack's rows
are merged into that cache just before their module is created (`trace/optix/modules.create`).
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import ctypes
import logging
import os
import threading
import warnings
from collections.abc import Generator
from ctypes import byref
from pathlib import Path
from typing import ClassVar, Final, final

import numpy as np
import warp as wp

from raywarp.cache.paths import cache_dir
from raywarp.core.build.kernel_cache import prebuilt
from raywarp.core.pipeline.platform.device import process_cores
from raywarp.core.trace.optix import api

# Bytes of the log buffer handed to compile / link calls (OptiX truncates longer logs).
_LOG_BYTES: Final = 8192
# The newest PTX target the programs are compiled for (`ptx_arch`).
MAX_PTX_ARCH: Final = 89
# OptiX's disk-cache size bounds (garbage collection past the high mark, down to the low one):
# above a module pack (1-3 GB per driver, `core/build/kernel_cache/prebuilt.py`) and the modules
# created beside it. OptiX's own variable OPTIX_CACHE_MAXSIZE, when set, takes precedence.
CACHE_LOW: Final = 6 << 30
CACHE_HIGH: Final = 8 << 30
# The directory inside OptiX's cache directory `DeviceContext.reload_cache` moves the cache to and
# back from (an empty database there).
RELOAD_DIR: Final = "reload"
_CACHE_SIZE_ENV: Final = "OPTIX_CACHE_MAXSIZE"


# OptiX modules of this process by outcome (`DeviceContext.module`): "compiled" (OptiX ran compile
# tasks) or "cached" (its disk cache served the module); `raywarp.warm` reports them. OptiX hands
# out a first task for a disk-cache hit too: a hit runs that one task and spawns none, a compile
# spawns more (RTX PRO 6000, driver 580: 1 against 5-9 tasks for the smallest modules).
MODULES: Final[collections.Counter[str]] = collections.Counter()
CACHED_TASKS: Final = 1
_MODULES_LOCK: Final = threading.Lock()


def _log_buffer() -> tuple[ctypes.Array[ctypes.c_char], ctypes.c_size_t]:
  """A log buffer and its size argument."""
  log = ctypes.create_string_buffer(_LOG_BYTES)
  return log, ctypes.c_size_t(_LOG_BYTES)


@final
class OptixObject:
  """A native OptiX object (module, program group, pipeline), destroyed explicitly.

  Attributes:
    handle: the native handle (0 once destroyed).
  """

  def __init__(self, handle: int, destroy: str) -> None:
    """Owns `handle`, released by the table entry `destroy` (e.g. "optixModuleDestroy")."""
    self.handle = handle
    self._destroy = destroy

  def destroy(self) -> None:
    """Releases the object (once).

    Raises:
      api.OptixError: when OptiX refuses.
    """
    handle, self.handle = self.handle, 0
    if handle:
      api.check(getattr(api.table(), self._destroy)(handle), self._destroy)


def address(a: wp.array, *, empty_ok: bool = False) -> int:
  """The device address of a buffer OptiX reads (`empty_ok`: 0 for an unallocated, empty one;
  OptiX accepts a null buffer of zero items).

  Raises:
    ValueError: for an unallocated buffer unless `empty_ok`.
  """
  if a.ptr is None:
    if empty_ok and a.size == 0:
      return 0
    raise ValueError("an OptiX buffer has no device memory")
  return int(a.ptr)


def set_stack_size(pipeline: OptixObject, dc_traversal: int, dc_state: int, continuation: int, depth: int) -> None:
  """Sets a pipeline's stack sizes and maximum traversable depth (`optixPipelineSetStackSize`)."""
  api.check(
    api.table().optixPipelineSetStackSize(pipeline.handle, dc_traversal, dc_state, continuation, depth),
    "optixPipelineSetStackSize",
  )


def _execute(task: int, room: int) -> list[int]:
  """Runs one OptiX compile task; the tasks it spawned (at most `room`)."""
  more = (api.Handle * room)()
  n = ctypes.c_uint(0)
  api.check(api.table().optixTaskExecute(task, more, room, byref(n)), "optixTaskExecute")
  return [int(more[i]) for i in range(n.value)]


def _run_tasks(first: int, stop: threading.Event | None) -> int:
  """Executes the task tree rooted at `first` on a thread pool (`process_cores` threads), until
  none is left, or until `stop` is set (the running tasks end; the module stays incomplete); the
  tasks executed."""
  workers = process_cores()  # the process's, not the calling thread's affinity
  executed = 0
  with concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix="rw_optix_task") as pool:
    pending = {pool.submit(_execute, first, workers)}
    while pending:
      done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
      for f in done:
        more = f.result()
        executed += 1
        if stop is None or not stop.is_set():
          pending |= {pool.submit(_execute, t, workers) for t in more}
  return executed


@final
class DeviceContext:
  """An OptiX device context (`optixDeviceContextCreate` on a CUDA context): what raywarp builds
  with it. Calls take ctypes structs of `api` and device addresses as ints.

  Attributes:
    handle: the native OptixDeviceContext.
    arch: the SM version of its device (the module pack it reads: `trace/optix/modules.create`).
  """

  def __init__(self, cuda_context: int, arch: int) -> None:
    """Creates the context on CUDA context `cuda_context` of a device of SM version `arch`, its
    log going to `_Logger`."""
    self._logger = _Logger()
    # Kept alive with the context: OptiX calls it until the context is destroyed.
    self._callback = api.LogCallback(self._logger.native)
    options = api.DeviceContextOptions(logCallbackFunction=self._callback, logCallbackLevel=4)
    handle = api.Handle()
    api.check(api.table().optixDeviceContextCreate(cuda_context, byref(options), byref(handle)), "optixDeviceContextCreate")
    self.handle = handle.value or 0
    self.arch = arch
    # OptiX's disk cache: its directory (None: OptiX's default, or off) and the lock of its reloads.
    self.location: str | None = None
    self._reload = threading.Lock()

  def close(self) -> None:
    """Destroys the context (once; its objects must be released first). A context of its own (the
    module pack builder's, one per module), never `device_context`'s shared one."""
    handle, self.handle = self.handle, 0
    if handle:
      api.check(api.table().optixDeviceContextDestroy(handle), "optixDeviceContextDestroy")

  def set_cache_enabled(self, enabled: bool) -> None:  # noqa: FBT001  # OptiX's switch
    """Turns OptiX's module disk cache on or off."""
    api.check(api.table().optixDeviceContextSetCacheEnabled(self.handle, int(enabled)), "optixDeviceContextSetCacheEnabled")

  def set_cache_location(self, location: str) -> None:
    """Moves OptiX's module disk cache to directory `location` (created)."""
    api.check(
      api.table().optixDeviceContextSetCacheLocation(self.handle, location.encode()), "optixDeviceContextSetCacheLocation"
    )
    self.location = location

  @property
  def cache_db(self) -> Path | None:
    """OptiX's disk-cache database (None: no location set, the cache off or OptiX's default)."""
    return None if self.location is None else Path(self.location) / prebuilt.OPTIX_DB

  def reload_cache(self) -> None:
    """Makes OptiX read its disk cache again: rows merged into it from outside after OptiX served a
    module from it are otherwise not seen (OptiX keeps its view of the database where that hit left
    it; setting the same location again or turning the cache off and on does not refresh it,
    moving it to another directory and back does: RTX PRO 6000, driver 580). Modules created
    meanwhile on other threads are unaffected (their rows land)."""
    location = self.location
    if location is None:
      return
    with self._reload, _own_location():
      self.set_cache_location(str(Path(location) / RELOAD_DIR))
      self.set_cache_location(location)

  def set_cache_sizes(self, low: int, high: int) -> None:
    """OptiX's disk-cache garbage collection: past `high` bytes it evicts down to `low`."""
    api.check(
      api.table().optixDeviceContextSetCacheDatabaseSizes(self.handle, low, high), "optixDeviceContextSetCacheDatabaseSizes"
    )

  def accel_sizes(self, options: api.AccelBuildOptions, inputs: ctypes.Array[api.BuildInput]) -> api.AccelBufferSizes:
    """Buffer sizes of a build of `inputs`."""
    sizes = api.AccelBufferSizes()
    t = api.table()
    api.check(
      t.optixAccelComputeMemoryUsage(self.handle, byref(options), inputs, len(inputs), byref(sizes)),
      "optixAccelComputeMemoryUsage",
    )
    return sizes

  def accel_build(  # noqa: PLR0913  # optixAccelBuild's arguments
    self,
    stream: int,
    options: api.AccelBuildOptions,
    inputs: ctypes.Array[api.BuildInput],
    temp: int,
    temp_size: int,
    out: int,
    out_size: int,
    emit: ctypes.Array[api.AccelEmitDesc] | None,
  ) -> int:
    """Builds (or refits) a structure on `stream`; its traversable handle."""
    handle = api.TraversableHandle()
    nemit = 0 if emit is None else len(emit)
    result = api.table().optixAccelBuild(
      self.handle, stream, byref(options), inputs, len(inputs), temp, temp_size, out, out_size, byref(handle), emit, nemit
    )
    api.check(result, "optixAccelBuild")
    return handle.value

  def relocation_info(self, handle: int) -> np.ndarray[[4], np.dtype[np.uint64]]:
    """The relocation information of built structure `handle` (`optixAccelGetRelocationInfo`)."""
    info = api.RelocationInfo()
    api.check(api.table().optixAccelGetRelocationInfo(self.handle, handle, byref(info)), "optixAccelGetRelocationInfo")
    return np.asarray(list(info.info), np.uint64)

  def relocation_compatible(self, info: np.ndarray[[4], np.dtype[np.uint64]]) -> bool:
    """Whether a structure of relocation information `info` can be relocated on this context
    (`optixCheckRelocationCompatibility`: the same device and driver)."""
    found, ok = api.RelocationInfo(), ctypes.c_int()
    found.info[:] = [int(x) for x in info]
    api.check(
      api.table().optixCheckRelocationCompatibility(self.handle, byref(found), byref(ok)), "optixCheckRelocationCompatibility"
    )
    return bool(ok.value)

  def accel_relocate(self, stream: int, info: np.ndarray[[4], np.dtype[np.uint64]], target: int, size: int) -> int:
    """Relocates the `size` bytes of a structure copied to device address `target` on `stream`
    (`optixAccelRelocate`; no relocate inputs: a GAS without micromaps); returns its handle."""
    found, handle = api.RelocationInfo(), api.TraversableHandle()
    found.info[:] = [int(x) for x in info]
    api.check(
      api.table().optixAccelRelocate(self.handle, stream, byref(found), None, 0, target, size, byref(handle)),
      "optixAccelRelocate",
    )
    return handle.value

  def accel_compact(self, stream: int, handle: int, out: int, size: int) -> int:
    """Compacts a structure into `out`; the new handle."""
    new = api.TraversableHandle()
    api.check(api.table().optixAccelCompact(self.handle, stream, handle, out, size, byref(new)), "optixAccelCompact")
    return new.value

  def module(
    self, mco: api.ModuleCompileOptions, pco: api.PipelineCompileOptions, ptx: bytes, stop: threading.Event | None = None
  ) -> OptixObject:
    """The OptiX module of `ptx`, compiled by parallel tasks (`optixModuleCreateWithTasks`); a set
    `stop` abandons the compile between tasks (the module is destroyed: RuntimeError).

    Raises:
      api.OptixError: when OptiX rejects the module or a compile task fails (OptiX's error messages
        logged during the compile attached as notes: a module compiled at the same time on this
        context may add its own).
      RuntimeError: when the compile did not complete.
    """
    t = api.table()
    log, size = _log_buffer()
    module, first = api.Handle(), api.Handle()
    result = t.optixModuleCreateWithTasks(
      self.handle, byref(mco), byref(pco), ptx, len(ptx), log, byref(size), byref(module), byref(first)
    )
    api.check(result, "optixModuleCreateWithTasks", log)
    obj = OptixObject(module.value or 0, "optixModuleDestroy")
    # The errors logged from here on: another thread's module compile (a build's modules created
    # together, `modules.Ahead`) may log meanwhile, so none is cleared.
    since = self._logger.logged
    try:
      tasks = _run_tasks(first.value, stop) if first.value else 0
      state = ctypes.c_int(0)
      api.check(t.optixModuleGetCompilationState(obj.handle, byref(state)), "optixModuleGetCompilationState")
      if state.value != api.MODULE_COMPILE_STATE_COMPLETED:
        raise RuntimeError(f"OptiX module compile ended in state {state.value:#x}, not COMPLETED")
      with _MODULES_LOCK:
        MODULES["compiled" if tasks > CACHED_TASKS else "cached"] += 1
    except BaseException as error:
      for line in self._logger.errors_since(since):
        error.add_note(f"OptiX: {line}")
      obj.destroy()
      raise
    return obj

  def builtin_is_module(
    self, mco: api.ModuleCompileOptions, pco: api.PipelineCompileOptions, options: api.BuiltinISOptions
  ) -> int:
    """OptiX's built-in intersection module of a primitive type (curves); owned by OptiX."""
    module = api.Handle()
    t = api.table()
    api.check(
      t.optixBuiltinISModuleGet(self.handle, byref(mco), byref(pco), byref(options), byref(module)), "optixBuiltinISModuleGet"
    )
    return module.value or 0

  def program_group(self, desc: api.ProgramGroupDesc, payload: api.PayloadType | None = None) -> OptixObject:
    """The program group of `desc` (`payload`: its programs' payload type; None: a raygen's)."""
    log, size = _log_buffer()
    group = api.Handle()
    options = api.ProgramGroupOptions()
    if payload is not None:
      options.payloadType = ctypes.pointer(payload)
    t = api.table()
    api.check(
      t.optixProgramGroupCreate(self.handle, byref(desc), 1, byref(options), log, byref(size), byref(group)),
      "optixProgramGroupCreate",
      log,
    )
    return OptixObject(group.value or 0, "optixProgramGroupDestroy")

  def pipeline(self, pco: api.PipelineCompileOptions, link: api.PipelineLinkOptions, groups: list[OptixObject]) -> OptixObject:
    """A pipeline of `groups`."""
    log, size = _log_buffer()
    handles = (api.Handle * len(groups))(*[g.handle for g in groups])
    pipeline = api.Handle()
    t = api.table()
    result = t.optixPipelineCreate(
      self.handle, byref(pco), byref(link), handles, len(groups), log, byref(size), byref(pipeline)
    )
    api.check(result, "optixPipelineCreate", log)
    return OptixObject(pipeline.value or 0, "optixPipelineDestroy")

  def micromap_sizes(self, build: api.OpacityMicromapArrayBuildInput) -> api.MicromapBufferSizes:
    """(output, temp) bytes of an opacity micromap array build (`trace/optix/omm/micromaps.py`)."""
    sizes = api.MicromapBufferSizes()
    t = api.table()
    api.check(
      t.optixOpacityMicromapArrayComputeMemoryUsage(self.handle, byref(build), byref(sizes)),
      "optixOpacityMicromapArrayComputeMemoryUsage",
    )
    return sizes

  def micromap_build(self, stream: int, build: api.OpacityMicromapArrayBuildInput, buffers: api.MicromapBuffers) -> None:
    """Builds an opacity micromap array into `buffers.output` on `stream`."""
    t = api.table()
    api.check(
      t.optixOpacityMicromapArrayBuild(self.handle, stream, byref(build), byref(buffers)), "optixOpacityMicromapArrayBuild"
    )


_log: Final = logging.getLogger("raywarp.optix")
_lock: Final = threading.Lock()
# Guards every `_Logger`'s state (OptiX logs from its own threads).
_log_lock: Final = threading.Lock()
# Distinct repeated messages remembered at most (beyond: logged every time; bounded memory).
_SEEN_MAX: Final = 1024
# Error messages kept for the exception of a failed module compile.
_ERRORS_KEPT: Final = 32
# One OptiX context per CUDA context handle (guarded by `_lock`). External contexts must
# remain alive while cached OptiX objects use them; a physical ordinal is not context identity.
_contexts: Final[dict[int, DeviceContext]] = {}


@final
class _Logger:
  """Forwards OptiX log messages to `_log`; OptiX may call it from its own threads: it never
  raises. A warning or print repeated word for word (one per build of a per-frame refit) is
  logged once per device context; fatal messages and errors every time."""

  # OptiX levels 1 (fatal) to 4 (print).
  LEVELS: ClassVar[dict[int, int]] = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING, 4: logging.INFO}

  def __init__(self) -> None:
    """No message seen yet."""
    self.seen: set[tuple[int, str, str]] = set()
    # The latest errors (fatal / error: "[tag] text") with their sequence numbers (`logged`: the
    # errors so far), attached to a failed compile's exception (`DeviceContext.module`: its tasks
    # report through the callback, not a log buffer).
    self.errors: collections.deque[tuple[int, str]] = collections.deque(maxlen=_ERRORS_KEPT)
    self.logged = 0

  def native(self, level: int, tag: bytes | None, message: bytes | None, _data: int | None) -> None:
    """The C callback (`api.LogCallback`): decodes and logs; never raises into OptiX."""
    try:
      self(level, (tag or b"").decode(errors="replace"), (message or b"").decode(errors="replace"))
    except Exception:  # noqa: BLE001  # an exception cannot cross into OptiX's thread
      _log.exception("OptiX log callback failed")

  def __call__(self, level: int, tag: str, message: str) -> None:
    """Logs one OptiX message (a repeated warning or print: once)."""
    text = message.rstrip()
    if level <= 2:
      with _log_lock:
        self.logged += 1
        self.errors.append((self.logged, f"[{tag}] {text}"))
    if level >= 3:
      with _log_lock:
        if (level, tag, text) in self.seen:
          return
        if len(self.seen) < _SEEN_MAX:
          self.seen.add((level, tag, text))
    _log.log(self.LEVELS.get(level, logging.INFO), "OptiX [%s]: %s", tag, text)

  def errors_since(self, since: int) -> list[str]:
    """The errors kept that were logged after the first `since` (`logged`)."""
    with _log_lock:
      return [text for n, text in self.errors if n > since]


def device_context(device: wp.Device) -> DeviceContext:
  """The OptiX device context on Warp's CUDA context of `device` (created on first use).

  Raises:
    RuntimeError: for a device without a CUDA context.
  """
  if device.context is None:
    raise RuntimeError(f"OptiX needs a CUDA device, got {device}")
  identity = int(device.context)
  with _lock:
    ctx = _contexts.get(identity)
    if ctx is None:
      # The CUDA context is passed explicitly: no device scope (rule 38).
      location = os.environ.get(prebuilt.OPTIX_CACHE_ENV)
      with _own_location():
        ctx = DeviceContext(identity, int(device.arch))
        _set_cache(ctx, location)
      _contexts[identity] = ctx
    return ctx


@contextlib.contextmanager
def _own_location() -> Generator[None]:
  """OPTIX_CACHE_PATH out of the environment for the block: OptiX would let it override every
  location the context sets, and `DeviceContext.reload_cache` moves the location (the directory
  it names is set through the API instead: `_set_cache`). Restored after; callers hold `_lock` or
  the context's reload lock."""
  saved = os.environ.pop(prebuilt.OPTIX_CACHE_ENV, None)
  try:
    yield
  finally:
    if saved is not None:
      os.environ[prebuilt.OPTIX_CACHE_ENV] = saved


def _set_cache(ctx: DeviceContext, own: str | None) -> None:
  """Points OptiX's module disk cache at OPTIX_CACHE_PATH's directory `own`, else at
  `<cache dir>/optix` (`prebuilt.optix_cache_dir`; see the module docstring), and bounds it above a
  module pack (`CACHE_HIGH`, unless OPTIX_CACHE_MAXSIZE is set: OptiX's default bound would evict a
  pack's modules)."""
  root = cache_dir()
  location = own or (None if root is None else str(prebuilt.optix_cache_dir(root)))
  try:
    if location is None:
      ctx.set_cache_enabled(False)
      return
    ctx.set_cache_location(location)
    if not os.environ.get(_CACHE_SIZE_ENV):
      ctx.set_cache_sizes(CACHE_LOW, CACHE_HIGH)
  except api.OptixError as e:
    warnings.warn(
      f"OptiX could not use its module cache at {location} ({e}); OptiX's default location is used",
      RuntimeWarning,
      stacklevel=3,
    )


def ptx_arch(device: wp.Device) -> int:
  """The PTX target of the programs: the device's architecture, at most sm_89.

  CUDA 13 NVVM for sm_100+ keeps floats in untyped .b32 registers, which OptiX 9.0 (driver 580)
  rejects as arguments of its intrinsics ("Argument[0] to function call
  _optix_report_intersection_4 is the incorrect type"), and an OptiX runtime older than the PTX
  target rejects it outright. sm_89 PTX is typed, accepted by every OptiX 9 runtime and compiled by
  OptiX for the device itself, so newer devices get it directly: one NVRTC compile per build.
  """
  return min(int(device.arch), MAX_PTX_ARCH)
