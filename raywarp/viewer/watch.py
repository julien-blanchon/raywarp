"""What the viewer process compiles or creates, live: the proof that a change compiled nothing.

Three kinds of event, from any thread (the renderer compiles in the background):

- "kernel": a Warp module of the renderer compiled (NVRTC / LLVM), not read from the kernel cache
  or the prebuilt bundle: a real compile (`Module._compile` returned True); "other": a module of
  other code compiled (MuJoCo Warp's, the caller's), which no raywarp bundle holds;
- "ptx": the OptiX programs' PTX compiled (not in the PTX cache or the bundle);
- "optix_module": OptiX created a module its disk cache lacked (OptiX's "Cache miss" log line):
  a first-use stall of seconds, not a compile; the driver's module pack removes it.

Each one is also seen while it runs (`running`: what, on which thread, for how long, and the
estimate of its kind: the median of this process's finished ones, OptiX's per-module estimate
`nearby.ESTIMATE_S` before the first), so that a page can say what a blocked render thread waits
for. The background warm's work is told apart by its threads (`Event.background`).

`install` wraps Warp's module compile, raywarp's PTX compile and OptiX's module creation
(`DeviceContext.module`) (Warp's is an internal method: the supported Warp range is pinned by the
package, and a Warp without it raises at install, never counting silently nothing) and adds a
handler to the `raywarp.optix` logger.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import statistics
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, final

from raywarp.core.build.variants import nearby

if TYPE_CHECKING:
  from pathlib import Path

  import warp as wp

type Kind = Literal["kernel", "ptx", "optix_module", "other"]
KINDS: Final[tuple[Kind, ...]] = ("kernel", "ptx", "optix_module", "other")
# The Python packages whose Warp modules the prebuilt bundle holds: raywarp's, and Warp's own
# built-in kernels it launches (mesh / BVH builds). Any other module ("other": a simulator's, the
# caller's) is not the renderer's.
BUNDLED: Final = ("raywarp", "warp.")
# OptiX's message for a module its disk cache lacked (logged at INFO by `raywarp.optix`).
_MISS: Final = "Cache miss for key"
# Events kept (a process that compiles thousands only shows the latest).
MAX_EVENTS: Final = 1000
# Durations kept per kind for the estimates.
_HISTORY: Final = 32
# The thread name prefixes of background work (`Renderer.warm`'s pool and its OptiX module workers).
BACKGROUND: Final = ("rw_warm", "rw_optix_modules")


def background() -> bool:
  """Whether the calling thread does background work (`BACKGROUND`)."""
  return threading.current_thread().name.startswith(BACKGROUND)


@final
@dataclasses.dataclass(frozen=True)
class Event:
  """One compile or creation.

  Attributes:
    seq: its position in the process's events (1 = first).
    time: wall-clock end time [s since the epoch].
    kind: "kernel", "ptx", "optix_module" or "other" (module doc).
    name: the Warp module, the PTX cache key or the OptiX cache key.
    seconds: how long it took (0: not timed, an OptiX module created on a thread of OptiX's own).
    background: done by the background warm (`BACKGROUND`), not for a frame or a command.
  """

  seq: int
  time: float
  kind: Kind
  name: str
  seconds: float
  background: bool = False

  def json(self) -> dict[str, object]:
    """The event as a JSON object."""
    return dataclasses.asdict(self)


@final
@dataclasses.dataclass(slots=True)
class Running:
  """A compile or creation in progress.

  Attributes:
    kind: its kind (an OptiX module: not yet known to be a cache miss).
    name: the Warp module, the PTX key or (once OptiX logged its cache miss) the OptiX key.
    start: its start (`time.monotonic`).
    background: on a background thread (`BACKGROUND`).
    miss: an OptiX module whose cache miss was logged on its thread (a creation, not a cache hit).
  """

  kind: Kind
  name: str
  start: float
  background: bool
  miss: bool = False


@final
class CompileWatch:
  """The process's compile events (`install` once; `since(seq)` from any thread)."""

  def __init__(self) -> None:
    """No event yet; nothing installed."""
    self._lock = threading.Lock()
    self._events: list[Event] = []
    self._counts = dict.fromkeys(KINDS, 0)
    self._inside_ptx = threading.local()
    self._running: dict[int, Running] = {}
    self._ids = itertools.count()
    self._threads = threading.local()  # the OptiX module creation running on this thread
    self._durations: dict[Kind, list[float]] = {k: [] for k in KINDS}
    self.installed = False

  def add(self, kind: Kind, name: str, seconds: float, *, back: bool | None = None) -> None:
    """Records one event (`back`: done in the background; None: the calling thread's)."""
    back = background() if back is None else back
    with self._lock:
      self._counts[kind] += 1
      seq = sum(self._counts.values())
      self._events = [*self._events[-(MAX_EVENTS - 1) :], Event(seq, time.time(), kind, name, seconds, back)]
      if seconds > 0:
        self._durations[kind] = [*self._durations[kind][-(_HISTORY - 1) :], seconds]

  def begin(self, kind: Kind, name: str) -> int:
    """Marks a compile or creation started on the calling thread; its id for `end`."""
    with self._lock:
      key = next(self._ids)
      self._running[key] = Running(kind, name, time.monotonic(), background())
      return key

  def end(self, key: int) -> Running:
    """Marks `begin`'s `key` finished; what it was."""
    with self._lock:
      return self._running.pop(key)

  def missed(self, name: str) -> bool:
    """Notes OptiX's cache miss of module `name` on the calling thread: True when a creation of
    `install`'s wrapper runs on it (its end records the event, timed)."""
    key = getattr(self._threads, "module", None)
    with self._lock:
      found = self._running.get(key) if key is not None else None
      if found is None:
        return False
      found.name, found.miss = name, True
      return True

  def estimate(self, kind: Kind) -> float | None:
    """Seconds one compile or creation of `kind` takes: the median of this process's finished
    ones, OptiX's per-module estimate before the first module (`nearby.ESTIMATE_S`, the default
    tiers' family), None for a kernel or PTX compile before the first."""
    with self._lock:
      done = list(self._durations[kind])
    if done:
      return statistics.median(done)
    return nearby.ESTIMATE_S["full"] if kind == "optix_module" else None

  def running(self, older: float = 0.0) -> list[dict[str, object]]:
    """The compiles and creations running for longer than `older` seconds, as JSON objects (kind,
    name, seconds so far, estimate, background), the longest first. An OptiX module not yet
    known to be a cache miss is shown once older than `older`: a cache hit takes milliseconds."""
    now = time.monotonic()
    with self._lock:
      items = [r for r in self._running.values() if now - r.start > older]
    return [
      {"kind": r.kind, "name": r.name, "seconds": now - r.start, "estimate": self.estimate(r.kind), "background": r.background}
      for r in sorted(items, key=lambda r: r.start)
    ]

  def counts(self) -> dict[Kind, int]:
    """Events so far per kind."""
    with self._lock:
      return dict(self._counts)

  def total(self) -> int:
    """Events so far."""
    with self._lock:
      return sum(self._counts.values())

  def since(self, seq: int) -> list[Event]:
    """The kept events after event `seq` (0: all kept)."""
    with self._lock:
      return [e for e in self._events if e.seq > seq]

  def install(self) -> None:
    """Starts recording (once per process: later calls do nothing). Sets the `raywarp.optix`
    logger to INFO when it is quieter (OptiX's cache misses are INFO lines): the viewer's process.

    Raises:
      RuntimeError: when this Warp has no `Module._compile` to observe.
    """
    if self.installed:
      return
    import warp._src.context as warp_context  # noqa: PLC0415  # the observed internals

    compile_module = getattr(warp_context.Module, "_compile", None)
    if compile_module is None:
      raise RuntimeError("this Warp release has no Module._compile: the viewer cannot observe kernel compiles")
    warp_context.Module._compile = self._wrap_kernel(compile_module)  # noqa: SLF001  # pyrefly: ignore[bad-assignment]  # a `*args` wrapper of the method
    self._install_ptx()
    self._install_optix()
    log = logging.getLogger("raywarp.optix")
    log.addHandler(OptixMisses(self))
    if log.getEffectiveLevel() > logging.INFO:
      log.setLevel(logging.INFO)
    self.installed = True

  def _wrap_kernel(self, compile_module: Callable[..., bool]):
    """Warp's module compile, recording each one that compiled (False: a cache or bundle hit)."""
    watch = self

    @functools.wraps(compile_module)
    def counted(module: object, *args: object, **kwargs: object) -> bool:
      """`Module._compile`, timed and recorded (running: `begin`; inside a PTX compile, the PTX
      compile's)."""
      ptx = getattr(watch._inside_ptx, "key", None)
      kind = "ptx" if ptx is not None else owner(module)
      name = ptx or str(getattr(module, "name", "?"))
      key = None if ptx is not None else watch.begin(kind, name)
      start = time.perf_counter()
      try:
        compiled = compile_module(module, *args, **kwargs)
      finally:
        if key is not None:
          watch.end(key)
      if compiled:
        watch.add(kind, name, time.perf_counter() - start)
      return compiled

    return counted

  def _install_ptx(self) -> None:
    """Wraps the OptiX programs' PTX compile."""
    from raywarp.core.trace.optix import pipeline  # noqa: PLC0415

    compile_ptx = pipeline.PtxCompile._compile  # noqa: SLF001
    watch = self

    def counted(module: wp.Module, arch: int, path: Path | None) -> bytes:
      """`PtxCompile._compile`: the Warp compile inside it (none when another process stored the
      PTX meanwhile) is recorded as "ptx", named by the PTX key."""
      watch._inside_ptx.key = name = str(getattr(path, "stem", path))
      key = watch.begin("ptx", name)
      try:
        return compile_ptx(module, arch, path)
      finally:
        watch._inside_ptx.key = None
        watch.end(key)

    pipeline.PtxCompile._compile = staticmethod(counted)  # noqa: SLF001

  def _install_optix(self) -> None:
    """Wraps OptiX's module creation (`DeviceContext.module`): running while it runs; a cache miss
    logged on its thread makes it a timed "optix_module" event at its end."""
    from raywarp.core.trace.optix.context import DeviceContext  # noqa: PLC0415

    create = DeviceContext.module
    watch = self

    @functools.wraps(create)
    def counted(ctx: DeviceContext, *args: object, **kwargs: object) -> object:
      """`DeviceContext.module`, watched."""
      key = watch.begin("optix_module", "OptiX module")
      outer = getattr(watch._threads, "module", None)
      watch._threads.module = key
      try:
        return create(ctx, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]
      finally:
        watch._threads.module = outer
        done = watch.end(key)
        if done.miss:
          watch.add("optix_module", done.name, time.monotonic() - done.start, back=done.background)

    DeviceContext.module = counted  # pyrefly: ignore[bad-assignment]


def owner(module: object) -> Kind:
  """The kind of a compiled Warp module: "kernel" for `BUNDLED` code (by its kernels' Python
  modules, else its name), "other" for any other."""
  kernels: dict[str, object] = getattr(module, "kernels", {})
  owners = {str(getattr(getattr(k, "func", None), "__module__", "")) for k in kernels.values()}
  owners = owners or {str(getattr(module, "name", ""))}
  return "kernel" if any(o.startswith(BUNDLED) for o in owners) else "other"


@final
class OptixMisses(logging.Handler):
  """Records OptiX's disk-cache misses (a module created) as "optix_module" events."""

  def __init__(self, watch: CompileWatch) -> None:
    """Records into `watch`."""
    super().__init__(logging.INFO)
    self.watch = watch

  def emit(self, record: logging.LogRecord) -> None:
    """One OptiX log line: a cache miss is recorded with its key."""
    message = record.getMessage()
    if _MISS in message:
      name = message.rsplit(" ", 1)[-1]
      if not self.watch.missed(name):  # logged off a watched creation's thread: untimed
        self.watch.add("optix_module", name, 0.0)


# The process's watch (the viewer installs it before its first scene).
WATCH: Final = CompileWatch()
