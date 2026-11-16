"""What a Warp device can do (CUDA graphs, pinned memory, OptiX, ...) and how CPU launches use
every core: the one place that tells a CPU device from a CUDA one.

Contract: no module tests `device.is_cuda`, `device.arch` or a stream's capture state itself; it
asks `device_caps(device)` what the device can do, by capability (`graphs`, `pinned`,
`device_textures`, ...), not by kind. `DeviceCaps` is a frozen record built once per Warp device
(`functools.cache` on the device alias); only `capturing()` reads live state (the current
stream). A new device kind (e.g. Metal) adds a `kind` and fills the same fields, and the code
that reads them does not change.

The OptiX traversal's device-level requirements (a CUDA device, the driver's OptiX, the bundled SDK
headers) are `DeviceCaps.optix`, probed on first access (it loads the driver's OptiX library); a
scene's own requirements are the trace backend's (`core/trace/backend.py`).

`launch_bands` is the one way to run a launch over all the CPU's cores (a CPU launch runs on one
thread): bands of the launch's range on the process's band threads, one per core of its affinity
and shared by every caller. That is the CPU budget of the bakes: worker threads of one process
share those cores (no oversubscription), and worker processes get theirs by pinning
(`taskset`, `os.sched_setaffinity`).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import os
import sys
import threading
from collections.abc import Callable
from typing import Final, Literal, final

import numpy as np
import warp as wp

# First SM version of the Blackwell RTX GPUs (RTX 50xx, RTX PRO 6000): its measured defaults differ
# (`resolve._tuning`: block size, ordered traversal).
_BLACKWELL_RTX_ARCH: Final = 120
# First SM version with RT cores (Turing), and the later SM versions of GPUs without them: GA100
# (8.0: A100 / A30), Jetson Orin (8.7), GH100 (9.0: H100 / H200 / GH200), GB100 / GB200 (10.0),
# Jetson Thor (10.1 before CUDA 13, 11.0 after; its published spec lists no RT cores) and GB300
# (10.3). CUDA exposes no RT-core attribute: the SM version is the only device-level record of the
# hardware (NVIDIA's architecture specs). An SM version not listed here (a future one) counts as
# having RT cores: OptiX then runs or raises, never a silent choice.
_FIRST_RT_ARCH: Final = 75
_COMPUTE_ONLY_ARCHS: Final = frozenset({80, 87, 90, 100, 101, 103, 110})


def has_rt_cores(arch: int) -> bool:
  """Whether a CUDA GPU of SM version `arch` (e.g. 89 for 8.9) has ray-tracing (RT) cores; 0 (the
  CPU) has none."""
  return arch >= _FIRST_RT_ARCH and arch not in _COMPUTE_ONLY_ARCHS


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class OptixSupport:
  """Whether this device can run the OptiX traversal at all (before any scene requirement).

  Attributes:
    available: a CUDA device whose driver serves the OptiX ABI of the binding
      (`trace.optix.api.unavailable_reason`), with the bundled OptiX SDK headers of that release.
    reason: why not (None when available), worded to follow "OptiX cannot run: ".
  """

  available: bool
  reason: str | None


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class DeviceCaps:
  """What a Warp device can do, resolved once (`device_caps`). Not slotted: `optix` is a
  `functools.cached_property` (it needs the instance `__dict__`).

  Attributes:
    alias: the Warp device alias ("cpu", "cuda:0", ...).
    kind: "cpu" or "cuda".
    arch: SM version of a CUDA device (e.g. 89, 120); 0 on the CPU.
    rt_cores: a GPU with ray-tracing cores (`has_rt_cores`): `traversal="auto"` traces with OptiX
      there and with the software traversal elsewhere (`trace.backend.select`).
    graphs: CUDA graph capture (`Renderer.capture`, `SensorCamera.capture`).
    pinned: page-locked host memory, so device -> host copies need no synchronisation.
    device_textures: textures are built on the device (sRGB decode and mip chain as kernels,
      device -> CUDA-array copies) and the texture unit decodes 8-bit sRGB color before filtering
      (`TextureStorage` "srgb8"); otherwise textures are built on the host and color is float16.
    parallel_launch: a kernel launch runs on many threads; a CPU launch runs on one, so host-side
      NumPy or launches split in bands on host threads are faster there.
    blackwell_rtx: a Blackwell RTX GPU (`arch` >= 120), whose tuned defaults differ.
  """

  alias: str
  kind: Literal["cpu", "cuda"]
  arch: int
  rt_cores: bool
  graphs: bool
  pinned: bool
  device_textures: bool
  parallel_launch: bool
  blackwell_rtx: bool

  @functools.cached_property
  def optix(self) -> OptixSupport:
    """The OptiX traversal's device-level requirements (probed on first access: loads the driver's
    OptiX function table)."""
    if self.kind != "cuda":
      return OptixSupport(available=False, reason="it needs a CUDA device")
    from raywarp.core.trace.optix import api  # noqa: PLC0415  # the OptiX stack: on first probe
    from raywarp.core.trace.optix.device import intrinsics as optix_device

    reason = api.unavailable_reason()
    if reason is not None:
      return OptixSupport(available=False, reason=reason)
    inc = optix_device.include_dir()
    if inc is None:
      return OptixSupport(available=False, reason=f"the bundled OptiX SDK headers are missing ({optix_device.INCLUDE_DIR})")
    reason = optix_device.headers_reason(inc)
    return OptixSupport(available=reason is None, reason=reason)

  def capturing(self) -> bool:
    """Whether the device's current stream is recording a CUDA graph (never on the CPU)."""
    return self.graphs and wp.get_stream(self.alias).is_capturing


def device_caps(device: wp.DeviceLike) -> DeviceCaps:
  """The capabilities of `device` (resolved once per device; None: Warp's default device)."""
  return _resolve(wp.get_device(device).alias)


def host_copy(a: wp.array) -> np.ndarray:
  """`a`'s values on the host, as `a.numpy()` returns them (a CPU array: its alias), without
  touching the device's current stream.

  `array.numpy()` copies a CUDA array on the device's null stream by making it the device's
  current stream for the copy (`wp.ScopedStream`). The current stream is process-wide: two
  threads doing that at once can leave the null stream current for good, and every later launch
  runs there (an OptiX launch then failed). This copy names the null stream instead (`wp.copy`'s
  `stream`): the same order (after the earlier work of the device's blocking streams), and a copy
  to pageable memory returns once it is done. raywarp reads device arrays only through it.
  """
  device = wp.get_device(a.device)
  if not device.is_cuda or not a.ptr:
    return a.numpy()
  out = wp.empty(a.shape, dtype=a.dtype, device="cpu")
  wp.copy(out, a, stream=device.null_stream)
  return out.numpy()


def stream_handle(stream: wp.Stream) -> int:
  """The CUDA handle of `stream` for a native call (OptiX, CUDA): 0 for the device's null stream,
  whose Warp handle is None (a thread of the caller's may hold it current: `host_copy`)."""
  return int(stream.cuda_stream or 0)


@functools.cache
def _resolve(alias: str) -> DeviceCaps:
  """Builds the record of the device `alias` (cached: devices do not change in a process)."""
  device = wp.get_device(alias)
  cuda = device.is_cuda
  arch = int(device.arch) if cuda else 0
  if cuda:
    from raywarp.core.build.kernel_cache import prebuilt  # noqa: PLC0415  # scene.model.wpbr imports this module

    prebuilt.seed(arch)  # before any module of this device loads
  return DeviceCaps(
    alias=alias,
    kind="cuda" if cuda else "cpu",
    arch=arch,
    rt_cores=cuda and has_rt_cores(arch),
    graphs=cuda,
    pinned=cuda,
    device_textures=cuda,
    parallel_launch=cuda,
    blackwell_rtx=cuda and arch >= _BLACKWELL_RTX_ARCH,
  )


# launch(a, b): the kernels of one band, over [a, b) of the launch's range.
type BandLaunch = Callable[[int, int], None]


def launch_bands(
  module: wp.Module, *, device: wp.DeviceLike, n: int, launch: BandLaunch, min_band: int, per_core: int = 1
) -> None:
  """Runs `launch(a, b)` (kernels of `module` launched on `device` over [a, b) of a range of `n`)
  over [0, n).

  One call where launches run in parallel; on the CPU (one thread per launch) `per_core` bands of
  at least `min_band` per core (more than one balances bands of uneven cost: a core that ends
  first takes the next), run on the shared band threads (`_POOL`): a launch releases the
  GIL, and each band writes its own outputs, so the result is that of the single launch.
  Concurrent callers (worker threads) queue their bands on the same threads; a call from a band
  itself runs as one launch. `module` is loaded first (one compile, not one per thread).
  `launch` passes `device=` to its launches: the bands set no default device (Warp's default
  device is process-wide: a `wp.ScopedDevice` per band thread raced, and its last restore could
  leave the caller's process on the CPU device).

  Raises:
    ValueError: for `min_band` or `per_core` < 1.
    Exception: the first failing band's, noting how many bands failed.
  """
  if min_band < 1:
    raise ValueError(f"launch_bands: min_band must be >= 1, got {min_band}")
  if per_core < 1:
    raise ValueError(f"launch_bands: per_core must be >= 1, got {per_core}")
  dev = wp.get_device(device)
  serial = device_caps(dev).parallel_launch or _BAND.inside
  nband = 1 if serial else min(process_cores() * per_core, n // min_band)
  if nband <= 1:
    launch(0, n)
    return
  wp.load_module(module, device=dev)

  def run(k: int) -> None:
    """Band k of `nband`."""
    launch(n * k // nband, n * (k + 1) // nband)

  # Every band ends before this returns, also when one raises (`Executor.map` would cancel the
  # pending bands and leave the running ones writing the caller's arrays after it raised).
  bands = [_POOL.get().submit(run, k) for k in range(nband)]
  concurrent.futures.wait(bands)
  errors = [e for band in bands if (e := band.exception()) is not None]
  if errors:
    errors[0].add_note(f"launch_bands: {len(errors)} of {nband} bands failed (this is the first)")
    raise errors[0]


@final
class _BandThread(threading.local):
  """Per-thread flag of the band threads (`launch_bands` from inside a band runs as one launch:
  no nested waits)."""

  inside = False


_BAND: Final = _BandThread()


def _band_thread(mask: set[int] | None) -> None:
  """Initialiser of a band thread: marks it and gives it the process's affinity `mask` (a new
  thread inherits its creator's, which may be a narrower-pinned worker thread)."""
  _BAND.inside = True
  if mask is not None and sys.platform == "linux":
    os.sched_setaffinity(0, mask)


@final
class _BandPool:
  """The process's band threads, one per core of its affinity, created on first use by any
  thread (sized and pinned by the process's mask, not the caller's) under a lock: concurrent
  first callers share one pool (an unlocked cache let each build its own, whose threads then
  idled for the process's life)."""

  def __init__(self) -> None:
    """No threads yet."""
    self._lock = threading.Lock()
    self._pool: concurrent.futures.ThreadPoolExecutor | None = None

  def get(self) -> concurrent.futures.ThreadPoolExecutor:
    """The pool (created by the first call)."""
    with self._lock:
      if self._pool is None:
        mask = _process_mask()
        self._pool = concurrent.futures.ThreadPoolExecutor(
          process_cores(), thread_name_prefix="rw_band", initializer=_band_thread, initargs=(mask,)
        )
      return self._pool

  def forget(self) -> None:
    """After a fork, in the child: none of the parent's threads (nor its lock state) exist there."""
    self._lock = threading.Lock()
    self._pool = None


# The band threads live as long as the process (`concurrent.futures` joins them at exit).
_POOL: Final = _BandPool()

# A forked child has none of the parent's threads (and maybe other cores): it makes its own pool
# (POSIX; Windows has no fork). A later `os.fork()` from a multi-threaded process gets Python's
# (3.12+) DeprecationWarning; the child does not deadlock: it never waits on the parent's band
# threads, nor on a lock a parent thread held while forking.
if sys.platform != "win32":
  os.register_at_fork(after_in_child=_POOL.forget)


def _process_mask() -> set[int] | None:
  """The process's CPU affinity (its main thread's on Linux: `sched_getaffinity(0)` would be the
  calling thread's); None where the OS has none."""
  if sys.platform == "linux":
    return os.sched_getaffinity(os.getpid())
  return None


def process_cores() -> int:
  """The cores this process may run on (its affinity mask where the OS has one: Linux)."""
  mask = _process_mask()
  return len(mask) if mask is not None else os.cpu_count() or 1
