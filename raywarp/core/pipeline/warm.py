"""`Renderer.warm`: a renderer configuration prepared ahead, in the background, so that its first
frame (in this process or a later one) finds everything on disk and in memory.

A warm takes the constructor's arguments (plus the simulator state its frame renders) and does
what the configuration's first frame would, keeping every result in the caches:

- `renderer`: a `Renderer` of the configuration: its first frame's Warp modules load (compiled
  into Warp's kernel cache when missing; the prebuilt kernel bundle seeded on first use) and the
  OptiX modules the frame binds are created into OptiX's disk cache (`Renderer.wait_compiles`);
- `frame`: one frame: the structures (GAS cache), the bakes (light grid, GI,
  environment prefilter: the bake cache) and the launch;
- `nearby` (with `nearby=N`): the modules of the settings one change away from the
  configuration's (`Renderer.nearby_modules`: the output sets given, every compile-time field over
  its domain, at most N modules, the plan's count and estimated seconds in the result), after the
  frame, several at once: OptiX modules created into OptiX's disk cache from bundled or cached PTX
  only (never NVRTC), so that a later settings change binds a module OptiX serves from its cache;
  on the software traversal the Warp modules of those settings' first frames (each output set's
  megakernel among them), fetched from the bundle or compiled into Warp's kernel cache.

The scene's own preparation is the setup's usual code, run before: an adapter's load converts a
USD stage into the `.wpbr` cache (`raywarp.formats.usd.load(..., cache=True)`), `upload` encodes
the material textures into the texture cache.

Concurrency: `warm` returns a `concurrent.futures.Future` at once (`asyncio.wrap_future` for an
event loop); many warms run together on a thread pool. Their OptiX module creation (the long,
CPU-bound part of a cold start) overlaps; the device steps (the construction with its Warp module
loads, the frame) of one device run one warm at a time (a lock per device: each warm's steps are
timed alone, and the device's work is not split between warms). Renderers on several threads are
safe without it (`Renderer`): raywarp never changes Warp's process-wide device or stream, and a
Warp module loads on one thread at a time (`module_lock`). A warm of a request already warming
returns its future (deduplication), and one OptiX module is created once across threads and
processes (`trace/optix/modules.create`'s lock).

The result reports each step's seconds and what it added to the caches (files of raywarp's cache
directory and entries of Warp's kernel cache; OptiX modules compiled vs served by OptiX's disk
cache): a warm of a configuration that is already warm adds nothing.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import os
import sys
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp.cache.paths import cache_dir
from raywarp.core.build.variants import nearby
from raywarp.core.build.variants.nearby import Plan
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, GpuScene
from raywarp.core.pipeline.warmup import WarpModule
from raywarp.core.pipeline.worlds.sharing import PerWorld
from raywarp.settings import Outputs, RenderSettings

if TYPE_CHECKING:
  from raywarp.core.pipeline.renderer import Renderer
  from raywarp.core.trace.optix.launch import Unit

# Warms running at once (their device steps take turns per device).
WORKERS: Final = 8
# Seconds between two looks at a renderer's Warp module loads (`Renderer.compile_status`).
_POLL_S: Final = 0.005

_LOCK: Final = threading.Lock()
# The running warms by request (`warm`'s key): deduplication.
_WARMS: Final[dict[tuple[object, ...], concurrent.futures.Future[Warmed]]] = {}
# One lock per device (its device steps).
_DEVICES: Final[dict[str, threading.Lock]] = {}
# The warms' thread pool (made on first use).
_POOL: Final[list[concurrent.futures.ThreadPoolExecutor]] = []


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Part:
  """One step of a warm.

  Attributes:
    name: "renderer", "frame" or "nearby" (see the module doc).
    seconds: its wall time.
    added: the cache entries it added (raywarp's cache files, Warp kernel cache entries; with
      other warms running, theirs too); 0: all cache hits.
    modules: OptiX modules (compiled, served by OptiX's disk cache) meanwhile, process-wide.
  """

  name: str
  seconds: float
  added: int
  modules: tuple[int, int] = (0, 0)

  def __str__(self) -> str:
    """E.g. "renderer 33.10 s (+2 cached, OptiX 1 compiled / 1 cached)"."""
    what = f"+{self.added} cached" if self.added else "hit"
    if any(self.modules):
      what += f", OptiX {self.modules[0]} compiled / {self.modules[1]} cached"
    return f"{self.name} {self.seconds:.2f} s ({what})"


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Warmed:
  """A warmed renderer configuration: its steps, in order, and the plan of its modules of the
  settings one change away (`nearby`: None without)."""

  parts: tuple[Part, ...]
  nearby: Plan[Unit] | Plan[WarpModule] | None = None

  @property
  def seconds(self) -> float:
    """Its total wall time."""
    return sum(p.seconds for p in self.parts)

  def __str__(self) -> str:
    """One line: total, then each step (and the nearby plan)."""
    line = f"warm {self.seconds:.2f} s: " + ", ".join(str(p) for p in self.parts)
    return line if self.nearby is None else f"{line}; nearby: {self.nearby}"


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Config:
  """A warm's renderer class (`Renderer`, given by `Renderer.warm`: this module builds renderers
  without importing it, which imports this module), constructor arguments (`Renderer.__init__`),
  its frame's simulator state and the modules of the settings one change away it creates (`nearby`:
  at most so many, 0: none; the output sets `nearby_outputs` among them, first)."""

  renderer: type[Renderer]
  nworld: int
  resolution: tuple[int, int]
  cameras: Sequence[CameraSpec | str | int] | None
  settings: RenderSettings | None
  per_world: PerWorld | None
  sim: object
  nearby: int = 0
  nearby_outputs: tuple[Outputs, ...] = ()


def warm(scene: GpuScene, config: Config) -> concurrent.futures.Future[Warmed]:
  """Starts warming `scene` at `config` (`Renderer.warm`; see the module doc); returns at once."""
  same = (repr(getattr(config, f.name)) for f in dataclasses.fields(config) if f.name != "sim")
  key = (id(scene), id(config.sim), *same)
  with _LOCK:
    found = _WARMS.get(key)
    if found is not None:
      return found
    if not _POOL:
      _POOL.append(concurrent.futures.ThreadPoolExecutor(WORKERS, thread_name_prefix="rw_warm"))
    future = _POOL[0].submit(_warm, scene, config)
    _WARMS[key] = future
  # Shared while it runs only: the objects' ids may be reused once they are gone.
  future.add_done_callback(lambda _: _forget(key))
  return future


def _forget(key: tuple[object, ...]) -> None:
  """Drops a finished warm from the deduplication table."""
  with _LOCK:
    _WARMS.pop(key, None)


def _warm(scene: GpuScene, config: Config) -> Warmed:
  """The warm's steps (see the module doc)."""

  def build() -> Renderer:
    """The renderer and its Warp module loads (a device step), then its OptiX modules waited for
    off the device (`Renderer.wait_compiles`)."""
    with _device(scene.device):
      r = config.renderer(
        scene,
        nworld=config.nworld,
        cameras=config.cameras,
        resolution=config.resolution,
        settings=config.settings,
        per_world=config.per_world,
      )
      try:
        while r.compile_status().pending:
          time.sleep(_POLL_S)
      except BaseException:
        r.close()
        raise
    try:
      r.wait_compiles()
    except BaseException:
      with _device(scene.device):
        r.close()
      raise
    return r

  r, made = _timed("renderer", build)
  try:

    def frame() -> None:
      """One frame (structures, bakes, launch), synchronized."""
      with _device(scene.device):
        r.render(config.sim)
        wp.synchronize_device(scene.device)

    _, shot = _timed("frame", frame)
    # After the frame: a build without bundled PTX has compiled its own by then (cached), and the
    # neighbours sharing it find it (`OptixRenderer.create_modules` never compiles PTX).
    plan = r.nearby_modules(config.nearby, config.nearby_outputs) if config.nearby > 0 else None
    near = None if plan is None else _timed("nearby", functools.partial(plan.create_all, nearby.WORKERS))[1]
    parts = (made, shot) if near is None else (made, shot, near)
  finally:
    with _device(scene.device):
      r.close()
  return Warmed(parts, plan)


@contextmanager
def _device(device: wp.DeviceLike) -> Generator[None]:
  """Holds `device`'s lock (one warm's device step at a time; Warp's default device, which is
  process-wide, is left as it is: the renderer names its device)."""
  alias = wp.get_device(device).alias
  with _LOCK:
    lock = _DEVICES.setdefault(alias, threading.Lock())
  with lock:
    yield


def _timed[T](name: str, step: Callable[[], T]) -> tuple[T, Part]:
  """(`step()`, its Part: seconds, cache entries added, OptiX modules compiled / cached)."""
  before, modules, start = _entries(), _modules(), time.perf_counter()
  value = step()
  seconds = time.perf_counter() - start
  after = _modules()
  return value, Part(name, seconds, max(0, _entries() - before), (after[0] - modules[0], after[1] - modules[1]))


def _entries() -> int:
  """Files of raywarp's cache directory (locks and temporaries aside) plus entries of Warp's
  kernel cache."""
  root, kernels = cache_dir(), wp.config.kernel_cache_dir
  return (_files(str(root)) if root is not None else 0) + (len(_names(kernels)) if kernels else 0)


def _files(root: str) -> int:
  """Files under `root` but locks and temporaries (entries removed meanwhile by another warm's
  writer are skipped)."""
  count = 0
  for name in _names(root):
    path = os.path.join(root, name)
    if os.path.isdir(path):
      count += _files(path)
    elif not name.endswith(".lock") and ".tmp" not in name and ".partial" not in name:
      count += 1
  return count


def _names(directory: str) -> list[str]:
  """The entries of `directory` (none when it is absent or vanished)."""
  try:
    return os.listdir(directory)
  except OSError:
    return []


def _modules() -> tuple[int, int]:
  """OptiX modules of this process so far: (compiled, served by the disk cache); none before
  OptiX is loaded."""
  context = sys.modules.get("raywarp.core.trace.optix.context")
  if context is None:
    return 0, 0
  counts = context.MODULES
  return counts["compiled"], counts["cached"]
