"""Compiles the Warp kernels of a renderer's first frame in the background, in parallel, so that
the first render does not compile them one after the other.

Warp compiles a module when one of its kernels is first launched, so a cold first frame compiles
its ~10 modules one after the other (the GI probe tracer alone takes 3-7 s of NVRTC). `start`
compiles them on background threads instead (NVRTC releases the GIL): the first frame waits for
the slowest module rather than for their sum. `first_frame` lists the kernels, as `wp.Kernel`
objects (imports: a renamed or moved kernel fails loudly), from the renderer's Features and its
trace backend's `first_frame_kernels`; a kernel missing from it still compiles on first launch.
Modules already in Warp's kernel cache load in milliseconds, so a warm start is unaffected.

Only renderer-owned kernels are listed: user binding modules compile on their ordinary launch
path, not on an unsolicited background thread. A module loads on one thread at a time, whoever
loads it (a warm-up, a launch on another thread, the caller's code):
`core/pipeline/platform/module_lock.py`.

The caller's own Warp work may run meanwhile (`tests/unit/test_warmup_threads.py`: a simulation,
host reads and a second Renderer while the modules load): the threads pass the device
explicitly (never `wp.ScopedDevice` / `ScopedStream`, whose save and restore are process-wide,
and no `array.numpy()`), `wp.force_load` restores the thread's CUDA context, and Warp (1.18)
generates code under its codegen lock and compiles in per-thread build directories.
"""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, final

import numpy as np
import warp as wp

from raywarp.core.build.kernel_cache import policy, prebuilt
from raywarp.core.kernels import probe
from raywarp.core.kernels.abi.features import software_view
from raywarp.core.pipeline import baked, cameras
from raywarp.core.pipeline.derive import fields
from raywarp.core.pipeline.platform import cadence
from raywarp.core.pipeline.platform.device import device_caps, process_cores
from raywarp.core.pipeline.worlds import optics, validator_kernels
from raywarp.core.pipeline.worlds import tables as per_world
from raywarp.core.post.display import deferred
from raywarp.core.post.sensor import effects
from raywarp.core.post.temporal import history
from raywarp.core.shade.gi import atlas as gi_atlas
from raywarp.core.shade.gi import cube as gi_cube
from raywarp.core.shade.gi import groups as gi_groups
from raywarp.core.shade.gi import kernels as gi_kernels
from raywarp.core.shade.gi import rebakes as gi_rebakes
from raywarp.core.shade.gi import relight as gi_relight
from raywarp.core.shade.gi import slots as gi_slots
from raywarp.core.shade.lights import flood, grid
from raywarp.core.shade.lights import pack as light_pack
from raywarp.core.shade.shadows import area as area_shadows
from raywarp.core.trace.accel import bounds, deformable_bvh, lbvh, packed

if TYPE_CHECKING:
  from types import ModuleType

  from raywarp.core.trace.backend import TraceBackend
  from raywarp.core.trace.context import FrameContext

# Compile threads per block size (each runs one module's compile): the process's cores (its
# affinity mask), at most 16.
MAX_WORKERS: Final = min(16, process_cores())


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CompileStatus:
  """What a Renderer's kernels are doing (`Renderer.compile_status`).

  Attributes:
    pending: the Warp modules still compiling or loading in the background (empty: ready).
    modules: the modules its current kernels need (`first_frame`).
    elapsed_s: seconds since that compile started.
  """

  pending: tuple[str, ...]
  modules: int
  elapsed_s: float


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FirstFrame:
  """The Warp kernels of a frame, by the block size they are launched with.

  Attributes:
    default: launched with Warp's default block size.
    render: launched with the renderer's (`Features.block_dim`): the megakernels.
  """

  default: tuple[wp.Kernel, ...] = ()
  render: tuple[wp.Kernel, ...] = ()

  def modules(self) -> tuple[list[wp.Module], list[wp.Module]]:
    """(default, render) Warp modules of the kernels, each once, in listing order."""
    return _unique(k.module for k in self.default), _unique(k.module for k in self.render)


class WarpModule(NamedTuple):
  """One Warp module a frame loads, at the block size it loads with (None: Warp's default): a unit
  of a software renderer's nearby warm (`Renderer.nearby_modules`)."""

  module: wp.Module
  block_dim: int | None


def frame_modules(ctx: FrameContext, backend: TraceBackend) -> tuple[WarpModule, ...]:
  """The Warp modules of a first frame of `ctx` on `backend` (`first_frame`), each once."""
  default, render = first_frame(ctx, backend).modules()
  return (*(WarpModule(m, ctx.features.block_dim) for m in render), *(WarpModule(m, None) for m in default))


def load(device: wp.DeviceLike, units: list[WarpModule], workers: int) -> tuple[int, int, float]:
  """Loads `units` on `device`, `workers` at a time, the bundled ones fetched first (each compiled
  into Warp's kernel cache when missing): (loaded, 0, seconds), `nearby.Plan.create`'s shape."""
  start = time.perf_counter()
  for block_dim in dict.fromkeys(u.block_dim for u in units):
    _fetch(device, [u.module for u in units if u.block_dim == block_dim], block_dim)
  with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="rw_module") as pool:
    list(pool.map(lambda u: _load_module(device, u.module, u.block_dim), units))
  return len(units), 0, time.perf_counter() - start


def _unique(modules: Iterable[wp.Module]) -> list[wp.Module]:
  """`modules` without repeats (each loads on one compile thread)."""
  return list(dict.fromkeys(modules))


def kernel(k: Callable[..., None]) -> wp.Kernel:
  """A module-level `@wp.kernel` as the `wp.Kernel` it is at run time (the Warp stubs type the
  decorated function as itself).

  Raises:
    TypeError: for a function that is not a Warp kernel.
  """
  if not isinstance(k, wp.Kernel):
    raise TypeError(f"warm-up entry {k!r} is not a Warp kernel")
  return k


def module_kernels(module: ModuleType) -> tuple[wp.Kernel, ...]:
  """The module-level kernels of a Python module whose kernels are private to it.

  Raises:
    RuntimeError: when it holds none (a warm-up entry that would warm nothing).
  """
  found = tuple(wp.get_module(module.__name__).kernels.values())
  if not found:
    raise RuntimeError(f"warm-up entry {module.__name__} holds no module-level Warp kernel")
  return found


def first_frame(ctx: FrameContext, backend: TraceBackend | type[TraceBackend]) -> FirstFrame:
  """The kernels of a renderer's first frame (as far as known: see the module doc)."""
  f = ctx.features
  own = backend.first_frame_kernels(f, ctx.camera_model, device_caps(ctx.scene.device))
  # Only explicitly listed renderer kernels. A binding's defining module can hold arbitrary
  # simulator/user kernels: sweeping it here races the caller's own Warp launches.
  default = [*module_kernels(cameras), *own.default]
  if ctx.scene.nlight:
    default.append(kernel(light_pack.pack))
  if _validates(ctx):
    default.append(kernel(validator_kernels.validate))
  # What the frames can bake and build: the build's GI and light grid, not only the settings'
  # (`baked.Baked.probes`): a run-time change turns them on without another build (an OptiX tier
  # build compiles both in, `tiers.BOUND_FREE`), so their kernels load now, not in that frame.
  gi, culling = f.gi, f.culling
  if gi or culling:
    default.append(kernel(bounds.instance_bounds))  # the software scene BVH of the probe / grid rays
    default.extend(packed.KERNELS)  # its packed nodes (the OptiX backend builds it on demand)
  if (gi and ctx.settings.gi.update_every > 0) or ctx.settings.performance.rebuild_every > 0:
    default.append(kernel(cadence.tick))  # periodic GI updates / BVH rebuilds
  world_tree = bool(np.asarray(ctx.layout.per_world, bool).any() or ctx.scene.deformables.count)
  if world_tree and (gi or culling or f.traversal == "software"):
    default.extend(module_kernels(lbvh))  # the software world BVH's rebuilds (checked at its build)
  # The relight build exists where the settings turn GI on (`resolve.relight_build`): an OptiX tier
  # build compiles GI in for every settings, GI off has nothing to relight.
  if gi and ctx.relight is not None and _relights(ctx):
    # Per-world GI inputs (declared or batched tables): the first re-randomisation relights the
    # shared bake (`gi.relight`): its kernels now, not in that frame (a multi-second stall).
    # Its tint by per-light slots and the re-bake bookkeeping too (`shade/gi/slots.py`,
    # `shade/gi/rebakes.py`).
    default += [baked.relight_kernel(ctx), *module_kernels(gi_relight), *module_kernels(gi_slots), *module_kernels(gi_rebakes)]
  if ctx.scene.deformables.count:
    default.append(kernel(deformable_bvh.face_normals))  # their triangles and normals, every frame
  if _writes_columns(ctx):
    default.extend(module_kernels(per_world))  # `InstanceColumns` writes / reads, the live camera gather
  if f.history:
    default.extend(module_kernels(history))  # frame bookkeeping
  default.extend(_feature_kernels(ctx, gi=gi, culling=culling, probes=gi and not backend.hardware_probes(f)))
  default.extend(_sensor_kernels(ctx))
  if f.any_rgb:
    default.extend(deferred.first_frame_kernels(ctx.settings))
  if f.any_rgb and ctx.settings.area_rays():
    # The soft area shadows' pass: its filter, and on the software traversal its tracing kernel
    # (OptiX: a program of its own, `programs.build_area`).
    default.append(kernel(area_shadows.apply_kernel))
    if f.traversal == "software":
      default.append(area_shadows.software_kernel(f))
  # The derived per-world fields' kernels (`derived_fields` imports this module: read at call time).
  default.extend(
    fields.kernels(
      ctx.per_world_fields,
      material_selection=fields.selects_shading(ctx.scene, ctx.per_world_fields),
      shininess=fields.reads_shininess(ctx.scene, ctx.per_world_fields),
    )
  )
  if f.world_optics:
    default.append(kernel(optics.compose_optics))  # the per-world instance optics
  return FirstFrame(default=tuple(default), render=own.render)


def _relights(ctx: FrameContext) -> bool:
  """Whether the worlds' GI radiance inputs can differ (`baked.gi_tracked`, from the declarations
  and the binding's tables: no frame yet), so that a bake can be relit."""
  if ctx.nworld <= 1:
    return False
  if any(gi_groups.gi_slot(name) for name in ctx.per_world_fields):
    return True
  slots = {**ctx.scene.tables, **ctx.scene.binding.slots(None)}
  return any(
    gi_groups.gi_slot(name) and isinstance(a, wp.array) and a.ndim >= 1 and a.shape[0] > 1 for name, a in slots.items()
  )


def _writes_columns(ctx: FrameContext) -> bool:
  """Whether the renderer launches `per_world`'s scatter / gather: instance columns of a declared
  geometry group or instance field (`Renderer.per_world.columns`, written after the first frame:
  their module loads now, not at the first write), or live scene-camera tables
  (`per_world.LiveCameras`)."""
  groups = ctx.layout.groups.values()
  declared = any(mask.any() for mask in groups) or any(f.startswith("instance.") for f in ctx.per_world_fields)
  return bool(declared or ctx.scene.binding.cameras(None))


def _validates(ctx: FrameContext) -> bool:
  """Whether the frame's validation kernel runs (`validator.Validator.active`, as far as the
  context tells): shared scenery, per-world tables or a baked effect checked across worlds."""
  f = ctx.features
  return bool(ctx.layout.nshared or ctx.per_world_fields or (ctx.nworld > 1 and (f.gi or f.culling)))


def _sensor_kernels(ctx: FrameContext) -> list[wp.Kernel]:
  """The sensor pass's kernels (`core/post/sensor/effects.py`) for the build's color / depth
  outputs: every sensor setting and the outputs are run time, so turning noise, vignette, response
  or the depth model on, or requesting the color or depth later, must load them, not compile them
  (both color-pass forms: in place, or from the clean copy)."""
  out: list[wp.Kernel] = []
  if ctx.features.any_rgb:
    out += [effects.make_color(direct=direct) for direct in (True, False)]
    out.append(effects.make_refresh(depth=False))
  out.append(effects.make_refresh(depth=True))
  out.append(effects.make_stereo())
  return [*out, *module_kernels(effects)]


def _feature_kernels(ctx: FrameContext, *, gi: bool, culling: bool, probes: bool) -> list[wp.Kernel]:
  """The factory-made kernels of `ctx`'s first frame: GI probes and light grid; as the software
  kernels beside the build are compiled (`software_view`), with the bakes the frames run (`gi`,
  `culling`; `probes`: the software probe-ray kernel traces the GI's probe rays,
  `TraceBackend.hardware_probes`)."""
  view = software_view(ctx.features, ctx.effective)
  f = dataclasses.replace(view, gi=gi, culling=culling)
  shape_types = tuple(f.shape_types)
  trees = f.trees
  out: list[wp.Kernel] = []
  if probes:
    out.append(probe.probe_kernel(f))
  if f.gi:
    q = {"trees": trees, "alpha_cutout": f.alpha_cutout, "live_uv": f.live_material_uv, "world_uv": f.world_uv}
    # The bake's own kernels (probe groups, atlas, cube, the hidden-instance colors).
    out += [k for m in (gi_groups, gi_atlas, gi_cube, baked) for k in module_kernels(m)]
    out += [
      kernel(gi_kernels.rotate_rays),
      gi_kernels.survey_kernel(shape_types, **q),
      gi_kernels.walls_kernel(shape_types, **q),
      gi_kernels.ao_kernel(shape_types, **q),
      gi_kernels.static_kernel(shape_types, **q),
    ]
  if f.culling:
    out += [
      kernel(flood.seed_kernel),
      grid.visibility_kernel(
        shape_types, area=f.area_lights, trees=trees, alpha_cutout=f.alpha_cutout, live_uv=f.live_material_uv
      ),
    ]
  return out


@final
class Warmup:
  """`first_frame`'s modules compiling on background threads (one per block size).

  A compile error is kept and raised by `wait`, before the frame launches anything.
  Workers finish before interpreter teardown, including when no first frame is rendered:
  abandoning native compiler work can race Warp's temporary-directory and runtime cleanup (an
  interrupted script exits once its compile is done).
  """

  def __init__(self, ctx: FrameContext, backend: TraceBackend, after: Warmup | None) -> None:
    """Starts the compile threads of a renderer's first-frame modules (see `first_frame`).

    `after`: an earlier warm-up of the same renderer not waited for (its Features changed
    meanwhile): these threads wait for it first, so that no module loads on two threads.
    """
    default, render = first_frame(ctx, backend).modules()
    self.modules = default + render
    self._after = after
    self._errors: list[Exception] = []
    self.started = time.perf_counter()
    # The modules loaded so far (`status`; set by the loader threads).
    self.loaded: set[str] = set()
    device = ctx.scene.device
    self._threads = [
      threading.Thread(target=self._load, args=(device, mods, block_dim), name="rw_warmup", daemon=False)
      for mods, block_dim in ((render, ctx.features.block_dim), (default, None))
      if mods
    ]
    for t in self._threads:
      t.start()

  def _load(self, device: wp.DeviceLike, mods: list[wp.Module], block_dim: int | None) -> None:
    """One compile thread: after the earlier warm-up, fetches the bundled modules of `mods` Warp's
    cache lacks (`prebuilt.fetch_modules`: one parallel batch), then loads `mods` (errors kept for
    `wait`)."""
    if self._after is not None:
      self._after.join()
    try:
      _fetch(device, mods, block_dim)
      self._run(device, mods, block_dim)
      self.loaded.update(m.name for m in mods)
    except Exception as e:  # noqa: BLE001 (any compile error: raised again by `wait`, on the caller's thread)
      self._errors.append(e)

  @staticmethod
  def _run(device: wp.DeviceLike, mods: list[wp.Module], block_dim: int | None) -> None:
    """Loads distinct modules `mods` on parallel threads (`_load_module`)."""
    if len(mods) == 1:
      _load_module(device, mods[0], block_dim)
      return
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="rw_module") as pool:
      futures = [pool.submit(_load_module, device, mod, block_dim) for mod in mods]
      for future in futures:
        future.result()

  def status(self) -> tuple[tuple[str, ...], int, float]:
    """(the modules still loading, the modules listed, seconds since the start)."""
    pending = tuple(m.name for m in self.modules if m.name not in self.loaded)
    return pending, len(self.modules), time.perf_counter() - self.started

  def join(self) -> None:
    """Blocks until its threads (and the earlier warm-up's) are done, errors or not."""
    if self._after is not None:
      self._after.join()
    for t in self._threads:
      t.join()

  def wait(self) -> None:
    """Blocks until every listed module is loaded (and the earlier warm-up's threads are done).

    Raises:
      Exception: the first compile error of its threads (Warp's build error; the earlier
        warm-up's are not raised: its Features are no longer the renderer's).
    """
    self.join()
    if self._errors:
      raise self._errors[0]


def start(ctx: FrameContext, backend: TraceBackend, after: Warmup | None = None) -> Warmup:
  """Starts compiling a renderer's first-frame modules on background threads (`after`: see
  `Warmup`)."""
  return Warmup(ctx, backend, after)


def _fetch(device: wp.DeviceLike, mods: list[wp.Module], block_dim: int | None) -> None:
  """Fetches the prebuilt Warp modules of `mods` (their cache directory names: Warp's content hash
  at the block size they load with) into Warp's kernel cache, on a CUDA device. With compiling off
  (`policy`), a module Warp's cache still lacks raises `policy.KernelsNotPrebuilt`."""
  arch = device_caps(device).arch
  names = [m.get_module_identifier(block_dim if block_dim is not None else m.options["block_dim"]) for m in mods]
  if arch:
    prebuilt.fetch_modules(arch, names)
  if not policy.current().compile:
    missing = [n for n in names if not _cached(n, arch)]
    if missing:
      where = f"sm_{arch}" if arch else "the CPU (no prebuilt CPU kernels)"
      policy.refuse(f"{len(missing)} Warp modules of this renderer on {where} (e.g. {missing[0]})")


def _cached(name: str, arch: int) -> bool:
  """Whether Warp's kernel cache holds module directory `name`'s binary for SM version `arch`
  (0: the CPU's object file)."""
  root = wp.config.kernel_cache_dir
  folder = Path(root) / name if root is not None else None
  if folder is None or not folder.is_dir():
    return False
  binary = re.compile(rf"\.sm{arch}[a-z]?\.(cubin|ptx)$") if arch else re.compile(r"\.o$")
  return any(binary.search(f.name) for f in folder.iterdir())


def _load_module(device: wp.DeviceLike, module: wp.Module, block_dim: int | None) -> None:
  """Loads `module` on `device` (`wp.force_load`; a module loads on one thread at a time:
  `module_lock`)."""
  wp.force_load(device=device, modules=[module], block_dim=block_dim, max_workers=0)
