"""Compiled OptiX programs: PTX, module, program groups, pipelines, shader binding tables, launches.

A build's programs (`programs.build`) compile once to PTX for `context.ptx_arch` (one NVRTC
compile; `PtxCompile` runs it off the render thread, NVRTC releasing the GIL; a warm start reads
the PTX from raywarp's cache or the prebuilt bundle by `programs.ptx_key` without defining the
programs, when it was generated from the installed kernel code: `prebuilt.Entry.fresh`), then
OptiX compiles the PTX into one module per `module_record` (register cap, bound values:
`trace/optix/modules.py`, OptiX's disk cache), which a `Pipeline` links. `Pipeline.trace`
launches a program over any number of launch indices, `nmajor` units (worlds, rays, lattice
points) of `minor` indices each: `optixLaunch` takes at most 2^30, so wider launches are split
into chunks of whole units, each reading its own copy of the launch parameters whose `offset` is
its first unit (the programs' index arithmetic stays in 32 bits).
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import dataclasses
import json
import os
import re
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.cache.paths import cache_dir
from raywarp.core.build.kernel_cache import kernel_code, policy, prebuilt
from raywarp.core.kernels.abi import toggles
from raywarp.core.pipeline.platform.device import device_caps, stream_handle
from raywarp.core.pipeline.platform.launches import struct_bytes
from raywarp.core.trace.optix import api
from raywarp.core.trace.optix.context import OptixObject, address, set_stack_size
from raywarp.core.trace.optix.device import intrinsics as device
from raywarp.core.trace.optix.device import params as params_lib
from raywarp.core.trace.optix.device.intrinsics import RAY_TYPES
from raywarp.core.trace.optix.device.programs import LAYER, LIGHT_VISIBILITY, RENDER, intersection_name
from raywarp.core.trace.optix.gas import curve_build_flags
from raywarp.core.trace.optix.group import CURVES, HitGroups
from raywarp.core.trace.rays.primary import LENS_MODELS

if TYPE_CHECKING:
  from collections.abc import Callable

  from raywarp.core.trace.optix.context import DeviceContext
  from raywarp.core.trace.optix.device.params import LaunchParams
  from raywarp.core.trace.optix.device.programs import SceneFlags

# Launch indices of one optixLaunch (OptiX's limit: width · height · depth <= 2^30).
MAX_LAUNCH: Final = 1 << 30
# The ray-generation program traces rays, through an IAS over GASes; each program it calls
# (`call_programs`) traces too, one level deeper (the layer program calls the light visibility
# program, which traces the shadow rays).
_TRACE_DEPTH: Final = 1
_MAX_TRAVERSABLE_DEPTH: Final = 2
# The programs called as miss programs (a trace of the null traversable), in their payload types'
# and miss records' order after the ray types' (`device.visibility_slot`), with their payload
# semantics.
_CALLS: Final = ((LAYER, device.layer_payload_semantics), (LIGHT_VISIBILITY, device.visibility_payload_semantics))
# Their PTX entries (`call_programs`; Warp's content hash suffix).
_CALL_ENTRIES: Final = tuple(
  (name, re.compile(rb"\.entry\s+" + name.encode() + rb"(?:_[0-9a-f]{8})?[\s(]")) for name, _ in _CALLS
)
# A lens camera model's render program (`programs.build_lens`): its module declares the layer
# call's payload type whatever the build (`module_calls`).
_LENS_ENTRY: Final = re.compile(
  rb"\.entry\s+" + RENDER.encode() + rb"_(?:" + b"|".join(m.name.encode() for m in LENS_MODELS) + rb")(?:_[0-9a-f]{8})?[\s(]"
)
# Warp suffixes PTX entry names with a content hash.
_ENTRY_HASH: Final = re.compile(r"_[0-9a-f]{8}$")


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CompileOptions:
  """A build's module and pipeline compile options, with the host arrays they point to.

  Attributes:
    module: OptixModuleCompileOptions (payload type, bound values, register cap).
    pipeline: OptixPipelineCompileOptions.
  """

  module: api.ModuleCompileOptions
  pipeline: api.PipelineCompileOptions


def compile_options(
  *,
  has_curves: bool,
  has_custom: bool,
  max_registers: int,
  opacity_micromaps: bool,
  bound: params_lib.BoundWords = (),
  calls: tuple[str, ...] = (),
) -> CompileOptions:
  """The compile options of a build's programs (`max_registers` 0: OptiX's default;
  `opacity_micromaps`: some GAS has them, `omm/micromaps.py`). The traversal supports triangles,
  plus linear curves (`has_curves`) and custom primitives (`has_custom`) only when the structures
  hold them. `bound`: launch-parameter words compiled in (`params.bound_words`). `calls`: the
  called programs the programs have (`call_programs`: the layer shading program, the light
  visibility program), each call a payload type of its own after the ray types': declared only
  then (OptiX compiles a program that cannot name its types, such as the built-in curve
  intersection, for every type of its module: a 25-register type there slowed a robot scene's
  frames 7 %)."""
  flags = api.PRIMITIVE_TYPE_FLAGS_TRIANGLE
  flags |= api.PRIMITIVE_TYPE_FLAGS_CUSTOM if has_custom else 0
  flags |= api.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR if has_curves else 0
  pco = api.PipelineCompileOptions(
    usesMotionBlur=0,
    traversableGraphFlags=api.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
    numPayloadValues=0,  # typed payloads: the module's `payloadTypes`
    numAttributeValues=device.NUM_ATTRIBUTE,
    exceptionFlags=api.EXCEPTION_FLAG_NONE,
    pipelineLaunchParamsVariableName=b"params",
    usesPrimitiveTypeFlags=flags,
    allowOpacityMicromaps=int(opacity_micromaps),
  )
  # Payload type 0: camera / occlusion traces; 1: shadow traces (`device.RAY_TYPES`); then the
  # calls' (`device.layer_call`, `device.visibility_call`).
  types = (device.payload_semantics(), device.shadow_payload_semantics())
  types += tuple(semantics() for name, semantics in _CALLS if name in calls)
  payload = (api.PayloadType * len(types))(*[api.PayloadType(len(sem), (ctypes.c_uint * len(sem))(*sem)) for sem in types])
  values = [ctypes.create_string_buffer(value, len(value)) for _, value in bound]
  entries = (api.ModuleCompileBoundValueEntry * max(len(bound), 1))()
  for entry, (offset, value), buf in zip(entries, bound, values, strict=False):
    entry.pipelineParamOffsetInBytes, entry.sizeInBytes = offset, len(value)
    entry.boundValuePtr = ctypes.cast(buf, ctypes.c_void_p)
    entry.annotation = f"params+{offset}".encode()
  mco = api.ModuleCompileOptions(
    maxRegisterCount=max_registers or api.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
    optLevel=api.COMPILE_OPTIMIZATION_DEFAULT,
    debugLevel=api.COMPILE_DEBUG_LEVEL_NONE,
    numBoundValues=len(bound),
    numPayloadTypes=len(types),
    payloadTypes=payload,
  )
  mco.boundValues = entries
  # The bound values' bytes: `boundValuePtr` is a plain address, so `entries` does not hold them.
  entries._values = values  # pyrefly: ignore[missing-attribute]  # ctypes arrays take attributes
  return CompileOptions(module=mco, pipeline=pco)


# The shape bits whose instances the pipeline traverses as curves / custom primitives
# (`trace/optix/device/trace.CURVE_TYPES`, `CUSTOM_TYPES`).
_CURVE_BITS: Final = toggles.SHAPE_SPHERE | toggles.SHAPE_CAPSULE
_CUSTOM_BITS: Final = toggles.SHAPE_CUSTOM


def module_record(scene: SceneFlags, max_registers: int, bound: params_lib.BoundWords, *, shapes: int) -> prebuilt.ModuleRecord:
  """The module options of a build's programs compiled for `scene`'s flags
  (`programs.program_flags`) with register cap `max_registers` (0: OptiX's default), binding
  `bound` (`params.bound_words`): one OptiX module (`trace/optix/modules.py`), as a bundle entry
  records it. Its pipeline traverses curves and custom primitives only where the shape bits
  `shapes` (`programs.traced_shapes`: the module word's) have them."""
  return prebuilt.ModuleRecord(
    has_curves=(scene.has_world_curves or scene.has_shared_curves) and bool(shapes & _CURVE_BITS),
    has_custom=bool(scene.primitive_types or scene.shared_primitive_types) and bool(shapes & _CUSTOM_BITS),
    opacity_micromaps=scene.has_omm,
    max_registers=max_registers,
    bound=tuple((offset, value.hex()) for offset, value in bound),
  )


def curves_module(ctx: DeviceContext, options: CompileOptions) -> int:
  """OptiX's built-in intersection module of the linear curves (sphere and capsule instances) beside
  a module of `options` (owned by OptiX, kept in its disk cache: `trace/optix/modules.create` makes
  it with each module whose record has curves, so a module pack holds it)."""
  curve_options = api.BuiltinISOptions(
    builtinISModuleType=api.PRIMITIVE_TYPE_ROUND_LINEAR,
    usesMotionBlur=0,
    buildFlags=curve_build_flags(),
    curveEndcapFlags=0,  # round end caps (the default for linear curves)
  )
  return ctx.builtin_is_module(options.module, options.pipeline, curve_options)


def call_programs(ptx: bytes) -> tuple[str, ...]:
  """The called programs programs `ptx` hold (`programs.LAYER`, `programs.LIGHT_VISIBILITY`), in
  their payload types' order."""
  return tuple(name for name, entry in _CALL_ENTRIES if entry.search(ptx) is not None)


def module_calls(ptx: bytes) -> tuple[str, ...]:
  """The called programs whose payload types a module of `ptx` declares (`compile_options`): its
  programs' (`call_programs`); a lens camera model's render program (`programs.build_lens`) the
  layer call's whatever the build (a ray-generation program names its payload types, and no other
  program shares the lens module; its light loops take their visibility inline)."""
  return (LAYER,) if _LENS_ENTRY.search(ptx) is not None else call_programs(ptx)


def record_options(record: prebuilt.ModuleRecord, *, calls: tuple[str, ...]) -> CompileOptions:
  """The compile options of a module record (`calls`: the called programs its PTX holds,
  `module_calls`)."""
  return compile_options(
    has_curves=record.has_curves,
    has_custom=record.has_custom,
    max_registers=record.max_registers,
    opacity_micromaps=record.opacity_micromaps,
    bound=tuple((offset, bytes.fromhex(value)) for offset, value in record.bound),
    calls=calls,
  )


@final
class PtxCompile:
  """The programs' PTX for `arch`: read from raywarp's PTX cache (`programs.ptx_key`) or the
  prebuilt bundle of the device's architecture `device_arch` (`prebuilt.ptx`; the programs are then
  never defined), else compiled on a background thread and stored in the cache (`compile_once`).
  `expected`: the bundle should hold the build (`tiers.prebuilt`): a miss warns.

  Attributes:
    module: the programs' Warp module (None: the PTX came from the cache).
  """

  def __init__(self, build: Callable[[], wp.Module], arch: int, key: str, *, device_arch: int, expected: bool) -> None:
    """Reads the cached or bundled PTX of `key` (`read_ptx`), else defines the programs (`build`)
    and starts compiling."""
    self._path = _ptx_path(key)
    self._bytes = read_ptx(key, device_arch=device_arch, expected=expected)
    self.module: wp.Module | None = None
    self._pool: concurrent.futures.ThreadPoolExecutor | None = None
    self._raised = False  # `result` raised the compile's failure
    if self._bytes is None:
      policy.refuse(f"the OptiX programs' PTX {key} (sm_{device_arch})")
      self.module = build()
      self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="rw_optix_ptx")
      self._future = self._pool.submit(self._compile, self.module, arch, self._path)

  @staticmethod
  def _compile(module: wp.Module, arch: int, path: Path | None) -> bytes:
    """NVRTC compile of the PTX for `arch` (releases the GIL), stored at `path` (atomically).

    No device: with an `arch` only, Warp compiles for that target alone (a `wp.ScopedDevice` here
    set the process-wide default device from this worker thread while the render thread
    allocated).
    """
    return compile_once(path, lambda: wp.compile_aot_module(module, arch=arch, use_ptx=True)[0].read_bytes())

  @property
  def ready(self) -> bytes | None:
    """The PTX when it was read (cache or bundle), None while it compiles."""
    return self._bytes if self._pool is None else None

  def result(self) -> bytes:
    """The PTX (waits; re-raises a compile error)."""
    if self._bytes is not None:
      return self._bytes
    if self._pool is None:
      raise RuntimeError("PtxCompile without a compile")
    try:
      self._bytes = self._future.result()
      return self._bytes
    except Exception:
      self._raised = True
      raise
    finally:
      self._pool.shutdown(wait=False)

  def join(self) -> None:
    """Waits for the compile to end (a release). Raises its failure only when `result` never did:
    a render that needed the programs already raised it (it is not raised again by the cleanup).

    Raises:
      Exception: the compile's failure, when no `result` call raised it.
    """
    if self._pool is None:
      return
    concurrent.futures.wait([self._future])
    self._pool.shutdown(wait=False)
    error = self._future.exception()
    if error is not None and not self._raised:
      self._raised = True
      raise error


# PTX files the cache keeps (the most recently used; every code edit makes a new key). Above what
# a prebuilt bundle's fill writes (`tools/prebuilt/fill.py`: every tier x output form x the fill's
# scenes, ~2-4 MB each): at 32 the fill of two scenes pruned its own first tiers' PTX, which the
# bundle then lacked (each compiled again on first use).
MAX_CACHED_PTX: Final = 256


def read_ptx(key: str, *, device_arch: int, expected: bool) -> bytes | None:
  """The PTX of `key` from raywarp's PTX cache or the prebuilt bundle of the device's
  architecture `device_arch` (None: neither holds it; `expected`: the bundle should, a miss warns).
  No compile."""
  return _read(_ptx_path(key)) or prebuilt.ptx(device_arch, key, expected=expected)


def _ptx_path(key: str) -> Path | None:
  """Where the PTX of `key` is cached (raywarp's cache directory's `ptx/`; None: caching off)."""
  root = cache_dir()
  return None if root is None else root / "ptx" / f"{key}.ptx"


def compile_once(path: Path | None, compile_ptx: Callable[[], bytes]) -> bytes:
  """The PTX of cache file `path`: compiled (`compile_ptx`) and stored by one process at a time
  (a lock beside it), so that a process starting the same build meanwhile waits and reads it
  (None: no cache, compiled)."""
  if path is None:
    return compile_ptx()
  with prebuilt.locked(path.with_suffix(".lock")):
    ptx = _read(path)
    if ptx is None:
      ptx = compile_ptx()
      _store(path, ptx)
    return ptx


def _read(path: Path | None) -> bytes | None:
  """The cached PTX at `path` (its use time refreshed), None when absent, unreadable or generated
  from other kernel code (its entry `<key>.json`: `prebuilt.Entry.fresh`); a truncated one (a PTX
  module ends with its last function's closing brace, then whitespace and Warp's NUL terminator)
  warns and is deleted (compiled again)."""
  if path is None:
    return None
  try:
    entry = prebuilt.Entry.of(json.loads(path.with_suffix(".json").read_text()))
    data = path.read_bytes() if entry.fresh() else None
    if data is None:
      return None
    os.utime(path)
  except (OSError, ValueError, KeyError, TypeError):
    return None
  if not data.rstrip(b"\0 \t\r\n").endswith(b"}"):
    warnings.warn(f"raywarp: the cached PTX {path} is truncated: compiling it again", RuntimeWarning, stacklevel=3)
    path.unlink(missing_ok=True)
    return None
  return data


def _store(path: Path, ptx: bytes) -> None:
  """Stores `ptx` at `path` with its entry (`<key>.json`: generated from every kernel-code function,
  `kernel_code.ALL`; each file atomically, the PTX first: the entry, written last, vouches for it.
  A key holds across kernel-code edits, so `path` may hold an older code's PTX: a process of this
  code reading meanwhile sees the old entry, stale for it, and waits for the compile's lock, never
  this code's entry beside the old PTX. The least recently used beyond `MAX_CACHED_PTX` are
  deleted). A cache that cannot be written warns: the build goes on with its PTX."""
  partial = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.partial")
  entry = prebuilt.Entry(ptx="", code=kernel_code.version(), reach=kernel_code.ALL, groups={})
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(ptx)
    partial.replace(path)
    partial.write_text(json.dumps(entry.json()))
    partial.replace(path.with_suffix(".json"))
    cached = sorted(path.parent.glob("*.ptx"), key=lambda p: p.stat().st_mtime)
    for old in cached[: max(0, len(cached) - MAX_CACHED_PTX)]:
      for suffix in (".ptx", ".json", ".lock"):
        old.with_suffix(suffix).unlink(missing_ok=True)
  except OSError as e:
    partial.unlink(missing_ok=True)
    warnings.warn(
      f"the OptiX PTX cache {path.parent} cannot be written ({e}): the programs compile again next start", stacklevel=2
    )


def forget(key: str) -> None:
  """Deletes the cached PTX of `key` (OptiX rejected it: the next build compiles it again)."""
  path = _ptx_path(key)
  if path is not None:
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)


def entry_names(ptx: bytes, required: list[str]) -> dict[str, str]:
  """OptiX program semantic name (e.g. '__raygen__render') -> PTX entry, for every `required` one.

  Raises:
    RuntimeError: when a program is not found (Warp mangles kernel names with a hash suffix).
  """
  names = {}
  for entry in re.findall(rb"\.entry\s+([A-Za-z_][A-Za-z0-9_]*)", ptx):
    entry = entry.decode()
    names[_ENTRY_HASH.sub("", entry)] = entry
  missing = [name for name in required if name not in names]
  if missing:
    raise RuntimeError(
      f"OptiX programs {missing} not found among the PTX entries {sorted(names.values())}: "
      "Warp's kernel name mangling may have changed (expected <name>_<8 hex digits>)"
    )
  return names


def _required(scene: SceneFlags, raygen: str, extras: tuple[str, ...], anyhit: frozenset[str]) -> list[str]:
  """The semantic names of a build's programs (`raygen`: its main ray-generation program)."""
  required = [raygen, "__miss__render", "__closesthit__triangle", "__closesthit__custom"]
  for shared, kind in HitGroups.of(scene).members():
    if kind == CURVES:
      required.append("__closesthit__curve" + ("_shared" if shared else ""))
    else:
      required += [intersection_name(kind, shared=shared, shadow=shadow) for shadow in (False, True)]
  return required + sorted(anyhit) + list(extras)


def _records(groups: list[OptixObject], device: wp.Device) -> wp.array[wp.uint8, [int]]:
  """The SBT records (headers only) of `groups`, on `device`."""
  header = api.SBT_RECORD_HEADER_SIZE
  out = np.zeros((len(groups), header), np.uint8)
  t = api.table()
  for g, row in zip(groups, out, strict=True):
    api.check(t.optixSbtRecordPackHeader(g.handle, row.ctypes.data), "optixSbtRecordPackHeader")
  return wp.array(out.reshape(-1), dtype=wp.uint8, device=device)


def _single(kind: int, module: OptixObject, entry: str) -> api.ProgramGroupDesc:
  """A raygen or miss program group description (`kind`) of `entry` in `module`."""
  desc = api.ProgramGroupDesc(kind=kind)
  one = desc.raygen if kind == api.PROGRAM_GROUP_KIND_RAYGEN else desc.miss
  one.module, one.entryFunctionName = module.handle, entry.encode()
  return desc


# (module handle, entry name) of a hit group's program (entry None: a built-in intersector).
type Program = tuple[int, str | None]


def _hitgroup(ch: Program | None, *, ah: Program | None = None, is_: Program | None = None) -> api.ProgramGroupDesc:
  """A hit group description: optional closest hit `ch`, any hit `ah` and intersection `is_`."""
  desc = api.ProgramGroupDesc(kind=api.PROGRAM_GROUP_KIND_HITGROUP)
  h = desc.hitgroup
  if ch is not None:
    h.moduleCH, h.entryFunctionNameCH = ch[0], (ch[1] or "").encode()
  if ah is not None:
    h.moduleAH, h.entryFunctionNameAH = ah[0], (ah[1] or "").encode()
  if is_ is not None:
    h.moduleIS = is_[0]
    h.entryFunctionNameIS = is_[1].encode() if is_[1] is not None else None
  return desc


@final
class Pipeline:
  """Compiled OptiX module, program groups, pipelines and shader binding tables of a build.

  Attributes:
    module: the OptiX module of the programs' PTX.
    record: its options (`module_record`).
    groups: every program group (destroyed with the pipeline).
    pipelines: ray-generation program (None: the render) -> its pipeline.
    sbts: ray-generation program (None: the render) -> its shader binding table.
    params: (ray-generation program (None: the render), launch chunks) -> one launch-parameter copy
      per chunk (`store`).
  """

  def __init__(
    self,
    *,
    ctx: DeviceContext,
    device: wp.Device,
    ptx: bytes,
    scene: SceneFlags,
    record: prebuilt.ModuleRecord,
    module: OptixObject,
    layout: params_lib.ParamsLayout,
    raygen: str = RENDER,
    lens: tuple[bytes, OptixObject] | None = None,
    extras: tuple[str, ...] = (),
    anyhit: frozenset[str] = frozenset(),
  ) -> None:
    """Creates the OptiX objects of a build's PTX on `device` (`ctx`'s) around `module`, its OptiX
    module of options `record` (`trace/optix/modules.create`; owned from here on, also when this
    raises): the program groups and SBT records of `scene`'s structures (`raygen`: its main
    ray-generation program, launched as program None; `lens`: (PTX, OptiX module) of a lens
    camera model's render program `raygen`, `programs.build_lens`, owned from here on too, which
    replaces the build's own; `extras`: its other ray-generation programs;
    `anyhit`: the hit groups' any-hit programs, by name, `programs.anyhit_programs`; the called
    programs the PTX has, `call_programs`, as the module's options).
    Never inside a capture: the SBT records are uploads."""
    self._destroyed = False
    self._owned: list[OptixObject] = [module] if lens is None else [module, lens[1]]
    self.device = device
    self.module, self.record = module, record
    try:
      self._build(
        ctx=ctx,
        ptx=ptx,
        scene=scene,
        record=record,
        layout=layout,
        raygen_name=raygen,
        lens=lens,
        extras=extras,
        anyhit=anyhit,
      )
    except BaseException as error:
      for cleanup in self._release():
        error.add_note(f"OptiX construction cleanup failed: {cleanup}")
      raise

  def _build(
    self,
    *,
    ctx: DeviceContext,
    ptx: bytes,
    scene: SceneFlags,
    record: prebuilt.ModuleRecord,
    layout: params_lib.ParamsLayout,
    raygen_name: str,
    lens: tuple[bytes, OptixObject] | None,
    extras: tuple[str, ...],
    anyhit: frozenset[str],
  ) -> None:
    """Acquires native objects in dependency order; the constructor rolls back on failure."""
    device = self.device
    calls = call_programs(ptx)
    options = record_options(record, calls=calls)
    own = RENDER if lens is not None else raygen_name
    entries = entry_names(ptx, _required(scene, own, extras, anyhit) + list(calls))
    raygen_module = self.module
    if lens is not None:
      entries[raygen_name] = entry_names(lens[0], [raygen_name])[raygen_name]
      raygen_module = lens[1]
    raygen, misses, hitgroups = self._groups(ctx, options, scene, entries, raygen=(raygen_module, raygen_name), anyhit=anyhit)
    # The calls' miss records and payload types, after one per ray type (`device.visibility_slot`).
    types = options.module.payloadTypes
    for k, name in enumerate(calls):
      kind = api.PROGRAM_GROUP_KIND_MISS
      misses.append(self._group(ctx, _single(kind, self.module, entries[name]), types[RAY_TYPES + k]))
    self._depth = _TRACE_DEPTH + len(calls)
    extra = {name: self._group(ctx, _single(api.PROGRAM_GROUP_KIND_RAYGEN, self.module, entries[name])) for name in extras}
    self.groups = [raygen, *misses, *hitgroups, *extra.values()]
    # The other ray-generation programs get pipelines of their own, so that they do not change the
    # render pipeline's stack sizes.
    pco = options.pipeline
    render = self._link(ctx, pco, [raygen, *misses, *hitgroups])
    self.pipelines = {None: render} | {name: self._link(ctx, pco, [g, *misses, *hitgroups]) for name, g in extra.items()}
    self.miss_record, self.hitgroup_records = _records(misses, device), _records(hitgroups, device)
    self.raygen_records = {name: _records([g], device) for name, g in [(None, raygen), *extra.items()]}
    header = api.SBT_RECORD_HEADER_SIZE
    self.sbts = {
      name: api.ShaderBindingTable(
        raygenRecord=address(raygen_record),
        missRecordBase=address(self.miss_record),
        missRecordStrideInBytes=header,
        missRecordCount=len(misses),
        hitgroupRecordBase=address(self.hitgroup_records),
        hitgroupRecordStrideInBytes=header,
        hitgroupRecordCount=len(hitgroups),
      )
      for name, raygen_record in self.raygen_records.items()
    }
    self.layout = layout
    self.params_size = params_lib.size_of(params_lib.params_type(layout))
    self.params: dict[tuple[str | None, int], wp.array[wp.uint32, [int, int]]] = {}
    # `params` key -> (parameter bytes, nmajor, minor) its buffer holds (None: unknown).
    self._stored: dict[tuple[str | None, int], tuple[bytes, int, int] | None] = {}
    # A capture recorded a copy into `params`: `store` never skips again (a replay rewrites the
    # buffer behind the host's back). Whether a graph may replay at all is the renderer's check
    # (`launch.OptixRenderer._check_graph`: the pipeline it captured is still alive).
    self._captured = False

  def _group(self, ctx: DeviceContext, desc: api.ProgramGroupDesc, payload: api.PayloadType | None = None) -> OptixObject:
    """The program group of `desc` (`payload`: its programs' payload type), owned by the
    pipeline."""
    group = ctx.program_group(desc, payload)
    self._owned.append(group)
    return group

  def _groups(
    self,
    ctx: DeviceContext,
    options: CompileOptions,
    scene: SceneFlags,
    entries: dict[str, str],
    *,
    raygen: tuple[OptixObject, str],
    anyhit: frozenset[str],
  ) -> tuple[OptixObject, list[OptixObject], list[OptixObject]]:
    """(raygen, miss per ray type, hit groups) of the main ray-generation program `raygen` (its
    module, its name).
    Hit groups in SBT order (`group.HitGroups`), each as `device.RAY_TYPES` records in a row, one
    per ray type (payload type 0: camera and occlusion traces, 1: shadow traces, no closest-hit
    program)."""
    module = self.module.handle
    types = options.module.payloadTypes
    raygen_group = self._group(ctx, _single(api.PROGRAM_GROUP_KIND_RAYGEN, raygen[0], entries[raygen[1]]))
    misses = [
      self._group(ctx, _single(api.PROGRAM_GROUP_KIND_MISS, self.module, entries["__miss__render"]), types[k])
      for k in range(device.RAY_TYPES)
    ]

    def ah(name: str) -> Program | None:
      """Any-hit program `name` when the build has it."""
      return (module, entries[name]) if name in anyhit else None

    def pair(ch: Program | None, *, base: str | None, is_: tuple[Program, Program] | None = None) -> list[OptixObject]:
      """The ray types' hit groups of closest hit `ch` (type 0 only), any hit `base` / `base`
      + "_shadow" (None: none), intersection `is_` (type 0, type 1)."""
      camera, shadow = (None, None) if base is None else (ah(base), ah(base + "_shadow"))
      return [
        self._group(ctx, _hitgroup(ch, ah=camera, is_=is_[0] if is_ else None), types[0]),
        self._group(ctx, _hitgroup(None, ah=shadow, is_=is_[1] if is_ else None), types[1]),
      ]

    hitgroups = pair((module, entries["__closesthit__triangle"]), base="__anyhit__triangle")
    curve_module = curves_module(ctx, options) if scene.has_world_curves or scene.has_shared_curves else 0
    for shared, kind in HitGroups.of(scene).members():
      suffix = "_shared" if shared else ""
      if kind == CURVES:
        builtin = (curve_module, None)
        closest = (module, entries["__closesthit__curve" + suffix])
        hitgroups += pair(closest, base="__anyhit__curve" + suffix, is_=(builtin, builtin))
      else:
        camera, shadow = (entries[intersection_name(kind, shared=shared, shadow=k)] for k in (False, True))
        hitgroups += pair((module, entries["__closesthit__custom"]), base=None, is_=((module, camera), (module, shadow)))
    return raygen_group, misses, hitgroups

  def _link(self, ctx: DeviceContext, pco: api.PipelineCompileOptions, groups: list[OptixObject]) -> OptixObject:
    """A pipeline of `groups` with the stack sizes their programs need."""
    pipeline = ctx.pipeline(pco, api.PipelineLinkOptions(maxTraceDepth=self._depth), groups)
    self._owned.append(pipeline)
    stack = api.accumulate_stack_sizes([g.handle for g in groups], pipeline.handle)
    dc_traversal, dc_state, continuation = api.compute_stack_sizes(stack, self._depth, 0, 0)
    set_stack_size(pipeline, dc_traversal, dc_state, continuation, _MAX_TRAVERSABLE_DEPTH)
    return pipeline

  @staticmethod
  def chunks(nmajor: int, minor: int) -> list[tuple[int, int]]:
    """(first major unit, launch indices) of each launch of `nmajor` x `minor` indices.

    A chunk holds whole major units (worlds, rays, lattice points: `minor` indices each), at most
    `MAX_LAUNCH` indices, so that every program's index arithmetic stays in 32 bits. No index
    (`nmajor` or `minor` 0), no chunk: OptiX would launch a zero-width grid.
    """
    if nmajor <= 0 or minor <= 0:
      return []
    per = max(1, MAX_LAUNCH // minor)
    return [(first, min(per, nmajor - first) * minor) for first in range(0, nmajor, per)]

  def store(self, params: LaunchParams, nmajor: int, minor: int, program: str | None = None) -> None:
    """Writes `program`'s launch parameters for `nmajor` x `minor` indices: one copy per chunk
    (`offset`), into the buffer of its chunk count (`params`: one per count, never reallocated, so
    that a captured copy and the launch `trace` makes at its replay read the same buffer).

    Skipped when the buffer already holds these values (`_stored`). A capture always records
    the copy, and afterwards the pipeline stops skipping: replaying the graph rewrites the buffer
    with the captured values behind the host's back.
    """
    chunks = self.chunks(nmajor, minor)
    if not chunks:
      return
    capturing = device_caps(self.device).capturing()
    self._captured = self._captured or capturing
    key = None if self._captured else (struct_bytes(params), nmajor, minor)
    slot = (program, len(chunks))
    buffer = self.params.get(slot)
    if key is not None and buffer is not None and self._stored.get(slot) == key:
      return
    words, nword, offset_word = params_lib.store_inputs(params)
    if buffer is None:
      # One row of the struct's words per chunk (row stride: the struct's size, `trace`).
      buffer = self.params[slot] = wp.zeros((len(chunks), nword), dtype=wp.uint32, device=self.device)
    per = chunks[1][0] if len(chunks) > 1 else 0
    wp.launch(
      params_lib.store_words,
      dim=(len(chunks), nword),
      inputs=[words, nword, offset_word, per],
      outputs=[buffer],
      device=self.device,
    )
    self._stored[slot] = key

  def trace(self, stream: wp.Stream, nmajor: int, minor: int, program: str | None = None) -> None:
    """Launches `program` (None: the render) over `nmajor` x `minor` indices on `stream`, in chunks
    (`store` first)."""
    chunks = self.chunks(nmajor, minor)
    if not chunks:
      return
    buffer = self.params.get((program, len(chunks)))
    if buffer is None or buffer.ptr is None:
      raise RuntimeError(f"OptiX launch of {program or 'the render'} before its launch parameters were stored")
    launch = api.table().optixLaunch
    pipeline, sbt, cuda_stream = self.pipelines[program].handle, ctypes.byref(self.sbts[program]), stream_handle(stream)
    for c, (_first, count) in enumerate(chunks):
      ptr = int(buffer.ptr) + c * self.params_size
      api.check(launch(pipeline, cuda_stream, ptr, self.params_size, sbt, count, 1, 1), "optixLaunch")

  @property
  def destroyed(self) -> bool:
    """Whether the OptiX objects were released (`destroy`, or a failed construction)."""
    return self._destroyed

  def _release(self) -> list[Exception]:
    """Detaches and releases all acquired handles in reverse dependency order."""
    self._destroyed = True
    owned, self._owned = self._owned, []
    errors: list[Exception] = []
    for obj in reversed(owned):
      try:
        obj.destroy()
      except Exception as error:  # noqa: BLE001  # Drain every handle before reporting failures.
        errors.append(error)
    return errors

  def destroy(self) -> None:
    """Destroys the OptiX pipelines, program groups and module (once; no launch may be pending)."""
    if self._destroyed:
      return
    for error in self._release():
      warnings.warn(f"destroying an OptiX object failed: {error}", stacklevel=2)
