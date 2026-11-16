"""`rw.formats.usd.scene` / `load` / `bake` / `read`: stage -> GpuScene / Scene, with the `.rayw`
cache.

Cache: `<cache dir>/<stem>-<key>.rayw` (or an explicit `.rayw` path), freshness through
`raywarp.scene.model.rayw`: the key (source file name, options, the converter version
`version.CONVERTER`, the versions of `LIBRARIES`) and every file the scene came from (USD layers,
.mdl modules, images; missing ones included; recorded relative to the source's directory) are
compared on load (no pxr import). raywarp's own MDL library and the SDK wheel's modules are
code, versioned by the key, not dependencies. Baked MDL textures are shared by every scene in
`<cache dir>/mdl/`. A rebuild first reads the files the stale cache file lists on a thread pool
(`raywarp.scene.images.prefetch`), while the stage opens.
"""

from __future__ import annotations

import dataclasses
import enum
import os
import time
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Final

from raywarp.cache import paths as cache_paths
from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
from raywarp.core.pipeline.upload.build import upload
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.formats.usd import version
from raywarp.formats.usd.options import DEFAULT_OPTIONS, MDL_PATH_VARIABLES, Options
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.images import blob
from raywarp.scene.images.prefetch import Prefetch
from raywarp.scene.lighting.rig import Lighting
from raywarp.scene.model import rayw
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import Json

if TYPE_CHECKING:
  import warp as wp
  from pxr import Usd  # pxr on the first stage read only

# Third-party distributions whose code decides a read's values (stage composition, image and EXR
# decoding, array arithmetic, MaterialX MDL generation, MDL distillation and bakes): their versions
# are part of the key.
LIBRARIES: Final = ("usd-core", "Pillow", "OpenEXR", "numpy", "mdl-sdk", "materialx")

# A stage: a file path, or an open Usd.Stage (never cached).
type StageSource = str | os.PathLike[str] | Usd.Stage


def _json(adapter: Options, options: SceneOptions) -> dict[str, Json]:
  """The host-scene options as JSON values (tuples as lists): the USD `Options` and the texture size
  cap (the other `SceneOptions` act at upload, after the cache)."""
  opts: dict[str, Json] = {"max_texture_size": options.max_texture_size}
  for k, v in dataclasses.asdict(adapter).items():
    opts[k] = list(v) if isinstance(v, tuple) else v.value if isinstance(v, enum.Enum) else v
  return opts


def cache_key(
  source: str | os.PathLike[str],
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  mips: bool = False,
) -> rayw.CacheKey:
  """The cache key of `load(source, options=options, adapter=adapter, mips=mips)`."""
  # A read's MDL materials depend on the MDL search paths, raywarp's MDL library (data: by content)
  # and the bundled MDL SDK release (`LIBRARIES`' "mdl-sdk": the build hook's).
  from raywarp.formats.usd.materials.mdl import sdk as mdl_sdk  # noqa: PLC0415  # its build record, on use

  mdl: list[Json] = [*(os.environ.get(v, "") for v in MDL_PATH_VARIABLES), version.mdl_library_digest()]
  return rayw.CacheKey(
    adapter="usd",
    source=os.path.abspath(os.fspath(source)),
    options={**_json(adapter, options), "mips": mips, "mdl": mdl},
    code=f"converter-{version.CONVERTER}",
    libraries={**rayw.installed_versions(*LIBRARIES), "mdl-sdk": mdl_sdk.version()},
  )


def cache_path(
  source: str | os.PathLike[str],
  cache: str | os.PathLike[str] | bool = True,  # noqa: FBT001, FBT002  # a location or True (the default one)
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  mips: bool = False,
) -> Path:
  """The `.rayw` file `load(source, cache=cache, ...)` reads or writes."""
  return rayw.cache_file(cache_key(source, options=options, adapter=adapter, mips=mips), cache)


def read(
  stage: StageSource,
  builder: SceneBuilder | None = None,
  /,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options | None = None,
  mdl_cache: Path | None = None,
) -> SceneBuilder:
  """Reads a stage (path or Usd.Stage) into a SceneBuilder without building it.

  `options.max_texture_size` caps the decoded images; `adapter`: the USD options (None: the
  defaults, with a supplied `builder`'s displacement preparation policy); `mdl_cache`: directory
  of the shared baked-MDL disk cache (`<mdl_cache>/mdl/`; None: off). A supplied `builder` keeps
  its displacement preparation policy: a given `adapter` must ask for the same.

  Raises:
    ValueError: for displacement options that conflict with the supplied builder's.
    FileNotFoundError: for a stage file that does not open.
  """
  from pxr import Usd  # noqa: PLC0415  # pxr on the first stage read only

  from raywarp.formats.usd import util  # noqa: PLC0415  # the readers import pxr
  from raywarp.formats.usd.materials.mdl import SdkBaker  # noqa: PLC0415
  from raywarp.formats.usd.stage.reader import StageReader, builder_options, displacement_options  # noqa: PLC0415

  # A supplied builder owns one preparation policy for all imported stages: a conflicting one
  # fails before opening a stage or mutating that builder.
  if adapter is None:
    adapter = DEFAULT_OPTIONS if builder is None else builder_options(builder)
  if builder is not None and builder.displacement_options != displacement_options(adapter):
    raise ValueError("usd.read: Options.displacement* conflicts with the supplied SceneBuilder preparation policy")
  builder = SceneBuilder(displacement_options=displacement_options(adapter)) if builder is None else builder
  report = builder.report
  report.describe = util.describe  # prims named by their paths
  if not isinstance(stage, Usd.Stage):
    report.source = os.fspath(stage)
    with report.timer("usd.open"):
      stage = util.open_stage(stage)
  else:
    report.source = report.source or stage.GetRootLayer().identifier
  with SdkBaker(mdl_cache) as baker, util.asset_cache() as assets:
    reader = StageReader(stage, adapter, baker, builder, max_texture_size=options.max_texture_size)
    reader.read()
  deps = {layer.realPath for layer in stage.GetUsedLayers() if layer.realPath}
  deps |= set(baker.dependencies())
  builder.dependencies |= deps | assets.missing
  builder.meta.update(
    source=report.source,
    adapter="usd",
    options=_json(adapter, options),
    meters_per_unit=reader.mpu,
    up_axis="Z" if reader.up_z else "Y",
  )
  return builder


def load(  # noqa: PLR0913  # the source and keyword-only options
  source: StageSource,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
  cache: str | os.PathLike[str] | bool | None = True,  # a location, True (the cache directory) or None
  mips: bool = False,
  threads: int | None = None,
  verify: rayw.Verify = "files",
) -> Scene:
  """Loads a USD stage as a host Scene, through the `.rayw` cache when `cache` is given.

  `lighting` is added after the cache (`Scene.with_lighting`): it does not change the cache key.

  Args:
    source: stage path (or a Usd.Stage, never cached).
    options: the options every adapter shares (`rw.SceneOptions`; its texture size cap applies
      at decode, the rest at upload).
    adapter: the USD options (`raywarp.formats.usd.Options`).
    lighting: lights, ambient, environment maps and headlight added to the stage's.
    cache: cache directory, a `.rayw` file path, True (the default: the cache directory,
      ~/.cache/raywarp or $RAYWARP_CACHE_DIR; no cache when that variable is empty) or None (no
      cache). A second load of an unchanged source maps its `.rayw` (0.4-0.6 s for an Isaac prop
      that converts in 3-9 s). The directory also holds the baked MDL textures shared by every
      scene (`mdl/`).
    mips: store full mip chains in the scene.
    threads: image decode threads.
    verify: what a cached scene's freshness compares besides its key: "files" every file it was
      read from, "source" the stage file only (`scene.model.rayw.fresh`: for prepared caches whose
      files were checked otherwise, e.g. on a network mount where each file costs a round trip).

  Raises:
    TypeError: for `options` that is not a `rw.SceneOptions` or `adapter` not a
      `raywarp.formats.usd.Options` (both are "Options": easily swapped).
    ValueError: for `threads` below 1 or an unknown `verify` (checked before any read).
  """
  check_arguments(options, adapter, threads=threads, verify=verify)
  start = time.perf_counter()
  cache = cache_paths.location(cache)
  if cache is None or not isinstance(source, str | os.PathLike):  # no cache, or an open stage
    host = read(source, options=options, adapter=adapter).build(mips=mips, threads=threads)
    host.report.timings["total"] = time.perf_counter() - start
  else:
    key = cache_key(source, options=options, adapter=adapter, mips=mips)
    path = rayw.cache_file(key, cache)
    cache_paths.remove_stale(path)  # temporary files of crashed writers
    found = Scene.load_fresh(path, key, verify=verify)
    if found is not None:
      found.report.timings["warm_load"] = time.perf_counter() - start
      host = found
    else:
      host = _rebuild(source, path, key, mips=mips, threads=threads, mdl_cache=path.parent, options=options, adapter=adapter)
      host.report.timings["total"] = time.perf_counter() - start
  return host if lighting is None else host.with_lighting(lighting)


def check_arguments(options: SceneOptions, adapter: Options, *, threads: int | None, verify: rayw.Verify) -> None:
  """Checks a read's options before any work (a cache hit would otherwise never check `threads`;
  `options` and `adapter` are both "Options" and easily swapped).

  Raises:
    TypeError: for `options` that is not a `rw.SceneOptions` or `adapter` not a
      `raywarp.formats.usd.Options`.
    ValueError: for `threads` below 1 or an unknown `verify`.
  """
  if not isinstance(options, SceneOptions):
    raise TypeError(f"options must be a rw.SceneOptions, got {type(options).__module__}.{type(options).__name__}")
  if not isinstance(adapter, Options):
    raise TypeError(f"adapter must be a raywarp.formats.usd.Options, got {type(adapter).__module__}.{type(adapter).__name__}")
  blob.worker_count(threads)
  if verify not in typing.get_args(rayw.Verify.__value__):
    raise ValueError(f"verify must be one of {list(typing.get_args(rayw.Verify.__value__))}, got {verify!r}")


def scene(  # noqa: PLR0913  # the source and keyword-only options
  source: StageSource,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
  device: wp.DeviceLike = None,
  cache: str | os.PathLike[str] | bool | None = True,  # a location, True (the cache directory) or None
  mips: bool = False,
  threads: int | None = None,
  verify: rayw.Verify = "files",
) -> GpuScene[None]:
  """A USD stage as a GpuScene on `device`: `load` (see its arguments), then `rw.upload` with
  `options` (the stage's poses are static)."""
  host = load(
    source, options=options, adapter=adapter, lighting=lighting, cache=cache, mips=mips, threads=threads, verify=verify
  )
  return upload(host, device, options=options)


def _rebuild(  # noqa: PLR0913  # the build's inputs, keyword-only
  source: str | os.PathLike[str],
  path: Path,
  key: rayw.CacheKey,
  *,
  mips: bool,
  threads: int | None,
  mdl_cache: Path | None,
  options: SceneOptions,
  adapter: Options,
) -> Scene:
  """Builds stage file `source` into `.rayw` file `path` (cache `key`), reading the layers and
  modules the stale `path` lists ahead (`Prefetch`: what the stage open and the MDL conversion
  read one after the other; images are read by the builder as they are found)."""
  prefetch = Prefetch(d for d in rayw.stale_dependencies(path, key.root) if _composed(d.path))
  started = time.time_ns()
  try:
    builder = read(source, options=options, adapter=adapter, mdl_cache=mdl_cache)
    known = prefetch.done() | builder.read_states()
    rayw.record(builder.meta, key, builder.dependencies, known=known, since_ns=started)
  finally:
    prefetch.close()
  return builder.build(path, mips=mips, threads=threads)


def _composed(path: str) -> bool:
  """Whether file `path` is a USD layer (of any file format plugin) or an MDL module."""
  from pxr import Sdf  # noqa: PLC0415  # pxr on the first stage read only

  ext = Path(path).suffix[1:]
  return ext == "mdl" or (bool(ext) and Sdf.FileFormat.FindByExtension(ext) is not None)


def bake(  # noqa: PLR0913  # the source, the destination and keyword-only options
  source: StageSource,
  destination: str | os.PathLike[str],
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  mips: bool = False,
  threads: int | None = None,
  mdl_cache: Path | None = None,
) -> Scene:
  """Converts a stage to a `.rayw` file (the cache format, usable as `load(...,
  cache=destination)`).

  Returns the memory-mapped Scene.

  Raises:
    TypeError: for `options` / `adapter` of the other class (`load`).
    ValueError: for `threads` below 1.
  """
  check_arguments(options, adapter, threads=threads, verify="files")
  if isinstance(source, str | os.PathLike):  # a stage file (an open stage is never cached)
    key = cache_key(source, options=options, adapter=adapter, mips=mips)
    dest = Path(destination)
    return _rebuild(source, dest, key, mips=mips, threads=threads, mdl_cache=mdl_cache, options=options, adapter=adapter)
  return read(source, options=options, adapter=adapter, mdl_cache=mdl_cache).build(destination, mips=mips, threads=threads)
