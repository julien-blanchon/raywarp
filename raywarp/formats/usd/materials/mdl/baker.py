"""`SdkBaker`: the `context.MaterialBaker` of MDL shaders and MaterialX materials, through the
NVIDIA MDL SDK.

A MaterialX material (a stage's MaterialX surface network, or a referenced `.mtlx` document's
`surfacematerial`) is first written as an MDL module by the MaterialX library's generator
(`formats.usd.materials.materialx`; modules named by their content, under `<cache_dir>/mdl/mtlx/` or
a directory of the read), then converted as an MDL material whose arguments are its files. Per
material (`steps`): its module is loaded, instantiated with the authored inputs, compiled,
distilled to `transmissive_pbr` and mapped to the canonical material (`surface`), varying values
baked by the SDK (`bake`). Results are cached by (module, material, authored inputs), and on disk
(`memo`): a material read again from unchanged files skips the SDK. The first conversion of a
read prefetches the stage's bound MDL materials: compiles, distills and bakes on a thread pool,
one transaction per lane. Without a usable MDL SDK the read raises `sdk.MdlSdkError` with the
cause (no grey materials: such a scene once rendered dark with nothing failing). A material whose
own conversion fails is reported and given the error material (`ERROR`, magenta), which no
authored material looks like.
"""

from __future__ import annotations

import dataclasses
import gc
import os
import queue
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

from pxr import Sdf, Usd, UsdShade

from raywarp.core.pipeline.platform.device import process_cores
from raywarp.formats.usd.materials.context import MaterialXError, MeshNeeds, ShadeContext, Where
from raywarp.formats.usd.materials.mdl import load, memo, sdk, steps
from raywarp.formats.usd.materials.mdl.bake import Bakes, Request, prefetch
from raywarp.formats.usd.materials.mdl.dag import Dag
from raywarp.formats.usd.materials.mdl.surface import material, plan
from raywarp.formats.usd.materials.mdl.values import UnsupportedError
from raywarp.formats.usd.materials.mdl.writer import Writer
from raywarp.formats.usd.materials.select import is_materialx
from raywarp.scene.images import files
from raywarp.scene.model.material import Material
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import Json

if TYPE_CHECKING:
  from collections.abc import Callable
  from types import TracebackType

  from raywarp.formats.usd.materials.materialx import Network
  from raywarp.formats.usd.materials.mdl import sdk_types as st

# The material of an MDL / MaterialX material whose conversion fails (reported): an unmistakable
# magenta.
ERROR: Final = Material(base_color=(1.0, 0.0, 1.0), base_metalness=0.0, specular_roughness=0.5, name="<mdl error>")
# The most lanes (threads, each with its own transactions) of a prefetch: the SDK's bake_texture
# is itself multithreaded, more lanes oversubscribe the cores.
_MAX_LANES: Final = 8

# A compile's result: (compiled, distilled) or the error.
type Compiled = tuple[st.ICompiled_material, st.ICompiled_material] | str
# A prefetched material: its shader prim, shading and conversion.
type Loaded = tuple[Usd.Prim, steps.Shading, _Conversion]

# The failures of one material's conversion: reported, the material ERROR (a MaterialX document
# the library rejects, an MDL construct the evaluators cannot represent). SDK failures are values
# (`steps.compile_material`, `load.Session.load`); an unusable SDK (`sdk.MdlSdkError`) and any
# other exception (a programming error) propagate.
MATERIAL_ERRORS: Final = (MaterialXError, UnsupportedError)


@final
@dataclasses.dataclass(slots=True)
class _Conversion:
  """One results key's conversion in a read: its load reports, the disk cache entry (None: a
  miss), its plan and readers (released once mapped), its dependencies and result."""

  record: LoadReport
  entry: dict[str, Json] | None
  file: Path | None  # the disk cache entry's (None: no cache)
  read: bool = False  # whether step 4 ran (a prefetch that fails after step 2 leaves it to `_map`)
  plan: steps.Plan | None = None
  distilled: tuple[Dag | None, Dag] | None = None
  deps: set[str] = dataclasses.field(default_factory=set)
  result: tuple[Material, MeshNeeds] | None = None


@final
class SdkBaker:
  """MDL materials through the MDL SDK (see the module docstring); a context manager that
  releases the SDK state of the stage read on exit."""

  def __init__(self, cache_dir: Path | None = None) -> None:
    """`cache_dir`: the shared disk cache of baked textures, conversions and MaterialX modules
    (`<cache_dir>/mdl/`; None: off, the modules in a directory removed on exit)."""
    self.bakes = Bakes(cache_dir)
    self.temporary = cache_dir is None
    self.generated = Path(tempfile.mkdtemp(prefix="raywarp_mtlx_")) if cache_dir is None else cache_dir / "mdl" / "mtlx"
    self.session: load.Session | None = None
    self.conversions: dict[tuple[str, ...], _Conversion] = {}
    self.keys: dict[Sdf.Path, tuple[str, ...]] = {}  # each converted shader's results key
    self.shadings: dict[Sdf.Path, tuple[steps.Shading, LoadReport]] = {}  # the prefetch's reads
    self.sources: set[str] = set()
    self.stats = memo.Stats()  # the read's file stats (the disk cache's dependencies)
    self.prepared = False
    self.meters_per_unit = 1.0  # the read's stage (`state::meters_per_scene_unit`)

  def __enter__(self) -> SdkBaker:
    """The baker."""
    return self

  def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None) -> None:
    """Releases the stage read's SDK scopes (a failure doing so does not hide the read's own)."""
    session, self.session = self.session, None
    self.conversions.clear()  # their SDK handles before the session's scopes
    self.bakes.jobs.clear()
    self.bakes.made.clear()  # the texels of bakes no slot bound (a prefetched material that lost)
    if self.temporary:
      shutil.rmtree(self.generated, ignore_errors=True)
    if session is None:
      return
    with sdk.LOCK:
      try:
        session.close()
      except Exception:
        if error is None:
          raise

  def dependencies(self) -> list[str]:
    """The .mdl modules read and the texture files their materials read (a packaged one: its
    package on disk), except the files of raywarp's MDL library and of the bundled SDK: code, which
    the `.wpbr` key versions (`api.cache_key`: library digest, SDK version), at other paths on
    every machine."""
    found = {files.container(s) for s in self.sources}
    code = (str(load.LIBRARY), str(self.generated), *(sdk.api().system_paths if found else ()))
    roots = tuple({os.path.join(r, "") for c in code for r in (c, os.path.realpath(c))})
    return sorted(f for f in found if not f.startswith(roots))

  def convert(self, ctx: ShadeContext, shader: UsdShade.Shader) -> tuple[Material, MeshNeeds]:
    """Material of an MDL shader or a MaterialX surface shader, cached by (module, material,
    arguments); `ERROR`, reported, when its conversion trips. The first one prefetches the
    stage's (`prepare`).

    Raises:
      sdk.MdlSdkError: without a usable MDL SDK.
    """
    prim = shader.GetPrim()
    self._prepare_once(ctx, prim)
    path = (prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim).GetPath()  # the prefetch's
    return self._guarded_convert(ctx, prim, path, lambda: self._shading(ctx, shader))

  def _prepare_once(self, ctx: ShadeContext, prim: Usd.Prim) -> None:
    """The read's prefetch, on its first conversion (`prepare`; best effort)."""
    if self.prepared:
      return
    self.prepared = True
    try:
      self.prepare(ctx, prim.GetStage())
    except MATERIAL_ERRORS as e:  # the read converts (and reports) every material on demand
      ctx.report.drop(f"mdl.prefetch_error:{type(e).__name__}", prim)

  def _guarded_convert(
    self, ctx: ShadeContext, where: Where, path: Sdf.Path, shading: Callable[[], steps.Shading]
  ) -> tuple[Material, MeshNeeds]:
    """`_convert`, a failure reported at `where` and given `ERROR`."""
    try:
      return self._convert(ctx, where, path, shading)
    except MATERIAL_ERRORS as e:
      steps.error_material(ctx, where, unsupported_key(e))
    return ERROR, MeshNeeds()

  def _shading(self, ctx: ShadeContext, shader: UsdShade.Shader) -> steps.Shading:
    """The shading of an MDL shader, or of a MaterialX surface shader's generated module.

    Raises:
      MaterialXError: a MaterialX network the library cannot generate.
    """
    if is_materialx(shader):
      from raywarp.formats.usd.materials import materialx  # noqa: PLC0415  # MaterialX only for MaterialX materials

      return self._generated(ctx, materialx.from_shader(shader, ctx.report), shader.GetPrim())
    return steps.Shading.of(ctx, shader)

  def _generated(self, ctx: ShadeContext, network: Network, where: Where) -> steps.Shading:
    """The shading of a MaterialX network's generated module (written under `generated`).

    Raises:
      MaterialXError: a network the library cannot generate.
    """
    from raywarp.formats.usd.materials import materialx  # noqa: PLC0415  # MaterialX only for MaterialX materials

    with sdk.LOCK:  # one generator context
      out = materialx.generate(network, ctx.report, where)
    ctx.report.count("mtlx.generated")
    module = materialx.module_file(self.generated, out.source)
    return steps.Shading.generated(module, materialx.MATERIAL, list(out.inputs))

  def _convert(
    self, ctx: ShadeContext, where: Where, path: Sdf.Path, make: Callable[[], steps.Shading]
  ) -> tuple[Material, MeshNeeds]:
    """`convert` once prepared: material `path` (reports at `where`), its shading by `make` unless
    the prefetch read it."""
    known = self.conversions.get(self.keys[path]) if path in self.keys else None
    if known is not None and known.result is not None:  # a shader bound again (another instance's)
      ctx.report.count("mdl.cached")
      return known.result
    shading, record = self.shadings.pop(path, None) or (make(), None)
    ctx.report.merge(record)
    self.keys[path] = shading.key
    conv = self._conversion(ctx, shading)
    if conv.result is not None:
      ctx.report.count("mdl.cached")
      return conv.result
    conv.result = self._replay(ctx, where, conv) or self._map(ctx, where, shading, conv)
    return conv.result

  def _conversion(self, ctx: ShadeContext, shading: steps.Shading) -> _Conversion:
    """The conversion of a shading's results key (its disk cache entry read on first use: one
    made on the same module search paths)."""
    if self.bakes.directory is not None and not self.conversions:  # the read's first conversion
      memo.prune(self.bakes.directory)
    if shading.key not in self.conversions:
      d, paths = self.bakes.directory, self._search_paths(shading)
      file = None if d is None or paths is None else memo.path(d, ctx, shading.key, list(shading.roots))
      entry = None if file is None or paths is None else memo.read(file, paths, self.stats)
      self.conversions[shading.key] = _Conversion(LoadReport(describe=ctx.report.describe), entry, file)
    return self.conversions[shading.key]

  def _search_paths(self, shading: steps.Shading) -> list[str] | None:
    """The module search paths of a shading's module (None: no module it names)."""
    session = self._session(sdk.api())
    ref = session.module_ref(shading.resolved, shading.authored)
    if ref is None:
      return None
    for directory in shading.roots:
      session.add_root(directory)
    return list(session.paths(ref.root))

  def prepare(self, ctx: ShadeContext, stage: Usd.Stage) -> None:
    """Loads, compiles and distills the bound MDL materials of `stage` and makes their bakes on a
    thread pool (the reports kept for the read's conversions); best effort: a material it trips
    on is converted (and reported) by the read."""
    api = sdk.api()
    self.meters_per_unit = ctx.frame.meters_per_unit
    with sdk.LOCK:
      loaded = self._load_all(ctx, stage, api)
      if not loaded:
        return
      self._session(api).commit()  # the loads (modules, input textures) visible to every lane
      # No collection while the lanes run: a SWIG destructor of a handle of one lane's
      # transaction must not run on another lane's thread.
      enabled = gc.isenabled()
      gc.collect()
      gc.disable()
      try:
        self._lanes_run(ctx, loaded)
      finally:
        if enabled:
          gc.enable()

  def _lanes_run(self, ctx: ShadeContext, loaded: list[Loaded]) -> None:
    """Steps 3 (a shared queue, each worker in its own lane), 4 and the bakes of the loaded
    materials."""
    n = _lanes()
    work: queue.SimpleQueue[Loaded] = queue.SimpleQueue()
    for item in loaded:
      work.put(item)
    with ThreadPoolExecutor(n, thread_name_prefix="mdl") as pool:  # waits for its tasks on exit
      compiled = [
        (item, result, lane) for lane, done in enumerate(pool.map(self._worker, [work] * n, range(n))) for item, result in done
      ]
      for (prim, shading, conv), result, lane in compiled:
        self._readers(ctx, prim, shading, conv, result, lane)
      del compiled  # the compiled materials: only the readers keep them
      prefetch(ctx, self._plans(ctx, loaded), self.bakes, pool)

  def _load_all(self, ctx: ShadeContext, stage: Usd.Stage, api: sdk.Api) -> list[Loaded]:
    """Step 2 of every bound MDL shader and MaterialX surface that no conversion or disk cache
    entry has (their reads kept for the read's conversions)."""
    out: list[Loaded] = []
    shaders = steps.bound_shaders(ctx, stage)
    for shader in shaders:  # every root first: each module's search paths are then the read's
      for directory in steps.layer_dirs(shader.GetPrim()):
        self._session(api).add_root(directory)
    for shader in shaders:
      prim = shader.GetPrim()
      try:
        record = LoadReport(describe=ctx.report.describe)
        shading = self._shading(dataclasses.replace(ctx, report=record), shader)
        self.shadings[prim.GetPath()] = shading, record
        conv = self._conversion(ctx, shading)
        if conv.entry is None and conv.plan is None and self._load(ctx, prim, shading, conv, api):
          out.append((prim, shading, conv))
      except MATERIAL_ERRORS:  # the read converts (and reports) it
        continue
    return out

  def _plans(self, ctx: ShadeContext, loaded: list[Loaded]) -> list[tuple[Dag, list[Request]]]:
    """The bakes the mappings of the prefetched materials ask for (a pass that binds and
    reports nothing)."""
    scratch = dataclasses.replace(ctx, report=LoadReport())
    out: list[tuple[Dag, list[Request]]] = []
    for prim, _, conv in loaded:
      if conv.distilled is not None:
        dag = conv.distilled[1]
        out.append((dag, _guarded(plan, Writer(scratch, dag, prim, self.bakes)) or []))
    return out

  def _session(self, api: sdk.Api) -> load.Session:
    """The read's SDK session (made on first use)."""
    self.session = self.session or load.Session(api)
    return self.session

  def _load(self, ctx: ShadeContext, prim: Where, shading: steps.Shading, conv: _Conversion, api: sdk.Api) -> bool:
    """Step 2 into `conv` (a failure reported in its record); whether it can compile."""
    rec = dataclasses.replace(ctx, report=conv.record)
    try:
      conv.plan = steps.load_material(rec, self._session(api), prim, shading)
    except MATERIAL_ERRORS as e:  # reported, the material ERROR (counted by the corpus metric)
      steps.error_material(rec, prim, unsupported_key(e))
      conv.plan = steps.Plan(ref=None, definition="", deps=frozenset())
    self.sources |= conv.plan.deps
    conv.deps |= conv.plan.deps
    return conv.plan.ref is not None

  def _worker(self, work: queue.SimpleQueue[Loaded], lane: int) -> list[tuple[Loaded, Compiled]]:
    """Step 3 of the queued materials this worker takes, in lane `lane` (a thread of the
    prefetch; never raises)."""
    out: list[tuple[Loaded, Compiled]] = []
    while True:
      try:
        item = work.get_nowait()
      except queue.Empty:
        return out
      out.append((item, self._compiled(item[1], item[2], lane)))

  def _compiled(self, shading: steps.Shading, conv: _Conversion, lane: int) -> Compiled:
    """Step 3 of one material, its error as a value."""
    if self.session is None or conv.plan is None:
      return "no session"
    try:
      return steps.compile_material(self.session, conv.plan, shading.inputs, self.meters_per_unit, lane)
    except MATERIAL_ERRORS as e:  # one material's failure is reported, its material ERROR
      return f"{type(e).__name__}: {e}"

  def _readers(  # noqa: PLR0913  # a conversion's state, then the compile's result and lane
    self, ctx: ShadeContext, prim: Where, shading: steps.Shading, conv: _Conversion, result: Compiled, lane: int
  ) -> None:
    """Step 4 into `conv` (a failure reported in its record)."""
    rec = dataclasses.replace(ctx, report=conv.record)
    conv.read = True
    out = None
    try:
      if self.session is not None and conv.plan is not None:
        out = steps.readers(rec, self.session, prim, shading, conv.plan, result, lane)
    except MATERIAL_ERRORS as e:  # reported, the material ERROR (counted by the corpus metric)
      steps.error_material(rec, prim, unsupported_key(e))
    if out is not None:
      conv.distilled, deps = out
      conv.deps |= deps
      self.sources |= deps

  def _map(self, ctx: ShadeContext, prim: Where, shading: steps.Shading, conv: _Conversion) -> tuple[Material, MeshNeeds]:
    """The conversion of a material no disk cache entry has: steps 2-4 here unless the prefetch
    made them, then the mapping (written to the disk cache); its reports added to the read's."""
    with sdk.LOCK:
      api = sdk.api()
      self.meters_per_unit = ctx.frame.meters_per_unit
      loads = conv.plan is None and self._load(ctx, prim, shading, conv, api)
      if loads or (conv.plan is not None and conv.plan.ref is not None and not conv.read):
        self._readers(ctx, prim, shading, conv, self._compiled(shading, conv, 0), 0)  # (or a failed prefetch's)
      ctx.report.merge(conv.record)
      if conv.distilled is None:  # a failed step, reported in its record
        return ERROR, MeshNeeds()
      (original, dag), conv.distilled = conv.distilled, None  # released once mapped
      record = LoadReport(describe=ctx.report.describe)
      mapping = dataclasses.replace(ctx, report=record)
      writer = Writer(mapping, dag, prim, self.bakes)
      mat = material(writer, sheen=None if original is None else Writer(mapping, original, prim, self.bakes))
      record.count("mdl.distilled")
      ctx.report.merge(record)
      self.bakes.forget(dag)
      if conv.file is not None and conv.plan is not None and conv.plan.ref is not None:  # the disk cache entry
        record.merge(conv.record)
        memo.save(conv.file, (mat, writer.needs()), writer.rec, record, conv.deps, list(conv.plan.paths), self.stats)
      return mat, writer.needs()

  def _replay(self, ctx: ShadeContext, prim: Where, conv: _Conversion) -> tuple[Material, MeshNeeds] | None:
    """The conversion of a disk cache entry, its reports and dependencies added (None: a miss; an
    entry that does not replay is removed)."""
    data, conv.entry = conv.entry, None
    result = None if data is None else memo.replay(ctx, prim, data, self.bakes.images)
    if data is None or result is None:
      if data is not None and conv.file is not None:
        conv.file.unlink(missing_ok=True)
      return None
    deps = data.get("deps")
    self.sources |= {str(d[0]) for d in deps if isinstance(d, list)} if isinstance(deps, list) else set()
    ctx.report.count("mdl.disk_cached")
    return result


def _lanes() -> int:
  """The lanes of a prefetch: the cores this process may run on, at most `_MAX_LANES`."""
  return min(_MAX_LANES, process_cores())


def unsupported_key(error: MaterialXError | UnsupportedError) -> str:
  """The report key of a material that failed with `error` (`MATERIAL_ERRORS`): a MaterialX
  error's own key (`mtlx.unknown_node:<id>`), an MDL construct's message cut to
  `steps.KEY_CHARS`."""
  if isinstance(error, MaterialXError):
    return f"mtlx.{error}"
  return f"mdl.unsupported:{str(error)[: steps.KEY_CHARS]}"


def _guarded[T, R](step: Callable[[T], R], arg: T) -> R | None:
  """`step(arg)`, None when it raises (the read's conversion makes and reports it again)."""
  try:
    return step(arg)
  except MATERIAL_ERRORS:  # the read's conversion makes and reports it again
    return None
