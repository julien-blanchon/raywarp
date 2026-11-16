"""The steps of an MDL material's conversion, apart: a prefetch runs the costly one in parallel.

1. `Shading.of`: an MDL shader's source and authored inputs (USD reads; its results key), or
   `Shading.generated`: a MaterialX material's generated module and arguments
   (`formats.usd.materials.materialx`);
2. `load_material` (the read's thread, lane 0): the module loaded, the material's definition
   found, the input textures made, reports; the `Plan` holds names only (no SDK handle survives
   the commit that makes them visible to the other lanes);
3. `compile_material` (any thread, in its own lane's transaction): instantiate, instance-compile,
   distill;
4. `readers` (the read's thread): the two `Dag` readers, the lossy lobes and the dependencies.

`bound_shaders` finds the shaders a read will hand the MDL baker: the bound materials' surfaces
that resolve to it (MDL shaders and MaterialX surfaces). Reports name `Where`: the shader prim, or
the stage path of a referenced `.mtlx` document's material.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

from pxr import Sdf, Usd, UsdShade

from raywarp.formats.usd import util
from raywarp.formats.usd.materials.mdl import keys, load, sdk
from raywarp.formats.usd.materials.mdl.dag import Dag
from raywarp.formats.usd.materials.mdl.surface import SHEENS, lossy
from raywarp.formats.usd.materials.select import baked_surface

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.context import ShadeContext, Where
  from raywarp.formats.usd.materials.mdl import sdk_types as st

SOURCE_ASSET: Final = "info:mdl:sourceAsset"
# The longest error message an MDL report key holds.
KEY_CHARS: Final = 60


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Shading:
  """An MDL material's source (authored and resolved module, material name), its arguments, its
  search roots (the directories of the layers authoring it) and its results key."""

  authored: str
  resolved: str
  name: str
  inputs: list[load.Input]
  key: tuple[str, ...]
  roots: tuple[str, ...] = ()

  @staticmethod
  def of(ctx: ShadeContext, shader: UsdShade.Shader) -> Shading:
    """The shading of `shader` (its inputs read: connections reported)."""
    prim = shader.GetPrim()
    asset = util.get(prim.GetAttribute(SOURCE_ASSET)) if prim.HasAttribute(SOURCE_ASSET) else shader.GetSourceAsset("mdl")
    authored = asset.path if isinstance(asset, Sdf.AssetPath) else ""
    name = shader.GetSourceAssetSubIdentifier("mdl") or Path(authored).stem
    inputs = _inputs(ctx, shader)
    resolved = util.asset_path(asset)
    roots = tuple(layer_dirs(prim))
    key = _key(resolved or authored, name, inputs)
    return Shading(authored=authored, resolved=resolved, name=name, inputs=inputs, key=key, roots=roots)

  @staticmethod
  def generated(module: Path, name: str, inputs: list[load.Input]) -> Shading:
    """The shading of material `name` of generated module file `module` with arguments
    `inputs`."""
    file = str(module)
    return Shading(authored=file, resolved=file, name=name, inputs=inputs, key=_key(file, name, inputs))


def _key(source: str, name: str, inputs: list[load.Input]) -> tuple[str, ...]:
  """The results key of a material: its module, name and arguments."""
  return (source, name, *sorted(f"{i.name}={i.value!r}@{i.space}" for i in inputs))


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Plan:
  """A material loaded (`ref` and `definition`, its DB name), or not (None, ''), and the files
  its conversion depends on."""

  ref: load.ModuleRef | None
  definition: str
  deps: frozenset[str]
  paths: tuple[str, ...] = ()  # the module's search paths at its load (its disk cache entry's)


def bound_shaders(ctx: ShadeContext, stage: Usd.Stage) -> list[UsdShade.Shader]:
  """The surface shaders the read hands the MDL baker: of each material the stage reader's
  scan found bound (`ShadeContext.bound`), the one `select.baked_surface` picks (collection
  bindings are not followed: their materials convert on demand)."""
  out: dict[Sdf.Path, UsdShade.Shader] = {}
  for target in ctx.bound:
    material = UsdShade.Material(stage.GetPrimAtPath(target))
    shader = baked_surface(ctx, material) if material else None
    if shader is not None:
      out.setdefault(shader.GetPrim().GetPath(), shader)
  return list(out.values())


def load_material(ctx: ShadeContext, session: load.Session, prim: Where, shading: Shading) -> Plan:
  """Loads a material's module and finds its material (step 2; reports in `ctx.report`, the
  error material's of a failed step included)."""
  authored, resolved, name = shading.authored, shading.resolved, shading.name
  for directory in shading.roots:
    session.add_root(directory)
  if "://" in authored and not resolved:  # not fetched: an https module is not loaded from the network
    scheme, _, rest = authored.partition("://")
    return _failed(ctx, prim, f"mdl.url_module:{scheme}://{rest.split('/', 1)[0]}(not fetched)")
  ref = session.module_ref(resolved, authored)
  if ref is None:
    return _failed(ctx, prim, f"mdl.unresolved_module:{authored or '?'}")
  module, err = session.load(ref)
  if module is None:  # the read depends on the files the module would be found at
    deps = {ref.file} if ref.file else set(session.candidates(ref))
    return _failed(ctx, prim, f"mdl.load_failed:{name}({err[:KEY_CHARS]})", deps=frozenset(deps))
  deps = session.files(ref.root, module) | set(session.candidates(ref, module.get_filename()))
  if err:
    ctx.report.drop(f"mdl.{err.replace(' ', '_')}", prim)
    deps |= load.resources(module.get_filename())
  _report_origin(ctx, prim, ref, module)
  _report_shadowed(ctx, prim, deps, session.api)
  definition = load.definition_name(module, name)
  fn = None if definition is None else load.definition(session, ref, definition)
  if definition is None or fn is None:
    return _failed(ctx, prim, f"mdl.unknown_material:{name}", deps=frozenset(deps))
  given = load.arguments(session, ref, fn, shading.inputs)  # makes the input textures
  for input_name in given.skipped:
    ctx.report.drop(f"mdl.input:{input_name}", prim)
  for parameter in given.zeroed:
    ctx.report.approx(f"mdl.parameter_without_default:{parameter}(default-constructed)", prim)
  return Plan(ref=ref, definition=definition, deps=frozenset(deps), paths=session.paths(ref.root))


def compile_material(
  session: load.Session, plan: Plan, inputs: list[load.Input], meters_per_unit: float, lane: int
) -> tuple[st.ICompiled_material, st.ICompiled_material] | str:
  """(the material instance-compiled, and distilled), or the error (step 3: SDK calls in lane
  `lane`'s transaction only; any thread)."""
  fn = None if plan.ref is None else load.definition(session, plan.ref, plan.definition, lane)
  if plan.ref is None or fn is None:
    return "definition not found"
  compiled = load.compile_material(session, fn, load.arguments(session, plan.ref, fn, inputs, lane).args, meters_per_unit)
  if isinstance(compiled, str):
    return compiled
  distilled = load.distill(session.api, compiled)
  return distilled if isinstance(distilled, str) else (compiled, distilled)


def readers(  # noqa: PLR0913  # a conversion's state, then the compile's result and lane
  ctx: ShadeContext,
  session: load.Session,
  prim: Where,
  shading: Shading,
  plan: Plan,
  result: tuple[st.ICompiled_material, st.ICompiled_material] | str,
  lane: int,
) -> tuple[tuple[Dag | None, Dag], set[str]] | None:
  """((the compiled material's reader when it has a sheen, the distilled one's), the
  conversion's dependencies) (step 4); None, the error material reported, for a failed compile."""
  if isinstance(result, str) or plan.ref is None:
    error_material(ctx, prim, f"mdl.compile_failed:{shading.name}({str(result)[:KEY_CHARS]})")
    return None
  tr = session.transaction(plan.ref.root, lane)
  original = Dag(session.api, tr, result[0], session.textures, lane=lane, modules=plan.deps)
  dag = Dag(session.api, tr, result[1], session.textures, lane=lane, modules=plan.deps)
  found = lossy(original)
  for node in sorted(found):
    ctx.report.approx(f"mdl.distilled:{node}", prim)
  deps = set(plan.deps) | keys.files(original)  # the distilled material reads a subset of its textures
  return (original if SHEENS & found.keys() else None, dag), deps


def error_material(ctx: ShadeContext, prim: Where, key: str) -> None:
  """Reports the error material (`baker.ERROR`, magenta) given for `prim` as `key`."""
  ctx.report.drop(f"{key}(error)", prim)


def _failed(ctx: ShadeContext, prim: Where, key: str, *, deps: frozenset[str] = frozenset()) -> Plan:
  """The plan of a material that does not load (the error material, reported as `key`)."""
  error_material(ctx, prim, key)
  return Plan(ref=None, definition="", deps=deps)


def layer_dirs(prim: Usd.Prim) -> list[str]:
  """Directories of the layers authoring the shader's source asset (its search roots)."""
  attr = prim.GetAttribute(SOURCE_ASSET)
  stack = attr.GetPropertyStack(Usd.TimeCode.Default()) if attr else []
  return [os.path.dirname(s.layer.realPath) for s in stack if s.layer.realPath]  # noqa: PTH120  # string paths


def _report_origin(ctx: ShadeContext, prim: Where, ref: load.ModuleRef, module: st.IModule) -> None:
  """Reports a module found in raywarp's library (the stage does not ship it) or in another
  file than the one the stage names (a module of the same name found first)."""
  filename = module.get_filename()
  if filename and load.LIBRARY in Path(filename).parents:
    ctx.report.approx(f"mdl.library_module:{Path(filename).name}(raywarp's copy)", prim)
  elif ref.file and filename and os.path.realpath(filename) != os.path.realpath(ref.file):
    ctx.report.approx(f"mdl.module_shadowed:{ref.name}", prim)


def _report_shadowed(ctx: ShadeContext, prim: Where, files: set[str], api: sdk.Api) -> None:
  """Reports `::nvidia::` modules loaded from outside the SDK's and raywarp's libraries (a stage
  that ships its own copy shadows the distiller's support modules)."""
  trusted = (str(load.LIBRARY), *api.system_paths)
  for f in sorted(files):
    if Path(f).parent.name == "nvidia" and f.endswith(".mdl") and not f.startswith(trusted):
      ctx.report.approx(f"mdl.nvidia_module_shadowed:{Path(f).name}", prim)


def _inputs(ctx: ShadeContext, shader: UsdShade.Shader) -> list[load.Input]:
  """The shader's authored input values (connections to other shaders and files that do not resolve:
  reported, skipped; an empty asset path: skipped)."""
  out: list[load.Input] = []
  for inp in shader.GetInputs():
    v, attr = util.producing(inp)
    if attr is not None and attr.GetName().startswith("outputs:"):
      ctx.report.drop(f"mdl.input_connection:{inp.GetBaseName()}", shader.GetPrim())
      continue
    if v is None or attr is None:
      continue
    if isinstance(v, Sdf.AssetPath) and not v.resolvedPath:
      # `@@` is "no texture" (the parameter's default, e.g. OmniPBR's unused slots): not reported.
      if v.path:
        ctx.report.drop(f"mdl.texture.unresolved:{inp.GetBaseName()}(parameter default)", shader.GetPrim())
      continue
    space = util.colorspace_of(attr, report=ctx.report, where=shader.GetPrim()) if isinstance(v, Sdf.AssetPath) else None
    out.append(load.Input(name=inp.GetBaseName(), value=v, space=space))
  return out
