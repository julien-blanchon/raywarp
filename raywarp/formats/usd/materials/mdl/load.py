"""MDL modules loaded, instantiated with authored arguments, compiled and distilled by the SDK.

Search paths, in order: the module's own root (its directory, or the root of the package its
absolute imports name), the directories of the stage layers that author MDL shaders,
`MDL_USER_PATH` / `MDL_SYSTEM_PATH`, Omniverse's own library shipped in `library/` (OmniPBR,
OmniGlass, OmniSurface and its presets, SimPBR, OmniHair, OmniUe4: a stage's own copy wins), and
the bundled SDK's modules (`::nvidia::core_definitions`). Modules live in one DB scope per root,
so two stages (or two folders of one stage) may each ship their own `OmniUe4Base.mdl`.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

import numpy as np
from pxr import Sdf

from raywarp.formats.usd.materials.mdl import sdk
from raywarp.formats.usd.materials.mdl.bake import PROCEDURAL_SIZE
from raywarp.formats.usd.options import MDL_LIBRARY, MDL_PATH_VARIABLES
from raywarp.formats.usd.util import UsdValue, floats
from raywarp.scene.images import colorspace, files
from raywarp.scene.images import decode as images
from raywarp.scene.images.colorspace import ColorSpace

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.mdl import sdk_types as st

# Omniverse's own MDL library shipped with raywarp (Omniverse Core Materials 2.1.4, BSD-3;
# library/NOTICE.md), and its search roots.
LIBRARY: Final = MDL_LIBRARY
LIBRARY_ROOTS: Final = tuple(str(LIBRARY / d) for d in ("Base", "Ue4", "mdl"))
# The distiller target: the SDK's PBR model closest to the OpenPBR subset the core renders.
TARGET: Final = "transmissive_pbr"
# Top-level packages found on the SDK's own paths (standard modules, the bundled nvidia::).
_SDK_PACKAGES: Final = frozenset({"df", "state", "math", "tex", "anno", "base", "debug", "limits", "scene", "std", "nvidia"})
_IMPORT: Final = re.compile(r"\b(?:import|using)\s+::(\w+)::")
_PARENT_IMPORT: Final = re.compile(r"\b(?:import|using)\s+((?:\.\.::)+)")
# A resource climbing out of the module's directory (`texture_2d("../textures/x.png")`).
_PARENT_RESOURCE: Final = re.compile(
  r'\b(?:texture_2d|texture_3d|texture_cube|light_profile|bsdf_measurement)\s*\(\s*"(?:\./)?((?:\.\./)+)'
)
# The most directories a module's imports / resources may climb to find its root; the largest
# module file read (the SDK parses what it loads: a bound on what a stage makes it read).
_MAX_CLIMB: Final = 4
MAX_MODULE_BYTES: Final = 32 << 20
_IDENTIFIER: Final = re.compile(r"[A-Za-z_]\w*")
# A resource a module body names (`texture_2d("./textures/x.png", ...)`).
_RESOURCE: Final = re.compile(r'\b(?:texture_2d|texture_3d|texture_cube|light_profile|bsdf_measurement)\s*\(\s*"([^"]+)"')
# The SDK's error code of a resource file it cannot find.
_MISSING_RESOURCE: Final = "C298"
# The SDK texture gamma of the sRGB transfer (the SDK has no piecewise sRGB) and of a file's own
# encoding (`gamma_default`: an input authored without a color space).
_SRGB_GAMMA: Final = 2.2
_DEFAULT_GAMMA: Final = 0.0
# Decoded channels -> the canvas's RGBA (grey, grey + alpha, RGB, RGBA).
_RGBA: Final = {1: [0, 0, 0], 2: [0, 0, 0, 1], 3: [0, 1, 2], 4: [0, 1, 2, 3]}
_RGB: Final = 3


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ModuleRef:
  """A module to load: its MDL name, the search root it needs ('' when it is searched on the
  session's paths) and its file ('' when only its name is known)."""

  name: str
  root: str
  file: str


def module_of(file: str) -> ModuleRef:
  """The module of file `file`, named under the root its imports need: the root of the package
  its absolute imports name when the file sits in it (vMaterials, `::nvidia::sdk_examples`), the
  ancestor its `..::` imports or `../` resources climb to (the OmniSurface presets'
  `..::Templates`), else its own directory (`::<stem>`)."""
  path = Path(file)
  text = _source(file)
  packages = set(_IMPORT.findall(text)) - (_SDK_PACKAGES - {"nvidia"})
  for ancestor in path.parents:
    if ancestor.name in packages:
      ref = _under(path, ancestor.parent)
      if ref is not None:
        return ref
  up = max((m.count("..") for m in (*_PARENT_IMPORT.findall(text), *_PARENT_RESOURCE.findall(text))), default=0)
  if 0 < up <= _MAX_CLIMB and up < len(path.parents):
    ref = _under(path, path.parents[up])
    if ref is not None:
      return ref
  return ModuleRef(name=f"::{path.stem}", root=str(path.parent), file=file)


def _under(path: Path, root: Path) -> ModuleRef | None:
  """The module of `path` under search root `root` (None: a directory name is not an MDL
  identifier)."""
  rel = path.relative_to(root).with_suffix("")
  if not all(_IDENTIFIER.fullmatch(p) for p in rel.parts):
    return None
  return ModuleRef(name="::" + "::".join(rel.parts), root=str(root), file=str(path))


def module_named(authored: str) -> ModuleRef | None:
  """The module of an unresolved authored path searched on the MDL paths (`OmniPBR.mdl`,
  `nvidia/core_definitions.mdl`; None for paths that cannot name one)."""
  rel = Path(authored.removeprefix("./")).with_suffix("")
  if "://" in authored or not rel.parts or not all(_IDENTIFIER.fullmatch(p) for p in rel.parts):
    return None
  return ModuleRef(name="::" + "::".join(rel.parts), root="", file="")


def resources(file: str) -> set[str]:
  """The files the relative resource literals of module `file` name next to it (a module that
  loads with missing resources depends on them)."""
  here = Path(file).parent
  return {os.path.normpath(here / r.removeprefix("./")) for r in _RESOURCE.findall(_source(file)) if not r.startswith("/")}


def _source(file: str) -> str:
  """The text of module file `file`, at most `MAX_MODULE_BYTES` ('' when it cannot be read)."""
  try:
    return files.read(file, MAX_MODULE_BYTES).decode(errors="replace")
  except OSError:
    return ""


def env_paths() -> tuple[str, ...]:
  """The directories of MDL_USER_PATH and MDL_SYSTEM_PATH."""
  return tuple(d for var in MDL_PATH_VARIABLES for d in os.environ.get(var, "").split(os.pathsep) if d)


@final
class Session:
  """One stage read's SDK state: a DB scope (and transaction) per module root, the stage's search
  roots, the textures created for authored inputs."""

  def __init__(self, api: sdk.Api) -> None:
    """A session of the started SDK `api`."""
    self.api = api
    self.roots: list[str] = []  # stage layer directories, in discovery order
    self.scopes: dict[str, st.IScope] = {}
    # The open transactions by (root, lane). Neuray documents transactions and const DB
    # access as thread-safe, but mutable DB elements and their interfaces need serial use.
    # Separate lanes also keep each worker's edit/commit and SDK handle lifetimes isolated;
    # lane 0 is the read's.
    self.transactions: dict[tuple[str, int], st.ITransaction] = {}
    # The file and authored color space (None: the file's) of each texture made for an input, by
    # DB name (one per module root's scope: a texture is visible in its own scope only).
    self.textures: dict[sdk.DbName, tuple[str, ColorSpace | None]] = {}
    self.refs: dict[str, ModuleRef | None] = {}  # `module_ref` by source
    self.undecodable: set[sdk.DbName] = set()  # the input textures that do not decode (tried once)

  def module_ref(self, resolved: str, authored: str) -> ModuleRef | None:
    """The module of a shader's source (`module_of` its resolved file, else `module_named` its
    authored path), once per session (a module file is scanned once)."""
    key = resolved or authored
    if key not in self.refs:
      self.refs[key] = module_of(resolved) if resolved else module_named(authored)
    return self.refs[key]

  def add_root(self, directory: str) -> None:
    """Adds a stage layer directory to the search roots."""
    if directory and directory not in self.roots:
      self.roots.append(directory)

  def transaction(self, root: str, lane: int = 0) -> st.ITransaction:
    """The transaction of lane `lane` in `root`'s scope (scope and transaction made on first
    use)."""
    if root not in self.scopes:
      db = self.api.database
      self.scopes[root] = db.create_scope(db.get_global_scope(), sdk.SCOPE_PRIVACY)
    if (root, lane) not in self.transactions:
      self.transactions[root, lane] = self.scopes[root].create_transaction()
    return self.transactions[root, lane]

  def commit(self) -> None:
    """Commits every open transaction: what they stored (modules, textures) is visible to the
    transactions opened after. The SWIG handles obtained through them must be gone first:
    collected here, since their reference cycles outlive the steps that used them."""
    gc.collect()
    open_, self.transactions = list(self.transactions.values()), {}
    for tr in open_:
      tr.commit()

  def paths(self, root: str) -> tuple[str, ...]:
    """The search paths of a module of `root`, in order."""
    return tuple(dict.fromkeys(p for p in (root, *self.roots, *env_paths(), *LIBRARY_ROOTS, *self.api.system_paths) if p))

  def search(self, root: str) -> None:
    """Sets the SDK's search paths (global to the process) for a module of `root`."""
    paths = self.paths(root)
    with sdk.LOCK:
      if paths == sdk.SEARCH.applied:
        return
      cfg = self.api.config
      cfg.clear_mdl_paths()
      cfg.clear_resource_paths()
      for p in paths:
        cfg.add_mdl_path(p)
        cfg.add_resource_path(p)
      sdk.SEARCH.applied = paths

  def candidates(self, ref: ModuleRef, found: str = "") -> list[str]:
    """The files a module of `ref`'s name would be found at on its search paths, up to the one
    it was `found` at (the read depends on them: one appearing later shadows it or makes a
    failed load succeed)."""
    rel = Path(*ref.name.removeprefix("::").split("::")).with_suffix(".mdl")
    out = [str(Path(p) / rel) for p in self.paths(ref.root)]
    real = os.path.realpath(found) if found else ""
    cut = next((i for i, c in enumerate(out) if real and os.path.realpath(c) == real), len(out))
    return out[:cut]

  def load(self, ref: ModuleRef) -> tuple[st.IModule | None, str]:
    """(the loaded module, '') or (None, the SDK's error). A module whose resources (textures)
    do not resolve loads with them invalid ('missing resources' as its note)."""
    if ref.file and files.size(ref.file) > MAX_MODULE_BYTES:
      return None, f"module larger than {MAX_MODULE_BYTES >> 20} MiB"
    self.search(ref.root)
    tr = self.transaction(ref.root)
    context = self.api.factory.create_execution_context()
    note = ""
    if self.api.impexp.load_module(tr, ref.name, context) < 0:
      error = sdk.messages(context) or f"cannot load {ref.name}"
      context = self.api.factory.create_execution_context()
      context.set_option("resolve_resources", False)  # noqa: FBT003
      if _MISSING_RESOURCE not in error or self.api.impexp.load_module(tr, ref.name, context) < 0:
        return None, error
      note = "missing resources"
    db_name = self.api.factory.get_db_module_name(ref.name)
    module = tr.access_as(self.api.bindings.IModule, db_name.get_c_str()) if sdk.valid(db_name) and db_name else None
    return (module, note) if sdk.valid(module) and module else (None, f"no module {ref.name}")

  def files(self, root: str, module: st.IModule) -> set[str]:
    """The files of `module` and of every module it imports, transitively (standard ones have
    none)."""
    tr = self.transaction(root)
    out: set[str] = set()
    todo, seen = [module], set[str]()
    while todo:
      m = todo.pop()
      if m.get_filename():
        out.add(m.get_filename())
      for i in range(m.get_import_count()):
        name = m.get_import(i)
        imported = tr.access_as(self.api.bindings.IModule, name) if name not in seen else None
        seen.add(name)
        if imported is not None and sdk.valid(imported):
          todo.append(imported)
    return out

  def texture(self, root: str, path: str, space: ColorSpace | None, lane: int = 0) -> sdk.DbName | None:
    """The DB name of a texture of file `path` read in color space `space` (None: the file's
    own encoding) in `root`'s scope, created once per session (in lane `lane`'s transaction);
    None when it does not decode."""
    digest = hashlib.sha1(f"{root}\0{path}\0{space}".encode(), usedforsecurity=False).hexdigest()[:20]
    name = sdk.DbName(f"raywarp::texture::{digest}")
    if name in self.textures or name in self.undecodable:
      return None if name in self.undecodable else name
    tr = self.transaction(root, lane)
    b = self.api.bindings
    image = tr.create_as(b.IImage, "Image")
    if image.reset_file(path) != 0 and not self._decoded(image, path):
      self.undecodable.add(name)
      return None
    tr.store(image, f"{name}::image")
    texture = tr.create_as(b.ITexture, "Texture")
    texture.set_image(f"{name}::image")
    texture.set_gamma(_gamma(space))
    tr.store(texture, name)
    self.textures[name] = path, space
    return name

  def _decoded(self, image: st.IImage, path: str) -> bool:
    """Sets `image` from raywarp's decode of a file the SDK's image plugins do not read (a
    Pillow format without an SDK image plugin); False when that fails too."""
    source = images.ImageSource(path=path, max_size=PROCEDURAL_SIZE)  # the bakes' largest side
    info = images.probe(source)
    texels, error = images.decode_base(source, info)
    if info.error or error:
      return False
    rgba = np.asarray(texels)[..., _RGBA[texels.shape[2]]]
    if rgba.shape[2] == _RGB:
      rgba = np.concatenate([rgba, np.full_like(rgba[..., :1], 255 if rgba.dtype == np.uint8 else 1.0)], axis=2)
    pixel, dtype = ("Rgba", np.uint8) if rgba.dtype == np.uint8 else ("Color", np.float32)
    canvas = self.api.images.create_canvas(pixel, rgba.shape[1], rgba.shape[0])
    view = canvas.get_tile().get_data_numpy()  # the canvas's own buffer, row 0 at the bottom
    view.reshape(-1)[:] = np.ascontiguousarray(rgba[::-1], dtype).view(view.dtype).reshape(-1)
    return image.set_from_canvas(canvas)

  def close(self) -> None:
    """Commits every transaction (`commit`) and removes the session's scopes."""
    self.commit()
    scopes, self.scopes = list(self.scopes.values()), {}
    for scope in scopes:
      self.api.database.remove_scope(scope.get_id())


def _gamma(space: ColorSpace | None) -> float:
  """The SDK texture gamma of an authored color space: its transfer's (`colorspace`; sRGB as
  2.2), the file's own for None. The SDK's bakes keep the space's primaries (reported:
  `mdl.bake_gamut`)."""
  if space is None:
    return _DEFAULT_GAMMA
  transfer = colorspace.texel_transfer(space)
  return _SRGB_GAMMA if transfer == "srgb" else 1.0 if transfer == "linear" else float(transfer)


def definition_name(module: st.IModule, material: str) -> str | None:
  """The DB name of material `material` of a loaded module (None: it has none)."""
  prefix = f"mdl{module.get_mdl_name()}::{material}("
  names = [module.get_material(i) for i in range(module.get_material_count())]
  return next((n for n in names if n.startswith(prefix)), None)


def definition(session: Session, ref: ModuleRef, name: str, lane: int = 0) -> st.IFunction_definition | None:
  """The material definition of DB name `name`, through lane `lane`'s transaction."""
  fn = session.transaction(ref.root, lane).access_as(session.api.bindings.IFunction_definition, name)
  return fn if sdk.valid(fn) else None


# Authored inputs -> arguments ----------------------------------------------------------------


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Input:
  """An authored shader input: name, value, and its color space (asset inputs)."""

  name: str
  value: UsdValue
  space: ColorSpace | None


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Arguments:
  """A material's argument list (`arguments`): the authored inputs, then a default-constructed
  value (zero, an invalid texture) for each parameter without a default that no input sets."""

  args: st.IExpression_list
  # The inputs that do not map: unknown parameter, type mismatch or missing texture.
  skipped: list[str]
  # The parameters without a default given their default-constructed value.
  zeroed: list[str]


def arguments(
  session: Session, ref: ModuleRef, fn: st.IFunction_definition, inputs: Sequence[Input], lane: int = 0
) -> Arguments:
  """The `Arguments` of `fn` for the authored `inputs`, made in lane `lane`'s transaction.

  A parameter without a default (SimPBR_RoadPaint's `TileOrigin` / `TileSize`) that no input
  sets takes its type's default-constructed value, as Omniverse RTX instantiates such a
  material (the SDK refuses a call without it: the material would be the error material)."""
  tr = session.transaction(ref.root, lane)
  values = session.api.factory.create_value_factory(tr)
  exprs = session.api.factory.create_expression_factory(tr)
  args = exprs.create_expression_list()
  types = fn.get_parameter_types()
  skipped: list[str] = []
  for inp in inputs:
    v = None
    if fn.get_parameter_index(inp.name) >= 0:
      v = _value(session, ref, values, types.get_type(inp.name).skip_all_type_aliases(), inp, lane)
    if v is None:
      skipped.append(inp.name)
      continue
    args.add_expression(inp.name, exprs.create_constant(v))
  zeroed = [name for name in _without_default(fn) if args.get_index(name) < 0]
  for name in zeroed:
    args.add_expression(name, exprs.create_constant(values.create(types.get_type(name))))
  return Arguments(args=args, skipped=skipped, zeroed=zeroed)


def _without_default(fn: st.IFunction_definition) -> list[str]:
  """The names of `fn`'s parameters that have no default (`get_index`: -1 for a name the list
  does not hold)."""
  types, defaults = fn.get_parameter_types(), fn.get_defaults()
  names = (types.get_name(i) for i in range(types.get_size()))
  return [name for name in names if defaults.get_index(name) < 0]


def _value(session: Session, ref: ModuleRef, vf: st.IValue_factory, t: st.IType, inp: Input, lane: int) -> st.IValue | None:  # noqa: PLR0913
  """The SDK value of type `t` of an authored input (None when it does not convert)."""
  kind, v = t.get_kind().name, inp.value
  try:
    if kind == "TK_TEXTURE":
      path = (v.resolvedPath or "") if isinstance(v, Sdf.AssetPath) else ""
      name = session.texture(ref.root, path, inp.space, lane) if path else None
      return vf.create_texture(t, name) if name else None
    if kind == "TK_ENUM":
      e = t.get_interface(session.api.bindings.IType_enumeration)
      index = e.find_value(v if isinstance(v, str) else int(floats(v)[0]))
      return vf.create_enum(e, index) if index >= 0 else None
    if kind == "TK_STRING":
      return vf.create_string(str(v))
    xs = floats(v)
  except (TypeError, ValueError):
    return None
  if kind == "TK_COLOR" and len(xs) == _RGB:
    return vf.create_color(xs[0], xs[1], xs[2])
  if kind in _SCALARS and len(xs) == 1:
    return _SCALARS[kind](vf, xs[0])
  if kind != "TK_VECTOR":
    return None
  vt = t.get_interface(session.api.bindings.IType_vector)
  element = _SCALARS.get(vt.get_element_type().get_kind().name)
  if vt.get_size() != len(xs) or element is None:
    return None
  vec = vf.create_vector(vt)
  for i, x in enumerate(xs):
    vec.set_value(i, element(vf, x))
  return vec


# The SDK value of a scalar kind (a double parameter takes a double: a float would not match).
_SCALARS: Final = {
  "TK_BOOL": lambda vf, x: vf.create_bool(bool(x)),
  "TK_INT": lambda vf, x: vf.create_int(int(x)),
  "TK_FLOAT": lambda vf, x: vf.create_float(float(x)),
  "TK_DOUBLE": lambda vf, x: vf.create_double(float(x)),
}


# Compile and distill ------------------------------------------------------------------------


def compile_material(
  session: Session, fn: st.IFunction_definition, args: st.IExpression_list, meters_per_unit: float
) -> st.ICompiled_material | str:
  """The material instance-compiled, or the error. The compile folds
  `state::meters_per_scene_unit` and inlines `noinline` helpers (OmniPBR's `base::` lookups
  become intrinsic calls)."""
  api = session.api
  rc = api.bindings.ReturnCode()
  call = fn.create_function_call(args, rc)
  if call is None or not sdk.valid(call) or rc.value != 0:
    return f"instantiate ({rc.value})"
  context = api.factory.create_execution_context()
  context.set_option("meters_per_scene_unit", meters_per_unit)
  context.set_option("fold_meters_per_scene_unit", True)  # noqa: FBT003
  context.set_option("internal_space", "coordinate_world")
  context.set_option("ignore_noinline", True)  # noqa: FBT003
  instance = call.get_interface(api.bindings.IMaterial_instance)
  compiled = instance.create_compiled_material(api.instance_compilation, context)
  return compiled if compiled is not None and sdk.valid(compiled) else f"compile: {sdk.messages(context)}"


def distill(api: sdk.Api, compiled: st.ICompiled_material) -> st.ICompiled_material | str:
  """A compiled material distilled to `TARGET`, or the error (thread-safe: the SDK releases the
  GIL)."""
  rc = api.bindings.ReturnCode()
  distilled = api.distiller.distill_material(compiled, TARGET, None, rc)
  if distilled is None or not sdk.valid(distilled) or rc.value != 0:
    return f"distill ({rc.value})"
  return distilled
