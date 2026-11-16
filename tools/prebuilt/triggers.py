"""The compile-trigger inventory, read from the code: every input of a compile key and where it
comes from.

A kernel compiles (NVRTC for an OptiX PTX, Warp's codegen + NVRTC for a Warp module, OptiX's
compile for an OptiX module) when its key is not in the prebuilt bundle (or the module pack). The
keys and their inputs:

- **OptiX PTX** (`trace/optix/device/programs.ptx_key`): the build's `Features` (every field,
`repr`), the
  scene flags its programs are compiled for (`SceneFlags` but `has_omm`), the output scale, the
  PTX target, the Warp release and codegen settings, the OptiX headers' version.
- **OptiX module** (`pipeline.module_record`, one per PTX and record): the record's fields, among
  them the bound launch-parameter values (`params.bound_words`: the build word's two halves of
  each copy, the output word), the register cap and the pipeline's curves / custom primitives /
  micromaps.
- **Warp modules**: Warp hashes a module's source with its closures' values, its `wp.constant`s,
  its struct layouts and its options. Every value that reaches a kernel's code passes a device-code
  factory's argument (`@specialize` / a function defining a `wp.func`, `wp.kernel` or `wp.struct`:
  `factories`), a `wp.constant` (`constants`) or the module options
  (`kernel_structs.module_options`: the trees, the traversal order, the software words).

`derive` reads all of it from the code: the `Features` fields and the settings / scene / device
values each is resolved from (`features_sources`: an AST walk of `core/build/variants/resolve.py`),
every factory argument and what its call sites pass (`factory_inputs`: a Features field, another
factory's argument, a constant, or something else), the PTX key's parts, the module record's fields
and the word bits. `INVENTORY` (`trigger_table.py`) states, for each trigger, its domain, whether
the bundle covers it and what a miss costs; `tests/contracts/test_compile_triggers.py` fails when
the code has a trigger the table lacks (a new compile-key input never slips in silently) or the
table one the code no longer has.

  python -m tools.prebuilt.triggers report internal/research/kb/COMPILE_TRIGGERS.md
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
PACKAGE: Final = ROOT / "raywarp"
# The Warp decorators that make device code (a function or class defining one is a factory).
_DEVICE: Final = ("wp.kernel", "wp.func", "wp.struct", "wp.func_native", "wp.func_grad", "wp.func_replay")
# Parameter annotations whose value is a Features or a projection of it: their fields are the
# Features inventory's.
FEATURE_TYPES: Final = frozenset({"Features", "ProbeFeatures", "TraceFeatures", "LookFlags", "Trees", "TraceSpec"})


@dataclasses.dataclass(frozen=True, order=True)
class Param:
  """One argument of a device-code factory: "<module>:<factory>.<name>" and its annotation."""

  factory: str
  name: str
  annotation: str

  @property
  def id(self) -> str:
    """The inventory id."""
    return f"{self.factory}.{self.name}"


def _module_name(path: Path) -> str:
  """The dotted module of a library file."""
  return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _decorators(node: ast.FunctionDef | ast.ClassDef) -> list[str]:
  """The decorators of `node` as source text (calls by their function)."""
  return [ast.unparse(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list]


def _defines_device_code(node: ast.FunctionDef) -> bool:
  """Whether `node` defines a device function, kernel or struct in its body (a factory)."""
  for inner in ast.walk(node):
    if inner is not node and isinstance(inner, (ast.FunctionDef, ast.ClassDef)):
      if any(d.startswith(_DEVICE) for d in _decorators(inner)):
        return True
  return False


@functools.cache
def _trees() -> dict[str, ast.Module]:
  """Every library module's AST, by dotted name."""
  return {_module_name(p): ast.parse(p.read_text()) for p in sorted(PACKAGE.rglob("*.py"))}


@functools.cache
def factories() -> dict[str, ast.FunctionDef]:
  """Every device-code factory: "<module>:<function>" -> its definition (`@specialize`, or a
  function defining device code; methods and nested factories by their own name)."""
  out: dict[str, ast.FunctionDef] = {}
  for module, tree in _trees().items():
    for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef) and ("specialize" in _decorators(node) or _defines_device_code(node)):
        out[f"{module}:{node.name}"] = node
  return out


def params() -> list[Param]:
  """Every argument of every factory (`self` and `cls` but)."""
  out = []
  for name, node in factories().items():
    for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
      if a.arg not in ("self", "cls"):
        out.append(Param(name, a.arg, ast.unparse(a.annotation) if a.annotation else ""))
  return sorted(out)


# --- What the call sites pass ---------------------------------------------------------------------


@dataclasses.dataclass
class Sources:
  """What one function argument's call sites pass, resolved through the arguments they come from:
  Features fields ("Features.<field>"), settings / scene / output / camera values ("settings
  <path>", ...), and the expressions none of those (unresolved: "<function>(<argument>)")."""

  sources: set[str] = dataclasses.field(default_factory=set)
  params: set[str] = dataclasses.field(default_factory=set)

  @property
  def features(self) -> set[str]:
    """The Features fields."""
    return {s for s in self.sources if s.startswith("Features.")}

  @property
  def other(self) -> set[str]:
    """The unresolved sources."""
    return {s for s in self.sources if not s.startswith(_KNOWN)}


# Argument annotation (its last name) -> the source kind of its attributes.
_ROOT_KINDS: Final = {
  **dict.fromkeys(FEATURE_TYPES, "Features"),
  "RenderSettings": "settings",
  "GpuScene": "scene",
  "SceneFlags": "scene flags",
  "Layout": "scene layout",
  "OutputLayout": "outputs",
  "Outputs": "outputs",
  "DeviceCaps": "device",
  "CameraSpec": "camera",
  "ParamsLayout": "Features",
}
# FrameContext attribute -> its kind (the frame's resolved Features, settings, scene).
_CONTEXT: Final = {
  "features": "Features",
  "effective": "Features",
  "settings": "settings",
  "scene": "scene",
  "layout": "scene layout",
}
_KNOWN: Final = ("Features.", "settings", "scene", "outputs", "device", "camera")


def _locals(fn: ast.FunctionDef) -> dict[str, list[ast.expr]]:
  """Local name -> the expressions assigned to it in `fn` (tuple targets unpacked by position;
  loop variables to their iterable)."""
  out: dict[str, list[ast.expr]] = {}

  def bind(target: ast.expr, value: ast.expr) -> None:
    if isinstance(target, ast.Name):
      out.setdefault(target.id, []).append(value)
    elif isinstance(target, ast.Tuple):
      parts = value.elts if isinstance(value, ast.Tuple) and len(value.elts) == len(target.elts) else [value] * len(target.elts)
      for t, v in zip(target.elts, parts, strict=True):
        bind(t, v)

  for node in ast.walk(fn):
    if isinstance(node, ast.Assign):
      for t in node.targets:
        bind(t, node.value)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
      bind(node.target, node.value)
    elif isinstance(node, ast.NamedExpr):
      bind(node.target, node.value)
    elif isinstance(node, (ast.For, ast.comprehension)):
      bind(node.target, node.iter)
  return out


@dataclasses.dataclass
class _Scope:
  """The enclosing functions of a call site, innermost first: (definition, id, argument
  annotations, locals)."""

  frames: list[tuple[ast.FunctionDef, str, dict[str, str], dict[str, list[ast.expr]]]]


def _root(annotation: str, attrs: list[str]) -> str | None:
  """The source of `<argument annotated annotation>.attrs` (None: the argument itself is a node)."""
  name = annotation.split("[")[0].split(".")[-1].split(" ")[0]
  if name == "FrameContext" and attrs and attrs[0] in _CONTEXT:
    kind = _CONTEXT[attrs[0]]
    rest = attrs[1:]
    return (
      f"Features.{rest[0]}"
      if kind == "Features" and rest
      else kind
      if kind == "Features"
      else f"{kind} {'.'.join(rest)}".strip()
    )
  if name in _ROOT_KINDS:
    kind = _ROOT_KINDS[name]
    return f"Features.{attrs[0]}" if kind == "Features" and attrs else f"{kind} {'.'.join(attrs)}".strip()
  return None


def _classify(expr: ast.expr, scope: _Scope, out: Sources, seen: set[tuple[str, str]] | None = None) -> None:
  """Adds the sources of `expr` to `out` (`Sources`): an annotated argument's attribute
  (`_root`), another function's argument (resolved later), nothing for constants and code; each
  local name expanded once (`seen`)."""
  seen = set() if seen is None else seen
  if isinstance(expr, ast.Constant):
    return
  chain = _chain(expr)
  if chain is not None:
    root, attrs = chain
    if root[:1].isupper() or root in ("wp", "np", "math", "int", "float", "bool", "len", "None", "True", "False"):
      return  # a module constant, a type or a builtin: code
    if root in ("self", "ctx") and attrs:
      kind = _CONTEXT.get(attrs[0])
      rest = attrs[1:]
      if kind == "Features":
        out.sources.add(f"Features.{rest[0]}" if rest else "Features")
      elif kind is not None:
        out.sources.add(f"{kind} {'.'.join(rest)}".strip())
      else:
        fid = scope.frames[0][1] if scope.frames else "?"
        out.sources.add(f"state {fid.split(':')[0]}.{root}.{attrs[0]}")
      return
    for _fn, fid, annotations, local in scope.frames:
      if root in annotations:
        kind = _root(annotations[root], attrs)
        if kind is not None:
          out.sources.add(kind)
        else:
          out.params.add(f"{fid}.{root}")
        return
      if root in local:
        if (fid, root) not in seen:
          seen.add((fid, root))
          for value in local[root]:
            _classify(value, scope, out, seen)
        return
      if root in _nested(fid):
        return  # a nested function or class: code
    return  # a module-level name: code
  for child in ast.iter_child_nodes(expr):
    if isinstance(child, ast.expr):
      _classify(child, scope, out, seen)


@functools.cache
def _nested(fid: str) -> frozenset[str]:
  """The functions and classes defined inside function `fid`."""
  fn = _functions().get(fid)
  if fn is None:
    return frozenset()
  return frozenset(d.name for d in ast.walk(fn) if d is not fn and isinstance(d, (ast.FunctionDef, ast.ClassDef)))


def _chain(expr: ast.expr) -> tuple[str, list[str]] | None:
  """(root name, attributes) of `name.a.b`, else None."""
  attrs: list[str] = []
  while isinstance(expr, ast.Attribute):
    attrs.insert(0, expr.attr)
    expr = expr.value
  return (expr.id, attrs) if isinstance(expr, ast.Name) else None


def _annotations(fn: ast.FunctionDef) -> dict[str, str]:
  """Argument name -> annotation text of `fn`."""
  args = (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)
  return {a.arg: ast.unparse(a.annotation) if a.annotation else "" for a in args}


@functools.cache
def _functions() -> dict[str, ast.FunctionDef]:
  """Every function of the library: "<module>:<name>" (the last definition of a name wins)."""
  out = {}
  for module, tree in _trees().items():
    for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef):
        out[f"{module}:{node.name}"] = node
  return out


def _calls() -> Iterator[tuple[ast.Call, _Scope]]:
  """Every call in the library with its enclosing functions."""
  for module, tree in _trees().items():
    yield from _walk_calls(tree, [], module)


def _walk_calls(node: ast.AST, frames: list, module: str) -> Iterator[tuple[ast.Call, _Scope]]:
  """`_calls` under `node` (`frames`: the enclosing functions, outermost first)."""
  for child in ast.iter_child_nodes(node):
    if isinstance(child, ast.FunctionDef):
      inner = [*frames, (child, f"{module}:{child.name}", _annotations(child), _locals(child))]
      yield from _walk_calls(child, inner, module)
      continue
    if isinstance(child, ast.Call):
      yield child, _Scope(list(reversed(frames)))
    yield from _walk_calls(child, frames, module)


def _callee(call: ast.Call) -> str | None:
  """The called name (a function or attribute name)."""
  f = call.func
  return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


@functools.cache
def _arguments() -> dict[str, Sources]:
  """Every function argument id -> what its call sites pass (direct; a name shared by several
  functions: every one of them)."""
  by_name: dict[str, list[str]] = {}
  for fid in _functions():
    by_name.setdefault(fid.split(":")[1], []).append(fid)
  out: dict[str, Sources] = {}
  # A name several functions define: the call's own module's definition, else every one (the
  # resolution over-approximates: more sources, never fewer).
  for call, scope in _calls():
    named = by_name.get(_callee(call) or "", [])
    module = scope.frames[0][1].split(":")[0] if scope.frames else ""
    local = [fid for fid in named if fid.split(":")[0] == module]
    for fid in local or named:
      node = _functions()[fid]
      args = [a.arg for a in (*node.args.posonlyargs, *node.args.args) if a.arg not in ("self", "cls")]
      for i, value in enumerate(call.args):
        if not isinstance(value, ast.Starred) and i < len(args):
          _classify(value, scope, out.setdefault(f"{fid}.{args[i]}", Sources()))
      for kw in call.keywords:
        if kw.arg is not None:
          _classify(kw.value, scope, out.setdefault(f"{fid}.{kw.arg}", Sources()))
  return out


@functools.cache
def factory_inputs() -> dict[str, Sources]:
  """Factory argument id -> its sources (`Sources`), resolved through the arguments it is passed
  from (a fixed point); an argument no call reaches: "<id> (no caller)"."""
  direct = _arguments()
  # The arguments the factories' arguments come from, transitively.
  reach, todo = set(), [p.id for p in params()]
  while todo:
    k = todo.pop()
    if k not in reach:
      reach.add(k)
      todo += direct[k].params if k in direct else []
  sources = {k: set(direct[k].sources) if k in direct else {f"{k} (no caller)"} for k in reach}
  changed = True
  while changed:  # a fixed point over the argument graph (recursive factories make cycles)
    changed = False
    for k in reach:
      for p in direct[k].params if k in direct else ():
        if not sources[p] <= sources[k]:
          sources[k] |= sources[p]
          changed = True
  return {p.id: Sources(sources.get(p.id, {f"{p.id} (no caller)"})) for p in params()}


def open_inputs() -> dict[str, Sources]:
  """The factory arguments with a source but a Features field, a constant or code: settings,
  scene, outputs, device or camera values read directly, or unresolved expressions (each needs
  its `trigger_table.INVENTORY` row)."""
  return {k: v for k, v in factory_inputs().items() if v.sources - v.features}


# --- Features fields and their sources ------------------------------------------------------------

# The roots `features_sources` maps to a source kind.
_SETTINGS_ROOTS: Final = ("s", "settings", "lit", "sh", "aa", "r", "a", "g")


def features_sources() -> dict[str, set[str]]:
  """Features field -> the values `resolve._resolve` and its helpers read for it: "settings
  <group>.<field>", "scene <attribute>", "declared <field>", "caps <attribute>" (the cameras),
  "device <attribute>", "tuning", "routing" (set by the tier routing, `tiers.py`)."""
  from raywarp.core.kernels.abi.features import Features

  tree = _trees()["raywarp.core.build.variants.resolve"]
  fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
  out: dict[str, set[str]] = {f.name: set() for f in dataclasses.fields(Features)}
  for fn in fns.values():
    if fn.name in _NOT_SOURCES:
      continue
    local = _locals(fn)
    for name, value in _assigned_fields(fn, set(out)):
      out[name] |= _reads(value, local, fns)
  for name in ("switches", "bound", "word"):
    out[name].add("routing tiers.with_switches / tiers.word")
  # Set from `_resolve`'s own arguments: the routing's (a tier build).
  for name in ("run_sampling", "run_counts", "scene_free"):
    out[name].add("routing resolve._resolve argument")
  out["light_forms"].add("routing resolve._light_forms (scene nlight, settings light grid)")
  return out


# `resolve.py` functions whose Features are the routing's or the lab's, not a source of a field
# (a tier build's re-resolution).
_NOT_SOURCES: Final = frozenset({"_tier_build", "resolve_features", "_resolve_and_route"})
# Calls returning whole Features: their fields are each one's own row ("Features (other fields)").
_FEATURES_CALLS: Final = frozenset({"Features", "replace", "_resolve", "_tier_build", "canonical"})


def _assigned_fields(fn: ast.FunctionDef, fields: set[str]) -> Iterator[tuple[str, ast.expr]]:
  """(Features field, its value) of the `Features(...)` keywords and the TypedDict literals
  (`{"field": value}`) in `fn`."""
  for node in ast.walk(fn):
    if isinstance(node, ast.Call) and _callee(node) in ("Features", "replace"):
      for kw in node.keywords:
        if kw.arg in fields:
          yield kw.arg, kw.value
    if isinstance(node, ast.Dict):
      for k, v in zip(node.keys, node.values, strict=True):
        if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value in fields:
          yield k.value, v


def _reads(expr: ast.expr, local: dict[str, list[ast.expr]], fns: dict[str, ast.FunctionDef], depth: int = 0) -> set[str]:
  """The sources `expr` reads (`features_sources`), through `fn`'s locals and the helpers it
  calls (their sources whole)."""
  out: set[str] = set()
  if depth > 8:
    return out
  for node in ast.walk(expr):
    if isinstance(node, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops):
      for c in (node.left,):
        if isinstance(c, ast.Constant) and isinstance(c.value, str):
          comparators = [ast.unparse(x) for x in node.comparators]
          kind = "declared" if "declared" in comparators else "free" if "free" in comparators else None
          if kind == "declared":
            out.add(f"declared {c.value}")
    chain = _chain(node) if isinstance(node, (ast.Attribute, ast.Name)) else None
    if chain is None:
      if isinstance(node, ast.Call) and _callee(node) in _FEATURES_CALLS - {"Features"}:
        out.add("Features (other fields)")
      elif isinstance(node, ast.Call) and _callee(node) in fns and depth < 3:
        helper = fns[str(_callee(node))]
        for value in (r.value for r in ast.walk(helper) if isinstance(r, ast.Return) and r.value is not None):
          out |= _reads(value, _locals(helper), fns, depth + 1)
      elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        out.add(f"call {ast.unparse(node.func)}")  # another module's helper (grading.flags_of)
      continue
    root, attrs = chain
    if root in local and not attrs:
      for v in local[root]:
        if isinstance(v, ast.Call) and _callee(v) in _FEATURES_CALLS:
          out.add("Features (other fields)")
          continue
        out |= _reads(v, local, fns, depth + 1)
    elif root in local and attrs:
      for v in local[root]:
        prefix = _chain(v)
        if prefix is not None:
          out |= _source(prefix[0], [*prefix[1], *attrs])
    else:
      out |= _source(root, attrs)
  return out


# Bare arguments of the resolution helpers that carry a source of their own.
_PARAMS: Final = {"casters": "scene shadow casters (count)", "alpha": "declared PerWorld.alpha"}


def _source(root: str, attrs: list[str]) -> set[str]:
  """The source of attribute chain `root.attrs` (`features_sources` kinds; empty: not a source)."""
  if not attrs:
    return {_PARAMS[root]} if root in _PARAMS else set()
  if root in ("s", "settings") and len(attrs) >= 2:
    return {f"settings {attrs[0]}.{attrs[1]}"}
  if root in ("scene", "sc", "env", "t", "layers", "prepared", "semantic") or (root == "caps" and attrs[0] == "scene"):
    return {f"scene {'.'.join(attrs[1:] if root == 'caps' else attrs)}"}
  if root == "caps":
    return {f"camera {attrs[0]}"}
  if root == "device":
    return {f"device {attrs[0]}"}
  if root in ("perf",):
    return {f"tuning {attrs[0]}"}
  if root in ("backend",):
    return {f"device backend.{attrs[0]}"}
  if root == "layout":
    return {f"scene layout.{attrs[0]}"}
  return set()


# --- OptiX keys ---------------------------------------------------------------------------------


def ptx_key_parts() -> list[str]:
  """The parts of `programs.ptx_key` (the `PtxParts` values `ptx_parts` builds, as source text)."""
  tree = _trees()["raywarp.core.trace.optix.device.programs"]
  fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ptx_parts")
  call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "PtxParts")
  return [ast.unparse(k.value) for k in call.keywords]


def scene_flags() -> list[str]:
  """The `programs.SceneFlags` fields (the PTX key's scene part but `has_omm`)."""
  from raywarp.core.trace.optix.device import programs

  return list(programs.SceneFlags.__annotations__)


def module_record_fields() -> list[str]:
  """The `prebuilt.ModuleRecord` fields (an OptiX module's options)."""
  from raywarp.core.build.kernel_cache import prebuilt

  return [f.name for f in dataclasses.fields(prebuilt.ModuleRecord)]


def word_bits() -> dict[str, int]:
  """The bound build-word bits (`kernels/abi/toggles.py`): name -> bit, the second half's as
  `toggles.high` values. Every bit is an OptiX module key input (`params.bound_words`) and, on the
  software path, a megakernel module's (`toggles.preamble`)."""
  from raywarp.core.build.variants import tiers
  from raywarp.core.kernels.abi import toggles

  out = {f"class {k}": v for k, v in toggles.CLASS_BITS.items()}
  out |= {f"shape {k}": v for k, v in toggles.SHAPE_BITS.items()}
  out |= {f"field {k}": v for k, v in tiers.FIELD_BITS.items()}
  out |= {f"bound {k}": v for k, v in tiers.BOUND_FIELDS.items()}
  out |= {f"setting {k}": toggles.high(v) for k, v in tiers.SETTING_FIELDS.items()}
  out |= {k: getattr(toggles, k) for k in dir(toggles) if k.startswith("GEO_") and k != "GEO_ALL"}
  out |= {"sampling SOFT_SHADOWS": toggles.high(toggles.SOFT_SHADOWS), "sampling ANISOTROPY": toggles.high(toggles.ANISOTROPY)}
  out |= {"FLEX": toggles.FLEX, "FLEX_MATERIALS": toggles.FLEX_MATERIALS}
  return out


def constants() -> dict[str, str]:
  """Every `wp.constant` of the library: "<module>:<name>" -> its value's source (code: a Warp
  module's hash holds it)."""
  out = {}
  for module, tree in _trees().items():
    for node in tree.body:
      target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0] if isinstance(node, ast.Assign) else None
      value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
      if isinstance(value, ast.Call) and ast.unparse(value.func) == "wp.constant" and isinstance(target, ast.Name):
        out[f"{module}:{target.id}"] = ast.unparse(value.args[0])
  return out


# Primitive argument annotations: a value the factory's code is specialised on.
_PRIMITIVE: Final = frozenset({"bool", "int", "float", "str", "tuple", "int | None", "float | None", "bool | None"})


def _base_type(annotation: str) -> str:
  """The type name of an annotation ("rays.PrimaryRay | None" -> "PrimaryRay")."""
  return annotation.split("|")[0].split("[")[0].strip().split(".")[-1]


def factory_values() -> set[str]:
  """The primitive arguments of the device-code factories that are not named after a Features
  field: values whose source the inventory states (`trigger_table.FACTORY_INPUTS`)."""
  from raywarp.core.kernels.abi.features import Features

  fields = {f.name for f in dataclasses.fields(Features)}
  return {
    p.name
    for p in params()
    if (p.annotation in _PRIMITIVE or p.annotation.split("[")[0] in _PRIMITIVE) and p.name not in fields
  }


def factory_types() -> set[str]:
  """The other argument types of the device-code factories (`trigger_table.FACTORY_TYPES`)."""
  return {
    _base_type(p.annotation)
    for p in params()
    if p.annotation not in _PRIMITIVE and p.annotation.split("[")[0] not in _PRIMITIVE
  }


def nested_constants() -> list[str]:
  """`wp.constant` calls inside a function (a value per call: a compile input the inventory does
  not hold); none expected."""
  out = []
  for module, tree in _trees().items():
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
      out += [f"{module}:{fn.name}" for c in ast.walk(fn) if isinstance(c, ast.Call) and ast.unparse(c.func) == "wp.constant"]
  return out


def derive() -> dict[str, set[str]]:
  """Every trigger id the code has, by kind: "features" (Features fields), "ptx" (the PTX key's
  parts), "flags" (scene flags), "record" (module record fields), "word" (bound bits), "factory"
  (primitive factory arguments not named after a Features field), "types" (the factories' other
  argument types)."""
  return {
    "features": set(features_sources()),
    "ptx": set(ptx_key_parts()),
    "flags": set(scene_flags()),
    "record": set(module_record_fields()),
    "word": set(word_bits()),
    "factory": factory_values(),
    "types": factory_types(),
  }


def report() -> str:
  """The inventory as markdown (`trigger_table.render`)."""
  from tools.prebuilt import trigger_table

  return trigger_table.render(derive(), features_sources())


if __name__ == "__main__":
  if sys.argv[1:2] == ["report"] and len(sys.argv) == 3:  # report OUT.md
    Path(sys.argv[2]).write_text(report())
  else:
    print(__doc__)
