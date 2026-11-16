"""The kernel-code list (`core/build/kernel_cache/kernel_code.py`) is whole: every module with
device code is listed (in the PTX list or as a plain Warp module), kernel code reads host modules
only through the listed constants and host-only names, and the OptiX programs' generated code comes
from the PTX list only."""

from __future__ import annotations

import ast
import enum
import importlib
import re
from pathlib import Path

import pytest
from conftest import REPO

from raywarp.cache import code_version
from raywarp.core.build.kernel_cache import kernel_code

ROOT = REPO / "raywarp"
# A device-code definition: Warp's decorators or calls, a native builtin, CUDA source in a string.
_DEVICE_CALLS = frozenset({"wp.func", "wp.kernel", "wp.struct", "wp.func_native", "wp.build_experimental.add_builtin"})
_CUDA = re.compile(r"#include\s*<|__device__|__forceinline__")
# Host values hashed by value when kernel code reads them (`HOST_CONSTANTS`).
_DATA = (bool, int, float, str, bytes, tuple, list, dict, frozenset, set, enum.Enum)


def _name(path: Path) -> str:
  """The dotted module name of a raywarp source file."""
  parts = list(path.relative_to(REPO).with_suffix("").parts)
  return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _sources() -> dict[str, ast.Module]:
  """Every raywarp module's syntax tree, by dotted name."""
  return {_name(p): ast.parse(p.read_text()) for p in sorted(ROOT.rglob("*.py")) if "__pycache__" not in p.parts}


def _callee(node: ast.expr) -> str:
  """A decorator's or call's callee as text (`wp.func(inline=False)` -> "wp.func")."""
  return ast.unparse(node.func if isinstance(node, ast.Call) else node)


def _device_code(tree: ast.Module) -> bool:
  """Whether a module defines device code."""
  for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and any(_callee(d) in _DEVICE_CALLS for d in node.decorator_list):
      return True
    if isinstance(node, ast.Call) and _callee(node) in _DEVICE_CALLS:
      return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and _CUDA.search(node.value):
      return True
  return False


def _runtime_nodes(tree: ast.Module) -> list[ast.AST]:
  """The nodes of `tree` outside `if TYPE_CHECKING:` blocks."""
  hidden: set[int] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
      hidden.update(id(n) for n in ast.walk(node))
  return [n for n in ast.walk(tree) if id(n) not in hidden]


def _host_imports(name: str, tree: ast.Module, kernel: set[str], modules: set[str]) -> list[tuple[str, str, str]]:
  """(source module, imported name or "*" for a whole module, local alias) of every raywarp
  import of kernel-code module `name` from a module outside `kernel` (any depth, but typing-only
  imports)."""
  out = []
  for node in _runtime_nodes(tree):
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("raywarp"):
      for alias in node.names:
        whole = f"{node.module}.{alias.name}"
        if whole in modules:
          if whole not in kernel:
            out.append((whole, "*", alias.asname or alias.name))
        elif node.module not in kernel:
          out.append((node.module, alias.name, alias.asname or alias.name))
    elif isinstance(node, ast.Import):
      out += [(a.name, "*", a.asname or a.name) for a in node.names if a.name.startswith("raywarp") and a.name not in kernel]
  return out


def _attributes(tree: ast.Module, alias: str) -> set[str]:
  """The attributes read on local name `alias` (`alias.x`)."""
  return {
    n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias
  }


def test_every_device_code_module_is_listed() -> None:
  """A module defining device code is kernel code (`PTX_PACKAGES` / `PTX_MODULES`) or a plain Warp
  module (`WARP_MODULES`); the lists name existing device-code modules only."""
  sources = _sources()
  kernel = set(kernel_code.ptx_modules())
  device = {name for name, tree in sources.items() if _device_code(tree)}
  assert not (kernel & kernel_code.WARP_MODULES)
  missing = sorted(device - kernel - kernel_code.WARP_MODULES)
  assert not missing, f"device code outside kernel_code's lists (PTX or WARP_MODULES): {missing}"
  assert set(kernel_code.PTX_MODULES) <= set(sources)
  stale = sorted(kernel_code.WARP_MODULES - device)
  assert not stale, f"WARP_MODULES without device code: {stale}"


def test_kernel_code_reads_host_modules_only_through_the_listed_names() -> None:
  """Every name a kernel-code module imports from host code is a listed constant (data, hashed by
  value) or a listed host-only object (no data); the tables hold no unused entry."""
  sources = _sources()
  kernel = set(kernel_code.ptx_modules())
  constants = {(m, n) for m, names in kernel_code.HOST_CONSTANTS.items() for n in names}
  host = {(m, n) for m, names in kernel_code.HOST_ONLY.items() for n in names}
  used: set[tuple[str, str]] = set()
  errors = []
  for name in sorted(kernel):
    tree = sources[name]
    for module, imported, alias in _host_imports(name, tree, kernel, set(sources)):
      source = importlib.import_module(module)
      if imported == "*":
        # A whole host module: what kernel code reads of it is host-only, or a listed constant.
        reads = {(module, a) for a in _attributes(tree, alias) if isinstance(getattr(source, a, None), _DATA)}
        errors += [f"{name}: {m}.{a} is data, not in HOST_CONSTANTS" for m, a in sorted(reads - constants)]
        used |= reads | {(module, "*")}
        entry = (module, "*")
      else:
        value = getattr(source, imported)
        entry = (module, imported)
        if isinstance(value, _DATA) and entry not in constants:
          errors.append(f"{name}: {module}.{imported} is data, not in HOST_CONSTANTS")
        if not isinstance(value, _DATA) and entry in constants:
          errors.append(f"{name}: {module}.{imported} is not data (HOST_ONLY, not HOST_CONSTANTS)")
      used.add(entry)
      if entry not in constants | host:
        errors.append(f"{name} imports {module}.{imported} from host code: list it in kernel_code (constant or host-only)")
  assert not errors, "\n".join(errors)
  stale = sorted((constants | host) - used)
  assert not stale, f"kernel_code entries no kernel-code module imports: {stale}"


def test_kernel_version_covers_kernel_code_and_constants_only(monkeypatch: pytest.MonkeyPatch) -> None:
  """The digest holds the kernel-code files and the host constants' values, no host module (the
  routing tables, resolve, upload, the adapters)."""
  names = set(kernel_code.sources())
  assert "raywarp.core.trace.optix.device/programs.py" in names
  assert "raywarp.core.kernels/pixel.py" in names
  host = ("class_sets", "tiers", "resolve", "renderer", "upload", "mujoco", "newton", "launch.py", "pipeline.py")
  assert not [n for n in names if any(h in n for h in host)]
  before = kernel_code.version()
  fields = importlib.import_module("raywarp.settings.fields")
  monkeypatch.setattr(fields, "REFLECTION_SHADOWS", {**fields.REFLECTION_SHADOWS, "new": 3})
  assert kernel_code.version() != before


def test_a_function_edit_changes_its_own_part_only() -> None:
  """`code_units`: a top-level function's edit changes its part alone, a module-level constant's the
  module's rest alone, a comment or docstring nothing."""
  name, data = next((n, d) for n, d in kernel_code._files() if n.endswith("shade/lights/loop.py"))
  parts = code_version.code_units([(name, data)])
  fn = next(u for u in parts if kernel_code.function(u))
  body = ast.parse(data).body
  node = next(n for n in body if isinstance(n, ast.FunctionDef) and f"{name}:{n.name}" == fn)
  lines = data.decode().splitlines(keepends=True)
  indent = " " * (node.body[-1].col_offset)
  edited = "".join([*lines[: node.end_lineno], f"{indent}_unused = 1\n", *lines[node.end_lineno :]]).encode()
  changed = code_version.code_units([(name, edited)])
  assert {u for u in parts if parts[u] != changed[u]} == {fn}
  constant = code_version.code_units([(name, data + b"\nX_NEW = 1\n")])
  assert {u for u in parts if parts[u] != constant[u]} == {f"{name}:"}
  assert code_version.code_units([(name, data + b"\n# a comment\n")]) == parts


def test_optix_programs_generate_code_of_the_kernel_list_only() -> None:
  """Every function and struct in the OptiX programs' generated code (each scene-free tier's
  render and GI programs) is defined in a kernel-code module (or outside raywarp: Warp, MuJoCo
  Warp)."""
  import warp as wp
  from fixtures.optix_codegen import preset_features, scene_flags

  from raywarp.core.build.variants import tiers
  from raywarp.core.trace.optix.device import intrinsics as device
  from raywarp.core.trace.optix.device import programs

  if device.include_dir() is None:
    pytest.skip("the OptiX SDK headers are not installed")
  from warp._src.context import ModuleBuilder

  kernel = set(kernel_code.ptx_modules())
  found: set[str] = set()
  for tier_settings in tiers.TIERS.values():
    for gi in (False, True):
      f = preset_features(tier_settings.replace(**{"gi.enabled": gi}))
      flags = scene_flags()
      modules = [programs.build(f, flags)] + ([programs.build_gi(f, flags)] if programs.gi_programs(f) else [])
      for module in modules:
        options = module.resolve_options(wp.config, block_dim=256)
        builder = ModuleBuilder(module, options | {"output_arch": None, "block_dim": 256})
        builder.codegen("cuda")
        found |= {getattr(getattr(fn, "func", None), "__module__", "") or "" for fn in builder.functions}
        found |= {getattr(getattr(st, "cls", None), "__module__", "") or "" for st in builder.structs}
  ours = {m for m in found if m.startswith("raywarp")}
  assert "raywarp.core.trace.optix.device.programs" in ours
  assert not ours - kernel, f"generated from modules outside kernel_code: {sorted(ours - kernel)}"


_RECORD = """
import json, pickle, sys
from tools.prebuilt import reach
reach.start()  # before raywarp is imported: a fill process's
import warp as wp
from warp._src.context import ModuleBuilder
from raywarp.core.trace.optix.device import programs
wp.init()
f, flags = pickle.loads(bytes.fromhex(sys.argv[1]))
module = programs.build(f, flags)
options = module.resolve_options(wp.config, block_dim=256) | {"output_arch": None, "block_dim": 256}
ModuleBuilder(module, options).codegen("cuda")
print(json.dumps(sorted(reach.of([module]))))
"""


def _reach(f: object, flags: object) -> frozenset[str]:
  """The reach of OptiX build (`f`, `flags`) recorded as a fill process records it (a fresh
  process, `reach.start` first; codegen without NVRTC)."""
  import json
  import pickle
  import subprocess
  import sys

  run = subprocess.run(
    [sys.executable, "-c", _RECORD, pickle.dumps((f, flags)).hex()], capture_output=True, text=True, cwd=REPO, check=True
  )
  return frozenset(json.loads(run.stdout.splitlines()[-1]))


def test_an_edit_changes_only_the_builds_whose_generation_used_it(monkeypatch: pytest.MonkeyPatch) -> None:
  """Per-build kernel code (`kernel_code.digest` of a recorded reach): editing a function only the
  translucent-surface class's code uses changes the code of the builds with that class and keeps
  every other build's; a function both use changes both; a module's other code changes both."""
  import dataclasses

  from fixtures.optix_codegen import preset_features, scene_flags

  from raywarp.core.build.variants import tiers
  from raywarp.core.trace.optix.device import intrinsics as device

  if device.include_dir() is None:
    pytest.skip("the OptiX SDK headers are not installed")
  plain = preset_features(tiers.TIERS["fast"])
  surface = dataclasses.replace(plain, transparency=True, translucent_shadows=True)
  flags = scene_flags()
  reach = {"plain": _reach(plain, flags), "surface": _reach(surface, flags)}
  only = sorted(reach["surface"] - reach["plain"])
  both = sorted(reach["surface"] & reach["plain"] - {u for u in reach["surface"] if "/types.py" in u})
  assert only and both
  table = dict(kernel_code.units())

  def codes(edited: str) -> dict[str, bool]:
    """Per build, whether its code changes when unit `edited` changes."""
    before = {k: kernel_code.digest(r) for k, r in reach.items()}
    monkeypatch.setattr(kernel_code, "units", lambda: table | {edited: "edited"})
    after = {k: kernel_code.digest(r) for k, r in reach.items()}
    monkeypatch.setattr(kernel_code, "units", lambda: table)
    return {k: before[k] != after[k] for k in reach}

  assert codes(only[0]) == {"plain": False, "surface": True}, only[0]
  assert codes(both[0]) == {"plain": True, "surface": True}
  assert codes(next(u for u in table if not kernel_code.function(u))) == {"plain": True, "surface": True}
  untouched = next(u for u in table if kernel_code.function(u) and u not in reach["surface"] | reach["plain"])
  assert codes(untouched) == {"plain": False, "surface": False}
