"""Import cycles: raywarp's module import graph (static AST pass, raywarp modules only).

* Module-level imports (run at import time; `if TYPE_CHECKING:` blocks excluded) form no cycle: a
  cycle there works only through import order (each module already in `sys.modules` when the next
  asks for it), and an innocent new import breaks it at import time.
* With the deferred imports (inside functions) added, the cycles are exactly `ALLOWED`: a deferred
  import that only works around a cycle hides real coupling. A new cycle fails; a fixed one must
  leave the list. None of them lies inside `core` (W20-CYCLES removed the last one there).
* The top-level packages form layers (`LAYERS`): no import reaches a higher layer, but the
  deferred ones of `UPWARD`.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from conftest import REPO

ROOT: Final = REPO / "raywarp"

# The cycles that remain through deferred imports, by their modules (under raywarp.), with why.
ALLOWED: Final = {
  # C4: the host scene evaluates "flux as rendered" through the upload's light derivation. Kept
  # until `pipeline/upload/lights.py` is reworked (the light derivation then moves into `scene/`).
  "C4": frozenset({"core.pipeline.upload.lights", "scene.model.lights", "scene.lighting.photometry"}),
  # C5: `Scene.with_lighting` (public API) builds through `scene.lighting.rig` -> `SceneBuilder`.
  # Kept by decision: one deferred import is cheaper than a worse API.
  "C5": frozenset({"scene.build.builder", "scene.lighting.rig", "scene.model.scene"}),
}

type Graph = dict[str, set[str]]


def _modules() -> dict[str, tuple[Path, bool]]:
  """Dotted module name -> (file, is a package `__init__`)."""
  out = {}
  for path in sorted(ROOT.rglob("*.py")):
    if "_native" in path.parts and path.name != "__init__.py":
      continue  # the native build hook's generated bindings (tools/native/hook.py), not source
    parts = list(path.relative_to(ROOT.parent).with_suffix("").parts)
    package = parts[-1] == "__init__"
    out[".".join(parts[:-1] if package else parts)] = (path, package)
  return out


def _type_checking(node: ast.If) -> bool:
  test = node.test
  return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
    isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
  )


def _imports(body: Sequence[ast.AST], *, deferred: bool) -> list[ast.Import | ast.ImportFrom]:
  """The import statements of `body` run at import time (also inside classes, `try`, `if`), plus
  those inside functions with `deferred`; never an `if TYPE_CHECKING:` block's."""
  out: list[ast.Import | ast.ImportFrom] = []
  for node in body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
      out.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      if deferred:
        out += _imports(node.body, deferred=True)
    elif isinstance(node, ast.If) and _type_checking(node):
      out += _imports(node.orelse, deferred=deferred)
    else:
      for field in ("body", "orelse", "finalbody", "handlers"):
        sub = getattr(node, field, None)
        if isinstance(sub, list):
          out += _imports([n for n in sub if isinstance(n, ast.AST)], deferred=deferred)
  return out


def _targets(node: ast.Import | ast.ImportFrom, name: str, package: bool, known: dict[str, tuple[Path, bool]]) -> list[str]:
  """The raywarp modules an import statement of module `name` loads (`from p import m`: the
  submodule p.m when m is one, else p)."""
  if isinstance(node, ast.Import):
    return [a.name for a in node.names]
  base = name if package else name.rpartition(".")[0]
  for _ in range(node.level - 1):
    base = base.rpartition(".")[0]
  module = (f"{base}.{node.module}" if node.module else base) if node.level else node.module or ""
  return [f"{module}.{a.name}" if f"{module}.{a.name}" in known else module for a in node.names]


def _graph(*, deferred: bool) -> Graph:
  known = _modules()
  graph: Graph = {m: set() for m in known}
  for name, (path, package) in known.items():
    for node in _imports(ast.parse(path.read_text()).body, deferred=deferred):
      graph[name] |= {t for t in _targets(node, name, package, known) if t in known and t != name}
  return graph


def _cycles(graph: Graph) -> list[frozenset[str]]:
  """The strongly connected components of more than one module (Tarjan, iterative)."""
  index: dict[str, int] = {}
  low: dict[str, int] = {}
  stack: list[str] = []
  on_stack: set[str] = set()
  out: list[frozenset[str]] = []
  for root in graph:
    if root in index:
      continue
    work = [(root, iter(sorted(graph[root])))]
    index[root] = low[root] = len(index)
    stack.append(root)
    on_stack.add(root)
    while work:
      v, edges = work[-1]
      w = next(edges, None)
      if w is not None:
        if w not in index:
          index[w] = low[w] = len(index)
          stack.append(w)
          on_stack.add(w)
          work.append((w, iter(sorted(graph[w]))))
        elif w in on_stack:
          low[v] = min(low[v], index[w])
        continue
      work.pop()
      if work:
        low[work[-1][0]] = min(low[work[-1][0]], low[v])
      if low[v] == index[v]:
        component = set()
        while True:
          w = stack.pop()
          on_stack.discard(w)
          component.add(w)
          if w == v:
            break
        if len(component) > 1:
          out.append(frozenset(c.removeprefix("raywarp.") for c in component))
  return out


def test_no_module_level_import_cycle() -> None:
  cycles = _cycles(_graph(deferred=False))
  assert not cycles, [sorted(c) for c in cycles]


def test_no_allowed_cycle_inside_core() -> None:
  inside = [name for name, c in ALLOWED.items() if all(m.startswith("core.") for m in c)]
  assert not inside, f"cycles inside core may not be allowed: {inside}"


def test_deferred_import_cycles_are_the_allowed_ones() -> None:
  cycles = set(_cycles(_graph(deferred=True)))
  allowed = set(ALLOWED.values())
  assert not cycles - allowed, f"new import cycles: {[sorted(c) for c in cycles - allowed]}"
  fixed = [name for name, c in ALLOWED.items() if c not in cycles]
  assert not fixed, f"no longer cycles (remove them from ALLOWED): {fixed}"


# The top-level layers (under raywarp.): a package imports its own layer's and lower ones only;
# the scene formats, then the simulator and framework integrations (those reading a format), then
# the app sit on the library. Inside `core`, the device band (kernel_code.PTX_PACKAGES: trace.rays /
# accel / optix.device, shade, post's display passes, kernels) reads the host band (pipeline,
# build, outputs, the trace backends, post.aa / sensor) only through kernel_code.HOST_ONLY /
# HOST_CONSTANTS names (tests/contracts/test_kernel_code.py).
LAYERS: Final = {
  "_base": 0,
  "_native": 0,
  "cache": 0,
  "data": 0,
  "settings": 1,
  "scene": 2,
  "core": 3,
  "randomize": 4,
  "kernels": 4,
  "video": 4,
  "formats": 4,
  "integrations": 5,
  "viewer": 6,
  "cli": 7,
}
# The deferred imports allowed upwards, (importer, imported) under raywarp.: C4 above.
UPWARD: Final = frozenset(
  {
    ("scene.lighting.photometry", "core.shade.lights.punctual"),
    ("scene.lighting.photometry", "core.pipeline.upload.lights"),
  }
)


def _upward(graph: Graph) -> set[tuple[str, str]]:
  """The edges of `graph` from a top-level package to a higher layer's (the root `raywarp`
  module, which re-exports the public names, excluded)."""
  out = set()
  for importer, imported in ((a, b) for a, targets in graph.items() for b in targets):
    if importer.count(".") and imported.count("."):
      low, high = importer.split(".")[1], imported.split(".")[1]
      if LAYERS[low] < LAYERS[high]:
        out.add((importer.removeprefix("raywarp."), imported.removeprefix("raywarp.")))
  return out


def test_layers() -> None:
  assert not _upward(_graph(deferred=False))
  assert _upward(_graph(deferred=True)) == UPWARD
