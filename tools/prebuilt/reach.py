"""The kernel code a build's PTX generation uses (`kernel_code.Reach`), recorded in a fill process.

A fill process records every Python function it starts (`start`, before raywarp is imported:
`sys.monitoring`, each code object reported once) and, after a build's code generation, the
device functions Warp reached (`reached`: their bodies are parsed, never run). The build's reach
is the kernel-code functions among both (`of`): the factories and helpers that chose its code, the
device functions it compiled, the import-time code every build shares. Each fill process makes
one build (`fill._compile`: its caches start empty), so a cached factory result never hides the
code that made it. The rest of each kernel-code module (constants, classes, imports) is in every
digest (`kernel_code.digest`).
"""

from __future__ import annotations

import ast
import bisect
import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
  from types import CodeType

  import warp as wp
  from warp._src.codegen import Adjoint
  from warp._src.context import Function

  from raywarp.core.build.kernel_cache.kernel_code import Reach

# The code objects this process started: (file, first line).
_STARTED: Final[set[tuple[str, int]]] = set()
_TOOL: Final = sys.monitoring.COVERAGE_ID


def start() -> None:
  """Records every Python function this process starts from now on (once per code object).

  Raises:
    RuntimeError: another tool holds the coverage slot of `sys.monitoring` (a coverage run).
  """
  if sys.monitoring.get_tool(_TOOL) is not None:
    raise RuntimeError(f"sys.monitoring tool {_TOOL} is {sys.monitoring.get_tool(_TOOL)!r}: the fill records its reach there")
  sys.monitoring.use_tool_id(_TOOL, "raywarp fill reach")
  sys.monitoring.register_callback(_TOOL, sys.monitoring.events.PY_START, _started)
  sys.monitoring.set_events(_TOOL, sys.monitoring.events.PY_START)


def _started(code: CodeType, _offset: int) -> object:
  """A started code object, recorded; its later starts not reported."""
  _STARTED.add((code.co_filename, code.co_firstlineno))
  return sys.monitoring.DISABLE


def of(modules: list[wp.Module]) -> Reach:
  """The reach of a build whose programs are `modules` (generated: their code built), in this
  process (`start`)."""
  lines = set(_STARTED)
  for module in modules:
    adjoints, functions = walk(module)
    for fn in [*(a.func for a in adjoints), *(f.func for f in functions)]:
      code = getattr(fn, "__code__", None)
      if code is not None:
        lines.add((code.co_filename, code.co_firstlineno))
  table = _functions()
  out = set()
  for file, line in lines:
    found = table.get(str(Path(file).resolve()))
    if found is not None:
      name, starts, ends = found
      i = bisect.bisect_right(starts, line) - 1
      if i >= 0 and line <= ends[i][1]:
        out.add(f"{name}:{ends[i][0]}")
  return frozenset(out)


def walk(module: wp.Module) -> tuple[list[Adjoint], list[Function]]:
  """The code the kernels of `module` reach (after its code was built or hashed): the adjoints of
  its kernels and of every user function they reach (references and static expressions,
  transitively, each overload once), and those functions."""
  from warp._src.context import Function  # Warp's function type (reached by references)

  seen: set[int] = set()
  adjoints: list[Adjoint] = []
  functions: list[Function] = []
  todo = [o.adj for k in module.kernels.values() for o in (k.overloads.values() if k.is_generic else (k,))]
  while todo:
    adj = todo.pop()
    if id(adj) in seen:
      continue
    seen.add(id(adj))
    adjoints.append(adj)
    refs = [*adj.get_references()[2], *(v for v in adj.resolved_static_expressions.values() if isinstance(v, Function))]
    user = [f for f in refs if not f.is_builtin()]  # a built-in's code is Warp's
    functions += user
    todo += [o.adj for f in user for o in (f.user_overloads | f.user_templates).values() if o.adj is not None]
  return adjoints, functions


@functools.cache
def _functions() -> dict[str, tuple[str, list[int], list[tuple[str, int]]]]:
  """Per kernel-code file path: its `kernel_code.units` name, the first lines of its top-level
  functions (decorators included) in order, and each one's (name, last line)."""
  from raywarp.core.build.kernel_cache import kernel_code

  out = {}
  for name, path in kernel_code.sources().items():
    if path.suffix != ".py":
      continue
    nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    spans = sorted((min([n.lineno, *(d.lineno for d in n.decorator_list)]), n.name, n.end_lineno or n.lineno) for n in nodes)
    out[str(path.resolve())] = (name, [s for s, _, _ in spans], [(n, end) for _, n, end in spans])
  return out
