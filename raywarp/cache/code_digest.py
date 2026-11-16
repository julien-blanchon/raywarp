"""Code digests of Python sources by syntax tree (`code_version`'s): a comment, docstring or
formatting edit keeps them. The standard library only, and no raywarp import: the wheel build loads
this file alone to ship the digests of the package's own sources (`tools/release/code_memo.py`),
so a first start reads them instead of parsing every source (`code_version._Memo`).
"""

from __future__ import annotations

import ast
import hashlib
import sys
from typing import Final

# Names `python_digest`'s normalization (bumped when it changes).
NORMALIZATION: Final = b"ast-1"
# Docstring owners: their first statement, a string constant, is dropped by `python_digest`.
_DOCUMENTED: Final = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
# The memo kinds: "code-digests" (`python_digest` per file), "code-units" (`python_units`).
DIGESTS: Final = "code-digests"
UNITS: Final = "code-units"


def memo_name(kind: str) -> str:
  """The memo file name of `kind` for this Python version (its parser's syntax trees)."""
  return f"{kind}-{NORMALIZATION.decode()}-py{sys.version_info[0]}{sys.version_info[1]}.json"


def raw_digest(data: bytes) -> str:
  """The memo key of a file's bytes."""
  return hashlib.blake2b(data, digest_size=16).hexdigest()


def python_digest(source: bytes) -> str:
  """Digest of a Python source's syntax tree without docstrings: comments, docstrings and
  formatting do not change it."""
  return _tree_digest(_stripped(source))


def python_units(source: bytes) -> dict[str, str]:
  """Digests of a Python source's parts (`python_digest`'s normalization): each top-level
  function by name (its decorators in it; a name defined twice, both definitions), and the rest
  of the module (imports, constants, classes, module-level code) under "" ."""
  parts: dict[str, list[ast.stmt]] = {"": []}
  for node in _stripped(source).body:
    name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""
    parts.setdefault(name, []).append(node)
  return {name: _tree_digest(ast.Module(body=nodes, type_ignores=[])) for name, nodes in parts.items()}


def _stripped(source: bytes) -> ast.Module:
  """The syntax tree of a Python source without docstrings."""
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if isinstance(node, _DOCUMENTED) and node.body:
      first = node.body[0]
      if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        node.body = node.body[1:] or [ast.Pass()]
  return tree


def _tree_digest(tree: ast.AST) -> str:
  """Digest of a syntax tree's text (`_dump`)."""
  out: list[str] = []
  _dump(tree, out)
  return hashlib.blake2b(NORMALIZATION + "".join(out).encode(), digest_size=16).hexdigest()


def _dump(node: object, out: list[str]) -> None:
  """Appends a syntax tree's text to `out`: node types, their fields but the unset ones (None,
  empty lists: Python versions differ in the optional fields they add and `ast.dump` shows) and
  constants by repr (1, 1.0 and True differ)."""
  if isinstance(node, ast.AST):
    out.append(f"{type(node).__name__}(")
    for name in node._fields:
      value = getattr(node, name, None)
      if value is not None and value != []:
        out.append(f"{name}=")
        _dump(value, out)
        out.append(",")
    out.append(")")
  elif isinstance(node, list):
    out.append("[")
    for item in node:
      _dump(item, out)
      out.append(",")
    out.append("]")
  else:
    out.append(repr(node))
