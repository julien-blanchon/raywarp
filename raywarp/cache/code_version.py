"""Code digests: what a cache keyed on code (the bake cache, the kernel-code digest, the texture
and emissive-fit caches, the MDL library) compares.

`code_version` is a digest of packages' syntax trees and data files (`code_digest`): a comment or
docstring edit keeps it; the trees are parsed once per file content (`_Memo`, kept in the cache
directory; the wheel ships the digests of raywarp's own sources, so a first start parses none).
`code_units` digests each top-level function apart (`kernel_code`'s per-function reach).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final, final

from raywarp.cache.code_digest import DIGESTS, UNITS, memo_name, python_units, raw_digest
from raywarp.cache.code_digest import python_digest as _python_digest
from raywarp.cache.paths import cache_dir, write_atomic

# The files of a package `code_version` reads: its Python sources and the MDL modules it ships.
CODE_SUFFIXES: Final = (".py", ".mdl")
# The memos of raywarp's own sources a wheel ships (`tools/release/code_memo.py`, absent in a
# source tree).
SHIPPED: Final = Path(__file__).with_name("code_memo")


@final
class _Memo[T: (str, dict[str, str])]:
  """Raw file digest -> `compute(file)`, kept in `<cache dir>/<kind>-<normalization>-<python>.json`
  (`code_digest.memo_name`: the parser of each Python version reads its own syntax; parsing costs
  ~1 s per MB of sources, a warm start only hashes their bytes), over the memo the wheel ships for
  raywarp's own sources (`SHIPPED`, read only: a first start parses none of them)."""

  def __init__(self, kind: str, compute: Callable[[bytes], T]) -> None:
    """The memo of `kind` (values of `compute`'s type) of this Python version: the shipped one,
    then the cache directory's (empty without either)."""
    root = cache_dir()
    name = memo_name(kind)
    self.path = None if root is None else root / name
    self.compute = compute
    self.known: dict[str, T] = {}
    self.added = False
    for path in (SHIPPED / name, self.path):
      if path is not None:
        with contextlib.suppress(OSError, ValueError):
          found = json.loads(path.read_bytes())
          if isinstance(found, dict):
            self.known |= {k: v for k, v in found.items() if isinstance(k, str) and _memo_value(v)}

  def get(self, data: bytes) -> T:
    """`compute(data)`, memoized by the raw digest of `data`."""
    raw = raw_digest(data)
    found = self.known.get(raw)
    if found is None:
      found = self.known[raw] = self.compute(data)
      self.added = True
    return found

  def save(self) -> None:
    """Writes the memo when it grew (`write_atomic`: a failed write warns and only loses the
    memo)."""
    if self.path is None or not self.added:
      return
    text = json.dumps(self.known, separators=(",", ":")).encode()
    write_atomic(self.path, lambda f: f.write(text), "code digest memo")


def _memo_value(value: object) -> bool:
  """Whether a memo file's value is one of `_Memo`'s (a digest, or part name -> digest)."""
  if isinstance(value, dict):
    return all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
  return isinstance(value, str)


def code_digest(files: Iterable[tuple[str, bytes]]) -> str:
  """Digest of code files given as (name, content): Python sources by `_python_digest` (comments,
  docstrings and formatting do not count), other files byte for byte. The order of `files` does
  not matter (sorted by name)."""
  memo = _Memo(DIGESTS, _python_digest)
  digest = hashlib.blake2b(digest_size=12)
  for name, data in sorted(files):
    part = memo.get(data) if name.endswith(".py") else hashlib.blake2b(data, digest_size=16).hexdigest()
    digest.update(f"{name}\0{part}\0".encode())
  memo.save()
  return digest.hexdigest()


def code_files(*packages: str) -> list[tuple[str, bytes]]:
  """(`<package>/<relative path>`, content) of every `CODE_SUFFIXES` file of `packages` (dotted
  package names, e.g. "raywarp.scene").

  Raises:
    ModuleNotFoundError: for a name that is not an importable package.
  """
  out: list[tuple[str, bytes]] = []
  for package in sorted(packages):
    spec = importlib.util.find_spec(package)
    roots = None if spec is None else spec.submodule_search_locations
    if not roots:
      raise ModuleNotFoundError(f"{package} is not a package")
    for root in roots:
      for file in sorted(Path(root).rglob("*")):
        if file.suffix in CODE_SUFFIXES and "__pycache__" not in file.parts and file.is_file():
          out.append((f"{package}/{file.relative_to(root).as_posix()}", file.read_bytes()))
  return out


def code_units(files: Iterable[tuple[str, bytes]]) -> dict[str, str]:
  """Digests of the parts of code files given as (name, content): each Python source's top-level
  functions as "<name>:<function>" and the rest of it as "<name>:" (`python_units`), any other
  file whole under its name."""
  memo = _Memo(UNITS, python_units)
  out: dict[str, str] = {}
  for name, data in files:
    if name.endswith(".py"):
      out |= {f"{name}:{part}": digest for part, digest in memo.get(data).items()}
    else:
      out[name] = hashlib.blake2b(data, digest_size=16).hexdigest()
  memo.save()
  return out


@functools.cache
def code_version(*packages: str) -> str:
  """`code_digest` of `packages`' `code_files`: changes with their code, not with their comments,
  docstrings or formatting."""
  return code_digest(code_files(*packages))
