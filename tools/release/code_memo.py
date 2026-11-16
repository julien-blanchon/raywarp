"""The wheel's code-digest memos: setuptools' `build_py` (pyproject.toml
`[tool.setuptools.cmdclass]`) that also writes, into the built `raywarp/cache/code_memo/`, the
syntax-tree digests of every raywarp source it built (`raywarp/cache/code_digest.py`:
`python_digest` and `python_units` per file, keyed by the file's raw digest, one file per memo
kind, named for the building Python). An
installed raywarp then reads them (`code_version._Memo`) instead of parsing its ~4 MB of sources
on a first start (the bake, texture and kernel-code digests: seconds). The memo is a pure cache
keyed by content: an entry whose file changed is never read; a Python of another version parses.

code_digest.py is loaded by its path: the build environment has no Warp, which `import raywarp`
needs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import override

from setuptools.command.build_py import build_py


def _code_digest(path: Path) -> ModuleType:
  """`raywarp/cache/code_digest.py` at `path`, imported alone (standard library only)."""
  spec = importlib.util.spec_from_file_location("_raywarp_code_digest", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def write_memos(package: Path) -> list[Path]:
  """The memos of every source under `package` (a built `raywarp/`) written to its
  `cache/code_memo/`; the files written."""
  digest = _code_digest(package / "cache" / "code_digest.py")
  sources = [p.read_bytes() for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts]
  memos = {
    digest.DIGESTS: {digest.raw_digest(s): digest.python_digest(s) for s in sources},
    digest.UNITS: {digest.raw_digest(s): digest.python_units(s) for s in sources},
  }
  out = package / "cache" / "code_memo"
  out.mkdir(exist_ok=True)
  written = []
  for kind, memo in memos.items():
    path = out / digest.memo_name(kind)
    path.write_text(json.dumps(memo, separators=(",", ":"), sort_keys=True))
    written.append(path)
  return written


class BuildPy(build_py):
  """`build_py`, then the code-digest memos of the built package (`write_memos`)."""

  @override
  def run(self) -> None:
    super().run()
    package = Path(self.build_lib) / "raywarp"
    if (package / "cache" / "code_digest.py").is_file():
      for path in write_memos(package):
        self.announce(f"code memo {path}", level=2)
