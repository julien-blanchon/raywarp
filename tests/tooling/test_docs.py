"""The docs cannot rot: every ```python block of docs/*.md and README.md runs, on the CPU device.

The blocks of one page run in order in one namespace (a page reads like a notebook), in a
temporary working directory. A block whose first line is a marker comment is skipped unless its
requirements are met:

  # docs: skip (why)        never run (a fragment, a placeholder path, a display)
  # docs: requires cuda     needs a CUDA device (graph capture, OptiX)
  # docs: requires torch    needs the named importable modules (several: space separated)
  # docs: requires assets   needs downloaded assets (internal/tools/fetch_assets.py): never run here

A page skipped block must not define what a later runnable block uses. The generated settings
regions (tools/settings_table.py) must be up to date.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import settings_table
import warp as wp
from conftest import REPO, library_defaults

ROOT = REPO
PAGES = [*sorted((ROOT / "docs").glob("*.md")), ROOT / "README.md"]
_FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
_MARKER = re.compile(r"^# docs: (skip(?: \(.+\))?|requires ([a-z_ ]+))$")


def blocks(page: Path) -> list[tuple[int, str]]:
  """The page's python blocks: (line of the block's first code line, code)."""
  text = page.read_text()
  return [(text.count("\n", 0, m.start(1)) + 1, m.group(1)) for m in _FENCE.finditer(text)]


def missing(code: str) -> str | None:
  """Why the block cannot run here (None: it runs)."""
  first = code.split("\n", 1)[0].strip()
  m = _MARKER.match(first)
  if m is None:
    return None
  if m.group(1).startswith("skip"):
    return "skip"
  for need in m.group(2).split():
    if need == "assets":
      return "assets"
    if need == "cuda":
      if not wp.is_cuda_available():
        return "cuda"
    elif importlib.util.find_spec(need) is None:
      return need
  return None


def test_markers_are_known():
  """A typo in a marker would silently run (or skip) a block."""
  for page in PAGES:
    for line, code in blocks(page):
      first = code.split("\n", 1)[0].strip()
      if first.startswith("# docs:"):
        assert _MARKER.match(first), f"{page.name}:{line}: unknown marker {first!r}"


# Pages whose snippets assert the library's default settings: run under them (the others render with
# the test session's GI default, off: a CPU bake at the library's GI grid took 213 s on quickstart).
_DEFAULT_PAGES = frozenset({"settings.md"})


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_snippets_run(page, tmp_path, monkeypatch):
  """Runs the page's python blocks in order on the CPU device."""
  found = blocks(page)
  if not found:
    pytest.skip("no python blocks")
  monkeypatch.chdir(tmp_path)
  if page.name in _DEFAULT_PAGES:
    library_defaults(monkeypatch)  # they assert the library's defaults, not the test session's
  # Warp inspects function source; Markdown is not a Python module. Preserve page line numbers
  # in a temporary real module so executable kernel examples are checked too.
  source = tmp_path / f"docs_{page.stem}.py"
  lines = ["\n"] * len(page.read_text().splitlines())
  for line, code in found:
    if missing(code) is None:
      body = code.splitlines(keepends=True)
      lines[line - 1 : line - 1 + len(body)] = body
  source.write_text("".join(lines))
  module = ModuleType(f"docs_{page.stem}")
  module.__file__ = str(source)
  monkeypatch.setitem(sys.modules, module.__name__, module)
  namespace: dict[str, object] = vars(module)
  ran = 0
  with wp.ScopedDevice("cpu"):
    for line, code in found:
      if missing(code) is not None:
        continue
      # Traceback line numbers still match the original page.
      exec(compile("\n" * (line - 1) + code, str(source), "exec"), namespace)
      ran += 1
  if ran == 0:
    pytest.skip("every block needs something missing here")


def test_generated_settings_regions_are_current(monkeypatch):
  """docs/settings.md and README.md list the fields as their metadata says (D1); docs/scenes.md the
  OpenPBR coverage and the MuJoCo material mapping as the code has them."""
  library_defaults(monkeypatch)
  assert settings_table.stale() == []
