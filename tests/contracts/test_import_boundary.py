"""Import boundary: the core (and the host scene) import only warp and numpy, no scene format."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO

ROOT = REPO / "raywarp"
FORBIDDEN = ("mujoco", "mujoco_warp", "pxr", "newton")
# Packages (and single modules) whose every module must stay format-free
# (`formats/usd/prim_paths.py`: the Newton and Isaac Lab adapters import it without the usd extra).
CORE_PACKAGES = ("_base", "cache", "settings", "core", "scene", "formats/usd/prim_paths.py")


def _files(package: str) -> list[Path]:
  """The `.py` files of `package` (a directory or one module) under `ROOT`."""
  path = ROOT / package
  return [path] if path.is_file() else sorted(path.rglob("*.py"))


def _modules() -> list[str]:
  names = []
  for package in CORE_PACKAGES:
    for path in _files(package):
      rel = path.relative_to(ROOT.parent).with_suffix("")
      parts = list(rel.parts)
      if parts[-1] == "__init__":
        parts = parts[:-1]
      names.append(".".join(parts))
  return names


def _top(name: str) -> str:
  return name.split(".")[0]


@pytest.mark.parametrize("package", CORE_PACKAGES)
def test_no_forbidden_import_statements(package):
  """Static check: no import of a scene format anywhere under the package (lazy ones included)."""
  offenders = []
  for path in _files(package):
    for node in ast.walk(ast.parse(path.read_text())):
      if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
      elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        names = [node.module]
      else:
        continue
      offenders += [f"{path.relative_to(ROOT)}: {n}" for n in names if _top(n) in FORBIDDEN]
  assert not offenders, offenders


def test_importing_the_core_loads_no_scene_format():
  """In a fresh interpreter, importing raywarp and every core / scene module loads none of them."""
  code = (
    "import importlib, json, sys\n"
    f"for name in {_modules()!r}:\n"
    "  importlib.import_module(name)\n"
    "import raywarp\n"
    f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {list(FORBIDDEN)!r})))\n"
  )
  out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT.parent)
  loaded = json.loads(out.stdout.strip().splitlines()[-1])
  assert loaded == [], loaded
