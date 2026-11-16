"""Every script of examples/ runs, on the CPU device at a tiny size (its `main`)."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest
from conftest import REPO

EXAMPLES = REPO / "examples"
# script -> (the modules it needs beyond raywarp[mujoco], its tiny-size arguments).
CASES: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
  "mujoco_render.py": ((), {"nworld": 2, "size": 24}),
  "outputs.py": ((), {"nworld": 2, "size": 24}),
  "domain_randomisation.py": ((), {"nworld": 3, "size": 24, "steps": 2}),
  "usd_scene.py": (("pxr",), {"size": 24, "stage": None}),
  "newton_sensor.py": (("newton",), {"nworld": 2, "size": 24}),
}


def test_every_example_is_tested() -> None:
  """A new script needs its row in `CASES` (and in examples/README.md)."""
  scripts = {p.name for p in EXAMPLES.glob("*.py")}
  assert scripts == set(CASES)
  table = (EXAMPLES / "README.md").read_text()
  assert all(f"`{name}`" in table for name in scripts)


@pytest.mark.parametrize("name", sorted(CASES))
def test_example_runs(name: str) -> None:
  """The script's `main` on the CPU at its tiny size."""
  needs, args = CASES[name]
  for module in needs:
    if importlib.util.find_spec(module) is None:
      pytest.skip(f"needs {module}")
  main = runpy.run_path(str(Path(EXAMPLES / name)))["main"]
  main(device="cpu", **args)
