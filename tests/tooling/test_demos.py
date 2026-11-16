"""The demo gallery (examples/demos/, docs/gallery.md): every demo is listed, has the common command
line, and its `main` records a clip on the CPU at a tiny size (the benchmark scenes replaced by
fixtures of tests/data/scenes/; network assets only when already cached)."""

from __future__ import annotations

import importlib.util
import inspect
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import REPO

DEMOS = REPO / "examples" / "demos"
GALLERY = REPO / "docs" / "gallery.md"
COMMON = ("device", "worlds", "size", "frames", "out")
TINY = {"device": "cpu", "worlds": 2, "size": "96x64", "frames": 3}
# demo -> (modules it needs beyond raywarp[mujoco], its tiny-size arguments beyond TINY).
CASES: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
  "newton_usd_robots": (("newton", "pxr"), {}),
  "newton_falling_objects": (("newton", "pxr"), {}),
  "newton_headless_viewer": (("newton", "pxr", "pyglet"), {"worlds": 1}),
  "parallel_worlds": ((), {"scene": "randomize"}),
  "domain_randomisation": ((), {"scene": "randomize", "every": 2}),
  "material_studio": (("pxr",), {"worlds": 1}),
  "daylight_timelapse": (("pxr",), {"scene": "shadows"}),
  "scene_tour": (("pxr",), {"worlds": 1, "scenes": "cornell_rect,geoms"}),
  "sponza_flythrough": (("pxr",), {"worlds": 1, "scene": "open_plan"}),
  "ground_truth": ((), {"worlds": 1, "size": "192x96", "scene": "randomize"}),
  "camera_models": ((), {"worlds": 1, "scene": "geoms"}),
  "sensor_effects": ((), {"scene": "geoms", "frames": 4}),
  "batch_mask": ((), {"worlds": 4, "scene": "randomize", "frames": 6}),
  "robot_multicam": ((), {"worlds": 1, "scene": "randomize"}),
  "gltf_asset": (("pygltflib", "pxr"), {"worlds": 1}),
  "compare_renderers": ((), {"worlds": 1, "size": "192x64", "scene": "randomize"}),
  "before_after": ((), {"worlds": 1, "size": "128x64", "setting": "contact_ao", "scene": "randomize"}),
}


def _main(name: str):
  return runpy.run_path(str(DEMOS / f"{name}.py"))["main"]


def test_every_demo_is_tested_and_in_the_gallery():
  """A new demo needs its row in CASES and its entry (command) in docs/gallery.md."""
  scripts = {p.stem for p in DEMOS.glob("*.py") if not p.name.startswith("_")}
  assert scripts == set(CASES)
  gallery = GALLERY.read_text()
  missing = [s for s in scripts if f"examples/demos/{s}.py" not in gallery]
  assert not missing, f"not in docs/gallery.md: {missing}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_demo_has_the_common_command_line(name, monkeypatch):
  """`main` takes the shared options (keyword arguments with defaults) and `--help` lists them."""
  monkeypatch.syspath_prepend(str(DEMOS))
  needs, _ = CASES[name]
  for module in needs:
    if importlib.util.find_spec(module) is None:
      pytest.skip(f"needs {module}")
  params = inspect.signature(_main(name)).parameters
  assert all(p in params for p in COMMON)
  assert all(p.default is not inspect.Parameter.empty for p in params.values())
  assert params["out"].default == f"{name}.mp4" or name in ("compare_renderers", "before_after")


@pytest.mark.slow  # compiles the CPU kernels of its settings on a cold Warp cache (30-90 s each)
@pytest.mark.parametrize("name", sorted(CASES))
def test_demo_records_a_clip(name, tmp_path, monkeypatch):
  """The demo's `main` on the CPU at its tiny size writes a decodable MP4."""
  if shutil.which("ffmpeg") is None:
    pytest.skip("needs the ffmpeg CLI")
  needs, extra = CASES[name]
  for module in needs:
    if importlib.util.find_spec(module) is None:
      pytest.skip(f"needs {module}")
  monkeypatch.syspath_prepend(str(DEMOS))
  if name == "gltf_asset":
    from raywarp.cache.paths import cache_dir

    if not ((cache_dir() or Path.home() / ".cache" / "raywarp") / "gallery" / "ToyCar.glb").is_file():
      pytest.skip("downloads a Khronos sample asset (not cached)")
  out = tmp_path / f"{name}.mp4"
  _main(name)(**{**TINY, **extra, "out": str(out)})
  probe = subprocess.run(
    ["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
    capture_output=True,
    text=True,
    check=True,
  )
  assert int(probe.stdout.strip()) == extra.get("frames", TINY["frames"])
