"""The public benchmark (`benchmarks/run.py`) and its publish layout
(`tools/release/benchmark_scenes.py`): a file:// mirror of the remote, the viewer's demo scene as
the showcase file, a tiny CPU run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks import run
from raywarp.viewer.scenes import demo
from tools.release import benchmark_scenes


@pytest.fixture
def mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """A remote laid out by the publish tool, the cache directory empty."""
  src = tmp_path / "src"
  src.mkdir()
  demo().save(src / "showcase.wpbr")
  shutil.copy(src / "showcase.wpbr", src / "sponza.wpbr")
  (src / "franka.tar").write_bytes(b"")
  view = [[2.0, -3.0, 1.5], [0.0, 0.0, 0.5], 50.0]
  (src / run.CAMERAS).write_text(json.dumps(dict.fromkeys(run.FILES, view)))
  remote = tmp_path / "remote"
  dest = benchmark_scenes.layout(src, remote)
  assert dest == remote / "benchmarks" / run.key() and (dest / "showcase.wpbr.xz").is_file()
  monkeypatch.setenv(run.URL_ENV, f"file://{remote}")
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path / "cache"))
  return tmp_path


def test_scene_download_render_and_png(mirror: Path) -> None:
  path, view = run.scene_file("showcase")
  assert path == mirror / "cache" / "benchmarks" / run.key() / "showcase.wpbr" and view[2] == 50.0
  png = mirror / "out.png"
  result = run.measure(path, view, worlds=2, res=16, settings="fast", frames=2, device="cpu", png=png)
  assert result.ms_per_frame > 0 and result.images_per_s > 0 and result.first_frame_s > 0
  assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_layout_refuses_missing_files(tmp_path: Path) -> None:
  with pytest.raises(SystemExit, match="missing scene files"):
    benchmark_scenes.layout(tmp_path, tmp_path / "out")
