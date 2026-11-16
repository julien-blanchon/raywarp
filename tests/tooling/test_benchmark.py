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
  demo().save(src / "showcase.rayw")
  shutil.copy(src / "showcase.rayw", src / "sponza.rayw")
  (src / "franka.tar").write_bytes(b"")
  view = [[2.0, -3.0, 1.5], [0.0, 0.0, 0.5], 50.0]
  (src / run.CAMERAS).write_text(json.dumps(dict.fromkeys(run.FILES, view)))
  remote = tmp_path / "remote"
  dest = benchmark_scenes.layout(src, remote)
  assert dest == remote / "benchmarks" / run.key() and (dest / "showcase.rayw.xz").is_file()
  monkeypatch.setenv(run.URL_ENV, f"file://{remote}")
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path / "cache"))
  return tmp_path


def test_scene_download_render_and_png(mirror: Path) -> None:
  path, view = run.scene_file("showcase")
  assert path == mirror / "cache" / "benchmarks" / run.key() / "showcase.rayw" and view[2] == 50.0
  png = mirror / "out.png"
  result = run.measure(path, view, worlds=2, res=16, settings="fast", frames=2, device="cpu", png=png)
  assert result.ms_per_frame > 0 and result.images_per_s > 0 and result.first_frame_s > 0
  assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_layout_refuses_missing_files(tmp_path: Path) -> None:
  with pytest.raises(SystemExit, match="missing scene files"):
    benchmark_scenes.layout(tmp_path, tmp_path / "out")


def test_compare_table_marks_errors_and_shows_flip_of_same_scene_renderers(tmp_path: Path) -> None:
  from benchmarks import compare_table

  def row(renderer: str, mode: str, worlds: int, status: str = "ok", **extra: object) -> dict:
    base = {
      "renderer": renderer,
      "mode": mode,
      "scene": "franka",
      "res": 512,
      "worlds": worlds,
      "outputs": "rgb",
      "status": status,
    }
    return {
      **base,
      **({"images_per_s": 2000.0 * worlds, "ms_per_frame": 0.5, "peak_mem_mb": 2048.0} if status == "ok" else {}),
      **extra,
    }

  data = {
    "meta": {"gpu": ["RTX"]},
    "renderers": {"raywarp": {"label": "Raywarp"}, "maniskill": {"label": "ManiSkill"}},
    "rows": [
      row("raywarp", "fast", 512, first_frame_s=1.25),
      row("raywarp", "fast", 1),
      row("maniskill", "minimal", 512),
      row("maniskill", "minimal", 1, status="error", error="OOM"),
    ],
    "quality": {"flip": {"raywarp/fast": 0.0712, "maniskill/minimal": 0.3}},
  }
  path = tmp_path / "compare.json"
  path.write_text(json.dumps(data))
  text = compare_table.tables(json.loads(path.read_text()))
  assert "| Raywarp fast | 2.0k | 1.02M |" in text and "| ManiSkill minimal | err | 1.02M |" in text
  assert "| Raywarp fast | 0.5 | 1.02M | - | 1.2 s | 2.0 GB | 0.071 |" in text  # FLIP: same MJCF, same lights
  assert "| ManiSkill minimal | 0.5 | 1.02M | - | - | 2.0 GB | - |" in text  # rebuilt scene: no FLIP
  assert [b["label"] for b in compare_table.bars(data)] == ["Raywarp fast", "Raywarp fast", "ManiSkill minimal"]


def test_scenes_bucket_is_buckets_env_assets_bucket() -> None:
  """The benchmark scenes live in the public assets bucket buckets.env names, one directory per
  .rayw format and USD converter version."""
  from tools.prebuilt import build_cache

  assert build_cache.buckets()["ASSETS_BUCKET"] == run.ASSETS_BUCKET == benchmark_scenes.BUCKET
  assert run.key().startswith("v") and "-c" in run.key()
