"""`raywarp` (`raywarp.cli`): the parser, and each subcommand on the CPU with fakes where a GPU,
an encoder or an extra would be needed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from raywarp import cli
from raywarp.cache import prune
from raywarp.cli import convert, record
from raywarp.core.build.kernel_cache import policy, store


def test_every_command_parses() -> None:
  p = cli.parser()
  assert p.parse_args(["view", "a.usd", "--port", "0"]).run.__module__ == "raywarp.viewer.command"
  args = p.parse_args(["record", "s.wpbr", "o.mp4", "--size", "64x32", "--worlds", "4", "--settings", "fast"])
  assert args.size == (64, 32) and args.worlds == 4 and args.run is record.run
  assert p.parse_args(["convert", "a.glb", "a.xml", "--normalize", "0.3"]).normalize == 0.3
  assert p.parse_args(["kernels", "fetch", "--arch", "86"]).arch == 86
  assert p.parse_args(["cache", "prune", "--max-gb", "2"]).max_gb == 2.0
  assert p.parse_args(["prepare", "a.usd", "-o", "b.wpbr"]).out == Path("b.wpbr")
  for bad in (["warm"], ["explain"], ["record", "s", "o", "--size", "0x3"], ["cache", "prune"]):
    with pytest.raises(SystemExit):
      p.parse_args(bad)


def test_cache_info_and_prune(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
  (tmp_path / "bakes").mkdir()
  (tmp_path / "bakes" / "old.npz").write_bytes(b"x" * 4096)
  (tmp_path / "kernels").mkdir()
  (tmp_path / "kernels" / "k").write_bytes(b"k" * 100)
  (tmp_path / "scene.wpbr").write_bytes(b"s" * 10)
  assert prune.sizes(tmp_path) == {"bakes": 4096, "kernels": 100, "wpbr": 10}
  assert cli.main(["cache", "info", "--root", str(tmp_path)]) == 0
  assert "bakes" in capsys.readouterr().out
  assert cli.main(["cache", "prune", "--root", str(tmp_path), "--max-gb", "0"]) == 0
  assert "removed 2 entries" in capsys.readouterr().out
  assert not (tmp_path / "bakes" / "old.npz").exists() and (tmp_path / "kernels" / "k").exists()  # the store stays


def test_kernels_info_reads_the_pins(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(store, "PIN", tmp_path / "kernels.json")
  store.PIN.write_text(json.dumps({"bundles": {a: {"content": f"c{a}", "files": "0" * 64} for a in ("86", "120")}}))
  monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
  monkeypatch.setenv(policy.COMPILE_ENV, "0")
  assert cli.main(["kernels", "info", "--arch", "86"]) == 0
  out = capsys.readouterr().out
  assert "c86" in out and "sm_86, sm_120" in out and "compile        off" in out
  assert cli.main(["kernels", "info", "--arch", "75"]) == 0
  assert "none pinned" in capsys.readouterr().out
  with pytest.raises(RuntimeError, match="pins no kernels bundle for sm_75"):
    cli.main(["kernels", "fetch", "--arch", "75"])


def test_info_on_a_cpu_machine(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  monkeypatch.setattr(wp, "get_cuda_device_count", lambda: 0)
  assert cli.main(["info"]) == 0
  assert "CPU (no CUDA device)" in capsys.readouterr().out


def test_convert_routes_by_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[tuple[str, ...]] = []
  pytest.importorskip("pygltflib", reason="needs the gltf extra")
  pytest.importorskip("pxr", reason="needs the usd extra")
  from raywarp.formats.gltf import to_usd
  from raywarp.integrations.mujoco import from_gltf as to_mjcf

  monkeypatch.setattr(to_usd, "convert", lambda src, out: calls.append(("usd", src.name, out.name)) or {})
  monkeypatch.setattr(
    to_mjcf, "convert", lambda src, d, name, s, n: calls.append(("mjcf", src.name, name, str(s), str(n))) or d
  )
  assert cli.main(["convert", "a.glb", str(tmp_path / "a.usdc")]) == 0
  assert cli.main(["convert", "a.gltf", str(tmp_path / "robot.xml"), "--scale", "2"]) == 0
  assert calls == [("usd", "a.glb", "a.usdc"), ("mjcf", "a.gltf", "robot", "2.0", "None")]
  with pytest.raises(SystemExit, match=r"write \.usd"):
    convert.run(argparse.Namespace(source=Path("a.glb"), out=Path("a.obj"), scale=1.0, normalize=None))
  with pytest.raises(SystemExit, match="not a glTF"):
    convert.run(argparse.Namespace(source=Path("a.usd"), out=Path("a.xml"), scale=1.0, normalize=None))


def test_orbit_turns_the_eye_about_the_target() -> None:
  view = ((2.0, 0.0, 1.0), (0.0, 0.0, 1.0), 45.0)
  start, half = record.orbit(view, 0.0), record.orbit(view, 180.0)
  np.testing.assert_allclose(start[:3], [2.0, 0.0, 1.0], atol=1e-6)
  np.testing.assert_allclose(half[:3], [-2.0, 0.0, 1.0], atol=1e-6)


def test_record_renders_and_writes_each_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`record` on the CPU with fake scene loading, renderer and recorder: one pose, render and write
  per frame, a mosaic for several worlds."""
  import raywarp as rw
  import raywarp.video
  from raywarp.viewer import scenes

  seen: dict[str, object] = {"poses": [], "writes": 0}

  class FakeRenderer:
    def __init__(
      self, gpu: object, *, nworld: int, cameras: list[rw.CameraSpec], resolution: tuple[int, int], settings: object
    ) -> None:
      seen["renderer"] = (nworld, resolution, cameras[0].fovy)
      self.camera_pose = self
      self.outputs = argparse.Namespace(rgb=wp.zeros((nworld, 1, resolution[1], resolution[0]), dtype=wp.vec4ub, device="cpu"))

    def assign(self, pose: np.ndarray) -> None:
      seen["poses"].append(pose.shape)  # pyrefly: ignore[missing-attribute]

    def render(self, sim: object) -> None:
      del sim

    def close(self) -> None:
      seen["closed"] = True

  class FakeRecorder:
    stats = "stats"

    def __enter__(self) -> FakeRecorder:
      return self

    def __exit__(self, *exc: object) -> None:
      del exc

    def write(self, rgb: object) -> None:
      seen["writes"] = int(seen["writes"]) + 1  # pyrefly: ignore[bad-argument-type]

  def fake_recorder(path: Path, shape: tuple[int, int, int, int], **kw: object) -> FakeRecorder:
    seen["recorder"] = (shape, kw["layout"], kw["fps"])
    return FakeRecorder()

  loaded = scenes.Loaded(scenes.Entry("x", "x", "files"), None, None, ((2.0, 0.0, 1.0), (0.0, 0.0, 1.0), 40.0), 1.0)  # pyrefly: ignore[bad-argument-type]
  monkeypatch.setattr(scenes, "file_entry", lambda path: loaded.entry)
  monkeypatch.setattr(scenes, "load", lambda entry, device: loaded)
  monkeypatch.setattr(rw, "Renderer", FakeRenderer)
  monkeypatch.setattr(raywarp.video, "recorder", fake_recorder)
  assert (
    cli.main(
      ["record", "s.wpbr", str(tmp_path / "o.mp4"), "--frames", "3", "--worlds", "2", "--size", "64x32", "--device", "cpu"]
    )
    == 0
  )
  assert seen["renderer"] == (2, (64, 32), 40.0) and seen["recorder"] == ((2, 1, 32, 64), "mosaic", 30)
  assert seen["poses"] == [(2, 1, 7)] * 3 and seen["writes"] == 3 and seen["closed"]
