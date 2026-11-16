"""tools/release/kernels.py: the remote layout of a bundle and its pack, the pins, and the publish
rule (a published content key or pack is never written again), without the network."""

from __future__ import annotations

import hashlib
import json
import lzma
import subprocess
from pathlib import Path

import pytest
import warp as wp

from raywarp.core.build.kernel_cache import prebuilt, store
from tools.release import kernels


def _bundle(root: Path) -> Path:
  """A complete bundle (one module, one PTX) and a pack of driver 580 beside it (one module
  file)."""
  b = root / "raywarp_kernels" / "sm120"
  (b / "warp" / "wp_m_0123456").mkdir(parents=True)
  (b / "warp" / "wp_m_0123456" / "wp_m_0123456.sm120.cubin").write_bytes(b"cubin")
  (b / "ptx").mkdir()
  (b / "ptx" / "a.ptx.xz").write_bytes(lzma.compress(b"ptx"))
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="c0").write(b / prebuilt.MANIFEST)
  p = root / "pack"
  (p / prebuilt.MODULE_DIR).mkdir(parents=True)
  (p / prebuilt.module_path("m0")).write_bytes(b"rows")
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="p0", driver="580").write(p / prebuilt.MANIFEST)
  return b


def test_layout_and_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Every file compressed (the PTX as is), the common modules as one archive, a pack under its
  driver; the pins name the content, the file list's sha256 and the Warp release."""
  monkeypatch.setattr(store, "PIN", tmp_path / "kernels.json")
  target = kernels.layout(_bundle(tmp_path), tmp_path / "out", pack=tmp_path / "pack", workers=2)
  assert target == tmp_path / "out" / "sm120" / "c0"
  assert (target / "warp/wp_m_0123456/wp_m_0123456.sm120.cubin.xz").is_file()
  assert (target / "ptx/a.ptx.xz").read_bytes() == lzma.compress(b"ptx")
  assert (target / "warp/common.tar.xz").is_file()
  listed = json.loads((target / store.FILES).read_text())["files"]
  assert store.COMMON in listed and not [rel for rel in listed if rel.startswith("optix/")]
  assert (target / "optix" / "580" / "m" / "m0.db.xz").is_file()
  pack = json.loads((target / "optix" / "580" / store.FILES).read_text())["files"]
  assert sorted(pack) == sorted([prebuilt.MANIFEST, prebuilt.module_path("m0")])
  pins = kernels.pin(tmp_path / "out")
  assert pins["120"]["content"] == "c0" and pins["120"]["warp"] == wp.config.version
  assert [u.relative_to(tmp_path / "out").as_posix() for u in kernels.units(tmp_path / "out")] == [
    "sm120/c0",
    "sm120/c0/optix/580",
  ]


def test_publish_refuses_a_published_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A content key or pack the bucket holds with other bytes is never uploaded again; new ones
  upload, each file list last; a bucket that cannot be listed stops the publish."""
  kernels.layout(_bundle(tmp_path), tmp_path / "out", pack=tmp_path / "pack", workers=2)
  calls: list[list[str]] = []
  held: set[str] = {"sm120/c0"}

  def fake(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str | bytes]:
    calls.append(cmd)
    if cmd[:3] == ["hf", "buckets", "list"]:
      rel = cmd[3].split("/", 2)[2].removesuffix(f"/{store.FILES}")
      return subprocess.CompletedProcess(cmd, 0, store.FILES if rel in held else "", "")
    if cmd[2] == "cp" and cmd[-1] == "-":
      return subprocess.CompletedProcess(cmd, 0, b"{}", b"")  # another fill's file list
    return subprocess.CompletedProcess(cmd, 0, "", "")

  monkeypatch.setattr(kernels.subprocess, "run", fake)
  with pytest.raises(SystemExit, match="already published with other bytes"):
    kernels.publish(tmp_path / "out")
  assert all(c[2] == "list" or c[-1] == "-" for c in calls)
  held.clear()
  calls.clear()
  kernels.publish(tmp_path / "out")
  uploads = [c for c in calls if c[2] in ("sync", "cp")]
  assert [c[2] for c in uploads] == ["sync", "cp", "sync", "cp"]
  assert "optix/*" in uploads[0] and uploads[1][-1].endswith(f"sm120/c0/{store.FILES}")

  calls.clear()
  kernels.publish(tmp_path / "out", prefix="scratch/dry/")  # a dry run's place: every path under it
  assert all("raywarp-kernels/scratch/dry/sm120/" in " ".join(c) for c in calls)

  def unlisted(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, 1, "", "401")

  monkeypatch.setattr(kernels.subprocess, "run", unlisted)
  with pytest.raises(SystemExit, match="cannot list"):
    kernels.publish(tmp_path / "out")


def test_one_pins_file_for_every_architecture(tmp_path: Path) -> None:
  """A multi-architecture layout (an OptiX bundle with its pack, a software-path bundle without)
  pins both in one file (format 4): the bucket they are published to, each bundle's content, file
  list and Warp release, and its packs by driver (the sha256 of the pack's file list); the
  runtime's `store.pin` reads it, and downloads from that bucket (`store.remote`)."""
  kernels.layout(_bundle(tmp_path), tmp_path / "out", pack=tmp_path / "pack", workers=2)
  soft = tmp_path / "soft" / "sm80"
  (soft / "warp" / "wp_m_0123456").mkdir(parents=True)
  (soft / "warp" / "wp_m_0123456" / "wp_m_0123456.sm80.cubin").write_bytes(b"cubin")
  prebuilt.Manifest(arch=80, warp=wp.config.version, content="s0").write(soft / prebuilt.MANIFEST)
  kernels.layout(soft, tmp_path / "out", pack=None, workers=2)
  to = tmp_path / "pins.json"
  kernels.pin(tmp_path / "out", to, bucket="owner/old-kernels")
  pins = json.loads(to.read_text())
  assert pins["format"] == kernels.PINS_FORMAT == 4 and sorted(pins["bundles"]) == ["120", "80"]
  assert pins["bucket"] == "owner/old-kernels"
  pack_files = tmp_path / "out" / "sm120" / "c0" / "optix" / "580" / store.FILES
  assert pins["bundles"]["120"]["packs"] == {"580": hashlib.sha256(pack_files.read_bytes()).hexdigest()}
  assert pins["bundles"]["80"]["packs"] == {} and pins["bundles"]["80"]["content"] == "s0"
  with pytest.MonkeyPatch.context() as m:
    m.setattr(store, "PIN", to)
    assert store.pin(80) == ("s0", pins["bundles"]["80"]["files"]) and store.pin(86) is None
    m.delenv(store.URL_ENV, raising=False)
    assert store.remote() == "https://huggingface.co/buckets/owner/old-kernels/resolve"
    to.write_text(json.dumps({k: v for k, v in pins.items() if k != "bucket"}))
    assert store.remote() == store.KERNELS_URL
