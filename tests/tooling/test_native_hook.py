"""tools/native/hook.py (the native build hook) without the network: the pin table, checksum
verification of pre-fetched artifacts, archive extraction, the patch applier, the platform check."""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from tools.native import hook

ROOT = Path(__file__).resolve().parents[2]
SHA256 = re.compile(r"[0-9a-f]{64}")


def _artifacts() -> list[hook.Artifact]:
  """Every pinned artifact."""
  return [hook.OPTIX_HEADERS, *hook.SOURCE_ARTIFACTS.values(), *(p.release for p in hook.PLATFORMS.values())]


def test_pin_table() -> None:
  """Every artifact: a sha256, an https URL on NVIDIA's GitHub at the pinned version / commit, a
  unique file name; every platform: a release archive of the pinned version and build."""
  artifacts = _artifacts()
  assert len({a.name for a in artifacts}) == len(artifacts)
  for a in artifacts:
    assert SHA256.fullmatch(a.sha256), a
    assert a.url.startswith(("https://github.com/NVIDIA/", "https://raw.githubusercontent.com/NVIDIA/")), a
    assert "/" not in a.name and a.url.rsplit("/", 1)[-1] in (a.name, f"{hook.OPTIX_COMMIT}.tar.gz", a.name.split("-", 3)[-1])
  assert hook.OPTIX_COMMIT in hook.OPTIX_HEADERS.url and len(hook.OPTIX_COMMIT) == 40
  assert all(f"/{hook.MDL_SDK_COMMIT}/" in a.url for a in hook.SOURCE_ARTIFACTS.values()) and len(hook.MDL_SDK_COMMIT) == 40
  for plat in hook.PLATFORMS.values():
    assert f"/download/{hook.MDL_SDK_VERSION}/" in plat.release.url and hook.MDL_SDK_BUILD in plat.release.name
    assert plat.release.name.endswith(".zip" if plat.suffix == ".dll" else ".tgz")
  # The build requirement the hook runs (SWIG) is pinned exactly; the build_ext command is the hook.
  pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
  assert any(re.fullmatch(r"swig==[\d.]+(\.post\d+)?", r) for r in pyproject["build-system"]["requires"])
  assert pyproject["tool"]["setuptools"]["cmdclass"]["build_ext"] == "tools.native.hook.BuildNative"
  assert pyproject["tool"]["setuptools"]["ext-modules"][0]["name"] == hook.BINDING
  # The sdist carries the hook and its patch.
  manifest = (ROOT / "MANIFEST.in").read_text()
  assert "tools/native/hook.py" in manifest and str(hook.PATCH.relative_to(ROOT)) in manifest


def _artifact(tmp: Path, name: str, data: bytes) -> hook.Artifact:
  """`data` written to `tmp`/`name`, and its artifact."""
  (tmp / name).write_bytes(data)
  return hook.Artifact(name, f"https://example.invalid/{name}", hashlib.sha256(data).hexdigest())


def test_fetch_reads_archives_dir_and_checks_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """RAYWARP_NATIVE_ARCHIVES: a matching file is used as is, a mismatch or a missing file is an
  error (no download: the URLs are unreachable)."""
  monkeypatch.setenv("RAYWARP_NATIVE_ARCHIVES", str(tmp_path))
  good = _artifact(tmp_path, "good.tgz", b"pinned bytes")
  assert hook.fetch(good) == tmp_path / "good.tgz"
  bad = hook.Artifact("good.tgz", good.url, "0" * 64)
  with pytest.raises(hook.NativeBuildError, match="sha256"):
    hook.fetch(bad)
  with pytest.raises(hook.NativeBuildError, match="missing"):
    hook.fetch(hook.Artifact("absent.tgz", good.url, good.sha256))


def test_fetch_cache_hit_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Without RAYWARP_NATIVE_ARCHIVES the cache is used; a corrupted cached file is an error, not a
  silent re-download."""
  monkeypatch.delenv("RAYWARP_NATIVE_ARCHIVES", raising=False)
  monkeypatch.setenv("RAYWARP_NATIVE_CACHE", str(tmp_path))
  (tmp_path / "archives").mkdir()
  art = _artifact(tmp_path / "archives", "a.bin", b"abc")
  assert hook.fetch(art) == tmp_path / "archives" / "a.bin"
  (tmp_path / "archives" / "a.bin").write_bytes(b"abd")
  with pytest.raises(hook.NativeBuildError, match="stale download"):
    hook.fetch(art)


def _tar(path: Path, files: dict[str, bytes]) -> None:
  with tarfile.open(path, "w:gz") as t:
    for name, data in files.items():
      info = tarfile.TarInfo(name)
      info.size = len(data)
      t.addfile(info, io.BytesIO(data))


def test_extract_tar_and_zip(tmp_path: Path) -> None:
  """Only the selected regular files below the prefix are written, where `select` says; `..`
  members fail."""
  files = {"root/include/a.h": b"a", "root/include/sub/b.h": b"b", "root/README": b"r", "other/x": b"x"}
  _tar(tmp_path / "a.tgz", files)
  with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
    for name, data in files.items():
      z.writestr(name, data)
  for archive in ("a.tgz", "a.zip"):
    out = tmp_path / f"out-{archive}"

    def select(rel: PurePosixPath, out: Path = out) -> Path | None:
      return out.joinpath(*rel.parts) if rel.parts[0] == "include" else None

    assert hook.extract(tmp_path / archive, "root", select) == 2
    assert (out / "include/sub/b.h").read_bytes() == b"b" and not (out / "README").exists()
  _tar(tmp_path / "evil.tgz", {"root/../escape": b"x"})
  with pytest.raises(hook.NativeBuildError, match="unsafe"):
    hook.extract(tmp_path / "evil.tgz", "root", lambda rel: tmp_path / "x")


def test_apply_patch(tmp_path: Path) -> None:
  """Hunks apply at their lines (offsets carried across hunks); a mismatching hunk fails."""
  (tmp_path / "f.i").write_text("".join(f"line{i}\n" for i in range(1, 11)))
  patch = "--- a/f.i\n+++ b/f.i\n@@ -2,2 +2,3 @@\n line2\n+added\n line3\n@@ -8,3 +9,2 @@\n line8\n-line9\n line10\n"
  hook.apply_patch(patch, tmp_path)
  assert (tmp_path / "f.i").read_text().split() == [
    "line1",
    "line2",
    "added",
    "line3",
    *[f"line{i}" for i in (4, 5, 6, 7, 8, 10)],
  ]
  with pytest.raises(hook.NativeBuildError, match="does not match"):
    hook.apply_patch(patch, tmp_path)
  # The real patch parses: its hunks' line counts match their bodies.
  hunks = re.findall(r"^@@ -\d+,(\d+) \+\d+,(\d+) @@", hook.PATCH.read_text(), re.M)
  assert hunks and all(int(new) > int(old) for old, new in hunks)


def test_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
  """A platform without an MDL SDK release is a clear error naming the supported ones."""
  monkeypatch.setattr(hook.sys, "platform", "freebsd14")
  with pytest.raises(hook.NativeBuildError, match=r"linux/x86_64.*this is freebsd14"):
    hook.host_platform()


def test_cuda_stub_covers_the_driver_calls() -> None:
  """The stub defines each imported driver symbol once and reports no device."""
  source = hook.cuda_stub_source()
  names = re.findall(r"^int (cu\w+)\(", source, re.M)
  assert len(names) == len(set(names)) and set(hook.CUDA_STUB_SYMBOLS) <= set(names)
  assert f"return {hook.CUDA_ERROR_NO_DEVICE};" in source
