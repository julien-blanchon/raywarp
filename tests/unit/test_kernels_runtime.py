"""`rw.kernels`: the download / compile switches, the per-architecture pins and bundle choice (CPU,
no network: a local store and fakes)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.request
from collections.abc import Generator
from pathlib import Path

import pytest
import warp as wp

import raywarp as rw
from raywarp.core.build.kernel_cache import policy, prebuilt, store
from raywarp.core.build.variants import resolve, tiers
from raywarp.core.pipeline import warmup
from raywarp.core.pipeline.platform.device import device_caps

ARCHS = (80, 86, 89, 90, 120)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
  """Each test starts from the environment's switches and no checked bundle."""
  monkeypatch.delenv(policy.DOWNLOAD_ENV, raising=False)
  monkeypatch.delenv(policy.COMPILE_ENV, raising=False)
  policy.reset()
  prebuilt._BUNDLES.clear()
  yield
  policy.reset()
  prebuilt._BUNDLES.clear()


def test_switches_default_on_follow_the_environment_and_configure(monkeypatch: pytest.MonkeyPatch) -> None:
  assert rw.kernels.current() == policy.Policy(download=True, compile=True)
  monkeypatch.setenv(policy.DOWNLOAD_ENV, "0")
  monkeypatch.setenv(policy.COMPILE_ENV, "off")
  assert rw.kernels.current() == policy.Policy(download=False, compile=False)
  assert rw.kernels.configure(compile=True) == policy.Policy(download=False, compile=True)  # None keeps download
  policy.reset()
  assert not rw.kernels.current().compile
  monkeypatch.setenv(policy.COMPILE_ENV, "maybe")
  with pytest.raises(ValueError, match="RAYWARP_KERNELS_COMPILE"):
    rw.kernels.current()


def test_download_off_never_opens_a_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  def no_network(*args: object, **kwargs: object) -> object:
    raise AssertionError("a download was attempted")

  monkeypatch.setattr(urllib.request, "urlopen", no_network)
  rw.kernels.configure(download=False)
  with pytest.raises(store.FetchError, match="downloads are off"):
    store.listing(tmp_path / "b", "https://example.invalid/sm86/abc", None)


def _pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[int, tuple[Path, str]]:
  """A local store holding a bundle (file list + manifest) per architecture of `ARCHS`, and the
  pins naming them (the format lane K's `tools/release/kernels.py pin` writes)."""
  root = tmp_path / "store"
  monkeypatch.setenv(store.STORE_ENV, str(root))
  pins: dict[str, dict[str, str]] = {}
  out: dict[int, tuple[Path, str]] = {}
  for arch in ARCHS:
    content = f"c{arch:03d}"
    bundle = root / f"sm{arch}" / content
    bundle.mkdir(parents=True)
    prebuilt.Manifest(arch=arch, warp=wp.config.version, content=content).write(bundle / prebuilt.MANIFEST)
    listing = store.Listing.of(bundle)
    (bundle / store.FILES).write_bytes(listing)
    pins[str(arch)] = {"content": content, "files": hashlib.sha256(listing).hexdigest(), "warp": wp.config.version}
    out[arch] = (bundle, content)
  monkeypatch.setattr(store, "PIN", tmp_path / "kernels.json")
  store.PIN.write_text(json.dumps({"bundles": pins}))
  return out


def test_each_architecture_gets_its_own_pinned_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  bundles = _pins(tmp_path, monkeypatch)
  rw.kernels.configure(download=False)  # the store holds them: nothing is fetched
  assert rw.kernels.pinned_architectures() == ARCHS
  for arch, (root, content) in bundles.items():
    found = prebuilt.bundle(arch)
    assert found is not None and found.root == root and found.manifest.arch == arch
    info = rw.kernels.info(arch)
    assert info.pinned == content and info.local and info.architectures == ARCHS
  assert prebuilt.bundle(75) is None and rw.kernels.info(75).pinned is None
  with pytest.raises(RuntimeError, match="pins no kernels bundle for sm_75"):
    rw.kernels.fetch(75)


def test_compile_off_refuses_what_is_not_prebuilt() -> None:
  rw.kernels.configure(compile=False)
  s = rw.RenderSettings()
  cuda = dataclasses.replace(device_caps("cpu"), kind="cuda", arch=120)
  with pytest.raises(rw.kernels.KernelsNotPrebuilt, match="CPU"):
    resolve.refuse_unprebuilt(s, None, device_caps("cpu"))
  with pytest.raises(rw.kernels.KernelsNotPrebuilt, match=r'performance\.kernels="scene"'):
    resolve.refuse_unprebuilt(s.replace(**{"performance.kernels": "scene"}), None, cuda)
  outside = tiers.Choice(tier=None, build=None, outside=("gi_rays",), nearest="default")  # pyrefly: ignore[bad-argument-type]  # a fake build
  with pytest.raises(rw.kernels.KernelsNotPrebuilt, match=r"gi_rays; nearest tier .default.*compile=True"):
    resolve.refuse_unprebuilt(s, outside, cuda)
  served = tiers.Choice(tier="default", build=None)  # pyrefly: ignore[bad-argument-type]  # a fake build
  resolve.refuse_unprebuilt(s, served, cuda)  # a tier's: no error
  rw.kernels.configure(compile=True)
  resolve.refuse_unprebuilt(s, None, device_caps("cpu"))  # compiling allowed: nothing to refuse


@dataclasses.dataclass
class _FakeModule:
  """A Warp module as `warmup._fetch` reads it."""

  name: str
  options: dict[str, int] = dataclasses.field(default_factory=lambda: {"block_dim": 256})

  def get_module_identifier(self, block_dim: int | None = None) -> str:
    return f"{self.name}_{block_dim}"


def test_compile_off_refuses_a_warp_module_the_cache_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path))
  cached = tmp_path / "rw_cached_256"
  cached.mkdir()
  (cached / "rw_cached_256.o").write_bytes(b"")
  rw.kernels.configure(compile=False)
  warmup._fetch("cpu", [_FakeModule("rw_cached")], None)  # pyrefly: ignore[bad-argument-type]  # a fake module
  with pytest.raises(rw.kernels.KernelsNotPrebuilt, match=r"1 Warp modules.*rw_missing_256"):
    warmup._fetch("cpu", [_FakeModule("rw_cached"), _FakeModule("rw_missing")], None)  # pyrefly: ignore[bad-argument-type]
  assert warmup._cached("rw_cached_256", 0) and not warmup._cached("rw_cached_256", 86)
