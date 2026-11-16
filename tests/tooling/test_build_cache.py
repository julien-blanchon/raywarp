"""The kernel fill's reuse (tools/prebuilt/build_cache.py, buildkey.py, fill.py's seed and cache):
a PTX unit or Warp module made once is taken from the build cache or the published bundle and never
compiled again while its code is unchanged; a key changes with what reaches a bundle and nothing
else; the prune keeps what the last releases reference; the bucket names are written once."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest
import warp as wp

from raywarp.core.build.kernel_cache import kernel_code, prebuilt, store
from tools.hf import config as hf_config
from tools.prebuilt import build_cache, buildkey, fill
from tools.prebuilt.build_cache import BuildCache
from tools.release import kernels

ROOT = Path(__file__).resolve().parents[2]


def _entry(reach: frozenset[str] = frozenset()) -> prebuilt.Entry:
  """An entry of this kernel code (`Entry.fresh`) of `reach`."""
  return prebuilt.Entry(ptx="", code=kernel_code.digest(reach), reach=reach, groups={})


def test_a_ptx_unit_kept_once_is_taken_while_its_code_is_unchanged(tmp_path: Path) -> None:
  """`keep_ptx` then `ptx_entry`: the PTX comes back byte for byte with its entry; a key the cache
  lacks, or holds only for other kernel code (a stale reach digest), is a miss; keeping twice
  writes nothing new."""
  cache = BuildCache(tmp_path / "cache", 120)
  src = tmp_path / "k1.ptx"
  src.write_text("// ptx k1")
  cache.keep_ptx("k1", src, _entry())
  cache.keep_ptx("k1", src, _entry())
  assert sorted(p.name for p in (tmp_path / "cache" / "ptx" / "k1").iterdir()) == [
    f"{_entry().code}.json",
    f"{_entry().code}.ptx.xz",
  ]
  out = tmp_path / "fill" / "k1.ptx"
  out.parent.mkdir()
  got = cache.ptx_entry("k1", out)
  assert got is not None and got.code == _entry().code and out.read_text() == "// ptx k1"
  assert cache.ptx_entry("k2", tmp_path / "fill" / "k2.ptx") is None
  stale = prebuilt.Entry(ptx="", code="0" * 24, reach=frozenset(), groups={})
  cache.keep_ptx("k3", src, stale)
  assert cache.ptx_entry("k3", tmp_path / "fill" / "k3.ptx") is None


def test_warp_modules_kept_once_are_extracted_into_a_fill_s_cache(tmp_path: Path) -> None:
  """A Warp kernel-cache entry kept (one tar per module, per architecture and Warp release) is
  extracted whole into another fill's cache; one the fill already holds, another release's or
  another architecture's are not."""
  made = tmp_path / "made" / "wp_m_0123456"
  made.mkdir(parents=True)
  (made / "wp_m_0123456.sm120.cubin").write_bytes(b"cubin")
  (made / "wp_m_0123456.meta").write_text("{}")
  BuildCache(tmp_path / "cache", 120).keep_warp("1.0", made)
  fresh = tmp_path / "fresh"
  assert BuildCache(tmp_path / "cache", 120).take_warp("1.0", {"wp_m_0123456", "wp_x_0000000"}, fresh) == {"wp_m_0123456"}
  assert (fresh / "wp_m_0123456" / "wp_m_0123456.sm120.cubin").read_bytes() == b"cubin"
  assert BuildCache(tmp_path / "cache", 120).take_warp("1.0", {"wp_m_0123456"}, fresh) == set()
  assert BuildCache(tmp_path / "cache", 120).take_warp("2.0", {"wp_m_0123456"}, tmp_path / "f2") == set()
  assert BuildCache(tmp_path / "cache", 89).take_warp("1.0", {"wp_m_0123456"}, tmp_path / "f3") == set()


def test_the_published_bundle_seeds_the_fill_over_https(tmp_path: Path) -> None:
  """`fill --seed URL`: the published bundle (laid out as `kernels.py layout` does, read through a
  file:// URL as through the bucket's) gives each PTX key of this kernel code and each Warp module
  of this architecture and Warp release, fetched one file at a time; a key it lacks is a miss."""
  bundle = tmp_path / "b" / "sm120"
  (bundle / "warp" / "wp_m_0123456").mkdir(parents=True)
  (bundle / "warp" / "wp_m_0123456" / "wp_m_0123456.sm120.cubin").write_bytes(b"cubin")
  (bundle / "ptx").mkdir()
  import lzma

  (bundle / "ptx" / "aa.ptx.xz").write_bytes(lzma.compress(b"// seeded"))
  (bundle / "ptx" / "k1.json").write_text(json.dumps(_entry().json() | {"ptx": "aa.ptx.xz"}))
  prebuilt.Manifest(arch=120, warp=wp.config.version, content="c0").write(bundle / prebuilt.MANIFEST)
  out = kernels.layout(bundle, tmp_path / "out", pack=None, workers=1)
  seed = fill.seed_bundle(out.as_uri())
  got = tmp_path / "fill" / "k1.ptx"
  got.parent.mkdir()
  assert fill._take(seed, "k1", got) is not None and got.read_bytes() == b"// seeded"
  assert fill._take(seed, "k9", tmp_path / "fill" / "k9.ptx") is None
  cache = tmp_path / "fill" / "warp"
  assert fill._take_modules(seed, cache, {"wp_m_0123456", "wp_n_0000000"}, 120) == 1
  assert (cache / "wp_m_0123456" / "wp_m_0123456.sm120.cubin").read_bytes() == b"cubin"
  assert fill._take_modules(seed, tmp_path / "other", {"wp_m_0123456"}, 89) == 0


def test_bundle_keys_follow_the_inputs_that_reach_a_bundle(tmp_path: Path) -> None:
  """A key is stable for the same tree and inputs, changes with an input, and the census key with
  the scene set (not its order); `changed` names the architectures the pins lack under that key
  or bucket."""
  one = buildkey.keyed(["arch=120", "backend=optix"])
  assert one == buildkey.keyed(["arch=120", "backend=optix"]) != buildkey.keyed(["arch=120", "backend=software"])
  a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
  a.write_text("x y")
  b.write_text("y\nx\n")
  c.write_text("x y z")
  assert buildkey.census_key(a) == buildkey.census_key(b) != buildkey.census_key(c)
  pins = tmp_path / "pins.json"
  pins.write_text(json.dumps({"bucket": "o/k", "bundles": {"120": {"key": "k1"}, "80": {"key": "old"}}}))
  assert buildkey.changed({"120": "k1", "80": "k2", "90": "k3"}, pins, "o/k") == ["80", "90"]
  assert buildkey.changed({"120": "k1"}, pins, "o/other") == ["120"]
  assert buildkey.changed({"120": "k1"}, tmp_path / "missing.json", "o/k") == ["120"]


def test_the_code_digest_ignores_the_orchestration_and_reads_the_committed_tree() -> None:
  """The code part digests committed blobs (fill_all.sh and the scene list are not code) and is
  the same from any directory."""
  listed = subprocess.run(
    ["git", "-C", str(ROOT), "ls-tree", "-r", "HEAD", "--", *buildkey.CODE], capture_output=True, text=True, check=True
  ).stdout
  assert "tools/prebuilt/fill.py" in listed and "raywarp/core/build/kernel_cache/store.py" in listed
  assert buildkey.NOT_CODE <= {line.split("\t")[1] for line in listed.splitlines()} | {
    "tools/prebuilt/buckets.env",
    "tools/prebuilt/buildkey.py",
  }
  assert buildkey.tree_digest(buildkey.CODE) != buildkey.tree_digest(tuple(c for c in buildkey.CODE if c != "raywarp"))
  assert buildkey.tree_digest(buildkey.CODE) == buildkey.tree_digest((*buildkey.CODE, "tools/prebuilt/fill_all.sh"))


def test_pins_keep_a_skipped_architecture_and_record_every_key(tmp_path: Path) -> None:
  """`pin --keys`: a laid-out architecture gets its key, one the fill skipped keeps the entry the
  pins held under the same key; a skipped one the pins lack (or hold under another key) refuses."""
  soft = tmp_path / "soft" / "sm80"
  (soft / "warp" / "wp_m_0123456").mkdir(parents=True)
  (soft / "warp" / "wp_m_0123456" / "wp_m_0123456.sm80.cubin").write_bytes(b"cubin")
  prebuilt.Manifest(arch=80, warp=wp.config.version, content="s0").write(soft / prebuilt.MANIFEST)
  kernels.layout(soft, tmp_path / "out", pack=None, workers=1)
  to = tmp_path / "pins.json"
  kept = {"content": "c9", "files": "f", "warp": "1", "packs": {}, "key": "k120"}
  to.write_text(json.dumps({"format": 4, "bucket": "o/k", "bundles": {"120": kept}}))
  kernels.pin(tmp_path / "out", to, bucket="o/k", keys={"80": "k80", "120": "k120"})
  pins = json.loads(to.read_text())
  assert pins["format"] == 4 and pins["bundles"]["120"] == kept and pins["bundles"]["80"]["key"] == "k80"
  with pytest.raises(SystemExit, match="sm120 is not laid out"):
    kernels.pin(tmp_path / "out", to, bucket="o/k", keys={"80": "k80", "120": "other"})


def test_publish_skips_a_unit_published_with_the_same_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A rebuilt bundle equal to the published one (same files.json) uploads nothing; one with other
  bytes under the same content key is refused."""
  soft = tmp_path / "soft" / "sm80"
  (soft / "warp" / "wp_m_0123456").mkdir(parents=True)
  (soft / "warp" / "wp_m_0123456" / "wp_m_0123456.sm80.cubin").write_bytes(b"cubin")
  prebuilt.Manifest(arch=80, warp=wp.config.version, content="s0").write(soft / prebuilt.MANIFEST)
  unit = kernels.layout(soft, tmp_path / "out", pack=None, workers=1)
  remote = {"bytes": (unit / store.FILES).read_bytes()}
  calls: list[list[str]] = []

  def fake(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[object]:
    calls.append(cmd)
    if cmd[2] == "list":
      return subprocess.CompletedProcess(cmd, 0, store.FILES, "")
    if cmd[2] == "cp" and cmd[-1] == "-":
      return subprocess.CompletedProcess(cmd, 0, remote["bytes"], b"")
    return subprocess.CompletedProcess(cmd, 0, "", "")

  monkeypatch.setattr(kernels.subprocess, "run", fake)
  kernels.publish(tmp_path / "out", bucket="o/k")
  assert not [c for c in calls if c[2] == "sync"]
  remote["bytes"] = b"{}"
  with pytest.raises(SystemExit, match="other bytes"):
    kernels.publish(tmp_path / "out", bucket="o/k")


def test_prune_keeps_the_last_releases_of_each_architecture() -> None:
  """Per architecture its last `keep` releases' units stay (a release that skipped it does not
  count), the shared PTX any of them names, every refs file and anything recent; the rest goes."""
  now = dt.datetime(2026, 11, 16, tzinfo=dt.UTC)
  old, new = now - dt.timedelta(days=30), now - dt.timedelta(hours=1)

  def ref(arch: int, n: int) -> dict[str, object]:
    return {
      "arch": arch,
      "ptx": [f"k{n}/c"],
      "warp": [f"1.0/wp_{n}"],
      "optix": [f"580/m{n}"],
      "list": f"l{n}",
      "census": f"c{n}",
    }

  releases = [[ref(120, 1), ref(80, 1)], [ref(120, 2)], [ref(120, 3)]]
  files = {
    "ptx/k1/c.json": old,
    "ptx/k2/c.ptx.xz": old,
    "ptx/k3/c.json": old,
    "ptx/k9/c.json": old,
    "ptx/k8/c.json": new,
    "sm80/warp/1.0/wp_1.tar": old,
    "sm120/warp/1.0/wp_1.tar": old,
    "sm120/warp/1.0/wp_3.tar": old,
    "sm120/optix/580/m/m2.db": old,
    "sm120/list/l1/part-0.json": old,
    "sm80/list/l1/part-0.json": old,
    "sm120/census/c3/office.json": old,
    "refs/t1.json": old,
  }
  gone = build_cache.doomed(files, releases, 2, dt.timedelta(days=7), now)
  # sm120 keeps releases 2 and 3; sm80 keeps release 1; k1 stays (sm80's release 1 names it).
  assert gone == ["ptx/k9/c.json", "sm120/list/l1/part-0.json", "sm120/warp/1.0/wp_1.tar"]


def test_bucket_names_are_written_in_one_place() -> None:
  """buckets.env names the kernels bucket the runtime downloads from by default, and the jobs'
  bucket tools.hf mounts; the workflow and fill_all.sh read it."""
  names = build_cache.buckets()
  assert names["KERNELS_BUCKET"] == store.BUCKET == kernels.BUCKET
  assert names["PROJECT_BUCKET"] == hf_config.BUCKET
  assert names["CACHE_BUCKET"] == "blanchon/raywarp-build-cache"
  assert "tools/prebuilt/buckets.env" in (ROOT / "tools" / "prebuilt" / "fill_all.sh").read_text()
  assert "tools/prebuilt/buckets.env" in (ROOT / ".github" / "workflows" / "kernels.yml").read_text()
