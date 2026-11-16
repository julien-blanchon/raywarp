"""The kernel fill's build cache: what earlier fills compiled, reused by content, never rebuilt.

A fill (`tools/prebuilt/fill_all.sh`) compiles units whose names are content addresses: a PTX key
(`programs.ptx_key`: the build's Features, flags and target; valid for the kernel code its entry's
reach digests, `prebuilt.Entry.fresh`), a Warp module directory (Warp's content hash of its
generated code), an OptiX module (`modules.digest` of its PTX and record), a listing part and a
scene's census rows (keyed by the code and inputs they are a pure function of:
`tools/prebuilt/buildkey.py`). The last published bundle holds the units of one fill; the cache
bucket (`CACHE_BUCKET`, tools/prebuilt/buckets.env; private; mounted at /cache in every fill job)
holds every unit any fill made, published or not (a failed run, a dry run, the census and the
listing, which no bundle carries), so a fill compiles only what no earlier fill made. Layout:

  ptx/<ptx key>/<code>.json              a PTX entry (`prebuilt.Entry`: the code digest, the reach;
                                         no module records: those are the census's, joined at merge)
  ptx/<ptx key>/<code>.ptx.xz            its PTX as NVRTC wrote it (a key names its PTX target, so
                                         sm_120 and sm_89 share theirs)
  sm<A>/warp/<Warp release>/<module>.tar one Warp kernel-cache entry of sm_<A> (cubin, meta)
  sm<A>/optix/<driver>/m/<module>.db     an OptiX module file (`build.py pack --keep`)
  sm<A>/list/<list key>/part-<i>.json    a listing part (`fill list`; the key names the part count)
  sm<A>/census/<census key>/<scene>.json a scene's census units for sm_<A>'s profile (word_reach)
  refs/<tag>.json                        the units release <tag> used, per architecture (`refs`)

Every entry is immutable: written once under a temporary name, then renamed (a reader never sees a
part); a name already present is never written again (same name, same content). A PTX entry's JSON
is written after its PTX: a listed entry is complete.

  python -m tools.prebuilt.build_cache refs --arch A --bundle DIR [--pack DIR] --list-key K
      [--census-key C] --out F.json
  python -m tools.prebuilt.build_cache prune [--keep 3] [--grace-days 7] [--apply]

`prune` (the workflow, after a publish: huggingface_hub and HF_TOKEN): keeps, per architecture,
what the last `--keep` releases that built it reference (`refs/`), the shared PTX any of them
references, and anything uploaded in the last `--grace-days` (a fill in flight); deletes the rest
(dry unless `--apply`).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import lzma
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
  from raywarp.core.build.kernel_cache.prebuilt import Entry

# The bucket names (KEY=value lines, `buckets`).
BUCKETS: Final = Path(__file__).with_name("buckets.env")
REFS: Final = "refs"


def buckets() -> dict[str, str]:
  """The bucket names of `BUCKETS` (KERNELS_BUCKET, CACHE_BUCKET, PROJECT_BUCKET)."""
  lines = (line.strip() for line in BUCKETS.read_text().splitlines())
  return dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))


def _place(write: Path, dest: Path) -> None:
  """`write` (a finished temporary file or directory beside `dest`) renamed to `dest`, unless
  `dest` exists (another fill placed the same content first)."""
  try:
    if dest.exists():
      return
    os.replace(write, dest)
  finally:
    if write.is_dir():
      shutil.rmtree(write, ignore_errors=True)
    else:
      write.unlink(missing_ok=True)


def _partial(dest: Path) -> Path:
  """A temporary name beside `dest` (its directory created)."""
  dest.parent.mkdir(parents=True, exist_ok=True)
  return dest.with_name(f".{dest.name}.{os.getpid()}.partial")


@dataclasses.dataclass(frozen=True, slots=True)
class BuildCache:
  """The build cache at `root` (the mounted bucket) as one architecture's fill sees it.

  Attributes:
    root: the cache's root directory.
    arch: the fill's SM version (its Warp modules', its OptiX modules').
  """

  root: Path
  arch: int

  def ptx_entry(self, key: str, path: Path) -> Entry | None:
    """The entry of PTX key `key` the cache holds for this kernel code (`Entry.fresh`; None:
    none), its PTX written at `path`."""
    from raywarp.core.build.kernel_cache import prebuilt

    held = self.root / "ptx" / key
    for meta in sorted(held.glob("*.json")) if held.is_dir() else []:
      try:
        entry = prebuilt.Entry.of(json.loads(meta.read_text()))
      except (OSError, ValueError, KeyError, TypeError):
        continue  # damaged: compiled again (and kept under its own name)
      if entry.code == meta.stem and entry.fresh():
        partial = _partial(path)
        partial.write_bytes(lzma.decompress(meta.with_suffix(".ptx.xz").read_bytes()))
        os.replace(partial, path)
        return entry
    return None

  def keep_ptx(self, key: str, ptx: Path, entry: Entry) -> None:
    """Keeps PTX file `ptx` of key `key` and its `entry` (the code it was generated from), unless
    the cache holds them."""
    meta = self.root / "ptx" / key / f"{entry.code}.json"
    if meta.exists():
      return
    blob = meta.with_suffix(".ptx.xz")
    partial = _partial(blob)
    partial.write_bytes(lzma.compress(ptx.read_bytes(), preset=6))
    _place(partial, blob)
    partial = _partial(meta)
    partial.write_text(json.dumps(dataclasses.replace(entry, ptx="", groups={}).json()))
    _place(partial, meta)

  def warp_dir(self, version: str) -> Path:
    """The cache's Warp modules of this architecture and Warp release `version`."""
    return self.root / f"sm{self.arch}" / "warp" / version

  def take_warp(self, version: str, wanted: Iterable[str], cache: Path) -> set[str]:
    """The Warp modules of `wanted` (module directory names) the cache holds for Warp release
    `version` and Warp kernel cache `cache` lacks, extracted into `cache` (each whole: under a
    temporary name, then renamed); their names."""
    held = self.warp_dir(version)
    names = {p.name.removesuffix(".tar") for p in held.glob("*.tar")} if held.is_dir() else set()
    taken = (set(wanted) & names) - {p.name for p in cache.iterdir()} if cache.is_dir() else set(wanted) & names
    for name in sorted(taken):
      partial = _partial(cache / name)
      shutil.rmtree(partial, ignore_errors=True)
      with tarfile.open(held / f"{name}.tar") as tar:
        tar.extractall(partial, filter="data")
      _place(partial, cache / name)
    return taken

  def keep_warp(self, version: str, module: Path) -> None:
    """Keeps Warp kernel-cache entry `module` (a directory: its files) unless the cache holds it."""
    dest = self.warp_dir(version) / f"{module.name}.tar"
    if dest.exists():
      return
    partial = _partial(dest)
    with tarfile.open(partial, "w") as tar:
      for f in sorted(module.iterdir()):
        tar.add(f, arcname=f.name)
    _place(partial, dest)


# --- references and prune ----------------------------------------------------------------------


def refs(arch: int, bundle: Path, pack: Path | None, list_key: str, census_key: str) -> dict[str, object]:
  """The cache units a release of architecture `arch` uses: its bundle's PTX entries (key/code),
  Warp modules (release/name) and its pack's OptiX modules (driver/name), its listing and census
  keys."""
  from raywarp.core.build.kernel_cache import prebuilt
  from tools.prebuilt import build

  manifest = prebuilt.Manifest.read(bundle / prebuilt.MANIFEST)
  entries = build.read_entries(bundle)
  warp = sorted(p.name for p in (bundle / "warp").iterdir()) if (bundle / "warp").is_dir() else []
  optix: list[str] = []
  if pack is not None:
    driver = prebuilt.Manifest.read(pack / prebuilt.MANIFEST).driver
    optix = sorted(f"{driver}/{p.stem}" for p in (pack / prebuilt.MODULE_DIR).glob(f"*{prebuilt.MODULE_SUFFIX}"))
  return {
    "arch": arch,
    "ptx": sorted(f"{k}/{e['code']}" for k, e in entries.items()),
    "warp": [f"{manifest.warp}/{n}" for n in warp],
    "optix": optix,
    "list": list_key,
    "census": census_key,
  }


def referenced(ref: dict[str, object]) -> set[str]:
  """The cache paths (and path prefixes, ending in "/") one architecture's `refs` names."""
  a = f"sm{ref['arch']}"
  out = {f"ptx/{p}.json" for p in ref["ptx"]} | {f"ptx/{p}.ptx.xz" for p in ref["ptx"]}  # pyrefly: ignore[not-iterable]
  out |= {f"{a}/warp/{p}.tar" for p in ref["warp"]}  # pyrefly: ignore[not-iterable]
  out |= {f"{a}/optix/{d}/m/{n}.db" for d, n in (p.split("/", 1) for p in ref["optix"])}  # pyrefly: ignore[not-iterable]
  prefixes = {f"{a}/list/{ref['list']}/"} | ({f"{a}/census/{ref['census']}/"} if ref["census"] else set())
  return out | prefixes


def doomed(
  files: dict[str, dt.datetime], releases: list[list[dict[str, object]]], keep: int, grace: dt.timedelta, now: dt.datetime
) -> list[str]:
  """The cache paths `prune` deletes: of `files` (path -> upload time), those no kept release
  references (per architecture its last `keep` releases in `releases`, oldest first: each a list of
  per-architecture `refs`), older than `grace`; `refs/` stays."""
  kept: dict[object, int] = {}
  wanted: set[str] = set()
  for release in reversed(releases):
    for ref in release:
      if kept.get(ref["arch"], 0) < keep:
        kept[ref["arch"]] = kept.get(ref["arch"], 0) + 1
        wanted |= referenced(ref)
  prefixes = tuple(w for w in wanted if w.endswith("/"))
  return sorted(
    p
    for p, when in files.items()
    if not p.startswith(f"{REFS}/") and p not in wanted and not p.startswith(prefixes) and now - when > grace
  )


def prune(keep: int, grace_days: float, *, apply: bool) -> list[str]:
  """`doomed` over the cache bucket (listed and its `refs/` read with huggingface_hub, HF_TOKEN),
  deleted when `apply`; the paths."""
  from huggingface_hub import HfApi

  bucket = buckets()["CACHE_BUCKET"]
  api = HfApi()
  listed = api.list_bucket_tree(bucket, recursive=True)
  files = {f.path: f.uploaded_at for f in listed if f.type == "file" and f.uploaded_at is not None}
  tags = sorted((when, p) for p, when in files.items() if p.startswith(f"{REFS}/"))
  releases: list[list[dict[str, object]]] = []
  with tempfile.TemporaryDirectory() as tmp:
    local = [(p, Path(tmp) / p.replace("/", "_")) for _, p in tags]
    api.download_bucket_files(bucket, local, raise_on_missing_files=True)  # pyrefly: ignore[bad-argument-type]
    releases = [json.loads(dest.read_text())["archs"] for _, dest in local]
  gone = doomed(files, releases, keep, dt.timedelta(days=grace_days), dt.datetime.now(dt.UTC))
  print(f"prune {bucket}: {len(gone)} of {len(files)} files unreferenced by the last {keep} releases per architecture")
  if apply and gone:
    for i in range(0, len(gone), 1000):
      api.batch_bucket_files(bucket, delete=gone[i : i + 1000])
  return gone


def main(argv: list[str] | None = None) -> None:
  """`refs`, `prune` (see the module doc)."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("refs")
  r.add_argument("--arch", type=int, required=True)
  r.add_argument("--bundle", type=Path, required=True)
  r.add_argument("--pack", type=Path, default=None)
  r.add_argument("--list-key", required=True)
  r.add_argument("--census-key", default="")
  r.add_argument("--out", type=Path, required=True)
  pr = sub.add_parser("prune")
  pr.add_argument("--keep", type=int, default=3, help="releases kept per architecture")
  pr.add_argument("--grace-days", type=float, default=7.0, help="never delete what is newer (a fill in flight)")
  pr.add_argument("--apply", action="store_true", help="delete (default: list only)")
  a = p.parse_args(argv)
  if a.cmd == "refs":
    a.out.write_text(json.dumps(refs(a.arch, a.bundle, a.pack, a.list_key, a.census_key), indent=0))
  else:
    prune(a.keep, a.grace_days, apply=a.apply)


if __name__ == "__main__":
  main()
