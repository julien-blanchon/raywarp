"""The prebuilt kernels' remote: the release bundles laid out for download
(`raywarp/core/build/kernel_cache/prebuilt.py`, `store.py`), the package's pins, the upload.

  python -m tools.release.kernels layout BUNDLE OUT [--pack DIR] [--workers N]
  python -m tools.release.kernels pin OUT [--to F] [--bucket OWNER/NAME]
  python -m tools.release.kernels publish OUT [--bucket OWNER/NAME] [--prefix P]  # user-approved

`layout`: a complete bundle directory (`tools/prebuilt/build.py merge` / `collect`: manifest.json,
warp/, ptx/) becomes `OUT/sm<arch>/<content>/`: every file xz-compressed (`<path>.xz`; the PTX is
already), `warp/common.tar.xz` (the common Warp modules in one file: what every renderer's first
start fetches, one request) and `files.json` (each file's sha256 and size as the store holds it,
decompressed). A driver's OptiX module pack (`build.py pack` / `merge-packs` output: manifest.json
with the driver, optix7cache.db) becomes `.../optix/<driver>/` the same way. A layout is
content-addressed and the remote is a mutable storage bucket, so the versioning is the layout
itself: `publish` uploads each bundle (its files minus `optix/`) and each pack as a unit, and
refuses a unit whose `files.json` the bucket already holds (a published content key or pack is never
written again; a new fill is a new content key, a later pack of an existing bundle is a new unit).
`--prefix P`: under `P/` of the bucket instead of its root (a dry run's scratch place; the package
reads the root).

`pin`: the release's pins (default: raywarp/core/build/kernel_cache/kernels.json), naming the
bucket the bundles are published to, then one per architecture laid out in OUT (its content, the
sha256 of its files.json, the Warp release it was built with and its OptiX module packs: driver ->
the sha256 of the pack's files.json; none on the software-path architectures sm_80 / sm_90): the
package then accepts exactly those bytes, and the release pins that Warp
(`tools/release/pin_warp.py`). A multi-architecture release (`tools/prebuilt/fill_all.sh`, ARCHS)
lays every architecture out into one OUT, then pins once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import lzma
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from raywarp.core.build.kernel_cache import prebuilt, store

# The pins' format version (`pin`; research/kb/lanes/K-ARCH.md "Pins format for lane P"): 2 adds
# `format` and each bundle's `packs` (NVIDIA driver -> sha256 of that pack's files.json); 3 the
# `bucket` the bundles were published to (the runtime's remote: `store.pinned_bucket`).
PINS_FORMAT = 3


def _compress(src: Path, dst: Path) -> None:
  """`src` xz-compressed into `dst` (copied when already compressed)."""
  dst.parent.mkdir(parents=True, exist_ok=True)
  if src.name.endswith(".xz"):
    shutil.copyfile(src, dst)
    return
  with src.open("rb") as r, lzma.open(dst, "wb", preset=9) as w:
    shutil.copyfileobj(r, w, store.CHUNK)


def _lay(src: Path, out: Path, workers: int, extra: dict[str, Path] | None = None) -> str:
  """Every file of `src` (and the `extra` files: bundle path -> file) compressed into `out`, and
  `out`/files.json listing them all; the sha256 of files.json."""
  files = store.Listing.under(src) | (extra or {})
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    jobs = [pool.submit(_compress, p, out / store.remote_name(rel)) for rel, p in files.items()]
    for job in jobs:
      job.result()
  listing = store.Listing.encode(files)
  (out / store.FILES).write_bytes(listing)
  return hashlib.sha256(listing).hexdigest()


def _common_tar(bundle: Path, dest: Path) -> Path:
  """The archive of `bundle`'s common Warp modules (`prebuilt.common_modules`: one request for
  them), deterministic (sorted, no times or owners)."""
  b = prebuilt.Bundle(bundle, prebuilt.Manifest.read(bundle / prebuilt.MANIFEST))
  modules = b.module_files()
  with tarfile.open(dest, "w", format=tarfile.PAX_FORMAT) as tar:
    for rel in sorted(rel for name in prebuilt.common_modules(b) for rel in modules[name]):
      info = tarfile.TarInfo(rel)
      info.size, info.mode = (bundle / rel).stat().st_size, 0o644
      with (bundle / rel).open("rb") as f:
        tar.addfile(info, f)
  return dest


def layout(bundle: Path, out: Path, *, pack: Path | None, workers: int) -> Path:
  """`bundle` (and the module `pack` of one driver) laid out under `out`; the bundle's directory."""
  manifest = prebuilt.Manifest.read(bundle / prebuilt.MANIFEST)
  target = out / f"sm{manifest.arch}" / manifest.content
  with tempfile.TemporaryDirectory() as scratch:
    common = _common_tar(bundle, Path(scratch) / "common.tar")
    sha = _lay(bundle, target, workers, {store.COMMON: common})
  print(f"{target}: files.json sha256 {sha}")
  if pack is not None:
    driver = prebuilt.Manifest.read(pack / prebuilt.MANIFEST).driver
    if not driver:
      raise SystemExit(f"{pack} is not a module pack (its manifest names no driver)")
    print(f"{target}/optix/{driver}: files.json sha256 {_lay(pack, target / 'optix' / driver, workers)}")
  return target


def pin(out: Path, to: Path | None = None, bucket: str = store.BUCKET) -> dict[str, dict[str, object]]:
  """The pins of the bundles laid out in `out` (one per architecture, `PINS_FORMAT`; published to
  storage bucket `bucket`, which the pins name: the runtime downloads from it) written to `to`
  (default `store.PIN`, the package's)."""
  bundles: dict[str, dict[str, object]] = {}
  for files in sorted(out.glob(f"sm*/*/{store.FILES}")):
    arch = files.parent.parent.name.removeprefix("sm")
    if arch in bundles:
      raise SystemExit(f"two bundles of sm{arch} in {out}: lay out one per architecture for a release")
    manifest = json.loads(lzma.decompress((files.parent / store.remote_name(prebuilt.MANIFEST)).read_bytes()))
    packs = {
      p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files.parent.glob(f"optix/*/{store.FILES}"))
    }
    bundles[arch] = {
      "content": files.parent.name,
      "files": hashlib.sha256(files.read_bytes()).hexdigest(),
      "warp": str(manifest["warp"]),
      "packs": packs,
    }
  pins = {"format": PINS_FORMAT, "bucket": bucket, "bundles": bundles}
  (to or store.PIN).write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
  return bundles


# The releases' storage bucket (the runtime's default remote).
BUCKET = store.BUCKET


def units(out: Path) -> list[Path]:
  """The publishable units of layout `out`: each bundle directory and each of its packs."""
  bundles = sorted(f.parent for f in out.glob(f"sm*/*/{store.FILES}"))
  packs = sorted(f.parent for f in out.glob(f"sm*/*/optix/*/{store.FILES}"))
  return bundles + packs


def _published(bucket: str, rel: str) -> bool:
  """Whether the bucket holds unit `rel` (its files.json).

  Raises:
    SystemExit: the bucket cannot be listed (never upload blind).
  """
  done = subprocess.run(
    ["hf", "buckets", "list", f"{bucket}/{rel}/{store.FILES}", "--json"], capture_output=True, text=True, check=False
  )
  if done.returncode:
    raise SystemExit(f"cannot list {bucket}/{rel}: {done.stderr.strip()}")
  return store.FILES in done.stdout


def publish(out: Path, bucket: str = BUCKET, prefix: str = "") -> None:
  """Uploads the units of layout `out` to storage bucket `bucket` (under `prefix`/ when given),
  each one's files.json last (a unit is listed once complete).

  Raises:
    SystemExit: a unit is already published (a content key or pack is never written again).
  """
  todo = units(out)
  root = f"{bucket}/{prefix.strip('/')}" if prefix.strip("/") else bucket
  taken = [u.relative_to(out).as_posix() for u in todo if _published(root, u.relative_to(out).as_posix())]
  if taken:
    raise SystemExit(f"already published, never overwritten (a new fill is a new content key): {taken}")
  for unit in todo:
    rel = unit.relative_to(out).as_posix()
    dest = f"hf://buckets/{root}/{rel}"
    sync = ["hf", "buckets", "sync", str(unit), dest, "--exclude", store.FILES]
    if unit.parent.name != "optix":
      sync += ["--exclude", "optix/*"]  # the packs are units of their own
    subprocess.run(sync, check=True)
    subprocess.run(["hf", "buckets", "cp", str(unit / store.FILES), f"{dest}/{store.FILES}"], check=True)
    print(f"published {dest}")


def main(argv: list[str] | None = None) -> None:
  """`layout`, `pin`, `publish` (see the module doc)."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  lay = sub.add_parser("layout")
  lay.add_argument("bundle", type=Path)
  lay.add_argument("out", type=Path)
  lay.add_argument("--pack", type=Path, default=None, help="a driver's module pack directory")
  lay.add_argument("--workers", type=int, default=4, help="xz processes")
  pin_p = sub.add_parser("pin")
  pin_p.add_argument("out", type=Path)
  pin_p.add_argument("--to", type=Path, default=None, help="the pins file (default: the package's kernels.json)")
  pin_p.add_argument("--bucket", default=BUCKET, help="the bucket the bundles are published to (the pins name it)")
  pub = sub.add_parser("publish")
  pub.add_argument("out", type=Path)
  pub.add_argument("--bucket", default=BUCKET)
  pub.add_argument("--prefix", default="", help="a place under the bucket's root (a dry run's scratch)")
  args = p.parse_args(argv)
  if args.cmd == "layout":
    layout(args.bundle, args.out, pack=args.pack, workers=args.workers)
  elif args.cmd == "pin":
    print(json.dumps(pin(args.out, args.to, args.bucket), indent=1))
  else:
    publish(args.out, args.bucket, args.prefix)


if __name__ == "__main__":
  main()
