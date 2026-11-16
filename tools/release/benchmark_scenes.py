"""The public benchmark's scene files on the public assets bucket (`blanchon/raywarp-assets`,
tools/prebuilt/buckets.env's ASSETS_BUCKET; `benchmarks/run.py` downloads them), one directory per
scene-file version (`benchmarks/<key>/`, `benchmarks.run.key()`: v<.rayw format>-c<USD converter>).

  python -m tools.release.benchmark_scenes layout SRC OUT    # SRC: franka.tar, showcase.rayw,
                                                             # sponza.rayw, cameras.json
  python -m tools.release.benchmark_scenes publish OUT [--bucket OWNER/NAME] [--prefix P]
                                                             # user-approved

`layout`: SRC's scene files and their cameras (built by the development repository's
public-scenes script) become `OUT/benchmarks/<key>/`: each xz-compressed and
`files.json` (their sha256 and size decompressed), the layout the kernel store reads
(`raywarp/core/build/kernel_cache/store.py`). `publish`: uploads it, files.json last, and refuses
a key the bucket already holds (a published version is never written again: new scene content
needs a new converter or format version). Run by the release pipeline after the kernels'
(`.github/workflows/kernels.yml`), never by hand without the user's go.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from benchmarks.run import ASSETS_BUCKET, CAMERAS, FILES, key
from raywarp.core.build.kernel_cache import store
from tools.release.kernels import _compress, _published

# The public assets bucket (tools/prebuilt/buckets.env's ASSETS_BUCKET), not the kernels'.
BUCKET = ASSETS_BUCKET


def layout(src: Path, out: Path) -> Path:
  """The scene files of `src` laid out under `out`; the version directory."""
  files = {rel: src / rel for rel in (*FILES.values(), CAMERAS)}
  missing = [str(p) for p in files.values() if not p.is_file()]
  if missing:
    raise SystemExit(f"missing scene files: {missing} (internal/benchmarks/public_scenes.py builds them)")
  dest = out / "benchmarks" / key()
  for rel, path in files.items():
    _compress(path, dest / store.remote_name(rel))
  (dest / store.FILES).write_bytes(store.Listing.encode(files))
  return dest


def publish(out: Path, bucket: str = BUCKET, prefix: str = "") -> None:
  """Uploads `out`'s version directory to `bucket` (under `prefix`/), files.json last.

  Raises:
    SystemExit: the bucket holds this version already.
  """
  rel = f"benchmarks/{key()}"
  root = f"{bucket}/{prefix.strip('/')}" if prefix.strip("/") else bucket
  if _published(root, rel) is not None:
    raise SystemExit(f"{root}/{rel} is published already (never overwritten)")
  dest = f"hf://buckets/{root}/{rel}"
  subprocess.run(["hf", "buckets", "sync", str(out / rel), dest, "--exclude", store.FILES], check=True)
  subprocess.run(["hf", "buckets", "cp", str(out / rel / store.FILES), f"{dest}/{store.FILES}"], check=True)
  print(f"published {dest}")


def main(argv: list[str] | None = None) -> None:
  """The command line."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("layout")
  s.add_argument("src", type=Path)
  s.add_argument("out", type=Path)
  s = sub.add_parser("publish")
  s.add_argument("out", type=Path)
  s.add_argument("--bucket", default=BUCKET)
  s.add_argument("--prefix", default="")
  a = p.parse_args(argv)
  if a.cmd == "layout":
    print(layout(a.src, a.out))
  else:
    publish(a.out, a.bucket, a.prefix)


if __name__ == "__main__":
  main()
