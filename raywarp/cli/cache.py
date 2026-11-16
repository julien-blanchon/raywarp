"""`raywarp cache info | prune --max-gb N`: raywarp's disk caches (`raywarp.cache`).

info: the cache directory (RAYWARP_CACHE_DIR) and what each cache in it holds.
prune: deletes the least recently used entries of the content-keyed caches (bakes, textures,
`.rayw` scenes, PTX, converted glTF) until they fit (`raywarp.cache.prune`); the kernel store and
OptiX's cache are left alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from raywarp.cli.common import Sub

_GB: Final = 1 << 30


def register(sub: Sub) -> None:
  """Adds `cache info` and `cache prune`."""
  p = sub.add_parser("cache", help="the disk caches: info, prune", description=__doc__)
  actions = p.add_subparsers(dest="action", required=True)
  for name, doc in (("info", "the cache directory and its sizes"), ("prune", "delete the least recently used entries")):
    a = actions.add_parser(name, help=doc)
    a.add_argument(
      "--root", type=Path, default=None, help="the cache directory (default: $RAYWARP_CACHE_DIR, else the user cache)"
    )
    if name == "prune":
      a.add_argument("--max-gb", type=float, required=True, help="the size to keep the pruned caches under [GiB]")
    a.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
  """Prints the sizes, or prunes."""
  from raywarp.cache import prune  # noqa: PLC0415
  from raywarp.cache.paths import cache_dir  # noqa: PLC0415

  root = args.root or cache_dir()
  if root is None:
    raise SystemExit("raywarp cache: caching is off (RAYWARP_CACHE_DIR is empty)")
  if args.action == "prune":
    if args.max_gb < 0:
      raise SystemExit("raywarp cache prune: --max-gb must be >= 0")
    print(prune.prune(int(args.max_gb * _GB), root))
    return 0
  found = prune.sizes(root)
  print(f"{root}: {sum(found.values()) / _GB:.2f} GB")
  for name, size in found.items():
    print(f"  {name:<12} {size / _GB:8.2f} GB")
  return 0
