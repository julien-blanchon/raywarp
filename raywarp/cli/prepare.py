"""`raywarp prepare IN.usd [-o OUT.rayw]`: a USD stage baked to a `.rayw` scene file
(`raywarp.formats.usd.bake`): it loads with `rw.Scene.load` (memory-mapped, no USD or MDL
code), in milliseconds. Needs the `usd` extra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from raywarp.cli.common import Sub


def register(sub: Sub) -> None:
  """Adds `prepare`."""
  p = sub.add_parser("prepare", help="bake a USD stage to a .rayw scene file", description=__doc__)
  p.add_argument("source", type=Path, help="the USD stage")
  p.add_argument("-o", "--out", type=Path, default=None, help="the .rayw file (default: beside the stage)")
  p.add_argument("--threads", type=int, default=None, help="texture threads (default: every core)")
  p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
  """Bakes; prints the scene's counts."""
  import raywarp.formats.usd  # noqa: PLC0415  # the usd extra

  out = args.out or args.source.with_suffix(".rayw")
  if out.suffix != ".rayw":
    raise SystemExit(f"raywarp prepare: {out}: the output is a .rayw file")
  print(f"{out}: {raywarp.formats.usd.bake(args.source, out, threads=args.threads)}")
  return 0
