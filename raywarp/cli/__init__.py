"""`raywarp`: the command line (the Python API first; this is a thin layer over it).

  raywarp view [SCENE ...]                the web viewer (`raywarp.viewer`)
  raywarp record SCENE OUT.mp4            an MP4 of a scene, the camera orbiting (`raywarp.video`)
  raywarp convert IN.gltf OUT.usdc|.xml   glTF -> USD / MJCF (`raywarp.formats.gltf`)
  raywarp prepare IN.usd [-o OUT.rayw]    a USD stage baked to a `.rayw` scene file
  raywarp kernels info | fetch            the prebuilt GPU kernels (`raywarp.kernels`)
  raywarp cache info | prune --max-gb N   the disk caches (`raywarp.cache`)
  raywarp info                            the environment: GPU, driver, OptiX, Warp, kernels

Each subcommand imports its adapters and extras when it runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from raywarp.cli import cache, convert, info, kernels, prepare, record, view

# The subcommands in `--help` order; each module's `register` adds its parser.
_COMMANDS = (view, record, convert, prepare, kernels, cache, info)


def parser() -> argparse.ArgumentParser:
  """The `raywarp` parser: one subparser per command, its `run(args) -> int` as `run`."""
  p = argparse.ArgumentParser(prog="raywarp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
  for command in _COMMANDS:
    command.register(sub)
  return p


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the command line `argv` (default: the process's); its exit status."""
  args = parser().parse_args(argv)
  return int(args.run(args))
