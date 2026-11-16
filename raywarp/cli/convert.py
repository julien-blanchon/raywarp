"""`raywarp convert IN.gltf OUT`: a glTF 2.0 asset (.gltf / .glb) to a USD stage (OUT .usd /
.usda / .usdc: `raywarp.formats.gltf.to_usd`, the stage `rw.formats.gltf.load` reads, for other
renderers) or to a visual-only MJCF model (OUT .xml: `raywarp.integrations.mujoco.from_gltf`,
its meshes and textures beside it). Needs the `gltf` extra (MJCF: the `mujoco` extra too).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from raywarp.cli.common import Sub

USD: Final = (".usd", ".usda", ".usdc")


def register(sub: Sub) -> None:
  """Adds `convert`."""
  p = sub.add_parser("convert", help="convert a glTF asset to USD (.usd/.usda/.usdc) or MJCF (.xml)", description=__doc__)
  p.add_argument("source", type=Path, help="the glTF asset (.gltf / .glb)")
  p.add_argument("out", type=Path, help="the output: .usd / .usda / .usdc, or .xml (MJCF)")
  p.add_argument("--scale", type=float, default=1.0, help="MJCF: scale the model (default: 1)")
  p.add_argument("--normalize", type=float, default=None, help="MJCF: largest extent in metres, centred, resting on z = 0")
  p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
  """Converts; prints what was written."""
  suffix = args.out.suffix.lower()
  if args.source.suffix.lower() not in (".gltf", ".glb"):
    raise SystemExit(f"raywarp convert: {args.source}: not a glTF asset (.gltf / .glb)")
  if suffix in USD:
    from raywarp.formats.gltf.to_usd import convert  # noqa: PLC0415  # the gltf extra

    print(f"{args.out}: {convert(args.source, args.out)}")
    return 0
  if suffix == ".xml":
    from raywarp.integrations.mujoco import from_gltf as to_mjcf  # noqa: PLC0415  # the gltf and mujoco extras

    print(to_mjcf.convert(args.source, args.out.parent, args.out.stem, args.scale, args.normalize))
    return 0
  raise SystemExit(f"raywarp convert: {args.out}: write .usd / .usda / .usdc (USD) or .xml (MJCF)")
