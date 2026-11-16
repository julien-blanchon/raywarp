"""`raywarp kernels info | fetch`: the prebuilt GPU kernels (`raywarp.kernels`).

info: what serves this machine (or `--arch`): the pinned bundle of the architecture, whether the
local store holds it, the remote, the driver, the download / compile switches.
fetch: the whole bundle of the architecture and the driver's OptiX module pack into the local store
(an offline mirror: copy the store, point RAYWARP_KERNELS at it, RAYWARP_KERNELS_DOWNLOAD=0).
"""

from __future__ import annotations

import argparse

from raywarp.cli.common import Sub


def register(sub: Sub) -> None:
  """Adds `kernels info` and `kernels fetch`."""
  p = sub.add_parser("kernels", help="the prebuilt GPU kernels: info, fetch", description=__doc__)
  actions = p.add_subparsers(dest="action", required=True)
  for name, doc in (("info", "what serves this machine's kernels"), ("fetch", "download a whole bundle (offline mirror)")):
    a = actions.add_parser(name, help=doc)
    a.add_argument("--arch", type=int, default=None, help="SM version, e.g. 86 (default: this machine's GPU)")
    if name == "fetch":
      a.add_argument("--driver", default="", help="NVIDIA driver of the OptiX module pack (default: this machine's)")
    a.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
  """Prints the info, or fetches."""
  from raywarp import kernels  # noqa: PLC0415  # imports Warp

  if args.action == "fetch":
    print(f"kernels in {kernels.fetch(args.arch, driver=args.driver)}")
    return 0
  found = kernels.info(args.arch)
  pinned = ", ".join(f"sm_{a}" for a in found.architectures) or "none (a source tree without pins)"
  print(f"architecture   {f'sm_{found.arch}' if found.arch else 'none (CPU: kernels compile)'}")
  print(f"bundle         {found.pinned or 'none pinned for it: kernels compile on first use'}")
  print(f"pinned         {pinned}")
  print(f"local store    {found.store or 'off (caching disabled)'}{' (holds it)' if found.local else ''}")
  print(f"remote         {found.remote}")
  print(f"driver         {found.driver or 'unknown (no OptiX module pack)'}")
  print(f"download       {'on' if found.policy.download else 'off (local store only)'}")
  print(f"compile        {'on' if found.policy.compile else 'off (a configuration not prebuilt raises)'}")
  return 0
