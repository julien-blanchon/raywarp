"""`raywarp info`: the environment check: the GPU (name, compute capability, RT cores), the NVIDIA
driver, OptiX, the Warp release, the traversal `traversal="auto"` picks, the prebuilt kernels of the
architecture and whether their remote answers (`--offline`: not asked).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from raywarp.cli.common import Sub


def register(sub: Sub) -> None:
  """Adds `info`."""
  p = sub.add_parser("info", help="check the environment: GPU, driver, OptiX, Warp, kernels", description=__doc__)
  p.add_argument("--device", default="cuda", help="Warp device (default: cuda)")
  p.add_argument("--offline", action="store_true", help="do not ask the kernels' remote")
  p.set_defaults(run=run)


def reachable(arch: int) -> str:
  """Whether the pinned bundle of SM version `arch` can be fetched: its file list downloaded into a
  scratch directory ("reachable", else why not)."""
  from raywarp.core.build.kernel_cache import store  # noqa: PLC0415

  pinned = store.pin(arch)
  if pinned is None:
    return "no bundle pinned for this architecture"
  content, sha = pinned
  with tempfile.TemporaryDirectory() as tmp:
    try:
      store.listing(Path(tmp), f"{store.remote()}/sm{arch}/{content}", sha)
    except store.FetchError as e:
      return f"NOT reachable: {e}"
  return "reachable"


def run(args: argparse.Namespace) -> int:
  """Prints the check."""
  import warp as wp  # noqa: PLC0415

  from raywarp import kernels  # noqa: PLC0415
  from raywarp.core.pipeline.platform.device import has_rt_cores  # noqa: PLC0415

  arch = kernels.detect(args.device)
  print(f"warp           {wp.config.version}")
  if not arch:
    print("device         CPU (no CUDA device): the software traversal; kernels compile on first use")
    return 0
  device = wp.get_device(args.device)
  found = kernels.info(arch)
  rt = has_rt_cores(arch)
  optix = _optix_reason()
  print(f"device         {device.alias}: {device.name}, compute capability {arch // 10}.{arch % 10} (sm_{arch})")
  print(f"driver         {found.driver or 'unknown'}")
  print(f"RT cores       {'yes' if rt else 'no'}")
  print(f"OptiX          {'available' if optix is None else f'unavailable: {optix}'}")
  print(f"traversal      auto -> {'optix' if rt and optix is None else 'software'}")
  print(f"kernels        {f'sm_{arch} bundle {found.pinned}' if found.pinned else 'none pinned: compile on first use'}")
  if found.pinned:
    print(f"  local store  {found.store}{' (holds it)' if found.local else ' (fetched on first use)'}")
    remote = "not asked (--offline)" if args.offline else reachable(arch)
    print(f"  remote       {found.remote}: {remote if found.policy.download else 'downloads off'}")
  return 0


def _optix_reason() -> str | None:
  """Why OptiX cannot run on this machine (None: it can): the driver's OptiX ABI and the bundled
  headers."""
  from raywarp.core.trace.optix import api  # noqa: PLC0415
  from raywarp.core.trace.optix.device import intrinsics  # noqa: PLC0415

  reason = api.unavailable_reason()
  if reason is not None:
    return reason
  inc = intrinsics.include_dir()
  return (
    f"the bundled OptiX SDK headers are missing ({intrinsics.INCLUDE_DIR})" if inc is None else intrinsics.headers_reason(inc)
  )
