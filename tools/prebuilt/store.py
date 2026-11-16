"""A fill's bundles (and module packs) as a kernel store of their own
(`raywarp/core/build/kernel_cache/store.py`), for the fill tooling and the benchmarks: a directory
of `tools/prebuilt/build.py --out` outputs (`raywarp_kernels/sm<arch>/`,
`raywarp_optix_modules/sm<arch>-drv<driver>/`) linked into `<store>/sm<arch>/fill/` (its
optix/<driver>/ the packs), then selected by RAYWARP_KERNELS and RAYWARP_KERNELS_BUNDLE=fill: the
renderer reads those complete bundles as they are, nothing is fetched.
"""

from __future__ import annotations

import os
from pathlib import Path

from raywarp.core.build.kernel_cache import prebuilt
from raywarp.core.build.kernel_cache import store as kernel_store

NAME = "fill"


def use(dist: Path, store: Path) -> dict[str, str]:
  """Links every bundle of `dist` (and its packs) into `store`; the environment that selects them
  (also set in this process, for its children too).

  Raises:
    SystemExit: `dist` holds no bundle.
  """
  bundles = sorted((dist / "raywarp_kernels").glob("sm*"))
  if not bundles:
    raise SystemExit(f"no raywarp_kernels/sm<arch> bundle in {dist}")
  for bundle in bundles:
    target = store / bundle.name / NAME
    target.mkdir(parents=True, exist_ok=True)
    for entry in bundle.iterdir():
      if not (target / entry.name).exists():
        (target / entry.name).symlink_to(entry.resolve())
    for pack in sorted((dist / "raywarp_optix_modules").glob(f"{bundle.name}-drv*")):
      link = target / "optix" / pack.name.removeprefix(f"{bundle.name}-drv")
      link.parent.mkdir(exist_ok=True)
      if not link.exists():
        link.symlink_to(pack.resolve())
  env = {kernel_store.STORE_ENV: str(store), prebuilt.BUNDLE_ENV: NAME}
  os.environ.update(env)
  return env
