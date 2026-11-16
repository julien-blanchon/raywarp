"""Prebuilt GPU kernels (`rw.kernels`): the switches, the environment check, a whole-bundle fetch.

The plain wheel carries no kernels: on first use, a renderer on a CUDA device downloads what it
needs of its architecture's bundle (`kernels.json`, pinned per architecture: sm_80, 86, 89, 90,
120) and of the running driver's OptiX module pack into the local store, from the release's
public bucket, every file checked against the installed package.

  rw.kernels.configure(download=False)   # never fetch: the local store only (offline machines)
  rw.kernels.configure(compile=False)    # a configuration that is not prebuilt raises
  rw.kernels.fetch()                      # this GPU's whole bundle and pack (an offline mirror)
  rw.kernels.info()                       # what serves this machine

Environment: RAYWARP_KERNELS_DOWNLOAD / RAYWARP_KERNELS_COMPILE (1 / 0, default 1),
RAYWARP_KERNELS (the local store), RAYWARP_KERNELS_URL (a mirror of the remote).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import final

from raywarp.core.build.kernel_cache import policy
from raywarp.core.build.kernel_cache import store as _store
from raywarp.core.build.kernel_cache.policy import KernelsNotPrebuilt, Policy

__all__ = ["Info", "KernelsNotPrebuilt", "Policy", "configure", "current", "detect", "fetch", "info", "pinned_architectures"]


def configure(*, download: bool | None = None, compile: bool | None = None) -> Policy:
  """Sets the prebuilt-kernel switches for this process (None: unchanged; the defaults come from
  RAYWARP_KERNELS_DOWNLOAD / RAYWARP_KERNELS_COMPILE, on when unset); the switches in force.

  download: False never fetches anything: the local store (filled ahead by `fetch` or copied, at
    RAYWARP_KERNELS) is read as it is.
  compile: False makes a configuration whose kernels are not prebuilt raise `KernelsNotPrebuilt`
    (naming it and how to get its kernels) instead of compiling: settings outside every prebuilt
    tier, `performance.kernels="scene"`, the CPU device, a kernel the bundle lacks.
  """
  return policy.configure(download=download, compile=compile)


def current() -> Policy:
  """The switches in force."""
  return policy.current()


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Info:
  """What serves a device's kernels.

  Attributes:
    arch: the device's SM version (0: the CPU, whose kernels always compile).
    pinned: the content key of the architecture's pinned bundle (None: this release has none).
    architectures: the SM versions this release pins a bundle for.
    store: the local store (None: caching off).
    local: the pinned bundle's file list is in the store (it was used or fetched here).
    remote: where missing files come from.
    driver: the NVIDIA driver ("" unknown: no OptiX module pack).
    policy: the switches in force.
  """

  arch: int
  pinned: str | None
  architectures: tuple[int, ...]
  store: Path | None
  local: bool
  remote: str
  driver: str
  policy: Policy


def pinned_architectures() -> tuple[int, ...]:
  """The SM versions the installed release pins a bundle for (`kernels.json`)."""
  return tuple(sorted(int(a) for a in _store.pins()))


def detect(device: str = "cuda") -> int:
  """The SM version of Warp device `device` (0: the CPU, or no CUDA device)."""
  import warp as wp  # noqa: PLC0415  # initialises Warp

  if device.startswith("cuda") and not wp.get_cuda_device_count():
    return 0
  found = wp.get_device(device)
  return int(found.arch) if found.is_cuda else 0


def info(arch: int | None = None) -> Info:
  """What would serve a device of SM version `arch` (default: `detect()`; 0: the CPU); reads
  nothing remote."""
  from raywarp.core.build.kernel_cache import prebuilt  # noqa: PLC0415  # imports Warp

  arch = detect() if arch is None else arch
  found = _store.pin(arch) if arch else None
  root = _store.store_root()
  local = found is not None and root is not None and (root / f"sm{arch}" / found[0] / _store.FILES).is_file()
  return Info(
    arch=arch,
    pinned=None if found is None else found[0],
    architectures=pinned_architectures(),
    store=root,
    local=local,
    remote=_store.remote(),
    driver=prebuilt.driver_version(),
    policy=policy.current(),
  )


def fetch(arch: int | None = None, *, driver: str = "") -> Path:
  """Downloads the whole pinned bundle of SM version `arch` (default: `detect()`) and driver
  `driver`'s OptiX module pack (default: this machine's driver) into the local store: an offline
  mirror (copy the store, point RAYWARP_KERNELS at it, `configure(download=False)`). No GPU needed.
  The bundle's directory.

  Raises:
    RuntimeError: no pinned or usable bundle for `arch`, files that could not be fetched, or
      downloads off.
  """
  from raywarp.core.build.kernel_cache import prebuilt  # noqa: PLC0415  # imports Warp

  arch = detect() if arch is None else arch
  if _store.pin(arch) is None:
    have = ", ".join(f"sm_{a}" for a in pinned_architectures()) or "none"
    raise RuntimeError(f"this release pins no kernels bundle for sm_{arch} (pinned: {have})")
  return prebuilt.prefetch(arch, driver=driver)
