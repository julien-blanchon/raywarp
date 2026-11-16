"""The two switches of the prebuilt kernels: whether a missing file is downloaded and whether a
kernel the prebuilt bundle does not hold may compile (`raywarp.kernels.configure`).

- download (default on; `RAYWARP_KERNELS_DOWNLOAD`): off, nothing is fetched from the remote: the
  local store (`store.store_root`, a store filled ahead by `raywarp.kernels.fetch` or copied) is
  read as it is, and what it lacks is missing.
- compile (default on; `RAYWARP_KERNELS_COMPILE`): off, a configuration whose kernels are not
  prebuilt raises `KernelsNotPrebuilt` (naming the configuration and how to get its kernels)
  where it would have compiled: at the kernel routing (settings outside every prebuilt tier,
  `performance.kernels="scene"`, the CPU device), at a PTX the bundle lacks and at a first-frame
  Warp module that neither the bundle nor Warp's kernel cache holds.

`configure` overrides the environment for the process; an argument left None keeps the current
value.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from typing import Final, final

DOWNLOAD_ENV: Final = "RAYWARP_KERNELS_DOWNLOAD"
COMPILE_ENV: Final = "RAYWARP_KERNELS_COMPILE"
_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off"})
# How to get kernels that are not prebuilt, for the errors.
HOW: Final = (
  "allow compiling (raywarp.kernels.configure(compile=True), or unset RAYWARP_KERNELS_COMPILE), use settings a prebuilt "
  "tier serves (RenderSettings() or a field set over it: docs/settings.md), or compile it once on a machine that may and "
  "copy Warp's kernel cache"
)


class KernelsNotPrebuilt(RuntimeError):
  """A kernel would compile while compiling is off (`configure(compile=False)`)."""


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Policy:
  """The switches in force.

  Attributes:
    download: missing prebuilt files are downloaded from the remote.
    compile: kernels the prebuilt bundle does not hold may compile.
  """

  download: bool
  compile: bool


_LOCK: Final = threading.Lock()
# The `configure` overrides (None: the environment's value).
_OVERRIDE: Final[dict[str, bool]] = {}


def _env(name: str) -> bool:
  """The boolean of environment variable `name` (unset or empty: True).

  Raises:
    ValueError: a value that is not a boolean word.
  """
  raw = os.environ.get(name, "").strip().lower()
  if not raw or raw in _TRUE:
    return True
  if raw in _FALSE:
    return False
  raise ValueError(f"{name}={raw!r}: expected 1 / 0 (true / false, yes / no, on / off)")


def current() -> Policy:
  """The switches in force: `configure`'s, else the environment's."""
  with _LOCK:
    download = _OVERRIDE.get("download")
    compile_ = _OVERRIDE.get("compile")
  return Policy(
    download=_env(DOWNLOAD_ENV) if download is None else download,
    compile=_env(COMPILE_ENV) if compile_ is None else compile_,
  )


def configure(*, download: bool | None = None, compile: bool | None = None) -> Policy:
  """Sets the switches for this process (None: unchanged); the switches now in force."""
  with _LOCK:
    if download is not None:
      _OVERRIDE["download"] = download
    if compile is not None:
      _OVERRIDE["compile"] = compile
  return current()


def reset() -> None:
  """Drops the `configure` overrides: the environment decides again."""
  with _LOCK:
    _OVERRIDE.clear()


def refuse(what: str) -> None:
  """Raises `KernelsNotPrebuilt` for `what` (the configuration or kernel that would compile) when
  compiling is off; returns otherwise."""
  if not current().compile:
    raise KernelsNotPrebuilt(f"{what} is not prebuilt and compiling is off ({COMPILE_ENV} / configure(compile=False)): {HOW}")
