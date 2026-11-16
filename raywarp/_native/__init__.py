"""Native components bundled with raywarp, written at install time by its native build hook: the
OptiX SDK headers (the constant `OPTIX_INCLUDE_DIR`, NVIDIA/optix-dev;
the OptiX runtime ships with the driver) and the NVIDIA MDL SDK (`raywarp._native.mdl`).
`build()` is the hook's record: versions, platform and build key.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Final, TypedDict

DIR: Final = Path(__file__).resolve().parent
# optix.h and the rest of optix-dev's include/ (LICENSE.txt next to it).
OPTIX_INCLUDE_DIR: Final = DIR / "optix" / "include"


class Build(TypedDict):
  """What the build hook bundled."""

  mdl_sdk: str
  mdl_sdk_build: str
  optix: str
  optix_commit: str
  platform: str
  key: str


class NativeBuildError(RuntimeError):
  """The native components are not built (a source tree that was never installed)."""


@functools.cache
def build() -> Build:
  """The build hook's record (_native/BUILD.json).

  Raises:
    NativeBuildError: when the hook never ran on this tree.
  """
  path = DIR / "BUILD.json"
  if not path.is_file():
    raise NativeBuildError(
      f"raywarp's native components are not built ({path} is missing): install raywarp with pip or uv "
      "(`pip install .`, `uv sync`), which runs its native build hook"
    )
  record: Build = json.loads(path.read_text())
  return record
