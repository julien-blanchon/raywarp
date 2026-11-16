"""What the `raywarp` subcommands share: the subparsers type, a WIDTHxHEIGHT option, the
settings."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
  import raywarp as rw

type Sub = argparse._SubParsersAction[argparse.ArgumentParser]

# `--settings` names: the default settings and the maximum-throughput field set.
SETTINGS: Final = ("default", "fast")


def size(text: str) -> tuple[int, int]:
  """A "WIDTHxHEIGHT" command-line value (positive integers).

  Raises:
    argparse.ArgumentTypeError: for another text.
  """
  try:
    width, height = (int(v) for v in text.lower().split("x"))
  except ValueError:
    raise argparse.ArgumentTypeError(f"size {text!r}: expected WIDTHxHEIGHT, e.g. 1024x576") from None
  if width < 1 or height < 1:
    raise argparse.ArgumentTypeError(f"size {text!r}: width and height must be positive")
  return width, height


def settings(name: str) -> rw.RenderSettings:
  """The `--settings` choice `name`: "default" (`RenderSettings()`) or "fast" (`rw.FAST` over
  it)."""
  import raywarp as rw  # noqa: PLC0415

  return rw.RenderSettings().replace(rw.FAST) if name == "fast" else rw.RenderSettings()
