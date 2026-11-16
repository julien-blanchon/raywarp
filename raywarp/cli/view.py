"""`raywarp view`: the web viewer (`raywarp.viewer.command`)."""

from __future__ import annotations

from raywarp.cli.common import Sub
from raywarp.viewer.command import add_arguments, run


def register(sub: Sub) -> None:
  """Adds `view`."""
  p = sub.add_parser("view", help="serve the web viewer of scene files (.wpbr, USD, glTF, MJCF)", description=__doc__)
  add_arguments(p)
  p.set_defaults(run=run)
