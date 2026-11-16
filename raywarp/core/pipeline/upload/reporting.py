"""Load-report helpers shared by the upload modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from raywarp.scene.model.types import Column

# `LoadReport.approx` / `drop`: (name, example, count).
type Record = Callable[[str, object, int], None]


def report_mask(record: Record, name: str, mask: Column[np.bool_], *, example: Callable[[int], object]) -> None:
  """Records `name` (`LoadReport.approx` or `drop`) once, counting the rows set in `mask`, with
  `example(i)` of the first such row i as the example; nothing when no row is set."""
  hits = np.flatnonzero(mask)
  if len(hits):
    record(name, example(int(hits[0])), len(hits))
