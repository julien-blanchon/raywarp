"""Fields of NumPy structured arrays (host copies of Warp struct arrays, packed tables)."""

from __future__ import annotations

import numpy as np


def record_field(records: np.ndarray, name: str) -> np.ndarray:
  """Field `name` of structured array `records`: a view, same shape (written through with
  `record_field(r, name)[...] = value`)."""
  return records[name]  # pyrefly: ignore[bad-index]  # stub gap NS1: NumPy's stubs type no string key
