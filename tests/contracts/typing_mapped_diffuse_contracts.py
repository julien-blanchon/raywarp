"""Static probes: optional texture width stays paired through host preparation."""

from typing import assert_type

import numpy as np

from raywarp.core.pipeline.upload.materials import MaterialTables, PreparedMaterials, TexAffines, scaled_normal_maps
from raywarp.scene.model.types import Rows


def _paired(table: PreparedMaterials) -> None:
  assert_type(table, MaterialTables[10] | MaterialTables[11] | MaterialTables[15])
  table.scaled_normal_maps()


def _mismatch(ids10: Rows[np.int32, 10], affine11: TexAffines[11]) -> None:
  scaled_normal_maps(ids10, affine11)  # pyrefly: ignore[bad-argument-type] - mismatched role dimensions must fail.
