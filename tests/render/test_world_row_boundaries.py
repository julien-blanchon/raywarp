"""Per-world appearance preparation must not silently discard authored world rows."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.pipeline.upload.build import upload
from raywarp.core.pipeline.worlds.tables import PerWorldTables


@pytest.fixture(scope="module")
def tables() -> Iterator[PerWorldTables]:
  builder = rw.SceneBuilder()
  builder.add_instance("sphere", material=builder.add_material(rw.Material()), size=(1, 0, 0))
  renderer = rw.Renderer(
    upload(builder.build(), device="cpu"),
    nworld=3,
    cameras=[rw.CameraSpec()],
    resolution=(8, 8),
    settings=rw.RenderSettings(outputs=rw.Outputs.of("depth"), performance=rw.settings.Performance(traversal="software")),
  )
  yield renderer._per_world
  renderer.close()


@pytest.mark.parametrize("rows", [0, 4])
def test_invalid_source_world_count_is_rejected(tables: PerWorldTables, rows: int) -> None:
  source = wp.ones((rows, 2), dtype=wp.vec4, device="cpu")
  with pytest.raises(ValueError, match="world rows"):
    tables._rows(source)


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_valid_shared_and_world_rows_keep_independent_copy(tables: PerWorldTables, rows: int) -> None:
  values = np.arange(rows * 8).astype(np.float32).reshape(rows, 2, 4)
  source = wp.array(values, dtype=wp.vec4, device="cpu")
  copied = tables._rows(source)
  assert copied.dtype == source.dtype and copied.device == source.device and copied.ptr != source.ptr
  assert copied.shape == (3, 2)
  np.testing.assert_array_equal(copied.numpy(), values[np.arange(3) % rows])  # K rows tile as world % K
  copied.zero_()
  np.testing.assert_array_equal(source.numpy(), values)


def test_rejected_count_never_downloads_source(tables: PerWorldTables, monkeypatch: pytest.MonkeyPatch) -> None:
  source = wp.ones(4, dtype=float, device="cpu")

  def no_readback(_: wp.array) -> None:
    raise AssertionError("invalid row count must fail before downloading source values")

  monkeypatch.setattr(wp.array, "numpy", no_readback)
  with pytest.raises(ValueError, match="world rows"):
    tables._rows(source)
