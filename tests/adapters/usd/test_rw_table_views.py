"""Host table views stay within the declared image rather than reading neighboring blob data."""

from __future__ import annotations

import numpy as np
import pytest

from raywarp.scene import SceneBuilder


@pytest.mark.parametrize("level", [-1, 1, 64])
def test_image_level_rejects_unstored_mips(level: int) -> None:
  """A single-level image cannot expose padding or a later image as its mip chain."""
  builder = SceneBuilder()
  builder.add_texture(np.full((1, 1, 4), 17, np.uint8))
  builder.add_texture(np.full((1, 1, 4), 209, np.uint8))
  images = builder.build(threads=1).images
  np.testing.assert_array_equal(images.levels, [1, 1])
  with pytest.raises(IndexError, match="mip"):
    images.level(0, level)


def test_declared_image_mips_remain_shared_views() -> None:
  """The final stored mip and negative image indexing preserve no-copy table semantics."""
  builder = SceneBuilder()
  builder.add_texture(np.full((4, 4, 4), 17, np.uint8))
  builder.add_texture(np.full((2, 2, 4), 209, np.uint8))
  images = builder.build(mips=True, threads=1).images
  first_last = images.level(0, int(images.levels[0]) - 1)
  second_last = images.level(-1, int(images.levels[-1]) - 1)
  assert first_last.shape == second_last.shape == (1, 1, 4)
  np.testing.assert_array_equal(first_last, 17)
  np.testing.assert_array_equal(second_last, 209)
  assert np.shares_memory(first_last, images.data)
  assert np.shares_memory(second_last, images.data)
  # The blob is read-only (as a loaded scene's): it cannot drift from the images' content digests.
  with pytest.raises(ValueError, match="read-only"):
    second_last[0, 0, 0] = 123
