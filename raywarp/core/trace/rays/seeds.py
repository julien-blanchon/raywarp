"""Independent random keys for render seeds, worlds and camera pixels."""

from __future__ import annotations

import warp as wp


@wp.func
def pixel_seed(
  # In:
  seed: wp.int32,
  world: wp.int32,
  pixel: wp.int32,
) -> wp.int32:
  """Hash the three coordinates separately, without a flattened world x pixels int32 index.

  Incrementing the frame seed cannot translate the previous frame's random field by one pixel.
  A finite 32-bit key can still collide by chance; no batch-size-dependent arithmetic wrap
  aliases entire worlds. The pixel index is the camera-buffer address, independent of launch order.
  """
  world_key = wp.int32(wp.rand_init(seed, world))
  return wp.int32(wp.rand_init(world_key, pixel))
