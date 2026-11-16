"""Lateral chromatic aberration of the deferred display (`settings.Sensor.aberration`).

`channel_positions`: lateral colour fringing, the red image magnified by 1 + a and the blue one by
1 - a around the image centre (the shift grows linearly with the radius, `a` the per-world
strength): the display samples each channel of the HDR at its own position.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def channel_positions(
  # In:
  x: int,
  y: int,
  width: int,
  height: int,
  aberration: float,
) -> wp.mat22:
  """Rows: the red and blue images' sample positions (pixel units) of pixel (x, y)."""
  c = wp.vec2(0.5 * float(width), 0.5 * float(height))
  d = wp.vec2(float(x) + 0.5, float(y) + 0.5) - c
  # The red image magnified by 1 + a: its pixel at d shows the scene at d / (1 + a).
  r = c + d / (1.0 + aberration)
  b = c + d / (1.0 - aberration)
  return wp.mat22(r[0], r[1], b[0], b[1])
