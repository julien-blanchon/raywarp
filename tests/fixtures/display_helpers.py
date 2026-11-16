"""`display_transform` on the host for tests: the device tone mapper itself, run on the CPU."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from raywarp.core.post.grading.color import display_transform

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _apply_display_transform(
  # In:
  tone_mapper: int,
  color_in: wp.array1d[wp.vec3],
  # Out:
  display_out: wp.array1d[wp.vec3],
):
  """`display_transform` of each color."""
  i = wp.tid()
  display_out[i] = display_transform(color_in[i], tone_mapper)


def apply_display_transform(colors: npt.ArrayLike, tone_mapper: int) -> np.ndarray:
  """`display_transform` of linear colors (..., 3), evaluated by the device function (CPU)."""
  c = np.asarray(colors, dtype=np.float32)
  with wp.ScopedDevice("cpu"):
    x = wp.array(c.reshape(-1, 3), dtype=wp.vec3)
    y = wp.empty_like(x)
    wp.launch(_apply_display_transform, dim=x.shape[0], inputs=[tone_mapper, x], outputs=[y])
    return y.numpy().reshape(c.shape)
