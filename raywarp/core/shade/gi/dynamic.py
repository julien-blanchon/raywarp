"""Dynamic lights (`PerWorld.dynamic_lights`) and the GI: they light the frame directly (direct
light and shadows, every frame, wherever they are), never enter the probes, and never re-light
the volume.

`StaticLights.of` gives the probe rays a light struct whose dynamic lights are switched off (a
copy of the frame's packed lights and hot records with `active` cleared, written on the device
every time the probes read the lights: graph-capturable); `groups.digest` skips the dynamic
lights' rows, so moving, recolouring or switching one changes no group or tint.
"""

from __future__ import annotations

from typing import final

import numpy as np
import warp as wp

from raywarp.core.shade.lights import punctual as lights_lib
from raywarp.core.shade.lights.punctual import Light, LightHot, Lights

wp.set_module_options({"enable_backward": False})


@wp.kernel
def hide_dynamic(
  # In:
  packed: wp.array2d[Light],
  dynamic: wp.array1d[wp.int32],
  # Out:
  out: wp.array2d[Light],
  hot: wp.array2d[LightHot],
):
  """Light `li` of row `row` as packed, switched off when it is dynamic, and its hot record."""
  row, li = wp.tid()
  light = packed[row, li]
  if li < dynamic.shape[0] and dynamic[li] != 0:
    light.active = False
  out[row, li] = light
  hot[row, li] = lights_lib.hot_of(light)


@final
class StaticLights:
  """The probes' view of a frame's lights: its dynamic lights off (see the module doc).

  Args:
    nlight: the scene's lights; dynamic: the dynamic light indices; device: the scene's.
  """

  def __init__(self, nlight: int, dynamic: tuple[int, ...], device: wp.Device) -> None:
    """Keeps the (nlight,) dynamic mask on the device; the copies are allocated on first use."""
    mask = np.zeros(max(nlight, 1), np.int32)
    mask[list(dynamic)] = 1
    self.ids = dynamic
    self.mask = wp.array(mask, dtype=wp.int32, device=device)
    self._lights: wp.array[Light, [int, int]] | None = None
    self._hot: wp.array[LightHot, [int, int]] | None = None

  def of(self, lit: Lights) -> Lights:
    """`lit` (a fresh struct of the frame, modified in place and returned) with its dynamic
    lights off; unchanged without dynamic lights."""
    if not self.ids:
      return lit
    packed = lit.lights
    if self._lights is None or self._hot is None or self._lights.shape != packed.shape:
      self._lights = wp.empty(packed.shape, dtype=Light, device=packed.device)
      self._hot = wp.empty(packed.shape, dtype=LightHot, device=packed.device)
    wp.launch(
      hide_dynamic, dim=packed.shape, inputs=[packed, self.mask], outputs=[self._lights, self._hot], device=packed.device
    )
    lit.lights, lit.hot = self._lights, self._hot
    return lit
