"""`Look`: one world's prepared display-transform constants (`Environment.look`), written each
frame by `grading.Grading.prepare` from the settings and the per-world grading / aberration
fields."""

from __future__ import annotations

from typing import Final

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# The look's per-world fields (`PerWorld(env=[...])`): name -> dtype; (kelvin, tint), saturation,
# lift, gamma, gain, chromatic aberration strength.
LOOK_FIELDS: Final = {
  "grading.white_balance": wp.vec2,
  "grading.saturation": float,
  "grading.lift": wp.vec3,
  "grading.gamma": wp.vec3,
  "grading.gain": wp.vec3,
  "sensor.aberration": float,
}


@wp.struct
class Look:
  """One world's imaging constants (`core/post/grading/transform.py`).

  Attributes:
    balance: white balance, linear Rec. 709 -> linear Rec. 709 (chromatic adaptation to D65).
    slope, offset, power: the ASC CDL in LogC (gain, lift, 1 / gamma).
    saturation: the saturation around Rec. 709 luminance.
    aberration: relative magnification difference of the red / blue images (deferred display).
  """

  balance: wp.mat33
  slope: wp.vec3
  offset: wp.vec3
  power: wp.vec3
  saturation: float
  aberration: float
