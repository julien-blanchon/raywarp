"""Distance fog and haze (`settings.Atmosphere`, `Features.fog`): far surfaces fade to the fog
color.

The camera ray's radiance L seen through the fog is L T + (1 - T) C, with the transmittance T
MuJoCo's linear fog between `start` and `end` (1 - linear(d)) times Beer-Lambert haze
exp(-haze d) (`fog_amount`, unchanged from MuJoCo), and C the world's fog color
(display-relative: divided by the pixel's exposure, the world's times the camera lens's); a ray
that misses runs MuJoCo's MAXVAL far. Fused into the one blend per camera sample after the layers
are composited; every value is per world (`EnvironmentArrays.fog_color`, `fog_range`), so
randomising them never recompiles.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from raywarp._base.types import MAXVAL as MJ_MAXVAL
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.structs import EnvironmentArrays
from raywarp.core.kernels.abi.tables import batched1d
from raywarp.core.kernels.generic import specialize

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def fog_amount(
  # In:
  fog_range: wp.vec3,
  dist: float,
) -> float:
  """Fraction of the fog color seen through `dist` of air; `fog_range` (start, end, haze)."""
  start = fog_range[0]
  end = fog_range[1]
  linear = float(0.0)
  if end > start:
    linear = wp.clamp((dist - start) / (end - start), 0.0, 1.0)
  return 1.0 - (1.0 - linear) * wp.exp(-fog_range[2] * dist)


@specialize
def make_apply_fog(setting: Callable[[int, int, int], bool]):
  """apply_fog(env, worldid, radiance, instance_id, dist, exposure): `radiance` of a camera ray
  whose first hit is `instance_id` at `dist` (-1: a miss, as far as MuJoCo's MAXVAL) seen through
  the world's fog, at the pixel's `exposure` (world x camera lens). A tier build compiles the fog
  in and reads it at run time (`setting`, the build's `toggles.setting_on`, `toggles.FOG`: the
  radiance as it is off)."""

  @wp.func
  def apply_fog(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    radiance: wp.vec3,
    instance_id: int,
    dist: float,
    exposure: float,
  ) -> wp.vec3:
    """The ray's radiance through the fog (see the module doc)."""
    if not setting(env.params.rt.flags, env.bits2, wp.static(toggles.FOG)):
      return radiance
    d = dist
    if instance_id == -1:
      d = float(MJ_MAXVAL)
    t = 1.0 - fog_amount(batched1d(env.fog_range, worldid), d)
    fog = batched1d(env.fog_color, worldid) / exposure
    return radiance * t + fog * (1.0 - t)

  return apply_fog
