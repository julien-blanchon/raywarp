"""What a camera ray that leaves the scene sees: the environment map, or the background color.

`make_background` builds the miss shading of the camera rays (which may see a separate
background map). A sun cut out of the lighting map as a light
(`core.pipeline.gpu_scene.gpu_scene.EnvSuns`) is drawn back over it as its disc (`sun_disc`): the
map holds only the refill there. Reflections need no disc: the sun light's specular term is its
reflection (`lights.evaluators.source_widening`: a mirror shows the disc's size and radiance), and
their misses read the split-sum environment.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params, env_lookup
from raywarp.core.kernels.abi.tables import batched1d, exposure_at
from raywarp.core.kernels.generic import specialize
from raywarp.core.kernels.surface import env_lod, env_radiance
from raywarp.core.shade.environment import envmap

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# background -> radiance of a ray leaving the scene (`make_background`).
type Background = Callable[[EnvironmentArrays, Params, int, wp.vec3, float, bool], wp.vec3]


@wp.func
def sun_disc(
  # Model:
  env: EnvironmentArrays,
  # In:
  worldid: int,
  direction: wp.vec3,
  spread: float,
) -> wp.vec3:
  """The radiance of the world's map's sun disc along `direction` (zero outside it, or for a map
  without one), turned by the world's yaw and scaled by its intensity; its edge blended over the
  ray cone `spread` (no stair-stepped rim)."""
  e, yaw, _scale = env_lookup(env, worldid)
  row = env.env_sun[e]
  if row[3] <= 0.0:
    return wp.vec3(0.0)
  toward = envmap.rotate_yaw(wp.vec3(row[0], row[1], row[2]), yaw)
  angle = 2.0 * wp.asin(wp.min(0.5 * wp.length(direction - toward), 1.0))  # exact near 0
  cover = wp.clamp(0.5 + (row[3] - angle) / wp.max(spread, 1.0e-6), 0.0, 1.0)
  if cover <= 0.0:
    return wp.vec3(0.0)
  return env.env_sun_radiance[e] * (cover * batched1d(env.env_intensity, worldid))


@specialize
def make_background(f: Features):
  """background(env, p, worldid, direction, spread, seen) -> radiance of a ray leaving the scene."""
  env_background = f.env_background
  on = toggles.on(f)

  @wp.func
  def background(
    # Model:
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    direction: wp.vec3,
    spread: float,
    seen: bool,
  ) -> wp.vec3:
    """The environment seen along `direction` (cone `spread`: `env_radiance`), or the background
    color.

    `seen`: a camera ray (through translucent layers), which sees a separate background map
    (`Params.background_map`); a mirror ray reflects the lighting map, as the split-sum specular
    term does (Filament's IBL reflections). The lighting map shows its sun's disc (`sun_disc`).
    """
    if wp.static(env_background):
      if on(p.rt.flags, wp.static(toggles.ENV_BACKGROUND)):
        if seen and p.background_map >= 0:
          yaw = batched1d(env.env_yaw, worldid)
          return (
            envmap.sample_radiance(env.env_radiance[p.background_map], direction, yaw, env_lod(p, spread)) * p.background_scale
          )
        return env_radiance(env, p, worldid, direction, spread) + sun_disc(env, worldid, direction, spread)
    return env.background / exposure_at(env.exposure, worldid)

  return background
