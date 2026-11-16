"""Newton's GL viewer look (ViewerRaywarpHeadless's default lighting), from ViewerGL's own values.

The lighting constants are read off ViewerGL (Newton 1.7 `newton/_src/viewer/gl/shaders.py` and
`gl/opengl.py`); the look then scales them (`LOOK_SUN_SCALE`, `LOOK_SKY_SCALE`) and tone maps
with `LOOK_TONE_MAPPER` (below):

* sun: direction `_GL_SUN[up_axis]` towards the light, color (1, 1, 1); the shader lights a
  surface with `albedo / pi * light_color * N.L * 3.0`, i.e. a directional light of 3 lux;
* ambient: `mix(ground, sky, (N.up + 1) / 2) * albedo * 0.7` with sky (0.8, 0.8, 0.85) and
  ground (0.3, 0.3, 0.35): a hemispheric environment of 0.7 x those radiances;
* exposure 1.6 (GL then applies Narkowicz's ACES fit): a linear exposure 2^-EV of 1.6, EV100
  log2(1 / 1.6);
* background: the sky gradient's lower color (40, 44, 55) / 255; ground planes in a checker of
  albedo x 0.7 (sRGB 0xDA = 0.7 linear) in 1 m squares.

Not reproduced: GL's gamma 2.2 (raywarp encodes sRGB), its percentage-closer shadow filter and the
metal-only specular ambient (raywarp's environment lighting is full PBR). With ViewerGL's values
alone, the dielectric specular raywarp adds (from the hemisphere and the sun) makes the frame much
brighter than ViewerGL's (mean |frame - ViewerGL| 56 / 255), and raywarp's ACES desaturates where
GL's fit keeps colors saturated. The look's scales (sun x 1.2, hemisphere x 0.4, PBR Neutral tone
mapping) bring ViewerRaywarpHeadless back to ViewerGL's appearance (~11 / 255). They are the
viewer's default look (what ViewerRaywarpHeadless shows without `lighting`), not a tuning of any
scene.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Final, final

from raywarp.integrations.newton.types import Hemisphere
from raywarp.scene.model.lights import Light

# Direction towards ViewerGL's sun per model up axis (opengl.py `_sun_dirs`).
_GL_SUN: Final = {0: (0.8, 0.2, -0.3), 1: (0.2, 0.8, -0.3), 2: (0.2, -0.3, 0.8)}
# The shader's direct-light scale (`* light_color * NdotL * 3.0`) as an illuminance [lux].
_GL_SUN_LUX: Final = 3.0
# The shader's ambient: `mix(ground_color, sky_color, ...) * albedo * 0.7`.
_GL_AMBIENT_SCALE: Final = 0.7
_GL_AMBIENT_SKY: Final = (0.8, 0.8, 0.85)
_GL_AMBIENT_GROUND: Final = (0.3, 0.3, 0.35)
# RendererGL.exposure.
_GL_EXPOSURE: Final = 1.6
# The look: the sun at 120 % and the hemispheric sky / ground at 40 % of ViewerGL's values, tone
# mapped with PBR Neutral, chosen so that ViewerRaywarpHeadless's frames match ViewerGL's appearance
# (module docstring).
LOOK_SUN_SCALE: Final = 1.2
LOOK_SKY_SCALE: Final = 0.4
LOOK_TONE_MAPPER: Final = "pbr_neutral"
GL_SKY_LOWER: Final = (40.0 / 255.0, 44.0 / 255.0, 55.0 / 255.0)
# sRGB bytes of the checker's albedo and albedo x 0.7 (0.7 linear is sRGB 0.854: 0xDA).
GL_CHECKER_COLORS: Final = (0xFF, 0xDA)


@final
@dataclasses.dataclass(frozen=True)
class GlLook:
  """The scene lighting and render settings of ViewerGL's look.

  Attributes:
    lights: the sun.
    ambient: the hemispheric ambient (an environment of absolute radiance, not shown).
    ev100: the exposure.
    settings: `RenderSettings.replace` keys (tone mapper, background).
  """

  lights: list[Light]
  ambient: Hemisphere
  ev100: float
  settings: Mapping[str, object]


def gl_look(up_axis: int) -> GlLook:
  """ViewerGL's look for a model whose up axis is `up_axis` (0, 1, 2)."""
  sun = _GL_SUN[up_axis]
  norm = math.hypot(*sun)
  light = Light.sun((-sun[0] / norm, -sun[1] / norm, -sun[2] / norm), illuminance=LOOK_SUN_SCALE * _GL_SUN_LUX)
  k = LOOK_SKY_SCALE * _GL_AMBIENT_SCALE
  sky = (k * _GL_AMBIENT_SKY[0], k * _GL_AMBIENT_SKY[1], k * _GL_AMBIENT_SKY[2])
  ground = (k * _GL_AMBIENT_GROUND[0], k * _GL_AMBIENT_GROUND[1], k * _GL_AMBIENT_GROUND[2])
  settings = {
    "environment.background": False,
    "environment.background_color": GL_SKY_LOWER,
    "post.tone_mapper": LOOK_TONE_MAPPER,
  }
  # EV100 of the common linear exposure 2^-EV equal to GL's.
  return GlLook([light], Hemisphere(sky, ground), math.log2(1.0 / _GL_EXPOSURE), settings)
