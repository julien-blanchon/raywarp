"""The `atmosphere` settings group: distance fog and haze that fade camera rays to a fog color
(`shade/environment/fog.py`)."""

from __future__ import annotations

import dataclasses
import math
from typing import final

from raywarp.settings.fields import is_display_rgb, rgb_tuple, runtime_field, validate


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Atmosphere:
  """Distance fog and haze between the camera and its first hit (off by default).

  A camera ray's color fades to the fog color with the distance d to its first hit (rays that
  hit nothing: fully fogged unless both terms are off): the fraction
  1 - (1 - linear(d)) * exp(-haze * d), with linear(d) MuJoCo's linear fog, 0 before `start` and
  1 after `end` (no linear fog when end <= start), and exp(-haze * d) Beer-Lambert haze. The
  values are runtime and per world (`PerWorld(env=["fog.color", "fog.range"])`: fog.color the
  linear color x exposure, `fog.range` (start, end, haze)); a domain-randomization knob. The fog
  is on where its values fog anything (`fog`) or declared per world; a tier build always compiles
  it, off at run time.

  Attributes:
    color: display (sRGB) color of a fully fogged pixel (R).
    start: distance where the linear fog starts [m] (R).
    end: distance where the linear fog is opaque [m] (R).
    haze: extinction coefficient of the exponential haze [1/m] (R).
  """

  color: tuple[float, float, float] = runtime_field((0.5, 0.5, 0.5))
  start: float = runtime_field(0.0, ge=0, lt=math.inf)
  end: float = runtime_field(0.0, ge=0, lt=math.inf)
  haze: float = runtime_field(0.0, ge=0, lt=math.inf)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "atmosphere")
    if not is_display_rgb(self.color):
      raise ValueError(f"atmosphere.color must be an sRGB triple in [0, 1], got {self.color!r}")
    object.__setattr__(self, "color", rgb_tuple(self.color))

  @property
  def fog(self) -> bool:
    """Whether these values fog anything: a linear fog (`end` > `start`) or a haze."""
    return self.end > self.start or self.haze > 0.0
