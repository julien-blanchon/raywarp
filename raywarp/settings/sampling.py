"""The `aa` settings group: how many samples a pixel takes (anti-aliasing)."""

from __future__ import annotations

import dataclasses
from typing import final

from raywarp.settings.fields import runtime_field, setting, validate


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class AntiAliasing:
  """Sampling of the pixel. Every frame renders on its own (no temporal accumulation): the sampled
  effects (area-light and sun-disc shadow targets) are fixed per pixel, deterministic per frame.

  Attributes:
    spp: stratified sub-pixel samples per axis, spp x spp per pixel (C).
    post: post anti-aliasing of the display image (R): FXAA 3.11's quality filter (Lottes
      2011) on the final color across the geometric edges only (silhouettes, thin wires, alpha
      cut-outs, creases: where the depth output is not one plane), at one sample per pixel;
      texture and shadow edges stay as rendered. Runs in the sensor pass
      (`core/post/sensor/effects.py`, before its other effects), needs `outputs.depth` and
      `outputs.rgb` or `outputs.luminance`; off it costs nothing.

  Specular anti-aliasing (roughness widened by the normal variance across the pixel footprint)
  is always on.
  """

  spp: int = setting(1, ge=1)
  post: bool = runtime_field(default=False)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "aa")
