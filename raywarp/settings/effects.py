"""The settings groups of the effects beyond direct light: `reflections` (traced reflection rays),
`transparency` (translucent layers) and `gi` (the irradiance probe volume)."""

from __future__ import annotations

import dataclasses
import math
from typing import Final, Literal, final

from raywarp.settings.fields import Box, ReflectionShadowMode, Tri, rebuild_field, runtime_field, setting, validate

# Coordinates of a `gi.bounds` corner.
_XYZ: Final = 3

# When a randomised world's GI is re-lit by rays (`GI.relight`).
type GIRelight = Literal["on_change", "on_request"]


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Reflections:
  """Traced specular reflections (core/kernels/reflections.py).

  Attributes:
    enabled: traced reflections: for surfaces smoother than `max_roughness`, one ray along the
      GGX dominant direction that replaces the environment's specular term (texture mips at the
      hit chosen by a ray cone widened by the roughness); off: no reflection ray (C). A material's
      mirror reflectance (MuJoCo `reflectance`) is its specular F0 whatever this: its reflection
      is this ray.
    max_roughness: perceptual roughness above which no reflection ray is traced; the traced
      reflection fades to the environment's specular term from half of it, and contributions of
      a faded specular albedo below `core.kernels.reflections.MIN_WEIGHT` fade out (no ray) (R).
    shadows: shadow rays at the reflection hits: "sun" (directional lights only, e.g. `env_sun`:
      what keeps sunlight out of indoor reflections; the other lights unshadowed) or "off" (every
      light unshadowed) (C).
    max_exact: at reflection hits, shade at most this many lights exactly (the strongest), the
      rest without a highlight (`lighting.max_exact`); None: every light of the cell (C).
  """

  enabled: bool = True
  max_roughness: float = runtime_field(0.5, ge=0, le=1)
  shadows: ReflectionShadowMode = "off"
  max_exact: int | None = setting(2, ge=1, le=16)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "reflections")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Transparency:
  """Translucent layers.

  Attributes:
    enabled: composite translucent layers front to back; "auto" = on when the scene has a
      translucent instance, material or texture, or `instance.color` / `material.base_color` are
      declared per world (C).
    max_layers: translucent layers a camera ray composites (a closed pane's exit face: only its
      coverage again, no layer; a solid's exit face: its medium's absorption, no layer); what is
      behind the last one is not traced, the background along the ray (environment or background
      color) shows through its transmittance (R: the layer loops' bound at run time; a change
      re-bakes the GI probes, whose rays follow the layers too).
  """

  enabled: Tri = "auto"
  max_layers: int = runtime_field(4, rebuild="gi", ge=1)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "transparency")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class GI:
  """Irradiance probe GI (core/shade/gi/).

  Domain randomization: worlds whose lights, materials, appearance and environment are equal share
  one set of probe maps; randomised worlds get their own (K stacked sets for K variants, one per
  world for continuous randomisation), found from the data and tracked every frame
  (core/shade/gi/groups.py). The grid, placement and distance maps are shared. The ambient
  cube holds every set: its texel budget lowers its resolution for many sets, and the renderer
  raises a ValueError when even one texel per probe spacing does not fit.

  Attributes:
    enabled: replace the diffuse ambient (environment SH / constant fill) by an irradiance probe
      grid (DDGI-style: sky visibility, bounce light, emissive surfaces lighting the scene; per
      group of equal worlds) and occlude the environment's specular term by the probes' sky
      visibility (the default: against path-traced references it cut the mean FLIP of
      window-lit, closed and open tabletop rooms by 47-83 % for no measurable frame cost); off:
      split-sum IBL, the environment unoccluded (C; rebuilds the volume).
    spacing: target probe spacing [m]; None (the default): derived from the scene, the median
      over survey points of their harmonic mean free distance to the geometry, divided by
      `shade.gi.volume.FREE_PATH_CELLS`: about 0.5 m in a furnished room (where a fixed 0.5 m cut
      the mean FLIP against Mitsuba of a closed workcell 0.144 -> 0.092, a window-lit room 0.120 ->
      0.098 from 1 m), coarser in open scenes, and the same image at any unit scale (mm to km)
      (R; rebuilds the volume). Large scenes reach `max_probes` first and get a coarser spacing.
    max_probes: probe budget; the spacing grows to fit it (R; rebuilds the volume).
    bounds: ((x0, y0, z0), (x1, y1, z1)) box covered by the probes; None = world 0's static
      instance bounds without planes, padded by one spacing, at least 2 spacings per axis (R;
      rebuilds the volume).
    rays: rays per probe per update (256: a window-lit room's mean FLIP 0.120 -> 0.080 from 64,
      the small opening's noise; a bake-time cost only; the bake traces them once); the static
      bake (distance moments) traces `shade.gi.volume.STATIC_RAYS` per probe in sets of `rays` (R;
      rebuilds the volume).
    normal_bias: lookup offset along the normal, fraction of the probe spacing (R; rebuilds).
    update_every: 0 = bake once at the first render (frozen); N = one more update every N
      renders (R).
    bake_updates: bounces of the initial bake: its first update traces the rays, each other one
      lights their hits again from the previous one (no ray traced) (R).
    relight: when a world of continuous randomisation (one shared volume tinted per world) whose
      GI inputs changed is re-lit by rays: "on_change" (the default) on the frame of the change;
      "on_request" only once `Renderer.request_gi_relight` names it (e.g. at each episode reset of
      a dataset generator that randomises every frame), keeping its tint until then. Changes of
      the lights' colours, intensities, activity and the environment intensity are recombined
      without rays in both (R).
  """

  enabled: bool = rebuild_field(default=True, rebuild="gi")
  spacing: float | None = runtime_field(None, rebuild="gi", gt=0, lt=math.inf)
  max_probes: int = runtime_field(32768, rebuild="gi", ge=1)
  bounds: Box | None = runtime_field(None, rebuild="gi")  # noqa: RUF009  # a dataclasses.field
  rays: int = runtime_field(256, rebuild="gi", ge=1)
  normal_bias: float = runtime_field(0.1, rebuild="gi", ge=0, lt=math.inf)
  update_every: int = runtime_field(0, ge=0)
  bake_updates: int = runtime_field(8, ge=1)
  relight: GIRelight = runtime_field("on_change")  # noqa: RUF009  # a dataclasses.field

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "gi")
    if self.bounds is not None:
      object.__setattr__(self, "bounds", _box(self.bounds))


def _box(bounds: Box) -> Box:
  """`gi.bounds` as float corners ((x0, y0, z0), (x1, y1, z1)).

  Raises:
    ValueError: for anything but two finite corners of 3 numbers with x0 < x1, y0 < y1, z0 < z1.
  """
  try:
    lo, hi = (tuple(float(v) for v in c) for c in bounds)  # any user value: malformed ones raise here
  except (TypeError, ValueError):
    lo = hi = ()
  finite = all(math.isfinite(v) for v in (*lo, *hi))
  if not (finite and len(lo) == len(hi) == _XYZ and all(a < b for a, b in zip(lo, hi, strict=True))):
    raise ValueError(f"gi.bounds must be finite ((x0, y0, z0), (x1, y1, z1)) with x0 < x1 ..., got {bounds!r}")
  return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])
