"""The settings groups of the lights: `shadows` (shadow rays), `lighting` (`Lights`: how lights
are shaded; the scene's light rig is `rw.Lighting`) and `environment` (image-based lighting and
the background)."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, final

from raywarp.settings.fields import Tri, is_display_rgb, is_rgb, rgb_tuple, runtime_field, setting, validate


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Shadows:
  """Shadow rays of the lights.

  Attributes:
    enabled: trace shadow rays for shadow-casting lights (C; off collapses the rest).
    samples: shadow rays per light; > 1 gives soft shadows from the light radius (C). Area lights
      and directional lights with an angular size (UsdLux DistantLight `angle`, an
      environment sun's disc: `SceneOptions.env_sun`) get soft shadows from points on the light:
      a fixed low-discrepancy set shifted per pixel and light (one sample: the hard shadow of its
      center; deterministic per pixel and seed).
    max_casters: shadow rays only for the K strongest shadow casters per shading point (None:
      every light; at most 16) (C). The casters beyond K: scaled by the mean of 1 and the traced
      ones' contribution-weighted visibility, the strongest of them traced too and blended in by
      its margin to the weakest traced caster (one more shadow ray; deterministic and continuous
      across changes of the traced set: where the strongest lights are blocked, e.g. under a desk
      below ceiling panels, the others still bring half their light instead of none).
    translucent: shadow rays of every light through translucent geometry (blend-mode materials,
      e.g. window glass, curtains): "transmittance" attenuates them by each translucent
      instance's RGB transmittance, base color x (1 - opacity) (the opacity map's minimum;
      colored glass tints the light), once per instance; "clear" lets them through unattenuated
      (as RTX RealTimePathTracing does for window glass), "off": every instance is opaque to
      shadow rays (C). One terminating walk per shadow ray (OptiX: the casters' any-hit programs
      multiply the transmittance in): a ray that meets no translucent caster costs a plain one.
    area_rays: soft shadows of the large, near area lights by a pass after the render (0: off,
      the default; `shade/shadows/area.py`): per pixel and area light subtending 0.05 rad or
      more (at most `max_casters` of them, by their share of the light), this many shadow rays to
      points of the light, filtered over the pixel's surface, replace the centre ray's hard shadow
      in that light's shading (every lobe: the pass shades the pixel's surface as the render
      does). Needs LTC area lights (`lighting.area_lights="ltc"`) and one sample per pixel
      (`aa.spp=1`). Off it costs nothing (the render's kernels are those of the setting off); on,
      the render writes the position (float32) and normal outputs it reads (those not requested)
      and the deferred display shows the image. Run time (R): turning it on creates (OptiX) the
      modules of the new output word, or compiles (software) its megakernel, once.
  """

  enabled: bool = True
  samples: int = setting(1, ge=1)
  max_casters: int | None = setting(4, ge=1, le=16)
  translucent: Literal["off", "transmittance", "clear"] = "transmittance"
  area_rays: int = runtime_field(0, ge=0, le=64)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "shadows")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Lights:
  """How the scene's lights are shaded.

  Attributes:
    area_lights: "point" shades every light as a point / spot (area lights as their point-light
      equivalent, soft shadows from `light.radius`); "ltc" shades lights with a `light.shape`
      (rect, disk, sphere, tube) as area lights: LTC specular and diffuse, shadows from
      `shadows.samples` points on the light (compiled in when the scene has area lights or
      `light.shape` is declared per world) (C).
    light_grid: shade only the lights visible from the point's cell of a static light visibility
      grid built on the first render; "auto" = on from `resolve.GRID_MIN_LIGHTS` (8) lights,
      where culling pays for the grid's lookup (C).
    light_grid_cell: cell size of the light visibility grid [m], a minimum: a large scene's cell
      grows to bound the grid's cells and its bake's cost (lattice points x lights,
      `light_grid.MAX_TESTS`) (R; rebuilds the grid).
    camera_light: camera-attached directional light, unshadowed; "auto" = as the scene authored it
      (its headlight), True: on (its color the scene's `Lighting.headlight` / MuJoCo
      `visual/headlight` diffuse, 0.4 grey without one) (C). Its intensity is display-relative to
      the reference exposure (`exposure.render_ev100`), which MuJoCo scenes and `Lighting` are
      calibrated for: in a scene in physical units (USD, Bitterli, a real interior) it is either
      negligible or blows the image out; use `fill` there.
    fill: a constant fill added to every lit surface, display-relative: a white diffuse
      surface (ambient occlusion 1) shows `fill` x `fill_color` of display white at the
      exposure its world is displayed at (under automatic exposure camera 0's metered one, from
      the frame after the world's first metering: the first frame after a reset has no fill and
      is metered without it), whatever the scene's light units. Diffuse only: base color x
      (1 - metalness) x ambient occlusion x fill (HyperLux's ambient), on top of GI, the
      environment and the lights (it replaces none of them, and the GI probes do not see it: a
      change re-bakes nothing). Not the scene's `Lighting.ambient` (an authored ambient light). 0
      (the default): none. A fill against black shadows, not light (GI brings the bounce light):
      at most 0.1, since a fill near middle grey (0.18) is what the automatic exposure would
      expose for (R).
    fill_color: linear RGB tint of `fill` (R).
    emissive_lights: emissive surfaces are also lights: the rect lights that `upload` made of
      flat rectangular emissive panels (`GpuScene.light_instance`,
      `raywarp.scene.lighting.emissive`) are active; off, the panels only glow where they are
      seen (C).
    max_exact: shade at most this many lights exactly per point (BRDF or LTC, shadow rays and
      the shadow budget `shadows.max_casters` among them): the strongest by a cheap estimate of
      their irradiance (the point-light form, no shadow ray); the others' estimated irradiance
      unshadowed, through the diffuse lobe and the specular lobe's albedo spread evenly (no
      highlight); None: every light of the point's cell exactly (`shade/lights/sources.py`) (C).
    visible_lights: camera rays see the emitting surface of the area lights flagged
      `Light.visible_in_primary_ray` (rect, disk, sphere and cylinder lights without geometry of
      their own; USD: Omniverse's `visibleInPrimaryRay`, false by default;
      `shade/lights/visible.py`); off: no light is seen (R).
    contact_ao: strength of the contact ambient occlusion of the dynamic objects (0: off): an
      analytic, deterministic occlusion of each pixel's indirect light (GI, environment, ambient;
      never the lights' direct light) by box and sphere proxies of the moving instances (a
      robot's links, grasped or randomised objects), which the GI probes and the environment's
      visibility do not see; applied by the deferred display, which it turns on
      (`shade/ao/contact.py`), from a planar depth output (a radial one raises). 0 (off) by
      default and then free (its code is in a display kernel variant of its own); on, its cost
      grows with the moving instances near what the pixels see (none: only the depth output). It
      reconstructs view-space points as a pinhole camera's (an orthographic, fisheye or distorted
      camera gets an approximate occlusion). Run time (R), but turning it on without a requested
      depth adds that output: on the software path a megakernel of the new output word compiles
      once, on OptiX a module of it is created.
  """

  area_lights: Literal["point", "ltc"] = "ltc"
  light_grid: Tri = "auto"
  light_grid_cell: float = runtime_field(1.0, rebuild="light_grid", gt=0, lt=math.inf)
  camera_light: Tri = "auto"
  fill: float = runtime_field(0.0, ge=0, le=0.1)
  fill_color: tuple[float, float, float] = runtime_field((1.0, 1.0, 1.0))
  emissive_lights: bool = True
  max_exact: int | None = setting(4, ge=1, le=16)
  visible_lights: bool = runtime_field(True)
  contact_ao: float = runtime_field(0.0, ge=0, le=1)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "lighting")
    if not is_rgb(self.fill_color) or min(self.fill_color) < 0:
      raise ValueError(f"lighting.fill_color must be three finite, non-negative linear values, got {self.fill_color!r}")
    object.__setattr__(self, "fill_color", rgb_tuple(self.fill_color))


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Environment:
  """Image-based lighting and what rays that hit nothing show.

  Attributes:
    enabled: image-based lighting from the scene's environment maps; "auto" = when the scene has
      maps that light it (C).
    background: rays that hit nothing show the environment map (C).
    background_color: display (sRGB) color of rays that hit nothing without a background map (R).
  """

  enabled: Tri = "auto"
  background: bool = True
  background_color: tuple[float, float, float] = runtime_field((0.0, 0.0, 0.0))

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "environment")
    if not is_display_rgb(self.background_color):
      raise ValueError(f"environment.background_color must be an sRGB triple in [0, 1], got {self.background_color!r}")
    object.__setattr__(self, "background_color", rgb_tuple(self.background_color))
