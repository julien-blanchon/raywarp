"""Traced glossy reflections (`reflections.enabled`): smooth surfaces
reflect what one traced ray sees instead of the prefiltered environment.

The split-sum specular term of a shading point is `L_spec x spec_albedo x energy_comp`, with
`L_spec` the occluded prefiltered environment along the GGX dominant direction. Where the surface
is smooth enough, `L_spec` is replaced by the radiance of one traced ray:

  w_r    = saturate((r_max - r) / (r_max - r_fade))     r: perceptual roughness
  w_r   *= c^2 / (alpha^2 + c^2)                         "dominant" only: the lobe's core share
  faded specular weight s = w_r * lum(spec_albedo * energy_comp)
  w_r *= smoothstep(min_weight / 2, min_weight, s)   (unless min_weight is zero)
  L_spec = mix(L_env, L_ray x spec_ao, w_r)          spec_ao: of the material's AO only

so the result keeps the DFG specular albedo (energy-consistent with direct lights and the DFG
LUT) and fades continuously to the environment term between `r_fade` and `r_max`.

* "dominant": the ray follows the dominant direction normalize(mix(r, n, alpha^2)) (Filament); the
  hit's textures are read through a ray cone widened by the lobe, spread + 2 alpha (Akenine-Moller
  ray cones), so they blur with roughness. One ray has no blur of the geometry it sees, so it
  stands for the lobe's sharp core only: w_r is scaled by the core's share of the GGX NDF,
  c^2 / (alpha^2 + c^2) (its slope CDF at slope c = `CORE_SLOPE`), and the rest of the lobe keeps
  the blurred environment term (a sharp core over a blurred tail, as GGX's own long-tailed
  shape). Deterministic.

Emissive panels that are also active lights (`Materials.instance_emitter_light`, the map GI probe
rays use too; compiled in with area lights as `light_panels`): the shading point's LTC term
already integrates the panel over the whole specular lobe, so a reflection ray that hits the panel
takes no emission from it (it would count the panel twice).

The reflection hit is shaded cheaply (`make_glossy`): emission, direct lights (unshadowed unless
`reflections.shadows`, then with the scene's shadow rays and budget), the headlight, and the
camera hits' ambient terms (`kernels/ambient.make_hit_ambient`: the diffuse irradiance, the
specular lobes' prefiltered environment about the hit's normal, the coat and fuzz); no further
reflection, no translucency (a translucent hit blends the environment by its opacity). Rays that
leave the scene see the unoccluded environment (by definition they saw the sky).

Directional lights at a hit without sun shadow rays (reflections.shadows "off") take the hit's
sky visibility as theirs (`sun_visibility`: the GI's cosine-weighted sky
fraction about the hit's normal, no ray): unshadowed, the sun lit every indoor wall a reflection
saw (steel table legs glowed in a room with a sunlit window). Outdoors the fraction about a
normal above the horizon is at least 1/2 (an open ground hides at most the lower half of the
hemisphere), so twice it, capped at 1, leaves open surfaces fully sunlit; a sunlit patch indoors
keeps only the share of sky its window shows (dimmer than a shadow ray would leave it).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

import warp as wp

from raywarp._base.types import MAXVAL as MJ_MAXVAL
from raywarp.core.kernels import ambient
from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import BodyClosest, GeometryFields
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params
from raywarp.core.kernels.abi.tables import emission_radiance, exposure_at
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.post.grading import color
from raywarp.core.shade.lights import punctual, sources
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials import brdf, material
from raywarp.core.shade.materials.material import Lobes, MaterialTables

if TYPE_CHECKING:
  from raywarp.core.shade.lights.loop import LightLoop
  from raywarp.core.shade.materials.evaluate import EvalSurface

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Faded specular albedo luminance below which a traced reflection fades out (to zero at half of
# it, skipping its ray): glossy reflections of dark or rough-faded surfaces are invisible.
MIN_WEIGHT: Final = 0.02
# The half-vector slope (tan theta_h) out to which the dominant ray stands for a GGX lobe: its
# sharp core (`core_share`). The NDF holds half its projected mass within slope alpha, so a lobe
# of alpha 0.05 (perceptual roughness 0.22) is half core, half the blurred environment term: a
# chrome finish (0.05) stays a 99.8 % mirror, roughness 0.25 steel 39 %, 0.3 gold 24 % (the
# dominant ray alone showed 0.25-0.3 metals as clear mirrors, read as glass, on tabletop, workcell,
# warehouse and office interiors). `alpha` includes the specular
# anti-aliasing widening (`shade/materials/evaluate._specular_aa`), so a small curved mirror far
# from the camera is partly blurred too (a 25-pixel chrome sphere: 56 % at 256^2, 84 % at 512^2),
# where one ray per pixel would alias.
CORE_SLOPE: Final = 0.05
# Reflected directions closer to the tangent plane than this are left to the environment term.
_MIN_NOL: Final = 1.0e-4
# A hit's sun visibility per unit of its sky fraction seen (`sun_visibility`): the inverse of the
# least sky fraction a normal above the horizon sees over an open ground, (1 + sin(elevation)) / 2.
SKY_TO_SUN: Final = 2.0

# The functions a reflection is shaded with (producers:
# `shade/materials/evaluate.make_eval_surface`, `light_loop.make_light_loop`,
# `kernels/lighting.make_bsdf_eval`, `kernels/ambient`).

# This module's functions.
# shade_hit(geo, mats, lit, env, p, worldid, origin, normal, direction, footprint, spread,
# unoccluded, headlight_dir, seed, stream) -> radiance arriving along a reflection ray.
type ShadeHit = Callable[
  [
    GeometryFields,
    MaterialTables,
    Lights,
    EnvironmentArrays,
    Params,
    int,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    float,
    float,
    wp.vec3,
    wp.vec3,
    int,
    int,
  ],
  wp.vec3,
]
# glossy(geo, mats, lit, env, p, worldid, lobes, outward, origin, normal, view, nov, footprint,
# spread, unoccluded, headlight_dir, seed, stream) -> (L_ray, w_r) (`make_glossy`).
type Glossy = Callable[
  [
    GeometryFields,
    MaterialTables,
    Lights,
    EnvironmentArrays,
    Params,
    int,
    Lobes,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    float,
    float,
    float,
    wp.vec3,
    wp.vec3,
    int,
    int,
  ],
  tuple[wp.vec3, float],
]


@wp.func
def trace_weight(
  # In:
  p: Params,
  perceptual_roughness: float,
) -> float:
  """w_r: 1 below the fade roughness, 0 above the maximum, linear in between."""
  span = p.reflection_max_roughness - p.reflection_fade_roughness
  if span <= 0.0:
    return wp.where(perceptual_roughness < p.reflection_max_roughness, 1.0, 0.0)
  return wp.clamp((p.reflection_max_roughness - perceptual_roughness) / span, 0.0, 1.0)


@specialize
def make_glossy(
  f: Features,
  trace_closest: BodyClosest,
  eval_surface: EvalSurface,
  hit_lights: LightLoop,
):
  """The traced reflection ray of the split sum's specular term.

  Returns glossy(geo, mats, lit, env, p, worldid, lobes, outward, origin, normal, view, nov,
  footprint, spread, unoccluded, headlight_dir, seed, stream) -> (L_ray, w_r): the radiance
  of the traced ray and its share w_r of the environment's specular term (module doc;
  `surface.make_indirect` mixes them); `unoccluded` is what a ray that leaves the scene sees (the
  environment and fill, `ambient.make_lobe_ambient`); `origin` is the shading point's
  secondary-ray origin and `outward` its geometric normal (`Surface.origin` / `outward`). A
  direction above the shading normal's horizon but under the surface is not traced (w_r 0).
  `hit_lights` is a light loop
  (`light_loop.make_light_loop`: geo, lit, worldid, lobes, point, normal, origin, view, nov, seed,
  stream, sun, head_dir, head), shadowed or not, with the headlight's entry; `footprint` /
  `spread` are the camera ray cone's width at the point and its angular spread; `seed`,
  `stream`: the hit's shadow samples.
  """
  shade_hit = make_shade_hit(f, trace_closest=trace_closest, eval_surface=eval_surface, hit_lights=hit_lights)
  return _make_dominant_glossy(shade_hit, anisotropy=f.specular_anisotropy)


@wp.func(inline=True)
def core_share(
  # In:
  alpha: float,
) -> float:
  """The share of a GGX lobe of roughness `alpha` that one ray along its dominant direction
  stands for: the NDF's projected mass within half-vector slope `CORE_SLOPE` of the normal,
  CORE_SLOPE^2 / (alpha^2 + CORE_SLOPE^2) (the GGX slope CDF; module doc)."""
  core = wp.static(CORE_SLOPE * CORE_SLOPE)
  return core / (alpha * alpha + core)


@wp.func(inline=True)
def shown_weight(
  # In:
  p: Params,
  lobes: Lobes,
  share: float,
) -> float:
  """Roughness blend with a continuous low-contribution transition to the environment; `share`
  scales the traced ray's part of it (`core_share` for the dominant ray, 1 for a lobe sample).

  Above `MIN_WEIGHT` retain the full blend. Below it, introduce weaker rays smoothly down
  to half the threshold; a hard cutoff here turns textured roughness into binary patches. This
  adds rays but never attenuates a contribution that the previous hard cutoff traced.
  """
  if p.rt.glossy == 0:
    return 0.0  # reflections off at run time (a tier build compiles them)
  w = trace_weight(p, lobes.perceptual_roughness) * share
  if w <= 0.0:
    return 0.0
  contribution = w * color.luminance(wp.cw_mul(lobes.spec_albedo, lobes.energy_comp))
  if contribution <= 0.0:
    return 0.0
  threshold = p.reflection_min_weight
  if threshold > 0.0 and contribution < threshold:
    blend = wp.clamp(2.0 * contribution / threshold - 1.0, 0.0, 1.0)
    w *= blend * blend * (3.0 - 2.0 * blend)
  return w


@specialize
def _make_dominant_glossy(shade_hit: ShadeHit, *, anisotropy: bool):
  """glossy(...) of "dominant" (`make_glossy`'s signature): one ray along the lobe's dominant
  direction (an anisotropic lobe's about its bent normal, as its image-based term:
  `material.bent_normal`)."""

  @wp.func
  @device_generic
  def glossy[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    lobes: Lobes,
    outward: wp.vec3,
    origin: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    _nov: float,
    footprint: float,
    spread: float,
    unoccluded: wp.vec3,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, float]:
    """(radiance of the dominant-direction ray, its weight)."""
    w = shown_weight(p, lobes, core_share(lobes.alpha))
    if w <= 0.0:
      return wp.vec3(0.0), 0.0
    alpha = lobes.alpha
    if wp.static(anisotropy):
      d = brdf.dominant_direction(view, material.bent_normal(lobes, normal, view), alpha)
    else:
      d = brdf.dominant_direction(view, normal, alpha)
    if wp.dot(d, normal) <= wp.static(_MIN_NOL) or wp.dot(d, outward) <= 0.0:
      return wp.vec3(0.0), 0.0
    traced = shade_hit(
      geo,
      mats,
      lit,
      env,
      p,
      worldid,
      origin,
      normal,
      d,
      footprint,
      spread + 2.0 * alpha,
      unoccluded,
      headlight_dir,
      seed,
      stream,
    )
    return traced, w

  return glossy


@wp.func(inline=True)
def sun_visibility(
  # In:
  sky_seen: float,
) -> float:
  """The directional lights' visibility at a reflection hit that traces no ray for them, from its
  cosine-weighted sky fraction seen `sky_seen` (module doc; 1 without GI)."""
  return wp.min(sky_seen * wp.static(SKY_TO_SUN), 1.0)


@wp.func(inline=True)
@device_generic
def panel_light[M: MaterialTables](
  # Model:
  mats: M,
  lit: Lights,
  # In:
  worldid: int,
  instance_id: int,
) -> bool:
  """Whether instance `instance_id` (< 0: none) is an emissive panel standing in for a light active
  in world `worldid` (its emission then comes from the light: module doc)."""
  if instance_id >= 0:
    li = mats.instance_emitter_light[instance_id]
    if li >= 0:
      return punctual.fetch(lit, worldid, li).active
  return False


@specialize
def make_shade_hit(f: Features, *, trace_closest: BodyClosest, eval_surface: EvalSurface, hit_lights: LightLoop):
  """shade_hit(geo, mats, lit, env, p, worldid, origin, normal, direction, footprint, spread,
  unoccluded, headlight_dir, seed, stream) -> radiance arriving along a reflection ray from
  the shading point at `origin` of shading normal `normal`.

  The hit is shaded cheaply (module doc): `hit_lights` (with the headlight's entry), the camera
  hits' ambient terms (`ambient.make_hit_ambient`) and emission; a miss returns `unoccluded`.
  """
  hit_ambient = ambient.make_hit_ambient(f)
  headlight = f.headlight
  on = toggles.on(f)
  transparency = f.transparency
  setting = toggles.setting_on(f)  # translucent layers: a tier build reads them at run time
  light_panels = f.light_panels
  relative_emission = material.make_relative_emission(f)
  area = toggles.bound_on(f.bound)
  # Whether the camera rays' light loop shaded a panel's light exactly at the shading point (its
  # LTC term then integrates the panel: the reflection skips the emission); a light the cap left
  # to the diffuse remainder has no specular term there, so its panel shows in the mirror.
  exact_light = sources.make_exact_test(
    nlight=f.nlight,
    culling=f.culling,
    cap=f.light_cap,
    has_spot=f.has_spot,
    ies=f.ies,
    bound=f.bound,
    counts=sources.counts_of(f, sources.CAMERA_CAP),
  )
  counting = f.counters
  shading_lobes = material.lobes_of(f)

  @wp.func
  @device_generic
  def shade_hit[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    origin: wp.vec3,
    normal: wp.vec3,
    direction: wp.vec3,
    footprint: float,
    spread: float,
    unoccluded: wp.vec3,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> wp.vec3:
    """Radiance arriving at `origin` from `direction` (cheap shading of the hit)."""
    if wp.static(counting):
      counters.add(geo, worldid, wp.static(counters.REFLECTION), 1)
    hit = trace_closest(geo, worldid, origin, direction, float(MJ_MAXVAL))
    if hit.instance_id == -1:
      return unoccluded
    q = origin + direction * hit.dist
    cone = footprint + spread * hit.dist
    surface, n = eval_surface(geo, mats, worldid, hit, q, direction, cone)
    view = -direction
    nov = wp.max(wp.dot(n, view), 1.0e-4)
    lobes = shading_lobes(env.dfg_lut, surface, nov)
    ambient_radiance, sky_seen = hit_ambient(env, worldid, lobes, surface, q, n, view, nov, spread)
    sun = sun_visibility(sky_seen)
    exposure = exposure_at(env.exposure, worldid)
    head = wp.vec3(0.0)  # the headlight: the light loop's last entry (zero: off)
    if wp.static(headlight):
      if on(geo.rt.flags, wp.static(toggles.HEADLIGHT)):
        head = p.headlight_color * (p.headlight_intensity / exposure)
    radiance = hit_lights(geo, lit, worldid, lobes, q, n, surface.origin, view, nov, seed, stream, sun, headlight_dir, head)
    radiance += ambient_radiance
    emissive = surface.emissive
    if wp.static(light_panels):
      # A panel that is also an area light: the shading point's LTC term already integrates its
      # emission over the specular lobe; seeing it again here would count it twice. Not when the
      # shading point is behind the panel light (it emits along `dir`: a double-sided panel's back,
      # USD stages being double-sided by default, glows but lights nothing there), nor when the
      # light cap left the light out of the exact set (`exact_light`).
      # Only an emitting hit is looked at (nested tests, not `and`: Warp evaluates both operands):
      # a reflection ray that meets no emitter reads nothing more.
      if area(lit.bits, wp.static(toggles.CLASS_AREA)):  # the area class bit (Features.bound)
        if emissive[0] + emissive[1] + emissive[2] > 0.0 and hit.instance_id >= 0:
          if on(geo.rt.flags, wp.static(toggles.LIGHT_PANELS)):
            li = mats.instance_emitter_light[hit.instance_id]
            if li >= 0:
              panel = punctual.fetch(lit, worldid, li)
              if panel.active and wp.dot(origin - panel.pos, panel.dir) > 0.0:
                if exact_light(geo, lit, worldid, li, origin, normal):
                  emissive = wp.vec3(0.0)
    relative = relative_emission(p.relative_emission, surface)
    radiance += emission_radiance(emissive, exposure, relative)
    if wp.static(transparency):
      # A translucent hit blends what is behind it, unshaded: the environment, by its opacity.
      if setting(geo.rt.flags, geo.bits2, wp.static(toggles.TRANSPARENCY)):
        opacity = wp.clamp(surface.alpha, 0.0, 1.0)
        radiance = radiance * opacity + unoccluded * (1.0 - opacity)
    return radiance

  return shade_hit
