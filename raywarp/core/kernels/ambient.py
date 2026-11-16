"""The light a surface receives from everything but the lights: the environment (split-sum IBL),
the ambient fill and the GI probes, occluded, through the lobes of the material. One formula for
camera hits (`kernels/surface.py`, where a traced glossy reflection can replace the base
specular's environment term) and glossy-reflection hits (`kernels/reflections.py`).

  diffuse_ambient  (fill, occluded diffuse irradiance, GI probe lookup, sky fraction seen)
  lobe_ambient     (occluded, unoccluded) prefiltered radiance of one GGX-like lobe: the base
                   specular, the coat, the fuzz
  base_ambient     the base lobes' ambient radiance from those terms
  layered_ambient  the base's ambient under the coat and fuzz layers plus theirs
                   (`material.make_layers`: the law of every light)
  hit_ambient      all of it at a reflection hit (no traced term)

Occlusion (Lagarde & de Rousiers 2014): the environment, the ambient fill (seen everywhere) and
the GI probes (a radiance, not a visibility) are occluded by the material's ambient occlusion
`ao`; specular terms by its specular occlusion (`specular_occlusion`). The user's ambient
(`lighting.fill`) is added to the diffuse irradiance in every configuration, occluded by `ao`
only: a display fill, not light (no specular, not seen by the GI probes).
"""

from __future__ import annotations

import warp as wp

from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.structs import EnvironmentArrays
from raywarp.core.kernels.abi.tables import batched1d, exposure_at
from raywarp.core.kernels.generic import specialize
from raywarp.core.shade import gi
from raywarp.core.shade.environment import envmap
from raywarp.core.shade.materials import brdf, material
from raywarp.core.shade.materials.material import Lobes, Surface

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def specular_occlusion(
  # In:
  nov: float,
  ao: float,
  alpha: float,
) -> float:
  """Specular occlusion of an ambient occlusion `ao` at roughness `alpha` (Lagarde & de Rousiers
  2014)."""
  return wp.clamp(wp.pow(nov + ao, wp.pow(2.0, -16.0 * alpha - 1.0)) - 1.0 + ao, 0.0, 1.0)


@wp.func(inline=True)
def prefiltered_at(
  # Model:
  env: EnvironmentArrays,
  # In:
  e: int,
  yaw: float,
  view: wp.vec3,
  normal: wp.vec3,
  alpha: float,
  perceptual_roughness: float,
  spread: float,
) -> wp.vec3:
  """Prefiltered radiance of environment map `e` (rotated by `yaw`, unscaled) along the GGX
  lobe's dominant direction."""
  dominant = brdf.dominant_direction(view, normal, alpha)
  return envmap.sample_specular(env.env_radiance[e], env.env_prefiltered[e], dominant, yaw, perceptual_roughness, spread)


@wp.func
def horizon_fade(
  # In:
  view: wp.vec3,
  normal: wp.vec3,
  outward: wp.vec3,
) -> float:
  """Filament's specular horizon occlusion (listing 39): min(1 + r . n_g, 1)², r the reflection
  of the view about the (mapped) `normal`, n_g the geometric normal `outward`: a mapped normal's
  reflection below the surface fades instead of fetching the environment under it."""
  r = normal * (2.0 * wp.dot(view, normal)) - view
  h = wp.clamp(1.0 + wp.dot(r, outward), 0.0, 1.0)
  return h * h


@wp.func
def display_exposure(
  # Model:
  env: EnvironmentArrays,
  # In:
  worldid: int,
) -> float:
  """The exposure that the display-relative fill (`lighting.fill`) is relative to: under
  automatic exposure camera 0's metered exposure (`EnvironmentArrays.view_exposure`), 0 before the
  world's first metering (no fill on that frame: relative to the reference exposure, a scene in
  physical units got a fill hundreds of times too bright, which the meter then exposed for); else
  the world's manual exposure."""
  n = env.view_exposure.shape[0]
  if n > 0:
    return env.view_exposure[worldid % n]
  return exposure_at(env.exposure, worldid)


@wp.func
def open_sky(
  # Model:
  env: EnvironmentArrays,
  # In:
  worldid: int,
  point: wp.vec3,
  normal: wp.vec3,
  probes: wp.vec4,
) -> wp.vec4:
  """The GI lookup `probes` (E / pi, sky visibility) at `point`, faded beyond the probe box to the
  environment's unoccluded irradiance (what a probe ray's miss returns) over one probe spacing
  (`gi.beyond`), for receivers facing up (+z, the environment's up) by the normal's z: a ground
  plane past the scene's objects is lit by the open sky, not by the box's boundary texels. A
  receiver facing down or sideways keeps the lookup: below the horizon a probe ray meets the
  ground, not the environment's lower half (the moving parts of a robot are outside the box of the
  static geometry, their undersides lit by the floor)."""
  fade = wp.min(gi.beyond(env.gi, point), 1.0) * wp.clamp(normal[2], 0.0, 1.0)
  if fade <= 0.0:
    return probes
  e, yaw, scale = kernel_structs.env_lookup(env, worldid)
  sky = envmap.eval_sh9(env.env_sh, e, envmap.rotate_yaw(normal, -yaw)) * scale
  return probes + (wp.vec4(sky[0], sky[1], sky[2], 1.0) - probes) * fade


@specialize
def make_diffuse_ambient(f: Features):
  """diffuse_ambient(env, worldid, point, normal, outward, ao) -> (the scene's ambient
  fill, diffuse irradiance occluded by `ao` (E / pi), GI probe lookup (irradiance, sky
  visibility; zero without GI), sky fraction seen (the probes' with GI, else 1))
  around the shading `normal`, the probe / sky lookups offset along the geometric normal
  `outward`: the GI probes' irradiance with GI, else the fill plus the environment's SH9
  irradiance; plus, in every case, the user's ambient (`Params.ambient`, display-relative: over
  the exposure the world is displayed at, `display_exposure`) occluded by `ao`."""
  gi_on = f.gi
  env_lit = f.env_lit
  on = toggles.on(f)
  gate = toggles.bound_on(f.bound)  # GI, environment bound (`toggles.SETTING_*`, `env.bits`)

  @wp.func
  def diffuse_ambient(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
    outward: wp.vec3,
    ao: float,
  ) -> tuple[wp.vec3, wp.vec3, wp.vec4, float]:
    """The ambient diffuse terms of a point (`make_diffuse_ambient`)."""
    fill = batched1d(env.ambient_radiance, worldid)
    shown = display_exposure(env, worldid)
    user = env.params.ambient * wp.where(shown > 0.0, ao / shown, 0.0)
    probes = wp.vec4(0.0)
    sky_seen = float(1.0)
    rt = env.params.rt
    if wp.static(gi_on):
      if gate(env.bits, wp.static(toggles.SETTING_GI)):
        if on(rt.flags, wp.static(toggles.GI)):
          probes = gi.irradiance(env.gi, worldid, point, normal, outward)
          if wp.static(env_lit):
            if gate(env.bits, wp.static(toggles.SETTING_ENV_LIT)) and on(rt.flags, wp.static(toggles.ENV_LIT)):
              probes = open_sky(env, worldid, point, normal, probes)
          return fill, wp.vec3(probes[0], probes[1], probes[2]) * ao + user, probes, probes[3]
    if wp.static(env_lit):
      if gate(env.bits, wp.static(toggles.SETTING_ENV_LIT)) and on(rt.flags, wp.static(toggles.ENV_LIT)):
        e, yaw, scale = kernel_structs.env_lookup(env, worldid)
        sky = envmap.eval_sh9(env.env_sh, e, envmap.rotate_yaw(normal, -yaw)) * scale
        return fill, (fill + sky) * ao + user, probes, sky_seen
    return fill, fill * ao + user, probes, sky_seen

  return diffuse_ambient


@specialize
def make_lobe_ambient(f: Features):
  """lobe_ambient(env, worldid, normal, view, nov, alpha, perceptual_roughness, fill, probes, ao,
  spread) -> (occluded, unoccluded) prefiltered ambient radiance of a GGX-like lobe of
  roughness (alpha, perceptual) about `normal` seen at NoV `nov`: the environment along the
  lobe's dominant direction (`spread`: the ray cone), with GI weighted by the probes' sky
  visibility and the rest from the probe irradiance (one probe lookup about the shading normal: a
  second one along the dominant direction cost +20 % of a large interior scene's frame on the GPU
  for no measured gain), plus the fill. Unoccluded: what a reflection
  ray that leaves the scene sees (the environment and fill, or the probes without an
  environment)."""
  env_lit = f.env_lit
  gi_on = f.gi
  on = toggles.on(f)
  gate = toggles.bound_on(f.bound)  # GI, environment bound (`toggles.SETTING_*`)

  @wp.func
  def lobe_ambient(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    alpha: float,
    perceptual_roughness: float,
    fill: wp.vec3,
    probes: wp.vec4,
    ao: float,
    spread: float,
  ) -> tuple[wp.vec3, wp.vec3]:
    """(occluded, unoccluded) prefiltered radiance of a lobe (`make_lobe_ambient`)."""
    occlusion = specular_occlusion(nov, ao, alpha)
    rt = env.params.rt
    if wp.static(env_lit):
      if gate(env.bits, wp.static(toggles.SETTING_ENV_LIT)) and on(rt.flags, wp.static(toggles.ENV_LIT)):
        e, yaw, scale = kernel_structs.env_lookup(env, worldid)
        sky = prefiltered_at(env, e, yaw, view, normal, alpha, perceptual_roughness, spread) * scale
        if wp.static(gi_on):
          if gate(env.bits, wp.static(toggles.SETTING_GI)):
            if on(rt.flags, wp.static(toggles.GI)):
              return (sky * probes[3] + wp.vec3(probes[0], probes[1], probes[2]) * (1.0 - probes[3])) * occlusion, sky
        return (fill + sky) * occlusion, fill + sky
    if wp.static(gi_on):
      if gate(env.bits, wp.static(toggles.SETTING_GI)):
        if on(rt.flags, wp.static(toggles.GI)):
          probe = wp.vec3(probes[0], probes[1], probes[2])
          return probe * occlusion, probe
    return fill * occlusion, fill

  return lobe_ambient


@specialize
def make_base_ambient(f: Features):
  """base_ambient(lobes, nov, irradiance, specular) -> the base lobes' ambient radiance: the
  diffuse albedo times the irradiance (a wrapped cloth, Features.fuzz: Filament's image-based
  tint saturate(tint + NoV); not its Fd_Wrap(NoV, w): the direct wrap is energy-normalised under
  uniform light, which the irradiance already is, and Fd_Wrap(NoV) would darken the image-based
  cloth by up to a third against a dome of lights of the same radiance), plus the specular albedo
  times the specular radiance (both already occluded)."""
  wrapped = f.fuzz

  @wp.func
  def base_ambient(
    # In:
    lobes: Lobes,
    nov: float,
    irradiance: wp.vec3,
    specular: wp.vec3,
  ) -> wp.vec3:
    """The base lobes' ambient radiance (`make_base_ambient`)."""
    diffuse = wp.cw_mul(irradiance, lobes.diffuse_color)
    if wp.static(wrapped):
      if lobes.wrap > 0.0:
        tint = lobes.wrap_tint + wp.vec3(nov)
        tint = wp.vec3(wp.clamp(tint[0], 0.0, 1.0), wp.clamp(tint[1], 0.0, 1.0), wp.clamp(tint[2], 0.0, 1.0))
        diffuse = wp.cw_mul(diffuse, tint)
    return diffuse + wp.cw_mul(specular, wp.cw_mul(lobes.spec_albedo, lobes.energy_comp))

  return base_ambient


@specialize
def make_layered_ambient(f: Features):
  """layered_ambient(env, worldid, lobes, surface, normal, view, nov, fill, probes, spread,
  base) -> `base` (the base lobes' ambient radiance) under the coat (its own normal and
  NoV, Features.coat_layer / coat_normal; its environment faded at the horizon like the base's,
  Features.horizon_fade) and the fuzz layer (Filament's sheen IBL: the prefiltered environment at
  the fuzz's perceptual roughness sqrt(α) along the normal's reflection, an approximation: the
  stack is GGX-prefiltered, not Charlie's), plus their prefiltered ambient radiance times their
  albedos (`material.make_layers`)."""
  clearcoat = f.clearcoat
  fuzz = f.fuzz
  layered_coat = f.coat_layer or f.coat_normal
  horizon = f.horizon_fade
  lobe_ambient = make_lobe_ambient(f)
  layers = material.layers_of(f)
  gate = toggles.bound_on(f.bound)  # the coat and layers class bits, the horizon fade (`Lobes.bits`)

  @wp.func
  def layered_ambient(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    lobes: Lobes,
    surface: Surface,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    fill: wp.vec3,
    probes: wp.vec4,
    spread: float,
    base: wp.vec3,
  ) -> wp.vec3:
    """`base` under the coat and fuzz layers plus theirs (`make_layered_ambient`)."""
    ao = surface.occlusion
    coat = wp.vec3(0.0)
    layered = False
    if wp.static(layered_coat or fuzz):
      layered = gate(lobes.bits, wp.static(toggles.CLASS_LAYERS))
    if wp.static(clearcoat):
      if gate(lobes.bits, wp.static(toggles.CLASS_COAT)) and lobes.coat > 0.0:
        cn = normal
        cnov = nov
        if wp.static(layered_coat):
          if layered:
            cn = lobes.coat_normal
            cnov = lobes.coat_nov
        light, _open = lobe_ambient(
          env, worldid, cn, view, cnov, lobes.coat_alpha, lobes.coat_roughness, fill, probes, ao, spread
        )
        if wp.static(horizon):
          if gate(lobes.bits, wp.static(toggles.SCENE_HORIZON_FADE)):
            light = light * horizon_fade(view, cn, surface.outward)
        coat = light * lobes.coat_albedo
    sheen = wp.vec3(0.0)
    if wp.static(fuzz):
      if layered and lobes.fuzz_weight > 0.0:
        light, _open = lobe_ambient(
          env, worldid, normal, view, nov, lobes.fuzz_alpha, wp.sqrt(lobes.fuzz_alpha), fill, probes, ao, spread
        )
        sheen = wp.cw_mul(light, lobes.fuzz_color) * (lobes.fuzz_weight * lobes.fuzz_albedo)
    return layers(lobes, base, coat, lobes.coat_nov, sheen)

  return layered_ambient


@specialize
def make_hit_ambient(f: Features):
  """hit_ambient(env, worldid, lobes, surface, point, normal, view, nov, spread) -> (the ambient
  radiance of a glossy-reflection hit: the camera hits' terms (`make_diffuse_ambient`,
  `make_lobe_ambient` about the anisotropic bent normal, `make_base_ambient`,
  `make_layered_ambient`) without a traced reflection; the cosine-weighted sky fraction seen about
  the normal, `make_diffuse_ambient`'s)."""
  diffuse_ambient = make_diffuse_ambient(f)
  lobe_ambient = make_lobe_ambient(f)
  base_ambient = make_base_ambient(f)
  layered_ambient = make_layered_ambient(f)
  anisotropy = f.specular_anisotropy
  horizon = f.horizon_fade
  layered = f.clearcoat or f.fuzz
  gate = toggles.bound_on(f.bound)  # the coat and layers class bits, the horizon fade (`Lobes.bits`)
  field = toggles.bound_on(f.bound, high=True)  # the layered lobes' field bits (`toggles.LAYER_*`)

  @wp.func
  def hit_ambient(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    lobes: Lobes,
    surface: Surface,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    spread: float,
  ) -> tuple[wp.vec3, float]:
    """(The ambient radiance of a reflection hit, its sky fraction seen) (`make_hit_ambient`)."""
    ao = surface.occlusion
    fill, irradiance, probes, sky_seen = diffuse_ambient(env, worldid, point, normal, surface.outward, ao)
    spec_normal = normal
    if wp.static(anisotropy):
      if gate(lobes.bits, wp.static(toggles.CLASS_LAYERS)) and field(lobes.bits2, wp.static(toggles.LAYER_ANISOTROPY)):
        spec_normal = material.bent_normal(lobes, normal, view)
    specular, _open = lobe_ambient(
      env, worldid, spec_normal, view, nov, lobes.alpha, lobes.perceptual_roughness, fill, probes, ao, spread
    )
    if wp.static(horizon):
      if gate(lobes.bits, wp.static(toggles.SCENE_HORIZON_FADE)):
        specular = specular * horizon_fade(view, normal, surface.outward)
    radiance = base_ambient(lobes, nov, irradiance, specular)
    if wp.static(layered):
      if gate(lobes.bits, wp.static(toggles.CLASS_COAT | toggles.CLASS_LAYERS)):
        radiance = layered_ambient(env, worldid, lobes, surface, normal, view, nov, fill, probes, spread, radiance)
    return radiance, sky_seen

  return hit_ambient
