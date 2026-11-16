"""Whether a point lit by a punctual light (point, spot or sun) sees it: the shadow rays and
their visibility fraction (`make_visibility` -> light_visibility).

Hard shadows trace one any-hit ray to the light; soft ones (`shadows.samples` > 1) sample the
emitter's disc; directional lights trace along their direction (through translucent layers with
`shadows.translucent`, attenuated by their transmittance) or sample the sun's disc
(`Features.sun_disc`). Rays leave from the shading point's `Surface.origin`. The visibility is a
fraction, or an RGB transmittance with tinted translucent shadows (`punctual.visibility_ops`).
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.queries import Queries
from raywarp.core.kernels.abi.rays import UNBOUNDED_SHADOW_DIST, GeometryFields
from raywarp.core.kernels.generic import compiled_out, concrete, device_generic, specialize
from raywarp.core.shade.lights import area, punctual
from raywarp.core.shade.lights.punctual import Light, Visibility
from raywarp.core.shade.materials import texture

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# The visibility functions return `V`: a float fraction, or an RGB transmittance (`wp.vec3`) with
# tinted shadows (`punctual.visibility_ops(Features.tinted_shadows)`); it is chosen at build time,
# so a dynamic Features value exposes the union instead of an unbound return parameter.
# shadow_ray(geo, worldid, origin, l, max_dist) -> visibility (`make_shadow_ray`).
type ShadowRay[V] = Callable[[GeometryFields, int, wp.vec3, wp.vec3, float], V]
# shadow_pass(geo, worldid, origin, l, max_dist) -> the transmitted fraction (`make_shadow_pass`).
type ShadowPass = Callable[[GeometryFields, int, wp.vec3, wp.vec3, float], float]
# light_visibility(geo, worldid, light, origin, normal, l, dist, seed, stream, nlight, li) ->
# visibility (`make_visibility`).
type LightVisibility[V] = Callable[[GeometryFields, int, Light, wp.vec3, wp.vec3, wp.vec3, float, int, int, int, int], V]


@specialize
def make_shadow_ray(f: Features, q: Queries):
  """shadow_ray(geo, worldid, origin, l, max_dist) -> visibility along one shadow ray of every
  light kind (a directional one: unbounded; a point, spot or area sample: to the light), through
  translucent casters (curtains, glass) with Features.translucent_shadows: one terminating walk
  whose translucent candidates multiply their transmittance in (`Queries.shadow`; tinted: RGB, else
  its first channel), so an unoccluded ray that meets none costs a plain occlusion query."""
  tinted = f.tinted_shadows
  on = toggles.on(f)
  setting = toggles.setting_on(f)  # shadows: a tier build reads them at run time (off: unoccluded)
  # `VisT` (not snake case, ruff N806) names the device functions' return type, as below: renaming
  # it would change their hashed source.
  VisT, vis_of = punctual.visibility_ops(tinted)
  trace_any = q.any_hit
  trace_shadow = q.shadow if q.shadow is not None else compiled_out
  through = q.shadow is not None

  @wp.func
  @concrete(**{"return": VisT})
  @device_generic
  def shadow_ray[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    origin: wp.vec3,
    l: wp.vec3,
    max_dist: float,
  ) -> Visibility:
    """Visibility along one shadow ray to `max_dist` (1: unoccluded)."""
    if not setting(geo.rt.flags, geo.bits2, wp.static(toggles.SHADOWS)):
      return vis_of(1.0)
    if wp.static(through):
      # A scene-free build compiles the walk for every scene; the scene's casters (and the
      # settings) switch it at run time. Not one trace site: without casters the walk is not the
      # occlusion query on the software traversal (it does not cull back faces; the any-hit query
      # does), so an opaque scene's image would change (one site measured: OptiX default, a large
      # textured office interior: 9 -> 6 raygen trace calls, continuation spills 30 -> 14 KB).
      if on(geo.rt.flags, wp.static(toggles.TRANSLUCENT_SHADOWS)):
        t = trace_shadow(geo, worldid, origin, l, max_dist)
        # if / else: a statement after a static branch is still generated (with the other type).
        if wp.static(tinted):
          return t
        else:
          return t[0]
    return vis_of(wp.where(trace_any(geo, worldid, origin, l, max_dist), 0.0, 1.0))

  return shadow_ray


@specialize
def make_shadow_pass(f: Features, q: Queries):
  """shadow_pass(geo, worldid, origin, l, max_dist) -> the fraction one shadow ray transmits
  (`make_shadow_ray`; tinted: the mean of its RGB transmittance), for the area lights' samples."""
  shadow_ray = make_shadow_ray(f, q)
  tinted = f.tinted_shadows

  @wp.func
  @device_generic
  def shadow_pass[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    origin: wp.vec3,
    l: wp.vec3,
    max_dist: float,
  ) -> float:
    """The transmitted fraction along one shadow ray (see `make_shadow_pass`)."""
    v = shadow_ray(geo, worldid, origin, l, max_dist)
    # if / else: a statement after a static branch is still generated (with the RGB type here).
    if wp.static(tinted):
      return (v[0] + v[1] + v[2]) / 3.0  # pyrefly: ignore[bad-index]  # RGB under this static flag
    else:
      return v  # pyrefly: ignore[bad-return]  # float without tinted shadows

  return shadow_pass


@specialize
def make_sun_visibility(f: Features, shadow_ray: ShadowRay[Visibility], *, runtime: bool):
  """sun_visibility(geo, worldid, light, origin, normal, l, seed, stream, nlight, li);
  `runtime`: the sample count and the disc read at run time (`Toggles`)."""
  shadow_samples = f.shadow_samples
  VisT, vis_of = punctual.visibility_ops(f.tinted_shadows)

  @wp.func
  @concrete(**{"return": VisT})
  @device_generic
  def sun_visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    light: Light,
    origin: wp.vec3,
    normal: wp.vec3,
    l: wp.vec3,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> Visibility:
    """Visible fraction of a directional light's disc (`radius`: its angular radius).

    `shadow_samples` directions uniform in the disc's cone (the fixed pattern shifted per pixel);
    one ray along `l` for a light without a disc. Samples below the surface's horizon are blocked
    by the surface itself.
    """
    nsample = int(1)
    hard = bool(False)  # run time without the sun disc: one ray along `l` (no horizon test)
    if wp.static(runtime):
      hard = geo.rt.sun_disc == 0
      if light.radius > 0.0 and not hard:
        nsample = geo.rt.shadow_samples
    elif light.radius > 0.0:
      nsample = wp.static(shadow_samples)
    one_minus_cos = 1.0 - wp.cos(light.radius)
    t1, t2 = texture.orthonormal_basis(l)
    visible = vis_of(0.0)
    shift = wp.vec2(0.0)
    if not hard:
      shift = area.shadow_rand2(seed, stream, nlight, li, nsample, 0)
    # A runtime trip count: one shadow-ray call site (a constant one would be unrolled).
    for k in range(nsample):
      u = area.fixed_rand2(k, shift)
      if nsample == 1:
        u = wp.vec2(0.0)
      cos_t = 1.0 - u[0] * one_minus_cos
      sin_t = wp.sqrt(wp.max(1.0 - cos_t * cos_t, 0.0))
      phi = wp.static(2.0 * wp.pi) * u[1]
      d = l * cos_t + (t1 * wp.cos(phi) + t2 * wp.sin(phi)) * sin_t
      if hard:
        d = l
      if hard or wp.dot(d, normal) > 0.0:
        visible += shadow_ray(geo, worldid, origin, d, wp.static(UNBOUNDED_SHADOW_DIST))  # pyrefly: ignore[unsupported-operation]  # both use f.tinted_shadows
    return visible * (1.0 / float(nsample))

  return sun_visibility


@specialize
def make_soft_visibility(f: Features, q: Queries, *, runtime: bool):
  """soft_visibility(geo, worldid, light, origin, l, dist, seed, stream, nlight, li): the
  visible fraction of a punctual light's emitter disc (`shadows.samples` > 1); `runtime`: the
  sample count read at run time (`Toggles`)."""
  shadow_samples = f.shadow_samples
  shadow_pass = make_shadow_pass(f, q)
  VisT, vis_of = punctual.visibility_ops(f.tinted_shadows)

  @wp.func
  @concrete(**{"return": VisT})
  @device_generic
  def soft_visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    light: Light,
    origin: wp.vec3,
    l: wp.vec3,
    dist: float,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> Visibility:
    """Visible fraction of light `li`'s disc of radius `radius` from `origin` (`l`, `dist`:
    towards it; read at run time, one shadow sample traces that one ray, as the hard shadows).

    A flat emitter is a disc in its own plane (one facing the point would reach behind the
    emitter, into the ceiling it hangs from); a bulb's silhouette faces the point.
    """
    axis = wp.where(light.type == wp.static(punctual.LIGHT_COSINE), light.dir, l)
    t1, t2 = texture.orthonormal_basis(axis)
    # A light without a radius: its rays would all be the same one.
    nsample = int(1)
    hard = bool(False)
    if wp.static(runtime):
      hard = geo.rt.shadow_samples <= 1
    if light.radius > 0.0 and not hard:
      if wp.static(runtime):
        nsample = geo.rt.shadow_samples
      else:
        nsample = wp.static(shadow_samples)
    visible = float(0.0)
    # A runtime trip count: one shadow-ray call site (a constant one would be unrolled).
    for k in range(nsample):
      ray = l
      reach = dist * wp.static(area.SHADOW_END)
      if not hard:
        u = area.shadow_rand2(seed, stream, nlight, li, nsample, k)
        r = light.radius * wp.sqrt(u[0])
        phi = wp.static(2.0 * wp.pi) * u[1]
        to_target = light.pos + (t1 * wp.cos(phi) + t2 * wp.sin(phi)) * r - origin
        d = wp.length(to_target)
        ray = to_target / d
        reach = d * wp.static(area.SHADOW_END)
      visible += shadow_pass(geo, worldid, origin, ray, reach)
    return vis_of(visible / float(nsample))

  return soft_visibility


@specialize
def make_visibility(f: Features, q: Queries):
  """light_visibility(geo, worldid, light, origin, normal, l, dist, seed, stream, nlight, li). A
  tier build reading the soft shadows at run time (`toggles.SOFT_SHADOWS`, on where the
  configuration's sampling differs from the build's, `tiers.soft`) takes the run-time sampling
  there, its own compiled sampling elsewhere."""
  shadow_samples = f.shadow_samples
  sun_disc = f.sun_disc
  switched = toggles.reads(f, toggles.SOFT_SHADOWS)
  setting = toggles.setting_on(f)
  VisT, _vis_of = punctual.visibility_ops(f.tinted_shadows)
  shadow_ray = make_shadow_ray(f, q)
  sun_visibility = make_sun_visibility(f, shadow_ray, runtime=False)
  soft_visibility = make_soft_visibility(f, q, runtime=False)
  run_sun = make_sun_visibility(f, shadow_ray, runtime=True) if switched else compiled_out
  run_soft = make_soft_visibility(f, q, runtime=True) if switched else compiled_out

  @wp.func
  @concrete(**{"return": VisT})
  @device_generic
  def light_visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    light: Light,
    origin: wp.vec3,
    normal: wp.vec3,
    l: wp.vec3,
    dist: float,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> Visibility:
    """Visibility of punctual light `li` from a shading point (1: unoccluded); `origin`: its
    secondary-ray origin (`Surface.origin`), `l` / `dist`: direction and distance to the light.

    Soft shadows (`shadow_samples` > 1) trace rays to a disc of radius `radius` around the light:
    a flat emitter's own disc (cosine lights: rects / disks in point form, facing `light.dir`),
    else the bulb's silhouette (facing the point).
    """
    if wp.static(switched):
      if setting(geo.rt.flags, geo.bits2, wp.static(toggles.SOFT_SHADOWS)):
        if light.type == wp.static(punctual.LIGHT_DIRECTIONAL):
          return run_sun(geo, worldid, light, origin, normal, l, seed, stream, nlight, li)
        return run_soft(geo, worldid, light, origin, l, dist, seed, stream, nlight, li)
    # Static branches with an else: the branch not taken is not emitted (code after a static
    # return still is, with its own inlined shadow-ray traversal).
    if light.type == wp.static(punctual.LIGHT_DIRECTIONAL):
      if wp.static(sun_disc > 0):
        return sun_visibility(geo, worldid, light, origin, normal, l, seed, stream, nlight, li)
      else:
        return shadow_ray(geo, worldid, origin, l, wp.static(UNBOUNDED_SHADOW_DIST))
    if wp.static(shadow_samples <= 1):
      return shadow_ray(geo, worldid, origin, l, dist * wp.static(area.SHADOW_END))
    else:
      return soft_visibility(geo, worldid, light, origin, l, dist, seed, stream, nlight, li)

  return light_visibility
