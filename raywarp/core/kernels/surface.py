"""How a surface a camera ray hits is colored: the light it reflects from the environment (split-
sum IBL, GI probes, the traced glossy reflection) and the lights (direct lights,
headlight), plus its own emission.

  env_radiance                  the environment seen by a ray (backgrounds, probe rays)
  make_indirect                 the occluded split-sum ambient radiance with the traced glossy
                                reflection, under the coat and fuzz (`kernels/ambient.py`)
  make_shade_surface            the outgoing radiance of a shaded hit (`shade_surface`)
  shade_unlit                   the same for `surfaces.shading="unlit"`
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, cast, overload

import warp as wp

from raywarp.core.kernels import ambient
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params
from raywarp.core.kernels.abi.tables import emission_radiance, exposure_at
from raywarp.core.kernels.ambient import horizon_fade
from raywarp.core.kernels.first_hit import emission_term
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.kernels.lighting import DirectLighting
from raywarp.core.shade.environment import envmap
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials import material
from raywarp.core.shade.materials.material import Lobes, MaterialTables, Surface

if TYPE_CHECKING:
  from raywarp.core.kernels.reflections import Glossy

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Floor of the environment background's mip level argument (log2 of a texel count).
_MIN_TEXELS: Final = 1.0e-8

# (Argument lists: the factories' docstrings; ruff format would put one argument type per line.)
# fmt: off
# shade_surface(geo, lit, env, p, worldid, mats, surface, point, normal, view, footprint, spread,
# headlight_dir, seed, stream) -> R (`make_shade_surface`).
type _SurfaceFunc[R] = Callable[[GeometryFields, Lights, EnvironmentArrays, Params, int, MaterialTables, Surface, wp.vec3,
  wp.vec3, wp.vec3, float, float, wp.vec3, int, int], R]
# fmt: on
# -> outgoing radiance (`make_shade_surface`, `shade_unlit`).
type ShadeSurface = _SurfaceFunc[wp.vec3]
# Split -> (radiance, direct, indirect, emission).
type ShadeSurfaceSplit = _SurfaceFunc[tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]]


@wp.func(inline=True)
def env_lod(
  # Model:
  p: Params,
  # In:
  spread: float,
) -> float:
  """Mip level of an environment radiance map whose texels match a ray cone of angular `spread`."""
  return wp.log2(wp.max(spread * p.inv_texel_angle, wp.static(_MIN_TEXELS)))


@wp.func
def env_radiance(
  # Model:
  env: EnvironmentArrays,
  p: Params,
  # In:
  worldid: int,
  direction: wp.vec3,
  spread: float,
) -> wp.vec3:
  """Environment radiance seen along `direction` by a ray cone of angular `spread` (the mip
  level whose texels match the cone): the background of camera and probe rays."""
  e, yaw, scale = kernel_structs.env_lookup(env, worldid)
  return envmap.sample_radiance(env.env_radiance[e], direction, yaw, env_lod(p, spread)) * scale


@specialize
def make_indirect(f: Features, glossy: Glossy | None):
  """indirect(...) -> the ambient radiance of a hit: the base lobes' diffuse + specular ambient, the
  specular's environment term mixed with the traced glossy reflection (`glossy`; misses see the
  unoccluded environment), under the coat and fuzz layers (`kernels/ambient.py`)."""
  traced = glossy is not None
  # Only the `traced` specialization calls this; Warp removes the other branch.
  trace_glossy = cast("Glossy", glossy)
  diffuse_ambient = ambient.make_diffuse_ambient(f)
  lobe_ambient = ambient.make_lobe_ambient(f)
  base_ambient = ambient.make_base_ambient(f)
  layered_ambient = ambient.make_layered_ambient(f)
  layered = f.clearcoat or f.fuzz
  anisotropy = f.specular_anisotropy
  horizon = f.horizon_fade
  gate = toggles.bound_on(f.bound)  # the coat and layers class bits, the horizon fade (`Lobes.bits`)
  field = toggles.bound_on(f.bound, high=True)  # the layered lobes' field bits (`toggles.LAYER_*`)

  @wp.func
  @device_generic
  def indirect[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    lobes: Lobes,
    surface: Surface,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> wp.vec3:
    """Image-based / ambient radiance towards `view` (stream + 1: the glossy reflection ray)."""
    ao = surface.occlusion
    fill, irradiance, probes, _sky_seen = diffuse_ambient(env, worldid, point, normal, surface.outward, ao)
    # An anisotropic lobe looks its environment up about its bent normal (an approximation of the
    # stretched lobe, `brdf.anisotropic_bent_normal`).
    spec_normal = normal
    if wp.static(anisotropy):
      if gate(lobes.bits, wp.static(toggles.CLASS_LAYERS)) and field(lobes.bits2, wp.static(toggles.LAYER_ANISOTROPY)):
        spec_normal = material.bent_normal(lobes, normal, view)
    specular, unoccluded = lobe_ambient(
      env, worldid, spec_normal, view, nov, lobes.alpha, lobes.perceptual_roughness, fill, probes, ao, spread
    )
    if wp.static(traced):
      # The traced reflection in place of the environment's share w of the specular radiance
      # (same split-sum weight; occluded by the material's AO only: the ray sees the scene).
      ray, w = trace_glossy(
        geo,
        mats,
        lit,
        env,
        p,
        worldid,
        lobes,
        surface.outward,
        surface.origin,
        normal,
        view,
        nov,
        footprint,
        spread,
        unoccluded,
        headlight_dir,
        seed,
        stream + 1,
      )
      if w > 0.0:
        shown = ambient.specular_occlusion(nov, ao, lobes.alpha) * w
        specular = specular * (1.0 - w) + ray * shown
    if wp.static(horizon):
      if gate(lobes.bits, wp.static(toggles.SCENE_HORIZON_FADE)):
        specular = specular * horizon_fade(view, normal, surface.outward)
    radiance = base_ambient(lobes, nov, irradiance, specular)
    if wp.static(layered):
      if gate(lobes.bits, wp.static(toggles.CLASS_COAT | toggles.CLASS_LAYERS)):
        radiance = layered_ambient(env, worldid, lobes, surface, normal, view, nov, fill, probes, spread, radiance)
    return radiance

  return indirect


@wp.func
def coat_emission(
  # In:
  lobes: Lobes,
  emissive: wp.vec3,
) -> wp.vec3:
  """Emission under an OpenPBR coat: lerp(1, T_coat, C) E (OpenPBR 1.1; T along the view twice:
  the emitted light crosses the coat once, T^(1 / mu_o') with T² = coat_color)."""
  if lobes.coat > 0.0 and lobes.coat_openpbr:
    t2 = material.coat_transmittance(lobes.coat_color, lobes.coat_ior, lobes.coat_nov, lobes.coat_nov)
    single = wp.vec3(wp.sqrt(t2[0]), wp.sqrt(t2[1]), wp.sqrt(t2[2]))  # T^(1 / mu_o')
    return wp.cw_mul(emissive, wp.vec3(1.0 - lobes.coat) + single * lobes.coat)
  return emissive


@wp.func
@device_generic
def shade_unlit[G: GeometryFields, M: MaterialTables](
  # Model:
  _geo: G,
  _lit: Lights,
  env: EnvironmentArrays,
  p: Params,
  # In:
  worldid: int,
  _mats: M,
  surface: Surface,
  _point: wp.vec3,
  _normal: wp.vec3,
  _view: wp.vec3,
  _footprint: float,
  _spread: float,
  _headlight_dir: wp.vec3,
  _seed: int,
  _stream: int,
) -> wp.vec3:
  """shade_surface(...) of `surfaces.shading="unlit"` (`make_shade_surface`'s signature): the
  surface's base color as display-relative radiance, plus its emission as a lit render shows it
  (lamps and emissive panels stay visible to a policy; `Params.relative_emission`). No light,
  shadow, environment, GI or reflection term (`canonical.canonical` collapses their Features); the
  other arguments are unused."""
  exposure = exposure_at(env.exposure, worldid)
  if p.relative_emission != 0:
    return (surface.base_color + surface.emissive) / exposure
  return surface.base_color / exposure + surface.emissive


@overload
def make_shade_surface(
  f: Features, lighting: DirectLighting, glossy: Glossy | None, *, split: Literal[False] = False
) -> ShadeSurface: ...


@overload
def make_shade_surface(
  f: Features, lighting: DirectLighting, glossy: Glossy | None, *, split: Literal[True]
) -> ShadeSurfaceSplit: ...


@overload
def make_shade_surface(
  f: Features, lighting: DirectLighting, glossy: Glossy | None, *, split: bool
) -> ShadeSurface | ShadeSurfaceSplit: ...


@specialize
def make_shade_surface(
  f: Features, lighting: DirectLighting, glossy: Glossy | None, *, split: bool = False
) -> ShadeSurface | ShadeSurfaceSplit:
  """shade_surface(geo, lit, env, p, worldid, mats, surface, point, normal, view, footprint,
  spread, headlight_dir, seed, stream) -> outgoing radiance (before exposure).

  `glossy`: the traced glossy reflection (`reflections.make_glossy`) in place of the environment's
  specular radiance, or None (no reflection ray). `split` (`ShadeSurfaceSplit`): also the lights'
  part, the rest but the emission and the emission (under the coat)."""
  shading_lobes = material.lobes_of(f)
  indirect = make_indirect(f, glossy)
  direct_lights = lighting.direct_lights
  headlight = f.headlight
  relative_emission = material.make_relative_emission(f)
  on = toggles.on(f)
  gate = toggles.bound_on(f.bound)  # the layers class bit (`Lobes.bits`)

  @wp.func
  @concrete(**{"return": tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3] if split else wp.vec3})
  @device_generic
  def shade_surface[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    mats: M,
    surface: Surface,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> wp.vec3 | tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    """Outgoing radiance towards `view` (before exposure); `footprint` / `spread`: the ray cone at
    the point (for the glossy reflection ray)."""
    nov = wp.max(wp.dot(normal, view), 1.0e-4)
    lobes = shading_lobes(env.dfg_lut, surface, nov)
    radiance = indirect(
      geo,
      mats,
      lit,
      env,
      p,
      worldid,
      lobes,
      surface,
      point,
      normal,
      view,
      nov,
      footprint,
      spread,
      headlight_dir,
      seed,
      stream,
    )
    if wp.static(split):
      indirect_part = radiance  # the light loop adds to it: the split's indirect part
    exposure = exposure_at(env.exposure, worldid)
    # The headlight: one more entry of the light loop (display-relative: its illuminance follows
    # the world's exposure); zero: off.
    head = wp.vec3(0.0)
    if wp.static(headlight):
      if on(geo.rt.flags, wp.static(toggles.HEADLIGHT)):
        head = p.headlight_color * (p.headlight_intensity / exposure)
    radiance = direct_lights(
      geo, lit, worldid, lobes, point, normal, surface.origin, view, nov, seed, stream, headlight_dir, head, radiance
    )

    if wp.static(f.coat_layer):
      if gate(lobes.bits, wp.static(toggles.CLASS_LAYERS)):
        surface.emissive = coat_emission(lobes, surface.emissive)
    relative = relative_emission(p.relative_emission, surface)
    if wp.static(split):
      emitted = emission_term(surface, exposure, relative)  # under the coat (`coat_emission`)
      return radiance + emitted, radiance - indirect_part, indirect_part, emitted  # pyrefly: ignore[unbound-name]
    else:
      return radiance + emission_radiance(surface.emissive, exposure, relative)

  # `concrete` gives Warp this exact return type; the host checker cannot narrow a
  # callable's union return from the factory's compile-time flag.
  return cast(ShadeSurfaceSplit, shade_surface) if split else cast(ShadeSurface, shade_surface)
