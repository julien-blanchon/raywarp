"""The color of one surface a camera ray meets, and how translucent layers stack up front to back.

One camera-ray hit of the megakernel shaded (`make_shade_layer`); `pixel.make_trace_pixel`
composites the layers front to back (`composite`, as the GI probe rays do). A MuJoCo
`reflectance` is the specular lobe's F0 (`evaluate.mirror_reflectance`): its reflection is the
glossy reflection of every specular lobe (`kernels/reflections.py`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, cast

import warp as wp

from raywarp._base.types import CLOSED_PANE, MEDIUM_IOR, ROW_MEDIUM
from raywarp.core.kernels import first_hit
from raywarp.core.kernels.abi import rays, toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params
from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.kernels.surface import ShadeSurface, ShadeSurfaceSplit
from raywarp.core.shade.environment import ibl
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials import brdf
from raywarp.core.shade.materials.material import MaterialTables, Surface, make_relative_emission

if TYPE_CHECKING:
  from raywarp.core.shade.materials.evaluate import EvalSurface

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Stop compositing translucent layers once the remaining weight cannot change an 8-bit pixel.
MIN_THROUGHPUT: Final = 1.0 / 255.0
# The cone widening per unit alpha of a rough thin-walled transmissive layer: both interfaces of a
# wall of IOR 1.5, (1 - 1 / 1.5) entering + (1.5 - 1) leaving (`transmit_step`).
_THIN_WALL_SPREAD: Final = 5.0 / 6.0

# (Argument lists: the factories' docstrings; ruff format would put one argument type per line.)
# fmt: off
# shade_layer(geo, mats, lit, env, p, worldid, hit, point, direction, cone, spread, headlight_dir,
# seed, stream) -> R (`make_shade_layer`).
type _LayerFunc[R] = Callable[[GeometryFields, MaterialTables, Lights, EnvironmentArrays, Params, int, rays.Hit, wp.vec3,
  wp.vec3, float, float, wp.vec3, int, int], R]
# fmt: on
# -> (radiance, its `Coverage`, normal, its `first_hit.SurfaceRecord`).
type ShadeLayer = _LayerFunc[tuple[wp.vec3, Coverage, wp.vec3, first_hit.SurfaceRecord]]


@wp.func(inline=True)
def composite(
  # In:
  radiance: wp.vec3,
  throughput: float,
  shaded: wp.vec3,
  opacity: float,
) -> tuple[wp.vec3, float]:
  """(radiance, throughput) after a layer of `opacity` shaded `shaded` (front-to-back "over")."""
  return radiance + shaded * (throughput * opacity), throughput * (1.0 - opacity)


@wp.func
def leaves_solid(
  # Model:
  instance_optics: wp.array2d[wp.vec4],
  # In:
  instance_id: int,
  normal: wp.vec3,
  direction: wp.vec3,
) -> bool:
  """Whether a ray along `direction` hitting instance `instance_id` (geometric normal `normal`)
  leaves a solid there: a back face of a refraction index > 0 (`pipeline/upload/optics.py`)."""
  return instance_optics[instance_id, wp.static(ROW_MEDIUM)][wp.static(MEDIUM_IOR)] > 0.0 and wp.dot(normal, direction) >= 0.0


@wp.func
def absorb_exit(
  # Model:
  instance_optics: wp.array2d[wp.vec4],
  # In:
  instance_id: int,
  normal: wp.vec3,
  direction: wp.vec3,
  dist: float,
  tint: wp.vec3,
  shaded: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(`tint`, `shaded`) at a transmissive layer of instance `instance_id` (geometric normal
  `normal`) reached after `dist`. Leaving a solid (a back face of a refraction index > 0): the ray
  crossed `dist` of its medium, Beer-Lambert absorbed by its optics row's extinction
  (`pipeline/upload/optics.py`), and the face adds no light of its own (its shading would reflect
  the lights outside the solid off the inside of the interface: a thin glass box would show its
  highlight twice). No medium state is carried between layers: the camera ray's layer loop keeps
  nothing live across its traces for it."""
  if not leaves_solid(instance_optics, instance_id, normal, direction):
    return tint, shaded
  row = instance_optics[instance_id, wp.static(ROW_MEDIUM)]
  return wp.cw_mul(tint, wp.vec3(wp.exp(-row[0] * dist), wp.exp(-row[1] * dist), wp.exp(-row[2] * dist))), wp.vec3(0.0)


@wp.func
def transmit_step(
  # Model:
  instance_optics: wp.array2d[wp.vec4],
  # In:
  instance_id: int,
  normal: wp.vec3,
  roughness: float,
  direction: wp.vec3,
  spread: float,
) -> tuple[wp.vec3, float]:
  """(direction, cone spread) of a camera ray past a transmissive layer of instance `instance_id`
  with geometric normal `normal` and perceptual roughness `roughness` (`pipeline/upload/optics.py`'s
  rows).

  * a solid (refraction index > 0) bends the ray by Snell's law, 1 / ior entering (`normal`
    facing the ray), ior leaving; past the critical angle it reflects and stays inside (its
    medium's absorption: `absorb_exit` at the face it leaves through).
  * a thin-walled one (index 0) keeps the ray straight.
  * a rough one (frosted glass) widens the ray's cone by what a microfacet tilted by alpha
    (= roughness^2) turns the refracted ray, |1 - eta| alpha (eta = n_in / n_out: 1 / ior entering,
    ior leaving), not a reflection's 2 alpha: glass of IOR 1.5 blurs 1/3 alpha entering, 1/2
    alpha leaving. A thin wall crosses both interfaces at once, `_THIN_WALL_SPREAD` alpha. What
    is behind is read through blurrier texture and environment mips. Cheap: the geometry behind
    stays sharp.
  """
  ior = instance_optics[instance_id, wp.static(ROW_MEDIUM)][wp.static(MEDIUM_IOR)]
  alpha = roughness * roughness
  if ior <= 0.0:
    return direction, spread + wp.static(_THIN_WALL_SPREAD) * alpha
  entering = wp.dot(normal, direction) < 0.0
  n = wp.where(entering, normal, -normal)
  eta = wp.where(entering, 1.0 / ior, ior)
  cosi = -wp.dot(n, direction)
  k = 1.0 - eta * eta * (1.0 - cosi * cosi)
  if k < 0.0:  # total internal reflection: the ray stays in the medium
    return wp.normalize(direction + 2.0 * cosi * n), spread + 2.0 * alpha
  spread = spread + wp.abs(1.0 - eta) * alpha
  return wp.normalize(eta * direction + (eta * cosi - wp.sqrt(k)) * n), spread


@wp.struct
class Coverage:
  """What compositing a translucent layer over the ones behind reads of its surface
  (`make_shade_layer`: evaluated before the shading's shadow and reflection rays, so that only
  these values, not the `Surface`, are live across them).

  Attributes:
    opacity: the layer's coverage in [0, 1] (a closed see-through pane's both faces, `_folded`).
    reflected: a transmissive layer's specular albedo E_spec(NoV), its RGB mean in [0, 1].
    color: a transmissive layer's base color clamped to [0, 1] (the transmission tint).
    specular_only: a translucent layer of a transmissive material.
    roughness: its perceptual roughness (the frosted glass's cone widening, `transmit_step`).
    base_color: the surface's base color (the first surface's guide, `pixel.make_trace_pixel`).
  """

  opacity: float
  reflected: float
  color: wp.vec3
  specular_only: bool
  roughness: float
  base_color: wp.vec3


@wp.func
@device_generic
def _folded[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  instance_id: int,
  alpha: float,
  specular_only: bool,
) -> float:
  """The coverage of a layer of coverage `alpha` on instance `instance_id`: a closed see-through
  pane's (`CLOSED_PANE`, set at upload; its back faces culled) covers for both its faces,
  1 - (1 - alpha)^2; other layers `alpha`. An opaque or glass (`specular_only`: never a pane)
  layer reads no flag."""
  if alpha >= 1.0 or specular_only or instance_id < 0:
    return alpha
  if (geo.instance_twosided[instance_id] & wp.static(CLOSED_PANE)) != 0:
    a = wp.clamp(alpha, 0.0, 1.0)
    return a * (2.0 - a)
  return alpha


@specialize
def make_coverage(f: Features, *, fold: bool):
  """coverage(geo, env, surface, normal, direction, hit) -> the `Coverage` of the layer `surface`
  at `hit` seen along `direction` (shading normal `normal`): Features.transmissive's
  transmission terms; `fold`: closed panes folded (`_folded`: camera rays, their back faces
  culled, where a tier build's culling switch is on; probe rays meet both faces)."""
  transmissive = f.transparency and f.transmissive
  fold_panes = fold
  setting = toggles.setting_on(f)  # culling: a tier build reads it at run time

  @wp.func(inline=True)
  @device_generic
  def coverage[G: GeometryFields](
    # Model:
    geo: G,
    env: EnvironmentArrays,
    # In:
    surface: Surface,
    normal: wp.vec3,
    direction: wp.vec3,
    hit: rays.Hit,
  ) -> Coverage:
    """The layer's coverage terms (`Coverage`)."""
    c = Coverage()
    alpha = surface.alpha
    if wp.static(fold_panes):
      if setting(geo.rt.flags, geo.bits2, wp.static(toggles.BACKFACE_CULLING)):
        alpha = _folded(geo, hit.instance_id, alpha, surface.specular_only)
    c.opacity = wp.clamp(alpha, 0.0, 1.0)
    c.specular_only = surface.specular_only
    c.roughness = surface.perceptual_roughness
    c.base_color = surface.base_color
    if wp.static(transmissive):
      if surface.specular_only:
        nov = wp.max(wp.dot(normal, -direction), 1.0e-4)
        pr, _alpha = brdf.roughness_to_alpha(surface.perceptual_roughness)
        f0 = brdf.surface_f0(surface.base_color, surface.metallic, surface.reflectance)
        e = brdf.specular_albedo(f0, ibl.sample_dfg(env.dfg_lut, nov, pr))
        c.reflected = wp.clamp((e[0] + e[1] + e[2]) * wp.static(1.0 / 3.0), 0.0, 1.0)
        b = surface.base_color
        c.color = wp.vec3(wp.clamp(b[0], 0.0, 1.0), wp.clamp(b[1], 0.0, 1.0), wp.clamp(b[2], 0.0, 1.0))
    return c

  return coverage


@wp.func
def composite_transmissive(
  # In:
  radiance: wp.vec3,
  throughput: float,
  tint: wp.vec3,
  shaded: wp.vec3,
  cover: Coverage,
) -> tuple[wp.vec3, float, wp.vec3]:
  """(radiance, throughput, tint) after a layer of coverage `cover`, with transmission semantics.

  Features.transmissive; `tint`: the color of the transmissive layers in front, which the light
  behind them is seen through:

  * a translucent layer of a transmissive material (`Coverage.specular_only`: physical
    transmission, Filament's TRANSPARENT blending, document §5.5.1; opacity 1 - t, the diffuse
    lobe already weighted by it) adds its shading unscaled by the opacity (a clear pane's
    reflection stays though it is invisible) and transmits (1 - opacity) of the light behind it
    that its Fresnel lets through, 1 - E_spec(NoV) (energy: reflection + diffuse + transmission
    <= 1; clear glass passes 1 - E_spec, OpenPBR's thin-walled interface), tinted by its base
    color (the derived `transmission_color`);
  * any other layer (coverage: OpenPBR `geometry_opacity`, glTF BLEND, MuJoCo rgba alpha;
    Filament's FADE) scales all its light by its opacity (`composite`).
  """
  if not cover.specular_only:
    out, through = composite(radiance, throughput, wp.cw_mul(shaded, tint), cover.opacity)
    return out, through, tint
  return (
    radiance + wp.cw_mul(shaded, tint) * throughput,
    throughput * (1.0 - cover.opacity) * (1.0 - cover.reflected),
    wp.cw_mul(tint, cover.color),
  )


@specialize
def make_shade_layer(
  f: Features,
  *,
  eval_surface: EvalSurface,
  shade_surface: ShadeSurface | ShadeSurfaceSplit,
):
  """shade_layer(...) -> (radiance, coverage, normal, its `first_hit.SurfaceRecord`) of one
  camera-ray hit (its `Coverage`, read before the shading's rays; the record: the surface
  outputs', `first_hit.capture`), the direct / indirect split from `shade_surface` built with
  `split` (`first_hit.split_lighting`). Features.world_optics: the coverage reads the world's
  closed-pane bits (`rays.world_hit`)."""
  split = first_hit.split_lighting(f)
  world_optics = f.world_optics
  coverage = make_coverage(f, fold=f.transparency and f.backface_culling)
  relative_emission = make_relative_emission(f)
  # The caller builds shade_surface with the same split flag. Each alias is called
  # only in its matching wp.static branch; neither adds a device wrapper.
  shade_split = cast(ShadeSurfaceSplit, shade_surface)
  shade_plain = cast(ShadeSurface, shade_surface)

  @wp.func
  @concrete(**{"return": tuple[wp.vec3, Coverage, wp.vec3, first_hit.SurfaceRecord]})
  @device_generic
  def shade_layer[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    hit: rays.Hit,
    point: wp.vec3,
    direction: wp.vec3,
    cone: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, Coverage, wp.vec3, first_hit.SurfaceRecord]:
    """The hit's surface shaded towards the ray."""
    surface, normal = eval_surface(geo, mats, worldid, hit, point, direction, cone)
    if wp.static(world_optics):
      cover = coverage(geo, env, surface, normal, direction, rays.world_hit(hit, geo.instance_type.shape[0], worldid))
    else:
      cover = coverage(geo, env, surface, normal, direction, hit)
    if wp.static(split):
      shaded, direct, rest, coated = shade_split(
        geo, lit, env, p, worldid, mats, surface, point, normal, -direction, cone, spread, headlight_dir, seed, stream
      )
    else:
      shaded = shade_plain(
        geo, lit, env, p, worldid, mats, surface, point, normal, -direction, cone, spread, headlight_dir, seed, stream
      )
    if wp.static(split):
      # The emission under the coat (the split is on for the emission output of coated
      # materials: `first_hit.split_lighting`).
      rec = first_hit.capture(surface, normal, point, coated, direct, rest)  # pyrefly: ignore[unbound-name]  # bound under `split`
    else:
      relative = relative_emission(p.relative_emission, surface)
      emitted = first_hit.emission_term(surface, exposure_at(env.exposure, worldid), relative)
      rec = first_hit.capture(surface, normal, point, emitted, wp.vec3(0.0), wp.vec3(0.0))
    return shaded, cover, normal, rec

  return shade_layer
