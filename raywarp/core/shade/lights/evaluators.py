"""The per-light evaluators of the light loops (`loop.Evaluators`): punctual lights and LTC
area lights, their unshadowed radiance and their shadow rays."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import warp as wp

from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.lights import area as area_lights
from raywarp.core.shade.lights import ies as ies_lib
from raywarp.core.shade.lights import ltc
from raywarp.core.shade.lights import punctual as lights
from raywarp.core.shade.lights.loop import AreaVisibility, Evaluators, Prepared
from raywarp.core.shade.lights.punctual import Light, Lights, Visibility
from raywarp.core.shade.materials.material import Lobes

if TYPE_CHECKING:
  from raywarp.core.kernels.lighting import BsdfEval
  from raywarp.core.shade.shadows.visibility import LightVisibility, ShadowPass

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def _punctual_context(
  # Model:
  _lit: Lights,
  # In:
  _lobes: Lobes,
  _normal: wp.vec3,
  _view: wp.vec3,
  _nov: float,
) -> int:
  """No per-point state (punctual lights)."""
  return 0


@wp.func
def source_widening(
  # In:
  light: Light,
  dist: float,
) -> float:
  """The roughness a light's size adds to the specular lobes of its punctual form (`bsdf_eval`'s
  `widen`, added in quadrature: alpha' = sqrt(alpha^2 + widen^2)): half its angular radius, the GGX
  normal spread that reflects its whole disc (the roughness-widening approximation of sphere
  lights, Karis 2013 "Real Shading in Unreal Engine 4"; in quadrature a lobe much wider than the
  source keeps its peak, as the LTC integral of a small rect shows):
  a mirror shows a lamp, a sun or an emissive panel shaded in point form as a soft spot of its
  size, not a dot, at the same reflected energy. `light.radius`: the emitter's radius [m] (a rect
  or disk: the disc of the same area, `upload/lights`), a directional light's angular radius
  [rad]. A punctual light
  (`shape` NONE: a point or spot, a `treatAsPoint` sphere, a MuJoCo bulb) adds none: its radius
  softens its shadows only."""
  if light.type == wp.static(lights.LIGHT_DIRECTIONAL):
    return 0.5 * light.radius
  if light.shape == wp.static(area_lights.SHAPE_NONE):
    return 0.0
  return 0.5 * light.radius / wp.max(dist, 1.0e-6)


def make_punctual(
  *,
  has_spot: bool,
  bsdf_eval: BsdfEval,
  light_visibility: LightVisibility[Visibility],
  wrapped: bool = False,
  ies: bool = False,
  bound: int = 0,
):
  """The evaluators of punctual lights (point, spot, directional, cosine, tube point forms; no
  per-point context): the light loop's, and the area-light evaluators' for their punctual lights.

  `bsdf_eval(lobes, normal, view, nov, l, nol, scale, widen)`: the shading's BSDF times
  NoL (`kernels/lighting.make_bsdf_eval`);
  `light_visibility(geo, worldid, light, origin, normal, l, dist, seed, stream, nlight, li)`:
  the shading's shadow rays of a `lights.Light`; `has_spot` as Features; `wrapped`
  (Features.fuzz): a cloth's wrapped diffuse lights it up to NoL = -w (`Lobes.wrap`); `ies`
  (Features.ies): the radiance scale times the light's profile (`shade/lights/ies.py`; the table is
  read only in a profiled light's term: no per-point context). `bound`
  (Features.bound): the IES profiles and the spot cones read the build word's bits
  (`toggles.CLASS_IES`, `SCENE_SPOT`; `toggles.bound_on`).
  """
  gate = toggles.bound_on(bound)

  @wp.func(inline=True)
  def term(
    # Model:
    lit: Lights,
    # In:
    light: Light,
    lobes: Lobes,
    _ctx: int,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ) -> tuple[wp.vec3, bool, wp.vec3, float]:
    """(unshadowed radiance, casts shadows, direction to the light, distance) of `light`: the
    prepared term (`loop.Prepared`), its direction and distance reused by the shadow rays."""
    if not light.active:
      return wp.vec3(0.0), False, wp.vec3(0.0), 0.0
    spotted = False
    if wp.static(has_spot):
      spotted = gate(lit.bits, wp.static(toggles.SCENE_SPOT))
    l, dist, scale = lights.eval_light(light, spotted, point, lit.falloff_floor)
    if wp.static(ies):
      if gate(lit.bits, wp.static(toggles.CLASS_IES)):
        scale = scale * ies_lib.factor(lit.ies, light, l)
    nol = wp.dot(normal, l)
    lit_from = float(0.0)
    if wp.static(wrapped):
      lit_from = -lobes.wrap
    # Behind the surface, or out of the light's range / cone / profile: no light.
    if nol <= lit_from or wp.length_sq(scale) == 0.0:
      return wp.vec3(0.0), False, wp.vec3(0.0), 0.0
    # The light's size widens its specular lobes (`source_widening`): a sized light is no
    # dot in a mirror.
    widen = source_widening(light, dist)
    return bsdf_eval(lobes, normal, view, nov, l, nol, scale, widen), light.cast_shadow, l, dist

  @wp.func(inline=True)
  def unshadowed(
    # Model:
    lit: Lights,
    # In:
    light: Light,
    lobes: Lobes,
    ctx: int,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ) -> tuple[wp.vec3, bool]:
    """(unshadowed radiance, casts shadows) of `light`: `term` without its direction and distance
    (the capped and budgeted loops keep no per-slot geometry)."""
    c, caster, _direction, _distance = term(lit, light, lobes, ctx, point, normal, view, nov)
    return c, caster

  @wp.func
  @device_generic
  def visibility[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    light: Light,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    seed: int,
    stream: int,
  ):
    """Visibility of punctual `light` from `point` (the shading's shadow rays, from `origin`)."""
    spotted = False
    if wp.static(has_spot):
      spotted = gate(lit.bits, wp.static(toggles.SCENE_SPOT))
    l, dist, _scale = lights.eval_light(light, spotted, point, lit.falloff_floor)
    return light_visibility(geo, worldid, light, origin, normal, l, dist, seed, stream, lit.nlight, li)

  return Evaluators(_punctual_context, unshadowed, visibility, int, Prepared(term, light_visibility))


def make_area_evaluators(
  *,
  punctual: Evaluators[int],
  ies: bool,
  shadow_samples: int,
  shadow_pass: ShadowPass,
  specular_curve: bool,
  tinted: bool,
  clearcoat: bool,
  specular_weight: bool = False,
  layers: tuple[bool, bool, bool, bool] = (False, False, False, False),
  bound: int = 0,
  switches: int = 0,
):
  """LTC area radiance/visibility with shared punctual evaluators and an LTC-frame context (`ies`:
  Features.ies, a profiled light takes the punctual evaluators). `bound` (Features.bound): the
  area lights and the IES profiles read the build word's bits (`toggles.CLASS_AREA` / `CLASS_IES`):
  with the area bit off every light takes the punctual evaluators and no LTC frame is set up.

  `punctual`: `make_punctual`'s evaluators; `shadow_samples`, `shadow_pass`: the area shadow rays
  (`make_area_visibility`); `tinted`: the visibility is an RGB transmittance (tinted shadows,
  `lights.visibility_ops`); `specular_curve`, `clearcoat`, `specular_weight` and `layers`
  (Features specular_anisotropy, fuzz, coat_normal, coat_layer): the lobes of the LTC frame and
  integrals (`ltc.make_frame_func`, `area_lights.make_eval_area_light`). A profiled area light is
  shaded in its point form (the punctual evaluators). `switches`: the build's run-time switches
  (the soft shadows' sampling, `make_area_visibility`); `lighting.area_lights="point"` packs the
  lights without their shape (`pack`), so they take the punctual evaluators."""
  anisotropy, fuzz, coat_normal, coat_layer = layers
  gate = toggles.bound_on(bound)  # the area and IES class bits (`bound`: Features.bound)
  # The material class bits (`bound`) of the area lights' lobes read the word's material copy
  # (`Lobes.bits`).
  ltc_frame = ltc.make_frame_func(
    specular_curve=specular_curve,
    clearcoat=clearcoat,
    specular_weight=specular_weight,
    anisotropy=anisotropy,
    coat_normal=coat_normal,
    coat_layer=coat_layer,
    fuzz=fuzz,
    bound=bound,
  )
  eval_area = area_lights.make_eval_area_light(
    clearcoat=clearcoat, anisotropy=anisotropy, fuzz=fuzz, coat_normal=coat_normal, coat_layer=coat_layer, bound=bound
  )
  area_visibility = make_area_visibility(shadow_pass, shadow_samples=shadow_samples, soft=(switches, bound))
  _VisT, vis_of = lights.visibility_ops(tinted)
  punctual_unshadowed, punctual_visibility = punctual.unshadowed, punctual.visibility

  @wp.func
  def frame_of(
    # Model:
    lit: Lights,
    # In:
    lobes: Lobes,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ) -> ltc.Frame:
    """The LTC frame of the point (none with the area class bit off)."""
    if gate(lit.bits, wp.static(toggles.CLASS_AREA)):
      return ltc_frame(lit, lobes, normal, view, nov)
    return ltc.Frame()

  @wp.func(inline=True)
  def is_area(
    # Model:
    lit: Lights,
    # In:
    light: Light,
  ) -> bool:
    """Whether `light` takes the LTC terms: an area shape (the area class bit on), not profiled."""
    area = gate(lit.bits, wp.static(toggles.CLASS_AREA))
    if area:
      area = light.shape != wp.static(area_lights.SHAPE_NONE)
    if wp.static(ies):
      if area and gate(lit.bits, wp.static(toggles.CLASS_IES)):
        area = light.ies == 0
    return area

  @wp.func
  def area_term(
    # In:
    light: Light,
    lobes: Lobes,
    fr: ltc.Frame,
    point: wp.vec3,
  ) -> tuple[wp.vec3, bool]:
    """(LTC radiance of active area light `light`, whether it casts shadows)."""
    a = area_lights.area_of(light)
    c = eval_area(a, fr, lobes, point)
    # A light that does not reach the point is no caster: the shadow budget would trace it.
    return c, a.cast_shadow and wp.length_sq(c) > 0.0

  @wp.func
  def unshadowed(
    # Model:
    lit: Lights,
    # In:
    light: Light,
    lobes: Lobes,
    fr: ltc.Frame,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ) -> tuple[wp.vec3, bool]:
    """(unshadowed radiance of `light`, whether it casts shadows)."""
    if not is_area(lit, light):
      return punctual_unshadowed(lit, light, lobes, 0, point, normal, view, nov)
    if not light.active:
      return wp.vec3(0.0), False
    return area_term(light, lobes, fr, point)

  @wp.func
  @device_generic
  def visibility[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    light: Light,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    seed: int,
    stream: int,
  ):
    """Visibility of `light` (area: sampled points; punctual: the megakernel's rays)."""
    if not is_area(lit, light):
      return punctual_visibility(geo, lit, worldid, light, li, point, normal, origin, seed, stream)
    return vis_of(area_visibility(geo, worldid, area_lights.area_of(light), origin, seed, stream, lit.nlight, li))

  return Evaluators(frame_of, unshadowed, visibility, ltc.Frame)


def make_area_visibility(shadow_pass: ShadowPass, *, shadow_samples: int, soft: tuple[int, int] = (0, 0)):
  """area_visibility(geo, worldid, light, origin, seed, stream, nlight, li) -> visible fraction.

  `shadow_samples` rays from `origin` (the shading point's `Surface.origin`) to points of the light
  (`area_lights.sample_rect` / `sample_curved`): the fixed pattern (`area_lights.fixed_rand2`
  shifted per pixel; one sample: the centre); `shadow_pass`: one shadow ray's transmitted fraction
  (`shadows.visibility.make_shadow_pass`: through translucent casters with translucent shadows).
  `soft`: the build's (switches, bound bits): a tier build reading the soft shadows at run time
  (`toggles.SOFT_SHADOWS`) takes the run-time sampling where they are on
  (`_make_switched_area_visibility`).
  """
  compiled = _make_area_visibility(shadow_pass, shadow_samples=shadow_samples)
  switches, bound = soft
  if not (switches & toggles.SOFT_SHADOWS or bound & toggles.high(toggles.SOFT_SHADOWS)):
    return compiled
  runtime = _make_runtime_area_visibility(shadow_pass)
  return _make_switched_area_visibility(compiled, runtime, toggles.setting(switches, bound))


@specialize
def _make_area_visibility(shadow_pass: ShadowPass, *, shadow_samples: int):
  """`make_area_visibility`'s compiled sampling: `shadow_samples` rays to the fixed pattern."""

  @wp.func(inline=True)
  def sample(
    # Model:
    light: area_lights.AreaLight,
    # In:
    is_rect: bool,
    rect: area_lights.SphericalRect,
    origin: wp.vec3,
    u: wp.vec2,
  ) -> wp.vec3:
    """Point `u` of the light (a rect through its shared setup `rect`)."""
    if is_rect:
      return area_lights.sample_rect(rect, u)
    return area_lights.sample_curved(light, origin, u)

  @wp.func
  @device_generic
  def area_visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    light: area_lights.AreaLight,
    origin: wp.vec3,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> float:
    """Visible fraction of area light `li` from a shading point (rays from its `origin`)."""
    visible = float(0.0)
    shift = wp.vec2(0.0)
    # A rect's solid-angle setup is shared by its samples (one shading point).
    is_rect = light.shape == wp.static(area_lights.SHAPE_RECT)
    rect = area_lights.SphericalRect()
    if wp.static(shadow_samples > 1):
      if is_rect:
        rect = area_lights.rect_of(light, origin)
      shift = area_lights.shadow_rand2(seed, stream, nlight, li, wp.static(shadow_samples), 0)
    # One sample: a constant trip count (the loop folds away); more: a run-time one, so that the
    # loop keeps one sampling and shadow-ray site (a constant count is unrolled, a site per sample).
    if wp.static(shadow_samples == 1):
      count = wp.static(shadow_samples)
    else:
      count = int(wp.static(shadow_samples))  # pyrefly: ignore[unnecessary-type-conversion]  # a run-time count: Warp unrolls a constant one
    for k in range(count):
      if wp.static(shadow_samples == 1):
        to_target = light.center - origin
      else:
        to_target = sample(light, is_rect, rect, origin, area_lights.fixed_rand2(k, shift)) - origin
      d = wp.length(to_target)
      # A target at the shading point is reached (no room for an occluder).
      if d <= 0.0:
        visible += 1.0
      else:
        visible += shadow_pass(geo, worldid, origin, to_target / d, d * wp.static(area_lights.SHADOW_END))
    return visible / float(wp.static(shadow_samples))

  return area_visibility


@specialize
def _make_switched_area_visibility(compiled: AreaVisibility, runtime: AreaVisibility, setting: Callable[[int, int, int], bool]):
  """area_visibility of a tier build reading the soft shadows at run time: the run-time sampling
  (`runtime`) where its `toggles.SOFT_SHADOWS` setting is on, its own compiled
  sampling (`compiled`) elsewhere."""

  @wp.func
  @device_generic
  def area_visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    light: area_lights.AreaLight,
    origin: wp.vec3,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> float:
    """Visible fraction of area light `li` (see `_make_switched_area_visibility`)."""
    if setting(geo.rt.flags, geo.bits2, wp.static(toggles.SOFT_SHADOWS)):
      return runtime(geo, worldid, light, origin, seed, stream, nlight, li)
    return compiled(geo, worldid, light, origin, seed, stream, nlight, li)

  return area_visibility


def _make_runtime_area_visibility(shadow_pass: ShadowPass):
  """`make_area_visibility`'s run-time sampling: `Toggles.shadow_samples` rays to the fixed pattern
  (one sample: the centre)."""

  @wp.func
  @device_generic
  def area_visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    light: area_lights.AreaLight,
    origin: wp.vec3,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> float:
    """Visible fraction of area light `li` from a shading point (rays from its `origin`)."""
    count = geo.rt.shadow_samples
    centre = count == 1
    visible = float(0.0)
    shift = wp.vec2(0.0)
    is_rect = light.shape == wp.static(area_lights.SHAPE_RECT)
    rect = area_lights.SphericalRect()
    if not centre and is_rect:
      rect = area_lights.rect_of(light, origin)
    if not centre:
      shift = area_lights.shadow_rand2(seed, stream, nlight, li, count, 0)
    for k in range(count):
      to_target = light.center - origin
      if not centre:
        u = area_lights.fixed_rand2(k, shift)
        if is_rect:
          to_target = area_lights.sample_rect(rect, u) - origin
        else:
          to_target = area_lights.sample_curved(light, origin, u) - origin
      d = wp.length(to_target)
      if d <= 0.0:
        visible += 1.0
      else:
        visible += shadow_pass(geo, worldid, origin, to_target / d, d * wp.static(area_lights.SHADOW_END))
    return visible / float(count)

  return area_visibility
