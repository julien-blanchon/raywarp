"""The main render kernel: every thread computes one pixel of one world's camera image (rays
traced, surfaces shaded, colors written): one thread per (world, camera ray).

For each of the N×N sub-pixel samples the thread traces the primary ray, shades the hit
(`kernels/surface.py`: direct lights with ray-traced shadows from `kernels/lighting.py`, split-sum
ambient, emission), optionally traces a glossy reflection ray in place of the environment's
specular term (core/kernels/reflections.py; a MuJoCo `reflectance` is one more specular F0)
and continues through translucent layers; the samples' mean HDR radiance is then tone mapped.
Discrete settings are compile-time constants (`Features`); continuous ones come from `Params`.

  make_render_pixel   the per-pixel body of a camera model (`primary.CameraModel`): also the OptiX
                      ray-generation programs' body and the temporal body
                      (`pixel.make_render_pixel`)
  megakernel          the software-traversal kernel (cached per `Features` and camera model)
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Final, cast, final

import warp as wp

from raywarp._base.types import MEDIUM_IOR, ROW_MEDIUM, ShapeType
from raywarp.core.kernels import ambient, center, first_hit, geometry, reflections
from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi import rays as trace
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.output_layout import OUT_ALPHA, OUT_FIRST
from raywarp.core.kernels.abi.queries import Queries, make_queries, with_visibility
from raywarp.core.kernels.abi.rays import (
  BodyClosest,
  BodyLeave,
  GeometryFields,
  SoftwareGeometry,
  Tracer,
  offset_floor,
  ray_offset,
  software_geometry_type,
)
from raywarp.core.kernels.abi.structs import Cameras, EnvironmentArrays, OutputArrays, Params
from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.kernels.abi.toggles import out_on
from raywarp.core.kernels.display import write_pixel
from raywarp.core.kernels.generic import compiled_out, concrete, device_generic, specialize
from raywarp.core.kernels.layer import (
  MIN_THROUGHPUT,
  Coverage,
  ShadeLayer,
  absorb_exit,
  composite,
  composite_transmissive,
  leaves_solid,
  make_shade_layer,
  transmit_step,
)
from raywarp.core.kernels.lighting import DirectLighting, make_lighting
from raywarp.core.kernels.surface import make_shade_surface, shade_unlit
from raywarp.core.post.grading.transform import make_display
from raywarp.core.post.temporal import pixel, state
from raywarp.core.post.temporal import store as temporal_store
from raywarp.core.shade.environment.background import Background, make_background
from raywarp.core.shade.environment.fog import make_apply_fog
from raywarp.core.shade.lights import visible as visible_lights
from raywarp.core.shade.lights.loop import LightVisibilityAt
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.evaluate import make_eval_surface
from raywarp.core.shade.materials.material import MaterialTables, material_type
from raywarp.core.trace.accel import query
from raywarp.core.trace.rays import camera, primary, seeds

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# trace_pixel(geo, mats, lit, env, p, worldid, origin, direction, max_dist, footprint, spread,
# headlight_dir, exposure, seed, stream) -> (radiance, first hit, its base color, its
# `first_hit.SurfaceRecord`) (`make_trace_pixel`).
type TracePixel = temporal_store.TracePixel
# render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid): the non-temporal body
# (`make_pixel_body`).
type RenderPixel = Callable[
  [GeometryFields, MaterialTables, Lights, Cameras, EnvironmentArrays, int, OutputArrays, int, int], None
]

# The factory returns exactly the signature selected by shaded/history specialization.
type PixelBody = geometry.Body | geometry.TemporalBody | RenderPixel | pixel.RenderPixel


# trace_pixel returns the first surface's `first_hit.SurfaceRecord` (its surface outputs, the
# shading normal: `shade_layer`'s record). Translucent layers are composited
# front to back, then seen through the fog (`Features.fog`, `shade/environment/fog.py`). Random
# streams: `stream` shades the camera ray's surfaces, `stream + 1` the reflection ray.
@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _TraceFlags:
  """What `make_trace_pixel` specialises its body on (`_make_trace_body`)."""

  fog: bool
  # The build's compile-time settings read at run time (`toggles.setting_on`: the fog, the
  # translucent layers).
  setting: Callable[[int, int, int], bool]
  transparency: bool
  # Transmissive layers keep their specular (`layer.composite_transmissive`).
  transmissive: bool
  # Solid transmissive layers bend and absorb the ray (Features.refraction).
  refraction: bool
  # Solid glass primitives: their analytic exit, the ambient past the budget
  # (Features.glass_shapes).
  glass_shapes: bool
  bound: int  # Features.bound: the bound bits (`toggles.bound_on`)
  # Camera rays see the area lights without geometry (`shade/lights/visible.py`).
  visible_lights: bool
  counting: bool
  # The layers' optics rows are the world's (Features.world_optics: `trace.world_hit`).
  world_optics: bool


@specialize
def make_trace_pixel(
  f: Features,
  *,
  closest: BodyClosest,
  shade_layer: ShadeLayer,
  background: Background,
  layer: LayerShade | None = None,
  leave: BodyLeave | None = None,
):
  """trace_pixel(...) -> (radiance, first hit, its base color, its record) of a camera ray (see
  the comment above `_TraceFlags`). `layer`: the shading of the layers behind the first surface,
  a call of a program of its own (`trace/optix/device/programs.layer_program`); None: `shade_layer`
  inline. `leave`: a layer's ray that continues into the instance it hit (into a solid, or
  through a see-through one) leaves it through the face `leave` gives (`Tracer.leave`; None:
  `closest`'s), the exit face where the medium absorbs and the ray bends again; past the layer
  budget inside solid glass, the ambient light ahead (`_make_through_ambient`): both with solid
  glass primitives only (Features.glass_shapes, its word bit; compiled into every glass build and
  bound on its shapes, the exit check cost a feature-dense room with glass but no glass sphere
  14 % in fast on RTX)."""
  flags = _trace_flags(f)
  return _make_trace_body(
    flags,
    closest=closest,
    shade_layer=shade_layer,
    background=background,
    layer=layer,
    leave=leave if flags.glass_shapes else None,
    through=_make_through_ambient(f) if flags.glass_shapes else None,
  )


def _trace_flags(f: Features) -> _TraceFlags:
  """The `_TraceFlags` of Features `f`."""
  return _TraceFlags(
    fog=f.fog,
    setting=toggles.setting_on(f),
    transparency=f.transparency,
    transmissive=f.transparency and f.transmissive,
    refraction=f.transparency and f.transmissive and f.refraction,
    glass_shapes=f.transparency and f.transmissive and f.refraction and f.glass_shapes,
    bound=f.bound,
    visible_lights=f.visible_lights,
    counting=f.counters,
    world_optics=f.world_optics,
  )


def _make_trace_body(  # noqa: PLR0915 (the blends' two forms: Features.world_optics)
  flags: _TraceFlags,
  *,
  closest: BodyClosest,
  shade_layer: ShadeLayer,
  background: Background,
  layer: LayerShade | None,
  leave: BodyLeave | None,
  through: ThroughAmbient | None,
):
  """The trace_pixel func of `flags` (`shade_layer`: the hit shaded, with its record; `layer`,
  `leave`: the loop's, see `make_trace_pixel`; `through`: past the layer budget inside solid
  glass, `_make_through_ambient`)."""
  # The solid glass primitives' steps (`_make_glass_steps`): out of the layer loop's body, whose
  # only compile-time choice stays the blend's hit form (no unrolled copies).
  track, leave_step, past_budget = _make_glass_steps(flags, leave=leave, through=through)
  apply_fog = make_apply_fog(flags.setting)
  shade = _make_layer_shade(shade_layer)
  # The layers behind the first surface: their own program's call, or the first surface's shading.
  shade_behind = shade if layer is None else layer
  miss, emitted, count_layer = _make_layer_steps(flags, background=background)
  blend = _make_blend(flags)
  see_light = _make_see_light(flags)
  layer_count = _make_layer_count(flags)

  @wp.func
  @device_generic
  def trace_pixel[G: GeometryFields, M: MaterialTables](  # noqa: PLR0915 (compile-time branches)
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    exposure: float,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, trace.Hit, wp.vec3, first_hit.SurfaceRecord]:
    """Radiance along a camera ray, its first hit, that surface's base color and record (or
    zero); `exposure`: the pixel's (world x camera lens), which the display-relative fog color is
    relative to."""
    radiance = wp.vec3(0.0)
    throughput = float(1.0)
    tint = wp.vec3(1.0)  # the transmissive layers' color (Features.transmissive; unused without)
    travelled = float(0.0)
    first_albedo = wp.vec3(0.0)
    first_record = first_hit.zero_record()
    inside = int(-1)  # the instance the ray continues into (`_make_glass_steps`; unread without)
    if wp.static(flags.counting):
      counters.add(geo, worldid, wp.static(counters.SAMPLES), 1)
    # The first surface, then a loop over the translucent layers behind it. The first surface's
    # shading is straight-line code before the loop (RTX: the same shading as the loop's first
    # iteration ran the own builds 2-4.7x slower with the same OptiX continuation stack and half
    # the PTX: a clear-coated interior 31 -> 62 ms, a feature-dense room 56 -> 167 ms; neither
    # cssRG nor PTX shows it). The loop is a loop, not copies of the whole surface shading: its
    # bound is read at run time (`Params.max_layers`), else Warp (a constant range whose body has a
    # wp.static) or the CUDA compiler unrolls it.
    first = closest(geo, worldid, origin, direction, max_dist)
    # A light's emitting surface in front of the surface hit (`shade/lights/visible.py`).
    seen, light = see_light(lit, worldid, origin, direction, wp.where(first.instance_id == -1, max_dist, first.dist))
    if seen:
      radiance = emitted(radiance, light, throughput, tint)
      throughput = 0.0
    elif first.instance_id == -1:
      radiance = miss(env, p, worldid, direction, spread, radiance, throughput, tint)
      throughput = 0.0
    else:
      count_layer(geo, worldid)
      point = origin + direction * first.dist
      cone = footprint + spread * first.dist
      shaded, cover, first_record = shade(
        geo, mats, lit, env, p, worldid, first, point, direction, cone, spread, headlight_dir, seed, stream
      )
      first_albedo = cover.base_color
      if wp.static(flags.world_optics):
        radiance, throughput, tint, direction, spread, _counted = blend(
          mats.instance_optics,
          mats.bits,
          p.rt.flags,
          geo.bits2,
          radiance,
          throughput,
          tint,
          shaded,
          cover,
          direction,
          trace.world_hit(first, geo.instance_type.shape[0], worldid),
          spread,
        )
      else:
        radiance, throughput, tint, direction, spread, _counted = blend(
          mats.instance_optics,
          mats.bits,
          p.rt.flags,
          geo.bits2,
          radiance,
          throughput,
          tint,
          shaded,
          cover,
          direction,
          first,
          spread,
        )
      travelled = first.dist
      origin = point + direction * ray_offset(point, offset_floor(geo, first.instance_id))
      inside = track(geo.bits2, inside, first, direction)
    if throughput >= wp.static(MIN_THROUGHPUT):
      layers = int(1)  # the surfaces counted against the budget (`blend`), the first one's
      for _layer in range(1, layer_count(p, mats.bits, geo.bits2)):
        # Past the budget (`p.max_layers` surfaces) what is behind the last surface is not traced:
        # an empty query, then the miss below adds the background through that surface's
        # transmittance (stacked panes stay see-through, not opaque). The same trace call site,
        # no state kept for it.
        remaining = wp.where(layers == p.max_layers, 0.0, max_dist - travelled)
        hit = closest(geo, worldid, origin, direction, remaining)
        hit = leave_step(geo, worldid, inside, origin, direction, hit)
        behind = wp.where(hit.instance_id == -1, remaining, hit.dist)
        seen, light = see_light(lit, worldid, origin, direction, behind)
        if seen:
          radiance = emitted(radiance, light, throughput, tint)
          break
        if hit.instance_id == -1:
          past, radiance = past_budget(
            geo, mats, env, worldid, inside, remaining, origin, direction, radiance, throughput, tint
          )
          if past:
            break
          radiance = miss(env, p, worldid, direction, spread, radiance, throughput, tint)
          break
        count_layer(geo, worldid)
        point = origin + direction * hit.dist
        cone = footprint + spread * (travelled + hit.dist)
        shaded, cover, _rec = shade_behind(
          geo, mats, lit, env, p, worldid, hit, point, direction, cone, spread, headlight_dir, seed, stream
        )
        if wp.static(flags.world_optics):
          radiance, throughput, tint, direction, spread, counted = blend(
            mats.instance_optics,
            mats.bits,
            p.rt.flags,
            geo.bits2,
            radiance,
            throughput,
            tint,
            shaded,
            cover,
            direction,
            trace.world_hit(hit, geo.instance_type.shape[0], worldid),
            spread,
          )
        else:
          radiance, throughput, tint, direction, spread, counted = blend(
            mats.instance_optics,
            mats.bits,
            p.rt.flags,
            geo.bits2,
            radiance,
            throughput,
            tint,
            shaded,
            cover,
            direction,
            hit,
            spread,
          )
        layers += counted
        if throughput < MIN_THROUGHPUT:
          break
        travelled += hit.dist
        origin = point + direction * ray_offset(point, offset_floor(geo, hit.instance_id))
        inside = track(geo.bits2, inside, hit, direction)
    if wp.static(flags.fog):
      radiance = apply_fog(env, worldid, radiance, first.instance_id, first.dist, exposure)
    return radiance, first, first_albedo, first_record

  return trace_pixel


# through(env, worldid, point, direction) -> the ambient radiance along `direction` from `point`
# (`_make_through_ambient`).
type ThroughAmbient = Callable[[EnvironmentArrays, int, wp.vec3, wp.vec3], wp.vec3]


@specialize
def _make_through_ambient(f: Features):
  """through(env, worldid, point, direction) -> what a camera ray past the layer budget inside
  solid glass sees ahead: the ambient diffuse irradiance about `direction` at `point` over pi
  (`ambient.make_diffuse_ambient`: the GI probes, else the environment's SH9 and the fill, plus
  the user's ambient), the mean radiance of the half of the scene the ray looks into."""
  diffuse_ambient = ambient.make_diffuse_ambient(f)

  @wp.func
  def through(
    # Model:
    env: EnvironmentArrays,
    # In:
    worldid: int,
    point: wp.vec3,
    direction: wp.vec3,
  ) -> wp.vec3:
    """The ambient radiance along `direction` (`_make_through_ambient`)."""
    _fill, seen, _probes, _sky = diffuse_ambient(env, worldid, point, direction, direction, 1.0)
    return seen

  return through


@specialize
def _make_glass_steps(flags: _TraceFlags, *, leave: BodyLeave | None, through: ThroughAmbient | None):
  """(track, leave_step, past_budget) of trace_pixel's layer loop for solid glass primitives
  (`leave`, `through`: `make_trace_pixel`; None: steps that change nothing, compiled away):
  - track(bits2, inside, hit, direction) -> the instance the ray continues into after `hit` (a
    front face: its instance; -1 otherwise), where the build word has `toggles.GLASS_SHAPES` (a
    module without glass primitives folds it: OptiX binds the word, a software module holds it);
  - leave_step(geo, worldid, inside, origin, direction, hit) -> `hit`, or the exit face of the
    instance `inside` where nearer (`Tracer.leave`);
  - past_budget(geo, mats, env, worldid, inside, remaining, origin, direction, radiance,
    throughput, tint) -> (whether the ray ends past the layer budget inside a solid glass
    primitive, `radiance` plus the ambient light ahead there, `_make_through_ambient`)."""
  leaving, tracking = leave is not None, through is not None
  leave_hit = leave if leave is not None else compiled_out
  through_ambient = through if through is not None else compiled_out
  in_solid = _make_in_solid(flags.world_optics)
  glassy = toggles.bound_on(flags.bound, high=True)

  @wp.func(inline=True)
  def track(
    # In:
    bits2: int,
    inside: int,
    hit: trace.Hit,
    direction: wp.vec3,
  ) -> int:
    """See `_make_glass_steps`."""
    if wp.static(tracking):
      if glassy(bits2, wp.static(toggles.GLASS_SHAPES)):
        return wp.where(wp.dot(hit.normal, direction) < 0.0, hit.instance_id, -1)
    return inside

  @wp.func(inline=True)
  @device_generic
  def leave_step[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    inside: int,
    origin: wp.vec3,
    direction: wp.vec3,
    hit: trace.Hit,
  ) -> trace.Hit:
    """See `_make_glass_steps`."""
    if wp.static(leaving):
      if inside >= 0:
        return leave_hit(geo, worldid, inside, origin, direction, hit)
    return hit

  @wp.func(inline=True)
  @device_generic
  def past_budget[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    env: EnvironmentArrays,
    # In:
    worldid: int,
    inside: int,
    remaining: float,
    origin: wp.vec3,
    direction: wp.vec3,
    radiance: wp.vec3,
    throughput: float,
    tint: wp.vec3,
  ) -> tuple[bool, wp.vec3]:
    """See `_make_glass_steps`: past the budget inside solid glass (the ray entered its last
    counted surface), the ambient light along the ray, not the background (a grid of glass balls
    stayed black behind the second one at `max_layers` 2)."""
    if wp.static(tracking):
      if remaining == 0.0 and inside >= 0:
        if in_solid(geo, mats.instance_optics, worldid, inside):
          ahead = wp.cw_mul(through_ambient(env, worldid, origin, direction), tint)
          return True, radiance + ahead * throughput
    return False, radiance

  return track, leave_step, past_budget


@specialize
def _make_in_solid(world_optics: bool):
  """in_solid(geo, optics, worldid, instance) -> whether `instance` is a solid glass primitive
  (an analytic shape, `Features.glass_shapes`: glass meshes keep the background on both backends,
  as an OptiX module without glass primitives folds the check; a refraction index > 0,
  `pipeline/upload/optics.py`; its world's row with Features.world_optics, `world_optics`)."""

  @wp.func(inline=True)
  @device_generic
  def in_solid[G: GeometryFields](
    # Model:
    geo: G,
    optics: wp.array2d[wp.vec4],
    # In:
    worldid: int,
    instance: int,
  ) -> bool:
    """See `_make_in_solid`."""
    gtype = geo.instance_type[instance]
    # The analytic shapes are the MuJoCo types sphere .. box.
    if gtype < int(ShapeType.SPHERE) or gtype > int(ShapeType.BOX):
      return False
    row = instance
    if wp.static(world_optics):
      row = trace.world_entry(geo.instance_type.shape[0], worldid, instance)
    return optics[row, wp.static(ROW_MEDIUM)][wp.static(MEDIUM_IOR)] > 0.0

  return in_solid


def _make_layer_count(flags: _TraceFlags):
  """layer_count(p, bits, bits2) -> the bound of trace_pixel's layer loop (its docstring)."""
  gate = toggles.bound_on(flags.bound)

  @wp.func(inline=True)
  def layer_count(
    # Model:
    p: Params,
    # In:
    bits: int,
    bits2: int,
  ) -> int:
    """The layer loop's bound: with translucent layers (compiled in and on: a tier build's setting,
    `toggles.TRANSPARENCY` in `p.rt.flags` / the word's second half `bits2`), `p.max_layers`
    surfaces and one untraced step past them (the background behind the last surface), else one
    surface. With solid glass (compiled in and on in the build word `bits`: the glass class bit,
    `toggles.CLASS_GLASS`), a solid's exit face is no layer (`blend`: it adds no light of its
    own): up to one exit per layer (a total internal reflection is one more exit, so the bound
    also ends a ray trapped in a solid)."""
    if wp.static(flags.transparency):
      if flags.setting(p.rt.flags, bits2, wp.static(toggles.TRANSPARENCY)):
        if wp.static(flags.refraction):
          if gate(bits, wp.static(toggles.CLASS_GLASS)):
            return 2 * p.max_layers + 1
        return p.max_layers + 1
    return 1

  return layer_count


# shade(geo, mats, lit, env, p, worldid, hit, point, direction, cone, spread, headlight_dir, seed,
# stream) -> (radiance, coverage, record): one layer's shading in trace_pixel's layer loop.
type LayerShade = Callable[..., tuple[wp.vec3, Coverage, first_hit.SurfaceRecord]]


def _make_layer_shade(shade_layer: ShadeLayer):
  """shade(...) -> (radiance, coverage, record) of trace_pixel's layer loop: a layer's shading and
  its `first_hit.SurfaceRecord` (the first surface's is kept)."""

  @wp.func(inline=True)
  @device_generic
  def shade[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    hit: trace.Hit,
    point: wp.vec3,
    direction: wp.vec3,
    cone: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, Coverage, first_hit.SurfaceRecord]:
    """(radiance, coverage, record) of the layer at `hit`."""
    shaded, cover, _normal, rec = shade_layer(
      geo, mats, lit, env, p, worldid, hit, point, direction, cone, spread, headlight_dir, seed, stream
    )
    return shaded, cover, rec

  return shade


def _make_see_light(flags: _TraceFlags):
  """see_light(lit, worldid, origin, direction, max_dist) -> (seen, radiance): the area light a
  camera ray sees before `max_dist` (`visible_lights.closest`; a scene without such a light, or
  with `lighting.visible_lights` off, pays one branch on `Lights.nvisible`);
  `flags.visible_lights` off: nothing compiled in; bound, where the build word has it
  (`toggles.SCENE_VISIBLE_LIGHTS`)."""
  gate = toggles.bound_on(flags.bound)

  @wp.func(inline=True)
  def see_light(
    # Model:
    lit: Lights,
    # In:
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> tuple[bool, wp.vec3]:
    """(Whether the ray sees an area light's surface before `max_dist`, its radiance)."""
    if wp.static(flags.visible_lights):
      if gate(lit.bits, wp.static(toggles.SCENE_VISIBLE_LIGHTS)) and lit.nvisible > 0:
        seen = visible_lights.closest(lit.lights, lit.visible, lit.nvisible, worldid, origin, direction, max_dist)
        return seen[3] > 0.0, wp.vec3(seen[0], seen[1], seen[2])
    return False, wp.vec3(0.0)

  return see_light


def _make_transmit(flags: _TraceFlags):
  """transmit(optics, bits, radiance, throughput, tint, shaded, cover, direction, hit, spread): the
  layer loop's blend of a transmissive build (`_make_layer_steps`' blend, its arguments but the
  second word half): leaving a solid, the light crossed its medium (`layer.absorb_exit`), and a
  transmissive layer bends and blurs what is behind it (`layer.transmit_step`; per-material
  run-time data). The solid glass reads the glass class bit of the build word (`flags.bound`)."""
  gate = toggles.bound_on(flags.bound)

  @wp.func(inline=True)
  def transmit(
    # Model:
    optics: wp.array2d[wp.vec4],
    # In:
    bits: int,
    radiance: wp.vec3,
    throughput: float,
    tint: wp.vec3,
    shaded: wp.vec3,
    cover: Coverage,
    direction: wp.vec3,
    hit: trace.Hit,
    spread: float,
  ) -> tuple[wp.vec3, float, wp.vec3, wp.vec3, float, int]:
    """See `_make_transmit`."""
    solid = False
    counted = int(1)
    if wp.static(flags.refraction):
      solid = gate(bits, wp.static(toggles.CLASS_GLASS))
    if solid:
      if cover.specular_only:
        counted = wp.where(leaves_solid(optics, hit.instance_id, hit.normal, direction), 0, 1)
        tint, shaded = absorb_exit(optics, hit.instance_id, hit.normal, direction, hit.dist, tint, shaded)
    out, through, tinted = composite_transmissive(radiance, throughput, tint, shaded, cover)
    if solid:
      if cover.specular_only:
        direction, spread = transmit_step(optics, hit.instance_id, hit.normal, cover.roughness, direction, spread)
    return out, through, tinted, direction, spread, counted

  return transmit


def _make_layer_steps(flags: _TraceFlags, *, background: Background):
  """(miss, emitted, count_layer) of trace_pixel's layer loop: the background and a light's
  surface seen through the layers in front (`_make_blend` composites a layer over them) and the
  counter of shaded layers (`Features.counters`)."""

  @wp.func(inline=True)
  def miss(
    # Model:
    env: EnvironmentArrays,
    p: Params,
    # In:
    worldid: int,
    direction: wp.vec3,
    spread: float,
    radiance: wp.vec3,
    throughput: float,
    tint: wp.vec3,
  ) -> wp.vec3:
    """`radiance` plus the background behind the layers (tinted by the transmissive ones)."""
    if wp.static(flags.transmissive):
      return radiance + wp.cw_mul(background(env, p, worldid, direction, spread, True), tint) * throughput
    return radiance + background(env, p, worldid, direction, spread, True) * throughput

  @wp.func(inline=True)
  def emitted(
    # In:
    radiance: wp.vec3,
    light: wp.vec3,
    throughput: float,
    tint: wp.vec3,
  ) -> wp.vec3:
    """`radiance` plus a light's surface radiance `light` behind the layers (tinted by the
    transmissive ones)."""
    if wp.static(flags.transmissive):
      return radiance + wp.cw_mul(light, tint) * throughput
    return radiance + light * throughput

  @wp.func(inline=True)
  @device_generic
  def count_layer[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
  ):
    """Counts a shaded layer (`Features.counters`; nothing otherwise)."""
    if wp.static(flags.counting):
      counters.add(geo, worldid, wp.static(counters.LAYERS), 1)

  return miss, emitted, count_layer


def _make_blend(flags: _TraceFlags):
  """blend(...) of trace_pixel's layer loop: a layer composited over the ones in front (the opaque
  build, or the translucent layers' setting off: added, ending the loop with throughput 0); a
  transmissive layer's blend is `_make_transmit`'s."""
  field = toggles.bound_on(flags.bound, high=True)
  transmit = _make_transmit(flags)

  @wp.func(inline=True)
  def blend(
    # Model:
    optics: wp.array2d[wp.vec4],
    # In:
    bits: int,
    switches: int,
    bits2: int,
    radiance: wp.vec3,
    throughput: float,
    tint: wp.vec3,
    shaded: wp.vec3,
    cover: Coverage,
    direction: wp.vec3,
    hit: trace.Hit,
    spread: float,
  ) -> tuple[wp.vec3, float, wp.vec3, wp.vec3, float, int]:
    """(radiance, throughput, tint, direction, cone spread, counted) after the layer `shaded` at
    `hit` of coverage `cover` (`layer.Coverage`: a closed pane's entry face covers for its culled
    exit face too); `counted`: 1 when the surface counts against `Params.max_layers`, 0 for a
    solid's exit face (as a closed pane's: the far side of the layer it entered).
    Features.transmissive: leaving a solid, the light crossed its medium (`layer.absorb_exit`),
    and a transmissive layer bends and blurs what is behind it (`layer.transmit_step`;
    per-material run-time data); bound, where the build word has the transmissive field bit
    (`bits2`, `toggles.SURF_TRANSMISSIVE`), else the plain composite. The translucent layers'
    setting: `switches` (`Toggles.flags`), `bits2` (the word's second half)."""
    if wp.static(flags.transparency):
      if flags.setting(switches, bits2, wp.static(toggles.TRANSPARENCY)):
        if wp.static(flags.transmissive):
          if field(bits2, wp.static(toggles.SURF_TRANSMISSIVE)):
            return transmit(optics, bits, radiance, throughput, tint, shaded, cover, direction, hit, spread)
        out, through = composite(radiance, throughput, shaded, cover.opacity)
        return out, through, tint, direction, spread, 1
    return radiance + shaded, 0.0, tint, direction, spread, 1

  return blend


# The HDR value a supersampled sample stores when it shares its pixel's first sample's surface
# (radiance is never negative): the downsample (`trace/optix/supersample.py`) counts the first
# sample's radiance in its place.
SHARED: Final = -1.0
# The smallest cosine between two samples' geometric normals on one shared surface: 60 degrees
# keeps the faces of an analytic box or of a cylinder (90 degree creases) apart, as multisampling
# shades each primitive apart, while a curved or finely tessellated surface turns by far less
# within one output pixel.
SHARED_COS: Final = 0.5


@specialize
def make_multisampling(
  f: Features,
  *,
  camera_rays: tuple[primary.PrimaryRay, primary.PixelCone],
  closest: BodyClosest,
  supersampled: bool,
):
  """(setup, footprint_scale) of the multisampled shading of the OptiX supersampled program
  (`supersampled`; otherwise `setup` is `camera.setup_pixel` and the scale 1):
  - setup(geo, lit, cams, out, worldid, rayid, store) -> the thread's pixel (`camera.setup_pixel`;
    supersampled: `camera.setup_sample`, sample-major warps), `camid` -1 when the sample sees its
    output pixel's first sample's surface, which then stores `SHARED` (not shaded): both primary
    rays hit the same instance and mesh, with geometric normals within `SHARED_COS`, the sample's
    hit point on the first one's tangent plane within the output pixel's footprint (no fold or
    step of one mesh), an opaque instance (no translucent shadow transmittance: what is behind a
    translucent layer differs per sample) and no visible light in front of either hit;
  - footprint_scale(pix) -> the factor of the sample's ray cone: the first sample shades for the
    samples sharing its surface, over the output pixel's footprint (texture filtering, reflection
    cones), as multisampling derives its from the pixel.
  Builds without it define none of its code (an OptiX module compiles every function it holds)."""
  if not supersampled:
    return _plain_setup(f.any_rgb), _unit_scale
  return _shared_setup(f, camera_rays=camera_rays, closest=closest), _first_sample_scale


@specialize
def _plain_setup(any_rgb: bool):
  """`make_multisampling`'s setup without multisampling: `camera.setup_pixel`."""

  @wp.func
  @device_generic
  def setup[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,  # noqa: ARG001  # the setup signature (`make_multisampling`)
    lit: Lights,  # noqa: ARG001
    cams: C,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    rayid: int,
    store: bool,  # noqa: ARG001
  ) -> camera.Pixel:
    """See `make_multisampling`."""
    return camera.setup_pixel(cams, out, wp.static(any_rgb), worldid, rayid)

  return setup


@wp.func
def _unit_scale(
  # In:
  pix: camera.Pixel,  # noqa: ARG001  # the footprint_scale signature (`make_multisampling`)
) -> float:
  """`make_multisampling`'s footprint scale without multisampling."""
  return 1.0


@wp.func
def _first_sample_scale(
  # In:
  pix: camera.Pixel,
) -> float:
  """`make_multisampling`'s footprint scale: the lens's samples per side for a pixel's first
  sample, else 1."""
  if pix.px % pix.lens.scale == 0 and pix.py % pix.lens.scale == 0:
    return float(pix.lens.scale)
  return 1.0


@specialize
def _shared_setup(f: Features, *, camera_rays: tuple[primary.PrimaryRay, primary.PixelCone], closest: BodyClosest):
  """`make_multisampling`'s setup of a supersampled pinhole build."""
  primary_ray, pixel_cone = camera_rays
  see_light = _make_see_light(_trace_flags(f))
  translucent = f.transparency
  world_optics = f.world_optics
  any_rgb = f.any_rgb

  @wp.func
  @device_generic
  def same_surface[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    cams: C,
    # In:
    pix: camera.Pixel,
    worldid: int,
  ) -> bool:
    """Whether sample `pix` sees its output pixel's first sample's surface (above)."""
    first = pix
    first.px = pix.px - pix.px % pix.lens.scale
    first.py = pix.py - pix.py % pix.lens.scale
    first.pixelid = first.py * first.width + first.px
    origin, direction, max_dist, _axis = primary_ray(cams, pix, 1, 0, 0)
    origin0, direction0, max_dist0, _axis0 = primary_ray(cams, first, 1, 0, 0)
    hit = closest(geo, worldid, origin, direction, max_dist)
    hit0 = closest(geo, worldid, origin0, direction0, max_dist0)
    # A miss or a deformable (`trace.FLEX_HIT`, negative ids) shades on its own.
    if hit.instance_id < 0 or hit.instance_id != hit0.instance_id or hit.mesh_id != hit0.mesh_id:
      return False
    if wp.dot(hit.normal, hit0.normal) < wp.static(SHARED_COS):
      return False
    # The output pixel's footprint at the first hit: the sample's cone over n samples.
    spread, width = pixel_cone(cams, first)
    footprint = (width + spread * hit0.dist) * float(pix.lens.scale)
    step = wp.dot(origin + direction * hit.dist - (origin0 + direction0 * hit0.dist), hit0.normal)
    if wp.abs(step) > footprint:
      return False
    if wp.static(translucent):
      if wp.static(world_optics):
        if wp.length_sq(trace.world_transmit(geo, worldid, hit.instance_id)) > 0.0:
          return False
      elif wp.length_sq(geo.instance_shadow_transmit[hit.instance_id]) > 0.0:
        return False
    seen, _light = see_light(lit, worldid, origin, direction, hit.dist)
    seen0, _light0 = see_light(lit, worldid, origin0, direction0, hit0.dist)
    return not (seen or seen0)

  @wp.func
  @device_generic
  def setup[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    cams: C,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    rayid: int,
    store: bool,
  ) -> camera.Pixel:
    """See `make_multisampling`."""
    pix = camera.setup_sample(cams, out, wp.static(any_rgb), worldid, rayid)
    if pix.camid != -1 and pix.want_rgb and (pix.px % pix.lens.scale != 0 or pix.py % pix.lens.scale != 0):
      if same_surface(geo, lit, cams, pix, worldid):
        write_pixel(out, worldid, pix, wp.vec3(0.0), wp.vec3(wp.static(SHARED)), 1.0, store)
        pix.camid = -1
    return pix

  return setup


@specialize
@specialize
def make_sample_axis(spp: int):
  """sample_axis(p) -> the samples per pixel axis of the per-frame body of a build of `Features.spp`
  `spp`: 1, or for a multi-sample build (2) `Params.spp`: any n > 1 at run time (`resolve._spp`)."""

  @wp.func
  def sample_axis(
    # In:
    p: Params,
  ) -> int:
    """Samples per pixel axis."""
    if wp.static(spp > 1):
      return p.spp
    return 1

  return sample_axis


def make_pixel_body(
  f: Features,
  *,
  trace_pixel: TracePixel,
  camera_rays: tuple[primary.PrimaryRay, primary.PixelCone],
  writes: first_hit.FirstHitWriters,
  geometry_only: primary.GeometryOnly,
  center_trace: center.CenterTrace,
  supersampled: bool,
  closest: BodyClosest,
):
  """render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid): the non-temporal body.
  Supersampled (the OptiX
  supersampled program: one thread per sample, n x n samples per output pixel), a sample whose
  surface is its pixel's first sample's (`make_multisampling`) is not shaded: it stores `SHARED`
  and the downsample counts the first sample's radiance in its place (multisampling: every sample
  traced, each surface of a pixel shaded once)."""
  primary_ray, pixel_cone = camera_rays
  center_ray, center_cone = primary.make_center_rays(camera_rays, supersampled=supersampled)
  sample_axis = make_sample_axis(f.spp)
  display = make_display(graded=False)
  write_observation = center.make_write(writes)
  # The first sample is off the pixel centre (the observation's ray): multisampling, the OptiX
  # supersampled program.
  off_center = f.spp > 1 or supersampled
  shading = center.shading_outputs(f)
  setup, footprint_scale = make_multisampling(f, camera_rays=camera_rays, closest=closest, supersampled=supersampled)

  @wp.func
  @device_generic
  def render_pixel[C: Cameras, G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    cams: C,
    env: EnvironmentArrays,
    # In:
    seed: int,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    rayid: int,
  ):
    """Renders ray `rayid` of world `worldid`: spp x spp samples averaged in HDR, then displayed."""
    p = env.params
    pix = setup(geo, lit, cams, out, worldid, rayid, p.skip_display == 0)
    if pix.camid == -1:
      return
    if not pix.want_rgb:
      geometry_only(geo, cams, out, pix, worldid, 1, 0)
      return

    n_aa = sample_axis(p)
    inv_aa = 1.0 / float(n_aa)
    # Ray-cone parameters from the neighbouring pixel: angular spread (perspective) and width
    # (orthographic), shared by the sub-pixel samples.
    pixel_spread, pixel_width = pixel_cone(cams, pix)
    spread = pixel_spread * inv_aa * footprint_scale(pix)
    width = pixel_width * inv_aa * footprint_scale(pix)
    headlight_dir = camera.headlight_dir(pix)

    pixel_seed = seeds.pixel_seed(seed, worldid, camera.buffer_index(pix))
    exposure = exposure_at(env.exposure, worldid) * pix.lens.exposure
    # Only the first high-resolution sample is copied by the OptiX downsample.
    selected = bool(True)
    if wp.static(supersampled):
      selected = pix.px % pix.lens.scale == 0 and pix.py % pix.lens.scale == 0
    # Off the centre, the outputs only the shading gives (direct / indirect) take the pixel's first
    # sample at the centre ray: one shading call site, the sample ray chosen first.
    centred = False
    if wp.static(off_center and shading != 0):
      centred = selected and out_on(out.bits, wp.static(shading))
    hdr = wp.vec3(0.0)
    covered = float(0.0)
    for smp in range(n_aa * n_aa):
      origin, direction, max_dist, depth_axis = primary_ray(cams, pix, n_aa, smp % n_aa, smp // n_aa)
      sample_width, sample_spread = width, spread
      if smp == 0 and centred:
        origin, direction, max_dist, depth_axis = center_ray(cams, pix, 1, 0, 0)
        sample_spread, sample_width = center_cone(cams, pix)
      # Random streams: 2 smp for the surfaces, 2 smp + 1 for the reflection rays.
      stream = 2 * smp
      radiance, hit, albedo, rec = trace_pixel(
        geo,
        mats,
        lit,
        env,
        p,
        worldid,
        origin,
        direction,
        max_dist,
        sample_width,
        sample_spread,
        headlight_dir,
        exposure,
        pixel_seed,
        stream,
      )
      # At spp 1 the shaded sample already is the output pixel center (or was traced there).
      if (wp.static(not off_center) or centred) and smp == 0:
        write_observation(geo, lit, out, worldid, pix, hit, depth_axis, origin, direction, albedo, rec, pixel_seed)
      hdr += radiance
      if out_on(out.bits, wp.static(OUT_ALPHA)):
        if hit.instance_id != -1:
          covered += 1.0

    if wp.static(off_center):
      # RGB-only cameras do no additional work: their geometry wants are false, the word has no
      # first-hit output.
      if selected and not centred and (pix.want_depth or pix.want_seg or out_on(out.bits, wp.static(OUT_FIRST))):
        origin, direction, max_dist, axis = center_ray(cams, pix, 1, 0, 0)
        center_spread, center_width = center_cone(cams, pix)
        hit, albedo, rec = center_trace(
          geo, mats, env, p, out, worldid, origin, direction, max_dist, center_width, center_spread
        )
        write_observation(geo, lit, out, worldid, pix, hit, axis, origin, direction, albedo, rec, pixel_seed)
    mean = hdr * (inv_aa * inv_aa)
    store = p.skip_display == 0
    shown = wp.vec3(0.0)
    if store:
      shown = display(env, worldid, mean * exposure)
    write_pixel(out, worldid, pix, shown, mean, covered * (inv_aa * inv_aa), store)

  return render_pixel


@specialize
def make_render_pixel(
  f: Features,
  tracer: Tracer,
  model: primary.CameraModel,
  *,
  supersampled: bool = False,
  layer: LayerShade | None = None,
  light_visibility: LightVisibilityAt | None = None,
):
  """Specialises the per-pixel body of the megakernel on `f` and the camera model `model` (its
  primary rays, `primary.make_camera_rays`).

  render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid) renders ray `rayid` (pixel of
  a camera) of world `worldid`; with `f.history` the temporal body
  (`pixel.make_render_pixel`: one more argument `tmp` before worldid). The ray queries
  are `tracer`'s (`TraceBackend.tracer`, see `queries.make_queries`), so the same body runs as a
  Warp kernel over the software traversal or as an OptiX ray-generation program (the temporal
  body is a `core/post/temporal` function). Without shaded outputs
  (`Features.any_rgb` False) the body is `geometry.make_geometry_body`'s. `layer`: the layer
  loop's shading call (`make_trace_pixel`); `light_visibility`: the light loops' visibility call
  (`Queries.light_visibility`).
  """
  if not f.any_rgb:
    return geometry.make_geometry_body(f, tracer, model)
  q = with_visibility(make_queries(f, tracer), light_visibility)
  camera_rays = primary.make_camera_rays(model)
  primary_ray, pixel_cone = camera_rays
  lighting = make_lighting(f, q)
  trace_pixel = _make_camera_trace(f, q, lighting, layer)
  writes = first_hit.make_writers(f, primary.write_geometry, lighting)
  center_trace = center.make_trace(f, closest=q.closest)
  if not f.history:
    geometry_only = primary.make_geometry_only(
      trace_closest=q.closest, primary_ray=primary.make_center_rays(camera_rays, supersampled=supersampled)[0], writes=writes
    )
    return make_pixel_body(
      f,
      trace_pixel=trace_pixel,
      camera_rays=camera_rays,
      writes=writes,
      geometry_only=geometry_only,
      center_trace=center_trace,
      supersampled=supersampled,
      closest=q.closest,
    )
  # The pixel centre's own shading for the outputs only it gives (an off-centre first sample).
  off_center = f.spp > 1
  return pixel.make_render_pixel(
    f,
    model,
    trace_closest=q.closest,
    trace_pixel=trace_pixel,
    primary_ray=primary_ray,
    pixel_cone=pixel_cone,
    writes=writes,
    center_trace=center_trace,
    shading=center.shading_outputs(f) if off_center else 0,
  )


def _make_camera_trace(f: Features, q: Queries, lighting: DirectLighting, layer: LayerShade | None):
  """The camera rays' trace_pixel of `f` (`make_trace_pixel`): its surfaces' shading, glossy
  reflections, background (`layer`: the layer loop's shading call)."""
  shade_layer = _camera_shade_layer(f, q, lighting)
  return make_trace_pixel(
    f, closest=q.closest, shade_layer=shade_layer, background=make_background(f), layer=layer, leave=q.leave
  )


def make_layer_body(f: Features, tracer: Tracer, light_visibility: LightVisibilityAt | None = None):
  """The shading of one layer behind the first surface, as trace_pixel's layer loop calls it
  (`LayerShade`): the camera rays' surface shading of `f` over `tracer`'s queries (and
  `light_visibility`, `make_render_pixel`), built as `make_render_pixel` builds it, for a program
  of its own (`trace/optix/device/programs.layer_program`)."""
  q = with_visibility(make_queries(f, tracer), light_visibility)
  return _make_layer_shade(_camera_shade_layer(f, q, make_lighting(f, q)))


def _camera_shade_layer(f: Features, q: Queries, lighting: DirectLighting) -> ShadeLayer:
  """The camera rays' layer shading of `f`: surface shading, glossy reflections, the direct /
  indirect split."""
  eval_surface = make_eval_surface(f)
  glossy = None
  if lighting.hit_lights is not None:
    glossy = reflections.make_glossy(
      f,
      trace_closest=q.closest,
      eval_surface=eval_surface,
      hit_lights=lighting.hit_lights,
    )
  # The camera rays' shading, with the direct / indirect outputs' split (every lit build).
  split = first_hit.split_lighting(f)
  if f.unlit:
    shade_surface = shade_unlit
  else:
    shade_surface = make_shade_surface(f, lighting, glossy, split=split)
  return make_shade_layer(f, eval_surface=eval_surface, shade_surface=shade_surface)


@functools.cache
def megakernel(f: Features, model: primary.CameraModel, *, banded: bool = False) -> wp.Kernel:
  """The software-traversal megakernel specialised on `f` and camera model `model` (cached per
  process: renderers and tests that share Features share kernels), in the module of its words: the
  words of a build share its device functions (`kernel_structs.in_module_of`).

  `banded`: the form a CPU launches in bands of rays over its cores (`device.launch_bands`): one
  more input `first` after the seed, the first ray of the launch's band (its launch is
  (nworld, band size)). A CUDA launch takes the plain form (one launch over every ray)."""
  return kernel_structs.in_module_of(_megakernel(kernel_structs.words_free(f), model, banded), f, outputs=True)


@functools.cache
def _megakernel(f: Features, model: primary.CameraModel, banded: bool):
  """`megakernel` of word-free Features `f`."""
  tracer = query.software_tracer(f, cull=f.backface_culling)
  if not f.any_rgb:
    return geometry.make_megakernel(f, tracer, model, banded=banded)
  # Narrow the chosen body at the same static branches used by make_render_pixel.
  body = make_render_pixel(f, tracer, model)
  if f.history:
    temporal_body = cast(pixel.RenderPixel, body)
    return _banded_temporal_megakernel(f, temporal_body) if banded else _temporal_megakernel(f, temporal_body)
  render_pixel = cast(RenderPixel, body)
  if banded:
    return _banded_megakernel(f, render_pixel)

  @wp.kernel(**kernel_structs.kernel_options(f))
  @concrete(
    geo=software_geometry_type(live_uv=f.live_material_uv),
    cams=Cameras,
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  @device_generic
  def render_megakernel(
    # Model:
    geo: SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    cams: Cameras,
    env: EnvironmentArrays,
    # In:
    seed: wp.array1d[wp.int32],
    # Out:
    out: OutputArrays,
  ):
    """One thread per (world, ray) of the (nworld, nray) launch (`camera.launch_ids`)."""
    i, j = wp.tid()
    worldid, rayid = camera.launch_ids(cams, cams.nworld, i, j)
    if worldid < 0:
      return  # a hole of the active list
    render_pixel(geo, mats, lit, cams, env, seed[0], out, worldid, rayid)

  return render_megakernel


def _temporal_megakernel(f: Features, render_pixel: pixel.RenderPixel) -> wp.Kernel:
  """The megakernel of the temporal body (one more input: the `state.Temporal` state)."""

  @wp.kernel(**kernel_structs.kernel_options(f))
  @concrete(
    geo=software_geometry_type(live_uv=f.live_material_uv),
    cams=Cameras,
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  @device_generic
  def render_temporal_megakernel(
    # Model:
    geo: SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    cams: Cameras,
    env: EnvironmentArrays,
    # In:
    seed: wp.array1d[wp.int32],
    # Out:
    out: OutputArrays,
    tmp: state.Temporal,
  ):
    """One thread per (world, ray); the temporal state is read and updated."""
    i, j = wp.tid()
    worldid, rayid = camera.launch_ids(cams, cams.nworld, i, j)
    if worldid < 0:
      return  # a hole of the active list
    render_pixel(geo, mats, lit, cams, env, seed[0], out, tmp, worldid, rayid)

  return render_temporal_megakernel


def _banded_megakernel(f: Features, render_pixel: RenderPixel):
  """`megakernel`'s banded form (CPU): thread (i, j) renders ray `first` + j."""

  @wp.kernel(**kernel_structs.kernel_options(f))
  @concrete(
    geo=software_geometry_type(live_uv=f.live_material_uv),
    cams=Cameras,
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  @device_generic
  def render_banded_megakernel(
    # Model:
    geo: SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    cams: Cameras,
    env: EnvironmentArrays,
    # In:
    seed: wp.array1d[wp.int32],
    first: wp.int32,
    # Out:
    out: OutputArrays,
  ):
    """One thread per (world, ray) of the band's rays."""
    i, j = wp.tid()
    worldid, rayid = camera.launch_ids(cams, cams.nworld, i, first + j)
    if worldid < 0:
      return  # a hole of the active list
    render_pixel(geo, mats, lit, cams, env, seed[0], out, worldid, rayid)

  return render_banded_megakernel


def _banded_temporal_megakernel(f: Features, render_pixel: pixel.RenderPixel) -> wp.Kernel:
  """`_temporal_megakernel`'s banded form (CPU): thread (i, j) renders ray `first` + j."""

  @wp.kernel(**kernel_structs.kernel_options(f))
  @concrete(
    geo=software_geometry_type(live_uv=f.live_material_uv),
    cams=Cameras,
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  @device_generic
  def render_banded_temporal_megakernel(
    # Model:
    geo: SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    cams: Cameras,
    env: EnvironmentArrays,
    # In:
    seed: wp.array1d[wp.int32],
    first: wp.int32,
    # Out:
    out: OutputArrays,
    tmp: state.Temporal,
  ):
    """One thread per (world, ray) of the band's rays; the temporal state is read and updated."""
    i, j = wp.tid()
    worldid, rayid = camera.launch_ids(cams, cams.nworld, i, first + j)
    if worldid < 0:
      return  # a hole of the active list
    render_pixel(geo, mats, lit, cams, env, seed[0], out, tmp, worldid, rayid)

  return render_banded_temporal_megakernel
