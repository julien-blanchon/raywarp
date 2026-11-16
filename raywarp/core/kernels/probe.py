"""What a GI probe sees: the light arriving at a probe along each of its rays, used to bake the
indirect light of a probe group (`core/shade/gi/`): the camera rays' surfaces and lights,
seen from a probe in the group's world (its materials, lights and environment; the static
geometry every group shares is traced in world 0).

`make_probe_shading(f, tracer)` returns shade_probe_ray(geo, mats, lit, env, world, origin,
direction, spread, seed, stream) -> (radiance, hit distance (negative: inside
geometry), sky weight, `gi.RayHit`: the radiance without its bounce, and where the bounce is read);
`probe_kernel(f)` wraps it in the GI update kernel (`gi.make_trace_kernel`).
"""

from __future__ import annotations

import dataclasses
from typing import Final

import warp as wp

from raywarp._base.types import MAXVAL as MJ_MAXVAL
from raywarp._base.types import ShapeType
from raywarp.core.build.variants import tiers
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import PLANES_SOME, Features
from raywarp.core.kernels.abi.queries import make_queries, with_visibility
from raywarp.core.kernels.abi.rays import GeometryFields, Hit, ProbeFeatures, Tracer, offset_floor, ray_offset
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params
from raywarp.core.kernels.abi.tables import emission_radiance, exposure_at, table_row
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.kernels.layer import MIN_THROUGHPUT, composite, composite_transmissive, make_coverage
from raywarp.core.kernels.lighting import DirectLights, make_scene_lights
from raywarp.core.kernels.reflections import panel_light
from raywarp.core.kernels.surface import env_radiance
from raywarp.core.shade import gi
from raywarp.core.shade.gi.kernels import BACKFACE_DISTANCE, ShadeProbeRay
from raywarp.core.shade.lights.loop import LightVisibilityAt
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials import material
from raywarp.core.shade.materials.evaluate import make_eval_surface
from raywarp.core.shade.materials.material import MaterialTables, Surface
from raywarp.core.trace.accel import query

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Hits a probe ray follows per camera-ray layer (it also meets the back faces camera rays cull).
_PROBE_LAYERS: Final = 3
# Opacity from which a hit is the probe ray's first solid surface (its returned distance).
_SOLID: Final = 0.5


@specialize
def make_probe_sky(f: Features):
  """probe_sky(env, p, world, direction, spread) -> radiance of a probe ray leaving the scene."""
  env_lit = f.env_lit
  on = toggles.on(f)
  gate = toggles.bound_on(f.bound)

  @wp.func
  def probe_sky(
    # Model:
    env: EnvironmentArrays,
    p: Params,
    # In:
    world: int,
    direction: wp.vec3,
    spread: float,
  ) -> wp.vec3:
    """Radiance of a probe ray that leaves world `world`: its lighting environment or fill."""
    ambient = env.ambient_radiance[table_row(world, env.ambient_radiance.shape[0])]
    if wp.static(env_lit):
      if gate(env.bits, wp.static(toggles.SETTING_ENV_LIT)) and on(p.rt.flags, wp.static(toggles.ENV_LIT)):
        return ambient + env_radiance(env, p, world, direction, spread)
    return ambient

  return probe_sky


@specialize
def make_probe_hit(f: Features, direct_lights: DirectLights):
  """shade_probe_hit(...) -> (radiance towards a probe from a hit (direct, emission), the weight of
  the irradiance at the hit in it (the bounce))."""
  clearcoat = f.clearcoat
  relative_emission = material.make_relative_emission(f)
  # Probe hits use the camera rays' lobes (specular curve, clearcoat).
  shading_lobes = material.lobes_of(f)

  @wp.func
  @device_generic
  def shade_probe_hit[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    env: EnvironmentArrays,
    # In:
    world: int,
    surface: Surface,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, wp.vec3]:
    """(Radiance towards a probe from direct lights and emission, weight of the bounce: the
    volume's irradiance at the hit)."""
    nov = wp.max(wp.dot(normal, view), 1.0e-4)
    lobes = shading_lobes(env.dfg_lut, surface, nov)
    radiance = direct_lights(
      geo, lit, world, lobes, point, normal, surface.origin, view, nov, seed, stream, wp.vec3(0.0),
      wp.vec3(0.0), wp.vec3(0.0),
    )  # fmt: skip
    bounce = lobes.diffuse_color * surface.occlusion
    if wp.static(clearcoat):
      # The base receives what the coat does not reflect (as `shade_surface`'s ambient term).
      bounce = bounce * (1.0 - lobes.coat_albedo)
    relative = relative_emission(env.params.relative_emission, surface)
    return radiance + emission_radiance(surface.emissive, exposure_at(env.exposure, world), relative), bounce

  return shade_probe_hit


@wp.func
def _add_layer(
  # In:
  record: gi.RayHit,
  placed: bool,
  light: wp.vec3,
  bounce: wp.vec3,
  t: float,
  normal: wp.vec3,
  outward: wp.vec3,
  opacity: float,
) -> tuple[gi.RayHit, bool]:
  """(`record` with a layer's light and bounce weight (its share) added, whether its bounce
  point is placed): the point is the first solid layer's (opacity from _SOLID), else the last
  one's (translucent layers in front of a solid one read its irradiance)."""
  fixed = record.fixed + light
  weight = record.weight + bounce
  if placed:
    return gi.ray_hit(fixed, weight, record.at, record.normal, record.outward), True
  return gi.ray_hit(fixed, weight, t, normal, outward), opacity >= wp.static(_SOLID)


def _make_probe_layers(f: Features):
  """layers(env, p) -> (whether the translucent layers are on, the layers a probe ray follows):
  compiled in (`Features.transparency`) and on (a tier build's setting, `toggles.TRANSPARENCY`),
  the settings' (`Params.max_layers`) times its back faces, else one."""
  transparency = f.transparency
  setting = toggles.setting_on(f)

  @wp.func(inline=True)
  def layers(
    # Model:
    env: EnvironmentArrays,
    p: Params,
  ) -> tuple[bool, int]:
    """(layered, nlayer)."""
    if wp.static(transparency):
      if setting(p.rt.flags, env.bits2, wp.static(toggles.TRANSPARENCY)):
        return True, wp.static(_PROBE_LAYERS) * p.max_layers
      return False, 1
    return True, 1

  return layers


def _make_probe_blend(f: Features):
  """blend(...) -> (the layer's share of what it sends, throughput, tint) of a probe ray's layer
  of opacity `opacity` (`composite`; Features.transmissive with the translucent layers on,
  `layered`: `composite_transmissive` of its coverage, `layer.Coverage`, probe rays meeting both
  faces of a closed pane)."""
  coverage = make_coverage(f, fold=False)
  transmissive = f.transmissive

  @wp.func(inline=True)
  @device_generic
  def blend[G: GeometryFields](
    # Model:
    geo: G,
    env: EnvironmentArrays,
    # In:
    layered: bool,
    throughput: float,
    tint: wp.vec3,
    surface: Surface,
    normal: wp.vec3,
    direction: wp.vec3,
    hit: Hit,
    opacity: float,
  ) -> tuple[wp.vec3, float, wp.vec3]:
    """(share, throughput, tint) past the layer."""
    if wp.static(transmissive):
      if layered:
        cover = coverage(geo, env, surface, normal, direction, hit)
        return composite_transmissive(wp.vec3(0.0), throughput, tint, wp.vec3(1.0), cover)
    share, through = composite(wp.vec3(0.0), throughput, wp.vec3(1.0), opacity)
    return share, through, tint

  return blend


@specialize
def make_probe_shading(f: Features, tracer: Tracer, light_visibility: LightVisibilityAt | None = None):
  """shade_probe_ray of the GI update (see the module docstring); `light_visibility`: its light
  loop's visibility call (`Queries.light_visibility`; None: inline)."""
  q = with_visibility(make_queries(f, tracer), light_visibility)
  closest = q.closest
  eval_surface, blend = make_eval_surface(f), _make_probe_blend(f)
  probe_sky = make_probe_sky(f)
  shade_probe_hit = make_probe_hit(f, make_scene_lights(f, q))
  transparency = f.transparency
  layers = _make_probe_layers(f)

  @wp.func
  @device_generic
  def shade_probe_ray[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: EnvironmentArrays,
    # In:
    world: int,
    origin: wp.vec3,
    direction: wp.vec3,
    spread: float,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, float, float, gi.RayHit]:
    """(radiance, distance to the first opaque hit (negative: inside geometry), sky weight, the
    radiance without the bounce and where the bounce is read)."""
    p = env.params
    radiance = wp.vec3(0.0)
    throughput = float(1.0)
    tint = wp.vec3(1.0)  # the transmissive layers' color (Features.transmissive; unused without)
    travelled = float(0.0)
    dist = float(gi.MISS_DISTANCE)
    found = bool(False)
    sky = float(0.0)
    # The record (`gi.RayHit`, `_add_layer`), placed at the first solid layer.
    record = gi.ray_hit(wp.vec3(0.0), wp.vec3(0.0), -1.0, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 1.0))
    placed = bool(False)
    # The layers a probe ray follows: the settings' (`Params.max_layers`, read at run time as the
    # camera rays' loop bound) times its back faces; one without translucent layers (not compiled
    # in, or their setting off: `layered`).
    layered, nlayer = layers(env, p)
    for _layer in range(nlayer):
      hit = closest(geo, 0, origin, direction, float(MJ_MAXVAL))
      if hit.instance_id == -1:
        light = probe_sky(env, p, world, direction, spread) * throughput
        if wp.static(f.transmissive):
          light = wp.cw_mul(light, tint)
        radiance += light
        record = gi.ray_hit(record.fixed + light, record.weight, record.at, record.normal, record.outward)
        sky = throughput
        break
      point = origin + direction * hit.dist
      t = travelled + hit.dist
      surface, normal = eval_surface(geo, mats, world, hit, point, direction, spread * t)
      opacity = float(1.0)
      if wp.static(transparency):
        if layered:
          opacity = wp.clamp(surface.alpha, 0.0, 1.0)
          if opacity < wp.static(material.OPAQUE) and wp.dot(hit.normal, direction) > 0.0:
            # The back face of a translucent layer (the far side of a glass pane): camera rays
            # cull it, so the pane attenuates once; probe rays (no culling) pass it the same way.
            travelled = t
            origin = point + direction * ray_offset(point, offset_floor(geo, hit.instance_id))
            continue
      if not found and opacity >= wp.static(_SOLID):
        found = True
        dist = t
        if opacity >= wp.static(material.OPAQUE) and gi.back_face(geo, hit, direction):
          # An opaque back face: the probe is inside geometry (DDGI's negative distance). A
          # translucent one (single-sided window glass seen from behind) is looked through.
          return (
            wp.vec3(0.0),
            wp.static(BACKFACE_DISTANCE) * t,
            0.0,
            gi.ray_hit(wp.vec3(0.0), wp.vec3(0.0), -1.0, record.normal, record.outward),
          )
      if panel_light(mats, lit, world, hit.instance_id):
        surface.emissive = wp.vec3(0.0)  # the light already brings the panel's emission
      light, bounce = shade_probe_hit(geo, lit, env, world, surface, point, normal, -direction, seed, stream)
      # The layer's share of what it sends (`composite` is linear in it): its light and its bounce,
      # the volume's irradiance (wall-aware, validity-checked: `shade.gi.cube`, as camera hits read
      # it: the 8-probe lookup's Chebyshev test passes probes behind room corners, and through these
      # bounces their light would build up inside closed rooms).
      share, throughput, tint = blend(geo, env, layered, throughput, tint, surface, normal, direction, hit, opacity)
      e = gi.irradiance(env.gi, world, point, normal, surface.outward)
      radiance += wp.cw_mul(share, light + wp.cw_mul(bounce, wp.vec3(e[0], e[1], e[2])))
      record, placed = _add_layer(
        record, placed, wp.cw_mul(share, light), wp.cw_mul(share, bounce), t, normal, surface.outward, opacity
      )
      if throughput < wp.static(MIN_THROUGHPUT):
        break
      travelled = t
      origin = point + direction * ray_offset(point, offset_floor(geo, hit.instance_id))
    return radiance, dist, sky, record

  return shade_probe_ray


# Features fields the probe rays never read (camera-ray sampling, culling, headlight, glossy
# reflections, outputs, launch options): `probe_features` resets them, so that changing one
# neither recompiles the probe kernel nor re-bakes the probes (`bake_keys.probe_key`).
# `tests/render/test_gi.py` checks the factory reads none of them.
CAMERA_ONLY: Final = frozenset(
  {
    "any_rgb",
    "backface_culling",
    "block_dim",
    "budget",
    "env_background",
    "fog",
    "glossy",
    "headlight",
    "hit_light_cap",
    "light_cap",
    "max_registers",
    "motion",
    "out_word",
    "reflection_shadows",
    "spp",
    "traversal",
    "two_sided",
    "unlit",
  }
)


def probe_features(f: Features) -> Features:
  """`f` with its `CAMERA_ONLY` fields reset (False / 0 / "") and their bits
  (`without_camera_bits`): what the probe kernel is built for."""
  values = vars(f)
  return without_camera_bits(dataclasses.replace(f, **{name: type(values[name])() for name in CAMERA_ONLY}))


def without_camera_bits(f: Features) -> Features:
  """`f` without the build-word bits of its `CAMERA_ONLY` fields (`tiers.word_bit`: reflection-hit
  sun shadows, the fog, back-face culling, two-sided instances): no probe ray reads them, so the
  GI programs and the relight kernel of tier variants that differ only there (e.g.
  `default_hit_sun` and `default`) are one build. The run-time switches stay (the switch
  functions' form is the build's)."""
  bound = sum(tiers.word_bit(name) for name in CAMERA_ONLY)
  return dataclasses.replace(f, bound=f.bound & ~bound, word=f.word & ~bound)


def probe_kernel(f: Features) -> wp.Kernel:
  """The GI probe-ray kernel of `f` (software traversal, no back-face culling; cached by
  `probe_features`)."""
  return _probe_kernel(probe_features(f))


@specialize
def _probe_kernel(f: Features) -> wp.Kernel:
  """`probe_kernel` of projected Features `f`, in the module of its word (the words of a build
  share its device functions: `kernel_structs.in_module_of`)."""
  free = kernel_structs.words_free(f)
  return kernel_structs.in_module_of(gi.make_trace_kernel(free, _probe_shading(free)), f, outputs=False)


# The scene axes of the relight build (`relight_features`): both trees, the plane loop, compact
# poses, the visibility table and the deformables' materials compiled in, each read from the
# relight word (`tiers.relight_word`); the unordered walk; no BLAS trace frames (read by no kernel
# factory).
_RELIGHT_AXES: Final = {
  "world_tree": True,
  "shared": True,
  "compact_poses": True,
  "visibility": True,
  "planes": PLANES_SOME,
  "flex_materials": True,
  "ordered_traversal": False,
  "mesh_frames": False,
}


def relight_features(f: Features) -> Features:
  """The relight kernel's build from `f`, its settings' build resolved without the scene in every
  field a tier build can take from it (`resolve.relight_build`): every shape kind the software
  traversal intersects and the deformables, every scene axis (`_RELIGHT_AXES`), bound as an OptiX
  scene-free build's (`tiers.with_switches`: every class, structure and compiled-in setting read
  from the relight word at run time, no module literal), the camera-only fields reset
  (`probe_features`). One build per probe-relevant settings: no scene, class set, sharing layout,
  light count or backend enters it, and a scene change never compiles it."""
  types = tuple(int(t) for t in ShapeType)
  axes = dataclasses.replace(f, shape_types=types, **_RELIGHT_AXES, traversal="optix", scene_free=True, word=0)
  return probe_features(tiers.with_switches(axes))


def relight_kernel(f: Features) -> wp.Kernel:
  """The per-world relight kernel of relight build `f` (`relight_features`; `gi.relight`; the
  probe kernel's shading; cached)."""
  return _relight_kernel(f)


@specialize
def _relight_kernel(f: Features) -> wp.Kernel:
  """`relight_kernel` of relight build `f`: one module, its gates reading the launch's words."""
  return gi.make_relight_kernel(f, _probe_shading(f))


def _probe_shading(f: Features) -> ShadeProbeRay:
  """The probe rays' shading of projected Features `f` (software traversal, no back-face
  culling; shadow rays through translucent casters with `f.translucent_shadows`, as the camera
  rays' and the OptiX GI programs')."""
  features = ProbeFeatures(
    f.shape_types,
    f.trees,
    translucent_shadows=f.translucent_shadows,
    world_optics=f.world_optics,
    alpha_cutout=f.alpha_cutout,
    live_material_uv=f.live_material_uv,
    world_uv=f.world_uv,
    bound=f.bound,
  )
  tracer = query.software_tracer(features, cull=False)
  return make_probe_shading(f, tracer)
