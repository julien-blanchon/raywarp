"""The first-hit outputs beyond depth / seg: one writer per kind of value, shared by every pass.

Two groups, by where their values come from:

* **Hit outputs** (`make_write_hit`): semantic id, position, uv, material id, geometric normal.
  They follow from the camera ray and its closest hit alone, so every pass that writes depth /
  seg writes them at the same place, from the same ray: the plain body, the motion body (the
  centre ray with multisampling), and the geometry-only bodies.
* **Surface outputs** (`make_write_surface`): albedo and the shading normal (the guides), and
  roughness, metallic, emission, direct, indirect and shadow. They come from the pixel center's
  surface: `pixel.make_trace_pixel` captures them at the first layer into a `SurfaceRecord`,
  which travels to the passes' surface writes.

Every store is compiled in and reads the output word (`output_layout`, `toggles.out_on`): a
build's modules fold the absent ones and their values (the record's fields, the shadow output's
rays) away.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple, cast

import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.output_layout import (
  OUT_ALBEDO,
  OUT_DIRECT,
  OUT_EMISSION,
  OUT_GEOMETRIC,
  OUT_INDIRECT,
  OUT_MATERIAL,
  OUT_METALLIC,
  OUT_NORMAL,
  OUT_POSITION,
  OUT_ROUGHNESS,
  OUT_SEMANTIC,
  OUT_SHADOW,
  OUT_UV,
  make_copy,
  put1,
  put2,
  put3,
)
from raywarp.core.kernels.abi.rays import FLEX_HIT, GeometryFields, Hit
from raywarp.core.kernels.abi.structs import OutputArrays, i32s, vec2hs, vec2s
from raywarp.core.kernels.abi.tables import batched, emission_radiance, material_id
from raywarp.core.kernels.abi.toggles import out_half, out_on
from raywarp.core.kernels.generic import device_generic
from raywarp.core.kernels.lighting import DirectLighting
from raywarp.core.shade.lights import ies as ies_lib
from raywarp.core.shade.lights import punctual
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.flex import flex_uv
from raywarp.core.shade.materials.material import Surface
from raywarp.core.shade.shadows.visibility import LightVisibility
from raywarp.core.trace.accel.cutout import make_hit_uv

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.struct
class SurfaceRecord:
  """The pixel center's first surface, as its surface outputs need it (`capture`).

  Attributes:
    normal: the shading normal (the guides' normal).
    point: the hit point; origin: its secondary-ray origin (`Surface.origin`, the shadow output's
      rays).
    roughness, metallic: the textured perceptual roughness and metallic.
    emission: the emission term of its radiance (before exposure).
    direct, indirect: its radiance from the lights / from everything else but the emission.
  """

  normal: wp.vec3
  point: wp.vec3
  origin: wp.vec3
  roughness: float
  metallic: float
  emission: wp.vec3
  direct: wp.vec3
  indirect: wp.vec3


def split_lighting(f: Features) -> bool:
  """Whether the camera rays' shading also returns its direct / indirect split and its emission
  (the direct, indirect and emission outputs; for coated materials the emission under the coat,
  `surface.coat_emission`): every lit build (unlit has no lights)."""
  return not f.unlit


@wp.func
def zero_record() -> SurfaceRecord:
  """The record of a ray that hits nothing: every field zero (a Warp struct starts zeroed)."""
  return SurfaceRecord()


@wp.func
def capture(
  # In:
  surface: Surface,
  normal: wp.vec3,
  point: wp.vec3,
  emission: wp.vec3,
  direct: wp.vec3,
  indirect: wp.vec3,
) -> SurfaceRecord:
  """The record of a shaded first surface (`emission`: the emission term of its radiance)."""
  r = SurfaceRecord()
  r.normal = normal
  r.point = point
  r.origin = surface.origin
  r.roughness = surface.perceptual_roughness
  r.metallic = surface.metallic
  r.emission = emission
  r.direct = direct
  r.indirect = indirect
  return r


def _make_hit_material(f: Features, *, uv: bool):
  """hit_material(geo, worldid, hit, point, direction) -> (material id, uv) of a camera-ray hit:
  -1 / 0 on a miss; flex hits read their flex's material and UVs; `uv`: else uv 0."""
  hit_uv = make_hit_uv(f.shape_types, compact=f.compact_poses, world_uv=f.world_uv, live_uv=f.live_material_uv, bound=f.bound)
  flex = int(ShapeType.FLEX) in f.shape_types  # deformables
  gate = toggles.bound_on(f.bound)  # the deformables bit

  @wp.func
  @device_generic
  def hit_material[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
  ) -> tuple[int, wp.vec2]:
    """The material and texture coordinate at `hit` (at `point` along `direction`)."""
    gi = hit.instance_id
    mat = int(-1)
    t = wp.vec2(0.0)
    if gi >= 0:
      mat = material_id(geo.instance_material, geo.material_base_color.shape[1], worldid, gi)
      if wp.static(uv):
        if mat >= 0:
          t, _has = hit_uv(geo, worldid, hit, point, direction, mat)
    if wp.static(flex):
      if gate(geo.bits, wp.static(toggles.FLEX)) and gi == wp.static(FLEX_HIT):
        mat = geo.flex_part[hit.mesh_id].material
        if wp.static(uv):
          t = flex_uv(geo, worldid, hit, mat)
    return mat, t

  return hit_material


def make_write_hit(f: Features):
  """write_hit(geo, out, worldid, adr, hit, origin, direction): the hit outputs of a camera ray
  (module doc): misses write 0 (semantic, position, uv, normal) and -1 (material)."""
  hit_material = _make_hit_material(f, uv=True)

  @wp.func
  @device_generic
  def write_hit[G: GeometryFields](
    # Model:
    geo: G,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    adr: int,
    hit: Hit,
    origin: wp.vec3,
    direction: wp.vec3,
  ):
    """Writes the hit outputs of `hit` (the ray (origin, direction)'s closest hit) the word has."""
    gi = hit.instance_id
    if out_on(out.bits, wp.static(OUT_SEMANTIC)):
      cls = int(0)
      if gi == wp.static(FLEX_HIT):
        cls = out.table.semantic_flex
      elif gi != -1:
        cls = batched(out.table.semantic_id, worldid, gi)
      i32s(out, out.table.semantic)[worldid, adr] = cls
    if out_on(out.bits, wp.static(OUT_POSITION)):
      p = wp.vec3(0.0)
      if gi != -1:
        p = origin + direction * hit.dist
      put3(out, out_half(out.half, wp.static(OUT_POSITION)), out.table.position, out.table.position16, worldid, adr, p)
    if out_on(out.bits, wp.static(OUT_UV | OUT_MATERIAL)):
      mat, uv = hit_material(geo, worldid, hit, origin + direction * hit.dist, direction)
      if out_on(out.bits, wp.static(OUT_MATERIAL)):
        i32s(out, out.table.material)[worldid, adr] = mat
      if out_on(out.bits, wp.static(OUT_UV)):
        put2(out, out_half(out.half, wp.static(OUT_UV)), out.table.uv, out.table.uv16, worldid, adr, uv)
    if out_on(out.bits, wp.static(OUT_GEOMETRIC)):
      n = wp.vec3(0.0)
      if gi != -1:
        n = wp.where(wp.dot(hit.normal, direction) > 0.0, -hit.normal, hit.normal)
      put3(out, out_half(out.half, wp.static(OUT_NORMAL)), out.table.normal, out.table.normal16, worldid, adr, n)

  return write_hit


def make_write_surface(f: Features, lighting: DirectLighting):
  """write_surface(geo, lit, out, worldid, adr, albedo, rec, seed): the surface outputs of
  the pixel center (module doc) from its `SurfaceRecord` `rec`; `lighting`: the camera rays' (the
  shadow output's visibility)."""
  shadow = make_shadow(f, lighting)

  @wp.func
  @device_generic
  def write_surface[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # Out:
    out: OutputArrays,
    # In:
    worldid: int,
    adr: int,
    albedo: wp.vec3,
    rec: SurfaceRecord,
    seed: int,
  ):
    """Writes the surface outputs the word has (`shadow`: one visibility query of the light)."""
    bits, half = out.bits, out.half
    if out_on(bits, wp.static(OUT_ALBEDO)):
      put3(out, out_half(half, wp.static(OUT_ALBEDO)), out.table.albedo, out.table.albedo16, worldid, adr, albedo)
    if out_on(bits, wp.static(OUT_NORMAL)):
      put3(out, out_half(half, wp.static(OUT_NORMAL)), out.table.normal, out.table.normal16, worldid, adr, rec.normal)
    if out_on(bits, wp.static(OUT_ROUGHNESS)):
      put1(
        out, out_half(half, wp.static(OUT_ROUGHNESS)), out.table.roughness, out.table.roughness16, worldid, adr, rec.roughness
      )
    if out_on(bits, wp.static(OUT_METALLIC)):
      put1(out, out_half(half, wp.static(OUT_METALLIC)), out.table.metallic, out.table.metallic16, worldid, adr, rec.metallic)
    if out_on(bits, wp.static(OUT_EMISSION)):
      put3(out, out_half(half, wp.static(OUT_EMISSION)), out.table.emission, out.table.emission16, worldid, adr, rec.emission)
    if out_on(bits, wp.static(OUT_DIRECT)):
      put3(out, out_half(half, wp.static(OUT_DIRECT)), out.table.direct, out.table.direct16, worldid, adr, rec.direct)
    if out_on(bits, wp.static(OUT_INDIRECT)):
      put3(out, out_half(half, wp.static(OUT_INDIRECT)), out.table.indirect, out.table.indirect16, worldid, adr, rec.indirect)
    if out_on(bits, wp.static(OUT_SHADOW)):
      v = shadow(geo, lit, worldid, out.table.shadow_light, rec, seed)
      put1(out, out_half(half, wp.static(OUT_SHADOW)), out.table.shadow, out.table.shadow16, worldid, adr, v)

  return write_surface


def make_shadow(f: Features, lighting: DirectLighting):
  """shadow(geo, lit, worldid, li, rec, seed) -> visibility of light `li` at the recorded
  surface: its shadow rays (`lighting.light_visibility`, the shading's samples: stream 0),
  0 where the surface faces away, 1 for a light that casts no shadow, with shadows off (also a
  tier build's run-time switch, as the shading's), or on a miss (a zero record's normal)."""
  has_spot = f.has_spot
  shadows = f.shadows
  on = toggles.on(f)
  tinted = f.tinted_shadows
  profiled = f.ies
  gate = toggles.bound_on(f.bound)  # the IES class bit
  # Lighting uses this same build-time tinted flag; aliases add no device wrapper.
  visibility_rgb = cast("LightVisibility[wp.vec3]", lighting.light_visibility)
  visibility_scalar = cast("LightVisibility[float]", lighting.light_visibility)

  @wp.func
  @device_generic
  def shadow[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    rec: SurfaceRecord,
    seed: int,
  ) -> float:
    """The light's visible fraction at `rec`."""
    if wp.length_sq(rec.normal) == 0.0:
      return 1.0
    light = punctual.fetch(lit, worldid, li)
    spotted = False
    if wp.static(has_spot):
      spotted = gate(lit.bits, wp.static(toggles.SCENE_SPOT))
    l, dist, scale = punctual.eval_light(light, spotted, rec.point, lit.falloff_floor)
    if wp.static(profiled):
      if gate(lit.bits, wp.static(toggles.CLASS_IES)):
        scale = scale * ies_lib.factor(lit.ies, light, l)
    if wp.dot(rec.normal, l) <= 0.0 or wp.length_sq(scale) == 0.0:
      return 0.0
    v = float(1.0)
    if wp.static(shadows):
      if light.cast_shadow and on(geo.rt.flags, wp.static(toggles.SHADOWS)):
        if wp.static(tinted):
          vis = visibility_rgb(geo, worldid, light, rec.origin, rec.normal, l, dist, seed, 0, lit.nlight, li)
          v = (vis[0] + vis[1] + vis[2]) * wp.static(1.0 / 3.0)
        else:
          v = visibility_scalar(geo, worldid, light, rec.origin, rec.normal, l, dist, seed, 0, lit.nlight, li)
    return v

  return shadow


@wp.func
def emission_term(
  # In:
  surface: Surface,
  exposure: float,
  relative: int,
) -> wp.vec3:
  """The emission a shaded surface adds to its radiance, the emission output
  (`emission_radiance`)."""
  return emission_radiance(surface.emissive, exposure, relative)


class FirstHitWriters(NamedTuple):
  """The first-hit writes of one body (`make_writers`): `geometry` (depth / seg:
  `rays.write_geometry`), `hit` and `surface` (the module doc)."""

  geometry: Callable
  hit: Callable
  surface: Callable


def make_writers(f: Features, write_geometry: Callable, lighting: DirectLighting | None):
  """The first-hit writers of `f` (`lighting`: the camera rays', for the shadow output; None in a
  geometry-only body, which writes no surface output: its `surface` is a stand-in, never
  emitted)."""
  write_surface = make_write_surface(f, lighting) if lighting is not None else write_geometry
  return FirstHitWriters(write_geometry, make_write_hit(f), write_surface)


# Copies of a vec3 / scalar output's element (`output_layout.make_copy`).
_copy3: Final = make_copy(wp.vec3, wp.vec3h)
_copy1: Final = make_copy(float, wp.float16)


@wp.func
def copy_first_hit(
  # In:
  hi: OutputArrays,
  # Out:
  out: OutputArrays,
  # In:
  worldid: int,
  src: int,
  dst: int,
):
  """Copies every first-hit buffer of the word (`out`'s: `hi` stores them alike) from a
  supersampled image's first sample `src` to pixel `dst` (`trace/optix/supersample.py`; depth / seg
  are copied there)."""
  bits, half = out.bits, out.half
  if out_on(bits, wp.static(OUT_ALBEDO)):
    a16 = out_half(half, wp.static(OUT_ALBEDO))
    _copy3(hi, out, a16, hi.table.albedo, hi.table.albedo16, out.table.albedo, out.table.albedo16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_NORMAL | OUT_GEOMETRIC)):
    n16 = out_half(half, wp.static(OUT_NORMAL))
    _copy3(hi, out, n16, hi.table.normal, hi.table.normal16, out.table.normal, out.table.normal16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_POSITION)):
    p16 = out_half(half, wp.static(OUT_POSITION))
    _copy3(hi, out, p16, hi.table.position, hi.table.position16, out.table.position, out.table.position16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_EMISSION)):
    e16 = out_half(half, wp.static(OUT_EMISSION))
    _copy3(hi, out, e16, hi.table.emission, hi.table.emission16, out.table.emission, out.table.emission16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_DIRECT)):
    d16 = out_half(half, wp.static(OUT_DIRECT))
    _copy3(hi, out, d16, hi.table.direct, hi.table.direct16, out.table.direct, out.table.direct16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_INDIRECT)):
    i16 = out_half(half, wp.static(OUT_INDIRECT))
    _copy3(hi, out, i16, hi.table.indirect, hi.table.indirect16, out.table.indirect, out.table.indirect16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_UV)):
    if out_half(half, wp.static(OUT_UV)):
      vec2hs(out, out.table.uv16)[worldid, dst] = vec2hs(hi, hi.table.uv16)[worldid, src]
    else:
      vec2s(out, out.table.uv)[worldid, dst] = vec2s(hi, hi.table.uv)[worldid, src]
  if out_on(bits, wp.static(OUT_ROUGHNESS)):
    r16 = out_half(half, wp.static(OUT_ROUGHNESS))
    _copy1(
      hi, out, r16, hi.table.roughness, hi.table.roughness16, out.table.roughness, out.table.roughness16, worldid, src, dst
    )
  if out_on(bits, wp.static(OUT_METALLIC)):
    m16 = out_half(half, wp.static(OUT_METALLIC))
    _copy1(hi, out, m16, hi.table.metallic, hi.table.metallic16, out.table.metallic, out.table.metallic16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_SHADOW)):
    s16 = out_half(half, wp.static(OUT_SHADOW))
    _copy1(hi, out, s16, hi.table.shadow, hi.table.shadow16, out.table.shadow, out.table.shadow16, worldid, src, dst)
  if out_on(bits, wp.static(OUT_SEMANTIC)):
    i32s(out, out.table.semantic)[worldid, dst] = i32s(hi, hi.table.semantic)[worldid, src]
  if out_on(bits, wp.static(OUT_MATERIAL)):
    i32s(out, out.table.material)[worldid, dst] = i32s(hi, hi.table.material)[worldid, src]
