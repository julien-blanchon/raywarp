"""How much light each lamp delivers to a surface point (physical units, falloff, spot cones), and
the packed per-frame light table the shading reads.

Intensity in lux (directional) or candela (point, spot), inverse-square falloff windowed at the
light's range, smooth spot cone. A point / spot light with range 0 and identity polynomial
attenuation (1, 0, 0) has no falloff (MuJoCo's default OpenGL light): its intensity is the
illuminance [lux] it gives a facing surface at any distance. Classic MuJoCo lights can instead
apply their constant/linear/quadratic polynomial. Adapters convert other light models to these
units (see `raywarp.integrations.mujoco.lights`).

The shading never reads the per-field light tables (`light.pos`, `light.strength`, ...): a pack
kernel (`shade/lights/pack.py`) gathers them once per frame into one `Light` per (row, light)
with the per-point constants precomputed, and every light loop reads one struct per
light (`Lights.lights`).

The same kernel also writes each light's hot record (`LightHot`, 48 B: position and range, color
and intensity, direction and flags; `Lights.hot`), what the cheap ranking of capped light loops
(`shade/lights/sources.py`) reads for every light of a cell. The full 132 B `Light` keeps the
cold fields (classic attenuation, cone, area shape, emitter radius) for the few lights evaluated
exactly. `eval_hot` is `eval_light` on the hot record, bit for bit: same values, same operations;
the cold fields are read only for the lights whose flags need them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal, cast, overload

import warp as wp

from raywarp._base.types import MAXVAL, LightType
from raywarp.core.kernels.abi.tables import batched, table_row
from raywarp.core.kernels.generic import specialize
from raywarp.core.shade.lights.ies_table import IesTable

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

type Visibility = float | wp.vec3
type VisibilityOf = Callable[[float], Visibility]

LIGHT_SPOT: Final = int(LightType.SPOT)
LIGHT_DIRECTIONAL: Final = int(LightType.DIRECTIONAL)
LIGHT_POINT: Final = int(LightType.POINT)
LIGHT_COSINE: Final = int(LightType.COSINE)
LIGHT_TUBE: Final = int(LightType.TUBE)

# Spot lights use `cutoff` as the outer cone; the inner cone is `light.cone_inner` x cutoff
# (`types.SPOT_INNER_FRACTION` of the outer cone where no table gives it: MuJoCo's spots).
# `Light.cos_outer` of lights without a cone: every direction is inside (the cone factor is 1).
NO_CONE: Final = -3.0
# Narrowest smooth edge of a spot cone [rad] (UsdLux softness 0: a hard cut-off).
CONE_MIN_WIDTH: Final = 1.0e-4
# Light-grid lookups (`cell_of`): offset of a surface point along its normal, in cells, so that a
# surface lying on a cell boundary does not pick a cell per pixel by rounding.
GRID_SURFACE_OFFSET: Final = 1.0e-3
# Distance `eval_light` reports for a directional light [m] (MuJoCo's infinity, `types.MAXVAL`).
_DIRECTIONAL_DISTANCE: Final = MAXVAL
# Floor of the light distance a direction is normalised by [m] (a point on the light).
_MIN_LIGHT_DISTANCE: Final = 1.0e-9
# Floor of the distance of the windowed inverse-square falloff [m] in a scene of 10 cm or more
# (`kernels.abi.rays.SCENE_REFERENCE`), its share of a smaller one (`falloff_floor`: a fixed 1 cm
# clamped every light of a millimetre-scale room, 20 % dark).
MIN_FALLOFF_DISTANCE: Final = 1.0e-2
# MuJoCo Warp's safe_div uses mjMINVAL for a zero classic attenuation denominator.
_MIN_CLASSIC_ATTENUATION: Final = 1.0e-15
# `LightHot.flags`: the light type in the low bits, then one bit each for: on (`Light.active`),
# casts shadows, a classic attenuation polynomial other than the identity (1, 0, 0), a spot cone
# (`Light.cos_outer` > NO_CONE).
HOT_TYPE_MASK: Final = 0xF
HOT_ACTIVE: Final = 0x10
HOT_CASTER: Final = 0x20
HOT_CLASSIC: Final = 0x40
HOT_CONED: Final = 0x80
# `LightHot.flags` >> HOT_IES_SHIFT: the light's IES profile instance (`Light.ies`).
HOT_IES_SHIFT: Final = 8


@wp.struct
class Light:
  """One light of one world, as the shading reads it (packed per frame: `shade/lights/pack.py`).

  Punctual form (every light): `type`, `pos`, `dir`, `color` x `intensity`, `range`,
  `attenuation` (classic polynomial, identity (1, 0, 0) otherwise), `radius`
  (point / spot: emitter radius of soft shadows; directional: angular radius of the disc [rad]),
  and the spot cone's precomputed `cos_outer`, `cone_start` / `cone_scale` (`cone_factor`;
  `NO_CONE`: no cone). Area form
  (`shape` != LightShape.NONE, `lighting.area_lights="ltc"`): `pos` the centre, `dir` the emission
  direction, `axis_x` / `axis_y` the shape's frame, `half` its half extents, `radiance` the
  uniform surface radiance. `active` False: the light is off (no radiance, no shadow rays).
  `ies`: 1 + the light's IES profile in `Lights.ies` (0: none; read only with Features.ies,
  `shade/lights/ies.py`; the instance carries the light's UsdLux angle scale).
  """

  type: wp.int32
  shape: wp.int32
  cast_shadow: bool
  active: bool
  pos: wp.vec3
  dir: wp.vec3
  color: wp.vec3
  intensity: wp.float32
  range: wp.float32
  attenuation: wp.vec3
  radius: wp.float32
  cos_outer: wp.float32
  cone_start: wp.float32
  cone_scale: wp.float32
  axis_x: wp.vec3
  axis_y: wp.vec3
  half: wp.vec2
  radiance: wp.vec3
  ies: wp.int32


@wp.struct
class LightHot:
  """The fields of a `Light` that ranking every light of a cell reads (module doc): `pos`,
  `range`, `color`, `intensity`, `dir` as the `Light`'s, `flags` its type and HOT_* bits."""

  pos: wp.vec3
  range: wp.float32
  color: wp.vec3
  intensity: wp.float32
  dir: wp.vec3
  flags: wp.int32


@wp.struct
class Lights:
  """The lights of a frame as the shading kernels read them.

  `lights` (1 or nworld, max(nlight, 1)): the packed lights (row worldid % rows); `hot` their hot
  records (same shape); `nlight` the light count; the light visibility grid
  (`shade/lights/grid.py` `Grid`: its movable lights at the end of the lists, its cells' far
  fields; a single cell listing every light when not built); the LTC tables of
  the area lights (read only with Features.area_lights); `ies` the IES profiles (read only with
  Features.ies, `shade/lights/ies.py`); `visible` (max(nvisible, 1),) the (light, LightShape) of the
  `nvisible` lights whose emitting surface camera rays see (`shade/lights/visible.py`).
  """

  lights: wp.array2d[Light]
  hot: wp.array2d[LightHot]
  nlight: wp.int32
  grid_origin: wp.vec3
  grid_inv_cell: wp.float32
  grid_dims: wp.vec3i
  grid_start: wp.array[wp.int32]
  grid_lights: wp.array[wp.int32]
  grid_valid: wp.array[wp.int32]
  grid_far: wp.array[wp.vec3]
  ltc_minv: wp.Texture2D
  ltc_mag: wp.Texture2D
  ies: IesTable
  visible: wp.array[wp.vec2i]
  nvisible: wp.int32
  # The build word's light copy (`tiers.word`: every bit but the geometry's `toggles.GEO_*`; the
  # light code reads its light classes, culling and `SCENE_*` bits through `toggles.bound_on`; an
  # OptiX module binds it: `trace/optix/device/params.bound_words`); `bits2` its second half
  # (`toggles.HIGH` bits, written together by `toggles.store`).
  bits: wp.int32
  bits2: wp.int32
  # The squared distance floor of the inverse-square falloff [m^2] (`falloff_floor`).
  falloff_floor: wp.float32


def falloff_floor(scale: float) -> float:
  """`Lights.falloff_floor` of a scene of floor scale `scale` (`kernels.abi.rays.scene_scale`):
  (MIN_FALLOFF_DISTANCE x scale)^2."""
  d = MIN_FALLOFF_DISTANCE * scale
  return d * d


@wp.func
def fetch(
  # Model:
  lit: Lights,
  # In:
  worldid: wp.int32,
  li: wp.int32,
) -> Light:
  """Light `li` of world `worldid`."""
  return batched(lit.lights, worldid, li)


@wp.func_native("""
const char* at = reinterpret_cast<const char*>(hot.data) + row * hot.strides[0] + li * hot.strides[1] + 16 * quad;
#if defined(__CUDA_ARCH__)
const float4 v = __ldg(reinterpret_cast<const float4*>(at));
return wp::vec4(v.x, v.y, v.z, v.w);
#else
const float* f = reinterpret_cast<const float*>(at);
return wp::vec4(f[0], f[1], f[2], f[3]);
#endif
""")
def _hot_quad(
  # In:
  hot: wp.array2d[LightHot],
  row: wp.int32,
  li: wp.int32,
  quad: wp.int32,
) -> wp.vec4:
  """16-byte word `quad` (0-2) of a hot record, one 128-bit load on the GPU (a `LightHot` is 48 B:
  its words are 16-byte aligned in a Warp allocation); Warp's 4-byte aligned vectors load field by
  field."""
  ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __float_as_int(x);
#else
int i;
memcpy(&i, &x, sizeof(i));
return i;
#endif
""")
def _bits(
  # In:
  x: wp.float32,
) -> wp.int32:
  """The bits of a float as an int (the flags word of a hot record read as a float)."""
  ...


@wp.func
def fetch_hot(
  # Model:
  lit: Lights,
  # In:
  worldid: wp.int32,
  li: wp.int32,
) -> LightHot:
  """The hot record of light `li` of world `worldid`, in three 128-bit loads (`_hot_quad`)."""
  row = table_row(worldid, lit.hot.shape[0])
  a = _hot_quad(lit.hot, row, li, 0)
  b = _hot_quad(lit.hot, row, li, 1)
  c = _hot_quad(lit.hot, row, li, 2)
  h = LightHot()
  h.pos = wp.vec3(a[0], a[1], a[2])
  h.range = a[3]
  h.color = wp.vec3(b[0], b[1], b[2])
  h.intensity = b[3]
  h.dir = wp.vec3(c[0], c[1], c[2])
  h.flags = _bits(c[3])
  return h


@wp.func
def hot_of(
  # In:
  light: Light,
) -> LightHot:
  """The hot record of a packed light."""
  h = LightHot()
  h.pos = light.pos
  h.range = light.range
  h.color = light.color
  h.intensity = light.intensity
  h.dir = light.dir
  flags = light.type
  if light.active:
    flags = flags | wp.static(HOT_ACTIVE)
  if light.cast_shadow:
    flags = flags | wp.static(HOT_CASTER)
  coeff = light.attenuation
  if coeff[0] != 1.0 or coeff[1] != 0.0 or coeff[2] != 0.0:
    flags = flags | wp.static(HOT_CLASSIC)
  if light.cos_outer > wp.static(NO_CONE):
    flags = flags | wp.static(HOT_CONED)
  # The IES profile instance (`Light.ies`, 0: none) above the HOT_* bits.
  h.flags = flags | (light.ies << wp.static(HOT_IES_SHIFT))
  return h


@wp.func
def cell_of(
  # In:
  origin: wp.vec3,
  inv_cell: wp.float32,
  dims: wp.vec3i,
  p: wp.vec3,
  n: wp.vec3,
) -> wp.int32:
  """Index of the light-grid cell (`shade/lights/grid.py`) holding surface point `p` of normal
  `n`.

  Clamped to the grid. Looked up GRID_SURFACE_OFFSET cells off the surface along `n`: a surface
  lying on a cell boundary (a floor at z = 0 with an origin at z = -1) would otherwise pick either
  cell from the rounding of its hit points, flickering per pixel between its own side's lights
  and the other side's.
  """
  q = (p - origin) * inv_cell + n * GRID_SURFACE_OFFSET
  x = wp.clamp(wp.int32(wp.floor(q[0])), 0, dims[0] - 1)
  y = wp.clamp(wp.int32(wp.floor(q[1])), 0, dims[1] - 1)
  z = wp.clamp(wp.int32(wp.floor(q[2])), 0, dims[2] - 1)
  return (z * dims[1] + y) * dims[0] + x


@wp.func
def _attenuate(
  # In:
  c: wp.vec3,
  v: wp.float32,
) -> wp.vec3:
  """Radiance `c` times a light's visibility (a fraction)."""
  return c * v


@wp.func
def _attenuate(
  # In:
  c: wp.vec3,
  v: wp.vec3,
) -> wp.vec3:
  """Radiance `c` times a light's visibility (an RGB transmittance: tinted shadows)."""
  return wp.cw_mul(c, v)


# Warp registers both overloads on one function object; the type checker sees only the last
# definition: the combined signature.
attenuate: Final = cast("Callable[[wp.vec3, Visibility], wp.vec3]", _attenuate)


@overload
def visibility_ops(tinted: Literal[False]) -> tuple[type[float], Callable[[float], float]]: ...


@overload
def visibility_ops(tinted: Literal[True]) -> tuple[type[wp.vec3], Callable[[float], wp.vec3]]: ...


@overload
def visibility_ops(tinted: bool) -> tuple[type[float] | type[wp.vec3], VisibilityOf]: ...


@specialize
def visibility_ops(tinted: bool) -> tuple[type[float] | type[wp.vec3], VisibilityOf]:
  """(visibility type, vis_of(float) -> that type): RGB with tinted shadows, else a float (the
  overloads give a literal `tinted` its exact converter type)."""
  if tinted:

    @wp.func
    def vis_rgb(
      # In:
      v: wp.float32,
    ) -> wp.vec3:
      """A visibility fraction as an RGB transmittance."""
      return wp.vec3(v)

    return wp.vec3, vis_rgb

  @wp.func
  def vis_float(
    # In:
    v: wp.float32,
  ) -> wp.float32:
    """A visibility fraction as itself."""
    return v

  return float, vis_float


@wp.func
def headlight(
  # In:
  color: wp.vec3,
  intensity: wp.float32,
  direction: wp.vec3,
) -> Light:
  """The headlight: a directional light along the camera's view, casting no shadow."""
  light = Light()
  light.type = wp.static(LIGHT_DIRECTIONAL)
  light.cast_shadow = False
  light.active = True
  light.dir = direction
  light.color = color
  light.intensity = intensity
  light.range = 1.0
  light.attenuation = wp.vec3(1.0, 0.0, 0.0)
  light.cos_outer = wp.static(NO_CONE)
  light.cone_start = 0.0
  light.cone_scale = 1.0
  return light


# The light id of the camera headlight's entry in a light loop (one more entry after the cell's
# lights: `light_at`).
HEADLIGHT: Final = -1


@wp.func(inline=True)
def light_at(
  # Model:
  lit: Lights,
  # In:
  worldid: wp.int32,
  li: wp.int32,
  head_dir: wp.vec3,
  head: wp.vec3,
) -> Light:
  """Light `li` of world `worldid`, or the camera headlight (`li` == HEADLIGHT): a directional
  light along `head_dir` of radiance scale `head` (color x illuminance), casting no shadow."""
  if li == wp.static(HEADLIGHT):
    return headlight(head, 1.0, head_dir)
  return fetch(lit, worldid, li)


@wp.func
def cone_factor(
  # In:
  light: Light,
  cos_theta: wp.float32,
) -> wp.float32:
  """UsdLux ShapingAPI cone: 1 inside the smooth-start angle, 0 beyond the cutoff, and
  1 - smoothstep of the off-axis angle in between (glTF / MuJoCo spots use the same curve, their
  inner cone as the start)."""
  if cos_theta <= light.cos_outer:
    return 0.0
  t = wp.clamp((wp.acos(wp.clamp(cos_theta, -1.0, 1.0)) - light.cone_start) * light.cone_scale, 0.0, 1.0)
  return 1.0 - t * t * (3.0 - 2.0 * t)


@wp.func
def _toward(
  # In:
  pos: wp.vec3,
  light_range: wp.float32,
  point: wp.vec3,
  floor: wp.float32,
) -> tuple[wp.vec3, wp.float32, wp.float32]:
  """(direction, distance, windowed inverse-square falloff) from `point` to a positional light at
  `pos` (range 0: no falloff, 1); the squared distance floored at `floor`
  (`Lights.falloff_floor`)."""
  attenuation = wp.float32(1.0)
  to_light = pos - point
  dist = wp.length(to_light)
  l = to_light / wp.max(dist, wp.static(_MIN_LIGHT_DISTANCE))
  if light_range > 0.0:
    d2 = dist * dist
    r = light_range
    factor = d2 / (r * r)
    window = wp.clamp(1.0 - factor * factor, 0.0, 1.0)
    attenuation = window * window / wp.max(d2, floor)
  return l, dist, attenuation


@wp.func
def _classic(
  # In:
  attenuation: wp.float32,
  coeff: wp.vec3,
  dist: wp.float32,
) -> wp.float32:
  """`attenuation` over a classic MuJoCo light's (constant, linear, quadratic) polynomial."""
  divisor = coeff[0] + coeff[1] * dist + coeff[2] * dist * dist
  return attenuation / wp.where(divisor != 0.0, divisor, wp.static(_MIN_CLASSIC_ATTENUATION))


@wp.func
def _lobe(
  # In:
  attenuation: wp.float32,
  ltype: wp.int32,
  cos_theta: wp.float32,
) -> wp.float32:
  """`attenuation` times the emission lobe of a cosine / tube light at `cos_theta` off its axis
  (other types: unchanged)."""
  if ltype == wp.static(LIGHT_COSINE):
    # A flat Lambertian emitter seen from a distance: its intensity falls with cos(theta).
    return attenuation * wp.max(cos_theta, 0.0)
  if ltype == wp.static(LIGHT_TUBE):
    # A Lambertian cylinder (no end caps) along `dir`: its projected area 2 r l sin(theta).
    return attenuation * wp.sqrt(wp.max(1.0 - cos_theta * cos_theta, 0.0))
  return attenuation


@wp.func
def eval_light(
  # In:
  light: Light,
  has_spot: bool,
  point: wp.vec3,
  floor: wp.float32,
) -> tuple[wp.vec3, wp.float32, wp.vec3]:
  """Returns (direction to light, distance, radiance scale); `floor`: the falloff's squared
  distance floor (`Lights.falloff_floor`).

  The scale multiplies the BRDF lobes times NoL (both lobes: one light color); zero means the
  light does not reach the point (outside cone or range). `has_spot` False skips the spot cone and
  the cosine / tube lobes (no SPOT / COSINE / TUBE lights).
  """
  dist = wp.float32(wp.static(_DIRECTIONAL_DISTANCE))
  attenuation = wp.float32(1.0)
  if light.type == wp.static(LIGHT_DIRECTIONAL):
    # Directions are unit vectors.
    l = -light.dir
  else:
    l, dist, attenuation = _toward(light.pos, light.range, point, floor)
    # Classic MuJoCo point/spot lights use this polynomial instead of a physical range.
    # Other lights carry the identity (1, 0, 0); keep the physical falloff above intact.
    coeff = light.attenuation
    if coeff[0] != 1.0 or coeff[1] != 0.0 or coeff[2] != 0.0:
      attenuation = _classic(attenuation, coeff, dist)
    if has_spot:
      cos_theta = wp.dot(-l, light.dir)
      attenuation = _lobe(attenuation, light.type, cos_theta)
      if light.cos_outer > wp.static(NO_CONE):
        attenuation = attenuation * cone_factor(light, cos_theta)

  return l, dist, light.color * (light.intensity * attenuation)


@wp.func
def eval_hot(
  # Model:
  lit: Lights,
  # In:
  worldid: wp.int32,
  li: wp.int32,
  hot: LightHot,
  has_spot: bool,
  point: wp.vec3,
) -> tuple[wp.vec3, wp.float32, wp.vec3]:
  """`eval_light` of light `li` of world `worldid` from its hot record `hot` (bit for bit); the
  full record is read only for a classic polynomial or a spot cone (`hot.flags`)."""
  dist = wp.float32(wp.static(_DIRECTIONAL_DISTANCE))
  attenuation = wp.float32(1.0)
  ltype = hot.flags & wp.static(HOT_TYPE_MASK)
  if ltype == wp.static(LIGHT_DIRECTIONAL):
    l = -hot.dir
  else:
    l, dist, attenuation = _toward(hot.pos, hot.range, point, lit.falloff_floor)
    if (hot.flags & wp.static(HOT_CLASSIC)) != 0:
      attenuation = _classic(attenuation, fetch(lit, worldid, li).attenuation, dist)
    if has_spot:
      cos_theta = wp.dot(-l, hot.dir)
      attenuation = _lobe(attenuation, ltype, cos_theta)
      if (hot.flags & wp.static(HOT_CONED)) != 0:
        attenuation = attenuation * cone_factor(fetch(lit, worldid, li), cos_theta)

  return l, dist, hot.color * (hot.intensity * attenuation)
