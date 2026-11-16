"""Area lights (lighting.area_lights="ltc"): LTC shading and solid-angle shadow sampling.

Shapes (`LightShape`): rect, disk, sphere, tube, of uniform surface radiance
`color * intensity / projected area` (the intensity is the point-light equivalent [cd]).

Unshadowed radiance: linearly transformed cosines (Heitz, Dupuy, Hill, Neubelt 2016) fitted to the
renderer's GGX (`raywarp/data/ltc_ggx.npz`, tools/fit_ltc.py), specular = LTC(M^-1) x (F0
norm + (1 - F0) fresnel) x the multiscatter compensation, diffuse = LTC(identity) x diffuse color
(the exact Lambert form factor; a rough EON lobe's color is its directional albedo, the analytic
albedo ratio over Lambert: `material.make_shading_lobes`), under the coat and fuzz layers
(`material.make_layers`):

* rect: the polygon integral of the transformed corners, clipped to the horizon (exact clip);
* disk: the transformed ellipse's form factor (Heitz & Hill 2017, ltc_code `ltc_disk`), horizon
  clipping by the equivalent sphere (Snyder's closed form instead of the course's table);
* sphere: the disk facing the shading point that subtends the same cone;
* tube: the rect facing the shading point that covers the cylinder's silhouette (2 r x l; the end
  caps are omitted).

Shadows: L = L_LTC x V, V = the fraction of `shadow_samples` rays reaching points on the light,
uniform in solid angle for rects (spherical rectangle sampling, Urena et al. 2013 / pbrt-v4) and
spheres (cone sampling), uniform in area for disks and tubes. The targets: the light's centre
for one sample (a hard shadow, noise-free), else an R2 low-discrepancy set shifted per pixel and
light (Cranley-Patterson rotation; deterministic per pixel and seed): stratified, less noisy than
independent points.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import warp as wp

from raywarp._base.types import LightShape
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.generic import specialize
from raywarp.core.shade.lights import ltc, punctual
from raywarp.core.shade.lights.ltc import Frame, ellipse_integral, quad_integral, to_local
from raywarp.core.shade.materials import brdf, material, texture
from raywarp.core.shade.materials.material import Lobes

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

SHAPE_NONE: Final = int(LightShape.NONE)
SHAPE_RECT: Final = int(LightShape.RECT)
SHAPE_DISK: Final = int(LightShape.DISK)
SHAPE_SPHERE: Final = int(LightShape.SPHERE)
SHAPE_TUBE: Final = int(LightShape.TUBE)
# Shadow rays towards a point on a light end at this fraction of its distance (2^-10 short of it:
# the light's own emitter geometry is not an occluder), at any scene scale (an absolute gap of
# 1 mm cut most of the ray in a millimetre-scale scene).
SHADOW_END: Final = 1.0 - 2.0**-10


@wp.struct
class AreaLight:
  """One area light of one world (world-space frame; `axis_y` = normal x axis_x)."""

  shape: int
  cast_shadow: bool
  center: wp.vec3
  normal: wp.vec3  # emission direction (rect, disk)
  axis_x: wp.vec3  # rect width / tube axis
  axis_y: wp.vec3
  half: wp.vec2  # rect half extents; tube (half length, -)
  radius: float  # disk, sphere, tube
  radiance: wp.vec3
  range: float


@wp.func
def area_of(
  # In:
  light: punctual.Light,
) -> AreaLight:
  """The area form of a packed light (`punctual.Light`, shape != SHAPE_NONE)."""
  a = AreaLight()
  a.shape = light.shape
  a.cast_shadow = light.cast_shadow
  a.center = light.pos
  a.normal = light.dir
  a.axis_x = light.axis_x
  a.axis_y = light.axis_y
  a.half = light.half
  a.radius = wp.max(light.radius, 1.0e-4)
  a.range = light.range
  a.radiance = light.radiance
  return a


@wp.func
def facing_axes(
  # Model:
  light: AreaLight,
  # In:
  point: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """(centre, axis u, axis v) of the disk (sphere) or rect (tube) facing `point`."""
  w = light.center - point
  if light.shape == wp.static(SHAPE_SPHERE):
    d2 = wp.length_sq(w)
    r2 = light.radius * light.radius
    # The disk at the centre that subtends the sphere's cone: r d / sqrt(d^2 - r^2).
    rd = light.radius * wp.sqrt(d2 / wp.max(d2 - r2, 1.0e-4 * d2))
    u, v = texture.orthonormal_basis(wp.normalize(w))
    return light.center, u * rd, v * rd
  # Tube: the silhouette rect, across = perpendicular to the axis and the view of the axis.
  across = wp.cross(light.axis_x, w)
  if wp.length_sq(across) < 1.0e-12:
    t1, _t2 = texture.orthonormal_basis(light.axis_x)
    across = t1
  return light.center, light.axis_x * light.half[0], wp.normalize(across) * light.radius


@wp.struct
class LocalShape:
  """An area light's shape in an LTC frame, relative to the shading point (`local_shape`): a quad's
  four corners (rect, tube), else an ellipse's centre and two axes (`p0`, `p1`, `p2`)."""

  quad: bool
  p0: wp.vec3
  p1: wp.vec3
  p2: wp.vec3
  p3: wp.vec3


@wp.func
def local_shape(
  # Model:
  light: AreaLight,
  # In:
  point: wp.vec3,
  fr: Frame,
) -> LocalShape:
  """The light's shape seen from `point`, in the frame `fr` (shared by the LTC integrals)."""
  s = LocalShape()
  if light.shape == wp.static(SHAPE_RECT) or light.shape == wp.static(SHAPE_TUBE):
    c = light.center
    u = light.axis_x * light.half[0]
    v = light.axis_y * light.half[1]
    if light.shape == wp.static(SHAPE_TUBE):
      c, u, v = facing_axes(light, point)
    q = c - point
    s.quad = True
    s.p0 = to_local(fr, q - u - v)
    s.p1 = to_local(fr, q + u - v)
    s.p2 = to_local(fr, q + u + v)
    s.p3 = to_local(fr, q - u + v)
    return s
  c = light.center
  u = light.axis_x * light.radius
  v = light.axis_y * light.radius
  if light.shape == wp.static(SHAPE_SPHERE):
    c, u, v = facing_axes(light, point)
  s.quad = False
  s.p0 = to_local(fr, c - point)
  s.p1 = to_local(fr, u)
  s.p2 = to_local(fr, v)
  return s


@wp.func
def shape_integral(
  # In:
  s: LocalShape,
  m: wp.mat33,
) -> float:
  """Clamped-cosine integral of a local shape through the LTC M^-1 `m`."""
  if s.quad:
    return quad_integral(m @ s.p0, m @ s.p1, m @ s.p2, m @ s.p3)
  return ellipse_integral(m @ s.p0, m @ s.p1, m @ s.p2)


@wp.func(inline=True)
def above_horizon(
  # Model:
  light: AreaLight,
  # In:
  point: wp.vec3,
  normal: wp.vec3,
) -> bool:
  """Whether any point of the emitter is above the shading point's horizon (plane of `normal`).

  The LTC transform shears the cosine's horizon, so clipping the transformed shape alone can light
  a receiver from an emitter wholly below its own horizon. The shape's support along `normal`
  (its extent from the centre; the light's axes are an orthonormal frame) rejects only emitters
  wholly below or tangent to it: one crossing the horizon keeps its LTC integral.
  """
  x = wp.dot(light.axis_x, normal)
  y = wp.dot(light.axis_y, normal)
  extent = light.radius * wp.length(normal)
  if light.shape == wp.static(SHAPE_RECT):
    extent = wp.abs(x) * light.half[0] + wp.abs(y) * light.half[1]
  elif light.shape == wp.static(SHAPE_DISK):
    extent = light.radius * wp.sqrt(x * x + y * y)
  elif light.shape == wp.static(SHAPE_TUBE):
    radial = normal - light.axis_x * x
    extent = wp.abs(x) * light.half[0] + light.radius * wp.length(radial)
  return wp.dot(light.center - point, normal) + extent > 0.0


@wp.func(inline=True)
def range_window(
  # Model:
  light: AreaLight,
  # In:
  point: wp.vec3,
  normal: wp.vec3,
) -> float:
  """The squared range window of the light at `point` (the punctual lights' window at the distance
  to the centre: light-grid consistent; range 0: 1, unbounded); 0 behind a one-sided emitter
  (rect, disk) or when the whole emitter is below the receiver's shading horizon."""
  if light.shape == wp.static(SHAPE_RECT) or light.shape == wp.static(SHAPE_DISK):
    if wp.dot(point - light.center, light.normal) <= 0.0:
      return 0.0
  if not above_horizon(light, point, normal):
    return 0.0
  if light.range <= 0.0:
    return 1.0  # range 0: no window (unbounded reach), as a punctual light (`punctual.eval_light`)
  d2 = wp.length_sq(light.center - point)
  factor = d2 / (light.range * light.range)
  window = wp.clamp(1.0 - factor * factor, 0.0, 1.0)
  return window * window


type EvalAreaLight = Callable[[AreaLight, Frame, Lobes, wp.vec3], wp.vec3]


@specialize
def make_eval_area_light(
  *,
  clearcoat: bool,
  anisotropy: bool = False,
  fuzz: bool = False,
  coat_normal: bool = False,
  coat_layer: bool = False,
  bound: int = 0,
):
  """eval_area_light(light, fr, lobes, point) -> unshadowed radiance of an area light at `point`
  towards the frame's view: the base lobes' LTC integrals (specular, diffuse) under the coat's LTC
  integral and the fuzz lobe (`material.make_layers`: the layering law of every light).

  The flags are their Features (each lobe compiled in only with its flag). Approximations of LTC
  integrals fitted to isotropic GGX: the anisotropic specular lobe's isotropic LTC (alpha,
  OpenPBR's mean roughness) about its bent normal (`ltc.make_frame_func`), the coat's LTC about
  its own normal (the base under it scaled at the view: `material.coat_base_scale` at the coat's
  NoV), the fuzz lobe evaluated toward the light's centre times its cosine form factor (pi FF L:
  exact for small lights; the Charlie lobe is broad). `bound` (Features.bound): the coat and layers
  class bits read the build word's material copy (`Lobes.bits`).
  """
  layers = material.make_layers(clearcoat=clearcoat, layered_coat=coat_normal or coat_layer, fuzz=fuzz, bound=bound)
  gate = toggles.bound_on(bound)
  field = toggles.bound_on(bound, high=True)  # the layered lobes' field bits (`toggles.LAYER_*`)

  @wp.func
  def eval_area_light(
    # Model:
    light: AreaLight,
    # In:
    fr: Frame,
    lobes: Lobes,
    point: wp.vec3,
  ) -> wp.vec3:
    """Unshadowed radiance of an area light at `point` towards the frame's view."""
    window = range_window(light, point, fr.n)
    if window <= 0.0:
      return wp.vec3(0.0)
    shape = local_shape(light, point, fr)
    spec_shape = shape
    layered = False
    if wp.static(anisotropy or fuzz or coat_normal):
      layered = gate(lobes.bits, wp.static(toggles.CLASS_LAYERS))
    if wp.static(anisotropy):
      if layered and field(lobes.bits2, wp.static(toggles.LAYER_ANISOTROPY)) and lobes.anisotropy > 0.0:
        spec_shape = local_shape(light, point, ltc.with_axes(fr, fr.spec_t1, fr.spec_t2, fr.spec_n))
    out = fr.spec * shape_integral(spec_shape, fr.minv)
    diffuse_ff = float(0.0)
    # No diffuse lobe (metals, specular-only translucent layers): no diffuse form factor.
    if wp.length_sq(lobes.diffuse_color) > 0.0 or (wp.static(fuzz) and layered):
      diffuse_ff = shape_integral(shape, wp.identity(3, dtype=float))
      out = lobes.diffuse_color * diffuse_ff + out
    radiance = light.radiance * window
    coat = wp.vec3(0.0)
    if wp.static(clearcoat):
      if gate(lobes.bits, wp.static(toggles.CLASS_COAT)) and lobes.coat > 0.0:
        coat_shape = shape
        if wp.static(coat_normal):
          if layered:
            coat_shape = local_shape(light, point, ltc.with_axes(fr, fr.coat_t1, fr.coat_t2, fr.coat_n))
        coat = radiance * (fr.coat_spec * shape_integral(coat_shape, fr.coat_minv))
    sheen = wp.vec3(0.0)
    if wp.static(fuzz):
      if layered and lobes.fuzz_weight > 0.0:
        sheen = wp.cw_mul(fuzz_toward(lobes, fr, wp.normalize(light.center - point)), radiance) * (wp.pi * diffuse_ff)
    return layers(lobes, wp.cw_mul(out, radiance), coat, lobes.coat_nov, sheen)

  return eval_area_light


@wp.func
def fuzz_toward(
  # In:
  lobes: Lobes,
  fr: Frame,
  l: wp.vec3,
) -> wp.vec3:
  """The fuzz lobe F color f_charlie(l, v) (normalised: `Lobes.fuzz_norm`; no cosine) for light
  from `l` and the frame's view (`Frame.view`)."""
  nol = wp.dot(fr.n, l)
  if nol <= 0.0:
    return wp.vec3(0.0)
  nov = wp.max(wp.dot(fr.n, fr.view), 1.0e-4)
  h = wp.normalize(l + fr.view)
  d = brdf.d_charlie(lobes.fuzz_alpha, wp.max(wp.dot(fr.n, h), 0.0))
  return lobes.fuzz_color * (lobes.fuzz_weight * lobes.fuzz_norm * d * brdf.v_neubelt(nov, nol))


# Shadow sampling -------------------------------------------------------------------------------


@wp.func
def _angle_between(
  # In:
  a: wp.vec3,
  b: wp.vec3,
) -> float:
  """Angle between unit vectors a and b (stable near 0 and pi)."""
  if wp.dot(a, b) < 0.0:
    return wp.pi - 2.0 * wp.asin(wp.min(wp.length(a + b) * 0.5, 1.0))
  return 2.0 * wp.asin(wp.min(wp.length(b - a) * 0.5, 1.0))


@wp.struct
class SphericalRect:
  """A rect s + [0, 1] ex + [0, 1] ey seen from a point p, set up for solid-angle sampling
  (`spherical_rect`; pbrt-v4's SphericalRectangle): the per-point part shared by its samples."""

  p: wp.vec3
  s: wp.vec3
  ex: wp.vec3
  ey: wp.vec3
  rx: wp.vec3
  ry: wp.vec3
  rz: wp.vec3
  x0: float
  x1: float
  y0: float
  y1: float
  z0: float
  b0: float
  b1: float
  g01: float  # g0 + g1 - 2 pi
  g23: float  # g2 + g3
  tiny: bool  # solid angle < 1e-3: sampled uniformly in area


@wp.func
def spherical_rect(
  # In:
  p: wp.vec3,
  s: wp.vec3,
  ex: wp.vec3,
  ey: wp.vec3,
) -> SphericalRect:
  """The sampling setup of the rect s + [0, 1] ex + [0, 1] ey seen from p."""
  r = SphericalRect()
  r.p = p
  r.s = s
  r.ex = ex
  r.ey = ey
  exl = wp.length(ex)
  eyl = wp.length(ey)
  rx = ex / exl
  ry = ey / eyl
  rz = wp.cross(rx, ry)
  dl = s - p
  x0 = wp.dot(dl, rx)
  y0 = wp.dot(dl, ry)
  z0 = wp.dot(dl, rz)
  if z0 > 0.0:
    rz = -rz
    z0 = -z0
  x1 = x0 + exl
  y1 = y0 + eyl
  v00 = wp.vec3(x0, y0, z0)
  v01 = wp.vec3(x0, y1, z0)
  v10 = wp.vec3(x1, y0, z0)
  v11 = wp.vec3(x1, y1, z0)
  n0 = wp.normalize(wp.cross(v00, v10))
  n1 = wp.normalize(wp.cross(v10, v11))
  n2 = wp.normalize(wp.cross(v11, v01))
  n3 = wp.normalize(wp.cross(v01, v00))
  g0 = _angle_between(-n0, n1)
  g1 = _angle_between(-n1, n2)
  g2 = _angle_between(-n2, n3)
  g3 = _angle_between(-n3, n0)
  solid_angle = g0 + g1 + g2 + g3 - 2.0 * wp.pi
  r.tiny = solid_angle < 1.0e-3
  r.rx = rx
  r.ry = ry
  r.rz = rz
  r.x0 = x0
  r.x1 = x1
  r.y0 = y0
  r.y1 = y1
  r.z0 = z0
  r.b0 = n0[2]
  r.b1 = n2[2]
  r.g01 = g0 + g1 - 2.0 * wp.pi
  r.g23 = g2 + g3
  return r


@wp.func
def sample_rect(
  # In:
  r: SphericalRect,
  u: wp.vec2,
) -> wp.vec3:
  """Point `u` of a set-up rect, uniform in solid angle seen from its point (pbrt-v4)."""
  if r.tiny:
    return r.s + r.ex * u[0] + r.ey * u[1]
  b0 = r.b0
  b1 = r.b1
  z0 = r.z0
  au = u[0] * r.g01 + (u[0] - 1.0) * r.g23
  fu = (wp.cos(au) * b0 - b1) / wp.sin(au)
  cu = wp.where(fu >= 0.0, 1.0, -1.0) / wp.sqrt(fu * fu + b0 * b0)
  cu = wp.clamp(cu, -0.99999994, 0.99999994)
  xu = -(cu * z0) / wp.sqrt(wp.max(1.0 - cu * cu, 0.0))
  xu = wp.clamp(xu, r.x0, r.x1)
  dd = wp.sqrt(xu * xu + z0 * z0)
  h0 = r.y0 / wp.sqrt(dd * dd + r.y0 * r.y0)
  h1 = r.y1 / wp.sqrt(dd * dd + r.y1 * r.y1)
  hv = h0 + u[1] * (h1 - h0)
  hvsq = hv * hv
  yv = wp.where(hvsq < 1.0 - 1.0e-6, (hv * dd) / wp.sqrt(wp.max(1.0 - hvsq, 1.0e-12)), r.y1)
  return r.p + r.rx * xu + r.ry * yv + r.rz * z0


@wp.func
def rect_of(
  # Model:
  light: AreaLight,
  # In:
  origin: wp.vec3,
) -> SphericalRect:
  """The sampling setup of a rect light seen from `origin` (`sample_rect`; shared by the samples of
  one shading point)."""
  ex = light.axis_x * light.half[0]
  ey = light.axis_y * light.half[1]
  return spherical_rect(origin, light.center - ex - ey, ex * 2.0, ey * 2.0)


@wp.func
def sample_curved(
  # Model:
  light: AreaLight,
  # In:
  origin: wp.vec3,
  u: wp.vec2,
) -> wp.vec3:
  """A point of a sphere, disk or tube light for a shadow ray from `origin` (a rect's:
  `sample_rect`)."""
  if light.shape == wp.static(SHAPE_SPHERE):
    w = light.center - origin
    d2 = wp.length_sq(w)
    r2 = light.radius * light.radius
    if d2 <= r2:
      return light.center
    d = wp.sqrt(d2)
    w = w / d
    # Uniform in the cone of the sphere, then the near intersection with it.
    cos_max = wp.sqrt(wp.max(1.0 - r2 / d2, 0.0))
    cos_t = 1.0 - u[0] * (1.0 - cos_max)
    sin_t = wp.sqrt(wp.max(1.0 - cos_t * cos_t, 0.0))
    phi = wp.static(2.0 * np.pi) * u[1]
    t1, t2 = texture.orthonormal_basis(w)
    dirn = w * cos_t + (t1 * wp.cos(phi) + t2 * wp.sin(phi)) * sin_t
    proj = d * cos_t
    t = proj - wp.sqrt(wp.max(r2 - d2 + proj * proj, 0.0))
    return origin + dirn * t
  if light.shape == wp.static(SHAPE_DISK):
    r = light.radius * wp.sqrt(u[0])
    phi = wp.static(2.0 * np.pi) * u[1]
    return light.center + (light.axis_x * wp.cos(phi) + light.axis_y * wp.sin(phi)) * r
  c, ax, ay = facing_axes(light, origin)
  return c + ax * (2.0 * u[0] - 1.0) + ay * (2.0 * u[1] - 1.0)


# R2 low-discrepancy sequence (Roberts 2018): point k = frac(1/2 + k (1/g, 1/g^2)), g^3 = g + 1.
_R2: Final = (0.7548776662466927, 0.5698402909980532)


@wp.func
def fixed_rand2(
  # In:
  k: int,
  shift: wp.vec2,
) -> wp.vec2:
  """Point k of the fixed shadow pattern: the R2 set, shifted by `shift`.

  The toroidal shift (Cranley-Patterson rotation) is per pixel and light (`shadow_rand2`'s first
  pair): every pixel sees the same low-discrepancy set at another offset, so a penumbra is the
  set's stratified average rather than `nsample` stacked copies of one hard shadow.
  """
  return wp.vec2(
    wp.mod(0.5 + float(k) * wp.static(_R2[0]) + shift[0], 1.0),
    wp.mod(0.5 + float(k) * wp.static(_R2[1]) + shift[1], 1.0),
  )


@wp.func
def shadow_rand2(
  # In:
  seed: int,
  stream: int,
  nlight: int,
  li: int,
  nsample: int,
  k: int,
) -> wp.vec2:
  """The megakernel's counter-based shadow random pair (same streams as punctual lights)."""
  counter = ((stream * nlight + li) * nsample + k) * 2
  return wp.vec2(wp.randf(wp.rand_init(seed, counter)), wp.randf(wp.rand_init(seed, counter + 1)))


@wp.func
def shape_points(
  # Model:
  a: AreaLight,
  # In:
  k: int,
) -> wp.vec3:
  """Point k (0-4) of a light for light-grid visibility: the centre, then four extremes."""
  if k == 0 or a.shape == wp.static(SHAPE_NONE):
    return a.center
  sx = wp.where(k == 1 or k == 4, 1.0, -1.0)
  sy = wp.where(k == 1 or k == 2, 1.0, -1.0)
  if a.shape == wp.static(SHAPE_RECT):
    return a.center + a.axis_x * (sx * a.half[0]) + a.axis_y * (sy * a.half[1])
  if a.shape == wp.static(SHAPE_TUBE):
    return a.center + a.axis_x * (sx * a.half[0])
  # Disk: the four ends of two diameters; sphere: +-x, +-z.
  sign = wp.where(k % 2 == 1, 1.0, -1.0)
  if a.shape == wp.static(SHAPE_DISK):
    return a.center + wp.where(k <= 2, a.axis_x, a.axis_y) * (sign * a.radius)
  return a.center + wp.where(k <= 2, wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0)) * (sign * a.radius)
