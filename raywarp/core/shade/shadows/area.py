"""Soft area-light shadows by a pass after the render (`shadows.area_rays` > 0; off by default).

The render shades every area light with one shadow ray to its centre (`lights.evaluators`): a large
light close to a blocker casts a hard, magnified umbra. This pass softens those shadows without
any code in the render kernels: their modules are those of the setting off, the render only
writes two more outputs (`RenderSettings.written_outputs`: position, normal) and its HDR image
goes through the deferred display.

Per pixel of the first hit (`make_pixel`):
- the camera ray to the recorded point is traced again and its surface shaded as the render shades
  it (`evaluate.make_eval_surface`, `material.lobes_of`, the light loop's evaluators
  `lighting.make_evaluators`): L, each light's unshadowed radiance towards the camera (every
  lobe and layer: diffuse, specular, coat, fuzz), and the rays' origin (`Surface.origin`);
- for each area light that casts shadows, subtends an angular radius of at least MIN_ANGLE and
  brings at least 1 / `shadows.max_casters` of these lights' radiance there (at most that many
  lights; the others keep the hard shadow; the correction fades in up to twice that angle and
  that share, so that no seam marks where a light starts to count; a pixel where no light counts
  traces nothing):
  - the render's own visibility, the shadow ray from that origin to the light's centre (`hard`);
  - `area_rays` shadow rays to points of the light, uniform in solid angle (rects; area-uniform
    for disks and tubes, `area.sample_rect` / `sample_curved`), each weighted by its
    cosine at the receiver (`soft`: the cosine-weighted visible fraction). The points are
    interleaved (Keller & Heidrich 2001): the pixels of a TILE x TILE block take consecutive runs
    of one R2 sequence (`sample_index`, shifted per world and light), so every TILE x TILE window
    of the filter holds the whole stratified set of TILE^2 x `area_rays` points.

Then `apply_kernel`, per camera: where a pixel's 5 x 5 window holds a partly visible light
(`noisy`), its soft sum is its unshadowed sum times the window's visible ratio (sum of soft over
sum of unshadowed over the window's pixels on the same surface: normals within FILTER_COS,
offsets within FILTER_SLOPE of its tangent plane), and the HDR radiance gains soft - hard. The
visibility is filtered, not the radiance: textures and highlights stay sharp. Where the pass'
centre ray differs from the render's (a light the render did not trace under its caster budget)
the pixel takes the soft light minus a visibility the render did not apply: those are lights the
point barely sees.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp.core.kernels import lighting
from raywarp.core.kernels.abi import rays as trace
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.output_layout import OUT_HDR, OUT_HDR_SQRT, OUT_NORMAL
from raywarp.core.kernels.abi.queries import Queries, make_queries
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.abi.structs import Cameras, EnvironmentArrays, OutputArrays, skipped, vec3hs, vec3s
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.kernels.abi.toggles import out_half, out_on
from raywarp.core.kernels.display import sqrt16, store_hdr
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.pipeline.platform.device import launch_bands
from raywarp.core.post.grading import color
from raywarp.core.shade.lights import area, punctual
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials import evaluate, material
from raywarp.core.shade.materials.material import MaterialTables, material_type
from raywarp.core.shade.shadows.visibility import ShadowPass, make_shadow_pass
from raywarp.core.trace.accel import query

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Smallest angular radius [rad] of a light (its extent over its distance) the pass softens: the
# penumbra of a blocker at height h above the receiver is about 2 x angle x h wide, so below
# 0.05 rad it is under a tenth of the blocker's height, as hard as the centre ray's.
MIN_ANGLE: Final = 0.05
# The filter: a (2 FILTER_RADIUS + 1)^2 neighbourhood; a neighbour counts where its normal is
# within acos(FILTER_COS) of the pixel's and its offset within FILTER_SLOPE (tangent of the angle)
# of the pixel's tangent plane: the same surface, never across a silhouette or crease.
FILTER_RADIUS: Final = 2
FILTER_COS: Final = 0.9
FILTER_SLOPE: Final = 0.1
# The interleaving's block side (`sample_index`): the filter's window, so that every window holds
# each pixel class once.
TILE: Final = 2 * FILTER_RADIUS + 1
# A pixel's state (`AreaLaunch.state`): no light the pass softens (NONE), each such light fully
# visible or fully hidden from its rays (EXACT), one partly visible (PARTIAL), one none of whose
# rays left above the horizon (UNKNOWN: no estimate, its neighbours' is taken). The filter runs
# at every pixel whose window holds a PARTIAL or UNKNOWN pixel: its EXACT pixels hold too few
# points for that to be exact there.
NONE: Final = 0
EXACT: Final = 1
PARTIAL: Final = 2
UNKNOWN: Final = 3
# The random stream of the lights' shifts (`light_shift`).
_SHIFT_STREAM: Final = 0xA5EA
# The camera ray traced again ends this far past the recorded point (relative: the point's float
# rounding), so that it finds the render's surface.
_REACH: Final = 1.0 + 2.0**-10
# Fewest pixels per CPU band of the pass' launches (`device.launch_bands`; one launch on a GPU).
BAND: Final = 1024


@wp.struct
class AreaLaunch:
  """The pass' launch: its per-pixel sums and how many rays per light.

  Attributes:
    soft: (nworld, npix) the soft sum of the pixel's lights, L x fade x the cosine-weighted
      visible fraction, as the float16 of its square root (`display.sqrt16`: any light unit's
      radiance in range).
    hard: (nworld, npix) the hard sum, L x fade x the centre ray's visibility (square root,
      float16).
    unshadowed: (nworld, npix) the sum of L x fade (square root, float16): the filter's
      denominator.
    state: (nworld, npix) the pixel's state (NONE, EXACT, PARTIAL, UNKNOWN).
    rays: shadow rays per light and pixel (`shadows.area_rays`).
    casters: at most this many lights per pixel (`shadows.max_casters`; 0: every light): those
      bringing at least 1 / casters of the softened lights' radiance.
    npix: pixels per world, every camera's (`make_pixel`'s launch: nworld x npix units).
  """

  soft: wp.array2d[wp.vec3h]
  hard: wp.array2d[wp.vec3h]
  unshadowed: wp.array2d[wp.vec3h]
  state: wp.array2d[wp.uint8]
  rays: int
  casters: int
  npix: int


@wp.func
def camera_of(
  # Model:
  cams: Cameras,
  # In:
  adr: int,
) -> wp.vec3i:
  """(camera, column, row) of pixel `adr` (its index in the output buffers, `Lens.adr`)."""
  cam = int(0)
  for c in range(1, cams.lens.shape[0]):
    if cams.lens[c].adr <= adr:
      cam = c
  i = adr - cams.lens[cam].adr
  w = cams.camera_res[cam][0]
  y = i // w
  return wp.vec3i(cam, i - y * w, y)


@wp.func
def sample_index(
  # In:
  pixel: wp.vec3i,
  rays: int,
) -> int:
  """The index in the light's R2 sequence of the first point of `pixel` (`camera_of`): its class
  in its camera image's TILE x TILE blocks times the rays per pixel (`TILE`)."""
  return ((pixel[2] % wp.static(TILE)) * wp.static(TILE) + pixel[1] % wp.static(TILE)) * rays


@wp.func
def light_shift(
  # In:
  worldid: int,
  li: int,
) -> wp.vec2:
  """The shift of light `li`'s R2 sequence in world `worldid` (Cranley-Patterson rotation: the
  same for every pixel and frame)."""
  state = wp.rand_init(wp.static(_SHIFT_STREAM) + worldid, li)
  return wp.vec2(wp.randf(state), wp.randf(state))


@wp.func
def softness(
  # In:
  light: punctual.Light,
  origin: wp.vec3,
) -> float:
  """How much the pass softens `light` seen from `origin` (0: not at all): an active,
  shadow-casting area light (no IES profile) faded in from MIN_ANGLE to twice it of angular
  radius (its radius: a rect's half diagonal, a disk's, sphere's or tube's half length), so that
  no seam marks where a light starts to count."""
  if not light.active or not light.cast_shadow or light.shape == wp.static(area.SHAPE_NONE) or light.ies != 0:
    return 0.0
  size = wp.max(light.radius, 1.0e-4)
  if light.shape == wp.static(area.SHAPE_RECT):
    size = wp.length(light.half)
  elif light.shape == wp.static(area.SHAPE_TUBE):
    size = wp.max(light.half[0], light.radius)
  dist = wp.length(light.pos - origin)
  return wp.clamp(size / wp.max(wp.static(MIN_ANGLE) * dist, 1.0e-30) - 1.0, 0.0, 1.0)


@wp.func
def light_point(
  # In:
  a: area.AreaLight,
  rect: area.SphericalRect,
  origin: wp.vec3,
  u: wp.vec2,
) -> wp.vec3:
  """Point `u` of the light seen from `origin` (a rect through its set-up `rect`)."""
  if a.shape == wp.static(area.SHAPE_RECT):
    return area.sample_rect(rect, u)
  return area.sample_curved(a, origin, u)


@wp.func
def read_normal(
  # In:
  out: OutputArrays,
  worldid: int,
  adr: int,
) -> wp.vec3:
  """The pixel's normal output (zero on a miss)."""
  if out_half(out.half, wp.static(OUT_NORMAL)):
    return wp.vec3(vec3hs(out, out.table.normal16)[worldid, adr])
  return vec3s(out, out.table.normal)[worldid, adr]


# visibility(geo, worldid, light, origin, normal, first, shift, rays) -> (hard, soft, cosine weight)
# of one area light (`make_light_visibility`).
type LightVisibility = Callable[[GeometryFields, int, area.AreaLight, wp.vec3, wp.vec3, int, wp.vec2, int], wp.vec3]


@specialize
def make_light_visibility(shadow_pass: ShadowPass):
  """visibility(geo, worldid, light, origin, normal, first, shift, rays): the centre ray's
  visibility of an area light (the render's hard shadow), the cosine-weighted visible fraction of
  `rays` points of it (points `first` .. of its R2 sequence shifted by `shift`; the centre ray's
  without any above the horizon) and the rays' summed cosine (0: no point above the horizon)."""

  @wp.func
  @device_generic
  def visibility[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    a: area.AreaLight,
    origin: wp.vec3,
    normal: wp.vec3,
    first: int,
    shift: wp.vec2,
    rays: int,
  ) -> wp.vec3:
    """(hard, soft, cosine weight) of light `a` from `origin` (see `make_light_visibility`)."""
    to_centre = a.center - origin
    dist = wp.length(to_centre)
    vh = shadow_pass(geo, worldid, origin, to_centre / dist, dist * wp.static(area.SHADOW_END))
    rect = area.SphericalRect()
    if a.shape == wp.static(area.SHAPE_RECT):
      rect = area.rect_of(a, origin)
    weight = float(0.0)
    seen = float(0.0)
    for k in range(rays):
      to_target = light_point(a, rect, origin, area.fixed_rand2(first + k, shift)) - origin
      d = wp.length(to_target)
      if d > 0.0:
        ray = to_target / d
        cos = wp.dot(ray, normal)
        if cos > 0.0:
          weight += cos
          seen += cos * shadow_pass(geo, worldid, origin, ray, d * wp.static(area.SHADOW_END))
    return wp.vec3(vh, wp.where(weight > 0.0, seen / wp.max(weight, 1.0e-20), vh), weight)

  return visibility


@wp.func
def write_sums(
  # Out:
  launch: AreaLaunch,
  # In:
  worldid: int,
  adr: int,
  soft: wp.vec3,
  hard: wp.vec3,
  total: wp.vec3,
  state: int,
):
  """Stores pixel `adr`'s sums (square roots, float16) and state."""
  launch.soft[worldid, adr] = sqrt16(soft)
  launch.hard[worldid, adr] = sqrt16(hard)
  launch.unshadowed[worldid, adr] = sqrt16(total)
  launch.state[worldid, adr] = wp.uint8(state)


@wp.func
def state_of(
  # In:
  state: int,
  weight: float,
  vs: float,
) -> int:
  """A pixel's state (`NONE` ...) with one more light of cosine weight `weight` and soft
  visibility `vs`."""
  if weight == 0.0:
    return wp.static(UNKNOWN)
  if vs > 0.0 and vs < 1.0:
    return wp.max(state, wp.static(PARTIAL))
  return wp.max(state, wp.static(EXACT))


@specialize
def make_pixel(f: Features, q: Queries):
  """area_pixel(geo, mats, lit, cams, env, out, launch, worldid, adr): one pixel of the pass (the
  module doc) for build `f`, with queries `q` (the render's: its camera ray's surface, its shadow
  rays through translucent casters as it traces them, its light evaluators)."""
  eval_surface = evaluate.make_eval_surface(f)
  shading_lobes = material.lobes_of(f)
  ev = lighting.make_evaluators(f, q)
  context, unshadowed = ev.context, ev.unshadowed
  visibility = make_light_visibility(make_shadow_pass(f, q))
  closest = q.closest

  @wp.func
  @device_generic
  def area_pixel[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    cams: Cameras,
    env: EnvironmentArrays,
    # Data in:
    out: OutputArrays,
    # Out:
    launch: AreaLaunch,
    # In:
    worldid: int,
    adr: int,
  ):
    """The soft, hard and unshadowed sums of pixel `adr` of world `worldid`."""
    if skipped(out, worldid):
      return
    soft = wp.vec3(0.0)
    hard = wp.vec3(0.0)
    total = wp.vec3(0.0)
    state = int(wp.static(NONE))  # pyrefly: ignore[unnecessary-type-conversion]  # a variable, not a constant
    pixel = camera_of(cams, adr)
    eye = cams.camera_pos[worldid, pixel[0]]
    to_point = vec3s(out, out.table.position)[worldid, adr] - eye
    seen_dist = wp.length(to_point)
    hit = trace.no_hit(0.0)
    direction = wp.vec3(0.0, 0.0, 1.0)
    near = float(0.0)  # how much the pass softens some light, seen from the recorded point
    if wp.length_sq(read_normal(out, worldid, adr)) > 0.0 and seen_dist > 0.0:  # a miss: the zero normal
      for li in range(lit.nlight):
        near = wp.max(near, softness(punctual.fetch(lit, worldid, li), eye + to_point))
    if near > 0.0:
      direction = to_point / seen_dist
      hit = closest(geo, worldid, eye, direction, seen_dist * wp.static(_REACH))
    if hit.instance_id == -1:
      write_sums(launch, worldid, adr, soft, hard, total, state)
      return
    point = eye + direction * hit.dist
    fovy = batched(cams.camera_fovy, worldid, pixel[0])
    footprint = hit.dist * 2.0 * wp.tan(fovy * wp.static(math.pi / 360.0)) / float(cams.camera_res[pixel[0]][1])
    surface, normal = eval_surface(geo, mats, worldid, hit, point, direction, footprint)
    view = -direction
    nov = wp.max(wp.dot(normal, view), 1.0e-4)
    lobes = shading_lobes(env.dfg_lut, surface, nov)
    ctx = context(lit, lobes, normal, view, nov)
    origin = surface.origin
    # The lights' total radiance (luminance) first: only those bringing at least 1 / `casters` of
    # it are traced (`AreaLaunch.casters`; the others keep the render's shadow).
    lum = float(0.0)
    for li in range(lit.nlight):
      light = punctual.fetch(lit, worldid, li)
      if softness(light, origin) > 0.0:
        c, _casts = unshadowed(lit, light, lobes, ctx, point, normal, view, nov)  # pyrefly: ignore[bad-argument-type]  # the evaluators' own context
        lum += color.luminance(c)
    least = wp.where(launch.casters > 0, lum / float(wp.max(launch.casters, 1)), 0.0)
    first = sample_index(pixel, launch.rays)
    for li in range(lit.nlight):
      light = punctual.fetch(lit, worldid, li)
      angle_fade = softness(light, origin)
      if angle_fade <= 0.0:
        continue
      c, _casts = unshadowed(lit, light, lobes, ctx, point, normal, view, nov)  # pyrefly: ignore[bad-argument-type]  # the evaluators' own context
      share = color.luminance(c)
      if share <= 0.0 or share < least:
        continue
      v = visibility(geo, worldid, area.area_of(light), origin, normal, first, light_shift(worldid, li), launch.rays)
      # Faded in over [least, 2 least]: no seam where the traced set changes across the image.
      fade = angle_fade * wp.where(least > 0.0, wp.clamp(share / wp.max(least, 1.0e-30) - 1.0, 0.0, 1.0), 1.0)
      # The faded share only: the render keeps (1 - fade) of the light with its hard shadow.
      soft += c * (v[1] * fade)
      hard += c * (v[0] * fade)
      total += c * fade
      state = state_of(state, v[2], v[1])
    write_sums(launch, worldid, adr, soft, hard, total, state)

  return area_pixel


@wp.func
def unsqrt(
  # In:
  s: wp.vec3h,
) -> wp.vec3:
  """The sum a pixel stores as the float16 of its square root (`AreaLaunch.soft`, `hard`)."""
  v = wp.vec3(s)
  return wp.cw_mul(v, v)


@wp.func
def read_hdr(
  # In:
  out: OutputArrays,
  worldid: int,
  adr: int,
) -> wp.vec3:
  """The pixel's HDR radiance as the word stores it (float32, float16 or float16 sqrt)."""
  if out_on(out.bits, wp.static(OUT_HDR_SQRT)):
    s = wp.vec3(vec3hs(out, out.table.hdr16)[worldid, adr])
    return wp.cw_mul(s, s)
  if out_half(out.half, wp.static(OUT_HDR)):
    return wp.vec3(vec3hs(out, out.table.hdr16)[worldid, adr])
  return vec3s(out, out.table.hdr)[worldid, adr]


@wp.func
def noisy(
  # In:
  launch: AreaLaunch,
  worldid: int,
  cam_adr: int,
  x: int,
  y: int,
  width: int,
  height: int,
) -> bool:
  """Whether pixel (x, y)'s filter window holds a PARTIAL or UNKNOWN pixel (its estimate is then
  the window's)."""
  for j in range(-wp.static(FILTER_RADIUS), wp.static(FILTER_RADIUS) + 1):
    for i in range(-wp.static(FILTER_RADIUS), wp.static(FILTER_RADIUS) + 1):
      xx = x + i
      yy = y + j
      if xx >= 0 and xx < width and yy >= 0 and yy < height:
        if int(launch.state[worldid, cam_adr + yy * width + xx]) >= wp.static(PARTIAL):
          return True
  return False


@wp.func
def visible_ratio(
  # In:
  out: OutputArrays,
  launch: AreaLaunch,
  worldid: int,
  cam_adr: int,
  x: int,
  y: int,
  width: int,
  height: int,
) -> float:
  """The visible share of the softened lights over pixel (x, y)'s window: the luminance of the
  soft sums over that of the unshadowed sums, of the window's pixels on its surface with an
  estimate (EXACT or PARTIAL; the module doc); -1 without any."""
  adr = cam_adr + y * width + x
  n = wp.normalize(read_normal(out, worldid, adr))
  p = vec3s(out, out.table.position)[worldid, adr]
  seen = float(0.0)
  total = float(0.0)
  for j in range(-wp.static(FILTER_RADIUS), wp.static(FILTER_RADIUS) + 1):
    for i in range(-wp.static(FILTER_RADIUS), wp.static(FILTER_RADIUS) + 1):
      xx = x + i
      yy = y + j
      if xx >= 0 and xx < width and yy >= 0 and yy < height:
        b = cam_adr + yy * width + xx
        state = int(launch.state[worldid, b])
        if state == wp.static(EXACT) or state == wp.static(PARTIAL):
          nb = read_normal(out, worldid, b)
          offset = vec3s(out, out.table.position)[worldid, b] - p
          same = wp.dot(nb, n) >= wp.static(FILTER_COS) * wp.length(nb)
          if same and wp.abs(wp.dot(offset, n)) <= wp.static(FILTER_SLOPE) * wp.length(offset):
            seen += color.luminance(unsqrt(launch.soft[worldid, b]))
            total += color.luminance(unsqrt(launch.unshadowed[worldid, b]))
  return wp.where(total > 0.0, seen / wp.max(total, 1.0e-30), -1.0)


@wp.kernel(module="unique", enable_backward=False)
def apply_kernel(
  # In:
  launch: AreaLaunch,
  cam_adr: int,
  width: int,
  height: int,
  first: int,
  # Out:
  out: OutputArrays,
):
  """Pixel `first` + j of the camera at `cam_adr` (width x height) of world w: its HDR radiance
  gains soft - hard, the soft sum its unshadowed sum times the window's visible ratio where its
  window is noisy (the module doc)."""
  w, j = wp.tid()
  i = first + j
  adr = cam_adr + i
  if skipped(out, w) or int(launch.state[w, adr]) == wp.static(NONE):
    return
  y = i // width
  x = i - y * width
  soft = unsqrt(launch.soft[w, adr])
  if noisy(launch, w, cam_adr, x, y, width, height):
    ratio = visible_ratio(out, launch, w, cam_adr, x, y, width, height)
    if ratio >= 0.0:
      soft = unsqrt(launch.unshadowed[w, adr]) * ratio
  delta = soft - unsqrt(launch.hard[w, adr])
  if wp.length_sq(delta) == 0.0:
    return
  store_hdr(out, w, adr, wp.max(read_hdr(out, w, adr) + delta, wp.vec3(0.0)))


@functools.cache
def software_kernel(f: Features):
  """The pass' tracing kernel on the software traversal for build `f` (its words in the module,
  as the megakernel's; cached): thread (w, j) is pixel `first` + j of world w (`make_pixel`), its
  queries the megakernel's."""
  area_pixel = make_pixel(f, make_queries(f, query.software_tracer(f, cull=f.backface_culling)))

  @wp.kernel(module="unique", enable_backward=False, module_options=kernel_structs.module_options(f))
  @device_generic
  @concrete(
    geo=trace.software_geometry_type(live_uv=f.live_material_uv),
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  def trace_area_shadows(
    # Model:
    geo: trace.SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    cams: Cameras,
    env: EnvironmentArrays,
    # Data in:
    out: OutputArrays,
    # Out:
    launch: AreaLaunch,
    # In:
    first: int,
  ):
    """Pixel `first` + j of world w."""
    w, j = wp.tid()
    area_pixel(geo, mats, lit, cams, env, out, launch, w, first + j)

  return trace_area_shadows


# trace(launch): the pass' tracing launch on a renderer's trace backend
# (`TraceBackend.area_shadows`).
type Trace = Callable[[AreaLaunch], None]


@final
class AreaShadows:
  """The pass of a Renderer (built while `shadows.area_rays` > 0): its per-pixel sums and the
  frame's launches (the module doc)."""

  def __init__(self, device: wp.Device, nworld: int, sizes: Sequence[tuple[int, int]], adr: Sequence[int]) -> None:
    """The sums of `nworld` worlds of cameras of `sizes` ((W, H) each) at buffer addresses
    `adr`."""
    self.device, self.nworld = device, nworld
    self.cameras = list(zip(adr, sizes, strict=True))
    npix = sum(w * h for w, h in sizes)
    self.launch = AreaLaunch()
    self.launch.soft = wp.zeros((nworld, npix), dtype=wp.vec3h, device=device)
    self.launch.hard = wp.zeros((nworld, npix), dtype=wp.vec3h, device=device)
    self.launch.unshadowed = wp.zeros((nworld, npix), dtype=wp.vec3h, device=device)
    self.launch.state = wp.zeros((nworld, npix), dtype=wp.uint8, device=device)
    self.launch.npix = npix

  def run(self, rays: int, casters: int, trace_pass: Trace, out: OutputArrays) -> None:
    """The frame's pass over the outputs `out` (the render's): `rays` per light and pixel for at
    most `casters` lights (0: every one) traced by `trace_pass`, then each camera's filter and HDR
    update."""
    self.launch.rays = rays
    self.launch.casters = casters
    trace_pass(self.launch)
    module = apply_kernel.module
    for cam_adr, (width, height) in self.cameras:

      def band(a: int, b: int, cam_adr: int = cam_adr, width: int = width, height: int = height) -> None:
        """Pixels [a, b) of the camera at `cam_adr`."""
        inputs = [self.launch, cam_adr, width, height, a]
        wp.launch(apply_kernel, dim=(self.nworld, b - a), inputs=inputs, outputs=[out], device=self.device)

      launch_bands(module, device=self.device, n=width * height, launch=band, min_band=BAND)
