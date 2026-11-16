"""SensorCamera's device work: Newton's ray bundle in, Newton's output formats out.

`load_rays` turns the multisample bundle into one ray per pixel; `view_shapes` writes the
renderer's segment table so the megakernel writes Newton's shape indices itself;
`newton_outputs` converts the rest of what the renderer wrote into the caller's arrays to
Newton's values in place (clear values, color space, coverage, depth along the bundle ray,
forward depth, packed albedo), only when something differs.
"""

from __future__ import annotations

from typing import Final

import warp as wp
from newton.sensors import SensorCamera as NewtonSensorCamera

from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.post.grading.color import pack_rgba, srgb_eotf, srgb_oetf, unpack_rgba

# Views with this world index keep their outputs untouched.
DISABLE_PRESERVE: Final = wp.constant(int(NewtonSensorCamera.WorldRenderFlag.DISABLE_PRESERVE))
# |ray_scale - 1| below which a bundle direction counts as a unit vector (depth taken as is).
_UNIT_SCALE_TOLERANCE: Final = wp.constant(1.0e-5)


@wp.struct
class Clear:
  """Newton's `ClearData` in the output formats (colors packed, in the output color space)."""

  color: wp.uint32
  depth: float
  shape: wp.uint32
  normal: wp.vec3
  albedo: wp.uint32


@wp.struct
class OutputSwitches:
  """Which outputs a call writes and how (runtime switches of `newton_outputs`)."""

  color: bool
  depth: bool
  forward: bool
  # The renderer's depth went into the forward-depth image (no depth image requested).
  forward_in_depth: bool
  shape: bool
  normal: bool
  albedo: bool
  hdr: bool
  # A shaded output was rendered: the color's coverage alpha marks the misses.
  shaded: bool
  multisample: bool
  # Misses show the environment map (opaque) in the color outputs.
  background: bool
  # ColorSpace.LINEAR outputs (else sRGB).
  linear: bool
  # Some output differs from what the renderer wrote (else only depth scaling or clears).
  pixel_pass: bool


@wp.struct
class Bundle:
  """Per-pixel rays of the Newton bundle (`load_rays`) and the depth-scale flag."""

  origin: wp.array1d[wp.vec3]
  direction: wp.array1d[wp.vec3]
  # Newton's ray parameter per unit of distance (1 / |direction|).
  scale: wp.array1d[float]
  # Fraction of the pixel's samples with a ray.
  coverage: wp.array1d[float]
  # [0]: some ray needs its depth scaled.
  flags: wp.array1d[int]


@wp.struct
class Shapes:
  """Newton shape of each instance per world (`NewtonBinding.shape_table`)."""

  shape0: wp.array1d[wp.int32]
  column: wp.array1d[wp.int32]
  rows: wp.array2d[wp.int32]
  ninstance: int


@wp.struct
class ViewArrays:
  """The caller's (view, height, width) output arrays (placeholders where not requested)."""

  color: wp.array3d[wp.uint32]
  depth: wp.array3d[float]
  forward: wp.array3d[float]
  shape: wp.array3d[wp.uint32]
  normal: wp.array3d[wp.vec3]
  albedo: wp.array3d[wp.uint32]
  hdr: wp.array3d[wp.vec3]


@wp.struct
class Views:
  """Which Newton world each view shows, and the renderer's per-world exposure."""

  world: wp.array1d[wp.int32]
  has_world: bool
  world_count: int
  exposure: wp.array1d[float]


@wp.func
def newton_pack(
  # In:
  rgb: wp.vec3,
  alpha: float,
) -> wp.uint32:
  """Newton's `pack_rgba_to_uint32`: clamped channels, truncated to bytes (0xAABBGGRR)."""
  return pack_rgba(
    wp.clamp(rgb[0], 0.0, 1.0) * 255.0,
    wp.clamp(rgb[1], 0.0, 1.0) * 255.0,
    wp.clamp(rgb[2], 0.0, 1.0) * 255.0,
    wp.clamp(alpha, 0.0, 1.0) * 255.0,
  )


@wp.func
def srgb_encode(
  # In:
  rgb: wp.vec3,
) -> wp.vec3:
  """Linear -> sRGB of each channel, negatives clamped (Newton's `linear_channel_to_srgb_wp`)."""
  return wp.vec3(srgb_oetf(wp.max(rgb[0], 0.0)), srgb_oetf(wp.max(rgb[1], 0.0)), srgb_oetf(wp.max(rgb[2], 0.0)))


@wp.func
def srgb_decode(
  # In:
  rgba: wp.vec4,
) -> wp.vec4:
  """SRGB -> linear of the color channels (Newton's `srgb_channel_to_linear_wp`), alpha kept."""
  return wp.vec4(srgb_eotf(rgba[0]), srgb_eotf(rgba[1]), srgb_eotf(rgba[2]), rgba[3])


@wp.func
def newton_color(
  # In:
  c: wp.uint32,
  linear: bool,
) -> wp.uint32:
  """Raywarp's packed display color (RGBA bytes, sRGB) in Newton's color space."""
  if linear:
    v = srgb_decode(unpack_rgba(c))
    return newton_pack(wp.vec3(v[0], v[1], v[2]), v[3])
  # The byte order (RGBA) already is Newton's 0xAABBGGRR packing.
  return c


@wp.func
def cover(
  # In:
  c: wp.uint32,
  clear: wp.uint32,
  coverage: float,
  srgb: bool,
) -> wp.uint32:
  """Color c covering a fraction of the pixel, the clear color the rest (Newton's resolve)."""
  if coverage >= 1.0:
    return c
  a, b = unpack_rgba(c), unpack_rgba(clear)
  if srgb:
    a, b = srgb_decode(a), srgb_decode(b)
  mix = a * coverage + b * (1.0 - coverage)
  rgb = wp.vec3(mix[0], mix[1], mix[2])
  if srgb:
    rgb = srgb_encode(rgb)
  return newton_pack(rgb, mix[3])


@wp.kernel(enable_backward=False)
def load_rays(
  # In:
  camera_rays: wp.array4d[wp.vec3],
  # Out:
  origin_out: wp.array1d[wp.vec3],
  direction_out: wp.array1d[wp.vec3],
  ray_scale_out: wp.array1d[float],
  coverage_out: wp.array1d[float],
  flags_out: wp.array1d[int],
):
  """Pixel rays of a Newton bundle: the sample (or mean of the samples with a direction).

  ray_scale is Newton's ray parameter per unit of distance (1 / |direction|, exactly 1 for unit
  directions), coverage the fraction of the pixel's samples that have a direction; flags[0]
  (zeroed before) becomes 1 when a pixel with a ray has ray_scale != 1 (depth needs scaling).
  """
  y, x = wp.tid()
  o = wp.vec3(0.0)
  d = wp.vec3(0.0)
  length = float(0.0)
  count = int(0)
  for s in range(camera_rays.shape[2]):
    ds = camera_rays[y, x, s, 1]
    if wp.length_sq(ds) > 0.0:
      o += camera_rays[y, x, s, 0]
      d += ds
      length += wp.length(ds)
      count += 1
  i = y * camera_rays.shape[1] + x
  if count == 0:
    origin_out[i] = camera_rays[y, x, 0, 0]
    direction_out[i] = wp.vec3(0.0)
    ray_scale_out[i] = 0.0
    coverage_out[i] = 0.0
    return
  origin_out[i] = o / float(count)
  direction_out[i] = d / float(count)
  scale = float(count) / length
  if wp.abs(scale - 1.0) < _UNIT_SCALE_TOLERANCE:
    scale = 1.0
  else:
    wp.atomic_max(flags_out, 0, 1)
  ray_scale_out[i] = scale
  coverage_out[i] = float(count) / float(camera_rays.shape[2])


@wp.kernel(enable_backward=False)
def view_active(
  # In:
  view_world: wp.array1d[wp.int32],
  world_count: int,
  # Out:
  active_out: wp.array1d[wp.int32],
):
  """The views the renderer renders (`Renderer.render(active=...)`): -1 for those it leaves
  alone, DISABLE_PRESERVE ones and invalid worlds (cleared later)."""
  v = wp.tid()
  w = view_world[v]
  active_out[v] = wp.where(w == DISABLE_PRESERVE or w < 0 or w >= world_count, -1, v)


@wp.func(inline=True)
def clear_pixel(
  # In:
  clear: Clear,
  want: OutputSwitches,
  view: int,
  y: int,
  x: int,
  # Out:
  images: ViewArrays,
):
  """Clear values in every requested output of pixel (y, x) of a view showing no valid world."""
  if want.color:
    images.color[view, y, x] = clear.color
  if want.hdr:
    images.hdr[view, y, x] = wp.vec3(0.0)
  if want.depth:
    images.depth[view, y, x] = clear.depth
  if want.forward:
    images.forward[view, y, x] = clear.depth
  if want.shape:
    images.shape[view, y, x] = clear.shape
  if want.normal:
    images.normal[view, y, x] = clear.normal
  if want.albedo:
    images.albedo[view, y, x] = clear.albedo


@wp.func(inline=True)
def color_pixel(
  # In:
  c: wp.uint32,
  hit: bool,
  coverage: float,
  clear: Clear,
  want: OutputSwitches,
) -> wp.uint32:
  """Newton's color of a pixel the renderer shaded as `c` (its alpha: coverage of the hit)."""
  out = clear.color
  if want.background:
    # A visible environment map is kept (opaque).
    out = newton_color(wp.uint32(int(c) | (255 << 24)), want.linear)
  elif hit:
    # Partial coverage (multisample): the background already is the clear color.
    a, ca = unpack_rgba(c)[3], unpack_rgba(clear.color)[3]
    alpha = int(wp.clamp(a + (1.0 - a) * ca, 0.0, 1.0) * 255.0 + 0.5)
    out = newton_color(wp.uint32((int(c) & 0x00FFFFFF) | (alpha << 24)), want.linear)
  if want.multisample:
    out = cover(out, clear.color, coverage, not want.linear)
  return out


@wp.func(inline=True)
def depth_pixel(
  # In:
  bundle: Bundle,
  clear: Clear,
  want: OutputSwitches,
  view: int,
  y: int,
  x: int,
  pixel: int,
  # Out:
  images: ViewArrays,
):
  """Newton's depth (along the bundle ray) and forward depth (along -z from the camera origin)."""
  # raywarp: distance along the (normalised) pixel ray, 0 on a miss.
  dist = float(0.0)
  if want.forward_in_depth:
    dist = images.forward[view, y, x]
  else:
    dist = images.depth[view, y, x]
  if want.forward:
    f = clear.depth
    if dist > 0.0:
      f = -(bundle.origin[pixel][2] + dist * wp.normalize(bundle.direction[pixel])[2])
    images.forward[view, y, x] = f
  if want.depth:
    images.depth[view, y, x] = wp.where(dist > 0.0, dist * bundle.scale[pixel], clear.depth)


@wp.kernel(enable_backward=False)
def view_shapes(
  # In:
  world: wp.array1d[wp.int32],
  has_world: bool,
  shapes: Shapes,
  # Out:
  segments: wp.array2d[wp.int32],
):
  """The renderer's segment table of a call: the Newton shape of instance i in view v's world
  (`Renderer.set_segment_table`): the megakernel writes Newton's shape index output itself."""
  v, i = wp.tid()
  w = v
  if has_world:
    w = wp.max(world[v], 0)  # an invalid world's view is cleared later
  col = shapes.column[i]
  s = shapes.shape0[i]
  if col >= 0:
    s = shapes.rows[w % shapes.rows.shape[0], col]
  segments[v, i] = s


@wp.func(inline=True)
def albedo_pixel(
  # In:
  albedo: wp.vec3,
  hit: bool,
  clear: Clear,
  linear: bool,
) -> wp.uint32:
  """Newton's packed albedo (the linear base color, sRGB encoded unless linear outputs)."""
  if not hit:
    return clear.albedo
  a = albedo
  if not linear:
    a = srgb_encode(a)
  return newton_pack(a, 1.0)


@wp.kernel(enable_backward=False)
def newton_outputs(
  # Data in:
  rgba: wp.array4d[wp.vec4ub],
  albedo_in: wp.array4d[wp.vec3],
  normal_in: wp.array4d[wp.vec3],
  # In:
  width: int,
  views: Views,
  bundle: Bundle,
  clear: Clear,
  want: OutputSwitches,
  # Out:
  images: ViewArrays,
):
  """Raywarp's outputs (rendered into the caller's arrays) -> Newton's, in place.

  Thread (view, pixel); views needing no conversion (pixel_pass off: the renderer already wrote
  Newton's values) return at once.
  """
  view, pixel = wp.tid()
  world = view
  if views.has_world:
    world = views.world[view]
    if world == DISABLE_PRESERVE:
      return
  # A view of an invalid world is cleared (the renderer skipped it).
  cleared = world < 0 or world >= views.world_count
  if not cleared and not want.pixel_pass and not (want.depth and bundle.flags[0] != 0):
    return
  y = pixel // width
  x = pixel - y * width
  if cleared:
    clear_pixel(clear, want, view, y, x, images)
    return
  e = exposure_at(views.exposure, world)
  # Shaded outputs: hit where the coverage alpha is not zero.
  hit = True
  if want.shaded:
    hit = rgba[view, 0, y, x][3] != wp.uint8(0)
  if want.color:
    images.color[view, y, x] = color_pixel(images.color[view, y, x], hit, bundle.coverage[pixel], clear, want)
  if want.hdr:
    images.hdr[view, y, x] = wp.where(hit or want.background, images.hdr[view, y, x] * e, wp.vec3(0.0))
  if want.depth or want.forward:
    depth_pixel(bundle, clear, want, view, y, x, pixel, images)
  if want.shape:
    # The megakernel wrote the Newton shape (`view_shapes`); misses get the clear value.
    s = images.shape[view, y, x]
    images.shape[view, y, x] = wp.where(s == wp.uint32(0xFFFFFFFF), clear.shape, s)
  # The guides come from the first sample (a unit normal where it hit): the other samples
  # (aa.spp > 1) can cover a pixel whose first sample missed.
  first_hit = hit
  if want.normal or want.albedo:
    first_hit = hit and wp.length_sq(normal_in[view, 0, y, x]) > 0.0
  if want.normal and not first_hit:
    images.normal[view, y, x] = clear.normal
  if want.albedo:
    images.albedo[view, y, x] = albedo_pixel(albedo_in[view, 0, y, x], first_hit, clear, want.linear)
