"""The post anti-aliasing (`aa.post`): FXAA 3.11 (quality) across the geometric edges of the final
color (`aa_offset`), the image each sensor effect starts from. Run by the sensor pass
(`sensor.Sensor.run`, step 1) before its effects: `aa_filter` writes the anti-aliased colors of the
edge pixels into the pass's buffer and marks them, `aa_write` writes them over the display (two
launches, no copy of the image: a pixel off the edges costs its depth reads and one mark).

A plain Warp module (`kernel_code.WARP_MODULES`).
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.kernels.abi.output_layout import OUT_DEPTH
from raywarp.core.kernels.abi.structs import WANT_RGB, Cameras, OutputArrays, f16s, f32s, skipped
from raywarp.core.kernels.abi.toggles import out_half
from raywarp.core.kernels.display import write_display
from raywarp.core.post.sensor.inputs import SensorParams, bilinear_corner, decode_display
from raywarp.settings.fields import MISS_DEPTHS

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Post anti-aliasing (`aa.post`, FXAA 3.11 quality, Lottes 2011): local contrast (perceptual luma
# range over the 4-neighbourhood) below max(_AA_EDGE_MIN, _AA_EDGE_REL x its maximum) is no edge
# (FXAA's "high quality" thresholds 0.0625 / 0.125); the sub-pixel blend's strength (its 0.75
# default); the steps [pixels] of the search for the edge's ends (its preset 39 schedule: 12
# steps, unit steps over the first 5 pixels, ends found up to 26.5 pixels away; preset 25's
# coarser start overestimated the span of short stair steps: the error of a slope-1/3 edge 0.58
# of the 1-spp image's with preset 39, 0.71 with 25. The geometric gate leaves FXAA a few % of
# the pixels, so the longer search costs little).
_AA_EDGE_MIN: Final = 0.0625
_AA_EDGE_REL: Final = 0.125
_AA_SUBPIX: Final = 0.75
_AA_STEPS: Final = (1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0)
_aa_steps = wp.constant(wp.types.vector(len(_AA_STEPS), wp.float32)(*_AA_STEPS))
# The geometric edges FXAA filters (`aa_offset`): the inverse depth's bend over three pixels
# (`bend`) above these, for a float32 / float16 depth output. A plane bends 0 (its inverse depth
# is affine in the image), a silhouette about 1, a crease theta x |tan a - tan b| (theta: the
# pixel's angle, a, b: the faces' slopes to the view), a sphere of radius R at depth z theta^2 z /
# R. float32 (rounding bends < 2^-16 on the fixtures' planes): 2^-11 flags at 512^2 and 60
# degrees (theta 2.3e-3) the creases of |tan a - tan b| > 0.22 and no sphere of a radius over 4.6
# pixels (texture edges on curved surfaces stay as rendered); at 1024^2 the creases > 0.43. float16
# (rounding bends up to 2^-10: 43 % of a 256^2 fixture's pixels above 2^-11, the same share as
# float32 at 2^-9): 2^-9.
_AA_BEND: Final = 2.0**-11
_AA_BEND_HALF: Final = 2.0**-9


@wp.func
def luma(
  # In:
  c: wp.vec4,
) -> float:
  """Perceptual luma of a display-linear color: the square root of its Rec. 709 luminance (close
  to the sRGB encoding FXAA reads)."""
  return wp.sqrt(wp.max(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2], 0.0))


@wp.func
def display_texel(
  # In:
  out: OutputArrays,
  srgb8: wp.array1d[float],
  worldid: int,
  cam_adr: int,
  width: int,
  height: int,
  x: int,
  y: int,
) -> wp.vec4:
  """The display color (`decode_display`) of pixel (x, y) of the camera at `cam_adr`, clamped to
  its image."""
  adr = cam_adr + wp.clamp(y, 0, height - 1) * width + wp.clamp(x, 0, width - 1)
  return decode_display(out, srgb8, worldid, adr, cam_adr, width * height)


@wp.func
def display_bilinear(
  # In:
  out: OutputArrays,
  srgb8: wp.array1d[float],
  worldid: int,
  cam_adr: int,
  width: int,
  height: int,
  p: wp.vec2,
) -> wp.vec4:
  """The display color of the camera at `cam_adr` at position p [pixels], bilinear, clamped to its
  image."""
  x0, y0, fx, fy = bilinear_corner(p)
  c00 = display_texel(out, srgb8, worldid, cam_adr, width, height, x0, y0)
  c10 = display_texel(out, srgb8, worldid, cam_adr, width, height, x0 + 1, y0)
  c01 = display_texel(out, srgb8, worldid, cam_adr, width, height, x0, y0 + 1)
  c11 = display_texel(out, srgb8, worldid, cam_adr, width, height, x0 + 1, y0 + 1)
  return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy


@wp.func
def edge_ends(
  # In:
  out: OutputArrays,
  srgb8: wp.array1d[float],
  worldid: int,
  cam_adr: int,
  width: int,
  height: int,
  origin: wp.vec2,
  along: wp.vec2,
  pair: float,
  gradient: float,
) -> wp.vec4:
  """FXAA's search for an edge's two ends from `origin` (half a pixel across the edge) in steps of
  `_AA_STEPS` along -/+ `along`: (distance back, distance forward, luma minus `pair` at each),
  an end where that difference reaches `gradient` (else the last step's)."""
  dn = float(0.0)
  dp = float(0.0)
  end_n = float(0.0)
  end_p = float(0.0)
  done_n = bool(False)
  done_p = bool(False)
  for k in range(wp.static(len(_AA_STEPS))):
    step = _aa_steps[k]
    if not done_n:
      dn += step
      end_n = luma(display_bilinear(out, srgb8, worldid, cam_adr, width, height, origin - along * dn)) - pair
      done_n = wp.abs(end_n) >= gradient
    if not done_p:
      dp += step
      end_p = luma(display_bilinear(out, srgb8, worldid, cam_adr, width, height, origin + along * dp)) - pair
      done_p = wp.abs(end_p) >= gradient
  return wp.vec4(dn, dp, end_n, end_p)


@wp.func
def inverse_depth(
  # In:
  out: OutputArrays,
  params: SensorParams,
  zfar: float,
  worldid: int,
  adr: int,
) -> float:
  """1 / the depth output at `adr`, 0 on a miss (infinitely far: the sky is one plane at
  infinity)."""
  d = float(0.0)
  if out_half(out.half, wp.static(OUT_DEPTH)):
    d = float(f16s(out, out.table.depth16, 1)[worldid, adr])
  else:
    d = f32s(out, out.table.depth)[worldid, adr]
  if not wp.isfinite(d) or d <= 0.0 or (params.miss == wp.static(MISS_DEPTHS["far"]) and d >= zfar and zfar > 0.0):
    return 0.0
  return 1.0 / d


@wp.func
def bend(
  # In:
  a: float,
  m: float,
  b: float,
) -> float:
  """How far inverse depths a, m, b of three pixels in a row are from one plane, relative to the
  nearest: |a - 2m + b| / max(a, m, b) (the inverse depth of a plane is affine in the image, so 0
  inside a surface, about 1 across a silhouette, the slope change across a crease)."""
  near = wp.max(wp.max(a, b), m)
  if near <= 0.0:
    return 0.0
  return wp.abs(a - 2.0 * m + b) / near


@wp.func
def aa_offset(
  # In:
  out: OutputArrays,
  srgb8: wp.array1d[float],
  params: SensorParams,
  zfar: float,
  worldid: int,
  cam_adr: int,
  width: int,
  height: int,
  x: int,
  y: int,
) -> wp.vec2:
  """The post anti-aliasing's resampling offset [pixels] of pixel (x, y): FXAA 3.11's (quality)
  across a geometric edge, 0 elsewhere.

  Geometric: the depth output's inverse is not one plane over the pixel and its two neighbours
  across the edge (`bend` above `_AA_BEND`): a silhouette (the sky included), a thin wire, an
  alpha cut-out or a crease; a texture or shadow edge on a surface stays as rendered. Across such
  an edge FXAA moves the pixel toward its other side by its distance to the ends of the edge (a
  long edge's stair steps become a ramp) or by the sub-pixel blend (a feature thinner than a
  pixel), the larger."""
  i = cam_adr + y * width + x
  um = inverse_depth(out, params, zfar, worldid, i)
  bend_x = float(0.0)
  bend_y = float(0.0)
  if x > 0 and x < width - 1:
    bend_x = bend(inverse_depth(out, params, zfar, worldid, i - 1), um, inverse_depth(out, params, zfar, worldid, i + 1))
  if y > 0 and y < height - 1:
    bend_y = bend(
      inverse_depth(out, params, zfar, worldid, i - width), um, inverse_depth(out, params, zfar, worldid, i + width)
    )
  tolerance = wp.static(_AA_BEND)
  if out_half(out.half, wp.static(OUT_DEPTH)):
    tolerance = wp.static(_AA_BEND_HALF)
  if wp.max(bend_x, bend_y) <= tolerance:
    return wp.vec2(0.0)
  p = wp.vec2(float(x) + 0.5, float(y) + 0.5)
  m = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x, y))
  n = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x, y - 1))
  s = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x, y + 1))
  e = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x + 1, y))
  w = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x - 1, y))
  hi = wp.max(wp.max(wp.max(n, s), wp.max(e, w)), m)
  lo = wp.min(wp.min(wp.min(n, s), wp.min(e, w)), m)
  contrast = hi - lo
  if contrast < wp.max(wp.static(_AA_EDGE_MIN), hi * wp.static(_AA_EDGE_REL)):
    return wp.vec2(0.0)
  nw = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x - 1, y - 1))
  ne = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x + 1, y - 1))
  sw = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x - 1, y + 1))
  se = luma(display_texel(out, srgb8, worldid, cam_adr, width, height, x + 1, y + 1))
  # Horizontal edge (luma changing along y) vs vertical edge; it must be a geometric one.
  horz = wp.abs(nw + sw - 2.0 * w) + 2.0 * wp.abs(n + s - 2.0 * m) + wp.abs(ne + se - 2.0 * e)
  vert = wp.abs(nw + ne - 2.0 * n) + 2.0 * wp.abs(w + e - 2.0 * m) + wp.abs(sw + se - 2.0 * s)
  horizontal = horz >= vert
  if wp.where(horizontal, bend_y, bend_x) <= tolerance:
    return wp.vec2(0.0)
  # The sub-pixel blend: the 3 x 3 low-pass's distance to the pixel, over the contrast.
  average = (2.0 * (n + s + e + w) + nw + ne + sw + se) / 12.0
  sub = wp.clamp(wp.abs(average - m) / contrast, 0.0, 1.0)
  sub = (-2.0 * sub + 3.0) * sub * sub
  subpix = sub * sub * wp.static(_AA_SUBPIX)
  # The side across the edge with the larger gradient: one luma step there, `across` toward it.
  across = wp.vec2(1.0, 0.0)
  along = wp.vec2(0.0, 1.0)
  lum_a = w
  lum_b = e
  if horizontal:
    across = wp.vec2(0.0, 1.0)
    along = wp.vec2(1.0, 0.0)
    lum_a = n
    lum_b = s
  grad_a = wp.abs(lum_a - m)
  grad_b = wp.abs(lum_b - m)
  side = lum_b
  if grad_a >= grad_b:
    across = -across
    side = lum_a
  pair = (side + m) * 0.5
  origin = p + across * 0.5
  ends = edge_ends(out, srgb8, worldid, cam_adr, width, height, origin, along, pair, wp.max(grad_a, grad_b) * 0.25)
  # The nearer end decides: an edge whose luma there crosses the pair's (a stair step's end) moves
  # the pixel across by its distance to that end over the span (0.5 at the end, 0 mid-span).
  below = m - pair < 0.0
  good = (ends[2] < 0.0) != below
  if ends[1] < ends[0]:
    good = (ends[3] < 0.0) != below
  offset = float(0.0)
  if good:
    offset = 0.5 - wp.min(ends[0], ends[1]) / (ends[0] + ends[1])
  return across * wp.max(offset, subpix)


@wp.kernel
def aa_filter(
  # In:
  out: OutputArrays,
  cams: Cameras,
  params: SensorParams,
  cam: int,
  srgb8: wp.array1d[float],
  # Out:
  color: wp.array2d[wp.vec4h],
  marked: wp.array2d[wp.uint8],
):
  """The post anti-aliasing of pixel i of camera `cam` of world w, read from the display: across a
  geometric edge (`aa_offset`) its anti-aliased color into `color`, marked for `aa_write`."""
  w, i = wp.tid()
  lens = cams.lens[cam]
  if skipped(out, w) or (lens.wants & wp.static(WANT_RGB)) == 0:
    return
  width = cams.camera_res[cam][0]
  height = cams.camera_res[cam][1]
  y = i // width
  x = i - y * width
  offset = aa_offset(out, srgb8, params, lens.zfar, w, lens.adr, width, height, x, y)
  if offset[0] != 0.0 or offset[1] != 0.0:
    p = wp.vec2(float(x) + 0.5, float(y) + 0.5) + offset
    color[w, lens.adr + i] = wp.vec4h(display_bilinear(out, srgb8, w, lens.adr, width, height, p))
    marked[w, lens.adr + i] = wp.uint8(1)


@wp.kernel
def aa_write(
  # In:
  cams: Cameras,
  color: wp.array2d[wp.vec4h],
  cam: int,
  # Out:
  marked: wp.array2d[wp.uint8],
  out: OutputArrays,
):
  """Writes pixel i of camera `cam` of world w from `color` where `aa_filter` marked it, and clears
  the mark."""
  w, i = wp.tid()
  lens = cams.lens[cam]
  adr = lens.adr + i
  if marked[w, adr] == wp.uint8(0):
    return
  marked[w, adr] = wp.uint8(0)
  c = wp.vec4(color[w, adr])
  write_display(out, w, adr, lens.adr, cams.camera_res[cam][0] * cams.camera_res[cam][1], wp.vec3(c[0], c[1], c[2]), c[3])
