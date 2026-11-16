"""The sensor pass: what a real camera and depth sensor add to the final images
(`settings.Sensor`: rolling shutter, motion blur, vignetting, shot / read noise, ADC
quantisation, the response curve; active-stereo or range depth noise, dropouts and edge
dropouts), and the post anti-aliasing (`aa.post`: FXAA across the geometric edges of the final
color, the image each color effect starts from).

It runs after every other pass (the display is final) and only when an effect is on: no launch
and no memory otherwise. All its parameters are runtime (`SensorParams`, the per-world tables),
so turning an effect on or off never recompiles; its kernels read the output word at run time
(one set for every output layout).

Per frame, per camera:
1. `aa_filter`, `aa_write` (only with `aa.post`; `fxaa.py`): FXAA 3.11 (quality) on the display
   where the depth output shows a geometric edge (`aa_offset`), read in place: the anti-aliased
   colors of those pixels go to the pass's buffer and are marked, then written over the display
   (two launches, no copy of the image: a pixel off the edges costs its depth reads and one mark).
2. `refresh`: the clean image (what the render wrote this frame) is copied into the pass's own
   buffers. A skipped world's views keep their earlier clean copy, so the effects never compound
   frame over frame. Skipped when no effect reads a neighbour: the color pass then reads each
   pixel in place.
3. `tiles` (only while rolling shutter or motion blur is on): the fastest motion vector of each
   `MOTION_TILE`-square tile, then of its 3 x 3 tiles (`neighbour_max`).
4. `color`: each pixel resamples the clean image along the motion (rolling shutter, motion blur;
   `exposure_sample`), then vignette, noise, quantisation and response, written through the
   color layout's writer (the luminance output follows). `stereo`: one thread per image row
   scans it right to left (the occlusion: the running minimum of x - d(x)) and adds disparity
   noise; `range_pass`: one thread per pixel adds range noise. Both drop pixels (at random, at
   depth edges) and write the depth in the output's convention, misses where the sensor has no
   depth.

A plain Warp module (`kernel_code.WARP_MODULES`): none of it reaches the OptiX programs, so its
edits keep the prebuilt render builds.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.output_layout import OUT_DEPTH, put1
from raywarp.core.kernels.abi.structs import WANT_DEPTH, WANT_RGB, Cameras, OutputArrays, f16s, f32s, skipped
from raywarp.core.kernels.abi.toggles import out_half
from raywarp.core.kernels.display import write_display
from raywarp.core.post.aa.fxaa import aa_filter, aa_write
from raywarp.core.post.grading import color
from raywarp.core.post.sensor.inputs import SensorParams, bilinear_corner, decode_display, power, srgb_decode, srgb_encode
from raywarp.core.trace.rays import camera, seeds
from raywarp.settings.fields import MISS_DEPTHS
from raywarp.settings.sensor import IDENTITY_RESPONSE

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray
  from raywarp.settings import RenderSettings

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Per-world fields (`PerWorld(env=[...])`): name -> dtype; (full_well, read_noise), (a1, a2, a3),
# (toe, shoulder, center, gamma).
PER_WORLD_FIELDS: Final = {"sensor.noise": wp.vec2, "sensor.vignette": wp.vec3, "sensor.response": wp.vec4}
# Motion-blur taps per pixel of motion over the exposure, and their cap: a streak of n pixels
# takes n taps (one per pixel: no gaps), at most 32 (longer streaks are sampled more sparsely).
_MAX_TAPS: Final = 32
# Side of the motion tiles [pixels] (McGuire et al. 2012, "A reconstruction filter for plausible
# motion blur": the tile size is the largest blur radius): a pixel finds the fastest motion among
# its tile and the 8 next ones (`neighbour_max`), so a moving object's streak reaches at least
# `MOTION_TILE` pixels past its silhouette (the tap cap's length: longer streaks are sparse).
MOTION_TILE: Final = _MAX_TAPS
# Largest gap [pixels] between where a dominant-motion candidate was at an instant and the pixel
# for it to cover the pixel then (half a pixel: the point's footprint).
_COVER: Final = 0.5
# Photo-electron counts below which shot noise is drawn from the Poisson law itself; above, from
# its normal limit N(n, n) (same mean and variance; skewness 1 / sqrt(n) < 0.18, below the read
# noise's effect; the exact sampler's rejection loop is the pass's cost at a 1000s e- well).
_POISSON_EXACT: Final = 32.0
# Smallest response exponent / distance of its center from 0 and 1 (a per-world row's clamp).
_MIN_EXPONENT: Final = 1.0e-3
# Random streams of the color noise and of the depth noise / dropout.
_COLOR_STREAM: Final = 0x5E450
_DEPTH_STREAM: Final = 0xDE97


@wp.func
def normal_pair(
  # In:
  state: wp.uint32,
) -> tuple[wp.vec2, wp.uint32]:
  """Two independent standard normal draws from one Box-Muller transform (`wp.randn` keeps one
  of its two)."""
  s = state
  u = wp.max(wp.randf(s), wp.static(2.0**-24))
  angle = wp.static(2.0 * 3.141592653589793) * wp.randf(s)
  r = wp.sqrt(-2.0 * wp.log(u))
  return wp.vec2(r * wp.cos(angle), r * wp.sin(angle)), s


@wp.func
def response(
  # In:
  x: float,
  curve: wp.vec4,
) -> float:
  """The camera response curve (toe, shoulder, center, gamma) at encoded value x (`Sensor`)."""
  # A per-world row is not validated: its parameters are clamped into the curve's domain.
  v = wp.clamp(x, 0.0, 1.0)
  toe = wp.max(curve[0], wp.static(_MIN_EXPONENT))
  shoulder = wp.max(curve[1], wp.static(_MIN_EXPONENT))
  center = wp.clamp(curve[2], wp.static(_MIN_EXPONENT), wp.static(1.0 - _MIN_EXPONENT))
  a = shoulder * center / ((shoulder - toe) * center + toe)
  y = float(0.0)
  if v <= center:
    y = a * power(v / center, toe)
  else:
    y = 1.0 - (1.0 - a) * power((1.0 - v) / (1.0 - center), shoulder)
  return power(y, wp.max(curve[3], wp.static(_MIN_EXPONENT)))


@wp.func
def vignette(
  # In:
  coeff: wp.vec3,
  x: float,
  y: float,
  width: int,
  height: int,
) -> float:
  """The lens falloff 1 + a1 r^2 + a2 r^4 + a3 r^6 in [0, 1] at pixel position (x, y)."""
  scale = 1.0 / float(wp.max(width, height))
  dx = (x - 0.5 * float(width)) * scale
  dy = (y - 0.5 * float(height)) * scale
  r2 = dx * dx + dy * dy
  return wp.clamp(1.0 + r2 * (coeff[0] + r2 * (coeff[1] + r2 * coeff[2])), 0.0, 1.0)


@wp.func
def shot_read_noise(
  # In:
  v: float,
  well: wp.vec2,
  n: float,
  state: wp.uint32,
) -> tuple[float, wp.uint32]:
  """Display-linear v with Poisson shot noise of v x full_well electrons and Gaussian read noise
  (`well`: (full_well, read_noise)), from the standard normal draw n; clamped to [0, 1] (the well
  and the ADC saturate). Below `_POISSON_EXACT` electrons the shot noise is drawn from the
  Poisson law; above, shot and read noise are one normal draw of their summed variance (the
  Poisson's normal limit plus an independent normal is normal)."""
  s = state
  full = wp.max(well[0], 1.0)  # a per-world row is not validated: at least one electron
  mean = wp.max(v, 0.0) * full
  read = wp.max(well[1], 0.0)
  electrons = float(0.0)
  if mean < wp.static(_POISSON_EXACT):
    electrons = float(wp.poisson(s, mean)) + n * read
  else:
    electrons = mean + wp.sqrt(mean + read * read) * n
  return wp.clamp(electrons / full, 0.0, 1.0), s


@wp.func
def frame_seed(
  # In:
  seed: wp.array1d[wp.int32],
  frame: wp.array1d[int],
  worldid: int,
) -> int:
  """The world's noise seed of the frame: the render seed and the world's sensor frame."""
  return int(wp.rand_init(seed[0], frame[worldid]))


@wp.func
def clean_texel(
  # In:
  clean: wp.array2d[wp.vec4h],
  worldid: int,
  adr: int,
  width: int,
  height: int,
  x: int,
  y: int,
) -> wp.vec4:
  """The clean image's pixel (x, y) of the camera at `adr`, clamped to its image."""
  return wp.vec4(clean[worldid, adr + wp.clamp(y, 0, height - 1) * width + wp.clamp(x, 0, width - 1)])


@wp.func
def bilinear(
  # In:
  clean: wp.array2d[wp.vec4h],
  worldid: int,
  adr: int,
  width: int,
  height: int,
  p: wp.vec2,
) -> wp.vec4:
  """The clean image of the camera at `adr` at position p [pixels], bilinear, clamped to it."""
  x0, y0, fx, fy = bilinear_corner(p)
  c00 = clean_texel(clean, worldid, adr, width, height, x0, y0)
  c10 = clean_texel(clean, worldid, adr, width, height, x0 + 1, y0)
  c01 = clean_texel(clean, worldid, adr, width, height, x0, y0 + 1)
  c11 = clean_texel(clean, worldid, adr, width, height, x0 + 1, y0 + 1)
  return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy


@wp.func
def motion_at(
  # In:
  motion: wp.array2d[wp.vec2],
  worldid: int,
  adr: int,
  width: int,
  height: int,
  p: wp.vec2,
) -> wp.vec2:
  """The motion vector of the pixel at position p [pixels] (nearest, clamped to the image)."""
  x = wp.clamp(int(wp.floor(p[0])), 0, width - 1)
  y = wp.clamp(int(wp.floor(p[1])), 0, height - 1)
  return motion[worldid, adr + y * width + x]


@wp.func
def exposure_sample(
  # In:
  params: SensorParams,
  clean: wp.array2d[wp.vec4h],
  motion: wp.array2d[wp.vec2],
  near: wp.array2d[wp.vec2],
  worldid: int,
  adr: int,
  tile_adr: int,
  x: int,
  y: int,
  width: int,
  height: int,
) -> wp.vec4:
  """The pixel's color over its exposure.

  The image (anti-aliased with `aa.post`: `aa_filter` ran before) at its row's readout (rolling
  shutter), averaged over the shutter interval (motion blur). A point now at q was at q + s m(q)
  s frame intervals ago, so at each instant s the pixel p shows the point that was there: one of
  two candidates (McGuire et al. 2012's scatter-as-gather with a gather of two motions):

  - its own: the current image at p - s m(p) (exact where the motion is smooth around p: a
    camera's motion, an object's inside; on an object's trailing edge it reads the background next
    to it, which the object uncovers);
  - the dominant one: q = p - s m(p - s v) with v the fastest motion of the neighbouring tiles
    (`near`: the step along v finds the moving object, the step along its own motion the point of
    it that was at p), taken when q's own motion brings it onto p at that instant (|s (m(q) -
    m(p - s v))| <= `_COVER`) and it moves faster than the own candidate: a moving object streaks
    over the background next to it. Faster stands for nearer (image motion grows as the depth
    shrinks under a camera's translation; a moving object over a static background): the paper's
    depth test without a depth buffer.

  Under an exactly uniform image motion both candidates coincide (the own-motion resampling
  alone); under a camera's rotation or translation with parallax a slightly faster neighbour's
  point can take a tap.
  """
  p = wp.vec2(float(x) + 0.5, float(y) + 0.5)
  i = adr + y * width + x
  if params.rolling_shutter <= 0.0 and params.motion_blur <= 0.0:
    return wp.vec4(clean[worldid, i])
  m = motion[worldid, i]
  tiles_x = (width + wp.static(MOTION_TILE - 1)) // wp.static(MOTION_TILE)
  v = near[worldid, tile_adr + (y // wp.static(MOTION_TILE)) * tiles_x + x // wp.static(MOTION_TILE)]
  start = params.rolling_shutter * (1.0 - p[1] / float(height))
  taps = 1
  if params.motion_blur > 0.0:
    reach = wp.max(wp.length(m), wp.length(v))
    taps = wp.clamp(int(wp.ceil(reach * params.motion_blur)), 1, wp.static(_MAX_TAPS))
  c = wp.vec4(0.0)
  for k in range(taps):
    s = start + params.motion_blur * (float(k) + 0.5) / float(taps)
    own = p - m * s
    # The dominant candidate, refined once by its own motion (an object's image motion varies
    # over it: perspective, rotation).
    mq = motion_at(motion, worldid, adr, width, height, p - v * s)
    q = p - mq * s
    mq2 = motion_at(motion, worldid, adr, width, height, q)
    mo = motion_at(motion, worldid, adr, width, height, own)
    if wp.length(mq2 - mq) * s <= wp.static(_COVER) and wp.length_sq(mq2) > wp.length_sq(mo):
      c += bilinear(clean, worldid, adr, width, height, q)
    else:
      c += bilinear(clean, worldid, adr, width, height, own)
  return c / float(taps)


@wp.kernel
def tile_max(
  # In:
  cams: Cameras,
  out: OutputArrays,
  motion: wp.array2d[wp.vec2],
  cam: int,
  tile_adr: int,
  # Out:
  tiles: wp.array2d[wp.vec2],
):
  """The fastest motion vector of tile t of camera `cam` of world w (`MOTION_TILE` square)."""
  w, t = wp.tid()
  if skipped(out, w) or (cams.lens[cam].wants & wp.static(WANT_RGB)) == 0:
    return
  width = cams.camera_res[cam][0]
  height = cams.camera_res[cam][1]
  adr = cams.lens[cam].adr
  tiles_x = (width + wp.static(MOTION_TILE - 1)) // wp.static(MOTION_TILE)
  x0 = (t % tiles_x) * wp.static(MOTION_TILE)
  y0 = (t // tiles_x) * wp.static(MOTION_TILE)
  best = wp.vec2(0.0)
  for y in range(y0, wp.min(y0 + wp.static(MOTION_TILE), height)):
    for x in range(x0, wp.min(x0 + wp.static(MOTION_TILE), width)):
      m = motion[w, adr + y * width + x]
      if wp.length_sq(m) > wp.length_sq(best):
        best = m
  tiles[w, tile_adr + t] = best


@wp.kernel
def neighbour_max(
  # In:
  cams: Cameras,
  out: OutputArrays,
  tiles: wp.array2d[wp.vec2],
  cam: int,
  tile_adr: int,
  # Out:
  near: wp.array2d[wp.vec2],
):
  """The fastest tile motion among tile t of camera `cam` of world w and its 8 neighbours."""
  w, t = wp.tid()
  if skipped(out, w) or (cams.lens[cam].wants & wp.static(WANT_RGB)) == 0:
    return
  tiles_x = (cams.camera_res[cam][0] + wp.static(MOTION_TILE - 1)) // wp.static(MOTION_TILE)
  tiles_y = (cams.camera_res[cam][1] + wp.static(MOTION_TILE - 1)) // wp.static(MOTION_TILE)
  tx = t % tiles_x
  ty = t // tiles_x
  best = wp.vec2(0.0)
  for y in range(wp.max(ty - 1, 0), wp.min(ty + 2, tiles_y)):
    for x in range(wp.max(tx - 1, 0), wp.min(tx + 2, tiles_x)):
      m = tiles[w, tile_adr + y * tiles_x + x]
      if wp.length_sq(m) > wp.length_sq(best):
        best = m
  near[w, tile_adr + t] = best


def tile_count(width: int, height: int) -> int:
  """Motion tiles of a width x height camera (`MOTION_TILE` squares, the last ones cut)."""
  return -(-width // MOTION_TILE) * -(-height // MOTION_TILE)


@functools.cache
def make_refresh(*, depth: bool):
  """refresh: the clean copies of camera `cam`'s pixels of the worlds rendered (color in
  the output word's layout, display-linear; `depth`: the depth output, 0 on a miss; the stereo
  model's as planar depth, through the rays of the undistorted perspective cameras it takes,
  `check_cameras`)."""
  camera_ray = camera.make_camera_ray(distortion=False)

  @wp.kernel(module="unique", enable_backward=False)
  def refresh(
    # In:
    out: OutputArrays,
    cams: Cameras,
    params: SensorParams,
    cam: int,
    srgb8: wp.array1d[float],
    # Out:
    clean: wp.array2d[wp.vec4h],
    clean_depth: wp.array2d[float],
  ):
    """Pixel i of camera `cam` of world w."""
    w, i = wp.tid()
    lens = cams.lens[cam]
    want = wp.static(WANT_DEPTH if depth else WANT_RGB)
    if skipped(out, w) or (lens.wants & want) == 0:
      return
    width = cams.camera_res[cam][0]
    adr = lens.adr + i
    if wp.static(not depth):
      clean[w, adr] = wp.vec4h(decode_display(out, srgb8, w, adr, lens.adr, width * cams.camera_res[cam][1]))
    else:
      # The miss value is tested on the stored value (radial "far" misses are zfar along the ray).
      d = float(0.0)
      if out_half(out.half, wp.static(OUT_DEPTH)):
        d = float(f16s(out, out.table.depth16, 1)[w, adr])
      else:
        d = f32s(out, out.table.depth)[w, adr]
      if not wp.isfinite(d) or (params.miss == wp.static(MISS_DEPTHS["far"]) and d >= lens.zfar and lens.zfar > 0.0):
        d = 0.0
      elif params.radial and params.stereo:
        pix = camera.pixel_at(cams, out, False, w, cam, i % width, i // width)
        _origin, _direction, _span, axis = camera_ray(pix, 1, 0, 0)
        d = d * -axis[2]
      clean_depth[w, adr] = d

  return refresh


@functools.cache
def make_color(*, direct: bool):
  """color: camera `cam`'s pixels through the exposure, vignette, noise and response, written in
  the output word's layout; `direct`: each pixel reads its own display color in place (no clean
  copy: every view was rewritten this frame and no effect reads a neighbour), else the clean
  copy."""

  @wp.kernel(module="unique", enable_backward=False)
  def color_pass(
    # In:
    cams: Cameras,
    params: SensorParams,
    clean: wp.array2d[wp.vec4h],
    motion: wp.array2d[wp.vec2],
    near: wp.array2d[wp.vec2],
    well: wp.array1d[wp.vec2],
    falloff: wp.array1d[wp.vec3],
    curve: wp.array1d[wp.vec4],
    seed: wp.array1d[wp.int32],
    frame: wp.array1d[int],
    cam: int,
    tile_adr: int,
    srgb8: wp.array1d[float],
    # Out:
    out: OutputArrays,
  ):
    """Pixel i of camera `cam` of world w (`tile_adr`: the camera's first motion tile)."""
    w, i = wp.tid()
    lens = cams.lens[cam]
    if skipped(out, w) or (lens.wants & wp.static(WANT_RGB)) == 0:
      return
    width = cams.camera_res[cam][0]
    height = cams.camera_res[cam][1]
    y = i // width
    x = i - y * width
    if wp.static(direct):
      c = decode_display(out, srgb8, w, lens.adr + i, lens.adr, width * height)
    else:
      c = exposure_sample(params, clean, motion, near, w, lens.adr, tile_adr, x, y, width, height)
    rgb = wp.vec3(c[0], c[1], c[2]) * vignette(falloff[w % falloff.shape[0]], float(x) + 0.5, float(y) + 0.5, width, height)
    if params.noise:
      state = wp.rand_init(seeds.pixel_seed(frame_seed(seed, frame, w), w, lens.adr + i), wp.static(_COLOR_STREAM))
      fw = well[w % well.shape[0]]
      n01, state = normal_pair(state)
      n23, state = normal_pair(state)
      r, state = shot_read_noise(rgb[0], fw, n01[0], state)
      g, state = shot_read_noise(rgb[1], fw, n01[1], state)
      b, state = shot_read_noise(rgb[2], fw, n23[0], state)
      rgb = wp.vec3(r, g, b)
    if params.levels > 0.0:
      for k in range(3):
        rgb[k] = wp.round(wp.clamp(rgb[k], 0.0, 1.0) * params.levels) / params.levels
    rc = curve[w % curve.shape[0]]
    if not (rc == wp.static(wp.vec4(*IDENTITY_RESPONSE))):
      for k in range(3):
        rgb[k] = srgb_decode(response(srgb_encode(wp.clamp(rgb[k], 0.0, 1.0)), rc))
    write_display(out, w, lens.adr + i, lens.adr, width * height, rgb, c[3])

  return color_pass


@wp.func
def jump(
  # In:
  clean_depth: wp.array2d[float],
  worldid: int,
  adr: int,
  z: float,
  threshold: float,
) -> bool:
  """Whether the clean depth at `adr` is a miss or differs from z by more than `threshold` z."""
  zn = clean_depth[worldid, adr]
  return zn <= 0.0 or wp.abs(zn - z) > threshold * z


@wp.func
def at_edge(
  # In:
  clean_depth: wp.array2d[float],
  worldid: int,
  base: int,
  x: int,
  y: int,
  width: int,
  height: int,
  threshold: float,
) -> bool:
  """Whether pixel (x, y) of the camera at `base` is at a depth edge: a 4-neighbour inside the
  image is a miss or jumps by more than `threshold` times the pixel's depth (`jump`)."""
  i = base + y * width + x
  z = clean_depth[worldid, i]
  edge = False
  if x > 0:
    edge = edge or jump(clean_depth, worldid, i - 1, z, threshold)
  if x < width - 1:
    edge = edge or jump(clean_depth, worldid, i + 1, z, threshold)
  if y > 0:
    edge = edge or jump(clean_depth, worldid, i - width, z, threshold)
  if y < height - 1:
    edge = edge or jump(clean_depth, worldid, i + width, z, threshold)
  return edge


@wp.func
def range_sigma(
  # In:
  coeff: wp.vec3,
  z: float,
) -> float:
  """The range model's noise [m rms] at depth z: sigma0 + k (z - z0)^2 (coeff: sigma0, z0, k)."""
  dz = z - coeff[1]
  return coeff[0] + coeff[2] * dz * dz


@wp.func
def miss_value(
  # In:
  params: SensorParams,
  zfar: float,
) -> float:
  """The depth output's miss value (`settings.MISS_DEPTHS`)."""
  miss = float(0.0)
  if params.miss == wp.static(MISS_DEPTHS["inf"]):
    miss = wp.inf
  elif params.miss == wp.static(MISS_DEPTHS["far"]):
    miss = wp.where(zfar > 0.0, zfar, wp.inf)
  return miss


@wp.func
def dropped(
  # In:
  params: SensorParams,
  clean_depth: wp.array2d[float],
  worldid: int,
  base: int,
  x: int,
  y: int,
  width: int,
  height: int,
  state: wp.uint32,
) -> tuple[bool, wp.uint32]:
  """Whether a valid pixel is dropped: at random (`dropout`), or at a depth edge (`edge_dropout`,
  `at_edge`)."""
  s = state
  out = wp.randf(s) < params.dropout
  if params.edge_dropout > 0.0 and wp.randf(s) < params.edge_dropout:
    out = out or at_edge(clean_depth, worldid, base, x, y, width, height, params.edge_threshold)
  return out, s


@functools.cache
def make_stereo():
  """stereo: camera `cam`'s rows through the active-stereo model (of undistorted perspective
  cameras, `check_cameras`), stored as the output word's depth."""
  camera_ray = camera.make_camera_ray(distortion=False)

  @wp.kernel(module="unique", enable_backward=False)
  def stereo_pass(
    # In:
    cams: Cameras,
    params: SensorParams,
    clean_depth: wp.array2d[float],
    seed: wp.array1d[wp.int32],
    frame: wp.array1d[int],
    cam: int,
    # Out:
    out: OutputArrays,
  ):
    """Row y of camera `cam` of world w, right to left."""
    w, y = wp.tid()
    lens = cams.lens[cam]
    if skipped(out, w) or (lens.wants & wp.static(WANT_DEPTH)) == 0:
      return
    width = cams.camera_res[cam][0]
    height = cams.camera_res[cam][1]
    pix = camera.pixel_at(cams, out, False, w, cam, 0, y)
    window = camera.frustum(pix)
    # Focal length [pixels] times the baseline: disparity = fb / z.
    # Horizontal disparity: fx [pixels] = W x image plane / window width.
    fb = float(width) * camera.image_plane(lens) / (window[1] - window[0]) * params.baseline
    miss = miss_value(params, lens.zfar)
    nearest = float(wp.inf)  # min over x' > x of x' - d(x')
    for k in range(width):
      x = width - 1 - k
      adr = lens.adr + y * width + x
      z = clean_depth[w, adr]
      value = miss
      if z > 0.0:
        d = fb / z
        hidden = nearest <= float(x) - d  # the right imager sees a nearer point there
        nearest = wp.min(nearest, float(x) - d)
        state = wp.rand_init(seeds.pixel_seed(frame_seed(seed, frame, w), w, adr), wp.static(_DEPTH_STREAM))
        dn = d + wp.randn(state) * params.disparity_noise
        if params.disparity_step > 0.0:
          dn = wp.round(dn / params.disparity_step) * params.disparity_step
        drop, state = dropped(params, clean_depth, w, lens.adr, x, y, width, height, state)
        if not hidden and not drop and dn > 0.0 and z >= params.min_depth and z <= params.max_depth:
          value = fb / dn
          if params.radial:
            pix.px = x
            _origin, _direction, _span, axis = camera_ray(pix, 1, 0, 0)
            value = value / -axis[2]
      put1(out, out_half(out.half, wp.static(OUT_DEPTH)), out.table.depth, out.table.depth16, w, adr, value)

  return stereo_pass


@wp.kernel(enable_backward=False)
def range_pass(
  # In:
  cams: Cameras,
  params: SensorParams,
  clean_depth: wp.array2d[float],
  seed: wp.array1d[wp.int32],
  frame: wp.array1d[int],
  cam: int,
  # Out:
  out: OutputArrays,
):
  """Pixel i of camera `cam` of world w through the range model, on the depth output's own value
  (any camera)."""
  w, i = wp.tid()
  lens = cams.lens[cam]
  if skipped(out, w) or (lens.wants & wp.static(WANT_DEPTH)) == 0:
    return
  width = cams.camera_res[cam][0]
  adr = lens.adr + i
  z = clean_depth[w, adr]
  value = miss_value(params, lens.zfar)
  if z > 0.0:
    state = wp.rand_init(seeds.pixel_seed(frame_seed(seed, frame, w), w, adr), wp.static(_DEPTH_STREAM))
    zn = z + wp.randn(state) * range_sigma(params.range_noise, z)
    y = i // width
    drop, state = dropped(params, clean_depth, w, lens.adr, i - y * width, y, width, cams.camera_res[cam][1], state)
    if not drop and zn > 0.0 and z >= params.min_depth and z <= params.max_depth:
      value = zn
  put1(out, out_half(out.half, wp.static(OUT_DEPTH)), out.table.depth, out.table.depth16, w, adr, value)


@wp.kernel
def _advance(
  # In:
  skip: wp.array1d[bool],
  # Out:
  frame: wp.array1d[int],
):
  """One more sensor frame of world w unless it is skipped."""
  w = wp.tid()
  if not skip[w % skip.shape[0]]:
    frame[w] = frame[w] + 1


@wp.kernel
def _reset(
  # In:
  mask: wp.array1d[bool],
  # Out:
  frame: wp.array1d[int],
):
  """Restarts the sensor frame of the masked worlds (all with an empty mask)."""
  w = wp.tid()
  if mask.shape[0] == 0 or mask[w]:
    frame[w] = 0


def defaults(settings: RenderSettings) -> dict[str, tuple[float, ...]]:
  """The settings' row of each per-world field (`PER_WORLD_FIELDS`)."""
  s = settings.sensor
  return {"sensor.noise": (s.full_well, s.read_noise), "sensor.vignette": s.vignette, "sensor.response": s.response}


def params(settings: RenderSettings) -> SensorParams:
  """The runtime parameters of `settings`."""
  s, depth = settings.sensor, settings.outputs.depth
  p = SensorParams()
  p.rolling_shutter = float(s.rolling_shutter)
  p.motion_blur = float(s.motion_blur)
  p.noise = s.noise
  p.levels = float(2**s.bits - 1) if s.bits > 0 else 0.0
  p.stereo = s.depth == "stereo"
  p.baseline = float(s.baseline)
  p.disparity_noise = float(s.disparity_noise)
  p.disparity_step = float(s.disparity_step)
  p.range_noise = wp.vec3(*s.range_noise)
  p.min_depth = float(s.min_depth)
  p.max_depth = float(s.max_depth)
  p.dropout = float(s.dropout)
  p.edge_dropout = float(s.edge_dropout)
  p.edge_threshold = float(s.edge_threshold)
  p.miss = MISS_DEPTHS[depth.miss] if depth is not None else 0
  p.radial = depth is not None and depth.convention == "radial"
  return p


@final
class Sensor:
  """The sensor pass of a Renderer (allocated while an effect or the post anti-aliasing is on).

  Memory: 8 B per pixel (the clean color, display-linear float16 + alpha) with a color effect or
  the post anti-aliasing (its anti-aliased colors), 1 B per pixel with the post anti-aliasing (its
  marks), 4 B per pixel (the clean depth) with a depth model, 16 B per `MOTION_TILE`-square tile
  with rolling shutter or motion blur (the tile motions), and one int per world. The color pass
  reads the display in place (no copy) when no effect reads a neighbour (noise, vignette and
  response alone).
  """

  def __init__(self, device: wp.Device, nworld: int, sizes: Sequence[tuple[int, int]]) -> None:
    """The pass of `nworld` worlds of cameras of `sizes` ((width, height) each) on `device`."""
    self.device = device
    self.nworld = nworld
    self.sizes = list(sizes)
    self.total = sum(w * h for w, h in self.sizes)
    self.tile_adr = [sum(tile_count(*size) for size in self.sizes[:cam]) for cam in range(len(self.sizes))]
    self.ntile = sum(tile_count(*size) for size in self.sizes)
    self.frame: wp.array[int, [int]] = wp.zeros(nworld, dtype=int, device=device)
    self.clean: wp.array[wp.vec4h, [int, int]] = wp.zeros((1, 1), dtype=wp.vec4h, device=device)
    self.clean_depth: wp.array[float, [int, int]] = wp.zeros((1, 1), dtype=float, device=device)
    self.marked: wp.array[wp.uint8, [int, int]] = wp.zeros((1, 1), dtype=wp.uint8, device=device)
    self._no_mask = wp.zeros(0, dtype=bool, device=device)
    self._no_motion = wp.zeros((1, 1), dtype=wp.vec2, device=device)
    self.tiles: wp.array[wp.vec2, [int, int]] = self._no_motion
    self.near: wp.array[wp.vec2, [int, int]] = self._no_motion
    # The 8-bit sRGB decode table (`decode_display`): 1 KB.
    self.srgb8: wp.array[float, [int]] = wp.array(color.srgb_to_linear(np.arange(256) / 255.0), dtype=float, device=device)
    self.params = SensorParams()
    self.color_on = False
    self.reads_motion = False
    self.antialias = False
    self.depth_on = False

  def configure(self, settings: RenderSettings, *, per_world: bool) -> None:
    """Takes the settings' parameters and allocates the buffers of the effects that are on
    (`per_world`: a per-world sensor field is declared: the color effects run, their per-world
    values deciding)."""
    self.params = params(settings)
    self.color_on = settings.sensor.color() or per_world
    self.reads_motion = settings.sensor.reads_motion()
    self.antialias = settings.aa.post
    self.depth_on = settings.sensor.depth != "off"
    if not self.reads_motion:  # freed while the motion effects are off
      self.tiles = self.near = self._no_motion
    elif self.tiles is self._no_motion:  # the first configuration that needs them
      self.tiles = wp.zeros((self.nworld, self.ntile), dtype=wp.vec2, device=self.device)
      self.near = wp.zeros((self.nworld, self.ntile), dtype=wp.vec2, device=self.device)
    shape = (self.nworld, self.total)
    if self.depth_on and self.clean_depth.shape != shape:
      self.clean_depth = wp.zeros(shape, dtype=float, device=self.device)
    if self.antialias:
      self._clean()
      if self.marked.shape != shape:
        self.marked = wp.zeros(shape, dtype=wp.uint8, device=self.device)

  def run(  # noqa: PLR0913  # the frame's tables, keyword-only
    self,
    out: OutputArrays,
    cams: Cameras,
    *,
    seed: wp.array[wp.int32, [1]],
    motion: AnyDeviceArray | None,
    world: Mapping[str, AnyDeviceArray],
    skip: wp.array[bool, [int]],
  ) -> None:
    """The pass over the frame's final outputs (`world`: the per-world tables by `PER_WORLD_FIELDS`
    name)."""
    motion = self._no_motion if motion is None else motion
    p, dev = self.params, self.device
    direct = not self.reads_motion
    for cam, (width, height) in enumerate(self.sizes):
      dim = (self.nworld, width * height)
      if self.antialias:
        inputs = [out, cams, p, cam, self.srgb8]
        wp.launch(aa_filter, dim=dim, inputs=inputs, outputs=[self.clean, self.marked], device=dev)
        wp.launch(aa_write, dim=dim, inputs=[cams, self.clean, cam], outputs=[self.marked, out], device=dev)
      if self.color_on:
        if self.reads_motion:
          self._tiles(out, cams, motion, cam)
        if not direct:
          self._clean()
          refresh = make_refresh(depth=False)
          inputs = [out, cams, p, cam, self.srgb8]
          wp.launch(refresh, dim=dim, inputs=inputs, outputs=[self.clean, self.clean_depth], device=dev)
        per_world = (world[name] for name in PER_WORLD_FIELDS)
        inputs = [cams, p, self.clean, motion, self.near, *per_world, seed, self.frame, cam, self.tile_adr[cam], self.srgb8]
        wp.launch(make_color(direct=direct), dim=dim, inputs=inputs, outputs=[out], device=dev)
      if self.depth_on:
        refresh = make_refresh(depth=True)
        inputs = [out, cams, p, cam, self.srgb8]
        wp.launch(refresh, dim=dim, inputs=inputs, outputs=[self.clean, self.clean_depth], device=dev)
        inputs = [cams, p, self.clean_depth, seed, self.frame, cam]
        if p.stereo:
          wp.launch(make_stereo(), dim=(self.nworld, height), inputs=inputs, outputs=[out], device=dev)
        else:
          wp.launch(range_pass, dim=dim, inputs=inputs, outputs=[out], device=dev)
    wp.launch(_advance, dim=self.nworld, inputs=[skip], outputs=[self.frame], device=dev)

  def _clean(self) -> None:
    """Allocates the clean color on its first use (the color effects' copy, the post
    anti-aliasing's colors: one buffer, `aa_write` done before the copy)."""
    if self.clean.shape != (self.nworld, self.total):
      self.clean = wp.zeros((self.nworld, self.total), dtype=wp.vec4h, device=self.device)

  def _tiles(self, out: OutputArrays, cams: Cameras, motion: AnyDeviceArray, cam: int) -> None:
    """Camera `cam`'s tile motions and their neighbourhood maxima (`tile_max`, `neighbour_max`)."""
    dim = (self.nworld, tile_count(*self.sizes[cam]))
    adr, dev = self.tile_adr[cam], self.device
    wp.launch(tile_max, dim=dim, inputs=[cams, out, motion, cam, adr], outputs=[self.tiles], device=dev)
    wp.launch(neighbour_max, dim=dim, inputs=[cams, out, self.tiles, cam, adr], outputs=[self.near], device=dev)

  def reset(self, worlds: wp.array[bool, [int]] | None) -> None:
    """Restarts the noise sequence of the masked worlds (None: all)."""
    wp.launch(
      _reset, dim=self.nworld, inputs=[self._no_mask if worlds is None else worlds], outputs=[self.frame], device=self.device
    )


def check_cameras(settings: RenderSettings, *, perspective: bool) -> None:
  """Raises ValueError for the stereo depth model on cameras that are not all undistorted
  perspective pinholes (`perspective`; ray bundles, orthographic, fisheye and distorted cameras
  have no rectified stereo pair whose rows are epipolar lines). The range model takes any camera."""
  if settings.sensor.depth == "stereo" and not perspective:
    raise ValueError(f"sensor.depth={settings.sensor.depth!r} models a rectified pair of undistorted perspective cameras")
