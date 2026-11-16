"""Which camera pixel each GPU thread renders, and the ray a camera's projection casts through
it, plus the inverse projection of a point back to the image.

`setup_pixel` maps a thread to (camera, pixel) (one camera search: `locate_ray`); `camera_ray`
is the primary ray through a sub-pixel cell: pinhole / orthographic / intrinsics (MuJoCo Warp's
`compute_ray`, `projection.py`), optionally through OpenCV's lens distortion (`undistort`), or a
Kannala-Brandt fisheye (`fisheye_frame_ray`); `project` is its inverse. Each camera's constants
(projection, sensor, clip planes, exposure, buffer address, the outputs it writes) are one
`Lens`; its lens coefficients are per world (`Cameras.camera_distortion`). The kernels' primary
rays (camera projection or ray bundle) are built by `primary.make_camera_rays`.

Fisheye cameras (`ProjectionType.FISHEYE`, OpenCV's `cv::fisheye` model): a ray at angle theta
from the optical axis lands at distorted angle theta_d = theta (1 + k1 theta^2 + k2 theta^4 +
k3 theta^6 + k4 theta^8) from the principal point, on the image plane of the intrinsics (or, with
fovy, the equidistant image whose height spans fovy degrees of theta_d); k = 0 is the f-theta
lens. Rays beyond 90 degrees (fields wider than 180) are exact; image points past the model's
fold (theta_d no longer increasing) or beyond theta = 180 degrees have no ray and render as misses.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import MAXVAL as MJ_MAXVAL
from raywarp._base.types import ProjectionType, RenderOrder
from raywarp.core.kernels.abi.structs import WANT_DEPTH, WANT_RGB, WANT_SEG, Cameras, Lens, OutputArrays, lens_coefficients
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.trace.rays.projection import camera_window, compute_ray

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Heights of the pixel bands of RenderOrder.TILED (`Cameras.order`, `tile_rows`): 4 x 8 warps
# keep shaded rays coherent; 8 x 4 warps fill 32-byte sectors of 4-byte outputs, faster where the
# stores dominate (unshaded, unlit).
TILE_ROWS: Final = 8
STORE_TILE_ROWS: Final = 4
# Worlds sharing a warp in RenderOrder.MIXED (with 32 / MIXED_WORLDS pixels of a row each) and in
# RenderOrder.WORLD (one pixel each).
MIXED_WORLDS: Final = 4
WORLD_WORLDS: Final = 32
# Floor of -z of camera-frame ray directions when converting the planar far plane to a distance.
_MIN_FORWARD: Final = 1.0e-6
# Fixed-point iterations of `undistort` (OpenCV's undistortPoints default is 5; 10 converge to
# float precision for the usual |k1| < 0.5 lenses).
UNDISTORT_ITERATIONS: Final = 10
# Floor of the depth -z of a camera-frame point projected by `project` (a point at the eye).
_MIN_DEPTH: Final = 1.0e-12
# Newton iterations of `fisheye_theta` (OpenCV's fisheye::undistortPoints default criteria: 10;
# quadratic convergence reaches float precision in 3-5 for calibrated lenses).
FISHEYE_ITERATIONS: Final = 10
# Largest |theta_d(theta) - theta_d| [rad] of a converged `fisheye_theta` (float32 precision on
# angles up to pi, with margin).
_FISHEYE_RESIDUAL: Final = 1.0e-5
# Points at which `fisheye_increasing` checks the model's slope on [0, theta].
_FISHEYE_SLOPE_SAMPLES: Final = 8


@wp.func
def order_worlds(
  # In:
  order: wp.int32,
) -> wp.int32:
  """Worlds interleaved in a warp by RenderOrder `order` (1: one world per warp)."""
  if order == wp.static(int(RenderOrder.WORLD)):
    return wp.static(WORLD_WORLDS)
  if order == wp.static(int(RenderOrder.MIXED)):
    return wp.static(MIXED_WORLDS)
  return 1


@wp.func
def order_rows(
  # In:
  order: wp.vec2i,
) -> wp.int32:
  """Height of the pixel bands of the frame's order (`Cameras.order`: RenderOrder, TILED's band
  height); 1: row-major pixels."""
  if order[0] == wp.static(int(RenderOrder.TILED)):
    return order[1]
  return 1


def tile_rows(*, store_bound: bool) -> int:
  """TILED's band height: `STORE_TILE_ROWS` for builds whose output stores dominate (no shading or
  unlit shading: measured 6 % faster on a large interior, unlit, at 1024²), else `TILE_ROWS` (1-2 %
  faster shaded)."""
  return STORE_TILE_ROWS if store_bound else TILE_ROWS


@wp.func
def thread_ids(
  # In:
  order: wp.int32,
  nworld: wp.int32,
  nray: wp.int32,
  i: wp.int32,
  j: wp.int32,
) -> tuple[wp.int32, wp.int32]:
  """(worldid, rayid) of thread (i, j) = wp.tid() of an (nworld, nray) launch (flat thread
  i * nray + j), the bijection of RenderOrder `order`.

  Worlds are taken in groups of `order_worlds` (the last one smaller); inside a group consecutive
  threads take the group's worlds for one ray, then the next ray. One world per group is the
  identity.
  """
  group = order_worlds(order)
  if group == 1:
    return i, j
  first = (i // group) * group
  size = wp.min(group, nworld - first)
  # The thread's index within its group's threads (< group * nray: no int32 overflow).
  r = (i - first) * nray + j
  ray = r // size
  return first + r - ray * size, ray


@wp.func
@device_generic
def launch_ids[C: Cameras](
  # Model:
  cams: C,
  # In:
  nworld: wp.int32,
  i: wp.int32,
  j: wp.int32,
) -> tuple[wp.int32, wp.int32]:
  """(worldid, rayid) of thread (i, j) of an (nworld units, cams.nray) render launch in the
  frame's order (`Cameras.order`); worldid -1: a hole of the active list (nothing to render)."""
  unit, ray = thread_ids(cams.order[0][0], nworld, cams.nray, i, j)
  return launch_world(cams, unit), ray


@wp.func
@device_generic
def launch_world[C: Cameras](
  # Model:
  cams: C,
  # In:
  unit: wp.int32,
) -> wp.int32:
  """The world of render-launch unit `unit`: the unit itself without an active list
  (`Cameras.worlds` empty), else the list's entry (-1: a hole)."""
  if cams.worlds.shape[0] == 0:
    return unit
  return cams.worlds[unit]


@wp.func
def thread_pixel(
  # In:
  rows: wp.int32,
  index: wp.int32,
  width: wp.int32,
  height: wp.int32,
) -> wp.vec2i:
  """Thread index within a camera image -> (px, py): bands of `rows` rows (`order_rows`) traversed
  column by column (rows 1: row-major)."""
  if rows > 1:
    band = index // (width * rows)
    n = wp.min(rows, height - band * rows)
    i = index - band * width * rows
    return wp.vec2i(i // n, band * rows + i % n)
  return wp.vec2i(index % width, index // width)


@wp.struct
class Pixel:
  """A camera pixel rendered by one thread.

  Attributes:
    camid: rendering camera (index into the context's active cameras), -1 if nothing to do.
    pixelid: row-major pixel index within the camera image.
    px: pixel column.
    py: pixel row.
    width: image width.
    height: image height.
    want_rgb: whether the camera renders RGB.
    want_depth: whether the camera renders depth.
    want_seg: whether the camera renders segmentation.
    lens: the camera's constants (projection, sensor, clip planes, exposure, buffer address,
      outputs).
    fovy: vertical field of view.
    intrinsics: camera intrinsics.
    distortion: the world's lens coefficients of the camera (`lens_coefficients`).
    rot: camera frame in world.
    pos: camera position in world.
  """

  camid: wp.int32
  pixelid: wp.int32
  px: wp.int32
  py: wp.int32
  width: wp.int32
  height: wp.int32
  want_rgb: bool
  want_depth: bool
  want_seg: bool
  lens: Lens
  fovy: wp.float32
  intrinsics: wp.vec4
  distortion: lens_coefficients
  rot: wp.mat33
  pos: wp.vec3


@wp.func
def buffer_index(
  # In:
  pix: Pixel,
) -> wp.int32:
  """The pixel's flat index in the output buffers (its camera's `Lens.adr` + `pixelid`)."""
  return pix.lens.adr + pix.pixelid


@wp.func
@device_generic
def setup_pixel[C: Cameras](
  # Model:
  cams: C,
  out: OutputArrays,
  # In:
  any_rgb: bool,
  worldid: wp.int32,
  rayid: wp.int32,
) -> Pixel:
  """The pixel of global ray `rayid` (active cameras' images concatenated) and its camera.

  `camid` -1: nothing to do (past the last camera, or a camera rendering none of rgb / depth /
  seg). The render launch maps its threads to rendered worlds only (`launch_ids`).
  """
  pix = Pixel()
  pix.camid = -1
  loc = locate_ray(cams, order_rows(cams.order[0]), rayid)
  if loc[0] == -1:
    return pix
  pix = pixel_at(cams, out, any_rgb, worldid, loc[0], loc[1], loc[2])
  if not pix.want_rgb and not pix.want_depth and not pix.want_seg:
    pix.camid = -1
  return pix


@wp.func
@device_generic
def setup_sample[C: Cameras](
  # Model:
  cams: C,
  out: OutputArrays,
  # In:
  any_rgb: bool,
  worldid: wp.int32,
  rayid: wp.int32,
) -> Pixel:
  """`setup_pixel` of a supersampled image (n x n samples per output pixel, n the lens's `scale`)
  in sample-major warps (`locate_sample`)."""
  return _setup_at(cams, out, any_rgb, worldid, locate_sample(cams, order_rows(cams.order[0]), rayid))


@wp.func
@device_generic
def _setup_at[C: Cameras](
  # Model:
  cams: C,
  out: OutputArrays,
  # In:
  any_rgb: bool,
  worldid: wp.int32,
  loc: wp.vec3i,
) -> Pixel:
  """The pixel of thread location `loc` ((camid, px, py)): `setup_pixel`'s steps (kept apart from
  it so that the builds without supersampling keep their code)."""
  pix = Pixel()
  pix.camid = -1
  if loc[0] == -1:
    return pix
  pix = pixel_at(cams, out, any_rgb, worldid, loc[0], loc[1], loc[2])
  if not pix.want_rgb and not pix.want_depth and not pix.want_seg:
    pix.camid = -1
  return pix


@wp.func
@device_generic
def pixel_at[C: Cameras](
  # Model:
  cams: C,
  out: OutputArrays,
  # In:
  any_rgb: bool,
  worldid: wp.int32,
  camid: wp.int32,
  px: wp.int32,
  py: wp.int32,
) -> Pixel:
  """The pixel (px, py) of rendering camera `camid` (`setup_pixel` without the thread mapping):
  the outputs it writes are the Renderer's (`OutputTable.wants`) the camera keeps (`Lens.wants`).

  Unlike `setup_pixel` it keeps cameras that render none of rgb / depth / seg (motion only).
  """
  pix = Pixel()
  pix.camid = camid
  pix.lens = cams.lens[camid]
  wants = out.table.wants & pix.lens.wants
  pix.want_rgb = False
  if any_rgb:
    pix.want_rgb = (wants & wp.static(WANT_RGB)) != 0
  pix.want_depth = (wants & wp.static(WANT_DEPTH)) != 0
  pix.want_seg = (wants & wp.static(WANT_SEG)) != 0
  pix.width = cams.camera_res[camid][0]
  pix.height = cams.camera_res[camid][1]
  pix.px = px
  pix.py = py
  pix.pixelid = py * pix.width + px
  pix.fovy = batched(cams.camera_fovy, worldid, camid)
  pix.intrinsics = batched(cams.camera_intrinsics, worldid, camid)
  # Read only by the lens models (`primary.CameraModel.distortion`) and the reprojection
  # (`project`): the load is dead code elsewhere.
  pix.distortion = batched(cams.camera_distortion, worldid, camid)
  pix.rot = cams.camera_rot[worldid, camid]
  pix.pos = cams.camera_pos[worldid, camid]
  return pix


@wp.func
@device_generic
def locate_ray[C: Cameras](
  # Model:
  cams: C,
  # In:
  rows: wp.int32,
  rayid: wp.int32,
) -> wp.vec3i:
  """(camid, px, py) of global ray `rayid` (camid -1 past the last camera) with bands of `rows`
  rows (`thread_pixel`): the one camera search."""
  offset = wp.int32(0)
  for i in range(cams.nrender):
    npix = cams.camera_res[i][0] * cams.camera_res[i][1]
    if rayid < offset + npix:
      p = thread_pixel(rows, rayid - offset, cams.camera_res[i][0], cams.camera_res[i][1])
      return wp.vec3i(i, p[0], p[1])
    offset += npix
  return wp.vec3i(-1, 0, 0)


# Threads of one SIMT warp (the CUDA warp size): `locate_sample` gives a warp one sample index.
WARP_THREADS: Final = 32


@wp.func
@device_generic
def locate_sample[C: Cameras](
  # Model:
  cams: C,
  # In:
  rows: wp.int32,
  rayid: wp.int32,
) -> wp.vec3i:
  """(camid, px, py) of global ray `rayid` of a supersampled image (n x n samples per output
  pixel, n the lens's `scale`): each run of `WARP_THREADS` threads takes one sample index s of
  `WARP_THREADS` output pixels (in `thread_pixel` order of the output image), so the samples a
  shading-sharing build skips (`pixel.make_pixel_body`) leave whole warps idle."""
  offset = wp.int32(0)
  for i in range(cams.nrender):
    npix = cams.camera_res[i][0] * cams.camera_res[i][1]
    if rayid < offset + npix:
      n = cams.lens[i].scale
      w = cams.camera_res[i][0] // n
      h = cams.camera_res[i][1] // n
      k = n * n
      local = rayid - offset
      full = (w * h) // wp.static(WARP_THREADS) * wp.static(WARP_THREADS)
      s = wp.int32(0)
      low = wp.int32(0)
      if local < full * k:
        block = local // (wp.static(WARP_THREADS) * k)
        within = local - block * wp.static(WARP_THREADS) * k
        s = within // wp.static(WARP_THREADS)
        low = block * wp.static(WARP_THREADS) + within - s * wp.static(WARP_THREADS)
      else:
        rest = w * h - full
        t = local - full * k
        s = t // rest
        low = full + t - s * rest
      p = thread_pixel(rows, low, w, h)
      return wp.vec3i(i, p[0] * n + s % n, p[1] * n + s // n)
    offset += npix
  return wp.vec3i(-1, 0, 0)


@wp.func
def frustum(
  # In:
  pix: Pixel,
) -> wp.vec4:
  """(left, right, top, bottom) of the pixel's camera window (`projection.camera_window`).

  Perspective: on the plane z = -`image_plane`; orthographic: on the camera plane; fisheye: in
  distorted angles (`fisheye_window`).
  """
  lens = pix.lens
  if lens.projection == wp.static(int(ProjectionType.FISHEYE)):
    return fisheye_window(pix)
  plane = image_plane(lens)
  return camera_window(lens.projection, pix.fovy, lens.sensorsize, pix.intrinsics, pix.width, pix.height, plane)


@wp.func
def image_plane(
  # In:
  lens: Lens,
) -> wp.float32:
  """Distance of a perspective camera's image window (`frustum`, `camera_frame_ray`): the near
  plane, or 1 for a camera without one (znear 0: a window of zero size would cast no ray)."""
  return wp.where(lens.znear > 0.0, lens.znear, 1.0)


@wp.func
def fisheye_window(
  # In:
  pix: Pixel,
) -> wp.vec4:
  """(left, right, top, bottom) of a fisheye camera's image in distorted normalised coordinates
  (theta_d times the direction around the axis, x right, y up): the intrinsics' window on the
  plane at distance 1 (OpenCV's x_d = (u - cx) / fx), else the equidistant window whose height
  spans `fovy` degrees of theta_d."""
  lens = pix.lens
  if lens.sensorsize[1] != 0.0:
    perspective = wp.static(int(ProjectionType.PERSPECTIVE))
    return camera_window(perspective, pix.fovy, lens.sensorsize, pix.intrinsics, pix.width, pix.height, 1.0)
  top = pix.fovy * wp.static(wp.pi / 360.0)
  right = top * wp.float32(pix.width) / wp.float32(pix.height)
  return wp.vec4(-right, right, top, -top)


@wp.func
def radial_terms(
  # In:
  k: lens_coefficients,
  r2: wp.float32,
) -> wp.vec2:
  """(numerator, denominator) of OpenCV's radial factor at r^2 = `r2`: 1 + k1 r^2 + k2 r^4 +
  k3 r^6 over 1 + k4 r^2 + k5 r^4 + k6 r^6 (the rational model; k4 = k5 = k6 = 0: the
  polynomial one)."""
  num = 1.0 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2
  den = 1.0 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2
  return wp.vec2(num, den)


@wp.func
def offset_terms(
  # In:
  k: lens_coefficients,
  x: wp.float32,
  y: wp.float32,
  r2: wp.float32,
) -> wp.vec2:
  """OpenCV's additive terms at normalised (x, y): tangential (p1, p2) and thin prism (s1 r^2 +
  s2 r^4, s3 r^2 + s4 r^4)."""
  dx = 2.0 * k[2] * x * y + k[3] * (r2 + 2.0 * x * x) + (k[9] * r2 + k[8]) * r2
  dy = k[2] * (r2 + 2.0 * y * y) + 2.0 * k[3] * x * y + (k[11] * r2 + k[10]) * r2
  return wp.vec2(dx, dy)


@wp.func
def distort(
  # In:
  k: lens_coefficients,
  xn: wp.float32,
  yn: wp.float32,
) -> wp.vec2:
  """OpenCV's lens distortion (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4: `cv::projectPoints`
  without the tilt) of normalised image coordinates (x right, y down)."""
  r2 = xn * xn + yn * yn
  radial = radial_terms(k, r2)
  d = offset_terms(k, xn, yn, r2)
  return wp.vec2(xn, yn) * (radial[0] / radial[1]) + d


@wp.func
def undistort(
  # In:
  k: lens_coefficients,
  xd: wp.float32,
  yd: wp.float32,
) -> wp.vec2:
  """The inverse of `distort`: OpenCV's fixed-point iteration (`cv::undistortPoints`).

  Like OpenCV (`undistortPoints.regression_14583`), an iterate where the radial factor is not
  positive stops the iteration at the distorted point itself: past the lens model's fold (a
  strong barrel distortion's image corners have no preimage) the iteration would otherwise flip
  the point through the centre (a mirrored ray) or diverge. OpenCV tests 1 / radial < 0; a zero
  numerator or denominator (a division by zero) stops it here as well.
  """
  x = xd
  y = yd
  converging = bool(True)
  for _ in range(wp.static(UNDISTORT_ITERATIONS)):
    if converging:
      r2 = x * x + y * y
      radial = radial_terms(k, r2)
      if radial[0] * radial[1] > 0.0:
        inv_radial = radial[1] / radial[0]
        d = offset_terms(k, x, y, r2)
        x = (xd - d[0]) * inv_radial
        y = (yd - d[1]) * inv_radial
      else:
        converging = False
        x = xd
        y = yd
  return wp.vec2(x, y)


@wp.func
def fisheye_theta_d(
  # In:
  k: lens_coefficients,
  theta: wp.float32,
) -> wp.float32:
  """The Kannala-Brandt distorted angle theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 +
  k4 theta^8) of the angle `theta` from the optical axis."""
  t2 = theta * theta
  return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


@wp.func
def fisheye_slope(
  # In:
  k: lens_coefficients,
  theta: wp.float32,
) -> wp.float32:
  """The slope of `fisheye_theta_d` (d theta_d / d theta)."""
  t2 = theta * theta
  return 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])))


@wp.func
def fisheye_increasing(
  # In:
  k: lens_coefficients,
  theta: wp.float32,
) -> bool:
  """Whether the model is increasing on [0, theta] (the slope positive at `_FISHEYE_SLOPE_SAMPLES`
  evenly spaced angles up to theta) and theta within [0, pi]: past its fold the model maps two
  angles to one image point, and an angle beyond pi is no ray."""
  ok = theta >= 0.0 and theta <= wp.pi
  n = wp.static(_FISHEYE_SLOPE_SAMPLES)
  for i in range(n):
    ok = ok and fisheye_slope(k, theta * wp.float32(i + 1) / wp.float32(n)) > 0.0
  return ok


@wp.func
def fisheye_theta(
  # In:
  k: lens_coefficients,
  theta_d: wp.float32,
) -> tuple[wp.float32, bool]:
  """(theta, valid) of distorted angle `theta_d` >= 0: OpenCV's Newton iteration
  (`cv::fisheye::undistortPoints`) from theta = theta_d, valid when it converged to an angle on
  the model's increasing branch within [0, pi] (`fisheye_increasing`)."""
  theta = theta_d
  converging = bool(True)
  for _ in range(wp.static(FISHEYE_ITERATIONS)):
    if converging:
      slope = fisheye_slope(k, theta)
      if slope > 0.0:
        theta = theta - (fisheye_theta_d(k, theta) - theta_d) / slope
      else:
        converging = False
  residual = wp.abs(fisheye_theta_d(k, theta) - theta_d)
  return theta, converging and residual <= wp.static(_FISHEYE_RESIDUAL) and fisheye_increasing(k, theta)


@wp.func
def fisheye_frame_ray(
  # In:
  pix: Pixel,
  subdiv: wp.int32,
  sx: wp.int32,
  sy: wp.int32,
) -> tuple[wp.vec3, bool]:
  """(camera-frame direction, valid) of the fisheye ray through sub-pixel cell (sx, sy) (as
  `camera_frame_ray`): the cell centre's distorted angle and azimuth (`fisheye_window`) through
  `fisheye_theta`. Invalid past the model's valid range (the direction is then that of the
  unconverged angle)."""
  window = fisheye_window(pix)
  u = (wp.float32(pix.px * subdiv + sx) + 0.5) / wp.float32(pix.width * subdiv)
  v = (wp.float32(pix.py * subdiv + sy) + 0.5) / wp.float32(pix.height * subdiv)
  x = window[0] + (window[1] - window[0]) * u
  y = window[2] + (window[3] - window[2]) * v
  theta_d = wp.sqrt(x * x + y * y)
  theta, ok = fisheye_theta(pix.distortion, theta_d)
  scale = wp.float32(1.0)
  if theta_d > 0.0:
    scale = wp.sin(theta) / theta_d
  return wp.vec3(x * scale, y * scale, -wp.cos(theta)), ok


@wp.func
def project(
  # In:
  pix: Pixel,
  window: wp.vec4,
  p_cam: wp.vec3,
) -> tuple[wp.vec2, bool]:
  """Image position [pixels, pixel centres at i + 0.5] of camera-frame point `p_cam`.

  The inverse of the camera rays (`window` = `frustum`, the lens distortion / fisheye model
  included); False behind a perspective camera or outside a fisheye model's valid range.
  """
  if pix.lens.projection == wp.static(int(ProjectionType.FISHEYE)):
    x = p_cam[0]
    y = p_cam[1]
    r = wp.sqrt(x * x + y * y)
    theta = wp.atan2(r, -p_cam[2])
    scale = wp.float32(0.0)
    if r > 0.0:
      scale = fisheye_theta_d(pix.distortion, theta) / r
    return window_pixel(pix, window, x * scale, y * scale), fisheye_increasing(pix.distortion, theta)
  return pinhole_project(pix, window, p_cam)


@wp.func
def pinhole_project(
  # In:
  pix: Pixel,
  window: wp.vec4,
  p_cam: wp.vec3,
) -> tuple[wp.vec2, bool]:
  """`project` of a perspective (OpenCV distortion included) or orthographic camera."""
  lens = pix.lens
  x = p_cam[0]
  y = p_cam[1]
  ok = True
  if lens.projection != wp.static(int(ProjectionType.ORTHOGRAPHIC)):
    ok = p_cam[2] < 0.0
    plane = image_plane(lens)
    s = plane / wp.max(-p_cam[2], wp.static(_MIN_DEPTH))
    x = x * s
    y = y * s
    if wp.length_sq(pix.distortion) != 0.0:
      # On the window plane: normalised coordinates (y down) distorted by the lens.
      d = distort(pix.distortion, x / plane, -y / plane)
      x = d[0] * plane
      y = -d[1] * plane
  return window_pixel(pix, window, x, y), ok


@wp.func
def window_pixel(
  # In:
  pix: Pixel,
  window: wp.vec4,
  x: wp.float32,
  y: wp.float32,
) -> wp.vec2:
  """Image position [pixels] of window point (x, y) (`frustum`'s units)."""
  u = (x - window[0]) / (window[1] - window[0])
  v = (y - window[2]) / (window[3] - window[2])
  return wp.vec2(u * wp.float32(pix.width), v * wp.float32(pix.height))


@specialize
def make_project(*, fisheye: bool):
  """project_point(pix, p_cam): `project` through the camera's own window (`frustum`), with
  the fisheye models compiled in only with `fisheye` (`primary.CameraModel.distortion`): the
  programs of other camera models (motion) carry no fisheye code."""

  @wp.func
  def project_point(
    # In:
    pix: Pixel,
    p_cam: wp.vec3,
  ) -> tuple[wp.vec2, bool]:
    """Image position of camera-frame point `p_cam` and whether the camera images it."""
    if wp.static(fisheye):
      return project(pix, frustum(pix), p_cam)
    window = camera_window(
      pix.lens.projection, pix.fovy, pix.lens.sensorsize, pix.intrinsics, pix.width, pix.height, image_plane(pix.lens)
    )
    return pinhole_project(pix, window, p_cam)

  return project_point


@wp.func
def headlight_dir(
  # In:
  pix: Pixel,
) -> wp.vec3:
  """Direction of the headlight's light: the camera's viewing direction (-z)."""
  return -wp.vec3(pix.rot[0, 2], pix.rot[1, 2], pix.rot[2, 2])


@wp.func
def camera_frame_ray(
  # In:
  pix: Pixel,
  subdiv: wp.int32,
  sx: wp.int32,
  sy: wp.int32,
) -> tuple[wp.vec3, wp.vec3]:
  """(direction, origin offset) in the camera frame of the ray through sub-pixel cell (sx, sy).

  The pixel is divided into subdiv × subdiv cells; the ray goes through the centre of cell
  (sx, sy), so subdiv 1 and (0, 0) give the pixel centre (and (1, 0) the next pixel's). No lens
  distortion (`distorted_frame_ray`).
  """
  lens = pix.lens
  return compute_ray(
    lens.projection,
    pix.fovy,
    lens.sensorsize,
    pix.intrinsics,
    pix.width * subdiv,
    pix.height * subdiv,
    pix.px * subdiv + sx,
    pix.py * subdiv + sy,
    image_plane(lens),
  )


@wp.func
def distorted_frame_ray(
  # In:
  pix: Pixel,
  subdiv: wp.int32,
  sx: wp.int32,
  sy: wp.int32,
) -> tuple[wp.vec3, wp.vec3]:
  """`camera_frame_ray` through the camera's lens model: a fisheye's (`fisheye_frame_ray`; past
  the model's valid range the direction of the unconverged angle, which `camera_ray` traces as a
  miss), or a perspective camera's OpenCV distortion (`Pixel.distortion`; none: the pinhole ray
  itself), the image point undistorted before the ray is cast."""
  if pix.lens.projection == wp.static(int(ProjectionType.FISHEYE)):
    fisheye_direction, _valid = fisheye_frame_ray(pix, subdiv, sx, sy)
    return fisheye_direction, wp.vec3(0.0)
  direction, offset = camera_frame_ray(pix, subdiv, sx, sy)
  if pix.lens.projection == wp.static(int(ProjectionType.ORTHOGRAPHIC)) or wp.length_sq(pix.distortion) == 0.0:
    return direction, offset
  inv_z = -1.0 / direction[2]
  u = undistort(pix.distortion, direction[0] * inv_z, -direction[1] * inv_z)
  return wp.normalize(wp.vec3(u[0], -u[1], -1.0)), offset


@specialize
def make_camera_ray(*, distortion: bool):
  """camera_ray(pix, subdiv, sx, sy) -> (world origin, world direction, max distance, depth
  axis) of the camera's primary ray through sub-pixel cell (sx, sy) (`primary.make_camera_rays`),
  through the lens models with `distortion` (`primary.CameraModel.distortion`: OpenCV distortion,
  fisheye).

  Pinhole / orthographic rays start on the near plane z = -znear and end on the far plane (the
  depth axis is the camera-frame direction: planar depth); fisheye rays start znear along
  themselves and end zfar along themselves (depth axis -z: the distance along the ray, which
  stays defined beyond 90 degrees); a fisheye pixel without a ray (`fisheye_frame_ray`) gets max
  distance 0 along the optical axis (a miss, as a ray bundle's pixel without a ray).
  """

  @wp.func
  def fisheye_ray(
    # In:
    pix: Pixel,
    subdiv: wp.int32,
    sx: wp.int32,
    sy: wp.int32,
  ) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.vec3]:
    """The fisheye camera's primary ray (see `make_camera_ray`)."""
    axis = wp.vec3(0.0, 0.0, -1.0)
    ray_dir_cam, ok = fisheye_frame_ray(pix, subdiv, sx, sy)
    far = max_ray_dist(axis, pix.lens.zfar)
    if not ok:
      ray_dir_cam = axis
      far = 0.0
    near = near_ray_dist(axis, pix.lens.znear)
    direction = pix.rot @ ray_dir_cam
    return pix.pos + direction * near, direction, wp.max(far - near, 0.0), axis

  @wp.func
  def camera_ray(
    # In:
    pix: Pixel,
    subdiv: wp.int32,
    sx: wp.int32,
    sy: wp.int32,
  ) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.vec3]:
    """The primary ray through cell (sx, sy) (see `make_camera_ray`)."""
    if wp.static(distortion):
      if pix.lens.projection == wp.static(int(ProjectionType.FISHEYE)):
        return fisheye_ray(pix, subdiv, sx, sy)
      ray_dir_cam, ray_offset_cam = distorted_frame_ray(pix, subdiv, sx, sy)
    else:
      ray_dir_cam, ray_offset_cam = camera_frame_ray(pix, subdiv, sx, sy)
    # The ray starts on the near plane: nearer hits are clipped.
    near = near_ray_dist(ray_dir_cam, pix.lens.znear)
    far = wp.max(max_ray_dist(ray_dir_cam, pix.lens.zfar) - near, 0.0)
    direction = pix.rot @ ray_dir_cam
    return pix.pos + pix.rot @ ray_offset_cam + direction * near, direction, far, ray_dir_cam

  return camera_ray


@wp.func
def max_ray_dist(
  # In:
  ray_dir_cam: wp.vec3,
  zfar: wp.float32,
) -> wp.float32:
  """Distance to the far plane along a camera ray (unbounded when zfar <= 0)."""
  if zfar > 0.0:
    return zfar / wp.max(-ray_dir_cam[2], wp.static(_MIN_FORWARD))
  return wp.float32(MJ_MAXVAL)


@wp.func
def near_ray_dist(
  # In:
  depth_axis: wp.vec3,
  znear: wp.float32,
) -> wp.float32:
  """Distance to the near plane along a camera ray (0 when znear <= 0): primary rays start there,
  so hits nearer than znear are clipped.

  `depth_axis` is the ray's camera-frame direction (`primary.make_camera_rays`): the near plane is
  planar for pinhole cameras (z = -znear) and a distance along the ray for ray bundles."""
  if znear > 0.0:
    return znear / wp.max(-depth_axis[2], wp.static(_MIN_FORWARD))
  return 0.0


@wp.func
def eye_distance(
  # In:
  dist: wp.float32,
  depth_axis: wp.vec3,
  znear: wp.float32,
) -> wp.float32:
  """Distance from the eye of a primary hit `dist` along the ray from its near-plane origin."""
  return dist + near_ray_dist(depth_axis, znear)


@wp.func
def view_depth(
  # In:
  lens: Lens,
  p_cam: wp.vec3,
) -> wp.float32:
  """Depth of camera-frame point `p_cam` in the convention of the camera's primary rays
  (`make_camera_ray`'s depth axis): planar (-z), the distance from the eye for fisheye cameras."""
  if lens.projection == wp.static(int(ProjectionType.FISHEYE)):
    return wp.length(p_cam)
  return -p_cam[2]
