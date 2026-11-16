"""The first ray of each pixel (from the pinhole cameras or from user-given per-pixel ray bundles)
and the depth / segmentation written from its hit.

`make_camera_rays(model)` returns the kernels' (primary_ray, pixel_cone) funcs of a camera model
(`CameraModel`): the cameras' pinhole projection, optionally through their lens models
(`camera.make_camera_ray`), or per-pixel ray bundles (cameras with `CameraSpec.rays`, see
`Renderer.set_camera_rays`): per-pixel camera-frame origins and
directions sampled at pixel centres. A sub-pixel sample interpolates bilinearly between the four
nearest bundle rays (extrapolating linearly past the outer pixel centres): directions on the
image plane z = -1 when the four point forward (exact for pinhole bundles), else on the unit
sphere, then renormalised. A bundle ray with a zero direction marks a pixel without a ray,
rendered as a miss; samples next to it take the nearest ray. Ray cones (texture filtering,
specular anti-aliasing) take their spread and width from the differences between neighbouring
bundle rays (`bundle.update_cones`); the bundles' device buffers are `trace/rays/bundle.py`'s.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp._base.types import MAXVAL as MJ_MAXVAL
from raywarp.core.kernels.abi.output_layout import (
  OUT_DEPTH,
  OUT_DEPTH_RADIAL,
  OUT_MISS_FAR,
  OUT_MISS_INF,
  OUT_SEG,
  OUT_SEG_SEMANTIC,
  put1,
)
from raywarp.core.kernels.abi.rays import FLEX_HIT, BodyClosest, GeometryFields, Hit
from raywarp.core.kernels.abi.structs import Cameras, OutputArrays, i32s
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.kernels.abi.toggles import out_half, out_on
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.trace.rays.camera import (
  Pixel,
  buffer_index,
  camera_frame_ray,
  distorted_frame_ray,
  eye_distance,
  make_camera_ray,
  near_ray_dist,
)

if TYPE_CHECKING:
  from raywarp.core.kernels.first_hit import FirstHitWriters

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Depth axis of bundle rays: depth = dist * -(0, 0, -1).z, the distance along the ray.
_ALONG_RAY: Final = wp.constant(wp.vec3(0.0, 0.0, -1.0))


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CameraModel:
  """The primary rays a render program is compiled for. A renderer's cameras take one model
  (`resolve.camera_model`): on OptiX a lens model is a ray-generation program in a module of its
  own (`trace/optix/device/programs.build_lens`), linked into the pinhole build's pipeline; on the
  software path a megakernel of its own. Not a build key: no model changes what else a build
  compiles.

  Attributes:
    name: the model's name (its OptiX ray-generation program's suffix).
    distortion: the lens models (OpenCV distortion, fisheye: `camera.make_camera_ray`); a camera
      without coefficients renders as a pinhole.
    bundle: per-pixel ray bundles (`CameraSpec.rays`), which replace the projection.
  """

  name: str
  distortion: bool
  bundle: bool


PINHOLE: Final = CameraModel("pinhole", distortion=False, bundle=False)
# A fisheye, OpenCV distortion or per-world lens coefficients.
DISTORTION: Final = CameraModel("distortion", distortion=True, bundle=False)
BUNDLE: Final = CameraModel("bundle", distortion=False, bundle=True)
# The models besides the pinhole.
LENS_MODELS: Final = (DISTORTION, BUNDLE)


@wp.func
def _bundle_axis(
  # In:
  i: wp.int32,
  n: wp.int32,
  nb: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.float32]:
  """Bundle rays (x0, x1) and weight t of sample i of n along an image axis of nb bundle rays.

  Sample i sits at image coordinate (i + 0.5) / n, i.e. bundle coordinate x = x0 + t with
  x = (i + 0.5) nb / n - 0.5 = num / den. Integer arithmetic keeps t exactly 0 when the sample
  falls on a bundle pixel centre (e.g. one sample per pixel at the bundle's resolution).
  """
  num = wp.int64(2 * i + 1) * wp.int64(nb) - wp.int64(n)
  den = wp.int64(2 * n)
  # num > -den: floor division of the positive num + den.
  x0 = wp.clamp(wp.int32((num + den) // den) - 1, 0, wp.max(nb - 2, 0))
  x1 = wp.min(x0 + 1, nb - 1)
  return x0, x1, wp.float32(num - wp.int64(x0) * den) / wp.float32(den)


@wp.func
def _lerp(
  # In:
  a: wp.vec3,
  b: wp.vec3,
  t: wp.float32,
) -> wp.vec3:
  """Linear interpolation a + (b - a) t."""
  return a + (b - a) * t


@wp.func
@device_generic
def bundle_ray[C: Cameras](
  # Model:
  cams: C,
  # In:
  pix: Pixel,
  subdiv: wp.int32,
  sx: wp.int32,
  sy: wp.int32,
) -> tuple[wp.vec3, wp.vec3]:
  """(direction, origin) in the camera frame of the bundle ray through sub-pixel cell (sx, sy).

  The direction is normalised, or zero where the pixel has no ray.
  """
  res = cams.ray_res[pix.camid]
  adr = cams.ray_adr[pix.camid]
  if subdiv == 1 and sx == 0 and sy == 0 and res[0] == pix.width and res[1] == pix.height:
    # The pixel centre at the bundle's resolution: the pixel's own ray.
    own = adr + pix.py * res[0] + pix.px
    d = cams.ray_direction[own]
    if wp.length_sq(d) > 0.0:
      d = wp.normalize(d)
    return d, cams.ray_origin[own]
  x0, x1, tx = _bundle_axis(pix.px * subdiv + sx, pix.width * subdiv, res[0])
  y0, y1, ty = _bundle_axis(pix.py * subdiv + sy, pix.height * subdiv, res[1])
  r0 = adr + y0 * res[0]
  r1 = adr + y1 * res[0]
  if (tx == 0.0 or tx == 1.0) and (ty == 0.0 or ty == 1.0):
    # On a bundle ray (e.g. one sample per pixel at the bundle's resolution): that ray.
    exact = wp.where(ty == 0.0, r0, r1) + wp.where(tx == 0.0, x0, x1)
    direction = cams.ray_direction[exact]
    if wp.length_sq(direction) > 0.0:
      direction = wp.normalize(direction)
    return direction, cams.ray_origin[exact]
  d00 = cams.ray_direction[r0 + x0]
  d10 = cams.ray_direction[r0 + x1]
  d01 = cams.ray_direction[r1 + x0]
  d11 = cams.ray_direction[r1 + x1]
  if wp.length_sq(d00) == 0.0 or wp.length_sq(d10) == 0.0 or wp.length_sq(d01) == 0.0 or wp.length_sq(d11) == 0.0:
    # Next to a pixel without a ray: the nearest bundle ray (weights rounded), no interpolation.
    near = wp.where(ty < 0.5, r0, r1) + wp.where(tx < 0.5, x0, x1)
    direction = cams.ray_direction[near]
    if wp.length_sq(direction) > 0.0:
      direction = wp.normalize(direction)
    return direction, cams.ray_origin[near]
  o00 = cams.ray_origin[r0 + x0]
  o10 = cams.ray_origin[r0 + x1]
  o01 = cams.ray_origin[r1 + x0]
  o11 = cams.ray_origin[r1 + x1]
  if d00[2] < 0.0 and d10[2] < 0.0 and d01[2] < 0.0 and d11[2] < 0.0:
    # Forward rays: interpolate on the image plane z = -1 (exact for pinhole bundles).
    d00 = d00 * (-1.0 / d00[2])
    d10 = d10 * (-1.0 / d10[2])
    d01 = d01 * (-1.0 / d01[2])
    d11 = d11 * (-1.0 / d11[2])
  direction = _lerp(_lerp(d00, d10, tx), _lerp(d01, d11, tx), ty)
  origin = _lerp(_lerp(o00, o10, tx), _lerp(o01, o11, tx), ty)
  if wp.length_sq(direction) > 0.0:
    direction = wp.normalize(direction)
  return direction, origin


@wp.func
@device_generic
def bundle_cone[C: Cameras](
  # Model:
  cams: C,
  # In:
  pix: Pixel,
) -> tuple[wp.float32, wp.float32]:
  """(spread, width) of one image pixel: ray difference to the next bundle ray along x.

  As for pinhole cameras, the neighbour is the next pixel of the row (in the last column, the
  linear extrapolation of the previous one; precomputed per bundle ray, `bundle.update_cones`).
  Differences are scaled from bundle pixels to image pixels (the image may be rendered at a
  multiple of the bundle's resolution).
  """
  res = cams.ray_res[pix.camid]
  b = cams.ray_adr[pix.camid] + ((pix.py * res[1]) // pix.height) * res[0] + (pix.px * res[0]) // pix.width
  cone = cams.ray_cone[b] * (wp.float32(res[0]) / wp.float32(pix.width))
  return cone[0], cone[1]


# primary_ray(cams, pix, subdiv, sx, sy) -> (world origin, world direction, max distance, depth
# axis) and pixel_cone(cams, pix) -> (spread, width) (`make_camera_rays`).
type PrimaryRay = Callable[[Cameras, Pixel, int, int, int], tuple[wp.vec3, wp.vec3, float, wp.vec3]]
type PixelCone = Callable[[Cameras, Pixel], tuple[float, float]]
# geometry_only(geo, cams, out, pix, worldid, subdiv, centre) (`make_geometry_only`).
type GeometryOnly = Callable[[GeometryFields, Cameras, OutputArrays, Pixel, int, int, int], None]


@specialize
def make_pinhole_rays(*, distortion: bool):
  """(primary_ray, pixel_cone) of the pinhole / orthographic / fisheye cameras (see
  `make_camera_rays`), through the cameras' lens models with `distortion`."""
  camera_ray = make_camera_ray(distortion=distortion)
  frame_ray = distorted_frame_ray if distortion else camera_frame_ray

  @wp.func
  @device_generic
  def pinhole_primary_ray[C: Cameras](
    # Model:
    _cams: C,
    # In:
    pix: Pixel,
    subdiv: wp.int32,
    sx: wp.int32,
    sy: wp.int32,
  ) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.vec3]:
    """The camera's ray through sub-pixel cell (sx, sy) (`camera.make_camera_ray`)."""
    return camera_ray(pix, subdiv, sx, sy)

  @wp.func
  @device_generic
  def pinhole_pixel_cone[C: Cameras](
    # Model:
    _cams: C,
    # In:
    pix: Pixel,
  ) -> tuple[wp.float32, wp.float32]:
    """(spread, width) of one pixel: the difference to the next pixel's ray."""
    d0, o0 = frame_ray(pix, 1, 0, 0)
    d1, o1 = frame_ray(pix, 1, 1, 0)
    return wp.length(d1 - d0), wp.length(o1 - o0)

  return pinhole_primary_ray, pinhole_pixel_cone


@wp.func
@device_generic
def bundle_primary_ray[C: Cameras](
  # Model:
  cams: C,
  # In:
  pix: Pixel,
  subdiv: wp.int32,
  sx: wp.int32,
  sy: wp.int32,
) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.vec3]:
  """The interpolated bundle ray through sub-pixel cell (sx, sy) (see `make_camera_rays`)."""
  ray_dir_cam, ray_org_cam = bundle_ray(cams, pix, subdiv, sx, sy)
  max_dist = wp.float32(MJ_MAXVAL)
  if pix.lens.zfar > 0.0:
    max_dist = pix.lens.zfar
  if wp.length_sq(ray_dir_cam) == 0.0:
    max_dist = 0.0
    ray_dir_cam = _ALONG_RAY
  # The ray starts znear along itself: nearer hits are clipped.
  near = near_ray_dist(_ALONG_RAY, pix.lens.znear)
  origin = pix.pos + pix.rot @ (ray_org_cam + ray_dir_cam * near)
  return origin, pix.rot @ ray_dir_cam, wp.max(max_dist - near, 0.0), _ALONG_RAY


@specialize
def make_camera_rays(model: CameraModel):
  """(primary_ray, pixel_cone) funcs of camera model `model`: the pinhole cameras' (through their
  lens models with `model.distortion`), or a ray bundle's (`model.bundle`).

  primary_ray(cams, pix, subdiv, sx, sy) -> (world origin, world direction, max distance,
  depth axis): the ray through sub-pixel cell (sx, sy) of the pixel divided into subdiv × subdiv
  cells (subdiv 1 and (0, 0): the pixel centre), through the camera's lens model with
  `model.distortion` (OpenCV distortion, fisheye; `camera.make_camera_ray`). The origin is on the
  near plane (`Lens.znear`: hits nearer are clipped, `camera.near_ray_dist`) and the hit distances
  count from it; depth is the eye distance (`camera.eye_distance`) * -depth_axis.z: MuJoCo Warp's
  planar depth for pinhole cameras, the distance along the ray for fisheye cameras and bundles
  (whose rays need not share an image plane). Likewise the near / far planes `Lens.znear` /
  `Lens.zfar` are planar for pinhole cameras and distances along the ray for fisheye cameras and
  bundles; a fisheye or bundle pixel without a ray gets max distance 0 (nothing is hit).

  pixel_cone(cams, pix) -> (spread, width): angular spread and width (orthographic origins) of
  one pixel, the ray-cone parameters. The two variants share the signature (the pinhole one reads
  no `cams` field).
  """
  if model.bundle:
    return bundle_primary_ray, bundle_cone
  return make_pinhole_rays(distortion=model.distortion)


@specialize
def make_center_rays(camera_rays: tuple[PrimaryRay, PixelCone], *, supersampled: bool = False):
  """Primary rays and footprints in output-pixel coordinates: of a pixel of the OptiX supersampled
  image (`supersampled`: n x n pixels per output pixel, n its `Lens.scale`), else the camera's."""
  if not supersampled:
    return camera_rays
  primary_ray, pixel_cone = camera_rays

  @wp.func
  def output_pixel(
    # In:
    pix: Pixel,
  ) -> Pixel:
    """The low-resolution observation pixel containing this high-resolution color sample."""
    low = pix
    n = pix.lens.scale
    low.width = pix.width // n
    low.height = pix.height // n
    low.px = pix.px // n
    low.py = pix.py // n
    low.pixelid = low.py * low.width + low.px
    return low

  @wp.func
  @device_generic
  def center_ray[C: Cameras](
    # Model:
    cams: C,
    # In:
    pix: Pixel,
    subdiv: wp.int32,
    sx: wp.int32,
    sy: wp.int32,
  ) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.vec3]:
    """The low-resolution output pixel's ray, including bundle interpolation and distortion."""
    return primary_ray(cams, output_pixel(pix), subdiv, sx, sy)

  @wp.func
  @device_generic
  def center_cone[C: Cameras](
    # Model:
    cams: C,
    # In:
    pix: Pixel,
  ) -> tuple[wp.float32, wp.float32]:
    """The ray cone of the output pixel's center (an off-center cone would change the texture and
    specular anti-aliasing)."""
    return pixel_cone(cams, output_pixel(pix))

  return center_ray, center_cone


@wp.func
def write_geometry(
  # Out:
  out: OutputArrays,
  # In:
  worldid: wp.int32,
  pix: Pixel,
  hit: Hit,
  depth_axis: wp.vec3,
) -> None:
  """Writes the depth and segment id of a camera ray's first hit (the miss values without one).

  The conventions are the output word's (`output_layout`): a hit's segment id is its row of the
  id table (`OutputTable.segment_id`: the scene's ids, or the instance index without them), with
  `OUT_SEG_SEMANTIC` the table's class ids (0 on a miss, flex: `OutputTable.flex_segment`);
  `OUT_DEPTH_RADIAL` the distance along the ray (else planar); `OUT_MISS_INF` / `OUT_MISS_FAR` the
  depth of a miss (inf, the camera's zfar; else 0); a float16 depth buffer.
  """
  adr = buffer_index(pix)
  if pix.want_seg and out_on(out.bits, wp.static(OUT_SEG)):
    semantic = out_on(out.bits, wp.static(OUT_SEG_SEMANTIC))
    seg = wp.where(semantic, 0, -1)
    if hit.instance_id == wp.static(FLEX_HIT):
      seg = out.table.flex_segment + wp.where(semantic, 0, hit.mesh_id)
    elif hit.instance_id != -1:
      seg = batched(out.table.segment_id, worldid, hit.instance_id)
    i32s(out, out.table.seg)[worldid, adr] = seg
  if pix.want_depth and out_on(out.bits, wp.static(OUT_DEPTH)):
    depth = wp.float32(0.0)
    if out_on(out.bits, wp.static(OUT_MISS_INF)):
      depth = wp.inf
    elif out_on(out.bits, wp.static(OUT_MISS_FAR)):
      depth = wp.where(pix.lens.zfar > 0.0, pix.lens.zfar, wp.inf)
    if hit.instance_id != -1:
      # From the eye: the ray started on the near plane (`make_camera_rays`).
      dist = eye_distance(hit.dist, depth_axis, pix.lens.znear)
      # Planar depth: the distance projected onto the optical axis (-z in the camera frame;
      # along the ray for bundles, `make_camera_rays`).
      depth = wp.where(out_on(out.bits, wp.static(OUT_DEPTH_RADIAL)), dist, dist * -depth_axis[2])
    put1(out, out_half(out.half, wp.static(OUT_DEPTH)), out.table.depth, out.table.depth16, worldid, adr, depth)


@specialize
def make_geometry_only(*, trace_closest: BodyClosest, primary_ray: PrimaryRay, writes: FirstHitWriters):
  """geometry_only(geo, cams, out, pix, worldid, subdiv, centre): depth / segmentation and the
  hit outputs (`first_hit.make_write_hit`).

  Traces the primary ray (`make_camera_rays`) through sub-pixel cell (centre, centre) without
  shading, for cameras that render no RGB.
  """
  write_geometry, write_hit = writes.geometry, writes.hit

  @wp.func
  @device_generic
  def geometry_only[C: Cameras, G: GeometryFields](
    # Model:
    geo: G,
    cams: C,
    out: OutputArrays,
    # In:
    pix: Pixel,
    worldid: wp.int32,
    subdiv: wp.int32,
    centre: wp.int32,
  ) -> None:
    """Depth / segmentation of the pixel's ray through cell (centre, centre), unshaded."""
    origin, direction, max_dist, depth_axis = primary_ray(cams, pix, subdiv, centre, centre)
    hit = trace_closest(geo, worldid, origin, direction, max_dist)
    write_geometry(out, worldid, pix, hit, depth_axis)
    write_hit(geo, out, worldid, buffer_index(pix), hit, origin, direction)

  return geometry_only
