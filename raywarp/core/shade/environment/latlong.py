"""How environment-map pixels map to directions on the device (the equirectangular "latlong"
layout: `scene.images.latlong`), and resampling a map seen through a rotation.

`direction_to_uv` / `uv_to_direction` are the device definition of the layout (the NumPy one:
`scene.images.latlong.latlong_directions` / `latlong_uv`); `rotate_latlong` resamples a map seen
through a rotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import device_caps, host_copy
from raywarp.scene.images.latlong import latlong_directions, latlong_uv

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import Floats
  from raywarp.scene.images.latlong import LatLong

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def direction_to_uv(
  # In:
  direction: wp.vec3,
  yaw: wp.float32,
) -> wp.vec2:
  """Equirectangular coordinates of a world direction seen through a map turned by `yaw` about +z.

  Row 0 = +z, centre column = +x, u = 0.25 = +y (longitude phi grows to the left: u = 0.5 -
  phi / 2 pi, Blender's layout).
  """
  phi = wp.atan2(direction[1], direction[0]) - yaw
  u = 0.5 - phi * wp.static(0.5 / wp.pi)
  v = wp.acos(wp.clamp(direction[2], -1.0, 1.0)) * wp.static(1.0 / wp.pi)
  return wp.vec2(u - wp.floor(u), v)


@wp.func
def uv_to_direction(
  # In:
  uv: wp.vec2,
) -> wp.vec3:
  """Map-frame direction of equirectangular coordinates `uv` (the inverse of `direction_to_uv`)."""
  phi = (0.5 - uv[0]) * wp.static(2.0 * wp.pi)
  theta = uv[1] * wp.pi
  s = wp.sin(theta)
  return wp.vec3(s * wp.cos(phi), s * wp.sin(phi), wp.cos(theta))


@wp.kernel
def _rotate_latlong(
  # In:
  src: wp.array3d[wp.float32],
  rotation: wp.mat33,
  # Out:
  dst_out: wp.array3d[wp.float32],
) -> None:
  """Texel (y, x) of `src` (h, w, c) seen through `rotation` (map -> world), bilinear."""
  y, x = wp.tid()
  h = src.shape[0]
  w = src.shape[1]
  d = wp.transpose(rotation) @ uv_to_direction(
    wp.vec2((wp.float32(x) + 0.5) / wp.float32(w), (wp.float32(y) + 0.5) / wp.float32(h))
  )
  uv = direction_to_uv(d, 0.0)
  su = uv[0] * wp.float32(w) - 0.5
  sv = uv[1] * wp.float32(h) - 0.5
  x0 = wp.int32(wp.floor(su))
  y0 = wp.int32(wp.floor(sv))
  fx = su - wp.float32(x0)
  fy = sv - wp.float32(y0)
  xa = ((x0 % w) + w) % w
  xb = (xa + 1) % w
  ya = wp.clamp(y0, 0, h - 1)
  yb = wp.clamp(y0 + 1, 0, h - 1)
  for k in range(src.shape[2]):
    top = src[ya, xa, k] * (1.0 - fx) + src[ya, xb, k] * fx
    bottom = src[yb, xa, k] * (1.0 - fx) + src[yb, xb, k] * fx
    dst_out[y, x, k] = top * (1.0 - fy) + bottom * fy


def rotate_latlong[H: IntVar, W: IntVar, C: IntVar, D: np.generic](
  image: LatLong[H, W, C, D], rotation: Floats[[3, 3]], *, device: wp.Device
) -> LatLong[H, W, C, D]:
  """The equirectangular `image` (h, w, c) seen through `rotation` (3, 3, map -> world frame).

  Output texel direction d samples the input at rotation^T d (bilinear, wrapping in longitude).
  Kernel lookups only rotate about +z (the yaw), so a tilted map (a DomeLight whose pole is not
  world +z) is resampled once: by a kernel on `device` where launches run in parallel (CUDA,
  `DeviceCaps.parallel_launch`; ~50x faster), else on the host (float32 both). A yaw needs no
  resample (`roll_yaw`).
  """
  if device_caps(device).parallel_launch:
    src = wp.array(np.ascontiguousarray(image, dtype=np.float32), dtype=float, device=device)
    dst = wp.empty_like(src)
    rot = wp.mat33(*np.asarray(rotation, np.float32).flatten())
    wp.launch(_rotate_latlong, dim=image.shape[:2], inputs=[src, rot], outputs=[dst], device=device)
    # The device copy keeps the (h, w, c) shape.
    return cast("LatLong[H, W, C, D]", host_copy(dst).astype(image.dtype))
  h, w = image.shape[:2]
  src = latlong_directions(h, w).astype(np.float32) @ np.asarray(rotation, np.float32)  # rows: rotation^T d
  u, v = latlong_uv(src.reshape(h, w, 3))
  su = u * w - 0.5
  sv = v * h - 0.5
  x0 = np.floor(su).astype(np.int64)
  y0 = np.floor(sv).astype(np.int64)
  fx = (su - x0)[..., None].astype(np.float32)
  fy = (sv - y0)[..., None].astype(np.float32)
  xa, xb = x0 % w, (x0 + 1) % w
  ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
  top = image[ya, xa] * (1.0 - fx) + image[ya, xb] * fx
  bottom = image[yb, xa] * (1.0 - fx) + image[yb, xb] * fx
  return (top * (1.0 - fy) + bottom * fy).astype(image.dtype)
