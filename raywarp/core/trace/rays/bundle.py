"""The device buffers of per-pixel ray bundles (cameras given their own rays, `CameraSpec.rays`):
validated uploads of the users' ray images and the rays' cones, read by `trace/rays/primary.py`.

`load_camera_rays` packs one (height, width, 2) ray image per camera (camera-frame origin and
direction per pixel) into flat buffers (`Cameras.ray_*`); `update_cones` recomputes the cones
(the ray difference to the next ray of the row) after the rays change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
import warp as wp

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import Floats

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Ray images: (height, width, 2) vec3 (origin, direction), or NumPy (height, width, 2, 3) floats.
_RAY_PAIR: Final = 2
_RAY_IMAGE_RANK: Final = 3
_RAY_FLOATS_RANK: Final = 4
# Fewest bundle rays per row with a neighbour to take a ray cone from (`_cones`).
_MIN_CONE_WIDTH: Final = 2

# One camera's rays, (height, width, 2): camera-frame origin [..., 0] and direction [..., 1].
type CameraRayImage = wp.array[wp.vec3, [int, int, 2]]
# One camera's rays as accepted on input: a ray image, or NumPy floats (height, width, 2, 3).
type RayImageInput = CameraRayImage | Floats[[int, int, 2, 3]]
# The kernels' bundle buffers over N rays of C cameras: (ray_origin (N,), ray_direction (N,),
# ray_adr (C,), ray_res (C,), ray_cone (N,)) (`Cameras` fields).
type RayBuffers[N: IntVar, C: IntVar] = tuple[
  wp.array[wp.vec3, [N]], wp.array[wp.vec3, [N]], wp.array[int, [C]], wp.array[wp.vec2i, [C]], wp.array[wp.vec2, [N]]
]


@wp.kernel
def _copy_rays(
  # In:
  rays: wp.array3d[wp.vec3],
  adr: wp.int32,
  # Out:
  origin_out: wp.array1d[wp.vec3],
  direction_out: wp.array1d[wp.vec3],
  invalid_out: wp.array1d[wp.int32],
) -> None:
  """Copies one camera's ray image into the flat origin / direction buffers at `adr`; a ray with a
  NaN / inf component becomes a pixel without a ray (zero origin and direction: a miss) and is
  counted in `invalid_out[0]`."""
  y, x = wp.tid()
  i = adr + y * rays.shape[1] + x
  origin = rays[y, x, 0]
  direction = rays[y, x, 1]
  if not (wp.isfinite(origin) and wp.isfinite(direction)):
    origin = wp.vec3(0.0)
    direction = wp.vec3(0.0)
    wp.atomic_add(invalid_out, 0, 1)
  origin_out[i] = origin
  direction_out[i] = direction


def _as_image(image: RayImageInput, device: wp.Device) -> CameraRayImage:
  """One camera's rays as a validated (height, width, 2) vec3 array on `device`."""
  if isinstance(image, np.ndarray):
    if image.ndim != _RAY_FLOATS_RANK or image.shape[-2:] != (_RAY_PAIR, 3):
      raise ValueError(f"a camera's rays as a NumPy array must have shape (height, width, 2, 3), got {image.shape}")
    rays = np.ascontiguousarray(image, dtype=np.float32)
    if not np.isfinite(rays).all():
      bad = np.argwhere(~np.isfinite(rays).all(axis=(2, 3)))
      first = tuple(bad[0].tolist())
      raise ValueError(f"camera rays must be finite: NaN / inf at {len(bad)} pixels, first (row, column) {first}")
    return wp.array(rays, dtype=wp.vec3, device=device)
  if not isinstance(image, wp.array):
    raise TypeError(f"camera rays must be Warp or NumPy arrays, got {type(image).__name__}")
  if image.dtype != wp.vec3:
    raise ValueError(f"camera rays must have dtype vec3, got {wp.types.type_repr(image.dtype)}")
  if image.ndim != _RAY_IMAGE_RANK or image.shape[2] != _RAY_PAIR:
    raise ValueError(f"a camera's rays must have shape (height, width, 2), got {tuple(image.shape)}")
  if image.device != device:
    raise ValueError(f"camera rays must be on the renderer's device {device}, got {image.device}")
  return image


def load_camera_rays(
  images: Sequence[RayImageInput],
  *,
  width: int,
  height: int,
  device: wp.Device,
  invalid: wp.array[wp.int32, [1]],
  buffers: RayBuffers[int, int] | None = None,
) -> RayBuffers[int, int]:
  """(ray_origin, ray_direction, ray_adr, ray_res, ray_cone) of one (H, W, 2) ray image per camera.

  Each camera's rays are in its camera frame (-z forward, +y up, +x right): ``rays[..., 0]`` the
  origin, ``rays[..., 1]`` the direction (normalised by the renderer; a zero direction marks a
  pixel without a ray, rendered as a miss: background, the miss depth). Depth is then the distance
  along each ray and `zfar` bounds it; sub-pixel samples interpolate between neighbouring rays.
  `buffers` (a previous result for the same cameras and resolution: its `ray_adr` / `ray_res` are
  kept) is written in place. `invalid[0]` (device) is set to the number of rays with a NaN / inf
  component, written as misses; no host read (the validation word reports it, `Word.RAYS`).

  Raises:
    ValueError / TypeError: for rays of the wrong shape, dtype or device, or NumPy rays with NaN /
      inf components (Warp arrays: `invalid`).
  """
  checked = [_as_image(image, device) for image in images]
  for i, image in enumerate(checked):
    if tuple(image.shape[:2]) != (height, width):
      raise ValueError(
        f"the rays of camera {i} must have shape ({height}, {width}, 2) (the resolution), got {tuple(image.shape)}"
      )
  npix = width * height
  total = npix * len(checked)
  if buffers is None or buffers[0].shape[0] != total or buffers[2].shape[0] != len(checked):
    # Origins, directions (`_copy_rays`) and cones (`update_cones`) are written in full below.
    buffers = (
      wp.empty(total, dtype=wp.vec3, device=device),
      wp.empty(total, dtype=wp.vec3, device=device),
      wp.array(np.arange(len(checked)).astype(np.int32) * npix, dtype=int, device=device),
      wp.array(np.tile([width, height], (len(checked), 1)).astype(np.int32), dtype=wp.vec2i, device=device),
      wp.empty(total, dtype=wp.vec2, device=device),
    )
  invalid.zero_()
  for i, image in enumerate(checked):
    outputs = [buffers[0], buffers[1], invalid]
    wp.launch(_copy_rays, dim=(height, width), inputs=[image, i * npix], outputs=outputs, device=device)
  update_cones(buffers, width)
  return buffers


@wp.func
def _step(
  # Model:
  origin: wp.array1d[wp.vec3],
  direction: wp.array1d[wp.vec3],
  # In:
  a: wp.int32,
  b: wp.int32,
  extrapolate: bool,
) -> wp.vec2:
  """(angle, distance) between the directions and the origins of bundle rays a and b.

  With `extrapolate`, between ray b and its linear extrapolation beyond b instead: on the image
  plane z = -1 when both rays point forward (exact for pinhole bundles), else on the sphere.
  """
  da = direction[a]
  db = direction[b]
  if wp.length_sq(da) == 0.0 or wp.length_sq(db) == 0.0:
    return wp.vec2(0.0)
  da = wp.normalize(da)
  db = wp.normalize(db)
  width = wp.length(origin[b] - origin[a])
  if extrapolate:
    if da[2] < 0.0 and db[2] < 0.0:
      return wp.vec2(wp.length(wp.normalize(db * (-2.0 / db[2]) + da * (1.0 / da[2])) - db), width)
    return wp.vec2(wp.length(wp.normalize(db * 2.0 - da) - db), width)
  return wp.vec2(wp.length(db - da), width)


@wp.kernel
def _cones(
  # In:
  origin: wp.array1d[wp.vec3],
  direction: wp.array1d[wp.vec3],
  width: wp.int32,
  # Out:
  cone_out: wp.array1d[wp.vec2],
) -> None:
  """Ray cone of each bundle ray: the difference to the next ray of its row (see `update_cones`)."""
  i = wp.tid()
  if width < wp.static(_MIN_CONE_WIDTH):
    cone_out[i] = wp.vec2(0.0)
    return
  if (i % width) + 1 < width:
    cone_out[i] = _step(origin, direction, i, i + 1, False)
  else:
    cone_out[i] = _step(origin, direction, i - 1, i, True)


def update_cones(buffers: RayBuffers[int, int], width: int) -> None:
  """Recomputes the ray cones (`buffers[4]`) after the rays changed (graph capturable).

  Cone of bundle ray b: (angle, distance) between its direction / origin and the next ray's of
  the row (in the last column, the linear extrapolation of the previous one), shared by every
  world: the renderer computes it once per frame instead of once per pixel and world.
  """
  wp.launch(
    _cones, dim=buffers[0].shape[0], inputs=[buffers[0], buffers[1], width], outputs=[buffers[4]], device=buffers[0].device
  )
