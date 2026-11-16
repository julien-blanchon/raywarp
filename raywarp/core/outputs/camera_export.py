"""The cameras of the last render as a dataset records them (`Renderer.camera_info`): per world and
camera the intrinsics K in pixels, the lens model and its coefficients, the resolution and the
camera-to-world pose, computed on the device from the frame's per-world fields and the last
render's poses (one launch, no host read; with `out` nothing is allocated, so a graph can capture
it).

Conventions (OpenCV's, which BOP, COLMAP and most SLAM / reconstruction tools read):

* the camera frame: x right, y down, z forward (the renderer's own camera frame is x
  right, y up, z backward: `CameraInfo.cam_to_world` is that frame times diag(1, -1, -1));
* pixel coordinates: x right, y down, the centre of pixel (0, 0) at (0, 0) (the rendered ray of
  pixel (px, py) passes through image point (px, py));
* perspective (and distorted) cameras: (u, v, 1) ~ K (x_d, y_d, 1) with (x_d, y_d) the normalised
  point (x / z, y / z) through the lens distortion (OpenCV's (k1, k2, p1, p2, k3, k4, k5, k6, s1,
  s2, s3, s4) for "opencv", Kannala-Brandt (k1, k2, k3, k4) for "kannala_brandt": `cv::fisheye`,
  x_d = theta_d (x / r));
* orthographic cameras ("orthographic"): (u, v) = K (x, y, 1), x and y in metres in the camera
  frame (fx, fy in pixels per metre).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal, final

import warp as wp

from raywarp._base.types import ProjectionType
from raywarp.core.kernels.abi.structs import Lens, lens_coefficients
from raywarp.core.trace.rays.projection import camera_window

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.pipeline.cameras import CameraRig

wp.set_module_options({"enable_backward": False})

# The lens model of each projection / distortion (`CameraInfo.models`).
type LensModel = Literal["pinhole", "opencv", "kannala_brandt", "orthographic"]


@wp.func
def _intrinsics(
  # In:
  lens: Lens,
  fovy: float,
  intrinsics: wp.vec4,
  res: wp.vec2i,
) -> wp.mat33:
  """K in pixels of a camera (see the module doc): from its image window on the plane at
  distance 1 (`projection.camera_window`; fisheye: `trace/rays/camera.fisheye_window`'s)."""
  w = float(res[0])
  h = float(res[1])
  window = wp.vec4(0.0)
  if lens.projection == wp.static(int(ProjectionType.FISHEYE)) and lens.sensorsize[1] == 0.0:
    top = fovy * wp.static(wp.pi / 360.0)
    right = top * w / h
    window = wp.vec4(-right, right, top, -top)
  else:
    projection = lens.projection
    if projection == wp.static(int(ProjectionType.FISHEYE)):
      projection = wp.static(int(ProjectionType.PERSPECTIVE))
    window = camera_window(projection, fovy, lens.sensorsize, intrinsics, res[0], res[1], 1.0)
  left, right, top, bottom = window[0], window[1], window[2], window[3]
  # u + 0.5 = W (x - left) / (right - left); v + 0.5 = H (top - y) / (top - bottom), y = -y_cv.
  fx = w / (right - left)
  fy = h / (top - bottom)
  cx = -w * left / (right - left) - 0.5
  cy = h * top / (top - bottom) - 0.5
  return wp.mat33(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)


@wp.kernel
def export(
  # In:
  lens: wp.array1d[Lens],
  res: wp.array1d[wp.vec2i],
  pos: wp.array2d[wp.vec3],
  rot: wp.array2d[wp.mat33],
  fovy: wp.array2d[float],
  intrinsics: wp.array2d[wp.vec4],
  distortion: wp.array2d[lens_coefficients],
  skip: wp.array1d[bool],
  # Out:
  k_out: wp.array2d[wp.mat33],
  coefficients_out: wp.array2d[lens_coefficients],
  pose_out: wp.array2d[wp.mat44],
):
  """Camera c of world w: K, lens coefficients, OpenCV camera-to-world pose; untouched for a
  world the last render skipped (`skip`, 1 or nworld entries)."""
  w, c = wp.tid()
  if skip[wp.min(w, skip.shape[0] - 1)]:
    return
  k_out[w, c] = _intrinsics(lens[c], fovy[w % fovy.shape[0], c], intrinsics[w % intrinsics.shape[0], c], res[c])
  coefficients_out[w, c] = distortion[w % distortion.shape[0], c]
  r = rot[w, c]
  p = pos[w, c]
  # Columns x, -y, -z of the renderer's camera frame.
  pose_out[w, c] = wp.mat44(
    r[0, 0], -r[0, 1], -r[0, 2], p[0],
    r[1, 0], -r[1, 1], -r[1, 2], p[1],
    r[2, 0], -r[2, 1], -r[2, 2], p[2],
    0.0, 0.0, 0.0, 1.0,
  )  # fmt: skip


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class CameraInfo[NWorld: IntVar, NCam: IntVar]:
  """The cameras of the last render (`Renderer.camera_info`; conventions: the module doc).

  Attributes:
    k: (nworld, ncam) mat33 intrinsics in pixels (NumPy: (nworld, ncam, 3, 3)).
    distortion: (nworld, ncam) lens coefficients (12 floats: "opencv" (k1, k2, p1, p2, k3, k4, k5,
      k6, s1, s2, s3, s4), "kannala_brandt" (k1, k2, k3, k4, then 0); zero for "pinhole" /
      "orthographic").
    cam_to_world: (nworld, ncam) mat44 camera-to-world transform of the OpenCV camera frame [m].
    models: the lens model of each camera.
    resolution: (width, height) of each camera.
    frame: frames rendered so far, eager renders and `Renderer.capture` graph replays (the last
      frame's index + 1; 0 before the first).
  """

  k: wp.array[wp.mat33, [NWorld, NCam]]
  distortion: wp.array[lens_coefficients, [NWorld, NCam]]
  cam_to_world: wp.array[wp.mat44, [NWorld, NCam]]
  models: tuple[LensModel, ...]
  resolution: tuple[tuple[int, int], ...]
  frame: int


def models(rig: CameraRig, distorted: bool) -> tuple[LensModel, ...]:
  """The lens model of each camera of `rig` (`distorted`: per-world lens coefficients declared)."""
  out: list[LensModel] = []
  for spec in rig.specs:
    if spec.projection == "fisheye":
      out.append("kannala_brandt")
    elif spec.projection == "orthographic":
      out.append("orthographic")
    elif distorted or any(v != 0.0 for v in spec.distortion or ()):
      out.append("opencv")
    else:
      out.append("pinhole")
  return tuple(out)


def compute[NWorld: IntVar](
  rig: CameraRig[NWorld],
  view: tuple[AnyDeviceArray, AnyDeviceArray, AnyDeviceArray],
  *,
  distorted: bool,
  frame: int,
  skip: wp.array[bool, [int]],
  out: CameraInfo[NWorld, int] | None = None,
) -> CameraInfo[NWorld, int]:
  """The cameras of `rig` at its last poses with the frame's per-world `view` tables (fovy,
  intrinsics, lens coefficients), written into `out` when given (no allocation); the worlds of
  the last render's `skip` mask (1 or nworld entries) are left as they are (zero in a new info).

  Raises:
    ValueError: for ray-bundle cameras (no projection), or an `out` of another rig.
  """
  if rig.rays is not None:
    raise ValueError("camera_info: ray-bundle cameras have no projection to export")
  nworld, ncam = rig.pos.shape[0], rig.ncam
  shape = (nworld, ncam)
  if out is None:
    k = wp.zeros(shape, dtype=wp.mat33, device=rig.device)
    coefficients = wp.zeros(shape, dtype=lens_coefficients, device=rig.device)
    pose = wp.zeros(shape, dtype=wp.mat44, device=rig.device)
  elif out.k.shape != shape or out.k.device != rig.device:
    raise ValueError(f"camera_info(out=...) needs an info of {shape} cameras on {rig.device}, got {out.k.shape}")
  else:
    k, coefficients, pose = out.k, out.distortion, out.cam_to_world
  fovy, intrinsics, distortion = view
  wp.launch(
    export,
    dim=shape,
    inputs=[rig.lens, rig.res, rig.pos, rig.rot, fovy, intrinsics, distortion, skip],
    outputs=[k, coefficients, pose],
    device=rig.device,
  )
  return CameraInfo(
    k=k,
    distortion=coefficients,
    cam_to_world=pose,
    models=models(rig, distorted),
    resolution=tuple(rig.sizes),
    frame=frame,
  )
