"""Camera projection: the image window a pinhole / orthographic / intrinsics camera sees and the
ray through each pixel.

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp@6c33934b3cc0); the image window is
factored out of `compute_ray` (`camera_window`) so that the inverse projection (`camera.project`,
temporal reprojection) uses the same one. The one arithmetic deviation: an orthographic ray's x is
left + (right - left) u with right = left + sensor width, where upstream multiplies u by the
sensor width directly (equal up to one rounding).
"""

from __future__ import annotations

import warp as wp

from raywarp._base.types import ProjectionType

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def camera_window(
  # In:
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  znear: float,
) -> wp.vec4:
  """(left, right, top, bottom) of a camera's image window.

  Perspective: on the plane z = -znear, from fovy or, when the sensor height is set, the
  intrinsics (fx, fy, cx, cy) cropped to the image's aspect; orthographic: on the camera plane,
  `fovy` being the window height.
  """
  inv_img_h = 1.0 / float(img_h)
  aspect = float(img_w) * inv_img_h
  if projection == ProjectionType.ORTHOGRAPHIC:
    sensor_h = fovy
    sensor_w = sensor_h * aspect
    left = -0.5 * sensor_w
    top = 0.5 * sensor_h
    return wp.vec4(left, left + sensor_w, top, -top)
  sensor_h = sensorsize[1]
  # Check if we have intrinsics (sensorsize[1] != 0)
  if sensor_h != 0.0:
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    sensor_w = sensorsize[0]

    target_aspect = aspect
    sensor_aspect = sensor_w / sensor_h
    if target_aspect > sensor_aspect:
      sensor_h = sensor_w / target_aspect
    elif target_aspect < sensor_aspect:
      sensor_w = sensor_h * target_aspect

    inv_fx_znear = znear / fx
    inv_fy_znear = znear / fy
    half_sensor_w = 0.5 * sensor_w
    half_sensor_h = 0.5 * sensor_h
    return wp.vec4(
      -inv_fx_znear * (half_sensor_w - cx),
      inv_fx_znear * (half_sensor_w + cx),
      inv_fy_znear * (half_sensor_h - cy),
      -inv_fy_znear * (half_sensor_h + cy),
    )
  fovy_rad = fovy * wp.static(wp.pi / 180.0)
  half_height = znear * wp.tan(0.5 * fovy_rad)
  half_width = half_height * aspect
  return wp.vec4(-half_width, half_width, half_height, -half_height)


@wp.func
def compute_ray(
  # In:
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  px: int,
  py: int,
  znear: float,
) -> tuple[wp.vec3, wp.vec3]:
  """(direction, origin offset) in the camera frame of the ray through the centre of pixel
  (px, py) of an img_w x img_h image, through the window of `camera_window`.

  The offset from the camera's centre is non-zero for orthographic cameras only.
  """
  window = camera_window(projection, fovy, sensorsize, intrinsic, img_w, img_h, znear)
  left = window[0]
  right = window[1]
  top = window[2]
  bottom = window[3]
  u = (float(px) + 0.5) / float(img_w)
  v = (float(py) + 0.5) * (1.0 / float(img_h))
  x = left + (right - left) * u
  y = top + (bottom - top) * v
  if projection == ProjectionType.ORTHOGRAPHIC:
    # Always pointing forward; the ray is offset from the camera's center.
    return wp.vec3(0.0, 0.0, -1.0), wp.vec3(x, y, 0.0)
  # Ray offset from center not used for perspective cameras
  return wp.normalize(wp.vec3(x, y, -znear)), wp.vec3(0.0, 0.0, 0.0)
