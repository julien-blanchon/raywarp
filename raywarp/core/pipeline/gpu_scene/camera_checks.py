"""Checks of a camera's parameters (projection, field of view, intrinsics, clip planes,
resolution, lens distortion, outputs), run when a `scene.CameraSpec` is built and when a
Renderer's outputs are set (`camera_outputs`), so that a bad value fails at once with the
field's name instead of deep in a kernel. The number, resolution and clip checks are
`arguments`' (shared with the Renderer's)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp._base.arguments import check_clip, check_number, check_numbers, check_resolution
from raywarp._base.named import construct_named
from raywarp.settings.outputs import Outputs, is_kind

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec

_PROJECTIONS: Final = ("perspective", "orthographic", "fisheye")
# Widest perspective field of view [degrees] (exclusive): a pinhole cannot see 180 degrees.
_MAX_FOVY_DEG: Final = 180.0
# Widest fisheye field of view [degrees] (inclusive): the image height spans theta_d up to 180
# degrees on each side of the axis.
_MAX_FISHEYE_FOVY_DEG: Final = 360.0
# Distortion coefficient counts: OpenCV's distortion vector of a perspective camera (4, 5, 8 or 12:
# `cv::projectPoints` without the tilt's 14), Kannala-Brandt (k1, k2, k3, k4) of a fisheye.
_DISTORTION_SIZES: Final = {"perspective": (4, 5, 8, 12), "fisheye": (4,)}
# The outputs a camera selects as one: segment ids and semantic classes (`camera_outputs`).
_SEG_OUTPUTS: Final = frozenset({"seg", "semantic"})
# Entries of an intrinsics tuple (fx, fy, cx, cy), of a position and of a rotation matrix.
_NINTRINSICS: Final = 4
_NPOS: Final = 3
_NROT: Final = 9
# Entries of a (width, height) size.
_NSIZE: Final = 2
# `CameraSpec.conform` values.
_CONFORMS: Final = ("crop", "fit")
# Largest |R R^T - I| entry of a `CameraSpec.local` rotation: float32 round-off of a rotation
# composed in a few steps is ~1e-6; 1e-3 still rejects any real scale or shear.
_ROTATION_TOLERANCE: Final = 1e-3
# Largest |CameraSpec.exposure| [stops]: 2^exposure stays a normal float32 (2^128 is inf,
# 2^-150 is 0).
_MAX_EXPOSURE_STOPS: Final = 126.0


def _check_sensor(spec: CameraSpec) -> None:
  """Validates `spec`'s conform rule, intrinsics and sensor (ValueError naming the field)."""
  if spec.conform not in _CONFORMS:
    raise ValueError(f"CameraSpec.conform must be one of {_CONFORMS}, got {spec.conform!r}")
  window = spec.conform == "fit" and spec.projection != "fisheye" and spec.intrinsics is None
  if spec.intrinsics is not None and spec.sensor_size is None:
    raise ValueError("CameraSpec.intrinsics and sensor_size go together (pinhole intrinsics on a sensor)")
  if spec.sensor_size is not None and spec.intrinsics is None and not window:
    raise ValueError("CameraSpec.sensor_size without intrinsics is a conform='fit' window (perspective / orthographic)")
  if window and spec.sensor_size is not None:
    size = check_numbers("CameraSpec.sensor_size", spec.sensor_size)
    if len(size) != _NSIZE or min(size) <= 0.0:
      raise ValueError(f"CameraSpec.sensor_size of a conform='fit' window must be (w, h) > 0, got {spec.sensor_size}")
  if spec.intrinsics is not None and spec.sensor_size is not None:
    intrinsics = check_numbers("CameraSpec.intrinsics", spec.intrinsics)
    sensor = check_numbers("CameraSpec.sensor_size", spec.sensor_size)
    if len(intrinsics) != _NINTRINSICS or len(sensor) != _NSIZE:
      raise ValueError(f"CameraSpec needs intrinsics (fx, fy, cx, cy) and sensor_size (w, h), got {intrinsics}, {sensor}")
    fx, fy = intrinsics[:2]
    # A (0, 0) sensor is MuJoCo's "no intrinsics": the camera uses fovy.
    if any(v != 0.0 for v in sensor) and (min(sensor) <= 0.0 or fx <= 0.0 or fy <= 0.0):
      raise ValueError(f"CameraSpec needs fx, fy > 0 on a sensor of positive size, got {spec.intrinsics} on {spec.sensor_size}")


def check_lens(spec: CameraSpec) -> None:
  """Validates the projection fields of `spec` (fovy, intrinsics, sensor, distortion).

  Raises:
    ValueError: for an invalid field (named in the message).
  """
  if spec.projection not in _PROJECTIONS:
    raise ValueError(f"CameraSpec.projection must be one of {_PROJECTIONS}, got {spec.projection!r}")
  fovy = check_number("CameraSpec.fovy", spec.fovy)
  if (
    fovy <= 0.0
    or (spec.projection == "perspective" and fovy >= _MAX_FOVY_DEG)
    or (spec.projection == "fisheye" and fovy > _MAX_FISHEYE_FOVY_DEG)
  ):
    raise ValueError(
      f"CameraSpec.fovy must be in (0, 180) degrees (fisheye: (0, 360]; orthographic: a view height > 0), got {fovy}"
    )
  _check_sensor(spec)
  if spec.distortion is not None:
    check_numbers("CameraSpec.distortion", spec.distortion)
    if len(spec.distortion) not in _DISTORTION_SIZES.get(spec.projection, ()):
      raise ValueError(
        "CameraSpec: lens distortion needs a perspective camera with OpenCV's (k1, k2, p1, p2[, k3[, k4, k5, k6[, "
        "s1, s2, s3, s4]]]) or a "
        f"fisheye with (k1, k2, k3, k4), got {spec.distortion} on a {spec.projection!r} camera"
      )
  if spec.rays is not None and spec.projection == "fisheye":
    raise ValueError("CameraSpec.rays replace the projection: a fisheye camera has none (its rays are its model's)")
  if spec.outputs is not None and (
    not isinstance(spec.outputs, tuple)
    or not spec.outputs
    or not all(isinstance(k, str) and is_kind(k) for k in spec.outputs)
    or len(set(spec.outputs)) != len(spec.outputs)
  ):
    raise ValueError(f"CameraSpec.outputs must be a non-empty tuple of distinct output names, got {spec.outputs!r}")


def check_frame(spec: CameraSpec) -> None:
  """Validates the clip, resolution, exposure and mount fields of `spec`.

  Raises:
    ValueError: for an invalid field (named in the message).
  """
  check_clip(spec.znear, spec.zfar, "CameraSpec")
  if spec.resolution is not None:
    check_resolution("CameraSpec.resolution", spec.resolution)
  stops = check_number("CameraSpec.exposure", spec.exposure)
  if abs(stops) > _MAX_EXPOSURE_STOPS:
    limit = f"[-{_MAX_EXPOSURE_STOPS:g}, {_MAX_EXPOSURE_STOPS:g}]"
    raise ValueError(f"CameraSpec.exposure must be in {limit} stops (2^exposure a finite float32 > 0), got {stops}")
  if spec.mount is not None and not (isinstance(spec.mount, str) and spec.mount):
    raise ValueError(f"CameraSpec.mount must be a scene camera name, got {spec.mount!r}")
  if spec.local is not None:
    pos, rot = spec.local
    entries = check_numbers("CameraSpec.local rotation", np.ravel(rot).tolist())
    if len(check_numbers("CameraSpec.local position", np.ravel(pos).tolist())) != _NPOS or len(entries) != _NROT:
      raise ValueError("CameraSpec.local must be (position (3,), rotation (3, 3))")
    m = np.reshape(entries, (3, 3))
    # A scaled, sheared or mirrored matrix would scale the rays (another field of view and depth)
    # or flip the image.
    if not (np.allclose(m @ m.T, np.eye(3), atol=_ROTATION_TOLERANCE) and np.linalg.det(m) > 0.0):
      raise ValueError(f"CameraSpec.local rotation must be a rotation matrix (orthonormal, det +1), got {m.tolist()}")


def camera_outputs(names: tuple[str, ...], outputs: Outputs) -> tuple[bool, bool, bool]:
  """(rgb, depth, seg): which output groups a camera of `CameraSpec.outputs` `names` writes
  among the Renderer's requested `outputs`. A camera selects whole groups: every shaded output
  (rgb, hdr, albedo, ...: the camera is shaded or not), depth, the ids (seg and semantic); the
  other outputs (position, uv, material, a geometric normal, motion) are written for every camera
  that writes a group.

  Raises:
    ValueError: for an output the Renderer does not request, part of a group, an ungrouped
      output left out, or no group at all.
  """
  requested = outputs.kinds
  chosen = frozenset(names)
  if chosen - requested:
    raise ValueError(f"CameraSpec.outputs {sorted(chosen - requested)} are not among the Renderer's {sorted(requested)}")
  shaded = frozenset(k for k in requested if construct_named(Outputs, {k: outputs.spec(k)}).shaded)
  groups = (shaded, requested & {"depth"}, requested & _SEG_OUTPUTS)
  for group in groups:
    if chosen & group and not group <= chosen:
      raise ValueError(f"CameraSpec.outputs selects whole groups: {sorted(group)} together, got {sorted(chosen)}")
  every = requested - shaded - _SEG_OUTPUTS - {"depth"}
  if not every <= chosen:
    raise ValueError(
      f"CameraSpec.outputs: {sorted(every - chosen)} are written for every camera (only the shaded outputs, depth "
      "and seg / semantic are selected per camera): list them"
    )
  rgb, depth, seg = (bool(chosen & g) for g in groups)
  if not (rgb or depth or seg):
    raise ValueError(f"CameraSpec.outputs needs a shaded output, depth, seg or semantic, got {sorted(chosen)}")
  return rgb, depth, seg
