"""The viewers' Z-up fly camera and the views that frame a scene.

Conventions (every raywarp scene is Z-up: the USD importer turns Y-up stages): yaw and pitch in
radians, yaw 0 looks along +Y, positive yaw turns toward +X, positive pitch looks up, no roll;
`fov` is the vertical field of view in degrees. A camera frame looks along its -Z with +Y up.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np

import raywarp as rw
from raywarp.scene.model.xform import quat_from_matrix

# A free view: eye, target (world units, Z up) and vertical field of view [degrees].
type View = tuple[tuple[float, float, float], tuple[float, float, float], float]
type Vec3 = np.ndarray[tuple[int], np.dtype[np.float64]]

# Pitch limit [rad]: just short of straight up / down (the yaw is undefined there).
MAX_PITCH: Final = 1.569
FOV_RANGE: Final = (5.0, 150.0)


@dataclasses.dataclass(frozen=True)
class Camera:
  """A fly camera: position in world units, yaw / pitch in radians, vertical FOV in degrees."""

  position: tuple[float, float, float] = (3.0, -5.0, 2.5)
  yaw: float = -0.54
  pitch: float = -0.26
  fov: float = 50.0

  @classmethod
  def parse(cls, value: object) -> Camera:
    """The camera of a JSON object {position, yaw, pitch, fov}.

    Raises:
      ValueError: for missing or extra keys, non-finite numbers, a pitch beyond `MAX_PITCH` or a
        FOV outside `FOV_RANGE`.
    """
    if not isinstance(value, dict) or set(value) != {"position", "yaw", "pitch", "fov"}:
      raise ValueError("camera requires position, yaw, pitch, fov")
    pos = value["position"]
    if not isinstance(pos, list | tuple) or len(pos) != 3:
      raise ValueError("position must have three coordinates")
    numbers = [*pos, value["yaw"], value["pitch"], value["fov"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers):
      raise ValueError("camera values must be finite numbers")
    if not -MAX_PITCH <= value["pitch"] <= MAX_PITCH or not FOV_RANGE[0] <= value["fov"] <= FOV_RANGE[1]:
      raise ValueError(f"pitch must be within ±{MAX_PITCH} radians and fov within {list(FOV_RANGE)}")
    x, y, z = (float(v) for v in pos)
    return cls((x, y, z), float(value["yaw"]), float(value["pitch"]), float(value["fov"]))

  def forward(self) -> Vec3:
    """The unit view direction."""
    cp = math.cos(self.pitch)
    return np.array([math.sin(self.yaw) * cp, math.cos(self.yaw) * cp, math.sin(self.pitch)])

  def pose(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """(7,) float32 camera transform: position, then the rotation quaternion (x, y, z, w) taking
    the camera frame (x right, y up, looking along -z) to the world."""
    forward = self.forward()
    right = np.array([math.cos(self.yaw), -math.sin(self.yaw), 0.0])
    up = np.cross(right, forward)
    return np.concatenate([np.asarray(self.position), quat_from_matrix(np.column_stack((right, up, -forward)))]).astype(
      np.float32
    )

  def view(self) -> View:
    """The (eye, target, fovy) of the camera (target 1 unit ahead)."""
    eye = np.asarray(self.position, float)
    return as_view(eye, eye + self.forward(), self.fov)


def norm(v: Vec3) -> float:
  """The Euclidean length of a vector."""
  return math.hypot(*(float(c) for c in v.tolist()))


def fit(center: Vec3, radius: float) -> Camera:
  """A camera framing a bounding sphere with a margin, from above and in front."""
  radius = max(float(radius), 0.25)
  direction = np.array([0.6, -1.0, 0.5])
  direction = direction / norm(direction)
  eye = np.asarray(center, float) + direction * radius / math.sin(math.radians(22)) * 1.15
  forward = -direction
  x, y, z = (float(v) for v in eye)
  return Camera((x, y, z), math.atan2(forward[0], forward[1]), math.asin(forward[2]))


def as_view(eye: Vec3, target: Vec3, fovy: float) -> View:
  """The plain-float view tuple of `eye`, `target` and `fovy`."""
  e, t = [float(v) for v in eye], [float(v) for v in target]
  return ((e[0], e[1], e[2]), (t[0], t[1], t[2]), float(fovy))


def view_camera(view: View) -> Camera:
  """The fly camera of an (eye, target, fovy) view (pitch clamped to `MAX_PITCH`).

  Raises:
    ValueError: when eye and target coincide.
  """
  eye, target, fovy = view
  forward = np.asarray(target, float) - np.asarray(eye, float)
  length = norm(forward)
  if length < 1e-9:
    raise ValueError("eye and target coincide")
  forward = forward / length
  pitch = max(-MAX_PITCH, min(MAX_PITCH, math.asin(float(forward[2]))))
  x, y, z = (float(v) for v in eye)
  return Camera((x, y, z), math.atan2(forward[0], forward[1]), pitch, float(fovy))


def primitive_half(kind: int, size: np.ndarray[tuple[int], np.dtype[np.float32]]) -> Vec3:
  """Half extents of a public primitive (MuJoCo size convention)."""
  if kind in (rw.scene.InstanceKind.BOX, rw.scene.InstanceKind.ELLIPSOID):
    return np.asarray(size, float)
  if kind == rw.scene.InstanceKind.CYLINDER:
    return np.array([size[0], size[0], size[1]], float)
  if kind == rw.scene.InstanceKind.CAPSULE:
    return np.array([size[0], size[0], size[0] + size[1]], float)
  return np.full(3, max(float(size[0]), 0.01))


def bounds(host: rw.Scene) -> tuple[Vec3, Vec3] | None:
  """(low, high) corners of the transformed primitive / mesh bounds, infinite planes excluded
  (None: nothing bounded)."""
  lows, highs = [], []
  inst = host.instances
  for kind, size, mesh, matrix in zip(inst.kind, inst.size, inst.mesh, inst.xform, strict=True):
    if kind == rw.scene.InstanceKind.PLANE:
      continue
    if mesh >= 0:
      box = host.meshes.aabb[mesh]
      center, half = (box[0] + box[1]) * 0.5, (box[1] - box[0]) * 0.5
    else:
      center, half = np.zeros(3), primitive_half(int(kind), size)
    lows.append(matrix[:, :3] @ center + matrix[:, 3] - np.abs(matrix[:, :3]) @ half)
    highs.append(matrix[:, :3] @ center + matrix[:, 3] + np.abs(matrix[:, :3]) @ half)
  if not lows:
    return None
  return np.min(lows, axis=0).astype(float), np.max(highs, axis=0).astype(float)


def home_view(host: rw.Scene) -> View:
  """A view framing the scene's bounds (`bounds`)."""
  box = bounds(host)
  if box is None:
    return fit(np.zeros(3), 2.0).view()
  lo, hi = box
  return fit((lo + hi) * 0.5, norm(hi - lo) * 0.5).view()


def authored_view(host: rw.Scene) -> View | None:
  """The view of the scene's first perspective camera (None: it has none)."""
  cams = host.cameras
  for k in range(host.ncamera):
    if (
      int(cams.projection[k]) != rw.scene.Projection.PERSPECTIVE
      or float(cams.focal_length[k]) <= 0
      or float(cams.vertical_aperture[k]) <= 0
    ):
      continue
    xform = np.asarray(cams.xform[k], float)
    fovy = math.degrees(2 * math.atan(float(cams.vertical_aperture[k]) / (2 * float(cams.focal_length[k]))))
    eye = xform[:, 3]
    return as_view(eye, eye - xform[:, 2] / max(norm(xform[:, 2]), 1e-12), min(max(fovy, FOV_RANGE[0]), FOV_RANGE[1]))
  return None


# The largest image a viewer streams (a browser frame, bounded per request).
MAX_SIZE: Final = (1920, 1080)


def resolution(width: object, height: object) -> tuple[int, int]:
  """A viewer image size (bools rejected).

  Raises:
    ValueError: for a width outside [16, 1920] or a height outside [16, 1080], or a non-integer.
  """
  if type(width) is not int or type(height) is not int or not 16 <= width <= MAX_SIZE[0] or not 16 <= height <= MAX_SIZE[1]:
    raise ValueError(f"resolution must be integer width [16, {MAX_SIZE[0]}], height [16, {MAX_SIZE[1]}]")
  return width, height
