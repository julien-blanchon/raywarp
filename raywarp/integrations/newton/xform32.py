"""Warp pose arithmetic of the Newton adapter: (p, q xyzw) transforms composed and inverted in
float32, as the pose kernel does (the numpy helpers are `raywarp.scene.model.xform`)."""

from __future__ import annotations

from typing import Final

import numpy as np
import warp as wp

type Pose = np.ndarray[[7], np.dtype[np.float64]]

IDENTITY: Final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def compose(a: Pose, b: Pose) -> Pose:
  """Warp transform a * b (float32 arithmetic, as the pose kernel)."""
  return np.asarray(wp.transform_multiply(wp.transformf(*a.tolist()), wp.transformf(*b.tolist())), np.float64)


def inverse(t: Pose) -> Pose:
  """Warp transform t^-1."""
  return np.asarray(wp.transform_inverse(wp.transformf(*t.tolist())), np.float64)


# Pose arrays (..., 7) of any leading shape, broadcast against each other.
type Poses = np.ndarray[tuple[int, ...], np.dtype[np.float32]]


def compose_many(a: Poses, b: Poses) -> Poses:
  """`compose` of broadcast pose arrays (..., 7): Warp's `transform_multiply` (quat_rotate(a.q,
  b.p) + a.p, a.q * b.q) in float32 numpy, in the native code's operation order (bit-equal;
  Python scalars keep float32 arithmetic)."""
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  qx, qy, qz, qw = (a[..., 3 + k] for k in range(4))
  x0, x1, x2 = (b[..., k] for k in range(3))
  c = 2.0 * qw * qw - 1.0
  d = 2.0 * (qx * x0 + qy * x1 + qz * x2)
  p0 = x0 * c + qx * d + (qy * x2 - qz * x1) * qw * 2.0 + a[..., 0]
  p1 = x1 * c + qy * d + (qz * x0 - qx * x2) * qw * 2.0 + a[..., 1]
  p2 = x2 * c + qz * d + (qx * x1 - qy * x0) * qw * 2.0 + a[..., 2]
  bx, by, bz, bw = (b[..., 3 + k] for k in range(4))
  q0 = qw * bx + bw * qx + qy * bz - by * qz
  q1 = qw * by + bw * qy + qz * bx - bz * qx
  q2 = qw * bz + bw * qz + qx * by - bx * qy
  q3 = qw * bw - qx * bx - qy * by - qz * bz
  return np.stack([p0, p1, p2, q0, q1, q2, q3], axis=-1)
