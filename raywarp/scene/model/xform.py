"""NumPy transform helpers shared by the host scene, the adapters and the upload.

Conventions (`types.py`): (3, 4) matrices `[A | t]` map local points as `A @ p + t` (column
vectors); quaternions are (x, y, z, w), Warp's order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp.scene.model.types import Array

if TYPE_CHECKING:
  from shape_extensions import IntVar

# An affine transform: None (identity), (3, 4), or homogeneous (4, 4); nested sequences allowed.
type XformLike = np.ndarray | Sequence[Sequence[float]] | None

type Mat3 = np.ndarray[[3, 3], np.dtype[np.float64]]
type Quat = np.ndarray[[4], np.dtype[np.float64]]
type Vector3 = np.ndarray[[3], np.dtype[np.float64]] | np.ndarray[[3], np.dtype[np.float32]]


# `light_frame` completes a frame with +Z unless the axis is within acos(0.9) of it.
_HELPER_COS: Final = 0.9


def as_xform(xform: XformLike) -> np.ndarray[[3, 4], np.dtype[np.float32]]:
  """A (3, 4) float32 affine transform from None, (3, 4), or a homogeneous (4, 4) matrix.

  Raises:
    ValueError: for another matrix shape, a batch, a last row other than (0, 0, 0, 1),
      or values that are not finite when stored as float32.
  """
  if xform is None:
    return np.ascontiguousarray(np.eye(4)[:3], dtype=np.float32)
  m = np.asarray(xform, dtype=np.float64)
  if m.shape not in ((3, 4), (4, 4)):
    raise ValueError(f"xform must have shape (3, 4) or (4, 4), got {m.shape}")
  if m.shape == (4, 4) and not np.array_equal(m[3], (0.0, 0.0, 0.0, 1.0)):
    raise ValueError("xform must have affine last row (0, 0, 0, 1)")
  with np.errstate(over="ignore"):
    stored = np.ascontiguousarray(m[:3, :], dtype=np.float32)
  if not np.all(np.isfinite(stored)):
    raise ValueError("xform must contain finite float32 values")
  return stored


def polar[N: IntVar](
  a: np.ndarray[[N, 3, 3], np.dtype[np.float32]] | np.ndarray[[N, 3, 3], np.dtype[np.float64]],
) -> tuple[np.ndarray[[N, 3, 3], np.dtype[np.float64]], np.ndarray[[N, 3, 3], np.dtype[np.float64]]]:
  """(rotations R, stretches S) with A = R S and det R = +1 (a reflection stays in S).

  Polar decomposition through the SVD A = U Σ Vᵀ: R = U Vᵀ, with U's last column negated
  where that product is a reflection.
  """
  # Batched SVD (pyrefly-numpy-stubs type only the 2-D one: the batch is typed `Array`).
  batch: Array = a.astype(np.float64)
  svd: tuple[Array, Array, Array] = np.linalg.svd(batch, full_matrices=False)
  u, _, vt = svd
  r = u @ vt
  flip = np.linalg.det(r) < 0.0
  if np.any(flip):
    u = u.copy()
    u[flip, :, 2] *= -1.0
    r = u @ vt
  s = np.transpose(r, (0, 2, 1)) @ a
  return r, s


def orthonormalize(
  m: np.ndarray[[4, 4], np.dtype[np.float64]],
) -> tuple[np.ndarray[[3, 4], np.dtype[np.float64]], np.ndarray[[3], np.dtype[np.float64]]]:
  """(orthonormal (3, 4) transform, per-axis scales) of a (4, 4) matrix (columns normalised, then
  the nearest orthonormal matrix: a mirroring transform stays mirrored)."""
  lin = m[:3, :3]
  scales = np.linalg.norm(lin, axis=0)
  rot = lin / np.maximum(scales, 1e-30)
  u, _, vt = np.linalg.svd(rot, full_matrices=False)  # square: the same as the full SVD
  out = np.zeros((3, 4))
  out[:, :3], out[:, 3] = u @ vt, m[:3, 3]
  return out, scales


def quat_to_matrix(q: Sequence[float] | Quat) -> Mat3:
  """Rotation matrix of a unit quaternion (x, y, z, w)."""
  x, y, z, w = (float(v) for v in q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def quat_from_matrix(r: Mat3) -> Quat:
  """Unit quaternion (x, y, z, w) of a rotation matrix (`quats_from_matrices` of one)."""
  return quats_from_matrices(np.asarray(r, np.float64)[None])[0]


def quats_from_matrices[N: IntVar](r: np.ndarray[[N, 3, 3], np.dtype[np.float64]]) -> np.ndarray[[N, 4], np.dtype[np.float64]]:
  """(n, 4) unit quaternions (x, y, z, w) of rotation matrices (Warp's `quat_from_matrix`, in
  float64): from the trace when it is >= 0, else from the largest diagonal entry (the first on
  ties: the best-conditioned component)."""
  m = np.asarray(r, np.float64)
  n = len(m)
  rows = np.arange(n)
  q = np.zeros((n, 4))
  tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
  pos = tr >= 0.0
  h = np.sqrt(np.where(pos, tr, 0.0) + 1.0)
  g = 0.5 / h
  q[pos, 3] = 0.5 * h[pos]
  q[pos, 0] = ((m[:, 2, 1] - m[:, 1, 2]) * g)[pos]
  q[pos, 1] = ((m[:, 0, 2] - m[:, 2, 0]) * g)[pos]
  q[pos, 2] = ((m[:, 1, 0] - m[:, 0, 1]) * g)[pos]
  i = np.argmax(np.diagonal(m, axis1=1, axis2=2), axis=1)
  j, k = (i + 1) % 3, (i + 2) % 3
  mi, mj, mk = m[rows, i, i], m[rows, j, j], m[rows, k, k]
  h = np.sqrt(np.where(pos, 1.0, (mi - (mj + mk)) + 1.0))
  g = 0.5 / h
  neg = rows[~pos]
  q[neg, i[neg]] = 0.5 * h[neg]
  q[neg, j[neg]] = ((m[rows, i, j] + m[rows, j, i]) * g)[neg]
  q[neg, k[neg]] = ((m[rows, k, i] + m[rows, i, k]) * g)[neg]
  q[neg, 3] = ((m[rows, k, j] - m[rows, j, k]) * g)[neg]
  return q / np.sqrt(_compensated_sum(q * q))[:, None]


def _compensated_sum[N: IntVar, K: IntVar](
  x: np.ndarray[[N, K], np.dtype[np.float64]],
) -> np.ndarray[[N], np.dtype[np.float64]]:
  """(n,) row sums of `x` in order, Neumaier-compensated (bit-equal to Python's float `sum`)."""
  total = np.zeros(len(x))
  comp = np.zeros(len(x))
  for col in x.T:
    t = total + col
    comp += np.where(np.abs(total) >= np.abs(col), (total - t) + col, (col - t) + total)
    total = t
  return total + comp


def pose_matrix(pose: Sequence[float] | np.ndarray[[7], np.dtype[np.float64]]) -> np.ndarray[[3, 4], np.dtype[np.float64]]:
  """(3, 4) transform of a rigid pose (position, quaternion x y z w)."""
  p = np.asarray(pose, np.float64)
  return np.concatenate([quat_to_matrix(p[3:7]), p[:3, None]], axis=1)


def light_frame(
  direction: Sequence[float] | Vector3, pos: Sequence[float] | Vector3
) -> np.ndarray[[3, 4], np.dtype[np.float64]]:
  """(3, 4) transform at `pos` whose -Z axis is `direction` (the way host lights emit).

  Raises:
    ValueError: for a zero or non-finite `direction`.
  """
  d = np.asarray(direction, np.float64)
  length = math.sqrt(float(d @ d))
  if not 0.0 < length < math.inf:
    raise ValueError(f"a light direction must be non-zero and finite, got {d.tolist()}")
  z = -d / length
  # Any axis not parallel to z completes the frame.
  helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < _HELPER_COS else np.array([1.0, 0.0, 0.0])
  x = np.cross(helper, z)
  x = x / math.sqrt(float(x @ x))
  return np.concatenate([np.stack([x, np.cross(z, x), z], axis=1), np.asarray(pos, np.float64)[:, None]], axis=1)


def rigid[N: IntVar, R: IntVar](m: np.ndarray[[N, R, 4], np.dtype[np.float64]]) -> np.ndarray[[N, 7], np.dtype[np.float64]]:
  """(n, 7) rigid poses (position, quaternion x y z w) of (n, 3, 4) or (n, 4, 4) transforms (scale
  and shear dropped): one batched polar decomposition."""
  a = np.asarray(m, np.float64)
  r, _ = polar(a[:, :3, :3])
  return np.concatenate([a[:, :3, 3], quats_from_matrices(r)], axis=1)
