"""`scene.model.xform.rigid` / `quats_from_matrices`: batched, bit-equal to the per-matrix
reference."""

from __future__ import annotations

import math

import numpy as np

from raywarp.scene.model import xform


def _reference_quat(r: np.ndarray) -> np.ndarray:
  """Warp's `quat_from_matrix` in float64 Python scalars (the former per-matrix code)."""
  m = [[float(r[i, j]) for j in range(3)] for i in range(3)]
  tr = m[0][0] + m[1][1] + m[2][2]
  q = [0.0, 0.0, 0.0, 0.0]
  if tr >= 0.0:
    h = math.sqrt(tr + 1.0)
    q[3], h = 0.5 * h, 0.5 / h
    q[0], q[1], q[2] = (m[2][1] - m[1][2]) * h, (m[0][2] - m[2][0]) * h, (m[1][0] - m[0][1]) * h
  else:
    diag = [m[0][0], m[1][1], m[2][2]]
    i = diag.index(max(diag))
    j, k = (i + 1) % 3, (i + 2) % 3
    h = math.sqrt((m[i][i] - (m[j][j] + m[k][k])) + 1.0)
    q[i], h = 0.5 * h, 0.5 / h
    q[j] = (m[i][j] + m[j][i]) * h
    q[k] = (m[k][i] + m[i][k]) * h
    q[3] = (m[k][j] - m[j][k]) * h
  norm = math.sqrt(math.fsum(c * c for c in q))
  return np.array([c / norm for c in q])


def test_batched_quaternions_and_poses_equal_the_per_matrix_ones():
  """Random rotations (both trace branches) and the axis flips (diagonal ties): every quaternion
  bit-equal; `rigid` of (n, 3, 4) and (n, 4, 4) transforms with scale and shear keeps the
  translation and the polar rotation's quaternion."""
  rng = np.random.default_rng(0)
  a = rng.normal(size=(2000, 3, 3))
  r, _ = xform.polar(a)
  flips = np.array(
    [
      np.eye(3),
      np.diag(np.array([1.0, -1, -1])),
      np.diag(np.array([-1.0, 1, -1])),
      np.diag(np.array([-1.0, -1, 1])),
      [[0, 1, 0], [1, 0, 0], [0, 0, -1.0]],
    ]
  )
  r = np.concatenate([r, flips])
  assert np.any(np.trace(r, axis1=1, axis2=2) < 0) and np.any(np.trace(r, axis1=1, axis2=2) >= 0)
  np.testing.assert_array_equal(xform.quats_from_matrices(r), np.array([_reference_quat(x) for x in r]))
  m = np.concatenate([a[:50], rng.normal(size=(50, 3, 1))], axis=2)
  poses = xform.rigid(m)
  np.testing.assert_array_equal(poses[:, :3], m[:, :, 3])
  np.testing.assert_array_equal(poses[:, 3:], xform.quats_from_matrices(xform.polar(a[:50])[0]))
  square = np.concatenate([m, np.tile([0.0, 0.0, 0.0, 1.0], (50, 1, 1))], axis=1)
  np.testing.assert_array_equal(xform.rigid(square), poses)
