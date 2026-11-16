"""Vectorised polygon-mesh helpers shared by adapters (numpy only, no Python loop per face)."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from raywarp.scene.model.types import Column, Rows

# Corners of a triangle (faces of this many corners need no fan).
_TRIANGLE: Final = 3


def triangulate(
  counts: npt.ArrayLike,
  indices: npt.ArrayLike,
  holes: npt.ArrayLike | None = None,
  *,
  left_handed: bool = False,
) -> tuple[Rows[np.int64, 3], Rows[np.int64, 3], Column[np.int64]]:
  """Fan-triangulates polygons given as (face vertex counts, face vertex indices).

  Args:
    counts: (F,) vertices per face; faces with fewer than 3 are skipped.
    indices: (sum(counts),) vertex index of every face corner.
    holes: indices of faces to skip (USD holeIndices); as in Hydra, an index that names no face
      (negative or >= F) skips nothing.
    left_handed: reverse the winding (USD orientation = leftHanded) to counter-clockwise.

  Returns:
    (tris (T, 3) vertex indices, corners (T, 3) face-corner indices (for faceVarying data),
    tri_face (T,) source face of each triangle).
  """
  counts = np.asarray(counts, dtype=np.int64)
  indices = np.asarray(indices, dtype=np.int64)
  starts = np.zeros(len(counts), dtype=np.int64)
  if len(counts) > 1:
    np.cumsum(counts[:-1], out=starts[1:])
  ntri = np.maximum(counts - 2, 0)
  hole_ids = np.zeros(0, np.int64) if holes is None else np.asarray(holes, dtype=np.int64).reshape(-1)
  hole_ids = hole_ids[(hole_ids >= 0) & (hole_ids < len(counts))]  # a negative index must not wrap
  ntri[hole_ids] = 0
  if np.all(counts == _TRIANGLE) and not len(hole_ids):
    corners = np.arange(3 * len(counts)).astype(np.int64).reshape(-1, 3)
    tri_face = np.arange(len(counts)).astype(np.int64)
  else:
    tri_face = np.repeat(np.arange(len(counts)).astype(np.int64), ntri)
    first = np.cumsum(ntri) - ntri
    k = np.arange(len(tri_face)).astype(np.int64) - first[tri_face]
    base = starts[tri_face]
    corners = np.stack([base, base + k + 1, base + k + 2], axis=1)
  if left_handed:
    corners = corners[:, [0, 2, 1]]
  return indices[corners], corners, tri_face


# Welding tolerance relative to a point set's extent: float32 keeps ~6e-8 of relative precision, so
# seam copies of one authored position (computed apart, e.g. cos 0 vs cos 2 pi) land well within it,
# while distinct vertices of one mesh are never a millionth of its size apart in practice.
WELD_TOLERANCE: Final = 1e-6


def weld(points: Rows[np.float32, 3] | Rows[np.float64, 3]) -> tuple[Column[np.int64], Column[np.int64]]:
  """(representative, inverse): one source index per welded position and each point's welded index.

  Points are welded when they round to the same cell of `WELD_TOLERANCE` x the set's largest
  extent (exact equality would leave seams whose copies differ by rounding open).
  """
  p = np.asarray(points, np.float64)
  extent = float(np.ptp(p, axis=0).max()) if len(p) else 0.0
  keys = np.round(p / max(extent * WELD_TOLERANCE, 1e-12)).astype(np.int64)
  _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
  return first.astype(np.int64), inverse.reshape(-1).astype(np.int64)


def smooth_normals(
  points: Rows[np.float32, 3] | Rows[np.float64, 3], tris: Rows[np.int32, 3] | Rows[np.int64, 3]
) -> Rows[np.float32, 3]:
  """Area-weighted vertex normals (what Storm shows for subdivision meshes without normals)."""
  p = points.astype(np.float64)
  # Unnormalised face normals: their length is twice the triangle's area.
  fn = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
  normals = np.zeros((len(points), 3), dtype=np.float64)
  flat = tris.reshape(-1)
  for axis in range(3):
    normals[:, axis] = np.bincount(flat, weights=np.repeat(fn[:, axis], 3), minlength=len(points))
  length = np.linalg.norm(normals, axis=1, keepdims=True)
  return (normals / np.maximum(length, 1e-20)).astype(np.float32)


def cone_mesh(
  radius: float, height: float, segments: int = 32
) -> tuple[Rows[np.float32, 3], Rows[np.int32, 3], Rows[np.float32, 3], Rows[np.int32, 3]]:
  """A closed cone along +Z centred at the origin (apex at +height/2), USD Cone's shape.

  Returns:
    (points, tris, normals, normal_idx): corner-indexed normals, smooth on the side, flat on the
    base.
  """
  angles = np.arange(segments) * (2.0 * math.pi / segments)
  ring = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(segments, -0.5 * height)], axis=1)
  points = np.concatenate([ring, [[0.0, 0.0, 0.5 * height]], [[0.0, 0.0, -0.5 * height]]]).astype(np.float32)
  apex, centre = segments, segments + 1
  i = np.arange(segments)
  j = (i + 1) % segments
  side = np.stack([i, j, np.full(segments, apex)], axis=1)
  base = np.stack([j, i, np.full(segments, centre)], axis=1)
  slope = radius / max(height, 1e-12)
  side_normals = np.stack([np.cos(angles), np.sin(angles), np.full(segments, slope)], axis=1)
  side_normals /= np.linalg.norm(side_normals, axis=1, keepdims=True)
  normals = np.concatenate([side_normals, [[0.0, 0.0, -1.0]]]).astype(np.float32)
  # Apex corners take the normal halfway between their two ring normals.
  mid = np.stack([np.cos(angles + math.pi / segments), np.sin(angles + math.pi / segments), np.full(segments, slope)], axis=1)
  mid /= np.linalg.norm(mid, axis=1, keepdims=True)
  normals = np.concatenate([normals, mid.astype(np.float32)])
  side_idx = np.stack([i, j, segments + 1 + i], axis=1)
  base_idx = np.full((segments, 3), segments)
  tris = np.concatenate([side, base]).astype(np.int32)
  normal_idx = np.concatenate([side_idx, base_idx]).astype(np.int32)
  return points, tris, normals, normal_idx


def axis_rotation(axis: str) -> np.ndarray[[3, 3], np.dtype[np.float64]]:
  """Rotation mapping local +Z to the USD `axis` (a cyclic permutation, right-handed)."""
  if axis == "X":
    return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
  if axis == "Y":
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
  return np.eye(3)
