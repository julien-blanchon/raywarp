"""An emitter against its fixture (numpy): the side its light leaves the fixture's triangles by
(ray tests from the emitter's surface), and coplanar rectangles merged.

`open_side` gives a tube's light the direction in which its emission leaves the instance's own
geometry (`emitter_shapes.mesh_tube`: a light on the axis would be inside the tube's own
triangles, which shadow it). `open_sides` drops the side of a two-sided emitter that emits into
its own fixture (rays from its face hit the fixture's triangles), and `merge_coplanar` makes one
rectangle of coplanar rectangles that together fill it (a fixture's diffuser split by its frame
bars).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp.scene.lighting.emitter_shapes import Disk, Emitter, Panel, Tube, Vec3, area, frame, min_area_rect, norm3
from raywarp.scene.model.types import Rows

if TYPE_CHECKING:
  from shape_extensions import IntVar

# Points per side of the grid `open_sides` casts rays from, and the share of its rays hitting the
# fixture's triangles above which a side is closed.
SIDE_GRID: Final = 4
BLOCKED_SIDE: Final = 0.9
# `merge_coplanar`: normals within this cosine and planes within this distance (x the larger
# side, plus the panels' thicknesses) are one plane; the merged rectangle must be covered at
# least MERGE_FILL by the panels.
COPLANAR_COS: Final = 0.999
COPLANAR_DIST: Final = 1e-3
MERGE_FILL: Final = 0.75
# Directions across a tube's axis tried by `open_side`, and points along it per direction.
OPEN_DIRECTIONS: Final = 16
OPEN_POINTS: Final = 5
# Determinants below _DET_EPS are singular (ray parallel to the triangle).
_DET_EPS: Final = 1e-12
# Ray-triangle pairs `_hits` tests at once (each pair holds ~10 float64 3-vectors: ~250 MB).
_HIT_PAIRS: Final = 1 << 20


def _hits[V: IntVar](
  points: np.ndarray[[V, 3], np.dtype[np.float64]],
  tris: Rows[np.int64, 3],
  origins: Rows[np.float64, 3],
  *,
  direction: Vec3,
) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(n,) whether each ray (origin, `direction`) hits a triangle (Moller-Trumbore, t > 0).

  The rays are parallel: triangles outside their footprint across `direction` (bounding boxes)
  or behind every origin cannot be hit and are culled first; the rest are tested in chunks of
  at most _HIT_PAIRS ray-triangle pairs (a 1M-triangle fixture stays within that memory)."""
  near = tris[_in_reach(points, tris, origins, direction)]
  hit = np.zeros(len(origins), np.bool_)
  step = max(1, _HIT_PAIRS // max(len(origins), 1))
  for start in range(0, len(near), step):
    hit |= _hits_of(points, near[start : start + step], origins, direction)
  return hit


def _in_reach[V: IntVar](
  points: np.ndarray[[V, 3], np.dtype[np.float64]], tris: Rows[np.int64, 3], origins: Rows[np.float64, 3], direction: Vec3
) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(t,) whether a triangle's bounding box across `direction` overlaps the origins' and any of
  its points lies ahead of an origin (the others cannot be hit by the parallel rays)."""
  e1, e2 = frame(direction)
  scale = float(np.abs(points).max()) if len(points) else 0.0
  margin = 1e-9 * max(scale, float(np.abs(origins).max()), 1.0)  # rounding of the projections
  keep = np.ones(len(tris), np.bool_)
  for axis in (e1, e2):
    o = origins @ axis
    v = (points @ axis)[tris]
    keep &= (v.max(1) >= o.min() - margin) & (v.min(1) <= o.max() + margin)
  return keep & ((points @ direction)[tris].max(1) > float((origins @ direction).min()) - margin)


def _hits_of[V: IntVar](
  points: np.ndarray[[V, 3], np.dtype[np.float64]], tris: Rows[np.int64, 3], origins: Rows[np.float64, 3], direction: Vec3
) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """`_hits` of every ray against every triangle of `tris` (no culling)."""
  a = points[tris[:, 0]]
  e1, e2 = points[tris[:, 1]] - a, points[tris[:, 2]] - a
  p = np.cross(direction, e2)
  det = (e1 * p).sum(1)
  ok = np.abs(det) > _DET_EPS
  inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
  s = origins[:, None, :] - a[None]
  u = (s * p[None]).sum(2) * inv
  qv = np.cross(s, e1[None])
  v = (qv @ direction) * inv
  t = (qv * e2[None]).sum(2) * inv
  return np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0), axis=1)


def open_side[V: IntVar](
  tube: Tube, points: np.ndarray[[V, 3], np.dtype[np.float64]], tris: Rows[np.int64, 3], *, offset: float
) -> Vec3 | None:
  """The unit direction across the axis in which a tube's emission leaves the triangles `tris`
  (its fixture), or None (enclosed, or open all around: no side dominates).

  OPEN_DIRECTIONS directions, rays from OPEN_POINTS points on the tube's surface each (`offset`
  outside it); the direction is the mean of the directions weighted by the rays escaping.
  """
  e1, e2 = frame(tube.axis)
  along = np.linspace(-0.4, 0.4, OPEN_POINTS) * tube.length
  total, free_sum = np.zeros(3), 0.0
  for k in range(OPEN_DIRECTIONS):
    ang = 2.0 * math.pi * k / OPEN_DIRECTIONS
    d = e1 * math.cos(ang) + e2 * math.sin(ang)
    origins = tube.centre + np.outer(along, tube.axis) + d * (tube.radius + offset)
    free = 1.0 - float(_hits(points, tris, origins, direction=d).mean())
    total, free_sum = total + d * free, free_sum + free
  norm = norm3(total)
  # A free half circle gives |total| ~ 0.64 x free_sum, a free full circle 0.
  if free_sum <= 0.0 or norm < 0.25 * free_sum:
    return None
  return total / norm


def _face_grid(e: Emitter) -> Rows[np.float64, 3]:
  """SIDE_GRID x SIDE_GRID points of an emitter's mid-plane, inside its face (cell centres)."""
  t = (np.arange(SIDE_GRID) + 0.5) / SIDE_GRID * 2.0 - 1.0
  a, b = (g.ravel() for g in np.meshgrid(t, t))
  if isinstance(e, Disk):
    e1, e2 = frame(e.normal)
    inside = a * a + b * b <= 1.0
    a, b = a[inside], b[inside]
    return e.centre + np.outer(a * e.radius, e1) + np.outer(b * e.radius, e2)
  e2 = np.cross(e.normal, e.axis)
  return e.centre + np.outer(a * e.half[0], e.axis) + np.outer(b * e.half[1], e2)


def open_sides[V: IntVar](
  e: Emitter, points: np.ndarray[[V, 3], np.dtype[np.float64]], tris: Rows[np.int64, 3], *, offset: float
) -> Emitter:
  """Two-sided `e` without the side that emits into its fixture `tris`: rays from its face grid
  (`_face_grid`, `offset` beyond the face) along that side's normal hit the fixture at least
  BLOCKED_SIDE of the time (a diffuser's back facing its housing). A one-sided emitter, or one
  open (or closed) on both sides, is returned unchanged."""
  if e.side != 0 or len(tris) == 0:
    return e
  grid = _face_grid(e)
  closed = []
  for side in (1, -1):
    d = e.normal * side
    origins = grid + d * (e.half_thickness + offset)
    closed.append(float(_hits(points, tris, origins, direction=d).mean()) >= BLOCKED_SIDE)
  if closed[0] == closed[1]:
    return e
  return dataclasses.replace(e, side=-1 if closed[0] else 1)


def _corners(p: Panel) -> Rows[np.float64, 3]:
  """The four corners of a panel's mid-plane rectangle."""
  e2 = np.cross(p.normal, p.axis)
  signs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], np.float64)
  return p.centre + np.outer(signs[:, 0] * p.half[0], p.axis) + np.outer(signs[:, 1] * p.half[1], e2)


def _merged(a: Panel, b: Panel) -> Panel | None:
  """The rectangle covering coplanar panels `a` and `b` emitting to the same side, or None (other
  side, plane or normal)."""
  cos = float(a.normal @ b.normal)
  side_b = b.side if cos > 0 else -b.side
  if abs(cos) < COPLANAR_COS or side_b != a.side:
    return None
  extent = float(max(a.half.max(), b.half.max()))
  if abs(float((b.centre - a.centre) @ a.normal)) > COPLANAR_DIST * extent + a.half_thickness + b.half_thickness:
    return None
  e1, e2 = frame(a.normal)
  q = np.concatenate([_corners(a), _corners(b)]) - a.centre
  centre2, u2, half = min_area_rect(np.stack([q @ e1, q @ e2], 1))
  return Panel(
    centre=a.centre + e1 * centre2[0] + e2 * centre2[1],
    normal=a.normal,
    axis=e1 * u2[0] + e2 * u2[1],
    half=half,
    half_thickness=max(a.half_thickness, b.half_thickness),
    side=a.side,
  )


def merge_coplanar(items: list[tuple[Emitter, Vec3]]) -> list[tuple[Emitter, Vec3]]:
  """(emitter, radiance scale) list with coplanar rectangles merged pairwise (the best fill first)
  while the merged rectangle stays at least MERGE_FILL covered by the source rectangles it holds
  (their summed area, carried through repeated merges: a merged rectangle's own area would count
  its gaps as cover); a merged rectangle emits their power (scales area-weighted over its area).
  Disks and the other emitters are kept."""
  out = [(e, k, area(e) if isinstance(e, Panel) else 0.0) for e, k in items]
  while True:
    best: tuple[float, int, int, Panel] | None = None
    for i, (a, _, cover_a) in enumerate(out):
      for j in range(i + 1, len(out)):
        b, _, cover_b = out[j]
        if not isinstance(a, Panel) or not isinstance(b, Panel):
          continue
        m = _merged(a, b)
        fill = 0.0 if m is None else (cover_a + cover_b) / max(area(m), 1e-30)
        if m is not None and fill >= MERGE_FILL and (best is None or fill > best[0]):
          best = (fill, i, j, m)
    if best is None:
      return [(e, k) for e, k, _ in out]
    _, i, j, panel = best
    (a, ka, cover_a), (b, kb, cover_b) = out[i], out[j]
    scale = (ka * area(a) + kb * area(b)) / area(panel)
    out = [item for k, item in enumerate(out) if k not in (i, j)] + [(panel, scale, cover_a + cover_b)]
