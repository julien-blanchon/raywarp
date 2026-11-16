"""Shapes fitted to emitting triangles (numpy; in a mesh's frame or world space): flat rectangles,
flat disks, tubes; `placed` moves a fit by a similarity transform.

`raywarp.scene.lighting.emissive` turns them into lights:

* a flat rectangle (`mesh_panel`): the points' oriented bounding box is thin and the faces cover
  `min_fill` of its rectangle;
* a flat disk (`mesh_disk`): thin, square bounding rectangle, the faces cover `min_fill` of the
  disk through the farthest point (a round downlight's diffuser);
* a tube (`mesh_tube`): elongated along its principal axis, side faces (normals across the axis)
  covering `min_fill` of the cylinder through the farthest point, their centroids near that
  radius (a fluorescent tube; a square bar is no tube). Its light is the tube's silhouette (2 r x
  l) facing the direction in which its emission leaves the instance's own geometry
  (`emitter_fixture.open_side`; none when no side dominates, e.g. a bare tube).

`components` splits emitting faces into connected parts (points welded), one shape per part.
`emitter_fixture` relates the fits to their fixture (open sides, coplanar rectangles merged).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp.scene.model import geometry
from raywarp.scene.model.types import Column, Rows

if TYPE_CHECKING:
  from shape_extensions import IntVar

type Vec3 = np.ndarray[[3], np.dtype[np.float64]]
type Vec2 = np.ndarray[[2], np.dtype[np.float64]]

# Points of a hull edge and a polygon; fewest points of a tube; a helper axis is used when
# the normal's cosine with it is below _HELPER_COS.
_SEGMENT: Final = 2
_TRIANGLE: Final = 3
_TUBE_POINTS: Final = 6
_HELPER_COS: Final = 0.9


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Panel:
  """A flat emitting rectangle in world space.

  Attributes:
    centre: centre of the rectangle (mid-thickness).
    normal: unit normal.
    axis: unit direction of the first side (`half[0]`).
    half: half extents along `axis` and `normal x axis`.
    half_thickness: half the panel's extent along `normal`.
    side: 1 (emits along `normal`), -1 (against it) or 0 (both: two-sided).
  """

  centre: Vec3
  normal: Vec3
  axis: Vec3
  half: Vec2
  half_thickness: float
  side: int


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Disk:
  """A flat emitting disk in world space (`Panel`'s conventions: centre, normal, side)."""

  centre: Vec3
  normal: Vec3
  radius: float
  half_thickness: float
  side: int


type Emitter = Panel | Disk


def area(e: Emitter) -> float:
  """Emitting area of one side [m^2]."""
  if isinstance(e, Disk):
    return math.pi * e.radius * e.radius
  return 4.0 * float(e.half[0] * e.half[1])


def face_areas(points: Rows[np.float64, 3], tris: Rows[np.int32, 3] | Rows[np.int64, 3]) -> Column[np.float64]:
  """(T,) area of each triangle of `tris` over `points` [m^2]."""
  a, b, c = (points[tris[:, k]] for k in range(3))
  return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def norm3(v: Vec3) -> float:
  """Euclidean length of a 3-vector (as `np.linalg.norm`)."""
  return math.sqrt(float(v @ v))


def _cross2(a: Vec2, b: Vec2) -> float:
  """Z of the cross product of two 2D vectors."""
  return float(a[0] * b[1] - a[1] * b[0])


def _hull_half(points: Rows[np.float64, 2]) -> list[Vec2]:
  """One half of the convex hull of lexicographically sorted points (monotone chain)."""
  out: list[Vec2] = []
  for q in points:
    while len(out) >= _SEGMENT and _cross2(out[-1] - out[-2], q - out[-2]) <= 0:
      out.pop()
    out.append(q)
  return out


def _outer(p: Rows[np.float64, 2]) -> Rows[np.float64, 2]:
  """`p` without the points strictly inside the polygon of its extreme points along x, y, x + y
  and x - y (Akl-Toussaint): they are not hull vertices, so the hull is unchanged; a margin of
  1e-9 x the extent squared keeps points near an edge (rounding)."""
  keys = (p[:, 0], p[:, 1], p[:, 0] + p[:, 1], p[:, 0] - p[:, 1])
  ext = np.unique(np.array([np.argmin(k) for k in keys] + [np.argmax(k) for k in keys]))
  if len(ext) < _TRIANGLE:
    return p
  poly = p[ext]
  c = poly.mean(0)
  poly = poly[np.argsort(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]))]  # counter-clockwise
  edge = np.roll(poly, -1, axis=0) - poly
  rel = p[:, None, :] - poly[None]  # (n, k, 2)
  cross = edge[None, :, 0] * rel[..., 1] - edge[None, :, 1] * rel[..., 0]
  margin = 1e-9 * float(np.ptp(p, axis=0).max()) ** 2
  return p[~np.all(cross > margin, axis=1)]


def min_area_rect(pts: Rows[np.float64, 2]) -> tuple[Vec2, Vec2, Vec2]:
  """(centre, unit axis u, half extents) of the minimum-area rectangle of 2D points."""
  p = np.unique(np.asarray(pts, np.float64), axis=0)
  if len(p) < _TRIANGLE:
    lo, hi = p.min(0), p.max(0)
    return 0.5 * (lo + hi), np.array([1.0, 0.0]), 0.5 * (hi - lo)
  p = _outer(p)
  p = p[np.lexsort((p[:, 1], p[:, 0]))]
  hull = np.array(_hull_half(p)[:-1] + _hull_half(p[::-1])[:-1])
  # The minimum-area rectangle has a side along a hull edge: try every edge direction.
  edges = np.roll(hull, -1, axis=0) - hull
  angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
  best: tuple[float, Vec2, Vec2, Vec2] | None = None
  for a in angles:
    u = np.array([np.cos(a), np.sin(a)])
    v = np.array([-u[1], u[0]])
    x, y = hull @ u, hull @ v
    rect = float((x.max() - x.min()) * (y.max() - y.min()))
    if best is None or rect < best[0]:
      c = u * 0.5 * (x.max() + x.min()) + v * 0.5 * (y.max() + y.min())
      best = (rect, c, u, 0.5 * np.array([x.max() - x.min(), y.max() - y.min()]))
  if best is None:
    raise ValueError("a minimum-area rectangle needs a hull of at least one edge")
  return best[1], best[2], best[3]


def frame(normal: Vec3) -> tuple[Vec3, Vec3]:
  """Two unit vectors completing `normal` to an orthonormal frame."""
  t = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < _HELPER_COS else np.array([0.0, 1.0, 0.0])
  e1 = np.cross(normal, t)
  e1 = e1 / norm3(e1)
  return e1, np.cross(normal, e1)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Flat:
  """A thin triangle set: its minimum-area rectangle, coverage and emitting side."""

  panel: Panel
  cover: float  # faces' area covering the plane, counted on the side that covers most
  radius: float  # farthest point from the rectangle's centre, in the plane


def flat[V: IntVar](
  points: np.ndarray[[V, 3], np.dtype[np.float64]], tris: Rows[np.int64, 3], max_thickness: float
) -> Flat | None:
  """The rectangle of a world-space triangle set thinner than `max_thickness` x its largest side,
  or None."""
  if len(tris) == 0:
    return None
  a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
  cross = np.cross(b - a, c - a)
  if 0.5 * np.linalg.norm(cross, axis=1).sum() <= 0:
    return None
  # Normal: the least-variance axis of the points; then the minimum-area rectangle in its plane.
  centroid = points.mean(0)
  _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
  normal = vt[2] / norm3(vt[2])
  e1, e2 = frame(normal)
  q = points - centroid
  h = q @ normal
  thickness = float(h.max() - h.min())
  plane = np.stack([q @ e1, q @ e2], 1)
  centre2, u2, half = min_area_rect(plane)
  if half.min() <= 0 or thickness > max_thickness * 2.0 * half.max():
    return None
  # Coverage of the plane by the faces on each side (a closed thin box covers it twice).
  signed = 0.5 * (cross @ normal)
  cover = float(max(signed[signed > 0].sum(), -signed[signed < 0].sum()))
  # One-sided when the faces agree on a side (90 % of their signed area), else two-sided.
  total = np.abs(signed).sum()
  side = int(np.sign(signed.sum())) if total > 0 and abs(signed.sum()) > 0.9 * total else 0
  centre = centroid + e1 * centre2[0] + e2 * centre2[1] + normal * (0.5 * (h.max() + h.min()))
  radius = float(np.sqrt(((plane - centre2) ** 2).sum(1).max()))
  return Flat(
    panel=Panel(
      centre=centre, normal=normal, axis=e1 * u2[0] + e2 * u2[1], half=half, half_thickness=0.5 * thickness, side=side
    ),
    cover=cover,
    radius=radius,
  )


def mesh_panel(f: Flat | None, *, min_fill: float) -> Panel | None:
  """The panel of a thin triangle set (`flat`), or None when it is not a flat rectangle."""
  if f is None or f.cover < min_fill * area(f.panel):
    return None
  return f.panel


def mesh_disk(f: Flat | None, *, min_fill: float) -> Disk | None:
  """The disk of a thin triangle set (`flat`), or None when it is not a flat disk."""
  if f is None or f.panel.half.min() < min_fill * f.panel.half.max():
    return None
  p = f.panel
  if f.cover < min_fill * math.pi * f.radius * f.radius:
    return None
  return Disk(centre=p.centre, normal=p.normal, radius=f.radius, half_thickness=p.half_thickness, side=p.side)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Tube:
  """An emitting cylinder in world space: centre, unit axis, radius, length."""

  centre: Vec3
  axis: Vec3
  radius: float
  length: float


def mesh_tube[V: IntVar](
  points: np.ndarray[[V, 3], np.dtype[np.float64]], tris: Rows[np.int64, 3], min_fill: float
) -> Tube | None:
  """The tube of a world-space triangle mesh, or None when it is not a round tube at least two
  diameters long."""
  if len(tris) == 0 or len(points) < _TUBE_POINTS:
    return None
  centroid = points.mean(0)
  _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
  axis = vt[0] / norm3(vt[0])
  q = points - centroid
  t = q @ axis
  radius = float(np.linalg.norm(q - np.outer(t, axis), axis=1).max())
  length = float(t.max() - t.min())
  if radius <= 0 or length < 4.0 * radius:
    return None
  a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
  cross = np.cross(b - a, c - a)
  face_area = 0.5 * np.linalg.norm(cross, axis=1)
  across = np.abs(cross @ axis) < np.maximum(face_area, 1e-30)  # |n . axis| < 0.5 (|cross| = 2 x area)
  if face_area[across].sum() < min_fill * 2.0 * math.pi * radius * length:
    return None
  # Round: the side faces' centroids near the radius (a regular n-gon's at cos(pi / n)).
  mid = (a + b + c)[across] / 3.0 - centroid
  dist = np.linalg.norm(mid - np.outer(mid @ axis, axis), axis=1)
  if float((dist * face_area[across]).sum() / face_area[across].sum()) < 0.85 * radius:
    return None
  return Tube(centre=centroid + axis * (0.5 * (t.max() + t.min())), axis=axis, radius=radius, length=length)


def components(points: Rows[np.float64, 3], tris: Rows[np.int64, 3]) -> list[np.ndarray[[int], np.dtype[np.int64]]]:
  """Face indices of each connected part of a triangle set (coincident points welded: meshes
  split vertices at UV seams)."""
  _, weld = geometry.weld(points)
  v = weld[tris]
  label = np.arange(int(weld.max()) + 1 if len(weld) else 0)
  while True:
    # Every vertex takes the smallest label of its faces, then labels jump to their roots.
    low = np.minimum(np.minimum(label[v[:, 0]], label[v[:, 1]]), label[v[:, 2]])
    new = label.copy()
    for k in range(3):
      np.minimum.at(new, v[:, k], low)
    new = new[new]
    if np.array_equal(new, label):
      break
    label = new
  face_label = label[v[:, 0]]
  return [np.nonzero(face_label == g)[0] for g in np.unique(face_label)]


# A transform is a similarity when its Gram matrix is s^2 I within this relative tolerance; its
# scale is keyed to _SCALE_DIGITS significant digits (fits are shared by equal scales).
_SIMILARITY_TOL: Final = 1e-6
_SCALE_DIGITS: Final = 6


def similarity(xform: np.ndarray[[3, 4], np.dtype[np.float64]]) -> float | None:
  """The scale s of a transform whose linear part is s x a rotation (det > 0), else None."""
  lin = xform[:, :3]
  gram = lin.T @ lin
  s2 = float(np.trace(gram)) / 3.0
  if s2 <= 0.0 or np.linalg.det(lin) <= 0.0 or float(np.abs(gram - s2 * np.eye(3)).max()) > _SIMILARITY_TOL * s2:
    return None
  return float(f"{math.sqrt(s2):.{_SCALE_DIGITS}g}")


def placed(e: Emitter, xform: np.ndarray[[3, 4], np.dtype[np.float64]], scale: float) -> Emitter:
  """Emitter `e` fitted in a mesh's frame, placed by the similarity `xform` of scale `scale`
  (directions rotated and renormalised, lengths scaled)."""
  lin = xform[:, :3]
  rot = lin / scale
  centre = lin @ e.centre + xform[:, 3]
  normal = rot @ e.normal
  normal = normal / norm3(normal)
  if isinstance(e, Disk):
    return Disk(centre=centre, normal=normal, radius=e.radius * scale, half_thickness=e.half_thickness * scale, side=e.side)
  axis = rot @ e.axis
  return Panel(
    centre=centre,
    normal=normal,
    axis=axis / norm3(axis),
    half=e.half * scale,
    half_thickness=e.half_thickness * scale,
    side=e.side,
  )
