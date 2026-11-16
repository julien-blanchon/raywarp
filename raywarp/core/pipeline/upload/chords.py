"""The thickness of closed instances: their mean chord, the path a ray crossing them travels inside
on average (Cauchy's formula, exact for convex bodies under isotropic uniform lines: 4 V / S,
V the enclosed volume, S the surface area). A solid glass without an authored
`transmission_depth` reaches its `transmission_color` over the median mean chord of its
instances (`pipeline/upload/optics.py`). Closed: a primitive but the (open) plane, or a mesh whose
every edge joins two triangles.

Mean chords of the primitives (MuJoCo sizes, `InstanceKind`; the instance's scale applied per
axis): a box of side lengths a, b, c: 2 abc / (ab + bc + ca) (a thin pane of thickness t: about 2t,
its rays at all angles); a sphere of radius r: 4r / 3; a cylinder of radius r and height h:
2rh / (r + h); a capsule: 4 V / S of its cylinder and the two half spheres; an ellipsoid: 4 V / S,
S by Thomsen's approximation (p = 1.6075, 1 % at most).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import Column, InstanceKind, Rows

# The triangles every edge of a closed mesh joins.
_CLOSED_EDGE_TRIANGLES: Final = 2
# Thomsen's exponent for an ellipsoid's surface area.
_THOMSEN_P: Final = 1.6075


def closed_mesh(scene: Scene, mesh: int) -> bool:
  """Whether mesh `mesh` of `scene` is closed (False for none, `closed_triangles`)."""
  if mesh < 0:
    return False
  view = scene.meshes.get(mesh)
  return closed_triangles(np.asarray(view.points, np.float32), np.asarray(view.tris, np.int32))


def closed_triangles(points: Rows[np.float32, 3], tris: Rows[np.int32, 3]) -> bool:
  """Whether the triangles `tris` (k, 3) over `points` (v, 3) are closed: with the vertices
  welded by position, every edge joins exactly two triangles (False for none)."""
  p = np.asarray(points, np.float32).reshape(-1, 3)
  t = np.asarray(tris, np.int32).reshape(-1, 3)
  if len(t) == 0:
    return False
  _, weld = np.unique(p.round(6), axis=0, return_inverse=True)
  w = weld.reshape(-1)[t]
  edges = np.sort(np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]]), axis=1)
  _, counts = np.unique(edges, axis=0, return_counts=True)
  return bool(np.all(counts == _CLOSED_EDGE_TRIANGLES))


def closed_instances(scene: Scene, ids: Column[np.int64]) -> Column[np.bool_]:
  """(k,) whether the shapes of instances `ids` are closed (module doc); each mesh tested once."""
  kind, mesh = np.asarray(scene.instances.kind)[ids], np.asarray(scene.instances.mesh)[ids]
  closed = kind != int(InstanceKind.PLANE)
  meshed = kind == int(InstanceKind.MESH)
  tested = {m: closed_mesh(scene, m) for m in {int(i) for i in mesh[meshed]}}
  closed[meshed] = [tested[int(m)] for m in mesh[meshed]]
  return closed


def mean_chords(scene: Scene, ids: Column[np.int64]) -> Column[np.float64]:
  """(k,) the mean chords [m] (module doc) of closed instances `ids` of `scene`, their world
  scale applied; 0 for a shape that encloses nothing (a zero size, a flat mesh)."""
  inst = scene.instances
  out = np.zeros(len(ids), np.float64)
  for k, i in enumerate(ids):
    linear = np.asarray(inst.xform[i], np.float64)[:, :3]
    if int(inst.kind[i]) == int(InstanceKind.MESH):
      volume, area = _mesh_measures(scene, int(inst.mesh[i]), linear)
    else:
      volume, area = _primitive_measures(InstanceKind(int(inst.kind[i])), np.asarray(inst.size[i], np.float64), linear)
    out[k] = 4.0 * volume / area if area > 0.0 and volume > 0.0 else 0.0
  return out


def _mesh_measures(scene: Scene, mesh: int, linear: Rows[np.float64, 3]) -> tuple[float, float]:
  """(enclosed volume, surface area) of mesh `mesh` under the linear map `linear` (3, 3)."""
  view = scene.meshes.get(mesh)
  return triangle_measures(np.asarray(view.points, np.float64), np.asarray(view.tris, np.int64), linear)


def triangle_measures(points: Rows[np.float64, 3], tris: Rows[np.int64, 3], linear: Rows[np.float64, 3]) -> tuple[float, float]:
  """(enclosed volume, surface area) of the triangles `tris` (k, 3) over `points` (v, 3) under the
  linear map `linear` (3, 3)."""
  p = np.asarray(points, np.float64).reshape(-1, 3) @ np.asarray(linear, np.float64).T
  t = np.asarray(tris, np.int64).reshape(-1, 3)
  a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
  volume = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())) / 6.0
  area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
  return volume, area


def _primitive_measures(kind: InstanceKind, size: Column[np.float64], linear: Rows[np.float64, 3]) -> tuple[float, float]:
  """(volume, surface area) of a primitive of MuJoCo size `size` scaled by the column lengths of
  `linear` (a sheared primitive: the lengths' approximation)."""
  sx, sy, sz = np.linalg.norm(linear, axis=0)
  if kind == InstanceKind.BOX:
    a, b, c = 2.0 * size[0] * sx, 2.0 * size[1] * sy, 2.0 * size[2] * sz
    return a * b * c, 2.0 * (a * b + b * c + c * a)
  if kind in (InstanceKind.SPHERE, InstanceKind.ELLIPSOID):
    radii = size[:1].repeat(3) if kind == InstanceKind.SPHERE else size
    a, b, c = radii[0] * sx, radii[1] * sy, radii[2] * sz
    mean = ((a * b) ** _THOMSEN_P + (a * c) ** _THOMSEN_P + (b * c) ** _THOMSEN_P) / 3.0
    return 4.0 / 3.0 * math.pi * a * b * c, 4.0 * math.pi * mean ** (1.0 / _THOMSEN_P)
  r, h = size[0] * math.sqrt(sx * sy), 2.0 * size[1] * sz  # radius in the XY plane, height along Z
  cylinder = (math.pi * r * r * h, 2.0 * math.pi * r * h)
  if kind == InstanceKind.CAPSULE:
    return cylinder[0] + 4.0 / 3.0 * math.pi * r**3, cylinder[1] + 4.0 * math.pi * r * r
  return cylinder[0], cylinder[1] + 2.0 * math.pi * r * r
