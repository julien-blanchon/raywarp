"""Deformable geometry in the core's terms (e.g. MuJoCo flex).

A deformable is a range of vertices whose positions change every step (the "flex.vert" slot:
(nworld, nvert) world positions, e.g. MuJoCo Warp's `d.flexvert_xpos`) plus a static topology:

* dim 1 (rods, ropes): segments (`Part.edges`), each a capsule of the part's radius;
* dim 2 (cloth, shells): mid-surface triangles (`Part.faces`) thickened by the radius on both
  sides along smooth vertex normals, with side walls along the boundary edges (`Part.borders`);
* dim 3 (soft bodies): the closed surface triangles (`Part.faces`), one-sided, no thickness.

An adapter describes each deformable as a `Part` and builds the record with `Deformables.create`.
The core owns everything else: the scene-BVH primitives (`prim_flexid` / `prim_edgeid`: one
capsule per segment, one box per surface), the per-surface triangle BVHs and their per-frame
refit (`core/trace/accel/deformable_bvh.py`), and the corner UVs of those triangles (`corner_uv`).

A part may name a material (MuJoCo's flex material): its color (when the part's rgba is the
default), metallic / roughness / specular / emission, and, for a surface with texture
coordinates, its texture maps (`core/shade/materials/flex.py`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal, final

import numpy as np
import warp as wp

from raywarp.scene.model.types import NO_MATERIAL, MaterialId

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import Ids


# Host topology and positions: (k, 2) vertex pairs, (k, 3) vertex triples, (n, 3) positions.
type Pairs = np.ndarray[[int, 2], np.dtype[np.int32]]
type Triangles = np.ndarray[[int, 3], np.dtype[np.int32]]
type Points = np.ndarray[[int, 3], np.dtype[np.float32]]
type Uvs = np.ndarray[[int, 2], np.dtype[np.float32]]
type Dim = Literal[1, 2, 3]

# Part dimensions (`Part.dim`).
SEGMENTS: Final = 1  # rods, ropes: capsules
THICK_SURFACE: Final = 2  # cloth, shells: offset triangle copies and side walls
CLOSED_SURFACE: Final = 3  # soft bodies: the surface triangles
_DIMS: Final = (SEGMENTS, THICK_SURFACE, CLOSED_SURFACE)


def surface_triangles(dim: int, nface: int, nborder: int) -> int:
  """Triangles per world of a part's mesh: dim 2 two offset copies per face and two per boundary
  edge (the side wall), dim 3 its faces, dim 1 none (`core/trace/accel/deformable_bvh.py`)."""
  if dim == THICK_SURFACE:
    return 2 * nface + 2 * nborder
  return nface if dim == CLOSED_SURFACE else 0


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Part:
  """One deformable, on the host (vertex ids are local: 0 is the part's first vertex).

  Attributes:
    dim: 1 segments, 2 thick surface, 3 closed surface (see the module doc).
    vertadr, vertnum: the part's vertex range in the "flex.vert" columns.
    radius: capsule radius (dim 1) or half thickness (dim 2) [m]; unused for dim 3.
    rgba: color (alpha 0 hides it).
    edges: (k, 2) segments (dim 1).
    faces: (k, 3) triangles (dim 2: mid-surface; dim 3: surface, outward winding).
    borders: (k, 2) boundary edges of a dim-2 surface (its side walls).
    material: material id (-1: none, the default material).
    texcoord: (m, 2) texture coordinates of a surface (dim 2 / 3; empty: none, no texture maps).
    face_texcoord: (k, 3) `texcoord` row of each corner of `faces` (with `texcoord`). The side
      walls read the row of their vertex id (MuJoCo's convention).
  """

  dim: Dim
  vertadr: int
  vertnum: int
  radius: float
  rgba: tuple[float, float, float, float]
  edges: Pairs = dataclasses.field(default_factory=lambda: np.zeros((0, 2), np.int32))
  faces: Triangles = dataclasses.field(default_factory=lambda: np.zeros((0, 3), np.int32))
  borders: Pairs = dataclasses.field(default_factory=lambda: np.zeros((0, 2), np.int32))
  material: MaterialId = NO_MATERIAL
  texcoord: Uvs = dataclasses.field(default_factory=lambda: np.zeros((0, 2), np.float32))
  face_texcoord: Triangles = dataclasses.field(default_factory=lambda: np.zeros((0, 3), np.int32))


def _check(i: int, p: Part, nvert: int) -> None:
  """Raises ValueError when part `i` is invalid: an unknown dim, a negative or non-finite radius,
  a color outside [0, 1], a material id below -1, or a vertex range or topology indexing past its
  vertices (the kernels read them unchecked)."""
  if p.dim not in _DIMS:
    raise ValueError(f"deformable {i}: dim must be 1, 2 or 3, got {p.dim!r}")
  if p.dim != CLOSED_SURFACE and not (np.isfinite(p.radius) and p.radius >= 0.0):
    raise ValueError(f"deformable {i}: radius must be a finite length >= 0 [m], got {p.radius}")
  rgba = np.asarray(p.rgba, np.float64)
  if rgba.shape != (4,) or not (np.all(np.isfinite(rgba)) and np.all((rgba >= 0.0) & (rgba <= 1.0))):
    raise ValueError(f"deformable {i}: rgba must be 4 values in [0, 1], got {p.rgba}")
  if p.material < -1:
    raise ValueError(f"deformable {i}: material must be an id >= 0 or -1 (none), got {p.material}")
  if p.vertadr < 0 or p.vertnum < 0 or p.vertadr + p.vertnum > nvert:
    raise ValueError(f"deformable {i}: vertices [{p.vertadr}, {p.vertadr + p.vertnum}) outside the {nvert} given")
  for name, ids in (("edges", p.edges), ("faces", p.faces), ("borders", p.borders)):
    if len(ids) and (ids.min() < 0 or ids.max() >= p.vertnum):
      raise ValueError(f"deformable {i}: {name} index vertices outside its {p.vertnum}")
  if len(p.texcoord):
    uv_ids = np.concatenate([p.face_texcoord.reshape(-1), p.borders.reshape(-1)])
    if p.face_texcoord.shape != p.faces.shape or (len(uv_ids) and (uv_ids.min() < 0 or uv_ids.max() >= len(p.texcoord))):
      raise ValueError(f"deformable {i}: face_texcoord must index its {len(p.texcoord)} texcoords per face corner")


def _corner_uvs(p: Part) -> Uvs:
  """(triangles * 3, 2) UVs of the corners of part `p`'s mesh triangles, in the mesh's index order
  (`core/trace/accel/deformable_bvh.py`: dim 2 the +radius copy, the -radius copy wound (0, 2, 1),
  then the side walls (b0+, b1-, b1+) and (b1-, b0+, b0-); dim 3 the faces)."""
  t = p.face_texcoord
  if p.dim == CLOSED_SURFACE:
    return p.texcoord[t.reshape(-1)]
  pairs = np.stack([t, t[:, [0, 2, 1]]], axis=1).reshape(-1)
  b = p.borders
  walls = np.stack([b[:, [0, 1, 1]], b[:, [1, 0, 0]]], axis=1).reshape(-1)
  return p.texcoord[np.concatenate([pairs, walls])]


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Deformables[NDeformable: IntVar, NFlexPrim: IntVar]:
  """The deformables of a scene, concatenated (device tables + host topology).

  Built by `create` (or `empty`); never mutated. Vertex ids of `edge` are local to their part (the
  traversal adds `vertadr`); `faces` / `borders` hold global vertex ids (the BVH kernels read them
  directly).

  Attributes:
    nprim: scene-BVH primitives: one per segment of a dim-1 part, one per dim-2 / 3 part.
    dim: (n,) host part dimensions.
    vertadr, vertnum, radius, rgba, material: (n,) per part.
    nface: (n,) triangles per world of each part's mesh (`surface_triangles`).
    uvadr: (n,) row of `corner_uv` of each part's first triangle corner (-1: no UVs).
    corner_uv: UVs of every triangle corner of the parts with UVs (`_corner_uvs`).
    edge: (nedge,) segments of every dim-1 part (local ids); `prim_edgeid` indexes it.
    prim_flexid: (nprim,) part of each primitive; `prim_edgeid` its segment, -1 for a surface.
    max_material: the largest material id of the parts (-1: none); `check_materials`.
    faces, borders: host per-part global-id triangles / boundary edges of the dim-2 / 3 parts.
    init_vert: (nvert, 3) float32 vertex positions the BVHs are first built from.
  """

  nprim: int
  dim: Ids[NDeformable]
  vertadr: wp.array[int, [NDeformable]]
  vertnum: wp.array[int, [NDeformable]]
  radius: wp.array[float, [NDeformable]]
  rgba: wp.array[wp.vec4, [NDeformable]]
  material: wp.array[int, [NDeformable]]
  nface: wp.array[int, [NDeformable]]
  uvadr: wp.array[int, [NDeformable]]
  corner_uv: wp.array[wp.vec2, [int]]
  edge: wp.array[wp.vec2i, [int]]
  prim_flexid: wp.array[int, [NFlexPrim]]
  prim_edgeid: wp.array[int, [NFlexPrim]]
  max_material: MaterialId
  faces: tuple[Triangles, ...]
  borders: tuple[Pairs, ...]
  init_vert: Points

  @property
  def count(self) -> int:
    """Number of deformables (parts)."""
    return len(self.dim)

  @property
  def has_material(self) -> bool:
    """A part has a material (`Features.flex_materials`)."""
    return self.max_material >= 0

  @staticmethod
  def create(parts: Sequence[Part], init_vert: Points, *, device: wp.Device) -> Deformables:
    """The record of `parts` (in order: part i is deformable i) at vertex positions `init_vert`,
    its arrays on `device`.

    Raises:
      ValueError: a part's vertices or topology outside `init_vert` / its own vertex range.
    """
    init_vert = np.asarray(init_vert, np.float32).reshape(-1, 3)
    for i, p in enumerate(parts):
      _check(i, p, len(init_vert))
    flexid: list[int] = []
    edgeid: list[int] = []
    edges: list[Pairs] = []
    nedge = 0
    for i, p in enumerate(parts):
      if p.dim == SEGMENTS:
        flexid.extend([i] * len(p.edges))
        edgeid.extend(range(nedge, nedge + len(p.edges)))
        edges.append(p.edges)
        nedge += len(p.edges)
      else:
        flexid.append(i)
        edgeid.append(-1)
    edge = np.concatenate(edges).astype(np.int32) if edges else np.zeros((0, 2), np.int32)
    uvs = [_corner_uvs(p) if len(p.texcoord) and p.dim != SEGMENTS else np.zeros((0, 2), np.float32) for p in parts]
    starts = np.cumsum([0] + [len(u) for u in uvs])[:-1]
    uvadr = [int(a) if len(u) else -1 for a, u in zip(starts, uvs, strict=True)]
    corner_uv = np.concatenate(uvs).astype(np.float32) if uvs else np.zeros((0, 2), np.float32)
    return Deformables(
      nprim=len(flexid),
      dim=np.array([p.dim for p in parts], np.int32),
      vertadr=wp.array(np.array([p.vertadr for p in parts], np.int32), dtype=int, device=device),
      vertnum=wp.array(np.array([p.vertnum for p in parts], np.int32), dtype=int, device=device),
      radius=wp.array(np.array([p.radius for p in parts], np.float32), dtype=float, device=device),
      rgba=wp.array(np.array([p.rgba for p in parts], np.float32).reshape(-1, 4), dtype=wp.vec4, device=device),
      material=wp.array(np.array([p.material for p in parts], np.int32), dtype=int, device=device),
      nface=wp.array(
        np.array([surface_triangles(p.dim, len(p.faces), len(p.borders)) for p in parts], np.int32), dtype=int, device=device
      ),
      uvadr=wp.array(np.array(uvadr, np.int32), dtype=int, device=device),
      corner_uv=wp.array(corner_uv.reshape(-1, 2), dtype=wp.vec2, device=device),
      edge=wp.array(edge, dtype=wp.vec2i, device=device),
      prim_flexid=wp.array(np.array(flexid, np.int32), dtype=int, device=device),
      prim_edgeid=wp.array(np.array(edgeid, np.int32), dtype=int, device=device),
      max_material=max((p.material for p in parts), default=NO_MATERIAL),
      faces=tuple((p.faces + p.vertadr).astype(np.int32) for p in parts),
      borders=tuple((p.borders + p.vertadr).astype(np.int32) for p in parts),
      init_vert=init_vert,
    )

  def check_materials(self, nmaterial: int) -> None:
    """Raises ValueError when a part's material id is not below the scene's `nmaterial` (the flex
    shading kernels read it unchecked)."""
    if self.max_material >= nmaterial:
      raise ValueError(f"a deformable has material {self.max_material}, the scene {nmaterial} materials")

  @staticmethod
  def empty(device: wp.Device) -> Deformables:
    """No deformable (the kernels' placeholder tables of a scene without any, on `device`)."""
    return Deformables.create([], np.zeros((0, 3), np.float32), device=device)
