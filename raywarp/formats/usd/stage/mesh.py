"""UsdGeom.Mesh -> triangles with corner-indexed normals, UVs and colors (vectorised numpy).

Polygons are fan-triangulated; faceVarying and indexed primvars keep their own pools and are
indexed per triangle corner, so positions are never unwelded. GeomSubsets of the materialBind
family split the faces into parts (one mesh per part, sharing the vertex data).

What is read besides the geometry depends on the bound materials (`MeshNeeds`): the UV primvar of
set 0 (else the first of UV_NAMES), the second UV set a texture names (e.g. st1), and vertex
colors: the primvar a material multiplies its base color with, or the displayColor (and
displayOpacity) of an unbound mesh when it varies over the mesh.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, final

import numpy as np
from pxr import Usd, UsdGeom

from raywarp.formats.usd.util import Color, Time, valued_attributes
from raywarp.scene.model import geometry
from raywarp.scene.model.report import LoadReport

# UV primvar names tried when the material names none (Omniverse / Maya / Blender exports).
UV_NAMES: Final = ("st", "st0", "st_0", "UVMap", "uv", "map1", "UVW")
# Primvars read here or elsewhere (materials, flags); every other authored primvar is reported.
_CONSUMED_PRIMVARS: Final = {"normals", "displayColor", "displayOpacity", "doNotCastShadows"}
DISPLAY_COLOR: Final = "displayColor"
_SKEL_PREFIX: Final = "skel:"
# Authored Mesh attributes read here (the rest is reported as dropped).
CONSUMED_ATTRS: Final = {
  "points",
  "faceVertexCounts",
  "faceVertexIndices",
  "holeIndices",
  "orientation",
  "normals",
  "doubleSided",
  "subdivisionScheme",
  "extent",
  "purpose",
  "visibility",
  "proxyPrim",
  "xformOpOrder",
  "kind",
}
# Ignored without a report: authoring metadata (user properties; a GeomSubset family's type, which
# does not change what is drawn), physics and skinning bind data.
IGNORED_PREFIXES: Final = (
  "xformOp:",
  "userProperties:",
  "subsetFamily:",
  "physics:",
  "physx",
  "newton:",
  "omni:",
  "mjc:",
  "material:binding",
  "semantic:",
  "skel:",
)

type Points = np.ndarray[[int, 3], np.dtype[np.float32]]
type Corners = np.ndarray[[int, 3], np.dtype[np.int64]]  # per-triangle corner indices
type Faces = np.ndarray[[int], np.dtype[np.int64]]
type CornerIndex = np.ndarray[[int, 3], np.dtype[np.int32]]
type Uvs = np.ndarray[[int, 2], np.dtype[np.float32]]
type Normals = np.ndarray[[int, 3], np.dtype[np.float32]]
type Rgba = np.ndarray[[int, 4], np.dtype[np.float32]]
type PrimvarValues = np.ndarray[[int, int], np.dtype[np.float64]]  # (values, components)
# A UV primvar's components, and the face sizes the triangulation reports.
UV_DIMS: Final = 2
TRIANGLE: Final = 3
QUAD: Final = 4


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MeshNeeds:
  """Mesh attributes the bound materials read (see the module docstring).

  Attributes:
    uv0: UV primvar of set 0 (None: the first of UV_NAMES found).
    uv1: UV primvar of set 1 (None: not read).
    color: vertex-color primvar (None: not read).
    unbound: no material: `color` is displayColor, read only when it is not constant.
    inherit: the material comes from an instancing prototype's binding (unknown here).
  """

  uv0: str | None = None
  uv1: str | None = None
  color: str | None = None
  unbound: bool = False
  inherit: bool = False


@final
@dataclasses.dataclass(frozen=True)
class Topology:
  """A triangulated polygon mesh: triangles, their face corners and source faces."""

  tris: Corners  # (T, 3) vertex indices
  corners: Corners  # (T, 3) face-corner indices (faceVarying data)
  tri_face: Faces  # (T,) source face


@final
@dataclasses.dataclass(frozen=True)
class Primvar:
  """A primvar's values (as (n, components)), interpolation and indices."""

  values: PrimvarValues
  interpolation: str
  indices: Faces | None


@final
@dataclasses.dataclass
class MeshData:
  """A triangulated mesh: attribute pools with per-corner indices, each triangle's source face,
  sidedness and the mean display color / opacity."""

  points: Points
  tris: CornerIndex
  normals: Normals | None
  normal_idx: CornerIndex | None
  uvs: Uvs | None
  uv_idx: CornerIndex | None
  uvs1: Uvs | None
  uv1_idx: CornerIndex | None
  colors: Rgba | None  # (C, 4) linear RGBA
  color_idx: CornerIndex | None
  tri_face: Faces
  nface: int
  double_sided: bool
  display_color: Color | None
  display_opacity: float | None


def _corner_index(interp: str, indices: Faces | None, nvalues: int, topo: Topology) -> Corners | None:
  """(T, 3) indices into a primvar's value pool, or None if the sizes do not match."""
  if interp in ("vertex", "varying"):
    base, expected = topo.tris, int(topo.tris.max()) + 1 if len(topo.tris) else 0
  elif interp == "faceVarying":
    base, expected = topo.corners, int(topo.corners.max()) + 1 if len(topo.corners) else 0
  elif interp == "uniform":
    tri_face = topo.tri_face
    base, expected = np.repeat(tri_face[:, None], 3, axis=1), int(tri_face.max()) + 1 if len(tri_face) else 0
  elif interp == "constant":
    base, expected = np.zeros_like(topo.tris), 1
  else:
    return None
  if indices is not None:
    if len(indices) < expected:
      return None
    out = indices[base]
    return out if (not len(out) or (int(out.min()) >= 0 and int(out.max()) < nvalues)) else None
  return base if nvalues >= expected else None


def _primvar(api: UsdGeom.PrimvarsAPI, name: str, time: Time) -> Primvar | None:
  """An authored, non-empty primvar (values reshaped to (n, components))."""
  pv = api.GetPrimvar(name)
  if not pv or not pv.HasAuthoredValue():
    return None
  values = pv.Get(time)
  if values is None or len(values) == 0:
    return None
  indices = pv.GetIndices(time) if pv.IsIndexed() else None
  v = np.asarray(values, np.float64)
  return Primvar(v.reshape(len(v), -1), pv.GetInterpolation(), None if indices is None else np.asarray(indices, dtype=np.int64))


def material_subsets(prim: Usd.Prim, time: Time, report: LoadReport) -> list[tuple[Usd.Prim, Faces]]:
  """(GeomSubset prim, face indices) of the materialBind subsets of a mesh. A binding on a subset
  of another family is not a per-face binding (USD's and Hydra's rule): reported."""
  for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Mesh(prim)):
    sub = subset.GetPrim()
    bound = any(r.GetName().startswith("material:binding") for r in sub.GetRelationships())
    if bound and subset.GetFamilyNameAttr().Get() != "materialBind":
      report.drop("GeomSubset.binding_outside_materialBind_family", sub)
  subsets = []
  for subset in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Mesh(prim), UsdGeom.Tokens.face, "materialBind"):
    faces = subset.GetIndicesAttr().Get(time)
    if faces is not None and len(faces):
      subsets.append((subset.GetPrim(), np.asarray(faces, dtype=np.int64)))
  return subsets


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _Read:
  """What every primvar read of one mesh takes: its primvars, time, triangulation and report."""

  api: UsdGeom.PrimvarsAPI
  time: Time
  topo: Topology
  prim: Usd.Prim
  report: LoadReport


def _uv_set(read: _Read, names: tuple[str, ...]) -> tuple[Uvs | None, CornerIndex | None, str | None]:
  """(UV pool, corner indices, primvar name) of the first readable primvar of `names`."""
  api, time, topo, prim, report = read.api, read.time, read.topo, read.prim, read.report
  for name in names:
    pv = _primvar(api, name, time)
    if pv is None or pv.values.shape[1] < UV_DIMS:
      continue
    idx = _corner_index(pv.interpolation, pv.indices, len(pv.values), topo)
    if idx is None:
      report.drop(f"Mesh.primvars:{name}.size_mismatch", prim)
      continue
    return np.ascontiguousarray(pv.values[:, :2], dtype=np.float32), idx.astype(np.int32), name
  return None, None, None


def _colors(read: _Read, wanted: MeshNeeds) -> tuple[Rgba | None, CornerIndex | None]:
  """(per-corner RGBA pool (3T, 4), corner indices (T, 3)) of the wanted color primvar."""
  api, time, topo, prim, report = read.api, read.time, read.topo, read.prim, read.report
  name = wanted.color
  pv = _primvar(api, name, time) if name else None
  if pv is None:
    if name and not wanted.unbound:
      report.count("meshes.vertex_colors_absent(white)")
    return None, None
  if wanted.unbound and (pv.interpolation == "constant" or len(pv.values) <= 1):
    return None, None  # a constant displayColor is the fallback material's color
  v = pv.values.astype(np.float32)
  idx = _corner_index(pv.interpolation, pv.indices, len(v), topo)
  if idx is None:
    report.drop(f"Mesh.primvars:{name}.size_mismatch", prim)
    return None, None
  ntri = len(topo.tris)
  rgba = np.ones((ntri * 3, 4), np.float32)
  n = min(v.shape[1], 4)
  rgba[:, :n] = v[idx.reshape(-1), :n]
  if name == DISPLAY_COLOR:
    op = _primvar(api, "displayOpacity", time)
    if op is not None:
      oidx = _corner_index(op.interpolation, op.indices, len(op.values), topo)
      if oidx is not None:
        rgba[:, 3] = op.values[:, 0].astype(np.float32)[oidx.reshape(-1)]
  report.count("meshes.vertex_colors")
  return rgba, np.arange(ntri * 3).astype(np.int32).reshape(-1, 3)


def _topology(mesh: UsdGeom.Mesh, time: Time, prim: Usd.Prim, report: LoadReport) -> tuple[Points, Topology, int] | None:
  """(points, triangulation, face count) of a mesh (None when it has no usable faces, reported)."""
  points = mesh.GetPointsAttr().Get(time)
  counts = mesh.GetFaceVertexCountsAttr().Get(time)
  indices = mesh.GetFaceVertexIndicesAttr().Get(time)
  if not points or not counts or not indices:
    report.drop("Mesh.empty", prim)
    return None
  p = np.asarray(points, dtype=np.float32)
  c = np.asarray(counts, dtype=np.int64)
  idx = np.asarray(indices, dtype=np.int64)
  if int(c.sum()) != len(idx) or (len(idx) and (int(idx.max()) >= len(p) or int(idx.min()) < 0)):
    report.drop("Mesh.invalid_topology", prim)
    return None
  if any(a.ValueMightBeTimeVarying() for a in (mesh.GetPointsAttr(), mesh.GetFaceVertexIndicesAttr())):
    report.approx("Mesh.time_samples(one time read)", prim)
  holes = mesh.GetHoleIndicesAttr().Get(time)
  left = mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
  tris, corners, tri_face = geometry.triangulate(c, idx, None if not holes else np.asarray(holes), left_handed=left)
  if len(tris) == 0:
    report.drop("Mesh.no_triangles", prim)
    return None
  if np.any(c > QUAD):
    report.approx("Mesh.ngon_fan_triangulation", prim)
  if np.any(c < TRIANGLE):
    report.drop("Mesh.degenerate_faces", prim, len(np.flatnonzero(c < TRIANGLE)))
  return p, Topology(tris, corners, tri_face), len(c)


def _normals(read: _Read, mesh: UsdGeom.Mesh, p: Points) -> tuple[Normals | None, CornerIndex | None]:
  """Authored normals (primvars:normals, else the normals attribute); smooth ones of a subdivision
  cage without any."""
  api, time, topo, prim, report = read.api, read.time, read.topo, read.prim, read.report
  normals = normal_idx = None
  pv = _primvar(api, "normals", time)
  if pv is None:
    n = mesh.GetNormalsAttr().Get(time)
    if n is not None and len(n):
      pv = Primvar(np.asarray(n, np.float64).reshape(-1, 3), mesh.GetNormalsInterpolation(), None)
  if pv is not None:
    idx = _corner_index(pv.interpolation, pv.indices, len(pv.values), topo)
    if idx is None:
      report.drop("Mesh.normals.size_mismatch", prim)
    else:
      normals, normal_idx = pv.values.astype(np.float32).reshape(-1, 3), idx.astype(np.int32)
  scheme = mesh.GetSubdivisionSchemeAttr().Get()
  if scheme != UsdGeom.Tokens.none:
    report.approx(f"Mesh.subdivisionScheme.{scheme}(cage)", prim)
    if normals is None:
      normals, normal_idx = geometry.smooth_normals(p, topo.tris), topo.tris.astype(np.int32)
  return normals, normal_idx


def display(
  api: UsdGeom.PrimvarsAPI, time: Time, report: LoadReport, *, prototype: bool = False
) -> tuple[Color | None, float | None]:
  """Mean displayColor / displayOpacity (the fallback material of an unbound gprim).

  Values outside [0, 1] (HDR authoring) are clamped and reported; a non-finite mean is reported
  and left unset (the grey fallback / opaque). `prototype`: a varying displayColor is reported as
  averaged (a prototype mesh's fallback).
  """
  prim = api.GetPrim()
  color = opacity = None
  dc = _primvar(api, DISPLAY_COLOR, time)
  if dc is not None and len(dc.values):
    mean = dc.values[:, :3].mean(axis=0)
    if not np.isfinite(mean).all():
      report.drop("Mesh.primvars:displayColor.non_finite", prim)
    else:
      if ((mean < 0.0) | (mean > 1.0)).any():
        report.approx("Mesh.primvars:displayColor.clamped", prim)
        mean = np.clip(mean, 0.0, 1.0)
      color = (float(mean[0]), float(mean[1]), float(mean[2]))
    if prototype and dc.interpolation != "constant" and len(dc.values) > 1:
      report.approx("Mesh.primvars:displayColor.averaged(prototype fallback)", prim)
  do = _primvar(api, "displayOpacity", time)
  if do is not None and len(do.values):
    value = float(do.values.mean())
    if not math.isfinite(value):
      report.drop("Mesh.primvars:displayOpacity.non_finite", prim)
    elif not 0.0 <= value <= 1.0:
      report.approx("Mesh.primvars:displayOpacity.clamped", prim)
      opacity = min(max(value, 0.0), 1.0)
    else:
      opacity = value
  return color, opacity


def _report_primvars(api: UsdGeom.PrimvarsAPI, read: set[str | None], prim: Usd.Prim, report: LoadReport) -> None:
  """Reports the authored primvars nothing reads, and skinning (rendered in the bind pose)."""
  for pv in api.GetAuthoredPrimvars():
    name = pv.GetPrimvarName()
    if name in _CONSUMED_PRIMVARS or name in read or name.startswith(_SKEL_PREFIX):
      continue
    report.drop(f"Mesh.primvars:{name}", prim)
  if "SkelBindingAPI" in prim.GetAppliedSchemas() or api.HasPrimvar("skel:jointIndices"):
    report.approx("Mesh.skinning(bind_pose)", prim)


def read_mesh(prim: Usd.Prim, time: Time, wanted: MeshNeeds, report: LoadReport) -> MeshData | None:
  """Reads and triangulates a UsdGeom.Mesh (None when it has no faces)."""
  mesh = UsdGeom.Mesh(prim)
  topology = _topology(mesh, time, prim, report)
  if topology is None:
    return None
  p, topo, nface = topology
  api = UsdGeom.PrimvarsAPI(prim)
  read = _Read(api, time, topo, prim, report)
  normals, normal_idx = _normals(read, mesh, p)
  names0 = ((wanted.uv0,) if wanted.uv0 else ()) + tuple(n for n in UV_NAMES if n != wanted.uv0)
  uvs, uv_idx, used_uv = _uv_set(read, names0)
  uvs1 = uv1_idx = used_uv1 = None
  if wanted.uv1 and wanted.uv1 != used_uv:
    uvs1, uv1_idx, used_uv1 = _uv_set(read, (wanted.uv1,))
    if uvs1 is None:
      report.drop(f"Mesh.primvars:{wanted.uv1}.missing(second UV set)", prim)
  colors, color_idx = _colors(read, wanted)
  display_color, display_opacity = display(api, time, report, prototype=wanted.inherit)
  _report_primvars(api, {used_uv, used_uv1, wanted.color}, prim, report)
  return MeshData(
    points=p,
    tris=topo.tris.astype(np.int32),
    normals=normals,
    normal_idx=normal_idx,
    uvs=uvs,
    uv_idx=uv_idx,
    uvs1=uvs1,
    uv1_idx=uv1_idx,
    colors=colors,
    color_idx=color_idx,
    tri_face=topo.tri_face,
    nface=nface,
    double_sided=bool(mesh.GetDoubleSidedAttr().Get()),
    display_color=display_color,
    display_opacity=display_opacity,
  )


def report_attributes(prim: Usd.Prim, consumed: set[str], report: LoadReport) -> None:
  """Reports every attribute of `prim` with an authored value that the adapter does not read."""
  type_name = prim.GetTypeName()
  for attr in valued_attributes(prim):
    name = attr.GetName()
    if name in consumed or name.startswith(("primvars:", *IGNORED_PREFIXES)):
      continue
    report.drop(f"{type_name}.{name}", prim)
