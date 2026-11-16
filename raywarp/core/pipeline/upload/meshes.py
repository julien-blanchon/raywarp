"""Host instances' shapes -> core poses, shape types and sizes, `Meshes` and `MeshInstances`.

An instance transform A = R S (polar decomposition, `scene.model.xform.polar`) gives the rigid pose
(rotation R) and a stretch S. A primitive folds S into its sizes (a non-uniformly scaled sphere
becomes an ellipsoid). A mesh instance keeps the uniform part s = |det S|^(1/3) of its stretch in
its pose, s R (`instance_poses`: the kernels map rays and scale what they read from the mesh by
it, `kernels/abi/rays.py`), and reads its mesh as it is when the rest S / s is the identity:
scattered copies of a mesh at random sizes share one mesh, one BLAS / GAS. Only the rest of a
non-uniform, sheared or mirrored stretch is baked into a copy of the mesh, one per distinct (mesh, S
/ s) pair (`bake_stretches`). A binding that writes rigid rotations (`Binding.scaled_poses` False:
Newton) gets every stretch baked. Its material's UV set and UV transform are the instance's
(`MeshInstances.texcoord` / `uv_xform`): no mesh copy per UV variant.

Core meshes are the host meshes some instance uses, in host order; their positions, triangles,
corner-indexed normals / UVs / colors are the host pools as they are (`vertadr` / `faceadr` /
`normaladr` / `texcoord_adr` / `coloradr` point into them). Meshes without normals point at one
zero normal (flat shading); meshes without UVs have no texcoord address (constant texel, as
Hydra's fallback st). With a second UV set in use, its corner rows follow the first set's in
`facetexcoord`. Everything here is vectorised over instances; the Python loops run over
meshes or distinct (mesh, stretch) pairs.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import TYPE_CHECKING, Final

import numpy as np
import warp as wp

from raywarp._base.records import record_field
from raywarp._base.types import UV_IDENTITY, UV_MESH, MaterialUvMapping, ShapeType
from raywarp.core.pipeline.gpu_scene.binding import InstancePoses
from raywarp.core.pipeline.gpu_scene.gpu_scene import MaterialUvTables, Meshes, MeshInstances
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.upload.reporting import report_mask
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.tables import Meshes as HostMeshes
from raywarp.scene.model.types import Array, Column, InstanceKind, MaterialFlags, Matrices, Rows
from raywarp.scene.model.xform import Mat3, polar

if TYPE_CHECKING:
  from raywarp.core.pipeline.upload.uv_plans import UvPlans

# Host InstanceKind (values 0, 1, ...) -> the core ShapeType of the same name.
_SHAPE_TYPES: Final = np.array([int(ShapeType[kind.name]) for kind in InstanceKind], np.int32)
# A stretch within this of the identity (`np.allclose` tolerances) is none.
_STRETCH_TOL: Final = 1e-5
# A stretch determinant below this flattens the mesh (it is left unscaled, reported).
_DEGENERATE_DET: Final = 1.0e-30
# Decimals the baked rest of a stretch (S / s, entries of order 1) is rounded to before mesh
# copies are shared: float32 transforms carry ~7 significant digits, so equal authored stretches
# land on one key whatever their rounding.
_STRETCH_DECIMALS: Final = 6
# Bound radius of an infinite plane (light grid extent).
_PLANE_RADIUS: Final = 1.0e6
# The smallest length a stretched normal is divided by.
_MIN_NORMAL_LENGTH: Final = 1e-30
# A smooth tangent shorter than this is none (zero).
_MIN_TANGENT_LENGTH: Final = 1.0e-12

type Paths = list[str]


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class Transforms:
  """The instances' transforms split into rigid poses and stretches.

  Attributes:
    pos: (n, 3) positions.
    rot: (n, 3, 3) rotations (det +1).
    stretch: (n, 3, 3) stretches S (A = R S; reflections included), exactly the identity within
      `_STRETCH_TOL` of it.
  """

  pos: Rows[np.float64, 3]
  rot: Matrices[np.float64, 3, 3]
  stretch: Matrices[np.float64, 3, 3]


def split_transforms(xform: Matrices[np.float32, 3, 4]) -> Transforms:
  """The rigid poses and stretches of (n, 3, 4) instance transforms (`Transforms`)."""
  a = np.asarray(xform, np.float64).reshape(-1, 3, 4)
  if not len(a):
    empty = np.zeros((0, 3, 3))
    return Transforms(pos=np.zeros((0, 3)), rot=empty, stretch=empty)
  rot, stretch = polar(a[:, :, :3])
  eye = np.eye(3)
  near = np.all(np.abs(stretch - eye) <= _STRETCH_TOL + _STRETCH_TOL * eye, axis=(1, 2))
  stretch[near] = eye
  return Transforms(pos=a[:, :, 3], rot=rot, stretch=stretch)


# The per-mesh columns of a host mesh table (the rest are its pools).
_MESH_COLUMNS: Final = tuple(name for name in HostMeshes.SCHEMA if name.endswith(("_adr", "_num")) or name == "aabb")
# The per-triangle pools of a host mesh table ("tris" first: `tri_adr` addresses them all).
_PER_TRIANGLE: Final = ("tris", "normal_idx", "uv_idx", "uv1_idx", "color_idx")


# Rows a host mesh pool may hold: its addresses (`vert_adr`, `normal_adr`, ...) and the core's are
# int32.
_MAX_POOL_ROWS: Final = np.iinfo(np.int32).max


@dataclasses.dataclass(slots=True, frozen=True)
class _Pools:
  """A host mesh table's columns grown to their final size for `bake_stretches`' copies (allocated
  once: no per-copy parts joined after), and the rows written so far."""

  arrays: dict[str, Array]
  sizes: dict[str, int]

  @staticmethod
  def grown(arrays: dict[str, Array], extra: dict[str, int]) -> _Pools:
    """The columns `arrays` with room for `extra` more rows each (the given rows copied in).

    Raises:
      ValueError: a column past `_MAX_POOL_ROWS` rows (its int32 addresses would overflow).
    """
    over = {name: len(arrays[name]) + n for name, n in extra.items() if len(arrays[name]) + n > _MAX_POOL_ROWS}
    if over:
      raise ValueError(
        f"the mesh copies of the scene's stretched instances need {over} rows, past {_MAX_POOL_ROWS} (int32 mesh "
        "addresses): too many non-uniformly scaled, sheared or mirrored mesh instances"
      )
    out: dict[str, Array] = {}
    for name, a in arrays.items():
      grown = np.empty((len(a) + extra.get(name, 0), *a.shape[1:]), a.dtype)
      grown[: len(a)] = a
      out[name] = grown
    return _Pools(out, {name: len(a) for name, a in arrays.items()})

  def append(self, name: str, rows: Array) -> int:
    """Writes `rows` after column `name`'s rows; returns their first row's address."""
    start = self.sizes[name]
    self.arrays[name][start : start + len(rows)] = rows
    self.sizes[name] = start + len(rows)
    return start

  def table(self) -> HostMeshes:
    """The extended mesh table (every column filled)."""
    return HostMeshes.from_columns(self.arrays)


def _uniform_scales(stretch: Matrices[np.float64, 3, 3]) -> tuple[Column[np.float64], Column[np.bool_]]:
  """((n,) uniform part |det S|^(1/3) of each stretch S (1 where degenerate), (n,) the degenerate
  ones: |det S| below `_DEGENERATE_DET`)."""
  det = np.abs(np.linalg.det(stretch))
  flat = det < _DEGENERATE_DET
  return np.where(flat, 1.0, np.cbrt(det)), flat


def _is_mesh(scene: Scene) -> Column[np.bool_]:
  """(n,) the instances that read a mesh."""
  inst = scene.instances
  return (np.asarray(inst.kind) == int(InstanceKind.MESH)) & (np.asarray(inst.mesh, np.int64) >= 0)


def pose_scales(scene: Scene, xf: Transforms) -> Column[np.float64]:
  """(n,) the uniform scale s each instance's pose carries (`instance_poses`; `xf`: the scene's
  `split_transforms`): a mesh instance's |det S|^(1/3) (1 within `_STRETCH_TOL` of it, or for a
  degenerate stretch), 1 for the others."""
  uniform, _flat = _uniform_scales(xf.stretch)
  return np.where(_is_mesh(scene) & (np.abs(uniform - 1.0) > _STRETCH_TOL), uniform, 1.0)


def instance_poses(scene: Scene, xf: Transforms | None = None) -> InstancePoses:
  """The instances' poses (float32): positions and rotations R, a mesh instance's scaled by the
  uniform part s of its stretch, s R (`pose_scales`; `bake_stretches` bakes only the rest).
  `xf`: the scene's `split_transforms` (None: computed)."""
  xf = split_transforms(scene.instances.xform) if xf is None else xf
  scale = pose_scales(scene, xf)
  return InstancePoses(xf.pos.astype(np.float32), (xf.rot * scale[:, None, None]).astype(np.float32))


def bake_stretches(scene: Scene, *, scaled: bool, report: LoadReport, paths: Paths) -> Scene:
  """`scene` with every mesh instance's stretch baked into a mesh but, with `scaled`
  (`Binding.scaled_poses`), its uniform part: an instance whose transform A = R S keeps s R
  (s = |det S|^(1/3), `instance_poses`; without `scaled` R) and reads the copy of its mesh for
  the rest S / s (S) when that is not the identity: one copy per distinct (mesh, shape S / s,
  and without `scaled` s), the shape rounded to `_STRETCH_DECIMALS` and s to float32, so equal
  authored stretches share one. A copy holds points S p / s, normals (S / s)^-T n normalised, its
  triangles' winding reversed for a mirror (det S < 0), so that faces keep the side their normals
  face; UVs and colors shared. A degenerate S (|det| below `_DEGENERATE_DET`) is dropped,
  reported. The kernels read every mesh as it is, in its instance's frame (s R)."""
  inst, hm = scene.instances, scene.meshes
  if not len(inst):
    return scene
  xf = split_transforms(inst.xform)
  mesh = np.asarray(inst.mesh, np.int64)
  stretched = _is_mesh(scene) & ~np.all(xf.stretch == np.eye(3), axis=(1, 2))
  if not np.any(stretched):
    return scene
  uniform, flat = _uniform_scales(xf.stretch)
  flat &= stretched
  report_mask(report.approx, "gpu.instance.degenerate_mesh_scale(unscaled)", flat, example=paths.__getitem__)
  scale = np.where(scaled & stretched & ~flat & (np.abs(uniform - 1.0) > _STRETCH_TOL), uniform, 1.0)
  shape = np.rint(xf.stretch / uniform[:, None, None] * 10.0**_STRETCH_DECIMALS) / 10.0**_STRETCH_DECIMALS
  shape[np.all(np.abs(shape - np.eye(3)) <= _STRETCH_TOL, axis=(1, 2))] = np.eye(3)
  rest = stretched & ~flat & (~np.all(shape == np.eye(3), axis=(1, 2)) if scaled else True)
  baked = np.flatnonzero(rest)
  xform = np.array(inst.xform, np.float32)
  xform[stretched, :, :3] = xf.rot[stretched] * scale[stretched, None, None]
  instances = dataclasses.replace(inst, xform=xform)
  if not len(baked):
    return dataclasses.replace(scene, instances=instances)
  size = np.zeros(len(inst)) if scaled else uniform.astype(np.float32).astype(np.float64)
  key = np.concatenate([mesh[baked, None].astype(np.float64), shape[baked].reshape(-1, 9), size[baked, None]], axis=1)
  pairs, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
  sources = baked[first]
  arrays = {name: np.asarray(a) for name, a in hm.arrays().items()}
  pools = _Pools.grown(arrays, _copy_rows(arrays, mesh[sources], np.linalg.det(xf.stretch[sources]) < 0.0))
  for i in sources.tolist():
    _append_stretched(pools, arrays, int(mesh[i]), xf.stretch[i] / scale[i])
  meshes = pools.table()
  mesh[baked] = len(hm) + inverse.reshape(-1)
  report.count("gpu.meshes_baked_stretch", len(pairs))
  return dataclasses.replace(scene, meshes=meshes, instances=dataclasses.replace(instances, mesh=mesh.astype(np.int32)))


def _copy_rows(arrays: dict[str, Array], meshes: Column[np.int64], mirrored: Column[np.bool_]) -> dict[str, int]:
  """Rows each column of a host mesh table (`arrays`) grows by for copies of host meshes `meshes`
  (`_append_stretched`; `mirrored`: copies whose triangles are reversed)."""
  extra = dict.fromkeys(_MESH_COLUMNS, len(meshes))
  extra["points"] = int(np.asarray(arrays["vert_num"], np.int64)[meshes].sum())
  extra["normals"] = int(np.maximum(np.asarray(arrays["normal_num"], np.int64)[meshes], 0).sum())
  tris = int(np.asarray(arrays["tri_num"], np.int64)[meshes][mirrored].sum())
  extra.update(dict.fromkeys(_PER_TRIANGLE, tris))
  return extra


def _append_stretched(pools: _Pools, arrays: dict[str, Array], h: int, stretch: Mat3) -> None:
  """Appends to `pools` the copy of host mesh h (columns `arrays`) stretched by `stretch`: its
  points and normals, its triangles (winding reversed for a mirror) and its per-mesh row."""
  row = {name: arrays[name][h : h + 1].copy() for name in _MESH_COLUMNS}
  va, nv = int(arrays["vert_adr"][h]), int(arrays["vert_num"][h])
  points = np.asarray(arrays["points"][va : va + nv], np.float64) @ stretch.T
  row["vert_adr"][:] = pools.append("points", points.astype(np.float32))
  na, nn = int(arrays["normal_adr"][h]), int(arrays["normal_num"][h])
  if nn > 0:
    n = np.asarray(arrays["normals"][na : na + nn], np.float64) @ np.linalg.inv(stretch)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), _MIN_NORMAL_LENGTH)
    row["normal_adr"][:] = pools.append("normals", n.astype(np.float32))
  if np.linalg.det(stretch) < 0.0:
    ta, nt = int(arrays["tri_adr"][h]), int(arrays["tri_num"][h])
    row["tri_adr"][:] = pools.sizes["tris"]
    for name in _PER_TRIANGLE:
      pools.append(name, arrays[name][ta : ta + nt][:, [0, 2, 1]])
  row["aabb"][:] = np.stack([points.min(axis=0), points.max(axis=0)]) if nv else 0.0
  for name, value in row.items():
    pools.append(name, value)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class PrimitiveSizes:
  """Core shape types, sizes and bound radii of the instances (stretches folded into primitives).

  Attributes:
    types: (n,) ShapeType (mesh instances: MESH).
    sizes: (n, 3) core sizes (0 for meshes).
    radius: (n,) bound radius (meshes: 0, set by `mesh_radii`).
  """

  types: Column[np.int32]
  sizes: Rows[np.float32, 3]
  radius: Column[np.float32]


def primitive_sizes(
  kind: Column[np.uint8], size: Rows[np.float32, 3], stretch: Matrices[np.float64, 3, 3], *, report: LoadReport, paths: Paths
) -> PrimitiveSizes:
  """The instances' core shape types, sizes and radii (`PrimitiveSizes`), vectorised.

  A primitive's stretch scales its sizes along its axes: a non-uniform sphere becomes an
  ellipsoid; sheared primitives and elliptic capsules / cylinders are approximated (reported).
  """
  kind = np.asarray(kind, np.int64)
  n = len(kind)
  types = _SHAPE_TYPES[kind]
  s = np.abs(np.diagonal(stretch, axis1=1, axis2=2))
  off = np.abs(stretch - s[:, :, None] * np.eye(3)).max(axis=(1, 2))
  sized = np.asarray(size, np.float64).copy()
  radius = np.zeros(n)
  primitive = kind != int(InstanceKind.MESH)
  report_mask(
    report.approx,
    "gpu.instance.sheared_primitive(axis scale)",
    primitive & (off > _STRETCH_TOL * np.maximum(s.max(axis=1), 1.0)),
    example=paths.__getitem__,
  )
  # Spheres: an ellipsoid when the scales differ.
  sphere = kind == int(InstanceKind.SPHERE)
  oval = sphere & (np.ptp(s, axis=1) > _STRETCH_TOL * s.max(axis=1))
  types[oval] = int(ShapeType.ELLIPSOID)
  sized[oval] = size[oval, :1] * s[oval]
  round_ = sphere & ~oval
  sized[round_, 0] = size[round_, 0] * s[round_, 0]
  radius[oval] = sized[oval].max(axis=1)
  radius[round_] = sized[round_, 0]
  # Capsules and cylinders: the mean radius of an elliptic section (reported).
  tube = (kind == int(InstanceKind.CAPSULE)) | (kind == int(InstanceKind.CYLINDER))
  elliptic = tube & (np.abs(s[:, 0] - s[:, 1]) > _STRETCH_TOL * np.maximum(s[:, 0], s[:, 1]))
  report_mask(report.approx, "gpu.instance.elliptic_capsule_or_cylinder(mean radius)", elliptic, example=paths.__getitem__)
  sized[tube, 0] = size[tube, 0] * 0.5 * (s[tube, 0] + s[tube, 1])
  sized[tube, 1] = size[tube, 1] * s[tube, 2]
  capsule = kind == int(InstanceKind.CAPSULE)
  radius[capsule] = sized[capsule, 0] + sized[capsule, 1]
  cylinder = kind == int(InstanceKind.CYLINDER)
  radius[cylinder] = np.hypot(sized[cylinder, 0], sized[cylinder, 1])
  # Boxes and ellipsoids: scaled half sizes / radii.
  box = kind == int(InstanceKind.BOX)
  ellipsoid = kind == int(InstanceKind.ELLIPSOID)
  sized[box | ellipsoid] = size[box | ellipsoid] * s[box | ellipsoid]
  radius[box] = np.linalg.norm(sized[box], axis=1)
  radius[ellipsoid] = sized[ellipsoid].max(axis=1)
  # Planes: half extents; infinite (a size of 0) ones bound everything.
  plane = kind == int(InstanceKind.PLANE)
  sized[plane, :2] = size[plane, :2] * s[plane, :2]
  finite = plane & (sized[:, 0] > 0.0) & (sized[:, 1] > 0.0)
  radius[finite] = np.hypot(sized[finite, 0], sized[finite, 1])
  radius[plane & ~finite] = _PLANE_RADIUS
  sized[~primitive] = 0.0
  return PrimitiveSizes(types=types.astype(np.int32), sizes=sized.astype(np.float32), radius=radius.astype(np.float32))


def _corner_rows(
  hm: HostMeshes, face: Rows[np.int32, 3], *, second_set: bool
) -> tuple[Rows[np.int32, 3], Rows[np.int32, 3], Rows[np.int32, 3]]:
  """The core's (normal, UV, color) corner-index tables: -1 entries -> 0, and the rows of meshes
  without the attribute -> the face rows (so a table equal to `face` is stored once,
  `_index_table`); with `second_set` the second UV set's rows follow the first's."""
  tri_adr, tri_num = np.asarray(hm.tri_adr, np.int64), np.asarray(hm.tri_num, np.int64)

  def rows_of(table: Rows[np.int32, 3], num: Column[np.int32]) -> Rows[np.int32, 3]:
    """One attribute's table (`num`: its per-mesh count)."""
    out = np.where(table < 0, 0, table).astype(np.int32)
    none = np.asarray(num) <= 0
    for a, n in zip(tri_adr[none].tolist(), tri_num[none].tolist(), strict=True):
      out[a : a + n] = face[a : a + n]
    return out

  texcoord = rows_of(hm.uv_idx, hm.uv_num)
  if second_set:
    texcoord = np.concatenate([texcoord, rows_of(hm.uv1_idx, hm.uv1_num)])
  return rows_of(hm.normal_idx, hm.normal_num), texcoord, rows_of(hm.color_idx, hm.color_num)


def _index_table(
  index: Rows[np.int32, 3], face: Rows[np.int32, 3], face_wp: wp.array[wp.vec3i, [int]], uploaded: Uploaded
) -> wp.array[wp.vec3i, [int]]:
  """Device table of a per-corner index (normals, UVs, colors): `face_wp` itself when equal to
  `face` (`uploaded`: the (host, device) pairs of the new tables).

  One index buffer serves every attribute in glTF and USD meshes, usually (a table of a second UV
  set is longer than `face`: always its own).
  """
  if np.array_equal(index, face):
    return face_wp
  table: wp.array[wp.vec3i, [int]] = wp.array(index, dtype=wp.vec3i, device=face_wp.device)
  uploaded.append((index, table))
  return table


def _padded(pool: Array, width: int, value: float) -> np.ndarray[[int, int], np.dtype[np.float32]]:
  """(n + 1, width) float32 rows of a host pool with one row of `value` after them, written in
  place into one array (no float64 temporary: a pool of millions of rows is GBs)."""
  rows = np.asarray(pool).reshape(-1, width)
  out = np.empty((len(rows) + 1, width), np.float32)
  out[:-1] = rows
  out[-1] = value
  return out


# Threads of `mesh_radii` (array operations that release the GIL).
_RADIUS_THREADS: Final = 16


# (host array, its device copy) of the mesh tables an upload makes (`register_digests`).
type Uploaded = list[tuple[np.ndarray, wp.array]]


def register_digests(uploaded: Uploaded) -> None:
  """Registers each device table's content digest (`bake_cache.register_array`, from its host
  array) for the bake keys: the tables are never written after the upload. Hashing releases the
  GIL: it runs on a thread while the upload goes on."""
  for host, device in uploaded:
    bake_cache.register_array(device, bake_cache.array_digest(host))


def core_meshes(
  scene: Scene, used: Column[np.int64], *, second_set: bool, report: LoadReport, device: wp.Device
) -> tuple[Meshes, Uploaded]:
  """(core `Meshes` of host meshes `used` (core mesh k = host mesh used[k]) over the host pools,
  the (host, device) pairs of its vertex and corner tables for `register_digests`).

  `second_set`: an instance reads a second UV set: its corner rows follow the first set's.
  """
  hm = scene.meshes
  face = np.ascontiguousarray(hm.tris, np.int32).reshape(-1, 3)
  tri_adr, tri_num = np.asarray(hm.tri_adr, np.int64), np.asarray(hm.tri_num, np.int64)
  facenormal, facetexcoord, facecolor = _corner_rows(hm, face, second_set=second_set)
  va, vn = np.asarray(hm.vert_adr, np.int64)[used], np.asarray(hm.vert_num, np.int64)[used]
  points = [hm.points[a : a + k] for a, k in zip(va.tolist(), vn.tolist(), strict=True)]
  faces = [face[a : a + k] for a, k in zip(tri_adr[used].tolist(), tri_num[used].tolist(), strict=True)]
  normaladr = np.where(np.asarray(hm.normal_num)[used] > 0, np.asarray(hm.normal_adr)[used], -1)
  texcoord_adr = np.where(np.asarray(hm.uv_num)[used] > 0, np.asarray(hm.uv_adr)[used], -1)
  coloradr = np.where(np.asarray(hm.color_num)[used] > 0, np.asarray(hm.color_adr)[used], -1)
  if np.any(texcoord_adr < 0):
    report.count("gpu.meshes_without_uv", int(np.sum(texcoord_adr < 0)))
  report.count("gpu.meshes", len(used))
  colored = bool(np.any(coloradr >= 0))
  colors = _padded(hm.colors, 4, 1.0)
  uploaded: Uploaded = []

  def table[T](host: np.ndarray, dtype: type[T]) -> wp.array[T, [int]]:
    """The device copy of a vertex / corner table (recorded in `uploaded`)."""
    host = np.ascontiguousarray(host)
    out = wp.array(host, dtype=dtype, device=device)
    uploaded.append((host, out))
    return out

  face_wp = table(face, wp.vec3i)

  def column(values: Column[np.integer]) -> wp.array[int, [int]]:
    """A per-core-mesh int column on the device."""
    return wp.array(np.asarray(values, np.int32), dtype=int, device=device)

  return Meshes(
    vertadr=column(va),
    faceadr=column(tri_adr[used]),
    normaladr=column(normaladr),
    vert=table(np.ascontiguousarray(hm.points, np.float32).reshape(-1, 3), wp.vec3),
    normal=table(_padded(hm.normals, 3, 0.0), wp.vec3),
    face=face_wp,
    facenormal=_index_table(facenormal, face, face_wp, uploaded),
    facetexcoord=_index_table(facetexcoord, face, face_wp, uploaded),
    texcoord=table(_padded(hm.uvs, 2, 0.0), wp.vec2),
    texcoord_adr=column(texcoord_adr),
    points=points,
    faces=faces,
    hfield_bvh_id=wp.zeros(0, dtype=wp.uint64, device=device),
    hfield_bounds_size=wp.zeros(0, dtype=wp.vec3, device=device),
    coloradr=column(coloradr),
    facecolor=_index_table(facecolor, face, face_wp, uploaded),
    color=table(colors if colored else colors[-1:], wp.vec4),
  ), uploaded


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class MeshPlan:
  """How the instances read their meshes, on the host (`mesh_instances` uploads it).

  Attributes:
    core: (n,) core mesh of each instance (-1: a primitive, or no mesh).
    used: (k,) host mesh of each core mesh.
    own_uvs: (n,) the instance reads its own UVs (another UV set or a UV transform).
    second_set: (n,) it reads its mesh's second UV set.
    vertex_colors: (n,) its material multiplies its mesh's vertex colors in (the scene's
      capability, `pipeline/upload/capabilities.py`; the kernels read the material's
      `ShadingFlag.VERTEX_COLOR`, the mesh's color address).
  """

  core: Column[np.int32]
  used: Column[np.int64]
  own_uvs: Column[np.bool_]
  second_set: Column[np.bool_]
  vertex_colors: Column[np.bool_]


def mesh_plan(scene: Scene, plans: UvPlans, *, report: LoadReport) -> MeshPlan:
  """The instances' core meshes and what each reads of its mesh (`MeshPlan`), vectorised."""
  inst, hm, mats = scene.instances, scene.meshes, scene.materials
  n, m = len(inst), len(mats)
  host = np.where(np.asarray(inst.kind) == int(InstanceKind.MESH), np.asarray(inst.mesh, np.int64), -1)
  is_mesh = host >= 0
  used = np.unique(host[is_mesh])
  core_of_host = np.full(len(hm) + 1, -1, np.int32)  # [-1]: no mesh
  core_of_host[used] = np.arange(len(used)).astype(np.int32)
  core = core_of_host[host]
  mat = np.asarray(inst.material, np.int64)
  valid = (mat >= 0) & (mat < m)
  mid = np.where(valid, mat, 0)
  # The material's UV set (set 1 without one on the mesh: set 0, reported) and transform.
  set1 = is_mesh & valid & (plans.uv_set[mid] == 1) if m else np.zeros(n, bool)
  missing = set1 & (np.asarray(hm.uv1_num)[np.maximum(host, 0)] == 0) if len(hm) else set1

  def material_name(i: int) -> str:
    """The name of instance i's material (the reports' example)."""
    return mats.names[int(mid[i])]

  report_mask(report.drop, "gpu.mesh.uv_set1_missing(set 0 used)", missing, example=material_name)
  set1 &= ~missing
  transformed = valid & plans.transformed[mid] if m else np.zeros(n, bool)
  own = is_mesh & (set1 | transformed)
  # Analytic primitive UVs keep the scale of a transform only (its repeat).
  primitive_uvs = ~is_mesh & transformed
  report_mask(report.approx, "gpu.material.uv_transform(primitive UVs: scale only)", primitive_uvs, example=material_name)
  vc_flag = (np.asarray(mats.ext_flags, np.int64)[mid] & int(MaterialFlags.VERTEX_COLOR)) != 0 if m else np.zeros(n, bool)
  colored = is_mesh & valid & vc_flag & (np.asarray(hm.color_num)[np.maximum(host, 0)] > 0) if len(hm) else np.zeros(n, bool)
  return MeshPlan(core=core, used=used, own_uvs=own, second_set=set1, vertex_colors=colored)


def mesh_instances(scene: Scene, plan: MeshPlan, *, plans: UvPlans, report: LoadReport, device: wp.Device) -> MeshInstances:
  """The device `MeshInstances` (on `device`) of a `MeshPlan`: UV addresses and transforms."""
  inst, hm = scene.instances, scene.meshes
  n = len(inst)
  if not n:
    return MeshInstances.plain(0, device)
  host = np.where(plan.core >= 0, plan.used[np.maximum(plan.core, 0)] if len(plan.used) else 0, 0)
  ntri = len(hm.tris)
  # Own UVs: (texcoord address of the set, its corner rows); the set's absence: no UVs (-1).
  adr = np.where(plan.second_set, np.asarray(hm.uv1_adr)[host], np.asarray(hm.uv_adr)[host]) if len(hm) else np.zeros(n)
  num = np.where(plan.second_set, np.asarray(hm.uv1_num)[host], np.asarray(hm.uv_num)[host]) if len(hm) else np.zeros(n)
  row = np.asarray(hm.tri_adr)[host] + np.where(plan.second_set, ntri, 0) if len(hm) else np.zeros(n)
  texcoord = np.stack([np.where(num > 0, adr, -1), row], axis=1)
  texcoord[~plan.own_uvs] = (UV_MESH, 0)
  mat = np.asarray(inst.material, np.int64)
  m = len(scene.materials)
  xform = np.tile(UV_IDENTITY, (n, 1, 1))
  if m:
    with_xform = plan.own_uvs & (mat >= 0) & (mat < m)
    xform[with_xform] = plans.xform[mat[with_xform]]
  if np.any(plan.own_uvs):
    report.count("gpu.instances.own_uvs", int(plan.own_uvs.sum()))
  return MeshInstances.build(texcoord=texcoord.astype(np.int32), uv=xform, device=device)


def mesh_radii(scene: Scene, plan: MeshPlan, scale: Column[np.float64]) -> Column[np.float32]:
  """(n,) bound radius of each mesh instance about its origin (max |p| over its mesh's points,
  times its pose's uniform `scale`), 0 for others; one computation per core mesh."""
  radius = np.zeros(len(plan.core), np.float32)
  if not len(plan.used):
    return radius
  hm = scene.meshes

  def of(h: int) -> float:
    """The radius of host mesh h."""
    a, n = int(hm.vert_adr[h]), int(hm.vert_num[h])
    p = np.asarray(hm.points[a : a + n], np.float64)
    return float(np.sqrt((p * p).sum(axis=1).max())) if n else 0.0

  # NumPy releases the GIL for these array operations: the meshes run on threads.
  with concurrent.futures.ThreadPoolExecutor(min(_RADIUS_THREADS, len(plan.used))) as pool:
    per_mesh = np.asarray(list(pool.map(of, plan.used.tolist())), np.float32)
  meshes = plan.core >= 0
  radius[meshes] = per_mesh[plan.core[meshes]] * scale[meshes]
  return radius


def material_uv_tables(scene: Scene, plan: MeshPlan, plans: UvPlans, *, device: wp.Device) -> MaterialUvTables | None:
  """The materials' UV mappings (`MaterialUvMapping`: UV set, transform) and each used mesh's
  second-UV-set addresses (`MaterialUvTables`); None when no material transforms its UVs or reads
  set 1."""
  if not np.any(plans.transformed | (plans.uv_set == 1)):
    return None
  records = wp.zeros(len(scene.materials), dtype=MaterialUvMapping, device="cpu").numpy()
  record_field(records, "uv_set")[...] = plans.uv_set
  record_field(records, "xform")[...] = plans.xform
  hm = scene.meshes
  used = plan.used
  # A mesh without a second set reads its first (UV_MESH; `mesh_plan` reported it); a second set's
  # corner rows follow the first set's (`core_meshes`, `second_set`).
  second = np.stack(
    [
      np.where(np.asarray(hm.uv1_num)[used] > 0, np.asarray(hm.uv1_adr)[used], UV_MESH),
      np.asarray(hm.tri_adr)[used] + len(hm.tris),
    ],
    axis=1,
  ).astype(np.int32)
  return MaterialUvTables(
    material=wp.array(records, dtype=MaterialUvMapping, device=device),
    mesh_second=wp.array(second, dtype=wp.vec2i, device=device),
    missing_second=tuple(int(i) for i in np.flatnonzero(second[:, 0] == UV_MESH)) if np.any(plans.uv_set == 1) else (),
  )


# A UV triangle of |det| below this has no tangent (degenerate or unmapped).
_UV_DET_EPS: Final = 1.0e-14


def vertex_tangents(scene: Scene, plan: MeshPlan) -> Rows[np.float32, 4]:
  """(nnormal + 1, 4) smooth UV tangents of the meshes `plan` uses, addressed like the core's
  normals (`MeshAdr.normal` + the face's normal corners; the last row: none): each triangle's
  dp/du (first UV set) normalised and area-weighted onto its corner normals, then made
  orthogonal to them (Gram-Schmidt); w the handedness sign(dot(n x t, dp/dv)). Zero where a
  vertex has no mapped triangle (the anisotropy then falls back to a fixed frame)."""
  hm = scene.meshes
  face = np.ascontiguousarray(hm.tris, np.int32).reshape(-1, 3)
  facenormal, facetexcoord, _ = _corner_rows(hm, face, second_set=False)
  normals = np.asarray(hm.normals, np.float64).reshape(-1, 3)
  points = np.asarray(hm.points, np.float64).reshape(-1, 3)
  uvs = np.asarray(hm.uvs, np.float64).reshape(-1, 2)
  acc = np.zeros((len(normals) + 1, 3))
  hand = np.zeros(len(normals) + 1)
  for k in np.unique(plan.used).tolist():
    if int(hm.normal_num[k]) <= 0 or int(hm.uv_num[k]) <= 0:
      continue
    a, n = int(hm.tri_adr[k]), int(hm.tri_num[k])
    p = points[int(hm.vert_adr[k]) + face[a : a + n]]
    uv = uvs[int(hm.uv_adr[k]) + facetexcoord[a : a + n]]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > _UV_DET_EPS
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)[:, None]
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * inv
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * inv
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    length = np.linalg.norm(t, axis=1)
    weight = np.where(ok & (length > 0.0), area / np.where(length > 0.0, length, 1.0), 0.0)[:, None]
    corners = int(hm.normal_adr[k]) + facenormal[a : a + n]
    sign = np.sign(np.einsum("ij,ij->i", np.cross(np.cross(e1, e2), t), b))
    for c in range(3):
      np.add.at(acc, corners[:, c], t * weight)
      np.add.at(hand, corners[:, c], sign * area)
  nrm = np.concatenate([normals, np.zeros((1, 3))])
  acc -= nrm * np.einsum("ij,ij->i", nrm, acc)[:, None]
  length = np.linalg.norm(acc, axis=1)
  out = np.zeros((len(acc), 4), np.float32)
  good = length > _MIN_TANGENT_LENGTH
  out[good, :3] = acc[good] / length[good, None]
  out[:, 3] = np.where(hand < 0.0, -1.0, 1.0)
  return out
