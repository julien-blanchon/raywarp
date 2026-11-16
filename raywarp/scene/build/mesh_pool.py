"""The mesh pools of a `SceneBuilder`: meshes deduplicated by content, chunks joined at build."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Final, TypedDict, final

import numpy as np
import numpy.typing as npt

from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.tables import Meshes
from raywarp.scene.model.types import NO_MESH, Array, Column, MeshId, Rows

# Rows an int32 pool address reaches (`Meshes` *_adr columns, the kernels' int indices).
_INT32_MAX: Final = 2**31 - 1
# Vertex colors given as RGB get alpha 1.
_RGB: Final = 3


class MeshAttributes(TypedDict, total=False):
  """Optional attributes of a mesh: value pools with per-corner indices.

  A pool given without its index must be per vertex (one value per point, indexed like `tris`).

  Attributes:
    normals: (N, 3) normal pool.
    normal_idx: (T, 3) corner indices into `normals`.
    uvs: (U, 2) UV pool.
    uv_idx: (T, 3) corner indices into `uvs`.
    uvs1: (U1, 2) second UV set (TextureRef.uv_set 1).
    uv1_idx: (T, 3) corner indices into `uvs1`.
    colors: (C, 3 | 4) linear vertex colors (alpha 1 if absent).
    color_idx: (T, 3) corner indices into `colors`.
  """

  normals: npt.ArrayLike | None
  normal_idx: npt.ArrayLike | None
  uvs: npt.ArrayLike | None
  uv_idx: npt.ArrayLike | None
  uvs1: npt.ArrayLike | None
  uv1_idx: npt.ArrayLike | None
  colors: npt.ArrayLike | None
  color_idx: npt.ArrayLike | None


# Each value pool of `MeshAttributes` and its corner index.
_POOLS: Final = (("normals", "normal_idx"), ("uvs", "uv_idx"), ("uvs1", "uv1_idx"), ("colors", "color_idx"))
# The joined pools of `Meshes` (the second UV set's values join the uvs pool): (name, width, dtype).
_COLUMNS: Final = (
  ("points", 3, np.float32),
  ("tris", 3, np.int32),
  ("normals", 3, np.float32),
  ("normal_idx", 3, np.int32),
  ("uvs", 2, np.float32),
  ("uv_idx", 3, np.int32),
  ("uv1_idx", 3, np.int32),
  ("colors", 4, np.float32),
  ("color_idx", 3, np.int32),
)


def finite_attributes(attributes: MeshAttributes, report: LoadReport, where: object, *, source: str = "Mesh") -> MeshAttributes:
  """`attributes` without the value pools holding NaN / inf (each dropped with its index and
  reported as `<source>.<pool>.non_finite` at `where`): an adapter's way to keep a mesh whose
  authored normals, UVs or colors are broken (flat normals, no UVs, no colors) where
  `SceneBuilder.add_mesh` rejects them."""
  out = MeshAttributes(**attributes)
  for pool, index in _POOLS:
    values = out.get(pool)
    if values is not None and not np.all(np.isfinite(np.asarray(values, np.float64))):
      report.drop(f"{source}.{pool}.non_finite", where)
      out[pool] = None
      out[index] = None
  return out


@final
@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class MeshRow:
  """The address / count columns of one `Meshes` row."""

  vert_adr: int
  vert_num: int
  tri_adr: int
  tri_num: int
  normal_adr: int
  normal_num: int
  uv_adr: int
  uv_num: int
  uv1_adr: int
  uv1_num: int
  color_adr: int
  color_num: int


# A corner attribute: (values (n, width) float32, corner indices (T, 3) int32), both None if absent.
type Corner = tuple[np.ndarray[[int, int], np.dtype[np.float32]] | None, Rows[np.int32, 3] | None]


@final
@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Mesh:
  """A validated mesh, triangles grouped by part: points, tris and its corner attributes."""

  points: Rows[np.float32, 3]
  tris: Rows[np.int32, 3]
  normals: Corner
  uvs: Corner
  uvs1: Corner
  colors: Corner
  counts: Column[np.int64]  # triangles per part

  def key(self) -> bytes:
    """Content hash (dedup key)."""
    h = hashlib.blake2b(digest_size=20)
    pools = (self.normals, self.uvs, self.uvs1, self.colors)
    for a in (self.points, self.tris, *(x for pair in pools for x in pair), self.counts):
      if a is None:
        h.update(b"-")
      else:
        h.update(np.asarray(a.shape, np.int64).tobytes())
        h.update(a)  # validated pools are C-contiguous: hash their buffer without a payload copy
    return h.digest()


def _check_rows(values: Array, *, name: str, width: int) -> None:
  """Checks that `values` are rows of `width` values: (n, width), or flat with a multiple of
  `width` values (a (n, 2) array of 6 values must not read as (2, 3)).

  Raises:
    ValueError: for another last dimension or a flat size that is not a multiple of `width`.
  """
  if (values.ndim > 1 and values.shape[-1] != width) or values.size % width:
    raise ValueError(f"mesh {name}: rows of {width} values, got shape {values.shape}")


def _indices(values: npt.ArrayLike, *, name: str, count: int) -> Column[np.int32]:
  """Nonnegative pool indices, validated before narrowing to the stored int32 format."""
  indices = np.asarray(values)
  if indices.size:
    if indices.dtype.kind not in "iu":
      raise ValueError(f"mesh {name}: indices must be integers, got {indices.dtype}")
    low, high = indices.min(), indices.max()
    if low < 0 or high >= count or high > _INT32_MAX:
      raise ValueError(f"mesh {name} index out of range [0, {min(count, _INT32_MAX + 1)}): {low}..{high}")
  return np.ascontiguousarray(indices, dtype=np.int32).reshape(-1)


def _corner(  # noqa: PLR0913  # named pool and index, then keyword-only shape/context
  name: str, values: npt.ArrayLike | None, index: npt.ArrayLike | None, *, width: int, tris: Rows[np.int32, 3], npoint: int
) -> Corner:
  """A validated corner attribute (`MeshAttributes`): values reshaped to `width`, indices
  checked."""
  if values is None:
    return None, None
  v = np.asarray(values, dtype=np.float32)
  if v.ndim > 1 and v.shape[-1] != width and not (name == "colors" and v.shape[-1] == _RGB):
    raise ValueError(f"mesh {name}: rows of {width} values, got shape {v.shape}")
  v = v.reshape(-1, v.shape[-1] if v.ndim > 1 else width)
  if name == "colors" and v.shape[1] == _RGB:
    v = np.concatenate([v, np.ones((len(v), 1), np.float32)], axis=1)
  v = np.ascontiguousarray(v.reshape(-1, width))
  if not np.all(np.isfinite(v)):
    raise ValueError(f"mesh {name}: NaN / inf values")
  i = tris if index is None else _indices(index, name=name, count=len(v)).reshape(-1, 3)
  if index is None and len(v) != npoint:
    raise ValueError(f"{name} without an index must be per vertex: {len(v)} values for {npoint} points")
  if i.shape != tris.shape:
    raise ValueError(f"mesh {name}: corner indices must have shape {tris.shape}, got {i.shape}")
  return v, i


def _mesh(
  points: npt.ArrayLike, tris: npt.ArrayLike, attributes: MeshAttributes, *, tri_part: npt.ArrayLike | None, nparts: int
) -> _Mesh:
  """Validates a mesh and sorts its triangles by part (stable)."""
  p = np.ascontiguousarray(points, dtype=np.float32)
  _check_rows(p, name="points", width=3)
  p = p.reshape(-1, 3)
  if not np.all(np.isfinite(p)):
    raise ValueError("mesh points: NaN / inf values")
  t = np.asarray(tris)
  _check_rows(t, name="triangle", width=3)
  t = _indices(t, name="triangle", count=len(p)).reshape(-1, 3)
  a = attributes
  normals = _corner("normals", a.get("normals"), a.get("normal_idx"), width=3, tris=t, npoint=len(p))
  uvs = _corner("uvs", a.get("uvs"), a.get("uv_idx"), width=2, tris=t, npoint=len(p))
  uvs1 = _corner("uvs1", a.get("uvs1"), a.get("uv1_idx"), width=2, tris=t, npoint=len(p))
  colors = _corner("colors", a.get("colors"), a.get("color_idx"), width=4, tris=t, npoint=len(p))
  part = None if tri_part is None else _indices(tri_part, name="tri_part", count=nparts)
  if part is not None and len(part) != len(t):
    raise ValueError(f"tri_part must give each of the {len(t)} triangles a part in [0, {nparts})")
  if part is None or nparts == 1:
    counts = np.zeros(nparts, np.int64)
    counts[0] = len(t)
    return _Mesh(points=p, tris=t, normals=normals, uvs=uvs, uvs1=uvs1, colors=colors, counts=counts)
  order = np.argsort(part, kind="stable")

  def sort(corner: Corner) -> Corner:
    """The corner attribute with its indices in part order."""
    return corner[0], None if corner[1] is None else corner[1][order]

  counts = np.bincount(part, minlength=nparts).astype(np.int64)
  return _Mesh(
    points=p, tris=t[order], normals=sort(normals), uvs=sort(uvs), uvs1=sort(uvs1), colors=sort(colors), counts=counts
  )


@final
class MeshPool:
  """Meshes being built: pools of chunks (joined by `table`), one row per mesh part."""

  def __init__(self) -> None:
    """An empty pool."""
    self._chunks: dict[str, list[Array]] = {name: [] for name, _, _ in _COLUMNS}
    self._sizes = {"points": 0, "tris": 0, "normals": 0, "uvs": 0, "colors": 0}
    self._rows: list[MeshRow] = []
    self._aabb: list[Rows[np.float32, 3]] = []
    self._keys: dict[bytes, tuple[MeshId, ...]] = {}

  def __len__(self) -> int:
    """Number of meshes (parts)."""
    return len(self._rows)

  def add(  # noqa: PLR0913  # mesh pools, then keyword-only report and part layout
    self,
    points: npt.ArrayLike,
    tris: npt.ArrayLike,
    attributes: MeshAttributes,
    *,
    report: LoadReport,
    tri_part: npt.ArrayLike | None = None,
    nparts: int = 1,
  ) -> list[MeshId]:
    """Adds a mesh split into `nparts` parts by `tri_part`; one mesh id per part (NO_MESH: empty).

    A mesh equal to one added before (same content and parts) returns that one's ids.
    """
    mesh = _mesh(points, tris, attributes, tri_part=tri_part, nparts=nparts)
    key = mesh.key()
    if key in self._keys:
      report.count("meshes.deduplicated")
      return list(self._keys[key])
    ids = self._append(mesh)
    self._keys[key] = tuple(ids)
    return ids

  def _append(self, mesh: _Mesh) -> list[MeshId]:
    """Appends a new mesh's pools and one row per non-empty part; returns the part ids."""
    sizes, chunks = self._sizes, self._chunks
    v0, t0, n0, c0, u0 = sizes["points"], sizes["tris"], sizes["normals"], sizes["colors"], sizes["uvs"]
    (n, ni), (u, ui), (u1, u1i), (c, ci) = mesh.normals, mesh.uvs, mesh.uvs1, mesh.colors
    u10 = u0 + (0 if u is None else len(u))  # the second UV set follows the first in the uvs pool
    chunks["points"].append(mesh.points)
    chunks["tris"].append(mesh.tris)
    for name, idx in (("normal_idx", ni), ("uv_idx", ui), ("uv1_idx", u1i), ("color_idx", ci)):
      chunks[name].append(idx if idx is not None else np.full_like(mesh.tris, -1))
    for name, values in (("normals", n), ("uvs", u), ("uvs", u1), ("colors", c)):
      if values is not None:
        chunks[name].append(values)
        sizes[name] += len(values)
    sizes["points"] += len(mesh.points)
    sizes["tris"] += len(mesh.tris)
    p = mesh.points
    aabb = np.stack([p.min(axis=0), p.max(axis=0)]) if len(p) else np.zeros((2, 3), np.float32)

    def num(a: Array | None) -> int:
      """Rows of an optional pool (0: absent)."""
      return 0 if a is None else len(a)

    # Parts share the vertex and attribute ranges; each has its own triangle range.
    ids: list[MeshId] = []
    start = t0
    for count in mesh.counts.tolist():
      if count == 0:
        ids.append(NO_MESH)
        continue
      ids.append(MeshId(len(self._rows)))
      self._rows.append(
        MeshRow(
          vert_adr=v0,
          vert_num=len(p),
          tri_adr=start,
          tri_num=count,
          normal_adr=n0,
          normal_num=num(n),
          uv_adr=u0,
          uv_num=num(u),
          uv1_adr=u10,
          uv1_num=num(u1),
          color_adr=c0,
          color_num=num(c),
        )
      )
      self._aabb.append(aabb)
      start += count
    return ids

  def table(self) -> Meshes:
    """The `Meshes` table of every mesh added.

    Raises:
      ValueError: when a pool holds more rows than its int32 addresses reach (2^31 - 1; the
        second UV set shares the UV pool).
    """
    over = {name: n for name, n in self._sizes.items() if n > _INT32_MAX}
    if over:
      raise ValueError(f"mesh pools beyond int32 addresses ({_INT32_MAX} rows): {over}")
    names = [f.name for f in dataclasses.fields(MeshRow)]
    rows = np.asarray([dataclasses.astuple(r) for r in self._rows], dtype=np.int32).reshape(-1, len(names))
    return Meshes.from_columns(
      {
        **{name: rows[:, i] for i, name in enumerate(names)},
        "aabb": np.asarray(self._aabb, np.float32).reshape(-1, 2, 3),
        **{
          name: np.concatenate(self._chunks[name]) if self._chunks[name] else np.zeros((0, width), dtype)
          for name, width, dtype in _COLUMNS
        },
      }
    )
