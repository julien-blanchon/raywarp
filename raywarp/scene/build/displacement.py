"""Bounded, view-independent scalar displacement into ordinary shared meshes.

The source mesh is linearly refined. Position-welded normals and seam-averaged heights keep
UV/hard-normal seams closed; incompatible height values at a seam are reported. This is not
Catmull-Clark limit evaluation. Prepared meshes use the usual backend and cache paths.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, final

import numpy as np

from raywarp.scene.build.mesh_pool import MeshAttributes, MeshPool
from raywarp.scene.images import decode, mips
from raywarp.scene.images.blob import Probed
from raywarp.scene.model import geometry
from raywarp.scene.model.material import Displacement, Material
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.tables import Instances, Meshes, MeshView
from raywarp.scene.model.types import Column, ImageId, InstanceKind, MeshId, Rows, TextureRef, Wrap

if TYPE_CHECKING:
  from shape_extensions import IntVar


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class DisplacementOptions:
  """Preparation limits. The triangle budget covers newly displaced meshes, once per variant.

  Attributes:
    mode: "true" (displace the geometry) or "off" (authored displacement reported and ignored).
    max_level: the most four-way refinements of a textured mesh (a constant amount is not refined).
    max_triangles: the triangles all displaced variants may add together (int32 corner addresses).
  """

  mode: Literal["true", "off"] = "true"
  max_level: int = 6
  max_triangles: int = 16_000_000

  def __post_init__(self) -> None:
    """Reject invalid budgets before any mesh or image work."""
    if self.mode not in ("true", "off"):
      raise ValueError("displacement mode must be 'true' or 'off'; bump mode is not implemented")
    for name, value in (("max_level", self.max_level), ("max_triangles", self.max_triangles)):
      if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"displacement {name} must be a nonnegative integer")
    if self.max_triangles > (2**31 - 1) // 3:
      raise ValueError("displacement triangle budget exceeds int32 corner addresses")


def _address(indices: Column[np.int64], size: int, mode: int) -> tuple[Column[np.int64], Column[np.bool_]]:
  """Pixel indices and valid taps for normalized texture addressing."""
  valid = np.ones(len(indices), np.bool_)
  if mode == Wrap.REPEAT:
    return indices % size, valid
  if mode == Wrap.MIRROR:
    folded = indices % (2 * size)
    return np.minimum(folded, 2 * size - 1 - folded), valid
  if mode == Wrap.BLACK:
    valid = (indices >= 0) & (indices < size)
  return np.clip(indices, 0, size - 1), valid


def sample_height(height: Rows[np.float32, int], uv: Rows[np.float32, 2], wrap: tuple[int, int]) -> Column[np.float32]:
  """Bilinear scalar texels, bottom-left host UVs, half-texel centers and authored wrap."""
  h, w = height.shape
  xy = uv.astype(np.float64) * np.asarray((w, h), np.float64) - 0.5
  lower = np.floor(xy).astype(np.int64)
  fraction = xy - lower
  out = np.zeros(len(uv), np.float64)
  for dy in (0, 1):
    y, vy = _address(lower[:, 1] + dy, h, wrap[1])
    wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
    for dx in (0, 1):
      x, vx = _address(lower[:, 0] + dx, w, wrap[0])
      wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
      out += height[h - 1 - y, x] * wx * wy * vx * vy
  return out.astype(np.float32)


def _split_corners[D: IntVar](
  values: np.ndarray[[int, 3, D], np.dtype[np.float32]],
) -> np.ndarray[[int, 3, D], np.dtype[np.float32]]:
  """Linearly interpolate each triangle's attributes without joining UV/color seams."""
  a, b, c = values[:, 0], values[:, 1], values[:, 2]
  ab, bc, ca = (a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5
  return np.concatenate([np.stack(v, axis=1) for v in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])


def _refine(points: Rows[np.float32, 3], tris: Rows[np.int32, 3]) -> tuple[Rows[np.float32, 3], Rows[np.int32, 3]]:
  """One four-way linear split, with a single midpoint per shared edge."""
  edges = np.sort(tris[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1).astype(np.int64)
  # One int64 key per edge: the same (lexicographic) order as a row-wise unique, sorted far faster.
  keys, inverse = np.unique(edges[:, 0] * len(points) + edges[:, 1], return_inverse=True)
  unique = np.stack(np.divmod(keys, len(points)), axis=1)
  mid = inverse.reshape(-1, 3) + len(points)
  a, b, c = tris.T
  ab, bc, ca = mid.T
  children = np.concatenate([np.stack(v, axis=1) for v in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
  return np.concatenate([points, (points[unique[:, 0]] + points[unique[:, 1]]) * 0.5]), children.astype(np.int32)


def _uv(mesh: MeshView, ref: TextureRef) -> Rows[np.float32, 2]:
  """Per-corner texture coordinates in the authored binding's frame."""
  values, indices = (mesh.uvs, mesh.uv_idx) if ref.uv_set == 0 else (mesh.uvs1, mesh.uv1_idx)
  if not len(values):
    raise ValueError(f"displacement requires UV set {ref.uv_set}")
  transform = np.asarray(ref.uv_xform, np.float32)
  mapped = values[indices].reshape(-1, 2) @ transform[:, :2].T + transform[:, 2]
  if not np.all(np.isfinite(mapped)):
    raise ValueError("displacement UV transform exceeds finite float32 values")
  return mapped


def _replaces_authored_normals(mesh: MeshView, points: Rows[np.float32, 3], tris: Rows[np.int32, 3]) -> bool:
  """Whether the welded smooth normals differ from `mesh`'s authored corner normals (compared one
  corner at a time: the temporaries stay one corner's size)."""
  if not len(mesh.normals):
    return False
  smooth = geometry.smooth_normals(points, tris)
  for corner in range(3):
    authored = mesh.normals[mesh.normal_idx[:, corner]].astype(np.float64)
    lengths = np.linalg.norm(authored, axis=-1, keepdims=True)
    authored = np.divide(authored, lengths, out=np.zeros_like(authored), where=lengths > 0)
    if not np.allclose(smooth[tris[:, corner]], authored, rtol=0, atol=1e-4):
      return True
  return False


def displaced_mesh(
  mesh: MeshView,
  amount: float | TextureRef,
  height: Rows[np.float32, int] | None,
  *,
  wrap: tuple[int, int],
  level: int,
  report: LoadReport,
) -> tuple[Rows[np.float32, 3], Rows[np.int32, 3], MeshAttributes]:
  """(points, triangles, attributes) of `mesh` refined `level` times and moved along its welded
  smooth normals by `amount` (a distance, or a texture binding read from the `height` map with
  `wrap`); budgets must be checked before this allocation.

  Raises:
    ValueError: for a missing UV set, a non-finite UV transform or non-finite moved positions.
  """
  used = np.unique(mesh.tris)
  first, inverse = geometry.weld(mesh.points[used])
  points = mesh.points[used][first]
  tris = inverse[np.searchsorted(used, mesh.tris)].astype(np.int32)
  if _replaces_authored_normals(mesh, points, tris):
    report.approx("displacement.authored_normals_reconstructed", "welded smooth directions replace authored corner normals")
  # Triangles welded to a point or an edge (a UV sphere's pole fans) have no area; refined, their
  # edge midpoints would be extra vertices at the same place without a normal, which stay behind
  # when the surface moves (a hole at the pole).
  keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])
  tris = tris[keep]
  uv = mesh.uvs[mesh.uv_idx[keep]] if len(mesh.uvs) else None
  uv1 = mesh.uvs1[mesh.uv1_idx[keep]] if len(mesh.uvs1) else None
  colors = mesh.colors[mesh.color_idx[keep]] if len(mesh.colors) else None
  texuv = _uv(mesh, amount).reshape(-1, 3, 2)[keep] if isinstance(amount, TextureRef) else None
  for _ in range(level):
    points, tris = _refine(points, tris)
    uv = None if uv is None else _split_corners(uv)
    uv1 = None if uv1 is None else _split_corners(uv1)
    colors = None if colors is None else _split_corners(colors)
    texuv = None if texuv is None else _split_corners(texuv)
  normals = geometry.smooth_normals(points, tris)
  distance = _distances(len(points), tris, amount, height, texuv, wrap=wrap, report=report)
  moved = points.astype(np.float64) + normals * distance[:, None]
  if not np.all(np.isfinite(moved)) or np.any(np.abs(moved) > np.finfo(np.float32).max):
    raise ValueError("displacement produced non-finite float32 positions")
  points = moved.astype(np.float32)
  attributes = MeshAttributes(normals=geometry.smooth_normals(points, tris), normal_idx=tris)
  indices = np.arange(tris.size).astype(np.int32).reshape(-1, 3)
  if uv is not None:
    attributes.update(uvs=uv.reshape(-1, 2), uv_idx=indices)
  if uv1 is not None:
    attributes.update(uvs1=uv1.reshape(-1, 2), uv1_idx=indices)
  if colors is not None:
    attributes.update(colors=colors.reshape(-1, 4), color_idx=indices)
  return points, tris, attributes


def _distances(  # noqa: PLR0913 (the refined mesh, then the height binding)
  npoint: int,
  tris: Rows[np.int32, 3],
  amount: float | TextureRef,
  height: Rows[np.float32, int] | None,
  texuv: np.ndarray[[int, 3, 2], np.dtype[np.float32]] | None,
  *,
  wrap: tuple[int, int],
  report: LoadReport,
) -> Column[np.float64]:
  """(npoint,) displacement distance of each refined vertex: `amount`, or the heights its binding
  reads from `height` at the corners' UVs `texuv` (with `wrap`), averaged over the corners of a
  welded vertex (a seam whose heights disagree is reported).

  Raises:
    ValueError: for a textured `amount` without a decoded height map or UVs.
  """
  if not isinstance(amount, TextureRef):
    return np.full(npoint, amount, np.float64)
  if height is None or texuv is None:
    raise ValueError("textured displacement requires decoded height and UV coordinates")
  heights = sample_height(height, texuv.reshape(-1, 2), wrap) * amount.scale[0] + amount.bias[0]
  flat = tris.reshape(-1)
  counts = np.bincount(flat, minlength=npoint)
  distance = np.bincount(flat, weights=heights, minlength=npoint) / np.maximum(counts, 1)
  if np.any(np.abs(heights - distance[flat]) > 1e-6 * max(1.0, float(np.max(np.abs(heights))))):
    report.approx("displacement.uv_seam_averaged", "incompatible heights at welded positions")
  return distance


def _height(source: decode.ImageSource, probed: Probed, channel: int) -> Rows[np.float32, int]:
  """Channel `channel` of an image as linear float32 heights (its read-ahead top level, else decoded
  now).

  Raises:
    ValueError: for an undecodable image or non-finite heights.
  """
  pixels, error = (probed.base, probed.error) if probed.base is not None else decode.decode_base(source, probed.info)
  if error or probed.info.error:
    raise ValueError(error or probed.info.error)
  rgba = decode.linear_rgba(pixels, transfer=source.transfer)
  result = rgba[..., channel].copy()
  if not np.all(np.isfinite(result)):
    raise ValueError("displacement height contains non-finite values")
  return result


def _limit_level(requested: int, triangles: int, options: DisplacementOptions, remaining: int) -> int:
  """Largest uniform level fitting both budgets (-1 when even the source does not fit)."""
  level = min(requested, options.max_level)
  while level and triangles * 4**level > remaining:
    level -= 1
  return level if triangles <= remaining else -1


def _texel_span(mesh: MeshView, amount: TextureRef, height: Rows[np.float32, int]) -> tuple[float, int]:
  """(the longest triangle edge of `mesh` in texels of the height map `height`, the refinement
  level that brings every edge to at most one texel: ceil(log2(longest)), 0 below one texel)."""
  uv = _uv(mesh, amount).reshape(-1, 3, 2)
  edges = (uv.astype(np.float64) - np.roll(uv, 1, axis=1)) * np.asarray((height.shape[1], height.shape[0]))
  longest = float(np.max(np.linalg.norm(edges, axis=2))) if len(edges) else 0.0
  return longest, max(0, math.ceil(math.log2(max(longest, 1.0))))


def _filtered_height(height: Rows[np.float32, int], longest: float, level: int) -> Rows[np.float32, int]:
  """Band-limit height to prepared edge spacing; finer residual bump is not implemented."""
  mip = min(max(0, math.floor(math.log2(max(longest / 2**level, 1.0)))), max(height.shape).bit_length() - 1)
  image = height[..., None]
  for _ in range(mip):
    # One large factor pads a 1xW image to WxW. Bounded reductions keep the short
    # axis at one texel and the working allocation proportional to source pixels.
    image = mips.box_reduce(image, 2)
  return image[..., 0].astype(np.float32)


@dataclasses.dataclass(slots=True, kw_only=True)
class _Preparation:
  """One `prepare` pass: its inputs, the height maps decoded so far and the triangle budget left."""

  pool: MeshPool
  original: Meshes
  textures: Sequence[tuple[ImageId, int, int, int]]
  sources: Sequence[decode.ImageSource]
  probed: Sequence[Probed]
  options: DisplacementOptions
  report: LoadReport
  remaining: int
  decoded: dict[tuple[ImageId, int], Rows[np.float32, int]] = dataclasses.field(default_factory=dict)

  def height(self, amount: TextureRef) -> tuple[Rows[np.float32, int], tuple[int, int]]:
    """(the height map of a textured `amount`, decoded once per image and channel, its wrap)."""
    image, _, ws, wt = self.textures[amount.texture]
    key = (image, amount.channel)
    if key not in self.decoded:
      self.decoded[key] = _height(self.sources[image], self.probed[image], amount.channel)
    return self.decoded[key], (ws, wt)

  def variant(self, mesh_id: int, amount: float | TextureRef, name: str) -> MeshId | None:
    """The new mesh of mesh `mesh_id` displaced by `amount` (material `name`), refined as far as
    the budgets allow; None when even the source mesh exceeds the triangle budget (reported)."""
    mesh = self.original.get(mesh_id)
    height, wrap, level, longest = None, (int(Wrap.REPEAT), int(Wrap.REPEAT)), 0, 0.0
    if isinstance(amount, TextureRef):
      height, wrap = self.height(amount)
      longest, level = _texel_span(mesh, amount, height)
    requested = level
    level = _limit_level(requested, len(mesh.tris), self.options, self.remaining)
    if level < 0:
      self.report.drop("displacement.triangle_budget", name)
      return None
    if level < requested:
      self.report.approx("displacement.refinement_limited", f"{name}: {requested} -> {level}")
    if height is not None:
      height = _filtered_height(height, longest, level)
    if len(np.flatnonzero(self.original.vert_adr == self.original.vert_adr[mesh_id])) > 1:  # a part of a split mesh
      self.report.approx(
        "displacement.subset_boundary", "parts are refined independently; boundary stitching is not implemented"
      )
    points, tris, attributes = displaced_mesh(mesh, amount, height, wrap=wrap, level=level, report=self.report)
    new_id = self.pool.add(points, tris, attributes, report=self.report)[0]
    self.remaining -= len(tris)
    self.report.count("displacement.triangles", len(tris))
    self.report.count("displacement.mesh_variants")
    return new_id


def prepare(  # noqa: PLR0913 (explicit scene preparation inputs)
  pool: MeshPool,
  instances: Instances,
  materials: Sequence[Material],
  textures: Sequence[tuple[ImageId, int, int, int]],
  sources: Sequence[decode.ImageSource],
  probed: Sequence[Probed],
  options: DisplacementOptions,
  report: LoadReport,
) -> None:
  """Remap authored (mesh, displacement) variants once, before final mesh-table assembly.

  Original mesh IDs remain valid. There is no mesh pass or extra array when no displacement
  is authored. Prepared geometry is serialized by the existing scene cache without rebaking.
  The triangle budget and analytic primitives are reported limits (the flat mesh kept).

  Raises:
    ValueError: for invalid displacement inputs (a missing UV set, an undecodable height map, a
      non-finite result), naming the material.
  """
  active = {i: m.displacement for i, m in enumerate(materials) if m.displacement is not None and m.displacement.amount != 0.0}
  if not active:
    return
  if options.mode == "off":
    report.drop("displacement.disabled", "scene preparation options")
    return
  prep = _Preparation(
    pool=pool,
    original=pool.table(),
    textures=textures,
    sources=sources,
    probed=probed,
    options=options,
    report=report,
    remaining=options.max_triangles,
  )
  variants: dict[tuple[int, Displacement], MeshId] = {}
  pairs = np.unique(np.stack([instances.mesh, instances.material, instances.kind], axis=1), axis=0)
  for mesh_id, material_id, kind in pairs:
    spec = active.get(int(material_id))
    if spec is None:
      continue
    name = materials[material_id].name
    if kind != InstanceKind.MESH:
      report.drop("displacement.analytic_primitive", name)
      continue
    key = (int(mesh_id), spec)
    if key not in variants:
      try:
        new_id = prep.variant(int(mesh_id), spec.amount, name)
      except ValueError as error:  # invalid authored inputs: explicit, never the flat mesh instead
        raise ValueError(f"displacement of material {name!r}: {error}") from error
      if new_id is None:
        continue
      variants[key] = new_id
    selected = (instances.mesh == mesh_id) & (instances.material == material_id) & (instances.kind == kind)
    instances.mesh[selected] = variants[key]
