"""Ingest: Newton model -> host Scene of its shapes, and the tables its binding is built from.

`load(model)` is the adapter's Ingest step: the host `raywarp.Scene` of the model's own shapes
(primitives, meshes, colors, mesh textures) and the added lights / environment / headlight. Its Bind
step, `api.binding(scene, model)`, rebuilds the binding of a host scene of model shapes from the
instances' Newton shape ids (`Instances.source`: `source_rows`, `shape_tables`), so a cached scene
binds to a live model; `api.scene(model)` is `rw.upload(load(model), binding=binding(...))`.
`load_usd(model, stage)` (`from_usd`) ingests the stage's visuals instead; `binding(scene, model,
stage=stage)` binds those to the bodies labelled with their rigid-body prim paths.

Worlds: the scene holds the worlds' pooled shapes (`pooled`: one instance per shape of world 0,
plus one per shape of another world that world 0 has no equal of, e.g. a different object to pick
in each world) and the global ones, and a (world, instance) table of bodies, so render world v
shows Newton world v (or `StateView.world_indices[v]`): its own shapes, the pooled instances of
other worlds hidden there (alpha 0 per world, `binding.WorldColors`: run-time data, no recompile
when a world shows another object). Worlds of the same shapes give world 0's instances only. With
`merge_worlds`, one render world holds every Newton world's shapes (a viewer). Instances the
simulation moves per world are per world (`NewtonBinding.dynamic`); a Renderer shares the others
(global shapes on no body: stored and traced once for all worlds) unless its `PerWorld` declares
them. Instances are named by their shape label (USD visuals: prim path).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import warnings
from collections.abc import Collection
from typing import TYPE_CHECKING, Final, cast, final

import newton
import numpy as np
from newton import GeoType

from raywarp._base.dims import Mask
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.core.post.grading.color import srgb_to_linear
from raywarp.integrations.newton import xform32 as xf
from raywarp.integrations.newton.binding import BindingTables, IntTable
from raywarp.integrations.newton.deformables import report_particles
from raywarp.integrations.newton.types import DEFAULT_OPTIONS, Checkerboard, Options
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.build.mesh_pool import MeshAttributes, finite_attributes
from raywarp.scene.images.colorspace import auto_colorspace
from raywarp.scene.lighting.rig import Lighting, add_lighting
from raywarp.scene.model import geometry
from raywarp.scene.model.material import Material, RaywarpExt
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import (
  NO_MESH,
  NO_TEXTURE,
  AlphaMode,
  Array,
  InstanceKind,
  Json,
  MaterialFlags,
  MeshId,
  Slot,
  TextureId,
  TextureRef,
)
from raywarp.scene.model.xform import pose_matrix

if TYPE_CHECKING:
  from pxr import Usd

  from raywarp._base.dims import AnyDeviceArray
  from raywarp.scene.model.report import LoadReport

_PRIMITIVES: Final = {
  int(GeoType.PLANE): InstanceKind.PLANE,
  int(GeoType.SPHERE): InstanceKind.SPHERE,
  int(GeoType.CAPSULE): InstanceKind.CAPSULE,
  int(GeoType.ELLIPSOID): InstanceKind.ELLIPSOID,
  int(GeoType.CYLINDER): InstanceKind.CYLINDER,
  int(GeoType.BOX): InstanceKind.BOX,
}
_MESHES: Final = {int(GeoType.MESH), int(GeoType.CONVEX_MESH), int(GeoType.CONE), int(GeoType.HFIELD)}
# Every rendered shape type.
_RENDERED_TYPES: Final = (*_PRIMITIVES, *_MESHES)
# Shapes whose source (`model.shape_source`) is their geometry.
_SOURCED: Final = (int(GeoType.MESH), int(GeoType.CONVEX_MESH), int(GeoType.HFIELD))
# Newton shape colors are display (sRGB) values.
_DEFAULT_COLOR: Final = (0.7, 0.7, 0.7)
# Perceptual roughness of every shape without its own (Newton has no material model).
_ROUGHNESS: Final = 0.5
_CONE_SEGMENTS: Final = 32
# `Scene.meta` key of a `load_usd` scene: {"instances": the stage's instance count (the first
# ones), "copies": its copies (merge_worlds: once per Newton world)}.
USD_META: Final = "newton.usd"
# `Scene.meta` key of the instances colored per world (`binding.WorldColors`): their indices.
WORLD_COLORS_META: Final = "newton.world_colors"

# A USD stage: its path or an open Usd.Stage (`load_usd`).
type Stage = str | os.PathLike[str] | Usd.Stage

# A mesh texture image: (h, w) grey or (h, w, 3 | 4) channels, uint8 or floats in [0, 1].
type Image = np.ndarray[[int, int], np.dtype[np.generic]] | np.ndarray[[int, int, int], np.dtype[np.generic]]


def _required[T](value: T | None, name: str) -> T:
  """`value`, or a ValueError naming the missing model array."""
  if value is None:
    raise ValueError(f"raywarp.integrations.newton: the Newton model has no {name}; pass a finalized newton.Model")
  return value


# Rows of the scene: which Newton shape each (world, instance) is ---------------------------------


type Source = newton.Mesh | newton.Heightfield | None
type ShapeIds = np.ndarray[[int], np.dtype[np.intp]]


def _digest(*arrays: object) -> bytes:
  """A digest of arrays' shapes, dtypes and bytes (None: absent)."""
  h = hashlib.blake2b(digest_size=16)
  for a in arrays:
    if a is None:
      h.update(b"\0none")
      continue
    data = np.ascontiguousarray(a)
    h.update(f"{data.dtype}{data.shape}".encode())
    h.update(data.tobytes())
  return h.digest()


def _source_key(source: Source) -> tuple[object, ...]:
  """What a pooled column renders of a mesh or height-field source (two shapes of equal keys share
  one column): its geometry and, for a mesh, its normals, UVs, texture and PBR scalars."""
  if isinstance(source, newton.Heightfield):
    return ("hfield", source.hx, source.hy, source.min_z, source.max_z, _digest(source.data))
  if not isinstance(source, newton.Mesh):
    return ("none", id(source))
  texture = source.texture
  image = os.fspath(texture) if isinstance(texture, (str, os.PathLike)) else _digest(texture)
  geometry = _digest(source.vertices, source.indices, source.normals, source.uvs, source.texture_transform)
  return ("mesh", geometry, image, source.roughness, source.metallic)


# Scales agreeing to this many mantissa bits (~1e-5 relative, `np.isclose`'s default) key alike:
# equal shapes of differently rounded imports pool into one column (two values straddling a
# rounding boundary still key apart: an extra hidden column, never a wrong shape).
_SCALE_BITS: Final = 17


def _row_groups(table: np.ndarray[[int, int], np.dtype[np.int64]]) -> np.ndarray[[int], np.dtype[np.intp]]:
  """(rows,) a group id per row of `table`, equal for equal rows (one lexsort, not a row-wise
  `np.unique`)."""
  order = np.lexsort(table.T[::-1])
  ordered = table[order]
  fresh = np.ones(len(order), bool)
  fresh[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
  out = np.empty(len(order), np.intp)
  out[order] = np.cumsum(fresh) - 1
  return out


def _shape_keys(model: newton.Model, ids: ShapeIds) -> np.ndarray[[int], np.dtype[np.intp]]:
  """(len(ids),) key of each shape `ids`: equal for shapes that render alike (type, scale to
  `_SCALE_BITS` and source by value, `_source_key`; the color is per world, `WorldColors`)."""
  shape_type = host_copy(_required(model.shape_type, "shape_type"))[ids]
  scale = host_copy(_required(model.shape_scale, "shape_scale"))[ids].astype(np.float64)
  mantissa, exponent = np.frexp(scale)
  group = np.full(len(ids), -1, np.int64)
  sourced = np.flatnonzero(np.isin(shape_type, _SOURCED))
  if len(sourced):
    sources = cast("list[Source]", model.shape_source)
    by_id: dict[int, int] = {}  # id(source) -> its value group (each source hashed once)
    groups: dict[tuple[object, ...], int] = {}
    for k in sourced.tolist():
      source = sources[int(ids[k])]
      if id(source) not in by_id:
        by_id[id(source)] = groups.setdefault(_source_key(source), len(groups))
      group[k] = by_id[id(source)]
  quantised = np.round(mantissa * (1 << _SCALE_BITS)).astype(np.int64)
  top = np.abs(quantised) == 1 << _SCALE_BITS  # rounded up to 1: the next exponent's 0.5
  quantised[top] //= 2
  exponent[top] += 1
  table = np.column_stack([shape_type.astype(np.int64), quantised, exponent.astype(np.int64), group])
  return _row_groups(table)


def _renderable(model: newton.Model, shapes: Collection[int] | None, *, non_usd: bool) -> Mask[int]:
  """(shape_count,) mask of the rendered shapes: `shapes` (default visible, non-site) drawn; warns
  about the selected shapes of a type not rendered.

  `non_usd` keeps only the shapes not imported from a stage (labels not starting with "/").
  Shapes of opacity 0 neither show nor block rays (Newton's sensor rule).
  """
  shape_type = host_copy(_required(model.shape_type, "shape_type"))
  visible = _selected(model, shapes, non_usd=non_usd)
  skipped = sorted({GeoType(t).name for t in shape_type[visible] if t not in _RENDERED_TYPES})
  if skipped:
    warnings.warn(f"raywarp.integrations.newton: shape types {skipped} are not rendered", stacklevel=4)
  return visible & np.isin(shape_type, _RENDERED_TYPES)


def _selected(model: newton.Model, shapes: Collection[int] | None, *, non_usd: bool) -> Mask[int]:
  """(shape_count,) `shapes` (default the visible, non-site ones) of opacity > 0 (`_renderable`)."""
  shape_type = host_copy(_required(model.shape_type, "shape_type"))
  if shapes is None:
    flags = host_copy(_required(model.shape_flags, "shape_flags"))
    visible = (flags & int(newton.ShapeFlags.VISIBLE)) != 0
    visible &= (flags & int(newton.ShapeFlags.SITE)) == 0
  else:
    ids = np.asarray(list(shapes), dtype=np.int64)
    if np.any((ids < 0) | (ids >= len(shape_type))):
      # A negative id would select a shape from the end, silently.
      raise ValueError(f"Options.shapes contains an out-of-range shape index (the model has {len(shape_type)} shapes)")
    visible = np.zeros(len(shape_type), dtype=bool)
    visible[ids] = True
  if model.shape_opacity is not None:
    visible &= host_copy(model.shape_opacity) > 0.0
  if non_usd:
    labels = list(model.shape_label) if model.shape_label is not None else [""] * len(shape_type)
    visible &= np.array([not label.startswith("/") for label in labels], dtype=bool)
  return visible


def _world_shapes(shape_world: Array, mask: Mask[int], nworld: int) -> list[ShapeIds]:
  """Per world 0..nworld-1, its shapes in `mask` (ascending ids): one stable sort by world, not a
  pass over every shape per world."""
  ids = np.flatnonzero(mask & (shape_world >= 0) & (shape_world < nworld))
  worlds = shape_world[ids]
  order = ids[np.argsort(worlds, kind="stable")]  # stable: ascending ids within each world
  ends = np.cumsum(np.bincount(worlds, minlength=nworld)).tolist()
  return [order[a:b] for a, b in zip([0, *ends[:-1]], ends, strict=True)]


def pooled(model: newton.Model, per_world: list[ShapeIds]) -> IntTable:
  """(nworld, ncolumn) Newton shape per (world, column), -1 where the world has none: worlds of
  different shapes share one scene of the pooled shapes, each world showing its own.

  The j-th shape of a key (`_shape_keys`) in a world takes that key's j-th column: worlds of the
  same shapes give one column per shape (world 0's, in its order), a world with another object
  adds columns that the other worlds hide. Columns are ordered by first appearance (world 0's
  shapes first).
  """
  nworld = len(per_world)
  counts = np.array([len(p) for p in per_world], np.intp)
  shapes = np.concatenate(per_world).astype(np.intp) if nworld else np.zeros(0, np.intp)
  if not len(shapes):
    return np.zeros((nworld, 0), np.int32)
  world = np.repeat(np.arange(nworld), counts)
  key = _shape_keys(model, shapes)
  # Occurrence of each entry among its world's shapes of the same key (in shape order).
  order = np.lexsort((np.arange(len(shapes)), key, world))
  fresh = np.ones(len(order), bool)
  fresh[1:] = (world[order][1:] != world[order][:-1]) | (key[order][1:] != key[order][:-1])
  starts = np.flatnonzero(fresh)
  occurrence = np.empty(len(shapes), np.intp)
  occurrence[order] = np.arange(len(order)) - np.repeat(starts, np.diff(np.append(starts, len(order))))
  pair = key * (int(occurrence.max()) + 1) + occurrence
  _, first, inverse = np.unique(pair, return_index=True, return_inverse=True)
  rank = np.empty(len(first), np.intp)
  rank[np.argsort(first, kind="stable")] = np.arange(len(first))
  rows = np.full((nworld, len(first)), -1, np.int32)
  rows[world, rank[inverse.reshape(-1)]] = shapes
  return rows


def representatives(rows: IntTable) -> np.ndarray[[int], np.dtype[np.int32]]:
  """(n,) the shape each column of `rows` is built from: its first world's (columns of every world:
  world 0's)."""
  if not rows.size:
    return np.zeros(rows.shape[1], np.int32)
  return rows[np.argmax(rows >= 0, axis=0), np.arange(rows.shape[1])].astype(np.int32)


def shape_rows(model: newton.Model, shapes: Collection[int] | None, *, merge_worlds: bool, non_usd: bool = False) -> IntTable:
  """(rows, n) Newton shape per (world, instance), -1 where the world has none (`pooled`): the
  worlds' pooled shapes then the global ones."""
  renderable = _renderable(model, shapes, non_usd=non_usd)
  if merge_worlds:
    return np.flatnonzero(renderable)[None].astype(np.int32)
  shape_world = host_copy(_required(model.shape_world, "shape_world"))
  nworld = max(model.world_count, 1)
  global_shapes = np.flatnonzero(renderable & (shape_world < 0))
  table = pooled(model, _world_shapes(shape_world, renderable, nworld))
  return np.concatenate([table, np.tile(global_shapes, (nworld, 1))], axis=1).astype(np.int32)


# The model's own shapes ---------------------------------------------------------------------------


def _texture(
  builder: SceneBuilder, texture: str | os.PathLike[str] | Image, max_size: int | None, cache: dict[str | int, TextureId]
) -> TextureId:
  """Host texture of a Newton mesh texture (a path, or an image keyed by its id); NO_TEXTURE if
  unusable."""
  key = os.fspath(texture) if isinstance(texture, (str, os.PathLike)) else id(texture)
  if key not in cache:
    if isinstance(texture, (str, os.PathLike)):
      # A base color file: sRGB if 8-bit, linear if floating point (EXR, HDR).
      space = auto_colorspace(os.fspath(texture), color=True)
      cache[key] = builder.add_texture(texture, colorspace=space, max_size=max_size)
    else:
      # Float images hold display values in [0, 1]: quantised to bytes (grey and RGB stay as
      # they are: the builder widens them).
      image = np.asarray(texture)
      if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
      cache[key] = builder.add_texture(np.ascontiguousarray(image), max_size=max_size)
  return cache[key]


@final
@dataclasses.dataclass
class _ShapeReader:
  """Host copies of the model's shape arrays and the builder's shared meshes / textures."""

  model: newton.Model
  builder: SceneBuilder
  checkerboard: Checkerboard | None
  load_textures: bool
  max_texture_size: int | None
  # Host mesh per cone ("cone", radius, half height) or per mesh source (its id).
  meshes: dict[tuple[str, float, float] | int, MeshId] = dataclasses.field(default_factory=dict)
  textures: dict[str | int, TextureId] = dataclasses.field(default_factory=dict)
  # Whether the host mesh of a mesh source (its id) kept its UVs.
  mesh_uvs: dict[int, bool] = dataclasses.field(default_factory=dict)
  checker_texture: TextureId = NO_TEXTURE

  def __post_init__(self) -> None:
    """Reads the shape arrays once."""
    m = self.model
    self.shape_type = host_copy(_required(m.shape_type, "shape_type"))
    self.scale = host_copy(_required(m.shape_scale, "shape_scale")).astype(np.float64)
    self.body = host_copy(_required(m.shape_body, "shape_body"))
    self.transform = host_copy(_required(m.shape_transform, "shape_transform")).astype(np.float64)
    self.world = host_copy(_required(m.shape_world, "shape_world"))
    self.color = host_copy(m.shape_color) if m.shape_color is not None else None
    self.opacity = host_copy(m.shape_opacity) if m.shape_opacity is not None else None
    self.body_q = host_copy(m.body_q).astype(np.float64) if m.body_q is not None and m.body_count else np.zeros((0, 7))
    self.checkered = frozenset() if self.checkerboard is None else self.checkerboard.shapes

  def alpha(self, shape: int) -> float:
    """Display opacity of a shape."""
    return float(self.opacity[shape]) if self.opacity is not None else 1.0

  def material(self, shape: int) -> Material:
    """OpenPBR material of a shape: its linear color and opacity (BLEND below 1), a dielectric
    of roughness 0.5 (Newton's shapes carry neither)."""
    srgb = self.color[shape] if self.color is not None else _DEFAULT_COLOR
    alpha = self.alpha(shape)
    r, g, b = (float(c) for c in srgb_to_linear(np.asarray(srgb, np.float64)[:3]))
    mode = AlphaMode.BLEND if alpha < 1.0 else AlphaMode.OPAQUE
    return Material(
      base_color=(r, g, b),
      geometry_opacity=alpha,
      base_metalness=0.0,
      specular_roughness=_ROUGHNESS,
      ext=RaywarpExt(alpha_mode=mode),
    )

  def cone(self, scale: np.ndarray[[3], np.dtype[np.float64]]) -> MeshId:
    """Host mesh of a cone shape (one per radius / half height)."""
    key = ("cone", float(scale[0]), float(scale[1]))
    if key not in self.meshes:
      points, tris, normals, normal_idx = geometry.cone_mesh(float(scale[0]), 2.0 * float(scale[1]), _CONE_SEGMENTS)
      self.meshes[key] = self.builder.add_mesh(points, tris, normals=normals, normal_idx=normal_idx)
    return self.meshes[key]

  def _per_vertex(
    self, values: np.ndarray[[int, int], np.dtype[np.float32]] | None, pool: str, nvert: int, where: str
  ) -> np.ndarray[[int, int], np.dtype[np.float32]] | None:
    """A newton.Mesh's per-vertex `pool` (normals / uvs) as float32; None when absent, or dropped
    and reported (`newton.Mesh.<pool>.length`) when it does not hold one value per vertex."""
    if values is None:
      return None
    out = np.asarray(values, np.float32)
    if len(out) != nvert:
      self.builder.report.drop(f"newton.Mesh.{pool}.length", f"{where} ({len(out)} {pool} for {nvert} vertices)")
      return None
    return out

  def _add_mesh(self, source: newton.Mesh, where: str) -> None:
    """The host mesh of a newton.Mesh, once per source: wrong-length or broken (NaN / inf) normals
    and UVs are dropped and reported once (flat shading, no texture)."""
    verts = np.asarray(source.vertices, np.float32)
    normals = self._per_vertex(source.normals, "normals", len(verts), where)
    uvs = self._per_vertex(source.uvs, "uvs", len(verts), where)
    attributes = finite_attributes(MeshAttributes(normals=normals, uvs=uvs), self.builder.report, where, source="newton.Mesh")
    tris = np.asarray(source.indices, np.int32).reshape(-1, 3)
    self.meshes[id(source)] = self.builder.add_mesh(verts, tris, **attributes)
    self.mesh_uvs[id(source)] = attributes.get("uvs") is not None

  def height_field(self, shape: int) -> MeshId:
    """Host mesh of a height field shape (one per source): Newton's own watertight mesh of it
    (`newton.Mesh.create_heightfield`, as Newton's viewer draws it), in shape units."""
    source = self.model.shape_source[shape]
    if not isinstance(source, newton.Heightfield):
      raise TypeError(f"raywarp.integrations.newton: height field shape {shape} has no newton.Heightfield source")
    if id(source) not in self.meshes:
      heights = source.min_z + np.asarray(source.data, np.float64) * (source.max_z - source.min_z)
      mesh = newton.Mesh.create_heightfield(
        heights.T, 2.0 * source.hx, 2.0 * source.hy, ground_z=source.min_z, compute_inertia=False
      )
      verts = np.asarray(mesh.vertices, np.float32)
      self.meshes[id(source)] = self.builder.add_mesh(verts, np.asarray(mesh.indices, np.int32).reshape(-1, 3))
    return self.meshes[id(source)]

  def mesh(self, shape: int, material: Material) -> tuple[MeshId, Material]:
    """Host mesh of a mesh shape (one per source), and `material` with its PBR scalars / texture."""
    source = self.model.shape_source[shape]
    if not isinstance(source, newton.Mesh):
      raise TypeError(
        f"raywarp.integrations.newton: mesh shape {shape} has no newton.Mesh source (got {type(source).__name__})"
      )
    if id(source) not in self.meshes:
      self._add_mesh(source, f"shape{shape}")
    if source.roughness is not None:
      material = dataclasses.replace(material, specular_roughness=float(source.roughness))
    if source.metallic is not None:
      material = dataclasses.replace(material, base_metalness=float(source.metallic))
    if self.load_textures and self.mesh_uvs[id(source)] and source.texture is not None and shape not in self.checkered:
      tex = _texture(self.builder, source.texture, self.max_texture_size, self.textures)
      if tex >= 0:
        m = np.asarray(source.texture_transform, np.float64).reshape(2, 3)
        uv = ((float(m[0, 0]), float(m[0, 1]), float(m[0, 2])), (float(m[1, 0]), float(m[1, 1]), float(m[1, 2])))
        # The texture replaces the color (the opacity stays the shape's).
        material = dataclasses.replace(
          material, base_color=(1.0, 1.0, 1.0), textures=((Slot.BASE_COLOR, TextureRef(tex, uv_xform=uv)),)
        )
    return self.meshes[id(source)], material

  def checker(self, shape: int, material: Material) -> Material:
    """`material` with the checkerboard texture (planes: in world units, MuJoCo's plane mapping of
    one texture per 2 m, so the default two squares per texture are 1 m, as ViewerGL's)."""
    if self.checkerboard is None or shape not in self.checkered:
      return material
    if self.checker_texture < 0:
      self.checker_texture = self.builder.add_texture(self.checkerboard.image())
    flags = int(MaterialFlags.UV_WORLD) if int(self.shape_type[shape]) == int(GeoType.PLANE) else material.ext.flags
    return dataclasses.replace(
      material,
      textures=((Slot.BASE_COLOR, TextureRef(self.checker_texture)),),
      ext=dataclasses.replace(material.ext, flags=flags),
    )

  def add(self, shape: int, *, tint: bool = False) -> bool:
    """Adds shape `shape` (posed at the model's initial state) as one instance; with `tint`, an
    untextured one gets a white material (its color comes per world from the instance color,
    `binding.WorldColors`). Whether it was tinted."""
    gtype = int(self.shape_type[shape])
    scale = self.scale[shape]
    material = self.material(shape)
    body = int(self.body[shape])
    pose = xf.compose(self.body_q[body], self.transform[shape]) if body >= 0 else self.transform[shape]
    xform = pose_matrix(pose)
    mesh = NO_MESH
    size = (0.0, 0.0, 0.0)
    if gtype in _PRIMITIVES:
      kind = _PRIMITIVES[gtype]
      sx, sy, sz = (float(v) for v in scale)
      size = (0.5 * sx, 0.5 * sy, 0.0) if gtype == int(GeoType.PLANE) else (sx, sy, sz)
    elif gtype == int(GeoType.CONE):
      kind, mesh = InstanceKind.MESH, self.cone(scale)
    elif gtype == int(GeoType.HFIELD):
      kind, mesh = InstanceKind.MESH, self.height_field(shape)
      # Newton scales a height field's extents and heights (`add_shape_heightfield(scale=)`).
      xform[:, :3] = xform[:, :3] * scale[None, :]
    else:
      kind = InstanceKind.MESH
      mesh, material = self.mesh(shape, material)
      # Mesh vertices are in shape units: the scale goes into the instance transform.
      xform[:, :3] = xform[:, :3] * scale[None, :]
    tint = tint and not material.textures  # a mesh texture replaces the color
    if tint:
      material = dataclasses.replace(material, base_color=(1.0, 1.0, 1.0))
    mat = self.builder.add_material(self.checker(shape, material))
    label = (self.model.shape_label[shape] if self.model.shape_label is not None else "") or f"shape{shape}"
    self.builder.add_instance(kind, mesh=mesh, material=mat, size=size, xform=xform, path=label, source=shape)
    return tint


def _differs(table: AnyDeviceArray, rows: IntTable) -> Mask[int]:
  """(n,) the columns of `rows` where some world's entry of the per-shape `table` differs from its
  column's representative (`representatives`; absent entries, -1, do not count)."""
  values = host_copy(table)
  first = values[representatives(rows)][None]  # (1, n, ...)
  present = (rows >= 0).reshape(*rows.shape, *(1,) * (values.ndim - 1))
  picked = np.where(present, values[np.maximum(rows, 0)], first)
  return (picked != first).reshape(*rows.shape, -1).any(axis=(0, 2))


def report_world_appearance(model: newton.Model, rows: IntTable, report: LoadReport, *, colors: bool) -> None:
  """Counts in `report` the instances whose shape opacity (and, with `colors`, color: without
  `Options.world_colors`) differs between the worlds showing them: the scene holds each column's
  representative shape (`model_shapes`: the first world's showing it, world 0's for shapes of
  every world), so every world shows its (`newton.shape_opacity(per world: world 0's shown)`;
  `scene` calls it)."""
  if len(rows) < 2 or rows.shape[1] == 0:  # one world (or merged worlds), or no shapes
    return
  tables = (("shape_color", model.shape_color),) if colors else ()
  for name, table in (*tables, ("shape_opacity", model.shape_opacity)):
    if table is None:
      continue
    differs = _differs(table, rows)
    if differs.any():
      first = representatives(rows)[np.flatnonzero(differs)[0]].item()
      report.drop(f"newton.{name}(per world: world 0's shown)", f"shape {first}", int(differs.sum()))


def model_shapes(  # noqa: PLR0913  # the model, its rows, the builder and keyword-only options
  model: newton.Model,
  rows: IntTable,
  builder: SceneBuilder,
  *,
  adapter: Options,
  max_texture_size: int | None,
  tint: Mask[int] | None = None,
) -> list[int]:
  """Adds the representative shape of every column of `rows` (`representatives`) to `builder`
  (bound by `binding`, from `Instances.source`), with `adapter`'s checkerboard and textures,
  decoded at most `max_texture_size` (None: no cap); the columns `tint` marks get a white material
  where untextured (`_ShapeReader.add`). The indices of the tinted instances."""
  reader = _ShapeReader(
    model, builder, adapter.checkerboard, load_textures=adapter.load_textures, max_texture_size=max_texture_size
  )
  tinted = []
  for k, shape in enumerate(representatives(rows).tolist()):
    instance = builder.ninstance
    if reader.add(shape, tint=tint is not None and bool(tint[k])):
      tinted.append(instance)
  return tinted


def varying_colors(model: newton.Model, rows: IntTable) -> Mask[int]:
  """(n,) whether column k's `shape_color` differs between the rows (Newton worlds) showing it."""
  if model.shape_color is None or len(rows) < 2 or rows.shape[1] == 0:  # one world
    return np.zeros(rows.shape[1], bool)
  return _differs(model.shape_color, rows)


def absent(rows: IntTable) -> Mask[int]:
  """(n,) the columns of `rows` some world does not show (pooled shapes of other worlds)."""
  return np.any(rows < 0, axis=0)


# The binding transform of an absent (row, column): the identity (the instance is hidden there).
_IDENTITY: Final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)


def shape_tables(model: newton.Model, rows: IntTable) -> BindingTables:
  """Binding tables of the shapes `rows` (rows, n): their bodies, local transforms and worlds (an
  absent entry, -1: no body, the identity, no world, no shape)."""
  body = host_copy(_required(model.shape_body, "shape_body"))
  transform = host_copy(_required(model.shape_transform, "shape_transform"))
  world = host_copy(_required(model.shape_world, "shape_world"))
  present = rows >= 0
  at = np.maximum(rows, 0)
  return BindingTables(
    np.where(present, body[at], -1).astype(np.int32),
    np.where(present[..., None], transform[at], _IDENTITY).astype(np.float32),
    np.where(present, world[at], -1).astype(np.int32),
    np.ascontiguousarray(rows, np.int32),
  )


# Binding ------------------------------------------------------------------------------------------


def source_rows(
  model: newton.Model,
  shapes: np.ndarray[[int], np.dtype[np.int32]],
  *,
  merge_worlds: bool,
  selected: Collection[int] | None = None,
  non_usd: bool = False,
) -> IntTable:
  """(rows, n) Newton shape per (world, instance) of a scene's instances of Newton shapes
  `shapes`: the column of the per-world shapes (`pooled`) each represents, -1 where a world has
  none; global shapes in every row. The rendered shapes are `selected` (the `shapes` the scene was
  built with; default the visible, non-site ones), `non_usd`: outside the stage of a `load_usd`
  scene.

  Raises:
    ValueError: for a world shape that represents no column (a scene built with merge_worlds, or
      with other `shapes`), or a column no scene instance represents (a scene of another model).
  """
  if merge_worlds:
    return shapes[None].astype(np.int32)
  shape_world = host_copy(_required(model.shape_world, "shape_world"))
  nworld = max(model.world_count, 1)
  shape_type = host_copy(_required(model.shape_type, "shape_type"))
  rendered = _selected(model, selected, non_usd=non_usd) & np.isin(shape_type, _RENDERED_TYPES)
  table = pooled(model, _world_shapes(shape_world, rendered, nworld))
  position = np.full(len(shape_world), -1, np.intp)  # representative shape -> its column
  position[representatives(table)] = np.arange(table.shape[1])
  local = shape_world[shapes] >= 0  # world shapes (global ones stay in every row)
  unknown = local & (position[shapes] < 0)
  if np.any(unknown):
    raise ValueError(
      f"binding: shape {int(shapes[unknown][0])} of world {int(shape_world[shapes[unknown][0]])} stands for no per-world"
      " column: bind with the `shapes` and merge_worlds the scene was built with (merge_worlds=True for a merged scene)"
    )
  covered = np.zeros(table.shape[1], bool)
  covered[position[shapes[local]]] = True
  if not covered.all():
    missing = int(representatives(table)[np.flatnonzero(~covered)[0]])
    raise ValueError(
      f"binding: the scene has no instance of shape {missing} (world {int(shape_world[missing])}): it was built from"
      " another model or other `shapes` (rebuild it, e.g. a stale cached scene)"
    )
  rows = np.repeat(shapes[None].astype(np.int32), nworld, axis=0)
  rows[:, local] = table[:, position[shapes[local]]]
  return rows


def load(
  model: newton.Model,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
) -> Scene:
  """Ingest: host `Scene` of a Newton model's shapes (`api.binding(scene, model, adapter=adapter)`
  binds it to the bodies; `api.scene` does both and uploads).

  Args:
    model: the finalized Newton model.
    options: the options every adapter shares (`rw.SceneOptions`; its texture size cap applies
      at decode).
    adapter: the Newton options (`raywarp.integrations.newton.Options`).
    lighting: lights, ambient, environment and headlight added (Newton models carry none).
  """
  rows = shape_rows(model, adapter.shapes, merge_worlds=adapter.merge_worlds)
  return ingest(model, rows, options=options, adapter=adapter, lighting=lighting)


def ingest(model: newton.Model, rows: IntTable, *, options: SceneOptions, adapter: Options, lighting: Lighting | None) -> Scene:
  """`load` of the shapes `rows` (`shape_rows`; `api.scene` computes them once for both)."""
  builder = SceneBuilder()
  # Ingest; the binding is rebuilt from the instances' shapes (`api.binding`).
  tint = varying_colors(model, rows) if adapter.world_colors else None
  tinted = model_shapes(model, rows, builder, adapter=adapter, max_texture_size=options.max_texture_size, tint=tint)
  if tinted:
    ids: list[Json] = list(tinted)  # (`Json` lists are invariant)
    builder.meta[WORLD_COLORS_META] = ids
  if adapter.particle_meshes:
    report_particles(model, builder.report)
  if lighting is not None:
    add_lighting(builder, lighting)
  return builder.build()
