"""The geometry, material and image tables of a host scene (see `table.py` for the layout)."""

from __future__ import annotations

import dataclasses
import hashlib
import operator
from collections.abc import Sequence
from typing import ClassVar, Final, final, override

import numpy as np
import numpy.typing as npt

from raywarp.scene.model.material import INPUTS, Material, OpenPbrExtras, RaywarpExt
from raywarp.scene.model.table import F32, F64, I8, I16, I32, I64, U8, U32, StringTable, Table, record_columns
from raywarp.scene.model.types import (
  ALIGN,
  IDENTITY_UV,
  NSLOT,
  PIXEL_DTYPES,
  Array,
  Column,
  Matrices,
  PixelType,
  Rows,
  TextureRef,
)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class MeshView:
  """Views of one mesh's attributes (no copy); see `Meshes` for the layout."""

  points: Rows[np.float32, 3]
  tris: Rows[np.int32, 3]
  normals: Rows[np.float32, 3]
  normal_idx: Rows[np.int32, 3]
  uvs: Rows[np.float32, 2]
  uv_idx: Rows[np.int32, 3]
  uvs1: Rows[np.float32, 2]
  uv1_idx: Rows[np.int32, 3]
  colors: Rows[np.float32, 4]
  color_idx: Rows[np.int32, 3]


@final
@dataclasses.dataclass
class Meshes(Table):
  """Triangle meshes in local coordinates, corner-indexed attributes.

  Row i uses `points[vert_adr[i] : vert_adr[i] + vert_num[i]]` and triangles
  `tris[tri_adr[i] : tri_adr[i] + tri_num[i]]` (indices local to the mesh's vertices). Normals
  and UVs are separate pools indexed per triangle corner (`normal_idx`, `uv_idx`, local to the
  mesh's `normal_adr` / `uv_adr`): USD faceVarying data and seams need no vertex unwelding.
  `normal_num == 0` = no authored normals (flat shading); `uv_num == 0` = no UVs. Meshes split
  by material (GeomSubsets) share their vertex, normal and UV ranges.

  A second UV set (the one a material's texture names, e.g. USD primvars:st1, glTF TEXCOORD_1)
  is stored in the same `uvs` pool at `uv1_adr` (`uv1_num` values, 0 = none) with its own
  corner indices `uv1_idx`. Vertex colors (linear RGBA, multiplying the base color of
  materials with MaterialFlags.VERTEX_COLOR) are the `colors` pool at `color_adr` with corner
  indices `color_idx` (`color_num == 0`: none). Index rows of meshes without the attribute are -1.
  """

  # The per-mesh address / count columns (`mesh_pool.MeshRow`), then the pools.
  SCHEMA: ClassVar = {
    "vert_adr": (I32, ()),
    "vert_num": (I32, ()),
    "tri_adr": (I32, ()),
    "tri_num": (I32, ()),
    "normal_adr": (I32, ()),
    "normal_num": (I32, ()),
    "uv_adr": (I32, ()),
    "uv_num": (I32, ()),
    "uv1_adr": (I32, ()),
    "uv1_num": (I32, ()),
    "color_adr": (I32, ()),
    "color_num": (I32, ()),
    "aabb": (F32, (2, 3)),
    "points": (F32, (3,)),
    "tris": (I32, (3,)),
    "normals": (F32, (3,)),
    "normal_idx": (I32, (3,)),
    "uvs": (F32, (2,)),
    "uv_idx": (I32, (3,)),
    "uv1_idx": (I32, (3,)),
    "colors": (F32, (4,)),
    "color_idx": (I32, (3,)),
  }
  vert_adr: Column[np.int32]
  vert_num: Column[np.int32]
  tri_adr: Column[np.int32]
  tri_num: Column[np.int32]
  normal_adr: Column[np.int32]
  normal_num: Column[np.int32]
  uv_adr: Column[np.int32]
  uv_num: Column[np.int32]
  uv1_adr: Column[np.int32]
  uv1_num: Column[np.int32]
  color_adr: Column[np.int32]
  color_num: Column[np.int32]
  aabb: Matrices[np.float32, 2, 3]
  points: Rows[np.float32, 3]
  tris: Rows[np.int32, 3]
  normals: Rows[np.float32, 3]
  normal_idx: Rows[np.int32, 3]
  uvs: Rows[np.float32, 2]
  uv_idx: Rows[np.int32, 3]
  uv1_idx: Rows[np.int32, 3]
  colors: Rows[np.float32, 4]
  color_idx: Rows[np.int32, 3]

  def stored_rows(self) -> dict[str, Rows[np.int32, 3]]:
    """The optional corner-index tables (`SPARSE_ROWS`) as a `.rayw` stores them: the rows of the
    meshes with the attribute only (`expanded_rows` restores them).

    Raises:
      ValueError: for a row of a mesh without the attribute that is not -1 (the table's rule).
    """
    out: dict[str, Rows[np.int32, 3]] = {}
    for name, count in SPARSE_ROWS.items():
      rows = getattr(self, name)
      if not len(rows):
        out[name] = rows
        continue
      has = _rows_with(self.tri_adr, self.tri_num, getattr(self, count), len(rows))
      if (rows[~has] != -1).any():
        raise ValueError(f"Meshes.{name}: rows of meshes without the attribute ({count} 0) must be -1")
      out[name] = rows if has.all() else np.ascontiguousarray(rows[has])
    return out

  @staticmethod
  def expanded_rows(arrays: dict[str, Array]) -> dict[str, Array]:
    """`arrays` (a `.rayw`'s mesh columns) with the optional corner-index tables of `stored_rows`
    restored to one row per triangle (-1 for the meshes without the attribute). A table no mesh
    has is a read-only broadcast of -1 (no memory: 12 B a triangle otherwise, ~110 MB on a 9 M
    triangle stage); a partial one is expanded."""
    ntri = len(arrays["tris"])
    out = dict(arrays)
    for name, count in SPARSE_ROWS.items():
      stored = arrays[name]
      if len(stored) == ntri:
        continue
      if len(stored) == 0:
        out[name] = np.broadcast_to(np.full(1, -1, np.int32), (ntri, 3))
        continue
      has = _rows_with(arrays["tri_adr"], arrays["tri_num"], arrays[count], ntri)
      rows = np.full((ntri, 3), -1, np.int32)
      rows[has] = stored
      out[name] = rows
    return out

  def get(self, i: int) -> MeshView:
    """Views of mesh `i`'s attributes."""
    v, nv, t, nt = int(self.vert_adr[i]), int(self.vert_num[i]), int(self.tri_adr[i]), int(self.tri_num[i])
    n, nn, u, nu = int(self.normal_adr[i]), int(self.normal_num[i]), int(self.uv_adr[i]), int(self.uv_num[i])
    u1, nu1, c, nc = int(self.uv1_adr[i]), int(self.uv1_num[i]), int(self.color_adr[i]), int(self.color_num[i])
    return MeshView(
      points=self.points[v : v + nv],
      tris=self.tris[t : t + nt],
      normals=self.normals[n : n + nn],
      normal_idx=self.normal_idx[t : t + nt],
      uvs=self.uvs[u : u + nu],
      uv_idx=self.uv_idx[t : t + nt],
      uvs1=self.uvs[u1 : u1 + nu1],
      uv1_idx=self.uv1_idx[t : t + nt],
      colors=self.colors[c : c + nc],
      color_idx=self.color_idx[t : t + nt],
    )


# Corner-index tables of optional attributes -> their per-mesh count column: a `.rayw` stores the
# rows of the meshes with the attribute only (`Meshes.stored_rows`).
SPARSE_ROWS: Final = {"uv1_idx": "uv1_num", "color_idx": "color_num"}


def _rows_with(
  tri_adr: Column[np.int32], tri_num: Column[np.int32], num: Column[np.int32], ntri: int
) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(ntri,) the triangle rows of the meshes whose attribute count `num` is positive."""
  starts = np.asarray(tri_adr, np.int64)[np.asarray(num) > 0]
  ends = starts + np.asarray(tri_num, np.int64)[np.asarray(num) > 0]
  edges = np.zeros(ntri + 1, np.int64)
  np.add.at(edges, starts, 1)
  np.add.at(edges, ends, -1)
  return np.cumsum(edges[:-1]) > 0


@final
@dataclasses.dataclass
class Instances(Table):
  """Placed shapes. `xform` is the world transform (scale included, non-uniform allowed).

  `prim` / `body` index the `paths` / `bodies` string tables: the source prim and the nearest
  rigid-body ancestor (the Newton adapter binds instances to the bodies of that label). `source`
  is the instance's id in what its adapter binds (a Newton shape index, -1: none): the source-id
  column a `binding()` entry rebuilds a binding from. `label` indexes the `labels` string table:
  the instance's semantic class (UsdSemantics, -1: none), the default of `Renderer.set_semantics`.
  """

  SCHEMA: ClassVar = {
    "kind": (U8, ()),
    "mesh": (I32, ()),
    "material": (I32, ()),
    "size": (F32, (3,)),
    "xform": (F32, (3, 4)),
    "color": (F32, (4,)),
    "flags": (U32, ()),
    "segment": (I32, ()),
    "source": (I32, ()),
    "prim": (I32, ()),
    "body": (I32, ()),
    "label": (I32, ()),
  }
  STRINGS: ClassVar = ("paths", "bodies", "labels")
  kind: Column[np.uint8]
  mesh: Column[np.int32]
  material: Column[np.int32]
  size: Rows[np.float32, 3]
  xform: Matrices[np.float32, 3, 4]
  color: Rows[np.float32, 4]
  flags: Column[np.uint32]
  segment: Column[np.int32]
  source: Column[np.int32]
  prim: Column[np.int32]
  body: Column[np.int32]
  label: Column[np.int32]
  paths_blob: Column[np.uint8]
  paths_offsets: Column[np.int64]
  bodies_blob: Column[np.uint8]
  bodies_offsets: Column[np.int64]
  labels_blob: Column[np.uint8]
  labels_offsets: Column[np.int64]

  @property
  def paths(self) -> StringTable:
    """Source prim paths (`prim` indexes them)."""
    return self.strings("paths")

  @property
  def bodies(self) -> StringTable:
    """Rigid-body paths (`body` indexes them; lights share the table)."""
    return self.strings("bodies")

  @property
  def labels(self) -> StringTable:
    """Semantic class labels (`label` indexes them)."""
    return self.strings("labels")

  def label_of(self) -> list[str]:
    """Every instance's semantic label ("" for none)."""
    labels = self.labels.tolist()
    return [labels[int(i)] if i >= 0 else "" for i in self.label]


# The `RaywarpExt` fields, stored as `ext_<field>` columns.
_EXT_FIELDS: Final = tuple(f.name for f in dataclasses.fields(RaywarpExt))


def _texture_columns(refs: Sequence[Sequence[tuple[int, TextureRef]]], width: int | None) -> dict[str, Array]:
  """The TextureRef columns of `refs`: per row, (key, ref) pairs; `width` keys per row ((n, width,
  ...) columns indexed by key: material slots), or None (one column row per pair: a pool)."""
  n = len(refs) if width is not None else sum(len(r) for r in refs)
  lead: tuple[int, ...] = (n,) if width is None else (n, width)
  cols: dict[str, Array] = {
    "tex": np.full(lead, -1, np.int32),
    "tex_channel": np.full(lead, -1, np.int8),
    "tex_scale": np.ones((*lead, 4), np.float32),
    "tex_bias": np.zeros((*lead, 4), np.float32),
    "uv_xform": np.broadcast_to(IDENTITY_UV, (*lead, 2, 3)).copy(),
    "uv_set": np.zeros(lead, np.uint8),
  }
  k = 0
  for i, pairs in enumerate(refs):
    for key, ref in pairs:
      at = k if width is None else (i, key)
      cols["tex"][at], cols["tex_channel"][at], cols["uv_set"][at] = ref.texture, ref.channel, ref.uv_set
      cols["tex_scale"][at], cols["tex_bias"][at], cols["uv_xform"][at] = ref.scale, ref.bias, ref.uv_xform
      k += 1
  return cols


@final
@dataclasses.dataclass
class Materials(Table):
  """The canonical OpenPBR materials (`scene.model.material.Material`).

  Columns: the `Material` fields (`geometry_thin_walled` as U8; `specular_weight` float64, the
  input of a non-linear derivation), `ext_<field>` the `RaywarpExt` fields, `extras` the row of
  the material's `OpenPbrExtras` in `MaterialExtras` (-1: none), and the texture slots as
  (M, NSLOT) columns.
  """

  SCHEMA: ClassVar = {
    "base_weight": (F32, ()),
    "base_color": (F32, (3,)),
    "base_metalness": (F32, ()),
    "specular_weight": (F64, ()),
    "specular_color": (F32, (3,)),
    "specular_roughness": (F32, ()),
    "specular_ior": (F32, ()),
    "transmission_weight": (F32, ()),
    "transmission_color": (F32, (3,)),
    "coat_weight": (F32, ()),
    "coat_roughness": (F32, ()),
    "coat_ior": (F32, ()),
    "emission_luminance": (F32, ()),
    "emission_color": (F32, (3,)),
    "geometry_opacity": (F32, ()),
    "geometry_thin_walled": (U8, ()),
    "ext_alpha_mode": (U8, ()),
    "ext_alpha_cutoff": (F32, ()),
    "ext_occlusion_strength": (F32, ()),
    "ext_mirror": (F32, ()),
    "ext_flags": (U32, ()),
    "ext_anisotropy_rotation": (F32, ()),
    "ext_wrap": (F32, ()),
    "extras": (I32, ()),
    "tex": (I32, (NSLOT,)),
    "tex_channel": (I8, (NSLOT,)),
    "tex_scale": (F32, (NSLOT, 4)),
    "tex_bias": (F32, (NSLOT, 4)),
    "uv_xform": (F32, (NSLOT, 2, 3)),
    "uv_set": (U8, (NSLOT,)),
  }
  STRINGS: ClassVar = ("names",)
  base_weight: Column[np.float32]
  base_color: Rows[np.float32, 3]
  base_metalness: Column[np.float32]
  specular_weight: Column[np.float64]
  specular_color: Rows[np.float32, 3]
  specular_roughness: Column[np.float32]
  specular_ior: Column[np.float32]
  transmission_weight: Column[np.float32]
  transmission_color: Rows[np.float32, 3]
  coat_weight: Column[np.float32]
  coat_roughness: Column[np.float32]
  coat_ior: Column[np.float32]
  emission_luminance: Column[np.float32]
  emission_color: Rows[np.float32, 3]
  geometry_opacity: Column[np.float32]
  geometry_thin_walled: Column[np.uint8]
  ext_alpha_mode: Column[np.uint8]
  ext_alpha_cutoff: Column[np.float32]
  ext_occlusion_strength: Column[np.float32]
  ext_mirror: Column[np.float32]
  ext_flags: Column[np.uint32]
  ext_anisotropy_rotation: Column[np.float32]
  ext_wrap: Column[np.float32]
  extras: Column[np.int32]
  # Texture slots (the second dimension is NSLOT).
  tex: Rows[np.int32, 9]
  tex_channel: Rows[np.int8, 9]
  tex_scale: Matrices[np.float32, 9, 4]
  tex_bias: Matrices[np.float32, 9, 4]
  uv_xform: np.ndarray[[int, 9, 2, 3], np.dtype[np.float32]]
  uv_set: Rows[np.uint8, 9]
  names_blob: Column[np.uint8]
  names_offsets: Column[np.int64]

  @property
  def names(self) -> StringTable:
    """Source names of the materials (diagnostics)."""
    return self.strings("names")

  def emission(self) -> Rows[np.float64, 3]:
    """(m, 3) emitted radiance `emission_color` x `emission_luminance` [nits] (float64)."""
    return np.asarray(self.emission_color, np.float64) * np.asarray(self.emission_luminance, np.float64)[:, None]

  @staticmethod
  def from_materials(mats: Sequence[Material]) -> Materials:
    """Table of `mats` (fields as columns, `ext` flattened, texture slots as (M, NSLOT) columns;
    the `extras` rows number the materials with extras in order, as
    `MaterialExtras.from_materials`)."""
    fields = [name for name in Materials.SCHEMA if not name.startswith(("tex", "uv_", "ext"))]
    cols: dict[str, npt.ArrayLike] = dict(record_columns(mats, fields))
    for name in _EXT_FIELDS:
      get = operator.attrgetter(name)
      cols[f"ext_{name}"] = [get(mat.ext) for mat in mats]
    has = np.array([mat.extras is not None for mat in mats], bool)
    cols["extras"] = np.where(has, np.cumsum(has) - 1, -1)
    cols.update(_texture_columns([mat.textures for mat in mats], NSLOT))
    return Materials.from_columns(cols, {"names": [mat.name for mat in mats]})


# The `OpenPbrExtras` scalar / color inputs (scene/model/material.py), in field order.
_EXTRA_SCALARS: Final = tuple(f.name for f in dataclasses.fields(OpenPbrExtras) if type(f.default) is float)
_EXTRA_COLORS: Final = tuple(f.name for f in dataclasses.fields(OpenPbrExtras) if f.name in INPUTS and type(f.default) is tuple)


@final
@dataclasses.dataclass
class MaterialExtras(Table):
  """The `OpenPbrExtras` of the materials that have some.

  Rows of the extras columns: one per material with extras (`Materials.extras` indexes them). The
  `tex_*` columns are a pool of their own length, one row per texture bound to an extra input:
  `tex_input` indexes `openpbr.INPUTS`, `tex_material` the material, then the TextureRef fields.
  """

  SCHEMA: ClassVar = {
    **dict.fromkeys(_EXTRA_SCALARS, (F32, ())),
    **dict.fromkeys(_EXTRA_COLORS, (F32, (3,))),
    "tex_input": (I16, ()),
    "tex_material": (I32, ()),
    "tex": (I32, ()),
    "tex_channel": (I8, ()),
    "tex_scale": (F32, (4,)),
    "tex_bias": (F32, (4,)),
    "uv_xform": (F32, (2, 3)),
    "uv_set": (U8, ()),
  }
  base_diffuse_roughness: Column[np.float32]
  specular_roughness_anisotropy: Column[np.float32]
  transmission_depth: Column[np.float32]
  transmission_scatter_anisotropy: Column[np.float32]
  transmission_dispersion_scale: Column[np.float32]
  transmission_dispersion_abbe_number: Column[np.float32]
  subsurface_weight: Column[np.float32]
  subsurface_radius: Column[np.float32]
  subsurface_scatter_anisotropy: Column[np.float32]
  fuzz_weight: Column[np.float32]
  fuzz_roughness: Column[np.float32]
  coat_darkening: Column[np.float32]
  coat_roughness_anisotropy: Column[np.float32]
  thin_film_weight: Column[np.float32]
  thin_film_thickness: Column[np.float32]
  thin_film_ior: Column[np.float32]
  transmission_scatter: Rows[np.float32, 3]
  subsurface_color: Rows[np.float32, 3]
  subsurface_radius_scale: Rows[np.float32, 3]
  fuzz_color: Rows[np.float32, 3]
  coat_color: Rows[np.float32, 3]
  tex_input: Column[np.int16]
  tex_material: Column[np.int32]
  tex: Column[np.int32]
  tex_channel: Column[np.int8]
  tex_scale: Rows[np.float32, 4]
  tex_bias: Rows[np.float32, 4]
  uv_xform: Matrices[np.float32, 2, 3]
  uv_set: Column[np.uint8]

  @staticmethod
  def from_materials(mats: Sequence[Material]) -> MaterialExtras:
    """The extras of `mats` (rows in material order, as `Materials.extras` numbers them)."""
    owned = [(i, mat.extras) for i, mat in enumerate(mats) if mat.extras is not None]
    get = {name: operator.attrgetter(name) for name in (*_EXTRA_SCALARS, *_EXTRA_COLORS)}
    cols: dict[str, npt.ArrayLike] = {name: [g(extra) for _, extra in owned] for name, g in get.items()}
    index = {name: k for k, name in enumerate(INPUTS)}
    pairs = [[(index[name], ref) for name, ref in extra.textures] for _, extra in owned]
    cols.update(_texture_columns(pairs, None))
    cols["tex_input"] = [k for p in pairs for k, _ in p]
    cols["tex_material"] = [i for (i, _), p in zip(owned, pairs, strict=True) for _ in p]
    return MaterialExtras.from_columns(cols)


@final
@dataclasses.dataclass
class Textures(Table):
  """Samplers: an image, the color space of its texels (`colorspace`:
  `scene.images.colorspace.code`, the index in the frozen `COLOR_SPACES`; read with `from_code`) and
  how coordinates wrap."""

  SCHEMA: ClassVar = {"image": (I32, ()), "colorspace": (U8, ()), "wrap": (U8, (2,))}
  image: Column[np.int32]
  colorspace: Column[np.uint8]
  wrap: Rows[np.uint8, 2]


# Bytes of an image's content digest (`Images.digest`).
DIGEST_BYTES: Final = 16


def texel_digest(levels: Sequence[Array]) -> np.ndarray[[int], np.dtype[np.uint8]]:
  """(DIGEST_BYTES,) content digest of an image's stored levels: their dtypes, shapes and texels."""
  h = hashlib.blake2b(digest_size=DIGEST_BYTES)
  for level in levels:
    contiguous = np.ascontiguousarray(level)
    h.update(f"{contiguous.dtype.str}{contiguous.shape}".encode())
    h.update(contiguous.data)
  return np.frombuffer(h.digest(), np.uint8)


@final
@dataclasses.dataclass
class Images(Table):
  """Decoded images in one texel blob, each with its mip chain (`levels` >= 1).

  Image i, level l is a (max(1, height >> l), max(1, width >> l), channels) array of
  `PIXEL_DTYPES[pixel[i]]`, stored after level l - 1 starting at byte `offset[i]` of `data`.
  Images of equal content share one copy of their bytes (one offset: `SceneBuilder` stores each
  content once). `digest[i]` is the content digest of its stored levels (`texel_digest`;
  `SceneBuilder` sets it as they are decoded and leaves the blob read-only): what an upload keys
  the textures made from it by, without reading its texels. A table assembled otherwise must keep
  it equal to its data.

  `appended`: texel blobs placed after `data` (`append`): offsets past the end of `data` address
  them as if the blobs were one, so appending images to a memory-mapped scene copies none of its
  texels; `arrays()` joins them (writing the scene to a file).
  """

  SCHEMA: ClassVar = {
    "width": (I32, ()),
    "height": (I32, ()),
    "channels": (U8, ()),
    "pixel": (U8, ()),
    "levels": (U8, ()),
    "offset": (I64, ()),
    "digest": (U8, (DIGEST_BYTES,)),
    "data": (U8, ()),
  }
  STRINGS: ClassVar = ("paths",)
  width: Column[np.int32]
  height: Column[np.int32]
  channels: Column[np.uint8]
  pixel: Column[np.uint8]
  levels: Column[np.uint8]
  offset: Column[np.int64]
  digest: Rows[np.uint8, 16]
  data: Column[np.uint8]
  paths_blob: Column[np.uint8]
  paths_offsets: Column[np.int64]
  appended: tuple[Column[np.uint8], ...] = ()

  @property
  def paths(self) -> StringTable:
    """Source of each image (a file, `<derived:...>` or `<array>`)."""
    return self.strings("paths")

  @property
  def blob_size(self) -> int:
    """Bytes of the texel blob (`data`, then the `appended` blobs)."""
    return len(self.data) + sum(len(b) for b in self.appended)

  @override
  def arrays(self) -> dict[str, Array]:
    """Every array, the `appended` blobs joined to `data` (a copy when there are any)."""
    out = super().arrays()
    if self.appended:
      out["data"] = np.concatenate([self.data, *self.appended])
    return out

  def append(self, other: Images) -> Images:
    """This table's images followed by `other`'s, whose blob is appended (ALIGN-aligned) without
    copying this table's; `other`'s image i becomes image `len(self) + i`."""
    end = self.blob_size
    pad = -end % ALIGN
    theirs = other.arrays()  # `other`'s appended blobs joined once
    rows = {name: theirs[name] for name in self.SCHEMA if name != "data"}
    rows["offset"] = rows["offset"] + (end + pad)
    mine = super().arrays()
    columns = {name: np.concatenate([mine[name], rows[name]]) for name in rows}
    table = Images.from_columns(columns, {"paths": self.paths.tolist() + other.paths.tolist()})
    blob = np.concatenate([np.zeros(pad, np.uint8), theirs["data"]])
    return dataclasses.replace(table, data=self.data, appended=(*self.appended, blob))

  def level(self, i: int, level: int = 0) -> np.ndarray[[int, int, int]]:
    """View of image `i` at a stored mip `level` (no copy).

    Raises:
      IndexError: for an image outside the table or a mip outside [0, levels[i]).
    """
    if not 0 <= level < int(self.levels[i]):
      raise IndexError(f"image {i}: mip level {level} is outside [0, {int(self.levels[i])})")
    w, h, c = int(self.width[i]), int(self.height[i]), int(self.channels[i])
    blob, offset = self.data, int(self.offset[i])
    for tail in self.appended:  # the blob holding the image (offsets past `data` as one blob)
      if offset < len(blob):
        break
      offset -= len(blob)
      blob = tail
    # Images start and end ALIGN-aligned in the blob: the rest of it is whole texels.
    texels = blob[offset:].view(PIXEL_DTYPES[PixelType(int(self.pixel[i]))])
    start = sum(max(1, h >> lv) * max(1, w >> lv) * c for lv in range(level))
    hl, wl = max(1, h >> level), max(1, w >> level)
    return texels[start : start + hl * wl * c].reshape(hl, wl, c)
