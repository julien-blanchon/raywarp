"""How kernels read per-world scene tables (colors, materials, textures, visibility) safely: one
shared row or one row per world, with out-of-range ids clamped instead of read out of bounds.

A batched table has 1 or `nworld` rows (`_base/dims.py`, `Table`): world `w` reads row
`min(w, rows - 1)` (`table_row`). These funcs are the single definition of the instance rules the
software traversal, the OptiX build and the shading all apply:

* `instance_rgba`: the color of an instance with a material by its scene's `ColorRule` (the
  adapter's explicit convention): the instance color tints the material's (host scenes), or
  MuJoCo's rule (the default geom rgba shows the material's color, any other replaces it).
* `material_id`: an instance's live material id, clamped below the material count (an
  out-of-range id, e.g. a MuJoCo `m.geom_matid` write, is reported by the validation).
* `texture_id`: a material's live texture id of one role, clamped below the texture count (an
  out-of-range id, e.g. a MuJoCo `m.mat_texid` write, reads the last texture; none without
  textures; reported by the validation).
* `instance_visible`: the per-world visibility table (`Features.visibility`), by visibility slot.
* `table_row` / `batched` / `batched1d`: the generic row read every batched table uses (typed
  through `core/kernels/generic.py`: exact element types for the type checker, one Warp generic at
  runtime).
* `in_table`: an index read from a live table (mesh ids, environment ids) clamped into the table
  it indexes: out-of-range values are reported by the per-frame validation
  (`core/pipeline/worlds/validator.py`) and never read out of bounds meanwhile.
"""

from __future__ import annotations

import warp as wp

from raywarp._base.types import ColorRule
from raywarp.core.kernels.generic import device_generic

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def material_id(
  # Model:
  instance_material: wp.array2d[int],
  nmaterial: int,
  # In:
  worldid: int,
  instance_id: int,
) -> int:
  """Material of instance `instance_id` in world `worldid` (< 0: none), at most `nmaterial` - 1 (a
  live id out of range reads the last material, never out of bounds;
  `core/pipeline/worlds/validator.py` reports it)."""
  return wp.min(batched(instance_material, worldid, instance_id), nmaterial - 1)


@wp.func
def texture_id(
  # Model:
  material_texture: wp.array3d[int],
  ntexture: int,
  # In:
  worldid: int,
  mat_id: int,
  role: int,
) -> int:
  """Texture of material `mat_id`'s `role` in world `worldid` (< 0: none), at most `ntexture` - 1
  (a live id out of range reads the last texture, never out of bounds; -1 without textures;
  `core/pipeline/worlds/validator.py` reports it)."""
  t = material_texture[table_row(worldid, material_texture.shape[0]), mat_id, role]
  return wp.where(t >= ntexture, ntexture - 1, t)


@wp.func
def instance_rgba(
  # Model:
  color_rule: int,
  instance_material: wp.array2d[int],
  instance_color: wp.array2d[wp.vec4],
  material_base_color: wp.array2d[wp.vec4],
  # In:
  worldid: int,
  instance_id: int,
) -> tuple[wp.vec4, int]:
  """(rgba, material id) instance `instance_id` renders with in world `worldid` (`color_rule`: the
  scene's `ColorRule`)."""
  rgba = batched(instance_color, worldid, instance_id)
  mat_id = material_id(instance_material, material_base_color.shape[1], worldid, instance_id)
  return material_rgba(color_rule, material_base_color, worldid, mat_id, rgba), mat_id


@wp.func
def material_value(
  # In:
  table: wp.array2d[float],
  worldid: int,
  mat_id: int,
  default: float,
) -> float:
  """Material `mat_id`'s value in the batched `table` in world `worldid`, `default` without a
  material (< 0)."""
  v = default
  if mat_id >= 0:
    v = batched(table, worldid, mat_id)
  return v


@wp.func
def material_rgba(
  # Model:
  color_rule: int,
  material_base_color: wp.array2d[wp.vec4],
  # In:
  worldid: int,
  mat_id: int,
  rgba: wp.vec4,
) -> wp.vec4:
  """The color a shape of color `rgba` and material `mat_id` (< 0: none) renders with in world
  `worldid` (`color_rule`: the scene's `ColorRule`; MuJoCo geoms and flexes alike)."""
  if mat_id >= 0:
    if color_rule == wp.static(int(ColorRule.TINT)):
      rgba = wp.cw_mul(rgba, batched(material_base_color, worldid, mat_id))
    elif rgba[0] == 0.5 and rgba[1] == 0.5 and rgba[2] == 0.5 and rgba[3] == 1.0:
      # MuJoCo: the default geom (flex) rgba defers to the material's color.
      rgba = batched(material_base_color, worldid, mat_id)
  return finite_rgba(rgba)


@wp.func
def finite_rgba(
  # In:
  rgba: wp.vec4,
) -> wp.vec4:
  """`rgba` with non-finite channels read as black / opaque (the validation reports them on the
  next frame), so no world turns NaN meanwhile."""
  return wp.vec4(
    wp.where(wp.isfinite(rgba[0]), rgba[0], 0.0),
    wp.where(wp.isfinite(rgba[1]), rgba[1], 0.0),
    wp.where(wp.isfinite(rgba[2]), rgba[2], 0.0),
    wp.where(wp.isfinite(rgba[3]), rgba[3], 1.0),
  )


@wp.func
def instance_visible(
  # Model:
  visible_slot: wp.array1d[int],
  # Data in:
  visible_in: wp.array2d[int],
  # In:
  worldid: int,
  gi: int,
) -> bool:
  """Whether instance `gi` is shown in world `worldid`: not declared (slot -1), or its column of
  the per-world visibility table is not 0."""
  slot = visible_slot[gi]
  if slot < 0:
    return True
  return batched(visible_in, worldid, slot) != 0


@wp.func
def instance_shown(
  # Model:
  visible_slot: wp.array1d[int],
  # Data in:
  visible_in: wp.array2d[int],
  # In:
  rgba: wp.vec4,
  use_visible: bool,
  worldid: int,
  gi: int,
) -> bool:
  """Whether instance `gi` of colour `rgba` is drawn in world `worldid`: alpha > 0 and, with the
  per-world visibility table (`use_visible`), not hidden there (`instance_visible`)."""
  show = rgba[3] != 0.0
  if use_visible:
    show = show and instance_visible(visible_slot, visible_in, worldid, gi)
  return show


@wp.func
def in_table(
  # In:
  index: int,
  n: int,
) -> int:
  """`index` clamped to [0, n) (identity for a valid index; see the module docstring)."""
  return wp.clamp(index, 0, n - 1)


@wp.func
def table_row(
  # In:
  worldid: int,
  nrow: int,
) -> int:
  """World `worldid`'s row of a batched table of `nrow` rows, 1 (shared) or one per world
  (`dims.Table`): one min, no branch and no integer remainder. Every scene and batch shares the
  build: the row count is run-time data, never an OptiX bound value (this form costs about as
  little as a bound row count would)."""
  return wp.min(worldid, nrow - 1)


@wp.func
@device_generic
def batched[T](
  # In:
  table: wp.array2d[T],
  worldid: int,
  i: int,
) -> T:
  """Entry `i` of world `worldid`'s row of a (1 or nworld, n) table (row 0 when shared)."""
  return table[table_row(worldid, table.shape[0]), i]


@wp.func
@device_generic
def batched1d[T](
  # In:
  table: wp.array1d[T],
  worldid: int,
) -> T:
  """World `worldid`'s entry of a (1 or nworld,) table (entry 0 when shared)."""
  return table[table_row(worldid, table.shape[0])]


@wp.func
def exposure_at(
  # In:
  exposure: wp.array1d[float],
  worldid: int,
) -> float:
  """World `worldid`'s photometric exposure; an invalid one (not finite and > 0: the validation
  reports it on the next frame) reads as 1, so no world turns NaN / inf meanwhile."""
  e = batched1d(exposure, worldid)
  return wp.where(e > 0.0 and wp.isfinite(e), e, 1.0)


@wp.func
def emission_radiance(
  # In:
  emissive: wp.vec3,
  exposure: float,
  relative: int,
) -> wp.vec3:
  """The radiance of emission `emissive` (before exposure): as is (radiance, `relative` 0) or
  display-relative (emission 1 shows the base color at full brightness: divided by the world's
  `exposure`; `Params.relative_emission`)."""
  if relative != 0:
    return emissive / exposure
  return emissive


@wp.kernel(enable_backward=False)
def tile_rows(
  # In:
  src: wp.array2d[wp.uint8],
  # Out:
  dst: wp.array2d[wp.uint8],
):
  """Byte j of row w of `dst` from row w % K of K-row table `src` (rows as bytes: one kernel for
  every dtype; an adapter's K-row batched table tiled to one row per world, so `table_row` reads
  it as any other)."""
  w, j = wp.tid()
  dst[w, j] = src[w % src.shape[0], j]
