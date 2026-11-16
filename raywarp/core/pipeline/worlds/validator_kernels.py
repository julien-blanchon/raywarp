"""The GPU side of the per-frame validation (`validator.Validator`): the snapshot copy, the
per-frame validation kernel and the renderer-owned checks, each setting its bit of one int32 flag
word for the host (`Word`, `raise_flag`: `atomic_or`).
"""

from __future__ import annotations

import enum
from typing import Final

import warp as wp

from raywarp._base.types import ShapeType

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


class Word(enum.IntEnum):
  """Flag bits of the validation word: one int32 whose bits above the flags hold the epoch
  (`EPOCH_SHIFT`). The host reads it with one aligned 32-bit load, so a pinned copy landing
  meanwhile is read whole or not at all (a word of int32 slots could be read half old, half new)."""

  MOVED = 1 << 0
  ENV_ID = 1 << 1
  MESH_ID = 1 << 2
  COMPILED_OUT = 1 << 3
  MATERIAL_ID = 1 << 4
  TEXTURE_ID = 1 << 5
  EXPOSURE = 1 << 6
  COLOR = 1 << 7
  DERIVED = 1 << 8
  RAYS = 1 << 9
  CASTERS = 1 << 10
  CAMERA = 1 << 11
  WORLD_IDS = 1 << 12


# The epoch above the 13 flag bits, 18 bits wide (sign bit clear). It wraps: a stale word would
# need 2^18 new epochs between its copy and its read to pass for a current one.
EPOCH_SHIFT: Final = 13
EPOCH_MASK: Final = (1 << 18) - 1
FLAG_MASK: Final = (1 << EPOCH_SHIFT) - 1


@wp.func
def raise_flag(
  # In:
  bit: wp.int32,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Sets flag `bit` (a `Word`) of the validation word `flags[0]`; a set bit is only read (no
  atomic: many threads flag the same error)."""
  if (flags[0] & bit) == 0:
    wp.atomic_or(flags, 0, bit)


@wp.kernel
def snap(
  # In:
  rows_pos: wp.array2d[wp.vec3],
  rows_rot: wp.array2d[wp.mat33],
  transforms: wp.array1d[wp.transformf],
  pose_ids: wp.array1d[int],
  rows: bool,
  npose: int,
  instance_size: wp.array2d[wp.vec3],
  instance_mesh: wp.array2d[int],
  shape_ids: wp.array1d[int],
  nshape: int,
  # Out:
  snap_pos: wp.array1d[wp.vec3],
  snap_rot: wp.array1d[wp.mat33],
  snap_tf: wp.array1d[wp.transformf],
  snap_size: wp.array1d[wp.vec3],
  snap_mesh: wp.array1d[int],
):
  """Copies world 0's shared poses, sizes and meshes into the snapshot buffers."""
  k = wp.tid()
  if k < npose:
    if rows:
      snap_pos[k] = rows_pos[0, pose_ids[k]]
      snap_rot[k] = rows_rot[0, pose_ids[k]]
    else:
      snap_tf[k] = transforms[pose_ids[k]]
  if k < nshape:
    snap_size[k] = instance_size[0, shape_ids[k]]
    snap_mesh[k] = instance_mesh[0, shape_ids[k]]


@wp.func
def check_pose(
  # In:
  rows_pos: wp.array2d[wp.vec3],
  rows_rot: wp.array2d[wp.mat33],
  transforms: wp.array1d[wp.transformf],
  pose_ids: wp.array1d[int],
  rows: bool,
  snap_pos: wp.array1d[wp.vec3],
  snap_rot: wp.array1d[wp.mat33],
  snap_tf: wp.array1d[wp.transformf],
  w: int,
  k: int,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Shared pose k of world w against the snapshot (the transforms, one for every world: in world 0
  only)."""
  i = pose_ids[k]
  if rows:
    if not (rows_pos[w, i] == snap_pos[k] and rows_rot[w, i] == snap_rot[k]):
      raise_flag(wp.int32(Word.MOVED), flags)
  elif w == 0:
    a = transforms[i]
    b = snap_tf[k]
    if not (
      wp.transform_get_translation(a) == wp.transform_get_translation(b)
      and wp.transform_get_rotation(a) == wp.transform_get_rotation(b)
    ):
      raise_flag(wp.int32(Word.MOVED), flags)


@wp.func
def check_shape(
  # In:
  instance_size: wp.array2d[wp.vec3],
  instance_mesh: wp.array2d[int],
  shape_ids: wp.array1d[int],
  snap_size: wp.array1d[wp.vec3],
  snap_mesh: wp.array1d[int],
  w: int,
  k: int,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Shared instance k's size and mesh in world w's table row against the snapshot."""
  gi = shape_ids[k]
  s = instance_size[w % instance_size.shape[0], gi]
  m = instance_mesh[w % instance_mesh.shape[0], gi]
  if not (s == snap_size[k] and m == snap_mesh[k]):
    raise_flag(wp.int32(Word.MOVED), flags)


@wp.func
def check_mesh_id(
  # In:
  instance_type: wp.array1d[int],
  instance_mesh: wp.array2d[int],
  nmesh: int,
  nhfield: int,
  w: int,
  k: int,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Mesh / height-field id of instance k in table row w (each row once) within its table."""
  if w >= instance_mesh.shape[0] or k >= instance_mesh.shape[1]:
    return
  t = instance_type[k]
  m = instance_mesh[w, k]
  if t == ShapeType.MESH or t == ShapeType.SDF:
    if m < 0 or m >= nmesh:
      raise_flag(wp.int32(Word.MESH_ID), flags)
  elif t == ShapeType.HFIELD and (m < 0 or m >= nhfield):
    raise_flag(wp.int32(Word.MESH_ID), flags)


@wp.func
def check_material_id(
  # In:
  instance_material: wp.array2d[int],
  nmaterial: int,
  w: int,
  k: int,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Material id of instance k in table row w (each row once): -1 (none) or below `nmaterial`."""
  if w >= instance_material.shape[0] or k >= instance_material.shape[1]:
    return
  if instance_material[w, k] >= nmaterial:
    raise_flag(wp.int32(Word.MATERIAL_ID), flags)


@wp.func
def check_color(
  # In:
  table: wp.array2d[wp.vec4],
  w: int,
  k: int,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Color k of table row w (each row once): finite (a NaN color makes the world's HDR NaN)."""
  if w >= table.shape[0] or k >= table.shape[1]:
    return
  c = table[w, k]
  if not (wp.isfinite(c[0]) and wp.isfinite(c[1]) and wp.isfinite(c[2]) and wp.isfinite(c[3])):
    raise_flag(wp.int32(Word.COLOR), flags)


@wp.func
def check_texture_id(
  # In:
  material_texture: wp.array3d[int],
  ntexture: int,
  w: int,
  k: int,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Texture ids of material k in table row w (each row once): -1 (none) or below `ntexture`."""
  if w >= material_texture.shape[0] or k >= material_texture.shape[1]:
    return
  for role in range(material_texture.shape[2]):
    if material_texture[w, k, role] >= ntexture:
      raise_flag(wp.int32(Word.TEXTURE_ID), flags)


@wp.kernel
def validate(
  # In:
  rows_pos: wp.array2d[wp.vec3],
  rows_rot: wp.array2d[wp.mat33],
  transforms: wp.array1d[wp.transformf],
  pose_ids: wp.array1d[int],
  rows: bool,
  npose: int,
  snap_pos: wp.array1d[wp.vec3],
  snap_rot: wp.array1d[wp.mat33],
  snap_tf: wp.array1d[wp.transformf],
  instance_size: wp.array2d[wp.vec3],
  instance_mesh: wp.array2d[int],
  shape_ids: wp.array1d[int],
  nshape: int,
  snap_size: wp.array1d[wp.vec3],
  snap_mesh: wp.array1d[int],
  env_id: wp.array1d[int],
  nenv: int,
  instance_type: wp.array1d[int],
  check_mesh: bool,
  nmesh: int,
  nhfield: int,
  instance_material: wp.array2d[int],
  check_material: bool,
  nmaterial: int,
  material_texture: wp.array3d[int],
  check_texture: bool,
  ntexture: int,
  exposure: wp.array1d[float],
  check_exposure: bool,
  instance_color: wp.array2d[wp.vec4],
  material_base_color: wp.array2d[wp.vec4],
  check_colors: bool,
  epoch: wp.array1d[wp.int32],
  # Out:
  flags: wp.array1d[wp.int32],
):
  """One frame's checks (see the module docstring); thread (w, k): world w, entry k."""
  w, k = wp.tid()
  if w == 0 and k == 0:
    wp.atomic_or(flags, 0, epoch[0])
  if k < npose:
    check_pose(rows_pos, rows_rot, transforms, pose_ids, rows, snap_pos, snap_rot, snap_tf, w, k, flags)
  if k < nshape:
    check_shape(instance_size, instance_mesh, shape_ids, snap_size, snap_mesh, w, k, flags)
  if k == 0 and w < env_id.shape[0]:
    e = env_id[w]
    if e < 0 or e >= nenv:
      raise_flag(wp.int32(Word.ENV_ID), flags)
  if check_exposure and k == 0 and w < exposure.shape[0]:
    x = exposure[w]
    if not (x > 0.0 and wp.isfinite(x)):
      raise_flag(wp.int32(Word.EXPOSURE), flags)
  if check_mesh:
    check_mesh_id(instance_type, instance_mesh, nmesh, nhfield, w, k, flags)
  if check_material:
    check_material_id(instance_material, nmaterial, w, k, flags)
  if check_texture:
    check_texture_id(material_texture, ntexture, w, k, flags)
  if check_colors:
    check_color(instance_color, w, k, flags)
    check_color(material_base_color, w, k, flags)


@wp.kernel
def check_range(
  # In:
  values: wp.array2d[float],
  low: float,
  high: float,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Entry (w, m) of a declared per-world table is a number in [low, high] (`low` / `high` the
  largest finite float32: finite; `high` inf: +inf allowed, e.g. an unbounded light.range;
  `derive_kernels.RANGES`; the derive kernels read the scene's value meanwhile)."""
  w, m = wp.tid()
  x = values[w, m]
  if not (x >= low and x <= high):
    raise_flag(wp.int32(Word.DERIVED), flags)


@wp.kernel
def check_casters(
  # In:
  cast_shadow: wp.array2d[bool],
  casters: wp.array1d[bool],
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Light l casts a shadow in row w but the Features were resolved without it as a potential
  caster (`scene.shadow_casters`)."""
  w, l = wp.tid()
  if cast_shadow[w, l] and not casters[l]:
    raise_flag(wp.int32(Word.CASTERS), flags)


@wp.kernel
def check_camera_poses(
  # In:
  pos: wp.array2d[wp.vec3],
  rot: wp.array2d[wp.mat33],
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Sets `Word.CAMERA` when camera c of world w is posed at a non-finite position or rotation (a
  NaN position, a zero / NaN quaternion: every ray of the view would be NaN, a black image)."""
  w, c = wp.tid()
  p = pos[w, c]
  r = rot[w, c]
  finite = wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
  for i in range(3):
    for j in range(3):
      finite = finite and wp.isfinite(r[i, j])
  if not finite:
    raise_flag(wp.static(int(Word.CAMERA)), flags)


@wp.kernel
def check_count(
  # In:
  count: wp.array1d[wp.int32],
  bit: wp.int32,
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Sets flag `bit` when `count[0]` (e.g. the invalid camera rays of the last
  `set_camera_rays`) is not 0."""
  if count[0] != 0:
    raise_flag(bit, flags)


@wp.kernel
def take_count(
  # In:
  bit: wp.int32,
  # In / Out:
  count: wp.array1d[wp.int32],
  # Out:
  flags: wp.array1d[wp.int32],
):
  """Sets flag `bit` when `count[0]` is not 0 and clears it: a count set by any frame since the
  last check (`active.ActiveWorlds.invalid`), reported once whatever `tuning.validate_every`."""
  if count[0] != 0:
    raise_flag(bit, flags)
    count[0] = 0
