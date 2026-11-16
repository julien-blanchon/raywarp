"""Shared scenery: which instances every world shares, decided once per Renderer.

Shared instances are stored and traced once for all worlds; this module decides which they are
(`decide`) and declares what a Renderer randomises per world (`PerWorld`).

An instance is *per world* when
* its binding moves it per world (`Binding.dynamic()`: MuJoCo geoms on a joint chain, on a mocap
  body posed differently per world when the scene was built, or with differing batched geom / body
  poses; Newton shapes on a body or in a world; host scenes: none),
* the Renderer's `PerWorld` declares a geometry field of it (pose, size, mesh),
* a table the binding supplies batched per world differs between worlds for it in size, mesh or
  visibility (alpha 0) (e.g. a batched mjw `geom_size`: an implicit declaration), or
* it is a plane and other instances are per world (cheaper in the worlds' own trees then).

Every other instance is *shared*: one pose for all worlds, stored once in its own BVH / OptiX IAS
traced by every world. Appearance (material, color, lights, cameras, environment) never affects
sharing, with one exception: a per-world material choice that can change the instances' optics
(`worlds/optics.py`: glass or see-through materials in the scene) makes every instance but the
planes per world, since an OptiX instance record culls back faces (or not) for every world it
serves (planes test their two-sided bit per world in their hit test). Otherwise those tables are
indexed per world as they always are. Neither does a declared per-world
visibility (`PerWorld.visible`): the declared instances get a column of the (nworld, nvisible)
"instance.visible" table (`Layout.visible_slot`), tested per world where rays meet them (the
software traversal skips a hidden primitive; OptiX gives a hidden per-world instance mask 0 and
tests a shared one in an any-hit / intersection program).

Pose layouts (`Layout.compact`): bindings that write poses into Renderer buffers
(`Binding.writes_poses`: host scenes, Newton) get compact (nworld, ncolumn) buffers of the per-world
instances only, and the shared instances one pose each in the shared pose table; bindings that alias
their simulator's poses (MuJoCo's aliased d.geom_xpos, every geom a column) read a shared instance's
pose from world 0's row.

The per-frame validation (`core/pipeline/worlds/validator.py`) checks on the device that the shared
poses (and the sizes / meshes of live tables) did not change since the shared trees were built
(`Binding.shared_motion`), and reads its flag word back one frame later from pinned memory (no
synchronisation): shared scenery is static, so a change raises (in one world or all alike) with
the `PerWorld` declaration to add.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np

from raywarp._base.types import ColorRule, ShapeType
from raywarp.core.pipeline.platform.device import host_copy

if TYPE_CHECKING:
  from raywarp._base.dims import AnyHostArray, Ids, Mask
  from raywarp.core.pipeline.gpu_scene.binding import Binding, Slots
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene

# PerWorld geometry fields -> the Renderer fields they let `per_world.columns(field, pattern)`
# request.
GEOMETRY_FIELDS: Final = {
  "pose": ("instance.pos", "instance.rot"),
  "size": ("instance.size",),
  "mesh": ("instance.mesh",),
  "visible": ("instance.visible",),
}
FIELD_GROUP: Final = {field: group for group, fields in GEOMETRY_FIELDS.items() for field in fields}
# Short names of PerWorld.material entries that are instance tables.
_INSTANCE_APPEARANCE: Final = frozenset({"color", "shading"})
# The RGBA color tables (`PerWorld.alpha`: whether their alpha varies per world).
COLOR_FIELDS: Final = ("material.base_color", "instance.color")
# MuJoCo's default geom rgba (`ColorRule.MUJOCO`: an instance showing its material's color).
_MUJOCO_DEFAULT: Final = np.array([0.5, 0.5, 0.5, 1.0], np.float32)
# Instance names an error message lists at most.
_SHOWN: Final = 8
_LISTED: Final = 5
_HINTED: Final = 3

# An instance pattern: an `fnmatch` glob on the instance names, or an instance index.
type Pattern = str | int
# One instance pattern or several (`match`, `Renderer.per_world.columns(field, instances)`).
type Patterns = Pattern | Sequence[Pattern]


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class PerWorld:
  """What a Renderer randomises per world, declared at construction (tables sized once).

  Geometry fields take instance patterns: `fnmatch` globs on `GpuScene.instance_names` (MuJoCo
  instance names, Newton shape labels, USD prim paths) or instance indices. The matched instances
  get per-world poses / sizes / meshes and leave the shared scenery (one tree entry per world each);
  `Renderer.per_world.columns("instance.pos", pattern)` etc. then give their per-world values.
  Visibility (`visible`) keeps shared instances shared: a per-world mask tested where rays meet
  them. Appearance fields take field names and never affect sharing.

  Attributes:
    pose: instances posed per world ("instance.pos" / "instance.rot"; static instances only:
      poses the simulator writes come from it).
    size: instances with per-world sizes ("instance.size").
    mesh: instances with per-world meshes ("instance.mesh").
    visible: instances shown or hidden per world ("instance.visible"; they stay in the shared
      scenery).
    material: per-world material / instance appearance fields: the OpenPBR "base_color" (the
      derived base), "base_diffuse_roughness" (constant EON substrate), "base_metalness",
      "specular_roughness" (under a coupled coat: the final, coat-roughened value),
      "specular_weight", "specular_ior", "coat_weight", "coat_ior", "coat_roughness",
      "emission_luminance" (derived into their device slots every frame,
      `core/pipeline/derive/fields.py`); the extensions "mirror", "texture", "tex_affine"
      (texel maps: tints, texture scale / bias, contrast), "tex_variation" (UV offset / rotation and
      hue per material, `rw.randomize`), "tex_repeat", "alpha_cutoff", "uv_world" (material
      tables); "color", "shading" (instance tables). Qualified names
      ("material.base_color") work too. The color tables ("base_color", "color") are RGB tints
      unless `alpha` is set: their alpha stays the scene's.
    alpha: the declared color tables' alpha varies per world: compiles translucency in (the
      layer loop of `transparency.enabled="auto"`; a declared color table is needed). Without
      it no translucency is compiled in for the declaration (alpha < 1 written per world renders
      opaque; hide instances with `visible`).
    light: per-world light fields ("strength": peak cd / lux, "color": normalized, "pos", "dir",
      ...).
    dynamic_lights: optional light indices whose geometric footprint may change. These lights
      bypass the static visibility grid (evaluated at every point); another light that moves
      makes its world's grid row evaluate every light (the grid's reach check). This hint does
      not allocate or declare any per-world table.
    camera: per-world camera fields ("fovy", "intrinsics", "distortion").
    env: per-world environment fields ("id", "intensity", "yaw"); "exposure" / "ambient", the
      fog's "fog.color" / "fog.range" (`settings.Atmosphere`), the
      look's "grading.*" / "sensor.aberration" (`settings.Grading`, `settings.Sensor`) and the
      sensor's "sensor.noise" / "sensor.vignette" / "sensor.response" (`settings.Sensor`) too.
  """

  pose: Sequence[Pattern] = ()
  size: Sequence[Pattern] = ()
  mesh: Sequence[Pattern] = ()
  visible: Sequence[Pattern] = ()
  material: Sequence[str] = ()
  alpha: bool = False
  light: Sequence[str] = ()
  dynamic_lights: Sequence[int] = ()
  camera: Sequence[str] = ()
  env: Sequence[str] = ()

  def __post_init__(self) -> None:
    """Checks the declaration's types and that `alpha` comes with a color table.

    Raises:
      TypeError: for a pattern that is neither a glob nor an index, a field name that is not a
        string, or a non-bool `alpha` (e.g. "no", which is truthy).
      ValueError: for `alpha` without "base_color" or "color" in `material`.
    """
    if not isinstance(self.alpha, bool):
      raise TypeError(f"PerWorld.alpha must be a bool, got {self.alpha!r}")
    for group, patterns in self.geometry().items():
      for p in patterns:
        _check_pattern(p, f"PerWorld.{group}")
    for group, names in (("material", self.material), ("light", self.light), ("camera", self.camera), ("env", self.env)):
      bad = [n for n in _as_list(names) if not isinstance(n, str)]
      if bad:
        raise TypeError(f"PerWorld.{group} takes field names (str), got {bad!r}")
    if self.alpha and not set(self.fields()) & set(COLOR_FIELDS):
      raise ValueError(
        'PerWorld(alpha=True) varies the alpha of a declared color table: add "base_color" or "color" to material'
      )

  def light_ids(self, nlight: int) -> tuple[int, ...]:
    """The sorted `dynamic_lights` of a scene with `nlight` lights.

    Raises:
      TypeError: for an index that is not an integer (or a bool).
      ValueError: for a repeated index or one outside [0, nlight).
    """
    ids = tuple(self.dynamic_lights)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in ids):
      raise TypeError("PerWorld.dynamic_lights takes integer light indices")
    if len(set(ids)) != len(ids):
      raise ValueError("PerWorld.dynamic_lights contains duplicate light indices")
    if any(i < 0 or i >= nlight for i in ids):
      raise ValueError(f"PerWorld.dynamic_lights indices must be in [0, {nlight})")
    return tuple(sorted(ids))

  def geometry(self) -> dict[str, tuple[Pattern, ...]]:
    """Geometry group -> its instance patterns."""
    patterns = {"pose": self.pose, "size": self.size, "mesh": self.mesh, "visible": self.visible}
    return {g: tuple(_as_list(patterns[g])) for g in GEOMETRY_FIELDS}

  def fields(self) -> list[str]:
    """The qualified appearance fields declared (Renderer.per_world names)."""
    material = _as_list(self.material)
    out = [n if "." in n else f"instance.{n}" if n in _INSTANCE_APPEARANCE else f"material.{n}" for n in material]
    out += [n if "." in n else f"light.{n}" for n in _as_list(self.light)]
    out += [n if "." in n else f"camera.{n}" for n in _as_list(self.camera)]
    out += [n if "." in n or n in ("exposure", "ambient") else f"env.{n}" for n in _as_list(self.env)]
    return out


def _as_list[T: Pattern](value: T | Sequence[T] | None) -> list[T]:
  """`value` as a list: None -> [], one pattern -> [it], a sequence -> its items."""
  if value is None:
    return []
  if isinstance(value, (str, int, np.integer)):
    return [value]
  return list(value)


def _check_pattern(p: object, what: str) -> None:
  """Raises TypeError unless `p` is a glob (str) or an instance index (an int, not a bool: True
  would select instance 1)."""
  if isinstance(p, (bool, np.bool_)) or not isinstance(p, (str, int, np.integer)):
    raise TypeError(f"{what}: an instance pattern is a name glob (str) or an index (int), got {p!r}")


def instance_names(scene: GpuScene) -> list[str]:
  """Names of the scene's instances (`GpuScene.instance_names`).

  Raises:
    ValueError: when the scene has not one name per instance (patterns would match the wrong
      instances).
  """
  names = scene.instance_names
  if len(names) != scene.ninstance:
    raise ValueError(f"GpuScene.instance_names has {len(names)} names for {scene.ninstance} instances")
  return list(names)


def match(names: Sequence[str], patterns: Patterns, what: str = "PerWorld") -> Mask[int]:
  """Bool mask of the instances matched by `patterns` (globs or indices).

  Raises:
    TypeError: for a pattern that is neither a glob nor an index.
    ValueError: for a pattern that matches no instance.
  """
  mask = np.zeros(len(names), bool)
  for p in _as_list(patterns):
    _check_pattern(p, what)
    if isinstance(p, (int, np.integer)):
      i = operator.index(p)
      if not 0 <= i < len(names):
        raise ValueError(f"{what}: instance index {p} out of range (the scene has {len(names)} instances)")
      mask[i] = True
      continue
    hit = np.array([fnmatch.fnmatchcase(n, p) for n in names], bool)
    if not hit.any():
      sample = ", ".join(repr(n) for n in names[:_SHOWN])
      more = ", ..." if len(names) > _SHOWN else ""
      raise ValueError(f"{what}: pattern {p!r} matches no instance (names: {sample}{more})")
    mask |= hit
  return mask


def _rows_differ(values: AnyHostArray) -> Mask[int]:
  """(n,) whether an instance's value differs between the rows of a (rows, n, ...) table."""
  if values.shape[0] <= 1:
    return np.zeros(values.shape[1], bool)
  diff = values != values[:1]
  return diff.reshape(diff.shape[0], diff.shape[1], -1).any(axis=(0, 2))


def _shown_alpha(
  color: np.ndarray[[int, 4], np.dtype[np.float32]],
  material: Ids[int],
  base: np.ndarray[[int, 4], np.dtype[np.float32]],
  *,
  rule: ColorRule,
) -> np.ndarray[[int], np.dtype[np.float32]]:
  """(n,) alpha the instances show in one table row (`kernels.abi.tables.instance_rgba` on the
  host)."""
  alpha = color[:, 3].copy()
  has_mat = (material >= 0) & (material < base.shape[0])
  if rule == ColorRule.MUJOCO:
    has_mat &= np.all(color == _MUJOCO_DEFAULT, axis=1)
    alpha[has_mat] = base[material[has_mat], 3]
  else:
    alpha[has_mat] *= base[material[has_mat], 3]
  return alpha


def implicit_per_world(slots: Slots, n: int, rule: ColorRule) -> Mask[int]:
  """Instances whose size, mesh or visibility (alpha 0) differs between batched table rows
  (`rule`: the scene's `ColorRule`).

  An implicit per-world declaration, e.g. a batched mjw geom_size.
  """
  out = np.zeros(n, bool)
  for a in (slots.get("instance.size"), slots.get("instance.mesh")):
    if a is not None and tuple(a.shape[1:]) == (n,) and a.shape[0] > 1:
      out |= _rows_differ(host_copy(a))
  color, material, base = slots.get("instance.color"), slots.get("instance.material"), slots.get("material.base_color")
  if color is None or material is None or base is None or max(color.shape[0], material.shape[0], base.shape[0]) <= 1:
    return out
  rgba, mat, mrgba = host_copy(color), host_copy(material), host_copy(base)
  rows = max(rgba.shape[0], mat.shape[0], mrgba.shape[0])
  hidden = np.zeros((rows, n), bool)
  for r in range(rows):
    hidden[r] = _shown_alpha(rgba[r % rgba.shape[0]], mat[r % mat.shape[0]], mrgba[r % mrgba.shape[0]], rule=rule) == 0.0
  return out | np.any(hidden != hidden[:1], axis=0)


@final
@dataclasses.dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class Layout:
  """Which instances are shared, and where every instance's pose lives.

  Attributes:
    per_world: (ninstance,) bool: per-world instances (the others are shared).
    dynamic: (ninstance,) bool: per-world instances the binding moves.
    groups: geometry group -> (ninstance,) bool mask of the instances PerWorld declares.
    compact: per-world pose arrays hold only the per-world instances (`columns`).
    columns: (ncolumn,) instance of each per-world pose column (compact; else every instance).
    shared: every shared instance (rendered or not), by shared slot (compact: the shared pose
      table).
    shared_ids: (nrendered_shared,) the rendered shared instances (primitives of the shared trees).
    pose_slot: (ninstance,) per-world column (>= 0) or -1 - shared slot (non-compact: slot =
      instance).
    planes: rendered per-world planes intersected analytically by the software traversal, not
      stored in the worlds' BVH (only with shared scenery: the worlds' trees then bound their
      own small instances, which most rays miss).
    visible_slot: (ninstance,) column of each instance in the (nworld, nvisible) visibility table
      (`PerWorld.visible`), -1: shown in every world.
  """

  per_world: Mask[int]
  dynamic: Mask[int]
  groups: dict[str, Mask[int]]
  compact: bool
  columns: Ids[int]
  shared: Ids[int]
  shared_ids: Ids[int]
  pose_slot: Ids[int]
  planes: Ids[int]
  visible_slot: Ids[int]

  @property
  def ncolumn(self) -> int:
    """Width of the per-world pose arrays."""
    return len(self.columns)

  @property
  def nshared(self) -> int:
    """Number of rendered shared instances."""
    return len(self.shared_ids)

  @property
  def nvisible(self) -> int:
    """Width of the visibility table (instances declared `PerWorld.visible`)."""
    return int((self.visible_slot >= 0).sum())


def decide[S](
  scene: GpuScene[S], binding: Binding[S], *, spec: PerWorld | None, slots: Slots, optics: Mask[int] | None = None
) -> Layout:
  """The Renderer's `Layout` (see the module docstring).

  `slots`: the scene's tables as resolved without a simulator state (scene tables, then the
  binding's model tables). `optics`: (ninstance,) the instances whose optics flags can differ
  per world (`optics.switched_instances`; their non-plane ones per world).
  """
  n = scene.ninstance
  names = instance_names(scene)
  spec = spec if spec is not None else PerWorld()
  dynamic = np.asarray(binding.dynamic(n), bool).reshape(n)
  groups = {g: match(names, p) for g, p in spec.geometry().items()}
  rendered = np.zeros(n, bool)
  rendered[np.asarray(scene.rendered, np.int64)] = True
  per_world, plane = _per_world_mask(scene, dynamic=dynamic, groups=groups, slots=slots, rendered=rendered)
  if optics is not None:
    per_world |= optics & ~plane
  # Nothing is shared unless a rendered instance is; then bindings writing poses into Renderer
  # buffers get compact ones.
  shared_ids = np.flatnonzero(rendered & ~per_world).astype(np.int32)
  compact = binding.writes_poses and len(shared_ids) > 0
  shared = np.flatnonzero(~per_world).astype(np.int32) if len(shared_ids) else shared_ids
  columns, pose_slot = _pose_columns(per_world, shared, compact=compact)
  planes = np.flatnonzero(rendered & per_world & plane).astype(np.int32) if len(shared_ids) else shared_ids
  visible = groups["visible"]
  visible_slot = np.full(n, -1, np.int32)
  visible_slot[visible] = np.arange(int(visible.sum())).astype(np.int32)
  return Layout(
    per_world=per_world,
    dynamic=dynamic & per_world,
    groups=groups,
    compact=compact,
    columns=columns,
    shared=shared,
    shared_ids=shared_ids,
    pose_slot=pose_slot,
    planes=planes,
    visible_slot=visible_slot,
  )


def _per_world_mask(
  scene: GpuScene, *, dynamic: Mask[int], groups: dict[str, Mask[int]], slots: Slots, rendered: Mask[int]
) -> tuple[Mask[int], Mask[int]]:
  """(per-world instances, planes): the moved, declared and implicitly per-world instances, and
  the planes when other rendered instances are per world."""
  n = scene.ninstance
  per_world = dynamic.copy()
  for group, mask in groups.items():
    if group != "visible":  # a visibility mask keeps shared instances shared
      per_world |= mask
  per_world |= implicit_per_world(slots, n, scene.color_rule)
  types = host_copy(scene.instance_type) if n else np.zeros(0, np.int32)
  # Planes stay in the worlds' own trees when those hold other instances (a ground plane there
  # cuts off what is behind it); alone there, they are shared too (one tree traversed, not two).
  plane = types == ShapeType.PLANE
  if np.any(rendered & per_world & ~plane) or scene.deformables.count:
    per_world |= plane
  return per_world, plane


def _pose_columns(per_world: Mask[int], shared: Ids[int], *, compact: bool) -> tuple[Ids[int], Ids[int]]:
  """(columns, pose_slot) of `Layout`: compact, the per-world instances' columns and the shared
  ones' slots; else every instance a column (a shared one's slot -1 - its index)."""
  n = len(per_world)
  pose_slot = np.arange(n).astype(np.int32)
  if compact:
    columns = np.flatnonzero(per_world).astype(np.int32)
    pose_slot[columns] = np.arange(len(columns)).astype(np.int32)
    pose_slot[shared] = -1 - np.arange(len(shared)).astype(np.int32)
  else:
    columns = np.arange(n).astype(np.int32)
    pose_slot[shared] = -1 - shared
  return columns, pose_slot


def check_request(layout: Layout, names: Sequence[str], *, field: str, mask: Mask[int]) -> None:
  """Raises when per-world geometry `field` is requested for instances without it (`mask`):
  shared instances (poses, sizes, meshes), undeclared ones (visibility)."""
  group = FIELD_GROUP[field]
  visible = group == "visible"
  bad = np.flatnonzero(mask & (layout.visible_slot < 0 if visible else ~layout.per_world))
  if not len(bad):
    return
  listed = ", ".join(repr(names[i]) for i in bad[:_LISTED]) + (", ..." if len(bad) > _LISTED else "")
  what = "shown in every world (not declared)" if visible else "shared by every world (static scenery)"
  one = len(bad) == 1
  raise ValueError(
    f"per_world.columns({field!r}): {listed} {'is' if one else 'are'} {what}. "
    f"Declare {'it' if one else 'them'} at construction: Renderer(..., "
    f"per_world=rw.PerWorld({group}=[{', '.join(repr(names[i]) for i in bad[:_HINTED])}]))"
  )
