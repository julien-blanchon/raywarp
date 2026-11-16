"""Derived per-world fields: OpenPBR values a world writes whose device slots are derived from
them per frame, as the upload derives them (`core/pipeline/derive/materials.py`).

  specular_weight, specular_ior, coat_weight, coat_ior -> material.reflectance
  coat_weight, coat_roughness                          -> material.specular_roughness
  coat_weight, coat_roughness                          -> instance.shading (the coat bytes)
  coat_ior                                             -> material.coat_eta (Features.coat_layer)
  emission_luminance                                   -> material.emission
  live shininess (MuJoCo's batched m.mat_shininess)    -> material.specular_roughness

Declaring a field (`PerWorld(material=["coat_weight"])`) allocates the user's (nworld, nmat)
float32 table and every slot it feeds (`DERIVED_SLOTS`); the slot's other inputs are the user
tables when declared too, else the scene's values (one row). The user table starts at the scene's
values, or, when the slot is live simulator state (MuJoCo's `m.mat_specular` / `m.mat_emission`),
at the values inverted from it, so a batched or edited model is not reset. Every frame runs one
kernel per derived slot before the frame's kernels (`DerivedTables.derive`: launches recorded on
the first frame, captured in CUDA graphs; nothing runs when nothing is declared;
`warmup.first_frame` compiles them ahead).

Slots derived only with a coat field (`material.specular_roughness`, `instance.shading`) are then
not declarable themselves (`check_declared`); the coat roughens the scene's pre-coat roughness,
and the shading bits other than the coat bytes stay the scene's unless `instance.material` is
also declared. Material selection gathers canonical curve, transmission, vertex-color and coat
bits for the current material. An explicit `instance.shading` declaration wins instead; it
cannot be combined with coat input declarations.

Precision: the reflectance, roughness and coat are computed in float64 from the float32 tables
and rounded once, as the upload computes them from its float32 columns; the emission is one
float32 product. Equal to a fresh upload at float32 values (the roughening's 4th root within 1
ulp; `derive_kernels.py`, no FMA contraction); a float64-authored value is rounded to float32 in
the user table. A non-finite or out-of-range value derives from the scene's (`RANGES`) and the
validation raises (`Word.DERIVED`). The scene's value is the upload's: on MuJoCo, an invalid
`specular_weight` falls back to the model's specular at scene creation, not its live
`m.mat_specular` (whose inverse is the undeclared weight).

Memory: per declared field one (nworld, nmat) float32 table, and per slot it feeds one more
(`instance.shading`: (nworld, ninstance) int32); `coat_weight` alone feeds three slots.
Material selection with nonzero prepared flags also allocates that shading output and shared
canonical input columns, reusing the same slot if coat fields are declared. Material selection
itself introduces no extra render material carrier or per-world mesh storage.
Canonical specular weight shading, when active, aliases the declared weight table (otherwise one
shared row) and reads shared authored model bits. It does not duplicate per-world textures.

Live inputs: a simulator's batched material table the slots derive from without a declaration
(`Binding.live_inputs`, `LIVE_INPUTS`): MuJoCo Warp's per-world `m.mat_shininess` (more than one
row when the Renderer is created) gives the roughness 1 - shininess of the materials that leave it
unset (`SHININESS_SLOT`, as MuJoCo Warp renders them per world), re-read every frame. A declared
`specular_roughness`, or coat fields (whose roughening starts from the scene's roughness), take
the slot instead.

Emission: a material that emits nothing at upload gets its emission color as a tint map with the
declaration (`emission_maps`: the white placeholder texture under the tint emission_color / peak,
as the upload registers it for emitting materials), so a per-world luminance lights it as a fresh
upload does. Scenes without such a declaration keep their tables and kernels. On an emissive panel
turned into a rect light, a per-world emission changes only the visible surface, not the fitted
light or the GI.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.types import ShadingFlag, TexRole
from raywarp.core.pipeline.derive import kernels as derive_kernels
from raywarp.core.pipeline.derive.kernels import RANGES
from raywarp.core.pipeline.derive.materials import thin_share
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.platform.launches import CachedLaunch, recorded
from raywarp.core.pipeline.worlds.validator import Range
from raywarp.scene.model.types import MaterialFlags, Slot

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.pipeline.derive.materials import DerivedMaterials
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.scene.model.tables import Materials
  from raywarp.scene.model.types import Column, Rows


# A static per-material column, on the host.
type _Column = Column[np.float64] | Column[np.float32]
# Material ids a declaration warning lists at most.
_LISTED: Final = 8
# The user fields that feed the coat-only slots, and those slots.
COAT_FIELDS: Final = ("material.coat_weight", "material.coat_ior", "material.coat_roughness")
COAT_SLOTS: Final = ("material.specular_roughness", "instance.shading")
# The user fields of the coat bytes (`derive_kernels.derive_shading`).
_SHADING_FIELDS: Final = ("material.coat_weight", "material.coat_roughness")


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DeriveInputs:
  """The per-material host values the derived fields start from and derive with.

  Attributes:
    shading: (nmat,) packed canonical shading flags and coat bytes (empty: keep instance bits).
    values: user field -> (nmat,) the scene's float32 value (the declared table's first value;
      MuJoCo: `m.mat_specular` / `m.mat_emission` give the live ones).
    ior_scale: (nmat,) 5 |eta - 1| / (eta + 1) at the scene's values (the live reflectance's
      inverse; MuJoCo: 1).
    peak_color: (nmat,) max(`emission_color`) (MuJoCo: 1: its emission scales the base color).
    gltf_coat: (nmat,) the coat is a glTF clearcoat (`MaterialFlags.COAT_GLTF`: not coupled).
    thin: (nmat,) the thin-walled pane share of the specular (`derive.thin_share`; MuJoCo: 0).
    roughenable: (nmat,) a coupled coat roughens the roughness (SPECULAR_ROUGHNESS untextured).
    base_roughness: (nmat,) `specular_roughness` before the coat roughening.
    tint: (nmat, 3) emission_color / peak (zeros without a color).
    diffuse_source: (nmat, 2) `base_weight`, and 1 where neither subsurface nor transmission
      replaces the diffuse lobe (`pipeline.derive.materials.DerivedMaterials`).
    base_lit: (nmat,) a per-world luminance needs an emission tint map (`emission_maps`): none
      registered (nothing emitted at upload) and a peak color > 0 (MuJoCo: none, its emission is
      the base color's by definition).
    from_shininess: (nmat,) the roughness is 1 - shininess (MuJoCo: roughness unset; its scene
      shininess is `values["material.shininess"]`), derived from the live shininess
      (`SHININESS_SLOT`); none elsewhere.
  """

  diffuse_source: Rows[np.float32, 2]
  shading: Column[np.int32]
  values: Mapping[str, Column[np.float32]]
  ior_scale: Column[np.float64]
  peak_color: Column[np.float32]
  gltf_coat: Column[np.bool_]
  thin: Column[np.float32]
  roughenable: Column[np.bool_]
  base_roughness: Column[np.float32]
  tint: Rows[np.float32, 3]
  base_lit: Column[np.bool_]
  from_shininess: Column[np.bool_]

  @staticmethod
  def of(mats: Materials, derived: DerivedMaterials, texid: Column[np.int32] | None) -> DeriveInputs:
    """The inputs of the material table `mats` derived as `derived`, `texid` the materials'
    EMISSIVE texture ids (None: MuJoCo, peak color 1, the emission a scale of the base color)."""
    color = np.asarray(mats.emission_color, np.float32)
    peak = np.ones(len(color), np.float32) if texid is None else color.max(axis=1)
    lit = peak > 0.0
    tint = np.zeros_like(color)
    tint[lit] = (color[lit].astype(np.float64) / peak[lit, None]).astype(np.float32)
    columns = (
      ("specular_weight", mats.specular_weight),
      ("specular_ior", mats.specular_ior),
      ("coat_weight", mats.coat_weight),
      ("coat_ior", mats.coat_ior),
      ("coat_roughness", mats.coat_roughness),
      ("emission_luminance", mats.emission_luminance),
    )
    return DeriveInputs(
      diffuse_source=derived.diffuse_source,
      shading=derived.shading(np.asarray(mats.ext_flags, np.int32)),
      values={f"material.{n}": np.asarray(c, np.float32) for n, c in columns},
      ior_scale=derived.ior_scale,
      peak_color=peak,
      gltf_coat=(np.asarray(mats.ext_flags, np.int64) & int(MaterialFlags.COAT_GLTF)) != 0,
      thin=thin_share(mats),
      roughenable=np.asarray(mats.tex[:, Slot.SPECULAR_ROUGHNESS]) < 0,
      base_roughness=np.asarray(mats.specular_roughness, np.float32),
      tint=tint,
      base_lit=np.zeros(len(color), bool) if texid is None else (texid < 0) & lit,
      from_shininess=np.zeros(len(color), bool),
    )


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DerivedSlot:
  """How a derived device slot is computed.

  Attributes:
    slot: the device slot the kernel writes ((nworld, nmat), or (nworld, ninstance)).
    fields: the user fields the kernel reads first, in its order ((1 or nworld, nmat) tables).
    kernel: In: those tables, their scene columns (the fallback of invalid values), `statics`;
      Out: the slot's table.
    statics: the static per-material columns the kernel reads after them.
    per_instance: the kernel also reads `instance.material` and the scene's slot table (before
      its output) over (nworld, ninstance).
    inverse: In: the live slot, `inverse_column`, the scene's values of `fields[0]`; Out: the
      values of `fields[0]` the live slot comes from, every frame (None: the
      slot is never live state).
    inverse_column: the static per-material column `inverse` reads (with it).
  """

  slot: str
  fields: tuple[str, ...]
  kernel: wp.Kernel
  statics: Callable[[DeriveInputs], tuple[Column[np.bool_] | Column[np.float32] | Column[np.int32], ...]]
  per_instance: bool = False
  inverse: wp.Kernel | None = None
  inverse_column: Callable[[DeriveInputs], _Column] | None = None


# The instance shading update is shared by coat fields and declared material-id selection.
SHADING_SLOT: Final = DerivedSlot(
  slot="instance.shading",
  fields=_SHADING_FIELDS,
  kernel=derive_kernels.derive_shading,
  statics=lambda i: (i.gltf_coat, i.shading),
  per_instance=True,
)

# Every derived slot.
DERIVED: Final = (
  DerivedSlot(
    slot="material.reflectance",
    fields=("material.specular_weight", "material.specular_ior", "material.coat_weight", "material.coat_ior"),
    kernel=derive_kernels.derive_reflectance,
    statics=lambda i: (i.gltf_coat, i.thin),
    inverse=derive_kernels.weight_of,
    inverse_column=lambda i: i.ior_scale,
  ),
  DerivedSlot(
    slot="material.specular_roughness",
    fields=("material.coat_weight", "material.coat_roughness"),
    kernel=derive_kernels.derive_roughness,
    statics=lambda i: (i.gltf_coat, i.roughenable, i.base_roughness),
  ),
  SHADING_SLOT,
  DerivedSlot(
    slot="material.coat_eta",
    fields=("material.coat_ior",),
    kernel=derive_kernels.derive_coat_eta,
    statics=lambda i: (i.gltf_coat,),
  ),
  DerivedSlot(
    slot="material.emission",
    fields=("material.emission_luminance",),
    kernel=derive_kernels.derive_emission,
    statics=lambda i: (i.peak_color,),
    inverse=derive_kernels.luminance_of,
    inverse_column=lambda i: i.peak_color,
  ),
)
# The live inputs (`Binding.live_inputs`): simulator tables, never declared.
LIVE_INPUTS: Final = ("material.shininess",)
# The roughness of the materials that leave it unset, from the live shininess (module docstring).
SHININESS_SLOT: Final = DerivedSlot(
  slot="material.specular_roughness",
  fields=("material.shininess",),
  kernel=derive_kernels.roughness_of_shininess,
  statics=lambda i: (i.from_shininess, i.base_roughness),
)
# The user fields (each feeds the slots that list it).
DERIVED_FIELDS: Final = tuple(dict.fromkeys(f for d in DERIVED for f in d.fields))
# Slots always derived from one user field (never declarable): slot -> that field.
DERIVED_SLOTS: Final = {"material.reflectance": "material.specular_weight", "material.emission": "material.emission_luminance"}


def slots_of(field: str) -> tuple[DerivedSlot, ...]:
  """The derived slots user field `field` feeds."""
  return tuple(d for d in DERIVED if field in d.fields)


def selects_shading[S](scene: GpuScene[S], declared: Iterable[str]) -> bool:
  """Whether a per-world "instance.material" (without a declared "instance.shading") gathers the
  shading bits of each world's material (`derive_kernels.derive_shading`): some prepared material
  has shading bits that matter (vertex colors only when a prepared material has them)."""
  names = set(declared)
  if "instance.material" not in names or "instance.shading" in names:
    return False
  prepared = scene.capabilities.prepared_materials
  colored = prepared.vertex_colors if prepared is not None else scene.capabilities.vertex_colors
  mask = -1 if colored else ~int(ShadingFlag.VERTEX_COLOR)
  return bool(np.any(scene.derive_inputs.shading & mask))


def _batched(table: AnyDeviceArray | None) -> bool:
  """Whether a live (rows, n) input table has more than one row (per-world values)."""
  return table is not None and table.shape[0] > 1


def _shininess_applies[S](scene: GpuScene[S], declared: Iterable[str]) -> bool:
  """Whether the live shininess would give a Renderer of `scene` declaring `declared` roughness
  values: the binding has one, some material's roughness is unset, and neither
  `specular_roughness` nor a coat field is declared."""
  if "material.shininess" not in scene.binding.live_inputs():
    return False
  names = set(declared)
  if "material.specular_roughness" in names or names & set(COAT_FIELDS):
    return False
  return bool(np.any(scene.derive_inputs.from_shininess))


def reads_shininess[S](scene: GpuScene[S], declared: Iterable[str]) -> bool:
  """Whether a Renderer of `scene` declaring `declared` derives the roughness from the live
  shininess (`SHININESS_SLOT`): it applies (`_shininess_applies`) and the binding's table is
  batched when the Renderer is created."""
  return _shininess_applies(scene, declared) and _batched(scene.binding.live_inputs().get("material.shininess"))


def kernels(declared: Iterable[str], *, material_selection: bool = False, shininess: bool = False) -> list[wp.Kernel]:
  """The derive kernels of the `declared` fields, the shading one also with `material_selection`
  (`selects_shading`), the live shininess' with `shininess` (`reads_shininess`)."""
  names = set(declared)
  out = [d.kernel for d in DERIVED if names & set(d.fields) or (d is SHADING_SLOT and material_selection)]
  return [*out, SHININESS_SLOT.kernel] if shininess else out


def check_declared(declared: Iterable[str]) -> None:
  """Raises ValueError when a coat field is declared with a slot it derives (`COAT_SLOTS`)."""
  names = set(declared)
  both = sorted(names & set(COAT_SLOTS))
  if both and names & set(COAT_FIELDS):
    raise ValueError(f"{both} are derived per frame from the declared coat fields: declare one or the other")


def emission_maps(scene: GpuScene) -> tuple[AnyDeviceArray, AnyDeviceArray] | None:
  """The scene's `material.texture` / `material.tex_affine` tables (one row) with the emission
  tint map of every `base_lit` material (the placeholder texture 0: white, under the texel map
  tint), or None: no such material, or no such scene table (live ones, the binding's, warn)."""
  inputs = scene.derive_inputs
  ids = np.flatnonzero(inputs.base_lit)
  if not len(ids):
    return None
  texture, affine = scene.tables.get("material.texture"), scene.tables.get("material.tex_affine")
  live = scene.binding.slots(None)
  if texture is None or affine is None or "material.texture" in live or "material.tex_affine" in live:
    warn_base_lit(inputs)
    return None
  texid, rows = host_copy(texture)[:1].copy(), host_copy(affine)[:1].copy()
  texid[0, ids, TexRole.EMISSIVE] = 0
  rows[0, ids, TexRole.EMISSIVE, 0, :3] = inputs.tint[ids]
  dev = texture.device
  return wp.array(texid, dtype=texture.dtype, device=dev), wp.array(rows, dtype=affine.dtype, device=dev)


def warn_base_lit(inputs: DeriveInputs) -> None:
  """Warns about the materials a per-world luminance lights by their base color (no tint map:
  `emission_maps` found no scene texture table)."""
  ids = np.flatnonzero(inputs.base_lit)
  if len(ids):
    listed = ", ".join(str(int(i)) for i in ids[:_LISTED]) + (", ..." if len(ids) > _LISTED else "")
    warnings.warn(
      f"per-world emission_luminance: {len(ids)} material(s) emit nothing at upload (ids {listed}) and the "
      "scene's texture tables are live: a per-world luminance L lights them as base color x L x max(emission_color)",
      stacklevel=5,  # the Renderer's caller
    )


@final
class DerivedTables:
  """The derived fields a Renderer declared, their slots and derive launches."""

  def __init__(self, scene: GpuScene, declared: frozenset[str]) -> None:
    """None declared yet (`declare`) of the per-world fields `declared` of `scene` (a slot other
    than its tables' is live state); the emission tint maps (`emission_maps`) of a declared
    `emission_luminance`.

    Raises:
      ValueError: for a coat field declared with a slot it derives (`check_declared`).
    """
    check_declared(declared)
    self.select_shading = selects_shading(scene, declared)
    self.shininess = reads_shininess(scene, declared)
    self._live_inputs = scene.binding.live_inputs
    # The live shininess applies but was not batched at creation: warned once if it becomes so.
    self._unread_shininess = not self.shininess and _shininess_applies(scene, declared)
    # An empty canonical column preserves the explicit/initial instance bits in the coat kernel.
    self.inputs = (
      scene.derive_inputs if self.select_shading else dataclasses.replace(scene.derive_inputs, shading=np.empty(0, np.int32))
    )
    self.specular_weight = False
    self.tables = scene.tables
    self.device = scene.device
    lit = "material.emission_luminance" in declared
    self.emission_maps = emission_maps(scene) if lit else None
    self.user: dict[str, AnyDeviceArray] = {}
    # Derived slot -> (its spec, the renderer's slot table, the scene's slot table).
    self.slots: dict[str, tuple[DerivedSlot, AnyDeviceArray, AnyDeviceArray]] = {}
    # Derived slot -> its kernel's scene and static columns on the device (`add_slot`).
    self._static: dict[str, list[AnyDeviceArray]] = {}
    # The undeclared inputs' tables: the scene's values (one row), or those inverted from a live
    # slot (`DerivedSlot.inverse`: the values it holds now).
    self._scene: dict[str, AnyDeviceArray] = {}
    # The recorded launch of each derived slot (re-recorded when an input array changes).
    self._launches: dict[str, CachedLaunch] = {}
    # Undeclared field -> the recorded launch inverting its live slot into `_scene` every frame.
    self._live: dict[str, wp.Launch] = {}

  def add_slot(self, d: DerivedSlot, slot: AnyDeviceArray, source: AnyDeviceArray) -> None:
    """Derives `slot` (the renderer's (nworld, n) table, a copy of `source`, the slot's current
    array) from the next `derive`. A live `source` (simulator state, edited in place) gives its
    first field's values every frame while that field is not declared (`DerivedSlot.inverse`), so
    the model's edits keep working; a declaration starts from them."""
    self.slots[d.slot] = (d, slot, source)
    # The scene columns and static columns, uploaded once (a re-record reuses them).
    columns = [wp.array(self.inputs.values[f], dtype=float, device=self.device) for f in d.fields]
    columns += [wp.array(c, dtype=wp.dtype_from_numpy(c.dtype), device=self.device) for c in d.statics(self.inputs)]
    self._static[d.slot] = columns
    if d.inverse is not None and d.inverse_column is not None and source is not self.tables.get(d.slot):
      first = d.fields[0]
      column = d.inverse_column(self.inputs)
      inputs = [source, wp.array(column, dtype=wp.dtype_from_numpy(column.dtype), device=self.device)]
      inputs.append(wp.array(self.inputs.values[first], dtype=float, device=self.device))
      table = self._scene[first] = wp.empty(source.shape, dtype=float, device=self.device)
      cmd = recorded(
        wp.launch(d.inverse, dim=source.shape, inputs=inputs, outputs=[table], device=self.device, record_cmd=True)
      )
      cmd.launch()  # the values now: a declaration starts from them
      self._live[first] = cmd

  def declare(self, field: str, nworld: int) -> AnyDeviceArray:
    """The user's (nworld, nmat) table of `field`, starting at its `_input` values (the slots it
    feeds are added first)."""
    values = np.broadcast_to(host_copy(self._input(field)), (nworld, len(self.inputs.peak_color)))
    user = self.user[field] = wp.array(np.ascontiguousarray(values), dtype=float, device=self.device)
    return user

  def table(self, field: str) -> AnyDeviceArray:
    """The (1 or nworld, nmat) table of user field `field` a frame reads: the declared one, else
    the scene's values (`_input`)."""
    return self._input(field)

  def _input(self, field: str) -> AnyDeviceArray:
    """The declared user table of `field`, a live input's current simulator table
    (`Binding.live_inputs`, re-read every call), else the undeclared input's table (`_scene`)."""
    if field in LIVE_INPUTS:
      return self._live_inputs()[field]
    if field in self.user:
      return self.user[field]
    if field not in self._scene:
      self._scene[field] = wp.array(self.inputs.values[field][None], dtype=float, device=self.device)
    return self._scene[field]

  def material_inputs(self) -> Mapping[str, AnyDeviceArray]:
    """The derived slots' extra kernel inputs: "material.specular_weight" (the declared table,
    else the scene's one row) while the build shades with it (`specular_weight`), else none."""
    if self.specular_weight:
      return {"material.specular_weight": self._input("material.specular_weight")}
    return {}

  def ranges(self) -> list[Range]:
    """The range of each declared field's user table: what the validation checks
    (`validator.Validator.ranges`)."""
    return [Range(field, table, *RANGES[field]) for field, table in self.user.items()]

  def bind(self, field: str, array: AnyDeviceArray) -> None:
    """Retains a validated replacement user table (the launches reading it re-record)."""
    if field in self.user:
      self.user[field] = array

  def _arguments(self, d: DerivedSlot, source: AnyDeviceArray, material: AnyDeviceArray | None) -> list[object]:
    """The derive kernel's inputs of slot `d` (`material`: the call's `instance.material`).

    Raises:
      ValueError: for a per-instance slot without `material`.
    """
    args: list[object] = [*(self._input(f) for f in d.fields), *self._static[d.slot]]
    if not d.per_instance:
      return args
    if material is None:
      raise ValueError(f"the derived {d.slot} needs the scene's instance.material table")
    return [*args, material, source]

  def derive(self, material: Callable[[], AnyDeviceArray | None]) -> None:
    """Writes every derived slot from its inputs (device work only; `material`: the call's
    `instance.material` table, read by the per-instance slots)."""
    if self._unread_shininess and _batched(self._live_inputs()["material.shininess"]):
      self._unread_shininess = False
      warnings.warn(
        "the simulator's material shininess (MuJoCo m.mat_shininess) was batched per world after the Renderer was "
        "created: its roughness keeps the values at creation; create the Renderer after batching it",
        stacklevel=4,
      )
    for field, cmd in self._live.items():
      if field not in self.user:  # a declared field is the user's table
        cmd.launch()
    for name, (d, slot, source) in self.slots.items():
      if not slot.shape[1]:
        continue
      args = self._arguments(d, source, material() if d.per_instance else None)
      launch = self._launches.setdefault(name, CachedLaunch())
      launch.launch(d.kernel, slot.shape, inputs=args, outputs=[slot], device=self.device)
