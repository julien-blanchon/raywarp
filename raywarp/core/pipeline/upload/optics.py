"""The per-instance optics rows the camera rays read at transmissive layers (`kernels/layer.py`
`transmit_step`, `absorb_exit`): run-time data, no kernel key. Each instance has `OPTICS_ROWS`
vec4 rows, its material's.

- `ROW_MEDIUM`: (sigma_r, sigma_g, sigma_b, ior). ior: the refraction index of a solid
  transmissive material (`specular_ior`: a transmission weight of 1, not thin-walled), 0 for a
  thin-walled, partially or non-transmissive one (no bend); the solid-glass feature class
  compiles only for scenes with one. sigma: the Beer-Lambert extinction [1/m] of a *volume*
  material, -ln(transmission_color) / depth: OpenPBR's transmission color is reached at
  `transmission_depth`. Without one, at the median mean chord of the material's instances
  (`pipeline/upload/chords.py`: the path a ray crossing a closed shape travels inside on average):
  the color is the one authored on the typical object, a thicker one of the same glass tints more, a
  thinner one less, and every ray by the length it travels inside.

A volume material: solid, with a `transmission_depth` or every instance of it closed (a ray that
enters leaves through a back face, where the kernels absorb). Its surface tints nothing (its
derived base color's transmitted share is white, `derive._tint`): the color is the volume's.
Another solid (an open instance among its instances: a single quad has no exit) or a thin-walled
material tints at each surface it crosses (OpenPBR's thin-walled tint).

Shadow rays meet both faces of a closed solid and carry no distance: each face passes
exp(-sigma chord / 2) (`compose`), the volume's absorption over the instance's mean chord.

The instance tables (shadow transmittance, optics rows, two-sided / closed-pane bits) are composed
from per-material and per-instance inputs (`OpticsInputs`, `compose`): at upload from the authored
materials, and per frame on the device from each world's (`core/pipeline/worlds/optics.py`, the same
formulas) when a renderer switches materials per world. A material's volume class and depth stay
the authored scene's (a material's look does not change with the objects another world gives it).
"""

from __future__ import annotations

import dataclasses
from typing import Final

import numpy as np

from raywarp._base.types import CLOSED_PANE, MEDIUM_IOR, OPTICS_ROWS, ROW_MEDIUM, TWO_SIDED
from raywarp.core.pipeline.derive.materials import material_values
from raywarp.core.pipeline.upload.chords import closed_instances, mean_chords
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.tables import MaterialExtras, Materials
from raywarp.scene.model.types import Column, InstanceKind, Rows

# The darkest transmission color the extinction is solved for (ln 0 is -inf: a black filter
# absorbs within a few depths instead).
_MIN_TRANSMISSION: Final = 1.0e-4
# The transmission weight from which a solid material is glass (refraction, Beer-Lambert).
_FULL_TRANSMISSION: Final = 0.999


# (n, OPTICS_ROWS, 4) optics rows of n instances.
type OpticsRows = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class MaterialOptics:
  """The materials' optics (module doc) and the instance chords their shadows need.

  Attributes:
    rows: (m, OPTICS_ROWS, 4) the materials' optics rows.
    volume: (m,) volume materials (their transmission color absorbs inside).
    chord: (n,) the mean chord of each closed instance of a volume material (0 else) [m].
  """

  rows: OpticsRows
  volume: Column[np.bool_]
  chord: Column[np.float64]


def _solid(mats: Materials, extras: MaterialExtras) -> Column[np.bool_]:
  """(m,) solid glass: fully transmissive (a partial weight blends a dielectric into another base,
  e.g. MDL tinted plastics: its continuation ray stays straight, as the transmissive layers'
  blend treats it) and not thin-walled."""
  full = material_values(mats, extras, "transmission_weight") >= _FULL_TRANSMISSION
  return full & (np.asarray(mats.geometry_thin_walled) == 0)


def material_optics(scene: Scene) -> MaterialOptics:
  """The optics of `scene`'s materials (module doc)."""
  mats, extras = scene.materials, scene.material_extras
  m = len(mats)
  solid = _solid(mats, extras)
  material = np.asarray(scene.instances.material, np.int64)
  valid = (material >= 0) & (material < m)
  glass = np.zeros(len(material), bool)
  glass[valid] = solid[material[valid]]
  ids: Column[np.int64] = np.flatnonzero(glass).astype(np.int64)
  closed = closed_instances(scene, ids)
  used, opened = np.zeros(m, bool), np.zeros(m, bool)
  used[material[ids]] = True
  opened[material[ids[~closed]]] = True
  depth = material_values(mats, extras, "transmission_depth")
  volume = solid & ((depth > 0.0) | (used & ~opened))
  chord = np.zeros(len(material), np.float64)
  shut: Column[np.int64] = ids[closed & volume[material[ids]]]
  chord[shut] = mean_chords(scene, shut)
  for i in np.flatnonzero(volume & (depth <= 0.0)):
    depth[i] = np.median(chord[shut[material[shut] == i]])
  absorbing = volume & (depth > 0.0)
  color = np.clip(material_values(mats, extras, "transmission_color"), _MIN_TRANSMISSION, 1.0)
  rows = np.zeros((m, OPTICS_ROWS, 4), np.float32)
  rows[:, ROW_MEDIUM, :3] = np.where(absorbing[:, None], -np.log(color) / np.where(absorbing, depth, 1.0)[:, None], 0.0)
  rows[:, ROW_MEDIUM, MEDIUM_IOR] = np.where(solid, material_values(mats, extras, "specular_ior"), 0.0)
  return MaterialOptics(rows=rows, volume=volume, chord=chord)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class OpticsInputs:
  """What the instance optics tables are composed of (`compose`): per material (m) and per
  instance (n).

  Attributes:
    medium: (m, 4) each material's medium row (extinction RGB, refraction index; 0: not solid).
    volume: (m,) volume materials (their instances' chords absorb shadow rays).
    see: (m, 3) the shadow transmittance of a blended material before its surface's Fresnel:
      clip(base color) x (1 - opacity) (`see_through` of `see_alpha` and `base`).
    see_alpha: (m,) the opacity factor of a blended material's see-through part (-1: not
      blended; `materials._see_alpha`): the per-world tables recompose `see` from the world's
      base colors (`optics.compose_optics`).
    transmissive: (m,) transmissive materials (their surface reflects E_F(ior) of a shadow ray).
    base: (m, 4) the derived base color and opacity (an instance color tints them).
    color: (n, 4) the instances' authored colors.
    doubled: (n,) USD doubleSided instances.
    closed: (n,) closed shapes (`chords.closed_instances`): every primitive but planes, the mesh
      instances whose authored material is a volume, or doubled and see-through (`optics_inputs`).
    chord: (n,) each closed instance's mean chord [m] (scenes with a volume material; else 0).
    material: (n,) the instances' authored material ids (-1: none).
  """

  medium: Rows[np.float32, 4]
  volume: Column[np.bool_]
  see: Rows[np.float32, 3]
  see_alpha: Column[np.float32]
  transmissive: Column[np.bool_]
  base: Rows[np.float32, 4]
  color: Rows[np.float32, 4]
  doubled: Column[np.bool_]
  closed: Column[np.bool_]
  chord: Column[np.float32]
  material: Column[np.int32]

  @property
  def switchable(self) -> bool:
    """Whether a per-world material choice can change an instance's tables (a solid, volume or
    see-through material, or a see-through instance color)."""
    return bool(
      np.any(self.medium != 0.0) or np.any(self.see > 0.0) or np.any(self.base[:, 3] < 1.0) or np.any(self.color[:, 3] < 1.0)
    )

  @property
  def solid(self) -> Column[np.bool_]:
    """(n,) the instances of a solid material (a refraction index in the medium row): a per-world
    `specular_ior` sets their rows and, through its sign, their two-sided bit."""
    m = len(self.medium)
    has = (self.material >= 0) & (self.material < m)
    return has & (self.medium[np.where(has, self.material, 0), MEDIUM_IOR] > 0.0) if m else has

  @property
  def ior_dependent(self) -> bool:
    """Whether a per-world `specular_ior` alone (the authored materials kept) changes an instance's
    tables: one of a solid or transmissive material (its shadow rays lose the surface's Fresnel)."""
    m = len(self.medium)
    has = (self.material >= 0) & (self.material < m)
    return bool(np.any(self.solid | (has & self.transmissive[np.where(has, self.material, 0)]))) if m else False


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class InstanceOptics:
  """The instances' optics tables (`compose`).

  Attributes:
    shadow: (n, 3) shadow-ray RGB transmittance (0: opaque).
    rows: (n, OPTICS_ROWS, 4) their materials' optics rows (zeros without).
    twosided: (n,) `TWO_SIDED` / `CLOSED_PANE` bits.
  """

  shadow: Rows[np.float32, 3]
  rows: OpticsRows
  twosided: Column[np.int32]


def optics_inputs(
  scene: Scene,
  optics: MaterialOptics,
  *,
  see_alpha: Column[np.float32],
  transmissive: Column[np.bool_],
  base: Rows[np.float32, 4],
  doubled: Column[np.bool_],
) -> OpticsInputs:
  """The `OpticsInputs` of `scene` (materials' `optics`, `see_alpha`, `transmissive`, derived
  `base`; instances' `doubled`). Closed shapes: every primitive but planes (no test); a mesh
  instance only where its authored material needs it (a volume: its chord; doubled and
  see-through: a pane), each mesh tested once: testing every mesh doubled a 1k-instance
  interior's upload (+0.6 s; a renderer choosing materials per world tests the others:
  `optics.every_closed`). Chords of the closed ones where the scene has a volume material."""
  inst = scene.instances
  n = len(inst.material)
  m = len(optics.volume)
  color = np.asarray(inst.color, np.float32).reshape(-1, 4)
  material = np.asarray(inst.material, np.int64)
  has = (material >= 0) & (material < m)
  mid = np.where(has, material, 0)
  see = see_through(base, see_alpha)
  volume = has & (optics.volume[mid] if m else False)
  shown_through = has & ((np.any(see[mid] > 0.0, axis=1) | (base[mid, 3] < 1.0)) if m else False) | (color[:, 3] < 1.0)
  meshed = np.asarray(inst.kind) == int(InstanceKind.MESH)
  ids: Column[np.int64] = np.flatnonzero(~meshed | volume | (doubled & shown_through)).astype(np.int64)
  closed = np.zeros(n, bool)
  closed[ids] = closed_instances(scene, ids)
  chord = np.zeros(n, np.float32)
  if np.any(optics.volume):
    shut: Column[np.int64] = np.flatnonzero(closed).astype(np.int64)
    chord[shut] = mean_chords(scene, shut)
  return OpticsInputs(
    medium=optics.rows[:, ROW_MEDIUM].astype(np.float32),
    volume=optics.volume,
    see=see,
    see_alpha=np.asarray(see_alpha, np.float32),
    transmissive=transmissive,
    base=base.astype(np.float32),
    color=color,
    doubled=doubled,
    closed=closed,
    chord=chord,
    material=material.astype(np.int32),
  )


def see_through(base: Rows[np.float32, 4], see_alpha: Column[np.float32]) -> Rows[np.float32, 3]:
  """(m, 3) the shadow transmittance of blended materials before their surface's Fresnel: base
  color x (1 - opacity x `see_alpha`), clipped (0 where `see_alpha` < 0: not blended);
  `optics.compose_optics` computes the same per world."""
  b = np.asarray(base, np.float32).reshape(-1, 4)
  a = np.asarray(see_alpha, np.float32)
  part = np.where(a >= 0.0, np.clip(1.0 - b[:, 3] * a, 0.0, 1.0), 0.0).astype(np.float32)
  return (np.clip(b[:, :3], 0.0, 1.0) * part[:, None]).astype(np.float32)


def compose(inputs: OpticsInputs, material: Column[np.int32], ior: Column[np.float32]) -> InstanceOptics:
  """The instance tables of material ids `material` (n,) (-1 or out of range: none) under the
  materials' refraction indices `ior` (m,) (`specular_ior`): `optics.compose_optics` on the
  device (one world)."""
  m = len(inputs.medium)
  has = (material >= 0) & (material < m)
  mid = np.where(has, material, 0)
  medium = np.zeros((len(material), 4), np.float32)
  if m:
    medium[has] = inputs.medium[mid[has]]
    medium[has, MEDIUM_IOR] = np.where(medium[has, MEDIUM_IOR] > 0.0, ior[mid[has]], 0.0)
  volume = has & (inputs.volume[mid] if m else False)
  chord = np.where(volume & inputs.closed, inputs.chord, 0.0).astype(np.float64)
  face = np.exp(-medium[:, :3].astype(np.float64) * (0.5 * chord)[:, None])
  transmissive = has & (inputs.transmissive[mid] if m else False)
  passed = np.where(transmissive, 1.0 - fresnel_average(ior[mid].astype(np.float64) if m else np.ones(len(material))), 1.0)
  own = (inputs.see[mid].astype(np.float64) if m else np.zeros((len(material), 3))) * passed[:, None]
  shown = inputs.color.astype(np.float64).copy()
  if m:
    shown[has] *= inputs.base[mid[has]]
  white = np.all(inputs.color == 1.0, axis=1)
  tinted = np.clip(shown[:, :3], 0.0, 1.0) * np.clip(1.0 - shown[:, 3:4], 0.0, 1.0)
  shadow = (face * np.where((has & white)[:, None], own, tinted)).astype(np.float32)
  solid = medium[:, MEDIUM_IOR] > 0.0
  two = inputs.doubled | solid
  pane = inputs.doubled & np.any(shadow > 0.0, axis=1) & ~transmissive & ~solid & inputs.closed
  bits = np.where(two & ~pane, TWO_SIDED, 0) | np.where(pane, CLOSED_PANE, 0)
  return InstanceOptics(shadow=shadow, rows=medium[:, None, :].repeat(OPTICS_ROWS, axis=1), twosided=bits.astype(np.int32))


def fresnel_average(ior: Column[np.float64]) -> Column[np.float64]:
  """(m,) hemispherical cosine-weighted average dielectric Fresnel E_F(eta) (d'Eon & Irving 2011
  fit; `brdf.fresnel_average`)."""
  eta = np.maximum(ior, 1.0e-3)
  inside = 0.997118 + 0.1014 * eta - 0.965241 * eta**2 - 0.130607 * eta**3
  return np.where(eta >= 1.0, (eta - 1.0) / (4.08567 + 1.00071 * eta), inside)
