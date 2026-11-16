"""How each host material maps UVs (`UvPlans`): one UV set and transform per material, the
base color slot's (else the first bound slot's, else its first extra input's). A transform's
axis scales are the material's `material.tex_repeat`; the rest of a transform and a second UV set
are read per mesh instance (`MeshInstances`); a world-projected material's transform is its
`material.uv_world`."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp._base.types import UV_IDENTITY
from raywarp.core.pipeline.derive.materials import MAPPED_INPUTS
from raywarp.scene.model.material import INPUTS
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import UV_SET_WORLD, Column, Matrices, Rows, Slot

if TYPE_CHECKING:
  from shape_extensions import IntVar

# A UV transform term below this is zero (a scale-only transform).
_UV_TOL: Final = 1e-6
# A UV scale below this is degenerate (treated as 1).
_SCALE_EPS: Final = 1e-12


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class UvPlans:
  """How each material maps UVs (one UV set and transform per material).

  Attributes:
    repeat: (m, 2) the transform's axis scales (`material.tex_repeat`).
    uv_set: (m,) UV set index (0 or 1).
    transformed: (m,) the transform has a rotation or translation (applied per mesh instance).
    xform: (m, 2, 3) that transform divided by the repeat the kernels multiply (identity else).
    world: (m, 2, 3) the transform of a world-projected material (TextureRef.uv_set
      UV_SET_WORLD: `material.uv_world`), zeros for one reading mesh UVs.
  """

  repeat: Rows[np.float32, 2]
  uv_set: Column[np.int32]
  transformed: Column[np.bool_]
  xform: Matrices[np.float32, 2, 3]
  world: Matrices[np.float32, 2, 3]


def uv_plans(scene: Scene, report: LoadReport) -> UvPlans:
  """The materials' `UvPlans`: the base color slot's (else the first bound slot's) set and
  transform (vectorized over the materials: scenes hold thousands)."""
  first, bindings = _uv_bindings(scene, report)
  xf, uv_set = first.xf, first.uv_set
  m = len(uv_set)
  repeat = np.ones((m, 2), np.float32)
  transformed = np.zeros(m, bool)
  xform = np.tile(UV_IDENTITY, (m, 1, 1))
  world = np.zeros((m, 2, 3), np.float32)
  projected = bindings & (uv_set == UV_SET_WORLD)  # projected from the hit's world position (`uv_frame`)
  world[projected] = xf[projected]
  uv_set[projected] = 0
  mapped = bindings & ~projected
  for i in np.flatnonzero(mapped & (uv_set > 1)).tolist():
    report.drop("gpu.material.uv_set>1(set 0 used)", scene.materials.names[i])
  uv_set[mapped & (uv_set > 1)] = 0
  offdiag = np.abs(xf[:, [0, 1, 0, 1], [1, 0, 2, 2]]).max(axis=1)
  scaled = mapped & (offdiag < _UV_TOL)
  repeat[scaled] = xf[scaled][:, [0, 1], [0, 1]]
  general = np.flatnonzero(mapped & ~scaled)
  rx, ry = np.hypot(xf[general, 0, 0], xf[general, 1, 0]), np.hypot(xf[general, 0, 1], xf[general, 1, 1])
  rx, ry = np.where(rx > _SCALE_EPS, rx, 1.0), np.where(ry > _SCALE_EPS, ry, 1.0)
  repeat[general] = np.stack([rx, ry], axis=1)
  transformed[general] = True
  scale = np.zeros((len(general), 2, 2))
  scale[:, 0, 0], scale[:, 1, 1] = 1.0 / rx, 1.0 / ry
  xform[general] = scale @ xf[general]
  return UvPlans(repeat=repeat, uv_set=uv_set, transformed=transformed, xform=xform, world=world)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _FirstBinding:
  """Per material, the binding whose UVs it reads (`_uv_bindings`).

  Attributes:
    xf: (m, 2, 3) its UV transform (float64).
    uv_set: (m,) its UV set index.
  """

  xf: Matrices[np.float64, 2, 3]
  uv_set: Column[np.int32]


def bound_slots[K: IntVar](tex: Rows[np.int32, K], ntex: int) -> Rows[np.bool_, K]:
  """(m, nslot) whether each material slot binds a texture: an id in [0, ntex) (an id past the
  scene's textures binds none)."""
  return (tex >= 0) & (tex < ntex)


def first_bound_slot[K: IntVar](bound: Rows[np.bool_, K]) -> Column[np.intp]:
  """(m,) each material's UV reference slot: its base color slot when bound, else its first bound
  slot (0 when it binds none: test `bound.any(axis=1)`)."""
  base = int(Slot.BASE_COLOR)
  return np.where(bound[:, base], base, np.argmax(bound, axis=1))


def _uv_bindings(scene: Scene, report: LoadReport) -> tuple[_FirstBinding, Column[np.bool_]]:
  """(each material's first UV binding: the base color slot's, else the first bound slot's, else
  its first extra input's; which materials bind any), reporting the materials whose other bindings
  read another transform or UV set (one per material)."""
  mats = scene.materials
  m, ntex = len(mats), len(scene.textures)
  bound = bound_slots(np.asarray(mats.tex), ntex)
  slot_xf = np.asarray(mats.uv_xform, np.float64)
  slot_set = np.asarray(mats.uv_set).astype(np.int32)
  rows = np.arange(m)
  first_slot = first_bound_slot(bound)
  xf, uv_set = slot_xf[rows, first_slot], slot_set[rows, first_slot]
  extra_first: dict[int, str] = {}  # materials reading an extra input's UVs: its name
  xf_differs = (bound & ~np.isclose(slot_xf, xf[:, None]).all(axis=(2, 3))).any(axis=1)
  set_differs = (bound & (slot_set != uv_set[:, None])).any(axis=1)
  bindings = bound.any(axis=1)
  extras = scene.material_extras
  # The extra inputs read through the material's UVs (the detail normal has its own transform).
  shared = {INPUTS.index(n) for n in MAPPED_INPUTS if n != "geometry_detail_normal"}
  extra_rows: dict[int, list[int]] = {}
  for r in range(len(extras.tex_input)):
    if int(extras.tex_input[r]) in shared and 0 <= int(extras.tex[r]) < ntex:
      extra_rows.setdefault(int(extras.tex_material[r]), []).append(r)
  for i, extra in extra_rows.items():
    pairs = [
      (INPUTS[int(extras.tex_input[r])], np.asarray(extras.uv_xform[r], np.float64), int(extras.uv_set[r])) for r in extra
    ]
    if not bindings[i]:
      extra_first[i], xf[i], uv_set[i] = pairs[0]
      bindings[i] = True
    xf_differs[i] |= any(not np.allclose(e_xf, xf[i]) for _, e_xf, _ in pairs)
    set_differs[i] |= any(e_set != uv_set[i] for _, _, e_set in pairs)
  for i in np.flatnonzero(bindings & (xf_differs | set_differs)).tolist():
    name = extra_first.get(i) or Slot(int(first_slot[i])).name.lower()
    if xf_differs[i]:
      report.approx(f"gpu.material.uv_transform(one per material: the {name} slot's)", mats.names[i])
    if set_differs[i]:
      report.approx(f"gpu.material.uv_set(one per material: the {name} slot's)", mats.names[i])
  uv_set[~bindings] = 0
  return _FirstBinding(xf=xf, uv_set=uv_set), bindings
