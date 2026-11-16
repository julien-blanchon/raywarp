"""Per-world texture randomization without texture copies (`rw.randomize`: domain randomization
helpers for the per-world material tables).

Every material has a texture variation (`material.tex_variation`, one vec4: u offset, v offset,
UV rotation about the texture's center, hue rotation of the color roles), applied where its
textures are sampled (`shade/materials/maps.py` `vary_uv` / `color_texel_map`): every role moves
alike, so its maps stay aligned. Declared per world (`PerWorld(material=["tex_variation"])`) the
table is (nworld, nmaterial) vec4: each world reads the same textures with its own offsets, rotation
and hue, at 16 bytes per (world, material). Contrast and per-role tints are the texel maps' scale /
bias rows (`material.tex_affine`, (nworld, nmaterial, NTEXROLE) 2 x 4: 320 bytes per (world,
material)); RGB tints of the base color are cheaper as `PerWorld(material=["base_color"])`.

`jitter` draws a variation table, `contrast` a texel-map table (any texture-role width the
renderer allocates: MuJoCo, OpenPBR or layered):

```python
r = rw.Renderer(scene, nworld=4096, ..., per_world=rw.PerWorld(material=["tex_variation"]))
table = r.per_world["material.tex_variation"]
table.assign(rw.randomize.jitter(table.numpy(), rng, uv_offset=1.0, uv_rotation=np.pi, hue=0.3))
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy as np

from raywarp._base.types import COLOR_ROLES, NLAYERED_TEXROLE, NOPENPBR_TEXROLE, NTEXROLE

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyHostArray
  from raywarp.scene.model.types import MaterialId

# Linear value a contrast change keeps (mid gray, 18 % reflectance).
CONTRAST_PIVOT: Final = 0.18
# Texture-role widths of a texel-map table (`material.tex_affine`): MuJoCo, OpenPBR, layered.
_ROLE_WIDTHS: Final = (NTEXROLE, NOPENPBR_TEXROLE, NLAYERED_TEXROLE)

# A per-world variation table: (nworld, nmaterial, 4) (u offset, v offset, UV rotation, hue).
type VariationTable[NWorld: IntVar, NMaterial: IntVar] = np.ndarray[[NWorld, NMaterial, 4], np.dtype[np.float32]]
# A per-world texel-map table: (nworld, nmaterial, roles, 2, 4) scale / bias rows.
type AffineTable[NWorld: IntVar, NMaterial: IntVar, NRoles: IntVar] = np.ndarray[
  [NWorld, NMaterial, NRoles, 2, 4], np.dtype[np.float32]
]
# Material ids to vary: indices into the table's material axis.
type MaterialIds = Sequence[MaterialId] | np.ndarray[[int], np.dtype[np.integer]]


def _materials(nmat: int, materials: MaterialIds | None) -> np.ndarray[[int], np.dtype[np.intp]]:
  """The materials to vary (default: all).

  Raises:
    ValueError: for a material id outside [0, nmat) (a negative one would wrap around).
  """
  if materials is None:
    return np.arange(nmat)
  mats = np.asarray(materials, np.intp).reshape(-1)
  if len(mats) and (mats.min() < 0 or mats.max() >= nmat):
    raise ValueError(f"materials: ids must be in [0, {nmat}), got {mats.tolist()}")
  return mats


def _check_range(name: str, value: float, high: float = np.inf) -> None:
  """Raises ValueError unless `value` is finite and in [0, high]."""
  if not (np.isfinite(value) and 0.0 <= value <= high):
    raise ValueError(f"{name} must be a finite value in [0, {high}], got {value}")


def _check_table(name: str, table: AnyHostArray, inner: tuple[int, ...]) -> None:
  """Raises ValueError unless `table` is (nworld, nmaterial, *inner)."""
  if table.ndim != 2 + len(inner) or tuple(table.shape[2:]) != inner:
    raise ValueError(f"{name}: a (nworld, nmaterial, {', '.join(map(str, inner))}) table, got shape {table.shape}")


def jitter[NWorld: IntVar, NMaterial: IntVar](  # noqa: PLR0913  # the ranges are keyword-only
  table: VariationTable[NWorld, NMaterial],
  rng: np.random.Generator,
  *,
  uv_offset: float = 0.0,
  uv_rotation: float = 0.0,
  hue: float = 0.0,
  materials: MaterialIds | None = None,
) -> VariationTable[NWorld, NMaterial]:
  """A copy of `table` with random texture variation added per (world, material).

  Args:
    table: (nworld, nmaterial, 4) variations (`Renderer.per_world["material.tex_variation"]`).
    rng: the random generator.
    uv_offset: UV offsets drawn uniformly in [0, uv_offset) (1: any position of a repeating
      texture).
    uv_rotation: UV rotation drawn uniformly in [-uv_rotation, uv_rotation] [rad].
    hue: hue rotation of the color roles drawn uniformly in [-hue, hue] [rad].
    materials: the materials to vary (default: all).

  Raises:
    ValueError: for a table of another shape, a negative or non-finite range, or a material id
      out of range.
  """
  _check_table("jitter", table, (4,))
  for name, value in (("uv_offset", uv_offset), ("uv_rotation", uv_rotation), ("hue", hue)):
    _check_range(name, value)
  out = np.array(table, np.float32, copy=True)
  mats = _materials(out.shape[1], materials)
  shape = (out.shape[0], len(mats))
  low = np.array([0.0, 0.0, -uv_rotation, -hue], np.float32)
  high = np.array([uv_offset, uv_offset, uv_rotation, hue], np.float32)
  out[:, mats] += rng.uniform(0.0, 1.0, (*shape, 4)).astype(np.float32) * (high - low) + low
  return out


def contrast[NWorld: IntVar, NMaterial: IntVar, NRoles: IntVar](
  table: AffineTable[NWorld, NMaterial, NRoles],
  rng: np.random.Generator,
  *,
  amount: float,
  materials: MaterialIds | None = None,
) -> AffineTable[NWorld, NMaterial, NRoles]:
  """A copy of `table` with a random color contrast per (world, material), folded into the color
  roles' scale / bias rows about CONTRAST_PIVOT: factor drawn uniformly in [1 - amount, 1 + amount].

  Args:
    table: (nworld, nmaterial, roles, 2, 4) texel maps (`Renderer.per_world["material.tex_affine"]`;
      roles: 10 MuJoCo, 11 OpenPBR or 15 layered).
    rng: the random generator.
    amount: the contrast range, in [0, 1] (more would draw negative factors: inverted colors).
    materials: the materials to vary (default: all).

  Raises:
    ValueError: for a table of another shape, an amount outside [0, 1], or a material id out of
      range.
  """
  _check_range("amount", amount, 1.0)
  if table.ndim != 5 or table.shape[2] not in _ROLE_WIDTHS:
    raise ValueError(f"contrast: a (nworld, nmaterial, roles, 2, 4) table with roles in {_ROLE_WIDTHS}, got {table.shape}")
  _check_table("contrast", table, (table.shape[2], 2, 4))
  out = np.array(table, np.float32, copy=True)
  mats = _materials(out.shape[1], materials)
  c = rng.uniform(1.0 - amount, 1.0 + amount, (out.shape[0], len(mats)))[..., None, None].astype(np.float32)
  w = np.arange(out.shape[0])[:, None, None]
  m = mats[None, :, None]
  color = np.asarray(COLOR_ROLES)  # hue and contrast apply to the color roles
  scale = out[w, m, color, 0, :3]
  bias = out[w, m, color, 1, :3]
  out[w, m, color, 0, :3] = scale * c
  out[w, m, color, 1, :3] = bias * c + (1.0 - c) * CONTRAST_PIVOT
  return out
