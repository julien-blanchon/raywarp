"""MuJoCo materials as canonical OpenPBR materials.

`openpbr_materials` gives the meaning of every MuJoCo material in OpenPBR terms:

  mat_rgba rgb, a          base_color, geometry_opacity (ext.alpha_mode BLEND when a < 1)
  mat_specular s           specular_weight = 4 s^2 at specular_ior 1.5 (F0 = 0.16 s^2, exact): the
                           device reflectance r is s. MuJoCo metals ignore s, and s = 0 keeps F90:
                           ext.flags DIELECTRIC_SPECULAR_LEVEL preserves these native semantics
  mat_shininess            specular_roughness = 1 - shininess, clipped, when roughness is unset
                           (a batched live m.mat_shininess: per world, `shininess_inputs`)
  mat_roughness / metallic specular_roughness / base_metalness (-1, unset: `unset_*`, else
                           1 - shininess / 0, glTF)
  mat_reflectance          ext.mirror (MuJoCo's mirror reflection)
  mat_emission e           emission_luminance = e in MuJoCo's display-relative unit
                           (`Capabilities.physical_emission` False); emission_color = rgb with the
                           base texture (the core scales the textured base color), or the
                           role="emissive" texture (white constant)
  mat_texid[role]          rgb -> BASE_COLOR (wins over rgba); rgba -> BASE_COLOR + GEOMETRY_OPACITY
                           (alpha); opacity -> GEOMETRY_OPACITY (red; with an rgba layer the device
                           multiplies both: reported); orm -> OCCLUSION / SPECULAR_ROUGHNESS /
                           BASE_METALNESS (red / green / blue; wins over the separate layers, read
                           red); normal -> GEOMETRY_NORMAL (the plain unpack 2 t - 1, +Y up: the
                           device's own decode of MuJoCo's top-left UVs); emissive -> EMISSION_COLOR
  mat_texrepeat            each TextureRef's uv_xform scale
  mat_texuniform           ext.flags UV_WORLD
  <numeric raywarp:alpha_cutoff:<mat>> c > 0
                           ext.alpha_mode MASK, ext.alpha_cutoff c. The device tests the opacity
                           layer's texel against c and draws what passes opaque: geometry_opacity
                           1, GEOMETRY_OPACITY from the opacity layer only (the rgba alpha is
                           ignored; without an opacity layer the surface is opaque: both reported)

`m.mat_rgba`, `mat_specular`, `mat_emission`, `mat_texid`, `mat_texrepeat` and `mat_texuniform`
stay aliased (their live meaning is the derived one); metalness, roughness, mirror and the cutoff
are derived from these materials (`build.py`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import mujoco
import numpy as np

from raywarp._base.types import TexRole
from raywarp.scene.model.material import Material, RaywarpExt
from raywarp.scene.model.normal_map import NORMAL_UNPACK
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import (
  CHANNEL_A,
  CHANNEL_B,
  CHANNEL_G,
  CHANNEL_R,
  CHANNEL_RGB,
  AlphaMode,
  Column,
  MaterialFlags,
  MaterialId,
  Slot,
  TextureId,
  TextureRef,
)

if TYPE_CHECKING:
  from raywarp.core.pipeline.derive.fields import DeriveInputs

# MuJoCo's default geom rgba: geoms with a material and this color show the material's color.
DEFAULT_GEOM_RGBA: Final = (0.5, 0.5, 0.5, 1.0)

# Custom numeric naming a material's alpha cutoff (the glTF -> MJCF converter writes it for glTF
# MASK materials: MJCF materials have no cutoff).
ALPHA_CUTOFF_PREFIX: Final = "raywarp:alpha_cutoff:"

# Dielectric-only weight per squared MuJoCo specular level (F0 = 0.16 s^2 = 4 s^2 x 0.04).
WEIGHT_PER_LEVEL2: Final = 4.0

# Data roles bound to one slot, read from the red channel (device `data_texel_map(...)[0]`).
_DATA_ROLES: Final = (
  (TexRole.OCCLUSION, Slot.OCCLUSION),
  (TexRole.ROUGHNESS, Slot.SPECULAR_ROUGHNESS),
  (TexRole.METALLIC, Slot.BASE_METALNESS),
)
_ORM: Final = ((Slot.OCCLUSION, CHANNEL_R), (Slot.SPECULAR_ROUGHNESS, CHANNEL_G), (Slot.BASE_METALNESS, CHANNEL_B))


def _metalness_roughness(
  mjm: mujoco.MjModel, unset_metallic: float | None, unset_roughness: float | None
) -> tuple[Column[np.float32], Column[np.float32]]:
  """(metalness, perceptual roughness) per material, float32 (nmat,), by the unset rules."""
  metallic = np.asarray(mjm.mat_metallic, np.float32)
  roughness = np.asarray(mjm.mat_roughness, np.float32)
  m = np.full(len(metallic), 0.0 if unset_metallic is None else unset_metallic, np.float32)
  shininess = np.asarray(mjm.mat_shininess, np.float64)
  r = np.clip(1.0 - shininess, 0.0, 1.0) if unset_roughness is None else np.full(len(shininess), unset_roughness)
  return np.where(metallic >= 0.0, metallic, m), np.where(roughness >= 0.0, roughness, r.astype(np.float32))


def shininess_inputs(mjm: mujoco.MjModel, inputs: DeriveInputs, *, unset_roughness: float | None) -> DeriveInputs:
  """`inputs` with the materials whose roughness is 1 - shininess (`_metalness_roughness`:
  roughness unset, no `unset_roughness`) and the model's shininess: a batched live
  `m.mat_shininess` gives their roughness per world (`derived_fields.SHININESS_SLOT`)."""
  unset = np.asarray(mjm.mat_roughness, np.float32) < 0.0
  from_shininess = unset if unset_roughness is None else np.zeros(mjm.nmat, bool)
  values = {**inputs.values, "material.shininess": np.asarray(mjm.mat_shininess, np.float32)}
  return dataclasses.replace(inputs, from_shininess=from_shininess, values=values)


def _alpha_cutoffs(mjm: mujoco.MjModel) -> dict[MaterialId, float]:
  """Material id -> alpha-test cutoff in (0, 1] of the `ALPHA_CUTOFF_PREFIX` custom numerics (0:
  no cutoff, left out).

  Raises:
    ValueError: for a cutoff outside [0, 1] or naming no material.
  """
  cutoff: dict[MaterialId, float] = {}
  for i in range(mjm.nnumeric):
    name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_NUMERIC, i) or ""
    if not name.startswith(ALPHA_CUTOFF_PREFIX) or int(mjm.numeric_size[i]) <= 0:
      continue
    mat = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, name[len(ALPHA_CUTOFF_PREFIX) :])
    if mat < 0:
      raise ValueError(f"rw.integrations.mujoco: custom numeric {name!r} names no material of the model")
    value = float(mjm.numeric_data[int(mjm.numeric_adr[i])])
    if not 0.0 <= value <= 1.0:
      raise ValueError(
        f"rw.integrations.mujoco: custom numeric {name!r} = {value}: an alpha cutoff must be in [0, 1] (0: none)"
      )
    if value > 0.0:
      cutoff[MaterialId(mat)] = value
  return cutoff


def _textures(texid: Mapping[TexRole, TextureRef], *, emissive_from_base: bool, mask: bool) -> dict[Slot, TextureRef]:
  """The slot bindings of one material's MuJoCo texture roles (`texid`: bound role -> its ref) by
  the device's rules (module docstring); `emissive_from_base`: the untextured MuJoCo emission;
  `mask`: a cut-out (the opacity layer alone binds GEOMETRY_OPACITY)."""

  def ref(role: TexRole, channel: int = CHANNEL_RGB) -> TextureRef:
    """`role`'s texture read through `channel`."""
    return dataclasses.replace(texid[role], channel=channel)

  base = next((role for role in (TexRole.RGB, TexRole.RGBA) if role in texid), None)
  out: dict[Slot, TextureRef] = {} if base is None else {Slot.BASE_COLOR: ref(base)}
  if TexRole.OPACITY in texid:
    out[Slot.GEOMETRY_OPACITY] = ref(TexRole.OPACITY, CHANNEL_R)
  elif base == TexRole.RGBA and not mask:
    out[Slot.GEOMETRY_OPACITY] = ref(TexRole.RGBA, CHANNEL_A)
  if TexRole.ORM in texid:
    out.update((slot, ref(TexRole.ORM, c)) for slot, c in _ORM)
  else:
    out.update((slot, ref(role, CHANNEL_R)) for role, slot in _DATA_ROLES if role in texid)
  if TexRole.NORMAL in texid:
    scale, bias = NORMAL_UNPACK
    out[Slot.GEOMETRY_NORMAL] = dataclasses.replace(ref(TexRole.NORMAL), scale=scale, bias=bias)
  if TexRole.EMISSIVE in texid:
    out[Slot.EMISSION_COLOR] = ref(TexRole.EMISSIVE)
  elif emissive_from_base and base is not None:
    out[Slot.EMISSION_COLOR] = ref(base)
  return out


def _report_alpha(texid: Mapping[TexRole, TextureRef], report: LoadReport, *, name: str, alpha: float, mask: bool) -> None:
  """Reports where MuJoCo's alpha has no exact OpenPBR form (module docstring)."""
  rgba = TexRole.RGBA in texid and TexRole.RGB not in texid
  if mask and (alpha < 1.0 or rgba):
    report.approx("mujoco.alpha_cutoff(rgba alpha ignored)", name)
  if mask and TexRole.OPACITY not in texid:
    report.approx("mujoco.alpha_cutoff(no opacity layer: opaque)", name)
  if not mask and rgba and TexRole.OPACITY in texid:
    report.approx("mujoco.texture.rgba_alpha(times the opacity layer)", name)


def _material(
  mjm: mujoco.MjModel, i: MaterialId, report: LoadReport, *, resolved: tuple[float, float], cutoff: float | None
) -> Material:
  """Canonical material of MuJoCo material `i` (`resolved`: its metalness / roughness; `cutoff`:
  its alpha cutoff or None)."""
  r, g, b, a = (float(v) for v in mjm.mat_rgba[i])
  s, e = float(mjm.mat_specular[i]), float(mjm.mat_emission[i])
  name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"material{i}"
  rx, ry = (float(v) for v in mjm.mat_texrepeat[i])
  uv = ((rx, 0.0, 0.0), (0.0, ry, 0.0))
  texid = {
    TexRole(role): TextureRef(TextureId(int(t)), uv_xform=uv)
    for role, t in enumerate(mjm.mat_texid[i])
    if t >= 0 and role != int(TexRole.USER)
  }
  mask = cutoff is not None
  _report_alpha(texid, report, name=name, alpha=a, mask=mask)
  textures = _textures(texid, emissive_from_base=e > 0.0, mask=mask)
  flags = int(MaterialFlags.DIELECTRIC_SPECULAR_LEVEL)
  if mjm.mat_texuniform[i]:
    flags |= int(MaterialFlags.UV_WORLD)
  ext = RaywarpExt(mirror=float(mjm.mat_reflectance[i]), flags=flags)
  if cutoff is not None:
    ext = dataclasses.replace(ext, alpha_mode=AlphaMode.MASK, alpha_cutoff=cutoff)
  elif a < 1.0 or Slot.GEOMETRY_OPACITY in textures:
    ext = dataclasses.replace(ext, alpha_mode=AlphaMode.BLEND)
  return Material(
    base_color=(r, g, b),
    base_metalness=resolved[0],
    specular_weight=WEIGHT_PER_LEVEL2 * s * s,
    specular_roughness=resolved[1],
    emission_luminance=e,
    emission_color=(1.0, 1.0, 1.0) if TexRole.EMISSIVE in texid else (r, g, b),
    geometry_opacity=1.0 if mask else a,
    textures=tuple(sorted(textures.items())),
    ext=ext,
    name=name,
  )


def openpbr_materials(
  mjm: mujoco.MjModel, report: LoadReport, *, unset_metallic: float | None = None, unset_roughness: float | None = None
) -> tuple[Material, ...]:
  """The canonical OpenPBR material of every MuJoCo material, in id order (module docstring);
  `report` gets what has no exact OpenPBR form.

  Raises:
    ValueError: for a bad alpha cutoff numeric.
  """
  metalness, roughness = _metalness_roughness(mjm, unset_metallic, unset_roughness)
  cutoffs = _alpha_cutoffs(mjm)
  return tuple(
    _material(mjm, i, report, resolved=(float(metalness[i]), float(roughness[i])), cutoff=cutoffs.get(i))
    for i in map(MaterialId, range(mjm.nmat))
  )
