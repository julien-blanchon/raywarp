"""The derived GPU material table: the device values computed from the canonical OpenPBR host
table.

The one place where OpenPBR inputs become device values: the base color (subsurface LOD,
transmission color, base weight), the core opacity, the coat-roughened roughness, the IOR under a
coupled coat, the reflectance r (F0 = 0.16 r^2) from `specular_weight`, the coat layer's values
(the coat's own IOR: its F0 on the device), the emission peak and tint, and the flags. Host
arithmetic is float64 on the stored columns, rounded once to float32.
`core/pipeline/upload/materials.py::material_tables` calls `derive` first and reads only its output;
`core/pipeline/upload/capabilities.py::capabilities` reads it for the Features flags.

A coat is *coupled* (OpenPBR's coat) when `coat_weight` > 0 and the material is not
`MaterialFlags.COAT_GLTF`: only coupled coats change the base IOR ratio and roughen the base.
What the core does not render is reported as `gpu.material.<input>`, one entry per input with
the number of materials (approximations name the approximation in parentheses): the MISSING
inputs of `openpbr.COVERAGE` authored off their default while their lobe is on (`Coverage.when`),
and the approximations below.

The coat formulas follow the OpenPBR specification 1.1.1 and its MaterialX reference graph
(`open_pbr_surface.mtlx`, Apache-2.0, Academy Software Foundation).
"""

from __future__ import annotations

import dataclasses
import operator
from typing import Final, Literal

import numpy as np

from raywarp._base.types import LayerFlag, ShadingFlag, pack_clearcoat
from raywarp.core.pipeline.upload.reporting import report_mask
from raywarp.scene.model.material import INPUTS, Material, OpenPbrExtras
from raywarp.scene.model.openpbr import COVERAGE, Status
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.tables import MaterialExtras, Materials
from raywarp.scene.model.types import AlphaMode, Array, Column, MaterialFlags, MaterialId, Rows, Slot

# The IOR of a glTF-law coat (KHR_materials_clearcoat: F0 0.04).
GLTF_COAT_IOR: Final = 1.5
# Opacity of a fully transmissive material: alpha 0 hides an instance (the visibility convention,
# `sharing._shown_alpha`), so clear glass keeps this much, below one 10-bit display level.
_TRANSMISSIVE_MIN_OPACITY: Final = 2.0**-10
# The reflectance r is at most this (F0 = 0.16 r^2 <= 1).
MAX_REFLECTANCE: Final = 2.5
# Slots carried but not rendered: reported when bound.
_UNRENDERED_SLOTS: Final = (Slot.TRANSMISSION_WEIGHT, Slot.SPECULAR_WEIGHT)
# The extra OpenPBR inputs a layered texture role reads (`materials._register_layers`).
LAYERED_INPUTS: Final = (
  "specular_roughness_anisotropy",
  "fuzz_color",
  "fuzz_roughness",
  "geometry_coat_normal",
  "geometry_detail_normal",
)
# Every extra input with a texture role (rendered: never reported as a dropped texture).
MAPPED_INPUTS: Final = ("base_diffuse_roughness", *LAYERED_INPUTS)
# The records at their defaults (what the reports compare against).
_DEFAULT_MATERIAL: Final = Material()
_DEFAULT_EXTRAS: Final = OpenPbrExtras()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Alpha:
  """A material's core opacity, alpha-test cutoff and whether it is blended (translucent).

  Attributes:
    opacity: core opacity in [0, 1]: `geometry_opacity`, the see-through of transmission
      1 - t (transmission weight t) folded in.
    cutoff: MASK threshold (`ext.alpha_cutoff`), 0 for none.
    translucent: blended (BLEND, or transmissive).
  """

  opacity: float
  cutoff: float
  translucent: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Layers:
  """The layered lobes' device values of every material (m rows; `LayeredMaterials`).

  Attributes:
    anisotropy: (m, 2) OpenPBR anisotropy a and its rotation [rad] (`ext.anisotropy_rotation`).
    fuzz: (m, 2) fuzz weight F and Charlie roughness alpha (`fuzz_roughness`).
    fuzz_color: (m, 3) `fuzz_color`.
    coat_color: (m, 3) `coat_color` (white for a glTF-law coat).
    coat_ior: (m,) `coat_ior` (1.5 for a glTF-law coat: F0 0.04).
    coat_darkening: (m,) `coat_darkening` of an OpenPBR coat (0 for a glTF-law coat).
    wrap: (m,) cloth wrap width `ext.wrap` (fuzz materials only, else 0).
    wrap_tint: (m, 3) the cloth's tint: `subsurface_color` x `subsurface_weight`.
    flags: (m,) LayerFlag bits (the texture bits are set by the texture packing).
  """

  anisotropy: Rows[np.float32, 2]
  fuzz: Rows[np.float32, 2]
  fuzz_color: Rows[np.float32, 3]
  coat_color: Rows[np.float32, 3]
  coat_ior: Column[np.float32]
  coat_darkening: Column[np.float32]
  wrap: Column[np.float32]
  wrap_tint: Rows[np.float32, 3]
  flags: Column[np.int32]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DerivedMaterials:
  """The device values of every material (m rows), before texture packing.

  Attributes:
    base: (m, 4) derived base rgb (subsurface, transmission color, base weight applied) and the
      core opacity (alpha).
    base_diffuse_roughness: (m,) `base_diffuse_roughness`.
    diffuse_source: (m, 2) `base_weight`, and 1 where neither subsurface nor transmission replaces
      the base diffuse lobe (else 0).
    base_metalness: (m,) `base_metalness`.
    specular_roughness: (m,) perceptual roughness, coat-roughened under a coupled (non
      COAT_GLTF) coat when SPECULAR_ROUGHNESS is not textured.
    reflectance: (m,) r = min(sqrt(w / 4) x ior_scale, 2.5), F0 = 0.16 r^2.
    ior_scale: (m,) 5 |eta - 1| / (eta + 1), eta the base IOR (`_ior_scale`): what the per-world
      `specular_weight` derives the reflectance with (`core/pipeline/derive/fields.py`).
    emission: (m,) peak radiance max(emission_color x emission_luminance) [nits].
    emission_tint: (m, 3) emitted color / peak (the EMISSION_COLOR texel map's tint).
    coat: (m,) device coat weight `coat_weight` (its IOR is its own Fresnel law), before byte
      packing.
    coat_roughness: (m,) `coat_roughness`.
    layers: the layered lobes' values (`Layers`).
    curve: (m,) `ext.flags` has SPECULAR_CURVE.
    transmissive: (m,) `transmission_weight` > 0.
    mirror: (m,) `ext.mirror`.
    alpha: per material, opacity / cutoff / translucency.
  """

  base: Rows[np.float32, 4]
  base_diffuse_roughness: Column[np.float32]
  diffuse_source: Rows[np.float32, 2]
  base_metalness: Column[np.float32]
  specular_roughness: Column[np.float32]
  reflectance: Column[np.float32]
  ior_scale: Column[np.float64]
  emission: Column[np.float32]
  emission_tint: Rows[np.float32, 3]
  coat: Column[np.float64]
  coat_roughness: Column[np.float64]
  curve: Column[np.bool_]
  transmissive: Column[np.bool_]
  mirror: Column[np.float32]
  alpha: tuple[Alpha, ...]
  layers: Layers

  def shading(self, flags: Column[np.int32]) -> Column[np.int32]:
    """Canonical per-material shading bits, including independently selectable vertex colors."""
    bits = np.where(self.curve, int(ShadingFlag.SPECULAR_CURVE), 0).astype(np.int64)
    bits |= np.where(self.transmissive, int(ShadingFlag.TRANSMISSIVE), 0)
    bits |= np.where((flags & int(MaterialFlags.DIELECTRIC_SPECULAR_LEVEL)) != 0, int(ShadingFlag.DIELECTRIC_SPECULAR_LEVEL), 0)
    bits |= np.where((flags & int(MaterialFlags.VERTEX_COLOR)) != 0, int(ShadingFlag.VERTEX_COLOR), 0)
    bits |= np.where((flags & int(MaterialFlags.UNLIT)) != 0, int(ShadingFlag.UNLIT), 0)
    bits |= np.where((flags & int(MaterialFlags.DISPLAY_EMISSION)) != 0, int(ShadingFlag.DISPLAY_EMISSION), 0)
    bits |= np.where(self.coat > 0.0, pack_clearcoat(self.coat, self.coat_roughness), 0)
    return bits.astype(np.int32)


def derive(mats: Materials, extras: MaterialExtras, report: LoadReport, *, volume: Column[np.bool_]) -> DerivedMaterials:
  """The device values of `mats` (with their `extras` rows; the module docstring); reports
  what the core does not render (`gpu.material.<input>`, driven by `openpbr.COVERAGE`).

  Args:
    mats: the canonical material table.
    extras: its `OpenPbrExtras` rows (`mats.extras` indexes them).
    report: load report.
    volume: (m,) the materials whose transmission color is a volume's (`pipeline/upload/optics.py`:
      absorbed inside, not at the surface).
  """
  coupled = _coupled_coat(mats)
  alpha = tuple(_alpha(mats, MaterialId(i), report=report) for i in range(len(mats)))
  base = np.empty((len(mats), 4), np.float32)
  base[:, :3] = _base_rgb(mats, extras, volume)
  base[:, 3] = [a.opacity for a in alpha]
  emission, tint = _emission(mats)
  ior_scale = _ior_scale(mats, coupled)
  _report(mats, extras, report, coupled=coupled)
  return DerivedMaterials(
    base=base,
    diffuse_source=np.column_stack(
      (
        np.asarray(mats.base_weight, np.float32),
        (material_values(mats, extras, "subsurface_weight") == 0.0) & (np.asarray(mats.transmission_weight) == 0.0),
      )
    ).astype(np.float32),
    base_diffuse_roughness=material_values(mats, extras, "base_diffuse_roughness").astype(np.float32),
    base_metalness=np.asarray(mats.base_metalness, np.float32),
    specular_roughness=_roughness(mats, coupled & (np.asarray(mats.tex[:, Slot.SPECULAR_ROUGHNESS]) < 0)),
    reflectance=_reflectance(np.asarray(mats.specular_weight, np.float64), ior_scale),
    ior_scale=ior_scale,
    emission=emission,
    emission_tint=tint,
    coat=np.clip(np.asarray(mats.coat_weight, np.float64), 0.0, 1.0),
    coat_roughness=np.asarray(mats.coat_roughness, np.float64),
    curve=_has_flag(mats, MaterialFlags.SPECULAR_CURVE),
    transmissive=np.asarray(mats.transmission_weight) > 0.0,
    mirror=np.asarray(mats.ext_mirror, np.float32),
    alpha=alpha,
    layers=_layers(mats, extras),
  )


def _layers(mats: Materials, extras: MaterialExtras) -> Layers:
  """The layered lobes' values (`Layers`): OpenPBR coats keep their IOR, color and darkening;
  glTF-law coats (COAT_GLTF) are white, IOR 1.5, undarkened. The coat law follows the material
  (not COAT_GLTF), whatever its authored weight: a per-world coat weight on a material authored
  without a coat is OpenPBR's coat, as a fresh upload of that coat is."""
  fuzz_weight = material_values(mats, extras, "fuzz_weight")
  openpbr = ~_has_flag(mats, MaterialFlags.COAT_GLTF)
  flags = np.where(openpbr, int(LayerFlag.COAT_OPENPBR), 0).astype(np.int64)
  flags |= np.where(_has_flag(mats, MaterialFlags.COAT_BASE_NORMAL), int(LayerFlag.COAT_BASE_NORMAL), 0)
  white = np.ones((len(mats), 3))
  cloth = fuzz_weight > 0.0
  tint = material_values(mats, extras, "subsurface_color") * material_values(mats, extras, "subsurface_weight")[:, None]
  return Layers(
    anisotropy=np.column_stack(
      (material_values(mats, extras, "specular_roughness_anisotropy"), np.asarray(mats.ext_anisotropy_rotation, np.float64))
    ).astype(np.float32),
    fuzz=np.column_stack((fuzz_weight, material_values(mats, extras, "fuzz_roughness"))).astype(np.float32),
    fuzz_color=material_values(mats, extras, "fuzz_color").astype(np.float32),
    coat_color=np.where(openpbr[:, None], material_values(mats, extras, "coat_color"), white).astype(np.float32),
    coat_ior=np.where(openpbr, np.asarray(mats.coat_ior, np.float64), GLTF_COAT_IOR).astype(np.float32),
    coat_darkening=np.where(openpbr, material_values(mats, extras, "coat_darkening"), 0.0).astype(np.float32),
    wrap=np.where(cloth, np.asarray(mats.ext_wrap, np.float64), 0.0).astype(np.float32),
    wrap_tint=np.where(cloth[:, None], tint, 0.0).astype(np.float32),
    flags=flags.astype(np.int32),
  )


def _has_flag(mats: Materials, flag: MaterialFlags) -> Column[np.bool_]:
  """(m,) the materials whose `ext.flags` hold any bit of `flag`."""
  return (np.asarray(mats.ext_flags, np.int64) & int(flag)) != 0


def material_values(mats: Materials, extras: MaterialExtras, name: str) -> Array:
  """Every material's value of OpenPBR input `name` as float64, (m,) or (m, 3): its core column,
  or its extras row (the default for a material without extras)."""
  if name in mats.SCHEMA:
    return np.asarray(mats.arrays()[name], np.float64)
  rows = np.asarray(mats.extras, np.intp)
  default = np.asarray(operator.attrgetter(name)(_DEFAULT_EXTRAS), np.float64)
  out = np.broadcast_to(default, (len(rows), *default.shape)).copy()
  out[rows >= 0] = np.asarray(extras.arrays()[name], np.float64)[rows[rows >= 0]]
  return out


def _base_rgb(mats: Materials, extras: MaterialExtras, volume: Column[np.bool_]) -> Rows[np.float64, 3]:
  """(m, 3) the base color: lerped to `subsurface_color` by `subsurface_weight` (the spec's LOD
  rule), then to the transmitted tint (`_tint`) by `transmission_weight`, then times
  `base_weight`. The cases at weight 1 are exact assignments (lerp(a, b, 1) is not always b in
  floating point). A base-color texture multiplies it on the device, its texel map folded to the
  opaque share (`base_shares`, `pipeline/upload/materials.py`)."""
  c = _lerp(np.asarray(mats.base_color, np.float64), *_subsurface(mats, extras))
  c = _lerp(c, _tint(mats, volume), np.asarray(mats.transmission_weight, np.float64))
  w = np.asarray(mats.base_weight, np.float64)[:, None]
  return np.where(w == 1.0, c, w * c)


def _subsurface(mats: Materials, extras: MaterialExtras) -> tuple[Rows[np.float64, 3], Column[np.float64]]:
  """((m, 3) `subsurface_color`, (m,) `subsurface_weight`)."""
  return material_values(mats, extras, "subsurface_color"), material_values(mats, extras, "subsurface_weight")


def _tint(mats: Materials, volume: Column[np.bool_]) -> Rows[np.float64, 3]:
  """(m, 3) the color the surface gives transmitted light. OpenPBR: `transmission_color`, or
  white for a `volume` material (its color is absorbed inside: Beer-Lambert,
  `pipeline/upload/optics.py`); glTF (`MaterialFlags.TRANSMISSION_TINT`): the base color."""
  openpbr = np.where(volume[:, None], 1.0, np.asarray(mats.transmission_color, np.float64))
  return np.where(_has_flag(mats, MaterialFlags.TRANSMISSION_TINT)[:, None], np.asarray(mats.base_color, np.float64), openpbr)


def base_shares(
  mats: Materials, extras: MaterialExtras, volume: Column[np.bool_]
) -> tuple[Rows[np.float64, 3], Rows[np.float64, 3]]:
  """((m, 3) opaque share, (m, 3) transmitted share) of the base color (their sum is `_base_rgb`'s,
  to rounding): base_weight x (1 - t) x the subsurface-blended base color, base_weight x t x the
  tint (`_tint`), t = `transmission_weight`. OpenPBR: a base-color texture multiplies only the
  opaque share (a fully transmissive material's tint is `transmission_color` alone). A
  `MaterialFlags.TRANSMISSION_TINT` material's texture tints both: its opaque share is the
  whole."""
  t = np.asarray(mats.transmission_weight, np.float64)[:, None]
  w = np.asarray(mats.base_weight, np.float64)[:, None]
  opaque = w * (1.0 - t) * _lerp(np.asarray(mats.base_color, np.float64), *_subsurface(mats, extras))
  transmitted = w * t * _tint(mats, volume)
  tinted = _has_flag(mats, MaterialFlags.TRANSMISSION_TINT)[:, None]
  return np.where(tinted, opaque + transmitted, opaque), np.where(tinted, 0.0, transmitted)


def base_color_affine(
  mats: Materials,
  extras: MaterialExtras,
  volume: Column[np.bool_],
  *,
  field: Literal["base_color", "transmission_color"] = "base_color",
) -> tuple[Column[np.float64], Rows[np.float64, 3]]:
  """The canonical upload's affine dependence on a declared live base or transmission color.

  The map preserves base weight and the subsurface/transmission color blends, including zero
  original RGB channels. Static upload retains its exact endpoint/rounding arithmetic. A
  `MaterialFlags.TRANSMISSION_TINT` material's transmitted share follows its base color; a
  `volume` material's is white (its color is in its static extinction, `pipeline/upload/optics.py`):
  a live transmission color does not reach it.
  """
  subsurface = material_values(mats, extras, "subsurface_weight")
  transmission = np.asarray(mats.transmission_weight, np.float64)
  weight = np.asarray(mats.base_weight, np.float64)
  tinted = _has_flag(mats, MaterialFlags.TRANSMISSION_TINT)
  through = np.where(tinted, 0.0, transmission)  # the share the surface tint (`_tint`) colors
  live = np.where(volume, 0.0, through)  # the share that follows transmission_color
  white = (through - live)[:, None]  # a volume's share
  if field == "transmission_color":
    base = np.asarray(mats.base_color, np.float64)
    underlying = _lerp(base, material_values(mats, extras, "subsurface_color"), subsurface)
    bias = underlying * (1.0 - transmission)[:, None] + base * (transmission - through)[:, None] + white
    return weight * live, weight[:, None] * bias
  scale = weight * ((1.0 - subsurface) * (1.0 - transmission) + transmission - through)
  bias = material_values(mats, extras, "subsurface_color") * (subsurface * (1.0 - transmission))[:, None]
  bias += np.asarray(mats.transmission_color, np.float64) * live[:, None] + white
  return scale, weight[:, None] * bias


def _lerp(a: Rows[np.float64, 3], b: Rows[np.float64, 3], t: Column[np.float64]) -> Rows[np.float64, 3]:
  """(m, 3) rows a + (b - a) t: a at t <= 0, b exactly at t == 1."""
  k = t[:, None]
  return np.where(k == 1.0, b, np.where(k > 0.0, a + (b - a) * k, a))


def _roughness(mats: Materials, roughened: Column[np.bool_]) -> Column[np.float32]:
  """(m,) perceptual roughness; `roughened` rows (a coupled coat over an untextured roughness)
  get OpenPBR's coat roughening r + (min(1, 2 c^4 + r^4)^(1/4) - r) x coat_weight."""
  r = np.asarray(mats.specular_roughness, np.float64)
  c = np.asarray(mats.coat_roughness, np.float64)
  under = np.minimum(1.0, 2.0 * c**4 + r**4) ** 0.25
  return np.where(roughened, r + (under - r) * np.asarray(mats.coat_weight, np.float64), r).astype(np.float32)


def _ior_scale(mats: Materials, coupled: Column[np.bool_]) -> Column[np.float64]:
  """(m,) 5 |eta - 1| / (eta + 1), the reflectance of w = 4 at IOR eta (F0 = 0.16 r^2 = w F0(eta)),
  a thin-walled pane's two interfaces folded in (`thin_wall_scale`).
  Under a coupled coat eta is the base IOR of the reference graph (`eta_s`): `specular_ior` mixed
  by `coat_weight` with its ratio to `coat_ior`, inverted below 1 (the spec's Coat TIR rule; F0 is
  the same for a ratio and its inverse)."""
  ior = np.asarray(mats.specular_ior, np.float64)
  ratio = ior / np.asarray(mats.coat_ior, np.float64)
  ratio = np.where(ratio < 1.0, 1.0 / ratio, ratio)
  eta = np.where(coupled, ior + (ratio - ior) * np.asarray(mats.coat_weight, np.float64), ior)
  scale = 5.0 * np.abs(eta - 1.0) / (eta + 1.0)
  return thin_wall_scale(scale, thin_share(mats))


def thin_share(mats: Materials) -> Column[np.float32]:
  """(m,) the share of each material's specular that is a thin-walled transmissive pane:
  `transmission_weight` (clamped to [0, 1]) where `geometry_thin_walled`, else 0."""
  t = np.clip(np.asarray(mats.transmission_weight, np.float64), 0.0, 1.0)
  return np.where(np.asarray(mats.geometry_thin_walled) != 0, t, 0.0).astype(np.float32)


def thin_wall_scale(scale: Column[np.float64], thin: Column[np.float32]) -> Column[np.float64]:
  """(m,) the IOR scale with a thin-walled pane's two interfaces: a pane's total reflectance with
  every internal bounce is 2 F / (1 + F) of one interface's F (OpenPBR's thin-walled dielectric,
  `openpbr_thin_wall_fresnel`; it transmits (1 - F) / (1 + F)). F0 -> F0 (1 + t (1 - F0) /
  (1 + F0)) over the thin share t (`thin_share`), F0 = (scale / 5)^2 = ((eta - 1) / (eta + 1))^2;
  t = 0 leaves the scale bit for bit. `derive_kernels.derive_reflectance` computes the same."""
  q = scale / 5.0
  f0 = q * q
  return scale * np.sqrt(1.0 + np.asarray(thin, np.float64) * (1.0 - f0) / (1.0 + f0))


def _reflectance(weight: Array, ior_scale: Column[np.float64]) -> Column[np.float32]:
  """(m,) the dielectric reflectance r = min(sqrt(w / 4) x ior_scale, 2.5) of `specular_weight` w
  (float64, rounded once; `core/pipeline/derive/fields.py`'s kernel computes the same).

  sqrt(w / 4) is exactly the level s of w = 4 s^2 (MuJoCo, OmniPBR), and with SPECULAR_CURVE
  (eta = 1.5, IOR factor exactly 1) the Unreal curve sqrt(s / 2) of w = 2 s."""
  return np.minimum(np.sqrt(weight / 4.0) * ior_scale, MAX_REFLECTANCE).astype(np.float32)


def _coupled_coat(mats: Materials) -> Column[np.bool_]:
  """(m,) the materials with an OpenPBR coat (coupled: `coat_weight` > 0, not COAT_GLTF; a glTF-law
  coat changes neither the base IOR nor its roughness)."""
  return (np.asarray(mats.coat_weight, np.float64) > 0.0) & ~_has_flag(mats, MaterialFlags.COAT_GLTF)


def _emission(mats: Materials) -> tuple[Column[np.float32], Rows[np.float32, 3]]:
  """((m,) peak radiance [nits], (m, 3) tint): e = `Materials.emission`, peak = max(e), tint =
  e / peak (zeros where nothing is emitted)."""
  e = mats.emission()
  peak = e.max(axis=1) if len(e) else np.zeros(0)
  lit = peak > 0.0
  tint = np.zeros_like(e)
  tint[lit] = e[lit] / peak[lit, None]
  return peak.astype(np.float32), tint.astype(np.float32)


def _alpha(mats: Materials, i: MaterialId, *, report: LoadReport) -> Alpha:
  """The core opacity of material i from its alpha mode, `geometry_opacity` and transmission."""
  mode = int(mats.ext_alpha_mode[i])
  opacity = float(mats.geometry_opacity[i])
  # Transmission: a translucent layer of opacity 1 - t, whatever the alpha mode (glTF / OmniGlass
  # transmissive materials are OPAQUE; solid glass also bends the camera rays: `optics.py`). Its
  # Fresnel reflection takes the rest (`kernels/layer.composite_transmissive`): clear glass passes
  # 1 - E_spec.
  transmission = float(mats.transmission_weight[i])
  see_through = 1.0
  if transmission > 0.0:
    see_through = max(1.0 - transmission, _TRANSMISSIVE_MIN_OPACITY)
    report.approx("gpu.material.transmission(alpha)", mats.names[i])
  if mode == AlphaMode.OPAQUE:
    return Alpha(opacity=see_through, cutoff=0.0, translucent=transmission > 0.0)
  if mode == AlphaMode.MASK:
    # Alpha test of factor x texel >= cutoff (cutoff > 0: `checks.material`): on the texel
    # against cutoff / factor (an opaque surface where it passes); without an opacity map the
    # whole surface passes or vanishes.
    c = float(mats.ext_alpha_cutoff[i])
    if int(mats.tex[i, Slot.GEOMETRY_OPACITY]) >= 0 and opacity > 0.0:
      if transmission > 0.0:
        report.drop("gpu.material.transmission(cut-out surfaces are opaque)", mats.names[i])
      return Alpha(opacity=1.0, cutoff=c / opacity, translucent=False)
    return Alpha(opacity=see_through if opacity >= c else 0.0, cutoff=0.0, translucent=transmission > 0.0)
  return Alpha(opacity=opacity * see_through, cutoff=0.0, translucent=True)


def _flag(mats: Materials, report: LoadReport, key: str, mask: Column[np.bool_], *, approx: bool = False) -> None:
  """Reports `key` once for the materials of `mask` (the first one as the example)."""
  report_mask(report.approx if approx else report.drop, key, mask, example=lambda i: mats.names[i])


def _report(mats: Materials, extras: MaterialExtras, report: LoadReport, *, coupled: Column[np.bool_]) -> None:
  """Reports what the core does not render or approximates (`gpu.material.<input>`)."""
  for slot in _UNRENDERED_SLOTS:
    _flag(mats, report, f"gpu.material.texture.{slot.name.lower()}", np.asarray(mats.tex[:, slot]) >= 0)
  _report_specular_weight(mats, report)
  key = "gpu.material.subsurface_weight(subsurface_color as the diffuse color)"
  _flag(mats, report, key, material_values(mats, extras, "subsurface_weight") > 0.0, approx=True)
  _report_coat(mats, report, coupled=coupled)
  diffuse_rough = material_values(mats, extras, "base_diffuse_roughness") > 0.0
  mixed = (material_values(mats, extras, "subsurface_weight") > 0.0) | (np.asarray(mats.transmission_weight) > 0.0)
  _flag(mats, report, "gpu.material.base_diffuse_roughness(subsurface/transmission fallback)", diffuse_rough & mixed)
  _flag(
    mats, report, "gpu.material.base_diffuse_roughness(existing specular/coat coupling)", diffuse_rough & ~mixed, approx=True
  )
  _report_missing_inputs(mats, extras, report)
  _report_unmapped_textures(mats, extras, report)


def _report_specular_weight(mats: Materials, report: LoadReport) -> None:
  """Reports the approximations of a `specular_weight` other than 0 or 1 under neither the
  SPECULAR_CURVE nor the DIELECTRIC_SPECULAR_LEVEL law."""
  w = np.asarray(mats.specular_weight, np.float64)
  metal = np.asarray(mats.base_metalness)
  mapped_metal = np.asarray(mats.tex[:, Slot.BASE_METALNESS]) >= 0
  laws = MaterialFlags.SPECULAR_CURVE | MaterialFlags.DIELECTRIC_SPECULAR_LEVEL
  weighted = (w != 1.0) & (w > 0.0) & ~_has_flag(mats, laws)
  _flag(
    mats,
    report,
    "gpu.material.specular_weight(Schlick dielectric Fresnel)",
    weighted & ((metal < 1.0) | mapped_metal),
    approx=True,
  )
  _flag(
    mats,
    report,
    "gpu.material.specular_weight(mixed-metal compensation)",
    weighted & (((metal > 0.0) & (metal < 1.0)) | mapped_metal),
    approx=True,
  )


def _report_missing_inputs(mats: Materials, extras: MaterialExtras, report: LoadReport) -> None:
  """Reports the MISSING inputs of `openpbr.COVERAGE` authored off their default while their lobe
  is on (`Coverage.when`)."""
  for name, cov in COVERAGE.items():
    if cov.status != Status.MISSING or (name not in mats.SCHEMA and name not in extras.SCHEMA):
      continue
    default = operator.attrgetter(name)(_DEFAULT_EXTRAS if name in extras.SCHEMA else _DEFAULT_MATERIAL)
    stored = material_values(mats, extras, name).astype(np.float32)
    off = np.any(np.atleast_2d(stored.T).T != np.asarray(default, np.float32).reshape(-1), axis=1)
    for gate in cov.when:
      off &= material_values(mats, extras, gate) > 0.0
    _flag(mats, report, f"gpu.material.{name}", off)


def _report_unmapped_textures(mats: Materials, extras: MaterialExtras, report: LoadReport) -> None:
  """Reports the textures bound to extra inputs without a texture role (not `MAPPED_INPUTS`)."""
  inputs, first, counts = np.unique(np.asarray(extras.tex_input, np.int64), return_index=True, return_counts=True)
  for k, row, n in zip(inputs.tolist(), first.tolist(), counts.tolist(), strict=True):
    if INPUTS[k] in MAPPED_INPUTS:
      continue
    report.drop(f"gpu.material.texture.{INPUTS[k]}", mats.names[int(extras.tex_material[row])], n)


def _report_coat(mats: Materials, report: LoadReport, *, coupled: Column[np.bool_]) -> None:
  """Reports what the coupled (OpenPBR) coats approximate or drop."""
  textured = np.asarray(mats.tex[:, Slot.SPECULAR_ROUGHNESS]) >= 0
  _flag(mats, report, "gpu.material.coat_roughening(textured roughness)", coupled & textured, approx=True)
  emitting = np.asarray(mats.emission_luminance) > 0.0
  _flag(mats, report, "gpu.material.emission_color(coat absorption only, no Fresnel)", coupled & emitting, approx=True)
