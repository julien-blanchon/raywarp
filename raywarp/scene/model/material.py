"""The canonical host material: an OpenPBR Surface 1.1.1 surface.

`Material` holds 16 OpenPBR inputs (the ones the core renders or derives from, and the carried
`specular_color` / `geometry_thin_walled`), with the spec's names and
defaults; `OpenPbrExtras` the other OpenPBR inputs (sparse: None when all are at their default);
`RaywarpExt` the documented raywarp extensions (glTF alpha modes, occlusion, the MuJoCo mirror,
flags, anisotropy rotation, cloth wrap). Every adapter writes OpenPBR values; the device table is
derived from them at upload (`core.pipeline.derive.materials`), the one place where conversions
happen. Colors are linear Rec.709.

This is `rw.Material`: the builder, the host tables (`tables.Materials`,
`MaterialExtras`), the `.rayw` format and the upload hold it; every adapter writes it.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import Final

from raywarp.scene.model.types import AlphaMode, Slot, TextureRef

type Color = tuple[float, float, float]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Displacement:
  """Scalar object-space displacement prepared into shared triangle geometry at build.

  `amount` is a constant distance or a texture binding whose scale/bias give distance.
  There is no implicit midpoint or unit conversion. Runtime material swaps affect shading,
  not the prepared geometry. Vector displacement and per-world height updates are not yet
  supported. No device material field is allocated for this preparation-only record.
  """

  amount: float | TextureRef


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RaywarpExt:
  """raywarp extensions of the OpenPBR surface (not OpenPBR inputs).

  Attributes:
    alpha_mode: AlphaMode: OPAQUE / BLEND / MASK (glTF). `geometry_opacity` < 1 (or a
      GEOMETRY_OPACITY texture) requires BLEND or MASK.
    alpha_cutoff: MASK threshold on the opacity, in (0, 1].
    occlusion_strength: blend of the OCCLUSION map (glTF AO), in [0, 1].
    mirror: MuJoCo's mirror reflectance `mat_reflectance` in [0, 1] (one extra ray,
      `Features.reflection`); host scenes: 0.
    flags: MaterialFlags: TWO_SIDED, UNLIT, UV_WORLD, SPECULAR_CURVE, VERTEX_COLOR, COAT_GLTF
      (a coat that leaves the base unchanged: glTF / UsdPreviewSurface / OmniPBR / OmniSurface),
      DIELECTRIC_SPECULAR_LEVEL (MuJoCo's specular level: dielectric F0 only),
      COAT_BASE_NORMAL (the coat shares the base's mapped normal: UsdPreviewSurface),
      TRANSMISSION_TINT (the base color tints transmitted light: glTF).
    anisotropy_rotation: rotation [rad] of the specular anisotropy's stretch direction about the
      shading normal, counter-clockwise from the tangent dp/du (MaterialX standard_surface
      `specular_rotation` x 2 pi, glTF `anisotropyRotation`); OpenPBR itself has no rotation
      (its `geometry_tangent` is the stretch direction).
    wrap: wrapped-diffuse width w of a cloth material (Filament's cloth model, `Fd_Wrap`), in
      [0, 1]: the diffuse lobe reaches acos(-w) past the terminator, tinted by the subsurface
      color; 0 (default): Lambert. Read with a fuzz lobe (`fuzz_weight` > 0) only.
  """

  alpha_mode: AlphaMode = AlphaMode.OPAQUE
  alpha_cutoff: float = 0.5
  occlusion_strength: float = 1.0
  mirror: float = 0.0
  flags: int = 0
  anisotropy_rotation: float = 0.0
  wrap: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class OpenPbrExtras:
  """Additional OpenPBR inputs at the spec's defaults, stored only when authored.

  Rendering coverage is recorded in `openpbr.COVERAGE`; unsupported authored inputs are reported
  under `gpu.material.<input>`. Diffuse roughness supports scalar values and prepared textures;
  subsurface currently uses the documented diffuse-color fallback.

  Attributes:
    base_diffuse_roughness: EON diffuse roughness, [0, 1]; multiplies its bound texture.
      Use 1 for the texture alone (OpenPBR connected inputs prepare this unit factor).
    specular_roughness_anisotropy: [0, 1].
    transmission_depth: [m], >= 0.
    transmission_scatter: interior scattering color, >= 0.
    transmission_scatter_anisotropy: [-1, 1].
    transmission_dispersion_scale: >= 0.
    transmission_dispersion_abbe_number: > 0.
    subsurface_weight: [0, 1].
    subsurface_color: [0, 1].
    subsurface_radius: mean free path scale [m], >= 0.
    subsurface_radius_scale: per-channel radius multiplier, >= 0.
    subsurface_scatter_anisotropy: [-1, 1].
    fuzz_weight: [0, 1].
    fuzz_color: [0, 1].
    fuzz_roughness: [0, 1].
    coat_color: [0, 1].
    coat_darkening: [0, 1].
    coat_roughness_anisotropy: [0, 1].
    thin_film_weight: [0, 1].
    thin_film_thickness: [um], >= 0.
    thin_film_ior: > 0.
    textures: textures bound to other OpenPBR inputs, keyed by input name (`openpbr.INPUTS`, not
      one a `Slot` drives: "geometry_coat_normal", "geometry_tangent", "fuzz_color" ...), sorted by
      name, one per input. Diffuse roughness is rendered as its scalar factor times
      clamp(scale * selected texel + bias, 0, 1); other extra textures remain reported omissions.
  """

  base_diffuse_roughness: float = 0.0
  specular_roughness_anisotropy: float = 0.0
  transmission_depth: float = 0.0
  transmission_scatter: Color = (0.0, 0.0, 0.0)
  transmission_scatter_anisotropy: float = 0.0
  transmission_dispersion_scale: float = 0.0
  transmission_dispersion_abbe_number: float = 20.0
  subsurface_weight: float = 0.0
  subsurface_color: Color = (0.8, 0.8, 0.8)
  subsurface_radius: float = 1.0
  subsurface_radius_scale: Color = (1.0, 0.5, 0.25)
  subsurface_scatter_anisotropy: float = 0.0
  fuzz_weight: float = 0.0
  fuzz_color: Color = (1.0, 1.0, 1.0)
  fuzz_roughness: float = 0.5
  coat_color: Color = (1.0, 1.0, 1.0)
  coat_darkening: float = 1.0
  coat_roughness_anisotropy: float = 0.0
  thin_film_weight: float = 0.0
  thin_film_thickness: float = 0.5
  thin_film_ior: float = 1.4
  textures: tuple[tuple[str, TextureRef], ...] = ()


# No slots: the table is built from each material's `vars` (`table.record_columns`).
@dataclasses.dataclass(frozen=True, kw_only=True)
class Material:
  """An OpenPBR Surface 1.1.1 material: the inputs the core renders, spec names and defaults.

  The core renders OpenPBR's real-time subset ("Flexibility of implementation"; the inputs it
  renders and how: `COVERAGE`). What each value becomes on the device:
  `core.pipeline.derive.materials`.

  Attributes:
    base_weight: [0, 1], multiplies `base_color`.
    base_color: [0, 1] each.
    base_metalness: [0, 1] (a linear mix of metal and dielectric).
    specular_weight: >= 0 (soft range [0, 1]); dielectric F0 = w F0(ior), capped at 1.
    specular_color: [0, 1]: dielectric tint and metal edge tint (carried, not rendered).
    specular_roughness: [0, 1], perceptual (GGX alpha = r^2).
    specular_ior: > 0, relative to the outside medium.
    transmission_weight: [0, 1]: a specular-only see-through layer; 1 on a solid (not thin-walled)
      bends the camera rays and absorbs along their path inside (`transmission_color`).
    transmission_color: [0, 1]: tints the transmitted light (the derived base color).
    coat_weight: [0, 1].
    coat_roughness: [0, 1].
    coat_ior: > 0 (checked even without a coat). COAT_GLTF coats use 1.5.
    emission_luminance: >= 0 [nits] (MuJoCo: its display-relative scale, `physical_emission`
      False).
    emission_color: >= 0, finite; radiance = emission_color x emission_luminance.
    geometry_opacity: [0, 1] (presence; < 1 needs `ext.alpha_mode` BLEND or MASK).
    geometry_thin_walled: thin-walled geometry.
    textures: slot -> TextureRef, sorted by slot, one per slot. A bound slot's value is
      `field constant x (scale x texel + bias)`.
    extras: the other OpenPBR inputs (None: every one at its default).
    ext: the raywarp extensions.
    displacement: optional load-time scalar displacement, baked into ordinary mesh geometry.
    name: source name (diagnostics, per-material writes by name: part of the dedup key).
  """

  base_weight: float = 1.0
  base_color: Color = (0.8, 0.8, 0.8)
  base_metalness: float = 0.0
  specular_weight: float = 1.0
  specular_color: Color = (1.0, 1.0, 1.0)
  specular_roughness: float = 0.3
  specular_ior: float = 1.5
  transmission_weight: float = 0.0
  transmission_color: Color = (1.0, 1.0, 1.0)
  coat_weight: float = 0.0
  coat_roughness: float = 0.0
  coat_ior: float = 1.6
  emission_luminance: float = 0.0
  emission_color: Color = (1.0, 1.0, 1.0)
  geometry_opacity: float = 1.0
  geometry_thin_walled: bool = False
  textures: tuple[tuple[Slot, TextureRef], ...] = ()
  extras: OpenPbrExtras | None = None
  ext: RaywarpExt = RaywarpExt()
  displacement: Displacement | None = None
  name: str = ""

  def texture(self, slot: Slot) -> TextureRef | None:
    """Binding of `slot`, or None."""
    return next((ref for s, ref in self.textures if s == slot), None)

  def with_texture(self, slot: Slot, ref: TextureRef | None) -> Material:
    """Copy with `slot` bound to `ref` (None unbinds it)."""
    rest = tuple((s, r) for s, r in self.textures if s != slot)
    textures = rest if ref is None else (*rest, (slot, ref))
    return dataclasses.replace(self, textures=tuple(sorted(textures, key=operator.itemgetter(0))))


# Fields of the records that are not OpenPBR inputs.
_NOT_INPUTS: Final = frozenset({"textures", "extras", "ext", "displacement", "name"})
# OpenPBR's vector inputs defaulting to the geometry's own frame (a normal map is their only form).
# `geometry_detail_normal` is the raywarp extension's detail normal map: a second tangent-space
# normal map combined with `geometry_normal` by reoriented normal mapping at the hit (its own
# TextureRef UV transform: an independent tiling), the base layer of OmniPBR's
# `detail_normalmap_texture` / MDL `add_detail_normal`.
GEOMETRY_INPUTS: Final = (
  "geometry_normal",
  "geometry_coat_normal",
  "geometry_tangent",
  "geometry_coat_tangent",
  "geometry_detail_normal",
)
# Every OpenPBR 1.1.1 input name, in a fixed order: `Material`'s, `OpenPbrExtras`', then the
# geometry vectors (with the detail normal extension; `MaterialExtras.tex_input` indexes it).
INPUTS: Final = (
  *(f.name for f in dataclasses.fields(Material) if f.name not in _NOT_INPUTS),
  *(f.name for f in dataclasses.fields(OpenPbrExtras) if f.name not in _NOT_INPUTS),
  *GEOMETRY_INPUTS,
)
# The inputs a texture `Slot` drives: the slot each binds (a texture on any other input goes to
# `OpenPbrExtras.textures`; OCCLUSION is a raywarp extension, not an input).
TEXTURE_SLOTS: Final = {
  "base_color": Slot.BASE_COLOR,
  "base_metalness": Slot.BASE_METALNESS,
  "specular_weight": Slot.SPECULAR_WEIGHT,
  "specular_roughness": Slot.SPECULAR_ROUGHNESS,
  "transmission_weight": Slot.TRANSMISSION_WEIGHT,
  "emission_color": Slot.EMISSION_COLOR,
  "geometry_opacity": Slot.GEOMETRY_OPACITY,
  "geometry_normal": Slot.GEOMETRY_NORMAL,
}
