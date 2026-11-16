"""Color spaces: the ASWF Color Interop Forum interop IDs, with their OCIO / MaterialX aliases.

One registry for every color-managed value of a scene: textures (`SceneBuilder.add_texture`,
`Textures.colorspace`), environment maps, MaterialX constants and images, and the render outputs
(`srgb_rec709_display`, `core.outputs.render_outputs.RenderOutputs.color_spaces`). Member values are
the interop IDs that OCIO 2.4+ configs and MaterialX's `*_scene` names use.

Storage: `Textures.colorspace` (uint8) holds `code(space)`, the index in `COLOR_SPACES`, a frozen
order: SRGB_REC709 is 0 and LIN_REC709 1, the values of the former SRGB / LINEAR enum, so the
`.wpbr` bytes of existing scenes are unchanged. New members go at the end.

A space's transfer applies to every texel, 8-bit codes and float values alike (one rule, whatever
the primaries, as OCIO and MaterialX convert): a float texture explicitly tagged with an sRGB or
gamma space holds encoded values and is decoded (USD "auto" and unauthored float files are
LIN_REC709: `auto_colorspace`).

The upload stores the `STORED_AS_IS` spaces without touching the texels (8-bit SRGB_REC709 sRGB
encoded, decoded after sampling; the others linear), except float SRGB_REC709 texels, which are
converted like any other space (`stored_as_is`). Color textures in any other space are
converted per texel (`decode`: the transfer decoded, the primaries converted to Rec.709, negatives
clamped; `core.pipeline.textures.pool`). Data textures are never converted.

Conversion matrices (linear RGB in a space -> linear Rec.709, D65) are the normalized primary
matrices of the spaces' chromaticities, with a Bradford chromatic adaptation from the ACES white
(x 0.32168, y 0.33767) to D65 for AP0 / AP1. Chromaticities:
- Rec.709: ITU-R BT.709-6 (R 0.64 0.33, G 0.30 0.60, B 0.15 0.06; D65 0.3127 0.3290);
- AP1: Academy S-2014-004 (R 0.713 0.293, G 0.165 0.830, B 0.128 0.044);
- AP0: SMPTE ST 2065-1 (R 0.7347 0.2653, G 0.0 1.0, B 0.0001 -0.0770);
- P3-D65: SMPTE EG 432-1 (R 0.680 0.320, G 0.265 0.690, B 0.150 0.060);
- Rec.2020: ITU-R BT.2020-2 (R 0.708 0.292, G 0.170 0.797, B 0.131 0.046);
- Adobe RGB (1998): Adobe RGB (1998) Color Image Encoding, version 2005-05, §4.2 (R 0.64 0.33,
  G 0.21 0.71, B 0.15 0.06; D65).

Transfers: sRGB's piecewise curve (IEC 61966-2-1); pure gamma 1.8 / 2.2 / 2.4 (MaterialX's
`gamma18` / `gamma22` / `gamma24` and `g22_ap1`, OCIO's `Gamma 2.2 AP1 - Texture`, exponent 2.2);
Adobe RGB's pure gamma 563/256 = 2.19921875 (the same encoding, §4.3.1.2, as OCIO's
`g22_adobergb_scene`).

They are the values OCIO publishes (the `ACES-AP1_to_LINEAR-REC709_BFD` builtin and the ACES
studio config's `*_to_Linear Rec.709` transforms, derived the same way); the test suite
re-derives each from the chromaticities within 1e-6.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Final, Literal

import numpy as np

from raywarp.scene.images import mips
from raywarp.scene.images.decode import is_float, linear_rgba
from raywarp.scene.model.types import Array, LinearRGBA, Texels

# A linear or encoded RGB triple.
type Color = tuple[float, float, float]
# A row-major 3x3 matrix applied to column RGB vectors.
type Matrix = tuple[Color, Color, Color]


class ColorSpace(enum.StrEnum):
  """A color space, valued by its Color Interop Forum interop ID."""

  SRGB_REC709 = "srgb_rec709_scene"  # sRGB piecewise transfer, Rec.709 primaries (8-bit color images)
  LIN_REC709 = "lin_rec709_scene"  # linear Rec.709: the renderer's working space
  G22_REC709 = "g22_rec709_scene"  # pure gamma 2.2, Rec.709 primaries
  G18_REC709 = "g18_rec709_scene"  # pure gamma 1.8
  G24_REC709 = "g24_rec709_scene"  # pure gamma 2.4
  LIN_AP1 = "lin_ap1_scene"  # ACEScg
  LIN_AP0 = "lin_ap0_scene"  # ACES2065-1
  LIN_P3D65 = "lin_p3d65_scene"  # linear Display P3 primaries, D65 white
  SRGB_P3D65 = "srgb_p3d65_scene"  # sRGB transfer, P3-D65 primaries
  LIN_REC2020 = "lin_rec2020_scene"  # linear Rec.2020
  DATA = "data"  # raw values (normals, roughness ...): never converted
  SRGB_REC709_DISPLAY = "srgb_rec709_display"  # display-referred sRGB: outputs only
  G22_ADOBERGB = "g22_adobergb_scene"  # Adobe RGB (1998): gamma 563/256, Adobe RGB primaries
  LIN_ADOBERGB = "lin_adobergb_scene"  # linear Adobe RGB primaries, D65 white
  G22_AP1 = "g22_ap1_scene"  # pure gamma 2.2, AP1 (ACEScg) primaries


# The frozen storage order (`code`): append only.
COLOR_SPACES: Final = (
  ColorSpace.SRGB_REC709,
  ColorSpace.LIN_REC709,
  ColorSpace.G22_REC709,
  ColorSpace.G18_REC709,
  ColorSpace.G24_REC709,
  ColorSpace.LIN_AP1,
  ColorSpace.LIN_AP0,
  ColorSpace.LIN_P3D65,
  ColorSpace.SRGB_P3D65,
  ColorSpace.LIN_REC2020,
  ColorSpace.DATA,
  ColorSpace.SRGB_REC709_DISPLAY,
  ColorSpace.G22_ADOBERGB,
  ColorSpace.LIN_ADOBERGB,
  ColorSpace.G22_AP1,
)
_CODES: Final = {space: i for i, space in enumerate(COLOR_SPACES)}

# Spaces that only describe outputs (rejected for textures).
OUTPUT_ONLY: Final = frozenset({ColorSpace.SRGB_REC709_DISPLAY})
# Spaces the upload stores without converting the texels (sRGB-encoded 8-bit or linear).
STORED_AS_IS: Final = frozenset({ColorSpace.SRGB_REC709, ColorSpace.LIN_REC709, ColorSpace.DATA})
# The luminance Y of linear Rec.709 RGB (ITU-R BT.709-6, D65): the one set of luma weights.
REC709_Y: Final = (0.2126, 0.7152, 0.0722)


def stored_as_is(space: ColorSpace, *, float_texels: bool) -> bool:
  """Whether the upload stores texels of `space` untouched (`STORED_AS_IS`; float SRGB_REC709 texels
  are sRGB-encoded values the samplers would read as linear: converted)."""
  return space in STORED_AS_IS and not (float_texels and space == ColorSpace.SRGB_REC709)


# Lower-case alias -> space: the interop IDs, OCIO / MaterialX names (current and deprecated), MDL
# gamma modes, the color space names of OCIO's ACES configs (ACES 1.x "<family> - <name>" and the
# 2.x studio / CG config names) and common spellings.
ALIASES: Final = {
  **{space.value: space for space in ColorSpace},
  **dict.fromkeys(
    (
      "lin_rec709",
      "linear",
      "lin_srgb",
      "linear rec.709 (srgb)",
      "linear rec.709",
      "linear srgb",
      "scene-linear rec.709-srgb",
      "utility - linear - srgb",
      "utility - linear - rec.709",
      "gamma_linear",
    ),
    ColorSpace.LIN_REC709,
  ),
  **dict.fromkeys(
    (
      "srgb_texture",
      "srgb",
      "srgb - texture",
      "srgb_tx",
      "utility - srgb - texture",
      "input - generic - srgb - texture",
      "srgb encoded rec.709 (srgb)",
      "srgb encoded rec.709",
      "gamma_srgb",
    ),
    ColorSpace.SRGB_REC709,
  ),
  **dict.fromkeys(("acescg", "lin_ap1", "aces - acescg", "acescg linear"), ColorSpace.LIN_AP1),
  **dict.fromkeys(("aces2065-1", "lin_ap0", "aces - aces2065-1", "aces2065-1 linear"), ColorSpace.LIN_AP0),
  # MaterialX's gamma18 / gamma22 / gamma24 (cmlib): pure gamma, Rec.709 primaries.
  **dict.fromkeys(
    (
      "g22_rec709",
      "gamma22",
      "gamma 2.2 rec.709 - texture",
      "utility - gamma 2.2 - rec.709 - texture",
      "gamma 2.2 encoded rec.709",
    ),
    ColorSpace.G22_REC709,
  ),
  **dict.fromkeys(
    (
      "g18_rec709",
      "gamma18",
      "gamma 1.8 rec.709 - texture",
      "utility - gamma 1.8 - rec.709 - texture",
      "gamma 1.8 encoded rec.709",
    ),
    ColorSpace.G18_REC709,
  ),
  **dict.fromkeys(
    (
      "g24_rec709",
      "gamma24",
      "gamma 2.4 rec.709 - texture",
      "utility - gamma 2.4 - rec.709 - texture",
      "gamma 2.4 encoded rec.709",
    ),
    ColorSpace.G24_REC709,
  ),
  **dict.fromkeys(
    ("lin_p3d65", "lin_displayp3", "linear p3-d65", "utility - linear - p3-d65", "linear display p3"),
    ColorSpace.LIN_P3D65,
  ),
  **dict.fromkeys(
    ("srgb_p3d65", "srgb_displayp3", "srgb encoded p3-d65", "utility - srgb - p3-d65 - texture"),
    ColorSpace.SRGB_P3D65,
  ),
  **dict.fromkeys(("lin_rec2020", "linear rec.2020", "utility - linear - rec.2020"), ColorSpace.LIN_REC2020),
  # MaterialX's adobergb / lin_adobergb / g22_ap1 (cmlib), the Color Interop Forum's display names
  # and the ACES configs' AP1 gamma 2.2 texture names.
  **dict.fromkeys(("adobergb", "g22_adobergb", "adobe rgb (1998)", "gamma 2.2 encoded adobergb"), ColorSpace.G22_ADOBERGB),
  **dict.fromkeys(("lin_adobergb", "linear adobergb", "linear adobe rgb (1998)"), ColorSpace.LIN_ADOBERGB),
  **dict.fromkeys(
    ("g22_ap1", "gamma 2.2 ap1 - texture", "utility - gamma 2.2 - ap1 - texture", "gamma 2.2 encoded ap1"),
    ColorSpace.G22_AP1,
  ),
  # "identity": the data name of some USD exporters.
  **dict.fromkeys(("raw", "none", "utility - raw", "non-color", "identity"), ColorSpace.DATA),
}

# Linear AP1 (ACEScg) -> linear Rec.709, Bradford-adapted (module docstring).
ACESCG_TO_REC709: Final = (
  (1.70505099, -0.62179212, -0.08325887),
  (-0.13025642, 1.14080474, -0.01054832),
  (-0.02400336, -0.12896898, 1.15297233),
)
# Linear AP0 (ACES2065-1) -> linear Rec.709, Bradford-adapted.
ACES2065_TO_REC709: Final = (
  (2.52168619, -1.13413099, -0.38755520),
  (-0.27647991, 1.37271909, -0.09623917),
  (-0.01537806, -0.15297534, 1.16835340),
)
# Linear P3-D65 -> linear Rec.709 (same white: no adaptation).
P3D65_TO_REC709: Final = (
  (1.22494018, -0.22494018, 0.0),
  (-0.04205695, 1.04205695, 0.0),
  (-0.01963755, -0.07863605, 1.09827360),
)
# Linear Rec.2020 -> linear Rec.709 (same white).
REC2020_TO_REC709: Final = (
  (1.66049100, -0.58764114, -0.07284986),
  (-0.12455047, 1.13289990, -0.00834942),
  (-0.01815076, -0.10057890, 1.11872966),
)
# Linear Adobe RGB (1998) -> linear Rec.709 (same white).
ADOBERGB_TO_REC709: Final = (
  (1.39835574, -0.39835574, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, -0.04292899, 1.04292899),
)
# Adobe RGB (1998)'s encoding exponent (§4.3.1.2: 2 + 51/256).
_ADOBE_GAMMA: Final = 563.0 / 256.0

# Color channels of an RGB(A) texel.
_RGB: Final = 3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Encoding:
  """How a space encodes linear Rec.709: its transfer (with a pure gamma's exponent) and its
  primaries (None: Rec.709's)."""

  transfer: Literal["linear", "srgb", "gamma"]
  gamma: float = 1.0
  to_rec709: Matrix | None = None


_LINEAR: Final = _Encoding(transfer="linear")
_ENCODINGS: Final = {
  ColorSpace.SRGB_REC709: _Encoding(transfer="srgb"),
  ColorSpace.LIN_REC709: _LINEAR,
  ColorSpace.G22_REC709: _Encoding(transfer="gamma", gamma=2.2),
  ColorSpace.G18_REC709: _Encoding(transfer="gamma", gamma=1.8),
  ColorSpace.G24_REC709: _Encoding(transfer="gamma", gamma=2.4),
  ColorSpace.LIN_AP1: _Encoding(transfer="linear", to_rec709=ACESCG_TO_REC709),
  ColorSpace.LIN_AP0: _Encoding(transfer="linear", to_rec709=ACES2065_TO_REC709),
  ColorSpace.LIN_P3D65: _Encoding(transfer="linear", to_rec709=P3D65_TO_REC709),
  ColorSpace.SRGB_P3D65: _Encoding(transfer="srgb", to_rec709=P3D65_TO_REC709),
  ColorSpace.LIN_REC2020: _Encoding(transfer="linear", to_rec709=REC2020_TO_REC709),
  ColorSpace.DATA: _LINEAR,
  ColorSpace.G22_ADOBERGB: _Encoding(transfer="gamma", gamma=_ADOBE_GAMMA, to_rec709=ADOBERGB_TO_REC709),
  ColorSpace.LIN_ADOBERGB: _Encoding(transfer="linear", to_rec709=ADOBERGB_TO_REC709),
  ColorSpace.G22_AP1: _Encoding(transfer="gamma", gamma=2.2, to_rec709=ACESCG_TO_REC709),
}


def parse(name: str) -> ColorSpace | None:
  """The space named `name` (an interop ID or an alias of `ALIASES`, any case; surrounding blanks
  ignored), None when unknown (the caller reports it with its own key and source). "" and "auto"
  are context defaults the caller resolves (`auto_colorspace`): None too.
  """
  return ALIASES.get(name.strip().lower())


def auto_colorspace(path: str, *, color: bool) -> ColorSpace:
  """The color space of an image file without an authored one (USD "auto", Newton envmaps):
  DATA for data maps (not `color`), else LIN_REC709 for floating-point (EXR, HDR) images and
  SRGB_REC709 for 8-bit ones. DATA and LIN_REC709 are stored alike
  (`core.pipeline.textures.pool`)."""
  if not color:
    return ColorSpace.DATA
  return ColorSpace.LIN_REC709 if is_float(path) else ColorSpace.SRGB_REC709


def texture_space(value: ColorSpace | str, *, where: str) -> ColorSpace:
  """The space of a texture given as a member or a name (`parse`).

  Raises:
    ValueError: for an unknown name or an output-only space.
  """
  space = value if isinstance(value, ColorSpace) else parse(value)
  if space is None:
    known = ", ".join(s.value for s in COLOR_SPACES if s not in OUTPUT_ONLY)
    raise ValueError(f"{where}: colorspace {value!r} is not a known name (interop IDs: {known}; or an alias)")
  if space in OUTPUT_ONLY:
    raise ValueError(f"{where}: colorspace {space.value!r} describes outputs only, not a texture")
  return space


def code(space: ColorSpace) -> int:
  """The stored uint8 code of `space` (`Textures.colorspace`): its index in COLOR_SPACES."""
  return _CODES[space]


def from_code(value: int) -> ColorSpace:
  """The space of stored code `value`.

  Raises:
    ValueError: for a code outside COLOR_SPACES.
  """
  if not 0 <= value < len(COLOR_SPACES):
    raise ValueError(f"color space code {value} is not in [0, {len(COLOR_SPACES)})")
  return COLOR_SPACES[value]


def _encoding(space: ColorSpace) -> _Encoding:
  """The encoding of a scene-referred space.

  Raises:
    ValueError: for an output-only space.
  """
  if space in OUTPUT_ONLY:
    raise ValueError(f"colorspace {space.value!r} describes outputs only: no conversion to linear Rec.709")
  return _ENCODINGS[space]


def _srgb_decode(c: float) -> float:
  """The sRGB transfer function's inverse (sRGB-encoded value -> linear)."""
  return c / 12.92 if c <= mips.SRGB_KNEE else ((c + 0.055) / 1.055) ** 2.4


def to_lin_rec709(rgb: Color, space: ColorSpace) -> Color:
  """`rgb` authored in `space` as linear Rec.709 (MaterialX constants): the transfer decoded,
  then the 3x3 to Rec.709 primaries (Bradford adaptation to D65 where the white differs). DATA and
  LIN_REC709 return `rgb` unchanged; out-of-gamut results keep their negative components (the
  caller clamps or rejects them).

  Raises:
    ValueError: for an output-only space.
  """
  enc = _encoding(space)
  r, g, b = rgb
  if enc.transfer == "srgb":
    r, g, b = _srgb_decode(r), _srgb_decode(g), _srgb_decode(b)
  elif enc.transfer == "gamma":
    r, g, b = max(r, 0.0) ** enc.gamma, max(g, 0.0) ** enc.gamma, max(b, 0.0) ** enc.gamma
  if enc.to_rec709 is None:
    return r, g, b
  x, y, z = (sum(m * c for m, c in zip(row, (r, g, b), strict=True)) for row in enc.to_rec709)
  return x, y, z


def _transfer(x: Array, enc: _Encoding) -> Array:
  """Encoded float32 values -> linear (`enc`'s transfer), float32 (Python float operands keep
  float32 arrays float32)."""
  if enc.transfer == "srgb":
    return mips.srgb_decode(x)
  if enc.transfer == "gamma":
    return np.maximum(x, 0.0) ** enc.gamma
  return x


# The 8-bit transfer decode of each space with a non-linear transfer: code -> linear (float32),
# the tables of `mips` (one per transfer, shared with the mips and the core).
_LUTS: Final = {
  space: mips.SRGB_TO_LINEAR if enc.transfer == "srgb" else mips.gamma_lut(enc.gamma)
  for space, enc in _ENCODINGS.items()
  if enc.transfer != "linear"
}


def rec709_primaries(space: ColorSpace) -> bool:
  """Whether `space` has Rec.709 primaries (or none: data)."""
  enc = _ENCODINGS.get(space)
  return enc is None or enc.to_rec709 is None


def texel_transfer(space: ColorSpace) -> mips.Transfer:
  """The 8-bit transfer of `space`'s texels ("srgb", a pure gamma's exponent, else "linear"): how
  its mips decode before filtering. Output-only spaces are sRGB-encoded."""
  if space in OUTPUT_ONLY:
    return "srgb"
  enc = _ENCODINGS[space]
  if enc.transfer == "gamma":
    return enc.gamma
  return "srgb" if enc.transfer == "srgb" else "linear"


def linear_rec709(texels: Texels, space: ColorSpace) -> LinearRGBA:
  """(h, w, c) texels of a color image authored in `space` as (h, w, 4) linear Rec.709 RGBA
  (float32, not clamped: out-of-gamut colors keep negative components): 8-bit values scaled to
  [0, 1] (their transfer through a 256-entry table), grey broadcast, the transfer decoded, the
  primaries converted; alpha (or 1) in channel 3.

  Raises:
    ValueError: for DATA or an output-only space (data texels are never converted).
  """
  if space == ColorSpace.DATA:
    raise ValueError("colorspace 'data': data texels are never converted")
  enc = _encoding(space)
  rgba = linear_rgba(texels, transfer="linear")
  if texels.dtype == np.uint8 and space in _LUTS:
    codes = texels[..., :3] if texels.shape[2] >= _RGB else np.repeat(texels[..., :1], 3, axis=2)
    rgba[..., :3] = _LUTS[space][codes]
  else:
    rgba[..., :3] = _transfer(rgba[..., :3], enc)
  if enc.to_rec709 is not None:
    rgba[..., :3] = rgba[..., :3] @ np.asarray(enc.to_rec709, np.float32).T
  return rgba


def stored_space(value: int, *, where: str) -> ColorSpace:
  """The space of a stored texture code (`Textures.colorspace`, e.g. read from a `.wpbr`).

  Raises:
    ValueError: for a code outside COLOR_SPACES or of an output-only space.
  """
  space = from_code(value)
  if space in OUTPUT_ONLY:
    raise ValueError(f"{where}: colorspace code {value} ({space.value!r}) describes outputs only, not a texture")
  return space


def decode(texels: Texels, space: ColorSpace) -> LinearRGBA:
  """(h, w, c) texels of a color image authored in `space` as (h, w, 4) linear Rec.709 RGBA
  (float32): `linear_rec709` with negatives clamped to 0. The upload re-encodes 8-bit sources as
  8-bit sRGB.

  Raises:
    ValueError: for DATA or an output-only space (data texels are never converted).
  """
  rgba = linear_rec709(texels, space)
  np.maximum(rgba[..., :3], 0.0, out=rgba[..., :3])
  return rgba
