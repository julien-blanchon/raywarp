"""OpenPBR Surface 1.1.1 inputs of the canonical `Material`: the coverage table of what the core
renders, `clamp_material` (front ends that build a `Material` directly: imported assets load with
out-of-range inputs clamped and reported, while the builder's `checks.material` fails loud) and
the anisotropic roughness inversion of the MDL mapping.

`Material` is an OpenPBR surface (`material.Material`). What each input becomes on the device (the
coat, the base IOR under it, the roughening, the transmission / subsurface colors, `base_weight`)
and the reports of what does not render (`gpu.material.<input>`) belong to the upload derivation
(`core.pipeline.derive.materials`).
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Final

from raywarp.scene.model.checks import COLOR_RANGES, FLOAT_RANGES, POSITIVE
from raywarp.scene.model.material import INPUTS, Color, Material, OpenPbrExtras
from raywarp.scene.model.report import LoadReport

# The spec defaults of the OpenPBR inputs, read from `Material` / `OpenPbrExtras` (one source of
# truth), by value type.
_FIELDS: Final = (*dataclasses.fields(Material), *dataclasses.fields(OpenPbrExtras))
FLOAT_DEFAULTS: Final = {f.name: f.default for f in _FIELDS if f.name in INPUTS and type(f.default) is float}
# The smallest MASK threshold (cutoffs are in (0, 1]): below half an 8-bit opacity step, so a glTF
# alphaCutoff of 0 keeps every texel but the fully transparent ones.
MIN_ALPHA_CUTOFF: Final = 1.0 / 512.0


class Status(enum.StrEnum):
  """How the core renders an OpenPBR input."""

  SUPPORTED = "supported"  # same semantics
  APPROXIMATED = "approximated"  # rendered with a documented approximation
  MISSING = "missing"  # carried, not rendered (reported `gpu.material.<input>` when authored)


@dataclasses.dataclass(frozen=True, slots=True)
class Coverage:
  """How the core renders an OpenPBR input.

  Attributes:
    status: Status.
    how: what the core does with it (the documentation's coverage table).
    when: a MISSING input is reported by the upload (`core.pipeline.derive.materials`) when authored
      off its default while each of these inputs is > 0 (its lobe is on); (): always.
  """

  status: Status
  how: str
  when: tuple[str, ...] = ()


# Every input: how the core renders it. The documentation shows this table; it drives the upload's
# `gpu.material.<input>` reports (`core.pipeline.derive.materials`).
COVERAGE: Final = {
  "base_weight": Coverage(Status.SUPPORTED, "multiplies the base color"),
  "base_color": Coverage(Status.SUPPORTED, "the Lambert base color"),
  "base_diffuse_roughness": Coverage(
    Status.APPROXIMATED,
    "constant/textured EON with sampled area/environment; existing specular/coat coupling and probe GI; "
    "subsurface/transmission combinations unsupported; shared material UV transform",
  ),
  "base_metalness": Coverage(Status.SUPPORTED, "a linear mix of metal and dielectric"),
  "specular_weight": Coverage(
    Status.APPROXIMATED,
    "weighted dielectric F0 with index-match zero; whole metal lobe scaled; Schlick and mixed-metal compensation approximate",
  ),
  "specular_color": Coverage(Status.MISSING, "carried: the dielectric tint and the metal edge tint are not rendered"),
  "specular_roughness": Coverage(Status.SUPPORTED, "GGX alpha = r^2, roughened under an OpenPBR coat"),
  "specular_ior": Coverage(Status.SUPPORTED, "F0 (the ratio to `coat_ior` under an OpenPBR coat), Schlick Fresnel"),
  "specular_roughness_anisotropy": Coverage(
    Status.SUPPORTED,
    "anisotropic GGX D / Smith V (alpha_t = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_b = (1 - a) alpha_t) along the "
    "mesh's UV tangent (generated per vertex) or the primitive's, turned by `ext.anisotropy_rotation`; bent-normal "
    "image-based lighting, anisotropic VNDF reflection rays; area lights approximate it with the bent normal",
  ),
  "transmission_weight": Coverage(
    Status.APPROXIMATED,
    "a specular-only see-through layer; a solid (weight 1, not thin-walled) bends camera rays (Snell, total internal "
    "reflection), shadow and reflection rays pass it straight",
  ),
  "transmission_color": Coverage(
    Status.APPROXIMATED,
    "a closed solid's Beer-Lambert extinction (reached at `transmission_depth`, else at its instances' median mean "
    "chord), else blended into the base color (the tint at each surface crossed)",
  ),
  "transmission_depth": Coverage(
    Status.SUPPORTED, "the depth at which a solid's absorption reaches `transmission_color`", ("transmission_weight",)
  ),
  "transmission_scatter": Coverage(Status.MISSING, "no volume scattering", ("transmission_weight",)),
  "transmission_scatter_anisotropy": Coverage(Status.MISSING, "no volume scattering", ("transmission_weight",)),
  "transmission_dispersion_scale": Coverage(Status.MISSING, "no dispersion", ("transmission_weight",)),
  "transmission_dispersion_abbe_number": Coverage(
    Status.MISSING, "no dispersion", ("transmission_weight", "transmission_dispersion_scale")
  ),
  "subsurface_weight": Coverage(Status.APPROXIMATED, "`subsurface_color` blended into the base color (the spec's LOD rule)"),
  "subsurface_color": Coverage(Status.APPROXIMATED, "the diffuse color under `subsurface_weight`"),
  "subsurface_radius": Coverage(Status.MISSING, "no subsurface transport", ("subsurface_weight",)),
  "subsurface_radius_scale": Coverage(Status.MISSING, "no subsurface transport", ("subsurface_weight",)),
  "subsurface_scatter_anisotropy": Coverage(Status.MISSING, "no subsurface transport", ("subsurface_weight",)),
  "fuzz_weight": Coverage(
    Status.APPROXIMATED,
    "a Charlie sheen lobe (Neubelt visibility) over the coated base, albedo-scaled 1 - F E_fuzz (E from the cloth "
    "table, at most 1); image-based light from the prefiltered environment at the fuzz roughness",
  ),
  "fuzz_color": Coverage(Status.SUPPORTED, "the fuzz lobe's tint (constant or textured)"),
  "fuzz_roughness": Coverage(Status.SUPPORTED, "the Charlie roughness alpha (glTF sheenRoughness^2)"),
  "coat_weight": Coverage(Status.SUPPORTED, "the coat's coverage C: C f_coat + lerp(1, T (1 - E_coat), C) f_base"),
  "coat_color": Coverage(
    Status.SUPPORTED, "the coat's transmittance T^(1 / mu_i + 1 / mu_o) (coat_color = T^2 at normal incidence)"
  ),
  "coat_roughness": Coverage(Status.SUPPORTED, "the coat's GGX roughness"),
  "coat_roughness_anisotropy": Coverage(Status.MISSING, "isotropic coat", ("coat_weight",)),
  "coat_ior": Coverage(Status.SUPPORTED, "the coat's own Schlick Fresnel F0((n - 1) / (n + 1))^2 and the base IOR ratio"),
  "coat_darkening": Coverage(
    Status.SUPPORTED, "OpenPBR's lerp(1, (1 - K) / (1 - E_b K), C d) on the base under an OpenPBR coat"
  ),
  "thin_film_weight": Coverage(Status.MISSING, "no thin-film iridescence"),
  "thin_film_thickness": Coverage(Status.MISSING, "no thin-film iridescence", ("thin_film_weight",)),
  "thin_film_ior": Coverage(Status.MISSING, "no thin-film iridescence", ("thin_film_weight",)),
  "emission_luminance": Coverage(Status.SUPPORTED, "radiance [nits]"),
  "emission_color": Coverage(Status.APPROXIMATED, "not attenuated by an OpenPBR coat (reported)"),
  "geometry_opacity": Coverage(Status.SUPPORTED, "presence, with `ext.alpha_mode` BLEND"),
  "geometry_thin_walled": Coverage(
    Status.SUPPORTED,
    "the core's surfaces are thin; a transmissive thin wall reflects its two interfaces' 2 F / (1 + F) (Schlick on that F0)",
  ),
  "geometry_normal": Coverage(Status.SUPPORTED, "the normal map (`Slot.GEOMETRY_NORMAL`; strengths / flips: its scale / bias)"),
  "geometry_coat_normal": Coverage(
    Status.SUPPORTED, "the coat's own normal map (unbound: the unmapped normal; COAT_BASE_NORMAL: the base's)"
  ),
  "geometry_tangent": Coverage(Status.MISSING, "the mesh's UV tangent (per vertex) or the primitive's"),
  "geometry_coat_tangent": Coverage(Status.MISSING, "isotropic coat: no tangent needed"),
  "geometry_detail_normal": Coverage(
    Status.SUPPORTED, "raywarp extension: a detail normal map combined with geometry_normal by RNM, own UV transform"
  ),
}


@dataclasses.dataclass(slots=True, kw_only=True)
class _Inputs:
  """The clamping of one material's inputs: the report of what was clamped, and where."""

  report: LoadReport
  where: object

  def clamped(self, key: str, v: float, low: float, high: float) -> float:
    """`v` clamped to [low, high], reported `openpbr.<key>(clamped ...)` when outside."""
    if low <= v <= high:
      return v
    self.report.approx(f"openpbr.{key}(clamped to [{low}, {high}])", self.where)
    return min(max(v, low), high)

  def finite_in_range(self, name: str, v: float) -> float:
    """Float input `name` = `v` in its range when finite (else `v`): clamped and reported outside;
    a non-positive IOR / Abbe number reads as its default (reported)."""
    if not math.isfinite(v):
      return v
    if name not in POSITIVE:
      return self.clamped(name, v, *FLOAT_RANGES[name])
    if v > 0.0:
      return v
    self.report.approx(f"openpbr.{name}(not positive: the default)", self.where)
    return FLOAT_DEFAULTS[name]

  def finite_color_in_range(self, name: str, v: Color) -> Color:
    """Color input `name` = `v` in its range when finite (else `v`): clamped and reported
    outside."""
    low, high = COLOR_RANGES[name]
    if not all(map(math.isfinite, v)) or all(low <= x <= high for x in v):
      return v
    self.report.approx(f"openpbr.{name}(clamped to [{low}, {high}])", self.where)
    r, g, b = (min(max(x, low), high) for x in v)
    return r, g, b

  def finite_fraction(self, key: str, v: float, low: float = 0.0) -> float:
    """`ext.` fraction `v` clamped to [low, 1] when finite, else `v`."""
    return self.clamped(key, v, low, 1.0) if math.isfinite(v) else v


def anisotropic_roughness(alpha_t: float, alpha_b: float) -> tuple[float, float]:
  """(specular_roughness r, specular_roughness_anisotropy a) whose OpenPBR GGX alphas are
  (alpha_t, alpha_b), 0 <= alpha_b <= alpha_t: OpenPBR's alpha_t = r^2 sqrt(2 / (1 + (1 - a)^2)),
  alpha_b = (1 - a) alpha_t inverted, a = 1 - alpha_b / alpha_t and
  r = sqrt(alpha_t sqrt((1 + (1 - a)^2) / 2)) (r = sqrt(alpha) when isotropic)."""
  if alpha_t <= 0.0:
    return 0.0, 0.0
  k = min(max(alpha_b / alpha_t, 0.0), 1.0)
  return math.sqrt(alpha_t * math.sqrt(0.5 * (1.0 + k * k))), 1.0 - k


def clamp_material(mat: Material, *, report: LoadReport, where: object = None) -> Material:
  """`mat` with its inputs (core and `extras`) and `ext` fractions in range: clamped and reported
  (`openpbr.<input>(clamped ...)`), a non-positive IOR read as its default; `mat` itself when
  every value is in range. For front ends that build a `Material` directly (UsdPreviewSurface),
  so an imported asset loads where the builder's `checks.material` fails. NaN / inf
  values are kept, for `checks.material` to reject.
  """
  inp = _Inputs(report=report, where=where)
  x, c, ext = inp.finite_in_range, inp.finite_color_in_range, mat.ext
  clamped = dataclasses.replace(
    mat,
    base_weight=x("base_weight", mat.base_weight),
    base_color=c("base_color", mat.base_color),
    base_metalness=x("base_metalness", mat.base_metalness),
    specular_weight=x("specular_weight", mat.specular_weight),
    specular_color=c("specular_color", mat.specular_color),
    specular_roughness=x("specular_roughness", mat.specular_roughness),
    specular_ior=x("specular_ior", mat.specular_ior),
    transmission_weight=x("transmission_weight", mat.transmission_weight),
    transmission_color=c("transmission_color", mat.transmission_color),
    coat_weight=x("coat_weight", mat.coat_weight),
    coat_roughness=x("coat_roughness", mat.coat_roughness),
    coat_ior=x("coat_ior", mat.coat_ior),
    emission_luminance=x("emission_luminance", mat.emission_luminance),
    emission_color=c("emission_color", mat.emission_color),
    geometry_opacity=x("geometry_opacity", mat.geometry_opacity),
    extras=None if mat.extras is None else _clamp_extras(inp, mat.extras),
    ext=dataclasses.replace(
      ext,
      alpha_cutoff=inp.finite_fraction("ext.alpha_cutoff", ext.alpha_cutoff, MIN_ALPHA_CUTOFF),
      occlusion_strength=inp.finite_fraction("ext.occlusion_strength", ext.occlusion_strength),
      mirror=inp.finite_fraction("ext.mirror", ext.mirror),
      wrap=inp.finite_fraction("ext.wrap", ext.wrap),
    ),
  )
  return mat if clamped == mat else clamped


def _clamp_extras(inp: _Inputs, e: OpenPbrExtras) -> OpenPbrExtras:
  """`e` with every input in range (`clamp_material`)."""
  x, c = inp.finite_in_range, inp.finite_color_in_range
  return dataclasses.replace(
    e,
    base_diffuse_roughness=x("base_diffuse_roughness", e.base_diffuse_roughness),
    specular_roughness_anisotropy=x("specular_roughness_anisotropy", e.specular_roughness_anisotropy),
    transmission_depth=x("transmission_depth", e.transmission_depth),
    transmission_scatter=c("transmission_scatter", e.transmission_scatter),
    transmission_scatter_anisotropy=x("transmission_scatter_anisotropy", e.transmission_scatter_anisotropy),
    transmission_dispersion_scale=x("transmission_dispersion_scale", e.transmission_dispersion_scale),
    transmission_dispersion_abbe_number=x("transmission_dispersion_abbe_number", e.transmission_dispersion_abbe_number),
    subsurface_weight=x("subsurface_weight", e.subsurface_weight),
    subsurface_color=c("subsurface_color", e.subsurface_color),
    subsurface_radius=x("subsurface_radius", e.subsurface_radius),
    subsurface_radius_scale=c("subsurface_radius_scale", e.subsurface_radius_scale),
    subsurface_scatter_anisotropy=x("subsurface_scatter_anisotropy", e.subsurface_scatter_anisotropy),
    fuzz_weight=x("fuzz_weight", e.fuzz_weight),
    fuzz_color=c("fuzz_color", e.fuzz_color),
    fuzz_roughness=x("fuzz_roughness", e.fuzz_roughness),
    coat_color=c("coat_color", e.coat_color),
    coat_darkening=x("coat_darkening", e.coat_darkening),
    coat_roughness_anisotropy=x("coat_roughness_anisotropy", e.coat_roughness_anisotropy),
    thin_film_weight=x("thin_film_weight", e.thin_film_weight),
    thin_film_thickness=x("thin_film_thickness", e.thin_film_thickness),
    thin_film_ior=x("thin_film_ior", e.thin_film_ior),
  )
