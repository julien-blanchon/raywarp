"""How much light a surface reflects from a light toward the viewer: the BRDF terms every shading
path uses (Filament's standard model: GGX, height-correlated Smith, Schlick, Lambert).

Formulas follow Filament's `shaders/src/surface_brdf.fs` and the Filament PBR document.
Roughness arguments named `alpha` are the remapped roughness α = perceptual_roughness².
"""

from __future__ import annotations

from typing import Final

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Filament clamps perceptual roughness on desktop so that α² stays representable.
MIN_PERCEPTUAL_ROUGHNESS: Final = 0.045
# Dielectric F0 per squared reflectance (Filament: F0 = 0.16 reflectance², 4 % at reflectance 0.5).
DIELECTRIC_F0_SCALE: Final = 0.16
# Floor of the DFG table's directional albedo in the multiscatter compensation 1 / dfg.y.
_MIN_DFG_ALBEDO: Final = 1.0e-4


@wp.func
def d_ggx(
  # In:
  noh: wp.float32,
  alpha: wp.float32,
) -> wp.float32:
  """GGX / Trowbridge-Reitz normal distribution."""
  a2 = alpha * alpha
  f = (noh * a2 - noh) * noh + 1.0
  return a2 / (wp.pi * f * f)


@wp.func
def v_smith_ggx_correlated(
  # In:
  nov: wp.float32,
  nol: wp.float32,
  alpha: wp.float32,
) -> wp.float32:
  """Height-correlated Smith visibility, including the 1 / (4 NoV NoL) factor."""
  a2 = alpha * alpha
  gv = nol * wp.sqrt(nov * nov * (1.0 - a2) + a2)
  gl = nov * wp.sqrt(nol * nol * (1.0 - a2) + a2)
  return 0.5 / (gv + gl)


@wp.func
def f_schlick_rgb(
  # In:
  f0: wp.vec3,
  f90: wp.vec3,
  voh: wp.float32,
) -> wp.vec3:
  """Schlick's Fresnel approximation f0 + (f90 - f0) (1 - VoH)^5, as f0 (1 - Fc) + f90 Fc."""
  x = 1.0 - voh
  x2 = x * x
  fc = x2 * x2 * x
  return f0 * (1.0 - fc) + f90 * fc


@wp.func
def specular_lobe_rgb(
  # In:
  f0: wp.vec3,
  f90: wp.vec3,
  alpha: wp.float32,
  noh: wp.float32,
  nov: wp.float32,
  nol: wp.float32,
  loh: wp.float32,
) -> wp.vec3:
  """Cook-Torrance specular D·V·F with a colored F90 (without the NoL cosine)."""
  return f_schlick_rgb(f0, f90, loh) * (d_ggx(noh, alpha) * v_smith_ggx_correlated(nov, nol, alpha))


@wp.func
def coat_lobe(
  # In:
  f0: wp.float32,
  alpha: wp.float32,
  noh: wp.float32,
  nov: wp.float32,
  nol: wp.float32,
  loh: wp.float32,
) -> wp.float32:
  """Specular D V F of a dielectric coat of normal reflectance f0 (no NoL cosine)."""
  x = 1.0 - loh
  x2 = x * x
  fresnel = f0 + (1.0 - f0) * (x2 * x2 * x)
  return fresnel * d_ggx(noh, alpha) * v_smith_ggx_correlated(nov, nol, alpha)


@wp.func
def diffuse_lambert() -> wp.float32:
  """Lambertian BRDF 1 / pi (times the diffuse albedo)."""
  return wp.static(1.0 / wp.pi)


@wp.func
def surface_f0(
  # In:
  base_color: wp.vec3,
  metallic: wp.float32,
  reflectance: wp.float32,
) -> wp.vec3:
  """Specular reflectance at normal incidence; dielectric F0 = 0.16 · reflectance²."""
  dielectric = wp.static(DIELECTRIC_F0_SCALE) * reflectance * reflectance * (1.0 - metallic)
  return base_color * metallic + wp.vec3(dielectric)


@wp.func
def weighted_fresnel(
  # In:
  base_color: wp.vec3,
  metallic: wp.float32,
  reflectance: wp.float32,
  weight: wp.float32,
) -> tuple[wp.vec3, wp.vec3]:
  """Weighted Schlick coefficients, with no dielectric reflection at index match.

  OpenPBR scales the complete metallic response by weight, while dielectric weight changes
  its IOR (already encoded in reflectance). Schlick remains approximate away from index match.
  Compute multiscatter compensation from surface_f0 *before* applying this metal weight;
  otherwise the compensation would spuriously change as a pure metal's weight changes.
  """
  dielectric = wp.static(DIELECTRIC_F0_SCALE) * reflectance * reflectance * (1.0 - metallic)
  metal = metallic * weight
  f0 = base_color * metal + wp.vec3(dielectric)
  grazing = wp.float32(0.0)
  if reflectance > 0.0:
    grazing = 1.0 - metallic
  return f0, wp.vec3(grazing + metal)


@wp.func
def energy_compensation(
  # In:
  f0: wp.vec3,
  dfg: wp.vec2,
) -> wp.vec3:
  """Multiscatter energy compensation 1 + f0 (1 / dfg.y − 1) (Kulla-Conty via Filament)."""
  return wp.vec3(1.0) + f0 * (1.0 / wp.max(dfg[1], wp.static(_MIN_DFG_ALBEDO)) - 1.0)


@wp.func
def specular_albedo_rgb(
  # In:
  f0: wp.vec3,
  f90: wp.vec3,
  dfg: wp.vec2,
) -> wp.vec3:
  """Directional albedo of the specular lobe (DFG LUT), colored F90: (f90 - f0) dfg.x + f0 dfg.y.

  dfg.x = ∫ V Fc, dfg.y = ∫ V.
  """
  return (f90 - f0) * dfg[0] + f0 * dfg[1]


@wp.func
def specular_albedo(
  # In:
  f0: wp.vec3,
  dfg: wp.vec2,
) -> wp.vec3:
  """`specular_albedo_rgb` with F90 = 1: mix(dfg.x, dfg.y, f0)."""
  return specular_albedo_rgb(f0, wp.vec3(1.0), dfg)


@wp.func
def coupled_diffuse(
  # In:
  base_color: wp.vec3,
  metallic: wp.float32,
  spec_albedo: wp.vec3,
) -> wp.vec3:
  """Lambertian albedo weighted by the energy the specular lobe leaves, (1 - metallic)(1 - E).

  Filament applies the (1 - E) factor to image-based lighting only; using it for every light
  keeps direct and image-based lighting consistent with one energy-conserving BRDF.
  """
  return wp.cw_mul(base_color * (1.0 - metallic), wp.vec3(1.0) - spec_albedo)


@wp.func
def f90_curve(
  # In:
  base_color: wp.vec3,
  metallic: wp.float32,
  reflectance: wp.float32,
  perceptual_roughness: wp.float32,
) -> wp.vec3:
  """F90 of the Unreal / OmniUe4Base specular (ShadingFlag.SPECULAR_CURVE).

  The dielectric layer is df::custom_curve_layer(weight: s, normal_reflectivity: 0.08,
  grazing_reflectivity: 1 - roughness): F0 = 0.08 s, F90 = s (1 - roughness), with the specular
  level s = 2 reflectance^2 (F0 = 0.16 reflectance^2). The metallic lobe is GGX tinted by the
  base color without a Fresnel curve: F90 = F0 = base color.
  """
  # The level s at most 1 (a MuJoCo mirror's reflectance can exceed the curve's range).
  dielectric = wp.min(2.0 * reflectance * reflectance, 1.0) * wp.max(1.0 - perceptual_roughness, 0.0)
  return wp.vec3(dielectric * (1.0 - metallic)) + base_color * metallic


@wp.func
def dominant_direction(
  # In:
  view: wp.vec3,
  normal: wp.vec3,
  alpha: wp.float32,
) -> wp.vec3:
  """Filament's dominant direction of the GGX lobe: normalize(mix(reflect(-view), n, alpha^2))."""
  r = normal * (2.0 * wp.dot(view, normal)) - view
  a2 = alpha * alpha
  return wp.normalize(r * (1.0 - a2) + normal * a2)


@wp.func
def roughness_to_alpha(
  # In:
  perceptual_roughness: wp.float32,
) -> tuple[wp.float32, wp.float32]:
  """Clamps perceptual roughness and returns (perceptual_roughness, alpha)."""
  pr = wp.clamp(perceptual_roughness, wp.static(MIN_PERCEPTUAL_ROUGHNESS), 1.0)
  return pr, pr * pr


# Smallest GGX roughness of an anisotropic axis (Filament's MIN_ROUGHNESS: α² stays representable).
MIN_ANISOTROPIC_ALPHA: Final = MIN_PERCEPTUAL_ROUGHNESS * MIN_PERCEPTUAL_ROUGHNESS
# Smallest Charlie roughness α (Filament clamps the sheen's perceptual roughness like the base's).
MIN_CHARLIE_ALPHA: Final = MIN_PERCEPTUAL_ROUGHNESS * MIN_PERCEPTUAL_ROUGHNESS
# Floor of sin²θh in the Charlie distribution (Filament: 2^-7, keeps the power finite at NoH = 1).
_CHARLIE_MIN_SIN2: Final = 0.0078125


@wp.func
def anisotropic_alpha(
  # In:
  alpha: wp.float32,
  anisotropy: wp.float32,
) -> wp.vec2:
  """(α_t, α_b) of OpenPBR's `specular_roughness_anisotropy` a in [0, 1] (OpenPBR 1.1 §Specular):
  α_t = α sqrt(2 / (1 + (1 - a)²)), α_b = (1 - a) α_t, so α_t² + α_b² = 2 α² (the average
  roughness is kept). The tangent axis is the one the distribution is stretched along."""
  b = 1.0 - wp.clamp(anisotropy, 0.0, 1.0)
  at = alpha * wp.sqrt(2.0 / (1.0 + b * b))
  return wp.vec2(wp.max(at, wp.static(MIN_ANISOTROPIC_ALPHA)), wp.max(at * b, wp.static(MIN_ANISOTROPIC_ALPHA)))


@wp.func
def d_ggx_anisotropic(
  # In:
  at: wp.float32,
  ab: wp.float32,
  toh: wp.float32,
  boh: wp.float32,
  noh: wp.float32,
) -> wp.float32:
  """Anisotropic GGX distribution (Burley 2012; Filament `D_GGX_Anisotropic`): roughness `at`
  along the tangent, `ab` along the bitangent, the half vector's frame cosines ToH, BoH, NoH."""
  a2 = at * ab
  d = wp.vec3(ab * toh, at * boh, a2 * noh)
  b2 = a2 / wp.dot(d, d)
  return a2 * b2 * b2 * wp.static(1.0 / wp.pi)


@wp.func
def v_smith_ggx_correlated_anisotropic(
  # In:
  at: wp.float32,
  ab: wp.float32,
  tov: wp.float32,
  bov: wp.float32,
  tol: wp.float32,
  bol: wp.float32,
  nov: wp.float32,
  nol: wp.float32,
) -> wp.float32:
  """Anisotropic height-correlated Smith visibility with the 1 / (4 NoV NoL) factor (Heitz 2014;
  Filament `V_SmithGGXCorrelated_Anisotropic`). Equals `v_smith_ggx_correlated` at at = ab."""
  lambda_v = nol * wp.length(wp.vec3(at * tov, ab * bov, nov))
  lambda_l = nov * wp.length(wp.vec3(at * tol, ab * bol, nol))
  return 0.5 / (lambda_v + lambda_l)


@wp.func
def anisotropic_bent_normal(
  # In:
  view: wp.vec3,
  normal: wp.vec3,
  tangent: wp.vec3,
  bitangent: wp.vec3,
  anisotropy: wp.float32,
  perceptual_roughness: wp.float32,
) -> wp.vec3:
  """The normal an anisotropic lobe's image-based lookup reflects about (Filament
  `getReflectedVector`, McAuley's bent normal; an approximation of the stretched lobe).

  The lobe is stretched along the tangent for a > 0 (OpenPBR: always), so its reflections
  smear along the bitangent's perpendicular plane: the normal bends toward the plane spanned by
  the stretch direction's grooves (Filament's `anisotropicB` for a >= 0, `anisotropicT` for
  a < 0, its signed convention), by |a| x saturate(5 perceptual roughness) (mirror-like lobes
  stay unbent).
  """
  direction = bitangent
  if anisotropy < 0.0:
    direction = tangent
  across = wp.cross(direction, view)
  bent = wp.cross(across, direction)
  bend = wp.abs(anisotropy) * wp.clamp(5.0 * perceptual_roughness, 0.0, 1.0)
  mixed = normal * (1.0 - bend) + bent * bend
  if wp.length_sq(mixed) <= 1.0e-12:
    return normal
  return wp.normalize(mixed)


@wp.func
def d_charlie(
  # In:
  alpha: wp.float32,
  noh: wp.float32,
) -> wp.float32:
  """Charlie sheen distribution (Estevez & Kulla 2017; Filament `D_Charlie`) of roughness α."""
  inv_alpha = 1.0 / alpha
  sin2h = wp.max(1.0 - noh * noh, wp.static(_CHARLIE_MIN_SIN2))
  return (2.0 + inv_alpha) * wp.pow(sin2h, inv_alpha * 0.5) * wp.static(1.0 / (2.0 * wp.pi))


@wp.func
def v_neubelt(
  # In:
  nov: wp.float32,
  nol: wp.float32,
) -> wp.float32:
  """Neubelt & Pettineo 2013 cloth visibility, including 1 / (4 NoV NoL) (Filament `V_Neubelt`)."""
  return 1.0 / (4.0 * (nol + nov - nol * nov))


@wp.func
def wrap_diffuse(
  # In:
  nol: wp.float32,
  wrap: wp.float32,
) -> wp.float32:
  """Energy-conserving wrapped cosine saturate((NoL + w) / (1 + w)²) (Filament `Fd_Wrap`): the
  cloth model's cheap thin-surface scattering, light reaching up to acos(-w) past the terminator."""
  return wp.clamp((nol + wrap) / ((1.0 + wrap) * (1.0 + wrap)), 0.0, 1.0)


@wp.func
def dielectric_f0(
  # In:
  ior: wp.float32,
) -> wp.float32:
  """Normal-incidence Fresnel reflectance ((n - 1) / (n + 1))² of a dielectric of index `ior`."""
  r = (ior - 1.0) / (ior + 1.0)
  return r * r


@wp.func
def fresnel_average(
  # In:
  ior: wp.float32,
) -> wp.float32:
  """Hemispherical cosine-weighted average of the dielectric Fresnel reflectance E_F(η) (d'Eon &
  Irving 2011 fit, as in OpenPBR's reference implementation): the internal reflectance input of
  the coat darkening K_r = 1 - (1 - E_F(η)) / η²."""
  eta = wp.max(ior, 1.0e-3)
  if eta >= 1.0:
    return (eta - 1.0) / (4.08567 + 1.00071 * eta)
  return 0.997118 + 0.1014 * eta - 0.965241 * eta * eta - 0.130607 * eta * eta * eta
