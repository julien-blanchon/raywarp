"""The OpenPBR rough diffuse lobe (EON), separate from its lighting integrators.

EON follows Portsmouth, Kutz and Hill, JCGT 14(1), 2025 (2026 errata), and Adobe openpbr-bsdf
c91aad1d, impl/openpbr_diffuse_lobe.h (Apache-2.0). It is evaluated from cosines only (no tangent
frame): `eon_split` once per shading point (the view's factors), `eon_eval_split` per light, and
`eon_albedo` for the integrated lights. `rho` is the underlying diffuse single-scattering color,
not the specular-coupled one; layer weights apply after evaluation.
"""

from __future__ import annotations

from typing import Final

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_FON_A: Final = 0.5 - 2.0 / (3.0 * wp.pi)
_FON_B: Final = 2.0 / 3.0 - 28.0 / (15.0 * wp.pi)


@wp.func
def _fon_loss_factor(
  # In:
  mu: float,
) -> float:
  """(1 - E_FON(mu)) / (roughness * A), finite even at grazing incidence."""
  c = wp.clamp(mu, 0.0, 1.0)
  s = wp.sqrt(wp.max(1.0 - c * c, 0.0))
  g = s * (wp.acos(c) - s * c) + (2.0 / 3.0) * (s * c * (1.0 + s + s * s) / (1.0 + s) - s)
  return wp.max(wp.static(_FON_A) - g / wp.pi, 0.0)


@wp.func
def _eon_multiscatter(
  # In:
  rho: wp.vec3,
  roughness: float,
) -> wp.vec3:
  """Multiple-scattering color before its angular and normalization factors."""
  average_loss = roughness * wp.static(_FON_A - _FON_B) / (1.0 + wp.static(_FON_A) * roughness)
  return wp.cw_div(wp.cw_mul(rho, rho) * (1.0 - average_loss), wp.vec3(1.0) - rho * average_loss)


@wp.func
def eon_split(
  # In:
  rho: wp.vec3,
  roughness: float,
  mu_o: float,
) -> tuple[wp.vec3, wp.vec3]:
  """The factors of `eon_eval_split` that depend on the view only (roughness > 0): (single, multi)
  with EON(mu_i, wi.wo) = single (1 + roughness st) + multi (1 - E_FON(mu_i)) / (roughness A). A
  shading point computes them once (`material.make_shading_lobes`), each light the rest."""
  a = 1.0 / (1.0 + wp.static(_FON_A) * roughness)
  single = rho * (a / wp.pi)
  multi = _eon_multiscatter(rho, roughness) * (roughness * a * _fon_loss_factor(mu_o) / wp.static((_FON_A - _FON_B) * wp.pi))
  return single, multi


@wp.func
def eon_eval_split(
  # In:
  single: wp.vec3,
  multi: wp.vec3,
  roughness: float,
  mu_i: float,
  mu_o: float,
  wi_dot_wo: float,
) -> wp.vec3:
  """EON BRDF from its view factors (`eon_split`) and the invariant cosines, avoiding a tangent
  frame per direct light."""
  if mu_i <= 0.0 or mu_o <= 0.0:
    return wp.vec3(0.0)
  s = wi_dot_wo - mu_i * mu_o
  st = s
  if s > 0.0:
    st = s / wp.max(mu_i, mu_o)
  return single * (1.0 + roughness * st) + multi * _fon_loss_factor(mu_i)


@wp.func
def eon_albedo(
  # In:
  rho: wp.vec3,
  roughness: float,
  mu: float,
) -> wp.vec3:
  """Exact directional hemispherical reflectance under uniform incident radiance: the albedo
  every integrated light scales (`material.make_shading_lobes`; exact in a white furnace, the
  analytic albedo ratio over Lambert's integral for a varying environment or a finite light)."""
  if roughness == 0.0:
    return rho
  loss = roughness * _fon_loss_factor(mu) / (1.0 + wp.static(_FON_A) * roughness)
  return rho * (1.0 - loss) + _eon_multiscatter(rho, roughness) * loss
