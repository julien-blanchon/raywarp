"""The layered lobes' terms against independent closed forms and quadrature (Filament V3/V5/V6):
anisotropic GGX (Heitz 2018's general form, its isotropic limit, the VNDF sampler's weights),
the Charlie sheen and its cloth table, the coat's transmittance, reoriented normal mapping."""

import math

import numpy as np
import pytest
import warp as wp

from raywarp.core.shade.environment import ibl
from raywarp.core.shade.materials import brdf, material, uv_maps

wp.set_module_options({"enable_backward": False})


# The anisotropic GGX visible-normal sampler (spherical caps, Dupuy & Benyoub 2023) and Smith's
# G1: the importance-sampled estimator of the lobe's albedo.
@wp.func
def sample_vndf_anisotropic(
  # In:
  view: wp.vec3,
  normal: wp.vec3,
  tangent: wp.vec3,
  bitangent: wp.vec3,
  alpha_t: float,
  alpha_b: float,
  u: wp.vec2,
) -> wp.vec3:
  """Microfacet normal from the anisotropic GGX distribution of visible normals (spherical caps,
  Dupuy & Benyoub 2023; Heitz 2018's stretch by (alpha_t, alpha_b) in the lobe's tangent frame).
  Equals `sample_vndf` (up to the frame's azimuth) at alpha_t = alpha_b."""
  v = wp.vec3(wp.dot(view, tangent), wp.dot(view, bitangent), wp.max(wp.dot(view, normal), 1.0e-4))
  vh = wp.normalize(wp.vec3(alpha_t * v[0], alpha_b * v[1], v[2]))
  phi = wp.static(2.0 * wp.pi) * u[0]
  z = (1.0 - u[1]) * (1.0 + vh[2]) - vh[2]
  sin_theta = wp.sqrt(wp.clamp(1.0 - z * z, 0.0, 1.0))
  c = wp.vec3(sin_theta * wp.cos(phi), sin_theta * wp.sin(phi), z) + vh
  m = wp.normalize(wp.vec3(alpha_t * c[0], alpha_b * c[1], wp.max(c[2], 0.0)))
  return tangent * m[0] + bitangent * m[1] + normal * m[2]


@wp.func
def smith_g1_anisotropic(
  # In:
  tov: float,
  bov: float,
  nov: float,
  alpha_t: float,
  alpha_b: float,
) -> float:
  """Smith's anisotropic GGX masking G1 = 2 NoV / (NoV + |(alpha_t ToV, alpha_b BoV, NoV)|)."""
  return 2.0 * nov / (nov + wp.length(wp.vec3(alpha_t * tov, alpha_b * bov, nov)))


@wp.kernel
def _aniso_terms(
  # In:
  h: wp.array[wp.vec3],
  v: wp.array[wp.vec3],
  l: wp.array[wp.vec3],
  ab: wp.array[wp.vec2],
  # Out:
  out: wp.array[wp.vec4],
):
  i = wp.tid()
  hh, vv, ll = h[i], v[i], l[i]
  at, bt = ab[i][0], ab[i][1]
  d = brdf.d_ggx_anisotropic(at, bt, hh[0], hh[1], hh[2])
  vis = brdf.v_smith_ggx_correlated_anisotropic(at, bt, vv[0], vv[1], ll[0], ll[1], vv[2], ll[2])
  di = brdf.d_ggx(hh[2], at)
  vi = brdf.v_smith_ggx_correlated(vv[2], ll[2], at)
  out[i] = wp.vec4(d, vis, di, vi)


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
  x = rng.normal(size=(n, 3))
  x[:, 2] = np.abs(x[:, 2]) + 0.05
  return x / np.linalg.norm(x, axis=1, keepdims=True)


def _heitz_d(h: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
  """Heitz 2014 eq. 85: 1 / (pi ax ay (x^2/ax^2 + y^2/ay^2 + z^2)^2)."""
  s = (h[:, 0] / ax) ** 2 + (h[:, 1] / ay) ** 2 + h[:, 2] ** 2
  return 1.0 / (np.pi * ax * ay * s * s)


def _heitz_lambda(w: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
  """Smith Lambda of the anisotropic GGX (Heitz 2014 eq. 86)."""
  a2t = ((w[:, 0] * ax) ** 2 + (w[:, 1] * ay) ** 2) / w[:, 2] ** 2
  return 0.5 * (-1.0 + np.sqrt(1.0 + a2t))


def test_anisotropic_ggx_matches_heitz_and_isotropic_limit():
  rng = np.random.default_rng(1)
  n = 512
  v, l = _unit(rng, n), _unit(rng, n)
  h = (v + l) / np.linalg.norm(v + l, axis=1, keepdims=True)
  ab = np.column_stack((rng.uniform(0.05, 1.0, n), rng.uniform(0.05, 1.0, n))).astype(np.float32)
  ab[: n // 4, 1] = ab[: n // 4, 0]  # isotropic rows
  out = wp.zeros(n, dtype=wp.vec4, device="cpu")
  arrays = [
    wp.array(x.astype(np.float32), dtype=t, device="cpu") for x, t in ((h, wp.vec3), (v, wp.vec3), (l, wp.vec3), (ab, wp.vec2))
  ]
  wp.launch(_aniso_terms, dim=n, inputs=arrays, outputs=[out], device="cpu")
  d, vis, di, vi = np.transpose(out.numpy())
  ax, ay = ab[:, 0].astype(np.float64), ab[:, 1].astype(np.float64)
  np.testing.assert_allclose(d, _heitz_d(h, ax, ay), rtol=2e-4)
  g2 = 1.0 / (1.0 + _heitz_lambda(v, ax, ay) + _heitz_lambda(l, ax, ay))
  np.testing.assert_allclose(vis, g2 / (4.0 * v[:, 2] * l[:, 2]), rtol=2e-4)
  iso = slice(0, n // 4)
  np.testing.assert_allclose(d[iso], di[iso], rtol=2e-4)
  np.testing.assert_allclose(vis[iso], vi[iso], rtol=2e-4)


def test_openpbr_anisotropic_alpha_keeps_mean_roughness():
  @wp.kernel
  def k(a: wp.array[float], out: wp.array[wp.vec2]):
    i = wp.tid()
    out[i] = brdf.anisotropic_alpha(0.3, a[i])

  a = np.array([0.0, 0.25, 0.5, 0.9, 1.0], np.float32)
  out = wp.zeros(len(a), dtype=wp.vec2, device="cpu")
  wp.launch(k, dim=len(a), inputs=[wp.array(a, dtype=float, device="cpu")], outputs=[out], device="cpu")
  at, ab = np.transpose(out.numpy())
  np.testing.assert_allclose(at[:-1] ** 2 + ab[:-1] ** 2, 2 * 0.09, rtol=1e-5)  # OpenPBR 1.1 §Specular
  np.testing.assert_allclose(ab / at, np.maximum(1 - a, brdf.MIN_ANISOTROPIC_ALPHA / at), rtol=1e-5)
  assert at[0] == pytest.approx(0.3) and ab[0] == pytest.approx(0.3)


@wp.kernel
def _vndf_weight(
  # In:
  view: wp.vec3,
  at: float,
  ab: float,
  n: int,
  # Out:
  out: wp.array[float],
):
  i = wp.tid()
  u = ibl.hammersley(i, 1.0 / float(n))
  u = wp.vec2(u[0] + 0.5 * (1.0 / float(n)), u[1])
  t = wp.vec3(1.0, 0.0, 0.0)
  b = wp.vec3(0.0, 1.0, 0.0)
  nn = wp.vec3(0.0, 0.0, 1.0)
  h = sample_vndf_anisotropic(view, nn, t, b, at, ab, u)
  d = h * (2.0 * wp.dot(view, h)) - view
  w = float(0.0)
  if d[2] > 0.0:
    v2 = brdf.v_smith_ggx_correlated_anisotropic(at, ab, view[0], view[1], d[0], d[1], view[2], d[2])
    w = 4.0 * view[2] * d[2] * v2 / smith_g1_anisotropic(view[0], view[1], view[2], at, ab)
  out[i] = w


def _albedo_quadrature(view: np.ndarray, at: float, ab: float, m: int = 1200) -> float:
  """Single-scattering albedo of the white anisotropic GGX: int D G2 / (4 NoV) over l (midpoint
  rule over (cos theta, phi))."""
  mu = (np.arange(m) + 0.5) / m
  phi = (np.arange(2 * m) + 0.5) / (2 * m) * 2 * np.pi
  mu_g, phi_g = np.meshgrid(mu, phi, indexing="ij")
  s = np.sqrt(1 - mu_g**2)
  l = np.stack((s * np.cos(phi_g), s * np.sin(phi_g), mu_g), -1).reshape(-1, 3)
  h = l + view
  h /= np.linalg.norm(h, axis=1, keepdims=True)
  ax, ay = np.full(len(l), at), np.full(len(l), ab)
  g2 = 1.0 / (1.0 + _heitz_lambda(np.broadcast_to(view, l.shape), ax, ay) + _heitz_lambda(l, ax, ay))
  f = _heitz_d(h, ax, ay) * g2 / (4.0 * view[2])  # f cos(l) / cos(l) cancelled: f = D G2 / (4 NoV NoL)
  return float(np.sum(f) * (1.0 / m) * (2 * np.pi / (2 * m)))


@pytest.mark.parametrize(("at", "ab"), [(0.6, 0.1), (0.2, 0.5)])
def test_anisotropic_vndf_weights_integrate_the_albedo(at: float, ab: float):
  view = np.array([0.5, 0.3, 0.0])
  view[2] = math.sqrt(1 - view[0] ** 2 - view[1] ** 2)
  n = 1 << 14
  out = wp.zeros(n, dtype=float, device="cpu")
  wp.launch(_vndf_weight, dim=n, inputs=[wp.vec3(*view), at, ab, n], outputs=[out], device="cpu")
  assert float(out.numpy().mean()) == pytest.approx(_albedo_quadrature(view, at, ab), rel=5e-3)


def test_charlie_is_normalised_and_cloth_table_matches_quadrature():
  @wp.kernel
  def k(alpha: float, noh: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = brdf.d_charlie(alpha, noh[i])

  m = 20000
  mu = (np.arange(m) + 0.5) / m
  out = wp.zeros(m, dtype=float, device="cpu")
  for alpha in (0.1, 0.4, 1.0):
    wp.launch(k, dim=m, inputs=[alpha, wp.array(mu.astype(np.float32), dtype=float, device="cpu")], outputs=[out], device="cpu")
    # int D (n.h) dw_h = 2 pi int D mu dmu (the sin^2 floor adds < 1e-2 at small alpha).
    assert float(np.sum(out.numpy() * mu) * 2 * np.pi / m) == pytest.approx(1.0, abs=1.5e-2)
  table = ibl.cloth_table()
  size = table.shape[0]
  for j, i in ((40, 100), (90, 20), (127, 127)):
    alpha = ((j + 0.5) / size) ** 2
    nov = (i + 0.5) / size
    view = np.array([math.sqrt(1 - nov * nov), 0.0, nov])
    q = 600
    mu_g, phi_g = np.meshgrid((np.arange(q) + 0.5) / q, (np.arange(2 * q) + 0.5) / (2 * q) * 2 * np.pi, indexing="ij")
    s = np.sqrt(1 - mu_g**2)
    l = np.stack((s * np.cos(phi_g), s * np.sin(phi_g), mu_g), -1).reshape(-1, 3)
    h = (l + view) / np.linalg.norm(l + view, axis=1, keepdims=True)
    sin2 = np.maximum(1 - h[:, 2] ** 2, 0.0078125)
    d = (2 + 1 / alpha) * sin2 ** (0.5 / alpha) / (2 * np.pi)
    vis = 1 / (4 * (l[:, 2] + nov - l[:, 2] * nov))
    e = float(np.sum(d * vis * l[:, 2]) / q * (2 * np.pi / (2 * q)))
    assert float(table[j, i]) == pytest.approx(e, rel=2e-2, abs=2e-3)


def test_coat_transmittance_and_fresnel_average():
  @wp.kernel
  def k(out: wp.array[wp.vec4]):
    t0 = material.coat_transmittance(wp.vec3(0.25, 1.0, 0.0), 1.5, 1.0, 1.0)
    t1 = material.coat_transmittance(wp.vec3(0.25, 1.0, 0.0), 1.5, 0.3, 0.3)
    out[0] = wp.vec4(t0[0], t0[1], t1[0], brdf.fresnel_average(1.5))

  out = wp.zeros(1, dtype=wp.vec4, device="cpu")
  wp.launch(k, dim=1, outputs=[out], device="cpu")
  t0, white, t1, favg = out.numpy()[0]
  assert t0 == pytest.approx(0.25, rel=1e-5) and white == pytest.approx(1.0)  # coat_color = T^2 at normal incidence
  mu_t = math.sqrt(1 - (1 - 0.09) / 2.25)
  assert t1 == pytest.approx(0.5 ** (2 / mu_t), rel=1e-4)  # T^(1/mu_i' + 1/mu_o')
  # Hemispherical average of the exact dielectric Fresnel at eta 1.5: 2 int F mu dmu.
  mu = (np.arange(20000) + 0.5) / 20000
  st = np.sqrt(1 - mu**2) / 1.5
  ct = np.sqrt(1 - st**2)
  rs = ((mu - 1.5 * ct) / (mu + 1.5 * ct)) ** 2
  rp = ((ct - 1.5 * mu) / (ct + 1.5 * mu)) ** 2
  # The fit (Kulla & Conty 2017) is within 3 % of the exact integral (0.0895 vs 0.0918).
  assert favg == pytest.approx(float(np.sum(0.5 * (rs + rp) * mu) * 2 / 20000), rel=3e-2)


def test_reoriented_normal_mapping():
  @wp.kernel
  def k(base: wp.array[wp.vec3], detail: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = uv_maps.reoriented_normal(base[i], detail[i])

  rng = np.random.default_rng(3)
  base = _unit(rng, 64)
  flat = np.tile([0.0, 0.0, 1.0], (64, 1))
  detail = _unit(rng, 64)
  for b, d, expect in ((base, flat, base), (flat, detail, detail)):
    out = wp.zeros(64, dtype=wp.vec3, device="cpu")
    args = [wp.array(x.astype(np.float32), dtype=wp.vec3, device="cpu") for x in (b, d)]
    wp.launch(k, dim=64, inputs=args, outputs=[out], device="cpu")
    np.testing.assert_allclose(out.numpy(), expect, atol=2e-5)
  # A detail normal turns with the base: its angle to the base equals its angle to +z.
  out = wp.zeros(64, dtype=wp.vec3, device="cpu")
  args = [wp.array(x.astype(np.float32), dtype=wp.vec3, device="cpu") for x in (base, detail)]
  wp.launch(k, dim=64, inputs=args, outputs=[out], device="cpu")
  r = out.numpy() / np.linalg.norm(out.numpy(), axis=1, keepdims=True)
  np.testing.assert_allclose(np.einsum("ij,ij->i", r, base), detail[:, 2], atol=2e-4)
