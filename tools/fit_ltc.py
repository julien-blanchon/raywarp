"""Fits linearly transformed cosine (LTC) tables to the renderer's GGX BRDF.

  python tools/fit_ltc.py --out ltc.npz [--size 64] [--samples 4096]

Heitz, Dupuy, Hill, Neubelt 2016, "Real-Time Polygonal-Light Shading with Linearly Transformed
Cosines" (the fitting of selfshadow/ltc_code, re-implemented for our lobe): isotropic GGX with
height-correlated Smith visibility (brdf.d_ggx / v_smith_ggx_correlated), Fresnel = 1.

Table axes: u = sqrt(1 - cos(theta_v)) (Heitz's parametrisation), v = perceptual roughness
(alpha = v^2, as brdf.roughness_to_alpha). For each cell, the LTC matrix in the frame where the
view direction lies in the xz plane is M = [[a, 0, b], [0, c, 0], [d, 0, 1]]; the table stores
M^-1 normalised by its (2, 2) entry as (m00, m02, m11, m20) plus
  norm: hemispherical integral of BRDF * cos (Fresnel = 1),
  fresnel: the part of it weighted by (1 - VoH)^5 (Schlick: F = F0 * norm + (1 - F0) * fresnel).
The error minimised is the L3 distance between the normalised BRDF * cos and the LTC
distribution, by importance sampling both (multiple importance sampling, balance heuristic).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize

MIN_ALPHA = 0.045**2  # brdf.MIN_PERCEPTUAL_ROUGHNESS squared


def ggx_cos(v: np.ndarray, l: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
  """BRDF * cos(theta_l) (Fresnel 1) and the (1 - VoH)^5-weighted version, for unit l (n = +z)."""
  nol = l[:, 2]
  nov = v[2]
  h = v[None, :] + l
  h /= np.linalg.norm(h, axis=1, keepdims=True)
  noh = np.clip(h[:, 2], 0.0, 1.0)
  voh = np.clip(h @ v, 0.0, 1.0)
  a2 = alpha * alpha
  f = (noh * a2 - noh) * noh + 1.0
  d = a2 / (np.pi * f * f)
  gv = nol * np.sqrt(nov * nov * (1.0 - a2) + a2)
  gl = nov * np.sqrt(nol * nol * (1.0 - a2) + a2)
  vis = 0.5 / np.maximum(gv + gl, 1e-12)
  val = np.where(nol > 0.0, d * vis * nol, 0.0)
  return val, val * (1.0 - voh) ** 5


def ggx_pdf(v: np.ndarray, l: np.ndarray, alpha: float) -> np.ndarray:
  """pdf of `sample_ggx` (visible normals, Heitz 2018) at l."""
  nol = l[:, 2]
  nov = max(v[2], 1e-6)
  h = v[None, :] + l
  h /= np.linalg.norm(h, axis=1, keepdims=True)
  noh = np.clip(h[:, 2], 0.0, 1.0)
  voh = np.clip(h @ v, 0.0, 1.0)
  a2 = alpha * alpha
  f = (noh * a2 - noh) * noh + 1.0
  d = a2 / (np.pi * f * f)
  g1 = 2.0 * nov / (nov + np.sqrt(a2 + (1.0 - a2) * nov * nov))
  pdf_h = g1 * voh * d / nov
  return np.where(nol > 0.0, pdf_h / np.maximum(4.0 * voh, 1e-12), 0.0)


def sample_ggx(v: np.ndarray, alpha: float, u: np.ndarray) -> np.ndarray:
  """Directions l from GGX visible normals for view v (Heitz 2018)."""
  vh = np.array([alpha * v[0], alpha * v[1], v[2]])
  vh /= np.linalg.norm(vh)
  lensq = vh[0] ** 2 + vh[1] ** 2
  t1 = np.array([-vh[1], vh[0], 0.0]) / np.sqrt(lensq) if lensq > 0 else np.array([1.0, 0.0, 0.0])
  t2 = np.cross(vh, t1)
  r = np.sqrt(u[:, 0])
  phi = 2.0 * np.pi * u[:, 1]
  p1 = r * np.cos(phi)
  p2 = r * np.sin(phi)
  s = 0.5 * (1.0 + vh[2])
  p2 = (1.0 - s) * np.sqrt(np.maximum(1.0 - p1 * p1, 0.0)) + s * p2
  nh = p1[:, None] * t1 + p2[:, None] * t2 + np.sqrt(np.maximum(1.0 - p1 * p1 - p2 * p2, 0.0))[:, None] * vh
  h = np.stack([alpha * nh[:, 0], alpha * nh[:, 1], np.maximum(nh[:, 2], 0.0)], axis=1)
  h /= np.linalg.norm(h, axis=1, keepdims=True)
  return 2.0 * (h @ v)[:, None] * h - v[None, :]


def cosine_dirs(u: np.ndarray) -> np.ndarray:
  r = np.sqrt(u[:, 0])
  phi = 2.0 * np.pi * u[:, 1]
  return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(1.0 - u[:, 0], 0.0))], axis=1)


def ltc_matrix(p: np.ndarray, frame: np.ndarray) -> np.ndarray:
  """M = frame @ [[a, 0, b], [0, c, 0], [d, 0, 1]] (frame: columns = average-direction basis)."""
  a, b, c, d = p
  return frame @ np.array([[a, 0.0, b], [0.0, c, 0.0], [d, 0.0, 1.0]])


def ltc_eval(minv: np.ndarray, det_minv: float, l: np.ndarray) -> np.ndarray:
  """D(l) = D_o(M^-1 l / |M^-1 l|) * |M^-1| / |M^-1 l|^3 (D_o: clamped cosine / pi)."""
  lo = l @ minv.T
  length = np.linalg.norm(lo, axis=1)
  lo = lo / length[:, None]
  return np.maximum(lo[:, 2], 0.0) / np.pi * abs(det_minv) / length**3


def fit_cell(v: np.ndarray, alpha: float, u: np.ndarray, start: np.ndarray, frame: np.ndarray) -> tuple:
  l_brdf = sample_ggx(v, alpha, u)
  l_brdf = l_brdf[l_brdf[:, 2] > 0.0]
  pdf_b = ggx_pdf(v, l_brdf, alpha)
  f_b, fres_b = ggx_cos(v, l_brdf, alpha)
  norm = float(np.mean(np.where(pdf_b > 0, f_b / np.maximum(pdf_b, 1e-30), 0.0))) * len(l_brdf) / len(u)
  fresnel = float(np.mean(np.where(pdf_b > 0, fres_b / np.maximum(pdf_b, 1e-30), 0.0))) * len(l_brdf) / len(u)
  norm = max(norm, 1e-8)
  cos_dirs = cosine_dirs(u)

  def error(p: np.ndarray) -> float:
    if p[0] <= 1e-5 or p[2] <= 1e-5:
      return 1e10
    m = ltc_matrix(p, frame)
    det = np.linalg.det(m)
    if abs(det) < 1e-12:
      return 1e10
    minv = np.linalg.inv(m)
    det_minv = 1.0 / det
    # Samples from the LTC: M @ cosine directions.
    l_ltc = cos_dirs @ m.T
    l_ltc /= np.linalg.norm(l_ltc, axis=1, keepdims=True)
    err = 0.0
    for ls in (l_brdf, l_ltc):
      p_ltc = ltc_eval(minv, det_minv, ls)
      f, _ = ggx_cos(v, ls, alpha)
      p_brdf = ggx_pdf(v, ls, alpha)
      target = f / norm
      w = p_ltc + p_brdf
      err += float(np.mean(np.where(w > 0, np.abs(p_ltc - target) ** 3 / np.maximum(w, 1e-30), 0.0)))
    return err

  res = minimize(error, start, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 400})
  return res.x, norm, fresnel


def average_direction(v: np.ndarray, alpha: float, u: np.ndarray) -> np.ndarray:
  l = sample_ggx(v, alpha, u)
  l = l[l[:, 2] > 0.0]
  f, _ = ggx_cos(v, l, alpha)
  pdf = ggx_pdf(v, l, alpha)
  w = np.where(pdf > 0, f / np.maximum(pdf, 1e-30), 0.0)
  avg = (l * w[:, None]).sum(0)
  avg[1] = 0.0  # isotropic: in the xz plane
  return avg / max(np.linalg.norm(avg), 1e-12)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--out", required=True)
  p.add_argument("--size", type=int, default=64)
  p.add_argument("--samples", type=int, default=4096)
  args = p.parse_args()
  n = args.size
  rng = np.random.default_rng(0)
  u = rng.random((args.samples, 2))
  minv_table = np.zeros((n, n, 4), np.float32)  # (m00, m02, m11, m20) of M^-1 / M^-1[2, 2]
  mag_table = np.zeros((n, n, 2), np.float32)  # (norm, fresnel)
  t0 = time.perf_counter()
  # Fit from smooth to rough... Heitz fits from rough (where M ~ identity) towards smooth.
  prev_first = np.array([1.0, 0.0, 1.0, 0.0])  # each roughness row starts from the previous row's first fit
  for a_idx in range(n - 1, -1, -1):
    rough = a_idx / (n - 1)
    alpha = max(rough * rough, MIN_ALPHA)
    start = np.array([1.0, 0.0, 1.0, 0.0])
    for t_idx in range(n):
      x = t_idx / (n - 1)
      cos_v = max(1.0 - x * x, 1e-4)  # u = sqrt(1 - cos theta)
      v = np.array([np.sqrt(1.0 - cos_v * cos_v), 0.0, cos_v])
      avg = average_direction(v, alpha, u)
      # Frame: x' = perpendicular to avg in the xz plane, z' = avg.
      z = avg
      xv = np.array([z[2], 0.0, -z[0]])
      frame = np.stack([xv, np.array([0.0, 1.0, 0.0]), z], axis=1)
      if t_idx == 0:
        start = prev_first
      params, norm, fresnel = fit_cell(v, alpha, u, start, frame)
      if t_idx == 0:
        prev_first = params.copy()
      start = params
      minv = np.linalg.inv(ltc_matrix(params, frame))
      minv = minv / minv[2, 2]
      entry = np.array([minv[0, 0], minv[0, 2], minv[1, 1], minv[2, 0]], np.float32)
      if entry[2] <= 0.0 and t_idx > 0:
        # The most grazing views of smooth lobes (cos theta_v = 1e-4: the lobe a sliver at the
        # horizon) fit flipped matrices (M^-1[2, 2] < 0: the normalisation turns the hemisphere
        # over): keep the previous view angle's lobe there.
        entry = minv_table[t_idx - 1, a_idx]
      minv_table[t_idx, a_idx] = entry
      mag_table[t_idx, a_idx] = [norm, fresnel]
    print(f"roughness {rough:.3f} done ({time.perf_counter() - t0:.0f} s)", flush=True)
  np.savez(args.out, minv=minv_table, mag=mag_table)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
