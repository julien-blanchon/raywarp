"""Checks fitted LTC tables: the LTC integral of a polygonal light against Monte Carlo.

  python tools/check_ltc.py ltc.npz

For several roughnesses / view angles / quads, compares
  LTC:  norm * E(M^-1 polygon)   (E: clipped polygon irradiance integral, Heitz 2016)
  MC :  integral of BRDF * cos (Fresnel 1) over the polygon's solid angle (uniform sphere samples)
and prints the relative errors. The table lookup is bilinear, as the GPU texture will be.
"""

import sys

import fit_ltc as F
import numpy as np


def lookup(table: np.ndarray, cos_v: float, rough: float) -> np.ndarray:
  n = table.shape[0]
  x = np.sqrt(max(1.0 - cos_v, 0.0)) * (n - 1)
  y = rough * (n - 1)
  i, j = min(int(x), n - 2), min(int(y), n - 2)
  fx, fy = x - i, y - j
  return (
    table[i, j] * (1 - fx) * (1 - fy)
    + table[i + 1, j] * fx * (1 - fy)
    + table[i, j + 1] * (1 - fx) * fy
    + table[i + 1, j + 1] * fx * fy
  )


def clip_horizon(poly: np.ndarray) -> np.ndarray:
  """Sutherland-Hodgman clip of a polygon (k, 3) to z >= 0."""
  out = []
  for k in range(len(poly)):
    a, b = poly[k], poly[(k + 1) % len(poly)]
    if a[2] >= 0:
      out.append(a)
    if (a[2] >= 0) != (b[2] >= 0):
      t = a[2] / (a[2] - b[2])
      out.append(a + t * (b - a))
  return np.array(out)


def polygon_integral(poly: np.ndarray) -> float:
  """Irradiance-normalised form factor of a polygon (directions from the origin), clipped."""
  p = clip_horizon(poly)
  if len(p) < 3:
    return 0.0
  p = p / np.linalg.norm(p, axis=1, keepdims=True)
  s = 0.0
  for k in range(len(p)):
    a, b = p[k], p[(k + 1) % len(p)]
    c = np.cross(a, b)
    ln = np.linalg.norm(c)
    if ln < 1e-12:
      continue
    theta = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    s += theta * c[2] / ln
  return abs(s) / (2.0 * np.pi)


def mc_integral(v: np.ndarray, alpha: float, quad: np.ndarray, n: int = 2_000_000, seed: int = 0) -> float:
  """Integral of BRDF * cos over the quad, importance-sampling GGX visible normals."""
  rng = np.random.default_rng(seed)
  d = F.sample_ggx(v, alpha, rng.random((n, 2)))
  pdf = F.ggx_pdf(v, d, alpha)
  # Ray (origin) vs the quad's plane, then inside test with the two edge vectors.
  o, e1, e2 = quad[0], quad[1] - quad[0], quad[3] - quad[0]
  nrm = np.cross(e1, e2)
  denom = d @ nrm
  t = np.where(np.abs(denom) > 1e-12, (o @ nrm) / np.where(np.abs(denom) > 1e-12, denom, 1.0), -1.0)
  hit = d * t[:, None] - o
  a = hit @ e1 / (e1 @ e1)
  b = hit @ e2 / (e2 @ e2)
  inside = (t > 0) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
  f, _ = F.ggx_cos(v, d, alpha)
  ok = inside & (pdf > 0) & (d[:, 2] > 0)
  return float(np.mean(np.where(ok, f / np.maximum(pdf, 1e-30), 0.0)))


def main():
  t = np.load(sys.argv[1])
  minv_t, mag_t = t["minv"], t["mag"]
  rng = np.random.default_rng(3)
  errs = []
  for rough in (0.1, 0.3, 0.5, 0.8, 1.0):
    alpha = max(rough * rough, F.MIN_ALPHA)
    for cos_v in (0.95, 0.7, 0.4, 0.15):
      v = np.array([np.sqrt(1 - cos_v * cos_v), 0.0, cos_v])
      m00, m02, m11, m20 = lookup(minv_t, cos_v, rough)
      minv = np.array([[m00, 0, m02], [0, m11, 0], [m20, 0, 1]])
      norm = lookup(mag_t, cos_v, rough)[0]
      for _ in range(3):
        c = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0.5, 2.0)])
        ex = rng.normal(size=3)
        ex /= np.linalg.norm(ex)
        ey = np.cross(ex, rng.normal(size=3))
        ey /= np.linalg.norm(ey)
        w, h = rng.uniform(0.3, 1.5), rng.uniform(0.3, 1.5)
        quad = np.array([c - ex * w - ey * h, c + ex * w - ey * h, c + ex * w + ey * h, c - ex * w + ey * h])
        ref = mc_integral(v, alpha, quad)
        ltc = norm * polygon_integral(quad @ minv.T)
        if ref > 1e-3:
          errs.append(abs(ltc - ref) / ref)
          print(f"rough {rough:.1f} cos_v {cos_v:.2f}: MC {ref:.4f} LTC {ltc:.4f} rel err {errs[-1]:.3f}")
  e = np.array(errs)
  print(f"median rel err {np.median(e):.3f}, 90% {np.percentile(e, 90):.3f}, max {e.max():.3f} ({len(e)} cases)")


if __name__ == "__main__":
  main()
