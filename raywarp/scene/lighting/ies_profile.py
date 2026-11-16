"""An IES profile on a regular angular grid (`IesGrid`, what the GPU stores), its flux, and the
lookup the shading does (`sample`, the numpy oracle of `core/shade/lights/ies.py`).

Light frame (UsdLux: the light emits along local -Z): a direction d = (x, y, z) in the light's
local frame. The profile's symmetries are expanded on the host, so the grid always covers the
profile's whole domain; directions outside the listed vertical (and, for types A / B, horizontal)
range get 0, as LM-63 specifies.

* Type C: vertical angle theta from the nadir (local -Z), horizontal angle phi = atan2(y, x) (0
  along local +X, 90 along +Y: counter-clockwise seen from above). One horizontal angle: axially
  symmetric, a 1D profile (no azimuth lookup); 0-90 (quadrant), 0-180 / 90-270 (bilateral)
  are mirrored to 0-360, a periodic grid.
* Types A and B: the photometric axis (H = V = 0) is the emission axis, local -Z; the luminaire's
  up is local +Y, its lateral axis local +X. Type A's polar axis is up: V = asin(y), H =
  atan2(x, -z). Type B's polar axis is lateral: V = asin(x), H = atan2(y, -z). Ranges starting at
  0 are mirrored (V -> -V, H -> -H); the back hemisphere (|H| > 90) is dark.

The grid's spacing is the angles' common step, so knots stay knots and the bilinear lookup equals
the file's piecewise-linear interpolation exactly; profiles without a common step (or one too
fine for `MAX_SAMPLES`) are resampled at their smallest step (`IesGrid.exact` False).

`flux` integrates the gridded interpolant exactly (bilinear cells times the solid-angle element).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Final

import numpy as np

from raywarp.scene.model.ies import ANGLE_TOL, IesData, PhotometricType, parse
from raywarp.scene.model.types import Column, Rows

# Most samples along one axis of a grid (0.1 degree over 180 degrees).
MAX_SAMPLES: Final = 1801
# Grid kinds (`IesGrid.kind`, `core/shade/lights/ies.py`).
KIND_C: Final = 0
KIND_A: Final = 1
KIND_B: Final = 2
_KINDS: Final = {PhotometricType.C: KIND_C, PhotometricType.A: KIND_A, PhotometricType.B: KIND_B}
_DEG: Final = math.pi / 180.0

type Dirs = np.ndarray[[int, 3], np.dtype[np.float64]]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class IesGrid:
  """A profile on a regular (vertical a, horizontal b) grid, relative to its peak.

  Attributes:
    kind: KIND_C, KIND_A or KIND_B (see the module doc for the angles).
    a0, da: first vertical angle and step [rad] (na > 1 samples).
    b0, db: first horizontal angle and step [rad] (nb samples); type C: periodic over 2 pi from
      the file's first horizontal angle when nb > 1 (db = 2 pi / nb); nb = 1: no horizontal
      dependence.
    values: (nb, na) intensity / peak, in [0, 1].
    peak: the largest intensity [cd] (file candela x multipliers).
    flux: luminous flux of the gridded intensity [lm].
    exact: the grid holds the file's knots (False: resampled at its smallest step).
    tilt_angles, tilt_factors: the TILT=INCLUDE table ([deg], multipliers; empty: none).
  """

  kind: int
  a0: float
  da: float
  b0: float
  db: float
  values: Rows[np.float32, int]
  peak: float
  flux: float
  exact: bool
  tilt_angles: Column[np.float64] = dataclasses.field(default_factory=lambda: np.zeros(0))
  tilt_factors: Column[np.float64] = dataclasses.field(default_factory=lambda: np.zeros(0))

  @property
  def na(self) -> int:
    """Vertical samples."""
    return int(self.values.shape[1])

  @property
  def nb(self) -> int:
    """Horizontal samples."""
    return int(self.values.shape[0])

  @property
  def mean(self) -> float:
    """Mean intensity over the sphere [cd]: flux / 4 pi (the normalised profile's divisor)."""
    return self.flux / (4.0 * math.pi)

  @property
  def periodic(self) -> bool:
    """Type C with an azimuth (the horizontal axis wraps)."""
    return self.kind == KIND_C and self.nb > 1


def _mirror(
  angles: Column[np.float64], rows: Rows[np.float64, int], axis: int
) -> tuple[Column[np.float64], Rows[np.float64, int]]:
  """(angles [0, x] mirrored to [-x, x], rows along `axis` mirrored with them)."""
  back = np.arange(len(angles))[1:][::-1]
  return np.concatenate([-angles[back], angles]), np.concatenate([np.take(rows, back, axis=axis), rows], axis=axis)


def _expand_c(h: Column[np.float64], rows: Rows[np.float64, int]) -> tuple[Column[np.float64], Rows[np.float64, int]]:
  """Type C horizontal angles and rows (nh, nv) expanded to a closed 360 degree range (the last
  angle is the first + 360, its row the first's)."""
  if len(h) == 1:
    return np.array([0.0]), rows
  if abs(h[0] - 90.0) <= ANGLE_TOL and abs(h[-1] - 270.0) <= ANGLE_TOL:
    # Bilateral about the 90-270 plane: value(phi) = value(180 - phi), phi in (-90, 90).
    lo = 180.0 - h[::-1][1:-1]
    h2 = np.concatenate([np.where(lo < 0.0, lo + 360.0, lo), h])
    rows2 = np.concatenate([rows[::-1][1:-1], rows])
    order = np.argsort(h2, kind="stable")
    h, rows = h2[order], rows2[order]
  elif abs(h[0]) <= ANGLE_TOL and abs(h[-1] - 90.0) <= ANGLE_TOL:
    h, rows = np.concatenate([h, 180.0 - h[::-1][1:]]), np.concatenate([rows, rows[::-1][1:]])
  if abs(h[0]) <= ANGLE_TOL and abs(h[-1] - 180.0) <= ANGLE_TOL:
    h, rows = np.concatenate([h, 360.0 - h[::-1][1:]]), np.concatenate([rows, rows[::-1][1:]])
  if h[-1] - h[0] < 360.0 - ANGLE_TOL:
    # Wrap the last plane back to the first (periodic).
    h, rows = np.concatenate([h, [h[0] + 360.0]]), np.concatenate([rows, rows[:1]])
  return h, rows


def _step(knots: Column[np.float64]) -> tuple[float, bool]:
  """(grid step [deg] of increasing `knots`, whether every knot is on the grid): their common step,
  else the smallest one."""
  diffs = np.diff(knots)
  smallest = float(diffs.min())
  span = float(knots[-1] - knots[0])
  for divisor in range(1, 64):
    step = smallest / divisor
    if span / step + 1.0 > MAX_SAMPLES:
      break
    q = (knots - knots[0]) / step
    if np.all(np.abs(q - np.round(q)) * step <= ANGLE_TOL):
      return step, True
  return max(smallest, span / (MAX_SAMPLES - 1)), False


def _regular(knots: Column[np.float64]) -> tuple[Column[np.float64], bool]:
  """(regular grid over [knots[0], knots[-1]] [deg], exact). An inexact grid ends on the last
  knot (its step span / (n - 1), close to `_step`'s): a periodic axis keeps its 360 sample at
  360 and no tail past the grid samples as 0."""
  if len(knots) == 1:
    return knots.copy(), True
  step, exact = _step(knots)
  n = round(float(knots[-1] - knots[0]) / step) + 1
  if not exact:
    return np.linspace(knots[0], knots[-1], n), False
  return knots[0] + step * np.arange(n), exact


def _resample(
  v: Column[np.float64], h: Column[np.float64], rows: Rows[np.float64, int]
) -> tuple[Column[np.float64], Column[np.float64], Rows[np.float64, int], bool]:
  """(regular vertical grid, regular horizontal grid, values on it, exact) of rows (nh, nv)."""
  va, exact_v = _regular(v)
  hb, exact_h = _regular(h)
  along_v = np.array([np.interp(va, v, r) for r in rows]) if len(v) > 1 else rows
  out = np.array([np.interp(hb, h, along_v[:, j]) for j in range(along_v.shape[1])]).T if len(h) > 1 else along_v
  return va, hb, out, exact_v and exact_h


def _weighted(a0: float, da: float, p: Rows[np.float64, int], *, cos_weight: bool) -> Column[np.float64]:
  """(nb,) integral over the vertical angle of each row's piecewise-linear interpolant times
  sin (type C: theta from the nadir) or cos (types A / B: latitude), exactly."""
  t = a0 + da * np.arange(p.shape[1])
  t0, t1 = t[:-1], t[1:]
  if cos_weight:
    w0 = np.sin(t1) - np.sin(t0)
    w1 = (t1 * np.sin(t1) + np.cos(t1)) - (t0 * np.sin(t0) + np.cos(t0))
  else:
    w0 = np.cos(t0) - np.cos(t1)
    w1 = (np.sin(t1) - t1 * np.cos(t1)) - (np.sin(t0) - t0 * np.cos(t0))
  # p0 + (p1 - p0) (t - t0) / da integrated against w: p0 w0 + (p1 - p0) (w1 - t0 w0) / da.
  slope = (p[:, 1:] - p[:, :-1]) / da
  return (p[:, :-1] * w0 + slope * (w1 - t0 * w0)).sum(axis=1)


def _flux(kind: int, a0: float, da: float, db: float, p: Rows[np.float64, int]) -> float:
  """Luminous flux [lm] of grid values `p` (nb, na) [cd]: the vertical integrals (`_weighted`)
  over the horizontal angle (exact: the interpolant is linear between rows)."""
  rows = _weighted(a0, da, p, cos_weight=kind != KIND_C)
  if kind == KIND_C:
    return float(rows.sum() * db) if len(rows) > 1 else float(rows[0] * 2.0 * math.pi)
  return float(0.5 * db * (rows[:-1] + rows[1:]).sum())


def grid(data: IesData) -> IesGrid:
  """The regular-grid profile of a parsed file (see the module doc).

  Raises:
    ValueError: for a profile without light (every value 0) or covering no solid angle, or
      angles outside the photometric type's range (C: vertical [0, 180], horizontal [0, 360];
      A / B: [-90, 90]).
  """
  v, h, rows = data.vertical, data.horizontal, data.candela
  kind = _KINDS[data.photometric_type]
  if kind == KIND_C:
    if v[0] < -ANGLE_TOL or v[-1] > 180.0 + ANGLE_TOL or h[0] < -ANGLE_TOL or h[-1] > 360.0 + ANGLE_TOL:
      raise ValueError(f"type C angles must be vertical [0, 180], horizontal [0, 360] deg, got {v[0]}-{v[-1]}, {h[0]}-{h[-1]}")
    h, rows = _expand_c(h, rows)
  else:
    if v[0] < -90.0 - ANGLE_TOL or v[-1] > 90.0 + ANGLE_TOL or h[0] < -90.0 - ANGLE_TOL or h[-1] > 90.0 + ANGLE_TOL:
      raise ValueError(f"type A / B angles must be within [-90, 90] deg, got {v[0]}-{v[-1]}, {h[0]}-{h[-1]}")
    if abs(v[0]) <= ANGLE_TOL and len(v) > 1:
      v, rows = _mirror(v, rows, axis=1)
    if abs(h[0]) <= ANGLE_TOL and len(h) > 1:
      h, rows = _mirror(h, rows, axis=0)
  va, hb, values, exact = _resample(v, h, rows)
  if kind == KIND_C and len(hb) > 1:
    values, hb = values[:-1], hb[:-1]  # periodic: 360 is 0
  peak = float(values.max())
  if not peak > 0.0:
    raise ValueError("the profile emits no light (every candela value is 0)")
  if len(va) == 1 or (kind != KIND_C and len(hb) == 1):
    raise ValueError(
      "the profile covers no solid angle (one vertical angle, or one horizontal angle of a type A / B profile): its flux is 0"
    )
  a0, da = float(va[0]) * _DEG, float(va[1] - va[0]) * _DEG
  b0 = float(hb[0]) * _DEG
  db = (2.0 * math.pi / len(hb) if kind == KIND_C else float(hb[1] - hb[0]) * _DEG) if len(hb) > 1 else 1.0
  return IesGrid(
    kind=kind,
    a0=a0,
    da=da,
    b0=b0,
    db=db,
    values=(values / peak).astype(np.float32),
    peak=peak,
    flux=_flux(kind, a0, da, db, values),
    exact=exact,
    tilt_angles=np.zeros(0) if data.tilt is None else data.tilt.angles,
    tilt_factors=np.zeros(0) if data.tilt is None else data.tilt.factors,
  )


@functools.lru_cache(maxsize=256)
def profile(text: str) -> IesGrid:
  """The grid of LM-63 file `text`, parsed once per process (shared by every light, upload and
  world using it)."""
  return grid(parse(text))


def scale_angle(theta: Column[np.float64], angle_scale: float) -> Column[np.float64]:
  """UsdLux ShapingAPI `ies:angleScale` applied to the angle from the emission axis [rad]: theta /
  s (s > 0), (theta - pi) / -s + pi (s < 0), unchanged (s = 0); clamped to [0, pi]."""
  if angle_scale > 0.0:
    theta = theta / angle_scale
  elif angle_scale < 0.0:
    theta = (theta - math.pi) / -angle_scale + math.pi
  return np.clip(theta, 0.0, math.pi)


def _bilinear(g: IesGrid, a: Column[np.float64], b: Column[np.float64]) -> Column[np.float64]:
  """The grid at angles (a, b) [rad]: bilinear, 0 outside the vertical (and A / B horizontal)
  range, periodic in b for type C."""
  vals = g.values.astype(np.float64)
  ua = (a - g.a0) / g.da
  inside = (ua >= -1e-6) & (ua <= g.na - 1 + 1e-6)
  ia = np.clip(np.floor(ua), 0, max(g.na - 2, 0)).astype(np.int64)
  ta = np.clip(ua - ia, 0.0, 1.0) if g.na > 1 else np.zeros_like(ua)
  ja = np.minimum(ia + 1, g.na - 1)
  if g.nb == 1:
    ib = jb = np.zeros_like(ia)
    tb = np.zeros_like(ua)
  elif g.periodic:
    ub = np.mod(b - g.b0, 2.0 * math.pi) / g.db
    ib = np.floor(ub).astype(np.int64) % g.nb
    tb, jb = ub - np.floor(ub), (ib + 1) % g.nb
  else:
    ub = (b - g.b0) / g.db
    inside &= (ub >= -1e-6) & (ub <= g.nb - 1 + 1e-6)
    ib = np.clip(np.floor(ub), 0, g.nb - 2).astype(np.int64)
    tb, jb = np.clip(ub - ib, 0.0, 1.0), np.minimum(ib + 1, g.nb - 1)
  row0 = vals[ib, ia] * (1.0 - ta) + vals[ib, ja] * ta
  row1 = vals[jb, ia] * (1.0 - ta) + vals[jb, ja] * ta
  return np.where(inside, row0 * (1.0 - tb) + row1 * tb, 0.0)


def sample(g: IesGrid, local: Dirs, angle_scale: float = 0.0) -> Column[np.float64]:
  """(n,) relative intensity (peak 1) toward unit light-local directions `local` (n, 3), after
  the angle scale (see the module doc for the frames)."""
  d = np.asarray(local, np.float64)
  if angle_scale != 0.0:
    theta = np.arccos(np.clip(-d[:, 2], -1.0, 1.0))
    scaled = scale_angle(theta, angle_scale)
    rho = np.hypot(d[:, 0], d[:, 1])
    safe = np.where(rho > 0.0, rho, 1.0)
    cx, cy = np.where(rho > 0.0, d[:, 0] / safe, 1.0), np.where(rho > 0.0, d[:, 1] / safe, 0.0)
    d = np.stack([np.sin(scaled) * cx, np.sin(scaled) * cy, -np.cos(scaled)], axis=1)
  if g.kind == KIND_C:
    a = np.arccos(np.clip(-d[:, 2], -1.0, 1.0))
    b = np.arctan2(d[:, 1], d[:, 0])
  elif g.kind == KIND_A:
    a, b = np.arcsin(np.clip(d[:, 1], -1.0, 1.0)), np.arctan2(d[:, 0], -d[:, 2])
  else:
    a, b = np.arcsin(np.clip(d[:, 0], -1.0, 1.0)), np.arctan2(d[:, 1], -d[:, 2])
  return _bilinear(g, a, b)


def sphere_directions(ntheta: int, nphi: int) -> tuple[Dirs, Column[np.float64]]:
  """(midpoint directions on the unit sphere (ntheta x nphi, 3), their solid angles [sr]): a
  midpoint quadrature in (theta from local -Z, phi), as `photometry.luminous_flux` integrates."""
  theta = (np.arange(ntheta) + 0.5) * (math.pi / ntheta)
  phi = (np.arange(nphi) + 0.5) * (2.0 * math.pi / nphi)
  t, p = np.meshgrid(theta, phi, indexing="ij")
  dirs = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), -np.cos(t)], axis=-1).reshape(-1, 3)
  dw = (np.cos(theta - 0.5 * math.pi / ntheta) - np.cos(theta + 0.5 * math.pi / ntheta)) * (2.0 * math.pi / nphi)
  return dirs, np.repeat(dw, nphi)


# Knots of the coarse profile (`coarse`) in cos(theta) from the nadir; samples per knot interval
# and azimuths per sample ring it takes its bound and mean from.
COARSE_KNOTS: Final = 33
_COARSE_SUBSTEPS: Final = 16
_COARSE_AZIMUTHS: Final = 128


def coarse(g: IesGrid, angle_scale: float) -> Rows[np.float32, 2]:
  """(COARSE_KNOTS, 2) coarse profile of `g` with its angle scale, knots uniform in cos(theta)
  from the nadir in [-1, 1]: column 0 a bound (the profile's maximum over every azimuth and over
  the knot's two neighbouring intervals, so the linear interpolant never falls below the
  profile), column 1 the azimuthal mean at the knot (the light's irradiance estimate). Relative to
  the peak, as `sample`. Types A / B use the same cos(theta) from the emission axis."""
  n, sub = COARSE_KNOTS, _COARSE_SUBSTEPS
  # Samples of every interval (ends included), _COARSE_AZIMUTHS azimuths each.
  c = np.linspace(-1.0, 1.0, (n - 1) * sub + 1)
  phi = (np.arange(_COARSE_AZIMUTHS) + 0.5) * (2.0 * math.pi / _COARSE_AZIMUTHS)
  st = np.sqrt(np.maximum(1.0 - c * c, 0.0))
  dirs = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)), -np.outer(c, np.ones_like(phi))], axis=-1).reshape(
    -1, 3
  )
  vals = sample(g, dirs, angle_scale).reshape(len(c), len(phi))
  ring_max, ring_mean = vals.max(axis=1), vals.mean(axis=1)
  out = np.zeros((n, 2), np.float32)
  for k in range(n):
    lo, hi = max(k - 1, 0) * sub, min(k + 1, n - 1) * sub
    out[k, 0] = ring_max[lo : hi + 1].max()
    out[k, 1] = ring_mean[k * sub]
  return out
