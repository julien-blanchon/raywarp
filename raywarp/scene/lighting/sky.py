"""A procedural clear sky with its sun: the Hosek-Wilkie sky model baked to an equirectangular
radiance map [nits] (`sun_sky`), for outdoor scenes and their randomisation.

The map feeds the ordinary environment path: `rw.Lighting(envmap=sun_sky(...), envmap_intensity=1)`
(physical nits; automatic exposure meters it). The upload cuts the sun disc out of the map as a
directional light with shadow rays and the disc's angular size (`latlong.extract_sun`; a low sun:
below), and the
rest lights the scene as image-based light and GI. Several skies (sun positions, turbidities) are
several maps: `Renderer.per_world["env.id"]` picks one per world. Nothing in the render kernels
changes; a 512-row map bakes in about 0.3 s on the host (NumPy), at scene build.

Sky: the Hosek-Wilkie RGB model (Hosek & Wilkie 2012, "An Analytic Model for Full Spectral
Sky-Dome Radiance"; dataset `data/hosek_rgb.npz` converted from the authors' reference
implementation `ArHosekSkyModelData_RGB.h`, version 1.4a, BSD 3-clause, Copyright (c) 2012-2013
Lukas Hosek and Alexander Wilkie): radiance x 683 lm / W [nits], linear Rec.709. It is fitted for
a sun from the horizon to the zenith, turbidity 1-10 and ground albedo 0-1.

Sun: the extraterrestrial solar illuminance (Darula, Kittler & Gueymard 2005) over the disc's solid
angle, a 5778 K blackbody's colour (`blackbody.blackbody_rgb`), times the transmittance
exp(-m (tau_R + tau_A)) at one wavelength per channel: Rayleigh optical depth 0.008735 l^-4.08
(Leckner 1978, l in um), aerosol beta l^-1.3 with beta = 0.04608 T - 0.04586 (Preetham, Shirley &
Smits 1999), relative air mass m of Kasten & Young 1989. The disc is drawn with its texels'
coverage (supersampled, normalised to the disc's solid angle), so the map holds the sun's exact
power (the part of the disc above the horizon).

A low or hazy sun (about 3 degrees and below at turbidity 3, 10 degrees at turbidity 6-10) is too
weak against its sky to be cut into a light (`latlong.SUN_MIN_SHARE`): it stays in the map as image
light, without a sharp shadow, as a real sunset HDR's does.

Below the horizon the map shows a Lambertian ground of the albedo under the sky and sun's
horizontal illuminance.
"""

from __future__ import annotations

import functools
import importlib.resources
import math
from typing import Final

import numpy as np

from raywarp.scene.images.colorspace import REC709_Y
from raywarp.scene.images.latlong import latlong_directions, texel_solid_angles
from raywarp.scene.lighting.blackbody import blackbody_rgb

# Luminous efficacy of the dataset's radiometric radiance [lm / W].
LUMINOUS_EFFICACY: Final = 683.0
# Extraterrestrial solar illuminance [lux] (Darula, Kittler & Gueymard 2005).
SOLAR_ILLUMINANCE: Final = 133_800.0
# The sun's angular radius [rad] (diameter 0.533 deg, `Light.sun`'s 0.53).
SUN_RADIUS: Final = math.radians(0.533 / 2.0)
# The sun's photosphere temperature [K] (its colour above the atmosphere).
SUN_KELVIN: Final = 5778.0
# The wavelength of each channel [um] (Rec.709 red, green, blue dominant wavelengths) and the
# aerosols' Angstrom exponent.
CHANNEL_UM: Final = (0.611, 0.549, 0.464)
ANGSTROM_ALPHA: Final = 1.3
# Supersamples per texel side of the sun disc's coverage.
DISC_SUPERSAMPLES: Final = 8
# The model's fitted ranges.
TURBIDITY_RANGE: Final = (1.0, 10.0)


@functools.cache
def _dataset() -> tuple[np.ndarray, np.ndarray]:
  """(coefficients (3 channels, 2 albedos, 10 turbidities, 6 control points, 9), radiances (3, 2,
  10, 6)) of the Hosek-Wilkie RGB dataset."""
  with importlib.resources.files("raywarp.data").joinpath("hosek_rgb.npz").open("rb") as f, np.load(f) as data:
    return np.asarray(data["coeffs"], np.float64), np.asarray(data["radiance"], np.float64)


def _bezier(control: np.ndarray, s: float) -> np.ndarray:
  """The quintic Bezier at s of the 6 control points along the first axis of `control`."""
  weights = np.array([math.comb(5, k) * (1.0 - s) ** (5 - k) * s**k for k in range(6)])
  return weights @ control


def _cooked(turbidity: float, albedo: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
  """((3, 9) distribution coefficients, (3,) zenith radiances) of the model at a solar `elevation`
  [rad], bilinear in albedo and turbidity (`ArHosekSkyModel_CookConfiguration`)."""
  coeffs, radiance = _dataset()
  s = (elevation / (0.5 * math.pi)) ** (1.0 / 3.0)
  t0 = int(turbidity)
  rem = turbidity - t0
  cfg, rad = np.zeros((3, 9)), np.zeros(3)
  for t, wt in ((t0, 1.0 - rem), (t0 + 1, rem)):
    if wt == 0.0 or t > coeffs.shape[2]:
      continue
    for a, wa in ((0, 1.0 - albedo), (1, albedo)):
      cfg += wa * wt * np.stack([_bezier(coeffs[c, a, t - 1], s) for c in range(3)])
      rad += wa * wt * np.array([_bezier(radiance[c, a, t - 1], s) for c in range(3)])
  return cfg, rad


def _sky(cfg: np.ndarray, cos_theta: np.ndarray, cos_gamma: np.ndarray) -> np.ndarray:
  """(..., 3) relative radiance of the model's distribution (`ArHosekSkyModel_GetRadianceInternal`)
  at view zenith cosines `cos_theta` (>= 0) and cosines to the sun `cos_gamma`."""
  c = cfg.T  # (9, 3): c[i] broadcasts over the channels
  ct, cg = cos_theta[..., None], cos_gamma[..., None]
  gamma = np.arccos(np.clip(cg, -1.0, 1.0))
  mie = (1.0 + cg * cg) / (1.0 + c[8] * c[8] - 2.0 * c[8] * cg) ** 1.5
  first = 1.0 + c[0] * np.exp(c[1] / (ct + 0.01))
  return first * (c[2] + c[3] * np.exp(c[4] * gamma) + c[5] * cg * cg + c[6] * mie + c[7] * np.sqrt(ct))


def sun_direction(elevation: float, azimuth: float) -> np.ndarray:
  """The unit direction towards the sun [map frame, +z up] at `elevation` above the horizon and
  `azimuth` counter-clockwise from +x [degrees]."""
  e, a = math.radians(elevation), math.radians(azimuth)
  return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def sun_radiance(elevation: float, turbidity: float) -> np.ndarray:
  """(3,) the sun disc's linear Rec.709 radiance [nits] seen from the ground (module doc)."""
  zenith = 90.0 - elevation
  airmass = 1.0 / (math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364)
  beta = 0.04608 * turbidity - 0.04586
  um = np.asarray(CHANNEL_UM)
  depth = 0.008735 * um**-4.08 + beta * um**-ANGSTROM_ALPHA
  solid_angle = 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS))
  color = np.asarray(blackbody_rgb(SUN_KELVIN), np.float64)
  color /= float(color @ np.asarray(REC709_Y))
  return SOLAR_ILLUMINANCE / solid_angle * color * np.exp(-airmass * depth)


def _disc_coverage(dirs: np.ndarray, sun: np.ndarray, height: int) -> tuple[tuple[slice, slice], np.ndarray]:
  """(rows / columns window, its texels' covered fraction by the sun disc): supersampled, then
  scaled so that the texels hold the disc's solid angle exactly (a texel's sub-samples are not
  equal areas near the poles; a disc no sub-sample sees: the texel nearest the sun), the part
  below the horizon left out."""
  cos_reach = math.cos(SUN_RADIUS + 2.0 * math.pi / height)
  rows, cols = np.nonzero(dirs @ sun > cos_reach)
  r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
  n = DISC_SUPERSAMPLES
  sub = (np.arange(n) + 0.5) / n
  ys = (r0 + np.arange(r1 - r0)[:, None] + sub[None]).reshape(-1)
  xs = (c0 + np.arange(c1 - c0)[:, None] + sub[None]).reshape(-1)
  theta = ys / height * math.pi
  phi = (0.5 - xs / (2 * height)) * 2.0 * math.pi
  st = np.sin(theta)[:, None]
  d = np.stack(np.broadcast_arrays(st * np.cos(phi)[None], st * np.sin(phi)[None], np.cos(theta)[:, None]), axis=-1)
  inside = (d @ sun >= math.cos(SUN_RADIUS)).astype(np.float64)
  coverage = inside.reshape(r1 - r0, n, c1 - c0, n).mean(axis=(1, 3))
  window = (slice(r0, r1), slice(c0, c1))
  if not coverage.any():
    # A disc smaller than the sub-sample spacing (low heights) can fall between every sub-sample:
    # the texel nearest the sun holds it.
    coverage[np.unravel_index(np.argmax(dirs[window] @ sun), coverage.shape)] = 1.0
  omega = texel_solid_angles(height, 2 * height)[window]
  coverage *= 2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS)) / float((coverage * omega).sum())
  return window, np.where(dirs[window][..., 2] > 0.0, coverage, 0.0)


def sun_sky(
  elevation: float,
  azimuth: float = 0.0,
  *,
  turbidity: float = 3.0,
  ground_albedo: float = 0.3,
  height: int = 512,
) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
  """A clear sky with its sun as a (height, 2 height, 3) equirectangular linear Rec.709 radiance
  map [nits] (module doc; +z up, row 0 the zenith, the centre column along +x).

  Args:
    elevation: the sun's elevation above the horizon [degrees], in [0, 90].
    azimuth: its azimuth counter-clockwise from +x [degrees].
    turbidity: the atmosphere's turbidity, in [1, 10] (2: very clear, 3: clear, 6: hazy).
    ground_albedo: the ground's albedo, in [0, 1] (the sky's lower light and the map's ground).
    height: the map's rows (at least 32).

  Raises:
    ValueError: for an argument outside its range.
  """
  if not 0.0 <= elevation <= 90.0:
    raise ValueError(f"sun_sky: elevation must be in [0, 90] degrees, got {elevation}")
  if not TURBIDITY_RANGE[0] <= turbidity <= TURBIDITY_RANGE[1]:
    raise ValueError(f"sun_sky: turbidity must be in [1, 10], got {turbidity}")
  if not 0.0 <= ground_albedo <= 1.0:
    raise ValueError(f"sun_sky: ground_albedo must be in [0, 1], got {ground_albedo}")
  if height < 32 or not math.isfinite(azimuth):
    raise ValueError(f"sun_sky: height must be >= 32 and azimuth finite, got {height}, {azimuth}")
  dirs = latlong_directions(height, 2 * height)
  sun = sun_direction(elevation, azimuth)
  cfg, zenith = _cooked(turbidity, ground_albedo, math.radians(elevation))
  up = dirs[..., 2] > 0.0
  sky = _sky(cfg, np.maximum(dirs[..., 2], 0.0), dirs @ sun) * zenith * LUMINOUS_EFFICACY
  image = np.where(up[..., None], sky, 0.0)
  # The ground: Lambertian under the sky's and the sun's horizontal illuminance.
  omega = texel_solid_angles(height, 2 * height)
  sun_rgb = sun_radiance(elevation, turbidity)
  e_sky = (image * (omega * np.maximum(dirs[..., 2], 0.0))[..., None]).sum(axis=(0, 1))
  e_sun = sun_rgb * (2.0 * math.pi * (1.0 - math.cos(SUN_RADIUS))) * math.sin(math.radians(elevation))
  image[~up] = ground_albedo * (e_sky + e_sun) / math.pi
  window, coverage = _disc_coverage(dirs, sun, height)
  image[window] += coverage[..., None] * sun_rgb
  return image.astype(np.float32)
