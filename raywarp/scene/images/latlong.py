"""The equirectangular "latlong" layout of environment maps on the host: texel directions and
solid angles, a yaw roll, and cutting a sun out as a light.

Maps are equirectangular images in a Z-up world, Blender's (and three.js's) layout: image row 0
is the zenith (+z), the horizon is the middle row, the image centre column looks along +x and
longitude grows towards the left of the image (u = 0.25 looks along +y), so a panorama seen from
inside the sphere is not mirrored. `latlong_directions` / `latlong_uv` are the NumPy definition
(the device one: `core.shade.environment.latlong`); `roll_yaw` turns a map about +z, `extract_sun`
cuts a physical sky's sun out as a distant light.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Final, final

import numpy as np

from raywarp.scene.images.colorspace import REC709_Y

if TYPE_CHECKING:
  from shape_extensions import IntVar

# An equirectangular map (height, width, channels) of element type D.
type LatLong[H: IntVar, W: IntVar, C: IntVar, D: np.generic] = np.ndarray[[H, W, C], np.dtype[D]]


def latlong_directions(h: int, w: int) -> np.ndarray[[int, int, 3], np.dtype[np.float64]]:
  """Directions of the texel centres of an (h, w) equirectangular map (`uv_to_direction`)."""
  theta = (np.arange(h) + 0.5) / h * np.pi
  phi = (0.5 - (np.arange(w) + 0.5) / w) * 2.0 * np.pi
  st = np.sin(theta)[:, None]
  return np.stack(np.broadcast_arrays(st * np.cos(phi)[None], st * np.sin(phi)[None], np.cos(theta)[:, None]), axis=-1)


def latlong_uv[F: (np.float32, np.float64)](
  d: np.ndarray[[int, int, 3], np.dtype[F]],
) -> tuple[np.ndarray[[int, int], np.dtype[F]], np.ndarray[[int, int], np.dtype[F]]]:
  """(u, v) of map-frame directions (`direction_to_uv` with no yaw; u in [0, 1))."""
  u = 0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2.0 * np.pi)
  return u - np.floor(u), np.arccos(np.clip(d[..., 2], -1.0, 1.0)) / np.pi


def texel_solid_angles(h: int, w: int) -> np.ndarray[[int, int], np.dtype[np.float64]]:
  """Solid angle of each texel of an (h, w) equirectangular map."""
  st = np.sin((np.arange(h) + 0.5) / h * np.pi)[:, None]
  return np.broadcast_to(st * (2.0 * np.pi / w) * (np.pi / h), (h, w))


def roll_yaw[H: IntVar, W: IntVar, C: IntVar, D: np.generic](image: LatLong[H, W, C, D], yaw: float) -> LatLong[H, W, C, D]:
  """The equirectangular `image` (h, w, c) turned by `yaw` about +z: its columns rolled.

  `rotate_latlong(image, Rz(yaw))` without the per-texel directions: a yaw only moves every row
  by the same fractional number of columns, interpolated linearly (float32).
  """
  w = image.shape[1]
  shift = yaw * w / (2.0 * math.pi)  # output column x samples input column x + shift
  k = math.floor(shift)
  f = np.asarray(shift - k, np.float32)  # the weights in float32
  a = np.roll(image, -k, axis=1)
  b = np.roll(image, -(k + 1), axis=1)
  return (a * (1.0 - f) + b * f).astype(image.dtype)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MapSun:
  """A compact light source cut out of an environment map (`extract_sun`; lit as a shadowed
  directional light, `core.pipeline.gpu_scene.gpu_scene.EnvSuns`).

  Attributes:
    direction: unit direction towards the sun, map frame.
    irradiance: (3,) its irradiance at normal incidence (sum of the cut-out radiance x solid angle).
    angle: angular radius of the sun disc [rad]: the uniform disc of the cut-out power's angular
      spread (at most the cut-out region's radius).
    extent: angular radius of the cut-out region about `direction` [rad] (>= `angle`): where the
      map holds the refill, which the background covers with the sun drawn back.
    share: its share of the irradiance of a surface facing it (the deepest its shadow darkens).
  """

  direction: np.ndarray[[3], np.dtype[np.float64]]
  irradiance: np.ndarray[[3], np.dtype[np.float64]]
  angle: float
  extent: float
  share: float


# The sun rule (`extract_sun`). A texel is bright above SUN_SKY_RATIO x the map's median radiance
# and SUN_PEAK_RATIO x its peak. The sun is the connected bright region around the peak texel,
# cut out when it is
# - compact: within SUN_MAX_ANGLE (+ one texel: a sun a low-resolution map spreads over a few
#   texels) of its centre, and
# - dominant: at least SUN_MIN_SHARE of the irradiance of a surface facing it (the deepest its
#   shadow darkens; below, the shadow is faint and not worth its ray).
# Measured on 9 Poly Haven 1k maps: the suns of 2 sunny skies span 0.7-1.8 degrees and hold 0.69 /
# 0.82 of the facing irradiance; studio lamps and windows span 3.6-20 degrees and hold <= 0.25
# (compact ones) or are extended; a sunset behind haze holds 0.05; 3 overcast / hazy skies have no
# bright texel (peak < 50 x median). So: sunny skies -> one sun; studios with several softboxes,
# overcast skies, hazy sunsets -> none. At most one sun per map.
SUN_SKY_RATIO: Final = 50.0


SUN_PEAK_RATIO: Final = 1.0e-3


SUN_MAX_ANGLE: Final = math.radians(5.0)


SUN_MIN_SHARE: Final = 0.3


# Fewest map rows to look for a sun in: a coarser texel (over 5.6 degrees) cannot tell a sun from a
# lamp (the compactness bound grows by a texel).
SUN_MIN_ROWS: Final = 32


# The refill ring around the cut-out: out to twice its radius plus RING_MARGIN [rad].
_RING_MARGIN: Final = math.radians(0.5)


# Floor of the power sums `extract_sun` divides by (a black map returns earlier: `peak <= 0`).
_MIN_POWER: Final = 1.0e-30


def _grow(
  seed: tuple[int, int], allowed: np.ndarray[[int, int], np.dtype[np.bool_]]
) -> np.ndarray[[int, int], np.dtype[np.bool_]]:
  """The 8-connected region of `allowed` texels (h, w) holding `seed` (columns wrap in longitude;
  `allowed` bounds it, so this takes as many steps as the region's radius in texels)."""
  rows = np.flatnonzero(allowed.any(axis=1))
  r0, r1 = int(rows.min()), int(rows.max()) + 1
  band = allowed[r0:r1]
  region = np.zeros_like(band)
  region[seed[0] - r0, seed[1]] = True
  while True:
    grown = region | np.roll(region, 1, axis=1) | np.roll(region, -1, axis=1)
    grown[1:] |= grown[:-1].copy()
    grown[:-1] |= grown[1:].copy()
    grown &= band
    if np.array_equal(grown, region):
      break
    region = grown
  out = np.zeros_like(allowed)
  out[r0:r1] = region
  return out


def extract_sun[H: IntVar, W: IntVar, D: np.generic](image: LatLong[H, W, 3, D]) -> tuple[LatLong[H, W, 3, D], MapSun] | None:
  """(map without the sun, the sun) of an equirectangular radiance map (h, w, 3), or None when it
  has no dominant compact source (the rule above `SUN_SKY_RATIO`).

  The sun of a physical sky HDR (a disc of ~0.5 degree holding a large part of the light) cannot
  be resolved by the GI probes' rays or the SH irradiance: it becomes a directional light (shadow
  rays) and its texels are refilled with the median of the ring around them; the sun holds the
  excess over that fill, so the map's power is the remainder's plus the sun's.
  """
  h, w = image.shape[:2]
  if h < SUN_MIN_ROWS:
    return None
  rgb = image[..., :3].astype(np.float64)
  lum = rgb @ np.asarray(REC709_Y)
  y, x = divmod(int(lum.argmax()), lum.shape[1])
  peak = float(lum[y, x])
  if peak <= 0.0:
    return None
  dirs = latlong_directions(h, w)
  omega = texel_solid_angles(h, w)
  reach = SUN_MAX_ANGLE + math.pi / h  # plus one texel
  bright = lum > max(peak * SUN_PEAK_RATIO, SUN_SKY_RATIO * float(np.median(lum)))
  if not bright[y, x]:
    return None  # no texel stands out of the sky (overcast)
  cos_to_peak = dirs @ dirs[y, x]
  region = _grow((y, x), bright & (cos_to_peak > math.cos(2.0 * reach)))
  if np.any(region & (cos_to_peak <= math.cos(2.0 * reach - math.pi / h))):
    return None  # it reaches the bound: an extended source
  centre = (dirs[region] * (lum[region] * omega[region])[:, None]).sum(0)
  centre /= np.linalg.norm(centre, keepdims=True)
  cos_to_centre = dirs @ centre
  radius = float(np.arccos(np.clip(cos_to_centre[region].min(), -1.0, 1.0)))
  if radius > reach:
    return None
  ring = ~region & (cos_to_centre > math.cos(min(2.0 * radius + _RING_MARGIN, 2.0 * reach)))
  fill = np.median(rgb[ring], axis=0) if np.any(ring) else np.zeros(3)
  excess = np.maximum(rgb[region] - fill, 0.0) * omega[region][:, None]
  excess_lum = excess @ np.asarray(REC709_Y)
  facing = float((lum * omega * np.maximum(cos_to_centre, 0.0)).sum())
  share = float((excess_lum * np.maximum(cos_to_centre[region], 0.0)).sum()) / max(facing, _MIN_POWER)
  if share < SUN_MIN_SHARE:
    return None
  direction = (dirs[region] * excess_lum[:, None]).sum(0)
  direction /= np.linalg.norm(direction)
  # Angular radius of the uniform disc with the excess power's spread about the direction
  # (E[theta^2] = R^2 / 2), each texel's radiance spread over its own area (`_texel_moments`): the
  # sun's size as the map resolves it, not the cut-out's (a one-texel sun: the texel's size; a sun
  # at a pole: the cap its first rows cover).
  ys, xs = np.nonzero(region)
  spread = _texel_moments(ys, xs, direction, h, w)
  disc = math.sqrt(2.0 * float((excess_lum * spread).sum()) / max(float(excess_lum.sum()), _MIN_POWER))
  half = _half_diagonal(ys, h, w)
  # The cut-out's reach from the sun's direction: its farthest texel centre plus a texel.
  extent = float(np.arccos(np.clip(dirs[region] @ direction, -1.0, 1.0)).max()) + half
  out = image.copy()
  out[region, :3] = fill.astype(image.dtype)
  return out, MapSun(direction=direction, irradiance=excess.sum(0), angle=min(disc, radius + half), extent=extent, share=share)


# Sub-cells per texel side in `_texel_moments`: the residual (the sub-cells' own spread, taken as
# flat rectangles) is 1 / _SUB^2 of a texel's, and the polar wedges' curvature is resolved.
_SUB: Final = 4


def _texel_moments(
  ys: np.ndarray[[int], np.dtype[np.intp]],
  xs: np.ndarray[[int], np.dtype[np.intp]],
  direction: np.ndarray[[3], np.dtype[np.float64]],
  h: int,
  w: int,
) -> np.ndarray[[int], np.dtype[np.float64]]:
  """E[theta^2] of each texel (ys, xs) of an (h, w) map about `direction`, its radiance uniform
  over its solid angle: _SUB x _SUB sub-cells weighted by their solid angle, plus each sub-cell's
  spread as a flat rectangle ((a^2 + b^2) / 12). Near a pole a texel is a thin wedge, not the disc
  of its solid angle: the row-0 texels around the pole spread over the cap of one texel's height."""
  f = (np.arange(_SUB) + 0.5) / _SUB
  theta = (ys[:, None, None] + f[None, :, None]) * (np.pi / h)  # (n, sub, 1)
  phi = (0.5 - (xs[:, None, None] + f[None, None, :]) / w) * (2.0 * np.pi)  # (n, 1, sub)
  st = np.sin(theta)
  d = np.stack(np.broadcast_arrays(st * np.cos(phi), st * np.sin(phi), np.cos(theta)), axis=-1)  # (n, sub, sub, 3)
  # The angle to `direction` (2 asin(|d - s| / 2): exact near 0).
  angle = 2.0 * np.arcsin(np.clip(0.5 * np.linalg.norm(d - direction, axis=-1), 0.0, 1.0))
  weight = np.broadcast_to(st, angle.shape)
  a, b = np.pi / (h * _SUB), st * (2.0 * np.pi / (w * _SUB))  # sub-cell height, width (n, sub, 1)
  local = angle * angle + (a * a + np.broadcast_to(b * b, angle.shape)) / 12.0
  return (local * weight).sum((1, 2)) / np.maximum(weight.sum((1, 2)), _MIN_POWER)


def _half_diagonal(ys: np.ndarray[[int], np.dtype[np.intp]], h: int, w: int) -> float:
  """The largest half diagonal of the texels of rows `ys` of an (h, w) map [rad]: how far a texel's
  light reaches past its centre (the polar extent of a row-0 texel at a pole, where its solid
  angle is tiny)."""
  a = np.pi / h
  b = np.sin((ys + 0.5) * a) * (2.0 * np.pi / w)
  return float(0.5 * np.sqrt(a * a + b * b).max())
