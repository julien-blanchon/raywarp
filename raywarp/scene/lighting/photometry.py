"""Light authoring units converted once to the canonical UsdLux record (`scene.model.lights.Light`).

The canonical record keeps UsdLux's inputs (radiance [nits], `normalize`, shaping); these helpers
derive the `intensity` that gives an authored photometric quantity, from the record as the
renderer shades it (`core/pipeline/upload/lights.py`: the point form's lobe, cone, IES profile, area
normalisation and stage units):

* `with_power`: luminous flux [lm] (Filament §5.2.2.1). Spot cones either keep the flux in the
  cone ("reflector", Filament's focused spot: I = flux / (2 pi (1 - cos outer)) for a hard cone)
  or absorb what falls outside it ("absorber": the flux of the light without its cone, I = flux /
  4 pi for a point). An IES profile always shapes the flux it is given.
* `with_illuminance`: illuminance [lux] on a surface facing the light at a distance [m] (exact
  on the axis of spheres, disks and rects; the point form elsewhere), the OVRTX
  `PhotometricAreaLightAPI` target.
* `watts_to_lumens`: electrical power times luminous efficacy [lm/W], or times efficiency x 683
  lm/W (Filament's `intensity(watts, efficiency)`).
* `cct_color`: a correlated colour temperature as a linear Rec.709 colour of luminance 1:
  "blackbody" (UsdLux / pxr's table, what `enable_color_temperature` renders), "planckian"
  (Krystek's 1985 Planckian locus, Filament's `Color::cct`), "daylight" (CIE D series, Filament's
  `Color::illuminantD`, 4000-25000 K); `spectrum_color` for a sampled spectrum.

Luminous quantities use the luminance of the light's colour (Rec.709 Y). The flux is the point
form's integral over the sphere (a falloff window is ignored), 16384 polar steps (512 x 256 for a
2D IES profile or a tube): it equals the area form's (pi L A for a one-sided rect or disk,
4 pi^2 r^2 L for a sphere).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Final

import numpy as np

from raywarp._base.types import SPOT_INNER_FRACTION, LightType
from raywarp.scene.images.colorspace import REC709_Y
from raywarp.scene.lighting import ies_profile
from raywarp.scene.lighting.blackbody import blackbody_rgb
from raywarp.scene.model import checks
from raywarp.scene.model.lights import MAX_ANGLE_DEG, Illuminant, Light, Lights, SpotMode
from raywarp.scene.model.types import Column, LightKind

type Dirs = np.ndarray[[int, 3], np.dtype[np.float64]]

# Luminous efficacy of 555 nm light [lm/W] (the maximum).
LUMENS_PER_WATT: Final = 683.0
# The XYZ -> linear Rec.709 matrix (D65).
_XYZ_TO_709: Final = np.array(
  [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]
)
# Validity of the CIE daylight series [K].
DAYLIGHT_RANGE: Final = (4000.0, 25000.0)
# Krystek's 1985 rational fits of the Planckian locus in CIE 1960 uv (Filament's `Color::cct`),
# u or v = (a + b T + c T^2) / (1 + d T + e T^2), valid over PLANCKIAN_RANGE [K]: the one locus
# (the white balance's device form reads the same coefficients,
# `core/post/grading/colorimetry.planckian_xy`).
PLANCKIAN_U: Final = (0.860117757, 1.54118254e-4, 1.28641212e-7, 8.42420235e-4, 7.08145163e-7)
PLANCKIAN_V: Final = (0.317398726, 4.22806245e-5, 4.20481691e-8, -2.89741816e-5, 1.61456053e-7)
PLANCKIAN_RANGE: Final = (1000.0, 15000.0)
# The CIE daylight series x(T) = a + b 1e3 / T + c 1e6 / T^2 + d 1e9 / T^3 below / above 7000 K
# (y = -3 x^2 + 2.87 x - 0.275).
DAYLIGHT_X_LOW: Final = (0.244063, 0.09911, 2.9678, -4.6070)
DAYLIGHT_X_HIGH: Final = (0.237040, 0.24748, 1.9018, -2.0064)
DAYLIGHT_SPLIT: Final = 7000.0
_POLAR_STEPS: Final = 16384
_PROFILE_STEPS: Final = (512, 256)


def watts_to_lumens(watts: float, *, efficacy: float | None = None, efficiency: float | None = None) -> float:
  """Luminous flux [lm] of `watts` of electrical power: times `efficacy` [lm/W], or times
  `efficiency` (0-1) x 683 lm/W (exactly one of them).

  Raises:
    ValueError: for none or both, a negative or non-finite value, or an efficiency above 1.
  """
  where = "watts_to_lumens"
  checks.in_range("watts", watts, where, low=0.0)
  if efficacy is not None and efficiency is None:
    return watts * checks.in_range("efficacy [lm/W]", efficacy, where, low=0.0)
  if efficiency is not None and efficacy is None:
    return watts * (LUMENS_PER_WATT * checks.in_range("efficiency", efficiency, where, low=0.0, high=1.0))
  raise ValueError("watts_to_lumens needs exactly one of efficacy [lm/W] or efficiency (0-1)")


def _xy_color(x: float, y: float) -> tuple[float, float, float]:
  """CIE xy chromaticity as linear Rec.709 of luminance 1 (negative channels clipped, then
  renormalised)."""
  xyz = np.array([x / y, 1.0, (1.0 - x - y) / y])
  rgb = np.maximum(_XYZ_TO_709 @ xyz, 0.0)
  rgb /= float(np.dot(rgb, REC709_Y))
  r, g, b = (float(c) for c in rgb)
  return r, g, b


def _rational(c: tuple[float, ...], t: float) -> float:
  """(a + b t + c t^2) / (1 + d t + e t^2) of coefficients `c`."""
  return (c[0] + c[1] * t + c[2] * t * t) / (1.0 + c[3] * t + c[4] * t * t)


def planckian_xy(kelvin: float) -> tuple[float, float]:
  """Krystek's rational approximation of the Planckian locus (PLANCKIAN_RANGE), as CIE xy."""
  u, v = _rational(PLANCKIAN_U, kelvin), _rational(PLANCKIAN_V, kelvin)
  d = 2.0 * u - 8.0 * v + 4.0
  return 3.0 * u / d, 2.0 * v / d


def daylight_xy(kelvin: float) -> tuple[float, float]:
  """CIE daylight series chromaticity (D illuminants) at `kelvin` in DAYLIGHT_RANGE.

  Raises:
    ValueError: outside DAYLIGHT_RANGE.
  """
  if not DAYLIGHT_RANGE[0] <= kelvin <= DAYLIGHT_RANGE[1]:
    raise ValueError(f"the CIE daylight series spans {DAYLIGHT_RANGE} K, got {kelvin}")
  a, b, c, d = DAYLIGHT_X_LOW if kelvin <= DAYLIGHT_SPLIT else DAYLIGHT_X_HIGH
  it = 1.0e3 / kelvin
  x = ((d * it + c) * it + b) * it + a
  return x, -3.0 * x * x + 2.87 * x - 0.275


def cct_color(kelvin: float, illuminant: Illuminant = "blackbody") -> tuple[float, float, float]:
  """Linear Rec.709 colour of luminance 1 of a colour temperature (see the module doc).
  "planckian" outside PLANCKIAN_RANGE (Krystek's fit) uses the nearest end, with a warning (the
  helpers that call it have no load report; USD imports report their own clamps).

  Raises:
    ValueError: for a temperature <= 0, an unknown illuminant, or daylight outside its range.
  """
  if not kelvin > 0.0:
    raise ValueError(f"colour temperature must be > 0 K, got {kelvin}")
  if illuminant == "blackbody":
    return blackbody_rgb(kelvin)
  if illuminant == "planckian":
    clamped = min(max(kelvin, PLANCKIAN_RANGE[0]), PLANCKIAN_RANGE[1])
    if clamped != kelvin:
      message = f"planckian colour temperature {kelvin} K clamped to {clamped} K (fit range {PLANCKIAN_RANGE} K)"
      warnings.warn(message, stacklevel=2)
    return _xy_color(*planckian_xy(clamped))
  if illuminant == "daylight":
    return _xy_color(*daylight_xy(kelvin))
  raise ValueError(f"illuminant must be 'blackbody', 'planckian' or 'daylight', got {illuminant!r}")


def _lobe(x: Column[np.float64], mu: float, s1: float, s2: float) -> Column[np.float64]:
  """Piecewise Gaussian of the Wyman-Sloan-Shirley CIE 1931 fit."""
  s = np.where(x < mu, s1, s2)
  return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz(nm: Column[np.float64]) -> np.ndarray[[int, 3], np.dtype[np.float64]]:
  """(n, 3) CIE 1931 2-degree colour matching functions at wavelengths `nm` (Wyman, Sloan and
  Shirley 2013 multi-lobe fit, within about 1 % of the tables)."""
  x = 1.056 * _lobe(nm, 599.8, 37.9, 31.0) + 0.362 * _lobe(nm, 442.0, 16.0, 26.7) - 0.065 * _lobe(nm, 501.1, 20.4, 26.2)
  y = 0.821 * _lobe(nm, 568.8, 46.9, 40.5) + 0.286 * _lobe(nm, 530.9, 16.3, 31.1)
  z = 1.217 * _lobe(nm, 437.0, 11.8, 36.0) + 0.681 * _lobe(nm, 459.0, 26.0, 13.8)
  return np.stack([x, y, z], axis=1)


def spectrum_color(nm: Column[np.float64], values: Column[np.float64]) -> tuple[float, float, float]:
  """Linear Rec.709 colour of luminance 1 of a spectrum sampled at increasing wavelengths `nm`
  (piecewise linear, 0 outside them), integrated over 360-830 nm at 1 nm.

  Raises:
    ValueError: for wavelengths that do not increase, negative or non-finite values, or a
      spectrum without luminance.
  """
  nm, values = np.asarray(nm, np.float64), np.asarray(values, np.float64)
  finite = bool(np.all(np.isfinite(nm)) and np.all(np.isfinite(values)))
  if not finite or len(nm) < 2 or np.any(np.diff(nm) <= 0.0) or len(values) != len(nm) or np.any(values < 0.0):
    raise ValueError("a spectrum needs >= 2 finite increasing wavelengths and as many finite values >= 0")
  grid = np.linspace(360.0, 830.0, 471)
  power = np.asarray(np.interp(grid, nm, values, left=0.0, right=0.0), np.float64)
  xyz: np.ndarray[[3], np.dtype[np.float64]] = (cie_xyz(grid) * power[:, None]).sum(axis=0)
  if not xyz[1] > 0.0:
    raise ValueError("the spectrum has no luminance in 360-830 nm")
  total = float(xyz.sum())
  return _xy_color(float(xyz[0]) / total, float(xyz[1]) / total)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _PointForm:
  """A record's point form as the renderer shades it (`core/pipeline/upload/lights.py`,
  `eval_light`).

  Attributes:
    luminous: peak luminous intensity [cd] (the luminance of the intensity colour).
    ltype: the core's LightType.
    cos_outer: cosine of the cone's cutoff angle (_NO_CONE: no cone).
    start, scale: the cone's smooth-start angle [rad] and 1 / (cutoff - start) [1/rad]
      (`lights.cone_factor`).
    grid, angle_scale: the IES profile (None: none) and its UsdLux angle scale.
    disk_radius: a disk light's radius [m] (0: not a disk).
    rect: a rect light's (width, height) [m] ((0, 0): not a rect).
  """

  luminous: float
  ltype: int
  cos_outer: float
  start: float
  scale: float
  grid: ies_profile.IesGrid | None
  angle_scale: float
  disk_radius: float
  rect: tuple[float, float]


# `_PointForm.cos_outer` without a cone (below every cosine).
_NO_CONE: Final = -2.0


def _point_form(light: Light) -> _PointForm:
  """The record derived as an upload would (one light)."""
  # Imported here: the upload module imports the host scene's records (scene/ imports no core at
  # module level).
  from raywarp.core.pipeline.upload import lights as upload  # noqa: PLC0415
  from raywarp.core.shade.lights.punctual import CONE_MIN_WIDTH  # noqa: PLC0415

  derived = upload.derive_lights(Lights.from_records([light], bodies=[-1]))
  slots, _ = upload.light_arrays(derived)
  ltype = int(slots["light.type"][0, 0])
  # The cone as `core/shade/lights/pack.cone` derives it (cutoff in degrees: < 90 is a cosine
  # lobe's cone).
  cutoff_deg = float(slots["light.cutoff"][0, 0])
  cutoff = math.radians(cutoff_deg)
  inner = float(slots["light.cone_inner"][0, 0]) if "light.cone_inner" in slots else SPOT_INNER_FRACTION
  coned = ltype == int(LightType.SPOT) or (ltype == int(LightType.COSINE) and cutoff_deg < 90.0)
  start = cutoff * min(max(inner, 0.0), 1.0)
  text = derived.ies.tolist()[0]
  color = np.asarray(slots["light.color"][0, 0], np.float64) * float(slots["light.strength"][0, 0])
  return _PointForm(
    luminous=float(np.dot(color, REC709_Y)),
    ltype=ltype,
    cos_outer=math.cos(cutoff) if coned else _NO_CONE,
    start=start,
    scale=1.0 / max(cutoff - start, CONE_MIN_WIDTH),
    grid=ies_profile.profile(text) if text else None,
    angle_scale=float(derived.ies_angle_scale[0]),
    disk_radius=float(derived.radius[0]) if light.kind == LightKind.DISK else 0.0,
    rect=(float(derived.width[0]), float(derived.height[0])),
  )


def _relative(pf: _PointForm, local: Dirs) -> Column[np.float64]:
  """(n,) intensity toward light-local unit directions, relative to the peak (`eval_light`'s
  angular terms times the IES profile)."""
  from raywarp._base.types import LightType  # noqa: PLC0415

  axis = np.array([1.0, 0.0, 0.0]) if pf.ltype == int(LightType.TUBE) else np.array([0.0, 0.0, -1.0])
  cos_t = local @ axis
  out = np.ones(len(local))
  if pf.ltype == int(LightType.COSINE):
    out *= np.maximum(cos_t, 0.0)
  elif pf.ltype == int(LightType.TUBE):
    out *= np.sqrt(np.maximum(1.0 - cos_t * cos_t, 0.0))
  if pf.cos_outer > _NO_CONE:
    t = np.clip((np.arccos(np.clip(cos_t, -1.0, 1.0)) - pf.start) * pf.scale, 0.0, 1.0)
    out *= np.where(cos_t <= pf.cos_outer, 0.0, 1.0 - t * t * (3.0 - 2.0 * t))
  if pf.grid is not None:
    out *= ies_profile.sample(pf.grid, local, pf.angle_scale)
  return out


def luminous_flux(light: Light) -> float:
  """The flux [lm] the record emits as rendered (see the module doc).

  Raises:
    ValueError: for a distant or display light (no flux).
  """
  if light.kind not in (LightKind.SPHERE, LightKind.RECT, LightKind.DISK, LightKind.CYLINDER):
    raise ValueError(f"a {light.kind.name.lower()} light has no luminous flux (distant lights are given in lux)")
  from raywarp._base.types import LightType  # noqa: PLC0415

  pf = _point_form(light)
  # Axially symmetric about local -Z (no tube lobe, no 2D profile): one azimuth sample per ring.
  axial = (pf.grid is None or pf.grid.nb == 1) and pf.ltype != int(LightType.TUBE)
  dirs, dw = ies_profile.sphere_directions(*((_POLAR_STEPS, 1) if axial else _PROFILE_STEPS))
  return pf.luminous * float((_relative(pf, dirs) * dw).sum())


def with_power(light: Light, lumens: float, *, spot_mode: SpotMode = "reflector") -> Light:
  """`light` with the intensity that emits `lumens` (see the module doc for `spot_mode`).

  Raises:
    ValueError: for negative lumens, an unknown mode, a light without flux, or one emitting
      nothing (zero colour, a closed cone).
  """
  if not lumens >= 0.0:
    raise ValueError(f"luminous power must be >= 0 lm, got {lumens}")
  if spot_mode not in ("reflector", "absorber"):
    raise ValueError(f"spot_mode must be 'reflector' or 'absorber', got {spot_mode!r}")
  probe = dataclasses.replace(light, intensity=1.0)
  if spot_mode == "absorber" and light.shaping is not None:
    open_cone = MAX_ANGLE_DEG if light.kind in (LightKind.SPHERE, LightKind.CYLINDER) else 90.0
    probe = dataclasses.replace(probe, shaping=dataclasses.replace(light.shaping, cone_angle=open_cone, cone_softness=0.0))
  flux = luminous_flux(probe)
  if not flux > 0.0:
    raise ValueError(f"{light.path or 'light'}: emits no light, its power cannot be set")
  return dataclasses.replace(light, intensity=lumens / flux)


def _facing_factor(pf: _PointForm, distance: float) -> float:
  """Illuminance [lux] at `distance` [m] on the emission axis of the record at intensity 1, on a
  surface facing it: exact for spheres (I / d^2), disks (I / (r^2 + d^2)) and rects (the
  rectangle's on-axis form factor), the point form I / d^2 otherwise; times the cone / profile
  toward that point."""
  angular = float(_relative(pf, np.array([[0.0, 0.0, -1.0]]))[0])
  d2 = distance * distance
  if pf.disk_radius > 0.0:
    return pf.luminous * angular / (pf.disk_radius * pf.disk_radius + d2)
  w, h = pf.rect
  if w > 0.0 and h > 0.0:
    a, b = 0.5 * w / distance, 0.5 * h / distance
    ra, rb = math.sqrt(1.0 + a * a), math.sqrt(1.0 + b * b)
    form = 2.0 * (a / ra * math.atan(b / ra) + b / rb * math.atan(a / rb))
    # Radiance x the form factor's solid-angle term; luminous = L w h.
    return pf.luminous / (w * h) * form * angular
  return pf.luminous * angular / d2


def with_illuminance(light: Light, lux: float, distance: float) -> Light:
  """`light` with the intensity giving `lux` on a surface facing it `distance` [m] along its
  emission axis (local -Z; see `_facing_factor`).

  Raises:
    ValueError: for negative lux, a distance <= 0 (or inside a sphere), a distant light, or a
      light that sends nothing along its axis.
  """
  if not lux >= 0.0:
    raise ValueError(f"illuminance must be >= 0 lux, got {lux}")
  if light.kind not in (LightKind.SPHERE, LightKind.RECT, LightKind.DISK, LightKind.CYLINDER):
    raise ValueError(f"a {light.kind.name.lower()} light's intensity is its illuminance already")
  if not distance > (light.radius if light.kind == LightKind.SPHERE else 0.0):
    raise ValueError(f"the reference distance must be > 0 m (outside a sphere), got {distance}")
  e = _facing_factor(_point_form(dataclasses.replace(light, intensity=1.0)), distance)
  if not e > 0.0:
    raise ValueError(f"{light.path or 'light'}: sends no light along its axis")
  return dataclasses.replace(light, intensity=lux / e)
