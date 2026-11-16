"""Light authoring units (`scene/lighting/photometry.py`): flux of each light kind, lumens with
reflector / absorber spots, watts, illuminance targets, colour temperatures and spectra."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from raywarp.scene.lighting import photometry
from raywarp.scene.model.lights import Light, Shaping
from raywarp.scene.model.types import LightKind

POINT = Light(kind=LightKind.SPHERE, radius=0.0, treat_as_point=True)


@pytest.mark.parametrize(
  "light,flux",
  [
    (POINT, 4.0 * math.pi),
    (Light(kind=LightKind.SPHERE, radius=0.1), 4.0 * math.pi**2 * 0.01),  # pi L x 4 pi r^2
    (Light(kind=LightKind.RECT, width=0.5, height=0.2), math.pi * 0.1),  # pi L A
    (Light(kind=LightKind.DISK, radius=0.3), math.pi * math.pi * 0.09),
    (Light(kind=LightKind.CYLINDER, radius=0.05, length=1.0), math.pi * 2.0 * math.pi * 0.05),
    (Light(kind=LightKind.RECT, width=1.0, height=1.0, normalize=True), math.pi),  # radiance / area
    (dataclasses.replace(POINT, exposure=1.0, color=(1.0, 0.0, 0.0)), 4.0 * math.pi * 2.0 * 0.2126),
  ],
)
def test_flux_of_each_kind(light, flux):
  """Point-form flux = the area form's (pi L A ...), times exposure and colour luminance."""
  assert photometry.luminous_flux(light) == pytest.approx(flux, rel=1e-4)


@pytest.mark.parametrize("cone", [10.0, 30.0, 60.0, 120.0])
def test_lumens_reflector_and_absorber_spots(cone):
  """Reflector: I = flux / (2 pi (1 - cos cone)) for a hard cone; absorber: I = flux / 4 pi."""
  spot = dataclasses.replace(POINT, shaping=Shaping(cone_angle=cone))
  hard = 2.0 * math.pi * (1.0 - math.cos(math.radians(cone)))
  assert photometry.with_power(spot, 1000.0).intensity == pytest.approx(1000.0 / hard, rel=3e-3)
  assert photometry.with_power(spot, 1000.0, spot_mode="absorber").intensity == pytest.approx(
    1000.0 / (4.0 * math.pi), rel=1e-6
  )
  # A soft cone: the flux of the smooth falloff is kept exactly.
  soft = dataclasses.replace(spot, shaping=Shaping(cone_angle=cone, cone_softness=0.5))
  assert photometry.luminous_flux(photometry.with_power(soft, 1000.0)) == pytest.approx(1000.0, rel=1e-6)


def test_watts_and_errors():
  """Efficacy or efficiency x 683; invalid inputs raise."""
  assert photometry.watts_to_lumens(10.0, efficacy=90.0) == 900.0
  assert photometry.watts_to_lumens(10.0, efficiency=0.1) == pytest.approx(683.0)
  with pytest.raises(ValueError, match="exactly one"):
    photometry.watts_to_lumens(10.0)
  with pytest.raises(ValueError, match="lux"):
    photometry.luminous_flux(Light(kind=LightKind.DISTANT))
  with pytest.raises(ValueError, match="no light"):
    photometry.with_power(dataclasses.replace(POINT, color=(0.0, 0.0, 0.0)), 10.0)


def _numeric_illuminance(light: Light, distance: float) -> float:
  """On-axis illuminance of a disk / rect of radiance 1 by quadrature over its surface."""
  n = 400
  if light.kind == LightKind.DISK:
    r = (np.arange(n) + 0.5) / n * light.radius
    rings = 2.0 * math.pi * r * (light.radius / n)
    return float((rings * distance**2 / (r * r + distance**2) ** 2).sum())
  x = (np.arange(n) + 0.5) / n * light.width - 0.5 * light.width
  y = (np.arange(n) + 0.5) / n * light.height - 0.5 * light.height
  xx, yy = np.meshgrid(x, y)
  da = light.width * light.height / n / n
  return float((da * distance**2 / (xx * xx + yy * yy + distance**2) ** 2).sum())


@pytest.mark.parametrize("light", [Light(kind=LightKind.DISK, radius=0.5), Light(kind=LightKind.RECT, width=1.2, height=0.4)])
@pytest.mark.parametrize("distance", [0.3, 1.0, 5.0])
def test_illuminance_targets_are_exact_on_axis(light, distance):
  """Disk and rect illuminance at a distance: exact (near field too)."""
  solved = photometry.with_illuminance(light, 250.0, distance)
  assert solved.intensity * _numeric_illuminance(light, distance) == pytest.approx(250.0, rel=2e-4)
  point = photometry.with_illuminance(POINT, 250.0, distance)
  assert point.intensity == pytest.approx(250.0 * distance**2)


def test_colour_temperatures_and_spectra():
  """D65 is the Rec.709 white; Krystek's locus agrees with UsdLux's blackbody within 1 %; a flat
  spectrum is near equal-energy white; all of luminance 1."""
  d65 = photometry.cct_color(6504.0, "daylight")
  np.testing.assert_allclose(d65, 1.0, atol=3e-3)
  for kelvin in (2700.0, 4000.0, 6500.0):
    planck = np.array(photometry.cct_color(kelvin, "planckian"))
    np.testing.assert_allclose(planck, photometry.cct_color(kelvin, "blackbody"), rtol=0.03)
    assert float(planck @ np.array(photometry.REC709_Y)) == pytest.approx(1.0)
  flat = np.array(photometry.spectrum_color(np.array([360.0, 830.0]), np.array([1.0, 1.0])))
  assert float(flat @ np.array(photometry.REC709_Y)) == pytest.approx(1.0) and flat.max() / flat.min() < 1.4
  with pytest.raises(ValueError, match="daylight"):
    photometry.cct_color(2000.0, "daylight")


@pytest.mark.parametrize(
  "watts,kw",
  [
    (10.0, {"efficiency": 1.5}),
    (10.0, {"efficiency": -0.1}),
    (10.0, {"efficacy": math.inf}),
    (10.0, {"efficacy": math.nan}),
    (math.inf, {"efficacy": 90.0}),
    (-1.0, {"efficacy": 90.0}),
    (10.0, {"efficacy": 90.0, "efficiency": 0.1}),
  ],
)
def test_watts_to_lumens_rejects_impossible_inputs(watts: float, kw: dict[str, float]):
  """An efficiency above 1 (more than 683 lm/W), a negative or non-finite value, or both factors
  raise instead of returning a flux."""
  with pytest.raises(ValueError, match="watts_to_lumens"):
    photometry.watts_to_lumens(watts, **kw)


@pytest.mark.parametrize(
  "nm,values",
  [([400.0, 500.0, 600.0], [1.0, math.nan, 1.0]), ([400.0, math.inf], [1.0, 1.0]), ([400.0, 500.0], [1.0, math.inf])],
)
def test_spectrum_with_non_finite_samples_is_rejected(nm: list[float], values: list[float]):
  """A NaN / inf wavelength or value is named as such (not as a spectrum without luminance)."""
  with pytest.raises(ValueError, match="finite"):
    photometry.spectrum_color(np.array(nm), np.array(values))


def test_planckian_outside_its_fit_range_is_clamped_with_a_warning():
  """A planckian temperature outside Krystek's fit range uses the nearest end and says so."""
  with pytest.warns(UserWarning, match="clamped to 15000.0 K"):
    hot = photometry.cct_color(40000.0, "planckian")
  assert hot == pytest.approx(photometry.cct_color(15000.0, "planckian"))
