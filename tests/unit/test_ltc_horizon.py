"""Reject physically invisible emitters without changing crossing/above LTC terms."""

from typing import Literal

import numpy as np
import pytest
import warp as wp

from raywarp.core.shade.environment.ibl import dfg_texture
from raywarp.core.shade.lights import area, ltc, punctual
from raywarp.core.shade.lights.area import SHAPE_DISK, SHAPE_RECT, AreaLight, local_shape, shape_integral
from raywarp.core.shade.lights.ltc import Frame
from raywarp.core.shade.materials import material
from raywarp.core.shade.materials.material import Lobes

_frame = ltc.make_frame_func(specular_curve=False, clearcoat=True)
_lobes = material.make_shading_lobes(specular_curve=False, clearcoat=True, specular_layers=False)
_coated = area.make_eval_area_light(clearcoat=True)
_plain = area.make_eval_area_light(clearcoat=False)


@wp.func
def form_factor(
  # Model:
  light: AreaLight,
  # In:
  point: wp.vec3,
  m: wp.mat33,
  fr: Frame,
) -> float:
  """Clamped-cosine integral of the light through the LTC M^-1 `m` (local frame `fr`)."""
  return shape_integral(local_shape(light, point, fr), m)


# Frozen production bodies from core/shade/lights/area.py at
# dd65770ea868b79bdd394b50196a85917f9364cd. Only function names/call targets
# change; the factory-local coat body is dedented. Retain guards and operation order.
@wp.func(inline=True)
def _old_range_window(
  # Model:
  light: AreaLight,
  # In:
  point: wp.vec3,
) -> float:
  """The squared range window of the light at `point` (the punctual lights' window at the distance
  to the centre: light-grid consistent); 0 behind a one-sided emitter (rect, disk)."""
  if light.shape == wp.static(SHAPE_RECT) or light.shape == wp.static(SHAPE_DISK):
    if wp.dot(point - light.center, light.normal) <= 0.0:
      return 0.0
  d2 = wp.length_sq(light.center - point)
  r = wp.max(light.range, 1.0e-6)
  factor = d2 / (r * r)
  window = wp.clamp(1.0 - factor * factor, 0.0, 1.0)
  return window * window


@wp.func
def _old_eval_area_light(
  # Model:
  light: AreaLight,
  # In:
  fr: Frame,
  lobes: Lobes,
  point: wp.vec3,
) -> wp.vec3:
  """Unshadowed radiance of an area light at `point` towards the frame's view."""
  window = _old_range_window(light, point)
  if window <= 0.0:
    return wp.vec3(0.0)
  out = fr.spec * form_factor(light, point, fr.minv, fr)
  # No diffuse lobe (metals, specular-only translucent layers): no diffuse form factor.
  if wp.length_sq(lobes.diffuse_color) > 0.0:
    out = lobes.diffuse_color * form_factor(light, point, wp.identity(3, dtype=float), fr) + out
  return wp.cw_mul(out, light.radiance) * window


@wp.func
def _old_eval_coated(
  # Model:
  light: AreaLight,
  # In:
  fr: Frame,
  lobes: Lobes,
  point: wp.vec3,
) -> wp.vec3:
  """`eval_area_light` plus the clearcoat lobe's LTC term (the base attenuated by the coat's
  albedo)."""
  out = _old_eval_area_light(light, fr, lobes, point)
  if lobes.coat > 0.0:
    window = _old_range_window(light, point)
    if window > 0.0:
      coat = fr.coat_spec * form_factor(light, point, fr.coat_minv, fr) * window
      out = out * (1.0 - lobes.coat_albedo) + light.radiance * coat
  return out


@wp.kernel
def _support(
  # Model:
  emitters: wp.array[area.AreaLight],
  # Out:
  visible: wp.array[int],
):
  i = wp.tid()
  visible[i] = int(area.above_horizon(emitters[i], wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0)))


@wp.kernel
def _evaluate(
  # Model:
  lit: punctual.Lights,
  dfg: wp.Texture2D,
  emitters: wp.array[area.AreaLight],
  # In:
  roughness: float,
  nov: float,
  # Out:
  out: wp.array2d[float],
):
  i = wp.tid()
  a = emitters[i]
  s = material.Surface()
  s.base_color = wp.vec3(0.6)
  s.alpha = 1.0
  s.perceptual_roughness = roughness
  s.reflectance = 0.5
  s.occlusion = 1.0
  s.f90 = wp.vec3(1.0)
  s.clearcoat = 0.5
  s.clearcoat_roughness = roughness
  lobes = _lobes(dfg, s, nov)
  n = wp.vec3(0.0, 0.0, 1.0)
  view = wp.vec3(wp.sqrt(1.0 - nov * nov), 0.0, nov)
  fr = _frame(lit, lobes, n, view, nov)
  p = wp.vec3(0.0)
  previous = _old_eval_area_light(a, fr, lobes, p)
  previous_coat = _old_eval_coated(a, fr, lobes, p)
  diffuse = form_factor(a, p, wp.identity(3, dtype=float), fr)
  out[i, 0] = _plain(a, fr, lobes, p)[0]
  out[i, 1] = previous[0]
  out[i, 2] = _coated(a, fr, lobes, p)[0]
  out[i, 3] = previous_coat[0]
  out[i, 4] = diffuse


def _emitter(shape: int, z: float, vertical: bool = True) -> area.AreaLight:
  a = area.AreaLight()
  a.shape = shape
  a.center = wp.vec3(-1.0, 0.0, z)
  a.normal = wp.vec3(1.0, 0.0, 0.0) if vertical else wp.vec3(0.0, 0.0, 1.0)
  a.axis_x = wp.vec3(0.0, 0.0, 1.0) if vertical else wp.vec3(1.0, 0.0, 0.0)
  a.axis_y = wp.vec3(0.0, -1.0, 0.0) if vertical else wp.vec3(0.0, 1.0, 0.0)
  a.half = wp.vec2(0.5, 0.3)
  a.radius = 0.25
  a.radiance = wp.vec3(1.0)
  a.range = 1000.0
  return a


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request: pytest.FixtureRequest) -> wp.Device:
  name = str(request.param)
  if name != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  return wp.get_device(name)


def test_shape_support_below_tangent_crossing(device: wp.Device) -> None:
  # Analytic support of unit-normal horizontal/vertical primitives. Tangency has zero measure.
  shapes = (area.SHAPE_RECT, area.SHAPE_DISK, area.SHAPE_SPHERE, area.SHAPE_TUBE)
  emitters: list[area.AreaLight] = []
  expected: list[int] = []
  for vertical, extents in ((True, (0.5, 0.25, 0.25, 0.5)), (False, (0.0, 0.0, 0.25, 0.25))):
    for shape, extent in zip(shapes, extents, strict=True):
      for offset in (-0.125, 0.0, 0.125):
        emitters.append(_emitter(shape, -extent + offset, vertical))
        expected.append(int(offset > 0))
  # Oblique axes exercise projected disk/cylinder radii, not just axis-aligned extents.
  # The emitter's x axis has z=.8, so radial cylinder support is .25*.6.
  for shape, extent in zip(shapes, (0.4, 0.2, 0.25, 0.55), strict=True):
    for offset in (-0.125, 0.125):
      a = _emitter(shape, -extent + offset)
      a.axis_x = wp.vec3(0.6, 0.0, 0.8)
      a.axis_y = wp.vec3(0.0, -1.0, 0.0)
      a.normal = wp.vec3(0.8, 0.0, -0.6)
      emitters.append(a)
      expected.append(int(offset > 0))
  result = wp.zeros(len(emitters), dtype=int, device=device)
  wp.launch(_support, len(emitters), inputs=[wp.array(emitters, dtype=area.AreaLight, device=device), result], device=device)
  np.testing.assert_array_equal(result.numpy(), expected)


def _values(emitters: list[area.AreaLight], device: wp.Device) -> np.ndarray[tuple[int, Literal[5]], np.dtype[np.float32]]:
  lit = punctual.Lights()
  lit.ltc_minv, lit.ltc_mag = ltc.ltc_textures(device)
  result: wp.array[wp.float32, [int, 5]] = wp.zeros((len(emitters), 5), dtype=wp.float32, device=device)
  wp.launch(
    _evaluate,
    len(emitters),
    inputs=[lit, dfg_texture(device), wp.array(emitters, dtype=area.AreaLight, device=device), 0.25, 1.0 / 17.0, result],
    device=device,
  )
  return np.asarray(result.numpy(), dtype=np.float32).reshape((-1, 5))


def test_actual_lut_below_horizon_specular_and_coat(device: wp.Device) -> None:
  emitters = [_emitter(area.SHAPE_RECT, -0.1, False), _emitter(area.SHAPE_DISK, -0.1, False)]
  values = _values(emitters, device)
  assert np.all(values[:, 1] > 0.0), "Original fitted LTC must expose the below-horizon leakage"
  assert np.all(values[:, 3] > 0.0), "Coat shares the same transformed-horizon issue"
  np.testing.assert_array_equal(values[:, [0, 2, 4]], 0.0)
  # The curved-shape support guards also reach both actual evaluator lobes. Their old LTC
  # silhouette approximations need not leak in this particular case; physical zero is decisive.
  curved = _values([_emitter(shape, -0.625) for shape in (area.SHAPE_SPHERE, area.SHAPE_TUBE)], device)
  np.testing.assert_array_equal(curved[:, [0, 2]], 0.0)


def test_crossing_and_above_keep_existing_ltc(device: wp.Device) -> None:
  emitters = [
    _emitter(shape, z)
    for shape in (area.SHAPE_RECT, area.SHAPE_DISK, area.SHAPE_SPHERE, area.SHAPE_TUBE)
    for z in (-0.125, 1.0)
  ]
  values = _values(emitters, device)
  assert np.all(np.isfinite(values))
  assert np.all(values[:, 1] > 0.0)
  # The same arithmetic as the old evaluator, compiled apart: CUDA's fused multiply-adds may round
  # the two differently (2 ulp seen on L4); the CPU build is exact.
  np.testing.assert_array_max_ulp(values[:, 0], values[:, 1], maxulp=4)
  np.testing.assert_array_max_ulp(values[:, 2], values[:, 3], maxulp=4)
