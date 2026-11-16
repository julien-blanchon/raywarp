"""F2 area lights: LTC integrals, LTC vs Monte Carlo, the switch, host lights, emissive panels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp
from conftest import render_host, required

import raywarp as rw
from raywarp._base.types import LightShape, LightType
from raywarp.core.pipeline.upload import lights as host_lights
from raywarp.core.shade.environment.ibl import dfg_texture
from raywarp.core.shade.lights import area as A
from raywarp.core.shade.lights import ltc, punctual
from raywarp.core.shade.materials import brdf, material
from raywarp.scene.lighting import emissive


def _panel_lights(host: rw.Scene) -> host_lights.DerivedLights:
  """The derived light table of `host` plus its emissive panel lights."""
  lit = emissive.add_panel_lights(host)
  return host_lights.derive_lights(lit.lights)


# The plain LTC frame (F90 = 1, no coat) and evaluator.
_make_frame = ltc.make_frame_func(specular_curve=False, clearcoat=False)
_eval_area = A.make_eval_area_light(clearcoat=False)


def _polygon_integral(poly: np.ndarray) -> float:
  """Exact clamped-cosine integral of a polygon seen from the origin (clipped to z >= 0)."""
  out = []
  for k in range(len(poly)):
    a, b = poly[k], poly[(k + 1) % len(poly)]
    if a[2] >= 0:
      out.append(a)
    if (a[2] >= 0) != (b[2] >= 0):
      out.append(a + a[2] / (a[2] - b[2]) * (b - a))
  if len(out) < 3:
    return 0.0
  p = np.array(out) / np.linalg.norm(np.array(out), axis=1, keepdims=True)
  s = 0.0
  for k in range(len(p)):
    c = np.cross(p[k], p[(k + 1) % len(p)])
    s += np.arccos(np.clip(p[k] @ p[(k + 1) % len(p)], -1, 1)) * c[2] / max(math.sqrt(float(c @ c)), 1e-12)
  return abs(s) / (2 * np.pi)


# The lobes of the plain metallic-roughness model (no specular curve, no clearcoat).
_shading_lobes = material.make_shading_lobes(specular_curve=False, clearcoat=False, specular_layers=False)


@wp.kernel
def _quads(q: wp.array2d[wp.vec3], out: wp.array[float]):
  i = wp.tid()
  out[i] = ltc.quad_integral(q[i, 0], q[i, 1], q[i, 2], q[i, 3])


def test_quad_integral_clips_exactly():
  rng = np.random.default_rng(0)
  quads = []
  for _ in range(64):
    c = rng.uniform([-1, -1, -0.5], [1, 1, 1.5])
    ex = rng.normal(size=3)
    ex /= np.linalg.norm(ex, keepdims=True)
    ey = np.cross(ex, rng.normal(size=3))
    ey /= np.linalg.norm(ey, keepdims=True)
    w, h = rng.uniform(0.2, 1.0, 2)
    quads.append([c - ex * w - ey * h, c + ex * w - ey * h, c + ex * w + ey * h, c - ex * w + ey * h])
  quads = np.array(quads, np.float32)
  out = wp.zeros(len(quads), dtype=float, device="cpu")
  wp.launch(_quads, dim=len(quads), inputs=[wp.array(quads, dtype=wp.vec3, device="cpu"), out], device="cpu")
  ref = [_polygon_integral(q.astype(np.float64)) for q in quads]
  np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)


def test_quad_integral_of_a_panel_at_the_receiver_is_exact():
  """A panel crossing the horizon millimetres in front of the receiver (glossy receivers next to
  a rect light, after the LTC transform): the two horizon crossings are nearly opposite, where the
  edge fit's 1 / sin(theta) lost float32 precision (off by 0.08 here)."""
  quads = []
  for d in (1.0e-4, 3.0e-4, 1.0e-3, 0.0025):
    for tilt in (0.0, 0.3):
      quads.append([[-0.5, d, -0.5 + tilt], [0.5, d, -0.5 - tilt], [0.5, d, 0.5], [-0.5, d, 0.5]])
  quads = np.array(quads, np.float32)
  out = wp.zeros(len(quads), dtype=float, device="cpu")
  wp.launch(_quads, dim=len(quads), inputs=[wp.array(quads, dtype=wp.vec3, device="cpu"), out], device="cpu")
  ref = [_polygon_integral(q.astype(np.float64)) for q in quads]
  np.testing.assert_allclose(out.numpy(), ref, atol=2e-3)


@wp.kernel
def _bulbs(radius: wp.array[float], out: wp.array[float]):
  i = wp.tid()
  # A sphere of radius r at distance 1, 30 degrees off the normal: its facing disk (`facing_axes`).
  c = wp.vec3(0.5, 0.0, 0.8660254)
  r = radius[i]
  rd = r / wp.sqrt(1.0 - r * r)
  out[i] = ltc.ellipse_integral(c, wp.vec3(0.0, 1.0, 0.0) * rd, wp.cross(wp.vec3(0.0, 1.0, 0.0), c) * rd)


def test_small_ellipses_take_the_small_source_limit():
  """GPU-3: the ellipse cubic is ill-conditioned for small sources (CUDA's fast math: a 2 cm bulb
  at 1 m was 4.9 % bright). Under ltc.SMALL_ELLIPSE the form factor is the small-source limit,
  within SMALL_ELLIPSE (relative) of a sphere's exact cos(theta) sin^2(alpha), on both sides of the
  switch."""
  radius = np.array([0.01, 0.02, 0.05, 0.0625, 0.0640, 0.07, 0.1, 0.3], np.float32)
  out = wp.zeros(len(radius), dtype=float, device="cpu")
  wp.launch(_bulbs, dim=len(radius), inputs=[wp.array(radius, device="cpu"), out], device="cpu")
  exact = 0.8660254 * radius.astype(np.float64) ** 2
  np.testing.assert_allclose(out.numpy(), exact, rtol=ltc.SMALL_ELLIPSE + 1e-3)


@wp.kernel
def _ltc_vs_mc(
  lit: punctual.Lights, dfg: wp.Texture2D, shape: int, view: wp.vec3, rough: float, nsamp: int, out: wp.array[float]
):
  a = A.AreaLight()
  a.shape = shape
  a.center = wp.vec3(0.3, 0.2, 1.0)
  a.normal = wp.normalize(wp.vec3(-0.2, 0.1, -1.0))
  a.axis_x = wp.normalize(wp.cross(a.normal, wp.vec3(0.0, 1.0, 0.0)))
  a.axis_y = wp.cross(a.normal, a.axis_x)
  a.half = wp.vec2(0.5, 0.3)
  a.radius = 0.3
  a.radiance = wp.vec3(1.0)
  a.range = 1.0e6
  n = wp.vec3(0.0, 0.0, 1.0)
  nov = wp.dot(n, view)
  s = material.Surface()
  s.base_color = wp.vec3(0.6)
  s.alpha = 1.0
  s.perceptual_roughness = rough
  s.reflectance = 0.5
  s.occlusion = 1.0
  s.f90 = wp.vec3(1.0)
  lobes = _shading_lobes(dfg, s, nov)
  out[0] = _eval_area(a, _make_frame(lit, lobes, n, view, nov), lobes, wp.vec3(0.0))[0]
  total = float(0.0)
  st = wp.rand_init(7)
  for _k in range(nsamp):
    u = wp.vec2(wp.randf(st), wp.randf(st))
    y = a.center + a.axis_x * (a.half[0] * (2.0 * u[0] - 1.0)) + a.axis_y * (a.half[1] * (2.0 * u[1] - 1.0))
    area = 4.0 * a.half[0] * a.half[1]
    if shape == A.SHAPE_DISK:
      r = a.radius * wp.sqrt(u[0])
      y = a.center + (a.axis_x * wp.cos(6.2831853 * u[1]) + a.axis_y * wp.sin(6.2831853 * u[1])) * r
      area = wp.pi * a.radius * a.radius
    d2 = wp.length_sq(y)
    l = y / wp.sqrt(d2)
    cos_e = wp.dot(a.normal, -l)
    nol = l[2]
    if cos_e > 0.0 and nol > 0.0:
      h = wp.normalize(l + view)
      f = brdf.specular_lobe_rgb(lobes.f0, wp.vec3(1.0), lobes.alpha, h[2], nov, nol, wp.dot(l, h))
      f = wp.cw_mul(f, lobes.energy_comp) + lobes.diffuse_color * brdf.diffuse_lambert()
      total += f[0] * nol * area * cos_e / d2
  out[1] = total / float(nsamp)


def test_ltc_matches_monte_carlo():
  dev = wp.get_device("cpu")
  lit = punctual.Lights()
  lit.ltc_minv, lit.ltc_mag = ltc.ltc_textures(dev)
  out = wp.zeros(2, dtype=float, device=dev)
  for shape in (A.SHAPE_RECT, A.SHAPE_DISK):
    for rough, cos_v in ((0.3, 0.9), (0.6, 0.5), (0.9, 0.7)):
      view = wp.vec3(-np.sqrt(1 - cos_v**2), 0.0, cos_v)
      wp.launch(_ltc_vs_mc, dim=1, inputs=[lit, dfg_texture(dev), shape, view, rough, 100000, out], device=dev)
      value, ref = out.numpy()
      assert abs(value - ref) < 0.05 * ref, (shape, rough, cos_v, value, ref)


XML = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <light type="point" pos="0 0 1" intensity="10" range="10" bulbradius="{r}" castshadow="true"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="sphere" pos="0.2 0 0.2" size="0.1"/>
    <camera name="cam" pos="0 -1.5 1" xyaxes="1 0 0 0 0.55 0.85"/>
  </worldbody>
</mujoco>
"""


def test_area_light_without_range_has_no_falloff(scene):
  """A MuJoCo point light of range 0 has no falloff (`punctual.eval_light`): its sphere (LTC) form
  windowed it to nothing (black with area_lights="ltc"). It stays a point light: "ltc" renders it
  as "point" does, with and without the light grid."""
  xml = XML.format(r=0.05).replace('range="10"', 'range="0"')
  out = {}
  for form in ("ltc", "point"):
    for grid in (False, True):
      lighting = rw.settings.Lights(area_lights=form, light_grid=grid, light_grid_cell=0.25)
      s = scene(xml, lighting=lighting, shadows=rw.settings.Shadows(samples=4), outputs=rw.Outputs.of("hdr")).render()
      assert not s.r.features.area_lights
      out[form, grid] = s.hdr().copy()
  point = out["point", False].mean()
  assert point > 0.0
  for key, hdr in out.items():
    assert abs(hdr.mean() / point - 1) < 0.05, key


def test_switch_compiles_in_only_with_area_lights(scene):
  s = scene(
    XML.format(r=0.05),
    lighting=rw.settings.Lights(area_lights="ltc"),
    shadows=rw.settings.Shadows(samples=4),
    outputs=rw.Outputs.of("hdr"),
  ).render()
  assert s.r.features.area_lights
  ltc = s.hdr()
  p = scene(
    XML.format(r=0.05),
    lighting=rw.settings.Lights(area_lights="point"),
    shadows=rw.settings.Shadows(samples=4),
    outputs=rw.Outputs.of("hdr"),
  ).render()
  assert not p.r.features.area_lights
  # A 5 cm bulb: the diffuse light matches the point light's to a few percent.
  assert np.isfinite(ltc).all() and abs(ltc.mean() / p.hdr().mean() - 1) < 0.05
  # No bulb radius: no area light, nothing compiled in.
  assert not scene(XML.format(r=0), lighting=rw.settings.Lights(area_lights="ltc")).r.features.area_lights


def test_fixed_area_shadows_are_noise_free(scene):
  """One shadow sample per frame: the light's centre (fixed targets), seed-independent."""
  auto = rw.settings.Shadows()
  fixed = scene(XML.format(r=0.2), lighting=rw.settings.Lights(area_lights="ltc"), shadows=auto, outputs=rw.Outputs.of("hdr"))
  assert fixed.r.features.area_lights
  np.testing.assert_array_equal(fixed.render(seed=1).hdr().copy(), fixed.render(seed=2).hdr())


def test_host_light_tables():
  b = rw.SceneBuilder()
  b.add_light(rw.Light(kind=rw.scene.LightKind.RECT, intensity=2, width=1.0, height=0.5))
  b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=1, radius=0.0))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3, angle=0.0))
  b.add_light(rw.Light(kind=rw.scene.LightKind.CYLINDER, intensity=1, radius=0.1, length=2.0))
  host = b.build()
  t, info = host_lights.light_arrays(host_lights.derive_lights(host.lights))
  # Rect: a Lambertian (cos theta) point light.
  assert t["light.type"][0].tolist() == [LightType.COSINE, LightType.POINT, LightType.DIRECTIONAL, LightType.TUBE]
  assert t["light.shape"][0].tolist() == [LightShape.RECT, LightShape.NONE, LightShape.NONE, LightShape.TUBE]
  np.testing.assert_allclose(t["light.strength"][0], [1.0, 1.0, 3.0, 0.4])  # cd = L x area; lux
  np.testing.assert_allclose(t["light.size"][0][[0, 3]], [[0.5, 0.25], [1.0, 0.0]])
  np.testing.assert_allclose(t["light.dir"][0][0], [0, 0, -1])
  np.testing.assert_allclose(t["light.dir"][0][3], [1, 0, 0])  # a tube's axis (its local X)
  assert "light.cone_inner" not in t
  assert info == host_lights.LightInfo(
    has_spot=True,
    has_area_lights=True,
    sun_disc=False,
    has_ies=False,
    visible_lights=False,  # no light flagged `visible_in_primary_ray`
  )


def test_emissive_panels():
  b = rw.SceneBuilder()
  glow = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0, emission_luminance=100.0, emission_color=(1, 1, 1))
  )
  b.add_instance("plane", size=(5, 5, 0))
  panel = b.add_instance("box", size=(0.6, 0.2, 0.004), xform=np.c_[np.eye(3), [0, 0, 2.5]], material=glow)
  quad = b.add_mesh(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32), np.array([[0, 1, 2], [0, 2, 3]]))
  up = b.add_instance("mesh", mesh=quad, xform=np.c_[np.eye(3), [2, 0, 0.5]], material=glow)
  ell = b.add_mesh(
    np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], np.float32),
    np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]),
  )
  b.add_instance("mesh", mesh=ell, xform=np.c_[np.eye(3), [-2, 0, 2.5]], material=glow)
  b.add_instance("sphere", size=(0.3, 0, 0), xform=np.c_[np.eye(3), [0, 2, 1]], material=glow)
  lit = _panel_lights(b.build())
  assert lit.instance.tolist() == [panel, panel, up]
  np.testing.assert_allclose(sorted([lit.width[0], lit.height[0]]), [0.4, 1.2], rtol=1e-5)
  # The (two-sided) box emits from both faces, each light in front of its face; the quad along
  # its normal.
  np.testing.assert_allclose(-lit.xform[0][:, 2], [0, 0, 1], atol=1e-6)
  assert lit.xform[0][2, 3] > 2.5 + 0.004
  np.testing.assert_allclose(-lit.xform[1][:, 2], [0, 0, -1], atol=1e-6)
  assert lit.xform[1][2, 3] < 2.5 - 0.004
  np.testing.assert_allclose(-lit.xform[2][:, 2], [0, 0, 1], atol=1e-6)
  np.testing.assert_allclose(lit.xform[2][:2, 3], [2.5, 0.5], atol=1e-6)
  np.testing.assert_allclose(lit.radiance[:2], [[100, 100, 100]] * 2)


def test_textured_emissive_panel_is_its_emitting_part():
  # A fixture: a 1 x 1 m quad whose emission mask (texture alpha) lights only its centre
  # 0.5 x 0.5 m (u, v in [0.25, 0.75]); a sign-like prop whose mask lights an L is no panel.
  b = rw.SceneBuilder()
  mask = np.zeros((4, 4, 4), np.uint8)
  mask[1:3, 1:3, 3] = 255
  tex = b.add_texture(mask, colorspace="linear")
  ref = rw.scene.TextureRef(tex, channel=3, scale=(2.0, 2.0, 2.0, 1.0))
  glow = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      specular_roughness=1.0,
      emission_luminance=100.0,
      emission_color=(1, 1, 1),
      textures=((rw.scene.Slot.EMISSION_COLOR, ref),),
    )
  )
  g = np.linspace(0.0, 1.0, 5, dtype=np.float32)
  pts = np.array([[x, y, 0.0] for y in g for x in g], np.float32)
  tris = np.array(
    [[r * 5 + c, r * 5 + c + 1, r * 5 + c + 6] for r in range(4) for c in range(4)]
    + [[r * 5 + c, r * 5 + c + 6, r * 5 + c + 5] for r in range(4) for c in range(4)]
  )
  uvs = pts[:, :2].copy()
  quad = b.add_mesh(pts, tris, uvs=uvs)
  fixture = b.add_instance("mesh", mesh=quad, xform=np.c_[np.eye(3), [0, 0, 2.5]], material=glow)
  ring = np.zeros((4, 4, 4), np.uint8)
  ring[[0, 0, 1], [0, 1, 0], 3] = 255
  prop = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      specular_roughness=1.0,
      emission_luminance=100.0,
      emission_color=(1, 1, 1),
      textures=((rw.scene.Slot.EMISSION_COLOR, rw.scene.TextureRef(b.add_texture(ring, colorspace="linear"), channel=3)),),
    ),
  )
  b.add_instance("mesh", mesh=quad, xform=np.c_[np.eye(3), [3, 0, 2.5]], material=prop)
  b.add_instance("plane", size=(5, 5, 0))
  lit = _panel_lights(b.build())
  assert lit.instance.tolist() == [fixture]
  np.testing.assert_allclose([lit.width[0], lit.height[0]], [0.5, 0.5], atol=1e-5)
  np.testing.assert_allclose(lit.xform[0][:2, 3], [0.5, 0.5], atol=1e-5)
  np.testing.assert_allclose(lit.radiance[0], [200, 200, 200], rtol=1e-5)  # factor x mean texel (2 x 1)


def _panel_ceiling(size: float, height: float, n: int, step: float) -> rw.Scene:
  """A matte floor under an n x n grid of square rect lights `size` wide, `step` apart, `height`
  above it (the same total power per panel area)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_weight=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(8.0, 8.0, 0.05), material=m, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
  c = (n - 1) / 2
  for i in range(n):
    for j in range(n):
      at = np.c_[np.eye(3), [(i - c) * step, (j - c) * step, height]]
      b.add_light(rw.Light(kind=rw.scene.LightKind.RECT, intensity=2.0e3 * size * size, width=size, height=size, xform=at))
  return b.build()


@pytest.mark.parametrize(("size", "height", "n", "step"), [(2.0, 0.5, 3, 2.4), (1.2, 0.3, 5, 1.4)])
def test_capped_area_lights_remainder_stays_close_to_exact(size: float, height: float, n: int, step: float) -> None:
  """Under a ceiling of large, near rect lights, the default cap (`lighting.max_exact` 4: the
  strongest by the point form, the rest through the diffuse lobe unshadowed) stays within 2 % of
  every light shaded by LTC, pixel by pixel: the near panels, whose point form overestimates
  them most, rank into the exact set; the far ones' point form is close to their form factor."""
  host = _panel_ceiling(size, height, n, step)
  fovy = 2.0 * math.degrees(math.atan(0.5 * n * step / 4.0))
  out = {}
  for cap in (None, 4):
    s = {"lighting": rw.settings.Lights(area_lights="ltc", max_exact=cap), "shadows": rw.settings.Shadows(enabled=False)}
    r = render_host(host, (0.0, 0.0, 4.0 + height), fovy=fovy, res=32, settings=s)
    assert r.features.light_cap == (cap or 0)
    out[cap] = required(r.outputs.hdr).numpy()[0, 0, ..., :3].mean(-1)
  ratio = out[4] / out[None]
  assert 0.98 < ratio.min() and ratio.max() < 1.005, (ratio.min(), ratio.max())
