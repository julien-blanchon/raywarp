"""Soft-shadow sampling: point-form emitters, fixed sample sets, zero-radius lights."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pytest
from conftest import caster_transmittance, disc_visible_fraction, render_host, required

import raywarp as rw
from raywarp.core.pipeline.platform import tuning


def _xform(pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
  """A translation as a 3 x 4 host transform."""
  return np.c_[np.eye(3), np.asarray(pos, np.float64)]


def _render(
  host: rw.Scene, eye: tuple[float, float, float], settings: Mapping[str, object], res: int = 4, seed: int = 0
) -> np.ndarray:
  """(res, res, 3) HDR radiance of a narrow camera at `eye` looking straight down."""
  r = render_host(host, eye, fovy=0.05, res=res, settings=settings, seed=seed)
  return required(r.outputs.hdr).numpy()[0, 0, ..., :3].copy()


def _ceiling_panel_scene() -> rw.Scene:
  """A floor and a ceiling slab (underside at z = 2) with a 0.5 m rect light 5 mm below it."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(6.0, 6.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))
  b.add_instance("box", size=(6.0, 6.0, 0.05), material=m, xform=_xform((0.0, 0.0, 2.05)))
  b.add_light(rw.Light(kind=rw.scene.LightKind.RECT, intensity=50.0, width=0.5, height=0.5, xform=_xform((0.0, 0.0, 1.995))))
  return b.build()


def test_point_form_rect_soft_shadows_do_not_hit_the_ceiling():
  """B41: a rect light in point form (lighting.area_lights="point") samples its own disc, in its
  plane
  under the ceiling: a floor point off to the side with nothing in between is fully lit, as with
  one hard shadow ray (a disc facing the point reached into the ceiling: ~half the rays blocked)."""
  host = _ceiling_panel_scene()
  for x in (0.0, 1.0, 2.0):
    hard = _render(
      host, (x, 0.0, 0.5), {"lighting": rw.settings.Lights(area_lights="point"), "shadows": rw.settings.Shadows(samples=1)}
    )
    soft = _render(
      host, (x, 0.0, 0.5), {"lighting": rw.settings.Lights(area_lights="point"), "shadows": rw.settings.Shadows(samples=8)}
    )
    assert hard.mean() > 0.0
    np.testing.assert_allclose(soft, hard, rtol=1e-4, err_msg=f"x={x}")


def _slab_scene(light: str) -> rw.Scene:
  """A floor under an opaque slab (x < 0) 2 m up; a 4 degree sun overhead or a 1 m rect light 3 m
  up."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(20.0, 20.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=m, xform=_xform((-5.0, 0.0, 2.0)))
  if light == "sun":
    b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=4.0, xform=_xform()))
  else:
    b.add_light(rw.Light(kind=rw.scene.LightKind.RECT, intensity=50.0, width=1.0, height=1.0, xform=_xform((0.0, 0.0, 3.0))))
  return b.build()


def test_fixed_sun_samples_are_shifted_per_pixel():
  """B44: 4 fixed sun-disc samples give every pixel the same R2 set at its own offset: across a
  patch of the penumbra the pixels differ (not one of 5 levels everywhere: 4 stacked hard
  shadows) and their mean is the disc's visible fraction; the image depends only on the seed."""
  host = _slab_scene("sun")
  fixed = {"shadows": rw.settings.Shadows(samples=4)}
  lit = _render(host, (5.0, 0.0, 0.5), fixed).mean()
  half = math.radians(2.0)
  for t in (-0.5, 0.0, 0.5):
    x = t * 2.0 * math.tan(half)  # the floor point whose disc centre is t radii past the edge
    v = _render(host, (x, 0.0, 0.5), fixed, res=24) / lit
    assert v.std() > 0.05, t
    assert abs(v.mean() - disc_visible_fraction(t)) < 0.04, (t, v.mean(), disc_visible_fraction(t))
  a, b = (_render(host, (0.0, 0.0, 0.5), fixed, res=8, seed=1) for _ in range(2))
  np.testing.assert_array_equal(a, b)


ZERO_RADIUS_XML = """
<mujoco>
  <visual><headlight active="0" ambient="0 0 0"/></visual>
  <worldbody>
    <light type="point" pos="0 0 2" intensity="20" range="10" bulbradius="0" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="sphere" pos="0 0 0.3" size="0.2"/>
    <geom type="box" pos="0.6 0.3 0.3" size="0.08 0.08 0.3"/>
    <camera name="cam" pos="0 -3 2.5" xyaxes="1 0 0 0 0.64 0.77"/>
  </worldbody>
</mujoco>
"""


def test_zero_radius_light_traces_one_shadow_ray(scene):
  """P6: a point light without a bulb radius traces one shadow ray whatever `shadow_samples`
  (its rays would all be the same): the hard shadow, at the cost of one sample. Counted with the
  performance counters (shadow rays per pixel), not timed."""

  def render(n: int) -> tuple[np.ndarray, float]:
    with tuning.override(counters=True):
      s = scene(
        ZERO_RADIUS_XML, cam_res=(32, 32), shadows=rw.settings.Shadows(samples=n), outputs=rw.Outputs.of("hdr")
      ).render()
    return s.hdr().copy(), s.r.counters()["shadow_rays"]

  hard, rays1 = render(1)
  soft, rays16 = render(16)
  np.testing.assert_allclose(soft, hard, atol=1e-6)
  assert 0.0 < rays1 <= 1.0 and rays16 == rays1, (rays1, rays16)  # one ray per lit pixel, not 16


def test_fixed_area_samples_are_shifted_per_pixel():
  """B44 for area lights (lighting.area_lights="ltc"): across a patch of the penumbra under a rect
  light,
  4 fixed samples average to the visible fraction of 64 random ones (stacked copies: one of 5
  levels per patch), and the image is deterministic per seed."""
  host = _slab_scene("rect")
  fixed = {"lighting": rw.settings.Lights(area_lights="ltc"), "shadows": rw.settings.Shadows(samples=4)}
  ref = {"lighting": rw.settings.Lights(area_lights="ltc"), "shadows": rw.settings.Shadows(samples=64)}
  for x in (-0.6, -0.3, 0.0, 0.3, 0.6):
    eye = (x, 0.0, 0.5)
    unshadowed = _render(
      host, eye, {"lighting": rw.settings.Lights(area_lights="ltc"), "shadows": rw.settings.Shadows(enabled=False)}, res=24
    ).mean()
    v = _render(host, eye, fixed, res=24).mean() / unshadowed
    want = _render(host, eye, ref, res=24).mean() / unshadowed
    assert abs(v - want) < 0.03, (x, v, want)
  a, b = (_render(host, (0.0, 0.0, 0.5), fixed, res=8, seed=1) for _ in range(2))
  np.testing.assert_array_equal(a, b)


def _floor_row(host: rw.Scene, shadows: rw.settings.Shadows, res: int = 64) -> np.ndarray:
  """Floor luminance along x in [-0.5, 0.5] (y = 0), seen straight down from 2 m."""
  fovy = 2.0 * math.degrees(math.atan(0.25))  # 1 m wide at 2 m
  settings = {"lighting": rw.settings.Lights(area_lights="point"), "shadows": shadows}
  r = render_host(host, (0.0, 0.0, 2.0), fovy=fovy, res=res, settings=settings)
  return required(r.outputs.hdr).numpy()[0, 0, res // 2, :, :3].mean(axis=-1)


def test_shadow_budget_is_continuous_where_the_traced_light_changes():
  """B49: two equal lights over a floor, one budgeted shadow ray (the stronger caster), the right
  light's shadow falling across x = 0 where they swap ranks. "ratio" scaled the untraced light by
  the traced one's visibility, so the floor jumped from lit to dark at x = 0."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(3.0, 3.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))
  # A post between the right light and the floor around x = 0.
  b.add_instance("box", size=(0.1, 0.5, 0.1), material=m, xform=_xform((0.5, 0.0, 0.6)))
  for x in (-1.0, 1.0):
    b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=5.0e4, radius=0.05, xform=_xform((x, 0.0, 1.2))))
  host = b.build()
  budget = _floor_row(host, rw.settings.Shadows(samples=1, max_casters=1))
  every = _floor_row(host, rw.settings.Shadows(samples=1, max_casters=None))
  # Away from the true shadow edges (where every-caster tracing is smooth) the budget's floor
  # must be smooth too: a ramp across the band, not the lit / dark step at x = 0.
  smooth = np.abs(np.diff(every)) < 0.05 * every.mean()
  assert np.abs(np.diff(budget))[smooth].max() < 0.3 * every.mean()


def test_shadow_budget_band_spreads_the_transition():
  """Phase A decision 2 (Warehouse fast streaks): three lights, two budgeted shadow rays,
  a post shadowing the right light around x = 0. The margin blend (the strongest untraced caster
  traced too, weighted by its margin) ramps the third caster in wherever it is weaker than the
  second: steps <= 2.1 % of the mean between neighbouring pixels and a 4.7 % mean error vs
  every caster traced (band 0.25: 6.2 % / 13 %)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(3.0, 3.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))
  b.add_instance("box", size=(0.1, 0.5, 0.1), material=m, xform=_xform((0.5, 0.0, 0.6)))
  for x, s in ((-1.0, 5.0e4), (1.0, 5.0e4), (0.0, 3.0e4)):
    b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=s, radius=0.05, xform=_xform((x, 0.0, 1.2))))
  host = b.build()
  budget = _floor_row(host, rw.settings.Shadows(samples=1, max_casters=2))
  every = _floor_row(host, rw.settings.Shadows(samples=1, max_casters=None))
  smooth = np.abs(np.diff(every)) < 0.05 * every.mean()
  assert np.abs(np.diff(budget))[smooth].max() < 0.03 * every.mean()
  assert np.abs(budget - every).mean() < 0.07 * every.mean()


def _pane_shadow(count: int) -> float:
  """Sunlit floor radiance under `count` stacked glass panes (boxes of opacity 0.2) over the open
  floor's."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      base_metalness=0.0,
      specular_roughness=0.5,
      geometry_opacity=0.2,
      ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
    )
  )
  b.add_instance("box", size=(6.0, 6.0, 0.05), material=floor, xform=_xform((0.0, 0.0, -0.05)))
  for k in range(count):
    b.add_instance("box", size=(1.0, 1.0, 0.02), material=glass, xform=_xform((-2.0, 0.0, 1.0 + 0.2 * k)))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0))
  host = b.build()
  settings = {"shadows": rw.settings.Shadows(translucent="transmittance")}
  return float(_render(host, (-2.0, 0.0, 0.5), settings).mean() / _render(host, (2.0, 0.0, 0.5), settings).mean())


def test_translucent_shadows_count_entering_faces():
  """B12: a glass pane (a closed box) attenuates the sun once by its transmittance (0.8, not 0.64
  per face pair), and at the layer limit (4 faces: double glazing) the shadow ray keeps the
  transmittance so far instead of blocking the sun (it went black)."""
  assert abs(_pane_shadow(1) - 0.8) < 0.01
  assert abs(_pane_shadow(2) - 0.8**2) < 0.01


def _curtain_shadow(light: rw.Light, translucent: str) -> float:
  """Floor radiance under a 0.5-opacity curtain (a sheet of blended material) between the floor and
  `light` above it, over the floor's without the curtain (the same point, the curtain hidden by
  placing it far away)."""

  def scene(curtain_z: float) -> rw.Scene:
    b = rw.SceneBuilder()
    floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
    cloth = b.add_material(
      rw.Material(
        base_color=(1.0, 1.0, 1.0),
        specular_roughness=1.0,
        geometry_opacity=0.5,
        ext=rw.scene.RaywarpExt(alpha_mode=rw.scene.AlphaMode.BLEND),
      )
    )
    b.add_instance("box", size=(6.0, 6.0, 0.05), material=floor, xform=_xform((0.0, 0.0, -0.05)))
    b.add_instance("box", size=(2.0, 2.0, 0.005), material=cloth, xform=_xform((0.0, 0.0, curtain_z)))
    b.add_light(light)
    return b.build()

  settings = {
    "shadows": rw.settings.Shadows(translucent=translucent, samples=1),  # pyrefly: ignore[bad-argument-type]
    "lighting.area_lights": "ltc",  # a rect light as an area light (its samples pass translucent casters)
  }
  under = _render(scene(1.0), (0.0, 0.0, 0.5), settings).mean()
  open_ = _render(scene(100.0), (0.0, 0.0, 0.5), settings).mean()
  return float(under / open_)


def test_local_lights_pass_translucent_casters():
  """A rect or point light behind a 0.5-opacity curtain lights the floor at its transmittance
  (base color x (1 - opacity) = 0.5) with translucent shadows, as a directional light does (the
  Bitterli bedroom's window lights behind its curtains were blocked: the room went dark): one
  terminating walk per shadow ray, the curtain's transmittance multiplied in; with
  `shadows.translucent="off"` the curtain is opaque."""
  point = rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=50.0, radius=0.0, xform=_xform((0.0, 0.0, 2.0)))
  rect = rw.Light(kind=rw.scene.LightKind.RECT, intensity=50.0, width=0.2, height=0.2, xform=_xform((0.0, 0.0, 2.0)))
  assert abs(_curtain_shadow(rect, "transmittance") - 0.5) < 0.03
  assert _curtain_shadow(rect, "off") < 0.02
  assert abs(_curtain_shadow(point, "transmittance") - 0.5) < 0.03
  assert _curtain_shadow(point, "off") < 0.02


@pytest.mark.parametrize("caster", ["pane", "sheet", "cylinder"])
def test_a_translucent_caster_attenuates_once(caster):
  """One rule on both backends (`tests/optix_backend` compares OptiX with this): a translucent
  caster attenuates a shadow ray once, its transmittance 0.6, whether a closed single-sided pane
  (two faces on the ray), a two-sided sheet or a solid cylinder."""
  assert caster_transmittance(caster, "software") == pytest.approx(0.6, rel=0.02)


def test_a_caster_of_no_light_takes_no_shadow_ray():
  """A light whose unshadowed term at the point is 0 (a black, specular-free floor: a zero BSDF)
  takes no budgeted caster slot and so no shadow ray (`shadow_budget`: a slot used to start at
  luminance -1, which 0 beat)."""
  b = rw.SceneBuilder()
  black = b.add_material(rw.Material(base_color=(0.0, 0.0, 0.0), specular_weight=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(3.0, 3.0, 0.05), material=black, xform=_xform((0.0, 0.0, -0.05)))
  for x in (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0):  # beyond the small light loop (every light traced)
    b.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=5.0e4, radius=0.05, xform=_xform((x, 0.0, 1.2))))
  settings = {"lighting": rw.settings.Lights(area_lights="point"), "shadows": rw.settings.Shadows(max_casters=1)}
  with tuning.override(counters=True):
    r = render_host(b.build(), (0.0, 0.0, 2.0), fovy=40.0, res=8, settings=settings)
  assert r.counters()["shadow_rays"] == 0.0


@pytest.mark.parametrize("half", [1.0e2, 1.0e4])
def test_a_huge_ground_quad_has_no_shadow_acne(half: float):
  """A 2-triangle ground quad of half size `half` [m] under a point light 1 m above it: every
  pixel lit (no shadow ray starting under the surface). Its hit points round by ~1e-3 m at
  1e4 m vertices, which the metre floor of the origin offset (1e-4 m) left under it: 42 of 256
  pixels black before the floor took the instance's vertex magnitude (`trace.ray_floors`)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), specular_roughness=1.0))
  quad = b.add_mesh([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], [[0, 1, 2], [0, 2, 3]])
  b.add_instance("mesh", mesh=quad, material=m, xform=_xform())
  b.add_light(rw.Light.point((0.0, 0.0, 1.0), intensity=10.0))
  r = render_host(b.build(), (0.0, 0.0, 2.0), fovy=60.0, res=16)
  hdr = required(r.outputs.hdr).numpy()[0, 0, ..., :3]
  assert (hdr.max(axis=-1) > 0.0).all(), int((hdr.max(axis=-1) <= 0.0).sum())


def _scaled_room(s: float) -> rw.Scene:
  """A floor, a cube and a post at length scale `s` (1: metres) under a sun and a point light of
  intensity scaled by s^2 (the same irradiance at the scaled distances)."""
  b = rw.SceneBuilder()
  m = b.add_material(rw.Material(base_color=(0.7, 0.7, 0.7), specular_roughness=0.6))
  b.add_instance("box", size=(2.0 * s, 2.0 * s, 0.05 * s), material=m, xform=_xform((0.0, 0.0, -0.05 * s)))
  b.add_instance("box", size=(0.2 * s, 0.2 * s, 0.2 * s), material=m, xform=_xform((0.3 * s, 0.0, 0.2 * s)))
  b.add_instance("box", size=(0.02 * s, 0.02 * s, 0.4 * s), material=m, xform=_xform((-0.4 * s, 0.3 * s, 0.4 * s)))
  b.add_light(rw.Light.sun((0.3, 0.2, -0.9), illuminance=2.0))
  b.add_light(rw.Light.point((-0.2 * s, -0.3 * s, 0.6 * s), intensity=0.5 * s * s))
  return b.build()


@pytest.mark.parametrize("area_lights", ["point", "ltc"])
def test_a_millimetre_and_a_kilometre_scene_render_as_the_metre_one(area_lights: Literal["point", "ltc"]):
  """The same room at 1 mm, 1 m and 1 km (every length scaled, the point light's intensity by
  s^2): the same image within 1 % mean (the offsets and falloff floors scale with the scene:
  `trace.ray_floors`, the light's range; a fixed 1e-4 m offset was a tenth of the millimetre
  room, a 1 cm falloff floor clamped its point light)."""
  out = {}
  for s in (1.0e-3, 1.0, 1.0e3):
    settings = {"lighting": rw.settings.Lights(area_lights=area_lights)}
    r = render_host(_scaled_room(s), (0.0, -1.5 * s, 1.2 * s), (0.383, 0.0, 0.0, 0.924), fovy=50.0, res=32, settings=settings)
    out[s] = required(r.outputs.hdr).numpy()[0, 0, ..., :3].mean(-1)
  base = out[1.0]
  for s in (1.0e-3, 1.0e3):
    rel = float(np.abs(out[s] - base).mean() / base.mean())
    assert rel < 0.01, (s, rel)
