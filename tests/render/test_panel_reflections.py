"""A glossy floor reflects emissive panels that are also lights (`reflections.panel_light`): the
reflection ray skips a panel's emission because the shading point's light term integrates the
panel; a panel whose light the light cap (`lighting.max_exact`) leaves to the diffuse remainder
reflected black (balanced: black rectangles under the panels): the reflection hit now skips the
emission only of a panel whose light the shading point shaded exactly
(`light_sources.make_exact_test`). CPU."""

from __future__ import annotations

import numpy as np
from conftest import DOWN, render_host, required

import raywarp as rw

RES = 96
FOVY = 60.0
XS = (-0.6, 0.0, 0.6)


def _pixel(x: float, y: float, z: float) -> tuple[int, int]:
  """(row, col) of a point in front of the DOWN camera at the origin."""
  f = (RES / 2) / np.tan(np.radians(FOVY / 2))
  return int(RES / 2 - f * y / -z), int(RES / 2 + f * x / -z)


def _reflections(settings: dict[str, object]) -> np.ndarray:
  """(3,) mean radiance of the floor's mirror images of three coloured panels (lights: two per
  two-sided box)."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.05, 0.05, 0.05), specular_roughness=0.1))
  b.add_instance("plane", size=(20, 20, 0), xform=np.c_[np.eye(3), [0, 0, -4]], material=floor)
  for x, c in zip(XS, ((1, 0.2, 0.2), (0.2, 1, 0.2), (0.2, 0.2, 1)), strict=True):
    glow = b.add_material(rw.Material(base_color=(0, 0, 0), specular_weight=0.0, emission_luminance=200.0, emission_color=c))
    b.add_instance("box", size=(0.15, 0.25, 0.005), xform=np.c_[np.eye(3), [x, 0.4, -1.5]], material=glow)
  s = {"lighting.area_lights": "ltc", "reflections.enabled": True, "lighting.max_exact": None, **settings}
  r = render_host(b.build(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=RES, settings=s)
  img = required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2)
  rows = [_pixel(x, 0.4, -6.5) for x in XS]
  return np.array([img[p[0] - 1 : p[0] + 2, p[1] - 1 : p[1] + 2].mean() for p in rows])


def test_panels_reflect_when_every_light_is_exact():
  """Uncapped, every panel's reflection is within 15 % of its glow seen as a plain emitter."""
  np.testing.assert_allclose(_reflections({}), _reflections({"lighting.emissive_lights": False}), rtol=0.15)


def test_capped_out_panels_still_reflect():
  """With one light shaded exactly, the panels whose lights are capped out still reflect (about
  their glow), not black (measured 0.146 vs 3.46)."""
  np.testing.assert_allclose(
    _reflections({"lighting.max_exact": 1}), _reflections({"lighting.emissive_lights": False}), rtol=0.25
  )


def _quad_glow(eye_z: float, quat: tuple[float, float, float, float], flags: int) -> float:
  """Mean radiance of a 1 m emissive quad facing +z (its winding), seen from `eye_z` on its axis
  (back faces not culled: the back face is hit; no environment)."""
  b = rw.SceneBuilder()
  points = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
  mesh = b.add_mesh(points, np.array([[0, 1, 2], [0, 2, 3]]))
  glow = b.add_material(rw.Material(base_color=(0, 0, 0), specular_weight=0.0, emission_luminance=100.0))
  b.add_instance("mesh", mesh=mesh, material=glow, flags=flags)
  s = {"surfaces.backface_culling": False, "lighting.emissive_lights": False}
  r = render_host(b.build(), (0.0, 0.0, eye_z), quat, fovy=10.0, res=8, settings=s)
  return float(required(r.outputs.hdr).numpy().mean())


def test_single_sided_emitters_emit_from_their_front_only():
  """A single-sided emissive quad glows seen from its front and is dark from behind (USD /
  Mitsuba: its panel light lights only the front side too); a double-sided one glows both ways."""
  up = (1.0, 0.0, 0.0, 0.0)  # the camera turned half a turn about x: looking along +z
  single = int(rw.scene.InstanceFlags.DEFAULT)
  double = single | int(rw.scene.InstanceFlags.DOUBLE_SIDED)
  front = _quad_glow(1.0, DOWN, single)
  assert front > 0.0
  assert _quad_glow(-1.0, up, single) < 1e-3 * front
  np.testing.assert_allclose(_quad_glow(-1.0, up, double), _quad_glow(1.0, DOWN, double), rtol=1e-3)


def _back_reflection(settings: dict[str, object]) -> float:
  """Mean radiance of a glossy floor's mirror image of a double-sided emissive quad (0.3 x 0.5 m,
  winding and panel light facing the camera, +z): the floor below sees its back."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.05, 0.05, 0.05), specular_roughness=0.1))
  b.add_instance("plane", size=(20, 20, 0), xform=np.c_[np.eye(3), [0, 0, -4]], material=floor)
  points = np.array([[-0.15, 0.15, -1.5], [0.15, 0.15, -1.5], [0.15, 0.65, -1.5], [-0.15, 0.65, -1.5]])
  mesh = b.add_mesh(points, np.array([[0, 1, 2], [0, 2, 3]]))
  glow = b.add_material(rw.Material(base_color=(0, 0, 0), specular_weight=0.0, emission_luminance=200.0))
  double = int(rw.scene.InstanceFlags.DEFAULT) | int(rw.scene.InstanceFlags.DOUBLE_SIDED)
  b.add_instance("mesh", mesh=mesh, material=glow, flags=double)
  s = {"lighting.area_lights": "ltc", "reflections.enabled": True, "lighting.max_exact": None, **settings}
  r = render_host(b.build(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=RES, settings=s)
  img = required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2)
  row, col = _pixel(0.0, 0.4, -6.5)
  return float(img[row - 1 : row + 2, col - 1 : col + 2].mean())


def test_a_double_sided_panel_reflects_from_behind_its_light():
  """USD panels are double-sided (both faces glow) with a one-sided panel light: a floor behind
  that light sees the panel's back in its mirror image at the panel's glow (the light term covers
  nothing there, so the reflection keeps the emission), not black (balanced: black floor
  rectangles under the reflections scene's panels)."""
  glow = _back_reflection({"lighting.emissive_lights": False})
  assert glow > 0.0
  np.testing.assert_allclose(_back_reflection({}), glow, rtol=0.15)
