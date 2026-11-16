"""Camera-visible area lights (`shade/lights/visible.py`): which lights camera rays see, and the
radiance they see, against an emissive surface of the same luminance (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DOWN, render_host, required

import raywarp as rw
from raywarp._base.types import LightShape
from raywarp.core.pipeline.upload import lights as host_lights
from raywarp.scene.lighting import emissive
from raywarp.scene.model.types import LightKind

RES = 64
FOVY = 60.0
# Emitting along +Z (toward a camera on the +Z side): local -Z is the emission direction.
TOWARD_CAMERA = np.diag(np.array([1.0, -1.0, -1.0]))


def _frame(rot: np.ndarray, pos: tuple[float, float, float]) -> np.ndarray:
  return np.c_[rot, pos].astype(np.float32)


def _pixel(x: float, y: float, z: float) -> tuple[int, int]:
  """(row, col) of a point in front of the DOWN camera at the origin (rows grow downward)."""
  f = (RES / 2) / np.tan(np.radians(FOVY / 2))
  return int(RES / 2 - f * y / -z), int(RES / 2 + f * x / -z)


def test_visible_lights_table():
  b = rw.SceneBuilder()
  seen = {"visible_in_primary_ray": True}
  b.add_light(rw.Light(kind=LightKind.RECT, intensity=2, width=1.0, height=0.5, **seen))
  b.add_light(rw.Light(kind=LightKind.SPHERE, intensity=1, radius=0.0, **seen))  # a point: nothing to see
  b.add_light(rw.Light(kind=LightKind.DISTANT, intensity=3, angle=0.5, **seen))
  b.add_light(rw.Light(kind=LightKind.CYLINDER, intensity=1, radius=0.1, length=2.0, **seen))
  b.add_light(rw.Light(kind=LightKind.DISK, intensity=1, radius=0.2, shaping=rw.scene.Shaping(cone_angle=30.0), **seen))
  b.add_light(rw.Light(kind=LightKind.SPHERE, intensity=1, radius=0.2, treat_as_point=True, **seen))
  b.add_light(
    rw.Light(kind=LightKind.CYLINDER, intensity=1, radius=0.1, length=1.0, shaping=rw.scene.Shaping(cone_angle=30.0), **seen)
  )
  b.add_light(rw.Light(kind=LightKind.RECT, intensity=2, width=1.0, height=0.5))  # not flagged: unseen
  glow = b.add_material(rw.Material(emission_luminance=100.0, emission_color=(1, 1, 1)))
  b.add_instance("box", size=(0.6, 0.2, 0.004), xform=np.c_[np.eye(3), [0, 0, 2.5]], material=glow)
  lit = host_lights.derive_lights(emissive.add_panel_lights(b.build()).lights)
  # The shaped disk is seen (its cone darkens it off axis); the panel's lights are the box's own
  # emission (instance-backed), the shaped cylinder has no area frame, the unflagged rect is not
  # seen (Omniverse RTX's `visibleInPrimaryRay` default).
  assert host_lights.visible_lights(lit).tolist() == [[0, LightShape.RECT], [3, LightShape.TUBE], [4, LightShape.DISK]]


def test_area_lights_are_unseen_by_default():
  """An area light made without `visible` lights the scene but camera rays do not see it (the
  wall behind it shows), as Omniverse RTX renders UsdLux lights; `visible=True` shows it."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), specular_roughness=1.0))
  b.add_instance("plane", size=(20, 20, 0), xform=np.c_[np.eye(3), [0, 0, -6]], material=grey)
  b.add_light(rw.Light.rect((0.0, 0.0, -3.0), (0.0, 0.0, 1.0), 1.0, 1.0, intensity=100.0))
  b.add_light(rw.Light.sphere((-1.0, 0.0, -3.0), 0.2, intensity=50.0, visible=True))
  r = render_host(b.build(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=RES, settings={"lighting.area_lights": "ltc"})
  lum = required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2)
  sphere, rect = lum[_pixel(-1.0, 0.0, -3.0)], lum[_pixel(0.0, 0.0, -3.0)]
  assert rect < 0.05 * sphere  # the lit wall behind the rect, not its 100 nits
  assert host_lights.visible_lights(host_lights.derive_lights(b.build().lights)).tolist() == [[1, LightShape.SPHERE]]


def _scene() -> rw.Scene:
  """In front of a camera at the origin looking down -Z (every light flagged visible): a 1 x 1 m
  rect light of 100 nits facing it (centre), an emissive sphere of 100 nits (right), a sphere
  light of 50 nits (left), a rect light of 100 nits facing away (up), all at 3 m; a grey wall at
  6 m and a sphere light of 400 nits behind it (down)."""
  b = rw.SceneBuilder()
  grey = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), specular_roughness=1.0))
  glow = b.add_material(
    rw.Material(base_color=(0.0, 0.0, 0.0), specular_weight=0.0, emission_luminance=100.0, emission_color=(1, 1, 1))
  )
  b.add_instance("plane", size=(20, 20, 0), xform=np.c_[np.eye(3), [0, 0, -6]], material=grey)
  b.add_instance("sphere", size=(0.2, 0, 0), xform=np.c_[np.eye(3), [1.0, 0, -3]], material=glow)
  seen = {"visible_in_primary_ray": True}
  b.add_light(
    rw.Light(kind=LightKind.RECT, intensity=100.0, width=1.0, height=1.0, xform=_frame(TOWARD_CAMERA, (0, 0, -3)), **seen)
  )
  b.add_light(rw.Light(kind=LightKind.SPHERE, intensity=50.0, radius=0.2, xform=_frame(np.eye(3), (-1.0, 0, -3)), **seen))
  b.add_light(
    rw.Light(kind=LightKind.RECT, intensity=100.0, width=0.4, height=0.4, xform=_frame(np.eye(3), (0, 1.0, -3)), **seen)
  )
  b.add_light(rw.Light(kind=LightKind.SPHERE, intensity=400.0, radius=0.5, xform=_frame(np.eye(3), (0, -2.0, -8)), **seen))
  return b.build()


@pytest.mark.parametrize("form", ["point", "ltc"])
def test_camera_sees_area_lights_at_their_radiance(form: str):
  r = render_host(
    _scene(),
    (0.0, 0.0, 0.0),
    DOWN,
    fovy=FOVY,
    res=RES,
    settings={"lighting.area_lights": form, "lighting.emissive_lights": False},
  )
  hdr = required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3)
  lum = hdr.mean(axis=2)
  emissive_ref = lum[_pixel(1.0, 0.0, -3.0)]
  assert emissive_ref > 0.0
  # The rect light facing the camera at the radiance of an emissive surface of the same luminance,
  # the sphere light at half of it.
  np.testing.assert_allclose(lum[_pixel(0.0, 0.0, -3.0)], emissive_ref, rtol=5e-3)
  np.testing.assert_allclose(lum[_pixel(-1.0, 0.0, -3.0)], 0.5 * emissive_ref, rtol=5e-3)
  # A rect seen from behind and a light behind the wall: the lit wall (a few nits), not the light.
  for point in ((0.0, 1.0, -3.0), (0.0, -2.0, -8.0)):
    assert lum[_pixel(*point)] < 0.05 * emissive_ref, point


def test_compiled_in_only_with_visible_lights():
  """Scenes without an area light of their own compile nothing of it (`Features.visible_lights`)."""
  b = rw.SceneBuilder()
  b.add_instance("plane", size=(5, 5, 0))
  b.add_light(rw.Light(kind=LightKind.SPHERE, intensity=10.0, radius=0.0, xform=_frame(np.eye(3), (0, 0, -3))))
  plain = render_host(b.build(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=8)
  assert not plain.features.visible_lights
  assert render_host(_scene(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=8).features.visible_lights


def test_visible_lights_switch_is_run_time():
  """`lighting.visible_lights` off hides the lights from camera rays (the wall behind shows) and
  keeps the kernels; on again, they show as before."""
  settings = {"lighting.area_lights": "ltc", "lighting.emissive_lights": False}
  r = render_host(_scene(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=RES, settings=settings)
  build = r.context().features
  seen = required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2).copy()
  r.update_settings(**{"lighting.visible_lights": False})
  r.render()
  hidden = required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2).copy()
  assert r.context().features == build
  centre, emissive = _pixel(0.0, 0.0, -3.0), _pixel(1.0, 0.0, -3.0)
  assert hidden[centre] < 0.05 * seen[centre]
  np.testing.assert_allclose(hidden[emissive], seen[emissive], rtol=1e-5)  # geometry stays
  r.update_settings(**{"lighting.visible_lights": True})
  r.render()
  np.testing.assert_allclose(required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2), seen, rtol=1e-5)
