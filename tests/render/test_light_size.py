"""A light's size widens the specular lobes of its punctual form
(`light_evaluators.source_widening`): a smooth metal reflects a sphere light as a spot of the
light's angular size, not a dot, at about the same reflected energy (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DOWN, render_host, required

import raywarp as rw
from raywarp.scene.model.types import LightKind

RES = 96
FOVY = 40.0


def _reflection(radius: float, *, metal: bool = True) -> np.ndarray:
  """(RES, RES) luminance of a smooth metal (else matte) plane 4 m in front of a camera at the
  origin, lit by a sphere light of `radius` 1 m in front of the camera, off axis (the light
  itself hidden: `lighting.visible_lights` off), its point form shaded."""
  b = rw.SceneBuilder()
  surface = (
    rw.Material(base_color=(0.9, 0.9, 0.9), base_metalness=1.0, specular_roughness=0.1)
    if metal
    else rw.Material(base_color=(0.9, 0.9, 0.9), specular_weight=0.0, specular_roughness=1.0)
  )
  mat = b.add_material(surface)
  b.add_instance("plane", size=(20, 20, 0), xform=np.c_[np.eye(3), [0, 0, -4]], material=mat)
  b.add_light(rw.Light(kind=LightKind.SPHERE, intensity=20.0, radius=radius, xform=np.c_[np.eye(3), [0.4, 0.0, -1.0]]))
  settings = {"lighting.area_lights": "point", "lighting.visible_lights": False, "shadows.enabled": False}
  r = render_host(b.build(), (0.0, 0.0, 0.0), DOWN, fovy=FOVY, res=RES, settings=settings)
  return required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3).mean(axis=2)


def test_sized_light_reflects_as_a_spot_of_its_size():
  """A 0.3 m sphere light's reflection covers several times the pixels of a 0.05 m one's, and
  reflects the energy its power gives (the widened lobe is normalised: the ratio of the two
  reflections' sums is the ratio of the lights' power, measured on a matte plane)."""
  small, sized = _reflection(0.05), _reflection(0.3)
  spot = (sized > 0.1 * sized.max()).sum()
  dot = (small > 0.1 * small.max()).sum()
  assert spot > 2.5 * dot, (spot, dot)
  power = _reflection(0.3, metal=False).sum() / _reflection(0.05, metal=False).sum()
  assert sized.sum() / small.sum() == pytest.approx(power, rel=0.15)
