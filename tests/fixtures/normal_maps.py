"""A normal-mapped quad under a point light (render/test_normal_maps.py)."""

from __future__ import annotations

import math

import numpy as np

import raywarp as rw
from raywarp.scene.model import normal_map

TILT = np.array([0.35, 0.5, 0.0])
TILT[2] = math.sqrt(1.0 - TILT[0] ** 2 - TILT[1] ** 2)
QUAD = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
# UVs with the image's bottom-left at (-0.5, -0.5): +v runs along +y.
UV_UP = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def encode(n: np.ndarray) -> np.ndarray:
  """(4, 4, 3) uint8 texels of the uniform tangent-space normal `n` (OpenGL, [0, 1] encoding)."""
  return np.broadcast_to(np.round((n + 1.0) / 2.0 * 255.0).astype(np.uint8), (4, 4, 3)).copy()


def quad(texels: np.ndarray, xy: tuple[float, float] = (1.0, 1.0), *, uvs: np.ndarray = UV_UP) -> rw.Scene:
  """A 1 m quad in z = 0 (UVs `uvs`) under normal map `texels` read with per-axis scale `xy`, a
  point light on the +y side."""
  b = rw.SceneBuilder()
  mesh = b.add_mesh(QUAD, np.array([[0, 1, 2], [0, 2, 3]]), uvs=uvs)
  scale, bias = normal_map.unpack(xy)
  ref = rw.scene.TextureRef(b.add_texture(texels, colorspace="linear"), scale=scale, bias=bias)
  mat = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), textures=((rw.scene.Slot.GEOMETRY_NORMAL, ref),)))
  b.add_instance("mesh", mesh=mesh, material=mat, xform=np.c_[np.eye(3), np.zeros(3)])
  light = rw.Light(
    kind=rw.scene.LightKind.SPHERE, intensity=50.0, radius=0.0, treat_as_point=True, shadow_enable=False,
    xform=np.c_[np.eye(3), [0.0, 2.0, 0.5]],
  )  # fmt: skip
  b.add_light(light)
  return b.build()
