"""DomeLight `texture:format`s resampled to the latlong environment
(`formats.usd.stage.dome_formats`).

Each source image is drawn from its format's definition (pixel -> direction, written here apart
from the loader's direction -> pixel mapping) with color 0.5 + 0.5 e for the direction e in
OpenEXR's latlong frame (pole +Y, centre column +Z, u = 0.25 towards +X); the loaded latlong map
must show those colors in the same directions.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import required
from rw_helpers import new_stage, write_hdr

pytest.importorskip("pxr")

from pxr import Sdf, UsdLux

import raywarp as rw
import raywarp.formats.usd
from raywarp.scene.images.latlong import latlong_uv

_N = 64  # ball sizes; a cross has 16-texel faces
_AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64)
_DIAGONALS = np.array([[1, 1, 1], [-1, 0.5, 1], [0.3, -1, -0.6], [-0.7, 0.2, -1], [1, -0.4, -0.3]], np.float64)


def _ball_uv(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """(u right, v up, radius) of the texel centres of an (n, n) ball image, in [-1, 1]."""
  t = (np.arange(n) + 0.5) / n * 2.0 - 1.0
  u, v = np.meshgrid(t, -t)
  return u, v, np.hypot(u, v)


def _angular(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Directions of an angular map: angle pi * radius from +Z towards (u, v)."""
  u, v, rho = _ball_uv(n)
  theta = np.pi * rho
  s = np.sin(theta) / np.maximum(rho, 1e-12)
  return np.stack([u * s, v * s, np.cos(theta)], axis=-1), rho < 1.0


def _mirrored_ball(n: int) -> tuple[np.ndarray, np.ndarray]:
  """Directions of a mirrored ball seen along -Z: the view ray (0, 0, -1) reflected about the
  sphere's normal (u, v, sqrt(1 - rho^2))."""
  u, v, rho = _ball_uv(n)
  normal = np.stack([u, v, np.sqrt(np.maximum(1.0 - rho**2, 0.0))], axis=-1)
  view = np.array([0.0, 0.0, -1.0])
  return view - 2.0 * (normal @ view)[..., None] * normal, rho < 1.0


def _cross(face: int) -> tuple[np.ndarray, np.ndarray]:
  """Directions of a vertical cross: each cell's (a right, b up) in [-1, 1] -> (r, u, f), with
  r = -x the right of a viewer facing +Z (front), folded from inside the cube."""
  t = (np.arange(face) + 0.5) / face * 2.0 - 1.0
  a, b = np.meshgrid(t, -t)
  one = np.ones_like(a)
  cells = {
    (1, 1): (a, b, one),  # front +Z
    (2, 1): (one, b, -a),  # right -X
    (0, 1): (-one, b, a),  # left +X
    (1, 0): (a, one, -b),  # up +Y
    (1, 2): (a, -one, b),  # down -Y
    (1, 3): (a, -b, -one),  # back -Z, turned half
  }
  e = np.zeros((4 * face, 3 * face, 3))
  inside = np.zeros((4 * face, 3 * face), bool)
  for (col, row), (r, up, f) in cells.items():
    d = np.stack([-r, up, f], axis=-1)
    e[row * face : (row + 1) * face, col * face : (col + 1) * face] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    inside[row * face : (row + 1) * face, col * face : (col + 1) * face] = True
  return e, inside


_SOURCES = {"angular": _angular(_N), "mirroredBall": _mirrored_ball(_N), "cubeMapVerticalCross": _cross(16)}


def _dome_map(tmp_path, fmt: str, image: np.ndarray) -> tuple[rw.Scene, np.ndarray | None]:
  """The scene and latlong texels of a stage whose DomeLight shows `image` in format `fmt`."""
  stage = new_stage(tmp_path)
  write_hdr(tmp_path / "env.hdr", image)
  dome = UsdLux.DomeLight.Define(stage, "/Dome")
  dome.CreateTextureFileAttr(Sdf.AssetPath("env.hdr"))
  dome.CreateTextureFormatAttr(fmt)
  s = rw.formats.usd.read(stage).build()
  texture = int(s.environments.texture[0])
  return s, (None if texture < 0 else s.images.level(int(s.textures.image[texture])))


@pytest.mark.parametrize("fmt", sorted(_SOURCES))
def test_a_resampled_dome_shows_each_direction_where_its_format_puts_it(tmp_path, fmt):
  e, inside = _SOURCES[fmt]
  s, texels = _dome_map(tmp_path, fmt, np.where(inside[..., None], 0.5 + 0.5 * e, 0.0))
  texels = required(texels)
  assert not any("texture" in k for k in s.report.dropped), dict(s.report.dropped)
  h, w = texels.shape[:2]
  assert w == 2 * h
  # A mirrored ball's rim squeezes everything behind it: its -Z texel is not checked.
  dirs = np.concatenate([_AXES[:5] if fmt == "mirroredBall" else _AXES, _DIAGONALS])
  dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
  # OpenEXR's frame -> the core map's: centre column +x = +Z, u = 0.25 (+y) = +X, pole +z = +Y.
  u, v = latlong_uv(dirs[:, [2, 0, 1]][None])
  got = texels[np.minimum((v[0] * h).astype(int), h - 1), np.minimum((u[0] * w).astype(int), w - 1), :3]
  np.testing.assert_allclose(got, 0.5 + 0.5 * dirs, atol=0.06)


@pytest.mark.parametrize("fmt", ["automatic", "latlong"])
def test_latlong_formats_read_the_file_as_it_is(tmp_path, fmt):
  image = np.random.default_rng(0).uniform(0.1, 1.0, (8, 16, 3))
  s, texels = _dome_map(tmp_path, fmt, image)
  texels = required(texels)
  assert texels.shape[:2] == (8, 16)
  np.testing.assert_allclose(texels[..., :3], image, atol=0.01)
  assert not s.report.dropped


@pytest.mark.parametrize(
  ("fmt", "shape"),
  [("cubeMapVerticalCross", (64, 64)), ("angular", (32, 64)), ("mirroredBall", (64, 32)), ("fisheye", (32, 64))],
  ids=["cross-not-3x4", "angular-not-square", "ball-not-square", "unknown-token"],
)
def test_a_dome_image_that_cannot_hold_its_format_raises(tmp_path, fmt, shape):
  with pytest.raises(ValueError, match=f"/Dome: texture:format '{fmt}'"):
    _dome_map(tmp_path, fmt, np.full((*shape, 3), 0.5))
