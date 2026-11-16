"""DomeLight texture formats (UsdLux `inputs:texture:format`) resampled to a lat-long map at load.

The UsdLux schema names the formats but not their geometry; its latlong is OpenEXR's
(`ImfEnvmap.cpp`: pole +Y, centre column +Z, u = 0.25 towards +X: `EXR_LATLONG`). The other
formats are expressed in that frame (directions e = (x, y, z) below), all centred on the latlong's
centre column (+Z) with +Y up, so a converted map shows the directions the same environment shows
as a latlong map (as HDRShop converts between them):
- automatic: the files read here (EXR, HDR, 8-bit images) carry no parameterization, so latlong,
  as Hydra Storm reads every dome texture (it has no other format);
- mirroredBall: an orthographic photograph of a mirrored sphere taken along -Z (OpenGL's
  GL_SPHERE_MAP; Cycles' `mirrorball_to_direction`): image centre +Z, image right +X, image up
  +Y, radius sin(theta / 2) for the angle theta from +Z (the rim is -Z, straight behind);
- angular: Debevec's angular map (pauldebevec.com/Probes: (u, v) = (x, y) r with
  r = acos(z) / (pi sqrt(x^2 + y^2))): mirroredBall's layout with radius theta / pi;
- cubeMapVerticalCross: a 3 x 4 cross of square faces unfolded as seen from inside (HDRShop /
  Debevec crosses): front +Z in the centre, +Y above it, -Y below, +X left, -X right, and -Z at
  the bottom turned half (it folds on from -Y).

Ball images are mirror images of the view (a reflection): +X is on the image's right, while the
latlong and the cross have +X left of their centre.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from raywarp.scene.images import colorspace
from raywarp.scene.images import decode as images
from raywarp.scene.images.colorspace import ColorSpace, texel_transfer
from raywarp.scene.images.latlong import latlong_directions
from raywarp.scene.model.types import LinearRGBA

# Formats stored as latlong (read as the texture file itself).
LATLONG_FORMATS: Final = frozenset({"automatic", "latlong"})
# Formats resampled to latlong here.
RESAMPLED_FORMATS: Final = frozenset({"mirroredBall", "angular", "cubeMapVerticalCross"})
# The UsdLux spec's latlong (OpenEXR's) in the core's convention (`core.shade.environment.latlong`).
# Columns: the core map's +x (centre column), +y (u = 0.25) and +z (pole) in OpenEXR's frame.
EXR_LATLONG: Final = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

# Directions (h, w, 3) and texel coordinates (h, w) (texel i spans [i, i + 1]).
type Directions = np.ndarray[[int, int, 3], np.dtype[np.float64]]
type Coords = np.ndarray[[int, int], np.dtype[np.float64]]

# Cube faces by major axis of (r, u, f) = (-x, y, z) (r: right of a viewer facing +Z, u: up,
# f: front) and its sign (+r, -r, +u, -u, +f, -f): cross cell (column, row), then the in-face
# coordinates a (right) and b (up) as (component, sign) over the major component's magnitude.
_CELL_COL: Final = np.array([2, 0, 1, 1, 1, 1])
_CELL_ROW: Final = np.array([1, 1, 0, 2, 1, 3])
_A_AXIS: Final = np.array([2, 2, 0, 0, 0, 0])
_A_SIGN: Final = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
_B_AXIS: Final = np.array([1, 1, 2, 2, 1, 1])
_B_SIGN: Final = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])


def read_linear(path: str, space: ColorSpace) -> tuple[LinearRGBA | None, ColorSpace, str]:
  """(the (h, w, 4) texels of image file `path` authored in `space` as linear values, their
  space, "") through the library's decoders; (None, space, the error) when it cannot be decoded.
  Color spaces convert to linear Rec.709 (`colorspace.decode`); data stays data."""
  source = images.ImageSource(path=path, transfer=texel_transfer(space))
  data, _ = images.read_file(source)
  info = images.probe(source, data)
  texels, error = images.decode_base(source, info, data)
  if error:
    return None, space, error
  if space == ColorSpace.DATA:
    return images.linear_rgba(texels, transfer="linear"), space, ""
  return colorspace.decode(texels, space), ColorSpace.LIN_REC709, ""


def to_latlong(image: LinearRGBA, fmt: str, *, where: str) -> LinearRGBA:
  """`image` (h, w, 4) of a resampled format as a (rows, 2 rows, 4) latlong map in the core's
  convention (bilinear). `rows` matches the source's finest angular resolution: angular h / 2
  (pi rows per h / 2 texels of radius), mirroredBall pi h / 4 (its centre: d radius / d theta =
  1 / 2), cubeMapVerticalCross 2 x the face size (a face spans pi / 2).

  Raises:
    ValueError: for an unknown format or an image whose shape cannot hold it (`where` names it).
  """
  h, w = image.shape[:2]
  rows = _rows(fmt, h, w, where)
  e = latlong_directions(rows, 2 * rows) @ EXR_LATLONG.T
  x, y = _cross_coords(e, w // 3) if fmt == "cubeMapVerticalCross" else _ball_coords(e, h, angular=fmt == "angular")
  return _bilinear(image, x, y)


def _rows(fmt: str, h: int, w: int, where: str) -> int:
  """The latlong height of a source (h, w) of format `fmt` (`to_latlong`).

  Raises:
    ValueError: for an unknown format or a shape that cannot hold it.
  """
  if fmt not in RESAMPLED_FORMATS:
    known = ", ".join(sorted(LATLONG_FORMATS | RESAMPLED_FORMATS))
    raise ValueError(f"{where}: texture:format {fmt!r} is not a UsdLux format ({known})")
  if fmt == "cubeMapVerticalCross":
    if 3 * h != 4 * w or w % 3:
      raise ValueError(f"{where}: texture:format 'cubeMapVerticalCross' needs a 3:4 (w x h) image, got {w} x {h}")
    return 2 * (w // 3)
  if h != w:
    raise ValueError(f"{where}: texture:format {fmt!r} needs a square image, got {w} x {h}")
  return max(1, round(h / 2 if fmt == "angular" else math.pi * h / 4))


def _ball_coords(e: Directions, n: int, *, angular: bool) -> tuple[Coords, Coords]:
  """Texel coordinates (x, y) in an (n, n) mirroredBall / angular image of directions `e`."""
  theta = np.arccos(np.clip(e[..., 2], -1.0, 1.0))
  radius = theta / math.pi if angular else np.sin(0.5 * theta)
  s = np.hypot(e[..., 0], e[..., 1])
  k = radius / np.maximum(s, 1e-12)
  return _clamp(0.5 * (1.0 + e[..., 0] * k) * n, 0, n), _clamp(0.5 * (1.0 - e[..., 1] * k) * n, 0, n)


def _cross_coords(e: Directions, face: int) -> tuple[Coords, Coords]:
  """Texel coordinates (x, y) in a vertical cross of `face`-texel faces of directions `e`, kept
  inside each direction's face (no blending with the empty cells)."""
  c = np.stack([-e[..., 0], e[..., 1], e[..., 2]], axis=-1)
  axis = np.argmax(np.abs(c), axis=-1)
  major = np.sum(c * (axis[..., None] == np.arange(3)), axis=-1)  # the signed major component
  f = 2 * axis + (major < 0)
  m = np.abs(major)
  a = _A_SIGN[f] * np.take_along_axis(c, _A_AXIS[f][..., None], axis=-1)[..., 0] / m
  b = _B_SIGN[f] * np.take_along_axis(c, _B_AXIS[f][..., None], axis=-1)[..., 0] / m
  col, row = _CELL_COL[f] * face, _CELL_ROW[f] * face
  return _clamp(col + 0.5 * (1.0 + a) * face, col, col + face), _clamp(row + 0.5 * (1.0 - b) * face, row, row + face)


def _clamp(t: Coords, lo: Coords | int, hi: Coords | int) -> Coords:
  """Coordinates `t` clamped to the texel centres of [lo, hi]."""
  return np.clip(t, np.add(lo, 0.5), np.subtract(hi, 0.5))


def _bilinear(image: LinearRGBA, x: Coords, y: Coords) -> LinearRGBA:
  """`image` (h, w, c) read bilinearly at texel coordinates (x, y) (texel centres at i + 0.5)."""
  h, w = image.shape[:2]
  fx, fy = x - 0.5, y - 0.5
  x0, y0 = np.floor(fx), np.floor(fy)
  tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
  x0i, y0i = np.clip(x0.astype(np.int64), 0, w - 1), np.clip(y0.astype(np.int64), 0, h - 1)
  x1i, y1i = np.minimum(x0i + 1, w - 1), np.minimum(y0i + 1, h - 1)
  top = image[y0i, x0i] * (1.0 - tx) + image[y0i, x1i] * tx
  bottom = image[y1i, x0i] * (1.0 - tx) + image[y1i, x1i] * tx
  return np.ascontiguousarray(top * (1.0 - ty) + bottom * ty, np.float32)
