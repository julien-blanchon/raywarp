"""Where the deferred display reads a frame's scene-linear HDR image (`Source`): the "hdr" output
(float32, float16 or float16 of sqrt(radiance); unexposed: times the world's and the camera's
exposure). `make_fetch(kind)` reads one pixel's manually exposed color; `make_sample` a bilinear
sample (clamped to its own image)."""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.kernels.abi.structs import Lens
from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.kernels.generic import specialize

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# `Source` kinds: the hdr output, float32 / float16 / float16 of sqrt(radiance) (the internal
# image: `FloatOutput.sqrt`).
HDR: Final = 0
HDR16: Final = 1
HDR_SQRT16: Final = 2


@wp.struct
class Source:
  """A frame's HDR image as the deferred display reads it (unused arrays: placeholders).

  Attributes:
    hdr, hdr16: the "hdr" output (unexposed radiance), float32 / float16 (or float16 of its
      square root: `HDR_SQRT16`).
    exposure: (1 or nworld,) the worlds' exposure; lens: the cameras (their exposure, address).
    worlds: the world of each launch unit (`Renderer.render(active=...)`'s list, -1 a hole;
      empty: unit u is world u). The passes over a frame's images launch one unit per entry, so
      the worlds not listed launch nothing.
  """

  hdr: wp.array2d[wp.vec3]
  hdr16: wp.array2d[wp.vec3h]
  exposure: wp.array1d[float]
  lens: wp.array1d[Lens]
  worlds: wp.array1d[int]


@wp.func
def manual_exposure(
  # In:
  src: Source,
  w: int,
  cam: int,
) -> float:
  """The exposure the source's colors carry: the world's times the camera's."""
  return exposure_at(src.exposure, w) * src.lens[cam].exposure


@wp.func
def unit_world(
  # In:
  src: Source,
  unit: int,
) -> int:
  """The world of launch unit `unit` (-1: a hole of the active list, nothing to do)."""
  if src.worlds.shape[0] == 0:
    return unit
  return src.worlds[unit]


@specialize
def make_fetch(kind: int):
  """fetch(src, w, cam, i) of source `kind`."""

  @wp.func
  def fetch(
    # In:
    src: Source,
    w: int,
    cam: int,
    i: int,
  ) -> wp.vec3:
    """Pixel i (within its camera image) of world w, exposed."""
    adr = src.lens[cam].adr + i
    if wp.static(kind == HDR16):
      return wp.vec3(src.hdr16[w, adr]) * manual_exposure(src, w, cam)
    if wp.static(kind == HDR_SQRT16):
      r = wp.vec3(src.hdr16[w, adr])
      return wp.cw_mul(r, r) * manual_exposure(src, w, cam)
    return src.hdr[w, adr] * manual_exposure(src, w, cam)

  return fetch


@specialize
def make_sample(kind: int):
  """sample(src, w, cam, width, height, p) of source `kind`: bilinear at pixel position p (pixel
  centres at integer + 0.5), clamped to the image's own border."""
  fetch = make_fetch(kind)

  @wp.func
  def sample(
    # In:
    src: Source,
    w: int,
    cam: int,
    width: int,
    height: int,
    p: wp.vec2,
  ) -> wp.vec3:
    """The bilinear exposed color at p."""
    x = wp.clamp(p[0] - 0.5, 0.0, float(width - 1))
    y = wp.clamp(p[1] - 0.5, 0.0, float(height - 1))
    x0 = wp.min(int(x), width - 1)
    y0 = wp.min(int(y), height - 1)
    x1 = wp.min(x0 + 1, width - 1)
    y1 = wp.min(y0 + 1, height - 1)
    fx = x - float(x0)
    fy = y - float(y0)
    a = fetch(src, w, cam, y0 * width + x0) * (1.0 - fx) + fetch(src, w, cam, y0 * width + x1) * fx
    b = fetch(src, w, cam, y1 * width + x0) * (1.0 - fx) + fetch(src, w, cam, y1 * width + x1) * fx
    return a * (1.0 - fy) + b * fy

  return sample
