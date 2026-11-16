"""A Renderer's first-image check: one warning when a camera's first image is blown out or black.

The first `Renderer.render` outside a graph capture reads back the display color of up to
`MAX_WORLDS` worlds once (one host read, the only one `render` ever does) and warns, per camera,
when more than `CLIP_SHARE` of the pixels have a channel at full scale (exposure too high, a light
far too strong) or when every pixel is black (no light reaches the camera, exposure far too low, a
camera inside geometry). Host-only: no kernel, no setting, nothing per frame.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy

if TYPE_CHECKING:
  from raywarp.core.outputs.render_outputs import RenderOutputs

# Share of a camera's pixels with a channel at full scale above which its first image warns
# (Alloy's camera uses the same 0.2: a fifth of the frame white is never a deliberate exposure).
CLIP_SHARE: Final = 0.2
# Worlds read back (the first ones): enough to see the scene, small at any batch size.
MAX_WORLDS: Final = 16

type Pixels = np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]


def camera_pixels(rgb: wp.array, nworld: int) -> list[Pixels]:
  """Per camera, the RGB of the first `MAX_WORLDS` worlds as (worlds, H, W, 3) floats in [0, 1].

  Args:
    rgb: `RenderOutputs.rgb` in any `ColorOutput` layout: (nworld, ncam, H, W) vec4ub / vec3ub, or
      planar (nworld * ncam, 3, H, W) uint8 / float16.
    nworld: the renderer's world count (the planar layout's rows are world-major).
  """
  worlds = min(nworld, MAX_WORLDS)
  if rgb.dtype not in (wp.uint8, wp.float16):  # vec4ub / vec3ub
    host = host_copy(rgb[:worlds])  # (worlds, ncam, H, W, C)
    planes = [host[:, c, ..., :3] for c in range(host.shape[1])]
  else:  # planar: (nworld * ncam, 3, H, W)
    ncam = rgb.shape[0] // nworld
    host = host_copy(rgb[: worlds * ncam]).reshape(worlds, ncam, 3, *rgb.shape[2:])
    planes = [np.moveaxis(host[:, c], 1, -1) for c in range(ncam)]
  scale = 255.0 if planes and planes[0].dtype == np.uint8 else 1.0
  return [np.asarray(p, np.float32) / scale for p in planes]


def verdict(pixels: Pixels) -> str | None:
  """What is wrong with one camera's first image (None: nothing)."""
  if not np.any(pixels > 0.0):
    return "is black (no light reaches the camera, an exposure far too low, or a camera inside geometry)"
  clipped = float(np.mean(np.any(pixels >= 1.0, axis=-1)))
  if clipped > CLIP_SHARE:
    return f"has {clipped:.0%} of its pixels clipped to white (exposure too high or a light far too strong)"
  return None


def check(outputs: RenderOutputs, nworld: int) -> None:
  """Warns once per camera whose first image `verdict` flags (no color output: nothing)."""
  views = [(c, out.rgb) for c, out in enumerate(outputs.cameras)] if outputs.rgb is None else [(None, outputs.rgb)]
  for camera, rgb in views:
    if rgb is None:
      continue
    for index, pixels in enumerate(camera_pixels(rgb, nworld)):
      found = verdict(pixels)
      if found is not None:
        name = camera if camera is not None else index
        warnings.warn(
          f"raywarp: camera {name}'s first image {found}; see docs/faq.md (Too dark or too bright)",
          RuntimeWarning,
          stacklevel=3,
        )
