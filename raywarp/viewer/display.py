"""Display images of a renderer's outputs (host side): the colour image and its debug views.

Views are display images (H, W, 3) uint8 of one camera of world 0. Radiance views (hdr, direct,
indirect, emission) share one scale: the 99th luminance percentile of the HDR image, so they compare
directly. `ev` is a false-colour map of each pixel's luminance in stops relative to middle grey at
the camera exposure (the image median with automatic exposure). `nan` marks NaN / Inf / negative
HDR pixels in magenta over the display image.
"""

from __future__ import annotations

import dataclasses
import io
from typing import Final

import numpy as np

import raywarp as rw
from raywarp._base.dims import AnyHostArray
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.scene.images.colorspace import REC709_Y
from raywarp.settings.fields import exposure_from_ev100

# A display image: (H, W, 3) uint8 sRGB.
type Image = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
# Host copies of outputs by name (`read`): (H, W, ...) arrays.
type Outputs = dict[str, AnyHostArray]

# The debug outputs a view can show (every tier build stores every output: switching is run time).
DEBUG_OUTPUTS: Final = (
  "hdr",
  "albedo",
  "normal",
  "depth",
  "position",
  "uv",
  "roughness",
  "metallic",
  "emission",
  "direct",
  "indirect",
  "seg",
  "material",
  "motion",
)


# The views: the display colour, the EV false colour, the bad-HDR overlay, then each output.
VIEWS: Final = ("rgb", "ev", "nan", *DEBUG_OUTPUTS)


# Rec. 709 luminance weights of linear RGB.
REC709: Final = np.array(REC709_Y, np.float32)


# One colour per stop from -5 to +5 EV around middle grey (index 5 = middle grey).
EV_COLORS: Final = np.array(
  [
    [0, 0, 0],
    [40, 0, 90],
    [0, 0, 220],
    [0, 140, 255],
    [0, 200, 120],
    [128, 128, 128],
    [150, 220, 0],
    [255, 230, 0],
    [255, 140, 0],
    [255, 0, 0],
    [255, 255, 255],
  ],
  np.uint8,
)


def with_outputs(settings: rw.RenderSettings, outputs: list[str]) -> rw.RenderSettings:
  """`settings` with the display colour (RGBA bytes, the layout `read` takes) plus exactly the
  outputs the views named need (`needed`; ValueError for an unknown view)."""
  names = sorted({"rgb", *(n for o in outputs for n in needed(o))})
  wanted, current = rw.Outputs.parse(names), settings.outputs
  specs = {f"outputs.{n}": getattr(current, n) or getattr(wanted, n) for n in names}
  rgb = specs["outputs.rgb"]
  if not isinstance(rgb, rw.settings.ColorOutput) or rgb.channels != 4:
    specs["outputs.rgb"] = rw.settings.ColorOutput()  # the viewer reads RGBA bytes
  return settings.replace(specs)


def written(settings: rw.RenderSettings) -> list[str]:
  """The outputs a renderer with `settings` writes, in field order."""
  return [f.name for f in dataclasses.fields(settings.outputs) if getattr(settings.outputs, f.name) is not None]


def read(renderer: rw.Renderer, names: tuple[str, ...], cam: int = 0) -> Outputs:
  """Host copies of outputs of world 0, camera `cam` (rgb: (H, W, 3) uint8; the others their
  natural trailing shape). Raises for an output the renderer does not write."""
  out = renderer.outputs
  result = {}
  for name in names:
    array = getattr(out, name)
    if array is None:
      raise ValueError(f"the renderer does not write {name!r}")
    host = host_copy(array[0, cam])
    result[name] = host[..., :3] if name == "rgb" else host
  return result


def needed(view: str) -> tuple[str, ...]:
  """The outputs view `view` reads (ValueError for an unknown view)."""
  if view not in VIEWS:
    raise ValueError(f"unknown view {view!r}; known: {list(VIEWS)}")
  if view == "rgb":
    return ("rgb",)
  if view in {"ev", "nan"}:
    return ("rgb", "hdr")
  if view in {"direct", "indirect", "emission"}:
    return (view, "hdr")
  return (view,)


def luminance(rgb: AnyHostArray) -> AnyHostArray:
  """Rec. 709 luminance of linear RGB (..., 3)."""
  return np.asarray(rgb[..., :3], np.float32) @ REC709


def srgb(linear: AnyHostArray) -> Image:
  """uint8 sRGB of linear values (clipped to [0, 1])."""
  x = np.clip(np.nan_to_num(linear, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
  encoded = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
  return (encoded * 255.0 + 0.5).astype(np.uint8)


def gray(values: AnyHostArray, lo: float, hi: float) -> Image:
  """(H, W, 3) uint8 grey ramp of values over [lo, hi]."""
  span = hi - lo if hi > lo else 1.0
  g = (np.clip((np.nan_to_num(values) - lo) / span, 0.0, 1.0) * 255).astype(np.uint8)
  return np.repeat(g[..., None], 3, axis=-1)


def id_colors(ids: AnyHostArray) -> Image:
  """A stable colour per id (black for negative ids)."""
  h = np.bitwise_and(ids.astype(np.int64) * 2654435761, 0xFFFFFF)
  rgb = np.stack([np.right_shift(h, 16) & 255, np.right_shift(h, 8) & 255, h & 255], axis=-1).astype(np.uint8)
  rgb[ids < 0] = 0
  return rgb


def exposure_scale(settings: rw.RenderSettings, lum: AnyHostArray) -> float:
  """The multiplier of scene luminance before "middle grey = 0.18": the renderer's camera exposure
  of a fixed EV100 or, with automatic exposure, the one putting the image median at middle grey."""
  if settings.exposure.mode == "manual":
    return exposure_from_ev100(settings.exposure.ev100)
  finite = lum[np.isfinite(lum) & (lum > 0)]
  return 0.18 / float(np.median(finite)) if finite.size else 1.0


def view_image(view: str, data: Outputs, settings: rw.RenderSettings) -> Image:
  """The (H, W, 3) uint8 display image of `view`."""
  if view == "rgb":
    return data["rgb"]
  if view == "nan":
    image = data["rgb"].copy()
    image[bad_mask(data["hdr"])] = (255, 0, 255)
    return image
  if view == "ev":
    lum = luminance(data["hdr"])
    stops = np.log2(np.maximum(lum * exposure_scale(settings, lum) / 0.18, 1e-12))
    return EV_COLORS[np.clip(np.round(np.nan_to_num(stops, nan=-99.0)) + 5, 0, 10).astype(np.int64)]
  value = data[view]
  if view in {"hdr", "direct", "indirect", "emission"}:
    return srgb(value / radiance_scale(data["hdr"]))
  if view == "albedo":
    return srgb(value)
  if view == "normal":
    return (np.clip(value * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
  if view == "depth":
    return _depth(value)
  if view in {"position", "motion"}:
    return _spread(value)
  if view == "uv":
    frac = np.mod(value, 1.0)
    return (np.concatenate([frac, np.zeros_like(frac[..., :1])], axis=-1) * 255).astype(np.uint8)
  if view in {"roughness", "metallic"}:
    return gray(value, 0.0, 1.0)
  return id_colors(value)


def radiance_scale(hdr: AnyHostArray) -> float:
  """The shared scale of radiance views: the HDR image's 99th luminance percentile."""
  lum = luminance(hdr)
  finite = lum[np.isfinite(lum)]
  p99 = float(np.percentile(finite, 99)) if finite.size else 1.0
  return p99 if p99 > 0 else 1.0


def _depth(depth: AnyHostArray) -> Image:
  """Inverse depth over its 1-99th percentiles (misses black)."""
  hit = np.isfinite(depth) & (depth > 0)
  if not hit.any():
    return np.zeros((*depth.shape, 3), np.uint8)
  inv = np.where(hit, 1.0 / np.where(hit, depth, 1.0), 0.0)
  lo, hi = np.percentile(inv[hit], [1, 99])
  image = gray(inv, float(lo), float(hi))
  image[~hit] = 0
  return image


def _spread(value: AnyHostArray) -> Image:
  """Each channel over its 1-99th percentiles (2-channel values get a zero blue channel)."""
  channels = []
  for c in range(value.shape[-1]):
    v = value[..., c]
    lo, hi = np.percentile(v, [1, 99])
    channels.append(gray(v, float(lo), float(hi))[..., 0])
  while len(channels) < 3:
    channels.append(np.zeros_like(channels[0]))
  return np.stack(channels, axis=-1)


def bad_mask(hdr: AnyHostArray) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
  """(H, W) pixels with a NaN, Inf or negative HDR channel."""
  return ~np.isfinite(hdr).all(axis=-1) | (np.nan_to_num(hdr) < 0).any(axis=-1)


def encode(image: Image, fmt: str, quality: int = 90) -> bytes:
  """PNG or JPEG (`quality`) bytes of an (H, W, 3) uint8 image (Pillow: the `viewer` extra)."""
  from PIL import Image as PilImage  # noqa: PLC0415  # the viewer extra, on use

  if fmt not in {"png", "jpeg"}:
    raise ValueError("format must be png or jpeg")
  buffer = io.BytesIO()
  PilImage.fromarray(image).save(buffer, format="PNG" if fmt == "png" else "JPEG", quality=quality, compress_level=1)
  return buffer.getvalue()
