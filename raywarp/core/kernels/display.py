"""Writes a pixel's final display color in the layouts the output word asks for
(`output_layout.OUT_RGBA` ... `OUT_CHW16`), and the luminance output.

Every layout is a per-world buffer at the pixel's flat index (world-major, as every output):
packed 8-bit RGBA (the default; its alpha byte is the coverage the OptiX downsample and the
deferred display carry), RGB
bytes, or channels-first planes, uint8 or float16 in [0, 1], at 3 x the camera's buffer address
(camera c's planes are (3, H_c, W_c) at `3 * Lens.adr`: a policy reads the buffer as its
observation tensor with no conversion pass). Every pass that finalises a pixel's color (the
megakernel bodies, the OptiX downsample, the deferred display) writes through
`write_display`, so the luminance output (Rec. 709 luminance of the display color,
sRGB-encoded) follows every one of them. The stores read the output word (`toggles.out_on`):
folded in the render kernels, at run time in the passes after them.
"""

from __future__ import annotations

import warp as wp

from raywarp.core.kernels.abi.output_layout import (
  OUT_ALPHA,
  OUT_CHW8,
  OUT_CHW16,
  OUT_HDR,
  OUT_HDR_SQRT,
  OUT_LUMINANCE,
  OUT_RGB8,
  OUT_RGBA,
)
from raywarp.core.kernels.abi.structs import OutputArrays, f16s, rgb8s, u8s, u32s, vec3hs, vec3s
from raywarp.core.kernels.abi.toggles import out_half, out_on
from raywarp.core.post.grading import color
from raywarp.core.trace.rays import camera

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def write_display(
  # Out:
  out: OutputArrays,
  # In:
  worldid: int,
  adr: int,
  cam_adr: int,
  npix: int,
  display: wp.vec3,
  alpha: float,
):
  """Writes a pixel's display color in the word's layout (packed RGBA, RGB bytes or channel
  planes) and its luminance.

  `adr`: the pixel's flat index; `cam_adr` / `npix`: its camera's first index and pixel count;
  `display`: display-linear color in [0, 1]; `alpha`: coverage (1 without `ColorOutput.alpha`).
  """
  # The 8-bit layouts encode the sRGB bytes once (the display color is already in [0, 1]).
  encoded = int(color.encode_display(display, alpha))
  if out_on(out.bits, wp.static(OUT_RGBA)):
    u32s(out, out.table.rgb)[worldid, adr] = wp.uint32(encoded)
  if out_on(out.bits, wp.static(OUT_RGB8)):
    rgb8s(out, out.table.rgb_hwc)[worldid, adr] = wp.vec3ub(
      wp.uint8(encoded & 255), wp.uint8((encoded >> 8) & 255), wp.uint8((encoded >> 16) & 255)
    )
  # Channel k of the pixel at 3 * cam_adr + k * npix + (adr - cam_adr).
  base = 2 * cam_adr + adr
  if out_on(out.bits, wp.static(OUT_CHW8)):
    for k in range(3):
      u8s(out, out.table.chw8, 3)[worldid, base + k * npix] = wp.uint8((encoded >> (8 * k)) & 255)
  if out_on(out.bits, wp.static(OUT_CHW16)):
    for k in range(3):
      f16s(out, out.table.chw16, 3)[worldid, base + k * npix] = wp.float16(color.srgb_oetf(wp.clamp(display[k], 0.0, 1.0)))
  if out_on(out.bits, wp.static(OUT_LUMINANCE)):
    y = color.srgb_oetf(wp.clamp(color.luminance(display), 0.0, 1.0))
    if out_half(out.half, wp.static(OUT_LUMINANCE)):
      f16s(out, out.table.luminance16, 1)[worldid, adr] = wp.float16(y)
    else:
      u8s(out, out.table.luminance, 1)[worldid, adr] = wp.uint8(int(y * 255.0 + 0.5))


@wp.func
def sqrt16(
  # In:
  hdr: wp.vec3,
) -> wp.vec3h:
  """sqrt(radiance) per channel in float16 (`FloatOutput.sqrt`), negatives at 0, saturated at
  float16's largest finite value (an infinite half would turn NaN in the tone mapper)."""
  c = wp.max(hdr, wp.vec3(0.0))
  return wp.vec3h(wp.min(wp.vec3(wp.sqrt(c[0]), wp.sqrt(c[1]), wp.sqrt(c[2])), wp.vec3(65504.0)))


@wp.func
def store_hdr(
  # Out:
  out: OutputArrays,
  # In:
  worldid: int,
  adr: int,
  hdr: wp.vec3,
):
  """Writes pixel `adr`'s HDR radiance into the "hdr" buffer as the word stores it (float32,
  float16, or float16 sqrt: the deferred display's internal image)."""
  if out_on(out.bits, wp.static(OUT_HDR)):
    if out_on(out.bits, wp.static(OUT_HDR_SQRT)):
      vec3hs(out, out.table.hdr16)[worldid, adr] = sqrt16(hdr)
    elif out_half(out.half, wp.static(OUT_HDR)):
      vec3hs(out, out.table.hdr16)[worldid, adr] = wp.vec3h(hdr)
    else:
      vec3s(out, out.table.hdr)[worldid, adr] = hdr


@wp.func
def coverage(
  # In:
  out: OutputArrays,
  covered: float,
  inv_samples: float,
) -> float:
  """The coverage alpha of a pixel whose `covered` samples of 1 / `inv_samples` hit geometry (1
  without `ColorOutput.alpha`)."""
  alpha = float(1.0)
  if out_on(out.bits, wp.static(OUT_ALPHA)):
    alpha = covered * inv_samples
  return alpha


@wp.func
def write_pixel(
  # Out:
  out: OutputArrays,
  # In:
  worldid: int,
  pix: camera.Pixel,
  display: wp.vec3,
  hdr: wp.vec3,
  covered: float,
  store: bool,
):
  """Writes a pixel's display color (encoded; only when `store`: a deferred display pass writes
  it otherwise), its mean HDR radiance and its coverage: the fraction `covered` of its samples
  that hit geometry."""
  adr = camera.buffer_index(pix)
  store_hdr(out, worldid, adr, hdr)
  if store:
    alpha = coverage(out, covered, 1.0)
    write_display(out, worldid, adr, pix.lens.adr, pix.width * pix.height, display, alpha)
