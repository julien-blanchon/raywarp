"""Supersampling of OptiX renders (spp n > 1): an n x resolution spp-1 image, then averaged.

Trace calls inside the megakernel's sample loop make OptiX keep (spill) much of the program state
across every trace, several times slower than one launch index per sample. The program renders the
n x n samples of each pixel as pixels of `Supersampling.hi`, in sample-major warps
(`camera.locate_sample`), and shades each surface of a pixel once (multisampling: a sample on the
first sample's surface stores `pixel.SHARED`, `pixel.make_multisampling`); `downsample_kernel`
then averages them into the renderer's outputs (lower resolution, same cameras) with the sample
loop's rules, a shared sample counting the first one's radiance: the samples' HDR mean is
displayed (`grading.make_display`) and encoded. Depth, segmentation and guides are traced at the
original pixel center by the first high-resolution sample and copied from its slot.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, final

import numpy as np
import warp as wp

from raywarp._base.warp_structs import warp_struct
from raywarp.core.kernels import first_hit
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.output_layout import OUT_ALPHA, OUT_DEPTH
from raywarp.core.kernels.abi.structs import (
  WANT_DEPTH,
  WANT_RGB,
  WANT_SEG,
  Cameras,
  EnvironmentArrays,
  Lens,
  OutputArrays,
  f16s,
  f32s,
  i32s,
  u32s,
  vec3s,
)
from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.kernels.abi.toggles import out_half, out_on
from raywarp.core.kernels.display import coverage, store_hdr, write_display
from raywarp.core.kernels.generic import specialize
from raywarp.core.kernels.pixel import SHARED
from raywarp.core.outputs.render_outputs import Buffers
from raywarp.core.post.grading.transform import make_display
from raywarp.core.trace.rays.camera import locate_ray
from raywarp.settings import ColorOutput, FloatOutput, Outputs

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.trace.context import FrameContext


def factor(ctx: FrameContext) -> int:
  """The supersampling factor n of a frame of `ctx`: the settings' `aa.spp` where its build
  supersamples (`Features.spp` > 1 with a shaded output: a supersampling backend's spp is 1 or 2,
  the factor run-time data), else 1."""
  f = ctx.features
  return ctx.settings.aa.spp if f.spp > 1 and f.any_rgb else 1


@wp.func
def _average(
  # In:
  hi: OutputArrays,
  out: OutputArrays,
  worldid: int,
  hi_adr: int,
  first: int,
  w: int,
  n: int,
) -> tuple[wp.vec3, float]:
  """(hdr, alpha x 255) sums of the n x n samples of a pixel (`first`: its first sample in the
  w · n wide image; the alpha with `out`'s word's `OUT_ALPHA`). A sample that shared the first
  sample's surface (`pixel.SHARED`: not shaded) counts the first sample's radiance."""
  hdr = wp.vec3(0.0)
  covered = float(0.0)
  shaded = vec3s(hi, hi.table.hdr)[worldid, hi_adr + first]
  for s in range(n * n):
    sub = hi_adr + first + (s // n) * w * n + s % n
    sample = vec3s(hi, hi.table.hdr)[worldid, sub]
    hdr += wp.where(sample[0] == wp.static(SHARED), shaded, sample)
    if out_on(out.bits, wp.static(OUT_ALPHA)):
      covered += float(int(u32s(hi, hi.table.rgb)[worldid, sub] >> wp.uint32(24)))
  return hdr, covered


@specialize
def downsample_kernel(f: Features):
  """Kernel averaging the n x n sub-pixel samples of the supersampled image (see the module doc)
  into the output word's outputs; n is the supersampled lens table's `Lens.scale` (run-time data:
  one kernel for every factor)."""
  display_of = make_display(graded=False)

  @wp.kernel(module="unique", enable_backward=False, module_options=kernel_structs.module_options(f))
  def downsample(
    # In:
    cams: Cameras,
    hi_lens: wp.array1d[Lens],
    env: EnvironmentArrays,
    hi: OutputArrays,
    # Out:
    out: OutputArrays,
  ):
    """Pixel `rayid` of world `worldid`: the mean of its n x n samples in `hi`."""
    n = hi_lens[0].scale
    worldid, rayid = wp.tid()
    if kernel_structs.skipped(out, worldid):
      return
    loc = locate_ray(cams, 1, rayid)  # row-major threads, as samples and outputs are stored
    camid, px, py = loc[0], loc[1], loc[2]
    if camid == -1:
      return
    w = cams.camera_res[camid][0]
    adr = cams.lens[camid].adr + py * w + px
    # The outputs this camera writes (`Lens.wants`): the others keep their contents.
    wants = out.table.wants & cams.lens[camid].wants
    # The first sample of the pixel's n x n block in the supersampled image (w · n wide).
    hi_adr = hi_lens[camid].adr
    first = py * n * w * n + px * n
    if (wants & wp.static(WANT_RGB)) != 0:
      e = exposure_at(env.exposure, worldid) * cams.lens[camid].exposure
      hdr, covered = _average(hi, out, worldid, hi_adr, first, w, n)
      inv = 1.0 / float(n * n)
      mean = hdr * inv
      store_hdr(out, worldid, adr, mean)
      if env.params.skip_display == 0:
        h = cams.camera_res[camid][1]
        display = display_of(env, worldid, mean * e)
        write_display(out, worldid, adr, cams.lens[camid].adr, w * h, display, coverage(out, covered, inv / 255.0))
    if (wants & wp.static(WANT_DEPTH)) != 0:
      if out_half(out.half, wp.static(OUT_DEPTH)):
        f16s(out, out.table.depth16, 1)[worldid, adr] = f16s(hi, hi.table.depth16, 1)[worldid, hi_adr + first]
      else:
        f32s(out, out.table.depth)[worldid, adr] = f32s(hi, hi.table.depth)[worldid, hi_adr + first]
    if (wants & wp.static(WANT_SEG)) != 0:
      i32s(out, out.table.seg)[worldid, adr] = i32s(hi, hi.table.seg)[worldid, hi_adr + first]
    first_hit.copy_first_hit(hi, out, worldid, hi_adr + first, adr)

  return downsample


def program_features(f: Features) -> Features:
  """The Features of the program of a supersampled build `f` (spp n > 1): spp 1, the n x
  resolution outputs (`hi_outputs`: their word is the frame's, `Supersampling.hi`). The render's
  (`Supersampling.features`) and the bundle fill's (`tools/prebuilt/fill.py`)."""
  return dataclasses.replace(f, spp=1)


def hi_outputs(outputs: Outputs) -> Outputs:
  """The outputs of the supersampled image of a render writing `outputs`: a float32 hdr (the
  downsample tone maps it), the color as packed RGBA (its alpha is averaged; the downsample
  writes the requested color layout and the luminance), the rest as requested."""
  rgb = ColorOutput(alpha=outputs.rgb.alpha) if outputs.rgb is not None else None
  return dataclasses.replace(outputs, hdr=FloatOutput(), rgb=rgb, luminance=None)


def module_outputs(f: Features, outputs: Outputs) -> Outputs:
  """The outputs the OptiX programs of build `f` store for a render writing `outputs` (their
  module's output word): the supersampled image's (`hi_outputs`) with spp > 1 and a shaded
  output, else `outputs`."""
  return hi_outputs(outputs) if f.spp > 1 and f.any_rgb else outputs


@final
class Supersampling:
  """The supersampled image of a Renderer with spp n > 1 (see the module doc).

  Attributes:
    hi: the n x resolution buffers the program renders (every output, plus HDR: `buffers`).
    features: the program's Features (spp 1, HDR, n² times the rays).
  """

  def __init__(self, ctx: FrameContext, n: int) -> None:
    """Allocates the supersampled buffers of the renderer of `ctx` (n x n samples per pixel)."""
    f = ctx.features
    c = ctx.cameras
    self._sizes = [(w * n, h * n) for w, h in c.sizes]
    self._outputs: Outputs | None = None
    self.hi = self.buffers(ctx)
    self.features = program_features(f)
    self.n = n
    self.kernel = downsample_kernel(f)
    self.camera_res = wp.array(np.asarray(self._sizes, np.int32).reshape(-1, 2), dtype=wp.vec2i, device=ctx.buffers.device)
    self._rig = c

  def buffers(self, ctx: FrameContext) -> Buffers:
    """The supersampled buffers of the outputs `ctx`'s settings write, as `hi_outputs` stores them:
    reallocated when those change (the outputs are run-time data: the same programs)."""
    outputs = module_outputs(ctx.features, ctx.settings.written_outputs())
    if outputs != self._outputs:
      self.hi = Buffers(outputs, nworld=ctx.nworld, sizes=self._sizes, device=ctx.buffers.device)
      self.hi.flat["seg"].fill_(-1)
      self._outputs = outputs
    return self.hi

  def cameras(self, cams: Cameras) -> Cameras:
    """The cameras at the supersampled resolution (ray bundles keep their own resolution)."""
    hi = Cameras()
    for name in warp_struct(Cameras).vars:
      setattr(hi, name, getattr(cams, name))
    hi.camera_res = self.camera_res
    hi.lens = self._rig.scaled_lens(self.n)
    hi.nray = cams.nray * self.n * self.n
    return hi

  def downsample(self, ctx: FrameContext, cams: Cameras, env: EnvironmentArrays, hi: OutputArrays, out: OutputArrays) -> None:
    """Averages the samples of `hi` into `out` (the renderer's outputs); `env`: the frame's
    environment struct (exposure, tone mapper, look)."""
    inputs = [cams, self._rig.scaled_lens(self.n), env, hi]
    wp.launch(self.kernel, dim=(ctx.nworld, cams.nray), inputs=inputs, outputs=[out], device=ctx.buffers.device)
