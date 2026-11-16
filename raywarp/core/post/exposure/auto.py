"""Automatic exposure per camera image (`settings.Exposure`; Filament §8.1.4), all on the device:
no host readback, graph-capturable, per-world state.

1. `make_meter`: each image's scene luminance (the source's exposed color divided by the manual
   exposure it carries), on the centre pixels of a grid of at least METER_SAMPLES cells
   (`meter_stride`), into a 256-bin EV100 histogram, weighted by the metering mode (integer
   weights 0..64: the sums are order independent, so a frame always meters the same; pixels of
   zero luminance, e.g. misses without an environment, carry no signal and are skipped). One block
   of `BLOCK` threads takes `BLOCK x PIXELS_PER_THREAD` cells of one image and accumulates a
   shared-memory histogram (`wp.tile_scatter_add`), then adds it to the image's global row once:
   one global atomic per bin and block, whatever the image content.
2. `_solve`: per (world, camera), the weighted mean EV of the bins between the `low` and `high`
   fractions of the weight (fractional bins at both ends) shown at the key, raised by the
   highlight protection (`protection`: a continuous function of the bright area, at most
   `_HIGHLIGHT_STOPS`), minus the compensation, clamped to [min_ev, max_ev]. Then, by
   `Exposure.adaptation` ("locked" keeps the EV, "continuous" follows the target at rate
   speed_up / speed_down (1 - exp(-frame_time speed)), "per_frame" takes it); a reset or first
   camera frame takes the target directly. Writes the factor the source's exposed colors are
   scaled by (2^-EV / the world's manual exposure; the camera's compensation is kept) and clears
   the histogram for the next frame. An image with no weighted pixel takes the manual EV on its
   first frame (a locked view then meters again on its next frame) and later keeps its EV, or
   takes the manual one per frame.

The display runs after `_solve` in the same frame, so every frame, the first included, shows at
its own exposure. A locked view that has its EV skips the metering (no cost per frame).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp.core.kernels.abi.tables import exposure_at
from raywarp.core.post.exposure import source
from raywarp.core.post.exposure.source import Source, make_fetch
from raywarp.core.post.grading import color
from raywarp.settings.imaging import ADAPTATION, METERING

if TYPE_CHECKING:
  from raywarp.settings.imaging import Exposure

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Histogram bins and their EV100 range.
BINS: Final = 256
EV_LO: Final = -8.0
EV_HI: Final = 24.0
# Threads per metering block and pixels each thread meters.
BLOCK: Final = 256
PIXELS_PER_THREAD: Final = 16
# Pixels metered per image, at least: an image is metered on the centres of `meter_stride` x
# `meter_stride` cells (stride 1 below 4 x this). 128^2 samples put a 1/8 EV bin's share of the
# histogram within 1 % (binomial, any percentile), and the meter reads 1/16 of a 512^2 image.
METER_SAMPLES: Final = 1 << 14
# Largest integer pixel weight.
WEIGHT_MAX: Final = 64
# EV100 = log2(L x S / K) with S = 100, K = 12.5 (the reflected-light meter equation).
_METER_SCALE: Final = 100.0 / 12.5
# The exposed luminance of the metered luminance at its own EV100: 2^-EV100 L = K / S.
_METER_KEY: Final = 12.5 / 100.0
# The histogram band: the fractions of the weighted pixel count below / above which pixels are
# not metered (Unreal's default 10-90 %: a hot pixel or the sun disc does not darken the frame).
_LOW: Final = 0.1
_HIGH: Final = 0.9
# The highlight protection (`protection`): the pixels that would show above _WHITE (exposed
# luminance; PBR Neutral 0.96, i.e. clipped) darken the exposure by their mean overshoot in stops,
# scaled by a smoothstep of their weighted share of the frame from _AREA_LO to _AREA_HI, by at
# most _HIGHLIGHT_STOPS. A small bright area (a lamp, the sun disc, a specular highlight) is
# allowed to clip; a large one (a sunlit floor) keeps some texture; a window never crushes the
# room it lights. The percentile pairs used before (the brightest 5 % at <= 2.0 and 20 % at <=
# 0.9, up to 3 stops) metered interiors with a bright window 1.3 stops too dark (country kitchen:
# mean 45/255 against 105 with the window out of view) and, since a percentile jumps across the
# gap of a two-level histogram (room / window), made the exposure jump frame to frame (bedroom
# tour: +/- 6-8 / 255, max 30). Every term here is continuous in the image: the share and the
# overshoot use fractional bins, and the ramp starts at 0.
_WHITE: Final = 2.0
_AREA_LO: Final = 0.02
_AREA_HI: Final = 0.25
_HIGHLIGHT_STOPS: Final = 1.0
# Adaptation rates [1/s] to brighter / darker scenes (Unreal's defaults).
_SPEED_UP: Final = 3.0
_SPEED_DOWN: Final = 1.0
# Radius of the spot and scale of the centre weighting, a fraction of half the image height.
_SPOT: Final = 0.2


@wp.struct
class Metering:
  """The runtime metering settings (`settings.Exposure`).

  Attributes:
    mode: `settings.imaging.METERING` id.
    low, high: the histogram's rejected fractions.
    key: log2(key / 0.125) [EV]: the metered mean shows at `Exposure.key`.
    white: log2(_WHITE / 0.125) [EV]: a pixel this far above the exposure's EV clips.
    area_lo, area_hi: the bright share's ramp (`_AREA_LO`, `_AREA_HI`).
    highlight_stops: most the highlight protection darkens [EV].
    compensation, min_ev, max_ev: [EV].
    adaptation: `settings.imaging.ADAPTATION` id.
    rate_up, rate_down: 1 - exp(-frame_time x speed).
    spot: the spot radius / centre scale (fraction of half the image height).
  """

  mode: int
  low: float
  high: float
  key: float
  white: float
  area_lo: float
  area_hi: float
  highlight_stops: float
  compensation: float
  min_ev: float
  max_ev: float
  adaptation: int
  rate_up: float
  rate_down: float
  spot: float


@wp.func
def ev_bin(
  # In:
  luminance: float,
) -> int:
  """The histogram bin of scene luminance `luminance` (<= 0: bin 0)."""
  if not (luminance > 0.0):
    return 0
  ev = wp.log2(luminance * wp.static(_METER_SCALE))
  b = int(wp.floor((ev - wp.static(EV_LO)) * wp.static(BINS / (EV_HI - EV_LO))))
  return wp.clamp(b, 0, wp.static(BINS - 1))


@wp.func
def weight(
  # In:
  m: Metering,
  x: int,
  y: int,
  width: int,
  height: int,
) -> int:
  """Integer metering weight (0..WEIGHT_MAX) of pixel (x, y)."""
  if m.mode == wp.static(METERING["average"]):
    return wp.static(WEIGHT_MAX)
  half = 0.5 * float(height)
  dx = (float(x) + 0.5 - 0.5 * float(width)) / half
  dy = (float(y) + 0.5 - half) / half
  r = wp.sqrt(dx * dx + dy * dy) / m.spot
  if m.mode == wp.static(METERING["spot"]):
    return wp.where(r <= 1.0, wp.static(WEIGHT_MAX), 0)
  # Centre-weighted: a Gaussian of the radius, never below 1 (the whole frame still counts).
  return wp.max(int(wp.round(float(wp.static(WEIGHT_MAX)) * wp.exp(-0.5 * r * r))), 1)


@functools.cache
def make_meter(kind: int):
  """The metering kernel of source `kind` (launch_tiled over (launch units, blocks), block
  `BLOCK`)."""
  fetch = make_fetch(kind)

  @wp.kernel(module="unique", enable_backward=False)
  def meter(
    # In:
    src: Source,
    m: Metering,
    cam: int,
    width: int,
    height: int,
    stride: int,
    ncam: int,
    valid: wp.array2d[wp.int32],
    # Out:
    histogram: wp.array2d[wp.int32],
  ):
    """One block of one image's metered cells: its shared histogram, then one atomic add per bin
    (nothing for a locked view that has its EV)."""
    unit, block, lane = wp.tid()
    w = source.unit_world(src, unit)
    if w < 0:
      return  # a hole of the active list (the whole block: one unit per block row)
    bins = wp.tile_zeros(shape=wp.static(BINS), dtype=wp.int32, storage="shared")
    cols = (width + stride - 1) // stride
    ncell = cols * ((height + stride - 1) // stride)
    skip = locked(m, valid, w, cam)
    inv = 1.0 / source.manual_exposure(src, w, cam)
    first = block * wp.static(BLOCK * PIXELS_PER_THREAD)
    # Threads of the block (BLOCK on CUDA; one lane on the CPU, which then walks every pixel).
    lanes = wp.block_dim()
    for k in range(wp.static(BLOCK * PIXELS_PER_THREAD) // lanes):
      c = first + k * lanes + lane
      b = int(0)
      wgt = int(0)
      if c < ncell and not skip:
        # The cell's centre pixel (clamped into a partial last cell).
        cy = c // cols
        x = wp.min((c - cy * cols) * stride + stride // 2, width - 1)
        y = wp.min(cy * stride + stride // 2, height - 1)
        lum = color.luminance(fetch(src, w, cam, y * width + x)) * inv
        b = ev_bin(lum)
        # Pixels without light (a miss without environment: no signal) are not metered.
        if lum > 0.0:
          wgt = weight(m, x, y, width, height)
      wp.tile_scatter_add(bins, b, wgt, wgt > 0)
    wp.tile_atomic_add(histogram[w * ncam + cam], bins)

  return meter


@wp.func
def histogram_ev(
  # In:
  m: Metering,
  histogram: wp.array2d[wp.int32],
  row: int,
) -> wp.vec2:
  """(mean EV100 of the metered bins, total weight of the row) of a histogram row (fractional
  bins at the band's ends; mean 0 for an empty row)."""
  total = float(0.0)
  for b in range(wp.static(BINS)):
    total += float(histogram[row, b])
  lo = total * m.low
  hi = total * m.high
  below = float(0.0)
  num = float(0.0)
  den = float(0.0)
  for b in range(wp.static(BINS)):
    c = float(histogram[row, b])
    # The part of this bin between the cumulative counts lo and hi.
    take = wp.max(wp.min(below + c, hi) - wp.max(below, lo), 0.0)
    num += take * (wp.static(EV_LO) + (float(b) + 0.5) * wp.static((EV_HI - EV_LO) / BINS))
    den += take
    below += c
  # The band holds (high - low) of the weight: below 1 for a lone metered pixel of weight 1.
  return wp.vec2(wp.where(den > 0.0, num / wp.max(den, 1.0e-30), 0.0), total)


@wp.func
def locked(
  # In:
  m: Metering,
  valid: wp.array2d[wp.int32],
  w: int,
  cam: int,
) -> bool:
  """Whether view (w, cam) keeps its EV this frame (locked adaptation, already metered)."""
  return m.adaptation == wp.static(ADAPTATION["locked"]) and valid[w, cam] != 0


@wp.func
def protection(
  # In:
  m: Metering,
  histogram: wp.array2d[wp.int32],
  row: int,
  total: float,
  ev: float,
) -> float:
  """How far [EV] the highlight protection darkens exposure `ev` of a histogram row of weight
  `total` (`_WHITE`, `_AREA_LO`, `_AREA_HI`): the mean overshoot of the clipped weight times a
  smoothstep of its share, at most highlight_stops (fractional bins: continuous in `ev` and in
  the image)."""
  width = wp.static((EV_HI - EV_LO) / BINS)
  clip = ev + m.white
  share = float(0.0)
  over = float(0.0)
  for b in range(wp.static(BINS)):
    lo = wp.static(EV_LO) + float(b) * width
    part = wp.clamp((lo + width - clip) / width, 0.0, 1.0)
    if part > 0.0:
      c = float(histogram[row, b]) * part
      share += c
      over += c * (0.5 * (wp.max(lo, clip) + lo + width) - clip)
  if not (share > 0.0):
    return 0.0
  t = wp.clamp((share / total - m.area_lo) / (m.area_hi - m.area_lo), 0.0, 1.0)
  return wp.min(over / share, m.highlight_stops * t * t * (3.0 - 2.0 * t))


@wp.func
def target_ev(
  # In:
  m: Metering,
  histogram: wp.array2d[wp.int32],
  row: int,
  measured: wp.vec2,
) -> float:
  """The exposure EV100 of a metered histogram row (`histogram_ev`): the band mean at the key,
  darkened by the highlight protection; minus the compensation, clamped to [min_ev, max_ev]."""
  ev = measured[0] - m.key
  ev += protection(m, histogram, row, measured[1], ev)
  return wp.clamp(ev - m.compensation, m.min_ev, m.max_ev)


@wp.kernel
def _solve(
  # In:
  src: Source,
  m: Metering,
  ncam: int,
  # Data in / out:
  histogram: wp.array2d[wp.int32],
  ev: wp.array2d[float],
  valid: wp.array2d[wp.int32],
  # Out:
  scale: wp.array2d[float],
  view: wp.array1d[float],
):
  """The exposure of view (w, cam) from its histogram (see the module doc); `view`: camera 0's
  exposure scale 2^-EV per world (the next frame's display-relative fill,
  `Environment.view_exposure`)."""
  unit, cam = wp.tid()
  w = source.unit_world(src, unit)
  if w < 0:
    return  # a hole of the active list
  row = w * ncam + cam
  # The world's exposure (the colors' manual exposure without the camera's compensation): read
  # directly, not divided back out of the product (an underflowing product gave 0 / NaN).
  manual = exposure_at(src.exposure, w)
  first = valid[w, cam] == 0
  prev = wp.where(first, -wp.log2(manual), ev[w, cam])
  per_frame = m.adaptation == wp.static(ADAPTATION["per_frame"])
  out = prev
  if not locked(m, valid, w, cam):
    measured = histogram_ev(m, histogram, row)
    # No metered pixel: a view keeps its EV, or takes the manual one per frame (and on its
    # first frame, which a locked view then meters again).
    target = wp.where(per_frame, -wp.log2(manual), prev)
    if measured[1] > 0.0:
      target = target_ev(m, histogram, row, measured)
      valid[w, cam] = 1
    elif m.adaptation != wp.static(ADAPTATION["locked"]):
      valid[w, cam] = 1
    out = target
    if m.adaptation == wp.static(ADAPTATION["continuous"]) and not first:
      rate = wp.where(target > prev, m.rate_up, m.rate_down)
      out = prev + (target - prev) * rate
  ev[w, cam] = out
  scale[w, cam] = wp.pow(2.0, -out) / manual
  if cam == 0:
    view[w] = wp.pow(2.0, -out)
  for b in range(wp.static(BINS)):
    histogram[row, b] = 0


@wp.kernel
def _reset(
  # In:
  worlds: wp.array1d[bool],
  ncam: int,
  # Out:
  valid: wp.array2d[wp.int32],
  view: wp.array1d[float],
):
  """Forgets the adaptation (and the metered exposure) of the masked worlds' cameras."""
  w = wp.tid()
  if worlds[w]:
    view[w] = 0.0
    for c in range(ncam):
      valid[w, c] = 0


# The solve and reset kernels (`deferred.first_frame_kernels`: the warm-up list).
STATE_KERNELS: Final = (_solve, _reset)


def meter_stride(width: int, height: int) -> int:
  """The metering cell size of a `width` x `height` image: the largest that keeps at least
  METER_SAMPLES cells (1 below 4 x METER_SAMPLES pixels)."""
  return max(1, math.isqrt(width * height // METER_SAMPLES))


def metering(settings: Exposure) -> Metering:
  """The `Metering` struct of the settings."""
  m = Metering()
  m.mode = METERING[settings.metering]
  m.low, m.high = _LOW, _HIGH
  m.key = math.log2(settings.key / _METER_KEY)
  m.white = math.log2(_WHITE / _METER_KEY)
  m.area_lo, m.area_hi = _AREA_LO, _AREA_HI
  m.highlight_stops = _HIGHLIGHT_STOPS
  m.compensation, m.min_ev, m.max_ev = float(settings.compensation), float(settings.min_ev), float(settings.max_ev)
  m.adaptation = ADAPTATION[settings.adaptation]
  m.rate_up = 1.0 - math.exp(-settings.frame_time * _SPEED_UP)
  m.rate_down = 1.0 - math.exp(-settings.frame_time * _SPEED_DOWN)
  m.spot = _SPOT
  return m


@final
class AutoExposure:
  """The metering state of a Renderer's cameras: (nworld x ncam) histograms, EV, validity and
  the per-view scale the deferred display applies (`scale`)."""

  def __init__(self, device: wp.Device, nworld: int, ncam: int) -> None:
    """Zero state for `nworld` worlds of `ncam` cameras (every view starts unadapted)."""
    self.device, self.nworld, self.ncam = device, nworld, ncam
    self.histogram: wp.array[wp.int32, [int, int]] = wp.zeros((nworld * ncam, BINS), dtype=wp.int32, device=device)
    self.ev: wp.array[float, [int, int]] = wp.zeros((nworld, ncam), dtype=float, device=device)
    self.valid: wp.array[wp.int32, [int, int]] = wp.zeros((nworld, ncam), dtype=wp.int32, device=device)
    self.scale: wp.array[float, [int, int]] = wp.ones((nworld, ncam), dtype=float, device=device)
    # Camera 0's metered exposure scale per world (0: none yet, the reference exposure applies).
    self.view: wp.array[float, [int]] = wp.zeros(nworld, dtype=float, device=device)
    self.metering = Metering()

  def configure(self, settings: Exposure) -> None:
    """Takes the runtime settings."""
    self.metering = metering(settings)

  def run(self, src: Source, kind: int, sizes: list[tuple[int, int]], nunit: int) -> None:
    """Meters every view of the frame's `nunit` launch units (`Source.worlds`) and solves its
    exposure (`sizes`: each camera's (W, H))."""
    meter = make_meter(kind)
    for cam, (width, height) in enumerate(sizes):
      stride = meter_stride(width, height)
      cells = -(-width // stride) * -(-height // stride)
      blocks = -(-cells // (BLOCK * PIXELS_PER_THREAD))
      wp.launch_tiled(
        meter,
        dim=(nunit, blocks),
        inputs=[src, self.metering, cam, width, height, stride, self.ncam, self.valid],
        outputs=[self.histogram],
        block_dim=BLOCK,
        device=self.device,
      )
    wp.launch(
      _solve,
      dim=(nunit, self.ncam),
      inputs=[src, self.metering, self.ncam, self.histogram, self.ev, self.valid],
      outputs=[self.scale, self.view],
      device=self.device,
    )

  def reset(self, worlds: wp.array[bool, [int]] | None) -> None:
    """The masked worlds' cameras (None: all) take their next metering without adaptation."""
    if worlds is None:
      self.valid.zero_()
      self.view.zero_()
      return
    wp.launch(_reset, dim=self.nworld, inputs=[worlds, self.ncam], outputs=[self.valid, self.view], device=self.device)
