"""The `sensor` settings group: what a real camera and depth sensor add to the rendered image
(`core/post/sensor/effects.py`): exposure-time effects (motion blur, rolling shutter), the lens's
vignetting, photon shot and read noise, the ADC's quantisation, the camera response curve, and the
noise and dropouts of a depth sensor (active stereo, or a range model of time-of-flight /
structured-light sensors).

Every field is runtime (R): turning an effect on or off never recompiles (the chromatic aberration
under manual exposure aside; rolling shutter and motion blur read `outputs.motion`, whose kernels
are a build form of their own: requesting that output can compile). Everything is off by
default, and with everything off no pass runs (no cost). The effects run after the display is
final (after the deferred display), on the "rgb" (and "luminance") and "depth" outputs; the
scene-referred outputs (hdr, albedo, ...) stay clean.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from typing import Final, Literal, final

from raywarp.settings.fields import is_numbers, runtime_field, validate
from raywarp.settings.outputs import Outputs

type DepthSensor = Literal["off", "stereo", "range"]

# The camera response curve that changes nothing: (toe, shoulder, center, gamma).
IDENTITY_RESPONSE: Final = (1.0, 1.0, 0.5, 1.0)
# Element counts of the tuple fields.
_VIGNETTE_TERMS: Final = 3
_RESPONSE_TERMS: Final = 4
_RANGE_TERMS: Final = 3
# The range model's default (sigma0 [m], z0 [m], k [1/m]): Nguyen, Izadi, Lovell 2012, "Modeling
# Kinect sensor noise for improved 3D reconstruction and tracking", axial noise
# sigma_z = 0.0012 + 0.0019 (z - 0.4)^2 [m] (Kinect v1, 10-60 degrees off the surface normal).
KINECT_RANGE_NOISE: Final = (0.0012, 0.4, 0.0019)
# The largest r^2 of a pixel (the corners of a square image: r is over max(W, H)).
_CORNER_R2: Final = 0.5


def _brightens(vignette: tuple[float, ...]) -> bool:
  """Whether the falloff 1 + t (a1 + a2 t + a3 t^2), t = r^2, exceeds 1 somewhere on the image
  (t in (0, `_CORNER_R2`]): where the quadratic a1 + a2 t + a3 t^2 is positive (its largest value
  on the interval is at an end or, concave, at its vertex)."""
  a1, a2, a3 = vignette
  ts = [0.0, _CORNER_R2]
  if a3 < 0 and 0 < -a2 / (2 * a3) < _CORNER_R2:
    ts.append(-a2 / (2 * a3))
  return max(a1 + a2 * t + a3 * t * t for t in ts) > 0


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Sensor:
  """Camera and depth sensor effects (all runtime, all off by default).

  Color, in the order a camera forms the image:

  1. `rolling_shutter`: a CMOS sensor reads its rows one after the other: row y shows the scene
     `rolling_shutter` x (1 - (y + 0.5) / H) frame intervals before the last row (the frame's
     render time).
  2. `motion_blur`: the shutter stays open `motion_blur` frame intervals, ending at each row's
     readout. Both move along the first hit's motion vectors (needs `outputs.motion`; linear
     motion over the frame): at each instant a pixel shows the image at its position then, along
     its own vector or, where a faster neighbour moves over it, along that one (McGuire et al.
     2012's tile maximum: a moving object streaks over the static background next to it, up to
     `core.post.sensor.effects.MOTION_TILE` = 32 pixels past its silhouette).
  3. `vignette`: the lens's radial falloff 1 + a1 r^2 + a2 r^4 + a3 r^6 (clamped to [0, 1]; r:
     distance to the image centre over max(W, H): corners at r^2 = 0.5 on a square image),
     Isaac Lab PPISP's polynomial.
  4. `noise`: photon shot noise (Poisson, of the photo-electrons) and read noise (Gaussian,
     `read_noise` electrons rms), with display white at `full_well` electrons: per pixel and
     channel, seeded by the render seed, the world, the pixel and the world's sensor frame
     (`Renderer.reset_history` restarts it), so worlds are independent of the batch size.
  5. `bits`: the ADC quantises the linear signal to 2^bits - 1 steps (display white the last);
     0: none (the output's own encoding still quantises, e.g. 8-bit sRGB).
  6. `response`: the camera response curve (toe, shoulder, center, gamma) on the sRGB-encoded
     value x in [0, 1] (PPISP's curve in natural units): a * (x / center)^toe below the center,
     1 - b * ((1 - x) / (1 - center))^shoulder above it, then ^gamma; (1, 1, 0.5, 1) is the
     identity.

  Noise, vignette and response act on the display-linear color (after the tone mapper): the
  rendered display stands for the camera's linear signal (`full_well` electrons at white).
  `noise` is the switch for every world: per-world (full_well, read_noise) rows set its level.
  Per world (randomisation, no recompile): `PerWorld(env=["sensor.noise", "sensor.vignette",
  "sensor.response"])` ((full_well, read_noise), (a1, a2, a3), (toe, shoulder, center, gamma)).

  Chromatic aberration (`aberration`, `core/post/display/lens.py`): lateral, the red and blue images
  magnified by 1 + / - `aberration` around the image centre (colour fringing growing with the
  radius), read from the HDR image by the deferred display (before the display transform, unlike
  the color effects above). With automatic exposure (the default) changing it never recompiles;
  with manual exposure, turning it from 0 on turns the deferred display on (C), as `exposure.mode`
  does. Per world: "sensor.aberration".

  Depth, `depth`="stereo": an active-stereo sensor (Intel RealSense D400 class) with the
  rendered camera as its left (reference) imager and a second one `baseline` metres to its
  right, same focal length f [pixels]. Disparity d = f baseline / z gets Gaussian noise of
  `disparity_noise` pixels rms (Intel's depth RMS error z^2 subpixel / (f baseline)) and is
  quantised to `disparity_step` pixels (D400: 5-bit subpixel, 1 / 32); pixels seen by the left
  imager only (the band left of near objects whose disparity covers them), outside [min_depth,
  max_depth] or dropped at random (`dropout`) read the depth output's miss value. Undistorted
  perspective cameras only.

  Depth, `depth`="range": a time-of-flight or structured-light sensor, on the depth output's own
  value (planar or radial, any camera): Gaussian noise of sigma0 + k (z - z0)^2 metres rms
  (`range_noise`, default Kinect v1's axial fit: `KINECT_RANGE_NOISE`); pixels outside
  [min_depth, max_depth] or dropped at random (`dropout`) read the miss value.

  Both models drop edge pixels: where a 4-neighbour's depth differs by more than `edge_threshold`
  times the pixel's (or is a miss), the pixel is dropped with probability `edge_dropout` (the
  mixed / flying pixels and failed matches at depth discontinuities of real sensors).

  Attributes:
    rolling_shutter: row readout time over the frame interval, in [0, 1] (0: global shutter).
    motion_blur: exposure time over the frame interval, in [0, 1] (0: none).
    vignette: (a1, a2, a3) of the radial falloff ((0, 0, 0): none); it darkens: the falloff
      stays at most 1 over the image (a1 + a2 r^2 + a3 r^4 <= 0 for r^2 in [0, 0.5]).
    noise: shot and read noise on.
    full_well: photo-electrons at display white (> 0).
    read_noise: read noise [electrons rms] (>= 0).
    bits: ADC bit depth of the linear signal, 0 (none) or 1-16.
    response: (toe, shoulder, center, gamma) of the response curve (toe, shoulder, gamma > 0,
      center in (0, 1)).
    aberration: relative magnification difference of red / blue at the image edge; 0: none.
    depth: "off", "stereo" or "range".
    baseline: stereo baseline [m] (> 0; D435: 0.05).
    disparity_noise: disparity noise [pixels rms] (>= 0; D400: ~0.08).
    disparity_step: disparity quantum [pixels] (>= 0; 0: none).
    range_noise: (sigma0 [m], z0 [m], k [1/m]) of the range model's noise (sigma0, k >= 0).
    min_depth, max_depth: valid depth range [m] (planar for "stereo", the output's for "range").
    dropout: probability that a valid pixel is dropped, in [0, 1].
    edge_dropout: probability that an edge pixel is dropped, in [0, 1].
    edge_threshold: relative depth jump to a 4-neighbour that makes an edge pixel (> 0).
  """

  rolling_shutter: float = runtime_field(0.0, ge=0, le=1)
  motion_blur: float = runtime_field(0.0, ge=0, le=1)
  vignette: tuple[float, float, float] = runtime_field((0.0, 0.0, 0.0))
  noise: bool = runtime_field(default=False)
  full_well: float = runtime_field(4000.0, gt=0, lt=math.inf)
  read_noise: float = runtime_field(3.0, ge=0, lt=math.inf)
  bits: int = runtime_field(0, ge=0, le=16)
  response: tuple[float, float, float, float] = runtime_field(IDENTITY_RESPONSE)
  aberration: float = runtime_field(0.0, ge=0, le=0.1)
  depth: DepthSensor = runtime_field("off")  # noqa: RUF009  # a dataclasses.field
  baseline: float = runtime_field(0.05, gt=0, lt=math.inf)
  disparity_noise: float = runtime_field(0.08, ge=0, lt=math.inf)
  disparity_step: float = runtime_field(1.0 / 32.0, ge=0, lt=math.inf)
  range_noise: tuple[float, float, float] = runtime_field(KINECT_RANGE_NOISE)
  min_depth: float = runtime_field(0.1, ge=0, lt=math.inf)
  max_depth: float = runtime_field(10.0, gt=0, le=math.inf)
  dropout: float = runtime_field(0.0, ge=0, le=1)
  edge_dropout: float = runtime_field(0.0, ge=0, le=1)
  edge_threshold: float = runtime_field(0.05, gt=0, lt=math.inf)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "sensor")
    terms = {"vignette": _VIGNETTE_TERMS, "response": _RESPONSE_TERMS, "range_noise": _RANGE_TERMS}
    for name, count in terms.items():
      value = getattr(self, name)
      entries = list(value) if is_numbers(value) else []
      reals = [float(v) for v in entries if isinstance(v, numbers.Real) and not isinstance(v, bool)]
      if len(entries) != count or len(reals) != count or not all(math.isfinite(v) for v in reals):
        raise ValueError(f"sensor.{name} needs {count} finite numbers, got {value}")
      object.__setattr__(self, name, tuple(reals))
    sigma0, _z0, k = self.range_noise
    if min(sigma0, k) < 0:
      raise ValueError(f"sensor.range_noise needs sigma0, k >= 0, got {self.range_noise}")
    if _brightens(self.vignette):
      raise ValueError(
        f"sensor.vignette {self.vignette} brightens part of the image (the falloff is clamped to 1 there: no "
        "effect): a vignette darkens, a1 + a2 r^2 + a3 r^4 <= 0 for r^2 in [0, 0.5] (e.g. (-0.5, 0, 0))"
      )
    toe, shoulder, center, gamma = self.response
    if min(toe, shoulder, gamma) <= 0 or not 0 < center < 1:
      raise ValueError(f"sensor.response needs toe, shoulder, gamma > 0 and center in (0, 1), got {self.response}")
    if not self.min_depth < self.max_depth:
      raise ValueError(f"need sensor.min_depth < sensor.max_depth, got {self.min_depth} and {self.max_depth}")

  def color(self) -> bool:
    """Whether a color effect is on (the settings alone; per-world fields:
    `core/post/sensor/effects.py`)."""
    return self.reads_motion() or self.noise or self.bits > 0 or any(self.vignette) or self.response != IDENTITY_RESPONSE

  def check_outputs(self, outputs: Outputs) -> None:
    """Raises ValueError for effects whose input or output `outputs` does not request."""
    if self.color() and outputs.rgb is None and outputs.luminance is None:
      raise ValueError("the sensor's color effects need outputs.rgb or outputs.luminance")
    if self.reads_motion() and outputs.motion is None:
      raise ValueError("sensor.rolling_shutter / sensor.motion_blur move along the motion vectors: request outputs.motion")
    if self.depth != "off" and outputs.depth is None:
      raise ValueError(f"sensor.depth={self.depth!r} needs outputs.depth")

  def reads_motion(self) -> bool:
    """Whether an effect reads the motion vectors (rolling shutter, motion blur)."""
    return self.rolling_shutter > 0 or self.motion_blur > 0
