"""The settings groups of the camera's imaging pipeline: `exposure` (automatic exposure, the
default, or a manual EV100) and `grading` (camera white balance and colour grading); all but
automatic exposure off by default (the chromatic aberration: `sensor.aberration`).

What runs where (`core/post/display/deferred.py`): automatic exposure, chromatic aberration
(`sensor.aberration`) and the grading read the frame's scene-linear HDR image, so each turns on the
deferred display pass (the "hdr" output is then written; `deferred`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final, Literal, final

from raywarp.settings.fields import DEFAULT_EV100, is_rgb, rgb_tuple, runtime_field, validate

type ExposureMode = Literal["auto", "manual"]
type MeteringMode = Literal["average", "center", "spot"]
type Adaptation = Literal["locked", "continuous", "per_frame"]

# Device ids of the runtime choices (`core/post/exposure/auto.py`).
METERING: Final = {"average": 0, "center": 1, "spot": 2}
ADAPTATION: Final = {"locked": 0, "continuous": 1, "per_frame": 2}
# The D65 white point's correlated colour temperature [K]: the white balance that changes nothing.
D65_KELVIN: Final = 6504.0
# Middle grey: the exposed luminance the meter shows the metered mean at by default
# (`Exposure.key`).
MIDDLE_GREY: Final = 0.18


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Exposure:
  """The camera exposure: automatic per camera (`mode` "auto", the default) or a manual EV100
  (`mode` "manual": every image is exposed by 2^-`ev100`, whatever the tone mapper).

  Under automatic exposure the render kernels expose at the reference `render_ev100()` and the
  deferred display rescales every view to its metered exposure. Switching `mode` turns that pass
  on or off (run time: the deferred display's modules are prebuilt and the HDR image it reads is an
  output word, no render kernel changes); changing `ev100` or a metering field never recompiles.

  Metering reads each camera image's scene-linear luminance L (before exposure, the "hdr" output's
  units; on the centres of a grid of at least 128 x 128 cells; pixels of zero luminance, such as
  misses without an environment, are not metered) into a 256-bin histogram of EV100 = log2(L x 100 /
  12.5) over [-8, 24] (integer pixel weights: the same frame always gives the same exposure), and
  averages the EV of the pixels between 10 % and 90 % of the weighted pixel count (outliers
  rejected: a hot pixel or the sun disc does not darken the frame). That mean shows at the exposed
  luminance `key` (middle grey). Highlight protection then darkens the exposure by at most 1 stop,
  and only for large bright areas: by the mean overshoot (in stops) of the pixels that would show
  above 2.0, scaled by a smooth ramp of their share of the frame (nothing below 2 %, all of it from
  25 %). The scene is exposed, not its light sources: a window or a lamp in view does not crush the
  room, and the exposure changes continuously as a bright area enters the view
  (`post/exposure/auto.py`). The exposure is 2^-(EV - compensation) (`ev100`'s convention),
  times the camera's own compensation (`CameraSpec.exposure`), with EV clamped to [min_ev, max_ev].

  `adaptation` says how a camera's EV evolves over frames (per world and camera):
  - "locked" (the default): the first frame with any metered pixel is metered, displayed at its own
    EV, and the EV is kept; `Renderer.reset_history(worlds)` (episode resets, re-randomisation,
    camera cuts) re-meters the next frame. Later frames do not meter (no metering cost).
  - "continuous": every frame is metered and the EV follows the metered one exponentially, EV +=
    (target - EV) (1 - exp(-frame_time x speed)), 3 / s when the scene gets brighter and 1 / s when
    it gets darker (Unreal's speeds): smooth, never overshooting. The first frame and a reset take
    the metered EV at once.
  - "per_frame": every frame is metered on its own (stateless, deterministic per frame).
  In every mode the first frame (and the first after a reset) is shown at its own metered EV: the
  meter runs before the display in the same frame. Skipped worlds keep their state.

  Attributes:
    mode: "auto" (metered, the default) or "manual" (`ev100`) (R but for the switch, above).
    ev100: the manual exposure value (R; read under "manual" only).
    metering: pixel weights: "average" (all equal), "center" (a smooth fall-off around the image
      centre, a fifth of the half height wide) or "spot" (a disc of that radius) (R).
    key: exposed luminance of the metered mean (middle grey) (R).
    compensation: exposure compensation [EV] (+1: one stop brighter) (R).
    min_ev, max_ev: range of the metered EV100 (R).
    adaptation: "locked", "continuous" or "per_frame" (above) (R; the state is always kept).
    frame_time: time between two renders [s]: the continuous adaptation's clock (R).
  """

  mode: ExposureMode = runtime_field("auto")  # noqa: RUF009  # a dataclasses.field
  ev100: float = runtime_field(DEFAULT_EV100, gt=-math.inf, lt=math.inf)
  metering: MeteringMode = runtime_field("average")  # noqa: RUF009  # a dataclasses.field
  key: float = runtime_field(MIDDLE_GREY, gt=0, lt=math.inf)
  compensation: float = runtime_field(0.0, gt=-math.inf, lt=math.inf)
  min_ev: float = runtime_field(-6.0, gt=-math.inf, lt=math.inf)
  max_ev: float = runtime_field(20.0, gt=-math.inf, lt=math.inf)
  adaptation: Adaptation = runtime_field("locked")  # noqa: RUF009  # a dataclasses.field
  frame_time: float = runtime_field(1.0 / 30.0, gt=0, lt=math.inf)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "exposure")
    if not self.min_ev < self.max_ev:
      raise ValueError(f"need exposure.min_ev < exposure.max_ev, got {self.min_ev} and {self.max_ev}")

  def render_ev100(self) -> float:
    """The EV100 the render kernels expose at: `ev100` under "manual", else the reference
    DEFAULT_EV100 that the display-relative quantities (MuJoCo lights and emission, the background
    and fog colors, `Lighting`'s default) are calibrated for."""
    return self.ev100 if self.mode == "manual" else DEFAULT_EV100


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Grading:
  """Camera white balance and colour grading, applied by the deferred display pass before the
  display transform (Filament's colour grading order, scene-referred before the tone mapper, then
  display-referred).

  1. White balance: Bradford chromatic adaptation from the white of a `kelvin` / `tint`
     illuminant (Planckian locus, tint along the isotherm in Duv / 100) to D65. It is the camera's
     setting, not the lights' colour: a tungsten-lit (2856 K) grey card balanced at 2856 K renders
     neutral.
  2. Lift / gamma / gain: the ASC CDL (offset = lift, power = 1 / gamma, slope = gain) in LogC.
  3. Saturation around Rec. 709 luminance.
  4. The tone mapper (`post.tone_mapper`).

  White balance, saturation and lift / gamma / gain are per world when declared
  (`PerWorld(env=["grading.white_balance", "grading.saturation", "grading.lift",
  "grading.gamma", "grading.gain"])`: (kelvin, tint), saturation, rgb, rgb, rgb).

  Attributes:
    enabled: grade the image: the switch of the values below, also of their per-world tables
      (off: they are ignored, every world ungraded; on: applied, identity values grading
      nothing at the pass's cost). Run time (R): it selects the deferred display's graded kernel
      (both prebuilt) and turns that pass on, the render kernels writing the HDR image it reads
      as with automatic exposure (a new output word under manual exposure). Not derived from the
      values: per-world values may start at identity and be randomised later.
    kelvin, tint: the white balance (R).
    lift, gamma, gain: rgb (R; gamma > 0).
    saturation: scale of the chroma around the luminance, 1: unchanged, 0: grey (R).
  """

  enabled: bool = runtime_field(False)
  kelvin: float = runtime_field(D65_KELVIN, ge=1000, le=40000)
  tint: float = runtime_field(0.0, ge=-1, le=1)
  lift: tuple[float, float, float] = runtime_field((0.0, 0.0, 0.0))
  gamma: tuple[float, float, float] = runtime_field((1.0, 1.0, 1.0))
  gain: tuple[float, float, float] = runtime_field((1.0, 1.0, 1.0))
  saturation: float = runtime_field(1.0, ge=0, le=4)

  def __post_init__(self) -> None:
    """Validates the group (ValueError for an invalid value)."""
    validate(self, "grading")
    for name in ("lift", "gamma", "gain"):
      value = getattr(self, name)
      if not is_rgb(value):
        raise ValueError(f"grading.{name} must be an rgb triple, got {value!r}")
      object.__setattr__(self, name, rgb_tuple(value))
    if min(self.gamma) <= 0:
      raise ValueError("grading.gamma must be positive")


def deferred(*, auto_exposure: bool, aberration: float, grading: Grading, contact_ao: float, area_rays: int) -> bool:
  """Whether these settings read the frame's HDR image (the deferred display pass): automatic
  exposure, chromatic aberration, the grading, the contact AO or the soft area shadows (the render
  kernels' display is the plain transform: a look change compiles no render kernel)."""
  return auto_exposure or aberration > 0.0 or grading.enabled or contact_ao > 0.0 or area_rays > 0
