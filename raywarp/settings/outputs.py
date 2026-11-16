"""The `outputs` settings group: which images a Renderer writes, one typed request per image.

Each output is a field of `Outputs` holding its spec (`ColorOutput`, `DepthOutput`, ...), or None
when it is not requested; only requested outputs are allocated and written. The outputs are run-time
data (an output word each frame reads: requesting another one compiles nothing), except `motion`
(its temporal history is a build of its own) and a request without a shaded output (the
geometry-only form, prebuilt). What each output holds (first hit, misses, spp behavior) is
documented on `RenderOutputs` and in docs/robotics.md ("Outputs: what you can request and what it
costs").

    rw.Outputs(rgb=rw.settings.ColorOutput(channels=3), depth=rw.settings.DepthOutput())
    rw.Outputs.of("rgb", "depth", "seg")  # default specs by name
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Iterable, Mapping
from typing import Final, Literal, TypeGuard, final

import numpy as np

from raywarp._base.named import construct_named, field_value
from raywarp.settings.fields import LAYOUTS, ColorLayout, MissDepth, OutputKind, choices, validate

# The element precision of a float output.
type Precision = Literal["float32", "float16"]


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class ColorOutput:
  """The display color, `RenderOutputs.rgb`: sRGB after exposure and the tone mapper.

  Every layout keeps the world-major axis order of every output; the layout changes the channels,
  the element type and whether the channels are interleaved or planar:

  - channels=4 (the default): (nworld, ncam, H, W) vec4ub RGBA bytes.
  - channels=3: (nworld, ncam, H, W) vec3ub RGB bytes (no alpha byte: 3 of 4 bytes).
  - channels=3, planar=True: channels-first planes, uint8 or float16 sRGB in [0, 1], (nworld *
    ncam, 3, H, W) in Warp (world-major rows: Warp arrays have at most four axes),
    (nworld, ncam, 3, H, W) from `RenderOutputs.torch()`; a policy's observation tensor, written by
    the kernels with no conversion pass.

  Attributes:
    channels: 4 (RGBA) or 3 (RGB).
    dtype: "uint8" (sRGB bytes) or "float16" (sRGB in [0, 1]; planar only).
    planar: channels-first planes (channels=3 only).
    alpha: the alpha byte (channels=4) holds the coverage (the fraction of the pixel's samples whose
      camera ray hit geometry) instead of 255 (no extra memory).
  """

  channels: Literal[3, 4] = 4
  dtype: Literal["uint8", "float16"] = "uint8"
  planar: bool = False
  alpha: bool = False

  def __post_init__(self) -> None:
    """Validates the fields (ValueError)."""
    validate(self, "outputs.rgb")
    if self.channels == 4 and (self.planar or self.dtype != "uint8"):
      raise ValueError("outputs.rgb: channels=4 is interleaved uint8 RGBA; planar or float16 color needs channels=3")
    if self.alpha and self.channels != 4:
      raise ValueError("outputs.rgb: alpha=True needs channels=4 (the alpha byte holds the coverage)")
    if self.dtype == "float16" and not self.planar:
      raise ValueError("outputs.rgb: float16 color is planar (planar=True); use the hdr output for linear radiance")

  @property
  def color_layout(self) -> int:
    """The kernels' color layout (`settings.LAYOUTS` id)."""
    name: ColorLayout = "rgba8" if self.channels == 4 else "rgb8"
    if self.planar:
      name = "rgb8_planar" if self.dtype == "uint8" else "rgb16f_planar"
    return LAYOUTS[name]


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class DepthOutput:
  """The depth of the first hit, (nworld, ncam, H, W).

  Attributes:
    convention: "planar" (distance to the camera plane along the optical axis: MuJoCo Warp,
      Isaac `distance_to_image_plane`) or "radial" (distance to the camera center along the
      ray: Isaac `distance_to_camera`); ray-bundle cameras write the distance along each ray.
    miss: the depth of rays that hit nothing: "zero" (MuJoCo Warp), "inf" (Isaac Lab) or "far"
      (the camera's zfar; inf for an unbounded camera: `mujoco.Renderer`).
    dtype: "float32" or "float16" (half the bytes; depths above 65504 m become inf).
  """

  convention: Literal["planar", "radial"] = "planar"
  miss: MissDepth = "zero"
  dtype: Precision = "float32"

  def __post_init__(self) -> None:
    """Validates the fields (ValueError)."""
    validate(self, "outputs.depth")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class IdOutput:
  """An int32 id per pixel (seg, semantic, material); no option."""


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class FloatOutput:
  """A float output (hdr, albedo, position, uv, roughness, metallic, emission, direct, indirect).

  Attributes:
    dtype: "float32" or "float16" (half the bytes; values beyond 65504 become inf).
    sqrt: float16 storing sqrt(value) per channel (needs dtype "float16"): the deferred display's
      internal HDR image when "hdr" is not requested (`RenderSettings.written_outputs`; never a
      user view). 3.7e-9 .. 4.3e9 at 2^-10 relative precision whatever the exposure, half the
      bytes of float32.
  """

  dtype: Precision = "float32"
  sqrt: bool = False

  def __post_init__(self) -> None:
    """Validates the fields (ValueError; `Outputs.parse` names the output, a spec built directly
    is named by its type: it serves every float output)."""
    validate(self, "FloatOutput")
    if self.sqrt and self.dtype != "float16":
      raise ValueError('FloatOutput(sqrt=True) needs dtype="float16"')


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class NormalOutput:
  """The world-space unit normal of the first hit, facing the camera.

  Attributes:
    frame: "shading" (the shading normal: smooth vertex normals and normal maps; a shaded
      output) or "geometric" (the hit's face / primitive normal: no shading, so a render of
      geometric outputs only takes the geometry-only path).
    dtype: "float32" or "float16".
  """

  frame: Literal["shading", "geometric"] = "shading"
  dtype: Precision = "float32"

  def __post_init__(self) -> None:
    """Validates the fields (ValueError)."""
    validate(self, "outputs.normal")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class ShadowOutput:
  """The visibility in [0, 1] of one light at the first hit (its shadow rays), (nworld, ncam, H, W).

  Attributes:
    light: the light: an index into the scene's lights, "sun" (the first directional light) or
      "strongest" (the largest intensity x luminance of its color), chosen when the Renderer is
      built.
    dtype: "float32" or "float16".
  """

  light: int | Literal["sun", "strongest"] = "strongest"  # int id: public settings; LightId inside
  dtype: Precision = "float32"

  def __post_init__(self) -> None:
    """Validates the fields (ValueError)."""
    validate(self, "outputs.shadow")
    light = self.light
    index = isinstance(light, numbers.Integral) and not isinstance(light, bool | np.bool_)
    if not (index or (isinstance(light, str) and light in ("sun", "strongest"))):
      raise ValueError(f'outputs.shadow.light must be a light index, "sun" or "strongest", got {light!r}')
    if index:
      if int(light) < 0:
        raise ValueError(f"outputs.shadow.light must be a non-negative light index, got {light}")
      object.__setattr__(self, "light", int(light))  # a NumPy integer stored as the Python int


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class LuminanceOutput:
  """The Rec. 709 luminance of the display color, sRGB-encoded: one channel (a gray camera).

  Attributes:
    dtype: "uint8" (sRGB bytes) or "float16" (sRGB in [0, 1]).
  """

  dtype: Literal["uint8", "float16"] = "uint8"

  def __post_init__(self) -> None:
    """Validates the fields (ValueError)."""
    validate(self, "outputs.luminance")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class MotionOutput:
  """The screen-space motion of the first hit to the previous frame, vec2 float32 pixels.

  Attributes:
    status: also write `RenderOutputs.flow_status`, (nworld, ncam, H, W) uint8: why each pixel's
      motion is valid or not (`RenderOutputs.FlowStatus`: VALID, MISS, OUT_OF_FRAME, BEHIND,
      OCCLUDED, UNKNOWN). Costs 1 B per pixel, plus the previous frame's depth (2 x float16 per
      pixel).
  """

  status: bool = False

  def __post_init__(self) -> None:
    """Validates the fields (ValueError)."""
    validate(self, "outputs.motion")


type OutputSpec = (
  ColorOutput | DepthOutput | IdOutput | FloatOutput | NormalOutput | ShadowOutput | LuminanceOutput | MotionOutput
)

# The spec class of each output field.
# Final[...]: inferred, the keys would be str; `KINDS`, `Outputs.kinds` iterate OutputKinds.
SPECS: Final[dict[OutputKind, type[OutputSpec]]] = {
  "rgb": ColorOutput,
  "depth": DepthOutput,
  "seg": IdOutput,
  "semantic": IdOutput,
  "hdr": FloatOutput,
  "albedo": FloatOutput,
  "normal": NormalOutput,
  "position": FloatOutput,
  "uv": FloatOutput,
  "material": IdOutput,
  "roughness": FloatOutput,
  "metallic": FloatOutput,
  "emission": FloatOutput,
  "direct": FloatOutput,
  "indirect": FloatOutput,
  "shadow": ShadowOutput,
  "luminance": LuminanceOutput,
  "motion": MotionOutput,
}
# Outputs that need the surfaces shaded (the others are geometry: a render of geometry outputs
# only skips shading entirely); "normal" only in its shading frame (`Outputs.shaded`).
_SHADED: Final = frozenset(
  {"rgb", "hdr", "albedo", "roughness", "metallic", "emission", "direct", "indirect", "shadow", "luminance"}
)


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Outputs:
  """What a Renderer writes: one field per output, its spec when requested, else None.

  Attributes:
    rgb: the display color (`ColorOutput`).
    depth: depth of the first hit (`DepthOutput`).
    seg: instance segment id of the first hit, -1 on a miss (`IdOutput`).
    semantic: semantic class id of the first hit (`Renderer.set_semantics`), 0 on a miss, 1
      unlabeled (`IdOutput`).
    hdr: linear radiance before exposure and tone mapping, averaged like rgb (`FloatOutput`).
    albedo: base color of the first hit (`FloatOutput`).
    normal: normal of the first hit (`NormalOutput`).
    position: world-space point of the first hit [m] (`FloatOutput`).
    uv: texture coordinate of the first hit (`FloatOutput`).
    material: material id of the first hit, -1 without one (`IdOutput`).
    roughness, metallic: the first hit's perceptual roughness and metallic as shaded (textured;
      the roughness widened by specular AA, always on) (`FloatOutput`).
    emission: the emission term of the first hit's radiance (`FloatOutput`).
    direct, indirect: the first hit's radiance from the lights / from everything else but its
      emission (`FloatOutput`).
    shadow: visibility of one light at the first hit (`ShadowOutput`).
    luminance: luminance of the display color (`LuminanceOutput`).
    motion: screen-space motion of the first hit (`MotionOutput`).

  Color spaces (`RenderOutputs.color_spaces`, Color Interop Forum IDs): rgb and luminance are
  `srgb_rec709_display`, after exposure and the tone mapper; hdr, albedo, emission, direct and
  indirect are `lin_rec709_scene`; the other outputs are `data`.
  """

  rgb: ColorOutput | None = None
  depth: DepthOutput | None = None
  seg: IdOutput | None = None
  semantic: IdOutput | None = None
  hdr: FloatOutput | None = None
  albedo: FloatOutput | None = None
  normal: NormalOutput | None = None
  position: FloatOutput | None = None
  uv: FloatOutput | None = None
  material: IdOutput | None = None
  roughness: FloatOutput | None = None
  metallic: FloatOutput | None = None
  emission: FloatOutput | None = None
  direct: FloatOutput | None = None
  indirect: FloatOutput | None = None
  shadow: ShadowOutput | None = None
  luminance: LuminanceOutput | None = None
  motion: MotionOutput | None = None

  def __post_init__(self) -> None:
    """Checks each field's spec type and that something is requested (ValueError, as every
    settings group reports a value of the wrong type)."""
    for kind, spec_type in SPECS.items():
      spec = field_value(self, kind)
      if spec is not None and not isinstance(spec, spec_type):
        raise ValueError(f"outputs.{kind} must be a {spec_type.__name__} or None, got {type(spec).__name__}")
    if not self.kinds:
      raise ValueError(f"outputs: request at least one output of {list(SPECS)}")

  @classmethod
  def of(cls, *kinds: OutputKind, **specs: OutputSpec) -> Outputs:
    """The outputs `kinds` with their default specs, plus `specs` given in full
    (`Outputs.of("rgb", depth=DepthOutput(miss="inf"))`; ValueError for an unknown name)."""
    _check_names([*kinds, *specs])
    return construct_named(cls, {**{k: SPECS[k]() for k in kinds}, **specs})

  @classmethod
  def parse(cls, config: str | Iterable[str] | Mapping[str, Mapping[str, object]]) -> Outputs:
    """The outputs of a plain-data config (JSON, a command line): output names with their
    default specs ("rgb,depth", spaces allowed, or ["rgb", "depth"]), or name -> spec options
    ({"depth": {"miss": "inf"}}).

    Raises:
      ValueError: for an unknown output or option, or options that are not a mapping.
    """
    if isinstance(config, str):
      config = [name.strip() for name in config.split(",") if name.strip()]
    options = dict(config) if isinstance(config, Mapping) else {name: {} for name in config}
    specs: dict[str, OutputSpec] = {}
    for k, o in _checked(options).items():
      if not isinstance(o, Mapping):
        raise ValueError(f"outputs.{k}: its options are a mapping (e.g. {{}} for the defaults), not {o!r}")
      try:
        specs[k] = construct_named(SPECS[k], dict(o))
      except (TypeError, ValueError) as e:  # TypeError: an option the spec does not have
        raise ValueError(f"outputs.{k} {dict(o)}: {e}") from e
    return construct_named(cls, specs)

  @property
  def kinds(self) -> frozenset[OutputKind]:
    """The requested outputs' names."""
    return frozenset(k for k in SPECS if field_value(self, k) is not None)

  def spec(self, kind: OutputKind) -> OutputSpec | None:
    """The spec of output `kind` (None: not requested)."""
    match value := field_value(self, kind):
      case None | ColorOutput() | DepthOutput() | IdOutput() | FloatOutput() | NormalOutput() | ShadowOutput():
        return value
      case LuminanceOutput() | MotionOutput():
        return value
      case _:
        raise TypeError(f"outputs.{kind} holds {type(value).__name__}, not an output spec")

  @property
  def shaded(self) -> bool:
    """Whether any output needs the surfaces shaded (all but depth, seg, semantic, position, uv,
    material, motion and a geometric normal)."""
    normal = self.normal is not None and self.normal.frame == "shading"
    return normal or bool(self.kinds & _SHADED)


def is_kind(name: str) -> TypeGuard[OutputKind]:
  """Whether `name` is an output's name."""
  return name in SPECS


def _check_names(names: Iterable[str]) -> None:
  """Raises ValueError for a name that is no output's."""
  unknown = sorted(n for n in names if not is_kind(n))
  if unknown:
    raise ValueError(f"unknown outputs {unknown}: the outputs are {list(SPECS)}")


def _checked[V](config: Mapping[str, V]) -> dict[OutputKind, V]:
  """`config` with its keys checked as output names (ValueError for an unknown one)."""
  _check_names(config)
  return {k: v for k, v in config.items() if is_kind(k)}


# Every output name, in field order (`Outputs`, `RenderOutputs`).
KINDS: Final = tuple(SPECS)
if choices(OutputKind) != KINDS or tuple(f.name for f in dataclasses.fields(Outputs)) != KINDS:
  raise ImportError(f"OutputKind {choices(OutputKind)}, SPECS {KINDS} and the Outputs fields differ")
