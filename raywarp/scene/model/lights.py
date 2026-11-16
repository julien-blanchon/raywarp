"""The canonical UsdLux light records (`Light`, `Shaping`, `LightExt`, `EnvironmentLight`; validated
by `lighting/checks.py`) and the light, environment and camera tables of a host scene (see
`table.py` for the layout).

The tables hold the UsdLux inputs, not derived values: the upload derives the device's radiance,
point intensity, angles in radians and flags (`core/pipeline/upload/lights.py::derive_lights`).
Every input of a non-linear derivation is stored in float64 (the same rule as `specular_weight`).
"""

from __future__ import annotations

import dataclasses
import math
import operator
import os
from collections.abc import Sequence
from itertools import starmap
from typing import ClassVar, Final, Literal, final

import numpy as np

from raywarp.scene.model import ies as ies_lib
from raywarp.scene.model.table import F32, F64, I32, U8, StringTable, Table, record_columns
from raywarp.scene.model.types import (
  NO_INSTANCE,
  NO_TEXTURE,
  Camera,
  Column,
  EnvFormat,
  InstanceId,
  LightKind,
  Matrices,
  Rows,
  TextureId,
)
from raywarp.scene.model.xform import light_frame

# The default light frame (shared, read-only).
_IDENTITY_FRAME: Final = np.zeros((3, 4), np.float32)
_IDENTITY_FRAME[:, :3] = np.eye(3)
_IDENTITY_FRAME.setflags(write=False)
type Falloff = Literal["inverse_square", "none"]
type Rgb = tuple[float, float, float]
type Vec3 = tuple[float, float, float] | Sequence[float]
# A colour temperature's illuminant (`scene.lighting.photometry.cct_color`).
type Illuminant = Literal["blackbody", "planckian", "daylight"]
# How a spot's cone treats the flux of a light given in lumens
# (`scene.lighting.photometry.with_power`).
type SpotMode = Literal["reflector", "absorber"]
WHITE: Final = (1.0, 1.0, 1.0)
# How an IES profile scales a light (`Shaping.ies_mode`): "mask", the profile relative to its peak
# times the light's own intensity; "measured", the file's candela values (intensity and exposure
# stay multipliers). UsdLux `ies:normalize` overrides both: the profile over its mean intensity.
type IesMode = Literal["mask", "measured"]
# Final[tuple[IesMode, ...]]: keeps the Literal members, inference would widen them to str.
IES_MODES: Final[tuple[IesMode, ...]] = ("mask", "measured")
# A light's (3, 4) world frame (stored as float32).
type LightFrame = np.ndarray[[3, 4], np.dtype[np.float32]] | np.ndarray[[3, 4], np.dtype[np.float64]]

# Angles (DistantLight `angle`, ShapingAPI cone half angle) span at most this [deg].
MAX_ANGLE_DEG: Final = 180.0
_FALLOFFS: Final = ("inverse_square", "none")
# The photometric float64 inputs a light and an environment share.
_PHOTOMETRIC: Final = (
  ("intensity", (F64, ())),
  ("exposure", (F64, ())),
  ("color", (F64, (3,))),
  ("enable_color_temperature", (U8, ())),
  ("color_temperature", (F64, ())),
)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Shaping:
  """UsdLux ShapingAPI (a spot light's cone and focus).

  Attributes:
    cone_angle: cone half angle [deg], in [0, 180] (>= 90 for a rect / disk, 180 for a sphere /
      cylinder: no cone).
    cone_softness: [0, 1].
    focus: >= 0, the emission's concentration along the axis.
    focus_tint: linear color at the focus' edge (reported when focus > 0: not rendered).
    ies: an IES LM-63 profile, the file's text (tilt file inlined: `scene.model.ies.read`); "" none.
    ies_angle_scale: UsdLux `ies:angleScale` (0: the profile's own angles).
    ies_normalize: UsdLux `ies:normalize`: the profile divided by its mean intensity (the light
      keeps the flux of the unshaped light).
    ies_mode: IesMode, without `ies_normalize`.
  """

  cone_angle: float = 90.0
  cone_softness: float = 0.0
  focus: float = 0.0
  focus_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
  ies: str = ""
  ies_angle_scale: float = 0.0
  ies_normalize: bool = False
  ies_mode: IesMode = "mask"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class LightExt:
  """raywarp extensions of a UsdLux light.

  Attributes:
    range: falloff window radius [m] (glTF `range`, MuJoCo `range`); None: no window (the
      inverse square law at any distance).
    falloff: "inverse_square" (UsdLux), or "none" (MuJoCo's classic lights: no falloff).
    relative: the intensity is display-relative (a `Light.sun` / `point` / `spot` made without
      one): `scene.lighting.rig.Lighting` resolves it at its `ev100` when it adds the light; a scene
      never holds one (`SceneBuilder.add_light` rejects it).
  """

  range: float | None = None
  falloff: Falloff = "inverse_square"
  relative: bool = False


# No slots: the table is built from each record's `vars` (`table.record_columns`).
@dataclasses.dataclass(frozen=True, kw_only=True)
class Light:
  """A light: the UsdLux inputs, UsdLux names (snake case) and defaults, physical units.
  The device values (radiance, point intensity, ...) are derived at upload
  (`core/pipeline/upload/lights.py::derive_lights`).

  Attributes:
    kind: LightKind: DISTANT, SPHERE, RECT, DISK, CYLINDER (UsdLux); CAMERA (the headlight),
      AMBIENT (raywarp display lights).
    intensity: radiance scale [nits]; DistantLight: illuminance [lux]; a zero-radius sphere:
      luminous intensity [cd] (projected area 1).
    exposure: [stops], the radiance is x 2^exposure.
    color: linear Rec.709, >= 0.
    enable_color_temperature: multiply by blackbody(color_temperature).
    color_temperature: [K], > 0.
    normalize: divide the radiance by the surface area in stage units (DistantLight: ignored).
    meters_per_unit: the stage's, for `normalize`.
    radius, width, height, length: sphere / disk / cylinder radius, rect width (local X) and
      height (local Y), cylinder length (local X) [m], world sizes (the transform's scale applied).
    angle: DistantLight angular diameter [deg].
    treat_as_point, treat_as_line: a sphere emitting from its centre, a cylinder from its axis.
    diffuse, specular: diffuse / specular lobe multipliers, >= 0.
    shaping: ShapingAPI, None when not applied.
    shadow_enable: ShadowAPI `shadow:enable`.
    visible_in_primary_ray: camera rays see the light's emitting surface (Omniverse RTX's
      `visibleInPrimaryRay`, false by default for every light with a surface: UsdLux lights have
      no geometry, so a renderer shows them only when asked; a dome is the background whatever
      this, a distant light has no surface in view). Area lights only (rect, disk, sphere,
      cylinder with a size).
    texture: RectLight texture (not rendered: dropped at upload).
    xform: (3, 4) orthonormal world frame (emits along local -Z).
    body: name of the rigid body the light moves with (None: static), resolved by the builder.
    instance: the emissive instance this light stands for (`scene.lighting.emissive`), NO_INSTANCE
    for an
      authored light.
    path: source path.
    ext: raywarp extensions.
  """

  kind: LightKind
  intensity: float = 1.0
  exposure: float = 0.0
  color: tuple[float, float, float] = (1.0, 1.0, 1.0)
  enable_color_temperature: bool = False
  color_temperature: float = 6500.0
  normalize: bool = False
  meters_per_unit: float = 1.0
  radius: float = 0.5
  width: float = 1.0
  height: float = 1.0
  length: float = 1.0
  angle: float = 0.53
  treat_as_point: bool = False
  treat_as_line: bool = False
  diffuse: float = 1.0
  specular: float = 1.0
  shaping: Shaping | None = None
  shadow_enable: bool = True
  visible_in_primary_ray: bool = False
  texture: TextureId = NO_TEXTURE
  xform: LightFrame = dataclasses.field(default_factory=lambda: _IDENTITY_FRAME)
  body: str | None = None
  instance: InstanceId = NO_INSTANCE
  path: str = ""
  ext: LightExt = LightExt()

  @property
  def direction(self) -> np.ndarray[[3], np.dtype[np.float64]]:
    """The direction the light emits along (its frame's -Z axis)."""
    return -np.asarray(self.xform[:, 2], np.float64)

  @classmethod
  def sun(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    direction: Vec3,
    *,
    illuminance: float | None = None,
    color: Rgb = WHITE,
    angle: float = 0.0,
    temperature: float | None = None,
    illuminant: Illuminant = "blackbody",
    shadows: bool = True,
    path: str = "",
  ) -> Light:
    """A distant light travelling along `direction`: `illuminance` [lux] x `color` on a surface
    facing it (None: display-relative, a white Lambertian surface facing it shows `color` at the
    `Lighting`'s ev100); `angle`: angular diameter [deg] (0: hard shadows; the sun is 0.53);
    `temperature` [K] tints `color` by the `illuminant`'s colour of luminance 1."""
    tint = _tinted(color, temperature, illuminant)
    frame = light_frame(direction, (0.0, 0.0, 0.0))
    value = 1.0 if illuminance is None else illuminance
    ext = LightExt(relative=illuminance is None)
    return cls(
      kind=LightKind.DISTANT, intensity=value, color=tint, angle=angle, shadow_enable=shadows, xform=frame, path=path, ext=ext
    )

  @classmethod
  def point(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    position: Vec3,
    *,
    intensity: float | None = None,
    power: float | None = None,
    radius: float = 0.0,
    color: Rgb = WHITE,
    temperature: float | None = None,
    illuminant: Illuminant = "blackbody",
    shadows: bool = True,
    ies: Ies | None = None,
    path: str = "",
  ) -> Light:
    """A point light at `position`: `intensity` [cd] x `color`, or `power` [lm] (None for both:
    display-relative, a white surface facing it at its distance to the scene centre shows
    `color`); `radius` > 0: a sphere emitter of that radius [m] (soft shadows), the same
    intensity; `ies`: a photometric profile (`Ies`)."""
    return _bulb(cls, position, None, intensity, power, radius, color, temperature, illuminant, shadows, ies, path)

  @classmethod
  def spot(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    position: Vec3,
    direction: Vec3,
    *,
    cone_angle: float = 45.0,
    cone_softness: float = 0.0,
    intensity: float | None = None,
    power: float | None = None,
    spot_mode: SpotMode = "reflector",
    radius: float = 0.0,
    color: Rgb = WHITE,
    temperature: float | None = None,
    illuminant: Illuminant = "blackbody",
    shadows: bool = True,
    ies: Ies | None = None,
    path: str = "",
  ) -> Light:
    """A spot light at `position` emitting along `direction` in a cone of half angle
    `cone_angle` [deg] (UsdLux ShapingAPI, `cone_softness` in [0, 1]); `intensity` [cd] /
    `power` [lm] (`spot_mode`: "reflector" keeps the flux in the cone, "absorber" the flux of
    the light without its cone) as `point`."""
    cone = _Cone(angle=cone_angle, softness=cone_softness, mode=spot_mode)
    return _bulb(cls, position, (direction, cone), intensity, power, radius, color, temperature, illuminant, shadows, ies, path)

  @classmethod
  def sphere(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    position: Vec3,
    radius: float,
    *,
    intensity: float = 1.0,
    color: Rgb = WHITE,
    shadows: bool = True,
    visible: bool = False,
  ) -> Light:
    """A UsdLux SphereLight of `radius` [m] at `position`, radiance `intensity` [nits] x `color`;
    `visible`: camera rays see it (`visible_in_primary_ray`)."""
    frame = np.c_[np.eye(3), np.asarray(position, np.float64)]
    return cls(
      kind=LightKind.SPHERE,
      intensity=intensity,
      color=color,
      radius=radius,
      shadow_enable=shadows,
      visible_in_primary_ray=visible,
      xform=frame,
    )

  @classmethod
  def rect(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    position: Vec3,
    direction: Vec3,
    width: float,
    height: float,
    *,
    intensity: float = 1.0,
    color: Rgb = WHITE,
    shadows: bool = True,
    visible: bool = False,
  ) -> Light:
    """A UsdLux RectLight of `width` x `height` [m] centred at `position`, emitting along
    `direction` (one-sided), radiance `intensity` [nits] x `color`; `visible`: camera rays see it
    (`visible_in_primary_ray`)."""
    frame = light_frame(direction, position)
    return cls(
      kind=LightKind.RECT,
      intensity=intensity,
      color=color,
      width=width,
      height=height,
      shadow_enable=shadows,
      visible_in_primary_ray=visible,
      xform=frame,
    )

  @classmethod
  def disk(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    position: Vec3,
    direction: Vec3,
    radius: float,
    *,
    intensity: float = 1.0,
    color: Rgb = WHITE,
    shadows: bool = True,
    visible: bool = False,
  ) -> Light:
    """A UsdLux DiskLight of `radius` [m] centred at `position`, emitting along `direction`,
    radiance `intensity` [nits] x `color`; `visible`: camera rays see it
    (`visible_in_primary_ray`)."""
    frame = light_frame(direction, position)
    return cls(
      kind=LightKind.DISK,
      intensity=intensity,
      color=color,
      radius=radius,
      shadow_enable=shadows,
      visible_in_primary_ray=visible,
      xform=frame,
    )

  @classmethod
  def tube(  # noqa: PLR0913  # the light's keyword-only inputs
    cls,
    position: Vec3,
    axis: Vec3,
    length: float,
    radius: float,
    *,
    intensity: float = 1.0,
    color: Rgb = WHITE,
    shadows: bool = True,
    visible: bool = False,
  ) -> Light:
    """A UsdLux CylinderLight of `length` x `radius` [m] centred at `position` along `axis`,
    radiance `intensity` [nits] x `color`; `visible`: camera rays see it
    (`visible_in_primary_ray`)."""
    x = np.asarray(axis, np.float64)
    frame = light_frame(x, position)  # -Z along the axis; rolled so that +X is the axis
    frame = np.c_[-frame[:, 2], frame[:, 1], frame[:, 0], frame[:, 3]]
    return cls(
      kind=LightKind.CYLINDER,
      intensity=intensity,
      color=color,
      length=length,
      radius=radius,
      shadow_enable=shadows,
      visible_in_primary_ray=visible,
      xform=frame,
    )


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Ies:
  """An IES LM-63 photometric profile for `Light.point` / `Light.spot` (UsdLux ShapingAPI
  `ies:*`).

  Attributes:
    file: the profile file (`scene.model.ies.read`: a TILT file inlined).
    angle_scale: `ies:angleScale` (0: the profile's own angles).
    normalize: `ies:normalize`: the profile over its mean intensity (the light keeps the flux of
      the unshaped light).
    mode: "mask" (the profile relative to its peak x the light's intensity) or "measured" (the
      file's candela values x the intensity: give intensity 1).
    direction: world direction of the profile's vertical angle 0 (its nadir, the light's local
      -Z) on a `Light.point`; a `Light.spot`'s is its own direction.
    axis: world direction of the profile's horizontal angle 0 (the light's local +X; None: any
      axis perpendicular to the emission axis, enough for an axially symmetric profile).
  """

  file: str | os.PathLike[str]
  angle_scale: float = 0.0
  normalize: bool = False
  mode: IesMode = "mask"
  direction: Vec3 = (0.0, 0.0, -1.0)
  axis: Vec3 | None = None


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Cone:
  """A spot's cone (`Light.spot`)."""

  angle: float
  softness: float
  mode: SpotMode


def _tinted(color: Rgb, temperature: float | None, illuminant: Illuminant) -> Rgb:
  """`color` times the colour of luminance 1 of `temperature` [K] (None: `color`)."""
  if temperature is None:
    return color
  from raywarp.scene.lighting.photometry import cct_color  # noqa: PLC0415  # photometry imports this module

  tint = cct_color(temperature, illuminant)
  return (color[0] * tint[0], color[1] * tint[1], color[2] * tint[2])


def _bulb(  # noqa: PLR0913  # a point / spot light's inputs
  cls: type[Light],
  position: Vec3,
  spot: tuple[Vec3, _Cone] | None,
  intensity: float | None,
  power: float | None,
  radius: float,
  color: Rgb,
  temperature: float | None,
  illuminant: Illuminant,
  shadows: bool,  # noqa: FBT001  # positional in this private helper
  ies: Ies | None,
  path: str,
) -> Light:
  """A point (`spot` None) or spot light: a SphereLight of `radius` (a zero-radius sphere: a point
  light of projected area 1) whose radiance x projected area is `intensity` [cd]; a spot gets a
  ShapingAPI cone, a profile its `ies` shaping. A "measured" profile (not normalized) is already
  the radiance x projected area (the upload's peak / projected area): its `intensity` stays the
  multiplier of the file's candela."""
  if intensity is not None and power is not None:
    raise ValueError("Light.point / spot: give intensity [cd] or power [lm], not both")
  measured = ies is not None and ies.mode == "measured" and not ies.normalize
  area = math.pi * radius * radius if radius > 0.0 and not measured else 1.0
  text = "" if ies is None else ies_lib.read(os.fspath(ies.file))
  shaping = None
  if spot is not None or text:
    cone = spot[1] if spot is not None else None
    shaping = Shaping(
      cone_angle=cone.angle if cone is not None else MAX_ANGLE_DEG,
      cone_softness=cone.softness if cone is not None else 0.0,
      ies=text,
      ies_angle_scale=0.0 if ies is None else ies.angle_scale,
      ies_normalize=False if ies is None else ies.normalize,
      ies_mode="mask" if ies is None else ies.mode,
    )
  direction = spot[0] if spot is not None else ies.direction if ies is not None else (0.0, 0.0, -1.0)
  frame = _ies_frame(direction, position, None if ies is None else ies.axis) if shaping is not None else None
  light = cls(
    kind=LightKind.SPHERE,
    intensity=(1.0 if intensity is None else intensity) / area,
    color=_tinted(color, temperature, illuminant),
    radius=radius,
    treat_as_point=radius == 0.0,
    shaping=shaping,
    shadow_enable=shadows,
    xform=frame if frame is not None else np.c_[np.eye(3), np.asarray(position, np.float64)],
    path=path,
    ext=LightExt(relative=intensity is None and power is None),
  )
  if power is None:
    return light
  from raywarp.scene.lighting.photometry import with_power  # noqa: PLC0415  # photometry imports this module

  return with_power(light, power, spot_mode=spot[1].mode if spot is not None else "reflector")


def _ies_frame(direction: Vec3, position: Vec3, axis: Vec3 | None) -> np.ndarray[[3, 4], np.dtype[np.float64]]:
  """The (3, 4) frame of a spot / profiled light: -Z along `direction`, +X along `axis` made
  perpendicular to it when given.

  Raises:
    ValueError: for an `axis` parallel to `direction`.
  """
  frame = light_frame(direction, position)
  if axis is None:
    return frame
  z = frame[:, 2]
  a = np.asarray(axis, np.float64)
  x = a - z * float(a @ z)
  norm = math.sqrt(float(x @ x))
  if not norm > 1e-9:
    raise ValueError(f"Ies.axis {axis} is parallel to the light's direction {direction}")
  x /= norm
  return np.c_[x, np.cross(z, x), z, frame[:, 3]]


# No slots: the table is built from each record's `vars` (`table.record_columns`).
@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentLight:
  """An environment light (UsdLux DomeLight, skyboxes, user HDRs): `texture` (its color space is
  the texture's) x the radiance scale derived from the UsdLux inputs.

  Attributes:
    texture: its texture (NO_TEXTURE: a constant environment).
    intensity, exposure, color, enable_color_temperature, color_temperature: as a `Light`'s (the
      radiance scale).
    rotation: (3, 3) map -> world rotation.
    format: EnvFormat.
    path: source path.
    illuminance: > 0: the environment is scaled at upload so that an upward-facing surface
      receives this illuminance [lux] from it (Omniverse `PhotometricDomeLightAPI`; the colour's
      chromaticity kept); 0: the radiance scale as is.
  """

  texture: TextureId = NO_TEXTURE
  intensity: float = 1.0
  exposure: float = 0.0
  color: tuple[float, float, float] = (1.0, 1.0, 1.0)
  enable_color_temperature: bool = False
  color_temperature: float = 6500.0
  rotation: np.ndarray[[3, 3], np.dtype[np.float64]] = dataclasses.field(default_factory=lambda: np.eye(3))
  format: EnvFormat = EnvFormat.LATLONG
  path: str = ""
  illuminance: float = 0.0


@final
@dataclasses.dataclass
class Lights(Table):
  """UsdLux lights (`Light` records): the UsdLux inputs, physical units.

  `xform` is orthonormal (position + axes; sizes are in world metres). `angle` and `cone_angle`
  are degrees (UsdLux); `shaping` says whether ShapingAPI is applied (its columns hold the
  `Shaping` defaults otherwise). `ext_has_range` / `ext_range` is `LightExt.range` (None / a
  value); `ext_falloff` indexes `FALLOFFS`. `body` indexes `Instances.bodies`. `instance` is the
  emissive instance a light was made from (`scene.lighting.emissive`; -1: an authored light): the
  light sits just in front of that surface, so the surface does not shadow it.
  """

  SCHEMA: ClassVar = {
    "kind": (U8, ()),
    **dict(_PHOTOMETRIC),
    "normalize": (U8, ()),
    "meters_per_unit": (F64, ()),
    "radius": (F64, ()),
    "width": (F64, ()),
    "height": (F64, ()),
    "length": (F64, ()),
    "angle": (F64, ()),
    "treat_as_point": (U8, ()),
    "treat_as_line": (U8, ()),
    "diffuse": (F64, ()),
    "specular": (F64, ()),
    "shaping": (U8, ()),
    "cone_angle": (F64, ()),
    "cone_softness": (F64, ()),
    "focus": (F64, ()),
    "focus_tint": (F64, (3,)),
    "ies_angle_scale": (F64, ()),
    "ies_normalize": (U8, ()),
    "ies_mode": (U8, ()),
    "shadow_enable": (U8, ()),
    "visible_in_primary_ray": (U8, ()),
    "texture": (I32, ()),
    "xform": (F32, (3, 4)),
    "body": (I32, ()),
    "instance": (I32, ()),
    "ext_has_range": (U8, ()),
    "ext_range": (F64, ()),
    "ext_falloff": (U8, ()),
  }
  STRINGS: ClassVar = ("paths", "ies")
  FALLOFFS: ClassVar[tuple[Falloff, ...]] = _FALLOFFS
  kind: Column[np.uint8]
  intensity: Column[np.float64]
  exposure: Column[np.float64]
  color: Rows[np.float64, 3]
  enable_color_temperature: Column[np.uint8]
  color_temperature: Column[np.float64]
  normalize: Column[np.uint8]
  meters_per_unit: Column[np.float64]
  radius: Column[np.float64]
  width: Column[np.float64]
  height: Column[np.float64]
  length: Column[np.float64]
  angle: Column[np.float64]
  treat_as_point: Column[np.uint8]
  treat_as_line: Column[np.uint8]
  diffuse: Column[np.float64]
  specular: Column[np.float64]
  shaping: Column[np.uint8]
  cone_angle: Column[np.float64]
  cone_softness: Column[np.float64]
  focus: Column[np.float64]
  focus_tint: Rows[np.float64, 3]
  ies_angle_scale: Column[np.float64]
  ies_normalize: Column[np.uint8]
  ies_mode: Column[np.uint8]
  shadow_enable: Column[np.uint8]
  visible_in_primary_ray: Column[np.uint8]
  texture: Column[np.int32]
  xform: Matrices[np.float32, 3, 4]
  body: Column[np.int32]
  instance: Column[np.int32]
  ext_has_range: Column[np.uint8]
  ext_range: Column[np.float64]
  ext_falloff: Column[np.uint8]
  paths_blob: Column[np.uint8]
  paths_offsets: Column[np.int64]
  ies_blob: Column[np.uint8]
  ies_offsets: Column[np.int64]

  @property
  def paths(self) -> StringTable:
    """Source paths of the lights."""
    return self.strings("paths")

  @property
  def ies(self) -> StringTable:
    """IES profile text of each light ("" none)."""
    return self.strings("ies")

  @staticmethod
  def from_records(lights: Sequence[Light], *, bodies: Sequence[int]) -> Lights:
    """The table of UsdLux records; `bodies[i]`: the `Instances.bodies` index of `lights[i].body`
    (-1: static), resolved by the caller (the builder, or a pass over a built scene).

    Raises:
      ValueError: when `bodies` does not hold one id per light.
    """
    if len(bodies) != len(lights):
      raise ValueError(f"Lights.from_records: {len(bodies)} body ids for {len(lights)} lights")
    rows = list(starmap(_light_row, zip(lights, bodies, strict=True)))
    columns = {name: np.asarray([row[name] for row in rows], dtype) for name, (dtype, _) in Lights.SCHEMA.items()}
    ies = [light.shaping.ies if light.shaping is not None else "" for light in lights]
    return Lights.from_columns(columns, {"paths": [light.path for light in lights], "ies": ies})


def _light_row(light: Light, body: int) -> dict[str, object]:
  """A record's `Lights` columns: its fields, shaping and ext flattened (None as a flag), the
  body as its `Instances.bodies` index."""
  shaping = light.shaping if light.shaping is not None else Shaping()
  return vars(light) | {
    "body": body,
    "shaping": light.shaping is not None,
    "cone_angle": shaping.cone_angle,
    "cone_softness": shaping.cone_softness,
    "focus": shaping.focus,
    "focus_tint": shaping.focus_tint,
    "ies_angle_scale": shaping.ies_angle_scale,
    "ies_normalize": shaping.ies_normalize,
    "ies_mode": IES_MODES.index(shaping.ies_mode),
    "ext_has_range": light.ext.range is not None,
    "ext_range": 0.0 if light.ext.range is None else light.ext.range,
    "ext_falloff": _FALLOFFS.index(light.ext.falloff),
  }


@final
@dataclasses.dataclass
class Environments(Table):
  """Environment lights (`EnvironmentLight` records: DomeLight, skyboxes, user HDRs): texture
  (-1 = constant) x the radiance scale of the UsdLux inputs (derived at upload)."""

  SCHEMA: ClassVar = {
    "texture": (I32, ()),
    **dict(_PHOTOMETRIC),
    "rotation": (F32, (3, 3)),
    "format": (U8, ()),
    "illuminance": (F64, ()),
  }
  STRINGS: ClassVar = ("paths",)
  texture: Column[np.int32]
  intensity: Column[np.float64]
  exposure: Column[np.float64]
  color: Rows[np.float64, 3]
  enable_color_temperature: Column[np.uint8]
  color_temperature: Column[np.float64]
  rotation: Matrices[np.float32, 3, 3]
  format: Column[np.uint8]
  illuminance: Column[np.float64]
  paths_blob: Column[np.uint8]
  paths_offsets: Column[np.int64]

  @property
  def paths(self) -> StringTable:
    """Source paths of the environments."""
    return self.strings("paths")

  @staticmethod
  def from_records(envs: Sequence[EnvironmentLight]) -> Environments:
    """The table of UsdLux environment records."""
    return Environments.from_columns(record_columns(envs, Environments.SCHEMA), {"paths": [env.path for env in envs]})


@final
@dataclasses.dataclass
class Cameras(Table):
  """Cameras (`types.Camera` fields); `names` are the source names (USD prim paths)."""

  SCHEMA: ClassVar = {
    "xform": (F32, (3, 4)),
    "projection": (U8, ()),
    "focal_length": (F32, ()),
    "horizontal_aperture": (F32, ()),
    "vertical_aperture": (F32, ()),
    "horizontal_offset": (F32, ()),
    "vertical_offset": (F32, ()),
    "near": (F32, ()),
    "far": (F32, ()),
    "fstop": (F32, ()),
    "focus_distance": (F32, ()),
    "exposure": (F32, ()),
  }
  STRINGS: ClassVar = ("names",)
  xform: Matrices[np.float32, 3, 4]
  projection: Column[np.uint8]
  focal_length: Column[np.float32]
  horizontal_aperture: Column[np.float32]
  vertical_aperture: Column[np.float32]
  horizontal_offset: Column[np.float32]
  vertical_offset: Column[np.float32]
  near: Column[np.float32]
  far: Column[np.float32]
  fstop: Column[np.float32]
  focus_distance: Column[np.float32]
  exposure: Column[np.float32]
  names_blob: Column[np.uint8]
  names_offsets: Column[np.int64]

  @property
  def names(self) -> StringTable:
    """Source names of the cameras."""
    return self.strings("names")

  def get(self, i: int) -> Camera:
    """Row `i` as a `types.Camera` (the float columns as Python floats)."""
    floats = {name: float(operator.attrgetter(name)(self)[i]) for name in self.SCHEMA if name not in {"xform", "projection"}}
    return Camera(xform=np.asarray(self.xform[i]), projection=int(self.projection[i]), **floats)
