"""Host UsdLux light records (`raywarp.scene.model.lights.Lights`) -> derived light table -> core
light slots.

`derive_lights` turns the UsdLux inputs into the device quantities, then `light_arrays` builds
the slots.

Called by `upload.upload` (the environment's suns are shaded after these lights: `with_suns`
gives their capabilities). Every light gets both representations, so `lighting.area_lights`
switches between them without re-uploading:

* the punctual one ("point"): distant -> directional (lux), sphere -> point
  (cd = radiance x pi r^2, soft shadows from its radius), rect / disk -> a COSINE light along the
  emission direction (cd = radiance x area x cos theta, a Lambertian emitter's intensity;
  `light.radius` = the radius of the same area: soft shadows from that disc in the emitter's
  plane),
  cylinder -> a TUBE light (cd = radiance x 2 r l sin theta, theta from the axis, `light.dir` =
  the axis); ShapingAPI cones make spots of that cone, inner cone (1 - softness) x the cone angle
  (`light.cone_inner`, only when some cone's differs from the core's default fraction). Distant
  lights: `light.radius` = the disc's angular radius (half the UsdLux `angle`), the soft sun of
  `Features.sun_disc`.
* the area one ("ltc"): `light.shape` RECT / DISK / SPHERE / TUBE with `light.axis` (local X) and
  `light.size` (rect half extents, tube (half length, 0)); radii in `light.radius`. Shaped
  (spot) lights, `treatAsPoint` spheres and zero-size emitters stay punctual.

The falloff window `light.range` is the light's own (`LightExt.range`) or none (infinite):
UsdLux lights fall off with the inverse square law at any distance, whatever the scene's units or
exposure (an absolute illuminance cutoff took 12 % of a dim interior's light inside its window,
cut lights below 1 cd at 1 m and blackened kilometre-scale scenes). The light grid moves each
cell's weak distant lights into its far field (`light_grid`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Final, final

import numpy as np
import warp as wp

from raywarp._base.types import SPOT_INNER_FRACTION, LightShape, LightType
from raywarp.core.pipeline.gpu_scene.gpu_scene import EnvSuns, GpuHeadlight
from raywarp.core.pipeline.upload.reporting import report_mask
from raywarp.core.shade.lights.ies import IesLights
from raywarp.scene.lighting import ies_profile
from raywarp.scene.lighting.blackbody import scaled_colors
from raywarp.scene.model.lights import IES_MODES, MAX_ANGLE_DEG, Environments, Lights
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.table import F32, F64, I32, U8, U32, StringTable, Table
from raywarp.scene.model.types import Column, LightFlags, LightKind, Matrices, Rows

# `light.range` of a light without a window: the inverse square law at every distance.
_UNBOUNDED_RANGE: Final = math.inf
# A cosine lobe with this cutoff [deg] or more lights the whole hemisphere (no cone); a ShapingAPI
# cone this wide or wider is none on a rect / disk (MAX_ANGLE_DEG on a sphere / cylinder).
_HEMISPHERE_DEG: Final = 90.0
# Inner cone fractions this close to the core's are its own.
_INNER_TOL: Final = 1e-6
# Light types with an angular profile (`Capabilities.has_spot`).
_SHAPED: Final = (int(LightType.SPOT), int(LightType.COSINE), int(LightType.TUBE))
# A cone half angle [rad] this close to pi is no cone.
_NO_CONE_EPS: Final = 1e-6
_DEG_TO_RAD: Final = math.pi / 180.0  # `math.radians`' factor (the same float64 product)
# The smallest peak the light color is divided by (a black light keeps a black color).
_MIN_PEAK: Final = 1e-30


@final
@dataclasses.dataclass
class DerivedLights(Table):
  """The derived light table `light_arrays` reads, from the UsdLux records.

  `radiance` is the emitted surface radiance [nits] (after color temperature, exposure and
  `normalize`), or the illuminance [lux] of a distant light; `intensity` is the same light as a
  point, luminous intensity [cd] = radiance x projected area (distant: lux). Sizes a kind does
  not have are 0. `angle` (a distant light's angular diameter) and `cone_angle` (the ShapingAPI
  half angle, pi = no cone) are radians. `has_range` / `range`: an explicit falloff window [m]
  (`LightExt.range`; 0: no falloff, `LightExt.falloff` "none"); without one, no window.
  `instance`: the emissive instance of a panel light (-1: none). `ies`: the light's
  IES profile (strings, "" none), whose peak and normalisation its `intensity` already holds
  (`_ies_scales`), `ies_angle_scale` its UsdLux angle scale.
  """

  SCHEMA: ClassVar = {
    "kind": (U8, ()),
    "flags": (U32, ()),
    "xform": (F32, (3, 4)),
    "radiance": (F32, (3,)),
    "intensity": (F32, (3,)),
    "radius": (F32, ()),
    "width": (F32, ()),
    "height": (F32, ()),
    "length": (F32, ()),
    "angle": (F32, ()),
    "cone_angle": (F32, ()),
    "cone_softness": (F32, ()),
    "focus": (F32, ()),
    "diffuse": (F32, ()),
    "specular": (F32, ()),
    "texture": (I32, ()),
    "instance": (I32, ()),
    "has_range": (U8, ()),
    "range": (F64, ()),
    "ies_angle_scale": (F32, ()),
  }
  STRINGS: ClassVar = ("paths", "ies")
  kind: Column[np.uint8]
  flags: Column[np.uint32]
  xform: Matrices[np.float32, 3, 4]
  radiance: Rows[np.float32, 3]
  intensity: Rows[np.float32, 3]
  radius: Column[np.float32]
  width: Column[np.float32]
  height: Column[np.float32]
  length: Column[np.float32]
  angle: Column[np.float32]
  cone_angle: Column[np.float32]
  cone_softness: Column[np.float32]
  focus: Column[np.float32]
  diffuse: Column[np.float32]
  specular: Column[np.float32]
  texture: Column[np.int32]
  instance: Column[np.int32]
  has_range: Column[np.uint8]
  range: Column[np.float64]
  ies_angle_scale: Column[np.float32]
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
    """IES profile text of each light ("" none; distant lights: none)."""
    return self.strings("ies")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class _Kinds:
  """(n,) masks of the light kinds of a `Lights` table."""

  distant: Column[np.bool_]
  sphere: Column[np.bool_]
  rect: Column[np.bool_]
  disk: Column[np.bool_]
  cylinder: Column[np.bool_]


def _kinds(lights: Lights) -> _Kinds:
  """The kind masks of `lights`."""
  k = np.asarray(lights.kind, np.int64)
  return _Kinds(
    distant=k == int(LightKind.DISTANT),
    sphere=k == int(LightKind.SPHERE),
    rect=k == int(LightKind.RECT),
    disk=k == int(LightKind.DISK),
    cylinder=k == int(LightKind.CYLINDER),
  )


def _profiled(lights: Lights | DerivedLights) -> Column[np.bool_]:
  """(n,) lights with an IES profile."""
  return np.array([bool(t) for t in lights.ies.tolist()], bool)


def _coned(cone_angle: Column[np.float32] | Column[np.float64]) -> Column[np.bool_]:
  """(n,) lights whose cone half angle [rad] is a cone (narrower than pi)."""
  return np.asarray(cone_angle, np.float64) < math.pi - _NO_CONE_EPS


def _has_flag(lights: DerivedLights, flag: LightFlags) -> Column[np.bool_]:
  """(n,) lights whose `flags` hold `flag`."""
  return (np.asarray(lights.flags, np.int64) & int(flag)) != 0


def _sizes(lights: Lights, k: _Kinds) -> dict[str, Column[np.float64]]:
  """Each light's sizes [m] its kind has (sphere / disk / cylinder radius, rect width / height,
  cylinder length), 0 for the others whatever the record holds (UsdLux defaults), and a distant
  light's angle [rad]."""
  zero = np.zeros(len(lights))
  return {
    "radius": np.where(k.sphere | k.disk | k.cylinder, lights.radius, zero),
    "width": np.where(k.rect, lights.width, zero),
    "height": np.where(k.rect, lights.height, zero),
    "length": np.where(k.cylinder, lights.length, zero),
    "angle": np.where(k.distant, np.asarray(lights.angle, np.float64) * _DEG_TO_RAD, zero),
  }


def _areas(k: _Kinds, size: dict[str, Column[np.float64]]) -> tuple[Column[np.float64], Column[np.float64]]:
  """(projected area, surface area) [m^2] per light: the area a point approximation multiplies
  the radiance by (pi r^2 for spheres / disks, w h, 2 r l; 1 for points, distant and display
  lights, l for a zero-radius cylinder), and the one UsdLux `normalize` divides by (4 pi r^2,
  pi r^2, w h, 2 pi r l; 0 for distant and display lights)."""
  r, w, h, length = size["radius"], size["width"], size["height"], size["length"]
  pi_r2, wh = math.pi * r * r, w * h
  projected = np.where((k.sphere | k.disk) & (r > 0.0), pi_r2, 1.0)
  projected = np.where(k.rect, wh, projected)
  projected = np.where(k.cylinder, np.where(r > 0.0, 2.0 * r * length, length), projected)
  surface = np.where(k.sphere, 4.0 * math.pi * r * r, np.where(k.disk, pi_r2, 0.0))
  surface = np.where(k.rect, wh, np.where(k.cylinder, 2.0 * math.pi * r * length, surface))
  return projected, surface


def _cones(lights: Lights, k: _Kinds) -> tuple[Column[np.float64], Column[np.uint32]]:
  """(cone half angle [rad], LightFlags) per light: the ShapingAPI angle when
  narrower than the emitter's spread (180 deg for spheres / cylinders, 90 otherwise), else pi;
  CAST_SHADOW, TREAT_AS_POINT (spheres), TREAT_AS_LINE (cylinders), VISIBLE (seen by camera
  rays), SHAPED for a cone, a focus or an IES profile (not on distant lights)."""
  shaped = np.asarray(lights.shaping, bool)
  angle = np.asarray(lights.cone_angle, np.float64)
  limit = np.where(k.sphere | k.cylinder, MAX_ANGLE_DEG, _HEMISPHERE_DEG)
  cone = np.where(shaped & (angle < limit), angle * _DEG_TO_RAD, math.pi)
  flags = np.where(np.asarray(lights.shadow_enable, bool), int(LightFlags.CAST_SHADOW), 0)
  flags |= np.where(k.sphere & np.asarray(lights.treat_as_point, bool), int(LightFlags.TREAT_AS_POINT), 0)
  flags |= np.where(k.cylinder & np.asarray(lights.treat_as_line, bool), int(LightFlags.TREAT_AS_LINE), 0)
  flags |= np.where(np.asarray(lights.visible_in_primary_ray, bool), int(LightFlags.VISIBLE), 0)
  spot = _coned(cone) | (shaped & (np.asarray(lights.focus) > 0.0)) | (_profiled(lights) & ~k.distant)
  return cone, (flags | np.where(spot, int(LightFlags.SHAPED), 0)).astype(np.uint32)


def derive_lights(lights: Lights) -> DerivedLights:
  """The derived light table of the UsdLux records (float64 on the stored columns, elementwise,
  rounded once to float32): color x blackbody, intensity x 2^exposure, `normalize`, radiance /
  point intensity, angles in radians, cones, flags, `ext.range` / `ext.falloff`, then the IES
  profiles' scale (`_ies_scales`)."""
  k = _kinds(lights)
  size = _sizes(lights, k)
  projected, surface = _areas(k, size)
  rad = _scaled(lights)
  measured = _measured(lights, k)
  apply = np.asarray(lights.normalize, bool) & (surface > 0.0) & ~measured
  # The world area [m^2] in stage units: mpu ** 2 as Python's pow (bit-identical to
  # formats/usd/stage/lux.py's).
  mpu2 = np.array([m**2 for m in np.asarray(lights.meters_per_unit, np.float64).tolist()])
  rad = np.where(apply[:, None], rad * np.divide(mpu2, surface, out=np.ones(len(lights)), where=apply)[:, None], rad)
  cone, flags = _cones(lights, k)
  no_falloff = np.asarray(lights.ext_falloff) == Lights.FALLOFFS.index("none")
  shaped = np.asarray(lights.shaping, bool)
  ies_scale = _ies_scales(lights, k, projected)
  if ies_scale is not None:
    rad = rad * ies_scale[:, None]
  derived = DerivedLights.from_columns(
    {
      "kind": lights.kind, "flags": flags, "xform": lights.xform, "radiance": rad, "intensity": rad * projected[:, None],
      **size, "cone_angle": cone, "cone_softness": np.where(shaped, lights.cone_softness, 0.0),
      "focus": np.where(shaped, lights.focus, 0.0), "diffuse": lights.diffuse, "specular": lights.specular,
      "texture": lights.texture, "instance": lights.instance,
      "has_range": np.asarray(lights.ext_has_range, bool) | no_falloff,
      "range": np.where(no_falloff, 0.0, lights.ext_range),
      "ies_angle_scale": np.where(shaped, lights.ies_angle_scale, 0.0),
    },
    {"ies": [t if not d else "" for t, d in zip(lights.ies.tolist(), k.distant.tolist(), strict=True)]},
  )  # fmt: skip
  return dataclasses.replace(derived, paths_blob=lights.paths_blob, paths_offsets=lights.paths_offsets)


def _measured(lights: Lights, k: _Kinds) -> Column[np.bool_]:
  """(n,) lights whose IES profile gives their intensity in candela ("measured" without
  `ies:normalize`; not distant lights)."""
  mode = np.asarray(lights.ies_mode) == IES_MODES.index("measured")
  return _profiled(lights) & ~k.distant & mode & ~np.asarray(lights.ies_normalize, bool)


def _ies_scales(lights: Lights, k: _Kinds, projected: Column[np.float64]) -> Column[np.float64] | None:
  """(n,) factor of each light's radiance for its IES profile (None: no profile), the device
  profile being relative to its peak (`shade/lights/ies.py`): `ies_normalize`, peak / mean intensity
  (the unshaped light's flux kept); "mask", 1 (the light's intensity is the peak's); "measured",
  peak [cd] / projected area (the radiance whose point intensity is the file's candela, the
  light's intensity and exposure multipliers). The TILT factor follows the light's orientation
  per frame (`lights.pack.pack`)."""
  texts = lights.ies.tolist()
  if not any(texts):
    return None
  scale = np.ones(len(texts))
  normalize = np.asarray(lights.ies_normalize, bool)
  measured = _measured(lights, k)
  for i, text in enumerate(texts):
    if not text or k.distant[i]:
      continue
    g = ies_profile.profile(text)
    if normalize[i]:
      scale[i] = g.peak / g.mean
    elif measured[i]:
      scale[i] = g.peak / projected[i]
  return scale


def _scaled(table: Lights | Environments) -> Rows[np.float64, 3]:
  """(n, 3) c x s of a table's photometric columns (`blackbody.scaled_colors`)."""
  return scaled_colors(
    table.color,
    intensity=table.intensity,
    exposure=table.exposure,
    kelvin=table.color_temperature,
    enabled=table.enable_color_temperature,
  )


def derive_environments(envs: Environments) -> Rows[np.float32, 3]:
  """(nenv, 3) device radiance scale of each environment: color x blackbody x intensity x
  2^exposure."""
  return _scaled(envs).astype(np.float32).reshape(len(envs), 3)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class LightInfo:
  """What a light table uses (`Capabilities` fields).

  Attributes:
    has_spot: a light has a cone or an angular profile (spot, cosine, tube).
    has_area_lights: a light has a `light.shape`.
    sun_disc: a shadow-casting directional light has an angular size.
    has_ies: a light has an IES profile.
    visible_lights: camera rays see a light's surface (`visible_lights`).
  """

  has_spot: bool
  has_area_lights: bool
  sun_disc: bool
  has_ies: bool
  visible_lights: bool


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class DisplayLights:
  """The CAMERA / AMBIENT rows of a host light table, summed (`LightKind`).

  Attributes:
    headlight: the camera headlight (off without a CAMERA row).
    ambient: constant ambient radiance (zero without an AMBIENT row).
  """

  headlight: GpuHeadlight
  ambient: tuple[float, float, float]


def split_display(lights: DerivedLights) -> tuple[DerivedLights, DisplayLights]:
  """(the emitters of a host light table, its display lights): CAMERA / AMBIENT rows are no
  emitters (no position, no light slot)."""
  kind = np.asarray(lights.kind, np.int64)
  camera, ambient = kind == int(LightKind.CAMERA), kind == int(LightKind.AMBIENT)
  display = camera | ambient
  radiance = np.asarray(lights.radiance, np.float64).reshape(-1, 3)
  head = GpuHeadlight(active=True, color=_rgb_sum(radiance[camera])) if np.any(camera) else GpuHeadlight(active=False)
  emitters = lights.take(np.flatnonzero(~display)) if np.any(display) else lights
  return emitters, DisplayLights(headlight=head, ambient=_rgb_sum(radiance[ambient]))


def _rgb_sum(rows: Rows[np.float64, 3]) -> tuple[float, float, float]:
  """The sum of (n, 3) rgb rows (zeros for none)."""
  r, g, b = (float(v) for v in rows.sum(axis=0))
  return r, g, b


def _light_types(lights: DerivedLights) -> Column[np.int32]:
  """(n,) core LightType of each host light (rect / disk: COSINE, cylinder: TUBE, a sphere or
  cylinder with a cone: SPOT; one shaped without a cone, an IES profile alone, keeps its own
  POINT / TUBE lobe)."""
  kind = np.asarray(lights.kind, np.int64)
  coned = _has_flag(lights, LightFlags.SHAPED) & _coned(lights.cone_angle)
  ltype = np.full(len(kind), int(LightType.POINT), np.int32)
  ltype[kind == int(LightKind.CYLINDER)] = int(LightType.TUBE)
  # A focus alone (not rendered) or a profile alone: no cone, no cut-off.
  ltype[coned] = int(LightType.SPOT)
  # Rects and disks (with a ShapingAPI cone too: cutoff < 90 degrees).
  ltype[(kind == int(LightKind.RECT)) | (kind == int(LightKind.DISK))] = int(LightType.COSINE)
  ltype[kind == int(LightKind.DISTANT)] = int(LightType.DIRECTIONAL)  # last: a shaped distant light stays directional
  return ltype


def _emitters(lights: DerivedLights) -> tuple[Column[np.int32], Column[np.float64], Rows[np.float64, 2]]:
  """(LightShape, emitter radius, LTC half sizes) of each host light.

  The area shapes of unshaped lights with a size (`treatAsPoint` spheres stay punctual); the
  point / spot radius of soft shadows is the emitter's or a disk of the rect's area; a distant
  light's radius is its disc's angular radius (UsdLux `angle` is its diameter).
  """
  kind = np.asarray(lights.kind, np.int64)
  radius = np.asarray(lights.radius, np.float64)
  width, height = np.asarray(lights.width, np.float64), np.asarray(lights.height, np.float64)
  length = np.asarray(lights.length, np.float64)
  # Shaped lights stay punctual, except an IES profile alone on a rect / disk / sphere: its area
  # form is modulated by the profile toward the point (`shade/lights/evaluators.py`).
  coned = _coned(lights.cone_angle)
  profile_only = _profiled(lights) & ~coned & (np.asarray(lights.focus) <= 0.0) & (kind != int(LightKind.CYLINDER))
  shape = _surface_shapes(lights, area=~_has_flag(lights, LightFlags.SHAPED) | profile_only)
  rect, tube = kind == int(LightKind.RECT), kind == int(LightKind.CYLINDER)
  bulb = np.where(rect, np.sqrt(np.maximum(width * height, 0.0) / math.pi), radius)
  bulb = np.where(kind == int(LightKind.DISTANT), 0.5 * np.maximum(np.asarray(lights.angle, np.float64), 0.0), bulb)
  bulb = np.where(tube & _has_flag(lights, LightFlags.TREAT_AS_LINE), 0.0, bulb)
  size = np.stack([np.where(tube, 0.5 * length, 0.5 * width), np.where(tube, 0.0, 0.5 * height)], axis=1)
  return shape, bulb, size


def _surface_shapes(lights: DerivedLights, *, area: Column[np.bool_]) -> Column[np.int32]:
  """(n,) LightShape of the emitting surface of each light where `area`: rects, disks, spheres and
  cylinders with a size (`treatAsPoint` spheres have none), NONE elsewhere."""
  kind = np.asarray(lights.kind, np.int64)
  radius = np.asarray(lights.radius, np.float64)
  width, height = np.asarray(lights.width, np.float64), np.asarray(lights.height, np.float64)
  point_like = _has_flag(lights, LightFlags.TREAT_AS_POINT)
  shape = np.full(len(kind), int(LightShape.NONE), np.int32)
  shape[(kind == int(LightKind.RECT)) & area & (width > 0) & (height > 0)] = int(LightShape.RECT)
  shape[(kind == int(LightKind.DISK)) & area & (radius > 0)] = int(LightShape.DISK)
  shape[(kind == int(LightKind.SPHERE)) & area & (radius > 0) & ~point_like] = int(LightShape.SPHERE)
  shape[(kind == int(LightKind.CYLINDER)) & area & (np.asarray(lights.length, np.float64) > 0)] = int(LightShape.TUBE)
  return shape


def visible_lights(lights: DerivedLights) -> Rows[np.int32, 2]:
  """(k, 2) (light index, LightShape) of the lights whose emitting surface camera rays see
  (`shade/lights/visible.py`): the rects, disks, spheres and cylinders with a size, shaped or
  not (a shaped cylinder's axis is its IES frame's: none), that are flagged VISIBLE
  (`Light.visible_in_primary_ray`, off by default as in Omniverse RTX), except the lights
  standing for an emissive instance, whose own geometry shows their emission."""
  kind = np.asarray(lights.kind, np.int64)
  shape = _surface_shapes(lights, area=(kind != int(LightKind.CYLINDER)) | ~_has_flag(lights, LightFlags.SHAPED))
  shape[(np.asarray(lights.instance, np.int64) >= 0) | ~_has_flag(lights, LightFlags.VISIBLE)] = int(LightShape.NONE)
  index = np.flatnonzero(shape != int(LightShape.NONE))
  return np.stack([index, shape[index]], axis=1).astype(np.int32).reshape(-1, 2)


type LightRows = np.ndarray[[int, int], np.dtype[np.generic]] | np.ndarray[[int, int, int], np.dtype[np.generic]]


def _row[T: np.generic](a: Column[T] | Rows[T, 2] | Rows[T, 3], dtype: type = np.float32) -> LightRows:
  """The (1, n, ...) table of per-light values `a` (n = len(a))."""
  return np.asarray(a, dtype).reshape(1, *np.shape(a))


def _falloff(lights: DerivedLights, *, distant: Column[np.bool_]) -> Column[np.float64]:
  """(n,) falloff window [m]: the light's own (`LightExt`), else none (_UNBOUNDED_RANGE); 1 for
  distant lights."""
  return np.where(distant, 1.0, np.where(np.asarray(lights.has_range, bool), lights.range, _UNBOUNDED_RANGE))


def light_arrays(lights: DerivedLights) -> tuple[dict[str, LightRows], LightInfo]:
  """(slot -> (1, n) numpy table, what the lights use) of a host light table.

  Pose slots ("light.pos", "light.dir", "light.axis") are (1, n) too: one row shared by every
  world (`Renderer.per_world["light.pos"]` gives them per-world rows).
  """
  n = len(lights.kind)
  xform = np.asarray(lights.xform, np.float64).reshape(n, 3, 4)
  shaped = _has_flag(lights, LightFlags.SHAPED)
  cast_shadow = _has_flag(lights, LightFlags.CAST_SHADOW)
  distant = np.asarray(lights.kind, np.int64) == int(LightKind.DISTANT)
  # Color and peak: lux of distant lights (radiance), candela of the others (intensity).
  ref = np.where(
    distant[:, None],
    np.asarray(lights.radiance, np.float64).reshape(n, 3),
    np.asarray(lights.intensity, np.float64).reshape(n, 3),
  )
  peak = ref.max(axis=1)
  ltype = _light_types(lights)
  cutoff, inner = _cone_lobes(lights, ltype, shaped=shaped)
  shape, bulb, size = _emitters(lights)

  tables = {
    "light.type": _row(ltype, np.int32),
    "light.cast_shadow": _row(cast_shadow, bool),
    "light.active": _row(np.ones(n, bool), bool),
    "light.color": _row(ref / np.maximum(peak, _MIN_PEAK)[:, None]),
    "light.cutoff": _row(cutoff),
    "light.strength": _row(peak),
    "light.range": _row(_falloff(lights, distant=distant)),
    "light.radius": _row(bulb),
    "light.pos": _row(xform[:, :, 3]),
    # Local -Z: the emission direction of rects / disks / spots, the direction of distant lights;
    # the axis (local X) of tubes.
    "light.dir": _row(np.where((ltype == int(LightType.TUBE))[:, None], xform[:, :, 0], -xform[:, :, 2])),
    # A shaped tube is punctual: its IES frame's nadir (local -Z; `shade/lights/ies.py`).
    "light.axis": _row(np.where(((ltype == int(LightType.TUBE)) & shaped)[:, None], -xform[:, :, 2], xform[:, :, 0])),
    "light.shape": _row(shape, np.int32),
    "light.size": _row(size),
  }
  coned = (ltype == int(LightType.SPOT)) | ((ltype == int(LightType.COSINE)) & (cutoff < _HEMISPHERE_DEG))
  if np.any(coned & (np.abs(inner - SPOT_INNER_FRACTION) > _INNER_TOL)):
    tables["light.cone_inner"] = _row(inner)
  return tables, LightInfo(
    has_spot=bool(np.any(np.isin(ltype, _SHAPED))),
    has_area_lights=bool(np.any(shape)),
    sun_disc=bool(np.any(distant & (bulb > 0.0) & cast_shadow)),
    has_ies=bool(np.any(_profiled(lights))),
    visible_lights=bool(len(visible_lights(lights))),
  )


def with_suns(info: LightInfo, suns: EnvSuns) -> LightInfo:
  """`info` with the environment suns shaded after its lights: shadowed directional lights with
  a disc (`EnvSuns.angle`)."""
  if not len(suns):
    return info
  return dataclasses.replace(info, sun_disc=info.sun_disc or bool(np.any(suns.angle > 0.0)))


def _cone_lobes(
  lights: DerivedLights, ltype: Column[np.int32], *, shaped: Column[np.bool_]
) -> tuple[Column[np.float64], Column[np.float64]]:
  """((n,) cone cutoff [deg], (n,) inner cone fraction) of each light: a shaped light's cone and
  softness, else a hemisphere and the core's default fraction."""
  cone = np.asarray(lights.cone_angle, np.float64)
  # A spot's cone reaches up to 180 degrees (a sphere); a cosine lobe's at most its hemisphere.
  limit = np.where(ltype == int(LightType.SPOT), math.pi, math.pi / 2)
  cutoff = np.where(shaped, np.degrees(np.minimum(cone, limit)), _HEMISPHERE_DEG)
  # UsdLux cone softness (glTF: 1 - inner / outer): the falloff starts at (1 - softness) x cone.
  softness = np.clip(np.asarray(lights.cone_softness, np.float64), 0.0, 1.0)
  return cutoff, np.where(shaped, 1.0 - softness, SPOT_INNER_FRACTION)


def ies_lights(lights: DerivedLights) -> IesLights:
  """The lights' IES profiles: each distinct profile once (`ies_profile.profile`, cached per
  process), one instance per distinct (profile, angle scale), each light's instance."""
  grid_of: dict[str, int] = {}
  instance_of: dict[tuple[int, float], int] = {}
  grids: list[ies_profile.IesGrid] = []
  instances: list[tuple[int, float]] = []
  profile = np.full(len(lights), -1, np.int32)
  scales = np.asarray(lights.ies_angle_scale, np.float32).tolist()
  for i, text in enumerate(lights.ies.tolist()):
    if not text:
      continue
    if text not in grid_of:
      grid_of[text] = len(grids)
      grids.append(ies_profile.profile(text))
    key = (grid_of[text], float(scales[i]))
    if key not in instance_of:
      instance_of[key] = len(instances)
      instances.append(key)
    profile[i] = instance_of[key]
  return IesLights(grids=tuple(grids), instances=tuple(instances), light_profile=profile)


# Light slot -> Warp dtype of its table.
DTYPES: Final = {
  "light.type": int,
  "light.cast_shadow": bool,
  "light.active": bool,
  "light.color": wp.vec3,
  "light.cutoff": float,
  "light.strength": float,
  "light.range": float,
  "light.radius": float,
  "light.pos": wp.vec3,
  "light.dir": wp.vec3,
  "light.axis": wp.vec3,
  "light.shape": int,
  "light.size": wp.vec2,
  "light.cone_inner": float,
}


def report_lights(lights: DerivedLights, report: LoadReport) -> None:
  """Counts the light attributes `light_arrays` does not render."""
  multipliers = (np.asarray(lights.diffuse) != 1.0) | (np.asarray(lights.specular) != 1.0)
  for name, mask in (
    ("gpu.light.shaping_focus", np.asarray(lights.focus) > 0.0),
    ("gpu.light.diffuse_specular_multipliers", multipliers),
    ("gpu.light.texture", np.asarray(lights.texture) >= 0),
  ):
    report_mask(report.drop, name, mask, example=lambda i: lights.paths[i])
