"""Light added to a scene from Python: extra lights, an ambient fill, environment maps and a
camera headlight, given the same way to every adapter (`Lighting`).

A source without lights (a Newton model), or one that is too dark to debug, gets them here before
upload: `rw.integrations.newton.scene(model, lighting=...)`, `rw.formats.usd.scene(path,
lighting=...)`, `rw.integrations.mujoco.scene(mjm, m, lighting=...)`, or `scene.with_lighting(...)`
on a host `Scene` that is already built (a `.wpbr` file, say). The display-relative quantities (a
`Light.sun` / `point` / `spot` made without an intensity, the ambient, the headlight, an envmap
without an intensity) are calibrated so that a white Lambertian surface shows the given color at
`ev100`: render with the same `exposure.ev100`.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Sequence
from typing import Final, Literal, final

import numpy as np

from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.images.decode import ImageSource, decode_linear, is_float, probe
from raywarp.scene.images.hdr import read_hdr
from raywarp.scene.images.latlong import latlong_directions
from raywarp.scene.lighting.checks import check_light
from raywarp.scene.model import checks
from raywarp.scene.model.lights import EnvironmentLight, Light
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.tables import Materials
from raywarp.scene.model.types import NO_TEXTURE, Column, LightKind, MaterialFlags, Wrap
from raywarp.settings.fields import DEFAULT_EV100, exposure_from_ev100

type Rgb = tuple[float, float, float]
type Vec3 = tuple[float, float, float]
# An equirectangular environment: an image path (.hdr / .exr linear, 8-bit images sRGB) or a
# (height, width, 3) linear radiance array. Row 0 is the zenith (+z); the centre column looks
# along +x.
type Envmap = str | os.PathLike[str] | np.ndarray[[int, int, int], np.dtype[np.float32]]
type Point = np.ndarray[[3], np.dtype[np.float64]]

# An envmap array: (height, width, channels) with 3 (RGB) or 4 (RGBA, alpha ignored) channels.
IMAGE_NDIM: Final = 3
ENVMAP_CHANNELS: Final = (3, 4)
# World up axes (`Lighting.studio(up=)`, `Lighting.fill_lights(up=)`) and their indices.
type UpAxis = Literal["x", "y", "z"]
# Final[tuple[UpAxis, ...]]: the literal axes, not three strings.
UP_AXES: Final[tuple[UpAxis, ...]] = ("x", "y", "z")
Y_AXIS: Final = 1
Z_AXIS: Final = 2


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Headlight:
  """A MuJoCo-style camera headlight, display-relative at `Lighting.ev100`.

  Attributes:
    ambient: linear RGB of its constant ambient term.
    diffuse: linear RGB of the directional light attached to the camera.
  """

  ambient: Rgb
  diffuse: Rgb


@final
@dataclasses.dataclass(frozen=True)
class Lighting:
  """Lights, ambient, environment maps and headlight added to a scene (see the module doc).

  Attributes:
    lights: extra lights (`Light.sun`, `Light.point`, ...; display-relative ones resolved at
      `ev100`).
    ambient: constant ambient radiance, display-relative (a white surface shows this color).
    envmap: equirectangular environment (see `Envmap`), or several (one environment each:
      `Renderer.per_world["env.id"]` picks one per world; the first is the default).
    envmap_intensity: their radiance scale (None: display-relative, a texel of 1 shows white).
    envmap_rotation: rotation about +z [degrees].
    headlight: camera headlight.
    ev100: exposure the display-relative quantities are calibrated for (render with the same
      `exposure.ev100`).
    replace: drop the scene's own lights, ambient, headlight and image-based lighting (False:
      add to them). A MuJoCo model's lights are switched off (intensity 0), not removed.
  """

  lights: Sequence[Light] = ()
  _: dataclasses.KW_ONLY
  ambient: Rgb | None = None
  envmap: Envmap | Sequence[Envmap] | None = None
  envmap_intensity: float | None = None
  envmap_rotation: float = 0.0
  headlight: Headlight | None = None
  ev100: float = DEFAULT_EV100
  replace: bool = False

  @classmethod
  def studio(cls, *, up: UpAxis = "z", fill: bool = True, replace: bool = False, ev100: float = DEFAULT_EV100) -> Lighting:
    """A neutral studio rig that makes any scene readable at `ev100` (the default exposure).

    A sky dome (a soft daylight gradient: bluish zenith, white horizon, dim warm ground), a warm
    shadowed key light 50 deg above the horizon and, with `fill`, a weak cool unshadowed fill
    from the opposite side. Every value is display-relative (calibrated for `ev100`), so it
    exposes well at the default `exposure.ev100` whatever the scene's own light units.

    Args:
      up: the scene's world up axis: "z" for every USD scene (the importer turns a Y-up stage
        Z-up; `scene.meta["up_axis"]` records the stage's authored axis, not the scene's),
        Newton `model.up_axis`.
      fill: add the fill light.
      replace: drop the scene's own lighting instead of adding to it.
      ev100: the exposure it is calibrated for.

    Raises:
      ValueError: for an unknown `up` axis.
    """
    axis = _up_axis(up)
    lights = [Light.sun(_from_z_up(STUDIO_KEY_DIR, axis), color=STUDIO_KEY_COLOR)]
    if fill:
      lights.append(Light.sun(_from_z_up(STUDIO_FILL_DIR, axis), color=STUDIO_FILL_COLOR, shadows=False))
    return cls(lights=lights, envmap=studio_dome(axis), ev100=ev100, replace=replace)

  @staticmethod
  def fill_lights(
    n: int = 3,
    *,
    brightness: float = 0.3,
    elevation: float = 45.0,
    azimuth: float = 0.0,
    color: Rgb = (1.0, 1.0, 1.0),
    cast_shadow: bool = False,
    up: UpAxis = "z",
  ) -> tuple[Light, ...]:
    """`n` directional fill lights for `Lighting(lights=...)`: evenly spaced in azimuth (the
    first at `azimuth` degrees from +x about the up axis), `elevation` degrees above the horizon,
    shining down onto the scene from every side. Display-relative (no intensity): a white surface
    facing one of them shows `brightness / n` x `color` at `Lighting.ev100`, so together they
    add about `brightness` x sin(elevation) to a white floor whatever the scene's light units.

    Unshadowed by default (`cast_shadow`): they cost no shadow rays and reach inside closed rooms
    and under furniture, as HyperLux's fill lights for scenes without lights. They are ordinary
    lights of the scene: they light the GI probes too.

    Raises:
      ValueError: for `n` < 1, a negative or non-finite `brightness`, an `elevation` outside
        (0, 90] degrees, or an unknown `up` axis.
    """
    if n < 1:
      raise ValueError(f"fill_lights: n must be >= 1, got {n}")
    checks.in_range("brightness", brightness, "fill_lights", low=0.0)
    if not 0.0 < elevation <= 90.0:
      raise ValueError(f"fill_lights: elevation must be in (0, 90] degrees, got {elevation}")
    axis = _up_axis(up)
    e = math.radians(elevation)
    rgb = tuple(float(c) * brightness / n for c in color)
    out = []
    for k in range(n):
      a = math.radians(azimuth) + 2.0 * math.pi * k / n
      # The light travels down towards the scene from azimuth a (Z up).
      d = (-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), -math.sin(e))
      out.append(Light.sun(_from_z_up(d, axis), color=(rgb[0], rgb[1], rgb[2]), shadows=cast_shadow, path=f"fill_light_{k}"))
    return tuple(out)

  def __post_init__(self) -> None:
    """Validates the values (at construction: not deep in an upload)."""
    for i, envmap in enumerate(self.envmaps):
      if not isinstance(envmap, (str, os.PathLike)):
        _check_envmap_shape(np.shape(envmap), f"Lighting.envmap[{i}]")
    where = "Lighting"
    if not isinstance(self.lights, Sequence) or isinstance(self.lights, str):
      raise TypeError(f"{where}.lights takes a sequence of rw.Light (e.g. [Light.sun(...)]), got {type(self.lights).__name__}")
    for i, light in enumerate(self.lights):
      if not isinstance(light, Light):
        raise TypeError(f"{where}.lights[{i}] must be a rw.Light (Light.sun, Light.point, ...), got {type(light).__name__}")
      unresolved = dataclasses.replace(light, ext=dataclasses.replace(light.ext, relative=False))
      check_light(unresolved, ntexture=0, ninstance=0, where=f"{where}.lights[{i}]")
    if self.ambient is not None:
      checks.color("ambient", self.ambient, where)
    if self.headlight is not None:
      checks.color("headlight.ambient", self.headlight.ambient, where)
      checks.color("headlight.diffuse", self.headlight.diffuse, where)
    if self.envmap_intensity is not None:
      checks.in_range("envmap_intensity (None: display-relative)", self.envmap_intensity, where, low=0.0)
    checks.finite("envmap_rotation [deg]", self.envmap_rotation, where)
    checks.finite("ev100", self.ev100, where)

  @property
  def exposure(self) -> float:
    """Photometric exposure of `ev100`."""
    return exposure_from_ev100(self.ev100)

  @property
  def envmaps(self) -> list[Envmap]:
    """`envmap` as a list (empty without one)."""
    if self.envmap is None:
      return []
    if isinstance(self.envmap, (str, os.PathLike, np.ndarray)):
      return [self.envmap]
    return list(self.envmap)

  @property
  def env_intensity(self) -> float:
    """The environments' radiance scale (display-relative 1 / exposure by default)."""
    return 1.0 / self.exposure if self.envmap_intensity is None else self.envmap_intensity

  def ambient_radiance(self) -> Rgb:
    """Constant ambient radiance of `ambient` plus the headlight's ambient term (linear)."""
    total = np.zeros(3, np.float64)
    for rgb in (self.ambient, self.headlight.ambient if self.headlight is not None else None):
      if rgb is not None:
        # float32 colors, as the light table holds them.
        total += np.asarray(rgb, np.float32).astype(np.float64)
    r, g, b = (float(v) for v in total / self.exposure)
    return r, g, b


# The studio rig (`Lighting.studio`), display-relative linear RGB: a white surface facing a
# light shows its color; the dome's texels are radiance relative to white. Together a white
# Lambertian surface facing up shows about 0.75 (key 0.53 x cos 40 deg + fill + dome): exposed for
# the default `exposure.ev100` with headroom for highlights.
# Key: 50 deg elevation, from the front-left (light travels along this direction, Z up).
STUDIO_KEY_DIR: Final = (-0.4132, 0.4924, -0.7660)
STUDIO_KEY_COLOR: Final = (0.55, 0.53, 0.49)  # warm white (about 4500 K)
# Fill: 45 deg elevation, from the opposite side.
STUDIO_FILL_DIR: Final = (0.6124, -0.3536, -0.7071)
STUDIO_FILL_COLOR: Final = (0.10, 0.11, 0.12)  # cool, about a fifth of the key
STUDIO_ZENITH: Final = (0.24, 0.27, 0.32)  # soft daylight blue (about 8000 K)
STUDIO_HORIZON: Final = (0.30, 0.30, 0.30)
STUDIO_GROUND: Final = (0.09, 0.085, 0.08)  # dim warm floor bounce


def _check_envmap_shape(shape: tuple[int, ...], where: str) -> None:
  """Checks the shape of an envmap array: (height, width, 3 or 4), at least 1 x 1.

  Raises:
    ValueError: for another shape.
  """
  if len(shape) != IMAGE_NDIM or shape[2] not in ENVMAP_CHANNELS or min(shape[:2]) < 1:
    raise ValueError(f"{where} must be a path or a (height, width, 3) radiance array, got shape {shape}")


def _up_axis(up: str) -> int:
  """The index of world up axis `up` ("x", "y" or "z").

  Raises:
    ValueError: for another value.
  """
  if up not in UP_AXES:
    raise ValueError(f"up must be one of {UP_AXES}, got {up!r}")
  return UP_AXES.index(up)


def _from_z_up(v: Vec3, axis: int) -> Vec3:
  """`v` given for a Z-up scene, in a scene whose up axis is `axis` (a rotation taking +z to it)."""
  x, y, z = v
  if axis == Z_AXIS:
    return v
  if axis == Y_AXIS:
    return (x, z, -y)  # +z -> +y, +y -> -z
  return (z, x, y)  # +z -> +x


def studio_dome(axis: int, height: int = 32) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
  """The studio sky dome: an (height, 2 height, 3) equirect gradient about up axis `axis`
  (zenith -> horizon -> ground, smoothstep blends), in the core's layout (row 0 is +z)."""
  directions = latlong_directions(height, 2 * height)
  h = directions[..., axis : axis + 1]  # sine of the elevation
  zenith, horizon, ground = (np.asarray(c, np.float64) for c in (STUDIO_ZENITH, STUDIO_HORIZON, STUDIO_GROUND))
  up = np.clip(h, 0.0, 1.0)
  down = np.clip(-h * 4.0, 0.0, 1.0)  # a short blend below the horizon
  sky = horizon + (zenith - horizon) * (up * up * (3.0 - 2.0 * up))
  return np.where(h >= 0.0, sky, horizon + (ground - horizon) * (down * down * (3.0 - 2.0 * down))).astype(np.float32)


def add_lights(builder: SceneBuilder, lights: Sequence[Light], *, center: Point, extent: float, exposure: float) -> None:
  """Adds lights to a builder, display-relative ones (`LightExt.relative`) resolved at `exposure`
  (`resolved`)."""
  for light in lights:
    builder.add_light(resolved(light, center=center, extent=extent, exposure=exposure))


def resolved(light: Light, *, center: Point, extent: float, exposure: float) -> Light:
  """`light` in physical units: a display-relative one gets the intensity that shows its color on
  a white Lambertian surface facing it at `exposure` (pi / exposure lux; a point or spot light at
  its distance to `center`, at least a tenth of `extent`); others are returned as they are."""
  if not light.ext.relative:
    return light
  value = math.pi / exposure
  if light.kind != LightKind.DISTANT:
    offset = np.asarray(light.xform[:, 3], np.float64) - center
    dist = max(math.sqrt(float(offset @ offset)), 0.1 * extent, 1e-3)
    value *= dist * dist
  ext = dataclasses.replace(light.ext, relative=False)
  return dataclasses.replace(light, intensity=light.intensity * value, ext=ext)


def envmap_radiance(envmap: Envmap) -> np.ndarray[[int, int, int], np.dtype[np.float32]]:
  """An envmap as a (height, width, 3 or 4) linear radiance array: Radiance .hdr files read
  exactly, other image files decoded (8-bit sRGB, float linear), arrays checked."""
  if isinstance(envmap, (str, os.PathLike)):
    path = os.fspath(envmap)
    if path.lower().endswith(".hdr"):
      return read_hdr(path)
    source = ImageSource(path=path, transfer="linear" if is_float(path) else "srgb")
    info = probe(source)
    if info.error:
      raise ValueError(f"envmap {path}: {info.error}")
    return decode_linear(source, info.width, info.height)[..., :3]
  image = np.asarray(envmap, np.float32)
  _check_envmap_shape(image.shape, "envmap")
  return image


def add_environment(builder: SceneBuilder, envmap: Envmap, *, intensity: float, rotation: float) -> None:
  """Adds an equirectangular environment: a path (8-bit images sRGB, float ones linear) or linear
  radiance, times `intensity`, rotated by `rotation` degrees about +z.

  Raises:
    ValueError: for a path to a missing or unsupported image (a lighting the caller asked for:
      never a constant environment instead).
  """
  wrap = (Wrap.REPEAT, Wrap.CLAMP)
  if isinstance(envmap, (str, os.PathLike)):
    space = ColorSpace.LIN_REC709 if is_float(os.fspath(envmap)) else ColorSpace.SRGB_REC709
    texture = builder.add_texture(envmap, colorspace=space, wrap=wrap)
    if texture < 0:
      raise ValueError(f"envmap {os.fspath(envmap)!r}: a missing or unsupported image file")
  else:
    image = np.asarray(envmap, np.float32)
    _check_envmap_shape(image.shape, "envmap")
    rgb = np.ascontiguousarray(image[..., :3])
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1), np.float32)], axis=2)
    texture = builder.add_texture(rgba, colorspace=ColorSpace.LIN_REC709, wrap=wrap)
  c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
  rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  builder.add_environment(EnvironmentLight(texture=texture, intensity=intensity, rotation=rot))


def bounds(xforms: np.ndarray[[int, 3, 4], np.dtype[np.float32]]) -> tuple[Point, float]:
  """(centre, extent) of instance positions (where display-relative lights are placed from)."""
  if not len(xforms):
    return np.zeros(3), 1.0
  pos = xforms[:, :, 3]  # float32, as stored
  lo, hi = pos.min(axis=0), pos.max(axis=0)
  span = hi - lo
  # The centre computed in float32, widened exactly to float64 (`add_lights` works in float64).
  return (0.5 * (lo + hi)).astype(np.float64), max(math.sqrt(float(span @ span)), 1.0)


def add_lighting(builder: SceneBuilder, lighting: Lighting, around: tuple[Point, float] | None = None) -> None:
  """Adds all of `lighting` to a builder: its lights (display-relative ones placed around
  `around`, (centre, extent); default the builder's instances), environments, then the headlight
  (a CAMERA light) and the ambient (one AMBIENT light)."""
  center, extent = around if around is not None else bounds(builder.instance_xforms())
  add_lights(builder, lighting.lights, center=center, extent=extent, exposure=lighting.exposure)
  for envmap in lighting.envmaps:
    add_environment(builder, envmap, intensity=lighting.env_intensity, rotation=lighting.envmap_rotation)
  if lighting.headlight is not None:
    # float32 colors, as the light table holds them.
    r, g, b = (float(c) for c in np.asarray(lighting.headlight.diffuse, np.float32))
    builder.add_light(Light(kind=LightKind.CAMERA, color=(r, g, b), shadow_enable=False, path="headlight"))
  if lighting.ambient is not None or lighting.headlight is not None:
    ambient = lighting.ambient_radiance()
    builder.add_light(Light(kind=LightKind.AMBIENT, color=ambient, shadow_enable=False, path="lighting#ambient"))


def with_lighting(scene: Scene, lighting: Lighting) -> Scene:
  """`scene` (built) with `lighting` added (`add_lighting`; display-relative lights placed from
  its instances). Added environments come after the scene's own (the scene's first stays the
  default); their images are appended (`Images.append`: the scene's texel blob is not copied)."""
  builder = SceneBuilder()
  add_lighting(builder, lighting, bounds(np.asarray(scene.instances.xform, np.float32)))
  extra = builder.build(mips=bool(scene.meta.get("mips", False)))
  textures, images = scene.textures, scene.images
  if extra.nimage:
    images = images.append(extra.images)
    shifted = extra.textures.arrays()
    shifted["image"] = shifted["image"] + scene.nimage
    textures = textures.concat(type(textures).from_arrays(shifted))
  envs = extra.environments.arrays()
  envs["texture"] = _shifted(envs["texture"], scene.ntexture)
  environments = type(scene.environments).from_arrays(envs)
  light_columns = extra.lights.arrays()
  light_columns["texture"] = _shifted(light_columns["texture"], scene.ntexture)
  lights = type(scene.lights).from_arrays(light_columns)
  materials = scene.materials
  if lighting.envmaps and lighting.envmap_intensity is not None:
    materials = physical_display_emission(materials, lighting.envmap_intensity)
  if lighting.replace:
    # The scene's own lights and environments are dropped (their textures stay, unused).
    return dataclasses.replace(
      scene, materials=materials, textures=textures, images=images, lights=lights, environments=environments
    )
  return dataclasses.replace(
    scene,
    materials=materials,
    textures=textures,
    images=images,
    lights=scene.lights.concat(lights),
    environments=scene.environments.concat(environments),
  )


def physical_display_emission(materials: Materials, texel: float) -> Materials:
  """`materials` with their display-relative emission (`MaterialFlags.DISPLAY_EMISSION`: glTF's
  emissive) made a radiance in the units of an environment whose texel of 1 is `texel` nits
  (`Lighting.envmap_intensity`): emission x `texel`, the flag cleared.

  The convention of glTF viewers and path tracers (three.js, Filament's IBL, warp-optix): an
  emission of 1 is as bright as an environment texel of 1. A display-relative environment
  (`envmap_intensity` None, 1 / exposure per texel) keeps it display-relative (the same ratio at
  render time); an absolute one sets the scene's unit, which the emission must then follow, or a
  dim HDR sky of ~0.5 nits sits beside emission of 1 / exposure (~33 000 nits at the default EV)."""
  flag = int(MaterialFlags.DISPLAY_EMISSION)
  flags = np.asarray(materials.ext_flags, np.uint32)
  display = np.bitwise_and(flags, flag) != 0
  if not display.any():
    return materials
  columns = materials.arrays()
  luminance = np.asarray(materials.emission_luminance, np.float64)
  columns["emission_luminance"] = np.where(display, luminance * texel, luminance).astype(np.float32)
  columns["ext_flags"] = np.bitwise_and(flags, 0xFFFFFFFF ^ flag).astype(np.uint32)
  return type(materials).from_arrays(columns)


def _shifted(texture: Column[np.int32], ntexture: int) -> Column[np.int32]:
  """Texture ids of an added table, after the scene's `ntexture` textures (NO_TEXTURE kept)."""
  return np.where(texture >= 0, texture + ntexture, NO_TEXTURE).astype(np.int32)
