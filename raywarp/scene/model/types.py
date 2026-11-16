"""Enumerations and value types of the format-neutral scene description.

Conventions shared by every table:

* Lengths are metres, the world is Z-up, transforms are float32 (3, 4) matrices `[A | t]` that map
  local points to world points as `A @ p + t` (column vectors; `A` holds rotation and scale).
* Colors are linear Rec.709. Texture images keep their stored encoding (sRGB or linear) and
  carry the color space as a texture property: nothing is converted at load.
* Light units are physical (UsdLux / Omniverse): surface radiance [nits] for area lights,
  illuminance [lux] for distant lights, a radiance scale for environments.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Final, NewType

import numpy as np

if TYPE_CHECKING:
  from shape_extensions import IntVar

# Row ids of the scene tables (what `SceneBuilder.add_*` return): distinct types, so passing a
# texture id where a material id is expected fails the type check. -1 is "none" where allowed.
MeshId = NewType("MeshId", int)
ImageId = NewType("ImageId", int)
TextureId = NewType("TextureId", int)
MaterialId = NewType("MaterialId", int)
InstanceId = NewType("InstanceId", int)
LightId = NewType("LightId", int)
EnvironmentId = NewType("EnvironmentId", int)
CameraId = NewType("CameraId", int)
NO_MESH: Final = MeshId(-1)
NO_IMAGE: Final = ImageId(-1)
NO_TEXTURE: Final = TextureId(-1)
NO_MATERIAL: Final = MaterialId(-1)
NO_INSTANCE: Final = InstanceId(-1)
NO_ENVIRONMENT: Final = EnvironmentId(-1)


class InstanceKind(enum.IntEnum):
  """Shape of an instance. Primitive sizes follow MuJoCo's `geom_size` convention."""

  MESH = 0  # size unused; `mesh` indexes the mesh table
  PLANE = 1  # size = (half width x, half length y, 0); 0 half sizes = infinite; normal +Z
  SPHERE = 2  # size = (radius, 0, 0)
  CAPSULE = 3  # size = (radius, half length of the cylinder part along Z, 0)
  CYLINDER = 4  # size = (radius, half height along Z, 0)
  BOX = 5  # size = half extents
  ELLIPSOID = 6  # size = semi-axes


class InstanceFlags(enum.IntFlag):
  """Per-instance visibility flags."""

  VISIBLE = 1
  CAST_SHADOW = 2
  DOUBLE_SIDED = 4
  PRIMARY_VISIBLE = 8  # seen by camera rays (off: shadows / reflections only)

  DEFAULT = VISIBLE | CAST_SHADOW | PRIMARY_VISIBLE


class AlphaMode(enum.IntEnum):
  """How a material's opacity is applied (glTF)."""

  OPAQUE = 0
  BLEND = 1
  MASK = 2  # opaque where opacity >= alpha_cutoff, else fully transparent


class Slot(enum.IntEnum):
  """Texture slots of a material, named after the OpenPBR input each drives (the order and values
  are frozen: `NSLOT` and the `.wpbr` column shapes). A bound slot's value is
  `field constant x (scale x texel + bias)`; a texture on any other OpenPBR input goes to
  `OpenPbrExtras.textures` (scene/model/material.py)."""

  BASE_COLOR = 0  # rgb multiplies `base_weight` x `base_color`
  BASE_METALNESS = 1  # one channel
  SPECULAR_ROUGHNESS = 2  # one channel
  OCCLUSION = 3  # one channel: raywarp extension (glTF AO), `RaywarpExt.occlusion_strength`
  # Tangent-space normal map: scale x texel + bias is the normal (OpenGL +Y up; strengths and
  # flips are per-axis scale / bias: scene/model/normal_map.py).
  GEOMETRY_NORMAL = 4
  EMISSION_COLOR = 5  # rgb (or one channel broadcast) multiplies `emission_color` x `emission_luminance`
  GEOMETRY_OPACITY = 6  # one channel
  TRANSMISSION_WEIGHT = 7  # one channel; carried, not rendered (dropped at upload)
  # One channel; carried, not rendered: OpenPBR `specular_weight` (the Unreal family's w = 2 s).
  SPECULAR_WEIGHT = 8


NSLOT: Final = len(Slot)

# Channel of a texture a slot reads. RGB reads three channels (color slots, normal map); a single
# channel is broadcast to every component of the slot.
CHANNEL_RGB: Final = -1
CHANNEL_R: Final = 0
CHANNEL_G: Final = 1
CHANNEL_B: Final = 2
CHANNEL_A: Final = 3
CHANNELS: Final = {"rgb": CHANNEL_RGB, "rgba": CHANNEL_RGB, "r": CHANNEL_R, "g": CHANNEL_G, "b": CHANNEL_B, "a": CHANNEL_A}
# TextureRef.uv_set of a world-space box projection (see TextureRef).
UV_SET_WORLD: Final = 2


class MaterialFlags(enum.IntFlag):
  """Per-material switches."""

  TWO_SIDED = 1
  UNLIT = 4
  UV_WORLD = 8  # analytic primitives: texture repeat per metre instead of per object (MuJoCo texuniform)
  # Unreal / OmniUe4Base specular (MDL custom_curve_layer): F0 = 0.08 x specular, F90 =
  # specular x (1 - roughness); the metallic lobe reflects the base color at every angle.
  SPECULAR_CURVE = 16
  # The mesh's vertex colors (Meshes.colors) multiply the base color and opacity (glTF
  # COLOR_0, USD displayColor read by the material or of an unbound gprim).
  VERTEX_COLOR = 32
  # A glTF / UsdPreviewSurface / OmniPBR / OmniSurface coat: the base keeps its F0 against air and
  # is neither roughened nor darkened under it (the canonical Material's raywarp variant of the
  # OpenPBR coat). Read by `core/pipeline/derive/materials.py`.
  COAT_GLTF = 64
  # A native dielectric specular level: `specular_weight` scales dielectric F0 only;
  # the metal lobe is unscaled and dielectric F90 stays 1 even at zero weight (MuJoCo).
  DIELECTRIC_SPECULAR_LEVEL = 128
  # The coat reflects about the base's mapped normal (UsdPreviewSurface's one `normal` input).
  # Without it a coat uses its own normal: `geometry_coat_normal` when bound, else the unmapped
  # (interpolated) normal, OpenPBR's / glTF's default (`clearcoatNormalTexture` absent).
  COAT_BASE_NORMAL = 256
  # The base color (factor and texture) tints the transmitted light (glTF
  # KHR_materials_transmission / KHR_materials_volume: a colored foil at the surface). Without it
  # the OpenPBR rule holds: `transmission_color` is the tint and a base-color texture reaches only
  # the opaque share (1 - transmission_weight). Read by `core/pipeline/derive/materials.py`.
  TRANSMISSION_TINT = 512
  # The emission is display-relative (1: the brightest LDR value, divided by the world's exposure
  # at render time) instead of a radiance in nits: glTF's `emissiveFactor` (x
  # KHR_materials_emissive_strength), which the glTF converter marks. The surface stays lit.
  DISPLAY_EMISSION = 1024


class Wrap(enum.IntEnum):
  """Texture address mode per axis."""

  REPEAT = 0
  CLAMP = 1
  MIRROR = 2
  BLACK = 3  # border: zero outside [0, 1]


class PixelType(enum.IntEnum):
  """Texel component type of an image."""

  U8 = 0
  F16 = 1
  F32 = 2


PIXEL_DTYPES: Final = {
  PixelType.U8: np.dtype(np.uint8),
  PixelType.F16: np.dtype(np.float16),
  PixelType.F32: np.dtype(np.float32),
}


def itemsize(dtype: np.dtype) -> int:
  """Bytes per element of `dtype` (read from an empty array: pyrefly-numpy-stubs' `dtype` has no
  `itemsize`)."""
  return np.empty(0, dtype).itemsize


def pixel_itemsize(pixel: PixelType) -> int:
  """Bytes per texel component of `pixel`."""
  return itemsize(PIXEL_DTYPES[pixel])


# Byte alignment of every array of a `.wpbr` file and of every image in the texel blob.
ALIGN: Final = 256

# Shaped host arrays of the scene tables: one row per object (a first dimension of any length).
# `Column[np.int32]` is (n,), `Rows[np.float32, 3]` (n, 3), `Matrices[np.float32, 3, 4]` (n, 3, 4).
type Column[D: np.generic] = np.ndarray[[int], np.dtype[D]]
type Rows[D: np.generic, K: IntVar] = np.ndarray[[int, K], np.dtype[D]]
type Matrices[D: np.generic, J: IntVar, K: IntVar] = np.ndarray[[int, J, K], np.dtype[D]]

# An array of any rank and dtype (the arrays of a `.wpbr` file, keyed by name).
type Array = np.ndarray[tuple[int, ...]]
# Image texels (height, width, channels) of any dtype, and linear float RGBA texels.
type Texels = np.ndarray[[int, int, int]]
type LinearRGBA = np.ndarray[[int, int, 4], np.dtype[np.float32]]

# A JSON value (scene metadata, `.wpbr` headers).
type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


class LightKind(enum.IntEnum):
  """Emitter shape. Point and spot lights are spheres (radius 0 = point) with shaping.

  CAMERA and AMBIENT are display lights with no position (MuJoCo's headlight and ambient term):
  the rows of each kind add up; their `xform`, sizes and flags are unused.
  """

  DISTANT = 0  # emits along the light's -Z; `angle` = angular diameter [deg]
  SPHERE = 1  # radius (0 or TREAT_AS_POINT: point light)
  RECT = 2  # width (local X) x height (local Y), emits along -Z
  DISK = 3  # radius, emits along -Z
  CYLINDER = 4  # radius, length along local X, emits radially
  # The headlight: a directional light along each camera's view, casting no shadow; its
  # `color` is linear and display-relative (illuminance pi x color / the render's exposure).
  CAMERA = 5
  AMBIENT = 6  # constant ambient radiance: its `color` [nits], unoccluded (lights every surface)


class LightFlags(enum.IntFlag):
  """Per-light switches."""

  CAST_SHADOW = 1
  TREAT_AS_POINT = 2  # sphere emitting from its centre
  TREAT_AS_LINE = 4  # cylinder emitting from its axis
  SHAPED = 8  # a cone (`cone_angle` < pi) or focus is authored: spot light
  VISIBLE = 16  # camera rays see its emitting surface (`Light.visible_in_primary_ray`)


class Projection(enum.IntEnum):
  """Camera projection."""

  PERSPECTIVE = 0
  ORTHOGRAPHIC = 1


class EnvFormat(enum.IntEnum):
  """Mapping of an environment image."""

  LATLONG = 0  # equirectangular: row 0 = +Z, centre column = +X (after the environment rotation)


IDENTITY_UV: Final = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
IDENTITY_UV.setflags(write=False)  # shared: never written


@dataclasses.dataclass(slots=True, frozen=True)
class TextureRef:
  """A material slot's texture binding.

  Attributes:
    texture: texture id (SceneBuilder.add_texture).
    channel: CHANNEL_RGB or a single channel index (0-3) broadcast to the slot's components.
    scale: per-component multiplier applied to the texel (UsdUVTexture scale).
    bias: per-component offset added after `scale` (UsdUVTexture bias).
    uv_xform: (2, 3) affine map from mesh UVs to texture coordinates (UsdTransform2d, MDL
      texture_scale / translate / rotate).
    uv_set: UV source: 0 = the mesh's primary set, 1 = its second set `Meshes.uv1_*`, UV_SET_WORLD =
      the hit's world position (metres, Z-up) projected along the dominant axis of its geometric
      normal, (y, z) / (x, z) / (x, y) for x- / y- / z-facing points, then through `uv_xform`
      (Unreal's world-aligned / triplanar projection, hard blend: exact on axis-aligned faces).
  """

  texture: TextureId
  _: dataclasses.KW_ONLY
  channel: int = CHANNEL_RGB
  scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
  bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
  uv_xform: tuple[tuple[float, float, float], tuple[float, float, float]] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
  uv_set: int = 0


# No slots: the table is built from each camera's `vars` (`table.record_columns`).
@dataclasses.dataclass(frozen=True, kw_only=True)
class Camera:
  """A camera (UsdGeom.Camera / MJCF camera). Looks down local -Z with +Y up.

  Attributes:
    xform: (3, 4) world transform (rotation only, no scale).
    projection: Projection.
    focal_length: focal length, in the units of the apertures.
    horizontal_aperture: sensor width (same units as `focal_length`; or the orthographic width [m]).
    vertical_aperture: sensor height.
    horizontal_offset: sensor offset.
    vertical_offset: sensor offset.
    near: near clipping distance [m].
    far: far clipping distance [m] (0 = unbounded).
    fstop: lens aperture (0 = pinhole).
    focus_distance: [m].
    exposure: exposure compensation [stops] over the world's exposure (USD's camera exposure:
      log2 of `UsdGeom.Camera.ComputeLinearExposureScale`).
  """

  xform: np.ndarray[[3, 4]]
  projection: int = Projection.PERSPECTIVE
  focal_length: float = 50.0
  horizontal_aperture: float = 20.955
  vertical_aperture: float = 15.2908
  horizontal_offset: float = 0.0
  vertical_offset: float = 0.0
  near: float = 0.01
  far: float = 0.0
  fstop: float = 0.0
  focus_distance: float = 0.0
  exposure: float = 0.0

  @property
  def fovy(self) -> float:
    """Vertical field of view [deg]."""
    return float(np.degrees(2.0 * np.arctan2(0.5 * self.vertical_aperture, self.focal_length)))
