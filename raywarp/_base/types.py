"""The renderer's shared vocabulary: enums and constants every part uses (no format-specific
code; the typed ids are `raywarp.scene.model.types`').

The integer values of `ShapeType`, `LightType`, `TexRole` and `ProjectionType` are MuJoCo's, so a
MuJoCo model's arrays are read by the kernels with no conversion; other adapters write the same
values.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Final, NewType

import numpy as np
import warp as wp

if TYPE_CHECKING:
  from raywarp._base.dims import AnyRank

# Slot of a Renderer camera (`Renderer.cameras`, which also holds free cameras): not a scene table
# row (`scene/model/types.py`).
CameraSlot = NewType("CameraSlot", int)

# The inner cone of a spot light as a fraction of its outer cone where no table gives one (MuJoCo's
# spots; `light.cone_inner`): a host constant (the scene's photometry reads it without the kernels).
SPOT_INNER_FRACTION: Final = 0.75

# Largest distance / bound used by ray queries, and the smallest denominator (as MuJoCo).
MAXVAL: Final = 1.0e10
MINVAL: Final = 1.0e-15


class ShapeType(enum.IntEnum):
  """Instance shape kinds (MuJoCo mjtGeom values; FLEX marks deformable triangle meshes)."""

  PLANE = 0
  HFIELD = 1
  SPHERE = 2
  CAPSULE = 3
  ELLIPSOID = 4
  CYLINDER = 5
  BOX = 6
  MESH = 7
  SDF = 8
  FLEX = 105


# Shapes traced through a triangle-mesh BVH.
MESH_LIKE: Final = (int(ShapeType.MESH), int(ShapeType.SDF), int(ShapeType.HFIELD))


class LightType(enum.IntEnum):
  """Punctual light kinds (MuJoCo mjtLightType values, then the core's own).

  COSINE: the point-light form of a flat Lambertian emitter (host rect / disk lights): intensity
  x cos(angle to `light.dir`), zero behind it; `light.cutoff` < 90 degrees adds a spot cone.
  TUBE: the point-light form of a Lambertian cylinder without end caps (host cylinder lights):
  intensity x sin(angle to `light.dir`, the cylinder's axis), its projected area 2 r l sin.
  """

  SPOT = 0
  DIRECTIONAL = 1
  POINT = 2
  IMAGE = 3
  COSINE = 4
  TUBE = 5


class LightShape(enum.IntEnum):
  """Emitter shape of a light, read with lighting.area_lights="ltc" ("point" ignores it).

  NONE is a punctual light (LightType). The others are area lights of uniform surface radiance
  `light.color * light.strength / projected area` (the strength stays the point-light
  equivalent [cd], so "point" renders the same light as a point or spot):

  * RECT: center `light.pos`, emits along `light.dir`, width along `light.axis`; half extents
    `light.size` (projected area 4 hx hy).
  * DISK: center, emits along `light.dir`, radius `light.radius` (pi r^2).
  * SPHERE: center, radius `light.radius` (pi r^2).
  * TUBE: cylinder of axis `light.axis`, half length `light.size[0]`, radius `light.radius`
    (2 r l; emits radially).
  """

  NONE = 0
  RECT = 1
  DISK = 2
  SPHERE = 3
  TUBE = 4


# Metallic, perceptual roughness and dielectric reflectance of an instance without a material
# (MuJoCo's default material: shininess 0.5, specular 0.5).
DEFAULT_MATERIAL_PARAMS: Final = (0.0, 0.5, 0.5)


class ProjectionType(enum.IntEnum):
  """Camera projections (FISHEYE is ours, after MuJoCo's two): a pinhole (with OpenCV's
  radial-tangential distortion), a parallel projection, and the Kannala-Brandt fisheye model of
  OpenCV's `cv::fisheye` (an equidistant f-theta lens without coefficients)."""

  PERSPECTIVE = 0
  ORTHOGRAPHIC = 1
  FISHEYE = 2


class TexRole(enum.IntEnum):
  """Fixed MuJoCo texture roles followed by optional OpenPBR roles."""

  USER = 0
  RGB = 1
  OCCLUSION = 2
  ROUGHNESS = 3
  METALLIC = 4
  NORMAL = 5
  OPACITY = 6
  EMISSIVE = 7
  RGBA = 8
  ORM = 9
  DIFFUSE_ROUGHNESS = 10  # optional OpenPBR role, after the fixed MuJoCo roles
  # The layered roles (Features.layered materials only: `NLAYERED_TEXROLE` wide tables).
  ANISOTROPY = 11  # x: the anisotropy strength factor
  FUZZ = 12  # rgb: the fuzz color factor, a: the fuzz roughness factor
  COAT_NORMAL = 13  # tangent-space normal map of the coat
  DETAIL_NORMAL = 14  # tangent-space detail normal map (RNM over NORMAL; own UV transform)


NTEXROLE: Final = 10  # the fixed MuJoCo-compatible prefix (also the untextured-OpenPBR width)
NOPENPBR_TEXROLE: Final = 11
NLAYERED_TEXROLE: Final = 15


# A 2D affine map (2 x 3): texture coordinates of a mesh instance's UVs, uv' = M (u, v, 1).
uv_xform = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


@wp.struct
class MeshAdr:
  """Where mesh m's rows start in the shared mesh pools (`Meshes.adr`, one entry per mesh: one
  launch-parameter array instead of five).

  Attributes:
    vert: its first vertex (`mesh_vert`); face: its first triangle (`mesh_face` and the corner
      tables aligned with it); normal: its first normal (`mesh_normal`, -1: none); color: its
      first vertex color (`mesh_color`, -1: none); texcoord: its first UV (`mesh_texcoord`, -1:
      none).
  """

  vert: wp.int32
  face: wp.int32
  normal: wp.int32
  color: wp.int32
  texcoord: wp.int32


@wp.struct
class MaterialUvMapping:
  """Authored mesh UV set and residual affine transform of one material."""

  uv_set: int
  xform: uv_xform


# `MeshInstances.texcoord` of an instance that reads its mesh's own UVs (set 0, no transform).
UV_MESH: Final = -2
# The identity UV transform (2 x 3: [linear | offset]).
UV_IDENTITY: Final = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)

# A material's texel map of one texture role (2 x 4 rows: scale, bias):
# texel' = scale * texel + bias per channel (`material.tex_affine`: tints, UsdUVTexture scale
# / bias, occlusion strength, normal scale, emissive color, contrast; the core textures hold only
# the selected channels). Texture variation without copies is one vec4 per material
# (`material.tex_variation`: u offset, v offset, UV rotation [rad] about the texture's center, hue
# rotation [rad] of the color roles), per material or per world
# (`PerWorld(material=["tex_variation"])`, `rw.randomize`).
tex_affine = wp.types.matrix(shape=(2, 4), dtype=wp.float32)
# The identity texel map.
TEX_IDENTITY: Final = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], np.float32)

# A core texture's texel layout (`pipeline/textures/formats.pack_channels`,
# `shade/materials/texture.apply_layout`): bits 3k..3k+2 give read channel k's stored channel (0-3)
# or TEX_LAYOUT_ONE (the constant 1).
TEX_LAYOUT_ONE: Final = 4
TEX_LAYOUT_IDENTITY: Final = 0 | (1 << 3) | (2 << 6) | (3 << 9)


# Bits of `GeometryFields.instance_twosided`: never back-face culled (USD doubleSided); a closed
# see-through pane (a closed alpha-blended shape, not glass: single-sided, its exit face culled,
# its entry face composites both faces, `kernels/layer._folded`).
TWO_SIDED: Final = 1
CLOSED_PANE: Final = 2


class ShadingFlag(enum.IntFlag):
  """Bits of the `instance.shading` table (host scenes; absent = 0).

  Attributes:
    SPECULAR_CURVE: Unreal / OmniUe4Base specular (`df::custom_curve_layer`): dielectric F0 =
      0.08 x level (stored as reflectance sqrt(level / 2), so 0.16 r^2 still gives F0) and F90 =
      level x (1 - roughness) instead of 1; the metallic lobe reflects the base color at every
      angle (F90 = F0).
    TRANSMISSIVE: the material transmits light (glTF KHR_materials_transmission, OmniGlass;
      a thin translucent layer in the core): its specular, and its diffuse lobe weighted by its
      opacity 1 - t (OpenPBR's diffuse / transmission mix; `make_shading_lobes`). Alpha-blended
      materials (coverage: glTF BLEND, UsdPreviewSurface opacity, MuJoCo rgba) keep their whole
      diffuse lobe.
    VERTEX_COLOR: multiply the mesh's color/alpha attributes in (the material's
      `MaterialFlags.VERTEX_COLOR`; a mesh without colors ignores it). The kernels read it from
      the instance's shading bits: the prepared material's, per world under declared material
      selection; a declared "instance.shading" table turns an instance's vertex colors on or off
      per world (where the scene compiles vertex colors: some material uses them).
    DIELECTRIC_SPECULAR_LEVEL: preserve a native dielectric-level model: the material weight
      is already folded into dielectric reflectance; metal reflection is unscaled and
      dielectric F90 remains 1, including at zero level.
    UNLIT: an unlit material (glTF KHR_materials_unlit, MaterialX surface_unlit;
      `MaterialFlags.UNLIT`): no light reaches it, its base color plus emission shows
      display-relative (divided by the world's exposure, as `Params.relative_emission` shows a
      MuJoCo scene's emission; `Features.unlit_emission`).
    DISPLAY_EMISSION: the material's emission is display-relative (`MaterialFlags.DISPLAY_EMISSION`,
      glTF's emissive factor): shown as `UNLIT`'s emission (divided by the world's exposure),
      its surface lit as any other (`Features.unlit_emission`, the same gate).
    OPTICS: the instance's material has optics rows its surface evaluation reads beyond the
      transmissive layers' (`MaterialTables.instance_optics`: reserved for specular colour and
      thin film, run-time data).

  Bits 8-15 and 16-23 hold the clearcoat layer (`pack_clearcoat`): its weight and perceptual
  roughness quantized to 0-255 (a weight of 0: no coat).
  """

  SPECULAR_CURVE = 1
  TRANSMISSIVE = 2
  VERTEX_COLOR = 4
  DIELECTRIC_SPECULAR_LEVEL = 8
  OPTICS = 16
  UNLIT = 32
  DISPLAY_EMISSION = 64


class LayerFlag(enum.IntFlag):
  """Bits of a material's `LayerConstants.flags` (Features.layered; `pipeline/derive/materials.py`).

  Attributes:
    COAT_OPENPBR: an OpenPBR coat: its own IOR, color and darkening, and the base under it
      albedo-scaled lerp(1, T (1 - E_coat), C) (without it: glTF's 1 - C F(LoH), F0 0.04).
    COAT_BASE_NORMAL: the coat reflects about the base's mapped normal (MaterialFlags).
    COAT_NORMAL_MAP: the material binds `geometry_coat_normal` (TexRole.COAT_NORMAL).
    DETAIL_NORMAL: the material binds `geometry_detail_normal` (TexRole.DETAIL_NORMAL).
  """

  COAT_OPENPBR = 1
  COAT_BASE_NORMAL = 2
  COAT_NORMAL_MAP = 4
  DETAIL_NORMAL = 8


# instance.shading: the clearcoat layer's weight and roughness bytes (ShadingFlag).
COAT_WEIGHT_SHIFT: Final = 8
COAT_ROUGHNESS_SHIFT: Final = 16


def pack_clearcoat(weight: float | AnyRank[np.float64], roughness: float | AnyRank[np.float64]) -> AnyRank[np.int64]:
  """instance.shading bits of a clearcoat layer (weight and roughness in [0, 1], clipped)."""
  w = np.rint(np.clip(np.asarray(weight, np.float64), 0.0, 1.0) * 255.0).astype(np.int64)
  r = np.rint(np.clip(np.asarray(roughness, np.float64), 0.0, 1.0) * 255.0).astype(np.int64)
  return (w << COAT_WEIGHT_SHIFT) | (r << COAT_ROUGHNESS_SHIFT)


# Slots whose texels are colors (decoded from sRGB unless the texture says linear).
COLOR_ROLES: Final = (int(TexRole.RGB), int(TexRole.RGBA), int(TexRole.EMISSIVE))


class ToneMapper(enum.IntEnum):
  """Display transform applied to exposed linear HDR before sRGB encoding.

  Attributes:
    PBR_NEUTRAL: Khronos PBR Neutral, preserves base colors under neutral lighting.
    ACES: ACES RRT + ODT fit.
    CLAMP: no tone mapping, clamp to [0, 1].
    ACES_RTX: Omniverse RTX's "acesApproximation" (Narkowicz's fit), using the same
      L / 2^EV100 exposure convention as every other curve.
  """

  PBR_NEUTRAL = 0
  ACES = 1
  CLAMP = 2
  ACES_RTX = 3


class RenderOrder(enum.IntEnum):
  """Assignment of pixels to threads (outputs are always stored row-major; pixels are the same in
  every order: `trace/rays/camera.thread_ids`).

  Attributes:
    AUTO: chosen every frame on the device (`cameras.CameraRig.pose`): MIXED when every world
      sees through the same cameras (equal poses and lenses) and there are at least 4 worlds,
      else TILED.
    TILED: bands of rows traversed column by column, one world per warp: 8-row bands (4 x 8 warps)
      for shaded builds, 4-row bands (8 x 4 warps, full 32-byte sectors of 4-byte outputs) where
      the stores dominate (`trace/rays/camera.tile_rows`).
    WORLD: a warp renders one pixel in 32 consecutive worlds. Fast on the software traversal
      when worlds render identical images, slow otherwise (and on OptiX).
    MIXED: a warp renders 8 pixels of a row in 4 consecutive worlds: faster than TILED when the
      cameras are the same in every world, slower when they differ.
  """

  AUTO = 0
  TILED = 1
  WORLD = 2
  MIXED = 3


class ColorRule(enum.IntEnum):
  """How an instance with a material combines its color with the material's base color (the
  source's convention, `GpuScene.color_rule`; `kernels.abi.tables.instance_rgba`).

  Attributes:
    TINT: the instance color multiplies the material's (host scenes: white shows the material).
    MUJOCO: MuJoCo's rule: the default geom rgba (0.5, 0.5, 0.5, 1) shows the material's color,
      any other geom rgba replaces it.
  """

  TINT = 0
  MUJOCO = 1


# vec4 optics rows per instance (`pipeline/upload/optics.py`, `MaterialTables.instance_optics`): the
# medium row (Beer-Lambert extinction rgb, refraction index).
OPTICS_ROWS: Final = 1
# The optics row of the medium values (extinction RGB, then the refraction index at
# component MEDIUM_IOR; 0: thin-walled).
ROW_MEDIUM: Final = 0
MEDIUM_IOR: Final = 3
