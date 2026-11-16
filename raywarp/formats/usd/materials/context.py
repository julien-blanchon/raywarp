"""UsdShade materials -> core materials: the conversion context (`ShadeContext`) and the MDL
baker role, shared by the converters.

Every converter writes the canonical OpenPBR `Material`. `select.py` picks the surface of a
material by USD's render-context rule (`Options.material_contexts`) and dispatches it:
UsdPreviewSurface networks to `preview.py`; MDL and MaterialX materials to the context's
`MaterialBaker` (`mdl.SdkBaker`, through the NVIDIA MDL SDK; MaterialX through the MaterialX
library's MDL generator, `materialx.py`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Protocol, final

import numpy as np
from pxr import Sdf, Usd, UsdShade

from raywarp.formats.usd.stage.items import Display
from raywarp.formats.usd.util import Time, Vec3
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.images.colorspace import ColorSpace, auto_colorspace
from raywarp.scene.model.material import Material
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import NO_TEXTURE, AlphaMode, MaterialFlags, MaterialId, Slot, TextureId, Wrap

if TYPE_CHECKING:
  from raywarp.formats.usd.materials.udim import Atlas

# What a conversion's reports name: its shader prim.
type Where = Usd.Prim
# Slots whose texels are colors: "auto" color space means sRGB for them, raw for the others.
COLOR_SLOTS: Final = (Slot.BASE_COLOR, Slot.EMISSION_COLOR)
# The IOR of a glTF clearcoat (KHR_materials_clearcoat: a fixed 1.5), UsdPreviewSurface's too.
GLTF_COAT_IOR: Final = 1.5
# Material of gprims without a binding: Hydra's grey display color.
FALLBACK: Final = Material(base_color=(0.5, 0.5, 0.5), base_metalness=0.0, specular_roughness=0.5, name="<fallback>")
# Mesh UV primvars by set (MaterialX texcoord index, MDL texture_coordinate index).
UV_PRIMVARS: Final = ("st", "st1")
# Custom attribute of a UsdShade.Material naming a vertex-color primvar that multiplies the base
# color (written by the glTF converter, `raywarp.formats.gltf.to_usd`, for COLOR_0;
# UsdPreviewSurface cannot express it).
VERTEX_COLOR_ATTR: Final = "raywarp:vertexColor"
# Custom bool attribute of a UsdShade.Material: an unlit material (`MaterialFlags.UNLIT`: its base
# color and emission shown display-relative, no light), written by the glTF converter for
# KHR_materials_unlit (UsdPreviewSurface cannot express it).
UNLIT_ATTR: Final = "raywarp:unlit"
# Custom bool attribute of a UsdShade.Material: its emission is display-relative
# (`MaterialFlags.DISPLAY_EMISSION`: 1 is the brightest LDR value, not 1 nit), written by the glTF
# converter for an `emissiveFactor` (UsdPreviewSurface's `emissiveColor` is a radiance).
DISPLAY_EMISSION_ATTR: Final = "raywarp:displayEmission"
# The MaterialX unlit surface's nodedef prefix (`ND_surface_unlit`): its color is its emission.
MTLX_UNLIT: Final = "ND_surface_unlit"

type UvMatrix = np.ndarray[[2, 3], np.dtype[np.float64]]
# A (2, 3) affine UV map as TextureRef.uv_xform rows.
type UvRows = tuple[tuple[float, float, float], tuple[float, float, float]]


def uv_matrix(
  scale: Sequence[float] = (1.0, 1.0), rotation_deg: float = 0.0, translation: Sequence[float] = (0.0, 0.0)
) -> UvMatrix:
  """(2, 3) affine T * R * S (UsdTransform2d / OmniPBR texture_scale, rotate, translate)."""
  c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
  sx, sy = float(scale[0]), float(scale[1])
  return np.array([[c * sx, -s * sy, float(translation[0])], [s * sx, c * sy, float(translation[1])]])


def uv_rows(m: UvMatrix) -> UvRows:
  """A (2, 3) UV matrix as TextureRef.uv_xform rows."""
  return (
    (float(m[0, 0]), float(m[0, 1]), float(m[0, 2])),
    (float(m[1, 0]), float(m[1, 1]), float(m[1, 2])),
  )


def compose_uv(outer: UvMatrix, inner: UvMatrix) -> UvMatrix:
  """Outer o inner for (2, 3) affine UV maps."""
  a = np.vstack([outer, [0.0, 0.0, 1.0]]) @ np.vstack([inner, [0.0, 0.0, 1.0]])
  return a[:2]


@final
@dataclasses.dataclass(frozen=True)
class MeshNeeds:
  """Mesh attributes a material reads: UV primvars (set 0, set 1), vertex-color primvar."""

  uv: tuple[str, ...] = ()
  color: str | None = None


# The material binding relationships of the three purposes (UsdShadeMaterialBindingAPI).
BINDINGS: Final = ("material:binding", "material:binding:full", "material:binding:preview")


class MaterialXError(ValueError):
  """A MaterialX material the official library cannot read or generate MDL for (`materialx.py`)."""


class MaterialBaker(Protocol):
  """Converts MDL and MaterialX materials to core materials and baked textures."""

  def convert(self, ctx: ShadeContext, shader: UsdShade.Shader) -> tuple[Material, MeshNeeds]:
    """(material, mesh attributes it reads) of an MDL shader or a MaterialX surface shader;
    textures go through `ctx`."""
    ...

  def dependencies(self) -> list[str]:
    """Files the converted materials came from (.mdl modules, baked texture sources)."""
    ...


def set_ext(mat: Material, **fields: float) -> Material:
  """`mat` with RaywarpExt fields set (alpha mode and cutoff, flags)."""
  # Field names vary: pyrefly cannot match a dynamic keyword set to RaywarpExt's fields.
  return dataclasses.replace(mat, ext=dataclasses.replace(mat.ext, **fields))  # pyrefly: ignore[bad-argument-type]


def with_flags(mat: Material, flags: MaterialFlags) -> Material:
  """`mat` with `flags` added to its `ext.flags`."""
  return set_ext(mat, flags=mat.ext.flags | int(flags))


def gltf_coat(mat: Material, *, base_normal: bool) -> Material:
  """`mat`'s coat, if any, as glTF's (UsdPreviewSurface's): COAT_GLTF at IOR 1.5, the base
  unchanged under it (coat_darkening 0 in extras the material has); `base_normal`: the coat
  reflects about the base's mapped normal (COAT_BASE_NORMAL: one `normal` input for every lobe)."""
  if mat.coat_weight <= 0.0:
    return mat
  extras = None if mat.extras is None else dataclasses.replace(mat.extras, coat_darkening=0.0)
  flags = MaterialFlags.COAT_GLTF | (MaterialFlags.COAT_BASE_NORMAL if base_normal else MaterialFlags(0))
  return with_flags(dataclasses.replace(mat, coat_ior=GLTF_COAT_IOR, extras=extras), flags)


@final
@dataclasses.dataclass(frozen=True)
class StageFrame:
  """The stage's world frame against the scene's (metres, Z-up).

  Attributes:
    meters_per_unit: the stage's metersPerUnit (`state::meters_per_scene_unit`).
    stage_from_scene: (3, 3) map from scene coordinates to stage world coordinates (the inverse
      of `reader.stage_root`'s linear part).
  """

  meters_per_unit: float
  stage_from_scene: np.ndarray[[3, 3], np.dtype[np.float64]]


@final
@dataclasses.dataclass(kw_only=True)
class ShadeContext:
  """Material conversion state shared by a stage read."""

  builder: SceneBuilder
  report: LoadReport
  max_texture_size: int | None
  time: Time
  baker: MaterialBaker
  frame: StageFrame
  # Render contexts tried after the universal one, in order (`Options.material_contexts`).
  contexts: tuple[str, ...]
  materials: dict[str, tuple[MaterialId, MeshNeeds]] = dataclasses.field(default_factory=dict)
  # The materials a prim (or a prototype's prim) binds directly, in scan order (the stage
  # reader's `_scan`; collection bindings not followed).
  bound: dict[Sdf.Path, None] = dataclasses.field(default_factory=dict)

  def texture(
    self,
    path: str,
    slot: Slot,
    colorspace: ColorSpace | None = None,
    wrap: tuple[int, int] = (Wrap.REPEAT, Wrap.REPEAT),
    *,
    image: Atlas | None = None,
  ) -> TextureId:
    """Texture id for a file (NO_TEXTURE if missing / unsupported, reported by the builder), or
    for the decoded `image` of a UDIM set (`path`: its template, which names the color space)."""
    if not path:
      return NO_TEXTURE
    if colorspace is None:
      colorspace = auto_colorspace(path, color=slot in COLOR_SLOTS)
    source = path if image is None else image
    return self.builder.add_texture(source, colorspace=colorspace, wrap=wrap, max_size=self.max_texture_size)

  def fallback(self, display: Display) -> MaterialId:
    """Material of an unbound gprim: its displayColor / displayOpacity, or grey.

    `display.vertex`: the mesh carries its displayColor / displayOpacity as vertex colors (white
    base color; blended when its mean opacity is below 1: the vertex alpha is the opacity).
    """
    mat = FALLBACK
    if display.vertex:
      vertex = dataclasses.replace(mat, base_color=(1.0, 1.0, 1.0), name="<displayColor>")
      if display.opacity is not None and display.opacity < 1.0:
        vertex = dataclasses.replace(vertex, ext=dataclasses.replace(vertex.ext, alpha_mode=AlphaMode.BLEND))
      return self.builder.add_material(with_flags(vertex, MaterialFlags.VERTEX_COLOR))
    if display.color is not None:
      mat = dataclasses.replace(mat, base_color=display.color, name="<displayColor>")
    if display.opacity is not None and display.opacity < 1.0:
      blend = dataclasses.replace(mat.ext, alpha_mode=AlphaMode.BLEND)
      mat = dataclasses.replace(mat, geometry_opacity=display.opacity, ext=blend)
    return self.builder.add_material(mat)

  def emitting(self, material: MaterialId, radiance: Vec3) -> MaterialId:
    """Material `material` emitting `radiance` (linear RGB [nits]) from its surface
    (MeshLightAPI)."""
    peak = float(np.max(radiance))
    if peak <= 0.0:
      return material
    mat = self.builder.material(material)
    tint = (float(radiance[0]) / peak, float(radiance[1]) / peak, float(radiance[2]) / peak)
    return self.builder.add_material(dataclasses.replace(mat, emission_color=tint, emission_luminance=peak))
