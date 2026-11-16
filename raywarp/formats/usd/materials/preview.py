"""UsdPreviewSurface networks -> canonical OpenPBR materials.

UsdUVTexture nodes become texture slots with their channel, scale / bias (of that channel for a
single-channel output), wrap modes, color space and the UsdTransform2d feeding their `st`; the
UsdPrimvarReader_float2 `varname` selects the mesh UV set; a file that does not resolve gives the
node's `fallback`. Inputs that are authored but not representable are counted in the load report.

The specular workflow becomes the metallic one (Khronos' conversion; per texel when textured); a
constant dielectric keeps its diffuse color with `specular_weight` = max(specularColor) / 0.04 and
the normalized specularColor as `specular_color`. The clearcoat is glTF's (COAT_GLTF, IOR 1.5) over
the base's mapped normal (COAT_BASE_NORMAL: the one `normal` input shades every lobe).
"""

from __future__ import annotations

import dataclasses
from typing import Final, final

import numpy as np
from pxr import Sdf, Usd, UsdShade

from raywarp.formats.usd import util
from raywarp.formats.usd.materials import udim
from raywarp.formats.usd.materials.context import (
  MeshNeeds,
  ShadeContext,
  UvMatrix,
  compose_uv,
  gltf_coat,
  set_ext,
  uv_matrix,
  uv_rows,
  with_flags,
)
from raywarp.formats.usd.util import Color, UsdValue
from raywarp.scene.images import decode as imaging
from raywarp.scene.images.colorspace import ColorSpace, auto_colorspace
from raywarp.scene.model import normal_map, openpbr
from raywarp.scene.model.material import Displacement, Material
from raywarp.scene.model.types import CHANNEL_RGB, CHANNELS, AlphaMode, MaterialFlags, Slot, TextureId, TextureRef, Wrap

_WRAPS: Final = {
  "repeat": Wrap.REPEAT,
  "clamp": Wrap.CLAMP,
  "mirror": Wrap.MIRROR,
  "black": Wrap.BLACK,
  "useMetadata": Wrap.REPEAT,
}
# UsdPreviewSurface's own defaults, written against OpenPBR's: diffuseColor 0.18, roughness 0.5,
# clearcoatRoughness 0.01, emissiveColor black as radiance [nits] (luminance 1).
PREVIEW_DEFAULT: Final = Material(
  base_color=(0.18, 0.18, 0.18),
  specular_roughness=0.5,
  coat_roughness=0.01,
  emission_color=(0.0, 0.0, 0.0),
  emission_luminance=1.0,
)
# The dielectric F0 of specular weight 1 at IOR 1.5 (the specular workflow's weight unit).
_DIELECTRIC_F0: Final = 0.04
# The inputs UsdPreviewSurface defines (others are reported).
_PREVIEW_INPUTS: Final = frozenset(
  (
    *("diffuseColor", "emissiveColor", "useSpecularWorkflow", "specularColor", "metallic", "roughness", "clearcoat"),
    *("clearcoatRoughness", "opacity", "opacityThreshold", "opacityMode", "ior", "normal", "occlusion", "displacement"),
  )
)
# Scalar inputs: (UsdPreviewSurface input, slot, Material field).
_SCALARS: Final = (
  ("metallic", Slot.BASE_METALNESS, "base_metalness"),
  ("roughness", Slot.SPECULAR_ROUGHNESS, "specular_roughness"),
  ("opacity", Slot.GEOMETRY_OPACITY, "geometry_opacity"),
)
_IDENTITY_UV: Final = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Longest UsdTransform2d chain followed from a UsdUVTexture's `st` (guards a connection cycle).
_MAX_ST_CHAIN: Final = 8
type Vec4 = tuple[float, float, float, float]
type Texels = np.ndarray[[int, int, int], np.dtype[np.float32]]
type Values = np.ndarray[[int, int, int], np.dtype[np.float64]]
type Colors = np.ndarray[[int, int, 3], np.dtype[np.float64]]
type Scalars = np.ndarray[[int, int], np.dtype[np.float64]]


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UvTexture:
  """A UsdUVTexture node as one material slot reads it (one output: rgb or a channel)."""

  path: str  # resolved file ('' when it does not resolve)
  channel: int  # CHANNEL_RGB or the output's channel
  scale: Vec4  # of the output: that channel's, broadcast, for a single-channel one
  bias: Vec4
  wrap: tuple[int, int]
  colorspace: ColorSpace | None
  uv: UvMatrix  # UsdTransform2d chain feeding `st`
  varname: str  # UV primvar
  fallback: Vec4  # the output when the file does not resolve
  # A UDIM set's tiles as one atlas (`udim`; `path` is then the set's template and `uv`
  # ends with the atlas' 1 / (cols, rows) scale); None for a single file.
  atlas: udim.Atlas | None = dataclasses.field(default=None, compare=False)

  def fallback_value(self, n: int) -> tuple[float, ...]:
    """`n` components of the fallback output: rgb, or the channel's broadcast."""
    return self.fallback[:n] if self.channel < 0 else (self.fallback[self.channel],) * n


def _varname(inputs: dict[str, UsdShade.Input]) -> str | None:
  """The primvar a UsdPrimvarReader reads (None: unset)."""
  name = util.input_value(inputs, "varname")
  return str(name) if name else None


def _vec4(inputs: dict[str, UsdShade.Input], name: str, default: Vec4) -> Vec4:
  """A float4 input's value, else `default`."""
  x, y, z, w = util.input_floats(inputs, name, default)[:4]
  return x, y, z, w


def _st(ctx: ShadeContext, inputs: dict[str, UsdShade.Input]) -> tuple[UvMatrix, str]:
  """(UV transform, primvar name) of the `st` chain of a UsdUVTexture (UsdTransform2d nodes)."""
  uv = _IDENTITY_UV
  st_attr = util.connected_output(inputs["st"]) if "st" in inputs else None
  for _ in range(_MAX_ST_CHAIN):
    if st_attr is None:
      break
    ins = util.shader_inputs(st_attr.GetPrim())
    node_id = str(UsdShade.Shader(st_attr.GetPrim()).GetIdAttr().Get())
    if node_id.startswith("UsdPrimvarReader"):
      return uv, _varname(ins) or "st"
    if node_id != "UsdTransform2d":
      ctx.report.drop(f"UsdUVTexture.st_node:{node_id}", st_attr.GetPrim())
      break
    scale, translation = util.input_floats(ins, "scale", (1.0, 1.0)), util.input_floats(ins, "translation", (0.0, 0.0))
    uv = compose_uv(uv, uv_matrix(scale, util.input_floats(ins, "rotation", (0.0,))[0], translation))
    st_attr = util.connected_output(ins["in"]) if "in" in ins else None
  return uv, "st"


def texture_node(ctx: ShadeContext, attr: Usd.Attribute) -> UvTexture | None:
  """The UsdUVTexture behind a connected output attribute (None for other node types)."""
  prim = attr.GetPrim()
  node = UsdShade.Shader(prim)
  if node.GetIdAttr().Get() != "UsdUVTexture":
    ctx.report.drop(f"preview.connected_node:{node.GetIdAttr().Get()}", prim)
    return None
  inputs = util.shader_inputs(prim)
  path, file_attr = "", None
  fallback = _vec4(inputs, "fallback", (0.0, 0.0, 0.0, 1.0))
  atlas, udim_scale = None, None
  if "file" in inputs:
    v, file_attr = util.producing(inputs["file"])
    path = util.asset_path(v)
    if isinstance(v, Sdf.AssetPath) and file_attr is not None and udim.is_template(v.path):
      path, atlas, udim_scale = _udim(ctx, prim, v.path, file_attr, fallback)
    elif isinstance(v, Sdf.AssetPath) and v.path and not path:
      ctx.report.drop("UsdUVTexture.file.unresolved(fallback used)", prim)
  ws, wt = (_WRAPS.get(str(util.input_value(inputs, n)), Wrap.REPEAT) for n in ("wrapS", "wrapT"))
  scale, bias = _vec4(inputs, "scale", (1.0, 1.0, 1.0, 1.0)), _vec4(inputs, "bias", (0.0, 0.0, 0.0, 0.0))
  space = util.input_value(inputs, "sourceColorSpace")
  colorspace = util.colorspace_of(file_attr, "auto" if space is None else str(space), report=ctx.report, where=prim)
  uv, varname = _st(ctx, inputs)
  if udim_scale is not None:
    uv = compose_uv(uv_matrix(udim_scale), uv)
  channel = CHANNELS.get(attr.GetBaseName(), CHANNEL_RGB)
  if channel >= 0:
    # outputs:r / g / b / a = texel.c * scale.c + bias.c, broadcast to the slot's components.
    s, b = scale[channel], bias[channel]
    scale, bias = (s, s, s, s), (b, b, b, b)
  return UvTexture(
    path=path,
    channel=channel,
    scale=scale,
    bias=bias,
    wrap=(ws, wt),
    colorspace=colorspace,
    uv=uv,
    varname=varname,
    fallback=fallback,
    atlas=atlas,
  )


def _udim(
  ctx: ShadeContext, prim: Usd.Prim, template: str, file_attr: Usd.Attribute, fallback: Vec4
) -> tuple[str, udim.Atlas | None, tuple[float, float] | None]:
  """(template path, atlas, its UV scale) of a UDIM file value (`udim`); ("", None, None),
  reported, when no tile resolves or a tile is not an 8-bit image."""
  layer = udim.template_layer(file_attr)
  found = udim.tiles(template, layer) if layer is not None else {}
  if not found:
    ctx.report.drop("UsdUVTexture.udim.no_tile", prim)
    return "", None, None
  if any(imaging.is_float(p) for p in found.values()):
    ctx.report.drop("UsdUVTexture.udim.float_tiles", prim)
    return "", None, None
  image, cols, rows = udim.atlas(found, fallback)
  ctx.report.count("UsdUVTexture.udim")
  anchored = layer.ComputeAbsolutePath(template) if layer is not None else template
  return anchored, image, (1.0 / cols, 1.0 / rows)


def specular_to_metallic_arrays(diffuse: Colors, specular: Colors) -> tuple[Scalars, Colors]:
  """(metallic (...), base color (..., 3)) of specular-workflow (diffuse, F0) arrays (..., 3).

  The Khronos KHR_materials_pbrSpecularGlossiness -> metallic-roughness conversion: metallic
  solved from the perceived brightness of both colors against a dielectric F0 of 0.04; the base
  color blends the diffuse (dielectric) and specular (metal) readings by metallic^2.
  """
  d = np.clip(np.asarray(diffuse, np.float64)[..., :3], 0.0, 1.0)
  sp = np.clip(np.asarray(specular, np.float64)[..., :3], 0.0, 1.0)
  f0 = _DIELECTRIC_F0

  def brightness(c: Colors) -> Scalars:
    """Perceived brightness."""
    return np.sqrt(0.299 * c[..., 0] ** 2 + 0.587 * c[..., 1] ** 2 + 0.114 * c[..., 2] ** 2)

  one_minus = 1.0 - sp.max(axis=-1)
  bd, bs = brightness(d), brightness(sp)
  b = bd * one_minus / (1.0 - f0) + bs - 2.0 * f0
  c = f0 - bs
  root = (-b + np.sqrt(np.maximum(b * b - 4.0 * f0 * c, 0.0))) / (2.0 * f0)
  metallic = np.where(bs >= f0, np.clip(root, 0.0, 1.0), 0.0)
  from_diffuse = d * (one_minus / (1.0 - f0) / np.maximum(1.0 - metallic, 1e-6))[..., None]
  from_specular = (sp - f0 * (1.0 - metallic)[..., None]) / np.maximum(metallic, 1e-6)[..., None]
  t = (metallic * metallic)[..., None]
  base = np.clip(from_diffuse * (1.0 - t) + from_specular * t, 0.0, 1.0)
  return metallic, base


def specular_to_metallic(diffuse: Color, specular: Color) -> tuple[float, Color]:
  """(metallic, base color) of one specular-workflow (diffuse, F0 color) pair (see above)."""
  metallic, base = specular_to_metallic_arrays(np.asarray(diffuse)[None, None], np.asarray(specular)[None, None])
  return float(metallic[0, 0]), (float(base[0, 0, 0]), float(base[0, 0, 1]), float(base[0, 0, 2]))


def _plan(tex: UvTexture | None, const: Color, sources: list[imaging.ImageSource], ctx: ShadeContext):
  """Per-texel color of a specular-workflow input: a texture output (added to `sources`) or a
  constant."""
  if tex is None:
    value = np.asarray(const, np.float32)
    return lambda inputs: np.broadcast_to(value, (*inputs[0].shape[:2], 3)).astype(np.float64)
  sources.append(imaging.ImageSource(path=tex.path, max_size=ctx.max_texture_size, transfer="srgb" if _srgb(tex) else "linear"))
  index, channel = len(sources) - 1, tex.channel
  scale, bias = np.asarray(tex.scale, np.float32), np.asarray(tex.bias, np.float32)

  def value_of(inputs: list[Texels]) -> Colors:
    """The texture output's color per texel."""
    texel = inputs[index]
    rgba = np.repeat(texel[..., channel : channel + 1], 4, axis=2) if channel >= 0 else texel
    return np.asarray((rgba * scale + bias)[..., :3], np.float64)

  return value_of


def _srgb(tex: UvTexture) -> bool:
  """Whether a specular-workflow input (a color) decodes as sRGB."""
  cs = tex.colorspace if tex.colorspace is not None else auto_colorspace(tex.path, color=True)
  return cs == ColorSpace.SRGB_REC709


def _key(tex: UvTexture | None, const: Color) -> tuple[object, ...]:
  """Identity of a specular-workflow input in the derived images' key."""
  if tex is None:
    return tuple(float(x) for x in const)
  return (tex.path, _srgb(tex), tex.channel, tex.scale, tex.bias)


def bake_specular_workflow(
  ctx: ShadeContext, prim: Usd.Prim, diffuse: tuple[UvTexture | None, Color], specular: tuple[UvTexture | None, Color]
) -> tuple[TextureId, TextureId, UvTexture] | None:
  """Bakes the specular -> metallic conversion per texel.

  Returns (base color texture, metallic texture, the texture whose UV / wrap they use), None
  when the two textures' UVs differ.
  """
  texs = [t for t, _ in (diffuse, specular) if t is not None]
  ref = texs[0]
  if any(t.atlas is not None for t in texs):
    ctx.report.drop("UsdPreviewSurface.useSpecularWorkflow.udim", prim)
    return None
  if any(t.varname != ref.varname or not np.allclose(t.uv, ref.uv) for t in texs):
    ctx.report.drop("UsdPreviewSurface.useSpecularWorkflow.textures_on_different_uvs", prim)
    return None
  sources: list[imaging.ImageSource] = []
  plans = [_plan(tex, const, sources, ctx) for tex, const in (diffuse, specular)]

  def base_fn(inputs: list[Texels]) -> Values:
    """Base color per texel (alpha 1)."""
    _, base = specular_to_metallic_arrays(plans[0](inputs), plans[1](inputs))
    return np.concatenate([base, np.ones((*base.shape[:2], 1))], axis=2)

  def metal_fn(inputs: list[Texels]) -> Values:
    """Metallic per texel."""
    metallic, _ = specular_to_metallic_arrays(plans[0](inputs), plans[1](inputs))
    return metallic[..., None]

  ident = ("specular_workflow", _key(*diffuse), _key(*specular), ctx.max_texture_size)
  base_img = ctx.builder.add_derived_image(
    imaging.Derived(sources=tuple(sources), fn=base_fn, channels=4, encoding="srgb", key=(*ident, "base"))
  )
  metal_img = ctx.builder.add_derived_image(
    imaging.Derived(sources=tuple(sources), fn=metal_fn, channels=1, encoding="linear", key=(*ident, "metallic"))
  )
  if base_img < 0 or metal_img < 0:
    return None
  base_tex = ctx.builder.add_texture(base_img, colorspace=ColorSpace.SRGB_REC709, wrap=ref.wrap)
  metal_tex = ctx.builder.add_texture(metal_img, colorspace=ColorSpace.LIN_REC709, wrap=ref.wrap)
  return base_tex, metal_tex, ref


@final
class _Preview:
  """Builds the Material of one UsdPreviewSurface shader (`preview_surface`)."""

  def __init__(self, ctx: ShadeContext, shader: UsdShade.Shader) -> None:
    """Starts from UsdPreviewSurface's defaults."""
    self.ctx = ctx
    self.shader = shader
    self.prim = shader.GetPrim()
    self.mat = PREVIEW_DEFAULT
    self.uv_names: list[str] = []
    self.bound: dict[Slot, UvTexture] = {}

  def set(self, **fields: float | tuple[float, ...]) -> None:
    """Sets Material fields."""
    # Field names vary: pyrefly cannot match a dynamic keyword set to Material's fields.
    self.mat = dataclasses.replace(self.mat, **fields)  # pyrefly: ignore[bad-argument-type]

  def read(self, name: str) -> tuple[UsdValue, UvTexture | None]:
    """(value, None) of an input, (None, texture) when a UsdUVTexture feeds it, else (None,
    None)."""
    inp = self.shader.GetInput(name)
    if not inp:
      return None, None
    v, attr = util.producing(inp)
    if attr is not None and attr.GetName().startswith("outputs:"):
      return None, texture_node(self.ctx, attr)
    return v, None

  def primvar_reader(self, name: str) -> str | None:
    """The primvar read by a UsdPrimvarReader_float3 / float4 connected to input `name`."""
    inp = self.shader.GetInput(name)
    attr = util.connected_output(inp) if inp else None
    if attr is None:
      return None
    if str(UsdShade.Shader(attr.GetPrim()).GetIdAttr().Get()) not in ("UsdPrimvarReader_float3", "UsdPrimvarReader_float4"):
      return None
    return _varname(util.shader_inputs(attr.GetPrim()))

  def uv_index(self, varname: str) -> int:
    """Index of primvar `varname` among the UV primvars the material reads (added if new)."""
    if varname not in self.uv_names:
      self.uv_names.append(varname)
    return self.uv_names.index(varname)

  def uv_set(self, varname: str) -> int:
    """Mesh UV set of primvar `varname` (meshes keep two sets: a third reads set 0, reported)."""
    uv_set = self.uv_index(varname)
    if uv_set > 1:
      self.ctx.report.drop(f"UsdPreviewSurface.uv_set{uv_set}(set 0 used)", self.prim)
      uv_set = 0
    return uv_set

  def bind(self, slot: Slot, tex: UvTexture, *, normal: bool = False) -> bool:
    """Binds `slot` to the texture; False when its file does not resolve (use its fallback).

    `normal`: a normal map, read through its scale / bias as authored (any per-axis strength or
    flip: `raywarp.scene.model.normal_map`); unauthored ones default to [0, 1] -> [-1, 1], and one
    whose z is never positive is dropped (reported; the fallback, the geometry's normal, is used).
    """
    scale, bias = tex.scale, tex.bias
    implicit = normal and scale == (1.0, 1.0, 1.0, 1.0) and bias == (0.0, 0.0, 0.0, 0.0)
    if implicit:
      scale, bias = normal_map.NORMAL_UNPACK
    if normal and not normal_map.outward(scale, bias):  # before the texture is registered
      self.ctx.report.drop("UsdPreviewSurface.normal.inward(z never positive: no normal map)", self.prim)
      return False
    tex_id = self.ctx.texture(tex.path, slot, tex.colorspace, tex.wrap, image=tex.atlas)
    if tex_id < 0:
      return False
    if implicit:
      self.ctx.report.approx("UsdPreviewSurface.normal.implicit_scale_bias", self.prim)
    self.bound[slot] = tex
    ref = TextureRef(
      tex_id, channel=tex.channel, scale=scale, bias=bias, uv_xform=uv_rows(tex.uv), uv_set=self.uv_set(tex.varname)
    )
    self.mat = self.mat.with_texture(slot, ref)
    return True

  def colors_and_scalars(self) -> str | None:
    """Base color, emission, metallic, roughness, opacity, occlusion, normal, clearcoat, ior.

    A connected texture replaces the value (factor 1); a primvar reader on diffuseColor
    (displayColor, a glTF COLOR_0) makes the vertex colors the base color (returned).
    """
    color_primvar = self.primvar_reader("diffuseColor")
    v, tex = (None, None) if color_primvar else self.read("diffuseColor")
    if color_primvar:
      self.set(base_color=(1.0, 1.0, 1.0))
      self.mat = with_flags(self.mat, MaterialFlags.VERTEX_COLOR)
    elif tex is not None:
      self.set(base_color=(1.0, 1.0, 1.0) if self.bind(Slot.BASE_COLOR, tex) else tex.fallback_value(3))
    elif v is not None:
      self.set(base_color=util.vec3(v))
    v, tex = self.read("emissiveColor")
    if tex is not None:
      self.set(emission_color=(1.0, 1.0, 1.0) if self.bind(Slot.EMISSION_COLOR, tex) else tex.fallback_value(3))
    elif v is not None:
      self.set(emission_color=util.vec3(v))
    for name, slot, field in _SCALARS:
      v, tex = self.read(name)
      if tex is not None:
        self.set(**{field: 1.0 if self.bind(slot, tex) else tex.fallback_value(1)[0]})
      elif v is not None:
        self.set(**{field: util.as_float(v)})
    for name, slot in (("occlusion", Slot.OCCLUSION), ("normal", Slot.GEOMETRY_NORMAL)):
      _, tex = self.read(name)
      if tex is not None:
        self.bind(slot, tex, normal=slot == Slot.GEOMETRY_NORMAL)
    for name, field in (("clearcoat", "coat_weight"), ("clearcoatRoughness", "coat_roughness"), ("ior", "specular_ior")):
      v, tex = self.read(name)
      if tex is not None:
        self.ctx.report.drop(f"UsdPreviewSurface.{name}.texture", self.prim)
      elif v is not None:
        self.set(**{field: util.as_float(v)})
    return color_primvar

  def specular_workflow(self) -> None:
    """useSpecularWorkflow: the metallic-roughness equivalent (baked per texel when textured)."""
    v, tex = self.read("specularColor")
    report, prim = self.ctx.report, self.prim
    self.set(base_metalness=0.0)
    self.mat = self.mat.with_texture(Slot.BASE_METALNESS, None)
    specular = (0.0, 0.0, 0.0) if v is None else util.vec3(v)  # UsdPreviewSurface default
    diffuse_tex = self.bound.get(Slot.BASE_COLOR) if self.mat.texture(Slot.BASE_COLOR) is not None else None
    diffuse = util.vec3(self.mat.base_color)
    baked = None
    if tex is not None or diffuse_tex is not None:
      # Textured: the conversion per texel, baked into base color and metallic textures.
      baked = bake_specular_workflow(self.ctx, prim, (diffuse_tex, diffuse), (tex, specular))
    if baked is not None:
      self._bind_baked(*baked, diffuse_textured=diffuse_tex is not None)
      report.approx("UsdPreviewSurface.useSpecularWorkflow(baked: dielectric F0 0.04)", prim)
      return
    if tex is not None:
      report.drop("UsdPreviewSurface.specularColor.texture", prim)
    elif self.mat.texture(Slot.BASE_COLOR) is None:
      # Constant diffuse / specular colors: the metallic-roughness equivalent (glTF's
      # KHR_materials_pbrSpecularGlossiness conversion), e.g. a gold specularColor is a metal.
      metallic, base = specular_to_metallic(diffuse, specular)
      # F0 comes from specularColor, not from the ior.
      self.set(base_metalness=metallic, base_color=base, specular_ior=1.5)
      if metallic == 0.0:
        # A dielectric: the diffuse color as authored, F0 = specularColor = weight x 0.04 x tint.
        peak = max(specular)
        tint = (1.0, 1.0, 1.0) if peak <= 0.0 else (specular[0] / peak, specular[1] / peak, specular[2] / peak)
        self.set(base_color=diffuse, specular_weight=peak / _DIELECTRIC_F0, specular_color=tint)
    elif diffuse_tex is not None:
      self.set(specular_color=specular)
    report.approx("UsdPreviewSurface.useSpecularWorkflow", prim)

  def _bind_baked(self, base_tex: TextureId, metal_tex: TextureId, ref: UvTexture, *, diffuse_textured: bool) -> None:
    """Binds the baked base color and metallic textures (`bake_specular_workflow`) on the UV set of
    the diffuse texture, else of `ref`."""
    base_ref = self.mat.texture(Slot.BASE_COLOR)
    uv_set = base_ref.uv_set if diffuse_textured and base_ref is not None else self.uv_set(ref.varname)
    xf = uv_rows(ref.uv)
    self.mat = self.mat.with_texture(Slot.BASE_COLOR, TextureRef(base_tex, channel=CHANNEL_RGB, uv_xform=xf, uv_set=uv_set))
    self.mat = self.mat.with_texture(Slot.BASE_METALNESS, TextureRef(metal_tex, channel=0, uv_xform=xf, uv_set=uv_set))
    self.set(base_color=(1.0, 1.0, 1.0), base_metalness=1.0, specular_ior=1.5, specular_weight=1.0)

  def opacity(self) -> None:
    """Alpha mode from opacity, opacityThreshold and opacityMode.

    opacityThreshold > 0: a cut-out in either mode (opaque where opacity >= threshold). Otherwise
    opacity < 1 is "presence" (coverage: every lobe attenuated, the core's blending) or
    "transparent" (the default: glass-like, the specular reflection kept at full strength),
    which the core can only render as coverage.
    """
    threshold, _ = self.read("opacityThreshold")
    mode_value, _ = self.read("opacityMode")
    mode = "transparent" if mode_value is None else str(mode_value)
    if mode not in ("transparent", "presence"):
      self.ctx.report.drop(f"UsdPreviewSurface.opacityMode.{mode}(transparent used)", self.prim)
      mode = "transparent"
    if threshold is not None and util.as_float(threshold) > 0.0:
      self.mat = set_ext(self.mat, alpha_mode=AlphaMode.MASK, alpha_cutoff=util.as_float(threshold))
    elif self.mat.geometry_opacity < 1.0 or self.mat.texture(Slot.GEOMETRY_OPACITY) is not None:
      self.mat = set_ext(self.mat, alpha_mode=AlphaMode.BLEND)
      if mode == "transparent":
        self.ctx.report.approx("UsdPreviewSurface.opacityMode.transparent(as presence: specular attenuated too)", self.prim)

  def displacement(self) -> None:
    """Reads scalar displacement, even when the material omits its displacement terminal."""
    value, tex = self.read("displacement")
    amount: float | TextureRef = 0.0
    if tex is not None:
      if tex.channel < 0:
        self.ctx.report.drop("UsdPreviewSurface.displacement.non_scalar_texture", self.prim)
        return
      if tex.scale[0] == 0.0:
        amount = tex.bias[0]
      else:
        texture = self.ctx.builder.add_texture(
          tex.path if tex.atlas is None else tex.atlas,
          colorspace=tex.colorspace or ColorSpace.DATA,
          wrap=tex.wrap,
          max_size=self.ctx.max_texture_size,
        )
        if texture < 0:
          amount = tex.scale[0] * tex.fallback_value(1)[0] + tex.bias[0]
        else:
          amount = TextureRef(
            texture,
            channel=tex.channel,
            scale=tex.scale,
            bias=tex.bias,
            uv_xform=uv_rows(tex.uv),
            uv_set=self.uv_set(tex.varname),
          )
    elif isinstance(value, (int, float)):
      amount = float(value)
    if amount != 0.0:
      self.mat = dataclasses.replace(self.mat, displacement=Displacement(amount=amount))

  def report_unread(self) -> None:
    """Reports every authored input UsdPreviewSurface does not define."""
    for inp in self.shader.GetInputs():
      name = inp.GetBaseName()
      if name not in _PREVIEW_INPUTS and (inp.GetAttr().HasAuthoredValue() or inp.HasConnectedSource()):
        self.ctx.report.drop(f"UsdPreviewSurface.{name}", self.prim)


def preview_surface(ctx: ShadeContext, shader: UsdShade.Shader) -> tuple[Material, MeshNeeds]:
  """UsdPreviewSurface -> Material (values, or textures with their scale / bias / transform)."""
  p = _Preview(ctx, shader)
  color_primvar = p.colors_and_scalars()
  use_specular, _ = p.read("useSpecularWorkflow")
  if use_specular:
    p.specular_workflow()
  p.opacity()
  p.displacement()
  p.report_unread()
  # Out-of-range constants are clamped and reported, as on the MaterialX path: the asset loads.
  mat = openpbr.clamp_material(p.mat, report=ctx.report, where=p.prim)
  return gltf_coat(mat, base_normal=True), MeshNeeds(tuple(p.uv_names) or ("st",), color_primvar)
