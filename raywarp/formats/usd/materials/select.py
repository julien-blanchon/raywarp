"""UsdShade.Material -> core material: the surface shader the render contexts pick (USD's rule,
`Options.material_contexts`), converted by `preview.py` (UsdPreviewSurface) or the context's
`MaterialBaker` (MaterialX and MDL), cached per material in the `ShadeContext` (`context.py`).
"""

from __future__ import annotations

import dataclasses

from pxr import UsdShade

from raywarp.formats.usd.materials.context import (
  DISPLAY_EMISSION_ATTR,
  FALLBACK,
  MTLX_UNLIT,
  UNLIT_ATTR,
  VERTEX_COLOR_ATTR,
  MeshNeeds,
  ShadeContext,
  with_flags,
)
from raywarp.formats.usd.materials.preview import preview_surface
from raywarp.scene.model.material import Material
from raywarp.scene.model.types import MaterialFlags, MaterialId


def _is_mdl(shader: UsdShade.Shader) -> bool:
  """Whether a shader is an MDL material the MDL baker converts: its implementation is a `.mdl`
  source asset (`info:mdl:sourceAsset`, or a universal `info:sourceAsset`, which USD returns for
  every source type: an OSL or GLSL file there is not MDL)."""
  if shader.GetImplementationSource() != "sourceAsset":
    return False
  asset = shader.GetSourceAsset("mdl")
  return asset is not None and asset.path.lower().endswith(".mdl")


def is_materialx(shader: UsdShade.Shader) -> bool:
  """Whether `shader` is a MaterialX surface shader (`materialx.is_surface`; the MaterialX library
  is imported only for an `ND_` id: a stage without MaterialX never loads it)."""
  if not str(shader.GetIdAttr().Get() or "").startswith("ND_"):
    return False
  from raywarp.formats.usd.materials.materialx import is_surface  # noqa: PLC0415  # MaterialX only for MaterialX networks

  return is_surface(shader)


def _maps(shader: UsdShade.Shader, context: str, contexts: tuple[str, ...]) -> bool:
  """Whether the converters map `shader`, connected to the surface output of render context
  `context` ("": universal): "mtlx" a MaterialX surface shader, "mdl" an MDL material; universal
  a UsdPreviewSurface, a MaterialX surface shader, or an MDL material when `contexts` names
  "mdl"."""
  if context == "mtlx":
    return is_materialx(shader)
  if context == "mdl":
    return _is_mdl(shader)
  preview = shader.GetIdAttr().Get() == "UsdPreviewSurface"
  return preview or is_materialx(shader) or ("mdl" in contexts and _is_mdl(shader))


def material(ctx: ShadeContext, material: UsdShade.Material) -> tuple[MaterialId, MeshNeeds]:
  """(material id, mesh attributes it reads) of a bound UsdShade.Material, cached by path."""
  key = str(material.GetPath())
  if key not in ctx.materials:
    ctx.report.count("formats.usd.materials.select")
    mat, needs = _convert(ctx, material)
    if mat.displacement is not None:
      terminal = material.GetDisplacementOutput()
      if not terminal or not terminal.HasConnectedSource():
        ctx.report.approx("UsdPreviewSurface.displacement.no_terminal", material.GetPrim())
    attr = material.GetPrim().GetAttribute(VERTEX_COLOR_ATTR)
    primvar = attr.Get() if attr and attr.HasAuthoredValue() else None
    if primvar:
      mat = with_flags(mat, MaterialFlags.VERTEX_COLOR)
      needs = dataclasses.replace(needs, color=str(primvar))
    for name, flag in ((UNLIT_ATTR, MaterialFlags.UNLIT), (DISPLAY_EMISSION_ATTR, MaterialFlags.DISPLAY_EMISSION)):
      attr = material.GetPrim().GetAttribute(name)
      if attr and attr.HasAuthoredValue() and bool(attr.Get()):
        mat = with_flags(mat, flag)
    ctx.materials[key] = (ctx.builder.add_material(mat), needs)
  return ctx.materials[key]


def _surface(material: UsdShade.Material, context: str) -> UsdShade.Shader | None:
  """The shader whose output feeds the material's surface output of render context `context`,
  through any NodeGraph outputs between them (USD's value-producing attribute)."""
  output = material.GetSurfaceOutput(context)
  if not (output and output.HasConnectedSource()):
    return None
  producers = UsdShade.Utils.GetValueProducingAttributes(output)
  prim = producers[0].GetPrim() if producers else None
  return UsdShade.Shader(prim) if prim is not None and prim.IsA(UsdShade.Shader) else None


def _pick(ctx: ShadeContext, material: UsdShade.Material) -> UsdShade.Shader | None:
  """The surface shader converted, by USD's render-context rule: the first of `contexts`
  whose surface output connects a shader it maps (`_maps`), else the universal output's.
  Without one: the first connected surface shader (converted as unknown: reported), else the
  first shader child that is a surface the universal context maps (never a texture or
  primvar reader), else None."""
  connected = [(c, _surface(material, c)) for c in (*ctx.contexts, "")]
  for context, shader in connected:
    if shader is not None and _maps(shader, context, ctx.contexts):
      return shader
  unknown = next((shader for _, shader in connected if shader is not None), None)
  if unknown is not None:
    return unknown
  children = (UsdShade.Shader(c) for c in material.GetPrim().GetChildren() if c.IsA(UsdShade.Shader))
  return next((c for c in children if _maps(c, "", ctx.contexts)), None)


def baked_surface(ctx: ShadeContext, material: UsdShade.Material) -> UsdShade.Shader | None:
  """The shader `_convert` hands the MDL baker for `material`: a MaterialX surface shader or an
  MDL shader (None: another surface wins)."""
  shader = _pick(ctx, material)
  if shader is None or shader.GetIdAttr().Get() == "UsdPreviewSurface":
    return None
  return shader if is_materialx(shader) or ("mdl" in ctx.contexts and _is_mdl(shader)) else None


def _convert(ctx: ShadeContext, material: UsdShade.Material) -> tuple[Material, MeshNeeds]:
  """Material of a UsdShade.Material: UsdPreviewSurface, else a MaterialX surface or an MDL
  shader (`ShadeContext.baker`)."""
  name = str(material.GetPath())
  shader = _pick(ctx, material)
  if shader is None:
    ctx.report.drop("material.no_surface", material.GetPrim())
    return dataclasses.replace(FALLBACK, name=name), MeshNeeds()
  if shader.GetIdAttr().Get() == "UsdPreviewSurface":
    mat, needs = preview_surface(ctx, shader)
  elif is_materialx(shader):
    from raywarp.formats.usd.materials.materialx import extensions  # noqa: PLC0415  # MaterialX only for MaterialX networks

    mat, needs = ctx.baker.convert(ctx, shader)
    mat = extensions(ctx, shader, mat)
    if str(shader.GetIdAttr().Get() or "").startswith(MTLX_UNLIT):
      # Unlit: the distilled emission is its color, nothing else shows (no base to light).
      mat = with_flags(dataclasses.replace(mat, base_color=(0.0, 0.0, 0.0)), MaterialFlags.UNLIT)
  elif "mdl" in ctx.contexts and _is_mdl(shader):
    mat, needs = ctx.baker.convert(ctx, shader)
  else:
    kind = shader.GetIdAttr().Get() or "/".join(shader.GetSourceTypes()) or "?"
    ctx.report.drop(f"material.unknown_shader:{kind}", shader.GetPrim())
    mat, needs = FALLBACK, MeshNeeds()
  return dataclasses.replace(mat, name=name), needs
