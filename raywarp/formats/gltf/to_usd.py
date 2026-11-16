"""A glTF 2.0 scene (.gltf / .glb) as a USD stage with UsdPreviewSurface and UsdLux (`convert`).

What `rw.formats.gltf.load` reads through the USD importer, and what other renderers of a
comparison read (the same stage file):

  meshes            one UsdGeom.Mesh per triangle primitive (points, vertex normals, every
                    TEXCOORD_n as primvars:st / st1 ..., V flipped to USD's bottom-left origin),
                    `doubleSided` from the material. A mesh used by one node is written under the
                    node's Xform; a mesh used by several nodes becomes one PointInstancer (its
                    world transforms as positions / orientations / scales).
  nodes             Xforms with the node matrix; the stage stays Y-up, metres (USD's up axis).
  materials         UsdPreviewSurface: base color (+ alpha), metallic (B) / roughness (G),
                    normal (scale folded into the UsdUVTexture scale / bias), occlusion (R),
                    emissive (x KHR_materials_emissive_strength; display-relative: the
                    material's `raywarp:displayEmission`,
                    `formats.usd.materials.context.DISPLAY_EMISSION_ATTR`), alpha mode (MASK ->
                    opacityThreshold, BLEND -> opacity), KHR_texture_transform (UsdTransform2d),
                    KHR_materials_ior, KHR_materials_clearcoat (factors only),
                    KHR_materials_pbrSpecularGlossiness (specular workflow: specularColor =
                    specular RGB, roughness = 1 - glossiness from the alpha channel; raywarp's
                    loader bakes the per-texel metallic-roughness conversion),
                    KHR_materials_transmission (approximated: opacity 1 - transmission),
                    KHR_materials_unlit (the material's `raywarp:unlit` attribute,
                    `formats.usd.materials.context.UNLIT_ATTR`: base color display-relative, no
                    light). What each value means in raywarp's canonical OpenPBR material, end to
                    end: the glTF table of the canonical material design
                    (`raywarp.formats.usd.materials.preview` reads the UsdPreviewSurface; MaterialX
                    `gltf_pbr` nodes go through MaterialX's own library,
                    `raywarp.formats.usd.materials.materialx`).
  lights            KHR_lights_punctual: point -> SphereLight, spot -> SphereLight + ShapingAPI
                    (cone = outer angle, softness = 1 - inner / outer), directional ->
                    DistantLight; `normalize` on, so the intensity is the glTF candela / lux.
  vertex colors     COLOR_0 -> primvars:displayColor (+ displayOpacity), multiplied into the base
                    color through the material's `raywarp:vertexColor` attribute
                    (`formats.usd.materials.context.VERTEX_COLOR_ATTR`: UsdPreviewSurface has no
                    product node; other renderers show them only on unbound / untextured meshes).
  cameras           perspective cameras -> UsdGeom.Camera (orthographic ones are skipped).
  images            copied under textures/ next to the stage (a self-contained folder); embedded
                    images and formats outside PNG / JPEG are decoded to PNG. A texture whose
                    `source` file is missing uses its MSFT_texture_dds image (decoded to PNG).

Not converted: primitives other than triangles (points, lines, strips, fans; counted as
`skipped_primitives`), sheen, iridescence, volume, anisotropy, specular (KHR_materials_specular),
variants, skins, morph targets, animations, sparse accessors, a light's `range`.
"""

from __future__ import annotations

import collections
import concurrent.futures
import io
import math
import os
import shutil
import urllib.parse
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final, cast, final

import numpy as np
import pygltflib
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from raywarp.formats.gltf.reader import TRIANGLES, Gltf
from raywarp.formats.gltf.values import Object, TextureInfo, num, nums, obj
from raywarp.formats.usd.materials.context import DISPLAY_EMISSION_ATTR, UNLIT_ATTR, VERTEX_COLOR_ATTR

# glTF sampler wrap modes (glTF 2.0 §5.26) -> UsdUVTexture wrap tokens; REPEAT by default.
_WRAPS: Final = {33071: "clamp", 33648: "mirror", 10497: "repeat"}


def _index(value: int | None, what: str) -> int:
  """A required glTF index.

  Raises:
    ValueError: when it is missing.
  """
  if value is None:
    raise ValueError(f"glTF: {what} has no index")
  return value


def _texture_info(o: Object) -> pygltflib.TextureInfo:
  """A textureInfo given as a JSON object (an extension's).

  Raises:
    ValueError: without its texture index.
  """
  if "index" not in o:
    raise ValueError(f"glTF: textureInfo without an index: {o!r}")
  return pygltflib.TextureInfo(
    index=int(num(o, "index", 0)), texCoord=int(num(o, "texCoord", 0)), extensions=dict(obj(o, "extensions"))
  )


# Image files copied as they are (other formats are decoded to PNG).
_DIRECT_IMAGE: Final = (".png", ".jpg", ".jpeg")
# Point / spot lights: a small sphere [m] (glTF lights are points; `treatAsPoint` is set).
_LIGHT_RADIUS: Final = 0.02
# glTF's default sampler wrap (REPEAT, glTF 2.0 §5.26).
_REPEAT: Final = 10497
# A camera without an aspect ratio (the viewport's in glTF): 3:2, the 36 x 24 mm film's.
_ASPECT: Final = 1.5
# The vertical aperture of converted cameras [mm] (only its ratio to the focal length matters).
_APERTURE_MM: Final = 24.0
# glTF's infinite far plane (`zfar` absent) as a finite clipping range [m].
_FAR_M: Final = 10000.0
_NEAR_M: Final = 0.1

type Matrix4 = np.ndarray[[4, 4], np.dtype[np.float64]]
type Vec3 = np.ndarray[[3], np.dtype[np.float64]]
type Quat = np.ndarray[[4], np.dtype[np.float64]]


def _name(prefix: str, index: int, name: str | None) -> str:
  """A valid, unique prim name."""
  clean = "".join(c if c.isalnum() or c == "_" else "_" for c in (name or ""))[:48]
  return f"{prefix}{index}" + (f"_{clean}" if clean else "")


def _decompose(m: Matrix4) -> tuple[Vec3, Quat, Vec3]:
  """(translation, quaternion w x y z, scale) of a (4, 4) column-vector matrix (shear dropped)."""
  t = m[:3, 3].copy()
  a = m[:3, :3]
  s = np.linalg.norm(a, axis=0)
  if np.linalg.det(a) < 0:
    s[0] = -s[0]
  r = a / np.where(np.abs(s) > 1e-30, s, 1.0)
  # Gf matrices are row-vector: pass the transpose.
  q = Gf.Matrix4d(Gf.Matrix3d(*r.T.flatten().tolist()), Gf.Vec3d(0.0)).ExtractRotationQuat()
  return t, np.array([q.GetReal(), *q.GetImaginary()]), s


@final
class _Converter:
  """Writes one glTF document as a new USD stage (`run`)."""

  def __init__(self, gltf: Gltf, out: Path) -> None:
    """A converter of `gltf` into a new stage file `out` (Y-up, metres)."""
    self.gltf = gltf
    self.doc = gltf.doc
    self.out = out
    self.stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
    self.world = UsdGeom.Xform.Define(self.stage, "/World")
    self.stage.SetDefaultPrim(self.world.GetPrim())
    UsdGeom.Scope.Define(self.stage, "/World/Looks")
    self.materials: dict[int, UsdShade.Material] = {}
    self.images: dict[int, str] = {}
    self.pending: dict[Path, Path | int] = {}  # image file -> its source file or glTF image
    self.uses: dict[int, list[Matrix4]] = {}  # mesh -> world transforms of its instances
    self.instanced: set[int] = set()  # meshes used by several nodes (`instancers`)
    self.counts = dict.fromkeys(
      ("meshes", "triangles", "skipped_primitives", "instancers", "instances", "lights", "cameras", "materials", "images"), 0
    )

  # Images and materials --------------------------------------------------------------------

  def image(self, texture_index: int) -> str:
    """Asset path (relative to the stage) of a glTF texture's image, under textures/ next to the
    stage (the stage folder is self-contained). External PNG / JPEG files are copied; anything
    else (embedded, another format, or a missing file whose MSFT_texture_dds twin exists) is
    decoded to PNG. Both are queued for `write_images`.
    """
    texture = self.doc.textures[texture_index]
    source = _index(texture.source, f"texture {texture_index}'s image")
    if source in self.images:
      return self.images[source]
    src, name = self._image_source(texture, source)
    parts = PurePosixPath(name).parts
    if not parts or any(p in ("..", "/") for p in parts):  # a URI leaving the asset's folder
      name = f"image{source}{PurePosixPath(name).suffix or '.png'}"
    rel = PurePosixPath("textures", name)  # PurePosixPath drops "." parts
    # Two images can map to one name (`a.png` and a decoded `a.tga`, an embedded image and a file
    # named `image<n>.png`): the later one gets its own.
    while self.pending.get(self.out.parent / rel, src) != src:
      rel = rel.with_name(f"{rel.stem}_image{source}{rel.suffix}")
    self.pending[self.out.parent / rel] = src
    self.images[source] = str(rel)
    return str(rel)

  def _image_source(self, texture: pygltflib.Texture, source: int) -> tuple[Path | int, str]:
    """(what image `source` is written from: a file, or a glTF image index whose bytes are
    decoded; its file name under textures/). A missing file is replaced by the texture's
    MSFT_texture_dds image when it has one."""
    raw = self.doc.images[source].uri
    uri, path = urllib.parse.unquote(raw or ""), self.gltf.uri_path(raw)
    if path is None:  # embedded
      return source, f"image{source}.png"
    png = Path(uri).with_suffix(".png").as_posix()
    if path.exists():
      return path, Path(uri).as_posix() if uri.lower().endswith(_DIRECT_IMAGE) else png
    dds = obj(texture.extensions or {}, "MSFT_texture_dds")
    if dds:
      dds_source = int(num(dds, "source", -1))
      dds_path = self.gltf.uri_path(self.doc.images[dds_source].uri)
      return (dds_source if dds_path is None else dds_path), png
    return path, png  # missing: the write raises FileNotFoundError

  def write_images(self) -> None:
    """Copies or decodes the queued images (in parallel)."""

    def one(item: tuple[Path, Path | int]) -> None:
      """Writes queued image `item` (destination, source file or glTF image) once, atomically."""
      dst, src = item
      if dst.exists():
        return
      dst.parent.mkdir(parents=True, exist_ok=True)
      tmp = dst.with_name(dst.name + ".part")
      if isinstance(src, Path) and src.suffix.lower() == dst.suffix.lower():
        shutil.copyfile(src, tmp)
      else:
        if isinstance(src, Path):
          img = Image.open(src)
        else:
          img = Image.open(io.BytesIO(self.gltf.image_bytes(src)))
        img.save(tmp, format="PNG", compress_level=4)
      tmp.rename(dst)

    with concurrent.futures.ThreadPoolExecutor(min(32, os.cpu_count() or 4)) as pool:
      list(pool.map(one, self.pending.items()))
    self.counts["images"] = len(self.pending)

  def _texture(  # noqa: PLR0913  # the texture's material, name, info and keyword-only options
    self,
    mat: UsdShade.Material,
    name: str,
    info: TextureInfo,
    colorspace: str,
    *,
    scale: list[float] | None = None,
    bias: list[float] | None = None,
  ) -> UsdShade.Shader:
    """A UsdUVTexture node (with its primvar reader and KHR_texture_transform) for a textureInfo."""
    index = _index(info.index, f"the {name} textureInfo")
    base = mat.GetPath().AppendChild(name)
    tex = UsdShade.Shader.Define(self.stage, base)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(self.image(index))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
    texture = self.doc.textures[index]
    sampler = self.doc.samplers[texture.sampler] if texture.sampler is not None else pygltflib.Sampler()
    for token, wrap in (("wrapS", sampler.wrapS), ("wrapT", sampler.wrapT)):
      tex.CreateInput(token, Sdf.ValueTypeNames.Token).Set(_WRAPS.get(wrap if wrap is not None else _REPEAT, "repeat"))
    if scale is not None:
      tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
    if bias is not None:
      tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
    reader = UsdShade.Shader.Define(self.stage, base.GetParentPath().AppendChild(f"{name}_uv"))
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_set = info.texCoord or 0
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st" if uv_set == 0 else f"st{uv_set}")
    st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    transform = obj(info.extensions or {}, "KHR_texture_transform")
    if transform:
      st = self._uv_transform(mat, f"{name}_xf", transform, st)
      if transform.get("texCoord") is not None:
        uv_set = int(num(transform, "texCoord", 0))
        reader.GetInput("varname").Set("st" if uv_set == 0 else f"st{uv_set}")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
    return tex

  def _uv_transform(self, mat: UsdShade.Material, name: str, t: Object, st: UsdShade.Output) -> UsdShade.Output:
    """KHR_texture_transform (in glTF's top-left UV space) as a UsdTransform2d on flipped st."""
    ox, oy = nums(t, "offset", (0.0, 0.0))
    rot = num(t, "rotation", 0.0)
    sx, sy = nums(t, "scale", (1.0, 1.0))
    c, s = math.cos(rot), math.sin(rot)
    # glTF: uv' = T R S uv with R = [[c, s], [-s, c]]. USD st = F uv, F: (u, v) -> (u, 1 - v), so
    # the USD transform is F (T R S) F: a rotation by +rot, the same scale, a new translation.
    a = np.array([[c * sx, s * sy, ox], [-s * sx, c * sy, oy], [0, 0, 1.0]])
    f = np.array([[1.0, 0, 0], [0, -1.0, 1.0], [0, 0, 1.0]])
    m = f @ a @ f
    tx, ty = m[0, 2], m[1, 2]
    rot_deg = math.degrees(rot)
    node = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild(name))
    node.CreateIdAttr("UsdTransform2d")
    node.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
    node.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(rot_deg)
    node.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(sx), float(sy)))
    node.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(tx), float(ty)))
    return node.CreateOutput("result", Sdf.ValueTypeNames.Float2)

  def material(self, index: int | None) -> UsdShade.Material | None:
    """The UsdShade material of glTF material `index` (None: none), made once."""
    if index is None:
      return None
    if index in self.materials:
      return self.materials[index]
    m = self.doc.materials[index]
    ext = m.extensions or {}
    mat = UsdShade.Material.Define(self.stage, f"/World/Looks/{_name('mat', index, m.name)}")
    surf = UsdShade.Shader.Define(self.stage, mat.GetPath().AppendChild("surface"))
    surf.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    alpha_mode = m.alphaMode or "OPAQUE"
    self._base_color(mat, surf, m, alpha_mode)
    sg = obj(ext, "KHR_materials_pbrSpecularGlossiness")
    if sg:
      self._specular_glossiness(mat, surf, sg)
    else:
      self._metallic_roughness(mat, surf, m.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness())
    self._maps(mat, surf, m)
    self._extensions(surf, ext, alpha_mode)
    if "KHR_materials_unlit" in ext:  # the base color shown as is (display-relative), unlit
      mat.GetPrim().CreateAttribute(UNLIT_ATTR, Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    self.materials[index] = mat
    self.counts["materials"] += 1
    return mat

  def _base_color(self, mat: UsdShade.Material, surf: UsdShade.Shader, m: pygltflib.Material, alpha_mode: str) -> None:
    """The base color and opacity (texture x factor, or the factor) and the MASK cutoff."""
    sg = obj(m.extensions or {}, "KHR_materials_pbrSpecularGlossiness")
    if sg:
      diffuse = nums(sg, "diffuseFactor", (1.0, 1.0, 1.0, 1.0))
      base_info = _texture_info(obj(sg, "diffuseTexture")) if "diffuseTexture" in sg else None
    else:
      pbr = m.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness()
      diffuse = list(pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0])
      base_info = pbr.baseColorTexture
    color = surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    opacity = surf.CreateInput("opacity", Sdf.ValueTypeNames.Float) if alpha_mode != "OPAQUE" else None
    if base_info is not None:
      tex = self._texture(mat, "baseColor", base_info, "sRGB", scale=diffuse)
      color.ConnectToSource(tex.ConnectableAPI(), "rgb")
      if opacity is not None:
        opacity.ConnectToSource(tex.ConnectableAPI(), "a")
    else:
      color.Set(Gf.Vec3f(*diffuse[:3]))
      if opacity is not None:
        opacity.Set(float(diffuse[3]))
    if alpha_mode == "MASK":
      cutoff = float(m.alphaCutoff if m.alphaCutoff is not None else 0.5)
      surf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(cutoff)

  def _specular_glossiness(self, mat: UsdShade.Material, surf: UsdShade.Shader, sg: Object) -> None:
    """KHR_materials_pbrSpecularGlossiness: the specular workflow, roughness = 1 - glossiness."""
    surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    spec = nums(sg, "specularFactor", (1.0, 1.0, 1.0))
    gloss = num(sg, "glossinessFactor", 1.0)
    color = surf.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f)
    roughness = surf.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    if "specularGlossinessTexture" in sg:
      info = _texture_info(obj(sg, "specularGlossinessTexture"))
      tex = self._texture(mat, "specGloss", info, "sRGB", scale=[*spec, -gloss], bias=[0.0, 0.0, 0.0, 1.0])
      color.ConnectToSource(tex.ConnectableAPI(), "rgb")
      roughness.ConnectToSource(tex.ConnectableAPI(), "a")
    else:
      color.Set(Gf.Vec3f(*spec))
      roughness.Set(1.0 - gloss)

  def _metallic_roughness(self, mat: UsdShade.Material, surf: UsdShade.Shader, pbr: pygltflib.PbrMetallicRoughness) -> None:
    """Metalness (B) / roughness (G) of the metallicRoughness texture x factors, or the factors."""
    metallic = 1.0 if pbr.metallicFactor is None else float(pbr.metallicFactor)
    roughness = 1.0 if pbr.roughnessFactor is None else float(pbr.roughnessFactor)
    m_in = surf.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    r_in = surf.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    if pbr.metallicRoughnessTexture is not None:
      scale = [1.0, roughness, metallic, 1.0]
      tex = self._texture(mat, "metallicRoughness", pbr.metallicRoughnessTexture, "raw", scale=scale)
      m_in.ConnectToSource(tex.ConnectableAPI(), "b")
      r_in.ConnectToSource(tex.ConnectableAPI(), "g")
    else:
      m_in.Set(metallic)
      r_in.Set(roughness)

  def _maps(self, mat: UsdShade.Material, surf: UsdShade.Shader, m: pygltflib.Material) -> None:
    """Normal (scale folded into the texture's scale / bias), occlusion (R) and emission
    (x KHR_materials_emissive_strength)."""
    if m.normalTexture is not None:
      k = float(m.normalTexture.scale if m.normalTexture.scale is not None else 1.0)
      tex = self._texture(mat, "normal", m.normalTexture, "raw", scale=[2 * k, 2 * k, 2.0, 1.0], bias=[-k, -k, -1.0, 0.0])
      surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    if m.occlusionTexture is not None:
      # glTF's occlusion strength s: 1 + s (ao - 1), the texture's scale / bias.
      s = float(m.occlusionTexture.strength if m.occlusionTexture.strength is not None else 1.0)
      strength = {} if s == 1.0 else {"scale": [s, s, s, 1.0], "bias": [1.0 - s, 1.0 - s, 1.0 - s, 0.0]}
      tex = self._texture(mat, "occlusion", m.occlusionTexture, "raw", **strength)
      surf.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "r")
    emissive = np.array(m.emissiveFactor or [0.0, 0.0, 0.0], np.float64)
    emissive *= num(obj(m.extensions or {}, "KHR_materials_emissive_strength"), "emissiveStrength", 1.0)
    if not emissive.any():
      return
    # glTF's emission is display-referred (1: the brightest LDR value), UsdPreviewSurface's a
    # radiance in nits: the material is marked (`context.DISPLAY_EMISSION_ATTR`).
    mat.GetPrim().CreateAttribute(DISPLAY_EMISSION_ATTR, Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    color = surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    if m.emissiveTexture is not None:
      tex = self._texture(mat, "emissive", m.emissiveTexture, "sRGB", scale=[*emissive.tolist(), 1.0])
      color.ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
      color.Set(Gf.Vec3f(*emissive.tolist()))

  @staticmethod
  def _extensions(surf: UsdShade.Shader, ext: Object, alpha_mode: str) -> None:
    """KHR_materials_ior, clearcoat (factors) and transmission (approximated: opacity 1 - t)."""
    if "KHR_materials_ior" in ext:
      surf.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(num(obj(ext, "KHR_materials_ior"), "ior", 1.5))
    if "KHR_materials_clearcoat" in ext:
      cc = obj(ext, "KHR_materials_clearcoat")
      surf.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(num(cc, "clearcoatFactor", 0.0))
      surf.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(num(cc, "clearcoatRoughnessFactor", 0.0))
    transmission = num(obj(ext, "KHR_materials_transmission"), "transmissionFactor", 0.0)
    if transmission > 0.0 and alpha_mode == "OPAQUE":
      surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(max(1.0 - transmission, 0.1))

  # Geometry --------------------------------------------------------------------------------

  def mesh(self, parent: Sdf.Path, mesh_index: int) -> None:
    """The primitives of a glTF mesh as UsdGeom.Mesh prims under `parent`."""
    for p, prim in enumerate(self.doc.meshes[mesh_index].primitives):
      if (prim.mode if prim.mode is not None else TRIANGLES) != TRIANGLES:
        self.counts["skipped_primitives"] += 1
        continue
      attr = prim.attributes
      pos = self.gltf.accessor(_index(attr.POSITION, "a primitive's POSITION")).astype(np.float32)
      faces = (
        self.gltf.accessor(prim.indices).astype(np.int32).reshape(-1, 3)
        if prim.indices is not None
        else np.arange(len(pos)).astype(np.int32).reshape(-1, 3)
      )
      mesh = UsdGeom.Mesh.Define(self.stage, parent.AppendChild(f"prim{p}"))
      mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
      mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos))
      mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
      mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
      lo, hi = pos.min(axis=0), pos.max(axis=0)
      mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
      if attr.NORMAL is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(self.gltf.accessor(attr.NORMAL).astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
      colored = self._primvars(UsdGeom.PrimvarsAPI(mesh), attr)
      mat = self.material(prim.material)
      double_sided = False
      if mat is not None:
        if colored:
          mat.GetPrim().CreateAttribute(VERTEX_COLOR_ATTR, Sdf.ValueTypeNames.Token, custom=True).Set("displayColor")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        double_sided = bool(self.doc.materials[_index(prim.material, "a primitive's material")].doubleSided)
      mesh.CreateDoubleSidedAttr(double_sided)
      self.counts["meshes"] += 1
      self.counts["triangles"] += len(faces)

  def _primvars(self, api: UsdGeom.PrimvarsAPI, attr: pygltflib.Attributes) -> bool:
    """Every TEXCOORD_n as st / st<n> (V flipped to USD's bottom-left origin) and COLOR_0 as
    displayColor / displayOpacity; returns whether the primitive has vertex colors."""
    for k in range(4):
      acc = getattr(attr, f"TEXCOORD_{k}", None)  # pygltflib keeps TEXCOORD_2+ as extra attributes
      if acc is None:
        continue
      uv = self.gltf.accessor(acc).astype(np.float32)
      uv[:, 1] = 1.0 - uv[:, 1]
      pv = api.CreatePrimvar("st" if k == 0 else f"st{k}", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
      pv.Set(Vt.Vec2fArray.FromNumpy(uv))
    color = attr.COLOR_0
    if color is None:
      return False
    # Vertex colors (linear; integer ones are normalised by the spec): displayColor /
    # displayOpacity, multiplied into the base color by the material's VERTEX_COLOR_ATTR.
    c = self.gltf.accessor(color).astype(np.float32)
    api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
      Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(c[:, :3]))
    )
    if c.shape[1] > 3:
      api.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(np.ascontiguousarray(c[:, 3]))
      )
    return True

  def light(self, path: Sdf.Path, index: int) -> None:
    """KHR_lights_punctual light `index` at `path` (normalised: a point / spot light's intensity in
    cd, a directional light's in lux, whatever its node's scale)."""
    lights = obj(self.doc.extensions or {}, "KHR_lights_punctual").get("lights")
    if not isinstance(lights, list) or not 0 <= index < len(lights) or not isinstance(lights[index], Mapping):
      raise ValueError(f"glTF: KHR_lights_punctual has no light {index}")
    spec = cast("Object", lights[index])
    kind = spec.get("type")
    if kind == "directional":
      light = UsdLux.DistantLight.Define(self.stage, path)
      light.CreateAngleAttr(0.53)
    else:
      light = UsdLux.SphereLight.Define(self.stage, path)
      light.CreateRadiusAttr(_LIGHT_RADIUS)
      light.CreateTreatAsPointAttr(True)
      if kind == "spot":
        spot = obj(spec, "spot")
        outer = num(spot, "outerConeAngle", math.pi / 4)
        inner = num(spot, "innerConeAngle", 0.0)
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(math.degrees(outer))
        shaping.CreateShapingConeSoftnessAttr(1.0 - inner / outer if outer > 0 else 0.0)
    intensity = num(spec, "intensity", 1.0)
    if kind != "directional":
      # Candela: a normalized sphere's radiance is intensity / (4 pi r^2), its point intensity
      # radiance x pi r^2 (`core/pipeline/upload/lights.py`), at any radius or node scale (metres).
      intensity *= 4.0
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(*nums(spec, "color", (1.0, 1.0, 1.0))))
    light.CreateNormalizeAttr(True)
    self.counts["lights"] += 1

  def camera(self, path: Sdf.Path, index: int) -> None:
    """Perspective camera `index` at `path` (the focal length from the vertical field of view;
    both look down -Z with +Y up); an orthographic one is skipped."""
    spec = self.doc.cameras[index]
    if spec.type != "perspective" or spec.perspective is None:
      return
    cam = UsdGeom.Camera.Define(self.stage, path)
    aspect = spec.perspective.aspectRatio or _ASPECT
    cam.CreateVerticalApertureAttr(_APERTURE_MM)
    cam.CreateHorizontalApertureAttr(_APERTURE_MM * aspect)
    cam.CreateFocalLengthAttr(0.5 * _APERTURE_MM / math.tan(0.5 * spec.perspective.yfov))
    cam.CreateClippingRangeAttr(Gf.Vec2f(spec.perspective.znear or _NEAR_M, spec.perspective.zfar or _FAR_M))
    self.counts["cameras"] += 1

  def node(self, index: int, parent: Sdf.Path, world: Matrix4) -> None:
    """Node `index` (and its subtree) as an Xform under `parent`; `world`: the parent's world
    transform. A mesh used by several nodes is collected for `instancers` instead."""
    node = self.doc.nodes[index]
    local = self.gltf.node_matrix(node)
    world = world @ local
    path = parent.AppendChild(_name("node", index, node.name))
    xform = UsdGeom.Xform.Define(self.stage, path)
    xform.AddTransformOp().Set(Gf.Matrix4d(*local.T.flatten().tolist()))
    if node.mesh is not None:
      if node.mesh in self.instanced:
        self.uses.setdefault(node.mesh, []).append(world)
      else:
        self.mesh(path.AppendChild("mesh"), node.mesh)
    light = obj(node.extensions or {}, "KHR_lights_punctual")
    if light:
      self.light(path.AppendChild("light"), int(num(light, "light", -1)))
    if node.camera is not None:
      self.camera(path.AppendChild("camera"), node.camera)
    for child in node.children or []:
      self.node(child, path, world)

  def instancers(self) -> None:
    """One PointInstancer per mesh used by several nodes (world transforms)."""
    root = UsdGeom.Scope.Define(self.stage, "/World/Instancers").GetPath()
    for mesh_index, xforms in sorted(self.uses.items()):
      pi = UsdGeom.PointInstancer.Define(
        self.stage, root.AppendChild(_name("mesh", mesh_index, self.doc.meshes[mesh_index].name))
      )
      proto = UsdGeom.Xform.Define(self.stage, pi.GetPath().AppendChild("proto"))
      self.mesh(proto.GetPath(), mesh_index)
      pi.CreatePrototypesRel().SetTargets([proto.GetPath()])
      parts = [_decompose(m) for m in xforms]
      pi.CreateProtoIndicesAttr(Vt.IntArray([0] * len(parts)))
      pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.array([p[0] for p in parts], np.float32)))
      pi.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(*(float(x) for x in p[1])) for p in parts]))
      pi.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.array([p[2] for p in parts], np.float32)))
      self.counts["instancers"] += 1
      self.counts["instances"] += len(parts)

  def run(self) -> dict[str, int]:
    """Converts the document's default scene (else its first), writes the images and saves the
    stage; returns the counts of what was converted."""
    scene = self.doc.scenes[self.doc.scene if self.doc.scene is not None else 0]
    users = collections.Counter(n.mesh for n in self.doc.nodes if n.mesh is not None)
    self.instanced = {m for m, n in users.items() if n > 1}
    for index in scene.nodes or []:
      self.node(index, self.world.GetPath(), np.eye(4))
    self.instancers()
    self.write_images()
    self.stage.GetRootLayer().customLayerData = {"source": self.gltf.path.name, "converter": "raywarp.formats.gltf"}
    self.stage.Save()
    return self.counts


def convert(path: Path, out: Path) -> dict[str, int]:
  """Converts a glTF / GLB file to the USD stage `out` (.usda / .usdc); returns its counts."""
  out.parent.mkdir(parents=True, exist_ok=True)
  if out.exists():
    out.unlink()
  return _Converter(Gltf(path), out).run()
