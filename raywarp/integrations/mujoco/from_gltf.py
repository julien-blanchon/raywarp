"""Convert a glTF 2.0 asset (.gltf / .glb) into a visual-only MJCF model.

The output uses native MuJoCo material fields only. Each glTF material's constants go through
the canonical OpenPBR material (`raywarp.scene.model.material.Material`) and `_mjcf_attributes`,
the inverse of the MuJoCo adapter's mapping (`raywarp.integrations.mujoco.materials`: the `gltf`
and `mujoco` extras):

  baseColorFactor              -> material rgba
  metallicFactor/roughness     -> material metallic / roughness
  KHR_materials_ior, KHR_materials_specular specularFactor
                               -> material specular: the reflectance level 0.5 sqrt(w) x
                                  5 |ior - 1| / (ior + 1) (left at MuJoCo's 0.5 for w = 1,
                                  ior 1.5; clamped to 1 and reported above)
  baseColorTexture             -> texture layer role="rgb" (role="rgba" for BLEND), sRGB
  alphaMode MASK               -> opaque role="rgb" layer, the alpha channel as a role="opacity"
                                  layer and the cutoff as `<custom><numeric
                                  name="raywarp:alpha_cutoff:<material>">` (MJCF materials have
                                  no cutoff; raywarp's MuJoCo adapter reads it, MuJoCo's own
                                  renderer blends the opacity layer instead)
  normalTexture                -> texture layer role="normal", linear
  occlusion + metallicRoughness-> one packed texture layer role="orm" (R=AO, G=rough, B=metal)
  emissiveTexture/Factor       -> texture layer role="emissive", sRGB, factor baked in
  KHR_materials_emissive_strength -> the material's `emission` scalar (not clipped into the
                                  8-bit texture)
  KHR_texture_transform        -> baked into the mesh UVs (the base color texture's transform,
                                  else the first other texture's; one transform per material)
  KHR_materials_transmission   -> approximated as a thin translucent surface: alpha scaled by
                                  max(1 - transmissionFactor, 0.2) (no refraction or volume)

Every glTF image is written once per use (key: image, color space, baked factor), however
many materials share it. Every glTF triangle primitive becomes one OBJ mesh (node transforms baked
in, Y-up converted
to MuJoCo's Z-up) and one static, non-colliding mesh geom.

  KHR_lights_punctual          -> MJCF <light>s (photometric: intensity in candela, lux for a
                                  directional light; color -> diffuse; spot outerConeAngle ->
                                  cutoff; range -> range), placed at their node's world pose

Not converted: other KHR_* extensions (clearcoat, sheen, iridescence, volume, the specular
color, ...), per-texture transforms that differ within a material, TEXCOORD_1, vertex colors,
skins, morph targets, cameras. What a model authors that MJCF cannot carry is reported (printed
and listed in a comment at the top of the MJCF): sampler wrap modes other than REPEAT (MJCF
textures repeat), `doubleSided` (MJCF has no per-material sidedness), a spot's innerConeAngle,
and KHR_materials_emissive_strength's scale beyond MuJoCo's display-relative emission.

With --normalize SIZE the model is centred in x/y, rests on z = 0, and is scaled so that its
largest bounding-box extent equals SIZE metres (applied after --scale).

`raywarp convert MODEL.glb OUT_DIR/NAME.xml [--scale S] [--normalize SIZE]` (the CLI) or `convert`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pygltflib
from PIL import Image

from raywarp.formats.gltf.reader import TRIANGLES, Gltf, Matrix4
from raywarp.formats.gltf.values import Object, TextureInfo, num, nums, obj
from raywarp.integrations.mujoco.materials import ALPHA_CUTOFF_PREFIX, WEIGHT_PER_LEVEL2
from raywarp.scene.model.material import Material, RaywarpExt
from raywarp.scene.model.types import AlphaMode

if TYPE_CHECKING:
  from shape_extensions import IntVar

# glTF sampler wrap mode REPEAT (CLAMP_TO_EDGE 33071, MIRRORED_REPEAT 33648: not in MJCF).
_REPEAT: Final = 10497
_WRAP_NAMES: Final = {33071: "clamp", 33648: "mirror"}
# glTF is Y-up; MuJoCo is Z-up: (x, y, z) -> (x, -z, y).
_Y_UP_TO_Z_UP: Final = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


# Smallest alpha of a transmissive material: fully transmissive glass stays faintly visible.
_MIN_TRANSMISSION_ALPHA: Final = 0.2
_MJ_DEFAULT_SPECULAR: Final = 0.5  # MJCF default: w = 1 at ior 1.5
# Stands for KHR_materials_ior's ior 0 (Fresnel 1): its IOR factor 5 |ior - 1| / (ior + 1) is 5.
_IOR_FRESNEL_ONE: Final = 1e30


def _texture_transform(info: TextureInfo) -> np.ndarray | None:
  """KHR_texture_transform of a texture reference as a 3x3 UV matrix (T * R * S), or None."""
  ext = obj(info.extensions or {}, "KHR_texture_transform")
  if not ext:
    return None
  ox, oy = nums(ext, "offset", [0.0, 0.0])
  r = num(ext, "rotation", 0.0)
  sx, sy = nums(ext, "scale", [1.0, 1.0])
  t = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
  rot = np.array([[np.cos(r), np.sin(r), 0.0], [-np.sin(r), np.cos(r), 0.0], [0.0, 0.0, 1.0]])
  return t @ rot @ np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def _write_obj[N: IntVar](
  path: Path,
  pos: np.ndarray[[N, 3], np.dtype[np.floating]],
  normal: np.ndarray[[N, 3], np.dtype[np.floating]] | None,
  uv: np.ndarray[[N, 2], np.dtype[np.floating]] | None,
  faces: np.ndarray[[int, 3], np.dtype[np.int64]],
) -> None:
  """Writes an OBJ mesh of vertices `pos` (and their `normal` / `uv`, when given) and `faces`."""
  lines = [f"v {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}" for p in pos]
  if uv is not None:
    # MuJoCo flips v when loading OBJ, so write OBJ's bottom-left convention.
    lines += [f"vt {t[0]:.7g} {1.0 - t[1]:.7g}" for t in uv]
  if normal is not None:
    lines += [f"vn {n[0]:.7g} {n[1]:.7g} {n[2]:.7g}" for n in normal]
  for f in faces + 1:
    if uv is not None and normal is not None:
      lines.append("f " + " ".join(f"{i}/{i}/{i}" for i in f))
    elif uv is not None:
      lines.append("f " + " ".join(f"{i}/{i}" for i in f))
    elif normal is not None:
      lines.append("f " + " ".join(f"{i}//{i}" for i in f))
    else:
      lines.append("f " + " ".join(str(i) for i in f))
  path.write_text("\n".join(lines) + "\n")


def _canonical(index: int, mat: pygltflib.Material, rgba: list[float], *, emissive: bool) -> Material:
  """The canonical OpenPBR material of glTF material `index`'s constants as the MJCF material
  carries them: `rgba` is the base color with the opacity left after the alpha mode and the
  transmission approximation; with an emissive layer (factor baked in) the luminance is
  KHR_materials_emissive_strength; KHR_materials_ior / the specularFactor of
  KHR_materials_specular set the dielectric reflectance."""
  pbr = mat.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness()
  extensions = mat.extensions or {}
  specular = obj(extensions, "KHR_materials_specular")
  if "specularColorFactor" in specular or "specularColorTexture" in specular or "specularTexture" in specular:
    print(f"  material {index}: KHR_materials_specular color / textures not converted (its specularFactor is)")
  strength = num(obj(extensions, "KHR_materials_emissive_strength"), "emissiveStrength", 1.0)
  mask = (mat.alphaMode or "OPAQUE") == "MASK"
  cutoff = float(mat.alphaCutoff if mat.alphaCutoff is not None else 0.5)
  return Material(
    base_color=(float(rgba[0]), float(rgba[1]), float(rgba[2])),
    geometry_opacity=float(rgba[3]),
    base_metalness=1.0 if pbr.metallicFactor is None else float(pbr.metallicFactor),
    specular_roughness=1.0 if pbr.roughnessFactor is None else float(pbr.roughnessFactor),
    specular_weight=num(specular, "specularFactor", 1.0),
    specular_ior=_ior(index, extensions),
    emission_luminance=strength if emissive else 0.0,
    ext=RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=cutoff) if mask else RaywarpExt(),
  )


def _ior(index: int, extensions: Object) -> float:
  """The KHR_materials_ior ior (default 1.5). Its special value 0 (the specular-glossiness
  compatibility mode: Fresnel 1 at every angle) maps to `_IOR_FRESNEL_ONE`, the reflectance
  clamped to MJCF's largest specular 1, reported.

  Raises:
    ValueError: for an ior that is neither 0 nor >= 1 (invalid glTF).
  """
  ior = num(obj(extensions, "KHR_materials_ior"), "ior", 1.5)
  if ior == 0.0:
    print(f"  material {index}: KHR_materials_ior ior 0 (Fresnel 1): MuJoCo's largest specular 1")
    return _IOR_FRESNEL_ONE
  if ior < 1.0:
    raise ValueError(f"material {index}: KHR_materials_ior ior {ior} (valid: 0 or >= 1)")
  return ior


def _mjcf_attributes(index: int, mat: Material) -> str:
  """The `<material>` attributes of `mat`'s constants, the inverse of
  `raywarp.integrations.mujoco.materials.openpbr_materials`: rgba = (base_color, geometry_opacity),
  MuJoCo's specular = the device reflectance sqrt(w / 4) x 5 |ior - 1| / (ior + 1) (clamped to
  MJCF's 1, reported), emission = `emission_luminance`; MuJoCo's defaults (specular 0.5, emission 0)
  are left out."""
  rgba = (*mat.base_color, mat.geometry_opacity)
  attrs = (
    f'name="mat{index}" rgba="{" ".join(f"{c:.6g}" for c in rgba)}" metallic="{mat.base_metalness:.6g}"'
    f' roughness="{mat.specular_roughness:.6g}"'
  )
  w, ior = mat.specular_weight, mat.specular_ior
  level = math.sqrt(w / WEIGHT_PER_LEVEL2) * 5.0 * abs(ior - 1.0) / (ior + 1.0)
  if level > 1.0 and ior != _IOR_FRESNEL_ONE:
    print(f"  material {index}: the dielectric reflectance exceeds MJCF's specular range: clamped to 1")
  if level != _MJ_DEFAULT_SPECULAR:
    attrs += f' specular="{min(level, 1.0):.6g}"'
  if mat.emission_luminance > 0.0:
    attrs += f' emission="{mat.emission_luminance:.6g}"'
  return attrs


# (mesh name, positions, normals, UVs, faces) of a converted primitive (`_write_obj`'s arrays).
type _Primitive = tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]


class _Converter:
  """One glTF document's conversion into `out_dir` (`run`: the MJCF file, its meshes and
  textures)."""

  def __init__(self, gltf: Gltf, out_dir: Path, name: str, scale: float, normalize: float | None) -> None:
    """A conversion of `gltf` into `out_dir/name.xml` (`scale`, `normalize`: `convert`)."""
    self.gltf = gltf
    self.out_dir = out_dir
    self.name = name
    self.scale = scale
    self.normalize = normalize
    self.primitives: list[_Primitive] = []  # (name, pos, normal, uv, faces) before normalisation
    self.uv_transforms: dict[int, np.ndarray] = {}  # material index -> KHR_texture_transform
    self.textures: list[tuple[str, str, str, int]] = []  # (name, file, colorspace, channels)
    self.written: dict[tuple[object, ...], str] = {}  # (kind, glTF image ids, baked parameters) -> name
    self.materials: dict[int, str] = {}  # gltf material index -> MJCF <material> element string
    self.meshes: list[tuple[str, str]] = []  # (name, file)
    self.geoms: list[str] = []  # MJCF <geom> element strings
    self.cutoffs: dict[str, float] = {}  # material name -> alpha cutoff (MASK)
    self.lights: list[tuple[Matrix4, Object]] = []  # (node world matrix, KHR_lights_punctual light)
    self.notes: list[str] = []  # what the MJCF cannot carry (printed, and a comment in the MJCF)
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "textures").mkdir(parents=True, exist_ok=True)

  def note(self, message: str) -> None:
    """Reports something the MJCF cannot carry (printed; listed in the MJCF's header comment)."""
    print(f"  {message}")
    self.notes.append(message)

  def _check_material(self, index: int, mat: pygltflib.Material, infos: list[TextureInfo]) -> None:
    """Notes the material's sidedness, its textures' non-repeating wrap modes and an emissive
    strength (MuJoCo's emission is display-relative)."""
    if mat.doubleSided:
      self.note(f"material {index}: doubleSided (MJCF has no per-material sidedness)")
    for info in infos:
      if info.index is None:
        raise ValueError(f"material {index}: a textureInfo without its texture index")
      texture = self.gltf.doc.textures[info.index]
      sampler = self.gltf.doc.samplers[texture.sampler] if texture.sampler is not None else None
      wraps = {w for w in (sampler.wrapS, sampler.wrapT) if w is not None and w != _REPEAT} if sampler else set()
      if wraps:
        names = ", ".join(sorted(_WRAP_NAMES.get(w, str(w)) for w in wraps))
        self.note(f"material {index}: texture {info.index} wrap {names} (MJCF textures repeat)")
    value = num(obj(mat.extensions or {}, "KHR_materials_emissive_strength"), "emissiveStrength", 1.0)
    if value != 1.0:
      self.note(f"material {index}: KHR_materials_emissive_strength {value:g} as MuJoCo's display-relative emission")

  def _save_texture(self, name: str, image: Image.Image, colorspace: str) -> str:
    """Writes `image` as texture `name` (a PNG under textures/) and returns its name."""
    file = f"textures/{name}.png"
    image.save(self.out_dir / file)
    self.textures.append((name, file, colorspace, len(image.getbands())))
    return name

  def _texture(self, key: tuple[object, ...], name: str, make: Callable[[], Image.Image], colorspace: str) -> str:
    """The texture of `key` (written once: `make()` builds its image the first time)."""
    if key not in self.written:
      self.written[key] = self._save_texture(name, make(), colorspace)
    return self.written[key]

  def _source(self, texture_index: int | None) -> int | None:
    """The glTF image index of texture `texture_index`."""
    return None if texture_index is None else self.gltf.doc.textures[texture_index].source

  def _orm_texture(self, index: int, mat: pygltflib.Material) -> str | None:
    """The packed ORM texture of material `index` (None without occlusion and metallic-roughness
    textures)."""
    pbr = mat.pbrMetallicRoughness
    mr = pbr.metallicRoughnessTexture if pbr else None
    occ = mat.occlusionTexture
    if mr is None and occ is None:
      return None
    strength = occ.strength if occ is not None and occ.strength is not None else 1.0

    def make() -> Image.Image:
      """The packed ORM image (R occlusion, G roughness, B metallic)."""
      mr_img = np.asarray(self.gltf.image(mr.index).convert("RGB")) if mr else None
      occ_img = np.asarray(self.gltf.image(occ.index).convert("RGB")) if occ else None
      shape = next(img for img in (mr_img, occ_img) if img is not None).shape
      orm = np.full(shape, 255, dtype=np.uint8)
      if mr_img is not None:
        orm[..., 1:] = mr_img[..., 1:]
      if occ_img is not None:
        if occ_img.shape != shape:
          occ_img = np.asarray(Image.fromarray(occ_img).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
        orm[..., 0] = np.clip(255.0 - strength * (255.0 - occ_img[..., 0]), 0, 255).astype(np.uint8)
      return Image.fromarray(orm)

    key = ("orm", self._source(mr.index if mr else None), self._source(occ.index if occ else None), strength)
    return self._texture(key, f"mat{index}_orm", make, "linear")

  def _emissive_texture(self, index: int, mat: pygltflib.Material) -> str | None:
    """The emissive texture of material `index`, factor baked in (None for a black factor)."""
    factor = np.array(mat.emissiveFactor or [0.0, 0.0, 0.0], dtype=np.float32)
    if not factor.any():
      return None
    texture = mat.emissiveTexture.index if mat.emissiveTexture is not None else None

    def make() -> Image.Image:
      """The emissive texture with the factor (<= 1, glTF) baked in."""
      if texture is not None:
        img = np.asarray(self.gltf.image(texture).convert("RGB"), dtype=np.float32) / 255.0
      else:
        img = np.ones((1, 1, 3), dtype=np.float32)
      # Bake the factor in sRGB space via linear: decode, scale, re-encode.
      lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4) * factor
      srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
      return Image.fromarray(np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8))

    key = ("emissive", self._source(texture), tuple(float(f) for f in factor))
    return self._texture(key, f"mat{index}_emissive", make, "sRGB")

  def material(self, index: int | None) -> str | None:
    """The MJCF name of glTF material `index`, converted at its first use (None: no material)."""
    if index is None:
      return None
    if index in self.materials:
      return f"mat{index}"
    mat = self.gltf.doc.materials[index]
    pbr = mat.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness()
    rgba = list(pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0])
    alpha_mode = mat.alphaMode or "OPAQUE"
    if alpha_mode in ("OPAQUE", "MASK"):
      rgba[3] = 1.0
    extensions = mat.extensions or {}
    transmission = num(obj(extensions, "KHR_materials_transmission"), "transmissionFactor", 0.0)
    if transmission > 0.0:
      rgba[3] *= max(1.0 - transmission, _MIN_TRANSMISSION_ALPHA)
    infos = [pbr.baseColorTexture, mat.normalTexture, pbr.metallicRoughnessTexture, mat.occlusionTexture, mat.emissiveTexture]
    self._check_material(index, mat, [info for info in infos if info is not None])
    transforms = [_texture_transform(info) for info in infos if info is not None]
    transforms = [t for t in transforms if t is not None]
    if transforms:
      self.uv_transforms[index] = transforms[0]
      if any(not np.allclose(t, transforms[0]) for t in transforms[1:]):
        print(f"  material {index}: textures with different KHR_texture_transforms, using the first one")

    layers = self._base_layers(index, pbr, alpha_mode)
    if mat.normalTexture is not None:
      normal = mat.normalTexture.index
      key = ("normal", self._source(normal))
      layers.append(
        (self._texture(key, f"mat{index}_normal", lambda: self.gltf.image(normal).convert("RGB"), "linear"), "normal")
      )
    orm = self._orm_texture(index, mat)
    if orm is not None:
      layers.append((orm, "orm"))
    emissive = self._emissive_texture(index, mat)
    if emissive is not None:
      layers.append((emissive, "emissive"))

    canonical = _canonical(index, mat, rgba, emissive=emissive is not None)
    if canonical.ext.alpha_mode == AlphaMode.MASK:
      self.cutoffs[f"mat{index}"] = canonical.ext.alpha_cutoff
    attrs = _mjcf_attributes(index, canonical)
    body = "".join(f'\n      <layer texture="{tex}" role="{role}"/>' for tex, role in layers)
    self.materials[index] = f"    <material {attrs}>{body}\n    </material>" if body else f"    <material {attrs}/>"
    return f"mat{index}"

  def _base_layers(self, index: int, pbr: pygltflib.PbrMetallicRoughness, alpha_mode: str) -> list[tuple[str, str]]:
    """The (texture, role) layers of material `index`'s base color texture (none without one):
    rgb (rgba when blended) and, for MASK, the opacity layer (factor alpha x texel alpha, tested
    against the cutoff)."""
    if pbr.baseColorTexture is None:
      return []
    role = "rgba" if alpha_mode == "BLEND" else "rgb"
    mode = "RGB" if role == "rgb" else "RGBA"
    base = pbr.baseColorTexture.index
    source = self._source(base)
    color = self._texture(
      ("basecolor", source, mode), f"mat{index}_basecolor", lambda: self.gltf.image(base).convert(mode), "sRGB"
    )
    if alpha_mode != "MASK":
      return [(color, role)]
    factor_alpha = float((pbr.baseColorFactor or [1.0] * 4)[3])

    def opacity() -> Image.Image:
      """The opacity layer: texel alpha x factor alpha."""
      alpha = np.asarray(self.gltf.image(base).convert("RGBA"), np.float32)[..., 3] * factor_alpha
      return Image.fromarray((alpha + 0.5).astype(np.uint8), "L")

    mask = self._texture(("opacity", source, factor_alpha), f"mat{index}_opacity", opacity, "linear")
    return [(color, role), (mask, "opacity")]

  def primitive(self, mesh_index: int, prim_index: int, prim: pygltflib.Primitive, world: Matrix4) -> None:
    """Converts one mesh primitive at node matrix `world` (non-triangle primitives skipped)."""
    if (prim.mode if prim.mode is not None else TRIANGLES) != TRIANGLES:
      print(f"  skipping mesh {mesh_index} primitive {prim_index}: mode {prim.mode} is not TRIANGLES")
      return
    attr = prim.attributes
    if attr.POSITION is None:
      raise ValueError(f"mesh {mesh_index} primitive {prim_index} has no POSITION attribute")
    pos = self.gltf.accessor(attr.POSITION).astype(np.float64)
    normal = self.gltf.accessor(attr.NORMAL).astype(np.float64) if attr.NORMAL is not None else None
    uv = self.gltf.accessor(attr.TEXCOORD_0).astype(np.float64) if attr.TEXCOORD_0 is not None else None
    if prim.indices is not None:
      faces = self.gltf.accessor(prim.indices).astype(np.int64).reshape(-1, 3)
    else:
      faces = np.arange(len(pos)).reshape(-1, 3)

    pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3] * self.scale
    if normal is not None:
      normal = normal @ np.linalg.inv(world[:3, :3])  # inverse transpose, row vectors.
      normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    if np.linalg.det(world[:3, :3]) < 0.0:
      faces = faces[:, ::-1]

    mat = self.material(prim.material)
    transform = self.uv_transforms.get(prim.material) if prim.material is not None else None
    if uv is not None and transform is not None:
      uv = (np.c_[uv, np.ones(len(uv))] @ transform.T)[:, :2]
    if uv is None and prim.material is not None and "<layer" in self.materials[prim.material]:
      # A textured material on a primitive without TEXCOORD_0 (glTF allows it for a constant
      # emissive color, baked here into a 1 x 1 texture): constant UVs, so the texture is read.
      uv = np.zeros((len(pos), 2))
    name = f"mesh{mesh_index}_{prim_index}_{len(self.primitives)}"
    self.primitives.append((name, pos, normal, uv, faces))
    mat_attr = f' material="{mat}"' if mat else ""
    self.geoms.append(f'    <geom type="mesh" mesh="{name}"{mat_attr} contype="0" conaffinity="0" group="1" mass="0"/>')

  def node(self, node_index: int, parent: Matrix4) -> None:
    """Converts node `node_index` and its subtree (`parent`: its parent's world matrix)."""
    node = self.gltf.doc.nodes[node_index]
    world = parent @ self.gltf.node_matrix(node)
    if node.mesh is not None:
      for prim_index, prim in enumerate(self.gltf.doc.meshes[node.mesh].primitives):
        self.primitive(node.mesh, prim_index, prim, world)
    light = obj(node.extensions or {}, "KHR_lights_punctual").get("light")
    if light is not None:
      lights = obj(self.gltf.doc.extensions or {}, "KHR_lights_punctual").get("lights")
      if not isinstance(lights, list) or not isinstance(light, int) or not 0 <= light < len(lights):
        raise ValueError(f"node {node_index}: KHR_lights_punctual light {light!r} is not one of the document's lights")
      spec = lights[light]
      if not isinstance(spec, dict):
        raise ValueError(f"KHR_lights_punctual light {light} must be an object, got {spec!r}")
      self.lights.append((world, spec))
    for child in node.children or []:
      self.node(child, world)

  def _light(self, index: int, world: Matrix4, light: Object, offset: np.ndarray, factor: float) -> str:
    """The MJCF <light> of a KHR_lights_punctual light at its node's world matrix (a glTF light
    shines along its node's -Z), placed like the meshes (`offset`, `factor`)."""
    kind = str(light.get("type", "point"))
    if kind not in ("point", "spot", "directional"):
      raise ValueError(f"light {index}: KHR_lights_punctual type {kind!r}")
    if kind == "spot" and "innerConeAngle" in obj(light, "spot"):
      self.note(f"light {index}: spot innerConeAngle (MJCF cuts off at the outer cone)")
    pos = (world[:3, 3] * self.scale + offset) * factor
    direction = world[:3, :3] @ np.array([0.0, 0.0, -1.0])
    length = math.hypot(*(float(c) for c in direction.tolist()))
    return _light_element(index, kind, pos, direction / max(length, 1e-12), light)

  def _notes_comment(self) -> str:
    """The MJCF header comment listing the notes (empty without notes)."""
    if not self.notes:
      return ""
    rows = "".join(f"\n       {n.replace('--', '-')}" for n in self.notes)
    return f"\n  <!-- Not carried by MJCF:{rows} -->"

  def run(self) -> Path:
    """Converts the default scene and writes the MJCF file (returned), its meshes and textures."""
    doc = self.gltf.doc
    scene = doc.scenes[doc.scene if doc.scene is not None else 0]
    for node_index in scene.nodes or []:
      self.node(node_index, _Y_UP_TO_Z_UP)

    offset, factor = np.zeros(3), 1.0
    if self.normalize is not None and self.primitives:
      lower = np.min([p[1].min(axis=0) for p in self.primitives], axis=0)
      upper = np.max([p[1].max(axis=0) for p in self.primitives], axis=0)
      factor = self.normalize / max(float((upper - lower).max()), 1e-12)
      offset = -np.array([0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]), lower[2]])
    for name, pos, normal, uv, faces in self.primitives:
      file = f"meshes/{name}.obj"
      _write_obj(self.out_dir / file, (pos + offset) * factor, normal, uv, faces)
      self.meshes.append((name, file))

    # MuJoCo keeps `nchannel` (default 3) channels of a texture file: RGBA ones need nchannel="4".
    textures = "\n".join(
      f'    <texture name="{n}" type="2d" file="{f}" colorspace="{c}"{' nchannel="4"' if k == 4 else ""}/>'
      for n, f, c, k in self.textures
    )
    meshes = "\n".join(f'    <mesh name="{n}" file="{f}" inertia="shell"/>' for n, f in self.meshes)
    lights = [self._light(i, world, light, offset, factor) for i, (world, light) in enumerate(self.lights)]
    custom = ""
    if self.cutoffs:
      rows = "\n".join(f'    <numeric name="{ALPHA_CUTOFF_PREFIX}{m}" data="{c:.6g}"/>' for m, c in self.cutoffs.items())
      custom = f"  <custom>\n{rows}\n  </custom>\n"
    xml = f"""<mujoco model="{self.name}">
  <!-- Converted from {self.gltf.path.name} by raywarp/integrations/mujoco/from_gltf.py -->{self._notes_comment()}
  <compiler meshdir="." texturedir="."/>
  <asset>
{textures}
{chr(10).join(self.materials.values())}
{meshes}
  </asset>
  <worldbody>
{chr(10).join([*lights, *self.geoms])}
  </worldbody>
{custom}</mujoco>
"""
    out = self.out_dir / f"{self.name}.xml"
    out.write_text(xml)
    return out


def _light_element(index: int, kind: str, pos: np.ndarray, direction: np.ndarray, light: Object) -> str:
  """An MJCF <light> of glTF punctual light `light` at world `pos`, shining along `direction`."""
  color = nums(light, "color", [1.0, 1.0, 1.0])
  attrs = (
    f'name="light{index}" type="{kind}"'
    f' pos="{" ".join(f"{c:.6g}" for c in pos)}" dir="{" ".join(f"{c:.6g}" for c in direction)}"'
    f' diffuse="{" ".join(f"{c:.6g}" for c in color)}" intensity="{num(light, "intensity", 1.0):.6g}"'
  )
  if light.get("range") is not None:
    attrs += f' range="{num(light, "range", 0.0):.6g}"'
  if kind == "spot":
    outer = num(obj(light, "spot"), "outerConeAngle", math.pi / 4.0)
    attrs += f' cutoff="{math.degrees(outer):.6g}"'
  return f"    <light {attrs}/>"


def convert(path: Path, out_dir: Path, name: str | None = None, scale: float = 1.0, normalize: float | None = None) -> Path:
  """Convert a glTF/GLB file and return the path of the written MJCF file."""
  out_dir.mkdir(parents=True, exist_ok=True)
  return _Converter(Gltf(path), out_dir, name or path.stem, scale, normalize).run()
