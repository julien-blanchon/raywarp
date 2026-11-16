"""Material texture maps at a hit: the texture footprint, the texel fetch and the sampled maps.

  make_footprint    the ray cone's footprint in UV space (`Footprint`)
  make_fetch        a texel averaged over the footprint (isotropic, or anisotropic taps)
  make_sample_maps  base color, ORM / occlusion / roughness / metallic, diffuse roughness, the
                    layered roles, opacity, emissive (each texel through its role's texel map)
(the texture frame: `uv_frame.py`; the normal maps and all maps at a hit's texture frame:
`uv_maps.py`). Texels are read at the mip level of the hit's ray cone (`texture.texture_lod`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import warp as wp

from raywarp._base.types import TexRole, tex_affine
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.tables import texture_id
from raywarp.core.kernels.generic import device_generic
from raywarp.core.shade.materials import texture
from raywarp.core.shade.materials.material import MaterialTables, Surface

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_ROLE_RGB: Final = int(TexRole.RGB)
_ROLE_RGBA: Final = int(TexRole.RGBA)
_ROLE_ORM: Final = int(TexRole.ORM)
_ROLE_OCCLUSION: Final = int(TexRole.OCCLUSION)
_ROLE_ROUGHNESS: Final = int(TexRole.ROUGHNESS)
_ROLE_METALLIC: Final = int(TexRole.METALLIC)
_ROLE_EMISSIVE: Final = int(TexRole.EMISSIVE)
_ROLE_OPACITY: Final = int(TexRole.OPACITY)
_ROLE_DIFFUSE_ROUGHNESS: Final = int(TexRole.DIFFUSE_ROUGHNESS)
_ROLE_ANISOTROPY: Final = int(TexRole.ANISOTROPY)
_ROLE_FUZZ: Final = int(TexRole.FUZZ)


@wp.struct
class Footprint:
  """A hit's texture footprint: the ray cone on the surface (an ellipse) in UV space.

  Attributes:
    cone: isotropic width (anisotropy 1: sets the mip level, `texture.texture_lod`).
    major: UV vector of the ellipse's major axis (its full length; anisotropic filtering).
    minor: UV vector of its minor axis (its full length); zero when the frame is degenerate (the
      isotropic tap then).
    taps: the most taps along the major axis (1: the isotropic tap; `make_fetch`).
  """

  cone: float
  major: wp.vec2
  minor: wp.vec2
  taps: int


type FetchTexel = Callable[[MaterialTables, int, wp.vec2, Footprint, float], wp.vec4]


@wp.func
def surface_to_uv(
  # In:
  d: wp.vec3,
  tangent: wp.vec3,
  bitangent: wp.vec3,
) -> wp.vec2:
  """UV offset of the surface step `d`: d = T du + B dv in the least-squares sense (zero for a
  degenerate frame)."""
  tt = wp.dot(tangent, tangent)
  tb = wp.dot(tangent, bitangent)
  bb = wp.dot(bitangent, bitangent)
  det = tt * bb - tb * tb
  if det <= 1.0e-20:
    return wp.vec2(0.0)
  rt = wp.dot(tangent, d)
  rb = wp.dot(bitangent, d)
  return wp.vec2(bb * rt - tb * rb, tt * rb - tb * rt) / det


def make_footprint(anisotropy: int):
  """footprint(cone, width, local_dir, local_normal, tangent, bitangent) -> Footprint."""

  @wp.func
  def footprint(
    # In:
    cone: float,
    width: float,
    local_dir: wp.vec3,
    local_normal: wp.vec3,
    tangent: wp.vec3,
    bitangent: wp.vec3,
  ) -> Footprint:
    """The footprint of a ray cone of `width` hitting the surface along `local_dir`.

    Isotropic (anisotropy 1): one tap of `cone`, the geometric mean of the footprint's axes
    (width across the view, width / |n.d| along it). Anisotropic: the ellipse's axes in UV
    space through the dual basis of (tangent, bitangent) = (dp/du, dp/dv); `make_fetch` spreads
    its taps along the axis that is longer in texels.
    """
    fp = Footprint()
    fp.taps = wp.static(anisotropy)
    fp.cone = cone
    fp.major = wp.vec2(0.0)
    fp.minor = wp.vec2(0.0)
    if wp.static(anisotropy > 1):
      nod = wp.max(wp.abs(wp.dot(local_normal, local_dir)), wp.static(texture.MIN_NOD))
      along = local_dir - local_normal * wp.dot(local_dir, local_normal)
      if wp.length_sq(along) <= 1.0e-12:  # seen head-on: a circle, any in-plane axes
        along = tangent - local_normal * wp.dot(tangent, local_normal)
      if wp.length_sq(along) > 1.0e-12:
        a = wp.normalize(along)
        fp.major = surface_to_uv(a * (width / nod), tangent, bitangent)
        fp.minor = surface_to_uv(wp.cross(local_normal, a) * width, tangent, bitangent)
    return fp

  return footprint


@wp.func
def vary_uv(
  # In:
  v: wp.vec4,
  uv: wp.vec2,
  tangent: wp.vec3,
  bitangent: wp.vec3,
) -> tuple[wp.vec2, wp.vec3, wp.vec3]:
  """`uv` and its frame (tangent, bitangent) = (dp/du, dp/dv) through a material's texture
  variation `v` (`material.tex_variation`: offset, then rotation about the texture's centre);
  unchanged without one.

  The varied coordinates are w = R (uv - c) + c, so their frame is dp/dw = (dp/du) R^T: the
  footprint (`surface_to_uv`) and the normal map's tangent frame follow the rotation (a UV
  rotation by +theta shows the texture, and its relief, turned by -theta).
  """
  uv = uv + wp.vec2(v[0], v[1])
  if v[2] != 0.0:
    c = wp.cos(v[2])
    s = wp.sin(v[2])
    uv = _rotate(uv - wp.vec2(0.5), c, s) + wp.vec2(0.5)
    t = tangent * c - bitangent * s
    bitangent = tangent * s + bitangent * c
    tangent = t
  return uv, tangent, bitangent


@wp.func
def _rotate(
  # In:
  p: wp.vec2,
  c: float,
  s: float,
) -> wp.vec2:
  """`p` rotated by the angle of cosine `c` and sine `s`."""
  return wp.vec2(c * p[0] - s * p[1], s * p[0] + c * p[1])


@wp.func
def ellipse_axes(
  # In:
  major: wp.vec2,
  minor: wp.vec2,
  size: wp.vec2,
) -> tuple[wp.vec2, float, float]:
  """Principal UV axis and texel lengths of an ellipse spanned by `major` / `minor`.

  UV transforms and non-square textures need not preserve perpendicular axes. Eigenvectors of
  A A^T, A = [size * major, size * minor], describe the same ellipse in texture space. Computing
  the short length from |det A| / long avoids cancellation for highly elongated footprints.
  """
  a = wp.cw_mul(major, size)
  b = wp.cw_mul(minor, size)
  if wp.dot(a, b) == 0.0:  # Perpendicular axes are already principal: their lengths directly.
    la = wp.length(a)
    lb = wp.length(b)
    return wp.where(la >= lb, major, minor), wp.max(la, lb), wp.min(la, lb)
  # Covariance squares lengths, and its discriminant squares them again. Scale first so
  # finite large footprints do not overflow.
  scale = wp.max(wp.max(wp.abs(a[0]), wp.abs(a[1])), wp.max(wp.abs(b[0]), wp.abs(b[1])))
  a = a / scale
  b = b / scale
  xx = a[0] * a[0] + b[0] * b[0]
  yy = a[1] * a[1] + b[1] * b[1]
  xy = a[0] * a[1] + b[0] * b[1]
  delta = wp.sqrt((xx - yy) * (xx - yy) + 4.0 * xy * xy)
  longest = 0.5 * (xx + yy + delta)
  if longest <= 0.0:
    return wp.vec2(0.0), 0.0, 0.0
  if delta == 0.0:  # A circle has no preferred axis; the one-tap path only needs its size.
    return major, wp.sqrt(longest) * scale, wp.sqrt(longest) * scale
  direction = wp.vec2(longest - yy, xy)
  if yy > xx:
    direction = wp.vec2(xy, longest - xx)
  longer = wp.sqrt(longest)
  axis = wp.cw_div(wp.normalize(direction) * (longer * scale), size)
  shorter = wp.abs(a[0] * b[1] - a[1] * b[0]) / longer
  return axis, longer * scale, shorter * scale


def make_fetch(anisotropy: int):
  """fetch(mats, tex_id, uv, fp, uv_per_length) -> texel averaged over the footprint's taps
  (`anisotropy`: the most compiled in, `Footprint.taps` the settings' up to it, 1 the isotropic
  tap)."""

  @wp.func
  @device_generic
  def fetch[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    tex_id: int,
    uv: wp.vec2,
    fp: Footprint,
    uv_per_length: float,
  ) -> wp.vec4:
    """Texel of texture `tex_id` around `uv` over the footprint (`uv` / `fp` already varied:
    `vary_uv`).

    Isotropic: one tap at the mip level of `fp.cone`. Anisotropic (the hardware scheme, in this
    texture's texels): the ellipse's principal axes measured in texels (a non-square texture or
    UV mapping can also shear the spanning axes), up to `anisotropy` taps along the longer one,
    each at the mip level of the longer axis / taps (at least the shorter axis); a footprint
    less than 2:1 in texels takes one tap at the axes' geometric mean.
    """
    size = mats.tex_size[tex_id]
    layout = mats.tex_layout[tex_id]
    taps = fp.taps
    if wp.static(anisotropy > 1):
      if taps > 1:
        axis, longer, shorter = ellipse_axes(fp.major, fp.minor, size)
        if longer > 0.0 and shorter > 0.0:
          if longer <= 2.0 * shorter:
            # Nearly round: one tap at the axes' geometric mean (the isotropic filter's size).
            return texture.sample_layout(mats.textures[tex_id], layout, uv, wp.log2(wp.max(wp.sqrt(longer * shorter), 1.0e-8)))
          # The ratio is capped before the int conversion (a needle footprint's can exceed the int
          # range); the int min keeps the loop's trip count bounded by `taps` for the compiler.
          ntap = wp.min(int(wp.ceil(wp.min(longer / shorter, float(taps)))), taps)
          lod = wp.log2(wp.max(wp.max(longer / float(ntap), shorter), 1.0e-8))
          texel = wp.vec4(0.0)
          step = axis / float(ntap)
          centre = 0.5 * float(ntap - 1)
          for k in range(ntap):
            texel += texture.sample_layout(mats.textures[tex_id], layout, uv + step * (float(k) - centre), lod)
          return texel / float(ntap)
    lod = texture.texture_lod(fp.cone, uv_per_length, size)
    return texture.sample_layout(mats.textures[tex_id], layout, uv, lod)

  return fetch


@wp.func
def texel_map(
  # In:
  affine: tex_affine,
  texel: wp.vec4,
) -> wp.vec4:
  """`texel` through a material's texel map of its role: scale * texel + bias."""
  return wp.cw_mul(affine[0], texel) + affine[1]


@wp.func
def color_texel_map(
  # In:
  affine: tex_affine,
  hue: float,
  texel: wp.vec4,
) -> wp.vec4:
  """`texel_map` of a color texel, then its hue rotation by `hue` [rad] (the material's
  `tex_variation`: rgb turned about the grey axis, Rodrigues: the channel mean is kept; clamped
  at 0)."""
  t = texel_map(affine, texel)
  if hue == 0.0:
    return t
  c = wp.cos(hue)
  s = wp.sin(hue) * wp.static(1.0 / math.sqrt(3.0))
  v = wp.vec3(t[0], t[1], t[2])
  grey = (v[0] + v[1] + v[2]) * (1.0 - c) / 3.0
  cross = wp.vec3(v[2] - v[1], v[0] - v[2], v[1] - v[0])
  r = wp.max(v * c + cross * s + wp.vec3(grey), wp.vec3(0.0))
  return wp.vec4(r[0], r[1], r[2], t[3])


@wp.func
def data_texel_map(
  # In:
  affine: tex_affine,
  texel: wp.vec4,
) -> wp.vec4:
  """`texel_map` of a data texel (ORM, normal, opacity), saturated to [0, 1] like its texels."""
  t = texel_map(affine, texel)
  return wp.vec4(wp.clamp(t[0], 0.0, 1.0), wp.clamp(t[1], 0.0, 1.0), wp.clamp(t[2], 0.0, 1.0), wp.clamp(t[3], 0.0, 1.0))


@wp.func
def widen_roughness(
  # In:
  perceptual_roughness: float,
  alpha2_extra: float,
) -> float:
  """Perceptual roughness whose GGX α² is increased by `alpha2_extra` (capped at 1)."""
  pr = wp.clamp(perceptual_roughness, 0.0, 1.0)
  a2 = wp.min(pr * pr * pr * pr + alpha2_extra, 1.0)
  return wp.sqrt(wp.sqrt(a2))


@wp.func
@device_generic
def optional_texture[M: MaterialTables](
  # Model:
  mats: M,
  # In:
  mat_world: int,
  mat_id: int,
  role: int,
) -> int:
  """Texture id of an optional role (-1: unbound, or the prepared table is narrower)."""
  if mats.material_texture.shape[2] <= role:
    return -1
  return texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, role)


def make_diffuse_map(fetch: FetchTexel):
  """diffuse_map(mats, mat_world, affine_world, mat_id, uv, fp, uv_per_length, roughness) -> the EON
  roughness times the material's diffuse-roughness map (TexRole.DIFFUSE_ROUGHNESS, an optional role
  of the shared texture and texel-map tables)."""

  @wp.func
  @device_generic
  def diffuse_map[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    mat_world: int,
    affine_world: int,
    mat_id: int,
    uv: wp.vec2,
    fp: Footprint,
    uv_per_length: float,
    roughness: float,
  ) -> float:
    """EON roughness `roughness` times material `mat_id`'s diffuse-roughness map (unchanged for a
    smooth lobe, an unbound role or a prepared table without the role)."""
    if roughness <= 0.0 or mats.material_texture.shape[2] <= wp.static(_ROLE_DIFFUSE_ROUGHNESS):
      return roughness
    tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_DIFFUSE_ROUGHNESS)
    if tex_id < 0:
      return roughness
    affine = tex_affine(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    if mats.material_tex_affine.shape[2] > wp.static(_ROLE_DIFFUSE_ROUGHNESS):
      affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_DIFFUSE_ROUGHNESS]
    return roughness * data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))[0]

  return diffuse_map


def make_layer_maps(f: Features, fetch: FetchTexel):
  """layer_maps(...) -> `surface` with the layered roles' maps (Features.layered_maps): the
  anisotropy strength (x) and the fuzz color (rgb, hue-varied) and roughness (a) factors."""
  anisotropy = f.specular_anisotropy
  fuzz = f.fuzz

  @wp.func
  @device_generic
  def layer_maps[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    mat_world: int,
    affine_world: int,
    mat_id: int,
    uv: wp.vec2,
    fp: Footprint,
    uv_per_length: float,
    hue: float,
    surface: Surface,
  ) -> Surface:
    """`surface` with the anisotropy / fuzz maps of material `mat_id` (as `sample_maps`)."""
    if wp.static(anisotropy):
      if surface.anisotropy != 0.0:
        tex_id = optional_texture(mats, mat_world, mat_id, _ROLE_ANISOTROPY)
        if tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_ANISOTROPY]
          surface.anisotropy = surface.anisotropy * data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))[0]
    if wp.static(fuzz):
      if surface.fuzz_weight > 0.0:
        tex_id = optional_texture(mats, mat_world, mat_id, _ROLE_FUZZ)
        if tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_FUZZ]
          texel = color_texel_map(affine, hue, fetch(mats, tex_id, uv, fp, uv_per_length))
          surface.fuzz_color = wp.cw_mul(surface.fuzz_color, wp.vec3(texel[0], texel[1], texel[2]))
          surface.fuzz_alpha = surface.fuzz_alpha * wp.clamp(texel[3], 0.0, 1.0)
    return surface

  return layer_maps


def make_sample_maps(f: Features, fetch: FetchTexel):  # noqa: C901, PLR0915 (one static branch and class gate per lobe / role)
  """sample_maps(...) -> (surface, emission) with the material's color / ORM / opacity /
  emissive maps applied (each compiled in with its `Features` flag)."""
  base_color_map = f.base_color_map
  orm_map = f.orm_map
  opacity_map = f.opacity_map
  diffuse_map = make_diffuse_map(fetch)
  emissive_map = f.emissive_map
  layer_maps = make_layer_maps(f, fetch)
  gate = toggles.bound_on(f.bound)  # the base-map, maps and layer-maps class bits
  field = toggles.bound_on(f.bound, high=True)  # each map slot's field bit (`toggles.MAP_*`)

  @wp.func
  @device_generic
  def sample_maps[M: MaterialTables](  # noqa: C901 (one static branch and class gate per lobe / role)
    # Model:
    mats: M,
    # In:
    mat_world: int,
    affine_world: int,
    mat_id: int,
    uv: wp.vec2,
    fp: Footprint,
    uv_per_length: float,
    hue: float,
    surface: Surface,
    emission: float,
  ) -> tuple[Surface, float]:
    """`surface` (and its emission strength, 0 once an emissive map took it) with the maps.

    `mat_world` / `affine_world`: the rows of the texture and texel-map tables of the hit's world;
    `uv` / `fp`: varied (`vary_uv`); `hue`: the color roles' hue rotation.
    """
    other_maps = False
    if wp.static(orm_map or opacity_map or emissive_map or f.diffuse_roughness_map):
      other_maps = gate(surface.bits, wp.static(toggles.CLASS_MAPS))
    if wp.static(base_color_map):
      if gate(surface.bits, wp.static(toggles.CLASS_BASE_MAP)):
        role = _ROLE_RGB
        tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_RGB)
        if tex_id < 0:
          role = _ROLE_RGBA
          tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_RGBA)
        if tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, role]
          texel = color_texel_map(affine, hue, fetch(mats, tex_id, uv, fp, uv_per_length))
          surface.base_color = wp.cw_mul(surface.base_color, wp.vec3(texel[0], texel[1], texel[2]))
          surface.alpha = surface.alpha * texel[3]

    if wp.static(orm_map):
      if other_maps and field(surface.bits2, wp.static(toggles.MAP_ORM)):
        tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_ORM)
        if tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_ORM]
          texel = data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))
          surface.occlusion = texel[0]
          surface.perceptual_roughness = surface.perceptual_roughness * texel[1]
          surface.metallic = surface.metallic * texel[2]
        else:
          tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_OCCLUSION)
          if tex_id >= 0:
            affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_OCCLUSION]
            surface.occlusion = data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))[0]
          tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_ROUGHNESS)
          if tex_id >= 0:
            affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_ROUGHNESS]
            surface.perceptual_roughness = (
              surface.perceptual_roughness * data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))[0]
            )
          tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_METALLIC)
          if tex_id >= 0:
            affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_METALLIC]
            surface.metallic = surface.metallic * data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))[0]

    if wp.static(f.diffuse_roughness_map):
      if other_maps and field(surface.bits2, wp.static(toggles.MAP_DIFFUSE_ROUGHNESS)):
        surface.diffuse_roughness = diffuse_map(
          mats, mat_world, affine_world, mat_id, uv, fp, uv_per_length, surface.diffuse_roughness
        )

    if wp.static(f.layered_maps):
      if gate(surface.bits, wp.static(toggles.CLASS_LAYER_MAPS)):
        surface = layer_maps(mats, mat_world, affine_world, mat_id, uv, fp, uv_per_length, hue, surface)

    if wp.static(opacity_map):
      if other_maps and field(surface.bits2, wp.static(toggles.MAP_OPACITY)):
        tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_OPACITY)
        if tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_OPACITY]
          surface.alpha = surface.alpha * data_texel_map(affine, fetch(mats, tex_id, uv, fp, uv_per_length))[0]

    if wp.static(emissive_map):
      if other_maps and field(surface.bits2, wp.static(toggles.MAP_EMISSIVE)):
        tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_EMISSIVE)
        if tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_EMISSIVE]
          texel = color_texel_map(affine, hue, fetch(mats, tex_id, uv, fp, uv_per_length))
          surface.emissive = wp.vec3(texel[0], texel[1], texel[2]) * emission
          emission = 0.0
    return surface, emission

  return sample_maps
