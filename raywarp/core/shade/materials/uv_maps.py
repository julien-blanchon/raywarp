"""The tangent-space normal map and every map at a hit's texture frame.

make_normal_map   the tangent-space normal map (Toksvig roughness, back faces); its
                  per-axis strengths / offsets variant (`Features.scaled_normal_map`)
make_uv_maps      all maps at a hit's texture frame (`uv_frame.make_apply_maps`,
                  `shade/materials/flex.py`)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import warp as wp

from raywarp._base.types import TexRole, tex_affine
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.abi.tables import batched, table_row, texture_id
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.materials import maps, texture
from raywarp.core.shade.materials.material import LayeredMaterials, MaterialTables, Surface

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_ROLE_NORMAL: Final = int(TexRole.NORMAL)
_ROLE_COAT_NORMAL: Final = int(TexRole.COAT_NORMAL)
_ROLE_DETAIL_NORMAL: Final = int(TexRole.DETAIL_NORMAL)
# Length error of an 8-bit normal-map texel: each component of the [-1, 1] encoding rounds to a
# step of 2/255, so a unit normal decodes up to sqrt(3)/255 short; one full step (2/255) also
# covers the bilinear blend of rounded texels. Toksvig ignores shortening below it.
_NORMAL_QUANTISATION: Final = 2.0 / 255.0


type NormalMaps = Callable[
  [
    MaterialTables,
    int,
    int,
    int,
    wp.vec2,
    maps.Footprint,
    float,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    wp.vec3,
    float,
    bool,
    Surface,
  ],
  tuple[Surface, wp.vec3],
]


@wp.func
def _toksvig_roughness(
  # In:
  perceptual_roughness: float,
  nts: wp.vec3,
) -> float:
  """Perceptual roughness widened by the shortening of a filtered tangent-space normal (Toksvig).

  Mip / bilinear filtering averages normals, and their spread shortens the average: the variance
  (1 - |n|) / |n| is added to α². Normal maps are 8-bit at every mip level, so a texel is short
  by up to _NORMAL_QUANTISATION from rounding alone: that much is not spread and is ignored.
  """
  length = wp.clamp(wp.length(nts) + wp.static(_NORMAL_QUANTISATION), 1.0e-3, 1.0)
  return maps.widen_roughness(perceptual_roughness, (1.0 - length) / length)


def make_normal_map(specular_aa: bool, scaled: bool, fetch: maps.FetchTexel):
  """normal_map(...) -> (surface, shading normal) with the tangent-space normal map applied.

  `scaled`: Features.scaled_normal_map, some material's normal texel map changes the decoded
  length (a strength, an offset: `pipeline.upload.materials.scaled_normal_maps`): Toksvig reads the
  fetched texel's own length (the filtered map's spread, whatever the strength), and the texel map
  applies unclamped (a strength above 1 decodes beyond [-1, 1]); without it the code is the plain
  decode's (unit per-axis scales: the unpack and its flips)."""

  @wp.func
  @device_generic
  def normal_map[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    tex_id: int,
    affine: tex_affine,
    uv: wp.vec2,
    fp: maps.Footprint,
    uv_per_length: float,
    tangent: wp.vec3,
    bitangent: wp.vec3,
    shading: wp.vec3,
    local_normal: wp.vec3,
    side: float,
    aa: bool,
    surface: Surface,
  ) -> tuple[Surface, wp.vec3]:
    """`surface` (Toksvig-widened roughness with specular AA on, `aa`) and the mapped normal (else
    `shading`).

    `side`: +1 front, -1 back (the back's normal is minus the front's mapped one);
    `affine`: the material's normal texel map (its binding's per-axis scales / flips:
    `pipeline.upload.materials.normal_rows`).
    """
    raw = fetch(mats, tex_id, uv, fp, uv_per_length)
    if wp.static(scaled):
      texel = maps.texel_map(affine, raw)
      if wp.static(specular_aa):
        if aa:
          spread = wp.vec3(raw[0], raw[1], raw[2]) * 2.0 - wp.vec3(1.0)
          surface.perceptual_roughness = _toksvig_roughness(surface.perceptual_roughness, spread)
      nts = wp.vec3(texel[0], texel[1], texel[2]) * 2.0 - wp.vec3(1.0)
    else:
      texel = maps.data_texel_map(affine, raw)
      nts = wp.vec3(texel[0], texel[1], texel[2]) * 2.0 - wp.vec3(1.0)
      if wp.static(specular_aa):
        if aa:
          surface.perceptual_roughness = _toksvig_roughness(surface.perceptual_roughness, nts)
    return surface, _mapped_normal(nts, tangent, bitangent, shading, local_normal, side)

  return normal_map


@wp.func
def _mapped_normal(
  # In:
  nts: wp.vec3,
  tangent: wp.vec3,
  bitangent: wp.vec3,
  shading: wp.vec3,
  local_normal: wp.vec3,
  side: float,
) -> wp.vec3:
  """The shading normal bent by the decoded tangent-space normal `nts` (else `shading`).

  The device's frame is (T, -B, N): UVs with a top-left origin (MuJoCo) and +Y as image up put
  tangent +y along -dp/dv; host scenes invert the stored green, which compensates their vertical
  image flip (scene/model/normal_map.py: +y along +dp/dv). From the back the relief is inverted;
  the result stays in the visible hemisphere of the geometric normal `local_normal`.
  """
  t = tangent - shading * wp.dot(shading, tangent)
  if wp.length_sq(t) <= 1.0e-20:
    return shading
  t = wp.normalize(t)
  b = wp.cross(shading, t)
  if wp.dot(b, bitangent) > 0.0:
    b = -b
  mapped = wp.normalize((t * nts[0] + b * nts[1]) * side + shading * nts[2])
  if wp.dot(mapped, local_normal) > 0.0:
    return mapped
  return shading


@wp.func
def _decoded(
  # In:
  raw: wp.vec4,
) -> wp.vec3:
  """A fetched normal texel's unit-scale decode 2 t - 1 (its length: the filtered spread)."""
  return wp.vec3(raw[0], raw[1], raw[2]) * 2.0 - wp.vec3(1.0)


@wp.func
def reoriented_normal(
  # In:
  base: wp.vec3,
  detail: wp.vec3,
) -> wp.vec3:
  """Reoriented normal mapping (Barré-Brisebois & Hill 2012; Filament listing 40): the detail
  tangent-space normal `detail` turned from +z onto `base` (both unit: the result is unit; each
  map's Toksvig reads its own fetched length, `make_layer_normals`). A flat detail (0, 0, 1)
  returns `base`, a flat base returns `detail`."""
  t = base + wp.vec3(0.0, 0.0, 1.0)
  u = wp.vec3(-detail[0], -detail[1], detail[2])
  return t * (wp.dot(t, u) / wp.max(t[2], 1.0e-6)) - u


def make_layer_normals(specular_aa: bool, *, detail: bool, coat: bool, fetch: maps.FetchTexel):
  """layer_normals(...) -> (surface, shading normal): the base normal map with its detail map
  (Features.detail_normal: RNM, the detail's own UV transform `LayerConstants.detail_uv`) and
  the coat's own normal map (Features.coat_normal: from the unmapped normal, its Toksvig on the
  coat roughness), for `LayeredMaterials`."""

  @wp.func
  def layer_normals(
    # Model:
    mats: LayeredMaterials,
    # In:
    mat_world: int,
    affine_world: int,
    mat_id: int,
    uv: wp.vec2,
    fp: maps.Footprint,
    uv_per_length: float,
    tangent: wp.vec3,
    bitangent: wp.vec3,
    shading: wp.vec3,
    local_normal: wp.vec3,
    side: float,
    aa: bool,
    surface: Surface,
  ) -> tuple[Surface, wp.vec3]:
    """`surface` (Toksvig-widened roughness with specular AA on, `aa`; the coat's normal) and the
    mapped normal."""
    unmapped = shading
    nts = wp.vec3(0.0, 0.0, 1.0)
    mapped = False
    tex_id = maps.optional_texture(mats, mat_world, mat_id, wp.static(_ROLE_NORMAL))
    if tex_id >= 0:
      affine = mats.material_tex_affine[affine_world, mat_id, wp.static(_ROLE_NORMAL)]
      raw = fetch(mats, tex_id, uv, fp, uv_per_length)
      texel = maps.texel_map(affine, raw)
      nts = wp.vec3(texel[0], texel[1], texel[2]) * 2.0 - wp.vec3(1.0)
      if wp.static(specular_aa):  # each map's filtered spread adds its variance to α²
        if aa:
          surface.perceptual_roughness = _toksvig_roughness(surface.perceptual_roughness, _decoded(raw))
      mapped = True
    if wp.static(detail):
      tex_id = maps.optional_texture(mats, mat_world, mat_id, wp.static(_ROLE_DETAIL_NORMAL))
      if tex_id >= 0:
        xf = mats.material_layer[mat_id].detail_uv
        affine = mats.material_tex_affine[affine_world, mat_id, wp.static(_ROLE_DETAIL_NORMAL)]
        # The detail's footprint: the base's through the detail transform's linear part.
        dfp = maps.Footprint()
        scale = wp.sqrt(wp.abs(xf[0, 0] * xf[1, 1] - xf[0, 1] * xf[1, 0]))
        dfp.cone = fp.cone
        dfp.taps = fp.taps
        dfp.major = wp.vec2(xf[0, 0] * fp.major[0] + xf[0, 1] * fp.major[1], xf[1, 0] * fp.major[0] + xf[1, 1] * fp.major[1])
        dfp.minor = wp.vec2(xf[0, 0] * fp.minor[0] + xf[0, 1] * fp.minor[1], xf[1, 0] * fp.minor[0] + xf[1, 1] * fp.minor[1])
        raw = fetch(mats, tex_id, texture.affine_uv(xf, uv), dfp, uv_per_length * scale)
        texel = maps.texel_map(affine, raw)
        if wp.static(specular_aa):
          if aa:
            surface.perceptual_roughness = _toksvig_roughness(surface.perceptual_roughness, _decoded(raw))
        nts = reoriented_normal(wp.normalize(nts), wp.normalize(wp.vec3(texel[0], texel[1], texel[2]) * 2.0 - wp.vec3(1.0)))
        mapped = True
    if mapped:
      shading = _mapped_normal(nts, tangent, bitangent, shading, local_normal, side)
    if wp.static(coat):
      tex_id = maps.optional_texture(mats, mat_world, mat_id, wp.static(_ROLE_COAT_NORMAL))
      if tex_id >= 0:
        affine = mats.material_tex_affine[affine_world, mat_id, wp.static(_ROLE_COAT_NORMAL)]
        raw = fetch(mats, tex_id, uv, fp, uv_per_length)
        texel = maps.texel_map(affine, raw)
        cts = wp.vec3(texel[0], texel[1], texel[2]) * 2.0 - wp.vec3(1.0)
        if wp.static(specular_aa):
          if aa:
            surface.clearcoat_roughness = _toksvig_roughness(surface.clearcoat_roughness, _decoded(raw))
        surface.coat_normal_local = _mapped_normal(cts, tangent, bitangent, unmapped, local_normal, side)
    return surface, shading

  # The caller's carrier is `LayeredMaterials` under the same Features (`material_type`).
  return cast(NormalMaps, layer_normals)


@specialize
def make_taps(f: Features):
  """taps(geo, mats) -> the anisotropic filter's most taps at a hit (`maps.Footprint.taps`): the
  settings' (`MaterialTables.texture_taps`) in a tier build where its `toggles.ANISOTROPY` is on
  (`tiers.aniso`) or, below its own, its `toggles.RUN_COUNTS` (`tiers.below`), else the build's
  own."""
  runtime = toggles.reads(f, toggles.ANISOTROPY)
  below = toggles.reads(f, toggles.RUN_COUNTS) and f.anisotropy > 1
  setting = toggles.setting_on(f)
  own = f.anisotropy

  @wp.func
  @device_generic
  def taps[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
  ) -> int:
    """The hit's most texture taps (see `make_taps`)."""
    if wp.static(runtime):
      if setting(geo.rt.flags, geo.bits2, wp.static(toggles.ANISOTROPY)):
        return mats.texture_taps
    if wp.static(below):
      if setting(geo.rt.flags, geo.bits2, wp.static(toggles.RUN_COUNTS)):
        return wp.min(mats.texture_taps, wp.static(own))
    return wp.static(own)

  return taps


def make_uv_maps(f: Features):
  """uv_maps(...) -> (surface, shading normal, emission): the material's maps at a hit whose
  texture frame is known (`uv_frame.make_apply_maps` for instances, `shade/materials/flex.py` for
  flex)."""
  normal_maps = f.normal_map
  # The most taps compiled: the build's, the capacity where it reads them at run time.
  most = toggles.TAPS_CAPACITY if toggles.reads(f, toggles.ANISOTROPY) else f.anisotropy
  footprint = maps.make_footprint(most)
  fetch = maps.make_fetch(most)
  sample_maps = maps.make_sample_maps(f, fetch)
  normal_map = make_normal_map(f.specular_aa, f.scaled_normal_map, fetch)
  layer_normals = f.detail_normal or f.coat_normal_map
  composed = make_layer_normals(f.specular_aa, detail=f.detail_normal, coat=f.coat_normal_map, fetch=fetch)
  gate = toggles.bound_on(f.bound)  # the maps and layer-maps class bits
  field = toggles.bound_on(f.bound, high=True)  # the normal map's field bit (`toggles.MAP_NORMAL`)

  @wp.func
  @device_generic
  def uv_maps[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    worldid: int,
    mat_id: int,
    has_uv: bool,
    uv: wp.vec2,
    tangent: wp.vec3,
    bitangent: wp.vec3,
    uv_per_length: float,
    cone: float,
    width: float,
    local_dir: wp.vec3,
    local_normal: wp.vec3,
    side: float,
    shading: wp.vec3,
    aa: bool,
    taps: int,
    surface: Surface,
    emission: float,
  ) -> tuple[Surface, wp.vec3, float]:
    """The maps at texture frame (uv, tangent = dp/du, bitangent = dp/dv) of the hit (shape frame:
    `local_dir` the ray, `local_normal` the viewer-facing geometric normal); `width`: the ray
    cone's width at the hit, `cone` its footprint on the surface; `aa`: specular AA on (the
    normal maps' Toksvig widening), `taps`: the anisotropic filter's most taps (`make_taps`)."""
    mat_world = table_row(worldid, mats.material_texture.shape[0])
    affine_world = table_row(worldid, mats.material_tex_affine.shape[0])
    variation = batched(mats.material_tex_variation, worldid, mat_id)
    # Without UVs a map reads its texture's average (coarsest mip: a cone past any texture, no
    # frame): a constant role applies exactly (a 1 x 1 texture, e.g. a constant emissive color).
    uv_cone = float(1.0e30)
    if has_uv:
      # The material's texture variation: every role's UVs moved alike (its maps stay aligned),
      # the frame with them (footprint, normal map).
      uv, tangent, bitangent = maps.vary_uv(variation, uv, tangent, bitangent)
      uv_cone = cone
    else:
      uv = wp.vec2(0.5)
      uv_per_length = 1.0
    fp = footprint(uv_cone, width, local_dir, local_normal, tangent, bitangent)
    fp.taps = taps
    surface, emission = sample_maps(
      mats, mat_world, affine_world, mat_id, uv, fp, uv_per_length, variation[3], surface, emission
    )
    # The layered normals (base, detail and coat maps) with the layer-maps class, else the base's.
    layered = False
    if wp.static(layer_normals):
      layered = gate(mats.bits, wp.static(toggles.CLASS_LAYER_MAPS))
    if layered:
      if has_uv:
        surface, shading = composed(
          mats, mat_world, affine_world, mat_id, uv, fp, uv_per_length, tangent, bitangent, shading, local_normal, side, aa,
          surface,
        )  # fmt: skip
    elif wp.static(normal_maps):
      if gate(mats.bits, wp.static(toggles.CLASS_MAPS)) and field(surface.bits2, wp.static(toggles.MAP_NORMAL)):
        tex_id = texture_id(mats.material_texture, mats.textures.shape[0], mat_world, mat_id, _ROLE_NORMAL)
        if has_uv and tex_id >= 0:
          affine = mats.material_tex_affine[affine_world, mat_id, _ROLE_NORMAL]
          surface, shading = normal_map(
            mats, tex_id, affine, uv, fp, uv_per_length, tangent, bitangent, shading, local_normal, side, aa, surface
          )
    return surface, shading, emission

  return uv_maps
