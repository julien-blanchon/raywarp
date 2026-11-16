"""Host materials -> core material tables, textures and the instances' material tables.

The device values come from `derive.derive` (the OpenPBR inputs -> base color, opacity,
roughness, reflectance, coat, emission: the one place they are converted); this module packs the
textures and texel maps around them. Texture slots become core textures
(`pool.TexturePool`): base color -> RGB (mips weighted by the opacity map for cut-out /
blended materials), metalness / roughness / occlusion -> one packed ORM texture, normal ->
NORMAL, emission color -> EMISSIVE (a constant emission color reads a 1 x 1 white texture),
opacity -> OPACITY. The core textures hold only selected channels (normal-map green inverted:
images are flipped vertically; an opacity of negative scale inverted); every per-material factor
is the material's texel map per role (`material.tex_affine`: UsdUVTexture scale / bias, occlusion
strength, a normal map's per-axis strength / flip (`normal_rows`), the emission color as a tint),
applied where the texture is sampled, so materials sharing an image share its core texture.
A material samples every slot with one UV set and transform (`uv_plans.py`): a scale is its
`material.tex_repeat`, the rest of a transform and a second UV set are its mesh instances'
(`MeshInstances`). Emission stays physical [nits] (`Capabilities.physical_emission`).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np
import warp as wp

from raywarp._base.types import NLAYERED_TEXROLE, NOPENPBR_TEXROLE, NTEXROLE, TEX_IDENTITY, UV_IDENTITY, LayerFlag, TexRole
from raywarp.core.pipeline.derive.materials import LAYERED_INPUTS, MAPPED_INPUTS, Alpha, DerivedMaterials, base_shares, derive
from raywarp.core.pipeline.textures.formats import DEFAULT_STORAGE, TextureStorage
from raywarp.core.pipeline.textures.pool import ONE, Channel, Coverage, Recipe, TexturePool
from raywarp.core.pipeline.upload.optics import (
  InstanceOptics,
  MaterialOptics,
  OpticsInputs,
  compose,
  material_optics,
  optics_inputs,
)
from raywarp.core.pipeline.upload.uv_plans import bound_slots, first_bound_slot
from raywarp.scene.model.material import INPUTS
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.tables import MaterialExtras, Materials
from raywarp.scene.model.types import AlphaMode, Column, InstanceFlags, MaterialFlags, Matrices, Rows, Slot, TextureId

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

# Texel maps (`types.tex_affine`) of materials x slots: (m, NTEXROLE, 2, 4).
type TexAffines[NRoles: IntVar] = np.ndarray[[int, NRoles, 2, 4], np.dtype[np.float32]]
# One material's texel maps per slot: (NTEXROLE, 2, 4).
type RoleAffines[NRoles: IntVar] = np.ndarray[[NRoles, 2, 4], np.dtype[np.float32]]
# Texel map entries this close to a unit / flip map's (float32 rounding) are those.
_UNIT_TOL: Final = 1e-6
# An alpha-test cutoff no texel in [0, 1] passes.
_NEVER_PASSES: Final = 2.0
# Condition number of a UV transform's linear part above which it has no usable inverse (a zero
# scale): float32 UVs keep ~7 digits, so larger ratios lose the inverse's precision entirely.
_SINGULAR: Final = 1e7


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class MaterialTables[NRoles: IntVar]:
  """Core material tables of the host materials.

  Attributes:
    derived: the device values of the materials (`derive.derive`): base color and opacity
      (`derived.base`), metalness, roughness, reflectance, emission, coat, flags.
    pool: their core textures.
    texid: (m, NTEXROLE) core texture id per slot (-1: none).
    tex_affine: (m, NTEXROLE, 2, 4) texel map per slot ([scale, bias] rows, `types.tex_affine`).
    uniform: (m,) world-space (texuniform) UVs.
    alpha_map: (m,) an opacity map with texels below 1 (not a cut-out).
    see_alpha: (m,) the opacity factor of blended materials' see-through part (`_see_alpha`;
      -1: not blended): their shadow transmittance before the surface's Fresnel is
      `optics.see_through` of it and their base color.
    optics: the materials' optics (`optics.py`: refraction, volume absorption).
    cutoff: (m,) alpha-test cutoff on the opacity texture's texel (before its texel map; 0: none).
    detail_uv: (m, 2, 3) each detail normal map's UV transform on the material's base UVs.
    layer_flags: (m,) LayerFlag bits (`derive.Layers.flags` and the bound layered maps).
  """

  derived: DerivedMaterials
  pool: TexturePool
  texid: Rows[np.int32, NRoles]
  tex_affine: TexAffines[NRoles]
  uniform: Column[np.bool_]
  alpha_map: Column[np.bool_]
  see_alpha: Column[np.float32]
  optics: MaterialOptics
  cutoff: Column[np.float32]
  detail_uv: Matrices[np.float32, 2, 3]
  layer_flags: Column[np.int32]

  def scaled_normal_maps(self) -> Column[np.bool_]:
    """The normal-map mask with this table's matched texture/affine role width."""
    return scaled_normal_maps(self.texid, self.tex_affine)


type PreparedMaterials = MaterialTables[10] | MaterialTables[11] | MaterialTables[15]


class _Mapped(NamedTuple):
  """A slot component: its recipe channel and the texel map scale * texel + bias applied to it."""

  channel: Channel
  scale: float
  bias: float


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Bindings:
  """The texture bindings of host material `i`: its slots (`Materials.tex*` row i, bound to one of
  the `ntex` host textures) and its extra inputs' maps (`extra`, `_extra_maps`)."""

  mats: Materials
  ntex: int
  i: int
  extra: dict[str, _Extra]

  def texture(self, slot: Slot) -> TextureId | None:
    """The host texture bound to `slot`, or None."""
    t = TextureId(int(self.mats.tex[self.i, slot]))
    return t if 0 <= t < self.ntex else None

  def channel(self, slot: Slot, ch: int) -> _Mapped | None:
    """Slot `slot`'s component `ch` (UsdUVTexture scale / bias), or None if unbound."""
    t = self.texture(slot)
    if t is None:
      return None
    mats, i = self.mats, self.i
    c = int(mats.tex_channel[i, slot])
    s, o = float(mats.tex_scale[i, slot, min(ch, 3)]), float(mats.tex_bias[i, slot, min(ch, 3)])
    return _Mapped(Channel(t, ch if c < 0 else c), s, o)

  def rgb(self, slot: Slot) -> tuple[_Mapped, _Mapped, _Mapped] | None:
    """Slot `slot`'s three rgb components when the slot is bound, else None."""
    r, g, b = (self.channel(slot, k) for k in range(3))
    return None if r is None or g is None or b is None else (r, g, b)

  def opacity(self) -> _Mapped | None:
    """The opacity component with a non-negative scale (a negative one inverts the channel:
    s t + o = -s (1 - t) + s + o), so the alpha test is a threshold on the texel (`_cutoff`)."""
    m = self.channel(Slot.GEOMETRY_OPACITY, 0)
    if m is None or m.scale >= 0.0:
      return m
    return _Mapped(m.channel._replace(invert=True), -m.scale, m.bias + m.scale)


def _cutoff(alpha: Alpha, opacity: _Mapped | None) -> float | None:
  """The alpha-test cutoff on the opacity texel t, passing where scale * t + bias >= the cutoff
  (0: no test; None: every texel passes, the surface is opaque).
  """
  if alpha.cutoff <= 0.0 or opacity is None:
    return alpha.cutoff
  if opacity.scale <= 0.0:  # a constant opacity: it passes everywhere or nowhere
    return None if opacity.bias >= alpha.cutoff else _NEVER_PASSES
  cut = (alpha.cutoff - opacity.bias) / opacity.scale
  return cut if cut > 0.0 else None


def _normal_recipe(texture: TextureId) -> Recipe:
  """The recipe of a normal map: its rgb channels, green inverted (`normal_rows`)."""
  return (Channel(texture, 0), Channel(texture, 1, invert=True), Channel(texture, 2), ONE)


def _mapped_rows(
  mapped: Sequence[_Mapped | None], scale: float = 1.0, bias: float = 0.0
) -> np.ndarray[[2, 4], np.dtype[np.float32]]:
  """The texel map (2, 4) of up to four slot components, each then scaled by `scale` and offset by
  `bias` (unbound components and the rest: identity).
  """
  rows = TEX_IDENTITY.copy()
  for k, m in enumerate(mapped):
    if m is not None:
      rows[:, k] = (scale * m.scale, scale * m.bias + bias)
  return rows


def _register_textures[NRoles: IntVar](
  pool: TexturePool,
  bound: _Bindings,
  *,
  nroles: Int[NRoles],
  opacity: _Mapped | None,
  tint: np.ndarray[[3], np.dtype[np.float32]] | None,
) -> tuple[np.ndarray[[NRoles], np.dtype[np.int32]], RoleAffines[NRoles]]:
  """(core texture id per slot, texel map per slot) of a material's `bound` textures, registered
  in `pool`; `opacity`: its opacity channel (`_Bindings.opacity`); `tint`: its derived emission
  tint (None: it emits nothing).
  """
  mats, i, extra = bound.mats, bound.i, bound.extra
  texid = np.full(nroles, -1, np.int32)
  affine = np.tile(TEX_IDENTITY, (nroles, 1, 1))
  blended = int(mats.ext_alpha_mode[i]) != AlphaMode.OPAQUE
  # Base color: a cut-out's or blended surface's mips weighted by its opacity map.
  rgb = bound.rgb(Slot.BASE_COLOR)
  if rgb is not None:
    coverage = Coverage(*opacity) if blended and opacity is not None else None
    texid[TexRole.RGB] = pool.get((rgb[0].channel, rgb[1].channel, rgb[2].channel, ONE), color=True, coverage=coverage)
    affine[TexRole.RGB] = _mapped_rows(rgb)
  # Metallic / roughness / occlusion -> ORM; occlusion o -> strength o + 1 - strength.
  orm = [bound.channel(s, 0) for s in (Slot.OCCLUSION, Slot.SPECULAR_ROUGHNESS, Slot.BASE_METALNESS)]
  if any(m is not None for m in orm):
    chans = [ONE if m is None else m.channel for m in orm]
    texid[TexRole.ORM] = pool.get((chans[0], chans[1], chans[2], ONE), color=False)
    strength = float(mats.ext_occlusion_strength[i])
    affine[TexRole.ORM] = _mapped_rows(orm)
    affine[TexRole.ORM, :, 0] = _mapped_rows(orm[:1], strength, 1.0 - strength)[:, 0]
  # Normal map (green inverted: `normal_rows`), its binding's scale / bias in the texel map.
  nrm = bound.texture(Slot.GEOMETRY_NORMAL)
  if nrm is not None:
    texid[TexRole.NORMAL] = pool.get(_normal_recipe(nrm), color=False)
    affine[TexRole.NORMAL] = normal_rows(mats.tex_scale[i, Slot.GEOMETRY_NORMAL], mats.tex_bias[i, Slot.GEOMETRY_NORMAL])
  if tint is not None:
    texid[TexRole.EMISSIVE], affine[TexRole.EMISSIVE] = _register_emission(pool, bound, tint=tint)
  # Opacity (blend / mask only).
  if blended and opacity is not None:
    texid[TexRole.OPACITY] = pool.get((opacity.channel, ONE, ONE, ONE), color=False, red_range=True)
    affine[TexRole.OPACITY] = _mapped_rows([opacity])
  diffuse = extra.get("base_diffuse_roughness")
  if diffuse is not None:
    texid[TexRole.DIFFUSE_ROUGHNESS] = pool.get((diffuse.channel(0), ONE, ONE, ONE), color=False)
    affine[TexRole.DIFFUSE_ROUGHNESS] = _mapped_rows([diffuse.mapped(0)])
  if nroles >= NLAYERED_TEXROLE:
    _register_layers(pool, extra, texid, affine)
  return texid, affine


def _register_layers(
  pool: TexturePool,
  extra: dict[str, _Extra],
  texid: np.ndarray[[int], np.dtype[np.int32]],
  affine: RoleAffines[int],
) -> None:
  """The layered roles of one material (`texid` / `affine` filled in place): the anisotropy
  strength (x), the fuzz color (rgb, sRGB-decoded) and roughness (a), the coat and detail normal
  maps (green inverted, their bindings' strengths in the texel map: `normal_rows`)."""
  aniso = extra.get("specular_roughness_anisotropy")
  if aniso is not None:
    texid[TexRole.ANISOTROPY] = pool.get((aniso.channel(0), ONE, ONE, ONE), color=False)
    affine[TexRole.ANISOTROPY] = _mapped_rows([aniso.mapped(0)])
  color, rough = extra.get("fuzz_color"), extra.get("fuzz_roughness")
  if color is not None or rough is not None:
    r, g, b = (ONE, ONE, ONE) if color is None else (color.channel(0), color.channel(1), color.channel(2))
    texid[TexRole.FUZZ] = pool.get((r, g, b, ONE if rough is None else rough.channel(0)), color=color is not None)
    rows = _mapped_rows([None if color is None else color.mapped(k) for k in range(3)])
    if rough is not None:
      rows[:, 3] = _mapped_rows([rough.mapped(0)])[:, 0]
    affine[TexRole.FUZZ] = rows
  for name, role in (("geometry_coat_normal", TexRole.COAT_NORMAL), ("geometry_detail_normal", TexRole.DETAIL_NORMAL)):
    ref = extra.get(name)
    if ref is not None:
      texid[role] = pool.get(_normal_recipe(ref.texture), color=False)
      affine[role] = normal_rows(ref.scale, ref.bias)


def normal_rows(
  scale: np.ndarray[[4], np.dtype[np.float32]], bias: np.ndarray[[4], np.dtype[np.float32]]
) -> np.ndarray[[2, 4], np.dtype[np.float32]]:
  """The texel map (2, 4) of a normal-map binding whose tangent-space normal is n = scale x
  texel + bias per component (scene/model/normal_map.py: any per-axis strength or flip, exactly).

  The device decodes a mapped texel v as 2 v - 1 along (T, -B, N) (`maps.normal_map`: its
  bitangent runs along -dp/dv, MuJoCo's top-left UVs), and host images are flipped vertically, so
  the core texture's green is inverted (t' = 1 - t): v = (s t + b + 1) / 2 for x and z, and
  2 v - 1 = -(s (1 - t') + b) for y. The plain unpack (scale 2, bias -1) gives the identity map
  and the green inversion; a strength k (scale 2 k, bias -k) gives k t + (1 - k) / 2.
  """
  s, b = scale.astype(np.float64), bias.astype(np.float64)
  rows = TEX_IDENTITY.astype(np.float64)
  rows[0, :3] = 0.5 * s[:3]
  rows[1, 0], rows[1, 2] = 0.5 * (b[0] + 1.0), 0.5 * (b[2] + 1.0)
  rows[1, 1] = 0.5 * (1.0 - (s[1] + b[1]))
  return rows.astype(np.float32)


def scaled_normal_maps[NRoles: IntVar](texid: Rows[np.int32, NRoles], tex_affine: TexAffines[NRoles]) -> Column[np.bool_]:
  """(m,) materials (`MaterialTables.texid` / `tex_affine`) whose normal texel map changes the
  decoded length: a per-axis strength or offset (the plain unpack and its flips map each
  component's texel by (1, 0) or (-1, 1) and keep the length). `Features.scaled_normal_map`
  compiles their decode in: Toksvig on the fetched texel, the texel map unclamped."""
  rows = tex_affine[:, TexRole.NORMAL, :, :3].astype(np.float64)
  unit = (np.abs(rows[:, 0] - 1.0) < _UNIT_TOL) & (np.abs(rows[:, 1]) < _UNIT_TOL)
  flip = (np.abs(rows[:, 0] + 1.0) < _UNIT_TOL) & (np.abs(rows[:, 1] - 1.0) < _UNIT_TOL)
  return ~np.all(unit | flip, axis=1) & (texid[:, TexRole.NORMAL] >= 0)


def _register_emission(
  pool: TexturePool, bound: _Bindings, *, tint: np.ndarray[[3], np.dtype[np.float32]]
) -> tuple[int, np.ndarray[[2, 4], np.dtype[np.float32]]]:
  """(core texture id, texel map) of a material's emission: its emissive texture (a 1 x 1 white
  one for a constant color) tinted by the derived `tint` (emitted color / peak)."""
  emissive = bound.rgb(Slot.EMISSION_COLOR)
  if emissive is None:
    rows = TEX_IDENTITY.copy()
    rows[0, :3] = tint
    return pool.get((ONE, ONE, ONE, ONE), color=True), rows
  texture = pool.get((emissive[0].channel, emissive[1].channel, emissive[2].channel, ONE), color=True, hdr=True)
  rows = _mapped_rows(emissive)
  rows[:, :3] *= tint
  return texture, rows


def material_tables(
  scene: Scene, report: LoadReport, *, device: wp.Device, storage: TextureStorage = DEFAULT_STORAGE
) -> PreparedMaterials:
  """The core material tables of `scene` on `device` (`MaterialTables`): `NTEXROLE` texture roles
  wide, `NOPENPBR_TEXROLE` when a diffuse-roughness map is bound, `NLAYERED_TEXROLE` when a
  layered map is bound (`LAYERED_INPUTS`)."""
  mapped = _extra_maps(scene.material_extras, ntexture=len(scene.textures))
  inputs = {name for maps in mapped.values() for name in maps}
  table = functools.partial(_material_tables, scene, report, device=device, storage=storage, mapped=mapped)
  if inputs & set(LAYERED_INPUTS):
    return table(nroles=NLAYERED_TEXROLE)
  if "base_diffuse_roughness" in inputs:
    return table(nroles=NOPENPBR_TEXROLE)
  return table(nroles=NTEXROLE)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class _Extra:
  """A texture bound to an extra OpenPBR input (one `MaterialExtras` texture row).

  Attributes:
    texture: its texture id.
    selected: its channel (-1: CHANNEL_RGB, the components in order).
    scale, bias: per-component texel map (UsdUVTexture scale / bias).
    uv_xform: its UV transform (2, 3).
  """

  texture: TextureId
  selected: int
  scale: np.ndarray[[4], np.dtype[np.float32]]
  bias: np.ndarray[[4], np.dtype[np.float32]]
  uv_xform: np.ndarray[[2, 3], np.dtype[np.float32]]

  def channel(self, k: int) -> Channel:
    """Recipe channel of component k (the selected channel broadcast, else channel k)."""
    return Channel(self.texture, k if self.selected < 0 else self.selected)

  def mapped(self, k: int) -> _Mapped:
    """Component k with its texel map."""
    return _Mapped(self.channel(k), float(self.scale[min(k, 3)]), float(self.bias[min(k, 3)]))


def _extra_maps(extras: MaterialExtras, *, ntexture: int) -> dict[int, dict[str, _Extra]]:
  """Material -> extra input -> its bound texture, for the extra inputs with a texture role."""
  out: dict[int, dict[str, _Extra]] = {}
  for i in range(len(extras.tex_input)):
    name = INPUTS[int(extras.tex_input[i])]
    if name not in MAPPED_INPUTS or not 0 <= int(extras.tex[i]) < ntexture:
      continue
    out.setdefault(int(extras.tex_material[i]), {})[name] = _Extra(
      texture=TextureId(int(extras.tex[i])),
      selected=int(extras.tex_channel[i]),
      scale=np.asarray(extras.tex_scale[i], np.float32),
      bias=np.asarray(extras.tex_bias[i], np.float32),
      uv_xform=np.asarray(extras.uv_xform[i], np.float32),
    )
  return out


def _fold_base_maps(affine: TexAffines[int], opaque: Rows[np.float64, 3], transmitted: Rows[np.float64, 3]) -> None:
  """Folds the base color's split (`derive.base_shares`) into the base-color texel maps of
  `affine` (in place): the device multiplies the base color S = opaque + transmitted by the
  texel, so a texel map tex -> (opaque tex + transmitted) / S makes the texture reach only the
  opaque share (OpenPBR's translucent base: at full transmission the texel is white and
  `transmission_color` alone tints). No device cost: the fold is in the existing texel map. A
  zero channel of S keeps its map (the product is 0 either way); opaque materials keep theirs
  (ratio 1, bias 0)."""
  total = opaque + transmitted
  folded = np.any(transmitted > 0.0, axis=1)
  safe = total > 0.0
  ratio = np.where(safe, opaque / np.where(safe, total, 1.0), 1.0)[folded]
  offset = np.where(safe, transmitted / np.where(safe, total, 1.0), 0.0)[folded]
  for role in (TexRole.RGB, TexRole.RGBA):
    scale, bias = affine[folded, role, 0, :3].astype(np.float64), affine[folded, role, 1, :3].astype(np.float64)
    affine[folded, role, 0, :3] = ratio * scale
    affine[folded, role, 1, :3] = ratio * bias + offset


def _material_tables[NRoles: IntVar](
  scene: Scene,
  report: LoadReport,
  *,
  device: wp.Device,
  storage: TextureStorage,
  nroles: Int[NRoles],
  mapped: dict[int, dict[str, _Extra]],
) -> MaterialTables[NRoles]:
  """The core material tables and textures on `device` of the host materials (`MaterialTables`;
  `storage`: how the textures are stored)."""
  mats = scene.materials
  m = len(mats)
  ntex = len(scene.textures)
  optics = material_optics(scene)
  derived = derive(mats, scene.material_extras, report, volume=optics.volume)
  pool = TexturePool(scene, report, storage, device=device)
  texid = np.full((m, nroles), -1, np.int32)
  affine = np.tile(TEX_IDENTITY, (m, nroles, 1, 1))
  cutoff = np.zeros(m, np.float32)
  for i in range(m):
    bound = _Bindings(mats=mats, ntex=ntex, i=i, extra=mapped.get(i, {}))
    opacity = bound.opacity()
    tint = derived.emission_tint[i] if derived.emission[i] > 0.0 else None
    texid[i], affine[i] = _register_textures(pool, bound, nroles=nroles, opacity=opacity, tint=tint)
    cut = _cutoff(derived.alpha[i], opacity)
    if cut is None:  # every texel passes: an opaque surface
      affine[i, TexRole.OPACITY, :2] = [[0.0] * 4, [1.0] * 4]
    cutoff[i] = cut or 0.0
  pool.finish()
  _fold_base_maps(affine, *base_shares(mats, scene.material_extras, optics.volume))
  min_alpha = _min_alpha(pool, texid, affine)
  return MaterialTables(
    derived=derived,
    pool=pool,
    texid=texid,
    tex_affine=affine,
    uniform=(np.asarray(mats.ext_flags, np.int64) & int(MaterialFlags.UV_WORLD)) != 0,
    alpha_map=(texid[:, TexRole.OPACITY] >= 0) & (min_alpha < 1.0) & (cutoff <= 0.0),
    see_alpha=_see_alpha(derived, min_alpha),
    optics=optics,
    cutoff=cutoff,
    detail_uv=_detail_uv(mats, len(scene.textures), mapped, report),
    layer_flags=_layer_flags(derived, texid),
  )


def _min_alpha[NRoles: IntVar](
  pool: TexturePool, texid: Rows[np.int32, NRoles], affine: TexAffines[NRoles]
) -> Column[np.float32]:
  """(m,) each material's smallest opacity texel through its texel map (scale >= 0), 1 without an
  opacity map; `pool` finished."""
  opacity_mapped = texid[:, TexRole.OPACITY] >= 0
  min_alpha = np.ones(len(texid), np.float32)
  low = np.array([pool.red_range.get(int(t), (0.0, 1.0))[0] for t in texid[opacity_mapped, TexRole.OPACITY]], np.float32)
  rows = affine[opacity_mapped, TexRole.OPACITY]
  min_alpha[opacity_mapped] = np.clip(rows[:, 0, 0] * low + rows[:, 1, 0], 0.0, 1.0)
  return min_alpha


def _see_alpha(derived: DerivedMaterials, min_alpha: Column[np.float32]) -> Column[np.float32]:
  """(m,) the opacity factor of blended (blend-mode or transmissive) materials: the opacity map's
  minimum `min_alpha` (the see-through part of an atlas, e.g. a window's glass next to its frame),
  -1 for the others. Their shadow transmittance before the surface's Fresnel is base color x
  (1 - opacity x it) (`optics.see_through`); the derived base color is the tint of colored glass
  (OpenPBR `transmission_color`). A transmissive material also reflects part of the light at its
  surface: shadow rays pass 1 - E_F(ior) of it (`optics.compose`; the hemispherical average
  Fresnel, as camera rays pass 1 - E_spec(NoV), `layer.composite_transmissive`)."""
  translucent = np.array([a.translucent for a in derived.alpha], bool)
  return np.where(translucent, min_alpha, -1.0).astype(np.float32)


def _layer_flags[NRoles: IntVar](derived: DerivedMaterials, texid: Rows[np.int32, NRoles]) -> Column[np.int32]:
  """(m,) LayerFlag bits: the derived ones and the bound coat / detail normal maps (layered
  tables only)."""
  flags = derived.layers.flags
  if texid.shape[1] >= NLAYERED_TEXROLE:
    flags = flags | np.where(texid[:, TexRole.COAT_NORMAL] >= 0, int(LayerFlag.COAT_NORMAL_MAP), 0)
    flags |= np.where(texid[:, TexRole.DETAIL_NORMAL] >= 0, int(LayerFlag.DETAIL_NORMAL), 0)
  return flags.astype(np.int32)


def _detail_uv(
  mats: Materials, ntex: int, mapped: dict[int, dict[str, _Extra]], report: LoadReport
) -> Matrices[np.float32, 2, 3]:
  """(m, 2, 3) the detail normal map's UV transform relative to its material's base UVs: D B^-1,
  D the detail binding's transform, B the material's (its base color slot's, else its first bound
  slot's of the `ntex` textures: `uv_plans.first_bound_slot`), as affine 3 x 3 maps (identity
  without a detail map). A singular B (a zero UV scale: every base UV one point) has no inverse:
  the detail map reads the base UVs (reported)."""
  out = np.tile(UV_IDENTITY, (len(mats), 1, 1))
  bound = bound_slots(np.asarray(mats.tex), ntex)
  first = first_bound_slot(bound)
  for i, maps in mapped.items():
    ref = maps.get("geometry_detail_normal")
    if ref is None:
      continue
    base = np.eye(3)
    if bound[i].any():
      base[:2] = np.asarray(mats.uv_xform[i, first[i]], np.float64)
    detail = np.eye(3)
    detail[:2] = ref.uv_xform
    if not np.less(np.linalg.cond(base[:2, :2]), _SINGULAR):  # inf for a zero scale
      report.approx("gpu.material.detail_uv(singular base UV transform: the base's UVs)", mats.names[i])
      continue
    out[i] = (detail @ np.linalg.inv(base))[:2]
  return out.astype(np.float32)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class InstanceMaterials:
  """The instances' material tables.

  Attributes:
    material: (n,) material id (-1: none).
    rgba: (n, 4) instance color, the tint of the material's color (`ColorRule.TINT`: the kernels
      multiply it by the live material table, so `material.base_color` overrides apply).
    alpha: (n,) effective alpha.
    shading: (n,) ShadingFlag bits (specular curve, clearcoat bytes).
    curve: (n,) the Unreal specular curve.
    transmissive: (n,) the material transmits light (ShadingFlag.TRANSMISSIVE).
    coat: (n,) clearcoat weight.
    optics: their shadow transmittance, optics rows and two-sided bits (`optics.compose`).
    inputs: what those are composed of (a per-world material choice recomposes them).
  """

  material: Column[np.int32]
  rgba: Rows[np.float32, 4]
  alpha: Column[np.float32]
  shading: Column[np.int32]
  curve: Column[np.bool_]
  transmissive: Column[np.bool_]
  coat: Column[np.float64]
  optics: InstanceOptics
  inputs: OpticsInputs


def instance_materials(scene: Scene, tables: PreparedMaterials) -> InstanceMaterials:
  """The instances' material tables (`InstanceMaterials`) from their materials and colors."""
  inst, derived = scene.instances, tables.derived
  m = len(scene.materials)
  matid = np.asarray(inst.material, np.int32)
  matid = np.where((matid >= 0) & (matid < m), matid, -1)
  has_mat = matid >= 0
  mid = np.maximum(matid, 0)
  color = np.asarray(inst.color, np.float32).reshape(-1, 4)
  # The color each instance shows (`ColorRule.TINT`, as the kernels compute it).
  shown = color.copy()
  shown[has_mat] = color[has_mat] * derived.base[matid[has_mat]]

  def per_material[T: np.generic](values: Column[T], default: float) -> Column[np.float64]:
    """The instances' material values (`default` without a material)."""
    return np.where(has_mat, np.asarray(values, np.float64)[mid] if m else 0.0, default)

  curve = has_mat & (per_material(derived.curve, 0.0) != 0.0)
  transmissive = has_mat & (per_material(derived.transmissive, 0.0) != 0.0)
  coat = per_material(derived.coat, 0.0)
  canonical_shading = derived.shading(np.asarray(scene.materials.ext_flags, np.int32))
  shading = np.where(has_mat, canonical_shading[mid] if m else 0, 0).astype(np.int32)
  doubled = (np.asarray(inst.flags, np.int64) & int(InstanceFlags.DOUBLE_SIDED)) != 0
  transmissive_mat = np.asarray(derived.transmissive, bool)
  inputs = optics_inputs(
    scene, tables.optics, see_alpha=tables.see_alpha, transmissive=transmissive_mat, base=derived.base, doubled=doubled
  )
  ior = np.asarray(scene.materials.specular_ior, np.float32)
  return InstanceMaterials(
    material=matid,
    rgba=color,
    alpha=shown[:, 3],
    shading=shading,
    curve=curve,
    transmissive=transmissive,
    coat=coat,
    optics=compose(inputs, matid, ior),
    inputs=inputs,
  )
