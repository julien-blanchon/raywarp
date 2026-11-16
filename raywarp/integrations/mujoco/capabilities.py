"""What a MuJoCo scene's content uses (`Capabilities`), from every batched row of `m`.

The kernels compile in translucency, texture slots, spot lights, ... only when the scene uses them.
Since `m`'s tables are aliased live, the scan reads every row of the batched arrays (per-world
`geom_rgba`, `geom_matid`, `mat_rgba`, `mat_texid`, `light_type`), not only the initial model.
Data written later can still need a feature compiled out: `LiveCapabilityCheck` (a
`core.pipeline.gpu_scene.binding.FeatureCheck`, from `MjwBinding.feature_checks()`) detects it on
the device.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Final, final, override

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from raywarp._base.dims import Mask
from raywarp._base.types import NTEXROLE, ColorRule, LightType, TexRole
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.tables import batched, instance_rgba
from raywarp.core.pipeline.frame import Frame
from raywarp.core.pipeline.gpu_scene.binding import FeatureCheck
from raywarp.core.pipeline.gpu_scene.gpu_scene import Meshes, TextureSlots
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.platform.launches import CachedLaunch
from raywarp.core.pipeline.worlds.validator_kernels import Word, raise_flag
from raywarp.integrations.mujoco import materials as mj_materials
from raywarp.integrations.mujoco.hfield import HeightFields
from raywarp.integrations.mujoco.textures import OPAQUE_U8, MjTextureId, texels

# Forward-only kernels: no adjoint code.
wp.set_module_options({"enable_backward": False})

# Texture roles per core slot (`TextureSlots` field order: base_color, normal, orm, emissive,
# opacity).
SLOT_ROLES: Final = (
  (int(TexRole.RGB), int(TexRole.RGBA)),
  (int(TexRole.NORMAL),),
  (int(TexRole.ORM), int(TexRole.OCCLUSION), int(TexRole.ROUGHNESS), int(TexRole.METALLIC)),
  (int(TexRole.EMISSIVE),),
  (int(TexRole.OPACITY),),
)
_SPOT: Final = int(LightType.SPOT)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Content:
  """What the rendered geoms use across every batched row of `m`.

  Attributes:
    materials: ids of the materials rendered geoms (in any world) and flexes use.
    texid: (n, NTEXROLE) every row of `m.mat_texid` of those materials, stacked.
    opacity: (len(materials),) whether each binds an opacity texture in some row.
    slots: texture slots bound.
    alpha_map: an RGBA / opacity texture bound has texels with alpha < 1.
    has_alpha: a rendered geom or its material has 0 < alpha < 1 in some world (or a flex).
    has_spot: a spot light in some world.
  """

  materials: np.ndarray[[int], np.dtype[np.int64]]
  texid: np.ndarray[[int, int], np.dtype[np.int32]]
  opacity: Mask[int]
  slots: TextureSlots
  alpha_map: bool
  has_alpha: bool
  has_spot: bool


def _effective_alpha(
  m: mjw.Model, matid: np.ndarray[[int, int], np.dtype[np.int32]], rgba: np.ndarray[[int, int, 4], np.dtype[np.float32]]
) -> np.ndarray[[int, int], np.dtype[np.float32]]:
  """(rows, n) alpha each of n shapes (geoms or flexes) of material ids `matid` and colors `rgba`
  ((1 or rows, n)) shows in every row of `m.mat_rgba` too (`tables.material_rgba`: the default
  rgba defers to the material's)."""
  matid = np.asarray(matid)
  rgba = np.asarray(rgba, np.float32)
  mat_rgba = host_copy(m.mat_rgba)
  if not mat_rgba.size:
    return np.broadcast_to(rgba[..., 3], np.broadcast_shapes(matid.shape, rgba.shape[:-1]))
  mat_rgba = mat_rgba.reshape(-1, mat_rgba.shape[-2], 4)
  nrow = max(matid.shape[0], rgba.shape[0], mat_rgba.shape[0])
  shape = (nrow, matid.shape[1])
  matid, alpha = np.broadcast_to(matid, shape), np.broadcast_to(rgba[..., 3], shape)
  inherit = (matid >= 0) & np.all(
    np.broadcast_to(rgba, (*shape, 4)) == np.asarray(mj_materials.DEFAULT_GEOM_RGBA, np.float32), axis=-1
  )
  rows = np.arange(nrow)[:, None] % mat_rgba.shape[0]
  return np.where(inherit, mat_rgba[rows, np.maximum(matid, 0), 3], alpha)


def _alpha_map(mjm: mujoco.MjModel, texid: np.ndarray[[int, int], np.dtype[np.int32]]) -> bool:
  """Whether an RGBA (alpha channel) or opacity (red channel) texture of `texid` has alpha < 1."""
  for role, channel in ((TexRole.RGBA, 3), (TexRole.OPACITY, 0)):
    for t in np.unique(texid[:, int(role)]):
      if t < 0:
        continue
      data = texels(mjm, MjTextureId(int(t)))
      if channel < data.shape[2] and np.any(data[..., channel] < OPAQUE_U8):
        return True
  return False


def content(mjm: mujoco.MjModel, m: mjw.Model, rendered: Mask[int]) -> Content:
  """The `Content` of the rendered geoms (`rendered`: (ngeom,) mask) and flexes over every row of
  `m`."""
  matid = host_copy(m.geom_matid)[:, rendered]
  flex_matid = np.asarray(mjm.flex_matid)
  materials = np.unique(np.concatenate([matid[matid >= 0], flex_matid[flex_matid >= 0]])).astype(np.int64)
  by_row = host_copy(m.mat_texid)[:, materials] if mjm.nmat else np.full((1, 0, NTEXROLE), -1, np.int32)
  opacity = np.any(by_row[..., int(TexRole.OPACITY)] >= 0, axis=0)
  texid = np.asarray(by_row, np.int32).reshape(-1, NTEXROLE)
  slots = TextureSlots(*(bool(np.any(texid[:, list(roles)] >= 0)) for roles in SLOT_ROLES))
  alpha = _effective_alpha(m, matid, host_copy(m.geom_rgba)[:, rendered])
  # Flex material ids and colors are static (`mjm`); their materials' rows are live.
  flex_alpha = _effective_alpha(m, flex_matid[None], np.asarray(mjm.flex_rgba).reshape(1, -1, 4))
  has_alpha = bool(np.any((alpha > 0.0) & (alpha < 1.0))) or bool(np.any(flex_alpha < 1.0))
  light_type = host_copy(m.light_type)
  has_spot = bool(mjm.nlight and np.any(light_type == _SPOT))
  return Content(
    materials=materials,
    texid=texid,
    opacity=opacity,
    slots=slots,
    alpha_map=_alpha_map(mjm, texid),
    has_alpha=has_alpha,
    has_spot=has_spot,
  )


@wp.kernel
def _check_geoms(
  # Model:
  instance_material: wp.array2d[int],
  instance_color: wp.array2d[wp.vec4],
  material_base_color: wp.array2d[wp.vec4],
  material_texture: wp.array3d[int],
  # In:
  rendered: wp.array[int],
  transparency: bool,
  role_mask: int,
  bit: wp.int32,
  # Out:
  flags_out: wp.array[wp.int32],
):
  """Sets flag `bit` of the validation word when rendered instance k in row w needs alpha or a
  texture role the kernels were compiled without."""
  w, k = wp.tid()
  rgba, mat = instance_rgba(
    wp.static(int(ColorRule.MUJOCO)), instance_material, instance_color, material_base_color, w, rendered[k]
  )
  need = not transparency and rgba[3] > 0.0 and rgba[3] < 1.0
  if mat >= 0:
    for role in range(wp.static(NTEXROLE)):
      if material_texture[w % material_texture.shape[0], mat, role] >= 0 and (role_mask & (1 << role)) == 0:
        need = True
  if need:
    raise_flag(bit, flags_out)


@wp.kernel
def _check_lights(
  # Model:
  light_type: wp.array2d[int],
  # In:
  bit: wp.int32,
  # Out:
  flags_out: wp.array[wp.int32],
):
  """Sets flag `bit` of the validation word when light i in row w is a spot light of a build
  without spots."""
  w, i = wp.tid()
  if batched(light_type, w, i) == wp.static(_SPOT):
    raise_flag(bit, flags_out)


@wp.kernel
def _check_meshes(
  # Model:
  instance_mesh: wp.array2d[int],
  used: wp.array[bool],
  # In:
  mesh_geoms: wp.array[int],
  bit: wp.int32,
  # Out:
  flags_out: wp.array[wp.int32],
):
  """Sets flag `bit` of the validation word when rendered mesh geom k in row w shows a mesh that
  got no BLAS (`Meshes.used`: it would render as nothing)."""
  w, k = wp.tid()
  mesh = batched(instance_mesh, w, mesh_geoms[k])
  if mesh >= 0 and mesh < used.shape[0] and not used[mesh]:
    raise_flag(bit, flags_out)


def _role_mask(features: Features) -> int:
  """Bit r set for every texture role r the compiled build samples."""
  if not features.textures:
    return 0
  compiled = (features.base_color_map, features.normal_map, features.orm_map, features.emissive_map, features.opacity_map)
  return functools.reduce(
    operator.or_, (1 << r for on, roles in zip(compiled, SLOT_ROLES, strict=True) if on for r in roles), 0
  )


@final
class LiveCapabilityCheck(FeatureCheck):
  """`core.pipeline.gpu_scene.binding.FeatureCheck` of a MuJoCo scene: the frame's live tables
  (aliased `m`, per-world overrides) need a feature the kernels were compiled without (translucency,
  a texture role, spot lights), a mesh built without a BLAS (`Meshes.used`) or a height field built
  without one (`HeightFields.ids`). Device only, graph-capturable.

  Attributes:
    rendered: (n,) the scene's rendered geom ids, on its device; `mesh_geoms` the mesh ones,
      `hfield_geoms` the height-field ones.
    used: `Meshes.used_now()` when the check was made, in its Renderer's constructor: the mask of
      that Renderer's BLAS / GAS tables (`RendererTables.mesh_used`; None: all).
    hfield_used: (max(nhfield, 1),) the height fields that have a BLAS (on the device).
    bit: the validation flag set (`core.pipeline.worlds.validator_kernels.Word.COMPILED_OUT`).
  """

  def __init__(  # noqa: PLR0913  # the binding's geom sets and tables
    self,
    rendered: wp.array[int, [int]],
    mesh_geoms: wp.array[int, [int]],
    meshes: Meshes,
    hfield_geoms: wp.array[int, [int]],
    hfields: HeightFields,
    scanned: Content,
  ) -> None:
    """The check of the geoms `rendered` (meshes: `mesh_geoms` against `meshes.used`; height
    fields: `hfield_geoms` against the BLASes of `hfields`); `scanned`: what the scene creation
    found. A feature the scan found but the settings turned off (no translucency, no textures) is
    no live change: those tables render as the settings say."""
    self.rendered = rendered
    self.scanned_alpha = scanned.has_alpha
    self.scanned_roles = functools.reduce(
      operator.or_, (1 << r for r in range(scanned.texid.shape[1]) if np.any(scanned.texid[:, r] >= 0)), 0
    )
    self.scanned_spot = scanned.has_spot
    self.mesh_geoms = mesh_geoms
    self.used = meshes.used_now()  # a later use_all keeps it, as the Renderer's tables do
    self.bit = int(Word.COMPILED_OUT)
    self._used: wp.array[bool, [int]] | None = None  # the device copy of `used`
    self.hfield_geoms = hfield_geoms
    built = np.zeros(hfields.bvh_id.shape[0], bool)
    built[hfields.ids] = True
    self.hfield_used = wp.array(built, dtype=bool, device=hfield_geoms.device)
    self._geoms, self._lights, self._meshes = CachedLaunch(), CachedLaunch(), CachedLaunch()
    self._hfields = CachedLaunch()

  @override
  def enqueue(self, features: Features, frame: Frame, flags: wp.array[wp.int32, [int]]) -> None:
    """Launches the check of `frame`'s tables against `features` (sets `bit` of `flags[0]`).

    Meshes are checked for every output; the appearance only with shaded outputs: depth /
    segmentation read none of those features (`canonical.canonical` collapses them, e.g. `has_spot`,
    which would else flag every spot light).
    """
    used = self._used_device(flags.device)
    if used is not None and self.mesh_geoms.shape[0]:
      dim = (frame.instance_mesh.shape[0], self.mesh_geoms.shape[0])
      inputs = [frame.instance_mesh, used, self.mesh_geoms, self.bit]
      self._meshes.launch(_check_meshes, dim, inputs=inputs, outputs=[flags], device=flags.device)
    if self.hfield_geoms.shape[0]:
      # A height-field geom's `instance_mesh` is its height field id: the mesh check over the
      # height fields' BLAS mask.
      dim = (frame.instance_mesh.shape[0], self.hfield_geoms.shape[0])
      inputs = [frame.instance_mesh, self.hfield_used, self.hfield_geoms, self.bit]
      self._hfields.launch(_check_meshes, dim, inputs=inputs, outputs=[flags], device=flags.device)
    if not features.any_rgb:
      return
    rows = max(
      frame.instance_material.shape[0],
      frame.instance_color.shape[0],
      frame.material_base_color.shape[0],
      frame.material_texture.shape[0],
    )
    if self.rendered.shape[0]:
      inputs = [frame.instance_material, frame.instance_color, frame.material_base_color, frame.material_texture]
      allowed = features.transparency or self.scanned_alpha
      inputs += [self.rendered, allowed, _role_mask(features) | self.scanned_roles, self.bit]
      self._geoms.launch(_check_geoms, (rows, self.rendered.shape[0]), inputs=inputs, outputs=[flags], device=flags.device)
    # A build that reads no lights (unlit shading, `nlight` 0) has no spot feature to miss.
    if features.nlight and not (features.has_spot or self.scanned_spot) and frame.light_type.shape[1]:
      dim = frame.light_type.shape
      self._lights.launch(_check_lights, dim, inputs=[frame.light_type, self.bit], outputs=[flags], device=flags.device)

  def _used_device(self, device: wp.Device | None) -> wp.array[bool, [int]] | None:
    """The device copy of `used` (None: every mesh has its BLAS)."""
    if self.used is None:
      return None
    if self._used is None:
      self._used = wp.array(self.used, dtype=bool, device=device)
    return self._used

  @override
  def explain(self, features: Features) -> str:
    """What the warning says when the check fired."""
    return (
      "the MuJoCo model's live tables (m.geom_rgba / mat_rgba alpha, m.mat_texid, m.light_type) now need a feature this "
      "Renderer was compiled without (translucency, a texture slot or spot lights): set it before creating the Renderer "
      "(the capabilities are scanned over every batched row of m at scene creation); or m.geom_dataid now shows a mesh "
      "no rendered geom showed at scene creation (it has no BLAS): declare it per world (PerWorld(mesh=[...])) or render "
      "that geom from the start; or m.geom_dataid now shows a height field no rendered geom showed in any batched row at "
      "scene creation (it has no BLAS): write it into a batched m.geom_dataid row before creating the scene"
    )
