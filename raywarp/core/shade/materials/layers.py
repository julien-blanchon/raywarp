"""The layered lobes' surface parameters at a hit (Features.layered): anisotropy and its tangent
frame, the fuzz layer, the OpenPBR coat, the cloth wrap (`LayeredMaterials` tables).

  make_layer_inputs   the material's layer values before its texture maps (`eval_surface`)
  make_layer_frame    the anisotropy's tangent frame and the fuzz albedo after the maps

Tangent frame of the anisotropy: the mesh's per-vertex UV tangents (`mesh_tangent`, generated at
upload, interpolated: no faceting on smooth meshes), the analytic dp/du of a primitive's UV
mapping, else a fixed basis of the shape-frame normal (meshes without UVs); turned about the
shading normal by the material's rotation, a negative anisotropy by a further quarter turn
(Filament's signed convention), then made orthogonal to the final shading normal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import warp as wp

from raywarp._base.types import LayerFlag, ShapeType
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields, Hit
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.kernels.generic import device_generic
from raywarp.core.shade.environment import ibl
from raywarp.core.shade.materials import texture
from raywarp.core.shade.materials.material import LayeredMaterials, MaterialTables, Surface
from raywarp.core.shade.materials.shape_normal import face_viewer

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_MESH: Final = int(ShapeType.MESH)
_COAT_OPENPBR: Final = int(LayerFlag.COAT_OPENPBR)
_COAT_BASE_NORMAL: Final = int(LayerFlag.COAT_BASE_NORMAL)

# layer_inputs(mats, worldid, mat_id, surface) -> surface (`make_layer_inputs`).
type LayerInputs = Callable[[MaterialTables, int, int, Surface], Surface]
# coat_frame(mats, mat_id, unmapped, shading, local_dir, gmat, normal, direction, surface)
# -> surface (`make_coat_frame`).
type CoatFrame = Callable[[MaterialTables, int, wp.vec3, wp.vec3, wp.vec3, wp.mat33, wp.vec3, wp.vec3, Surface], Surface]
# layer_frame(geo, mats, worldid, hit, gtype, is_mesh, local, local_normal, gmat, normal, nov,
# surface) -> surface (`make_layer_frame`).
type LayerFrame = Callable[
  [GeometryFields, MaterialTables, int, Hit, int, bool, wp.vec3, wp.vec3, wp.mat33, wp.vec3, float, Surface], Surface
]


def make_layer_inputs(f: Features):
  """layer_inputs(...) -> `surface` with the material's anisotropy, fuzz, coat and cloth values
  (each compiled in with its Features flag); a hit without a material keeps the zeros."""
  anisotropy = f.specular_anisotropy
  fuzz = f.fuzz
  coat_layer = f.coat_layer
  field = toggles.bound_on(f.bound, high=True)  # each lobe's field bit (`toggles.LAYER_*`)

  @wp.func
  def layer_inputs(
    # Model:
    mats: LayeredMaterials,
    # In:
    worldid: int,
    mat_id: int,
    surface: Surface,
  ) -> Surface:
    """The layered lobes' constants of material `mat_id` in world `worldid` (before maps)."""
    if mat_id < 0:
      return surface
    if wp.static(anisotropy):
      if field(surface.bits2, wp.static(toggles.LAYER_ANISOTROPY)):
        a = batched(mats.material_specular_anisotropy, worldid, _row(mats.material_specular_anisotropy.shape[1], mat_id))
        surface.anisotropy = a[0]
        surface.tangent = wp.vec3(a[1], 0.0, 0.0)  # the rotation, until `layer_frame`
    if wp.static(fuzz):
      if field(surface.bits2, wp.static(toggles.LAYER_FUZZ)):
        fz = batched(mats.material_fuzz, worldid, _row(mats.material_fuzz.shape[1], mat_id))
        surface.fuzz_weight = fz[0]
        surface.fuzz_alpha = fz[1]
        surface.fuzz_color = batched(mats.material_fuzz_color, worldid, _row(mats.material_fuzz_color.shape[1], mat_id))
        constants = mats.material_layer[mat_id]
        surface.wrap = constants.wrap
        surface.wrap_tint = constants.wrap_tint
    if wp.static(coat_layer):
      constants = mats.material_layer[mat_id]
      surface.coat_openpbr = (constants.flags & wp.static(_COAT_OPENPBR)) != 0
      surface.coat_darkening = constants.coat_darkening
      surface.coat_color = batched(mats.material_coat_color, worldid, _row(mats.material_coat_color.shape[1], mat_id))
      surface.coat_ior = batched(mats.material_coat_ior, worldid, _row(mats.material_coat_ior.shape[1], mat_id))
    return surface

  # The caller's carrier is `LayeredMaterials` under the same Features (`material_type`).
  return cast(LayerInputs, layer_inputs)


@wp.func
def _row(
  # In:
  ncolumn: int,
  mat_id: int,
) -> int:
  """The column of material `mat_id` in an optional per-material table of `ncolumn` columns: a
  (1, 1) placeholder (scenes whose adapter writes no layer table, e.g. MuJoCo) holds the default
  for every material."""
  if mat_id < ncolumn:
    return mat_id
  return 0


@wp.func
def _rotate_about(
  # In:
  t: wp.vec3,
  n: wp.vec3,
  angle: float,
) -> wp.vec3:
  """Tangent `t` (orthogonal to unit `n`) turned by `angle` counter-clockwise about `n`."""
  if angle == 0.0:
    return t
  return t * wp.cos(angle) + wp.cross(n, t) * wp.sin(angle)


def make_layer_frame(f: Features, *, has_mesh: bool, has_primitive: bool):
  """layer_frame(...) -> `surface` with its world tangent frame (Features.specular_anisotropy)
  and the fuzz albedo E_fuzz(NoV) (Features.fuzz), at the final shading normal."""
  anisotropy = f.specular_anisotropy
  fuzz = f.fuzz
  primitive_uv = texture.make_primitive_uv(f.shape_types, f.bound)
  field = toggles.bound_on(f.bound, high=True)  # each lobe's field bit (`toggles.LAYER_*`)

  @wp.func
  @device_generic
  def layer_frame[G: GeometryFields](
    # Model:
    geo: G,
    mats: LayeredMaterials,
    # In:
    worldid: int,
    hit: Hit,
    gtype: int,
    is_mesh: bool,
    local: wp.vec3,
    local_normal: wp.vec3,
    gmat: wp.mat33,
    normal: wp.vec3,
    nov: float,
    surface: Surface,
  ) -> Surface:
    """`surface` with the tangent frame of its anisotropy (world, orthogonal to the world
    shading normal `normal`) and its fuzz albedo at `nov`."""
    if wp.static(fuzz):
      if field(surface.bits2, wp.static(toggles.LAYER_FUZZ)) and surface.fuzz_weight > 0.0:
        surface.fuzz_albedo = ibl.sample_cloth(mats.cloth_lut, nov, surface.fuzz_alpha)
    if wp.static(anisotropy):
      rotation = surface.tangent[0]
      a = surface.anisotropy
      surface.tangent = wp.vec3(0.0)
      if field(surface.bits2, wp.static(toggles.LAYER_ANISOTROPY)) and a != 0.0:
        if a < 0.0:  # Filament's signed convention: stretched along the bitangent.
          a = -a
          rotation = rotation + wp.static(0.5 * wp.pi)
        surface.anisotropy = wp.min(a, 1.0)
        t = wp.vec3(0.0)
        if wp.static(has_mesh):
          if is_mesh and hit.mesh_id >= 0 and hit.face_id >= 0 and mats.mesh_tangent.shape[0] > 0:
            mesh = mats.mesh_adr[hit.mesh_id]
            nadr = mesh.normal
            if nadr >= 0:
              tri = mats.mesh_facenormal[mesh.face + hit.face_id]
              b2 = 1.0 - hit.bary_u - hit.bary_v
              t4 = (
                mats.mesh_tangent[nadr + tri[0]] * hit.bary_u
                + mats.mesh_tangent[nadr + tri[1]] * hit.bary_v
                + mats.mesh_tangent[nadr + tri[2]] * b2
              )
              t = wp.vec3(t4[0], t4[1], t4[2])
        if wp.static(has_primitive):
          if not is_mesh:
            size = batched(geo.instance_size, worldid, hit.instance_id)
            t1, t2 = texture.orthonormal_basis(local_normal)
            _uv, _rep, _offset, grad_u, grad_v = primitive_uv(mats.bits, gtype, local, local_normal, size, wp.vec2(1.0), False)
            t, _b, _density = texture.frame_from_uv_jacobian(
              t1, t2, wp.vec2(wp.dot(grad_u, t1), wp.dot(grad_u, t2)), wp.vec2(wp.dot(grad_v, t1), wp.dot(grad_v, t2))
            )
        world = gmat @ t
        world = world - normal * wp.dot(normal, world)
        if wp.length_sq(world) <= 1.0e-12:
          # No UVs / a degenerate frame (a pole): a fixed basis of the shape-frame normal.
          b0, _b1 = texture.orthonormal_basis(local_normal)
          world = gmat @ b0
          world = world - normal * wp.dot(normal, world)
        if wp.length_sq(world) <= 1.0e-12:
          surface.anisotropy = 0.0
          return surface
        t = _rotate_about(wp.normalize(world), normal, rotation)
        surface.tangent = t
        surface.bitangent = wp.cross(normal, t)
    return surface

  # The caller's carrier is `LayeredMaterials` under the same Features (`material_type`).
  return cast(LayerFrame, layer_frame)


def make_coat_frame(f: Features):
  """coat_frame(...) -> `surface` with the coat's world normal and NoV (Features.coat_layer /
  coat_normal): its own normal map's (`uv_maps.make_layer_normals`), else the unmapped normal
  (OpenPBR / glTF), or the base's mapped one for LayerFlag.COAT_BASE_NORMAL; the base's normal
  without Features.coat_normal."""
  own = f.coat_normal

  @wp.func
  def coat_frame(
    # Model:
    mats: LayeredMaterials,
    # In:
    mat_id: int,
    unmapped: wp.vec3,
    shading: wp.vec3,
    local_dir: wp.vec3,
    gmat: wp.mat33,
    normal: wp.vec3,
    direction: wp.vec3,
    surface: Surface,
  ) -> Surface:
    """The coat's frame from the shape-frame `unmapped` / mapped `shading` normals (`gmat`: shape
    to world, `local_dir` the ray in the shape frame), `normal` the base's world normal."""
    surface.coat_normal = normal
    if wp.static(own):
      coat = surface.coat_normal_local
      if wp.length_sq(coat) == 0.0:
        coat = unmapped
        if mat_id >= 0:
          if (mats.material_layer[mat_id].flags & wp.static(_COAT_BASE_NORMAL)) != 0:
            coat = shading
      surface.coat_normal = wp.normalize(gmat @ face_viewer(coat, -local_dir))
    surface.coat_nov = wp.max(wp.dot(surface.coat_normal, -direction), 1.0e-4)
    return surface

  # The caller's carrier is `LayeredMaterials` under the same Features (`material_type`).
  return cast(CoatFrame, coat_frame)
