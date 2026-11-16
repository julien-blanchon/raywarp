"""The texture frame of a hit: UV, tangent frame and UV density, from a mesh triangle's UVs or a
primitive's analytic mapping (`texture.make_primitive_uv`), and the material's maps there
(`make_apply_maps`: the frame, then `uv_maps.make_uv_maps`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import warp as wp

from raywarp._base.types import UV_MESH, TexRole, uv_xform
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import Cutout, GeometryFields, Hit, LiveUvCutout
from raywarp.core.kernels.abi.tables import batched, table_row
from raywarp.core.kernels.generic import device_generic
from raywarp.core.shade.materials import texture
from raywarp.core.shade.materials.material import MaterialTables, Surface
from raywarp.core.shade.materials.uv_maps import make_taps, make_uv_maps

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_ROLE_RGB: Final = int(TexRole.RGB)
_ROLE_RGBA: Final = int(TexRole.RGBA)


# A paired accessor: used only when both the factory and geometry carrier select live UVs.
type UvAddress = Callable[[Cutout | LiveUvCutout | MaterialTables, int, int], tuple[wp.vec2i, uv_xform]]


@wp.func(inline=True)
@device_generic
def _live_uv_address[C: LiveUvCutout | MaterialTables](
  # Model:
  cut: C,
  # In:
  mesh_id: int,
  mat_id: int,
) -> tuple[wp.vec2i, uv_xform]:
  """(texcoord address, corner row address) of mesh `mesh_id`'s UV set selected by material
  `mat_id`'s live UV mapping (its second set where the mesh has one), and the mapping's
  transform."""
  mapping = cut.material_uv_mapping[mat_id]
  mesh = cut.mesh_adr[mesh_id]
  adr = wp.vec2i(mesh.texcoord, mesh.face)
  if mapping.uv_set == 1:
    second = cut.mesh_uv_second[mesh_id]
    if second[0] != wp.static(UV_MESH):
      adr = second
  return adr, mapping.xform


def live_uv_address() -> UvAddress:
  """Accessor narrowed by the same static flag that selects LiveUvCutout at launch (an OptiX shaded
  build's alpha-test tables are its material carrier, which holds the live-UV tables too)."""
  return cast(UvAddress, _live_uv_address)


def make_mesh_uv_frame(*, uv_maps: bool, live_uv: bool, bound: int):
  """The UV, tangent frame and density of a mesh hit using its selected material.

  `uv_maps`: Features.uv_maps (the instance's own UVs; `bound`, Features.bound: read with the maps
  class bit); `live_uv`: Features.live_material_uv (the selected material's live UV mapping, in
  its place where the word's bit has it, `toggles.MAP_LIVE_UV`).
  """
  address = live_uv_address()
  gate = toggles.bound_on(bound)
  field = toggles.bound_on(bound, high=True)  # the field bits (`toggles.MAP_UV_SETS`, `MAP_LIVE_UV`)

  @wp.func
  @device_generic
  def mesh_uv_frame[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    # In:
    hit: Hit,
    mat_id: int,
    repeat: wp.vec2,
  ) -> tuple[bool, wp.vec2, wp.vec3, wp.vec3, float]:
    """The texture frame of a mesh triangle hit, in the shape frame."""
    # The corner UVs: the instance's UV set and transform (uv_maps), else the mesh's.
    adr = wp.vec2i(wp.static(UV_MESH), 0)
    m = uv_xform(0.0)
    live = bool(False)
    if wp.static(live_uv):
      live = field(geo.bits2, wp.static(toggles.MAP_LIVE_UV))
      if live:
        adr, m = address(geo.cut, hit.mesh_id, mat_id)
    if wp.static(uv_maps):
      if not live and gate(mats.bits, wp.static(toggles.CLASS_MAPS)) and field(geo.bits2, wp.static(toggles.MAP_UV_SETS)):
        own = geo.mesh_instance[hit.instance_id]
        adr = own.texcoord
        m = own.uv_xform
    mesh = mats.mesh_adr[hit.mesh_id]
    has_uv, uv0, uv1, uv2 = texture.triangle_uvs(
      mats.mesh_facetexcoord, mats.mesh_texcoord, mesh.texcoord, mesh.face, adr, m, hit.face_id
    )
    if not has_uv:
      return False, wp.vec2(0.0), wp.vec3(0.0), wp.vec3(0.0), 0.0
    uv, tangent, bitangent, uv_per_length = texture.mesh_triangle_frame(
      mats.mesh_vert, mats.mesh_face, mesh, hit.face_id, hit.bary_u, hit.bary_v, uv0, uv1, uv2, repeat
    )
    return True, uv, tangent, bitangent, uv_per_length

  return mesh_uv_frame


@wp.func
def world_projected(
  # In:
  m: uv_xform,
) -> bool:
  """Whether `material.uv_world` row `m` is a world projection (zeros: mesh UVs)."""
  return m[0, 0] != 0.0 or m[0, 1] != 0.0 or m[1, 0] != 0.0 or m[1, 1] != 0.0


@wp.func
def world_uv_frame(
  # In:
  m: uv_xform,
  point: wp.vec3,
  normal: wp.vec3,
  gmat_t: wp.mat33,
) -> tuple[wp.vec2, wp.vec3, wp.vec3, float]:
  """(uv, tangent, bitangent, uv_per_length) of a world-projected material (`material.uv_world`
  `m`): the world `point` on the plane of the dominant axis of the geometric `normal` (y, z /
  x, z / x, y for an x- / y- / z-facing point; Unreal's world-aligned projection with a hard
  blend), through `m`. The frame (dp/du, dp/dv) is returned in the shape frame (`gmat_t`)."""
  a = wp.abs(normal)
  e0 = wp.vec3(1.0, 0.0, 0.0)
  e1 = wp.vec3(0.0, 1.0, 0.0)
  if a[0] >= a[1] and a[0] >= a[2]:
    e0 = wp.vec3(0.0, 1.0, 0.0)
    e1 = wp.vec3(0.0, 0.0, 1.0)
  elif a[1] >= a[2]:
    e1 = wp.vec3(0.0, 0.0, 1.0)
  q = wp.vec2(wp.dot(point, e0), wp.dot(point, e1))
  uv = wp.vec2(m[0, 0] * q[0] + m[0, 1] * q[1] + m[0, 2], m[1, 0] * q[0] + m[1, 1] * q[1] + m[1, 2])
  det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
  if wp.abs(det) < 1.0e-20:
    return uv, wp.vec3(0.0), wp.vec3(0.0), 0.0
  # dq / d(uv) = M^-1: the world directions of +u and +v on the plane.
  tangent = (e0 * m[1, 1] - e1 * m[1, 0]) / det
  bitangent = (e1 * m[0, 0] - e0 * m[0, 1]) / det
  return uv, gmat_t @ tangent, gmat_t @ bitangent, wp.sqrt(wp.abs(det))


def make_uv_frame(f: Features, *, has_mesh: bool, has_primitive: bool):
  """uv_frame(...) -> (has_uv, uv, tangent, bitangent, uv_per_length) of a hit, shape frame.

  Features.uv_maps: `make_mesh_uv_frame`; Features.world_uv: `world_uv_frame` for
  the materials with a `material.uv_world` map; the primitives' analytic mappings of
  Features.shape_types.
  """
  mesh_uv_frame = make_mesh_uv_frame(uv_maps=f.uv_maps, live_uv=f.live_material_uv, bound=f.bound)
  primitive_uv = texture.make_primitive_uv(f.shape_types, f.bound)
  world_uv = f.world_uv
  gate = toggles.bound_on(f.bound)  # the maps class bit
  field = toggles.bound_on(f.bound, high=True)  # the world UV's field bit (`toggles.MAP_WORLD_UV`)

  @wp.func
  @device_generic
  def uv_frame[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    # In:
    worldid: int,
    hit: Hit,
    gtype: int,
    is_mesh: bool,
    mat_id: int,
    point: wp.vec3,
    gmat_t: wp.mat33,
    scale: float,
    local: wp.vec3,
    local_normal: wp.vec3,
    repeat: wp.vec2,
    uniform: bool,
  ) -> tuple[bool, wp.vec2, wp.vec3, wp.vec3, float]:
    """The hit's texture frame: the material's world projection, mesh triangle UVs, or a
    primitive's analytic mapping.

    `point`: the hit in the world (`gmat_t`: world to shape frame, world units; `scale`: a mesh
    instance's, which scales its triangles); `local`: the hit in the shape frame (primitives);
    tangent = dp/du, bitangent = dp/dv.
    """
    if wp.static(world_uv):
      if gate(mats.bits, wp.static(toggles.CLASS_MAPS)) and field(geo.bits2, wp.static(toggles.MAP_WORLD_UV)):
        m = batched(mats.material_uv_world, worldid, mat_id)
        if world_projected(m):
          uv, tangent, bitangent, uv_per_length = world_uv_frame(m, point, hit.normal, gmat_t)
          return True, uv, tangent, bitangent, uv_per_length
    has_uv = False
    uv = wp.vec2(0.0)
    tangent = wp.vec3(0.0)
    bitangent = wp.vec3(0.0)
    uv_per_length = float(0.0)
    if wp.static(has_mesh):
      if is_mesh and hit.mesh_id >= 0 and hit.face_id >= 0:
        has_uv, uv, tangent, bitangent, uv_per_length = mesh_uv_frame(geo, mats, hit, mat_id, repeat)
        tangent = tangent * scale
        bitangent = bitangent * scale
        uv_per_length = uv_per_length / scale
    if wp.static(has_primitive):
      if not is_mesh:
        size = batched(geo.instance_size, worldid, hit.instance_id)
        t1, t2 = texture.orthonormal_basis(local_normal)
        uv0, rep, offset, grad_u, grad_v = primitive_uv(mats.bits, gtype, local, local_normal, size, repeat, uniform)
        # The repeated UVs' derivatives along the tangent basis (t1, t2).
        du = grad_u * rep[0]
        dv = grad_v * rep[1]
        tangent, bitangent, uv_per_length = texture.frame_from_uv_jacobian(
          t1, t2, wp.vec2(wp.dot(du, t1), wp.dot(du, t2)), wp.vec2(wp.dot(dv, t1), wp.dot(dv, t2))
        )
        uv = wp.cw_mul(uv0, rep) + offset
        has_uv = True
    return has_uv, uv, tangent, bitangent, uv_per_length

  return uv_frame


# mapped(mats, worldid, mat_id) -> whether the material binds a sampled map (`make_mapped`).
type Mapped = Callable[[MaterialTables, int, int], bool]


def map_roles(f: Features, classes: int = -1) -> int:
  """The bit mask of the texture roles the maps of `f` sample (`TexRole` bits), of the map classes
  `classes` (`toggles.CLASS_BASE_MAP / MAPS / LAYER_MAPS` bits; -1: every class)."""
  base, other, layer = (classes & c != 0 for c in (toggles.CLASS_BASE_MAP, toggles.CLASS_MAPS, toggles.CLASS_LAYER_MAPS))
  roles = (
    (base and f.base_color_map, (TexRole.RGB, TexRole.RGBA)),
    (other and f.orm_map, (TexRole.ORM, TexRole.OCCLUSION, TexRole.ROUGHNESS, TexRole.METALLIC)),
    (other and f.normal_map, (TexRole.NORMAL,)),
    (other and f.opacity_map, (TexRole.OPACITY,)),
    (other and f.emissive_map, (TexRole.EMISSIVE,)),
    (other and f.diffuse_roughness_map, (TexRole.DIFFUSE_ROUGHNESS,)),
    (layer and f.layered_maps, (TexRole.ANISOTROPY, TexRole.FUZZ)),
    (layer and f.coat_normal_map, (TexRole.COAT_NORMAL,)),
    (layer and f.detail_normal, (TexRole.DETAIL_NORMAL,)),
  )
  return sum({1 << int(r) for on, rs in roles if on for r in rs})


def make_mapped(f: Features):
  """mapped(mats, worldid, mat_id): whether material `mat_id` binds a texture in world `worldid` to
  a role the maps of `f` sample (`map_roles`: the others' ids are not read; unrolled, one load and
  one max per compiled role of the material's row, no branch: an id >= 0 is bound, its range
  is `texture_id`'s)."""
  mask = map_roles(f)
  nrole = max(mask.bit_length(), 1)
  # The roles of each map class (read with its class bit, `toggles.bound_on`).
  gate = toggles.bound_on(f.bound)
  base, other, layer = (map_roles(f, c) for c in (toggles.CLASS_BASE_MAP, toggles.CLASS_MAPS, toggles.CLASS_LAYER_MAPS))

  @wp.func
  @device_generic
  def mapped[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    worldid: int,
    mat_id: int,
  ) -> bool:
    """Whether the material has a map the build samples (`make_mapped`)."""
    ids = mats.material_texture
    row = table_row(worldid, ids.shape[0])
    roles = ids.shape[2]
    sampled = int(0)  # the roles of the map classes on (a constant once bound)
    if gate(mats.bits, wp.static(toggles.CLASS_BASE_MAP)):
      sampled = sampled | wp.static(base)
    if gate(mats.bits, wp.static(toggles.CLASS_MAPS)):
      sampled = sampled | wp.static(other)
    if gate(mats.bits, wp.static(toggles.CLASS_LAYER_MAPS)):
      sampled = sampled | wp.static(layer)
    bound = int(-1)
    for role in range(wp.static(nrole)):
      if wp.static(mask) >> role & 1 != 0:
        if (sampled >> role) & 1 != 0 and role < roles:
          bound = wp.max(bound, ids[row, mat_id, role])
    return bound >= 0

  return mapped


def make_apply_maps(f: Features, *, has_mesh: bool, has_primitive: bool):
  """apply_maps(...) -> (surface, shading normal, emission) with the material's texture maps."""
  uv_frame = make_uv_frame(f, has_mesh=has_mesh, has_primitive=has_primitive)
  uv_maps = make_uv_maps(f)
  taps = make_taps(f)
  mapped = make_mapped(f)

  @wp.func
  @device_generic
  def apply_maps[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    # In:
    worldid: int,
    hit: Hit,
    gtype: int,
    is_mesh: bool,
    mat_id: int,
    point: wp.vec3,
    gmat_t: wp.mat33,
    scale: float,
    local: wp.vec3,
    local_dir: wp.vec3,
    local_normal: wp.vec3,
    cone: float,
    width: float,
    side: float,
    shading: wp.vec3,
    surface: Surface,
    emission: float,
  ) -> tuple[Surface, wp.vec3, float]:
    """The hit's texture frame, then its maps: (surface, shading normal, emission strength).

    `point` / `local`: the hit in the world / shape frame (`gmat_t`: world to shape; `scale`: a
    mesh instance's, `uv_frame`), `local_dir` the ray; `width`: the ray cone's width at the hit,
    `cone` its footprint on the surface.
    """
    if not mapped(mats, worldid, mat_id):
      # No map: surface, normal and emission as they are (every role's code reads a bound id).
      return surface, shading, emission
    repeat = batched(mats.material_tex_repeat, worldid, mat_id)
    uniform = batched(mats.material_tex_uniform, worldid, mat_id)
    has_uv, uv, tangent, bitangent, uv_per_length = uv_frame(
      geo, mats, worldid, hit, gtype, is_mesh, mat_id, point, gmat_t, scale, local, local_normal, repeat, uniform
    )
    return uv_maps(
      mats,
      worldid,
      mat_id,
      has_uv,
      uv,
      tangent,
      bitangent,
      uv_per_length,
      cone,
      width,
      local_dir,
      local_normal,
      side,
      shading,
      True,
      taps(geo, mats),
      surface,
      emission,
    )

  return apply_maps
