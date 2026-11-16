"""Alpha cutout: rays pass through the transparent texels of a leaf or fence texture (glTF
alphaMode MASK, UsdPreviewSurface opacityThreshold), by an alpha test at each hit.

A hit on a material with a cutoff > 0 (`GeometryFields.cut.material_alpha_cutoff`, the
"material.alpha_cutoff" slot) whose opacity texel is below the cutoff is no hit: the ray queries of
a kernel compiled with `Features.alpha_cutout` continue past it (camera, shadow, reflection and GI
probe rays, the light grid's visibility rays). The surfaces that pass are opaque. The test
point-samples level 0 of the opacity map at the hit's UV (the authored edge at any distance).
Without the switch nothing of this is emitted.

Two forms, each testing a candidate where the traversal finds it:
- OptiX: the hit programs (`trace/optix/device/programs.py`). Triangles: an any-hit program on the
triangle hit
  group (`make_cut_triangle`) ignores the candidates that fail the test; only the cut-out instances
  run it (the others disable any-hit programs). Spheres and capsules: the curve hit groups' any-hit
  programs (their GAS instance enforces them when one of its curves is cut out). Ellipsoids and
  cylinders: their intersection programs report the first analytic hit the test keeps. Planes:
  the analytic plane queries (`query.make_plane_hit`). The ray-generation program has the call
  sites and state of a build without cutouts.
- software traversal: `trace.accel.instances.make_intersect` with `make_cut_away` (a cut-out mesh's
  BLAS is walked once, candidate by candidate).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import warp as wp

from raywarp._base.types import ShapeType, TexRole, uv_xform
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.rays import GeometryFields, Hit, instance_pose
from raywarp.core.kernels.abi.tables import batched, in_table, material_id, texture_id
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.materials import texture
from raywarp.core.shade.materials.uv_frame import live_uv_address, world_projected, world_uv_frame

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_ROLE_OPACITY: Final = int(TexRole.OPACITY)

# cut_away(geo, worldid, hit, point, direction) -> whether the alpha test discards `hit`.
type CutAway = Callable[[GeometryFields, int, Hit, wp.vec3, wp.vec3], bool]


@wp.func
@device_generic
def cut_material[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: int,
  gi: int,
) -> tuple[int, float, int]:
  """(material, cutoff, opacity texture) of instance gi; cutoff 0 (no test) without them."""
  mat_id = material_id(geo.instance_material, geo.material_base_color.shape[1], worldid, gi)
  if mat_id < 0:
    return mat_id, 0.0, -1
  c = geo.cut
  cutoff = batched(c.material_alpha_cutoff, worldid, mat_id)
  if cutoff <= 0.0:
    return mat_id, 0.0, -1
  tex_id = texture_id(c.material_texture, c.textures.shape[0], worldid, mat_id, wp.static(_ROLE_OPACITY))
  if tex_id < 0:
    return mat_id, 0.0, -1
  return mat_id, cutoff, tex_id


@specialize
def make_mesh_uv(*, live_uv: bool, bound: int):
  """mesh_uv(geo, gi, mesh_id, face, bary_u, bary_v, repeat, mat_id) -> (uv, has_uv): the tiled
  texture coordinate at barycentrics (u, v) of a face of instance gi's mesh, shared by the alpha
  test, the uv output (`make_hit_uv`) and the OptiX any-hit test (`make_cut_triangle`); `live_uv`:
  the live material UV mapping where the word's bit has it (`toggles.MAP_LIVE_UV`; `bound`:
  Features.bound)."""
  address = live_uv_address()
  field = toggles.bound_on(bound, high=True)

  @wp.func
  @device_generic
  def mesh_uv[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    gi: int,
    mesh_id: int,
    face: int,
    bary_u: float,
    bary_v: float,
    repeat: wp.vec2,
    mat_id: int,
  ) -> tuple[wp.vec2, bool]:
    """(texture coordinate, whether the face has one) at barycentrics (u, v) of a face of instance
    gi's mesh."""
    c = geo.cut
    adr = wp.vec2i(0)
    xf = uv_xform(0.0)
    live = bool(False)
    if wp.static(live_uv):
      live = field(geo.bits2, wp.static(toggles.MAP_LIVE_UV))
      if live:
        adr, xf = address(c, mesh_id, mat_id)
    if not live:
      own = geo.mesh_instance[gi]
      adr = own.texcoord
      xf = own.uv_xform
    mesh = c.mesh_adr[mesh_id]
    has_uv, uv0, uv1, uv2 = texture.triangle_uvs(c.mesh_facetexcoord, c.mesh_texcoord, mesh.texcoord, mesh.face, adr, xf, face)
    if not has_uv:
      return wp.vec2(0.0), False
    w2 = 1.0 - bary_u - bary_v
    return wp.cw_mul(uv0 * bary_u + uv1 * bary_v + uv2 * w2, repeat), True

  return mesh_uv


@specialize  # the alpha test and the uv output of one kernel share one copy
def make_hit_uv(shape_types: tuple[int, ...], *, compact: bool, world_uv: bool = False, live_uv: bool = False, bound: int = 0):
  """hit_uv(geo, worldid, hit, point, direction, mat_id) -> (uv, has_uv): the texture coordinate
  the maps of material `mat_id` read at a camera-ray hit (mesh UVs or the primitive projection,
  the material's tiling applied; the alpha test's, the uv output's). `world_uv`
  (Features.world_uv; the uv output): a material's world projection first, as `uv_frame`. `bound`:
  Features.bound (`instance_pose`, the primitives' shape bits)."""
  mesh_uv = make_mesh_uv(live_uv=live_uv, bound=bound)
  pose = instance_pose(compact=compact, bound=bound)
  primitive_uv = texture.make_primitive_uv(shape_types, bound)
  has_mesh = int(ShapeType.MESH) in shape_types or int(ShapeType.SDF) in shape_types
  has_primitive = any(t not in (int(ShapeType.MESH), int(ShapeType.SDF), int(ShapeType.FLEX)) for t in shape_types)

  @wp.func
  @device_generic
  def hit_uv[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
    mat_id: int,
  ) -> tuple[wp.vec2, bool]:
    """(texture coordinate, whether the hit has one) of instance hit (a material `mat_id` >= 0)."""
    gi = hit.instance_id
    c = geo.cut
    if wp.static(world_uv):
      m = batched(c.material_uv_world, worldid, mat_id)
      if world_projected(m):
        world, _tangent, _bitangent, _per_length = world_uv_frame(m, point, hit.normal, wp.identity(3, dtype=float))
        return world, True
    repeat = batched(c.material_tex_repeat, worldid, mat_id)
    gtype = geo.instance_type[gi]
    is_mesh = gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF)
    uv = wp.vec2(0.0)
    has_uv = False
    if wp.static(has_mesh):
      if is_mesh and hit.mesh_id >= 0 and hit.face_id >= 0:
        uv, has_uv = mesh_uv(geo, gi, hit.mesh_id, hit.face_id, hit.bary_u, hit.bary_v, repeat, mat_id)
    if wp.static(has_primitive):
      if not is_mesh:
        pos, mat = pose(geo, worldid, gi)
        mat_t = wp.transpose(mat)
        local_normal = mat_t @ hit.normal
        if wp.dot(local_normal, mat_t @ direction) > 0.0:
          local_normal = -local_normal
        uniform = batched(c.material_tex_uniform, worldid, mat_id)
        size = batched(geo.instance_size, worldid, gi)
        local = mat_t @ (point - pos)
        uv0, rep, offset, _grad_u, _grad_v = primitive_uv(geo.bits, gtype, local, local_normal, size, repeat, uniform)
        uv = wp.cw_mul(uv0, rep) + offset
        has_uv = True
    return uv, has_uv

  return hit_uv


@specialize
def make_cut_away(
  shape_types: tuple[int, ...], *, compact: bool, live_uv: bool = False, world_uv: bool = False, bound: int = 0
):
  """The alpha test of a hit (`CutAway`) for these instance types and pose layout (`bound`:
  Features.bound)."""
  hit_uv = make_hit_uv(shape_types, compact=compact, live_uv=live_uv, world_uv=world_uv, bound=bound)

  @wp.func
  @device_generic
  def cut_away[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
  ) -> bool:
    """Whether the alpha test discards `hit` (its opacity texel below the material's cutoff)."""
    gi = hit.instance_id
    if gi < 0:
      return False
    mat_id, cutoff, tex_id = cut_material(geo, worldid, gi)
    if cutoff <= 0.0:
      return False
    uv, has_uv = hit_uv(geo, worldid, hit, point, direction, mat_id)
    if not has_uv:
      return False
    c = geo.cut
    return texture.sample_layout(c.textures[tex_id], c.tex_layout[tex_id], uv, 0.0)[0] < cutoff

  return cut_away


@specialize
def make_word_cut_away(
  shape_types: tuple[int, ...], *, compact: bool, live_uv: bool = False, world_uv: bool = False, bound: int = 0
):
  """`make_cut_away` behind the build word's opacity-map bit (`toggles.MAP_OPACITY`, where `bound`,
  Features.bound, binds it): a module whose scene has no opacity map folds the test away. The OptiX
  analytic hits' test (planes, curves, custom primitives), which runs on every hit of its kind."""
  cut_away = make_cut_away(shape_types, compact=compact, live_uv=live_uv, world_uv=world_uv, bound=bound)
  field = toggles.bound_on(bound, high=True)

  @wp.func
  @device_generic
  def word_cut_away[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
  ) -> bool:
    """Whether the alpha test discards `hit` (never where the word has no opacity map)."""
    if not field(geo.bits2, wp.static(toggles.MAP_OPACITY)):
      return False
    return cut_away(geo, worldid, hit, point, direction)

  return word_cut_away


@wp.func
@device_generic
def world_cutout[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: int,
  gi: int,
  point: wp.vec3,
  normal: wp.vec3,
) -> tuple[bool, bool]:
  """(handled, rejected) of a world-projected opacity map, used by OptiX triangle any-hit."""
  mat_id, cutoff, tex_id = cut_material(geo, worldid, gi)
  if cutoff <= 0.0:
    return False, False
  c = geo.cut
  mapping = batched(c.material_uv_world, worldid, mat_id)
  if not world_projected(mapping):
    return False, False
  uv, _tangent, _bitangent, _density = world_uv_frame(mapping, point, normal, wp.identity(3, dtype=float))
  rejected = texture.sample_layout(c.textures[tex_id], c.tex_layout[tex_id], uv, 0.0)[0] < cutoff
  return True, rejected


@specialize
def make_cut_triangle(*, has_mesh: bool, has_box: bool, live_uv: bool = False, bound: int = 0):
  """cut_triangle(geo, worldid, gi, face, bary, object_origin, object_direction, t) -> whether the
  OptiX any-hit program ignores the candidate triangle hit (its alpha test). Mesh
  instance gi: face `face` at OptiX barycentrics `bary` (b1, b2). Box instance gi (accel's unit cube
  scaled to the half sizes): triangle `face` of the cube, the object-space ray (the cube's frame)
  at distance t. `bound`: Features.bound (the live UV mapping's bit).
  """
  mesh_uv = make_mesh_uv(live_uv=live_uv, bound=bound)
  primitive_uv = texture.make_primitive_uv((int(ShapeType.BOX),), 0)  # a box candidate: its own program

  @wp.func
  @device_generic
  def cut_triangle[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    gi: int,
    face: int,
    bary: wp.vec2,
    object_origin: wp.vec3,
    object_direction: wp.vec3,
    t: float,
  ) -> bool:
    """Whether the alpha test discards candidate triangle `face` of instance gi (OptiX any-hit)."""
    mat_id, cutoff, tex_id = cut_material(geo, worldid, gi)
    if cutoff <= 0.0:
      return False
    c = geo.cut
    repeat = batched(c.material_tex_repeat, worldid, mat_id)
    uv = wp.vec2(0.0)
    has_uv = False
    if wp.static(has_mesh):
      gtype = geo.instance_type[gi]
      if gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF):
        mesh_id = in_table(batched(geo.instance_mesh, worldid, gi), geo.nmesh)
        # OptiX (b1, b2) weight vertices 1 and 2; MuJoCo Warp's (u, v) vertices 0 and 1.
        uv, has_uv = mesh_uv(geo, gi, mesh_id, face, 1.0 - bary[0] - bary[1], bary[0], repeat, mat_id)
    if wp.static(has_box):
      if geo.instance_type[gi] == int(ShapeType.BOX):
        size = batched(geo.instance_size, worldid, gi)
        # Unit-cube face `face // 2` (accel): outward normal, towards the ray as cut_away's.
        k = face // 2
        # The face's axis by selects (a dynamic component write puts the vector in local memory).
        sign = wp.where(k % 2 == 0, 1.0, -1.0)
        axis = k // 2
        normal = wp.vec3(wp.where(axis == 0, sign, 0.0), wp.where(axis == 1, sign, 0.0), wp.where(axis == 2, sign, 0.0))
        if wp.dot(normal, object_direction) > 0.0:
          normal = -normal
        local = wp.cw_mul(object_origin + object_direction * t, size)
        uniform = batched(c.material_tex_uniform, worldid, mat_id)
        uv0, rep, offset, _grad_u, _grad_v = primitive_uv(geo.bits, int(ShapeType.BOX), local, normal, size, repeat, uniform)
        uv = wp.cw_mul(uv0, rep) + offset
        has_uv = True
    if not has_uv:
      return False
    return texture.sample_layout(c.textures[tex_id], c.tex_layout[tex_id], uv, 0.0)[0] < cutoff

  return cut_triangle


@wp.func
@device_generic
def cut_out[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: int,
  gi: int,
) -> bool:
  """Whether instance gi (< 0: none, or a deformable) is alpha tested: a cutoff and an opacity
  map."""
  if gi < 0:
    return False
  _mat_id, cutoff, _tex_id = cut_material(geo, worldid, gi)
  return cutoff > 0.0
