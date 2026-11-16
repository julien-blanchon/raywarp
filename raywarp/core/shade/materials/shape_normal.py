"""The normal used for shading at a hit (smooth interpolated normals on meshes, vertex colors, the
curvature of primitives): a piece of `make_eval_surface`.

Pieces: `vertex_color` (per-vertex colors, Features.vertex_colors), `smooth_normal`
(interpolated mesh normals, back-face side, Hanika lift, curvature for specular AA),
`primitive_curvature` (analytic curvature of spheres, capsules, ...).
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import ShadingFlag, ShapeType
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import GeometryFields, Hit
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.shade.materials.material import MaterialTables, Surface
from raywarp.core.shade.materials.uv import vertex_normal

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Floor of the lengths the curvature estimates divide by [m] (a degenerate triangle, a zero-size
# primitive): a huge but finite curvature, widened by specular AA to its threshold.
_MIN_CURVATURE_LENGTH: Final = 1.0e-9


# Smallest n.v a shading normal keeps toward the viewer (`face_viewer`): a normal bent to exactly
# 0 would sit on the BRDF's grazing singularity.
_MIN_NOV: Final = 0.01


@wp.func
def face_viewer(
  # In:
  normal: wp.vec3,
  to_eye: wp.vec3,
) -> wp.vec3:
  """The shading normal bent toward the viewer just enough to face it (unchanged if it does).

  A smooth (interpolated or mapped) normal on a front face can point away from the eye near a
  silhouette. Flipping it mirrors it into the surface (dark rims); bending it along `to_eye` until
  n.v reaches _MIN_NOV keeps its direction across the view instead.
  """
  nov = wp.dot(normal, to_eye)
  if nov >= wp.static(_MIN_NOV):
    return normal
  return wp.normalize(normal + to_eye * (wp.static(_MIN_NOV) - nov))


@wp.func
@device_generic
def vertex_color[M: MaterialTables](
  # Model:
  mats: M,
  # In:
  hit: Hit,
  surface: Surface,
  enabled: bool,
) -> Surface:
  """`surface` with its mesh's vertex color (glTF COLOR_0, USD displayColor) multiplied in.

  Only where `enabled` (the instance's `ShadingFlag.VERTEX_COLOR`) and the mesh has colors.
  """
  mesh = mats.mesh_adr[hit.mesh_id]
  cadr = mesh.color
  if cadr >= 0 and enabled:
    ci = mats.mesh_facecolor[mesh.face + hit.face_id]
    vc = (
      mats.mesh_color[cadr + ci[0]] * hit.bary_u
      + mats.mesh_color[cadr + ci[1]] * hit.bary_v
      + mats.mesh_color[cadr + ci[2]] * (1.0 - hit.bary_u - hit.bary_v)
    )
    surface.base_color = wp.cw_mul(surface.base_color, wp.vec3(vc[0], vc[1], vc[2]))
    surface.alpha = surface.alpha * vc[3]
  return surface


@specialize
def make_smooth_normal(*, specular_aa: bool):
  """smooth_normal(mats, hit, nadr, local_normal) -> (shading, side, lift, curvature)."""

  @wp.func
  @device_generic
  def smooth_normal[M: MaterialTables](
    # Model:
    mats: M,
    # In:
    hit: Hit,
    nadr: wp.int32,
    local_normal: wp.vec3,
  ) -> tuple[wp.vec3, wp.float32, wp.vec3, wp.float32]:
    """A smooth mesh's shading normal at a hit, in the mesh frame.

    Returns (interpolated normal, side the authored normals face: +1 / -1, the Hanika lift of
    the secondary-ray origin, normal curvature for specular AA). `local_normal`: the geometric
    normal turned to the viewer.
    """
    face_adr = mats.mesh_adr[hit.mesh_id].face + hit.face_id
    b0 = hit.bary_u
    b1 = hit.bary_v
    b2 = 1.0 - hit.bary_u - hit.bary_v
    n0, n1, n2, side = _corner_normals(mats, nadr, face_adr, b0, b1, b2, local_normal)
    # vertex_normal keeps each corner within ~37 degrees of the viewer-side face normal, so
    # the interpolated normal is on that side: it is never flipped on the view direction
    # (near a silhouette it may still face away from the viewer: bent by the caller).
    shading = wp.normalize(n0 * b0 + n1 * b1 + n2 * b2)
    p0, p1, p2 = _corner_points(mats, hit, face_adr)
    # Shadow-terminator offset (Hanika 2021): the hit projected onto each corner's
    # tangent plane (only when below it), blended barycentrically, approximates the smooth
    # surface the normals describe; rays leave from there, above the flat triangle.
    q = p0 * b0 + p1 * b1 + p2 * b2
    lift = (
      n0 * (-b0 * wp.min(wp.dot(q - p0, n0), 0.0))
      + n1 * (-b1 * wp.min(wp.dot(q - p1, n1), 0.0))
      + n2 * (-b2 * wp.min(wp.dot(q - p2, n2), 0.0))
    )
    curvature = wp.float32(0.0)
    if wp.static(specular_aa):
      edge = wp.sqrt(wp.length(wp.cross(p1 - p0, p2 - p0)))
      spread = wp.max(wp.length(n0 - n1), wp.max(wp.length(n1 - n2), wp.length(n2 - n0)))
      curvature = spread / wp.max(edge, wp.static(_MIN_CURVATURE_LENGTH))
    return shading, side, lift, curvature

  return smooth_normal


@wp.func(inline=True)
@device_generic
def _corner_normals[M: MaterialTables](
  # Model:
  mats: M,
  # In:
  nadr: wp.int32,
  face_adr: wp.int32,
  b0: wp.float32,
  b1: wp.float32,
  b2: wp.float32,
  local_normal: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.float32]:
  """(the hit triangle's three vertex normals in the mesh frame, viewer side (`vertex_normal`),
  the side the authored normals face)."""
  tri = mats.mesh_facenormal[face_adr]
  r0 = mats.mesh_normal[nadr + tri[0]]
  r1 = mats.mesh_normal[nadr + tri[1]]
  r2 = mats.mesh_normal[nadr + tri[2]]
  # The side the authored normals face: seen from behind (or wound against its
  # normals) a smooth mesh keeps its interpolated normals, turned to the viewer.
  side = wp.float32(1.0)
  if wp.dot(r0 * b0 + r1 * b1 + r2 * b2, local_normal) < 0.0:
    side = -1.0
    r0 = -r0
    r1 = -r1
    r2 = -r2
  return vertex_normal(r0, local_normal), vertex_normal(r1, local_normal), vertex_normal(r2, local_normal), side


@wp.func(inline=True)
@device_generic
def _corner_points[M: MaterialTables](
  # Model:
  mats: M,
  # In:
  hit: Hit,
  face_adr: wp.int32,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """The hit triangle's vertices in the mesh frame."""
  f = mats.mesh_face[face_adr]
  vadr = mats.mesh_adr[hit.mesh_id].vert
  return mats.mesh_vert[vadr + f[0]], mats.mesh_vert[vadr + f[1]], mats.mesh_vert[vadr + f[2]]


@wp.func
@device_generic
def primitive_curvature[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: wp.int32,
  instance_id: wp.int32,
  gtype: wp.int32,
) -> wp.float32:
  """Normal curvature of a primitive for specular AA (1 / radius; 0: flat)."""
  size = batched(geo.instance_size, worldid, instance_id)
  if gtype == ShapeType.SPHERE or gtype == ShapeType.CAPSULE or gtype == ShapeType.CYLINDER:
    return 1.0 / wp.max(size[0], wp.static(_MIN_CURVATURE_LENGTH))
  if gtype == ShapeType.ELLIPSOID:
    return 1.0 / wp.max(wp.min(size[0], wp.min(size[1], size[2])), wp.static(_MIN_CURVATURE_LENGTH))
  return 0.0


@specialize
def make_shape_normal(f: Features, *, has_mesh: bool, has_primitive: bool):
  """shape_normal(...) -> (surface, shading, side, lift, curvature): the shape's normal at a hit."""
  vertex_colors = f.vertex_colors and has_mesh
  vertex_normals = f.vertex_normals and has_mesh
  primitive_aa = f.specular_aa and has_primitive
  smooth_normal = make_smooth_normal(specular_aa=f.specular_aa)
  gate = toggles.bound_on(f.bound)  # the materials class bit (vertex colors)
  field = toggles.bound_on(f.bound, high=True)  # the vertex colours' field bit (`toggles.MAT_VERTEX_COLORS`)

  @wp.func
  @device_generic
  def shape_normal[G: GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    # In:
    worldid: wp.int32,
    hit: Hit,
    gtype: wp.int32,
    is_mesh: bool,
    scale: wp.float32,
    local_normal: wp.vec3,
    side: wp.float32,
    surface: Surface,
  ) -> tuple[Surface, wp.vec3, wp.float32, wp.vec3, wp.float32]:
    """(surface with vertex colors, shading normal, side, Hanika lift, curvature), shape frame
    (world units: a mesh's lift and curvature, read from its vertices, scaled by its instance's
    `scale`).

    Flat shapes and meshes without normals keep the geometric normal `local_normal` (viewer
    side) and no lift.
    """
    shading = local_normal
    # Secondary rays leave from the hit (moved onto a smooth mesh's vertex tangent planes).
    lift = wp.vec3(0.0)
    curvature = wp.float32(0.0)
    on_triangle = is_mesh and hit.mesh_id >= 0 and hit.face_id >= 0
    if wp.static(vertex_colors):
      colors = gate(surface.bits, wp.static(toggles.CLASS_MATERIALS)) and field(
        surface.bits2, wp.static(toggles.MAT_VERTEX_COLORS)
      )
      if colors and on_triangle:
        # The instance's material multiplies its mesh's vertex colors in (`ShadingFlag.VERTEX_COLOR`
        # of its shading bits: the prepared material's, also under declared material selection).
        colored = (batched(mats.instance_shading, worldid, hit.instance_id) & wp.static(int(ShadingFlag.VERTEX_COLOR))) != 0
        surface = vertex_color(mats, hit, surface, colored)
    if wp.static(vertex_normals):
      # Meshes without normals (`MeshAdr.normal` -1) keep the flat face normal.
      if on_triangle:
        nadr = mats.mesh_adr[hit.mesh_id].normal
        if nadr >= 0:
          shading, side, lift, curvature = smooth_normal(mats, hit, nadr, local_normal)
          lift = lift * scale
          curvature = curvature / scale
    if wp.static(primitive_aa):
      if not is_mesh:
        curvature = primitive_curvature(geo, worldid, hit.instance_id, gtype)
    return surface, shading, side, lift, curvature

  return shape_normal
