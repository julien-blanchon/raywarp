"""The per-hit texture frame: UVs, tangents and ray-cone level of detail.

Primitive UVs use MuJoCo's mapping (texrepeat / texuniform, see uv.py). Mesh UVs are the mesh's
own, or the instance's (`MeshInstances`: another UV set's corner rows and an affine UV
transform, `triangle_uvs`); for meshes the UV derivatives come from the hit triangle, stretched
by the instance's stretch; for primitives they are the analytic gradients of the
mapping. The derivatives give both the normal-map tangent frame and the
texture LOD of a ray cone (Akenine-Möller et al. 2019).
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import TEX_LAYOUT_IDENTITY, UV_MESH, MeshAdr, ShapeType, uv_xform
from raywarp.core.kernels.abi import toggles
from raywarp.core.shade.materials.uv import (
  texture_box,
  texture_capsule,
  texture_cylinder,
  texture_hfield,
  texture_plane,
  texture_sphere_ellipsoid,
)

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def orthonormal_basis(
  # In:
  n: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """Branchless orthonormal basis (Duff et al. 2017)."""
  sign = wp.where(n[2] >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + n[2])
  b = n[0] * n[1] * a
  return wp.vec3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]), wp.vec3(b, sign + n[1] * n[1] * a, -n[1])


# Floor of squared distances to an azimuth axis and of cos^2 elevation (UV gradients at the poles).
_MIN_AXIS_SQ: Final = 1.0e-12


@wp.func
def azimuth_gradient(
  # In:
  x: float,
  y: float,
) -> wp.vec3:
  """Gradient of atan2(y, x) / 2 pi (the azimuthal u of spheres, cylinders, capsules) in (x, y)."""
  inv = wp.static(0.5 / wp.pi) / wp.max(x * x + y * y, wp.static(_MIN_AXIS_SQ))
  return wp.vec3(-y * inv, x * inv, 0.0)


@wp.func
def plane_gradients(
  # In:
  size: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(grad u, grad v) of `uv.texture_plane` (infinite axes: 0.5 per unit)."""
  gu = wp.where(size[0] > 0.0, 0.5 / wp.max(size[0], 1e-6), 0.5)
  gv = wp.where(size[1] > 0.0, -0.5 / wp.max(size[1], 1e-6), -0.5)
  return wp.vec3(gu, 0.0, 0.0), wp.vec3(0.0, gv, 0.0)


@wp.func
def ellipsoid_gradients(
  # In:
  local: wp.vec3,
  size: wp.vec3,
  is_sphere: bool,
) -> tuple[wp.vec3, wp.vec3]:
  """(grad u, grad v) of `uv.texture_sphere_ellipsoid` (azimuth, 1/2 - elevation / pi)."""
  radii = size
  if is_sphere:
    radii = wp.vec3(size[0], size[0], size[0])
  inv_r = wp.vec3(1.0 / wp.max(radii[0], 1e-6), 1.0 / wp.max(radii[1], 1e-6), 1.0 / wp.max(radii[2], 1e-6))
  pt = wp.cw_mul(local, inv_r)
  length = wp.max(wp.length(pt), 1e-6)
  dz = wp.clamp(pt[2] / length, -1.0, 1.0)
  # d(dz)/d(pt) = (e_z - d dz) / |pt|; d(asin)/d(dz) = 1 / sqrt(1 - dz^2).
  ddz = (wp.vec3(0.0, 0.0, 1.0) - pt * (dz / length)) / length
  gv = ddz * (-wp.static(1.0 / wp.pi) / wp.sqrt(wp.max(1.0 - dz * dz, wp.static(_MIN_AXIS_SQ))))
  return wp.cw_mul(azimuth_gradient(pt[0], pt[1]), inv_r), wp.cw_mul(gv, inv_r)


@wp.func
def cylinder_gradients(
  # In:
  local: wp.vec3,
  local_normal: wp.vec3,
  size: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(grad u, grad v) of `uv.texture_cylinder` (caps: planar disks; body: azimuth, height)."""
  r = wp.max(size[0], 1e-6)
  if wp.abs(local_normal[2]) > 0.7071:
    return wp.vec3(0.5 / r, 0.0, 0.0), wp.vec3(0.0, 0.5 / r, 0.0)
  return azimuth_gradient(local[0], local[1]), wp.vec3(0.0, 0.0, -0.5 / wp.max(size[1], 1e-6))


@wp.func
def capsule_gradients(
  # In:
  local: wp.vec3,
  size: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(grad u, grad v) of `uv.texture_capsule` (azimuth; height on the body, arc on the caps)."""
  r = wp.max(size[0], 1e-6)
  h = wp.max(size[1], 1e-6)
  z = local[2]
  gv = -0.5 / h
  if z > h or z < -h:
    # v = 1 - asin((z - h) / r) 2 / pi on the top cap, asin((-h - z) / r) 2 / pi on the bottom.
    dz = wp.clamp((wp.abs(z) - h) / r, 0.0, 1.0)
    gv = -wp.static(2.0 / wp.pi) / (r * wp.sqrt(wp.max(1.0 - dz * dz, wp.static(_MIN_AXIS_SQ))))
  return azimuth_gradient(local[0], local[1]), wp.vec3(0.0, 0.0, gv)


@wp.func
def box_gradients(
  # In:
  local_normal: wp.vec3,
  size: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(grad u, grad v) of `uv.texture_box` (the face's planar mapping)."""
  hx = 0.5 / wp.max(size[0], 1e-6)
  hy = 0.5 / wp.max(size[1], 1e-6)
  hz = 0.5 / wp.max(size[2], 1e-6)
  ax = wp.abs(local_normal[0])
  ay = wp.abs(local_normal[1])
  az = wp.abs(local_normal[2])
  if az >= ax and az >= ay:
    return wp.vec3(hx, 0.0, 0.0), wp.vec3(0.0, wp.where(local_normal[2] > 0.0, -hy, hy), 0.0)
  if ax >= ay and ax >= az:
    return wp.vec3(0.0, wp.where(local_normal[0] > 0.0, hy, -hy), 0.0), wp.vec3(0.0, 0.0, -hz)
  return wp.vec3(wp.where(local_normal[1] < 0.0, hx, -hx), 0.0, 0.0), wp.vec3(0.0, 0.0, -hz)


@wp.func
def hfield_gradients(
  # In:
  size: wp.vec3,
) -> tuple[wp.vec3, wp.vec3]:
  """(grad u, grad v) of `uv.texture_hfield`."""
  return wp.vec3(0.5 / wp.max(size[0], 1e-6), 0.0, 0.0), wp.vec3(0.0, -0.5 / wp.max(size[1], 1e-6), 0.0)


def make_primitive_uv(instance_types: tuple[int, ...], bound: int):  # noqa: C901 (a static branch and shape gate per type)
  """Build MuJoCo's analytic UV mapping for the primitive types present.

  Returns primitive_uv(bits, gtype, local, local_normal, size, repeat, uniform) -> (uv, repeat,
  offset, grad u, grad v): the texture coordinate is uv ⊙ repeat + offset, and the gradients of
  the unrepeated uv in the shape's frame are the mapping's analytic ones (no finite differences,
  no seam folding). `bound` (Features.bound): an analytic type's
  mapping runs where the build word `bits` has its shape bit (`toggles.SHAPE_BITS`).
  """
  gate = toggles.bound_on(bound)
  sphere, ellipsoid = int(ShapeType.SPHERE), int(ShapeType.ELLIPSOID)
  round_bits = toggles.SHAPE_BITS[sphere] | toggles.SHAPE_BITS[ellipsoid]

  @wp.func
  def primitive_uv(  # noqa: C901 (a static branch and shape gate per type)
    # In:
    bits: int,
    gtype: int,
    local: wp.vec3,
    local_normal: wp.vec3,
    size: wp.vec3,
    repeat: wp.vec2,
    uniform: bool,
  ) -> tuple[wp.vec2, wp.vec2, wp.vec2, wp.vec3, wp.vec3]:
    """UV, repeat, offset and UV gradients of primitive type `gtype` at shape-frame point
    `local`."""
    uv = wp.vec2(0.0)
    offset = wp.vec2(0.0)
    rep = repeat
    gu = wp.vec3(0.0)
    gv = wp.vec3(0.0)
    if wp.static(int(ShapeType.PLANE) in instance_types):
      if gtype == ShapeType.PLANE:
        uv, offset, rep = texture_plane(local, size, repeat, uniform)
        gu, gv = plane_gradients(size)
    if wp.static(sphere in instance_types or ellipsoid in instance_types):
      if gate(bits, wp.static(round_bits)):
        if gtype == ShapeType.SPHERE or gtype == ShapeType.ELLIPSOID:
          uv, offset, rep = texture_sphere_ellipsoid(local, size, repeat, uniform, gtype == ShapeType.SPHERE)
          gu, gv = ellipsoid_gradients(local, size, gtype == ShapeType.SPHERE)
    if wp.static(int(ShapeType.CYLINDER) in instance_types):
      if gate(bits, wp.static(toggles.SHAPE_BITS[int(ShapeType.CYLINDER)])):
        if gtype == ShapeType.CYLINDER:
          uv, offset, rep = texture_cylinder(local, local_normal, size, repeat, uniform)
          gu, gv = cylinder_gradients(local, local_normal, size)
    if wp.static(int(ShapeType.CAPSULE) in instance_types):
      if gate(bits, wp.static(toggles.SHAPE_CAPSULE)):
        if gtype == ShapeType.CAPSULE:
          uv, offset, rep = texture_capsule(local, size, repeat, uniform)
          gu, gv = capsule_gradients(local, size)
    if wp.static(int(ShapeType.BOX) in instance_types):
      if gate(bits, wp.static(toggles.SHAPE_BOX)):
        if gtype == ShapeType.BOX:
          uv, offset, rep = texture_box(local, local_normal, size, repeat, uniform)
          gu, gv = box_gradients(local_normal, size)
    if wp.static(int(ShapeType.HFIELD) in instance_types):
      if gtype == ShapeType.HFIELD:
        uv, offset, rep = texture_hfield(local, size, repeat, uniform)
        gu, gv = hfield_gradients(size)
    return uv, rep, offset, gu, gv

  return primitive_uv


@wp.func
def frame_from_uv_jacobian(
  # In:
  t1: wp.vec3,
  t2: wp.vec3,
  du_dt: wp.vec2,
  dv_dt: wp.vec2,
) -> tuple[wp.vec3, wp.vec3, float]:
  """Tangent (∂p/∂u), bitangent (∂p/∂v) and UV-per-length from the UV Jacobian on (t1, t2)."""
  det = du_dt[0] * dv_dt[1] - du_dt[1] * dv_dt[0]
  inv = 1.0 / wp.where(wp.abs(det) > 1.0e-20, det, 1.0e-20)
  tangent = (t1 * dv_dt[1] - t2 * dv_dt[0]) * inv
  bitangent = (t2 * du_dt[0] - t1 * du_dt[1]) * inv
  # Isotropic footprint scale: square root of the UV area per unit surface area.
  uv_per_length = wp.sqrt(wp.abs(det))
  return tangent, bitangent, uv_per_length


@wp.func
def affine_uv(
  # In:
  m: uv_xform,
  uv: wp.vec2,
) -> wp.vec2:
  """`uv` through the 2 x 3 affine map `m`."""
  return m @ wp.vec3(uv[0], uv[1], 1.0)


@wp.func
def triangle_uvs(
  # Model:
  mesh_facetexcoord: wp.array1d[wp.vec3i],
  mesh_texcoord: wp.array1d[wp.vec2],
  # In:
  mesh_texcoord_adr: int,
  mesh_face_adr: int,
  adr: wp.vec2i,
  m: uv_xform,
  face_id: int,
) -> tuple[bool, wp.vec2, wp.vec2, wp.vec2]:
  """(has UVs, the corner UVs) of a mesh face as an instance reads them (`mesh_texcoord_adr` /
  `mesh_face_adr`: the mesh's first UV, -1 none, and first triangle).

  `adr` / `m`: the instance's (texcoord address, corner row address) and UV transform when it has
  its own (`MeshInstance.texcoord`), else (UV_MESH, -): the mesh's UVs.
  """
  own = adr[0] != wp.static(UV_MESH)
  tc_adr = mesh_texcoord_adr
  row = mesh_face_adr + face_id
  if own:
    tc_adr = adr[0]
    row = adr[1] + face_id
  if tc_adr < 0:
    return False, wp.vec2(0.0), wp.vec2(0.0), wp.vec2(0.0)
  tc = mesh_facetexcoord[row]
  if tc[0] < 0 or tc[1] < 0 or tc[2] < 0:
    return False, wp.vec2(0.0), wp.vec2(0.0), wp.vec2(0.0)
  uv0 = mesh_texcoord[tc_adr + tc[0]]
  uv1 = mesh_texcoord[tc_adr + tc[1]]
  uv2 = mesh_texcoord[tc_adr + tc[2]]
  if own:
    uv0 = affine_uv(m, uv0)
    uv1 = affine_uv(m, uv1)
    uv2 = affine_uv(m, uv2)
  return True, uv0, uv1, uv2


@wp.func
def triangle_frame(
  # In:
  e1: wp.vec3,
  e2: wp.vec3,
  bary_u: float,
  bary_v: float,
  uv0: wp.vec2,
  uv1: wp.vec2,
  uv2: wp.vec2,
  repeat: wp.vec2,
) -> tuple[wp.vec2, wp.vec3, wp.vec3, float]:
  """(uv, tangent, bitangent, uv_per_length) at barycentrics (u, v) of a triangle of edges
  e1 = p1 - p0, e2 = p2 - p0 and corner UVs uv0..uv2 (times `repeat`); the frame in the edges'
  frame."""
  uv0 = wp.cw_mul(uv0, repeat)
  uv1 = wp.cw_mul(uv1, repeat)
  uv2 = wp.cw_mul(uv2, repeat)
  uv = uv0 * bary_u + uv1 * bary_v + uv2 * (1.0 - bary_u - bary_v)
  d1 = uv1 - uv0
  d2 = uv2 - uv0
  det = d1[0] * d2[1] - d2[0] * d1[1]
  inv = 1.0 / wp.where(wp.abs(det) > 1.0e-20, det, 1.0e-20)
  tangent = (e1 * d2[1] - e2 * d1[1]) * inv
  bitangent = (e2 * d1[0] - e1 * d2[0]) * inv
  area = wp.length(wp.cross(e1, e2))
  uv_per_length = wp.sqrt(wp.abs(det) / wp.max(area, 1.0e-20))
  return uv, tangent, bitangent, uv_per_length


@wp.func
def mesh_triangle_frame(
  # Model:
  mesh_vert: wp.array1d[wp.vec3],
  mesh_face: wp.array1d[wp.vec3i],
  # In:
  mesh: MeshAdr,
  face_id: int,
  bary_u: float,
  bary_v: float,
  uv0: wp.vec2,
  uv1: wp.vec2,
  uv2: wp.vec2,
  repeat: wp.vec2,
) -> tuple[wp.vec2, wp.vec3, wp.vec3, float]:
  """(uv, tangent, bitangent, uv_per_length) of a mesh hit with corner UVs uv0..uv2, shape frame."""
  vadr = mesh.vert
  f = mesh_face[mesh.face + face_id]
  p0 = mesh_vert[vadr + f[0]]
  return triangle_frame(mesh_vert[vadr + f[1]] - p0, mesh_vert[vadr + f[2]] - p0, bary_u, bary_v, uv0, uv1, uv2, repeat)


# Smallest |n.d| of the footprint (a ray grazing the surface): its elongation is at most 400.
MIN_NOD: Final = 0.0025


@wp.func(inline=True)
def surface_cone(
  # In:
  footprint: float,
  local_normal: wp.vec3,
  local_dir: wp.vec3,
) -> float:
  """Ray-cone footprint on the surface: width w across the view direction, w / |n.d| along it.

  One isotropic mip level for both axes is sized by their geometric mean, w / sqrt(|n.d|): the
  major axis alone (w / |n.d|) blurs surfaces seen at grazing angles, the minor one aliases
  (moiré); surfaces.anisotropy > 1 spreads taps along the major axis instead.
  """
  return footprint / wp.sqrt(wp.max(wp.abs(wp.dot(local_normal, local_dir)), wp.static(MIN_NOD)))


@wp.func
def texture_lod(
  # In:
  footprint: float,
  uv_per_length: float,
  tex_size: wp.vec2,
) -> float:
  """Mip level for a ray-cone footprint (world width at the hit, already divided by |n·d|)."""
  texels = footprint * uv_per_length * wp.max(tex_size[0], tex_size[1])
  return wp.log2(wp.max(texels, 1.0e-8))


@wp.func
def sample(
  # Model:
  tex: wp.Texture2D,
  # In:
  uv: wp.vec2,
  lod: float,
) -> wp.vec4:
  """Texel at `uv` and mip level `lod` (bilinear, trilinear between levels)."""
  return wp.texture_sample(tex, uv, dtype=wp.vec4, lod=wp.max(lod, 0.0))


@wp.func(inline=True)
def _read_channel(
  # In:
  texel: wp.vec4,
  field: int,
) -> float:
  """Stored channel `field` (0-3) of a sampled texel, 1 for TEX_LAYOUT_ONE."""
  return wp.where(
    field == 0,
    texel[0],
    wp.where(field == 1, texel[1], wp.where(field == 2, texel[2], wp.where(field == 3, texel[3], 1.0))),
  )


@wp.func
def apply_layout(
  # In:
  texel: wp.vec4,
  layout: int,
) -> wp.vec4:
  """The four read channels of a texel sampled from a texture of texel layout `layout`
  (`types.TEX_LAYOUT_*`: stored channel or 1 per read channel)."""
  if layout == wp.static(TEX_LAYOUT_IDENTITY):
    return texel
  out = wp.vec4(
    _read_channel(texel, layout & 7),
    _read_channel(texel, (layout >> 3) & 7),
    _read_channel(texel, (layout >> 6) & 7),
    _read_channel(texel, (layout >> 9) & 7),
  )
  return out


@wp.func
def sample_layout(
  # Model:
  tex: wp.Texture2D,
  # In:
  layout: int,
  uv: wp.vec2,
  lod: float,
) -> wp.vec4:
  """`sample` read through the texture's texel layout (`apply_layout`)."""
  return apply_layout(sample(tex, uv, lod), layout)
