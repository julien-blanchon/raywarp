"""Build inputs of the static triangle GASes (`gas.mesh_gases`): mesh face normals, packed once
at build time with the octahedral snorm16 code the hit programs use
(`core/shade/materials/octahedral.py`), and the unit cube every box instance scales."""

from __future__ import annotations

import numpy as np
import warp as wp

from raywarp.core.shade.materials.octahedral import pack_unit

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def face_normals(
  # Model:
  mesh_vertadr: wp.array1d[int],
  mesh_vert: wp.array1d[wp.vec3],
  mesh_face: wp.array1d[wp.vec3i],
  mesh_frame: wp.array1d[wp.mat33],
  # In:
  face_mesh: wp.array1d[int],
  # Out:
  normal_out: wp.array1d[wp.uint32],
):
  """Packed object-space normal of face f in its mesh's GAS frame (+z when degenerate)."""
  f = wp.tid()
  m = face_mesh[f]
  tri = mesh_face[f]
  va = mesh_vertadr[m]
  rot = mesh_frame[m]
  v0 = rot @ mesh_vert[va + tri[0]]
  n = wp.cross(rot @ mesh_vert[va + tri[1]] - v0, rot @ mesh_vert[va + tri[2]] - v0)
  if wp.abs(n[0]) + wp.abs(n[1]) + wp.abs(n[2]) == 0.0:
    n = wp.vec3(0.0, 0.0, 1.0)
  normal_out[f] = pack_unit(n)


def unit_cube() -> tuple[np.ndarray[[8, 3], np.dtype[np.float32]], np.ndarray[[12, 3], np.dtype[np.int32]]]:
  """Vertices and triangles of the cube [-1, 1]^3.

  Triangles 2a and 2a + 1 form face a (normal `trace.cube_face_normal(a)`), wound
  counter-clockwise seen from outside.
  """
  vertices = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float32)
  faces = []
  for a in range(6):
    axis, sign = a // 2, 1.0 if a % 2 == 0 else -1.0
    u, v = (axis + 1) % 3, (axis + 2) % 3
    corner = []
    for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
      p = np.zeros(3)
      p[axis], p[u], p[v] = sign, cu * sign, cv
      corner.append(int(np.flatnonzero(np.all(vertices == p, axis=1))[0]))
    faces += [(corner[0], corner[1], corner[2]), (corner[0], corner[2], corner[3])]
  return vertices, np.array(faces, dtype=np.int32)
