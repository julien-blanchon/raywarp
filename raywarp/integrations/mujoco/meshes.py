"""MuJoCo meshes and height fields -> core `Meshes`, aliasing MuJoCo Warp's mesh arrays.

The vertex, normal and face arrays are `m`'s own (no copy); the per-corner normal / UV index
tables are `m.mesh_face` itself where equal. Meshes are re-framed when that shrinks their BLAS
boxes (`mesh_frames`). Height fields: `hfield.py`.
"""

from __future__ import annotations

from typing import Final

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from raywarp._base.dims import Mask
from raywarp.core.pipeline.gpu_scene.gpu_scene import Meshes
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.integrations.mujoco.hfield import HeightFields
from raywarp.scene.model.types import MeshId

type Points = np.ndarray[[int, 3], np.dtype[np.floating]]
type Faces = np.ndarray[[int, 3], np.dtype[np.int32]]
type Rotation = np.ndarray[[3, 3], np.dtype[np.float64]]

# Re-frame a mesh BLAS when its summed triangle bounding-box area shrinks by at least this factor.
_MESH_FRAME_MIN_GAIN: Final = 1.25


def _bbox_area_sum(points: Points, faces: Faces) -> float:
  """Summed surface area of the triangles' axis-aligned bounding boxes (a BVH cost proxy)."""
  # Corner by corner (elementwise): a reduction over the corner axis of (n, 3, 3) is ~10x slower.
  a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
  e = np.maximum(np.maximum(a, b), c) - np.minimum(np.minimum(a, b), c)
  return float(np.sum(e[:, 0] * e[:, 1] + e[:, 1] * e[:, 2] + e[:, 2] * e[:, 0]))


def mesh_frames(mjm: mujoco.MjModel) -> dict[MeshId, tuple[Rotation, Points]]:
  """Authored-frame rotations of the meshes whose BLAS they improve: {mesh_id: (R, R v)}.

  MuJoCo's mesh frame is the principal inertia frame; R (from mesh_quat) rotates its vertices v
  back to the authored frame. CAD-style meshes are axis-aligned as authored, and rotated into
  the inertia frame their long triangles get large overlapping boxes (up to ~5x slower
  rendering). A mesh is kept when re-framing shrinks its summed triangle bounding-box area by
  at least _MESH_FRAME_MIN_GAIN.
  """
  out: dict[MeshId, tuple[Rotation, Points]] = {}
  for mesh_id in map(MeshId, range(mjm.nmesh)):
    if np.allclose(mjm.mesh_quat[mesh_id], [1.0, 0.0, 0.0, 0.0]):
      continue
    v0, nv = mjm.mesh_vertadr[mesh_id], mjm.mesh_vertnum[mesh_id]
    f0, nf = mjm.mesh_faceadr[mesh_id], mjm.mesh_facenum[mesh_id]
    points = np.asarray(mjm.mesh_vert[v0 : v0 + nv], np.float64).reshape(-1, 3)
    faces = np.asarray(mjm.mesh_face[f0 : f0 + nf], np.int32).reshape(-1, 3)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, mjm.mesh_quat[mesh_id])
    rot = rot.reshape(3, 3)
    framed = (points @ rot.T).reshape(-1, 3)
    if _bbox_area_sum(points, faces) < _MESH_FRAME_MIN_GAIN * _bbox_area_sum(framed, faces):
      continue
    out[mesh_id] = (rot, framed)
  return out


def _index_table(
  mjm: mujoco.MjModel,
  index: np.ndarray[[int], np.dtype[np.int32]],
  face: wp.array[wp.vec3i, [int]],
  adr: np.ndarray[[int], np.dtype[np.int32]] | None,
  device: wp.Device,
) -> wp.array[wp.vec3i, [int]]:
  """Device table of a per-corner index (normals, UVs): MuJoCo Warp's `mesh_face` itself when equal.

  `adr`: the per-mesh address of the indexed attribute (-1: none); the rows of meshes without
  it are never read and compare as equal.
  """
  index = np.array(index, np.int32).reshape(-1, 3)
  faces = mjm.mesh_face.reshape(-1, 3)
  if adr is not None:
    for mesh_id in np.flatnonzero(np.asarray(adr) < 0):
      span = slice(mjm.mesh_faceadr[mesh_id], mjm.mesh_faceadr[mesh_id] + mjm.mesh_facenum[mesh_id])
      index[span] = faces[span]
  return face if np.array_equal(index, faces) else wp.array(index, dtype=wp.vec3i, device=device)


def meshes(mjm: mujoco.MjModel, m: mjw.Model, hfields: HeightFields, *, device: wp.Device) -> Meshes:
  """The core meshes of the model: `m`'s arrays, host copies for the BLASes, the height fields
  (the tables MuJoCo Warp lacks on `device`, `m`'s)."""
  points = [mjm.mesh_vert[a : a + n] for a, n in zip(mjm.mesh_vertadr, mjm.mesh_vertnum, strict=True)]
  faces = [mjm.mesh_face[a : a + n] for a, n in zip(mjm.mesh_faceadr, mjm.mesh_facenum, strict=True)]
  reframed = mesh_frames(mjm)
  frames = None
  if reframed:
    frames = np.tile(np.eye(3, dtype=np.float32), (mjm.nmesh, 1, 1))
    for mesh_id, (rot, _) in reframed.items():
      frames[mesh_id] = rot
  return Meshes(
    vertadr=m.mesh_vertadr,
    faceadr=m.mesh_faceadr,
    normaladr=m.mesh_normaladr,
    vert=m.mesh_vert,
    normal=m.mesh_normal,
    face=m.mesh_face,
    facenormal=_index_table(mjm, mjm.mesh_facenormal, m.mesh_face, None, device),
    facetexcoord=_index_table(mjm, mjm.mesh_facetexcoord, m.mesh_face, mjm.mesh_texcoordadr, device),
    texcoord=wp.array(mjm.mesh_texcoord, dtype=wp.vec2, device=device),
    texcoord_adr=wp.array(mjm.mesh_texcoordadr, dtype=int, device=device),
    coloradr=wp.full(max(mjm.nmesh, 1), -1, dtype=int, device=device),  # MuJoCo meshes have no vertex colors
    facecolor=m.mesh_face,
    color=wp.ones(1, dtype=wp.vec4, device=device),
    points=points,
    faces=faces,
    frames=frames,
    framed_points={i: framed for i, (_, framed) in reframed.items()},
    hfield_bvh_id=hfields.bvh_id,
    hfield_bounds_size=hfields.half,
    hfield_registry=dict(zip(hfields.ids, hfields.meshes, strict=True)),
    hfield_nodes=hfields.nodes.addresses,
  )


def used_hfields(mjm: mujoco.MjModel, m: mjw.Model, rendered: Mask[int]) -> list[int]:
  """Ids of the height fields a rendered height-field geom shows in some batched row of
  `m.geom_dataid` (the others get no BLAS, as `build.used_meshes`)."""
  if not mjm.nhfield or not mjm.ngeom:
    return []
  ids = host_copy(m.geom_dataid).reshape(-1, mjm.ngeom)[:, rendered & (mjm.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)]
  return sorted({int(h) for h in ids[(ids >= 0) & (ids < mjm.nhfield)]})
