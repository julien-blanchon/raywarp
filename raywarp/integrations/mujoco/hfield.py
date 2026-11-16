"""MuJoCo height fields -> triangle BLASes, built on the device from the aliased `m.hfield_*`.

The triangle set is MuJoCo's own (`mjr_uploadHField`): every grid cell (r, c) is split along its
(r, c)-(r + 1, c + 1) diagonal, vertex (r, c) at (size[0] (c / w - 1), size[1] (r / h - 1),
size[2] data[r, c]) with w, h the half column / row counts, in float32 like MuJoCo's renderer;
only the top surface is traced (no sides or base, as MuJoCo Warp). Live terrain: after writing
`m.hfield_data`, `HeightFields.refit()` rewrites the vertices with the same kernel and refits the
BLASes and repacks their packed children (`trace.accel.packed`: same mesh ids and node addresses,
captured graphs stay valid). The scene BVHs bound a height field by its
`hfield_size` box (data in [0, 1], MuJoCo's convention), fixed at construction.
"""

from __future__ import annotations

import dataclasses
from typing import final

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from raywarp.core.trace.accel.packed import PackedNodes

# Forward-only kernels: no adjoint code.
wp.set_module_options({"enable_backward": False})


@wp.kernel
def _hfield_points(
  # Model:
  hfield_size: wp.array[wp.vec4],
  hfield_nrow: wp.array[int],
  hfield_ncol: wp.array[int],
  hfield_adr: wp.array[int],
  hfield_data: wp.array[float],
  # In:
  hfieldid: int,
  # Out:
  points_out: wp.array[wp.vec3],
):
  """Vertex (r, c) = thread r * ncol + c of height field `hfieldid` (module docstring)."""
  i = wp.tid()
  nc = hfield_ncol[hfieldid]
  r = i // nc
  c = i - r * nc
  sz = hfield_size[hfieldid]
  width = 0.5 * float(nc - 1)
  height = 0.5 * float(hfield_nrow[hfieldid] - 1)
  z = hfield_data[hfield_adr[hfieldid] + i]
  points_out[i] = wp.vec3(sz[0] * (float(c) / width - 1.0), sz[1] * (float(r) / height - 1.0), z * sz[2])


@wp.kernel
def _hfield_faces(
  # In:
  ncol: int,
  # Out:
  indices_out: wp.array[wp.int32],
):
  """The two triangles of cell (r, c) = thread r * (ncol - 1) + c, wound to face +z."""
  i = wp.tid()
  r = i // (ncol - 1)
  c = i - r * (ncol - 1)
  v00 = r * ncol + c
  v01 = v00 + 1
  v10 = v00 + ncol
  v11 = v10 + 1
  k = 6 * i
  indices_out[k + 0] = v00
  indices_out[k + 1] = v01
  indices_out[k + 2] = v11
  indices_out[k + 3] = v00
  indices_out[k + 4] = v11
  indices_out[k + 5] = v10


@wp.kernel
def _hfield_half(
  # Model:
  hfield_size: wp.array[wp.vec4],
  # Out:
  half_out: wp.array[wp.vec3],
):
  """Half extents of each height field's box (x, y half sizes, half height; `bounds.world_aabb`)."""
  h = wp.tid()
  sz = hfield_size[h]
  half_out[h] = wp.vec3(sz[0], sz[1], 0.5 * sz[2])


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class HeightFields:
  """The BLASes of the rendered height fields, over the aliased `m.hfield_*` arrays.

  Attributes:
    m: the device model (its hfield arrays are read by every build / refit).
    ids: (n,) the rendered height fields' ids.
    meshes: their wp.Mesh (built over `ids`, same order).
    placeholder: the BLAS of the others (one degenerate triangle, never hit; None: every height
      field is rendered): a later `m.geom_dataid` write onto one traces nothing, safely, and the
      live check reports it (`capabilities.LiveCapabilityCheck`).
    bvh_id: (nhfield,) uint64 BLAS id per height field (`placeholder`'s: not rendered),
      `Meshes.hfield_bvh_id`.
    half: (nhfield,) vec3 half extents, `Meshes.hfield_bounds_size`.
    nodes: the packed children of the BLASes (`Meshes.hfield_nodes`), repacked by `refit`.
  """

  m: mjw.Model
  ids: list[int]
  meshes: list[wp.Mesh]
  placeholder: wp.Mesh | None
  bvh_id: wp.array[wp.uint64, [int]]
  half: wp.array[wp.vec3, [int]]
  nodes: PackedNodes

  def refit(self) -> None:
    """Re-reads the terrain (`m.hfield_data`), refits the BLASes in place and repacks their
    children (the queries read the packed copy, not Warp's nodes)."""
    for hid, mesh in zip(self.ids, self.meshes, strict=True):
      _write_points(self.m, hid, mesh.points)
      mesh.refit()
    self.nodes.refresh()


def _write_points(m: mjw.Model, hfieldid: int, points: wp.array[wp.vec3, [int]]) -> None:
  """Writes height field `hfieldid`'s vertices from `m` into `points`."""
  wp.launch(
    _hfield_points,
    dim=points.shape[0],
    inputs=[m.hfield_size, m.hfield_nrow, m.hfield_ncol, m.hfield_adr, m.hfield_data, hfieldid],
    outputs=[points],
    device=points.device,
  )


def height_fields(mjm: mujoco.MjModel, m: mjw.Model, used: list[int]) -> HeightFields:
  """The BLASes of height fields `used` (ids), built on `m`'s device from `m`'s arrays.

  The grid sizes come from `mjm` (topology is static); the vertices from the device model.
  """
  device, nhfield = m.hfield_data.device, mjm.nhfield
  placeholder = _placeholder(device) if len(used) < nhfield else None
  bvh_id = [wp.uint64(0 if placeholder is None else placeholder.id)] * max(nhfield, 1)
  half = wp.zeros(max(nhfield, 1), dtype=wp.vec3, device=device)
  meshes = []
  for hid in used:
    nr, nc = int(mjm.hfield_nrow[hid]), int(mjm.hfield_ncol[hid])
    points = wp.empty(nr * nc, dtype=wp.vec3, device=device)
    _write_points(m, hid, points)
    indices = wp.empty(6 * (nr - 1) * (nc - 1), dtype=wp.int32, device=device)
    wp.launch(_hfield_faces, dim=(nr - 1) * (nc - 1), inputs=[nc], outputs=[indices], device=device)
    meshes.append(wp.Mesh(points=points, indices=indices, bvh_constructor="cubql", bvh_leaf_size=2))
    bvh_id[hid] = meshes[-1].id
  if nhfield:
    wp.launch(_hfield_half, dim=nhfield, inputs=[m.hfield_size], outputs=[half], device=device)
  ids = wp.array(np.array(bvh_id, np.uint64), dtype=wp.uint64, device=device)
  nodes = PackedNodes(ids, meshes=True)
  return HeightFields(m=m, ids=list(used), meshes=meshes, placeholder=placeholder, bvh_id=ids, half=half, nodes=nodes)


def _placeholder(device: wp.Device) -> wp.Mesh:
  """The BLAS of the height fields no rendered geom shows: one degenerate triangle, never hit."""
  points = wp.zeros(3, dtype=wp.vec3, device=device)
  indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
  return wp.Mesh(points=points, indices=indices, bvh_constructor="cubql", bvh_leaf_size=2)
