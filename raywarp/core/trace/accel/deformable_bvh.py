"""Triangle BVHs of the deformable surfaces (`core/pipeline/gpu_scene/deformables.py`), built once,
refit per frame.

Each dim-2 / dim-3 part gets one `wp.Mesh` holding its world-space triangles of every world,
grouped by world (`group_root`: the root of world w's subtree, what `meshes.ray_flex_with_bvh`
starts from). The triangles are rewritten from the current vertices every frame, then the mesh
BVH is refit (its topology comes from the initial vertices):

* dim 2: two offset copies of each mid-surface triangle (+radius and -radius along the smooth
  vertex normals, the second wound the other way: both face outwards) and two triangles per
  boundary edge (the side wall), as MuJoCo renders flex;
* dim 3: the surface triangles as they are.

Vertex normals are gathered per vertex from its adjacent face normals (no atomics: deterministic
on every device, sums in face order) into persistent buffers: a frame allocates nothing.

Vendored from MuJoCo Warp (google-deepmind/mujoco_warp 3.17.0, `_src/bvh.py`: `build_flex_bvh`,
`refit_flex_bvh`, `accumulate_flex_vertex_normals`, `_update_flex_2d_face_points`,
`_update_flex_3d_face_points`); the per-triangle arithmetic is unchanged (same operation order,
bit-identical points). Changed: the core's `Deformables` record instead of the MuJoCo model, the
normals gathered instead of accumulated with atomics, the build writing the refit's triangles
(MuJoCo's build kernels emit the side walls' second triangle in another vertex order, which its
first refit overwrites), one 1-row vertex table read by every world (no host tile per world),
group roots written on the device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.rays import FlexPart
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.pipeline.gpu_scene.deformables import surface_triangles
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.trace.accel.packed import PackedNodes

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import Ids, Table
  from raywarp.core.pipeline.gpu_scene.deformables import Deformables, Triangles

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# MuJoCo Warp's flex BVH construction (`build_flex_bvh` defaults).
CONSTRUCTOR: Final = "sah"
LEAF_SIZE: Final = 2
# Dimension of a thick-surface part (two offset copies of its mid-surface plus side walls; 3: a
# closed volume's surface as it is, 1: segments, traced as scene-BVH capsules).
THICK_SURFACE: Final = 2


@wp.kernel
def part_table(
  # Model:
  rgba: wp.array[wp.vec4],
  material: wp.array[int],
  nface: wp.array[int],
  uvadr: wp.array[int],
  mesh: wp.array[wp.uint64],
  # Out:
  part_out: wp.array[FlexPart],
):
  """`FlexPart` p from the deformables' per-part tables and its mesh."""
  p = wp.tid()
  out = FlexPart()
  out.rgba = rgba[p]
  out.mesh = mesh[p]
  out.material = material[p]
  out.nface = nface[p]
  out.uvadr = uvadr[p]
  part_out[p] = out


@wp.kernel
def _hide(
  # In:
  part_in: wp.array[FlexPart],
  # Out:
  part_out: wp.array[FlexPart],
):
  """Part p with a zero color (not the default one: its material's is not read; `intersect_flex`
  skips it)."""
  p = wp.tid()
  out = part_in[p]
  out.rgba = wp.vec4(0.0)
  part_out[p] = out


def hidden(parts: wp.array[FlexPart, [int]]) -> wp.array[FlexPart, [int]]:
  """A copy of `parts` with every part hidden (alpha 0: their hits are skipped)."""
  out = wp.empty_like(parts)
  wp.launch(_hide, dim=parts.shape[0], inputs=[parts], outputs=[out], device=parts.device)
  return out


@wp.kernel
def face_normals(
  # Model:
  faces: wp.array[wp.vec3i],
  # Data in:
  vert_in: wp.array2d[wp.vec3],
  # Out:
  normal_out: wp.array2d[wp.vec3],
):
  """Unit normal of thick-surface triangle f in world w."""
  worldid, f = wp.tid()
  face = faces[f]
  v0 = batched(vert_in, worldid, face[0])
  v1 = batched(vert_in, worldid, face[1])
  v2 = batched(vert_in, worldid, face[2])
  normal_out[worldid, f] = wp.normalize(wp.cross(v1 - v0, v2 - v0))


@wp.kernel
def vertex_normals(
  # Model:
  vert_faceadr: wp.array[int],
  vert_face: wp.array[int],
  # In:
  face_normal: wp.array2d[wp.vec3],
  # Out:
  normal_out: wp.array2d[wp.vec3],
):
  """Smooth normal of vertex v in world w: its adjacent face normals summed in face order,
  normalised."""
  worldid, v = wp.tid()
  n = wp.vec3(0.0)
  for k in range(vert_faceadr[v], vert_faceadr[v + 1]):
    n += face_normal[worldid, vert_face[k]]
  normal_out[worldid, v] = wp.normalize(n)


@wp.kernel
def thick_surface_points(
  # Model:
  faces: wp.array[wp.vec3i],
  borders: wp.array[wp.vec2i],
  radius: float,
  # Data in:
  vert_in: wp.array2d[wp.vec3],
  # In:
  normal_in: wp.array2d[wp.vec3],
  nface: int,
  # Out:
  point_out: wp.array[wp.vec3],
):
  """Triangle pair `workid` of world w (dim 2): a face's two offset copies, or a border's side
  wall."""
  worldid, workid = wp.tid()
  nelem = faces.shape[0]
  world_face_offset = worldid * nface
  if workid < nelem:
    face = faces[workid]
    v0 = batched(vert_in, worldid, face[0])
    v1 = batched(vert_in, worldid, face[1])
    v2 = batched(vert_in, worldid, face[2])
    n0 = batched(normal_in, worldid, face[0])
    n1 = batched(normal_in, worldid, face[1])
    n2 = batched(normal_in, worldid, face[2])
    base0 = (world_face_offset + 2 * workid) * 3
    point_out[base0 + 0] = v0 + radius * n0
    point_out[base0 + 1] = v1 + radius * n1
    point_out[base0 + 2] = v2 + radius * n2
    base1 = (world_face_offset + 2 * workid + 1) * 3
    point_out[base1 + 0] = v0 - radius * n0
    point_out[base1 + 1] = v1 - radius * n1
    point_out[base1 + 2] = v2 - radius * n2
  else:
    # Side wall of a boundary edge: two triangles spanning -radius..+radius.
    shellid = workid - nelem
    edge = borders[shellid]
    v0 = batched(vert_in, worldid, edge[0])
    v1 = batched(vert_in, worldid, edge[1])
    n0 = batched(normal_in, worldid, edge[0])
    n1 = batched(normal_in, worldid, edge[1])
    base0 = (world_face_offset + 2 * nelem + 2 * shellid) * 3
    point_out[base0 + 0] = v0 + radius * n0
    point_out[base0 + 1] = v1 - radius * n1
    point_out[base0 + 2] = v1 + radius * n1
    base1 = (world_face_offset + 2 * nelem + 2 * shellid + 1) * 3
    point_out[base1 + 0] = v1 - radius * n1
    point_out[base1 + 1] = v0 + radius * n0
    point_out[base1 + 2] = v0 - radius * n0


@wp.kernel
def surface_points(
  # Model:
  faces: wp.array[wp.vec3i],
  # Data in:
  vert_in: wp.array2d[wp.vec3],
  # Out:
  point_out: wp.array[wp.vec3],
):
  """Triangle f of world w of a closed surface (dim 3), as it is."""
  worldid, f = wp.tid()
  face = faces[f]
  base = (worldid * faces.shape[0] + f) * 3
  point_out[base + 0] = batched(vert_in, worldid, face[0])
  point_out[base + 1] = batched(vert_in, worldid, face[1])
  point_out[base + 2] = batched(vert_in, worldid, face[2])


@wp.kernel
def mesh_group_roots(
  # In:
  mesh_id: wp.uint64,
  column: int,
  # Out:
  group_root_out: wp.array2d[int],
):
  """Root node of world w's group of the mesh, into column `column`."""
  worldid = wp.tid()
  group_root_out[worldid, column] = wp.mesh_get_group_root(mesh_id, worldid)


def _vertex_faces(faces: Triangles, nvert: int) -> tuple[Ids[int], Ids[int]]:
  """(nvert + 1,) CSR offsets and the face ids adjacent to each vertex, ascending."""
  if not len(faces):
    return np.zeros(nvert + 1, np.int32), np.zeros(1, np.int32)
  vert = faces.reshape(-1)
  face = np.repeat(np.arange(len(faces)).astype(np.int32), 3)
  order = np.argsort(vert, kind="stable")  # stable: each vertex's faces stay ascending
  counts = np.bincount(vert, minlength=nvert)
  adr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
  return adr, face[order]


@final
class _Surface:
  """One part's triangle mesh over every world (see the module doc)."""

  def __init__(self, d: Deformables, part: int, *, nworld: int, radius: float) -> None:
    """Device topology of part `part` (of thickness `radius`) over `nworld` worlds; the mesh itself
    is made by `DeformableBvhs`."""
    device = d.radius.device
    self.dim = int(d.dim[part])
    self.radius = radius
    self.nworld = nworld
    self.faces = wp.array(d.faces[part], dtype=wp.vec3i, device=device)
    self.borders = wp.array(d.borders[part].reshape(-1, 2), dtype=wp.vec2i, device=device)
    self.nelem, nborder = len(d.faces[part]), len(d.borders[part])
    self.nface = surface_triangles(self.dim, self.nelem, nborder)
    self.points = wp.zeros(nworld * self.nface * 3, dtype=wp.vec3, device=device)

  def indices(self) -> Ids[int]:
    """(nworld * nface * 3,) vertex indices of the triangles: the points in order, except the
    -radius copy of each mid-surface triangle (dim 2), wound the other way (it faces outwards)."""
    index = np.arange(self.nworld * self.nface * 3).astype(np.int32).reshape(self.nworld, self.nface, 3)
    if self.dim == THICK_SURFACE:
      bottom = index[:, 1 : 2 * self.nelem : 2]
      bottom[:, :, [1, 2]] = bottom[:, :, [2, 1]]
    return index.reshape(-1)

  def write[NRow: IntVar, NVert: IntVar](self, vert: Table[wp.vec3, NRow, NVert], normal: Table[wp.vec3, NRow, NVert]) -> None:
    """Rewrites the triangles of every world from `vert` and its vertex normals `normal` (1 or
    nworld rows)."""
    device = self.points.device
    if self.dim == THICK_SURFACE:
      inputs = [self.faces, self.borders, self.radius, vert, normal, self.nface]
      wp.launch(thick_surface_points, dim=(self.nworld, self.nface // 2), inputs=inputs, outputs=[self.points], device=device)
    else:
      wp.launch(surface_points, dim=(self.nworld, self.nface), inputs=[self.faces, vert], outputs=[self.points], device=device)


@final
class DeformableBvhs[NWorld: IntVar, NDeformable: IntVar]:
  """The triangle BVHs of a scene's deformables over `nworld` worlds.

  Attributes:
    ids: (n,) mesh id per part (0 for a dim-1 part: its segments are scene-BVH capsules).
    group_root: (nworld, n) root of each world's subtree per part (-1 for dim 1).
    meshes: (part, mesh) of each surface part: its triangles of every world, world w's
      `nworld`-th share of `mesh.points` (the OptiX backend's GAS inputs, `trace/optix/dynamic.py`).
    parts: (max(n, 1),) `FlexPart` of each part (its mesh: `ids`), what the shading reads.
  """

  def __init__(
    self, d: Deformables[NDeformable, int], nworld: Int[NWorld], *, device: wp.Device, constructor: str = CONSTRUCTOR
  ) -> None:
    """Builds every surface's BVH on `device` (the scene's, where `d` lives) from `d.init_vert`
    (one row, read by every world) with Warp's `constructor` (the OptiX backend, which never
    queries them, takes the fastest build)."""
    self.device = device
    n = d.count
    nvert = len(d.init_vert)
    self.group_root: wp.array[int, [NWorld, NDeformable]] = wp.full((nworld, n), -1, dtype=int, device=device)
    radius = host_copy(d.radius)  # one copy to the host for every part
    self._surfaces = {i: _Surface(d, i, nworld=nworld, radius=float(radius[i])) for i in range(n) if d.dim[i] != 1}
    thick = [d.faces[i] for i in self._surfaces if d.dim[i] == THICK_SURFACE]
    faces = np.concatenate(thick) if thick else np.zeros((0, 3), np.int32)
    adr, adjacent = _vertex_faces(faces, nvert)
    self._faces = wp.array(faces, dtype=wp.vec3i, device=device)
    self._vert_faceadr = wp.array(adr, dtype=int, device=device)
    self._vert_face = wp.array(adjacent, dtype=int, device=device)
    self._face_normal = wp.zeros((nworld, len(faces)), dtype=wp.vec3, device=device)
    self._normal = wp.zeros((nworld, nvert), dtype=wp.vec3, device=device)
    init = wp.array(d.init_vert, dtype=wp.vec3, device=device).reshape((1, nvert))
    self.write(init, wp.zeros((1, nvert), dtype=wp.vec3, device=device))
    ids = [wp.uint64(0)] * n
    self.meshes: list[tuple[int, wp.Mesh]] = []  # kept alive: `ids` point at them
    for i, s in self._surfaces.items():
      index = wp.array(s.indices(), dtype=wp.int32, device=device)
      group = wp.array(np.repeat(np.arange(nworld).astype(np.int32), s.nface), dtype=int, device=device)
      mesh = wp.Mesh(points=s.points, indices=index, groups=group, bvh_constructor=constructor, bvh_leaf_size=LEAF_SIZE)
      self.meshes.append((i, mesh))
      ids[i] = mesh.id
      wp.launch(mesh_group_roots, dim=nworld, inputs=[mesh.id, i], outputs=[self.group_root], device=device)
    self.ids: wp.array[wp.uint64, [NDeformable]] = wp.array(ids, dtype=wp.uint64, device=device)
    self.parts = wp.zeros(max(n, 1), dtype=FlexPart, device=device)
    if n:
      inputs = [d.rgba, d.material, d.nface, d.uvadr, self.ids]
      wp.launch(part_table, dim=n, inputs=inputs, outputs=[self.parts], device=device)
    self.nodes = PackedNodes(self.ids, meshes=True)  # repacked after every refit

  def write[NRow: IntVar, NVert: IntVar](
    self, vert: Table[wp.vec3, NRow, NVert], normal: Table[wp.vec3, NRow, NVert] | None = None
  ) -> None:
    """Vertex normals (into `normal`, `vert`'s row count; None: the BVHs' own buffer) and every
    surface's triangles from `vert`, without refitting (the OptiX backend refits its GASes)."""
    if normal is None:
      normal = self._normal[: vert.shape[0]]
    rows, nface = vert.shape[0], self._faces.shape[0]
    if nface:
      wp.launch(face_normals, dim=(rows, nface), inputs=[self._faces, vert], outputs=[self._face_normal], device=self.device)
      wp.launch(
        vertex_normals,
        dim=(rows, vert.shape[1]),
        inputs=[self._vert_faceadr, self._vert_face, self._face_normal],
        outputs=[normal],
        device=self.device,
      )
    for s in self._surfaces.values():
      s.write(vert, normal)

  def refit[NRow: IntVar, NVert: IntVar](self, vert: Table[wp.vec3, NRow, NVert]) -> None:
    """Rewrites the triangles from the (1 or nworld, nvert) vertices `vert` ("flex.vert") and
    refits."""
    if not self._surfaces:
      return
    self.write(vert)
    for _part, mesh in self.meshes:
      mesh.refit()
    self.nodes.refresh()
