"""Newton's triangle-mesh particles (cloth, soft-body surfaces) as core deformables.

A Newton model's triangles (`model.tri_indices`, particle ids) are the surfaces its viewer draws
over `state.particle_q`. Each connected set of triangles is one core `Part`
(`raywarp.core.pipeline.gpu_scene.deformables`): an open sheet (cloth: some edge has one triangle)
is a thick surface of half thickness the particles' mean collision radius (`model.particle_radius`:
the surface the solver keeps other bodies away from), a closed one (a soft body's skin) a closed
surface. Their vertex columns are one Newton world's particles: every render gathers, for render
world v, the particles of the Newton world it shows (the rigid instances' rule:
`StateView.world_indices[v]`, else v; world 0 beyond the model's worlds), plus that world's offset
and the view's root, into the binding's "flex.vert" table (`ParticleMeshes.gather`), so each step's
positions are what the next render traces. With `merge_worlds` one render world holds every world's
triangles over all the particles (no world offsets then). Every world must repeat world 0's
triangles.
"""

from __future__ import annotations

import dataclasses
from typing import Final, final

import newton
import numpy as np
import warp as wp

from raywarp.core.pipeline.gpu_scene.deformables import (
  CLOSED_SURFACE,
  THICK_SURFACE,
  Deformables,
  Pairs,
  Part,
  Points,
  Triangles,
)
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.post.grading.color import srgb_to_linear
from raywarp.scene.model.report import LoadReport

# A triangle without `model.tri_color`: Newton's shape default (display sRGB 0.7).
_SRGB: Final = (0.7, 0.7, 0.7)


@wp.kernel(enable_backward=False)
def _gather(  # noqa: PLR0913  # the particle state, the world mapping and the layout
  # In:
  particle_q: wp.array1d[wp.vec3],
  world_indices: wp.array1d[wp.int32],
  world_offsets: wp.array1d[wp.vec3],
  start: int,
  stride: int,
  nsource: int,
  mapped: bool,
  offset: bool,
  root: wp.transformf,
  # Out:
  vert: wp.array2d[wp.vec3],
):
  """vert[v, i] = vertex i of the Newton world render world v shows (its offset and the root
  applied)."""
  v, i = wp.tid()
  w = v
  if mapped:
    w = world_indices[v]
  if w < 0 or w >= nsource:
    w = 0
  p = particle_q[start + w * stride + i]
  if offset and w < world_offsets.shape[0]:
    p = p + world_offsets[w]
  vert[v, i] = wp.transform_point(root, p)


@final
@dataclasses.dataclass(frozen=True)
class ParticleRows:
  """Where each Newton world's deformable vertices are in `state.particle_q`.

  Attributes:
    start: the first vertex's particle of world 0.
    count: vertices per world.
    stride: particles between the first vertices of consecutive worlds.
    nsource: Newton worlds (1 with merge_worlds: every world's particles in one row).
    merged: one row holds several Newton worlds (merge_worlds on a multi-world model).
  """

  start: int
  count: int
  stride: int
  nsource: int
  merged: bool = False


@final
@dataclasses.dataclass(frozen=True)
class ParticleMeshes:
  """The deformables of a Newton model and where their vertices are read.

  Attributes:
    parts: one per connected triangle set (vertex ids local to the row).
    init_vert: (count, 3) the row-0 vertices at the model's initial particle positions.
    rows: the vertex columns of each Newton world.
  """

  parts: tuple[Part, ...]
  init_vert: Points
  rows: ParticleRows

  def deformables(self, device: wp.DeviceLike) -> Deformables:
    """The core deformables on `device`."""
    return Deformables.create(self.parts, self.init_vert, device=wp.get_device(device))

  def gather(  # noqa: PLR0913  # the state and the view's world mapping, keyword-only
    self,
    particle_q: wp.array[wp.vec3, [int]],
    vert: wp.array[wp.vec3, [int, int]],
    *,
    world_indices: wp.array[wp.int32, [int]] | None,
    world_offsets: wp.array[wp.vec3, [int]] | None,
    root: wp.transformf,
    empty_ids: wp.array[wp.int32, [int]],
    empty_offsets: wp.array[wp.vec3, [int]],
  ) -> None:
    """Writes the (nworld, count) vertices of every render world into `vert` (`_gather`).

    Raises:
      ValueError: for a particle array shorter than the model's, too few world indices, or world
        offsets with merge_worlds (one row holds every world).
    """
    r = self.rows
    if particle_q.shape[0] < r.start + (r.nsource - 1) * r.stride + r.count:
      raise ValueError(f"state.particle_q has {particle_q.shape[0]} particles, the model's particle meshes read more")
    if world_indices is not None and world_indices.shape[0] < vert.shape[0]:
      raise ValueError(f"StateView.world_indices has {world_indices.shape[0]} entries for {vert.shape[0]} render worlds")
    if world_offsets is not None and r.merged:
      raise ValueError("StateView.world_offsets cannot move a merged model's particle meshes (one row of every world)")
    wp.launch(
      _gather,
      dim=vert.shape,
      inputs=[
        particle_q,
        world_indices if world_indices is not None else empty_ids,
        world_offsets if world_offsets is not None else empty_offsets,
        r.start,
        r.stride,
        r.nsource,
        world_indices is not None,
        world_offsets is not None,
        root,
      ],
      outputs=[vert],
      device=vert.device,
    )


def _components(tris: Triangles) -> list[np.ndarray[[int], np.dtype[np.intp]]]:
  """The rows of `tris` of each vertex-connected set (union-find over shared vertices)."""
  parent = np.arange(int(tris.max()) + 1 if len(tris) else 0)

  def root(i: int) -> int:
    """The set representative of vertex `i` (path halving)."""
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = int(parent[i])
    return i

  for a, b, c in tris.tolist():
    ra = root(a)
    for v in (b, c):
      rv = root(v)
      if rv != ra:
        parent[rv] = ra
  labels = np.array([root(int(t[0])) for t in tris], np.int64)
  return [np.flatnonzero(labels == k) for k in dict.fromkeys(labels.tolist())]


def _borders(tris: Triangles) -> Pairs:
  """(k, 2) edges used by one triangle only (an open sheet's boundary), in triangle order."""
  edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
  keys = np.sort(edges, axis=1)
  _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
  return edges[counts[inverse.reshape(-1)] == 1].astype(np.int32)


type Rgba = tuple[float, float, float, float]


def _rgba(model: newton.Model, ids: np.ndarray[[int], np.dtype[np.intp]], report: LoadReport) -> Rgba:
  """Linear color and opacity of the triangles `ids` (`model.tri_color`: display sRGB, and
  `tri_opacity`); a part has one: differing triangles give their mean (reported)."""
  srgb = host_copy(model.tri_color)[ids] if model.tri_color is not None else np.tile(_SRGB, (len(ids), 1))
  alpha = host_copy(model.tri_opacity)[ids] if model.tri_opacity is not None else np.ones(len(ids))
  if np.ptp(srgb, axis=0).max() > 0.0 or np.ptp(alpha) > 0.0:
    report.approx("newton.tri_color(per_triangle: part mean)", n=len(ids))
  r, g, b = (float(c) for c in srgb_to_linear(np.mean(srgb, axis=0).astype(np.float64)))
  return (r, g, b, float(np.clip(np.mean(alpha), 0.0, 1.0)))


def _part(tris: Triangles, radius: np.ndarray[[int], np.dtype[np.float32]], rgba: Rgba, report: LoadReport) -> Part:
  """The core part of one connected triangle set (row-local vertex ids); a sheet without particle
  radii is a zero-thickness surface (reported)."""
  low, high = int(tris.min()), int(tris.max())
  faces = (tris - low).astype(np.int32)
  borders = _borders(faces)
  if not len(borders):
    return Part(dim=CLOSED_SURFACE, vertadr=low, vertnum=high - low + 1, radius=0.0, rgba=rgba, faces=faces)
  used = np.unique(tris)
  thickness = float(np.mean(radius[used])) if len(radius) else 0.0
  if thickness <= 0.0:
    report.approx("newton.cloth.thickness(no particle radius: zero)", n=len(tris))
  return Part(dim=THICK_SURFACE, vertadr=low, vertnum=high - low + 1, radius=thickness, rgba=rgba, faces=faces, borders=borders)


def _rows(model: newton.Model, used: np.ndarray[[int], np.dtype[np.int64]], *, merge_worlds: bool) -> ParticleRows:
  """The vertex columns of the triangles' particles `used`.

  Raises:
    ValueError: for triangles on global particles in a model of several worlds, or worlds whose
      particles are not equal, contiguous, evenly spaced blocks (render with merge_worlds=True).
  """
  nparticle = model.particle_count
  nworld = max(model.world_count, 1)
  if merge_worlds or nworld == 1:
    return ParticleRows(start=0, count=nparticle, stride=nparticle, nsource=1, merged=nworld > 1)
  world = host_copy(model.particle_world) if model.particle_world is not None else np.zeros(nparticle, np.int32)
  if np.any(world[used] < 0):
    raise ValueError("raywarp.integrations.newton: triangles on global particles render only with merge_worlds=True")
  blocks = [np.flatnonzero(world == w) for w in range(nworld)]
  count = len(blocks[0])
  contiguous = all(len(b) == count and (count == 0 or b[-1] - b[0] == count - 1) for b in blocks)
  starts = np.array([int(b[0]) if len(b) else 0 for b in blocks])
  stride = int(starts[1] - starts[0])
  if not contiguous or not np.all(np.diff(starts) == stride) or stride < count:
    raise ValueError(
      "raywarp.integrations.newton: every world needs the same number of particles in one contiguous block at evenly"
      " spaced ids to render its triangles per world (render with merge_worlds=True)"
    )
  return ParticleRows(start=int(starts[0]), count=count, stride=stride, nsource=nworld)


def _check_repeats(tris: np.ndarray[[int, 3], np.dtype[np.int64]], rows: ParticleRows) -> None:
  """Raises ValueError unless every world's triangles are world 0's (shifted by its block)."""
  local = tris - rows.start
  world = local[:, 0] // rows.stride
  first = np.sort(local[world == 0], axis=0)
  for w in range(1, rows.nsource):
    if not np.array_equal(np.sort(local[world == w] - w * rows.stride, axis=0), first):
      raise ValueError(
        f"raywarp.integrations.newton: world {w}'s cloth / soft-body triangles differ from world 0's (merge_worlds=True)"
      )


def report_particles(model: newton.Model, report: LoadReport) -> None:
  """Counts in `report` the particles no triangle uses (`newton.particles(unrendered)`: loose
  particles have no surface to render)."""
  used = 0
  if model.tri_count and model.tri_indices is not None:
    used = len(np.unique(host_copy(model.tri_indices)))
  if model.particle_count > used:
    report.drop("newton.particles(unrendered)", n=model.particle_count - used)


def particle_meshes(model: newton.Model, report: LoadReport, *, merge_worlds: bool) -> ParticleMeshes | None:
  """The deformables of `model`'s triangles (None without any).

  Raises:
    ValueError: see `_rows` and `_check_repeats`.
  """
  if not model.tri_count or model.tri_indices is None or model.particle_q is None:
    return None
  tris = np.asarray(host_copy(model.tri_indices), np.int64).reshape(-1, 3)
  rows = _rows(model, np.unique(tris), merge_worlds=merge_worlds)
  _check_repeats(tris, rows)
  local = tris - rows.start
  in_row = np.all((local >= 0) & (local < rows.count), axis=1)  # row 0's triangles (the others repeat them)
  radius = (
    host_copy(model.particle_radius)[rows.start : rows.start + rows.count] if model.particle_radius is not None else np.zeros(0)
  )
  row0: np.ndarray[[int], np.dtype[np.intp]] = np.flatnonzero(in_row)
  radius = np.asarray(radius, np.float32)
  parts = tuple(_part(local[row0[c]], radius, _rgba(model, row0[c], report), report) for c in _components(local[row0]))
  q = host_copy(model.particle_q).reshape(-1, 3)
  init = np.ascontiguousarray(q[rows.start : rows.start + rows.count], np.float32)
  return ParticleMeshes(parts=parts, init_vert=init, rows=rows)
