"""Our mesh BLAS queries (`native.mesh_closest` / `mesh_any` over packed children) against Warp's
`mesh_query_ray` / `mesh_query_ray_anyhit` on randomized meshes: closest hits, back-face culling,
group (flex) roots, a full traversal stack (built with a 2-entry stack), and the resumable walk of
alpha-tested meshes (`nodes.mesh_walk_next`) against the queries on the kept faces alone."""

from typing import Final

import numpy as np
import pytest
import warp as wp

from raywarp.core.kernels import native
from raywarp.core.trace.accel import deformable_bvh, nodes
from raywarp.core.trace.accel.packed import PackedNodes

_DEVICES: Final = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]
_FAR: Final = 100.0
_OPTIONS: Final = {"extra_build_options": nodes.BUILD_OPTIONS}
# The same queries with a 4-entry stack: deeper trees fill it.
_SMALL: Final = "#define MJPBR_MESH_STACK 4\n" + nodes.BUILD_OPTIONS.extra_cpu_preamble
_SMALL_OPTIONS: Final = {"extra_build_options": wp.ModuleBuildOptions(extra_cuda_preamble=_SMALL, extra_cpu_preamble=_SMALL)}


@wp.kernel(module="unique", enable_backward=False, module_options=_OPTIONS)
def _ours(
  # In:
  mesh: wp.uint64,
  halves: wp.uint64,
  origin: wp.array[wp.vec3],
  direction: wp.array[wp.vec3],
  root: int,
  cull: wp.bool,
  # Out:
  t_out: wp.array[float],
  face_out: wp.array[int],
  normal_out: wp.array[wp.vec3],
  any_out: wp.array[wp.bool],
):
  """Ray i against the mesh: our closest hit (t -1: none), its face and normal, our any hit."""
  i = wp.tid()
  q = native.mesh_closest(mesh, halves, origin[i], direction[i], _FAR, root, cull, wp.uint64(0))
  t_out[i] = q.t
  face_out[i] = q.face
  normal_out[i] = q.normal
  any_out[i] = native.mesh_any(mesh, halves, origin[i], direction[i], _FAR, root, wp.uint64(0))


@wp.kernel(module="unique", enable_backward=False, module_options=_SMALL_OPTIONS)
def _ours_small_stack(
  # In:
  mesh: wp.uint64,
  halves: wp.uint64,
  origin: wp.array[wp.vec3],
  direction: wp.array[wp.vec3],
  # Out:
  t_out: wp.array[float],
  any_out: wp.array[wp.bool],
):
  """`_ours` (no cull, the root) with a 4-entry stack."""
  i = wp.tid()
  t_out[i] = native.mesh_closest(mesh, halves, origin[i], direction[i], _FAR, -1, False, wp.uint64(0)).t
  any_out[i] = native.mesh_any(mesh, halves, origin[i], direction[i], _FAR, -1, wp.uint64(0))


@wp.kernel(module="unique", enable_backward=False, module_options=_OPTIONS)
def _walk(
  # In:
  mesh: wp.uint64,
  halves: wp.uint64,
  origin: wp.array[wp.vec3],
  direction: wp.array[wp.vec3],
  cull: wp.bool,
  ignored: int,
  # Out:
  t_out: wp.array[float],
  face_out: wp.array[int],
  normal_out: wp.array[wp.vec3],
  any_out: wp.array[wp.bool],
):
  """The walk's closest and any hit of ray i, ignoring the faces f with f % `ignored` == 0 (as an
  alpha test would; 0: none)."""
  i = wp.tid()
  walk = nodes.MeshWalk()
  nodes.mesh_walk_start(walk, mesh, _FAR, -1)
  t_out[i] = -1.0
  face_out[i] = -1
  while nodes.mesh_walk_next(walk, mesh, halves, origin[i], direction[i], cull, wp.uint64(0)):
    if ignored == 0 or walk.face % ignored != 0:
      t_out[i] = walk.t
      face_out[i] = walk.face
      normal_out[i] = walk.normal
      walk.limit = walk.t
  nodes.mesh_walk_start(walk, mesh, _FAR, -1)
  any_out[i] = False
  while nodes.mesh_walk_next(walk, mesh, halves, origin[i], direction[i], False, wp.uint64(0)):
    if ignored == 0 or walk.face % ignored != 0:
      any_out[i] = True
      break


@wp.kernel(module="unique", enable_backward=False)
def _warps(
  # In:
  mesh: wp.uint64,
  origin: wp.array[wp.vec3],
  direction: wp.array[wp.vec3],
  root: int,
  # Out:
  t_out: wp.array[float],
  face_out: wp.array[int],
  normal_out: wp.array[wp.vec3],
  any_out: wp.array[wp.bool],
):
  """Ray i against the mesh: Warp's closest hit (t -1: none), face, unit normal, any hit."""
  i = wp.tid()
  q = wp.mesh_query_ray(mesh, origin[i], direction[i], _FAR, root)
  t_out[i] = wp.where(q.result, q.t, -1.0)
  face_out[i] = wp.where(q.result, q.face, -1)
  normal_out[i] = wp.where(q.result, wp.normalize(q.normal), wp.vec3(0.0))
  any_out[i] = wp.mesh_query_ray_anyhit(mesh, origin[i], direction[i], _FAR, root)


def _soup(rng: np.random.Generator, ntri: int) -> tuple[np.ndarray, np.ndarray]:
  """(3 ntri, 3) vertices, (ntri, 3) indices: random triangles of mixed sizes in the unit cube,
  a few slivers and coplanar pairs (the watertight test's edge cases)."""
  centre = rng.uniform(-1, 1, (ntri, 1, 3))
  size = np.exp(rng.uniform(np.log(0.01), np.log(0.5), (ntri, 1, 1)))
  tris = centre + size * rng.normal(size=(ntri, 3, 3))
  tris[: ntri // 20, 2] = 0.5 * (tris[: ntri // 20, 0] + tris[: ntri // 20, 1]) + 1e-4  # slivers
  tris[ntri // 20 : ntri // 10, :, 2] = 0.25  # coplanar at z = 0.25
  return tris.reshape(-1, 3).astype(np.float32), np.arange(3 * ntri).astype(np.int32).reshape(-1, 3)


def _rays(rng: np.random.Generator, n: int, direction: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
  """n rays from a sphere of radius 3 towards the unit cube (all along `direction` if given)."""
  start = rng.normal(size=(n, 3))
  start = 3.0 * start / np.linalg.norm(start, axis=1, keepdims=True)
  if direction is None:
    d = rng.uniform(-0.8, 0.8, (n, 3)) - start
  else:
    d = np.broadcast_to(direction, (n, 3)).copy()
    start = rng.uniform(-1.2, 1.2, (n, 3)) - 3.0 * direction
  return start.astype(np.float32), (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)


def _mesh(points: np.ndarray, indices: np.ndarray, groups: np.ndarray | None = None) -> wp.Mesh:
  """A BLAS over `points` / `indices` as the scene builds them: cubql, or with `groups` (one
  per world) a flex's (`trace.accel.deformable_bvh`: SAH); leaf size 2."""
  return wp.Mesh(
    points=wp.array(points, dtype=wp.vec3),
    indices=wp.array(indices.ravel(), dtype=wp.int32),
    groups=None if groups is None else wp.array(groups, dtype=int),
    bvh_constructor="cubql" if groups is None else deformable_bvh.CONSTRUCTOR,
    bvh_leaf_size=2,
  )


def _query(mesh: wp.Mesh, start: np.ndarray, d: np.ndarray, *, root: int = -1, cull: bool = False):
  """((t, face, normal, any) ours, (t, face, normal, any) Warp's) of the rays."""
  ids = wp.array([mesh.id], dtype=wp.uint64)
  packed = PackedNodes(ids, meshes=True)
  n = len(start)
  origin, direction = wp.array(start, dtype=wp.vec3), wp.array(d, dtype=wp.vec3)
  results = []
  for kernel, extra in ((_ours, [wp.uint64(int(packed.host_addresses[0]))]), (_warps, [])):
    out = [wp.zeros(n, dtype=float), wp.zeros(n, dtype=int), wp.zeros(n, dtype=wp.vec3), wp.zeros(n, dtype=wp.bool)]
    inputs = [mesh.id, *extra, origin, direction, root] + ([cull] if kernel is _ours else [])
    wp.launch(kernel, dim=n, inputs=inputs, outputs=out)
    results.append(tuple(a.numpy() for a in out))
  return results


def _same_closest(t: np.ndarray, other: np.ndarray) -> np.ndarray:
  """Asserts that closest hits `t` on one tree and `other` on another tree of the same
  triangles agree: the same rays hit, at the same distance up to one float ulp (a near tie, e.g.
  coplanar triangles, can be skipped by its box's rounded entry distance in one tree only).
  Returns where they are bit-identical (nearly all)."""
  np.testing.assert_array_equal(t > 0, other > 0)
  np.testing.assert_allclose(t, other, rtol=2e-7)
  same = t == other
  assert same.mean() > 0.99
  return same & (t > 0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closest_and_any_hit_match_warps(device: str, seed: int) -> None:
  """Same t (bit for bit), face and normal as `mesh_query_ray`, same any hit as its any-hit
  query, on random triangle soups."""
  rng = np.random.default_rng(seed)
  with wp.ScopedDevice(device):
    mesh = _mesh(*_soup(rng, 3000))
    (t, face, normal, hit), (wt, wface, wnormal, whit) = _query(mesh, *_rays(rng, 4096))
  assert (t > 0).mean() > 0.3  # most rays meet the soup
  np.testing.assert_array_equal(t, wt)
  np.testing.assert_array_equal(face, wface)
  np.testing.assert_allclose(normal, wnormal, atol=1e-6)
  np.testing.assert_array_equal(hit, whit)
  np.testing.assert_array_equal(hit, t > 0)


@pytest.mark.parametrize("device", _DEVICES)
def test_culling_is_the_closest_front_face(device: str) -> None:
  """Parallel rays: culling back faces gives Warp's closest hit on the front faces alone."""
  rng = np.random.default_rng(3)
  points, indices = _soup(rng, 3000)
  direction = np.array([0.3, -0.5, -0.8]) / np.sqrt(0.98)  # unit
  tri = points[indices]
  front = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ direction < 0.0
  with wp.ScopedDevice(device):
    start, d = _rays(rng, 4096, direction)
    (t, face, _, _), _ = _query(_mesh(points, indices), start, d, cull=True)
    _, (wt, wface, _, _) = _query(_mesh(points, indices[front]), start, d)
  assert (t > 0).mean() > 0.3
  same = _same_closest(t, wt)
  np.testing.assert_array_equal(face[same], np.flatnonzero(front)[wface[same]])


@pytest.mark.parametrize("device", _DEVICES)
def test_group_roots_query_their_group_only(device: str) -> None:
  """A grouped mesh (one group per world, as a flex's): a query below a group's root gives Warp's
  result for that root, and that of the group's triangles alone."""
  rng = np.random.default_rng(4)
  points, indices = _soup(rng, 2000)
  groups = rng.integers(0, 3, len(indices)).astype(np.int32)
  with wp.ScopedDevice(device):
    mesh = _mesh(points, indices, groups)
    roots = wp.zeros(3, dtype=int)
    wp.launch(_group_roots, dim=3, inputs=[mesh.id], outputs=[roots])
    start, d = _rays(rng, 2048)
    for g, root in enumerate(roots.numpy()):
      (t, face, _, hit), (wt, wface, _, whit) = _query(mesh, start, d, root=int(root))
      np.testing.assert_array_equal(t, wt)
      np.testing.assert_array_equal(face, wface)
      np.testing.assert_array_equal(hit, whit)
      assert np.all(groups[face[t > 0]] == g)
      (alone, _, _, _), _ = _query(_mesh(points, indices[groups == g]), start, d)
      _same_closest(t, alone)


@wp.kernel(module="unique", enable_backward=False)
def _group_roots(
  # In:
  mesh: wp.uint64,
  # Out:
  root_out: wp.array[int],
):
  """Root node of group g of the mesh."""
  g = wp.tid()
  root_out[g] = wp.mesh_get_group_root(mesh, g)


def test_a_full_stack_drops_the_far_child_and_goes_on() -> None:
  """With a 4-entry stack the 3000-triangle tree (over 10 levels) fills it: the queries drop the
  far child and go on. Hits stay real (never closer than the closest) and nearly every ray that
  meets the mesh still finds a hit; ending the query at a full stack (as Warp's does) found 2.6 %
  of them on this mesh."""
  rng = np.random.default_rng(5)
  with wp.ScopedDevice("cpu"):
    mesh = _mesh(*_soup(rng, 3000))
    start, d = _rays(rng, 4096)
    (t, _, _, hit), _ = _query(mesh, start, d)
    packed = PackedNodes(wp.array([mesh.id], dtype=wp.uint64), meshes=True)
    small_t, small_any = wp.zeros(len(start), dtype=float), wp.zeros(len(start), dtype=wp.bool)
    inputs = [mesh.id, wp.uint64(int(packed.host_addresses[0])), wp.array(start, dtype=wp.vec3), wp.array(d, dtype=wp.vec3)]
    wp.launch(_ours_small_stack, dim=len(start), inputs=inputs, outputs=[small_t, small_any])
  small, small_hit = small_t.numpy(), small_any.numpy()
  assert not np.any(small_hit & ~hit) and not np.any((small > 0) & ~hit)  # never a hit that is not there
  found = small > 0
  assert np.all(small[found] >= t[found])
  assert found[hit].mean() > 0.95
  assert small_hit[hit].mean() > 0.95


@wp.func_native("return *wp::mesh_get(mesh).bvh.root;")
def _root(
  # In:
  mesh: wp.uint64,
) -> int:
  """The root node index of the mesh's BVH."""
  ...


@wp.kernel(module="unique", enable_backward=False)
def _roots(
  # In:
  mesh: wp.uint64,
  # Out:
  root_out: wp.array[int],
):
  """The root node index of the mesh's BVH."""
  root_out[0] = _root(mesh)


def _depth(mesh: wp.Mesh) -> int:
  """Levels of the mesh's tree, from its packed children: the deepest leaf's depth below the root
  (root = 0)."""
  packed = PackedNodes(wp.array([mesh.id], dtype=wp.uint64), meshes=True)
  root = wp.zeros(1, dtype=int)
  wp.launch(_roots, dim=1, inputs=[mesh.id], outputs=[root])
  halves = packed.halves.numpy().view(np.uint32).reshape(-1, 4, 4)  # (node, half, xyz + i|b)
  word = halves[:, [0, 2], 3]  # both children's lower halves: index | leaf bit
  inner: np.ndarray = np.right_shift(word, 31) == 0
  depth, level = 0, root.numpy()
  while len(level) and depth <= 64:
    level = (word[level] & 0x7FFFFFFF)[inner[level]]
    depth += 1
  return depth


@pytest.mark.parametrize("device", _DEVICES)
def test_scene_builders_trees_fit_the_stack(device: str) -> None:
  """The mesh queries' 32-entry stack holds one far sibling per level: the BLAS builders' trees
  (cubql, as the scene's; SAH, as a flex's) stay within 32 levels, also on a degenerate mesh
  (triangles at geometrically shrinking distances, a classic deep-tree input)."""
  rng = np.random.default_rng(6)
  scale = 2.0 ** -np.arange(100).astype(np.float64)
  chain = (scale[:, None, None] * (1.0 + 0.1 * rng.normal(size=(100, 3, 3)))).astype(np.float32)
  for points, indices in (_soup(rng, 20000), (chain.reshape(-1, 3), np.arange(300).astype(np.int32).reshape(-1, 3))):
    for groups in (None, np.zeros(len(indices), np.int32)):
      with wp.ScopedDevice(device):
        depth = _depth(_mesh(points, indices, groups))
      assert 1 < depth <= 32, (depth, groups is None)


def _walked(mesh: wp.Mesh, start: np.ndarray, d: np.ndarray, *, cull: bool, ignored: int):
  """(t, face, normal, any) of the walk over `mesh`, ignoring faces f % `ignored` == 0."""
  ids = wp.array([mesh.id], dtype=wp.uint64)
  packed = PackedNodes(ids, meshes=True)
  n = len(start)
  out = [wp.zeros(n, dtype=float), wp.zeros(n, dtype=int), wp.zeros(n, dtype=wp.vec3), wp.zeros(n, dtype=wp.bool)]
  inputs = [mesh.id, wp.uint64(int(packed.host_addresses[0])), wp.array(start, dtype=wp.vec3), wp.array(d, dtype=wp.vec3)]
  wp.launch(_walk, dim=n, inputs=[*inputs, cull, ignored], outputs=out)
  return tuple(a.numpy() for a in out)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cull", [False, True])
def test_walk_without_ignored_faces_is_the_closest_query(device: str, cull: bool) -> None:
  """Accepting every candidate, the walk gives `mesh_closest`'s hit bit for bit."""
  rng = np.random.default_rng(4)
  with wp.ScopedDevice(device):
    mesh = _mesh(*_soup(rng, 3000))
    start, d = _rays(rng, 4096)
    (t, face, normal, hit), _ = _query(mesh, start, d, cull=cull)
    wt, wface, wnormal, whit = _walked(mesh, start, d, cull=cull, ignored=0)
  assert (t > 0).mean() > 0.3
  np.testing.assert_array_equal(wt, t)
  np.testing.assert_array_equal(wface, face)
  np.testing.assert_allclose(wnormal[t > 0], normal[t > 0], atol=1e-6)
  np.testing.assert_array_equal(whit, hit)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cull", [False, True])
def test_walk_ignoring_faces_is_the_query_on_the_kept_faces(device: str, cull: bool) -> None:
  """Ignoring every third face (an alpha test's rejections), the walk's closest and any hit are
  those of the mesh without those faces; the ignored layers in front cost no restart."""
  rng = np.random.default_rng(5)
  points, indices = _soup(rng, 3000)
  kept = np.arange(len(indices)) % 3 != 0
  with wp.ScopedDevice(device):
    start, d = _rays(rng, 4096)
    wt, wface, _, whit = _walked(_mesh(points, indices), start, d, cull=cull, ignored=3)
    (t, face, _, hit), _ = _query(_mesh(points, indices[kept]), start, d, cull=cull)
  assert (t > 0).mean() > 0.3
  same = _same_closest(wt, t)
  # Equal distances: the same face, except exact ties (coplanar faces) visited in another order.
  assert (wface[same] == np.flatnonzero(kept)[face[same]]).mean() > 0.999
  np.testing.assert_array_equal(whit, hit)
