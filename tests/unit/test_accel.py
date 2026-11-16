"""The software traversal's pieces (`core/trace/accel`): B23, P12, P10, and every factory variant
compiles."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels import native
from raywarp.core.kernels.abi import rays as trace
from raywarp.core.kernels.abi.features import PLANES_NONE, PLANES_SOME
from raywarp.core.kernels.abi.rays import SoftwareGeometry
from raywarp.core.trace.accel import bounds, cast, nodes, ordered, primitives, query
from raywarp.core.trace.accel.packed import PackedNodes

wp.set_module_options({"enable_backward": False})


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _expand_root(
  # In:
  tree_id: wp.uint64,
  halves: wp.uint64,
  start: wp.array[int],
  # Out:
  out: wp.array[int],
):
  """Expands the root of a two-leaf tree (both entered) with `start[0]` stack entries pending."""
  c = ordered.Cursor()
  s = ordered.NearStack()
  origin = wp.vec3(-10.0, 0.5, 0.5)
  rcp_dir = nodes.reciprocal(wp.vec3(1.0, 0.0, 0.0))
  ordered.start_ordered(c, tree_id, halves, 0, origin, rcp_dir, 1.0e9)
  root = int(c.next_left)  # pyrefly: ignore[unnecessary-type-conversion]  # a Warp copy: a field read aliases the field
  c.count = start[0]
  c.has_next = 0
  ordered.expand(c, s, root, origin, rcp_dir, 1.0e9)
  # The packed slots of the root's children: the near one is the box at x in [0, 1].
  left_near = native.packed_child(halves, 4 * root).lower[0] < 1.0
  near = wp.where(left_near, 4 * root, 4 * root + 2)
  out[0] = c.count
  out[1] = c.has_next
  out[2] = nodes.int_at(s.node, wp.max(c.count - 1, 0))
  out[3] = wp.where(left_near, 4 * root + 2, 4 * root)  # the far child's slot
  # The near child's reference (its slot range: a leaf) arrives with its box.
  out[4] = int(c.next_left == native.packed_child(halves, near).left and c.next_leaf == 1)


def _expand(start: int) -> list[int]:
  """(count, near child held, stacked slot, far child's slot, near child's range kept)."""
  lo = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float32)
  bvh = wp.Bvh(wp.array(lo, dtype=wp.vec3, device="cpu"), wp.array(lo + 1.0, dtype=wp.vec3, device="cpu"))
  halves = PackedNodes(wp.array([bvh.id], dtype=wp.uint64, device="cpu"), meshes=False)
  out = wp.zeros(5, dtype=int, device="cpu")
  inputs = [bvh.id, wp.uint64(int(halves.host_addresses[0])), wp.array([start], dtype=int, device="cpu")]
  wp.launch(_expand_root, dim=1, inputs=inputs, outputs=[out], device="cpu")
  return out.numpy().tolist()


def test_ordered_push_keeps_near_child_when_one_slot_is_free():
  """B6: one free slot keeps both children, with the near child outside the stack."""
  count, held, stacked, far, kept = _expand(nodes.STACK - 1)
  assert (count, held, stacked, kept) == (nodes.STACK, 1, far, 1)


def test_ordered_push_far_first_when_room():
  """Only the far child needs a stack entry; the near child is expanded immediately."""
  count, held, stacked, far, kept = _expand(0)
  assert (count, held, stacked, kept) == (1, 1, far, 1)


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _walk_full_stack(
  # In:
  tree_id: wp.uint64,
  halves: wp.uint64,
  # Out:
  out: wp.array[int],
):
  """Primitives an unordered walk yields from a stack holding the root's two children `STACK - 1`
  times between them (one slot free)."""
  walk = cast.Walk()
  s = cast.WalkStack()
  origin = wp.vec3(-10.0, 0.5, 0.5)
  rcp_dir = nodes.reciprocal(wp.vec3(1.0, 0.0, 0.0))
  cast.start_unordered(walk, s, tree_id, halves, 0, origin, rcp_dir, 1.0e9)
  walk.next = -1
  for k in range(nodes.STACK - 1):
    nodes.unordered_store(s.node, k, wp.where(k % 2 == 0, 0, 2))  # the root's children, alternately
  walk.count = nodes.STACK - 1
  n = int(0)
  while cast.next_unordered(walk, s, origin, rcp_dir, 1.0e9) >= 0:
    n += 1
  out[0] = n


def test_unordered_walk_keeps_both_children_when_one_slot_is_free():
  """B59: popping an inner node with one free slot dropped both children (a missed subtree)."""
  n = 4
  lo = np.array([[2.0 * i, 0.0, 0.0] for i in range(n)], np.float32)
  bvh = wp.Bvh(wp.array(lo, dtype=wp.vec3, device="cpu"), wp.array(lo + 1.0, dtype=wp.vec3, device="cpu"))
  halves = PackedNodes(wp.array([bvh.id], dtype=wp.uint64, device="cpu"), meshes=False)
  out = wp.zeros(1, dtype=int, device="cpu")
  wp.launch(_walk_full_stack, dim=1, inputs=[bvh.id, wp.uint64(int(halves.host_addresses[0]))], outputs=[out], device="cpu")
  # The root's two subtrees split the n primitives the ray crosses: every one of the STACK - 1
  # entries hands out its whole subtree.
  assert out.numpy()[0] == (nodes.STACK // 2) * (n // 2) + (nodes.STACK // 2 - 1) * (n // 2)


@wp.kernel
def _aabbs(
  types: wp.array[int],
  pos: wp.array[wp.vec3],
  rot: wp.array[wp.mat33],
  size: wp.array[wp.vec3],
  lo: wp.array[wp.vec3],
  hi: wp.array[wp.vec3],
):
  """`world_aabb` of shape i."""
  i = wp.tid()
  lower, upper = bounds.world_aabb(types[i], pos[i], rot[i], size[i])
  lo[i] = lower
  hi[i] = upper


def _rotations(n: int, rng: np.random.Generator) -> np.ndarray:
  """n random rotation matrices (QR of Gaussian matrices, det +1)."""
  q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
  q *= np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
  q[np.linalg.det(q) < 0, :, 0] *= -1
  return q


def _local_points(gtype: int, size: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  """Points on the surface of a shape in its frame (the extremes of its box among them)."""
  if gtype in (ShapeType.BOX, ShapeType.MESH):
    return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
  if gtype == ShapeType.HFIELD:
    return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 2)]) * size
  if gtype == ShapeType.PLANE:
    s = 2.0 * max(size[0], size[1])
    return np.array([[sx * s, sy * s, 0.0] for sx in (-1, 1) for sy in (-1, 1)])
  u = rng.normal(size=(20000, 3))
  u /= np.linalg.norm(u, axis=1, keepdims=True)
  if gtype == ShapeType.SPHERE:
    return u * size[0]
  if gtype == ShapeType.ELLIPSOID:
    return u * size
  if gtype == ShapeType.CAPSULE:
    return u * size[0] + np.where(u[:, 2:] >= 0, 1.0, -1.0) * np.array([0.0, 0.0, size[1]])
  # Cylinder: cap rims.
  a = rng.uniform(0, 2 * np.pi, 20000)
  z = np.where(rng.random(20000) < 0.5, -size[1], size[1])
  return np.stack([size[0] * np.cos(a), size[0] * np.sin(a), z], axis=1)


@pytest.mark.parametrize("gtype", [t for t in ShapeType if t not in (ShapeType.SDF, ShapeType.FLEX)])
def test_world_aabb_closed_form(gtype: ShapeType):
  """P12: the closed-form box contains the shape and is tight (plane: plus its pad)."""
  rng = np.random.default_rng(int(gtype))
  n = 16
  rot = _rotations(n, rng)
  pos = rng.normal(size=(n, 3))
  size = rng.uniform(0.1, 2.0, size=(n, 3))
  args = [np.full(n, int(gtype), np.int32), pos, rot, size]
  lo = wp.zeros(n, dtype=wp.vec3, device="cpu")
  hi = wp.zeros(n, dtype=wp.vec3, device="cpu")
  dtypes = (int, wp.vec3, wp.mat33, wp.vec3)
  wp.launch(
    _aabbs,
    dim=n,
    inputs=[wp.array(a, dtype=d, device="cpu") for a, d in zip(args, dtypes, strict=False)],
    outputs=[lo, hi],
    device="cpu",
  )
  pad = bounds.PLANE_PAD if gtype == ShapeType.PLANE else 0.0
  for i in range(n):
    world = _local_points(int(gtype), size[i], rng) @ rot[i].T + pos[i]
    assert np.all(world >= lo.numpy()[i] - 1e-5) and np.all(world <= hi.numpy()[i] + 1e-5)
    # Tight: the sampled extremes reach the box (sampled curved shapes up to 1%).
    tol = 1e-5 if gtype in (ShapeType.BOX, ShapeType.MESH, ShapeType.HFIELD, ShapeType.PLANE) else 0.01 * float(size[i].max())
    np.testing.assert_allclose(world.min(axis=0) - pad, lo.numpy()[i], atol=tol)
    np.testing.assert_allclose(world.max(axis=0) + pad, hi.numpy()[i], atol=tol)


def test_instance_bounds_one_pose_row_equals_tiled():
  """P10: the BVH build reads one row of initial poses for every world (no host tile to nworld)."""
  rng = np.random.default_rng(0)
  nworld, n = 3, 5
  rot = _rotations(n, rng)
  pos = rng.normal(size=(n, 3))
  types = wp.array(
    np.array([ShapeType.BOX, ShapeType.SPHERE, ShapeType.CAPSULE, ShapeType.CYLINDER, ShapeType.ELLIPSOID]),
    dtype=int,
    device="cpu",
  )
  common = [
    types,
    wp.full((1, n), -1, dtype=int, device="cpu"),
    wp.array(rng.uniform(0.1, 1.0, size=(1, n, 3)), dtype=wp.vec3, device="cpu"),
  ]
  rest = [
    n,
    wp.array(np.arange(n), dtype=int, device="cpu"),
    wp.zeros(1, dtype=wp.vec3, device="cpu"),
    wp.zeros(1, dtype=wp.vec3, device="cpu"),
    wp.zeros(1, dtype=wp.vec3, device="cpu"),
    wp.array(np.arange(n), dtype=int, device="cpu"),
  ]
  out = []
  for rows in (1, nworld):
    lo, hi, group = (wp.zeros(nworld * n, dtype=d, device="cpu") for d in (wp.vec3, wp.vec3, int))
    poses = [
      wp.array(np.tile(pos[None], (rows, 1, 1)), dtype=wp.vec3, device="cpu"),
      wp.array(np.tile(rot[None], (rows, 1, 1, 1)), dtype=wp.mat33, device="cpu"),
    ]
    wp.launch(bounds.instance_bounds, dim=(nworld, n), inputs=[*common, *poses, *rest], outputs=[lo, hi, group], device="cpu")
    out.append((lo.numpy(), hi.numpy(), group.numpy()))
  for a, b in zip(*out, strict=False):
    np.testing.assert_array_equal(a, b)
  np.testing.assert_array_equal(out[0][2], np.repeat(np.arange(nworld), n))


@pytest.mark.parametrize("cancel", [False, True])
def test_mesh_instance_box_is_its_own_box_off_origin(cancel: bool) -> None:
  """A mesh whose vertices lie far from its origin (authored in world space) is bounded by its own
  box under the instance pose, not by a box centred on the origin: it contains every vertex and
  is tight (up to the rounding pad). `cancel`: the pose brings the mesh back near the world origin
  (pos = -R c: the box's world coordinates are small, its terms large)."""
  rng = np.random.default_rng(7)
  offset = np.array([3000.0, -2000.0, 1500.0]) if cancel else np.array([40.0, -25.0, 3.0])
  points = (rng.uniform(-0.5, 0.5, size=(2000, 3)) + offset).astype(np.float32)
  lo, hi = points.min(axis=0).astype(np.float64), points.max(axis=0).astype(np.float64)
  n = 8
  rot = _rotations(n, rng).astype(np.float32)
  centre = (0.5 * (lo + hi)).astype(np.float32)
  pos = (-(rot @ centre) if cancel else rng.normal(size=(n, 3))).astype(np.float32)
  dev = "cpu"
  out_lo, out_hi, group = (wp.zeros(n, dtype=d, device=dev) for d in (wp.vec3, wp.vec3, int))
  wp.launch(
    bounds.instance_bounds,
    dim=(1, n),
    inputs=[
      wp.full(n, int(ShapeType.MESH), dtype=int, device=dev),
      wp.zeros((1, n), dtype=int, device=dev),
      wp.zeros((1, n), dtype=wp.vec3, device=dev),
      wp.array(pos[None], dtype=wp.vec3, device=dev),
      wp.array(rot[None], dtype=wp.mat33, device=dev),
      n,
      wp.array(np.arange(n), dtype=int, device=dev),
      wp.array([centre], dtype=wp.vec3, device=dev),
      wp.array([(0.5 * (hi - lo)).astype(np.float32)], dtype=wp.vec3, device=dev),
      wp.zeros(1, dtype=wp.vec3, device=dev),
      wp.array(np.arange(n), dtype=int, device=dev),
    ],
    outputs=[out_lo, out_hi, group],
    device=dev,
  )
  corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
  for i in range(n):
    # The vertices in world space as the ray queries see them (float32 pose applied in float64).
    world = points.astype(np.float64) @ rot[i].T.astype(np.float64) + pos[i]
    assert np.all(world >= out_lo.numpy()[i]) and np.all(world <= out_hi.numpy()[i])
    box = corners @ rot[i].T.astype(np.float64) + pos[i]
    np.testing.assert_allclose(out_lo.numpy()[i], box.min(axis=0), atol=1e-2 if cancel else 1e-3)
    np.testing.assert_allclose(out_hi.numpy()[i], box.max(axis=0), atol=1e-2 if cancel else 1e-3)


_ALL_TYPES = tuple(int(t) for t in ShapeType)
# (world_tree, shared, compact_poses, visibility, planes): `Features.trees` (the plane ids are
# run-time data; the last entry whether their code is compiled in).
_TREES = [
  (True, False, False, False, PLANES_NONE),
  (False, True, True, False, PLANES_SOME),
  (True, True, True, True, PLANES_SOME),
]


@pytest.mark.parametrize("ordered_traversal", [False, True])
@pytest.mark.parametrize("trees", _TREES, ids=["world", "shared_compact_planes", "both_visibility_planes"])
def test_trace_variants_compile(ordered_traversal: bool, trees: tuple):
  """Every traversal variant (all shape types, culling, two-sided) generates and compiles (S1: no
  NameError)."""
  closest, any_hit = query.make_trace(
    types=_ALL_TYPES, cull_backfaces=True, ordered=ordered_traversal, trees=trees, two_sided=True
  )

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees, ordered=ordered_traversal))
  def probe(geo: SoftwareGeometry, out: wp.array[float]):
    hit = closest(geo, 0, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), 1.0)
    out[0] = hit.dist + wp.where(any_hit(geo, 0, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), 1.0), 1.0, 0.0)

  probe.module.load(wp.get_device("cpu"))


@wp.kernel
def _primitive_hits(
  # In:
  origins: wp.array[wp.vec3],
  directions: wp.array[wp.vec3],
  # Out:
  dist_out: wp.array2d[float],
):
  """Distance along ray i to each analytic primitive (plane, sphere, capsule, ellipsoid, cylinder,
  box) at the origin, of unit size."""
  i = wp.tid()
  o = origins[i]
  v = directions[i]
  pos = wp.vec3(0.0)
  mat = wp.identity(3, float)
  size = wp.vec3(1.0, 1.0, 1.0)
  d0, _n0 = primitives.ray_plane(pos, mat, size, o, v)
  d1, _n1 = primitives.ray_sphere(pos, 1.0, o, v)
  d2, _n2 = primitives.ray_capsule(pos, mat, size, o, v)
  d3, _n3 = primitives.ray_ellipsoid(pos, mat, size, o, v)
  d4, _n4 = primitives.ray_cylinder(pos, mat, size, o, v)
  d5, _n5 = primitives.ray_box(pos, mat, size, o, v)
  dist_out[i, 0] = d0
  dist_out[i, 1] = d1
  dist_out[i, 2] = d2
  dist_out[i, 3] = d3
  dist_out[i, 4] = d4
  dist_out[i, 5] = d5


def test_primitives_reject_hits_at_distance_zero():
  """Regression (sweep S1): a ray starting on a primitive's surface does not hit it at t = 0 (the
  plane returned -0.0, a "hit" that depth reports as a miss): it hits the far side or nothing."""
  down, up = (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)
  # On the plane (down, up); on the top of the sphere / ellipsoid / cylinder / box (up: leaving,
  # down: through to the bottom at z = -1); on the capsule's top (z = 2), down to its bottom.
  rays = [((0.2, 0.1, 0.0), down), ((0.2, 0.1, 0.0), up), ((0.0, 0.0, 1.0), up), ((0.0, 0.0, 1.0), down)]
  rays += [((0.0, 0.0, 2.0), down), ((0.0, 0.0, 2.0), up)]
  o = wp.array([r[0] for r in rays], dtype=wp.vec3, device="cpu")
  v = wp.array([r[1] for r in rays], dtype=wp.vec3, device="cpu")
  dist = wp.zeros((len(rays), 6), dtype=float, device="cpu")
  wp.launch(_primitive_hits, dim=len(rays), inputs=[o, v], outputs=[dist], device="cpu")
  d = dist.numpy()
  assert np.all((d > 0.0) | (d == -1.0)), d
  np.testing.assert_array_equal(d[:2, 0], -1.0)
  np.testing.assert_array_equal(d[2, [1, 3, 4, 5]], -1.0)
  np.testing.assert_allclose(d[3, [1, 3, 4, 5]], 2.0, rtol=1e-6)
  np.testing.assert_allclose(d[4, 2], 4.0, rtol=1e-6)
  assert d[5, 2] == -1.0


@wp.kernel(module="unique", enable_backward=False)
def _slab(
  # In:
  origins: wp.array[wp.vec3],
  directions: wp.array[wp.vec3],
  # Out:
  entry_out: wp.array[float],
):
  """`nodes.ray_aabb` of ray i into the unit box [0, 1]^3."""
  i = wp.tid()
  rcp = nodes.reciprocal(directions[i])
  entry_out[i] = nodes.ray_aabb(origins[i], rcp, wp.vec3(0.0), wp.vec3(1.0), 1.0e30)


def test_slab_test_keeps_rays_lying_in_a_box_face():
  """An axis-aligned ray lying in a face plane of the box (origin on the face, no motion along its
  axis: 0 x inf = NaN in that slab) enters the closed box; one beside it misses (both signs of a
  zero component)."""
  down, up = (0.0, 0.0, -1.0), (0.0, -0.0, 1.0)
  rays = [((0.0, 0.5, 5.0), down), ((1.0, 0.5, 5.0), down), ((0.0, 0.0, 5.0), down), ((1.0, 1.0, 5.0), down)]
  rays += [((0.5, 0.0, -5.0), up), ((0.5, 1.0, -5.0), up), ((-0.1, 0.5, 5.0), down), ((0.5, 1.1, -5.0), up)]
  o = wp.array([r[0] for r in rays], dtype=wp.vec3, device="cpu")
  v = wp.array([r[1] for r in rays], dtype=wp.vec3, device="cpu")
  out = wp.zeros(len(rays), dtype=float, device="cpu")
  wp.launch(_slab, dim=len(rays), inputs=[o, v], outputs=[out], device="cpu")
  np.testing.assert_array_equal(out.numpy(), [4.0, 4.0, 4.0, 4.0, 5.0, 5.0, -1.0, -1.0])


def test_ray_floors_leave_planes_out_and_scale_below_ten_centimetres() -> None:
  """`trace.ray_floors`: an unbounded plane (an upload's 1e6 m radius) neither lifts its own rays
  nor sets the scene's extent; a 10 cm object keeps the metre floor (1e-4 m), a millimetre one
  scales it (1e-6 m); a 10 km mesh's own floor is 2^-20 of its radius."""
  plane, box, mesh = int(ShapeType.PLANE), int(ShapeType.BOX), int(ShapeType.MESH)
  at = np.zeros((2, 3), np.float32)
  floors = trace.ray_floors(np.array([1.0e6, 0.05], np.float32), at, np.array([plane, box], np.int32))
  np.testing.assert_allclose(floors, [1.0e-4, 1.0e-4, 1.0e-4], rtol=1e-6)
  floors = trace.ray_floors(np.array([1.0e6, 5.0e-4], np.float32), at, np.array([plane, box], np.int32))
  np.testing.assert_allclose(floors, [1.0e-6, 1.0e-6, 1.0e-6], rtol=1e-5)
  floors = trace.ray_floors(np.array([1.0e4, 0.05], np.float32), at, np.array([mesh, box], np.int32))
  np.testing.assert_allclose(floors, [1.0e4 * 2.0**-20, 1.0e-4, 1.0e-4], rtol=1e-6)
