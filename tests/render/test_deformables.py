"""Core deformables (`core/pipeline/gpu_scene/deformables.py`, `core/trace/accel/deformable_bvh.py`)
without any simulator."""

import dataclasses

import numpy as np
import pytest
import warp as wp

from raywarp.core.pipeline.gpu_scene.deformables import Deformables, Part
from raywarp.core.trace.accel.deformable_bvh import DeformableBvhs
from raywarp.scene.model.types import MaterialId

_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]

# A unit square cloth at z = 1 (two triangles, four border edges), a rope of two segments and a
# tetrahedron's surface.
_SQUARE = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], np.float32)
_ROPE = np.array([[3, 0, 0], [3, 0, 1], [3, 0, 2]], np.float32)
_TET = np.array([[5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1]], np.float32)
_VERT = np.concatenate([_SQUARE, _ROPE, _TET])
_PARTS = [
  Part(
    dim=2,
    vertadr=0,
    vertnum=4,
    radius=0.05,
    rgba=(1.0, 0.0, 0.0, 1.0),
    faces=np.array([[0, 1, 2], [0, 2, 3]], np.int32),
    borders=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], np.int32),
  ),
  Part(dim=1, vertadr=4, vertnum=3, radius=0.02, rgba=(0.0, 1.0, 0.0, 1.0), edges=np.array([[0, 1], [1, 2]], np.int32)),
  Part(
    dim=3,
    vertadr=7,
    vertnum=4,
    radius=0.0,
    rgba=(0.0, 0.0, 1.0, 1.0),
    faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32),
  ),
]


@wp.kernel
def _cast(
  # In:
  mesh: wp.uint64,
  root: int,
  origin: wp.vec3,
  direction: wp.vec3,
  # Out:
  dist_out: wp.array[float],
  sign_out: wp.array[float],
):
  """Closest hit of one ray with one group of a mesh (-1: miss) and the face side (+1: front)."""
  q = wp.mesh_query_ray(mesh, origin, direction, 100.0, root)
  dist_out[0] = wp.where(q.result, q.t, -1.0)
  sign_out[0] = q.sign


def _ray(bvhs: DeformableBvhs, part: int, world: int, origin, direction) -> tuple[float, float]:
  """(distance, side) of a ray against world `world`'s triangles of part `part`."""
  dist, sign = wp.zeros(1, dtype=float), wp.zeros(1, dtype=float)
  mesh = wp.uint64(int(bvhs.ids.numpy()[part]))
  root = int(bvhs.group_root.numpy()[world, part])
  wp.launch(_cast, dim=1, inputs=[mesh, root, wp.vec3(*origin), wp.vec3(*direction)], outputs=[dist, sign])
  return float(dist.numpy()[0]), float(sign.numpy()[0])


def test_create_primitives():
  d = Deformables.create(_PARTS, _VERT, device=wp.get_device())
  assert d.count == 3 and d.nprim == 4  # the cloth, two rope segments, the tetrahedron
  assert d.prim_flexid.numpy().tolist() == [0, 1, 1, 2]
  assert d.prim_edgeid.numpy().tolist() == [-1, 0, 1, -1]
  assert d.edge.numpy().tolist() == [[0, 1], [1, 2]]  # local to the rope
  assert d.faces[2].min() == 7  # global vertex ids
  empty = Deformables.empty(wp.get_device())
  assert empty.count == 0 and empty.nprim == 0


def test_create_rejects_out_of_range_topology():
  """The kernels index the vertices unchecked: a part past the given vertices, or topology past
  its own range, is refused at creation."""
  rope = _PARTS[1]
  with pytest.raises(ValueError, match="outside the 11 given"):
    Deformables.create([dataclasses.replace(rope, vertadr=9)], _VERT, device=wp.get_device())
  with pytest.raises(ValueError, match="edges index vertices outside its 3"):
    Deformables.create([dataclasses.replace(rope, edges=np.array([[1, 3]], np.int32))], _VERT, device=wp.get_device())


def test_create_rejects_invalid_parts():
  """Review C O1: an unknown dim (built as a surface), a NaN or negative radius, a color outside
  [0, 1] and a material id below -1 were accepted and reached the kernels."""
  rope = _PARTS[1]
  for bad, match in (
    (dataclasses.replace(rope, dim=4), "dim must be 1, 2 or 3"),  # pyrefly: ignore[bad-argument-type]  # deliberately invalid
    (dataclasses.replace(rope, radius=float("nan")), "radius must be a finite length"),
    (dataclasses.replace(rope, radius=-0.1), "radius"),
    (dataclasses.replace(rope, rgba=(1.0, 0.0, 0.0, 2.0)), r"rgba must be 4 values in \[0, 1\]"),
    (dataclasses.replace(rope, rgba=(1.0, float("nan"), 0.0, 1.0)), "rgba"),
    (dataclasses.replace(rope, material=MaterialId(-2)), "material must be an id"),
  ):
    with pytest.raises(ValueError, match=match):
      Deformables.create([bad], _VERT, device=wp.get_device())
  Deformables.create([dataclasses.replace(_PARTS[2], radius=float("nan"))], _VERT, device=wp.get_device())  # dim 3: no radius


def test_check_materials_bounds_the_material_ids():
  """Review C O1: a part's material id >= the scene's material count reached the flex shading
  kernels, which read it unchecked (`Renderer` checks it against `GpuScene.nmaterial`)."""
  d = Deformables.create([dataclasses.replace(_PARTS[0], material=MaterialId(3)), _PARTS[1]], _VERT, device=wp.get_device())
  assert d.max_material == 3
  d.check_materials(4)
  with pytest.raises(ValueError, match="a deformable has material 3, the scene 3 materials"):
    d.check_materials(3)
  Deformables.create([_PARTS[1]], _VERT, device=wp.get_device()).check_materials(0)  # -1: no material


@pytest.mark.parametrize("device", _DEVICES)
def test_thick_surface_faces_outwards_and_refits_per_world(device):
  with wp.ScopedDevice(device):
    bvhs = DeformableBvhs(Deformables.create(_PARTS, _VERT, device=wp.get_device()), nworld=2, device=wp.get_device())
    assert bvhs.group_root.numpy()[:, 1].tolist() == [-1, -1]  # the rope has no triangle BVH
    # Both offset copies face outwards: a ray from above hits the top (z = 1.05) front side, a ray
    # from below the bottom (z = 0.95) front side.
    dist, sign = _ray(bvhs, 0, 0, (0.6, 0.3, 3.0), (0.0, 0.0, -1.0))
    assert dist == pytest.approx(1.95, abs=1e-5) and sign > 0
    dist, sign = _ray(bvhs, 0, 0, (0.6, 0.3, -1.0), (0.0, 0.0, 1.0))
    assert dist == pytest.approx(1.95, abs=1e-5) and sign > 0
    # A side wall closes the border: a horizontal ray at the cloth's height hits x = 0.
    dist, _ = _ray(bvhs, 0, 0, (-1.0, 0.5, 1.0), (1.0, 0.0, 0.0))
    assert dist == pytest.approx(1.0, abs=1e-5)
    # World 1's cloth rises by 1 m; world 0's stays.
    vert = np.stack([_VERT, _VERT])
    vert[1, :4, 2] += 1.0
    bvhs.refit(wp.array(vert, dtype=wp.vec3))
    assert _ray(bvhs, 0, 0, (0.6, 0.3, 3.0), (0.0, 0.0, -1.0))[0] == pytest.approx(1.95, abs=1e-5)
    assert _ray(bvhs, 0, 1, (0.6, 0.3, 3.0), (0.0, 0.0, -1.0))[0] == pytest.approx(0.95, abs=1e-5)
    # The tetrahedron (closed surface, no thickness).
    assert _ray(bvhs, 2, 1, (5.2, 0.2, 3.0), (0.0, 0.0, -1.0))[0] == pytest.approx(2.4, abs=1e-5)


def test_vertex_normals_are_smooth():
  # A cloth folded along its diagonal: the vertices on the fold average both faces' normals.
  vert = _VERT.copy()
  vert[3, 2] = 2.0  # lift vertex 3: face (0, 2, 3) tilts
  bvhs = DeformableBvhs(Deformables.create(_PARTS, vert, device=wp.get_device()), nworld=1, device=wp.get_device())
  bvhs.refit(wp.array(vert[np.newaxis], dtype=wp.vec3))
  normal = bvhs._normal.numpy()[0]
  n1 = np.array([0.0, 0.0, 1.0])
  n2 = np.array([1.0, -1.0, 1.0]) / 3.0**0.5  # (v2 - v0) x (v3 - v0), normalised
  expected = (n1 + n2) / np.linalg.norm(n1 + n2).item()
  np.testing.assert_allclose(normal[0], expected, atol=1e-6)
  np.testing.assert_allclose(normal[1], n1, atol=1e-6)
  np.testing.assert_allclose(normal[3], n2, atol=1e-6)


def test_corner_uvs_follow_the_mesh_triangles():
  """Each mesh triangle's corner UVs are its vertices' (the +radius copy, the rewound -radius copy,
  the side walls by vertex id); a part without texcoords has no UV row."""
  uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
  cloth = dataclasses.replace(_PARTS[0], material=MaterialId(3), texcoord=uv, face_texcoord=_PARTS[0].faces)
  d = Deformables.create([cloth, *_PARTS[1:]], _VERT, device=wp.get_device())
  assert d.has_material and d.material.numpy().tolist() == [3, -1, -1]
  assert d.nface.numpy().tolist() == [12, 0, 4] and d.uvadr.numpy().tolist() == [0, -1, -1]
  corners = d.corner_uv.numpy().reshape(-1, 3, 2)
  assert len(corners) == 12
  np.testing.assert_array_equal(corners[0], uv[[0, 1, 2]])  # face 0, +radius copy
  np.testing.assert_array_equal(corners[1], uv[[0, 2, 1]])  # face 0, -radius copy (rewound)
  np.testing.assert_array_equal(corners[4], uv[[0, 1, 1]])  # border (0, 1): (b0+, b1-, b1+)
  np.testing.assert_array_equal(corners[5], uv[[1, 0, 0]])  # (b1-, b0+, b0-)
  bad = dataclasses.replace(cloth, face_texcoord=np.array([[0, 1, 4], [0, 2, 3]], np.int32))
  with pytest.raises(ValueError, match="face_texcoord"):
    Deformables.create([bad], _VERT[:4], device=wp.get_device())


def test_negative_face_texcoord_is_rejected():
  """A negative `face_texcoord` row passed the check (only the maximum was bounded) and NumPy's
  indexing then read a corner UV from the end of `texcoord`."""
  uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
  bad = dataclasses.replace(_PARTS[0], texcoord=uv, face_texcoord=np.array([[0, 1, -1], [0, 2, 3]], np.int32))
  with pytest.raises(ValueError, match="face_texcoord"):
    Deformables.create([bad], _VERT[:4], device=wp.get_device())
