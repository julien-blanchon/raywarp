"""Mesh pool indexing validates authored values before narrowing and keeps corner rows aligned."""

from __future__ import annotations

import tracemalloc
from typing import Final

import numpy as np
import pytest

from raywarp.scene import SceneBuilder
from raywarp.scene.build.mesh_pool import MeshAttributes
from raywarp.scene.model.types import NO_MESH

_POINTS: Final = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
_TRI: Final = [[0, 1, 2]]


@pytest.mark.parametrize("bad", [2**32, 2**32 + 1, -(2**32)])
def test_triangle_index_cannot_wrap_to_a_valid_vertex(bad: int) -> None:
  with pytest.raises(ValueError, match="triangle index out of range"):
    SceneBuilder().add_mesh(_POINTS, np.array([[bad, 1, 2]], np.int64))


@pytest.mark.parametrize("bad", [0.5, 1.5, float("nan")])
def test_fractional_or_nonfinite_indices_are_not_truncated(bad: float) -> None:
  with pytest.raises(ValueError, match="indices must be integers"):
    SceneBuilder().add_mesh(_POINTS, [[bad, 1, 2]])


@pytest.mark.parametrize("kind", ["normals", "uvs", "uvs1", "colors"])
@pytest.mark.parametrize("rows", [0, 2])
def test_every_corner_pool_has_one_index_row_per_triangle(kind: str, rows: int) -> None:
  attrs: MeshAttributes
  index = np.zeros((rows, 3), np.int32)
  match kind:
    case "normals":
      attrs = {"normals": [[0, 0, 1]], "normal_idx": index}
    case "uvs":
      attrs = {"uvs": [[0, 0]], "uv_idx": index}
    case "uvs1":
      attrs = {"uvs1": [[0, 0]], "uv1_idx": index}
    case _:
      attrs = {"colors": [[1, 1, 1]], "color_idx": index}
  with pytest.raises(ValueError, match="corner indices must have shape"):
    SceneBuilder().add_mesh(_POINTS, _TRI, **attrs)


def test_corner_index_cannot_wrap_to_a_valid_normal() -> None:
  with pytest.raises(ValueError, match="normals index out of range"):
    SceneBuilder().add_mesh(_POINTS, _TRI, normals=[[0, 0, 1]], normal_idx=np.full((1, 3), 2**32, np.int64))


@pytest.mark.parametrize("part", [[], [1], [-1], [0, 0], [0.5]])
def test_single_part_still_validates_authored_part_ids(part: list[int] | list[float]) -> None:
  with pytest.raises(ValueError, match="tri_part"):
    SceneBuilder().add_mesh_parts(_POINTS, _TRI, tri_part=part, nparts=1)


def test_missing_part_ids_put_triangles_in_first_part_and_keep_empty_parts() -> None:
  builder = SceneBuilder()
  parts = builder.add_mesh_parts(_POINTS, _TRI, nparts=3)
  assert parts == [0, NO_MESH, NO_MESH]
  assert builder.add_mesh_parts(_POINTS, _TRI, tri_part=[0], nparts=3) == parts
  scene = builder.build()
  assert scene.nmesh == 1
  np.testing.assert_array_equal(scene.meshes.get(parts[0]).tris, _TRI)


def test_rejected_corner_input_does_not_shift_later_mesh_indices() -> None:
  builder = SceneBuilder()
  with pytest.raises(ValueError, match="corner indices"):
    builder.add_mesh(_POINTS, _TRI, normals=[[0, 0, 1]], normal_idx=[[0, 0, 0], [0, 0, 0]])
  mesh = builder.add_mesh(_POINTS, _TRI, normals=[[0, 0, 1]], normal_idx=[[0, 0, 0]])
  assert mesh == 0
  stored = builder.build().meshes.get(mesh)
  np.testing.assert_array_equal(stored.normal_idx, [[0, 0, 0]])
  np.testing.assert_array_equal(stored.normals[stored.normal_idx], [[[0, 0, 1]] * 3])


@pytest.mark.parametrize(
  ("points", "tris"),
  [
    (np.zeros((6, 2), np.float32), _TRI),  # 12 values would read as 4 points
    (_POINTS, [[0, 1], [2, 0], [1, 2]]),  # 6 indices would read as 2 triangles
    (_POINTS, [0, 1, 2, 0]),  # a flat index list of 4
  ],
)
def test_points_and_triangles_must_be_rows_of_three(points: np.ndarray, tris: list[list[int]] | list[int]) -> None:
  with pytest.raises(ValueError, match="rows of 3 values"):
    SceneBuilder().add_mesh(points, tris)


def test_mesh_registration_does_not_copy_payloads_for_deduplication() -> None:
  """Registration hashes valid contiguous pools without allocating another pool-sized buffer."""
  points = np.arange(300_000).astype(np.float32).reshape(-1, 3)
  triangles = np.arange(99_999).astype(np.int32).reshape(-1, 3)
  builder = SceneBuilder()
  mesh = builder.add_mesh(points, triangles)
  tracemalloc.start()
  try:
    assert builder.add_mesh(points, triangles) == mesh
    _, peak = tracemalloc.get_traced_memory()
  finally:
    tracemalloc.stop()
  assert peak < points.nbytes


def test_stored_corner_tables_expand_without_memory_when_no_mesh_has_them() -> None:
  """A `.wpbr`'s optional corner tables (`Meshes.stored_rows` / `expanded_rows`): a table no mesh
  has comes back as a read-only -1 broadcast (no per-triangle memory), a partial one row-aligned."""
  from raywarp.scene.model.tables import Meshes

  builder = SceneBuilder()
  builder.add_mesh(_POINTS, _TRI)
  builder.add_mesh(_POINTS + 1.0, _TRI, uvs1=[[0, 0], [1, 0]], uv1_idx=[[0, 1, 0]])
  meshes = builder.build().meshes
  arrays = {name: np.asarray(getattr(meshes, name)) for name in ("tris", "tri_adr", "tri_num", "uv1_num", "color_num")}
  restored = Meshes.expanded_rows({**arrays, **meshes.stored_rows()})
  assert restored["color_idx"].shape == (2, 3) and restored["color_idx"].strides == (0, 0)
  assert (restored["color_idx"] == -1).all() and not np.asarray(restored["color_idx"]).flags["WRITEABLE"]
  np.testing.assert_array_equal(restored["uv1_idx"], [[-1, -1, -1], [0, 1, 0]])
