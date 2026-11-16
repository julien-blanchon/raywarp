"""Pose factories preserve world/column dimensions through both binding implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import numpy as np
import warp as wp

from raywarp.core.pipeline.gpu_scene.binding import Binding, InstancePoses, PoseBuffers, StaticPoses
from raywarp.integrations.newton.binding import BindingTables, NewtonBinding

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import Ids


def _allocate[S, NWorld: IntVar, NColumn: IntVar](
  binding: Binding[S], worlds: Int[NWorld], columns: Ids[NColumn]
) -> PoseBuffers[NWorld, NColumn] | None:
  """A caller through the base binding preserves both dimensions, including aliasing None."""
  return binding.allocate(worlds, columns, device=wp.get_device("cpu"))


def test_static_pose_allocation_preserves_named_dimensions_and_values() -> None:
  positions = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], np.float32)
  rotations = np.zeros((3, 3, 3), dtype=np.float32)
  rotations[:] = np.eye(3)
  binding = StaticPoses(InstancePoses(positions, rotations))
  columns = np.array([2, 0], np.int32)
  poses = assert_type(binding.allocate(3, columns, device=wp.get_device("cpu")), PoseBuffers[3, 2])
  assert_type(poses.pos, "wp.array[wp.vec3, [3, 2]]")
  assert_type(poses.rot, "wp.array[wp.mat33, [3, 2]]")
  generic = assert_type(_allocate(binding, 3, columns), "PoseBuffers[3, 2] | None")
  assert generic is not None
  np.testing.assert_array_equal(poses.pos.numpy(), np.broadcast_to(positions[[2, 0]], (3, 2, 3)))
  np.testing.assert_array_equal(poses.rot.numpy(), np.broadcast_to(np.eye(3), (3, 2, 3, 3)))
  assert poses.pos.device == wp.get_device("cpu")


def test_newton_pose_allocation_preserves_named_dimensions_and_values() -> None:
  local = np.zeros((1, 3, 7), dtype=np.float32)
  local[0, :, :3] = [[0, 0, 0], [1, 2, 3], [4, 5, 6]]
  local[:, :, 6] = 1
  tables = BindingTables(
    body=np.full((1, 3), -1, np.int32),
    local=local,
    world=np.full((1, 3), -1, np.int32),
    shape=np.array([[0, 1, 2]], np.int32),
  )
  binding = NewtonBinding(tables=tables, nsource=1, device="cpu", shape_transform=None)
  columns = np.array([2, 0], np.int32)
  poses = assert_type(binding.allocate(3, columns, device=wp.get_device("cpu")), PoseBuffers[3, 2])
  assert_type(poses.pos, "wp.array[wp.vec3, [3, 2]]")
  assert_type(poses.rot, "wp.array[wp.mat33, [3, 2]]")
  np.testing.assert_array_equal(poses.pos.numpy(), np.broadcast_to(local[0, [2, 0], :3], (3, 2, 3)))
  np.testing.assert_array_equal(poses.rot.numpy(), np.broadcast_to(np.eye(3), (3, 2, 3, 3)))
  assert poses.pos.device == wp.get_device("cpu")
