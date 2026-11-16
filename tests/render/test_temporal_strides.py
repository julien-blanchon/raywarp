"""Actual native descriptor/offset regressions without allocating or accessing large images."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp


@wp.func_native("return wp::uint64(wp::byte_offset(a, i, j, k));")
def _color_offset(
  # Model (descriptor only, never dereferenced):
  a: wp.array3d[wp.vec4h],
  # In:
  i: int,
  j: int,
  k: int,
) -> wp.uint64:
  """Use the actual native array-address helper, stopping before any data access."""
  return wp.uint64(0)  # Native snippet supplies the compiled implementation.


@wp.func_native("return wp::uint64(wp::byte_offset(a, i, j, k));")
def _depth_offset(
  # Model (descriptor only, never dereferenced):
  a: wp.array3d[float],
  # In:
  i: int,
  j: int,
  k: int,
) -> wp.uint64:
  """Return the native byte offset for a planar-depth descriptor."""
  return wp.uint64(0)


@wp.func_native("return wp::uint64(wp::byte_offset(a, i, j, k));")
def _id_offset(
  # Model (descriptor only, never dereferenced):
  a: wp.array3d[int],
  # In:
  i: int,
  j: int,
  k: int,
) -> wp.uint64:
  """Return the native byte offset for an instance-ID descriptor."""
  return wp.uint64(0)


@wp.kernel
def _offsets(
  # Model (descriptor-only advertised extents):
  color: wp.array3d[wp.vec4h],
  depth: wp.array3d[float],
  ids: wp.array3d[int],
  # Data in:
  color_coordinates: wp.array1d[wp.vec3i],
  geometry_coordinates: wp.array1d[wp.vec3i],
  # Out:
  output: wp.array2d[wp.uint64],
):
  """Only the tiny coordinates/output are read/written; image addresses are never accessed."""
  n = wp.tid()
  c = color_coordinates[n]
  g = geometry_coordinates[n]
  output[n, 0] = _color_offset(color, c[0], c[1], c[2])
  output[n, 1] = _depth_offset(depth, g[0], g[1], g[2])
  output[n, 2] = _id_offset(ids, g[0], g[1], g[2])


def test_original_native_descriptor_overflow_is_reproduced_without_image_allocation() -> None:
  """Python's correct stride tuple is insufficient: inspect actual exported ctypes fields."""
  owner = wp.zeros(1, dtype=wp.uint8, device="cpu")
  color = wp.array(ptr=owner.ptr, shape=(2, 512, 1048576), dtype=wp.vec4h, device="cpu")
  depth = wp.array(ptr=owner.ptr, shape=(2, 512, 1048576), dtype=wp.float32, device="cpu")
  assert owner.capacity == 1
  assert color.strides[0] == 2**32
  assert depth.strides[0] == 2**31
  # No .numpy(), .fill_(), copy or data kernel may touch these synthetic extents.
  assert color.__ctype__().strides[0] == 0
  assert depth.__ctype__().strides[0] == -(2**31)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("resolution", [512, 1024])
def test_pool_and_world_major_native_offsets_are_exact_beyond_four_gib(device: str, resolution: int) -> None:
  """Actual native offset calculation covers distant worlds, both slots and last pixels."""
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  owner = wp.zeros(1, dtype=wp.uint8, device=device)
  pixels = resolution**2
  total = 4 * pixels
  color = wp.array(ptr=owner.ptr, shape=(2112, 1, pixels), dtype=wp.vec4h, device=device)
  depth = wp.array(ptr=owner.ptr, shape=(512, 2, total), dtype=wp.float32, device=device)
  ids = wp.array(ptr=owner.ptr, shape=(512, 2, total), dtype=wp.int32, device=device)
  for view in (color, depth, ids):
    assert tuple(view.__ctype__().strides)[:3] == view.strides
    assert all(stride is not None and 0 < stride < 2**31 for stride in view.strides)
  cs = [(0, 0, 0), (1, 0, pixels - 1), (1024, 0, 1), (2047, 0, pixels - 1), (2111, 0, pixels - 1)]
  gs = [(0, 0, 0), (0, 1, total - 1), (256, 0, 1), (511, 0, total - 1), (511, 1, total - 1)]
  c = wp.array(cs, dtype=wp.vec3i, device=device)
  g = wp.array(gs, dtype=wp.vec3i, device=device)
  output = wp.zeros((len(cs), 3), dtype=wp.uint64, device=device)
  wp.launch(_offsets, len(cs), inputs=[color, depth, ids, c, g], outputs=[output], device=device)
  expected = np.array(
    [
      [(ci * pixels + ck) * 8, (gi * 2 * total + gj * total + gk) * 4, (gi * 2 * total + gj * total + gk) * 4]
      for (ci, _, ck), (gi, gj, gk) in zip(cs, gs, strict=True)
    ],
    dtype=np.uint64,
  )
  np.testing.assert_array_equal(output.numpy(), expected)
  assert int(expected[-1, 0]) > 2**32
  assert owner.capacity == 1


@pytest.mark.parametrize("resolution", [512, 1024])
def test_production_temporal_allocations_export_safe_large_strides(monkeypatch: pytest.MonkeyPatch, resolution: int) -> None:
  """Exercise the real allocator decisions with one-byte borrowed descriptors, never storage."""
  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.post.temporal import history

  with wp.ScopedDevice("cpu"):
    owner = wp.zeros(1, dtype=wp.uint8, device="cpu")

    def zeros(shape: int | tuple[int, ...], dtype: type, *, device: wp.Device) -> AnyDeviceArray:
      """Record exactly the production shape/dtype in a real exported Warp descriptor."""
      return wp.array(ptr=owner.ptr, shape=shape, dtype=dtype, device=device)

    with monkeypatch.context() as patch:
      patch.setattr(wp, "zeros", zeros)
      temporal = history._temporal(None, status=True, shape=(512, 4, 4 * resolution**2), device=wp.get_device("cpu"))
    assert temporal.gz.shape == (512, 2, 4 * resolution**2)
    for array in (temporal.frame, temporal.count, temporal.gz, temporal.prev_camera):
      native = tuple(array.__ctype__().strides)[: array.ndim]
      assert native == array.strides
      assert all(0 < stride < 2**31 for stride in native)
    assert owner.capacity == 1
