"""device_generic: typed PEP 695 generics compile to Warp generics with exact results."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.tables import batched, batched1d, tile_rows
from raywarp.core.kernels.generic import _erase, device_generic, specialize


@wp.kernel
def _read_all(
  f: wp.array2d[float],
  n: wp.array2d[int],
  b: wp.array2d[wp.bool],
  v: wp.array2d[wp.vec3],
  m: wp.array2d[wp.mat33],
  row: wp.array1d[float],
  out: wp.array2d[float],
):
  w, i = wp.tid()
  value = batched(f, w, i) + float(batched(n, w, i)) + batched(v, w, i)[1] + batched(m, w, i)[2, 2]
  if batched(b, w, i):
    value += 100.0
  out[w, i] = value + batched1d(row, w)


def test_batched_reads_every_table_type_with_row_sharing():
  """Shared (1-row) and per-world tables give the numpy result for every element type."""
  nworld, n = 3, 4
  rng = np.random.default_rng(0)
  f = rng.random((nworld, n)).astype(np.float32)
  ints = rng.integers(0, 9, (1, n)).astype(np.int32)  # shared row
  flags = rng.random((nworld, n)) > 0.5
  vecs = rng.random((1, n, 3)).astype(np.float32)  # shared row
  mats = rng.random((nworld, n, 3, 3)).astype(np.float32)
  row = rng.random(nworld).astype(np.float32)
  out = wp.zeros((nworld, n), dtype=float, device="cpu")
  wp.launch(
    _read_all,
    dim=(nworld, n),
    inputs=[
      wp.array(f, dtype=float, device="cpu"),
      wp.array(ints, dtype=int, device="cpu"),
      wp.array(flags, dtype=wp.bool, device="cpu"),
      wp.array(vecs, dtype=wp.vec3, device="cpu"),
      wp.array(mats, dtype=wp.mat33, device="cpu"),
      wp.array(row, dtype=float, device="cpu"),
    ],
    outputs=[out],
    device="cpu",
  )
  expected = f + ints[0][None] + vecs[0, :, 1][None] + mats[:, :, 2, 2] + 100.0 * flags + row[:, None]
  np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)


def test_table_row_reads_world_mod_rows_for_every_row_count():
  """World w reads row w % rows of a batched table for every row count: 1 and nworld (or more)
  rows directly (`table_row`: min(w, rows - 1), branch-free), K rows (1 < K < nworld: MuJoCo
  Warp's domain randomisation) through their nworld-row tiling (`tile_rows`, by the adapter)."""
  nworld = 7
  for nrow in (1, 2, 3, nworld, nworld + 4):
    table = wp.array(np.arange(nrow * 3).astype(np.int32).reshape(nrow, 3) + 100, dtype=int, device="cpu")
    if 1 < nrow < nworld:
      tiled = wp.empty((nworld, 3), dtype=int, device="cpu")
      src = wp.array(ptr=table.ptr, dtype=wp.uint8, shape=(nrow, 12), device="cpu")
      dst = wp.array(ptr=tiled.ptr, dtype=wp.uint8, shape=(nworld, 12), device="cpu")
      wp.launch(tile_rows, dim=(nworld, 12), inputs=[src], outputs=[dst], device="cpu")
      table = tiled
    out = wp.zeros((nworld, 3), dtype=int, device="cpu")
    wp.launch(_read_rows, dim=(nworld, 3), inputs=[table], outputs=[out], device="cpu")
    expected = (np.arange(nworld)[:, None] % nrow) * 3 + np.arange(3)[None] + 100
    np.testing.assert_array_equal(out.numpy(), expected, err_msg=f"{nrow} rows")


@wp.kernel
def _read_rows(
  # In:
  table: wp.array2d[int],
  # Out:
  out: wp.array2d[int],
):
  """`batched` reads of every (world, column) (fixture)."""
  w, i = wp.tid()
  out[w, i] = batched(table, w, i)


def _mixed[T](a: wp.array1d[T], x: T) -> T:
  """Generic over T only (fixture)."""
  return x


def test_type_parameters_become_any_and_warp_type_vars_are_kept():
  """Our type parameters (bare and as array dtypes) become Any; wp.Scalar stays a Warp generic."""
  hints = device_generic(_mixed).__annotations__
  assert hints["x"] is Any and hints["return"] is Any
  assert hints["a"].dtype is Any and hints["a"].ndim == 1
  assert _erase(wp.Scalar) is wp.Scalar


def _make_sum(n: int, generic: bool):
  """A kernel summing an n-vector through a function whose signature names the factory-local
  vector type only (S1: with string annotations Warp alone would raise NameError)."""
  VecN = wp.types.vector(n, dtype=float)

  if generic:

    @wp.func
    @device_generic
    def total[T: float](v: VecN, scale: T) -> float:
      s = scale * 0.0
      for i in range(wp.static(n)):
        s += v[i] * scale
      return s

  else:

    @wp.func
    @device_generic
    def total(v: VecN, scale: float) -> float:
      s = float(0.0)
      for i in range(wp.static(n)):
        s += v[i] * scale
      return s

  @wp.kernel(module="unique", enable_backward=False)
  @device_generic
  def kernel(out: wp.array1d[float], v: VecN):
    out[0] = total(v, 1.0)

  return kernel, VecN


def test_factory_local_names_in_device_signatures():
  """device_generic evaluates string annotations in the defining scope, with or without type
  parameters."""
  for generic in (False, True):
    kernel, vec = _make_sum(3, generic)
    out = wp.zeros(1, dtype=float, device="cpu")
    wp.launch(kernel, dim=1, inputs=[out, vec(1.0, 2.0, 3.0)], device="cpu")
    assert out.numpy()[0] == 6.0


def test_specialize_keys_on_bound_arguments():
  """One build per bound argument set: positional / keyword / default spellings share it, a tuple
  argument does not collide with a keyword, `True` and `1` stay apart (the key was
  `(*args, *sorted(kwargs))`: all three failed)."""
  calls: list[tuple[object, object]] = []

  @specialize
  def g(a: object = None, b: object = False) -> object:
    calls.append((a, b))
    return object()

  assert g(1, False) is g(1, b=False) is g(1) is g(a=1)
  assert len(calls) == 1
  # A positional tuple equal to a keyword item, and True vs 1: separate builds.
  assert g(("b", True)) is not g(b=True)
  assert g(True) is not g(1)
  assert len(calls) == 4
