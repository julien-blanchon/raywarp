"""The process-wide function-hash memo (`core/build/kernel_cache/module_hash.py`)."""

from __future__ import annotations

import warp as wp
from warp._src import context as warp_context

from raywarp.core.build.kernel_cache import module_hash


@wp.func
def _shared(x: float) -> float:
  return x * 2.0 + 1.0


@wp.func
def _caller(x: float) -> float:
  return _shared(x) + _shared(x * 0.5)


def _kernels() -> tuple[wp.Kernel, wp.Kernel]:
  """Two kernels in two modules that both call `_caller`."""

  def body_a(out: wp.array[float]):
    out[0] = _caller(1.0)

  def body_b(out: wp.array[float]):
    out[0] = _caller(2.0) - 1.0

  a = wp.kernel(body_a, module=wp.Module("rw_test_hash_a", None))
  b = wp.kernel(body_b, module=wp.Module("rw_test_hash_b", None))
  return a, b


def test_module_hashes_equal_warps_and_shared_functions_hash_once(monkeypatch) -> None:
  """With the memo, module hashes are Warp's own, and a function reached from two modules is
  hashed by Warp once."""
  module_hash.install()
  a, b = _kernels()
  options = {"enable_backward": False}
  monkeypatch.setattr(warp_context.ModuleHasher, "hash_function", module_hash._WARP_HASH_FUNCTION)
  plain = [warp_context.ModuleHasher([k], options).hash for k in (a, b)]
  monkeypatch.setattr(warp_context.ModuleHasher, "hash_function", module_hash._hash_function)
  module_hash._HASHES.clear()
  calls: list[str] = []
  original = module_hash._WARP_HASH_FUNCTION

  def counted(hasher: warp_context.ModuleHasher, func: warp_context.Function) -> bytes:
    calls.append(func.key)
    return original(hasher, func)

  monkeypatch.setattr(module_hash, "_WARP_HASH_FUNCTION", counted)
  memo = [warp_context.ModuleHasher([k], options).hash for k in (a, b)]
  assert memo == plain
  # A wp.func is a Function (the stubs type it as the Python function).
  assert calls.count(_caller.key) == 1
  assert calls.count(_shared.key) == 1
