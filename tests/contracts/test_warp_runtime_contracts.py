"""Private BVH compatibility and bounded compile coordination, without renderer startup."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast, final

import numpy as np
import pytest
import warp as wp

from raywarp.core.pipeline import warmup
from raywarp.core.pipeline.platform import module_lock
from raywarp.core.trace.accel import cast as traversal
from raywarp.core.trace.accel import nodes, query
from raywarp.core.trace.accel.packed import PackedNodes
from raywarp.core.trace.backend import TraceBackend
from raywarp.core.trace.context import FrameContext
from raywarp.core.trace.rays.primary import PINHOLE


@pytest.mark.parametrize("version", ["1.17.2", "1.19.0", "1.20.0", "2.0.0", "unknown"])
def test_unverified_private_bvh_contract_fails_before_compile(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
  monkeypatch.setattr(wp, "__version__", version)
  single = (True, False, False, False, 0)
  shared = (True, True, False, False, 0)
  assert query.module_options((False, False, False, False, 0), ordered=False) == {}
  for trees, ordered in ((single, False), (single, True), (shared, False)):
    with pytest.raises(RuntimeError, match=r"validated warp-lang \(1\.18\.x\)"):
      query.module_options(trees, ordered=ordered)


@pytest.mark.parametrize("version", ["1.18.0", "1.18.3", "1.18.0+owned"])
def test_validated_warp_family_can_select_private_nodes(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
  monkeypatch.setattr(wp, "__version__", version)
  assert query.module_options((True, False, False, False, 0), ordered=True)


@wp.kernel(module="unique", enable_backward=False, module_options={"extra_build_options": nodes.BUILD_OPTIONS})
def _query_nodes(
  # In:
  tree: wp.uint64,
  halves: wp.uint64,
  lower: wp.array1d[wp.vec3],
  upper: wp.array1d[wp.vec3],
  # Out:
  private_hits: wp.array2d[int],
  public_hits: wp.array2d[int],
) -> None:
  """Compare node interpretation and primitive indirection against Warp's public ray query."""
  ray = wp.tid()
  origin = wp.vec3(-2.0, float(ray) + 0.25, 0.25)
  direction = wp.vec3(1.0, 0.0, 0.0)
  reciprocal = nodes.reciprocal(direction)
  root = wp.bvh_get_group_root(tree, 0)
  walk = traversal.Walk()
  stack = traversal.WalkStack()
  traversal.start_unordered(walk, stack, tree, halves, root, origin, reciprocal, 100.0)
  primitive = traversal.next_unordered(walk, stack, origin, reciprocal, 100.0)
  while primitive >= 0:
    # A packed leaf can contain boxes the ray misses; compare actual primitive boxes.
    if nodes.ray_aabb(origin, reciprocal, lower[primitive], upper[primitive], 100.0) >= 0.0:
      private_hits[ray, primitive] = 1
    primitive = traversal.next_unordered(walk, stack, origin, reciprocal, 100.0)
  public = wp.bvh_query_ray(tree, origin, direction, root)
  item = int(0)
  while wp.bvh_query_next(public, item):
    public_hits[ray, item] = 1


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("block_dim", [128, 256])
def test_private_bvh_matches_public_query_after_refit(device: str, block_dim: int) -> None:
  """Mixed native/private queries fit at shared-stack and default local-fallback block sizes."""
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  nodes.require_compatible()
  boxes = np.array([[float(i % 4), float(i // 4), 0.0] for i in range(12)], dtype=np.float32)
  lower = wp.array(boxes, dtype=wp.vec3, device=device)
  upper = wp.array(boxes + 0.5, dtype=wp.vec3, device=device)
  groups = wp.zeros(12, dtype=wp.int32, device=device)
  tree = wp.Bvh(lower, upper, groups=groups, constructor="sah")
  packed = PackedNodes(wp.array([tree.id], dtype=wp.uint64, device=device), meshes=False)
  private_hits = wp.zeros((4, 12), dtype=int, device=device)
  public_hits = wp.zeros((4, 12), dtype=int, device=device)
  for shifted in (False, True):
    if shifted:
      boxes[:, 1] += 1.0
      lower.assign(boxes)
      upper.assign(boxes + 0.5)
      tree.refit()
      packed.refresh()
    private_hits.zero_()
    public_hits.zero_()
    wp.launch(
      _query_nodes,
      dim=4,
      inputs=[tree.id, wp.uint64(int(packed.host_addresses[0])), lower, upper],
      outputs=[private_hits, public_hits],
      device=device,
      block_dim=block_dim,
    )
    observed = private_hits.numpy()
    np.testing.assert_array_equal(observed, public_hits.numpy())
    np.testing.assert_array_equal(observed.sum(axis=1), [0, 4, 4, 4] if shifted else [4, 4, 4, 0])


def test_warmups_serialize_the_same_module(monkeypatch: pytest.MonkeyPatch) -> None:
  module = wp.get_module(__name__)
  entered, release, overlap = threading.Event(), threading.Event(), threading.Event()
  lock = threading.Lock()
  active = 0

  def load(_module: wp.Module, *_args: object, **_kwargs: object) -> None:
    """Warp's `Module.load`, stubbed: counts the loads running at once (`module_lock` wraps it)."""
    nonlocal active
    with lock:
      active += 1
      if active > 1:
        overlap.set()
    entered.set()
    try:
      if not release.wait(3):
        raise TimeoutError("mock module load was not released")
    finally:
      with lock:
        active -= 1

  # The lock is `module_lock`'s, around Warp's own loader: stub that loader, keep the real
  # `wp.force_load` and the installed `Module.load`.
  monkeypatch.setattr(module_lock, "_WARP_LOAD", load)
  with ThreadPoolExecutor(max_workers=2) as pool:
    first = pool.submit(warmup.Warmup._run, "cpu", [module], None)
    try:
      assert entered.wait(3)
      second = pool.submit(warmup.Warmup._run, "cpu", [module], 128)
      assert not overlap.wait(0.1), "the same module loaded concurrently across warmups/block sizes"
    finally:
      release.set()
    first.result(timeout=3)
    second.result(timeout=3)


def test_warmup_preserves_distinct_module_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
  rendezvous = threading.Barrier(2, timeout=3)

  def load(*, device: wp.DeviceLike, modules: list[wp.Module], block_dim: int | None, max_workers: int) -> None:
    rendezvous.wait()

  monkeypatch.setattr(wp, "force_load", load)
  monkeypatch.setattr(warmup, "MAX_WORKERS", 2)
  warmup.Warmup._run("cpu", [wp.get_module(__name__), wp.get_module("raywarp.core.pipeline.cameras")], None)


def test_failed_module_load_releases_coordination_lock(monkeypatch: pytest.MonkeyPatch) -> None:
  calls = 0

  def load(*, device: wp.DeviceLike, modules: list[wp.Module], block_dim: int | None, max_workers: int) -> None:
    nonlocal calls
    calls += 1
    if calls == 1:
      raise RuntimeError("mock compile failure")

  monkeypatch.setattr(wp, "force_load", load)
  module = wp.get_module(__name__)
  with pytest.raises(RuntimeError, match="mock compile failure"):
    warmup.Warmup._run("cpu", [module], None)
  with ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(warmup.Warmup._run, "cpu", [module], None).result(timeout=3)
  assert calls == 2


@final
class _Binding:
  """A user binding in a module that also holds unrelated Warp kernels."""

  def cameras(self, sim: None) -> dict[str, wp.array]:  # `Binding.cameras`
    """No live camera tables."""
    return {}

  def live_inputs(self) -> dict[str, wp.array]:  # `Binding.live_inputs`
    """No live material tables."""
    return {}


@wp.kernel
def _user_kernel(
  # Out:
  output: wp.array1d[int],
) -> None:
  """A user kernel whose module must not be swept into renderer warmup."""
  output[0] = 1


def test_binding_module_kernels_are_not_compiled_without_permission(monkeypatch: pytest.MonkeyPatch) -> None:
  features = SimpleNamespace(gi=False, culling=False, history=False, any_rgb=False, world_optics=False)
  # Only metadata read by first_frame is needed; no renderer/device buffers are constructed.
  context = cast(
    FrameContext,
    SimpleNamespace(
      features=features,
      effective=features,
      scene=SimpleNamespace(nlight=0, deformables=SimpleNamespace(count=0), binding=_Binding(), device="cpu"),
      settings=SimpleNamespace(performance=SimpleNamespace(rebuild_every=0)),
      layout=SimpleNamespace(per_world=np.zeros(1, bool), groups={}),
      per_world_fields=frozenset(),
      camera_model=PINHOLE,
    ),
  )
  backend = cast(TraceBackend, SimpleNamespace(first_frame_kernels=lambda _features, _model, _caps: warmup.FirstFrame()))
  monkeypatch.setattr(warmup, "_validates", lambda _ctx: False)
  monkeypatch.setattr(warmup, "_feature_kernels", lambda _ctx, **_bakes: [])
  monkeypatch.setattr(warmup, "_sensor_kernels", lambda _ctx: [])
  modules, _ = warmup.first_frame(context, backend).modules()
  assert warmup.kernel(_user_kernel).module not in modules
