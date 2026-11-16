"""Host-only capture composition and external-buffer ownership contracts."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, final

import pytest
import warp as wp

from raywarp.core.pipeline import graph
from raywarp.core.pipeline.gpu_scene.binding import StaleBinding
from raywarp.core.pipeline.renderer import Renderer

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray


@final
class _Renderer:
  def __init__(self) -> None:
    self.device = wp.get_device("cpu")
    self.events: list[str] = []
    self.arrays: dict[str, AnyDeviceArray] = {}
    self._closed = False

  def _check_open(self, call: str) -> None:
    Renderer._check_open(self, call)  # pyrefly: ignore[bad-argument-type]  # the stand-in has its state

  def _graph_version(self) -> int:
    return 0

  def _graph_arrays(self, sim: AnyDeviceArray | None) -> dict[str, AnyDeviceArray]:
    return {**self.arrays, **({"simulation": sim} if sim is not None else {})}

  def _before_replay(self, sim: AnyDeviceArray | None) -> None:
    self.events.append("before")

  def _replayed(self) -> None:
    pass

  def _seed_array(self, seed: int | wp.array[wp.int32, [1]]) -> wp.array[wp.int32, [1]]:
    assert isinstance(seed, wp.array)
    return seed

  def render(
    self, sim: AnyDeviceArray | None, *, seed: wp.array[wp.int32, [1]], active: wp.array[wp.int32, [int]] | None = None
  ) -> None:
    self.events.append("render")


@final
class _Segments:
  def __init__(self) -> None:
    self.launches = 0

  def launch(self) -> None:
    self.launches += 1


def test_capture_orders_policy_after_current_observation(monkeypatch: pytest.MonkeyPatch) -> None:
  renderer, segments = _Renderer(), _Segments()

  def record(device: wp.DeviceLike, work: Callable[[], None], *, before_record: Callable[[], None] | None = None) -> _Segments:
    work()  # The eager preparation and recording both execute the complete composed work.
    if before_record is not None:
      before_record()
    work()
    return segments

  monkeypatch.setattr(graph, "capture", record)
  seed = wp.zeros(1, dtype=wp.int32, device="cpu")
  captured = Renderer.capture(
    renderer,  # pyrefly: ignore[bad-argument-type]  # Exercise orchestration with a host-only stand-in.
    step=lambda: renderer.events.append("physics"),
    seed=seed,
    after=lambda: renderer.events.append("policy"),
  )
  assert renderer.events == ["physics", "render", "policy"] * 2
  captured.launch()
  assert segments.launches == 1


@pytest.mark.parametrize("change", ["pointer", "shape", "strides", "dtype", "removed", "added"])
def test_external_buffer_changes_reject_replay(change: str) -> None:
  renderer, segments = _Renderer(), _Segments()
  storage = wp.zeros(8, dtype=wp.float32, device="cpu")
  arrays: dict[str, AnyDeviceArray] = {"policy.weights": storage}
  captured = graph.RenderGraph(
    renderer,
    segments,  # pyrefly: ignore[bad-argument-type]  # Host-only segments.
    None,
    arrays=lambda: arrays,
  )
  captured.launch()
  assert segments.launches == 1
  renderer.events.clear()
  if change == "pointer":
    arrays["policy.weights"] = wp.zeros(8, dtype=wp.float32, device="cpu")
  elif change == "shape":
    arrays["policy.weights"] = storage.reshape((2, 4))
  elif change == "strides":
    arrays["policy.weights"] = wp.array(ptr=storage.ptr, shape=(8,), strides=(8,), dtype=wp.float32, device="cpu")
  elif change == "dtype":
    arrays["policy.weights"] = storage.view(wp.int32)
  elif change == "removed":
    arrays.clear()
  else:
    arrays["policy.bias"] = wp.zeros(8, dtype=wp.float32, device="cpu")
  with pytest.raises(StaleBinding, match="caller"):
    captured.launch()
  assert not renderer.events  # No before_replay work queued on a stale caller binding.
  assert segments.launches == 1


def test_graph_retains_external_storage_and_callback_owner() -> None:
  renderer, segments = _Renderer(), _Segments()
  storage = wp.zeros(2, dtype=wp.float32, device="cpu")
  owner = _Renderer()
  array_ref, owner_ref = weakref.ref(storage), weakref.ref(owner)
  arrays: dict[str, AnyDeviceArray] = {"policy.actions": storage}
  captured = graph.RenderGraph(
    renderer,  # Protocol-conforming host renderer.
    segments,  # pyrefly: ignore[bad-argument-type]  # Host-only segments.
    None,
    arrays=lambda: arrays,
    work=lambda held=owner: held.events.clear(),
  )
  arrays.clear()
  del storage, owner
  gc.collect()
  assert array_ref() is not None
  assert owner_ref() is not None
  del captured
  gc.collect()
  assert array_ref() is None
  assert owner_ref() is None


def test_capture_rejects_replaced_declared_buffer_while_recording(monkeypatch: pytest.MonkeyPatch) -> None:
  renderer, segments = _Renderer(), _Segments()
  arrays: dict[str, AnyDeviceArray] = {"actions": wp.zeros(2, device="cpu")}
  previous = weakref.ref(arrays["actions"])
  recording = False

  def policy() -> None:
    if recording:
      arrays["actions"] = wp.zeros(2, device="cpu")
      gc.collect()
      assert previous() is not None  # The original captured storage remains owned until rejection.

  def record(device: wp.DeviceLike, work: Callable[[], None], *, before_record: Callable[[], None] | None = None) -> _Segments:
    nonlocal recording
    work()
    assert before_record is not None
    before_record()
    recording = True
    work()
    return segments

  monkeypatch.setattr(graph, "capture", record)
  with pytest.raises(StaleBinding, match="during capture"):
    Renderer.capture(
      renderer,  # pyrefly: ignore[bad-argument-type]  # Host-only orchestration stand-in.
      seed=wp.zeros(1, dtype=wp.int32, device="cpu"),
      after=policy,
      arrays=lambda: arrays,
    )
  assert segments.launches == 0


def test_removed_renderer_binding_rejects_before_replay() -> None:
  renderer, segments = _Renderer(), _Segments()
  renderer.arrays["instance.pos"] = wp.zeros(2, dtype=wp.vec3, device="cpu")
  captured = graph.RenderGraph(
    renderer,
    segments,  # pyrefly: ignore[bad-argument-type]  # Host-only segments.
    None,
  )
  renderer.arrays.clear()
  with pytest.raises(StaleBinding, match="renderer array names"):
    captured.launch()
  assert not renderer.events
  assert segments.launches == 0


def test_default_replay_retains_and_reuses_captured_simulator_state() -> None:
  renderer, segments = _Renderer(), _Segments()
  state = wp.zeros(2, dtype=wp.float32, device="cpu")
  state_ref = weakref.ref(state)
  captured = graph.RenderGraph(
    renderer,
    segments,  # pyrefly: ignore[bad-argument-type]  # Host-only segments.
    state,
  )
  del state
  gc.collect()
  assert state_ref() is not None
  captured.launch()
  assert segments.launches == 1
  with pytest.raises(StaleBinding, match="simulation"):
    captured.launch(wp.zeros(2, dtype=wp.float32, device="cpu"))
  assert segments.launches == 1


@final
class _FrameRenderer:
  """A `graph.Recordable` stand-in: its version, arrays and the frames it ran."""

  def __init__(self) -> None:
    self.device = wp.get_device("cpu")
    self.version = 0
    self.arrays: dict[str, AnyDeviceArray] = {"instance.pos": wp.zeros(2, dtype=wp.vec3, device="cpu")}
    self.events: list[str] = []
    self.recording = False
    self.bump_on_replay = False

  def _graph_version(self) -> int:
    return self.version

  def _graph_arrays(self, sim: AnyDeviceArray | None) -> dict[str, AnyDeviceArray]:
    return {**self.arrays, **({"simulation": sim} if sim is not None else {})}

  def _before_replay(self, sim: AnyDeviceArray | None) -> None:
    self.events.append("checks")
    if self.bump_on_replay:
      self.version += 1

  def _replayed(self) -> None:
    self.events.append("replayed")

  def eager_frame(self, sim: AnyDeviceArray | None, seed: wp.array[wp.int32, [1]], *, polled: bool) -> None:
    self.events.append("record" if self.recording else "eager" if not polled else "eager-polled")


@final
class _FrameSegments:
  def __init__(self, events: list[str]) -> None:
    self.events = events
    self.reason: str | None = None

  def stale(self) -> str | None:
    return self.reason

  def launch(self) -> None:
    self.events.append("replay")


type _Render = Callable[[AnyDeviceArray | None, wp.array[wp.int32, [1]]], None]


def _frame_graphs(monkeypatch: pytest.MonkeyPatch) -> tuple[_FrameRenderer, _Render, list[_FrameSegments]]:
  """A FrameGraphs' render of a stand-in renderer whose recordings are host-only segments."""
  renderer, recorded = _FrameRenderer(), []

  def record(device: wp.Device, work: Callable[[], None]) -> _FrameSegments:
    renderer.recording = True
    work()
    renderer.recording = False
    recorded.append(_FrameSegments(renderer.events))
    return recorded[-1]

  monkeypatch.setattr(graph, "record", record)
  frames = graph.FrameGraphs()

  def render(sim: AnyDeviceArray | None, seed: wp.array[wp.int32, [1]]) -> None:
    frames.render(renderer, sim, seed)

  return renderer, render, recorded


def test_render_records_the_second_frame_with_repeated_inputs_then_replays(monkeypatch: pytest.MonkeyPatch) -> None:
  """The first frame runs eagerly, the second (same inputs) is recorded and launched, later ones
  replay after the host checks."""
  r, render, recorded = _frame_graphs(monkeypatch)
  seed = wp.zeros(1, dtype=wp.int32, device="cpu")
  for _ in range(4):
    render(None, seed)
  assert r.events == ["eager", "checks", "record", "replay", "checks", "replay", "checks", "replay"]
  assert len(recorded) == 1


def test_render_runs_eagerly_on_new_inputs_and_keeps_alternating_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
  """A new simulator buffer, seed array or version runs eagerly; two alternating state buffers
  (double-buffered simulators) each get their own graph."""
  r, render, recorded = _frame_graphs(monkeypatch)
  seed = wp.zeros(1, dtype=wp.int32, device="cpu")
  a, b = wp.zeros(4, device="cpu"), wp.zeros(4, device="cpu")
  for state in (a, b, a, b, a, b):
    render(state, seed)
  assert r.events == ["eager", "eager"] + ["checks", "record", "replay"] * 2 + ["checks", "replay"] * 2
  assert len(recorded) == 2
  r.events.clear()
  render(a, wp.zeros(1, dtype=wp.int32, device="cpu"))  # another seed array
  r.version += 1
  render(a, seed)  # a changed renderer: every graph dropped
  render(a, seed)
  assert r.events == ["eager", "eager", "checks", "record", "replay"]
  assert len(recorded) == 3


def test_render_checks_change_the_renderer_or_refuse_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
  """A replay whose host checks change the renderer, or whose graph is refused (`Segments.stale`),
  runs eagerly without polling twice, and the refused graph is recorded again."""
  r, render, recorded = _frame_graphs(monkeypatch)
  seed = wp.zeros(1, dtype=wp.int32, device="cpu")
  render(None, seed)
  render(None, seed)
  r.events.clear()
  r.bump_on_replay = True
  render(None, seed)
  r.bump_on_replay = False
  assert r.events == ["checks", "eager-polled"]
  render(None, seed)
  recorded[-1].reason = "the OptiX pipeline of this graph was released"
  r.events.clear()
  render(None, seed)
  render(None, seed)
  assert r.events == ["checks", "eager-polled", "checks", "record", "replay"]
  assert len(recorded) == 3


def test_render_keeps_a_graph_while_arrays_are_rewrapped(monkeypatch: pytest.MonkeyPatch) -> None:
  """A new array object over the same storage (e.g. a tensor wrapped again every frame) keeps the
  graph: the inputs are compared by address, shape, strides, dtype and device."""
  r, render, recorded = _frame_graphs(monkeypatch)
  seed = wp.zeros(1, dtype=wp.int32, device="cpu")
  storage = wp.zeros(4, device="cpu")
  for _ in range(3):
    render(wp.array(ptr=storage.ptr, shape=(4,), dtype=wp.float32, device="cpu"), seed)
  assert r.events == ["eager", "checks", "record", "replay", "checks", "replay"]
  assert len(recorded) == 1


def test_frame_graphs_keep_no_reference_to_their_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
  """The renderer owns its FrameGraphs and is each call's argument: after recorded frames the
  graphs hold no reference back to it, so a dropped renderer is freed by reference counting (its
  device arrays at once, no collection)."""

  def record(_device: wp.Device, work: Callable[[], None]) -> _FrameSegments:
    work()
    return _FrameSegments([])

  monkeypatch.setattr(graph, "record", record)
  frames, r = graph.FrameGraphs(), _FrameRenderer()
  seed = wp.zeros(1, dtype=wp.int32, device="cpu")
  for _ in range(3):
    frames.render(r, None, seed)
  alive = weakref.ref(r)
  gc.disable()
  try:
    del r
    assert alive() is None
  finally:
    gc.enable()
