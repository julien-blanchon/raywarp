"""CUDA graphs of a renderer's frames: recorded once, replayed while the arrays they read are the
captured ones.

`Renderer.render` replays its own graphs (`FrameGraphs`): a frame is recorded when its inputs (the
arrays it reads: `Replayable._graph_arrays`, the seed array, `_graph_version`) repeat those of an
earlier frame, and replayed for every later frame with the same inputs; any other frame runs
eagerly. `capture(device, work)` (Renderer.capture, Newton's SensorCamera.capture) records a
caller's work around a render (`RenderGraph`): run once eagerly (compiles, builds, first-frame
allocations), then recorded. Stream capture cannot record OptiX launches, so a recording is cut
around them (`core/trace/optix/graph.py` `Segments`): a single graph segment without OptiX.

`wp.ScopedDevice` (rule 38's one exception): `work()` runs with the capture's device as Warp's
default because it holds the caller's simulator step (MuJoCo Warp, Newton launch on the default
device); the renderer's own work names its device (tests/render/test_explicit_devices.py). A
replay on the current stream changes neither Warp's default device nor its current stream; one on
another stream makes that stream current meanwhile (`Segments.launch`, core/trace/optix/graph.py).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, Protocol, final

import warp as wp

from raywarp.core.pipeline.gpu_scene.binding import StaleBinding
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.trace.optix.graph import Segments

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray


def capture(device: wp.DeviceLike, work: Callable[[], None], *, before_record: Callable[[], None] | None = None) -> Segments:
  """Captures `work()` on `device` after one eager run; `Segments.launch()` replays it.

  Raises:
    RuntimeError: on a device without graph capture (CPU).
  """
  device = wp.get_device(device)
  if not device_caps(device).graphs:
    raise RuntimeError("CUDA graph capture needs a CUDA device")
  with wp.ScopedDevice(device):
    work()
    if before_record is not None:
      before_record()
    return record(device, work)


def record(device: wp.Device, work: Callable[[], None]) -> Segments:
  """Records `work()` on `device`'s current stream (nothing runs: `Segments.launch()` does)."""
  with Segments(wp.get_stream(device)) as segments:
    work()
  return segments


# What a captured graph bakes in of an array: its address (None: empty), shape and strides (a
# different array allocated at the same address, e.g. by a caching allocator, can have another
# shape).
type Fingerprint = tuple[int | None, tuple[int, ...], tuple[int | None, ...], str, wp.Device | None]
type ArraySource = Callable[[], Mapping[str, AnyDeviceArray]]


def fingerprint(array: AnyDeviceArray) -> Fingerprint:
  """Storage address, shape, strides, element type and device baked into captured work."""
  ptr = array.ptr
  return (
    None if ptr is None else int(ptr),
    tuple(int(n) for n in array.shape),
    tuple(array.strides),
    str(array.dtype),
    array.device,
  )


# A call's arrays: (name, fingerprint) in the order the renderer lists them.
type Prints = tuple[tuple[str, Fingerprint], ...]


@final
class Bindings:
  """Named arrays and their fingerprints, compared with a call's: by identity first (a Warp array
  object keeps its address, shape and strides), by fingerprint for another object.

  Attributes:
    prints: the fingerprints by name, in order (the same tuple object while nothing changes).
  """

  def __init__(self, arrays: Mapping[str, AnyDeviceArray]) -> None:
    """The fingerprints of `arrays` (held: an id is not reused while held)."""
    self._held: dict[str, tuple[AnyDeviceArray, Fingerprint]] = {}
    self.prints: Prints = ()
    self.update(arrays)

  def update(self, arrays: Mapping[str, AnyDeviceArray]) -> bool:
    """Takes `arrays` as the current ones; whether any is another object or the names differ.
    Only the arrays of other objects are fingerprinted again."""
    held = self._held
    if len(arrays) == len(held) and all((h := held.get(name)) is not None and h[0] is a for name, a in arrays.items()):
      return False
    self._held = held = {
      name: (a, h[1] if (h := held.get(name)) is not None and h[0] is a else fingerprint(a)) for name, a in arrays.items()
    }
    self.prints = tuple((name, h[1]) for name, h in held.items())
    return True


def _address(fp: Fingerprint | None) -> str:
  """A fingerprint in an error message."""
  if fp is None:
    return "none"
  ptr = "empty" if fp[0] is None else f"0x{fp[0]:x}"
  return f"{ptr} shape {fp[1]} strides {fp[2]} dtype {fp[3]} device {fp[4]}"


class Replayable[S](Protocol):
  """What a `RenderGraph` checks on the renderer it captured (implemented by `Renderer`)."""

  device: wp.Device

  def _graph_version(self) -> int:
    """Changes whenever a captured graph becomes stale (settings, outputs, rebuilt structures)."""
    ...

  def _graph_arrays(self, sim: S | None) -> dict[str, AnyDeviceArray]:
    """Slot / binding source name -> array of a call: what a captured graph must find unchanged."""
    ...

  def _before_replay(self, sim: S | None) -> None:
    """The host work of a frame that a graph cannot replay (the validation word's poll, the
    binding's host staging, the GI groups' overflow read)."""
    ...

  def _replayed(self) -> None:
    """Counts a replayed frame (`Renderer.camera_info`'s frame index)."""
    ...


class Recordable[S](Replayable[S], Protocol):
  """A renderer whose frames `FrameGraphs` records (implemented by `Renderer`)."""

  def eager_frame(self, sim: S | None, seed: wp.array[wp.int32, [1]], *, polled: bool) -> None:
    """Enqueues one frame's work for `sim` (recorded when the device's stream is capturing);
    `polled`: `_before_replay` already ran for it."""
    ...


@final
class RenderGraph[S]:
  """A CUDA graph of `step` + `render(sim)` + optional `after` (see `Renderer.capture`).

  `launch(sim)` re-resolves the slots and raises `StaleBinding` when an array differs from the
  captured one (address, shape or strides: e.g. a swapped simulator state) or the renderer changed
  since the capture. It never synchronises: launches queue on the current stream.

  Attributes:
    renderer: the captured renderer.
    segments: the captured graphs and, with traversal="optix", the OptiX calls enqueued between
      them (`core/trace/optix/graph.py`; a single graph otherwise).
  """

  def __init__(
    self,
    renderer: Replayable[S],
    segments: Segments,
    sim: S | None,
    *,
    seed: wp.array[wp.int32, [1]] | None = None,
    arrays: ArraySource | None = None,
    work: Callable[[], None] | None = None,
    owners: Mapping[str, AnyDeviceArray] | None = None,
  ) -> None:
    """A graph of `renderer`'s frame for `sim` (its arrays fingerprinted now)."""
    self.renderer = renderer
    self._sim = sim  # the state `launch()` replays (the captured one)
    self.seed = seed  # the device seed every replay reads (kept alive)
    self.segments = segments
    self._version = renderer._graph_version()  # noqa: SLF001  # the renderer's graph protocol
    self._arrays = Bindings(renderer._graph_arrays(sim))  # noqa: SLF001  # the renderer's graph protocol
    self._captured = self._arrays.prints
    self._work = work  # the captured callbacks (and the simulator / policy state they hold)
    self._array_source = arrays
    self._owners = dict(owners) if owners is not None else (None if arrays is None else dict(arrays()))
    self._external = None if self._owners is None else Bindings(self._owners)
    self._external_captured = None if self._external is None else self._external.prints

  def launch(self, sim: S | None = None) -> None:
    """Replays the graph for `sim` (None: the captured simulator state) after checking its arrays
    (no device read).

    Raises:
      StaleBinding: for arrays other than the captured ones, or a renderer changed since.
    """
    sim = self._sim if sim is None else sim
    r = self.renderer
    if r._graph_version() != self._version:  # noqa: SLF001  # the renderer's graph protocol
      raise StaleBinding(
        "the renderer changed since the capture (update_settings, bound outputs or per-world storage, "
        "replaced tables or a rebuild request): capture again"
      )
    _check(self._arrays, self._captured, r._graph_arrays(sim), "renderer")  # noqa: SLF001  # the renderer's graph protocol
    if self._array_source is not None and self._external is not None and self._external_captured is not None:
      _check(self._external, self._external_captured, self._array_source(), "caller")
    r._before_replay(sim)  # noqa: SLF001  # the renderer's graph protocol
    if r._graph_version() != self._version:  # noqa: SLF001  # the renderer's graph protocol
      raise StaleBinding(
        "the renderer changed at this replay's checks (the shared OptiX structure rebuilt after it moved, the "
        "Features re-resolved for a new shadow caster, or the GI re-baked for worlds whose inputs match no baked "
        "group): capture again"
      )
    self.segments.launch()
    r._replayed()  # noqa: SLF001  # the renderer's graph protocol


def _check(held: Bindings, captured: Prints, current: Mapping[str, AnyDeviceArray], owner: str) -> None:
  """Raises StaleBinding when `current` differs from the `captured` fingerprints (`held` tracks
  the objects last seen: unchanged objects are not fingerprinted again)."""
  held.update(current)
  if held.prints is not captured:
    if held.prints != captured:
      check_arrays(current, dict(captured), owner)
    held.prints = captured  # equal: compared by identity from now on


def check_arrays(current: Mapping[str, AnyDeviceArray], captured: Mapping[str, Fingerprint], owner: str) -> None:
  """Raises StaleBinding when the `current` arrays of `owner` differ from the `captured`
  fingerprints (checked before any replay work is queued; no device read)."""
  if current.keys() != captured.keys():
    raise StaleBinding(f"{owner} array names changed since capture: capture again")
  for name, array in current.items():
    previous = captured[name]
    now = fingerprint(array)
    if previous != now:
      raise StaleBinding(
        f"{owner} {name}: captured {_address(previous)}, got {_address(now)} "
        "(keep captured storage; Newton state swaps need an even number of substeps)"
      )


# Graphs (and recent frame inputs) a renderer keeps: a simulator alternating between a few state
# buffers (Newton's double-buffered states) replays one graph per buffer.
CAPACITY: Final = 4


@final
class FrameGraphs[S]:
  """The CUDA graphs `Renderer.render` replays (see the module doc).

  A frame whose inputs (`_graph_version`, the fingerprints of `_graph_arrays` and of the seed array)
  equal those after one of the last `CAPACITY` frames is recorded (it then runs as a replay), and
  every later frame with those inputs replays it after `_before_replay`. Any other frame runs
  eagerly. A replay whose `_before_replay` changes the renderer, or whose OptiX objects were
  released (`Segments.stale`), runs eagerly instead. A version change drops every graph (they are
  stale). The renderer is each call's argument, never kept: the graphs it owns hold no reference
  back to it (a dropped renderer is freed by reference counting, no collection).
  """

  def __init__(self) -> None:
    """No graph yet."""
    self._inputs = Bindings({})
    self._graphs: list[tuple[int, Prints, Segments]] = []  # (version, inputs, graph), oldest first
    self._seen: list[tuple[int, Prints]] = []  # the inputs after the last eager frames

  def render(self, r: Recordable[S], sim: S | None, seed: wp.array[wp.int32, [1]]) -> None:
    """Enqueues renderer `r`'s frame of `sim` with seed array `seed`: replayed, recorded or
    eager."""
    version, prints = r._graph_version(), self._prints(r, sim, seed)  # noqa: SLF001  # the renderer's graph protocol
    segments = self._find(version, prints)
    if segments is None and not any(v == version and _same(p, prints) for v, p in self._seen):
      r.eager_frame(sim, seed, polled=False)
      self._remember(r, sim, seed)
      return
    r._before_replay(sim)  # noqa: SLF001  # the renderer's graph protocol
    if r._graph_version() != version or (segments is not None and segments.stale() is not None):  # noqa: SLF001  # the renderer's graph protocol
      self._graphs = [g for g in self._graphs if g[2] is not segments]
      r.eager_frame(sim, seed, polled=True)
      self._remember(r, sim, seed)
      return
    if segments is None:
      segments = self._record(r, sim, seed)
      if segments is None:
        return
    segments.launch()

  def clear(self) -> None:
    """Drops every graph (`Renderer.close`)."""
    self._graphs.clear()
    self._seen.clear()

  def _prints(self, r: Recordable[S], sim: S | None, seed: wp.array[wp.int32, [1]]) -> Prints:
    """The fingerprints of the frame's inputs."""
    arrays = r._graph_arrays(sim)  # noqa: SLF001  # the renderer's graph protocol
    arrays["render.seed"] = seed
    self._inputs.update(arrays)
    return self._inputs.prints

  def _find(self, version: int, prints: Prints) -> Segments | None:
    """The graph recorded with these inputs, or None."""
    for v, p, segments in self._graphs:
      if v == version and _same(p, prints):
        return segments
    return None

  def _record(self, r: Recordable[S], sim: S | None, seed: wp.array[wp.int32, [1]]) -> Segments | None:
    """Records the frame (kept with its inputs) and returns it to replay; None when the recording
    changed the renderer (dropped: the frame then ran eagerly)."""
    version, prints = r._graph_version(), self._inputs.prints  # noqa: SLF001  # the renderer's graph protocol
    segments = record(r.device, lambda: r.eager_frame(sim, seed, polled=True))
    if r._graph_version() != version:  # noqa: SLF001  # the renderer's graph protocol
      r.eager_frame(sim, seed, polled=True)
      self._remember(r, sim, seed)
      return None
    self._graphs = [*self._graphs[-(CAPACITY - 1) :], (version, prints, segments)]
    return segments

  def _remember(self, r: Recordable[S], sim: S | None, seed: wp.array[wp.int32, [1]]) -> None:
    """Notes the inputs after an eager frame (a next frame with them is recorded); drops the
    graphs and inputs of other versions."""
    version, prints = r._graph_version(), self._prints(r, sim, seed)  # noqa: SLF001  # the renderer's graph protocol
    self._graphs = [g for g in self._graphs if g[0] == version]
    seen = [k for k in self._seen if k[0] == version and not _same(k[1], prints)]
    self._seen = [*seen[-(CAPACITY - 1) :], (version, prints)]


def _same(a: Prints, b: Prints) -> bool:
  """Equal inputs (the same tuple object while nothing changed)."""
  return a is b or a == b
