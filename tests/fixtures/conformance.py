"""The adapter conformance kit: the checks every adapter passes, run on one `ConformanceCase` per
adapter by tests/contracts/test_conformance.py.

  HostAdapter        Ingest + Bind (`load(source) -> Scene`, `binding(scene, source)`): USD, Newton
  DirectAdapter      one step to a GpuScene (`scene(mjm, m) -> GpuScene`): MuJoCo
  host_case / direct_case  a ConformanceCase built through the adapter's protocol

  compare            the same scene from two sources (formats or adapters) renders the same
                     geometry and base colors: coverage, depth and albedo within `Tolerance`
                     (`measure`: the same measures of an adapter's own renders, e.g. an
                     integration's outputs, against `render_view`'s)
  check_no_recompile per-world randomisation writes never recompile (Features and kernels kept)
  allocations        the CPU allocations inside a block (steady frames allocate nothing)
  report_mismatches  a load report's counters against the expected ones (real assets)
  check_graph_replay step + render captured in one CUDA graph replays the eager frame (CUDA)

`compare` renders with unlit shading (`conftest.UNLIT_FAST`): base colors, not the sources' light
units, are what two ingests of one scene must agree on (lighting has its own references).

Known difference (decided by the user): an MJCF through Newton's importer does not match the
MuJoCo adapter's colors. Newton defines `shape_color` as sRGB (its viewer and SensorCamera decode
it; the Newton adapter follows them, so the drop-ins keep parity), while its MJCF importer stores
MuJoCo's linear `rgba` unconverted: the MJCF's colors come out sRGB-decoded through Newton (mean
base-color difference 0.24 on the feature scene). Compare such pairs on geometry only.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import threading
from collections.abc import Callable, Generator, Mapping
from typing import TYPE_CHECKING, Final, Protocol, final

import numpy as np
import warp as wp
import warp._src.context as warp_context
from conftest import UNLIT_FAST

from raywarp.core.pipeline.gpu_scene.binding import Binding
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, GpuScene
from raywarp.core.pipeline.renderer import Renderer
from raywarp.core.pipeline.upload.build import upload
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.settings import Outputs

if TYPE_CHECKING:
  from shape_extensions import IntVar

# Measurements compare one image, with height and width axes (no world / camera batch).
_IMAGE_NDIM: Final = 2

# A free camera pose (7,): position, then quaternion (x, y, z, w): a `wp.transformf` (OpenGL
# camera frame).
type Pose = np.ndarray[[7], np.dtype[np.float32]] | np.ndarray[[7], np.dtype[np.float64]]
# Per-world table values (`Renderer.per_world[field].assign`): (nworld,), (nworld, n) or
# (nworld, n, k).
type Values = (
  np.ndarray[[int], np.dtype[np.float32]]
  | np.ndarray[[int, int], np.dtype[np.float32]]
  | np.ndarray[[int, int, int], np.dtype[np.float32]]
)
# A host copy of one camera's output: (nworld, 1, H, W), with a trailing channel axis for vectors.
type HostOutput = (
  np.ndarray[[int, int, int, int], np.dtype[np.generic]] | np.ndarray[[int, int, int, int, int], np.dtype[np.generic]]
)

# One view's (depth (H, W), linear albedo (H, W, 3)) as `compare` measures them.
type ViewRender[H: IntVar, W: IntVar] = tuple[
  np.ndarray[[H, W], np.dtype[np.float32]], np.ndarray[[H, W, 3], np.dtype[np.float32]]
]


class HostAdapter[Source, S](Protocol):
  """An adapter in two steps (satisfied by its module: `raywarp.formats.usd`,
  `raywarp.integrations.newton`).

  `Source` is what it reads (a USD path, a Newton model); `S` the simulator state its binding
  reads (`Renderer.render(sim: S)`; None for a static stage). Adapter-specific options are
  keyword arguments with defaults beyond these signatures.
  """

  def load(self, source: Source, /) -> Scene:
    """Ingest: `source` -> host `Scene`, its `LoadReport` in `Scene.report`."""
    ...

  def binding(self, scene: Scene, source: Source, /) -> Binding[S]:
    """Bind: the `Binding` that drives `scene` from `source`'s per-step state."""
    ...


class DirectAdapter[Model, DeviceModel, S](Protocol):
  """An adapter that ingests and binds in one step, aliasing the simulator's tables
  (`raywarp.integrations.mujoco`: `scene(mjm, m) -> GpuScene[mjw.Data]`). It has no host Scene and
  no load report, and owns the per-frame capability check of its live tables."""

  def scene(self, model: Model, device_model: DeviceModel, /) -> GpuScene[S]:
    """Ingest + Bind: the GpuScene of `model`, aliasing `device_model`."""
    ...


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConformanceCase[S]:
  """One adapter's scene, as every conformance check takes it.

  Attributes:
    name: the adapter ("mujoco", "usd", ...).
    scene: its GpuScene.
    sim: the simulator state `Renderer.render` takes (None: static).
    report: the ingest's load report (None for a DirectAdapter: none is made).
    step: one simulation step, captured with the render by `check_graph_replay`.
  """

  name: str
  scene: GpuScene[S]
  sim: S
  report: LoadReport | None
  step: Callable[[], None] = lambda: None


def host_case[Source, S](
  name: str, adapter: HostAdapter[Source, S], source: Source, sim: S, *, device: wp.DeviceLike
) -> ConformanceCase[S]:
  """The case of a HostAdapter: `load(source)`, bound by `binding(scene, source)`, uploaded to
  `device`."""
  host = adapter.load(source)
  bound = upload(host, device, binding=adapter.binding(host, source))
  return ConformanceCase(name=name, scene=bound, sim=sim, report=host.report)


def direct_case[Model, DeviceModel, S](
  name: str, adapter: DirectAdapter[Model, DeviceModel, S], model: Model, device_model: DeviceModel, sim: S
) -> ConformanceCase[S]:
  """The case of a DirectAdapter: `scene(model, device_model)` (no load report)."""
  return ConformanceCase(name=name, scene=adapter.scene(model, device_model), sim=sim, report=None)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Tolerance:
  """What `compare` accepts.

  Attributes:
    coverage: least fraction of pixels both renders hit or both miss.
    depth: largest 95th-percentile relative depth difference where both hit.
    albedo: largest mean absolute base-color difference where both hit.
  """

  coverage: float = 0.98
  depth: float = 0.01
  albedo: float = 0.02

  def __post_init__(self) -> None:
    """Requires finite, meaningful comparison thresholds."""
    if not math.isfinite(self.coverage) or not 0.0 <= self.coverage <= 1.0:
      raise ValueError(f"coverage tolerance must be finite in [0, 1], got {self.coverage}")
    for name, value in (("depth", self.depth), ("albedo", self.albedo)):
      if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} tolerance must be finite and nonnegative, got {value}")


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Comparison:
  """`compare`'s measures and verdict (`failures` empty: within tolerance)."""

  coverage: float
  depth: float
  albedo: float
  failures: tuple[str, ...]


def render_view[S](
  scene: GpuScene[S], sim: S, camera: CameraSpec, pose: Pose, resolution: tuple[int, int]
) -> ViewRender[int, int]:
  """(depth (H, W), albedo (H, W, 3)) of world 0 seen by a free `camera` at `pose` (7,), unlit."""
  settings = UNLIT_FAST.replace(outputs=Outputs.of("depth", "albedo"))
  r = Renderer(scene, nworld=1, cameras=[camera], resolution=resolution, settings=settings)
  r.camera_pose.assign(np.asarray(pose, np.float32).reshape(1, 1, 7))
  r.render(sim)
  depth, albedo = r.outputs.depth, r.outputs.albedo
  if depth is None or albedo is None:
    raise RuntimeError("render_view: the renderer wrote no depth / albedo (both requested)")
  return depth.numpy()[0, 0], albedo.numpy()[0, 0]


def compare[A, B](  # noqa: PLR0913  # the two scenes, then the view keyword-only
  a: tuple[GpuScene[A], A],
  b: tuple[GpuScene[B], B],
  *,
  camera: CameraSpec,
  pose: Pose,
  resolution: tuple[int, int] = (64, 48),
  tol: Tolerance = Tolerance(),  # noqa: B008  # a frozen value
) -> Comparison:
  """Renders (scene, simulator state) `a` and `b` from one free camera and compares them."""
  return measure(render_view(*a, camera, pose, resolution), render_view(*b, camera, pose, resolution), tol=tol)


def _validate_view[H: IntVar, W: IntVar](view: ViewRender[H, W], *, name: str) -> None:
  """Checks measurement inputs; zero and positive infinity are valid depth misses."""
  depth, albedo = view
  if depth.ndim != _IMAGE_NDIM or depth.size == 0 or albedo.shape != (*depth.shape, 3):
    raise ValueError(f"{name}: expected nonempty depth (H, W) and albedo (H, W, 3), got {depth.shape}, {albedo.shape}")
  if np.isnan(depth).any() or (depth < 0.0).any():
    raise ValueError(f"{name}: depth must be nonnegative and not NaN (zero or +inf for misses)")
  hit = np.isfinite(depth) & (depth > 0.0)
  if not np.isfinite(albedo[hit]).all():
    raise ValueError(f"{name}: albedo must be finite at every depth hit")


def measure[H: IntVar, W: IntVar](
  a: ViewRender[H, W],
  b: ViewRender[H, W],
  *,
  tol: Tolerance = Tolerance(),  # noqa: B008  # a frozen value
) -> Comparison:
  """Compares two renders (depth (H, W), linear albedo (H, W, 3)) of one view: `compare`'s
  measures, for an adapter rendering through its own API (an integration's outputs).

  Raises:
    ValueError: for empty / mismatched shapes, invalid depth, or non-finite albedo at a hit.
  """
  _validate_view(a, name="first render")
  _validate_view(b, name="second render")
  (da, aa), (db, ab) = a, b
  if da.shape != db.shape:
    raise ValueError(f"render depth shapes must match, got {da.shape} and {db.shape}")
  hit_a, hit_b = np.isfinite(da) & (da > 0.0), np.isfinite(db) & (db > 0.0)
  coverage = float(np.mean(hit_a == hit_b))
  both = hit_a & hit_b
  depth = float(np.percentile(np.abs(da[both] - db[both]) / da[both], 95)) if both.any() else 0.0
  albedo = float(np.mean(np.abs(aa[both] - ab[both]))) if both.any() else 0.0
  failures: list[str] = []
  if coverage < tol.coverage:
    failures.append(f"coverage {coverage:.3f} < {tol.coverage}")
  if depth > tol.depth:
    failures.append(f"depth p95 {depth:.4f} > {tol.depth}")
  if albedo > tol.albedo:
    failures.append(f"albedo {albedo:.4f} > {tol.albedo}")
  return Comparison(coverage=coverage, depth=depth, albedo=albedo, failures=tuple(failures))


@contextlib.contextmanager
def module_loads() -> Generator[list[str]]:
  """The names of the Warp modules the process builds or loads from disk inside the block (every
  `Module.load` that finds no module already in the process, whatever wraps the factories).

  The warm-ups still running when the block starts (of renderers an earlier test made and never
  rendered: their threads outlive the test) are waited for first: their loads are not the block's.
  """
  for thread in threading.enumerate():
    if thread.name.startswith("rw_warmup") and thread is not threading.current_thread():
      thread.join()
  loads: list[str] = []
  build = warp_context.Module._compile

  def counted(module: warp_context.Module, *args: object, **kwargs: object) -> bool:
    """The module's build, recorded."""
    loads.append(module.name)
    return build(module, *args, **kwargs)  # pyrefly: ignore[bad-argument-type]

  warp_context.Module._compile = counted  # pyrefly: ignore[bad-assignment]
  try:
    yield loads
  finally:
    warp_context.Module._compile = build


@contextlib.contextmanager
def allocations() -> Generator[list[int]]:
  """The sizes [bytes] of the Warp CPU-device allocations inside the block (every new `wp.array`
  of a CPU renderer: a steady frame should make none)."""
  sizes: list[int] = []
  allocate = warp_context.CpuDefaultAllocator.allocate

  def counted(self: warp_context.CpuDefaultAllocator, size_in_bytes: int) -> int:
    """The allocation, recorded."""
    sizes.append(size_in_bytes)
    return allocate(self, size_in_bytes)

  warp_context.CpuDefaultAllocator.allocate = counted
  try:
    yield sizes
  finally:
    warp_context.CpuDefaultAllocator.allocate = allocate


def check_no_recompile[S](renderer: Renderer[S], sim: S, writes: Mapping[str, Values]) -> None:
  """Writes per-world tables (`renderer.per_world[field]`, declared fields) and renders, twice
  (the second time with the values halved): the Features stay the same and neither write nor
  frame loads a Warp module.

  Raises:
    AssertionError: when a write reconfigured the renderer or compiled a kernel.
  """
  features = renderer.features
  renderer.render(sim)
  with module_loads() as loads:
    for scale in (1.0, 0.5):
      for field, values in writes.items():
        renderer.per_world[field].assign(np.multiply(values, scale, dtype=np.float32))
      renderer.render(sim)
  # The kit's verdicts are AssertionErrors (it runs inside tests), raised explicitly: `python -O`
  # must not skip them.
  if renderer.features is not features:
    raise AssertionError("a per-world write reconfigured the renderer")
  if loads:
    raise AssertionError(f"a per-world write compiled kernels: {loads}")


def report_mismatches(report: LoadReport, expected: Mapping[str, int]) -> dict[str, tuple[int, int]]:
  """{counter: (expected, got)} of a load report's dropped / approximated counters that differ
  from `expected` (counters not in `expected` are expected 0)."""
  got = {
    **{f"dropped:{k}": v for k, v in report.dropped.items()},
    **{f"approximated:{k}": v for k, v in report.approximated.items()},
  }
  keys = set(got) | set(expected)
  return {k: (expected.get(k, 0), got.get(k, 0)) for k in sorted(keys) if expected.get(k, 0) != got.get(k, 0)}


def check_graph_replay[S](renderer: Renderer[S], step: Callable[[], None], sim: S) -> bool:
  """Captures `step()` + `render(sim)` in one CUDA graph and checks a replay (into cleared outputs)
  writes what an eager frame writes from the same state (the graph holds no host round trip).
  False on a device without graph capture (nothing checked).

  Raises:
    AssertionError: when the replay differs.
  """
  if not wp.get_device(renderer.device).is_cuda:
    return False
  graph = renderer.capture(step, sim)
  for a in renderer.outputs.arrays().values():  # the capture's eager frame: cleared, so a replay
    a.zero_()  # that writes nothing fails
  graph.launch()
  wp.synchronize_device(renderer.device)
  replayed = _snapshot(renderer)
  step()
  renderer.render(sim)
  wp.synchronize_device(renderer.device)
  for name, values in _snapshot(renderer).items():
    np.testing.assert_array_equal(values, replayed[name], err_msg=name)
  return True


def _snapshot[S](renderer: Renderer[S]) -> dict[str, HostOutput]:
  """Host copies of the renderer's per-camera outputs."""
  out = {}
  for c, v in enumerate(renderer.outputs.cameras):
    named = {"rgb": v.rgb, "depth": v.depth, "seg": v.seg, "hdr": v.hdr, "albedo": v.albedo}
    named |= {"normal": v.normal, "motion": v.motion}
    out |= {f"{name}{c}": a.numpy() for name, a in named.items() if a is not None}
  return out
