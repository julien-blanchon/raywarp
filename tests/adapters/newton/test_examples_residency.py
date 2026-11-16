"""GPU residency across Newton's examples: SensorCamera and the PBR viewer on
`newton.examples`.

Each example runs with its own simulation; after two warm-up frames a steady-state frame of
`SensorCamera.update` (and of `ViewerRaywarpHeadless`) allocates nothing and copies nothing but the
camera poses (device -> device) and the validation flag word (`core/pipeline/worlds/validator.py`).
On CPU every copy is a host copy, so the test checks what is copied; on CUDA (GPU checkpoint) it
also checks that no copy crosses the host / device boundary and that `update()` captures in one CUDA
graph with the physics step.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import math
from collections.abc import Callable, Generator
from typing import Protocol

import numpy as np
import pytest
import warp as wp
from conftest import required

newton = pytest.importorskip("newton")
pytest.importorskip("newton.examples")

from raywarp.integrations.newton.headless import ViewerRaywarpHeadless
from raywarp.integrations.newton.sensor import SensorCamera

# CPU-light examples (rigid bodies; XPBD / Featherstone).
EXAMPLES = ["basic.example_basic_shapes", "basic.example_basic_pendulum", "basic.example_basic_joints"]
_COPIES = ("wp_memcpy_h2d", "wp_memcpy_d2h", "wp_memcpy_d2d", "wp_memcpy_p2p", "wp_memcpy_h2h")
_ALLOCS = ("wp_alloc_host", "wp_alloc_pinned", "wp_alloc_device", "wp_alloc_device_default", "wp_alloc_device_async")
# Byte-count argument of each copy (warp/_src/context.py argtypes).
_SIZE_ARG = {"wp_memcpy_h2d": 3, "wp_memcpy_d2h": 3, "wp_memcpy_d2d": 3, "wp_memcpy_p2p": 4, "wp_memcpy_h2h": 2}
_FLAG_BYTES = 4  # core/pipeline/worlds/validator.py: the flag word (one int32) copied to pinned memory per frame
_RES = 32


@contextlib.contextmanager
def _traffic() -> Generator[list[tuple[str, int]]]:
  """Records the Warp runtime's copies and allocations (name, bytes) inside the block."""
  from warp._src.context import runtime

  core = required(runtime).core
  events: list[tuple[str, int]] = []
  originals = {name: getattr(core, name) for name in (*_COPIES, *_ALLOCS) if hasattr(core, name)}

  def wrap(name: str, fn: Callable[..., object]) -> Callable[..., object]:
    """`fn` recording its call (name, bytes) in `events`."""
    index = _SIZE_ARG.get(name)

    def wrapper(*args: object) -> object:
      size = _nbytes(args[index]) if index is not None and len(args) > index else 0
      events.append((name, size))
      return fn(*args)

    return wrapper

  for name, fn in originals.items():
    setattr(core, name, wrap(name, fn))
  try:
    yield events
  finally:
    for name, fn in originals.items():
      setattr(core, name, fn)


def _nbytes(arg: object) -> int:
  """A byte-count argument of a runtime copy: an int or a ctypes integer (its `value`)."""
  value = getattr(arg, "value", arg)
  return value if isinstance(value, int) else 0


class _Example(Protocol):
  """What the tests use of a Newton example (`newton.examples.*.Example`)."""

  model: newton.Model
  state_0: newton.State

  def step(self) -> None:
    """Advances the simulation one frame (its own CUDA graph when it has one)."""
    ...

  def simulate(self) -> None:
    """The simulation substeps of one frame (no graph launch)."""
    ...

  def render(self) -> None:
    """Logs the frame to the example's viewer."""
    ...


def _example(module: str, viewer: newton.viewer.ViewerBase) -> _Example:
  """Newton example `module` built with its default arguments on the current device."""
  example_module = importlib.import_module(f"newton.examples.{module}")
  args = newton.examples.create_parser().parse_args([])
  return example_module.Example(viewer, argparse.Namespace(**{**vars(args), "test": True}))


def _camera(
  model: newton.Model, count: int, target: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> wp.array[wp.transformf, [int]]:
  """`count` cameras 4 m away looking at `target` (default the origin; model up axis)."""
  up = int(newton.Axis.from_any(model.up_axis))
  offset = np.zeros(3)
  offset[(up + 1) % 3], offset[up] = -4.0, 1.5
  eye = np.asarray(target, np.float64) + offset
  forward = -offset / np.linalg.norm(offset, keepdims=True)
  world_up = np.eye(3)[up]
  right = np.cross(forward, world_up)
  right /= np.linalg.norm(right, keepdims=True)
  rot = np.stack([right, np.cross(right, forward), -forward], axis=1).astype(np.float32)
  cam = wp.transformf(wp.vec3f(*eye.astype(np.float32)), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
  return wp.array([cam] * count, dtype=wp.transformf)


def _check_steady(events: list[tuple[str, int]], frames: int, pose_bytes: int, device: wp.DeviceLike) -> None:
  """Steady-state traffic: no allocation; copies only of the camera poses and the flag word."""
  allocs = [e for e in events if e[0] in _ALLOCS]
  assert not allocs, f"steady-state frames allocated: {allocs}"
  copies = [e for e in events if e[0] in _COPIES]
  unexpected = [e for e in copies if e[1] not in (pose_bytes, _FLAG_BYTES)]
  assert not unexpected, f"unexpected copies: {unexpected}"
  assert len(copies) <= 2 * frames
  if wp.get_device(device).is_cuda:
    crossing = [
      e for e in copies if e[0] in ("wp_memcpy_h2d", "wp_memcpy_h2h") or (e[0] == "wp_memcpy_d2h" and e[1] > _FLAG_BYTES)
    ]
    assert not crossing, f"host round trips: {crossing}"


@pytest.mark.parametrize("module", EXAMPLES)
def test_sensor_camera_steady_state_on_newton_examples(module):
  """SensorCamera.update on a running example: renders, then allocates and copies nothing new."""
  device = wp.get_device()
  example = _example(module, newton.viewer.ViewerNull(num_frames=100))
  model = example.model
  views = min(int(model.world_count), 2)
  cams = _camera(model, views)
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=math.radians(60.0), device=device)
  sensor = SensorCamera(model)
  color = sensor.allocate_image_color(views, _RES, _RES)
  index = sensor.allocate_image_shape_index(views, _RES, _RES)

  def update() -> None:
    sensor.update(example.state_0, cams, rays, color_image=color, shape_index_image=index)

  for _ in range(2):  # warm up: compiles, buffers
    example.step()
    update()
  frames = 3
  events: list[tuple[str, int]] = []
  for _ in range(frames):
    example.step()  # the simulation's own traffic is Newton's: not measured
    with _traffic() as frame_events:
      update()
    events += frame_events
  _check_steady(events, frames, views * 7 * 4, device)
  # It renders the example: shapes hit, their indices are model shapes.
  ids = index.numpy()
  hit = ids != 0xFFFFFFFF
  assert hit.mean() > 0.01 and ids[hit].max() < model.shape_count
  assert (color.numpy()[hit] & 0xFFFFFF).any()


@pytest.mark.parametrize("module", EXAMPLES[:1])
def test_headless_viewer_runs_newton_example(module):
  """ViewerRaywarpHeadless as an example's viewer: every frame is ray traced, steady state
  resident."""
  pytest.importorskip("pyglet", reason="Newton's viewer Camera (ViewerRaywarpHeadless's) needs pyglet")
  viewer = ViewerRaywarpHeadless(width=_RES, height=_RES, num_frames=6)
  example = _example(module, viewer)
  frames = 0
  while viewer.is_running():
    example.step()
    if frames == 3:
      with _traffic() as events:
        example.render()
      _check_steady([e for e in events if e[0] in _ALLOCS], 1, -1, wp.get_device())
    else:
      example.render()
    frames += 1
  frame = viewer.get_frame().numpy()
  assert frame.shape == (_RES, _RES, 3) and frame.std() > 1.0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="GPU checkpoint: CUDA graphs and host / device copies")
@pytest.mark.parametrize("module", EXAMPLES)
def test_newton_example_step_and_update_capture_in_one_graph(module):
  """Physics step + SensorCamera.update captured together; replays follow the simulation."""
  with wp.ScopedDevice("cuda:0"):
    example = _example(module, newton.viewer.ViewerNull(num_frames=100))
    model = example.model
    views = min(int(model.world_count), 2)
    # Aimed at the bodies: the pendulum hangs ~5 m up, out of the origin camera's view, where 20
    # steps moved only its shadow and changed no pixel (the failure since GPU-3).
    bodies = example.state_0.body_q.numpy()[:, :3]
    cams = _camera(model, views, target=bodies.mean(axis=0) if len(bodies) else (0.0, 0.0, 0.0))
    rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=math.radians(60.0))
    sensor = SensorCamera(model)
    color = sensor.allocate_image_color(views, _RES, _RES)

    def frame() -> None:
      example.simulate()  # the example's step without its own graph launch
      sensor.update(example.state_0, cams, rays, color_image=color)

    graph = sensor.capture(frame)
    wp.synchronize()
    first = color.numpy().copy()
    with _traffic() as events:
      for _ in range(20):
        graph.launch()
      wp.synchronize()
    _check_steady([e for e in events if e[0] != "wp_memcpy_d2h"], 20, -1, "cuda:0")
    assert not np.array_equal(color.numpy(), first)  # bodies moved in the replays
