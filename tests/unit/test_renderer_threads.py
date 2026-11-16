"""Renderers built, warmed, rendered and closed on several threads at once (stress test).

Each thread uploads its own scene (different lights, materials, GI on or off), builds a Renderer,
renders and closes it, several rounds, while another thread reads device arrays back with
Warp's `array.numpy()` (which makes the device's null stream current for a moment: a launch of
the renderers meanwhile goes to the null stream). Checks: no error, every image equal to the
same scene rendered alone, Warp's default device and current stream unchanged. Then: raywarp
never changes the process-wide stream or device itself, and Warp's module loads are safe
against a concurrent change of the module (`module_lock`).
"""

from __future__ import annotations

import ast
import concurrent.futures
import gc
import pathlib
import threading
import time
import weakref
from typing import cast

import numpy as np
import pytest
import warp as wp
from conftest import recipe, required, whole_table

import raywarp as rw
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.shade import gi as gi_lib
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.trace.context import FrameContext

DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]
ROUNDS = 2
SIZE = (24, 24)
LIGHTS = {
  "sun": rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=3.0, angle=0.0),
  "sphere": rw.Light.point((0.4, -0.3, 1.2), power=40.0, radius=0.05),
}
# (light, base color, GI): one scene per thread.
SCENES = (("sun", (0.8, 0.2, 0.1), False), ("sphere", (0.1, 0.7, 0.2), False), ("sun", (0.2, 0.3, 0.9), True))


def _scene(light: str, color: tuple[float, float, float]) -> rw.Scene:
  """A floor and a box of `color` under `light`."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.6, 0.6, 0.6), specular_roughness=0.8))
  box = b.add_material(rw.Material(base_color=color, specular_roughness=0.3))
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]], material=floor)
  b.add_instance("box", size=(0.2, 0.2, 0.2), xform=np.c_[np.eye(3), [0.0, 0.0, 0.2]], material=box)
  b.add_light(LIGHTS[light])
  return b.build()


def _render(device: str, light: str, color: tuple[float, float, float], gi: bool) -> np.ndarray:
  """Upload, build, render (one frame), read and close, all on the calling thread."""
  settings = recipe("fast").replace(**{"gi.enabled": gi})
  scene = rw.upload(_scene(light, color), device)
  with rw.Renderer(scene, nworld=2, cameras=[rw.CameraSpec(fovy=50.0)], resolution=SIZE, settings=settings) as r:
    r.camera_pose.assign(np.array([[[0.0, -1.2, 0.8, 0.42, 0.0, 0.0, 0.91]]] * 2, np.float32))
    r.render()
    return host_copy(required(r.outputs.rgb)).copy()


@pytest.mark.parametrize("device", DEVICES)
def test_renderers_on_several_threads_render_as_alone(device: str) -> None:
  dev = wp.get_device(device)
  want = [_render(device, *s) for s in SCENES]
  default, stream = wp.get_device(), dev.stream if dev.is_cuda else None
  stop = threading.Event()
  probe = wp.array(np.arange(4096).astype(np.float32), device=dev)

  def caller_reads() -> None:
    """The caller's own host reads (Warp's numpy(): a moment on the null stream each)."""
    while not stop.is_set():
      assert probe.numpy()[7] == 7.0

  reader = threading.Thread(target=caller_reads)
  reader.start()
  try:
    with concurrent.futures.ThreadPoolExecutor(len(SCENES)) as pool:
      for _ in range(ROUNDS):
        got = [f.result() for f in [pool.submit(_render, device, *s) for s in SCENES]]
        for image, ref in zip(got, want, strict=True):
          np.testing.assert_array_equal(image, ref)
  finally:
    stop.set()
    reader.join()
  assert wp.get_device() == default
  if stream is not None:
    assert dev.stream is stream


@pytest.mark.parametrize("device", DEVICES)
def test_concurrent_warms_of_different_scenes(device: str) -> None:
  settings = recipe("fast")
  futures = [
    rw.Renderer.warm(
      rw.upload(_scene(light, color), device),
      nworld=2,
      cameras=[rw.CameraSpec(fovy=50.0)],
      resolution=SIZE,
      settings=settings,
    )
    for light, color, _gi in SCENES
  ]
  for f in futures:
    f.result()  # raises what a warm raised


# The process-wide state raywarp leaves alone: (call, the files allowed to make it, why).
_PROCESS_WIDE = {
  "ScopedStream": {"core/trace/optix/graph.py"},  # a stream the caller passes
  # The viewer application's own thread; a graph capture runs the caller's simulator step on the
  # capture's device (`core/pipeline/graph.py`). The recorder makes its device current for the CUDA
  # IPC calls that need its context (scoped: restored after); the encoder worker is its own process
  # and the CLI an application.
  "ScopedDevice": {
    "viewer/session.py",
    "viewer/scenes.py",
    "core/pipeline/graph.py",
    "video/recorder.py",
    "video/worker.py",
    "cli/record.py",
  },
  "set_stream": set(),
  "set_device": set(),
}


def test_raywarp_leaves_the_current_stream_and_device_alone() -> None:
  # `array.numpy()` swaps the process-wide current stream: host reads go through `host_copy`.
  root = pathlib.Path(rw.__file__).parent
  found: list[str] = []
  for path in root.rglob("*.py"):
    rel = path.relative_to(root).as_posix()
    for node in ast.walk(ast.parse(path.read_text())):
      if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        continue
      name = node.func.attr
      if name == "numpy" and not node.args and rel != "core/pipeline/platform/device.py":
        receiver = ast.unparse(node.func.value)
        if "device='cpu'" not in receiver:  # `ast.unparse` quotes with '
          found.append(f"{rel}:{node.lineno} {receiver}.numpy()")
      elif name in _PROCESS_WIDE and rel not in _PROCESS_WIDE[name]:
        found.append(f"{rel}:{node.lineno} {name}")
  assert not found, found


def test_module_loads_wait_for_a_concurrent_change_of_the_module() -> None:
  # A kernel factory adds a kernel to a module while another thread loads (compiles) that module:
  # Warp's own load then reads hashes the change cleared (FileNotFoundError / KeyError,
  # `module_lock`). The kernels are new to Warp's cache (`salt`): every load compiles.
  salt = float(time.time_ns() % 1_000_003)
  module = wp.get_module(f"{__name__}.factory_{int(salt)}")

  def add(k: int) -> None:
    step = salt + k

    def f(x: wp.array[float]) -> None:
      i = wp.tid()
      x[i] = x[i] + step

    wp.Kernel(f, key=f"added_{k}", module=module)

  add(0)
  errors: list[BaseException] = []
  done = threading.Event()

  def loader() -> None:
    try:
      while not done.is_set():
        module.load(wp.get_device("cpu"))
    except BaseException as e:  # noqa: BLE001  # reported by the assertion below
      errors.append(e)

  t = threading.Thread(target=loader)
  t.start()
  try:
    for k in range(1, 4):
      time.sleep(0.1)  # the loader is compiling
      add(k)
  finally:
    done.set()
    t.join()
  assert not errors, errors
  assert module.load(wp.get_device("cpu")) is not None


def _live_warp_objects() -> dict[int, object]:
  """The Warp arrays and textures alive (tracked by the collector), by id."""
  return {id(o): o for o in gc.get_objects() if isinstance(o, (wp.array, wp.Texture))}


def _render_relit() -> None:
  """The GI scene with per-world base colours, one per world: a relit GI volume (`gi.relight`)."""
  scene = rw.upload(_scene("sun", (0.8, 0.2, 0.1)), "cpu")
  settings = recipe("fast").replace(**{"gi.enabled": True})
  per_world = rw.PerWorld(material=["base_color"])
  with rw.Renderer(
    scene, nworld=4, cameras=[rw.CameraSpec(fovy=50.0)], resolution=SIZE, settings=settings, per_world=per_world
  ) as r:
    r.camera_pose.assign(np.array([[[0.0, -1.2, 0.8, 0.42, 0.0, 0.0, 0.91]]] * 4, np.float32))
    table = whole_table(r.per_world["material.base_color"])
    colors = host_copy(table)
    colors[..., :3] = np.random.default_rng(0).uniform(0.1, 1.0, colors[..., :3].shape)
    table.assign(colors)
    r.render()
    r.render()


@pytest.mark.parametrize("render", ["plain", "relit"])
def test_a_closed_renderer_leaves_no_warp_object_without_a_collection(render: str) -> None:
  # A scene switch: everything the scene and its renderer made is freed by reference counting
  # when they are dropped (no reference cycle, no cache that outlives them), so a data-generation
  # loop needs no `gc.collect()` between scenes and its device memory does not grow. "relit": a GI
  # volume relit per world, whose relight inputs the bake structures keep.
  def scenes() -> None:
    if render == "relit":
      _render_relit()
      return
    for scene in SCENES[:2]:
      _render("cpu", *scene)

  scenes()  # what the first renderers of the process keep (kernels, tables)
  gc.collect()
  before = set(_live_warp_objects())
  gc.disable()
  try:
    scenes()
    left = [type(o).__name__ for k, o in _live_warp_objects().items() if k not in before]
  finally:
    gc.enable()
  assert not left, left


def test_probe_lights_read_the_grid_current_when_called(monkeypatch) -> None:
  """The probes' light factory (`Baked._current_lights`, kept by the probe inputs a warm makes)
  reads the light grid when the probe rays run, not when it was made: a grid rebuilt between the
  warm and the frame is the one the relight uses (a stale grid wrote a bake under another key).
  It holds the owner weakly: the inputs `Baked` keeps make no reference cycle, and a released
  owner raises instead of lighting with released tables."""
  from raywarp.core.pipeline import baked as baked_lib  # the module under test

  class Static:
    """`StaticLights` stand-in: `of` returns what it lights."""

    def of(self, lit: object) -> object:
      return lit

  monkeypatch.setattr(baked_lib.frame_structs, "lights_struct", lambda now, grid: ("lights", now, grid))
  # Sentinels: the stand-ins only pass them through.
  warm_grid, frame_grid = cast(light_grid.Grid, object()), cast(light_grid.Grid, object())
  ctx = cast(FrameContext[None], object())
  owner = object.__new__(baked_lib.Baked)
  owner.grid = warm_grid

  def probe_lights(ctx: FrameContext[None]) -> gi_lib.StaticLights:  # `Baked._probe_lights`
    return cast(gi_lib.StaticLights, Static())

  owner._probe_lights = probe_lights  # the unit under test
  lights = owner._current_lights(ctx)
  owner.grid = frame_grid
  assert lights() == ("lights", ctx, frame_grid)
  gc.disable()
  try:
    ref = weakref.ref(owner)
    del owner
    assert ref() is None  # freed by reference counting: the factory holds it weakly
  finally:
    gc.enable()
  with pytest.raises(RuntimeError, match="released"):
    lights()


def test_a_software_warm_loads_every_output_view_s_modules(monkeypatch: pytest.MonkeyPatch) -> None:
  """On the software traversal each output set is its own megakernel module (its output word):
  `Renderer.warm(nearby=..., nearby_outputs=...)` plans the sets' first-frame Warp modules first
  and loads them, so a renderer of each set then finds every module of its first frame loaded."""
  from raywarp.core.pipeline import warmup

  scene = rw.upload(_scene("sun", (0.8, 0.2, 0.1)), "cpu")
  settings, cams = recipe("fast"), [rw.CameraSpec(fovy=50.0)]
  sets = [rw.Outputs.of("rgb", "albedo"), rw.Outputs.of("depth")]
  config = {"nworld": 1, "cameras": cams, "resolution": SIZE}
  plan = rw.Renderer.warm(scene, **config, settings=settings, nearby=6, nearby_outputs=sets).result().nearby
  assert plan is not None and plan.kind == "warp"
  assert [c for c, _, _ in plan.modules[:2]] == ["outputs='albedo,rgb'", "outputs='depth'"]
  cold: list[str] = []
  load = warmup._load_module  # the loads, recorded: a module without a loaded exec compiles

  def recorded(device: wp.DeviceLike, module: wp.Module, block_dim: int | None) -> None:
    if not module.execs:
      cold.append(module.name)
    load(device, module, block_dim)

  monkeypatch.setattr(warmup, "_load_module", recorded)
  for o in sets:
    with rw.Renderer(scene, **config, settings=settings.replace(outputs=o)) as r:
      r.wait_compiles()
  assert not cold


def test_a_gi_build_with_the_settings_gi_off_lists_no_relight_kernel() -> None:
  """An OptiX tier build compiles GI in whatever the settings (`Features.gi`); with GI off in the
  settings there is no relight build (`FrameContext.relight` None): a renderer of per-world light
  colours (relit inputs) listed the relight kernel and raised at construction (Isaac Lab on RTX).
  Here the tier build's GI is set on a CPU renderer's context."""
  import dataclasses

  from raywarp.core.pipeline import warmup

  scene = rw.upload(_scene("sun", (0.8, 0.2, 0.1)), "cpu")
  settings = recipe("fast").replace(**{"gi.enabled": False})
  per_world = rw.PerWorld(light=["color"])
  with rw.Renderer(
    scene, nworld=2, cameras=[rw.CameraSpec(fovy=50.0)], resolution=SIZE, settings=settings, per_world=per_world
  ) as r:
    ctx = r.context()
    assert ctx.relight is None
    tier = dataclasses.replace(ctx, features=dataclasses.replace(ctx.features, gi=True))
    warmup.first_frame(tier, r._backend)
