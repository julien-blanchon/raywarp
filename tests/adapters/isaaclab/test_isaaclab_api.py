"""Renderer-facing API fakes pinned to Isaac Lab 647e39c5a108e6b217014e4f57ab86efeef7cdaa.

The constructor, synchronous prepare_capture, camera calibration and reset signatures follow
source/isaaclab/isaaclab/renderers/base_renderer.py at that commit. This checks our adapter
boundary without claiming an installed Isaac Lab integration run.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import types
import warnings
from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pytest
import warp as wp
from conftest import REPO, UNLIT_FAST_CHANGES
from fixtures import isaaclab_fake as fk

from raywarp.integrations.isaaclab.session import CameraGroup, CameraSession

if TYPE_CHECKING:
  from torch import Tensor


@dataclasses.dataclass
class _MaterialBatch:
  channel: str
  material_paths: tuple[str, ...]
  values: Tensor
  input_names: tuple[str, ...] = ("diffuseColor",)


class _BaseRenderer:
  """The upstream constructor and synchronous capture-preparation contract."""

  def __init__(self, cfg: object, *, sim: object) -> None:
    self.cfg = cfg
    self._sim = sim
    self.base_initialized = True

  def prepare_capture(self, render_data: object, camera_data: object, frame: object) -> None:
    """Synchronous renderers need no asynchronous metadata snapshot."""


@dataclasses.dataclass
class _SimCfg:
  physics: object = dataclasses.field(default_factory=object)


@dataclasses.dataclass
class _Simulation:
  cfg: _SimCfg = dataclasses.field(default_factory=_SimCfg)
  device: str = "cpu"
  requires_newton_model: bool = False
  requires_usd_stage: bool = False

  @staticmethod
  def instance() -> None:
    raise AssertionError("renderer must use its owning sim, never the global singleton")


@dataclasses.dataclass
class _NewtonBackendCfg:
  physics_cfg: object
  device: str


@dataclasses.dataclass
class _Tensor:
  """A device-index tensor whose view conversion can be checked without importing torch."""

  array: wp.array[wp.int32, [int]]
  is_cuda: bool = False

  def to(self, *, dtype: type) -> Self:
    assert dtype is wp.int32
    return self

  def contiguous(self) -> Self:
    return self

  def reshape(self, *shape: int) -> Self:
    assert shape == (-1,)
    return self

  def tolist(self) -> list[int]:
    return self.array.numpy().tolist()


@pytest.fixture
def adapter_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
  """Load only our adapter against the pinned interface, without polluting later imports."""
  definitions: dict[str, dict[str, object]] = {
    "isaaclab": {},
    "isaaclab.renderers": {"BaseRenderer": _BaseRenderer, "RenderBufferKind": str, "RenderBufferSpec": object},
    "isaaclab.scene_data": {"REQUIRES_STAGE_AND_MODEL": {"newton_warp": (False, True)}, "SceneDataFormat": object},
    "isaaclab.sim": {"SimulationContext": _Simulation},
    "isaaclab.utils": {},
    "isaaclab.utils.version": {"has_kit": lambda: False},  # kit-less Isaac Lab
    "isaaclab_newton": {},
    "isaaclab_newton.physics": {"NewtonBackendCfg": _NewtonBackendCfg},
    "isaaclab_newton.renderers": {},
    "isaaclab_newton.renderers.segmentation": {"NewtonSegmentationMapper": object},
    "torch": {"Tensor": _Tensor, "int32": wp.int32},
  }
  with monkeypatch.context() as patch:
    import sys

    for name, values in definitions.items():
      module = ModuleType(name)
      module.__dict__.update(values)
      patch.setitem(sys.modules, name, module)
    source = REPO / "raywarp" / "integrations" / "isaaclab" / "renderer.py"
    spec = importlib.util.spec_from_file_location("_raywarp_isaac_api_contract", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module


def test_renderer_uses_owning_simulation(adapter_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
  """The current cfg.class_type(cfg, sim=owner) construction calls the base and keeps its owner."""
  sim = _Simulation()
  renderer = adapter_module.RaywarpRenderer(fk.Config(), sim=sim)
  assert renderer.base_initialized
  assert sim.requires_newton_model and sim.requires_usd_stage
  received: list[object] = []

  def simulation(owner: object, backend: object, config: object) -> object:
    # Isaac's Newton backend is asked for the owner's physics, with this renderer's config.
    assert backend.physics_cfg is sim.cfg.physics and config is renderer.cfg  # pyrefly: ignore[missing-attribute]
    received.append(owner)
    return owner

  monkeypatch.setattr(adapter_module, "IsaacSimulation", simulation)
  monkeypatch.setattr(renderer.cameras, "initialize", received.append)
  renderer.initialize()
  assert received == [sim, sim]
  # The inherited hook deliberately does nothing: captures are synchronous.
  assert renderer.prepare_capture(object(), object(), object()) is None
  renderer.close()
  renderer.close()


def test_renderer_batch_and_device_reset_forwarding(adapter_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
  """Batch dispatch keeps session identity; CPU tensor reset ids take the host path (a list; CUDA
  tensors become device int32 views on the consumer stream: CUDA only)."""
  renderer = adapter_module.RaywarpRenderer(fk.Config(), sim=_Simulation())
  sessions = [object.__new__(CameraSession), object.__new__(CameraSession)]
  # The device whose stream Torch's consumers wait for (CPU: none), through the cameras' group.
  group = types.SimpleNamespace(renderer=types.SimpleNamespace(device=wp.get_device("cpu")))
  for session in sessions:
    session.group = cast("CameraGroup", group)
  batches: list[list[CameraSession]] = []
  monkeypatch.setattr(renderer.cameras, "render_batch", lambda batch: batches.append(list(batch)))
  renderer.render_batch([])
  renderer.render_batch(sessions)
  assert batches == [[], sessions]  # one batch, in Isaac's order
  indices = wp.array([1], dtype=wp.int32, device="cpu")
  resets: list[object] = []

  def from_torch(tensor: _Tensor, *, dtype: type) -> wp.array[wp.int32, [int]]:
    assert dtype is wp.int32
    return tensor.array

  def reset(session: CameraSession, env_ids: object) -> None:
    assert session is sessions[0]
    resets.append(env_ids)

  monkeypatch.setattr(wp, "from_torch", from_torch)
  monkeypatch.setattr(renderer.cameras, "reset", reset)
  renderer.reset(sessions[0], _Tensor(indices))
  assert resets == [[1]]
  renderer.reset(sessions[0], [0])
  assert resets[1] == [0]
  renderer.close()


def test_body_poses_wait_for_torch_after_the_provider(adapter_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
  """`update_transforms` orders the Warp stream after Torch once the provider has published the
  poses (its conversion may queue Torch work) and before they are copied into the render state."""
  events: list[str] = []
  transforms = wp.zeros(2, dtype=wp.transformf, device="cpu")

  class Provider:
    def get_transforms(self, out: types.SimpleNamespace, *, mapping: object, count: int) -> bool:
      assert count == 2
      events.append("provider")
      out.transforms = transforms
      return True

  class Backend:
    model = types.SimpleNamespace(body_label=["a", "b"], body_count=2, device="cpu")

  class Owner(_Simulation):
    def get_or_create_backend(self, cfg: object) -> Backend:
      return Backend()

    def get_scene_data_provider(self) -> Provider:
      return Provider()

  provider_create = []
  Provider.create_mapping = lambda self, labels: provider_create.append(labels)
  monkeypatch.setattr(
    adapter_module, "SceneDataFormat", types.SimpleNamespace(Transform=lambda: types.SimpleNamespace(transforms=None))
  )
  # White-box seam: the stream wait is only observable as its call order (CPU has no streams).
  monkeypatch.setattr(adapter_module, "consume_torch", lambda device: events.append(f"wait {device}"))
  sim = adapter_module.IsaacSimulation(Owner(), object(), fk.Config())
  assert provider_create == [["a", "b"]]
  assert sim.body_poses() is transforms
  assert events == ["provider", "wait cpu"]  # the copy into the render state follows the wait


@pytest.mark.skipif(not wp.is_cuda_available(), reason="real Torch CUDA material interop")
@pytest.mark.parametrize("channel", ["color", "specular"])
def test_real_torch_material_values_on_another_stream(
  request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
  """Live Torch producers precede material writes into the shared prepared table."""
  import sys

  import numpy as np

  torch = pytest.importorskip("torch")
  if not torch.cuda.is_available():
    pytest.skip("Torch has no CUDA support")
  adapter = request.getfixturevalue("adapter_module")
  assert isinstance(adapter, ModuleType)
  with monkeypatch.context() as interop:
    interop.setitem(sys.modules, "torch", torch)
    interop.setattr(adapter, "torch", torch)
    renderer = adapter.RaywarpRenderer(fk.Config(settings=UNLIT_FAST_CHANGES), sim=_Simulation())
    with wp.ScopedDevice("cuda:0"):
      model, state = fk.boxes(2, device="cuda:0")
      renderer.cameras.initialize(fk.Simulation(model, state, materials=["/Looks/Paint"] * model.shape_count))
      values = torch.zeros((1, 3) if channel == "color" else (1,), dtype=torch.float32, device="cuda:0")
      name = "diffuseColor" if channel == "color" else "specular_level"
      writer = renderer.visual_material_writer((_MaterialBatch(channel, ("/Looks/Paint",), values, (name,)),))
      writer()
      session = renderer.cameras.create_render_data(fk.RenderSpec(fk.CameraCfg(8, 8, ["albedo"]), 2))
      table = session.renderer.per_world["material.base_color" if channel == "color" else "material.specular_weight"]
      pointer = table.ptr
      producer = torch.cuda.Stream(device="cuda:0")
      producer.wait_stream(torch.cuda.current_stream("cuda:0"))
      consumer = wp.Stream("cuda:0")
      consumer.wait_stream(wp.get_stream("cuda:0"))
      with torch.cuda.stream(producer), wp.ScopedStream(consumer):
        torch.cuda._sleep(1_000_000)  # Delay the producer enough to expose a missing consumer wait.
        if channel == "color":
          values[:, 2] = 0.75
        else:
          values[:] = 0.75
        writer()
      wp.synchronize_stream(consumer)
      actual = table.numpy()[..., 2] if channel == "color" else table.numpy()
      np.testing.assert_allclose(actual, 0.75 if channel == "color" else 1.5, atol=1e-7)
      assert table.ptr == pointer
      renderer.close()


def _isaac_simulation(adapter_module: ModuleType, model: types.SimpleNamespace, stage: object, plan: object) -> object:
  """The adapter's `IsaacSimulation` over a fake owner holding `model`, `stage` and clone `plan`."""

  class Provider:
    def create_mapping(self, labels: list[str]) -> list[str]:
      return labels

  class Owner(_Simulation):
    def get_or_create_backend(self, cfg: object) -> types.SimpleNamespace:
      return types.SimpleNamespace(model=model)

    def get_scene_data_provider(self) -> Provider:
      return Provider()

    def get_clone_plan(self) -> object:
      return plan

  owner = Owner()
  owner.stage = stage  # pyrefly: ignore[missing-attribute]
  # The module is the test's own (`adapter_module`): its stand-in for Isaac's pose format.
  setattr(adapter_module, "SceneDataFormat", types.SimpleNamespace(Transform=types.SimpleNamespace))  # noqa: B010
  return adapter_module.IsaacSimulation(owner, object(), fk.Config())


def test_segmentation_reads_isaacs_mapper(adapter_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
  """`segmentation(kind)` builds Isaac's Newton mapper once (model, stage, renderer config, clone
  plan), maps `kind` without forcing a rebuild, and returns its shape ids as int32 and its info."""
  calls: list[tuple[object, ...]] = []
  info = {"idToLabels": {"0": "BACKGROUND"}}

  class Mapper:
    def __init__(self, model: object, stage: object, cfg: object, plan: object) -> None:
      calls.append(("init", model, stage, cfg, plan))

    def build_mapping(self, kind: str, force_rebuild: bool) -> None:  # noqa: FBT001  # Isaac's positional flag
      calls.append(("build", kind, force_rebuild))

    def get_mapping(self, kind: str, force_rebuild: bool) -> types.SimpleNamespace:  # noqa: FBT001
      calls.append(("get", kind, force_rebuild))
      return types.SimpleNamespace(shape_to_id=wp.array([3, -1, 7], dtype=wp.int64, device="cpu"), info=info)

  monkeypatch.setattr(adapter_module, "NewtonSegmentationMapper", Mapper)
  model = types.SimpleNamespace(body_label=[], device="cpu")
  stage, plan = object(), object()
  sim = _isaac_simulation(adapter_module, model, stage, plan)
  for kind in ("semantic", "instance"):
    ids, got = sim.segmentation(kind)  # pyrefly: ignore[missing-attribute]
    assert ids.dtype == np.int32 and ids.tolist() == [3, -1, 7] and got is info
  init = calls[0]
  assert init[:3] == ("init", model, stage) and init[4] is plan and isinstance(init[3], fk.Config)
  assert calls[1:] == [
    ("build", "semantic", False),
    ("get", "semantic", False),
    ("build", "instance", False),
    ("get", "instance", False),
  ]


def test_shape_materials_read_the_isaaclab_namespace(adapter_module: ModuleType) -> None:
  """`shape_materials()` is the model's `isaaclab` namespace's per-shape visual material paths;
  without the namespace, none (warned once: material writes do not reach the render)."""
  paths = ["/World/Looks/A", ""]
  tagged = types.SimpleNamespace(body_label=[], device="cpu", isaaclab=types.SimpleNamespace(visual_material_path=paths))
  assert _isaac_simulation(adapter_module, tagged, None, None).shape_materials() == paths  # pyrefly: ignore[missing-attribute]
  bare = _isaac_simulation(adapter_module, types.SimpleNamespace(body_label=[], device="cpu"), None, None)
  with pytest.warns(UserWarning, match="visual_material_path"):
    assert bare.shape_materials() == []  # pyrefly: ignore[missing-attribute]
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert bare.shape_materials() == []  # pyrefly: ignore[missing-attribute]
