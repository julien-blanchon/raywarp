"""raywarp as mjlab's RGB backend, through mjlab's own API (`--env mjlab`; skipped without mjlab).

The CPU checks call the entry point (`create_groups`) as mjlab's sensor context does; the CUDA
check runs a mjlab Scene + Simulation end to end and reports the adapter's device memory.
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

pytest.importorskip("mjlab")

from conftest import recipe, required

import raywarp as rw
from raywarp.integrations.mjlab import RaywarpCameraSensorCfg, create_groups

# A room with a red box, three cameras.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <geom type="box" size="0.2 0.2 0.2" pos="0 0 0.2" rgba="0.9 0.1 0.1 1"/>
    <geom type="sphere" size="0.1" pos="0.5 0 0.1" group="3"/>
    <camera name="a" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="b" pos="0 -2 1.2" xyaxes="1 0 0 0 0.5 0.86"/>
    <camera name="c" pos="2 0 1" xyaxes="0 1 0 -0.45 0 0.9"/>
  </worldbody>
</mujoco>
"""


def _sensor(cam: int, **cfg: object) -> SimpleNamespace:
  """A camera sensor as mjlab's sensor context passes it (its MuJoCo camera id and config)."""
  return SimpleNamespace(camera_idx=cam, cfg=RaywarpCameraSensorCfg(name=f"cam{cam}", **cfg))  # pyrefly: ignore[bad-argument-type, unexpected-keyword]  # mjlab not installed (Any): CameraSensorCfg's fields unseen


def _model(nworld: int = 2) -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  """(mjm, m, d) of `_XML` on the CPU device."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with wp.ScopedDevice("cpu"):
    return mjm, mjw.put_model(mjm), mjw.put_data(mjm, mjd, nworld=nworld)


def test_groups_share_one_scene_per_geom_groups_and_use_final_settings() -> None:
  """Sensors of one geom-group set share one scene; one Renderer per resolution / configuration,
  built with its final settings (the default with mjlab's flags, or `raywarp_settings`) and
  observation-only color bound to each camera's tensor."""
  mjm, m, d = _model()
  sensors = [
    _sensor(0, width=32, height=24, use_shadows=False),
    _sensor(1, width=32, height=24, use_shadows=False),
    _sensor(2, width=16, height=16, raywarp_settings=recipe("default")),
    _sensor(2, width=16, height=16, enabled_geom_groups=(3, 0)),
  ]
  groups = create_groups(mjm, m, d, sensors, wp.get_device("cpu"))
  assert [sorted(g.rgb) for g in groups] == [[0, 1], [2], [2]]
  assert groups[0].renderer.scene is groups[1].renderer.scene is not groups[2].renderer.scene
  first, explicit = groups[0].renderer.settings, groups[1].renderer.settings
  assert first.outputs.rgb == rw.settings.ColorOutput(channels=3) and first.outputs.kinds == {"rgb"}
  assert first == rw.RenderSettings().replace(**{"shadows.enabled": False, "surfaces.textures": True}).replace(
    outputs=first.outputs
  )
  assert explicit.replace(outputs=recipe("default").outputs) == recipe("default")
  for g, (w, h) in zip(groups, [(32, 24), (16, 16), (16, 16)], strict=True):
    out = g.renderer.outputs
    assert out.rgb is not None and out.rgb.dtype == wp.vec3ub
    for i, cam in enumerate(sorted(g.rgb)):
      rgb = g.rgb[cam]
      assert rgb.shape == (2, h, w, 3)
      assert rgb.data_ptr() == required(out.rgb.ptr) + i * h * w * 3  # written in place, world-major


def test_invalid_raywarp_fields_and_settings_raise() -> None:
  """A wrong `raywarp_*` type raises; so do settings contradicting mjlab's flags and a device not
  the model's."""
  mjm, m, d = _model()
  cpu = wp.get_device("cpu")
  with pytest.raises(TypeError, match=r"must be a rw\.PerWorld"):
    create_groups(mjm, m, d, [_sensor(0, raywarp_per_world=["table/*"])], cpu)
  with pytest.raises(TypeError, match=r"must be a rw\.RenderSettings"):
    create_groups(mjm, m, d, [_sensor(0, raywarp_settings={"aa.spp": 1})], cpu)
  with pytest.raises(ValueError, match="contradict use_shadows=False"):
    create_groups(mjm, m, d, [_sensor(0, raywarp_settings=recipe("default"), use_shadows=False)], cpu)
  with pytest.raises(ValueError, match="use_textures=False"):
    create_groups(mjm, m, d, [_sensor(0, raywarp_settings=recipe("default"), use_textures=False)], cpu)
  if wp.is_cuda_available():
    with pytest.raises(ValueError, match="is not the model's"):
      create_groups(mjm, m, d, [_sensor(0)], wp.get_device("cuda:0"))


def test_a_group_failing_to_build_closes_the_groups_built_before(monkeypatch: pytest.MonkeyPatch) -> None:
  """`create_groups` returns nothing when a later group fails: the Renderers of the groups built
  before it are closed (nothing else owns them; the garbage collector releases no OptiX build)."""
  import raywarp.integrations.mjlab.rgb as mjlab_rgb

  mjm, m, d = _model()
  real, built, closed = mjlab_rgb.MjlabRgbRenderer, [], []

  def second_fails(*args: object, **kwargs: object) -> object:
    if built:
      raise RuntimeError("the second group fails")
    built.append(real(*args, **kwargs))  # pyrefly: ignore[bad-argument-type]
    return built[-1]

  monkeypatch.setattr(mjlab_rgb, "MjlabRgbRenderer", second_fails)
  monkeypatch.setattr(rw.Renderer, "close", lambda r: closed.append(r))
  sensors = [_sensor(0, width=32, height=24), _sensor(1, width=16, height=16)]
  with pytest.raises(RuntimeError, match="second group"):
    create_groups(mjm, m, d, sensors, wp.get_device("cpu"))
  assert closed == [built[0].renderer]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="mjlab's simulation and Torch streams need CUDA")
def test_mjlab_scene_renders_rgb_through_its_sensors() -> None:
  """A mjlab Scene + Simulation with two raywarp cameras: mjlab builds one renderer, its sensors
  read the rendered RGB (the red box in the centre); prints the adapter's device memory."""
  import torch
  from mjlab.entity import EntityCfg
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim import Simulation, SimulationCfg

  nworld, size = 4, 128
  sensors = tuple(
    RaywarpCameraSensorCfg(name=f"cam_{c}", camera_name=f"room/{c}", width=size, height=size)  # pyrefly: ignore[unexpected-keyword]  # mjlab not installed (Any): CameraSensorCfg's fields unseen
    for c in "ab"
  )
  scene = Scene(
    SceneCfg(num_envs=nworld, entities={"room": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(_XML))}, sensors=sensors),
    "cuda:0",
  )
  model = scene.compile()
  sim = Simulation(nworld, SimulationCfg(), model, "cuda:0")
  torch.cuda.synchronize()
  before = torch.cuda.memory_allocated(), wp.get_mempool_used_mem_current("cuda:0")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  context = scene.sensor_context
  assert context is not None
  sim.set_sensor_context(context)
  sim.forward()
  context.render_rgb()
  ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"room/{c}") for c in "ab"]
  images = [context.get_rgb(i) for i in ids]
  torch.cuda.synchronize()
  after = torch.cuda.memory_allocated(), wp.get_mempool_used_mem_current("cuda:0")
  for rgb in images:
    assert rgb.shape == (nworld, size, size, 3) and rgb.dtype == torch.uint8
    centre = rgb[:, size // 2, size // 2].float().cpu().numpy()
    assert np.all(centre[:, 0] > centre[:, 1] + 30), centre  # the red box
  torch_mib, warp_mib = ((a - b) / 2**20 for a, b in zip(after, before, strict=True))
  rgb_mib = 2 * nworld * size * size * 3 / 2**20
  print(f"\nmjlab raywarp adapter memory: torch {torch_mib:.1f} MiB (rgb {rgb_mib:.1f}), warp pool {warp_mib:.1f} MiB")
  context.close()
