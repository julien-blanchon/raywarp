"""Deformables never occlude a shared bake (the GI probes and their placement, the light grid) on
either backend: the software bakes hide their parts (`baked.Baked._hide_flex`), the OptiX bakes'
rays leave out their identity instances' visibility-mask bit (`trace.BAKE_RAY_MASK`). GPU-free:
the instance records through the OptiX stand-in, and the programs' generated source."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
import warp as wp
from conftest import make_scene, no_default_device
from fixtures import optix_fake
from fixtures.optix_codegen import preset_features, scene_flags

import raywarp as rw
from raywarp._base.records import record_field
from raywarp.core.trace.optix import accel
from raywarp.core.trace.optix.device import intrinsics as optix_device
from raywarp.core.trace.optix.device import programs
from raywarp.core.trace.optix.device import trace as rt_trace

# A cloth over a floor and a moving sphere (a per-world curve GAS beside the deformable's).
_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="box" size="1.2 1.2 0.05" pos="0 0 -0.05"/>
      <body pos="0.5 0.4 0.5"><freejoint/><geom type="sphere" size="0.15"/></body>
      <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.2 0.2 0.2" pos="-0.3 0 0.7" radius="0.02" dim="2"
        rgba="0.2 0.7 0.3 1"><edge equality="true"/><contact contype="0" conaffinity="0"/></flexcomp>
      <camera name="cam" pos="0 -2.4 1.6" xyaxes="1 0 0 0 0.6 0.8"/>
    </worldbody>
  </mujoco>
"""


def test_the_bake_mask_leaves_only_the_deformables_out():
  """The bakes' rays trace every instance kind but the deformables; the render's trace all."""
  every = rt_trace.WORLD_BITS | rt_trace.SHARED_BITS | rt_trace.GAS_BITS
  assert rt_trace.BAKE_RAY_MASK & every == every
  assert not rt_trace.BAKE_RAY_MASK & rt_trace.FLEX_BITS
  assert rt_trace.FLEX_BITS & rt_trace.RAY_MASK_CULL == 0
  assert len({rt_trace.WORLD_BITS, rt_trace.SHARED_BITS, rt_trace.GAS_BITS, rt_trace.FLEX_BITS, rt_trace.RAY_MASK_CULL}) == 5


@pytest.mark.usefixtures("_cpu_device")
def test_deformable_instances_carry_their_own_bit(monkeypatch):
  """The deformable parts' identity instances are on `FLEX_BITS`, the curve GAS's on `GAS_BITS`."""
  optix_fake.install(monkeypatch)
  s = make_scene(_XML, nworld=2, cam_res=(8, 8), performance=rw.settings.Performance(traversal="software"))
  with no_default_device():
    scene = accel.OptixScene(s.r.scene, 2, s.r.layout, device=wp.get_device("cpu"), mesh_used=None, mesh_frame=None)
    scene.write(s.r.debug.frame(s.d), visibility=False)
    scene.build(optix_fake.STREAM)
  rows = scene.worlds.instances.numpy()
  ids, masks = record_field(rows, "instance_id"), record_field(rows, "visibility_mask")
  flex = ids >= rt_trace.FLEX_INSTANCE
  assert flex.sum() == 1  # the cloth's one surface part
  assert (masks[flex] == rt_trace.FLEX_BITS).all()
  gas_rows = [slot.index for slot in scene.worlds.slots if slot.instance_id < rt_trace.FLEX_INSTANCE]
  assert gas_rows  # the sphere's curve GAS
  assert (masks[gas_rows] == rt_trace.GAS_BITS).all()


@pytest.fixture
def _cpu_device():
  """The scene on the CPU device (the stand-in's stream is the CPU's)."""
  with wp.ScopedDevice("cpu"):
    yield


@pytest.fixture
def _headers(tmp_path, monkeypatch):
  """`device.build_options` only checks that the SDK header exists: codegen never reads it."""
  (tmp_path / "optix_device.h").write_text("")
  monkeypatch.setattr(optix_device, "INCLUDE_DIR", tmp_path)


def _spec_masks(monkeypatch: pytest.MonkeyPatch, build: Callable[[], object]) -> set[int]:
  """The visibility masks of every query `build()` defines: each `TraceSpec`'s rays, culled and
  with culling switched off at run time (`TraceSpec.unculled`; every OptiX trace call passes one
  of the two, `trace/optix/device/trace.py`)."""
  specs: list[rt_trace.TraceSpec] = []
  with monkeypatch.context() as m:
    for name in ("make_trace", "make_shadow"):
      original = getattr(rt_trace, name)

      def spy(spec: rt_trace.TraceSpec, _original: Callable[[rt_trace.TraceSpec], object] = original) -> object:
        specs.append(spec)
        return _original(spec)

      m.setattr(rt_trace, name, spy)
    build()
  return {int(m) for spec in specs for m in (spec.ray_mask, spec.unculled.ray_mask)}


@pytest.mark.usefixtures("_headers")
def test_bake_programs_trace_without_the_deformables(monkeypatch):
  """Every query of the GI programs and of the light grid's visibility program leaves the
  deformables' bit out, culled or not; the render's queries keep it."""
  f = dataclasses.replace(preset_features(rw.RenderSettings()), gi=True, culling=True)
  flags = programs.program_flags(f, scene_flags())
  bake = {rt_trace.BAKE_RAY_MASK & ~rt_trace.RAY_MASK_CULL, rt_trace.BAKE_RAY_MASK | rt_trace.RAY_MASK_CULL}
  with wp.ScopedDevice("cpu"):
    gi = _spec_masks(monkeypatch, lambda: programs.build_gi(f, flags))
    render = _spec_masks(monkeypatch, lambda: programs.build(f, flags))
  assert gi and gi <= bake
  assert render & bake  # the light grid's visibility program
  assert any(m & rt_trace.FLEX_BITS for m in render)  # the camera rays'


@pytest.mark.usefixtures("_headers")
def test_gi_programs_read_none_of_the_fields_their_key_resets():
  """`programs.gi_features` resets fields (`GI_UNREAD`) the GI programs do not read: the GI module's
  generated CUDA source of a build with them set equals the source with them reset (over the GI
  launch parameters of the reset build, `programs.gi_layout`, as the host fills them), so the GI
  PTX keyed without them is the same code."""
  from warp._src import context as wcontext

  f = dataclasses.replace(preset_features(rw.RenderSettings()), gi=True, culling=True)
  rich = dataclasses.replace(f, budget=4, glossy=1, hit_light_cap=2, light_cap=3, motion=True, reflection_shadows=1)
  flags = programs.program_flags(rich, scene_flags())

  def source(g: object) -> str:
    """The CUDA source of the GI programs built literally for `g` (no reset)."""
    module = wp.Module(programs.MODULE_NAME, None)
    P = programs.params_lib.params_type(programs.gi_layout(g, flags))  # pyrefly: ignore[bad-argument-type]
    programs._gi_programs(module, P, g, flags)  # pyrefly: ignore[bad-argument-type]
    programs._hit_programs(module, P, programs.probe_lib.probe_features(g), flags)  # pyrefly: ignore[bad-argument-type]
    programs._module_options(module, g)  # pyrefly: ignore[bad-argument-type]
    options = module.resolve_options(wp.config, block_dim=256) | {"output_arch": 89}
    module.get_module_hash(256)
    text = wcontext.ModuleBuilder(module, options, hasher=module.hashers.get(256)).codegen("cuda")
    return text.replace(module.name, "M")

  with wp.ScopedDevice("cpu"):
    assert source(rich) == source(programs.gi_features(rich))
