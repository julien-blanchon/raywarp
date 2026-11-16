"""The OptiX backend's host logic on the CPU, through a stand-in for driver OptiX (`optix_fake.py`).

The acceleration structures' Warp kernels (cell bounds, instance records, curves, primitive
bounds) run for real; the OptiX calls are recorded. Regression tests of B19 (launch chunking),
B20 (cell bounds reset on rebuild frames, also in CUDA graphs), G1 (a diverged world stays out of
the cell), B21 (shared scenery visibility per
frame) and B22 (instance ids within OptiX's 28 bits). The OptiX objects are built and written
under `no_default_device`: every allocation, launch and stream names its device. Rendering through
OptiX: `test_optix.py`.
"""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import itertools
import logging
import re
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import warp as wp
from conftest import RECIPE_NAMES, internal_module, make_scene, no_default_device, recipe
from fixtures import optix_fake
from fixtures.optix_codegen import PARITY_XML, cuda_source, preset_features, scene_flags

import raywarp as rw
from raywarp._base.records import record_field
from raywarp._base.types import TEX_LAYOUT_IDENTITY, uv_xform
from raywarp._base.warp_structs import struct_mirror, warp_struct
from raywarp.core.kernels.abi import output_layout as ol
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import PLANES_NONE, Features
from raywarp.core.kernels.abi.output_layout import OutputLayout
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import DeviceCaps, device_caps
from raywarp.core.pipeline.textures import formats
from raywarp.core.shade.materials import texture
from raywarp.core.trace import omm_sources
from raywarp.core.trace.context import FrameContext
from raywarp.core.trace.optix import accel, api, context, gas, group, launch, pipeline
from raywarp.core.trace.optix.context import DeviceContext, OptixObject
from raywarp.core.trace.optix.device import intrinsics as optix_device
from raywarp.core.trace.optix.device import params, programs
from raywarp.core.trace.optix.device import trace as rt_trace
from raywarp.core.trace.optix.device.programs import SceneFlags
from raywarp.core.trace.optix.omm import bake as omm_bake
from raywarp.core.trace.optix.omm import kernels as omm_kernels
from raywarp.core.trace.optix.omm import micromaps as omm
from raywarp.core.trace.rays.primary import LENS_MODELS
from raywarp.scene.model.types import AlphaMode, InstanceId, Slot, TextureRef

if TYPE_CHECKING:
  import feature_scenes as fs  # development repository only (`internal_module`)

  from raywarp.core.trace.optix.context import DeviceContext
from typing import Final

import mujoco_warp as mjw
from conftest import required

from raywarp._base.dims import Floats
from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.rays import OptixFields, OptixGeometry
from raywarp.core.kernels.generic import concrete
from raywarp.core.trace.accel import reach
from raywarp.core.trace.optix import instances
from raywarp.core.trace.optix.device import hits, trace

_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom name="static_box" type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
      <geom type="{scenery_type}" pos="0 0 0.4" size="{scenery_size}"/>
      {hidden}
      <body pos="0.1 -0.5 0.35">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
  </mujoco>
"""


CPU = wp.get_device("cpu")


@pytest.fixture
def _cpu_device():
  """The MuJoCo scenes on the CPU device: the fake stream is the CPU's (on a CUDA machine the
  default device is the GPU, whose arrays the fake's CPU-stream launches crashed the worker on)."""
  with wp.ScopedDevice("cpu"):
    yield


def _setup(monkeypatch: pytest.MonkeyPatch, nworld: int = 2, rebuild_every: int = 0, hidden: int = 0):
  """(fake context, mjw scene, OptiX structures): a software Renderer's scene and layout."""
  _table, ctx = optix_fake.install(monkeypatch)
  geoms = "".join(f'<geom type="sphere" group="3" pos="0 0 {i}" size="0.01"/>' for i in range(hidden))
  xml = _XML.format(hidden=geoms, scenery_type="sphere", scenery_size="0.3")
  s = make_scene(xml, nworld=nworld, cam_res=(8, 8), performance=rw.settings.Performance(traversal="software"))
  with no_default_device():
    frames = s.r.scene.meshes.frames
    mesh_frame = None if frames is None else frames.astype(np.float32)  # as launch.py
    scene = accel.OptixScene(s.r.scene, nworld, s.r.layout, device=CPU, mesh_used=None, mesh_frame=mesh_frame)
  scene.rebuild_every = rebuild_every
  return ctx, s, scene


def _frame(s):
  """The renderer's frame of the current simulation state."""
  return s.r.debug.frame(s.d)


def _place_body(s, x: float) -> None:
  """Moves the free body of every world to x (forward kinematics included)."""
  import mujoco_warp as mjw

  qpos = np.tile(s.mjd.qpos, (s.d.nworld, 1))
  qpos[:, 0] = x
  s.d.qpos.assign(qpos.astype(np.float32))
  mjw.forward(s.m, s.d)


def _frame_step(scene, frame, capturing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
  """One frame: the Warp writes (captured in a graph or eager), then the OptiX builds."""
  with monkeypatch.context() as m, no_default_device():
    m.setattr(DeviceCaps, "capturing", lambda self: capturing)
    scene.write(frame, visibility=False)
    scene.build(optix_fake.STREAM)


def _cell_upper_x(scene) -> float:
  """The upper x of the cell the last frame was written with."""
  return float(scene.tables.cell.numpy()[rt_trace.CELL_UPPER][0])


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize("capturing", [False, True])
def test_cell_follows_the_current_frame_after_far_moves(monkeypatch, capturing):
  """BUG-012: a world's cell is its current geometry's, on refit frames too (eager or a replayed
  graph): after a move far away and back, the cell (hence every cell-space coordinate) is exactly
  the first frame's, so a reset renders identically."""
  _ctx, s, scene = _setup(monkeypatch, rebuild_every=0)
  _place_body(s, 0.1)
  frame = _frame(s)
  _frame_step(scene, frame, capturing, monkeypatch)  # frame 0: built
  first = scene.tables.cell.numpy().copy()
  instances = scene.worlds.instances.numpy().copy()
  for x in (1.0e2, 1.0e4):
    _place_body(s, x)
    _frame_step(scene, frame, capturing, monkeypatch)  # refit frames
    assert _cell_upper_x(scene) > 0.9 * x
  _place_body(s, 0.1)
  _frame_step(scene, frame, capturing, monkeypatch)
  np.testing.assert_array_equal(scene.tables.cell.numpy(), first)
  np.testing.assert_array_equal(scene.worlds.instances.numpy(), instances)


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize("value", [1e20, float("nan")])
def test_diverged_world_stays_out_of_the_cell(monkeypatch, value):
  """World 1's free box at 1e20 / NaN (a diverged simulation) is left out of world 1's cell (which
  stays finite) and hidden there, with a finite record inside its own cell; world 0's cell is
  unchanged."""
  _ctx, s, scene = _setup(monkeypatch, nworld=2)
  frame = _frame(s)
  _frame_step(scene, frame, False, monkeypatch)
  cell = scene.tables.cell.numpy().copy()
  pos = frame.instance_pos.numpy()
  box = int(np.flatnonzero(s.r.layout.per_world)[-1])  # the free box
  pos[1, box] = value
  frame.instance_pos.assign(pos)
  _frame_step(scene, frame, False, monkeypatch)
  after = scene.tables.cell.numpy()
  world0 = rt_trace.CELL_SIZE  # world 0's cell
  np.testing.assert_array_equal(after[:world0], cell[:world0])
  assert np.isfinite(after).all()
  ntri, k = len(scene.worlds.tri_ids), scene.worlds.tri_ids.index(InstanceId(box))
  records = scene.worlds.instances.numpy()[[k, ntri + k]]
  mask = record_field(records, "visibility_mask")
  assert mask[0] != 0 and mask[1] == 0
  hidden = np.stack([record_field(records, t)[1] for t in ("t0", "t1", "t2")])
  assert np.isfinite(hidden).all() and np.abs(hidden[:, 3]).max() < 10.0


@pytest.mark.usefixtures("_cpu_device")
def test_shared_visibility_follows_alpha_every_frame(monkeypatch):
  """B21: hiding a shared instance (alpha 0) after the first frame hides it in the shared IAS."""
  _ctx, s, scene = _setup(monkeypatch)
  box = s.mjm.geom("static_box").id
  group = scene.shared
  assert box in group.tri_ids
  rows = [group.tri_ids.index(InstanceId(box))]
  frame = _frame(s)
  _frame_step(scene, frame, False, monkeypatch)
  assert (record_field(group.instances.numpy(), "visibility_mask")[rows] != 0).all()
  rgba = frame.instance_color.numpy()
  rgba[:, box, 3] = 0.0
  frame.instance_color.assign(rgba)
  _frame_step(scene, frame, False, monkeypatch)
  assert (record_field(group.instances.numpy(), "visibility_mask")[rows] == 0).all()


def _bake_renderer(scene) -> tuple[launch.OptixRenderer, FrameContext]:
  """(an OptixRenderer's structure state over `scene`, without programs; the frame context fields
  its structure writes read)."""
  r = object.__new__(launch.OptixRenderer)
  r.scene, r.device = scene, CPU
  r.fresh = r._unbuilt = False
  r._apart = None
  ctx = SimpleNamespace(
    settings=SimpleNamespace(performance=SimpleNamespace(rebuild_every=0)),
    features=SimpleNamespace(visibility=False),
  )
  return r, cast("FrameContext", ctx)


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize("capturing", [False, True])
def test_gi_bake_structures_hide_the_probe_geometry_s_instances(monkeypatch, capturing):
  """The GI bake traces structures written with the probe geometry's colors: an instance they hide
  (alpha 0, a moving one: `baked._probe_geometry`) is out of the bake's IAS, as the software
  traversal skips it (only in the launch parameters, the colors hid nothing from an opaque build's
  rays: the probes saw a moving robot). Written once per frame and colors; the render's next
  frame writes its own colors again. Inside a capture the writes are recorded and the launch's
  cut rebuilds apart."""
  fake, s, scene = _setup(monkeypatch, nworld=2)
  r, ctx = _bake_renderer(scene)
  box = int(np.flatnonzero(s.r.layout.per_world)[-1])  # the free (moving) box
  rows = [scene.worlds.tri_ids.index(InstanceId(box)) + k * len(scene.worlds.tri_ids) for k in range(2)]
  frame = _frame(s)
  with no_default_device():
    r.refresh(ctx, frame)
  shown = lambda: record_field(scene.worlds.instances.numpy(), "visibility_mask")[rows]  # noqa: E731
  assert (shown() != 0).all()
  rgba = frame.instance_color.numpy().copy()
  rgba[:, box, 3] = 0.0
  colors = wp.array(rgba, dtype=wp.vec4, device=CPU)
  with monkeypatch.context() as m, no_default_device():
    m.setattr(DeviceCaps, "capturing", lambda self: capturing)
    builds = len(fake.builds)
    r._bake_structures(ctx, frame, colors)
    assert (shown() == 0).all() and not r.fresh
    assert r._unbuilt == capturing and (len(fake.builds) == builds) == capturing
    if not capturing:
      assert scene.stale and all(not b.update for b in fake.builds[builds:])  # rebuilt apart
    built = len(fake.builds)
    r._bake_structures(ctx, frame, colors)  # the next launch of the bake: nothing written again
    assert len(fake.builds) == built
  with no_default_device():
    r.begin_frame()
    r.refresh(ctx, frame)
  assert (shown() != 0).all() and r._apart is None and not scene.stale


@pytest.mark.usefixtures("_cpu_device")
def test_instance_ids_are_the_instances(monkeypatch):
  """B22: a triangle instance's OptiX id is its instance index, in every world (the world is the
  ray's payload): the ids stay within OptiX's 28 bits at any world count."""
  _ctx, s, scene = _setup(monkeypatch, nworld=3, hidden=40)
  ntri = len(scene.worlds.tri_ids)
  assert ntri == 1 and s.mjm.ngeom > 40  # the free box
  _frame_step(scene, _frame(s), False, monkeypatch)
  ids = record_field(scene.worlds.instances.numpy(), "instance_id")[: 3 * ntri]
  np.testing.assert_array_equal(ids, np.tile(np.asarray(scene.worlds.tri_ids), 3))
  shared = record_field(scene.shared.instances.numpy(), "instance_id")[: len(scene.shared.tri_ids)]
  np.testing.assert_array_equal(shared, np.asarray(scene.shared.tri_ids))


def _bare_pipeline() -> pipeline.Pipeline:
  """A Pipeline's launch state on the CPU, without OptiX objects (`store` / `trace` only)."""
  pipe = object.__new__(pipeline.Pipeline)
  pipe.device = CPU
  pipe.layout = params.ParamsLayout()
  pipe.params_size = params.size_of(params.params_type(pipe.layout))
  pipe.params = {}
  pipe._stored = {}
  pipe._captured = False
  pipe.pipelines = {None: optix_fake.pipeline_object()}
  pipe.sbts = {None: api.ShaderBindingTable()}
  return pipe


@pytest.mark.usefixtures("_cpu_device")
def test_launch_chunks_stay_under_optix_limit(monkeypatch):
  """B19: 4097 worlds x 2 cameras x 256^2 x spp 4 (over 2^31 launch indices) are split into launches
  OptiX accepts, of whole worlds, each reading launch parameters that start at its first world."""
  fake, _ctx = optix_fake.install(monkeypatch)
  pipe = _bare_pipeline()
  p = params.params_type(pipe.layout)()
  p.seed = wp.array([7], dtype=wp.int32, device=CPU)
  nworld, rays = 4097, 2 * 256 * 256 * 4
  with no_default_device():
    pipe.store(p, nworld, rays)
    pipe.trace(optix_fake.STREAM, nworld, rays)
  assert fake.launches == [1 << 30, 1 << 30, rays]
  mirror = type(struct_mirror(p))
  copies = [mirror.from_buffer_copy(row.tobytes()) for row in pipe.params[None, 3].numpy()]
  assert [c.offset for c in copies] == [0, 2048, 4096]
  assert [c.seed.data for c in copies] == [p.seed.ptr] * 3
  np.testing.assert_array_equal(p.seed.numpy(), [7])


class _SizedContext(optix_fake.FakeContext):
  """A fake context reporting odd byte sizes and recording the buffer sizes `accel_build` got."""

  sizes: tuple[int, int, int] = (13, 5, 1001)
  given: list[tuple[int, int]]

  def accel_sizes(self, options: api.AccelBuildOptions, inputs: object) -> api.AccelBufferSizes:
    """The temp, temp-update and output sizes of `sizes`."""
    temp, update, output = self.sizes
    return api.AccelBufferSizes(outputSizeInBytes=output, tempSizeInBytes=temp, tempUpdateSizeInBytes=update)

  def accel_build(self, stream: int, options: api.AccelBuildOptions, inputs: object, *args: object) -> int:
    """Records (temp size, output size) of the build."""
    self.given.append((int(args[1]), int(args[3])))  # pyrefly: ignore[bad-argument-type]
    return super().accel_build(stream, options, inputs, *args)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.usefixtures("_cpu_device")
def test_gas_buffers_are_words_above_the_int32_byte_limit(monkeypatch):
  """A GAS over > ~34M triangles needs > 2^31 bytes: the build buffers are uint64 words (a byte
  array of that size exceeds Warp's int32 shape), OptiX still gets the byte sizes it asked for."""
  optix_fake.install(monkeypatch)
  for nbytes in (2**31 + 1, 6 * 2**30 + 3):
    assert gas.buffer_words(nbytes) < 2**31 and 8 * gas.buffer_words(nbytes) >= nbytes
  assert gas.buffer_words(0) == 1 and gas.buffer_words(8) == 1 and gas.buffer_words(9) == 2
  ctx = _SizedContext()
  ctx.given = []
  with no_default_device():
    a = gas.Accel(ctx, api.BuildInput(), 0, device=CPU)  # pyrefly: ignore[bad-argument-type]
  assert a.temp.dtype is wp.uint64 and a.output.dtype is wp.uint64
  assert a.temp.capacity == 16 and a.output.capacity == 1008
  a.build(optix_fake.STREAM.cuda_stream)
  assert ctx.given == [(13, 1001)]


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize("primitive", ["sphere", "capsule", "cylinder", "ellipsoid", "box"])
def test_shared_scenery_is_its_own_group(monkeypatch, primitive):
  """One layout: shared instances are in the shared group, per-world ones in the worlds' cells,
  whatever their primitive type."""
  optix_fake.install(monkeypatch)
  xml = f"""<mujoco><worldbody>
    <geom name="shared" type="{primitive}" size="0.1 0.1 0.1"/>
    <body pos="1 0 0"><freejoint/><geom name="world" type="box" size="0.1 0.1 0.1"/></body>
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody></mujoco>"""
  s = make_scene(xml, nworld=2, cam_res=(8, 8), performance=rw.settings.Performance(traversal="software"))
  with no_default_device():
    scene = accel.OptixScene(s.r.scene, 2, s.r.layout, device=CPU, mesh_used=None)
  assert not scene.shared.empty and not scene.worlds.empty
  assert InstanceId(s.mjm.geom("world").id) in scene.worlds.tri_ids
  assert InstanceId(s.mjm.geom("shared").id) not in scene.worlds.tri_ids


@pytest.mark.usefixtures("_cpu_device")
def test_world_cells_are_independent(monkeypatch):
  """Moving one world's box changes that world's records only (each world's cell is its own)."""
  _ctx, s, scene = _setup(monkeypatch, nworld=3)
  _place_body(s, 1.5)
  frame = _frame(s)
  _frame_step(scene, frame, False, monkeypatch)
  ntri = len(scene.worlds.tri_ids)
  before = scene.worlds.instances.numpy().copy()
  cells_before = scene.tables.cell.numpy()[: 3 * trace.CELL_SIZE].reshape(3, trace.CELL_SIZE, 3).copy()
  pos = frame.instance_pos.numpy()
  pos[1, int(np.flatnonzero(s.r.layout.per_world)[-1])] += 5.0
  frame.instance_pos.assign(pos)
  _frame_step(scene, frame, False, monkeypatch)
  after = scene.worlds.instances.numpy()
  cells_after = scene.tables.cell.numpy()[: 3 * trace.CELL_SIZE].reshape(3, trace.CELL_SIZE, 3)
  for w in (0, 2):
    np.testing.assert_array_equal(after[w * ntri : (w + 1) * ntri], before[w * ntri : (w + 1) * ntri])
    np.testing.assert_array_equal(cells_after[w], cells_before[w])
  # World 1's cell follows its moved geometry (its only per-world box keeps its cell coordinates).
  assert not np.array_equal(cells_after[1, trace.CELL_CENTER], cells_before[1, trace.CELL_CENTER])


@pytest.mark.usefixtures("_cpu_device")
def test_scene_without_materials_has_no_cut_instances():
  """A scene without materials (empty cutoff table) built no OptiX structures: reshaping the empty
  table by rows raised (found by the L4 run of the two-sided test)."""
  b = rw.SceneBuilder()
  b.add_instance("box", size=(0.1, 0.1, 0.1))
  scene = rw.upload(b.build(), "cpu")
  assert not accel.cut_instances(scene, 1, cut_all=False).any()


_CUT_XML = """<mujoco><custom><numeric name="raywarp:alpha_cutoff:leaf" data="0.5"/></custom>
  <asset><material name="leaf"/><material name="plain"/></asset>
  <worldbody><geom type="box" size=".1 .1 .1" material="leaf"/><geom type="box" size=".1 .1 .1" pos="1 0 0" material="plain"/>
  <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody></mujoco>"""


@pytest.mark.usefixtures("_cpu_device")
def test_mujoco_live_material_write_keeps_the_alpha_test(monkeypatch):
  """MuJoCo's instance materials are live binding slots (`geom_matid`): a write moves a geom onto a
  cut-out material without a rebuild, so the build must not take the cut-out set from the
  initial ids (it did, after reading `geom_matid`: the moved geom lost its alpha test)."""
  from raywarp.core.trace.optix import launch

  optix_fake.install(monkeypatch)
  s = make_scene(_CUT_XML, cam_res=(8, 8), performance=rw.settings.Performance(traversal="software"))
  cut_all = launch._cut_all(s.r.context())
  assert cut_all
  with no_default_device():
    scene = accel.OptixScene(s.r.scene, 1, s.r.layout, device=CPU, mesh_used=None, alpha_cutout=True, cut_all=cut_all)
  s.m.geom_matid.assign(np.zeros(2, np.int32))  # the plain box now wears the leaf
  flags = scene.tables.flags.numpy()[:2]
  assert (flags & api.INSTANCE_FLAG_ENFORCE_ANYHIT).all()


@pytest.mark.usefixtures("_cpu_device")
def test_static_cut_out_materials_and_out_of_range_ids():
  """The static path: the cut-out instance is found from the tables (the binding's included), and
  a material id beyond the cutoff table cuts nothing (no IndexError: validation reports it)."""
  import mujoco
  import mujoco_warp as mjw

  mjm = mujoco.MjModel.from_xml_string(_CUT_XML)
  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm))
    assert accel.cut_instances(scene, 2, cut_all=False).tolist() == [True, False]
    scene.binding.slots(None)["instance.material"].assign(np.array([[0, 7]], np.int32))
    assert accel.cut_instances(scene, 2, cut_all=False).tolist() == [True, False]


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize(("nmajor", "minor"), [(0, 64), (3, 0)])
def test_no_launch_without_indices(monkeypatch, nmajor, minor):
  """No index, no launch: `chunks(0, m)` was one zero-width chunk, which `trace` launched."""
  fake, _ctx = optix_fake.install(monkeypatch)
  pipe = _bare_pipeline()
  assert pipeline.Pipeline.chunks(nmajor, minor) == []
  pipe.store(params.params_type(pipe.layout)(), nmajor, minor)
  pipe.trace(optix_fake.STREAM, nmajor, minor)
  assert fake.launches == [] and pipe.params == {}


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize(("curves", "custom"), [(False, False), (True, False), (False, True), (True, True)])
def test_primitive_type_flags_list_only_present_types(curves, custom):
  pco = pipeline.compile_options(has_curves=curves, has_custom=custom, max_registers=0, opacity_micromaps=False).pipeline
  expected = api.PRIMITIVE_TYPE_FLAGS_TRIANGLE
  expected |= api.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR if curves else 0
  expected |= api.PRIMITIVE_TYPE_FLAGS_CUSTOM if custom else 0
  assert pco.usesPrimitiveTypeFlags == expected


def test_hit_groups_order_and_instance_offsets():
  """The SBT's hit groups: the triangles' first, then per structure (per-world, shared) its curves'
  and one per custom type; an instance record points at its hit group's first record (each group
  `RAY_TYPES` records in a row)."""
  ellipsoid, cylinder = rt_trace.CUSTOM_TYPES
  groups = group.HitGroups(world_curves=True, world_types=(ellipsoid, cylinder), shared_curves=True, shared_types=(cylinder,))
  assert groups.members() == [
    (False, group.CURVES),
    (False, ellipsoid),
    (False, cylinder),
    (True, group.CURVES),
    (True, cylinder),
  ]
  stride = optix_device.RAY_TYPES
  assert groups.offset(shared=False, kind=group.CURVES) == stride
  assert groups.offset(shared=False, kind=ellipsoid) == 2 * stride
  assert groups.offset(shared=True, kind=group.CURVES) == 4 * stride
  assert groups.offset(shared=True, kind=cylinder) == 5 * stride
  # No per-world curves: the shared groups move up.
  assert group.HitGroups(False, (), True, (ellipsoid,)).offset(shared=True, kind=ellipsoid) == 2 * stride


@pytest.mark.usefixtures("_cpu_device")
def test_payload_semantics_match_the_trace_layout():
  options = pipeline.compile_options(
    has_curves=False,
    has_custom=False,
    max_registers=0,
    opacity_micromaps=False,
    calls=(programs.LAYER, programs.LIGHT_VISIBILITY),
  )
  mco, pco = options.module, options.pipeline
  assert pco.numPayloadValues == 0  # typed payloads: the module's types
  assert mco.numPayloadTypes == 4  # camera / occlusion traces, shadow traces, the layer and visibility calls
  camera, shadow = ([t.payloadSemantics[i] for i in range(t.numPayloadValues)] for t in mco.payloadTypes[:2])
  assert len(camera) == len(shadow) == 6  # `device.py`'s trace layout
  any_hit_r, any_hit_rw, is_rw, miss_write, caller_write = 1 << 6, (1 | 2) << 6, (1 | 2) << 8, 2 << 4, 2
  # Camera / occlusion traces: the ray's world (p1) reaches the any-hit and intersection programs,
  # the miss program clears the occlusion flag (p0), p3..p5 are outputs of the closest-hit
  # programs only (no program but the closest hit writes the payload).
  assert camera[1] & any_hit_r and camera[1] & (1 << 8) and camera[0] & miss_write
  assert not any(camera[k] & caller_write for k in (3, 4, 5))
  assert not any(camera[k] & (2 << 6 | 2 << 8) for k in range(6))
  # Shadow traces: p2..p5 (the last attenuating instance, the transmittance) read and written by
  # the any-hit and intersection programs; no closest-hit program.
  assert all(shadow[k] & any_hit_rw == any_hit_rw and shadow[k] & is_rw == is_rw for k in (2, 3, 4, 5))
  assert not any(shadow[k] & (3 << 2) for k in range(6))
  # The layer shading call (`device.layer_call`): its arguments in, the miss program
  # `__miss__layer` writes the result over the first `LAYER_OUT`; no hit program sees it (a trace
  # of no traversable).
  layer = [mco.payloadTypes[2].payloadSemantics[i] for i in range(mco.payloadTypes[2].numPayloadValues)]
  miss_read = 1 << 4
  assert len(layer) == optix_device.LAYER_IN
  assert all(v & miss_read and v & caller_write for v in layer)
  assert all(layer[k] & miss_write and layer[k] & 1 for k in range(optix_device.LAYER_OUT))
  assert not any(v & (3 << 2 | 3 << 6 | 3 << 8) for v in layer)
  # The light visibility call (`device.visibility_call`): the same form, after the layer call's
  # (`device.visibility_slot`).
  slot = optix_device.visibility_slot(layer=True)
  visibility = [mco.payloadTypes[slot].payloadSemantics[i] for i in range(mco.payloadTypes[slot].numPayloadValues)]
  assert len(visibility) == optix_device.VISIBILITY_IN
  assert all(v & miss_read and v & caller_write for v in visibility)
  assert all(visibility[k] & miss_write and visibility[k] & 1 for k in range(optix_device.VISIBILITY_OUT))
  assert not any(v & (3 << 2 | 3 << 6 | 3 << 8) for v in visibility)
  # OptiX rejects a value no program writes or none reads (the caller counts as both).
  for value in camera + shadow + layer + visibility:
    assert value & 0b10 or any(value & (2 << shift) for shift in (2, 4, 6, 8))
    assert value & 0b01 or any(value & (1 << shift) for shift in (2, 4, 6, 8))


@pytest.mark.usefixtures("_cpu_device")
def test_unchanged_launch_parameters_are_stored_once(monkeypatch):
  """`Pipeline.store` copies the parameters only when they changed, except once a capture
  recorded the copy (a graph replay rewrites the buffer)."""
  optix_fake.install(monkeypatch)
  pipe = _bare_pipeline()
  copies: list[int] = []
  launch = wp.launch
  monkeypatch.setattr(wp, "launch", lambda *a, **k: (copies.append(1), launch(*a, **k)))
  p = params.params_type(pipe.layout)()
  with no_default_device():
    pipe.store(p, 2, 64)
    pipe.store(p, 2, 64)
    assert len(copies) == 1
    p.nworld = 2
    pipe.store(p, 2, 64)
    pipe.store(p, 3, 64)  # other launch units
    assert len(copies) == 3
    with monkeypatch.context() as m:
      m.setattr(DeviceCaps, "capturing", lambda self: True)
      pipe.store(p, 3, 64)
    pipe.store(p, 3, 64)
    assert len(copies) == 5
  c = struct_mirror(p)
  np.testing.assert_array_equal(pipe.params[None, 1].numpy()[:, type(c).nworld.offset // 4], [2])


@pytest.mark.usefixtures("_cpu_device")
def test_a_launch_of_more_chunks_keeps_the_captured_parameter_buffer(monkeypatch):
  """A launch of more chunks than a captured one gets a buffer of its own: reallocating the one a
  graph's copy writes freed it under the graph, and the replayed launch read the new buffer."""
  fake, _ctx = optix_fake.install(monkeypatch)
  pipe = _bare_pipeline()
  p = params.params_type(pipe.layout)()
  minor = 1 << 29  # two units per chunk
  with no_default_device():
    with monkeypatch.context() as m:
      m.setattr(DeviceCaps, "capturing", lambda self: True)
      pipe.store(p, 2, minor)  # one chunk, recorded by a capture
    captured = pipe.params[None, 1]
    pipe.store(p, 3, minor)  # two chunks
    pipe.trace(optix_fake.STREAM, 2, minor)
    pipe.trace(optix_fake.STREAM, 3, minor)
  two = pipe.params[None, 2]
  assert pipe.params[None, 1] is captured and two.shape[0] == 2
  one_at, two_at = captured.ptr, two.ptr
  assert one_at is not None and two_at is not None
  assert fake.launch_params == [one_at, two_at, two_at + pipe.params_size]


@pytest.mark.parametrize(
  "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA"))]
)
def test_store_words_copies_any_layout(device):
  """`params.store_words`: one kernel for every layout writes chunk i's copy of the struct's bytes
  with its `offset` field at i · chunk (the bytes OptiX reads)."""
  import ctypes

  for layout in (params.ParamsLayout(), params.ParamsLayout(history=True, grid=True, mesh=True, layered=True)):
    p = params.params_type(layout)()
    p.nworld = 77
    words, nword, offset_word = params.store_inputs(p)
    out = wp.zeros((3, nword), dtype=wp.uint32, device=device)
    wp.launch(params.store_words, dim=(3, nword), inputs=[words, nword, offset_word, 5], outputs=[out], device=device)
    c = struct_mirror(p)
    raw = np.frombuffer(ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c)), np.uint32)
    assert nword * 4 == params.size_of(params.params_type(layout))
    for i, row in enumerate(out.numpy()):
      expected = raw.copy()
      expected[offset_word] = 5 * i
      np.testing.assert_array_equal(row, expected)


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize(
  "settings",
  [
    rw.RenderSettings(),
    rw.RenderSettings(openpbr=rw.settings.OpenPBR(diffuse_roughness=True)),
    rw.RenderSettings(outputs=rw.Outputs.of("depth")),
  ],
  ids=["default", "eon", "geometry-only"],
)
def test_frame_geometry_is_the_launch_parameters_geometry(settings):
  """BUG-105: the frame's OptiX geometry (a shaded build's: its material carrier as `geo.cut`) has
  the type of the launch parameters' `geo` (`params._fields`), so the parameters take it."""
  from raywarp.core.trace import frame_structs

  s = make_scene(PARITY_XML, cam_res=(8, 8), **{k: getattr(settings, k) for k in ("openpbr", "outputs")})
  r = s.r
  f = r.context().features  # the build's (its carrier: the frame's `geo.cut`)
  layout = programs.layout(f, scene_flags())
  p = params.params_type(layout)()
  geo = frame_structs.optix_geometry(r.context(), r.debug.frame(s.d), shaded=layout.shaded)
  p.geo = geo
  assert type(p.geo) is type(geo)
  if layout.shaded:
    assert hasattr(geo.cut, "material_alpha_cutoff") and hasattr(geo.cut, "mesh_vert")  # the carrier


def test_cached_ptx_skips_defining_the_programs(tmp_path, monkeypatch):
  """`PtxCompile`: a build whose PTX is in raywarp's cache (`programs.ptx_key`, stored with its
  kernel-code entry by `compile_once`) reads it and never defines its programs; another key
  compiles. Only the NVRTC step is stubbed."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  compiled: list[int] = []

  def nvrtc() -> bytes:
    compiled.append(1)
    return b"// ptx\n}\n"

  def compile_ptx(_module, _arch, path):
    return pipeline.compile_once(path, nvrtc)

  monkeypatch.setattr(pipeline.PtxCompile, "_compile", staticmethod(compile_ptx))
  built: list[int] = []

  def build():
    built.append(1)
    return wp.Module("ptx_cache_test", None)

  first = pipeline.PtxCompile(build, 89, "k", device_arch=120, expected=False)
  assert first.result() == b"// ptx\n}\n" and built == [1] and compiled == [1]
  again = pipeline.PtxCompile(build, 89, "k", device_arch=120, expected=False)
  assert again.module is None and again.result() == b"// ptx\n}\n" and built == [1] and compiled == [1]
  other = pipeline.PtxCompile(build, 89, "k2", device_arch=120, expected=False)
  assert other.result() == b"// ptx\n}\n" and built == [1, 1]


def test_failed_ptx_compile_raises_at_result_not_at_join(tmp_path, monkeypatch):
  """A programs' build error is raised where the programs are needed (`result`: the render); a
  release (`join`: `OptixRenderer.close`) waits for the compile and raises the error only if no
  render did (never silently dropped, never raised twice)."""
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))

  def fail(_module, _arch, _path):
    raise RuntimeError("Error while parsing function write_guides")

  monkeypatch.setattr(pipeline.PtxCompile, "_compile", staticmethod(fail))
  compile_ = pipeline.PtxCompile(lambda: wp.Module("ptx_fail_test", None), 89, "bad", device_arch=120, expected=False)
  with pytest.raises(RuntimeError, match="write_guides"):
    compile_.result()
  compile_.join()  # raised by `result` already
  unused = pipeline.PtxCompile(lambda: wp.Module("ptx_fail_test2", None), 89, "bad2", device_arch=120, expected=False)
  with pytest.raises(RuntimeError, match="write_guides"):
    unused.join()  # never needed by a render: the release raises it


def test_ptx_key_follows_what_the_programs_read(features, _headers):
  """The PTX key changes with the Features, the scene flags, the output scale and the target."""
  scene = scene_flags()
  key = programs.ptx_key(features, scene, supersampled=False, arch=89)
  assert key == programs.ptx_key(features, scene_flags(), supersampled=False, arch=89)
  assert key != programs.ptx_key(dataclasses.replace(features, spp=2), scene, supersampled=False, arch=89)
  assert key != programs.ptx_key(features, scene_flags(has_box=False), supersampled=False, arch=89)
  assert key != programs.ptx_key(features, scene, supersampled=True, arch=89)
  assert key != programs.ptx_key(features, scene, supersampled=False, arch=86)


def test_ptx_key_holds_on_any_install_location(features, tmp_path, monkeypatch):
  """The PTX key names the OptiX headers by version, not by path: prebuilt PTX (built elsewhere)
  is found on any install."""
  keys = []
  for where in ("a", "b"):
    (tmp_path / where).mkdir()
    (tmp_path / where / "optix_device.h").write_text("")
    (tmp_path / where / "optix.h").write_text("#define OPTIX_VERSION 90000\n")
    monkeypatch.setattr(optix_device, "INCLUDE_DIR", tmp_path / where)
    keys.append(programs.ptx_key(features, scene_flags(), supersampled=False, arch=89))
  assert keys[0] == keys[1]


def test_ptx_key_holds_whatever_mujoco_warp_is_installed(features, tmp_path, monkeypatch):
  """The PTX key does not depend on the MuJoCo Warp release (its kernels are vendored): a bundle
  filled with the `mujoco` extra serves a Newton or Isaac Lab install without it, and another
  MuJoCo Warp release (the key read the first `mujoco_warp-<version>.dist-info` on sys.path)."""
  from raywarp.scene.model.rayw import installed_versions

  key = programs.ptx_key(features, scene_flags(), supersampled=False, arch=89)
  (tmp_path / "mujoco_warp-0.0.1.dist-info").mkdir()
  monkeypatch.syspath_prepend(str(tmp_path))
  installed_versions.cache_clear()  # read once per process
  try:
    assert programs.ptx_key(features, scene_flags(), supersampled=False, arch=89) == key
  finally:
    installed_versions.cache_clear()


@pytest.mark.usefixtures("_cpu_device")
def test_bound_words_are_the_arrays_element_strides():
  """`params.bound_words`: the innermost stride of every array descriptor, its element size, from
  the struct's types alone: unset, empty and resized arrays, the world and ray counts change no
  word (one module per build, for every scene); pointers, extents and counts are never bound."""
  import ctypes

  from warp._src.types import array_t

  layout = params.ParamsLayout(mesh=True)
  p = params.params_type(layout)()
  c = struct_mirror(p)
  words = dict(params.bound_words(p))
  stride_at, shape_at, data_at = (
    type(c).face_normal.offset + array_t.strides.offset,
    array_t.shape.offset,
    array_t.data.offset,  # pyrefly: ignore[missing-attribute]  # Warp's array_t: ctypes fields, seen as instance-only
  )
  assert words[stride_at] == bytes(ctypes.c_int32(4))  # uint32 elements
  assert type(c).nworld.offset not in words and type(c).seed.offset not in words
  assert type(c).face_normal.offset + shape_at not in words and type(c).face_normal.offset + data_at not in words
  p.nworld = 512
  p.cams.nray = 4 * 512 * 512
  p.face_normal = wp.zeros(7, dtype=wp.uint32, device=CPU)
  p.env.gi.irradiance = wp.zeros((5, 3), dtype=wp.vec4h, device=CPU)
  assert dict(params.bound_words(p)) == words
  p.face_normal = wp.zeros(0, dtype=wp.uint32, device=CPU)
  assert dict(params.bound_words(p)) == words


@pytest.mark.usefixtures("_cpu_device")
def test_bound_words_bind_the_build_words():
  """Every copy of the build word is bound at its value (one module per word value): the geometry's
  `geo.bits` (every bound bit) in every layout; in a shaded one the light, material carrier and
  environment copies, which hold all but the geometry bits (`params.set_words`)."""
  import ctypes

  bits = toggles.CLASS_IES | toggles.CLASS_COAT | toggles.GEO_PLANES | toggles.GEO_SHARED | toggles.SETTING_GI
  bits |= toggles.high(toggles.MAP_NORMAL)
  for layout in (
    params.ParamsLayout(mesh=True),
    params.ParamsLayout(mesh=True, layered=True),
    params.ParamsLayout(shaded=False),
  ):
    struct = params.params_type(layout)
    p = struct()
    c = struct_mirror(p)
    at = params.word_offsets(struct)
    for half in ("bits", "bits2"):
      assert at[("geo",), half] == type(c).geo.offset + getattr(type(c.geo), half).offset
      assert ((("lit",), half) in at) == layout.shaded
    if layout.shaded:  # the material carrier: the first half (`GeometryFields.bits2` the second)
      assert at[("geo", "cut"), "bits"] == type(c).geo.offset + type(c.geo).cut.offset + type(c.geo.cut).bits.offset
      assert (("geo", "cut"), "bits2") not in at
    assert all(dict(params.bound_words(p))[o] == bytes(ctypes.c_int32(0)) for o in at.values())
    outputs = OutputLayout(bits=ol.OUT_RGBA | ol.OUT_DEPTH, half=ol.OUT_DEPTH)
    params.set_words(p, bits)
    params.set_outputs(p, outputs)
    words = dict(params.bound_words(p))
    for (path, half), o in at.items():
      held = bits if path == ("geo",) else bits & ~toggles.GEO_ALL
      value = held & toggles.HALF if half == "bits" else held >> toggles.HIGH
      assert words[o] == bytes(ctypes.c_int32(value))
    # The output word's copies: bound in every render layout, at their values.
    out = params.output_offsets(struct)
    assert out["bits"] == type(c).out.offset + type(c.out).bits.offset
    assert (words[out["bits"]], words[out["half"]]) == (
      bytes(ctypes.c_int32(outputs.bits)),
      bytes(ctypes.c_int32(outputs.half)),
    )
  # The GI programs' layout holds no output struct: their modules bind no output word.
  gi = params.params_type(params.ParamsLayout(mesh=True, gi=True))
  assert params.output_offsets(gi) == {} and "out" not in warp_struct(gi).vars


@pytest.mark.usefixtures("_cpu_device")
def test_a_strided_launch_array_raises():
  """A non-empty launch-parameter array whose innermost stride is not its element size (a strided
  view) raises: the bound module would misread it."""
  p = params.params_type(params.ParamsLayout(mesh=True))()
  table = wp.zeros(8, dtype=wp.uint32, device=CPU)
  p.face_normal = table[::2]
  with pytest.raises(ValueError, match="face_normal"):
    params.bound_words(p)


def _bound_value(entry: api.ModuleCompileBoundValueEntry) -> tuple[int, bytes]:
  """(offset, bytes) of a `ModuleCompileBoundValueEntry`."""
  assert entry.boundValuePtr is not None
  return entry.pipelineParamOffsetInBytes, ctypes.string_at(entry.boundValuePtr, entry.sizeInBytes)


def test_compile_options_bind_the_given_words():
  bound = ((16, b"\x07\x00\x00\x00"), (40, b"\x02\x00\x00\x00"))
  mco = pipeline.compile_options(
    has_curves=False, has_custom=False, max_registers=0, opacity_micromaps=False, bound=bound
  ).module
  gc.collect()  # the values' buffers live as long as the options
  assert [_bound_value(mco.boundValues[i]) for i in range(mco.numBoundValues)] == list(bound)


@pytest.mark.parametrize("reason", [None, "the NVIDIA driver's libnvoptix.so.1 cannot be loaded (missing)"])
def test_optix_support_reports_the_driver(monkeypatch, tmp_path, reason):
  """The device's OptiX support is the driver's function table plus the bundled headers of the
  binding's release (the trace backend raises with the reason)."""
  monkeypatch.setattr(api, "unavailable_reason", lambda: reason)
  (tmp_path / "optix.h").write_text("#define OPTIX_VERSION 90000\n")
  monkeypatch.setattr(optix_device, "include_dir", lambda: tmp_path)
  caps = DeviceCaps(
    alias="cuda:9",
    kind="cuda",
    arch=89,
    rt_cores=True,
    graphs=True,
    pinned=True,
    device_textures=True,
    parallel_launch=True,
    blackwell_rtx=False,
  )
  assert caps.optix.available == (reason is None) and caps.optix.reason == reason


@pytest.mark.parametrize(
  ("header", "reason"),
  [
    ("#define OPTIX_VERSION 90000\n", None),
    ("#define OPTIX_VERSION 90100\n", "are OptiX 9.1.0, the binding is OptiX 9.0.0"),
    ("// no version\n", "declares no OPTIX_VERSION"),
  ],
)
def test_optix_headers_must_match_the_binding_abi(tmp_path, header, reason):
  """OptiX 9.1 headers with the 9.0 binding compile programs against the wrong intrinsics: the
  OptiX traversal is then unavailable, with the cause."""
  (tmp_path / "optix.h").write_text(header)
  found = optix_device.headers_reason(tmp_path)
  assert found is None if reason is None else (found is not None and reason in found)


# ---------------------------------------------------------------------------------------------
# Native-handle rollback and CUDA-context identity without a driver or native build.


_handles = itertools.count(1)


@dataclass
class Handle:
  name: str
  destroyed: list[str]
  fail_destroy: bool = False
  handle: int = field(default_factory=lambda: next(_handles))

  def destroy(self) -> None:
    self.destroyed.append(self.name)
    if self.fail_destroy:
      raise RuntimeError(f"destroy {self.name}")


@dataclass
class PipelineContext:
  fail: str = ""
  cleanup_failure: bool = False
  created: list[str] = field(default_factory=list)
  destroyed: list[str] = field(default_factory=list)
  error: Exception = field(default_factory=lambda: MemoryError("injected construction failure"))
  groups: int = 0
  pipelines: int = 0

  def make(self, name: str) -> Handle:
    self.created.append(name)
    return Handle(name, self.destroyed, self.cleanup_failure)

  def module(self, mco: api.ModuleCompileOptions, pco: api.PipelineCompileOptions, ptx: bytes) -> Handle:
    return self.make("module")

  def program_group(self, desc: api.ProgramGroupDesc, payload: api.PayloadType | None = None) -> Handle:
    self.groups += 1
    if self.fail == "group" and self.groups == 2:
      raise self.error
    return self.make(f"group{self.groups}")

  def pipeline(self, pco: api.PipelineCompileOptions, link: api.PipelineLinkOptions, groups: list[Handle]) -> Handle:
    self.pipelines += 1
    if self.fail == "pipeline" and self.pipelines == 2:
      raise self.error
    return self.make(f"pipeline{self.pipelines}")


def make_pipeline(monkeypatch: pytest.MonkeyPatch, ctx: PipelineContext, calls: tuple[str, ...] = ()) -> pipeline.Pipeline:
  optix_fake.install(monkeypatch)
  monkeypatch.setattr(api, "accumulate_stack_sizes", lambda groups, pipe: api.StackSizes())

  def stack(*args: object) -> None:
    if ctx.fail == "stack":
      raise ctx.error

  monkeypatch.setattr(pipeline, "set_stack_size", stack)
  scene = cast(
    SceneFlags,
    SimpleNamespace(
      has_world_curves=False, has_shared_curves=False, has_omm=False, primitive_types=(), shared_primitive_types=()
    ),
  )
  names = ["__raygen__render", "__miss__render", "__closesthit__triangle", "__closesthit__custom", "__raygen__extra", *calls]
  ptx = b"" if ctx.fail == "entry" else "".join(f".entry {name}_abcdef01(\n" for name in names).encode()

  def records(groups: list[Handle], device: wp.Device) -> wp.array[wp.uint8, [int]]:
    if ctx.fail == "sbt":
      raise ctx.error
    return wp.zeros(32 * len(groups), dtype=wp.uint8, device=device)

  monkeypatch.setattr(pipeline, "_records", records)
  record = pipeline.module_record(scene, 0, (), shapes=toggles.SHAPE_ALL)
  options = pipeline.record_options(record, calls=calls)
  return pipeline.Pipeline(
    ctx=cast(DeviceContext, ctx),
    device=wp.get_device("cpu"),
    ptx=ptx,
    scene=scene,
    record=record,
    module=cast(OptixObject, ctx.module(options.module, options.pipeline, ptx)),
    layout=params.ParamsLayout(),
    extras=("__raygen__extra",),
  )


@pytest.mark.parametrize("stage", ["entry", "group", "pipeline", "stack", "sbt"])
def test_pipeline_failure_releases_every_acquired_handle(monkeypatch, stage):
  ctx = PipelineContext(fail=stage)
  with pytest.raises(RuntimeError if stage == "entry" else MemoryError) as caught:
    make_pipeline(monkeypatch, ctx)
  if stage != "entry":
    assert caught.value is ctx.error
  assert ctx.created
  assert ctx.destroyed == ctx.created[::-1]


def test_pipeline_cleanup_failure_preserves_original_and_drains_handles(monkeypatch):
  ctx = PipelineContext(fail="sbt", cleanup_failure=True)
  with pytest.raises(MemoryError) as caught:
    make_pipeline(monkeypatch, ctx)
  assert caught.value is ctx.error
  assert ctx.destroyed == ctx.created[::-1]
  assert len(caught.value.__notes__) == len(ctx.created)


def test_pipeline_close_releases_once_in_dependency_order(monkeypatch):
  ctx = PipelineContext()
  owner = make_pipeline(monkeypatch, ctx)
  owner.destroy()
  owner.destroy()
  assert ctx.destroyed == ctx.created[::-1]


@pytest.mark.parametrize(
  "calls", [(), (programs.LAYER,), (programs.LIGHT_VISIBILITY,), (programs.LAYER, programs.LIGHT_VISIBILITY)]
)
def test_called_programs_get_their_miss_records_payload_types_and_trace_depth(monkeypatch, calls):
  # The called programs' miss records follow one per ray type, in `call_programs`' order, each
  # with its own payload type (the light visibility call's slot: `device.visibility_slot`); every
  # call is one trace level deeper (the layer program calls the light visibility program).
  ctx = PipelineContext()
  payloads: list[int] = []
  group = ctx.program_group

  def program_group(desc: api.ProgramGroupDesc, payload: api.PayloadType | None = None) -> Handle:
    payloads.append(-1 if payload is None else payload.numPayloadValues)
    return group(desc, payload)

  ctx.program_group = program_group
  owner = make_pipeline(monkeypatch, ctx, calls)
  sizes = {programs.LAYER: optix_device.LAYER_IN, programs.LIGHT_VISIBILITY: optix_device.VISIBILITY_IN}
  # The traces' payload types have 6 registers (-1: no type, the ray-generation groups).
  assert [n for n in payloads if n not in (6, -1)] == [sizes[name] for name in calls]
  assert owner.sbts[None].missRecordCount == optix_device.RAY_TYPES + len(calls)
  assert owner._depth == 1 + len(calls)
  if programs.LIGHT_VISIBILITY in calls:
    slot = optix_device.visibility_slot(layer=programs.LAYER in calls)
    assert slot == optix_device.RAY_TYPES + calls.index(programs.LIGHT_VISIBILITY)
  owner.destroy()


def test_context_cache_uses_cuda_context_not_physical_ordinal(monkeypatch):
  calls: list[int] = []

  def create(identity: int, _arch: int) -> object:
    calls.append(identity)
    return SimpleNamespace(
      set_cache_location=lambda location: None, set_cache_enabled=lambda enabled: None, set_cache_sizes=lambda low, high: None
    )

  monkeypatch.setattr(context, "DeviceContext", create)
  monkeypatch.setattr(context, "_contexts", {})
  first = cast(wp.Device, SimpleNamespace(context=1001, ordinal=0, arch=120))
  second = cast(wp.Device, SimpleNamespace(context=1002, ordinal=0, arch=120))
  alias = cast(wp.Device, SimpleNamespace(context=1001, ordinal=0, arch=120))
  a, b = context.device_context(first), context.device_context(second)
  assert a is not b
  assert context.device_context(alias) is a
  assert calls == [1001, 1002]


@pytest.mark.parametrize(
  ("optix_env", "cache_env", "expected"),
  [
    (None, "{tmp}", [("location", "{tmp}/optix"), ("sizes", (context.CACHE_LOW, context.CACHE_HIGH))]),
    (None, "", [("enabled", False)]),
    ("/elsewhere", "{tmp}", [("location", "/elsewhere"), ("sizes", (context.CACHE_LOW, context.CACHE_HIGH))]),
  ],
)
def test_optix_module_cache_lives_in_the_raywarp_cache_dir(monkeypatch, tmp_path, optix_env, cache_env, expected):
  """OptiX's module disk cache is `<RAYWARP_CACHE_DIR>/optix` (one directory holds every warm-start
  cache), off with caching off, OPTIX_CACHE_PATH's directory when it names one (set through the
  API: `reload_cache` moves it); its size bound holds a module pack either way."""
  calls: list[tuple[str, object]] = []
  ctx = SimpleNamespace(
    set_cache_location=lambda location: calls.append(("location", location)),
    set_cache_enabled=lambda enabled: calls.append(("enabled", enabled)),
    set_cache_sizes=lambda low, high: calls.append(("sizes", (low, high))),
  )
  monkeypatch.delenv("OPTIX_CACHE_MAXSIZE", raising=False)
  monkeypatch.setenv("RAYWARP_CACHE_DIR", cache_env.format(tmp=tmp_path))
  if optix_env is None:
    monkeypatch.delenv("OPTIX_CACHE_PATH", raising=False)
  else:
    monkeypatch.setenv("OPTIX_CACHE_PATH", optix_env)
  context._set_cache(cast(context.DeviceContext, ctx), optix_env)
  assert calls == [(k, v.format(tmp=tmp_path) if isinstance(v, str) else v) for k, v in expected]
  calls.clear()
  monkeypatch.setenv("OPTIX_CACHE_MAXSIZE", "1000")
  context._set_cache(cast(context.DeviceContext, ctx), optix_env)
  assert ("sizes", (context.CACHE_LOW, context.CACHE_HIGH)) not in calls  # OptiX's own bound


# ---------------------------------------------------------------------------------------------
# OptiX launch parameters stay under the size whose by-value copy NVVM keeps in local memory
# (the programs then run several times slower: params.PARAMS_LIMIT). Runs without a GPU.


# An array field of a struct (pointer, shape, strides, ...).
_ARRAY = 56


def _size(layout: params.ParamsLayout) -> int:
  """Bytes of the launch parameters of `layout` (no warning above the limit)."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return params.size_of(params.params_type(layout))


# Every program of both groups: the scene flags that put the most tables in the parameters.
_EVERY_PROGRAM = SimpleNamespace(
  has_mesh=True,
  has_box=True,
  has_instances=True,
  has_shared=True,
  translucent_triangles=True,
  translucent_curves=(True, True),
  shared_visible=True,
  primitive_types=(4, 5),
  shared_primitive_types=(4, 5),
  has_world_curves=True,
  has_shared_curves=True,
  has_omm=False,
)


# The optional material / UV tables (diffuse_roughness, live_material_uv, specular_weight): 112
# bytes each.
_OPTIONS: Final = ("diffuse_roughness", "live_material_uv", "specular_weight")
# Every recipe's parameters, with every program and table and every optional table, fit: a shaded
# build's material carrier is its geometry's alpha-test tables (`geo.cut`), so the texture, UV and
# mesh-face tables they share are held once (BUG-105: they were held twice, 560 bytes).


def _layout_size(recipe_name: str, options: tuple[str, ...]) -> int:
  f = dataclasses.replace(preset_features(recipe(recipe_name), PARITY_XML), culling=True)
  f = dataclasses.replace(
    f,
    diffuse_roughness="diffuse_roughness" in options,
    live_material_uv="live_material_uv" in options,
    specular_weight="specular_weight" in options,
  )
  return _size(programs.layout(f, _EVERY_PROGRAM))  # a stand-in Scene


@pytest.mark.parametrize("recipe_name", sorted(RECIPE_NAMES))
def test_every_preset_layout_fits(recipe_name):
  """Each recipe's parameters, with every scene-dependent table (the light grid, the programs
  of both groups), fit under the limit."""
  assert _layout_size(recipe_name, ()) <= params.PARAMS_LIMIT


@pytest.mark.parametrize("recipe_name", sorted(RECIPE_NAMES))
@pytest.mark.parametrize("options", [(o,) for o in _OPTIONS] + [_OPTIONS])
def test_every_preset_layout_fits_with_optional_tables(recipe_name, options):
  """... and with the optional material / UV tables."""
  assert _layout_size(recipe_name, options) <= params.PARAMS_LIMIT


def test_fields_only_with_their_programs():
  """Optional tables cost their bytes only in builds whose programs read them."""
  base = _size(params.ParamsLayout())
  assert _size(params.ParamsLayout(history=True)) > base
  office = params.ParamsLayout(history=True, grid=True, mesh=True)
  assert _size(dataclasses.replace(office, mesh=False)) == _size(office) - _ARRAY
  assert params.params_type(params.ParamsLayout()) is params.params_type(params.ParamsLayout())


def test_rough_diffuse_layout_adds_only_its_two_tables():
  base = params.ParamsLayout()
  assert _size(dataclasses.replace(base, diffuse_roughness=True)) == _size(base) + 2 * _ARRAY


# ---------------------------------------------------------------------------------------------
# The OptiX programs still generate CUDA source, without a GPU, a driver or the OptiX SDK.
#
# `programs.build` runs over a stand-in for `accel.OptixScene` (its feature flags only) and Warp
# generates each build's CUDA source, so the programs, the OptiX `Tracer` (`scene_tracer`) and the
# native intrinsics (`core/kernels/native.py`) are checked on CPU-only machines: hand-picked builds
# and every recipe. Compiling and running them needs a GPU (`tests/optix_backend/test_optix.py`).


@pytest.fixture(scope="module")
def features():
  """The default Features of the parity scene (meshes and every primitive), as an OptiX build's."""
  return preset_features(rw.RenderSettings())


@pytest.fixture
def _headers(tmp_path, monkeypatch):
  """`device.build_options` only checks that the SDK header exists: codegen never reads it."""
  (tmp_path / "optix_device.h").write_text("")
  monkeypatch.setattr(optix_device, "INCLUDE_DIR", tmp_path)


def _check(f, scene) -> None:
  """Generates the build's CUDA source and checks its programs and intrinsics."""
  module = programs.build(f, scene)
  source = cuda_source(module)
  names = {k.key for k in module._get_live_kernels()}
  assert {"__raygen__render", "__miss__render", "__closesthit__triangle", "__closesthit__custom"} <= names
  assert ("__raygen__visibility" in names) == programs.visibility_program(f)
  # Custom primitives cull back faces for the culling queries' rays only (`trace.RAY_MASK_CULL`).
  culls = bool(f.backface_culling and (scene.primitive_types or scene.shared_primitive_types))
  assert ("mjpbr_optix::ray_mask(" in source) == culls
  # The shading compiles once per build: one definition, one call (the pixel centre's own shading,
  # for the direct / indirect outputs off the centre, goes through it too).
  assert "__raygen__extra" not in names
  defs = re.findall(r"static CUDA_CALLABLE \S+ (_make_trace_body__locals__trace_pixel_\d+)\(", source)
  assert len(defs) == (1 if f.any_rgb else 0), defs  # a geometry-only build shades nothing
  if defs:
    assert len(re.findall(r"(?<!\w)" + defs[0] + r"\(", source)) == 2  # its definition and one call
  # The any-hit programs of both ray types (camera / occlusion, shadow), as `anyhit_programs` lists.
  for name in programs.anyhit_programs(f, scene):
    assert name in names, name

  cut, seen = f.alpha_cutout, f.visibility and scene.shared_visible
  assert ("__anyhit__triangle" in names) == bool(cut or seen)
  assert ("__anyhit__curve" in names) == bool(cut and scene.has_world_curves)
  assert ("__anyhit__curve_shared" in names) == bool((cut or seen) and scene.has_shared_curves)
  # The alpha test of the analytic hits: in the curves' any-hit and the custom primitives'
  # intersection programs, which test a cut-out primitive's crossings unculled on occlusion rays.
  customs = scene.primitive_types or scene.shared_primitive_types
  assert ("mjpbr_optix::first_hit_ray(" in source) == bool(cut and customs)
  assert ("__closesthit__curve_shared" in names) == scene.has_shared_curves
  for t in scene.shared_primitive_types:
    for shadow in (False, True):
      assert programs.intersection_name(t, shared=True, shadow=shadow) in names
  assert "mjpbr_optix::trace_closest(" in source and "mjpbr_optix::launch_index(" in source
  # The shadow trace through translucent casters.
  assert ("mjpbr_optix::trace_shadow(" in source) == f.translucent_shadows


BUILDS = {
  "plain": ({}, {}),
  # The light grid's (unculled) program beside culling custom primitives.
  "culling_shadows_grid": ({"backface_culling": True, "culling": True}, {}),
  "translucent": ({"translucent_shadows": True}, {"translucent_triangles": True}),
  "translucent_cutout": (
    {"translucent_shadows": True, "alpha_cutout": True},
    {"translucent_triangles": True, "translucent_curves": (True, False)},
  ),
  "cutout_anyhit_shared_grid": (
    {"alpha_cutout": True, "culling": True, "shared": True},
    {"has_shared": True, "has_shared_curves": True, "shared_primitive_types": (4,)},
  ),
  # Per-world visibility of shared instances: the any-hit programs of the shared triangles and
  # curves, the shared intersection programs' test, with and without the alpha test.
  "shared_visibility": (
    {"visibility": True, "shared": True},
    {"shared_visible": True, "has_shared": True, "has_shared_curves": True, "shared_primitive_types": (4, 5)},
  ),
  "shared_visibility_cutout": (
    {"visibility": True, "shared": True, "alpha_cutout": True},
    {"shared_visible": True, "has_shared": True, "shared_primitive_types": (5,)},
  ),
  "translucent_curves": ({"translucent_shadows": True}, {"translucent_curves": (True, True), "has_shared_curves": True}),
  "shared_visibility_counted": (
    {"visibility": True, "shared": True, "alpha_cutout": True, "counters": True},
    {"shared_visible": True, "has_shared": True, "has_shared_curves": True},
  ),
  # Live material UVs read through the material carrier (a shaded build's `geo.cut`), by the
  # shading and the hit programs' alpha test; cut-out curves and custom primitives without
  # triangles.
  "live_uv": ({"live_material_uv": True}, {}),
  "live_uv_cutout": ({"live_material_uv": True, "alpha_cutout": True}, {}),
  "cutout_analytic_only": (
    {"alpha_cutout": True, "world_uv": True, "backface_culling": True},
    {"has_mesh": False, "has_box": False},
  ),
}


@pytest.mark.usefixtures("_headers")
@pytest.mark.parametrize("name", list(BUILDS))
def test_programs_generate_cuda_source(features, name):
  f_kw, scene_kw = BUILDS[name]
  _check(dataclasses.replace(features, **f_kw), scene_flags(**scene_kw))


@pytest.mark.usefixtures("_headers")
def test_culling_reaches_custom_primitives_only_through_the_ray_mask():
  """Culling and unculled queries share the intersection programs: the culling queries' rays
  carry `RAY_MASK_CULL`, the others clear it (the light grid's, translucent shadows', the alpha
  test's: a culling build had no grid program with cylinders / ellipsoids, and its unculled
  queries culled them)."""
  from raywarp.core.trace.optix.device import trace as rt_trace

  spec = rt_trace.TraceSpec(
    planes=PLANES_NONE,
    has_mesh=True,
    has_box=True,
    has_instances=True,
    cull=True,
    has_shared=False,
    compact=False,
    visibility=False,
  )
  assert int(spec.ray_mask) == 255
  assert int(dataclasses.replace(spec, cull=False).ray_mask) == 255 & ~rt_trace.RAY_MASK_CULL


@pytest.mark.usefixtures("_headers")
def test_any_hit_path_counters_are_specialized_to_debug_builds(features, monkeypatch):
  """The OptiX trace spec only emits path atomics for `performance.counters` builds."""
  from raywarp.core.trace.optix.device import trace as rt_trace

  specs = []
  make_trace, make_shadow = rt_trace.make_trace, rt_trace.make_shadow
  monkeypatch.setattr(rt_trace, "make_trace", lambda spec: specs.append(spec) or make_trace(spec))
  monkeypatch.setattr(rt_trace, "make_shadow", lambda spec: specs.append(spec) or make_shadow(spec))
  scene = scene_flags(has_shared=True, translucent_triangles=True)
  programs.scene_tracer(scene, dataclasses.replace(features, counters=False), cull=False)
  assert specs and all(not spec.counting for spec in specs)
  specs.clear()
  programs.scene_tracer(scene, dataclasses.replace(features, counters=True, translucent_shadows=True), cull=False)
  assert len(specs) == 2 and all(spec.counting for spec in specs)  # the queries, the shadow trace


@pytest.mark.usefixtures("_headers")
def test_two_sided_instances_reach_planes_and_custom_primitives(features, monkeypatch):
  """Two-sided instances (USD doubleSided) are never culled: the OptiX plane queries and
  intersection programs skip them as the software traversal does (they culled them)."""
  from raywarp.core.trace.optix.device import hits
  from raywarp.core.trace.optix.device import trace as rt_trace

  specs, intersections = [], []
  make_trace, make_intersection = rt_trace.make_trace, hits.make_intersection
  monkeypatch.setattr(rt_trace, "make_trace", lambda spec: specs.append(spec) or make_trace(spec))
  monkeypatch.setattr(hits, "make_intersection", lambda **k: intersections.append(k) or make_intersection(**k))
  f = dataclasses.replace(features, backface_culling=True, two_sided=True, culling=True)
  _check(f, scene_flags())
  assert specs and all(s.two_sided for s in specs)
  assert intersections and all(k["cull_backfaces"] and k["two_sided"] for k in intersections)


@pytest.mark.usefixtures("_headers")
@pytest.mark.parametrize("recipe_name", sorted(RECIPE_NAMES))
def test_every_preset_generates_cuda_source(recipe_name):
  """Every recipe's build over every program of both groups (shared scenery included)."""
  scene = scene_flags(has_shared=True, has_shared_curves=True, shared_primitive_types=(4, 5))
  f = preset_features(recipe(recipe_name))
  _check(f, scene)
  # Each lens model's render program alone in a module of its own (`build_lens`), shading once.
  for model in LENS_MODELS:
    lens = programs.build_lens(f, scene, model)
    assert {k.key for k in lens._get_live_kernels()} == {programs.raygen_name(model)}
    defs = re.findall(r"static CUDA_CALLABLE \S+ (_make_trace_body__locals__trace_pixel_\d+)\(", cuda_source(lens))
    assert len(defs) == (1 if f.any_rgb else 0), model


def _function_body(source: str, prefix: str) -> str:
  """Extract one generated device function, excluding explanatory source comments."""
  match = re.search(r"static (?:WP_FORCEINLINE )?CUDA_CALLABLE [^\n]* " + re.escape(prefix) + r"_\d+\(", source)
  assert match is not None, prefix
  start = source.index("\n{", match.start())
  end = source.index("\n}\n", start)
  return re.sub(r"//[^\n]*", "", source[start:end])


@pytest.mark.usefixtures("_headers")
def test_optix_ray_queries_read_cached_grid_offset(features):
  """Generated raygen reads the cell offset instead of recalculating the world grid index."""
  source = cuda_source(programs.build(features, scene_flags()))
  body = _function_body(source, "to_cell")
  assert "cell_offset_" in body
  assert "world_offset_" not in body
  assert "world_offset_" not in source


@pytest.mark.usefixtures("_headers")
@pytest.mark.parametrize("samples", [1, 4, 16])
def test_sample_loops_keep_one_call_site_without_rolling_transparency(features: Features, samples: int) -> None:
  """Sample budgets grow a loop bound; the four ordered transparency layers stay unrolled."""
  f = dataclasses.replace(
    features,
    glossy=2,
    area_lights=True,
    shadow_samples=samples,
    translucent_shadows=True,
  )
  module = programs.build(f, scene_flags(translucent_triangles=True))
  assert module.options["max_unroll"] == 6
  source = cuda_source(module)
  # The compiled sampling (`light_evaluators._make_area_visibility`, which `make_area_visibility`
  # picks).
  area = _function_body(source, "_make_area_visibility__locals__area_visibility")
  assert ("start_for_" in area) == (samples > 1)
  assert len(re.findall(r" = make_shadow_pass__locals__shadow_pass_\d+\(", area)) == 1
  directional = _function_body(source, "make_shadow_ray__locals__shadow_ray")
  assert "start_for_" not in directional


# ---------------------------------------------------------------------------------------------
# Opacity micromaps (`core/trace/optix/omm*.py`): the micro-triangle order, the bake's
# conservativeness (CPU), and OptiX renders with micromaps byte-identical to the any-hit test alone
# (GPU).


wp.set_module_options({"enable_backward": False})


# Points per micro-triangle (the alpha test checked there): its corners, edge midpoints, centroid
# and a few interior ones, as barycentric weights of its three corners.
_WEIGHTS = np.array(
  [
    *([1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [1 / 3, 1 / 3, 1 / 3]),
    *([0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.45, 0.45, 0.1], [0.1, 0.45, 0.45]),
    # Just inside the corners and edges (the bake's padding).
    *([0.998, 0.001, 0.001], [0.0005, 0.999, 0.0005], [0.5, 0.4999, 0.0001], [0.0001, 0.5, 0.4999]),
  ],
  np.float32,
)


def _micro2bary(index: int, level: int) -> set[tuple[float, float]]:
  """OptiX's micro2bary (optix_micromap_impl.h) in Python: micro-triangle `index`'s corners."""
  if level == 0:
    return {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)}

  def even(x: int) -> int:
    x &= 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    return (x | (x >> 8)) & 0x0000FFFF

  def eor(x: int) -> int:
    for s in (1, 2, 4, 8):
      x ^= x >> s
    return x

  m = 0xFFFFFFFF
  b0, b1 = even(index), even(index >> 1)
  fx, fy = eor(b0), eor(b0 & ~b1 & m)
  t = fy ^ b1
  iu = (fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t)
  iv = fy ^ b0
  iw = (~fx & ~t) | (b0 & ~t) | (~b0 & fx & t)
  mask = (1 << level) - 1
  iu, iv, iw = iu & mask, iv & mask, iw & mask
  y = (iu & 1) ^ (iv & 1) ^ (iw & 1) ^ 1
  d = 1.0 / (1 << level)
  u, v = iu * d, iv * d
  return {(u, v + y * d), (u + d, v + y * d), (u + y * d, v + (1 - y) * d)}


@wp.kernel
def _corners(
  # In:
  level: int,
  # Out:
  out: wp.array2d[wp.vec2],
):
  """Corners of every micro-triangle of `level`."""
  i = wp.tid()
  c0, c1, c2 = omm_kernels.micro_triangle(i, level)
  out[i, 0] = c0
  out[i, 1] = c1
  out[i, 2] = c2


@pytest.mark.parametrize("level", [1, 2, 3, 5])
def test_micro_triangles_follow_optix_order(level):
  """`micro_triangle` gives OptiX's micro-triangles (micro2bary's corner sets) and they tile the
  triangle."""
  n = 4**level
  out = wp.zeros((n, 3), dtype=wp.vec2, device="cpu")
  wp.launch(_corners, dim=n, inputs=[level, out], device="cpu")
  corners = out.numpy().astype(np.float64)
  for i in range(n):
    ref = _micro2bary(i, level)
    got = {(float(a), float(b)) for a, b in corners[i]}
    assert len(ref) == 3 and got == ref, (i, got, ref)
  e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
  area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
  np.testing.assert_allclose(area.sum(), 0.5)
  centroids = corners.mean(axis=1)
  assert len(np.unique(np.rint(centroids * 1.0e9), axis=0)) == n
  assert np.all(centroids.sum(axis=1) < 1.0) and np.all(centroids > 0.0)


@wp.kernel
def _check_states(
  # Model:
  mesh_faceadr: wp.array1d[int],
  mesh_facetexcoord: wp.array1d[wp.vec3i],
  mesh_texcoord: wp.array1d[wp.vec2],
  mesh_texcoord_adr: wp.array1d[int],
  # In:
  tex: wp.Texture2D,
  layout: int,
  cutoff: float,
  texcoord: wp.vec2i,
  xform: uv_xform,
  repeat: wp.vec2,
  mesh: int,
  index: wp.array1d[wp.int32],
  desc: wp.array1d[wp.uint32],
  data: wp.array1d[wp.uint8],
  weights: wp.array1d[wp.vec3],
  # Out:
  wrong_out: wp.array1d[int],
  resolved_out: wp.array1d[int],
):
  """Per (face, point of a micro-triangle grid): whether the alpha test there contradicts the
  baked state (`wrong_out[face]`), and how many points a resolved state covers."""
  face, w = wp.tid()
  _has, uv0, uv1, uv2 = omm_kernels.face_uvs(
    mesh_faceadr, mesh_facetexcoord, mesh_texcoord, mesh_texcoord_adr, texcoord, xform, repeat, wp.vec2(1.0), mesh, face
  )
  k = index[face]
  level = 6
  if k >= 0:
    level = int(desc[2 * k + 1]) & 0xFFFF
  for i in range(1 << (2 * level)):
    state = int(0)
    if k >= 0:
      byte = int(data[int(desc[2 * k]) + i // 4])
      state = (byte >> (2 * (i % 4))) & 3
    elif k == -1:
      state = 0
    elif k == -2:
      state = 1
    else:
      state = 3
    c0, c1, c2 = omm_kernels.micro_triangle(i, level)
    b = c0 * weights[w][0] + c1 * weights[w][1] + c2 * weights[w][2]
    uv = uv0 * (1.0 - b[0] - b[1]) + uv1 * b[0] + uv2 * b[1]
    cut = texture.sample_layout(tex, layout, uv, 0.0)[0] < cutoff
    if (state == 0 and not cut) or (state == 1 and cut):
      wp.atomic_add(wrong_out, face, 1)
    if state <= 1:
      wp.atomic_add(resolved_out, face, 1)


def _check_bake(scene: rw.GpuScene, source: omm_sources.CutSource, tex: wp.Texture2D, device: wp.Device) -> tuple[int, int]:
  """(points where a baked state contradicts the alpha test, resolved points) of `source`'s
  bake over `tex` (`_check_states` at every `_WEIGHTS` point of every micro-triangle)."""
  m = scene.meshes
  b = omm_bake.bake(m, source, tex, device=device)
  nface = len(m.faces[source.mesh])
  wrong, resolved = (wp.zeros(nface, dtype=int, device=device) for _ in range(2))
  wp.launch(
    _check_states,
    dim=(nface, len(_WEIGHTS)),
    inputs=[
      m.faceadr,
      m.facetexcoord,
      m.texcoord,
      m.texcoord_adr,
      tex,
      source.layout,
      source.cutoff,
      wp.vec2i(*source.texcoord),
      uv_xform(*source.uv_xform),
      wp.vec2(*source.repeat),
      source.mesh,
      b.index,
      b.desc,
      b.data,
      wp.array(_WEIGHTS, dtype=wp.vec3, device=device),
    ],
    outputs=[wrong, resolved],
    device=device,
  )
  return int(wrong.numpy().sum()), int(resolved.numpy().sum())


@pytest.mark.parametrize("name", ["glass", "cutouts"])
def test_bake_is_conservative(name):
  """Leaf cards, a tiled fence, soft-edged decals: every micro-triangle the bake resolves agrees
  with the alpha test at every checked point (OPAQUE: kept, TRANSPARENT: discarded), and some are
  resolved."""
  fs = internal_module("feature_scenes")
  rig = fs.load(name, device="cpu")
  scene = rig.gpu
  host = omm_sources.host_tables(scene)
  sources = {s for g in range(scene.ninstance) if (s := omm_sources.source_of(host, g)) is not None}
  assert sources, "the glass scene has static cut-out leaves"
  m = scene.meshes
  for source in sources:
    b = omm_bake.bake(m, source, scene.textures[source.texture], device=wp.get_device("cpu"))
    nface = len(m.faces[source.mesh])
    wrong, resolved = (wp.zeros(nface, dtype=int, device="cpu") for _ in range(2))
    wp.launch(
      _check_states,
      dim=(nface, len(_WEIGHTS)),
      inputs=[
        m.faceadr,
        m.facetexcoord,
        m.texcoord,
        m.texcoord_adr,
        scene.textures[source.texture],
        source.layout,
        source.cutoff,
        wp.vec2i(*source.texcoord),
        uv_xform(*source.uv_xform),
        wp.vec2(*source.repeat),
        source.mesh,
        b.index,
        b.desc,
        b.data,
        wp.array(_WEIGHTS, dtype=wp.vec3, device="cpu"),
      ],
      outputs=[wrong, resolved],
      device="cpu",
    )
    assert wrong.numpy().sum() == 0, (source, wrong.numpy())
    assert b.faces[3] > 0 or b.faces[1] > 0, b.faces
    assert resolved.numpy().sum() > 0


def _gpu_optix() -> bool:
  """CUDA with the OptiX traversal (driver OptiX and the bundled headers)."""
  return wp.is_cuda_available() and device_caps("cuda:0").optix.available


_GPU = pytest.mark.skipif(not _gpu_optix(), reason="needs CUDA and driver OptiX")


# The cut-out instances of each scene, posed per world in the identity runs.
_CUT = {"glass": "/World/Leaf*", "cutouts": "/World/Leaf*"}


def _renderer(name: str, recipe_name: str, *, per_world: rw.PerWorld | None, **settings: object) -> fs.Rig:
  """A 2-world 128^2 OptiX Renderer of feature scene `name` at recipe `recipe_name` (with the
  scene's settings and `settings` over it)."""
  fs = internal_module("feature_scenes")
  rig = fs.load(name, res=(8, 8), device="cuda:0")
  over = {**dict(fs.FEATURES[name].settings), "outputs": rw.Outputs.of("rgb", "hdr"), "performance.traversal": "optix"}
  r = rw.Renderer(
    rig.gpu,
    nworld=2,
    cameras=list(rig.r.cameras),
    resolution=(128, 128),
    settings=recipe(recipe_name).replace(**{**over, **settings}),
    per_world=per_world,
  )
  rig.r.close()
  rig = dataclasses.replace(rig, r=r)
  rig.set_views(fs.FEATURES[name].views)
  return rig


def _sequence(rig: fs.Rig, name: str) -> list[tuple[np.ndarray, np.ndarray]]:
  """(rgb, hdr) of four steps: the views; the camera moved along the scene's path; the cut-out
  instances moved in world 1 only; back to the views (the instances reset)."""
  fs = internal_module("feature_scenes")
  out = []
  cut = rig.r.per_world.columns("instance.pos", _CUT[name])
  start = cut.numpy().copy()
  views = fs.FEATURES[name].views
  for step in range(4):
    if step == 1:
      rig.set_views([fs.FEATURES[name].camera(0.35)] * len(views))
    if step == 2:
      moved = start.copy()
      moved[1] += np.array([-0.15, 0.1, 0.05], np.float32)
      cut.assign(moved)
    if step == 3:
      rig.set_views(views)
      cut.assign(start)
    rig.render(frames=2, seed=10 * step)
    out.append((rig.rgb(), rig.hdr()))
  return out


@_GPU
@pytest.mark.parametrize("recipe_name", ["default", "fast"])
@pytest.mark.parametrize("name", ["glass", "cutouts"])
def test_micromaps_render_identical_pixels(name, recipe_name):
  """Leaves, a tiled fence, soft-edged decals (OptiX): the cut-outs get micromaps (`has_omm`) and
  every frame (8-bit and HDR, two worlds, the cut-out instances posed per world; the views, a
  moved camera, instances moved in one world, the reset) is byte-identical to the any-hit test
  alone."""
  frames = {}
  for micromaps in (False, True):
    with tuning.override(opacity_micromaps=micromaps):
      rig = _renderer(name, recipe_name, per_world=rw.PerWorld(pose=[_CUT[name]]))
      frames[micromaps] = _sequence(rig, name)
    assert rig.r._backend.optix.scene.has_omm == micromaps
    rig.r.close()
  for (rgb0, hdr0), (rgb1, hdr1) in zip(frames[False], frames[True], strict=True):
    np.testing.assert_array_equal(rgb1, rgb0)
    np.testing.assert_array_equal(hdr1, hdr0)


@_GPU
def test_per_world_cutoff_has_no_micromaps():
  """A per-world alpha cutoff (and so a per-world alpha test) builds no micromap: world 1's leaves
  turn opaque (cutoff 0) while world 0's stay cut out."""
  rig = _renderer("cutouts", "default", per_world=rw.PerWorld(material=["alpha_cutoff"]))
  rig.render()
  assert not rig.r._backend.optix.scene.has_omm
  cut = rig.r.per_world["material.alpha_cutoff"]
  values = cut.numpy()
  values[1] = 0.0
  cut.assign(values)
  rig.render()
  rgb = rig.rgb()
  assert np.abs(rgb[0].astype(int) - rgb[1].astype(int)).max() > 64
  rig.r.close()


@_GPU
def test_micromaps_match_software():
  """cutouts: OptiX with micromaps renders the software traversal's image (the cut-out edges
  agree; the backends differ only by float rounding)."""
  out = {}
  for traversal in ("software", "optix"):
    rig = _renderer("cutouts", "default", per_world=None, **{"performance.traversal": traversal})
    rig.render()
    out[traversal] = rig.hdr()[0].reshape(-1, 3)
    if traversal == "optix":
      assert rig.r._backend.optix.scene.has_omm
    rig.r.close()
  diff = np.abs(out["optix"] - out["software"]).max(-1) / np.maximum(out["software"].max(-1), 1e-3)
  assert np.mean(diff > 0.02) < 0.01


def test_bake_is_stored_on_disk_by_content(monkeypatch, tmp_path):
  """A second process's bake of the same cut-out (same texels, texture content, alpha test and
  levels) loads the stored micromaps (no micro-state launch) and they equal the computed ones; a
  changed cutoff misses."""
  fs = internal_module("feature_scenes")
  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  cpu = wp.get_device("cpu")
  scene = fs.load("cutouts", device="cpu").gpu
  host = omm_sources.host_tables(scene)
  source = next(s for g in range(scene.ninstance) if (s := omm_sources.source_of(host, g)) is not None)
  tex = scene.textures[source.texture]
  first = omm_bake.bake(scene.meshes, source, tex, device=cpu)
  assert list((tmp_path / "bakes" / "omm").glob("*.npz"))
  baked: list[bool] = []
  monkeypatch.setattr(omm_bake, "_bake", lambda *a, **k: baked.append(True))
  again = omm_bake.bake(scene.meshes, source, tex, device=cpu)
  assert not baked
  for name in ("data", "desc", "index"):
    np.testing.assert_array_equal(getattr(again, name).numpy(), getattr(first, name).numpy())
  assert (again.histogram, again.faces, again.micro) == (first.histogram, first.faces, first.micro)
  with pytest.raises(AttributeError):  # the stub returns None: the changed cutoff was baked
    omm_bake.bake(scene.meshes, dataclasses.replace(source, cutoff=source.cutoff * 0.5), tex, device=cpu)
  assert baked


def test_bake_cache_keys_the_texture_and_keeps_the_last_cap(monkeypatch):
  """`omm.micromaps` bakes once per (meshes, texture, device, source): the same call is a hit, a
  new level cap replaces the old cap's entry, a texture replaced under the same index is baked
  again, and a collected texture drops its entries."""
  fs = internal_module("feature_scenes")
  import gc

  builds: list[omm_bake.Bake] = []

  def build_array(ctx: object, b: omm_bake.Bake, *, device: wp.Device) -> object:
    """The array build (OptiX): recorded."""
    builds.append(b)
    return object()

  monkeypatch.setattr(omm, "build_array", build_array)
  cpu = wp.get_device("cpu")
  scene = fs.load("cutouts", device="cpu").gpu
  host = omm_sources.host_tables(scene)
  source = next(s for g in range(scene.ninstance) if (s := omm_sources.source_of(host, g)) is not None)
  ctx = cast("DeviceContext", None)
  levels = omm_bake.face_levels(scene.meshes, source, scene.textures[source.texture], device=cpu)

  def entries() -> list[omm.BakeKey]:
    return [k for k in omm._cache if k[0] == id(scene.meshes)]

  first = omm.micromaps(ctx, scene, source, levels, 3, device=cpu)
  assert omm.micromaps(ctx, scene, source, levels, 3, device=cpu) is first and len(builds) == 1
  lower = omm.micromaps(ctx, scene, source, levels, 2, device=cpu)
  assert lower is not first and len(builds) == 2 and len(entries()) == 1  # the cap-3 entry is gone
  old = scene.textures[source.texture]
  scene.textures[source.texture] = wp.Texture2D(np.zeros((4, 4, 4), np.float32), device=cpu)
  assert omm.micromaps(ctx, scene, source, levels, 2, device=cpu) is not lower and len(builds) == 3
  assert len(entries()) == 2
  replaced = scene.textures[source.texture]
  scene.textures[source.texture] = old
  del replaced
  gc.collect()
  assert [k[1] for k in entries()] == [id(old)]  # the collected texture's bake is dropped


# The synthetic bakes' texture side [texels] and UV grid: vertices at uv i / _CELLS, on texel
# boundaries (16 / 4 = 4 texels per cell).
_SIDE = 16


_CELLS = 4


_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


def _checker(block: int, low: float, high: float) -> np.ndarray:
  """(_SIDE, _SIDE) checkerboard of `block`-texel squares, `low` / `high`."""
  i = np.arange(_SIDE) // block
  return np.where((i[:, None] + i[None, :]) % 2 == 0, high, low).astype(np.float32)


def _grid_scene(device: str, *, exact: bool = False) -> rw.GpuScene:
  """A unit quad of _CELLS^2 cells (two triangles each, UVs = positions on texel boundaries) per
  opacity texture of the scene path: sRGB RGBA (alpha apart), linear LA and L, each with
  (repeat, clamp) and (mirror, black) wrap, cutoff 0.5; `exact`: every texel exact
  (`EXACT_STORAGE`), else the default (8-bit sRGB color where the device decodes it)."""
  b = rw.SceneBuilder()
  g = np.arange(_CELLS + 1) / _CELLS
  uv = np.stack(np.meshgrid(g, g, indexing="xy"), axis=-1).reshape(-1, 2)
  k = np.arange(_CELLS)
  c = (k[:, None] * (_CELLS + 1) + k[None, :]).reshape(-1)
  tris = np.concatenate([np.stack([c, c + 1, c + _CELLS + 2], 1), np.stack([c, c + _CELLS + 2, c + _CELLS + 1], 1)])
  mesh = b.add_mesh(np.c_[uv, np.zeros(len(uv))], tris, uvs=uv)
  alpha = (255 * _checker(2, 0.0, 1.0)).astype(np.uint8)
  images = {
    "srgb": ("srgb_rec709_scene", np.dstack([alpha, 255 - alpha, alpha, alpha])),
    "la": ("linear", np.dstack([alpha, alpha])),
    "l": ("linear", alpha[..., None]),
  }
  for name, (space, image) in images.items():
    for wrap in (("repeat", "clamp"), ("mirror", "black")):
      tex = b.add_texture(image, colorspace=space, wrap=wrap)
      cut = rw.Material(ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5), name=f"{name}_{wrap[0]}")
      ref = TextureRef(tex, channel=3 if name == "srgb" else (1 if name == "la" else 0))
      b.add_instance("mesh", mesh=mesh, material=b.add_material(cut.with_texture(Slot.GEOMETRY_OPACITY, ref)))
  return rw.upload(
    b.build(),
    device,
    options=rw.SceneOptions(textures=formats.EXACT_STORAGE if exact else formats.TextureStorage()),
  )


def _variants(source: omm_sources.CutSource, scene: rw.GpuScene) -> list[omm_sources.CutSource]:
  """`source` with tiling 1 and 3.7, and with the mesh's UVs read as the instance's own through a
  rotation + offset (a non-identity `uv_xform`)."""
  m = source.mesh
  own = (int(scene.meshes.texcoord_adr.numpy()[m]), int(scene.meshes.faceadr.numpy()[m]))
  a = np.deg2rad(30.0)
  rotated = (np.cos(a), -np.sin(a), 0.37, np.sin(a), np.cos(a), -0.21)
  return [
    dataclasses.replace(source, repeat=(1.0, 1.0)),
    dataclasses.replace(source, repeat=(3.7, 3.7)),
    dataclasses.replace(source, repeat=(3.7, 1.0), texcoord=own, uv_xform=tuple(float(x) for x in rotated)),
  ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("storage", ["default", "exact"])
def test_bake_is_conservative_on_texel_boundaries(device, storage):
  """Scene-path opacity textures (sRGB alpha apart, LA, L; default or exact storage; every wrap
  mode) over UVs on texel boundaries, tiled 1 / 3.7, through a rotated UV transform: no baked
  state contradicts the alpha test at the micro-triangles' corners, edges and interiors."""
  scene = _grid_scene(device, exact=storage == "exact")
  host = omm_sources.host_tables(scene)
  sources = [s for g in range(scene.ninstance) if (s := omm_sources.source_of(host, g)) is not None]
  assert len(sources) == 6
  resolved = 0
  for source in sources:
    for variant in _variants(source, scene):
      wrong, ok = _check_bake(scene, variant, scene.textures[variant.texture], wp.get_device(device))
      assert wrong == 0, variant
      resolved += ok
  assert resolved > 0


_ADDRESS = [wp.TextureAddressMode.WRAP, wp.TextureAddressMode.CLAMP, wp.TextureAddressMode.MIRROR, wp.TextureAddressMode.BORDER]


# (block size, low, high) of the raw textures: texel-scale and coarser checkerboards, and an 8-bit
# one a quantisation step either side of the cutoff (0.5).
_PATTERNS = [(1, 0.0, 1.0), (2, 0.2, 0.9), (4, 0.0, 1.0), (4, 127 / 255, 128 / 255)]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("pattern", range(len(_PATTERNS)))
def test_bake_is_conservative_for_any_sampler(device, pattern):
  """Raw opacity textures: bilinear and nearest filtering, every address mode, float and 8-bit
  texels, a mip chain (the alpha test reads level 0): no contradicted state."""
  block, low, high = _PATTERNS[pattern]
  values = _checker(block, low, high)
  eight_bit = high - low < 0.01
  data = np.rint(values * 255).astype(np.uint8) if eight_bit else values
  scene = _grid_scene(device)
  base = next(s for g in range(scene.ninstance) if (s := omm_sources.source_of(omm_sources.host_tables(scene), g)) is not None)
  base = dataclasses.replace(base, layout=TEX_LAYOUT_IDENTITY)
  resolved = 0
  for i, (filt, address) in enumerate(
    (f, a) for f in (wp.TextureFilterMode.LINEAR, wp.TextureFilterMode.CLOSEST) for a in _ADDRESS
  ):
    tex = wp.Texture2D(data, filter_mode=filt, address_mode=address, num_mip_levels=1 + 3 * (i % 2), device=device)
    for variant in _variants(base, scene):
      wrong, ok = _check_bake(scene, variant, tex, wp.get_device(device))
      assert wrong == 0, (filt, address, variant)
      resolved += ok
  assert resolved > 0 or block == 1  # a texel-scale checker leaves every bilinear footprint mixed


# ---------------------------------------------------------------------------------------------
# A world's acceleration coordinates cannot depend on another world's geometry bounds.


type Observation = tuple[
  Floats[[int, int, int, int, 3]], Floats[[int, int, int, int]], np.ndarray[[int, int, int, int], np.dtype[np.int32]]
]


_CELLS_XML: Final = """
<mujoco>
  <worldbody>
    <light pos="0 -1 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1"/>
    <body name="robot" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" pos="-0.6 0 0" size="0.12 0.13 0.17"/>
      <geom type="sphere" pos="-0.2 0 0" size="0.16"/>
      <geom type="capsule" pos="0.2 0 0" size="0.09 0.12"/>
      <geom type="cylinder" pos="0.6 0 0" size="0.12 0.16"/>
      <geom type="ellipsoid" pos="0 0.5 0" size="0.2 0.1 0.14"/>
    </body>
    <camera name="cam" pos="0 -3 1.8" xyaxes="1 0 0 0 0.4 0.92"/>
  </worldbody>
</mujoco>
"""


def _bounds_rows(boxes: np.ndarray) -> wp.array:
  """`instances.BOUNDS` rows of (n, 6) boxes (lower xyz, upper xyz) and their centres as the
  reference positions."""
  boxes = np.asarray(boxes, np.float32).reshape(-1, 6)
  ref = 0.5 * (boxes[:, :3] + boxes[:, 3:])
  return wp.array(np.concatenate([boxes, ref], axis=1).reshape(-1), dtype=wp.float32, device="cpu")


@wp.kernel
def _original_and_cached_cell_point(
  # In:
  cells: wp.array1d[wp.vec3],
  grid: wp.vec3i,
  # Out:
  original: wp.array1d[wp.vec3],
  cached: wp.array1d[wp.vec3],
):
  """Use the old expression and the cached value on the same Warp CPU device."""
  worldid = wp.tid()
  point = wp.vec3(0.2, -0.7, 3.1)
  center = cells[worldid * wp.static(trace.CELL_SIZE) + wp.static(trace.CELL_CENTER)]
  original[worldid] = (point - center) * trace.world_scale(cells, worldid) + trace.world_offset(grid, worldid)
  cached[worldid] = trace.to_cell(cells, worldid, point)


def _cells(nworld: int) -> wp.array:
  """An empty cell table of `nworld` worlds on the CPU."""
  return wp.empty(nworld * trace.CELL_SIZE, dtype=wp.vec3, device="cpu")


@pytest.mark.parametrize("nworld", [1, 2, 17, 512])
def test_cached_grid_offsets_match_original_arithmetic(nworld: int) -> None:
  """The per-world cell write retains the old grid mapping at varied grid shapes."""
  grid = wp.vec3i(*accel._grid(nworld))
  bounds = _bounds_rows(np.tile(np.array([-1, -2, -3, 1, 2, 3], np.float32), (nworld, 1)))
  cells = _cells(nworld)
  original = wp.empty(nworld, dtype=wp.vec3, device="cpu")
  cached = wp.empty_like(original)
  wp.launch(instances.update_cell, dim=nworld, inputs=[bounds, grid], outputs=[cells], device="cpu")
  wp.launch(_original_and_cached_cell_point, dim=nworld, inputs=[cells, grid], outputs=[original, cached], device="cpu")
  np.testing.assert_array_equal(original.numpy(), cached.numpy())


def test_cell_frames_are_independent_and_bounded() -> None:
  """Changing one bound row cannot change another frame or unbound normalized cell size."""
  boxes = np.array([[-1, -2, -3, 1, 2, 3], [-1, -2, -3, 1, 2, 3]], np.float32)
  cells = _cells(2)
  grid = wp.vec3i(2, 1, 1)
  wp.launch(instances.update_cell, dim=2, inputs=[_bounds_rows(boxes), grid], outputs=[cells], device="cpu")
  before = cells.numpy().reshape(2, trace.CELL_SIZE, 3).copy()
  boxes[1] = [-10000, -2, -3, 10000, 2, 3]
  wp.launch(instances.update_cell, dim=2, inputs=[_bounds_rows(boxes), grid], outputs=[cells], device="cpu")
  after = cells.numpy().reshape(2, trace.CELL_SIZE, 3)
  np.testing.assert_array_equal(before[0], after[0])
  # Odd integers in one binade per axis: [4, 8) for 2 cells, [2, 4) for 1.
  np.testing.assert_array_equal(after[:, trace.CELL_OFFSET], [[5, 3, 3], [7, 3, 3]])
  assert after[1, trace.CELL_SCALE, 0] < before[1, trace.CELL_SCALE, 0]
  for c in range(2):
    extent = after[c, trace.CELL_UPPER] - after[c, trace.CELL_LOWER]
    scale = after[c, trace.CELL_SCALE, 0]
    assert np.max(extent * scale) < 1.0 and 0.5 <= np.max(extent) * scale  # the power of two that fits
    assert np.frexp(scale)[0] == 0.5  # a power of two


_TWO_BODIES = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <body name="stays" pos="-0.4 0 0.3">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
      <body name="moves" pos="0.4 0 0.3">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
  </mujoco>
"""


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize("distance", [0.5 * reach.NEAR_RADIUS, 1.5 * reach.NEAR_RADIUS, 1.0e4])
def test_a_body_beyond_near_radius_is_hidden(monkeypatch, distance):
  """BUG-E13 without a second cell: world 1's second body moved `distance` along x. Within
  `reach.NEAR_RADIUS` of the world's reference it is drawn (the cell spans it); beyond, it is
  hidden with a finite record and leaves the cell (and so the precision of the world's other
  geometry) to the body in reach. World 0 is unchanged."""
  optix_fake.install(monkeypatch)
  s = make_scene(_TWO_BODIES, nworld=2, cam_res=(8, 8), performance=rw.settings.Performance(traversal="software"))
  with no_default_device():
    scene = accel.OptixScene(s.r.scene, 2, s.r.layout, device=CPU, mesh_used=None, mesh_frame=None)
  frame = _frame(s)
  _frame_step(scene, frame, False, monkeypatch)
  cell = scene.tables.cell.numpy().copy()
  moves = int(np.flatnonzero(s.r.layout.per_world)[-1])
  pos = frame.instance_pos.numpy()
  pos[1, moves, 0] += distance
  frame.instance_pos.assign(pos)
  _frame_step(scene, frame, False, monkeypatch)
  after = scene.tables.cell.numpy()
  np.testing.assert_array_equal(after[: trace.CELL_SIZE], cell[: trace.CELL_SIZE])
  ntri, k = len(scene.worlds.tri_ids), scene.worlds.tri_ids.index(InstanceId(moves))
  records = scene.worlds.instances.numpy()[[k, ntri + k]]
  mask = record_field(records, "visibility_mask")
  world1 = after[trace.CELL_SIZE :]
  if distance <= reach.NEAR_RADIUS:
    assert mask[0] != 0 and mask[1] != 0
    assert world1[trace.CELL_UPPER, 0] > distance
  else:
    assert mask[0] != 0 and mask[1] == 0
    # The cell spans the body left in reach only: its scale is no finer than world 0's.
    assert world1[trace.CELL_UPPER, 0] < 1.0
    assert world1[trace.CELL_SCALE, 0] >= after[trace.CELL_SCALE, 0]
    hidden = np.stack([record_field(records, t)[1] for t in ("t0", "t1", "t2")])
    assert np.isfinite(hidden).all() and np.abs(hidden[:, 3]).max() < 10.0


@pytest.mark.parametrize("nworld", [1, 7, 64, 4096])
def test_world_cells_share_one_binade_per_axis(nworld: int) -> None:
  """Every cell (offset +- 1/2) lies in one binade [b, 2b) per axis, so identical worlds round their
  cell coordinates identically (`trace.py`'s module doc)."""
  ncell = nworld
  grid = accel._grid(ncell)
  bounds = _bounds_rows(np.tile(np.array([-1, -1, -1, 1, 1, 1], np.float32), (nworld, 1)))
  cells = _cells(nworld)
  wp.launch(instances.update_cell, dim=nworld, inputs=[bounds, wp.vec3i(*grid)], outputs=[cells], device="cpu")
  offset = cells.numpy().reshape(ncell, trace.CELL_SIZE, 3)[:, trace.CELL_OFFSET]
  for axis in range(3):
    lo, hi = offset[:, axis] - 0.5, offset[:, axis] + 0.5
    b = 2.0 ** np.floor(np.log2(lo.min()))
    assert lo.min() >= b and hi.max() < 2 * b, (axis, lo.min(), hi.max())
  assert len({tuple(o) for o in offset}) == ncell  # distinct cells


@pytest.mark.parametrize("gtype", [ShapeType.CYLINDER, ShapeType.ELLIPSOID])
@pytest.mark.parametrize("scale", [1.0, 5.0e-6, 250.0])
@pytest.mark.parametrize("shared", [False, True])
def test_custom_primitive_hits_preserve_scene_units(gtype: ShapeType, scale: float, shared: bool) -> None:
  """A millimetre primitive keeps outside/inside hits, normals and misses across cell scales."""
  primitive_hit = hits.make_primitive_hit(gtype=int(gtype), compact=False, shared=shared)

  @wp.kernel
  @concrete(geo=OptixGeometry)
  def query(
    # In:
    geo: OptixFields,
    # Out:
    out: wp.array1d[wp.vec4],
  ):
    ray = wp.tid()
    origin = wp.vec3(0.0, 0.0, -0.003)
    if ray == 1:
      origin = wp.vec3(0.0)
    elif ray == 2:
      origin[0] = 0.004
    if wp.static(not shared):
      origin = trace.to_cell(geo.rt_cell, 0, origin)
    distance, normal = primitive_hit(geo, 0, 0, origin, wp.vec3(0.0, 0.0, 1.0))
    out[ray] = wp.vec4(distance, normal[0], normal[1], normal[2])

  geo = OptixGeometry()
  geo.instance_pos = wp.zeros((1, 1), dtype=wp.vec3, device="cpu")
  geo.instance_rot = wp.array(np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3), dtype=wp.mat33, device="cpu")
  geo.instance_size = wp.full((1, 1), wp.vec3(0.001), dtype=wp.vec3, device="cpu")
  cells = np.zeros((trace.CELL_SIZE, 3), np.float32)
  cells[trace.CELL_SCALE] = scale
  geo.rt_cell = wp.array(cells, dtype=wp.vec3, device="cpu")
  geo.rt_grid = wp.vec3i(1, 1, 1)
  out = wp.empty(3, dtype=wp.vec4, device="cpu")
  wp.launch(query, dim=3, inputs=[geo], outputs=[out], device="cpu")
  observed = out.numpy()
  distance_scale = 1.0 if shared else scale
  np.testing.assert_allclose(observed[:2, 0], np.array([0.002, 0.001]) * distance_scale, rtol=1e-5, atol=0.0)
  np.testing.assert_allclose(observed[:2, 1:], [[0, 0, -1], [0, 0, 1]], atol=1e-6)
  assert observed[2, 0] < 0.0


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA and OptiX required")
def test_identical_worlds_render_identically() -> None:
  """Worlds in the same state render the same depth, segmentation and HDR in every world index
  (boxes, spheres, capsules, cylinders, ellipsoids; shadows on a shared plane), as the software
  traversal does: each world's cell coordinates equal up to exact offsets (`trace.py`)."""
  if not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  with wp.ScopedDevice("cuda:0"):
    s = make_scene(
      _CELLS_XML,
      nworld=12,
      cam_res=(64, 48),
      outputs=rw.Outputs.of("hdr", "depth", "seg"),
      aa=rw.settings.AntiAliasing(spp=1),
      performance=rw.settings.Performance(traversal="optix"),
    )
    try:
      mjw.forward(s.m, s.d)
      s.r.render(s.d)
      for out in (s.hdr(), s.depth(), s.seg()):
        np.testing.assert_array_equal(out, np.broadcast_to(out[:1], out.shape))
      assert len(np.unique(s.seg()[0])) == 7  # five shapes, the plane, the background
    finally:
      s.r.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA and OptiX required")
@pytest.mark.parametrize("changed", [0, 1])
@pytest.mark.parametrize("capture", [False, True])
def test_other_world_motion_preserves_observations(changed: int, capture: bool) -> None:
  """Cover triangle, built-in curve and custom primitives, eager and captured refits."""
  if not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  with wp.ScopedDevice("cuda:0"):
    s = make_scene(
      _CELLS_XML,
      nworld=2,
      cam_res=(32, 32),
      outputs=rw.Outputs.of("hdr", "depth", "seg"),
      aa=rw.settings.AntiAliasing(spp=1),
      performance=rw.settings.Performance(traversal="optix"),
    )
    try:
      assert s.r.features.traversal == "optix"
      # Compare GPU-forward poses on both sides, not CPU initialization against GPU rounding.
      mjw.forward(s.m, s.d)
      s.r.render(s.d)
      graph = s.r.capture(sim=s.d) if capture else None

      def render() -> Observation:
        if graph is None:
          out = s.r.render(s.d)
        else:
          graph.launch(s.d)
          out = s.r.outputs
        # The renderer resolves output shapes at runtime from the cameras and world count.
        return cast(
          Observation, (required(out.hdr).numpy().copy(), required(out.depth).numpy().copy(), required(out.seg).numpy().copy())
        )

      before = render()
      repeat = render()
      for first, second in zip(before, repeat, strict=True):
        np.testing.assert_array_equal(first, second)
      untouched_pos = s.d.geom_xpos.numpy()[1 - changed].copy()
      untouched_rot = s.d.geom_xmat.numpy()[1 - changed].copy()
      qpos = s.d.qpos.numpy()
      qpos[changed, :3] = (20.0, -10.0, 4.0)
      s.d.qpos.assign(qpos)
      mjw.forward(s.m, s.d)
      np.testing.assert_array_equal(s.d.geom_xpos.numpy()[1 - changed], untouched_pos)
      np.testing.assert_array_equal(s.d.geom_xmat.numpy()[1 - changed], untouched_rot)
      after = render()
      for first, second in zip(before, after, strict=True):
        np.testing.assert_array_equal(first[1 - changed], second[1 - changed])
    finally:
      s.r.close()


def test_curve_gas_refits_with_uniform_splitting(monkeypatch):
  """A curve GAS is refitted with PREFER_FAST_BUILD (uniform splitting): never ALLOW_UPDATE with
  PREFER_FAST_TRACE or the default adaptive splitting, whose every refit OptiX reports as a
  fallback. The built-in intersector gets the same flags."""
  _table, ctx = optix_fake.install(monkeypatch)
  flags = gas.curve_build_flags()
  assert flags == api.BUILD_FLAG_ALLOW_UPDATE | api.BUILD_FLAG_PREFER_FAST_BUILD
  assert not flags & api.BUILD_FLAG_PREFER_FAST_TRACE
  cpu = wp.get_device("cpu")
  accel = gas.curves(
    cast(DeviceContext, ctx),
    wp.zeros(4, dtype=wp.vec3, device=cpu),
    wp.zeros(4, dtype=float, device=cpu),
    wp.zeros(2, dtype=wp.uint32, device=cpu),
    device=cpu,
  )
  accel.build(0)
  accel.build(0, update=True)
  assert [b.update for b in ctx.builds] == [False, True]


def test_a_structure_built_without_allow_update_refuses_a_refit(monkeypatch):
  _table, ctx = optix_fake.install(monkeypatch)
  accel = gas.Accel(cast(DeviceContext, ctx), api.BuildInput(), api.BUILD_FLAG_PREFER_FAST_TRACE, device=wp.get_device("cpu"))
  accel.build(0)
  with pytest.raises(RuntimeError, match="ALLOW_UPDATE"):
    accel.build(0, update=True)


@dataclass
class _Released:
  """A pipeline or PTX compile stand-in recording its release (`OptixRenderer.close`)."""

  log: list[str]
  name: str
  error: Exception | None = None

  def join(self) -> None:
    self.log.append(f"join {self.name}")
    if self.error is not None:
      raise self.error

  def destroy(self) -> None:
    self.log.append(f"destroy {self.name}")


@pytest.mark.usefixtures("_cpu_device")
def test_closing_after_a_failed_compile_still_releases_everything():
  """`OptixRenderer.close` joins every compile and destroys every pipeline before it raises the
  render programs' compile failure (it once raised at the first join: the GI compile was never
  joined and the pipelines and module never released)."""
  log: list[str] = []
  failure = RuntimeError("NVRTC failed")
  r = object.__new__(launch.OptixRenderer)
  r.closed, r.device, r.module, r._ahead, r._ptx_bytes = False, CPU, None, {}, None
  r._ptx = cast("pipeline.PtxCompile", _Released(log, "render", failure))
  r._gi_ptx = cast("pipeline.PtxCompile", _Released(log, "gi"))
  r._lens_ptx, r._lens_bytes, r._area_ptx = None, None, None
  r._pipelines = {(): cast("pipeline.Pipeline", _Released(log, "pipeline"))}
  r._gi = {(): cast("pipeline.Pipeline", _Released(log, "gi pipeline"))}
  r._area = {}
  with pytest.raises(RuntimeError, match="NVRTC failed"):
    r.close()
  assert log == ["join render", "join gi", "destroy pipeline", "destroy gi pipeline"]
  assert r.closed and r._ptx is None and r._gi_ptx is None


def test_a_failed_compile_keeps_its_own_errors_when_another_compile_logs():
  """A module compile attaches the errors logged since it started: a second compile on the same
  context (a build's modules created together) starting meanwhile cleared them."""
  logger = context._Logger()
  logger(2, "COMPILER", "an earlier failure")
  first = logger.logged
  logger(2, "COMPILER", "the first module's error")
  second = logger.logged  # the second compile starts
  logger(2, "COMPILER", "the second module's error")
  assert logger.errors_since(first) == ["[COMPILER] the first module's error", "[COMPILER] the second module's error"]
  assert logger.errors_since(second) == ["[COMPILER] the second module's error"]


def test_repeated_optix_warnings_are_logged_once(caplog):
  logger = context._Logger()
  with caplog.at_level(logging.INFO, logger="raywarp.optix"):
    for _ in range(3):
      logger(3, "ACCEL", "Updating of adaptively split curve build inputs is not supported\n")
      logger(2, "ERROR", "a failure")
  assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.ERROR, logging.ERROR, logging.ERROR]


@pytest.mark.usefixtures("_cpu_device")
@pytest.mark.parametrize("usage", [[], [(3, 2, api.OPACITY_MICROMAP_FORMAT_4_STATE)]])
def test_micromap_build_input_has_no_empty_usage_array(usage):
  """OptiX rejects a non-null usage-count array with a zero count (every face predefined): the
  pointer is null exactly when there are no counts."""
  points = wp.zeros(3, dtype=wp.vec3, device=CPU)
  faces = wp.zeros(1, dtype=wp.vec3i, device=CPU)
  maps = omm.Micromaps(
    array=wp.zeros(1, dtype=wp.uint64, device=CPU), index=wp.zeros(1, dtype=wp.int32, device=CPU), usage=usage
  )
  m = omm.triangles(points, faces, maps).triangleArray.opacityMicromap
  assert m.numMicromapUsageCounts == len(usage)
  assert bool(m.micromapUsageCounts) == bool(usage)


@wp.kernel
def _shared_spans(
  # Model:
  cell: wp.array1d[wp.vec3],
  # In:
  origins: wp.array1d[wp.vec3],
  directions: wp.array1d[wp.vec3],
  reach: wp.array1d[float],
  # Out:
  out: wp.array1d[int],
):
  """Ray i's `trace.shared_span` (1: traced)."""
  i = wp.tid()
  out[i] = int(rt_trace.shared_span(cell, origins[i], directions[i], reach[i]))


def test_shared_span_launches_only_rays_that_meet_the_shared_box() -> None:
  """`trace.shared_span` against the shared IAS's box after the worlds' cells (the AABB its
  builds emit): a ray through it, one starting inside, one hitting its face exactly, one parallel
  inside its slab are traced; one passing beside it, one stopping short of it, one pointing away
  are not."""
  cell = np.zeros((2 * rt_trace.CELL_SIZE + rt_trace.SHARED_BOX, 3), np.float32)
  cell[-2:] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
  rays = [  # (origin, direction, reach, traced)
    ((0.5, 0.5, -1.0), (0.0, 0.0, 1.0), 1e30, True),
    ((0.5, 0.5, 0.5), (0.6, 0.0, 0.8), 1e30, True),
    ((1.0, 0.5, -1.0), (0.0, 0.0, 1.0), 1e30, True),
    ((0.5, 0.5, 2.0), (1.0, 0.0, 0.0), 1e30, False),
    ((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), 1e30, True),
    ((2.0, 0.5, -1.0), (0.0, 0.0, 1.0), 1e30, False),
    ((0.5, 0.5, -1.0), (0.0, 0.0, 1.0), 0.5, False),
    ((0.5, 0.5, -1.0), (0.0, 0.0, -1.0), 1e30, False),
  ]
  arrays = [
    wp.array(np.array([r[k] for r in rays], np.float32), dtype=dt, device="cpu")
    for k, dt in ((0, wp.vec3), (1, wp.vec3), (2, float))
  ]
  out = wp.zeros(len(rays), dtype=int, device="cpu")
  wp.launch(
    _shared_spans, dim=len(rays), inputs=[wp.array(cell, dtype=wp.vec3, device="cpu"), *arrays], outputs=[out], device="cpu"
  )
  assert out.numpy().astype(bool).tolist() == [r[3] for r in rays]


@pytest.mark.usefixtures("_cpu_device")
def test_pipeline_lookup_remembers_nothing_per_frame():
  """A build's pipeline is found by the bound words of each launch's parameters, checked at every
  launch: parameters that change per frame (counts, buffers) select the one pipeline and leave no
  per-value entry behind."""
  r = launch.OptixRenderer.__new__(launch.OptixRenderer)
  made: list[params.BoundWords] = []
  r.device = wp.get_device("cpu")
  r._pipelines = {}

  def new_pipeline(words: params.BoundWords) -> pipeline.Pipeline:
    made.append(words)
    return cast("pipeline.Pipeline", object())

  r._new_pipeline = new_pipeline
  p = params.params_type(params.ParamsLayout(mesh=True))()
  first = r._pipeline(p)
  for n in range(1, 200):
    p.nworld = n
    p.face_normal = wp.zeros(n, dtype=wp.uint32, device=CPU)
    assert r._pipeline(p) is first
  assert len(made) == 1 and len(r._pipelines) == 1
  p.face_normal = wp.zeros(8, dtype=wp.uint32, device=CPU)[::2]
  with pytest.raises(ValueError, match="face_normal"):
    r._pipeline(p)
