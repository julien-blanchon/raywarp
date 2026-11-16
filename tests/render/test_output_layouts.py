"""Color output layouts (`ColorOutput(channels=3)`, `planar=True`): RGB bytes and channels-first
planes written by the kernels, world-major like every output, equal to the RGBA bytes."""

import dataclasses
import functools
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import make_scene, required

import raywarp as rw
from raywarp.core.kernels.abi import output_layout as ol
from raywarp.core.kernels.abi.output_layout import layout_of
from raywarp.core.kernels.abi.structs import OutputArrays, OutputTable
from raywarp.core.kernels.display import write_display
from raywarp.core.outputs.render_outputs import Buffers
from raywarp.core.pipeline.gpu_scene.binding import StaleBinding
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.settings.fields import OutputKind

if TYPE_CHECKING:
  from conftest import MakeScene
import gc
import weakref
from typing import TYPE_CHECKING

from conftest import MakeScene
from test_temporal import XML as SPHERE_XML

from raywarp.core.outputs.render_outputs import _alias

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.6 0.4 1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.2 0.5 0.9 1"/>
    <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2" rgba="0.9 0.2 0.2 1"/>
    <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


RES = (24, 16)


PLANAR = rw.settings.ColorOutput(channels=3, planar=True)
PLANAR16 = rw.settings.ColorOutput(channels=3, planar=True, dtype="float16")
RGB3 = rw.settings.ColorOutput(channels=3)


def _chw(rgba: np.ndarray) -> np.ndarray:
  """(nworld, ncam, H, W, 4) bytes -> (nworld * ncam, 3, H, W)."""
  planes = np.moveaxis(rgba[..., :3], -1, 2)
  return planes.reshape(-1, *planes.shape[2:])


class _Opts(TypedDict, total=False):
  """The scene options of `_rgba`'s reference."""

  nworld: int
  cam_res: tuple[int, int]


def _rgba(scene: MakeScene, *extra: OutputKind, **kw: Unpack[_Opts]) -> np.ndarray:
  """The default RGBA bytes of the scene `kw` builds (with the outputs `extra`), rendered twice
  (the reference)."""
  return required(scene(XML, outputs=rw.Outputs.of("rgb", *extra), **kw).render().render().out.rgb).numpy()


@pytest.mark.parametrize("extra", [(), ("motion",)])
def test_rgb8_planes_are_the_rgba_bytes(scene, extra):
  """Every pass that finalises the color (the plain megakernel, the motion body) writes the
  planes: they equal the RGBA bytes channels-first."""
  kw: _Opts = {"nworld": 2, "cam_res": RES}
  s = scene(XML, outputs=rw.Outputs.of(*extra, rgb=PLANAR), **kw)
  assert s.r._buffers.layout.bits == ol.OUT_CHW8
  s.render().render()
  chw = s.out.rgb.numpy()
  assert chw.dtype == np.uint8 and chw.shape == (4, 3, RES[1], RES[0])
  np.testing.assert_array_equal(chw, _chw(_rgba(scene, *extra, **kw)))


def test_rgb16f_planes_hold_the_unquantised_display_color(scene):
  """float16 planes: the sRGB display value in [0, 1] before 8-bit quantisation."""
  s = scene(XML, cam_res=RES, outputs=rw.Outputs.of(rgb=PLANAR16)).render()
  chw = s.out.rgb.numpy()
  assert chw.dtype == np.float16
  np.testing.assert_allclose(chw.astype(np.float32), _chw(_rgba(scene, cam_res=RES)) / 255.0, atol=0.5 / 255 + 1e-3)


def test_bind_planes_to_an_observation_tensor(scene):
  """bind_outputs(rgb=...): the kernels write the policy's (nworld * ncam, 3, H, W) tensor."""
  s = scene(XML, nworld=2, cam_res=RES, outputs=rw.Outputs.of(rgb=PLANAR))
  obs = wp.zeros((4, 3, RES[1], RES[0]), dtype=wp.uint8)
  s.r.bind_outputs(rgb=obs)
  s.render()
  np.testing.assert_array_equal(obs.numpy(), _chw(_rgba(scene, nworld=2, cam_res=RES)))


def test_per_camera_planes(scene):
  """With per-camera resolutions each camera's planes are its own (nworld, 3, H_c, W_c) view."""
  s = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb"))
  cams = [s.scene.cameras["cam"], dataclasses.replace(s.scene.cameras["top"], resolution=(12, 10))]
  ref = rw.Renderer(s.scene, nworld=1, cameras=cams, resolution=RES, settings=s.r.settings)
  r = rw.Renderer(s.scene, nworld=1, cameras=cams, resolution=RES, settings=s.r.settings.replace(**{"outputs.rgb": PLANAR}))
  ref.render(s.d)
  r.render(s.d)
  assert r.outputs.rgb is None
  for c, (w, h) in enumerate([RES, (12, 10)]):
    view = r.outputs.cameras[c]
    assert required(view.rgb).shape == (1, 3, h, w)
    np.testing.assert_array_equal(required(view.rgb).numpy(), _chw(required(ref.outputs.cameras[c].rgb).numpy()))


def test_constructor_bound_buffers_allocate_nothing(monkeypatch):
  """`Renderer(output_arrays=...)`: a bound buffer is a view of the caller's array from the start
  (the constructor allocated a full-size own buffer first, then dropped it on binding)."""
  from raywarp.core.outputs import render_outputs as outputs

  allocated = []
  zeros = wp.zeros

  def recording_zeros(shape, dtype, **kw):
    allocated.append((tuple(np.atleast_1d(shape)), dtype))
    return zeros(shape, dtype=dtype, **kw)

  with wp.ScopedDevice("cpu"):
    rgb = wp.zeros((2, 4, 4), dtype=wp.uint32)
    monkeypatch.setattr(wp, "zeros", recording_zeros)
    b = outputs.Buffers(rw.Outputs.of("rgb"), nworld=2, sizes=[(4, 4)], device=wp.get_device("cpu"), arrays={"rgb": rgb})
  assert ((2, 16), wp.uint32) not in allocated
  assert b.flat["rgb"].ptr == rgb.ptr and required(b.outputs.rgb).shape == (2, 1, 4, 4)


# ---------------------------------------------------------------------------------------------
# Finite device gates for explicit HWC3 output: eager parity, graph reuse and owner lifetime.


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
@pytest.mark.parametrize("traversal", ["software", "optix"])
@pytest.mark.parametrize("order", ["tiled", "world"])
def test_hwc_device_motion_reset_graph_and_rebinding(
  traversal: Literal["software", "optix"], order: Literal["tiled", "world"]
) -> None:
  """Both raygen index orders preserve full RGB and detect replaced output bindings."""
  if traversal == "optix" and not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  xml = XML.replace(
    '<geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.2 0.5 0.9 1"/>',
    '<body pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.3" rgba="0.2 0.5 0.9 1"/></body>',
  )
  with wp.ScopedDevice("cuda:0"):
    prepare = functools.partial(
      make_scene,
      xml,
      nworld=3,
      cameras=["cam", "top"],
      cam_res=(16, 12),
      aa=rw.settings.AntiAliasing(spp=1),
      performance=rw.settings.Performance(traversal=traversal, render_order=order),
    )
    baseline = prepare(outputs=rw.Outputs(rgb=rw.settings.ColorOutput()))
    candidate = prepare(outputs=rw.Outputs(rgb=RGB3))
    try:
      assert baseline.r.features.traversal == candidate.r.features.traversal == traversal
      initial = baseline.d.qpos.numpy()
      captured = candidate.r.capture(sim=candidate.d, seed=0)
      for move in (0.0, 0.3, 0.0):
        positions = initial.copy()
        positions[1:, 0] += move
        for simulation in (baseline, candidate):
          simulation.d.qpos.assign(positions)
          mjw.forward(simulation.m, simulation.d)
        baseline.render()
        captured.launch()
        expected = required(baseline.out.rgb).numpy()[..., :3]
        np.testing.assert_array_equal(required(candidate.r.outputs.rgb).numpy(), expected)
      retained = required(candidate.r.outputs.rgb)
      replacement = wp.empty(retained.shape, dtype=wp.vec3ub, device=retained.device)
      candidate.r.bind_outputs(rgb=replacement)
      with pytest.raises(StaleBinding):
        captured.launch()
      recaptured = candidate.r.capture(sim=candidate.d, seed=0)
      recaptured.launch()
      np.testing.assert_array_equal(replacement.numpy(), expected)
      candidate.r.close()
      np.testing.assert_array_equal(retained.numpy(), expected)
    finally:
      baseline.r.close()
      candidate.r.close()


# ---------------------------------------------------------------------------------------------
# World-major RGB bytes preserve color, ownership and large-buffer addressing.


def test_color_layouts_are_output_word_bits() -> None:
  """Each color layout is one store of the output word (one table for every layout, no
  launch-parameter bytes per buffer): the packed RGBA by default, the RGB bytes or the planes for
  a three-channel color (no packed RGBA beside them), the buffers allocated accordingly."""
  default = rw.Outputs(rgb=rw.settings.ColorOutput())
  only = rw.Outputs(rgb=RGB3)
  assert layout_of(default).bits == ol.OUT_RGBA
  assert layout_of(rw.Outputs(rgb=PLANAR)).bits in (ol.OUT_CHW8, ol.OUT_CHW16)
  assert layout_of(only).bits == ol.OUT_RGB8
  cpu = wp.get_device("cpu")
  assert "rgb_hwc" not in Buffers(default, nworld=1, sizes=[(2, 2)], device=cpu).flat
  assert "rgb_hwc" in Buffers(only, nworld=1, sizes=[(2, 2)], device=cpu).flat
  with pytest.raises(ValueError, match="alpha=True needs channels=4"):
    rw.settings.ColorOutput(channels=3, alpha=True)


@pytest.mark.parametrize("sizes", [[(4, 3), (4, 3)], [(4, 3), (2, 2)]])
def test_rgb3_views_are_world_major_and_retain_bindings(sizes: list[tuple[int, int]]) -> None:
  """Unequal cameras pack without padding in each world's row, with caller aliases surviving
  renderer rebinding."""
  nworld = 3
  count = sum(w * h for w, h in sizes)
  caller = wp.zeros((nworld, count), dtype=wp.vec3ub, device="cpu")
  buffers = Buffers(rw.Outputs(rgb=RGB3), nworld=nworld, sizes=sizes, device=wp.get_device("cpu"), arrays={"rgb": caller})
  assert buffers.flat["rgb"].shape == (nworld, 1)
  old = buffers.outputs.cameras[1].rgb
  assert old is not None and old.shape == (nworld, 1, sizes[1][1], sizes[1][0])
  old.fill_(wp.vec3ub(1, 2, 3))
  actual = caller.numpy()
  start = sizes[0][0] * sizes[0][1]
  np.testing.assert_array_equal(actual[:, :start], 0)
  np.testing.assert_array_equal(actual[:, start:], np.broadcast_to([1, 2, 3], actual[:, start:].shape))
  replacement = wp.zeros_like(caller)
  assert buffers.bind({"rgb": replacement})
  old.fill_(wp.vec3ub(4, 5, 6))
  np.testing.assert_array_equal(replacement.numpy(), 0)
  assert not buffers.bind({"rgb": replacement})
  if sizes[0] == sizes[1]:
    assert buffers.outputs.rgb is not None
    assert buffers.outputs.rgb.shape == (nworld, 2, 3, 4)
  else:
    assert buffers.outputs.rgb is None


def test_rgb3_writer_matches_packed_bytes_for_every_world_and_camera() -> None:
  """The shared software/OptiX writer uses exactly the existing sRGB byte quantization: one
  write with the RGB bytes and the packed RGBA both in the word."""
  outputs = rw.Outputs(rgb=RGB3)

  @wp.kernel
  def exercise(out: OutputArrays):
    # Out: packed reference and world-major RGB bytes.
    world, pixel = wp.tid()
    camera_start = int(0)
    camera_pixels = int(5)
    if pixel >= 5:
      camera_start = 5
      camera_pixels = 3
    color = wp.vec3(float(world) * 0.13, float(pixel) * 0.12, 0.123)
    write_display(out, world, pixel, camera_start, camera_pixels, color, 1.0)

  buffers = Buffers(outputs, nworld=3, sizes=[(5, 1), (3, 1)], device=wp.get_device("cpu"))
  rgba = wp.zeros((3, 8), dtype=wp.uint32, device="cpu")
  table = OutputTable()
  table.rgb_hwc = wp.uint64(required(buffers.flat["rgb_hwc"].ptr))
  table.rgb = wp.uint64(required(rgba.ptr))
  out = OutputArrays()
  out.table = table
  out.rows, out.row = 3, 8
  out.bits = ol.OUT_RGB8 | ol.OUT_RGBA
  wp.launch(exercise, dim=(3, 8), inputs=[out], device="cpu")
  expected = rgba.numpy().view(np.uint8).reshape(3, 8, 4)[..., :3]
  for index, (first, pixels) in enumerate(((0, 5), (5, 3))):
    camera = buffers.outputs.cameras[index].rgb
    assert camera is not None
    np.testing.assert_array_equal(camera.numpy().reshape(3, pixels, 3), expected[:, first : first + pixels])


def test_rgb3_scene_matches_rgba_and_rebinding_invalidates_graphs(scene: MakeScene) -> None:
  """Real software shading keeps exact bytes, public RGBA and output-version semantics."""

  aa = rw.settings.AntiAliasing(spp=1)
  baseline = scene(
    XML, nworld=2, cameras=["cam", "top"], cam_res=(8, 6), aa=aa, outputs=rw.Outputs(rgb=rw.settings.ColorOutput())
  ).render()
  candidate = scene(XML, nworld=2, cameras=["cam", "top"], cam_res=(8, 6), aa=aa, outputs=rw.Outputs(rgb=RGB3)).render()
  assert baseline.out.rgb is not None and candidate.out.rgb is not None
  assert candidate.out.rgb.dtype == wp.vec3ub and candidate.out.rgb.shape == (2, 2, 6, 8)
  expected = baseline.out.rgb.numpy()[..., :3]
  np.testing.assert_array_equal(candidate.out.rgb.numpy(), expected)
  previous = candidate.out.rgb
  version = candidate.r.debug.graph_version()
  bound = wp.empty(previous.shape, dtype=wp.vec3ub, device=previous.device)
  candidate.r.bind_outputs(rgb=bound)
  assert candidate.r.debug.graph_version() != version
  candidate.render()
  np.testing.assert_array_equal(bound.numpy(), expected)
  candidate.r.close()
  np.testing.assert_array_equal(previous.numpy(), expected)


def test_target_sized_rgb3_alias_avoids_oversized_flat_extent() -> None:
  """A six-GiB logical RGB3 output aliases through two valid axes without allocating pixels."""
  from raywarp.core.outputs.render_outputs import _alias

  owner = wp.zeros(1, dtype=wp.uint8, device="cpu")
  source = wp.array(ptr=owner.ptr, dtype=wp.vec3ub, shape=(4, 512, 1024, 1024), device="cpu")
  with pytest.raises(ValueError, match="signed 32-bit"):
    source.reshape(source.size)
  alias = _alias(source, wp.vec3ub, (512, 4 * 1024**2), wp.get_device("cpu"))
  assert alias.ptr == source.ptr
  assert alias.strides == (4 * 1024**2 * 3, 3)
  assert alias.size * 3 == 6 * 1024**3
  assert alias._ref is not None


# ---------------------------------------------------------------------------------------------
# Three-channel storage: parsed from plain data, every reconstruction.


def test_three_channel_color_parses_from_plain_data() -> None:
  """A JSON / CLI request names the layout by its fields."""
  assert rw.settings.ColorOutput().channels == 4
  assert rw.Outputs.parse({"rgb": {"channels": 3, "planar": True}}).rgb == PLANAR


@pytest.mark.parametrize("color", [RGB3, PLANAR])
def test_three_channel_color_takes_supersampling(color: rw.settings.ColorOutput) -> None:
  """Supersampling writes through the layout's writer."""
  settings = rw.RenderSettings(outputs=rw.Outputs(rgb=color)).replace(**{"aa.spp": 2})
  assert settings.outputs.rgb == color


@pytest.mark.parametrize("dtype", ["uint8", "float16"])
def test_planes_exact_binding_and_retained_views(scene: MakeScene, dtype: Literal["uint8", "float16"]) -> None:
  """Both plane precisions hold exact values, allocate no RGBA and preserve caller ownership."""
  color = rw.settings.ColorOutput(channels=3, planar=True, dtype=dtype)
  aa = rw.settings.AntiAliasing(spp=1)
  full = scene(XML, nworld=2, cameras=["cam", "top"], cam_res=(12, 8), aa=aa, outputs=rw.Outputs.of("rgb")).render()
  only = scene(XML, nworld=2, cameras=["cam", "top"], cam_res=(12, 8), aa=aa, outputs=rw.Outputs(rgb=color)).render()
  assert only.r._buffers.flat["rgb"].shape == (2, 1)
  expected = _chw(required(full.out.rgb).numpy())
  previous = required(only.out.rgb)
  got = previous.numpy()
  if dtype == "float16":
    got = np.round(got.astype(np.float32) * 255.0)
  np.testing.assert_allclose(got, expected, atol=0 if dtype == "uint8" else 1)
  expected = previous.numpy().copy()
  key = "chw8" if dtype == "uint8" else "chw16"
  replacement = wp.empty_like(only.r._buffers.flat[key])
  only.r.bind_outputs(rgb=replacement)
  only.render()
  assert required(only.out.rgb).ptr == replacement.ptr
  np.testing.assert_array_equal(required(only.out.rgb).numpy(), expected)
  with pytest.raises(ValueError, match="cannot hold"):
    only.r.bind_outputs(rgb=full.r._buffers.flat["rgb"])
  only.r.close()
  np.testing.assert_array_equal(previous.numpy(), expected)
  np.testing.assert_array_equal(replacement.numpy(), expected.reshape(replacement.shape))


# ---------------------------------------------------------------------------------------------
# Valid multi-axis output bindings must not flatten through a signed32-bit extent.


@wp.func_native("return wp::uint64(wp::byte_offset(a, i, j));")
def _half_offset(
  # Data in: borrowed image descriptor and logical coordinates.
  a: wp.array2d[wp.float16],
  i: int,
  j: int,
) -> wp.uint64:
  """Native offset only: never read the synthetic image allocation."""
  return wp.uint64(0)


@wp.func_native("return wp::uint64(wp::byte_offset(a, i, j));")
def _packed_offset(
  # Data in: borrowed image descriptor and logical coordinates.
  a: wp.array2d[wp.uint32],
  i: int,
  j: int,
) -> wp.uint64:
  """Use the same native address arithmetic for packed RGBA output."""
  return wp.uint64(0)


@wp.kernel
def _offsets(
  # Data in: borrowed image descriptors and selected world indices.
  half: wp.array2d[wp.float16],
  packed: wp.array2d[wp.uint32],
  worlds: wp.array1d[wp.int32],
  # Out: tiny native byte-offset witnesses.
  output: wp.array2d[wp.uint64],
):
  """Only tiny worlds/output arrays are accessed; image descriptors are never dereferenced."""
  i = wp.tid()
  world = worlds[i]
  output[i, 0] = _half_offset(half, world, half.shape[1] - 1)
  output[i, 1] = _packed_offset(packed, world, packed.shape[1] - 1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_actual_alias_keeps_large_valid_shapes_and_native_offsets(device: str) -> None:
  """Exercise the production alias with target-sized descriptors over one borrowed byte."""
  if device != "cpu" and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  where = wp.get_device(device)
  owner = wp.zeros(1, dtype=wp.uint8, device=where)
  half = wp.array(ptr=owner.ptr, dtype=wp.float16, shape=(2048, 3, 1024, 1024), device=where)
  packed = wp.array(ptr=owner.ptr, dtype=wp.vec4ub, shape=(512, 4, 1024, 1024), device=where)
  # Preserve the original failure without attempting any image data operation/allocation.
  for source in (half, packed):
    with pytest.raises(ValueError, match="signed 32-bit"):
      source.reshape(source.size)
  flat_half = _alias(half, wp.float16, (512, 12 * 1024**2), where)
  flat_packed = _alias(packed, wp.uint32, (512, 4 * 1024**2), where)
  for source, alias in ((half, flat_half), (packed, flat_packed)):
    assert alias.ptr == source.ptr == owner.ptr
    assert alias.size == source.size
    assert tuple(alias.__ctype__().strides)[:2] == alias.strides
    assert all(stride is not None and 0 < stride < 2**31 for stride in alias.strides)
  worlds = wp.array([0, 1, 255, 511], dtype=wp.int32, device=where)
  output = wp.empty((4, 2), dtype=wp.uint64, device=where)
  wp.launch(_offsets, 4, inputs=[flat_half, flat_packed, worlds], outputs=[output], device=where)
  expected = np.array(
    [[(w * 12 * 1024**2 + 12 * 1024**2 - 1) * 2, (w * 4 * 1024**2 + 4 * 1024**2 - 1) * 4] for w in (0, 1, 255, 511)],
    dtype=np.uint64,
  )
  np.testing.assert_array_equal(output.numpy(), expected)
  assert expected[-1, 0] == half.size * 2 - 2 == 12 * 1024**3 - 2
  assert expected[-1, 1] == packed.size * 4 - 4 == 8 * 1024**3 - 4
  assert owner.capacity == 1


def test_bound_planes_render_dense_camera_writeback_lifetime_and_rebind(scene: MakeScene) -> None:
  """Real pixels and retained public aliases survive direct reinterpretation and rebinding."""
  with wp.ScopedDevice("cpu"):
    s = scene(
      SPHERE_XML,
      nworld=2,
      cam_res=(12, 8),
      cameras=["front"] * 2,
      outputs=rw.Outputs(rgb=PLANAR16),
      aa=rw.settings.AntiAliasing(spp=1),
    )
    s.render(seed=31)
    expected = required(s.out.rgb).numpy().copy()
    source = wp.empty((4, 3, 8, 12), dtype=wp.float16, device="cpu")
    owner = weakref.ref(source)
    s.r.bind_outputs(rgb=source)
    s.render(seed=31)
    dense = required(s.out.rgb)
    camera = required(s.out.cameras[1].rgb)
    assert dense.ptr == source.ptr
    np.testing.assert_array_equal(source.numpy(), expected)
    camera.fill_(0.25)
    modified = expected.copy()
    modified[1::2] = 0.25
    np.testing.assert_array_equal(dense.numpy(), modified)
    np.testing.assert_array_equal(source.numpy(), modified)
    replacement = wp.zeros((4, 3, 8, 12), dtype=wp.float16, device="cpu")
    s.r.bind_outputs(rgb=replacement)
    del source
    gc.collect()
    assert owner() is not None
    s.render(seed=31)
    np.testing.assert_array_equal(required(s.out.rgb).numpy(), expected)
    np.testing.assert_array_equal(dense.numpy(), modified)
    s.r.close()
    camera.fill_(0.5)
    modified[1::2] = 0.5
    np.testing.assert_array_equal(dense.numpy(), modified)
    del camera, dense
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("spp", [1, 2])
def test_half_albedo_and_normal_guides_build_and_match_float32(scene: MakeScene, spp: int) -> None:
  """float16 albedo + shading normal (no surface record: the guides' writer) failed to compile:
  the writer stored vec3 into the vec3h buffers."""
  aa = rw.settings.AntiAliasing(spp=spp)
  f32 = scene(XML, cam_res=(12, 8), aa=aa, outputs=rw.Outputs.of("albedo", "normal")).render()
  half = rw.Outputs(albedo=rw.settings.FloatOutput(dtype="float16"), normal=rw.settings.NormalOutput(dtype="float16"))
  f16 = scene(XML, cam_res=(12, 8), aa=aa, outputs=half).render()
  for name in ("albedo", "normal"):
    a, b = getattr(f32.out, name).numpy(), getattr(f16.out, name).numpy().astype(np.float32)
    np.testing.assert_allclose(b, a, atol=2e-3)


def test_output_struct_follows_rebound_buffers() -> None:
  """The cached output struct holds the buffers' addresses: after a re-binding it points at the new
  caller array (keyed by the buffers' addresses, not their views' ids, which a freed view's may
  share)."""
  from raywarp.core.outputs.render_outputs import OutputTables
  from raywarp.scene.model.types import LightId

  cpu = wp.get_device("cpu")
  ids = wp.zeros((1, 1), dtype=wp.int32, device=cpu)
  tables = OutputTables(
    segment_id=ids,
    flex_segment=0,
    world_skip=wp.zeros(2, dtype=bool, device=cpu),
    semantic_id=ids,
    semantic_flex=0,
    shadow_light=LightId(0),
  )
  buffers = Buffers(rw.Outputs(rgb=RGB3), nworld=2, sizes=[(4, 3)], device=cpu)
  field = buffers.layout.field("rgb_hwc")
  for _ in range(3):
    caller = wp.zeros((2, 12), dtype=wp.vec3ub, device=cpu)
    buffers.bind({"rgb": caller})
    assert getattr(buffers.struct(tables).table, field) == caller.ptr
  buffers.bind({})
  assert getattr(buffers.struct(tables).table, field) == buffers.flat["rgb_hwc"].ptr
