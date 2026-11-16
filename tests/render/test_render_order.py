"""The thread-to-pixel orders (`trace/rays/camera.thread_ids` / `thread_pixel`) are bijections with
their documented warp shapes."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import MjScene

import raywarp as rw
from raywarp._base.types import RenderOrder
from raywarp.core.trace.rays import camera

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _ids(
  # In:
  order: int,
  nworld: int,
  nray: int,
  # Out:
  out: wp.array2d[wp.vec2i],
):
  """(worldid, rayid) of every thread of an (nworld, nray) launch, by flat thread index."""
  i, j = wp.tid()
  w, r = camera.thread_ids(order, nworld, nray, i, j)
  out[i, j] = wp.vec2i(w, r)


@wp.kernel
def _pixels(
  # In:
  rows: int,
  width: int,
  height: int,
  # Out:
  out: wp.array1d[wp.vec2i],
):
  """(px, py) of every thread index of one camera image."""
  t = wp.tid()
  out[t] = camera.thread_pixel(rows, t, width, height)


def _thread_ids(order: RenderOrder, nworld: int, nray: int) -> np.ndarray:
  """(nworld * nray, 2) (worldid, rayid) by flat thread index."""
  out = wp.empty((nworld, nray), dtype=wp.vec2i, device="cpu")
  wp.launch(_ids, dim=(nworld, nray), inputs=[int(order), nworld, nray], outputs=[out], device="cpu")
  return out.numpy().reshape(-1, 2)


@pytest.mark.parametrize("order", [o for o in RenderOrder if o != RenderOrder.AUTO])
@pytest.mark.parametrize("nworld", [1, 5, 37])
def test_thread_ids_are_a_bijection(order: RenderOrder, nworld: int) -> None:
  ids = _thread_ids(order, nworld, 13)
  pairs = ids[:, 0] * 13 + ids[:, 1]
  assert sorted(pairs.tolist()) == list(range(nworld * 13))


@pytest.mark.parametrize(("order", "worlds"), [(RenderOrder.WORLD, 32), (RenderOrder.MIXED, 4)])
def test_world_groups_share_a_warp(order: RenderOrder, worlds: int) -> None:
  ids = _thread_ids(order, 64, 16)
  warp = ids[:32]
  assert sorted(set(warp[:, 0].tolist())) == list(range(worlds))
  assert sorted(set(warp[:, 1].tolist())) == list(range(32 // worlds))


@pytest.mark.parametrize(("width", "height"), [(16, 8), (37, 21)])
@pytest.mark.parametrize("rows", [camera.TILE_ROWS, camera.STORE_TILE_ROWS, 1])
def test_tiled_pixels_are_a_bijection_with_band_warps(width: int, height: int, rows: int) -> None:
  out = wp.empty(width * height, dtype=wp.vec2i, device="cpu")
  wp.launch(_pixels, dim=width * height, inputs=[rows, width, height], outputs=[out], device="cpu")
  p = out.numpy()
  assert sorted((p[:, 1] * width + p[:, 0]).tolist()) == list(range(width * height))
  # The first warp: 32 // rows columns x rows rows (8 x 4: 32-byte rows of 4-byte outputs).
  if width >= 32 // rows:
    assert set(map(tuple, p[:32].tolist())) == {(x, y) for x in range(32 // rows) for y in range(rows)}


_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.8 0.2 0.2 1"/>
    <camera name="a" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="b" pos="2 0 1" xyaxes="0 1 0 -0.4 0 0.9"/>
  </worldbody>
</mujoco>
"""


def _order(r: rw.Renderer) -> tuple[int, int]:
  """The frame's (RenderOrder, TILED band height) on the device."""
  o = r.context().cameras.order.numpy()[0]
  return int(o[0]), int(o[1])


def _frames(s: MjScene, n: int) -> list[int]:
  """The frame orders of `n` more renders."""
  out = []
  for _ in range(n):
    s.render()
    out.append(_order(s.r)[0])
  return out


@pytest.mark.parametrize("nworld", [2, 5])
def test_auto_order_follows_the_cameras(scene, nworld: int) -> None:
  """ "auto": MIXED while every world sees through the same cameras (with at least 4 worlds), TILED
  once one world's camera changes (its field of view here); the pixels are the same as a fixed
  order's."""
  kwargs = {"nworld": nworld, "cam_res": (37, 21), "outputs": rw.Outputs.of("rgb", "depth")}
  s = scene(_XML, per_world=rw.PerWorld(camera=["fovy"]), **kwargs)
  coherent = int(RenderOrder.MIXED) if nworld >= camera.MIXED_WORLDS else int(RenderOrder.TILED)
  s.render()
  assert _order(s.r) == (coherent, camera.TILE_ROWS)
  ref = scene(_XML, performance=rw.settings.Performance(render_order="tiled"), **kwargs).render()
  np.testing.assert_array_equal(s.out.rgb.numpy(), ref.out.rgb.numpy())
  fovy = s.r.per_world["camera.fovy"]
  wide = fovy.numpy()
  wide[nworld - 1, 1] += 1.0  # one world's second camera
  fovy.assign(wide)
  s.render()
  assert _order(s.r) == (int(RenderOrder.TILED), camera.TILE_ROWS)
  fovy.assign(np.ascontiguousarray(np.broadcast_to(wide[0], wide.shape)))
  s.render()
  assert _order(s.r)[0] == coherent


def test_auto_is_a_static_rule(scene) -> None:
  """ "auto" renders MIXED from the first frame with identical cameras and 4 worlds or more (no
  trial frames), and keeps it across settings changes."""
  s = scene(_XML, nworld=5, cam_res=(16, 8))
  assert _frames(s, 2) == [int(RenderOrder.MIXED)] * 2
  s.r.update_settings(exposure=rw.settings.Exposure(mode="manual", ev100=12.0))
  assert _frames(s, 1) == [int(RenderOrder.MIXED)]
  s.r.update_settings(**{"performance.render_order": "tiled"})
  assert _frames(s, 1) == [int(RenderOrder.TILED)]
  s.r.update_settings(**{"performance.render_order": "auto"})
  assert _frames(s, 1) == [int(RenderOrder.MIXED)]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="graph capture needs CUDA")
@pytest.mark.parametrize("traversal", ["software", "optix"])
def test_auto_under_graph_replay_keeps_the_pixels(scene, traversal: str) -> None:
  """Graph replays render "auto"'s MIXED (a device value the replay recomputes), with the pixels of
  a fixed TILED renderer."""
  kwargs = {"nworld": 5, "cam_res": (37, 21), "outputs": rw.Outputs.of("rgb", "depth")}
  with wp.ScopedDevice("cuda:0"):
    perf = {"performance.traversal": traversal}
    s = scene(_XML, **kwargs, **perf)
    if s.r.features.traversal != traversal:
      pytest.skip(f"{traversal} unavailable")
    ref = scene(_XML, **kwargs, **perf, **{"performance.render_order": "tiled"}).render()
    graph = s.r.capture(sim=s.d)
    orders = []
    for _ in range(3):
      graph.launch(s.d)
      orders.append(_order(s.r)[0])
      np.testing.assert_array_equal(s.out.rgb.numpy(), ref.out.rgb.numpy())
      np.testing.assert_array_equal(s.out.depth.numpy(), ref.out.depth.numpy())
  assert orders == [int(RenderOrder.MIXED)] * 3


def test_auto_order_follows_moved_free_cameras(scene) -> None:
  """A free camera posed differently in one world turns "auto" to TILED."""
  s = scene(_XML, nworld=4, cam_res=(16, 8), cameras=[rw.CameraSpec(fovy=45.0)])
  assert _frames(s, 1)[-1] == int(RenderOrder.MIXED)
  pose = s.r.camera_pose
  t = pose.numpy()
  t[3, 0, 0] += 0.01  # world 3's camera 1 cm along x
  pose.assign(t)
  s.render()
  assert _order(s.r)[0] == int(RenderOrder.TILED)


def test_fixed_orders_render_as_set(scene) -> None:
  """A fixed order renders from the first frame."""
  s = scene(_XML, nworld=5, cam_res=(16, 8), performance=rw.settings.Performance(render_order="mixed"))
  assert _frames(s, 3) == [int(RenderOrder.MIXED)] * 3


def test_store_bound_builds_tile_8x4(scene) -> None:
  """Depth-only and unlit builds use 4-row bands (8 x 4 warps), shaded ones 8-row bands."""
  s = scene(_XML, nworld=1, cam_res=(16, 8), outputs=rw.Outputs.of("depth")).render()
  assert _order(s.r) == (int(RenderOrder.TILED), camera.STORE_TILE_ROWS)
  s = scene(_XML, nworld=1, cam_res=(16, 8), surfaces=rw.settings.Surfaces(shading="unlit")).render()
  assert _order(s.r) == (int(RenderOrder.TILED), camera.STORE_TILE_ROWS)
