"""The output contract, per `ColorOutput` variant: one name per output (the one requested), one
world-major axis order, (nworld, ncam, ...), for the Warp arrays, the torch views and the
per-camera views.

`out.rgb` is (nworld, ncam, H, W) vec4ub RGBA or vec3ub RGB; the planar color is (nworld * ncam,
3, H, W) in Warp (four axes at most; row w * ncam + c) and (nworld, ncam, 3, H, W) in torch.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import MakeScene, MjScene

import raywarp as rw

# Two cameras looking at a red box from different sides: every (world, camera) image differs
# once the worlds' box colors differ.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    <geom name="box" type="box" size="0.3 0.3 0.3" pos="0 0 0.3" rgba="1 0 0 1"/>
    <camera name="a" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="b" pos="2.5 0 1.5" xyaxes="0 1 0 -0.5 0 0.86"/>
  </worldbody>
</mujoco>
"""
NWORLD, NCAM, W, H = 3, 2, 12, 8

# Per variant: (Warp shape, Warp dtype name, torch shape).
_CONTRACT = {
  "rgba": (rw.settings.ColorOutput(), (NWORLD, NCAM, H, W), "vec4ub", (NWORLD, NCAM, H, W, 4)),
  "rgb3": (rw.settings.ColorOutput(channels=3), (NWORLD, NCAM, H, W), "vec3ub", (NWORLD, NCAM, H, W, 3)),
  "planar": (rw.settings.ColorOutput(channels=3, planar=True), (NWORLD * NCAM, 3, H, W), "uint8", (NWORLD, NCAM, 3, H, W)),
  "planar_f16": (
    rw.settings.ColorOutput(channels=3, planar=True, dtype="float16"),
    (NWORLD * NCAM, 3, H, W),
    "float16",
    (NWORLD, NCAM, 3, H, W),
  ),
}


def _reference(scene: MakeScene) -> np.ndarray:
  """(nworld, ncam, H, W, 3) uint8 RGB of the default packed output, worlds tinted apart."""
  s = _render(scene, rw.settings.ColorOutput())
  rgb = s.r.outputs.rgb
  assert rgb is not None
  return rgb.numpy()[..., :3]


def _render(scene: MakeScene, color: rw.settings.ColorOutput) -> MjScene:
  """The scene with per-world box colors (red, green, blue), rendered once with `color`."""
  aa = rw.settings.AntiAliasing(spp=1)
  s = scene(_XML, nworld=NWORLD, cameras=["a", "b"], cam_res=(W, H), outputs=rw.Outputs(rgb=color), aa=aa)
  rgba = np.zeros((NWORLD, s.mjm.ngeom, 4), np.float32)
  rgba[:] = s.mjm.geom_rgba
  rgba[:, 1, :3] = np.eye(3)
  s.m.geom_rgba = wp.array(rgba, dtype=wp.vec4, device=s.m.geom_rgba.device)
  return s.render()


def _hwc(rgb: np.ndarray, planar: bool, dtype: str) -> np.ndarray:
  """(nworld, ncam, H, W, 3) uint8 of a torch / NumPy color in the contract's axis order."""
  if planar:
    rgb = np.moveaxis(rgb, 2, -1)
  rgb = rgb[..., :3]
  return np.round(rgb.astype(np.float32) * 255.0).astype(np.uint8) if dtype == "float16" else rgb


@pytest.mark.parametrize("variant", list(_CONTRACT))
def test_color_output_names_shapes_and_axis_order(scene: MakeScene, variant: str) -> None:
  """Each variant writes exactly "rgb", with its shape and dtype; the torch view and every
  per-camera view hold the same image per (world, camera) as the RGBA reference."""
  color, shape, dtype, _ = _CONTRACT[variant]
  ref = _reference(scene)
  assert len({ref[w, c].tobytes() for w in range(NWORLD) for c in range(NCAM)}) == NWORLD * NCAM
  out = _render(scene, color).r.outputs
  arrays = out.arrays()
  assert sorted(arrays) == ["rgb"]
  rgb = arrays["rgb"]
  assert (tuple(rgb.shape), rgb.dtype.__name__) == (shape, dtype)
  host = rgb.numpy()
  if color.planar:
    host = host.reshape(NWORLD, NCAM, 3, H, W)
  atol = 1 if color.dtype == "float16" else 0
  np.testing.assert_allclose(_hwc(host, color.planar, color.dtype), ref, atol=atol)
  for c, camera in enumerate(out.cameras):
    view = camera.rgb
    assert view is not None
    one = view.numpy().reshape(NWORLD, 1, 3, H, W) if color.planar else view.numpy()
    np.testing.assert_allclose(_hwc(one, color.planar, color.dtype)[:, 0], ref[:, c], atol=atol)


@pytest.mark.parametrize("variant", list(_CONTRACT))
def test_torch_views_are_world_major(scene: MakeScene, variant: str) -> None:
  """`torch()` gives every output (nworld, ncam, ...), the planar color split into its axes."""
  torch = pytest.importorskip("torch")
  color, _, _, torch_shape = _CONTRACT[variant]
  ref = _reference(scene)
  out = _render(scene, color).r.outputs
  t = out.torch()["rgb"]
  assert tuple(t.shape) == torch_shape
  host = t.cpu().to(torch.float32 if color.dtype == "float16" else torch.uint8).numpy()
  atol = 1 if color.dtype == "float16" else 0
  np.testing.assert_allclose(_hwc(host, color.planar, "float16" if color.dtype == "float16" else "uint8"), ref, atol=atol)


def test_color_layout_validation() -> None:
  """Every layout takes every reconstruction; invalid combinations raise."""
  rw.RenderSettings(outputs=rw.Outputs(rgb=rw.settings.ColorOutput(channels=3)), aa=rw.settings.AntiAliasing(spp=2))
  for bad in ({"channels": 4, "planar": True}, {"channels": 3, "alpha": True}, {"channels": 3, "dtype": "float16"}):
    with pytest.raises(ValueError, match=r"outputs\.rgb"):
      rw.settings.ColorOutput(**bad)  # pyrefly: ignore[bad-argument-type]


def test_flow_status_output_contract(scene: MakeScene) -> None:
  """`MotionOutput(status=True)` adds "flow_status": (nworld, ncam, H, W) uint8 data, the same in
  torch and per camera, bindable like every output; UNKNOWN on a first frame, VALID / MISS on a
  still second one (a box on a plane with background above)."""
  torch = pytest.importorskip("torch")
  outputs = rw.Outputs(depth=rw.settings.DepthOutput(), motion=rw.settings.MotionOutput(status=True))
  s = scene(_XML, nworld=NWORLD, cameras=["a", "b"], cam_res=(W, H), outputs=outputs)
  out = s.r.outputs
  assert sorted(out.arrays()) == ["depth", "flow_status", "motion"]
  assert out.color_spaces["flow_status"] == rw.scene.ColorSpace.DATA
  status = out.flow_status
  assert status is not None and (tuple(status.shape), status.dtype) == ((NWORLD, NCAM, H, W), wp.uint8)
  assert tuple(out.torch()["flow_status"].shape) == (NWORLD, NCAM, H, W)
  assert out.torch()["flow_status"].dtype == torch.uint8
  bound = wp.zeros((NWORLD * NCAM, H, W), dtype=wp.uint8, device=status.device)
  s.r.bind_outputs(flow_status=bound)
  s.render()
  F = rw.RenderOutputs.FlowStatus
  assert (bound.numpy() == F.UNKNOWN).all()
  s.render()
  host = bound.numpy().reshape(NWORLD, NCAM, H, W)
  depth = s.r.outputs.depth
  assert depth is not None
  np.testing.assert_array_equal(host, np.where(depth.numpy() > 0.0, int(F.VALID), int(F.MISS)))
  for c, camera in enumerate(s.r.outputs.cameras):
    view = camera.flow_status
    assert view is not None and tuple(view.shape) == (NWORLD, 1, H, W)
    np.testing.assert_array_equal(view.numpy()[:, 0], host[:, c])
