"""The API's use cases as contract tests: names, shapes and the world-major axis order of what a
user reads, for a MuJoCo Warp training loop, USD -> video and domain randomisation. (The Newton
sensor and Isaac camera drop-ins are pinned by tests/adapters/newton/test_newton_sensor.py and
tests/adapters/isaaclab/test_isaaclab.py.)"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import recipe, required

import raywarp as rw
import raywarp.integrations.mujoco

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.6 0.6 0.6 1"/>
    <body name="box" pos="0 0 0.3"><freejoint/><geom name="box" type="box" size="0.2 0.2 0.2" rgba="0.9 0.2 0.1 1"/></body>
    <camera name="front" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""
NWORLD, W, H = 3, 16, 12


def _mujoco() -> tuple[mujoco.MjModel, mjw.Model, mjw.Data]:
  mjm = mujoco.MjModel.from_xml_string(XML)
  return mjm, mjw.put_model(mjm), mjw.make_data(mjm, nworld=NWORLD)


@pytest.mark.parametrize("color", [rw.settings.ColorOutput(), rw.settings.ColorOutput(channels=3, planar=True)])
def test_mujoco_warp_training_loop(color: rw.settings.ColorOutput) -> None:
  """step -> render -> observations: every output (nworld, ncam, ...), named as requested; each
  frame sees the stepped state (the falling box's top face, read by the top camera's depth)."""
  with wp.ScopedDevice("cpu"):
    mjm, m, d = _mujoco()
    settings = recipe("fast").replace(outputs=rw.Outputs(rgb=color, depth=rw.settings.DepthOutput()), gi={"enabled": False})
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m), nworld=NWORLD, cameras=["front", "top"], resolution=(W, H), settings=settings
    )
    tops = []
    for _ in range(2):
      for _ in range(40):
        mjw.step(m, d)
      out = r.render(d)
      assert out.depth is not None
      # The top camera (z = 3, looking down) sees the box's top face (half size 0.2) at the center.
      top = d.xpos.numpy()[:, mjm.body("box").id, 2] + 0.2
      np.testing.assert_allclose(out.depth.numpy()[:, 1, H // 2, W // 2], 3.0 - top, atol=1e-4)
      tops.append(top)
    assert np.all(tops[1] < tops[0] - 0.01)  # the box fell between the frames
    assert sorted(out.arrays()) == ["depth", "rgb"]
    assert tuple(out.depth.shape) == (NWORLD, 2, H, W)
    assert out.rgb is not None
    rgb = out.rgb.numpy()
    if color.planar:
      assert rgb.shape == (NWORLD * 2, 3, H, W)
      rgb = np.moveaxis(rgb.reshape(NWORLD, 2, 3, H, W), 2, -1)
    assert rgb.shape[:4] == (NWORLD, 2, H, W)
    assert rgb[..., :3].max() > 0  # lit, not an untouched buffer
    assert r.explain().cost == "fast"
    r.close()


def test_usd_to_video(tmp_path) -> None:
  """A USD stage rendered frame after frame: each frame's color is (nworld, ncam, H, W) RGBA."""
  usd = pytest.importorskip("raywarp.formats.usd")
  from pxr import Usd, UsdGeom, UsdLux

  stage_path = tmp_path / "scene.usda"
  stage = Usd.Stage.CreateNew(str(stage_path))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  cube = UsdGeom.Cube.Define(stage, "/World/Cube")
  cube.CreateSizeAttr(0.5)
  sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
  sun.CreateIntensityAttr(3.0)
  sun.CreateNormalizeAttr(True)  # intensity: lux at normal incidence
  UsdGeom.XformCommonAPI(UsdGeom.Camera.Define(stage, "/World/Camera")).SetTranslate((0.0, 0.0, 4.0))
  stage.Save()
  scene = usd.scene(stage_path, device="cpu")
  settings = recipe("default").replace(gi={"enabled": False}, exposure={"mode": "manual", "ev100": 3.0})
  r = rw.Renderer(scene, nworld=1, cameras=None, resolution=(W, H), settings=settings)
  frames = []
  for _ in range(3):
    out = r.render()
    assert out.rgb is not None and out.rgb.dtype == wp.vec4ub
    frames.append(out.rgb.numpy()[0, 0].copy())  # world 0, camera 0: (H, W, 4)
  video = np.stack(frames)
  assert video.shape == (3, H, W, 4) and video[..., :3].max() > 0
  r.close()


def test_domain_randomisation_per_world() -> None:
  """Per-world colours through `r.per_world[...]`: world w's image is world w's colour."""
  with wp.ScopedDevice("cpu"):
    mjm, m, d = _mujoco()
    settings = recipe("fast").replace(outputs=rw.Outputs.of("rgb", "seg"), gi={"enabled": False})
    decl = rw.PerWorld(material=["color"])  # "instance.color": the geoms' rgba per world
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m),
      nworld=NWORLD,
      cameras=["top"],
      resolution=(W, H),
      settings=settings,
      per_world=decl,
    )
    assert list(r.per_world) == ["instance.color"]
    table = r.per_world["instance.color"]
    colors = table.numpy()
    colors[:, -1, :3] = np.eye(3)[:NWORLD]  # the box: red, green, blue per world
    table.assign(colors)
    mjw.forward(m, d)
    out = r.render(d)
    rgb, seg = required(out.rgb).numpy()[:, 0], required(out.seg).numpy()[:, 0]
    box = seg == seg[0, H // 2, W // 2]
    means = np.array([rgb[w][box[w]][:, :3].mean(axis=0) for w in range(NWORLD)])
    assert np.argmax(means, axis=1).tolist() == [0, 1, 2]
    r.close()
