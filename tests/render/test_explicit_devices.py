"""Every allocation and launch names its device (code-quality rule 17): the scene builds
and caches below run under `conftest.no_default_device`, where Warp's default device is a stand-in
that fails when used, so a forgotten `device=` fails on a CPU-only machine too."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import no_default_device
from test_deformables import _PARTS, _VERT

import raywarp as rw
from raywarp.core.pipeline.gpu_scene.binding import InstancePoses, StaticPoses
from raywarp.core.pipeline.gpu_scene.deformables import Deformables
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.shade.lights.punctual import Light
from raywarp.core.trace.accel.deformable_bvh import DeformableBvhs
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model.lights import EnvironmentLight as LightsEnvironment
from raywarp.scene.model.types import Wrap

CPU = wp.get_device("cpu")

# Textures (2D and skybox), a height field or a flex (MuJoCo Warp has no flex-hfield
# collision), a mesh, lights and a camera: every part of the MuJoCo scene build.
_XML = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".8 .8 .8" width="16" height="16"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="16" height="16"/>
    <material name="floor" texture="grid" texrepeat="2 2"/>
    <material name="cloth" rgba="0.8 0.2 0.2 1"/>
    <hfield name="h" nrow="4" ncol="4" size="1 1 0.2 0.1"/>
    <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
  </asset>
  <worldbody>
    <light type="directional" dir="0 0 -1"/>
    <light type="spot" pos="0 0 2" dir="0 0 -1" cutoff="30"/>
    <geom type="plane" size="2 2 0.1" material="floor"/>
    {hfield}
    <geom type="mesh" mesh="tet" pos="0 0 0.5"/>
    <geom type="sphere" size="0.2" pos="0.5 0 0.3"/>
    {flex}
    <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
  </worldbody>
</mujoco>
"""


_HFIELD = '<geom type="hfield" hfield="h" pos="2 0 0"/>'
_FLEX = '<flexcomp name="f" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 1" dim="2" material="cloth"/>'


@pytest.mark.parametrize(("hfield", "flex"), [(_HFIELD, ""), ("", _FLEX)], ids=["hfield", "flex"])
def test_mujoco_scene_is_built_on_the_model_device(monkeypatch: pytest.MonkeyPatch, hfield: str, flex: str):
  """`rw.integrations.mujoco.scene` builds on the device of `m` without a `wp.ScopedDevice` of it
  (that scope set the process-wide default device: a race with other threads)."""
  mjm = mujoco.MjModel.from_xml_string(_XML.format(hfield=hfield, flex=flex))
  with wp.ScopedDevice(CPU):
    m = mjw.put_model(mjm)

  def scoped(*_args: object) -> None:
    raise AssertionError("the scene build entered wp.ScopedDevice")

  monkeypatch.setattr(wp, "ScopedDevice", scoped)
  with no_default_device():
    scene = rw.integrations.mujoco.scene(mjm, m)
  assert scene.device == CPU and scene.texture_array.device == CPU


def test_static_poses_are_allocated_on_the_given_device():
  """`Binding.allocate` takes the renderer's device (it allocated on the current one)."""
  poses = InstancePoses(np.zeros((3, 3), np.float32), np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)))
  with no_default_device():
    buffers = StaticPoses(poses).allocate(2, np.array([0, 2], np.int32), device=CPU)
  assert buffers.pos.device == CPU and buffers.pos.shape == (2, 2) and buffers.rot.device == CPU


def test_light_grid_cache_loads_on_the_given_device():
  """A cached light grid loads on the scene's device (it loaded on the current one)."""
  stored = {
    "origin": np.zeros(3, np.float32),
    "cell": np.array(0.5, np.float32),
    "dims": np.ones(3, np.int32),
    "start": np.zeros(3, np.int32),
    "lights": np.zeros(1, np.int32),
    "snapshot": wp.zeros((1, 1), dtype=Light, device=CPU).numpy(),
    "dynamic_mask": np.zeros(1, np.int32),
    "far": np.zeros((0, 3), np.float32),
  }
  with no_default_device():
    grid = light_grid.from_arrays(stored, device=CPU)
  assert grid.start.device == CPU and grid.lights.device == CPU
  assert grid.snapshot.device == CPU and grid.valid.device == CPU
  assert grid.far.device == CPU and grid.dynamic_mask.device == CPU


def test_deformable_bvhs_are_built_on_the_given_device():
  """The deformables' BVHs are built on the scene's device, named by the caller."""
  d = Deformables.create(_PARTS, _VERT, device=CPU)
  with no_default_device():
    bvhs = DeformableBvhs(d, nworld=2, device=CPU)
  assert bvhs.ids.device == CPU and bvhs.group_root.shape == (2, len(_PARTS))


def _no_scope(monkeypatch: pytest.MonkeyPatch) -> None:
  """`wp.ScopedDevice` raises: the renderer entered one around its work (Warp's process-wide
  default device, a race with other threads)."""

  def scoped(*_args: object) -> None:
    raise AssertionError("the renderer entered wp.ScopedDevice")

  monkeypatch.setattr(wp, "ScopedDevice", scoped)


def test_a_host_scene_uploads_and_renders_without_the_default_device(monkeypatch: pytest.MonkeyPatch):
  """Upload, Renderer construction, frames (the motion output's history, per-world fields,
  settings and camera changes) name their device: no `wp.ScopedDevice`, no default-device
  allocation."""
  b = rw.SceneBuilder()
  mat = b.add_material(rw.Material(base_color=(0.8, 0.3, 0.2), coat_weight=0.5, emission_color=(1.0, 0.5, 0.0)))
  b.add_instance("sphere", size=(0.3, 0.0, 0.0), material=mat)
  b.add_instance("box", size=(2.0, 2.0, 0.05), xform=np.c_[np.eye(3), [0.0, 0.0, -0.4]])
  # A dome tilted about +x: the upload resamples it (`latlong.rotate_latlong`) on its device.
  sky = np.ones((8, 16, 4), np.float32)
  sky[:4, :, :3] = 2.0
  texture = b.add_texture(sky, colorspace=ColorSpace.LIN_REC709, wrap=(Wrap.REPEAT, Wrap.CLAMP))
  c, s = np.cos(0.4), np.sin(0.4)
  tilt = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
  b.add_environment(LightsEnvironment(texture=texture, intensity=1.0, rotation=tilt))
  host = b.build().with_lighting(rw.Lighting.studio())
  settings = rw.RenderSettings().replace(
    outputs=rw.Outputs.of("rgb", "depth", "hdr", "motion"),
    gi=rw.settings.GI(enabled=False),
  )
  _no_scope(monkeypatch)
  with no_default_device():
    scene = rw.upload(host, device=CPU)
    per_world = rw.PerWorld(material=["base_color", "coat_weight", "emission_luminance"], env=["exposure"])
    r = rw.Renderer(
      scene, nworld=2, cameras=[rw.CameraSpec(fovy=50.0)], resolution=(16, 12), settings=settings, per_world=per_world
    )
    r.camera_pose.assign(np.array([[[0.0, -2.0, 0.5, 0.7071068, 0.0, 0.0, 0.7071068]]] * 2, np.float32))
    r.per_world["material.coat_weight"].assign(np.array([[0.5], [1.0]], np.float32))
    for _ in range(2):
      out = r.render()
    r.update_settings(outputs=rw.Outputs.of("rgb", "seg"))
    r.set_clip(znear=0.05)
    r.reset_history()
    out = r.render()
  assert out.rgb is not None and out.rgb.device == CPU
  assert np.isfinite(out.rgb.numpy()).all()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs")
def test_a_graph_capture_names_its_device(monkeypatch: pytest.MonkeyPatch):
  """Capture and replay of a render on CUDA without the default device (the capture's own scope
  covers only the caller's step: `graph.capture`)."""
  b = rw.SceneBuilder()
  b.add_instance("sphere", size=(0.3, 0.0, 0.0), material=b.add_material(rw.Material()))
  cuda = wp.get_device("cuda:0")
  scene = rw.upload(b.build().with_lighting(rw.Lighting.studio()), device=cuda)
  r = rw.Renderer(scene, nworld=2, cameras=[rw.CameraSpec()], resolution=(16, 12))
  with no_default_device():
    g = r.capture()
    g.launch()
  wp.synchronize_device(cuda)
  assert r.outputs.rgb is not None and r.outputs.rgb.device == cuda
