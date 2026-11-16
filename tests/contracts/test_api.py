"""Renderer API: outputs, per-world overrides, settings updates, cameras, capture, determinism."""

from typing import Final

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import RECIPE_NAMES, recipe, required

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.settings.fields import OutputKind

XML = """
<mujoco>
  <worldbody>
    <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
    <light type="point" pos="0.5 -0.5 1.5" bulbradius="0.2" castshadow="true"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.8 0.3 0.2 1"/>
    <geom type="box" pos="0.6 0.2 0.2" size="0.2 0.2 0.2" rgba="0.2 0.4 0.9 1"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9" fovy="50"/>
    <camera name="side" pos="2 0 1" xyaxes="0 1 0 -0.4 0 0.9" fovy="60"/>
  </worldbody>
</mujoco>
"""
_ALL: Final = frozenset[OutputKind]({"rgb", "depth", "seg", "hdr", "albedo", "normal"})


def test_output_shapes_and_dtypes(scene):
  s = scene(XML, nworld=3, cam_res=(24, 16), outputs=rw.Outputs.of(*_ALL)).render()
  out = s.out
  shape = (3, 2, 16, 24)
  assert out.rgb.shape == shape and out.rgb.dtype == wp.vec4ub
  assert out.rgb.numpy().shape == (*shape, 4)
  assert out.depth.shape == shape and out.depth.dtype == wp.float32
  assert out.seg.shape == shape and out.seg.dtype == wp.int32
  for name in ("hdr", "albedo", "normal"):
    assert getattr(out, name).shape == shape and getattr(out, name).dtype == wp.vec3
  assert np.all(out.rgb.numpy()[..., 3] == 255)
  assert set(np.unique(s.seg())) <= {-1, 0, 1, 2} and (s.seg() == 1).any()
  only_depth = scene(XML, cam_res=(8, 8), outputs=rw.Outputs.of("depth"))
  assert only_depth.out.rgb is None and only_depth.out.seg is None and not only_depth.r.features.any_rgb


def test_per_world_override_wins_over_the_model(scene):
  s = scene(
    XML, nworld=2, cam_res=(24, 16), per_world=rw.PerWorld(material=["color"], camera=["fovy"], env=["exposure"])
  ).render()
  before = s.rgb().copy()
  color = s.r.per_world["instance.color"]
  assert color.shape == (2, s.mjm.ngeom)
  np.testing.assert_allclose(color.numpy()[1], s.mjm.geom_rgba, rtol=1e-6)
  assert s.r.per_world["instance.color"] is color  # one array per field
  # The override wins: replacing m.geom_rgba no longer changes the image.
  rgba = s.m.geom_rgba.numpy()
  rgba[..., :3] = 0.0
  s.m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  s.render()
  np.testing.assert_array_equal(s.rgb(), before)
  # Writing the override in place (world 1 only) changes world 1 only.
  values = color.numpy()
  values[1, 1, :3] = (0.1, 0.9, 0.1)
  color.assign(values)
  s.render()
  np.testing.assert_array_equal(s.rgb()[0], before[0])
  assert np.abs(s.rgb()[1] - before[1]).max() > 0.2
  fovy = s.r.per_world["camera.fovy"]
  assert fovy.shape == (2, 2)
  exposure = s.r.per_world["exposure"]
  assert exposure.shape == (2,)


def test_update_settings_reallocates_outputs(scene):
  s = scene(XML, cam_res=(16, 12)).render()
  old = s.out
  assert old.depth is None
  s.r.update_settings(outputs=rw.Outputs.of("rgb", "depth"))
  assert s.out is not old and s.out.depth is not None
  s.render()
  assert (s.depth() > 0).any()


def test_failed_update_settings_keeps_the_previous_settings(scene):
  # The change configures nothing (direct light of an unlit surface): the renderer stays on the old
  # settings and buffers and still renders.
  s = scene(XML, cam_res=(16, 12)).render()
  settings, out, features = s.r.settings, s.out, s.r.features
  bad = {"outputs": rw.Outputs.of("rgb", "depth", "direct"), "surfaces.shading": "unlit"}
  with pytest.raises(ValueError, match="unlit"):
    s.r.update_settings(**bad)
  assert s.r.settings is settings and s.out is out and s.r.features == features
  s.render()
  assert s.out.depth is None


def test_update_settings_rebuilds_only_what_the_field_names(scene):
  """A "gi.*" volume field re-bakes the probe volume on the next render; a runtime field
  without a rebuild keeps it; a dotted key and a group update the same settings."""
  s = scene(XML, cam_res=(8, 8), gi=rw.settings.GI(enabled=True, max_probes=64, rays=32, bake_updates=2)).render()
  vol = s.r.debug.gi_volume
  assert vol is not None
  s.r.update_settings(**{"performance.rebuild_every": 32})
  s.render()
  assert s.r.debug.gi_volume is vol and s.r.settings.performance.rebuild_every == 32
  s.r.update_settings(**{"gi.spacing": 0.25})
  assert s.r.settings.gi.spacing == 0.25 and s.r.settings.gi.enabled
  s.render()
  assert s.r.debug.gi_volume is not None and s.r.debug.gi_volume is not vol
  with pytest.raises(ValueError, match="unknown setting 'gi_spacing'"):
    s.r.update_settings(gi_spacing=0.5)


def test_alpha_output_is_coverage(scene):
  xml = XML.replace('<geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>', "")
  s = scene(xml, cam_res=(24, 16), outputs=rw.Outputs.of("rgb", "seg"), aa=rw.settings.AntiAliasing(spp=2)).render()
  opaque = s.out.rgb.numpy()
  assert np.all(opaque[..., 3] == 255)
  s.r.update_settings(outputs=rw.Outputs.of("seg", rgb=rw.settings.ColorOutput(alpha=True)))
  s.render()
  rgba = s.out.rgb.numpy()
  alpha, seg = rgba[..., 3], s.out.seg.numpy()
  # Segmentation is the center hit; alpha is coverage of the independent 2 x 2 color grid.
  center = scene(xml, cam_res=(24, 16), outputs=rw.Outputs.of("seg"), aa=rw.settings.AntiAliasing(spp=1)).render()
  np.testing.assert_array_equal(seg, center.out.seg.numpy())
  assert (alpha == 0).any() and (alpha == 255).any()
  assert ((alpha > 0) & (alpha < 255)).any()  # silhouettes: partial coverage of the 2 x 2 samples
  np.testing.assert_array_equal(rgba[..., :3], opaque[..., :3])  # the color is unchanged


def test_bind_outputs_renders_into_caller_arrays(scene):
  s = scene(XML, nworld=2, cam_res=(16, 12), outputs=rw.Outputs.of("rgb", "depth")).render()
  ref_rgb, ref_depth = s.out.rgb.numpy().copy(), s.out.depth.numpy().copy()
  rgb = wp.zeros((4, 12, 16), dtype=wp.uint32)  # (nworld * ncam, H, W)
  depth = wp.zeros((2, 2, 12, 16), dtype=wp.float32)
  s.r.bind_outputs(rgb=rgb, depth=depth)
  version = s.r.debug.graph_version()
  s.r.bind_outputs(rgb=rgb, depth=depth)  # the same arrays: a no-op
  assert s.r.debug.graph_version() == version
  assert s.out.rgb.ptr == rgb.ptr and s.out.depth.ptr == depth.ptr
  s.render()
  np.testing.assert_array_equal(rgb.numpy().view(np.uint8).reshape(ref_rgb.shape), ref_rgb)
  np.testing.assert_array_equal(depth.numpy(), ref_depth)
  with pytest.raises(ValueError, match="off"):
    s.r.bind_outputs(seg=wp.zeros((4, 12, 16), dtype=wp.int32))
  with pytest.raises(ValueError, match="elements"):
    s.r.bind_outputs(rgb=wp.zeros((3, 12, 16), dtype=wp.uint32))
  with pytest.raises(TypeError, match="Warp arrays"):  # not an AttributeError on the NumPy array
    s.r.bind_outputs(rgb=np.zeros((4, 12, 16), np.uint32))
  s.r.bind_outputs()  # back to own buffers
  assert s.out.rgb.ptr != rgb.ptr


def test_inactive_worlds_keep_outputs(scene):
  s = scene(XML, nworld=3, cam_res=(16, 12), outputs=rw.Outputs.of("rgb", "depth")).render()
  ref = s.out.depth.numpy().copy()
  s.out.depth.fill_(-1.0)
  s.r.render(s.d, active=wp.array([0, 2], dtype=wp.int32))
  depth = s.out.depth.numpy()
  assert np.all(depth[1] == -1.0)
  np.testing.assert_array_equal(depth[[0, 2]], ref[[0, 2]])
  with pytest.raises(ValueError, match="int32"):
    s.r.render(s.d, active=wp.zeros(2, dtype=bool))
  s.render()
  np.testing.assert_array_equal(s.out.depth.numpy(), ref)


def test_free_camera_matches_mounted_camera(scene):
  s = scene(XML, cam_res=(24, 16), cameras=["front"]).render()
  xpos = s.d.cam_xpos.numpy()[0, 0]
  xmat = s.d.cam_xmat.numpy()[0, 0]
  free = rw.CameraSpec(fovy=50.0, zfar=s.scene.cameras["front"].zfar)
  r = rw.Renderer(s.scene, nworld=1, cameras=[free], resolution=(24, 16), settings=s.r.settings)
  quat = wp.quat_from_matrix(wp.mat33(*xmat.ravel().tolist()))
  r.camera_pose.assign(np.array([[[*xpos, *quat]]], dtype=np.float32))
  r.render(s.d)
  free_rgb = required(r.outputs.rgb).numpy()[..., :3].astype(int)
  # Same pose up to the quaternion round trip: within rounding of the mounted camera's image.
  diff = np.abs(free_rgb - s.out.rgb.numpy()[..., :3].astype(int)).max(-1)
  assert np.mean(diff <= 2) > 0.98


def test_mounted_camera_with_local_offset(scene):
  s = scene(XML, cam_res=(16, 12), cameras=["front"]).render()
  spec = rw.CameraSpec(fovy=50.0, mount="front", local=(np.zeros(3), np.eye(3)))
  r = rw.Renderer(s.scene, nworld=1, cameras=[spec], resolution=(16, 12), settings=s.r.settings)
  r.render(s.d)
  np.testing.assert_array_equal(required(r.outputs.rgb).numpy(), s.out.rgb.numpy())


def test_render_is_deterministic(scene):
  s = scene(XML, nworld=2, cam_res=(16, 12), shadows=rw.settings.Shadows(samples=4), outputs=rw.Outputs.of("rgb", "depth"))
  a = s.render(seed=3).out.rgb.numpy().copy()
  b = s.render(seed=3).out.rgb.numpy().copy()
  np.testing.assert_array_equal(a, b)
  other = scene(
    XML, nworld=2, cam_res=(16, 12), shadows=rw.settings.Shadows(samples=4), outputs=rw.Outputs.of("rgb", "depth")
  ).render(seed=3)
  np.testing.assert_array_equal(other.out.rgb.numpy(), a)


def test_capture_needs_cuda(scene):
  s = scene(XML, cam_res=(8, 8))
  if wp.get_device().is_cuda:
    pytest.skip("CPU-only check")
  with pytest.raises(RuntimeError, match="CUDA device"):
    s.r.capture(sim=s.d)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs need a CUDA device")
def test_capture_and_stale_binding():
  import mujoco
  import mujoco_warp as mjw

  with wp.ScopedDevice("cuda:0"):
    mjm = mujoco.MjModel.from_xml_string(XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=2)
    r = rw.Renderer(
      rw.integrations.mujoco.scene(mjm, m),
      nworld=2,
      resolution=(16, 12),
      settings=rw.RenderSettings(performance=rw.settings.Performance(traversal="software")),
    )
    graph = r.capture(lambda: mjw.forward(m, d), sim=d)
    eager = required(r.render(d).rgb).numpy().copy()
    required(r.outputs.rgb).zero_()
    graph.launch(d)
    np.testing.assert_array_equal(required(r.outputs.rgb).numpy(), eager)
    with pytest.raises(rw.StaleBinding, match=r"instance\.pos"):
      graph.launch(mjw.put_data(mjm, mjd, nworld=2))
    r.update_settings(exposure=rw.settings.Exposure(mode="manual", ev100=12.0))
    with pytest.raises(rw.StaleBinding, match="changed since the capture"):
      graph.launch(d)


def test_slot_vocabulary():
  """Kernel fields are the slot names with "." -> "_"; the MuJoCo table maps only core slots."""
  from raywarp.core.pipeline.gpu_scene.binding import FIXED_SLOTS, SLOTS
  from raywarp.integrations.mujoco.binding import DATA_SLOTS, MODEL_SLOTS

  assert all(field == slot.replace(".", "_") for slot, field in SLOTS.items())
  assert FIXED_SLOTS <= SLOTS.keys()
  assert MODEL_SLOTS.keys() | DATA_SLOTS.keys() <= SLOTS.keys()


_BATCH_XML: Final = XML.replace(
  '<geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.8 0.3 0.2 1"/>',
  '<body name="ball" pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.3" rgba="0.8 0.3 0.2 1"/></body>',
)
# Recipes whose sampling is decorrelated per world (a world's noise depends on its index): there a
# world renders as alone only at index 0.
_PER_WORLD_NOISE: Final = frozenset({"quality", "reference"})


@pytest.mark.parametrize("recipe_name", sorted(RECIPE_NAMES))
def test_a_world_of_a_batch_renders_as_a_renderer_of_it_alone(scene, recipe_name: str) -> None:
  """1 vs N worlds: the last world of a batch of 3 (its ball moved) renders bit for bit as a
  one-world renderer of that state, and world 0 as one of world 0's (every recipe; with per-world
  noise, world 0 only). Depth may differ by 1 ulp (the batch's structure build rounds a hit's
  distance differently on some GPUs: L4); labels and colours are exact."""
  p = recipe(recipe_name)
  kw = {**{k: v for k, v in vars(p).items() if k != "outputs"}, "gi": rw.settings.GI(enabled=False)}
  kw["outputs"] = rw.Outputs.of("rgb", "depth", "seg")
  batch = scene(_BATCH_XML, nworld=3, cam_res=(16, 12), **kw)
  qpos = batch.d.qpos.numpy()
  qpos[2, 0] += 0.5  # world 2's ball, half a metre along x
  batch.d.qpos.assign(qpos)
  mjw.kinematics(batch.m, batch.d)
  batch.render(seed=5)
  for world in (0, 2):
    alone = scene(_BATCH_XML, cam_res=(16, 12), **kw)
    one = alone.d.qpos.numpy()
    one[0] = qpos[world]
    alone.d.qpos.assign(one)
    mjw.kinematics(alone.m, alone.d)
    alone.render(seed=5)
    np.testing.assert_array_max_ulp(batch.depth()[world], alone.depth()[0], maxulp=1)
    np.testing.assert_array_equal(batch.seg()[world], alone.seg()[0], err_msg=f"world {world}")
    if world == 0 or recipe_name not in _PER_WORLD_NOISE:
      np.testing.assert_array_equal(batch.rgb()[world], alone.rgb()[0], err_msg=f"world {world}")
  assert not np.array_equal(batch.depth()[0], batch.depth()[2])  # the worlds differ
