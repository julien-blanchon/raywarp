"""Isaac Lab adapter on CPU without Isaac Lab: Isaac's renderer calls driven with fakes of its
interface objects (`isaaclab_fake`), on a two-environment Newton model."""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
import pytest
import warp as wp
from conftest import UNLIT_FAST_CHANGES, required, whole_table

newton = pytest.importorskip("newton")

from fixtures import conformance
from fixtures import isaaclab_fake as fk

import raywarp as rw
from raywarp.core.pipeline.upload.build import upload
from raywarp.integrations.isaaclab import Cameras, contract
from raywarp.integrations.isaaclab.materials import MaterialLayout
from raywarp.integrations.isaaclab.scalars import SCALAR_CHANNELS, MaterialWriters, ScalarWriter, scalar_table
from raywarp.integrations.isaaclab.session import CameraSession

W, H = 32, 24


ALL = ["rgb", "depth", "distance_to_image_plane", "distance_to_camera", "albedo", "normals", "rgb_hdr"]


SEG = ["instance_segmentation", "semantic_segmentation"]


EYE = (-3.0, 0.0, 0.5)  # looking along +X at the box (front face at x = -0.3)


DEFAULT_SPAWN = fk.Spawn()


# The renderer config of the frames whose outputs, resets and calibration are checked: Isaac's
# defaults without probe GI (the default bakes it over the ground plane: 30k probes, ~80 s on
# the CPU; GI has its own tests).
NO_GI: Final = {"gi.enabled": False}


def _frame(names, config=None, sim=None, n=2, spawn=DEFAULT_SPAWN, scene_clip=None):
  """(Cameras, session, CameraData) after one frame of `names` seen from EYE in every env
  (`spawn` None: a camera without spawn config, so the scene's default clip planes, `scene_clip`
  when given)."""
  if sim is None:
    model, state = fk.boxes(n)
    q = required(state.body_q).numpy()
    q[1, 0] += 0.5  # env 1's box 0.5 m further away
    required(state.body_q).assign(q)
    sim = fk.Simulation(model, state)
  cams = Cameras(config or fk.Config(settings=NO_GI))
  cams.initialize(sim)
  if scene_clip is not None:
    cams.scene.clip = scene_clip
  spec = fk.RenderSpec(fk.CameraCfg(W, H, list(names), spawn=spawn), n)
  session = cams.create_render_data(spec)
  data = fk.CameraData.allocate(list(names), n, W, H)
  cams.set_outputs(session, data.output)
  pos, rot = fk.poses([EYE] * n)
  cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(n, W, H)))
  cams.update_transforms()
  cams.render(session)
  cams.read_output(session, data)
  return cams, session, data


def _out(data, name):
  return data.output[name].warp.numpy()


def test_outputs_follow_isaac_lab_contract():
  """Every output lands in Isaac's (N, H, W, C) buffers with Isaac's values: planar / radial depth
  (inf on a miss: Isaac RTX), rgb through Isaac's view of rgba, packed sRGB albedo, poses per
  env."""
  _, session, data = _frame(ALL)
  depth = _out(data, "depth")[..., 0]
  np.testing.assert_allclose(depth[:, H // 2, W // 2], [2.7, 3.2], rtol=1e-4)  # env 1's box moved
  np.testing.assert_array_equal(depth, _out(data, "distance_to_image_plane")[..., 0])
  assert np.isinf(depth[:, 0, :]).all()  # sky row: miss
  radial = _out(data, "distance_to_camera")[..., 0]
  # Radial / planar = |ray| on the unit-depth plane (fx = fy, cx = W / 2: Isaac's K).
  fx = 0.5 * W / math.tan(math.radians(30.0))
  y, x = H // 2 + 5, W // 2 + 7
  scale = math.sqrt(1.0 + ((x + 0.5 - W / 2) / fx) ** 2 + ((y + 0.5 - H / 2) / fx) ** 2)
  np.testing.assert_allclose(radial[0, y, x], depth[0, y, x] * scale, rtol=1e-5)
  rgba, rgb = _out(data, "rgba"), _out(data, "rgb")
  np.testing.assert_array_equal(rgb, rgba[..., :3])
  assert (rgba[..., 3] == 255).all() and rgba[0, H // 2, W // 2, 0] > rgba[0, H // 2, W // 2, 2]  # a red box
  # Newton's sRGB shape color (0.8, 0.2, 0.1) back as 8-bit sRGB.
  np.testing.assert_array_equal(_out(data, "albedo")[:, H // 2, W // 2], [[204, 51, 26, 255]] * 2)
  normal = _out(data, "normals")[0, H // 2, W // 2]
  np.testing.assert_allclose(normal, [-1.0, 0.0, 0.0], atol=1e-5)  # the face looking at the camera
  assert _out(data, "rgb_hdr")[0, H // 2, W // 2].max() > 0.0
  assert required(session.renderer.outputs.rgb).ptr == data.output["rgba"].warp.ptr  # rendered in place


def test_supported_outputs_are_isaac_layouts():
  """The contract lists Isaac's channel counts and dtypes; unknown outputs are refused."""
  assert contract.OUTPUTS["rgba"] == contract.Channels(4, wp.uint8)
  assert contract.OUTPUTS["distance_to_camera"] == contract.Channels(1, wp.float32)
  assert contract.OUTPUTS["instance_segmentation"] == contract.Channels(1, wp.int32)
  with pytest.raises(ValueError, match="motion_vectors"):
    contract.check_names(["rgb", "motion_vectors"])


def test_segmentation_ids_and_labels():
  """Isaac's Newton Warp segmentation contract (`isaaclab_newton/renderers/segmentation.py`): the
  env boxes are two instances (nearest labelled ancestor, ids from 2 in shape order) of one class
  (2); the unlabelled ground is 1, misses 0; `idToLabels` / `idToSemantics` keyed by int."""
  model, state = fk.boxes(2)
  box = {"class": ["box"], "material": ["wood"]}
  labels = {"/World/envs/env_0/Box": box, "/World/envs/env_1/Box": box}
  _, _, data = _frame(SEG, sim=fk.Simulation(model, state, labels=labels))
  inst, sem = _out(data, SEG[0])[..., 0], _out(data, SEG[1])[..., 0]
  assert (inst[:, 0] == 0).all() and (sem[:, 0] == 0).all()  # sky: background
  np.testing.assert_array_equal(inst[:, H // 2, W // 2], [2, 3])
  np.testing.assert_array_equal(sem[:, H // 2, W // 2], [2, 2])
  assert (inst[:, -1, W // 2] == 1).all() and (sem[:, -1, W // 2] == 1).all()  # the ground
  payload = {"class": "box", "material": "wood"}
  info, semantic = data.info["instance_segmentation"], data.info["semantic_segmentation"]
  assert isinstance(info, dict) and isinstance(semantic, dict)
  assert info["idToLabels"] == {0: "BACKGROUND", 1: "UNLABELLED", 2: "/World/envs/env_0/Box", 3: "/World/envs/env_1/Box"}
  assert info["idToSemantics"][3] == payload and info["idToSemantics"][1] == {"class": "UNLABELLED"}
  assert semantic["idToLabels"] == {
    0: {"class": "BACKGROUND"},
    1: {"class": "UNLABELLED"},
    2: payload,
  }


@pytest.mark.parametrize(("clipping", "miss"), [("none", math.inf), ("zero", 0.0), ("max", 50.0)])
def test_depth_clipping_behavior(clipping, miss):
  """Isaac's depth_clipping_behavior: misses are inf, zero, or exactly the far clip, planar and
  radial alike."""
  config = fk.Config(depth_clipping_behavior=clipping)
  names = ["depth", "distance_to_camera"]
  _, _, data = _frame(names, config=config, spawn=fk.Spawn(clipping_range=(0.01, 50.0)))
  for name in names:
    assert (_out(data, name)[:, 0, :, 0] == miss).all(), name


def test_max_clipping_without_spawn_keeps_the_scene_far_clip_on_radial_misses():
  """ "max" with no spawn (zfar from the scene's default): radial misses are the planar misses, the
  far clip the renderer writes, not scaled (they were: the radial pass took inf as the miss)."""
  names = ["depth", "distance_to_camera"]
  config = fk.Config(depth_clipping_behavior="max")
  _, _, data = _frame(names, config=config, spawn=None, scene_clip=(0.01, 50.0))
  for name in names:
    assert (_out(data, name)[:, 0, :, 0] == np.asarray(50.0, np.float32)).all(), name


def test_radial_misses_follow_the_renderers_far_clip_after_set_clip():
  """ "max" after `renderer.set_clip`: radial misses are the far clip the renderer now writes
  (`Renderer.clips`), not the one the session was built with (they were scaled)."""
  names = ["depth", "distance_to_camera"]
  config = fk.Config(depth_clipping_behavior="max")
  cams, session, data = _frame(names, config=config, spawn=fk.Spawn(clipping_range=(0.01, 50.0)))
  session.renderer.set_clip(zfar=30.0)
  assert session.renderer.clips == ((0.01, 30.0),)
  cams.render(session)
  cams.read_output(session, data)
  for name in names:
    assert (_out(data, name)[:, 0, :, 0] == np.asarray(30.0, np.float32)).all(), name


def test_intrinsics_per_environment():
  """Each env renders with its own K (no shared ray field): a narrower lens magnifies the box."""
  model, state = fk.boxes(2)
  cams, session, data = _frame(["depth"], sim=fk.Simulation(model, state))
  k = fk.intrinsics(2, W, H).numpy()
  k[1, 0, 0] *= 2.0
  k[1, 1, 1] *= 2.0
  cams.update_camera_intrinsics(session, wp.array(k, dtype=wp.mat33, device="cpu"))
  cams.render(session)
  hits = (_out(data, "depth")[..., 0] < 10.0)[:, H // 2].sum(axis=-1)
  assert hits[1] > 1.6 * hits[0], hits


def test_off_centre_principal_point():
  """An off-centre K (cx = W / 2 + 4 px) moves the image 4 px right; radial depth follows the
  shifted rays (the pixel next to the new principal point)."""
  model, state = fk.boxes(2)
  cams, session, data = _frame(["depth", "distance_to_camera"], sim=fk.Simulation(model, state))
  k = fk.intrinsics(2, W, H).numpy()
  k[1, 0, 2] += 4.0
  cams.update_camera_intrinsics(session, wp.array(k, dtype=wp.mat33, device="cpu"))
  cams.render(session)
  row = _out(data, "depth")[:, H // 2, :, 0] < 10.0
  first = row.argmax(axis=-1)
  assert first[1] - first[0] == 4 and row[0].sum() == row[1].sum(), row
  radial, planar = _out(data, "distance_to_camera")[1, ..., 0], _out(data, "depth")[1, ..., 0]
  # Pixel (W / 2 + 4, H / 2) is half a pixel from (cx, cy) in x and y.
  half = 0.5 / (0.5 * W / math.tan(math.radians(30.0)))
  np.testing.assert_allclose(radial[H // 2, W // 2 + 4], planar[H // 2, W // 2 + 4] * math.sqrt(1.0 + 2.0 * half**2), rtol=1e-6)


def test_graph_replays_when_isaac_swaps_the_poses():
  """CUDA: the frame replays from one graph; a new body-pose array (Isaac's provider swapped its
  buffer) is copied into the rendered state, not captured again, and the outputs `finish` writes
  (radial depth) follow (skipped on CPU)."""
  if not wp.get_cuda_device_count():
    pytest.skip("graph capture needs CUDA")
  with wp.ScopedDevice("cuda:0"):
    model, state = fk.boxes(2, device="cuda:0")
    sim = fk.Simulation(model, state)
    cams = Cameras(fk.Config())
    cams.initialize(sim)
    names = ["depth", "distance_to_camera"]
    session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(W, H, names), 2))
    data = fk.CameraData.allocate(names, 2, W, H, device="cuda:0")
    cams.set_outputs(session, data.output)
    pos, rot = fk.poses([EYE] * 2, device="cuda:0")
    k = fk.intrinsics(2, W, H, device="cuda:0")
    cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(k))
    graphs = []
    capture = session.renderer.capture

    def counted(*args, **kwargs):  # Renderer.capture's signature
      graphs.append(capture(*args, **kwargs))
      return graphs[-1]

    session.renderer.capture = counted
    for step in range(3):
      q = required(state.body_q).numpy()
      q[:, 0] = 0.25 * step  # both boxes move away
      state.body_q = wp.array(q, dtype=wp.transformf)  # a new array every frame: copied
      cams.update_transforms()
      cams.render(session)
      wp.synchronize()
      np.testing.assert_allclose(_out(data, "depth")[:, H // 2, W // 2, 0], 2.7 + 0.25 * step, rtol=1e-4)
      np.testing.assert_allclose(_out(data, "distance_to_camera")[:, H // 2, W // 2, 0], 2.7 + 0.25 * step, rtol=1e-3)
    assert len(graphs) == 1  # captured once, replayed after


def test_material_writes_tint_each_environment():
  """Isaac's visual-material writes (per-env color rows) change only their env's instances, in
  every camera, without recompiling."""
  model, state = fk.boxes(2)
  materials = [
    f"/World/envs/env_{w}/Looks/red" if label.endswith("Box/geom") else ""
    for w, label in zip(required(model.shape_world).numpy(), model.shape_label, strict=True)
  ]
  sim = fk.Simulation(model, state, materials=materials)
  cams = Cameras(fk.Config(settings=UNLIT_FAST_CHANGES))
  cams.initialize(sim)
  colors = wp.array([wp.vec3(0.8, 0.2, 0.1), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
  writer = cams.material_writer(["/World/envs/env_0/Looks/red", "/World/envs/env_1/Looks/red"], colors)
  writer()  # Isaac's full sync after creation: the composed (linear) rows as they are
  session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(W, H, ["albedo"]), 2))
  data = fk.CameraData.allocate(["albedo"], 2, W, H)
  cams.set_outputs(session, data.output)
  pos, rot = fk.poses([EYE] * 2)
  cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(2, W, H)))
  cams.render(session)
  before = _out(data, "albedo")[:, H // 2, W // 2].copy()
  np.testing.assert_allclose(before[0, :3], [231, 124, 89], atol=1)  # sRGB of linear (0.8, 0.2, 0.1)
  colors.assign(np.array([[0.8, 0.2, 0.1], [0.05, 0.6, 0.9]], np.float32))
  cpu = {"dtype": wp.int32, "device": "cpu"}
  features = session.renderer.features
  with conformance.module_loads() as loads:
    writer({"color": wp.array([0], **cpu)}, wp.array([1], **cpu))  # env 1 only
    cams.render(session)
  assert not loads, loads
  after = _out(data, "albedo")[:, H // 2, W // 2]
  np.testing.assert_array_equal(after[0], before[0])
  np.testing.assert_allclose(after[1, :3], [63, 203, 243], atol=1)  # sRGB of (0.05, 0.6, 0.9)
  assert session.renderer.features is features
  cams.close()


def test_graph_replays_the_frame():
  """On CUDA each frame is a graph replay matching the eager frame (skipped on CPU)."""
  if not wp.get_cuda_device_count():
    pytest.skip("graph capture needs CUDA")
  model, state = fk.boxes(2, device="cuda:0")
  with wp.ScopedDevice("cuda:0"):
    cams = Cameras(fk.Config())
    cams.initialize(fk.Simulation(model, state))
    session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(W, H, ["rgb", "depth"]), 2))
    assert conformance.check_graph_replay(session.renderer, lambda: None, state)


def test_lens_distortion_is_refused():
  """Distorted lenses are not rendered yet: refused, not silently pinhole."""
  with pytest.raises(NotImplementedError, match="distortion"):
    _frame(["rgb"], spawn=fk.Spawn(distortion=object()))


def test_arrays_on_another_device_are_refused():
  """Regression (L4 run 4): a CUDA K matrix passed to a CPU renderer crashed the process (a CPU
  launch reading device memory); now a ValueError."""
  if not wp.get_cuda_device_count():
    pytest.skip("needs a CUDA device besides the CPU")
  cams, session, _ = _frame(["depth"])
  with pytest.raises(ValueError, match="intrinsics is on cuda"):
    cams.update_camera_intrinsics(session, fk.intrinsics(2, W, H, device="cuda:0"))


def test_lighting_ev100_is_the_session_exposure():
  """`lighting.ev100` is the manual render exposure (`exposure.ev100`) unless the settings set it,
  as in the Newton front ends; without a lighting the default's automatic exposure."""
  from raywarp.integrations.isaaclab.cameras import render_settings

  lit = fk.Config(lighting=rw.Lighting.studio(ev100=9.0))
  assert render_settings(lit).exposure == rw.settings.Exposure(mode="manual", ev100=9.0)
  assert render_settings(fk.Config(lighting=rw.Lighting(ev100=9.0), settings={"exposure.ev100": 11.0})).exposure.ev100 == 11.0
  assert render_settings(fk.Config()).exposure == rw.RenderSettings().exposure


def test_camera_state_shapes_fail_before_writing_poses() -> None:
  """Incomplete calibration/pose batches cannot launch out-of-bounds device reads."""
  cams, session, _ = _frame(["depth"])
  before = session.renderer.camera_pose.numpy().copy()
  pos, rot = fk.poses([EYE] * 2)
  with pytest.raises(ValueError, match="intrinsics must have shape"):
    cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(1, W, H)))
  np.testing.assert_array_equal(session.renderer.camera_pose.numpy(), before)
  with pytest.raises(ValueError, match="intrinsics must have shape"):
    cams.update_camera_intrinsics(session, fk.intrinsics(1, W, H))
  cams.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA stream capture")
def test_session_render_inside_outer_capture() -> None:
  """A warmed session records into the caller's graph instead of nesting its own capture."""
  with wp.ScopedDevice("cuda:0"):
    model, state = fk.boxes(2, device="cuda:0")
    config = fk.Config(settings={"performance.traversal": "software"}, use_cuda_graph=True)
    cams = Cameras(config)
    cams.initialize(fk.Simulation(model, state))
    session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(W, H, ["depth"]), 2))
    data = fk.CameraData.allocate(["depth"], 2, W, H, device="cuda:0")
    cams.set_outputs(session, data.output)
    pos, rot = fk.poses([EYE] * 2, device="cuda:0")
    cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(2, W, H, device="cuda:0")))
    cams.render(session)
    expected = _out(data, "depth").copy()
    with wp.ScopedCapture(device="cuda:0") as capture:
      cams.render(session)
    assert capture.graph is not None
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(_out(data, "depth"), expected)
    cams.close()


@pytest.mark.parametrize("emissive", [True, False])
def test_emissive_lights_setting_decides_the_panel_fit(emissive: bool) -> None:
  """Every session renders with the renderer config's settings: `lighting.emissive_lights=False`
  uploads without fitting the emissive panel lights (no fit in the scene's load report), the
  default fits them."""
  model, state = fk.boxes(1)
  cams = Cameras(fk.Config(settings={"lighting.emissive_lights": emissive}))
  cams.initialize(fk.Simulation(model, state))
  assert ("lights.emissive_panels" in cams.scene.report.counts) == emissive


# ---------------------------------------------------------------------------------------------
# Canonical live material colors: address isolation, shared storage.


def test_zero_original_channels_can_be_recolored() -> None:
  """A black material can become blue in the actual frame tables; inverse tint cannot do this."""
  model, state = fk.boxes(2)
  assert model.shape_color is not None
  model.shape_color.zero_()
  paths = ["/Looks/Paint"] * model.shape_count
  cams = Cameras(fk.Config(settings=UNLIT_FAST_CHANGES, use_cuda_graph=False))
  cams.initialize(fk.Simulation(model, state, materials=paths))
  writer = cams.material_writer(["/Looks/Paint"], wp.array([(0, 0, 0.7)], dtype=wp.vec3, device="cpu"))
  writer()
  session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(8, 8, ["albedo"]), 2))
  frame = session.renderer.debug.frame()
  materials = frame.material_base_color.numpy()
  ids = frame.instance_material.numpy()[0]
  tint = frame.instance_color.numpy()
  for world in range(2):
    effective = materials[world % materials.shape[0], ids, :3] * tint[world % tint.shape[0], :, :3]
    np.testing.assert_allclose(effective, np.tile([0, 0, 0.7], (len(ids), 1)), atol=1e-7)
  cams.close()


def _rebinding_renderer() -> rw.Renderer:
  """A lit sphere seen by a camera of 2 worlds, base color / specular weight / size per world."""
  builder = rw.SceneBuilder()
  material = builder.add_material(rw.Material())
  builder.add_instance("sphere", material=material, size=(1, 0, 0))
  gpu = upload(builder.build().with_lighting(rw.Lighting.studio()), device="cpu")
  renderer = rw.Renderer(
    gpu,
    nworld=2,
    cameras=[rw.CameraSpec()],
    resolution=(8, 8),
    settings=rw.RenderSettings(outputs=rw.Outputs.of("hdr")),
    per_world=rw.PerWorld(material=["base_color", "specular_weight"], size=["*"]),
  )
  pose = np.zeros((2, 1, 7), np.float32)
  pose[..., 2], pose[..., 6] = 3.0, 1.0  # 3 m up the camera's back axis, identity rotation
  renderer.camera_pose.assign(pose)
  return renderer


def test_rebinding_requires_declared_matching_storage_without_readback(monkeypatch: pytest.MonkeyPatch) -> None:
  """The same prepared storage works for any core renderer, including derived OpenPBR inputs:
  a rebound specular weight renders like the same values written in place, and an invalid
  value in the bound storage raises at the next render."""
  renderer = _rebinding_renderer()
  default = required(renderer.render().hdr).numpy().copy()
  prepared = wp.clone(whole_table(renderer.per_world["material.base_color"]))
  weight = wp.full((2, 1), 0.5, dtype=float, device="cpu")
  version = renderer.debug.graph_version()

  def no_readback(_: wp.array) -> None:
    raise AssertionError("rebinding must not read device data")

  with monkeypatch.context() as patch:
    patch.setattr(wp.array, "numpy", no_readback)
    renderer.bind_per_world("material.base_color", prepared)
    renderer.bind_per_world("material.specular_weight", weight)
  assert renderer.per_world["material.base_color"] is prepared
  assert renderer.per_world["material.specular_weight"] is weight
  assert renderer.debug.graph_version() == version + 2
  rebound = required(renderer.render().hdr).numpy()
  in_place = _rebinding_renderer()
  whole_table(in_place.per_world["material.specular_weight"]).fill_(0.5)
  np.testing.assert_array_equal(rebound, required(in_place.render().hdr).numpy())
  assert not np.array_equal(rebound, default)  # the derived reflectance follows the bound weight
  renderer.bind_per_world("material.base_color", prepared)
  assert renderer.debug.graph_version() == version + 2
  with pytest.raises(ValueError, match="not declared"):
    renderer.bind_per_world("material.base_metalness", weight)
  with pytest.raises(ValueError, match="shape"):
    renderer.bind_per_world("material.base_color", wp.zeros((1, 1), dtype=wp.vec4, device="cpu"))
  with pytest.raises(ValueError, match="dtype"):
    renderer.bind_per_world("material.base_color", wp.zeros((2, 1), dtype=wp.vec3, device="cpu"))
  strided = wp.zeros((2, 2), dtype=wp.vec4, device="cpu")[:, ::2]
  with pytest.raises(ValueError, match="contiguous"):
    renderer.bind_per_world("material.base_color", strided)
  with pytest.raises(ValueError, match="geometry storage"):
    renderer.bind_per_world("instance.size", wp.zeros((2, 1), dtype=wp.vec3, device="cpu"))
  weight.fill_(-1.0)  # the validation checks the bound storage, not the replaced one
  renderer.render()  # the frame's check is read back by the next render (no sync)
  with pytest.raises(ValueError, match="per-world values must be numbers in range"):
    renderer.render()


def test_canonical_color_map_retains_weights_and_distinct_addresses() -> None:
  """Shared authored material parameters split only where composed prim addresses differ."""
  from raywarp.integrations.isaaclab.materials import MaterialLayout, MaterialWriter, PreparedColors

  builder = rw.SceneBuilder()
  material = builder.add_material(
    rw.Material(
      base_color=(0, 0, 0),
      base_weight=0.5,
      transmission_weight=0.25,
      transmission_color=(0.2, 0.4, 0.8),
      extras=rw.scene.OpenPbrExtras(subsurface_weight=0.4, subsurface_color=(0.8, 0.2, 0.1)),
    )
  )
  for _ in range(2):
    builder.add_instance("sphere", material=material, size=(1, 0, 0))
  host = builder.build()
  textures, images = host.textures, host.images
  layout = MaterialLayout.prepare(host, (("/A/Paint", "/B/Paint"), ("/C/Paint", "/D/Paint")))
  assert len(host.materials) == 2 and host.instances.material.tolist() == [0, 1]
  assert host.textures is textures and host.images is images
  gpu = upload(host, device="cpu")
  prepared = PreparedColors.create(gpu, layout)
  changed = wp.zeros(2, dtype=bool, device="cpu")
  colors = wp.array([(0, 0, 0)], dtype=wp.vec3, device="cpu")
  writer = MaterialWriter(
    gpu,
    layout=layout,
    material_paths=["/D/Paint"],
    colors=colors,
    prepared=prepared,
    reset=lambda mask: wp.copy(changed, mask),
    before_write=lambda: colors.fill_(wp.vec3(0, 1, 0)),
  )
  before = prepared.table.numpy().copy()
  writer()
  after = prepared.table.numpy()
  np.testing.assert_array_equal(after[0], before[0])
  np.testing.assert_array_equal(after[1, 0], before[1, 0])
  expected = 0.5 * (
    ((1 - 0.4) * np.array([0, 1, 0]) + 0.4 * np.array([0.8, 0.2, 0.1])) * 0.75 + 0.25 * np.array([0.2, 0.4, 0.8])
  )
  np.testing.assert_allclose(after[1, 1, :3], expected, atol=1e-7)
  assert after[1, 1, 3] == before[1, 1, 3]
  np.testing.assert_array_equal(changed.numpy(), [False, True])


def test_color_input_provenance_distinguishes_glass_transmission_tint() -> None:
  """Exact authored input names distinguish base color from glass tint (thin-walled: a surface
  tint); unknown inputs report."""
  from raywarp.integrations.isaaclab.materials import MaterialLayout, MaterialWriter, PreparedColors

  builder = rw.SceneBuilder()
  material = builder.add_material(
    rw.Material(base_color=(0.2, 0.3, 0.4), transmission_color=(0.7, 0, 0), transmission_weight=1, geometry_thin_walled=True)
  )
  for _ in range(3):
    builder.add_instance("sphere", material=material, size=(1, 0, 0))
  host = builder.build()
  paths = ("/A/Paint", "/B/Paint", "/C/Paint")
  layout = MaterialLayout.prepare(host, (paths,))
  gpu = upload(host, device="cpu")
  prepared = PreparedColors.create(gpu, layout)
  writer = MaterialWriter(
    gpu,
    layout=layout,
    material_paths=paths,
    colors=wp.array([(0, 0, 0.8)] * 3, dtype=wp.vec3, device="cpu"),
    prepared=prepared,
    reset=lambda _: None,
    input_names=("diffuseColor", "glass_color", "unknown_color"),
  )
  before = prepared.table.numpy().copy()
  writer()
  after = prepared.table.numpy()
  np.testing.assert_array_equal(after[0, 0], before[0, 0])  # A base color has no contribution through full transmission.
  np.testing.assert_allclose(after[0, 1, :3], [0, 0, 0.8], atol=1e-7)
  np.testing.assert_array_equal(after[0, 2], before[0, 2])
  assert gpu.report.dropped["isaac.material.input(unsupported)"] == 1


def test_live_glass_color_of_volume_glass_is_reported() -> None:
  """Solid glass on closed shapes is volume glass: its color is a static extinction
  (`pipeline/upload/optics.py`), so a live glass color leaves its row and is reported."""
  from raywarp.integrations.isaaclab.materials import MaterialLayout, MaterialWriter, PreparedColors

  builder = rw.SceneBuilder()
  material = builder.add_material(rw.Material(transmission_color=(0.7, 0, 0), transmission_weight=1))
  builder.add_instance("sphere", material=material, size=(1, 0, 0))
  host = builder.build()
  layout = MaterialLayout.prepare(host, (("/A/Glass",),))
  assert layout.volume.tolist() == [True]
  gpu = upload(host, device="cpu")
  prepared = PreparedColors.create(gpu, layout)
  before = prepared.table.numpy().copy()
  MaterialWriter(
    gpu,
    layout=layout,
    material_paths=("/A/Glass",),
    colors=wp.array([(0, 0, 0.8)], dtype=wp.vec3, device="cpu"),
    prepared=prepared,
    reset=lambda _: None,
    input_names=("glass_color",),
  )()
  np.testing.assert_array_equal(prepared.table.numpy(), before)
  assert gpu.report.approximated["isaac.material.glass_color(volume glass: static extinction)"] == 1


# ---------------------------------------------------------------------------------------------
# Live scalar material channels: canonical conversions and independent world updates.


def test_live_roughness_matches_fresh_upload_with_and_without_coupled_coats() -> None:
  """Canonical roughness updates retain the exact existing coupled/glTF coat distinction."""
  materials = (
    rw.Material(specular_roughness=0.2),
    rw.Material(specular_roughness=0.2, coat_weight=0.7, coat_roughness=0.4),
    rw.Material(
      specular_roughness=0.2,
      coat_weight=0.7,
      coat_roughness=0.4,
      coat_ior=1.5,
      ext=rw.scene.RaywarpExt(flags=int(rw.scene.MaterialFlags.COAT_GLTF)),
    ),
  )
  builder, fresh = rw.SceneBuilder(), rw.SceneBuilder()
  for material in materials:
    builder.add_instance("sphere", size=(1, 0, 0), material=builder.add_material(material))
    fresh.add_instance(
      "sphere", size=(1, 0, 0), material=fresh.add_material(dataclasses.replace(material, specular_roughness=0.65))
    )
  host = builder.build()
  paths = tuple(f"/Looks/Paint{i}" for i in range(3))
  layout = MaterialLayout.prepare(host, (paths,))
  gpu, expected = upload(host, device="cpu"), upload(fresh.build(), device="cpu")
  spec = SCALAR_CHANNELS["roughness"]
  table = scalar_table(gpu, 1, spec)
  writer = ScalarWriter(
    gpu,
    layout=layout,
    spec=spec,
    material_paths=paths,
    input_names=("roughness", "reflection_roughness_constant", "frosting_roughness"),
    values=wp.full(3, 0.65, dtype=float, device="cpu"),
    table=table,
    reset=lambda _: None,
  )
  writer()
  np.testing.assert_array_max_ulp(table.numpy(), expected.tables["material.specular_roughness"].numpy(), maxulp=1)
  assert table.numpy()[0, 1] > table.numpy()[0, 0]
  assert table.numpy()[0, 2] == table.numpy()[0, 0]


def test_scalar_channels_share_storage_and_change_only_selected_worlds() -> None:
  """A metallic edit changes one world in every camera while unrelated scalar rows stay intact."""
  model, state = fk.boxes(2)
  assert model.shape_world is not None
  paths = [f"/World{world}/Looks/Paint" if world >= 0 else "" for world in model.shape_world.numpy()]
  cams = Cameras(fk.Config(use_cuda_graph=False))
  cams.initialize(fk.Simulation(model, state, materials=paths))
  material_paths = ("/World0/Looks/Paint", "/World1/Looks/Paint")
  values = {name: wp.array([0.25, 0.5], dtype=float, device="cpu") for name in SCALAR_CHANNELS}
  inputs = {"roughness": "roughness", "metallic": "metallic", "specular": "specular_level"}
  writers = MaterialWriters(
    channels=tuple(
      cams.scalar_writer(spec, material_paths, (inputs[name],) * 2, values[name]) for name, spec in SCALAR_CHANNELS.items()
    )
  )
  writers()
  sessions = [cams.create_render_data(fk.RenderSpec(fk.CameraCfg(8, 8, ["rgb"]), 2)) for _ in range(2)]
  for spec in SCALAR_CHANNELS.values():
    assert sessions[0].renderer.per_world[spec.field] is sessions[1].renderer.per_world[spec.field]
  before = {name: sessions[0].renderer.per_world[spec.field].numpy().copy() for name, spec in SCALAR_CHANNELS.items()}
  values["metallic"].assign(np.array([0.1, 0.8], np.float32))
  writers({"metallic": wp.array([0], dtype=int, device="cpu")}, wp.array([1], dtype=int, device="cpu"))
  for name, spec in SCALAR_CHANNELS.items():
    after = sessions[0].renderer.per_world[spec.field].numpy()
    np.testing.assert_array_equal(after[0], before[name][0])
    if name != "metallic":
      np.testing.assert_array_equal(after, before[name])
    else:
      assert np.any(after[1] == values["metallic"].numpy()[1])
  # OmniPBR specular_level is F0=0.08*s, hence canonical OpenPBR weight=2*s.
  assert np.any(before["specular"][0] == 0.5)
  writers.close()
  cams.close()


type HdrImages = np.ndarray[[int, int, int, 3], np.dtype[np.float32]]


def render_scalar_edit(channel: str, width: int = 48) -> tuple[HdrImages, HdrImages]:
  """Lit sphere: roughness broadens highlights; metallic changes the red diffuse/specular balance.

  This shared diagnostic fixture returns linear HDR before/after one world-one update.
  """
  import math

  from pxr import Sdf, UsdGeom, UsdLux, UsdShade
  from test_isaaclab_usd_scope import ROOTS, _model, _stage

  from raywarp.integrations.isaaclab.usd_source import UsdSource

  stage, model = _stage(), _model(bodies=True)
  visual = ROOTS[0] + "/Body/Visual"
  UsdGeom.Sphere.Define(stage, visual).CreateRadiusAttr(0.75)
  for path in (ROOTS[0] + "/Lamp", "/Arena/GlobalLamp"):
    UsdLux.SphereLight(stage.GetPrimAtPath(path)).GetIntensityAttr().Set(10000.0)
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROOTS[0] + "/Lamp")).SetTranslate((-2.0, -1.0, 3.0))
  UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/Arena/GlobalLamp")).SetTranslate((-3.0, 3.0, 3.0))
  paths = tuple(root + "/Looks/Paint" for root in ROOTS)
  material = UsdShade.Material.Define(stage, paths[0])
  shader = UsdShade.Shader.Define(stage, paths[0] + "/Surface")
  shader.CreateIdAttr("UsdPreviewSurface")
  shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.7, 0.1, 0.05))
  shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
  shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
  material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
  UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(visual)).Bind(material)
  model.body_label = [root + "/Body" for root in ROOTS]
  cameras = Cameras(
    fk.Config(
      settings={"performance.traversal": "software", "aa.spp": 1, "exposure.mode": "manual", "exposure.ev100": 5.0},
      use_cuda_graph=False,
    )
  )
  cameras.initialize(fk.Simulation(model, model.state(), source=UsdSource.create(stage, model, ROOTS)))
  original, edited = (0.12, 0.7) if channel == "roughness" else (0.0, 1.0)
  values = wp.full(2, original, dtype=float, device="cpu")
  writer = cameras.scalar_writer(SCALAR_CHANNELS[channel], paths, (channel,) * 2, values)
  writer()
  session = cameras.create_render_data(fk.RenderSpec(fk.CameraCfg(width, width, ["rgb_hdr"]), 2))
  data = fk.CameraData.allocate(["rgb_hdr"], 2, width, width)
  cameras.set_outputs(session, data.output)
  pos, rot = fk.poses([(-3.0, 0.0, 1.0), (4.0, -1.0, 1.0)])
  orientations = rot.numpy()
  orientations[1] = (0, 0, math.sqrt(0.5), math.sqrt(0.5))
  rot.assign(orientations)
  cameras.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(2, width, width)))
  cameras.render(session)
  before = np.asarray(data.output["rgb_hdr"].warp.numpy(), dtype=np.float32).reshape((2, width, width, 3)).copy()
  values.assign(np.array([original, edited], np.float32))
  writer({channel: wp.array([0], dtype=int, device="cpu")}, wp.array([1], dtype=int, device="cpu"))
  cameras.render(session)
  after = np.asarray(data.output["rgb_hdr"].warp.numpy(), dtype=np.float32).reshape((2, width, width, 3)).copy()
  cameras.close()
  return before, after


def test_live_roughness_changes_only_edited_world_render() -> None:
  before, after = render_scalar_edit("roughness")
  np.testing.assert_array_equal(before[0], after[0])
  assert np.isfinite(after).all() and np.max(after) > 0
  assert np.max(np.abs(after[1] - before[1])) > 1e-3


def test_live_metallic_changes_only_edited_world_render() -> None:
  before, after = render_scalar_edit("metallic")
  np.testing.assert_array_equal(before[0], after[0])
  assert np.isfinite(after).all() and np.max(after) > 0
  assert np.max(np.abs(after[1] - before[1])) > 1e-3


def test_mapped_scalar_inputs_are_reported_without_changing_prepared_maps() -> None:
  """A prepared MDL lerp map is not a raw authored-constant multiplier, even at full influence."""
  for name, slot, authored in (
    ("roughness", rw.scene.Slot.SPECULAR_ROUGHNESS, "reflection_roughness_constant"),
    ("metallic", rw.scene.Slot.BASE_METALNESS, "metallic_constant"),
  ):
    builder = rw.SceneBuilder()
    tex = builder.add_texture(np.full((2, 2, 4), 180, np.uint8), colorspace="linear")
    mapped = rw.Material(
      specular_roughness=1.0,
      base_metalness=1.0,
      textures=((slot, rw.scene.TextureRef(tex)),),
    )
    plain = rw.Material(specular_roughness=0.2, base_metalness=0.2)
    for material in (mapped, plain):
      builder.add_instance("sphere", size=(1, 0, 0), material=builder.add_material(material))
    host = builder.build()
    paths = ("/Looks/Mapped", "/Looks/Plain")
    layout = MaterialLayout.prepare(host, (paths,))
    scene = upload(host, device="cpu")
    spec = SCALAR_CHANNELS[name]
    table = scalar_table(scene, 1, spec)
    before = table.numpy().copy()
    # The authored constant is0.2; its prepared map already contains the importer blend.
    values = wp.full(2, 0.2, dtype=float, device="cpu")
    writer = ScalarWriter(
      scene,
      layout=layout,
      spec=spec,
      material_paths=paths,
      input_names=(authored,) * 2,
      values=values,
      table=table,
      reset=lambda _: None,
    )
    writer()
    np.testing.assert_array_equal(table.numpy(), before)
    assert scene.report.dropped["isaac.material.mapped_input(unsupported)"] == 1
    values.fill_(0.8)
    writer()
    result = table.numpy()
    assert result[0, 0] == before[0, 0]
    assert result[0, 1] == values.numpy()[1]


def _camera(cams: Cameras, names: list[str], width: int = W, n: int = 2) -> tuple[CameraSession, fk.CameraData]:
  """A camera of `names` in every env, bound to its own Isaac buffers."""
  session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(width, H, names), n))
  data = fk.CameraData.allocate(names, n, width, H)
  cams.set_outputs(session, data.output)
  return session, data


def _pose(cams: Cameras, session: CameraSession, eye: tuple[float, float, float], width: int = W, n: int = 2) -> None:
  pos, rot = fk.poses([eye] * n)
  cams.update_camera(session, fk.ProxyArray(pos), fk.ProxyArray(rot), fk.ProxyArray(fk.intrinsics(n, width, H)))


def test_alike_cameras_render_in_one_batch_as_they_would_alone():
  """Alike cameras Isaac renders together share one renderer (a camera each): every camera's
  outputs equal its render alone; one rendered alone gets its own renderer and writes only its
  outputs; another size is another group."""
  names = ["rgb", "depth", "distance_to_camera", *SEG]
  eyes = [EYE, (-2.5, 0.4, 0.6)]
  alone = []
  for eye in eyes:
    model, state = fk.boxes(2)
    cams = Cameras(fk.Config(settings=NO_GI, use_cuda_graph=False))
    cams.initialize(fk.Simulation(model, state))
    session, data = _camera(cams, names)
    _pose(cams, session, eye)
    cams.render_batch([session])
    alone.append({name: _out(data, name).copy() for name in data.output})
    cams.close()
  model, state = fk.boxes(2)
  cams = Cameras(fk.Config(settings=NO_GI, use_cuda_graph=False))
  cams.initialize(fk.Simulation(model, state))
  pairs = [_camera(cams, names) for _ in eyes]
  other, other_data = _camera(cams, ["depth"], width=W + 8)
  for (session, _), eye in zip(pairs, eyes, strict=True):
    _pose(cams, session, eye)
  _pose(cams, other, EYE, width=W + 8)
  cams.render_batch([session for session, _ in pairs])
  # Rendered together: one renderer (the poses written before the regroup carried over).
  assert pairs[0][0].renderer is pairs[1][0].renderer and other.renderer is not pairs[0][0].renderer
  for (_, data), expected in zip(pairs, alone, strict=True):
    for name, image in expected.items():
      np.testing.assert_array_equal(_out(data, name), image, err_msg=name)
  assert not _out(other_data, "depth").any()  # not requested: untouched
  for _, data in pairs:
    data.output["depth"].warp.zero_()
  cams.render_batch([pairs[1][0]])  # rendered alone (Isaac's lazy reads): its own renderer again
  assert not _out(pairs[0][1], "depth").any()  # not requested: untouched
  np.testing.assert_array_equal(_out(pairs[1][1], "depth"), alone[1]["depth"])
  assert pairs[0][0].renderer is not pairs[1][0].renderer and len(cams.groups) == 3
  cams.close()


def test_a_failed_release_leaves_no_closed_renderer_and_releases_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
  """A renderer whose close raises is detached all the same (a later use of its group rebuilds,
  instead of rendering on the closed one forever) and `Cameras.close` still releases the other
  groups' renderers, then raises the failure."""
  cams, first, _ = _frame(["depth"])
  second, _ = _camera(cams, ["depth"], width=W + 8)
  _pose(cams, second, EYE, width=W + 8)
  cams.render(second)
  renderers = [first.renderer, second.renderer]
  groups = [first.group, second.group]
  closed: list[rw.Renderer] = []
  real = rw.Renderer.close

  def close(r: rw.Renderer) -> None:
    closed.append(r)
    real(r)
    if len(closed) == 1:
      raise RuntimeError("cleanup failed")

  monkeypatch.setattr(rw.Renderer, "close", close)
  with pytest.raises(ExceptionGroup, match="Isaac Lab cameras"):
    cams.close()
  assert sorted(map(id, closed)) == sorted(map(id, renderers))
  assert all(g._renderer is None for g in groups) and not cams.groups


def test_a_failed_bind_releases_the_renderer_of_the_replaced_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
  """`set_outputs` with arrays the renderer refuses: the renderer may still render into the
  replaced arrays, which the session no longer holds; it is released (rebuilt over the new ones
  at the next use). Arrays of another shape are refused before anything is replaced: the
  renderer and the bound arrays are kept."""
  cams, session, _ = _frame(["depth"])
  old, held = session.renderer, dict(session.arrays)
  bad = fk.CameraData.allocate(["depth"], 2, W + 1, H)  # another size than the camera's
  with pytest.raises(ValueError, match="Isaac Lab output 'depth' must be a"):
    cams.set_outputs(session, bad.output)
  assert session.group._renderer is old and not old._closed and session.arrays == held

  def refuse() -> None:
    raise ValueError("the renderer refuses the arrays")

  monkeypatch.setattr(session.group, "rebind", refuse)
  with pytest.raises(ValueError, match="refuses"):
    cams.set_outputs(session, fk.CameraData.allocate(["depth"], 2, W, H).output)
  assert session.group._renderer is None and old._closed


def test_background_color_is_shown_in_front_of_the_environment_map() -> None:
  """Isaac's `background_color` fills the misses; it only set `environment.background_color`, which
  a scene with an environment map never showed (its misses showed the map)."""
  model, state = fk.boxes(2)
  cams = Cameras(fk.Config(settings=NO_GI))
  cams.initialize(fk.Simulation(model, state))
  cfg = fk.CameraCfg(W, H, ["rgb"], background_color=(1.0, 0.0, 0.0))
  session = cams.create_render_data(fk.RenderSpec(cfg, 2))
  env = session.group.settings.environment
  assert env.background_color == (1.0, 0.0, 0.0) and env.background is False
  plain = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(W, H, ["rgb"]), 2))
  assert plain.group.settings.environment.background is True


def test_output_arrays_must_be_the_outputs_shape_and_dtype() -> None:
  """`flat` checked the byte count only: a transposed (N, W, H, C) array or a 3-channel albedo of
  as many bytes as rows passed, and `finish`'s passes wrote N x H x W pixels past its end."""
  model, state = fk.boxes(2)
  cams = Cameras(fk.Config(settings=NO_GI))
  cams.initialize(fk.Simulation(model, state))
  session = cams.create_render_data(fk.RenderSpec(fk.CameraCfg(W, H, ["albedo", "depth"]), 2))
  good = fk.CameraData.allocate(["albedo", "depth"], 2, W, H).output
  for name, bad in (
    ("albedo", wp.zeros((2, H, W, 3), dtype=wp.uint8)),
    ("albedo", wp.zeros((2, W, H, 4), dtype=wp.uint8)),
    ("depth", wp.zeros((2, H, W, 1), dtype=wp.int32)),
  ):
    arrays = {k: v.warp for k, v in good.items()} | {name: bad}
    with pytest.raises(ValueError, match=f"Isaac Lab output '{name}' must be a"):
      session.bind(arrays)  # pyrefly: ignore[bad-argument-type]  # wrong arrays on purpose
  session.bind({k: v.warp for k, v in good.items()})
