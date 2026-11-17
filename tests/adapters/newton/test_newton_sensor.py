"""raywarp.integrations.newton.sensor.SensorCamera as a drop-in for newton.sensors.SensorCamera
(Warp CPU device)."""

import math
from collections.abc import Callable

import numpy as np
import pytest
import warp as wp
from conftest import required

newton = pytest.importorskip("newton")
from newton.sensors import SensorCamera as NewtonSensorCamera

import raywarp as rw
from raywarp.integrations.newton.sensor import SensorCamera

type _Cams = wp.array[wp.transformf, [int]]

# Newton's own SensorCamera tests whose scenario does not depend on Newton's shading, run against
# raywarp.integrations.newton.sensor.SensorCamera.
_NEWTON_TESTS = [
  "test_update_validates_rays_and_transforms",
  "test_update_resolves_ssaa_rays",
  "test_update_resolves_msaa_rays",
  "test_update_rejects_invalid_anti_aliasing",
  "test_update_rejects_multisample_rays_without_anti_aliasing",
  "test_update_renders_from_camera_transforms",
  "test_update_respects_disable_clear_flag",
  "test_update_clears_reserved_and_out_of_range_world_indices",
  "test_update_respects_disable_preserve_flag",
  "test_update_defaults_world_indices_to_identity",
  "test_update_without_world_indices_maps_view_to_world",
  "test_world_indices_decouple_views_from_worlds",
  "test_update_rejects_default_world_indices_exceeding_world_count",
  "test_render_forward_depth_output",
  "test_forward_depth_uses_camera_origin",
  "test_update_supports_all_render_orders_with_3d_outputs",
]
_RES = 40
_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]
_FOV = math.radians(50.0)


@pytest.mark.parametrize("name", _NEWTON_TESTS)
def test_newton_scenario(monkeypatch, name):
  newton_tests = pytest.importorskip("newton.tests.test_sensor_camera")
  monkeypatch.setattr(newton_tests, "SensorCamera", SensorCamera)
  monkeypatch.setattr(newton_tests, "get_test_devices", lambda: ["cpu"])
  case = newton_tests.TestSensorCamera(name)
  with wp.ScopedDevice("cpu"):
    getattr(case, name)()


def _scene(nworld: int = 2, device: str = "cpu") -> tuple[newton.Model, newton.State, _Cams]:
  """Two worlds with every primitive type and a mesh; world 1 shifted; one camera per world."""
  world = newton.ModelBuilder()
  body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
  world.add_shape_sphere(body, xform=wp.transform((0.0, 0.0, 0.45), wp.quat_identity()), radius=0.2)
  arm = world.add_body(xform=wp.transform((0.7, 0.0, 0.4), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.6)))
  world.add_shape_capsule(arm, radius=0.1, half_height=0.25, color=(0.2, 0.3, 0.9))
  verts = np.array([[0.4, 0.4, 0.0], [0.8, 0.4, 0.0], [0.4, 0.8, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
  faces = np.array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], dtype=np.int32)
  world.add_shape_mesh(-1, mesh=newton.Mesh(verts, faces), xform=wp.transform((-1.0, -0.6, 0.0), wp.quat_identity()))
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(world, nworld)
  with wp.ScopedDevice(device):
    model = scene.finalize(device=device)
    state = model.state()
  q = state.body_q.numpy()
  q[len(q) // nworld :, 1] += 0.3
  state.body_q.assign(q)
  return model, state, _look_at([0.0, -3.0, 1.2], nworld, device)


def _look_at(eye: list[float], count: int, device: str = "cpu", target: tuple[float, float, float] = (0.0, 0.0, 0.3)) -> _Cams:
  """`count` camera poses at `eye` looking at `target` (z up)."""
  eye_np = np.asarray(eye, dtype=np.float64)
  forward = np.asarray(target) - eye_np
  forward /= np.linalg.norm(forward, keepdims=True)
  right = np.cross(forward, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right, keepdims=True)
  up = np.cross(right, forward)
  rot = np.stack([right, up, -forward], axis=1).astype(np.float32)
  cam = wp.transformf(wp.vec3f(*eye_np), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
  return wp.array([cam] * count, dtype=wp.transformf, device=device)


_OUTPUTS = ("color", "depth", "depth_forward", "shape_index", "normal", "albedo", "color_hdr")


def _render(cls, model, state, cams, rays, world_indices=None, **kwargs) -> dict[str, np.ndarray]:
  views = cams.shape[0]
  height, width = rays.shape[:2]
  with wp.ScopedDevice(model.device):
    sensor = cls(model, **kwargs)
    sensor.create_default_light()
    images = {name: getattr(sensor, f"allocate_image_{name}")(views, width, height) for name in _OUTPUTS}
    model.bvh_build_shapes(state)
    sensor.update(
      state,
      cams,
      rays,
      color_image=images["color"],
      depth_image=images["depth"],
      forward_depth_image=images["depth_forward"],
      shape_index_image=images["shape_index"],
      normal_image=images["normal"],
      albedo_image=images["albedo"],
      hdr_color_image=images["color_hdr"],
      world_indices=world_indices,
    )
  return {name: image.numpy().copy() for name, image in images.items()}


def _bytes(packed: np.ndarray) -> np.ndarray:
  return np.stack([np.right_shift(packed, s) & 0xFF for s in (0, 8, 16, 24)], axis=-1).astype(np.int32)


def _assert_geometry_matches(ours: dict[str, np.ndarray], ref: dict[str, np.ndarray], min_same: float = 0.995) -> np.ndarray:
  same = ours["shape_index"] == ref["shape_index"]
  assert same.mean() >= min_same
  hit = same & (ref["shape_index"] != 0xFFFFFFFF)
  assert hit.mean() > 0.3
  for name in ("depth", "depth_forward"):
    np.testing.assert_allclose(ours[name][hit], ref[name][hit], rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(ours[name][~hit & same], ref[name][~hit & same])
  np.testing.assert_allclose(ours["normal"][hit], ref["normal"][hit], atol=1e-2)
  # Albedo: the shape color in sRGB bytes (rounding of the color conversions differs).
  assert np.abs(_bytes(ours["albedo"])[hit] - _bytes(ref["albedo"])[hit]).max() <= 1
  np.testing.assert_array_equal(ours["albedo"][~hit & same], ref["albedo"][~hit & same])
  return hit


@pytest.fixture(scope="module", params=_DEVICES)
def scene(request):
  return _scene(device=request.param)


def test_pinhole_parity_with_newton(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  ours = _render(SensorCamera, model, state, cams, rays)
  ref = _render(NewtonSensorCamera, model, state, cams, rays)
  hit = _assert_geometry_matches(ours, ref)
  # Both worlds and every shape type are visible; color only sanity-checked (shading differs).
  assert len(np.unique(ours["shape_index"][hit])) >= 9
  box = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.BOX))[0])
  r, g, b, a = _bytes(ours["color"])[ours["shape_index"] == box].mean(axis=0)
  assert r > 2.0 * g and r > 2.0 * b and a == 255
  miss = ref["shape_index"] == 0xFFFFFFFF
  np.testing.assert_array_equal(ours["color"][miss], 0)  # ClearData().clear_color
  np.testing.assert_array_equal(ours["color_hdr"][miss], 0.0)
  assert np.all(ours["color_hdr"][hit] >= 0.0) and ours["color_hdr"][hit].max() > 0.05


def test_fisheye_parity_with_newton(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_fisheye_opencv(
    _RES, _RES, fx=14.0, fy=14.0, cx=_RES / 2, cy=_RES / 2, k1=0.05, max_fov=math.radians(170.0), device=model.device
  )
  ours = _render(SensorCamera, model, state, cams, rays)
  ref = _render(NewtonSensorCamera, model, state, cams, rays)
  _assert_geometry_matches(ours, ref, min_same=0.99)


def test_multisampling_keeps_first_sample_guides(scene):
  """B58: other samples covering a pixel whose first sample missed (the fisheye's horizon) left
  its albedo / normal as an opaque black hit instead of the clear values."""
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_fisheye_opencv(
    _RES, _RES, fx=14.0, fy=14.0, cx=_RES / 2, cy=_RES / 2, k1=0.05, max_fov=math.radians(170.0), device=model.device
  )
  ours = _render(SensorCamera, model, state, cams, rays, **{"aa.spp": 2})
  ref = _render(NewtonSensorCamera, model, state, cams, rays)
  _assert_geometry_matches(ours, ref, min_same=0.99)


def test_views_follow_world_indices(scene):
  model, state, _ = scene
  # Stereo pair of world 1, then world 0, a disabled view and an out-of-range one.
  cams = wp.array(
    [*_look_at([-0.1, -3.0, 1.2], 1).numpy(), *_look_at([0.1, -3.0, 1.2], 1).numpy(), *_look_at([0.0, -3.0, 1.2], 3).numpy()],
    dtype=wp.transformf,
    device=model.device,
  )
  flags = [1, 1, 0, int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR), 7]
  world_indices = wp.array(flags, dtype=wp.int32, device=model.device)
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  ours = _render(SensorCamera, model, state, cams, rays, world_indices=world_indices)
  ref = _render(NewtonSensorCamera, model, state, cams, rays, world_indices=world_indices)
  _assert_geometry_matches(ours, ref)
  # The two eyes see the same world from 20 cm apart; disabled / invalid views are cleared.
  assert not np.array_equal(ours["shape_index"][0], ours["shape_index"][1])
  for view in (3, 4):
    assert np.all(ours["shape_index"][view] == 0xFFFFFFFF) and np.all(ours["depth"][view] == 0.0)
    assert np.all(ours["color"][view] == 0)


def test_clear_data_and_linear_color_space(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  clear = SensorCamera.ClearData(
    clear_color=0xFF336699, clear_depth=-1.0, clear_shape_index=77, clear_normal=(0.0, 1.0, 0.0), clear_albedo=0x80112233
  )
  config = SensorCamera.RenderConfig(output_color_space=newton.utils.ColorSpace.LINEAR)
  results = {}
  for cls in (SensorCamera, NewtonSensorCamera):
    with wp.ScopedDevice(model.device):
      sensor = cls(model, default_clear_data=clear, default_render_config=config)
      color = sensor.allocate_image_color(2, _RES, _RES)
      albedo = sensor.allocate_image_albedo(2, _RES, _RES)
      depth = sensor.allocate_image_depth(2, _RES, _RES)
      index = sensor.allocate_image_shape_index(2, _RES, _RES)
      normal = sensor.allocate_image_normal(2, _RES, _RES)
      model.bvh_build_shapes(state)
      sensor.update(
        state,
        cams,
        rays,
        color_image=color,
        albedo_image=albedo,
        depth_image=depth,
        shape_index_image=index,
        normal_image=normal,
      )
    results[cls] = [a.numpy().copy() for a in (color, albedo, depth, index, normal)]
  ours, ref = results[SensorCamera], results[NewtonSensorCamera]
  miss = ref[3] == 77
  assert miss.any()
  np.testing.assert_array_equal(ours[3] == 77, miss)
  for a, b in zip(ours, ref, strict=True):
    np.testing.assert_array_equal(a[miss], b[miss])
  hit = ~miss & (ours[3] == ref[3])
  assert np.abs(_bytes(ours[1])[hit] - _bytes(ref[1])[hit]).max() <= 1  # linear albedo bytes


def test_shadows_follow_render_config(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  with wp.ScopedDevice(model.device):
    sensor = SensorCamera(model)
    sensor.create_default_light(direction=wp.vec3f(0.0, 0.6, -0.8))
    colors = []
    for shadows in (False, True):
      color = sensor.allocate_image_color(2, _RES, _RES)
      config = SensorCamera.RenderConfig(enable_shadows=shadows)
      sensor.update(state, cams, rays, color_image=color, render_config=config)
      colors.append(_bytes(color.numpy())[..., :3].sum())
  assert colors[1] < colors[0]


def test_render_options_and_lights():
  model, state, cams = _scene(nworld=1)
  with pytest.raises(TypeError, match="cam_res"):
    SensorCamera(model, cam_res=(8, 8))
  rays = SensorCamera.compute_camera_rays_pinhole(16, 16, camera_fov=_FOV, device="cpu")
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model, lighting=rw.Lighting([rw.Light.sun((0.0, 0.0, -1.0))]), **{"shadows.samples": 2})
    color = sensor.allocate_image_color(1, 16, 16)
    sensor.update(state, cams, rays, color_image=color)
  assert _bytes(color.numpy())[..., :3].max() > 50


def test_loose_particles_are_reported_not_rendered():
  """Particles no triangle uses have no surface to render: counted in the load report."""
  from raywarp.integrations.newton.api import scene as newton_scene

  builder = newton.ModelBuilder()
  builder.add_shape_sphere(-1, radius=0.3)
  builder.add_particle((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0, radius=0.05)
  with wp.ScopedDevice("cpu"):
    model = builder.finalize(device="cpu")
    SensorCamera(model)
    assert newton_scene(model).report.dropped["newton.particles(unrendered)"] == 1


def test_checkerboard_material(scene):
  model, state, cams = scene
  plane = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))[0])
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  with wp.ScopedDevice(model.device):
    sensor = SensorCamera(model)
    sensor.assign_checkerboard_material(shape_indices=[plane], resolution=8, checker_size=1)
    albedo = sensor.allocate_image_albedo(2, _RES, _RES)
    index = sensor.allocate_image_shape_index(2, _RES, _RES)
    for textures in (False, True):
      config = SensorCamera.RenderConfig(enable_textures=textures)
      sensor.update(state, cams, rays, albedo_image=albedo, shape_index_image=index, render_config=config)
      floor = _bytes(albedo.numpy())[index.numpy() == plane][:, 0]
      # Two greys (sRGB 0x80 and 0xBF times the shape color) with textures, one color without.
      assert (floor.max() > 1.3 * floor.min()) == textures
  with pytest.raises(ValueError, match="out-of-range"):
    sensor.assign_checkerboard_material(shape_indices=[model.shape_count])


def test_renders_only_requested_outputs_into_caller_arrays(scene):
  model, state, cams = scene
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  with wp.ScopedDevice(model.device):
    sensor = SensorCamera(model)
    sensor.create_default_light()
    color = sensor.allocate_image_color(2, _RES, _RES)
    depth = sensor.allocate_image_depth(2, _RES, _RES)
    sensor.update(state, cams, rays, color_image=color)
    r = required(sensor.renderer)
    assert r.settings.outputs == rw.Outputs(rgb=rw.settings.ColorOutput(alpha=True))
    out = r.outputs
    assert required(out.rgb).ptr == color.ptr  # zero copy
    assert out.depth is None and out.hdr is None
    sensor.update(state, cams, rays, depth_image=depth)
    assert sensor.renderer is r  # one renderer, its outputs switched
    assert r.settings.outputs.kinds == {"depth"} and not r.features.any_rgb
    assert required(r.outputs.depth).ptr == depth.ptr and r.outputs.rgb is None
  ref = _render(SensorCamera, model, state, cams, rays)
  np.testing.assert_array_equal(color.numpy(), ref["color"])
  np.testing.assert_array_equal(depth.numpy(), ref["depth"])


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs need a CUDA device")
def test_update_is_graph_capturable():
  model, state, cams = _scene(device="cuda:0")
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  world_indices = wp.array([1, 0], dtype=wp.int32, device=model.device)
  with wp.ScopedDevice(model.device):
    sensor = SensorCamera(model)
    color = sensor.allocate_image_color(2, _RES, _RES)
    index = sensor.allocate_image_shape_index(2, _RES, _RES)

    def update():
      sensor.update(state, cams, rays, color_image=color, shape_index_image=index, world_indices=world_indices)

    graph = sensor.capture(update)  # runs update() once eagerly, then captures it
    wp.synchronize()
    first = color.numpy().copy(), index.numpy().copy()
    # Move world 0's bodies away: the graph replays the pose sync from body_q.
    q = state.body_q.numpy()
    q[: len(q) // 2, 2] += 5.0
    state.body_q.assign(q)
    color.zero_()
    graph.launch()
    wp.synchronize()
    moved = color.numpy(), index.numpy()
  # View 0 shows world 1 (unchanged), view 1 world 0 (its bodies moved).
  np.testing.assert_array_equal(moved[0][0], first[0][0])
  np.testing.assert_array_equal(moved[1][0], first[1][0])
  assert (moved[1][1] != first[1][1]).any()


def _up_scene(up: str) -> tuple[newton.Model, newton.State, _Cams]:
  """A sphere on a ground plane, built Z-up or Y-up (the same scene rotated), and its camera."""
  to_up = (lambda p: p) if up == "Z" else (lambda p: (p[0], p[2], -p[1]))
  builder = newton.ModelBuilder(up_axis=newton.Axis.from_any(up))
  builder.add_ground_plane()
  body = builder.add_body(xform=wp.transform(to_up((0.2, 0.1, 0.35)), wp.quat_identity()))
  builder.add_shape_sphere(body, radius=0.35, color=(0.8, 0.8, 0.8))
  model = builder.finalize(device="cpu")
  eye, target, world_up = np.array(to_up((0.0, -3.0, 1.2))), np.array(to_up((0.0, 0.0, 0.3))), np.array(to_up((0.0, 0.0, 1.0)))
  forward = (target - eye) / np.linalg.norm(target - eye, keepdims=True)
  right = np.cross(forward, world_up)
  right /= np.linalg.norm(right, keepdims=True)
  rot = np.stack([right, np.cross(right, forward), -forward], axis=1).astype(np.float32)
  cam = wp.transformf(wp.vec3f(*eye.astype(np.float32)), wp.quat_from_matrix(wp.mat33f(rot.flatten())))
  return model, model.state(), wp.array([cam], dtype=wp.transformf, device="cpu")


def _color(
  model: newton.Model,
  state: newton.State,
  cams: _Cams,
  setup: Callable[[SensorCamera], None] | None = None,
  config: SensorCamera.RenderConfig | None = None,
) -> np.ndarray:
  """(1, H, W, 3) color bytes of the sensor of `model` (`setup` run on it first)."""
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device="cpu")
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model)
    if setup is not None:
      setup(sensor)
    color = sensor.allocate_image_color(1, _RES, _RES)
    sensor.update(state, cams, rays, color_image=color, render_config=config or SensorCamera.RenderConfig(enable_shadows=True))
  return _bytes(color.numpy())[..., :3]


@pytest.mark.parametrize("setup", ["default_lights", "create_default_light", "ambient"])
def test_lighting_follows_the_up_axis(setup):
  """B36: the default lights, Newton's default light and the ambient light follow model.up_axis."""

  def prepare(sensor: SensorCamera) -> None:
    if setup == "create_default_light":
      sensor.create_default_light()
    elif setup == "ambient":
      sensor.create_default_light()
      sensor.set_ambient_light((0.6, 0.6, 0.6), (0.05, 0.05, 0.05))

  z_up, y_up = (_color(*_up_scene(up), setup=prepare) for up in ("Z", "Y"))
  assert z_up.mean() > 5.0
  assert np.mean(np.abs(z_up.astype(int) - y_up.astype(int)) <= 2) > 0.98


def test_default_light_color_and_ambient_light():
  """B36 (Newton main): create_default_light(color=) and set_ambient_light(sky, ground)."""
  scene = _up_scene("Z")
  white = _color(*scene, setup=lambda s: s.create_default_light())
  red = _color(*scene, setup=lambda s: s.create_default_light(color=(1.0, 0.2, 0.2)))
  assert red[..., 0].sum() > 4.0 * red[..., 1].sum() and white[..., 0].sum() == white[..., 1].sum()  # tinted by the light

  def ambient(sensor: SensorCamera) -> None:
    sensor.create_default_light()
    sensor.set_ambient_light((0.6, 0.6, 0.6), (0.05, 0.05, 0.05))

  lit = _color(*scene, setup=ambient)
  assert lit.sum() > 1.2 * white.sum()  # the ambient lights the shadowed side
  unlit = _color(*scene, setup=ambient, config=SensorCamera.RenderConfig(enable_shadows=True, enable_ambient_lighting=False))
  np.testing.assert_array_equal(unlit, white)


def test_render_config_flags_may_be_truthy_ints():
  """Newton's RenderConfig takes truthy ints for its flags: the renderer's settings get bools."""
  model, state, cams = _scene(nworld=1)
  rays = SensorCamera.compute_camera_rays_pinhole(8, 8, camera_fov=_FOV, device="cpu")
  config = SensorCamera.RenderConfig(enable_shadows=1, enable_backface_culling=0)  # pyrefly: ignore[bad-argument-type]
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model)
    sensor.update(state, cams, rays, color_image=sensor.allocate_image_color(1, 8, 8), render_config=config)
  settings = required(sensor.renderer).settings
  assert settings.shadows.enabled is True and settings.surfaces.backface_culling is False


def test_infinite_max_distance_is_unbounded():
  """Newton takes `RenderConfig.max_distance=inf` (no far limit); it raised "zfar must be a finite
  distance". It renders unbounded (zfar 0), a finite one is the far clip, 0 or less raises."""
  model, state, cams = _scene(nworld=1)
  rays = SensorCamera.compute_camera_rays_pinhole(8, 8, camera_fov=_FOV, device="cpu")
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model)
    color = sensor.allocate_image_color(1, 8, 8)
    sensor.update(state, cams, rays, color_image=color, render_config=SensorCamera.RenderConfig(max_distance=float("inf")))
    assert required(sensor.renderer).clips[0][1] == 0.0
    sensor.update(state, cams, rays, color_image=color, render_config=SensorCamera.RenderConfig(max_distance=50.0))
    assert required(sensor.renderer).clips[0][1] == 50.0
    # 0 (Newton: every ray misses) is not taken as zfar 0 (unbounded): it raises.
    for bad in (0.0, -1.0, float("nan")):
      with pytest.raises(ValueError, match="max_distance must be > 0"):
        sensor.update(state, cams, rays, color_image=color, render_config=SensorCamera.RenderConfig(max_distance=bad))
    assert required(sensor.renderer).clips[0][1] == 50.0


@pytest.mark.parametrize("emissive", [True, False])
def test_emissive_lights_option_decides_the_panel_fit(emissive):
  """The render options are fixed: `lighting.emissive_lights=False` uploads without fitting the
  emissive panel lights (no fit in the scene's load report), the default fits them."""
  model, state, cams = _scene(nworld=1)
  rays = SensorCamera.compute_camera_rays_pinhole(8, 8, camera_fov=_FOV, device="cpu")
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model, **{"lighting.emissive_lights": emissive})
    sensor.update(state, cams, rays, color_image=sensor.allocate_image_color(1, 8, 8))
  assert ("lights.emissive_panels" in required(sensor.renderer).scene.report.counts) == emissive


def test_scene_changes_and_close_release_the_renderers(monkeypatch):
  """A light, ambient or checkerboard change rebuilds the scene: its renderers are closed (the
  garbage collector releases no OptiX build), as by `close`; a graph captured before
  refuses to replay (its kernels read the dropped scene's buffers)."""
  import types

  from raywarp.integrations.newton.sensor import camera as sensor_module

  model, state, cams = _scene()
  rays = SensorCamera.compute_camera_rays_pinhole(_RES, _RES, camera_fov=_FOV, device=model.device)
  closed: list[object] = []
  real = rw.Renderer.close

  def close(r: rw.Renderer) -> None:
    closed.append(r)
    real(r)

  monkeypatch.setattr(rw.Renderer, "close", close)
  with wp.ScopedDevice("cpu"):
    sensor = SensorCamera(model)
    color = sensor.allocate_image_color(2, _RES, _RES)

    def update() -> None:
      sensor.update(state, cams, rays, color_image=color)

    def fake_capture(device: wp.Device, step: Callable[[], None]) -> types.SimpleNamespace:
      """A CPU stand-in for the capture: the eager run, no recording (`Segments.launch` runs its
      checks first)."""
      del device
      step()
      return types.SimpleNamespace(checks=[])

    monkeypatch.setattr(sensor_module.render_graph, "capture", fake_capture)
    graph = sensor.capture(update)
    for check in graph.checks:
      check()  # unchanged: the replay goes on
    first = sensor.renderer
    sensor.create_default_light()
    assert closed == [first] and sensor.renderer is None
    with pytest.raises(RuntimeError, match="capture again"):
      for check in graph.checks:
        check()
    update()
    second = sensor.renderer
    assert second is not None and second is not first
    sensor.close()
    assert closed == [first, second]


def test_every_public_newton_method_is_on_the_drop_in():
  """raywarp's SensorCamera has every public method of Newton's (drop-in parity, including
  `allocate_image_color_hdr`, `compute_camera_transforms_usd`, `sync_deformable_meshes`)."""
  newton_methods = {n for n, v in vars(NewtonSensorCamera).items() if not n.startswith("_") and callable(v)}
  missing = sorted(n for n in newton_methods if not callable(getattr(SensorCamera, n, None)))
  assert not missing
  for name in ("allocate_image_color_hdr", "compute_camera_transforms_usd", "sync_deformable_meshes"):
    assert callable(getattr(SensorCamera, name))


def test_constructor_takes_newtons_keywords():
  """Every keyword of Newton's constructor is accepted (drop-in parity). The deformable surface
  mode: BOTH and SIMULATION_ONLY render the simulation triangles; VISUAL_ONLY (Newton's
  deformable visual meshes only) raises."""
  import inspect

  ours = inspect.signature(SensorCamera.__init__).parameters
  assert set(inspect.signature(NewtonSensorCamera.__init__).parameters) <= set(ours)
  assert ours["deformable_surface_mode"].default == NewtonSensorCamera.DeformableSurfaceMode.BOTH
  model, _state, _cams = _scene()
  SensorCamera(model, deformable_surface_mode=SensorCamera.DeformableSurfaceMode.SIMULATION_ONLY)
  with pytest.raises(NotImplementedError, match="VISUAL_ONLY"):
    SensorCamera(model, deformable_surface_mode=SensorCamera.DeformableSurfaceMode.VISUAL_ONLY)
