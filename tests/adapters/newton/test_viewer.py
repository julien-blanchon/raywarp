"""ViewerRaywarpHeadless (no window, no OpenGL) and the PBR frames on the Warp CPU device.

The windowed ViewerRaywarp (a ViewerGL + `BackgroundHook`, CUDA-GL texture copies) needs a display:
checked by hand / on the GPU checkpoint.
"""

import math

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton")
pytest.importorskip("pyglet")  # Newton's viewer camera (vector maths only)

from conftest import recipe

import raywarp as rw
import raywarp.integrations.newton as rw_newton
from raywarp.integrations.newton.binding import NewtonBinding
from raywarp.integrations.newton.headless import ViewerRaywarpHeadless
from raywarp.integrations.newton.headless.look import GL_SKY_LOWER
from raywarp.settings.fields import exposure_from_ev100

_W, _H = 48, 32
_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


def _model(nworld: int = 1, device: str = "cpu") -> tuple[newton.Model, newton.State]:
  """A box body and a sphere per world on a ground plane, and its state."""
  world = newton.ModelBuilder()
  body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
  world.add_shape_sphere(-1, xform=wp.transform((0.6, 0.3, 0.3), wp.quat_identity()), radius=0.2, color=(0.1, 0.8, 0.2))
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(world, nworld)
  with wp.ScopedDevice(device):
    model = scene.finalize(device=device)
  return model, model.state()


def _run(viewer: ViewerRaywarpHeadless, model: newton.Model, state: newton.State, frames: int = 2) -> np.ndarray:
  """The last frame of `viewer` run until it stops (`frames` frames) on `state`."""
  viewer.set_model(model)
  viewer.set_camera(wp.vec3(3.0, -2.0, 1.5), pitch=-20.0, yaw=150.0)
  count = 0
  while viewer.is_running():
    viewer.begin_frame(0.01 * count)
    viewer.log_state(state)
    viewer.end_frame()
    count += 1
  assert count == frames
  return viewer.get_frame().numpy()


@pytest.mark.parametrize("device", _DEVICES)
def test_headless_frame_matches_sensor_camera(device):
  model, state = _model(device=device)
  with wp.ScopedDevice(device):
    # SensorCamera's lighting (the viewer's default is Newton's GL look, tested below) and its
    # direct light only (the viewer's balanced look has probe GI; the sensor has none).
    viewer = ViewerRaywarpHeadless(
      width=_W,
      height=_H,
      num_frames=2,
      lighting=rw.Lighting(rw_newton.default_lights()),
      gi=recipe("fast").gi,
    )
    frame = _run(viewer, model, state)
    assert frame.shape == (_H, _W, 3) and frame.dtype == np.uint8
    assert frame.std() > 5.0  # box, sphere and floor, not a flat image
    # The same camera through SensorCamera (pinhole rays, Newton's camera frame).
    camera = viewer.camera
    front, up = np.asarray(camera.get_front()), np.asarray(camera.get_up())
    right = np.cross(front, up)
    rot = np.stack([right, up, -front], axis=1).astype(np.float32)
    pose = wp.transformf(wp.vec3f(*np.asarray(camera.pos, dtype=np.float32)), wp.quat_from_matrix(wp.mat33f(*rot.ravel())))
    from raywarp.integrations.newton.sensor import SensorCamera

    sensor = SensorCamera(model)
    rays = sensor.compute_camera_rays_pinhole(_W, _H, camera_fov=math.radians(camera.fov), device=device)
    color = sensor.allocate_image_color(1, _W, _H)
    config = sensor.RenderConfig(enable_shadows=True, enable_textures=True)
    sensor.update(state, wp.array([pose], dtype=wp.transformf, device=device), rays, color_image=color, render_config=config)
  rgb = np.stack([np.right_shift(color.numpy()[0], s) & 0xFF for s in (0, 8, 16)], axis=-1)
  diff = np.abs(rgb.astype(int) - frame.astype(int)).max(axis=-1)
  assert np.mean(diff <= 2) > 0.99


@pytest.mark.parametrize("device", _DEVICES)
def test_default_look_is_newton_gl(device):
  """Without lights or envmap: GL's horizon sky color as background and a checkered ground."""
  model, state = _model(device=device)
  with wp.ScopedDevice(device):
    viewer = ViewerRaywarpHeadless(width=96, height=64, num_frames=1)
    viewer.set_model(model)
    # Looking down at the ground from 3 m, then towards the horizon.
    viewer.set_camera(wp.vec3(0.3, 0.3, 3.0), pitch=-89.0, yaw=0.0)
    ground = _run(viewer, model, state, frames=1)
    viewer = ViewerRaywarpHeadless(width=96, height=64, num_frames=1)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(0.0, -5.0, 0.3), pitch=10.0, yaw=90.0)
    sky = _run(viewer, model, state, frames=1)
  # 1 m squares in a ~2.5 m wide view: both checker greys, far apart.
  lum = ground.astype(float).mean(axis=-1)
  assert np.percentile(lum, 90) - np.percentile(lum, 10) > 3.0
  expected = np.round(np.asarray(GL_SKY_LOWER) * 255.0)
  assert np.abs(sky[0].astype(float) - expected).max() <= 2.0


def test_default_look_uses_gl_values():
  """ViewerGL's own values (3 lux sun, 0.7 x hemisphere ambient, exposure 1.6) times the look
  preset (sun x 1.2, hemisphere x 0.4, PBR Neutral: ViewerGL's appearance)."""
  from raywarp.integrations.newton.headless.look import gl_look

  look = gl_look(2)
  (sun,) = look.lights
  assert (
    sun.intensity == pytest.approx(1.2 * 3.0)
    and np.allclose(np.linalg.norm(sun.direction, keepdims=True), 1.0)
    and sun.direction[2] < 0.0
  )
  sky, ground = (0.4 * 0.56, 0.4 * 0.56, 0.4 * 0.595), (0.4 * 0.21, 0.4 * 0.21, 0.4 * 0.245)
  assert look.ambient.sky == pytest.approx(sky) and look.ambient.ground == pytest.approx(ground)
  assert exposure_from_ev100(look.ev100) == pytest.approx(1.6)
  assert look.settings["post.tone_mapper"] == "pbr_neutral"
  # The hemisphere's Lambertian irradiance is GL's mix(ground, sky, (n.up + 1) / 2).
  n = 64
  env = look.ambient.envmap(2, height=n).astype(np.float64)  # (n, 2n, 3), row 0 is +z
  theta = (np.arange(n) + 0.5) / n * np.pi
  for normal_z, expected in ((1.0, look.ambient.sky), (-1.0, look.ambient.ground)):
    # E / pi = 1/pi * sum L cos+ dw, dw = sin(theta) (pi / n)^2 (a normal along +-z).
    cosine = np.maximum(normal_z * np.cos(theta), 0.0) * np.sin(theta) * (np.pi / n) ** 2 / np.pi
    np.testing.assert_allclose((env * cosine[:, None, None]).sum(axis=(0, 1)), expected, rtol=0.01)


def test_visibility_toggles_select_rendered_shapes(tmp_path):
  """Show Ground / Visual / Collision and the visible worlds: a visibility mask, no scene
  rebuild."""
  world = newton.ModelBuilder()
  body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25)  # visual and collision
  collision_cfg = newton.ModelBuilder.ShapeConfig(is_visible=False)
  world.add_shape_sphere(body, radius=0.2, cfg=collision_cfg)  # collision only
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(world, 2)
  model = scene.finalize(device="cpu")
  flags, kind, owner = model.shape_flags.numpy(), model.shape_type.numpy(), model.shape_world.numpy()
  ground = int(np.flatnonzero(kind == int(newton.GeoType.PLANE))[0])
  hidden_spheres = set(np.flatnonzero((flags & int(newton.ShapeFlags.VISIBLE)) == 0).tolist())
  assert len(hidden_spheres) == 2

  viewer = ViewerRaywarpHeadless(width=_W, height=_H, num_frames=1)
  viewer.set_model(model)
  gpu = viewer.pbr.scene
  assert gpu is not None and gpu.ninstance == model.shape_count  # every shape a toggle can show

  def shown() -> set[int]:
    """The shapes the PBR frame shows after rendering a frame."""
    viewer.log_state(model.state())
    viewer.render_pbr()
    r = viewer.pbr.renderer
    assert r is not None and viewer.pbr.scene is gpu  # no rebuild
    mask = r.per_world["instance.visible"].numpy()[0]
    binding = gpu.binding
    assert isinstance(binding, NewtonBinding)
    return {int(s) for s, m in zip(binding.shape0, mask, strict=True) if m}

  assert ground in shown() and not hidden_spheres & shown()
  viewer.show_ground = False
  assert ground not in shown()
  viewer.show_collision = True
  assert hidden_spheres <= shown()
  viewer.set_visible_worlds([1])
  assert all(owner[s] in (-1, 1) for s in shown())


def test_headless_lifecycle():
  model, state = _model(nworld=2)
  with wp.ScopedDevice("cpu"):
    viewer = ViewerRaywarpHeadless(width=_W, height=_H, num_frames=3, **{"aa.spp": 2})
    with pytest.raises(RuntimeError, match="no frame"):
      viewer.get_frame()
    frame = _run(viewer, model, state, frames=3)
    # Logging is a no-op without a window.
    viewer.log_lines("lines", wp.zeros(1, dtype=wp.vec3), wp.zeros(1, dtype=wp.vec3), (1.0, 0.0, 0.0))
    viewer.log_scalar("x", 1.0)
    out = wp.zeros((_H, _W, 3), dtype=wp.uint8)
    assert viewer.get_frame(output=out) is out
    np.testing.assert_array_equal(out.numpy(), frame)
    with pytest.raises(ValueError, match="Shape of `output`"):
      viewer.get_frame(output=wp.zeros((_H, _W + 1, 3), dtype=wp.uint8))
    # No UI to draw: `ViewerBase`'s contract for an unsupported capture option.
    with pytest.raises(NotImplementedError, match="render_ui"):
      viewer.get_frame(render_ui=True)
    viewer.close()
  # Both worlds are rendered (at the viewer's world offsets): two red boxes.
  red = (frame[..., 0] > 2 * frame[..., 1].astype(int)) & (frame[..., 0] > 60)
  columns = np.flatnonzero(red.any(axis=0))
  assert len(columns) and np.any(np.diff(columns) > 1)


def test_headless_takes_viewer_null_arguments():
  """A drop-in for ViewerNull: its positional `num_frames` and its benchmark arguments."""
  viewer = ViewerRaywarpHeadless(5, True, 2.0, 1, width=_W, height=_H)  # noqa: FBT003  # ViewerNull's signature
  assert viewer.num_frames == 5
  assert (viewer.benchmark, viewer.benchmark_timeout, viewer.benchmark_start_frame) == (True, 2.0, 1)
  assert (viewer.camera.width, viewer.camera.height) == (_W, _H)


def test_headless_frame_size_is_checked():
  """A width or height of 0 rendered 1 x 1 pixels (the frame size was clamped), silently."""
  for size in ({"width": 0}, {"height": -4}, {"width": 1.5}):
    with pytest.raises(ValueError, match=r"ViewerRaywarpHeadless: (width|height) must be at least 1"):
      ViewerRaywarpHeadless(**size)  # pyrefly: ignore[bad-argument-type]  # on purpose


def test_reserved_options():
  with pytest.raises(TypeError, match="cam_res"):
    ViewerRaywarpHeadless(cam_res=(8, 8))
  with pytest.raises(TypeError, match="sets"):
    ViewerRaywarpHeadless(**{"outputs": rw.Outputs.of("rgb")})


@pytest.mark.parametrize("emissive", [True, False])
def test_emissive_lights_option_decides_the_panel_fit(monkeypatch, emissive):
  """The viewer's render options are fixed: `lighting.emissive_lights=False` uploads without
  fitting the emissive panel lights, the default fits them."""
  import raywarp.integrations.newton.headless.frames as frames_module

  seen: list[bool] = []
  upload = frames_module.upload

  def recording(*args, **kwargs):
    seen.append(kwargs["options"].emissive_lights)
    return upload(*args, **kwargs)

  # Spies the viewer's call of the public `rw.upload` (the viewer exposes no scene or renderer).
  monkeypatch.setattr(frames_module, "upload", recording)
  model, state = _model()
  with wp.ScopedDevice("cpu"):
    viewer = ViewerRaywarpHeadless(width=_W, height=_H, num_frames=1, **{"lighting.emissive_lights": emissive})
    _run(viewer, model, state, frames=1)
    viewer.close()
  assert seen == [emissive]


def test_frames_follow_the_viewer_camera_near_plane():
  """The PBR frame clips at the viewer camera's near and far planes (ViewerGL's projection), so
  geometry ViewerGL's overlays clip is clipped in the ray-traced view too."""
  from newton._src.viewer.camera import Camera

  from raywarp.integrations.newton.headless.frames import PbrFrames

  model, state = _model()
  frames = PbrFrames(rw.Lighting(rw_newton.default_lights()), {}, owner="test", particles="-")
  frames.set_model(model)
  camera = Camera(width=_W, height=_H)
  camera.near, camera.far = 0.25, 40.0
  frames.render(state, camera, size=(_W, _H), world_offsets=None, root=None, seed=0)
  r = frames.renderer
  assert r is not None
  assert r._rig.clips[0] == pytest.approx((0.25, 40.0))  # the renderer's clip table (`set_clip`)


def test_frames_close_the_renderers_they_drop(monkeypatch):
  """A resize evicting a size, a new model and `set_model(None)` (the viewers' close) drop
  renderers: each is closed (its OptiX build and compile threads released), not left
  to the garbage collector, which never releases them."""
  from newton._src.viewer.camera import Camera

  from raywarp.integrations.newton.headless.frames import PbrFrames

  closed: list[object] = []
  real = rw.Renderer.close

  def close(r: rw.Renderer) -> None:
    closed.append(r)
    real(r)

  monkeypatch.setattr(rw.Renderer, "close", close)
  model, state = _model()
  frames = PbrFrames(rw.Lighting(rw_newton.default_lights()), {}, owner="test", particles="-")
  frames.set_model(model)
  camera = Camera(width=_W, height=_H)

  def render(size: tuple[int, int]) -> rw.Renderer:
    assert frames.render(state, camera, size=size, world_offsets=None, root=None, seed=0) is not None
    r = frames.renderer
    assert r is not None
    return r

  first = render((8, 8))
  second = render((12, 8))
  third = render((16, 8))  # each new size evicts the previous one
  assert closed == [first, second]
  frames.set_model(model)
  assert closed == [first, second, third]
  render((8, 8))
  frames.set_model(None)
  assert len(closed) == 4


def test_the_default_viewer_renders_the_bundled_build():
  """`ViewerRaywarpHeadless()` renders with the default's sampling (BUG-772: its old `accumulate=32`
  selected a build no tier holds: a 10-25 min compile on a Newton user's first frame)."""
  viewer = ViewerRaywarpHeadless(width=_W, height=_H, num_frames=1)
  assert viewer.pbr._settings().aa == rw.RenderSettings().aa


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"checker_size": 0}, "checker_size"),
    ({"checker_size": 2.5}, "checker_size"),
    ({"resolution": -4}, "resolution"),
    ({"colors": (0x80, 300)}, "colors"),
  ],
)
def test_checkerboard_rejects_invalid_sizes(kwargs: dict[str, object], match: str) -> None:
  """A checkerboard of a zero or fractional checker size (a NumPy division warning, then a broken
  texture), a negative resolution or a grey outside a byte raises at construction."""
  from raywarp.integrations.newton.types import Checkerboard

  with pytest.raises(ValueError, match=match):
    Checkerboard(frozenset({0}), **kwargs)  # pyrefly: ignore[bad-argument-type]  # invalid on purpose
  assert Checkerboard(frozenset({0})).image().shape == (64, 64, 3)
