"""MuJoCo Warp's renderer API on raywarp: same calls, same output dtypes and values for
depth and segmentation."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp
from conftest import recipe, required

from raywarp.integrations import mujoco as mjr

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1"/>
    <geom type="sphere" pos="0 0 0.4" size="0.3"/>
    <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    <camera name="top" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
  """(mjm, m, d) of the test scene, two worlds."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjw.put_model(mjm), mjw.put_data(mjm, mjd, nworld=2)


def _both(mod, mjm, m, d, **kwargs):
  """The same create_render_context / render on module `mod` (mujoco_warp or raywarp)."""
  rc = mod.create_render_context(mjm, nworld=2, **kwargs)
  if mod is mjw:
    mjw.refit_bvh(m, d, rc)
  mod.render(m, d, rc)
  return rc


def _silhouette_ties(ours: np.ndarray, ref: np.ndarray) -> np.ndarray:
  """(nworld, h, w) pixels whose geom ids differ at a silhouette: two ray-intersection codes may
  round a ray grazing an edge differently (CUDA: the top camera's pixel (7, 16) passes exactly
  through the box's vertical edge x = y = 0.4, by the camera's symmetry). A tie is a pixel whose
  id `ours` gives appears in the reference's 3x3 neighbourhood; at most one per world and camera
  (anything else is a convention difference and fails)."""
  differ = ours != ref
  padded = np.pad(ref, ((0, 0), (1, 1), (1, 1)), mode="edge")
  h, w = ref.shape[1:]
  near = np.zeros_like(differ)
  for dy in range(3):
    for dx in range(3):
      near |= padded[:, dy : dy + h, dx : dx + w] == ours
  ties = differ & near
  assert (ties.reshape(len(ties), -1).sum(axis=1) <= 1).all(), f"{ties.sum()} silhouette pixels differ"
  return ties


def test_depth_and_segmentation_match_mujoco_warp(model):
  """get_depth / get_segmentation give MuJoCo Warp's values per camera, with per-camera
  resolutions and a camera selection by name."""
  mjm, m, d = model
  kwargs = {
    "cam_res": [(40, 30), (24, 24)],
    "render_rgb": True,
    "render_depth": True,
    "render_seg": True,
    "cam_active": ["front", "top"],
  }
  ours, ref = _both(mjr, mjm, m, d, **kwargs), _both(mjw, mjm, m, d, **kwargs)
  for cam, (w, h) in enumerate(kwargs["cam_res"]):
    outs = {}
    for mod, rc in ((mjr, ours), (mjw, ref)):
      depth = wp.zeros((2, h, w), dtype=float)
      seg = wp.zeros((2, h, w), dtype=wp.vec2i)
      rgb = wp.zeros((2, h, w), dtype=wp.vec3)
      mod.get_depth(rc, cam, 5.0, depth)
      mod.get_segmentation(rc, cam, seg)
      mod.get_rgb(rc, cam, rgb)
      outs[mod.__name__] = (depth.numpy(), seg.numpy(), rgb.numpy())
    (d0, s0, c0), (d1, s1, c1) = outs[mjr.__name__], outs[mjw.__name__]
    ties = _silhouette_ties(s0[..., 0], s1[..., 0])
    np.testing.assert_array_equal(s0[~ties], s1[~ties])
    np.testing.assert_allclose(d0[~ties], d1[~ties], rtol=1e-4, atol=1e-5)
    assert c0.shape == c1.shape and 0.0 <= c0.min() and c0.max() <= 1.0 and c0.max() > 0.2


def test_settings_map_and_errors(model):
  """MuJoCo Warp arguments map to RenderSettings over the base settings; reading before the first
  render, or an output not rendered, raises."""
  mjm, m, d = model
  rc = mjr.create_render_context(
    mjm,
    nworld=2,
    cam_res=(16, 16),
    render_rgb=True,
    render_depth=False,
    render_seg=False,
    use_shadows=False,
    samples_per_pixel=2,
    use_fast_math=True,
  )
  assert not rc.settings.shadows.enabled and rc.settings.aa.spp == 2 and rc.settings.outputs.kinds == {"rgb"}
  with pytest.raises(ValueError, match="must run before"):
    mjr.get_rgb(rc, 0, wp.zeros((2, 16, 16), dtype=wp.vec3))
  mjr.render(m, d, rc)
  with pytest.raises(ValueError, match="does not render depth"):
    mjr.get_depth(rc, 0, 1.0, wp.zeros((2, 16, 16), dtype=float))
  with pytest.raises(ValueError, match="not found"):
    mjr.create_render_context(mjm, cam_active=["nope"])


def test_get_outputs_check_the_callers_array_shape(model):
  """An output array not shaped (nworld, height, width) like the camera's raises instead of
  reading past the rendered image."""
  mjm, m, d = model
  rc = mjr.create_render_context(mjm, nworld=2, cam_res=(16, 12), render_rgb=True, render_depth=True, render_seg=True)
  mjr.render(m, d, rc)
  with pytest.raises(ValueError, match=r"shape \(2, 12, 16\)"):
    mjr.get_rgb(rc, 0, wp.zeros((2, 16, 16), dtype=wp.vec3))
  with pytest.raises(ValueError, match="shape"):
    mjr.get_depth(rc, 0, 1.0, wp.zeros((3, 12, 16), dtype=float))
  with pytest.raises(ValueError, match="shape"):
    mjr.get_segmentation(rc, 0, wp.zeros((2, 12, 17), dtype=wp.vec2i))


def test_get_outputs_check_the_camera_index(model):
  """`get_rgb(rc, -1, ...)` read the last camera (a Python index wraps) and an index past the
  cameras raised "tuple index out of range": both raise ValueError naming the range."""
  mjm, m, d = model
  rc = mjr.create_render_context(mjm, nworld=2, cam_res=(16, 12), render_rgb=True)
  mjr.render(m, d, rc)
  for index in (-1, 2, True):
    with pytest.raises(ValueError, match=r"camera_index must be in \[0, 1\]"):
      mjr.get_rgb(rc, index, wp.zeros((2, 12, 16), dtype=wp.vec3))


def test_context_arguments_fail_at_creation():
  """nworld=0 and a geom group no geom can be in were accepted until the first render (or never
  raised: a group of 7 rendered nothing of it); an RGB background failed with "tuple index out of
  range" and a float camera id with a NumPy indexing error."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  for kwargs, match in (
    ({"nworld": 0}, "nworld must be at least 1"),
    ({"enabled_geom_groups": [0, 7]}, r"geom_groups must be integers in \[0, 6\)"),
    ({"background_color": (1.0, 0.0, 0.0)}, "background_color is opaque RGBA"),
    ({"cam_active": [0.0]}, "Invalid cam_active format"),
  ):
    with pytest.raises(ValueError, match=match):
      mjr.create_render_context(mjm, render_rgb=True, **kwargs)  # pyrefly: ignore[bad-argument-type]


def test_camera_ids_out_of_range_raise():
  """`cam_active` ids outside [0, ncam) name no camera: they raise like an unknown name (a
  negative id would else wrap to the last camera, a large one fail deep in the scan)."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  for bad in ([2], [-1], [0, 5]):
    with pytest.raises(ValueError, match="not found"):
      mjr.create_render_context(mjm, render_rgb=True, cam_active=bad)
  assert mjr.create_render_context(mjm, render_rgb=True, cam_active=[1]).cameras == [1]


def test_unnamed_camera_does_not_collide_with_a_named_one():
  """An unnamed camera 0 ("camera0") and a camera named "camera0" are two scene cameras: camera
  ids index `camera_names` (the context renders camera 1 from camera 1's pose)."""
  xml = XML.replace('<camera name="front"', "<camera").replace('name="top"', 'name="camera0"')
  mjm = mujoco.MjModel.from_xml_string(xml)
  scene = mjr.scene(mjm, mjw.put_model(mjm))
  assert scene.camera_names == ["camera0#", "camera0"]
  # The suffixed name taken too: still one name per camera.
  xml = xml.replace('name="camera0"', 'name="camera0#"').replace("</worldbody>", '<camera name="camera0"/></worldbody>')
  mjm = mujoco.MjModel.from_xml_string(xml)
  assert mjr.scene(mjm, mjw.put_model(mjm)).camera_names == ["camera0##", "camera0#", "camera0"]


def test_per_camera_render_flags_follow_mujoco_warp():
  """A `render_*` list of one flag per model camera reads the active cameras' entries (an
  inactive camera's True renders nothing, as in MuJoCo Warp); a list of another length raises."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  rc = mjr.create_render_context(mjm, cam_active=[0], render_rgb=True, render_depth=[False, True])
  assert rc.settings.outputs.kinds == {"rgb"}
  rc = mjr.create_render_context(mjm, cam_active=[1], render_rgb=True, render_depth=[False, True])
  assert rc.settings.outputs.kinds == {"rgb", "depth"}
  with pytest.raises(ValueError, match="render_depth length"):
    mjr.create_render_context(mjm, render_rgb=True, render_depth=[False, True, True])


def test_truthy_int_flags_are_accepted(model):
  """MuJoCo Warp takes truthy ints for its flags: `use_shadows=1` sets a bool setting."""
  mjm, _, _ = model
  rc = mjr.create_render_context(mjm, nworld=1, use_shadows=1, use_textures=0, enable_backface_culling=1)
  s = rc.settings
  assert s.shadows.enabled is True and s.surfaces.textures is False and s.surfaces.backface_culling is True


def test_options_past_use_textures_are_keyword_only(model):
  """MuJoCo Warp's 8th parameter is use_fast_math (ours would be use_shadows): a positional call
  past `use_textures` raises instead of binding the wrong option."""
  mjm, _, _ = model
  with pytest.raises(TypeError):
    mjr.create_render_context(mjm, 1, (8, 8), True, False, False, True, True)  # pyrefly: ignore[bad-argument-count]
  rc = mjr.create_render_context(mjm, 1, (8, 8), True, False, False, True, use_shadows=False)
  assert rc.settings.surfaces.textures is True and rc.settings.shadows.enabled is False


def test_every_mujoco_warp_parameter_is_accepted_by_name(model):
  """The drop-in takes every parameter of MuJoCo Warp's `create_render_context` by name, with
  MuJoCo Warp's defaults (a call written for MuJoCo Warp runs unchanged)."""
  import inspect

  mjm, _, _ = model
  ref = inspect.signature(mjw.create_render_context).parameters
  ours = inspect.signature(mjr.create_render_context).parameters
  assert set(ref) <= set(ours)
  assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ours.values())
  defaults = {
    k: p.default for k, p in ref.items() if k not in {"mjm", "cam_active"} and p.default is not inspect.Parameter.empty
  }
  rc = mjr.create_render_context(mjm, **defaults)
  assert rc.nworld == 1


def test_no_argument_is_silently_ignored(model):
  """An unknown argument (a typo) raises; MuJoCo Warp shading options off their default, splat
  data and a translucent background raise instead of being dropped."""
  mjm, _, _ = model
  with pytest.raises(TypeError):
    mjr.create_render_context(mjm, render_segmentaion=True)  # pyrefly: ignore[unexpected-keyword]
  with pytest.raises(ValueError, match="enable_specular=False"):
    mjr.create_render_context(mjm, enable_specular=False)
  with pytest.raises(ValueError, match="shadow_light_fraction"):
    mjr.create_render_context(mjm, shadow_light_fraction=0.0)
  with pytest.raises(ValueError, match="splats"):
    mjr.create_render_context(mjm, splat_position=np.zeros((1, 3), np.float32))
  with pytest.raises(ValueError, match="alpha must be 1"):
    mjr.create_render_context(mjm, background_color=(0.1, 0.2, 0.3, 0.5))
  rc = mjr.create_render_context(mjm, background_color=(0.1, 0.2, 0.3, 1.0), use_fast_math=False, deterministic=True)
  assert rc.settings.environment.background_color == pytest.approx((0.1, 0.2, 0.3))


def test_explicit_flags_beat_the_base_settings(model):
  """An explicit MuJoCo Warp flag wins over `settings=`; an unset one keeps the base's value."""

  mjm, _, _ = model
  base = recipe("fast").replace(**{"shadows.enabled": True, "aa.spp": 3})
  rc = mjr.create_render_context(mjm, nworld=1, use_shadows=False, settings=base)
  assert rc.settings.shadows.enabled is False and rc.settings.aa.spp == 3


def test_policy_path_binds_channels_first_planes(model):
  """`color=ColorOutput(channels=3, planar=True)` + `bind_outputs(rgb=...)` render the
  observation straight into the caller's (nworld * ncam, 3, H, W) uint8 array: the bytes of the
  default packed RGBA's channels; `get_rgb` then has no RGBA to expand."""
  import raywarp as rw

  mjm, m, d = model
  flags = {"cam_res": (16, 12), "render_rgb": True, "render_depth": False, "render_seg": False}
  ref = _both(mjr, mjm, m, d, **flags)
  rgba = ref.renderer.outputs.rgb.numpy()  # (nworld, ncam, H, W, 4)
  rc = mjr.create_render_context(mjm, nworld=2, color=rw.settings.ColorOutput(channels=3, planar=True), **flags)
  obs = wp.zeros((2 * mjm.ncam, 3, 12, 16), dtype=wp.uint8)
  mjr.bind_outputs(m, rc, rgb=obs)
  mjr.render(m, d, rc)
  planes = obs.numpy().reshape(2, mjm.ncam, 3, 12, 16)
  np.testing.assert_array_equal(planes, np.moveaxis(rgba[..., :3], -1, 2))
  with pytest.raises(ValueError, match="reads the RGBA color"):
    mjr.get_rgb(rc, 0, wp.zeros((2, 12, 16), dtype=wp.vec3))


@wp.kernel
def _count(n: wp.array1d[int]):
  n[0] += 1


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="graph capture needs CUDA")
def test_capture_replays_step_and_render():
  """`capture(m, d, rc, step)` records the step and the render: a replay runs the step's device
  work and writes the eager depth into the cleared output."""
  mjm = mujoco.MjModel.from_xml_string(XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  with wp.ScopedDevice("cuda:0"):
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd, nworld=2)
    rc = mjr.create_render_context(mjm, nworld=2, cam_res=(16, 12), render_rgb=False, render_depth=True, render_seg=False)
    mjr.render(m, d, rc)
    eager = wp.zeros((2, 12, 16), dtype=float)
    mjr.get_depth(rc, 0, 10.0, eager)
    steps = wp.zeros(1, dtype=int)

    def step() -> None:
      wp.launch(_count, 1, inputs=[steps])

    graph = mjr.capture(m, d, rc, step=step)
    assert steps.numpy()[0] == 1  # the eager run before the recording
    required(required(rc.renderer).outputs.depth).zero_()  # a replay that writes nothing fails below
    replayed = wp.zeros((2, 12, 16), dtype=float)
    graph.launch()
    assert steps.numpy()[0] == 2  # the replay ran the step
    mjr.get_depth(rc, 0, 10.0, replayed)
    np.testing.assert_array_equal(replayed.numpy(), eager.numpy())
    assert (eager.numpy() > 0).any()
