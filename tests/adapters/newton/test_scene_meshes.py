"""rw.integrations.newton.scene: Newton mesh attributes and textures, and models rendered on
another device."""

import numpy as np
import pytest
import warp as wp
from conftest import required

newton = pytest.importorskip("newton")

from test_scene import _lights, _scenery_model

import raywarp as rw
import raywarp.integrations.newton as newton_pbr
from raywarp.scene.images import colorspace
from raywarp.scene.model.tables import Images


def test_mesh_texture_files_get_the_auto_color_space(tmp_path):
  """A float (HDR) mesh texture is linear data, an 8-bit one sRGB (as every other adapter path)."""
  # A flat Radiance RGBE file of 0.5 everywhere (mantissa 128, exponent 128 + 0).
  hdr = tmp_path / "base.hdr"
  hdr.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n" + bytes([128, 128, 128, 128]) * 4)
  png = tmp_path / "base.png"
  from PIL import Image

  Image.fromarray(np.full((2, 2, 3), 128, np.uint8)).save(png)
  verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
  uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], np.float32)
  builder = newton.ModelBuilder()
  for path in (hdr, png):
    mesh = newton.Mesh(verts, np.array([0, 1, 2], np.int32), uvs=uvs, texture=str(path))
    builder.add_shape_mesh(-1, mesh=mesh)
  host = newton_pbr.load(builder.finalize(device="cpu"))
  tex = host.textures
  spaces = {host.images.paths[int(i)]: colorspace.from_code(int(c)) for i, c in zip(tex.image, tex.colorspace, strict=True)}
  assert spaces == {str(hdr): rw.scene.ColorSpace.LIN_REC709, str(png): rw.scene.ColorSpace.SRGB_REC709}


@pytest.mark.parametrize("kind", ["grey", "rgb", "float"])
def test_mesh_texture_arrays_match_their_rgba_bytes(kind):
  """Grey, RGB and float [0, 1] image textures decode as the bytes they stand for (the builder
  widens grey / RGB; float images are quantised to bytes, as Newton uploads them)."""
  rgb = np.random.default_rng(3).integers(0, 256, (4, 4, 3), dtype=np.uint8)
  grey = rgb[..., 0]
  given = {"grey": grey, "rgb": rgb, "float": rgb.astype(np.float32) / 255.0}[kind]
  expected = grey[..., None] if kind == "grey" else np.concatenate([rgb, np.full((4, 4, 1), 255, np.uint8)], axis=2)
  verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
  uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], np.float32)

  def images(texture: np.ndarray) -> Images:
    """The image table of a one-mesh model textured with `texture`."""
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh(verts, np.array([0, 1, 2], np.int32), uvs=uvs, texture=texture))
    return newton_pbr.load(builder.finalize(device="cpu")).images

  got, want = images(given), images(expected)
  assert len(got) == 1
  for name in ("width", "height", "channels", "pixel", "levels", "data"):
    np.testing.assert_array_equal(getattr(got, name), getattr(want, name), err_msg=name)


def test_mesh_normals_and_uvs_of_the_wrong_length_are_reported_once_per_mesh():
  """Normals / UVs that do not hold one value per vertex are dropped and reported with their
  counts, once per mesh source (two shapes share it here; they were dropped silently)."""
  verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
  mesh = newton.Mesh(
    verts, np.array([0, 1, 2], np.int32), normals=np.array([[0.0, 0.0, 1.0]] * 2, np.float32), uvs=np.zeros((4, 2), np.float32)
  )
  builder = newton.ModelBuilder()
  for _ in range(2):
    builder.add_shape_mesh(-1, mesh=mesh)
  host = newton_pbr.load(builder.finalize(device="cpu"))
  report = host.report
  assert report.dropped["newton.Mesh.normals.length"] == 1 and report.dropped["newton.Mesh.uvs.length"] == 1
  assert report.examples["newton.Mesh.normals.length"].endswith("(2 normals for 3 vertices)")
  assert report.examples["newton.Mesh.uvs.length"].endswith("(4 uvs for 3 vertices)")
  assert host.meshes.normal_num.tolist() == [0] and host.meshes.uv_num.tolist() == [0]


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="needs CUDA (a model and a render device apart)")
def test_shared_scenery_renders_on_another_device_than_the_model():
  """A CPU model rendered on CUDA (`binding(device=)`): the shared shapes' `shape_transform` is
  copied to the render device, for their poses and the moved-scenery check (a CPU array was read
  by CUDA launches), and a moved global shape still raises."""
  nworld = 2
  model, state = _scenery_model(nworld)  # on the CPU
  host = newton_pbr.load(model, lighting=_lights())
  scene = rw.upload(host, "cuda:0", binding=newton_pbr.binding(host, model, device="cuda:0"))
  cylinder = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.CYLINDER))[0])
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"), performance=rw.settings.Performance(traversal="software"))
  r = rw.Renderer(scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(24, 16), settings=settings)
  rot = wp.quat_from_matrix(wp.mat33(1.0, 0.0, 0.0, 0.0, 0.5, -0.866, 0.0, 0.866, 0.5))
  r.camera_pose.assign(np.tile(np.array([0.0, -2.5, 1.5, *rot], np.float32), (nworld, 1, 1)))
  before = required(r.render(state).depth).numpy().copy()
  assert np.isfinite(before).any()
  tf = model.shape_transform.numpy()
  tf[cylinder, 0] -= 0.5
  model.shape_transform.assign(tf)
  r.render(state)
  with pytest.raises(ValueError, match=r"shared scenery .* moved or changed after the first frame"):
    r.render(state)


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="needs CUDA (a model and a render device apart)")
def test_foreign_device_arrays_upload_once_per_frame_and_replay_in_a_graph(monkeypatch):
  """A CPU model on CUDA: each model array (body_q, shape_transform ...) is uploaded through a
  pinned buffer once per render (`Binding.stage`), with one reused event each; a CUDA graph's
  replays read the copies the upload before each launch refreshes: after a CPU-side body move
  the graph's frame is the eager render's."""
  from raywarp.integrations.newton import staging

  staged: list[str] = []
  real, events = staging.Staging.upload, []
  real_event = wp.Event

  def counting(self, name, array):
    staged.append(name)
    real(self, name, array)

  def event(*args, **kwargs):
    events.append(1)
    return real_event(*args, **kwargs)

  monkeypatch.setattr(staging.Staging, "upload", counting)
  monkeypatch.setattr(staging.wp, "Event", event)
  nworld = 2
  model, state = _scenery_model(nworld)
  host = newton_pbr.load(model, lighting=_lights())
  scene = rw.upload(host, "cuda:0", binding=newton_pbr.binding(host, model, device="cuda:0"))
  settings = rw.RenderSettings(outputs=rw.Outputs.of("depth"), performance=rw.settings.Performance(traversal="software"))
  r = rw.Renderer(scene, nworld=nworld, cameras=[rw.CameraSpec(fovy=60.0)], resolution=(24, 16), settings=settings)
  rot = wp.quat_from_matrix(wp.mat33(1.0, 0.0, 0.0, 0.0, 0.5, -0.866, 0.0, 0.866, 0.5))
  r.camera_pose.assign(np.tile(np.array([0.0, -2.5, 1.5, *rot], np.float32), (nworld, 1, 1)))
  # Shared scenery may stage shape_transform during upload/preparation. Count its event too,
  # then measure per-frame uploads independently of that initialization.
  staged.clear()
  first = required(r.render(state).depth).numpy().copy()
  r.render(state)
  assert staged.count("body_q") == 2 and staged.count("shape_transform") == 2, staged
  graph = r.capture(None, state)
  graph.launch(state)
  np.testing.assert_array_equal(required(r.outputs.depth).numpy(), first)
  q = state.body_q.numpy()
  q[:, 0] += 0.3  # every body moves on the CPU
  state.body_q.assign(q)
  graph.launch(state)
  replayed = required(r.outputs.depth).numpy().copy()
  assert np.any(replayed != first)
  np.testing.assert_array_equal(required(r.render(state).depth).numpy(), replayed)
  # One reused event per staged CPU array (body_q, shape_transform; no world indices / offsets).
  assert len(events) == 2, events


def test_a_model_on_the_render_device_stages_nothing():
  """Same device: the binding reads the model's arrays themselves (no copy, nothing staged)."""
  model, state = _scenery_model(2)
  host = newton_pbr.load(model, lighting=_lights())
  binding = newton_pbr.binding(host, model)
  binding.stage(state)
  sources = binding.sources(state)  # what the pose kernel reads: no staged copy among them
  assert set(sources) == {"state.body_q"} and sources["state.body_q"] is state.body_q
