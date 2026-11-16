"""Prepared scalar geometry: actual heights, UV semantics, budgets, sharing and cache reuse."""

from pathlib import Path

import numpy as np
import pytest

import raywarp as rw
from raywarp.scene.build.displacement import sample_height
from raywarp.scene.model.types import Wrap


def quad(builder: rw.SceneBuilder):
  return builder.add_mesh(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 1, 2], [0, 2, 3]],
    uvs=[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
  )


def test_constant_displacement_instances_share_geometry_and_cache(tmp_path: Path):
  builder = rw.SceneBuilder()
  mesh = quad(builder)
  material = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=0.2)))
  builder.add_instances("mesh", mesh=mesh, material=material, xforms=np.tile(np.eye(4), (4, 1, 1)))
  scene = builder.build(tmp_path / "displaced.wpbr")
  assert np.all(scene.instances.mesh == scene.instances.mesh[0])
  actual = scene.meshes.get(int(scene.instances.mesh[0]))
  np.testing.assert_allclose(actual.points[:, 2], 0.2)
  np.testing.assert_allclose(actual.normals, np.tile([0, 0, 1], (len(actual.points), 1)))
  loaded = rw.Scene.load(tmp_path / "displaced.wpbr")
  np.testing.assert_array_equal(loaded.meshes.points, scene.meshes.points)
  assert scene.report.counts["displacement.mesh_variants"] == 1


def test_texture_height_channel_scale_bias_and_bottom_left_uv():
  builder = rw.SceneBuilder(displacement_options=rw.scene.DisplacementOptions(max_level=0))
  tex = builder.add_texture(np.array([[0, 1], [2, 3]], np.float32)[..., None], colorspace="data")
  ref = rw.scene.TextureRef(tex, channel=0, scale=(0.1,) * 4, bias=(-0.1,) * 4)
  material = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=ref)))
  builder.add_instance("mesh", mesh=quad(builder), material=material)
  scene = builder.build()
  points = scene.meshes.get(int(scene.instances.mesh[0])).points
  # np.unique's point ordering is independent of the original quad's indexing.
  for x, y, z in points:
    expected = (2.0 * (1.0 - y) + x) * 0.1 - 0.1
    assert z == pytest.approx(expected, abs=2e-7)


def test_refinement_budget_and_zero_cost_unauthored_path():
  builder = rw.SceneBuilder(displacement_options=rw.scene.DisplacementOptions(max_level=6, max_triangles=8))
  tex = builder.add_texture(np.ones((32, 32, 1), np.float32), colorspace="data")
  mat = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=rw.scene.TextureRef(tex, channel=0))))
  base = quad(builder)
  builder.add_instance("mesh", mesh=base, material=mat)
  scene = builder.build()
  assert scene.meshes.tri_num[scene.instances.mesh[0]] == 8
  assert scene.report.approximated["displacement.refinement_limited"] == 1
  plain = rw.SceneBuilder()
  base = quad(plain)
  mat = plain.add_material(rw.Material(displacement=rw.scene.Displacement(amount=0.0)))
  plain.add_instance("mesh", mesh=base, material=mat)
  untouched = plain.build()
  assert untouched.instances.mesh[0] == base
  assert untouched.meshes.tris.shape == (2, 3)
  assert not untouched.report.counts["displacement.mesh_variants"]


@pytest.mark.parametrize("mode", [Wrap.REPEAT, Wrap.CLAMP, Wrap.MIRROR, Wrap.BLACK])
def test_height_sampler_half_texel_and_wrap(mode: Wrap):
  image = np.array([[1, 2], [3, 4]], np.float32)
  uv = np.array([[0.25, 0.25], [0.75, 0.75], [-0.25, 0.25]], np.float32)
  sampled = sample_height(image, uv, (int(mode), int(mode)))
  expected = {Wrap.REPEAT: 4, Wrap.CLAMP: 3, Wrap.MIRROR: 3, Wrap.BLACK: 0}[mode]
  np.testing.assert_allclose(sampled, [3, 2, expected])


def test_displacement_rejects_ambiguous_rgb_and_bad_limits():
  builder = rw.SceneBuilder()
  tex = builder.add_texture(np.ones((1, 1, 1), np.float32), colorspace="data")
  with pytest.raises(ValueError, match="scalar channel"):
    builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=rw.scene.TextureRef(tex))))
  with pytest.raises(ValueError, match="nonnegative integer"):
    rw.scene.DisplacementOptions(max_level=-1)
  with pytest.raises(ValueError, match="finite scalar"):
    builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=float("nan"))))


def test_preview_displacement_without_terminal_and_stage_units(tmp_path: Path):
  pytest.importorskip("pxr")
  from pxr import Sdf
  from rw_helpers import bind, new_stage, preview_material
  from rw_helpers import quad as usd_quad

  from raywarp.formats import usd

  stage = new_stage(tmp_path, mpu=0.01)
  mesh = usd_quad(stage, "/Mesh")
  material, shader = preview_material(stage, "/Looks/Height")
  shader.CreateInput("displacement", Sdf.ValueTypeNames.Float).Set(2.0)
  bind(mesh.GetPrim(), material)
  scene = usd.read(stage).build()
  instance = int(scene.instances.mesh[0])
  np.testing.assert_allclose(scene.meshes.get(instance).points[:, 2], 2.0)
  assert scene.instances.xform[0, 2, 2] == pytest.approx(0.01)
  assert scene.report.approximated["UsdPreviewSurface.displacement.no_terminal"] == 1
  disabled = usd.read(stage, adapter=rw.formats.usd.Options(displacement=False)).build()
  np.testing.assert_array_equal(disabled.meshes.get(int(disabled.instances.mesh[0])).points[:, 2], 0)
  assert disabled.report.dropped["displacement.disabled"] == 1


def test_uv_seams_keep_welded_geometry_closed():
  builder = rw.SceneBuilder(displacement_options=rw.scene.DisplacementOptions(max_level=0))
  mesh = builder.add_mesh(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 1, 2], [3, 4, 5]],
    uvs=[[0.25, 0.5]] * 3 + [[0.75, 0.5]] * 3,
  )
  tex = builder.add_texture(np.array([[[0], [1]]], np.float32), colorspace="data", wrap=(Wrap.CLAMP, Wrap.CLAMP))
  mat = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=rw.scene.TextureRef(tex, channel=0))))
  builder.add_instance("mesh", mesh=mesh, material=mat)
  scene = builder.build()
  actual = scene.meshes.get(int(scene.instances.mesh[0]))
  assert len(actual.points) == 4
  assert len(set(actual.tris[0]) & set(actual.tris[1])) == 2
  assert scene.report.approximated["displacement.uv_seam_averaged"] == 1


def test_near_coincident_seam_copies_weld_closed():
  # A procedural UV sphere: its u = 0 / u = 1 seam copies differ by sin(2 pi) ~ 1e-16 (float32 keeps
  # it); welding by exact equality left a crack along the seam once displaced.
  n_u, n_v = 16, 8
  u, v = np.meshgrid(np.linspace(0, 1, n_u + 1), np.linspace(0, 1, n_v + 1))
  phi, theta = u * 2 * np.pi, v * np.pi
  points = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)], -1).reshape(-1, 3)
  assert not np.array_equal(points[0 :: n_u + 1].astype(np.float32), points[n_u :: n_u + 1].astype(np.float32))
  idx = np.arange((n_v + 1) * (n_u + 1)).reshape(n_v + 1, n_u + 1)
  q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
  tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
  builder = rw.SceneBuilder(displacement_options=rw.scene.DisplacementOptions(max_level=2))
  mesh = builder.add_mesh(points, tris, uvs=np.stack([u, v], -1).reshape(-1, 2))
  # A uniform height map: refined like any texture (pole fans included), moved by exactly 0.2.
  tex = builder.add_texture(np.ones((64, 64, 1), np.float32), colorspace="data")
  height = rw.scene.TextureRef(tex, channel=0, scale=(0.2,) * 4)
  material = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=height)))
  builder.add_instance("mesh", mesh=mesh, material=material)
  scene = builder.build()
  actual = scene.meshes.get(int(scene.instances.mesh[0]))
  assert len(actual.tris) == (n_u * n_v * 2 - 2 * n_u) * 4**2  # the zero-area pole triangles dropped, refined twice
  edges = np.sort(actual.tris[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
  _, uses = np.unique(edges, axis=0, return_counts=True)
  assert np.all(uses >= 2), "open edges: the displaced surface cracked at the seam"
  # Linear refinement keeps chords inside the sphere; a pole midpoint left unmoved would sit at 1.
  assert np.linalg.norm(actual.points, axis=1).min() > 1.1


@pytest.mark.parametrize("normal,reported", [((1, 0, 0), 1), ((0, 0, 2), 0)])
def test_displacement_reports_replaced_authored_directions(normal: tuple[int, int, int], reported: int):
  builder = rw.SceneBuilder()
  material = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=0.2)))
  mesh = builder.add_mesh(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    [[0, 1, 2]],
    normals=[normal],
    normal_idx=[[0, 0, 0]],
  )
  builder.add_instance("mesh", mesh=mesh, material=material)
  scene = builder.build()
  assert scene.report.approximated["displacement.authored_normals_reconstructed"] == reported


def test_height_filtered_to_coarse_geometry_and_zero_scale_needs_no_uv():
  builder = rw.SceneBuilder(displacement_options=rw.scene.DisplacementOptions(max_level=0))
  checker = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float32)[..., None]
  tex = builder.add_texture(checker, colorspace="data")
  mat = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=rw.scene.TextureRef(tex, channel=0))))
  builder.add_instance("mesh", mesh=quad(builder), material=mat)
  constant = builder.add_material(
    rw.Material(
      displacement=rw.scene.Displacement(amount=rw.scene.TextureRef(tex, channel=0, scale=(0.0,) * 4, bias=(0.25,) * 4))
    )
  )
  no_uv = builder.add_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
  builder.add_instance("mesh", mesh=no_uv, material=constant)
  scene = builder.build()
  np.testing.assert_allclose(scene.meshes.get(int(scene.instances.mesh[0])).points[:, 2], 0.5)
  np.testing.assert_allclose(scene.meshes.get(int(scene.instances.mesh[1])).points[:, 2], 0.25)


def test_repeated_usd_import_preserves_builder_preparation_policy(tmp_path: Path):
  pytest.importorskip("pxr")
  from pxr import Sdf
  from rw_helpers import bind, new_stage, preview_material
  from rw_helpers import quad as usd_quad

  from raywarp.formats import usd

  stage = new_stage(tmp_path)
  material, shader = preview_material(stage, "/Looks/Height")
  shader.CreateInput("displacement", Sdf.ValueTypeNames.Float).Set(0.2)
  bind(usd_quad(stage, "/Mesh"), material)
  builder = rw.SceneBuilder(displacement_options=rw.scene.DisplacementOptions(mode="off", max_level=2))
  assert usd.read(stage, builder) is builder
  assert usd.read(stage, builder) is builder
  before = builder.ninstance
  with pytest.raises(ValueError, match="conflicts with the supplied"):
    usd.read(stage, builder, adapter=rw.formats.usd.Options(displacement=True))
  assert builder.ninstance == before
  assert builder.displacement_options == rw.scene.DisplacementOptions(mode="off", max_level=2)
  scene = builder.build()
  np.testing.assert_array_equal(scene.meshes.points[:, 2], 0)
  # Public convenience entrypoints must create their own builder with matching options.
  # In particular, cached rebuild must not look like a caller-supplied default builder.
  stage.GetRootLayer().Save()
  source = stage.GetRootLayer().realPath
  for cache in (None, tmp_path / "cache"):
    loaded = usd.load(source, cache=cache, adapter=rw.formats.usd.Options(displacement=False, displacement_max_level=2))
    np.testing.assert_array_equal(loaded.meshes.points[:, 2], 0)
    if cache is not None:
      warm = usd.load(source, cache=cache, adapter=rw.formats.usd.Options(displacement=False, displacement_max_level=2))
      assert "warm_load" in warm.report.timings
  baked = usd.bake(source, tmp_path / "off.wpbr", adapter=rw.formats.usd.Options(displacement=False, displacement_max_level=2))
  np.testing.assert_array_equal(baked.meshes.points[:, 2], 0)


def test_height_filter_never_pads_skinny_images_to_a_square(monkeypatch: pytest.MonkeyPatch):
  from raywarp.scene.build import displacement
  from raywarp.scene.model.types import Texels

  original = displacement.mips.box_reduce
  shapes: list[tuple[int, ...]] = []

  def bounded(image: Texels, factor: int) -> Texels:
    shapes.append(image.shape)
    assert factor <= 2
    return original(image, factor)

  monkeypatch.setattr(displacement.mips, "box_reduce", bounded)
  values = np.tile(np.array([0, 1], np.float32), 4096)[None, :]
  got = displacement._filtered_height(values, 8192, 0)
  np.testing.assert_array_equal(got, [[0.5]])
  assert all(shape[0] == 1 and shape[1] <= 8192 for shape in shapes)


def test_displaced_material_part_compacts_unused_source_vertices():
  builder = rw.SceneBuilder()
  parts = builder.add_mesh_parts(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [100, 0, 0], [101, 0, 0], [100, 1, 0]],
    [[0, 1, 2], [3, 4, 5]],
    tri_part=[0, 1],
    nparts=2,
  )
  material = builder.add_material(rw.Material(displacement=rw.scene.Displacement(amount=0.2)))
  builder.add_instance("mesh", mesh=parts[0], material=material)
  scene = builder.build()
  changed = int(scene.instances.mesh[0])
  assert scene.meshes.vert_num[changed] == 3
  np.testing.assert_allclose(scene.meshes.aabb[changed], [[0, 0, 0.2], [1, 1, 0.2]])


def test_invalid_displacement_inputs_raise_naming_the_material():
  """A textured displacement read through a UV set the mesh does not have is an invalid input:
  the build raises (it used to keep the flat mesh with a report)."""
  builder = rw.SceneBuilder()
  tex = builder.add_texture(np.zeros((2, 2, 1), np.float32), colorspace="data")
  ref = rw.scene.TextureRef(tex, channel=0, uv_set=1)  # the quad has no second UV set
  material = builder.add_material(rw.Material(name="bumpy", displacement=rw.scene.Displacement(amount=ref)))
  builder.add_instance("mesh", mesh=quad(builder), material=material)
  with pytest.raises(ValueError, match="displacement of material 'bumpy'"):
    builder.build()
