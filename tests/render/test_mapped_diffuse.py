"""Optional diffuse map storage, canonical input and live shared-texture contracts."""

import numpy as np
import pytest
import warp as wp
from conftest import DOWN, render_host, required, whole_table

import raywarp as rw
from raywarp._base.types import NOPENPBR_TEXROLE, NTEXROLE, TexRole
from raywarp.scene.model.report import LoadReport


def _host(*, mapped: bool, roughness: float = 1.0) -> rw.Scene:
  b = rw.SceneBuilder()
  refs = ()
  if mapped:
    texels = np.broadcast_to(np.array([0, 128, 255], np.uint8), (4, 4, 3)).copy()
    ref = rw.scene.TextureRef(b.add_texture(texels, colorspace="linear"), channel=1, scale=(0.5,) * 4, bias=(0.1,) * 4)
    refs = (("base_diffuse_roughness", ref),)
  mat = b.add_material(
    rw.Material(
      base_color=(0.1, 0.4, 0.8),
      specular_weight=0.0,
      extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=roughness, textures=refs),
    )
  )
  b.add_instance("box", size=(2.0, 2.0, 0.1), material=mat)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0))
  return b.build()


def test_optional_role_storage_is_absent_without_map():
  plain = rw.upload(_host(mapped=False, roughness=0.0), "cpu")
  mapped = rw.upload(_host(mapped=True), "cpu")
  assert plain.tables["material.texture"].shape == (1, 1, NTEXROLE)
  assert plain.tables["material.tex_affine"].shape == (1, 1, NTEXROLE)
  assert mapped.tables["material.texture"].shape == (1, 1, NOPENPBR_TEXROLE)
  assert mapped.tables["material.tex_affine"].shape == (1, 1, NOPENPBR_TEXROLE)
  assert mapped.capabilities.textures.diffuse_roughness
  assert "gpu.material.texture.base_diffuse_roughness" not in mapped.report.dropped


def test_mapped_channel_affine_matches_uniform_constant():
  factor = 0.8
  mapped = render_host(_host(mapped=True, roughness=factor), (0.0, 0.0, 2.0), fovy=0.1, res=2)
  expected = factor * (0.5 * (128.0 / 255.0) + 0.1)
  plain = render_host(_host(mapped=False, roughness=expected), (0.0, 0.0, 2.0), fovy=0.1, res=2)
  try:
    assert mapped.features.diffuse_roughness_map
    np.testing.assert_allclose(required(mapped.outputs.hdr).numpy(), required(plain.outputs.hdr).numpy(), rtol=2e-5, atol=1e-6)
  finally:
    mapped.close()
    plain.close()


@pytest.mark.parametrize(
  "device,backend",
  [
    ("cpu", "software"),
    pytest.param("cuda:0", "software", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph")),
    pytest.param("cuda:0", "optix", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="OptiX graph")),
  ],
)
def test_affine_updates_are_world_local_and_keep_texture_storage(device, backend):
  from raywarp.core.pipeline.platform.device import device_caps

  if backend == "optix" and not device_caps(device).optix.available:
    pytest.skip("OptiX unavailable")
  renderer = rw.Renderer(
    rw.upload(_host(mapped=True), device),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=0.1)],
    resolution=(2, 2),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      outputs=rw.Outputs(hdr=rw.settings.FloatOutput()),
      performance=rw.settings.Performance(traversal=backend),
    ),
    per_world=rw.PerWorld(material=["tex_affine"]),
  )
  try:
    renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, *DOWN]]] * 2, np.float32))
    renderer.render()
    captured = renderer.capture() if device != "cpu" else None
    before = required(renderer.outputs.hdr).numpy().copy()
    texture_table = renderer.scene.tables["material.texture"]
    tex_pointer = texture_table.ptr
    affine = whole_table(renderer.per_world["material.tex_affine"])
    pointer = affine.ptr
    rows = affine.numpy()
    rows[1, :, TexRole.DIFFUSE_ROUGHNESS, 0] = 0.0
    rows[1, :, TexRole.DIFFUSE_ROUGHNESS, 1] = 1.0
    affine.assign(rows)
    if captured is None:
      renderer.render()
    else:
      captured.launch()
    after = required(renderer.outputs.hdr).numpy().copy()
    assert affine.ptr == pointer and texture_table.ptr == tex_pointer
    np.testing.assert_array_equal(after[0], before[0])
    assert np.max(np.abs(after[1] - before[1])) > 0.01
    renderer.render()
    np.testing.assert_array_equal(required(renderer.outputs.hdr).numpy(), after)
  finally:
    renderer.close()


def test_declared_optional_role_reserves_once_without_authored_map():
  b = rw.SceneBuilder()
  texture = b.add_texture(np.full((2, 2, 3), 128, np.uint8), colorspace="linear")
  mat = b.add_material(
    rw.Material(
      base_color=(0.1, 0.4, 0.8), specular_weight=0.0, textures=((rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(texture)),)
    )
  )
  b.add_instance("box", size=(2.0, 2.0, 0.1), material=mat)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0))
  scene = rw.upload(b.build(), "cpu")
  renderer = rw.Renderer(
    scene,
    nworld=2,
    cameras=[rw.CameraSpec(fovy=0.1)],
    resolution=(2, 2),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      outputs=rw.Outputs(hdr=rw.settings.FloatOutput()),
      performance=rw.settings.Performance(traversal="software"),
    ),
    per_world=rw.PerWorld(material=["base_diffuse_roughness", "texture"]),
  )
  try:
    assert scene.tables["material.texture"].shape[2] == NTEXROLE
    assert scene.tables["material.tex_affine"].shape[2] == NTEXROLE
    ids = renderer.per_world["material.texture"]
    assert ids.shape == (2, 1, NOPENPBR_TEXROLE)
    assert np.all(ids.numpy()[:, :, TexRole.DIFFUSE_ROUGHNESS] == -1)
    renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, *DOWN]]] * 2, np.float32))
    renderer.render()
    before = required(renderer.outputs.hdr).numpy().copy()
    rows = ids.numpy()
    rows[1, :, TexRole.DIFFUSE_ROUGHNESS] = rows[1, :, TexRole.RGB]
    ids.assign(rows)
    renderer.per_world["material.base_diffuse_roughness"].assign(np.array([[0.0], [1.0]], np.float32))
    renderer.render()
    after = required(renderer.outputs.hdr).numpy()
    np.testing.assert_array_equal(after[0], before[0])
    assert np.max(np.abs(after[1] - before[1])) > 0.01
  finally:
    renderer.close()


def test_roughness_extra_uv_provenance_and_conflicts():
  from raywarp.core.pipeline.upload.uv_plans import uv_plans

  b = rw.SceneBuilder()
  texture = b.add_texture(np.full((2, 2, 3), 128, np.uint8), colorspace="linear")
  rough = rw.scene.TextureRef(texture, uv_set=1, uv_xform=((0.0, -2.0, 0.25), (3.0, 0.0, 0.5)))
  b.add_material(
    rw.Material(extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=1.0, textures=(("base_diffuse_roughness", rough),)))
  )
  b.add_material(
    rw.Material(
      textures=((rw.scene.Slot.BASE_COLOR, rw.scene.TextureRef(texture)),),
      extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=1.0, textures=(("base_diffuse_roughness", rough),)),
    )
  )
  host = b.build()
  report = LoadReport()
  plans = uv_plans(host, report)
  assert plans.uv_set.tolist() == [1, 0]
  assert plans.transformed.tolist() == [True, False]
  np.testing.assert_allclose(plans.repeat[0, :, None] * plans.xform[0], rough.uv_xform)
  assert report.approximated["gpu.material.uv_transform(one per material: the base_color slot's)"] == 1
  assert report.approximated["gpu.material.uv_set(one per material: the base_color slot's)"] == 1


def test_color_contrast_preserves_optional_scalar_affine():
  from raywarp import randomize
  from raywarp._base.types import TEX_IDENTITY

  table = np.tile(TEX_IDENTITY, (2, 3, NOPENPBR_TEXROLE, 1, 1))
  varied = randomize.contrast(table, np.random.default_rng(5), amount=0.5)
  np.testing.assert_array_equal(varied[:, :, TexRole.DIFFUSE_ROUGHNESS], table[:, :, TexRole.DIFFUSE_ROUGHNESS])
  assert np.any(varied[:, :, TexRole.RGB] != table[:, :, TexRole.RGB])


def test_usd_openpbr_connected_diffuse_texture_is_prepared(tmp_path):
  pytest.importorskip("pxr")
  pytest.importorskip("MaterialX")
  from PIL import Image
  from pxr import Sdf, Usd, UsdGeom, UsdShade

  import raywarp.formats.usd as usd

  Image.fromarray(np.full((2, 2), 128, np.uint8)).save(tmp_path / "roughness.png")
  stage = Usd.Stage.CreateNew(str(tmp_path / "mapped.usda"))
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  material = UsdShade.Material.Define(stage, "/Looks/Rough")
  surface = UsdShade.Shader.Define(stage, "/Looks/Rough/Surface")
  surface.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
  image = UsdShade.Shader.Define(stage, "/Looks/Rough/Image")
  image.CreateIdAttr("ND_image_float")
  image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("roughness.png")
  surface.CreateInput("base_diffuse_roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
    image.CreateOutput("out", Sdf.ValueTypeNames.Float)
  )
  material.CreateSurfaceOutput("mtlx").ConnectToSource(surface.CreateOutput("out", Sdf.ValueTypeNames.Token))
  cube = UsdGeom.Cube.Define(stage, "/Cube")
  UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
  builder = usd.read(stage)
  canonical = next(m for m in builder.materials if m.name == "/Looks/Rough")
  assert canonical.extras is not None
  assert canonical.extras.base_diffuse_roughness == 1.0
  assert canonical.extras.textures[0][0] == "base_diffuse_roughness"
  uploaded = rw.upload(builder.build(), "cpu")
  assert uploaded.capabilities.textures.diffuse_roughness
  assert not any("base_diffuse_roughness" in key for key in uploaded.report.dropped)


def test_disabled_diffuse_mapping_retains_lambert_output():
  mapped = render_host(_host(mapped=True), (0.0, 0.0, 2.0), fovy=0.1, res=2, settings={"openpbr.diffuse_roughness": False})
  plain = render_host(_host(mapped=False, roughness=0.0), (0.0, 0.0, 2.0), fovy=0.1, res=2)
  try:
    assert not mapped.features.diffuse_roughness_map
    assert not mapped.features.textures
    np.testing.assert_array_equal(required(mapped.outputs.hdr).numpy(), required(plain.outputs.hdr).numpy())
  finally:
    mapped.close()
    plain.close()


def test_material_swap_selects_live_mesh_uv_frame():
  b = rw.SceneBuilder()
  texture = b.add_texture(np.full((2, 2), 128, np.uint8), colorspace="linear")
  ref = rw.scene.TextureRef(texture, uv_xform=((0.0, -1.0, 0.25), (1.0, 0.0, 0.5)))
  first = b.add_material(
    rw.Material(extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=1.0, textures=(("base_diffuse_roughness", ref),)))
  )
  b.add_material(
    rw.Material(
      extras=rw.scene.OpenPbrExtras(
        base_diffuse_roughness=1.0, textures=(("base_diffuse_roughness", rw.scene.TextureRef(texture)),)
      )
    )
  )
  mesh = b.add_mesh(
    np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
    np.array([[0, 1, 2]]),
    uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
  )
  b.add_instance("mesh", mesh=mesh, material=first)
  renderer = rw.Renderer(
    rw.upload(b.build(), "cpu"),
    nworld=2,
    cameras=[rw.CameraSpec()],
    resolution=(2, 2),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      performance=rw.settings.Performance(traversal="software"),
    ),
    per_world=rw.PerWorld(material=["instance.material"]),
  )
  try:
    assert renderer.scene.capabilities.uv_maps
    key = "gpu.instance.material(mesh UV set/transform stays tied to prepared instance)"
    assert key not in renderer.scene.report.approximated
    assert renderer.features.live_material_uv
  finally:
    renderer.close()


def test_layered_maps_are_not_reported_dropped():
  # Every layered map has a texture role (the layered role width): rendered, so never "dropped".
  b = rw.SceneBuilder()
  texels = np.full((4, 4, 3), 128, np.uint8)
  ref = rw.scene.TextureRef(b.add_texture(texels, colorspace="linear"))
  extras = rw.scene.OpenPbrExtras(fuzz_weight=1.0, textures=(("fuzz_color", ref), ("fuzz_roughness", ref)))
  mat = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), extras=extras))
  b.add_instance("box", size=(1.0, 1.0, 0.1), material=mat)
  scene = rw.upload(b.build(), "cpu")
  assert not {k for k in scene.report.dropped if k.startswith("gpu.material.texture.")}
