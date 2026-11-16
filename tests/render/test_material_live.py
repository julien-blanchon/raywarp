"""Tests for resolving unset PBR material fields."""

import dataclasses
from typing import Literal

import mujoco
import numpy as np
import pytest
import warp as wp
from conftest import required, whole_table

import raywarp as rw
from raywarp._base.types import DEFAULT_MATERIAL_PARAMS
from raywarp.integrations.mujoco import materials as material
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import UV_SET_WORLD, AlphaMode, Slot

_XML = """
  <mujoco>
    <asset>
      <material name="authored" metallic="0.3" roughness="0.6" specular="0.7"/>
      <material name="legacy" shininess="0.8" rgba="0.5 0.5 0.5 1"/>
      <material name="white" rgba="0.9 0.9 0.9 1"/>
    </asset>
    <worldbody>
      <geom name="authored" type="sphere" size="0.1" material="authored"/>
      <geom name="legacy" type="sphere" size="0.1" material="legacy"/>
      <geom name="white" type="sphere" size="0.1" material="white"/>
      <geom name="nomat" type="sphere" size="0.1" rgba="0.1 0.1 0.1 1"/>
      <geom name="floor" type="plane" size="1 1 0.1" material="legacy"/>
    </worldbody>
  </mujoco>
"""


def _resolve(**kwargs):
  """(metallic, roughness, reflectance) per geom: its material's (`m.mat_specular` for the
  reflectance)."""
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mats = material.openpbr_materials(mjm, LoadReport(), **kwargs)
  out = {}
  for g in range(mjm.ngeom):
    mat = int(mjm.geom_matid[g])
    name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g)
    if mat < 0:
      out[name] = DEFAULT_MATERIAL_PARAMS
      continue
    m = mats[mat]
    out[name] = (m.base_metalness, m.specular_roughness, float(mjm.mat_specular[mat]))
  return out


def test_gltf_policy():
  r = _resolve()
  np.testing.assert_allclose(r["authored"], (0.3, 0.6, 0.7), rtol=1e-6)
  # Unset: metallic 0, roughness 1 - shininess, reflectance = material specular (default 0.5).
  np.testing.assert_allclose(r["legacy"], (0.0, 0.2, 0.5), rtol=1e-6)
  np.testing.assert_allclose(r["nomat"], (0.0, 0.5, 0.5), rtol=1e-6)


def test_unset_values_only_fill_unset_fields():
  r = _resolve(unset_metallic=1.0, unset_roughness=0.1)
  np.testing.assert_allclose(r["authored"][:2], (0.3, 0.6), rtol=1e-6)
  np.testing.assert_allclose(r["legacy"][:2], (1.0, 0.1), rtol=1e-6)


def test_unset_values_reach_the_scene():
  import mujoco_warp as mjw

  import raywarp as rw
  import raywarp.integrations.mujoco

  mjm = mujoco.MjModel.from_xml_string(_XML)
  scene = rw.integrations.mujoco.scene(mjm, mjw.put_model(mjm), adapter=rw.integrations.mujoco.Options(unset_roughness=0.25))
  roughness = scene.tables["material.specular_roughness"].numpy()[0]
  np.testing.assert_allclose(roughness[1], 0.25)  # "legacy" leaves roughness unset
  np.testing.assert_allclose(roughness[0], 0.6, rtol=1e-6)  # authored


# ---------------------------------------------------------------------------------------------
# Live material UV selection must match preparing the selected material directly.


def _scene(
  kind: Literal["set", "transform", "world"],
  *,
  selected: int,
  mapped_normal: bool = False,
  cutout: bool = False,
  initial_role: bool = True,
  second_uv: bool = True,
) -> rw.Scene:
  builder = rw.SceneBuilder()
  texels = np.zeros((8, 8, 3), np.uint8)
  texels[:4, :4] = (255, 32, 16)
  texels[:4, 4:] = (16, 255, 32)
  texels[4:, :4] = (32, 16, 255)
  texels[4:, 4:] = (255, 255, 255)
  texture = builder.add_texture(texels, colorspace="linear")
  first = rw.scene.TextureRef(texture)
  second = (
    rw.scene.TextureRef(texture, uv_set=1)
    if kind == "set"
    else rw.scene.TextureRef(texture, uv_xform=((0.0, -1.0, 1.0), (1.0, 0.0, 0.0)))
  )
  if kind == "world":
    second = rw.scene.TextureRef(texture, uv_set=UV_SET_WORLD, uv_xform=((0.0, -0.5, 0.5), (0.5, 0.0, 0.5)))
  normal_texture = builder.add_texture(np.full((2, 2, 3), (220, 128, 220), np.uint8), colorspace="linear")
  opacity = np.zeros((8, 8), np.uint8)
  opacity[:, :4] = 255
  opacity_texture = builder.add_texture(opacity, colorspace="linear")
  materials = []
  for index, ref in enumerate((first, second)):
    active = initial_role or index == 1
    material = rw.Material(specular_weight=0.0)
    if active or mapped_normal or cutout:
      material = material.with_texture(Slot.BASE_COLOR, ref)
    if mapped_normal and active:
      material = material.with_texture(
        Slot.GEOMETRY_NORMAL, rw.scene.TextureRef(normal_texture, uv_set=ref.uv_set, uv_xform=ref.uv_xform)
      )
    if cutout and active:
      material = dataclasses.replace(
        material, ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5)
      ).with_texture(Slot.GEOMETRY_OPACITY, rw.scene.TextureRef(opacity_texture, uv_set=ref.uv_set, uv_xform=ref.uv_xform))
    materials.append(builder.add_material(material))
  mesh = builder.add_mesh(
    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
    [[0, 1, 2], [0, 2, 3]],
    uvs=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    uvs1=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) if second_uv else None,
  )
  builder.add_instance("mesh", mesh=mesh, material=materials[selected])
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0))
  return builder.build()


def _renderer(
  scene: rw.Scene, *, worlds: int, vary: bool, device: str = "cpu", backend: Literal["software", "optix"] = "software"
) -> rw.Renderer[None]:
  renderer = rw.Renderer(
    rw.upload(scene, device),
    nworld=worlds,
    cameras=[rw.CameraSpec(fovy=40.0)],
    resolution=(16, 16),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      shadows=rw.settings.Shadows(enabled=False),
      lighting=rw.settings.Lights(camera_light=False),
      outputs=rw.Outputs.of("hdr", "uv", "normal"),
      performance=rw.settings.Performance(traversal=backend),
    ).replace(),
    per_world=rw.PerWorld(material=["instance.material"] if vary else []),
  )
  renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]] * worlds, np.float32))
  return renderer


def _assert_matches(kind: Literal["set", "transform", "world"], effect: str, *, initial_role: bool) -> None:
  dynamic = _renderer(
    _scene(kind, selected=0, mapped_normal=effect == "normal", cutout=effect == "cutout", initial_role=initial_role),
    worlds=2,
    vary=True,
  )
  fresh = _renderer(
    _scene(kind, selected=1, mapped_normal=effect == "normal", cutout=effect == "cutout", initial_role=initial_role),
    worlds=1,
    vary=False,
  )
  try:
    dynamic.render(seed=17)
    before = required(dynamic.outputs.hdr).numpy().copy()
    table = dynamic.per_world["instance.material"]
    values = table.numpy()
    values[1, 0] = 1
    table.assign(values)
    dynamic.reset_history(wp.array([False, True], dtype=bool, device="cpu"))
    dynamic.render(seed=17)
    fresh.render(seed=17)
    actual = required(dynamic.outputs.hdr).numpy()
    expected = required(fresh.outputs.hdr).numpy()
    np.testing.assert_array_equal(actual[0], before[0])
    # Establish that the reference selects a genuinely different mapping before checking parity.
    assert float(np.max(np.abs(expected[0] - before[0]))) > 0.1
    np.testing.assert_allclose(actual[1], expected[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
      required(dynamic.outputs.normal).numpy()[1], required(fresh.outputs.normal).numpy()[0], atol=1e-6
    )
    np.testing.assert_allclose(required(dynamic.outputs.uv).numpy()[1], required(fresh.outputs.uv).numpy()[0], atol=1e-6)
  finally:
    dynamic.close()
    fresh.close()


@pytest.mark.parametrize("kind", ["set", "transform"])
@pytest.mark.parametrize("effect", ["color", "normal", "cutout"])
def test_live_material_uv_matches_fresh_preparation(kind: Literal["set", "transform", "world"], effect: str):
  _assert_matches(kind, effect, initial_role=True)


@pytest.mark.parametrize(
  "kind,effect", [("set", "color"), ("transform", "normal"), ("transform", "cutout"), ("world", "color")]
)
def test_prepared_unused_material_role_matches_fresh(kind: Literal["set", "transform", "world"], effect: str):
  _assert_matches(kind, effect, initial_role=False)


@pytest.mark.parametrize("backend", ["software", "optix"])
@pytest.mark.parametrize("kind", ["set", "transform"])
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph material mapping")
def test_live_material_uv_graph_uses_current_material(backend: Literal["software", "optix"], kind: Literal["set", "transform"]):
  from raywarp.core.pipeline.platform.device import device_caps

  if backend == "optix" and not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  dynamic = _renderer(
    _scene(kind, selected=0, mapped_normal=True, cutout=True), worlds=2, vary=True, device="cuda:0", backend=backend
  )
  fresh = _renderer(
    _scene(kind, selected=1, mapped_normal=True, cutout=True), worlds=1, vary=False, device="cuda:0", backend=backend
  )
  try:
    dynamic.render(seed=17)
    before = required(dynamic.outputs.hdr).numpy().copy()
    table = whole_table(dynamic.per_world["instance.material"])
    pointer = table.ptr
    graph = dynamic.capture(seed=17)
    table.assign(np.array([[0], [1]], np.int32))
    graph.launch()
    actual = required(dynamic.outputs.hdr).numpy().copy()
    fresh.render(seed=17)
    np.testing.assert_array_equal(actual[0], before[0])
    np.testing.assert_allclose(actual[1], required(fresh.outputs.hdr).numpy()[0], atol=1e-6, rtol=1e-5)
    dynamic.render(seed=17)
    np.testing.assert_array_equal(required(dynamic.outputs.hdr).numpy(), actual)
    assert table.ptr == pointer
  finally:
    dynamic.close()
    fresh.close()


def test_live_uv_layout_only_adds_two_shared_array_descriptors():
  from raywarp.core.kernels.abi.rays import Cutout, LiveUvCutout
  from raywarp.core.trace.optix.device import params

  # A geometry-only build's alpha test is a Cutout: live UVs add exactly its two shared array
  # descriptors. A shaded build's alpha-test tables are its material carrier, which holds them
  # always (empty without live UVs): nothing is added.
  addition = wp.types.type_size_in_bytes(LiveUvCutout) - wp.types.type_size_in_bytes(Cutout)
  assert addition == 2 * 56
  geometry = params.ParamsLayout(shaded=False)
  live = dataclasses.replace(geometry, live_material_uv=True)
  assert params.size_of(params.params_type(live)) == params.size_of(params.params_type(geometry)) + addition
  shaded = params.ParamsLayout()
  assert params.size_of(params.params_type(dataclasses.replace(shaded, live_material_uv=True))) == params.size_of(
    params.params_type(shaded)
  )


@pytest.mark.parametrize(
  "device,backend",
  [
    ("cpu", "software"),
    pytest.param("cuda:0", "software", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA")),
    pytest.param("cuda:0", "optix", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="OptiX")),
  ],
)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("shape", ["mesh", "box"])
def test_world_projected_cutout_follows_world_mask(
  device: str, backend: Literal["software", "optix"], shared: bool, shape: str
):
  """A rotated mesh without UVs still cuts the authored world-x half plane."""
  from raywarp.core.pipeline.platform.device import device_caps

  if backend == "optix" and not device_caps(device).optix.available:
    pytest.skip("OptiX unavailable")
  builder = rw.SceneBuilder()
  opacity = np.zeros((8, 8), np.uint8)
  opacity[:, :4] = 255
  texture = builder.add_texture(opacity, colorspace="linear")
  mat = builder.add_material(
    rw.Material(ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5)).with_texture(
      Slot.GEOMETRY_OPACITY,
      rw.scene.TextureRef(texture, uv_set=UV_SET_WORLD, uv_xform=((0.5, 0.0, 0.5), (0.0, 0.5, 0.5))),
    )
  )
  mesh = builder.add_mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]])
  transform = np.array([[0, -1, 0, 0.15], [1, 0, 0, 0], [0, 0, 1, 0]], np.float64)
  if shape == "mesh":
    builder.add_instance("mesh", mesh=mesh, material=mat, xform=transform)
  else:
    transform[2, 3] = -0.05
    builder.add_instance("box", size=(1, 1, 0.05), material=mat, xform=transform)
  renderer = rw.Renderer(
    rw.upload(builder.build(), device),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=40)],
    resolution=(16, 16),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      outputs=rw.Outputs.of("depth"),
      performance=rw.settings.Performance(traversal=backend),
    ),
    per_world=rw.PerWorld(pose=[] if shared else [0]),
  )
  try:
    renderer.camera_pose.assign(np.array([[[0, 0, 2, 0, 0, 0, 1]]] * 2, np.float32))
    renderer.render()
    depth = required(renderer.outputs.depth).numpy()
    # Center rays on columns2..6 have world x<0 (opaque);9..13 have x>0 (cut away).
    np.testing.assert_allclose(depth[:, 0, 4:12, 2:7], 2.0, atol=1e-5)
    assert np.all(depth[:, 0, 4:12, 9:14] == 0.0)
  finally:
    renderer.close()


def test_unused_material_role_is_pruned_without_swap_declaration():
  scene = _scene("transform", selected=0, mapped_normal=True, initial_role=False)
  static = _renderer(scene, worlds=1, vary=False)
  dynamic = _renderer(scene, worlds=2, vary=True)
  try:
    assert not static.features.normal_map
    assert dynamic.features.normal_map
    assert not dynamic.features.orm_map and not dynamic.features.emissive_map
    assert not dynamic.features.opacity_map
    tables = dynamic.scene.material_uv
    assert tables is not None
    assert tables.material.shape == (2,)
    assert tables.mesh_second.shape == (1,)
  finally:
    static.close()
    dynamic.close()


def test_missing_second_uv_reports_and_uses_first_set():
  dynamic = _renderer(_scene("set", selected=0, second_uv=False), worlds=2, vary=True)
  fresh = _renderer(_scene("set", selected=1, second_uv=False), worlds=1, vary=False)
  try:
    dynamic.per_world["instance.material"].assign(np.array([[0], [1]], np.int32))
    dynamic.render(seed=17)
    fresh.render(seed=17)
    np.testing.assert_allclose(required(dynamic.outputs.hdr).numpy()[1], required(fresh.outputs.hdr).numpy()[0], atol=1e-6)
    assert dynamic.scene.report.dropped["gpu.material.swap_uv_set1_missing(set 0 used)"] == 1
  finally:
    dynamic.close()
    fresh.close()


def test_auxiliary_ray_factories_compile_live_and_world_uv():
  """Relocation, walls, static (sky / distance) and light-grid rays share the active
  geometry/cutout contract."""
  from raywarp.core.shade.gi import kernels
  from raywarp.core.shade.lights import grid

  renderer = _renderer(_scene("transform", selected=0, cutout=True), worlds=2, vary=True)
  try:
    features = renderer.features
    options = {"trees": features.trees, "alpha_cutout": True, "live_uv": True, "world_uv": True}
    for kernel in (
      kernels.survey_kernel(features.shape_types, **options),
      kernels.walls_kernel(features.shape_types, **options),
      kernels.static_kernel(features.shape_types, **options),
      grid.visibility_kernel(features.shape_types, area=False, **options),
    ):
      wp.load_module(module=kernel.module, device="cpu")
  finally:
    renderer.close()


# ---------------------------------------------------------------------------------------------
# Prepared material selection must preserve its shading capabilities and instance coupling.


type Effect = Literal["vertex_color", "vertex_alpha", "curve", "coat", "mirror", "transmission"]


def _swap_scene(effect: Effect, *, selected: int, coat: float | None = None, colors: bool = True) -> rw.Scene:
  builder = rw.SceneBuilder()
  plain = rw.Material(base_color=(0.6, 0.6, 0.6), specular_weight=0.2, specular_roughness=0.6)
  if effect in ("vertex_color", "vertex_alpha"):
    changed = dataclasses.replace(plain, ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.VERTEX_COLOR))
  elif effect == "curve":
    changed = dataclasses.replace(plain, ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.SPECULAR_CURVE))
  elif effect == "coat":
    changed = dataclasses.replace(plain, coat_weight=1.0, coat_roughness=0.2)
  elif effect == "transmission":
    changed = dataclasses.replace(plain, transmission_weight=0.8)
  else:
    changed = dataclasses.replace(plain, ext=rw.scene.RaywarpExt(mirror=1.0))
  if coat is not None:
    changed = dataclasses.replace(changed, coat_weight=coat)
  materials = [builder.add_material(material) for material in (plain, changed)]
  mesh = builder.add_mesh(
    [[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]],
    [[0, 1, 2], [0, 2, 3]],
    colors=[[0.05, 1.0, 0.1, 0.25 if effect == "vertex_alpha" else 1.0]] * 4 if colors else None,
  )
  builder.add_instance("mesh", mesh=mesh, material=materials[selected])
  if effect == "mirror":
    glow = builder.add_material(rw.Material(emission_luminance=100.0, emission_color=(1.0, 0.01, 0.01)))
    builder.add_instance("sphere", material=glow, size=(1.0, 1.0, 1.0), xform=np.c_[np.eye(3), [0, 0, 4]])
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=1000.0))
  return builder.build()


def _swap_renderer(
  scene: rw.Scene,
  *,
  worlds: int,
  vary: bool,
  fields: tuple[str, ...] = (),
  device: str = "cpu",
  backend: Literal["software", "optix"] = "software",
) -> rw.Renderer[None]:
  renderer = rw.Renderer(
    rw.upload(scene, device),
    nworld=worlds,
    cameras=[rw.CameraSpec(fovy=100.0)],
    resolution=(16, 16),
    settings=rw.RenderSettings(
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      shadows=rw.settings.Shadows(enabled=False),
      lighting=rw.settings.Lights(camera_light=False),
      reflections=rw.settings.Reflections(enabled=False),
      outputs=rw.Outputs.of("hdr"),
      performance=rw.settings.Performance(traversal=backend),
    ).replace(),
    per_world=rw.PerWorld(material=[*(["instance.material"] if vary else []), *fields]),
  )
  renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]]] * worlds, np.float32))
  return renderer


@pytest.mark.parametrize("effect", ["vertex_color", "vertex_alpha", "curve", "coat", "mirror", "transmission"])
def test_prepared_unused_shading_matches_fresh_selection(effect: Effect):
  dynamic = _swap_renderer(_swap_scene(effect, selected=0), worlds=2, vary=True)
  fresh = _swap_renderer(_swap_scene(effect, selected=1), worlds=1, vary=False)
  try:
    dynamic.render(seed=17)
    before = required(dynamic.outputs.hdr).numpy().copy()
    table = dynamic.per_world["instance.material"]
    values = table.numpy()
    values[1, 0] = 1
    table.assign(values)
    dynamic.reset_history(wp.array([False, True], dtype=bool, device="cpu"))
    dynamic.render(seed=17)
    fresh.render(seed=17)
    actual = required(dynamic.outputs.hdr).numpy()
    expected = required(fresh.outputs.hdr).numpy()
    np.testing.assert_array_equal(actual[0], before[0])
    # The target genuinely changes appearance; parity cannot pass with an inert fixture.
    assert float(np.max(np.abs(expected[0] - before[0]))) > 1e-3
    np.testing.assert_allclose(actual[1], expected[0], rtol=1e-5, atol=1e-6)
  finally:
    dynamic.close()
    fresh.close()


@pytest.mark.parametrize("effect", ["vertex_color", "curve"])
def test_selected_material_composes_live_coat_fields(effect: Effect):
  dynamic = _swap_renderer(_swap_scene(effect, selected=0), worlds=2, vary=True, fields=("coat_weight",))
  fresh = _swap_renderer(_swap_scene(effect, selected=1, coat=0.4), worlds=1, vary=False)
  try:
    dynamic.per_world["instance.material"].assign(np.array([[0], [1]], np.int32))
    dynamic.per_world["material.coat_weight"].assign(np.array([[0, 0], [0, 0.4]], np.float32))
    dynamic.render(seed=17)
    fresh.render(seed=17)
    np.testing.assert_allclose(
      required(dynamic.outputs.hdr).numpy()[1], required(fresh.outputs.hdr).numpy()[0], rtol=1e-5, atol=1e-6
    )
    # A material-id rebind must invalidate the recorded gather's pointer, including no-material.
    dynamic.bind_per_world("instance.material", wp.array([[0], [-1]], dtype=int, device="cpu"))
    dynamic.render(seed=17)
    assert dynamic.debug.frame().instance_shading.numpy()[1, 0] == 0
  finally:
    dynamic.close()
    fresh.close()


def test_explicit_shading_override_wins_over_selected_material():
  renderer = _swap_renderer(_swap_scene("curve", selected=0), worlds=2, vary=True, fields=("instance.shading",))
  try:
    renderer.render(seed=17)
    before = required(renderer.outputs.hdr).numpy().copy()
    renderer.per_world["instance.material"].assign(np.array([[0], [1]], np.int32))
    whole_table(renderer.per_world["instance.shading"]).zero_()
    renderer.render(seed=17)
    np.testing.assert_array_equal(required(renderer.outputs.hdr).numpy(), before)
    np.testing.assert_array_equal(renderer.debug.frame().instance_shading.numpy(), 0)
  finally:
    renderer.close()


@pytest.mark.parametrize("effect", ["mirror", "vertex_color"])
def test_material_selection_without_used_shading_flags_has_no_derived_storage(effect: Effect):
  renderer = _swap_renderer(_swap_scene(effect, selected=0, colors=False), worlds=2, vary=True)
  try:
    assert not renderer._per_world.derived.slots
    assert renderer.debug.frame().instance_shading.shape[0] == 1
  finally:
    renderer.close()


@pytest.mark.parametrize("backend", ["software", "optix"])
@pytest.mark.parametrize("effect", ["vertex_color", "coat"])
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph material shading")
def test_material_shading_selection_replays_without_reprepare(effect: Effect, backend: Literal["software", "optix"]):
  from raywarp.core.pipeline.platform.device import device_caps

  if backend == "optix" and not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  dynamic = _swap_renderer(_swap_scene(effect, selected=0), worlds=2, vary=True, device="cuda:0", backend=backend)
  fresh = _swap_renderer(_swap_scene(effect, selected=1), worlds=1, vary=False, device="cuda:0", backend=backend)
  try:
    dynamic.render(seed=17)
    before = required(dynamic.outputs.hdr).numpy().copy()
    table = whole_table(dynamic.per_world["instance.material"])
    pointer = table.ptr
    graph = dynamic.capture(seed=17)
    table.assign(np.array([[0], [1]], np.int32))
    graph.launch()
    fresh.render(seed=17)
    actual = required(dynamic.outputs.hdr).numpy()
    np.testing.assert_array_equal(actual[0], before[0])
    np.testing.assert_allclose(actual[1], required(fresh.outputs.hdr).numpy()[0], rtol=1e-5, atol=1e-6)
    assert table.ptr == pointer
    table.zero_()
    graph.launch()
    np.testing.assert_array_equal(required(dynamic.outputs.hdr).numpy(), before)
  finally:
    dynamic.close()
    fresh.close()
