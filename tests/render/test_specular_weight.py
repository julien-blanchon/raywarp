"""Canonical specular weight coefficients; renderer-path acceptance is separate."""

import dataclasses
from typing import Literal

import numpy as np
import pytest
import warp as wp
from conftest import DOWN, required, whole_table

import raywarp as rw
from raywarp.core.shade.materials import brdf
from raywarp.scene.model.types import MaterialFlags


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_metal_weight_scales_whole_compensated_lobe(weight: float) -> None:
  base = wp.vec3(0.12, 0.45, 0.8)
  dfg = wp.vec2(0.06, 0.7)
  compensation = brdf.energy_compensation(base, dfg)
  f0, f90 = brdf.weighted_fresnel(base, 1.0, 0.5 * weight**0.5, weight)
  for cosine in (0.0, 0.05, 0.5, 1.0):
    original = brdf.f_schlick_rgb(base, wp.vec3(1.0), cosine)
    actual = wp.cw_mul(brdf.f_schlick_rgb(f0, f90, cosine), compensation)
    expected = np.asarray(wp.cw_mul(original, compensation)) * weight
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=1e-7)
  weighted_albedo = brdf.specular_albedo_rgb(f0, f90, dfg)
  np.testing.assert_allclose(np.asarray(weighted_albedo), np.asarray(brdf.specular_albedo(base, dfg)) * weight, atol=1e-7)


@pytest.mark.parametrize("metallic", [0.0, 0.4, 1.0])
def test_zero_weight_has_no_specular_at_any_angle(metallic: float) -> None:
  f0, f90 = brdf.weighted_fresnel(wp.vec3(0.2, 0.4, 0.7), metallic, 0.0, 0.0)
  for cosine in (0.0, 0.01, 0.5, 1.0):
    assert brdf.f_schlick_rgb(f0, f90, cosine) == wp.vec3(0.0)
  assert brdf.specular_albedo_rgb(f0, f90, wp.vec2(0.1, 0.8)) == wp.vec3(0.0)


def test_index_matched_dielectric_does_not_disable_metal() -> None:
  base = wp.vec3(0.2, 0.4, 0.7)
  dielectric = brdf.weighted_fresnel(base, 0.0, 0.0, 1.0)
  metal = brdf.weighted_fresnel(base, 1.0, 0.0, 1.0)
  assert dielectric == (wp.vec3(0.0), wp.vec3(0.0))
  assert metal == (base, wp.vec3(1.0))


@pytest.mark.parametrize("metallic", [0.0, 0.4, 1.0])
def test_unit_weight_preserves_existing_coefficients(metallic: float) -> None:
  base = wp.vec3(0.2, 0.4, 0.7)
  f0, f90 = brdf.weighted_fresnel(base, metallic, 0.5, 1.0)
  np.testing.assert_array_equal(np.asarray(f0), np.asarray(brdf.surface_f0(base, metallic, 0.5)))
  assert f90 == wp.vec3(1.0)


# ---------------------------------------------------------------------------------------------
# Compiled weight paths, live canonical rows and unchanged custom-curve semantics.


def _scene(
  *,
  weight: float,
  metallic: float,
  curve: bool = False,
  lighting: Literal["sun", "area", "environment"] = "sun",
  native: bool = False,
  mixed: bool = False,
) -> rw.Scene:
  b = rw.SceneBuilder()
  m = b.add_material(
    rw.Material(
      base_color=(0.15, 0.4, 0.7),
      base_metalness=metallic,
      specular_weight=weight,
      specular_roughness=0.35,
      ext=rw.scene.RaywarpExt(
        flags=MaterialFlags.SPECULAR_CURVE if curve else MaterialFlags.DIELECTRIC_SPECULAR_LEVEL if native else MaterialFlags(0)
      ),
    )
  )
  b.add_instance("box", size=(2.0, 2.0, 0.1), material=m)
  if mixed:
    canonical = b.add_material(rw.Material(specular_weight=0.0))
    b.add_instance("box", size=(0.1, 0.1, 0.1), material=canonical, xform=np.c_[np.eye(3), [20.0, 0.0, 0.0]])
  if lighting == "sun":
    b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0))
  elif lighting == "area":
    b.add_light(
      rw.Light(kind=rw.scene.LightKind.RECT, intensity=10.0, width=1.0, height=1.0, xform=np.c_[np.eye(3), [0.0, 0.0, 3.0]])
    )
  scene = b.build()
  if lighting == "environment":
    scene = scene.with_lighting(rw.Lighting(envmap=np.ones((8, 16, 3), np.float32)))
  return scene


def _renderer(
  weight: float,
  metallic: float,
  *,
  declared: bool,
  curve: bool = False,
  lighting: Literal["sun", "area", "environment"] = "sun",
  device: str = "cpu",
  backend: Literal["software", "optix"] = "software",
  motion: bool = False,
  native: bool = False,
  mixed: bool = False,
  flex_material: bool = False,
) -> rw.Renderer[None]:
  scene = rw.upload(
    _scene(weight=weight, metallic=metallic, curve=curve, lighting=lighting, native=native, mixed=mixed), device
  )
  if flex_material:
    from raywarp.core.pipeline.gpu_scene.deformables import Deformables, Part
    from raywarp.scene.model.types import MaterialId

    parts = [
      Part(
        dim=2,
        vertadr=0,
        vertnum=3,
        radius=0.0,
        rgba=(1.0, 1.0, 1.0, 1.0),
        material=MaterialId(0),
        faces=np.array([[0, 1, 2]], np.int32),
      )
    ]
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    scene = dataclasses.replace(scene, deformables=Deformables.create(parts, vertices, device=wp.get_device(device)))
  r = rw.Renderer(
    scene,
    nworld=2,
    cameras=[rw.CameraSpec(fovy=15.0)],
    resolution=(4, 4),
    settings=rw.RenderSettings(
      outputs=rw.Outputs.of("hdr", "motion") if motion else rw.Outputs.of("hdr"),
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=lighting == "environment"),
      lighting=rw.settings.Lights(camera_light=False),
      shadows=rw.settings.Shadows(enabled=False),
      performance=rw.settings.Performance(traversal=backend),
    ),
    per_world=rw.PerWorld(material=["specular_weight"]) if declared else None,
  )
  r.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, *DOWN]]] * 2, np.float32))
  return r


@pytest.mark.parametrize("lighting", ["sun", "area", "environment"])
def test_compiled_metal_live_weight_is_world_local_and_linear(lighting: Literal["sun", "area", "environment"]) -> None:
  r = _renderer(1.0, 1.0, declared=True, lighting=lighting)
  try:
    assert r.features.specular_weight
    r.render()
    full = required(r.outputs.hdr).numpy().copy()
    assert full.min() > 0.0
    weight = whole_table(r.per_world["material.specular_weight"])
    weight.assign(np.array([[1.0], [0.0]], np.float32))
    r.render()
    zero = required(r.outputs.hdr).numpy().copy()
    np.testing.assert_array_equal(zero[0], full[0])
    np.testing.assert_array_equal(zero[1], np.zeros_like(zero[1]))
    weight.assign(np.array([[1.0], [0.5]], np.float32))
    r.render()
    half = required(r.outputs.hdr).numpy()
    np.testing.assert_array_equal(half[0], full[0])
    np.testing.assert_allclose(half[1], full[1] * 0.5, rtol=2e-6)
  finally:
    r.close()


@pytest.mark.parametrize("curve", [False, True])
def test_declared_unit_weight_matches_static_compiled_response(curve: bool) -> None:
  static = _renderer(1.0, 0.4, declared=False, curve=curve)
  active = _renderer(1.0, 0.4, declared=True, curve=curve)
  try:
    static.render()
    active.render()
    np.testing.assert_array_equal(required(static.outputs.hdr).numpy(), required(active.outputs.hdr).numpy())
    if curve:
      assert static.features.specular_curve and active.features.specular_curve
    else:
      assert not static.features.specular_weight and active.features.specular_weight
  finally:
    static.close()
    active.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("backend", ["software", "optix"])
def test_captured_weight_update_and_selective_history_reset(backend: Literal["software", "optix"]) -> None:
  from raywarp.core.pipeline.platform.device import device_caps

  if backend == "optix" and not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX unavailable")
  r = _renderer(0.0, 1.0, declared=True, device="cuda:0", backend=backend, motion=True)
  mask = wp.array([False, True], dtype=bool, device="cuda:0")

  def reset() -> None:
    r.reset_history(mask)

  try:
    assert r.features.traversal == backend
    assert r.features.specular_weight and r.features.history
    weight = whole_table(r.per_world["material.specular_weight"])
    graph = r.capture(reset, seed=17, arrays=lambda: {"reset": mask})
    r.reset_history()
    weight.assign(np.array([[0.0], [0.5]], np.float32))
    graph.launch()
    half = required(r.outputs.hdr).numpy().copy()
    assert half[1].min() > 0.0
    np.testing.assert_array_equal(half[0], np.zeros_like(half[0]))
    weight.assign(np.array([[0.0], [1.0]], np.float32))
    graph.launch()
    full = required(r.outputs.hdr).numpy().copy()
    np.testing.assert_allclose(full[1], half[1] * 2.0, rtol=2e-6)
    np.testing.assert_array_equal(full[0], half[0])
    reset()
    r.render(seed=17)
    np.testing.assert_array_equal(required(r.outputs.hdr).numpy(), full)
    replacement = wp.array([[0.0], [0.25]], dtype=float, device="cuda:0")
    r.bind_per_world("material.specular_weight", replacement)
    with pytest.raises(rw.StaleBinding, match="changed since the capture"):
      graph.launch()
    # Poison the former input: a recapture must retain and read the replacement instead.
    weight.fill_(2.0)
    rebound = r.capture(reset, seed=17, arrays=lambda: {"reset": mask})
    r.reset_history()
    rebound.launch()
    quarter = required(r.outputs.hdr).numpy().copy()
    np.testing.assert_array_equal(quarter[0], full[0])
    np.testing.assert_allclose(quarter[1], full[1] * 0.25, rtol=2e-6)
    reset()
    r.render(seed=17)
    np.testing.assert_array_equal(required(r.outputs.hdr).numpy(), quarter)
  finally:
    r.close()


@pytest.mark.parametrize("curve", [False, True], ids=["dielectric-level", "custom-curve"])
def test_mixed_native_model_keeps_unscaled_metal(curve: bool) -> None:
  reference = _renderer(0.0, 1.0, declared=False, native=not curve, curve=curve)
  mixed = _renderer(0.0, 1.0, declared=False, native=not curve, curve=curve, mixed=True)
  try:
    assert not reference.features.specular_weight
    assert mixed.features.specular_weight
    reference.render()
    mixed.render()
    expected = required(reference.outputs.hdr).numpy()
    assert expected.min() > 0.0
    np.testing.assert_array_equal(required(mixed.outputs.hdr).numpy(), expected)
  finally:
    reference.close()
    mixed.close()


def test_active_carriers_add_only_weight_and_shared_models() -> None:
  from warp.types import type_size_in_bytes

  from raywarp.core.shade.materials.material import Materials, RoughDiffuseMaterials, RoughWeightedMaterials, WeightedMaterials

  # 1624 B of shading and alpha-test tables (an OptiX shaded build's carrier is its geometry's
  # alpha-test tables; the per-mesh addresses one `MeshAdr` table since 45045475) + the general
  # kernel's texture taps (aligned).
  assert type_size_in_bytes(Materials) == 1624 + 8
  assert type_size_in_bytes(WeightedMaterials) == type_size_in_bytes(Materials) + 112
  assert type_size_in_bytes(RoughWeightedMaterials) == type_size_in_bytes(RoughDiffuseMaterials) + 112


def _flex_response(r: rw.Renderer[None]) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.float32]]:
  """Flex has no instance shading row; its material model must still reach the same lobes."""
  from raywarp.core.kernels.abi import rays
  from raywarp.core.kernels.generic import concrete
  from raywarp.core.shade.materials import material
  from raywarp.core.shade.materials.flex import make_flex_surface
  from raywarp.core.trace import frame_structs

  try:
    # The hand-built kernel below is specialised on the scene's own Features and declares their
    # struct types (`WeightedMaterials`, the software geometry): the renderer must run that build,
    # not a tier's (a complete tier compiles the layered carrier: another struct, which crashed the
    # CPU launch when fed to this kernel).
    r.update_settings(**{"performance.kernels": "scene"})
    ctx = r.context()
    assert ctx.features == r.features, "the frame structs must be those of the kernel's Features"
    assert (
      material.material_type(diffuse_roughness=r.features.diffuse_roughness, specular_weight=True, layered=r.features.layered)
      is material.WeightedMaterials
    )
    features = dataclasses.replace(r.features, flex_materials=True, textures=False)
    flex = make_flex_surface(features)
    lobes_of = material.make_shading_lobes(
      specular_curve=features.specular_curve, clearcoat=False, specular_layers=False, specular_weight=True
    )

    @wp.kernel
    @concrete(geo=rays.software_geometry_type(live_uv=False), mats=material.WeightedMaterials)
    def sample(
      # Model:
      geo: rays.SoftwareFields,
      mats: material.WeightedMaterials,
      dfg: wp.Texture2D,
      # Out:
      values: wp.array1d[wp.vec3],
    ):
      hit = rays.Hit()
      hit.mesh_id = wp.tid()
      hit.normal = wp.vec3(0.0, 0.0, 1.0)
      surface, normal = flex(geo, mats, 0, hit, wp.vec3(0.0), wp.vec3(0.0, 0.0, -1.0), 0.0)
      lobes = lobes_of(dfg, surface, 0.25 * wp.dot(normal, wp.vec3(0.0, 0.0, 1.0)))
      values[wp.tid()] = wp.cw_mul(lobes.spec_albedo, lobes.energy_comp)
      values[wp.tid() + 2] = lobes.f90

    frame = r.debug.frame()
    geo = frame_structs.geometry(ctx, frame)
    parts = []
    for mat in (0, 1 if r.scene.nmaterial > 1 else -1):
      part = rays.FlexPart()
      part.rgba, part.material, part.uvadr = wp.vec4(1.0, 1.0, 1.0, 1.0), mat, -1
      parts.append(part)
    geo.flex_part = wp.array(parts, dtype=rays.FlexPart, device="cpu")
    values = wp.empty(4, dtype=wp.vec3, device="cpu")
    wp.launch(sample, dim=2, inputs=[geo, frame_structs.materials(ctx, frame), ctx.tables.dfg], outputs=[values], device="cpu")
    # Warp vector elements expose their component axis at this host boundary.
    return np.asarray(values.numpy(), dtype=np.float32).reshape(4, 3)
  finally:
    r.close()


def test_mixed_flex_material_uses_shared_native_model() -> None:
  r = _renderer(0.0, 1.0, declared=False, native=True, mixed=True)
  actual = _flex_response(r)
  assert actual[0].min() > 0.0
  np.testing.assert_array_equal(actual[1], np.zeros(3, np.float32))


def test_curve_flex_is_independent_of_other_canonical_materials() -> None:
  fresh = _renderer(0.0, 1.0, declared=False, curve=True, flex_material=True)
  mixed = _renderer(0.0, 1.0, declared=False, curve=True, mixed=True, flex_material=True)
  assert fresh.features.specular_weight and fresh.features.specular_curve
  assert mixed.features.specular_weight and mixed.features.specular_curve
  # The second synthetic hit has no material in the fresh scene; only compare the real part.
  original, other = _flex_response(fresh), _flex_response(mixed)
  np.testing.assert_array_equal(original[0], other[0])
  np.testing.assert_array_equal(original[2], np.array([0.15, 0.4, 0.7], np.float32))
  np.testing.assert_array_equal(other[2], original[2])


def test_zero_weight_dielectric_matches_lambert_without_grazing_loss() -> None:
  r = _renderer(0.0, 0.0, declared=False)
  try:
    r.render()
    actual = required(r.outputs.hdr).numpy()
    expected = np.array([0.15, 0.4, 0.7], np.float32) * (10.0 / np.pi)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=2e-6)
  finally:
    r.close()


def test_rebound_weight_reaches_the_new_frame_storage() -> None:
  r = _renderer(1.0, 1.0, declared=True)
  try:
    r.render()
    full = required(r.outputs.hdr).numpy().copy()
    replacement = wp.array([[1.0], [0.5]], dtype=float, device="cpu")
    r.bind_per_world("material.specular_weight", replacement)
    assert r.debug.frame().material_specular_weight is replacement
    r.render()
    actual = required(r.outputs.hdr).numpy()
    np.testing.assert_array_equal(actual[0], full[0])
    np.testing.assert_allclose(actual[1], full[1] * 0.5, rtol=2e-6)
  finally:
    r.close()
