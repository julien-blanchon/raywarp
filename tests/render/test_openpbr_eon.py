"""Local EON contracts; these do not claim renderer integration or image acceptance."""

import math

import numpy as np
import numpy.typing as npt
import pytest
import warp as wp
from conftest import DOWN, render_host, required, whole_table

import raywarp as rw
from raywarp.core.shade.environment import ibl
from raywarp.core.shade.materials.material import Materials, RoughDiffuseMaterials, Surface, make_shading_lobes, material_type
from raywarp.core.shade.materials.openpbr_lobes import eon_albedo, eon_eval_split, eon_split
from raywarp.settings.openpbr import OpenPBR


@wp.func
def eon_eval(
  # In:
  rho: wp.vec3,
  roughness: float,
  wi: wp.vec3,
  wo: wp.vec3,
) -> wp.vec3:
  """EON BRDF without the incident cosine in the +Z frame, composed as the renderer does
  (`eon_split` once per view, `eon_eval_split` per light); zero roughness is Lambert's."""
  if roughness == 0.0:
    if wi[2] <= 0.0 or wo[2] <= 0.0:
      return wp.vec3(0.0)
    return rho * wp.static(1.0 / wp.pi)
  single, multi = eon_split(rho, roughness, wo[2])
  return eon_eval_split(single, multi, roughness, wi[2], wo[2], wp.dot(wi, wo))


@wp.kernel
def _evaluate(
  # In:
  directions: wp.array1d[wp.vec3],
  view: wp.vec3,
  color: wp.vec3,
  roughness: float,
  # Out:
  values: wp.array1d[wp.vec3],
  reverse: wp.array1d[wp.vec3],
):
  i = wp.tid()
  values[i] = eon_eval(color, roughness, directions[i], view)
  reverse[i] = eon_eval(color, roughness, view, directions[i])


def _integrate(color: tuple[float, float, float], roughness: float, mu: float) -> npt.NDArray[np.float64]:
  """Independent hemisphere quadrature, not the analytic albedo expression."""
  nodes, weights = np.polynomial.legendre.leggauss(64)
  z = (nodes + 1.0) / 2.0
  phi = (np.arange(128) + 0.5) * (2.0 * np.pi / 128)
  radius = np.sqrt(1.0 - z * z)
  directions = np.stack(
    (radius[:, None] * np.cos(phi), radius[:, None] * np.sin(phi), np.broadcast_to(z[:, None], (64, 128))), axis=-1
  ).reshape(-1, 3)
  incoming = wp.array(directions, dtype=wp.vec3, device="cpu")
  values = wp.empty(len(directions), dtype=wp.vec3, device="cpu")
  reverse = wp.empty_like(values)
  wp.launch(
    _evaluate,
    dim=len(directions),
    inputs=[incoming, wp.vec3(math.sqrt(1.0 - mu * mu), 0.0, mu), wp.vec3(*color), roughness],
    outputs=[values, reverse],
    device="cpu",
  )
  actual = values.numpy()
  np.testing.assert_allclose(actual, reverse.numpy(), rtol=2e-6, atol=2e-7)
  assert np.all(np.isfinite(actual)) and np.all(actual >= 0.0)
  return np.sum(actual.reshape(64, 128, 3) * (weights * z)[:, None, None], axis=(0, 1)) * (np.pi / 128)


@pytest.mark.parametrize("roughness,mu", [(0.0, 1.0), (1e-7, 0.01), (0.4, 0.3), (1.0, 0.02), (1.0, 1.0)])
def test_eon_white_furnace_and_reciprocity(roughness: float, mu: float):
  np.testing.assert_allclose(_integrate((1.0, 1.0, 1.0), roughness, mu), 1.0, atol=3e-4)


@pytest.mark.parametrize("mu", [0.02, 0.3, 1.0])
def test_colored_directional_albedo_matches_quadrature(mu: float):
  color = (0.05, 0.4, 0.9)
  integrated = _integrate(color, 0.8, mu)
  analytic = eon_albedo(wp.vec3(*color), 0.8, mu)
  np.testing.assert_allclose(integrated, np.asarray(analytic), atol=2e-4)
  assert np.all(integrated <= np.array(color) + 1e-5)


def test_eon_zero_roughness_and_dark_side():
  color = wp.vec3(0.1, 0.4, 0.9)
  view = wp.vec3(0.0, 0.0, 1.0)
  np.testing.assert_array_equal(np.asarray(eon_eval(color, 0.0, view, view)), np.asarray(color * (1.0 / wp.pi)))
  assert eon_eval(color, 1.0, -view, view) == wp.vec3(0.0)
  assert np.all(np.isfinite(np.asarray(eon_albedo(color, 1.0, 0.0))))


# ---------------------------------------------------------------------------------------------
# Focused end-to-end EON activation, weighting and inactive-carrier checks.


def _scene(roughness: float, weight: float) -> rw.Scene:
  builder = rw.SceneBuilder()
  material = builder.add_material(
    rw.Material(
      base_color=(0.1, 0.4, 0.8),
      base_weight=weight,
      specular_weight=0.0,
      extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=roughness),
    )
  )
  builder.add_instance("box", size=(2.0, 2.0, 0.1), material=material)
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0))
  return builder.build()


def _render(roughness: float, weight: float, mode: OpenPBR) -> rw.Renderer[None]:
  return render_host(
    _scene(roughness, weight),
    (0.0, 0.0, 2.0),
    fovy=0.1,
    res=2,
    settings={"openpbr": mode, "shadows.enabled": False, "lighting.camera_light": False},
  )


def test_inactive_carrier_and_forced_neutral_pixels():
  assert material_type(diffuse_roughness=False) is Materials
  assert material_type(diffuse_roughness=True) is RoughDiffuseMaterials
  off = _render(0.0, 1.0, OpenPBR(diffuse_roughness=False))
  forced = _render(0.0, 1.0, OpenPBR(diffuse_roughness=True))
  assert off.context().tables.diffuse_source is None
  assert forced.features.diffuse_roughness
  np.testing.assert_array_equal(required(off.outputs.hdr).numpy(), required(forced.outputs.hdr).numpy())


def test_diffuse_layer_weight_is_outside_nonlinear_eon():
  full = _render(1.0, 1.0, OpenPBR())
  half = _render(1.0, 0.5, OpenPBR())
  assert full.features.diffuse_roughness and half.features.diffuse_roughness
  np.testing.assert_allclose(
    required(half.outputs.hdr).numpy()[..., :3], required(full.outputs.hdr).numpy()[..., :3] * 0.5, rtol=2e-5, atol=1e-6
  )


def test_declared_zero_to_rough_change_is_world_local():
  settings = rw.RenderSettings(
    outputs=rw.Outputs(hdr=rw.settings.FloatOutput()),
    gi=rw.settings.GI(enabled=False),
    environment=rw.settings.Environment(enabled=False),
    performance=rw.settings.Performance(traversal="software"),
    lighting=rw.settings.Lights(camera_light=False),
    shadows=rw.settings.Shadows(enabled=False),
  )
  renderer = rw.Renderer(
    rw.upload(_scene(0.0, 1.0), "cpu"),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=0.1)],
    resolution=(2, 2),
    settings=settings,
    per_world=rw.PerWorld(material=["base_diffuse_roughness"]),
  )
  renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, *DOWN]]] * 2, np.float32))
  renderer.render()
  before = required(renderer.outputs.hdr).numpy().copy()
  values = renderer.per_world["material.base_diffuse_roughness"]
  values.assign(np.array([[0.0], [1.0]], np.float32))
  renderer.render()
  after = required(renderer.outputs.hdr).numpy()
  np.testing.assert_array_equal(after[0], before[0])
  assert np.max(np.abs(after[1, ..., :3] - before[1, ..., :3])) > 0.05


def test_mixed_substrate_omission_is_explicit():
  builder = rw.SceneBuilder()
  material = builder.add_material(
    rw.Material(
      extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=1.0, subsurface_weight=0.5),
    )
  )
  builder.add_instance("box", size=(1.0, 1.0, 0.1), material=material)
  uploaded = rw.upload(builder.build(), "cpu")
  assert uploaded.report.dropped["gpu.material.base_diffuse_roughness(subsurface/transmission fallback)"] == 1
  assert uploaded.derive_inputs.diffuse_source[0, 1] == 0.0


def test_declared_roughness_reports_initially_smooth_mixed_substrate():
  builder = rw.SceneBuilder()
  material = builder.add_material(rw.Material(extras=rw.scene.OpenPbrExtras(subsurface_weight=0.5)))
  builder.add_instance("box", size=(2.0, 2.0, 0.1), material=material)
  builder.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0))
  renderer = rw.Renderer(
    rw.upload(builder.build(), "cpu"),
    nworld=1,
    cameras=[rw.CameraSpec(fovy=0.1)],
    resolution=(2, 2),
    settings=rw.RenderSettings(
      outputs=rw.Outputs(hdr=rw.settings.FloatOutput()),
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      performance=rw.settings.Performance(traversal="software"),
      lighting=rw.settings.Lights(camera_light=False),
      shadows=rw.settings.Shadows(enabled=False),
    ),
    per_world=rw.PerWorld(material=["base_diffuse_roughness"]),
  )
  key = "gpu.material.base_diffuse_roughness(declared on subsurface/transmission fallback)"
  assert renderer.scene.report.dropped[key] == 1
  renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, *DOWN]]], np.float32))
  renderer.render()
  before = required(renderer.outputs.hdr).numpy().copy()
  whole_table(renderer.per_world["material.base_diffuse_roughness"]).fill_(1.0)
  renderer.render()
  np.testing.assert_array_equal(required(renderer.outputs.hdr).numpy(), before)
  assert renderer.scene.report.dropped[key] == 1
  renderer.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph replay")
@pytest.mark.parametrize("backend", ["software", "optix"])
def test_live_roughness_replays_without_rebinding(backend):
  from raywarp.core.pipeline.platform.device import device_caps

  if backend == "optix" and not device_caps("cuda:0").optix.available:
    pytest.skip("OptiX backend unavailable")
  renderer = rw.Renderer(
    rw.upload(_scene(0.0, 1.0), "cuda:0"),
    nworld=2,
    cameras=[rw.CameraSpec(fovy=0.1)],
    resolution=(8, 8),
    settings=rw.RenderSettings(
      outputs=rw.Outputs(hdr=rw.settings.FloatOutput()),
      gi=rw.settings.GI(enabled=False),
      environment=rw.settings.Environment(enabled=False),
      performance=rw.settings.Performance(traversal=backend),
      lighting=rw.settings.Lights(camera_light=False),
      shadows=rw.settings.Shadows(enabled=False),
    ),
    per_world=rw.PerWorld(material=["base_diffuse_roughness"]),
  )
  try:
    renderer.camera_pose.assign(np.array([[[0.0, 0.0, 2.0, *DOWN]]] * 2, np.float32))
    captured = renderer.capture(seed=31)
    before = required(renderer.outputs.hdr).numpy().copy()
    values = whole_table(renderer.per_world["material.base_diffuse_roughness"])
    pointer = values.ptr
    values.assign(np.array([[0.0], [1.0]], np.float32))
    captured.launch()
    replay = required(renderer.outputs.hdr).numpy().copy()
    assert values.ptr == pointer
    np.testing.assert_array_equal(replay[0], before[0])
    assert np.max(np.abs(replay[1, ..., :3] - before[1, ..., :3])) > 0.05
    renderer.render(seed=31)
    np.testing.assert_array_equal(required(renderer.outputs.hdr).numpy(), replay)
  finally:
    renderer.close()


_diffuse_lobes = make_shading_lobes(specular_curve=False, clearcoat=False, specular_layers=True, diffuse_roughness=True)


@wp.kernel
def _no_diffuse_substrate(
  # Model:
  dfg: wp.Texture2D,
  # In:
  metallic: float,
  specular_only: bool,
  # Out:
  result: wp.array1d[wp.vec4],
):
  surface = Surface()
  surface.base_color = wp.vec3(0.5)
  surface.perceptual_roughness = 0.5
  surface.reflectance = 0.5
  surface.metallic = metallic
  surface.specular_only = specular_only
  surface.alpha = 0.0  # opacity 1 - t of a fully transmissive layer
  surface.diffuse_roughness = 1.0
  surface.diffuse_base_weight = 1.0
  lobes = _diffuse_lobes(dfg, surface, 0.5)
  w = lobes.diffuse_single + lobes.diffuse_multi
  result[0] = wp.vec4(w[0], w[1], w[2], lobes.diffuse_roughness)


@pytest.mark.parametrize(("metallic", "specular_only", "roughness"), [(1.0, False, 0.0), (0.0, True, 1.0)])
def test_no_diffuse_substrate_has_no_eon_lobe(metallic, specular_only, roughness):
  # A metal has no EON lobe; a fully transmissive layer (opacity 0) keeps its lobe weighted by its
  # opacity 1 - t (OpenPBR's diffuse / transmission mix): zero weight.
  device = wp.get_device("cpu")
  result = wp.empty(1, dtype=wp.vec4, device=device)
  wp.launch(
    _no_diffuse_substrate,
    dim=1,
    inputs=[ibl.dfg_texture(device), metallic, specular_only],
    outputs=[result],
    device=device,
  )
  np.testing.assert_array_equal(result.numpy(), np.array([[0.0, 0.0, 0.0, roughness]], np.float32))
