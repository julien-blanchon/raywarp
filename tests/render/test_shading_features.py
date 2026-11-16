"""Physically-based shading features on small SceneBuilder scenes (CPU), against closed forms:
tinted translucent shadows, sun discs, tube lights in point form, spot cone softness, clearcoat,
and GI probe placement through alpha cutouts. Each feature is compiled in only when the scene
uses it (Features fields off otherwise)."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping

import numpy as np
import pytest
import warp as wp
from conftest import DOWN, disc_visible_fraction, recipe, render_host, required

import raywarp as rw
import raywarp.formats.usd
from raywarp._base import types as core_types
from raywarp.core.shade.materials import brdf
from raywarp.scene.model.types import AlphaMode, Slot, TextureRef

DEVICE = "cpu"


def _xform(pos=(0.0, 0.0, 0.0), rot=None) -> np.ndarray:
  return np.c_[np.eye(3) if rot is None else rot, np.asarray(pos, np.float64)]


def _pixel(
  host, eye, settings: Mapping[str, object] | None = None, quat=DOWN, fovy=1.0, res=2
) -> tuple[np.ndarray, rw.Renderer]:
  """(mean HDR, Renderer) of a narrow camera at `eye`."""
  r = render_host(host, eye, quat, fovy=fovy, res=res, settings=settings)
  return required(r.outputs.hdr).numpy()[0, 0, ..., :3].reshape(-1, 3).mean(0).astype(np.float64), r


def _floor(b: rw.SceneBuilder, albedo: float = 0.8) -> None:
  m = b.add_material(rw.Material(base_color=(albedo, albedo, albedo), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(20.0, 20.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))


def _pane_scene(color, opacity: float) -> rw.Scene:
  """A floor under a 2 x 2 m translucent pane (a thin closed box) 1 m up; a sun overhead."""
  b = rw.SceneBuilder()
  _floor(b)
  pane = b.add_material(
    rw.Material(
      base_color=color,
      base_metalness=0.0,
      specular_roughness=0.5,
      geometry_opacity=opacity,
      ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.BLEND),
    )
  )
  b.add_instance("box", size=(1.0, 1.0, 0.005), material=pane, xform=_xform((0.0, 0.0, 1.0)))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=_xform()))
  return b.build()


def test_colored_glass_tints_the_sun_shadow():
  """Under a red pane of opacity 0.5 the floor gets the sun times the pane's RGB transmittance,
  base color x (1 - opacity), once (B12: not once per face); a grey pane keeps the scalar path
  (no tinted build)."""
  red = (0.9, 0.2, 0.1)
  host = _pane_scene(red, 0.5)
  # The camera sits between the pane and the floor, looking down at the shadowed floor.
  under, r = _pixel(
    host,
    (0.0, 0.0, 0.5),
    {"shadows": rw.settings.Shadows(translucent="transmittance"), "transparency": rw.settings.Transparency(enabled=True)},
  )
  assert r.features.translucent_shadows and r.features.tinted_shadows
  open_, _ = _pixel(
    host,
    (3.0, 0.0, 0.5),
    {"shadows": rw.settings.Shadows(translucent="transmittance"), "transparency": rw.settings.Transparency(enabled=True)},
  )
  want = np.asarray(red) * 0.5
  np.testing.assert_allclose(under / open_, want, rtol=0.03, atol=1e-4)
  # "clear" lets the sun through untinted; a grey pane compiles no RGB visibility.
  clear, r = _pixel(
    host,
    (0.0, 0.0, 0.5),
    {"shadows": rw.settings.Shadows(translucent="clear"), "transparency": rw.settings.Transparency(enabled=True)},
  )
  assert not r.features.tinted_shadows
  np.testing.assert_allclose(clear, open_, rtol=0.02)
  grey, r = _pixel(
    _pane_scene((1.0, 1.0, 1.0), 0.2), (0.0, 0.0, 0.5), {"shadows": rw.settings.Shadows(translucent="transmittance")}
  )
  assert r.features.translucent_shadows and not r.features.tinted_shadows
  np.testing.assert_allclose(grey / open_, [0.8] * 3, rtol=0.03)


@pytest.mark.parametrize("color", [(1.0, 1.0, 1.0), (0.9, 0.2, 0.1)])
def test_recorded_shadow_averages_scalar_or_rgb_visibility(color: tuple[float, float, float]) -> None:
  """The requested scalar shadow output agrees with either visibility specialization."""
  _, renderer = _pixel(
    _pane_scene(color, 0.5),
    (0.0, 0.0, 0.5),
    {
      "shadows": rw.settings.Shadows(translucent="transmittance"),
      "transparency": rw.settings.Transparency(enabled=True),
      "outputs": rw.Outputs.of("hdr", "shadow"),
    },
  )
  assert renderer.features.tinted_shadows == (color != (1.0, 1.0, 1.0))
  np.testing.assert_allclose(required(renderer.outputs.shadow).numpy(), sum(color) / 6.0, rtol=0.03, atol=1e-4)


def test_host_shadow_transmittance_is_rgb():
  gpu = rw.upload(_pane_scene((0.9, 0.2, 0.1), 0.5), device=DEVICE)
  t = np.asarray(gpu.instance_shadow_transmit)
  assert t.shape == (2, 3)
  np.testing.assert_allclose(t[0], 0.0)
  np.testing.assert_allclose(t[1], [0.45, 0.1, 0.05], rtol=1e-5)


def _slab_scene(angle_deg: float, height: float = 2.0) -> rw.Scene:
  """A floor under an opaque slab (x < 0) `height` up; a sun overhead of diameter `angle_deg`."""
  b = rw.SceneBuilder()
  _floor(b)
  m = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(5.0, 5.0, 0.01), material=m, xform=_xform((-5.0, 0.0, height)))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=angle_deg, xform=_xform()))
  return b.build()


def test_sun_disc_penumbra_matches_the_disc():
  """A 4 degree sun 2 m above the floor: the penumbra spans 2 h tan(2 deg); the visible fraction
  of a uniform disc whose centre is d from the edge (in units of the disc's radius) is the
  circular-segment area. The fixed pattern converges to it."""
  h, half = 2.0, math.radians(2.0)
  host = _slab_scene(4.0, h)
  gpu = rw.upload(host, device=DEVICE)
  assert gpu.capabilities.sun_disc
  np.testing.assert_allclose(gpu.tables["light.radius"].numpy()[0, 0], half, rtol=1e-6)
  lit, _ = _pixel(host, (5.0, 0.0, 0.5))

  settings = {"shadows": rw.settings.Shadows(samples=64)}
  for t in (-0.5, 0.0, 0.5):
    x = t * h * math.tan(half)  # the floor point whose disc centre is t radii past the edge
    v, r = _pixel(host, (x, 0.0, 0.5), settings, fovy=0.05)
    assert r.features.sun_disc
    assert v.mean() / lit.mean() == pytest.approx(disc_visible_fraction(t), abs=0.06), t


def test_sun_disc_off_without_angle_or_with_one_fixed_sample():
  """No angle: no disc code; one fixed sample: the hard shadow."""
  _, r = _pixel(_slab_scene(0.0), (1.0, 0.0, 0.5), {"shadows": rw.settings.Shadows(samples=4)})
  assert not r.features.sun_disc
  edge, r = _pixel(_slab_scene(4.0), (0.02, 0.0, 0.5), fovy=0.05)
  assert not r.features.sun_disc  # the default: fixed, one sample
  lit, _ = _pixel(_slab_scene(4.0), (5.0, 0.0, 0.5))
  np.testing.assert_allclose(edge, lit, rtol=1e-3)  # just outside the centre's shadow: fully lit


def test_usd_distant_angle_is_the_disc_diameter(tmp_path):
  stage = tmp_path / "sun.usda"
  stage.write_text(
    '#usda 1.0\n(\n  upAxis = "Z"\n)\ndef Cube "Box"\n{\n}\ndef DistantLight "Sun"\n{\n'
    "  float inputs:angle = 0.53\n  float inputs:intensity = 1000\n}\n"
  )
  host = rw.formats.usd.load(str(stage))
  assert "gpu.light.distant_angle" not in host.report.dropped
  gpu = rw.upload(host, device=DEVICE)
  np.testing.assert_allclose(gpu.tables["light.radius"].numpy()[0, 0], math.radians(0.53) / 2, rtol=1e-5)
  assert gpu.capabilities.sun_disc


def test_tube_light_point_form_falls_off_with_the_projected_area():
  """A cylinder light along x, 1 m up: its point form's intensity is I0 sin(theta) from the axis,
  so a floor point 2 m along the axis gets h / r of a point 2 m across it (equal distance)."""
  b = rw.SceneBuilder()
  _floor(b)
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.CYLINDER,
      intensity=1.0e5,
      radius=0.01,
      length=0.2,
      shadow_enable=False,
      xform=_xform((0.0, 0.0, 1.0)),
    )
  )
  host = b.build()
  gpu = rw.upload(host, device=DEVICE)
  assert gpu.tables["light.type"].numpy()[0, 0] == core_types.LightType.TUBE
  np.testing.assert_allclose(gpu.tables["light.dir"].numpy()[0, 0], [1.0, 0.0, 0.0])
  along, r = _pixel(host, (2.0, 0.0, 0.5), {"lighting": rw.settings.Lights(area_lights="point")})
  assert r.features.has_spot and not r.features.area_lights
  across, _ = _pixel(host, (0.0, 2.0, 0.5), {"lighting": rw.settings.Lights(area_lights="point")})
  below, _ = _pixel(host, (0.0, 0.0, 0.5), {"lighting": rw.settings.Lights(area_lights="point")})
  dist = math.hypot(2.0, 1.0)
  assert along.mean() / across.mean() == pytest.approx(1.0 / dist, rel=0.02)
  # Straight below: E = I0 / h^2, I0 = radiance x 2 r l; Lambert 0.8.
  np.testing.assert_allclose(below, 1.0e5 * 2 * 0.01 * 0.2 * 0.8 / math.pi, rtol=0.02)


def _spot_scene(cone_deg: float, softness: float) -> rw.Scene:
  b = rw.SceneBuilder()
  _floor(b)
  b.add_light(
    rw.Light(
      kind=rw.scene.LightKind.SPHERE,
      intensity=50.0,
      radius=0.0,
      treat_as_point=True,
      shaping=rw.scene.Shaping(cone_angle=cone_deg, cone_softness=softness),
      shadow_enable=False,
      xform=_xform((0.0, 0.0, 1.0)),
    )
  )
  return b.build()


@pytest.mark.parametrize("softness", [0.0, 0.5, 1.0])
def test_spot_cone_softness(softness):
  """B54: UsdLux ShapingAPI cone softness s: full intensity inside (1 - s) x cone, then
  1 - smoothstep of the off-axis angle out to the cone, nothing beyond (it was glTF's
  ((cos theta - cos inner) / (cos inner - cos outer))^2: 0.03-0.16 off)."""
  host = _spot_scene(30.0, softness)
  outer = math.radians(30.0)
  inner = outer * (1.0 - softness)

  def expected(theta: float) -> float:
    t = min(max((theta - inner) / max(outer - inner, 1.0e-4), 0.0), 1.0)
    cone = 0.0 if theta > outer else 1.0 - t * t * (3.0 - 2.0 * t)
    # Floor 1 m below: E = I cos^3(theta) / h^2, Lambert 0.8.
    return 50.0 * cone * math.cos(theta) ** 3 * 0.8 / math.pi

  for deg in (0.0, 10.0, 18.0, 20.0, 24.0, 27.0, 32.0):
    theta = math.radians(deg)
    v, r = _pixel(host, (math.tan(theta), 0.0, 0.5), fovy=0.05)
    assert r.features.has_spot
    assert v.mean() == pytest.approx(expected(theta), rel=0.03, abs=1e-3), deg


def test_default_inner_cone_keeps_the_mujoco_fraction():
  """softness 0.25 is the core's fixed inner cone (0.75): no per-light table."""
  gpu = rw.upload(_spot_scene(30.0, 0.25), device=DEVICE)
  assert "light.cone_inner" not in gpu.tables
  _, r = _pixel(_spot_scene(30.0, 0.25), (0.0, 0.0, 0.5))
  assert r.features.has_spot


def _coat_scene(coat: float, roughness: float, light: bool = True, sky: float = 0.0, color=(0.6, 0.05, 0.05)) -> rw.Scene:
  b = rw.SceneBuilder()
  m = b.add_material(
    rw.Material(
      base_color=color,
      base_metalness=0.0,
      specular_roughness=0.7,
      coat_weight=coat,
      coat_roughness=roughness,
      coat_ior=1.5,
      ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.COAT_GLTF),
    )
  )
  # 6 m: the mirror point of the light and the sky-lit centre (the sky-occlusion grid, the CPU
  # cost of the sky tests, grows with the floor's area).
  b.add_instance("box", size=(3.0, 3.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))
  if light:
    # At the mirror image of the camera (-1, 0, 1) about the floor's normal.
    b.add_light(
      rw.Light(
        kind=rw.scene.LightKind.SPHERE,
        intensity=500.0,
        radius=0.0,
        treat_as_point=True,
        shadow_enable=False,
        xform=_xform((1.0, 0.0, 1.0)),
      )
    )
  if sky > 0.0:
    b.add_environment(rw.scene.EnvironmentLight(intensity=sky))
  return b.build()


def _look_at_origin_quat() -> tuple:
  """Camera at (-1, 0, 1) looking at the origin: -z turned onto (1, 0, -1) / sqrt 2 about +y."""
  a = math.radians(-45.0) / 2.0
  return (0.0, math.sin(a), 0.0, math.cos(a))


@pytest.mark.parametrize("roughness", [0.1, 0.4])
def test_clearcoat_direct_lobe(roughness):
  """At the mirror configuration (camera and light at 45 degrees) the coated surface's radiance
  is the uncoated one x (1 - c F) plus c F D V NoL I / d^2, F the coat's Schlick Fresnel (F0 0.04)
  at LoH = cos 45 deg, D / V its GGX and height-correlated Smith terms."""
  eye, quat = (-1.0, 0.0, 1.0), _look_at_origin_quat()
  base, r = _pixel(_coat_scene(0.0, 0.0), eye, quat=quat, fovy=0.02)
  assert not r.features.clearcoat
  coated, r = _pixel(_coat_scene(1.0, roughness), eye, quat=quat, fovy=0.02)
  assert r.features.clearcoat
  nov = nol = loh = math.sqrt(0.5)
  noh = 1.0
  alpha = max(round(roughness * 255) / 255, brdf.MIN_PERCEPTUAL_ROUGHNESS) ** 2
  a2 = alpha * alpha
  d = a2 / (math.pi * ((noh * a2 - noh) * noh + 1.0) ** 2)
  v = 0.5 / (nol * math.sqrt(nov * nov * (1 - a2) + a2) + nov * math.sqrt(nol * nol * (1 - a2) + a2))
  f = 0.04 + 0.96 * (1.0 - loh) ** 5
  e = 500.0 / 2.0  # 500 cd at sqrt(2) m (the range window is ~1 there)
  want = base * (1.0 - f) + f * d * v * nol * e
  np.testing.assert_allclose(coated, want, rtol=0.02)


def test_clearcoat_under_an_area_light():
  """A small rect light far away: its LTC shading (coat lobe through its own LTC) approaches the
  point form (the coat's GGX lobe at the light's direction) on a coated floor."""
  eye, quat = (-1.0, 0.0, 1.0), _look_at_origin_quat()
  out = {}
  for coat in (0.0, 1.0):
    b = rw.SceneBuilder()
    m = b.add_material(
      rw.Material(
        base_color=(0.6, 0.05, 0.05),
        base_metalness=0.0,
        specular_roughness=0.7,
        coat_weight=coat,
        coat_roughness=0.4,
        coat_ior=1.5,
        ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.COAT_GLTF),
      )
    )
    b.add_instance("box", size=(20.0, 20.0, 0.05), material=m, xform=_xform((0.0, 0.0, -0.05)))
    # A 4 cm square at the mirror position, facing the floor point (local -Z turned onto
    # (-1, 0, -1) / sqrt 2 by 45 degrees about +y).
    c = s = math.sqrt(0.5)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    b.add_light(
      rw.Light(
        kind=rw.scene.LightKind.RECT,
        intensity=3.0e5,
        width=0.04,
        height=0.04,
        shadow_enable=False,
        xform=_xform((1.0, 0.0, 1.0), rot),
      )
    )
    host = b.build()
    for form in ("point", "ltc"):
      v, r = _pixel(host, eye, {"lighting": rw.settings.Lights(area_lights=form)}, quat=quat, fovy=0.02)
      assert r.features.clearcoat == (coat > 0.0)
      out[(coat, form)] = v.mean()
  coat_point = out[(1.0, "point")] - out[(0.0, "point")]
  coat_ltc = out[(1.0, "ltc")] - out[(0.0, "ltc")]
  assert coat_point > 0.0
  assert coat_ltc == pytest.approx(coat_point, rel=0.15)


def test_clearcoat_reflects_the_sky():
  """A black dielectric under a uniform sky: the coat adds its ~4 % albedo on top of the base's
  attenuated reflection; a white one barely changes (energy is conserved: what the coat reflects
  the base does not get)."""
  eye = (0.0, 0.0, 1.0)
  black = (0.0, 0.0, 0.0)
  plain, _ = _pixel(
    _coat_scene(0.0, 0.0, light=False, sky=1.0, color=black), eye, {"environment": rw.settings.Environment(enabled=True)}
  )
  coated, r = _pixel(
    _coat_scene(1.0, 0.05, light=False, sky=1.0, color=black), eye, {"environment": rw.settings.Environment(enabled=True)}
  )
  assert r.features.clearcoat
  # coated = plain x (1 - E_c) + E_c x sky: the coat's albedo at normal incidence is F0 = 0.04.
  coat_albedo = (coated.mean() - plain.mean()) / (1.0 - plain.mean())
  assert coat_albedo == pytest.approx(0.04, abs=0.004)
  white = (1.0, 1.0, 1.0)
  plain, _ = _pixel(
    _coat_scene(0.0, 0.0, light=False, sky=1.0, color=white), eye, {"environment": rw.settings.Environment(enabled=True)}
  )
  coated, _ = _pixel(
    _coat_scene(1.0, 0.05, light=False, sky=1.0, color=white), eye, {"environment": rw.settings.Environment(enabled=True)}
  )
  assert coated.mean() <= plain.mean() * 1.01
  assert coated.mean() == pytest.approx(plain.mean(), rel=0.05)


def test_clearcoat_bits_round_trip():
  bits = int(core_types.pack_clearcoat(0.5, 0.25))
  assert (bits >> core_types.COAT_WEIGHT_SHIFT) & 255 == 128
  assert (bits >> core_types.COAT_ROUGHNESS_SHIFT) & 255 == 64
  gpu = rw.upload(_coat_scene(0.5, 0.25), device=DEVICE)
  assert gpu.capabilities.clearcoat
  assert int(gpu.tables["instance.shading"].numpy()[0, 0]) == bits


def _hedge_scene(cut: bool) -> rw.Scene:
  """A 1 m cube of foliage (an opacity map of 0.3, cutoff 0.5: entirely cut away when `cut`, else
  opaque) on a floor, under a sun."""
  b = rw.SceneBuilder()
  _floor(b)
  mat = rw.Material(base_color=(0.2, 0.6, 0.2), base_metalness=0.0, specular_roughness=1.0)
  if cut:
    img = np.full((2, 2, 4), round(0.3 * 255), np.uint8)
    tex = b.add_texture(img, colorspace="linear")
    mat = dataclasses.replace(mat, ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5))
    mat = mat.with_texture(Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=3))
  m = b.add_material(mat)
  # 2 cm above the floor (a face coplanar with the floor's would hide it from the continued rays).
  b.add_instance("box", size=(0.5, 0.5, 0.5), material=m, xform=_xform((0.0, 0.0, 0.52)))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=_xform()))
  return b.build()


def _foliage_stack_scene(layers: int) -> rw.Scene:
  """A floor under `layers` cut-away horizontal quads in one mesh (an opacity map of 0.3, cutoff
  0.5) and as many cut-away thin boxes, 1-2 m up; a sun overhead. `layers` 0: the floor alone."""
  b = rw.SceneBuilder()
  _floor(b)
  if layers:
    img = np.full((2, 2, 4), round(0.3 * 255), np.uint8)
    tex = b.add_texture(img, colorspace="linear")
    mat = rw.Material(base_color=(0.2, 0.6, 0.2), base_metalness=0.0, specular_roughness=1.0)
    mat = dataclasses.replace(mat, ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5))
    m = b.add_material(mat.with_texture(Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=3)))
    quad = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    points = np.concatenate([quad + np.array([0.0, 0.0, 1.0 + 0.05 * k]) for k in range(layers)])
    tris = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + 4 * k for k in range(layers)])
    uvs = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (layers, 1))
    b.add_instance("mesh", mesh=b.add_mesh(points, tris, uvs=uvs), material=m)
    for k in range(layers):
      b.add_instance("box", size=(1.0, 1.0, 0.01), material=m, xform=_xform((0.0, 0.0, 1.6 + 0.05 * k)))
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=_xform()))
  return b.build()


def test_a_build_without_the_configuration_s_alpha_test_binds_zero_cutoffs():
  """A build compiling the alpha test serving a configuration without it (textures off on a
  cut-out scene: `tiers._SUPERSETS`) binds (1, nmat) zero cutoffs, so no material is tested
  (`frame_structs.alpha_cutoffs`); with the test, the frame's own table. (The routing:
  `test_zero_compile.py::test_run_time_bits_route_to_a_fill_point`; the OptiX image identity is a
  GPU check.)"""
  from types import SimpleNamespace

  from raywarp.core.trace import frame_structs

  table = wp.array(np.full((1, 3), 0.5, np.float32), dtype=float, device=DEVICE)
  frame = SimpleNamespace(material_alpha_cutoff=table)
  scene = SimpleNamespace(device=DEVICE)
  for build, effective, zero in ((True, False, True), (True, True, False), (False, False, False)):
    ctx = SimpleNamespace(
      features=SimpleNamespace(alpha_cutout=build), effective=SimpleNamespace(alpha_cutout=effective), scene=scene
    )
    out = frame_structs.alpha_cutoffs(ctx, frame).numpy()  # pyrefly: ignore[bad-argument-type]
    assert out.shape == (1, 3) and (not out.any() if zero else (out == 0.5).all())


def test_cutout_rays_pass_any_number_of_layers():
  """Camera and shadow rays pass every cut-away layer of a stack of 12 quads in one mesh and 12
  boxes (more than the continuation cap of wrapped queries): the floor looks as without them."""
  eye = (0.0, 0.0, 3.0)
  bare, _ = _pixel(_foliage_stack_scene(0), eye)
  seen, r = _pixel(_foliage_stack_scene(12), eye)
  assert r.features.alpha_cutout
  np.testing.assert_allclose(seen, bare, rtol=1e-3)


def test_probe_placement_passes_cutout_texels():
  """Probes inside a cut-away cube stay active (the relocation rays pass its texels, as the probe
  rays do); inside an opaque cube they are buried (inactive)."""
  settings = {
    "gi": rw.settings.GI(enabled=True, spacing=0.25, rays=32, bake_updates=1, bounds=((-0.4, -0.4, 0.1), (0.4, 0.4, 0.9)))
  }
  active = {}
  for cut in (False, True):
    _, r = _pixel(_hedge_scene(cut), (0.0, -3.0, 0.5), settings, quat=(math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)))
    assert r.features.alpha_cutout == cut
    active[cut] = required(r.debug.gi_volume).struct.active.numpy()
  assert active[False].mean() < 0.5  # buried in the opaque cube
  assert active[True].all()


# ---------------------------------------------------------------------------------------------
# The shading counter packet records every resolved setting as valid JSON.


@pytest.mark.parametrize("recipe_name", ("fast", "default"))
def test_counter_settings_include_complete_output_specs(recipe_name: str) -> None:
  target = recipe(recipe_name).replace(outputs=rw.Outputs.of("rgb", "hdr"))
  payload = json.loads(json.dumps(dataclasses.asdict(target), allow_nan=False))
  assert set(payload) == {field.name for field in dataclasses.fields(target)}
  assert payload["outputs"]["rgb"] is not None
  assert payload["outputs"]["hdr"]["dtype"] == "float32"
  for field in dataclasses.fields(target):
    group = getattr(target, field.name)
    assert set(payload[field.name]) == {member.name for member in dataclasses.fields(group)}
