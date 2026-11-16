"""The one BSDF (`kernels/lighting.make_bsdf_eval`, `material.make_layers`, the LTC area
evaluator and `kernels/ambient.py`): energy conservation in a white furnace over roughness x
metalness x coat x the layered lobes, and the invariances that the duplicated evaluators broke
(the sun-disc coat law, EON with a coat under an area light)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

import raywarp as rw
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.lighting import make_bsdf_eval
from raywarp.core.shade.environment import ibl
from raywarp.core.shade.lights import area, ltc
from raywarp.core.shade.lights import punctual as lights
from raywarp.core.shade.materials import material
from raywarp.core.shade.materials.material import Surface
from raywarp.core.shade.materials.openpbr_lobes import eon_albedo
from raywarp.settings.fields import exposure_from_ev100

_WHITE = (1.0, 1.0, 1.0)
_ROUGHNESS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Camera on -y looking +y at a grid of 0.5 m spheres 1.5 m apart in the xz plane.
_RES = 384
_FOVY_DEG = 60.0
_DISTANCE = 12.0
_SPACING = 1.5
_INNER_PX = 4  # NoV > ~0.9 on every sphere (away from the Fresnel rim)


def _row(name: str, roughness: float) -> rw.Material:
  """The furnace material of row `name` at specular roughness `roughness` (all white)."""
  gltf = rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.COAT_GLTF)
  match name:
    case "dielectric":
      return rw.Material(base_color=_WHITE, specular_roughness=roughness)
    case "metal":
      return rw.Material(base_color=_WHITE, base_metalness=1.0, specular_roughness=roughness)
    case "half_metal":
      return rw.Material(base_color=_WHITE, base_metalness=0.5, specular_roughness=roughness)
    case "gltf_coat_dielectric":
      return rw.Material(
        base_color=_WHITE, specular_roughness=0.5, coat_weight=1.0, coat_roughness=roughness, coat_ior=1.5, ext=gltf
      )
    case "gltf_coat_metal":
      return rw.Material(
        base_color=_WHITE,
        base_metalness=1.0,
        specular_roughness=0.5,
        coat_weight=1.0,
        coat_roughness=roughness,
        coat_ior=1.5,
        ext=gltf,
      )
    case "eon":
      return rw.Material(
        base_color=_WHITE, specular_roughness=roughness, extras=rw.scene.OpenPbrExtras(base_diffuse_roughness=1.0)
      )
    case "openpbr_coat_metal":
      return rw.Material(
        base_color=_WHITE, base_metalness=1.0, specular_roughness=0.5, coat_weight=1.0, coat_roughness=roughness
      )
    case "fuzz":
      return rw.Material(
        base_color=_WHITE,
        specular_roughness=0.5,
        extras=rw.scene.OpenPbrExtras(fuzz_weight=1.0, fuzz_roughness=max(roughness, 0.1)),
      )
    case "anisotropic_metal":
      return rw.Material(
        base_color=_WHITE,
        base_metalness=1.0,
        specular_roughness=roughness,
        extras=rw.scene.OpenPbrExtras(specular_roughness_anisotropy=0.5),
      )
  raise ValueError(name)


_BASE_ROWS = ("dielectric", "metal", "half_metal", "gltf_coat_dielectric", "gltf_coat_metal", "eon")
_LAYERED_ROWS = (*_BASE_ROWS, "openpbr_coat_metal", "fuzz", "anisotropic_metal")


def _centers(rows: int) -> np.ndarray:
  """(rows, columns, 3) sphere centres."""
  x = (np.arange(len(_ROUGHNESS)) - 0.5 * (len(_ROUGHNESS) - 1)) * _SPACING
  z = (0.5 * (rows - 1) - np.arange(rows)) * _SPACING
  return np.stack(np.broadcast_arrays(x[None, :], 0.0, z[:, None]), axis=-1)


def _furnace(rows: tuple[str, ...]) -> np.ndarray:
  """(rows, columns) mean display-relative radiance (r, g, b averaged) of each sphere's inner disc
  under a uniform environment of 1 (balanced, sky occlusion off: nothing occludes the sky)."""
  b = rw.SceneBuilder()
  centers = _centers(len(rows))
  for i, name in enumerate(rows):
    for j, roughness in enumerate(_ROUGHNESS):
      m = b.add_material(_row(name, roughness))
      b.add_instance("sphere", size=(0.5, 0.0, 0.0), material=m, xform=np.c_[np.eye(3), centers[i, j]])
  scene = b.build().with_lighting(rw.Lighting(envmap=np.ones((8, 16, 3), np.float32)))
  # No GI (the session's GI default): the spheres' inter-reflection is not the lobes' albedo.
  settings = rw.RenderSettings().replace(**{"outputs": rw.Outputs.of("hdr")})
  r = rw.Renderer(
    rw.upload(scene, "cpu"), nworld=1, cameras=[rw.CameraSpec(fovy=_FOVY_DEG)], resolution=(_RES, _RES), settings=settings
  )
  r.camera_pose.fill_(wp.transformf((0.0, -_DISTANCE, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)))
  hdr = r.render().hdr
  assert hdr is not None
  image = hdr.numpy()[0, 0][..., :3] * exposure_from_ev100(settings.exposure.render_ev100())
  focal = 0.5 * _RES / math.tan(math.radians(0.5 * _FOVY_DEG))
  yy, xx = np.mgrid[:_RES, :_RES] + 0.5
  out = np.zeros(centers.shape[:2])
  for i in range(centers.shape[0]):
    for j in range(centers.shape[1]):
      c = centers[i, j]
      u = 0.5 * _RES + c[0] / _DISTANCE * focal
      v = 0.5 * _RES - c[2] / _DISTANCE * focal
      inside = (xx - u) ** 2 + (yy - v) ** 2 < _INNER_PX**2
      out[i, j] = image[inside].mean()
  return out


def test_white_furnace_grid_conserves_energy():
  """White dielectric, metal, glTF-coated dielectric / metal and a rough-diffuse (EON) dielectric
  over the specular / coat roughness range reflect 1 in a uniform unit environment (no lobe gains
  or loses energy; the base model's build)."""
  disc = _furnace(_BASE_ROWS)
  print({name: [round(float(x), 4) for x in r] for name, r in zip(_BASE_ROWS, disc, strict=True)})
  np.testing.assert_allclose(disc, 1.0, atol=0.002)


def test_white_furnace_grid_layered_lobes():
  """The layered build (OpenPBR coat, fuzz, anisotropy compiled in) keeps the base rows' furnace,
  and its OpenPBR-coated white metal and white fuzz reflect 1. Anisotropic GGX keeps the isotropic
  compensation (Filament's limitation, F1: 0.44-1.21 at a = 1): tracked within 5 % at a = 0.5."""
  disc = _furnace(_LAYERED_ROWS)
  print({name: [round(float(x), 4) for x in r] for name, r in zip(_LAYERED_ROWS, disc, strict=True)})
  np.testing.assert_allclose(disc, 1.0, atol=0.002)


# The invariances, on the device functions themselves -------------------------------------------


def _features(*, specular_anisotropy: bool = False, fuzz: bool = False, coat_layer: bool = False) -> Features:
  """The resolved Features of a small coated scene with these flags (no kernel compiled)."""
  b = rw.SceneBuilder()
  m = b.add_material(
    rw.Material(coat_weight=1.0, coat_ior=1.5, ext=rw.scene.RaywarpExt(flags=rw.scene.MaterialFlags.COAT_GLTF))
  )
  b.add_instance("sphere", size=(0.5, 0.0, 0.0), material=m)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=1.0, angle=0.0))
  r = rw.Renderer(rw.upload(b.build(), "cpu"), nworld=1, cameras=[rw.CameraSpec()], resolution=(4, 4))
  f = r.features
  r.close()
  assert f.clearcoat
  return dataclasses.replace(f, specular_anisotropy=specular_anisotropy, fuzz=fuzz, coat_layer=coat_layer)


def _bsdf_kernel(f: Features) -> wp.Kernel:
  """A kernel evaluating `make_bsdf_eval(f)` for coated surfaces over (view, light) pairs."""
  bsdf_eval = make_bsdf_eval(f)
  lobes_of = material.lobes_of(f)

  @wp.kernel(module="unique")
  def evaluate(
    # Model:
    dfg: wp.Texture2D,
    # In:
    views: wp.array[wp.vec3],
    lights: wp.array[wp.vec3],
    roughness: wp.array[float],
    # Out:
    out: wp.array[wp.vec3],
  ):
    i = wp.tid()
    surface = Surface()
    surface.base_color = wp.vec3(0.8, 0.3, 0.1)
    surface.perceptual_roughness = roughness[i]
    surface.reflectance = 0.5
    surface.metallic = 0.0
    surface.clearcoat = 1.0
    surface.clearcoat_roughness = roughness[i]
    surface.coat_normal = wp.vec3(0.0, 0.0, 1.0)
    surface.coat_nov = wp.max(views[i][2], 1.0e-4)
    normal = wp.vec3(0.0, 0.0, 1.0)
    nov = wp.max(views[i][2], 1.0e-4)
    lobes = lobes_of(dfg, surface, nov)
    l = lights[i]
    out[i] = bsdf_eval(lobes, normal, views[i], nov, l, wp.dot(normal, l), wp.vec3(1.0), 0.0)

  return evaluate


def _evaluate(f: Features) -> np.ndarray:
  """(n, 3) `bsdf_eval` of `f` over a fixed set of directions and roughnesses."""
  rng = np.random.default_rng(3)

  def hemisphere(n: int) -> np.ndarray:
    d = rng.normal(size=(n, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.05
    return (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32)

  n = 256
  views, lights = hemisphere(n), hemisphere(n)
  roughness = rng.uniform(0.0, 1.0, n).astype(np.float32)
  out = wp.empty(n, dtype=wp.vec3, device="cpu")
  wp.launch(
    _bsdf_kernel(f),
    dim=n,
    inputs=[
      ibl.dfg_texture(wp.get_device("cpu")),
      wp.array(views, dtype=wp.vec3, device="cpu"),
      wp.array(lights, dtype=wp.vec3, device="cpu"),
      wp.array(roughness, dtype=float, device="cpu"),
    ],
    outputs=[out],
    device="cpu",
  )
  return out.numpy()


def test_layered_build_matches_the_base_build():
  """Materials without layered lobes shade the same in a layered build (anisotropy, fuzz, the
  OpenPBR coat's normal compiled in) as in the base build: one BSDF, whatever is compiled in."""
  base = _evaluate(_features())
  layered = _evaluate(_features(specular_anisotropy=True, fuzz=True, coat_layer=True))
  np.testing.assert_allclose(layered, base, rtol=2e-5, atol=1e-7)


@wp.kernel(module="unique")
def _eon_under_coat(
  # Model:
  lit: lights.Lights,
  dfg: wp.Texture2D,
  # In:
  coat: float,
  diffuse_roughness: float,
  # Out:
  out: wp.array[wp.vec3],
):
  surface = Surface()
  surface.base_color = wp.vec3(0.9, 0.6, 0.3)
  surface.perceptual_roughness = 0.6
  surface.reflectance = 0.5
  surface.diffuse_roughness = diffuse_roughness
  surface.diffuse_base_weight = 1.0
  surface.clearcoat = coat
  surface.clearcoat_roughness = 0.2
  view = wp.vec3(0.6, 0.0, 0.8)
  normal = wp.vec3(0.0, 0.0, 1.0)
  lobes = _rough_coat_lobes(dfg, surface, view[2])
  fr = _frame(lit, lobes, normal, view, view[2])
  light = area.AreaLight()
  light.shape = wp.static(area.SHAPE_RECT)
  light.center = wp.vec3(0.0, 0.0, 1.0)
  light.normal = wp.vec3(0.0, 0.0, -1.0)
  light.axis_x = wp.vec3(1.0, 0.0, 0.0)
  light.axis_y = wp.vec3(0.0, 1.0, 0.0)
  light.half = wp.vec2(0.5)
  light.radius = 1.0e-4
  light.radiance = wp.vec3(1.0)
  light.range = 0.0
  out[0] = _coated_area(light, fr, lobes, wp.vec3(0.0))
  out[1] = lobes.diffuse_color
  out[2] = wp.vec3(lobes.coat_albedo)


_rough_coat_lobes = material.make_shading_lobes(
  specular_curve=False, clearcoat=True, specular_layers=False, diffuse_roughness=True
)
_frame = ltc.make_frame_func(specular_curve=False, clearcoat=True)
_coated_area = area.make_eval_area_light(clearcoat=True)


def test_eon_under_a_coat_under_an_area_light():
  """A rough-diffuse (EON) base under a coat, under a rect light: the diffuse is EON's directional
  albedo times the Lambert form factor (the analytic albedo ratio), scaled by the coat like every
  other term; never negative (the old quadrature replacement subtracted the Lambert term with the
  wrong weight). Range 0 is unbounded reach, as for punctual lights."""
  device = wp.get_device("cpu")
  lit = lights.Lights()
  lit.ltc_minv, lit.ltc_mag = ltc.ltc_textures(device)
  results = {}
  for coat, rough in ((0.0, 1.0), (1.0, 1.0), (0.0, 0.0)):
    out = wp.empty(3, dtype=wp.vec3, device=device)
    wp.launch(_eon_under_coat, dim=1, inputs=[lit, ibl.dfg_texture(device), coat, rough], outputs=[out], device=device)
    results[coat, rough] = out.numpy()
  bare, coated, lambert = results[0.0, 1.0], results[1.0, 1.0], results[0.0, 0.0]
  # The diffuse albedo is EON's directional albedo at NoV over Lambert's (same specular coupling).
  rho = np.array([0.9, 0.6, 0.3])
  ratio = np.asarray(eon_albedo(wp.vec3(*rho), 1.0, 0.8)) / rho
  np.testing.assert_allclose(bare[1], lambert[1] * ratio, rtol=1e-5)
  assert np.all(bare[0] > 0.0) and np.all(coated[0] > 0.0)
  # The coat scales the base by 1 - E_coat and adds its own (rough 0.2) lobe toward the light.
  under = 1.0 - coated[2][0]
  assert np.all(coated[0] >= bare[0] * under - 1e-6)


def _direct_kernel(f: Features) -> wp.Kernel:
  """A kernel integrating `make_bsdf_eval(f)` x NoL over the hemisphere (stratified in cos theta
  and phi) for a white surface seen at NoV `nov`: its directional albedo under direct light."""
  bsdf_eval = make_bsdf_eval(f)
  lobes_of = material.lobes_of(f)

  @wp.kernel(module="unique")
  def albedo(
    # Model:
    dfg: wp.Texture2D,
    # In:
    metallic: wp.array[float],
    roughness: wp.array[float],
    anisotropy: wp.array[float],
    coat: wp.array[float],
    nov: float,
    n_theta: int,
    n_phi: int,
    # Out:
    out: wp.array[wp.vec3],
  ):
    i = wp.tid()
    surface = Surface()
    surface.base_color = wp.vec3(1.0)
    surface.perceptual_roughness = roughness[i]
    surface.reflectance = 0.5
    surface.metallic = metallic[i]
    surface.anisotropy = anisotropy[i]
    surface.tangent = wp.vec3(1.0, 0.0, 0.0)
    surface.bitangent = wp.vec3(0.0, 1.0, 0.0)
    surface.clearcoat = coat[i]
    surface.clearcoat_roughness = roughness[i]
    normal = wp.vec3(0.0, 0.0, 1.0)
    view = wp.vec3(wp.sqrt(1.0 - nov * nov), 0.0, nov)
    lobes = lobes_of(dfg, surface, nov)
    total = wp.vec3(0.0)
    for a in range(n_theta):
      c = (float(a) + 0.5) / float(n_theta)
      s = wp.sqrt(1.0 - c * c)
      for b in range(n_phi):
        phi = (float(b) + 0.5) * (2.0 * wp.pi / float(n_phi))
        l = wp.vec3(s * wp.cos(phi), s * wp.sin(phi), c)
        total += bsdf_eval(lobes, normal, view, nov, l, c, wp.vec3(1.0), 0.0)
    out[i] = total * (2.0 * wp.pi / float(n_theta * n_phi))

  return albedo


def _direct_albedo(
  f: Features, metallic: list[float], roughness: list[float], anisotropy: list[float], coat: list[float], nov: float
) -> np.ndarray:
  """(n,) mean directional albedo of white surfaces under direct light (`_direct_kernel`)."""
  out = wp.empty(len(metallic), dtype=wp.vec3, device="cpu")
  arrays = [wp.array(np.asarray(v, np.float32), device="cpu") for v in (metallic, roughness, anisotropy, coat)]
  wp.launch(
    _direct_kernel(f),
    dim=len(metallic),
    inputs=[ibl.dfg_texture(wp.get_device("cpu")), *arrays, nov, 512, 512],
    outputs=[out],
    device="cpu",
  )
  return out.numpy().mean(axis=1)


@pytest.mark.parametrize("nov", [0.9, 0.5])
def test_direct_light_furnace(nov: float):
  """The punctual BSDF integrated over the hemisphere (a uniform dome of point lights) reflects
  what the image-based term does: 1 for white dielectrics and metals, glTF-coated or not (the
  coupled diffuse, the coat's albedo scaling and the
  multiscatter compensation, consistent between direct and image-based light). Anisotropic GGX
  keeps the isotropic compensation (Filament's limitation): at a = 0.5 a white metal reflects
  0.99-1.10 (over-reflecting at high roughness), tracked here; at a = 1 the bitangent lobe is at
  the alpha floor, too sharp for this quadrature (the image-based furnace cannot see it either)."""
  rough = [0.3, 0.5, 0.7, 1.0]
  f = _features(specular_anisotropy=True)
  iso = _direct_albedo(f, [0.0] * 4 + [1.0] * 4, rough * 2, [0.0] * 8, [0.0] * 8, nov)
  coated = _direct_albedo(f, [0.0] * 4 + [1.0] * 4, rough * 2, [0.0] * 8, [1.0] * 8, nov)
  aniso = _direct_albedo(f, [1.0] * 4, rough, [0.5] * 4, [0.0] * 4, nov)
  print(
    {
      "nov": nov,
      "isotropic": [round(float(x), 4) for x in np.ravel(iso)],
      "coated": [round(float(x), 4) for x in np.ravel(coated)],
    }
  )
  print({"anisotropic a=0.5": [round(float(x), 4) for x in np.ravel(aniso)]})
  np.testing.assert_allclose(coated, 1.0, atol=0.02)
  np.testing.assert_allclose(iso, 1.0, atol=0.02)
  assert np.all(aniso > 0.98) and np.all(aniso < 1.12)
