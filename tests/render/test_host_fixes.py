"""Host-scene semantics on small SceneBuilder scenes (CPU): alpha cutout, doubleSided, ior."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DOWN, render_host, required

import raywarp as rw
import raywarp.formats.usd
from raywarp.core.pipeline.upload.optics import fresnel_average
from raywarp.scene.model.types import AlphaMode, InstanceFlags, Slot, TextureRef

DEVICE = "cpu"
RES = 16
QUAD = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
QUAD_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _xform(z: float = 0.0, flip: bool = False) -> np.ndarray:
  rot = np.diag(np.array([1.0, -1.0, -1.0])) if flip else np.eye(3)
  return np.c_[rot, [0.0, 0.0, z]]


def _render(host, settings=None, eye=(0.0, 0.0, 3.0), quat=DOWN, fovy=40.0):
  """(hdr (RES, RES, 3), seg (RES, RES), Renderer) of a camera at `eye` (default: looking down
  -z)."""
  r = render_host(host, eye, quat, fovy=fovy, res=RES, kinds=frozenset({"hdr", "seg"}), settings=settings)
  return required(r.outputs.hdr).numpy()[0, 0].reshape(RES, RES, 3), required(r.outputs.seg).numpy()[0, 0].reshape(RES, RES), r


def _cutout_scene(alpha_mode=AlphaMode.MASK, cutoff=0.5, opacity=1.0, textured=True, flags=InstanceFlags.DEFAULT):
  """A floor (instance 0) under a 2 x 2 m quad at z = 1 (instance 1) whose opacity map is 1 on the
  left half (u < 0.5) and 0.3 on the right half; a sun straight down."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  mat = rw.Material(
    base_color=(0.2, 0.6, 0.2),
    base_metalness=0.0,
    specular_roughness=1.0,
    geometry_opacity=opacity,
    ext=rw.scene.RaywarpExt(alpha_mode=alpha_mode, alpha_cutoff=cutoff),
  )
  if textured:
    img = np.full((2, 2, 4), 255, np.uint8)
    img[:, 1, 3] = round(0.3 * 255)
    tex = b.add_texture(img, colorspace="linear")
    mat = mat.with_texture(Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=3))
  quad = b.add_material(mat)
  b.add_instance("plane", size=(5.0, 5.0, 0.0), material=floor)
  mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  b.add_instance("mesh", mesh=mesh, material=quad, xform=_xform(1.0), flags=flags)
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


def test_alpha_cutout_camera_and_shadow_rays():
  """MASK: the opaque half of the quad is seen and shadows the floor; the 0.3-alpha half is cut
  away for camera and shadow rays (the floor under it is fully lit), with no blending."""
  _, seg, r = _render(_cutout_scene())
  assert r.features.alpha_cutout and not r.features.transparency
  left, right = seg[:, 2 : RES // 2 - 2], seg[:, RES // 2 + 2 : -2]
  assert np.all(left == 1) and np.all(right == 0)
  # Cameras under the quad, looking down at the floor: the cut-away half casts no shadow.
  lit = _render(_cutout_scene(), eye=(0.5, 0.0, 0.5), fovy=5.0)[0].mean()
  shadowed = _render(_cutout_scene(), eye=(-0.5, 0.0, 0.5), fovy=5.0)[0].mean()
  assert shadowed < 0.01 * lit
  np.testing.assert_allclose(lit, 10.0 * 0.8 / np.pi, rtol=0.01)  # Lambert under 10 lux of sun (no GI)


def test_alpha_cutout_shadow_behind_an_opaque_occluder():
  """Shadow rays through the cut-away half still stop at an opaque instance above it (blocked at
  once by its any hit), and the cut-out half alone keeps its alpha-tested shadow."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  leaf = rw.Material(base_color=(0.2, 0.6, 0.2), ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5))
  img = np.full((2, 2, 4), 255, np.uint8)
  img[:, 1, 3] = round(0.3 * 255)
  leaf = leaf.with_texture(Slot.GEOMETRY_OPACITY, TextureRef(b.add_texture(img, colorspace="linear"), channel=3))
  quad = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  b.add_instance("plane", size=(5.0, 5.0, 0.0), material=floor)
  b.add_instance("mesh", mesh=quad, material=b.add_material(leaf), xform=_xform(1.0))
  # An opaque 0.5 m box above the cut-away half only.
  opaque = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5)))
  b.add_instance("box", size=(0.25, 0.25, 0.05), material=opaque, xform=np.c_[np.eye(3), [0.5, 0.0, 2.0]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  host = b.build()
  under_box = _render(host, eye=(0.5, 0.0, 0.5), fovy=5.0)[0].mean()
  under_leaf = _render(host, eye=(-0.5, 0.0, 0.5), fovy=5.0)[0].mean()
  through_cut = _render(host, eye=(0.5, 0.6, 0.5), fovy=5.0)[0].mean()
  np.testing.assert_allclose(through_cut, 10.0 * 0.8 / np.pi, rtol=0.01)
  assert under_box < 0.01 * through_cut and under_leaf < 0.01 * through_cut


@pytest.mark.parametrize(("opacity", "seen"), [(0.6, True), (0.4, False)])
def test_alpha_cutout_without_map_is_all_or_nothing(opacity, seen):
  """A constant opacity against the cutoff: the whole surface is opaque or gone (no blending)."""
  _, seg, r = _render(_cutout_scene(opacity=opacity, textured=False))
  assert not r.features.transparency and not r.features.alpha_cutout
  assert np.all(seg[4:-4, 4:-4] == (1 if seen else 0))


def test_alpha_cutout_factor_scales_the_texel():
  """factor x texel >= cutoff: an opacity factor of 0.4 drops the texels of 1 (0.4 < 0.5) too."""
  _, seg, _ = _render(_cutout_scene(opacity=0.4))
  assert np.all(seg[4:-4, 4:-4] == 0)
  _, seg, _ = _render(_cutout_scene(cutoff=0.25))  # 0.3 >= 0.25: everything stays
  assert np.all(seg[4:-4, 4:-4] == 1)


def test_double_sided_instances_are_not_culled():
  """A quad seen from behind: culled when single-sided, seen (shaded from its back) when the
  instance is doubleSided."""
  for flags, want in ((InstanceFlags.DEFAULT, 0), (InstanceFlags.DEFAULT | InstanceFlags.DOUBLE_SIDED, 1)):
    b = rw.SceneBuilder()
    m = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), base_metalness=0.0, specular_roughness=1.0))
    b.add_instance("plane", size=(5.0, 5.0, 0.0), material=m)
    mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
    b.add_instance("mesh", mesh=mesh, material=m, xform=_xform(1.0, flip=True), flags=flags)  # faces -z
    b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
    hdr, seg, r = _render(b.build())
    assert np.all(seg[4:-4, 4:-4] == want), flags
    assert r.features.two_sided == bool(want)
    if want:
      assert hdr[4:-4, 4:-4].mean() > 0.5  # lit from above on its (front-facing to us) back side


def test_ior_maps_to_the_dielectric_f0():
  b = rw.SceneBuilder()
  for ior in (1.0, 1.33, 1.5, 2.4):
    m = b.add_material(rw.Material(base_color=(0.5, 0.5, 0.5), base_metalness=0.0, specular_roughness=0.5, specular_ior=ior))
    b.add_instance("sphere", size=(0.1, 0, 0), material=m)
  m = b.add_material(
    rw.Material(base_color=(0.5, 0.5, 0.5), base_metalness=0.0, specular_weight=0.25, specular_roughness=0.5, specular_ior=1.33)
  )
  b.add_instance("sphere", size=(0.1, 0, 0), material=m)
  spec = rw.upload(b.build(), device=DEVICE).tables["material.reflectance"].numpy()[0]
  f0 = 0.16 * spec**2
  want = [((n - 1) / (n + 1)) ** 2 for n in (1.0, 1.33, 1.5, 2.4)]
  np.testing.assert_allclose(f0[:4], want, rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(f0[4], want[1] * 0.25, rtol=1e-5)  # specular_weight 0.25: a quarter of F0(ior)


@pytest.mark.parametrize("area_lights", ["point", "ltc"])
def test_rect_light_is_a_lambertian_emitter(area_lights):
  """A small 0.2 m rect light (radiance L, facing down) 2 m above the floor: the illuminance 2 m
  to the side (45 degrees off its axis) is L A cos^2(45) / d^2, its point-light form included
  (intensity L A cos(theta), not L A in every direction)."""
  b = rw.SceneBuilder()
  floor = b.add_material(
    rw.Material(base_color=(1.0, 1.0, 1.0), base_metalness=0.0, specular_weight=0.0, specular_roughness=1.0)
  )
  b.add_instance("plane", size=(10.0, 10.0, 0.0), material=floor)
  b.add_light(
    rw.Light(kind=rw.scene.LightKind.RECT, intensity=1000.0, width=0.2, height=0.2, xform=np.c_[np.eye(3), [0.0, 0.0, 2.0]])
  )
  hdr, _, _ = _render(
    b.build(),
    {"lighting": rw.settings.Lights(area_lights=area_lights), "shadows": rw.settings.Shadows(enabled=False)},
    eye=(2.0, 0.0, 0.5),
    fovy=2.0,
  )
  e = 1000.0 * 0.04 * 0.5 / 8.0  # L A cos(theta_light) cos(theta_floor) / d^2
  # Lambert (specular_weight 0); the old 1 lux falloff window took 8 % here.
  np.testing.assert_allclose(hdr.mean(), e / np.pi, rtol=0.03)


@pytest.mark.parametrize("bits", [8, 16])
def test_grey_png_decodes(tmp_path, bits):
  """Grey PNGs (roughness / height maps) decode (an "L" image was a 2D array: decode failed, mid
  grey); 16-bit ones keep their full range (Pillow's convert("L") clips at 255)."""
  from PIL import Image

  ramp = np.linspace(0, 255, 16).reshape(4, 4)
  stored = (ramp * 257).astype(np.uint16) if bits == 16 else ramp.astype(np.uint8)
  Image.fromarray(stored).save(tmp_path / "ramp.png")
  b = rw.SceneBuilder()
  tex = b.add_texture(str(tmp_path / "ramp.png"), colorspace="linear")
  host = b.build()
  assert not host.report.dropped
  img = host.images.level(int(host.textures.image[tex]), 0)
  np.testing.assert_allclose(img[..., 0].astype(np.float64), ramp, atol=1.0)


def test_transmission_of_an_opaque_mode_material_is_translucent():
  """glTF / OmniGlass transmissive materials are alphaMode OPAQUE: still a thin translucent layer
  (was reset to opaque after the transmission scaled it, while reported as approximated); shadow
  rays pass its see-through part times what its surface Fresnel lets through, 1 - E_F(ior)."""
  b = rw.SceneBuilder()
  glass = b.add_material(
    rw.Material(
      base_color=(1.0, 1.0, 1.0),
      base_metalness=0.0,
      specular_roughness=0.05,
      transmission_weight=1.0,
      transmission_color=(1.0, 1.0, 1.0),
    )
  )
  b.add_instance("sphere", size=(1.0, 0.0, 0.0), material=glass)
  gpu = rw.upload(b.build(), device=DEVICE)
  alpha = float(gpu.tables["material.base_color"].numpy()[0, 0, 3])
  assert alpha == pytest.approx(2.0**-10)
  assert gpu.capabilities.has_alpha
  assert gpu.instance_shadow_transmit[0] == pytest.approx((1.0 - 2.0**-10) * (1.0 - fresnel_average(np.array([1.5]))[0]))


def test_specular_workflow_constants_convert_to_metallic():
  """UsdPreviewSurface useSpecularWorkflow: a gold specularColor over a black diffuse is a gold
  metal (was a black plastic of F0 0.04); a black specularColor is a dielectric without F0."""
  from raywarp.formats.usd.materials.preview import specular_to_metallic

  metallic, base = specular_to_metallic((0.02, 0.02, 0.02), (1.0, 0.77, 0.34))
  assert metallic == pytest.approx(1.0, abs=1e-6)
  np.testing.assert_allclose(base, (1.0, 0.77, 0.34), atol=1e-6)
  metallic, base = specular_to_metallic((0.5, 0.2, 0.1), (0.04, 0.04, 0.04))
  assert metallic == 0.0
  np.testing.assert_allclose(base, (0.5, 0.2, 0.1), atol=1e-6)
  host = rw.formats.usd.load("tests/data/scenes/materials.usda")
  i = list(host.materials.names).index("/World/Looks/SpecularGold")
  assert host.materials.base_metalness[i] == pytest.approx(1.0, abs=1e-5)
  np.testing.assert_allclose(host.materials.base_color[i], (1.0, 0.77, 0.34), atol=1e-5)


def test_a_zero_scale_uv_transform_with_a_detail_normal_map_uploads():
  """A material whose base UV transform has a zero scale (every base UV one point) and a detail
  normal map: the detail map's transform relative to the base has no inverse; the upload reads
  the base UVs and reports it instead of aborting."""
  b = rw.SceneBuilder()
  tex = b.add_texture(np.full((2, 2, 4), 128, np.uint8), colorspace="linear")
  flat = ((0.0, 0.0, 0.5), (0.0, 0.0, 0.5))
  mat = rw.Material(
    textures=((Slot.BASE_COLOR, TextureRef(tex, uv_xform=flat)),),
    extras=rw.scene.OpenPbrExtras(textures=(("geometry_detail_normal", TextureRef(tex)),)),
  )
  quad = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  b.add_instance("mesh", mesh=quad, material=b.add_material(mat))
  host = b.build()
  scene = rw.upload(host, device="cpu")
  assert scene.report.approximated["gpu.material.detail_uv(singular base UV transform: the base's UVs)"] == 1


def test_a_detail_normal_map_ignores_an_out_of_range_texture_id():
  """The detail map's UV transform is relative to the material's UV reference slot, which binds a
  texture of the scene: a base color slot whose id lies past the scene's textures binds none (as
  `uv_plans` reads it), so the next bound slot (roughness, UVs scaled by 2) is the reference, not
  the unbound slot's singular transform."""
  b = rw.SceneBuilder()
  tex = b.add_texture(np.full((2, 2, 4), 128, np.uint8), colorspace="linear")
  flat = ((0.0, 0.0, 0.5), (0.0, 0.0, 0.5))
  twice = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
  mat = rw.Material(
    textures=((Slot.BASE_COLOR, TextureRef(tex, uv_xform=flat)), (Slot.SPECULAR_ROUGHNESS, TextureRef(tex, uv_xform=twice))),
    extras=rw.scene.OpenPbrExtras(textures=(("geometry_detail_normal", TextureRef(tex)),)),
  )
  quad = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  b.add_instance("mesh", mesh=quad, material=b.add_material(mat))
  host = b.build()
  host.materials.tex[0, Slot.BASE_COLOR] = len(host.textures)  # one past the last texture
  scene = rw.upload(host, device="cpu")
  assert not scene.report.approximated
  assert scene.layers is not None
  np.testing.assert_allclose(scene.layers.detail_uv[0], ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0)), atol=1e-7)
