"""`raywarp.integrations.mujoco.from_gltf` on a synthetic glTF quad."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib", reason="needs the gltf extra")
from fixtures.gltf import quad_glb

from raywarp.integrations.mujoco.from_gltf import convert  # after the importorskip


def _world_vertices(m: mujoco.MjModel) -> np.ndarray:
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  adr, num = m.mesh_vertadr[0], m.mesh_vertnum[0]
  return m.mesh_vert[adr : adr + num] @ d.geom_xmat[0].reshape(3, 3).T + d.geom_xpos[0]


def test_material_layers_and_factors(tmp_path):
  quad_glb(tmp_path / "quad.glb")
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out")))

  assert m.nmat == 1
  np.testing.assert_allclose(m.mat_metallic, [0.25])
  np.testing.assert_allclose(m.mat_roughness, [0.75])
  # mujoco-stubs: model arrays have no rank (m.tex_width[i] is not a scalar) and the enums are not
  # accepted as numpy indices.
  roles = mujoco.mjtTextureRole
  texid = m.mat_texid[0]
  for role in (roles.mjTEXROLE_RGB, roles.mjTEXROLE_NORMAL, roles.mjTEXROLE_ORM):
    assert texid[role] >= 0  # pyrefly: ignore[bad-index]
  assert texid[roles.mjTEXROLE_EMISSIVE] == -1  # pyrefly: ignore[bad-index]

  # ORM packs occlusion (R) with glTF metallicRoughness green/blue.
  orm = texid[roles.mjTEXROLE_ORM]  # pyrefly: ignore[bad-index]
  adr, w, h, c = m.tex_adr[orm], m.tex_width[orm], m.tex_height[orm], m.tex_nchannel[orm]
  texel = m.tex_data[adr : adr + w * h * c].reshape(h, w, c)[0, 0]  # pyrefly: ignore[bad-argument-type]
  np.testing.assert_array_equal(texel[:3], [50, 100, 200])


def test_y_up_to_z_up_and_texcoords(tmp_path):
  quad_glb(tmp_path / "quad.glb")
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out")))

  # glTF (x, y, z) -> MuJoCo (x, -z, y); node translation y=2 becomes z=2.
  verts = _world_vertices(m)
  np.testing.assert_allclose(verts.min(axis=0), [0, 0, 2], atol=1e-6)
  np.testing.assert_allclose(verts.max(axis=0), [1, 0, 3], atol=1e-6)

  # MuJoCo stores texcoords with glTF's top-left origin.
  uv = sorted(map(tuple, m.mesh_texcoord.round(6)))
  assert uv == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


@pytest.mark.parametrize("normalize", [None, 0.5])
def test_negative_scale_keeps_outward_winding(tmp_path, normalize):
  quad_glb(tmp_path / "quad.glb", scale=(-1.0, 1.0, 1.0))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out", normalize=normalize)))

  # Mirroring flips handedness; the converter flips the winding so the face normal follows the
  # (inverse-transpose transformed) vertex normal, which stays glTF +z -> MuJoCo -y.
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  face = m.mesh_face[m.mesh_faceadr[0]]
  verts = m.mesh_vert[m.mesh_vertadr[0] :][:4] @ d.geom_xmat[0].reshape(3, 3).T
  n = np.cross(verts[face[1]] - verts[face[0]], verts[face[2]] - verts[face[0]])
  assert n[1] < 0.0

  if normalize is not None:
    extent = np.ptp(_world_vertices(m), axis=0)
    assert extent.max() == pytest.approx(normalize)


def test_texture_transform_is_baked_into_uvs(tmp_path):
  """KHR_texture_transform: uv' = T(offset) R(rotation) S(scale) uv, in glTF's UV space."""
  offset, rotation, scale = (0.5, 0.25), np.pi / 2, (2.0, 3.0)
  quad_glb(tmp_path / "quad.glb", texture_transform={"offset": list(offset), "rotation": rotation, "scale": list(scale)})
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out")))
  c, s = np.cos(rotation), np.sin(rotation)
  expected = []
  for u, v in [(0, 1), (1, 1), (1, 0), (0, 0)]:
    su, sv = scale[0] * u, scale[1] * v
    expected.append((round(c * su + s * sv + offset[0], 5), round(-s * su + c * sv + offset[1], 5)))
  assert sorted(map(tuple, m.mesh_texcoord.round(5))) == sorted(expected)


@pytest.mark.parametrize(("transmission", "alpha"), [(0.5, 0.5), (1.0, 0.2)])
def test_transmission_becomes_translucent(tmp_path, transmission, alpha):
  ext = {"KHR_materials_transmission": {"transmissionFactor": transmission}}
  quad_glb(tmp_path / "quad.glb", material_extensions=ext)
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "out")))
  assert m.mat_rgba[0, 3] == pytest.approx(alpha)


def test_blend_base_color_keeps_its_alpha_channel(tmp_path):
  """BLEND / MASK base color textures are RGBA: MuJoCo keeps only `nchannel` (default 3)
  channels of a file, so the texture needs nchannel="4" (its alpha was dropped)."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].alphaMode = "BLEND"
  doc.save_binary(str(tmp_path / "blend.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "blend.glb", tmp_path / "out")))
  rgba = m.mat_texid[0][mujoco.mjtTextureRole.mjTEXROLE_RGBA]  # pyrefly: ignore[bad-index]
  assert rgba >= 0 and m.tex_nchannel[rgba] == 4


def test_mask_writes_an_opacity_layer_and_its_cutoff(tmp_path):
  """MASK: MJCF materials have no cutoff. The base color is opaque, its alpha an opacity layer
  and the cutoff a custom numeric that raywarp's MuJoCo adapter reads (alpha test)."""
  from raywarp.integrations.mujoco.materials import openpbr_materials
  from raywarp.scene.model.report import LoadReport
  from raywarp.scene.model.types import AlphaMode, Slot

  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].alphaMode = "MASK"
  doc.materials[0].alphaCutoff = 0.3
  doc.save_binary(str(tmp_path / "mask.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "mask.glb", tmp_path / "out")))
  texid = m.mat_texid[0]
  assert texid[mujoco.mjtTextureRole.mjTEXROLE_OPACITY] >= 0  # pyrefly: ignore[bad-index]
  assert texid[mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0  # pyrefly: ignore[bad-index]
  assert m.mat_rgba[0, 3] == 1.0
  (mat,) = openpbr_materials(m, LoadReport())
  assert mat.ext.alpha_mode == AlphaMode.MASK
  np.testing.assert_allclose(mat.ext.alpha_cutoff, 0.3, rtol=1e-6)
  opacity = mat.texture(Slot.GEOMETRY_OPACITY)
  assert opacity is not None and opacity.channel == 0  # the opacity layer's red channel
  # The MuJoCo adapter: the cutoff table and the alpha test compiled in.
  import mujoco_warp as mjw
  import warp as wp

  import raywarp as rw
  import raywarp.integrations.mujoco

  with wp.ScopedDevice("cpu"):
    scene = rw.integrations.mujoco.scene(m, mjw.put_model(m))
  assert scene.capabilities.alpha_cutout
  np.testing.assert_allclose(scene.tables["material.alpha_cutoff"].numpy(), [[0.3]], rtol=1e-6)


def test_shared_images_are_written_once(tmp_path):
  """W13: two materials sharing the glTF images write each texture once (dedup by glTF image)."""
  import copy

  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials.append(copy.deepcopy(doc.materials[0]))
  doc.materials[1].pbrMetallicRoughness.metallicFactor = 0.9  # another material, same images
  second = copy.deepcopy(doc.meshes[0].primitives[0])
  second.material = 1
  doc.meshes[0].primitives.append(second)
  doc.save_binary(str(tmp_path / "two.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "two.glb", tmp_path / "out")))
  assert m.nmat == 2 and m.ntex == 3  # base color, normal, ORM: once each
  np.testing.assert_array_equal(m.mat_texid[0], m.mat_texid[1])
  assert len(list((tmp_path / "out" / "textures").glob("*.png"))) == 3


def test_emissive_strength_becomes_the_emission_scalar(tmp_path):
  """W13: KHR_materials_emissive_strength goes to MJCF `emission` (not clipped into 8 bits)."""
  quad_glb(tmp_path / "quad.glb", material_extensions={"KHR_materials_emissive_strength": {"emissiveStrength": 7.5}})
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].emissiveFactor = [1.0, 0.5, 0.25]
  doc.save_binary(str(tmp_path / "glow.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "glow.glb", tmp_path / "out")))
  np.testing.assert_allclose(m.mat_emission, [7.5])
  assert m.mat_texid[0][mujoco.mjtTextureRole.mjTEXROLE_EMISSIVE] >= 0  # pyrefly: ignore[bad-index]


def test_constant_emissive_on_a_primitive_without_uvs_is_readable(tmp_path):
  """A constant emissive color (baked into a 1 x 1 texture) on a primitive without TEXCOORD_0
  gets constant UVs: without them the texture is never sampled (the emitter rendered black)."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.meshes[0].primitives[0].attributes.TEXCOORD_0 = None
  mat = doc.materials[0]
  mat.pbrMetallicRoughness.baseColorTexture = mat.pbrMetallicRoughness.metallicRoughnessTexture = None
  mat.normalTexture = mat.occlusionTexture = None
  mat.emissiveFactor = [0.2, 0.4, 1.0]
  doc.save_binary(str(tmp_path / "flat.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "flat.glb", tmp_path / "out")))
  adr, num = m.mesh_texcoordadr[0], m.mesh_texcoordnum[0]
  assert adr >= 0 and num > 0
  uv = m.mesh_texcoord[adr : adr + num]
  np.testing.assert_array_equal(uv, np.broadcast_to(uv[0], uv.shape))  # one constant texel


def test_ior_and_specular_factor_set_the_reflectance(tmp_path):
  """KHR_materials_ior / the specularFactor of KHR_materials_specular reach MuJoCo's specular as
  the reflectance level 0.5 sqrt(w) x 5 |ior - 1| / (ior + 1) (canonical_design.md §3.1 inverse);
  w = 1 at ior 1.5 leaves MuJoCo's default 0.5."""
  quad_glb(tmp_path / "quad.glb")
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "quad.glb", tmp_path / "plain")))
  assert m.mat_specular[0] == np.asarray(0.5, np.float32)
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].extensions = {"KHR_materials_ior": {"ior": 2.0}, "KHR_materials_specular": {"specularFactor": 0.64}}
  doc.save_binary(str(tmp_path / "ior.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "ior.glb", tmp_path / "ior")))
  np.testing.assert_allclose(m.mat_specular[0], 0.5 * 0.8 * 5.0 / 3.0, rtol=1e-5)
  # ior 0: KHR_materials_ior's Fresnel-1 mode, MuJoCo's largest specular; below 1: invalid glTF.
  doc.materials[0].extensions = {"KHR_materials_ior": {"ior": 0.0}}
  doc.save_binary(str(tmp_path / "zero.glb"))
  m = mujoco.MjModel.from_xml_path(str(convert(tmp_path / "zero.glb", tmp_path / "zero")))
  assert m.mat_specular[0] == 1.0
  doc.materials[0].extensions = {"KHR_materials_ior": {"ior": 0.5}}
  doc.save_binary(str(tmp_path / "bad.glb"))
  with pytest.raises(ValueError, match=r"ior 0\.5"):
    convert(tmp_path / "bad.glb", tmp_path / "bad")


def test_punctual_lights_and_what_mjcf_cannot_carry(tmp_path, capsys):
  """KHR_lights_punctual -> MJCF lights at their node's pose (photometric intensity, colour, spot
  cutoff); clamp / mirror wrap modes, doubleSided, a spot's inner cone and an emissive strength are
  reported (printed and listed in the MJCF's header comment), never dropped silently."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.samplers = [pygltflib.Sampler(wrapS=33071, wrapT=33648)]
  doc.textures[0].sampler = 0
  material = doc.materials[0]
  material.doubleSided = True
  material.extensions = {**(material.extensions or {}), "KHR_materials_emissive_strength": {"emissiveStrength": 4.0}}
  doc.extensions = {
    "KHR_lights_punctual": {
      "lights": [
        {"type": "point", "color": [1.0, 0.5, 0.25], "intensity": 50.0, "range": 8.0},
        {"type": "spot", "intensity": 20.0, "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.5}},
        {"type": "directional", "intensity": 3.0},
      ]
    }
  }
  base = len(doc.nodes)
  doc.nodes += [
    pygltflib.Node(translation=[1.0, 2.0, 3.0], extensions={"KHR_lights_punctual": {"light": 0}}),
    pygltflib.Node(translation=[0.0, 4.0, 0.0], extensions={"KHR_lights_punctual": {"light": 1}}),
    pygltflib.Node(rotation=[-0.7071068, 0.0, 0.0, 0.7071068], extensions={"KHR_lights_punctual": {"light": 2}}),
  ]
  doc.scenes[0].nodes += [base, base + 1, base + 2]
  doc.extensionsUsed = [*(doc.extensionsUsed or []), "KHR_lights_punctual", "KHR_materials_emissive_strength"]
  doc.save_binary(str(tmp_path / "lit.glb"))
  xml = convert(tmp_path / "lit.glb", tmp_path / "out")
  m = mujoco.MjModel.from_xml_path(str(xml))
  assert m.nlight == 3
  np.testing.assert_allclose(m.light_pos[0], [1.0, -3.0, 2.0], atol=1e-6)  # glTF Y-up -> Z-up
  np.testing.assert_allclose(m.light_diffuse[0], [1.0, 0.5, 0.25])
  np.testing.assert_allclose(m.light_intensity, [50.0, 20.0, 3.0])
  np.testing.assert_allclose(m.light_cutoff[1], np.degrees(0.5), rtol=1e-5)
  np.testing.assert_allclose(m.light_dir[0], [0.0, 1.0, 0.0], atol=1e-6)  # glTF -Z -> Z-up +Y
  np.testing.assert_allclose(m.light_dir[2], [0.0, 0.0, -1.0], atol=1e-5)  # turned to point down
  text = xml.read_text()
  printed = capsys.readouterr().out
  for needle in ("doubleSided", "wrap clamp, mirror", "innerConeAngle", "emissive_strength 4"):
    assert needle in text and needle in printed, needle
