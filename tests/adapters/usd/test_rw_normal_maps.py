"""Each importer derives a normal map's per-axis transform from the authored data
(scene/model/normal_map.py): UsdUVTexture scale / bias, MaterialX `normalmap` (file and USD
readers), glTF `normalTexture.scale`, MuJoCo's normal layer."""

from __future__ import annotations

import numpy as np
import pytest
from rw_helpers import bind, new_stage, preview_material, quad, write_png
from rw_helpers import material_named as material_of

import raywarp as rw
from raywarp._base.types import TexRole

S = rw.scene.Slot


def _preview_normal(tmp_path, scale, bias) -> rw.SceneBuilder:
  """A stage whose UsdPreviewSurface normal reads n.png through UsdUVTexture `scale` / `bias`."""
  from pxr import Sdf, UsdShade

  stage = new_stage(tmp_path)
  write_png(tmp_path / "n.png", np.full((2, 2, 3), 128, np.uint8))
  mat, shader = preview_material(stage, "/Looks/M")
  tex = UsdShade.Shader.Define(stage, "/Looks/M/N")
  tex.CreateIdAttr("UsdUVTexture")
  tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("n.png")
  tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
  tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(bias)
  shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
  bind(quad(stage, "/Q"), mat)
  return rw.formats.usd.read(stage)


@pytest.mark.parametrize(
  ("scale", "bias"),
  [((2.0, -2.0, 2.0, 1.0), (-1.0, 1.0, -1.0, 0.0)), ((1.0, 1.0, 2.0, 1.0), (-0.5, -0.5, -1.0, 0.0))],
)
def test_usd_preview_scale_bias_is_the_binding(tmp_path, scale, bias):
  """UsdUVTexture scale / bias on the normal input (a +Y-down map, a strength 0.5) is the binding
  as authored, and the upload honours it (no `gpu.material.normal_scale_bias`)."""
  pytest.importorskip("pxr")
  import raywarp.formats.usd  # noqa: F401  # the adapter

  builder = _preview_normal(tmp_path, scale, bias)
  ref = material_of(builder, "/Looks/M").texture(S.GEOMETRY_NORMAL)
  assert ref is not None and ref.scale == scale and ref.bias == bias
  scene = rw.upload(builder.build(), "cpu")
  assert not [k for k in scene.report.approximated if "normal" in k]


def test_usd_preview_inward_normal_map_is_dropped(tmp_path):
  pytest.importorskip("pxr")
  import raywarp.formats.usd  # noqa: F401  # the adapter

  builder = _preview_normal(tmp_path, (2.0, 2.0, 0.0, 1.0), (-1.0, -1.0, 0.0, 0.0))
  assert material_of(builder, "/Looks/M").texture(S.GEOMETRY_NORMAL) is None
  assert builder.report.dropped["UsdPreviewSurface.normal.inward(z never positive: no normal map)"] == 1
  assert len(builder.build().textures) == 0  # dropped before its texture was registered


def test_usd_mtlx_normalmap_binds_the_normal_slot(tmp_path):
  """A MaterialX normal map (through the MaterialX MDL generator and the MDL SDK) binds the normal
  slot."""
  pytest.importorskip("pxr")
  pytest.importorskip("MaterialX")
  from pxr import Sdf, UsdShade

  import raywarp.formats.usd  # noqa: F401  # the adapter

  stage = new_stage(tmp_path)
  write_png(tmp_path / "n.png", np.full((2, 2, 3), 128, np.uint8))
  mat = UsdShade.Material.Define(stage, "/Looks/M")
  surface = UsdShade.Shader.Define(stage, "/Looks/M/S")
  surface.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
  image = UsdShade.Shader.Define(stage, "/Looks/M/Image")
  image.CreateIdAttr("ND_image_vector3")
  image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("n.png")
  nrm = UsdShade.Shader.Define(stage, "/Looks/M/N")
  nrm.CreateIdAttr("ND_normalmap_vector2")
  nrm.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(image.CreateOutput("out", Sdf.ValueTypeNames.Float3))
  nrm.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((1.0, -1.0))
  surface.CreateInput("geometry_normal", Sdf.ValueTypeNames.Float3).ConnectToSource(
    nrm.CreateOutput("out", Sdf.ValueTypeNames.Float3)
  )
  mat.CreateSurfaceOutput("mtlx").ConnectToSource(surface.CreateOutput("out", Sdf.ValueTypeNames.Token))
  bind(quad(stage, "/Q"), mat)
  builder = rw.formats.usd.read(stage)
  assert material_of(builder, "/Looks/M").texture(S.GEOMETRY_NORMAL) is not None
  assert builder.report.counts["mtlx.generated"] == 1


def test_gltf_normal_texture_scale(tmp_path):
  """glTF normalTexture.scale k folds into the UsdUVTexture scale / bias
  (`raywarp.formats.gltf.to_usd`): the binding is the unpack with strength k on x and y."""
  pygltflib = pytest.importorskip("pygltflib", reason="needs the gltf extra")
  pytest.importorskip("pxr")
  from fixtures.gltf import quad_glb

  import raywarp.formats.usd  # noqa: F401  # the adapter
  from raywarp.formats.gltf.to_usd import convert

  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].normalTexture.scale = 0.5
  doc.save_binary(str(tmp_path / "s.glb"))
  convert(tmp_path / "s.glb", tmp_path / "usd" / "s.usda")
  scene = rw.formats.usd.read(tmp_path / "usd" / "s.usda").build()
  mid = int(scene.instances.material[0])
  np.testing.assert_allclose(scene.materials.tex_scale[mid, S.GEOMETRY_NORMAL], (1.0, 1.0, 2.0, 1.0))
  np.testing.assert_allclose(scene.materials.tex_bias[mid, S.GEOMETRY_NORMAL], (-0.5, -0.5, -1.0, 0.0))
  # The upload applies it (it used to read the plain unpack): x, y at half strength, z unchanged.
  rows = rw.upload(scene, "cpu").tables["material.tex_affine"].numpy()[0, mid, TexRole.NORMAL]
  np.testing.assert_allclose(rows[:, :3], [[0.5, 0.5, 1.0], [0.25, 0.25, 0.0]], atol=1e-7)
