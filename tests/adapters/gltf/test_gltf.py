"""rw.formats.gltf: the glTF -> USD conversion read by the USD importer, and its content-hash
cache."""

from __future__ import annotations

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib", reason="needs the gltf extra")
pytest.importorskip("pxr")
from fixtures.gltf import quad_glb
from pxr import Usd

import raywarp as rw
from raywarp.formats.gltf.to_usd import convert  # after the importorskip


def _with_colors(src, dst, colors: np.ndarray) -> None:
  doc = pygltflib.GLTF2().load(str(src))
  blob = doc.binary_blob()
  blob += b"\0" * (-len(blob) % 4)
  data = colors.astype(np.float32).tobytes()
  doc.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
  doc.accessors.append(pygltflib.Accessor(bufferView=len(doc.bufferViews) - 1, componentType=5126, count=4, type="VEC4"))
  doc.meshes[0].primitives[0].attributes.COLOR_0 = len(doc.accessors) - 1
  doc.buffers[0].byteLength = len(blob) + len(data)
  doc.set_binary_blob(blob + data)
  doc.save_binary(str(dst))


def test_color_0_multiplies_the_base_color(tmp_path):
  quad_glb(tmp_path / "quad.glb")
  colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0.5]], np.float32)
  _with_colors(tmp_path / "quad.glb", tmp_path / "c.glb", colors)
  convert(tmp_path / "c.glb", tmp_path / "usd" / "c.usda")
  builder = rw.formats.usd.read(tmp_path / "usd" / "c.usda")
  s = builder.build()
  mid = int(s.instances.material[0])
  assert s.materials.ext_flags[mid] & rw.scene.MaterialFlags.VERTEX_COLOR
  assert s.materials.tex[mid, rw.scene.Slot.BASE_COLOR] >= 0  # textured and vertex colored
  m = s.meshes.get(int(s.instances.mesh[0]))
  got = m.colors[m.color_idx].reshape(-1, 4)
  assert {tuple(c) for c in got.tolist()} == {tuple(c) for c in colors.tolist()}  # every vertex's own color
  assert not [k for k in builder.report.dropped if "display" in k or "vertexColor" in k]


def test_the_cache_converts_once_per_content(tmp_path):
  """The conversion is reused while the asset's content is unchanged (also after a move) and
  redone when it changes; the host scene goes through the .rayw cache next to it."""
  import raywarp.formats.gltf

  (tmp_path / "a").mkdir()
  quad_glb(tmp_path / "a" / "quad.glb")
  cache = tmp_path / "cache"
  first = raywarp.formats.gltf.stage(tmp_path / "a" / "quad.glb", cache=cache)
  host = raywarp.formats.gltf.load(tmp_path / "a" / "quad.glb", cache=cache)
  assert host.ninstance == 1 and host.materials.tex[int(host.instances.material[0]), rw.scene.Slot.BASE_COLOR] >= 0
  assert list(first.parent.glob("*.rayw"))  # the host scene cached next to the stage
  (tmp_path / "b").mkdir()
  (tmp_path / "a" / "quad.glb").rename(tmp_path / "b" / "quad.glb")
  assert raywarp.formats.gltf.stage(tmp_path / "b" / "quad.glb", cache=cache) == first  # same content: no conversion
  quad_glb(tmp_path / "b" / "quad.glb", scale=(2.0, 1.0, 1.0))
  assert raywarp.formats.gltf.stage(tmp_path / "b" / "quad.glb", cache=cache) != first


def test_scene_without_cache_uploads_a_temporary_conversion(tmp_path):
  import raywarp.formats.gltf

  quad_glb(tmp_path / "quad.glb")
  gpu = raywarp.formats.gltf.scene(tmp_path / "quad.glb", device="cpu")
  assert gpu.ninstance == 1
  assert sorted(p.name for p in tmp_path.iterdir()) == ["quad.glb"]


def _with_accessor(src, dst, data: bytes, **accessor) -> int:
  """`src` with one more buffer view of `data` and an accessor over it, saved as `dst`; returns
  the accessor's index."""
  doc = pygltflib.GLTF2().load(str(src))
  blob = doc.binary_blob()
  blob += b"\0" * (-len(blob) % 4)
  doc.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(data)))
  doc.accessors.append(pygltflib.Accessor(bufferView=len(doc.bufferViews) - 1, **accessor))
  doc.buffers[0].byteLength = len(blob) + len(data)
  doc.set_binary_blob(blob + data)
  doc.save_binary(str(dst))
  return len(doc.accessors) - 1


def test_signed_normalized_accessors_read_in_minus_one_to_one(tmp_path):
  """glTF 2.0 §3.11: a normalized signed component c reads max(c / 127, -1): -128 is -1."""
  from raywarp.formats.gltf.reader import Gltf

  quad_glb(tmp_path / "quad.glb")
  data = np.array([-128, -127, 0, 127], np.int8).tobytes()
  i = _with_accessor(
    tmp_path / "quad.glb", tmp_path / "n.glb", data, componentType=5120, count=4, type="SCALAR", normalized=True
  )
  got = Gltf(tmp_path / "n.glb").accessor(i)
  np.testing.assert_allclose(got, [-1.0, -1.0, 0.0, 1.0])


def test_an_accessor_overrunning_its_buffer_view_raises(tmp_path):
  """Elements past the view's bytes are an error, not a read of whatever memory follows."""
  from raywarp.formats.gltf.reader import Gltf

  quad_glb(tmp_path / "quad.glb")
  data = np.zeros(4, np.float32).tobytes()  # 1 VEC4, the accessor claims 1000
  i = _with_accessor(tmp_path / "quad.glb", tmp_path / "bad.glb", data, componentType=5126, count=1000, type="VEC4")
  with pytest.raises(ValueError, match="overruns its buffer view"):
    Gltf(tmp_path / "bad.glb").accessor(i)


def test_images_mapping_to_one_file_name_keep_their_own_files(tmp_path):
  """An embedded image (written as textures/image0.png) and an external file of that name are
  two images: each texture reads its own."""
  from PIL import Image

  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.images[1] = pygltflib.Image(uri="image0.png")  # the normal map, now a file named like image 0's
  doc.save_binary(str(tmp_path / "two.glb"))
  Image.fromarray(np.full((2, 2, 3), (1, 2, 3), np.uint8)).save(tmp_path / "image0.png")
  convert(tmp_path / "two.glb", tmp_path / "usd" / "two.usda")
  stage = Usd.Stage.Open(str(tmp_path / "usd" / "two.usda"))
  files = {
    p.GetName(): p.GetAttribute("inputs:file").Get().resolvedPath
    for p in stage.Traverse()
    if p.GetAttribute("info:id").Get() == "UsdUVTexture"
  }
  assert files["baseColor"] != files["normal"]
  assert np.asarray(Image.open(files["baseColor"]))[0, 0, :3].tolist() == [200, 10, 10]
  assert np.asarray(Image.open(files["normal"]))[0, 0, :3].tolist() == [1, 2, 3]


def test_a_cache_directory_under_home_is_expanded(tmp_path, monkeypatch):
  """`cache="~/..."` is the user's directory (as `rw.formats.usd.load`'s), not a `~` folder in the
  working directory."""
  import raywarp.formats.gltf

  monkeypatch.setenv("HOME", str(tmp_path / "home"))
  monkeypatch.chdir(tmp_path)
  quad_glb(tmp_path / "quad.glb")
  converted = raywarp.formats.gltf.stage(tmp_path / "quad.glb", cache="~/cache")
  assert converted.is_relative_to(tmp_path / "home" / "cache")
  assert not (tmp_path / "~").exists()


def test_punctual_lights_keep_their_candela_and_lux(tmp_path):
  """KHR_lights_punctual: a point / spot light's intensity is its candela (also under a scaled
  node), a directional light's its lux (the normalized sphere gave a quarter of the candela)."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.extensionsUsed = [*(doc.extensionsUsed or []), "KHR_lights_punctual"]
  spot = {"type": "spot", "intensity": 100.0, "spot": {"innerConeAngle": 0.3, "outerConeAngle": 0.6}}
  doc.extensions = {
    "KHR_lights_punctual": {"lights": [{"type": "point", "intensity": 100.0}, spot, {"type": "directional", "intensity": 5.0}]}
  }
  for k, scale in enumerate((1.0, 3.0, 1.0)):
    node = pygltflib.Node(translation=[0.0, 0.0, 2.0], scale=[scale] * 3, extensions={"KHR_lights_punctual": {"light": k}})
    doc.nodes.append(node)
    doc.scenes[0].nodes.append(len(doc.nodes) - 1)
  doc.save_binary(str(tmp_path / "lit.glb"))
  from raywarp.core.pipeline.upload.lights import derive_lights

  derived = derive_lights(rw.formats.gltf.load(tmp_path / "lit.glb").lights)
  np.testing.assert_allclose(np.asarray(derived.intensity)[:, 0], [100.0, 100.0, 5.0], rtol=1e-5)


def test_occlusion_strength_scales_the_occlusion(tmp_path):
  """glTF's occlusionTexture.strength s: occlusion 1 + s (ao - 1) (it was ignored: s = 1)."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].occlusionTexture.strength = 0.25
  doc.save_binary(str(tmp_path / "ao.glb"))
  m = rw.formats.gltf.load(tmp_path / "ao.glb").materials
  slot = rw.scene.Slot.OCCLUSION
  assert m.tex[0, slot] >= 0
  np.testing.assert_allclose(m.tex_scale[0, slot][:3], 0.25)
  np.testing.assert_allclose(m.tex_bias[0, slot][:3], 0.75)


def test_verify_is_usd_loads(tmp_path, monkeypatch):
  """`gltf.load(verify=)` is `rw.formats.usd.load`'s ("source": the cached scene is fresh by the
  converted stage alone; "files": by every file); an unknown value raises before any conversion."""
  import raywarp.formats.gltf
  from raywarp.formats.gltf import api as gltf_api

  quad_glb(tmp_path / "quad.glb")
  cache = tmp_path / "cache"
  first = raywarp.formats.gltf.load(tmp_path / "quad.glb", cache=cache, verify="source")
  assert (
    raywarp.formats.gltf.load(tmp_path / "quad.glb", cache=cache, verify="source").report.timings.get("warm_load") is not None
  )
  assert raywarp.formats.gltf.load(tmp_path / "quad.glb", cache=cache).ninstance == first.ninstance
  monkeypatch.setattr(gltf_api, "_convert", lambda *_: pytest.fail("converted before checking verify"))
  with pytest.raises(ValueError, match="verify must be one of"):
    raywarp.formats.gltf.load(tmp_path / "other.glb", verify="nope")  # pyrefly: ignore[bad-argument-type]


def test_khr_materials_unlit_sets_the_unlit_flag(tmp_path):
  """KHR_materials_unlit converts to the material's `raywarp:unlit` attribute, read back as
  `MaterialFlags.UNLIT` (rendered display-relative, `ShadingFlag.UNLIT`); a lit material has
  none."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].extensions = {"KHR_materials_unlit": {}}
  doc.extensionsUsed = [*(doc.extensionsUsed or []), "KHR_materials_unlit"]
  doc.save_binary(str(tmp_path / "unlit.glb"))
  for name, unlit in (("quad", False), ("unlit", True)):
    convert(tmp_path / f"{name}.glb", tmp_path / "usd" / f"{name}.usda")
    s = rw.formats.usd.read(tmp_path / "usd" / f"{name}.usda").build()
    flags = int(s.materials.ext_flags[int(s.instances.material[0])])
    assert bool(flags & rw.scene.MaterialFlags.UNLIT) == unlit, name


def test_emissive_factor_is_display_relative(tmp_path):
  """BUG-681: glTF's `emissiveFactor` is display-referred: the converter marks the material
  (`raywarp:displayEmission`), read back as `MaterialFlags.DISPLAY_EMISSION` (its emission shown
  display-relative, the surface lit); a material without emission has none."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].emissiveFactor = [1.0, 0.5, 0.0]
  doc.save_binary(str(tmp_path / "emissive.glb"))
  for name, display in (("quad", False), ("emissive", True)):
    convert(tmp_path / f"{name}.glb", tmp_path / "usd" / f"{name}.usda")
    s = rw.formats.usd.read(tmp_path / "usd" / f"{name}.usda").build()
    flags = int(s.materials.ext_flags[int(s.instances.material[0])])
    assert bool(flags & rw.scene.MaterialFlags.DISPLAY_EMISSION) == display, name
    assert not flags & rw.scene.MaterialFlags.UNLIT, name


def test_emission_follows_an_absolute_environment(tmp_path):
  """glTF emission is in the environment's units (glTF viewers: emission 1 = an environment texel
  of 1): under `Lighting(envmap=..., envmap_intensity=k)` (absolute) it becomes a radiance of
  emission x k nits, no longer display-relative (it was e / exposure, ~33 000 x e nits at the
  default EV, beside a k-nit sky: blown out under automatic exposure); a display-relative
  environment (no intensity) and lighting without an environment keep it display-relative."""
  quad_glb(tmp_path / "quad.glb")
  doc = pygltflib.GLTF2().load(str(tmp_path / "quad.glb"))
  doc.materials[0].emissiveFactor = [1.0, 0.5, 0.0]
  doc.save_binary(str(tmp_path / "emissive.glb"))
  sky = np.full((8, 16, 3), 0.5, np.float32)
  display = rw.scene.MaterialFlags.DISPLAY_EMISSION
  for lighting, scale, flagged in (
    (rw.Lighting(envmap=sky, envmap_intensity=3.0), 3.0, False),
    (rw.Lighting(envmap=sky, envmap_intensity=3.0, replace=True), 3.0, False),
    (rw.Lighting(envmap=sky), 1.0, True),
    (rw.Lighting(ambient=(0.1, 0.1, 0.1)), 1.0, True),
  ):
    s = rw.formats.gltf.load(tmp_path / "emissive.glb", lighting=lighting, cache=None)
    m = int(s.instances.material[0])
    assert bool(int(s.materials.ext_flags[m]) & display) == flagged, lighting
    np.testing.assert_allclose(s.materials.emission()[m], np.array([1.0, 0.5, 0.0]) * scale, rtol=1e-6)
