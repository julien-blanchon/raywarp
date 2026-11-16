"""The outputs request API (`rw.Outputs` and its specs) and every output it can request: what each
holds against closed forms or other outputs, float16 variants, the geometry-only path, and that
requesting nothing new keeps every other output bit-identical. CPU, plus CUDA where available."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, assert_type

import numpy as np
import pytest
import warp as wp
from conftest import DOWN, MakeScene, MjScene, render_host, required
from fixtures.conformance import module_loads

import raywarp as rw
from raywarp._base.named import replace_named
from raywarp.core.build.variants import registers
from raywarp.core.kernels.abi import output_layout as ol
from raywarp.core.kernels.abi.output_layout import OutputLayout, layout_of
from raywarp.core.outputs.render_outputs import Buffers, RenderOutputs
from raywarp.core.trace.optix.supersample import hi_outputs
from raywarp.scene.model.types import AlphaMode, Slot, TextureRef
from raywarp.settings.fields import OutputKind

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp.core.outputs.render_outputs import Color
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.trace.context import FrameContext
import gc
import weakref

from raywarp.core.outputs.render_outputs import camera_view

_DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


RES = (32, 24)


# A textured floor, a box shadowing it from a directional sun, an emissive sphere, a metal one.
XML = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="64" height="64" rgb1=".1 .2 .3" rgb2=".8 .8 .8"/>
    <material name="floor" texture="grid" texrepeat="2 2" roughness="0.7"/>
    <material name="glow" rgba=".9 .5 .2 1" emission="2"/>
    <material name="metal" rgba=".8 .8 .8 1" metallic="1" roughness="0.3"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0.3 0.2 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 .1" material="floor"/>
    <geom name="box" type="box" size=".25 .25 .25" pos="0 0 .4" rgba=".8 .2 .2 1"/>
    <geom name="glow" type="sphere" size=".2" pos=".7 0 .3" material="glow"/>
    <geom name="metal" type="sphere" size=".2" pos="-.7 0 .3" material="metal"/>
    <camera name="cam" pos="0 -2.2 1.6" xyaxes="1 0 0 0 .6 .8"/>
  </worldbody>
</mujoco>
"""


def _img(a: wp.array) -> np.ndarray:
  """Camera 0 of world 0 as float64."""
  return a.numpy()[0, 0].astype(np.float64)


def test_outputs_of_and_parse_build_the_same_request():
  """`Outputs.of` (names, plus full specs) and `Outputs.parse` (plain data) agree."""
  a = rw.Outputs.of("rgb", "seg", depth=rw.settings.DepthOutput(miss="inf", dtype="float16"))
  b = rw.Outputs.parse({"rgb": {}, "seg": {}, "depth": {"miss": "inf", "dtype": "float16"}})
  assert a == b
  assert a.kinds == {"rgb", "seg", "depth"}
  assert rw.Outputs.parse(["rgb", "hdr"]) == rw.Outputs.parse("rgb,hdr") == rw.Outputs.of("rgb", "hdr")
  assert rw.RenderSettings().outputs == rw.Outputs.of("rgb")


@pytest.mark.parametrize(
  ("make", "error", "match"),
  [
    (lambda: rw.Outputs(), ValueError, "at least one output"),
    (lambda: rw.Outputs.of("rgb", "alpha"), ValueError, "unknown outputs"),  # pyrefly: ignore[bad-argument-type]  # an unknown kind on purpose
    (lambda: rw.Outputs.parse({"depth": {"precision": "half"}}), ValueError, "precision"),
    (lambda: rw.Outputs(depth=rw.settings.FloatOutput()), ValueError, "DepthOutput"),  # pyrefly: ignore[bad-argument-type]  # invalid on purpose
    (lambda: rw.settings.ColorOutput(dtype="float16"), ValueError, "channels=3"),
    (lambda: rw.settings.DepthOutput(miss="nan"), ValueError, "miss"),  # pyrefly: ignore[bad-argument-type]  # invalid on purpose
    (lambda: rw.settings.FloatOutput(dtype="float64"), ValueError, "FloatOutput.dtype"),  # pyrefly: ignore[bad-argument-type]  # invalid on purpose
    (lambda: rw.Outputs.parse({"albedo": {"dtype": "float64"}}), ValueError, r"outputs\.albedo .*dtype"),
    (lambda: rw.settings.ShadowOutput(light=-1), ValueError, "non-negative"),
    (lambda: rw.settings.ShadowOutput(light="moon"), ValueError, "sun"),  # pyrefly: ignore[bad-argument-type]  # invalid on purpose
  ],
)
def test_invalid_requests_fail_at_the_boundary(make, error, match):
  """Bad requests raise when built, naming what is wrong."""
  with pytest.raises(error, match=match):
    make()


def test_shaded_and_geometry_outputs():
  """Geometry outputs (and a geometric normal) need no shading; the others do."""
  geometry = ("depth", "seg", "semantic", "position", "uv", "material", "motion")
  assert not rw.Outputs.of(*geometry, normal=rw.settings.NormalOutput(frame="geometric")).shaded
  for kind in (
    "rgb",
    "hdr",
    "albedo",
    "normal",
    "roughness",
    "metallic",
    "emission",
    "direct",
    "indirect",
    "shadow",
    "luminance",
  ):
    assert rw.Outputs.of(kind).shaded, kind


def test_output_word_of_requests():
  """The output word of a request: one store bit per output (the packed RGBA with every shaded
  output but a three-channel color), the conventions, the float16 buffers at their bit."""
  rgba = ol.OUT_RGBA
  assert layout_of(rw.Outputs.of("rgb")) == OutputLayout(bits=rgba)
  assert layout_of(rw.Outputs.of("rgb", "depth", "seg")) == OutputLayout(bits=rgba | ol.OUT_DEPTH | ol.OUT_SEG)
  assert layout_of(rw.Outputs.of("depth")) == OutputLayout(bits=ol.OUT_DEPTH)
  assert layout_of(rw.Outputs.of("albedo")) == OutputLayout(bits=rgba | ol.OUT_ALBEDO)
  # Semantic ids alone go into the seg buffer; with instance ids into their own.
  assert layout_of(rw.Outputs.of("semantic")) == OutputLayout(bits=ol.OUT_SEG | ol.OUT_SEG_SEMANTIC)
  assert layout_of(rw.Outputs.of("seg", "semantic")) == OutputLayout(bits=ol.OUT_SEG | ol.OUT_SEMANTIC)
  depth = rw.settings.DepthOutput(dtype="float16", convention="radial", miss="far")
  assert layout_of(rw.Outputs.of(depth=depth)) == OutputLayout(
    bits=ol.OUT_DEPTH | ol.OUT_DEPTH_RADIAL | ol.OUT_MISS_FAR, half=ol.OUT_DEPTH
  )
  geometric = layout_of(rw.Outputs.of(normal=rw.settings.NormalOutput(frame="geometric", dtype="float16")))
  assert geometric == OutputLayout(bits=ol.OUT_GEOMETRIC, half=ol.OUT_NORMAL)
  planar = layout_of(rw.Outputs.of(rgb=rw.settings.ColorOutput(channels=3, planar=True, dtype="float16")))
  assert planar == OutputLayout(bits=ol.OUT_CHW16)
  word = layout_of(rw.Outputs.of("rgb", hdr=rw.settings.FloatOutput(dtype="float16"))).word
  assert OutputLayout.of_word(word) == OutputLayout(bits=rgba | ol.OUT_HDR, half=ol.OUT_HDR)


def test_direct_indirect_need_lit_surfaces(scene):
  """direct / indirect with unlit surfaces fail when the renderer is built."""
  with pytest.raises(ValueError, match="unlit"):
    scene(XML, cam_res=RES, outputs=rw.Outputs.of("direct"), **{"surfaces.shading": "unlit"})


@pytest.mark.parametrize("device", _DEVICES)
def test_hit_outputs_against_depth_and_seg(scene, device):
  """position lies on the camera ray at the radial depth; material / uv / geometric normal are
  set exactly where seg hits; misses write 0 / -1."""
  outputs = rw.Outputs.of(
    "seg",
    "position",
    "uv",
    "material",
    depth=rw.settings.DepthOutput(convention="radial"),
    normal=rw.settings.NormalOutput(frame="geometric"),
  )
  with wp.ScopedDevice(device):
    s = scene(XML, cam_res=RES, outputs=outputs).render()
    assert not s.r.features.any_rgb  # the geometry-only path
    seg, depth = _img(s.out.seg), _img(s.out.depth)
    position, material = s.out.position.numpy()[0, 0], _img(s.out.material)
    normal, uv = s.out.normal.numpy()[0, 0], s.out.uv.numpy()[0, 0]
    eye = s.mjd.cam_xpos[0]
  hit = seg >= 0
  assert hit.any() and (~hit).any()
  np.testing.assert_allclose(np.linalg.norm(position - eye, axis=-1)[hit], depth[hit], rtol=1e-4)
  np.testing.assert_array_equal(position[~hit], 0.0)
  np.testing.assert_allclose(np.linalg.norm(normal, axis=-1)[hit], 1.0, atol=1e-5)
  floor = seg == 0
  np.testing.assert_allclose(normal[floor], [[0.0, 0.0, 1.0]] * int(floor.sum()), atol=1e-5)
  assert (material[~hit] == -1).all() and (material[floor] == s.mjm.geom_matid[0]).all()
  assert (material[seg == 1] == -1).all()  # the box has no material
  # The floor's checker UVs are tiled (texrepeat 2): not all zero, zero without a material.
  assert np.abs(uv[floor]).max() > 0.5
  np.testing.assert_array_equal(uv[~hit], 0.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_geometry_only_path_matches_the_shaded_geometry(scene, device):
  """A geometry-only render writes the same depth / seg as a shaded one (spp 1: the same centre
  ray), and the same position (on CUDA up to an ulp: the kernels' float contraction differs)."""
  with wp.ScopedDevice(device):
    geo = scene(XML, cam_res=RES, outputs=rw.Outputs.of("depth", "seg", "position")).render()
    shaded = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb", "depth", "seg", "position")).render()
    assert not geo.r.features.any_rgb and shaded.r.features.any_rgb
    for kind in ("depth", "seg"):
      np.testing.assert_array_equal(getattr(geo.out, kind).numpy(), getattr(shaded.out, kind).numpy(), err_msg=kind)
    position = (geo.out.position.numpy(), shaded.out.position.numpy())
  if device == "cpu":
    np.testing.assert_array_equal(*position)
  else:
    np.testing.assert_allclose(*position, atol=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_lighting_split_sums_to_hdr(scene, device):
  """direct + indirect + emission = hdr at spp 1 on opaque surfaces without fog; the emissive
  sphere's emission is non-zero, the box's zero; direct is zero in the box's shadow."""
  outputs = rw.Outputs.of("rgb", "hdr", "direct", "indirect", "emission", "seg", "shadow")
  # The emissive sphere is a mirror too (its reflectance is its specular F0): emission unattenuated.
  xml = XML.replace('emission="2"', 'emission="2" reflectance="0.5"')
  with wp.ScopedDevice(device):
    # No headlight: the sun is the only direct light.
    s = scene(xml, cam_res=RES, outputs=outputs, **{"aa.spp": 1, "lighting.camera_light": False, "reflections.enabled": True})
    s.render()
    assert s.r.features.reflection
    hdr, direct, indirect, emission = (
      s.out.hdr.numpy()[0, 0],
      s.out.direct.numpy()[0, 0],
      s.out.indirect.numpy()[0, 0],
      s.out.emission.numpy()[0, 0],
    )
    seg, shadow = _img(s.out.seg), _img(s.out.shadow)
  np.testing.assert_allclose(direct + indirect + emission, hdr, rtol=1e-4, atol=1e-4 * hdr.max())
  glow = seg == 2
  assert glow.any() and (emission[glow].sum(axis=-1) > 0).all() and (emission[seg == 1] == 0).all()
  shadowed = (seg == 0) & (shadow == 0)
  lit = (seg == 0) & (shadow == 1)
  assert shadowed.any() and lit.any()
  np.testing.assert_array_equal(direct[shadowed], 0.0)
  assert (direct[lit].sum(axis=-1) > 0).all()
  assert (shadow[seg < 0] == 1).all() and ((shadow >= 0) & (shadow <= 1)).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_surface_outputs(scene, device):
  """roughness / metallic are the materials' as shaded (the sphere's roughness widened by specular
  AA, always on: never narrowed, alpha^2 by at most its threshold 0.2); albedo alone is written
  without the normal buffer; luminance is the Rec. 709 luminance of the display color (to the
  8-bit encodings)."""
  outputs = rw.Outputs.of("rgb", "albedo", "roughness", "metallic", "luminance", "seg")
  with wp.ScopedDevice(device):
    s = scene(XML, cam_res=RES, outputs=outputs).render()
    out = s.out
    seg, rough, metal = _img(out.seg), _img(out.roughness), _img(out.metallic)
    rgba, luma = out.rgb.numpy()[0, 0], _img(out.luminance)
  assert out.normal is None
  np.testing.assert_allclose(metal[seg == 3], 1.0)
  widest = (0.3**4 + 0.2) ** 0.25  # perceptual roughness of alpha^2 + the AA threshold
  assert ((rough[seg == 3] >= 0.3 - 1e-6) & (rough[seg == 3] <= widest + 1e-6)).all()
  np.testing.assert_allclose(metal[seg == 0], 0.0)
  srgb = rgba[..., :3] / 255.0
  linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
  y = linear @ np.array([0.2126, 0.7152, 0.0722])
  expected = np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055) * 255.0
  assert np.abs(luma - expected).max() <= 3.0  # the luminance of the unquantised color


@pytest.mark.parametrize("device", _DEVICES)
def test_float16_outputs_match_float32(scene, device):
  """Every float16 output equals its float32 one rounded to half precision (and is half the
  bytes); the rgb is bit-identical either way."""
  kinds: tuple[OutputKind, ...] = ("depth", "hdr", "normal", "position", "emission", "roughness", "uv", "shadow", "direct")
  half = {k: replace_named(required(rw.Outputs.of(k).spec(k)), {"dtype": "float16"}) for k in kinds}
  with wp.ScopedDevice(device):
    full = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb", *kinds)).render()
    halved = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb", **half)).render()
    for k in kinds:
      a, b = getattr(full.out, k), getattr(halved.out, k)
      assert wp.types.type_size_in_bytes(b.dtype) * 2 == wp.types.type_size_in_bytes(a.dtype), k
      with np.errstate(over="ignore"):  # values beyond float16's range round to inf, as on the device
        expected = a.numpy().astype(np.float16)
      np.testing.assert_array_equal(b.numpy(), expected, err_msg=k)
    np.testing.assert_array_equal(full.out.rgb.numpy(), halved.out.rgb.numpy())


@pytest.mark.parametrize("device", _DEVICES)
def test_new_outputs_leave_the_others_unchanged(scene, device):
  """Adding every new output changes none of the existing outputs' pixels: bit-identical on the
  CPU; on CUDA the other kernel's float contraction may move hdr and position by an ulp (the
  8-bit color, depth, seg and guides stay equal)."""
  base = rw.Outputs.of("rgb", "depth", "seg", "hdr", "albedo", "normal")
  extra = (
    "semantic",
    "position",
    "uv",
    "material",
    "roughness",
    "metallic",
    "emission",
    "direct",
    "indirect",
    "shadow",
    "luminance",
  )
  with wp.ScopedDevice(device):
    a = scene(XML, cam_res=RES, outputs=base).render()
    b = scene(XML, cam_res=RES, outputs=replace_named(base, {k: rw.Outputs.of(k).spec(k) for k in extra})).render()
    for k in ("rgb", "depth", "seg", "hdr", "albedo", "normal"):
      want, got = getattr(a.out, k).numpy(), getattr(b.out, k).numpy()
      if device == "cpu" or k != "hdr":
        np.testing.assert_array_equal(got, want, err_msg=k)
      else:
        np.testing.assert_allclose(got, want, rtol=1e-6, err_msg=k)


def test_shadow_light_resolution(scene):
  """ "sun" is the first directional light, an index is checked against the scene's lights."""
  s = scene(XML, cam_res=RES, outputs=rw.Outputs.of(shadow=rw.settings.ShadowOutput(light="sun")))
  assert s.r.context().shadow_light == 0
  with pytest.raises(ValueError, match="out of range"):
    scene(XML, cam_res=RES, outputs=rw.Outputs.of(shadow=rw.settings.ShadowOutput(light=3)))


def test_bind_an_extra_output(scene):
  """An extra output's buffer binds to a caller array like the base ones."""
  s = scene(XML, cam_res=RES, outputs=rw.Outputs.of("rgb", "position"))
  position = wp.zeros((1, 1, RES[1], RES[0]), dtype=wp.vec3)
  s.r.bind_outputs(position=position)
  s.render()
  assert s.out.position.ptr == position.ptr and np.abs(position.numpy()).max() > 0


# ---------------------------------------------------------------------------------------------
# The first-hit outputs against each other across render modes (supersampling, the motion body,
# alpha cut-outs, deformables, cameras of two sizes, the OptiX supersampled copy), and the domain of
# the lighting split's identity direct + indirect + emission = hdr (`RenderOutputs.direct`).


# Every hit output, the depth radial (position lies at that distance from the eye).
HITS = rw.Outputs.of(
  "seg",
  "position",
  "uv",
  "material",
  depth=rw.settings.DepthOutput(convention="radial"),
  normal=rw.settings.NormalOutput(frame="geometric"),
)


# A textured cloth (texture coordinates per vertex) over a floor, seen from above.
FLEX_XML = """
<mujoco>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="64" height="64" rgb1=".9 .1 .1" rgb2=".1 .1 .9"/>
    <material name="checker" texture="checker" texrepeat="2 2"/>
    <material name="floor" rgba=".5 .5 .5 1"/>
  </asset>
  <worldbody>
    <light type="directional" dir="0 0.3 -1"/>
    <geom type="plane" size="3 3 .1" material="floor"/>
    <flexcomp name="cloth" type="grid" count="5 5 1" spacing="0.15 0.15 0.15" pos="0 0 0.5" radius="0.02" dim="2"
      material="checker"/>
    <camera name="cam" pos="0 -1.1 1.6" xyaxes="1 0 0 0 0.7 0.7" fovy="45"/>
  </worldbody>
</mujoco>
"""


_EYE = (0.0, 0.0, 3.0)


def _images(out: rw.RenderOutputs, camera: int = 0) -> dict[str, np.ndarray]:
  """World 0's hit outputs of one camera as float64 / int arrays."""
  view = out.cameras[camera] if out.cameras else out
  return {
    k: getattr(view, k).numpy()[0, 0 if out.cameras else camera]
    for k in ("seg", "depth", "position", "material", "uv", "normal")
  }


def _check_hits(img: dict[str, np.ndarray], eye: np.ndarray) -> np.ndarray:
  """Checks the hit outputs against each other; returns the hit mask (depth > 0: "zero" misses)."""
  depth, position = img["depth"].astype(np.float64), img["position"].astype(np.float64)
  hit = depth > 0.0
  assert hit.any()
  np.testing.assert_allclose(np.linalg.norm(position - eye, axis=-1)[hit], depth[hit], rtol=1e-4)
  np.testing.assert_allclose(np.linalg.norm(img["normal"], axis=-1)[hit], 1.0, atol=1e-4)
  for kind, value in (("position", 0.0), ("normal", 0.0), ("uv", 0.0), ("material", -1), ("seg", -1)):
    np.testing.assert_array_equal(img[kind][~hit], value, err_msg=kind)
  return hit


@pytest.mark.parametrize("case", ["spp2_motion", "two_sizes", "flex", "cutout"])
def test_hit_outputs_agree_across_render_modes(scene, case):
  """position at the radial depth, unit normals, misses 0 / -1, in every mode; supersampling in
  the motion body writes the hit outputs of the plain body's render (its first sample); a second
  camera of another size has its own consistent outputs; a flex hit has its material and texture
  coordinates; a cut-away texel shows the surface behind it."""
  outputs = dataclasses.replace(HITS, rgb=rw.settings.ColorOutput())
  if case == "spp2_motion":
    # The software backend: OptiX renders spp > 1 as one supersampled image, without motion.
    sw = {"aa.spp": 2, "performance.traversal": "software"}
    moving = dataclasses.replace(outputs, motion=rw.settings.MotionOutput())
    s = scene(XML, cam_res=(32, 24), outputs=moving, **sw).render()
    ref = scene(XML, cam_res=(32, 24), outputs=outputs, **sw).render()
    assert s.r.features.history and not ref.r.features.history
    img = _images(s.out)
    _check_hits(img, s.mjd.cam_xpos[0])
    for kind, value in _images(ref.out).items():
      np.testing.assert_array_equal(img[kind], value, err_msg=kind)
  elif case == "two_sizes":
    cams = [rw.CameraSpec(fovy=50.0, resolution=(32, 24)), rw.CameraSpec(fovy=50.0, resolution=(16, 12))]
    s = scene(XML, cameras=cams, outputs=outputs)
    s.r.camera_pose.assign(np.array([[[0.0, -2.2, 1.6, 0.5, 0.0, 0.0, 0.8660254]] * 2], np.float32))
    s.render()
    for c, (w, h) in enumerate(((32, 24), (16, 12))):
      img = _images(s.out, c)
      assert img["seg"].shape == (h, w)
      _check_hits(img, np.array([0.0, -2.2, 1.6]))
  elif case == "flex":
    s = scene(FLEX_XML, cam_res=(48, 48), outputs=outputs).render()
    img = _images(s.out)
    _check_hits(img, s.mjd.cam_xpos[0])
    cloth = img["seg"] == s.scene.flex_segment
    assert cloth.sum() > 200
    assert (img["material"][cloth] == s.mjm.flex_matid[0]).all() and s.mjm.flex_matid[0] >= 0
    assert np.abs(img["uv"][cloth]).max() > 0.5  # the checker's tiled UVs, not zero
  else:
    r = render_host(_cutout_scene(), _EYE, DOWN, fovy=40.0, res=(24, 24), settings={"outputs": outputs})
    assert r.features.alpha_cutout
    img = _images(r.outputs)
    hit = _check_hits(img, np.array(_EYE))
    assert hit.all() and (img["seg"] == 0).all()  # the cut-away box over the centre shows the floor
    assert (img["material"] == img["material"][0, 0]).all() and img["material"][0, 0] >= 0


def _cutout_scene() -> rw.Scene:
  """A floor (instance 0) under a box of fully cut-away texels (opacity 0.3, cutoff 0.5)."""
  b = rw.SceneBuilder()
  floor = b.add_material(rw.Material(base_color=(0.8, 0.8, 0.8), base_metalness=0.0, specular_roughness=1.0))
  b.add_instance("box", size=(20.0, 20.0, 0.05), material=floor, xform=np.c_[np.eye(3), [0.0, 0.0, -0.05]])
  tex = b.add_texture(np.full((2, 2, 4), round(0.3 * 255), np.uint8), colorspace="linear")
  cut = rw.Material(
    base_color=(0.2, 0.6, 0.2),
    base_metalness=0.0,
    specular_roughness=1.0,
    ext=rw.scene.RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5),
  )
  m = b.add_material(cut.with_texture(Slot.GEOMETRY_OPACITY, TextureRef(tex, channel=3)))
  b.add_instance("box", size=(1.0, 1.0, 0.5), material=m, xform=np.c_[np.eye(3), [0.0, 0.0, 0.6]])
  b.add_light(rw.Light(kind=rw.scene.LightKind.DISTANT, intensity=10.0, angle=0.0, xform=np.c_[np.eye(3), np.zeros(3)]))
  return b.build()


@pytest.mark.parametrize(
  "color", [rw.settings.ColorOutput(channels=3, planar=True, dtype="float16"), rw.settings.ColorOutput(channels=3)]
)
def test_supersampled_layout_matches_its_kernel(color):
  """`hi_outputs` (the supersampled buffers): a float32 hdr and the packed RGBA; the color's layout
  and luminance are written at the output resolution only (the downsample), the first-hit outputs
  as requested."""
  out = rw.Outputs.of(
    "seg",
    "position",
    rgb=color,
    hdr=rw.settings.FloatOutput(dtype="float16"),
    luminance=rw.settings.LuminanceOutput(dtype="float16"),
  )
  hi = hi_outputs(out)
  word, own = layout_of(hi), layout_of(out)
  assert not word.bits & ol.OUT_DISPLAY and word.has("rgb") and word.has("hdr") and not word.is_half("hdr")
  assert word.bits & ol.OUT_FIRST == own.bits & ol.OUT_FIRST and word.has("seg")
  assert hi.luminance is None and hi.rgb == rw.settings.ColorOutput() and hi.hdr == rw.settings.FloatOutput()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="OptiX supersampling runs on CUDA")
def test_optix_supersampled_copy(scene):
  """The OptiX supersampled render (spp 2: the downsample kernel's first-hit copy) writes
  consistent hit outputs of the sample loop's first sample, and its chw / luminance outputs exist
  at the output size."""
  outputs = dataclasses.replace(
    HITS,
    rgb=rw.settings.ColorOutput(channels=3, planar=True, dtype="float16"),
    luminance=rw.settings.LuminanceOutput(dtype="float16"),
  )
  common = {"performance.traversal": "optix"}
  with wp.ScopedDevice("cuda:0"):
    s = scene(XML, cam_res=(32, 24), outputs=outputs, **common, **{"aa.spp": 2}).render()
    if s.r.features.traversal != "optix":
      pytest.skip("OptiX is unavailable")
    software = {**common, "performance.traversal": "software", "aa.spp": 2}
    ref = scene(XML, cam_res=(32, 24), outputs=outputs, **software).render()
    assert s.out.luminance.shape == (1, 1, 24, 32) and s.out.rgb.shape == (1, 3, 24, 32)
  img, want = _images(s.out), _images(ref.out)
  _check_hits(img, s.mjd.cam_xpos[0])
  # The same first sample as the software sample loop's (hits equal up to the backends' rounding).
  for kind in ("seg", "material"):
    assert (img[kind] == want[kind]).mean() > 0.99, kind
  np.testing.assert_allclose(img["depth"], want["depth"], atol=1e-3)


_SPLIT = rw.Outputs.of("rgb", "hdr", "direct", "indirect", "emission", "seg")


_ENV = np.full((8, 16, 3), 0.5, np.float32)


_TRANSLUCENT = XML.replace('rgba=".8 .2 .2 1"', 'rgba=".8 .2 .2 .5"')


# case -> (scene xml, scene keywords, whether the identity holds on every hit)
_CASES = {
  "plain": (XML, {}, True),
  "environment": (XML, {"envmap": _ENV}, True),
  "gi": (XML, {"gi": rw.settings.GI(enabled=True, spacing=0.5, rays=16)}, True),
  "reflections": (XML.replace('metallic="1"', 'metallic="1" reflectance="0.8"'), {"reflections.enabled": True}, True),
  "fog": (XML, {"atmosphere.start": 0.0, "atmosphere.end": 6.0}, False),
  "transparency": (_TRANSLUCENT, {"transparency.enabled": True}, False),
  "spp2": (XML, {"aa.spp": 2}, False),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_lighting_split_identity_domain(scene, case):
  """direct + indirect + emission = hdr on every hit of an opaque, unfogged spp-1 render, with GI
  and mirror reflections too; fog (applied after the shading), translucent layers
  (composited) and supersampling (hdr averages the samples, the split is the first one's) break
  it, as documented."""
  xml, kw, holds = _CASES[case]
  s = scene(xml, cam_res=(32, 24), outputs=_SPLIT, **{"aa.spp": 1, **kw}).render()
  f = s.r.features
  assert {"gi": f.gi, "reflections": f.reflection}.get(case, True)
  assert {"environment": f.env_lit, "fog": f.fog, "transparency": f.transparency}.get(case, True)
  out = s.out
  hdr, direct, indirect, emission = (
    getattr(out, k).numpy()[0, 0].astype(np.float64) for k in ("hdr", "direct", "indirect", "emission")
  )
  hit = out.seg.numpy()[0, 0] >= 0
  residual = np.abs(direct + indirect + emission - hdr)[hit].max(axis=-1)
  tol = 1e-4 * hdr.max() + 1e-6
  if holds:
    assert residual.max() <= tol, residual.max()
  else:
    assert residual.max() > 100 * tol, residual.max()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="OptiX runs on CUDA")
def test_optix_geometry_build_at_its_register_cap(scene):
  """A geometry-only OptiX build with uv and motion (the temporal state) compiles at the
  geometry cap (64, whatever the shading cap: `FAST`'s 96) and writes consistent hit
  outputs."""
  outputs = dataclasses.replace(HITS, motion=rw.settings.MotionOutput())
  settings = {"performance.traversal": "optix"}
  with wp.ScopedDevice("cuda:0"):
    s = scene(XML, cam_res=(32, 24), outputs=outputs, **settings)
    if s.r.features.traversal != "optix":
      pytest.skip("OptiX is unavailable")
    assert not s.r.features.any_rgb and registers.optix_registers(registers.register_caps(s.r.settings), s.r.features, 0) == 64
    s.render()
    s.render()
    motion = s.out.motion.numpy()
  _check_hits(_images(s.out), s.mjd.cam_xpos[0])
  assert np.isfinite(motion).all()


# ---------------------------------------------------------------------------------------------
# Depth and segmentation conventions (`settings.Outputs`: depth, miss_depth, segmentation).


# A camera looking slightly down at a sphere and a box over a floor that ends: the top rows see
# the sky (misses).
_CONVENTIONS_XML = """
<mujoco>
  <visual><map znear="0.01" zfar="8"/></visual>
  <statistic extent="1"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <geom name="ball" type="sphere" pos="0 0 0.4" size="0.3"/>
    <geom name="crate" type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
    <camera name="cam" pos="0 -2.5 0.8" xyaxes="1 0 0 0 0.2 1" fovy="60"/>
  </worldbody>
</mujoco>
"""


def _depth(scene: MakeScene, outputs: rw.Outputs) -> tuple[MjScene, np.ndarray]:
  """(the rendered scene, the depth output of camera 0, world 0) of `_CONVENTIONS_XML` with
  `outputs`."""
  s = scene(_CONVENTIONS_XML, cam_res=RES, outputs=outputs).render()
  return s, s.depth()[0]


def _seg(scene: MakeScene) -> np.ndarray:
  """The seg output of camera 0, world 0 of `_CONVENTIONS_XML`."""
  return scene(_CONVENTIONS_XML, cam_res=RES, outputs=rw.Outputs.of("seg")).render().seg()[0]


def test_radial_depth_is_planar_depth_over_the_ray_cosine(scene):
  """outputs.depth="radial" is the distance along each pixel's ray: planar / cos(off-axis angle)."""
  kinds = rw.Outputs.of("depth")
  _, planar = _depth(scene, kinds)
  _, radial = _depth(scene, rw.Outputs.of(depth=rw.settings.DepthOutput(convention="radial")))
  w, h = RES
  half = math.tan(math.radians(30.0))
  x = (np.arange(w) + 0.5 - 0.5 * w) / (0.5 * h) * half
  y = (np.arange(h) + 0.5 - 0.5 * h) / (0.5 * h) * half
  cosine = 1.0 / np.sqrt(1.0 + x[None, :] ** 2 + y[:, None] ** 2)
  hit = planar > 0.0
  assert hit.any() and (~hit).any()
  np.testing.assert_allclose(radial[hit] * cosine[hit], planar[hit], rtol=2e-4)
  assert np.all(radial[~hit] == 0.0)


@pytest.mark.parametrize(("miss", "value"), [("zero", 0.0), ("inf", np.inf), ("far", 8.0)])
def test_miss_depth(scene, miss, value):
  """Rays that hit nothing get 0 (MuJoCo Warp), inf (Isaac Lab) or the camera's zfar."""
  s, depth = _depth(scene, rw.Outputs.of(depth=rw.settings.DepthOutput(miss=miss)))
  assert s.r.cameras[0].zfar is None and s.scene.clip[1] == 8.0
  sky = depth[0]  # the top row sees only sky
  assert np.all(sky == value)
  assert np.all(np.isfinite(depth[-1])) and np.all(depth[-1] > 0.0)


def test_depth_seg_only_renders_again_with_spot_lights(scene):
  """Regression: a depth / segmentation-only render of a MuJoCo model with spot lights raised on
  its second frame (the live capability check flagged the spot lights `canonical` compiles out)."""
  s = scene(_CONVENTIONS_XML, cam_res=RES, outputs=rw.Outputs.of("depth", "seg"))
  assert int(s.mjm.light_type[0]) == 0 and not s.r.features.has_spot  # a spot light, compiled out
  s.render().render()


def test_authored_reserved_label_is_unlabelled():
  """An authored label named like a reserved class ("background": the miss id 0) counts as
  unlabelled, as `set_semantics` rejects that name: no instance reads as a miss."""

  from raywarp.core.outputs.semantic_classes import UNLABELLED, Semantics

  s = Semantics(["wall", "cup"], ["background", "cup"], device="cpu")
  assert s.labels == ("background", "unlabelled", "cup")
  assert s.ids.numpy().tolist() == [[UNLABELLED, 2]]
  with pytest.raises(ValueError, match="reserved"):
    s.assign({"background": "wall"})


def _semantic(s):
  """The semantic ids of camera 0, world 0."""
  return s.out.semantic.numpy()[0, 0]


def test_semantics_rejects_mismatched_authored_labels_and_empty_names():
  """Review C O1: more authored labels than instances failed with an IndexError deep in
  `assign` (fewer silently left the rest unlabelled), and an empty rule label became a class
  named ""."""

  from raywarp.core.outputs.semantic_classes import Semantics

  with pytest.raises(ValueError, match="3 authored labels for 2 instances"):
    Semantics(["wall", "cup"], ["a", "b", "c"], device="cpu")
  s = Semantics(["wall", "cup"], device="cpu")
  with pytest.raises(ValueError, match="non-empty"):
    s.assign({"": "wall"})
  for label in (5, 0):  # an int label became a class "named" 5; 0 read as an empty name
    with pytest.raises(TypeError, match="semantic labels must be names"):
      s.assign({label: [0]})  # pyrefly: ignore[bad-assignment]  # an int label on purpose


def test_semantic_segmentation(scene):
  """The semantic output: 0 on a miss, 1 unlabelled, then the labels' class ids; relabelling
  writes the table in place (same Features)."""
  instance = _seg(scene)
  s = scene(_CONVENTIONS_XML, cam_res=RES, outputs=rw.Outputs.of("semantic")).render()
  assert s.out.seg is None
  np.testing.assert_array_equal(_semantic(s), np.where(instance == -1, 0, 1))
  features = s.r.features
  with module_loads() as loads:
    labels = s.r.set_semantics({"object": ["ball", "crate"], "ground": "floor", "box": 2})
    semantic = _semantic(s.render())
  assert s.r.features is features and not loads, loads
  assert labels == ("background", "unlabelled", "object", "ground", "box")
  assert s.r.semantic_labels == labels
  expected = np.choose(instance + 1, [0, 3, 2, 4]).astype(np.int32)  # miss, floor, ball, crate
  np.testing.assert_array_equal(semantic, expected)
  with pytest.raises(ValueError, match="matches no instance"):
    s.r.set_semantics({"x": "nothing*"})
  with pytest.raises(ValueError, match="reserved"):
    s.r.set_semantics({"background": "ball"})


def test_instance_and_semantic_ids_together(scene):
  """seg and semantic requested together: both buffers, each as when requested alone."""
  instance = _seg(scene)
  s = scene(_CONVENTIONS_XML, cam_res=RES, outputs=rw.Outputs.of("seg", "semantic")).render()
  s.r.set_semantics({"object": ["ball", "crate"], "ground": "floor", "box": 2})
  s.render()
  np.testing.assert_array_equal(s.seg()[0], instance)
  np.testing.assert_array_equal(_semantic(s), np.choose(instance + 1, [0, 3, 2, 4]).astype(np.int32))


def test_eye_on_a_plane_agrees_in_depth_and_seg(scene):
  """Regression (sweep S1): with the eye exactly on the floor plane, the floor's hits at distance
  0 were kept: seg said floor while depth said miss (0). A hit at t <= 0 is no hit."""
  on_floor = _CONVENTIONS_XML.replace('pos="0 -2.5 0.8" xyaxes="1 0 0 0 0.2 1"', 'pos="0 -1.5 0" xyaxes="1 0 0 0 0 1"')
  s = scene(on_floor, cam_res=RES, outputs=rw.Outputs.of("depth", "seg")).render()
  depth, seg = s.depth()[0], s.seg()[0]
  assert (seg >= 0).any()
  np.testing.assert_array_equal(seg >= 0, depth > 0.0)
  assert not (seg[-1] == 0).any()  # the bottom row looks down at the floor from on it: no hit


# ---------------------------------------------------------------------------------------------
# Shape contracts of the render outputs (`core/outputs/render_outputs.py`): `RenderOutputs[NWorld,
# NCam, H, W]` binds its dims from its arrays and a generic consumer keeps them (checked by pyrefly:
# `uv run pyrefly check tests/render/test_outputs.py`; the asserts below run as ordinary tests).


def _color[NW: IntVar, NC: IntVar, H: IntVar, W: IntVar](out: RenderOutputs[NW, NC, H, W]) -> Color[NW, NC, H, W]:
  """A consumer generic over the outputs' dims: the color batch, its shape kept."""
  if out.rgb is None:
    raise ValueError("no color output")
  return out.rgb


def test_render_outputs_bind_their_dims():
  """Literal sizes bind the class's dims; a consumer's result keeps them."""
  with wp.ScopedDevice("cpu"):
    out = RenderOutputs(rgb=wp.zeros((4, 2, 3, 5), dtype=wp.vec4ub), depth=wp.zeros((4, 2, 3, 5), dtype=wp.float32))
  assert_type(out, "RenderOutputs[4, 2, 3, 5]")
  color = assert_type(_color(out), "Color[4, 2, 3, 5]")
  assert color.shape == (4, 2, 3, 5)
  assert sorted(out.arrays()) == ["depth", "rgb"]


def test_buffers_views_per_camera_and_dense():
  """`Buffers` binds NWorld from its count; the dense and per-camera views alias one buffer."""
  with wp.ScopedDevice("cpu"):
    outputs = rw.Outputs(rgb=rw.settings.ColorOutput(channels=3, planar=True), depth=rw.settings.DepthOutput())
    b = Buffers(outputs, nworld=3, sizes=[(5, 4), (5, 4)], device=wp.get_device("cpu"))
  assert_type(b, "Buffers[3]")
  out = assert_type(b.outputs, "RenderOutputs[3, int, int, int]")
  assert out.rgb is not None and out.depth is not None
  assert out.rgb.shape == (6, 3, 4, 5) and out.depth.shape == (3, 2, 4, 5)
  cam = out.cameras[1]
  assert cam.depth is not None and cam.rgb is not None and cam.seg is None
  assert cam.depth.shape == (3, 1, 4, 5) and cam.rgb.shape == (3, 3, 4, 5)
  b.flat["depth"].assign(np.arange(3 * 40).astype(np.float32).reshape(3, 40))
  np.testing.assert_array_equal(cam.depth.numpy()[:, 0], out.depth.numpy()[:, 1])
  b.flat["chw8"].assign(np.arange(3 * 120).astype(np.uint8).reshape(3, 120))
  np.testing.assert_array_equal(cam.rgb.numpy(), out.rgb.numpy()[1::2])


def test_bound_array_on_another_device_is_refused():
  """A caller array must be on the buffers' device (captured at construction)."""
  with wp.ScopedDevice("cpu"):
    b = Buffers(rw.Outputs.of("rgb"), nworld=1, sizes=[(2, 2)], device=wp.get_device("cpu"))
  if not wp.is_cuda_available():
    pytest.skip("needs a second device")
  with pytest.raises(ValueError, match="renders on cpu"):
    b.bind({"rgb": wp.zeros(4, dtype=wp.uint32, device="cuda:0")})


def _context_keeps_simulator_state[S](renderer: rw.Renderer[S]) -> tuple[FrameContext[S], GpuScene[S]]:
  """Static assertion: the renderer/context boundary must preserve its simulator-state type."""
  context = assert_type(renderer.context(), "FrameContext[S]")
  return context, assert_type(context.scene, "GpuScene[S]")


# ---------------------------------------------------------------------------------------------
# Output aliases retain their owners and validate rebinding before changing storage.


def test_camera_array_retains_flat_storage_after_owner_release() -> None:
  """A caller may keep one camera array after closing/reconfiguring its renderer."""
  flat = wp.array(np.arange(26).astype(np.float32).reshape(2, 13), dtype=float, device="cpu")
  owner = weakref.ref(flat)
  camera = assert_type(camera_view(flat, adr=3, width=2, height=2), "wp.array[wp.float32, [2, 1, 2, 2]]")
  expected = np.array([[[[3, 4], [5, 6]]], [[[16, 17], [18, 19]]]], np.float32)
  del flat
  gc.collect()
  assert owner() is not None, "per-camera raw pointer outlived its owning array"
  np.testing.assert_array_equal(camera.numpy(), expected)
  del camera
  gc.collect()
  assert owner() is None


@pytest.mark.parametrize("kind", ["short", "strided", "element_size"])
def test_same_pointer_rebinding_still_validates_metadata(kind: str) -> None:
  source = wp.zeros(4, dtype=wp.uint32, device="cpu")
  buffers = Buffers(rw.Outputs.of("rgb"), nworld=1, sizes=[(2, 2)], device=wp.get_device("cpu"), arrays={"rgb": source})
  if kind == "short":
    bad = source[:2]
  elif kind == "strided":
    bad = source[::2]
  else:
    bad = wp.array(ptr=source.ptr, dtype=wp.uint16, shape=(4,), device="cpu")
  assert bad.ptr == source.ptr
  with pytest.raises(ValueError):
    buffers.bind({"rgb": bad})
  assert buffers.bound["rgb"] is source


def test_rejected_binding_keeps_owned_buffers_alive_and_reusable() -> None:
  buffers = Buffers(rw.Outputs.of("rgb", "depth"), nworld=1, sizes=[(2, 2)], device=wp.get_device("cpu"))
  own = buffers.flat["rgb"]
  own.fill_(19)
  source = wp.zeros(4, dtype=wp.uint32, device="cpu")
  bad = wp.zeros(2, dtype=float, device="cpu")
  with pytest.raises(ValueError):
    buffers.bind({"rgb": source, "depth": bad})
  # A later legitimate binding must not silently allocate a new cleared packed-color buffer.
  buffers.bind({"depth": wp.zeros(4, dtype=float, device="cpu")})
  assert buffers.flat["rgb"] is own
  np.testing.assert_array_equal(own.numpy(), [[19] * 4])


@pytest.mark.parametrize("entry,value", [((0, 1), 0.2), ((1, 0), 0.2), ((2, 0), 0.2), ((2, 2), 2.0), ((0, 1), np.nan)])
def test_opencv_matrix_rejects_entries_that_cannot_be_represented(entry: tuple[int, int], value: float) -> None:
  matrix = np.array([[100, 0, 16], [0, 110, 12], [0, 0, 1]], np.float64)
  matrix[entry] = value
  with pytest.raises(ValueError, match="OpenCV"):
    rw.CameraSpec.from_opencv(matrix, (32, 24))


@pytest.mark.parametrize("adr,width,height,channels", [(-1, 2, 2, 1), (0, 0, 2, 1), (0, 2, 2, 0), (10, 2, 2, 1)])
def test_camera_view_rejects_invalid_storage_span(adr: int, width: int, height: int, channels: int) -> None:
  flat = wp.zeros((2, 13), dtype=float, device="cpu")
  with pytest.raises(ValueError, match="dimensions and offset"):
    camera_view(flat, adr=adr, width=width, height=height, channels=channels)


def test_opencv_canonical_matrix_preserves_intrinsics() -> None:
  camera = rw.CameraSpec.from_opencv([[100, 0, 15.5], [0, 110, 11.5], [0, 0, 1]], (32, 24))
  assert camera.intrinsics == (100, 110, 0, 0)
