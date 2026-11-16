"""Tangent-space normal maps (scene/model/normal_map.py): the canonical convention (+Y along
+dp/dv), the per-axis transform in the binding's scale / bias, the device's texel map, the gated
unclamped decode, validation and the `.wpbr` round trip."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from conftest import render_host, required
from fixtures.normal_maps import QUAD, TILT, UV_UP, encode, quad
from hypothesis import given
from hypothesis import strategies as st

import raywarp as rw
from raywarp._base.types import NTEXROLE, TEX_IDENTITY, TexRole
from raywarp.core.build.variants import canonical as canonical_lib
from raywarp.core.kernels import pixel
from raywarp.core.kernels.abi.features import Features
from raywarp.core.pipeline.upload.materials import normal_rows, scaled_normal_maps
from raywarp.core.trace.rays.primary import PINHOLE
from raywarp.scene.model import normal_map, wpbr
from raywarp.scene.model.types import TextureId

# A tilt of 30 degrees towards +x and 30 towards +y (both tangent axes, so a flip of one shows).
_FINITE = st.floats(-4.0, 4.0, allow_nan=False, width=32)


def _directx(texels: np.ndarray) -> np.ndarray:
  """The same relief encoded +Y down (DirectX): green inverted."""
  out = texels.copy()
  out[..., 1] = 255 - out[..., 1]
  return out


def _render(scene: rw.Scene) -> tuple[np.ndarray, np.ndarray, bool]:
  """(mean world normal, hdr image, Features.scaled_normal_map) of `scene` seen from above."""
  r = render_host(scene, (0.0, 0.0, 1.0), fovy=10.0, res=4, kinds=frozenset({"normal", "hdr"}))
  normal = required(r.outputs.normal).numpy()[0, 0].reshape(-1, 3).mean(0)
  return normal, required(r.outputs.hdr).numpy()[0, 0, ..., :3], r.features.scaled_normal_map


def test_canonical_convention_is_y_along_dp_dv():
  """+Y (green) runs along +dp/dv, x along +dp/du, and the bitangent follows the UV handedness: a
  V-mirrored mapping turns the relief with it."""
  normal, _, _ = _render(quad(encode(TILT)))
  np.testing.assert_allclose(normal, TILT, atol=0.01)
  mirrored, _, _ = _render(quad(encode(TILT), uvs=UV_UP * np.array([1.0, -1.0]) + np.array([0.0, 1.0])))
  np.testing.assert_allclose(mirrored, TILT * np.array([1.0, -1.0, 1.0]), atol=0.01)


def test_directx_map_with_its_flip_renders_as_the_opengl_one():
  """One relief, lit from +y: encoded OpenGL (identity) and DirectX (`Y_DOWN`) render identically;
  the DirectX map read without its flip mirrors the relief (y negated, darker)."""
  opengl = encode(TILT)
  gl_normal, gl_hdr, gl_wide = _render(quad(opengl))
  dx_normal, dx_hdr, dx_wide = _render(quad(_directx(opengl), normal_map.Y_DOWN))
  np.testing.assert_allclose(dx_hdr, gl_hdr, rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(dx_normal, gl_normal, atol=1e-6)
  wrong_normal, wrong_hdr, _ = _render(quad(_directx(opengl)))
  np.testing.assert_allclose(wrong_normal, gl_normal * np.array([1.0, -1.0, 1.0]), atol=0.01)
  assert wrong_hdr.mean() < 0.9 * gl_hdr.mean()
  assert not gl_wide and not dx_wide  # flips keep the plain decode (and its kernels)


def test_x_flip_mirrors_x_only():
  normal, _, _ = _render(quad(encode(TILT), normal_map.X_FLIPPED))
  np.testing.assert_allclose(normal, TILT * np.array([-1.0, 1.0, 1.0]), atol=0.01)


def test_strength_above_one_compiles_the_unclamped_decode():
  """A strength 2 on a 40-degree tilt decodes x = 1.29 > 1: exact under Features.scaled_normal_map
  (the clamped decode would give 1, a 6-degree error); any strength compiles it, a flip does not."""
  tilt = np.array([math.sin(math.radians(40.0)), 0.0, math.cos(math.radians(40.0))])
  normal, _, wide = _render(quad(encode(tilt), (2.0, 2.0)))
  texel = np.round((tilt + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
  expected = texel * np.array([2.0, 2.0, 1.0])
  assert wide
  np.testing.assert_allclose(normal, expected / np.linalg.norm(expected, keepdims=True), atol=0.01)
  _, _, half = _render(quad(encode(tilt), (0.5, 0.5)))
  assert half


def _glossy_pixel(texels: np.ndarray, uv_extent: float, xy: tuple[float, float]) -> np.ndarray:
  """Mean radiance of a 1 m metal quad (roughness 0.2) under normal map `texels` (UVs 0..uv_extent)
  read with strength `xy`, seen straight down with a point light just behind the camera."""
  b = rw.SceneBuilder()
  mesh = b.add_mesh(QUAD, np.array([[0, 1, 2], [0, 2, 3]]), uvs=UV_UP * uv_extent)
  scale, bias = normal_map.unpack(xy)
  ref = rw.scene.TextureRef(b.add_texture(texels, colorspace="linear"), scale=scale, bias=bias)
  mat = rw.Material(base_color=(0.9, 0.7, 0.3), base_metalness=1.0, specular_roughness=0.2)
  b.add_instance("mesh", mesh=mesh, material=b.add_material(mat.with_texture(rw.scene.Slot.GEOMETRY_NORMAL, ref)))
  light = rw.Light(
    kind=rw.scene.LightKind.SPHERE, intensity=100.0, radius=0.0, treat_as_point=True, shadow_enable=False,
    xform=np.c_[np.eye(3), [0.0, 0.0, 1.5]],
  )  # fmt: skip
  b.add_light(light)
  r = render_host(b.build(), (0.0, 0.0, 1.0), fovy=1.0, res=2)
  assert r.features.scaled_normal_map == (xy != (1.0, 1.0)) and r.features.specular_aa
  return required(r.outputs.hdr).numpy()[0, 0, ..., :3].reshape(-1, 3).mean(0)


def test_toksvig_ignores_the_strength_of_an_unfiltered_map():
  """A uniform tilted map (no filtering spread) at strength 0.5: the scaled normal is short by
  design, not by filtering, so specular AA must not widen the roughness (it used to read the
  scaled vector: perceptual roughness 0.3 became about 0.57): it shades as the same normal
  stored at strength 1 (up to the texels' 8-bit rounding)."""
  tilt = np.array([math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0))])
  scaled = np.array([0.5 * tilt[0], 0.0, tilt[2]])
  np.testing.assert_allclose(
    _glossy_pixel(encode(tilt), 1.0, (0.5, 0.5)),
    _glossy_pixel(encode(scaled / math.hypot(*(float(x) for x in scaled))), 1.0, (1.0, 1.0)),
    rtol=0.06,
  )


def test_toksvig_keeps_the_filtered_spread_of_a_strong_map():
  """A minified 1-texel checker of +-30 degree normals at strength 2: the filtered texel is short
  (cos 30), so specular AA still widens the lobe (the scaled vector, longer than 1, used to hide
  the spread)."""
  tilt = np.array([[np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)], [-np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)]])
  texels = np.round((tilt + 1.0) / 2.0 * 255.0).astype(np.uint8)
  checker = texels[(np.arange(64)[:, None] + np.arange(64)[None, :]) % 2]
  wide = _glossy_pixel(checker, 32.0, (2.0, 2.0))
  sharp = _glossy_pixel(encode(np.array([0.0, 0.0, 1.0])), 1.0, (1.0, 1.0))  # the mean normal, unfiltered
  assert wide.mean() < 0.5 * sharp.mean()


def _features(host: rw.Scene, monkeypatch: pytest.MonkeyPatch, per_world: rw.PerWorld | None = None) -> Features:
  """The resolved Features of a CPU Renderer of `host` (no kernel compiled: Features only)."""
  monkeypatch.setattr(rw.Renderer, "_start_compiles", lambda _self: None)
  scene = rw.upload(host, "cpu")
  return rw.Renderer(scene, nworld=1, cameras=[rw.CameraSpec(fovy=10.0)], resolution=(4, 4), per_world=per_world).features


def test_per_world_texel_maps_compile_the_scaled_decode(monkeypatch: pytest.MonkeyPatch):
  """A per-world `material.tex_affine` may write any normal strength: its declaration compiles the
  scaled decode in, on a scene whose own maps would not need it."""
  host = quad(encode(TILT))
  assert not _features(host, monkeypatch).scaled_normal_map
  assert _features(host, monkeypatch, rw.PerWorld(material=["tex_affine"])).scaled_normal_map


def test_scaled_decode_reaches_the_module_hash(monkeypatch: pytest.MonkeyPatch):
  """Features.scaled_normal_map reaches the megakernel (its module hash changes with it), and is
  collapsed without normal maps."""
  plain = _features(quad(encode(TILT)), monkeypatch)
  scaled = dataclasses.replace(plain, scaled_normal_map=True)
  assert plain.normal_map and not plain.scaled_normal_map
  assert pixel.megakernel(plain, PINHOLE).module.hash_module() != pixel.megakernel(scaled, PINHOLE).module.hash_module()
  off = dataclasses.replace(scaled, normal_map=False)
  assert not canonical_lib.canonical(off).scaled_normal_map


@given(st.tuples(_FINITE, _FINITE), st.tuples(_FINITE, _FINITE))
def test_scale_xy_composes(a: tuple[float, float], b: tuple[float, float]):
  ref = rw.scene.TextureRef(TextureId(0), scale=normal_map.NORMAL_UNPACK[0], bias=normal_map.NORMAL_UNPACK[1])
  twice = normal_map.scale_xy(normal_map.scale_xy(ref, a), b)
  once = normal_map.scale_xy(ref, (a[0] * b[0], a[1] * b[1]))
  np.testing.assert_allclose(twice.scale, once.scale, rtol=1e-12)
  np.testing.assert_allclose(twice.bias, once.bias, rtol=1e-12)


@given(st.tuples(_FINITE, _FINITE), st.floats(0.0, 1.0))
def test_unpack_decodes_scaled_xy(xy: tuple[float, float], t: float):
  scale, bias = normal_map.unpack(xy)
  n = np.array(scale[:3]) * t + np.array(bias[:3])
  np.testing.assert_allclose(n, [xy[0] * (2 * t - 1), xy[1] * (2 * t - 1), 2 * t - 1], atol=1e-12)


@given(st.lists(_FINITE, min_size=6, max_size=6), st.lists(st.floats(0.0, 1.0), min_size=3, max_size=3))
def test_device_texel_map_decodes_the_canonical_normal(sb: list[float], t: list[float]):
  """The device's decode 2 v - 1 along (T, -B, N) of the green-inverted core texel through
  `normal_rows` is the binding's scale x texel + bias (any per-axis scale / bias, not clamped)."""
  scale, bias = np.array([*sb[:3], 1.0], np.float32), np.array([*sb[3:], 0.0], np.float32)
  rows = normal_rows(scale, bias).astype(np.float64)
  stored = np.array([t[0], 1.0 - t[1], t[2]])
  device = 2.0 * (rows[0, :3] * stored + rows[1, :3]) - 1.0
  canonical = scale[:3].astype(np.float64) * np.array(t) + bias[:3]
  np.testing.assert_allclose(device * np.array([1.0, -1.0, 1.0]), canonical, atol=1e-5)
  np.testing.assert_array_equal(rows[:, 3], TEX_IDENTITY[:, 3])


@pytest.mark.parametrize("k", [1.0, 0.5, 0.25, 0.7, 1.3])
def test_strength_texel_map_is_bit_identical_to_the_scalar_normal_scale(k: float):
  """A strength k (scale 2 k, bias -k: was `ext.normal_scale` k) gives the texel map the core used
  to build, bit for bit: [[k, k], [(1 - k) / 2] * 2] on x / y, identity on z."""
  scale, bias = normal_map.unpack((k, k))
  rows = normal_rows(np.array(scale, np.float32), np.array(bias, np.float32))
  ns = float(np.asarray(k, np.float32))
  old = TEX_IDENTITY.copy()
  old[:2, :2] = [[ns, ns], [0.5 * (1.0 - ns)] * 2]
  np.testing.assert_array_equal(rows, old)


def test_scaled_normal_maps_flags_strengths_and_offsets():
  def table(*xys: tuple[float, float]) -> np.ndarray:
    return np.stack([np.tile(TEX_IDENTITY, (NTEXROLE, 1, 1)) for _ in xys])

  xys = [(1.0, 1.0), (1.0, -1.0), (2.0, 2.0), (0.5, -1.0), (3.0, 3.0), (-1.0, 1.0)]
  affine = table(*xys)
  for i, xy in enumerate(xys):
    scale, bias = normal_map.unpack(xy)
    affine[i, TexRole.NORMAL] = normal_rows(np.array(scale, np.float32), np.array(bias, np.float32))
  texid = np.full((len(xys), NTEXROLE), -1, np.int32)
  texid[[0, 1, 2, 3, 5], TexRole.NORMAL] = 0  # material 4 has no normal map
  assert scaled_normal_maps(texid, affine).tolist() == [False, False, True, True, False, False]


def test_inward_normal_maps_are_rejected():
  """A binding whose decoded z is never positive (no outward normal) fails at the builder, on the
  slot and on an extra normal input."""
  b = rw.SceneBuilder()
  tex = b.add_texture(np.zeros((2, 2, 3), np.uint8), colorspace="linear")
  flat = rw.scene.TextureRef(tex, scale=(2.0, 2.0, 0.0, 1.0), bias=(-1.0, -1.0, 0.0, 0.0))
  with pytest.raises(ValueError, match=r"slot geometry_normal: a normal map's z"):
    b.add_material(rw.Material(textures=((rw.scene.Slot.GEOMETRY_NORMAL, flat),)))
  inward = rw.scene.TextureRef(tex, scale=(2.0, 2.0, -2.0, 1.0), bias=(-1.0, -1.0, 0.0, 0.0))
  extras = rw.scene.OpenPbrExtras(textures=(("geometry_coat_normal", inward),))
  with pytest.raises(ValueError, match=r"geometry_coat_normal: a normal map's z"):
    b.add_material(rw.Material(extras=extras))
  # Flipped z alone is fine as long as some texel points out (z = 1 - 2 t).
  b.add_material(
    rw.Material(textures=((rw.scene.Slot.GEOMETRY_NORMAL, rw.scene.TextureRef(tex, scale=(2, 2, -2, 1), bias=(-1, -1, 1, 0))),))
  )


def test_wpbr_round_trip_keeps_the_per_axis_transform(tmp_path):
  b = rw.SceneBuilder()
  tex = b.add_texture(encode(TILT), colorspace="linear")
  scale, bias = normal_map.unpack((0.5, -2.0))
  ref = rw.scene.TextureRef(tex, scale=scale, bias=bias)
  mat = b.add_material(rw.Material(textures=((rw.scene.Slot.GEOMETRY_NORMAL, ref),)))
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, np.array([[0, 1, 2], [0, 2, 3]]), uvs=UV_UP), material=mat)
  b.build().save(tmp_path / "n.wpbr")
  loaded = rw.Scene.load(tmp_path / "n.wpbr")
  assert wpbr.VERSION >= 9 and "ext_normal_scale" not in loaded.materials.SCHEMA
  np.testing.assert_array_equal(loaded.materials.tex_scale[0, rw.scene.Slot.GEOMETRY_NORMAL], scale)
  np.testing.assert_array_equal(loaded.materials.tex_bias[0, rw.scene.Slot.GEOMETRY_NORMAL], bias)
