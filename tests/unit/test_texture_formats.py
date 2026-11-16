"""GPU texture storage (`core/formats.py`, `core/pipeline/textures/uploader.py`): packed channels
(lossless), shared textures, 8-bit sRGB color (the default on CUDA), the resolution cap and the disk
cache of texture levels."""

from __future__ import annotations

import dataclasses
import gc

import numpy as np
import pytest
import warp as wp
from fixtures.textures import make_texture

import raywarp as rw
from raywarp._base.types import TEX_LAYOUT_IDENTITY, TEX_LAYOUT_ONE, TexRole
from raywarp.core.pipeline.textures import formats
from raywarp.core.pipeline.textures import uploader as textures
from raywarp.core.pipeline.textures.formats import DEFAULT_STORAGE, EXACT_STORAGE, TextureStorage, pack_channels
from raywarp.core.shade.materials import texture
from raywarp.scene.model.types import Slot, TextureRef

DEVICE = wp.get_device()
QUAD = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], np.float32)
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
QUAD_UV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)


def _fields(layout: int) -> list[int]:
  """The stored channel (or TEX_LAYOUT_ONE) of each read channel."""
  return [(layout >> (3 * k)) & 7 for k in range(4)]


def test_pack_channels_keeps_distinct_non_constant_channels():
  rng = np.random.default_rng(0)
  a, b = (rng.integers(0, 255, (8, 8), dtype=np.uint8) for _ in range(2))
  one = np.full((8, 8), 255, np.uint8)
  grey = pack_channels(np.stack([a, a, a, one], axis=-1))
  assert grey.kept == (0,) and _fields(grey.layout) == [0, 0, 0, TEX_LAYOUT_ONE]
  orm = pack_channels(np.stack([one, a, b, one], axis=-1))  # no occlusion: roughness, metallic
  assert orm.kept == (1, 2) and _fields(orm.layout) == [TEX_LAYOUT_ONE, 0, 1, TEX_LAYOUT_ONE]
  rgb = pack_channels(np.stack([a, b, a ^ b, one], axis=-1))  # 3 channels: padded to 4
  assert rgb.kept == (0, 1, 2, 3) and rgb.layout == TEX_LAYOUT_IDENTITY - (3 << 9) + (TEX_LAYOUT_ONE << 9)
  ones = pack_channels(np.stack([one] * 4, axis=-1))
  assert ones.kept == (0,) and _fields(ones.layout) == [0, TEX_LAYOUT_ONE, TEX_LAYOUT_ONE, TEX_LAYOUT_ONE]
  full = pack_channels(np.stack([a, b, a ^ b, b ^ 7], axis=-1))
  assert full.kept == (0, 1, 2, 3) and full.layout == TEX_LAYOUT_IDENTITY


@wp.kernel
def _sample_many(tex: wp.Texture2D, layout: int, uv: wp.array1d[wp.vec2], lod: wp.array1d[float], out: wp.array1d[wp.vec4]):
  i = wp.tid()
  out[i] = texture.sample_layout(tex, layout, uv[i], lod[i])


def _samples(tex: wp.Texture2D, layout: int, n: int = 256) -> np.ndarray:
  """Texels of `tex` read through `layout` at `n` random (uv, lod), the same points every call."""
  rng = np.random.default_rng(3)
  uv = wp.array(rng.uniform(-0.5, 1.5, (n, 2)).astype(np.float32), dtype=wp.vec2, device=DEVICE)
  lod = wp.array(rng.uniform(0.0, 4.0, n).astype(np.float32), dtype=float, device=DEVICE)
  out = wp.zeros(n, dtype=wp.vec4, device=DEVICE)
  wp.launch(_sample_many, dim=n, inputs=[tex, layout, uv, lod, out], device=DEVICE)
  return out.numpy()


def _rgba_texture(rgba: np.ndarray, srgb: bool) -> wp.Texture2D:
  """Today's 4-channel texture of `rgba` (linear float16 for sRGB, else as stored)."""
  texels = rgba
  if srgb:
    texels = np.concatenate([formats.SRGB_F16[rgba[..., :3]], formats.UNIT_F16[rgba[..., 3:]]], axis=-1)
  wrap = wp.TextureAddressMode.WRAP
  return textures.MipTexture2D(textures.box_chain(texels), address=(wrap, wrap), device=wp.get_device())


_KINDS = ["grey", "orm", "opacity", "grey_alpha", "alpha_is_red", "grey_alpha_is_grey"]
# Stored channels of each kind: data / sRGB color (alpha never copies an sRGB color channel).
_CHANNELS = {"grey": (1, 1), "orm": (2, 2), "opacity": (1, 1), "grey_alpha": (2, 2), "alpha_is_red": (4, 4)}
_CHANNELS["grey_alpha_is_grey"] = (1, 2)


def _kind(kind: str) -> np.ndarray:
  """The (16, 32, 4) uint8 test image of `kind`."""
  rng = np.random.default_rng(1)
  a, b, c = (rng.integers(0, 255, (16, 32), dtype=np.uint8) for _ in range(3))
  one = np.full_like(a, 255)
  return {
    "grey": np.stack([a, a, a, one], -1),
    "orm": np.stack([one, a, b, one], -1),
    "opacity": np.stack([a, one, one, one], -1),
    "grey_alpha": np.stack([a, a, a, b], -1),
    "alpha_is_red": np.stack([a, b, c, a], -1),
    "grey_alpha_is_grey": np.stack([a, a, a, a], -1),
  }[kind]


@pytest.mark.parametrize("srgb", [False, True])
@pytest.mark.parametrize("kind", _KINDS)
def test_packed_texture_samples_like_the_rgba_texture(kind, srgb):
  """A packed texture (fewer channels) reads bit for bit as the 4-channel one it replaces; alpha
  copies a color channel only when both are stored alike (not sRGB)."""
  rgba = _kind(kind)
  core = make_texture(rgba, srgb=srgb, storage=EXACT_STORAGE, device=wp.get_device())
  assert core.texture.num_channels == _CHANNELS[kind][int(srgb)]
  np.testing.assert_array_equal(_samples(core.texture, core.layout), _samples(_rgba_texture(rgba, srgb), TEX_LAYOUT_IDENTITY))


@pytest.mark.parametrize("kind", ["alpha_is_red", "grey_alpha_is_grey", "grey"])
def test_packed_coverage_texture_samples_like_the_rgba_texture(kind):
  """With coverage-weighted color mips (alpha box filtered) alpha never copies a color channel."""
  rgba = _kind(kind)
  coverage = (np.random.default_rng(2).random(rgba.shape[:2]) * (np.arange(rgba.shape[1]) % 3 != 0)).astype(np.float32)
  core = make_texture(rgba, srgb=False, coverage=coverage, storage=EXACT_STORAGE, device=wp.get_device())
  wrap = (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.WRAP)
  base = np.ascontiguousarray(rgba)
  shapes = formats.level_shapes(base.shape[1], base.shape[0])
  ref = textures.MipTexture2D(formats.coverage_mip_chain(base, coverage, shapes), address=wrap, device=wp.get_device())
  np.testing.assert_array_equal(_samples(core.texture, core.layout), _samples(ref, TEX_LAYOUT_IDENTITY))


def _quad_scene(images: list[np.ndarray]) -> rw.Scene:
  """One quad per image, each with its own host texture as its base color."""
  b = rw.SceneBuilder()
  mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  for img in images:
    mat = rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0).with_texture(
      Slot.BASE_COLOR, TextureRef(b.add_texture(img))
    )
    b.add_instance("mesh", mesh=mesh, material=b.add_material(mat))
  return b.build()


def test_equal_texels_share_one_gpu_texture():
  """Two host textures of equal texels (two files of one image) are one GPU texture."""
  img = np.random.default_rng(2).integers(0, 256, (8, 8, 4), dtype=np.uint8)
  with wp.ScopedDevice(DEVICE):
    gpu = rw.upload(_quad_scene([img, img.copy()]), options=rw.SceneOptions(textures=EXACT_STORAGE))
  texid = gpu.tables["material.texture"].numpy()[0, :, int(TexRole.RGB)]
  assert texid[0] != texid[1]  # two recipes (two host textures) ...
  assert gpu.textures[int(texid[0])] is gpu.textures[int(texid[1])]  # ... one texture


@pytest.mark.parametrize("srgb", [False, True])
@pytest.mark.parametrize("value", [(200, 120, 40, 255), (90, 90, 90, 255), (0, 0, 0, 0)])
def test_uniform_texture_is_stored_1x1_and_samples_alike(value, srgb):
  """A texture whose every texel is one value (`pool._uniform`) is stored 1 x 1 in its
  format and layout, keeps the size it is sampled at, and reads as the full one at any uv / lod
  (CUDA's fixed-point filter weights: bit for bit; Warp's CPU blend v (1 - f) + v f rounds in the
  last bit, differently per weight: within 2 ulp); equal values share it; a BORDER-addressed one
  stays full size (its edge blend)."""
  from raywarp.core.pipeline.textures import pool

  rgba = np.empty((16, 32, 4), np.uint8)
  rgba[:] = value
  host = textures.host_texture(rgba, srgb=srgb, device=wp.get_device())
  folded = pool._uniform(host)
  assert folded is not None and folded.texels.shape[:2] == (1, 1) and folded.layout == host.layout
  assert folded.texels.dtype == host.texels.dtype and folded.texels.srgb8 == host.texels.srgb8
  uploader = textures.TextureUploader(wp.get_device())
  full, one = uploader.add(host), uploader.add(folded)
  uploader.finish()
  assert one.texture.width == 1 and full.texture.width == 32
  got, want = _samples(one.texture, one.layout), _samples(full.texture, full.layout)
  np.testing.assert_allclose(got, want, rtol=0.0 if DEVICE.is_cuda else 2.5e-7, atol=0.0)
  border = wp.TextureAddressMode.BORDER
  assert pool._uniform(textures.host_texture(rgba, srgb=srgb, address=(border, border), device=DEVICE)) is None
  rgba[3, 5, 0] ^= 1
  assert pool._uniform(textures.host_texture(rgba, srgb=srgb, device=DEVICE)) is None
  # Through the pool: two images of one value are one 1 x 1 texture sampled at their own sizes.
  with wp.ScopedDevice(DEVICE):
    gpu = rw.upload(_quad_scene([np.full((8, 8, 4), 77, np.uint8), np.full((4, 16, 4), 77, np.uint8)]))
  texid = gpu.tables["material.texture"].numpy()[0, :, int(TexRole.RGB)]
  assert gpu.textures[int(texid[0])] is gpu.textures[int(texid[1])] and gpu.textures[int(texid[0])].width == 1
  assert gpu.tex_size.numpy()[texid].tolist() == [[8.0, 8.0], [16.0, 4.0]]


@pytest.mark.parametrize("coverage", [False, True])
def test_grey_srgb8_color_is_one_channel_with_the_rgba_levels(coverage):
  """An sRGB color image of one stored channel without alpha (grey) keeps one 8-bit channel with
  the texture unit's sRGB decode: its levels are channel 0 of the RGBA chain (sRGB-linear
  filtering per channel); on CUDA it reads bit for bit as the RGBA texture."""
  rng = np.random.default_rng(9)
  a = rng.integers(0, 256, (16, 32), dtype=np.uint8)
  rgba = np.stack([a, a, a, np.full_like(a, 255)], -1)
  weights = (rng.random(a.shape) * (np.arange(32) % 3 != 0)).astype(np.float32) if coverage else None
  packed = pack_channels(rgba, alpha_apart=True)
  assert packed.kept == (0,)
  one, _ = textures._host_levels(rgba[..., :1].copy(), srgb=True, srgb8=True, ncolor=1, coverage=weights)
  four, _ = textures._host_levels(rgba, srgb=True, srgb8=True, ncolor=3, coverage=weights)
  for lv1, lv4 in zip(one, four, strict=True):
    np.testing.assert_array_equal(lv1[..., 0], lv4[..., 0])
  if not DEVICE.is_cuda:
    return
  core = make_texture(rgba, srgb=True, coverage=weights, device=DEVICE)
  assert core.texture.num_channels == 1 and core.texture.dtype == wp.uint8
  wrap = wp.TextureAddressMode.WRAP
  ref = textures.MipTexture2D(four, address=(wrap, wrap), device=DEVICE, srgb8=True)
  np.testing.assert_array_equal(_samples(core.texture, core.layout), _samples(ref, TEX_LAYOUT_IDENTITY))


def test_srgb8_color_is_8bit_with_linear_mips():
  """storage.color "srgb8" (the default): 8-bit sRGB texels decoded before filtering (the texture
  unit on CUDA; elsewhere the exact float16 storage); the mips are filtered in linear values (a
  black / white checker's 1 x 1 mip is linear 0.5, not sRGB 0.5)."""
  rgba = np.full((2, 2, 4), 255, np.uint8)
  rgba[0, 0, :3] = rgba[1, 1, :3] = 0
  core = make_texture(rgba, srgb=True, device=wp.get_device())
  assert core.texture.dtype == (wp.uint8 if DEVICE.is_cuda else wp.float16)
  assert textures.host_texture(rgba, srgb=True, device=wp.get_device()).texels.srgb8 == DEVICE.is_cuda
  exact = make_texture(rgba, srgb=True, storage=EXACT_STORAGE, device=wp.get_device())
  out = wp.zeros(2, dtype=wp.vec4, device=DEVICE)
  uv = wp.array([wp.vec2(0.25, 0.25), wp.vec2(0.5, 0.5)], dtype=wp.vec2, device=DEVICE)
  lod = wp.array([0.0, 1.0], dtype=float, device=DEVICE)
  for packed in (core, exact):
    wp.launch(_sample_many, dim=2, inputs=[packed.texture, packed.layout, uv, lod, out], device=DEVICE)
    texel = out.numpy()
    np.testing.assert_allclose(texel[0, :3], 0.0, atol=1e-6)  # a black texel centre
    np.testing.assert_allclose(texel[1, :3], 0.5, atol=4e-3)  # the 1 x 1 mip: linear average (not 0.21)


def test_max_size_halves_large_textures():
  """storage.max_size: a 64 x 32 texture capped at 16 is stored 16 x 8, as its mip level 2."""
  rng = np.random.default_rng(4)
  rgba = rng.integers(0, 256, (32, 64, 4), dtype=np.uint8)
  cap = dataclasses.replace(EXACT_STORAGE, max_size=16)
  capped = make_texture(rgba, srgb=True, storage=cap, device=wp.get_device())
  assert capped.size == (16.0, 8.0) and capped.texture.width == 16 and capped.texture.dtype == wp.float16
  exact = make_texture(rgba, srgb=True, storage=EXACT_STORAGE, device=wp.get_device())
  out = wp.zeros(4, dtype=wp.vec4, device=DEVICE)
  uv = wp.array(rng.uniform(0.0, 1.0, (4, 2)).astype(np.float32), dtype=wp.vec2, device=DEVICE)
  got = []
  for core, lod in ((capped, 0.0), (exact, 2.0)):
    lods = wp.full(4, lod, dtype=float, device=DEVICE)
    wp.launch(_sample_many, dim=4, inputs=[core.texture, core.layout, uv, lods, out], device=DEVICE)
    got.append(out.numpy())
  np.testing.assert_allclose(got[0], got[1], atol=2e-3)


@pytest.mark.parametrize("kind", ["orm", "opacity", "ones"])
def test_packed_black_border_texture_reads_zero_outside(kind):
  """BLACK wrap (BORDER): outside [0, 1] and across the edge texels' blend every channel fades to
  0 as in the 4-channel texture, the constant-1 channels included (they are stored there)."""
  rng = np.random.default_rng(5)
  a, b = (rng.integers(0, 255, (16, 32), dtype=np.uint8) for _ in range(2))
  one = np.full_like(a, 255)
  rgba = {
    "orm": np.stack([one, a, b, one], -1),
    "opacity": np.stack([a, one, one, one], -1),
    "ones": np.stack([one] * 4, -1),
  }[kind]
  border = wp.TextureAddressMode.BORDER
  core = make_texture(rgba, srgb=False, address=(border, border), storage=EXACT_STORAGE, device=wp.get_device())
  ref = textures.MipTexture2D(textures.box_chain(rgba), address=(border, border), device=wp.get_device())
  got, want = _samples(core.texture, core.layout), _samples(ref, TEX_LAYOUT_IDENTITY)
  assert (want == 0.0).any()  # some samples fall outside
  np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
  "bad",
  [
    {"color": "bc7"},
    {"color": "exact"},
    {"max_size": -1},
    {"max_size": 2.5},
    {"max_size": True},
  ],
)
def test_texture_storage_rejects_unknown_options(bad):
  with pytest.raises(ValueError, match="TextureStorage"):
    TextureStorage(**bad)


def test_texture_storage_is_not_top_level():
  assert "TextureStorage" not in vars(rw)
  assert TextureStorage(max_size=1000).max_size == 1000  # any size >= 1, not only powers of two


def test_default_storage_is_exact_without_the_srgb_decode():
  """The default stores 8-bit sRGB color where the texture unit decodes it (CUDA); elsewhere
  color is float16 (exact); `max_size` is kept."""
  default = TextureStorage(max_size=8)
  assert default.color == "srgb8"
  assert default.on(srgb8=True) is default
  assert default.on(srgb8=False) == dataclasses.replace(EXACT_STORAGE, max_size=8)
  rgba = np.random.default_rng(3).integers(0, 256, (8, 8, 4), dtype=np.uint8)
  cpu = make_texture(rgba, srgb=True, device=wp.get_device("cpu"))
  assert cpu.texture.dtype == wp.float16


def test_equal_texels_with_other_wrap_modes_stay_apart():
  """Equal texels sampled with different address modes are two GPU textures."""
  img = np.random.default_rng(6).integers(0, 256, (8, 8, 4), dtype=np.uint8)
  b = rw.SceneBuilder()
  mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  for wrap in ("repeat", "clamp"):
    tex = b.add_texture(img.copy(), wrap=(wrap, wrap))
    mat = rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0).with_texture(Slot.BASE_COLOR, TextureRef(tex))
    b.add_instance("mesh", mesh=mesh, material=b.add_material(mat))
  with wp.ScopedDevice(DEVICE):
    gpu = rw.upload(b.build())
  texid = gpu.tables["material.texture"].numpy()[0, :, int(TexRole.RGB)]
  assert gpu.textures[int(texid[0])] is not gpu.textures[int(texid[1])]


@pytest.mark.parametrize("shape", [(12, 20), (1, 1), (1, 7)])
@pytest.mark.parametrize(
  ("srgb", "coverage", "storage"),
  [(True, False, None), (True, True, None), (False, False, None), (False, True, None), (True, False, "exact")],
)
def test_texture_levels_round_trip_through_the_disk_cache(tmp_path, srgb, coverage, storage, shape):
  """`write_entry` / `read_entry`: a texture rebuilt from its cached host levels (8-bit sRGB with
  linear mips, coverage-weighted, box filtered on the device, exact color) samples exactly as the
  one built from the texels, at every level, 1 x 1 and 1 x N images included (size-1 axes keep
  odd NumPy strides); the entry keeps the caller's fields; a missing texel file is a miss."""
  rng = np.random.default_rng(8)
  rgba = rng.integers(0, 256, (*shape, 4), dtype=np.uint8)
  weights = rng.random(shape).astype(np.float32) if coverage else None
  kept = EXACT_STORAGE if storage == "exact" else DEFAULT_STORAGE
  cache = tmp_path / "textures"
  host = textures.host_texture(rgba, srgb=srgb, coverage=weights, storage=kept, device=DEVICE)
  textures.write_entry(cache / "e.json", host, {"note": [1, 2]})
  found = textures.read_entry(cache / "e.json")
  assert found is not None
  read, extra = found
  assert extra == {"note": [1, 2]} and read.texels == host.texels and read.layout == host.layout
  # Two uploaders: the second is not deduplicated against the first.
  built, again = textures.TextureUploader(DEVICE).add(host), textures.TextureUploader(DEVICE).add(read)
  assert (again.layout, again.size, again.texture.dtype) == (built.layout, built.size, built.texture.dtype)
  uv = wp.array(rng.uniform(-0.5, 1.5, (64, 2)).astype(np.float32), dtype=wp.vec2, device=DEVICE)
  out = wp.zeros(64, dtype=wp.vec4, device=DEVICE)
  for lod in (0.0, 1.5, 4.0):
    lods = wp.full(64, lod, dtype=float, device=DEVICE)
    got = []
    for core in (built, again):
      wp.launch(_sample_many, dim=64, inputs=[core.texture, core.layout, uv, lods, out], device=DEVICE)
      got.append(out.numpy())
    np.testing.assert_array_equal(got[0], got[1])
  for f in cache.glob("*.npy"):
    f.unlink()
  assert textures.read_entry(cache / "e.json") is None


def test_second_pool_upload_reads_the_cached_levels(tmp_path, monkeypatch):
  """A second upload of the same images builds every pool texture from its cached host levels
  (no texel read, pack or mip filter): the same layouts, sizes, digests and sampled texels."""
  from raywarp.core.pipeline.platform import bake_cache
  from raywarp.core.pipeline.textures import pool

  monkeypatch.setenv("RAYWARP_CACHE_DIR", str(tmp_path))
  rng = np.random.default_rng(9)
  color = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
  orm = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
  b = rw.SceneBuilder()
  base, data = b.add_texture(color), b.add_texture(orm, colorspace="data")
  mat = rw.Material(
    textures=(
      (Slot.BASE_COLOR, TextureRef(base)),
      (Slot.SPECULAR_ROUGHNESS, TextureRef(data, channel=1)),
      (Slot.GEOMETRY_NORMAL, TextureRef(data)),
    )
  )
  b.add_instance("mesh", mesh=b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV), material=b.add_material(mat))
  scene = b.build()

  def upload() -> tuple[rw.GpuScene[None], np.ndarray]:
    with wp.ScopedDevice(DEVICE):
      gpu = rw.upload(scene)
    uv = wp.array(rng.uniform(0.0, 1.0, (32, 2)).astype(np.float32), dtype=wp.vec2, device=DEVICE)
    out, lods = wp.zeros(32, dtype=wp.vec4, device=DEVICE), wp.full(32, 0.7, dtype=float, device=DEVICE)
    layouts = gpu.tex_layout.numpy()
    texels = []
    for i, tex in enumerate(gpu.textures):
      wp.launch(_sample_many, dim=32, inputs=[tex, int(layouts[i]), uv, lods, out], device=DEVICE)
      texels.append(out.numpy())
    return gpu, np.stack(texels)

  rng_state = rng.bit_generator.state
  first, first_texels = upload()
  assert list(tmp_path.glob("textures/pool-*.json")) and list(tmp_path.glob("textures/*.npy"))
  monkeypatch.setattr(pool.TexturePool, "_texels", lambda *a: pytest.fail("texels read on a warm upload"))
  monkeypatch.setattr(textures, "pack_channels", lambda *a, **k: pytest.fail("texels packed on a warm upload"))
  rng.bit_generator.state = rng_state
  again, again_texels = upload()
  assert again.tex_layout.numpy().tolist() == first.tex_layout.numpy().tolist()
  assert again.tex_size.numpy().tolist() == first.tex_size.numpy().tolist()
  assert [bake_cache._digests.get(t) for t in again.textures] == [bake_cache._digests.get(t) for t in first.textures]
  np.testing.assert_array_equal(again_texels, first_texels)


CUDA = pytest.mark.skipif(not DEVICE.is_cuda, reason="the texture unit's sRGB decode needs CUDA")
ADDRESS = [wp.TextureAddressMode.WRAP, wp.TextureAddressMode.CLAMP, wp.TextureAddressMode.MIRROR, wp.TextureAddressMode.BORDER]


def _display(linear: np.ndarray) -> np.ndarray:
  """sRGB display values [0, 1] of linear ones."""
  x = np.clip(linear, 0.0, 1.0)
  return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


@CUDA
@pytest.mark.parametrize("mode", ADDRESS)
@pytest.mark.parametrize("shape", [(18, 30), (1, 1)])
def test_srgb8_samples_like_its_linear_float16_levels(mode, shape):
  """The texture unit's sRGB decode and filter equal a float16 texture of the same levels
  decoded per texel (as the hardware decodes before filtering) within 3.5 display levels on a
  smooth image (the hardware filters with 8-bit weights: on noise their error, steepened by the
  sRGB curve near black, reaches ~6 levels), bilinear and trilinear, per address mode,
  mipmapped and single-level."""
  y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
  smooth = np.stack([128 + 100 * np.sin(x / 4.0), 128 + 90 * np.cos(y / 3.0), 40 + 4 * x + y, 255 - 6 * y], -1)
  rgba = (smooth + np.random.default_rng(10).normal(0, 3, smooth.shape)).clip(0, 255).astype(np.uint8)
  host = textures.host_texture(rgba, srgb=True, address=(mode, mode), device=DEVICE)
  core = textures.TextureUploader(DEVICE).add(host)
  assert host.texels.srgb8 and core.texture.dtype == wp.uint8
  shapes = formats.level_shapes(shape[1], shape[0])
  levels = formats.srgb8_mip_chain(rgba, None, shapes, ncolor=3, nsrgb=3)
  linear = [formats.linear_f16(lv, 3) for lv in levels]
  ref = textures.MipTexture2D(linear, address=(mode, mode), device=wp.get_device("cpu"))
  uv = np.random.default_rng(11).uniform(-1.5, 2.5, (256, 2)).astype(np.float32)
  for lod in (0.0, 0.5, 2.25, 9.0):
    got, want = [], []
    for tex, dev, out in ((core.texture, DEVICE, got), (ref, wp.get_device("cpu"), want)):
      res = wp.zeros(256, dtype=wp.vec4, device=dev)
      uvs = wp.array(uv, dtype=wp.vec2, device=dev)
      wp.launch(
        _sample_many, dim=256, inputs=[tex, core.layout, uvs, wp.full(256, lod, dtype=float, device=dev), res], device=dev
      )
      out.append(res.numpy())
    diff = np.abs(_display(got[0][:, :3]) - _display(want[0][:, :3])) * 255.0
    assert diff.max() <= 3.5, (lod, diff.max())
    np.testing.assert_allclose(got[0][:, 3], want[0][:, 3], atol=1.5 / 255)


@CUDA
def test_srgb8_textures_are_freed():
  """Dropping an 8-bit sRGB texture frees its CUDA array (Warp's destructor frees the replaced
  texture object's array and the sRGB one)."""
  rgba = np.random.default_rng(7).integers(0, 256, (1024, 1024, 4), dtype=np.uint8)
  nbytes = rgba.nbytes * 4 // 3
  wp.synchronize_device(DEVICE)
  free = DEVICE.free_memory
  texs = [make_texture(rgba, srgb=True, address=(m, m), device=DEVICE) for m in ADDRESS * 8]
  wp.synchronize_device(DEVICE)
  assert free - DEVICE.free_memory >= 32 * nbytes * 0.9
  del texs
  gc.collect()
  wp.synchronize_device(DEVICE)
  assert free - DEVICE.free_memory < 4 * nbytes


@CUDA
def test_parallel_uploads_keep_the_device_stream():
  """Building textures on the upload's thread pool (sRGB texture objects included) leaves the
  device's current stream and default device as they were."""
  rng = np.random.default_rng(6)
  b = rw.SceneBuilder()
  mesh = b.add_mesh(QUAD, QUAD_TRIS, uvs=QUAD_UV)
  for _ in range(16):
    mat = rw.Material(base_color=(1.0, 1.0, 1.0), specular_roughness=1.0).with_texture(
      Slot.BASE_COLOR, TextureRef(b.add_texture(rng.integers(0, 256, (32, 32, 4), dtype=np.uint8)))
    )
    b.add_instance("mesh", mesh=mesh, material=b.add_material(mat))
  with wp.ScopedDevice(DEVICE):
    stream = wp.get_stream()
    gpu = rw.upload(b.build())
    assert wp.get_stream() is stream and wp.get_device() == DEVICE
  assert sum(t.dtype == wp.uint8 and t.num_channels == 4 for t in gpu.textures[1:]) == 16


def test_a_failed_upload_unblocks_the_readers_waiting_for_slots() -> None:
  """When the consumer of `as_done` fails, the readers blocked on slots that only unconsumed
  results hold are woken (`abort`: `Staging.close`) instead of hanging the pool's join."""
  import queue
  import threading

  slots: queue.SimpleQueue[int] = queue.SimpleQueue()
  slots.put(0)  # one slot for many readers: every reader but the first waits for it

  def read(i: int) -> int:
    if slots.get() < 0:
      slots.put(-1)
      raise RuntimeError("closed")
    return i  # holds the slot until the consumer releases it (it never does)

  def consume() -> None:
    with pytest.raises(ValueError, match="upload failed"):
      for _ in textures.as_done(read, range(8), 4, abort=lambda: slots.put(-1)):
        raise ValueError("upload failed")

  t = threading.Thread(target=consume, daemon=True)
  t.start()
  t.join(timeout=20.0)
  assert not t.is_alive(), "as_done hung after its consumer failed"
