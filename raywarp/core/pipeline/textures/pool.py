"""Core textures built from host textures by channel recipes (`TexturePool`).

A recipe gives each of a core texture's four channels as one host texture's channel, optionally
inverted (1 - texel), or the constant 1 (host texture `NO_TEXTURE`): material slots are packed (ORM)
and flipped vertically (host UVs have their origin at the image's bottom-left, the core's at the
top-left) on the host. Recipes only select channels: per-material factors (tints, UsdUVTexture scale
/ bias, occlusion strength, normal scale, emissive color) are the material's texel maps
(`material.tex_affine`, applied where the texture is sampled), so one image shared by materials of
different tints is one core texture. Recipes are deduplicated by every input of the texels and built
on a thread pool; core textures of equal stored texels share one GPU texture (each core id keeps its
own texel layout, `formats.pack_channels`).

A core texture whose every texel of every mip level is one value (`_uniform`; not BORDER-addressed:
its border blend depends on the size) is stored as a 1 x 1 texture of that texel, same format and
layout, keeping the size it is sampled at (`sizes`: its mip level and anisotropic taps unchanged):
the texture unit returns the value either way, and equal values share one GPU texture.

A core texture is remembered on disk by the digest of its every input (`_entry_key`: the source
images' content digests `Images.digest`, channels, color spaces, wraps, request flags, storage,
the code that makes it): `<cache dir>/textures/pool-<key>.json` (`uploader.write_entry`) holds its
layout, red range, conversion notes and names its host levels' file, so a repeat upload reads the
levels (`texture_cache.read_entry`, then `texture_cache.load_levels` or `uploader.read_staged`)
without reading, converting, packing, filtering or hashing any texel.

Color spaces: color recipes decode sRGB sources (`host_texture` for 8-bit ones), read LIN_REC709
/ DATA sources as linear, and convert sources in any other space per texel at upload
(`colorspace.linear_rec709`, negatives clamped to 0): 8-bit ones are re-encoded as 8-bit sRGB
(stored and decoded as SRGB_REC709 ones, same memory; components above 1 clipped), float ones kept
linear. Emission recipes (`hdr`) of 8-bit sources whose converted values exceed 1 keep them as
linear float instead (the fitted emissive lights read the same values). Data recipes read raw
channels whatever the space. Host textures of one image and wrap in LIN_REC709 and DATA (stored
identically) are one source, so their recipes share a core texture.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import os
import sys
import warnings
from pathlib import Path
from typing import Final, NamedTuple, final

import numpy as np
import warp as wp

from raywarp._base import types as core_types
from raywarp._base.types import TEX_LAYOUT_IDENTITY
from raywarp.cache.code_version import code_digest
from raywarp.cache.paths import cache_dir
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.pipeline.textures import cache as texture_cache
from raywarp.core.pipeline.textures import formats, uploader
from raywarp.core.pipeline.textures.formats import DEFAULT_STORAGE, TextureStorage
from raywarp.core.pipeline.textures.uploader import (
  CoreTexture,
  HostTexture,
  Staging,
  TextureUploader,
  as_done,
  host_texture,
  placeholder_texture,
  read_staged,
  write_entry,
)
from raywarp.core.post.grading import color as shade_color
from raywarp.core.post.grading.color import srgb_to_linear
from raywarp.scene.images import colorspace, mips
from raywarp.scene.images.colorspace import ColorSpace, stored_as_is
from raywarp.scene.images.mips import linear_to_srgb_u8
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Scene
from raywarp.scene.model.types import NO_TEXTURE, Array, Json, PixelType, Texels, TextureId, Wrap

# Stored channel counts of host images: grey + alpha, RGBA (grey alone is 1).
_GREY_ALPHA: Final = 2
_RGBA: Final = 4
_ADDRESS: Final = {
  int(Wrap.REPEAT): wp.TextureAddressMode.WRAP,
  int(Wrap.CLAMP): wp.TextureAddressMode.CLAMP,
  int(Wrap.MIRROR): wp.TextureAddressMode.MIRROR,
  int(Wrap.BLACK): wp.TextureAddressMode.BORDER,
}


class Channel(NamedTuple):
  """A core texture channel: host texture `texture`'s channel `channel` (1 - it when `invert`).

  Host texture `NO_TEXTURE` is the constant 1 (`ONE`).
  """

  texture: TextureId
  channel: int
  invert: bool = False


class Coverage(NamedTuple):
  """The opacity weighting a color texture's mips: clip(scale * channel + bias, 0, 1)."""

  channel: Channel
  scale: float
  bias: float


# Float32 texels of any shape (`_unit_float`, `_resampled`).
type _Floats = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
# The four channels of a core texture.
type Recipe = tuple[Channel, Channel, Channel, Channel]
# A core texture's (u, v) address modes.
type Address = tuple[wp.TextureAddressMode, wp.TextureAddressMode]

# The constant-one channel.
ONE: Final = Channel(NO_TEXTURE, 0)
# Color spaces whose sources are stored alike (linear): one source for the recipes.
_LINEAR_STORAGE: Final = frozenset({ColorSpace.LIN_REC709, ColorSpace.DATA})


def _rgba_view(scene: Scene, texture: TextureId) -> tuple[Texels, ColorSpace]:
  """(level-0 image of a host texture as (h, w, 4) in its stored dtype, color space); images are
  stored with 1 (grey), 2 (grey, alpha) or 4 channels."""
  image = int(scene.textures.image[texture])
  img = scene.images.level(image, 0)
  c = img.shape[2]
  if c == _RGBA:
    rgba = img
  else:
    one = np.array(255 if img.dtype == np.uint8 else 1.0, img.dtype)
    alpha = img[..., 1:2] if c == _GREY_ALPHA else np.broadcast_to(one, (*img.shape[:2], 1))
    rgba = np.concatenate([np.repeat(img[..., :1], 3, axis=2), alpha], axis=2)
  return rgba, colorspace.from_code(int(scene.textures.colorspace[texture]))


class Converted(NamedTuple):
  """A color source converted to Rec.709 at upload.

  Attributes:
    texels: (h, w, 4) 8-bit sRGB or float32 linear RGBA.
    space: their space (SRGB_REC709 or LIN_REC709).
    negative: colors left the gamut below 0 (clamped to 0).
    clipped: 8-bit sRGB storage clipped components above 1.
  """

  texels: Texels
  space: ColorSpace
  negative: bool
  clipped: bool


def image_digest(scene: Scene, image: int) -> bytes:
  """The content digest of host image `image` (`Images.digest`).

  Raises:
    ValueError: for an image without a content digest (every `SceneBuilder` scene has them).
  """
  digest = np.asarray(scene.images.digest[image])
  if not digest.any():
    raise ValueError(f"image {image} has no content digest (Images.digest): a scene not made by SceneBuilder")
  return digest.tobytes()


def converted_view(texels: Texels, space: ColorSpace, *, hdr: bool = False) -> Converted:
  """(h, w, 4) texels of a color image not stored as is (`colorspace.stored_as_is`), converted to
  Rec.709: 8-bit sources re-encoded as 8-bit sRGB (alpha bytes kept), float ones linear float32.
  `hdr`: 8-bit sources whose converted values exceed 1 stay linear float32 (emission)."""
  linear = colorspace.linear_rec709(texels, space)
  rgb = linear[..., :3]
  negative = bool((rgb < 0.0).any())
  np.maximum(rgb, 0.0, out=rgb)
  above = bool((rgb > 1.0).any())
  if texels.dtype != np.uint8 or (hdr and above):
    return Converted(linear, ColorSpace.LIN_REC709, negative, clipped=False)
  out = np.empty(texels.shape, np.uint8)
  out[..., :3] = linear_to_srgb_u8(rgb)
  out[..., 3] = texels[..., 3]
  return Converted(out, ColorSpace.SRGB_REC709, negative, clipped=above)


@dataclasses.dataclass
class _Request:
  """A core texture to build.

  Attributes:
    recipe: its four channels.
    color: linearise sRGB sources, convert the other color spaces.
    red_range: record the smallest and largest red texels.
    coverage: the opacity weighting the color mips, or None.
    hdr: converted 8-bit sources keep values above 1 (linear float; emission).
  """

  recipe: Recipe
  color: bool
  red_range: bool
  coverage: Coverage | None = None
  hdr: bool = False


# Pinned staging slots of a warm upload (`TexturePool.finish`): reads run at ~2 GB/s per thread
# into them, so a few keep the one uploading thread fed; each slot costs pinned memory
# (allocation ~0.3 s per GB).
_STAGING_SLOTS: Final = 8
# Threads reading or building the host side of the pool's textures (at most the CPU count).
_MAX_THREADS: Final = 32
# Bumped when a pool entry's content changes (3: uniform textures stored 1 x 1, their size kept).
_ENTRY_FORMAT: Final = 3
# Values of an entry's flag pair (negative, clipped) and of its red range (smallest, largest).
_PAIR: Final = 2


def _entry_code() -> str:
  """`code_digest` of the modules deciding what a request becomes (texels, color conversions, mip
  filter, formats, the texel layout codes)."""
  modules = (sys.modules[__name__], uploader, formats, shade_color, colorspace, mips, core_types)
  return code_digest((f"{m.__name__}.py", Path(m.__file__).read_bytes()) for m in modules if m.__file__ is not None)


_ENTRY_CODE: Final = _entry_code()


@final
class _Entry(NamedTuple):
  """What a pool entry remembers besides its texture (`uploader.write_entry`'s extra fields).

  Attributes:
    red: (smallest, largest) red texel, when the request records it.
    converted: per converted source (`_conversions` order): (negative, clipped) of `Converted`.
    size: the (width, height) a uniform texture stored 1 x 1 is sampled at (`_uniform`), else None.
  """

  red: tuple[float, float] | None
  converted: tuple[tuple[bool, bool], ...]
  size: tuple[float, float] | None = None

  def fields(self) -> dict[str, Json]:
    """The entry's extra fields."""
    return {
      "red": None if self.red is None else list(self.red),
      "converted": [list(c) for c in self.converted],
      "size": None if self.size is None else list(self.size),
    }

  @staticmethod
  def parse(fields: dict[str, Json]) -> _Entry:
    """The entry of an entry file's extra fields.

    Raises:
      ValueError: for a missing or malformed field.
    """
    converted = fields.get("converted")
    if not isinstance(converted, list):
      raise ValueError(f"pool entry fields {fields!r}")
    flags = [(c[0], c[1]) for c in converted if isinstance(c, list) and len(c) == _PAIR]
    if len(flags) != len(converted) or not all(isinstance(f, bool) for pair in flags for f in pair):
      raise ValueError(f"pool entry fields {fields!r}")
    red, size = (_pair(fields.get(name), fields) for name in ("red", "size"))
    return _Entry(red, tuple((bool(a), bool(b)) for a, b in flags), size)


def _pair(value: Json, fields: dict[str, Json]) -> tuple[float, float] | None:
  """A pool entry field of two numbers (`fields`: the entry's, for the error), or None.

  Raises:
    ValueError: for a value that is neither None nor two numbers.
  """
  if value is None:
    return None
  numbers = [float(x) for x in value if isinstance(x, float | int)] if isinstance(value, list) else []
  if len(numbers) != _PAIR or not isinstance(value, list) or len(value) != _PAIR:
    raise ValueError(f"pool entry fields {fields!r}")
  return numbers[0], numbers[1]


@final
class TexturePool:
  """Core textures derived from host textures by channel recipes (deduplicated, parallel build).

  Attributes:
    textures: the built core textures (0: the unused-slot placeholder; equal texels: one texture).
    sizes: the (width, height) they are sampled at: the stored one (a uniform texture's before its
      1 x 1 storage).
    layouts: their texel layouts (`types.TEX_LAYOUT_*`).
    red_range: core id -> (smallest, largest) red texel (opacity textures).
  """

  def __init__(self, scene: Scene, report: LoadReport, storage: TextureStorage = DEFAULT_STORAGE, *, device: wp.Device) -> None:
    """An empty pool over `scene`'s host textures (`report`: wrap conflicts; `storage`: how the
    textures are stored), building on `device`."""
    self.scene = scene
    self.report = report
    self.storage = storage
    self.device = device
    self.textures: list[wp.Texture2D] = [placeholder_texture(device)]
    self.sizes: list[tuple[float, float]] = [(1.0, 1.0)]
    self.layouts: list[int] = [TEX_LAYOUT_IDENTITY]
    self.red_range: dict[int, tuple[float, float]] = {}
    self._source = self._sources(scene)
    # (image, space, hdr) -> the converted texels of color sources not stored as is (built in
    # `finish`, kept for the pool's lifetime: equal images and spaces convert once).
    self._converted: dict[tuple[int, ColorSpace, bool], Converted] = {}
    # (recipe, color, coverage, hdr): the texture's every input (sources, channels, color space,
    # mip weights; the wrap modes come from the sources) -> core id.
    self._keys: dict[tuple[Recipe, bool, Coverage | None, bool], int] = {}
    self._pending: list[_Request] = []

  @staticmethod
  def _sources(scene: Scene) -> list[TextureId]:
    """Host texture -> the host texture whose texels its recipes read: the first one of the same
    image, wrap and storage (LIN_REC709 and DATA are stored alike).

    Raises:
      ValueError: for a stored color space code that is unknown or output-only (a corrupt `.rayw`).
    """
    textures = scene.textures
    first: dict[tuple[int, int, int, int], TextureId] = {}
    out: list[TextureId] = []
    for t in range(len(textures)):
      space = colorspace.stored_space(int(textures.colorspace[t]), where=f"texture {t}")
      storage = ColorSpace.LIN_REC709 if space in _LINEAR_STORAGE else space
      key = (int(textures.image[t]), colorspace.code(storage), int(textures.wrap[t, 0]), int(textures.wrap[t, 1]))
      out.append(first.setdefault(key, TextureId(t)))
    return out

  def _channel(self, channel: Channel) -> Channel:
    """`channel` reading its host texture's source (`_sources`)."""
    if channel.texture < 0:
      return channel
    return channel._replace(texture=self._source[channel.texture])

  def get(
    self, recipe: Recipe, *, color: bool, red_range: bool = False, coverage: Coverage | None = None, hdr: bool = False
  ) -> int:
    """Core texture id of a 4-channel recipe; color recipes are linearised (sRGB sources) and
    converted to Rec.709 (other spaces).

    `red_range`: also record the smallest and largest red texels (`red_range`), whichever request
    of a shared recipe asks for it. `coverage`: the opacity weighting the color mips. `hdr`: an
    emission recipe (converted 8-bit sources keep values above 1; only part of the key when a
    source is converted, so other recipes still share the texture).

    Raises:
      ValueError: for `red_range` on a recipe whose texture `finish` already built without it.
    """
    a, b, c, d = (self._channel(ch) for ch in recipe)
    recipe = (a, b, c, d)
    if coverage is not None:
      coverage = coverage._replace(channel=self._channel(coverage.channel))
    hdr = hdr and color and any(not self._as_is(ch.texture) for ch in recipe if ch.texture >= 0)
    key = (recipe, color, coverage, hdr)
    if key not in self._keys:
      self._keys[key] = len(self.textures) + len(self._pending)
      self._pending.append(_Request(recipe, color, red_range, coverage, hdr))
      self._check_wrap(recipe)
    elif (slot := self._keys[key] - len(self.textures)) >= 0:
      self._pending[slot].red_range |= red_range
    elif red_range and self._keys[key] not in self.red_range:
      raise ValueError(f"texture {self._keys[key]} was built (`finish`) without the red range now requested")
    return self._keys[key]

  def _check_wrap(self, recipe: Recipe) -> None:
    """Reports a recipe whose sources disagree on their wrap modes (the first source's is kept)."""
    sources = _textures_of(recipe)
    wraps = {tuple(int(w) for w in self.scene.textures.wrap[t]) for t in sources}
    if len(wraps) > 1:
      self.report.approx("gpu.texture.wrap(sources disagree: the first one's)", f"texture {sources[0]}")

  def finish(self) -> None:
    """Builds the textures registered by `get`: from their disk entries (`_cached`), else from
    the texels (`_build`: preparation and mips; each then gets an entry). The host side is read or
    built on a thread pool while this thread uploads each texture as it completes
    (`TextureUploader`); the textures keep the requests' order."""
    pending = self._pending
    root = cache_dir()
    keys = [self._entry_key(item) for item in pending]  # with caching off too: the digests are checked
    files = [None if root is None else root / "textures" / f"pool-{key}.json" for key in keys]
    upload = TextureUploader(self.device)
    threads = min(_MAX_THREADS, os.cpu_count() or 1)
    done: list[tuple[CoreTexture, _Entry] | None] = [None] * len(pending)
    staging = Staging(_STAGING_SLOTS, self.device) if self.device.is_cuda else None
    abort = staging.close if staging is not None else None
    try:
      for i, hit in as_done(lambda k: self._read(pending[k], files[k], staging), range(len(pending)), threads, abort=abort):
        if hit is not None:
          done[i] = (upload.add(hit[0]), hit[2])
          if staging is not None:
            staging.release(hit[1])
      with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        misses = [i for i, d in enumerate(done) if d is None]
        self._convert(pool, [pending[i] for i in misses], {i: d[1] for i, d in enumerate(done) if d is not None})
      for k, (host, entry) in as_done(lambda i: self._build(pending[i], files[i]), misses, threads):
        done[misses[k]] = (upload.add(host), entry)
    finally:
      upload.finish()  # the queued copies read the staging slots: done before they are freed
    for d in done:
      if d is None:
        raise RuntimeError("a pool texture was neither read nor built")
      self._append(*d)
    self._pending = []

  def _read(self, item: _Request, path: Path | None, staging: Staging | None) -> tuple[HostTexture, int, _Entry] | None:
    """A request's host texture from its disk entry at `path` (in a `staging` slot on CUDA, else
    slot -1) and what the entry remembers, or None (a miss)."""
    found = self._cached(item, path)
    if found is None or path is None:
      return None
    if staging is None:
      levels = texture_cache.load_levels(path.parent, found[0].texels)
      return None if levels is None else (HostTexture(levels, found[0].texels, found[0].layout), -1, found[1])
    staged = read_staged(path, found[0], staging)
    return None if staged is None else (*staged, found[1])

  def _append(self, texture: CoreTexture, entry: _Entry) -> None:
    """Adds a built core texture and its red range under the next core id."""
    if entry.red is not None:
      self.red_range[len(self.textures)] = entry.red
    self.textures.append(texture.texture)
    self.sizes.append(texture.size if entry.size is None else entry.size)
    self.layouts.append(texture.layout)

  def _entry_key(self, item: _Request) -> str:
    """The digest of a request's every input (the module docstring): its disk entry's name."""
    storage = self.storage.on(srgb8=device_caps(self.device).device_textures)
    parts: list[object] = [_ENTRY_FORMAT, _ENTRY_CODE, repr(storage), item.color, item.red_range, item.hdr]
    parts += [self._source_key(c) for c in item.recipe]
    if item.coverage is not None:
      parts += [self._source_key(item.coverage.channel), item.coverage.scale, item.coverage.bias]
    return hashlib.blake2b(repr(parts).encode(), digest_size=20).hexdigest()

  def _source_key(self, channel: Channel) -> tuple[object, ...]:
    """A channel's inputs: its source image's content digest, color space and wrap, the channel.

    Raises:
      ValueError: for an image without a content digest (`Images.digest`; every `SceneBuilder`
        scene has them).
    """
    t = channel.texture
    if t < 0:
      return ("one", channel.channel, channel.invert)
    textures = self.scene.textures
    digest = image_digest(self.scene, int(textures.image[t]))
    wrap = (int(textures.wrap[t, 0]), int(textures.wrap[t, 1]))
    return (digest.hex(), int(textures.colorspace[t]), wrap, channel.channel, channel.invert)

  def _cached(self, item: _Request, path: Path | None) -> tuple[texture_cache.Entry, _Entry] | None:
    """A request's disk entry and what it remembers, or None (a miss); its levels are read next
    (`finish`)."""
    found = None if path is None else texture_cache.read_entry(path)
    if found is None:
      return None
    try:
      entry = _Entry.parse(found.extra)
    except ValueError as e:
      warnings.warn(f"raywarp cache: unreadable {path} ({e}); building the texture again", RuntimeWarning, stacklevel=2)
      return None
    if (item.red_range and entry.red is None) or len(entry.converted) != len(self._conversions(item)):
      return None
    return found, entry

  def _build(self, item: _Request, path: Path | None) -> tuple[HostTexture, _Entry]:
    """One host texture built from its texels, and what its entry remembers (written at `path`)."""
    texels, srgb, address = self._texels(item)
    coverage = None if item.coverage is None else self._coverage(item.coverage, texels.shape[0], texels.shape[1])
    host = host_texture(texels, srgb=srgb, address=address, coverage=coverage, storage=self.storage, device=self.device)
    folded = _uniform(host)
    size = None if folded is None else host.size
    host = host if folded is None else folded
    red = None
    if item.red_range:
      unit = 255.0 if texels.dtype == np.uint8 else 1.0
      red = (float(texels[..., 0].min()) / unit, float(texels[..., 0].max()) / unit)
    converted = tuple((self._converted[k].negative, self._converted[k].clipped) for k in self._conversions(item))
    entry = _Entry(red, converted, size)
    if path is not None:
      write_entry(path, host, entry.fields())
    return host, entry

  def _conversions(self, item: _Request) -> list[tuple[int, ColorSpace, bool]]:
    """(image, space, hdr) of a request's sources converted to Rec.709 (color requests of a space
    not stored as is)."""
    if not item.color:
      return []
    converted = [t for t in _textures_of(item.recipe) if not self._as_is(t)]
    return [(int(self.scene.textures.image[t]), self._space(t), item.hdr) for t in converted]

  def _convert(self, pool: concurrent.futures.ThreadPoolExecutor, misses: list[_Request], entries: dict[int, _Entry]) -> None:
    """Converts (on `pool`) the color sources of the requests to build (`misses`) whose space is
    not stored as is, once per (image, space); reports each conversion and gamut clamp of every
    pending request, those of cached ones (`entries`, by pending index) from their entries."""
    todo: dict[tuple[int, ColorSpace, bool], TextureId] = {}
    for item in self._pending:
      for t in _textures_of(item.recipe) if item.color else []:
        key = (int(self.scene.textures.image[t]), self._space(t), item.hdr)
        if not self._as_is(t) and key not in self._converted:
          todo.setdefault(key, t)
    needed = {key for item in misses for key in self._conversions(item)}
    remembered = {
      key: flags
      for i, entry in entries.items()
      for key, flags in zip(self._conversions(self._pending[i]), entry.converted, strict=True)
    }
    work = [(key, t) for key, t in todo.items() if key in needed]

    def convert(item: tuple[tuple[int, ColorSpace, bool], TextureId]) -> Converted:
      """The converted texels of a host texture."""
      (_, _, hdr), t = item
      return converted_view(*_rgba_view(self.scene, t), hdr=hdr)

    for (key, _), done in zip(work, pool.map(convert, work), strict=True):
      self._converted[key] = done
    for key, t in todo.items():
      done = self._converted.get(key)
      negative, clipped = (done.negative, done.clipped) if done is not None else remembered[key]
      where = f"texture {t}"
      self.report.approx(f"gpu.texture.colorspace:{key[1].value}(converted to Rec.709 at upload)", where)
      if negative:
        self.report.approx("gpu.texture.colorspace.gamut_clamped", where)
      if clipped:
        self.report.approx("gpu.texture.colorspace.gamut_clamped(above 1 clipped)", where)

  def _space(self, texture: TextureId) -> ColorSpace:
    """The color space of host texture `texture`."""
    return colorspace.from_code(int(self.scene.textures.colorspace[texture]))

  def _as_is(self, texture: TextureId) -> bool:
    """Whether host texture `texture`'s texels are stored untouched (`colorspace.stored_as_is`)."""
    pixel = int(self.scene.images.pixel[int(self.scene.textures.image[texture])])
    return stored_as_is(self._space(texture), float_texels=pixel != PixelType.U8)

  def _view(self, texture: TextureId, item: _Request) -> tuple[Texels, ColorSpace]:
    """(level-0 RGBA texels, space) a request reads from a host texture: its stored texels, or for
    color requests of a space not stored as is the converted ones (`_convert`)."""
    space = self._space(texture)
    if not item.color or self._as_is(texture):
      return _rgba_view(self.scene, texture)
    done = self._converted[int(self.scene.textures.image[texture]), space, item.hdr]
    return done.texels, done.space

  def _coverage(self, coverage: Coverage, h: int, w: int) -> np.ndarray[[int, int], np.dtype[np.float32]]:
    """(h, w) weights in [0, 1] of an opacity, resampled (nearest) and flipped like the texels."""
    t, ch, invert = coverage.channel
    img, _ = _rgba_view(self.scene, t)
    f = _unit_float(img[..., ch])
    if invert:
      f = 1.0 - f
    return np.clip(coverage.scale * _resampled(f, h, w) + coverage.bias, 0.0, 1.0).astype(np.float32)

  def _address(self, sources: list[TextureId]) -> Address:
    """The (u, v) address modes of a recipe: its first source's wrap modes (repeat for
    constants)."""
    if not sources:
      return (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.WRAP)
    wrap = self.scene.textures.wrap[sources[0]]
    return (_ADDRESS[int(wrap[0])], _ADDRESS[int(wrap[1])])

  def _texels(self, item: _Request) -> tuple[Texels, bool, Address]:
    """(texels flipped to the core's row order, whether `host_texture` linearises them, address)."""
    recipe, color = item.recipe, item.color
    sources = _textures_of(recipe)
    address = self._address(sources)
    views = {t: self._view(t, item) for t in sources}
    if len(sources) == 1 and views[sources[0]][0].dtype == np.uint8:
      img, space = views[sources[0]]
      fast = _copied_channels(recipe, img, space, color=color)
      if fast is not None:
        return fast[0], fast[1], address
    # General path: float32 per channel, sources resampled (nearest) to the largest one.
    h = max((views[t][0].shape[0] for t in sources), default=1)
    w = max((views[t][0].shape[1] for t in sources), default=1)
    out = np.empty((h, w, 4), np.float32)
    cache: dict[TextureId, np.ndarray[[int, int, int], np.dtype[np.float32]]] = {}
    for k, (t, ch, invert) in enumerate(recipe):
      if t < 0:
        out[..., k] = 1.0
        continue
      if t not in cache:
        img, cs = views[t]
        f = _unit_float(img)
        if color and cs == ColorSpace.SRGB_REC709:
          f[..., :3] = srgb_to_linear(f[..., :3])
        cache[t] = _resampled(f, h, w)
      out[..., k] = 1.0 - cache[t][..., ch] if invert else cache[t][..., ch]
    if color:
      return out.astype(np.float16), False, address
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), False, address


def _uniform(host: HostTexture) -> HostTexture | None:
  """The 1 x 1 texture of `host`'s one texel value when every texel of its every level is that
  value (same format, layout and address modes), else None; BORDER-addressed textures are kept
  (their blend with the border spans one texel of the stored size)."""
  if int(wp.TextureAddressMode.BORDER) in host.texels.address:
    return None
  texel = host.levels[0][:1, :1]
  bits = f"u{texel.itemsize}"  # bitwise: -0 and NaN texels compare as stored
  if not all(np.array_equal(lv.view(bits), np.broadcast_to(texel.view(bits), lv.shape)) for lv in host.levels):
    return None
  one = np.ascontiguousarray(texel)
  texels = host.texels
  digest = bake_cache.texture_digest(one, texels.srgb8, *texels.address, "uniform")
  return HostTexture([one], texels._replace(digest=digest, shape=one.shape), host.layout)


def _textures_of(recipe: Recipe) -> list[TextureId]:
  """The distinct host textures a recipe reads, sorted (constants excluded)."""
  return sorted({c.texture for c in recipe if c.texture >= 0})


def _unit_float(img: Array) -> _Floats:
  """`img` as float32, 8-bit values scaled to [0, 1]."""
  f = img.astype(np.float32)
  if img.dtype == np.uint8:
    f /= 255.0
  return f


def _resampled(f: _Floats, h: int, w: int) -> _Floats:
  """(h, w, ...) `f` resampled (nearest) and flipped to the core's row order."""
  ys = np.arange(h) * f.shape[0] // h
  xs = np.arange(w) * f.shape[1] // w
  return f[np.ix_(ys, xs)][::-1]


def _copied_channels(recipe: Recipe, img: Texels, space: ColorSpace, *, color: bool) -> tuple[Texels, bool] | None:
  """The path of one 8-bit source: channels copied, inverted (1 - x) or the constant 1.

  Returns (texels, sRGB to linearise), or None for an inverted sRGB color channel (inverted after
  linearisation: the general path). A color texture of sRGB texels is linearised by
  `host_texture` (rgb only); linear 8-bit texels stay normalised uint8.
  """
  srgb = color and space == ColorSpace.SRGB_REC709
  if srgb and any(c.invert for c in recipe[:3]):
    return None
  perm = [c.channel if c.texture >= 0 else 0 for c in recipe]
  flipped = img[::-1]
  # Always a copy: the writes below must not reach the source (a 1-row flip is contiguous, and
  # converted sources are shared by every recipe of the pool).
  out = flipped.copy() if perm == [0, 1, 2, 3] else flipped[..., perm]
  for k, (t, _, invert) in enumerate(recipe):
    if t < 0:
      out[..., k] = 255
    elif invert:
      np.subtract(255, out[..., k], out=out[..., k])
  return out, srgb
