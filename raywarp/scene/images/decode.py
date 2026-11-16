"""Image decoding: header probes, decode-time downscale, threaded decode into one texel blob.

Images are stored upload-ready: 1, 2 or 4 channels (RGB is padded to RGBA, the layouts GPU
textures accept), uint8 for 8-bit files (sRGB or linear, as stored), float32 for HDR files (sun
discs overflow float16) and float16 for float arrays given directly (float32 when a value
overflows float16). Decoding happens in two passes
so that no image is held twice: a header pass gives every image's final size and the blob
layout, then a thread pool decodes each image and copies it into the in-memory blob or writes it
to the cache file (`os.pwrite`). One decoder per format: Pillow (8-bit formats), `hdr`
(Radiance RGBE) and the OpenEXR package (`exr`).
"""

from __future__ import annotations

import dataclasses
import io
import re
from collections.abc import Callable, Hashable
from typing import Final, Literal

import numpy as np

from raywarp.scene.images import exr, files, hdr, mips
from raywarp.scene.images.mips import Transfer, ceil_div, decode_lut
from raywarp.scene.model.types import PIXEL_DTYPES, Array, LinearRGBA, PixelType, Texels, pixel_itemsize
from raywarp.scene.model.wpbr import Dependency

# Extensions decoded with Pillow, Radiance HDR (hdr.py) and OpenEXR (exr.py).
PIL_EXTENSIONS: Final = frozenset({".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".gif", ".webp"})
HDR_EXTENSIONS: Final = frozenset({".hdr"})
EXR_EXTENSIONS: Final = frozenset({".exr"})
# Floating-point (linear HDR) formats.
FLOAT_EXTENSIONS: Final = HDR_EXTENSIONS | EXR_EXTENSIONS
_F16_MAX: Final = 65504.0
# Array ranks of grey and color images; channel counts of RGB and grey + alpha texels.
_GREY_NDIM: Final = 2
_COLOR_NDIM: Final = 3
_RGB: Final = 3
_GREY_ALPHA: Final = 2

# How a `Derived` image stores its values: 8-bit sRGB color (alpha linear), 8-bit linear clipped
# to [0, 1], or float16.
type Encoding = Literal["srgb", "linear", "float"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Derived:
  """An image computed at decode time from other images (texture baking).

  E.g. specular-glossiness -> metallic-roughness, or an MDL expression of several texture channels.

  Attributes:
    sources: the input images (files; each with its own encoding and size limit).
    fn: `fn(inputs)` -> (h, w, channels) linear float32 values, from the inputs decoded as linear
      float32 (h, w, 4) RGBA at the output size (sRGB sources decoded, alpha kept, grey
      broadcast to RGB).
    channels: 1, 2 or 4 stored channels.
    encoding: "srgb" (8-bit sRGB color, alpha linear), "linear" (8-bit, clipped to [0, 1]) or
      "float" (float16).
    key: identity of the computation (images are deduplicated by it).
    size: (width, height) of the output (None: the largest source's; sources resampled to it).
  """

  sources: tuple[ImageSource, ...]
  fn: Callable[[list[LinearRGBA]], Array] = dataclasses.field(compare=False, hash=False)
  channels: int
  encoding: Encoding
  key: tuple[Hashable, ...]
  size: tuple[int, int] | None = None


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class ImageSource:
  """An image to decode: a file path, an array or a `Derived` image, and the size limit."""

  path: str = ""
  array: Array | None = dataclasses.field(default=None, compare=False, hash=False)
  max_size: int | None = None
  transfer: Transfer = "srgb"  # the 8-bit color channels' encoding (mips filter in linear)
  derived: Derived | None = None

  @property
  def name(self) -> str:
    """The image's source for reports: its path, `<derived:a+b>` or `<array>`."""
    if self.path:
      return self.path
    if self.derived is not None:
      return "<derived:" + "+".join(d.path for d in self.derived.sources) + ">"
    return "<array>"


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class ImageInfo:
  """Final layout of a decoded image (after downscale).

  Attributes:
    width: stored width [texels].
    height: stored height [texels].
    channels: stored channels (1, 2 or 4).
    pixel: stored component type.
    factor: downscale factor applied at decode (a power of two).
    error: the probe failure ("": none): the image becomes a 1 x 1 fallback.
    decode_work: estimated bytes the decode holds before the stored top level exists: the file's
      bytes and the decoder's full-resolution output and conversions (`decode_estimate`).
  """

  width: int
  height: int
  channels: int
  pixel: PixelType
  factor: int = 1
  error: str = ""
  decode_work: int = 0


def _failed(error: str) -> ImageInfo:
  """The fallback layout of an unreadable image, with its error."""
  return ImageInfo(width=1, height=1, channels=4, pixel=PixelType.U8, error=error)


# A decode failure's own report key in its message (`texture.exr.<reason>`). The message is
# `<ExceptionType>: texture.exr.<reason>: <detail>` (`probe` / `decode_base` prefix the exception's
# type); the key must open the message, not appear in a path.
_REPORT_KEY: Final = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_.]*: )?(texture\.exr\.[a-z0-9_]+): ")


def report_key(error: str) -> str:
  """The report key of a decode failure: the `texture.exr.<reason>` its message names (an EXR
  without the OpenEXR package), else `texture.decode_failed`."""
  found = _REPORT_KEY.match(error)
  return "texture.decode_failed" if found is None else found.group(1)


def supported(path: str) -> bool:
  """Whether `path` has an image format decoded here (by its extension)."""
  return files.suffix(path) in PIL_EXTENSIONS | FLOAT_EXTENSIONS


def is_float(path: str) -> bool:
  """Whether `path` is a floating-point (linear HDR) image: an EXR / Radiance file."""
  return files.suffix(path) in FLOAT_EXTENSIONS


def check_max_size(max_size: int | None, where: str) -> None:
  """Checks a texture size limit: None (no limit) or at least 1 texel.

  Raises:
    ValueError: for a limit below 1 (0 is not "no limit": pass None).
  """
  if max_size is not None and max_size < 1:
    raise ValueError(f"{where}: max_size must be at least 1 texel (None: no limit), got {max_size}")


def _reduction(width: int, height: int, max_size: int | None) -> int:
  """Power-of-two downscale factor that fits an image within `max_size` (None: no limit, 1).

  Raises:
    ValueError: for a `max_size` below 1 (0 is not "no limit").
  """
  if max_size is None:
    return 1
  check_max_size(max_size, "probe")
  factor = 1
  while max(width, height) > max_size * factor:
    factor *= 2
  return factor


def _scaled(  # noqa: PLR0913 (keyword-only layout values)
  width: int, height: int, channels: int, *, pixel: PixelType, max_size: int | None, work_per_texel: int, file_bytes: int
) -> ImageInfo:
  """The layout of a `width` x `height` image downscaled to fit `max_size` (`_reduction`), RGB
  stored as RGBA; its decode holds `work_per_texel` bytes per source texel plus the file's bytes."""
  factor = _reduction(width, height, max_size)
  return ImageInfo(
    width=ceil_div(width, factor),
    height=ceil_div(height, factor),
    channels=4 if channels == _RGB else channels,
    pixel=pixel,
    factor=factor,
    decode_work=width * height * work_per_texel + file_bytes,
  )


def _file_bytes(path: str, data: bytes | None) -> int:
  """Size of an image file [bytes] (of `data` when read already; 0 when it cannot be stat-ed: the
  probe reports the file)."""
  if data is not None:
    return len(data)
  try:
    return files.size(path)
  except OSError:
    return 0


def probe_bytes(source: ImageSource) -> int:
  """Encoded file bytes a read or probe of `source` may hold, to reserve before reading (a derived
  image: its largest source's, as they are probed one at a time; 0 for an array). An estimate from
  the file's size: a file that changes during the build is not accounted for."""
  if source.derived is not None:
    return max((probe_bytes(s) for s in source.derived.sources), default=0)
  if not source.path:
    return 0
  return _file_bytes(source.path, None)


# Pillow grey modes wider than 8 bits -> the scale of their values to [0, 1] (16-bit PNGs open as
# "I;16" or "I"; "F" holds floats).
_WIDE_GREY: Final = {"I;16": 1.0 / 65535.0, "I;16B": 1.0 / 65535.0, "I;16L": 1.0 / 65535.0, "I": 1.0 / 65535.0, "F": 1.0}
# Channel counts an array image may have (3 is padded to 4).
_ARRAY_CHANNELS: Final = (1, 2, 3, 4)
# Working bytes per source texel of a decode, before the stored top level exists (measured with
# tracemalloc, rounded up): Radiance RGBE (the run-length bytes, float32 RGB, the RGBA copy and the
# box filter's float32 pad); OpenEXR (planes, the float32 stack, the RGBA copy); Pillow 8-bit
# (the decoded image, its mode conversion and the array: 4 bytes per channel; JPEG downscales
# decode smaller: over-estimated) and 16-bit grey (the float64 rescale and its 8-bit copy); arrays
# downscaled through float32 (a copy and the mean); an 8-bit color image downscaled in linear light
# adds `mips.reduce_u8_work`.
_HDR_WORK: Final = 48
_F32_BYTES: Final = 4
_EXR_WORK: Final = 48
_PIL_WORK_PER_CHANNEL: Final = 4
_WIDE_GREY_WORK: Final = 24
_ARRAY_WORK_PER_CHANNEL: Final = 8


def _pil_channels(mode: str) -> tuple[int, str]:
  """(stored channels, Pillow mode to convert to) of a Pillow image mode."""
  if mode in ("L", "1") or mode in _WIDE_GREY:
    return 1, "L"
  if mode in ("LA", "La"):
    return 2, "LA"
  if mode in ("RGBA", "RGBa", "P", "PA"):
    return 4, "RGBA"
  return 4, "RGB"  # RGB, CMYK, YCbCr, LAB, HSV: padded to RGBA


def check_array(array: Array, where: str) -> None:
  """Checks an image given as an array: (h, w) or (h, w, 1-4), at least 1 x 1, uint8 (8-bit
  texels) or floating point with finite values.

  Raises:
    ValueError: for another rank, size, channel count or dtype (uint16 would be read as float
      values up to 65535, not [0, 1]), or a NaN / inf texel.
  """
  shape = tuple(array.shape)
  channels = 1 if array.ndim == _GREY_NDIM else (shape[2] if array.ndim == _COLOR_NDIM else 0)
  if channels not in _ARRAY_CHANNELS or 0 in shape[:2]:
    raise ValueError(f"{where}: an image array is (h, w) or (h, w, 1-4) with h, w >= 1, got shape {shape}")
  if array.dtype != np.uint8 and not np.issubdtype(array.dtype, np.floating):
    raise ValueError(f"{where}: an image array is uint8 or floating point, got {array.dtype} (scale it to [0, 1])")
  if array.dtype != np.uint8 and not np.all(np.isfinite(array)):
    raise ValueError(f"{where}: an image array must be finite (NaN / inf texels)")


def _probe_array(source: ImageSource, a: Array) -> ImageInfo:
  """The layout of an array image (`check_array`ed): uint8 stays 8-bit, floats are float16 unless
  a value overflows it (float32)."""
  h, w = a.shape[:2]
  c = 1 if a.ndim == _GREY_NDIM else a.shape[2]
  pixel = PixelType.U8 if a.dtype == np.uint8 else PixelType.F16
  if pixel == PixelType.F16 and float(np.max(np.abs(a))) > _F16_MAX:
    pixel = PixelType.F32
  info = _scaled(w, h, c, pixel=pixel, max_size=source.max_size, work_per_texel=0, file_bytes=0)
  if info.factor > 1:  # downscaled through float32 copies of the whole array
    linear = mips.reduce_u8_work(w, h, c) if pixel == PixelType.U8 and source.transfer != "linear" else 0
    return dataclasses.replace(info, decode_work=w * h * c * _ARRAY_WORK_PER_CHANNEL + linear)
  return info


def _probe_derived(d: Derived) -> ImageInfo:
  """The layout of a derived image: its size, else its largest source's (first error kept)."""
  infos = [probe(src) for src in d.sources]
  errors = [i.error for i in infos if i.error]
  w, h = d.size or (max((i.width for i in infos), default=1), max((i.height for i in infos), default=1))
  # Each source decoded (one at a time) then held as linear float32 RGBA at the output size.
  inputs = len(infos) * w * h * 4 * _F32_BYTES
  return ImageInfo(
    width=w,
    height=h,
    channels=d.channels,
    pixel=PixelType.F16 if d.encoding == "float" else PixelType.U8,
    error=errors[0] if errors else "",
    decode_work=max((i.decode_work for i in infos), default=0) + 2 * inputs,
  )


def _probe_file(source: ImageSource, data: bytes | None) -> ImageInfo:
  """The layout of an image file from its header (of `data` when given; raises when unreadable)."""
  path = source.path
  suffix = files.suffix(path)
  size = _file_bytes(path, data)
  if suffix in HDR_EXTENSIONS:
    h, w = hdr.size(path)
    return _scaled(w, h, 4, pixel=PixelType.F32, max_size=source.max_size, work_per_texel=_HDR_WORK, file_bytes=size)
  if suffix in EXR_EXTENSIONS:
    header = exr.read_header(path)
    pixel = PixelType.F16 if header.half else PixelType.F32
    w, h, n = header.width, header.height, len(header.channels)
    return _scaled(w, h, n, pixel=pixel, max_size=source.max_size, work_per_texel=_EXR_WORK, file_bytes=size)
  from PIL import Image  # noqa: PLC0415 (optional dependency: the usd / newton extras)

  if data is None and files.is_packaged(path):
    data = files.read(path)  # Pillow opens file paths only
  with Image.open(io.BytesIO(data) if data is not None else path) as im:
    w, h = im.size
    channels, _ = _pil_channels(im.mode)
    work = _WIDE_GREY_WORK if im.mode in _WIDE_GREY else _PIL_WORK_PER_CHANNEL * max(channels, len(im.getbands()))
  info = _scaled(w, h, channels, pixel=PixelType.U8, max_size=source.max_size, work_per_texel=work, file_bytes=size)
  if info.factor > 1 and _numpy_reduce(source.transfer, channels):
    return dataclasses.replace(info, decode_work=info.decode_work + mips.reduce_u8_work(w, h, channels))
  return info


def probe(source: ImageSource, data: bytes | None = None) -> ImageInfo:
  """The decoded layout of `source`, reading only the file header (of `data`, the file's bytes,
  when given: a Pillow format is then not opened again). An unreadable file is a 1 x 1
  fallback with its error."""
  if source.array is not None:
    return _probe_array(source, np.asarray(source.array))
  if source.derived is not None:
    return _probe_derived(source.derived)
  try:
    return _probe_file(source, data)
  except Exception as e:  # noqa: BLE001 (any unreadable file becomes a reported 1x1 fallback)
    return _failed(f"{type(e).__name__}: {e}")


def linear_rgba(pixels: Texels, *, transfer: Transfer) -> LinearRGBA:
  """(h, w, 4) linear float32 RGBA of stored texels (1, 2, 3 or 4 channels, uint8 or float).

  8-bit values are scaled to [0, 1] and decoded with `transfer` (alpha stays linear), grey is
  broadcast to RGB, alpha is 1 when absent.
  """
  f = pixels.astype(np.float32)
  if pixels.dtype == np.uint8:
    f /= 255.0
  c = f.shape[2]
  if c <= _GREY_ALPHA:
    f = np.concatenate(
      [np.repeat(f[..., :1], 3, axis=2), f[..., 1:2] if c == _GREY_ALPHA else np.ones_like(f[..., :1])], axis=2
    )
  elif c == _RGB:
    f = np.concatenate([f, np.ones_like(f[..., :1])], axis=2)
  lut = decode_lut(transfer) if pixels.dtype == np.uint8 else None
  if lut is not None:
    f[..., :3] = lut[pixels[..., :3]] if c >= _RGB else lut[np.repeat(pixels[..., :1], 3, axis=2)]
  return f


def decode_linear(source: ImageSource, width: int, height: int) -> LinearRGBA:
  """(height, width, 4) linear float32 RGBA texels of an image at `width` x `height`.

  As `linear_rgba`; resampled by box / nearest.
  """
  data = read_file(source)[0]
  info = probe(source, data)
  if info.error:
    raise ValueError(info.error)
  f = linear_rgba(_decode_pixels(source, info, data), transfer=source.transfer)
  h, w = f.shape[:2]
  factor = h // height
  if factor > 1 and (h, w) == (factor * height, factor * width):  # the same integer factor on both axes
    f = mips.box_reduce(f, factor)
  elif (h, w) != (height, width):
    ys = (np.arange(height) * h) // height
    xs = (np.arange(width) * w) // width
    f = f[np.ix_(ys, xs)]
  return np.ascontiguousarray(f, np.float32)


def encode(values: Array, encoding: Encoding, channels: int) -> Array:
  """Linear float values (h, w, c) -> the stored texels of a `Derived` image."""
  v = values if values.ndim == _COLOR_NDIM else values[..., None]
  if v.shape[2] != channels:
    raise ValueError(f"derived image: {v.shape[2]} channels computed, {channels} declared")
  if encoding == "float":
    return np.clip(v, -_F16_MAX, _F16_MAX).astype(np.float16)
  return mips.encode_u8(v, transfer="srgb" if encoding == "srgb" else "linear")


def read_file(source: ImageSource) -> tuple[bytes | None, Dependency | None]:
  """(the bytes of a Pillow-decoded image file, its state: `files.read_stat`), read once
  (probed and decoded from memory); (None, None) for arrays, derived images, the float formats
  (read by their own decoders) and unreadable files (`probe` reports them)."""
  if not source.path or files.suffix(source.path) not in PIL_EXTENSIONS:
    return None, None
  try:
    data, holder, st = files.read_stat(source.path)
  except OSError:
    return None, None
  return data, Dependency(path=holder, mtime_ns=st.st_mtime_ns, size=st.st_size)


def _decode_pixels(source: ImageSource, info: ImageInfo, data: bytes | None = None) -> Array:
  """Decoded (height, width, channels) pixels in the stored type of `info` (from `data`, the
  file's bytes, when given); RGB stays RGB (`_stored` pads it)."""
  if source.derived is not None:
    d = source.derived
    inputs = [decode_linear(src, info.width, info.height) for src in d.sources]
    return encode(np.asarray(d.fn(inputs), np.float32), d.encoding, d.channels)
  if source.array is not None:
    a = np.asarray(source.array)
    a = a[..., None] if a.ndim == _GREY_NDIM else a
    if info.factor > 1:
      a = mips.reduce_u8(a, info.factor, transfer=source.transfer) if a.dtype == np.uint8 else mips.box_reduce(a, info.factor)
    return a
  suffix = files.suffix(source.path)
  if suffix in HDR_EXTENSIONS:
    return mips.box_reduce(hdr.read_hdr(source.path), info.factor)
  if suffix in EXR_EXTENSIONS:
    return mips.box_reduce(exr.decode(source.path), info.factor)
  return _decode_pil(info, data if data is not None else (read_file(source)[0] or b""), transfer=source.transfer)


def _numpy_reduce(transfer: mips.Transfer, channels: int) -> bool:
  """Whether a Pillow image of `channels` stored channels is downscaled by `mips.reduce_u8`: color
  images (filtered in linear light) and every image with alpha. Pillow premultiplies RGBA / LA by
  alpha when it resamples, which zeroes the texels of a channel-packed data image whose alpha is
  another map (an Unreal ao / roughness / metallic / emissive-mask texture with a mask of 0 read
  as metalness and roughness 0)."""
  return transfer != "linear" or channels in (_GREY_ALPHA, 4)


def _decode_pil(info: ImageInfo, data: bytes, *, transfer: mips.Transfer) -> Array:
  """`_decode_pixels` of a Pillow format from the file's bytes `data`. A downscale filters the
  8-bit color channels in linear light (`mips.reduce_u8`, as the mips), and every channel of an
  image with alpha unpremultiplied (`_numpy_reduce`); data images without alpha (`transfer`
  "linear") use Pillow's box reduce and JPEG's DCT-domain draft."""
  from PIL import Image  # noqa: PLC0415 (optional dependency: the usd / newton extras)

  with Image.open(io.BytesIO(data)) as im:
    target = (info.width, info.height)
    if info.factor > 1 and im.format == "JPEG" and transfer == "linear":
      im.draft(im.mode, target)  # DCT-domain downscale: nearly free (averages the codes: data only)
    _, mode = _pil_channels(im.mode)
    if im.mode in _WIDE_GREY:
      # 16-bit / 32-bit grey (roughness, height maps): Pillow's convert("L") clips at 255 instead
      # of rescaling; scale the full range to 8 bits.
      grey = np.asarray(im, np.float64) * _WIDE_GREY[im.mode]
      im = Image.fromarray((np.clip(grey, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "L")
    if im.mode != mode:
      im = im.convert(mode)
    if im.size != target and _numpy_reduce(transfer, len(im.getbands())):
      pixels = np.asarray(im)
      pixels = pixels[..., None] if pixels.ndim == _GREY_NDIM else pixels
      return mips.reduce_u8(pixels, info.factor, transfer=transfer)
    if im.size != target:
      factor = im.size[0] // target[0] if im.size[0] % target[0] == 0 else 0
      im = (
        im.reduce(factor)
        if factor > 1 and ceil_div(im.size[1], factor) == target[1]
        else im.resize(target, Image.Resampling.BOX)
      )
    pixels = np.asarray(im)
    return pixels[..., None] if pixels.ndim == _GREY_NDIM else pixels  # grey: (h, w, 1)


def _stored(pixels: Array, shape: tuple[int, int, int], dtype: np.dtype) -> Texels:
  """Decoded `pixels` as the stored top level of `shape` and `dtype` (RGB padded to RGBA).

  Raises:
    ValueError: when the decoded size differs from the probed one.
  """
  if pixels.shape[2] == _RGB:
    if pixels.shape[:2] != shape[:2]:
      raise ValueError(f"decoded {pixels.shape}, expected {shape}")
    base = np.empty(shape, dtype)
    base[..., :3] = pixels
    base[..., 3] = 255 if dtype == np.uint8 else 1.0
    return base
  if pixels.shape != shape:
    raise ValueError(f"decoded {pixels.shape}, expected {shape}")
  return np.ascontiguousarray(pixels, dtype=dtype)


def decode_base(source: ImageSource, info: ImageInfo, data: bytes | None = None) -> tuple[Texels, str]:
  """(the stored top level of `source`, "" or the error message: mid-grey when it cannot be
  decoded); from `data`, the file's bytes, when given."""
  dtype = PIXEL_DTYPES[info.pixel]
  shape = (info.height, info.width, info.channels)
  base = None
  error = info.error
  if not error:
    try:
      base = _stored(_decode_pixels(source, info, data), shape, dtype)
    except Exception as e:  # noqa: BLE001 (any decode failure becomes a reported mid-grey image)
      error = f"{type(e).__name__}: {e}"
  if base is None:
    base = np.full(shape, 128 if dtype == np.uint8 else 0.5, dtype)
  return base, error


def level_shapes(info: ImageInfo, levels: int) -> list[tuple[int, int, int]]:
  """(height, width, channels) of an image's first `levels` mip levels."""
  return [(max(1, info.height >> i), max(1, info.width >> i), info.channels) for i in range(levels)]


def image_nbytes(info: ImageInfo, levels: int) -> int:
  """Bytes of an image's `levels` mip levels in the blob."""
  return sum(h * w * c for h, w, c in level_shapes(info, levels)) * pixel_itemsize(info.pixel)
