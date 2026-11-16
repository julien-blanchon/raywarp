"""Material texture upload: 8-bit / float16 images -> host levels (`host_texture`, on any thread)
-> mipmapped GPU textures (`TextureUploader`, one thread, no synchronisation), and their disk
cache (`write_entry`, `read_entry`).

A texture stores only its distinct channels and reads them through its texel layout
(`formats.pack_channels`); textures of equal content are uploaded once
(`TextureUploader`). 8-bit sRGB color is stored per `TextureStorage.color`: 8-bit sRGB RGBA (one
channel for a grey image without alpha) decoded by the texture unit before filtering ("srgb8",
CUDA; its mips filtered in linear values on the host, `srgb8_mip_chain`), or linear float16.
Every other texture is exact. Every mip level is built on the host (`host_texture`: box filtered,
coverage-weighted or 8-bit sRGB in linear values) and cached with the texture, so a warm upload
only copies levels. Box reductions of even sides average pairs
(Warp's own arithmetic); odd ones integrate the complete image extent (`box_chain`).

Disk cache (`texture_cache`): `write_entry` writes a texture's host levels once per content and a
caller's entry naming them; `read_entry` reads the host texture back without reading,
converting, packing, filtering or hashing a source texel.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import functools
import itertools
import queue
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, final, override

import numpy as np
import warp as wp
from warp._src.texture import MemoryType

from raywarp._base.types import TEX_LAYOUT_ONE
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import device_caps, host_copy
from raywarp.core.pipeline.textures import cache as texture_cache
from raywarp.core.pipeline.textures.cache import HostLevel, TexelFile
from raywarp.core.pipeline.textures.formats import (
  DEFAULT_STORAGE,
  NCOLOR,
  Coverage,
  Level,
  TextureStorage,
  capped_levels,
  coverage_mip_chain,
  halve_pairs,
  layout_code,
  level_shapes,
  linear_f16,
  pack_channels,
  reduced,
  srgb8_mip_chain,
)
from raywarp.scene.images.mips import area_half_sum
from raywarp.scene.model.types import Json

# A texture's (u, v) address modes.
type Address = tuple[wp.TextureAddressMode, wp.TextureAddressMode]

_WRAP: Final = (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.WRAP)

# cuda.h: CUresourcetype, CUfilter_mode, CUaddress_mode, CU_TRSF_* flags.
_RESOURCE_ARRAY: Final = 0
_RESOURCE_MIPMAPPED_ARRAY: Final = 1
_FILTER_LINEAR: Final = 1
_ADDRESS_CLAMP: Final = 1
_TRSF_NORMALIZED_COORDINATES: Final = 0x02
_TRSF_SRGB: Final = 0x10


@final
class CoreTexture(NamedTuple):
  """A built material texture: the GPU texture, its texel layout and stored (width, height)."""

  texture: wp.Texture2D
  layout: int
  size: tuple[float, float]


def _halve_u8(x: Level[np.uint8], w: int, h: int) -> Level[np.uint8]:
  """Area filter of a uint8 (H, W, C) level to (h, w), with ties rounded to even.

  Warp averages pairs in float64 (width, then height) and rounds half to even; on integer sums
  that is round(sum / 2^k) with ties to even. Even reductions keep that exact path;
  odd reductions include the complete source extent with integer area weights.
  """
  if (x.shape[1] != w and x.shape[1] % 2) or (x.shape[0] != h and x.shape[0] % 2):
    total, divisor = area_half_sum(x.astype(np.uint64), width=w, height=h)
    q, rem = total // divisor, total % divisor
    q += (2 * rem > divisor) | ((2 * rem == divisor) & ((q & 1) == 1))
    return q.astype(np.uint8)
  s = x.astype(np.uint16)
  k = 0
  if x.shape[1] != w:
    s = s[:, 0 : 2 * w : 2] + s[:, 1 : 2 * w : 2]
    k += 1
  if x.shape[0] != h:
    s = s[0 : 2 * h : 2] + s[1 : 2 * h : 2]
    k += 1
  if k == 0:
    return x.copy()
  q = np.right_shift(s, k)
  r = np.bitwise_and(s, (1 << k) - 1)
  half = 1 << (k - 1)
  q += (r > half) | ((r == half) & (np.bitwise_and(q, 1) == 1))
  return q.astype(np.uint8)


def _halve_f16(x: Level[np.float16], w: int, h: int) -> Level[np.float16]:
  """Area filter of a float16 (H, W, C) level to (h, w), averaged in float32 (`halve_pairs`)."""
  return halve_pairs(x.astype(np.float32), w, h).astype(np.float16)


def box_chain[T: (np.uint8, np.float16)](base: Level[T]) -> list[Level[T]]:
  """Every mip level of a (h, w, c) uint8 / float16 `base` down to 1 x 1 (`level_shapes`):
  `_halve_u8` / `_halve_f16` level by level."""
  shapes = level_shapes(base.shape[1], base.shape[0])
  halve: Callable[[Level[T], int, int], Level[T]] = _halve_u8 if base.dtype == np.uint8 else _halve_f16  # pyrefly: ignore[bad-assignment]  # the dtype test does not narrow T
  levels = [base]
  for w, h, _ in shapes[1:]:
    levels.append(halve(levels[-1], w, h))
  return levels


@final
class MipTexture2D(wp.Texture2D):
  """A mipmapped texture of given host levels (bilinear, trilinear between them): the material
  textures (`box_chain`, coverage-weighted or 8-bit sRGB chains, `host_texture`) and the
  environment maps' radiance.

  `srgb8`: 8-bit sRGB RGBA or one-channel texels decoded by the texture unit before filtering
  (CUDA; Warp's own texture object has no sRGB read, so it is replaced by one that has, over the
  same CUDA array).
  On CUDA the level copies are queued on the device's stream without waiting
  (`_upload_mipmap_chain`); the texture keeps no reference to the host levels.
  """

  def __init__(self, levels: Sequence[HostLevel], *, address: Address, device: wp.Device, srgb8: bool = False) -> None:
    """A texture on `device` of host `levels` (every level of the full chain of level 0's size);
    `address`: its (u, v) address modes; `srgb8`: read as sRGB.

    Raises:
      ValueError: for levels that are not the full chain, `srgb8` for texels that are not 8-bit
        RGBA or one-channel on CUDA.
    """
    self._levels: list[HostLevel] = list(levels)
    # Level 0 aliased (Warp copies a NumPy array first), with canonical strides: NumPy keeps any
    # stride on a size-1 axis of a contiguous array; Warp rejects it.
    base = levels[0].reshape(-1).reshape(levels[0].shape)
    dtype = wp.uint8 if base.dtype == np.uint8 else wp.float16
    linear = wp.TextureFilterMode.LINEAR
    super().__init__(
      wp.array(base, dtype=dtype, device="cpu", copy=False),
      filter_mode=linear,
      address_mode_u=address[0],
      address_mode_v=address[1],
      num_mip_levels=0,
      mip_filter_mode=linear,
      device=device,
    )
    self._levels = []
    if srgb8:
      self._read_srgb()

  def _read_srgb(self) -> None:
    """Replaces Warp's texture object by one decoding the 8-bit sRGB texels before filtering."""
    runtime = self._runtime
    if not self.device.is_cuda or self.dtype != wp.uint8 or self.num_channels not in (1, 4) or runtime is None:
      raise ValueError("the texture unit's sRGB decode needs 8-bit RGBA or one-channel texels on CUDA")
    address = (int(self._address_mode_u), int(self._address_mode_v))
    with self.device.context_guard:
      handle = _srgb_object(self._array_handle, mipmapped=self._is_mipmapped, address=address, nlevel=self._num_mip_levels)
      old, self._tex_handle = self._tex_handle, handle
      runtime.core.wp_texture_object_destroy_device(self.device.context, old)

  @override
  def _generate_mipmap_chain(self, data_np: HostLevel) -> list[HostLevel]:
    """The given levels (Warp's level shapes checked)."""
    if [lv.shape[1::-1] for lv in self._levels] != [s[:2] for s in self._mip_shapes]:
      raise ValueError(f"host levels {[lv.shape for lv in self._levels]} do not match the texture's {self._mip_shapes}")
    return self._levels

  @override
  def _upload_mipmap_chain(self, data: wp.array) -> None:
    """Queues the host -> CUDA-array copies of the given levels on the device's stream without
    waiting (a copy from pageable memory returns once its source is staged: the levels may be
    dropped at once; later work on the stream sees the texels). A CPU texture takes Warp's
    path."""
    if not self.device.is_cuda:
      super()._upload_mipmap_chain(data)
      return
    runtime = self._runtime
    if runtime is None:
      raise RuntimeError("a CUDA texture without Warp's runtime")
    core, context = runtime.core, self.device.context
    stream = wp.get_stream(self.device).cuda_stream
    item = wp.types.type_size_in_bytes(self.dtype) * self.num_channels
    for level, arr in enumerate(self._generate_mipmap_chain(self._levels[0])):
      w, h, d = self._mip_shapes[level]
      target = core.wp_texture_get_mip_level_array_device(context, self._array_handle, level)
      if not target:
        raise RuntimeError(f"Failed to get mip level {level} CUDA array: {runtime.get_error_string()}")
      src = np.ascontiguousarray(arr)
      ok = core.wp_texture_copy_device(
        context, w * item, h, d, MemoryType.ARRAY, target, 0, 0, MemoryType.HOST, src.ctypes.data, w * item, h, stream
      )
      if not ok:
        raise RuntimeError(f"Failed to copy mip level {level} to texture: {runtime.get_error_string()}")


def placeholder_texture(device: wp.Device) -> wp.Texture2D:
  """A 1 x 1 white texture on `device` standing for an unused texture slot."""
  texels = np.full((1, 1, 4), 255, dtype=np.uint8)
  texture = wp.Texture2D(texels, filter_mode=wp.TextureFilterMode.LINEAR, device=device)
  return bake_cache.register_texture(texture, bake_cache.texture_digest(texels, "placeholder"))


class _ResourceDesc(ctypes.Structure):
  """CUDA_RESOURCE_DESC of a CUDA array or a mipmapped one (the union's other members:
  `reserved`)."""

  _fields_ = (
    ("resType", ctypes.c_uint),
    ("handle", ctypes.c_void_p),
    ("reserved", ctypes.c_int * 30),  # the union is 128 bytes: 8 for the handle, 120 more
    ("flags", ctypes.c_uint),
  )


class _TextureDesc(ctypes.Structure):
  """CUDA_TEXTURE_DESC."""

  _fields_ = (
    ("addressMode", ctypes.c_uint * 3),
    ("filterMode", ctypes.c_uint),
    ("flags", ctypes.c_uint),
    ("maxAnisotropy", ctypes.c_uint),
    ("mipmapFilterMode", ctypes.c_uint),
    ("mipmapLevelBias", ctypes.c_float),
    ("minMipmapLevelClamp", ctypes.c_float),
    ("maxMipmapLevelClamp", ctypes.c_float),
    ("borderColor", ctypes.c_float * 4),
    ("reserved", ctypes.c_int * 12),
  )


@functools.cache
def _driver() -> ctypes.CDLL:
  """The CUDA driver library (the one Warp loaded), with the argument types of the functions used
  here (each returns a CUresult: ctypes' default int result)."""
  lib = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
  ptr = ctypes.POINTER
  lib.cuGetErrorString.argtypes = (ctypes.c_int, ptr(ctypes.c_char_p))
  lib.cuTexObjectCreate.argtypes = (ptr(ctypes.c_uint64), ptr(_ResourceDesc), ptr(_TextureDesc), ctypes.c_void_p)
  return lib


def _srgb_object(array: int, *, mipmapped: bool, address: tuple[int, int], nlevel: int) -> int:
  """A texture object of Warp's CUDA `array` (a mipmapped one when `mipmapped`) decoding its 8-bit
  sRGB texels before filtering, with the sampler of every material texture (bilinear, trilinear,
  normalised coordinates, border color 0, `address` (u, v) modes); the calling thread's current
  context is the array's.

  Raises:
    RuntimeError: naming the driver's error.
  """
  kind = _RESOURCE_MIPMAPPED_ARRAY if mipmapped else _RESOURCE_ARRAY
  res = _ResourceDesc(resType=kind, handle=array, flags=0)
  tex = _TextureDesc(
    filterMode=_FILTER_LINEAR,
    flags=_TRSF_NORMALIZED_COORDINATES | _TRSF_SRGB,
    maxAnisotropy=0,
    mipmapFilterMode=_FILTER_LINEAR,
    mipmapLevelBias=0.0,
    minMipmapLevelClamp=0.0,
    maxMipmapLevelClamp=float(nlevel - 1),
  )
  tex.addressMode[0], tex.addressMode[1], tex.addressMode[2] = address[0], address[1], _ADDRESS_CLAMP
  handle = ctypes.c_uint64()
  result = _driver().cuTexObjectCreate(ctypes.byref(handle), ctypes.byref(res), ctypes.byref(tex), None)
  if result != 0:
    text = ctypes.c_char_p()
    _driver().cuGetErrorString(result, ctypes.byref(text))
    raise RuntimeError(f"CUDA driver cuTexObjectCreate failed with error {result}: {(text.value or b'?').decode()}")
  return handle.value


@final
class HostTexture(NamedTuple):
  """A material texture's host side, ready to upload (`host_texture`, `read_entry`).

  Attributes:
    levels: its host levels (the full chain).
    texels: their description (content digest, shape, format, address modes).
    layout: its texel layout (`formats.pack_channels`).
  """

  levels: Sequence[HostLevel]
  texels: TexelFile
  layout: int

  @property
  def size(self) -> tuple[float, float]:
    """The stored (width, height)."""
    h, w, _ = self.texels.shape
    return (float(w), float(h))


def host_texture(  # noqa: PLR0913 (keyword-only build options)
  rgba: HostLevel,
  *,
  srgb: bool,
  address: Address = _WRAP,
  coverage: Coverage | None = None,
  storage: TextureStorage = DEFAULT_STORAGE,
  device: wp.Device,
) -> HostTexture:
  """The host side of the mipmapped texture of an (h, w, 4) image (row 0 is sampled at v = 0),
  wrapping by default, for `device` (`TextureUploader` makes it).

  Only the image's distinct channels are stored (`pack_channels`, read through the returned
  layout). 8-bit sRGB color (`srgb`) is stored as `storage.color` says (`TextureStorage.on`:
  float16 on a device without the sRGB decode): "srgb8" keeps the four 8-bit channels (one when
  the color is a single stored channel without alpha: a grey image), decoded by the texture unit,
  its mips filtered in linear values; "float16" linearises them. Data images
  stay normalised uint8, float16 images are taken as linear. `address` is the (u, v) address
  mode. `coverage`: (h, w) weights in [0, 1] of a color image's texels (its opacity): the color
  mips are coverage-weighted (built on the host). `storage.max_size` halves larger images first.
  The content digest (`TexelFile.digest`) keys equal textures and the bake caches.
  """
  storage = storage.on(srgb8=device_caps(device).device_textures)
  srgb8 = srgb and storage.color == "srgb8" and rgba.dtype == np.uint8
  k = capped_levels(rgba.shape[1], rgba.shape[0], storage.max_size)
  if k:
    rgba, srgb, coverage = reduced(rgba, coverage, k, srgb=srgb, srgb8=srgb8)
  border = wp.TextureAddressMode.BORDER in address
  kept, layout = pack_channels(rgba, border=border, alpha_apart=srgb or coverage is not None)
  if srgb8 and kept != (0,):  # the texture unit's sRGB read: one color channel, else RGBA
    alpha = NCOLOR if NCOLOR in kept else TEX_LAYOUT_ONE
    kept, layout = (0, 1, 2, NCOLOR), layout_code((0, 1, 2, alpha))
  stored = np.ascontiguousarray(rgba[..., list(kept)])
  ncolor = sum(1 for c in kept if c < NCOLOR)
  cover = None if coverage is None else bake_cache.texture_digest(coverage)
  height, width = stored.shape[:2]
  mip_key = ("odd-area-mips-v1",) if (width & (width - 1)) or (height & (height - 1)) else ()
  mode = ("srgb8",) if srgb8 else ()
  digest = bake_cache.texture_digest(stored, srgb, ncolor, int(address[0]), int(address[1]), cover, *mode, *mip_key)
  levels, texels = _host_levels(stored, srgb=srgb, srgb8=srgb8, ncolor=ncolor, coverage=coverage)
  return HostTexture(levels, texels._replace(digest=digest, address=(int(address[0]), int(address[1]))), layout)


@final
class TextureUploader:
  """GPU textures of host textures (`HostTexture`) on one device; equal content is uploaded once.

  Every texture is made on the calling thread and its copies queued on the device's stream
  without waiting (`MipTexture2D`): textures made one after another overlap their copies, and the
  CUDA driver serialises texture creation anyway (threads only contend for it). The host levels
  may be dropped as soon as `add` returns (a copy from pageable memory returns once its source is
  staged: CUDA's API synchronisation behaviour); `finish` waits for the copies themselves.
  """

  def __init__(self, device: wp.Device) -> None:
    """An uploader to `device`."""
    self.device = device
    self._built: dict[str, wp.Texture2D] = {}

  def finish(self) -> None:
    """Waits for every queued copy (one synchronisation): the textures are then complete for any
    stream, not only for later work on the device's stream."""
    if self.device.is_cuda:
      wp.synchronize_stream(wp.get_stream(self.device))

  def add(self, host: HostTexture) -> CoreTexture:
    """The GPU texture of `host` (registered with its content digest for the bake caches)."""
    texels = host.texels
    texture = self._built.get(texels.digest)
    if texture is None:
      texture = bake_cache.register_texture(_texture(host.levels, texels, self.device), texels.digest)
      self._built[texels.digest] = texture
    return CoreTexture(texture, host.layout, host.size)


def as_done[T, R](
  fn: Callable[[T], R], items: Sequence[T], threads: int, *, abort: Callable[[], None] | None = None
) -> Iterator[tuple[int, R]]:
  """(index, `fn` of the item) of each item, computed on `threads` threads, yielded as they
  complete (a large item does not hold back the others) with at most 2 x `threads` results
  pending (host levels are large): the host side of textures read or built while the caller
  uploads them (`TextureUploader`). When an item or the caller fails (the generator is closed
  early), `abort` runs before the pool is joined: it must unblock workers waiting for what the
  unconsumed results hold (`Staging.close`)."""
  window = 2 * threads
  todo = iter(enumerate(items))
  with concurrent.futures.ThreadPoolExecutor(threads) as pool:
    running: dict[concurrent.futures.Future[R], int] = {}

    def submit() -> None:
      """Fills the window."""
      for i, item in itertools.islice(todo, window - len(running)):
        running[pool.submit(fn, item)] = i

    try:
      submit()
      while running:
        done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
          yield running.pop(future), future.result()
        submit()
    except BaseException:  # GeneratorExit included: the caller stopped on an error
      if abort is not None:
        abort()
      pool.shutdown(wait=False, cancel_futures=True)
      raise


def _host_levels(
  stored: HostLevel, *, srgb: bool, srgb8: bool, ncolor: int, coverage: Coverage | None
) -> tuple[Sequence[HostLevel], TexelFile]:
  """Every host level of a texture and their description (no digest, wrap address): 8-bit sRGB
  color filtered in linear values (`srgb8`), else 8-bit `srgb` color with stored color channels
  (`ncolor` > 0) linearised to float16 first; then coverage-weighted (its first `ncolor`
  channels) or box filtered (`box_chain`)."""
  shapes = level_shapes(stored.shape[1], stored.shape[0])
  levels: Sequence[HostLevel]
  if srgb8:
    levels = srgb8_mip_chain(stored.astype(np.uint8, copy=False), coverage, shapes, ncolor=NCOLOR, nsrgb=NCOLOR)
  elif stored.dtype == np.uint8 and not (srgb and ncolor):  # no sRGB channel stored: exact uint8
    u8 = stored.astype(np.uint8, copy=False)
    levels = box_chain(u8) if coverage is None else coverage_mip_chain(u8, coverage, shapes, ncolor)
  else:
    as_u8 = stored.dtype == np.uint8
    f16 = linear_f16(stored.astype(np.uint8, copy=False), ncolor) if as_u8 else stored.astype(np.float16, copy=False)
    levels = box_chain(f16) if coverage is None else coverage_mip_chain(f16, coverage, shapes, ncolor)
  h, w, c = levels[0].shape
  dtype: Literal["uint8", "float16"] = "uint8" if levels[0].dtype == np.uint8 else "float16"
  return levels, TexelFile("", (h, w, c), dtype, srgb8, (0, 0))


def _texture(levels: Sequence[HostLevel], texels: TexelFile, device: wp.Device) -> wp.Texture2D:
  """The texture of host `levels` described by `texels` (the sRGB decode when `texels.srgb8`)."""
  address = (wp.TextureAddressMode(texels.address[0]), wp.TextureAddressMode(texels.address[1]))
  return MipTexture2D(levels, address=address, device=device, srgb8=texels.srgb8)


def write_entry(path: Path, host: HostTexture, extra: Mapping[str, Json]) -> None:
  """Writes the disk entry of `host` at `path`: its host levels (once per content, in `path`'s
  directory), its layout and the caller's `extra` fields (`texture_cache.write_entry`)."""
  texture_cache.save_levels(path.parent, host.levels, host.texels)
  texture_cache.write_entry(path, host.layout, host.texels, extra)


def read_entry(path: Path) -> tuple[HostTexture, dict[str, Json]] | None:
  """The host texture of the disk entry at `path` (`write_entry`) from its cached host levels, and
  the entry's `extra` fields; None when either is missing. Host work only (thread safe)."""
  entry = texture_cache.read_entry(path)
  levels = None if entry is None else texture_cache.load_levels(path.parent, entry.texels)
  if entry is None or levels is None:
    return None
  return HostTexture(levels, entry.texels, entry.layout), entry.extra


# The `Staging` queue's marker of closed slots.
_CLOSED: Final = -1


@final
class Staging:
  """Pinned host slots texel files are read into (`read_staged`), so the uploader's copies from
  them are DMA on the stream: from pageable memory the CUDA driver first copies every byte into
  its own staging buffer on the uploading thread. A slot is reused once the copies from it are
  done (an event recorded on the device's stream per slot, `release`); its pinned buffer grows
  to the largest texture read into it (pinning costs ~0.3 s per GB, so the slots are few)."""

  def __init__(self, nslot: int, device: wp.Device) -> None:
    """`nslot` slots for uploads to CUDA `device`."""
    self.device = device
    self._buffers: list[wp.array[wp.uint8, [int]] | None] = [None] * nslot
    self._done: list[wp.Event | None] = [None] * nslot
    self._free: queue.SimpleQueue[int] = queue.SimpleQueue()
    for i in range(nslot):
      self._free.put(i)

  def acquire(self, nbytes: int) -> tuple[int, np.ndarray[[int], np.dtype[np.uint8]]]:
    """(index, at least `nbytes` bytes) of a free slot, waiting for one and for the copies last
    made from it.

    Raises:
      RuntimeError: once the slots are closed (`close`).
    """
    i = self._free.get()
    if i == _CLOSED:
      self._free.put(i)  # wakes the next waiter
      raise RuntimeError("texture staging closed: an upload failed")
    done = self._done[i]
    if done is not None:
      wp.synchronize_event(done)
    buffer = self._buffers[i]
    if buffer is None or buffer.shape[0] < nbytes:
      buffer = self._buffers[i] = wp.empty(max(nbytes, 1), dtype=wp.uint8, device="cpu", pinned=True)
    return i, host_copy(buffer)

  def release(self, slot: int) -> None:
    """Returns `slot` once the copies queued so far on the device's stream are done."""
    self._done[slot] = wp.get_stream(self.device).record_event()
    self._free.put(slot)

  def close(self) -> None:
    """Fails every waiting and later `acquire` (an upload failed: the slots held by results never
    consumed are not released, and the readers waiting for them would block the pool's join)."""
    self._free.put(_CLOSED)


def read_staged(path: Path, entry: texture_cache.Entry, staging: Staging) -> tuple[HostTexture, int] | None:
  """(host texture of disk entry `entry` read from `path`'s directory into a staging slot, the
  slot), or None (a missing or damaged texel file: the slot is released). The caller uploads it,
  then releases the slot (`Staging.release`)."""
  slot, buffer = staging.acquire(texture_cache.level_bytes(entry.texels))
  levels = texture_cache.load_levels(path.parent, entry.texels, into=buffer)
  if levels is None:
    staging.release(slot)
    return None
  return HostTexture(levels, entry.texels, entry.layout), slot
