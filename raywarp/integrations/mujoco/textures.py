"""MuJoCo 2D textures -> core textures, color space per (texture, role) over rendered geoms.

Texels are read with `texels`: a texture's payload in `tex_data`, or, for a by-reference texture
(MuJoCo >= 3.17 `embed="false"` / `<compiler passthrough="textures"/>`: `tex_adr` -1, width and
height 0), decoded from its file by `mujoco.mj_loadTexture`.

Color space: an explicit `colorspace` wins; "auto" is sRGB where the texture is bound to a
color role (rgb, rgba, emissive) and linear where bound to a data role (normal, ORM, opacity, ...)
by a material of a rendered geom (every row of `m.mat_texid`). A texture bound both ways gets two
GPU textures: its color one at its MuJoCo id, its linear copy after the model's textures; the
material table then remaps the data roles (`TexturePack.material_texture`), else `m.mat_texid`
itself is the table (aliased, live).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import os
import sys
from pathlib import Path
from typing import Final, NewType, final

import mujoco
import numpy as np
import warp as wp

from raywarp._base import types as core_types
from raywarp._base.types import COLOR_ROLES, NTEXROLE, TEX_LAYOUT_IDENTITY, TexRole
from raywarp.cache.code_version import code_digest
from raywarp.cache.paths import cache_dir
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.pipeline.textures import formats
from raywarp.core.pipeline.textures import uploader as core_textures
from raywarp.core.pipeline.textures.formats import DEFAULT_STORAGE, Level, TextureStorage
from raywarp.core.pipeline.textures.uploader import (
  CoreTexture,
  HostTexture,
  TextureUploader,
  as_done,
  host_texture,
  placeholder_texture,
)
from raywarp.core.post.grading import color as shade_color
from raywarp.scene.images import mips

type TexIds = np.ndarray[[int, int], np.dtype[np.int32]]

# A row of MuJoCo's texture tables (`mjm.tex_*`, `mat_texid`, `light_texid`); -1: none.
MjTextureId = NewType("MjTextureId", int)
NO_MJ_TEXTURE: Final = MjTextureId(-1)

# An opaque 8-bit alpha.
OPAQUE_U8: Final = 255
# Threads decoding / preparing the texels (`upload_textures`; uploaded as each completes).
PREPARE_THREADS: Final = min(8, os.cpu_count() or 1)

_DATA_ROLES: Final = tuple(r for r in range(NTEXROLE) if r not in COLOR_ROLES)
# The disk entries' format (`_Entry`) and the address mode of MuJoCo textures (they wrap).
_ENTRY_FORMAT: Final = 2
_WRAP: Final = (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.WRAP)

# An explicit `colorspace` -> whether the texels are sRGB ("auto": by the role read, `srgb`).
_EXPLICIT_SRGB: Final = {
  int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB): True,
  int(mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR): False,
}


def texels(mjm: mujoco.MjModel, tex_id: MjTextureId) -> Level[np.uint8]:
  """(height, width, nchannel) uint8 texels of texture `tex_id` (by-reference ones decoded)."""
  return loaded(mjm, tex_id)[0]


def loaded(mjm: mujoco.MjModel, tex_id: MjTextureId) -> tuple[Level[np.uint8], int]:
  """((height, width, nchannel) uint8 texels, `mjtColorSpace`) of texture `tex_id`.

  A by-reference texture is decoded with the color space MuJoCo resolves from the file, as
  compiling would (a PNG without an sRGB chunk is linear: `tex_colorspace` stays "auto" in the
  model), so it reads like the same texture embedded."""
  adr = int(mjm.tex_adr[tex_id])
  if adr < 0:
    pixels, space = mujoco.mj_loadTexture(mjm, tex_id)  # pyrefly: ignore[missing-attribute]  # MuJoCo >= 3.17; mujoco-stubs lag
    return pixels, int(space)
  h, w, nc = int(mjm.tex_height[tex_id]), int(mjm.tex_width[tex_id]), int(mjm.tex_nchannel[tex_id])
  return mjm.tex_data[adr : adr + h * w * nc].reshape(h, w, nc), int(mjm.tex_colorspace[tex_id])


def srgb(colorspace: int, *, color: bool) -> bool:
  """Whether a texture of `colorspace` (`mjtColorSpace`, resolved: `loaded`) read in a color
  (`color`) or data role holds sRGB-encoded values ("auto": by the role)."""
  return _EXPLICIT_SRGB.get(colorspace, color)


@final
@dataclasses.dataclass(frozen=True)
class TexturePack:
  """The GPU textures of a MuJoCo scene.

  Attributes:
    textures: core texture id -> wp.Texture2D (MuJoCo ids first, then the linear copies).
    sizes: (ntexture, 2) stored (width, height).
    layouts: (ntexture,) texel layouts (`types.TEX_LAYOUT_*`).
    material_texture: (nrow, nmat, NTEXROLE) core texture ids, or None when they are MuJoCo's
      (`m.mat_texid` is then the aliased table).
  """

  textures: list[wp.Texture2D]
  sizes: np.ndarray[[int, 2], np.dtype[np.float32]]
  layouts: np.ndarray[[int], np.dtype[np.int32]]
  material_texture: np.ndarray[[int, int, int], np.dtype[np.int32]] | None


def _bound(texid: TexIds, roles: tuple[int, ...], ntex: int) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """(ntex,) whether each texture is bound to one of `roles` in `texid` (rows x NTEXROLE)."""
  out = np.zeros(ntex, bool)
  ids = texid[:, list(roles)].ravel()
  out[ids[ids >= 0]] = True
  return out


def rgba_texels(data: Level[np.uint8]) -> Level[np.uint8]:
  """(h, w, 4) uint8 of (h, w, nchannel) texels: grey broadcast to RGB, missing alpha opaque."""
  h, w, nc = data.shape
  rgba = np.full((h, w, 4), OPAQUE_U8, dtype=np.uint8)
  if nc == 1:
    rgba[..., :3] = data
  else:
    rgba[..., :nc] = data
  return rgba


def _coverage(rgba: Level[np.uint8], *, weighted: bool) -> np.ndarray[[int, int], np.dtype[np.float32]] | None:
  """The alpha weighting the color mips of a base-color texture with alpha, else None."""
  if not weighted or bool(np.all(rgba[..., 3] == OPAQUE_U8)):
    return None
  return rgba[..., 3].astype(np.float32) / 255.0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Request:
  """One GPU texture of a MuJoCo texture: its texels and how they are stored (`host_texture`)."""

  srgb: bool
  weighted: bool  # coverage-weighted color mips (a base color with alpha, `_coverage`)


@functools.cache
def _entry_code() -> str:
  """`code_digest` of the modules deciding what a texture's texels become (this one's padding and
  coverage, the formats, mips and layouts)."""
  modules = (sys.modules[__name__], core_textures, formats, shade_color, mips, core_types)
  return code_digest((f"{m.__name__}.py", Path(m.__file__).read_bytes()) for m in modules if m.__file__ is not None)


def _entry_key(texels: Level[np.uint8], request: _Request, storage: TextureStorage) -> str:
  """The digest of a texture's every input: its texels (content and shape), how they are stored
  and the code making them (`_entry_code`)."""
  digest = hashlib.blake2b(np.ascontiguousarray(texels).data, digest_size=20)
  head = (_ENTRY_FORMAT, _entry_code(), texels.shape, repr(storage), request.srgb, request.weighted)
  digest.update(repr(head).encode())
  return digest.hexdigest()


def upload_textures(
  mjm: mujoco.MjModel,
  mat_texid: np.ndarray[[int, int, int], np.dtype[np.int32]],
  materials: np.ndarray[[int], np.dtype[np.int64]],
  *,
  storage: TextureStorage = DEFAULT_STORAGE,
  device: wp.Device,
) -> TexturePack:
  """Mipmapped textures on `device` of the 2D textures the rendered `materials` bind in any row of
  `mat_texid` ((nrow, nmat, NTEXROLE), `m.mat_texid`), in their color spaces (module docstring),
  stored as `storage`. Unbound textures are 1 x 1 placeholders. A texture is remembered on disk by
  the digest of its texels and storage (`<cache dir>/textures/mjcf-<key>.json`,
  `textures.write_entry`): a repeat upload reads its host levels, without padding or filtering a
  texel. The texels are read (by-reference ones decoded), keyed, and on a miss padded to RGBA
  and built, on a thread pool; the textures keep their texture order."""
  ntex = mjm.ntex
  rows = mat_texid[:, materials].reshape(-1, NTEXROLE)
  color, data = _bound(rows, COLOR_ROLES, ntex), _bound(rows, _DATA_ROLES, ntex)
  base = _bound(rows, (int(TexRole.RGB), int(TexRole.RGBA)), ntex) & ~_bound(rows, (int(TexRole.OPACITY),), ntex)
  is_2d = np.asarray(mjm.tex_type) == int(mujoco.mjtTexture.mjTEXTURE_2D) if ntex else np.zeros(0, bool)
  wanted = [MjTextureId(int(t)) for t in np.flatnonzero((color | data) & is_2d)]
  stored = storage.on(srgb8=device_caps(device).device_textures)
  root = cache_dir()

  def requests(t: MjTextureId) -> tuple[Level[np.uint8], list[_Request], list[str]]:
    """(texels, the texture's requests: its own then its linear copy, their entry keys)."""
    pixels, colorspace = loaded(mjm, t)
    space = srgb(colorspace, color=bool(color[t]))
    own = [_Request(srgb=space, weighted=bool(base[t]))]
    if data[t] and space != srgb(colorspace, color=False):
      own.append(_Request(srgb=False, weighted=False))
    return pixels, own, [_entry_key(pixels, r, stored) for r in own]

  def built(t: MjTextureId) -> list[HostTexture]:
    """The texture's host textures (read or built on a thread): from their entries, else from
    the texels."""
    pixels, own, keys = requests(t)
    out = []
    rgba: Level[np.uint8] | None = None
    for request, key in zip(own, keys, strict=True):
      path = None if root is None else root / "textures" / f"mjcf-{key}.json"
      found = None if path is None else core_textures.read_entry(path)
      if found is not None:
        host = found[0]
      else:
        rgba = rgba_texels(pixels) if rgba is None else rgba
        coverage = _coverage(rgba, weighted=request.weighted)
        host = host_texture(rgba, srgb=request.srgb, address=_WRAP, coverage=coverage, storage=storage, device=device)
        if path is not None:
          core_textures.write_entry(path, host, {})
      out.append(host)
    return out

  uploader = TextureUploader(device)
  uploaded = {wanted[i]: [uploader.add(host) for host in hosts] for i, hosts in as_done(built, wanted, PREPARE_THREADS)}
  uploader.finish()
  return _pack(ntex, uploaded, mat_texid, device)


def _pack(
  ntex: int,
  uploaded: dict[MjTextureId, list[CoreTexture]],
  mat_texid: np.ndarray[[int, int, int], np.dtype[np.int32]],
  device: wp.Device,
) -> TexturePack:
  """The `TexturePack` of the `uploaded` textures (each its own texture, then its linear copy if
  any): a placeholder at every other MuJoCo id, the linear copies after the model's textures."""
  placeholder = CoreTexture(placeholder_texture(device), TEX_LAYOUT_IDENTITY, (1.0, 1.0))
  textures: list[CoreTexture] = []
  twins: dict[int, int] = {}
  extra: list[CoreTexture] = []  # linear copies
  for t in map(MjTextureId, range(ntex)):
    own = uploaded.get(t)
    if own is None:
      textures.append(placeholder)
      continue
    textures.append(own[0])
    if len(own) > 1:
      twins[t] = ntex + len(extra)
      extra.append(own[1])
  textures += extra
  return TexturePack(
    [c.texture for c in textures],
    np.array([c.size for c in textures], np.float32).reshape(-1, 2),
    np.array([c.layout for c in textures], np.int32),
    _remap(mat_texid, twins),
  )


def _remap(
  mat_texid: np.ndarray[[int, int, int], np.dtype[np.int32]], twins: dict[int, int]
) -> np.ndarray[[int, int, int], np.dtype[np.int32]] | None:
  """`mat_texid` with the data roles of textures in `twins` pointing at their linear copies (None:
  no copy)."""
  if not twins:
    return None
  out = mat_texid.copy()
  data = out[..., list(_DATA_ROLES)]
  for t, twin in twins.items():
    data[data == t] = twin
  out[..., list(_DATA_ROLES)] = data
  return out
