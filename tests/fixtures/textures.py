"""One material texture built and uploaded in a test (`core.pipeline.textures.uploader.host_texture`
then a `TextureUploader` of its own)."""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.pipeline.textures import uploader as textures
from raywarp.core.pipeline.textures.cache import HostLevel
from raywarp.core.pipeline.textures.formats import DEFAULT_STORAGE, Coverage, TextureStorage

# Wrapping in u and v (`host_texture`'s default).
WRAP: Final = (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.WRAP)


def make_texture(  # noqa: PLR0913 (keyword-only build options)
  rgba: HostLevel,
  *,
  srgb: bool,
  address: textures.Address = WRAP,
  coverage: Coverage | None = None,
  storage: TextureStorage = DEFAULT_STORAGE,
  device: wp.Device,
) -> textures.CoreTexture:
  """The mipmapped texture on `device` of an (h, w, 4) image: `host_texture` (its arguments)
  uploaded."""
  host = textures.host_texture(rgba, srgb=srgb, address=address, coverage=coverage, storage=storage, device=device)
  uploader = textures.TextureUploader(device)
  texture = uploader.add(host)
  uploader.finish()
  return texture
