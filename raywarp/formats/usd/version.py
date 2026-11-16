"""The USD converter's code version: `CONVERTER` (its output version) and the digest of raywarp's
MDL library, which the `.rayw` cache key (`api.cache_key`) and the MDL conversion cache
(`mdl.memo`) read.
"""

from __future__ import annotations

import functools
from typing import Final

from raywarp.cache.code_version import code_digest
from raywarp.formats.usd.options import MDL_LIBRARY

# The converter's output version (part of the cache key): bumped by every change that changes what
# a stage converts to, wherever it is (raywarp.formats.usd / scene, the core modules the build
# imports such as `raywarp._base.types`, the MDL library); the converter golden test compares
# fixture conversions with a checked-in golden and fails until it is bumped. A refactor
# that keeps the output keeps every cache and prepared scene.
# 9: data textures with alpha downscaled unpremultiplied; lights' visible_in_primary_ray; a
# surface output through a NodeGraph resolves to its shader (and the other arbitrary-USD fixes).
# 10: `.mtlx` documents a stage references are reported dropped, no longer read (their materials
# and looks unbound); `usd.Options.mtlx_look` removed.
# 11: `usd.Options.cone_segments` removed (no longer in the scene's recorded options).
# 12: images of equal content stored once (`Images.offset` shared; with `.rayw` VERSION 14).
# 13: an MDL parameter without a default takes its type's default-constructed value (SimPBR road
# markings were the error material).
# 14: PointInstancer orientations normalised (USD composes a rounded `quath` unnormalised: a ~1e-3
# stretch that made scattered instances mesh copies at upload).
# 15: MDL world lookups of incommensurate periods bake over the finest world window, read through
# the world projection (SimPBR_Road's asphalt was stretched over the road tile's UV square).
CONVERTER: Final = 15


@functools.cache
def mdl_library_digest() -> str:
  """The content digest of raywarp's MDL library (`MDL_LIBRARY`: modules the build reads, not
  recorded as dependencies: they sit at another path on every machine)."""
  files = sorted(p for p in MDL_LIBRARY.rglob("*") if p.is_file())
  return code_digest((p.relative_to(MDL_LIBRARY).as_posix(), p.read_bytes()) for p in files)
