"""The disk cache of material textures: host levels by content, entries by every input.

A texture's host levels (`TexelFile`: every level of its mip chain, down to 1 x 1) are written
once per content to `<cache>/<digest>-t<TEXEL_FORMAT>-<code>.npy` (one flat array, level after
level; the digest of the stored texels and sampling options, the code digest of the modules making
the levels: `uploader`, `formats`, the sRGB table). A caller's entry next to it
(`write_entry`: the texture pool's, MuJoCo's, named by the texture's every input) holds the
texture's layout, its `TexelFile` and the caller's fields, so a repeat upload reads the levels
(`read_entry`, `load_levels`: one read per file) without reading, converting, packing, filtering or
hashing a source texel. Writes are atomic (a writer's own temporary file renamed into place); a
damaged file warns and is rebuilt.
"""

from __future__ import annotations

import functools
import importlib
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, cast, final

import numpy as np

from raywarp.cache.code_version import code_digest
from raywarp.cache.paths import write_atomic
from raywarp.core.pipeline.textures import formats
from raywarp.core.pipeline.textures.formats import Level, level_shapes
from raywarp.scene.images import mips
from raywarp.scene.model.types import Json

# A host image level: (h, w, channels) uint8 / float16 texels.
type HostLevel = Level[np.uint8] | Level[np.float16]

# Bumped when a texel file's layout or an entry's fields change.
TEXEL_FORMAT: Final = 2
# A content digest (hexadecimal: it names a texel file).
_HEX: Final = re.compile(r"[0-9a-f]+")
_DTYPES: Final = {"uint8": np.uint8, "float16": np.float16}
_ITEMSIZE: Final = {"uint8": 1, "float16": 2}


@final
class TexelFile(NamedTuple):
  """A texture's host levels in the disk cache: what an entry rebuilds the texture from.

  Attributes:
    digest: the content digest of its stored texels and sampling options (`host_texture`; the
      bake caches' texture digest, `bake_cache.register_texture`).
    shape: level 0's (h, w, channels).
    dtype: "uint8" or "float16".
    srgb8: 8-bit sRGB RGBA or one-channel levels, decoded by the texture unit before filtering
      (CUDA).
    address: (u, v) address modes (`wp.TextureAddressMode` values).
  """

  digest: str
  shape: tuple[int, int, int]
  dtype: Literal["uint8", "float16"]
  srgb8: bool
  address: tuple[int, int]


@final
class Entry(NamedTuple):
  """A disk entry: the texture's texel layout, its host levels and the caller's fields."""

  layout: int
  texels: TexelFile
  extra: dict[str, Json]


@functools.cache
def _code() -> str:
  """`code_digest` of the modules deciding a texture's host levels (`uploader` imported here:
  it imports this module)."""
  modules = (importlib.import_module("raywarp.core.pipeline.textures.uploader"), formats, mips)
  return code_digest((f"{m.__name__}.py", Path(m.__file__).read_bytes()) for m in modules if m.__file__ is not None)


def _texel_path(cache: Path, digest: str) -> Path:
  """The texel file of content `digest` in `cache` (keyed by the code making the levels too)."""
  return cache / f"{digest}-t{TEXEL_FORMAT}-{_code()}.npy"


def save_levels(cache: Path, levels: Sequence[HostLevel], texels: TexelFile) -> None:
  """Writes `levels` (one flat array, level after level) to their texel file, unless it exists."""
  path = _texel_path(cache, texels.digest)
  if path.exists():
    return
  flat = np.concatenate([lv.reshape(-1) for lv in levels])
  write_atomic(path, lambda f: np.save(f, flat, allow_pickle=False), "texture cache")


def level_bytes(texels: TexelFile) -> int:
  """Bytes of every host level of `texels` (its texel file's payload)."""
  h, w, c = texels.shape
  return sum(lw * lh * c for lw, lh, _ in level_shapes(w, h)) * _ITEMSIZE[texels.dtype]


def load_levels(
  cache: Path, texels: TexelFile, into: np.ndarray[[int], np.dtype[np.uint8]] | None = None
) -> list[HostLevel] | None:
  """The host levels of `texels` (views of one array read from the file; read into the uint8
  buffer `into` when given: a pinned staging slot of at least `level_bytes`), or None (a miss, a
  damaged file or one of another size: the caller builds and rewrites it).

  One read of the whole file, not a memory map: the map's page faults made the warm texture
  upload of a 400-texture interior (1.4 GB of levels, 32 upload threads, L4) 0.95-1.02 s against
  0.73-0.74 s."""
  path = _texel_path(cache, texels.digest)
  h, w, c = texels.shape
  shapes = level_shapes(w, h)
  total = sum(lw * lh * c for lw, lh, _ in shapes)
  dtype = _DTYPES[texels.dtype]
  try:
    with path.open("rb") as f:
      version = np.lib.format.read_magic(f)
      header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
      shape, fortran, stored = header(f)
      if shape != (total,) or fortran or stored != dtype:
        return None
      if into is None:
        flat = np.fromfile(f, dtype=dtype, count=total)
      else:
        target = into[: total * _ITEMSIZE[texels.dtype]]
        flat = target.view(dtype) if f.readinto(memoryview(target)) == target.nbytes else np.empty(0, dtype)
  except FileNotFoundError:
    return None
  except (OSError, ValueError, EOFError) as e:
    warnings.warn(f"raywarp cache: unreadable {path} ({e}); building the texture again", RuntimeWarning, stacklevel=2)
    return None
  if flat.shape != (total,):  # a truncated file
    return None
  levels: list[HostLevel] = []
  at = 0
  for lw, lh, _ in shapes:
    levels.append(cast("HostLevel", flat[at : at + lw * lh * c].reshape(lh, lw, c)))
    at += lw * lh * c
  return levels


def write_entry(path: Path, layout: int, texels: TexelFile, extra: Mapping[str, Json]) -> None:
  """Writes the entry of a texture (texel `layout`, its levels `texels` in `path`'s directory) and
  the caller's `extra` fields at `path`."""
  body = {**extra, "format": TEXEL_FORMAT, "layout": layout, "texels": list(texels)}
  write_atomic(path, lambda f: f.write(json.dumps(body).encode()), "texture cache")


def read_entry(path: Path) -> Entry | None:
  """The entry at `path`, or None (a miss, another format; a damaged entry warns)."""
  try:
    body = json.loads(path.read_bytes())
    if body.pop("format") != TEXEL_FORMAT:
      return None
    return Entry(int(body.pop("layout")), _texel_file(body.pop("texels")), body)
  except FileNotFoundError:
    return None
  except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
    warnings.warn(f"raywarp cache: unreadable {path} ({e}); building the texture again", RuntimeWarning, stacklevel=2)
    return None


def _ints(value: Json, n: int) -> tuple[int, ...]:
  """`n` integers of a JSON list.

  Raises:
    ValueError: for another value.
  """
  if not isinstance(value, list) or len(value) != n or not all(isinstance(v, int) for v in value):
    raise ValueError(f"expected {n} integers, got {value!r}")
  return tuple(v for v in value if isinstance(v, int))


def _texel_file(fields: Json) -> TexelFile:
  """The `TexelFile` of an entry's `texels` list.

  Raises:
    ValueError: for a field of the wrong kind.
  """
  if not isinstance(fields, list) or len(fields) != len(TexelFile._fields):
    raise ValueError(f"texels {fields!r}")
  digest, shape, dtype, srgb8, address = fields
  if not (isinstance(digest, str) and _HEX.fullmatch(digest)) or not isinstance(srgb8, bool):
    raise ValueError(f"texels {fields!r}")
  if dtype not in ("uint8", "float16"):
    raise ValueError(f"texels {fields!r}")
  h, w, c = _ints(shape, 3)
  u, v = _ints(address, 2)
  return TexelFile(digest, (h, w, c), "uint8" if dtype == "uint8" else "float16", srgb8, (u, v))
