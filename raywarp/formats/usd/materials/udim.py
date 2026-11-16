"""UDIM texture sets (`<UDIM>` in a texture's file path) as one atlas texture.

Tile 1001 + u + 10 v covers the UV square [u, u + 1] x [v, v + 1]. The tiles found are laid out
as a `cols` x `rows` grid (tile (u, v) at column u, row v counted from the bottom: the image's v
axis), so that the atlas read at (u / cols, v / rows) is the tile set read at (u, v): one UV
scale, no per-tile lookup in the kernels. Missing tiles of the grid are filled with the
fallback color (no UV of the mesh reads them). Every tile is resized to the first one's size
(sets have one tile size in practice). Filtering across a tile seam blends the neighbour tile
(at most half a texel at the finest mip); mips coarser than one tile mix tiles (UDIM sets are
authored with a gutter around each island for that).
"""

from __future__ import annotations

import io
from typing import Final

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdShade

from raywarp.scene.images import files

# Tiles per UDIM row (1001..1010 on v = 0).
_ROW: Final = 10
_FIRST: Final = 1001

# A tile set's atlas: (rows h, cols w, 4) 8-bit RGBA texels.
type Atlas = np.ndarray[[int, int, 4], np.dtype[np.uint8]]


def is_template(path: str) -> bool:
  """Whether an authored file path is a UDIM set's template."""
  return bool(UsdShade.UdimUtils.IsUdimIdentifier(path))


def template_layer(attr: Usd.Attribute) -> Sdf.Layer | None:
  """The layer whose authored file value anchors attribute `attr`'s UDIM path (None: none)."""
  spec = next((s for s in attr.GetPropertyStack(Usd.TimeCode.Default()) if s.HasDefaultValue()), None)
  return spec.layer if spec is not None else None


def tiles(template: str, layer: Sdf.Layer) -> dict[tuple[int, int], str]:
  """(u, v) tile -> resolved file of the UDIM set `template` authored in `layer` (empty: none)."""
  out: dict[tuple[int, int], str] = {}
  for path, tile in UsdShade.UdimUtils.ResolveUdimTilePaths(template, layer):
    k = int(tile) - _FIRST
    if k >= 0:
      out[(k % _ROW, k // _ROW)] = path
  return out


def atlas(found: dict[tuple[int, int], str], fill: tuple[float, ...]) -> tuple[Atlas, int, int]:
  """((rows h, cols w, 4) uint8 atlas, cols, rows) of the tiles `found` (`tiles`); missing grid
  cells are `fill` (rgba in [0, 1], stored as is)."""
  cols = 1 + max(u for u, _ in found)
  rows = 1 + max(v for _, v in found)
  decoded = {k: Image.open(io.BytesIO(files.read(p))).convert("RGBA") for k, p in found.items()}
  w, h = next(iter(decoded.values())).size
  pad = np.clip(np.round(np.asarray(fill, np.float64)[:4] * 255.0), 0, 255).astype(np.uint8)
  out = np.empty((rows * h, cols * w, 4), np.uint8)
  out[:] = pad
  for (u, v), img in decoded.items():
    tile = np.asarray(img if img.size == (w, h) else img.resize((w, h), Image.Resampling.LANCZOS))
    top = (rows - 1 - v) * h
    out[top : top + h, u * w : (u + 1) * w] = tile
  return out, cols, rows
