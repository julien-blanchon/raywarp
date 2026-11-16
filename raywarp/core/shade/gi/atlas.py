"""The probe maps as hardware-filtered texture atlases: what the shading's lookup samples.

The blends write the maps in their (row, texel) arrays (`probes.Volume.irradiance` / `distance`);
after each blend `Atlas.refresh` copies them into a float16 2D texture of tiles, one tile of
(n + 2)² texels per map row: the n x n octahedral map plus a one-texel border holding the texels
across the octahedron's edges (RTXGI's border, `fold`). A bilinear texture fetch at the
direction's tile coordinates then filters the 4 texels around it, across the edges too, in one
fetch per map instead of 4 loads and the wrap arithmetic (the hardware's filter weights have 8
fractional bits: < 2^-9 of a texel step from exact bilinear weights).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp.core.pipeline.platform import bake_cache

if TYPE_CHECKING:
  from shape_extensions import IntVar

wp.set_module_options({"enable_backward": False})

# Most texels per atlas side (CUDA's 2D texture limit is larger; this keeps tiles square-ish).
MAX_SIDE: Final = 32768


@wp.func
def fold(
  # In:
  x: int,
  y: int,
  n: int,
) -> int:
  """Row-major index of texel (x, y) of an n x n octahedral map, x / y in [-1, n] (out-of-range
  taps fold across the octahedron's edges)."""
  tx = x
  ty = y
  if tx < 0:
    tx = 0
    ty = n - 1 - ty
  elif tx >= n:
    tx = n - 1
    ty = n - 1 - ty
  if ty < 0:
    ty = 0
    tx = n - 1 - tx
  elif ty >= n:
    ty = n - 1
    tx = n - 1 - tx
  return ty * n + tx


@wp.kernel
def fill_irradiance(
  # In:
  maps: wp.array2d[wp.vec4h],
  n: int,
  tiles: int,
  # Out:
  texels: wp.array3d[wp.float16],
):
  """Texel (bx, by) of map row `row`'s tile (border included)."""
  row, by, bx = wp.tid()
  v = maps[row, fold(bx - 1, by - 1, n)]
  y = (row // tiles) * (n + 2) + by
  x = (row % tiles) * (n + 2) + bx
  for c in range(4):
    texels[y, x, c] = v[c]


@wp.kernel
def fill_distance(
  # In:
  maps: wp.array2d[wp.vec2h],
  n: int,
  tiles: int,
  # Out:
  texels: wp.array3d[wp.float16],
):
  """Texel (bx, by) of map row `row`'s tile (border included)."""
  row, by, bx = wp.tid()
  v = maps[row, fold(bx - 1, by - 1, n)]
  y = (row // tiles) * (n + 2) + by
  x = (row % tiles) * (n + 2) + bx
  texels[y, x, 0] = v[0]
  texels[y, x, 1] = v[1]


@wp.func
def tile_uv(
  # In:
  uv: wp.vec2,
  row: int,
  n: int,
  tiles: int,
) -> wp.vec2:
  """Atlas texel coordinates of octahedral coordinates `uv` ([-1, 1]²) in map row `row`'s tile
  (texel centres at +0.5: interior texel i spans [1 + i, 2 + i) of the tile; unnormalised, so
  a tile's filtering does not depend on the atlas size, i.e. on the number of groups)."""
  x = float((row % tiles) * (n + 2)) + 1.0 + (uv[0] + 1.0) * 0.5 * float(n)
  y = float((row // tiles) * (n + 2)) + 1.0 + (uv[1] + 1.0) * 0.5 * float(n)
  return wp.vec2(x, y)


def layout(rows: int, n: int) -> tuple[int, int, int]:
  """(tiles per atlas row, atlas width, atlas height) [texels] of `rows` tiles of (n + 2)².

  Power-of-two tile counts per side: with power-of-two tiles (the irradiance maps' 8 texels) the
  atlas size is a power of two, so the normalisation of a texel coordinate (Warp's CPU sampler
  divides by the size) is exact and a tile filters the same whatever the number of groups.
  """
  side = n + 2
  tiles = 1 << (max(1, min(rows, math.ceil(math.sqrt(rows)), MAX_SIDE // side)).bit_length() - 1)
  height = 1 << (math.ceil(rows / tiles) - 1).bit_length()
  return tiles, tiles * side, height * side


@final
class Atlas:
  """A map array's texture atlas and its staging texels (`refresh` copies the maps in).

  Args:
    rows: map rows (tiles); n: texels per map side; channels: 4 (irradiance) or 2 (distance).
  """

  def __init__(self, rows: int, n: int, channels: int, device: wp.Device) -> None:
    """Allocates the staging texels and the texture."""
    self.n, self.channels, self.device = n, channels, device
    self.tiles, width, height = layout(rows, n)
    self.texels = wp.zeros((height, width, channels), dtype=wp.float16, device=device)
    texture = wp.Texture2D(
      self.texels,
      filter_mode=wp.TextureFilterMode.LINEAR,
      address_mode=wp.TextureAddressMode.CLAMP,
      normalized_coords=False,
      device=device,
    )
    # Derived from the maps (hashed as arrays by the bake keys): named by its layout only.
    self.texture = bake_cache.register_texture(texture, f"gi-atlas:{self.texels.shape}")

  @property
  def nbytes(self) -> int:
    """Device memory of the staging texels and the texture."""
    return 2 * int(self.texels.capacity)

  def refresh[NRow: IntVar, NTexel: IntVar](
    self, maps: wp.array[wp.vec4h, [NRow, NTexel]] | wp.array[wp.vec2h, [NRow, NTexel]]
  ) -> None:
    """Copies `maps` into the atlas (tiles and borders), then into the texture."""
    kernel = fill_irradiance if self.channels == 4 else fill_distance
    side = self.n + 2
    wp.launch(
      kernel, dim=(maps.shape[0], side, side), inputs=[maps, self.n, self.tiles], outputs=[self.texels], device=self.device
    )
    self.texture.copy_from(self.texels)
