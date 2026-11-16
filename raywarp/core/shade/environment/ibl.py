"""Helpers of lighting by environment maps (image-based lighting): GGX importance sampling and
the precomputed DFG table of the split-sum approximation (Filament / Karis 2013).

`hammersley` / `importance_sample_ggx` sample the GGX lobe of the prefiltered stacks (`envmap.py`;
the offline generator of the tables uses the same sampling). The DFG and cloth tables are shipped
as package data, uploaded as textures and sampled per hit (`sample_dfg`, `sample_cloth`).
"""

from __future__ import annotations

import functools
import importlib.resources
from typing import TYPE_CHECKING, Final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform import bake_cache

if TYPE_CHECKING:
  from raywarp._base.dims import AnyRank

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# The DFG table shipped in the package data (`dfg_table`), its size per axis and samples per texel.
DFG_FILE: Final = "dfg_lut.npz"
DFG_LUT_SIZE: Final = 128
DFG_LUT_SAMPLES: Final = 65536
# Samples per texel of the cloth (Charlie / Neubelt) table (Filament cmgen's 4096).
CLOTH_LUT_SAMPLES: Final = 4096


# Bit-reversal masks as typed constants: a bare literal above INT32_MAX would first be emitted as an
# int32 by Warp's code generator (a narrowing conversion the C++ compiler warns about).
_MASK_1: Final = wp.constant(wp.uint32(0x55555555))
_MASK_1_HI: Final = wp.constant(wp.uint32(0xAAAAAAAA))
_MASK_2: Final = wp.constant(wp.uint32(0x33333333))
_MASK_2_HI: Final = wp.constant(wp.uint32(0xCCCCCCCC))
_MASK_4: Final = wp.constant(wp.uint32(0x0F0F0F0F))
_MASK_4_HI: Final = wp.constant(wp.uint32(0xF0F0F0F0))
_MASK_8: Final = wp.constant(wp.uint32(0x00FF00FF))
_MASK_8_HI: Final = wp.constant(wp.uint32(0xFF00FF00))


@wp.func
def hammersley(
  # In:
  i: wp.int32,
  inv_n: wp.float32,
) -> wp.vec2:
  """Point `i` of the Hammersley set of 1 / inv_n points (radical inverse in base 2)."""
  bits = wp.uint32(i)
  bits = (bits << wp.uint32(16)) | (bits >> wp.uint32(16))
  bits = ((bits & _MASK_1) << wp.uint32(1)) | ((bits & _MASK_1_HI) >> wp.uint32(1))
  bits = ((bits & _MASK_2) << wp.uint32(2)) | ((bits & _MASK_2_HI) >> wp.uint32(2))
  bits = ((bits & _MASK_4) << wp.uint32(4)) | ((bits & _MASK_4_HI) >> wp.uint32(4))
  bits = ((bits & _MASK_8) << wp.uint32(8)) | ((bits & _MASK_8_HI) >> wp.uint32(8))
  return wp.vec2(wp.float32(i) * inv_n, wp.float32(bits) * wp.static(1.0 / 4294967296.0))


@wp.func
def importance_sample_ggx(
  # In:
  u: wp.vec2,
  alpha: wp.float32,
) -> wp.vec3:
  """Half vector around +z distributed as D(h) (h · n)."""
  a2 = alpha * alpha
  phi = wp.static(2.0 * wp.pi) * u[0]
  cos_theta2 = (1.0 - u[1]) / (1.0 + (a2 - 1.0) * u[1])
  cos_theta = wp.sqrt(cos_theta2)
  sin_theta = wp.sqrt(wp.max(1.0 - cos_theta2, 0.0))
  return wp.vec3(sin_theta * wp.cos(phi), sin_theta * wp.sin(phi), cos_theta)


@functools.cache
def dfg_table() -> np.ndarray[[int, int, 2], np.dtype[np.float32]]:
  """Filament's multiscatter DFG table (DFG_LUT_SIZE, DFG_LUT_SIZE, 2) indexed [perceptual
  roughness, NoV], shipped as package data (`tools/make_dfg_lut.py` computes it).

  Channel x = ∫ V·Fc, channel y = ∫ V, so the specular albedo is mix(x, y, f0) and the
  multiscatter compensation is 1 + f0 (1 / y − 1).
  """
  return _packaged("dfg")


@functools.cache
def cloth_table() -> np.ndarray[[int, int], np.dtype[np.float32]]:
  """The sheen lobe's directional albedo int D_Charlie V_Neubelt NoL (DFG_LUT_SIZE, DFG_LUT_SIZE)
  indexed [sqrt(Charlie α), NoV] (`tools/make_dfg_lut.py`): Filament's cloth DG channel,
  kept in its own texture so that kernels without the fuzz lobe sample the same two-channel
  table."""
  return _packaged("cloth")


def _packaged(name: str) -> AnyRank[np.float32]:
  """Array `name` of the package's table file (`DFG_FILE`), contiguous float32."""
  with importlib.resources.as_file(importlib.resources.files("raywarp.data") / DFG_FILE) as path:
    return np.ascontiguousarray(np.load(path)[name], np.float32)


def cloth_texture(device: wp.Device) -> wp.Texture2D:
  """The cloth DG table as a bilinear 1-channel texture on `device`, sampled at (NoV, sqrt(α))."""
  return bake_cache.table_texture("cloth_dg", cloth_table, device)


@wp.func
def sample_cloth(
  # Model:
  cloth_lut: wp.Texture2D,
  # In:
  nov: wp.float32,
  alpha: wp.float32,
) -> wp.float32:
  """The sheen lobe's directional albedo E_fuzz(NoV, α) (`cloth_table`), bilinear."""
  return wp.texture_sample(cloth_lut, wp.vec2(nov, wp.sqrt(alpha)), dtype=float)


def dfg_texture(device: wp.Device) -> wp.Texture2D:
  """The DFG table as a bilinear 2-channel texture on `device`, sampled at (NoV, perceptual
  roughness) (cached)."""
  return bake_cache.table_texture("dfg", dfg_table, device)


@wp.func
def sample_dfg(
  # Model:
  dfg_lut: wp.Texture2D,
  # In:
  nov: wp.float32,
  perceptual_roughness: wp.float32,
) -> wp.vec2:
  """(int V Fc, int V) at (NoV, perceptual roughness), bilinear."""
  return wp.texture_sample(dfg_lut, wp.vec2(nov, perceptual_roughness), dtype=wp.vec2)
