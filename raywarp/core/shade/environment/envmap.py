"""Environment maps (HDR skies that light the scene and show behind it): their upload to the GPU,
the precomputed blurred copies used for rough reflections and diffuse light, and their lookups.

Environment maps are equirectangular images in the layout of `shade/environment/latlong.py`
(Blender's: row 0 the zenith, centre column +x, u = 0.25 +y). A per-world yaw rotates the map about
+z.

For each map we keep
  * the radiance as a mipmapped 2D texture (background and sharp reflections),
  * a GGX-prefiltered stack as a 3D texture whose depth axis is perceptual roughness
    (Karis 2013, with filtered importance sampling: Křivánek & Colbert 2008),
  * 9 spherical-harmonic coefficients of the diffuse irradiance, premultiplied by the cosine
    lobe and 1/π (Ramamoorthi & Hanrahan 2001), so they evaluate to Lambertian radiance.

The textures are float16 (largest finite value 65504) while a physical sky's sun disc reaches
~1e8: each map is stored divided by its `scale` (peak at most MAX_STORED_RADIANCE), and the
lookups multiply it back (`EnvironmentTextures.scale`, applied with the per-world intensity).
All three representations hold the same scaled radiance, so they stay consistent.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import host_copy, launch_bands
from raywarp.core.pipeline.textures.uploader import MipTexture2D, box_chain
from raywarp.core.shade.environment import ibl, latlong
from raywarp.core.shade.materials import brdf
from raywarp.scene.images.latlong import latlong_directions, texel_solid_angles

if TYPE_CHECKING:
  from shape_extensions import IntVar
  from warp.shape_checks import expect_shape

  from raywarp._base.dims import AnyRank, Floats

# A linear radiance image, (height, width, 3) float32 (what `scene.images.hdr.read_hdr` returns).
type HdrImage[H: IntVar, W: IntVar] = np.ndarray[[H, W, 3], np.dtype[np.float32]]
# An image as accepted on input: (height, width, channels >= 3) linear radiance, floats (extra
# channels are ignored; 8-bit images are display-encoded, not radiance: `_radiance` rejects them).
type Image = Floats[[int, int, int]] | np.ndarray[[int, int, int], np.dtype[np.float16]]

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Prefiltered stack: width × height × number of roughness levels.
PREFILTER_WIDTH: Final = 256
PREFILTER_LEVELS: Final = 8
PREFILTER_SAMPLES: Final = 512
# Radiance textures larger than this are downsampled (box filter) before upload.
MAX_RADIANCE_WIDTH: Final = 2048
# Rank and fewest channels of an input map (height, width, RGB[A...]).
_MAP_RANK: Final = 3
_MIN_CHANNELS: Final = 3
# Side of the box filter of `downsample`.
_BOX: Final = 2
# Narrowest map: the prefiltered stack is (width // 2) rows high.
MIN_WIDTH: Final = 2
# Largest texel stored in the float16 textures (headroom below 65504 for the bilinear and
# prefilter sums); brighter maps are stored divided by a scale.
MAX_STORED_RADIANCE: Final = 30000.0
# Floors of the prefilter's GGX alpha (level 0: a mirror lobe), of a sample's pdf and of the
# accumulated sample weight (a texel whose samples all fall under the horizon).
_MIN_ALPHA: Final = 1.0e-4
_MIN_PDF: Final = 1.0e-8
_MIN_WEIGHT: Final = 1.0e-8
# |n.z| above which the prefilter's tangent frame is built around +x instead of +z.
_POLE_COS: Final = 0.999


def downsample[C: IntVar](
  image: np.ndarray[[int, int, C], np.dtype[np.float32]], max_width: int
) -> np.ndarray[[int, int, C], np.dtype[np.float32]]:
  """Box-filtered halvings of `image` until it is at most `max_width` wide (a single row is only
  halved along its width)."""
  while image.shape[1] > max_width:
    fy = _BOX if image.shape[0] >= _BOX else 1
    h, w = image.shape[0] // fy * fy, image.shape[1] // _BOX * _BOX
    image = image[:h, :w].reshape(h // fy, fy, w // _BOX, _BOX, -1).mean(axis=(1, 3))
  return image


@wp.kernel
def _prefilter(
  # In:
  radiance: wp.Texture2D,
  source_height: wp.int32,
  nsample: wp.int32,
  row_offset: wp.int32,
  # Out:
  stack_out: wp.array4d[wp.float16],
) -> None:
  """Texel (level, row + row_offset, x) of the GGX-prefiltered stack (Karis 2013, filtered
  importance sampling)."""
  level, row, x = wp.tid()
  y = row + row_offset
  nlevel = stack_out.shape[0]
  height = stack_out.shape[1]
  width = stack_out.shape[2]
  n = latlong.uv_to_direction(wp.vec2((wp.float32(x) + 0.5) / wp.float32(width), (wp.float32(y) + 0.5) / wp.float32(height)))
  pr = wp.float32(level) / wp.float32(nlevel - 1)
  alpha = wp.max(pr * pr, wp.static(_MIN_ALPHA))

  # Tangent frame around n (= v = r for the split-sum approximation).
  up = wp.where(wp.abs(n[2]) < wp.static(_POLE_COS), wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))
  t = wp.normalize(wp.cross(up, n))
  b = wp.cross(n, t)
  # A source texel's solid angle as the mips see it: its larger angular side squared, the
  # polar step pi / height (the azimuth step 2 pi sin(theta) / width is no larger on a 2:1 map;
  # the mean texel, 4 pi / (width height), is 2 / pi of it: +0.33 LOD of blur everywhere).
  texel_solid_angle = (wp.static(wp.pi) / wp.float32(source_height)) ** 2.0

  color = wp.vec3(0.0)
  weight = wp.float32(0.0)
  inv_n = 1.0 / wp.float32(nsample)
  for s in range(nsample):
    h_local = ibl.importance_sample_ggx(ibl.hammersley(s, inv_n), alpha)
    h = t * h_local[0] + b * h_local[1] + n * h_local[2]
    noh = h_local[2]
    l = 2.0 * noh * h - n
    nol = wp.dot(n, l)
    if nol > 0.0:
      # Filtered importance sampling: read the source mip whose texels match the sample's lobe.
      # Level 0 (the mirror) reads the full-resolution map: its floored alpha makes D(h) at
      # h = n divide by zero in float32 (Warp debug mode's div(0, 0) assertion).
      lod = float(0.0)
      if level > 0:
        pdf = brdf.d_ggx(noh, alpha) * 0.25
        sample_solid_angle = inv_n / wp.max(pdf, wp.static(_MIN_PDF))
        lod = 0.5 * wp.log2(sample_solid_angle / texel_solid_angle) + 1.0
      c = wp.texture_sample(radiance, latlong.direction_to_uv(l, 0.0), dtype=wp.vec4, lod=wp.max(lod, 0.0))
      color += wp.vec3(c[0], c[1], c[2]) * nol
      weight += nol
  color = color / wp.max(weight, wp.static(_MIN_WEIGHT))
  stack_out[level, y, x, 0] = wp.float16(color[0])
  stack_out[level, y, x, 1] = wp.float16(color[1])
  stack_out[level, y, x, 2] = wp.float16(color[2])
  stack_out[level, y, x, 3] = wp.float16(1.0)


# The SH9 Lambertian kernel (clamped cosine, 1/4 + c/2 + 5/32 (3c^2 - 1)) dips to -0.04 near
# 122 degrees from a light: a compact very bright source (a sun left in the map, no `env_sun`)
# drives the irradiance on the side facing away from it negative, clamped to black. Maps whose
# reconstruction goes negative are windowed per band by the nearest window (least squares in
# (sigma_1, sigma_2)) that makes the kernel non-negative: a sun then adds >= 0 everywhere and the
# sky keeps its own light. Only the map's peaks are windowed (radiance above SH_PEAK x the map's
# mean without them, projected apart): the window lowers the L1 band by 15 %, and windowing a
# whole map that merely rings (a sky over a black ground dips to -0.1 % of its peak under the
# horizon, where nothing is lit) read its upward irradiance 7.9 % low. SH_PEAK: the kernel's
# negative lobe is 4 % of a source's peak irradiance, so only a source 25 x brighter than the
# rest can blacken a side.
# Rows of the SH9 projection's quadrature (`_sh9_irradiance`).
_SH_ROWS: Final = 256
SH_DERING: Final = np.array([1.0, 0.8474, 0.8474, 0.8474, 0.965, 0.965, 0.965, 0.965, 0.965])
SH_PEAK: Final = 25.0
# The share of the brightest direction's irradiance from which a direction counts as lit, and the
# (rows, columns) direction grid the ringing is checked on (`_blackens`).
_LIT_SHARE: Final = 0.01
_PROBE: Final = (32, 64)
# The peak level's fixed point: relative tolerance and step bound (`_peak_level`).
_LEVEL_TOL: Final = 0.01
_LEVEL_STEPS: Final = 64


def _sh9_basis(d: AnyRank[np.float64]) -> AnyRank[np.float64]:
  """(9, ...) real SH basis of unit directions `d` (..., 3) (the order of `eval_sh9`)."""
  x, y, z = d[..., 0], d[..., 1], d[..., 2]
  return np.stack(
    [
      np.full(x.shape, 0.282095),
      0.488603 * y,
      0.488603 * z,
      0.488603 * x,
      1.092548 * x * y,
      1.092548 * y * z,
      0.315392 * (3.0 * z * z - 1.0),
      1.092548 * x * z,
      0.546274 * (x * x - y * y),
    ]
  )


def _sh9_irradiance[H: IntVar, W: IntVar](
  image: np.ndarray[[H, W, 3], np.dtype[np.float32]],
) -> np.ndarray[[9, 3], np.dtype[np.float32]]:
  """SH9 coefficients of Lambertian radiance (irradiance / π) for an equirect radiance image, its
  peaks (`_peak_level`) windowed (`SH_DERING`)."""
  # The texel-centre quadrature is exact to ~1e-5 from 256 rows and 512 columns; a smaller map (a
  # constant environment is a few texels) is projected as its piecewise-constant texels resampled to
  # at least that (an 8 x 16 constant map read 1.5 % bright at the zenith, a white furnace above 1).
  # Rows and columns are repeated apart: the same factor for a 2:1 map, while a narrower one still
  # reaches the `_blackens` probe grid's columns.
  ky = -(-_SH_ROWS // image.shape[0])
  kx = -(-2 * _SH_ROWS // image.shape[1])
  if ky > 1 or kx > 1:
    image = np.repeat(np.repeat(image, ky, axis=0), kx, axis=1)
  h, w, _ = image.shape
  solid = texel_solid_angles(h, w)
  weights = _sh9_basis(latlong_directions(h, w)) * solid
  radiance = image.astype(np.float64)
  smooth = np.minimum(radiance, _peak_level(radiance, solid))
  band = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3] + [np.pi / 4] * 5)[:, None] / np.pi
  rest = np.einsum("khw,hwc->kc", weights, smooth) * band
  peaks = np.einsum("khw,hwc->kc", weights, radiance - smooth) * band
  coeffs = rest + peaks * SH_DERING[:, None]
  if _blackens(radiance, solid, coeffs):
    # A bright region below the peak level (a large softbox) still rings a lit side black.
    coeffs = (rest + peaks) * SH_DERING[:, None]
  return coeffs.astype(np.float32)


def _blackens(
  radiance: np.ndarray[[int, int, 3], np.dtype[np.float64]],
  solid: np.ndarray[[int, int], np.dtype[np.float64]],
  coeffs: np.ndarray[[9, 3], np.dtype[np.float64]],
) -> bool:
  """Whether the reconstruction `coeffs` (9, 3) goes below zero, on a `_PROBE` direction grid,
  in a direction whose exact irradiance (the map box-filtered to the grid, power kept) exceeds
  _LIT_SHARE of the brightest direction's (below it a black side is within the SH9 kernel's own
  error: its negative lobe is 4 % of a source's peak)."""
  h, w = _PROBE
  dirs = latlong_directions(h, w).reshape(-1, 3)
  reconstruction = _sh9_basis(dirs).T @ coeffs
  if reconstruction.min() >= 0.0:
    return False
  ih, iw, _ = radiance.shape
  fy, fx = ih // h, iw // w
  power = (radiance * solid[..., None])[: fy * h, : fx * w].reshape(h, fy, w, fx, 3).sum(axis=(1, 3)).reshape(-1, 3)
  exact = np.maximum(dirs @ dirs.T, 0.0) @ power / np.pi
  lit = exact > _LIT_SHARE * exact.max(axis=0, keepdims=True)
  return bool(np.any((reconstruction < 0.0) & lit))


def _peak_level(
  radiance: np.ndarray[[int, int, 3], np.dtype[np.float64]], solid: np.ndarray[[int, int], np.dtype[np.float64]]
) -> np.ndarray[[3], np.dtype[np.float64]]:
  """(3,) radiance above which texels are peaks, per channel: SH_PEAK x the mean of the map
  clipped at that level. The map is clipped ever lower from its plain mean (a sun dominates it):
  the level decreases monotonically to the fixed point, reached when it moves by less than
  _LEVEL_TOL (or the channel is black); _LEVEL_STEPS bounds the descent (a channel that is all
  peak descends towards 0: everything windowed)."""
  level = np.full(3, np.inf)
  total = solid.sum()
  for _ in range(_LEVEL_STEPS):
    new = SH_PEAK * (np.minimum(radiance, level) * solid[..., None]).sum(axis=(0, 1)) / total
    done = bool(np.all((new >= (1.0 - _LEVEL_TOL) * level) | (new == 0.0)))
    level = np.minimum(new, level)
    if done:
      break
  return level


def fold_sh9(coeffs: np.ndarray[[9, 3], np.dtype[np.float32]]) -> np.ndarray[[9, 3], np.dtype[np.float32]]:
  """SH9 coefficients with the basis constants folded in (`eval_sh9`'s polynomial): the
  constant term (L0 and the -1 of the z^2 band), y, z, x, xy, yz, z^2, xz, x^2 - y^2 (float64,
  rounded once)."""
  c = np.asarray(coeffs, np.float64)
  k = np.array([0.282095, 0.488603, 0.488603, 0.488603, 1.092548, 1.092548, 3.0 * 0.315392, 1.092548, 0.546274])
  folded = c * k[:, None]
  folded[0] = c[0] * 0.282095 - c[6] * 0.315392
  return folded.astype(np.float32)


@wp.func
def eval_sh9(
  # Model:
  sh: wp.array2d[wp.vec3],
  # In:
  env: wp.int32,
  n: wp.vec3,
) -> wp.vec3:
  """Lambertian radiance of folded SH9 coefficients `sh[env]` (`fold_sh9`: the basis constants
  pre-multiplied) around unit normal `n` (clamped at 0)."""
  x = n[0]
  y = n[1]
  z = n[2]
  c = (
    sh[env, 0]
    + sh[env, 1] * y
    + sh[env, 2] * z
    + sh[env, 3] * x
    + sh[env, 4] * (x * y)
    + sh[env, 5] * (y * z)
    + sh[env, 6] * (z * z)
    + sh[env, 7] * (x * z)
    + sh[env, 8] * (x * x - y * y)
  )
  return wp.max(c, wp.vec3(0.0))


@wp.func
def rotate_yaw(
  # In:
  direction: wp.vec3,
  yaw: wp.float32,
) -> wp.vec3:
  """`direction` rotated by `yaw` about +z."""
  c = wp.cos(yaw)
  s = wp.sin(yaw)
  return wp.vec3(c * direction[0] - s * direction[1], s * direction[0] + c * direction[1], direction[2])


@wp.func(inline=True)
def _prefiltered_uv(
  # Model:
  stack: wp.Texture3D,
  # In:
  uv: wp.vec2,
  perceptual_roughness: wp.float32,
) -> wp.vec3:
  """Prefiltered radiance at latlong `uv` at a perceptual roughness (trilinear in the stack)."""
  levels = wp.float32(stack.depth)
  w = (wp.clamp(perceptual_roughness, 0.0, 1.0) * (levels - 1.0) + 0.5) / levels
  c = wp.texture_sample(stack, wp.vec3(uv[0], uv[1], w), dtype=wp.vec4)
  return wp.vec3(c[0], c[1], c[2])


@wp.func(inline=True)
def _radiance_uv(
  # Model:
  radiance: wp.Texture2D,
  # In:
  uv: wp.vec2,
  lod: wp.float32,
) -> wp.vec3:
  """Radiance at latlong `uv` at mip level `lod` of the radiance texture."""
  c = wp.texture_sample(radiance, uv, dtype=wp.vec4, lod=wp.max(lod, 0.0))
  return wp.vec3(c[0], c[1], c[2])


@wp.func
def sample_specular(
  # Model:
  radiance: wp.Texture2D,
  stack: wp.Texture3D,
  # In:
  direction: wp.vec3,
  yaw: wp.float32,
  perceptual_roughness: wp.float32,
  spread: wp.float32,
) -> wp.vec3:
  """Prefiltered radiance; the sharp part also filters the ray cone's angular `spread`.

  Use this map's width, not the largest map in a per-world bank. Below one texel the footprint
  reads mip zero. An isotropic ray-cone approximation, without normal / curvature derivatives;
  the rough lobe uses its precomputed stack. Both maps share one latlong lookup.
  """
  uv = latlong.direction_to_uv(direction, yaw)
  blurred = _prefiltered_uv(stack, uv, perceptual_roughness)
  t = perceptual_roughness * (wp.float32(stack.depth) - 1.0)
  if t >= 1.0:
    return blurred
  texels = spread * float(radiance.width) / wp.static(2.0 * wp.pi)
  lod = wp.log2(wp.max(texels, 1.0))
  sharp = _radiance_uv(radiance, uv, lod)
  return sharp * (1.0 - t) + blurred * t


@wp.func
def sample_radiance(
  # Model:
  radiance: wp.Texture2D,
  # In:
  direction: wp.vec3,
  yaw: wp.float32,
  lod: wp.float32,
) -> wp.vec3:
  """Radiance along `direction` at mip level `lod` of the radiance texture."""
  return _radiance_uv(radiance, latlong.direction_to_uv(direction, yaw), lod)


def _launch_prefilter(radiance: wp.Texture2D, *, height: int, stack: wp.array[wp.float16, [int, int, int, 4]]) -> None:
  """Runs `_prefilter` over the stack; on the CPU in row bands on several threads.

  Warp runs a CPU launch on one thread (and releases the GIL meanwhile): prefiltering a 1k map
  serially takes ~30 s. The bands compute exactly the texels of the single launch.
  """
  nlevel, rows, cols = stack.shape[0], stack.shape[1], stack.shape[2]

  def run(row0: int, row1: int) -> None:
    """Prefilters rows [row0, row1)."""
    wp.launch(
      _prefilter,
      dim=(nlevel, row1 - row0, cols),
      inputs=[radiance, height, PREFILTER_SAMPLES, row0],
      outputs=[stack],
      device=stack.device,
    )

  launch_bands(wp.get_module(__name__), device=stack.device, n=rows, launch=run, min_band=1)


def _radiance(image: Image) -> HdrImage[int, int]:
  """The linear radiance (h, w, 3) float32 of an input map, downsampled to MAX_RADIANCE_WIDTH;
  negative radiance (out-of-gamut colors) clamped to 0 with a RuntimeWarning, as the texture
  upload clamps its converted texels (the prefilter and SH sums would subtract light).

  Raises:
    ValueError: for a map that is not (h, w, >= 3) floats with h >= 1 and w >= MIN_WIDTH, or that
      holds NaN or infinite radiance.
  """
  array = np.asarray(image)
  shape, dtype = array.shape, array.dtype
  if len(shape) != _MAP_RANK or shape[2] < _MIN_CHANNELS or shape[0] < 1 or shape[1] < MIN_WIDTH:
    raise ValueError(
      f"environment map: expected (height >= 1, width >= {MIN_WIDTH}, channels >= 3) linear radiance, got shape {shape}"
    )
  if not np.issubdtype(dtype, np.floating):
    raise ValueError(
      f"environment map: expected float linear radiance, got {dtype} (8-bit images are display-encoded: decode them, "
      "e.g. color.srgb_to_linear(image / 255))"
    )
  radiance = downsample(array.astype(np.float32, copy=False)[..., :3], MAX_RADIANCE_WIDTH)
  if not np.all(np.isfinite(radiance)):
    raise ValueError("environment map: NaN or infinite radiance (pass float32 radiance, not an overflowed float16 cast)")
  negative = int((radiance < 0.0).sum())
  if negative:
    warnings.warn(f"environment map: {negative} negative radiance components clamped to 0", RuntimeWarning, stacklevel=3)
    radiance = np.maximum(radiance, 0.0)  # a new array: `radiance` can view the caller's image
  return radiance


@final
class EnvironmentMap:
  """GPU resources for one environment map.

  Attributes:
    radiance: radiance texture (float16, divided by `scale`).
    prefiltered: GGX-prefiltered stack (float16, divided by `scale`; a 1-texel placeholder for a
      background-only map).
    sh: SH9 Lambertian radiance coefficients, basis constants folded (`fold_sh9`; divided by
      `scale`; zero for a background-only map).
    scale: radiance per stored unit: 1, or peak / MAX_STORED_RADIANCE for maps beyond it.
    width: width of the radiance texture.
  """

  radiance: wp.Texture2D
  prefiltered: wp.Texture3D
  sh: np.ndarray[[9, 3], np.dtype[np.float32]]
  scale: float
  width: int

  def __init__(self, image: Image, *, lighting: bool, device: wp.Device) -> None:
    """Uploads `image` ((h, w, >= 3) linear radiance) to `device` and, for a `lighting` map, its
    prefilter stack and SH (from the bake cache when it holds them, `bake_cache`). A
    background-only map (`lighting` False) is only looked up by rays leaving the scene: no
    prefilter, zero SH.

    Raises:
      ValueError: for a map of the wrong shape or with NaN or infinite radiance (`_radiance`).
    """
    radiance = _radiance(image)
    # Rescale before the float16 cast: a sun disc beyond float16's range would become inf, and
    # its bilinear / prefilter / SH sums NaN.
    peak = max(float(radiance.max()), 0.0)
    self.scale = max(peak / MAX_STORED_RADIANCE, 1.0)
    if self.scale > 1.0:
      radiance = radiance / np.asarray(self.scale, np.float32)
    h, w, _ = radiance.shape
    rgba = np.concatenate([radiance, np.ones((h, w, 1), np.float32)], axis=-1).astype(np.float16)
    self.radiance = MipTexture2D(
      box_chain(rgba), address=(wp.TextureAddressMode.WRAP, wp.TextureAddressMode.CLAMP), device=device
    )
    mip_key = ("odd-area-mips-v1",) if (w & (w - 1)) or (h & (h - 1)) else ()
    digest = bake_cache.texture_digest(rgba, "envmap", *mip_key)
    bake_cache.register_texture(self.radiance, digest)
    self.width = w
    if not lighting:
      self.prefiltered = _placeholder_stack(device)
      self.sh = np.zeros((9, 3), np.float32)
      return
    stack, sh = _lighting(self.radiance, radiance, device)
    self.sh = fold_sh9(sh)
    texture = wp.Texture3D(
      stack,
      filter_mode=wp.TextureFilterMode.LINEAR,
      address_mode_u=wp.TextureAddressMode.WRAP,
      address_mode_v=wp.TextureAddressMode.CLAMP,
      address_mode_w=wp.TextureAddressMode.CLAMP,
      device=device,
    )
    self.prefiltered = bake_cache.register_texture(texture, f"{digest}-prefiltered")


def _lighting(
  radiance: wp.Texture2D, image: HdrImage[int, int], device: wp.Device
) -> tuple[wp.array[wp.float16, [int, int, int, 4]], np.ndarray[[9, 3], np.dtype[np.float32]]]:
  """(GGX-prefiltered stack on `device`, SH9) of a map (float32 texels `image`): from the bake
  cache, else computed and stored."""
  h, w, _ = image.shape
  pw = min(PREFILTER_WIDTH, w)
  key = bake_cache.key("envmap", device, host=(bake_cache.texture_digest(image), PREFILTER_LEVELS, PREFILTER_SAMPLES, pw))
  stored = bake_cache.load(key)
  if stored is not None:
    return wp.array(stored["stack"], dtype=wp.float16, device=device), stored["sh"]
  # Every texel is written by `_prefilter`.
  stack = wp.empty((PREFILTER_LEVELS, pw // 2, pw, 4), dtype=wp.float16, device=device)
  if TYPE_CHECKING:
    expect_shape(stack, (PREFILTER_LEVELS, pw // 2, pw, 4))
  _launch_prefilter(radiance, height=h, stack=stack)
  sh = _sh9_irradiance(image)
  bake_cache.save(key, {"stack": host_copy(stack), "sh": sh})
  return stack, sh


def _placeholder_stack(device: wp.Device) -> wp.Texture3D:
  """The 1-texel prefilter stack of a background-only map or of no map (never sampled)."""
  return bake_cache.register_texture(wp.Texture3D(np.zeros((1, 1, 1, 4), np.float16), device=device), "placeholder-stack")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentTextures[NEnv: IntVar]:
  """Device resources of the environment maps of a scene
  (`core.pipeline.gpu_scene.gpu_scene.GpuEnvironment.maps`).

  Attributes:
    maps: the maps owning the textures (kept alive with the scene; empty without a map).
    radiance: radiance texture per map                                  (nenv,)
    prefiltered: GGX-prefiltered stack per map                          (nenv,)
    sh: SH9 Lambertian radiance coefficients                            (nenv, 9)
    scale: radiance per stored unit of each map (`EnvironmentMap.scale`) (nenv,)
    width: width of the largest radiance texture
    placeholders: the textures of `empty_environments` (kept alive likewise; empty with maps).
  """

  maps: tuple[EnvironmentMap, ...]
  radiance: wp.array[wp.Texture2D, [NEnv]]
  prefiltered: wp.array[wp.Texture3D, [NEnv]]
  sh: wp.array[wp.vec3, [NEnv, 9]]
  scale: wp.array[float, [NEnv]]
  width: float
  placeholders: tuple[wp.Texture, ...] = ()


def build_environments(
  images: Sequence[Image], lighting: Sequence[bool] | None = None, *, device: wp.Device
) -> EnvironmentTextures[int]:
  """GPU resources on `device` for environment maps; `lighting`: per map, whether it lights the
  scene (default: all; a background-only map skips its prefilter and SH).

  Raises:
    ValueError: without a map, for a `lighting` list of another length, or for an invalid map
      (`EnvironmentMap`).
  """
  if not images:
    raise ValueError("build_environments needs at least one map (`empty_environments` for none)")
  lit = [True] * len(images) if lighting is None else list(lighting)
  if len(lit) != len(images):
    raise ValueError(f"build_environments: {len(lit)} lighting flags for {len(images)} maps")
  maps = tuple(EnvironmentMap(im, lighting=on, device=device) for im, on in zip(images, lit, strict=True))
  return EnvironmentTextures(
    maps=maps,
    radiance=wp.array([e.radiance for e in maps], dtype=wp.Texture2D, device=device),
    prefiltered=wp.array([e.prefiltered for e in maps], dtype=wp.Texture3D, device=device),
    sh=wp.array(np.stack([e.sh for e in maps]), dtype=wp.vec3, device=device),
    scale=wp.array([e.scale for e in maps], dtype=float, device=device),
    width=float(max(e.width for e in maps)),
  )


def empty_environments(*, device: wp.Device) -> EnvironmentTextures[1]:
  """Placeholder resources on `device` of a context without environment map (constant ambient)."""
  radiance = bake_cache.register_texture(wp.Texture2D(np.zeros((1, 1, 4), np.float16), device=device), "placeholder-radiance")
  prefiltered = _placeholder_stack(device)
  return EnvironmentTextures(
    maps=(),
    radiance=wp.array([radiance], dtype=wp.Texture2D, device=device),
    prefiltered=wp.array([prefiltered], dtype=wp.Texture3D, device=device),
    sh=wp.zeros((1, 9), dtype=wp.vec3, device=device),
    scale=wp.ones(1, dtype=float, device=device),
    width=1.0,
    placeholders=(radiance, prefiltered),
  )


def texel_angle(width: float) -> float:
  """Angular size [rad] of one radiance texel at mip 0 of a map `width` texels wide."""
  return 2.0 * math.pi / width
