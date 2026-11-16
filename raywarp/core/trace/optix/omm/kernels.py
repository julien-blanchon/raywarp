"""The opacity micromap bake's device code (`bake.bake` launches it).

For each triangle a subdivision level gives micro-triangles of about `MICRO_TEXELS` texels per
edge (`micro_levels`: level 1 to `MAX_LEVEL`); for each micro-triangle (`micro_states`) every
texel the alpha test's bilinear lookup (`texture.sample_layout`, level 0, the material's tiling)
can read anywhere on it (its UV box, padded for rounding) is sampled at its centre: all >= cutoff
(with a margin) -> OPAQUE, all < cutoff -> TRANSPARENT, else (or a footprint wider than
`MAX_SPAN` texels) UNKNOWN_OPAQUE. Bilinear and nearest lookups are convex combinations of those
texels, and the sRGB decode is monotonic, so a resolved state agrees with the test at every point.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import uv_xform
from raywarp.core.shade.materials import texture

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Target micro-triangle edge [texels]: finer gains nothing (the alpha test reads texels) and costs
# 4x memory per level.
MICRO_TEXELS: Final = 2.0
# Subdivision cap: level 9 is 262144 micro-triangles (64 KiB) per triangle, a 1024-texel leaf card
# triangle at MICRO_TEXELS. OptiX allows 12.
MAX_LEVEL: Final = 9
# Micro-triangles whose texel footprint is wider than this [texels per axis] stay UNKNOWN (tiled
# textures on large triangles: the any-hit test decides, as without micromaps).
MAX_SPAN: Final = 48
# The opaque / transparent decision keeps this margin from the cutoff: a texel-centre sample
# can mix in up to 1/256 of a neighbour (hardware coordinates and filter weights are fixed
# point), times the sRGB decode's slope (< 2.5).
_MARGIN: Final = 1.0 / 64.0
# UV box padding [texels]: absolute, and relative to the texel coordinate (float rounding of the
# hit's barycentrics and UVs).
_PAD_ABS: Final = 0.05
_PAD_REL: Final = 4.0e-6
# OptiX's 4-state values (optix_types.h OPTIX_OPACITY_MICROMAP_STATE_*).
STATE_TRANSPARENT: Final = 0
STATE_OPAQUE: Final = 1
STATE_UNKNOWN_OPAQUE: Final = 3
# Level of a face without UVs (`micro_levels`): no micromap, never cut.
_NO_UV: Final = -1


@wp.func
def _even_bits(
  # In:
  x: int,
) -> int:
  """The even bits of x (>= 0, < 2^31), packed (optix_micromap_impl.h extractEvenBits)."""
  x = x & 0x55555555
  x = (x | (x >> 1)) & 0x33333333
  x = (x | (x >> 2)) & 0x0F0F0F0F
  x = (x | (x >> 4)) & 0x00FF00FF
  return (x | (x >> 8)) & 0x0000FFFF


@wp.func
def _prefix_eor(
  # In:
  x: int,
) -> int:
  """Prefix exclusive-or of x's bits (>= 0, 16 bits) from the top (optix_micromap_impl.h
  prefixEor)."""
  x = x ^ (x >> 1)
  x = x ^ (x >> 2)
  x = x ^ (x >> 4)
  return x ^ (x >> 8)


@wp.func
def micro_triangle(
  # In:
  index: int,
  level: int,
) -> tuple[wp.vec2, wp.vec2, wp.vec2]:
  """Barycentrics (b1, b2) of the corners of micro-triangle `index` at subdivision `level`, in
  OptiX's micromap order (optix_micromap_impl.h micro2bary; the corner order, which only the
  winding depends on, is not kept)."""
  if level == 0:
    return wp.vec2(0.0, 0.0), wp.vec2(1.0, 0.0), wp.vec2(0.0, 1.0)
  # Signed ints: `~` sets the high bits, which `mask` (level <= 12 bits) clears.
  b0 = _even_bits(index)
  b1 = _even_bits(index >> 1)
  fx = _prefix_eor(b0)
  fy = _prefix_eor(b0 & ~b1)
  t = fy ^ b1
  mask = (1 << level) - 1
  iu = ((fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t)) & mask
  iv = (fy ^ b0) & mask
  iw = ((~fx & ~t) | (b0 & ~t) | (~b0 & fx & t)) & mask
  flipped = ((iu ^ iv ^ iw) & 1) == 0
  d = 1.0 / float(1 << level)
  u = float(iu) * d
  v = float(iv) * d
  if flipped:
    return wp.vec2(u, v + d), wp.vec2(u + d, v + d), wp.vec2(u + d, v)
  return wp.vec2(u, v), wp.vec2(u + d, v), wp.vec2(u, v + d)


@wp.func
def face_uvs(
  # Model:
  mesh_faceadr: wp.array1d[int],
  mesh_facetexcoord: wp.array1d[wp.vec3i],
  mesh_texcoord: wp.array1d[wp.vec2],
  mesh_texcoord_adr: wp.array1d[int],
  # In:
  texcoord: wp.vec2i,
  xform: uv_xform,
  repeat: wp.vec2,
  size: wp.vec2,
  mesh: int,
  face: int,
) -> tuple[bool, wp.vec2, wp.vec2, wp.vec2]:
  """(has UVs, corner coordinates in texels) of a face as the alpha test reads it
  (`cutout.make_mesh_uv`: the instance's UVs, the material's tiling), times the texture size."""
  has_uv, uv0, uv1, uv2 = texture.triangle_uvs(
    mesh_facetexcoord, mesh_texcoord, mesh_texcoord_adr[mesh], mesh_faceadr[mesh], texcoord, xform, face
  )
  scale = wp.cw_mul(repeat, size)
  return has_uv, wp.cw_mul(uv0, scale), wp.cw_mul(uv1, scale), wp.cw_mul(uv2, scale)


@wp.kernel
def micro_levels(
  # Model:
  mesh_faceadr: wp.array1d[int],
  mesh_facetexcoord: wp.array1d[wp.vec3i],
  mesh_texcoord: wp.array1d[wp.vec2],
  mesh_texcoord_adr: wp.array1d[int],
  # In:
  tex: wp.Texture2D,
  texcoord: wp.vec2i,
  xform: uv_xform,
  repeat: wp.vec2,
  mesh: int,
  # Out:
  level_out: wp.array1d[int],
):
  """Subdivision level of a face: micro-triangle edges of about MICRO_TEXELS texels (1 to
  MAX_LEVEL; _NO_UV for a face without UVs)."""
  face = wp.tid()
  size = wp.vec2(float(tex.width), float(tex.height))
  has_uv, p0, p1, p2 = face_uvs(
    mesh_faceadr, mesh_facetexcoord, mesh_texcoord, mesh_texcoord_adr, texcoord, xform, repeat, size, mesh, face
  )
  if not has_uv:
    level_out[face] = wp.static(_NO_UV)
    return
  edge = wp.max(wp.max(wp.length(p1 - p0), wp.length(p2 - p1)), wp.length(p0 - p2))
  level = int(wp.ceil(wp.log2(wp.max(edge / wp.static(MICRO_TEXELS), 1.0))))
  level_out[face] = wp.clamp(level, 1, wp.static(MAX_LEVEL))


@wp.func
def micro_state(
  # In:
  tex: wp.Texture2D,
  layout: int,
  cutoff: float,
  p0: wp.vec2,
  p1: wp.vec2,
  p2: wp.vec2,
) -> int:
  """4-state value of the micro-triangle with texel-space corners p0, p1, p2 (see the module
  doc): every texel a lookup on it can read, sampled at its centre."""
  lo = wp.min(wp.min(p0, p1), p2)
  hi = wp.max(wp.max(p0, p1), p2)
  reach = wp.max(wp.max(wp.abs(lo[0]), wp.abs(lo[1])), wp.max(wp.abs(hi[0]), wp.abs(hi[1])))
  pad = wp.static(_PAD_ABS) + wp.static(_PAD_REL) * reach
  # Bilinear footprint: texels floor(x - 0.5) and the next one (nearest lookups read a subset).
  x0 = int(wp.floor(lo[0] - 0.5 - pad))
  y0 = int(wp.floor(lo[1] - 0.5 - pad))
  x1 = int(wp.floor(hi[0] - 0.5 + pad)) + 1
  y1 = int(wp.floor(hi[1] - 0.5 + pad)) + 1
  if x1 - x0 >= wp.static(MAX_SPAN) or y1 - y0 >= wp.static(MAX_SPAN):
    return wp.static(STATE_UNKNOWN_OPAQUE)
  inv = wp.vec2(1.0 / float(tex.width), 1.0 / float(tex.height))
  # keeps / cuts: some texel the test may keep (>= cutoff) / discard (< cutoff), within the margin.
  keeps = bool(False)
  cuts = bool(False)
  for y in range(y0, y1 + 1):
    for x in range(x0, x1 + 1):
      uv = wp.cw_mul(wp.vec2(float(x) + 0.5, float(y) + 0.5), inv)
      a = texture.sample_layout(tex, layout, uv, 0.0)[0]
      keeps = keeps or a >= cutoff - wp.static(_MARGIN)
      cuts = cuts or a < cutoff + wp.static(_MARGIN)
      if keeps and cuts:
        return wp.static(STATE_UNKNOWN_OPAQUE)
  if not cuts:
    return wp.static(STATE_OPAQUE)
  if not keeps:
    return wp.static(STATE_TRANSPARENT)
  return wp.static(STATE_UNKNOWN_OPAQUE)


@wp.kernel
def micro_states(
  # Model:
  mesh_faceadr: wp.array1d[int],
  mesh_facetexcoord: wp.array1d[wp.vec3i],
  mesh_texcoord: wp.array1d[wp.vec2],
  mesh_texcoord_adr: wp.array1d[int],
  # In:
  tex: wp.Texture2D,
  layout: int,
  cutoff: float,
  texcoord: wp.vec2i,
  xform: uv_xform,
  repeat: wp.vec2,
  mesh: int,
  faces: wp.array1d[int],
  levels: wp.array1d[int],
  starts: wp.array1d[int],
  ends: wp.array1d[int],
  # Out:
  data_out: wp.array1d[wp.uint8],
  opaque_out: wp.array1d[int],
  transparent_out: wp.array1d[int],
):
  """Byte b of the dense micromap data: the 4 states of its micro-triangles (2 bits each, the
  lowest first), and their opaque / transparent counts per micromap k (`faces[k]`)."""
  b = wp.tid()
  k = wp.lower_bound(ends, b + 1)
  face = faces[k]
  level = levels[k]
  size = wp.vec2(float(tex.width), float(tex.height))
  _has_uv, p0, p1, p2 = face_uvs(
    mesh_faceadr, mesh_facetexcoord, mesh_texcoord, mesh_texcoord_adr, texcoord, xform, repeat, size, mesh, face
  )
  e1 = p1 - p0
  e2 = p2 - p0
  bits = int(0)
  opaque = int(0)
  clear = int(0)
  for j in range(4):
    c0, c1, c2 = micro_triangle((b - starts[k]) * 4 + j, level)
    s = micro_state(
      tex,
      layout,
      cutoff,
      p0 + e1 * c0[0] + e2 * c0[1],
      p0 + e1 * c1[0] + e2 * c1[1],
      p0 + e1 * c2[0] + e2 * c2[1],
    )
    bits = bits | (s << (2 * j))
    opaque += int(s == wp.static(STATE_OPAQUE))
    clear += int(s == wp.static(STATE_TRANSPARENT))
  data_out[b] = wp.uint8(bits)
  if opaque > 0:
    wp.atomic_add(opaque_out, k, opaque)
  if clear > 0:
    wp.atomic_add(transparent_out, k, clear)


@wp.kernel
def pack_kept(
  # In:
  data: wp.array1d[wp.uint8],
  source: wp.array1d[int],
  starts: wp.array1d[int],
  ends: wp.array1d[int],
  # Out:
  data_out: wp.array1d[wp.uint8],
):
  """Byte b of the kept micromaps' data, from kept micromap k's dense offset `source[k]`."""
  b = wp.tid()
  k = wp.lower_bound(ends, b + 1)
  data_out[b] = data[source[k] + b - starts[k]]
