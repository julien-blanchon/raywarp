"""Stores a 3D direction in two numbers (octahedral mapping onto the square [-1, 1]²), or packed in
32 bits: one definition for every user.

`oct_encode` / `oct_decode` (float coordinates: the GI probes' maps, `shade/gi/probes.py`) and
`pack_unit` / `unpack_unit` (two snorm16 coordinates in one uint32, error < 1e-4 rad: the OptiX
payload's face normals, `trace/optix/device/trace.py`, and the mesh face normals packed at build
time, `trace/optix/mesh_inputs.face_normals`).
"""

from __future__ import annotations

from typing import Final

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# snorm16 scale of a packed coordinate.
_SNORM16: Final = 32767.0
# Sign bit of a 16-bit two's complement code, and the code range (2^15, 2^16).
_SIGN16: Final = 32768
_RANGE16: Final = 65536


@wp.func
def _sign_not_zero(
  # In:
  v: wp.float32,
) -> wp.float32:
  """+1 for v >= 0, else -1 (the octahedral fold's sign)."""
  return wp.where(v >= 0.0, 1.0, -1.0)


@wp.func
def oct_encode(
  # In:
  d: wp.vec3,
) -> wp.vec2:
  """Unit direction -> octahedral coordinates in [-1, 1]²."""
  inv_l1 = 1.0 / (wp.abs(d[0]) + wp.abs(d[1]) + wp.abs(d[2]))
  u = d[0] * inv_l1
  v = d[1] * inv_l1
  if d[2] < 0.0:
    # Lower hemisphere: fold the octahedron's lower half over the square's corners.
    uu = (1.0 - wp.abs(v)) * _sign_not_zero(u)
    v = (1.0 - wp.abs(u)) * _sign_not_zero(v)
    u = uu
  return wp.vec2(u, v)


@wp.func
def oct_decode(
  # In:
  uv: wp.vec2,
) -> wp.vec3:
  """Octahedral coordinates in [-1, 1]² -> unit direction."""
  x = uv[0]
  y = uv[1]
  z = 1.0 - wp.abs(x) - wp.abs(y)
  if z < 0.0:
    xx = (1.0 - wp.abs(y)) * _sign_not_zero(x)
    y = (1.0 - wp.abs(x)) * _sign_not_zero(y)
    x = xx
  return wp.normalize(wp.vec3(x, y, z))


@wp.func
def _snorm16(
  # In:
  x: wp.float32,
) -> wp.uint32:
  """X in [-1, 1] as a 16-bit two's complement snorm (low 16 bits)."""
  return wp.uint32(wp.int32(wp.round(x * wp.static(_SNORM16))) & 0xFFFF)


@wp.func
def _from_snorm16(
  # In:
  bits: wp.uint32,
) -> wp.float32:
  """The value of a 16-bit snorm (sign-extended from bit 15)."""
  i = wp.int32(bits & wp.uint32(0xFFFF))
  if i >= wp.static(_SIGN16):
    i -= wp.static(_RANGE16)
  return wp.float32(i) * wp.static(1.0 / _SNORM16)


@wp.func
def pack_unit(
  # In:
  n: wp.vec3,
) -> wp.uint32:
  """A non-zero vector's direction as two snorm16 octahedral coordinates (u low, v high)."""
  uv = oct_encode(n)
  return _snorm16(uv[0]) | (_snorm16(uv[1]) << wp.uint32(16))


@wp.func
def unpack_unit(
  # In:
  packed: wp.uint32,
) -> wp.vec3:
  """The unit direction of `pack_unit`'s code."""
  return oct_decode(wp.vec2(_from_snorm16(packed), _from_snorm16(packed >> wp.uint32(16))))
