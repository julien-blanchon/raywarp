"""Tangent-space normal maps: the canonical convention and its one per-axis transform.

Convention. A normal-map binding (`Slot.GEOMETRY_NORMAL`, an `extras.textures` normal such as
`geometry_coat_normal`) follows the binding rule of every slot: its value is
`scale x texel + bias` per component (`TextureRef.scale` / `bias`), and that value IS the
tangent-space normal n = (x, y, z), renormalised where it is shaded. Its axes:

  x  along T, the tangent dp/du made orthogonal to the shading normal;
  y  along B = +-(N x T), the sign taken so that B points along +dp/dv;
  z  along N, the shading normal (back faces: minus the front's mapped normal).

(u, v) are the host UVs, origin at the image's bottom-left (USD `st`, MaterialX, OpenGL): +y is
"image up". This is the OpenGL / glTF / Blender / USD / MaterialX convention; B follows the UV
handedness, so mirrored UVs need no flag. A +Y-down (DirectX / Unreal) map is the same map with y
negated: `scale_xy(ref, Y_DOWN)`.

The one exception is MuJoCo: its UVs have a top-left origin and its images are not flipped, so its
device path decodes the plain unpack along (T, -dp/dv, N), the same +Y-up (OpenGL) maps in
MuJoCo's own UVs (`integrations.mujoco.materials` records the plain unpack; its texel maps stay the
identity).

The transform. Every authored normal-map modifier is an affine map per tangent axis on the decoded
normal, so it composes into the binding's scale / bias and needs no other record field:
  - UsdUVTexture scale / bias on a normal input (scale (2, -2, 2, 1), bias (-1, 1, -1, 0): a
    +Y-down map), taken as authored;
  - MaterialX `normalmap` scale (float or vector2; negative components flip an axis);
  - glTF `normalTexture.scale` (folded by the glTF-to-USD conversion into the UsdUVTexture
    scale / bias);
  - MDL: n = x T_u + y T_v + z N with T_u / T_v = +-state::texture_tangent_u / _v() (dp/du,
    dp/dv: the canonical T / B); a negated tangent is a negated axis, so OmniPBR-style
    `flip_tangent_u` / `flip_tangent_v` give sx / sy = -1 (`X_FLIPPED`, `Y_DOWN`) and the
    strength scales both; bind the expression's affine function of the texel as the TextureRef
    scale / bias, or `scale_xy` an unpacked binding (the hook for the MDL importer).
Mixing the axes (an XY swap, a rotation) is not per-axis; no source authors one (a UV rotation
turns the tangent frame with the UVs: `TextureRef.uv_xform`, `material.tex_variation`).

The core (`core.pipeline.upload.materials`) honours any finite per-component scale / bias exactly.
Unit per-axis scales (the unpack and its flips) are texel-map values only; a strength or an offset
compiles the scaled decode in (`Features.scaled_normal_map`: Toksvig on the fetched texel, the
texel map unclamped, so a strength above 1 decodes beyond [-1, 1]); other scenes keep their
generated code.
"""

from __future__ import annotations

import dataclasses
from typing import Final

from raywarp.scene.model.types import TextureRef

type Vec4 = tuple[float, float, float, float]
type XY = tuple[float, float]

# An 8-bit normal map's unpacking [0, 1] -> [-1, 1] (n = 2 t - 1): TextureRef (scale, bias).
NORMAL_UNPACK: Final = ((2.0, 2.0, 2.0, 1.0), (-1.0, -1.0, -1.0, 0.0))
# Per-axis scale reading a +Y-down (DirectX / Unreal) map in the canonical +Y-up convention.
Y_DOWN: Final = (1.0, -1.0)
# Per-axis scale of an X-flipped map (MDL `flip_tangent_u`).
X_FLIPPED: Final = (-1.0, 1.0)
# The decoded z of a usable normal map must reach this somewhere in the texel range [0, 1]: a map
# whose normals never leave the tangent plane has no outward direction.
MIN_OUTWARD_Z: Final = 1e-6


def unpack(xy: XY = (1.0, 1.0)) -> tuple[Vec4, Vec4]:
  """(scale, bias) of a binding that unpacks [0, 1] texels to [-1, 1] and scales the decoded
  tangent-plane components by `xy` = (sx, sy): n = (sx (2 r - 1), sy (2 g - 1), 2 b - 1)."""
  return _scaled(*NORMAL_UNPACK, xy)


def scale_xy(ref: TextureRef, xy: XY) -> TextureRef:
  """`ref` with its decoded tangent-plane components scaled by `xy` = (sx, sy) (a strength, or
  a flip: `Y_DOWN`, `X_FLIPPED`); z is unchanged."""
  if xy == (1.0, 1.0):
    return ref
  scale, bias = _scaled(ref.scale, ref.bias, xy)
  return dataclasses.replace(ref, scale=scale, bias=bias)


def _scaled(scale: Vec4, bias: Vec4, xy: XY) -> tuple[Vec4, Vec4]:
  """(scale, bias) with components x and y scaled by `xy`."""
  (sx, sy), (s, b) = xy, (scale, bias)
  return (sx * s[0], sy * s[1], s[2], s[3]), (sx * b[0], sy * b[1], b[2], b[3])


def outward(scale: Vec4, bias: Vec4) -> bool:
  """Whether a normal-map binding's decoded z (`scale[2]` x texel + `bias[2]`) is positive for
  some texel in [0, 1]: else every mapped normal lies in or below the tangent plane (degenerate)."""
  return max(bias[2], scale[2] + bias[2]) > MIN_OUTWARD_Z
