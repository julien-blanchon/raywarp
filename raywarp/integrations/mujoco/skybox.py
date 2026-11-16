"""MuJoCo cube textures (skybox, cube) as linear equirectangular environment images.

A MuJoCo cube texture stores its six square faces stacked vertically in OpenGL order (+X, -X,
+Y, -Y, +Z, -Z), and MuJoCo maps a world direction to cube-map space by a 90 degree turn about X
(render_gl3.c: s = x, t = z, r = -y). `cube_face_uv` is that lookup (the arithmetic of MuJoCo
Warp's renderer, so the resampled map matches it texel for texel).
"""

from __future__ import annotations

from typing import Final

import mujoco
import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.textures.formats import Level
from raywarp.core.post.grading.color import srgb_eotf
from raywarp.core.shade.environment.latlong import uv_to_direction
from raywarp.integrations.mujoco.textures import NO_MJ_TEXTURE, MjTextureId, rgba_texels

# Forward-only kernels: no adjoint code.
wp.set_module_options({"enable_backward": False})

# MuJoCo's mjMINVAL: the denominator of a zero major axis.
_MINVAL: Final = float(mujoco.mjMINVAL)


@wp.func
def _safe_div(
  # In:
  x: float,
  y: float,
) -> float:
  """X / y, with a zero y replaced by mjMINVAL."""
  return x / wp.where(y != 0.0, y, wp.static(_MINVAL))


@wp.func
def cube_face_uv(
  # In:
  d: wp.vec3,
  face_width_inv: float,
) -> wp.vec2:
  """Texture coordinates of world direction `d` in a vertical strip of six cube faces."""
  rx = d[0]
  ry = d[2]
  rz = -d[1]
  arx = wp.abs(rx)
  ary = wp.abs(ry)
  arz = wp.abs(rz)
  face = int(0)
  sc = float(0.0)
  tc = float(0.0)
  ma = float(1.0)
  if arx >= ary and arx >= arz:
    ma = arx
    face = wp.where(rx > 0.0, 0, 1)
    sc = wp.where(rx > 0.0, -rz, rz)
    tc = -ry
  elif ary >= arz:
    ma = ary
    face = wp.where(ry > 0.0, 2, 3)
    sc = rx
    tc = wp.where(ry > 0.0, rz, -rz)
  else:
    ma = arz
    face = wp.where(rz > 0.0, 4, 5)
    sc = wp.where(rz > 0.0, rx, -rx)
    tc = -ry
  s = (_safe_div(sc, ma) + 1.0) * 0.5
  t = (_safe_div(tc, ma) + 1.0) * 0.5
  # Keep the linear filter from bleeding between adjacent faces of the strip.
  t_min = 0.5 * face_width_inv
  t = wp.clamp(t, t_min, 1.0 - t_min)
  return wp.vec2(s, (float(face) + t) * wp.static(1.0 / 6.0))


@wp.kernel
def _cube_to_equirect(
  # In:
  cube: wp.Texture2D,
  face_width_inv: float,
  srgb: bool,
  # Out:
  image_out: wp.array3d[float],
):
  """Texel (y, x) of the equirect image: the cube texture seen along its direction, linear."""
  y, x = wp.tid()
  h = image_out.shape[0]
  w = image_out.shape[1]
  d = uv_to_direction(wp.vec2((float(x) + 0.5) / float(w), (float(y) + 0.5) / float(h)))
  c = wp.texture_sample(cube, cube_face_uv(d, face_width_inv), dtype=wp.vec4)
  for i in range(3):
    image_out[y, x, i] = wp.where(srgb, srgb_eotf(c[i]), c[i])


def cube_equirect(
  pixels: Level[np.uint8], *, srgb: bool, device: wp.Device, width: int = 1024
) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
  """The cube texture of texels `pixels` (six faces stacked vertically, `textures.loaded`) as a
  (width / 2, width, 3) linear equirect image, resampled on `device`.

  `srgb`: the texels are display (sRGB) colors, decoded to linear.
  """
  rgba = rgba_texels(pixels)
  face = rgba.shape[1]
  # float32 texels x float32(1 / 255), MuJoCo Warp's conversion (bit-identical maps).
  scale = np.full(1, 1.0 / 255.0, np.float32)
  cube = wp.Texture2D(rgba.astype(np.float32) * scale, filter_mode=wp.TextureFilterMode.LINEAR, device=device)
  image = wp.empty((width // 2, width, 3), dtype=float, device=device)
  inputs = [cube, 1.0 / float(face), srgb]
  wp.launch(_cube_to_equirect, dim=(width // 2, width), inputs=inputs, outputs=[image], device=device)
  return np.asarray(host_copy(image), np.float32).reshape(width // 2, width, 3)


def skybox_id(mjm: mujoco.MjModel) -> MjTextureId:
  """The model's first skybox texture, else NO_MJ_TEXTURE."""
  ids = np.flatnonzero(mjm.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX) if mjm.ntex else []
  return MjTextureId(int(ids[0])) if len(ids) else NO_MJ_TEXTURE
