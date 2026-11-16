"""The Isaac Lab adapter's device work: Isaac's camera state in, Isaac's output values out.

Inputs (`update_camera`): world-convention poses (+X forward, +Z up, quaternion x y z w) to the
renderer's OpenGL-convention camera poses (-Z forward, +Y up), and the (N,) 3 x 3 pixel
intrinsics to the per-world "camera.intrinsics" table. Outputs (after each render, only for the
outputs that differ from what the megakernel writes into Isaac's buffers): radial depth from
planar depth, packed sRGB albedo, Isaac's reserved segmentation ids (0 background) and semantic
classes. Every kernel runs over (world, pixel) of flat (nworld, H * W) views.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp.core.pipeline.derive.kernels import roughen_under_coat
from raywarp.core.post.grading.color import pack_rgba, srgb_oetf

wp.set_module_options({"enable_backward": False})

# World-convention camera frame (+X forward, +Z up) -> OpenGL camera frame (-Z forward, +Y up): the
# rotation whose columns are the OpenGL camera's right (-Y), up (+Z) and back (-X) axes in the
# world-convention frame (Isaac Lab's NewtonWarpRenderer applies the same quaternion).
WORLD_TO_OPENGL: Final = wp.constant(wp.quatf(0.5, -0.5, -0.5, 0.5))
# Isaac Lab's reserved segmentation ids (isaaclab.renderers.segmentation_colors).
BACKGROUND_ID: Final = wp.constant(0)
UNLABELLED_ID: Final = wp.constant(1)


@wp.kernel
def camera_poses(
  # In:
  positions: wp.array1d[wp.vec3],
  orientations: wp.array1d[wp.quatf],
  cam: int,
  # Out:
  poses: wp.array2d[wp.transformf],
) -> None:
  """poses[w, cam] = the OpenGL-convention pose of camera `cam` in world w (Isaac's
  world-convention pose)."""
  w = wp.tid()
  poses[w, cam] = wp.transformf(positions[w], orientations[w] * WORLD_TO_OPENGL)


@wp.kernel
def camera_intrinsics(
  # In:
  matrices: wp.array1d[wp.mat33],
  width: float,
  height: float,
  cam: int,
  # Out:
  table: wp.array2d[wp.vec4],
  changed: wp.array1d[bool],
) -> None:
  """table[w, cam] = (fx, fy, principal point offsets) of camera `cam`'s pixel matrix in world w
  (pixel centres at i + 0.5, Isaac Lab's convention: cx = width / 2 is centred) on a sensor of
  width x height px."""
  w = wp.tid()
  k = matrices[w]
  calibration = wp.vec4(k[0, 0], k[1, 1], 0.5 * width - k[0, 2], 0.5 * height - k[1, 2])
  if changed.shape[0] > 0:
    previous = table[w, cam]
    changed[w] = (
      previous[0] != calibration[0]
      or previous[1] != calibration[1]
      or previous[2] != calibration[2]
      or previous[3] != calibration[3]
    )
  table[w, cam] = calibration


@wp.kernel
def radial_depth(
  # In:
  planar: wp.array2d[float],
  table: wp.array2d[wp.vec4],
  width: int,
  height: int,
  miss: float,
  cam: int,
  # Out:
  radial: wp.array2d[float],
) -> None:
  """Radial = distance to the camera centre of each pixel's planar depth; misses (`miss`: 0, inf
  or the far clip) keep their value exactly."""
  w, p = wp.tid()
  d = planar[w, p]
  if d == miss:
    radial[w, p] = miss
    return
  k = table[w % table.shape[0], cam]
  # The pixel's ray direction on the plane at unit depth (offsets: (0.5 W - cx, 0.5 H - cy)).
  x = (float(p % width) + 0.5 - (0.5 * float(width) - k[2])) / k[0]
  y = (float(p / width) + 0.5 - (0.5 * float(height) - k[3])) / k[1]
  radial[w, p] = d * wp.sqrt(1.0 + x * x + y * y)


@wp.kernel
def packed_albedo(
  # In:
  albedo: wp.array2d[wp.vec3],
  # Out:
  packed: wp.array2d[wp.uint32],
) -> None:
  """Packed = the linear base color sRGB-encoded as RGBA8 (alpha 255; background black)."""
  w, p = wp.tid()
  c = albedo[w, p]
  r = wp.clamp(srgb_oetf(c[0]), 0.0, 1.0) * 255.0 + 0.5
  g = wp.clamp(srgb_oetf(c[1]), 0.0, 1.0) * 255.0 + 0.5
  b = wp.clamp(srgb_oetf(c[2]), 0.0, 1.0) * 255.0 + 0.5
  packed[w, p] = pack_rgba(r, g, b, 255.0)


@wp.kernel
def semantic_ids(
  # In:
  seg: wp.array2d[wp.int32],
  classes: wp.array1d[wp.int32],
  # Out:
  semantic: wp.array2d[wp.int32],
) -> None:
  """Semantic = the class of each pixel's segment (`classes[seg]`; background where seg < 0)."""
  w, p = wp.tid()
  s = seg[w, p]
  semantic[w, p] = wp.where(s < 0, BACKGROUND_ID, classes[s])


@wp.kernel
def background_ids(
  # Out:
  seg: wp.array2d[wp.int32],
) -> None:
  """Isaac's background id where the renderer wrote a miss (-1)."""
  w, p = wp.tid()
  if seg[w, p] < 0:
    seg[w, p] = BACKGROUND_ID


@wp.kernel
def scatter_colors(
  # In:
  colors: wp.array1d[wp.vec3],
  rows: wp.array1d[wp.int32],
  envs: wp.array1d[wp.int32],
  start: wp.array1d[wp.int32],
  targets: wp.array1d[wp.vec2i],
  scale: wp.array1d[float],
  bias: wp.array1d[wp.vec3],
  # Out:
  table: wp.array2d[wp.vec4],
  dirty: wp.array1d[int],
) -> None:
  """Write canonical colors, preserve alpha and mark changed worlds."""
  m, e, k = wp.tid()
  row = rows[m] + envs[e]
  if row < 0 or row >= colors.shape[0]:
    return
  slot = start[row] + k
  if slot >= start[row + 1]:
    return
  t = targets[slot]
  c = colors[row] * scale[slot] + bias[slot]
  old = table[t[0], t[1]]
  if c[0] != old[0] or c[1] != old[1] or c[2] != old[2]:
    table[t[0], t[1]] = wp.vec4(c[0], c[1], c[2], old[3])
    wp.atomic_max(dirty, t[0], 1)


@wp.kernel
def scatter_scalars(
  # In:
  values: wp.array1d[float],
  rows: wp.array1d[wp.int32],
  envs: wp.array1d[wp.int32],
  start: wp.array1d[wp.int32],
  targets: wp.array1d[wp.vec2i],
  multiplier: float,
  roughness: bool,
  coat_weight: wp.array1d[float],
  coat_roughness: wp.array1d[float],
  roughenable: wp.array1d[bool],
  # Out:
  table: wp.array2d[float],
  dirty: wp.array1d[int],
) -> None:
  """Convert explicit authored scalar inputs to canonical tables, preserving coat roughening."""
  m, e, k = wp.tid()
  row = rows[m] + envs[e]
  if row < 0 or row >= values.shape[0]:
    return
  slot = start[row] + k
  if slot >= start[row + 1]:
    return
  target = targets[slot]
  column = target[1]
  value = values[row] * multiplier
  if roughness:
    if roughenable[column] and coat_weight[column] > 0.0:
      value = wp.float32(
        roughen_under_coat(wp.float64(value), wp.float64(coat_weight[column]), wp.float64(coat_roughness[column]))
      )
  if value != table[target[0], column]:
    table[target[0], column] = value
    wp.atomic_max(dirty, target[0], 1)


@wp.kernel
def changed_worlds(
  # In:
  dirty: wp.array1d[int],
  # Out:
  changed: wp.array1d[bool],
) -> None:
  """Convert atomic change flags to the renderer's world history-reset mask."""
  world = wp.tid()
  changed[world] = dirty[world] != 0


@wp.kernel
def reset_mask(
  # In:
  ids: wp.array1d[wp.int32],
  # Out:
  mask: wp.array1d[bool],
) -> None:
  """Mark Isaac's validated reset ids, without downloading a CUDA tensor to the host."""
  i = ids[wp.tid()]
  if i >= 0 and i < mask.shape[0]:
    mask[i] = True
