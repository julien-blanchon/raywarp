"""What the "motion" output keeps between frames: the state (`Temporal`, kept by
`history.History`) and the first hit's history (`make_write_history`: exact motion through the
previous instance and camera poses and projection, with its status: `FlowStatus`).

Compiled in only when the motion output is (`Features.history`); otherwise the megakernel has the
plain per-pixel body (`pixel.make_render_pixel`). The per-pixel body is `temporal/pixel.py`.

Determinism: no atomics feed a color. Same seed, inputs and reset pattern -> bitwise identical
images.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import warp as wp

from raywarp._base.types import ProjectionType
from raywarp.core.kernels.abi import rays as trace
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.output_layout import OUT_FLOW_STATUS, FlowStatus
from raywarp.core.kernels.abi.rays import GeometryFields, Hit
from raywarp.core.kernels.abi.structs import OutputArrays, lens_coefficients, u8s
from raywarp.core.kernels.abi.toggles import out_on
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.trace.accel import instances
from raywarp.core.trace.rays import camera, primary

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Largest finite float16: the previous depths are clamped to it (`Temporal.gz`).
HALF_MAX: Final = 65504.0
# Motion status: relative depth precision of the occlusion test, 2 x the float16 rounding of the
# stored previous depth (2^-11 relative): equal depths never read as occluded.
OCCLUSION_TOLERANCE: Final = 2.0**-10


@wp.struct
class PrevCamera:
  """A rendering camera as the previous frame saw it (`Temporal.prev_camera`)."""

  pos: wp.vec3
  rot: wp.mat33
  fovy: float
  intrinsics: wp.vec4
  distortion: lens_coefficients


@wp.struct
class Temporal:
  """Per-world temporal state and buffers ((1, ..., 1) placeholders where not compiled in).

  Attributes:
    frame: (nworld,) frames rendered since the last reset (the previous-depth slot).
    count: (nworld, ncam) frames since the last reset.
    gz: (nworld, 2, ncam * H * W) planar depth of the first hit (0: miss; float16, clamped to
      HALF_MAX: 5e-4 relative), slot frame % 2: this frame's and the previous one's (the motion
      status's occlusion test).
    motion: (nworld, ncam * H * W) the "motion" output.
    prev_camera: (nworld, ncam) the rendering cameras of the previous frame (pose, projection).
    prev_instance_pos: (nworld, ncolumn) per-world pose columns of the previous frame.
    prev_instance_rot: (nworld, ncolumn) the same frames.
  """

  frame: wp.array1d[int]
  count: wp.array2d[int]
  gz: wp.array3d[wp.float16]
  motion: wp.array2d[wp.vec2]
  prev_camera: wp.array2d[PrevCamera]
  prev_instance_pos: wp.array2d[wp.vec3]
  prev_instance_rot: wp.array2d[wp.mat33]


@wp.func
def previous_pixel(
  # Data in:
  tmp: Temporal,
  # In:
  pix: camera.Pixel,
  worldid: int,
) -> camera.Pixel:
  """`pix` as the previous frame saw it: its camera pose, field of view and intrinsics."""
  prev = pix
  c = tmp.prev_camera[worldid, pix.camid]
  prev.pos = c.pos
  prev.rot = c.rot
  prev.fovy = c.fovy
  prev.intrinsics = c.intrinsics
  prev.distortion = c.distortion
  return prev


# write_history(geo, out, tmp, pix, worldid, adr, hit, origin, direction, depth_axis): a first
# hit's previous depth, motion and motion status.
type WriteHistory = Callable[
  [GeometryFields, OutputArrays, Temporal, camera.Pixel, int, int, Hit, wp.vec3, wp.vec3, wp.vec3], None
]


@wp.func
def _previous_status(
  # In:
  pix: camera.Pixel,
  uv_prev: wp.vec2,
  ok_prev: bool,
  zprev: float,
) -> int:
  """The `FlowStatus` of a point at depth `zprev` (`camera.view_depth`: planar, along the ray for
  fisheye) and image position `uv_prev` (`ok_prev`: `camera.project`'s) in the previous camera,
  before the occlusion test: BEHIND its near plane (or a pinhole's eye), OUT_OF_FRAME of its image
  (a fisheye's model range included) or beyond its far plane, else VALID."""
  fisheye = pix.lens.projection == wp.static(int(ProjectionType.FISHEYE))
  if zprev <= wp.max(pix.lens.znear, 0.0) or (not ok_prev and not fisheye):
    return wp.static(int(FlowStatus.BEHIND))
  if not ok_prev:
    return wp.static(int(FlowStatus.OUT_OF_FRAME))
  if pix.lens.zfar > 0.0 and zprev > pix.lens.zfar:
    return wp.static(int(FlowStatus.OUT_OF_FRAME))
  if uv_prev[0] < 0.0 or uv_prev[1] < 0.0 or uv_prev[0] >= float(pix.width) or uv_prev[1] >= float(pix.height):
    return wp.static(int(FlowStatus.OUT_OF_FRAME))
  return wp.static(int(FlowStatus.VALID))


@specialize
def make_reproject(f: Features, model: primary.CameraModel):
  """reproject(geo, tmp, pix, worldid, hit, origin, direction, depth) -> (motion [px],
  planar depth in the previous frame: 0 miss, -1 invalid, `FlowStatus` without the occlusion
  test) of a first hit.

  Exact motion: the previous instance pose of the hit point and the previous camera (pose and
  projection, `previous_pixel`); flex hits and moving ray bundles have no reprojection (-1,
  UNKNOWN). A point behind the previous camera's near plane has no motion (-1, BEHIND); one
  outside its image keeps its motion (OUT_OF_FRAME). The projection is camera model `model`'s.
  """
  bundle = model.bundle
  project = camera.make_project(fisheye=model.distortion)
  compact = f.compact_poses
  pose = trace.instance_pose(compact=compact, bound=f.bound)
  column = instances.pose_column(compact=compact, bound=f.bound)
  unknown = int(FlowStatus.UNKNOWN)
  miss = int(FlowStatus.MISS)
  valid = int(FlowStatus.VALID)

  @wp.func
  @device_generic
  def reproject[G: GeometryFields](
    # Model:
    geo: G,
    # Data in:
    tmp: Temporal,
    # In:
    pix: camera.Pixel,
    worldid: int,
    hit: trace.Hit,
    origin: wp.vec3,
    direction: wp.vec3,
    depth: float,
  ) -> tuple[wp.vec2, float, int]:
    """(motion [px], planar depth in the previous frame: 0 miss, -1 invalid, `FlowStatus`
    without the occlusion test) of a first hit."""
    # A reset view has no previous frame: stale poses must not affect its motion.
    if tmp.count[worldid, pix.camid] == 0:
      return wp.vec2(0.0), -1.0, wp.static(unknown)
    prev = previous_pixel(tmp, pix, worldid)
    camera_static = prev.pos == pix.pos and prev.rot == pix.rot and prev.fovy == pix.fovy and prev.intrinsics == pix.intrinsics
    camera_static = camera_static and prev.distortion == pix.distortion
    g = hit.instance_id
    if g == wp.static(trace.FLEX_HIT):
      return wp.vec2(0.0), -1.0, wp.static(unknown)  # flex: no previous vertex positions
    if g == -1:
      if camera_static:
        return wp.vec2(0.0), 0.0, wp.static(miss)
      if wp.static(bundle):
        return wp.vec2(0.0), -1.0, wp.static(miss)
      if pix.lens.projection == wp.static(int(ProjectionType.ORTHOGRAPHIC)):
        return wp.vec2(0.0), -1.0, wp.static(miss)
      # Background at infinity: the direction seen by both cameras (rotation and projection).
      uv, ok = project(pix, wp.transpose(pix.rot) @ direction)
      uv_prev, ok_prev = project(prev, wp.transpose(prev.rot) @ direction)
      if ok and ok_prev:
        return uv_prev - uv, 0.0, wp.static(miss)
      return wp.vec2(0.0), -1.0, wp.static(miss)
    point = origin + direction * hit.dist
    gpos, gmat = pose(geo, worldid, g)
    # Shared instances of a compact layout are static (no previous pose column).
    ppos = gpos
    pmat = gmat
    col = column(geo, g)
    if wp.static(not compact) or col >= 0:
      ppos = tmp.prev_instance_pos[worldid, col]
      pmat = tmp.prev_instance_rot[worldid, col]
    instance_static = ppos == gpos and pmat == gmat
    if camera_static and instance_static:
      return wp.vec2(0.0), depth, wp.static(valid)
    if wp.static(bundle):
      return wp.vec2(0.0), -1.0, wp.static(unknown)
    prev_point = point
    if not instance_static:
      prev_point = pmat @ (trace.frame_inverse(gmat) @ (point - gpos)) + ppos
    uv, ok = project(pix, wp.transpose(pix.rot) @ (point - pix.pos))
    # The previous position through the previous frame's camera pose and projection.
    pc = wp.transpose(prev.rot) @ (prev_point - prev.pos)
    uv_prev, ok_prev = project(prev, pc)
    if not ok:
      return wp.vec2(0.0), -1.0, wp.static(unknown)
    zprev = camera.view_depth(prev.lens, pc)
    status = _previous_status(prev, uv_prev, ok_prev, zprev)
    # No motion where the previous camera did not image the point (behind it, or outside a
    # fisheye's model, where `project`'s position is meaningless).
    if status == wp.static(int(FlowStatus.BEHIND)) or not ok_prev:
      return wp.vec2(0.0), -1.0, status
    return uv_prev - uv, zprev, status

  return reproject


@wp.func
def _previous_depth_range(
  # Data in:
  tmp: Temporal,
  # In:
  worldid: int,
  slot: int,
  base: int,
  width: int,
  height: int,
  p: wp.vec2,
) -> wp.vec2:
  """(min, max) of the previous frame's first-hit planar depths (`Temporal.gz` slot `slot`; the
  camera's pixels from `base`) at the pixel centres around image position `p` (one pixel, or up to
  four between centres); max inf when one of them is a miss."""
  q = p - wp.vec2(0.5)
  x0 = wp.clamp(int(wp.floor(q[0])), 0, width - 1)
  x1 = wp.clamp(int(wp.ceil(q[0])), 0, width - 1)
  y0 = wp.clamp(int(wp.floor(q[1])), 0, height - 1)
  y1 = wp.clamp(int(wp.ceil(q[1])), 0, height - 1)
  lo = float(wp.inf)
  hi = float(0.0)
  for k in range(4):
    x = wp.where(k % 2 == 0, x0, x1)
    y = wp.where(k < 2, y0, y1)
    z = float(tmp.gz[worldid, slot, base + y * width + x])
    if z == 0.0:
      hi = wp.inf  # the background: nothing in front of the point
    lo = wp.min(lo, z)
    hi = wp.max(hi, z)
  return wp.vec2(lo, hi)


@wp.func
def occluded(
  # Data in:
  tmp: Temporal,
  # In:
  pix: camera.Pixel,
  worldid: int,
  slot: int,
  mv: wp.vec2,
  zexp: float,
) -> bool:
  """Whether the point now at pixel `pix`, seen at planar depth `zexp` and image position pixel
  centre + `mv` by the previous frame, was hidden there by a nearer surface (previous depths in
  slot `slot`).

  It was visible when `zexp` is at most the farthest previous depth around that position plus
  their spread and the depth precision: a continuous surface's depth inside a pixel cell stays
  within its corners' range widened by that range (a plane or a convex surface stays within the
  range; a concave crease exceeds it by at most a pixel's depth change), while an occluder hides
  the point behind all four corners.
  """
  p = wp.vec2(float(pix.px) + 0.5, float(pix.py) + 0.5) + mv
  r = _previous_depth_range(tmp, worldid, slot, pix.camid * pix.width * pix.height, pix.width, pix.height, p)
  z = wp.min(zexp, wp.static(HALF_MAX))
  return z > r[1] + (r[1] - r[0]) + r[1] * wp.static(OCCLUSION_TOLERANCE)


@specialize
def make_write_history(f: Features, model: primary.CameraModel) -> WriteHistory:
  """write_history(geo, out, tmp, pix, worldid, adr, hit, origin, direction, depth_axis): the
  first hit's motion and motion status for the outputs (through camera model `model`'s
  projection), and its depth for the next frame's occlusion test."""
  reproject = make_reproject(f, model)

  @wp.func
  @device_generic
  def write_history[G: GeometryFields](
    # Model:
    geo: G,
    # Out:
    out: OutputArrays,
    tmp: Temporal,
    # In:
    pix: camera.Pixel,
    worldid: int,
    adr: int,
    hit: trace.Hit,
    origin: wp.vec3,
    direction: wp.vec3,
    depth_axis: wp.vec3,
  ):
    """Motion / motion status of the pixel's first hit for the outputs, its depth for the next
    frame."""
    if wp.static(f.motion):
      depth = float(0.0)
      if hit.instance_id != -1:
        depth = camera.eye_distance(hit.dist, depth_axis, pix.lens.znear) * -depth_axis[2]
      # Depth slots by frame parity: this frame writes one, reads the other.
      dslot = tmp.frame[worldid] % 2
      mv, zexp, status = reproject(geo, tmp, pix, worldid, hit, origin, direction, depth)
      tmp.motion[worldid, adr] = mv
      if out_on(out.bits, wp.static(OUT_FLOW_STATUS)):
        if status == wp.static(int(FlowStatus.VALID)) and occluded(tmp, pix, worldid, 1 - dslot, mv, zexp):
          status = wp.static(int(FlowStatus.OCCLUDED))
        u8s(out, out.table.flow_status, 1)[worldid, camera.buffer_index(pix)] = wp.uint8(status)
        # The previous frame's depth of the occlusion test.
        tmp.gz[worldid, dslot, adr] = wp.float16(wp.min(depth, wp.static(HALF_MAX)))

  return write_history
