"""Ground-truth images computed from the last render's outputs, outside the render.

Opt-in: nothing runs or is allocated until asked (`Renderer.camera_normals` / `Renderer.nocs`).

* camera-frame normals: the "normal" output (world space, facing the camera) rotated into each
  camera's OpenCV frame (x right, y down, z forward; `camera_export`'s conventions), zero on a miss;
* NOCS (normalized object coordinates, Wang et al. 2019): the first hit's point in its instance's
  own frame, from the depth output unprojected through the camera ray (`pointcloud.make_unproject`:
  pose, projection, intrinsics, lens distortion), the instance of the "seg" output's id (the
  inverse of the segment table, `instance_of`: one instance per id) and the instance's pose,
  centred on the instance's local box and divided by the box's diagonal, + 0.5: every point of the
  instance's box lies in the unit cube, its centre at (0.5, 0.5, 0.5); zero on a miss and on hits
  without an instance (deformables). The local box is the one the scene BVH bounds the instance by
  (`trace/accel/bounds.world_aabb` at the identity rotation: a mesh's vertex box, a primitive's from
  its per-world size; an infinite plane's is huge: its NOCS ~0.5).

One launch each over (nworld, pixels) of the flat output order (camera c's pixels at its buffer
address, row-major), no host read; with `out` (a previous result) nothing is allocated, so a
graph can capture it. Module-level kernels (the prebuilt bundle holds them): the unprojection
compiles the lens models in, exact for undistorted cameras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi.rays import frame_inverse
from raywarp.core.kernels.abi.structs import Lens, lens_coefficients
from raywarp.core.kernels.abi.tables import in_table
from raywarp.core.outputs.pointcloud import make_unproject
from raywarp.core.trace.accel.bounds import world_aabb

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray

wp.set_module_options({"enable_backward": False})

_unproject = make_unproject(distortion=True)


@wp.struct
class Views:
  """The last render's cameras (`cameras.CameraRig`) and per-world projection.

  Attributes:
    lens: (ncam,) the camera constants (buffer address, projection, sensor, clip planes).
    res: (ncam,) resolutions; pos, rot: (nworld, ncam) the last render's camera poses.
    fovy, intrinsics, distortion: (1 or nworld, ncam) the frame's per-world projection.
  """

  lens: wp.array1d[Lens]
  res: wp.array1d[wp.vec2i]
  pos: wp.array2d[wp.vec3]
  rot: wp.array2d[wp.mat33]
  fovy: wp.array2d[float]
  intrinsics: wp.array2d[wp.vec4]
  distortion: wp.array2d[lens_coefficients]


@wp.struct
class Instances:
  """The instances' shapes and the frame's poses (`Frame`, `sharing.Layout`).

  Attributes:
    shape: (ninstance,) `ShapeType`; mesh, size: (1 or nworld, ninstance) the frame's mesh ids and
      sizes; mesh_centre, mesh_half: (nmesh,) the meshes' local boxes; hfield_half: (nhfield,).
    pose_slot: (ninstance,) per-world pose column (>= 0) or -1 - shared slot; compact: the layout
      keeps shared poses in `shared_pos` / `shared_rot` (else in world 0's columns).
    pos, rot: (nworld, ncolumn) the frame's per-world poses; shared_pos, shared_rot: (nshared,).
  """

  shape: wp.array1d[int]
  mesh: wp.array2d[int]
  size: wp.array2d[wp.vec3]
  mesh_centre: wp.array1d[wp.vec3]
  mesh_half: wp.array1d[wp.vec3]
  hfield_half: wp.array1d[wp.vec3]
  pose_slot: wp.array1d[int]
  compact: bool
  pos: wp.array2d[wp.vec3]
  rot: wp.array2d[wp.mat33]
  shared_pos: wp.array1d[wp.vec3]
  shared_rot: wp.array1d[wp.mat33]


@wp.func
def _camera_of(
  # In:
  lens: wp.array1d[Lens],
  i: int,
) -> int:
  """The camera whose pixels hold flat index `i` (the last one starting at or before it)."""
  c = int(0)
  for k in range(1, lens.shape[0]):
    if lens[k].adr <= i:
      c = k
  return c


@wp.func
def _to_opencv(
  # In:
  rot: wp.mat33,
  v: wp.vec3,
) -> wp.vec3:
  """World vector `v` in the OpenCV frame of a camera of frame `rot` (columns x right, y up, z
  backward)."""
  local = wp.transpose(rot) @ v
  return wp.vec3(local[0], -local[1], -local[2])


@wp.kernel
def camera_normals(
  # In:
  lens: wp.array1d[Lens],
  rot: wp.array2d[wp.mat33],
  normal: wp.array2d[wp.vec3],
  normal16: wp.array2d[wp.vec3h],
  half: bool,
  skip: wp.array1d[bool],
  # Out:
  out: wp.array2d[wp.vec3],
):
  """Pixel i of world w: its world-space normal (float32, or float16 with `half`) in its
  camera's OpenCV frame; untouched for a world the last render skipped (`skip`, 1 or nworld
  entries: its image is older than its camera's pose)."""
  w, i = wp.tid()
  if skip[wp.min(w, skip.shape[0] - 1)]:
    return
  n = wp.vec3(0.0)
  if half:
    n = wp.vec3(normal16[w, i])
  else:
    n = normal[w, i]
  out[w, i] = _to_opencv(rot[w, _camera_of(lens, i)], n)


@wp.func
def _pose(
  # Data in:
  inst: Instances,
  # In:
  w: int,
  k: int,
) -> tuple[wp.vec3, wp.mat33]:
  """(position, rotation) of instance k in world w (its per-world column, or its shared slot)."""
  slot = inst.pose_slot[k]
  if inst.compact:
    if slot >= 0:
      return inst.pos[w, slot], inst.rot[w, slot]
    return inst.shared_pos[-1 - slot], inst.shared_rot[-1 - slot]
  row = wp.where(slot >= 0, w, 0)  # a shared instance's pose: world 0's column
  return inst.pos[row, k], inst.rot[row, k]


@wp.func
def _local_box(
  # Data in:
  inst: Instances,
  # In:
  w: int,
  k: int,
) -> tuple[wp.vec3, wp.vec3]:
  """(lower, upper) of instance k's box in its own frame (as the scene BVH bounds it)."""
  shape = inst.shape[k]
  size = inst.size[w % inst.size.shape[0], k]
  centre = wp.vec3(0.0)
  hfield = shape == wp.static(int(ShapeType.HFIELD))
  if shape == wp.static(int(ShapeType.MESH)) or shape == wp.static(int(ShapeType.SDF)) or hfield:
    mesh = inst.mesh[w % inst.mesh.shape[0], k]
    if mesh < 0:
      return wp.vec3(0.0), wp.vec3(0.0)
    if hfield:
      size = inst.hfield_half[in_table(mesh, inst.hfield_half.shape[0])]
    else:
      m = in_table(mesh, inst.mesh_half.shape[0])
      size = inst.mesh_half[m]
      centre = inst.mesh_centre[m]
  return world_aabb(shape, centre, wp.identity(3, dtype=float), size)


@wp.kernel
def nocs(
  # Data in:
  views: Views,
  inst: Instances,
  # In:
  depth: wp.array2d[float],
  depth16: wp.array2d[wp.float16],
  half: bool,
  radial: bool,
  seg: wp.array2d[int],
  instance_of: wp.array1d[int],
  skip: wp.array1d[bool],
  # Out:
  out: wp.array2d[wp.vec3],
):
  """Pixel i of world w: its first hit's normalized object coordinates (see the module doc;
  `instance_of`: the instance of each segment id, -1 for none); untouched for a world the last
  render skipped (`skip`, 1 or nworld entries: its image is older than the poses)."""
  w, i = wp.tid()
  if skip[wp.min(w, skip.shape[0] - 1)]:
    return
  s = seg[w, i]
  k = int(-1)
  if s >= 0 and s < instance_of.shape[0]:
    k = instance_of[s]
  if k < 0:
    out[w, i] = wp.vec3(0.0)
    return
  c = _camera_of(views.lens, i)
  lens = views.lens[c]
  d = float(0.0)
  if half:
    d = float(depth16[w, i])
  else:
    d = depth[w, i]
  point = _unproject(
    lens,
    views.fovy[w % views.fovy.shape[0], c],
    views.intrinsics[w % views.intrinsics.shape[0], c],
    views.distortion[w % views.distortion.shape[0], c],
    views.pos[w, c],
    views.rot[w, c],
    views.res[c],
    i - lens.adr,
    d,
    radial,
  )
  pos, rot = _pose(inst, w, k)
  local = frame_inverse(rot) @ (point - pos)  # a mesh instance's frame s R: its mesh frame
  lower, upper = _local_box(inst, w, k)
  diagonal = wp.max(wp.length(upper - lower), 1.0e-12)
  out[w, i] = (local - 0.5 * (lower + upper)) / diagonal + wp.vec3(0.5)


def image_array(like: AnyDeviceArray, out: AnyDeviceArray | None, name: str) -> wp.array[wp.vec3, [int, int]]:
  """`out` checked against flat output `like` ((nworld, pixels) vec3 on its device), or a new one.

  Raises:
    ValueError: for an `out` of another shape, dtype or device.
  """
  shape = (like.shape[0], like.shape[1])
  if out is None:
    return wp.zeros(shape, dtype=wp.vec3, device=like.device)
  if out.shape != shape or out.dtype != wp.vec3 or out.device != like.device:
    raise ValueError(
      f"{name}(out=...) needs a {shape} vec3 array on {like.device}, got {out.dtype} {out.shape} on {out.device}"
    )
  return out


def instance_of(segments: np.ndarray[[int, int], np.dtype[np.int32]], device: wp.Device) -> wp.array[int, [int]]:
  """The (max id + 1,) instance of each segment id of a (1, ninstance) segment table (-1: none).

  Raises:
    ValueError: for per-world segment tables, or an id that several instances write.
  """
  if segments.shape[0] != 1:
    raise ValueError("nocs: per-world segment tables (Renderer.set_segment_table) have no instance per id")
  ids = segments[0]
  kept = ids[ids >= 0]
  if len(np.unique(kept)) != len(kept):
    raise ValueError("nocs needs one instance per segment id (the seg output names the hit instance)")
  table = np.full(int(kept.max()) + 1 if len(kept) else 1, -1, np.int32)
  table[kept] = np.flatnonzero(ids >= 0).astype(np.int32)
  return wp.array(table, dtype=int, device=device)
