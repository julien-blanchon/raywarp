"""World-space point clouds of the depth output (`Renderer.point_cloud`): every hit pixel of the
chosen cameras unprojected through its camera ray (pose, projection, intrinsics, lens distortion:
the inverse of `trace/rays/camera.project`), fused per world, optionally filtered by segment / class
id and resampled to a fixed number of points.

Outside the render: three small launches over the last render's outputs (count the kept pixels
per block of each world, prefix-sum the blocks, gather the points), no host synchronisation; with
`out` (a previous cloud of the same request) nothing is allocated, so a graph can capture it. The
worlds the last render skipped (`render(active=...)`) keep their rows (their kept depth is older
than this frame's poses). One gather kernel serves every camera: the lens models are compiled in
and a pinhole camera (no coefficients) takes the pinhole ray at run time
(`trace/rays/camera.distorted_frame_ray`).

Resampling to N points per world is deterministic: stratified over the kept pixels in camera /
row-major order, one random offset per world from `seed` (no point twice when a world keeps at
least N pixels; fewer are repeated cyclically, `count` says how many are distinct).
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal, final

import numpy as np
import warp as wp

from raywarp._base import arguments
from raywarp.core.kernels.abi.structs import Lens, lens_coefficients
from raywarp.core.kernels.generic import specialize
from raywarp.core.trace.rays import camera

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.outputs.render_outputs import Buffers
  from raywarp.core.pipeline.cameras import CameraRig
  from raywarp.settings import DepthOutput

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Pixels per counting block (the gather scans at most one block per point).
BLOCK: Final = 256
# How the color is read (`_Sources.color_mode`), by buffer: none, packed RGBA, RGB bytes, uint8 /
# float16 planes.
_COLOR_MODES: Final = {None: 0, "rgb": 1, "rgb_hwc": 2, "chw8": 3, "chw16": 4}


@wp.struct
class _Sources:
  """The last render's buffers and cameras a cloud reads (placeholders (1, 1) for the unused).

  Attributes:
    lens: (ncam,) the renderer's camera constants.
    fovy, intrinsics, distortion: (1 or nworld, ncam) the frame's per-world projection.
    pos, rot: (nworld, ncam) the last render's camera poses.
    res: (ncam,) camera resolutions.
    cams: (nsel,) the chosen cameras; start: (nsel + 1,) their first pixel in the fused order.
    miss: (nsel,) the depth a miss writes (DepthOutput.miss "far": the camera's zfar), 0 otherwise.
    depth, depth16: the float32 / float16 depth buffer (the other a placeholder), half: which.
    radial: the depth is the distance along the ray (else planar).
    ids, keep: the id buffer read per point and the ids kept (empty: all).
    rgba, rgb3, chw8, chw16, color_mode: the color buffers and which one holds the color.
  """

  lens: wp.array1d[Lens]
  fovy: wp.array2d[float]
  intrinsics: wp.array2d[wp.vec4]
  distortion: wp.array2d[lens_coefficients]
  pos: wp.array2d[wp.vec3]
  rot: wp.array2d[wp.mat33]
  res: wp.array1d[wp.vec2i]
  cams: wp.array1d[int]
  start: wp.array1d[int]
  miss: wp.array1d[float]
  depth: wp.array2d[float]
  depth16: wp.array2d[wp.float16]
  half: bool
  radial: bool
  ids: wp.array2d[int]
  keep: wp.array1d[int]
  rgba: wp.array2d[wp.uint32]
  rgb3: wp.array2d[wp.vec3ub]
  chw8: wp.array2d[wp.uint8]
  chw16: wp.array2d[wp.float16]
  color_mode: int


@wp.func
def _locate(
  # Data in:
  src: _Sources,
  # In:
  i: int,
) -> wp.vec3i:
  """(chosen camera index, pixel id, flat buffer index) of fused pixel `i`."""
  k = int(0)
  while i >= src.start[k + 1]:
    k += 1
  c = src.cams[k]
  p = i - src.start[k]
  return wp.vec3i(k, p, src.lens[c].adr + p)


@wp.func
def _depth(
  # Data in:
  src: _Sources,
  # In:
  w: int,
  adr: int,
) -> float:
  """The depth buffer's value at (world `w`, flat index `adr`)."""
  if src.half:
    return float(src.depth16[w, adr])
  return src.depth[w, adr]


@wp.func
def _kept(
  # Data in:
  src: _Sources,
  # In:
  w: int,
  i: int,
) -> bool:
  """Whether fused pixel `i` of world `w` hit a surface (a finite positive depth, not the miss
  value) whose id is kept."""
  loc = _locate(src, i)
  d = _depth(src, w, loc[2])
  if not (d > 0.0 and wp.isfinite(d) and d != src.miss[loc[0]]):
    return False
  if src.keep.shape[0] == 0:
    return True
  v = src.ids[w, loc[2]]
  for j in range(src.keep.shape[0]):
    if src.keep[j] == v:
      return True
  return False


@wp.func
def _skipped(
  # In:
  skip: wp.array1d[bool],
  w: int,
) -> bool:
  """Whether the last render skipped world `w` (`skip`: 1 or nworld entries)."""
  return skip[wp.min(w, skip.shape[0] - 1)]


@wp.kernel
def _count(
  # Data in:
  src: _Sources,
  # In:
  npix: int,
  skip: wp.array1d[bool],
  # Out:
  counts: wp.array2d[int],
):
  """Kept pixels of block b of world w (a skipped world's: untouched)."""
  w, b = wp.tid()
  if _skipped(skip, w):
    return
  n = int(0)
  for i in range(b * wp.static(BLOCK), wp.min((b + 1) * wp.static(BLOCK), npix)):
    if _kept(src, w, i):
      n += 1
  counts[w, b] = n


@wp.kernel
def _offsets(
  # In:
  skip: wp.array1d[bool],
  # In / Out:
  counts: wp.array2d[int],
  # Out:
  total: wp.array1d[int],
):
  """World w's block counts -> exclusive prefix sums; its kept pixels (a skipped world's:
  untouched)."""
  w = wp.tid()
  if _skipped(skip, w):
    return
  s = int(0)
  for b in range(counts.shape[1]):
    n = counts[w, b]
    counts[w, b] = s
    s += n
  total[w] = s


@wp.func
def _kth(
  # Data in:
  src: _Sources,
  counts: wp.array2d[int],
  # In:
  w: int,
  k: int,
  npix: int,
) -> int:
  """The fused pixel index of world `w`'s `k`-th kept pixel (`counts`: exclusive block offsets)."""
  lo = int(0)
  hi = counts.shape[1] - 1
  while lo < hi:  # the last block whose offset is <= k
    mid = (lo + hi + 1) // 2
    if counts[w, mid] <= k:
      lo = mid
    else:
      hi = mid - 1
  r = k - counts[w, lo]
  i = lo * wp.static(BLOCK)
  end = wp.min(i + wp.static(BLOCK), npix)
  while i < end:
    if _kept(src, w, i):
      if r == 0:
        return i
      r -= 1
    i += 1
  return -1


@wp.func
def _color(
  # Data in:
  src: _Sources,
  # In:
  w: int,
  loc: wp.vec3i,
) -> wp.vec3ub:
  """The sRGB bytes of the display color at `loc` (`_locate`) in the render's color layout."""
  adr = loc[2]
  if src.color_mode == 1:
    v = int(src.rgba[w, adr])  # packed RGBA bytes, R lowest
    return wp.vec3ub(wp.uint8(v & 255), wp.uint8((v >> 8) & 255), wp.uint8((v >> 16) & 255))
  if src.color_mode == 2:
    return src.rgb3[w, adr]
  c = src.cams[loc[0]]
  npix = src.res[c][0] * src.res[c][1]
  base = 3 * src.lens[c].adr + loc[1]
  if src.color_mode == 3:
    return wp.vec3ub(src.chw8[w, base], src.chw8[w, base + npix], src.chw8[w, base + 2 * npix])
  out = wp.vec3ub()
  for ch in range(3):
    out[ch] = wp.uint8(int(wp.clamp(float(src.chw16[w, base + ch * npix]), 0.0, 1.0) * 255.0 + 0.5))
  return out


@specialize
def make_unproject(*, distortion: bool):
  """unproject(lens, fovy, intrinsics, coefficients, pos, rot, res, pixel, depth, radial) -> the
  world-space point at `depth` (planar, or along the ray with `radial`) of row-major `pixel` of a
  camera of resolution `res`, constants `lens`, pose (pos, rot) and per-world projection (fovy,
  intrinsics, lens coefficients): the camera's primary ray through the pixel centre
  (`trace/rays/camera.make_camera_ray`; `distortion`: through the lens models), whose depth counts
  from the eye (fisheye: along the ray, whose depth axis is the ray itself)."""
  camera_ray = camera.make_camera_ray(distortion=distortion)

  @wp.func
  def unproject(
    # In:
    lens: Lens,
    fovy: float,
    intrinsics: wp.vec4,
    coefficients: lens_coefficients,
    pos: wp.vec3,
    rot: wp.mat33,
    res: wp.vec2i,
    pixel: int,
    depth: float,
    radial: bool,
  ) -> wp.vec3:
    """The world-space point of `pixel` at `depth` (see `make_unproject`)."""
    pix = camera.Pixel()
    pix.width = res[0]
    pix.height = res[1]
    pix.pixelid = pixel
    pix.py = pixel // pix.width
    pix.px = pixel - pix.py * pix.width
    pix.lens = lens
    pix.fovy = fovy
    pix.intrinsics = intrinsics
    pix.distortion = coefficients
    pix.rot = rot
    pix.pos = pos
    # The ray starts on the near plane.
    origin, direction, _, axis = camera_ray(pix, 1, 0, 0)
    d = depth
    if not radial:
      d = d / wp.max(-axis[2], 1.0e-6)  # planar -> along the ray
    return origin + direction * (d - camera.near_ray_dist(axis, lens.znear))

  return unproject


# The unprojection of every camera: the lens models compiled in (a pinhole camera's zero
# coefficients take the pinhole ray at run time).
_point_of = make_unproject(distortion=True)


@wp.func
def _unproject(
  # Data in:
  src: _Sources,
  # In:
  w: int,
  i: int,
) -> wp.vec3:
  """The world-space point of fused pixel `i` of world `w` from its depth."""
  loc = _locate(src, i)
  c = src.cams[loc[0]]
  return _point_of(
    src.lens[c],
    src.fovy[w % src.fovy.shape[0], c],
    src.intrinsics[w % src.intrinsics.shape[0], c],
    src.distortion[w % src.distortion.shape[0], c],
    src.pos[w, c],
    src.rot[w, c],
    src.res[c],
    loc[1],
    _depth(src, w, loc[2]),
    src.radial,
  )


@wp.kernel
def _gather(
  # Data in:
  src: _Sources,
  counts: wp.array2d[int],
  total: wp.array1d[int],
  # In:
  npix: int,
  resample: bool,
  seed: int,
  with_ids: bool,
  skip: wp.array1d[bool],
  # Out:
  points: wp.array2d[wp.vec3],
  colors: wp.array2d[wp.vec3ub],
  ids: wp.array2d[int],
):
  """Point j of world w: the k-th kept pixel (k = j, or stratified with `resample`) unprojected
  through its camera's ray; a skipped world's: untouched."""
  w, j = wp.tid()
  if _skipped(skip, w):
    return
  n = total[w]
  npoint = points.shape[1]
  k = j
  if resample and n >= npoint:
    u = wp.randf(wp.rand_init(seed, w))
    k = wp.min(int((float(j) + u) * float(n) / float(npoint)), n - 1)
  elif resample and n > 0:
    k = j % n
  i = int(-1)
  if k < n:
    i = _kth(src, counts, w, k, npix)
  if i < 0:
    points[w, j] = wp.vec3(0.0)
    if src.color_mode != 0:
      colors[w, j] = wp.vec3ub(wp.uint8(0))
    if with_ids:
      ids[w, j] = -1
    return
  points[w, j] = _unproject(src, w, i)
  loc = _locate(src, i)
  if src.color_mode != 0:
    colors[w, j] = _color(src, w, loc)
  if with_ids:
    ids[w, j] = src.ids[w, loc[2]]


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class PointCloud[NWorld: IntVar, N: IntVar]:
  """World-space points of the depth output, per world (`Renderer.point_cloud`).

  Attributes:
    points: (nworld, N) vec3 world-space points [m] (NumPy: (nworld, N, 3)); zero past `count`
      without resampling.
    count: (nworld,) int32 kept pixels per world (distinct points; resampled clouds repeat them
      when fewer than N).
    colors: (nworld, N) vec3ub sRGB display color of each point, None unless asked.
    ids: (nworld, N) int32 segment / class id of each point (the `labels` output), -1 past
      `count`; None without `labels`.
  """

  points: wp.array[wp.vec3, [NWorld, N]]
  count: wp.array[wp.int32, [NWorld]]
  colors: wp.array[wp.vec3ub, [NWorld, N]] | None
  ids: wp.array[wp.int32, [NWorld, N]] | None
  # The request it holds and its scratch (`compute(out=...)`).
  _key: tuple[object, ...] = dataclasses.field(repr=False, compare=False)
  _counts: wp.array[wp.int32, [NWorld, int]] = dataclasses.field(repr=False, compare=False)
  # The device tables of the last call (at most one): a call over the same arrays reuses them.
  _cache: list[_Tables] = dataclasses.field(default_factory=list, repr=False, compare=False)


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _Tables:
  """A cloud's device tables, reused by every call over the same arrays (no allocation, no
  host-to-device copy: capture-safe).

  Attributes:
    identity: the identities of the arrays they were built from, and what they copy by value.
    arrays: those arrays (held: the identities stay those of live arrays).
    src: the kernels' sources; npix: fused pixels per world.
    colors, ids: the cloud's color / id arrays, or (1, 1) placeholders.
  """

  identity: tuple[object, ...]
  arrays: tuple[AnyDeviceArray, ...]
  src: _Sources
  npix: int
  colors: AnyDeviceArray
  ids: AnyDeviceArray


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Request:
  """What a cloud holds (`Renderer.point_cloud`'s arguments).

  Attributes:
    npoints: points per world (resampled), None: every kept pixel (N = the cameras' pixels).
    cameras: the cameras fused (renderer camera indices), None: all.
    labels: the id output read per point ("seg" or "semantic"), None: none.
    keep: the ids kept (needs `labels`), None: all.
    colors: gather the display color.
    seed: the resampling's random seed.
  """

  npoints: int | None = None
  cameras: Sequence[int] | None = None
  labels: Literal["seg", "semantic"] | None = None
  keep: Sequence[int] | None = None
  colors: bool = False
  seed: int = 0

  def __post_init__(self) -> None:
    """Stores `cameras` and `keep` as tuples of ints: a request compares by value (`out=`), and a
    caller's list or NumPy array changed later never reaches a cloud's cached tables. `npoints`,
    `seed` and the kept ids are checked here: the kernels read them as int32 (a larger value
    would wrap, a bool would count as 1).

    Raises:
      TypeError: for a camera or id that is not an integer.
      ValueError: for npoints that is not an integer >= 1, a seed or kept id outside int32.
    """
    for name in ("cameras", "keep"):
      values = getattr(self, name)
      if values is not None:
        object.__setattr__(self, name, tuple(operator.index(v) for v in values))
    if self.npoints is not None:
      object.__setattr__(
        self, "npoints", arguments.check_int("point_cloud: npoints", self.npoints, low=1, high=arguments.INT32_MAX)
      )
    object.__setattr__(
      self, "seed", arguments.check_int("point_cloud: seed", self.seed, low=arguments.SEED_MIN, high=arguments.SEED_MAX)
    )
    for i in self.keep or ():
      arguments.check_int("point_cloud: keep id", i, low=arguments.INT32_MIN, high=arguments.INT32_MAX)


def compute[NWorld: IntVar](  # noqa: PLR0913  # the last render's tables, keyword-only
  req: Request,
  *,
  rig: CameraRig[NWorld],
  buffers: Buffers[NWorld],
  depth: DepthOutput | None,
  view: tuple[AnyDeviceArray, AnyDeviceArray, AnyDeviceArray],
  skip: wp.array[bool, [int]],
  out: PointCloud[NWorld, int] | None = None,
) -> PointCloud[NWorld, int]:
  """The point cloud `req` of the last render (`rig`: its cameras; `buffers`: its outputs;
  `depth`: the depth output's spec; `view`: the frame's (fovy, intrinsics, distortion) tables;
  `skip`: the worlds it skipped, 1 or nworld entries, whose rows are left as they are), written
  into `out` when given (same request: no allocation).

  Raises:
    ValueError: without the depth output (or the color / id output asked for), for ray-bundle
      cameras, an unknown camera, npoints < 1, `keep` without `labels`, or an `out` of another
      request.
  """
  arrays = (rig.lens, rig.pos, rig.rot, rig.res, *view, *buffers.flat.values())
  # The arrays by identity (held by the cache), and what the tables copy by value: each camera's
  # clip distances (the miss depth of "far", `Renderer.set_clip` writes them in place) and the
  # depth output's spec.
  identity = (*(id(a) for a in arrays), tuple(rig.clips), depth)
  cached = out._cache if out is not None else []  # noqa: SLF001  # the cloud's own tables
  tables = cached[0] if cached and cached[0].identity == identity else None
  if tables is None:
    src, npix = _sources(req, rig=rig, buffers=buffers, depth=depth, view=view)
  else:
    src, npix = tables.src, tables.npix
  nworld = rig.pos.shape[0]
  npoint = req.npoints if req.npoints is not None else npix
  key = (req, nworld, npix)
  if out is None:
    out = _allocate(req, key, nworld=nworld, npoint=npoint, nblock=(npix + BLOCK - 1) // BLOCK, device=rig.device)
  elif out._key != key:  # noqa: SLF001  # the cloud's own request
    raise ValueError("point_cloud(out=...) needs a cloud of the same request (npoints, cameras, labels, keep, colors, seed)")
  device = rig.device
  counts = out._counts  # noqa: SLF001
  if tables is None:
    tables = _Tables(
      identity=identity,
      arrays=arrays,
      src=src,
      npix=npix,
      colors=out.colors if out.colors is not None else _placeholder(wp.vec3ub, device),
      ids=out.ids if out.ids is not None else _placeholder(wp.int32, device),
    )
    out._cache[:] = [tables]  # noqa: SLF001
  wp.launch(_count, dim=counts.shape, inputs=[src, npix, skip], outputs=[counts], device=device)
  wp.launch(_offsets, dim=nworld, inputs=[skip], outputs=[counts, out.count], device=device)
  wp.launch(
    _gather,
    dim=(nworld, npoint),
    inputs=[src, counts, out.count, npix, req.npoints is not None, req.seed, req.labels is not None, skip],
    outputs=[out.points, tables.colors, tables.ids],
    device=device,
  )
  return out


def _allocate[NWorld: IntVar](
  req: Request, key: tuple[object, ...], *, nworld: Int[NWorld], npoint: int, nblock: int, device: wp.Device
) -> PointCloud[NWorld, int]:
  """A new cloud of `req` for `nworld` worlds of `npoint` points (`nblock` counting blocks)."""
  shape = (nworld, npoint)
  return PointCloud(
    points=wp.zeros(shape, dtype=wp.vec3, device=device),
    count=wp.zeros(nworld, dtype=wp.int32, device=device),
    colors=wp.zeros(shape, dtype=wp.vec3ub, device=device) if req.colors else None,
    ids=wp.zeros(shape, dtype=wp.int32, device=device) if req.labels is not None else None,
    _key=key,
    _counts=wp.zeros((nworld, nblock), dtype=wp.int32, device=device),
  )


def _sources[NWorld: IntVar](
  req: Request,
  *,
  rig: CameraRig[NWorld],
  buffers: Buffers[NWorld],
  depth: DepthOutput | None,
  view: tuple[AnyDeviceArray, AnyDeviceArray, AnyDeviceArray],
) -> tuple[_Sources, int]:
  """(the kernels' sources, fused pixels per world) of `req` (`compute`)."""
  device = rig.device
  found = buffers.flat_of("depth")
  if depth is None or found is None:
    raise ValueError("point_cloud needs the depth output (outputs.depth)")
  if rig.rays is not None:
    raise ValueError("point_cloud: ray-bundle cameras have no pinhole ray to unproject")
  cams = list(range(rig.ncam)) if req.cameras is None else list(req.cameras)
  if not cams or any(not 0 <= c < rig.ncam for c in cams):
    raise ValueError(f"point_cloud: cameras must be indices in [0, {rig.ncam}), got {cams}")
  sizes = [rig.sizes[c][0] * rig.sizes[c][1] for c in cams]
  src = _Sources()
  src.lens, (src.fovy, src.intrinsics, src.distortion), src.pos, src.rot, src.res = rig.lens, view, rig.pos, rig.rot, rig.res
  src.cams = wp.array(cams, dtype=int, device=device)
  src.start = wp.array(np.concatenate([[0], np.cumsum(sizes)]), dtype=int, device=device)
  far = [rig.clips[c][1] for c in cams]
  miss = [(f if f > 0.0 else np.inf) if depth.miss == "far" else 0.0 for f in far]
  half = depth.dtype == "float16"
  src.miss = wp.array(np.asarray(miss, np.float16 if half else np.float32).astype(np.float32), dtype=float, device=device)
  src.half, src.radial = half, depth.convention == "radial"
  src.depth = _placeholder(float, device) if half else found[1]
  src.depth16 = found[1] if half else _placeholder(wp.float16, device)
  _id_sources(src, req, buffers, device)
  _color_sources(src, req, buffers, device)
  return src, int(sum(sizes))


def _placeholder(dtype: type, device: wp.Device) -> AnyDeviceArray:
  """A (1, 1) array standing for a buffer the cloud does not read."""
  return wp.zeros((1, 1), dtype=dtype, device=device)


def _id_sources(src: _Sources, req: Request, buffers: Buffers, device: wp.Device) -> None:
  """The id buffer and kept ids of `req` into `src`."""
  if req.keep is not None and req.labels is None:
    raise ValueError('point_cloud: keep filters the ids of an output: give labels="seg" or "semantic"')
  src.ids = _placeholder(int, device)
  if req.labels is not None:
    found = buffers.flat_of(req.labels)
    if found is None:
      raise ValueError(f"point_cloud: labels={req.labels!r} needs that output (outputs.{req.labels})")
    src.ids = found[1]
  keep = [] if req.keep is None else list(req.keep)
  src.keep = wp.array(np.asarray(keep, np.int32), dtype=int, device=device)


def _color_sources(src: _Sources, req: Request, buffers: Buffers, device: wp.Device) -> None:
  """The color buffer of `req` (the render's color layout) into `src`."""
  found = buffers.flat_of("rgb") if req.colors else None
  if req.colors and found is None:
    raise ValueError("point_cloud: colors=True needs the rgb output (outputs.rgb)")
  name = None if found is None else found[0]
  src.color_mode = _COLOR_MODES[name]

  def buffer(kind: str, dtype: type) -> AnyDeviceArray:
    """The color buffer when it is `kind`, else a placeholder."""
    return found[1] if found is not None and found[0] == kind else _placeholder(dtype, device)

  src.rgba = buffer("rgb", wp.uint32)
  src.rgb3 = buffer("rgb_hwc", wp.vec3ub)
  src.chw8 = buffer("chw8", wp.uint8)
  src.chw16 = buffer("chw16", wp.float16)
