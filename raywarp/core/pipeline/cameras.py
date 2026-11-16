"""The cameras of one Renderer: which cameras it renders, where they are each frame and how
their rays are shaped (`CameraRig`: specs, device tables, per-frame poses, ray bundles).

A camera is either mounted on a scene camera (`CameraSpec.mount`: its pose follows the scene
camera's, offset by `CameraSpec.local`) or free (its pose is the Renderer's `camera_pose` table,
written in place by the caller). Each camera has its own projection, resolution, clip planes,
exposure compensation, intrinsics, lens coefficients and outputs: one `kernel_structs.Lens` per
camera and its pixels at `Lens.adr` in the flat output buffers (`outputs.Buffers`); the field of
view, intrinsics and lens coefficients are per-world fields ("camera.fovy", "camera.intrinsics",
"camera.distortion") whose defaults are the rig's `fovy`, `intrinsics` and `coefficients`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.arguments import check_clip
from raywarp._base.types import CameraSlot, ProjectionType, RenderOrder
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.tables import batched
from raywarp.core.pipeline.gpu_scene.camera_checks import camera_outputs
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec
from raywarp.core.pipeline.platform.launches import CachedLaunch
from raywarp.core.trace.rays import bundle, camera

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import Table
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.settings.outputs import Outputs

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.kernel
def _pose_cameras(
  # Model:
  mount: wp.array1d[int],
  local_pos: wp.array1d[wp.vec3],
  local_rot: wp.array1d[wp.mat33],
  has_local: wp.array1d[bool],
  # Data in:
  scene_pos: wp.array2d[wp.vec3],
  scene_rot: wp.array2d[wp.mat33],
  free_pose: wp.array2d[wp.transformf],
  # Out:
  camera_pos: wp.array2d[wp.vec3],
  camera_rot: wp.array2d[wp.mat33],
):
  """Camera frames: a mounted camera copies (and offsets) its scene camera, a free one its pose."""
  w, c = wp.tid()
  m = mount[c]
  if m >= 0:
    pos = scene_pos[w % scene_pos.shape[0], m]
    rot = scene_rot[w % scene_rot.shape[0], m]
    if has_local[c]:
      pos = pos + rot @ local_pos[c]
      rot = rot @ local_rot[c]
  else:
    t = free_pose[w, c]
    pos = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)
    # Raw simulator quaternions need not be unit; a zero / NaN one stays NaN (no silent identity).
    rot = wp.quat_to_matrix(q * (1.0 / wp.length(q)))
  camera_pos[w, c] = pos
  camera_rot[w, c] = rot


@wp.kernel
def _camera_order(
  # Data in:
  camera_pos: wp.array2d[wp.vec3],
  camera_rot: wp.array2d[wp.mat33],
  camera_fovy: wp.array2d[float],
  camera_intrinsics: wp.array2d[wp.vec4],
  camera_distortion: wp.array2d[kernel_structs.lens_coefficients],
  # In:
  worlds: wp.array1d[int],
  tiled_rows: int,
  # Out:
  order: wp.array1d[wp.vec2i],
):
  """RenderOrder.AUTO after `order` was set to MIXED: TILED as soon as camera c of launch unit u's
  world differs from unit 0's (pose, field of view, intrinsics or lens coefficients; `worlds`: the
  render launch's, `active.ActiveWorlds.units`, empty: unit u is world u; holes compare nothing);
  every writer writes the same value."""
  u, c = wp.tid()
  w = u
  r = 0
  if worlds.shape[0] > 0:
    w = worlds[u]
    r = worlds[0]
  if w < 0 or r < 0:
    return
  same = camera_pos[w, c] == camera_pos[r, c] and camera_rot[w, c] == camera_rot[r, c]
  same = same and batched(camera_fovy, w, c) == batched(camera_fovy, r, c)
  same = same and batched(camera_intrinsics, w, c) == batched(camera_intrinsics, r, c)
  same = same and batched(camera_distortion, w, c) == batched(camera_distortion, r, c)
  if not same:
    order[0] = wp.vec2i(wp.static(int(RenderOrder.TILED)), tiled_rows)


def _scene_camera(scene: GpuScene, c: CameraSpec | str | int) -> CameraSpec:
  """The CameraSpec of `c`: itself, or the scene camera of that name or index.

  Raises:
    ValueError: for an unknown camera or mount.
    TypeError: for something else than a spec, name or index (a bool is not an index).
  """
  names = scene.camera_names
  if isinstance(c, int | np.integer) and not isinstance(c, bool):
    if not 0 <= c < len(names):
      raise ValueError(f"camera index {c} out of range: the scene has {len(names)} cameras")
    c = names[c]
  if isinstance(c, str):
    if c not in scene.cameras:
      raise ValueError(f"camera {c!r} not found among the scene cameras {names}")
    c = scene.cameras[c]
  if not isinstance(c, CameraSpec):
    raise TypeError(f"cameras must be CameraSpec, scene camera names or indices, got {type(c).__name__}")
  if c.mount is not None and c.mount not in scene.cameras:
    raise ValueError(f"camera mount {c.mount!r} not found among the scene cameras {names}")
  return c


def camera_specs(
  scene: GpuScene, cameras: Sequence[CameraSpec | str | int] | CameraSpec | str | int | None
) -> list[CameraSpec]:
  """The CameraSpecs of `cameras` (specs, scene camera names or indices; one of them: that camera
  alone, a name not read as its characters; None: every scene camera; each spec validates its own
  fields, `CameraSpec.__post_init__`).

  Raises:
    ValueError: for an unknown camera or mount, no camera, or ray bundles on some cameras only.
    TypeError: for something else than a spec, name or index.
  """
  if isinstance(cameras, str | CameraSpec | int | np.integer):
    cameras = [cameras]
  specs = [_scene_camera(scene, c) for c in (scene.camera_names if cameras is None else cameras)]
  if not specs:
    raise ValueError("a Renderer needs at least one camera")
  with_rays = [s.rays is not None for s in specs]
  if any(with_rays) and not all(with_rays):
    raise ValueError("either every camera of a Renderer has rays or none")
  return specs


def _sizes(specs: list[CameraSpec], width: int, height: int) -> list[tuple[int, int]]:
  """(width, height) of each camera (its own resolution, checked by `CameraSpec`, else the
  Renderer's, checked by the Renderer).

  Raises:
    ValueError: for ray-bundle cameras of different resolutions.
  """
  sizes = [(s.resolution[0], s.resolution[1]) if s.resolution is not None else (width, height) for s in specs]
  if specs[0].rays is not None and len(set(sizes)) > 1:
    raise ValueError("ray-bundle cameras share one resolution")
  return sizes


_PROJECTIONS: Final = {
  "perspective": ProjectionType.PERSPECTIVE,
  "orthographic": ProjectionType.ORTHOGRAPHIC,
  "fisheye": ProjectionType.FISHEYE,
}


def _lens_table[NCam: IntVar](  # noqa: PLR0913  # one sequence per Lens field group
  specs: Sequence[CameraSpec],
  sizes: Sequence[tuple[int, int]],
  clips: Sequence[tuple[float, float]],
  adr: Sequence[int],
  wants: Sequence[int],
  *,
  device: wp.Device,
  scale: int = 1,
) -> wp.array[kernel_structs.Lens, [NCam]]:
  """The (ncam,) `kernel_structs.Lens` table of the cameras of image `sizes` on `device` (`wants`:
  each camera's `Lens.wants`; `scale`: the rendered image's scale over `sizes`, `Lens.scale`)."""
  lenses = []
  for spec, size, (znear, zfar), a, want in zip(specs, sizes, clips, adr, wants, strict=True):
    lens = kernel_structs.Lens()
    lens.projection = int(_PROJECTIONS[spec.projection])
    lens.sensorsize = wp.vec2(*conformed(spec, size)[1])
    lens.znear, lens.zfar = float(znear), float(zfar)
    lens.exposure = float(2.0 ** float(spec.exposure))
    lens.adr = a
    lens.wants = want
    lens.scale = scale
    lenses.append(lens)
  return wp.array(lenses, dtype=kernel_structs.Lens, device=device)


def conformed(spec: CameraSpec, size: tuple[int, int]) -> tuple[float, tuple[float, float]]:
  """(fovy, `Lens.sensorsize`) of `spec` on an image of `size` (width, height): its own, or with
  `conform="fit"` its window extended to the image's aspect along its shorter side (Hydra's
  CameraUtilConformedWindow, CameraUtilFit: the window wider than the image keeps its width, a
  narrower one its height; the centre stays). A fovy camera's field of view widens by the same
  factor (perspective: of tan(fovy / 2); orthographic: of the view height). Without intrinsics
  the sensor is (0, 0): the kernels read fovy."""
  sensor = spec.sensor_size if spec.intrinsics is not None and spec.sensor_size is not None else (0.0, 0.0)
  sw, sh = (float(v) for v in sensor)
  fit = spec.conform == "fit" and spec.sensor_size is not None and min(spec.sensor_size) > 0.0
  if not fit or spec.sensor_size is None or spec.projection == "fisheye":
    return float(spec.fovy), (sw, sh)
  image = size[0] / size[1]
  window = float(spec.sensor_size[0]) / float(spec.sensor_size[1])
  stretch = max(window / image, 1.0)  # > 1: the window is wider, its height extended
  if spec.intrinsics is not None:
    return float(spec.fovy), (sw, sh * stretch) if stretch > 1.0 else (sh * image, sh)
  if spec.projection == "orthographic":
    return float(spec.fovy) * stretch, (0.0, 0.0)
  half = math.atan(math.tan(math.radians(0.5 * float(spec.fovy))) * stretch)
  return math.degrees(2.0 * half), (0.0, 0.0)


def _coefficients(spec: CameraSpec) -> tuple[float, ...]:
  """The `kernel_structs.lens_coefficients` of `spec`: its distortion padded with zeros to
  `kernel_structs.NLENS` (OpenCV's (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]), a
  fisheye's (k1, k2, k3, k4))."""
  k = tuple(float(v) for v in spec.distortion or ())
  return k + (0.0,) * (kernel_structs.NLENS - len(k))


# `Lens.wants` of a camera writing every output the Renderer requests.
_ALL_WANTS: Final = kernel_structs.WANT_RGB | kernel_structs.WANT_DEPTH | kernel_structs.WANT_SEG


def _wants(spec: CameraSpec, outputs: Outputs) -> int:
  """`Lens.wants` of `spec` among the Renderer's `outputs` (ValueError: `camera_outputs`)."""
  if spec.outputs is None:
    return _ALL_WANTS
  rgb, depth, seg = camera_outputs(spec.outputs, outputs)
  bits = (kernel_structs.WANT_RGB, rgb), (kernel_structs.WANT_DEPTH, depth), (kernel_structs.WANT_SEG, seg)
  return sum(bit for bit, on in bits if on)


def _has_lens_model(spec: CameraSpec) -> bool:
  """Whether `spec` needs the lens models compiled in (`rays.DISTORTION`): a fisheye, or a
  perspective camera with a nonzero distortion coefficient."""
  return spec.projection == "fisheye" or any(v != 0.0 for v in _coefficients(spec))


@final
class CameraRig[NWorld: IntVar]:
  """The cameras of a Renderer: their specs, device tables and the frame's poses.

  Attributes:
    device: the device of every table (the scene's).
    specs: the CameraSpecs, one per camera; `ncam` their count.
    sizes: (width, height) of each camera; `uniform` whether they are all equal; `adr` each
      camera's first flat pixel index in the output buffers; `nray` every camera's pixels.
    clips: (znear, zfar) of each camera (zfar 0: unbounded; `set_clip`).
    distortion: some camera has a lens model, a fisheye or lens distortion (`rays.DISTORTION`;
      also for a per-world "camera.distortion", `resolve.camera_model`).
    wants: each camera's `Lens.wants` (`set_outputs`; every output until then).
    lens: (ncam,) `kernel_structs.Lens` per camera (projection, sensor, clip, exposure, buffer
      address, outputs).
    fovy: (1, ncam) vertical field of view; `intrinsics` (1, ncam) (fx, fy, cx, cy) [sensor
      units]; `coefficients` (1, ncam) `kernel_structs.lens_coefficients`: the defaults of the
      per-world "camera.fovy" / "camera.intrinsics" / "camera.distortion" fields.
    res: (ncam,) resolution per camera.
    mount: (ncam,) scene camera index of mounted cameras (-1: free); `local_pos` / `local_rot`
      / `has_local` their offsets; `has_mount`: any camera is mounted.
    free_pose: (nworld, ncam) transforms of the free cameras (`Renderer.camera_pose`).
    pos, rot: (nworld, ncam) the frame's camera poses (`pose`).
    rays: the ray-bundle buffers (`bundle.load_camera_rays`), None for pinhole cameras.
    order: (1,) the frame's thread-to-pixel order: `RenderOrder` (AUTO resolved per frame by
      `pose`), TILED's band height (`set_order`).
  """

  def __init__(self, scene: GpuScene, specs: list[CameraSpec], *, nworld: Int[NWorld], width: int, height: int) -> None:
    """The tables of `specs` on the scene's device (see the class doc).

    Raises:
      ValueError: for a camera whose clip planes (its spec's, else the scene's) leave znear at or
        beyond zfar.
    """
    self.device = device = wp.get_device(scene.device)
    self.specs = specs
    self.ncam = len(specs)
    self.sizes = _sizes(specs, width, height)
    self.uniform = len(set(self.sizes)) == 1
    npix = [w * h for w, h in self.sizes]
    self.adr = [sum(npix[:c]) for c in range(len(npix))]
    self.nray = sum(npix)
    znear, zfar = scene.clip
    self.clips = [(s.znear if s.znear is not None else znear, s.zfar if s.zfar is not None else zfar) for s in specs]
    for c, (near, far) in enumerate(self.clips):  # a spec's plane against the scene's other one
      check_clip(near, far, f"camera {c}")
    self.distortion = any(_has_lens_model(s) for s in specs)
    self.wants = [_ALL_WANTS] * self.ncam
    self.lens: wp.array[kernel_structs.Lens, [int]] = _lens_table(
      specs, self.sizes, self.clips, self.adr, self.wants, device=device
    )
    self._scaled: dict[int, wp.array[kernel_structs.Lens, [int]]] = {}
    self._tables(scene, specs)
    self._locals(specs)
    self.free_pose: wp.array[wp.transformf, [NWorld, int]] = wp.zeros((nworld, self.ncam), dtype=wp.transformf, device=device)
    self.free_pose.fill_(wp.transform_identity())
    self.pos: wp.array[wp.vec3, [NWorld, int]] = wp.zeros((nworld, self.ncam), dtype=wp.vec3, device=device)
    self.rot: wp.array[wp.mat33, [NWorld, int]] = wp.zeros((nworld, self.ncam), dtype=wp.mat33, device=device)
    # The scene camera poses of a scene without cameras: one row, never read (no camera is mounted).
    self._no_scene: tuple[Table[wp.vec3, int, int], Table[wp.mat33, int, int]] = (
      wp.zeros((1, 1), dtype=wp.vec3, device=device),
      wp.zeros((1, 1), dtype=wp.mat33, device=device),
    )
    self.rays: bundle.RayBuffers[int, int] | None = None
    self._order = (RenderOrder.TILED, camera.TILE_ROWS)
    self._pose_launch, self._order_launch = CachedLaunch(), CachedLaunch()
    self.order: wp.array[wp.vec2i, [1]] = wp.array([wp.vec2i(*self._order)], dtype=wp.vec2i, device=device)
    # Bundle rays with a NaN / inf component (written as misses), reported by the validation.
    self.invalid_rays: wp.array[wp.int32, [1]] = wp.zeros(1, dtype=wp.int32, device=device)
    images = [s.rays for s in specs if s.rays is not None]
    if images:
      width, height = self.sizes[0]
      self.rays = bundle.load_camera_rays(images, width=width, height=height, device=device, invalid=self.invalid_rays)

  def _tables(self, scene: GpuScene, specs: list[CameraSpec]) -> None:
    """The per-camera device tables: default fovy / intrinsics / lens coefficients, resolutions,
    mounts."""
    ncam, device = self.ncam, self.device
    names = scene.camera_names
    intrinsics = [s.intrinsics if s.intrinsics is not None else (0.0, 0.0, 0.0, 0.0) for s in specs]
    mount = np.array([names.index(s.mount) if s.mount is not None else -1 for s in specs], dtype=np.int32)
    fovy = [conformed(s, size)[0] for s, size in zip(specs, self.sizes, strict=True)]
    self.fovy = wp.array(np.array([fovy], dtype=np.float32), dtype=float, device=device)
    self.intrinsics = wp.array(np.asarray(intrinsics, np.float32).reshape(1, ncam, 4), dtype=wp.vec4, device=device)
    coefficients = np.asarray([_coefficients(s) for s in specs], np.float32).reshape(1, ncam, kernel_structs.NLENS)
    self.coefficients = wp.array(coefficients, dtype=kernel_structs.lens_coefficients, device=device)
    self.res = wp.array(np.asarray(self.sizes, np.int32).reshape(ncam, 2), dtype=wp.vec2i, device=device)
    self.mount = wp.array(mount, dtype=int, device=device)
    self.has_mount = bool(np.any(mount >= 0))

  def set_order(self, order: RenderOrder, tiled_rows: int) -> None:
    """Sets the thread-to-pixel order of the next frames and TILED's band height (in place:
    captured graphs follow; AUTO: resolved by every `pose`)."""
    if (order, tiled_rows) != self._order:
      self._order = (order, tiled_rows)
      if order == RenderOrder.AUTO and self.pos.shape[0] < camera.MIXED_WORLDS:
        order = RenderOrder.TILED  # too few worlds to fill MIXED's warps: AUTO's rule, once
      if order != RenderOrder.AUTO:
        self.order.fill_(wp.vec2i(int(order), tiled_rows))

  def _resolve_order(
    self,
    fovy: Table[float, int, int],
    intrinsics: Table[wp.vec4, int, int],
    distortion: Table[kernel_structs.lens_coefficients, int, int],
    worlds: wp.array[int, [int]],
  ) -> None:
    """RenderOrder.AUTO's choice for the frame's poses (device work only): MIXED when every world
    the render launches (`worlds`: its units' worlds, empty: every world) sees through the first
    one's cameras and there are enough worlds to fill its warps (`camera.MIXED_WORLDS`), else
    TILED. A static rule: with identical cameras MIXED was faster
    than TILED on every measured scene and backend (OptiX -1.7 to -3.6 %, software -8.6 to -19 %,
    L4)."""
    order, rows = self._order
    if order != RenderOrder.AUTO:
      return
    nworld = self.pos.shape[0]
    if nworld < camera.MIXED_WORLDS:
      return  # TILED (`set_order`)
    self.order.fill_(wp.vec2i(int(RenderOrder.MIXED), rows))
    self._order_launch.launch(
      _camera_order,
      (worlds.shape[0] or nworld, self.ncam),
      inputs=[self.pos, self.rot, fovy, intrinsics, distortion, worlds, rows],
      outputs=[self.order],
      device=self.device,
    )

  def set_clip(
    self,
    znear: float | None,
    zfar: float | None,
    camera: CameraSlot | None,
  ) -> bool:
    """Changes the clip distances of camera `camera` (None: every camera); in place, no recompile.

    The values were validated by `Renderer.set_clip` (`_base/arguments.py`), the only caller.

    Returns:
      Whether any camera's clip planes changed.
    """
    clips = list(self.clips)
    for c in range(self.ncam) if camera is None else (camera,):
      near, far = clips[c]
      clips[c] = (near if znear is None else znear, far if zfar is None else zfar)
    if clips == self.clips:
      return False
    self.clips = clips
    self._write_lens()
    return True

  def _write_lens(self) -> None:
    """Rewrites the lens table and the supersampled ones in place (captured graphs keep reading
    the same arrays)."""
    self.lens.assign(self._table(1))
    for n, table in self._scaled.items():
      table.assign(self._table(n))

  def _table(self, n: int) -> wp.array[kernel_structs.Lens, [int]]:
    """The lens table of the cameras rendered at n x their resolution (buffer addresses x n²)."""
    adr = [a * n * n for a in self.adr]
    return _lens_table(self.specs, self.sizes, self.clips, adr, self.wants, device=self.device, scale=n)

  def set_outputs(self, outputs: Outputs, *, depth_with_rgb: bool) -> None:
    """Sets the outputs each camera writes (`CameraSpec.outputs`) for the Renderer's requested
    `outputs`, in place (no recompile: `Lens.wants`); `depth_with_rgb`: every camera writing RGB
    writes its depth too (the contact AO's input, `shade/ao/contact.py`).

    Raises:
      ValueError: for camera outputs the Renderer does not request, or that a camera cannot
        select (`camera_checks.camera_outputs`).
    """
    extra = kernel_structs.WANT_DEPTH if depth_with_rgb else 0
    wants = [w | (extra if w & kernel_structs.WANT_RGB else 0) for w in (_wants(s, outputs) for s in self.specs)]
    if wants != self.wants:
      self.wants = wants
      self._write_lens()

  def scaled_lens(self, n: int) -> wp.array[kernel_structs.Lens, [int]]:
    """The lens table of the cameras rendered at n x their resolution (buffer addresses x n²).

    Raises:
      ValueError: for a factor below 1.
    """
    if n < 1:
      raise ValueError(f"scaled_lens: the supersampling factor must be >= 1, got {n}")
    if n not in self._scaled:
      self._scaled[n] = self._table(n)
    return self._scaled[n]

  def _locals(self, specs: list[CameraSpec]) -> None:
    """The mounted cameras' offsets from their scene cameras (identity without one)."""
    n = len(specs)
    lpos = np.zeros((n, 3), np.float32)
    lrot = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    for i, s in enumerate(specs):
      if s.local is not None:
        lpos[i] = np.asarray(s.local[0], dtype=np.float32)
        lrot[i] = np.asarray(s.local[1], dtype=np.float32).reshape(3, 3)
    self.local_pos = wp.array(lpos, dtype=wp.vec3, device=self.device)
    self.local_rot = wp.array(lrot, dtype=wp.mat33, device=self.device)
    self.has_local = wp.array(np.array([s.local is not None for s in specs]), dtype=bool, device=self.device)

  def set_rays(self, images: Sequence[bundle.RayImageInput]) -> None:
    """Replaces the bundles' rays: one (H, W, 2) ray image per camera, written in place.

    Raises:
      ValueError: when the cameras were not created with rays, or for a shape / camera count
        mismatch.
    """
    if self.rays is None:
      raise ValueError("set_camera_rays needs cameras created with CameraSpec(rays=...)")
    if len(images) != self.ncam:
      raise ValueError(f"set_camera_rays needs rays for each of the {self.ncam} cameras, got {len(images)}")
    width, height = self.sizes[0]
    self.rays = bundle.load_camera_rays(
      list(images), width=width, height=height, device=self.device, invalid=self.invalid_rays, buffers=self.rays
    )

  def pose(
    self,
    scene_pos: Table[wp.vec3, int, int] | None,
    scene_rot: Table[wp.mat33, int, int] | None,
    fovy: Table[float, int, int],
    intrinsics: Table[wp.vec4, int, int],
    distortion: Table[kernel_structs.lens_coefficients, int, int],
    worlds: wp.array[int, [int]],
  ) -> None:
    """Writes the frame's camera poses (`pos`, `rot`) from the scene cameras' poses of the call
    (None: the scene has none), the bundles' ray cones and RenderOrder.AUTO's choice (with the
    frame's per-world `fovy` / `intrinsics` / lens coefficients `distortion`, over the render
    launch's `worlds`, `Cameras.worlds`; device work only).

    Raises:
      ValueError: for mounted cameras without the scene camera poses ("camera.pos" / "camera.rot").
    """
    pos_in, rot_in = self._no_scene
    if scene_pos is not None and scene_rot is not None and scene_pos.shape[1] > 0:
      pos_in, rot_in = scene_pos, scene_rot
    elif self.has_mount:
      raise ValueError("mounted cameras need the scene camera poses (camera.pos / camera.rot)")
    self._pose_launch.launch(
      _pose_cameras,
      (self.free_pose.shape[0], self.ncam),
      inputs=[self.mount, self.local_pos, self.local_rot, self.has_local, pos_in, rot_in, self.free_pose],
      outputs=[self.pos, self.rot],
      device=self.device,
    )
    self._resolve_order(fovy, intrinsics, distortion, worlds)
    if self.rays is not None:
      # The bundle may have been written in place (`Renderer.camera_rays`): its ray cones follow.
      bundle.update_cones(self.rays, self.sizes[0][0])

  def struct(self, frame: Frame, worlds: wp.array[int, [int]]) -> kernel_structs.Cameras:
    """The kernel struct of the cameras for `frame` (its per-world fovy / intrinsics / lens
    coefficients) and the render launch's worlds `worlds` (`Cameras.worlds`; empty: every
    world)."""
    cams = kernel_structs.Cameras()
    cams.lens = self.lens
    cams.camera_fovy = frame.camera_fovy
    cams.camera_intrinsics = frame.camera_intrinsics
    cams.camera_distortion = frame.camera_distortion
    cams.camera_pos = self.pos
    cams.camera_rot = self.rot
    cams.nrender = self.ncam
    cams.nray = self.nray
    cams.nworld = worlds.shape[0] or self.pos.shape[0]
    cams.worlds = worlds
    cams.order = self.order
    cams.camera_res = self.res
    if self.rays is not None:
      cams.ray_origin, cams.ray_direction, cams.ray_adr, cams.ray_res, cams.ray_cone = self.rays
    return cams
