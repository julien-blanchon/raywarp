"""Host side of the motion output's state: `History`, the per-world state a Renderer keeps between
frames (frame index, history length, previous poses and cameras).

Per frame, all on the device (no host synchronisation, graph-capturable): `begin` (the previous
poses' buffers), the render and `end` (the poses for the next frame, frame index and history
length += 1). A skipped world (one `Renderer.render(active=...)` does not list) keeps its outputs
and its state: its frame index, history length and previous poses and cameras (the per-world rows;
a table shared by every world moves on) stay; the Renderer restarts it (`reset`) when the world is
rendered again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

import warp as wp

from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.structs import lens_coefficients
from raywarp.core.kernels.generic import device_generic
from raywarp.core.pipeline.frame import Frame
from raywarp.core.post.temporal.state import PrevCamera, Temporal

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import AnyDeviceArray

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# The frame index wraps to 0 after FRAME_WRAP frames (it stays a non-negative int32: frame % 2
# picks the previous-depth slot).
FRAME_WRAP: Final = 1 << 30


@wp.kernel
@device_generic
def _keep_rows[T](
  # In:
  cur: wp.array2d[T],
  skip: wp.array1d[bool],
  # Out:
  prev: wp.array2d[T],
):
  """Copies row w of the pose table `cur` into `prev` unless world w is skipped (`skip`: (nworld,)
  or (1,) for all); a table shared by every world ((1, n)) is copied unless a (1,) `skip` skips
  all (a per-world `skip` always moves it on, as the module doc says)."""
  w, i = wp.tid()
  if (cur.shape[0] == 1 and skip.shape[0] > 1) or not skip[w % skip.shape[0]]:
    prev[w, i] = cur[w, i]


for _dtype in (wp.vec3, wp.mat33):
  wp.overload(_keep_rows, {"cur": wp.array2d[_dtype], "skip": wp.array1d[bool], "prev": wp.array2d[_dtype]})


@wp.kernel
def _end_frame(
  # In / Out:
  frame: wp.array1d[int],
  count: wp.array2d[int],
  # In:
  skip: wp.array1d[bool],
):
  """One more frame of world w: frame index + 1 (wrapped at FRAME_WRAP), history length + 1
  (capped at FRAME_WRAP); a skipped world's (`skip`: (nworld,) or (1,) for all) stay."""
  w = wp.tid()
  if skip[w % skip.shape[0]]:
    return
  frame[w] = (frame[w] + 1) % wp.static(FRAME_WRAP)
  for c in range(count.shape[1]):
    count[w, c] = wp.min(count[w, c] + 1, wp.static(FRAME_WRAP))


@wp.kernel
def _reset(
  # In:
  mask: wp.array1d[bool],
  # Out:
  frame: wp.array1d[int],
  count: wp.array2d[int],
):
  """Restarts the history of the masked worlds (all with an empty mask)."""
  w = wp.tid()
  if mask.shape[0] == 0 or mask[w]:
    frame[w] = 0
    for c in range(count.shape[1]):
      count[w, c] = 0


@wp.kernel
def _store_cameras(
  # In:
  pos: wp.array2d[wp.vec3],
  rot: wp.array2d[wp.mat33],
  fovy: wp.array2d[float],
  intrinsics: wp.array2d[wp.vec4],
  distortion: wp.array2d[lens_coefficients],
  skip: wp.array1d[bool],
  # Out:
  prev: wp.array2d[PrevCamera],
):
  """Records camera c of world w (pose and projection) unless the world is skipped (`skip`:
  (nworld,) or (1,) for all)."""
  w, c = wp.tid()
  if skip[w % skip.shape[0]]:
    return
  record = PrevCamera()
  record.pos = pos[w, c]
  record.rot = rot[w, c]
  record.fovy = fovy[w % fovy.shape[0], c]
  record.intrinsics = intrinsics[w % intrinsics.shape[0], c]
  record.distortion = distortion[w % distortion.shape[0], c]
  prev[w, c] = record


def _dummy(dtype: type, ndim: int, device: wp.Device) -> AnyDeviceArray:
  """A (1, ..., 1) placeholder of a buffer not compiled in (any rank: the struct field's)."""
  return wp.zeros((1,) * ndim, dtype=dtype, device=device)


def _temporal[NWorld: IntVar, NPixel: IntVar](
  motion: wp.array[wp.vec2, [NWorld, NPixel]] | None,
  *,
  status: bool,
  shape: tuple[Int[NWorld], int, int],
  device: wp.Device,
) -> Temporal:
  """The `Temporal` struct for `shape` (worlds, cameras, camera pixels per world), the buffers not
  used (1, ..., 1) placeholders (`motion`: the "motion" output, None without it; `status`: its
  `flow_status` is written)."""
  nworld, ncam, total = shape

  def buffer(shape: tuple[int, ...], dtype: type, *, on: bool) -> AnyDeviceArray:
    """A zeroed `shape` buffer when compiled in (`on`), else the placeholder of its rank."""
    return wp.zeros(shape, dtype=dtype, device=device) if on else _dummy(dtype, len(shape), device)

  t = Temporal()
  t.frame = wp.zeros(nworld, dtype=int, device=device)
  t.count = wp.zeros((nworld, ncam), dtype=int, device=device)
  # The two previous-depth slots of the motion status's occlusion test (4 B per pixel, only with
  # `MotionOutput(status=True)`).
  t.gz = buffer((nworld, 2, total), wp.float16, on=status)
  t.motion = motion if motion is not None else _dummy(wp.vec2, 2, device)
  t.prev_camera = wp.zeros((nworld, ncam), dtype=PrevCamera, device=device)
  t.prev_instance_pos = _dummy(wp.vec3, 2, device)
  t.prev_instance_rot = _dummy(wp.mat33, 2, device)
  return t


@final
class History[NWorld: IntVar]:
  """The motion output's state of a Renderer (allocated when `Features.history`); the motion
  status's previous depths retain 4 B per pixel."""

  def __init__[NPixel: IntVar](
    self,
    f: Features,
    *,
    nworld: Int[NWorld],
    ncam: int,
    resolution: tuple[int, int],
    motion: wp.array[wp.vec2, [NWorld, NPixel]] | None,
    status: bool,
    device: wp.Device,
  ) -> None:
    """Allocates on `device` the state of Features `f` for these worlds and cameras of
    `resolution` (width, height) each (`motion`: the "motion" output, None without it; `status`:
    its `flow_status` output, which reads the previous depths)."""
    self.features = f
    self.motion: wp.array[wp.vec2, [int, int]] | None = motion
    self.status = status
    self.nworld, self.ncam = nworld, ncam
    self.width, self.height = resolution
    self.device = device
    self.total = ncam * self.width * self.height
    self.struct = _temporal(motion, status=status, shape=(nworld, ncam, self.total), device=device)
    # Before a first frame the previous cameras are its own (filled by `begin`).
    self._cameras_known = False
    self._prev: dict[str, AnyDeviceArray] = {}
    self._no_mask = wp.zeros(0, dtype=bool, device=device)
    self._no_skip = wp.zeros(1, dtype=bool, device=device)

  def adopt[NPixel: IntVar](
    self, f: Features, motion: wp.array[wp.vec2, [NWorld, NPixel]] | None, *, status: bool, resolution: tuple[int, int]
  ) -> bool:
    """Takes the Features `f` when this state's buffers serve them (the same "motion" output
    `motion` and its `status`, the same `resolution` (width, height)); False: a new History is
    needed."""
    if motion is not self.motion or status != self.status or (self.width, self.height) != resolution:
      return False
    self.features = f
    return True

  def _prev_of(self, name: str, a: AnyDeviceArray) -> AnyDeviceArray:
    """The previous-frame copy of frame array `a` (allocated on first use as a copy: a first frame
    is its own previous one, no motion)."""
    prev = self._prev.get(name)
    if prev is None or prev.shape != a.shape or prev.dtype != a.dtype:
      prev = self._prev[name] = wp.clone(a, device=self.device)
      if name == "instance_pos":
        self.struct.prev_instance_pos = prev
      elif name == "instance_rot":
        self.struct.prev_instance_rot = prev
    return prev

  def _poses(self, frame: Frame) -> dict[str, AnyDeviceArray]:
    """The frame's poses kept for the next frame (the instances': motion), by name."""
    return {"instance_pos": frame.instance_pos, "instance_rot": frame.instance_rot}

  def begin[NCam: IntVar](
    self, frame: Frame, camera_pos: wp.array[wp.vec3, [NWorld, NCam]], camera_rot: wp.array[wp.mat33, [NWorld, NCam]]
  ) -> None:
    """Before the render: previous-pose buffers (the cameras posed for the frame: `camera_pos` /
    `camera_rot`)."""
    for name, a in self._poses(frame).items():
      self._prev_of(name, a)
    if not self._cameras_known:
      # A first frame reprojects through its own cameras (no motion).
      self._store_cameras(frame, camera_pos, camera_rot, skip=self._no_skip)

  def _store_cameras[NCam: IntVar](
    self,
    frame: Frame,
    camera_pos: wp.array[wp.vec3, [NWorld, NCam]],
    camera_rot: wp.array[wp.mat33, [NWorld, NCam]],
    *,
    skip: wp.array[bool, [int]],
  ) -> None:
    """The frame's cameras (pose, field of view, intrinsics) as the previous ones, but the skipped
    worlds' (`skip`: (nworld,) or (1,) for all)."""
    wp.launch(
      _store_cameras,
      dim=(self.nworld, self.ncam),
      inputs=[camera_pos, camera_rot, frame.camera_fovy, frame.camera_intrinsics, frame.camera_distortion, skip],
      outputs=[self.struct.prev_camera],
      device=self.device,
    )
    self._cameras_known = True

  def end[NCam: IntVar](
    self,
    frame: Frame,
    camera_pos: wp.array[wp.vec3, [NWorld, NCam]],
    camera_rot: wp.array[wp.mat33, [NWorld, NCam]],
    *,
    world_skip: wp.array[bool, [int]],
  ) -> None:
    """After the render: poses and cameras for the next frame, frame index and history length of
    the worlds not skipped (`world_skip`: the renderer's (nworld,) or (1,) skip table)."""
    t = self.struct
    for name, a in self._poses(frame).items():
      if a.size:
        wp.launch(_keep_rows, dim=a.shape, inputs=[a, world_skip], outputs=[self._prev[name]], device=self.device)
    self._store_cameras(frame, camera_pos, camera_rot, skip=world_skip)
    wp.launch(_end_frame, dim=self.nworld, inputs=[t.frame, t.count, world_skip], device=self.device)

  def reset(self, worlds: wp.array[bool, [NWorld]] | None) -> None:
    """Restarts the history (frame index, history length) of the masked worlds (None: all).

    Raises:
      ValueError: when `worlds` is not a (nworld,) bool array on the renderer's device.
    """
    if worlds is not None and (worlds.dtype != wp.bool or worlds.shape != (self.nworld,) or worlds.device != self.device):
      raise ValueError(
        f"reset_history needs a ({self.nworld},) bool array on {self.device}, "
        f"got {worlds.dtype} {worlds.shape} on {worlds.device}"
      )
    mask = self._no_mask if worlds is None else worlds
    wp.launch(_reset, dim=self.nworld, inputs=[mask, self.struct.frame, self.struct.count], device=self.device)
