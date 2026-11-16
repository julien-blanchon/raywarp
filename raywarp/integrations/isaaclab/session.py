"""Isaac Lab cameras rendered in groups: one raywarp `Renderer` per group of cameras that render
alike (same environments, size, outputs, clips and settings), one camera of it per Isaac camera.

Render world w is environment w; each camera is a free camera posed from Isaac's camera state
(`CameraSession.update`). A group renders all its cameras in one frame (`CameraGroup.render`): one
pose / BVH / light / GI update for every camera instead of one per camera. A group of one camera
writes straight into Isaac's (N, H, W, C) output arrays where the layouts match (color, HDR,
normals, the first depth, instance ids: `Renderer.bind_outputs`, zero copy); with several cameras
the renderer writes its world-major (N, ncam, H, W) buffers and `finish` copies each requested
camera's column into Isaac's arrays. `finish` also converts the rest in place after each frame (a
second depth, packed albedo, semantic classes, Isaac's background id), with copies and launches
prepared once per binding. No host copy or synchronisation per frame: the host only enqueues work
(on CUDA a captured graph replay, captured again when Isaac swaps an array, plus the `finish`
launches).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, NamedTuple, final

import numpy as np
import warp as wp

from raywarp._base import arguments
from raywarp.core.pipeline.gpu_scene.binding import StaleBinding
from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec
from raywarp.core.pipeline.platform.device import device_caps
from raywarp.core.pipeline.renderer import Renderer
from raywarp.core.pipeline.worlds.sharing import PerWorld
from raywarp.integrations.isaaclab import contract, kernels

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.graph import RenderGraph
  from raywarp.integrations.isaaclab.segments import SegmentTables
  from raywarp.integrations.isaaclab.types import Buffer, SegmentInfo
  from raywarp.integrations.newton.types import NewtonState
  from raywarp.settings import RenderSettings

# Flat (nworld, H * W) views of an output, by Warp element type.
type Flat = wp.array[object, [int, int]]

# Isaac output -> (renderer output it is written from, element type of a pixel).
_DIRECT: Final = {
  "rgba": ("rgb", wp.uint32),
  "rgb_hdr": ("hdr", wp.vec3),
  "normals": ("normal", wp.vec3),
  contract.INSTANCE: ("seg", wp.int32),
}


class _Pass(NamedTuple):
  """One `finish` launch over (world, pixel): a conversion kernel and its arguments."""

  kernel: Callable[..., None]
  inputs: Sequence[object]
  outputs: Sequence[Flat]


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CameraConfig:
  """What a camera renders (from Isaac's `CameraRenderSpec` and the renderer config).

  Attributes:
    nworld: environments (render worlds).
    width, height: image size [px].
    names: Isaac output names requested (`contract.OUTPUTS` keys).
    clipping: Isaac's `depth_clipping_behavior` ("none", "zero", "max").
    znear, zfar: clip planes (`spawn.clipping_range`; None: the scene's defaults).
    graph: replay each frame as a CUDA graph (CUDA devices only).
    period: Isaac's `update_period` [s] (cameras due together render together).
  """

  nworld: int
  width: int
  height: int
  names: tuple[str, ...]
  clipping: str = "none"
  znear: float | None = None
  zfar: float | None = None
  graph: bool = True
  period: float = 0.0


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Appearance:
  """What every renderer of a scene shares: segment ids, visibility and live material tables.

  Attributes:
    segments: the scene's segment ids and labels (None without a segmentation output).
    visibility: (nworld, ninstance) instance visibility (None: everything visible).
    material_colors: (nworld, nmaterial) live base colors (None: the scene's).
    material_scalars: live scalar material tables by per-world field.
  """

  segments: SegmentTables | None = None
  visibility: np.ndarray[[int, int], np.dtype[np.bool_]] | None = None
  material_colors: wp.array[wp.vec4, [int, int]] | None = None
  material_scalars: Mapping[str, wp.array[float, [int, int]]] = dataclasses.field(default_factory=dict)


def check_device(device: wp.Device, **arrays: AnyDeviceArray) -> None:
  """Raises ValueError for an array on another device than `device` (a Warp launch would read
  it as `device` memory: a CPU launch on CUDA memory crashes the process)."""
  for name, array in arrays.items():
    if wp.get_device(array.device) != device:
      raise ValueError(f"{name} is on {array.device}, the renderer renders on {device}")


def flat(array: AnyDeviceArray, dtype: type, nworld: int) -> Flat:
  """`array`'s memory (an Isaac (N, H, W, C) output) viewed as (nworld, H * W) `dtype` elements.

  Raises:
    ValueError: for a non-contiguous array or one of another byte size.
  """
  if not array.is_contiguous:
    raise ValueError("Isaac Lab output arrays must be contiguous")
  size = wp.types.type_size_in_bytes(array.dtype) * array.size
  element = wp.types.type_size_in_bytes(dtype)
  if size % (element * nworld):
    raise ValueError(f"an output of {size} bytes does not hold {nworld} rows of {dtype} elements")
  return wp.array(ptr=array.ptr, dtype=dtype, shape=(nworld, size // (element * nworld)), device=array.device)


def column(array: AnyDeviceArray, dtype: type, nworld: int, ncam: int, cam: int) -> Flat:
  """Camera `cam`'s (nworld, H * W) `dtype` pixels of a renderer's world-major (nworld, ncam, H,
  W) output (a strided view, no copy)."""
  rows = flat(array, dtype, nworld * ncam)
  if rows.ptr is None:
    raise ValueError("an empty renderer output has no camera columns")
  pixels, element = rows.shape[1], wp.types.type_size_in_bytes(dtype)
  return wp.array(
    ptr=rows.ptr + cam * pixels * element,
    dtype=dtype,
    shape=(nworld, pixels),
    strides=(ncam * pixels * element, element),
    device=array.device,
  )


@final
class CameraSession:
  """One Isaac camera: its column of its group's renderer, its Isaac output arrays and the
  conversions between them (Isaac's render data of the camera).

  Attributes:
    config: what it renders.
    group: the group rendering it.
  """

  def __init__(self, config: CameraConfig, group: CameraGroup) -> None:
    """Joins `group` (its renderer is rebuilt with one more camera on its next use)."""
    self.config = config
    self.group = group
    self.arrays: dict[str, Buffer] = {}
    self._copies: list[tuple[Flat, Flat]] = []
    self._passes: list[_Pass] = []
    # (planar depth written, radial depth) when both are bound: `finish`'s radial pass.
    self._radial: tuple[Flat, Flat] | None = None
    self.intrinsics_set = False
    # Isaac's camera state arrays of the last `update` (written again into a rebuilt renderer).
    self._state: tuple[wp.array[wp.vec3, [int]], wp.array[wp.quatf, [int]], wp.array[wp.mat33, [int]]] | None = None
    group.join(self)

  def move(self, group: CameraGroup) -> None:
    """Leaves its group for `group` (both renderers are rebuilt on their next use)."""
    self.group.leave(self)
    self.group = group
    group.join(self)

  def restore(self) -> None:
    """Writes the last camera state into a rebuilt renderer (its pose and intrinsics)."""
    if self._state is not None:
      self.update(*self._state)

  @property
  def renderer(self) -> Renderer[NewtonState]:
    """The group's renderer (built on first use)."""
    return self.group.renderer

  @property
  def camera(self) -> int:
    """This camera's index in its group's renderer."""
    return self.group.sessions.index(self)

  @property
  def segments(self) -> SegmentTables | None:
    """The scene's segment ids and labels (None without a segmentation output)."""
    return self.group.appearance.segments

  def bind(self, arrays: Mapping[str, Buffer]) -> None:
    """Renders into Isaac's output arrays (by output name; the (N, H, W, C) buffers of
    `CameraData.output`): the matching ones bound to a one-camera renderer, the others written by
    `finish` (its copies and launches prepared when the group binds).

    Raises:
      ValueError: for an output name not requested or not written, or an array that is not the
        output's (N, H, W, C) array of its dtype (`contract.OUTPUTS`): the passes of `finish`
        write N x H x W pixels into it.
    """
    contract.check_names(arrays)
    c = self.config
    for name, array in arrays.items():
      want = contract.OUTPUTS[name]
      shape = (c.nworld, c.height, c.width, want.channels)
      if tuple(array.shape) != shape or array.dtype != want.dtype:
        raise ValueError(
          f"Isaac Lab output {name!r} must be a {shape} {wp.types.type_repr(want.dtype)} array, got "
          f"{tuple(array.shape)} {wp.types.type_repr(array.dtype)}"
        )
    self.arrays = dict(arrays)
    try:
      self.group.rebind()
    except BaseException:
      # The renderer may still render into the arrays this call replaced (no longer referenced
      # here): released, it is rebuilt over the new ones at the next use (which raises again).
      self.group.release()
      raise

  def depths(self) -> list[str]:
    """The depth outputs bound, the first one written by the renderer."""
    return [d for d in (*contract.PLANAR, contract.RADIAL) if d in self.arrays]

  def direct(self) -> dict[str, tuple[str, Flat]]:
    """Isaac output -> (renderer output key, the Isaac array's flat view) of the outputs the
    renderer writes itself."""
    n = self.config.nworld
    out = {name: (key, flat(self.arrays[name], dtype, n)) for name, (key, dtype) in _DIRECT.items() if name in self.arrays}
    depths = self.depths()
    if depths:
      out[depths[0]] = ("depth", flat(self.arrays[depths[0]], wp.float32, n))
    return out

  def prepare(self) -> None:
    """The copies and launches of `finish` for the bound arrays, from the renderer's buffers."""
    r, n, cam = self.renderer, self.config.nworld, self.camera
    ncam = len(self.group.sessions)
    out = r.outputs

    def written(key: str, dtype: type) -> Flat:
      """This camera's pixels of the renderer's output `key`."""
      buffer = getattr(out, key)
      if buffer is None:
        raise RuntimeError(f"CameraSession: the renderer wrote no {key} (requested by contract.outputs)")
      return column(buffer, dtype, n, ncam, cam)

    self._copies, self._passes, self._radial = [], [], None
    for key, isaac in self.direct().values():
      source = written(key, isaac.dtype)
      if source.ptr != isaac.ptr:
        self._copies.append((isaac, source))
    depths = self.depths()
    for name in depths[1:]:
      planar, dest = flat(self.arrays[depths[0]], wp.float32, n), flat(self.arrays[name], wp.float32, n)
      if name == contract.RADIAL:
        self._radial = (planar, dest)
      else:
        self._copies.append((dest, planar))
    if "albedo" in self.arrays:
      self._passes.append(
        _Pass(kernels.packed_albedo, [written("albedo", wp.vec3)], [flat(self.arrays["albedo"], wp.uint32, n)])
      )
    segments = self.segments
    if segments is not None and out.seg is not None:
      ids = written("seg", wp.int32)
      if contract.SEMANTIC in self.arrays:
        semantic = flat(self.arrays[contract.SEMANTIC], wp.int32, n)
        self._passes.append(_Pass(kernels.semantic_ids, [ids, segments.classes], [semantic]))
      if contract.INSTANCE in self.arrays:  # after the semantic lookup: it reads the misses (-1)
        self._passes.append(_Pass(kernels.background_ids, [], [flat(self.arrays[contract.INSTANCE], wp.int32, n)]))

  def update(
    self,
    positions: wp.array[wp.vec3, [int]],
    orientations: wp.array[wp.quatf, [int]],
    intrinsics: wp.array[wp.mat33, [int]],
  ) -> None:
    """Poses this camera in every environment: (N,) vec3 positions, (N,) quatf orientations
    (x, y, z, w; Isaac's world convention: +X forward, +Z up) and, on the first call (and after
    its group was rebuilt), the (N,) mat33 pixel intrinsics (later changes: `set_intrinsics`)."""
    r = self.renderer
    check_device(r.device, positions=positions, orientations=orientations, intrinsics=intrinsics)
    self._camera_array("positions", positions, wp.vec3)
    self._camera_array("orientations", orientations, wp.quatf)
    self._camera_array("intrinsics", intrinsics, wp.mat33)
    self._state = (positions, orientations, intrinsics)
    wp.launch(
      kernels.camera_poses,
      self.config.nworld,
      inputs=[positions, orientations, self.camera],
      outputs=[r.camera_pose],
      device=r.device,
    )
    if not self.intrinsics_set:
      self.set_intrinsics(intrinsics)

  def set_intrinsics(self, intrinsics: wp.array[wp.mat33, [int]]) -> None:
    """Writes the (N,) mat33 pixel intrinsics into this camera's column of the per-world table (in
    place, any env apart); the environments whose intrinsics changed restart their history."""
    c, r = self.config, self.renderer
    check_device(r.device, intrinsics=intrinsics)
    self._camera_array("intrinsics", intrinsics, wp.mat33)
    changed = self.group.history_mask()
    wp.launch(
      kernels.camera_intrinsics,
      c.nworld,
      inputs=[intrinsics, float(c.width), float(c.height), self.camera],
      outputs=[self.group.intrinsics, changed],
      device=r.device,
    )
    if self.intrinsics_set and changed.size:
      r.reset_history(changed)
    self.intrinsics_set = True

  def _camera_array(self, name: str, array: AnyDeviceArray, dtype: type) -> None:
    """Reject incompatible state arrays before a camera kernel can index them."""
    if array.shape != (self.config.nworld,) or array.dtype != dtype:
      raise ValueError(f"{name} must have shape ({self.config.nworld},) and dtype {dtype}, got {array.shape} {array.dtype}")

  def reset(self, env_ids: Sequence[int] | wp.array[wp.int32, [int]] | None = None) -> None:
    """Reset the selected environments' histories (of every camera of the group: an environment
    resets as a whole). Device ids are Isaac-validated, out-of-range ids skipped, never
    downloaded; host sequences are bounds-checked."""
    r = self.renderer
    if env_ids is None:
      r.reset_history()
      return
    selected = self.group.history_mask()
    if isinstance(env_ids, wp.array):
      check_device(r.device, env_ids=env_ids)
      if env_ids.ndim != 1 or env_ids.dtype != wp.int32:
        raise ValueError("reset env_ids must be a one-dimensional int32 array")
      if selected.size:
        selected.zero_()
        if env_ids.size:
          wp.launch(kernels.reset_mask, dim=env_ids.size, inputs=[env_ids], outputs=[selected], device=r.device)
        r.reset_history(selected)
      return
    ids = [arguments.check_int("env_id", i, low=0, high=self.config.nworld - 1) for i in env_ids]
    if ids and selected.size:
      mask = np.zeros(self.config.nworld, dtype=bool)
      mask[ids] = True
      selected.assign(mask)
      r.reset_history(selected)

  def finish(self) -> None:
    """Writes the outputs the renderer does not write itself into Isaac's arrays (in place): the
    copies and launches `prepare` set up."""
    dim, device = (self.config.nworld, self.config.width * self.config.height), self.renderer.device
    for dest, src in self._copies:
      wp.copy(dest, src)
    for kernel, inputs, outputs in self._passes:
      wp.launch(kernel, dim, inputs=inputs, outputs=outputs, device=device)
    if self._radial is not None:
      c, (planar, radial) = self.config, self._radial
      # Misses keep the exact value the renderer wrote: inf, 0 or its far clip now.
      miss = contract.miss_value(contract.MISS_DEPTH[c.clipping], self.renderer.clips[self.camera][1])
      inputs = [planar, self.group.intrinsics, c.width, c.height, miss, self.camera]
      wp.launch(kernels.radial_depth, dim, inputs=inputs, outputs=[radial], device=device)

  def info(self) -> dict[str, SegmentInfo]:
    """Isaac's `CameraData.info` entries of the requested segmentation outputs."""
    segments = self.segments
    if segments is None:
      return {}
    return {name: info for name, info in segments.info.items() if name in self.arrays}

  def close(self) -> None:
    """Leaves the group (its renderer is rebuilt without this camera, or released when empty)."""
    self.group.leave(self)


@final
class CameraGroup:
  """The cameras one `Renderer` renders together, one camera each: alike cameras (equal config and
  settings) that Isaac submits in one `render_batch` (`Cameras.render_batch` regroups them).

  Attributes:
    config: what each of its cameras renders.
    settings: the render settings (outputs from `config`).
    appearance: the scene tables every renderer shares.
    sessions: its cameras, in renderer camera order.
  """

  def __init__(
    self, scene: GpuScene[NewtonState], config: CameraConfig, settings: RenderSettings, appearance: Appearance
  ) -> None:
    """No camera and no renderer yet."""
    self.scene = scene
    self.config = config
    self.settings = settings.replace(outputs=contract.outputs(config.names, config.clipping))
    self.appearance = appearance
    self.sessions: list[CameraSession] = []
    self._renderer: Renderer[NewtonState] | None = None
    self._graph: RenderGraph[NewtonState] | None = None
    self._reset_mask: wp.array[bool, [int]] | None = None

  def join(self, session: CameraSession) -> None:
    """Adds a camera (the renderer is rebuilt on next use)."""
    self.sessions.append(session)
    self._drop()

  def leave(self, session: CameraSession) -> None:
    """Removes a camera (the renderer is rebuilt on next use, released when none is left)."""
    self.sessions.remove(session)
    self._drop()

  def release(self) -> None:
    """Releases the renderer (rebuilt on the next use)."""
    self._drop()

  def _drop(self) -> None:
    """Releases the renderer: every camera re-binds and re-writes its intrinsics on its next use
    (detached first: a close that fails leaves no closed renderer in use)."""
    renderer = self._renderer
    self._renderer, self._graph, self._reset_mask = None, None, None
    for session in self.sessions:
      session.intrinsics_set = False
    if renderer is not None:
      renderer.close()

  @property
  def renderer(self) -> Renderer[NewtonState]:
    """The renderer of every camera (built on first use after a camera joined or left)."""
    if self._renderer is None:
      self._renderer = self._build()
      try:
        self.rebind()
        for session in self.sessions:
          session.restore()
      except BaseException:
        self._drop()  # half-bound: rebuilt (and checked again) on the next use
        raise
    return self._renderer

  def _build(self) -> Renderer[NewtonState]:
    """One camera per session (placeholder 90-degree intrinsics: each camera's first `update`
    writes Isaac's matrices), the shared appearance tables bound."""
    c, a = self.config, self.appearance
    w, h = c.width, c.height
    camera = CameraSpec(intrinsics=(0.5 * w, 0.5 * w, 0.0, 0.0), sensor_size=(float(w), float(h)), znear=c.znear, zfar=c.zfar)
    material_fields = ([] if a.material_colors is None else ["base_color"]) + list(a.material_scalars)
    decl = PerWorld(camera=["intrinsics"], material=material_fields, visible=["*"] if a.visibility is not None else ())
    ncam = max(len(self.sessions), 1)
    r = Renderer(
      self.scene, nworld=c.nworld, cameras=[camera] * ncam, resolution=(w, h), settings=self.settings, per_world=decl
    )
    try:
      if a.material_colors is not None:
        r.bind_per_world("material.base_color", a.material_colors)
      for field, table in a.material_scalars.items():
        r.bind_per_world(field, table)
      if a.visibility is not None:
        r.per_world["instance.visible"].assign(a.visibility.astype(np.int32))
      if a.segments is not None:
        r.set_segment_table(a.segments.ids)
    except BaseException:
      r.close()  # nothing else owns it yet
      raise
    return r

  @property
  def intrinsics(self) -> wp.array[wp.vec4, [int, int]]:
    """The (nworld, ncam) per-world camera intrinsics table."""
    table = self.renderer.per_world["camera.intrinsics"]
    if not isinstance(table, wp.array):
      raise TypeError("the camera intrinsics table must be a Warp array")
    return table

  def rebind(self) -> None:
    """Binds the outputs of a one-camera group to its Isaac arrays (zero copy), then prepares each
    camera's `finish` (a no-op before the renderer exists: it binds when built)."""
    r = self._renderer
    if r is None:
      return
    bound: dict[str, Flat] = {}
    if len(self.sessions) == 1:
      bound = dict(self.sessions[0].direct().values())
    r.bind_outputs(**bound)
    self._graph = None
    for session in self.sessions:
      session.prepare()

  def history_mask(self) -> wp.array[bool, [int]]:
    """Reusable (nworld,) reset mask; empty for a renderer without history."""
    if self._reset_mask is None:
      size = self.config.nworld if self.renderer.features.history else 0
      self._reset_mask = wp.empty(size, dtype=bool, device=self.renderer.device)
    return self._reset_mask

  def reset_worlds(self, mask: wp.array[bool, [int]]) -> None:
    """Restarts the history of the (nworld,) `mask`'s worlds (nothing before the renderer is
    built: a new renderer starts without history)."""
    if self._renderer is not None:
      self._renderer.reset_history(mask)

  def render(self, state: NewtonState, sessions: Sequence[CameraSession]) -> None:
    """Renders every camera from `state`'s body poses (one frame), then writes the outputs of
    `sessions` (the requested cameras; the others keep their last observation)."""
    r = self.renderer
    caps = device_caps(r.device)
    if caps.capturing() or not (self.config.graph and caps.graphs):
      r.render(state)
    elif self._graph is None:
      self._graph = r.capture(None, state)
    else:
      try:
        self._graph.launch(state)
      except StaleBinding:  # Isaac swapped an array (e.g. the body poses' buffer): capture again
        self._graph = r.capture(None, state)
    for session in sessions:
      session.finish()
