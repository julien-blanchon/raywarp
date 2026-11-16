"""`RendererControls`: the Renderer methods that change things at runtime without recompiling its
kernels.

Camera poses, rays, clip distances, caller output buffers, the segmentation /
semantic tables, the world skip mask and rebuild requests of baked structures (light grid, GI / sky
probes). Each change bumps the renderer's struct generation (and, unless it is written in place, the
graph version: captured graphs become stale). Also the read-only views of its runtime state (pose
buffers, history length, GI volume, outputs, cost counters), the point clouds of its depth
output (`point_cloud`), the ground truth computed from its outputs (`camera_normals`, `nocs`) and
its cameras' export (`camera_info`). `Renderer` inherits them; the attributes below are set by its
constructor.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import warp as wp

from raywarp._base import arguments
from raywarp._base.types import CameraSlot
from raywarp.core.build.variants import explain as explain_lib
from raywarp.core.build.variants.explain import Explanation
from raywarp.core.kernels.abi import counters as counters_lib
from raywarp.core.outputs import camera_export, ground_truth, pointcloud
from raywarp.core.outputs import semantic_classes as semantics_lib
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.worlds import sharing
from raywarp.core.pipeline.worlds.tables import PerWorldView

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray, Mask
  from raywarp.core.build.variants.tiers import Choice
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.outputs.camera_export import CameraInfo
  from raywarp.core.outputs.pointcloud import PointCloud
  from raywarp.core.outputs.render_outputs import Buffers, RenderOutputs
  from raywarp.core.pipeline.baked import Baked
  from raywarp.core.pipeline.cameras import CameraRig
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.binding import PoseBuffers
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.worlds.active import ActiveWorlds
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.pipeline.worlds.tables import PerWorldTables
  from raywarp.core.pipeline.worlds.watch import Watch
  from raywarp.core.post.temporal.history import History
  from raywarp.core.shade import gi as gi_lib
  from raywarp.core.trace.context import RendererTables
  from raywarp.core.trace.rays.bundle import RayImageInput
  from raywarp.settings import RenderSettings


class RendererControls[S]:
  """The runtime controls of a `Renderer` (see the module docstring)."""

  scene: GpuScene[S]
  device: wp.Device
  nworld: int
  ncam: int
  layout: Layout
  per_world_fields: frozenset[str]
  features: Features
  settings: RenderSettings
  _choice: Choice | None
  _rig: CameraRig
  _buffers: Buffers
  _tables: RendererTables
  _active: ActiveWorlds
  _baked: Baked
  _per_world: PerWorldTables
  _history: History | None
  _watch: Watch[S]
  # Counters: captured graphs are stale when `_version` or `Baked.generation` changed; kernel
  # structs are rebuilt when `_generation` changed (with `Baked.generation`).
  _version: int = 0
  _generation: int = 0

  _closed: bool
  # Frames rendered so far (`camera_info`'s frame index): eager renders and the replays of graphs
  # captured by `capture` (a frame recorded in another renderer's captured step is not counted).
  _renders: int = 0
  # NOCS: the segment table read and its inverse (`_instance_of`).
  _nocs_ids: tuple[AnyDeviceArray, wp.array[int, [int]]] | None = None

  def _frame(self, sim: S | None = None) -> Frame:
    """The kernel-field arrays of a call for `sim` (`Renderer._frame`)."""
    raise NotImplementedError

  def reset_history(self, worlds: wp.array[bool, [int]] | None = None) -> None:
    """Restarts the temporal history of the masked worlds (`Renderer.reset_history`)."""
    raise NotImplementedError

  def _check_open(self, call: str) -> None:
    """Raises RuntimeError after `close` (the backend's structures are released)."""
    if self._closed:
      raise RuntimeError(f"Renderer.{call} after close(): create a new Renderer")

  @property
  def per_world(self) -> PerWorldView:
    """The per-world values the kernels read, by field, owned by the renderer, written in place
    (`r.per_world["material.base_color"].assign(colors)`; `r.per_world.columns("instance.visible",
    "distractor_*")` for the columns of some instances; `print(r.per_world)` lists them).

    Fields: the non-pose slots of `binding.SLOTS` ("instance.color", "material.base_color",
    "light.strength", ...) but the derived ones, the OpenPBR "material.specular_weight",
    "specular_ior", the coat inputs and "emission_luminance" they derive from
    (`derived_fields.py`), "exposure", "ambient",
    "env.id" / "env.intensity" / "env.yaw",
    "camera.fovy" / "camera.intrinsics", "fog.color" / "fog.range", the look fields
    ("grading.white_balance", ...), and the instance geometry fields "instance.pos" /
    "instance.rot" (static instances), "instance.size", "instance.mesh", "instance.visible" (int,
    0 hides). Each is declared at construction (`Renderer(per_world=PerWorld(...))`), which
    allocates its (nworld, n) table (starting as the current value, winning over the scene /
    binding value) and decides what is compiled in (`PerWorld`), so writing it never
    reconfigures. A field not declared raises KeyError (naming what to declare).
    """
    return PerWorldView(self._per_world)

  def bind_per_world(self, field: str, array: AnyDeviceArray) -> None:
    """Binds caller device storage as the table of a declared appearance / environment field.

    Shape, element type and device must match `per_world[field]`; storage must be contiguous.
    No values are read back or copied. Binding replaces the pointer and invalidates captured graphs.
    Subsequent in-place writes keep the pointer and graph valid; after binding or writing,
    callers reset the histories of worlds whose material/lighting values changed.

    Raises:
      RuntimeError: after `close` (before any change).
    """
    self._check_open("bind_per_world")
    if self._per_world.bind(field, array):
      self._watch.validator.set_ranges(self._per_world.ranges())
      self._changed()

  @property
  def camera_pose(self) -> wp.array[wp.transformf, [int, int]]:
    """(nworld, ncam) transforms of the free cameras (mount None), written in place."""
    return self._rig.free_pose

  def set_camera_rays(self, rays: Sequence[RayImageInput]) -> None:
    """Replaces the per-pixel rays of the (ray-bundle) cameras: one (H, W, 2) ray image per camera.

    Raises:
      ValueError: when the cameras were not created with rays, or for a shape / camera count
        mismatch.
      RuntimeError: after `close` (before any change).
    """
    self._check_open("set_camera_rays")
    self._rig.set_rays(rays)
    self._changed(graphs=False)

  @property
  def camera_rays(self) -> tuple[wp.array[wp.vec3, [int]], wp.array[wp.vec3, [int]]] | None:
    """(origin, direction) ray buffers (ncam * H * W,) of ray-bundle cameras, writable in place."""
    rays = self._rig.rays
    return None if rays is None else (rays[0], rays[1])

  @property
  def clips(self) -> tuple[tuple[float, float], ...]:
    """(znear, zfar) of each camera as rendered: its spec's, else the scene's default, then
    `set_clip`'s (zfar 0: unbounded)."""
    return tuple(self._rig.clips)

  def set_clip(
    self,
    *,
    znear: float | None = None,
    zfar: float | None = None,
    camera: CameraSlot | None = None,
  ) -> None:
    """Changes the clip distances of camera `camera` (None: every camera); None keeps a value.

    Runtime: written into the cameras' table in place, no recompilation; zfar 0 = unbounded.

    Raises:
      ValueError: for a camera index out of range (or a bool), a distance that is not a finite
        number >= 0, or a camera left with znear >= zfar (zfar != 0).
      RuntimeError: after `close` (before any change).
    """
    self._check_open("set_clip")
    if camera is not None:
      camera = CameraSlot(arguments.check_int("camera", camera, low=0, high=self.ncam - 1))
    znear, zfar = arguments.check_clip(znear, zfar, "set_clip")
    for c in range(self.ncam) if camera is None else (camera,):
      near, far = self._rig.clips[c]
      arguments.check_clip(near if znear is None else znear, far if zfar is None else zfar, f"set_clip: camera {c}")
    if self._rig.set_clip(znear, zfar, camera):
      self._changed(graphs=False)

  def bind_outputs(self, **arrays: AnyDeviceArray) -> None:
    """Renders into caller arrays instead of the renderer's own output buffers (zero copy).

    Keywords: the requested outputs' names: "rgb" in its `ColorOutput` layout (uint32 / vec4ub
    RGBA, vec3ub RGB, or uint8 / float16 planes: 3 x the elements, a policy's observation
    tensor), "depth" (float32), "seg" (int32 / uint32), "hdr" (vec3), "normal" (vec3), ...; each
    array holds a world-major (nworld, ncam, H, W) output in any shape of that size (e.g. (nworld
    * ncam, H, W)). A float output takes its own dtype only, an integer one any integer element
    of its size. Buffers not named go back to own allocations. A no-op
    when the arrays are the bound ones; else `outputs` is replaced and captured graphs become
    stale. Allocates nothing for the bound buffers (and nothing at all on a pure re-binding).

    Raises:
      TypeError: for something else than a Warp array.
      ValueError: for a buffer that is off, or an array of the wrong size, element type, layout
        or device.
      RuntimeError: after `close` (before any change).
    """
    self._check_open("bind_outputs")
    if self._buffers.bind(arrays):
      self._changed()

  def set_semantics(
    self, rules: Mapping[str, sharing.Patterns], worlds: Sequence[int] | Mask[int] | None = None
  ) -> tuple[str, ...]:
    """Semantic classes of the semantic output.

    `rules`: label -> instances (`PerWorld` patterns: globs on `GpuScene.instance_names` or
    indices; later labels win; by MuJoCo body: `semantics.body_instances`). Class ids: 0
    background, 1 unlabelled (every instance no rule matches, and deformables), then the labels in
    order. `worlds`: None assigns every world (and restarts the numbering); world indices or a
    (nworld,) bool mask assign those worlds only: the other worlds keep their classes, every
    existing id keeps its label and new labels get the next ids (one label list for all worlds).
    Written in place: no recompilation, captured graphs keep reading the table, except on the
    first per-world call, which replaces the shared (1, ninstance) table by a (nworld, ninstance)
    one (captured graphs become stale: make that call before capturing).

    Returns:
      The labels by class id (`semantic_labels`).

    Raises:
      TypeError: for a label that is not a str.
      ValueError: for a pattern that matches no instance, an empty or reserved label, or a world
        index out of range / a mask of another length.
      RuntimeError: after `close` (before any change).
    """
    self._check_open("set_semantics")
    semantics = self._tables.semantics
    table = semantics.ids
    mask = None if worlds is None else semantics_lib.world_mask(worlds, self.nworld)
    labels = semantics.assign(rules, mask)  # written in place, but on a first per-world call
    if semantics.ids is not table:
      self._changed()
    return labels

  def set_segment_table(self, table: wp.array[int, [int, int]] | None) -> None:
    """The ids "seg" writes per (world, instance): a (1 or nworld, ninstance) int32 table read
    every render (write it in place, e.g. in a graph: Newton's shape ids per view); None: the
    scene's (`GpuScene.segment_id`). No recompilation.

    Raises:
      ValueError: for a table of another dtype / shape or device, or a scene without a segment
        table (MuJoCo: "seg" is the geom id).
      RuntimeError: after `close` (before any change).
    """
    self._check_open("set_segment_table")
    if self.scene.segment_id is None:
      raise ValueError("set_segment_table needs a scene with a segment table (GpuScene.segment_id)")
    if table is None:
      table = self.scene.segment_id
    elif table.dtype != wp.int32 or tuple(table.shape[1:]) != (self.scene.ninstance,) or table.shape[0] not in (1, self.nworld):
      raise ValueError(
        f"a segment table is (1 or {self.nworld}, {self.scene.ninstance}) int32, got {table.dtype} {tuple(table.shape)}"
      )
    elif table.device != self.device:
      raise ValueError(f"the segment table is on {table.device}, the renderer renders on {self.device}")
    if table is not self._tables.segment_id:
      self._tables.segment_id = table
      self._changed()

  def point_cloud(  # noqa: PLR0913  # keyword-only options of one request
    self,
    npoints: int | None = None,
    *,
    cameras: Sequence[int] | None = None,
    labels: Literal["seg", "semantic"] | None = None,
    keep: Sequence[int] | None = None,
    colors: bool = False,
    seed: int = 0,
    out: PointCloud | None = None,
  ) -> PointCloud:
    """World-space point clouds of the last render's depth output, one per world, the chosen
    cameras fused (`core/outputs/pointcloud.py`): every pixel that hit a surface, unprojected
    through its camera ray (pose, projection, intrinsics, distortion; planar or radial depth, any
    miss value).

    Args:
      npoints: N points per world, resampled deterministically (stratified over the kept pixels,
        a random offset per world from `seed`; repeated when a world keeps fewer); None: every
        kept pixel, compacted in camera / row order, zero past `count`.
      cameras: the camera indices fused (None: all).
      labels: also return each point's id from the "seg" or "semantic" output (`PointCloud.ids`).
      keep: keep only the points whose `labels` id is one of these.
      colors: also return each point's display color (sRGB bytes, `PointCloud.colors`).
      seed: the resampling's random seed.
      out: a cloud of an earlier call with the same arguments: written in place (no allocation,
        graph-capturable). The worlds the last render skipped (`render(active=...)`) are left as
        they are: with `out`, the cloud of their kept depth (empty in a new cloud).

    Returns:
      `PointCloud`: points (nworld, N) vec3 [m], count (nworld,), colors / ids (nworld, N).

    Raises:
      ValueError: without the depth output (or the rgb / `labels` output asked for), for
        ray-bundle cameras, an unknown camera, npoints < 1, `keep` without `labels`, or an `out`
        of another request.
      RuntimeError: the renderer is closed.
    """
    self._check_open("point_cloud")
    req = pointcloud.Request(npoints=npoints, cameras=cameras, labels=labels, keep=keep, colors=colors, seed=seed)
    depth = self.settings.written_outputs().depth
    skip = self._tables.world_skip
    return pointcloud.compute(req, rig=self._rig, buffers=self._buffers, depth=depth, view=self._view(), skip=skip, out=out)

  def _view(self) -> tuple[AnyDeviceArray, AnyDeviceArray, AnyDeviceArray]:
    """The frame's per-world (fovy, intrinsics, lens coefficients) tables, as the render reads
    them (`PerWorldTables.camera_field`)."""
    field = self._per_world.camera_field
    return field("camera.fovy"), field("camera.intrinsics"), field("camera.distortion")

  def camera_info(self, *, out: CameraInfo | None = None) -> CameraInfo:
    """The cameras of the last render, as a dataset records them (`core/outputs/camera_export.py`,
    OpenCV conventions): per world and camera the intrinsics K in pixels, the lens model and its
    coefficients, the camera-to-world pose; per camera the resolution; the frame index. Device
    arrays (one launch, no host read: read them when the frame's outputs are read). The worlds the
    last render skipped (`render(active=...)`) are left as they are: with `out`, the cameras their
    kept images were rendered with (zero in a new info).

    Args:
      out: an info of an earlier call: written in place (no allocation, graph-capturable).

    Raises:
      ValueError: for ray-bundle cameras, or an `out` of another renderer.
      RuntimeError: the renderer is closed.
    """
    self._check_open("camera_info")
    distorted = "camera.distortion" in self.per_world_fields
    skip = self._tables.world_skip
    return camera_export.compute(self._rig, self._view(), distorted=distorted, frame=self._renders, skip=skip, out=out)

  def camera_normals(self, *, out: AnyDeviceArray | None = None) -> wp.array[wp.vec3, [int, int]]:
    """The last render's "normal" output in each camera's OpenCV frame (x right, y down, z forward;
    `core/outputs/ground_truth.py`): (nworld, pixels) vec3 in the outputs' flat order (camera c's
    pixels row-major from its buffer address; with one resolution for every camera, reshape to
    (nworld, ncam, H, W)), zero on a miss. One launch, no host read. The rows of the worlds the last
    render skipped (`render(active=...)`) are left as they are (with `out`, the values of their kept
    images; zero in a new array).

    Args:
      out: the array of an earlier call: written in place (no allocation, graph-capturable).

    Raises:
      ValueError: without the normal output, or for an `out` of another shape / dtype / device.
      RuntimeError: the renderer is closed.
    """
    self._check_open("camera_normals")
    found = self._buffers.flat_of("normal")
    if found is None:
      raise ValueError("camera_normals needs the normal output (outputs.normal)")
    normal = found[1]
    half = normal.dtype == wp.vec3h
    result = ground_truth.image_array(normal, out, "camera_normals")
    full, halved = (_placeholder(wp.vec3, self.device), normal) if half else (normal, _placeholder(wp.vec3h, self.device))
    wp.launch(
      ground_truth.camera_normals,
      dim=result.shape,
      inputs=[self._rig.lens, self._rig.rot, full, halved, half, self._tables.world_skip],
      outputs=[result],
      device=self.device,
    )
    return result

  def nocs(self, sim: S | None = None, *, out: AnyDeviceArray | None = None) -> wp.array[wp.vec3, [int, int]]:
    """Normalized object coordinates of the last render (`core/outputs/ground_truth.py`): each
    pixel's first hit in its instance's frame, centred on the instance's local box and divided by
    its diagonal, + 0.5 (the box inside the unit cube), from the depth and "seg" outputs and the
    instances' poses of `sim` (the state the last render read): (nworld, pixels) vec3 in the
    outputs' flat order (`camera_normals`), zero on a miss or a deformable. One launch, no host read
    after the first call. The rows of the worlds the last render skipped are left as they are
    (`camera_normals`).

    Args:
      sim: the simulator state of the last render (MuJoCo: the mjw.Data).
      out: the array of an earlier call: written in place (no allocation, graph-capturable).

    The "seg" output names the hit instance through the segment table (one instance per id; read
    once per table object: a table written in place later is not seen).

    Raises:
      ValueError: without the depth and "seg" outputs, for a segment table with several instances
        per id or one per world, ray-bundle cameras, or an `out` of another shape / dtype / device.
      RuntimeError: the renderer is closed.
    """
    self._check_open("nocs")
    depth, seg = self._buffers.flat_of("depth"), self._buffers.flat_of("seg")
    spec = self.settings.written_outputs().depth
    if depth is None or seg is None or spec is None:
      raise ValueError("nocs needs the depth and seg outputs (outputs.depth, outputs.seg)")
    if self._rig.rays is not None:
      raise ValueError("nocs: ray-bundle cameras have no pinhole ray to unproject")
    result = ground_truth.image_array(depth[1], out, "nocs")
    half = spec.dtype == "float16"
    full, halved = (_placeholder(float, self.device), depth[1]) if half else (depth[1], _placeholder(wp.float16, self.device))
    inputs = [
      self._gt_views(),
      self._gt_instances(self._frame(sim)),
      full,
      halved,
      half,
      spec.convention == "radial",
      seg[1],
      self._instance_of(),
      self._tables.world_skip,
    ]
    wp.launch(ground_truth.nocs, dim=result.shape, inputs=inputs, outputs=[result], device=self.device)
    return result

  def _instance_of(self) -> wp.array[int, [int]]:
    """The instance of each segment id of the renderer's segment table (`ground_truth.instance_of`;
    one host read per table: kept while the table is the same object)."""
    table = self._tables.segment_id
    held = self._nocs_ids
    if held is None or held[0] is not table:
      held = self._nocs_ids = (table, ground_truth.instance_of(host_copy(table), self.device))
    return held[1]

  def _gt_views(self) -> ground_truth.Views:
    """The cameras of the last render as the ground-truth kernels read them."""
    v = ground_truth.Views()
    rig = self._rig
    v.lens, v.res, v.pos, v.rot = rig.lens, rig.res, rig.pos, rig.rot
    v.fovy, v.intrinsics, v.distortion = self._view()
    return v

  def _gt_instances(self, frame: Frame) -> ground_truth.Instances:
    """The instances of `frame` as the NOCS kernel reads them."""
    inst = ground_truth.Instances()
    inst.shape = self.scene.instance_type
    inst.mesh, inst.size = frame.instance_mesh, frame.instance_size
    inst.mesh_centre, inst.mesh_half = self.scene.meshes.local_boxes()
    hfields = self.scene.meshes.hfield_bounds_size
    inst.hfield_half = hfields if hfields is not None else _placeholder(wp.vec3, self.device).reshape((1,))
    inst.pose_slot, inst.compact = self._tables.pose_slot, self.layout.compact
    inst.pos, inst.rot = frame.instance_pos, frame.instance_rot
    inst.shared_pos, inst.shared_rot = frame.shared_pos, frame.shared_rot
    return inst

  @property
  def semantic_labels(self) -> tuple[str, ...]:
    """Semantic label names by class id ("background", "unlabelled", then `set_semantics`'s)."""
    return self._tables.semantics.labels

  def rebuild_light_grid(self) -> None:
    """Rebuilds the light visibility grid on the next render (after moving lights or walls)."""
    self._baked.drop_grid()
    self._changed()

  def rebuild_gi(self) -> None:
    """Rebuilds and re-bakes the probe volume (GI) on the next render (after moving walls)."""
    self._baked.drop_probes()
    self._changed()

  def request_gi_relight(self, worlds: wp.array[bool, [int]]) -> None:
    """Under gi.relight="on_request": lets the masked worlds (nworld,) bool re-light their GI by
    rays at the next render where their GI inputs changed (e.g. the worlds an episode reset
    re-randomises: the request may come before or with the frame of the table writes; it waits
    for the world's next change); the others keep their GI until requested. One device launch
    (no host read, no captured graph made stale); nothing to do without a per-world GI tint (no
    GI, worlds that share their GI, or before the first render).

    Raises:
      TypeError: for something else than a Warp array.
      ValueError: for an array of another dtype, shape or device.
      RuntimeError: after `close`.
    """
    self._check_open("request_gi_relight")
    if not isinstance(worlds, wp.array):
      raise TypeError(f"worlds must be a warp array ({self.nworld},) bool, got {type(worlds).__name__}")
    if worlds.dtype != wp.bool or worlds.shape != (self.nworld,):
      raise ValueError(f"worlds must be a ({self.nworld},) bool array, got {worlds.dtype} {tuple(worlds.shape)}")
    if worlds.device != self.device:
      raise ValueError(f"the worlds array is on {worlds.device}, the renderer renders on {self.device}")
    vol = self._baked.gi
    if vol is not None and vol.relight is not None:
      vol.relight.request(worlds)

  @property
  def poses(self) -> PoseBuffers | None:
    """The renderer's (nworld, ncolumn) pose buffers of the per-world instances `layout.columns`.

    None for bindings aliasing simulator poses. Prefer `per_world.columns("instance.pos", pattern)`.
    """
    return self._per_world.poses

  def _history_length(self) -> wp.array[int, [int, int]] | None:
    """(nworld, ncam) int frames of history (the motion output), None without history
    (`RendererDebug.history_length`)."""
    return None if self._history is None else self._history.struct.count

  def explain(self) -> Explanation:
    """What the settings resolve to on this scene: every feature with the setting that asked for
    it and why it is on or off, the frame's cost class and the kernel tier serving it
    (`core/build/variants/explain.py`; `print` it)."""
    return explain_lib.explain(self.settings, self.features, self._choice)

  def counters(self) -> dict[str, float]:
    """Per-pixel means of the last frame's light-cost counters (`kernels/abi/counters.py`; a host
    read).

    Raises:
      ValueError: for a Renderer not built under
      `core.pipeline.platform.tuning.override(counters=True)`.
    """
    if not self.features.counters:
      raise ValueError(
        "Renderer.counters needs a counters build: create it under "
        "raywarp.core.pipeline.platform.tuning.override(counters=True)"
      )
    return counters_lib.per_pixel(self._tables.counters, self._rig.nray)

  def _gi_volume(self) -> gi_lib.ProbeVolume | None:
    """The GI probe volume (None before the first render or without GI;
    `RendererDebug.gi_volume`)."""
    return self._baked.gi

  def _gi_rebakes(self) -> gi_lib.GIRebakes:
    """The GI's re-bake log so far (`RendererDebug.gi_rebakes`)."""
    return self._baked.log.snapshot(self.nworld)

  def _changed(self, *, graphs: bool = True) -> None:
    """The kernel structs must be rebuilt; `graphs`: captured graphs are stale too."""
    self._generation += 1
    if graphs:
      self._version += 1

  @property
  def outputs(self) -> RenderOutputs:
    """The output arrays (replaced only by update_settings(outputs=...) and bind_outputs)."""
    return self._buffers.outputs

  def _graph_version(self) -> int:
    """Changes whenever a captured graph becomes stale (`graph.Replayable`): a change of the
    renderer, or a light grid / GI volume an eager frame replaced by itself (a
    re-bake after an in-place write: the graph's kernels read the released structures). Both
    counters only grow, so their sum changes whenever either does."""
    return self._version + self._baked.generation


def _placeholder(dtype: type, device: wp.Device) -> AnyDeviceArray:
  """A (1, 1) array standing for a buffer a kernel does not read (one per dtype and device)."""
  return _placeholder_on(dtype, str(device))


@functools.cache
def _placeholder_on(dtype: type, device: str) -> AnyDeviceArray:
  """`_placeholder` by device name."""
  return wp.zeros((1, 1), dtype=dtype, device=device)
