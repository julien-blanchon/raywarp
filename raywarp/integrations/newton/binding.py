"""Bind: a GpuScene's instances posed from Newton's `state.body_q` on the GPU (`NewtonBinding`).

Every `Renderer.render(state)` runs one kernel writing `root * (body_q[body] * local + world
offset)` into the renderer's pose buffers, resolving `state.body_q` on every call (Newton swaps
state_0 / state_1). No pose goes through the host. The global shapes nothing moves per world are
shared by a Renderer (traced once for all worlds) and posed from `model.shape_transform`.

Instances whose Newton shape color differs between worlds, and pooled instances some worlds do not
show (worlds of different shapes, `build.pooled`), get a per-world "instance.color" table
(`WorldColors`) written by a second kernel on every call for the Newton world each render world
shows: the shape color re-read from `model.shape_color` (sRGB decoded on the device; their material
is white, `ColorRule.TINT`), alpha 0 where that world has no shape for the instance (hidden: the
traversal skips it, OptiX leaves it out of the world's instances).
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, final, override

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.tables import batched
from raywarp.core.pipeline.gpu_scene.binding import Binding, PoseBuffers, Slots, TransformSource
from raywarp.core.pipeline.gpu_scene.deformables import Deformables
from raywarp.core.post.grading.color import srgb_eotf
from raywarp.integrations.newton.staging import Staging
from raywarp.integrations.newton.types import NewtonState, StateView

if TYPE_CHECKING:
  from shape_extensions import Int, IntVar

  from raywarp._base.dims import AnyDeviceArray, Ids, Mask
  from raywarp.integrations.newton.deformables import ParticleMeshes

type IntTable = np.ndarray[[int, int], np.dtype[np.int32]]
type PoseTable = np.ndarray[[int, int, 7], np.dtype[np.float32]]
# Host instance indices.
type Indices = (
  np.ndarray[[int], np.dtype[np.int32]] | np.ndarray[[int], np.dtype[np.int64]] | np.ndarray[[int], np.dtype[np.intp]]
)


@final
@dataclasses.dataclass(frozen=True)
class BindingTables:
  """(rows, ninstance) tables binding instances to Newton: a row per Newton world (or one for all).

  Attributes:
    body: Newton body moving the instance (-1: static).
    local: (rows, n, 7) its transform: offset from the body, or the world pose of a static one.
    world: Newton world whose `world_offsets` entry moves it (-1: none).
    shape: its Newton shape (-1: none, e.g. a USD visual).
  """

  body: IntTable
  local: PoseTable
  world: IntTable
  shape: IntTable

  @property
  def rows(self) -> int:
    """Number of rows."""
    return int(self.body.shape[0])

  def _broadcast(self, rows: int) -> BindingTables:
    """Read-only shared-row views for preparation, without materializing repeated inputs."""
    if rows < 1 or self.rows not in (1, rows):
      raise ValueError(f"binding rows must be 1 or {rows}, got {self.rows}; target rows must be positive")
    return BindingTables(
      body=np.broadcast_to(self.body, (rows, self.body.shape[1])),
      local=np.broadcast_to(self.local, (rows, self.local.shape[1], 7)),
      world=np.broadcast_to(self.world, (rows, self.world.shape[1])),
      shape=np.broadcast_to(self.shape, (rows, self.shape.shape[1])),
    )

  def columns(self, ids: Indices) -> BindingTables:
    """Independent, contiguous tables of the instances `ids` (every row)."""
    return BindingTables(
      body=np.ascontiguousarray(self.body[:, ids]),
      local=np.ascontiguousarray(self.local[:, ids]),
      world=np.ascontiguousarray(self.world[:, ids]),
      shape=np.ascontiguousarray(self.shape[:, ids]),
    )

  def first_row(self) -> BindingTables:
    """Independent, contiguous tables of row 0 only."""
    return BindingTables(
      body=self.body[:1].copy(order="C"),
      local=self.local[:1].copy(order="C"),
      world=self.world[:1].copy(order="C"),
      shape=self.shape[:1].copy(order="C"),
    )

  def merged(self) -> BindingTables:
    """One row holding every row's instances, world by world (one render world showing all)."""
    return BindingTables(
      body=self.body.copy(order="C").reshape(1, -1),
      local=self.local.copy(order="C").reshape(1, -1, 7),
      world=self.world.copy(order="C").reshape(1, -1),
      shape=self.shape.copy(order="C").reshape(1, -1),
    )

  @staticmethod
  def concat(parts: Sequence[BindingTables]) -> BindingTables:
    """Instances of `parts` side by side (one-row parts repeated to the others' row count)."""
    if not parts:
      raise ValueError("binding rows require at least one part")
    rows = max(p.rows for p in parts)
    views = [part._broadcast(rows) for part in parts]  # noqa: SLF001  # same-class preparation helper
    return BindingTables(
      body=np.ascontiguousarray(np.concatenate([part.body for part in views], axis=1)),
      local=np.ascontiguousarray(np.concatenate([part.local for part in views], axis=1)),
      world=np.ascontiguousarray(np.concatenate([part.world for part in views], axis=1)),
      shape=np.ascontiguousarray(np.concatenate([part.shape for part in views], axis=1)),
    )


class ShapeTable(NamedTuple):
  """The Newton shape of an instance in each world (SensorCamera's shape index output).

  Attributes:
    shape0: (ninstance,) shape of each instance in world 0.
    column: (ninstance,) column in `rows` of the instances whose shape varies per row (-1: none).
    rows: (rows, ndynamic) their shapes.
  """

  shape0: np.ndarray[[int], np.dtype[np.int32]]
  column: np.ndarray[[int], np.dtype[np.int32]]
  rows: IntTable


@wp.struct
class PoseTables:
  """Device binding tables of the instances one launch poses (`BindingTables` columns)."""

  body: wp.array2d[wp.int32]
  local: wp.array2d[wp.transformf]
  world: wp.array2d[wp.int32]
  # Pose column written for each instance.
  column: wp.array1d[wp.int32]


@wp.func
def posed(
  # In:
  pos: wp.vec3,
  rot: wp.quatf,
  root: wp.transformf,
) -> tuple[wp.vec3, wp.mat33]:
  """World pose root * (pos, rot) as the renderer's (position, rotation matrix)."""
  q = wp.transform_get_rotation(root) * rot
  # Raw simulator quaternions need not be unit; a zero / NaN one stays NaN (no silent identity).
  return wp.transform_point(root, pos), wp.quat_to_matrix(q * (1.0 / wp.length(q)))


@wp.kernel(enable_backward=False)
def _instance_poses(
  # Data in:
  body_q: wp.array1d[wp.transformf],
  view_world: wp.array1d[wp.int32],
  world_offsets: wp.array1d[wp.vec3],
  # Model:
  tables: PoseTables,
  # In:
  nsource: int,
  has_view_world: bool,
  has_world_offsets: bool,
  root: wp.transformf,
  # Out:
  xpos_out: wp.array2d[wp.vec3],
  xmat_out: wp.array2d[wp.mat33],
):
  """Pose of instance i in render world v: root * (body_q[body] * local + world offset)."""
  v, i = wp.tid()
  # The Newton world render world v shows (world 0 when out of range).
  w = v
  if has_view_world:
    w = view_world[v]
  if w < 0 or w >= nsource:
    w = 0
  b = batched(tables.body, w, i)
  x = batched(tables.local, w, i)
  if b >= 0:
    x = wp.transform_multiply(body_q[b], x)
  pos = wp.transform_get_translation(x)
  if has_world_offsets:
    ow = batched(tables.world, w, i)
    if ow >= 0 and ow < world_offsets.shape[0]:
      pos = pos + world_offsets[ow]
  c = tables.column[i]
  p, r = posed(pos, wp.transform_get_rotation(x), root)
  xpos_out[v, c] = p
  xmat_out[v, c] = r


@wp.kernel(enable_backward=False)
def _instance_colors(  # noqa: PLR0913  # the state, the model tables and the layout
  # Data in:
  shape_color: wp.array1d[wp.vec3],
  view_world: wp.array1d[wp.int32],
  # Model:
  shape: wp.array2d[wp.int32],
  column: wp.array1d[wp.int32],
  tint: wp.array1d[wp.int32],
  base: wp.array1d[wp.vec4],
  # In:
  nsource: int,
  has_view_world: bool,
  # Out:
  color_out: wp.array2d[wp.vec4],
):
  """Color of instance column[k] in render world v, from its Newton shape in the Newton world v
  shows: hidden (alpha 0) without one there, else its scene color `base[k]` with, if `tint[k]`,
  the shape's sRGB `shape_color` decoded."""
  v, k = wp.tid()
  w = v
  if has_view_world:
    w = view_world[v]
  if w < 0 or w >= nsource:
    w = 0
  s = batched(shape, w, k)
  c = base[k]
  if s < 0:
    c = wp.vec4(c[0], c[1], c[2], 0.0)
  elif tint[k] != 0:
    rgb = shape_color[s]
    c = wp.vec4(srgb_eotf(rgb[0]), srgb_eotf(rgb[1]), srgb_eotf(rgb[2]), c[3])
  color_out[v, column[k]] = c


@final
@dataclasses.dataclass(frozen=True)
class WorldColors:
  """The instances colored per world: the pooled ones some Newton worlds do not show (alpha 0
  there) and those whose shape color differs between worlds (tinted).

  Attributes:
    shape_color: the model's `shape_color` (display sRGB, read on every call; None: nothing tinted).
    ids: (k,) their instance indices.
    tint: (k,) whether instance ids[k] shows its shape's color (`Scene.meta[WORLD_COLORS_META]`).
    base: (ninstance, 4) the scene's instance colors.
  """

  shape_color: wp.array[wp.vec3, [int]] | None
  ids: np.ndarray[[int], np.dtype[np.int32]]
  tint: np.ndarray[[int], np.dtype[np.bool_]]
  base: np.ndarray[[int, 4], np.dtype[np.float32]]


@wp.kernel(enable_backward=False)
def _shared_poses(
  # Model:
  shape_tf: wp.array1d[wp.transformf],
  shapes: wp.array1d[wp.int32],
  local: wp.array1d[wp.transformf],
  # Out:
  pos_out: wp.array1d[wp.vec3],
  rot_out: wp.array1d[wp.mat33],
):
  """Pose of shared instance k: its shape's current transform (else its static local pose)."""
  k = wp.tid()
  x = local[k]
  if shapes[k] >= 0:
    x = shape_tf[shapes[k]]
  # The per-world kernel's pose with an identity root (bit for bit).
  p, r = posed(wp.transform_get_translation(x), wp.transform_get_rotation(x), wp.transform_identity())
  pos_out[k] = p
  rot_out[k] = r


@final
@dataclasses.dataclass(frozen=True)
class _Shared:
  """Device inputs of the shared poses of instances `ids` (built once)."""

  ids: np.ndarray[[int], np.dtype[np.int64]]
  shapes: wp.array[wp.int32, [int]]
  local: wp.array[wp.transformf, [int]]


class _ColorInputs(NamedTuple):
  """Device inputs of `_instance_colors` (`WorldColors`, built once)."""

  shape: wp.array[wp.int32, [int, int]]
  column: wp.array[wp.int32, [int]]
  tint: wp.array[wp.int32, [int]]
  base: wp.array[wp.vec4, [int]]


class _Layout(NamedTuple):
  """One pose buffer's binding state (`NewtonBinding.allocate`): its moving instances' tables
  (None: none move) and its shared instance count."""

  buffer: weakref.ref[wp.array[wp.vec3, [int, int]]]
  driven: PoseTables | None
  nshared: int


@final
class NewtonBinding(Binding[NewtonState]):
  """Binds a GpuScene's instances to Newton bodies (see the module docstring).

  Built from `BindingTables`, indexed `world % rows`. Only the instances that differ between
  rows (`dynamic`: on a body, in a Newton world, or different per row) keep every row; the
  others keep row 0.

  Attributes:
    shape0: (ninstance,) Newton shape of each instance in world 0.
    nsource: number of Newton worlds the rows describe (render worlds beyond it show world 0).
    shape_transform: the model's `shape_transform` (poses the shared shapes; None: static).
    share: whether static instances may be shared (False with merge_worlds: `StateView.root`
      moves everything).
  """

  writes_poses = True
  hides_by_color = True

  def __init__(
    self,
    tables: BindingTables,
    nsource: int,
    device: wp.DeviceLike,
    shape_transform: wp.array[wp.transformf, [int]] | None,
    *,
    share: bool = True,
    particles: ParticleMeshes | None = None,
    colors: WorldColors | None = None,
  ) -> None:
    """Keeps every row of the moving instances and row 0 of the others."""
    self.particles = particles
    self.colors = colors
    # Each renderer's (nworld, ninstance) "instance.color" table, by render world count; the last
    # one written (`write`) is the one the same render's slots return.
    self._color_tables: dict[int, wp.array[wp.vec4, [int, int]]] = {}
    self._color_last: wp.array[wp.vec4, [int, int]] | None = None
    self._color_inputs: _ColorInputs | None = None
    if colors is not None:
      self._color_inputs = _ColorInputs(
        shape=wp.array(np.ascontiguousarray(tables.shape[:, colors.ids]), dtype=wp.int32, device=device),
        column=wp.array(colors.ids, dtype=wp.int32, device=device),
        tint=wp.array(colors.tint.astype(np.int32), dtype=wp.int32, device=device),
        base=wp.array(np.ascontiguousarray(colors.base[colors.ids]), dtype=wp.vec4, device=device),
      )
    # Each renderer's (nworld, nvert) particle-mesh vertices ("flex.vert"), by render world count;
    # the last one written (`write`) is the one the same render's slots return.
    self._flex: dict[int, wp.array[wp.vec3, [int, int]]] = {}
    self._flex_last: wp.array[wp.vec3, [int, int]] | None = None
    self.device = wp.get_device(device)
    self.ninstance = int(tables.body.shape[1])
    dyn = np.ones(self.ninstance, bool)
    if share:
      dyn = (
        np.any(tables.body >= 0, axis=0)
        | np.any(tables.world >= 0, axis=0)
        | np.any(tables.local != tables.local[:1], axis=(0, 2))
        | np.any(tables.shape != tables.shape[:1], axis=0)
      )
    self._dyn: Mask[int] = dyn
    self._dyn_ids = np.flatnonzero(dyn)
    self._rows = tables.columns(self._dyn_ids)
    self._row0 = tables.first_row()
    self.shape0 = self._row0.shape[0]
    self._no_world = wp.zeros(1, dtype=wp.int32, device=self.device)
    self._no_offsets = wp.zeros(1, dtype=wp.vec3, device=self.device)
    self._no_bodies = wp.zeros(1, dtype=wp.transformf, device=self.device)
    self._no_colors = wp.zeros(1, dtype=wp.vec3, device=self.device)
    self.nsource = max(nsource, 1)
    self.shape_transform = shape_transform
    self.share = share
    self._staging = Staging(self.device)  # model arrays on another device, copied once per frame
    # Per pose buffer (one per Renderer: its own columns, `allocate`): the moving instances'
    # tables (written every render) and its shared instance count, keyed by id(buffers.pos).
    self._layouts: dict[int, _Layout] = {}
    self._shared: _Shared | None = None
    # (ids, their Newton shapes) of the last `shared_motion` call.
    self._motion_shapes: tuple[Ids[int], np.ndarray[[int], np.dtype[np.int32]]] | None = None

  @override
  def dynamic(self, ninstance: int) -> Mask[int]:
    """Instances moved per world (everything without `share`)."""
    return self._dyn

  def shape_table(self) -> ShapeTable:
    """The Newton shape of an instance in each world."""
    col = np.full(self.ninstance, -1, np.int32)
    col[self._dyn_ids] = np.arange(len(self._dyn_ids)).astype(np.int32)
    return ShapeTable(self.shape0, col, self._rows.shape)

  @staticmethod
  def _view(sim: NewtonState | None) -> StateView:
    """`sim` as a StateView (raises without a state)."""
    if sim is None:
      raise ValueError("a Newton scene renders the body poses of a newton.State: pass it to render(state)")
    return sim if isinstance(sim, StateView) else StateView(sim)

  @override
  def deformables(self, device: wp.Device) -> Deformables:
    """The model's triangle meshes over particles (cloth, soft-body skins:
    `raywarp.integrations.newton.deformables`)."""
    return Deformables.empty(device) if self.particles is None else self.particles.deformables(device)

  @override
  def slots(self, sim: NewtonState | None) -> Slots:
    """The particle meshes' vertices ("flex.vert") and the per-world instance colors
    ("instance.color", `WorldColors`) of this render, written by `write`; the instance poses are
    written into the Renderer's buffers (`write`)."""
    if sim is None:
      return {}
    out: Slots = {}
    if self._flex_last is not None:
      out["flex.vert"] = self._flex_last
    if self._color_last is not None:
      out["instance.color"] = self._color_last
    return out

  @override
  def sources(self, sim: NewtonState | None) -> dict[str, AnyDeviceArray]:
    """The per-call arrays the pose kernel reads (fingerprinted by captured graphs): body_q, ...,
    and the staged copies of arrays on another device (a reallocated one is stale).

    Empty without a state (e.g. when `Renderer.per_world` reads the tables).
    """
    if sim is None:
      return {}
    view = self._view(sim)
    named: dict[str, AnyDeviceArray | None] = {
      "state.body_q": view.state.body_q,
      "state.particle_q": view.state.particle_q if self.particles is not None else None,
      "state.world_indices": view.world_indices,
      "state.world_offsets": view.world_offsets,
      "model.shape_color": self.colors.shape_color if self.colors is not None else None,
    }
    return {**{name: array for name, array in named.items() if array is not None}, **self._staging.buffers()}

  @override
  def stage(self, sim: NewtonState | None) -> None:
    """Uploads the arrays on another device than the renderer (`staging`), outside any graph."""
    if sim is None:
      return
    view = self._view(sim)
    self._staging.upload("body_q", view.state.body_q)
    self._staging.upload("world_indices", view.world_indices)
    self._staging.upload("world_offsets", view.world_offsets)
    self._staging.upload("shape_transform", self.shape_transform)
    if self.colors is not None and self.colors.shape_color is not None:
      self._staging.upload("shape_color", self.colors.shape_color)
    if self.particles is not None:
      self._staging.upload("particle_q", view.state.particle_q)

  def _device_tables(self, ids: Indices, cols: Indices, *, driven: bool) -> PoseTables:
    """PoseTables of instances `ids` written at pose columns `cols`: every row (moving) or row 0."""
    src = self._rows if driven else self._row0
    at = np.searchsorted(self._dyn_ids, ids) if driven else ids
    tables = PoseTables()
    tables.body = wp.array(np.ascontiguousarray(src.body[:, at]), dtype=wp.int32, device=self.device)
    tables.local = wp.array(np.ascontiguousarray(src.local[:, at]), dtype=wp.transformf, device=self.device)
    tables.world = wp.array(np.ascontiguousarray(src.world[:, at]), dtype=wp.int32, device=self.device)
    tables.column = wp.array(cols.astype(np.int32), dtype=wp.int32, device=self.device)
    return tables

  def _launch[NWorld: IntVar, NColumn: IntVar](
    self, tables: PoseTables, poses: PoseBuffers[NWorld, NColumn], view: StateView | None
  ) -> None:
    """Runs the pose kernel of `tables` for `view` (None: the static pose, no state).

    Raises:
      ValueError: for `StateView.world_indices` shorter than the render worlds.
    """
    body_q = worlds = offsets = None
    if view is not None:
      body_q = self._staging.get("body_q", view.state.body_q)
      offsets = self._staging.get("world_offsets", view.world_offsets)
      worlds = self._staging.get("world_indices", view.world_indices)
    if worlds is not None and worlds.shape[0] < poses.pos.shape[0]:
      raise ValueError(f"StateView.world_indices has {worlds.shape[0]} entries for {poses.pos.shape[0]} render worlds")
    root = view.root if view is not None and view.root is not None else wp.transform_identity()
    wp.launch(
      _instance_poses,
      dim=(poses.pos.shape[0], tables.column.shape[0]),
      inputs=[
        body_q if body_q is not None else self._no_bodies,
        worlds if worlds is not None else self._no_world,
        offsets if offsets is not None else self._no_offsets,
        tables,
        self.nsource,
        worlds is not None,
        offsets is not None,
        root,
      ],
      outputs=[poses.pos, poses.rot],
      device=self.device,
    )

  @override
  def allocate[NWorld: IntVar, NColumn: IntVar](
    self, nworld: Int[NWorld], columns: Ids[NColumn], *, device: wp.Device
  ) -> PoseBuffers[NWorld, NColumn]:
    """Pose buffers of instances `columns`: static ones written now, moving ones every render.

    Also builds the inputs of the shared instances (the others) once.
    """
    cols = np.asarray(columns, np.int64)
    shape = (nworld, len(cols))
    out = PoseBuffers(wp.zeros(shape, dtype=wp.vec3, device=device), wp.zeros(shape, dtype=wp.mat33, device=device))
    dyn = self._dyn[cols]
    driven_tables = None
    for driven in (False, True):
      k = np.flatnonzero(dyn == driven)
      if not len(k):
        continue
      tables = self._device_tables(cols[k], k, driven=driven)
      if driven:
        driven_tables = tables
      else:
        self._launch(tables, out, None)
    self._layouts = {key: layout for key, layout in self._layouts.items() if layout.buffer() is not None}
    self._layouts[id(out.pos)] = _Layout(weakref.ref(out.pos), driven_tables, self.ninstance - len(cols))
    self._shared = self._shared_inputs(np.setdiff1d(np.arange(self.ninstance), cols))
    if self.particles is not None and nworld not in self._flex:
      self._flex[nworld] = wp.zeros((nworld, self.particles.rows.count), dtype=wp.vec3, device=device)
    if self.colors is not None and nworld not in self._color_tables:
      base = np.ascontiguousarray(np.broadcast_to(self.colors.base, (nworld, *self.colors.base.shape)))
      self._color_tables[nworld] = wp.array(base, dtype=wp.vec4, device=device)
    return out

  @override
  def write[NWorld: IntVar, NColumn: IntVar](self, poses: PoseBuffers[NWorld, NColumn], sim: NewtonState | None) -> None:
    """Launches the pose kernel of the moving instances for this call's state."""
    view = self._view(sim)
    self._staging.next_frame()
    layout = self._layouts.get(id(poses.pos))
    if layout is None or layout.buffer() is not poses.pos:
      raise ValueError("NewtonBinding.write: pose buffers not made by this binding's allocate")
    if view.root is not None and layout.nshared:
      raise ValueError("StateView.root needs a scene without shared instances (merge_worlds=True)")
    if layout.driven is not None:
      self._launch(layout.driven, poses, view)
    if self.colors is not None:
      self._write_colors(view, poses.pos.shape[0])
    if self.particles is not None:
      self._gather_particles(view, poses.pos.shape[0])

  def _gather_particles(self, view: StateView, nworld: int) -> None:
    """The particle meshes' vertices of the `nworld` render worlds (`ParticleMeshes.gather`).

    Raises:
      ValueError: for a state without particle positions.
    """
    particles = self.particles
    if particles is None:
      return
    particle_q = self._staging.get("particle_q", view.state.particle_q)
    if particle_q is None:
      raise ValueError("the Newton state has no particle_q for the model's particle meshes")
    vert = self._flex[nworld]
    particles.gather(
      particle_q,
      vert,
      world_indices=self._staging.get("world_indices", view.world_indices),
      world_offsets=self._staging.get("world_offsets", view.world_offsets),
      root=view.root if view.root is not None else wp.transform_identity(),
      empty_ids=self._no_world,
      empty_offsets=self._no_offsets,
    )
    self._flex_last = vert

  def _shared_shapes(self, ids: Indices) -> np.ndarray[[int], np.dtype[np.int32]]:
    """Newton shapes of the shared instances `ids` (-1 without `shape_transform`)."""
    if self.shape_transform is None:
      return np.full(len(ids), -1, np.int32)
    return self.shape0[np.asarray(ids, np.int64)].astype(np.int32)

  def _shared_inputs(self, ids: Indices) -> _Shared:
    """Device shapes and static poses of the shared instances `ids`."""
    rows = np.asarray(ids, np.int64)
    return _Shared(
      rows,
      wp.array(self._shared_shapes(rows), dtype=wp.int32, device=self.device),
      wp.array(np.ascontiguousarray(self._row0.local[0, rows]), dtype=wp.transformf, device=self.device),
    )

  def _write_colors(self, view: StateView, nworld: int) -> None:
    """Writes this call's per-world colors of `WorldColors`' instances (`_instance_colors`)."""
    colors, inputs = self.colors, self._color_inputs
    if colors is None or inputs is None:
      return
    table = self._color_tables[nworld]
    worlds = self._staging.get("world_indices", view.world_indices)
    shape_color = self._staging.get("shape_color", colors.shape_color) if colors.shape_color is not None else None
    wp.launch(
      _instance_colors,
      dim=(nworld, len(colors.ids)),
      inputs=[
        shape_color if shape_color is not None else self._no_colors,
        worlds if worlds is not None else self._no_world,
        inputs.shape,
        inputs.column,
        inputs.tint,
        inputs.base,
        self.nsource,
        worlds is not None,
      ],
      outputs=[table],
      device=self.device,
    )
    self._color_last = table

  @override
  def write_shared[NShared: IntVar](
    self, ids: Ids[NShared], pos: wp.array[wp.vec3, [NShared]], rot: wp.array[wp.mat33, [NShared]], sim: NewtonState | None
  ) -> None:
    """The shared instances' poses from their shapes' current `model.shape_transform`."""
    if not len(ids):
      return
    shared = self._shared
    if shared is None or not np.array_equal(shared.ids, ids):
      shared = self._shared = self._shared_inputs(ids)
    tf = self._staging.get("shape_transform", self.shape_transform)
    wp.launch(
      _shared_poses,
      dim=len(shared.ids),
      inputs=[tf if tf is not None else self._no_bodies, shared.shapes, shared.local],
      outputs=[pos, rot],
      device=self.device,
    )

  @override
  def shared_motion(self, sim: NewtonState | None, ids: Ids[int]) -> TransformSource | None:
    """The shared shapes' `model.shape_transform` (checked by the renderer's per-frame
    validation)."""
    tf = self._staging.get("shape_transform", self.shape_transform)
    if tf is None:
      return None
    cached = self._motion_shapes
    if cached is None or cached[0] is not ids:
      # The renderer passes the same (layout's) ids every frame: converted once.
      cached = self._motion_shapes = (ids, self._shared_shapes(ids))
    return TransformSource(tf, cached[1])
