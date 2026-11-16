"""Checks, every frame, that the data a Renderer reads stays valid (ids in range, finite colors,
shared scenery static), with one small kernel whose flags the host reads one frame late.

Every render enqueues `Validator.check` after its frame: one kernel compares what the frame read
with what the renderer assumed, and writes flags into a device word that is copied to pinned host
memory. The next render reads that copy (`Validator.poll`, no synchronisation: one frame late) and
acts on it. The flags (slots of `Word`):

* `moved`: a world's shared instances differ from the snapshot of the first frame, which the
  shared trees were built from (poses, and sizes / meshes of live tables, e.g. an aliased
  `m.geom_size`): an error. Shared scenery is static (stored and traced once); instances that
  move or change are declared per world.
* `env_id` / `mesh_id` / `material_id` / `texture_id`: an index table out of range ("env.id",
  "instance.mesh", a live "instance.material" or "material.texture"); the kernels clamp such
  indices meanwhile (never an out-of-bounds read).
* `exposure`: a per-world exposure that is not finite and > 0 (0 would make the world's HDR
  NaN / inf); the kernels read it as 1 meanwhile (`tables.exposure_at`).
* `color`: a per-world or live "instance.color" / "material.base_color" that is not finite
  (NaN would make the world's HDR NaN); the kernels read it as black (opaque) meanwhile
  (`tables.instance_rgba`).
* `compiled_out`: reserved for checks that live data needs a feature the kernels were compiled
  without: `Binding.feature_checks()` enqueue them (`FeatureCheck`).
* `derived` / `rays` / `camera`: a declared per-world OpenPBR input out of range (`ranges`), camera
  rays with NaN / inf (`invalid_rays`), a camera posed at a non-finite position or rotation
  (`camera_poses`): the renderer-owned checks, recorded once.
* `world_ids`: the render's active world ids held one out of [0, nworld) or a repeat
  (`active.ActiveWorlds.invalid`; such an entry is not rendered meanwhile).
* `casters`: a light casts a shadow that the Features were resolved without as a potential caster
  (`casters`, `scene.shadow_casters`; armed only while one more caster would turn the shadow
  budget on). Not an error: the Renderer re-resolves its Features.

The word (one int32, `Word`) also holds the epoch it was checked against, which changes with the
snapshot and every report `poll` returns (on the device's stream): a word checked before the last
report (a late pinned copy on CUDA) is ignored, so a fixed error does not repeat. `poll` never
writes the pinned word and reads it in one aligned 32-bit load: a copy landing meanwhile is read
whole or not at all.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, NamedTuple, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.gpu_scene.binding import RowsSource, TransformSource
from raywarp.core.pipeline.platform.device import device_caps, host_copy
from raywarp.core.pipeline.platform.launches import recorded
from raywarp.core.pipeline.worlds import validator_kernels as checks
from raywarp.core.pipeline.worlds.validator_kernels import EPOCH_MASK, EPOCH_SHIFT, FLAG_MASK, Word

if TYPE_CHECKING:
  from raywarp._base.dims import Ids
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.binding import FeatureCheck


@final
class Flags(NamedTuple):
  """The flags of one polled word (all False: nothing to report), in `Word` order (checked at
  import: `Flags(*bits)` is built positionally)."""

  moved: bool
  env_id: bool
  mesh_id: bool
  compiled_out: bool
  material_id: bool
  texture_id: bool
  exposure: bool
  color: bool
  derived: bool
  rays: bool
  casters: bool
  camera: bool
  world_ids: bool

  def any(self) -> bool:
    """Whether any flag is set."""
    return any(self)


@final
class Range(NamedTuple):
  """A declared per-world table the validation range-checks (`checks.check_range`).

  Attributes:
    field: the per-world field ("material.base_metalness", ...), named in the error.
    values: (nworld, n) float32: the table, or a float view of its components (vec3 (nworld, n)
      -> (nworld, 3 n)).
    low, high: every value is in [low, high] (NaN never; inf only where a bound is inf).
  """

  field: str
  values: wp.array[float, [int, int]]
  low: float
  high: float


if Flags._fields != tuple(w.name.lower() for w in Word):
  raise ImportError(f"validator.Flags {Flags._fields} differ from the Word bits {[w.name for w in Word]}")
NO_FLAGS: Final = Flags(*(False for _ in Word))

# The validation kernel's inputs by name, in its parameter order (`flags`, its output, last). Not a
# `wp.struct`: the recorded launch already packs them once, so a struct would add a second cache
# and change the per-frame kernel for no win; named (`Validator._inputs`), same-typed inputs
# cannot swap places.
VALIDATE_INPUTS: Final = tuple(inspect.signature(checks.validate).parameters)[:-1]  # the kernel wraps its function


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Snapshot:
  """What the shared trees were built from (the first frame's shared scenery), in device buffers
  allocated once (a captured graph keeps reading them).

  Attributes:
    pose_ids: (npose,) index of each checked shared instance in its pose source.
    pos, rot: (npose,) world 0's poses (a `RowsSource`).
    transforms: (npose,) source transforms (a `TransformSource`).
    shape_ids: (nshape,) the rendered shared instances whose size / mesh tables are live.
    size, mesh: (nshape,) world 0's sizes and mesh ids.
    rows: the pose source is per-world rows (else transforms).
    pose_instances, shape_instances: (npose,) / (nshape,) instance of each checked entry.
  """

  pose_ids: wp.array[int, [int]]
  pos: wp.array[wp.vec3, [int]]
  rot: wp.array[wp.mat33, [int]]
  transforms: wp.array[wp.transformf, [int]]
  shape_ids: wp.array[int, [int]]
  size: wp.array[wp.vec3, [int]]
  mesh: wp.array[int, [int]]
  rows: bool
  pose_instances: Ids[int]
  shape_instances: Ids[int]

  @property
  def npose(self) -> int:
    """Checked shared poses (`pose_ids` has one placeholder entry when there are none)."""
    return len(self.pose_instances)

  @property
  def nshape(self) -> int:
    """Checked shared sizes / meshes (`shape_ids` likewise)."""
    return len(self.shape_instances)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Watched:
  """What a Renderer's validation reads besides the frame (fixed at its construction).

  Attributes:
    shared_ids: (nshared,) the rendered shared instances (`Layout.shared_ids`).
    live_shapes: the frame's size / mesh tables can change after construction (a binding's or a
      per-world table, not the scene's upload): the shared ones are snapshot and checked.
    env_ids: the env-id table is a per-world declaration (checked against `nenv`).
    mesh_ids: the instance-mesh table is a per-world declaration or a binding's (checked).
    material_ids: the instance-material table is a binding's (checked against `nmaterial`).
    texture_ids: the material-texture table is a binding's (checked against `ntexture`).
    exposure: the exposure table is a per-world declaration (checked: finite and > 0).
    colors: a color table is a per-world declaration or a binding's (checked: finite).
    nenv, nmesh, nhfield, nmaterial, ntexture: table sizes the indices must stay below.
  """

  shared_ids: Ids[int]
  live_shapes: bool
  env_ids: bool
  mesh_ids: bool
  material_ids: bool
  texture_ids: bool
  exposure: bool
  colors: bool
  nenv: int
  nmesh: int
  nhfield: int
  nmaterial: int
  ntexture: int


@final
class Validator:
  """The per-frame validation of one Renderer (see the module docstring).

  Attributes:
    watched: what is checked (`Watched`).
    checks: the binding's feature checks (`FeatureCheck`), enqueued after the validation kernel.
    snapshot: the shared trees' snapshot (None before the first frame: nothing shared is checked).
    epoch: the current epoch (host; see the module doc); the device copy is read by the kernel.
    flags: the device word (one int32: epoch and flag bits, `Word`); `host` its pinned copy, read
      one frame late.
  """

  def __init__(self, watched: Watched, checks: tuple[FeatureCheck, ...], *, device: wp.DeviceLike) -> None:
    """Allocates the flag words on `device` (the snapshot comes with the first `snap`)."""
    self.watched = watched
    self.checks = checks
    self.device = wp.get_device(device)
    self.snapshot: Snapshot | None = None
    self.epoch = 0
    self._epoch = wp.zeros(1, dtype=wp.int32, device=self.device)
    self.flags = wp.zeros(1, dtype=wp.int32, device=self.device)
    self.host = wp.zeros(1, dtype=wp.int32, device="cpu", pinned=device_caps(self.device).pinned)
    self._empty = _Empty(self.device)
    # The declared per-world tables checked for range (`PerWorldTables.ranges`).
    self.ranges: list[Range] = []
    # The camera rig's invalid bundle-ray count (`Cameras.invalid_rays`; None: no ray bundle).
    self.invalid_rays: wp.array[wp.int32, [1]] | None = None
    # The frame's camera poses (`CameraRig.pos` / `rot`, written in place every frame; None: not
    # checked).
    self.camera_poses: tuple[wp.array[wp.vec3, [int, int]], wp.array[wp.mat33, [int, int]]] | None = None
    # The Features' potential shadow casters (nlight,), checked while a new caster would change the
    # shadow budget (None: not checked; `Renderer._configure`).
    self.casters: wp.array[bool, [int]] | None = None
    # The render's invalid active-world-id flag (`active.ActiveWorlds.invalid`; None: no active
    # list).
    self.world_ids: wp.array[wp.int32, [1]] | None = None
    self._renderer_launches: list[wp.Launch] | None = None
    # What frames check before the first `snap` (nothing shared moves): allocated here, so a steady
    # frame allocates and uploads nothing, also inside a captured CUDA graph.
    self._unshared = self._allocate(None)
    # (input identities, recorded launch) of the last check. The recorded launch keeps its argument
    # arrays alive (`wp.Launch.fwd_args`), so no other array can take a cached identity.
    self._launch: tuple[tuple[object, ...], wp.Launch] | None = None

  def set_world_ids(self, invalid: wp.array[wp.int32, [1]] | None) -> None:
    """Checks the active list's invalid flag `invalid` (None: none) from the next frame on."""
    self.world_ids = invalid
    self._renderer_launches = None

  def set_ranges(self, ranges: list[Range]) -> None:
    """Checks `ranges` from the next frame on (the recorded launches of the previous tables
    dropped)."""
    self.ranges = ranges
    self._renderer_launches = None

  def snap(self, source: RowsSource | TransformSource | None, frame: Frame) -> None:
    """Takes the snapshot of the shared scenery (once: the first frame's), under a new epoch."""
    snap = self.snapshot = self._allocate(source)
    if snap.npose or snap.nshape:
      rows_pos, rows_rot, transforms = self._source_arrays(source)
      wp.launch(
        checks.snap,
        dim=max(snap.npose, snap.nshape),
        inputs=[
          rows_pos,
          rows_rot,
          transforms,
          snap.pose_ids,
          snap.rows,
          snap.npose,
          frame.instance_size,
          frame.instance_mesh,
          snap.shape_ids,
          snap.nshape,
        ],
        outputs=[snap.pos, snap.rot, snap.transforms, snap.size, snap.mesh],
        device=self.device,
      )
    self._next_epoch()

  def _next_epoch(self) -> None:
    """A new epoch, enqueued on the device's stream: words of checks enqueued before are stale."""
    self.epoch += 1
    self._epoch.fill_((self.epoch & EPOCH_MASK) << EPOCH_SHIFT)

  def _allocate(self, source: RowsSource | TransformSource | None) -> Snapshot:
    """The snapshot buffers of `source`'s checked instances (sized once)."""
    ids = np.asarray(self.watched.shared_ids, np.int32)
    index = np.asarray(source.index, np.int32) if source is not None else ids[:0]
    keep = np.flatnonzero(index >= 0)
    shapes = ids if self.watched.live_shapes else ids[:0]
    npose, nshape, dev = max(len(keep), 1), max(len(shapes), 1), self.device
    return Snapshot(  # buffers of at least one entry (placeholders when nothing is checked)
      pose_ids=wp.array(np.resize(index[keep], npose), dtype=int, device=dev),
      pos=wp.zeros(npose, dtype=wp.vec3, device=dev),
      rot=wp.zeros(npose, dtype=wp.mat33, device=dev),
      transforms=wp.zeros(npose, dtype=wp.transformf, device=dev),
      shape_ids=wp.array(np.resize(shapes, nshape), dtype=int, device=dev),
      size=wp.zeros(nshape, dtype=wp.vec3, device=dev),
      mesh=wp.zeros(nshape, dtype=int, device=dev),
      rows=isinstance(source, RowsSource),
      pose_instances=ids[keep],
      shape_instances=shapes,
    )

  def _source_arrays(
    self, source: RowsSource | TransformSource | None
  ) -> tuple[wp.array[wp.vec3, [int, int]], wp.array[wp.mat33, [int, int]], wp.array[wp.transformf, [int]]]:
    """(rows pos, rows rot, transforms) of `source`, placeholders for the absent ones."""
    e = self._empty
    if isinstance(source, RowsSource):
      return source.pos, source.rot, e.transforms
    if isinstance(source, TransformSource):
      return e.rows_pos, e.rows_rot, source.transforms
    return e.rows_pos, e.rows_rot, e.transforms

  def active(self) -> bool:
    """Whether a frame has anything to check (else `check` enqueues nothing)."""
    w = self.watched
    snap = self.snapshot
    shared = snap is not None and (snap.npose or snap.nshape)
    return bool(
      shared
      or w.env_ids
      or w.mesh_ids
      or w.material_ids
      or w.texture_ids
      or w.exposure
      or w.colors
      or self.checks
      or self.ranges
      or self.invalid_rays is not None
      or self.camera_poses is not None
      or self.casters is not None
      or self.world_ids is not None
    )

  def check(
    self,
    source: RowsSource | TransformSource | None,
    frame: Frame,
    *,
    features: Features,
    instance_type: wp.array[int, [int]],
  ) -> None:
    """Enqueues the frame's checks and the copy of the word to pinned memory (no sync)."""
    if not self.active():
      return
    snap = self.snapshot if self.snapshot is not None else self._unshared
    npose = snap.npose if source is not None else 0
    nworld = int(frame.instance_pos.shape[0])
    named = self._inputs(source, frame, snap=snap, npose=npose, instance_type=instance_type)
    inputs = [named[name] for name in VALIDATE_INPUTS]
    # The packed launch is reused while its inputs are the same objects (33 arguments: packing them
    # is most of the check's host cost).
    key = (nworld, *(id(a) if isinstance(a, wp.array) else a for a in inputs))
    launch = self._recorded(key, (nworld, self._width(frame, snap, npose)), inputs)
    self.flags.zero_()
    launch.launch()
    for c in self.checks:
      c.enqueue(features, frame, self.flags)
    for cmd in self._renderer_checks():
      cmd.launch()
    if self.casters is not None and frame.light_cast_shadow.shape[1]:
      dim, inputs = frame.light_cast_shadow.shape, [frame.light_cast_shadow, self.casters]
      wp.launch(checks.check_casters, dim=dim, inputs=inputs, outputs=[self.flags], device=self.device)
    wp.copy(self.host, self.flags)

  def prepare(self) -> None:
    """Records the renderer-owned checks now (loading their kernels' module): a first check inside
    a graph capture would load it there (an allocation a captured conditional body rejects)."""
    self._renderer_checks()

  def _renderer_checks(self) -> list[wp.Launch]:
    """The recorded launches of the renderer-owned checks (`ranges`, `invalid_rays`,
    `camera_poses`, `world_ids`; recorded once: `set_ranges` / `set_world_ids` invalidate them
    when their storage is rebound)."""
    if self._renderer_launches is not None:
      return self._renderer_launches
    out, dev = [self.flags], self.device
    launches = [
      recorded(wp.launch(checks.check_range, dim=v.shape, inputs=[v, low, high], outputs=out, device=dev, record_cmd=True))
      for _, v, low, high in self.ranges
    ]
    if self.invalid_rays is not None:
      inputs = [self.invalid_rays, wp.int32(Word.RAYS)]
      launches.append(recorded(wp.launch(checks.check_count, dim=1, inputs=inputs, outputs=out, device=dev, record_cmd=True)))
    if self.camera_poses is not None:
      pos, rot = self.camera_poses
      launch = wp.launch(checks.check_camera_poses, dim=pos.shape, inputs=[pos, rot], outputs=out, device=dev, record_cmd=True)
      launches.append(recorded(launch))
    if self.world_ids is not None:
      inputs = [wp.int32(Word.WORLD_IDS), self.world_ids]
      launches.append(recorded(wp.launch(checks.take_count, dim=1, inputs=inputs, outputs=out, device=dev, record_cmd=True)))
    self._renderer_launches = launches
    return launches

  def _width(self, frame: Frame, snap: Snapshot, npose: int) -> int:
    """Entries per world of the validation launch: the widest checked table."""
    w = self.watched
    widths = [npose, snap.nshape, 1]
    widths += [int(frame.instance_mesh.shape[1])] if w.mesh_ids else []
    widths += [int(frame.instance_material.shape[1])] if w.material_ids else []
    widths += [int(frame.material_texture.shape[1])] if w.texture_ids else []
    widths += [int(frame.instance_color.shape[1]), int(frame.material_base_color.shape[1])] if w.colors else []
    return max(widths)

  def _inputs(
    self,
    source: RowsSource | TransformSource | None,
    frame: Frame,
    *,
    snap: Snapshot,
    npose: int,
    instance_type: wp.array[int, [int]],
  ) -> dict[str, object]:
    """The validation kernel's inputs by parameter name (`VALIDATE_INPUTS`: every parameter of
    `validator_kernels.validate` before `flags`)."""
    w = self.watched
    rows_pos, rows_rot, transforms = self._source_arrays(source)
    return {
      "rows_pos": rows_pos,
      "rows_rot": rows_rot,
      "transforms": transforms,
      "pose_ids": snap.pose_ids,
      "rows": snap.rows,
      "npose": npose,
      "snap_pos": snap.pos,
      "snap_rot": snap.rot,
      "snap_tf": snap.transforms,
      "instance_size": frame.instance_size,
      "instance_mesh": frame.instance_mesh,
      "shape_ids": snap.shape_ids,
      "nshape": snap.nshape,
      "snap_size": snap.size,
      "snap_mesh": snap.mesh,
      "env_id": frame.env_id if w.env_ids else self._empty.env_id,
      "nenv": w.nenv,
      "instance_type": instance_type,
      "check_mesh": w.mesh_ids,
      "nmesh": w.nmesh,
      "nhfield": w.nhfield,
      "instance_material": frame.instance_material,
      "check_material": w.material_ids,
      "nmaterial": w.nmaterial,
      "material_texture": frame.material_texture,
      "check_texture": w.texture_ids,
      "ntexture": w.ntexture,
      "exposure": frame.exposure,
      "check_exposure": w.exposure,
      "instance_color": frame.instance_color,
      "material_base_color": frame.material_base_color,
      "check_colors": w.colors,
      "epoch": self._epoch,
    }

  def _recorded(self, key: tuple[object, ...], dim: tuple[int, int], inputs: Sequence[object]) -> wp.Launch:
    """The recorded launch of the validation kernel for `inputs` (reused while `key` holds)."""
    if self._launch is not None and self._launch[0] == key:
      return self._launch[1]
    cmd = recorded(
      wp.launch(checks.validate, dim=dim, inputs=inputs, outputs=[self.flags], device=self.device, record_cmd=True)
    )
    self._launch = (key, cmd)
    return cmd

  def poll(self) -> Flags:
    """The flags of the last word copied for the current epoch (no synchronisation), consumed: a
    report starts a new epoch (see the module doc)."""
    word = int(host_copy(self.host)[0])  # one aligned 32-bit load (module doc)
    if word >> EPOCH_SHIFT != self.epoch & EPOCH_MASK or not word & FLAG_MASK:
      return NO_FLAGS
    self._next_epoch()
    return Flags(*(bool(word & bit) for bit in Word))


@final
class _Empty:
  """Placeholders of the absent validation inputs (allocated once)."""

  def __init__(self, device: wp.Device) -> None:
    """Zero-size / single-entry arrays on `device`."""
    self.rows_pos = wp.zeros((1, 1), dtype=wp.vec3, device=device)
    self.rows_rot = wp.zeros((1, 1), dtype=wp.mat33, device=device)
    self.transforms = wp.zeros(1, dtype=wp.transformf, device=device)
    self.env_id = wp.zeros(0, dtype=int, device=device)
