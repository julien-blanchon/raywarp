"""What a Renderer does with its per-frame validation (`core/pipeline/worlds/validator.py`): raise
for invalid data.

`Watch` decides what is checked (`Watched`), snapshots the shared scenery after the first frame,
enqueues the check after each frame, and acts on the word one frame later: raises for errors
(shared scenery that moved or changed: it is static; out-of-range indices, live data needing a
compiled-out feature) and records a new shadow caster (the Renderer re-resolves its Features).
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, final

import numpy as np

from raywarp.core.pipeline.gpu_scene.binding import RowsSource, TransformSource
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.worlds.sharing import COLOR_FIELDS
from raywarp.core.pipeline.worlds.validator import Range, Snapshot, Validator, Watched

if TYPE_CHECKING:
  from raywarp._base.dims import Ids
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.pipeline.gpu_scene.binding import Slots
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.pipeline.worlds.sharing import Layout
  from raywarp.core.pipeline.worlds.validator import Flags

# Instance names an error or warning lists (then "...").
_LISTED: Final = 5
# A bound at least this large is printed as inf (the largest float32 stands for "any finite").
_UNBOUNDED: Final = 1.0e38


def pose_hint(names: list[str], every: list[str]) -> str:
  """The `PerWorld(pose=[...])` patterns that declare exactly `names` among `every` instance name:
  one glob on their common path prefix (e.g. "room/*" for the geoms of an mjlab entity) when it
  matches no other instance, else the names themselves."""
  prefix = os.path.commonprefix(names)  # noqa: RUF071  # a character prefix, cut at its last '/'
  prefix = prefix[: prefix.rfind("/") + 1]
  if len(names) > 1 and prefix and set(fnmatch.filter(every, prefix + "*")) == set(names):
    return repr(prefix + "*")
  return ", ".join(repr(n) for n in names)


def watched(scene: GpuScene, layout: Layout, binding_slots: Slots, per_world_fields: frozenset[str]) -> Watched:
  """What a renderer's validation checks: the shared instances, live size / mesh tables (the
  binding's), the per-world env ids, mesh ids, the binding's material and texture ids, a per-world
  exposure and the per-world or live color tables."""
  meshes = scene.meshes
  hfields = meshes.hfield_bvh_id
  live_mesh = "instance.mesh" in binding_slots
  return Watched(
    shared_ids=layout.shared_ids,
    live_shapes="instance.size" in binding_slots or live_mesh,
    env_ids="env.id" in per_world_fields,
    mesh_ids=bool(live_mesh or layout.groups["mesh"].any()),
    material_ids="instance.material" in binding_slots,
    texture_ids="material.texture" in binding_slots,
    exposure="exposure" in per_world_fields,
    colors=any(f in per_world_fields or f in binding_slots for f in COLOR_FIELDS),
    nenv=int(scene.environment.maps.radiance.shape[0]),
    nmesh=meshes.nmesh,
    nhfield=int(hfields.shape[0]) if hfields is not None else 0,
    nmaterial=scene.nmaterial,
    ntexture=int(scene.texture_array.shape[0]),
  )


def _bound(x: float) -> str:
  """`x` printed ("inf" past `_UNBOUNDED`)."""
  return "inf" if abs(x) >= _UNBOUNDED else f"{x:g}"


def _out_of_range(ranges: list[Range]) -> str:
  """The checked fields holding a value out of their bounds now, each with its bounds, its first
  offending world and value and the count ("light.cutoff in [0, 180]: world 2 has 200 (3
  values)"); every checked field with its bounds when none does any more (the flag is a frame
  old). Host reads of the tables: the error path only."""
  failing = []
  for r in ranges:
    v = host_copy(r.values)
    bad = ~((v >= r.low) & (v <= r.high))
    if bad.any():
      world, _ = np.argwhere(bad)[0]
      first = v[bad][0]
      failing.append(f"{r.field} in [{_bound(r.low)}, {_bound(r.high)}]: world {world} has {first:g} ({bad.sum()} values)")
  if failing:
    return "; ".join(failing)
  listed = "; ".join(f"{r.field} in [{_bound(r.low)}, {_bound(r.high)}]" for r in ranges)
  return f"none out of range any more, checked: {listed}"


def _raise_errors(flags: Flags, w: Watched, ranges: list[Range]) -> None:
  """Raises for the first invalid-data flag of the word (out-of-range ids, exposure, color,
  per-world values)."""
  errors = (
    (
      flags.env_id,
      f"env.id must be in [0, {w.nenv}) (one per environment map); out-of-range values render with the nearest map",
    ),
    (flags.mesh_id, f"instance.mesh ids must be in range (meshes: [0, {w.nmesh}), height fields: [0, {w.nhfield}))"),
    (
      flags.material_id,
      f"instance.material ids (MuJoCo: m.geom_matid) must be -1 or in [0, {w.nmaterial}); out-of-range ids render "
      "with the last material",
    ),
    (
      flags.texture_id,
      f"material.texture ids (MuJoCo: m.mat_texid) must be -1 or in [0, {w.ntexture}); out-of-range ids render "
      "with the last texture",
    ),
    (flags.exposure, "a per-world exposure must be finite and > 0; invalid values render with exposure 1"),
    (
      flags.color,
      "instance.color / material.base_color must be finite (MuJoCo: m.geom_rgba / m.mat_rgba); invalid colors "
      "render black meanwhile",
    ),
    (
      flags.derived,
      f"per-world values must be numbers in range ({_out_of_range(ranges) if flags.derived else ''}); invalid derived "
      "OpenPBR inputs use the scene's value meanwhile, the other tables render as written (NaN / negative radiance)",
    ),
    (flags.rays, "camera rays must be finite: rays with NaN / inf components render as misses (no ray) meanwhile"),
    (
      flags.camera,
      "camera poses must be finite with a non-zero rotation quaternion (Renderer.camera_pose, the scene cameras' "
      "camera.pos / camera.rot): a NaN position or a zero / NaN quaternion renders the view black",
    ),
    (
      flags.world_ids,
      "render(active=...) ids must be distinct world ids in [0, nworld) or -1 (a hole); an invalid or repeated id "
      "is not rendered meanwhile",
    ),
  )
  for on, message in errors:
    if on:
      raise ValueError(message)


@final
class Watch[S]:
  """The per-frame validation of one Renderer (see the module docstring).

  Attributes:
    validator: the kernel, word and snapshot (`validator.Validator`).
    built: the shared scenery was snapshot (the first frame's end).
  """

  def __init__(self, scene: GpuScene[S], layout: Layout, names: list[str], what: Watched) -> None:
    """Nothing snapshot yet."""
    self.scene = scene
    self.layout = layout
    self.names = names
    self.validator = Validator(what, tuple(scene.binding.feature_checks()), device=scene.device)
    self.built = False
    # A light became a shadow caster the Features were resolved without (`Word.CASTERS`): the
    # renderer re-resolves them, then clears it.
    self.new_casters = False
    self._frames = 0  # eager frames since the last check, modulo `end`'s `every`

  def source(self, sim: S | None) -> RowsSource | TransformSource | None:
    """Where the shared instances' poses come from this call (`Binding.shared_motion`)."""
    ids = self.layout.shared_ids
    return self.scene.binding.shared_motion(sim, ids) if len(ids) else None

  def poll(self, sim: S | None, frame: Callable[[], Frame], features: Features) -> None:
    """Acts on an earlier frame's word (pinned memory, no synchronisation); `frame()` gives the
    call's frame (resolved only when the word reports a moved shared instance, to name it).

    Raises:
      ValueError: for out-of-range env / mesh / material / texture ids, an invalid per-world
        exposure, color or value, non-finite camera rays or poses, invalid active world ids, live
        data needing a compiled-out feature, or shared scenery that moved or changed since the
        first frame (shared scenery is static).
    """
    v = self.validator
    flags = v.poll()
    if not flags.any():
      return
    self.new_casters |= flags.casters
    _raise_errors(flags, v.watched, v.ranges)
    if flags.compiled_out:
      raise ValueError("; ".join(c.explain(features) for c in v.checks))
    if flags.moved and v.snapshot is not None:
      self._raise_moved(_moved_shared(v.snapshot, self.source(sim), frame()))

  def _raise_moved(self, moved: Ids[int]) -> None:
    """Raises the error of shared scenery that moved or changed (`moved`: the instances still away
    from the snapshot; empty when they moved back since the checked frame)."""
    names = [self.names[int(i)] for i in moved]
    listed = ", ".join(repr(n) for n in names[:_LISTED]) + (", ..." if len(names) > _LISTED else "")
    what = f"shared scenery {listed}" if names else "shared scenery (back in place by now)"
    raise ValueError(
      f"{what} moved or changed after the first frame (e.g. a mocap body posed alike in every world when the "
      "scene was built, moved later): shared scenery is static, stored and traced once for every world. Declare "
      "instances that move or change per world at construction "
      f"(Renderer(..., per_world=rw.PerWorld(pose=[{pose_hint(names, self.names) if names else '...'}]), or "
      "size= / mesh=)"
    )

  def end(self, sim: S | None, frame: Frame, features: Features, *, every: int, capturing: bool) -> None:
    """After the frame's work: the first frame snapshots the shared scenery, every `every`-th eager
    frame (and a captured frame: the graph checks every replay) checks."""
    if not self.built and self.layout.nshared:
      self.built = True
      self.validator.snap(self.source(sim), frame)
    due = capturing or self._frames == 0
    self._frames = (self._frames + 1) % every
    if not due:
      return
    self.validator.check(self.source(sim), frame, features=features, instance_type=self.scene.instance_type)


def _moved_shared(snapshot: Snapshot, source: RowsSource | TransformSource | None, frame: Frame) -> Ids[int]:
  """The shared instances whose pose, size or mesh differs from the snapshot in some world (host
  reads: only after a detection)."""
  moved: list[int] = []
  if snapshot.npose and source is not None:
    ids = host_copy(snapshot.pose_ids)[: snapshot.npose]
    inst = snapshot.pose_instances
    if isinstance(source, RowsSource):
      p, r = host_copy(source.pos)[:, ids], host_copy(source.rot)[:, ids]
      away = np.any(p != host_copy(snapshot.pos)[None, : len(ids)], axis=(0, 2))
      away |= np.any(r != host_copy(snapshot.rot)[None, : len(ids)], axis=(0, 2, 3))
      moved += inst[away].tolist()
    else:
      moved += inst[np.any(host_copy(source.transforms)[ids] != host_copy(snapshot.transforms)[: len(ids)], axis=1)].tolist()
  if snapshot.nshape:
    inst = snapshot.shape_instances
    size, mesh = host_copy(frame.instance_size)[:, inst], host_copy(frame.instance_mesh)[:, inst]
    away = np.any(size != host_copy(snapshot.size)[None, : len(inst)], axis=(0, 2))
    away |= np.any(mesh != host_copy(snapshot.mesh)[None, : len(inst)], axis=0)
    moved += inst[away].tolist()
  return np.unique(np.asarray(moved, np.int32))
