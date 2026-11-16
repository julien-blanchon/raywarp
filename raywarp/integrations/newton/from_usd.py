"""Ingest from the USD stage a Newton model was loaded from (`load_usd`; `[usd]` extra), and
the stage part of its binding (`stage_parts`, read by `api.binding`).

The stage's visuals, materials and lights come through `raywarp.formats.usd`; an instance under
a rigid body follows the Newton body labelled with that body's prim path (or ending with "/" +
it: prefixed replicas), others are static. USD is imported on the first call only.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, cast

import newton
import numpy as np

from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.formats.usd import api as usd_api
from raywarp.formats.usd import options as usd_options
from raywarp.formats.usd.prim_paths import under
from raywarp.integrations.newton import xform32 as xf
from raywarp.integrations.newton.binding import BindingTables, PoseTable
from raywarp.integrations.newton.build import USD_META, Stage, model_shapes, shape_rows
from raywarp.integrations.newton.deformables import report_particles
from raywarp.integrations.newton.types import DEFAULT_OPTIONS, Options
from raywarp.scene.build.builder import SceneBuilder
from raywarp.scene.lighting.rig import Lighting
from raywarp.scene.model import xform
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.scene import Instances, Scene
from raywarp.scene.model.types import Json

if TYPE_CHECKING:
  from pxr import Usd


def _opened(stage: Stage) -> Usd.Stage:
  """`stage` itself, or the stage opened from its path."""
  from pxr import Usd  # noqa: PLC0415  # pxr (the usd extra) on the first stage read only

  from raywarp.formats.usd.util import open_stage  # noqa: PLC0415  # imports pxr (the usd extra)

  return stage if isinstance(stage, Usd.Stage) else open_stage(stage)


def _rigid_poses(stage: Stage, paths: list[str]) -> np.ndarray[[int, 7], np.dtype[np.float64]]:
  """(n, 7) rigid world pose of each prim of `paths` (scale removed)."""
  from raywarp.formats.usd.stage.reader import prim_transforms  # noqa: PLC0415  # imports pxr (the usd extra)

  return xform.rigid(np.stack([m[:3] for m in prim_transforms(stage, paths)]).astype(np.float64))


def stage_has_lighting(stage: Stage) -> bool:
  """Whether the stage authors any UsdLux light (a DomeLight included), without reading it."""
  from pxr import UsdLux  # noqa: PLC0415  # pxr (the usd extra) on the first stage read only

  return any(prim.HasAPI(UsdLux.LightAPI) for prim in _opened(stage).Traverse())


def _body_world(model: newton.Model) -> np.ndarray[[int], np.dtype[np.int32]]:
  """(body_count,) Newton world of each body (-1: global)."""
  if not model.body_count or model.body_world is None:
    return np.zeros(0, np.int32)
  return cast("np.ndarray[[int], np.dtype[np.int32]]", host_copy(model.body_world))  # Newton's arrays are unshaped


def _suffixes(labels: Sequence[str]) -> dict[str, list[int]]:
  """Every "/"-separated tail of each label -> the bodies so labelled ("a/b/c" -> "b/c", "c")."""
  out: dict[str, list[int]] = {}
  for b, label in enumerate(labels):
    for i, c in enumerate(label):
      if c == "/":
        out.setdefault(label[i + 1 :], []).append(b)
  return out


def _matches(model: newton.Model, bodies: Sequence[str], body_paths: Sequence[str] | None = None) -> dict[int, dict[int, int]]:
  """Host body string id -> {Newton world: body} of the bodies labelled with its path.

  Exact label first, else labels ending with "/" + the path (prefixed replicas).

  Raises:
    ValueError: for a path that names several bodies of one world (an ambiguous label or
      suffix: pass `body_paths` to say which body is which).
  """
  if body_paths is not None and len(body_paths) != model.body_count:
    raise ValueError(f"body_paths needs {model.body_count} paths, got {len(body_paths)}")
  labels = list(body_paths) if body_paths is not None else list(model.body_label) if model.body_count else []
  body_world = _body_world(model)
  by_label: dict[str, list[int]] = {}
  for b, label in enumerate(labels):
    by_label.setdefault(label, []).append(b)
  by_suffix: dict[str, list[int]] | None = None  # built on the first miss
  out: dict[int, dict[int, int]] = {}
  for k, path in enumerate(bodies):
    found = by_label.get(path, [])
    if not found:
      by_suffix = by_suffix if by_suffix is not None else _suffixes(labels)
      found = by_suffix.get(path.lstrip("/"), [])
    per_world: dict[int, int] = {}
    for b, w in zip(found, body_world[found].tolist(), strict=True):
      if w in per_world:
        names = [labels[per_world[w]], labels[b]]
        raise ValueError(f"USD rigid body {path!r} matches several Newton bodies of world {w}: {names} (pass body_paths)")
      per_world[w] = b
    out[k] = per_world
  return out


# Two import transforms agree within these tolerances: positions [m] (float32 poses of scenes of
# up to ~1 km keep ~1e-4 m), and 1 - |q . q'| (1e-8: ~2e-4 rad, 1e-4 m at a 0.5 m lever arm).
_SAME_POSITION: Final = 1e-4
_SAME_ROTATION: Final = 1e-8


def _consensus(candidates: Sequence[xf.Pose]) -> tuple[xf.Pose, int, bool]:
  """(the pose most `candidates` agree on, how many disagree with it, whether another group is as
  large: a tie, decided for the first body)."""
  c = np.asarray(candidates, np.float64)  # (n, 7): position, quaternion
  near = np.linalg.norm(c[:, None, :3] - c[None, :, :3], axis=2) <= _SAME_POSITION
  aligned = 1.0 - np.abs(c[:, 3:] @ c[:, 3:].T) <= _SAME_ROTATION
  votes: list[int] = np.sum(near & aligned, axis=1).tolist()
  best = votes.index(max(votes))  # the first of the largest agreeing group
  tie = any(v == votes[best] and not (near[best, i] and aligned[best, i]) for i, v in enumerate(votes))
  return candidates[best], len(candidates) - votes[best], tie


def _import_transforms(
  model: newton.Model, stage: Stage, bodies: Sequence[str], matches: dict[int, dict[int, int]], report: LoadReport | None
) -> tuple[dict[int, xf.Pose], dict[int, xf.Pose]]:
  """(stage pose of each matched host body, Newton's import transform per world).

  Every matched body b of world w implies an import transform T = body_q0[b] P_body^-1 (e.g.
  `add_usd(xform=...)` or a replicate offset). A body whose initial pose was changed after the
  import (randomised, posed by joint coordinates) implies another one: T_w is the transform most
  of world w's bodies agree on, and the bodies that disagree are counted in `report`.
  """
  matched = [k for k, m in matches.items() if m]
  if not matched:
    return {}, {}
  body_q = host_copy(model.body_q).astype(np.float64) if model.body_q is not None else np.zeros((0, 7))
  rigid = _rigid_poses(stage, [bodies[k] for k in matched])
  stage_pose = {k: rigid[i] for i, k in enumerate(matched)}
  candidates: dict[int, list[xf.Pose]] = {}
  for k in matched:
    for w, b in matches[k].items():
      candidates.setdefault(w, []).append(xf.compose(body_q[b], xf.inverse(stage_pose[k])))
  import_tf: dict[int, xf.Pose] = {}
  for w, poses_w in candidates.items():
    import_tf[w], disagree, tie = _consensus(poses_w)
    if disagree and report is not None:
      report.approx("newton.import_transform(bodies_moved_after_import)", f"world {w}", n=disagree)
    if tie and report is not None:
      report.approx("newton.import_transform(tie: first body's)", f"world {w}")
  return stage_pose, import_tf


def _homes(world_local_roots: Sequence[str] | None, world_homes: Sequence[int] | None, nworld: int) -> tuple[int, ...]:
  """(nworld,) the root whose world-local content each world shows (default: every world root 0's).

  Raises:
    ValueError: for homes without roots, of another length than the worlds, out of range, or a home
      that does not show its own root.
  """
  if world_homes is None:
    return (0,) * nworld
  homes = tuple(world_homes)
  if world_local_roots is None:
    raise ValueError("world_homes needs world_local_roots")
  if len(homes) != nworld or any(not 0 <= h < nworld for h in homes) or any(homes[h] != h for h in homes):
    raise ValueError(f"world_homes needs {nworld} root indices, each showing its own root (world_homes[h] == h)")
  return homes


def _placements(
  model: newton.Model,
  stage: Stage,
  imported: dict[int, xf.Pose],
  world_local_roots: Sequence[str] | None,
  homes: Sequence[int],
) -> dict[int, list[xf.Pose]]:
  """Per home root h: the transform placing its world-local content in each world (T_w T_h^-1 of
  the root frames T; without roots, Newton's import transform of each world for root 0)."""
  nworld = max(model.world_count, 1)
  if world_local_roots is None:
    return {0: [imported.get(w, imported.get(-1, xf.IDENTITY)) for w in range(nworld)]}
  if len(world_local_roots) != nworld:
    raise ValueError(f"world_local_roots needs {nworld} paths, got {len(world_local_roots)}")
  opened = _opened(stage)
  roots = [root.rstrip("/") or "/" for root in world_local_roots]
  if any(not root.startswith("/") or not opened.GetPrimAtPath(root) for root in roots):
    raise ValueError("world_local_roots must name existing absolute prim paths")
  poses = list(_rigid_poses(opened, roots))
  # Explicit clone roots describe placement independently of initial articulated/body DR.
  inverses = {h: xf.inverse(poses[h]) for h in sorted(set(homes))}
  return {h: [xf.compose(pose, inverse) for pose in poses] for h, inverse in inverses.items()}


def _home(path: str, roots: Sequence[str] | None, home_set: Sequence[int]) -> int:
  """The home root (of the sorted distinct homes `home_set`) a prim's world-local content belongs
  to (component boundary, not prefix); -1: global content (placed once)."""
  if roots is None:
    return 0  # Existing Newton imports place the entire stage in each world.
  return next((h for h in home_set if under(path, roots[h])), -1)


def static_light_poses(  # noqa: PLR0913  # source scene, model and explicit clone scope
  model: newton.Model,
  host: Scene,
  stage: Stage,
  *,
  body_paths: Sequence[str] | None = None,
  world_local_roots: Sequence[str] | None = None,
  world_homes: Sequence[int] | None = None,
) -> tuple[PoseTable, np.ndarray[[int], np.dtype[np.bool_]]]:
  """World poses and active mask of authored USD lights using the instance scope rules.

  A body-attached light needs a live pose binding, which NewtonBinding does not yet provide.
  Such lights are reported and disabled instead of emitting from a stale initial pose.
  """
  bodies = host.instances.bodies.tolist()
  matches = _matches(model, bodies, body_paths)
  _, imported = _import_transforms(model, stage, bodies, matches, None)  # reported by the ingest
  homes = _homes(world_local_roots, world_homes, max(model.world_count, 1))
  placements = _placements(model, stage, imported, world_local_roots, homes)
  home_set = sorted(placements)
  poses = np.zeros((len(homes), host.nlight, 7), np.float32)
  active = np.asarray(host.lights.body < 0, bool)
  rigid = xform.rigid(np.asarray(host.lights.xform, np.float64).reshape(-1, 3, 4))
  for i, path in enumerate(host.lights.paths):
    if not active[i]:
      host.report.drop("newton.light.body(unsupported)", path)
    pose = rigid[i]
    home = _home(path, world_local_roots, home_set)
    for w in range(len(homes)):
      poses[w, i] = xf.compose(placements[home][w], pose) if home >= 0 else pose
  return poses, active


def usd_tables(  # noqa: PLR0913  # source scene, model and explicit clone scope
  model: newton.Model,
  host: Scene,
  n: int,
  stage: Stage,
  report: LoadReport | None = None,
  *,
  body_paths: Sequence[str] | None = None,
  world_local_roots: Sequence[str] | None = None,
  world_homes: Sequence[int] | None = None,
) -> BindingTables:
  """Binding tables of the first `n` instances of a USD host scene (instances on unmatched bodies
  counted in `report`: the ingest's, not the binding's).

  An instance under a matched rigid body follows it, one row per world: local = P_body^-1
  P_instance with the stage poses P. Newton's import transform of world w also places the static
  instances of that world (identity without matched bodies), which then belong to world w. One
  row unless a body matches in several worlds (or `world_local_roots` scopes the clones). With
  `world_homes`, the static content under root h is placed in world w by T_w T_h^-1 of the root
  frames T (`load_usd`).
  """
  inst = host.instances
  bodies = inst.bodies.tolist()
  matches = _matches(model, bodies, body_paths)
  # With explicit roots the import transforms place nothing (the roots do): not reported (bodies of
  # different home roots imply different ones).
  stage_pose, import_tf = _import_transforms(model, stage, bodies, matches, report if world_local_roots is None else None)
  rows = max(model.world_count, 1) if world_local_roots is not None or any(len(m) > 1 for m in matches.values()) else 1
  homes = _homes(world_local_roots, world_homes, max(model.world_count, 1))
  # Per home root, one placement per row (one row: world 0's).
  placements = _placements(model, stage, import_tf, world_local_roots, homes)
  paths = inst.paths.tolist()
  poses = xform.rigid(np.asarray(inst.xform[:n], np.float64))
  home_set = sorted(placements)
  home = np.array([_home(paths[int(p)], world_local_roots, home_set) for p in inst.prim[:n]], np.int64)
  # Static placement of every (world, instance): world-local ones by their home's placement.
  local = np.repeat(poses[None].astype(np.float32), rows, axis=0)
  for h, placed in placements.items():
    ids = np.flatnonzero(home == h)
    if len(ids):
      local[:, ids] = xf.compose_many(np.asarray(placed, np.float32)[:rows, None], poses[None, ids].astype(np.float32))
  out = BindingTables(
    np.full((rows, n), -1, np.int32), local, np.full((rows, n), -1, np.int32), np.full((rows, n), -1, np.int32)
  )
  if rows > 1:
    # Row w's world-local static instances are world w's copies (Newton's per-world offsets).
    out.world[:, home >= 0] = np.arange(rows).astype(np.int32)[:, None]
  unbound = _bind_bodies(out, inst.body[:n].tolist(), matches, stage_pose, poses, _body_world(model))
  if unbound and report is not None:
    report.drop("newton.body_not_found(static)", n=unbound)
  return out


def _bind_bodies(  # noqa: PLR0913  # the tables written and the matched bodies' poses
  out: BindingTables,
  host_bodies: list[int],
  matches: dict[int, dict[int, int]],
  stage_pose: dict[int, xf.Pose],
  poses: np.ndarray[[int, 7], np.dtype[np.float64]],
  body_world: np.ndarray[[int], np.dtype[np.int32]],
) -> int:
  """Binds each instance i on a matched host body `host_bodies[i]` to its Newton body in every
  row of `out` (local = P_body^-1 P_instance); the count of instances on unmatched bodies."""
  bound: dict[int, np.ndarray[[int], np.dtype[np.int32]]] = {}  # host body -> Newton body per world
  unbound = 0
  for i, k in enumerate(host_bodies):
    match = matches.get(k, {}) if k >= 0 else {}
    unbound += int(k >= 0 and not match)
    if not match:
      continue
    if k not in bound:
      bound[k] = _world_bodies(match, out.rows)
    b = bound[k]
    on = b >= 0  # worlds where a body carries the instance (else its static placement stays)
    out.body[on, i], out.world[on, i] = b[on], body_world[b[on]]
    out.local[on, i] = xf.compose(xf.inverse(stage_pose[k]), poses[i])
  return unbound


def _world_bodies(match: dict[int, int], rows: int) -> np.ndarray[[int], np.dtype[np.int32]]:
  """(rows,) Newton body carrying a host body in each world (its world's match, else a global
  one; -1: none). One row (no replication): the first matched world's."""
  b = np.array([match.get(w, match.get(-1, -1)) for w in range(rows)], np.int32)
  if rows == 1 and b[0] < 0:
    b[0] = match[min(match)]
  return b


def _replicated(host: Scene, n: int, copies: int) -> Scene:
  """`host` with its first `n` instances repeated `copies` times (in front of the others)."""
  arrays = host.instances.arrays()
  for name in Instances.SCHEMA:
    column = arrays[name]
    arrays[name] = np.concatenate([*([column[:n]] * copies), column[n:]])
  return dataclasses.replace(host, instances=Instances.from_arrays(arrays))


def load_usd(  # noqa: PLR0913  # the model, its stage and keyword-only options
  model: newton.Model,
  stage: Stage,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  usd: usd_options.Options = usd_options.DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
  cache: str | os.PathLike[str] | bool | None = True,  # a location, True (the cache directory) or None
  body_paths: Sequence[str] | None = None,
  world_local_roots: Sequence[str] | None = None,
  world_homes: Sequence[int] | None = None,
) -> Scene:
  """Ingest: host `Scene` of the stage a Newton model was loaded from; bind it with
  `binding(scene, model, adapter=adapter, stage=stage)` (the stage's rigid bodies matched by prim
  path) and upload it with `options`.

  The stage's visuals, materials, lights and cameras, plus the model's shapes not imported from a
  stage (labels not starting with "/", e.g. a ground plane added in code). The stage is read
  through the `.rayw` cache when no such shape is added to it. `meta[USD_META]` records the
  stage's instance count and copies for the binding.

  Args:
    model: the finalized Newton model.
    stage: the stage (path or Usd.Stage).
    options: the options every adapter shares (`rw.SceneOptions`; its texture size cap applies
      at decode).
    adapter: the Newton options: `merge_worlds` (one render world holding every Newton world, the
      stage's visuals once per world when Newton replicated them; else render world v shows
      Newton world v) and `shapes` (the model's shapes to add, default the visible, non-site
      ones, among those not imported from a stage).
    usd: the USD options of the stage read (`raywarp.formats.usd.Options`).
    lighting: lights, ambient, environment and headlight added to the stage's.
    cache: `.rayw` cache of the USD host scene (see `rw.formats.usd.load`; on by default).
    body_paths: optional canonical USD path per Newton body (same order as body_label).
    world_local_roots: one root prim per world. The first is the ingested prototype subtree;
      its static content follows world placement while content outside stays global. Root
      transforms define static placement independently of initial body poses. None preserves
      whole-stage imports. Pass the same explicit paths and roots to `binding`; light pose
      tables are prepared separately with `static_light_poses`.
    world_homes: per world, the root whose world-local content it shows (default every world root
      0's): for worlds of different prototypes, each prototype's content lies under its first
      world's root h (world_homes[h] == h) and is placed in world w by T_w T_h^-1 of the root
      frames T; content under other roots is global. Which worlds show a home's content is the
      caller's (a per-world visibility: `raywarp.integrations.isaaclab.usd_source`).
  """
  merge_worlds = adapter.merge_worlds
  if merge_worlds and world_homes is not None and len(set(world_homes)) > 1:
    raise ValueError(
      "load_usd: merge_worlds shows every world's copy at once: worlds of different homes need per-world rendering"
    )
  rows = shape_rows(model, adapter.shapes, merge_worlds=merge_worlds, non_usd=True)
  if rows.shape[1] > 0:
    # Model shapes added to the stage's scene: read it into a builder (no cache).
    builder = usd_api.read(stage, SceneBuilder(), options=options, adapter=usd)
    nusd = builder.ninstance
    model_shapes(model, rows, builder, adapter=adapter, max_texture_size=options.max_texture_size)
    host = builder.build()
  else:
    host = usd_api.load(stage, options=options, adapter=usd, cache=cache)
    nusd = host.ninstance
  if adapter.particle_meshes:
    report_particles(model, host.report)
  if lighting is not None:
    # Display-relative lights placed from the stage's and the model's instances (one copy); an
    # envmap's image is appended after the scene's texels, not copied (`Scene.with_lighting`).
    host = host.with_lighting(lighting)
  scope = {"body_paths": body_paths, "world_local_roots": world_local_roots, "world_homes": world_homes}
  tables = usd_tables(model, host, nusd, stage, host.report, **scope)
  copies = tables.rows if merge_worlds and tables.rows > 1 else 1
  if copies > 1:
    # One render world: the stage's visuals once per Newton world, each bound to its bodies.
    host = _replicated(host, nusd, copies)
  record: dict[str, Json] = {"instances": nusd, "copies": copies}
  meta: dict[str, Json] = {**host.meta, USD_META: record}
  return dataclasses.replace(host, meta=meta)


def stage_parts(  # noqa: PLR0913  # the scene, the model, its stage and the clone scope
  scene: Scene,
  model: newton.Model,
  stage: Stage,
  *,
  body_paths: Sequence[str] | None = None,
  world_local_roots: Sequence[str] | None = None,
  world_homes: Sequence[int] | None = None,
) -> tuple[BindingTables, int]:
  """(binding tables of a `load_usd` scene's stage instances, their count with copies)."""
  info = scene.meta[USD_META]
  nusd = info.get("instances") if isinstance(info, dict) else None
  copies = info.get("copies") if isinstance(info, dict) else None
  if not isinstance(nusd, int) or not isinstance(copies, int):
    raise TypeError(f"binding: meta[{USD_META!r}] must be {{'instances': int, 'copies': int}}, got {info!r}")
  scope = {"body_paths": body_paths, "world_local_roots": world_local_roots, "world_homes": world_homes}
  usd = usd_tables(model, scene, nusd, stage, **scope)
  return (usd.merged() if copies > 1 else usd), nusd * copies
