"""Bind: the `NewtonBinding` of a host scene of a Newton model's shapes (`binding`), and the
adapter's entry `scene(model)` (`build.load`, `binding` and `rw.upload`).

`binding(scene, model)` rebuilds the binding from the instances' Newton shape ids
(`Instances.source`), so a cached scene binds to a live model; a `load_usd` scene binds through its
stage (`from_usd.stage_parts`: its rigid-body prims matched to the model's bodies).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp

from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
from raywarp.core.pipeline.upload.build import upload
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.integrations.newton.binding import BindingTables, NewtonBinding, NewtonState, WorldColors
from raywarp.integrations.newton.build import (
  USD_META,
  WORLD_COLORS_META,
  Stage,
  absent,
  ingest,
  report_world_appearance,
  shape_rows,
  shape_tables,
  source_rows,
)
from raywarp.integrations.newton.deformables import particle_meshes
from raywarp.integrations.newton.from_usd import stage_parts
from raywarp.integrations.newton.types import DEFAULT_OPTIONS, Options

if TYPE_CHECKING:
  from raywarp.scene.lighting.rig import Lighting
  from raywarp.scene.model.scene import Scene


def binding(  # noqa: PLR0913  # the scene, the model and keyword-only options
  scene: Scene,
  model: newton.Model,
  *,
  adapter: Options = DEFAULT_OPTIONS,
  stage: Stage | None = None,
  body_paths: Sequence[str] | None = None,
  world_local_roots: Sequence[str] | None = None,
  world_homes: Sequence[int] | None = None,
  device: wp.DeviceLike = None,
) -> NewtonBinding:
  """Bind: the `NewtonBinding` of a host scene of `load` / `load_usd`, rebuilt from each model
  shape's Newton shape id (`Instances.source`; e.g. a `.rayw`-cached scene) and, for a `load_usd`
  scene, the stage's rigid-body prims matched to the model's bodies.

  Args:
    scene: the host scene.
    model: the finalized Newton model.
    adapter: the options the scene was loaded with (`merge_worlds`, `shapes`: world w's copies are
      found among them).
    stage: the stage of a `load_usd` scene (path or Usd.Stage).
    body_paths: canonical path per Newton body, as passed to `load_usd`.
    world_local_roots: per-world prototype root paths, as passed to `load_usd`.
    world_homes: the root whose world-local content each world shows, as passed to `load_usd`.
    device: Warp device (default: the model's).

  Raises:
    ValueError: for a `load_usd` scene without its stage, instances without a Newton shape
      outside the stage's, or world shapes that stand for no per-world column (`source_rows`).
  """
  merge_worlds, shapes = adapter.merge_worlds, adapter.shapes
  parts: list[BindingTables] = []
  start = 0
  if USD_META in scene.meta:
    if stage is None:
      raise ValueError("binding: a load_usd scene binds through its stage (stage=)")
    scope = {"body_paths": body_paths, "world_local_roots": world_local_roots, "world_homes": world_homes}
    usd, start = stage_parts(scene, model, stage, **scope)
    parts.append(usd)
  sources = np.asarray(scene.instances.source, np.int32)[start:]
  if np.any(sources < 0):
    raise ValueError("binding: instances without a Newton shape (Instances.source -1): USD visuals come from load_usd")
  hidden = np.zeros(0, np.int32)  # pooled instances some worlds do not show
  if len(sources) or not parts:
    outside = USD_META in scene.meta  # the shapes outside a load_usd stage
    rows = source_rows(model, sources, merge_worlds=merge_worlds, selected=shapes, non_usd=outside)
    parts.append(shape_tables(model, rows))
    hidden = (start + np.flatnonzero(absent(rows))).astype(np.int32)
  # Render world v shows Newton world v, or every world with `merge_worlds`.
  nsource = 1 if merge_worlds else max(model.world_count, 1)
  device = model.device if device is None else device
  particles = particle_meshes(model, scene.report, merge_worlds=merge_worlds) if adapter.particle_meshes else None
  tables = BindingTables.concat(parts)
  colors = _world_colors(scene, model, hidden, tinted=adapter.world_colors)
  return NewtonBinding(
    tables, nsource, device, model.shape_transform, share=not merge_worlds, particles=particles, colors=colors
  )


def _world_colors(
  scene: Scene, model: newton.Model, hidden: np.ndarray[[int], np.dtype[np.int32]], *, tinted: bool
) -> WorldColors | None:
  """The instances colored per world: `hidden` (the pooled ones some worlds do not show) and, with
  `tinted` (`Options.world_colors`), those whose shape color differs between worlds (None:
  neither)."""
  meta = scene.meta.get(WORLD_COLORS_META, []) if tinted and model.shape_color is not None else []
  if not isinstance(meta, list):
    raise TypeError(f"binding: meta[{WORLD_COLORS_META!r}] must be a list of instance indices, got {meta!r}")
  tint_ids = np.asarray(meta, np.int32)
  ids = np.union1d(hidden, tint_ids).astype(np.int32)
  if not len(ids):
    return None
  tint = np.isin(ids, tint_ids)
  base = np.asarray(scene.instances.color, np.float32).reshape(-1, 4)
  return WorldColors(model.shape_color if tint.any() else None, ids, tint, base)


def scene(
  model: newton.Model,
  *,
  options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  adapter: Options = DEFAULT_OPTIONS,
  lighting: Lighting | None = None,
  device: wp.DeviceLike = None,
) -> GpuScene[NewtonState]:
  """The GpuScene of a Newton model's shapes on `device` (default: the model's), bound to its
  bodies (`Renderer.render(state)`): `load`, `binding` and `rw.upload` with `options`."""
  rows = shape_rows(model, adapter.shapes, merge_worlds=adapter.merge_worlds)
  host = ingest(model, rows, options=options, adapter=adapter, lighting=lighting)
  # Without per-world colors, the scene shows world 0's in every world (Isaac Lab, which also
  # loads, writes its own per-world material tables).
  report_world_appearance(model, rows, host.report, colors=not adapter.world_colors)
  dev = model.device if device is None else device
  return upload(host, dev, binding=binding(host, model, adapter=adapter, device=dev), options=options)
