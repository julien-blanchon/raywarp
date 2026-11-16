"""The disk-cache keys of the bakes (`baked.Baked`: light grid, GI volume) and the
keys deciding when a bake is redone (`grid_key`, `probe_key`): what each bake reads, world 0's rows
only (`world0`), the geometry behind the BVH handles (`shapes`), never a BVH (the lookup comes
before any build)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

import numpy as np
import warp as wp

from raywarp.core.kernels import probe
from raywarp.core.kernels.abi.features import Features, software_view
from raywarp.core.pipeline.frame import Frame
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.shade.gi import cube as gi_cube
from raywarp.core.trace import frame_structs
from raywarp.settings.fields import exposure_from_ev100

if TYPE_CHECKING:
  from collections.abc import Set

  from raywarp._base.dims import AnyDeviceArray, AnyHostArray
  from raywarp.core.kernels.abi import rays
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.shade import gi as gi_lib
  from raywarp.core.trace.context import FrameContext


# Struct fields holding per-world (1 or nworld, ...) rows: every `Frame` field but the shared
# instances' poses (the struct builders keep the Frame's names), and the rows derived from them:
# the ray queries' instance frames, the packed lights, the environment's `Frame.ambient`. The
# bakes trace world 0 only, so their keys hash row 0 (`world0`): a bake does not depend on the
# world count (tests/unit/test_bake_keys.py checks every name is a field of a hashed struct).
_SHARED_FIELDS: Final = frozenset({"shared_pos", "shared_rot"})
_DERIVED_ROWS: Final = frozenset({"instance_rot_trace", "lights", "ambient_radiance"})
WORLD_ROWS: Final = frozenset(f.name for f in dataclasses.fields(Frame)) - _SHARED_FIELDS | _DERIVED_ROWS
# Struct fields that are not inputs of a bake, left out of the keys: BVH root nodes and mesh BVH
# handles (derived from hashed arrays; world 0's root index depends on the world count, the
# handles exist only once built) and the light-cost counters (an output: their contents change
# every frame), the lights' hot records (derived from `lights`), and the user's ambient fill
# (`Params.ambient`: added at camera and reflection hits only, the probes do not see it), and the
# build word copies (`bits`, `bits2`: the serving build's, whose probe-relevant part the host key
# holds, `probe_features`: a tier build and the settings' own bake alike).
UNHASHED: Final = frozenset(
  {"group_root", "flex_group_root", "counters", "hot", "mesh_bvh_id", "nmesh", "ambient", "bits", "bits2"}
)


def world0[S](s: S) -> S:
  """A copy of kernel struct `s` (nested structs too) for bake keys: its WORLD_ROWS arrays cut to
  row 0 (views), its UNHASHED arrays empty."""
  out = type(s)()
  for name, value in bake_cache.fields(s).items():
    if wp.types.is_struct(value):
      setattr(out, name, world0(value))
    elif isinstance(value, wp.array):
      setattr(out, name, _key_rows(name, value))
    elif name not in UNHASHED:
      setattr(out, name, value)
  return out


def _key_rows(name: str, value: AnyDeviceArray) -> AnyDeviceArray | None:
  """The rows of struct array `name` a bake key hashes (see `world0`; None: not hashed)."""
  if name in UNHASHED:
    return None
  if name not in WORLD_ROWS or value.shape[0] <= 1:
    return value
  return value[:1] if value.size else wp.empty((1, *value.shape[1:]), dtype=value.dtype, device=value.device)


def shapes(ctx: FrameContext) -> tuple[AnyHostArray | AnyDeviceArray, ...]:
  """The geometry behind the BVH handles of the geometry struct, for bake keys (the struct holds
  the BLAS ids, not the vertices): the meshes' vertices, faces, addresses and frames, the
  height fields' vertices and triangles."""
  m = ctx.scene.meshes
  out = [
    m.vert,  # device arrays: hashed by their registered digests when the upload made them
    m.face,
    m.vertadr,
    m.faceadr,
    np.asarray(m.frames if m.frames is not None else ()),
  ]
  for h in m.hfield_registry.values():
    out += [host_copy(h.points), host_copy(h.indices)]
  return tuple(out)


def grid_key(ctx: FrameContext) -> tuple[object, ...]:
  """What the light grid's visibility test depends on besides the geometry and light poses:
  the Features of its rays and light struct, the active lights, whether the lights' emission
  varies (`FrameContext.emission_varies`: dark lights flooded), the settings tagged
  `rebuild="light_grid"` (the cell size). The area lights are the settings' (`effective`, not
  the serving build's)."""
  f = software_view(ctx.features, ctx.effective)
  return (
    f.shape_types,
    ctx.effective.area_lights,
    f.trees,
    f.alpha_cutout,
    ctx.settings.lighting.emissive_lights,
    ctx.dynamic_lights,
    ctx.emission_varies,
    *ctx.settings.rebuild_inputs("light_grid"),
  )


# The light fields whose writes can light a lamp the grid was baked dark.
EMISSION_FIELDS: Final = frozenset({"light.strength", "light.color"})


def emission_varies(scene: GpuScene, per_world_fields: Set[str]) -> bool:
  """Whether the lights' emission is written after the bake: a live slot of the scene's source
  (a simulator's light intensity) or declared per world (randomised): the grid then floods the
  lights dark at the bake too (`light_grid.build`), so lighting one keeps its world culled. Fixed
  for a renderer (`FrameContext.emission_varies`)."""
  return not EMISSION_FIELDS.isdisjoint((*scene.binding.slots(None), *per_world_fields))


def probe_key(ctx: FrameContext) -> tuple[object, ...]:
  """What the GI volume was baked for besides the scene: the probe-relevant Features of the
  settings (`probe_features`), the settings tagged `rebuild="gi"` and, in a scene of
  display-relative emission, the renderer's exposure (the probes light that emission at it:
  `frame_structs.probe_environment`; physical emission does not depend on it)."""
  relative = () if ctx.scene.capabilities.physical_emission else (ctx.settings.exposure.render_ev100(),)
  return (probe_features(ctx), *ctx.settings.rebuild_inputs("gi"), *relative)


def probe_features(ctx: FrameContext) -> Features:
  """The Features the baked radiance depends on: the settings' own, specialised on the scene
  (`ctx.effective`), without the camera-ray / output / launch fields (`probe.probe_features`).
  Not the build serving them (a tier's or the settings' own, which shade the probe rays alike): a
  camera-only change that moves the settings onto another build keeps the bake."""
  return probe.probe_features(ctx.effective)


def gi_key(
  ctx: FrameContext,
  source: Frame,
  groups: gi_lib.Groups,
  *,
  see_host: np.ndarray[[int], np.dtype[np.int32]],
  hidden: np.ndarray[[int], np.dtype[np.bool_]],
) -> bake_cache.Digest | None:
  """The bake cache key of a GI volume baked from frame `source` over `groups` (`bake_cache.key`):
  every input of the placement and of the bake (world 0's frame before the volume exists;
  `see_host`, `hidden`: the instances the probe rays pass and do not see)."""
  inputs = (
    frame_structs.materials(ctx, source),
    frame_structs.probe_environment(ctx, source),
    frame_structs.lights_struct(ctx, None),
  )
  host = (
    probe_features(ctx),
    ctx.settings.gi,
    cube_layout(),
    exposure_from_ev100(ctx.settings.exposure.render_ev100()),
    ctx.settings.lighting.light_grid_cell,
    see_host,
    hidden,
    host_copy(groups.group_digest),
    *shapes(ctx),
  )
  return bake_cache.key("gi", ctx.scene.device, trace_inputs(ctx, source), *map(world0, inputs), host=host)


def cube_layout() -> tuple[int, int, int]:
  """The ambient cube's subdivision bounds and texel budgets (`shade.gi.cube`): a bake key part (a
  cached cube must have the texel layout the restoring volume allocates)."""
  return (gi_cube.SUBDIV, gi_cube.MAX_SUBDIV, gi_cube.MAX_TEXELS)


def trace_inputs(ctx: FrameContext, frame: Frame) -> rays.OptixFields:
  """World 0's rows of what a bake's geometry rays read besides the meshes (`shapes`): the
  instance frames, sizes, colors and cut-out tables, without any BVH (a bake key; `world0`)."""
  return world0(frame_structs.optix_geometry(ctx, frame))
