"""Opacity micromaps (OMM) of the alpha cut-outs: the any-hit alpha test resolved ahead of time.

A cut-out mesh instance (`trace/accel/cutout.py`) whose alpha test is the same in every world and
frame (`omm_sources.source_of`) gets a GAS of its own (one per `CutSource`: mesh, opacity texture,
cutoff, UVs) whose triangles carry a 4-state opacity micromap (`bake.py`, `kernels.py`). OptiX
then skips the micro-triangles the bake found fully transparent and accepts the fully opaque ones
without calling the any-hit program; only the `UNKNOWN` ones (the leaf edges) still run it, so the
pixels are those of the any-hit test alone.

This module builds the micromap arrays and their GASes and caches the bakes.
"""

from __future__ import annotations

import ctypes
import dataclasses
import weakref
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy, stream_handle
from raywarp.core.trace.omm_sources import CutSource
from raywarp.core.trace.optix import api, gas
from raywarp.core.trace.optix.context import address
from raywarp.core.trace.optix.omm.bake import Bake, bake, face_levels, level_cap

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene
  from raywarp.core.trace.optix.context import DeviceContext


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class Micromaps:
  """An opacity micromap array and the index buffer its GAS build reads (kept alive together with
  the GAS: OptiX reads the array at traversal).

  Attributes:
    array: the built micromap array.
    index: (nface,) micromap index of each face (`Bake.index`).
    usage: (count, level, format) of the micromaps the faces use (none: every face predefined).
  """

  array: wp.array[wp.uint64, [int]]
  index: wp.array[wp.int32, [int]]
  usage: list[tuple[int, int, int]]


def build_array(ctx: DeviceContext, b: Bake, *, device: wp.Device) -> Micromaps:
  """The micromap array of bake `b` on `device` (built on its current stream, synchronised)."""
  histogram = _counts(b.histogram)
  build = api.OpacityMicromapArrayBuildInput(
    flags=api.OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE,
    inputBuffer=address(b.data),
    perMicromapDescBuffer=address(b.desc),
    perMicromapDescStrideInBytes=0,
    numMicromapHistogramEntries=len(histogram),
    micromapHistogramEntries=histogram,
  )
  sizes = ctx.micromap_sizes(build)
  out_size, temp_size = int(sizes.outputSizeInBytes), int(sizes.tempSizeInBytes)
  out, temp = gas.byte_buffer(out_size, device), gas.byte_buffer(temp_size, device)
  on = wp.get_stream(device)
  buffers = api.MicromapBuffers(output=address(out), outputSizeInBytes=out_size, temp=address(temp), tempSizeInBytes=temp_size)
  ctx.micromap_build(stream_handle(on), build, buffers)
  wp.synchronize_stream(on)
  del temp
  return Micromaps(
    array=out,
    index=b.index,
    usage=b.histogram if b.faces[3] else [],
  )


def _counts(entries: list[tuple[int, int, int]]) -> ctypes.Array[api.OpacityMicromapUsageCount]:
  """(count, level, format) entries as OptiX usage counts / histogram entries."""
  return (api.OpacityMicromapUsageCount * len(entries))(*[api.OpacityMicromapUsageCount(*e) for e in entries])


def triangles(points: wp.array[wp.vec3, [int]], indices: wp.array[wp.vec3i, [int]], m: Micromaps) -> api.BuildInput:
  """Triangle build input over `points` and `indices` with micromaps `m` (indexed, int32): its
  any-hit program runs on the UNKNOWN micro-triangles only."""
  bi = gas.triangles(points, indices, api.GEOMETRY_FLAG_NONE)
  omm = bi.triangleArray.opacityMicromap
  omm.indexingMode = api.OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED
  omm.opacityMicromapArray, omm.indexBuffer = address(m.array), address(m.index)
  omm.indexSizeInBytes = 4
  # OptiX rejects a non-null array of zero counts (every face predefined: no usage).
  if m.usage:
    usage = _counts(m.usage)
    omm.numMicromapUsageCounts, omm.micromapUsageCounts = len(usage), usage
  return bi


# A bake's identity: (scene meshes id, opacity texture id, device, source).
type BakeKey = tuple[int, int, str, CutSource]
# Built micromaps by `BakeKey`, with their level cap (only the last cap's are kept), dropped with
# the meshes or the texture: a Renderer rebuilt for new features or world counts reuses its
# scene's bakes, and a texture replaced under the same index is baked again.
_cache: Final[dict[BakeKey, tuple[int, Micromaps]]] = {}
# The ids of the objects whose collection drops their entries (`_forget`), finalizers attached.
_watched: Final[set[int]] = set()


def _forget(identity: int) -> None:
  """Drops the cached micromaps of the meshes or texture whose id was `identity` (collected)."""
  _watched.discard(identity)
  for k in [k for k in _cache if identity in (k[0], k[1])]:
    del _cache[k]


def _watch(obj: object) -> None:
  """Drops `obj`'s cached micromaps once it is collected (one finalizer per object)."""
  if id(obj) not in _watched:
    _watched.add(id(obj))
    weakref.finalize(obj, _forget, id(obj))


def micromaps(
  ctx: DeviceContext,
  scene: GpuScene,
  source: CutSource,
  levels: np.ndarray[[int], np.dtype[np.int64]],
  cap: int,
  *,
  device: wp.Device,
) -> Micromaps:
  """The built micromaps of `source` on `device` (`levels`: its `face_levels`), levels at most
  `cap` (baked once per scene meshes, texture, device and cap)."""
  tex = scene.textures[source.texture]
  key = (id(scene.meshes), id(tex), str(device), source)
  found = _cache.get(key)
  if found is not None and found[0] == cap:
    return found[1]
  _watch(scene.meshes)
  _watch(tex)
  built = build_array(ctx, bake(scene.meshes, source, tex, levels=levels, cap=cap, device=device), device=device)
  _cache[key] = (cap, built)
  return built


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class CutGases:
  """The GASes with micromaps of a build's cut-out sources (`cut_gases`).

  Attributes:
    gases: one compacted GAS per source; `maps` their micromaps (alive as long as the GASes).
    handle: each source's GAS handle.
  """

  gases: list[gas.Accel]
  maps: list[Micromaps]
  handle: dict[CutSource, int]


def cut_gases(
  ctx: DeviceContext,
  scene: GpuScene,
  sources: list[CutSource],
  frames: np.ndarray[[int, 3, 3], np.dtype[np.float32]],
  *,
  device: wp.Device,
) -> CutGases:
  """The GAS with micromaps of each source on `device` (its mesh's vertices in its GAS frame
  `frames[mesh]`, as `gas.mesh_gases`; one level cap for all: `level_cap`)."""
  meshes = scene.meshes
  vertadr, faceadr = host_copy(meshes.vertadr), host_copy(meshes.faceadr)
  levels = [face_levels(meshes, s, scene.textures[s.texture], device=device) for s in sources]
  cap = level_cap(levels)
  maps = [micromaps(ctx, scene, s, lv, cap, device=device) for s, lv in zip(sources, levels, strict=True)]
  inputs: list[tuple[api.BuildInput, gas.InputBuffers]] = []
  for s, m in zip(sources, maps, strict=True):
    _, points = gas.mesh_points(meshes, frames[s.mesh], s.mesh, int(vertadr[s.mesh]), device=device)
    fa = int(faceadr[s.mesh])
    faces = meshes.face[fa : fa + len(meshes.faces[s.mesh])]
    inputs.append((triangles(points, faces, m), (points, faces)))
  gases = gas.build_static(ctx, inputs, device=device)
  return CutGases(gases=gases, maps=maps, handle={s: g.handle for s, g in zip(sources, gases, strict=True)})
