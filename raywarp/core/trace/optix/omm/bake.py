"""The opacity micromap bake of one cut-out source (`bake`): its faces' subdivision levels, the
4-state micromaps of the faces with UVs (`kernels.py`), and the kept ones packed.

The bake is conservative by construction (`kernels.py`). Levels are lowered under
`BUDGET_BYTES` for a whole build (`level_cap`). Triangles whose micro-triangles all agree use
OptiX's predefined indices (no micromap); faces without UVs never cut (FULLY_OPAQUE).
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp._base.types import uv_xform
from raywarp.cache.paths import cache_dir
from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.trace.optix import api
from raywarp.core.trace.optix.omm.kernels import MAX_LEVEL, micro_levels, micro_states, pack_kept

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp.core.pipeline.gpu_scene.gpu_scene import Meshes
  from raywarp.core.trace.omm_sources import CutSource

# Dense bake buffer cap [bytes] of a build (every source), before the uniform triangles are
# dropped: the cap level is lowered until the build's micromaps fit.
BUDGET_BYTES: Final = 1 << 27
# OptiX's predefined indices (optix_types.h OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_*).
FULLY_TRANSPARENT: Final = -1
FULLY_OPAQUE: Final = -2
FULLY_UNKNOWN_OPAQUE: Final = -4


def micromap_bytes(level: np.ndarray[[int], np.dtype[np.int64]]) -> np.ndarray[[int], np.dtype[np.int64]]:
  """Bytes of 4-state micromaps of these levels (>= 1): 4^level states of 2 bits."""
  return np.left_shift(np.int64(1), 2 * level) // 4


def level_cap(levels: list[np.ndarray[[int], np.dtype[np.int64]]]) -> int:
  """The highest subdivision cap whose dense micromaps (faces with UVs, every source) fit
  `BUDGET_BYTES`."""
  level = np.concatenate([lv[lv >= 1] for lv in levels]) if levels else np.zeros(0, np.int64)
  cap = MAX_LEVEL
  while cap > 1 and int(micromap_bytes(np.minimum(level, cap)).sum()) > BUDGET_BYTES:
    cap -= 1
  return cap


def face_levels(
  meshes: Meshes, source: CutSource, tex: wp.Texture2D, *, device: wp.Device
) -> np.ndarray[[int], np.dtype[np.int64]]:
  """The subdivision level `micro_levels` picks for each face of `source`'s mesh (before a cap)."""
  nface = len(meshes.faces[source.mesh])
  levels = wp.zeros(max(nface, 1), dtype=int, device=device)
  wp.launch(micro_levels, dim=nface, inputs=[*_mesh_model(meshes), tex, *_uv_args(source)], outputs=[levels], device=device)
  return host_copy(levels)[:nface].astype(np.int64)


@final
@dataclasses.dataclass(slots=True, kw_only=True)
class Bake:
  """A cut-out mesh's baked micromaps (`bake`), as the array and GAS builds read them.

  Attributes:
    data: the kept micromaps' 4-state data, back to back (at least one byte).
    desc: (nkept, 2) OptixOpacityMicromapDesc words: byte offset, level | format << 16.
    index: (nface,) micromap of each face, or a predefined index (FULLY_*).
    histogram: (count, level, format) of the kept micromaps (their array build and usage counts).
    faces: (opaque, transparent, unknown, micromap) face counts; `micro`: micro-triangles baked.
    seconds: bake time (synchronised).
  """

  data: wp.array[wp.uint8, [int]]
  desc: wp.array[wp.uint32, [int]]
  index: wp.array[wp.int32, [int]]
  histogram: list[tuple[int, int, int]]
  faces: tuple[int, int, int, int]
  micro: int
  seconds: float


def _uv_args(source: CutSource) -> list[object]:
  """The kernels' (texcoord, xform, repeat, mesh) inputs of `source`."""
  return [
    wp.vec2i(*source.texcoord),
    uv_xform(*source.uv_xform),
    wp.vec2(*source.repeat),
    source.mesh,
  ]


def _mesh_model(meshes: Meshes) -> list[object]:
  """The kernels' mesh UV tables."""
  return [meshes.faceadr, meshes.facetexcoord, meshes.texcoord, meshes.texcoord_adr]


def _dense[N: IntVar](
  meshes: Meshes, source: CutSource, tex: wp.Texture2D, level: np.ndarray[[N], np.dtype[np.int64]], *, device: wp.Device
) -> tuple[
  wp.array[wp.uint8, [int]],
  np.ndarray[[int], np.dtype[np.int64]],
  np.ndarray[[int], np.dtype[np.int64]],
  np.ndarray[[int, 2], np.dtype[np.int64]],
]:
  """The dense micromaps of the faces with UVs: (data, faces, starts, (opaque, transparent)
  counts per micromap)."""
  faces = np.flatnonzero(level >= 1).astype(np.int64)
  nbytes = micromap_bytes(level[faces])
  ends = np.cumsum(nbytes)
  starts = ends - nbytes
  total = int(ends[-1]) if len(ends) else 0
  data = wp.zeros(max(total, 1), dtype=wp.uint8, device=device)
  counts = [wp.zeros(max(len(faces), 1), dtype=int, device=device) for _ in range(2)]
  if total:
    ints = [wp.array(a.astype(np.int32), dtype=int, device=device) for a in (faces, level[faces], starts, ends)]
    wp.launch(
      micro_states,
      dim=total,
      inputs=[*_mesh_model(meshes), tex, source.layout, source.cutoff, *_uv_args(source), *ints],
      outputs=[data, *counts],
      device=device,
    )
  return data, faces, starts, np.stack([host_copy(c)[: len(faces)].astype(np.int64) for c in counts], axis=1)


def bake(
  meshes: Meshes,
  source: CutSource,
  tex: wp.Texture2D,
  *,
  levels: np.ndarray[[int], np.dtype[np.int64]] | None = None,
  cap: int = MAX_LEVEL,
  device: wp.Device,
) -> Bake:
  """The micromaps of `source`'s mesh on `device` (`tex`: its opacity texture; `levels`: its
  faces' `face_levels`, computed when None; levels at most `cap`; module doc), from the bake cache
  when it holds them (`bake_cache`, kind "omm", `_key`)."""
  t0 = time.perf_counter()
  level = np.minimum(face_levels(meshes, source, tex, device=device) if levels is None else levels, cap)
  key = _key(meshes, source, tex, level, device=device)

  def compute() -> tuple[Bake, bake_cache.Bake]:
    b = _bake(meshes, source, tex, level, device=device)
    return b, _stored(b)

  b = bake_cache.cached(key, compute, lambda stored: _restored(stored, device=device))
  b.seconds = time.perf_counter() - t0
  return b


def _key(
  meshes: Meshes, source: CutSource, tex: wp.Texture2D, level: np.ndarray[[int], np.dtype[np.int64]], *, device: wp.Device
) -> bake_cache.Digest | None:
  """The bake cache key of a bake (None: caching off, or an unregistered texture): the alpha
  test's values, the texture's content digest, the levels and what the faces' texel corners are
  computed from (the source's UV address and mesh, the meshes' UV tables: the upload registers
  their content digests, `bake_cache.register_array`, so nothing is copied back or hashed again)."""
  if cache_dir() is None:
    return None
  alpha_test = (source.layout, source.cutoff, source.repeat, source.uv_xform)
  return bake_cache.key("omm", device, host=(alpha_test, tex, level, source.texcoord, source.mesh, *_mesh_model(meshes)))


def _stored(b: Bake) -> bake_cache.Bake:
  """The cached arrays of a `Bake`."""
  return {
    "data": host_copy(b.data),
    "desc": host_copy(b.desc),
    "index": host_copy(b.index),
    "histogram": np.asarray(b.histogram, np.int64).reshape(-1, 3),
    "faces": np.asarray(b.faces, np.int64),
    "micro": np.asarray(b.micro, np.int64),
  }


def _restored(stored: bake_cache.Bake, *, device: wp.Device) -> Bake:
  """A `Bake` from its cached arrays (its time: the caller's)."""
  return Bake(
    data=wp.array(stored["data"], dtype=wp.uint8, device=device),
    desc=wp.array(stored["desc"], dtype=wp.uint32, device=device),
    index=wp.array(stored["index"], dtype=wp.int32, device=device),
    histogram=[(int(c), int(lv), int(f)) for c, lv, f in stored["histogram"]],
    faces=(int(stored["faces"][0]), int(stored["faces"][1]), int(stored["faces"][2]), int(stored["faces"][3])),
    micro=int(stored["micro"]),
    seconds=0.0,
  )


def _bake(
  meshes: Meshes, source: CutSource, tex: wp.Texture2D, level: np.ndarray[[int], np.dtype[np.int64]], *, device: wp.Device
) -> Bake:
  """The micromaps of `source`'s mesh at the faces' (capped) levels `level` (`bake`'s work)."""
  t0 = time.perf_counter()
  nface = len(meshes.faces[source.mesh])
  data, faces, starts, counts = _dense(meshes, source, tex, level, device=device)
  micro = np.left_shift(np.int64(1), 2 * level[faces])
  index = np.full(nface, FULLY_OPAQUE, np.int32)  # faces without UVs never cut
  opaque, clear = counts[:, 0] == micro, counts[:, 1] == micro
  unknown = counts.sum(axis=1) == 0
  index[faces[opaque]] = FULLY_OPAQUE
  index[faces[clear]] = FULLY_TRANSPARENT
  index[faces[unknown]] = FULLY_UNKNOWN_OPAQUE
  kept = np.flatnonzero(~(opaque | clear | unknown))
  index[faces[kept]] = np.arange(len(kept)).astype(np.int32)
  packed, desc, histogram = _pack(data, starts[kept], level[faces[kept]], device=device)
  wp.synchronize_device(device)
  return Bake(
    data=packed,
    desc=desc,
    index=wp.array(index, dtype=wp.int32, device=device),
    histogram=histogram,
    faces=(int(opaque.sum()) + nface - len(faces), int(clear.sum()), int(unknown.sum()), len(kept)),
    micro=int(micro.sum()),
    seconds=time.perf_counter() - t0,
  )


def _pack[K: IntVar](
  data: wp.array[wp.uint8, [int]],
  source: np.ndarray[[K], np.dtype[np.int64]],
  level: np.ndarray[[K], np.dtype[np.int64]],
  *,
  device: wp.Device,
) -> tuple[wp.array[wp.uint8, [int]], wp.array[wp.uint32, [int]], list[tuple[int, int, int]]]:
  """The kept micromaps (dense offsets `source`, levels `level`) packed: (data, descriptors,
  histogram). Without any, one unused all-unknown level-1 micromap (OptiX wants an array)."""
  if not len(level):
    one = wp.array(np.full(1, 0xFF, np.uint8), dtype=wp.uint8, device=device)
    return (
      one,
      wp.array(np.array([0, 1 | api.OPACITY_MICROMAP_FORMAT_4_STATE << 16], np.uint32), dtype=wp.uint32, device=device),
      [(1, 1, api.OPACITY_MICROMAP_FORMAT_4_STATE)],
    )
  nbytes = micromap_bytes(level)
  ends = np.cumsum(nbytes)
  starts = ends - nbytes
  out = wp.zeros(int(ends[-1]), dtype=wp.uint8, device=device)
  ints = [wp.array(a.astype(np.int32), dtype=int, device=device) for a in (source, starts, ends)]
  wp.launch(pack_kept, dim=int(ends[-1]), inputs=[data, *ints], outputs=[out], device=device)
  words = np.stack([starts, level | api.OPACITY_MICROMAP_FORMAT_4_STATE << 16], axis=1).astype(np.uint32).reshape(-1)
  levels, count = np.unique(level, return_counts=True)
  histogram = [(int(c), int(lv), api.OPACITY_MICROMAP_FORMAT_4_STATE) for lv, c in zip(levels, count, strict=True)]
  return out, wp.array(words, dtype=wp.uint32, device=device), histogram
