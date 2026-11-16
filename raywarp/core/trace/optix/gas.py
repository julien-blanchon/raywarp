"""OptiX geometry acceleration structures (GASes) and their build inputs.

`Accel` owns one structure (its build input, the device buffers that input points to, the temp and
output buffers) and builds or refits it on a stream. `mesh_gases` builds the static triangle
GASes of the scene's meshes and of the unit cube (`build_static`: every build first, the compacted
sizes read with one synchronisation, then every compaction; a warm start copies the compacted
bytes from the bake cache and relocates them, `optixAccelRelocate`). `curves` /
`custom_primitives` make the build inputs of the world-space GASes refitted every frame
(`group.InstanceGroup`).
"""

from __future__ import annotations

import ctypes
import dataclasses
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform import bake_cache
from raywarp.core.pipeline.platform.device import host_copy, stream_handle
from raywarp.core.trace.optix import api, mesh_inputs
from raywarp.core.trace.optix import instances as ops
from raywarp.core.trace.optix.context import DeviceContext, address

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyHostArray
  from raywarp.core.pipeline.gpu_scene.gpu_scene import Meshes

# The device buffers a build input points to, kept alive with it.
type InputBuffers = tuple[object, ...]
# The mesh GASes, the GAS of each mesh and each mesh's box (lower, upper): built or restored.
type _Built = tuple[list[Accel], AnyHostArray, AnyHostArray, AnyHostArray]


def buffer_words(nbytes: int) -> int:
  """Length in 8-byte words of an OptiX buffer of `nbytes` bytes (at least one word).

  OptiX buffers are allocated as uint64 words: a byte array caps a buffer at 2^31 - 1 bytes (Warp
  array shapes are int32), which a GAS over ~34M triangles exceeds. Warp's allocations are aligned
  well beyond OptiX's 128 B (`OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT`), whatever the dtype.
  """
  return max((nbytes + 7) // 8, 1)


def byte_buffer(nbytes: int, device: wp.Device) -> wp.array[wp.uint64, [int]]:
  """An uninitialised buffer on `device` of at least `nbytes` bytes (whole uint64 words)."""
  return wp.empty(buffer_words(nbytes), dtype=wp.uint64, device=device)


@final
class Accel:
  """An acceleration structure with its build input and buffers (kept alive together)."""

  temp: wp.array[wp.uint64, [int]]
  output: wp.array[wp.uint64, [int]]
  temp_bytes: int
  output_bytes: int
  handle: int

  def __init__(
    self, ctx: DeviceContext, build_input: api.BuildInput, flags: int, *, device: wp.Device, keep: InputBuffers = ()
  ) -> None:
    """Sizes the buffers of `build_input` on `device` (`ctx`'s; `keep`: the device buffers it
    points to); built by `build`."""
    self.ctx, self.device = ctx, device
    self.keep = keep
    # Built with ALLOW_UPDATE: later builds may refit (OptiX rejects a refit otherwise).
    self.refit = bool(flags & api.BUILD_FLAG_ALLOW_UPDATE)
    self.inputs: ctypes.Array[api.BuildInput] = (api.BuildInput * 1)(build_input)
    self.options = {
      update: api.AccelBuildOptions(buildFlags=flags, operation=op)
      for update, op in ((False, api.BUILD_OPERATION_BUILD), (True, api.BUILD_OPERATION_UPDATE))
    }
    sizes = ctx.accel_sizes(self.options[False], self.inputs)
    # The byte sizes OptiX asked for go to `accel_build` (the buffers round them up to words).
    self.temp_bytes = max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, 8)
    self.output_bytes = max(sizes.outputSizeInBytes, 8)
    self.temp = byte_buffer(self.temp_bytes, device)
    self.output = byte_buffer(self.output_bytes, device)
    self.handle = 0

  @classmethod
  def relocated(
    cls,
    ctx: DeviceContext,
    output: wp.array[wp.uint64, [int]],
    nbytes: int,
    handle: int,
    *,
    keep: InputBuffers,
    device: wp.Device,
  ) -> Accel:
    """A static structure restored by relocation (`build_static`'s disk cache): `output` holds its
    `nbytes` bytes (`keep`: the buffer `output` is a view of), `handle` its traversable handle;
    built and compacted, like `build_static`'s."""
    accel = cls.__new__(cls)
    accel.ctx, accel.device, accel.keep, accel.refit = ctx, device, keep, False
    accel.inputs, accel.options = (api.BuildInput * 0)(), {}
    accel.temp, accel.temp_bytes = byte_buffer(8, device), 8
    accel.output, accel.output_bytes, accel.handle = output, nbytes, handle
    return accel

  def build(self, stream: int, *, update: bool = False, emit: ctypes.Array[api.AccelEmitDesc] | None = None) -> int:
    """Builds (or, `update`, refits) the structure on `stream`; returns its handle.

    `emit`: OptiX's emitted properties (`AccelEmitDesc`, e.g. the compacted size).
    """
    if update and not self.refit:
      raise RuntimeError("a refit of an acceleration structure built without ALLOW_UPDATE")
    temp, out = address(self.temp), address(self.output)
    self.handle = self.ctx.accel_build(
      stream, self.options[update], self.inputs, temp, self.temp_bytes, out, self.output_bytes, emit
    )
    return self.handle


def build_static(ctx: DeviceContext, inputs: list[tuple[api.BuildInput, InputBuffers]], *, device: wp.Device) -> list[Accel]:
  """Static triangle GASes of `inputs` (build input, buffers it reads) on `device`, built (on its
  current stream), then compacted.

  Every GAS is built first, the compacted sizes are read with one synchronisation, then every
  GAS is compacted; the build buffers (temp, uncompacted output, inputs) are freed.
  """
  flags = api.BUILD_FLAG_PREFER_FAST_TRACE | api.BUILD_FLAG_ALLOW_COMPACTION
  on = wp.get_stream(device)
  stream = stream_handle(on)
  gases = [Accel(ctx, build_input, flags, device=device, keep=keep) for build_input, keep in inputs]
  sizes = wp.zeros(max(len(gases), 1), dtype=wp.uint64, device=device)
  for i, gas in enumerate(gases):
    emit = (api.AccelEmitDesc * 1)(api.AccelEmitDesc(result=address(sizes) + 8 * i, type=api.PROPERTY_TYPE_COMPACTED_SIZE))
    gas.build(stream, emit=emit)
  compacted = host_copy(sizes)  # one synchronisation for every build
  built = []  # the uncompacted outputs, alive until their compactions ran
  for gas, size in zip(gases, compacted, strict=False):
    if 0 < int(size) < gas.output_bytes:
      output = byte_buffer(int(size), device)
      gas.handle = ctx.accel_compact(stream, gas.handle, address(output), int(size))
      built.append(gas.output)
      gas.output, gas.output_bytes = output, int(size)
    gas.temp, gas.temp_bytes = byte_buffer(8, device), 8
    gas.inputs, gas.keep = (api.BuildInput * 0)(), ()
  wp.synchronize_stream(on)
  del built
  return gases


# Byte alignment of each structure in the stored blob (OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT).
_ALIGN: Final = 128


def _offsets(nbytes: np.ndarray[[int], np.dtype[np.int64]]) -> np.ndarray[[int], np.dtype[np.int64]]:
  """Byte offset of each structure in the blob (each aligned to `_ALIGN`), then the blob's size."""
  return np.concatenate([[0], np.cumsum((nbytes + _ALIGN - 1) // _ALIGN * _ALIGN)]).astype(np.int64)


def _store(ctx: DeviceContext, gases: list[Accel]) -> dict[str, np.ndarray]:
  """The cached form of built structures: their compacted bytes back to back (`blob`), their
  sizes and relocation information (`optixAccelGetRelocationInfo`)."""
  nbytes = np.asarray([g.output_bytes for g in gases], np.int64)
  at = _offsets(nbytes)
  blob = np.zeros(int(at[-1]) // 8, np.uint64)
  info = np.zeros((len(gases), 4), np.uint64)
  for k, g in enumerate(gases):
    blob[at[k] // 8 : (at[k] + nbytes[k] + 7) // 8] = host_copy(g.output)[: (nbytes[k] + 7) // 8]
    info[k] = ctx.relocation_info(g.handle)
  return {"blob": blob, "nbytes": nbytes, "info": info}


def _restored(ctx: DeviceContext, stored: bake_cache.Bake, *, device: wp.Device) -> list[Accel] | None:
  """The structures copied back from their cached form (`_store`: one copy of the blob, one
  relocation each), or None (a structure this driver and device cannot relocate)."""
  count = len(stored["nbytes"])
  info = np.asarray(stored["info"], np.uint64)
  if not all(ctx.relocation_compatible(row) for row in info):
    return None
  blob = wp.array(np.asarray(stored["blob"], np.uint64), dtype=wp.uint64, device=device)
  nbytes = np.asarray(stored["nbytes"], np.int64)
  at = _offsets(nbytes)
  stream = stream_handle(wp.get_stream(device))
  out = []
  for k in range(count):
    view = blob[int(at[k]) // 8 : int(at[k] + nbytes[k] + 7) // 8]
    handle = ctx.accel_relocate(stream, info[k], address(view), int(nbytes[k]))
    out.append(Accel.relocated(ctx, view, int(nbytes[k]), handle, keep=(blob,), device=device))
  return out


def dynamic_build_flags() -> int:
  """Build flags of the structures refitted every frame (custom primitives, instances)."""
  return api.BUILD_FLAG_ALLOW_UPDATE | api.BUILD_FLAG_PREFER_FAST_TRACE


def curve_build_flags() -> int:
  """Build flags of the curve GASes and of the built-in curve intersector (they must match):
  refitted every frame, with PREFER_FAST_BUILD, which also makes OptiX split the segments
  uniformly. OptiX splits curves adaptively by default, and a refit of an adaptively split GAS
  falls back to uniform splitting with a warning on every refit. Measured on L4 (144 capsules x
  256 worlds, 512^2): within noise of a plain refittable build (52.8 vs 52.0 ms per frame), 3 %
  faster than the former fast-trace refit (54.4 ms) and than a fast-trace rebuild every frame
  (54.2 ms)."""
  return api.BUILD_FLAG_ALLOW_UPDATE | api.BUILD_FLAG_PREFER_FAST_BUILD


def triangles(points: wp.array[wp.vec3, [int]], indices: wp.array[wp.vec3i, [int]], flag: int) -> api.BuildInput:
  """Triangle build input (one SBT record, geometry flag `flag`) over `points` and `indices`."""
  bi = api.BuildInput(type=api.BUILD_INPUT_TYPE_TRIANGLES)
  tri = bi.triangleArray
  tri.vertexFormat = api.VERTEX_FORMAT_FLOAT3
  tri.numVertices = points.shape[0]
  tri.vertexBuffers = (api.CUdeviceptr * 1)(_input(points))
  tri.indexFormat = api.INDICES_FORMAT_UNSIGNED_INT3
  tri.numIndexTriplets = indices.shape[0]
  tri.indexBuffer = _input(indices)
  tri.flags = (ctypes.c_uint * 1)(flag)
  tri.numSbtRecords = 1
  return bi


def _triangles(points: wp.array[wp.vec3, [int]], indices: wp.array[wp.vec3i, [int]]) -> tuple[api.BuildInput, InputBuffers]:
  """Triangle build input (no any-hit) over `points` and `indices`, and its buffers."""
  return triangles(points, indices, api.GEOMETRY_FLAG_DISABLE_ANYHIT), (points, indices)


def _input(a: wp.array) -> int:
  """The device address of a build input's buffer (0 for an empty one)."""
  return address(a, empty_ok=True)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MeshGases:
  """Triangle GAS per mesh (and of the unit cube), with what the instance writers read.

  Attributes:
    gases: one GAS per mesh (none without mesh instances), then the cube's (with box instances).
    handle: (max(nmesh, 1),) traversable handle of each mesh's GAS.
    cube: traversable handle of the unit cube's GAS (0: no boxes).
    lower, upper: (max(nmesh, 1),) each mesh's box in its GAS frame.
    frame_t: (max(nmesh, 1),) R^T of each mesh's GAS frame (identity: the mesh frame).
    face_normal: (max(nface, 1),) packed object-space normal of every face, by face address.
  """

  gases: list[Accel]
  handle: wp.array[wp.uint64, [int]]
  cube: int
  lower: wp.array[wp.vec3, [int]]
  upper: wp.array[wp.vec3, [int]]
  frame_t: wp.array[wp.mat33, [int]]
  face_normal: wp.array[wp.uint32, [int]]


def mesh_frames(
  frames: np.ndarray[[int, 3, 3], np.dtype[np.float32]] | None, nmesh: int
) -> np.ndarray[[int, 3, 3], np.dtype[np.float32]]:
  """The frame R of each mesh's GAS ((max(nmesh, 1), 3, 3); identity: the mesh frame)."""
  rot = np.tile(np.eye(3, dtype=np.float32), (max(nmesh, 1), 1, 1))
  if frames is not None and nmesh:
    rot[:] = np.asarray(frames, np.float32)
  return rot


def mesh_points(
  meshes: Meshes, rot: np.ndarray[[3, 3], np.dtype[np.float32]], m: int, va: int, *, device: wp.Device
) -> tuple[np.ndarray[[int, 3], np.dtype[np.float32]], wp.array[wp.vec3, [int]]]:
  """(host, device) vertices of mesh m's GAS (starting at vertex address `va`): in frame `rot`
  when re-framed (a device copy), else the scene's vertices."""
  p = meshes.points[m]
  if np.allclose(rot, np.eye(3)):
    return p, meshes.vert[va : va + len(p)]
  p = (p.astype(np.float64) @ rot.T.astype(np.float64)).astype(np.float32)
  return p, wp.array(p, dtype=wp.vec3, device=device)


def _driver(device: wp.Device) -> tuple[str, object]:
  """The device name and CUDA driver version (a GAS cache entry's builder: its bytes are the
  driver's)."""
  return device.name, wp.get_cuda_driver_version()


def _mesh_inputs(
  meshes: Meshes,
  rot: np.ndarray[[int, 3, 3], np.dtype[np.float32]],
  used: np.ndarray[[int], np.dtype[np.bool_]] | None,
  *,
  nmesh: int,
  has_box: bool,
  device: wp.Device,
) -> tuple[
  list[tuple[api.BuildInput, InputBuffers]],
  np.ndarray[[int], np.dtype[np.int64]],
  np.ndarray[[int, 3], np.dtype[np.float32]],
  np.ndarray[[int, 3], np.dtype[np.float32]],
]:
  """(build inputs of the mesh GASes (unused meshes: one shared empty placeholder), then the
  cube's; the GAS of each mesh; each mesh's box (lower, upper) in its GAS frame)."""
  vertadr, faceadr = host_copy(meshes.vertadr), host_copy(meshes.faceadr)
  inputs, lower, upper = [], [], []
  gas_of = np.zeros(max(nmesh, 1), np.int64)
  placeholder = -1
  for m in range(nmesh):
    if used is not None and not used[m]:
      if placeholder < 0:
        placeholder = len(inputs)
        empty = (wp.zeros(3, dtype=wp.vec3, device=device), wp.zeros(1, dtype=wp.vec3i, device=device))
        inputs.append(_triangles(*empty))
      gas_of[m] = placeholder
      lower.append(np.zeros(3, np.float32))
      upper.append(np.zeros(3, np.float32))
      continue
    p, points = mesh_points(meshes, rot[m], m, int(vertadr[m]), device=device)
    fa = int(faceadr[m])
    gas_of[m] = len(inputs)
    inputs.append(_triangles(points, meshes.face[fa : fa + len(meshes.faces[m])]))
    lower.append(p.min(axis=0))
    upper.append(p.max(axis=0))
  if has_box:
    vertices, faces = mesh_inputs.unit_cube()
    cube = (wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces, dtype=wp.vec3i, device=device))
    inputs.append(_triangles(*cube))
  box = (np.asarray(lower or [[0.0] * 3], np.float32), np.asarray(upper or [[0.0] * 3], np.float32))
  return inputs, gas_of, *box


def mesh_gases(
  ctx: DeviceContext,
  meshes: Meshes,
  frames: np.ndarray[[int, 3, 3], np.dtype[np.float32]] | None,
  *,
  used: np.ndarray[[int], np.dtype[np.bool_]] | None,
  has_mesh: bool,
  has_box: bool,
  device: wp.Device,
) -> MeshGases:
  """The static triangle GASes on `device` of the scene's `used` meshes (a Renderer's mask,
  `RendererTables.mesh_used`; None: all; vertices in frame R when re-framed) and the cube; from
  the disk cache when it holds them (their compacted bytes copied back and relocated, `_restored`;
  keyed by the meshes' content digests registered at upload, the frames, the mask, the device and
  the driver; an entry the driver cannot relocate is built and stored again)."""
  nmesh = len(meshes.points) if has_mesh else 0
  rot = mesh_frames(frames, nmesh)
  faceadr = host_copy(meshes.faceadr)
  face_mesh = np.zeros(max(int(meshes.face.shape[0]), 1), np.int32)
  for m in range(nmesh):
    face_mesh[int(faceadr[m]) : int(faceadr[m]) + len(meshes.faces[m])] = m
  key = bake_cache.key(
    "gas",
    device,
    host=(meshes.vert, meshes.face, meshes.vertadr, meshes.faceadr, rot, used, has_mesh, has_box, _driver(device)),
  )

  def build() -> tuple[_Built, bake_cache.Bake]:
    inputs, gas_of, lower, upper = _mesh_inputs(meshes, rot, used, nmesh=nmesh, has_box=has_box, device=device)
    gases = build_static(ctx, inputs, device=device)
    return (gases, gas_of, lower, upper), {**_store(ctx, gases), "gas_of": gas_of, "lower": lower, "upper": upper}

  def restore(stored: bake_cache.Bake) -> _Built | None:
    found = _restored(ctx, stored, device=device)
    return None if found is None else (found, stored["gas_of"], stored["lower"], stored["upper"])

  gases, gas_of, lower, upper = bake_cache.cached(key, build, restore)
  face_normal = wp.zeros(face_mesh.shape[0], dtype=wp.uint32, device=device)
  frame = wp.array(rot, dtype=wp.mat33, device=device)
  if nmesh:
    wp.launch(
      mesh_inputs.face_normals,
      dim=int(meshes.face.shape[0]),
      inputs=[meshes.vertadr, meshes.vert, meshes.face, frame, wp.array(face_mesh, dtype=int, device=device)],
      outputs=[face_normal],
      device=device,
    )
  return MeshGases(
    gases=gases,
    handle=wp.array(
      np.asarray([gases[int(g)].handle for g in gas_of[:nmesh]] or [0], np.uint64), dtype=wp.uint64, device=device
    ),
    cube=gases[-1].handle if has_box else 0,
    lower=wp.array(np.asarray(lower, np.float32), dtype=wp.vec3, device=device),
    upper=wp.array(np.asarray(upper, np.float32), dtype=wp.vec3, device=device),
    frame_t=wp.array(np.ascontiguousarray(np.transpose(rot, (0, 2, 1))), dtype=wp.mat33, device=device),
    face_normal=face_normal,
  )


def curves[N: IntVar, M: IntVar](
  ctx: DeviceContext,
  vertices: wp.array[wp.vec3, [N]],
  widths: wp.array[float, [N]],
  index: wp.array[wp.uint32, [M]],
  *,
  device: wp.Device,
) -> Accel:
  """Linear-curve GAS over `index.shape[0]` segments (`curve_build_flags`, the intersector's):
  segment k joins vertices index[k] and index[k] + 1, of radius `widths` at each vertex."""
  bi = api.BuildInput(type=api.BUILD_INPUT_TYPE_CURVES)
  c = bi.curveArray
  c.curveType = api.PRIMITIVE_TYPE_ROUND_LINEAR
  c.numPrimitives = index.shape[0]
  c.vertexBuffers = (api.CUdeviceptr * 1)(_input(vertices))
  c.numVertices = vertices.shape[0]
  c.vertexStrideInBytes = 12
  c.widthBuffers = (api.CUdeviceptr * 1)(_input(widths))
  c.widthStrideInBytes = 4
  c.indexBuffer = _input(index)
  c.indexStrideInBytes = 4
  c.flag = api.GEOMETRY_FLAG_DISABLE_ANYHIT
  return Accel(ctx, bi, curve_build_flags(), device=device, keep=(vertices, widths, index))


def custom_primitives[N: IntVar](
  ctx: DeviceContext, aabbs: wp.array[ops.Aabb, [N]], sbt: wp.array[wp.uint32, [N]], ntypes: int, *, device: wp.Device
) -> Accel:
  """Custom-primitive GAS over `aabbs` (`sbt`: SBT record of each primitive, `ntypes` records)."""
  bi = api.BuildInput(type=api.BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES)
  c = bi.customPrimitiveArray
  c.aabbBuffers = (api.CUdeviceptr * 1)(_input(aabbs))
  c.numPrimitives = aabbs.shape[0]
  c.flags = (ctypes.c_uint * ntypes)(*[api.GEOMETRY_FLAG_DISABLE_ANYHIT] * ntypes)
  c.numSbtRecords = ntypes
  c.sbtIndexOffsetBuffer = _input(sbt)
  c.sbtIndexOffsetSizeInBytes = 4
  c.sbtIndexOffsetStrideInBytes = 4
  return Accel(ctx, bi, dynamic_build_flags(), device=device, keep=(aabbs, sbt))


def instance_array(ctx: DeviceContext, instances: wp.array[ops.OptixInstance, [int]], *, device: wp.Device) -> Accel:
  """IAS over `instances` (refitted every frame)."""
  bi = api.BuildInput(type=api.BUILD_INPUT_TYPE_INSTANCES)
  bi.instanceArray.instances, bi.instanceArray.numInstances = _input(instances), instances.shape[0]
  return Accel(ctx, bi, dynamic_build_flags(), device=device, keep=(instances,))
