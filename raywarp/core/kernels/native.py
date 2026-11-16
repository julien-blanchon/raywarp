"""Makes the few C++ functions our kernels call (BVH node reads, OptiX intrinsics) callable from
Warp kernels: native builtins, registered once, with typed access.

Warp's experimental external-compilation API (`wp.build_experimental`) declares a C++ function
of a module preamble as a builtin. This module is the only place that registers ours, at import:
- the BVH node accessors of the traversals (`bvh_lowers`, `bvh_node_at`, ...; C++ in
  `core/trace/accel/nodes.py`'s preamble, which kernels get through `nodes.BUILD_OPTIONS`);
- the OptiX device intrinsics (`optix_*`; C++ in `core/trace/optix/device/intrinsics.py`'s
  preamble, which the programs get through `device.build_options()`).
Each preamble stays with the code that owns its layout; the ctypes mirrors below follow it field
by field.

Kernels call the builtins through this module (`native.bvh_node_at(...)`,
`native.optix_ray_tmax()`): Warp resolves the attribute to the registered builtin, and the type
checker reads their signatures from `native.pyi` (Warp itself registers them as
`wp.mjpbr_<name>`, which no stub declares).
"""

from __future__ import annotations

import ctypes
from typing import Final

import warp as wp


class BvhNode(ctypes.Structure):
  """`mjpbr::BvhNode`: bounds plus children (inner node) or primitive slot range (leaf)."""

  _fields_ = [
    ("lower", ctypes.c_float * 3),
    ("left", ctypes.c_int32),
    ("upper", ctypes.c_float * 3),
    ("right", ctypes.c_int32),
    ("leaf", ctypes.c_int32),
  ]


class MeshHit(ctypes.Structure):
  """`mjpbr::MeshHit`: a mesh query's closest hit (t < 0: none), mesh-frame unit normal."""

  _fields_ = [
    ("t", ctypes.c_float),
    ("u", ctypes.c_float),
    ("v", ctypes.c_float),
    ("face", ctypes.c_int32),
    ("normal", ctypes.c_float * 3),
  ]


class TraceResult(ctypes.Structure):
  """`mjpbr_optix::TraceResult`: a closest-hit trace's payload, decoded."""

  _fields_ = [
    ("t", ctypes.c_float),
    ("instance", ctypes.c_int32),
    ("primitive", ctypes.c_int32),
    ("a", ctypes.c_float),
    ("b", ctypes.c_float),
    ("c", ctypes.c_float),
    ("normal", ctypes.c_uint32),
  ]


class LayerArgs(ctypes.Structure):
  """`mjpbr_optix::LayerArgs`: the layer shading call's arguments
  (`trace/optix/device/intrinsics.py`)."""

  _fields_ = [
    ("world", ctypes.c_int32),
    ("instance", ctypes.c_int32),
    ("dist", ctypes.c_float),
    ("normal", ctypes.c_float * 3),
    ("bary_u", ctypes.c_float),
    ("bary_v", ctypes.c_float),
    ("face", ctypes.c_int32),
    ("mesh", ctypes.c_int32),
    ("point", ctypes.c_float * 3),
    ("direction", ctypes.c_float * 3),
    ("cone", ctypes.c_float),
    ("spread", ctypes.c_float),
    ("headlight", ctypes.c_float * 3),
    ("seed", ctypes.c_int32),
    ("stream", ctypes.c_int32),
  ]


class LayerResult(ctypes.Structure):
  """`mjpbr_optix::LayerResult`: the layer shading call's result
  (`trace/optix/device/intrinsics.py`)."""

  _fields_ = [
    ("radiance", ctypes.c_float * 3),
    ("opacity", ctypes.c_float),
    ("reflected", ctypes.c_float),
    ("color", ctypes.c_float * 3),
    ("specular_only", ctypes.c_int32),
    ("roughness", ctypes.c_float),
    ("base_color", ctypes.c_float * 3),
  ]


class VisibilityArgs(ctypes.Structure):
  """`mjpbr_optix::VisibilityArgs`: the light visibility call's arguments
  (`trace/optix/device/intrinsics.py`)."""

  _fields_ = [
    ("world", ctypes.c_int32),
    ("light", ctypes.c_int32),
    ("point", ctypes.c_float * 3),
    ("normal", ctypes.c_float * 3),
    ("origin", ctypes.c_float * 3),
    ("seed", ctypes.c_int32),
    ("stream", ctypes.c_int32),
  ]


wp.build_experimental.add_native_type(
  BvhNode,
  native_name="mjpbr::BvhNode",
  fields={"lower": wp.vec3, "left": wp.int32, "upper": wp.vec3, "right": wp.int32, "leaf": wp.int32},
)
wp.build_experimental.add_native_type(
  MeshHit,
  native_name="mjpbr::MeshHit",
  fields={"t": float, "u": float, "v": float, "face": wp.int32, "normal": wp.vec3},
)
wp.build_experimental.add_native_type(
  TraceResult,
  native_name="mjpbr_optix::TraceResult",
  fields={
    "t": float,
    "instance": wp.int32,
    "primitive": wp.int32,
    **dict.fromkeys(("a", "b", "c"), float),
    "normal": wp.uint32,
  },
)


wp.build_experimental.add_native_type(
  LayerArgs,
  native_name="mjpbr_optix::LayerArgs",
  fields={
    "world": wp.int32,
    "instance": wp.int32,
    "dist": float,
    "normal": wp.vec3,
    "bary_u": float,
    "bary_v": float,
    "face": wp.int32,
    "mesh": wp.int32,
    "point": wp.vec3,
    "direction": wp.vec3,
    "cone": float,
    "spread": float,
    "headlight": wp.vec3,
    "seed": wp.int32,
    "stream": wp.int32,
  },
)
wp.build_experimental.add_native_type(
  LayerResult,
  native_name="mjpbr_optix::LayerResult",
  fields={
    "radiance": wp.vec3,
    "opacity": float,
    "reflected": float,
    "color": wp.vec3,
    "specular_only": wp.int32,
    "roughness": float,
    "base_color": wp.vec3,
  },
)
wp.build_experimental.add_native_type(
  VisibilityArgs,
  native_name="mjpbr_optix::VisibilityArgs",
  fields={
    "world": wp.int32,
    "light": wp.int32,
    "point": wp.vec3,
    "normal": wp.vec3,
    "origin": wp.vec3,
    "seed": wp.int32,
    "stream": wp.int32,
  },
)


def _bvh(name: str, inputs: dict[str, type], value_type: type | None) -> wp.Function:
  """Registers `mjpbr::<name>` as the builtin `wp.mjpbr_<name>`."""
  return wp.build_experimental.add_builtin(f"mjpbr_{name}", inputs, value_type, native_name=f"mjpbr::{name}")


def _optix(name: str, inputs: dict[str, type], value_type: type | None) -> wp.Function:
  """Registers `mjpbr_optix::<name>` as the builtin `wp.mjpbr_optix_<name>`."""
  return wp.build_experimental.add_builtin(f"mjpbr_optix_{name}", inputs, value_type, native_name=f"mjpbr_optix::{name}")


# BVH node accessors (C++ in `core/trace/accel/nodes.py`'s preamble).
bvh_lowers = _bvh("bvh_lowers", {"bvh_id": wp.uint64}, wp.uint64)
bvh_uppers = _bvh("bvh_uppers", {"bvh_id": wp.uint64}, wp.uint64)
bvh_prims = _bvh("bvh_prims", {"bvh_id": wp.uint64}, wp.uint64)
bvh_node_at = _bvh("bvh_node_at", {"lowers": wp.uint64, "uppers": wp.uint64, "index": int}, BvhNode)
bvh_primitive_at = _bvh("bvh_primitive_at", {"prims": wp.uint64, "slot": int}, int)
# Mesh BLAS queries (same preamble).
mesh_closest = _bvh(
  "mesh_closest",
  {
    "mesh_id": wp.uint64,
    "halves": wp.uint64,
    "start": wp.vec3,
    "dir": wp.vec3,
    "max_t": float,
    "root": int,
    "cull": wp.bool,
    "counts": wp.uint64,
  },
  MeshHit,
)
mesh_any = _bvh(
  "mesh_any",
  {
    "mesh_id": wp.uint64,
    "halves": wp.uint64,
    "start": wp.vec3,
    "dir": wp.vec3,
    "max_t": float,
    "root": int,
    "counts": wp.uint64,
  },
  wp.bool,
)
mesh_max_nodes = _bvh("mesh_max_nodes", {"mesh_id": wp.uint64}, int)
mesh_num_items = _bvh("mesh_num_items", {"mesh_id": wp.uint64}, int)
mesh_pack_node = _bvh("mesh_pack_node", {"mesh_id": wp.uint64, "halves": wp.uint64, "index": int}, None)
bvh_max_nodes = _bvh("bvh_max_nodes", {"bvh_id": wp.uint64}, int)
mesh_max_depth = _bvh("mesh_max_depth", {"mesh_id": wp.uint64}, int)
bvh_max_depth = _bvh("bvh_max_depth", {"bvh_id": wp.uint64}, int)
bvh_pack_node = _bvh("bvh_pack_node", {"bvh_id": wp.uint64, "halves": wp.uint64, "index": int}, None)
packed_child = _bvh("packed_child", {"halves": wp.uint64, "slot": int}, BvhNode)

# OptiX device intrinsics (`trace/optix/device/intrinsics.py`): traces from the ray-generation
# program, then the program state the miss / hit / intersection programs read and write.
_RAY: Final = {"handle": wp.uint64, "o": wp.vec3, "d": wp.vec3, "tmin": float, "tmax": float}
optix_launch_index = _optix("launch_index", {}, int)
optix_trace_closest = _optix("trace_closest", {**_RAY, "ray_flags": wp.uint32, "mask": wp.uint32, "world": int}, TraceResult)
optix_trace_any = _optix("trace_any", {**_RAY, "mask": wp.uint32}, wp.bool)
optix_trace_any_world = _optix("trace_any_world", {**_RAY, "mask": wp.uint32, "world": int}, wp.bool)
optix_trace_shadow = _optix(
  "trace_shadow", {**_RAY, "ray_flags": wp.uint32, "mask": wp.uint32, "world": int, "t": wp.vec3}, wp.vec3
)
optix_layer_call = _optix(
  "layer_call",
  {
    "world": int,
    "instance": int,
    "dist": float,
    "normal": wp.vec3,
    "bary_u": float,
    "bary_v": float,
    "face": int,
    "mesh": int,
    "point": wp.vec3,
    "direction": wp.vec3,
    "cone": float,
    "spread": float,
    "headlight": wp.vec3,
    "seed": int,
    "stream": int,
  },
  LayerResult,
)
optix_layer_args = _optix("layer_args", {}, LayerArgs)
optix_visibility_call = _optix(
  "visibility_call",
  {
    "slot": int,
    "world": int,
    "light": int,
    "point": wp.vec3,
    "normal": wp.vec3,
    "origin": wp.vec3,
    "seed": int,
    "stream": int,
  },
  wp.vec3,
)
optix_visibility_args = _optix("visibility_args", {"slot": int}, VisibilityArgs)
optix_visibility_result = _optix("visibility_result", {"v": wp.vec3}, None)
optix_layer_result = _optix(
  "layer_result",
  {
    "radiance": wp.vec3,
    "opacity": float,
    "reflected": float,
    "color": wp.vec3,
    "specular_only": int,
    "roughness": float,
    "base_color": wp.vec3,
  },
  None,
)
optix_shadow_attenuate = _optix("shadow_attenuate", {"instance": int, "t": wp.vec3}, None)
optix_payload_type_camera = _optix("payload_type_camera", {}, None)
optix_payload_type_shadow = _optix("payload_type_shadow", {}, None)
optix_payload_type_both = _optix("payload_type_both", {}, None)
optix_miss = _optix("miss", {}, None)
optix_payload_world = _optix("payload_world", {}, int)
optix_instance_id = _optix("instance_id", {}, int)
optix_triangle_barycentrics = _optix("triangle_barycentrics", {}, wp.vec2)
optix_object_ray_origin = _optix("object_ray_origin", {}, wp.vec3)
optix_object_ray_direction = _optix("object_ray_direction", {}, wp.vec3)
optix_ignore_intersection = _optix("ignore_intersection", {}, None)
optix_closesthit_triangle = _optix("closesthit_triangle", {"normal": wp.uint32, "instance": int}, None)
optix_object_to_world_normal = _optix("object_to_world_normal", {"n": wp.vec3}, wp.vec3)
optix_triangle_face_normal = _optix("triangle_face_normal", {}, wp.vec3)
optix_closesthit_custom = _optix("closesthit_custom", {}, None)
optix_ray_tmax = _optix("ray_tmax", {}, float)
optix_ray_mask = _optix("ray_mask", {}, wp.uint32)
optix_first_hit_ray = _optix("first_hit_ray", {}, wp.bool)
optix_primitive_index = _optix("primitive_index", {}, int)
optix_world_ray_origin = _optix("world_ray_origin", {}, wp.vec3)
optix_world_ray_direction = _optix("world_ray_direction", {}, wp.vec3)
optix_set_hit = _optix("set_hit", {"t": float, "instance": int, "primitive": int, "n": wp.vec3}, None)
optix_report_intersection = _optix("report_intersection", {"t": float, "n": wp.vec3, "instance": int}, wp.bool)
