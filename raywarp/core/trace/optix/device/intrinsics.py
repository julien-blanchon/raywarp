"""OptiX device intrinsics exposed to Warp kernels.

`optixTrace` is variadic and takes its payload registers by reference, so it is wrapped in fixed
payload layouts, together with the launch-index, ray and hit accessors the programs need. The glue
is a CUDA preamble that includes `<optix_device.h>`; `build_options` adds the OptiX SDK include
directory. `core/kernels/native.py` registers its functions as builtins (and mirrors
`TraceResult`); programs call them as `native.optix_<name>`.

Payload layout of a closest-hit trace (6 registers):
  p0: hit distance (float bits), p1: primitive index (on input the ray's world, plus
  `trace.SHARED_RAY` for the shared IAS, read by the hit programs), p2: the hit's instance
  (0xffffffff on a miss),
  p3..p4: triangle barycentrics (b1, b2) and p5 a mesh triangle's packed world face normal
  (`core/shade/materials/octahedral.py`); or p3..p5 the world-space normal of a curve or custom
  primitive. Every hit program writes the instance itself.
An occlusion trace reads p0: 1 if anything is hit, 0 after the miss program (with the alpha
test, `trace_any_world`, p1 carries the ray's world).
A shadow trace through translucent casters (`trace_shadow`: its own payload type, 1, and ray type,
1) sets p0 to 1, p1 the ray's world, p2 the last attenuating instance (-1), p3..p5 the RGB
transmittance so far: the shadow any-hit and intersection programs of translucent instances
multiply it once per instance and ignore the candidate (`shadow_attenuate`); an opaque candidate
is accepted and ends the trace (p0 stays set), the miss program clears p0 (unoccluded: p3..p5 the
transmittance). Every trace calls `optixTrace` with an SBT stride of `RAY_TYPES`.

The registers' OptiX payload semantics (`payload_semantics`: which program reads or writes each)
let OptiX keep only what each stage needs live: p3..p5 are outputs only. One payload type serves
the closest-hit and occlusion traces: OptiX resolves each program group to a single type, and the
miss program and the triangle hit group serve both (types per trace would need a record per ray
type in the shader binding table).

The layer shading call (`layer_call`, payload type 2) is no ray: a trace of the null traversable
runs the miss program `__miss__layer` (`programs.layer_program`), its arguments and result in the
payload registers. The light visibility call (`visibility_call`, the payload type after it,
`visibility_slot`) runs `__miss__light_visibility` (`programs.light_visibility_program`) so.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import warp as wp

from raywarp._native import OPTIX_INCLUDE_DIR
from raywarp.core.trace.optix import api

# Attribute registers of a custom-primitive hit (`report_intersection`: world normal xyz, instance).
NUM_ATTRIBUTE: Final = 4

# OptixPayloadSemantics bits (optix_types.h): (read, write) of the trace caller and the closest-hit,
# miss, any-hit and intersection programs.
_CALLER_R: Final = 1 << 0
_CALLER_W: Final = 2 << 0
_CH_R: Final = 1 << 2
_CH_W: Final = 2 << 2
_MS_R: Final = 1 << 4
_MS_W: Final = 2 << 4
_AH_R: Final = 1 << 6
_AH_W: Final = 2 << 6
_IS_R: Final = 1 << 8
_IS_W: Final = 2 << 8
_CALLER_RW: Final = _CALLER_R | _CALLER_W


# The ray types, each with its payload type (`OptixPayloadTypeID`): the camera / occlusion traces'
# and the shadow traces' through translucent casters (`trace_shadow`), each with its own SBT records
# (ray type 0 / 1 of a hit group: `RAY_TYPES` records per hit group, the traces' SBT stride,
# `pipeline.Pipeline`): the camera rays keep their narrow payload whatever the shadow rays carry.
RAY_TYPES: Final = 2


def payload_semantics() -> tuple[int, ...]:
  """Per-register OptiX payload semantics of the camera / occlusion traces' payload type (type 0,
  the layout above)."""
  out = _CALLER_R | _CH_W  # set by the closest-hit programs, read by the caller
  return (
    _CALLER_RW | _CH_W | _MS_W,  # p0: distance, or the occlusion flag the miss program clears
    _CALLER_RW | _CH_R | _CH_W | _AH_R | _IS_R,  # p1: the ray's world in, the primitive out
    _CALLER_RW | _CH_W,  # p2: instance (-1 kept on a miss)
    out,
    out,
    out,
  )


# The layer shading call's payload (type 2, `layer_call`): `LAYER_IN` registers in, the first
# `LAYER_OUT` of them overwritten with the result.
LAYER_IN: Final = 23
LAYER_OUT: Final = 13
# The layer shading program's miss record (after one per ray type).
LAYER_MISS: Final = RAY_TYPES


def layer_payload_semantics() -> tuple[int, ...]:
  """Payload semantics of the layer shading call (type 2, `layer_call`: a trace of the null
  traversable, which runs the miss program `__miss__layer` alone)."""
  both = _CALLER_RW | _MS_R | _MS_W
  return (both,) * LAYER_OUT + (_CALLER_W | _MS_R,) * (LAYER_IN - LAYER_OUT)


# The light visibility call's payload (`visibility_call`): `VISIBILITY_IN` registers in, the first
# `VISIBILITY_OUT` of them overwritten with the visibility (RGB).
VISIBILITY_IN: Final = 13
VISIBILITY_OUT: Final = 3


def visibility_slot(*, layer: bool) -> int:
  """The light visibility call's payload type and miss record index (equal: `RAY_TYPES` payload
  types and miss records first, then the layer program's when the build has it)."""
  return LAYER_MISS + int(layer)


def visibility_payload_semantics() -> tuple[int, ...]:
  """Payload semantics of the light visibility call (`visibility_call`: a trace of the null
  traversable, which runs the miss program `__miss__light_visibility` alone)."""
  both = _CALLER_RW | _MS_R | _MS_W
  return (both,) * VISIBILITY_OUT + (_CALLER_W | _MS_R,) * (VISIBILITY_IN - VISIBILITY_OUT)


def shadow_payload_semantics() -> tuple[int, ...]:
  """Payload semantics of the shadow traces' type (type 1, `trace_shadow`; no closest-hit
  program runs)."""
  attenuated = _CALLER_RW | _AH_R | _AH_W | _IS_R | _IS_W  # the transmittance so far
  return (
    _CALLER_RW | _MS_W,  # p0: the occlusion flag the miss program clears
    _CALLER_W | _AH_R | _IS_R,  # p1: the ray's world
    _CALLER_W | _AH_R | _AH_W | _IS_R | _IS_W,  # p2: the last attenuating instance
    attenuated,
    attenuated,
    attenuated,
  )


_PREAMBLE: Final = r"""
// Warp's NVRTC compile has no CUDA toolkit include path: include only the device API (optix.h
// pulls optix_host.h -> <cuda.h>) and skip cooperative vectors (they include <cuda_fp16.h>).
#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 0
#include <optix_device.h>

namespace mjpbr_optix {

// a, b: barycentrics (triangles) or a, b, c: world normal (curves, custom primitives); normal:
// p5 itself, a triangle's packed world face normal (`octahedral.unpack_unit`).
struct TraceResult {
  float t;
  int instance;
  int primitive;
  float a;
  float b;
  float c;
  unsigned int normal;
};

__device__ inline int launch_index() { return int(optixGetLaunchIndex().x); }

// `mask`: the instances whose visibility mask shares a bit with it (trace.*_BITS). `world`: the
// ray's world (p1), read by the alpha test's any-hit program when `ray_flags` lets it run.
__device__ inline TraceResult trace_closest(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax,
                                            wp::uint32 ray_flags, wp::uint32 mask, int world) {
  // p3..p5 are outputs only (`payload_semantics`): read after a hit.
  unsigned int p0 = 0u, p1 = (unsigned int)world, p2 = 0xffffffffu, p3, p4, p5;
  optixTrace(OPTIX_PAYLOAD_TYPE_ID_0, OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]),
             make_float3(d[0], d[1], d[2]), tmin, tmax, 0.0f, OptixVisibilityMask(mask), ray_flags, 0u, 2u, 0u,
             p0, p1, p2, p3, p4, p5);
  TraceResult r;
  r.t = __uint_as_float(p0);
  r.instance = int(p2);
  r.primitive = int(p1);
  r.a = __uint_as_float(p3);
  r.b = __uint_as_float(p4);
  r.c = __uint_as_float(p5);
  r.normal = p5;
  return r;
}

__device__ inline bool trace_any(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax,
                                 wp::uint32 mask) {
  // The payload type's six registers; only p0 is read. p1..p5 stay uninitialised: they reach no
  // program. The ray flags disable the closest-hit and any-hit programs; the intersection programs
  // that read p1 (shared custom primitives testing their per-world visibility) are traced only by
  // `trace_any_world` (a build with that test traces its shared scenery so: `scene_tracer`'s
  // `shared_anyhit`); the miss program writes p0 alone.
  unsigned int p0 = 1u, p1, p2, p3, p4, p5;  // the miss program clears p0
  optixTrace(OPTIX_PAYLOAD_TYPE_ID_0, OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]),
             make_float3(d[0], d[1], d[2]), tmin, tmax, 0.0f, OptixVisibilityMask(mask),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             0u, 2u, 0u, p0, p1, p2, p3, p4, p5);
  return p0 != 0u;
}

// An occlusion query whose candidate hits on cut-out instances run the alpha test's any-hit
// program (p1: the ray's world). One pass: splitting it into opaque instances first, then the
// any-hit ones, was 12-17 % slower on the cutouts scene with opacity micromaps.
__device__ inline bool trace_any_world(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax,
                                       wp::uint32 mask, int world) {
  unsigned int p0 = 1u, p1 = (unsigned int)world, p2, p3, p4, p5;
  optixTrace(OPTIX_PAYLOAD_TYPE_ID_0, OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]),
             make_float3(d[0], d[1], d[2]), tmin, tmax, 0.0f, OptixVisibilityMask(mask),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0u, 2u, 0u,
             p0, p1, p2, p3, p4, p5);
  return p0 != 0u;
}

// A shadow ray through translucent casters (module doc): the transmittance `t` after them, 0 when an
// opaque candidate ends it. Any-hit programs run (no DISABLE_ANYHIT): the instances' flags enable
// them on the translucent and cut-out ones only, so an opaque hit costs a plain occlusion trace.
__device__ inline wp::vec3 trace_shadow(wp::uint64 handle, wp::vec3 o, wp::vec3 d, float tmin, float tmax,
                                        wp::uint32 ray_flags, wp::uint32 mask, int world, wp::vec3 t) {
  unsigned int p0 = 1u, p1 = (unsigned int)world, p2 = 0xffffffffu;
  unsigned int p3 = __float_as_uint(t[0]), p4 = __float_as_uint(t[1]), p5 = __float_as_uint(t[2]);
  optixTrace(OPTIX_PAYLOAD_TYPE_ID_1, OptixTraversableHandle(handle), make_float3(o[0], o[1], o[2]),
             make_float3(d[0], d[1], d[2]), tmin, tmax, 0.0f, OptixVisibilityMask(mask),
             ray_flags | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 1u, 2u, 1u,
             p0, p1, p2, p3, p4, p5);
  if (p0 != 0u) return wp::vec3(0.0f);
  return wp::vec3(__uint_as_float(p3), __uint_as_float(p4), __uint_as_float(p5));
}

// A translucent candidate of `instance` (transmittance `t`) on a shadow trace: its transmittance
// multiplies the payload's once per instance (a closed pane's two faces, one instance's triangles
// met in one GAS traversal: consecutive calls).
__device__ inline void shadow_attenuate(int instance, wp::vec3 t) {
  if (optixGetPayload_2() == (unsigned int)instance) return;
  optixSetPayload_2((unsigned int)instance);
  optixSetPayload_3(__float_as_uint(__uint_as_float(optixGetPayload_3()) * t[0]));
  optixSetPayload_4(__float_as_uint(__uint_as_float(optixGetPayload_4()) * t[1]));
  optixSetPayload_5(__float_as_uint(__uint_as_float(optixGetPayload_5()) * t[2]));
}

// The layer shading call (`programs.layer_program`): the shading of a translucent layer behind
// the first surface runs in the miss program `__miss__layer`, a program of its own (OptiX compiles
// it as its own task, beside the ray-generation program), reached by a trace of the null
// traversable (no traversal: the miss program at once). Payload type 2: the arguments in 23
// registers, the result back in the first 13.
struct LayerArgs {
  int world;
  int instance;
  float dist;
  wp::vec3 normal;
  float bary_u;
  float bary_v;
  int face;
  int mesh;
  wp::vec3 point;
  wp::vec3 direction;
  float cone;
  float spread;
  wp::vec3 headlight;
  int seed;
  int stream;
};

struct LayerResult {
  wp::vec3 radiance;
  float opacity;
  float reflected;
  wp::vec3 color;
  int specular_only;
  float roughness;
  wp::vec3 base_color;
};

#define MJPBR_F(x) __float_as_uint(x)
#define MJPBR_I(x) ((unsigned int)(x))
#define MJPBR_AF(x) __uint_as_float(x)

__device__ inline LayerResult layer_call(int world, int instance, float dist, wp::vec3 normal, float bary_u,
                                         float bary_v, int face, int mesh, wp::vec3 point, wp::vec3 direction,
                                         float cone, float spread, wp::vec3 headlight, int seed, int stream) {
  unsigned int p0 = MJPBR_I(world), p1 = MJPBR_I(instance), p2 = MJPBR_F(dist), p3 = MJPBR_F(normal[0]),
               p4 = MJPBR_F(normal[1]), p5 = MJPBR_F(normal[2]), p6 = MJPBR_F(bary_u), p7 = MJPBR_F(bary_v),
               p8 = MJPBR_I(face), p9 = MJPBR_I(mesh), p10 = MJPBR_F(point[0]), p11 = MJPBR_F(point[1]),
               p12 = MJPBR_F(point[2]), p13 = MJPBR_F(direction[0]), p14 = MJPBR_F(direction[1]),
               p15 = MJPBR_F(direction[2]), p16 = MJPBR_F(cone), p17 = MJPBR_F(spread), p18 = MJPBR_F(headlight[0]),
               p19 = MJPBR_F(headlight[1]), p20 = MJPBR_F(headlight[2]), p21 = MJPBR_I(seed), p22 = MJPBR_I(stream);
  optixTrace(OPTIX_PAYLOAD_TYPE_ID_2, OptixTraversableHandle(0), make_float3(0.0f, 0.0f, 0.0f),
             make_float3(0.0f, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f, OptixVisibilityMask(0), OPTIX_RAY_FLAG_NONE, 0u, 2u,
             2u, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20,
             p21, p22);
  LayerResult r;
  r.radiance = wp::vec3(MJPBR_AF(p0), MJPBR_AF(p1), MJPBR_AF(p2));
  r.opacity = MJPBR_AF(p3);
  r.reflected = MJPBR_AF(p4);
  r.color = wp::vec3(MJPBR_AF(p5), MJPBR_AF(p6), MJPBR_AF(p7));
  r.specular_only = int(p8);
  r.roughness = MJPBR_AF(p9);
  r.base_color = wp::vec3(MJPBR_AF(p10), MJPBR_AF(p11), MJPBR_AF(p12));
  return r;
}

// In `__miss__layer`: its arguments, then its result.
__device__ inline LayerArgs layer_args() {
  optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_2);
  LayerArgs a;
  a.world = int(optixGetPayload_0());
  a.instance = int(optixGetPayload_1());
  a.dist = MJPBR_AF(optixGetPayload_2());
  a.normal = wp::vec3(MJPBR_AF(optixGetPayload_3()), MJPBR_AF(optixGetPayload_4()), MJPBR_AF(optixGetPayload_5()));
  a.bary_u = MJPBR_AF(optixGetPayload_6());
  a.bary_v = MJPBR_AF(optixGetPayload_7());
  a.face = int(optixGetPayload_8());
  a.mesh = int(optixGetPayload_9());
  a.point = wp::vec3(MJPBR_AF(optixGetPayload_10()), MJPBR_AF(optixGetPayload_11()), MJPBR_AF(optixGetPayload_12()));
  a.direction = wp::vec3(MJPBR_AF(optixGetPayload_13()), MJPBR_AF(optixGetPayload_14()), MJPBR_AF(optixGetPayload_15()));
  a.cone = MJPBR_AF(optixGetPayload_16());
  a.spread = MJPBR_AF(optixGetPayload_17());
  a.headlight = wp::vec3(MJPBR_AF(optixGetPayload_18()), MJPBR_AF(optixGetPayload_19()), MJPBR_AF(optixGetPayload_20()));
  a.seed = int(optixGetPayload_21());
  a.stream = int(optixGetPayload_22());
  return a;
}

__device__ inline void layer_result(wp::vec3 radiance, float opacity, float reflected, wp::vec3 color,
                                    int specular_only, float roughness, wp::vec3 base_color) {
  optixSetPayload_0(MJPBR_F(radiance[0]));
  optixSetPayload_1(MJPBR_F(radiance[1]));
  optixSetPayload_2(MJPBR_F(radiance[2]));
  optixSetPayload_3(MJPBR_F(opacity));
  optixSetPayload_4(MJPBR_F(reflected));
  optixSetPayload_5(MJPBR_F(color[0]));
  optixSetPayload_6(MJPBR_F(color[1]));
  optixSetPayload_7(MJPBR_F(color[2]));
  optixSetPayload_8(MJPBR_I(specular_only));
  optixSetPayload_9(MJPBR_F(roughness));
  optixSetPayload_10(MJPBR_F(base_color[0]));
  optixSetPayload_11(MJPBR_F(base_color[1]));
  optixSetPayload_12(MJPBR_F(base_color[2]));
}

// The light visibility call (`programs.light_visibility_program`): one light's shadow rays from a
// shading point run in the miss program `__miss__light_visibility`, a program of its own (its own
// OptiX compile task), reached by a trace of the null traversable. `slot`: its payload type's
// index and its miss record's (`LIGHT_VISIBILITY_SLOTS`: after the layer program's when the build
// has both). The arguments in 13 registers, the visibility back in the first 3.
struct VisibilityArgs {
  int world;
  int light;
  wp::vec3 point;
  wp::vec3 normal;
  wp::vec3 origin;
  int seed;
  int stream;
};

__device__ inline wp::vec3 visibility_call(int slot, int world, int light, wp::vec3 point, wp::vec3 normal,
                                           wp::vec3 origin, int seed, int stream) {
  unsigned int p0 = MJPBR_I(world), p1 = MJPBR_I(light), p2 = MJPBR_F(point[0]), p3 = MJPBR_F(point[1]),
               p4 = MJPBR_F(point[2]), p5 = MJPBR_F(normal[0]), p6 = MJPBR_F(normal[1]), p7 = MJPBR_F(normal[2]),
               p8 = MJPBR_F(origin[0]), p9 = MJPBR_F(origin[1]), p10 = MJPBR_F(origin[2]), p11 = MJPBR_I(seed),
               p12 = MJPBR_I(stream);
  optixTrace(OptixPayloadTypeID(1u << slot), OptixTraversableHandle(0), make_float3(0.0f, 0.0f, 0.0f),
             make_float3(0.0f, 0.0f, 1.0f), 0.0f, 0.0f, 0.0f, OptixVisibilityMask(0), OPTIX_RAY_FLAG_NONE, 0u, 2u,
             (unsigned int)slot, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
  return wp::vec3(MJPBR_AF(p0), MJPBR_AF(p1), MJPBR_AF(p2));
}

// In `__miss__light_visibility`: its arguments (the program serves payload type `slot`), then its
// result.
__device__ inline VisibilityArgs visibility_args(int slot) {
  optixSetPayloadTypes(OptixPayloadTypeID(1u << slot));
  VisibilityArgs a;
  a.world = int(optixGetPayload_0());
  a.light = int(optixGetPayload_1());
  a.point = wp::vec3(MJPBR_AF(optixGetPayload_2()), MJPBR_AF(optixGetPayload_3()), MJPBR_AF(optixGetPayload_4()));
  a.normal = wp::vec3(MJPBR_AF(optixGetPayload_5()), MJPBR_AF(optixGetPayload_6()), MJPBR_AF(optixGetPayload_7()));
  a.origin = wp::vec3(MJPBR_AF(optixGetPayload_8()), MJPBR_AF(optixGetPayload_9()), MJPBR_AF(optixGetPayload_10()));
  a.seed = int(optixGetPayload_11());
  a.stream = int(optixGetPayload_12());
  return a;
}

__device__ inline void visibility_result(wp::vec3 v) {
  optixSetPayload_0(MJPBR_F(v[0]));
  optixSetPayload_1(MJPBR_F(v[1]));
  optixSetPayload_2(MJPBR_F(v[2]));
}

#undef MJPBR_F
#undef MJPBR_I
#undef MJPBR_AF

// The payload types a hit or miss program serves (`optixSetPayloadTypes`, first in the program):
// OptiX compiles it for those only (`RAY_TYPES`: 0 camera / occlusion, 1 shadow).
__device__ inline void payload_type_camera() { optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0); }
__device__ inline void payload_type_shadow() { optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_1); }
__device__ inline void payload_type_both() { optixSetPayloadTypes(OPTIX_PAYLOAD_TYPE_ID_0 | OPTIX_PAYLOAD_TYPE_ID_1); }

__device__ inline void miss() { optixSetPayload_0(0u); }

// Any-hit program state: a candidate triangle hit.
__device__ inline int payload_world() { return int(optixGetPayload_1()); }

__device__ inline int instance_id() { return int(optixGetInstanceId()); }

__device__ inline wp::vec2 triangle_barycentrics() {
  const float2 b = optixGetTriangleBarycentrics();
  return wp::vec2(b.x, b.y);
}

__device__ inline wp::vec3 object_ray_origin() {
  const float3 v = optixGetObjectRayOrigin();
  return wp::vec3(v.x, v.y, v.z);
}

__device__ inline wp::vec3 object_ray_direction() {
  const float3 v = optixGetObjectRayDirection();
  return wp::vec3(v.x, v.y, v.z);
}

__device__ inline void ignore_intersection() { optixIgnoreIntersection(); }

// Triangle hit: distance, instance (decoded by the program), primitive, barycentrics and the
// packed world face normal (a mesh's; 0 for a box).
__device__ inline void closesthit_triangle(wp::uint32 normal, int instance) {
  const float2 b = optixGetTriangleBarycentrics();
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2((unsigned int)instance);
  optixSetPayload_3(__float_as_uint(b.x));
  optixSetPayload_4(__float_as_uint(b.y));
  optixSetPayload_5(normal);
}

// An object-space normal of the hit instance in world space.
__device__ inline wp::vec3 object_to_world_normal(wp::vec3 n) {
  const float3 w = optixTransformNormalFromObjectToWorldSpace(make_float3(n[0], n[1], n[2]));
  return wp::vec3(w.x, w.y, w.z);
}

// The current triangle hit's object-space normal cross(b - a, c - a) from its GAS's vertices
// (the software traversal's, `trace/accel/nodes.py`), +z when degenerate.
__device__ inline wp::vec3 triangle_face_normal() {
  float3 v[3];
  optixGetTriangleVertexData(v);
  const wp::vec3 a(v[0].x, v[0].y, v[0].z);
  const wp::vec3 n = wp::cross(wp::vec3(v[1].x, v[1].y, v[1].z) - a, wp::vec3(v[2].x, v[2].y, v[2].z) - a);
  if (n[0] == 0.0f && n[1] == 0.0f && n[2] == 0.0f) return wp::vec3(0.0f, 0.0f, 1.0f);
  return n;
}

__device__ inline void closesthit_custom() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2(optixGetAttribute_3());
  optixSetPayload_3(optixGetAttribute_0());
  optixSetPayload_4(optixGetAttribute_1());
  optixSetPayload_5(optixGetAttribute_2());
}

__device__ inline float ray_tmax() { return optixGetRayTmax(); }

// The visibility mask the ray was traced with (`trace.RAY_MASK_CULL`: a culling query).
__device__ inline wp::uint32 ray_mask() { return optixGetRayVisibilityMask(); }

// Whether the current ray ends at its first accepted hit (an occlusion or shadow trace).
__device__ inline bool first_hit_ray() { return (optixGetRayFlags() & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) != 0u; }

__device__ inline int primitive_index() { return int(optixGetPrimitiveIndex()); }

__device__ inline wp::vec3 world_ray_origin() {
  const float3 v = optixGetWorldRayOrigin();
  return wp::vec3(v.x, v.y, v.z);
}

__device__ inline wp::vec3 world_ray_direction() {
  const float3 v = optixGetWorldRayDirection();
  return wp::vec3(v.x, v.y, v.z);
}

// Closest-hit payload of a curve hit: distance, instance, primitive, world normal.
__device__ inline void set_hit(float t, int instance, int primitive, wp::vec3 n) {
  optixSetPayload_0(__float_as_uint(t));
  optixSetPayload_1((unsigned int)primitive);
  optixSetPayload_2((unsigned int)instance);
  optixSetPayload_3(__float_as_uint(n[0]));
  optixSetPayload_4(__float_as_uint(n[1]));
  optixSetPayload_5(__float_as_uint(n[2]));
}

// Custom-primitive hit (hit kind 0): world-space normal and instance as attributes.
__device__ inline bool report_intersection(float t, wp::vec3 n, int instance) {
  return optixReportIntersection(t, 0u, __float_as_uint(n[0]), __float_as_uint(n[1]), __float_as_uint(n[2]),
                                 (unsigned int)instance);
}

}  // namespace mjpbr_optix
"""


# The OptiX SDK headers bundled with raywarp at install (the build hook fetches NVIDIA/optix-dev at
# the release of `api.VERSION`).
INCLUDE_DIR: Final = OPTIX_INCLUDE_DIR


def include_dir() -> Path | None:
  """The bundled OptiX SDK include directory (holding optix_device.h), None when absent (an install
  without the build hook's output)."""
  return INCLUDE_DIR if (INCLUDE_DIR / "optix_device.h").exists() else None


_HEADER_VERSION: Final = re.compile(rb"^#define OPTIX_VERSION (\d+)", re.MULTILINE)


def header_version(inc: Path) -> tuple[int, int, int] | None:
  """(major, minor, micro) of the OptiX SDK headers in `inc` (`OPTIX_VERSION` of its optix.h), None
  when it has none."""
  try:
    found = _HEADER_VERSION.search((inc / "optix.h").read_bytes())
  except OSError:
    return None
  if found is None:
    return None
  n = int(found.group(1))
  return n // 10000, n % 10000 // 100, n % 100


def headers_reason(inc: Path) -> str | None:
  """None when the OptiX headers in `inc` are the release the binding mirrors (`api.VERSION`),
  else why not: the programs' device code would call another release's intrinsics (worded to
  follow "OptiX cannot run: ")."""
  found = header_version(inc)
  want = ".".join(map(str, api.VERSION))
  if found is None:
    return f"{inc}/optix.h declares no OPTIX_VERSION (the programs need the OptiX {want} SDK headers)"
  if found != api.VERSION:
    return f"the OptiX headers in {inc} are OptiX {'.'.join(map(str, found))}, the binding is OptiX {want}"
  return None


def build_options() -> wp.ModuleBuildOptions:
  """Module build options of a module holding OptiX programs.

  Raises:
    RuntimeError: when the bundled OptiX SDK headers are missing (`include_dir`).
  """
  inc = include_dir()
  if inc is None:
    raise RuntimeError(f"the OptiX SDK headers are not installed at {INCLUDE_DIR} (reinstall raywarp: its build fetches them)")
  return wp.ModuleBuildOptions(
    extra_cuda_include_dirs=[str(inc)],
    extra_cuda_preamble=_PREAMBLE,
  )
