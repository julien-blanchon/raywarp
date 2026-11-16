# Type stub of `native.py`: the signatures of our native builtins as kernels call them.
# Keep in step with the registrations there (and with the C++ of the owning preambles).

import ctypes

import warp as wp

class BvhNode(ctypes.Structure):
  """`mjpbr::BvhNode`: bounds plus children (inner node) or primitive slot range [left, right)
  (leaf)."""

  lower: wp.vec3
  left: int
  upper: wp.vec3
  right: int
  leaf: int

class MeshHit(ctypes.Structure):
  """`mjpbr::MeshHit`: a mesh query's closest hit (t < 0: none), mesh-frame unit normal."""

  t: float
  u: float
  v: float
  face: int
  normal: wp.vec3

class TraceResult(ctypes.Structure):
  """`mjpbr_optix::TraceResult`: distance, instance, primitive, barycentrics / normal, packed face
  normal."""

  t: float
  instance: int
  primitive: int
  a: float
  b: float
  c: float
  normal: wp.uint32

class LayerArgs(ctypes.Structure):
  """`mjpbr_optix::LayerArgs`: the layer shading call's arguments."""

  world: int
  instance: int
  dist: float
  normal: wp.vec3
  bary_u: float
  bary_v: float
  face: int
  mesh: int
  point: wp.vec3
  direction: wp.vec3
  cone: float
  spread: float
  headlight: wp.vec3
  seed: int
  stream: int

class VisibilityArgs(ctypes.Structure):
  """`mjpbr_optix::VisibilityArgs`: the light visibility call's arguments."""

  world: int
  light: int
  point: wp.vec3
  normal: wp.vec3
  origin: wp.vec3
  seed: int
  stream: int

class LayerResult(ctypes.Structure):
  """`mjpbr_optix::LayerResult`: the layer shading call's result."""

  radiance: wp.vec3
  opacity: float
  reflected: float
  color: wp.vec3
  specular_only: int
  roughness: float
  base_color: wp.vec3

# BVH node accessors (`core/trace/accel/nodes.py` preamble).
def bvh_lowers(bvh_id: wp.uint64 | int) -> wp.uint64:
  """Address of the lower node halves of the Warp BVH `bvh_id`."""

def bvh_uppers(bvh_id: wp.uint64 | int) -> wp.uint64:
  """Address of the upper node halves of the Warp BVH `bvh_id`."""

def bvh_prims(bvh_id: wp.uint64 | int) -> wp.uint64:
  """Address of the leaf-slot primitive indices of the Warp BVH `bvh_id`."""

def bvh_node_at(lowers: wp.uint64, uppers: wp.uint64, index: int) -> BvhNode:
  """Node `index` of the tree whose node halves are at `lowers` / `uppers`."""

def bvh_primitive_at(prims: wp.uint64, slot: int) -> int:
  """Primitive index at leaf slot `slot` of the tree whose indices are at `prims`."""

def mesh_closest(
  mesh_id: wp.uint64,
  halves: wp.uint64,
  start: wp.vec3,
  dir: wp.vec3,
  max_t: float,
  root: int,
  cull: bool,
  counts: wp.uint64 | int,
) -> MeshHit:
  """Closest hit before `max_t` on mesh `mesh_id` (packed children at `halves`) below node `root`
  (-1: the root); `cull`: back faces skipped; `counts`: traversal counters' address (0: none)."""

def mesh_any(
  mesh_id: wp.uint64, halves: wp.uint64, start: wp.vec3, dir: wp.vec3, max_t: float, root: int, counts: wp.uint64 | int
) -> bool:
  """Whether the ray hits mesh `mesh_id` (packed children at `halves`) below node `root` (-1: the
  root) before `max_t`; `counts`: traversal counters' address (0: none)."""

def mesh_max_nodes(mesh_id: wp.uint64) -> int:
  """Node capacity of mesh `mesh_id`'s BVH (0 for id 0)."""

def mesh_num_items(mesh_id: wp.uint64) -> int:
  """Triangle count of mesh `mesh_id`'s BVH (0 for id 0)."""

def mesh_pack_node(mesh_id: wp.uint64, halves: wp.uint64, index: int) -> None:
  """Writes inner node `index`'s packed children of mesh `mesh_id` at `halves` (nothing for a
  leaf) and the triangle of leaf slot `index` after them (nothing past the triangle count)."""

def bvh_max_nodes(bvh_id: wp.uint64) -> int:
  """Node capacity of the Warp BVH `bvh_id` (0 for id 0)."""

def mesh_max_depth(mesh_id: wp.uint64) -> int:
  """Recorded depth of mesh `mesh_id`'s BVH (0: none or unrecorded)."""

def bvh_max_depth(bvh_id: wp.uint64) -> int:
  """Recorded depth of the Warp BVH `bvh_id` (0: none or unrecorded)."""

def bvh_pack_node(bvh_id: wp.uint64, halves: wp.uint64, index: int) -> None:
  """Writes inner node `index`'s packed children of the Warp BVH `bvh_id` at `halves`."""

def packed_child(halves: wp.uint64, slot: int) -> BvhNode:
  """The packed child whose lower half is at `slot`: bounds, then a leaf's slot range or an inner
  node's own index (`left`)."""

# OptiX device intrinsics (`core/trace/optix/device/intrinsics.py` preamble).
def optix_launch_index() -> int:
  """The thread's launch index (x)."""

def optix_trace_closest(
  handle: wp.uint64 | int,
  o: wp.vec3,
  d: wp.vec3,
  tmin: float,
  tmax: float,
  ray_flags: wp.uint32 | int,
  mask: wp.uint32 | int,
  world: int,
) -> TraceResult:
  """Closest hit of the ray through the traversable `handle` (visibility `mask`, ray `world`)."""

def optix_trace_any(handle: wp.uint64 | int, o: wp.vec3, d: wp.vec3, tmin: float, tmax: float, mask: wp.uint32 | int) -> bool:
  """Whether anything is hit in (tmin, tmax) (terminates on the first hit, no hit programs)."""

def optix_trace_any_world(
  handle: wp.uint64 | int, o: wp.vec3, d: wp.vec3, tmin: float, tmax: float, mask: wp.uint32 | int, world: int
) -> bool:
  """`optix_trace_any` whose candidate hits run the alpha test's any-hit program (ray `world`)."""

def optix_miss() -> None:
  """Miss program: clears payload 0."""

def optix_payload_world() -> int:
  """The ray's world (payload 1), in hit programs."""

def optix_instance_id() -> int:
  """The hit instance's id."""

def optix_triangle_barycentrics() -> wp.vec2:
  """Barycentrics of the candidate triangle hit."""

def optix_object_ray_origin() -> wp.vec3:
  """Ray origin in the hit instance's object space."""

def optix_object_ray_direction() -> wp.vec3:
  """Ray direction in the hit instance's object space."""

def optix_trace_shadow(
  handle: wp.uint64,
  o: wp.vec3,
  d: wp.vec3,
  tmin: float,
  tmax: float,
  ray_flags: wp.uint32,
  mask: wp.uint32,
  world: int,
  t: wp.vec3,
) -> wp.vec3:
  """A shadow ray through translucent casters: the transmittance `t` after them, 0 if blocked
  (`ray_flags`: the occlusion queries' culling)."""

def optix_layer_call(
  world: int,
  instance: int,
  dist: float,
  normal: wp.vec3,
  bary_u: float,
  bary_v: float,
  face: int,
  mesh: int,
  point: wp.vec3,
  direction: wp.vec3,
  cone: float,
  spread: float,
  headlight: wp.vec3,
  seed: int,
  stream: int,
) -> LayerResult:
  """Shades one translucent layer hit in the layer program (a miss program of its own): the hit
  and the ray's state in, the layer's shading out."""

def optix_layer_args() -> LayerArgs:
  """Inside the layer program: the arguments of the `optix_layer_call` it serves."""

def optix_layer_result(
  radiance: wp.vec3,
  opacity: float,
  reflected: float,
  color: wp.vec3,
  specular_only: int,
  roughness: float,
  base_color: wp.vec3,
) -> None:
  """Inside the layer program: the layer's shading, returned to the `optix_layer_call`."""

def optix_visibility_call(
  slot: int,
  world: int,
  light: int,
  point: wp.vec3,
  normal: wp.vec3,
  origin: wp.vec3,
  seed: int,
  stream: int,
) -> wp.vec3:
  """The visibility (RGB) of light `light` from a shading point, computed by the light visibility
  program (a miss program of its own, payload type and miss record `slot`)."""

def optix_visibility_args(slot: int) -> VisibilityArgs:
  """Inside the light visibility program (payload type `slot`): the arguments of the
  `optix_visibility_call` it serves."""

def optix_visibility_result(v: wp.vec3) -> None:
  """Inside the light visibility program: the visibility, back to its `optix_visibility_call`."""

def optix_shadow_attenuate(instance: int, t: wp.vec3) -> None:
  """A translucent candidate on a shadow trace: multiplies its transmittance in, once per
  instance."""

def optix_payload_type_camera() -> None:
  """A hit / miss program's first statement: it serves payload type 0 (camera, occlusion)."""

def optix_payload_type_shadow() -> None:
  """A hit / miss program's first statement: it serves payload type 1 (shadow traces)."""

def optix_payload_type_both() -> None:
  """A hit / miss program's first statement: it serves both payload types."""

def optix_ignore_intersection() -> None:
  """Any-hit program: discards the candidate hit."""

def optix_closesthit_triangle(normal: wp.uint32 | int, instance: int) -> None:
  """Closest-hit program of triangles: writes the payload (instance, packed world face normal)."""

def optix_object_to_world_normal(n: wp.vec3) -> wp.vec3:
  """An object-space normal of the hit instance in world space."""

def optix_triangle_face_normal() -> wp.vec3:
  """Closest-hit / any-hit of a triangle: its unnormalized object-space normal cross(b - a, c - a)
  from the GAS's vertices (+z when degenerate)."""

def optix_closesthit_custom() -> None:
  """Closest-hit program of custom primitives: copies the reported attributes to the payload."""

def optix_ray_tmax() -> float:
  """The current hit distance (ray tmax)."""

def optix_ray_mask() -> wp.uint32:
  """The visibility mask the current ray was traced with."""

def optix_first_hit_ray() -> bool:
  """Whether the current ray ends at its first accepted hit (an occlusion or shadow trace)."""

def optix_primitive_index() -> int:
  """The hit primitive's index within its build input."""

def optix_world_ray_origin() -> wp.vec3:
  """Ray origin in world space."""

def optix_world_ray_direction() -> wp.vec3:
  """Ray direction in world space."""

def optix_set_hit(t: float, instance: int, primitive: int, n: wp.vec3) -> None:
  """Closest-hit payload of a curve hit: distance, instance, primitive, normal."""

def optix_report_intersection(t: float, n: wp.vec3, instance: int) -> bool:
  """Intersection program: reports a custom-primitive hit (normal and instance as attributes)."""
