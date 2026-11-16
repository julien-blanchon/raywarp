"""Direct access to Warp BVH nodes and traversal stacks (`cast.py`, `ordered.py`), and our mesh
BLAS queries (`native.mesh_closest` / `mesh_any`, `meshes.py`).

Our traversals read nodes through native accessors (`native.bvh_lowers` / `bvh_uppers` /
`bvh_prims` once per tree, then `bvh_node_at` / `bvh_primitive_at`, registered by
`core/kernels/native.py`); their C++ is this module's preamble, which every kernel inlining them
gets through `BUILD_OPTIONS` (`query.module_options`). The unordered walk uses Warp's depth-major
CUDA shared-stack layout for blocks <=128; CPU and larger blocks keep the local vector.

The private packed-node contract is verified for the Warp release families `VALIDATED_WARP`.
Kernel selection rejects other families before compilation; widening the set requires the
private-contract tests (every private Warp symbol raywarp uses, the node-access parity on CPU
and CUDA) and moving the package's warp-lang cap with it. Named C++ fields avoid copying a
descriptor layout into Python.

Both single-tree and two-tree scene queries use these accessors: Warp 1.18's public query can
overflow its stack on depth-32 SAH trees, so it is no alternative path.
"""

from __future__ import annotations

from typing import Final

import warp as wp

# The Warp release families (major.minor) whose private contracts were verified; the package caps
# warp-lang below the family after the last one.
# 1.19 is excluded: its float32 texture sampling is 36-46 % slower on sm_120 (warp#2258).
VALIDATED_WARP: Final = ("1.18",)


def require_compatible() -> None:
  """Raises RuntimeError for a warp-lang outside `VALIDATED_WARP` (the direct node traversal reads
  Warp's private BVH contracts), before a kernel selects it."""
  if ".".join(wp.__version__.split(".")[:2]) not in VALIDATED_WARP:
    families = ", ".join(f"{s}.x" for s in VALIDATED_WARP)
    raise RuntimeError(
      f"direct Warp BVH traversal requires a validated warp-lang ({families}); found {wp.__version__}. "
      "Install a validated Warp version (pip install raywarp resolves one)."
    )


_PREAMBLE: Final = r"""
// The mesh queries' stack depth (a test builds them with a smaller one to exercise a full stack).
#ifndef MJPBR_MESH_STACK
#define MJPBR_MESH_STACK BVH_QUERY_STACK_SIZE
#endif
// The mesh queries are called, not inlined, on the GPU: one stack frame (the 32-entry traversal
// stacks) serves every call site of a kernel instead of one per inlined copy (the per-thread local
// memory the driver reserves for every resident thread), and one copy of the loop is compiled.
#if defined(__CUDA_ARCH__)
#define MJPBR_QUERY static __noinline__
#else
#define MJPBR_QUERY inline
#endif
namespace mjpbr {
static_assert(sizeof(wp::BVHPackedNodeHalf) == 16,
              "raywarp requires the validated Warp packed BVH node contract");
// One active unordered Walk per thread. Its scene-candidate calls may enter Warp mesh queries,
// whose stacks have separate storage; no candidate recursively enters another scene Walk.
// Match Warp's depth-major shared layout, capped at 16 KiB. A public Warp BVH query in the
// same kernel then uses another <=16 KiB, leaving room below CUDA's 48 KiB static limit.
// At larger blocks our local fallback leaves that shared budget to upstream query storage.
CUDA_CALLABLE inline int& unordered_stack_slot(int* local, int depth) {
#if defined(__CUDA_ARCH__) && WP_TILE_BLOCK_DIM <= 128
  __shared__ int stack[32 * WP_TILE_BLOCK_DIM];
  const int thread = int(threadIdx.x) + int(blockDim.x)
      * (int(threadIdx.y) + int(blockDim.y) * int(threadIdx.z));
  return stack[depth * WP_TILE_BLOCK_DIM + thread];
#else
  return local[depth];
#endif
}
// One BVH node: bounds plus children (inner node) or primitive slot range [left, right) (leaf).
struct BvhNode {
  wp::vec3 lower;
  int left;
  wp::vec3 upper;
  int right;
  int leaf;
};

// A tree's descriptor fields, read once per traversed tree (not once per node: `bvh_get(id)` is a
// load through the id on every call, which the compiler does not hoist out of the loop).
CUDA_CALLABLE inline wp::uint64 bvh_lowers(wp::uint64 id) { return (wp::uint64)wp::bvh_get(id).node_lowers; }
CUDA_CALLABLE inline wp::uint64 bvh_uppers(wp::uint64 id) { return (wp::uint64)wp::bvh_get(id).node_uppers; }
CUDA_CALLABLE inline wp::uint64 bvh_prims(wp::uint64 id) { return (wp::uint64)wp::bvh_get(id).primitive_indices; }

CUDA_CALLABLE inline BvhNode bvh_node_at(wp::uint64 lowers, wp::uint64 uppers, int index) {
  const wp::BVHPackedNodeHalf lo = wp::bvh_load_node((const wp::BVHPackedNodeHalf*)lowers, index);
  const wp::BVHPackedNodeHalf hi = wp::bvh_load_node((const wp::BVHPackedNodeHalf*)uppers, index);
  BvhNode n;
  n.lower = wp::vec3(lo.x, lo.y, lo.z);
  n.left = int(lo.i);
  n.upper = wp::vec3(hi.x, hi.y, hi.z);
  n.right = int(hi.i);
  n.leaf = int(lo.b);
  return n;
}

CUDA_CALLABLE inline int bvh_primitive_at(wp::uint64 prims, int slot) { return ((const int*)prims)[slot]; }

// Mesh BLAS queries: Warp's `mesh_query_ray(_anyhit)` traversal (near child first, the same
// box tests and leaf order: hits identical) over packed children (below), with the watertight triangle test's per-ray shear
// computed once per query (Warp's `intersect_ray_tri_woop` recomputes it, with 2-3 divisions, for
// every triangle and indexes the vertices dynamically: local memory) and closest-hit stack entries
// skipped unread once a nearer hit is known.
//
// Stack bound: the stack holds at most one far sibling per inner node on the path to the current
// node, so a tree whose leaves are at most MJPBR_MESH_STACK (BVH_QUERY_STACK_SIZE, 32) levels
// below the root never fills it. Warp's CPU (SAH, median) and LBVH builders stop splitting at that depth; a deeper
// tree (another builder, degenerate input) would fill it, and the far child of a full stack is
// dropped (its subtree unvisited) while the walk goes on: Warp's query ends there, missing the
// near subtree and everything still pending too.
struct MeshHit {
  float t;  // < 0: no hit
  float u;
  float v;
  int face;
  wp::vec3 normal;  // unit, mesh frame
};

// Component k of v by selects (constant indices only: registers).
CUDA_CALLABLE inline float pick(const wp::vec3& v, int k) { return k == 0 ? v[0] : (k == 1 ? v[1] : v[2]); }

// The per-ray part of the watertight test (Woop, Benthin, Wald 2013): permuted axes and shear.
struct WoopRay {
  int kx;
  int ky;
  int kz;
  float sx;
  float sy;
  float sz;
};

CUDA_CALLABLE inline WoopRay woop_ray(const wp::vec3& dir) {
  WoopRay r;
  r.kz = wp::max_dim(dir);
  r.kx = r.kz + 1;
  if (r.kx == 3) r.kx = 0;
  r.ky = r.kx + 1;
  if (r.ky == 3) r.ky = 0;
  const float dz = pick(dir, r.kz);
  if (dz < 0.0f) {
    const int tmp = r.kx;
    r.kx = r.ky;
    r.ky = tmp;
  }
  r.sx = pick(dir, r.kx) / dz;
  r.sy = pick(dir, r.ky) / dz;
  r.sz = 1.0f / dz;
  return r;
}

// `intersect_ray_tri_woop` after its per-ray setup, the same arithmetic in the same order.
CUDA_CALLABLE inline bool woop_triangle(const WoopRay& r, const wp::vec3& p, const wp::vec3& a, const wp::vec3& b,
                                        const wp::vec3& c, float& t, float& u, float& v) {
  const wp::vec3 A = a - p;
  const wp::vec3 B = b - p;
  const wp::vec3 C = c - p;
  const float Akz = pick(A, r.kz);
  const float Bkz = pick(B, r.kz);
  const float Ckz = pick(C, r.kz);
  const float Ax = pick(A, r.kx) - r.sx * Akz;
  const float Ay = pick(A, r.ky) - r.sy * Akz;
  const float Bx = pick(B, r.kx) - r.sx * Bkz;
  const float By = pick(B, r.ky) - r.sy * Bkz;
  const float Cx = pick(C, r.kx) - r.sx * Ckz;
  const float Cy = pick(C, r.ky) - r.sy * Ckz;
  float U = wp::diff_product(Cx, By, Cy, Bx);
  float V = wp::diff_product(Ax, Cy, Ay, Cx);
  float W = wp::diff_product(Bx, Ay, By, Ax);
  if (U == 0.0f || V == 0.0f || W == 0.0f) {
    const double CxBy = (double)Cx * (double)By;
    const double CyBx = (double)Cy * (double)Bx;
    U = (float)(CxBy - CyBx);
    const double AxCy = (double)Ax * (double)Cy;
    const double AyCx = (double)Ay * (double)Cx;
    V = (float)(AxCy - AyCx);
    const double BxAy = (double)Bx * (double)Ay;
    const double ByAx = (double)By * (double)Ax;
    W = (float)(BxAy - ByAx);
  }
  if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f)) return false;
  const float det = U + V + W;
  if (det == 0.0f) return false;
  const float Az = r.sz * Akz;
  const float Bz = r.sz * Bkz;
  const float Cz = r.sz * Ckz;
  const float T = U * Az + V * Bz + W * Cz;
  const int det_sign = wp::sign_mask(det);
  if (wp::xorf(T, det_sign) < 0.0f) return false;
  const float rcp_det = 1.0f / det;
  u = U * rcp_det;
  v = V * rcp_det;
  t = T * rcp_det;
  return true;
}

// Packed BVH2 children (Aila-Laine 2009): for inner node i, the lower and upper halves of its
// left child, then of its right child, in 64 contiguous bytes at halves[4 i] (one cache line
// instead of four sectors in two arrays). A leaf child keeps Warp's halves (leaf bit, primitive
// slot range); an inner child's halves carry its own node index (`i` of the lower half), so the
// traversal's node reference (`bvh_query_node_pack` of the halves) is the index to read next.
CUDA_CALLABLE inline void pack_node(const wp::BVH& bvh, wp::uint64 halves, int index) {
  if (index >= bvh.num_nodes) return;
  const wp::BVHPackedNodeHalf lo = bvh.node_lowers[index];
  if (lo.b) return;
  wp::BVHPackedNodeHalf* out = (wp::BVHPackedNodeHalf*)halves + 4 * index;
  const int child[2] = {int(lo.i), int(bvh.node_uppers[index].i)};
  for (int k = 0; k < 2; ++k) {
    wp::BVHPackedNodeHalf l = bvh.node_lowers[child[k]];
    wp::BVHPackedNodeHalf u = bvh.node_uppers[child[k]];
    if (!l.b) {
      l.i = (unsigned int)child[k];
      u.i = 0u;
    }
    out[2 * k] = l;
    out[2 * k + 1] = u;
  }
}

// A mesh's triangles gathered in leaf-slot order after its packed children: slot s (the
// triangle `primitive_indices[s]`) is three 16-byte vertices (w unused) at packed_tris(...)[3 s],
// so a leaf reads its triangles contiguously with three 128-bit loads each instead of the slot's
// primitive index, its three vertex indices and three scattered vertices (two dependent loads);
// the primitive index is read only for a hit. Same vertices, same hits.
struct alignas(16) Vertex4 {
  float x;
  float y;
  float z;
  float w;
};

CUDA_CALLABLE inline const Vertex4* packed_tris(wp::uint64 halves, const wp::BVH& bvh) {
  return (const Vertex4*)(halves + (wp::uint64)64 * (wp::uint64)bvh.max_nodes);
}

CUDA_CALLABLE inline wp::vec3 vertex_of(const Vertex4& q) { return wp::vec3(q.x, q.y, q.z); }

// Node capacity, triangle count and packing (children of node `index`, triangle of slot `index`)
// of a mesh's BVH (id 0: none), or node capacity and packing of a plain Warp BVH.
CUDA_CALLABLE inline int mesh_max_nodes(wp::uint64 id) { return id == 0 ? 0 : wp::mesh_get(id).bvh.max_nodes; }
CUDA_CALLABLE inline int mesh_num_items(wp::uint64 id) { return id == 0 ? 0 : wp::mesh_get(id).bvh.num_items; }
CUDA_CALLABLE inline void mesh_pack_node(wp::uint64 id, wp::uint64 halves, int index) {
  if (id == 0) return;
  const wp::Mesh& mesh = wp::mesh_get(id);
  pack_node(mesh.bvh, halves, index);
  if (index < mesh.bvh.num_items) {
    const int prim = mesh.bvh.primitive_indices[index];
    Vertex4* out = (Vertex4*)packed_tris(halves, mesh.bvh) + 3 * index;
    for (int k = 0; k < 3; ++k) {
      const wp::vec3 p = mesh.points[mesh.indices[prim * 3 + k]];
      out[k].x = p[0];
      out[k].y = p[1];
      out[k].z = p[2];
      out[k].w = 0.0f;
    }
  }
}
CUDA_CALLABLE inline int bvh_max_nodes(wp::uint64 id) { return id == 0 ? 0 : wp::bvh_get(id).max_nodes; }
// The recorded depth of a mesh's BVH or a plain BVH (0: none, or not recorded by a build that
// kept the tree within BVH_QUERY_STACK_SIZE levels).
CUDA_CALLABLE inline int mesh_max_depth(wp::uint64 id) { return id == 0 ? 0 : wp::mesh_get(id).bvh.max_depth; }
CUDA_CALLABLE inline int bvh_max_depth(wp::uint64 id) { return id == 0 ? 0 : wp::bvh_get(id).max_depth; }
CUDA_CALLABLE inline void bvh_pack_node(wp::uint64 id, wp::uint64 halves, int index) {
  if (id != 0) pack_node(wp::bvh_get(id), halves, index);
}

// A packed child (its lower half at `slot`) as a BvhNode: bounds, then the slot range of a
// leaf or, for an inner node, its own index in `left`.
CUDA_CALLABLE inline BvhNode packed_child(wp::uint64 halves, int slot) {
  const wp::BVHPackedNodeHalf lo = wp::bvh_load_node((const wp::BVHPackedNodeHalf*)halves, slot);
  const wp::BVHPackedNodeHalf hi = wp::bvh_load_node((const wp::BVHPackedNodeHalf*)halves, slot + 1);
  BvhNode n;
  n.lower = wp::vec3(lo.x, lo.y, lo.z);
  n.left = int(lo.i);
  n.upper = wp::vec3(hi.x, hi.y, hi.z);
  n.right = int(hi.i);
  n.leaf = int(lo.b);
  return n;
}

// The traversal's reference of node `index`: a leaf's packed halves, an inner node's index.
CUDA_CALLABLE inline wp::uint64 node_ref(const wp::BVH& bvh, int index) {
  const wp::BVHPackedNodeHalf lo = wp::bvh_load_node(bvh.node_lowers, index);
  if (lo.b) return wp::bvh_query_node_pack(lo, wp::bvh_load_node(bvh.node_uppers, index));
  return (wp::uint64)index;
}

// One inner node's expansion (`node`: its index in the packed reference): both children's box
// tests before `limit` (Warp's `mesh_query_ray_intersect_aabb`), the nearer hit child first.
struct Expansion {
  wp::uint64 near;  // the child to visit next (hits >= 1)
  wp::uint64 far;   // the other hit child (hits == 2)
  float far_t;      // its box entry distance
  int hits;         // children hit: 0, 1 or 2
};

CUDA_CALLABLE inline Expansion expand(const wp::BVHPackedNodeHalf* packed, wp::uint64 node, const wp::vec3& start,
                                      const wp::vec3& dir, const wp::vec3& rcp_dir, bool fast_aabb, float limit) {
  const int index = 4 * wp::bvh_query_node_lower_payload(node);
  const wp::BVHPackedNodeHalf ll = wp::bvh_load_node(packed, index);
  const wp::BVHPackedNodeHalf lu = wp::bvh_load_node(packed, index + 1);
  const wp::BVHPackedNodeHalf rl = wp::bvh_load_node(packed, index + 2);
  const wp::BVHPackedNodeHalf ru = wp::bvh_load_node(packed, index + 3);
  float t0 = FLT_MAX;
  float t1 = FLT_MAX;
  const bool h0 = wp::mesh_query_ray_intersect_aabb(start, dir, rcp_dir, fast_aabb, wp::vec3(ll.x, ll.y, ll.z),
                                                    wp::vec3(lu.x, lu.y, lu.z), t0) && t0 < limit;
  const bool h1 = wp::mesh_query_ray_intersect_aabb(start, dir, rcp_dir, fast_aabb, wp::vec3(rl.x, rl.y, rl.z),
                                                    wp::vec3(ru.x, ru.y, ru.z), t1) && t1 < limit;
  const wp::uint64 ln = wp::bvh_query_node_pack(ll, lu);
  const wp::uint64 rn = wp::bvh_query_node_pack(rl, ru);
  const bool near_left = h0 && (!h1 || t0 < t1);
  Expansion e;
  e.near = near_left ? ln : rn;
  e.far = near_left ? rn : ln;
  e.far_t = near_left ? t1 : t0;
  e.hits = int(h0) + int(h1);
  return e;
}

// Pops the next pending node the ray enters before `limit` into `node` (entries beyond it are
// skipped unread); false: none left.
CUDA_CALLABLE inline bool pop_nearer(const wp::uint64* stack, const float* stack_t, int& count, float limit,
                                     wp::uint64& node) {
  while (count > 0) {
    --count;
    if (stack_t[count] < limit) {
      node = stack[count];
      return true;
    }
  }
  return false;
}

// Traversal counts of a counted query (`Features.counters` builds): `counts` is the address of
// three consecutive int64 counters of a contiguous row of the counters table (`context.py`): BLAS
// nodes expanded, triangles tested, dropped stack entries; or 0 (no counting: the counting code is
// compiled out of every call site passing a constant 0).
struct QueryCounts {
  int nodes = 0;
  int tris = 0;
  int drops = 0;
};

CUDA_CALLABLE inline void add_count(wp::uint64 address, int n) {
  if (n == 0) return;
#if defined(__CUDA_ARCH__)
  atomicAdd((unsigned long long*)address, (unsigned long long)(long long)n);
#else
  *(long long*)address += n;  // CPU kernels run one thread at a time
#endif
}

CUDA_CALLABLE inline void add_counts(wp::uint64 counts, const QueryCounts& c) {
  add_count(counts, c.nodes);
  add_count(counts + 8, c.tris);
  add_count(counts + 16, c.drops);
}

// Closest hit of the ray (start, dir) before max_t on mesh `id` (packed children and triangles at `halves`)
// below node `root` (-1: the root). `cull`: back faces (dot(dir, normal) >= 0) are skipped inside
// the leaves.
template <bool COUNT>
CUDA_CALLABLE MJPBR_QUERY MeshHit mesh_closest_t(wp::uint64 id, wp::uint64 halves, const wp::vec3& start,
                                                 const wp::vec3& dir, float max_t, int root, bool cull,
                                                 wp::uint64 counts) {
  const wp::BVHPackedNodeHalf* packed = (const wp::BVHPackedNodeHalf*)halves;
  const wp::Mesh mesh = wp::mesh_get(id);
  wp::uint64 stack[MJPBR_MESH_STACK];
  float stack_t[MJPBR_MESH_STACK];
  int count = 0;
  QueryCounts n;
  wp::uint64 node = node_ref(mesh.bvh, root == -1 ? *mesh.bvh.root : root);
  const wp::vec3 safe = wp::mesh_query_ray_safe_dir(dir);
  const wp::vec3 rcp_dir(1.0f / safe[0], 1.0f / safe[1], 1.0f / safe[2]);
  const bool fast_aabb = wp::mesh_query_ray_use_fast_aabb(dir);
  const WoopRay ray = woop_ray(dir);
  MeshHit hit;
  hit.t = max_t;
  hit.face = -1;
  hit.u = 0.0f;
  hit.v = 0.0f;
  wp::vec3 normal(0.0f);
  const Vertex4* tris = packed_tris(halves, mesh.bvh);
  while (true) {
    if (wp::bvh_query_node_is_leaf(node)) {
      const int end = wp::bvh_query_node_upper_payload(node);
      for (int pc = wp::bvh_query_node_lower_payload(node); pc < end; ++pc) {
        if (COUNT) ++n.tris;
        const wp::vec3 a = vertex_of(tris[3 * pc]);
        const wp::vec3 b = vertex_of(tris[3 * pc + 1]);
        const wp::vec3 c = vertex_of(tris[3 * pc + 2]);
        float t, u, v;
        if (woop_triangle(ray, start, a, b, c, t, u, v) && t < hit.t && t >= 0.0f) {
          const wp::vec3 nrm = wp::cross(b - a, c - a);
          if (!cull || wp::dot(dir, nrm) < 0.0f) {
            hit.t = t;
            hit.u = u;
            hit.v = v;
            hit.face = pc;  // the slot; its primitive below
            normal = nrm;
          }
        }
      }
      if (!pop_nearer(stack, stack_t, count, hit.t, node)) break;
      continue;
    }
    if (COUNT) ++n.nodes;
    const Expansion e = expand(packed, node, start, dir, rcp_dir, fast_aabb, hit.t);
    if (e.hits == 2) {
      if (count < MJPBR_MESH_STACK) {  // full (a tree deeper than 32 levels, see above): the far child is dropped
        stack[count] = e.far;
        stack_t[count] = e.far_t;
        ++count;
      } else if (COUNT) {
        ++n.drops;
      }
      node = e.near;
    } else if (e.hits == 1) {
      node = e.near;
    } else {
      if (!pop_nearer(stack, stack_t, count, hit.t, node)) break;
    }
  }
  if (COUNT) add_counts(counts, n);
  if (hit.face < 0) {
    hit.t = -1.0f;
    hit.normal = wp::vec3(0.0f);
  } else {
    hit.face = mesh.bvh.primitive_indices[hit.face];
    hit.normal = wp::normalize(normal);
  }
  return hit;
}

// Whether the ray hits mesh `id` below node `root` (-1: the root) before max_t (shadow rays).
template <bool COUNT>
CUDA_CALLABLE MJPBR_QUERY bool mesh_any_t(wp::uint64 id, wp::uint64 halves, const wp::vec3& start, const wp::vec3& dir,
                                          float max_t, int root, wp::uint64 counts) {
  const wp::BVHPackedNodeHalf* packed = (const wp::BVHPackedNodeHalf*)halves;
  const wp::Mesh mesh = wp::mesh_get(id);
  wp::uint64 stack[MJPBR_MESH_STACK];
  int count = 0;
  QueryCounts n;
  wp::uint64 node = node_ref(mesh.bvh, root == -1 ? *mesh.bvh.root : root);
  const wp::vec3 safe = wp::mesh_query_ray_safe_dir(dir);
  const wp::vec3 rcp_dir(1.0f / safe[0], 1.0f / safe[1], 1.0f / safe[2]);
  const bool fast_aabb = wp::mesh_query_ray_use_fast_aabb(dir);
  const WoopRay ray = woop_ray(dir);
  const Vertex4* tris = packed_tris(halves, mesh.bvh);
  while (true) {
    if (wp::bvh_query_node_is_leaf(node)) {
      const int end = wp::bvh_query_node_upper_payload(node);
      for (int pc = wp::bvh_query_node_lower_payload(node); pc < end; ++pc) {
        if (COUNT) ++n.tris;
        const wp::vec3 a = vertex_of(tris[3 * pc]);
        const wp::vec3 b = vertex_of(tris[3 * pc + 1]);
        const wp::vec3 c = vertex_of(tris[3 * pc + 2]);
        float t, u, v;
        if (woop_triangle(ray, start, a, b, c, t, u, v) && t < max_t && t >= 0.0f) {
          if (COUNT) add_counts(counts, n);
          return true;
        }
      }
      if (count == 0) break;
      node = stack[--count];
      continue;
    }
    if (COUNT) ++n.nodes;
    const Expansion e = expand(packed, node, start, dir, rcp_dir, fast_aabb, max_t);
    if (e.hits == 2) {
      if (count < MJPBR_MESH_STACK) {
        stack[count++] = e.far;
      } else if (COUNT) {
        ++n.drops;  // full: the far child is dropped
      }
      node = e.near;
    } else if (e.hits == 1) {
      node = e.near;
    } else {
      if (count == 0) break;
      node = stack[--count];
    }
  }
  if (COUNT) add_counts(counts, n);
  return false;
}

// The queries: `counts` 0 (a constant at every uncounted call site, which removes the counting
// copy) or the address of the counters (`QueryCounts`).
CUDA_CALLABLE inline MeshHit mesh_closest(wp::uint64 id, wp::uint64 halves, const wp::vec3& start, const wp::vec3& dir,
                                          float max_t, int root, bool cull, wp::uint64 counts) {
  if (counts != 0) return mesh_closest_t<true>(id, halves, start, dir, max_t, root, cull, counts);
  return mesh_closest_t<false>(id, halves, start, dir, max_t, root, cull, 0);
}

CUDA_CALLABLE inline bool mesh_any(wp::uint64 id, wp::uint64 halves, const wp::vec3& start, const wp::vec3& dir,
                                   float max_t, int root, wp::uint64 counts) {
  if (counts != 0) return mesh_any_t<true>(id, halves, start, dir, max_t, root, counts);
  return mesh_any_t<false>(id, halves, start, dir, max_t, root, 0);
}
// A resumable closest-hit walk of one mesh BLAS for alpha-tested meshes: `mesh_walk_next` stops at
// every candidate triangle hit before the walk's limit (in nearest-node-first order, as
// `mesh_closest`) and hands it to the caller, which tests the triangle's texel and either ignores
// it or accepts it (a closest query then lowers `limit` to its distance; an any-hit query stops).
// One traversal visits each node once, however many cut-out layers the ray crosses. The walk's
// state is a Warp struct W (`nodes.MeshWalk`): its pending nodes and their entry distances, the
// current node (MESH_WALK_NONE: pop next) and leaf slots, the limit, and the candidate (t, u, v,
// primitive, mesh-frame unit normal).
constexpr wp::uint64 MESH_WALK_NONE = ~wp::uint64(0);

// The walk holds `nodes.STACK` (32) pending nodes: a smaller (test) stack fits, a larger one does not.
static_assert(MJPBR_MESH_STACK <= 32, "MeshWalk holds 32 pending nodes");

template <class W>
CUDA_CALLABLE inline void mesh_walk_start(W& w, wp::uint64 id, float max_t, int root) {
  const wp::Mesh& mesh = wp::mesh_get(id);
  w.count = 0;
  w.node = node_ref(mesh.bvh, root == -1 ? *mesh.bvh.root : root);
  w.slot = 0;
  w.slot_end = 0;
  w.limit = max_t;
}

template <class W, bool COUNT>
CUDA_CALLABLE MJPBR_QUERY bool mesh_walk_next_t(W& w, wp::uint64 id, wp::uint64 halves, const wp::vec3& start,
                                                const wp::vec3& dir, bool cull, wp::uint64 counts) {
  const wp::BVHPackedNodeHalf* packed = (const wp::BVHPackedNodeHalf*)halves;
  const wp::Mesh& mesh = wp::mesh_get(id);
  const wp::vec3 safe = wp::mesh_query_ray_safe_dir(dir);
  const wp::vec3 rcp_dir(1.0f / safe[0], 1.0f / safe[1], 1.0f / safe[2]);
  const bool fast_aabb = wp::mesh_query_ray_use_fast_aabb(dir);
  const WoopRay ray = woop_ray(dir);
  const Vertex4* tris = packed_tris(halves, mesh.bvh);
  const float limit = w.limit;
  QueryCounts n;
  int count = w.count;
  wp::uint64 node = w.node;
  int pc = w.slot;
  int end = w.slot_end;
  bool found = false;
  while (true) {
    if (pc < end) {
      for (; pc < end; ++pc) {
        if (COUNT) ++n.tris;
        const wp::vec3 a = vertex_of(tris[3 * pc]);
        const wp::vec3 b = vertex_of(tris[3 * pc + 1]);
        const wp::vec3 c = vertex_of(tris[3 * pc + 2]);
        float t, u, v;
        if (woop_triangle(ray, start, a, b, c, t, u, v) && t < limit && t >= 0.0f) {
          const wp::vec3 nrm = wp::cross(b - a, c - a);
          if (!cull || wp::dot(dir, nrm) < 0.0f) {
            w.t = t;
            w.u = u;
            w.v = v;
            w.face = mesh.bvh.primitive_indices[pc];
            w.normal = wp::normalize(nrm);
            found = true;
            ++pc;
            break;
          }
        }
      }
      if (found) break;
      continue;
    }
    if (node == MESH_WALK_NONE) {
      // Pending nodes entered beyond the limit are skipped unread (`pop_nearer`).
      bool popped = false;
      while (count > 0) {
        --count;
        if (w.stack_t.c[count] < limit) {
          node = w.stack.c[count];
          popped = true;
          break;
        }
      }
      if (!popped) break;
    }
    if (wp::bvh_query_node_is_leaf(node)) {
      pc = wp::bvh_query_node_lower_payload(node);
      end = wp::bvh_query_node_upper_payload(node);
      node = MESH_WALK_NONE;
      continue;
    }
    if (COUNT) ++n.nodes;
    const Expansion e = expand(packed, node, start, dir, rcp_dir, fast_aabb, limit);
    if (e.hits == 2) {
      if (count < MJPBR_MESH_STACK) {  // full: the far child is dropped (see `mesh_closest_t`)
        w.stack.c[count] = e.far;
        w.stack_t.c[count] = e.far_t;
        ++count;
      } else if (COUNT) {
        ++n.drops;
      }
      node = e.near;
    } else if (e.hits == 1) {
      node = e.near;
    } else {
      node = MESH_WALK_NONE;
    }
  }
  w.count = count;
  w.node = node;
  w.slot = pc;
  w.slot_end = end;
  if (COUNT) add_counts(counts, n);
  return found;
}

template <class W>
CUDA_CALLABLE inline bool mesh_walk_next(W& w, wp::uint64 id, wp::uint64 halves, const wp::vec3& start,
                                         const wp::vec3& dir, bool cull, wp::uint64 counts) {
  if (counts != 0) return mesh_walk_next_t<W, true>(w, id, halves, start, dir, cull, counts);
  return mesh_walk_next_t<W, false>(w, id, halves, start, dir, cull, 0);
}
}  // namespace mjpbr
"""

# Module option of every kernel that inlines a traversal reading nodes directly.
BUILD_OPTIONS: Final = wp.ModuleBuildOptions(extra_cuda_preamble=_PREAMBLE, extra_cpu_preamble=_PREAMBLE)

# Pending siblings for a traced subtree of depth <= 32. The current child is held separately;
# pushing both children would need 33 slots at Warp's SAH builder's deepest permitted leaf.
STACK: Final = 32
IntStack = wp.types.vector(length=STACK, dtype=int)
FloatStack = wp.types.vector(length=STACK, dtype=float)
NodeStack = wp.types.vector(length=STACK, dtype=wp.uint64)


@wp.struct
class MeshWalk:
  """State of a resumable mesh-BLAS walk (`mesh_walk_start` / `mesh_walk_next`, C++ above): pending
  node references and entry distances, the current node and leaf slots, the limit (the walk skips
  hits at or beyond it; a closest query lowers it to each accepted hit), and the last candidate:
  distance `t`, barycentrics (u, v) as `native.mesh_closest`'s, primitive `face` and mesh-frame
  unit `normal`."""

  stack: NodeStack
  stack_t: FloatStack
  count: int
  node: wp.uint64
  slot: int
  slot_end: int
  limit: float
  t: float
  u: float
  v: float
  face: int
  normal: wp.vec3


@wp.func_native("mjpbr::mesh_walk_start(walk, mesh_id, max_t, root);")
def mesh_walk_start(
  # Out:
  walk: wp.ref[MeshWalk],
  # In:
  mesh_id: wp.uint64,
  max_t: float,
  root: int,
) -> None:
  """Starts `walk` at node `root` (-1: the root) of mesh `mesh_id` with limit `max_t`."""
  ...


@wp.func_native("return mjpbr::mesh_walk_next(walk, mesh_id, halves, start, dir, cull, counts);")
def mesh_walk_next(
  # Out:
  walk: wp.ref[MeshWalk],
  # In:
  mesh_id: wp.uint64,
  halves: wp.uint64,
  start: wp.vec3,
  dir: wp.vec3,
  cull: bool,
  counts: wp.uint64,
) -> bool:
  """The next candidate triangle hit before `walk.limit` (in `walk.t` ...), False when none is left;
  `cull`: back faces skipped; `counts`: traversal counters' address (0: none)."""
  ...


@wp.func_native("return mjpbr::unordered_stack_slot(stack.c, i);")
def unordered_int_at(
  # In:
  stack: wp.ref[IntStack],
  i: int,
) -> int:
  """Entry `i` of the active unordered walk's stack (shared memory on CUDA blocks <= 128, else
  `stack` itself: local)."""
  ...


@wp.func_native("mjpbr::unordered_stack_slot(stack.c, i) = value;")
def unordered_store(
  # Out:
  stack: wp.ref[IntStack],
  # In:
  i: int,
  value: int,
) -> None:
  """Writes entry `i` of the active unordered walk's stack (storage as `unordered_int_at`): every
  entry is written before it is read, so the shared scratch needs no initialisation."""
  ...


# Reading an entry of a stack field (`s.node[s.count]`, `s` a `wp.ref` struct) makes Warp load the
# whole 32-entry vector into a temporary before extracting it: on GPU that copy lives in local
# memory (+400 B per thread) and makes the traversal 38-60% slower. These read the entry in place;
# writes (`s.node[i] = x`) are in place already.
@wp.func_native("return stack.c[i];")
def int_at(
  # In:
  stack: wp.ref[IntStack],
  i: int,
) -> int:
  """Entry `i` of `stack`, read in place (no copy of the vector)."""
  ...


@wp.func_native("stack.c[i] = value;")
def int_store(
  # Out:
  stack: wp.ref[IntStack],
  # In:
  i: int,
  value: int,
) -> None:
  """Writes entry `i` (0 <= i < STACK) in place, without Warp's negative-index wrap."""
  ...


@wp.func_native("stack.c[i] = value;")
def float_store(
  # Out:
  stack: wp.ref[FloatStack],
  # In:
  i: int,
  value: float,
) -> None:
  """Writes entry `i` (0 <= i < STACK) in place, without Warp's negative-index wrap."""
  ...


@wp.func_native("return stack.c[i];")
def float_at(
  # In:
  stack: wp.ref[FloatStack],
  i: int,
) -> float:
  """Entry `i` of `stack`, read in place (no copy of the vector)."""
  ...


@wp.func_native(
  "return (wp::uint64)counters.data + (wp::uint64)((worldid % counters.shape[0]) * counters.strides[0]"
  " + column * counters.strides[1]);"
)
def counter_address(
  # In:
  counters: wp.array2d[wp.int64],
  worldid: int,
  column: int,
) -> wp.uint64:
  """Address of world `worldid`'s counter `column` (`kernels/abi/counters.py`), what a counted mesh
  query adds its counts to."""
  ...


@wp.func
def ray_aabb(
  # In:
  origin: wp.vec3,
  rcp_dir: wp.vec3,
  lower: wp.vec3,
  upper: wp.vec3,
  max_t: float,
) -> float:
  """Entry distance (>= 0) of the ray into the closed box, or -1 if missed or beyond max_t.

  Slab test with the near / far face picked by the direction's sign (Williams et al. 2005): the
  ray is inside all three slabs on [tmin, tmax]. An axis the ray does not move along (`rcp_dir`
  +-inf, `reciprocal`) gives -inf / +inf inside its slab and an empty interval outside it; on a
  face of that slab, 0 x inf = NaN, which Warp's min / max drop (C fmin / fmax): the slab
  constrains nothing, so a ray lying in a box face still enters the box (picking the faces by
  min / max of both products made that axis +inf / +inf or -inf / -inf instead: a miss)."""
  near = wp.vec3(
    wp.where(rcp_dir[0] >= 0.0, lower[0], upper[0]),
    wp.where(rcp_dir[1] >= 0.0, lower[1], upper[1]),
    wp.where(rcp_dir[2] >= 0.0, lower[2], upper[2]),
  )
  far = wp.vec3(
    wp.where(rcp_dir[0] >= 0.0, upper[0], lower[0]),
    wp.where(rcp_dir[1] >= 0.0, upper[1], lower[1]),
    wp.where(rcp_dir[2] >= 0.0, upper[2], lower[2]),
  )
  t_near = wp.cw_mul(near - origin, rcp_dir)
  t_far = wp.cw_mul(far - origin, rcp_dir)
  tmin = wp.max(wp.max(t_near[0], t_near[1]), t_near[2])
  tmax = wp.min(wp.min(t_far[0], t_far[1]), t_far[2])
  entry = wp.max(tmin, 0.0)
  if tmax >= entry and entry < max_t:
    return entry
  return -1.0


@wp.func
def reciprocal(
  # In:
  direction: wp.vec3,
) -> wp.vec3:
  """Per-component 1 / direction (inf on zero components, as the slab test expects)."""
  return wp.vec3(1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2])
