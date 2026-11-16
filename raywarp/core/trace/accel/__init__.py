"""Software ray traversal (Warp BVHs), the trace backend available on every device.

- `trace.py`: the query types (`Ray`, `Hit`, geometry structs, `Tracer`, `TraceFeatures`).
- `query.py`: `make_trace` (analytic planes + scene BVHs), `software_tracer`, `module_options`.
- `cast.py` / `ordered.py`: unordered (MuJoCo Warp's) / nearest-first scene-BVH traversals.
- `instances.py`: what both traversals do with one BVH primitive (resolve, pose, intersect, merge).
- `nodes.py`: native BVH node accessors, traversal stacks, slab test.
- `primitives.py` / `meshes.py`: vendored analytic / mesh, height-field and flex intersections.
- `cutout.py`: the alpha test of cut-out materials at each hit.
- `bounds.py` / `bvh.py`: primitive world boxes and the scene BVHs (world tree, shared tree);
  `lbvh.py`: the world BVH's in-place rebuild; `packed.py`: the packed BVH2 children the queries
  read; `reach.py`: which per-world instances a world draws.
- `deformable_bvh.py`: the deformables' triangle BVHs, refit per frame.

Its `TraceBackend` is `core/trace/software_backend.py`'s `SoftwareBackend`.
"""

from __future__ import annotations
