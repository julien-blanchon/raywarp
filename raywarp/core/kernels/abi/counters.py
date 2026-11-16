"""Light-cost counters (`performance.counters`): how many samples, surfaces, lights and rays a
frame's shading used, per world, to see where render time goes.

A debug build (`Features.counters`) adds per-world events to `GeometryFields.counters`
(nworld, NCOUNTER) int64, zeroed before each frame. `lights`/`hit_lights` count candidate entries
sent to the evaluator, BEFORE active/range/backface/zero-term rejection. `cheap` counts
entries considered by light-cap ranking. Neither count is a BRDF/LTC execution count.

`light_terms`/`hit_light_terms` count evaluated unshadowed terms with positive squared
magnitude; the corresponding `*_casters` count those whose evaluator reports cast-shadow.
These are before visibility, sun-only selection and shadow budgeting; they are NOT shadow
queries. Capped diffuse remainder, ambient/environment and headlight are not counted as terms.
`hit_*` describes glossy secondary lighting (MuJoCo mirrors included: their reflection is the
glossy ray). Probe preparation can call that loop too, but frame counters are reset
before rendering, so its earlier preparation work is not a frame total.

Reflection rays include glossy and mirror. Ray queries are counted separately by kind.
`Renderer.counters` divides world totals by camera pixels, not by shaded samples. Builds
without counters contain none of these increments; counter-enabled runs are not timing runs.
OptiX any-hit callback counts include triangle alpha/visibility candidates and shared-curve
visibility candidates. The corresponding ignored columns count callbacks that reject a hit.
They are callback invocations, not unique rays or traversal time. Software tracing does not
execute these callbacks, so those columns remain zero there.

The software traversal counts its own work (zero under OptiX): scene-BVH walks by kind (an
alpha-tested query that continues past cut-out texels walks again), scene-BVH nodes tested,
instances intersected (BVH candidates), mesh BLAS inner nodes expanded and triangles tested, and
stack entries dropped by a full traversal stack (subtrees never visited: must stay 0).
The table remains allocated when disabled: 31 columns x 8 bytes = 248 bytes per world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import warp as wp

from raywarp.core.kernels.abi.rays import GeometryFields, RayQuery
from raywarp.core.kernels.abi.tables import table_row
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.pipeline.platform.device import host_copy

if TYPE_CHECKING:
  from shape_extensions import IntVar

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Columns of `GeometryFields.counters`.
SAMPLES: Final = 0  # camera-ray samples
LAYERS: Final = 1  # surfaces shaded along camera rays (first hits and translucent layers)
LIGHTS: Final = 2  # candidate light entries at camera/mirror surfaces
CHEAP: Final = 3  # lights ranked by their cheap estimate (light caps) at camera-ray surfaces
HIT_LIGHTS: Final = 4  # candidate light entries at glossy-reflection hits
HIT_CHEAP: Final = 5  # lights ranked cheaply at glossy-reflection hits
REFLECTION: Final = 6  # reflection rays (glossy and mirror)
CLOSEST: Final = 7  # closest-hit queries (camera, layers, reflections)
SHADOW: Final = 8  # shadow queries (any-hit or through translucent casters)
LIGHT_TERMS: Final = 9  # nonzero unshadowed terms at camera/mirror surfaces
LIGHT_CASTERS: Final = 10  # those terms whose evaluator reports cast-shadow
HIT_LIGHT_TERMS: Final = 11  # nonzero unshadowed terms at glossy-reflection hits
HIT_LIGHT_CASTERS: Final = 12  # those terms whose evaluator reports cast-shadow
TRIANGLE_ANYHIT: Final = 13  # OptiX triangle any-hit callback invocations
TRIANGLE_ANYHIT_IGNORED: Final = 14  # triangle candidates rejected by visibility/alpha
CURVE_ANYHIT: Final = 15  # OptiX shared-curve any-hit callback invocations
CURVE_ANYHIT_IGNORED: Final = 16  # shared-curve candidates rejected by visibility
PLANE_ANY_TERMINATED: Final = 17  # analytic plane ended an OptiX any-hit query
SHARED_ANY_CALLS: Final = 18  # shared IAS any-hit optixTrace calls
SHARED_ANY_HITS: Final = 19  # shared IAS any-hit calls that found a blocker
WORLD_CELL_REJECTED: Final = 20  # shared miss, but ray segment misses the selected world's cell
WORLD_ANY_CALLS: Final = 21  # per-world IAS any-hit optixTrace calls
WORLD_ANY_HITS: Final = 22  # per-world IAS any-hit calls that found a blocker
TRANSLUCENT_PASSES: Final = 23  # translucent casters a shadow ray passed (their transmittance in)
SCENE_CLOSEST_WALKS: Final = 24  # software closest-hit scene-BVH walks
SCENE_ANY_WALKS: Final = 25  # software any-hit scene-BVH walks
SCENE_NODES: Final = 26  # software scene-BVH nodes tested against the ray
INSTANCE_TESTS: Final = 27  # software instances intersected (scene-BVH candidates)
BLAS_NODES: Final = 28  # software mesh-BLAS inner nodes expanded (BLAS_NODES..STACK_DROPS: consecutive)
BLAS_TRIANGLES: Final = 29  # software mesh-BLAS triangles tested
STACK_DROPS: Final = 30  # software traversal stack entries dropped by a full stack (scene and BLAS)
NAMES: Final = (
  "samples",
  "layers",
  "lights",
  "cheap",
  "hit_lights",
  "hit_cheap",
  "reflection_rays",
  "closest_rays",
  "shadow_rays",
  "light_terms",
  "light_casters",
  "hit_light_terms",
  "hit_light_casters",
  "triangle_anyhit_candidates",
  "triangle_anyhit_ignored",
  "curve_anyhit_candidates",
  "curve_anyhit_ignored",
  "plane_any_terminated",
  "shared_any_calls",
  "shared_any_hits",
  "world_cell_rejected",
  "world_any_calls",
  "world_any_hits",
  "translucent_passes",
  "scene_closest_walks",
  "scene_any_walks",
  "scene_nodes",
  "instance_tests",
  "blas_nodes",
  "blas_triangles",
  "stack_drops",
)
NCOUNTER: Final = len(NAMES)


@wp.func
@device_generic
def add[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: int,
  column: int,
  n: int,
):
  """Adds `n` events of kind `column` to world `worldid`'s counters."""
  wp.atomic_add(geo.counters, table_row(worldid, geo.counters.shape[0]), column, wp.int64(n))


@wp.func
@device_generic
def term[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  worldid: int,
  column: int,
  radiance: wp.vec3,
  caster: bool,
):
  """Counts a nonzero evaluated term (and, for a shadow caster, its caster column) before
  visibility."""
  if wp.length_sq(radiance) > 0.0:
    add(geo, worldid, column, 1)
    if caster:
      add(geo, worldid, column + 1, 1)


@specialize
def counted[R](query: RayQuery[GeometryFields, R], column: int):
  """`query` (a ray query: closest hit or shadow) counting its calls in counter `column`."""

  @wp.func
  @device_generic
  def counted_query[G: GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ):
    """`query`, counted."""
    add(geo, worldid, wp.static(column), 1)
    return query(geo, worldid, origin, direction, max_dist)

  return counted_query


def per_pixel[NWorld: IntVar, NCounter: IntVar](
  counters: wp.array[wp.int64, [NWorld, NCounter]], pixels: int
) -> dict[str, float]:
  """Counter name -> mean per camera pixel of the (nworld, NCOUNTER) `counters` over `pixels` per
  world (a host read)."""
  totals = host_copy(counters)
  mean = totals.astype(np.float64).sum(axis=0) / float(max(pixels * totals.shape[0], 1))
  return dict(zip(NAMES, mean.tolist(), strict=True))
