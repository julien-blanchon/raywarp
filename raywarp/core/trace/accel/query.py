"""The software ray queries: `make_trace` builds the `trace_closest` / `trace_any` pair of a
`Tracer`.

A query intersects the analytic planes kept out of the BVHs (`GeometryFields.plane_ids`, run-time
data; `Features.planes` none: no plane code; with shared scenery the worlds' trees stay small),
then traverses the scene BVHs (`ordered.py` nearest-first, or `cast.py` unordered) with the
planes' closest hit as its bound.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, final

import warp as wp

from raywarp._base.types import TWO_SIDED
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import PLANES_NONE
from raywarp.core.kernels.abi.rays import (
  AlphaTest,
  BodyAny,
  BodyClosest,
  CastRay,
  GeometryFields,
  Hit,
  ProbeFeatures,
  RayQuery,
  SoftwareFields,
  TraceFeatures,
  Tracer,
  Trees,
  instance_pose,
  new_ray,
  no_hit,
  widen,
  world_transmit,
  world_twosided,
)
from raywarp.core.kernels.abi.tables import batched, instance_rgba, instance_visible
from raywarp.core.kernels.generic import compiled_out, device_generic, specialize
from raywarp.core.trace.accel import cutout, nodes
from raywarp.core.trace.accel.cast import make_cast_ray, make_shadow_cast
from raywarp.core.trace.accel.instances import BvhTrees, CutAway, closer, culled
from raywarp.core.trace.accel.ordered import make_ordered_cast
from raywarp.core.trace.accel.primitives import RAY_TOL_REL, ray_plane

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# plane_hit(geo, worldid, gi, origin, direction) -> (distance (< 0: no hit), normal).
type PlaneHit = Callable[[GeometryFields, int, int, wp.vec3, wp.vec3], tuple[float, wp.vec3]]


@specialize
def make_plane_hit(
  *,
  cull_backfaces: bool,
  compact: bool,
  visibility: bool,
  two_sided: bool,
  cut_away: CutAway | None = None,
  first_hit: bool = False,
  bound: int = 0,
  switches: int = 0,
  world_optics: bool = False,
):
  """The `PlaneHit` query: plane gi intersected analytically; transparent or hidden planes are not
  hit; `two_sided`: planes flagged `TWO_SIDED` in `geo.instance_twosided` are not culled (the
  world's flags with `world_optics`, Features.world_optics: `trace.world_entry`);
  `cut_away`: the alpha test (a plane is crossed once: a discarded hit is no hit; None: none);
  `first_hit`: the occlusion and shadow queries' test, which tests a cut-out plane unculled (as the
  traversals test a cut-out instance's crossings, `instances.make_cut_analytic`);
  `bound` (Features.bound): the visibility table and compact poses compiled in are read where the
  build word says so
  (`toggles.GEO_VISIBLE`, `GEO_COMPACT`); `switches` (Features.switches): culling a run-time
  setting read at run time (`toggles.setting`, `toggles.BACKFACE_CULLING`)."""
  pose = instance_pose(compact=compact, bound=bound)
  gate = toggles.bound_on(bound)
  setting = toggles.setting(switches, bound)
  alpha = cut_away is not None
  cut_first = alpha and first_hit  # the closest query's back faces of cut-out planes
  cut_out = cutout.cut_out if cut_first else compiled_out

  @wp.func
  @device_generic
  def plane_hit[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    origin: wp.vec3,
    direction: wp.vec3,
  ) -> tuple[float, wp.vec3]:
    """(distance (< 0: no hit), normal) of plane `gi`."""
    d = float(-1.0)
    n = wp.vec3(0.0)
    rgba, _ = instance_rgba(geo.color_rule, geo.instance_material, geo.instance_color, geo.material_base_color, worldid, gi)
    show = rgba[3] != 0.0
    if wp.static(visibility):
      if gate(geo.bits, wp.static(toggles.GEO_VISIBLE)):
        show = show and instance_visible(geo.visible_slot, geo.instance_visible, worldid, gi)
    if show:
      pos, mat = pose(geo, worldid, gi)
      d, n = ray_plane(pos, mat, batched(geo.instance_size, worldid, gi), origin, direction)
      if wp.static(cull_backfaces):
        cull = setting(geo.rt.flags, geo.bits2, wp.static(toggles.BACKFACE_CULLING))
        if wp.static(two_sided):
          if gate(geo.bits, wp.static(toggles.SCENE_TWO_SIDED)):
            if wp.static(world_optics):
              cull = cull and (world_twosided(geo, worldid, gi) & wp.static(TWO_SIDED)) == 0
            else:
              cull = cull and (geo.instance_twosided[gi] & wp.static(TWO_SIDED)) == 0
        if wp.static(cut_first):
          # A back face read only: a cut-out plane is tested unculled.
          if cull and d >= 0.0 and wp.dot(direction, n) > 0.0:
            cull = not cut_out(geo, worldid, gi)
        d = culled(direction, cull, d, n)
      if wp.static(alpha):
        if d >= 0.0:
          hit = no_hit(d)
          hit.instance_id = gi
          hit.normal = n
          if cut_away(geo, worldid, hit, origin + direction * d, direction):  # pyrefly: ignore[not-callable]  # wp.static(alpha)
            d = -1.0
    return d, n

  return plane_hit


@final
class PlaneQueries(NamedTuple):
  """The analytic planes' queries of one plane form (`make_plane_queries`), shared by both
  backends.

  Attributes:
    closest: closest(geo, worldid, origin, direction, hit) -> `hit` or a closer plane hit
      (`instances.closer`'s tie-break: the higher instance wins).
    any: any(geo, worldid, origin, direction, max_dist) -> the instance of a plane hit before
      max_dist, -1: none.
    shadow: shadow(geo, worldid, origin, direction, max_dist) -> the planes' RGB transmittance
      (a translucent plane multiplies its `instance_shadow_transmit`, an opaque one gives 0).
  """

  closest: Callable[[GeometryFields, int, wp.vec3, wp.vec3, Hit], Hit]
  any: RayQuery[GeometryFields, int]
  shadow: RayQuery[GeometryFields, wp.vec3]


@specialize
def make_plane_queries(plane_hit: PlaneHit, occluder_hit: PlaneHit, *, world_optics: bool = False):
  """The `PlaneQueries` over the instances of `GeometryFields.plane_ids` (run-time data): a loop
  over the table, one plane test in the code per query (unrolled chunks of 4 cost a one-plane
  robot scene's fast raygen 7 757 PTX instructions against 4 996; the plane ids compiled
  in, 4 589); `occluder_hit`: the occlusion and shadow queries' test (`make_plane_hit(first_hit=)`;
  `plane_hit` itself without an alpha test); `world_optics`: the shadow query reads the world's
  transmittance (Features.world_optics). Builds with PLANES_NONE compile none of it: the queries'
  callers skip them (`wp.static`)."""
  return PlaneQueries(
    _closest_planes(plane_hit), _any_plane(occluder_hit), _planes_shadow(occluder_hit, world_optics=world_optics)
  )


@specialize
def _closest_planes(plane_hit: PlaneHit):
  """`PlaneQueries.closest`."""

  @wp.func
  @device_generic
  def closest_step[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    k: int,
    origin: wp.vec3,
    direction: wp.vec3,
    best: Hit,
  ) -> Hit:
    """`best` or plane k's hit, whichever is closer."""
    gi = geo.plane_ids[k]
    d, n = plane_hit(geo, worldid, gi, origin, direction)
    hit = no_hit(d)
    hit.instance_id = gi
    hit.normal = n
    closer(best, hit)
    return best

  @wp.func
  @device_generic
  def closest[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    hit: Hit,
  ) -> Hit:
    """`hit` or a closer plane hit."""
    best = hit
    for k in range(geo.plane_ids.shape[0]):
      best = closest_step(geo, worldid, k, origin, direction, best)
    return best

  return closest


@specialize
def _any_plane(plane_hit: PlaneHit):
  """`PlaneQueries.any`."""

  @wp.func
  @device_generic
  def any_step[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    k: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> int:
    """Plane k's instance if it is hit before max_dist, -1: not."""
    gi = geo.plane_ids[k]
    d, _n = plane_hit(geo, worldid, gi, origin, direction)
    if d >= 0.0 and d < max_dist:
      return gi
    return -1

  @wp.func
  @device_generic
  def any_plane[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> int:
    """A plane hit before max_dist (its instance id), -1: none."""
    for k in range(geo.plane_ids.shape[0]):
      gi = any_step(geo, worldid, k, origin, direction, max_dist)
      if gi != -1:
        return gi
    return -1

  return any_plane


@specialize
def _planes_shadow(plane_hit: PlaneHit, *, world_optics: bool):
  """`PlaneQueries.shadow` (`world_optics`: the world's transmittance, `trace.world_entry`)."""

  @wp.func
  @device_generic
  def shadow_step[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    k: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
    t: wp.vec3,
  ) -> wp.vec3:
    """`t` times plane k's transmittance if it is crossed before max_dist (0: blocked)."""
    gi = geo.plane_ids[k]
    d, _n = plane_hit(geo, worldid, gi, origin, direction)
    if d >= 0.0 and d < max_dist:
      if wp.static(world_optics):
        pt = world_transmit(geo, worldid, gi)
      else:
        pt = geo.instance_shadow_transmit[gi]
      if wp.max(pt[0], wp.max(pt[1], pt[2])) <= 0.0:
        return wp.vec3(0.0)
      return wp.cw_mul(t, pt)
    return t

  @wp.func
  @device_generic
  def shadow[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> wp.vec3:
    """The planes' transmittance before max_dist (0: an opaque plane blocks)."""
    t = wp.vec3(1.0)
    for k in range(geo.plane_ids.shape[0]):
      t = shadow_step(geo, worldid, k, origin, direction, max_dist, t)
      if wp.max(t[0], wp.max(t[1], t[2])) <= 0.0:
        return t
    return t

  return shadow


@wp.func
def plane_or_hit(
  # In:
  plane: Hit,
  hit: Hit,
) -> Hit:
  """The BVH hit, or the plane hit when the BVHs found nothing or it wins the tie (deformables win
  ties)."""
  tol = RAY_TOL_REL * wp.max(hit.dist, 1.0)
  tie = wp.abs(plane.dist - hit.dist) <= tol and plane.instance_id > hit.instance_id and hit.instance_id >= 0
  if plane.instance_id >= 0 and (hit.instance_id == -1 or tie):
    return plane
  return hit


def module_options(trees: Trees, *, ordered: bool = False) -> dict[str, wp.ModuleBuildOptions]:
  """Warp module options of kernels inlining `make_trace`'s queries.

  The native BVH node accessors for either depth-safe traversal (`cast.make_cast_ray`,
  `ordered.make_ordered_cast`).
  """
  if ordered or trees[0] or trees[1]:
    nodes.require_compatible()
    return {"extra_build_options": nodes.BUILD_OPTIONS}
  return {}


@specialize
def make_culls(cull_backfaces: bool, switches: int):
  """culls(flags) -> whether the software queries' rays cull back faces: compiled in
  (`cull_backfaces`) and, a tier build's run-time switch (`toggles.BACKFACE_CULLING` among
  `switches`), on in the launch's `Toggles.flags`."""
  on = toggles.switch(switches)

  @wp.func
  def culls(
    # In:
    flags: int,
  ) -> bool:
    """Whether the rays cull back faces (see `make_culls`)."""
    if wp.static(cull_backfaces):
      return on(flags, wp.static(toggles.BACKFACE_CULLING))
    return False

  return culls


@specialize
def make_any(cast_any: CastRay, planes: PlaneQueries | None, *, culls: Callable[[int], bool]):
  """trace_any: whether anything is hit, from the planes' (None: no plane code) and the BVHs'
  any-hit queries (`culls`: `make_culls`)."""
  has_planes = planes is not None
  any_plane = planes.any if planes is not None else compiled_out

  @wp.func
  @device_generic
  def trace_any[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> bool:
    """Whether anything is hit before `max_dist` (the first hit found, not the closest)."""
    ray = new_ray(origin, direction, max_dist, culls(geo.rt.flags))
    if wp.static(has_planes):
      if any_plane(geo, worldid, origin, direction, max_dist) != -1:
        return True
    return cast_any(geo, worldid, ray).instance_id != -1

  return trace_any


@specialize
def make_trace(
  *,
  types: tuple[int, ...],
  cull_backfaces: bool,
  ordered: bool,
  trees: Trees,
  two_sided: bool,
  counted: bool = False,
  alpha: AlphaTest | None = None,
  switches: int = 0,
  bound: int = 0,
  world_optics: bool = False,
):
  """(trace_closest, trace_any) specialised on the instance types present.

  The queries take the software geometry (`SoftwareFields`; `trace.optix.device.trace.make_trace`
  builds the OptiX ones). `ordered`: the closest-hit query's nearest-first traversal (`ordered.py`),
  else MuJoCo Warp's (`cast.py`); the any-hit query always walks unordered: it stops at its first
  hit, so sorting children buys no pruning and costs a second box test per inner node. `trees`:
  `Features.trees` (`trace.Trees`: the BVHs, and the planes' loop form).
  `two_sided`: culling skips the instances flagged in `GeometryFields.instance_twosided`
  (Features.two_sided; `world_optics`: the world's flags, Features.world_optics). `counted`: the
  traversals count their work (Features.counters). `alpha`:
  every candidate surface is alpha-tested where the traversal finds it (Features.alpha_cutout:
  `instances.make_intersect`), so the queries pass cut-out texels in one walk; None: no test.
  `switches` (Features.switches): culling a run-time switch (`make_culls`). `bound`
  (Features.bound): the structure bits it binds (`toggles.GEO_ALL`: a build compiling every
  structure, plane loop, pose indexing and visibility table, `kernels/probe.relight_features`),
  each read from the build word.
  """
  two_sided = two_sided and cull_backfaces
  culls = make_culls(cull_backfaces, switches)
  cut_away = _cut_away(types, trees, alpha, bound)
  traced = {
    "types": types,
    "trees": _bvh_trees(trees),
    "two_sided": two_sided,
    "counted": counted,
    "cut_away": cut_away,
    "bound": bound,
    "world_optics": world_optics,
  }
  if ordered:
    cast_closest = make_ordered_cast(**traced)
  else:
    cast_closest = make_cast_ray(**traced, first_hit=False)
  cast_any = make_cast_ray(**traced, first_hit=True)
  planes = _planes(
    trees,
    cull_backfaces=cull_backfaces,
    two_sided=two_sided,
    cut_away=cut_away,
    switches=switches,
    bound=bound,
    world_optics=world_optics,
  )
  has_planes = planes is not None
  nearest_plane = planes.closest if planes is not None else compiled_out

  @wp.func
  @device_generic
  def trace_closest[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> Hit:
    """Closest hit before `max_dist` (instance_id -1: none)."""
    ray = new_ray(origin, direction, max_dist, culls(geo.rt.flags))
    if wp.static(has_planes):
      # Analytic planes first: their closest hit bounds the BVH queries.
      plane = nearest_plane(geo, worldid, origin, direction, no_hit(max_dist))
      ray.max_dist = plane.dist
      return plane_or_hit(plane, cast_closest(geo, worldid, ray))
    return cast_closest(geo, worldid, ray)

  return trace_closest, make_any(cast_any, planes, culls=culls)


def _bvh_trees(trees: Trees) -> BvhTrees:
  """`trees` without the planes' loop form: the BVH part the traversals are specialised on."""
  return trees[0], trees[1], trees[2], trees[3]


def _cut_away(types: tuple[int, ...], trees: Trees, alpha: AlphaTest | None, bound: int) -> CutAway | None:
  """The candidates' alpha test of `alpha` (`cutout.make_cut_away`; `bound`: its compact poses
  read where the build word says so), None without one."""
  if alpha is None:
    return None
  return cutout.make_cut_away(types, compact=trees[2], live_uv=alpha.live_uv, world_uv=alpha.world_uv, bound=bound)


def _planes(
  trees: Trees,
  *,
  cull_backfaces: bool,
  two_sided: bool,
  cut_away: CutAway | None,
  switches: int,
  bound: int,
  world_optics: bool,
) -> PlaneQueries | None:
  """The planes' queries of `trees`' loop form (`Features.planes`), None: PLANES_NONE
  (`world_optics`: Features.world_optics)."""
  if trees[4] == PLANES_NONE:
    return None
  hits = [
    make_plane_hit(
      cull_backfaces=cull_backfaces,
      compact=trees[2],
      visibility=trees[3],
      two_sided=two_sided,
      cut_away=cut_away,
      first_hit=first,
      bound=bound,
      switches=switches,
      world_optics=world_optics,
    )
    for first in (False, cut_away is not None)
  ]
  return make_plane_queries(hits[0], hits[1], world_optics=world_optics)


@specialize
def make_shadow(
  *,
  types: tuple[int, ...],
  trees: Trees,
  cull_backfaces: bool,
  two_sided: bool,
  counted: bool = False,
  alpha: AlphaTest | None = None,
  switches: int = 0,
  bound: int = 0,
  world_optics: bool = False,
):
  """trace_shadow(geo, worldid, origin, direction, max_dist) -> a shadow ray's RGB transmittance
  through translucent casters (0: blocked; culling as the occlusion query: `cull_backfaces`,
  `two_sided` instances never): the planes (a translucent one multiplies, an opaque one blocks),
  then one any-hit walk whose translucent candidates multiply theirs in (`cast.make_shadow_cast`),
  as the OptiX shadow trace does (`trace/optix/device/trace.make_shadow`); `bound`, `world_optics`:
  as `make_trace`'s."""
  cut_away = _cut_away(types, trees, alpha, bound)
  two_sided = two_sided and cull_backfaces
  bvh = _bvh_trees(trees)
  cast_shadow = make_shadow_cast(
    types=types, trees=bvh, two_sided=two_sided, counted=counted, cut_away=cut_away, bound=bound, world_optics=world_optics
  )
  planes = _planes(
    trees,
    cull_backfaces=cull_backfaces,
    two_sided=two_sided,
    cut_away=cut_away,
    switches=switches,
    bound=bound,
    world_optics=world_optics,
  )
  has_planes = planes is not None
  planes_shadow = planes.shadow if planes is not None else compiled_out
  culls = make_culls(cull_backfaces, switches)

  @wp.func
  @device_generic
  def trace_shadow[G: SoftwareFields](
    # In:
    geo: G,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> wp.vec3:
    """The transmittance along a shadow ray before `max_dist` (see `make_shadow`)."""
    t = wp.vec3(1.0)
    if wp.static(has_planes):
      t = planes_shadow(geo, worldid, origin, direction, max_dist)
      if wp.max(t[0], wp.max(t[1], t[2])) <= 0.0:
        return t
    return wp.cw_mul(t, cast_shadow(geo, worldid, new_ray(origin, direction, max_dist, culls(geo.rt.flags))))

  return trace_shadow


def software_tracer(f: TraceFeatures, *, cull: bool) -> Tracer:
  """The software traversal's `Tracer` of `f` (`SoftwareBackend.tracer`).

  `cull`: the closest / any queries cull back faces (skipping `f.two_sided` instances); with
  `f.translucent_shadows`, the shadow query through translucent casters (`make_shadow`). With
  `f.alpha_cutout` the queries alpha-test their candidates (`make_trace`).
  """
  alpha = AlphaTest(f.live_material_uv, f.world_uv) if f.alpha_cutout else None
  # The structure bits of the word (a build compiling every structure, `toggles.GEO_ALL`); the
  # others are the shading's.
  traced = {
    "types": f.shape_types,
    "ordered": f.ordered_traversal,
    "trees": f.trees,
    "counted": f.counters,
    "alpha": alpha,
    "bound": f.bound & toggles.GEO_ALL,
    "world_optics": f.world_optics,
  }
  closest, any_hit = make_trace(**traced, cull_backfaces=cull, two_sided=f.two_sided and cull, switches=f.switches)
  shadow = None
  if f.translucent_shadows:
    shadow = make_shadow(
      types=f.shape_types,
      trees=f.trees,
      cull_backfaces=cull,
      two_sided=f.two_sided,
      counted=f.counters,
      alpha=alpha,
      switches=f.switches,
      bound=f.bound & toggles.GEO_ALL,
      world_optics=f.world_optics,
    )
  return Tracer(closest, any_hit, shadow=shadow)


def unculled_queries(
  shape_types: tuple[int, ...], *, trees: Trees, alpha_cutout: bool, live_uv: bool = False, world_uv: bool = False
) -> tuple[BodyClosest, BodyAny]:
  """(closest, any) software queries without back-face culling of the probe / grid kernels
  (GI relocation, light-grid visibility): `ProbeFeatures(shape_types, trees)`'s,
  alpha-tested with `alpha_cutout` (Features.alpha_cutout).
  """
  features = ProbeFeatures(shape_types, trees, alpha_cutout=alpha_cutout, live_material_uv=live_uv, world_uv=world_uv)
  tracer = software_tracer(features, cull=False)
  return widen(tracer.closest), widen(tracer.any_hit)
