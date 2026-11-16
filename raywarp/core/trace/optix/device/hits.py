"""What the OptiX hit programs compute: instances, poses, analytic curve and primitive hits.

Every hit program reports the hit's instance itself: a triangle instance's OptiX id is its
instance (`instance_of`), the curve and intersection programs find theirs from the primitive
index (`world_instance`). Poses are those the structures were written with (`make_pose`: per
world on the world's cell, shared in scene coordinates with world 0's size), so the analytic hits
(MuJoCo Warp's `ray_*` functions) match the software traversal's.
"""

from __future__ import annotations

from typing import Final

import warp as wp

from raywarp._base.types import TWO_SIDED, ShapeType
from raywarp.core.kernels import native
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.rays import (
  AlphaTest,
  OptixFields,
  instance_pose,
  no_hit,
  offset_floor,
  ray_offset,
  world_transmit,
  world_twosided,
)
from raywarp.core.kernels.abi.tables import batched, instance_visible
from raywarp.core.kernels.generic import device_generic, specialize
from raywarp.core.trace.accel import cutout
from raywarp.core.trace.accel.instances import culled
from raywarp.core.trace.accel.primitives import ray_capsule, ray_cylinder, ray_ellipsoid, ray_sphere
from raywarp.core.trace.optix.device.trace import RAY_MASK_CULL, SHARED_RAY, from_cell, to_cell, world_scale

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Relative distance by which the analytic curve hit may differ from OptiX's (`make_curve_hit`).
_CURVE_HIT_TOL: Final = 1.0e-3


@wp.func
def world_instance(
  # In:
  index: int,
  instances: wp.array1d[int],
) -> tuple[int, int]:
  """(world, instance) of primitive `world · len(instances) + k` (instance instances[k]) of a
  world-space GAS."""
  n = instances.shape[0]
  worldid = index // n
  return worldid, instances[index - worldid * n]


@wp.func
def instance_of(
  # In:
  payload_world: int,
  instance_id: int,
) -> tuple[int, int]:
  """(world, instance) of a triangle hit: the ray's payload world (`SHARED_RAY` added for the
  shared IAS) and OptiX's instance id, the instance itself (`instances.write_instances`)."""
  world = wp.where(payload_world >= SHARED_RAY, payload_world - SHARED_RAY, payload_world)
  return world, instance_id


@specialize
def make_pose(*, compact: bool, shared: bool, bound: int = 0):
  """pose(geo, worldid, gi) -> (position, rotation, size) of instance gi as the IAS holds it.

  Per world: on its cell (`trace.to_cell`); shared: in scene coordinates with world 0's
  size. `compact`: Features.compact_poses (`bound`: Features.bound, `instance_pose`).
  """
  instance = instance_pose(compact=compact, bound=bound)

  @wp.func
  @device_generic
  def pose[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
  ) -> tuple[wp.vec3, wp.mat33, wp.vec3]:
    """Position, rotation and size of instance `gi` (see `make_pose`)."""
    if wp.static(shared):
      p, rot = instance(geo, 0, gi)
      return p, rot, batched(geo.instance_size, 0, gi)
    p, rot = instance(geo, worldid, gi)
    return to_cell(geo.rt_cell, worldid, p), rot, batched(geo.instance_size, worldid, gi) * world_scale(geo.rt_cell, worldid)

  return pose


@wp.func
@device_generic
def _curve_hit_at[G: OptixFields](
  # In:
  geo: G,
  gi: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  t: float,
) -> tuple[float, wp.vec3]:
  """Analytic hit and normal on sphere / capsule gi; OptiX's distance if they disagree."""
  capsule = geo.instance_type[gi] == int(ShapeType.CAPSULE)
  d = float(-1.0)
  n = wp.vec3(0.0)
  if capsule:
    d, n = ray_capsule(pos, rot, size, pnt, vec)
  else:
    d, n = ray_sphere(pos, size[0] * size[0], pnt, vec)
  if d < 0.0 or wp.abs(d - t) > wp.static(_CURVE_HIT_TOL) * wp.max(t, size[0]):
    # OptiX's hit stands; the normal is the analytic one at that distance.
    axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    p = pnt + t * vec
    s = float(0.0)
    if capsule:
      s = wp.clamp(wp.dot(p - pos, axis), -size[1], size[1])
    d = t
    n = wp.normalize(p - (pos + s * axis))
  return d, n


@specialize
def make_curve_hit(*, compact: bool, shared: bool, bound: int = 0):
  """curve_hit(geo, curve_instance) of a curve closest-hit program: (distance, world normal,
  instance).

  Segment `world · ncurve + k` is sphere / capsule `curve_instance[k]` (shared: segment k, world
  0). Re-runs MuJoCo Warp's `ray_sphere` / `ray_capsule` on the world ray; if that disagrees with
  OptiX's hit (it should not), keeps OptiX's distance with the analytic normal there.
  """
  pose = make_pose(compact=compact, shared=shared, bound=bound)

  @wp.func
  @device_generic
  def curve_hit[G: OptixFields](
    # In:
    geo: G,
    curve_instance: wp.array1d[int],
  ) -> tuple[float, wp.vec3, int]:
    """The analytic hit of the current curve primitive (see `make_curve_hit`)."""
    worldid, gi = world_instance(native.optix_primitive_index(), curve_instance)
    pos, rot, size = pose(geo, worldid, gi)
    pnt = native.optix_world_ray_origin()
    vec = native.optix_world_ray_direction()
    d, n = _curve_hit_at(geo, gi, pos, rot, size, pnt, vec, native.optix_ray_tmax())
    return d, n, gi

  return curve_hit


@specialize
def make_primitive_hit(*, gtype: int, compact: bool, shared: bool, bound: int = 0):
  """primitive_hit(geo, worldid, gi, pnt, vec) -> (distance (< 0: miss), world normal) on
  ellipsoid / cylinder gi (type `gtype`), intersected in scene-sized local coordinates to keep the
  analytic routines' tolerances.

  Normalizing a world's bounds must not shrink a cylinder's quadratic discriminant below the
  fixed scene-unit cutoff. Only the input ray and returned distance use acceleration units.
  `bound`: Features.bound (`instance_pose`).
  """
  instance = instance_pose(compact=compact, bound=bound)

  @wp.func
  @device_generic
  def primitive_hit[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
    pnt: wp.vec3,
    vec: wp.vec3,
  ) -> tuple[float, wp.vec3]:
    """Analytic hit on primitive gi, without OptiX intrinsic calls or visibility decisions."""
    pos, rot = instance(geo, worldid, gi)
    size = batched(geo.instance_size, worldid, gi)
    scale = float(1.0)
    if wp.static(not shared):
      scale = world_scale(geo.rt_cell, worldid)
      pnt = (pnt - to_cell(geo.rt_cell, worldid, pos)) / scale
      pos = wp.vec3(0.0)
    if wp.static(gtype == int(ShapeType.ELLIPSOID)):
      d, n = ray_ellipsoid(pos, rot, size, pnt, vec)
    else:
      d, n = ray_cylinder(pos, rot, size, pnt, vec)
    if d >= 0.0:
      d = d * scale
    return d, n

  return primitive_hit


@specialize
def make_cut_crossings(*, gtype: int, compact: bool, shared: bool, alpha: AlphaTest, bound: int = 0):
  """crossings(geo, worldid, ray_world, gi, pnt, vec, cull, d, n) of a cut-out custom primitive of
  type `gtype` hit first at (d, n) by the world ray (pnt, vec) (as `make_primitive_hit`'s, posed in
  world `worldid`; `ray_world`: the ray's, whose materials the test reads): a convex
  primitive has at most two crossings (in, out), each culled as `cull` and alpha-tested where it is
  found, as the software traversal's (`trace.accel.instances.make_cut_analytic`). `alpha`: the
  test's texture coordinates (Features.live_material_uv, world_uv); `bound`: Features.bound."""
  primitive_hit = make_primitive_hit(gtype=gtype, compact=compact, shared=shared, bound=bound)
  cut_away = cutout.make_cut_away((gtype,), compact=compact, live_uv=alpha.live_uv, world_uv=alpha.world_uv, bound=bound)

  @wp.func
  @device_generic
  def crossings[G: OptixFields](
    # In:
    geo: G,
    worldid: int,
    ray_world: int,
    gi: int,
    pnt: wp.vec3,
    vec: wp.vec3,
    cull: bool,
    d: float,
    n: wp.vec3,
  ) -> tuple[float, wp.vec3]:
    """The first crossing the alpha test keeps (distance -1: none)."""
    scale = float(1.0)
    if wp.static(not shared):
      scale = world_scale(geo.rt_cell, worldid)
    candidate = no_hit(-1.0)
    candidate.instance_id = gi
    for _crossing in range(2):
      if d < 0.0:
        break
      if culled(vec, cull, d, n) < 0.0:
        return -1.0, n
      # In scene coordinates (a per-world ray leaves the world's cell).
      point = pnt + vec * d
      if wp.static(not shared):
        point = from_cell(geo.rt_cell, worldid, point)
      candidate.normal = n
      if not cut_away(geo, ray_world, candidate, point, vec):
        return d, n
      # The next crossing, from just past this one (the offset in scene units).
      skip = d + ray_offset(point, offset_floor(geo, gi)) * scale
      d, n = primitive_hit(geo, worldid, gi, pnt + vec * skip, vec)
      d = wp.where(d >= 0.0, d + skip, -1.0)
    return -1.0, n

  return crossings


@specialize
def make_intersection(  # noqa: C901 (the program's compile-time branches)
  *,
  gtype: int,
  cull_backfaces: bool,
  two_sided: bool,
  compact: bool,
  shared: bool,
  visible: bool,
  shadow: bool = False,
  alpha: AlphaTest | None = None,
  bound: int = 0,
  world_optics: bool = False,
):
  """intersect(geo, prim_code) of custom primitive `world · nprim + k` (shared: k).

  `prim_code[world, k]` is its instance (type `gtype`), -1 - instance when hidden. Runs MuJoCo
  Warp's `ray_*` function in scene-sized local coordinates (`make_primitive_hit`). Returns
  (distance, world normal, instance); distance < 0 on a miss. `cull_backfaces`: the rays of culling
  queries (`trace.RAY_MASK_CULL`) drop back-face hits (`trace.accel.instances.culled`), except on
  the instances flagged two-sided (`two_sided`: Features.two_sided). `visible` (shared): a miss
  where the ray's world (the payload's, minus `trace.SHARED_RAY`) hides the instance. `shadow`: the
  shadow traces' program (ray type 1): a translucent caster multiplies its transmittance in and
  reports no intersection (`intrinsics.shadow_attenuate`). `alpha` (Features.alpha_cutout): a
  cut-out instance's hit is its first crossing the alpha test keeps (`make_cut_crossings`; the
  occlusion and shadow rays, which end at their first hit, test both crossings unculled, as the
  software traversal); the test runs where the build word has an opacity map
  (`toggles.MAP_OPACITY`). `bound` (Features.bound): the visibility test and the two-sided
  exception run where the build word has them (`toggles.GEO_SHARED_VISIBLE`, `SCENE_TWO_SIDED`).
  `world_optics` (Features.world_optics): the two-sided flags and the transmittance are the ray's
  world's (`trace.world_entry`).
  """
  primitive_hit = make_primitive_hit(gtype=gtype, compact=compact, shared=shared, bound=bound)
  gate = toggles.bound_on(bound)
  field = toggles.bound_on(bound, high=True)
  cut = alpha is not None
  crossings = make_cut_crossings(gtype=gtype, compact=compact, shared=shared, alpha=alpha, bound=bound) if alpha else None

  @wp.func
  @device_generic
  def intersect[G: OptixFields](  # noqa: C901 (compile-time branches)
    # In:
    geo: G,
    prim_code: wp.array2d[int],
  ) -> tuple[float, wp.vec3, int]:
    """The world ray's hit on the current custom primitive (see `make_intersection`)."""
    index = native.optix_primitive_index()
    worldid = index // prim_code.shape[1]
    code = prim_code[worldid, index - worldid * prim_code.shape[1]]
    d = float(-1.0)
    n = wp.vec3(0.0)
    if code < 0:
      return d, n, -1 - code
    gi = code
    # The ray's world (a shared primitive's: the payload's; its pose and size are world 0's).
    ray_world = worldid
    if wp.static(shared and (visible or cut)):
      ray_world = native.optix_payload_world() - SHARED_RAY
    if wp.static(visible):
      if gate(geo.bits, wp.static(toggles.GEO_SHARED_VISIBLE)):
        if not instance_visible(geo.visible_slot, geo.instance_visible, ray_world, gi):
          return d, n, gi
    pnt = native.optix_world_ray_origin()
    vec = native.optix_world_ray_direction()
    d, n = primitive_hit(geo, worldid, gi, pnt, vec)
    tested = False
    if wp.static(cut):
      if d >= 0.0:
        if field(geo.bits2, wp.static(toggles.MAP_OPACITY)):
          tested = cutout.cut_out(geo, ray_world, gi)
    cull = False
    if wp.static(cull_backfaces):
      cull = (int(native.optix_ray_mask()) & RAY_MASK_CULL) != 0
      if wp.static(two_sided):
        if cull and gate(geo.bits, wp.static(toggles.SCENE_TWO_SIDED)):  # read only by culling rays
          if wp.static(world_optics):
            cull = (world_twosided(geo, ray_world, gi) & wp.static(TWO_SIDED)) == 0
          else:
            cull = (geo.instance_twosided[gi] & wp.static(TWO_SIDED)) == 0
      if not tested:
        d = culled(vec, cull, d, n)
    if tested:
      # Occlusion and shadow rays test a cut-out primitive's crossings unculled.
      cull = cull and not native.optix_first_hit_ray()
      d, n = crossings(geo, worldid, ray_world, gi, pnt, vec, cull, d, n)  # pyrefly: ignore[not-callable]  # wp.static(cut)
    # A translucent shadow caster on a shadow trace: its transmittance multiplies the ray's, and it
    # reports no intersection (`intrinsics.shadow_attenuate`).
    if wp.static(shadow):
      if d >= 0.0:
        if wp.static(world_optics):
          t = world_transmit(geo, ray_world, gi)
        else:
          t = geo.instance_shadow_transmit[gi]
        if wp.max(t[0], wp.max(t[1], t[2])) > 0.0:
          native.optix_shadow_attenuate(gi, t)
          d = -1.0
    return d, n, gi

  return intersect
