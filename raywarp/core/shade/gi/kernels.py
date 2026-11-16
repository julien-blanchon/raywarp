"""The GPU / CPU kernels of the GI probes: placement (the relocation survey and move, the ambient
cube's walls, compact solids and contact occlusion), the static bake, ray directions and lobe
lists, probe rays, the bake's bounces and probe filter, map blends, the cube's resample, and the
update counter. The ray bodies (`make_probe_ray`, `make_static_ray`, `make_survey_ray`,
`make_walls_texel`, `make_ao_texel`) are shared by the software kernels and the OptiX programs
(`rays.optix.device.programs`).

Every update kernel reads the volume's device state only (`volume.ProbeVolume`: the update
counter, which picks the ray rotation and the random seed, a gate flag, and the groups'
representative worlds, `groups.Groups`), so an update is a fixed sequence of launches with no
host value in it: a CUDA graph replays it unchanged, and the gate makes the periodic cadence
(`gi.update_every`) a device decision (`Cadence.tick`). A group runs when the gate is open and a
world uses it this frame (`group_used`). The per-world relight kernel (`make_relight_kernel`)
serves `relight.Relight`.

Ray directions are the spherical Fibonacci set rotated once per update (`rotate_rays`); the
blends read per-update lobe lists (`lobe_rays`: the rays in the cosine lobe of each irradiance
texel and the power-50 lobe of each distance texel, with their weights), computed once per
(texel, ray) instead of once per (probe, texel, ray), and looped over instead of every ray (a
distance texel's lobe holds ~7 % of the rays).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import warp as wp

from raywarp._base.types import TWO_SIDED, ShapeType
from raywarp.core.kernels.abi import rays
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import BodyClosest, Trees, offset_floor, ray_offset
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.shade.gi.cube import (
  COMPACT_BIT,
  FACES,
  INSIDE_BIT,
  MAP_MAX,
  SOLID_SHIFT,
  VISIBLE_MIN,
  WALL_STEPS,
  Cube,
  free_distance,
  store_texel,
  wall_channel,
  write_texel,
)
from raywarp.core.shade.gi.groups import group_used
from raywarp.core.shade.gi.probes import (
  DIST_TEXELS,
  DISTANCE_EXPONENT,
  IRR_TEXELS,
  MISS_DISTANCE,
  Volume,
  fibonacci_direction,
  grid_position,
  irradiance,
  lookup,
  probe_cell,
  probe_coords,
  probe_corner,
  probe_index,
  probe_position,
  texel_direction,
)
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.material import MaterialTables, material_type
from raywarp.core.shade.materials.octahedral import oct_encode, pack_unit, unpack_unit
from raywarp.core.trace.accel import query

wp.set_module_options({"enable_backward": False})

# Relocation pass: fixed rays, back-face fraction above which a probe is inside; a survey ray's
# first hit (`survey_kernel`).
RELOCATE_RAYS: Final = 64
BACKFACE_FRACTION: Final = 0.25
SURVEY_NONE: Final = 0
SURVEY_BACK: Final = 1
SURVEY_FRONT: Final = 2
# A probe moves away from a front face closer than this share of the spacing (`relocate_step`;
# measured on the showcase room's window wall: 0.25 / 0.4 raise the floor's cube / gather ratio
# along it from 1.52 to 1.77 / 2.0).
MIN_FRONT: Final = 0.1
# Free-path survey: a point is enclosed when front faces cover this share of its directions.
ENCLOSED_FRACTION: Final = 0.75
# Seed stride between updates (update k's probe rays draw from seed k x SEED_STRIDE's streams).
SEED_STRIDE: Final = 7919
# See-through surfaces (window glass) a static ray passes at most.
SEE_THROUGH_LAYERS: Final = 4
# The bake's probe filter (`filter_irradiance`): a neighbour is used when the probe's mean static
# distance towards it reaches this share of the way (a wall in between stops the rays short).
FILTER_REACH: Final = 0.9
# ... and when its texel's light is within this factor of the probe's (noise, not another region).
# gi_stress cases against their Mitsuba references at 512 px: 2 smeared the falloff around
# emitters (emissive_sphere blotch 0.47 without the filter -> 0.65); 1.5: emissive_sphere 0.52,
# door_rooms 0.26 -> 0.20.
FILTER_RATIO: Final = 1.5
# How far a hidden back face is looked for past a miss [m] (`with_back_faces`): past any closed
# primitive the GI meets, near enough for float32 ray-quadric tests (from 1 km a 0.4 m sphere's
# test cancels to a miss).
BACK_REACH: Final = 20.0
# Distance-texel lobe weights below this are dropped from its ray list (`lobe_rays`).
LOBE_MIN: Final = 1.0e-4
# Distance of a back-face hit relative to its hit distance (DDGI's negative distance: inside).
BACKFACE_DISTANCE: Final = -0.2
# A texel centre is inside a closed solid when at least this many of its six axis rays first meet
# a back face (all six inside a lone convex solid; five tolerates one ray escaping through a seam;
# four also classified the overlap texels at a room's creases, whose fill then came from outdoors:
# bright crease pixels on the window case).
INSIDE_AXES: Final = 5
# The longest run of a compact solid's texels along an axis (`compact_solids`): two probe spacings
# (cube.SUBDIV texels each), objects up to about a metre at a 0.5 m spacing; walls and slabs run
# longer along two axes.
COMPACT_TEXELS: Final = 8
# Contact occlusion (`ao_kernel`): fixed spherical Fibonacci rays per texel, shared by its six
# faces (cosine weights).
AO_RAYS: Final = 48
# ... and the share of a ray's occlusion it applies: the texel's occlusion also darkens the surfaces
# a texel away that read it (a shelf base's floor). gi_stress cases against their Mitsuba
# references at 512 px: full strength shelves blotch 1.80, half 0.80; contact_spheres 0.143 ->
# 0.231, within its limit.
CONTACT_AO_STRENGTH: Final = 0.5


@specialize
def with_back_faces(trace_closest: BodyClosest):
  """`trace_closest` with the back faces it does not report: OptiX's curve intersector (spheres,
  capsules) reports no hit for a ray from inside the curve, and the placement reads those back
  faces (a texel or probe inside a solid: `_make_inside`, the relocation survey). A second ray
  from just before the forward hit (or BACK_REACH away on a miss) back towards the origin meets
  the surface the forward ray passed, the exit face of the primitive the origin lies in: reported
  as the hit at its distance from the origin (its normal faces along the ray: a back face). For
  the hardware placement programs over scenes with curves
  (`rays.optix.device.programs._gi_programs`; the acceleration structures are the render's, whose
  curve GASes only OptiX's own intersector reads)."""

  @wp.func
  @device_generic
  def closest[G: rays.GeometryFields](
    # Model:
    geo: G,
    # In:
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
  ) -> rays.Hit:
    """The closest hit, back faces included (see `with_back_faces`)."""
    hit = trace_closest(geo, worldid, origin, direction, max_dist)
    end = wp.where(hit.instance_id == -1, wp.min(max_dist, wp.static(BACK_REACH)), hit.dist)
    start = origin + direction * end
    end -= ray_offset(start, offset_floor(geo, hit.instance_id))
    if end <= 0.0:
      return hit
    back = trace_closest(geo, worldid, origin + direction * end, -direction, end)
    if back.instance_id == -1 or back.dist >= end:
      return hit
    back.dist = end - back.dist
    return back

  return closest


@wp.struct
class RayHit:
  """A probe ray's light without its bounce (`fixed`: direct light, emission, sky through its
  layers), and the bounce's weight (`weight`, summed over the layers) on the volume's irradiance
  at distance `at` along the ray (< 0: none) around shading normal `normal` (geometric normal
  `outward`): the light is `fixed + weight * irradiance` (`bounce_hits`)."""

  fixed: wp.vec3
  weight: wp.vec3
  at: float
  normal: wp.vec3
  outward: wp.vec3


@wp.func
def ray_hit(
  # In:
  fixed: wp.vec3,
  weight: wp.vec3,
  at: float,
  normal: wp.vec3,
  outward: wp.vec3,
) -> RayHit:
  """A `RayHit` of these fields."""
  out = RayHit()
  out.fixed = fixed
  out.weight = weight
  out.at = at
  out.normal = normal
  out.outward = outward
  return out


@wp.struct
class RayHits:
  """The `RayHit`s of a chunk's rays (rows as the ray buffers'), stored by the bake's traced update
  and bounced by its others (`bounce_hits`); empty (0 rows) for the periodic updates."""

  fixed: wp.array2d[wp.vec3]
  weight: wp.array2d[wp.vec3h]
  at: wp.array2d[float]
  # The shading and geometric normals (`octahedral.pack_unit`).
  normals: wp.array2d[wp.vec2ui]


# shade_probe_ray(geo, mats, lit, env, world, origin, direction, spread, seed, stream)
# -> (radiance, hit distance, sky weight, RayHit): the megakernel's probe shading
# (`kernels/probe.make_probe_shading`).
type ShadeProbeRay = Callable[
  [
    rays.GeometryFields,
    MaterialTables,
    Lights,
    kernel_structs.EnvironmentArrays,
    int,
    wp.vec3,
    wp.vec3,
    float,
    int,
    int,
  ],
  tuple[wp.vec3, float, float, RayHit],
]


@wp.func
@device_generic
def back_face[G: rays.GeometryFields](
  # Model:
  geo: G,
  # In:
  hit: rays.Hit,
  direction: wp.vec3,
) -> bool:
  """Whether `hit` (a ray along `direction`) is the back face of a solid: the ray started inside
  it. Closed primitives (sphere, capsule, ellipsoid, cylinder, box) have an interior whatever their
  two-sided render flag (a USD stage's gprims are all doubleSided unless its RTX settings cull
  faces: probes and texels inside them would otherwise stay active and dark). Two-sided meshes
  (USD doubleSided, `instance_twosided`) have no inside: both of their faces are front faces
  (single-sided quads of a room flagged doubleSided, wound outwards, would otherwise put every
  probe of the room "inside")."""
  if wp.dot(hit.normal, direction) <= 0.0:
    return False
  i = hit.instance_id
  if i < 0 or i >= geo.instance_twosided.shape[0] or (geo.instance_twosided[i] & wp.static(TWO_SIDED)) == 0:
    return True
  t = geo.instance_type[i]
  return t >= wp.static(int(ShapeType.SPHERE)) and t <= wp.static(int(ShapeType.BOX))


@wp.kernel
def rotate_rays(
  # In:
  rotations: wp.array1d[wp.mat33],
  updates: wp.array1d[int],
  gate: wp.array1d[int],
  nray: int,
  # Out:
  dirs: wp.array1d[wp.vec3],
):
  """Direction of probe ray r this update: the Fibonacci point r under the update's rotation."""
  r = wp.tid()
  if gate[0] == 0:
    return
  dirs[r] = rotations[updates[0] % rotations.shape[0]] @ fibonacci_direction(r, nray)


@wp.struct
class LobeRays:
  """The rays in each texel's lobe, in ray order (`lobe_rays`): texel t's k-th ray `ray[t, k]`
  and its weight `weight[t, k]`, k < `count[t]`; the blends loop over these only."""

  ray: wp.array2d[wp.int32]
  weight: wp.array2d[float]
  count: wp.array1d[wp.int32]


@wp.kernel
def lobe_rays(
  # In:
  dirs: wp.array1d[wp.vec3],
  gate: wp.array1d[int],
  nray: int,
  # Out:
  irradiance: LobeRays,
  distance: LobeRays,
):
  """The rays of texel t's lobe: cosine (irradiance texels: every ray in front of the texel's
  direction), power-50 cosine (distance texels: weights above LOBE_MIN, a cone of ~32 degrees, 7 %
  of the rays: 0.01 % of the lobe's weight dropped). One thread per texel, rays in order (the
  sums keep a fixed order: bit-reproducible bakes)."""
  t = wp.tid()
  if gate[0] == 0:
    return
  nirr = wp.static(IRR_TEXELS * IRR_TEXELS)
  n = int(0)
  if t < nirr:
    d = texel_direction(t, wp.static(IRR_TEXELS))
    for r in range(nray):
      w = wp.dot(d, dirs[r])
      if w > 0.0:
        irradiance.ray[t, n] = r
        irradiance.weight[t, n] = w
        n += 1
    irradiance.count[t] = n
  else:
    d = texel_direction(t - nirr, wp.static(DIST_TEXELS))
    for r in range(nray):
      w = wp.pow(wp.max(0.0, wp.dot(d, dirs[r])), wp.static(DISTANCE_EXPONENT))
      if w > wp.static(LOBE_MIN):
        distance.ray[t - nirr, n] = r
        distance.weight[t - nirr, n] = w
        n += 1
    distance.count[t - nirr] = n


@wp.func
def _sees(
  # Model:
  vol: Volume,
  # In:
  probe: int,
  towards: wp.vec3,
) -> bool:
  """Whether `probe`'s static rays reach `towards` (the mean distance of its distance texel
  nearest that direction covers it, FILTER_REACH): no wall between them."""
  v = towards - probe_position(vol, probe)
  dist = wp.length(v)
  uv = oct_encode(v / wp.max(dist, 1.0e-6))
  n = wp.static(DIST_TEXELS)
  tx = wp.clamp(int((uv[0] + 1.0) * 0.5 * float(n)), 0, n - 1)
  ty = wp.clamp(int((uv[1] + 1.0) * 0.5 * float(n)), 0, n - 1)
  return float(vol.distance[probe, ty * n + tx][0]) * vol.max_distance >= wp.static(FILTER_REACH) * dist


@wp.kernel
def filter_irradiance(
  # Model:
  vol: Volume,
  # Data in:
  src: wp.array2d[wp.vec4h],
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  # In:
  nworld: int,
  first_group: int,
  # Out:
  target: wp.array2d[wp.vec4h],
):
  """Irradiance texel `texel` of probe `probe` of group `first_group + i`: the 1-2-1 binomial mean
  over its 3 x 3 x 3 grid neighbours that are active, see it both ways (`_sees`: no wall between
  them) and hold light within FILTER_RATIO of its own. The bake's maps hold one set of rays each
  (no temporal mean): this averages the probes' noise (the blotches of a window-lit room, where
  few rays meet the sunlit patch) over a probe spacing (`ProbeVolume.bake`)."""
  i, probe, texel = wp.tid()
  g = first_group + i
  if not group_used(gate, rep, nworld, g) or vol.active[probe] == 0:
    return
  row = g * vol.nprobe
  here = probe_position(vol, probe)
  c = probe_coords(vol, probe)
  own = wp.vec4(src[row + probe, texel])
  lum = own[0] + own[1] + own[2]
  total = own * 8.0
  wsum = float(8.0)
  for k in range(27):
    d = wp.vec3i(k % 3 - 1, (k // 3) % 3 - 1, k // 9 - 1)
    w = float((2 - wp.abs(d[0])) * (2 - wp.abs(d[1])) * (2 - wp.abs(d[2])))
    n = c + d
    if k == 13 or wp.min(n) < 0 or n[0] >= vol.counts[0] or n[1] >= vol.counts[1] or n[2] >= vol.counts[2]:
      continue
    other = probe_index(vol, n)
    if vol.active[other] == 0:
      continue
    v = wp.vec4(src[row + other, texel])
    # A neighbour whose light differs by more than FILTER_RATIO is another region, not noise (a
    # probe outside a window seeing in through it).
    other_lum = v[0] + v[1] + v[2]
    if other_lum > wp.static(FILTER_RATIO) * lum or lum > wp.static(FILTER_RATIO) * other_lum:
      continue
    if not _sees(vol, probe, probe_position(vol, other)) or not _sees(vol, other, here):
      continue
    total += v * w
    wsum += w
  target[row + probe, texel] = wp.vec4h(total / wsum)


@wp.kernel
def bounce_hits(
  # Model:
  vol: Volume,
  # Data in:
  dirs: wp.array1d[wp.vec3],
  hits: RayHits,
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  # In:
  nworld: int,
  first_group: int,
  first: int,
  # Out:
  ray_radiance: wp.array2d[wp.vec4],
):
  """Ray r of probe `first + probe` of group `first_group + i` lit again from its `RayHit`: its
  light without the bounce plus the bounce's weight times the volume's irradiance now (a bake's
  update after the traced one: one more bounce, no ray traced). The hit distance stays."""
  i, probe, r = wp.tid()
  probe += first
  g = first_group + i
  if not group_used(gate, rep, nworld, g):
    return
  row = i * vol.nprobe + probe
  light = hits.fixed[row, r]
  at = hits.at[row, r]
  if at >= 0.0:
    n = hits.normals[row, r]
    e = irradiance(vol, rep[g], probe_position(vol, probe) + dirs[r] * at, unpack_unit(n[0]), unpack_unit(n[1]))
    light += wp.cw_mul(wp.vec3(hits.weight[row, r]), wp.vec3(e[0], e[1], e[2]))
  ray_radiance[row, r] = wp.vec4(light[0], light[1], light[2], ray_radiance[row, r][3])


@wp.kernel
def blend_irradiance(
  # Model:
  vol: Volume,
  # Data in:
  ray_radiance: wp.array2d[wp.vec4],
  ray_sky: wp.array2d[float],
  lobes: LobeRays,
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  # In:
  nworld: int,
  first_group: int,
  first: int,
  hysteresis: float,
  # Out:
  target: wp.array2d[wp.vec4h],
):
  """Blends the ray results of probe `first + probe` of group `first_group + i` (ray rows: the
  groups from `first_group`) into irradiance texel `texel` of `target` with hysteresis."""
  i, probe, texel = wp.tid()
  probe += first
  g = first_group + i
  if not group_used(gate, rep, nworld, g) or vol.active[probe] == 0:
    return
  h = hysteresis
  ray_row = i * vol.nprobe + probe
  row = g * vol.nprobe + probe
  total = wp.vec4(0.0)
  wsum = float(0.0)
  for k in range(lobes.count[texel]):
    r = lobes.ray[texel, k]
    ray = ray_radiance[ray_row, r]
    if ray[3] < 0.0:  # back face: no light
      continue
    sample = wp.vec4(ray[0] * vol.scale, ray[1] * vol.scale, ray[2] * vol.scale, ray_sky[ray_row, r])
    if not wp.isfinite(sample):  # NaN guard: one bad ray must not poison the maps
      continue
    w = lobes.weight[texel, k]
    total += sample * w
    wsum += w
  result = total / wp.max(wsum, 1.0e-9)
  old = wp.vec4(target[row, texel])
  mixed = result * (1.0 - h) + old * h
  # float16 maps: clamp the color to the finite range (MAP_MAX).
  for c in range(3):
    mixed[c] = wp.clamp(mixed[c], 0.0, wp.static(MAP_MAX))
  target[row, texel] = wp.vec4h(mixed)


@wp.kernel
def accumulate_static(
  # Model:
  vol: Volume,
  # Data in:
  ray_radiance: wp.array2d[wp.vec4],
  distance: LobeRays,
  # In:
  first: int,
  # Out:
  distance_sum: wp.array2d[wp.vec3],
):
  """Adds the static rays of probe `first + tid` to the weighted sums of its distance texel
  `texel` (distance, distance², weight): the static bake pools all its ray sets before dividing
  (`resolve_static`)."""
  probe, texel = wp.tid()
  probe += first
  if vol.active[probe] == 0:
    return
  acc = wp.vec3(0.0)
  for k in range(distance.count[texel]):
    dist = wp.min(wp.abs(ray_radiance[probe, distance.ray[texel, k]][3]), vol.max_distance)
    if not wp.isfinite(dist):
      continue
    w = distance.weight[texel, k]
    acc += wp.vec3(dist * w, dist * dist * w, w)
  distance_sum[probe, texel] = distance_sum[probe, texel] + acc


@wp.kernel
def resolve_static(
  # Data in:
  distance_sum: wp.array2d[wp.vec3],
  # Out:
  vol: Volume,
):
  """The static bake's distance moments of probe `probe`'s texel `texel` from its pooled sums."""
  probe, texel = wp.tid()
  d = distance_sum[probe, texel]
  # In units of the volume's max distance (`Volume.distance`: float16 squares overflow past 256 m).
  inv = 1.0 / wp.max(d[2], 1.0e-9)
  unit = 1.0 / vol.max_distance
  vol.distance[probe, texel] = wp.vec2h(wp.float16(d[0] * inv * unit), wp.float16(d[1] * inv * unit * unit))


@wp.func
def texel_centre(
  # Model:
  cube: Cube,
  # In:
  i: int,
  j: int,
  k: int,
) -> wp.vec3:
  """World position of the centre of cube texel (k, j, i)."""
  cell = wp.vec3(1.0 / cube.inv_cell[0], 1.0 / cube.inv_cell[1], 1.0 / cube.inv_cell[2])
  return cube.origin + wp.cw_mul(cell, wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5))


@wp.func
def _face_axis(
  # In:
  face: int,
) -> wp.vec3:
  """Unit axis of cube face `face` (2 a: +axis a, 2 a + 1: -axis a)."""
  axis = wp.vec3(0.0)
  axis[face // 2] = wp.where(face % 2 == 0, 1.0, -1.0)
  return axis


@wp.func
def contact_open(
  # Data in:
  ao: wp.array4d[wp.uint8],
  # In:
  i: int,
  j: int,
  k: int,
  face: int,
) -> float:
  """The open share (0-1) of texel (k, j, i)'s face `face` (`ao_kernel`)."""
  return float(ao[k, j, i, face]) * wp.static(1.0 / 255.0)


@wp.kernel
def resample_cube(
  # Model:
  vol: Volume,
  # Data in:
  gate: wp.array1d[int],
  rep: wp.array1d[wp.int32],
  ao: wp.array4d[wp.uint8],
  # In:
  nworld: int,
  row0: int,
  # Out:
  texels: wp.array4d[wp.float16],
):
  """Ambient-cube texel (k, j, i) of row row0 + r = k * ny + j of group g's six slabs (row0:
  the first row of a CPU band, `volume.ProbeVolume.refresh`): the probe lookup at
  the texel centre for each axis normal, biased along it by the normal bias; a face the probes do
  not see is flagged invalid (`cube.store_texel`, dilated by `cube.CubeTexture.finish`). Only the
  groups that updated (`group_used`)."""
  g, r, i = wp.tid()
  if not group_used(gate, rep, nworld, g):
    return
  cube = vol.cube
  k, j = (row0 + r) // cube.counts[1], (row0 + r) % cube.counts[1]
  p = texel_centre(cube, i, j, k)
  sight = wall_channel(cube, i, j, k, 3)
  row = g * vol.nprobe
  for face in range(wp.static(FACES)):
    axis = _face_axis(face)
    v, seen = lookup(vol, row, p, p + axis * vol.normal_bias, p, axis, sight)
    v = v * contact_open(ao, i, j, k, face)
    write_texel(cube, g * wp.static(FACES) + face, k, j, i, store_texel(v, seen >= wp.static(VISIBLE_MIN), 3), texels)


@wp.kernel
def advance(
  # In:
  gate: wp.array1d[int],
  count: int,
  # Out:
  updates: wp.array1d[int],
):
  """Counts `count` gated updates (the next one uses the next rotation and seed)."""
  if gate[0] != 0:
    updates[0] = updates[0] + count


@wp.struct
class SurveyLaunch:
  """One survey launch (`make_survey_ray`): `npoint` points from `first` (their `origins`, those
  still `moving`), RELOCATE_RAYS fixed directions each (thread (ray, point): ray-major), glass
  (`see_through`) passed; out: per ray its first hit as SURVEY_NONE / SURVEY_BACK / SURVEY_FRONT
  (`kinds`) and its distance (`dists`)."""

  see_through: wp.array1d[wp.int32]
  origins: wp.array1d[wp.vec3]
  moving: wp.array1d[wp.int32]
  first: int
  npoint: int
  kinds: wp.array2d[wp.int32]
  dists: wp.array2d[float]


@specialize
def make_survey_ray(trace_closest: BodyClosest):
  """survey_ray(geo, launch, r, point): survey ray r of point `launch.first + point` with this
  closest-hit query (unculled): the body of the software kernel (`survey_kernel`) and of the OptiX
  program (`rays.optix.device.programs`). The probe relocation (`relocate_step`) and the free-path
  survey (`volume.free_path`) reduce the rays. Rays ending on see-through geometry (single-sided
  window glass seen from behind) count neither as back nor as front faces."""

  @wp.func
  @device_generic
  def survey_ray[G: rays.GeometryFields](
    # Model:
    geo: G,
    # In:
    q: SurveyLaunch,
    r: int,
    point: int,
  ):
    """See `make_survey_ray`."""
    point += q.first
    if q.moving[point] == 0:
      return
    d = fibonacci_direction(r, wp.static(RELOCATE_RAYS))
    hit = trace_closest(geo, 0, q.origins[point], d, 1.0e30)
    kind = wp.static(SURVEY_NONE)
    if hit.instance_id != -1:
      passed = hit.instance_id >= 0 and hit.instance_id < q.see_through.shape[0] and q.see_through[hit.instance_id] != 0
      if not passed:
        kind = wp.where(back_face(geo, hit, d), wp.static(SURVEY_BACK), wp.static(SURVEY_FRONT))
    q.kinds[point, r] = kind
    q.dists[point, r] = hit.dist

  return survey_ray


@specialize
def survey_kernel(
  shape_types: tuple[int, ...], *, trees: Trees, alpha_cutout: bool, live_uv: bool = False, world_uv: bool = False
):
  """The survey's software ray kernel of these instance types and BVH trees (cached;
  `make_survey_ray`): one thread per (ray, point). `alpha_cutout`: its rays pass cut-out texels
  (Features.alpha_cutout), as the probe rays do (foliage does not bury probes)."""
  trace_closest, _ = query.unculled_queries(
    shape_types, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv
  )
  survey_ray = make_survey_ray(trace_closest)

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees))
  @concrete(geo=rays.software_geometry_type(live_uv=live_uv))
  def survey_rays(
    # Model:
    geo: rays.SoftwareGeometry,
    # In:
    q: SurveyLaunch,
  ):
    """Survey ray r of point `q.first + point` (`make_survey_ray`)."""
    r, point = wp.tid()
    survey_ray(geo, q, r, point)

  return survey_rays


@wp.kernel
def probe_positions(
  # Model:
  vol: Volume,
  # Out:
  out: wp.array1d[wp.vec3],
):
  """Probe tid's position (grid point + relocation offset)."""
  out[wp.tid()] = probe_position(vol, wp.tid())


@wp.kernel
def relocate_step(
  # Data in:
  kinds: wp.array2d[wp.int32],
  dists: wp.array2d[float],
  # In:
  last: int,
  first: int,
  # Out:
  moving: wp.array1d[wp.int32],
  origins: wp.array1d[wp.vec3],
  vol: Volume,
):
  """Probe `first + tid`, still moving, from its survey (`survey_kernel`, from `origins`):
  inside (mostly back faces) it moves to the closest back face, else away from a front face closer
  than MIN_FRONT spacing, at most 0.45 spacing from its grid point; it stops
  (classified: inactive when still inside) when it does not move or on the `last` step. The survey
  is a pure function of the position, so a probe that did not move has converged."""
  probe = first + wp.tid()
  if moving[probe] == 0:
    return
  backfaces = int(0)
  closest_back = float(1.0e27)
  closest_back_dir = wp.vec3(0.0)
  closest_front = float(1.0e27)
  closest_front_dir = wp.vec3(0.0)
  for r in range(wp.static(RELOCATE_RAYS)):
    kind = kinds[probe, r]
    dist = dists[probe, r]
    if kind == wp.static(SURVEY_BACK):
      backfaces += 1
      if dist < closest_back:
        closest_back = dist
        closest_back_dir = fibonacci_direction(r, wp.static(RELOCATE_RAYS))
    elif kind == wp.static(SURVEY_FRONT) and dist < closest_front:
      closest_front = dist
      closest_front_dir = fibonacci_direction(r, wp.static(RELOCATE_RAYS))
  fraction = float(backfaces) / float(wp.static(RELOCATE_RAYS))
  min_front = wp.static(MIN_FRONT) * wp.min(vol.spacing[0], wp.min(vol.spacing[1], vol.spacing[2]))
  offset = vol.offset[probe]
  full = offset
  moved = False
  if last == 0:
    if fraction > wp.static(BACKFACE_FRACTION):
      full = offset + closest_back_dir * (closest_back + 0.5 * min_front)
      moved = True
    elif closest_front < min_front:
      full = offset - closest_front_dir * (min_front - closest_front)
      moved = True
  # At most 0.45 spacing from the grid point (the cell's probes keep their order).
  normalized = wp.cw_div(full, vol.spacing)
  if wp.dot(normalized, normalized) >= 0.45 * 0.45:
    moved = False
  if moved:
    vol.offset[probe] = full
    origins[probe] = grid_position(vol, probe) + full
  else:
    moving[probe] = 0
    vol.active[probe] = wp.where(fraction > wp.static(BACKFACE_FRACTION), 0, 1)


@specialize
def _make_free(trace_closest: BodyClosest):
  """free(geo, see_through, origin, direction, length) -> the free distance from `origin` along
  `direction` in units of `length`, clamped to 1 (glass passed), with this closest-hit query."""

  @wp.func
  @device_generic
  def free[G: rays.GeometryFields](
    # Model:
    geo: G,
    see_through: wp.array1d[wp.int32],
    # In:
    origin: wp.vec3,
    direction: wp.vec3,
    length: float,
  ) -> float:
    """Distance to the first opaque surface (front or back face) along `direction` over
    `length` (1: none within it); see-through instances (window glass) are passed."""
    o = origin
    travelled = float(0.0)
    # A run-time bound: the loop is not unrolled (one traversal site per call: fast compiles).
    layers = int(SEE_THROUGH_LAYERS)
    for _layer in range(layers):
      hit = trace_closest(geo, 0, o, direction, length - travelled)
      if hit.instance_id == -1:
        return 1.0
      t = travelled + hit.dist
      if t >= length:
        return 1.0
      if hit.instance_id >= 0 and hit.instance_id < see_through.shape[0]:
        if see_through[hit.instance_id] != 0:
          travelled = t
          o = o + direction * (hit.dist + ray_offset(o + direction * hit.dist, offset_floor(geo, hit.instance_id)))
          continue
      return t / length
    return 1.0

  return free


@specialize
def _make_inside(trace_closest: BodyClosest):
  """inside(geo, see_through, p): whether `p` lies inside a closed solid, with this closest-hit
  query."""

  @wp.func
  @device_generic
  def inside[G: rays.GeometryFields](
    # Model:
    geo: G,
    see_through: wp.array1d[wp.int32],
    # In:
    p: wp.vec3,
  ) -> bool:
    """Whether at least INSIDE_AXES of the six axis rays from `p` first meet an opaque back face
    (one-sided geometry; two-sided geometry has no inside: `back_face`; glass is passed)."""
    backs = int(0)
    # Run-time bounds (not unrolled: one traversal site, fast compiles).
    axes = int(6)
    for d in range(axes):
      if backs + axes - d < wp.static(INSIDE_AXES):
        # Too few axes left to reach INSIDE_AXES (open space: after two front faces or misses).
        return False
      axis = wp.vec3(0.0)
      axis[d // 2] = wp.where(d % 2 == 0, 1.0, -1.0)
      o = p
      layers = int(SEE_THROUGH_LAYERS)
      for _layer in range(layers):
        hit = trace_closest(geo, 0, o, axis, wp.static(MISS_DISTANCE))
        if hit.instance_id == -1:
          break
        if hit.instance_id >= 0 and hit.instance_id < see_through.shape[0]:
          if see_through[hit.instance_id] != 0:
            o = o + axis * (hit.dist + ray_offset(o + axis * hit.dist, offset_floor(geo, hit.instance_id)))
            continue
        if back_face(geo, hit, axis):
          backs += 1
        break
    return backs >= wp.static(INSIDE_AXES)

  return inside


@wp.struct
class WallsLaunch:
  """One walls launch (`make_walls_texel`): every texel of volume `vol`'s cube (thread per texel,
  x fastest), glass (`see_through`) passed; out: the walls array (`cube.Cube.walls`)."""

  vol: Volume
  see_through: wp.array1d[wp.int32]
  walls: wp.array4d[wp.uint16]


@specialize
def make_walls_texel(trace_closest: BodyClosest):
  """walls_texel(geo, launch, i, j, k): texel (k, j, i)'s walls with this closest-hit query
  (unculled): per axis, the free distances [1/255 texel, rounded down: any wall within a texel is
  seen] towards +axis (high byte) and -axis (low byte); channel 3: bit k set when corner k of the
  probe cell around the texel centre is active and in sight of it (the resample's
  `probes.lookup` drops the others: the distance moments' Chebyshev test passes probes behind
  room corners); bit `cube.INSIDE_BIT` set when the centre lies inside a closed solid
  (`_make_inside`). The body of the software kernel (`walls_kernel`) and of the OptiX program."""
  free = _make_free(trace_closest)
  inside = _make_inside(trace_closest)

  @wp.func
  @device_generic
  def walls_texel[G: rays.GeometryFields](
    # Model:
    geo: G,
    # In:
    q: WallsLaunch,
    i: int,
    j: int,
    k: int,
  ):
    """See `make_walls_texel`."""
    vol = q.vol
    cube = vol.cube
    p = texel_centre(cube, i, j, k)
    base, alpha = probe_cell(vol, p)
    sight = int(0)
    # Run-time bounds (not unrolled: one traversal site per loop, fast compiles).
    corners = int(8)
    for c in range(corners):
      probe, _t = probe_corner(vol, base, alpha, c)
      if vol.active[probe] != 0:
        to = probe_position(vol, probe) - p
        dist = wp.length(to)
        if dist <= 1.0e-6 or free(geo, q.see_through, p, to / dist, dist) >= 1.0:
          sight = sight | (1 << c)
    if inside(geo, q.see_through, p):
      sight = sight | (1 << wp.static(INSIDE_BIT))
    q.walls[k, j, i, 3] = wp.uint16(sight)
    steps = float(wp.static(WALL_STEPS))
    directions = int(6)
    for d in range(directions):
      a = d // 2
      axis = wp.vec3(0.0)
      axis[a] = wp.where(d % 2 == 0, 1.0, -1.0)
      length = 1.0 / cube.inv_cell[a]
      step = int(wp.floor(free(geo, q.see_through, p, axis, length) * steps))
      if d % 2 == 0:
        q.walls[k, j, i, a] = wp.uint16(step * 256)
      else:
        q.walls[k, j, i, a] = wp.uint16(int(q.walls[k, j, i, a]) + step)

  return walls_texel


@specialize
def walls_kernel(
  shape_types: tuple[int, ...], *, trees: Trees, alpha_cutout: bool, live_uv: bool = False, world_uv: bool = False
):
  """The ambient cube's software walls kernel of these instance types and BVH trees (cached;
  `make_walls_texel`), geometry only."""
  trace_closest, _ = query.unculled_queries(
    shape_types, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv
  )
  walls_texel = make_walls_texel(trace_closest)

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees))
  @concrete(geo=rays.software_geometry_type(live_uv=live_uv))
  def trace_walls(
    # Model:
    geo: rays.SoftwareGeometry,
    # In:
    q: WallsLaunch,
  ):
    """Texel (k, j, i) (`make_walls_texel`)."""
    k, j, i = wp.tid()
    walls_texel(geo, q, i, j, k)

  return trace_walls


@wp.func
def _run(
  # Data in:
  walls: wp.array4d[wp.uint16],
  # In:
  start: wp.vec3i,
  a: int,
  step: int,
) -> int:
  """Inside texels from `start` (inside a solid) along axis `a` (`step` +-1) before the solid ends
  (an outside texel, the box's end, or a surface into another solid), at most COMPACT_TEXELS + 1."""
  at = wp.vec3i(start[0], start[1], start[2])
  n = walls.shape[2 - a]
  count = int(0)
  for _t in range(wp.static(COMPACT_TEXELS + 1)):
    if at[a] + step < 0 or at[a] + step >= n:
      return count
    if free_distance(int(walls[at[2], at[1], at[0], a]), step > 0) < 1.0:
      # A surface before the next texel: this solid ends (the next one is outside or another
      # solid's interior).
      return count
    at[a] = at[a] + step
    if (int(walls[at[2], at[1], at[0], 3]) >> wp.static(INSIDE_BIT)) & 1 == 0:
      return count
    count += 1
  return count


@wp.kernel
def compact_solids(
  # Out:
  walls: wp.array4d[wp.uint16],
):
  """Texel (k, j, i) inside a solid: `cube.COMPACT_BIT` when its solid spans at most
  COMPACT_TEXELS texels along each axis through it (an object in a room: one space all around it,
  whose fill holds that space's values), not a wall, floor or slab (long along two axes: the spaces
  on either side differ)."""
  k, j, i = wp.tid()
  bits = int(walls[k, j, i, 3])
  if (bits >> wp.static(INSIDE_BIT)) & 1 == 0:
    return
  c = wp.vec3i(i, j, k)
  for a in range(3):
    if 1 + _run(walls, c, a, 1) + _run(walls, c, a, -1) > wp.static(COMPACT_TEXELS):
      return
  walls[k, j, i, 3] = wp.uint16(bits | (1 << wp.static(COMPACT_BIT)))


@wp.kernel
def compact_neighbours(
  # Out:
  walls: wp.array4d[wp.uint16],
):
  """Texel (k, j, i) outside solids: bit `cube.SOLID_SHIFT` + d when its neighbour along direction
  d (2 a: +a, 2 a + 1: -a) is inside a compact solid (`compact_solids`): a footprint may reach it
  without a leak."""
  k, j, i = wp.tid()
  bits = int(walls[k, j, i, 3])
  if (bits >> wp.static(INSIDE_BIT)) & 1 != 0:
    return
  n = wp.vec3i(walls.shape[2], walls.shape[1], walls.shape[0])
  for d in range(6):
    a = d // 2
    near = wp.vec3i(i, j, k)
    near[a] = near[a] + wp.where(d % 2 == 0, 1, -1)
    if near[a] >= 0 and near[a] < n[a]:
      if (int(walls[near[2], near[1], near[0], 3]) >> wp.static(COMPACT_BIT)) & 1 != 0:
        bits = bits | (1 << (wp.static(SOLID_SHIFT) + d))
  walls[k, j, i, 3] = wp.uint16(bits)


@wp.kernel
def near_walls(
  # Data in:
  walls: wp.array4d[wp.uint16],
  # Out:
  ao: wp.array4d[wp.uint8],
  near: wp.array1d[wp.int32],
):
  """Texel (k, j, i): open contact occlusion (255 per face), and whether `ao_kernel` traces it: 1
  outside solids where an axis meets a wall within a texel (else nothing close: the probes' own
  occlusion holds)."""
  k, j, i = wp.tid()
  for f in range(wp.static(FACES)):
    ao[k, j, i, f] = wp.uint8(255)
  t = (k * walls.shape[1] + j) * walls.shape[2] + i
  near[t] = 0
  if (int(walls[k, j, i, 3]) >> wp.static(INSIDE_BIT)) & 1 != 0:
    return
  for a in range(3):
    raw = int(walls[k, j, i, a])
    if free_distance(raw, True) < 1.0 or free_distance(raw, False) < 1.0:
      near[t] = 1


@wp.kernel
def list_cells(
  # Data in:
  flags: wp.array1d[wp.int32],
  index: wp.array1d[wp.int32],
  # In:
  nx: int,
  ny: int,
  # Out:
  cells: wp.array1d[wp.vec3i],
):
  """The (i, j, k) of flagged texel t at its place in the exclusive scan `index`."""
  t = wp.tid()
  if flags[t] != 0:
    cells[index[t]] = wp.vec3i(t % nx, (t // nx) % ny, t // (nx * ny))


@wp.struct
class AoLaunch:
  """One contact-occlusion launch (`make_ao_texel`): the listed texels `cells` of cube `cube`
  (thread per texel), AO_RAYS fixed rays each reaching `reach`, glass and emitters
  (`see_through`) passed; out: the contact occlusion per texel face (`ao`)."""

  cube: Cube
  see_through: wp.array1d[wp.int32]
  cells: wp.array1d[wp.vec3i]
  reach: float
  ao: wp.array4d[wp.uint8]


@specialize
def make_ao_texel(trace_closest: BodyClosest):
  """ao_texel(geo, launch, t): texel `launch.cells[t]` (one near a wall, `near_walls`), with this
  closest-hit query (unculled): per face (+x, -x, +y, -y, +z, -z), 255 x (1 - the cosine-weighted
  occlusion of the AO_RAYS fixed rays from its centre), a ray occluding by
  CONTACT_AO_STRENGTH x (1 - its free distance / `reach`) (a smooth falloff). The body of the
  software kernel (`ao_kernel`) and of the OptiX program."""
  free = _make_free(trace_closest)

  @wp.func
  @device_generic
  def ao_texel[G: rays.GeometryFields](
    # Model:
    geo: G,
    # In:
    q: AoLaunch,
    t: int,
  ):
    """See `make_ao_texel`."""
    c = q.cells[t]
    p = texel_centre(q.cube, c[0], c[1], c[2])
    occluded = wp.vector(0.0, length=6, dtype=float)
    weight = wp.vector(0.0, length=6, dtype=float)
    for r in range(wp.static(AO_RAYS)):
      d = fibonacci_direction(r, wp.static(AO_RAYS))
      occ = (1.0 - free(geo, q.see_through, p, d, q.reach)) * wp.static(CONTACT_AO_STRENGTH)
      for f in range(6):
        w = wp.max(0.0, wp.dot(d, _face_axis(f)))
        occluded[f] += occ * w
        weight[f] += w
    for f in range(6):
      q.ao[c[2], c[1], c[0], f] = wp.uint8(int(wp.round(255.0 * (1.0 - occluded[f] / wp.max(weight[f], 1.0e-9)))))

  return ao_texel


@specialize
def ao_kernel(shape_types: tuple[int, ...], *, trees: Trees, alpha_cutout: bool, live_uv: bool = False, world_uv: bool = False):
  """The ambient cube's software contact-occlusion kernel of these instance types and BVH trees
  (cached; `make_ao_texel`): one thread per listed texel (`volume.ProbeVolume.place`: the texels
  with nothing close stay open, `near_walls`)."""
  trace_closest, _ = query.unculled_queries(
    shape_types, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv
  )
  ao_texel = make_ao_texel(trace_closest)

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees))
  @concrete(geo=rays.software_geometry_type(live_uv=live_uv))
  def trace_ao(
    # Model:
    geo: rays.SoftwareGeometry,
    # In:
    q: AoLaunch,
  ):
    """Texel `q.cells[tid]` (`make_ao_texel`)."""
    ao_texel(geo, q, wp.tid())

  return trace_ao


@wp.struct
class StaticLaunch:
  """One static-bake launch (`make_static_ray`): `nprobe` probes from `first` of a volume,
  `nray` rays each (thread (ray, probe): ray-major) along `dirs`, at most `length` long, window
  glass (`see_through`) passed; out: per ray the hit distance (negative: a back face;
  MISS_DISTANCE: none)."""

  # The volume's grid (`probes.probe_position`): its origin, spacing, counts, relocation offsets
  # and active flags (not the whole `Volume`: the OptiX launch parameters stay small).
  origin: wp.vec3
  spacing: wp.vec3
  counts: wp.vec3i
  offset: wp.array1d[wp.vec3]
  active: wp.array1d[wp.int32]
  see_through: wp.array1d[wp.int32]
  dirs: wp.array1d[wp.vec3]
  length: float
  first: int
  nprobe: int
  nray: int
  ray_radiance: wp.array2d[wp.vec4]


@specialize
def make_static_ray(trace_closest: BodyClosest):
  """static_ray(geo, launch, r, probe): static ray r of probe `launch.first + probe` with this
  closest-hit query (unculled: `query.unculled_queries`, or the OptiX scene's): the body of the
  software kernel (`static_kernel`) and of the OptiX program (`rays.optix.device.programs`). The
  GI's rays end at `Volume.max_distance` (its distance maps' clamp: a shorter traversal)."""

  @wp.func
  @device_generic
  def static_ray[G: rays.GeometryFields](
    # Model:
    geo: G,
    # In:
    q: StaticLaunch,
    r: int,
    probe: int,
  ):
    """See `make_static_ray`."""
    probe += q.first
    q.ray_radiance[probe, r] = wp.vec4(0.0, 0.0, 0.0, wp.static(MISS_DISTANCE))
    if q.active[probe] == 0:
      return
    d = q.dirs[r]
    x = probe % q.counts[0]
    y = (probe // q.counts[0]) % q.counts[1]
    z = probe // (q.counts[0] * q.counts[1])
    origin = q.origin + wp.cw_mul(q.spacing, wp.vec3(float(x), float(y), float(z))) + q.offset[probe]
    travelled = float(0.0)
    for _layer in range(wp.static(SEE_THROUGH_LAYERS)):
      hit = trace_closest(geo, 0, origin, d, q.length - travelled)
      if hit.instance_id == -1:
        return
      t = travelled + hit.dist
      if hit.instance_id >= 0 and hit.instance_id < q.see_through.shape[0]:
        if q.see_through[hit.instance_id] != 0:
          travelled = t
          origin = origin + d * (hit.dist + ray_offset(origin + d * hit.dist, offset_floor(geo, hit.instance_id)))
          continue
      if back_face(geo, hit, d):
        t = t * wp.static(BACKFACE_DISTANCE)
      q.ray_radiance[probe, r] = wp.vec4(0.0, 0.0, 0.0, t)
      return

  return static_ray


@specialize
def static_kernel(
  shape_types: tuple[int, ...], *, trees: Trees, alpha_cutout: bool, live_uv: bool = False, world_uv: bool = False
):
  """The static bake's software ray kernel (cached; `make_static_ray`): one thread per (ray,
  probe) (ray-major: a warp traces one direction from neighbouring probes, `make_trace_kernel`),
  geometry only, no back-face culling (`volume.ProbeVolume.bake_static`)."""
  trace_closest, _ = query.unculled_queries(
    shape_types, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv
  )
  static_ray = make_static_ray(trace_closest)

  @wp.kernel(module="unique", enable_backward=False, module_options=query.module_options(trees))
  @concrete(geo=rays.software_geometry_type(live_uv=live_uv))
  def trace_static(
    # Model:
    geo: rays.SoftwareGeometry,
    # In:
    q: StaticLaunch,
  ):
    """Static ray r of probe `q.first + probe` (`make_static_ray`)."""
    r, probe = wp.tid()
    static_ray(geo, q, r, probe)

  return trace_static


@wp.struct
class ProbeLaunch:
  """One probe-ray launch of an update (`make_probe_ray`): `nprobe` probes from `first` of the
  launch's groups from `first_group`, `nray` rays each (thread (group, ray, probe): ray-major),
  the update's directions and counter, its gate, the groups' representative worlds; out: the ray
  results and, unless empty, their hits (`RayHits`)."""

  dirs: wp.array1d[wp.vec3]
  updates: wp.array1d[int]
  gate: wp.array1d[int]
  rep: wp.array1d[wp.int32]
  nworld: int
  nray: int
  spread: float
  first_group: int
  first: int
  nprobe: int
  ray_radiance: wp.array2d[wp.vec4]
  ray_sky: wp.array2d[float]
  hits: RayHits


@specialize
def make_probe_ray(shade_probe_ray: ShadeProbeRay):
  """probe_ray(geo, mats, lit, env, launch, i, r, probe): ray r of probe `launch.first + probe`
  of group `launch.first_group + i`, shaded by `shade_probe_ray` (the megakernel's probe shading,
  kernels/probe.py: probe hits are lit by the camera rays' light code): its (radiance, hit
  distance), sky weight and `RayHit`, in the group's representative world (ray rows: the groups
  from `first_group`). The body of the software kernel (`make_trace_kernel`) and of the OptiX
  program (`rays.optix.device.programs`)."""
  Environment = kernel_structs.EnvironmentArrays

  @wp.func
  @device_generic
  def probe_ray[G: rays.GeometryFields, M: MaterialTables](
    # Model:
    geo: G,
    mats: M,
    lit: Lights,
    env: Environment,
    # In:
    q: ProbeLaunch,
    i: int,
    r: int,
    probe: int,
  ):
    """See `make_probe_ray`."""
    probe += q.first
    g = q.first_group + i
    if not group_used(q.gate, q.rep, q.nworld, g):
      return
    vol = env.gi
    row = i * vol.nprobe + probe
    record = q.hits.at.shape[0] != 0
    if vol.active[probe] == 0:
      q.ray_radiance[row, r] = wp.vec4(0.0, 0.0, 0.0, wp.static(MISS_DISTANCE))
      q.ray_sky[row, r] = 0.0
      if record:
        q.hits.fixed[row, r] = wp.vec3(0.0)
        q.hits.at[row, r] = -1.0
      return
    update_seed = wp.static(SEED_STRIDE) * q.updates[0]
    radiance, dist, sky, hit = shade_probe_ray(
      geo, mats, lit, env, q.rep[g], probe_position(vol, probe), q.dirs[r], q.spread, update_seed, probe * q.nray + r
    )
    q.ray_radiance[row, r] = wp.vec4(radiance[0], radiance[1], radiance[2], dist)
    q.ray_sky[row, r] = sky
    if record:
      q.hits.fixed[row, r] = hit.fixed
      q.hits.weight[row, r] = wp.vec3h(hit.weight)
      q.hits.at[row, r] = hit.at
      q.hits.normals[row, r] = wp.vec2ui(pack_unit(hit.normal), pack_unit(hit.outward))

  return probe_ray


@specialize
def make_trace_kernel(f: Features, shade_probe_ray: ShadeProbeRay):
  """The software probe-ray kernel of Features `f` (`make_probe_ray`; cached): one thread per
  (group, ray, probe). Ray-major: a warp traces one direction from neighbouring probes, coherent
  traversals and light loops (a 30 k-probe office interior on an RTX PRO 6000: 98 Mrays/s against
  51 probe-major; the primary hits alone 1126 against 602)."""
  Environment = kernel_structs.EnvironmentArrays
  probe_ray = make_probe_ray(shade_probe_ray)

  @wp.kernel(module="unique", enable_backward=False, module_options=kernel_structs.module_options(f))
  @device_generic
  @concrete(
    geo=rays.software_geometry_type(live_uv=f.live_material_uv),
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  def trace_probe_rays(
    # Model:
    geo: rays.SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    env: Environment,
    # In:
    q: ProbeLaunch,
  ):
    """Ray r of probe `q.first + probe` of group `q.first_group + i` (`make_probe_ray`)."""
    i, r, probe = wp.tid()
    probe_ray(geo, mats, lit, env, q, i, r, probe)

  return trace_probe_rays


@specialize
def make_relight_kernel(f: Features, shade_probe_ray: ShadeProbeRay):
  """The per-world relight kernel of Features `f` (`relight.Relight`; cached): one thread per
  (row, cell, sample probe), the probe's fixed rays shaded in the row's world and summed in the
  cosine lobe of each axis direction (+x, -x, +y, -y, +z, -z: entries s x 6 + face).

  A row is world `row` and runs when the gate is open or its world changed; or, with
  `fixed_world` >= 0, the shared reference (that world, when the gate is open). The rays are the
  same for every world and frame (fixed directions and seeds), so the sums of two worlds differ
  by their inputs only; a non-finite ray (a NaN light value) is dropped from its row's sums.
  """
  Environment = kernel_structs.EnvironmentArrays

  @wp.kernel(module="unique", enable_backward=False, module_options=kernel_structs.module_options(f))
  @device_generic
  @concrete(
    geo=rays.software_geometry_type(live_uv=f.live_material_uv),
    mats=material_type(diffuse_roughness=f.diffuse_roughness, specular_weight=f.specular_weight, layered=f.layered),
  )
  def relight_rays(
    # Model:
    geo: rays.SoftwareGeometry,
    mats: MaterialTables,
    lit: Lights,
    env: Environment,
    # Data in:
    dirs: wp.array1d[wp.vec3],
    samples: wp.array2d[wp.int32],
    gate: wp.array1d[int],
    changed: wp.array1d[wp.int32],
    # In:
    fixed_world: int,
    spread: float,
    # Out:
    sums: wp.array3d[wp.vec3],
  ):
    """Cosine-weighted sums of the radiance of the fixed rays of sample probe s of cell c, per
    axis direction, in the row's world."""
    row, c, s = wp.tid()
    w = row
    if fixed_world >= 0:
      w = fixed_world
      if gate[0] == 0:
        return
    elif gate[0] == 0 and changed[w] == 0:
      return
    vol = env.gi
    probe = samples[c, s]
    px = wp.vec3(0.0)
    nx = wp.vec3(0.0)
    py = wp.vec3(0.0)
    ny = wp.vec3(0.0)
    pz = wp.vec3(0.0)
    nz = wp.vec3(0.0)
    if vol.active[probe] != 0:
      origin = probe_position(vol, probe)
      nray = dirs.shape[0]
      for r in range(nray):
        d = dirs[r]
        radiance, _dist, _sky, _hit = shade_probe_ray(
          geo, mats, lit, env, w, origin, d, spread, 0, (c * samples.shape[1] + s) * nray + r
        )
        if not wp.isfinite(radiance):
          continue
        px += radiance * wp.max(d[0], 0.0)
        nx += radiance * wp.max(-d[0], 0.0)
        py += radiance * wp.max(d[1], 0.0)
        ny += radiance * wp.max(-d[1], 0.0)
        pz += radiance * wp.max(d[2], 0.0)
        nz += radiance * wp.max(-d[2], 0.0)
    at = s * 6
    sums[row, c, at] = px
    sums[row, c, at + 1] = nx
    sums[row, c, at + 2] = py
    sums[row, c, at + 3] = ny
    sums[row, c, at + 4] = pz
    sums[row, c, at + 5] = nz

  return relight_rays
