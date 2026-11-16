"""Ray-query types: the `Ray`, `Hit` and geometry structs and the `Tracer` contract.

The geometry structs gather every array a ray query reads: the common part (`GeometryFields`)
plus the software traversal's (`SoftwareGeometry`) or the OptiX backend's (`OptixGeometry`). A
`Tracer` is the set of ray queries a kernel body calls (`pixel.make_render_pixel`), built by a trace
backend (`core/trace/backend.py`) for some `TraceFeatures`: the software backend's come from
`query.make_trace` (MuJoCo Warp's scene BVH + per-instance intersection, `cast.py` / `ordered.py`,
specialised on the instance types present); the OptiX backend's (`trace/optix/device/trace.py`) have
the same signatures.

Mesh BVH frames: MuJoCo stores mesh vertices in the mesh's principal inertia frame, which for
CAD-style assets is an arbitrary rotation of the authored, axis-aligned frame. Long axis-aligned
triangles then get large, overlapping bounding boxes and the mesh BVH degrades (two orders of
magnitude on some assets). The software traversal can therefore query a mesh through a copy
built in a better-aligned frame `R` (vertices `R v`): ray queries read the rotation of a mesh
instance from `instance_rot_trace`, which holds `instance_rot @ R^T` (`mesh_trace_frames`, written
for the mesh columns only), so distances, faces, barycentrics and world normals are unchanged.

Instance frames: a primitive's rotation is a rotation R (its size scales it); a mesh instance's
may carry a uniform scale, s R (s > 0: `pipeline/upload/meshes.py` bakes only the rest of a stretch
into a mesh copy). Ray queries map the world ray into the mesh frame with `frame_inverse` (affine:
the distance t is the world one), normals with s R (normalised: unchanged by s); shading works in
the rotated frame in world units and scales what it reads from mesh vertices (`frame_scale`).
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import Final, NamedTuple, Protocol, cast, final

import numpy as np
import warp as wp

from raywarp._base.types import MaterialUvMapping, MeshAdr, ShapeType, uv_xform
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.tables import in_table, table_row
from raywarp.core.kernels.abi.toggles import Toggles
from raywarp.core.kernels.generic import device_generic, specialize, struct_schema
from raywarp.core.shade.materials.material import MaterialTables

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Origin offset of secondary rays along the surface normal (as in MuJoCo Warp) in a scene of
# SCENE_REFERENCE or more, and its share of the point's largest coordinate where that is larger
# (`ray_offset`). The floor scales with a smaller scene (`ray_floors`: x its extent over
# SCENE_REFERENCE; a fixed 1e-4 m was a tenth of a millimetre-scale room), and with an instance's
# largest vertex coordinate in its frame (RAY_OFFSET_VERTEX of it, 16 float32 ulps: a 2 x 10 km
# ground quad's hit points round by ~1e-3 m, which a 1e-4 m offset left under the surface: acne).
RAY_OFFSET: Final = 1.0e-4
RAY_OFFSET_RELATIVE: Final = 1.0 / 65536.0
RAY_OFFSET_VERTEX: Final = 2.0**-20
# The smallest scene extent [m] that keeps the metre-scale floors (`scene_scale`): a 10 cm object on
# its own (0.1 % of it the offset floor), so a robot or tabletop scene keeps today's values.
SCENE_REFERENCE: Final = 0.1
# Length of shadow rays towards directional lights and the environment.
UNBOUNDED_SHADOW_DIST: Final = 1.0e8
# An instance frame whose squared column length is within this of 1 is a rotation (a float32
# rotation's rounding is ~1e-7; the upload keeps scales within 1e-5 of 1 out:
# `pipeline/upload/meshes.py`).
FRAME_UNIT_TOL: Final = 2.0e-5


@wp.func
def ray_offset(
  # In:
  p: wp.vec3,
  floor: float,
) -> float:
  """Offset of a secondary ray leaving point `p`: `floor` (`offset_floor`: RAY_OFFSET in a scene
  of a metre or more), or RAY_OFFSET_RELATIVE x |p|_max where larger (beyond 6.6 m at the metre
  floor).

  A float32 hit point carries a rounding error proportional to its magnitude (the ray's origin,
  distance and the intersection all round): at 1 km a fixed 1e-4 m offset is below it and shadow
  rays start under the surface (acne). 2^-16 of the magnitude is Wachter & Binder's 256 ulps
  (Ray Tracing Gems ch. 6).
  """
  m = wp.max(wp.abs(p[0]), wp.max(wp.abs(p[1]), wp.abs(p[2])))
  return wp.max(floor, m * wp.static(RAY_OFFSET_RELATIVE))


def _sized(
  radius: np.ndarray[[int], np.dtype[np.float64]], types: np.ndarray[[int], np.dtype[np.int32]]
) -> np.ndarray[[int], np.dtype[np.bool_]]:
  """The instances with a size: a finite bound radius > 0, not a plane (an upload's unbounded
  plane has a radius of 1e6 m; MuJoCo's 0)."""
  return np.isfinite(radius) & (radius > 0.0) & (np.asarray(types) != int(ShapeType.PLANE))


def scene_scale(
  radius: np.ndarray[[int], np.dtype[np.float32]],
  pos: np.ndarray[[int, 3], np.dtype[np.float32]],
  types: np.ndarray[[int], np.dtype[np.int32]],
) -> float:
  """The share of the metre-scale floors a scene keeps (`ray_floors`, `punctual.falloff_floor`):
  the largest side of the box around its sized instances (`_sized`: bound radius `radius` about
  their initial positions `pos`, types `types`) over SCENE_REFERENCE, at most 1 (none: 1)."""
  r = np.asarray(radius, np.float64)
  sized = _sized(r, types)
  if not sized.any():
    return 1.0
  p = np.asarray(pos, np.float64)[sized]
  lo, hi = (p - r[sized, None]).min(axis=0), (p + r[sized, None]).max(axis=0)
  return min(float((hi - lo).max()) / SCENE_REFERENCE, 1.0)


def ray_floors(
  radius: np.ndarray[[int], np.dtype[np.float32]],
  pos: np.ndarray[[int, 3], np.dtype[np.float32]],
  types: np.ndarray[[int], np.dtype[np.int32]],
) -> np.ndarray[[int], np.dtype[np.float32]]:
  """(n + 1,) `ray_offset`'s floor of the rays leaving each of n instances (bound radius
  `radius` about its origin, the largest vertex magnitude of a mesh; initial position `pos`;
  types `types`), then the scene's (deformables, rays from no surface; `offset_floor`). The
  scene's floor is RAY_OFFSET x `scene_scale`; a sized instance's (not a plane) the larger of it
  and RAY_OFFSET_VERTEX x its radius."""
  r = np.asarray(radius, np.float64)
  scene = RAY_OFFSET * scene_scale(radius, pos, types)
  own = RAY_OFFSET_VERTEX * np.where(_sized(r, types), r, 0.0)
  return np.append(np.maximum(own, scene), scene).astype(np.float32)


@wp.func
@device_generic
def offset_floor[G: GeometryFields](
  # Model:
  geo: G,
  # In:
  instance: int,
) -> float:
  """`ray_offset`'s floor of a ray leaving instance `instance` (`GeometryFields.ray_floor`; a
  deformable or no surface, < 0: the scene's)."""
  last = geo.ray_floor.shape[0] - 1
  return geo.ray_floor[wp.where(instance >= 0 and instance < last, instance, last)]


@wp.func
def frame_scale_sq(
  # In:
  frame: wp.mat33,
) -> float:
  """s^2 of an instance frame s R (R a rotation; s > 0 on mesh instances,
  `pipeline/upload/meshes.py`): exactly 1 within `FRAME_UNIT_TOL` of it (a rotation's float32
  rounding), so a rigid frame takes the very arithmetic of a rotation."""
  s2 = frame[0, 0] * frame[0, 0] + frame[1, 0] * frame[1, 0] + frame[2, 0] * frame[2, 0]
  return wp.where(wp.abs(s2 - 1.0) <= wp.static(FRAME_UNIT_TOL), 1.0, s2)


@wp.func
def frame_scale(
  # In:
  frame: wp.mat33,
) -> float:
  """The uniform scale s of an instance frame s R (1 for a rotation, `frame_scale_sq`)."""
  s2 = frame_scale_sq(frame)
  return wp.where(s2 == 1.0, 1.0, wp.sqrt(s2))


@wp.func
def frame_inverse(
  # In:
  frame: wp.mat33,
) -> wp.mat33:
  """The inverse (s R)^T / s^2 of an instance frame s R (a rotation's transpose, exactly)."""
  return wp.transpose(frame) * (1.0 / frame_scale_sq(frame))


@wp.struct
class Ray:
  """A world-space ray query.

  Attributes:
    origin: start point.
    direction: unit direction.
    max_dist: any-hit queries ignore hits at or beyond this distance; closest-hit queries keep
      those within its tie window (`instances.closer`: RAY_TOL_REL x max(max_dist, 1) beyond).
    cull: drop back-facing hits (except instances flagged two-sided, `Features.two_sided`).
  """

  origin: wp.vec3
  direction: wp.vec3
  max_dist: float
  cull: bool


# `Hit.instance_id` of a deformable (flex) hit (-1: a miss).
FLEX_HIT: Final = -2


@wp.struct
class FlexPart:
  """A deformable part as its hits are shaded: its color, its world-space triangle mesh (every
  world's triangles grouped by world, `trace/accel/deformable_bvh.py`; 0 for a dim-1 part), material
  id (-1: none), triangles per world of its mesh, its first corner's row of `flex_uv` (-1: no
  UVs)."""

  rgba: wp.vec4
  mesh: wp.uint64
  material: wp.int32
  nface: wp.int32
  uvadr: wp.int32


@wp.struct
class Hit:
  """A ray-query result (closest hit, or the first hit found by an any-hit query).

  Attributes:
    instance_id: hit instance, -1 on a miss, FLEX_HIT for flex (then mesh_id holds the flex id).
    dist: distance along the (unit) ray direction.
    normal: geometric normal in world frame.
    bary_u: barycentric coordinate of the first vertex (meshes).
    bary_v: barycentric coordinate of the second vertex (meshes).
    face_id: triangle index within the mesh, -1 for primitives.
    mesh_id: mesh id for meshes, flex id for flex hits.
  """

  instance_id: int
  dist: float
  normal: wp.vec3
  bary_u: float
  bary_v: float
  face_id: int
  mesh_id: int


@wp.func
def new_ray(
  # In:
  origin: wp.vec3,
  direction: wp.vec3,
  max_dist: float,
  cull: bool,
) -> Ray:
  """A `Ray` from its fields."""
  ray = Ray()
  ray.origin = origin
  ray.direction = direction
  ray.max_dist = max_dist
  ray.cull = cull
  return ray


@wp.func
def no_hit(
  # In:
  dist: float,
) -> Hit:
  """A miss at distance `dist` (instance -1, no face or mesh)."""
  hit = Hit()
  hit.instance_id = -1
  hit.dist = dist
  hit.normal = wp.vec3(0.0)
  hit.bary_u = 0.0
  hit.bary_v = 0.0
  hit.face_id = -1
  hit.mesh_id = -1
  return hit


@wp.struct
class MeshInstance:
  """How an instance reads its mesh (static; one `GeometryFields.mesh_instance` entry per instance).

  One struct array: the OptiX launch parameters hold every array field of `OptixGeometry` and are
  near their size limit (`trace/optix/device/params.PARAMS_LIMIT`).

  Attributes:
    uv_xform: affine transform of the UVs it reads (with own UVs only).
    texcoord: (texcoord address, corner row address into `facetexcoord`) of its own UVs, or
      (UV_MESH, -) for its mesh's (set 0, no transform).
  """

  uv_xform: uv_xform
  texcoord: wp.vec2i


@wp.struct
class Cutout:
  """Arrays of the alpha test of cut-out materials (`core/trace/accel/cutout.py`).

  Read only by kernels compiled with `Features.alpha_cutout`; the others carry the descriptors.
  """

  # Cutoff on the opacity texel per material (1 or nworld, nmat); 0: no test.
  material_alpha_cutoff: wp.array2d[float]
  material_texture: wp.array3d[int]
  material_tex_repeat: wp.array2d[wp.vec2]
  material_tex_uniform: wp.array2d[bool]
  # World-projected UV map per material (`material.uv_world`; read by the uv output with
  # Features.world_uv: the coordinate the maps read).
  material_uv_world: wp.array2d[uv_xform]
  mesh_adr: wp.array1d[MeshAdr]
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  textures: wp.array1d[wp.Texture2D]
  tex_layout: wp.array1d[int]


@wp.struct
class LiveUvCutout:
  """`Cutout` plus the live material UV mappings (each material's UV set and transform, each
  mesh's second-set addresses: `MaterialUvTables`), read by the surface, output and alpha-test UVs.

  Only with `Features.live_material_uv`; the other builds keep the smaller `Cutout`.
  """

  # Cutoff on the opacity texel per material (1 or nworld, nmat); 0: no test.
  material_alpha_cutoff: wp.array2d[float]
  material_texture: wp.array3d[int]
  material_tex_repeat: wp.array2d[wp.vec2]
  material_tex_uniform: wp.array2d[bool]
  # World-projected UV map per material (`material.uv_world`; read by the uv output with
  # Features.world_uv: the coordinate the maps read).
  material_uv_world: wp.array2d[uv_xform]
  mesh_adr: wp.array1d[MeshAdr]
  mesh_facetexcoord: wp.array1d[wp.vec3i]
  mesh_texcoord: wp.array1d[wp.vec2]
  textures: wp.array1d[wp.Texture2D]
  tex_layout: wp.array1d[int]

  material_uv_mapping: wp.array1d[MaterialUvMapping]
  mesh_uv_second: wp.array1d[wp.vec2i]


type CutoutTables = Cutout | LiveUvCutout


@struct_schema
class GeometryFields(Protocol):
  """The geometry arrays every ray query backend and the surface evaluation read (the common part
  of `SoftwareGeometry` and `OptixGeometry`: device funcs take any of them, `geo: G` with
  `G: GeometryFields`, `core/kernels/generic.py`)."""

  # Model:
  instance_type: wp.array1d[int]
  instance_mesh: wp.array2d[int]
  instance_material: wp.array2d[int]
  instance_size: wp.array2d[wp.vec3]
  instance_color: wp.array2d[wp.vec4]
  material_base_color: wp.array2d[wp.vec4]
  # How instance_color combines with material_base_color (`ColorRule`, `GpuScene.color_rule`).
  color_rule: int
  # The run-time settings of a general kernel (`kernels/abi/toggles.py`; unread by a specialised
  # one).
  rt: Toggles
  # The build word's geometry copy: every bound bit of an OptiX build (`Features.bound`,
  # `kernels/abi/toggles.py`) the scene uses (`trace/optix/device/programs.build_word`); the OptiX
  # module binds it (`trace/optix/device/params.bound_words`). 0 elsewhere (never read). Both halves
  # (`toggles.store`): the material code reads the second from here (`Surface.bits2`; the carrier
  # holds the first only).
  bits: int
  bits2: int
  # Data:
  # Per-world poses (nworld, ncolumn): instance i at column pose_slot[i] (Features.compact_poses),
  # else at column i.
  instance_pos: wp.array2d[wp.vec3]
  instance_rot: wp.array2d[wp.mat33]
  # Shadow-ray RGB transmittance of each instance (translucent shadows; 0: opaque); with
  # Features.world_optics world-major per (world, instance) (`world_entry`).
  instance_shadow_transmit: wp.array1d[wp.vec3]
  # Per instance (per world as `instance_shadow_transmit`): `TWO_SIDED` never back-face culled (USD
  # doubleSided, Features.two_sided), plus `CLOSED_PANE` for a closed see-through pane.
  instance_twosided: wp.array1d[int]
  # The analytic planes' instance ids, intersected beside the backend's structures (run-time data:
  # software, the per-world planes `sharing.Layout.planes`; OptiX, every rendered plane,
  # `trace.optix.accel.OptixScene.plane_ids`; empty: none).
  plane_ids: wp.array1d[int]
  # Shared scenery (Features.shared, core/pipeline/worlds/sharing.py). pose_slot (ninstance,):
  # per-world column (>= 0) or -1 - shared slot. shared_pos / shared_rot: the shared poses by slot
  # (compact poses: the shared pose table; else world 0's row of instance_pos / instance_rot, slot =
  # instance; shared mesh instances read their trace frame from world 0's instance_rot_trace).
  pose_slot: wp.array1d[int]
  shared_pos: wp.array1d[wp.vec3]
  shared_rot: wp.array1d[wp.mat33]
  # Per-world visibility (Features.visibility): (nworld or 1, nvisible), 0 hidden, at column
  # visible_slot[i] of instance i (-1: shown in every world; `sharing.Layout.visible_slot`).
  instance_visible: wp.array2d[int]
  visible_slot: wp.array1d[int]
  # How each instance reads its mesh (`GpuScene.mesh_instances`):
  # own UVs (Features.uv_maps; also the alpha test).
  mesh_instance: wp.array1d[MeshInstance]
  # Members of the per-world structure (software: the world tree's instances; OptiX: triangle
  # instance records).
  enabled_instance_ids: wp.array1d[int]
  # The mesh count every mesh id is bounded with (the alpha test, the software mesh queries).
  nmesh: int
  # Deformables: per part (`FlexPart`, the backend's: `DeformableBvhs.parts`) and the UVs of every
  # triangle corner of the parts with UVs (`Deformables.corner_uv`), read by their shading
  # (`core/shade/materials/flex.py`).
  flex_part: wp.array1d[FlexPart]
  flex_uv: wp.array1d[wp.vec2]
  # Alpha test of cut-out materials (Features.alpha_cutout): a `Cutout`, or an OptiX shaded
  # build's material carrier (`optix_geometry_type(carrier=...)`), which holds the same tables.
  cut: CutoutTables | MaterialTables
  # Light-cost counters (nworld, kernels/abi/counters.NCOUNTER), written only by Features.counters
  # builds.
  counters: wp.array2d[wp.int64]
  # The secondary rays' origin offset floor per instance, then the scene's (`ray_floors`,
  # `offset_floor`).
  ray_floor: wp.array1d[float]


@struct_schema
class SoftwareFields(GeometryFields, Protocol):
  """The software traversal's part (`SoftwareGeometry`): Warp BVHs, flex primitives, trace
  frames."""

  flex_vertadr: wp.array1d[int]
  flex_edge: wp.array1d[wp.vec2i]
  flex_radius: wp.array1d[float]
  flex_vert: wp.array2d[wp.vec3]
  # Instance frames of ray queries: instance_rot, times R^T for meshes whose BVH is built in frame
  # R.
  instance_rot_trace: wp.array2d[wp.mat33]
  bvh_id: wp.uint64
  # Mesh BLAS ids (`nmesh` of them).
  mesh_bvh_id: wp.array1d[wp.uint64]
  # Device addresses of the world and shared BVHs' packed children (`packed.py`; 0: no tree).
  bvh_nodes: wp.uint64
  shared_nodes: wp.uint64
  group_root: wp.array1d[int]
  bvh_ninstance: int
  bvh_nflexprim: int
  hfield_bvh_id: wp.array1d[wp.uint64]
  # Packed BVH2 children of the mesh, height-field and deformable BLASes (`packed.py`), by the
  # index of their id table.
  mesh_nodes: wp.array1d[wp.uint64]
  hfield_nodes: wp.array1d[wp.uint64]
  flex_nodes: wp.array1d[wp.uint64]
  flex_prim_flexid: wp.array1d[int]
  flex_prim_edgeid: wp.array1d[int]
  flex_bvh_id: wp.array1d[wp.uint64]
  flex_group_root: wp.array2d[int]
  # BVH of the shared instances (primitive k: instance shared_ids[k]).
  shared_bvh_id: wp.uint64
  shared_root: int
  shared_ids: wp.array1d[int]


@struct_schema
class OptixFields(GeometryFields, Protocol):
  """The OptiX backend's part (`OptixGeometry`, optix/)."""

  rt_handle: wp.uint64  # OptiX instance acceleration structure over all worlds
  rt_cell: wp.array1d[wp.vec3]  # per-world cells (`trace/optix/device/trace.py` CELL_*), then the shared IAS's box (SHARED_BOX)
  rt_grid: wp.vec3i  # world cells per axis
  rt_shared: wp.uint64  # OptiX IAS of the shared instances (untranslated, every world)


def _geometry_struct[F: GeometryFields](
  name: str, part: type[F], *, doc: str, live_uv: bool = False, cut: type | None = None
) -> type[F]:
  """The Warp struct `name` of the common fields plus `part`'s own (`GeometryFields` subclass);
  `cut`: the type of its alpha-test tables (default: `Cutout` / `LiveUvCutout`)."""
  fields = {**inspect.get_annotations(GeometryFields, eval_str=True), **inspect.get_annotations(part, eval_str=True)}
  fields["cut"] = cut if cut is not None else LiveUvCutout if live_uv else Cutout
  attrs = {"__annotations__": fields, "__module__": __name__, "__qualname__": name, "__doc__": doc}
  # A Warp `Struct` is used as a type (annotations, constructor) whose instances hold `part`'s
  # fields.
  return cast("type[F]", wp.struct(type(name, (), attrs)))


# The geometry struct of each backend's kernels. Separate types so that an OptiX build's launch
# parameters (copied by value into every program: trace/optix/device/params.py) hold none of the
# software traversal's ~760 bytes of BVH tables, and the software kernels none of the OptiX handles.
SoftwareGeometry = _geometry_struct("SoftwareGeometry", SoftwareFields, doc="`SoftwareFields`: the software kernels' geometry.")
OptixGeometry = _geometry_struct("OptixGeometry", OptixFields, doc="`OptixFields`: the OptiX programs' geometry.")
LiveUvSoftwareGeometry = _geometry_struct(
  "LiveUvSoftwareGeometry", SoftwareFields, doc="Software geometry with live material UVs.", live_uv=True
)
LiveUvOptixGeometry = _geometry_struct(
  "LiveUvOptixGeometry", OptixFields, doc="OptiX geometry with live material UVs.", live_uv=True
)


def software_geometry_type(*, live_uv: bool) -> type[SoftwareFields]:
  """The software geometry struct of a build (`live_uv`: Features.live_material_uv)."""
  return LiveUvSoftwareGeometry if live_uv else SoftwareGeometry


def optix_geometry_type(*, live_uv: bool, carrier: type | None = None) -> type[OptixFields]:
  """The OptiX geometry struct of a build (`live_uv` as `software_geometry_type`). `carrier`: a
  shaded build's material carrier (`shade.materials.material`), which then holds the alpha test's
  tables (`cut`) instead of a `Cutout`: the launch parameters hold the shared texture / UV tables
  once (`trace/optix/device/params.py`; the carrier has the live-UV tables too)."""
  if carrier is not None:
    return _carrier_geometry(carrier)
  return LiveUvOptixGeometry if live_uv else OptixGeometry


@functools.cache
def _carrier_geometry(carrier: type) -> type[OptixFields]:
  """The OptiX geometry struct whose alpha-test tables are the material carrier `carrier`."""
  name = f"Optix{carrier.__name__}Geometry"
  return _geometry_struct(name, OptixFields, doc=f"OptiX geometry with `{carrier.__name__}` as its tables.", cut=carrier)


# (world_tree, shared, compact_poses, visibility, planes) of `Features.trees`: which BVHs hold
# primitives, how poses are indexed, whether per-world visibility is read, the analytic planes'
# loop form (`Features.planes`).
type Trees = tuple[bool, bool, bool, bool, int]
# A ray query over a backend's geometry G: query(geo, worldid, origin, direction, max_dist) -> R
# (`query.make_trace`: G = SoftwareFields; `trace.optix.device.trace.make_trace`: G = OptixFields).
type RayQuery[G: GeometryFields, R] = Callable[[G, int, wp.vec3, wp.vec3, float], R]
# A `Tracer`'s closest-hit query (the closest hit before max_dist, instance -1: none): the software
# traversal's or the OptiX backend's (a kernel body is built for one backend's geometry).
type ClosestQuery = RayQuery[SoftwareFields, Hit] | RayQuery[OptixFields, Hit]
# A `Tracer`'s any-hit query: whether anything is hit before max_dist.
type AnyQuery = RayQuery[SoftwareFields, bool] | RayQuery[OptixFields, bool]
# The queries as the device functions that call them see them: generic over the geometry G
# (`device_generic`, `[G: GeometryFields]`: render bodies, cut-out / counter wrappers, reflection,
# temporal and probe rays). A kernel is built for one backend, whose `Tracer` queries take that
# backend's geometry (`ClosestQuery` / `AnyQuery`); `widen` gives their generic view.
type BodyClosest = RayQuery[GeometryFields, Hit]
type BodyAny = RayQuery[GeometryFields, bool]
# A shadow ray's RGB transmittance through translucent casters (0: blocked).
type ShadowQuery = RayQuery[SoftwareFields, wp.vec3] | RayQuery[OptixFields, wp.vec3]
type BodyShadow = RayQuery[GeometryFields, wp.vec3]
# leave(geo, worldid, instance, origin, direction, hit) -> Hit: `hit`, the closest hit of a ray from
# `origin` inside `instance`, or the face the ray leaves that instance through where it is nearer
# (`Tracer.leave`).
type LeaveQuery = Callable[[OptixFields, int, int, wp.vec3, wp.vec3, Hit], Hit]
type BodyLeave = Callable[[GeometryFields, int, int, wp.vec3, wp.vec3, Hit], Hit]


@wp.func(inline=True)
def world_entry(
  # In:
  ninstance: int,
  worldid: int,
  gi: int,
) -> int:
  """Instance gi's entry of world `worldid` in the per-world optics tables (Features.world_optics,
  `core/pipeline/worlds/optics.py`: world-major, `ninstance` entries a world, the length of
  `GeometryFields.instance_type`)."""
  return gi + worldid * ninstance


@wp.func(inline=True)
@device_generic
def world_twosided[G: GeometryFields](
  # In:
  geo: G,
  worldid: int,
  gi: int,
) -> int:
  """Instance gi's two-sided / closed-pane bits in world `worldid` (Features.world_optics)."""
  return geo.instance_twosided[world_entry(geo.instance_type.shape[0], worldid, gi)]


@wp.func(inline=True)
@device_generic
def world_transmit[G: GeometryFields](
  # In:
  geo: G,
  worldid: int,
  gi: int,
) -> wp.vec3:
  """Instance gi's shadow transmittance in world `worldid` (Features.world_optics)."""
  return geo.instance_shadow_transmit[world_entry(geo.instance_type.shape[0], worldid, gi)]


@wp.func
def world_hit(
  # In:
  hit: Hit,
  ninstance: int,
  worldid: int,
) -> Hit:
  """`hit` whose instance id is its entry of world `worldid` in the per-world optics tables
  (`world_entry`; no instance, < 0, stays): what a layer's coverage and blend read the instance id
  for (`layer.make_coverage`, `pixel._make_blend`) with Features.world_optics."""
  entry = hit
  if hit.instance_id >= 0:
    entry.instance_id = world_entry(ninstance, worldid, hit.instance_id)
  return entry


# The one factory with a declared return: Pyrefly 1.4 infers the scalars of a tuple of Warp vectors
# returned through `@wp.func` as Unknown.
@specialize
def instance_pose(*, compact: bool, bound: int = 0) -> Callable[[GeometryFields, int, int], tuple[wp.vec3, wp.mat33]]:
  """pose(geo, worldid, instance) -> (position, rotation) of an instance (shading frame).

  Without compact poses instance i is column i of the per-world poses, shared instances included
  (their poses are the same in every world); with them, a negative pose slot is a shared slot.
  `bound` (Features.bound): compact poses compiled in are read where the build word says so
  (`toggles.GEO_COMPACT`), the plain columns elsewhere.
  """
  gate = toggles.bound_on(bound)

  @wp.func
  @device_generic
  def pose[G: GeometryFields](
    # In:
    geo: G,
    worldid: int,
    gi: int,
  ) -> tuple[wp.vec3, wp.mat33]:
    """Pose of instance `gi` in world `worldid`."""
    if wp.static(compact):
      if gate(geo.bits, wp.static(toggles.GEO_COMPACT)):
        s = geo.pose_slot[gi]
        if s < 0:
          return geo.shared_pos[-1 - s], geo.shared_rot[-1 - s]
        return geo.instance_pos[worldid, s], geo.instance_rot[worldid, s]
    return geo.instance_pos[worldid, gi], geo.instance_rot[worldid, gi]

  return pose


def widen[R](query: RayQuery[SoftwareFields, R] | RayQuery[OptixFields, R]) -> RayQuery[GeometryFields, R]:
  """A backend's ray query as the generic device functions call it (`BodyClosest` / `BodyAny`).

  The one cast between the backends' geometry types and the generic view: a kernel is built for
  one backend (its geometry struct and queries travel together), which the type checker cannot see
  through Warp's device-function generics.
  """
  return cast("RayQuery[GeometryFields, R]", query)


# cast_ray(geo, worldid, ray) -> Hit: a scene-BVH traversal (`cast.py`, `ordered.py`).
type CastRay = Callable[[SoftwareFields, int, Ray], Hit]


class Tracer(NamedTuple):
  """The ray queries of one kernel body (`TraceBackend.tracer`), all with the `make_trace`
  signatures.

  Attributes:
    closest: closest hit, back faces culled as the build asks.
    any_hit: whether anything is hit before max_dist.
    shadow: a shadow ray's RGB transmittance through translucent casters, one terminating walk
      (`Features.translucent_shadows`; None: none, the shadow rays are `any_hit` queries).
    leave: the exit face of a primitive a camera ray travels inside, where `closest` does not
      report it as the software traversal does (`optix.trace.make_leave`; None: `closest` does).
  """

  closest: ClosestQuery
  any_hit: AnyQuery
  shadow: ShadowQuery | None = None
  leave: LeaveQuery | None = None


class AlphaTest(NamedTuple):
  """How the software queries alpha-test their candidates (`Features.alpha_cutout`): the texture
  coordinates the opacity maps read (`cutout.make_cut_away`)."""

  live_uv: bool  # Features.live_material_uv
  world_uv: bool  # Features.world_uv


class TraceFeatures(Protocol):
  """What a `Tracer` is specialised on: `Features` satisfies it, `ProbeFeatures` names a few
  fields."""

  @property
  def shape_types(self) -> tuple[int, ...]:
    """Instance types the queries intersect."""
    ...

  @property
  def trees(self) -> Trees:
    """`Features.trees`."""
    ...

  @property
  def ordered_traversal(self) -> bool:
    """Nearest-first traversal (`ordered.py`) instead of MuJoCo Warp's."""
    ...

  @property
  def two_sided(self) -> bool:
    """Culling skips the instances flagged in `GeometryFields.instance_twosided`."""
    ...

  @property
  def translucent_shadows(self) -> bool:
    """Shadow rays pass translucent casters (`Tracer.shadow`)."""
    ...

  @property
  def world_optics(self) -> bool:
    """The two-sided bits and shadow transmittance are read per world (`world_entry`)."""
    ...

  @property
  def glass_shapes(self) -> bool:
    """Camera rays leave solid glass spheres and boxes through their analytic exit
    (`Tracer.leave`, OptiX)."""
    ...

  @property
  def alpha_cutout(self) -> bool:
    """Cut-out materials: the software queries alpha-test their candidates."""
    ...

  @property
  def live_material_uv(self) -> bool:
    """The alpha test reads live material UV mappings (`cutout.make_cut_away`)."""
    ...

  @property
  def world_uv(self) -> bool:
    """The alpha test reads world-projected opacity maps (`cutout.make_cut_away`)."""
    ...

  @property
  def counters(self) -> bool:
    """Diagnostic trace counts in a counter-enabled render build."""
    ...

  @property
  def bound(self) -> int:
    """The bound bits read from the build word (`Features.bound`; an OptiX build's)."""
    ...

  @property
  def switches(self) -> int:
    """The run-time switches (`Features.switches`; a tier build's): back-face culling compiled in
    is read from `Toggles.flags` (`toggles.BACKFACE_CULLING`)."""
    ...


@final
@dataclasses.dataclass(slots=True, frozen=True)
class ProbeFeatures:
  """`TraceFeatures` of the unordered, plain queries of the probe and light-grid rays.

  Their kernels are cached on a few fields rather than on a whole `Features` (`light_grid`,
  `gi`) and apply the alpha test themselves.
  """

  shape_types: tuple[int, ...]
  trees: Trees
  ordered_traversal: bool = False
  two_sided: bool = False
  translucent_shadows: bool = False
  world_optics: bool = False
  glass_shapes: bool = False
  alpha_cutout: bool = False
  live_material_uv: bool = False
  world_uv: bool = False
  counters: bool = False
  bound: int = 0
  switches: int = 0


@wp.kernel
def mesh_trace_frames(
  # Model:
  instance_mesh: wp.array2d[int],
  # Data in:
  instance_rot_in: wp.array2d[wp.mat33],
  # In:
  mesh_columns: wp.array1d[int],
  per_world: int,
  mesh_bvh_frame: wp.array1d[wp.mat33],
  # Out:
  instance_rot_trace_out: wp.array2d[wp.mat33],
):
  """Mesh-instance frames of ray queries: `instance_rot @ R^T` for meshes whose BVH is built in
  frame R, for the mesh (and SDF) columns `mesh_columns` only (the other instances' entries are
  never read: `instances.make_tree_pose`): its first `per_world` columns in every world, the
  shared ones after them in world 0 only (the shared tree reads world 0's row)."""
  worldid, k = wp.tid()
  if k >= per_world and worldid > 0:
    return
  gi = mesh_columns[k]
  mat = instance_rot_in[worldid, gi]
  mesh_id = instance_mesh[table_row(worldid, instance_mesh.shape[0]), gi]
  if mesh_id >= 0:
    mat = mat @ wp.transpose(mesh_bvh_frame[in_table(mesh_id, mesh_bvh_frame.shape[0])])
  instance_rot_trace_out[worldid, gi] = mat
