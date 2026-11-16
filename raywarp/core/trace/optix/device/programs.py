"""The OptiX programs of one specialised build, in one Warp module compiled to PTX.

- `__raygen__render`: one launch index per (world, ray) (from the launch chunk's first unit
  `offset`, `pipeline.Pipeline.chunks`); calls `pixel.make_render_pixel`'s func, whose ray
  queries are the OptiX `Tracer` (`scene_tracer`, `trace.py`), of the pinhole cameras; each lens
  model's (`__raygen__render_distortion`, `_bundle`: `primary.LENS_MODELS`, `raygen_name`)
  in a module of its own (`build_lens`), which a renderer whose cameras have that model launches
  instead, beside this module's other programs; `__miss__layer` (the shading of the translucent
  layers behind the first surface, `layer_program`), `__miss__light_visibility` (one light's
  shadow rays, `light_visibility_program`) and `__raygen__visibility` (the light grid's visibility
  test) when the build has them; the GI bake's programs (`GI_PROGRAMS`) in a module of their own
  (`build_gi`) with GI; the soft area shadows' pass (`__raygen__area_shadows`) in a module of its
  own (`build_area`), which a renderer with `shadows.area_rays` on launches after the render;
- `__miss__render`, `__closesthit__triangle` (+ `__anyhit__triangle`: the alpha test, the
  per-world visibility of shared instances), the curve closest-hit programs (+ `__anyhit__curve`,
  `__anyhit__curve_shared`: the alpha test, that visibility) and one `__intersection__` per
  custom-primitive type, per group (per-world / shared), which run MuJoCo Warp's `ray_*` functions
  (and the alpha test; the shared ones test that visibility too); curves use OptiX's built-in
  intersector.

All programs take the build's launch-parameter struct (`params.params_type`).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast, final

import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.build.variants import tiers
from raywarp.core.kernels import first_hit, geometry, lighting, native, pixel
from raywarp.core.kernels import probe as probe_lib
from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi.features import PLANES_NONE
from raywarp.core.kernels.abi.queries import make_queries
from raywarp.core.kernels.abi.rays import (
  AlphaTest,
  GeometryFields,
  Hit,
  OptixFields,
  TraceFeatures,
  Tracer,
  no_hit,
  widen,
  world_transmit,
)
from raywarp.core.kernels.abi.structs import EnvironmentArrays, Params
from raywarp.core.kernels.abi.tables import batched, in_table, instance_visible
from raywarp.core.kernels.generic import device_generic
from raywarp.core.kernels.layer import Coverage
from raywarp.core.post.temporal import pixel as temporal_pixel
from raywarp.core.shade.gi import kernels as gi_kernels
from raywarp.core.shade.lights import grid as light_grid
from raywarp.core.shade.lights import punctual
from raywarp.core.shade.lights.punctual import Light, Lights
from raywarp.core.shade.materials.material import MaterialTables
from raywarp.core.shade.materials.octahedral import pack_unit, unpack_unit
from raywarp.core.shade.shadows import area as area_shadows
from raywarp.core.trace.accel import cutout
from raywarp.core.trace.optix.device import hits, intrinsics
from raywarp.core.trace.optix.device import params as params_lib
from raywarp.core.trace.optix.device import trace as rt_trace
from raywarp.core.trace.optix.device.params import LaunchParams
from raywarp.core.trace.rays import camera, primary

if TYPE_CHECKING:
  from collections.abc import Callable

  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.shade.lights.loop import LightVisibilityAt

# Name of the programs' Warp modules (one per build, not registered with Warp; kernel cache
# entries are told apart by their content hash).
MODULE_NAME: Final = "raywarp_optix"
# Semantic name of the pinhole cameras' render program (`raygen_name`: the other camera models').
RENDER: Final = "__raygen__render"
# Semantic name of the light grid's ray-generation program (its own pipeline and launches).
VISIBILITY: Final = "__raygen__visibility"
# Semantic name of the layer shading program (`layer_program`: a miss program,
# `intrinsics.layer_call`).
LAYER: Final = "__miss__layer"
# Semantic name of the light visibility program (`light_visibility_program`: a miss program,
# `intrinsics.visibility_call`).
LIGHT_VISIBILITY: Final = "__miss__light_visibility"
# The GI bake's ray-generation programs (`gi_programs`): its probe rays, and its placement's
# static rays, surveys, walls and contact occlusion.
PROBE: Final = "__raygen__probe"
PROBE_STATIC: Final = "__raygen__probe_static"
PROBE_SURVEY: Final = "__raygen__probe_survey"
PROBE_WALLS: Final = "__raygen__probe_walls"
PROBE_AO: Final = "__raygen__probe_ao"
GI_PROGRAMS: Final = (PROBE, PROBE_STATIC, PROBE_SURVEY, PROBE_WALLS, PROBE_AO)
# The soft area shadows' ray-generation program (`build_area`).
AREA: Final = "__raygen__area_shadows"


class SceneFlags(Protocol):
  """What the programs read of a build's structures (`accel.OptixScene`; a stand-in in codegen
  tests), read only."""

  @property
  def has_mesh(self) -> bool:
    """Triangle meshes."""
    ...

  @property
  def has_box(self) -> bool:
    """Boxes (custom primitives)."""
    ...

  @property
  def has_instances(self) -> bool:
    """Per-world instances (the world structure)."""
    ...

  @property
  def has_shared(self) -> bool:
    """Instances shared by every world (the shared structure)."""
    ...

  @property
  def translucent_triangles(self) -> bool:
    """Translucent shadow casters among the triangles."""
    ...

  @property
  def translucent_curves(self) -> tuple[bool, bool]:
    """Translucent casters among the curves, per structure (world, shared)."""
    ...

  @property
  def shared_visible(self) -> bool:
    """Per-world visibility of the shared instances."""
    ...

  @property
  def primitive_types(self) -> tuple[int, ...]:
    """The custom primitive types of the world structure."""
    ...

  @property
  def shared_primitive_types(self) -> tuple[int, ...]:
    """The custom primitive types of the shared structure."""
    ...

  @property
  def has_world_curves(self) -> bool:
    """Curves (spheres, capsules) in the world structure."""
    ...

  @property
  def has_shared_curves(self) -> bool:
    """Curves in the shared structure."""
    ...

  @property
  def has_omm(self) -> bool:
    """Opacity micromaps."""
    ...


@final
@dataclasses.dataclass(frozen=True, slots=True)
class FreeFlags:
  """`SceneFlags` a scene-free build's programs are compiled for (`scene_free_flags`)."""

  has_mesh: bool
  has_box: bool
  has_instances: bool
  has_shared: bool
  translucent_triangles: bool
  translucent_curves: tuple[bool, bool]
  shared_visible: bool
  primitive_types: tuple[int, ...]
  shared_primitive_types: tuple[int, ...]
  has_world_curves: bool
  has_shared_curves: bool
  has_omm: bool


def scene_free_flags(f: Features, scene: SceneFlags) -> FreeFlags:
  """The flags a scene-free build `f` (`Features.scene_free`) compiles its programs for.

  - The geometry kinds of its shape types (`tiers.geometry_kinds`: the triangle kinds only, or
    with the boxes, curves and custom primitives too) in each structure the build traces (the
    scene's, or every one its word binds: `toggles.GEO_WORLD`, `GEO_SHARED`).
  - The translucent casters' transmittance as any-hit programs (a build with
    `Features.translucent_shadows`).
  - The shared instances' per-world visibility: the scene's, or bound (`GEO_SHARED_VISIBLE`).
  - Opacity micromaps where the scene has them (`accel.OptixScene.has_omm`) in a build with the
    alpha test (`Features.alpha_cutout`: the surface class): a pipeline option, not in the PTX key.

  The pipeline links the program groups of `scene`'s own structures only (`pipeline.Pipeline`,
  its SBT records): programs of absent kinds are compiled, never run. The planes' ids are run-time
  data (`GeometryFields.plane_ids`), their loop form the build's (`Features.planes`)."""
  types = set(f.shape_types)
  curves = bool(types & set(rt_trace.CURVE_TYPES))
  custom = tuple(t for t in rt_trace.CUSTOM_TYPES if t in types)
  # Bound structures (`Features.bound`): both compiled in, each traced where the build word says.
  world = scene.has_instances or bool(f.bound & toggles.GEO_WORLD)
  shared = scene.has_shared or bool(f.bound & toggles.GEO_SHARED)
  return FreeFlags(
    has_mesh=True,
    has_box=int(ShapeType.BOX) in types,
    has_instances=world,
    has_shared=shared,
    translucent_triangles=f.translucent_shadows,
    translucent_curves=(curves and world, curves and shared),
    shared_visible=scene.shared_visible or bool(f.bound & toggles.GEO_SHARED_VISIBLE),
    primitive_types=custom if world else (),
    shared_primitive_types=custom if shared else (),
    has_world_curves=curves and world,
    has_shared_curves=curves and shared,
    has_omm=f.alpha_cutout and scene.has_omm,
  )


def build_word(build: Features, effective: Features, scene: SceneFlags) -> int:
  """The value of build `build`'s word in its geometry copy (`GeometryFields.bits`, every bound bit)
  over `scene`'s structures for a configuration of Features `effective` (specialised on the scene):
  `tiers.word` and the geometry bits the scene uses (`toggles.GEO_*`: its planes, structures, pose
  indexing, per-world visibility), among the build's bound ones."""
  geometry = (
    (toggles.GEO_PLANES if effective.planes != PLANES_NONE else 0)
    | (toggles.GEO_WORLD if scene.has_instances else 0)
    | (toggles.GEO_SHARED if scene.has_shared else 0)
    | (toggles.GEO_COMPACT if effective.compact_poses else 0)
    | (toggles.GEO_VISIBLE if effective.visibility else 0)
    | (toggles.GEO_SHARED_VISIBLE if effective.visibility and scene.shared_visible else 0)
  )
  return tiers.word(build, effective) | (geometry & build.bound)


def traced_shapes(f: Features, word: int) -> int:
  """The analytic shape types (`toggles.SHAPE_BITS`) a module of build `f` with build word `word`
  traces: those its word has where `f` binds them, else every one it compiles."""
  return (word | ~f.bound) & toggles.SHAPE_ALL


def program_flags(f: Features, scene: SceneFlags) -> SceneFlags:
  """The flags build `f`'s programs are compiled for: `scene_free_flags` for a scene-free build,
  else the scene's own."""
  return scene_free_flags(f, scene) if f.scene_free else scene


def layout(f: Features, scene: SceneFlags) -> params_lib.ParamsLayout:
  """The launch-parameter layout of build `f` over `scene` (its programs and their tables)."""
  return params_lib.ParamsLayout(
    live_material_uv=f.live_material_uv,
    diffuse_roughness=f.diffuse_roughness,
    specular_weight=f.specular_weight,
    layered=f.layered,
    shaded=f.any_rgb,
    history=f.history,
    grid=visibility_program(f),
    world_curves=scene.has_world_curves,
    world_primitives=bool(scene.primitive_types),
    shared_curves=scene.has_shared_curves,
    shared_primitives=bool(scene.shared_primitive_types),
    mesh=scene.has_mesh,
  )


def intersection_name(gtype: int, *, shared: bool = False, shadow: bool = False) -> str:
  """The semantic name of the intersection program of custom type `gtype` (per-world or shared;
  `shadow`: the shadow traces' ray type)."""
  return f"__intersection__{ShapeType(gtype).name.lower()}{'_shared' if shared else ''}{'_shadow' if shadow else ''}"


def shared_visibility(f: Features, scene: SceneFlags) -> bool:
  """Whether shared instances have a per-world visibility the programs test (any-hit programs of
  the shared triangles and curves, the shared intersection programs)."""
  return f.visibility and scene.shared_visible


def anyhit_programs(f: Features, scene: SceneFlags) -> frozenset[str]:
  """The any-hit programs of the hit groups, by name. Ray type 0 (camera, occlusion traces):
  `__anyhit__triangle` (the alpha test of cut-out instances, the per-world visibility of shared
  ones), `__anyhit__curve` / `__anyhit__curve_shared` (the alpha test; that visibility). Ray type 1
  (shadow traces through translucent casters, `intrinsics.trace_shadow`): the same tests plus the
  translucent casters' transmittance, `__anyhit__triangle_shadow`, `__anyhit__curve_shadow`,
  `__anyhit__curve_shared_shadow`."""
  cut, seen = f.alpha_cutout, shared_visibility(f, scene)
  translucent = f.translucent_shadows
  names = set()
  if cut or seen:
    names.add("__anyhit__triangle")
  if translucent and (cut or seen or scene.translucent_triangles):
    names.add("__anyhit__triangle_shadow")
  if scene.has_world_curves and cut:
    names.add("__anyhit__curve")
  if translucent and scene.has_world_curves and (cut or scene.translucent_curves[0]):
    names.add("__anyhit__curve_shadow")
  if scene.has_shared_curves and (cut or seen):
    names.add("__anyhit__curve_shared")
  if translucent and scene.has_shared_curves and (cut or seen or scene.translucent_curves[1]):
    names.add("__anyhit__curve_shared_shadow")
  return frozenset(names)


def raygen_name(model: primary.CameraModel) -> str:
  """The semantic name of camera model `model`'s render program (`RENDER`: the pinhole's)."""
  return RENDER if model == primary.PINHOLE else f"{RENDER}_{model.name}"


def visibility_program(f: Features) -> bool:
  """Whether a build has the light grid's visibility program (`__raygen__visibility`): with the
  grid on (its unculled rays: `trace.RAY_MASK_CULL`)."""
  return f.culling


def layer_program(f: Features) -> bool:
  """Whether a build has the layer shading program (`__miss__layer`, `intrinsics.layer_call`).

  A shaded build with translucent layers: their layer loop shades each layer behind the first
  surface by calling it (`pixel.make_trace_pixel`'s `layer`), so the ray-generation program holds
  one inlined surface shading, the first surface's, and OptiX compiles the layers' as a program of
  its own.
  (Inline, the second shading site multiplied the state OptiX keeps across the raygen's trace
  calls: a translucent default module took 143 s to create, 24.5 s without the loop.)"""
  return f.any_rgb and f.transparency


def light_visibility_program(f: Features) -> bool:
  """Whether a build has the light visibility program (`__miss__light_visibility`,
  `intrinsics.visibility_call`).

  A shaded build with shadows that reads its light caps and shadow budget at run time (a counted
  tier build: `Features.run_counts`, the quality recipe's): its light loops (the first
  surface's, the reflection hits', the layer program's, the GI probe hits') take one light's
  visibility by calling it (`Queries.light_visibility`), so a ray-generation program holds one call
  site where it inlined every shadow-ray form of every light kind (the sun's disc, a soft light's
  samples, an area light's targets, each through both structures and both shadow forms) at each of
  the loop's sites (every caster, the budget's slots, the remainder), and OptiX compiles them
  once, as a program of its own.
  (Inline, a quality module's ray-generation program held 135 trace sites, 128 of them shadow
  rays, and took 2-16 min to create, its GI probe program up to 10 min; called, 21-46 s, and the
  quality frames ran 35-67 % faster, images within 2/255. The fast and default tiers' small and
  capped loops ran 2-12 % slower calling it: inline there.)"""
  return f.any_rgb and f.shadows and f.run_counts


def gi_programs(f: Features) -> bool:
  """Whether a build has the GI bake's programs (`GI_PROGRAMS`, `build_gi`: a module of their own):
  with GI on (the bake's rays traced by the hardware, `launch.OptixRenderer.gi_tracer`)."""
  return f.gi and f.any_rgb


def scene_tracer(scene: SceneFlags, f: TraceFeatures, *, cull: bool, mask: int = 255) -> Tracer:
  """The OptiX queries of `f` over `scene`'s structures (`OptixBackend.tracer`).

  `cull`: closest / any queries cull back faces (triangles by ray flag, the rest in the
  intersection programs; `f.two_sided` instances never). `mask`: the instances' visibility-mask
  bits the queries trace (`trace.BAKE_RAY_MASK`: a shared bake's, without the deformables). With
  `f.alpha_cutout` the hit programs alpha-test the candidates of the cut-out instances (the
  triangles' and curves' any-hit programs, the custom primitives' intersection programs) and the
  plane queries their planes' hits.
  """
  compact, visibility = f.trees[2], f.trees[3]
  anyhit = f.alpha_cutout
  shared_anyhit = anyhit or (visibility and scene.shared_visible)
  bound = f.bound
  alpha = AlphaTest(f.live_material_uv, f.world_uv) if f.alpha_cutout else None

  def spec_of(*, cull_faces: bool) -> rt_trace.TraceSpec:
    """The queries' spec, culling as `cull_faces`."""
    return rt_trace.TraceSpec(
      planes=f.trees[4],
      has_mesh=scene.has_mesh,
      has_box=scene.has_box,
      has_instances=scene.has_instances,
      cull=cull_faces,
      has_shared=scene.has_shared,
      compact=compact,
      visibility=visibility,
      anyhit=anyhit,
      alpha=alpha,
      shared_anyhit=shared_anyhit,
      two_sided=f.two_sided,
      world_optics=f.world_optics,
      counting=f.counters,
      bound=bound,
      mask=mask,
      switches=f.switches,
    )

  spec = spec_of(cull_faces=cull)
  closest, any_hit = rt_trace.make_trace(spec)
  # Shadow rays through translucent casters: one trace per structure, the casters' any-hit and
  # intersection programs multiplying the transmittance in (`trace.make_shadow`); culling as the
  # occlusion queries' (`cull`): one shadow-ray behaviour on both paths.
  shadow = rt_trace.make_shadow(spec) if f.translucent_shadows else None
  # A camera ray inside a solid glass sphere, capsule or box leaves it through the face the
  # software traversal reports (`trace.make_leave`: OptiX's curves report none from inside,
  # coincident faces tie unordered); built only for a build with solid glass primitives
  # (`Features.glass_shapes`; ellipsoids and cylinders are custom primitives, which report both
  # crossings).
  exits = (int(ShapeType.SPHERE), int(ShapeType.BOX), int(ShapeType.CAPSULE))
  leaving = tuple(t for t in f.shape_types if t in exits)
  leave = rt_trace.make_leave(spec, leaving) if f.glass_shapes and leaving else None
  return Tracer(closest, any_hit, shadow=shadow, leave=leave)


def _program(module: wp.Module, name: str, params: type[LaunchParams]):
  """Decorator: `fn(p)` as the OptiX program `name` of `module`, taking the launch parameters
  `p`."""

  def decorator(fn: Callable[[LaunchParams], None]):
    """The program kernel (Warp's external_constant_params entry point)."""
    fn.__annotations__ = {fn.__code__.co_varnames[0]: params}
    return wp.kernel(fn, module=module, name=name, enable_backward=False, entry_point_abi="external_constant_params")

  return decorator


# The `SceneFlags` fields the programs read (`ptx_key`): the protocol's read-only properties.
_SCENE_FLAGS: Final = tuple(name for name, value in vars(SceneFlags).items() if isinstance(value, property))
# The flags a build's PTX depends on: all but the micromaps (a module and pipeline compile option).
_PTX_FLAGS: Final = tuple(k for k in _SCENE_FLAGS if k != "has_omm")


@final
class PtxParts(NamedTuple):
  """What a build's PTX depends on but raywarp's kernel code (`ptx_key`).

  Attributes:
    program: "render" (`build`), "gi" (`build_gi`), "area" (`build_area`) or a lens model's
      render program ("render_<model>", `build_lens`).
    features: the build's Features (repr).
    flags: its scene flags (`_PTX_FLAGS`; the micromaps are a module option).
    supersampled: whether it renders the supersampled image (its factor is run-time data,
      `Lens.scale`).
    arch: the PTX target.
    warp: the Warp release.
    codegen: Warp's code-generation settings.
    headers: the OptiX headers' version, not their path: a key (and the PTX it names) holds on any
      install location (prebuilt kernels, `core/build/kernel_cache/prebuilt.py`).
  """

  program: str
  features: str
  flags: tuple[object, ...]
  supersampled: bool
  arch: int
  warp: str
  codegen: tuple[object, ...]
  headers: tuple[int, int, int] | None


# The Features fields no OptiX program's generated code nor its module options read (the software
# megakernel's launch shape, register cap and traversal order; the module's register cap is its
# record's, `registers.optix_registers`): reset in `ptx_parts`, so a tuning study keeps the bundle.
# Checked: the programs' CUDA source and module options are identical for two values of each
# (render and GI, fast and default builds).
KEY_UNREAD: Final = ("block_dim", "max_registers", "ordered_traversal")


def key_features(f: Features) -> Features:
  """`f` with the `KEY_UNREAD` fields reset (their type's zero, as `gi_features`): what a build's
  PTX key names."""
  values = vars(f)
  return dataclasses.replace(f, **{name: type(values[name])() for name in KEY_UNREAD})


def ptx_parts(
  f: Features,
  scene: SceneFlags,
  *,
  supersampled: bool,
  arch: int,
  gi: bool = False,
  lens: primary.CameraModel | None = None,
  area: bool = False,
) -> PtxParts:
  """The parts of `ptx_key`. Nothing installed beside raywarp but Warp is one (MuJoCo Warp: its
  kernels are vendored): a bundle filled with the `mujoco` extra serves a Newton or Isaac Lab
  install."""
  inc = intrinsics.include_dir()
  program = "gi" if gi else "area" if area else "render" if lens is None else f"render_{lens.name}"
  return PtxParts(
    program=program,
    features=repr(key_features(f)),
    flags=tuple(getattr(scene, k) for k in _PTX_FLAGS),
    supersampled=supersampled,
    arch=arch,
    warp=wp.config.version,
    codegen=(wp.config.mode, wp.config.optimization_level, wp.config.verify_fp, wp.config.verify_cuda, wp.config.lineinfo),
    headers=intrinsics.header_version(inc) if inc is not None else None,
  )


def key_of(parts: tuple[object, ...]) -> str:
  """The PTX key of key parts `parts` (`ptx_parts`, as a plain tuple)."""
  return hashlib.blake2b(repr(tuple(parts)).encode(), digest_size=16).hexdigest()


def ptx_key(
  f: Features,
  scene: SceneFlags,
  *,
  supersampled: bool,
  arch: int,
  gi: bool = False,
  lens: primary.CameraModel | None = None,
  area: bool = False,
) -> str:
  """The cache key of build `f`'s PTX (`gi`: `build_gi`'s; `lens`: that lens model's
  `build_lens`; `area`: `build_area`'s) for `arch`: everything its generated code depends on but
  raywarp's kernel code (`ptx_parts`), so that a warm start reads the PTX without defining the
  programs (`pipeline.PtxCompile`). The kernel code is the entry's (`prebuilt.Entry.fresh`: the
  digest of the code its generation used)."""
  parts = ptx_parts(gi_features(f) if gi else f, scene, supersampled=supersampled, arch=arch, gi=gi, lens=lens, area=area)
  return key_of(parts)


def build(f: Features, scene: SceneFlags, *, supersampled: bool = False) -> wp.Module:
  """Defines the programs of one specialised build in a new Warp module, returned (`supersampled`:
  rendering the OptiX supersampled image, `supersample.py`)."""
  module = wp.Module(MODULE_NAME, None)
  P = params_lib.params_type(layout(f, scene))
  tracer = scene_tracer(scene, f, cull=f.backface_culling)
  _raygen_programs(module, P, f, scene, tracer, supersampled=supersampled)
  _hit_programs(module, P, f, scene)
  _module_options(module, f)
  return module


def build_lens(f: Features, scene: SceneFlags, model: primary.CameraModel, *, supersampled: bool = False) -> wp.Module:
  """Defines lens model `model`'s render program of build `f` (`raygen_name`) alone in a new Warp
  module, returned: a renderer whose cameras have that model links it with the build's other
  programs (`pipeline.Pipeline`'s `lens`), on the build's launch parameters."""
  module = wp.Module(MODULE_NAME, None)
  P = params_lib.params_type(layout(f, scene))
  tracer = scene_tracer(scene, f, cull=f.backface_culling)
  layer = _layer_call() if layer_program(f) else None
  # Its light loops' visibility inline (no light visibility call): the module declares the layer
  # call's payload type only (`pipeline.call_programs`), whatever the build's calls.
  _render_program(
    module, P, f, raygen_name(model), pixel.make_render_pixel(f, tracer, model, supersampled=supersampled, layer=layer)
  )
  _module_options(module, f)
  return module


# The Features fields no GI program reads (`probe.CAMERA_ONLY`: the programs shade with
# `probe.probe_features`) nor its launch-parameter layout or module options do: reset in
# `gi_features`, so one GI build serves the render builds that differ only in them.
GI_UNREAD: Final = (
  "budget",
  "glossy",
  "hit_light_cap",
  "light_cap",
  "reflection_shadows",
  "motion",
)


def gi_features(f: Features) -> Features:
  """The Features `build_gi`'s programs (and their key, `ptx_key`) are compiled for: the build's
  with the `GI_UNREAD` fields reset and the camera-only fields' bits cleared
  (`probe.without_camera_bits`)."""
  values = vars(f)
  return probe_lib.without_camera_bits(dataclasses.replace(f, **{name: type(values[name])() for name in GI_UNREAD}))


def gi_layout(f: Features, scene: SceneFlags) -> params_lib.ParamsLayout:
  """The launch-parameter layout of `build_gi`'s programs: the render's (of `gi_features`), with
  the GI launches."""
  return dataclasses.replace(layout(gi_features(f), scene), gi=True)


def build_gi(f: Features, scene: SceneFlags) -> wp.Module:
  """Defines the GI bake's programs of one specialised build in a new Warp module, returned: its
  own module and pipeline (`launch.OptixRenderer.gi_launch`), so that the render's programs and
  launch parameters are those of a build without GI. Its hit programs are the probe rays'
  (`probe.probe_features`: no back-face culling, also for the custom primitives' intersection
  programs, whose culling is compiled in: rays from inside a closed primitive meet its back
  faces)."""
  f = gi_features(f)
  module = wp.Module(MODULE_NAME, None)
  P = params_lib.params_type(gi_layout(f, scene))
  _gi_programs(module, P, f, scene)
  _hit_programs(module, P, probe_lib.probe_features(f), scene)
  _module_options(module, f)
  return module


def area_layout(f: Features, scene: SceneFlags) -> params_lib.ParamsLayout:
  """The launch-parameter layout of `build_area`'s program: the render's, with the pass' launch."""
  return dataclasses.replace(layout(f, scene), area=True)


def build_area(f: Features, scene: SceneFlags) -> wp.Module:
  """Defines the soft area shadows' program of build `f` (`AREA`) and the build's hit programs in
  a new Warp module, returned: its own module and pipeline (`launch.OptixRenderer.area_shadows`),
  so that the render's module is the same with the pass on or off. Its queries, surface shading
  and light evaluators are the render's (over the camera rays' queries)."""
  module = wp.Module(MODULE_NAME, None)
  P = params_lib.params_type(area_layout(f, scene))
  area_pixel = area_shadows.make_pixel(f, make_queries(f, scene_tracer(scene, f, cull=f.backface_culling)))

  @_program(module, AREA, P)
  def raygen_area(p: LaunchParams):
    """Pixel unit p.offset + i of the launch: (world, pixel) (`area_shadows.make_pixel`)."""
    a = p.area[0]
    i = p.offset + native.optix_launch_index()
    worldid = i // a.npix
    # A shaded build's `geo.cut` is its material carrier (`params.ParamsLayout.shaded`).
    area_pixel(p.geo, p.geo.cut, p.lit, p.cams, p.env, p.out, a, worldid, i - worldid * a.npix)

  _hit_programs(module, P, f, scene)
  _module_options(module, f)
  return module


def _module_options(module: wp.Module, f: Features) -> None:
  """Sets the programs' Warp module options of build `f`."""
  options = kernel_structs.module_options(f)
  options["extra_build_options"] = intrinsics.build_options()
  options["enable_backward"] = False
  wp.set_module_options(options, module)


def _raygen_programs(
  module: wp.Module, P: type[LaunchParams], f: Features, scene: SceneFlags, tracer: Tracer, *, supersampled: bool
) -> None:
  """The ray-generation programs: the pinhole cameras' render (and the layer shading program it
  calls; the lens models': `build_lens`), the grid's test."""
  layer = _layer_call() if layer_program(f) else None
  visibility = None
  if light_visibility_program(f):
    slot = intrinsics.visibility_slot(layer=layer is not None)
    visibility = _visibility_call(slot, tinted=f.tinted_shadows)
    _light_visibility_program(module, P, f, lighting.light_visibility(f, make_queries(f, tracer)), slot)
  render_pixel = pixel.make_render_pixel(
    f, tracer, primary.PINHOLE, supersampled=supersampled, layer=layer, light_visibility=visibility
  )
  if layer is not None:
    _layer_program(module, P, f, tracer, visibility)
  _render_program(module, P, f, RENDER, render_pixel)
  if visibility_program(f):
    _visibility_program(module, P, f, scene)


def _render_program(module: wp.Module, P: type[LaunchParams], f: Features, name: str, render_pixel: pixel.PixelBody) -> None:
  """The render program `name` of body `render_pixel` (`pixel.make_render_pixel`: one camera
  model's)."""
  shaded = f.any_rgb
  history = f.history
  # Warp erases the unused branches below. Each emitted call has the signature selected by
  # make_render_pixel under exactly these shaded/history flags.
  geometry_body = cast(geometry.Body, render_pixel)
  geometry_history = cast(geometry.TemporalBody, render_pixel)
  shaded_body = cast(pixel.RenderPixel, render_pixel)
  shaded_history = cast(temporal_pixel.RenderPixel, render_pixel)

  @_program(module, name, P)
  def raygen(p: LaunchParams):
    """Renders camera ray `rayid` of world `worldid` (`pixel.make_render_pixel`)."""
    # Launch index i of the chunk starting at world p.offset: (world, ray) units of rays, mapped to
    # (world, ray) in the frame's order (`camera.launch_ids`).
    i = native.optix_launch_index()
    unit = i // p.cams.nray
    rayid = i - unit * p.cams.nray
    worldid = p.offset + unit
    if wp.static(history):
      tmp = p.tmp[0]  # the temporal state (`params.write_struct`, written before the launch)
    worldid, rayid = camera.launch_ids(p.cams, p.nworld, worldid, rayid)
    if worldid < 0:
      return  # a hole of the active list
    if wp.static(not shaded):
      # The geometry-only body (`kernels/geometry.py`): no shading structs.
      if wp.static(history):
        geometry_history(p.geo, p.cams, p.out, tmp, worldid, rayid)  # pyrefly: ignore[unbound-name]  # set under the same wp.static(history)
      else:
        geometry_body(p.geo, p.cams, p.out, worldid, rayid)
    elif wp.static(history):
      # A shaded build's `geo.cut` is its material carrier (`params.ParamsLayout.shaded`).
      shaded_history(
        p.geo,
        p.geo.cut,  # pyrefly: ignore[bad-argument-type]
        p.lit,
        p.cams,
        p.env,
        p.seed[0],
        p.out,
        tmp,  # pyrefly: ignore[unbound-name]  # set under the same wp.static(history)
        worldid,
        rayid,
      )
    else:
      shaded_body(p.geo, p.geo.cut, p.lit, p.cams, p.env, p.seed[0], p.out, worldid, rayid)  # pyrefly: ignore[bad-argument-type]


def _visibility_program(module: wp.Module, P: type[LaunchParams], f: Features, scene: SceneFlags) -> None:
  """`__raygen__visibility`: the light grid's visibility test (`light_grid.make_visit`) with the
  hardware traversal, unculled, without the deformables (`trace.BAKE_RAY_MASK`); launch index i is
  pair p.offset + i of the round (`light_flood.Pass`)."""
  grid = scene_tracer(scene, f, cull=False, mask=rt_trace.BAKE_RAY_MASK)
  visit = light_grid.make_visit(widen(grid.any_hit), f.area_lights)

  @_program(module, VISIBILITY, P)
  def raygen_visibility(p: LaunchParams):
    """The light grid: visibility of one light from one lattice point (one pair of the round)."""
    visit(p.geo, p.lit, p.grid, p.offset + native.optix_launch_index())


def _layer_call():
  """The layer loop's shading of one layer (`pixel.LayerShade`): a call of `__miss__layer`
  (`intrinsics.layer_call`); the layer's record is the first surface's only (zero here)."""

  @wp.func(inline=True)
  @device_generic
  def shade[G: GeometryFields, M: MaterialTables](
    # Model:
    _geo: G,
    _mats: M,
    _lit: Lights,
    _env: EnvironmentArrays,
    _p: Params,
    # In:
    worldid: int,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
    cone: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, Coverage, first_hit.SurfaceRecord]:
    """(radiance, coverage, record) of the layer at `hit` (see `_layer_call`)."""
    r = native.optix_layer_call(
      worldid,
      hit.instance_id,
      hit.dist,
      hit.normal,
      hit.bary_u,
      hit.bary_v,
      hit.face_id,
      hit.mesh_id,
      point,
      direction,
      cone,
      spread,
      headlight_dir,
      seed,
      stream,
    )
    cover = Coverage()
    cover.opacity = r.opacity
    cover.reflected = r.reflected
    cover.color = r.color
    cover.specular_only = r.specular_only != 0
    cover.roughness = r.roughness
    cover.base_color = r.base_color
    return r.radiance, cover, first_hit.zero_record()

  return shade


def _visibility_call(slot: int, *, tinted: bool):
  """The light loops' visibility of one light (`light_loop.Evaluators.visibility`): a call of
  `__miss__light_visibility` (`intrinsics.visibility_call`, payload type and miss record `slot`),
  which fetches the light itself (`punctual.fetch`: every traced light is a table light, the
  headlight casts no shadow)."""

  @wp.func(inline=True)
  @device_generic
  def visibility[G: GeometryFields](
    # Model:
    _geo: G,
    _lit: Lights,
    # In:
    worldid: int,
    _light: Light,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    seed: int,
    stream: int,
  ):
    """Visibility of light `li` from `point` (its rays from `origin`), RGB with tinted shadows."""
    v = native.optix_visibility_call(wp.static(slot), worldid, li, point, normal, origin, seed, stream)
    if wp.static(tinted):
      return v
    else:
      return v[0]

  return visibility


def _light_visibility_program(
  module: wp.Module, P: type[LaunchParams], f: Features, visibility: LightVisibilityAt, slot: int
) -> None:
  """`__miss__light_visibility`: one light's `visibility` (inline: `lighting.light_visibility`),
  with the launch parameters' structures, for `_visibility_call`."""
  tinted = f.tinted_shadows

  @_program(module, LIGHT_VISIBILITY, P)
  def miss_light_visibility(p: LaunchParams):
    """Traces the shadow rays of the call's light and returns its visibility."""
    a = native.optix_visibility_args(wp.static(slot))
    light = punctual.fetch(p.lit, a.world, a.light)
    v = visibility(p.geo, p.lit, a.world, light, a.light, a.point, a.normal, a.origin, a.seed, a.stream)
    if wp.static(tinted):
      native.optix_visibility_result(v)  # pyrefly: ignore[bad-argument-type]  # RGB (`punctual.visibility_ops`)
    else:
      native.optix_visibility_result(wp.vec3(v))


def _layer_program(
  module: wp.Module, P: type[LaunchParams], f: Features, tracer: Tracer, light_visibility: LightVisibilityAt | None
) -> None:
  """`__miss__layer`: one layer's shading (`pixel.make_layer_body`) over `tracer`'s queries (its
  light loops' visibility `light_visibility`: the light visibility program's call, or inline), with
  the launch parameters' structures, for `_layer_call`."""
  body = pixel.make_layer_body(f, tracer, light_visibility)

  @_program(module, LAYER, P)
  def miss_layer(p: LaunchParams):
    """Shades the layer of the call's arguments and returns its radiance and coverage."""
    a = native.optix_layer_args()
    hit = Hit()
    hit.instance_id = a.instance
    hit.dist = a.dist
    hit.normal = a.normal
    hit.bary_u = a.bary_u
    hit.bary_v = a.bary_v
    hit.face_id = a.face
    hit.mesh_id = a.mesh
    shaded, cover, _rec = body(
      p.geo,
      p.geo.cut,  # a shaded build's material carrier
      p.lit,
      p.env,
      p.env.params,
      a.world,
      hit,
      a.point,
      a.direction,
      a.cone,
      a.spread,
      a.headlight,
      a.seed,
      a.stream,
    )
    specular = int(0)
    if cover.specular_only:
      specular = 1
    native.optix_layer_result(shaded, cover.opacity, cover.reflected, cover.color, specular, cover.roughness, cover.base_color)


def _gi_programs(module: wp.Module, P: type[LaunchParams], f: Features, scene: SceneFlags) -> None:
  """The GI bake's programs (`gi_programs`): the software kernels' bodies (`shade.gi.kernels`:
  `make_probe_ray`, `make_static_ray`, `make_survey_ray`, `make_walls_texel`, `make_ao_texel`)
  over the hardware traversal, unculled, with the probe rays' Features (`probe.probe_features`).
  Launch index i is unit `p.offset + i` of the launch (ray-major for the rays of probes)."""
  pf = probe_lib.probe_features(f)
  tracer = scene_tracer(scene, pf, cull=False, mask=rt_trace.BAKE_RAY_MASK)
  # The probe hits' light loop calls the light visibility program, as the render's (no layer
  # program in this module: the first call slot).
  visibility = None
  if light_visibility_program(f):
    slot = intrinsics.visibility_slot(layer=False)
    visibility = _visibility_call(slot, tinted=pf.tinted_shadows)
    _light_visibility_program(module, P, pf, lighting.light_visibility(pf, make_queries(pf, tracer)), slot)
  closest = widen(tracer.closest)
  probe_ray = gi_kernels.make_probe_ray(probe_lib.make_probe_shading(pf, tracer, visibility))
  static_ray = gi_kernels.make_static_ray(closest)
  # The curve intersector reports no hit from inside a sphere or capsule: the relocation survey and
  # the walls' inside test need those back faces (a probe or texel inside a solid); the probe,
  # static and contact-occlusion rays start outside solids.
  inside = gi_kernels.with_back_faces(closest) if scene.has_world_curves or scene.has_shared_curves else closest
  survey_ray = gi_kernels.make_survey_ray(inside)
  walls_texel = gi_kernels.make_walls_texel(inside)
  ao_texel = gi_kernels.make_ao_texel(closest)

  @_program(module, PROBE, P)
  def raygen_probe(p: LaunchParams):
    """Probe ray of unit p.offset + i: (group, ray, probe) of the launch (`make_probe_ray`)."""
    q = p.gi_probe[0]
    i = p.offset + native.optix_launch_index()
    probe = i % q.nprobe
    rest = i // q.nprobe
    probe_ray(p.geo, p.geo.cut, p.lit, p.env, q, rest // q.nray, rest % q.nray, probe)

  @_program(module, PROBE_STATIC, P)
  def raygen_static(p: LaunchParams):
    """Static ray of unit p.offset + i: (ray, probe) of the launch (`make_static_ray`)."""
    q = p.gi_static[0]
    i = p.offset + native.optix_launch_index()
    static_ray(p.geo, q, i // q.nprobe, i % q.nprobe)

  @_program(module, PROBE_SURVEY, P)
  def raygen_survey(p: LaunchParams):
    """Survey ray of unit p.offset + i: (ray, point) of the launch (`make_survey_ray`)."""
    q = p.gi_survey[0]
    i = p.offset + native.optix_launch_index()
    survey_ray(p.geo, q, i // q.npoint, i % q.npoint)

  @_program(module, PROBE_WALLS, P)
  def raygen_walls(p: LaunchParams):
    """Walls of texel p.offset + i of the cube (x fastest, `make_walls_texel`)."""
    q = p.gi_walls[0]
    t = p.offset + native.optix_launch_index()
    counts = q.vol.cube.counts
    walls_texel(p.geo, q, t % counts[0], (t // counts[0]) % counts[1], t // (counts[0] * counts[1]))

  @_program(module, PROBE_AO, P)
  def raygen_ao(p: LaunchParams):
    """Contact occlusion of listed texel p.offset + i (`make_ao_texel`)."""
    ao_texel(p.geo, p.gi_ao[0], p.offset + native.optix_launch_index())


@wp.func
@device_generic
def _triangle_normal[G: OptixFields](
  # Model:
  geo: G,
  face_normal: wp.array1d[wp.uint32],
  # In:
  worldid: int,
  gi: int,
  face: int,
) -> wp.vec3:
  """Object-space normal of the current hit, triangle `face` of instance gi (not a box): a mesh's
  (or an SDF's, traced as its mesh) from the packed face table (`geo.cut`'s face addresses: the
  material carrier of a shaded build, else a Cutout); a height field's or deformable part's
  (`trace.FLEX_INSTANCE` + part) from its GAS's vertices (`dynamic.py`)."""
  if gi < wp.static(rt_trace.FLEX_INSTANCE):
    gtype = geo.instance_type[gi]
    if gtype == int(ShapeType.MESH) or gtype == int(ShapeType.SDF):
      mesh_id = in_table(batched(geo.instance_mesh, worldid, gi), geo.cut.mesh_adr.shape[0])
      return unpack_unit(face_normal[geo.cut.mesh_adr[mesh_id].face + face])
  return native.optix_triangle_face_normal()


def _hit_programs(module: wp.Module, P: type[LaunchParams], f: Features, scene: SceneFlags) -> None:
  """The miss, closest-hit, any-hit and intersection programs of both groups."""
  has_mesh = scene.has_mesh
  shared_visible = shared_visibility(f, scene)

  @_program(module, "__miss__render", P)
  def miss(_p: LaunchParams):
    """Miss: clears the hit flag of an occlusion query (closest hits keep instance -1)."""
    native.optix_payload_type_both()
    native.optix_miss()

  @_program(module, "__closesthit__triangle", P)
  def closesthit_triangle(p: LaunchParams):
    """Closest hit of a triangle: distance, OptiX's instance id (the instance), barycentrics,
    packed world face normal."""
    native.optix_payload_type_camera()
    # The hit's instance and, for a mesh, SDF, height field or deformable, its packed world face
    # normal.
    normal = wp.uint32(0)
    if wp.static(has_mesh):
      worldid, gi = hits.instance_of(native.optix_payload_world(), native.optix_instance_id())
      box = False
      if gi < wp.static(rt_trace.FLEX_INSTANCE):
        box = p.geo.instance_type[gi] == int(ShapeType.BOX)  # its normal: its cube face's (`trace.py`)
      if not box:
        n = _triangle_normal(p.geo, p.face_normal, worldid, gi, native.optix_primitive_index())
        normal = pack_unit(native.optix_object_to_world_normal(n))
    native.optix_closesthit_triangle(normal, native.optix_instance_id())

  anyhit = anyhit_programs(f, scene)
  for shadow in (False, True):
    if ("__anyhit__triangle_shadow" if shadow else "__anyhit__triangle") in anyhit:
      _triangle_anyhit(
        module,
        P,
        scene,
        cut=f.alpha_cutout,
        shared_visible=shared_visible,
        live_uv=f.live_material_uv,
        world_uv=f.world_uv,
        counting=f.counters,
        shadow=shadow,
        bound=f.bound,
        world_optics=f.world_optics,
      )
  alpha = AlphaTest(f.live_material_uv, f.world_uv) if f.alpha_cutout else None
  for name in ("__anyhit__curve", "__anyhit__curve_shadow", "__anyhit__curve_shared", "__anyhit__curve_shared_shadow"):
    if name in anyhit:
      shared = "_shared" in name
      curve = _CurveAnyhit(shared=shared, visible=shared and shared_visible, alpha=alpha, compact=f.compact_poses)
      _curve_anyhit(module, P, name, curve, counting=f.counters, bound=f.bound, world_optics=f.world_optics)

  @_program(module, "__closesthit__custom", P)
  def closesthit_custom(_p: LaunchParams):
    """Closest hit of a custom primitive: the attributes its intersection program reported."""
    native.optix_payload_type_camera()
    native.optix_closesthit_custom()

  cull = f.backface_culling
  for shared, has_curves, types in (
    (False, scene.has_world_curves, scene.primitive_types),
    (True, scene.has_shared_curves, scene.shared_primitive_types),
  ):
    if has_curves:
      _curve_program(module, P, compact=f.compact_poses, shared=shared, bound=f.bound)
    for gtype in types:
      for shadow in (False, True):
        _intersection_program(
          module,
          P,
          gtype,
          cull=(cull, f.two_sided),
          compact=f.compact_poses,
          shared=shared,
          visible=shared and shared_visible,
          shadow=shadow and f.translucent_shadows,
          alpha=alpha,
          name=intersection_name(gtype, shared=shared, shadow=shadow),
          bound=f.bound,
          world_optics=f.world_optics,
        )


@final
class _CurveAnyhit(NamedTuple):
  """What a curve any-hit program tests (`_curve_anyhit`).

  Attributes:
    shared: the shared group's curves (else the per-world group's).
    visible: a shared candidate is ignored where its world hides it.
    alpha: the alpha test (None: none): a cut-out candidate is ignored where its analytic hit
      (`hits.make_curve_hit`) fails it.
    compact: Features.compact_poses (the analytic hit's poses).
  """

  shared: bool
  visible: bool
  alpha: AlphaTest | None
  compact: bool


def _curve_anyhit(  # noqa: C901 (the program's compile-time tests)
  module: wp.Module, P: type[LaunchParams], name: str, curve: _CurveAnyhit, *, counting: bool, bound: int, world_optics: bool
) -> None:
  """A curve hit group's any-hit program `name` (`anyhit_programs`): a shared candidate is ignored
  where its world hides it (`curve.visible`; `bound`, Features.bound: where the build word has it,
  `toggles.GEO_SHARED_VISIBLE`), a cut-out one where the alpha test discards it (`curve.alpha`,
  where the word has an opacity map, `toggles.MAP_OPACITY`); on a shadow trace (a `_shadow`
  program), a translucent caster's multiplies the transmittance in (`_shadow_candidate`; the
  world's with `world_optics`, Features.world_optics)."""
  shadow = name.endswith("_shadow")
  shared, visible, alpha = curve.shared, curve.visible, curve.alpha
  gate = toggles.bound_on(bound)
  field = toggles.bound_on(bound, high=True)
  cut = alpha is not None
  curve_hit = hits.make_curve_hit(compact=curve.compact, shared=shared, bound=bound)
  cut_away = (
    cutout.make_cut_away(
      rt_trace.CURVE_TYPES, compact=curve.compact, live_uv=alpha.live_uv, world_uv=alpha.world_uv, bound=bound
    )
    if alpha is not None
    else None
  )

  @_program(module, name, P)
  def anyhit_curve(p: LaunchParams):
    """Candidate hit of a sphere / capsule (see `_curve_anyhit`)."""
    if wp.static(shadow):
      native.optix_payload_type_shadow()
    else:
      native.optix_payload_type_camera()
    index = native.optix_primitive_index()
    if wp.static(shared):
      worldid = native.optix_payload_world() - rt_trace.SHARED_RAY
      gi = p.shared_curve_instance[index]
    else:
      worldid, gi = hits.world_instance(index, p.curve_instance)
    if wp.static(counting):
      counters.add(p.geo, worldid, wp.static(counters.CURVE_ANYHIT), 1)
    hidden = False
    if wp.static(visible):
      if gate(p.geo.bits, wp.static(toggles.GEO_SHARED_VISIBLE)):
        hidden = not instance_visible(p.geo.visible_slot, p.geo.instance_visible, worldid, gi)
    if wp.static(cut):
      if not hidden and field(p.geo.bits2, wp.static(toggles.MAP_OPACITY)):
        if cutout.cut_out(p.geo, worldid, gi):
          if wp.static(shared):
            d, n, _gi = curve_hit(p.geo, p.shared_curve_instance)
          else:
            d, n, _gi = curve_hit(p.geo, p.curve_instance)
          direction = native.optix_world_ray_direction()
          point = native.optix_world_ray_origin() + direction * d
          if wp.static(not shared):
            point = rt_trace.from_cell(p.geo.rt_cell, worldid, point)  # out of the world's cell
          candidate = no_hit(d)
          candidate.instance_id = gi
          candidate.normal = n
          hidden = cut_away(p.geo, worldid, candidate, point, direction)  # pyrefly: ignore[not-callable]  # wp.static(cut)
    if hidden:
      if wp.static(counting):
        counters.add(p.geo, worldid, wp.static(counters.CURVE_ANYHIT_IGNORED), 1)
      native.optix_ignore_intersection()
    elif wp.static(shadow):
      if wp.static(world_optics):
        _world_shadow_candidate(p.geo, worldid, gi)
      else:
        _shadow_candidate(p.geo, gi)


def _triangle_anyhit(  # noqa: C901 (the program's compile-time tests)
  module: wp.Module,
  P: type[LaunchParams],
  scene: SceneFlags,
  *,
  cut: bool,
  shared_visible: bool,
  live_uv: bool,
  world_uv: bool,
  counting: bool,
  shadow: bool,
  bound: int,
  world_optics: bool,
) -> None:
  """`__anyhit__triangle` (ray type 0) / `__anyhit__triangle_shadow` (`shadow`: ray type 1):
  candidate hits of a cut-out instance (`cut`), of a shared one with a per-world visibility
  (`shared_visible`) and, on a shadow trace, of a translucent shadow caster (its transmittance,
  the world's with `world_optics`, multiplies the ray's, `intrinsics.shadow_attenuate`, and the
  candidate is ignored);
  `group.InstanceTables.flags` enforce the program on them. `bound` (Features.bound): the shared
  visibility test where the build word has it (`toggles.GEO_SHARED_VISIBLE`), the live material UV
  mappings (`toggles.MAP_LIVE_UV`)."""
  cut_triangle = cutout.make_cut_triangle(has_mesh=scene.has_mesh, has_box=scene.has_box, live_uv=live_uv, bound=bound)
  gate = toggles.bound_on(bound)

  @_program(module, "__anyhit__triangle_shadow" if shadow else "__anyhit__triangle", P)
  def anyhit_triangle(p: LaunchParams):  # noqa: C901 (compile-time tests)
    """Candidate triangle hit: ignored where the ray's world hides its instance, or the alpha test
    fails."""
    if wp.static(shadow):
      native.optix_payload_type_shadow()
    else:
      native.optix_payload_type_camera()
    world = native.optix_payload_world()
    worldid, gi = hits.instance_of(world, native.optix_instance_id())
    if wp.static(counting):
      counters.add(p.geo, worldid, wp.static(counters.TRIANGLE_ANYHIT), 1)
    hidden = False
    if wp.static(shared_visible):
      if gate(p.geo.bits, wp.static(toggles.GEO_SHARED_VISIBLE)):
        hidden = world >= rt_trace.SHARED_RAY and not instance_visible(p.geo.visible_slot, p.geo.instance_visible, worldid, gi)
    if wp.static(cut):
      if not hidden:
        handled = False
        if wp.static(world_uv):
          point = native.optix_world_ray_origin() + native.optix_world_ray_direction() * native.optix_ray_tmax()
          if world < wp.static(rt_trace.SHARED_RAY):
            point = rt_trace.from_cell(p.geo.rt_cell, worldid, point)
          normal = wp.vec3(0.0)
          face = native.optix_primitive_index()
          if p.geo.instance_type[gi] == int(ShapeType.BOX):
            side = face // 2
            normal[side // 2] = wp.where(side % 2 == 0, 1.0, -1.0)
          elif wp.static(scene.has_mesh):
            normal = _triangle_normal(p.geo, p.face_normal, worldid, gi, face)
          normal = native.optix_object_to_world_normal(normal)
          handled, hidden = cutout.world_cutout(p.geo, worldid, gi, point, normal)
        if not handled:
          bary = native.optix_triangle_barycentrics()
          o = native.optix_object_ray_origin()
          d = native.optix_object_ray_direction()
          hidden = cut_triangle(p.geo, worldid, gi, native.optix_primitive_index(), bary, o, d, native.optix_ray_tmax())
    if hidden:
      if wp.static(counting):
        counters.add(p.geo, worldid, wp.static(counters.TRIANGLE_ANYHIT_IGNORED), 1)
      native.optix_ignore_intersection()
    elif wp.static(shadow):
      if wp.static(world_optics):
        _world_shadow_candidate(p.geo, worldid, gi)
      else:
        _shadow_candidate(p.geo, gi)


@wp.func
@device_generic
def _shadow_candidate[G: OptixFields](
  # Model:
  geo: G,
  # In:
  gi: int,
):
  """A candidate of instance gi on a shadow trace: a translucent caster multiplies its transmittance
  in and is passed (`intrinsics.shadow_attenuate`); an opaque one is accepted (it ends the
  trace)."""
  t = geo.instance_shadow_transmit[gi]
  if wp.max(t[0], wp.max(t[1], t[2])) > 0.0:
    native.optix_shadow_attenuate(gi, t)
    native.optix_ignore_intersection()


@wp.func
@device_generic
def _world_shadow_candidate[G: OptixFields](
  # Model:
  geo: G,
  # In:
  worldid: int,
  gi: int,
):
  """`_shadow_candidate` with world `worldid`'s transmittance (Features.world_optics,
  `trace.world_entry`)."""
  t = world_transmit(geo, worldid, gi)
  if wp.max(t[0], wp.max(t[1], t[2])) > 0.0:
    native.optix_shadow_attenuate(gi, t)
    native.optix_ignore_intersection()


def _curve_program(module: wp.Module, P: type[LaunchParams], *, compact: bool, shared: bool, bound: int) -> None:
  """The closest-hit program of a group's curves (`hits.make_curve_hit`; `bound`:
  Features.bound)."""
  curve_hit = hits.make_curve_hit(compact=compact, shared=shared, bound=bound)
  name = "__closesthit__curve_shared" if shared else "__closesthit__curve"

  @_program(module, name, P)
  def closesthit_curve(p: LaunchParams):
    """Closest hit of a sphere / capsule: the analytic hit on the instance
    (`hits.make_curve_hit`)."""
    native.optix_payload_type_camera()
    if wp.static(shared):
      d, n, gi = curve_hit(p.geo, p.shared_curve_instance)
    else:
      d, n, gi = curve_hit(p.geo, p.curve_instance)
    native.optix_set_hit(d, gi, native.optix_primitive_index(), n)


def _intersection_program(  # noqa: PLR0913  # the program's compile-time options
  module: wp.Module,
  P: type[LaunchParams],
  gtype: int,
  *,
  cull: tuple[bool, bool],
  compact: bool,
  shared: bool,
  visible: bool,
  shadow: bool,
  alpha: AlphaTest | None,
  name: str,
  bound: int,
  world_optics: bool,
) -> None:
  """The intersection program `name` of a group's custom primitives of type `gtype`
  (`hits.make_intersection`; `cull`: (back-face culling, two-sided instances); `world_optics`:
  their flags and transmittance per world; `visible`: shared
  primitives test their per-world visibility; `shadow`: a translucent caster on a shadow trace
  multiplies its transmittance in and reports nothing; `alpha`: the alpha test (None: none);
  `bound`: Features.bound)."""
  intersect = hits.make_intersection(
    gtype=gtype,
    cull_backfaces=cull[0],
    two_sided=cull[1],
    compact=compact,
    shared=shared,
    visible=visible,
    shadow=shadow,
    alpha=alpha,
    bound=bound,
    world_optics=world_optics,
  )

  @_program(module, name, P)
  def intersection(p: LaunchParams):
    """Intersection of a custom primitive: reports the analytic hit (`hits.make_intersection`)."""
    if wp.static(name.endswith("_shadow")):
      native.optix_payload_type_shadow()
    else:
      native.optix_payload_type_camera()
    if wp.static(shared):
      d, n, gi = intersect(p.geo, p.shared_prim_code)
    else:
      d, n, gi = intersect(p.geo, p.prim_code)
    if d >= 0.0:
      native.optix_report_intersection(d, n, gi)
