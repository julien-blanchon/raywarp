"""Turns the user's settings into what the render kernels are compiled for: `resolve_features`
maps settings x scene capabilities x per-world declaration x device to the kernel
specialization (`Features`) and the runtime parameters (`Params`), made canonical
(`core/build/variants/canonical.py`: the fields a kernel cannot see collapsed, so equal code means
one cached kernel).

A pure function (host values only, nothing allocated on a device): the Renderer calls it when
its settings change, and anything that needs to know what a configuration compiles (the OptiX
backend's build key, the Newton adapters) reads its result instead of the Renderer's state.
"auto" settings resolve against what the scene uses (`GpuScene.capabilities`) and what the
`PerWorld` declaration randomizes (a per-world `material.mirror` compiles its reflectance in, ...):
per-world fields come only from that declaration, so writing a per-world table never
reconfigures.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, NamedTuple, final

import numpy as np
import warp as wp

from raywarp._base.types import MESH_LIKE, ROW_MEDIUM, TWO_SIDED, ShadingFlag, ShapeType, TexRole
from raywarp.core.build.kernel_cache import policy
from raywarp.core.build.variants import tiers
from raywarp.core.build.variants.canonical import NO_LAYERS, canonical
from raywarp.core.kernels import reflections
from raywarp.core.kernels.abi import structs as kernel_structs
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import (
  LIGHT_FORM_GENERAL,
  PLANES_SOME,
  RUNTIME_LIGHTS,
  SMALL_LIGHTS_MAX,
  Features,
  LayerFeatures,
  LightFeatures,
  MaterialFeatures,
  ShadowFeatures,
  SurfaceFeatures,
  TuningFeatures,
  plane_form,
  small_form,
)
from raywarp.core.kernels.abi.output_layout import check_outputs, layout_of
from raywarp.core.pipeline.derive.fields import COAT_FIELDS
from raywarp.core.pipeline.gpu_scene.gpu_scene import shadow_transmit
from raywarp.core.pipeline.platform import tuning
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.upload.optics import OpticsInputs
from raywarp.core.pipeline.worlds import optics as world_optics
from raywarp.core.shade.environment import envmap
from raywarp.core.trace import backend as trace_backend
from raywarp.core.trace.rays import primary
from raywarp.settings import RenderSettings
from raywarp.settings import fields as settings_fields

if TYPE_CHECKING:
  from raywarp.core.pipeline.gpu_scene.gpu_scene import GpuScene, PreparedMaterialCapabilities
  from raywarp.core.pipeline.platform.device import DeviceCaps
  from raywarp.core.pipeline.worlds.sharing import Layout, PerWorld
  from raywarp.core.trace.backend import TraceBackend

# Measured tuning defaults (per-architecture tuning sweeps).
DEFAULT_MAX_REGISTERS: Final = 96
PRIMITIVE_BLOCK_DIM: Final = 256
DEFAULT_BLOCK_DIM: Final = 64
# The optics inputs of no material choice (`_switched_optics`).
_NO_OPTICS: Final = OpticsInputs(
  medium=np.zeros((0, 4), np.float32),
  volume=np.zeros(0, bool),
  see=np.zeros((0, 3), np.float32),
  see_alpha=np.zeros(0, np.float32),
  transmissive=np.zeros(0, bool),
  base=np.zeros((0, 4), np.float32),
  color=np.zeros((0, 4), np.float32),
  doubled=np.zeros(0, bool),
  closed=np.zeros(0, bool),
  chord=np.zeros(0, np.float32),
  material=np.zeros(0, np.int32),
)
# Light count from which lighting.light_grid="auto" builds the light grid: below it the grid's
# cell lookup and dynamic light loop cost more than the lights it culls (the robotics scenes and
# 2-4 lights in closed rooms: off 0-4 % faster; 8 lights in 4 rooms: on 7 % faster; 512 px, L4).
GRID_MIN_LIGHTS: Final = 8
# `Features.glossy` with traced reflections (`reflections.enabled`: one ray along the GGX dominant
# direction; 0: none).
GLOSSY_DOMINANT: Final = 1
# Spread of a caster's RGB transmittance above which it tints shadows (else gray).
_TINT_SPREAD: Final = 1e-6


@final
@dataclasses.dataclass(frozen=True)
class RenderCaps:
  """What one Renderer offers the feature resolution, fixed at its construction.

  Attributes:
    scene: the GpuScene (its capabilities, environment, lights and tables).
    layout: the shared-scenery layout (`sharing.decide`).
    ncam: cameras.
    bundle: the cameras trace ray bundles (`CameraSpec.rays`).
    distortion: a camera has a lens model (a fisheye, `CameraSpec.distortion`).
    The last two choose the camera model (`camera_model`), never what a build compiles.
    uniform: every camera has the same resolution (the motion output needs it).
  """

  scene: GpuScene
  layout: Layout
  ncam: int
  bundle: bool
  distortion: bool
  uniform: bool


# Tier Features memoised per Renderer (`resolve_features(tier_builds=)`): the tiers x a few run-time
# variants; past it the memo starts over.
_TIER_MEMO: Final = 64


@final
class Resolution(NamedTuple):
  """What `resolve_features` resolves.

  Attributes:
    features: what the render kernels compile (a tier's, or the settings' own outside every tier;
      the process-global kernel cache key).
    params: the runtime parameters (`Params`, with the `Toggles` of the effective Features).
    effective: what the settings resolve to (the specialised Features: the host's decisions read
      it, e.g. whether the GI volume is baked; `features` itself outside the tiers).
    choice: the tier serving the settings (`tiers.route`; None: `performance.kernels="scene"`).
    relight: the GI relight kernel's build (`relight_build`; None without GI).
    camera: the cameras' model (`camera_model`): the ray-generation program the build launches.
  """

  features: Features
  params: kernel_structs.Params
  effective: Features
  choice: tiers.Choice | None
  relight: Features | None
  camera: primary.CameraModel


def resolve_features(
  settings: RenderSettings,
  caps: RenderCaps,
  per_world_decl: PerWorld | None,
  device: DeviceCaps,
  *,
  casters: int,
  tier_builds: dict[tuple[RenderSettings, int, tuning.Tuning, tiers.Axes], Features] | None = None,
) -> Resolution:
  """The kernels and runtime parameters of `settings` for `caps` on `device` (ValueError for
  settings the scene or backend cannot honor: environment=True without maps, the motion output
  with a supersampling backend at spp > 1). `casters`: `shadow_casters` of the call's slots;
  `tier_builds`: a memo of the tiers' Features for these `caps`, `per_world_decl` and `device`
  (the caller's, e.g. one per Renderer: a settings change then resolves only its own Features).

  The one routing point of the kernel build (`performance.kernels`): "prebuilt" serves a tier's
  build (`tiers.route`), resolved without the scene in its feature classes, when the settings are
  a tier's or differ from one only in switches turned off, so that neither a scene nor such a
  settings change ever compiles (the bundle holds those builds); else the settings' own. "scene"
  serves the settings' own Features, specialised on the scene.
  """
  effective = _resolve(settings, caps, per_world_decl, device, casters=casters)
  params = _params(settings, caps)
  params.rt = toggles.of(effective)  # also a specialised build's: reflections off reads it
  relight = relight_build(settings, caps, per_world_decl, device, casters=casters) if effective.gi else None
  declared = frozenset(per_world_decl.fields()) if per_world_decl is not None else frozenset()
  model = camera_model(caps, declared)
  if settings.performance.kernels == "scene":
    return Resolution(effective, params, effective, None, relight, model)

  def tier_build(s: RenderSettings, tier: str) -> Features:
    """Settings' scene-free Features (a tier's, `tiers.tier_settings`, or the configuration's own
    to compare with them: `tiers.route`) on the configuration's scene axes for tier `tier`
    (`tiers.axes`: a complete tier's every class, a sampled or counted tier's run-time sampling and
    counts), memoised."""
    of = tiers.axes(effective, tier)
    if tier_builds is None:
      return _tier_build(s, caps, per_world_decl, device, casters=casters, of=of)
    key = (s, casters, tuning.current(), of)  # the tuning goes into the Features (`_tuning`)
    if key not in tier_builds:
      if len(tier_builds) >= _TIER_MEMO:
        tier_builds.clear()
      tier_builds[key] = _tier_build(s, caps, per_world_decl, device, casters=casters, of=of)
    return tier_builds[key]

  moot = _scene_moot(settings, caps, per_world_decl, device, casters=casters)
  choice = tiers.route(settings, effective, tier_build, moot=moot)
  if effective.counters:  # a debug build (tier builds count nothing): its own kernels
    choice = tiers.Choice(tier=None, build=effective, outside=("counters",), nearest=choice.tier or choice.nearest)
  if choice.tier is not None:
    params.rt.flags |= tiers.run_flags(choice.build, effective)  # its run-time sampling
  return Resolution(choice.build, params, effective, choice, relight, model)


def refuse_unprebuilt(settings: RenderSettings, choice: tiers.Choice | None, device: DeviceCaps) -> None:
  """With compiling off (`policy.configure(compile=False)`), raises `policy.KernelsNotPrebuilt` for
  a configuration the prebuilt kernels cannot serve: the CPU device, `performance.kernels="scene"`,
  settings outside every prebuilt tier (naming the fields that differ from the nearest tier)."""
  if policy.current().compile:
    return
  if device.kind != "cuda":
    policy.refuse("a renderer on the CPU device (CPU kernels always compile)")
  if settings.performance.kernels == "scene":
    policy.refuse('performance.kernels="scene" (kernels specialised on the scene)')
  if choice is None or choice.tier is None:
    fields = ", ".join(choice.outside) if choice is not None and choice.outside else "debug counters"
    nearest = f"; nearest tier {choice.nearest!r}" if choice is not None and choice.nearest else ""
    policy.refuse(f"these settings (outside every prebuilt tier: {fields}{nearest})")


def camera_model(caps: RenderCaps, declared: frozenset[str]) -> primary.CameraModel:
  """The camera model of a renderer's cameras (`primary.CameraModel`): ray bundles (every camera has
  one or none); a lens model (a fisheye, OpenCV distortion, or per-world "camera.distortion");
  else the pinhole."""
  if caps.bundle:
    return primary.BUNDLE
  if caps.distortion or "camera.distortion" in declared:
    return primary.DISTORTION
  return primary.PINHOLE


def relight_build(
  settings: RenderSettings, caps: RenderCaps, per_world_decl: PerWorld | None, device: DeviceCaps, *, casters: int
) -> Features:
  """The GI relight kernel's build of `settings` (`kernels/probe.relight_features`): resolved
  without the scene in every field a tier build can take from it (`tiers.SCENE_FREE`,
  `tiers.BOUND_FREE`: every class, structure and light), the general light loop; the same for
  every scene, so that the relight kernel is one per probe-relevant settings (the prebuilt
  bundle's: one per tier). The translucent layers are compiled in whatever the settings say, read
  from the relight word (`tiers.SETTING_FIELDS`). The live material UV mappings stay the scene's:
  the relight launches the software geometry struct of the frame, whose layout they select
  (`frame_structs.geometry`)."""
  from raywarp.core.kernels import probe  # noqa: PLC0415  # import cycle: probe -> features.gi -> ... -> resolve

  free = (tiers.SCENE_FREE | tiers.BOUND_FREE) - {"live_material_uv"}
  layered = settings.replace(**{"transparency.enabled": "auto"})
  f = _resolve(layered, caps, per_world_decl, device, casters=casters, free=free, general=True)
  return probe.relight_features(f)


# The surface fields a scene without translucent content (no alpha, no transmissive or solid glass
# material) never reads, and the cut-out field a scene without cut-out material never reads.
_TRANSLUCENT_FIELDS: Final = frozenset({"transparency", "transmissive", "refraction", "glass_shapes"})
# The analytic primitives whose solid glass `Features.glass_shapes` is on for.
_GLASS_SHAPE_TYPES: Final = tuple(
  int(t) for t in (ShapeType.SPHERE, ShapeType.BOX, ShapeType.CAPSULE, ShapeType.ELLIPSOID, ShapeType.CYLINDER)
)


def _scene_moot(
  s: RenderSettings, caps: RenderCaps, per_world_decl: PerWorld | None, device: DeviceCaps, *, casters: int
) -> frozenset[str]:
  """The Features fields whose value no frame of `s` on this scene reads (`tiers.route`'s `moot`):
  the translucent layers on a scene that has none (resolved with transparency "auto": a complete
  tier's translucent build composites nothing there), the alpha test on a scene without cut-out
  material (no cutoff: no instance is tested, no surface made opaque, `trace/accel/cutout.py`)."""
  declared = frozenset(per_world_decl.fields()) if per_world_decl is not None else frozenset()
  moot = frozenset() if _cutouts(caps.scene, declared, frozenset()) else frozenset({"alpha_cutout"})
  if s.transparency.enabled is not False:
    return moot
  auto = _resolve(s.replace(**{"transparency.enabled": "auto"}), caps, per_world_decl, device, casters=casters)
  return moot if auto.transparency or auto.transmissive or auto.refraction else moot | _TRANSLUCENT_FIELDS


def _tier_build(
  s: RenderSettings,
  caps: RenderCaps,
  per_world_decl: PerWorld | None,
  device: DeviceCaps,
  *,
  casters: int,
  of: tiers.Axes,
) -> Features:
  """The tier build of settings `s` serving a configuration of scene axes `of`: resolved without
  the scene (`tiers.SCENE_FREE`) but in the scene's feature classes and the configuration's
  (`tiers.class_free`: a class neither uses is left out), its light-loop form on the
  configuration's light grid (`tiers.Axes.grid`)."""
  # A complete tier's general light loop (`tiers.COMPLETE`); the configuration's motion history.
  axes = {"grid": of.grid, "general": of.complete, "motion": of.motion, "sampled": of.sampled}
  own = _resolve(s, caps, per_world_decl, device, casters=casters, **axes)
  free = tiers.tier_free(own, of)
  build = _resolve(s, caps, per_world_decl, device, casters=casters, free=free, **axes)
  # A class the build does not compile is off, whatever the scene holds: the configuration uses
  # none of it (`of.classes`), though the tier's settings would resolve some of its fields on
  # (a cut-out scene with textures off: the tier's free texture slots would turn the cut-outs on).
  off = tiers.unserved_fields(own, of)
  flags: dict[str, bool] = dict.fromkeys(off, False)
  build = canonical(dataclasses.replace(build, **flags)) if off else build  # pyrefly: ignore[bad-argument-type]
  return _counted(build) if of.counted else build


def _counted(f: Features) -> Features:
  """Tier build `f` reading its light caps and shadow budget at run time (`Features.run_counts`): at
  the capacities (`toggles.COUNTS_CAPACITY`), both budget remainders compiled, the launch's values
  (`Toggles`) selecting below them."""
  cap = toggles.COUNTS_CAPACITY
  return dataclasses.replace(
    f,
    budget=cap["budget"] if f.shadows else 0,
    light_cap=cap["light_cap"],
    hit_light_cap=cap["hit_light_cap"] if f.glossy else 0,
    run_counts=True,
  )


def _resolve(
  settings: RenderSettings,
  caps: RenderCaps,
  per_world_decl: PerWorld | None,
  device: DeviceCaps,
  *,
  casters: int,
  free: frozenset[str] = frozenset(),
  grid: bool | None = None,
  general: bool = False,
  motion: bool | None = None,
  sampled: bool = False,
) -> Features:
  """The Features of `settings` (`resolve_features`); `free`: a tier build's fields resolved
  without the scene (`tiers.SCENE_FREE`, less the feature classes the scene does not use:
  `tiers.class_free`); `grid`: whether the scene's light loop takes the light grid (a tier
  build's: the configuration's, `tiers.Axes`), else the settings'; `general`: the general light
  loop whatever the scene's lights (a complete tier's); `motion`: the motion history (a tier
  build's: the configuration's), else the settings' outputs'; `sampled`: the shadows' sampling
  read at run time (`Features.run_sampling`, a `tiers.SAMPLED` tier's)."""
  s, scene, layout = settings, caps.scene, caps.layout
  declared = frozenset(per_world_decl.fields()) if per_world_decl is not None else frozenset()
  backend = trace_backend.select(s.performance.traversal, scene, device)
  shaded, sc, nflexprim = s.outputs.shaded, scene.capabilities, scene.deformables.nprim
  grid = _grid(s, scene.nlight) if grid is None else grid
  lights, glossy = _lights(s, scene, declared, free, grid=grid), _glossy(s)
  check_outputs(s, scene)
  surfaces = _surfaces(s, scene, declared, free, alpha=per_world_decl is not None and per_world_decl.alpha)
  types = tuple(sc.shape_types)
  motion = _motion(s, backend, uniform=caps.uniform, motion=motion)
  if free:
    # A tier build's shape types: the triangle kinds, or every shape the backend traces.
    inlined = backend.name != "optix"  # the software traversal inlines each kind's intersection
    types = tiers.geometry_kinds(types, backend.shape_types, every="shape_types" in free, flex="flex" in free, inlined=inlined)
  features = Features(
    shape_types=types,
    traversal=backend.name,
    # An OptiX tier build traces deformables in every scene (`tiers.BOUND_FREE`): the software
    # helper kernels beside it take the scene's traversal order (`features.software_view`).
    **_tuning(types, device, shaded=shaded, nflexprim=0 if "flex" in free else nflexprim),
    spp=_spp(s, motion=motion, supersamples=backend.supersamples),
    any_rgb=shaded,
    # The fog: its values', a per-world range's, or a tier build's (compiled in, a run-time switch).
    fog=shaded and ("fog" in free or s.atmosphere.fog or "fog.range" in declared),
    unlit=shaded and s.surfaces.shading == "unlit",
    **lights,
    **_shadows(s, scene, declared, free, casters=casters, glossy=glossy),
    **surfaces,
    **(_layers(scene, declared, free, shaded=shaded, surfaces=surfaces) if s.openpbr.layers is not False else NO_LAYERS),
    specular_aa=shaded,  # always on (no setting): `canonical` turns it off where unread
    **_materials(s, scene, declared, free, glossy=glossy, area_lights=lights["area_lights"]),
    mesh_frames=("mesh_frames" in free or sc.mesh_frames) and backend.mesh_frames,
    motion=motion,
    # An OptiX tier build compiles GI in, the setting read from the build word (`tiers.BOUND_FREE`).
    gi=(s.gi.enabled or "gi" in free) and shaded,
    glossy=glossy,
    visible_lights=shaded and ("visible_lights" in free or sc.visible_lights),
    # An OptiX tier build: every structure, the plane loop, compact poses and the visibility table
    # compiled in, each read from the build word (`tiers.BOUND_FREE`).
    shared="shared" in free or layout.nshared > 0,
    world_tree="world_tree" in free or len(scene.rendered) - layout.nshared - len(layout.planes) + nflexprim > 0,
    compact_poses="compact_poses" in free or layout.compact,
    visibility="visibility" in free or layout.nvisible > 0,
    planes=PLANES_SOME if "planes" in free else plane_form(backend.analytic_planes(scene, layout)),
    # A debug build (`core/pipeline/platform/tuning.Tuning.counters`): its own kernels, never a
    # tier's.
    counters=tuning.current().counters and shaded and not free,
    run_sampling=sampled,
    switches=0,
    scene_free=bool(free),
    # The written outputs (with the deferred display's "hdr") as a software build's modules define
    # them (`toggles.preamble`); an OptiX module binds them per frame.
    out_word=0 if backend.name == "optix" else layout_of(s.written_outputs()).word,
  )
  formed = dataclasses.replace(features, light_forms=_light_forms(features, scene.nlight, grid=grid or general))
  return canonical(dataclasses.replace(formed, nlight=_light_count(formed, scene.nlight)))


def _grid(s: RenderSettings, nlight: int) -> bool:
  """Whether the settings `s` build the light grid for a scene of `nlight` lights."""
  grid = s.lighting.light_grid
  return nlight >= GRID_MIN_LIGHTS if grid == "auto" else grid


def _light_forms(f: Features, nlight: int, *, grid: bool) -> int:
  """Features.light_forms of `f` for a scene of `nlight` lights (also in a tier build:
  `tiers.LOOP_FORM`): none without the camera rays' shadowed loop, else the small form when the
  table fits it (SMALL_LIGHTS_MAX lights, whatever the budget and the cap: a lower count never
  leaves it) and the scene builds no light grid (`grid`; an OptiX tier build compiles the grid's
  lookup in and binds it, `tiers.BOUND_FREE`), the general one when not. The small form has
  `nlight` entries, also in a tier build (each entry compiled in unused costs its light term and
  shadow-ray site: a 2-light fast raygen +9.6 % SASS with one more; the count bound within a
  larger form left +0.40-0.95 %), but with soft shadows (`Features.shadow_samples` > 1, the
  sampled variants: their few small-light scenes do not pay for their own builds) the largest.
  IES builds follow the same rule: an interior with IES profiles and 4 lights ran default 3.3x
  slower in the general form than in its small form (RTX PRO 6000, 512 worlds at 512^2)."""
  if not (f.any_rgb and f.shadows and not f.unlit):
    return 0
  size = SMALL_LIGHTS_MAX
  if not 0 < nlight <= size or grid:
    return LIGHT_FORM_GENERAL
  return size if f.nlight == RUNTIME_LIGHTS and f.shadow_samples > 1 else nlight


def _light_count(f: Features, nlight: int) -> int:
  """Features.nlight of `f` with light-loop form `f.light_forms` for a scene of `nlight` lights:
  a tier build's run-time count (RUNTIME_LIGHTS) is the scene's where its small form has exactly
  that many entries (one build per count either way; a constant bound: a 2-light fast raygen
  -7.1 % SASS)."""
  return nlight if f.nlight == RUNTIME_LIGHTS and 0 < small_form(f.light_forms) == nlight else f.nlight


def _env_lit(s: RenderSettings, scene: GpuScene) -> bool:
  """Whether the environment maps light the scene ("auto": as the scene says)."""
  env = scene.environment
  if s.environment.enabled == "auto":
    return env.lit
  if s.environment.enabled and not env.lit:
    raise ValueError(
      "environment.enabled=True needs a scene with a lighting environment map (this scene has none, or only a"
      " background map such as a MuJoCo skybox: pass envmap= to light it)"
    )
  return s.environment.enabled


def _tuning(shape_types: tuple[int, ...], device: DeviceCaps, *, shaded: bool, nflexprim: int) -> TuningFeatures:
  """block_dim, max_registers, ordered_traversal of a build for `shape_types`: `tuning.current()`,
  None = measured default."""
  perf = tuning.current()
  types = set(shape_types)
  has_meshes = bool(set(MESH_LIKE) & types) or int(ShapeType.FLEX) in types
  block_dim = perf.block_dim
  if block_dim is None:
    block_dim = DEFAULT_BLOCK_DIM if (has_meshes or device.blackwell_rtx) else PRIMITIVE_BLOCK_DIM
  ordered = perf.ordered_traversal
  if ordered is None:
    ordered = shaded and (has_meshes or not device.blackwell_rtx)
  return {
    "block_dim": block_dim,
    # A CUDA register cap (no meaning on the CPU: one CPU kernel for every cap).
    "max_registers": 0 if device.kind == "cpu" else DEFAULT_MAX_REGISTERS if perf.max_registers is None else perf.max_registers,
    # Deformables keep the unordered traversal: the ordered one gives the same pixels but is no
    # faster on CPU (cloth_mannequin 4 x 128^2: +1 %) and 7-17 % slower on the RTX PRO 6000
    # (4096 x 64^2, local memory +224 B).
    "ordered_traversal": ordered and nflexprim == 0,
  }


def _lights(s: RenderSettings, scene: GpuScene, declared: frozenset[str], free: frozenset[str], *, grid: bool) -> LightFeatures:
  """The light loop's Features: light caps, the light grid (`grid`: the scene's lights take it),
  area lights, headlight, environment (`free`: the fields resolved without the scene,
  `tiers.SCENE_FREE`)."""
  lit, n = s.lighting, scene.nlight
  some = n > 0 or "nlight" in free
  return {
    "nlight": RUNTIME_LIGHTS if "nlight" in free else n,
    "has_spot": "has_spot" in free or scene.capabilities.has_spot or "light.type" in declared,
    # IES profiles: a fixed per-light assignment (no per-world slot), so the scene's alone.
    "ies": some and ("ies" in free or scene.capabilities.has_ies),
    # "auto" builds the light grid from GRID_MIN_LIGHTS lights: with fewer, its lookup and the
    # cell's dynamic light loop cost shading time for little or nothing (a single sun is listed in
    # every cell: 15 % of fast on an L4). Scene-free: the lookup is compiled wherever a grid may be
    # built; a scene without one binds the one-cell placeholder.
    "culling": (lit.light_grid is not False) if "culling" in free else grid,
    "light_cap": (lit.max_exact or 0) if "light_cap" in free else _cap(lit.max_exact, n),
    "hit_light_cap": (s.reflections.max_exact or 0) if "hit_light_cap" in free else _cap(s.reflections.max_exact, n),
    "area_lights": _area_lights(s, scene, declared, free),
    # Scene-free: compiled in, its intensity data (0 when off: `_params`).
    "headlight": True if "headlight" in free else (scene.headlight.active if lit.camera_light == "auto" else lit.camera_light),
    # Scene-free: compiled wherever the settings allow it, switched at run time (`tiers.SWITCHES`).
    "env_lit": (s.environment.enabled is not False) if "env_lit" in free else _env_lit(s, scene),
    "env_background": s.environment.background and ("env_background" in free or scene.environment.has_maps),
  }


def _cap(cap: int | None, n: int) -> int:
  """A per-point light cap's Features value: 0 (none) when None or not below the `n` lights."""
  return cap if cap is not None and cap < n else 0


def _glossy(s: RenderSettings) -> int:
  """Features.glossy: the glossy reflection rays (`GLOSSY_DOMINANT`; 0 without reflections)."""
  return GLOSSY_DOMINANT if s.outputs.shaded and s.reflections.enabled else 0


def _area_lights(s: RenderSettings, scene: GpuScene, declared: frozenset[str], free: frozenset[str]) -> bool:
  """Features.area_lights: LTC area lights, when the scene has some or declares their shape (or
  resolved without the scene)."""
  return s.lighting.area_lights == "ltc" and (
    "area_lights" in free or scene.capabilities.has_area_lights or "light.shape" in declared
  )


def _prepared(scene: GpuScene, declared: frozenset[str]) -> PreparedMaterialCapabilities | None:
  """What every prepared material uses when the per-world declaration selects materials
  ("instance.material": any of them can be rendered), else None (the rendered ones count)."""
  return scene.capabilities.prepared_materials if "instance.material" in declared else None


def _switched_optics(scene: GpuScene, declared: frozenset[str]) -> OpticsInputs:
  """The optics inputs a per-world material choice composes per world (`world_optics.py`), empty
  without one: what can turn on refraction, translucent casters or two-sided instances."""
  inputs = world_optics.switched(scene, declared)
  return inputs if inputs is not None else _NO_OPTICS


def _glass_shapes(scene: GpuScene, declared: frozenset[str]) -> bool:
  """Whether a rendered analytic primitive can be solid glass (`Features.glass_shapes`): one has
  a medium row, or a per-world material choice composes one (`_switched_optics`) on a scene with
  primitives. Reads the instance types (one host copy) only for a scene with solid glass."""
  solid = np.any(scene.instance_optics[scene.rendered, ROW_MEDIUM] != 0.0, axis=-1)
  if np.any(_switched_optics(scene, declared).medium != 0.0):
    solid = np.ones_like(solid)  # any material on any instance, per world
  if not solid.any():
    return False
  shapes = np.isin(host_copy(scene.instance_type)[scene.rendered], _GLASS_SHAPE_TYPES)
  return bool(np.any(shapes & solid))


def _materials(
  s: RenderSettings, scene: GpuScene, declared: frozenset[str], free: frozenset[str], *, glossy: int, area_lights: bool
) -> MaterialFeatures:
  """The material capabilities a build compiles: what the rendered (or, with declared material
  selection, the prepared) materials use, or a declared per-world field that can turn them on;
  resolved without the scene, every field in `free` (`glossy`, `area_lights`: the build's)."""
  sc, shaded = scene.capabilities, s.outputs.shaded
  prepared = _prepared(scene, declared)
  flex_curve = scene.deformables.has_material and bool(np.any(scene.derive_inputs.shading & int(ShadingFlag.SPECULAR_CURVE)))
  live_shading = "instance.shading" in declared
  return {
    "reflection": shaded
    and (
      "reflection" in free or sc.has_mirror or (prepared is not None and prepared.has_mirror) or "material.mirror" in declared
    ),
    "uv_maps": "uv_maps" in free or sc.uv_maps,
    # An OptiX tier build compiles the live UV address in, read from its word (`tiers.BOUND_FREE`).
    "live_material_uv": "live_material_uv" in free or (scene.material_uv is not None and "instance.material" in declared),
    "world_uv": "world_uv" in free
    or sc.world_uv
    or (prepared is not None and prepared.world_uv)
    or "material.uv_world" in declared,
    "scaled_normal_map": "scaled_normal_map" in free
    or (prepared.textures.scaled_normal if prepared is not None else sc.textures.scaled_normal)
    or "material.tex_affine" in declared,
    "diffuse_roughness": shaded and _diffuse_roughness(s, scene, declared, free),
    "clearcoat": shaded
    and (
      "clearcoat" in free
      or sc.clearcoat
      or (prepared is not None and prepared.clearcoat)
      or live_shading
      or bool(declared & set(COAT_FIELDS))
    ),
    "specular_weight": shaded
    and ("specular_weight" in free or _specular_weight(scene, declared) or live_shading or flex_curve),
    "specular_curve": shaded
    and (
      "specular_curve" in free
      or sc.specular_curve
      or (prepared is not None and prepared.specular_curve)
      or live_shading
      or flex_curve
    ),
    "transmissive": shaded
    and s.transparency.enabled
    and ("transmissive" in free or sc.transmissive or (prepared is not None and prepared.transmissive) or live_shading),
    "refraction": shaded
    and s.transparency.enabled
    and (
      "refraction" in free
      or bool(np.any(scene.instance_optics[scene.rendered, ROW_MEDIUM] != 0.0))
      or bool(np.any(_switched_optics(scene, declared).medium != 0.0))
    ),
    "glass_shapes": shaded and s.transparency.enabled and ("glass_shapes" in free or _glass_shapes(scene, declared)),
    "light_panels": bool(glossy and area_lights and ("light_panels" in free or sc.light_panels) and s.lighting.emissive_lights),
    "vertex_colors": shaded
    and ("vertex_colors" in free or sc.vertex_colors or (prepared is not None and prepared.vertex_colors)),
    # Unlit materials (`ShadingFlag.UNLIT`, per instance; a declared shading table can set it).
    "unlit_emission": shaded
    and ("unlit_emission" in free or sc.unlit_emission or (prepared is not None and prepared.unlit_emission) or live_shading),
    "flex_materials": shaded and ("flex_materials" in free or scene.deformables.has_material),
  }


def _diffuse_roughness(s: RenderSettings, scene: GpuScene, declared: frozenset[str], free: frozenset[str]) -> bool:
  """Features.diffuse_roughness (of a shaded build): forced by the setting, else used by a
  material or texture of the scene, or settable per world (a declared roughness field, or a
  material selection over prepared roughness rows)."""
  setting, sc = s.openpbr.diffuse_roughness, scene.capabilities
  if setting is False:
    return False
  rows = scene.tables.get("material.base_diffuse_roughness")
  mutable = bool(declared & {"instance.material", "instance.visible"}) and rows is not None and rows.shape[1] > 0
  return (
    setting is True
    or "diffuse_roughness" in free
    or sc.diffuse_roughness
    or sc.textures.diffuse_roughness
    or mutable
    or "material.base_diffuse_roughness" in declared
  )


def _specular_weight(scene: GpuScene, declared: frozenset[str]) -> bool:
  """Whether a material of the canonical specular law (no specular curve, no dielectric level) has
  a specular weight other than 1 or a zero IOR scale, or can get one per world (a declared
  specular or coat field)."""
  inputs = scene.derive_inputs
  canonical_specular = (inputs.shading & int(ShadingFlag.SPECULAR_CURVE | ShadingFlag.DIELECTRIC_SPECULAR_LEVEL)) == 0
  return (
    bool(np.any(canonical_specular & (inputs.values["material.specular_weight"] != 1.0)))
    or bool(np.any(canonical_specular & (inputs.ior_scale == 0.0)))
    or (
      bool(np.any(canonical_specular)) and bool(declared & {"material.specular_weight", "material.specular_ior", *COAT_FIELDS})
    )
  )


def _shadows(
  s: RenderSettings,
  scene: GpuScene,
  declared: frozenset[str],
  free: frozenset[str],
  *,
  casters: int,
  glossy: int,
) -> ShadowFeatures:
  """The shadow Features: samples, budget (none when it covers every potential caster:
  `shadow_casters`; resolved without the scene: the setting, which traces every caster when they
  fit), sun discs, translucent casters (`glossy`: the build's)."""
  sh = s.shadows
  budget = (sh.max_casters or 0) if "budget" in free else _cap(sh.max_casters, casters)
  shaded = s.outputs.shaded
  transmit = shadow_transmit(scene)
  switched = _switched_optics(scene, declared)
  see = bool(np.any(transmit > 0.0)) or bool(np.any(switched.see > 0.0) or np.any(switched.base[:, 3] < 1.0))
  translucent = (
    sh.translucent != "off"
    and sh.enabled
    and ("translucent_shadows" in free or see or bool(np.any(switched.color[:, 3] < 1.0)))
  )
  # Soft sun shadows (the disc of a directional light's angular radius): a fixed pattern like the
  # area shadows' (one fixed sample is the hard shadow: off).
  discs = "sun_disc" in free or scene.capabilities.sun_disc or "light.radius" in declared
  sun_disc = shaded and sh.enabled and discs and sh.samples > 1
  return {
    "shadows": sh.enabled,
    "shadow_samples": sh.samples,
    "budget": budget,
    "reflection_shadows": settings_fields.REFLECTION_SHADOWS[s.reflections.shadows] if glossy and sh.enabled else 0,
    "translucent_shadows": translucent,
    "tinted_shadows": bool(
      translucent
      and sh.translucent == "transmittance"
      and ("translucent_shadows" in free or np.any(np.ptp(transmit, axis=1) > _TINT_SPREAD))
    ),
    "sun_disc": sun_disc,
  }


def _surfaces(
  s: RenderSettings, scene: GpuScene, declared: frozenset[str], free: frozenset[str], *, alpha: bool
) -> SurfaceFeatures:
  """Culling, normals, texture slots, translucency, cut-outs (`alpha`: per world; `free`: the
  fields resolved without the scene, "textures": the base color slot, "maps": every other slot
  "auto" compiles, each read only where a material binds a texture)."""
  sc = scene.capabilities
  auto = s.surfaces.textures == "auto"
  prepared = _prepared(scene, declared)
  t = prepared.textures if prepared is not None else sc.textures
  if "textures" in free:
    t = dataclasses.replace(t, base_color=True)
  if "maps" in free:
    t = dataclasses.replace(t, normal=True, orm=True, emissive=True, opacity=True, diffuse_roughness=True)
  tinted = "material.emission_luminance" in declared and bool(np.any(scene.derive_inputs.base_lit))  # tint maps
  texture_rows = scene.tables.get("material.texture")
  prepared_diffuse = texture_rows is not None and texture_rows.shape[2] > int(TexRole.DIFFUSE_ROUGHNESS)
  declared_diffuse = "material.base_diffuse_roughness" in declared and bool(
    declared & {"material.texture", "material.tex_affine"}
  )
  diffuse = s.openpbr.diffuse_roughness is not False and (
    t.diffuse_roughness or declared_diffuse or (prepared_diffuse and bool(declared & {"instance.material", "instance.visible"}))
  )
  layered = t.layered or t.coat_normal or t.detail_normal
  on = (
    (t.base_color or t.normal or t.orm or t.emissive or t.opacity or tinted or diffuse or layered)
    if auto
    else bool(s.surfaces.textures)
  )
  # A texture slot is compiled in when textures are on and ("auto") the scene binds it.
  bound = (t.base_color, t.normal, t.orm, t.emissive or tinted, t.opacity) if auto else (True,) * 5
  base_color, normal, orm, emissive, opacity = (on and b for b in bound)
  auto_transparency = "transparency" in free or sc.has_alpha or (sc.alpha_map and on) or alpha
  if prepared is not None:
    auto_transparency |= prepared.has_alpha or (prepared.alpha_map and on)
  transparency = auto_transparency if s.transparency.enabled == "auto" else s.transparency.enabled
  culling = s.surfaces.backface_culling
  per_world_optics = world_optics.switched(scene, declared) is not None
  return {
    "backface_culling": culling,
    "two_sided": bool(
      culling and ("two_sided" in free or np.any(scene.instance_twosided[scene.rendered] & TWO_SIDED) or per_world_optics)
    ),
    "world_optics": per_world_optics,
    "vertex_normals": s.outputs.shaded,  # always on (no setting): `canonical` turns it off where unread
    "textures": on,
    "anisotropy": s.surfaces.anisotropy if on else 1,
    "base_color_map": base_color,
    "normal_map": normal,
    "orm_map": orm,
    "emissive_map": emissive,
    "opacity_map": opacity,
    "diffuse_roughness_map": on and diffuse,
    "transparency": transparency,
    "alpha_cutout": _cutouts(scene, declared, free) and opacity,
  }


def _cutouts(scene: GpuScene, declared: frozenset[str], free: frozenset[str]) -> bool:
  """Whether a material of the scene can be cut out (its alpha cutoff; with declared material
  selection a prepared one; a declared per-world cutoff; resolved without the scene: `free`)."""
  prepared = _prepared(scene, declared)
  sc = scene.capabilities
  return (
    "alpha_cutout" in free
    or sc.alpha_cutout
    or (prepared is not None and prepared.alpha_cutout)
    or "material.alpha_cutoff" in declared
  )


def _layers(
  scene: GpuScene, declared: frozenset[str], free: frozenset[str], *, shaded: bool, surfaces: SurfaceFeatures
) -> LayerFeatures:
  """The layered lobes' Features (`Features.layered`): what the rendered (or, with declared
  material selection, the prepared) materials use, or a declared per-world layer field; resolved
  without the scene, every lobe (`free` "layers") and every layered map (`free` "layer_maps"), each
  read per material."""
  sc = scene.capabilities
  prepared = _prepared(scene, declared)
  layers = sc.layers if prepared is None else sc.layers.union(prepared.layers)
  t = prepared.textures if prepared is not None else sc.textures
  if "layers" in free:
    layers = dataclasses.replace(layers, anisotropy=True, fuzz=True, coat=True, coat_own_normal=True)
  if "layer_maps" in free:
    t = dataclasses.replace(t, coat_normal=True, detail_normal=True, layered=True)
  maps = surfaces["textures"]
  coated = (
    "layers" in free
    or sc.clearcoat
    or (prepared is not None and prepared.clearcoat)
    or "instance.shading" in declared
    or bool(declared & set(COAT_FIELDS))
  )
  coat_map = shaded and coated and maps and t.coat_normal
  return {
    "specular_anisotropy": shaded and (layers.anisotropy or "material.specular_anisotropy" in declared),
    "fuzz": shaded and (layers.fuzz or bool(declared & {"material.fuzz", "material.fuzz_color"})),
    # A per-world coat weight on a material without a coat makes it an OpenPBR coat (COAT_GLTF:
    # glTF's law, from its layer flags).
    "coat_layer": shaded
    and coated
    and (layers.coat or bool(declared & {"material.coat_weight", "material.coat_color", "material.coat_ior"})),
    # The coat's normal differs from the base's: its own map, or a mapped base it does not share
    # (outside the "layers" class, a base the scene maps: the "maps" class compiling the normal map
    # in must not turn a scene's own-normal coat into a layered one).
    "coat_normal": coat_map
    or (shaded and coated and surfaces["normal_map"] and layers.coat_own_normal and ("layers" in free or t.normal)),
    "coat_normal_map": coat_map,
    "detail_normal": shaded and maps and t.detail_normal,
    "layered_maps": shaded and maps and t.layered,
    "horizon_fade": shaded and (surfaces["normal_map"] or (maps and t.detail_normal)),
  }


def _motion(s: RenderSettings, backend: type[TraceBackend], *, uniform: bool, motion: bool | None) -> bool:
  """`Features.motion`: the motion history (`motion`: a tier build's, None: the outputs').

  Raises:
    ValueError: for motion with a supersampling backend at spp > 1, or with per-camera resolutions
      (`uniform` False: the history is one image size).
  """
  motion = s.written_outputs().motion is not None if motion is None else motion
  if backend.supersamples and s.aa.spp > 1 and motion:
    raise ValueError(f'motion with traversal="{backend.name}" needs aa.spp=1 (it renders spp > 1 as one supersampled image)')
  if not uniform and motion:
    raise ValueError("motion needs every camera at the same resolution (CameraSpec.resolution)")
  return motion


def _spp(s: RenderSettings, *, motion: bool, supersamples: bool) -> int:
  """`Features.spp`: 1, or 2 for any n > 1 samples per axis where n is run-time data: a
  supersampling backend's one n x n image (`trace/optix/supersample.py`, `Lens.scale`) and the
  software path's per-frame sample loop (`Params.spp`); the software path's motion body compiles
  its n (its sample cone and grid)."""
  if supersamples or not motion:
    return min(s.aa.spp, 2)
  return s.aa.spp


def _params(s: RenderSettings, caps: RenderCaps) -> kernel_structs.Params:
  """The runtime parameters of the kernels (`Params`)."""
  scene = caps.scene
  p = kernel_structs.Params()
  p.headlight_color = wp.vec3(*scene.headlight.color)
  # Data, not a switch: 0 when off (the scene's or the settings'), so a build with the headlight
  # compiled in adds nothing (`tiers.SWITCHES`).
  lit = s.lighting.camera_light
  on = s.outputs.shaded and (scene.headlight.active if lit == "auto" else lit)
  p.headlight_intensity = float(scene.headlight.intensity) if on else 0.0
  p.ambient = wp.vec3(*(float(c) * float(s.lighting.fill) for c in s.lighting.fill_color))
  env = scene.environment
  p.inv_texel_angle = 1.0 / float(envmap.texel_angle(env.maps.width))
  p.background_map = env.background
  p.background_scale = float(env.background_intensity * env.maps.maps[env.background].scale) if env.background >= 0 else 0.0
  p.tone_mapper = settings_fields.TONE_MAPPERS[s.post.tone_mapper]
  r = s.reflections
  p.reflection_max_roughness = float(r.max_roughness)
  p.reflection_fade_roughness = 0.5 * float(r.max_roughness)
  p.reflection_min_weight = reflections.MIN_WEIGHT
  p.max_layers = s.transparency.max_layers
  p.spp = s.aa.spp
  # Material emission as radiance [nits] (host scenes) or display-relative (MuJoCo: divided by the
  # world's exposure): a run-time value, not a kernel key.
  p.relative_emission = int(not scene.capabilities.physical_emission)
  rgb = s.written_outputs().rgb
  p.skip_display = int(s.deferred_display() and (rgb is None or not rgb.alpha))
  return p
