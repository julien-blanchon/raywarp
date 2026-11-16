"""The fixed kernel tiers: the few render kernels a release prebuilds, and which one serves a
configuration.

A tier is the compile-time settings of the default (`RenderSettings()`), of the fast field set
(`FAST`) or of unlit shading, or one of their variants (`TIERS`, cheapest first): its kernels
compile the tier's effects plus the `SWITCHES` (effects each launch turns on or off:
`Features.switches`, `kernels/abi/toggles.py`), at their most capable setting. Reflections off is
run time in every build (`Params.rt.glossy`). So a settings change that keeps a tier's compile-time
settings, or only turns switches off, renders with that tier's kernels: nothing compiles, and a
prebuilt bundle (`core/build/kernel_cache/prebuilt.py`) holds them for every configuration reachable
from the tiers by run-time changes.

A tier build is resolved **scene-free** (`SCENE_FREE`): the Features fields that would come from
the scene (light count, light kinds, caps zeroed by the light count, ...) take the value that
serves every scene (counts read at run time, capabilities compiled in and selected per light /
material at run time), so one build per tier serves every scene and a new scene compiles nothing.

`route` picks the first tier whose build serves the settings' own scene-free Features
(`differences`: every field equal but switches turned off); a configuration no tier serves compiles
its own kernels, specialised on the scene (`Choice.outside` names what leaves the nearest tier).
The backend and the GPU architecture are axes of their own; the OptiX register cap is a module
option, per module from its word (`registers.optix_registers`). GI (`gi.enabled`) is one on the
software path (two forms of each tier, both in the bundle); an OptiX tier build compiles it in and
reads it from its word (`BOUND_FIELDS`). The outputs and the exposure mode are not: a build
compiles every output store and selects from the output word at run time
(`kernels/abi/output_layout.py`).
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from collections.abc import Callable
from typing import Final, NamedTuple, final

from raywarp._base.types import ShapeType
from raywarp.core.build.variants import class_sets
from raywarp.core.kernels.abi import toggles
from raywarp.core.kernels.abi.features import PLANES_NONE, RUNTIME_LIGHTS, SMALL_LIGHTS_MAX, Features, small_form
from raywarp.settings import FAST, RenderSettings

# The variants of the fast and default tiers: their compile-time settings with one structural
# change each, code that compiled into the tier would cost every configuration of it (CPU proxy,
# fast raygen SASS compiled in: 2 x 2 supersampling +10.6 % (the off-centre observation trace; on
# OptiX another output scale), shadow rays at reflection hits (the hits' directional rays), glossy
# reflections +58.5 %, LTC area lights +76.9 % (bound off in every tier, the area class still left
# +0.6 % and 8 B of stack: a 2-light robot scene); the default has reflections and LTC). Temporal
# state is allowed in every tier.
# A counted variant's settings (`COUNTED`): its tier's with what no earlier variant serves (its
# build's counts are the capacities whatever the values): more shadow samples, and a shadow budget
# above the lights shaded exactly (a budget at or above the light cap collapses, `canonical`).
_COUNTS_MARK: Final = {"shadows.samples": 4}
_DEFAULT_COUNTS: Final = {"reflections.shadows": "sun", **_COUNTS_MARK}
VARIANTS: Final = {
  # The unlit tier's: 2 x 2 supersampling and the run-time texture taps (each `SAMPLED`: a lit
  # configuration's anisotropic filtering under "unlit" keeps its taps).
  "unlit": {
    "ss2": {"aa.spp": 2},
    "sampled": {"surfaces.anisotropy": 2},
  },
  "fast": {
    "ss2": {"aa.spp": 2},
    "reflections": {"reflections.enabled": True},
    "ltc": {"lighting.area_lights": "ltc"},
    "sampled": {"shadows.samples": 2},
    "counts": {"shadows.max_casters": 16, **_COUNTS_MARK},
  },
  "default": {
    "ss2": {"aa.spp": 2},
    "sampled": {"shadows.samples": 2},
    "hit_sun": {"reflections.shadows": "sun"},
    # A shadow budget below the lights shaded exactly (the default's budget equals its cap: every
    # exact light traced, no budget loop): the budget loop at 3 casters, fewer read at run time.
    "casters": {"shadows.max_casters": 3},
    "counts": _DEFAULT_COUNTS,
    # The documented looks over the default (docs/settings.md "Recipes"): quality, reference and
    # the RTX look (2 x 2 supersampling with the counts).
    "ss2_counts": {"aa.spp": 2, **_DEFAULT_COUNTS},
  },
}
_BASE: Final = {
  # Shape and color only (`surfaces.shading="unlit"`: the base color as is, no lights, shadows,
  # environment, reflections or GI), on the fast field set (isotropic filtering). Translucent
  # layers follow the scene's surface class ("auto"), as in every other tier.
  "unlit": RenderSettings()
  .replace(FAST)
  .replace(
    surfaces={"shading": "unlit"},
    shadows={"enabled": False},
    reflections={"enabled": False},
    lighting={"light_grid": False},
    gi={"enabled": False},
  ),
  "fast": RenderSettings().replace(FAST),
  "default": RenderSettings(),
}
# The tiers, cheapest first (a configuration takes the first that serves it): the unlit tier, the
# fast and the default tier, each followed by its variants.
TIERS: Final = {
  name: settings
  for tier, base in _BASE.items()
  for name, settings in ((tier, base), *((f"{tier}_{v}", base.replace(c)) for v, c in VARIANTS[tier].items()))
}
# The tiers whose builds compile every class (`served`: on OptiX both code splits, on the software
# path every class) and the general light loop (`resolve._light_forms`) whatever the scene uses:
# the variants, configurations few scenes render, so the bundle holds one build of each per output
# form instead of one per class set and loop form; a scene pays what it does not use
# (`CLASS_COST`: an opaque scene the translucent surface's; a few lights the loop). Translucent
# layers turned off on a scene without any are moot there (`resolve._scene_moot`). The unlit
# tier's variants take the surface code split from the configuration instead (`SURFACE_SPLIT`: two
# builds per output form, the smallest builds): opaque unlit surfaces on a translucent scene.
SURFACE_SPLIT: Final = frozenset(f"unlit_{v}" for v in VARIANTS["unlit"])
COMPLETE: Final = frozenset(f"{t}_{v}" for t, vs in VARIANTS.items() for v in vs)
# The variants that read the shadows' and the textures' sampling at run time
# (`Features.run_sampling`; the anisotropic taps, `_ANISO`, as the shadows'): the tier's settings
# with 2 samples (a configuration the tier itself serves never reaches them: their fill points route
# here), the sample count, sun-disc pattern and area-light targets the launch's (`Toggles`) where
# the configuration's differ (`soft`: more samples). A variant, not a word bit of every tier:
# compiled in and folded off, the run-time paths still cost the tiers' raygens: soft shadows
# +0.7-1.0 % SASS (CPU proxy), Franka default +1.4 % on RTX; the taps a textured robot's default
# raygen +2.5 % SASS.
# The variants that read the light caps and the shadow budget at run time too
# (`Features.run_counts`: compiled at their capacities, `toggles.COUNTS_CAPACITY`): a lit
# configuration with other caps or budget (`lighting.max_exact`, `reflections.max_exact`,
# `shadows.max_casters`), samples, taps or reflection-hit shadows.
# Variants, not the tiers' own builds: compiled at their capacity, the budget's slots and the
# caps' ranking would cost every frame of the default.
COUNTED: Final = frozenset({"fast_counts", "default_counts", "default_ss2_counts"})
SAMPLED: Final = frozenset({"unlit_ss2", "unlit_sampled", "fast_sampled", "default_sampled"}) | COUNTED

# The compile-time settings a tier build compiles in and reads at run time, each its bit
# (`toggles.setting`: one value, the software path's `Toggles.flags` switch and an OptiX module's
# word bit, second half, folded per module so that the tier's value costs nothing): shadows (off:
# every shadow ray unoccluded, `shadows.make_shadow_ray`; the loops keep their form, their other
# fields moot, `_MOOT_WITHOUT_SHADOWS`), the directional lights' shadow rays at reflection hits
# (off: the hit loop traces none, `light_loop._make_light_term`), back-face culling (the OptiX ray
# flags and mask, the planes' and the software walk's test), the meshes' interpolated vertex
# normals, specular anti-aliasing (the curvature and Toksvig roughness widening), the fog (compiled
# into every tier build, `COMPILED_ON` in `tier_free`: +0.8 % SASS of a fast raygen with distance
# fog), the translucent layers (off: every surface opaque, one per camera ray, as a build without
# them; the transmissive and solid-glass fields they carry moot, `_MOOT_WITHOUT_TRANSPARENCY`; a
# translucent scene with `transparency.enabled=False` keeps its tier build). As plain run-time
# switches on OptiX culling, vertex normals and specular AA cost a fast raygen +0.25-1 % SASS (CPU
# proxy, a legged robot).
SETTING_FIELDS: Final = {
  "shadows": toggles.SHADOWS,
  "reflection_shadows": toggles.REFLECTION_SHADOWS,
  "fog": toggles.FOG,
  "backface_culling": toggles.BACKFACE_CULLING,
  "transparency": toggles.TRANSPARENCY,
}
_SETTING_BITS: Final = {name: toggles.high(bit) for name, bit in SETTING_FIELDS.items()}
# The shadows' sampling (samples per light, the sun disc): a tier build compiles its own and the
# run-time sampling, the latter on where the configuration samples otherwise (`soft`: other samples,
# a sun disc), `toggles.SOFT_SHADOWS` (a word bit on OptiX, a switch on the software path). Compiled
# in unbound, a second shadow sample cost a fast raygen +10 % SASS (CPU proxy).
_SOFT: Final = toggles.high(toggles.SOFT_SHADOWS)
_SOFT_FIELDS: Final = ("shadow_samples", "sun_disc")
# The anisotropic taps: a `SAMPLED` variant with textures compiles its own count and the run-time
# one (`MaterialTables.texture_taps`, up to `toggles.TAPS_CAPACITY`), the latter on where the
# configuration's count differs (`aniso`), `toggles.ANISOTROPY` (as `_SOFT`).
_ANISO: Final = toggles.high(toggles.ANISOTROPY)
# The Features fields a tier compiles in and each launch switches (`Toggles.flags` bit): the
# environment's background (a scene without maps, or the settings' off, turns it off), translucent
# shadow casters, the area-light panels' emission skipped at reflection hits
# (`lighting.emissive_lights`: a test inside the emitting-hit branch alone) and, on the software
# path, `SETTING_FIELDS`. GI is not a switch (as one both ambient paths stayed compiled in: a robot
# scene's bits build: +1.4 k raygen instructions, its stack 192 -> 256 B, +3 % frame time on RTX)
# but a word bit on OptiX (`BOUND_FIELDS`, the GI-off module folds it) and a settings axis on the
# software path (`_OWN` "gi.enabled": a GI and a GI-off build, both prebuilt). The camera headlight
# is data: its intensity is 0 when off (`resolve._params`), the light loops' entry count reads it.
SWITCHES: Final = {
  "env_background": toggles.ENV_BACKGROUND,
  "translucent_shadows": toggles.TRANSLUCENT_SHADOWS,
  "light_panels": toggles.LIGHT_PANELS,
  **SETTING_FIELDS,
}
# The Features fields a tier build compiles in and reads from its build word (`toggles.SETTING_*`,
# `SCENE_*`): the module of a configuration or scene that has one off folds its code (`word`). The
# environment's lighting (a skybox-only scene, or the settings' off: as run-time switches their code
# stayed live, +15.6 % raygen instructions of a fast build), GI (an OptiX scene-free build compiles
# it both ways, `BOUND_FREE`) and the scene properties of `toggles.SCENE_ALL` (visible area lights,
# spot cones, two-sided instances, the specular curve, the horizon fade).
BOUND_FIELDS: Final = {
  "gi": toggles.SETTING_GI,
  "env_lit": toggles.SETTING_ENV_LIT,
  "visible_lights": toggles.SCENE_VISIBLE_LIGHTS,
  "has_spot": toggles.SCENE_SPOT,
  "two_sided": toggles.SCENE_TWO_SIDED,
  "specular_curve": toggles.SCENE_SPECULAR_CURVE,
  "horizon_fade": toggles.SCENE_HORIZON_FADE,
}
# The fields inside the bound classes (`CLASSES`) an OptiX tier build compiles in and reads from the
# second half of its word (`toggles.high` bits), one bit per field: a scene that uses one field of a
# class folds the class's others. CPU proxy (fast raygen SASS of a 16-rect-light room with a few
# maps, materials and layers: the maps fields compiled in unused +7.2 %, the materials and layers
# fields +10.6 %). The software path keeps them run time (its word is a module literal: a module per
# word). The live material UV mappings (`live_material_uv`, `BOUND_FREE`) have their bit too: off, a
# mesh reads its instance's own UVs.
FIELD_BITS: Final = {
  "reflection": toggles.high(toggles.MAT_REFLECTION),
  "diffuse_roughness": toggles.high(toggles.MAT_DIFFUSE_ROUGHNESS),
  "specular_weight": toggles.high(toggles.MAT_SPECULAR_WEIGHT),
  "vertex_colors": toggles.high(toggles.MAT_VERTEX_COLORS),
  "unlit_emission": toggles.high(toggles.MAT_UNLIT),
  "normal_map": toggles.high(toggles.MAP_NORMAL),
  "orm_map": toggles.high(toggles.MAP_ORM),
  "emissive_map": toggles.high(toggles.MAP_EMISSIVE),
  "opacity_map": toggles.high(toggles.MAP_OPACITY),
  "diffuse_roughness_map": toggles.high(toggles.MAP_DIFFUSE_ROUGHNESS),
  "uv_maps": toggles.high(toggles.MAP_UV_SETS),
  "world_uv": toggles.high(toggles.MAP_WORLD_UV),
  "live_material_uv": toggles.high(toggles.MAP_LIVE_UV),
  "specular_anisotropy": toggles.high(toggles.LAYER_ANISOTROPY),
  "fuzz": toggles.high(toggles.LAYER_FUZZ),
  # Inside the surface split: a translucent scene without transmissive materials folds them (RTX,
  # glass fast: -3.0 %).
  "transmissive": toggles.high(toggles.SURF_TRANSMISSIVE),
  # Inside the glass class: solid glass on spheres or boxes (an OptiX glass scene without them
  # folds the analytic exit check: a feature-dense room's fast +14 % on RTX).
  "glass_shapes": toggles.high(toggles.GLASS_SHAPES),
}


# The scene-derived Features fields (and field groups: "textures", "layers") a tier build resolves
# without the scene (`resolve._resolve` free): the light count (RUNTIME_LIGHTS: the loops' bound at
# run time), the light caps and shadow budget at the settings' values (a cap at or above the scene's
# lights or casters selects them all), and the light kinds (spot cones, IES profiles, LTC area
# lights, sun discs), each selected per light at run time; the material and mesh capabilities below,
# each selected per material / instance at run time. A lab switch: benchmarks narrow it to measure
# one field's cost against the specialised build.
SCENE_FREE: frozenset[str] = frozenset({
  "nlight",
  "light_cap",
  "hit_light_cap",
  "budget",
  "has_spot",
  "ies",
  "area_lights",
  "sun_disc",
  # The environment's light (a word bit, `BOUND_FIELDS`) and background (a run-time switch).
  "env_lit",
  "env_background",
  # The camera headlight (MuJoCo's), one more light-loop entry, switched at run time.
  "headlight",
  # Shadow rays through translucent casters (RGB with "transmittance"), switched at run time by
  # the scene's casters; an opaque scene's rays take the any-hit query.
  "translucent_shadows",
  # The materials' texture slots and bits, each read where a material uses it (a texture id, a
  # shading flag, a per-material row whose neutral value is exact): "textures" the base color
  # slot, "maps" every other slot (normal, ORM, emission, opacity, EON roughness).
  "textures",
  "maps",
  # The layered lobes' maps (coat normal, detail normal, anisotropy and fuzz maps).
  "layer_maps",
  "scaled_normal_map",
  "uv_maps",
  "world_uv",
  "reflection",
  "diffuse_roughness",
  "specular_weight",
  "specular_curve",
  "vertex_colors",
  "unlit_emission",
  "light_panels",
  # Camera rays test the area lights without geometry out of line, over `Lights.nvisible`.
  "visible_lights",
  # Every shape the backend traces (an instance's type selects its code at run time; the
  # "primitives" class) and the launch shape of scenes with meshes (block size, ordered traversal).
  "shape_types",
  # Translucent layers (the layer loop's bound and each surface's alpha at run time), alpha cut-outs
  # (the cut-out instances' test), transmissive glass, the clear coat and the layered lobes
  # ("layers": anisotropy, fuzz, the OpenPBR coat, coat / detail normal maps), each read per
  # material. Translucent shadow casters stay the scene's (they change every shadow query).
  "transparency",
  "alpha_cutout",
  "transmissive",
  "clearcoat",
  "layers",
  # Solid glass: refraction and Beer-Lambert absorption, read per instance (its medium row), and
  # the analytic exit of glass spheres and boxes (per instance type).
  "refraction",
  "glass_shapes",
  # Two-sided mesh instances (a per-instance table).
  "two_sided",
})  # fmt: skip


# The feature classes (zero recompile at the speed of the specialised kernels): groups of SCENE_FREE
# fields a tier build compiles all together, or none of when the scene uses none of them
# (`class_free`). Each class is a prebuilt variant of every tier (the bundle holds them all, as it
# holds both backends): a scene picks its classes from its data, no scene count, id or table enters
# a key, and a scene change never compiles.
# - surface: translucent layers, alpha cut-outs, transmissive glass, translucent shadow casters
#   (fast, Veach MIS, sm_120 ptxas spill stores: every path 8944 B, without these 956 B).
# - coat / layers: the clear coat; the layered lobes (anisotropy, fuzz, the OpenPBR coat, coat /
#   detail normal maps). Together ("lobes"), in a translucent build without them a large office
#   interior's default OptiX raygen continuation stack is 224 B smaller (1440 -> 1216 B; its own
#   build 832 B), its software spills 21.7 -> 13.5 KB with the shapes (own 12.0 KB). Apart:
#   a clear-coated interior without layered lobes compiled them all, the layered lobes the largest
#   single group of its bundle build (RTX: 76.8 ms, without them 45.9).
# - glass: refraction and Beer-Lambert absorption of solid transmissive instances (any rendered
#   instance with a medium row: an IOR or a transmission depth; `pipeline/upload/optics.py`); in the
#   layer loop of every translucent scene they cost a feature-dense room's fast +28 % on RTX
#   (OptiX raygen continuation stack 640 -> 736 B). Solid glass is transmissive: a glass build is a
#   surface build too.
# - headlight: the camera headlight's entry of the general light loops (compiled in unused it cost
#   a translucent room 6 % on RTX; the small form's straight-line entry is free).
# - maps: the texture slots but base color (normal, ORM, emission, opacity, EON roughness), UV
#   sets, world UV, scaled normal maps; materials: EON, specular weight, vertex colors, mirror
#   reflectance, unlit materials (the specular curve stays in every build: 0 raygen PTX unused on
#   a scene whose own build uses it alone and so compiled the whole class: EON 573, vertex colors
#   157, specular weight 124, reflection 46 PTX). Compiled in unused (RTX 512 worlds, fast bundle
#   vs the bundle with the group from the scene): maps 9-28 %, materials 4-12 % on an interior and
#   two robot scenes. (Transmissive apart from the surface class, -1 % on a large interior, and
#   the visible-light test, 0, stay in.)
# - layer_maps / base_map / env: the layered lobes' maps (coat normal, detail normal, anisotropy,
#   fuzz) apart from the lobes; the base color texture role; the environment's light (a word bit,
#   `BOUND_FIELDS`) and background (a run-time switch) within the class: bound off, the light
#   folds (+0.4 % SASS over the class compiled out, CPU proxy), the background code stays (+8 %),
#   so a scene without either keeps its env-free part. Compiled in unused (fast, RTX 512
#   worlds, bundle vs the bundle with the group from the scene): base color map -4.3 % on a legged
#   robot, -2.7 % on a clear-coated interior; environment -4.3 % on the robot; the layered maps
#   1.3 k of a feature-dense room's 3.2 k raygen PTX over its own build (CPU).
# - primitives: the analytic shapes (boxes, spheres, capsules, ellipsoids, cylinders) beside the
#   triangle kinds every build traces (`geometry_kinds`): their OptiX traversal (custom primitives
#   and curves in the pipeline flags, the box hit's normal and distance at each closest-hit trace)
#   and their texture coordinates. Without it a robot scene's bits build ran at its own build's
#   speed (RTX, 512 worlds: an arm 28.32 -> 27.92 ms, own 27.81; raygen 11 860 -> 10 548
#   instructions, continuation stack 256 -> 192 B). On OptiX a bound class: one word bit per
#   shape type (`toggles.SHAPE_BITS`; compiled in with every bit off: the shape-free build's SASS,
#   CPU proxy). Its analytic planes (`PLANE_FORM`, the
#   scene's value like the light-loop form): none (no plane code) or some (a run-time loop over
#   the plane table); the plane ids are run-time data. Compiled into every build, the plane code
#   cost two large plane-free interiors 15-24 % (RTX, fast, bundle builds).
# - ies: the IES profiles of the light loops (rare in scenes). Compiled in unused (RTX, 512 worlds,
#   fast, the bundle with the IES profiles from the scene): a clear-coated interior
#   36.77 -> 31.00 ms (its own build's speed, the whole of its bundle gap), a robot scene -2.7 %.
# - area: the area lights' shading and shadow targets and the glossy light panels (the default
#   and its variants; fast shades lights as points). Compiled in unused (RTX, 512 worlds):
#   a robot scene's default bundle build 1.39x its own build, mostly from this group.
# So up to 3 x 2^12 class combinations per tier (opaque / translucent / glass, each other class on
# or off; the layered maps only with the coat or the layers), times the light-loop and plane forms:
# the bundle holds what `tools/prebuilt/fill.py` enumerates.
CLASSES: Final = {
  "surface": ("transparency", "alpha_cutout", "transmissive", "translucent_shadows"),
  "coat": ("clearcoat",),
  "layers": ("layers",),
  "layer_maps": ("layer_maps",),
  "base_map": ("textures",),
  "maps": ("maps", "scaled_normal_map", "uv_maps", "world_uv"),
  "materials": ("reflection", "diffuse_roughness", "specular_weight", "vertex_colors", "unlit_emission"),
  "glass": ("refraction", "glass_shapes"),
  "env": ("env_lit", "env_background"),
  "headlight": ("headlight",),
  "primitives": ("shape_types",),
  "ies": ("ies",),
  "area": ("area_lights", "light_panels"),
}

# The analytic shape types of the "primitives" class: what a triangle-kinds build leaves out.
PRIMITIVE_TYPES: Final = frozenset(
  int(t) for t in (ShapeType.BOX, ShapeType.SPHERE, ShapeType.CAPSULE, ShapeType.ELLIPSOID, ShapeType.CYLINDER)
)


def geometry_kinds(
  types: tuple[int, ...], backend_types: tuple[int, ...], *, every: bool, flex: bool, inlined: bool
) -> tuple[int, ...]:
  """The shape types a tier build of a scene with shape types `types` compiles: the backend's
  (`backend_types`), less the primitives (`PRIMITIVE_TYPES`) unless the scene has one or `every`
  (the "primitives" class compiled in), plus the deformables when the scene has them or `flex` (an
  OptiX build: bound, `BOUND_FREE`; bound off they still cost a robot scene's fast raygen
  +3.1 % SASS by the CPU proxy, but as a scene axis they took the census from 106 to 262 builds).
  Meshes are in every build, planes, height fields and SDFs too where the backend traces them apart
  from its per-type code (OptiX: fast raygen PTX with textures: meshes alone 5 448 instructions,
  with planes, height fields and SDFs 5 682, with the primitives 6 838; the CPU proxy: 0 SASS); with
  `inlined` (the software traversal, whose every trace site inlines each kind's intersection: an
  interior's tier megakernel 3.1x its own build's SASS with every kind, +5.8 % with the scene's)
  only where the scene has them."""
  primitives = every or bool(PRIMITIVE_TYPES & set(types))
  deformables = (int(ShapeType.FLEX),) if flex or int(ShapeType.FLEX) in types else ()
  scene_kinds = _INLINED_KINDS if inlined else frozenset()
  kept = (t for t in backend_types if (primitives or t not in PRIMITIVE_TYPES) and (t in types or t not in scene_kinds))
  return (*kept, *deformables)


# The mesh-like kinds a software tier build takes from the scene (`geometry_kinds`).
_INLINED_KINDS: Final = frozenset({int(ShapeType.PLANE), int(ShapeType.HFIELD), int(ShapeType.SDF)})


def _uses(f: Features, name: str) -> bool:
  """Whether `f` compiles class field `name` ("layers": any layered lobe; "shape_types": a
  primitive)."""
  if name == "layers":
    return f.layered
  if name == "shape_types":
    return bool(PRIMITIVE_TYPES & set(f.shape_types))
  if name == "maps":
    return f.normal_map or f.orm_map or f.emissive_map or f.opacity_map or f.diffuse_roughness_map
  if name == "layer_maps":
    return f.coat_normal_map or f.detail_normal or f.layered_maps
  if name == "textures":
    return f.base_color_map
  return bool(getattr(f, name))


# What compiling a class unused costs, in percent of a frame (RTX, 512 worlds, fast, the
# measurements beside `CLASSES`): the price `cover` puts on a superset's extra classes.
CLASS_COST: Final = {
  "surface": 20,
  "coat": 5,
  "layers": 30,
  "layer_maps": 4,
  "base_map": 3,
  "maps": 15,
  "materials": 8,
  "glass": 25,
  "env": 4,
  "headlight": 6,
  "primitives": 2,
  "ies": 12,
  "area": 30,
}


def cover(used: frozenset[str], *, visibility: bool = False, flex: bool = False, aliased: bool = False) -> frozenset[str]:
  """The prebuilt class set (`class_sets.PREBUILT`, and every class: always prebuilt) serving a
  scene that uses the classes `used`: the cheapest superset (`cover_cost`; ties: fewer classes,
  then by name). Any scene is served (every class is a superset of all), so a scene never
  compiles; a combination the bundle lacks pays the classes of its superset it does not use. A
  scene with per-world visibility (`visibility`), deformables (`flex`) or aliased poses
  (`aliased`: shared instances without compact poses, MuJoCo) takes only the sets prebuilt with
  them as well (`class_sets.VISIBLE`, `FLEX`, `ALIASED`)."""
  every = frozenset(CLASSES)
  sets = [
    s
    for s in (*class_sets.PREBUILT, every)
    if used <= s <= every
    and (
      s == every
      or (
        (not visibility or s in class_sets.VISIBLE)
        and (not flex or s in class_sets.FLEX)
        and (not aliased or s in class_sets.ALIASED)
      )
    )
  ]
  return min(sets, key=lambda s: (cover_cost(used, s), len(s), sorted(s)))


def cover_cost(used: frozenset[str], superset: frozenset[str]) -> int:
  """What serving a scene that uses the classes `used` with the build of class set `superset`
  costs: the `CLASS_COST` of the classes it adds."""
  return sum(CLASS_COST[c] for c in superset - used)


def small_loop(light_forms: int) -> bool:
  """Whether a build of light-loop form `light_forms` takes the small loop, whose straight-line
  headlight entry costs nothing (so the "headlight" class is counted used: one build per small
  form instead of two)."""
  return 0 < light_forms <= SMALL_LIGHTS_MAX


def used_classes(own: Features) -> frozenset[str]:
  """The classes `own` (the settings' Features specialised on the scene) uses; with the headlight
  in a small-loop build (`small_loop`)."""
  small = small_loop(own.light_forms)
  return frozenset(n for n, fields in CLASSES.items() if (n == "headlight" and small) or any(_uses(own, f) for f in fields))


@final
class Axes(NamedTuple):
  """The scene axes a tier build takes from the configuration it serves (`axes`), not from the
  tier's own settings: a settings change that only moves the scene onto another prebuilt variant
  (transparency on: the translucent class; the light grid on or off: the light loop's form with or
  without the grid, `LOOP_FORM`) lands on that variant instead of leaving the tiers.

  Attributes:
    classes: the classes the configuration uses (`used_classes`), served beside the tier's own.
    grid: whether the configuration builds the light grid for the scene (`Features.culling` of
      its Features specialised on the scene): the tier's light-loop form follows it.
    motion: whether it keeps the motion history (`Features.motion`: a motion output; every tier is
      filled in both output forms, with and without it).
    complete: every class served and the general light loop (a `COMPLETE` tier's build, and the
      configuration resolved to compare with it).
    sampled: the shadows' sampling read at run time (a `SAMPLED` tier's).
    counted: the light caps and the shadow budget read at run time (a `COUNTED` tier's).
    split: a complete tier's surface code split the configuration's (a `SURFACE_SPLIT` tier's).
  """

  classes: frozenset[str]
  grid: bool
  motion: bool
  complete: bool = False
  sampled: bool = False
  counted: bool = False
  split: bool = False


def axes(cfg: Features, tier: str | None = None) -> Axes:
  """The scene axes of configuration `cfg` (its Features specialised on the scene) for tier `tier`
  (`COMPLETE`, `SAMPLED`, `COUNTED`, `SURFACE_SPLIT`: its every class, run-time sampling, run-time
  counts, the configuration's surface split)."""
  flags = (tier in COMPLETE, tier in SAMPLED, tier in COUNTED, tier in SURFACE_SPLIT)
  return Axes(used_classes(cfg), cfg.culling, cfg.motion, *flags)


# The classes of the build word (`toggles.CLASS_BITS`, and the primitives' shape bits,
# `toggles.SHAPE_BITS`: one per analytic shape type): a tier build compiles them all and reads them
# from its word (`word`; OptiX: bound per module, software: its megakernel modules' literal), so
# they are no routing axis. The others stay code splits, each a prebuilt variant: "surface" (the
# program set: cut-out any-hit programs and micromaps, the translucent shadow trace, the layer loop;
# its opaque build is half the translucent raygen), "env" (its background is a `Toggles.flags`
# switch).
CLASS_MASKS: Final = {**toggles.CLASS_BITS, "primitives": toggles.SHAPE_ALL}
BOUND_CLASSES: Final = frozenset(CLASSES) & frozenset(CLASS_MASKS)


def bound_classes(traversal: str) -> frozenset[str]:
  """The bound classes of a tier build on backend `traversal`: `BOUND_CLASSES`, but on the
  software path the primitives stay a code split (its traversal inlines each shape type's
  intersection at every trace site, which a word bit would leave in: the shape bits gate the
  texture mapping and the OptiX box code only)."""
  return BOUND_CLASSES if traversal == "optix" else BOUND_CLASSES - {"primitives"}


SPLIT_CLASSES: Final = frozenset(CLASSES) - BOUND_CLASSES


def served(own: Features, of: Axes | None = None) -> frozenset[str]:
  """The classes the tier build serving `own` (the settings' Features specialised on the scene)
  compiles on: on OptiX every bound class (its module binds `own`'s) and the cheapest prebuilt
  code-split part over `own`'s (`_split_cover`); on the software path the prebuilt set (`cover` of
  `used_classes(own)`, its visibility, deformables and pose aliasing), whose bound classes its
  word turns on. `of`: the configuration's axes, its classes served too (`Axes`; complete: every
  class)."""
  # A tier build serving a configuration compiles the configuration's classes (`of`), not those
  # the tier's own settings would use on the scene (transparency off on a translucent scene).
  used = of.classes if of is not None else used_classes(own)
  if of is not None and of.complete:
    if of.split and own.traversal == "optix":
      return BOUND_CLASSES | SPLIT_CLASSES - {"surface"} | _split_cover(used & SPLIT_CLASSES)
    return BOUND_CLASSES | SPLIT_CLASSES if own.traversal == "optix" else frozenset(CLASSES)
  if own.traversal == "optix":
    return _split_cover(used & SPLIT_CLASSES) | BOUND_CLASSES
  flex = int(ShapeType.FLEX) in own.shape_types
  return cover(used, visibility=own.visibility, flex=flex, aliased=own.shared and not own.compact_poses)


def _split_cover(used: frozenset[str]) -> frozenset[str]:
  """The cheapest code-split part of a prebuilt set (`class_sets.PREBUILT`; every split class
  always) holding the split classes `used`: an OptiX tier build's (visibility, pose aliasing and
  deformables are bound there)."""
  parts = {s & SPLIT_CLASSES for s in class_sets.PREBUILT} | {SPLIT_CLASSES}
  return min((s for s in parts if used <= s), key=lambda s: (cover_cost(used, s), len(s), sorted(s)))


def class_free(own: Features, of: Axes | None = None) -> frozenset[str]:
  """SCENE_FREE without the fields of every class neither bound nor `served(own, of)`: those
  resolve to the scene's values (none of them: the scene does not use them); the others compile
  in."""
  compiled = served(own, of) | bound_classes(own.traversal)
  return SCENE_FREE - {f for name, fields in CLASSES.items() if name not in compiled for f in fields}


def unserved_fields(own: Features, of: Axes | None = None) -> tuple[str, ...]:
  """The fields of the classes the tier build serving `own` compiles neither bound nor served
  (`class_free`'s complement) that are flags: off in the build (with `tinted_shadows`, which needs
  translucent casters)."""
  compiled = served(own, of) | bound_classes(own.traversal)
  fields = [f for name, group in CLASSES.items() if name not in compiled for f in group if isinstance(getattr(own, f), bool)]
  return (*fields, "tinted_shadows") if "translucent_shadows" in fields else tuple(fields)


# The scene-derived Features fields an OptiX tier build resolves without the scene and reads from
# the build word (`kernels/abi/toggles.GEO_*`, bound per module: OptiX folds the parts the scene
# does not use, `word`, `trace/optix/device/programs.build_word`): the analytic planes' loop
# (PLANES_SOME), both structures (per-world and shared), compact pose indexing (a non-compact
# layout's slots are the identity: `sharing._pose_columns`), the per-world visibility table, the
# light grid's lookup of the general loop (`toggles.CLASS_CULLING`; the loop form stays the
# scene's), the deformables ("flex": their shape type, `toggles.FLEX`; their materials,
# `FLEX_MATERIALS`), the GI setting (the probes' ambient terms compiled in, `toggles.SETTING_GI`;
# the GI programs a module of their own) and the live material UV mappings (declared material
# selection with live UV tables, MuJoCo: `toggles.MAP_LIVE_UV`; the shaded build's material carrier
# holds their tables in every build). The software backend keeps them as axes (`SCENE_AXES`,
# `LOOP_FORM`, GI a settings axis; the live UV mappings the scene's: its word is a module literal, a
# bit there would be a module).
BOUND_FREE: Final = frozenset({
  "planes",
  "shared",
  "world_tree",
  "compact_poses",
  "visibility",
  "culling",
  "flex",
  "flex_materials",
  "gi",
  "live_material_uv",
})  # fmt: skip


def tier_free(own: Features, of: Axes | None = None) -> frozenset[str]:
  """The fields the tier build serving `own` (the settings' Features specialised on the scene)
  resolves without the scene: `class_free`'s, and on OptiX `BOUND_FREE` (`of`: the configuration's
  axes)."""
  return class_free(own, of) | COMPILED_ON | (BOUND_FREE if own.traversal == "optix" else frozenset())


# The light loop's form, the scene's (one of three values, all prebuilt): the small table's
# straight-line loop (the lights fit min(4, budget, cap) with no grid), the general loop, or the
# general loop with the light grid's lookup (the settings allow a grid and the scene has enough
# lights; on OptiX a bound bit of the general loop, `BOUND_FREE`). Compiled in unused, the grid
# lookup cost a translucent room 11 % on RTX and both forms 20-32 %, so neither is scene-free; the
# small vs general form stays a code split (one PTX with both loops schedules the raygen with both
# live; moving the hot shading between a straight-line and a loop form changed RTX time 2-4.7x
# where no CPU metric predicted it), as do the small form's entry counts (bound, +0.4-0.95 % SASS).
LOOP_FORM: Final = ("light_forms", "culling")
# The "primitives" class's plane form (`Features.planes`: PLANES_NONE / SOME), the scene's.
PLANE_FORM: Final = ("planes",)


def feature_class(build: Features) -> tuple[bool | int, ...]:
  """The feature classes a build compiles (one flag per `CLASSES` entry), then its light loop's
  form (`LOOP_FORM` values) and its planes' (`PLANE_FORM`)."""
  classes = tuple(any(_uses(build, name) for name in fields) for fields in CLASSES.values())
  return (*classes, *(getattr(build, name) for name in (*LOOP_FORM, *PLANE_FORM)))


# The scene-derived Features fields a software tier build still takes from the scene (OptiX:
# `BOUND_FREE` binds the first four): prebuilt axes of the bundle, like the classes (a scene picks
# its variant; a scene change never compiles). Kept as axes on measurement, not run time: the
# shared-scenery layout (which structures a ray traverses: shared-only / world-only / both;
# compiling both for every scene costs a shared-only scene, 33 of 35 registry scenes, +26 % fast
# OptiX raygen instructions and 3x its trace sites), the pose indexing (`compact_poses`: +2.1 %
# instructions), per-world visibility, the software BLAS frames, deformables and the trace backend;
# and the scene's planes, height fields and SDFs among its shape types (`geometry_kinds` `inlined`;
# the primitives: their class).
SCENE_AXES: Final = (
  "shared",
  "world_tree",
  "compact_poses",
  "visibility",
  "mesh_frames",
  "flex_materials",
  "traversal",
)

# Fields a tier reads from the user's settings, never the tier's: the outputs (structure of the
# buffers), the trace backend (an axis of its own), GI and the kernel build choice.
_OWN: Final = ("outputs.", "performance.traversal", "performance.kernels", "gi.enabled")
# Features fields no kernel of the effective configuration reads when a switch is off: reflection
# hits without reflection rays.
_MOOT_WITHOUT_GLOSSY: Final = ("hit_light_cap", "reflection_shadows", "light_panels")
# The shadow fields a configuration without shadows collapses (`resolve.canonical`) and its light
# loop's form: unread by a build whose shadows its word turns off (`SETTING_FIELDS`).
_MOOT_WITHOUT_SHADOWS: Final = (
  "shadow_samples",
  "budget",
  "light_forms",
  "nlight",
  "reflection_shadows",
  "tinted_shadows",
  "sun_disc",
)
# The fields the translucent layers carry (`pixel.make_trace_pixel`: transmissive compositing and
# solid glass only through them): unread by a build whose layers its word or switch turns off.
_MOOT_WITHOUT_TRANSPARENCY: Final = ("transmissive", "refraction", "glass_shapes")
# Not compared: the build's switch and bound bits and its word (properties of the build).
_UNCOMPARED: Final = ("switches", "bound", "word")
# The light caps: a cap at or above the scene's light count shades every light exactly.
_CAPS: Final = ("light_cap", "hit_light_cap")
# The counts a tier build reads at run time within its compiled value (`toggles.RUN_COUNTS`: the
# camera rays' shadow budget, the light caps, the anisotropic texture taps; `run_flags` turns it
# on): a configuration's lower one (above 0) is served.
_RUN_COUNTS: Final = ("budget", *_CAPS, "anisotropy")


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Choice:
  """The kernels serving one configuration.

  Attributes:
    tier: the tier (a `TIERS` name), None outside every tier (own kernels, compiled).
    build: what the kernels compile (the tier's, with its `switches`; outside: the effective
      Features).
    off: the switches the configuration turns off at run time (Features field names; "glossy":
      reflections).
    outside: outside every tier, the Features fields that differ from the nearest tier's (empty
      in a tier).
    nearest: outside every tier, the tier differing in the fewest fields.
  """

  tier: str | None
  build: Features
  off: tuple[str, ...] = ()
  outside: tuple[str, ...] = ()
  nearest: str | None = None


def tier_settings(s: RenderSettings, tier: str) -> RenderSettings:
  """`s` with the compile-time settings of `tier` (its base settings', the switches at their
  most capable); the outputs, the backend, GI on or off and the run-time settings stay `s`'s."""
  return s.replace(_compiled(tier))


# Features every tier compiles on whatever its settings say, off at run time (`SETTING_FIELDS`,
# `SWITCHES`): the fog (derived from its run-time values, `settings.Atmosphere.fog`).
COMPILED_ON: Final = frozenset({"fog"})


@functools.cache
def _compiled(tier: str) -> dict[str, object]:
  """The compile-time settings of `tier`'s base settings (but `_OWN`), "group.field" -> value."""
  base = TIERS[tier].flat()
  return {f.name: base[f.name] for f in RenderSettings.fields_table() if not f.runtime and not f.name.startswith(_OWN)}


def differences(
  build: Features, f: Features, scene: Features | None = None, *, unread: frozenset[str] = frozenset()
) -> tuple[str, ...]:
  """The Features fields where `build` cannot render `f` (both scene-free, or both the scene's):
  every field equal, except a switch `build` compiles in and `f` turns off, reflections `f` turns
  off, the fields `f`'s kernels would not read (`_MOOT_*`: without reflections, shadows or
  translucent layers; `unread`: those no frame on the scene reads, `resolve._scene_moot`), and a
  light cap of `build` that every light of the scene fits under while `f`'s does too (`scene`: the
  configuration's Features specialised on the scene, its cap 0 then)."""
  moot = {*_UNCOMPARED, *unread}
  if not f.glossy:
    moot.update(_MOOT_WITHOUT_GLOSSY)
  if not f.shadows and build.shadows and (build.switches & toggles.SHADOWS or _bound_off(build, "shadows")):
    moot.update(_MOOT_WITHOUT_SHADOWS)
  if not f.transparency and build.transparency and (build.switches & toggles.TRANSPARENCY or _bound_off(build, "transparency")):
    moot.update(_MOOT_WITHOUT_TRANSPARENCY)
  if f.shadows and (build.switches & toggles.SOFT_SHADOWS or build.bound & _SOFT):
    moot.update(_SOFT_FIELDS)
  if build.switches & toggles.ANISOTROPY or build.bound & _ANISO or not f.textures:
    moot.add("anisotropy")
  if not f.normal_map and _bound_off(build, "normal_map"):
    moot.add("scaled_normal_map")  # the normal map's decode, folded with it
  if not f.translucent_shadows and build.switches & toggles.TRANSLUCENT_SHADOWS:
    moot.add("tinted_shadows")  # the translucent casters' transmittance, switched off with them
  out: list[str] = []
  for field in dataclasses.fields(Features):
    name = field.name
    mine, theirs = getattr(f, name), getattr(build, name)
    if name in moot or mine == theirs:
      continue
    off = name == "glossy" or build.switches & SWITCHES.get(name, 0) or _bound_off(build, name) or _class_off(build, f, name)
    if theirs and not mine and (off or _maps_off(build, f, name) or name in _SUPERSETS):
      continue
    if name in _CAPS and scene is not None and _covers(theirs, mine, scene, name):
      continue
    if name in _RUN_COUNTS and 0 < mine < theirs and toggles.reads(build, toggles.RUN_COUNTS):
      continue
    if name in _FORM_FIELDS and scene is not None and _small_covers(build, scene):
      continue
    out.append(name)
  return tuple(out)


# The Features fields a tier build reads from its word: `BOUND_FIELDS` and the light grid's lookup
# (`toggles.CLASS_CULLING`, an OptiX build's: the grid off is the module whose word has it off).
_WORD_FIELDS: Final = {**BOUND_FIELDS, **FIELD_BITS, "culling": toggles.CLASS_CULLING, **_SETTING_BITS}


def word_bit(name: str) -> int:
  """The build-word bit Features field `name` is read from in a tier build (`_WORD_FIELDS`; 0:
  none)."""
  return _WORD_FIELDS.get(name, 0)


# The light loop's form and count: a small form of the build covering the scene's lights serves a
# configuration whose own small form differs (soft shadows: the largest, `resolve._light_forms`).
_FORM_FIELDS: Final = ("light_forms", "nlight")


def _small_covers(build: Features, scene: Features) -> bool:
  """Whether `build`'s small light-loop form holds every light of `scene` (the configuration's
  Features specialised on it) where the configuration takes a small form there too (every light
  traced: its scene-free counts may still be below the scene's lights)."""
  size = small_form(build.light_forms)
  counts = build.nlight in (RUNTIME_LIGHTS, scene.nlight)
  return 0 < scene.nlight <= size and small_form(scene.light_forms) > 0 and counts


def soft(build: Features, own: Features) -> bool:
  """Whether a configuration resolving to `own` (specialised on the scene) samples its shadows
  otherwise than tier build `build` compiles (`_SOFT`): more or fewer samples, a sun disc of its
  own (a scene without one has none)."""
  if not (build.shadows and own.shadows):
    return False
  samples = own.shadow_samples != build.shadow_samples
  disc = own.sun_disc and not build.sun_disc
  return samples or disc


def aniso(build: Features, own: Features) -> bool:
  """Whether a configuration resolving to `own` filters its textures with another tap count than
  tier build `build` compiles (`_ANISO`)."""
  return own.textures and build.textures and own.anisotropy != build.anisotropy


def below(build: Features, own: Features) -> bool:
  """Whether a configuration resolving to `own` shades fewer lights exactly or traces fewer
  casters than tier build `build` compiles (`_RUN_COUNTS`; 0, every light or caster of the scene:
  the compiled count covers it), where `build` reads them (`toggles.RUN_COUNTS`)."""
  taps = own.textures and own.anisotropy < build.anisotropy
  fewer = taps or any(0 < getattr(own, n) < getattr(build, n) for n in ("budget", *_CAPS))
  return fewer and bool(toggles.count_bits(build))


def run_flags(build: Features, own: Features) -> int:
  """The run-time sampling and counts (in-half bits, `toggles.SOFT_SHADOWS`, `ANISOTROPY`,
  `RUN_COUNTS`) tier build `build` takes for a configuration resolving to `own`: the launch's
  `Toggles.flags` and, on OptiX, its module's word."""
  sampling = (toggles.SOFT_SHADOWS if soft(build, own) else 0) | (toggles.ANISOTROPY if aniso(build, own) else 0)
  return sampling | (toggles.RUN_COUNTS if below(build, own) else 0)


def _bound_off(build: Features, name: str) -> bool:
  """Whether `build` reads Features field `name` from its word (`_WORD_FIELDS`): a configuration
  turning it off is served by the module whose word has it off."""
  return bool(build.bound & _WORD_FIELDS.get(name, 0))


# The Features fields of each bound class (`_uses`' reading of `CLASSES`): a configuration using
# none of a class turns it off in the module's word (`word`).
_CLASS_FIELDS: Final = {
  "coat": ("clearcoat",),
  "layers": ("specular_anisotropy", "fuzz", "coat_layer", "coat_normal", "detail_normal"),
  "layer_maps": ("coat_normal_map", "detail_normal", "layered_maps"),
  "base_map": ("base_color_map",),
  "maps": (
    "normal_map",
    "orm_map",
    "emissive_map",
    "opacity_map",
    "diffuse_roughness_map",
    "scaled_normal_map",
    "uv_maps",
    "world_uv",
  ),
  "materials": ("reflection", "diffuse_roughness", "specular_weight", "vertex_colors", "unlit_emission"),
  "glass": ("refraction", "glass_shapes"),
  "headlight": ("headlight",),
  "ies": ("ies",),
  "area": ("area_lights", "light_panels"),
}


def _class_off(build: Features, f: Features, name: str) -> bool:
  """Whether field `name` belongs to bound classes of OptiX build `build` (its word's class bits)
  that the configuration `f` uses none of ("textures": the base color and the maps classes): the
  module whose word has them off serves it."""
  if build.traversal != "optix":
    return False  # a software build's word holds its served set (`word`), not the configuration's classes
  classes = ("base_map", "maps") if name == "textures" else tuple(c for c, fs in _CLASS_FIELDS.items() if name in fs)
  bound = all(build.bound & CLASS_MASKS[c] for c in classes)
  return bool(classes) and bound and not any(getattr(f, x) for c in classes for x in _CLASS_FIELDS[c])


# Features fields whose on value renders the off one's image exactly: tinted shadows (RGB
# visibilities) through gray or clear casters (their transmittance is run-time data,
# `context.transmittance`: equal channels, or 1 with "clear"); the alpha test, whose cutoffs a
# configuration without it binds as zeros (`frame_structs.alpha_cutoffs`: no material is tested, a
# surface keeps its alpha; the micromaps follow the configuration, `omm_sources.allowed`).
_SUPERSETS: Final = frozenset({"tinted_shadows", "alpha_cutout"})
# The texture slots a build samples, each with the word bit gating it (`uv_frame.map_roles`): the
# base color's class, the other slots' field bits, the layered maps' class.
_TEXTURE_SLOTS: Final = {
  "base_color_map": CLASS_MASKS["base_map"],
  **{n: FIELD_BITS[n] for n in ("normal_map", "orm_map", "emissive_map", "opacity_map", "diffuse_roughness_map")},
  **dict.fromkeys(("layered_maps", "coat_normal_map", "detail_normal"), CLASS_MASKS["layer_maps"]),
}


def _maps_off(build: Features, f: Features, name: str) -> bool:
  """Whether the module of OptiX build `build` whose word has `f`'s texture slots folds field
  `name` off: "textures" (no slot sampled: each slot `build` compiles is gated by its bit and off
  in `f`), a layered map's slot, or the coat's own normal (with neither a normal map nor the
  layered maps it equals the base's)."""
  if build.traversal != "optix":
    return False

  def folded(slot: str) -> bool:
    """Whether `build` samples slot `slot` in no module of `f`'s word."""
    return not getattr(build, slot) or (not getattr(f, slot) and bool(build.bound & _TEXTURE_SLOTS[slot]))

  if name == "textures":
    return all(folded(slot) for slot in _TEXTURE_SLOTS)
  if name == "coat_normal":
    return all(folded(slot) for slot in ("normal_map", "coat_normal_map", "detail_normal"))
  return name in _TEXTURE_SLOTS and _TEXTURE_SLOTS[name] == CLASS_MASKS["layer_maps"] and folded(name)


def _covers(build_cap: int, cap: int, scene: Features, name: str) -> bool:
  """Whether caps `build_cap` and `cap` (0: none) both shade every light of `scene` exactly (its
  own `name` 0: no cap below its light count)."""
  n = scene.nlight
  return getattr(scene, name) == 0 and n != RUNTIME_LIGHTS and all(c == 0 or c >= n for c in (build_cap, cap))


def switched_off(build: Features, f: Features) -> tuple[str, ...]:
  """The switches, word settings (`BOUND_FIELDS` of `toggles.SETTING_ALL`) and "glossy" `build`
  compiles in that `f` turns off."""
  settings = (n for n, bit in BOUND_FIELDS.items() if bit & toggles.SETTING_ALL)
  return tuple(n for n in (*SWITCHES, *settings, "glossy") if getattr(build, n) and not getattr(f, n))


def with_switches(f: Features) -> Features:
  """`f` as a tier build: the `SWITCHES` it compiles in read at run time, those that can be turned
  off without leaving it; its bound bits (`Features.bound`, `word`): the `BOUND_FIELDS` it
  compiles in (GI only in an OptiX scene-free build: elsewhere a settings axis); on OptiX the
  bound classes it compiles in and, scene-free, the geometry bits (`BOUND_FREE`), the headlight
  in a translucent small-loop build (`_bound`); on the software path, scene-free, its bound
  classes (`toggles.DEFINED`: the word its modules' literal). `SETTING_FIELDS` are switches on the
  software path, word bits on OptiX."""
  optix = f.traversal == "optix"
  words = set(SETTING_FIELDS) if optix else set()
  switches = sum(bit for name, bit in SWITCHES.items() if getattr(f, name) and name not in words)
  switches |= 0 if optix else toggles.sampling_bits(f) | toggles.count_bits(f)
  return dataclasses.replace(f, switches=switches, bound=_bound(f))


def _bound(f: Features) -> int:
  """The bound bits of tier build `f` (`with_switches`)."""
  settings = _field_bits(f) & ~(0 if f.traversal == "optix" and f.scene_free else toggles.SETTING_GI)
  if f.traversal != "optix":
    bits = (_class_bits(used_classes(f) & bound_classes(f.traversal)) if f.scene_free else 0) | settings
    # The solid glass primitives' check, a module literal (a module per value): a glass scene
    # without glass primitives folds it (compiled in, showcase's FAST megakernel +1.1 % SASS, +348 B
    # of spills, CPU proxy).
    bits |= toggles.high(toggles.GLASS_SHAPES) if f.scene_free and f.glass_shapes else 0
    return bits | toggles.DEFINED if bits else 0
  bound = _class_bits(used_classes(f) & BOUND_CLASSES) | (toggles.CLASS_CULLING if f.culling else 0) | settings
  bound |= _bits_of(f, FIELD_BITS) | _settings(f) | toggles.high(toggles.sampling_bits(f) | toggles.count_bits(f))
  if f.scene_free:
    bound |= toggles.GEO_ALL | _flex_bits(f)
  # The small loop's headlight entry: bound where the layer loop runs it per layer (translucent
  # builds: unbound, glass fast paid 2.7 % on RTX), compiled in on otherwise (its straight-line
  # entry is free there; bound, an opaque robot scene's fast build ran 1.0-1.3 % slower).
  return bound & ~toggles.CLASS_HEADLIGHT if small_loop(f.light_forms) and not f.transparency else bound


def _field_bits(f: Features) -> int:
  """The `BOUND_FIELDS` bits of the fields Features `f` has on."""
  return _bits_of(f, BOUND_FIELDS)


def _flex_bits(f: Features) -> int:
  """The deformables' bits of Features `f` (`toggles.FLEX`, `FLEX_MATERIALS`)."""
  flex = toggles.FLEX if int(ShapeType.FLEX) in f.shape_types else 0
  return flex | (toggles.FLEX_MATERIALS if f.flex_materials else 0)


def _bits_of(f: Features, table: dict[str, int]) -> int:
  """The bits of `table` (field -> bit; fields may share one) of the fields Features `f` has on."""
  return functools.reduce(operator.or_, (bit for name, bit in table.items() if getattr(f, name)), 0)


def _settings(f: Features) -> int:
  """The `SETTING_FIELDS` word bits (second half) of the settings Features `f` has on."""
  return _bits_of(f, _SETTING_BITS)


def _class_bits(classes: frozenset[str]) -> int:
  """The word bits of the bound `classes` (`CLASS_MASKS`)."""
  return sum(CLASS_MASKS[c] for c in classes)


def _shape_bits(f: Features) -> int:
  """The `toggles.SHAPE_BITS` of the shape types Features `f` has (or-ed: ellipsoid and cylinder
  share one bit; their sum set the next bit, `SCENE_SPOT`, and cleared it)."""
  return functools.reduce(operator.or_, (bit for t, bit in toggles.SHAPE_BITS.items() if t in f.shape_types), 0)


def _with_word(build: Features, own: Features) -> Features:
  """Tier build `build` serving `own`, with its word when its modules define it (software): but
  the bits read from the word copies at run time (`toggles.RUN_TIME`), so that one module serves
  every scene value of them."""
  if not build.bound & toggles.DEFINED:
    return build
  return dataclasses.replace(build, word=word(build, own) & ~toggles.RUN_TIME)


def word(build: Features, own: Features) -> int:
  """The value of tier build `build`'s word for a scene whose configuration resolves to `own`
  (specialised on the scene), but the OptiX geometry bits
  (`trace/optix/device/programs.build_word`): the bound bits it uses among the build's (its classes;
  on the software path the served set's, `served`), the shape bits of its shape types and its
  `BOUND_FIELDS`."""
  classes = used_classes(own) if build.traversal == "optix" else served(own)
  if build.traversal == "optix" and not own.headlight:
    # A small-loop build compiles the headlight's entry (`used_classes`); where it binds it
    # (`_bound`), the bit says whether the scene's configuration lights it.
    classes -= {"headlight"}
  bits = _class_bits(classes & BOUND_CLASSES - {"primitives"}) | _shape_bits(own)
  bits |= (toggles.CLASS_CULLING if own.culling else 0) | _flex_bits(own) | _bits_of(own, FIELD_BITS)
  bits |= _settings(own) | toggles.high(run_flags(build, own))
  return (bits | _field_bits(own)) & build.bound & toggles.BOUND_ALL


def relight_word(build: Features, own: Features) -> int:
  """The value of relight build `build`'s word (`kernels/probe.relight_features`, bound as an
  OptiX scene-free build) for a scene whose configuration resolves to `own`: `word`'s bits of the
  classes, shapes, deformables and settings `own` uses, and the software traversal's structure
  bits of its trees (`toggles.GEO_*`: the trees it walks, compact poses, the visibility table, the
  plane loop)."""
  trees = (
    (toggles.GEO_PLANES if own.planes != PLANES_NONE else 0)
    | (toggles.GEO_WORLD if own.world_tree else 0)
    | (toggles.GEO_SHARED if own.shared else 0)
    | (toggles.GEO_COMPACT if own.compact_poses else 0)
    | (toggles.GEO_VISIBLE if own.visibility else 0)
  )
  return word(dataclasses.replace(build, traversal="optix"), own) | (trees & build.bound)


# The shape types a backend's tier builds do not trace correctly: a configuration of a scene with
# one takes its own kernels there. OptiX: the scene-free programs miss the deformables' hits (their
# own build renders them).
UNSERVED_SHAPES: Final[dict[str, frozenset[int]]] = {"optix": frozenset({int(ShapeType.FLEX)})}


def unserved_shapes(effective: Features) -> frozenset[int]:
  """The shape types of `effective` (specialised on the scene) its backend's tiers do not serve
  (`UNSERVED_SHAPES`)."""
  return UNSERVED_SHAPES.get(effective.traversal, frozenset()) & frozenset(effective.shape_types)


def route(
  s: RenderSettings,
  effective: Features,
  resolve: Callable[[RenderSettings, str], Features],
  *,
  moot: frozenset[str] = frozenset(),
) -> Choice:
  """The kernels of settings `s` resolved to `effective` (specialised on the scene): the first
  tier whose build (`resolve(tier_settings(s, tier), tier)`, scene-free on the tier's axes, the
  switches run time: `axes`) serves the settings' own scene-free Features on them (`resolve(s,
  tier)`; `moot`: fields no frame on the scene reads, `differences`), else `effective`'s own
  kernels. A tier whose settings the renderer cannot take (`resolve` raises ValueError:
  supersampling with motion on OptiX) serves nothing; nor does any for a scene with a shape type its
  backend's tiers do not serve (`unserved_shapes`)."""
  if unserved_shapes(effective):
    return Choice(tier=None, build=effective, outside=("shape_types",))
  nearest: tuple[str, tuple[str, ...]] | None = None
  for tier in TIERS:
    try:
      build = with_switches(resolve(tier_settings(s, tier), tier))
      target = resolve(s, tier)
    except ValueError:
      continue  # the tier's settings do not fit this renderer (per-camera resolutions, motion)
    diff = differences(build, target, effective, unread=moot)
    if not diff:
      return Choice(tier=tier, build=_with_word(build, effective), off=switched_off(build, effective))
    if nearest is None or len(diff) < len(nearest[1]):
      nearest = (tier, diff)
  if nearest is None:
    return Choice(tier=None, build=effective)
  return Choice(tier=None, build=effective, outside=nearest[1], nearest=nearest[0])


def prebuilt(s: RenderSettings, build: Features) -> bool:
  """Whether the prebuilt bundle (`core/build/kernel_cache/prebuilt.py`) holds build `build` of
  settings `s`: a scene-free tier build (`Features.scene_free`; `tools/prebuilt/fill.py` enumerates
  them all) under `performance.kernels="prebuilt"`. A miss of such a build warns (`prebuilt.ptx`);
  the scene's own builds and the configurations outside every tier compile by design."""
  return s.performance.kernels == "prebuilt" and build.scene_free
