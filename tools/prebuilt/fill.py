"""The prebuilt bundle's enumeration: every build a scene can route to, per backend.

The bundle of an architecture holds one backend (`core/build/kernel_cache/prebuilt.py`: OptiX on
GPUs with RT cores, the software BVH elsewhere). Its builds are the product of the axes a tier build
still takes from the scene: the tiers (`tiers.TIERS`, every one scene-free), GI on / off, the
prebuilt feature-class sets (`class_sets.PREBUILT` and every class: `tiers.cover` routes a scene to
the cheapest superset; a complete tier, `tiers.COMPLETE`, every class only), the light-loop and
plane forms (one plain base scene per light count and plane form, `feature_space.base_scenes`) and
the sharing layout (shared-only, both, world-only: the scene's instances posed per world through
`PerWorld(pose=...)`, none, one, all).
The outputs are the output word's (run-time data) but for the body and the motion output's temporal
history: the tier settings' shaded outputs or a geometry-only request, each with or without motion
(`OUTPUT_FORMS`). The other bundle axes, each on the sets its scenes use (and the per-world optics'
glass point, `_optics_points`):
per-world visibility (the layouts' twins with every instance's visibility declared, on
`class_sets.VISIBLE`), the MuJoCo pose layout (the base scenes' MuJoCo copies, shared-only and both:
their shared instances read world 0's pose row, on `class_sets.ALIASED`) and deformables (the copies
with a cloth, on `class_sets.FLEX`).
Each fill point is resolved on the CPU (nothing compiled) to its tier build and PTX key.

  python -m tools.prebuilt.fill census --backend optix --out OUT.json

prints, per tier, the distinct builds and PTX keys and the axis values reached, and the totals:
the bundle's build count (each a PTX / Warp megakernel compile and, on OptiX, a module); writes
the rows to OUT.json.gz and each distinct build's Features to OUT.builds.json.gz.

  python -m tools.prebuilt.fill fill --root R --workers 32 [--census C.json ...]

compiles the PTX of every distinct OptiX build (and its GI programs) on the CPU, NVRTC only, into
`R/raywarp/ptx`, one per true code split, each with the module records the census reaches per tier
(`<key>.json`, a `prebuilt.Entry`; `prebuilt.ModuleRecord`, one per class-bits value: the module
pack's rows, `build.py pack`). The
census: the scene-key census outputs (registry, feature-space scenes; development repository
only), each row's `units` (PTX key, tier, record), most used first. `fill --warp`, a job of its
own, the Warp modules the frames launch (`warmup.first_frame`) into `R/warp`; for `build.py
collect --root R`. A root seeded with an earlier run's output resumes (what it holds is not
compiled).

  python -m tools.prebuilt.fill list --backend optix --part I/N --out D   # N processes, one each
  python -m tools.prebuilt.fill spread --ptx-shards 48 --warp-shards 4 --out D

list every fill point once (each part its base scenes) and write the shard files of a sharded
fill (`D/ptx-<i>.json`, `D/warp-<i>.json`: each shard's points, and its Warp modules by size); a
shard job then runs `fill --shard D/ptx-<i>.json` (or `fill --warp --shard D/warp-<i>.json`) and
resolves only its own points. `tools/prebuilt/fill_all.sh` runs the whole pipeline on HF jobs
(list, spread, the shards, merge, the module pack).

  python -m tools.prebuilt.fill fill --backend software --root R --workers 8

the software bundle (Warp modules only): each tier build's megakernel modules, one per word, linked
from one LTO-IR per build (`tools/prebuilt/software.py`, the NVRTC / nvJitLink wheels of Warp's
CUDA version in the fill environment), and every other module its frames launch. A point lists
its build's modules with each value of the scene bits a software word keeps as literals
(`literal_twins`: the visible lights, the spot cone), which no base scene sets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import gzip
import hashlib
import importlib
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final

from tools.prebuilt.build_cache import BuildCache

if TYPE_CHECKING:
  import warp as wp

  from raywarp.core.build.kernel_cache.prebuilt import Bundle, Entry, ModuleRecord
  from raywarp.core.build.variants.resolve import Resolution
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.kernels.abi.output_layout import OutputLayout
  from raywarp.core.pipeline.renderer import Renderer
  from raywarp.core.pipeline.worlds.sharing import PerWorld
  from raywarp.core.trace.optix.device.programs import SceneFlags
  from raywarp.settings import RenderSettings
  from raywarp.settings.outputs import Outputs
  from tools.prebuilt.probe import Resolver

# The GPU profiles of each backend's census (`probe.PROFILES`).
PROFILES: Final = {"optix": "rtx_pro_6000", "software": "a100"}
# The sharing layouts of a fill scene: the instances posed per world (none: shared-only; a prop,
# instance 1: both; every instance: world-only).
# Each with and without a per-world visibility mask over every instance (`PerWorld.visible`: the
# "visibility" axis).
LAYOUTS: Final[dict[str, tuple[tuple[int | str, ...], tuple[str, ...]]]] = {
  "shared": ((), ()),
  "both": ((1,), ()),
  "world": (("*",), ()),
  "shared_visible": ((), ("*",)),
  "both_visible": ((1,), ("*",)),
  "world_visible": (("*",), ("*",)),
}
# The per-world optics builds (`Features.world_optics`: a renderer choosing materials per world in a
# scene with see-through content, BUG-487) are those of one domain-randomisation glass point
# (`feature_space.OPTICS_POINT` with `OPTICS_PER_WORLD`, also a census form): every tier, GI on /
# off and output form on the class sets with the translucent layers ("surface"; the glass class is
# bound on OptiX). Its layout: every instance per world but the planes (`sharing.decide`, optics).
# The layouts of each base-scene geometry (`feature_space.base_scenes`): a MuJoCo copy's world-only
# layout has no shared instance, so the host scene's builds (compact poses only matter with shared
# instances).
GEOMETRY_LAYOUTS: Final = {
  "mesh": tuple(LAYOUTS),
  "mjmesh": ("shared", "both", "shared_visible", "both_visible"),
  "flex": tuple(LAYOUTS),
  "flexplain": tuple(LAYOUTS),
}


def geometry_layouts(geometry: str, backend: str) -> tuple[str, ...]:
  """The sharing layouts filled for a base scene of `geometry` on `backend` (`GEOMETRY_LAYOUTS`):
  none for a deformable scene on a backend whose tiers do not serve deformables
  (`tiers.UNSERVED_SHAPES`: such a scene renders its own build, BUG-795)."""
  from raywarp._base.types import ShapeType
  from raywarp.core.build.variants import tiers

  if geometry in ("flex", "flexplain") and int(ShapeType.FLEX) in tiers.UNSERVED_SHAPES.get(backend, frozenset()):
    return ()
  return GEOMETRY_LAYOUTS[geometry]


def _sets(geometry: str, layout: str, backend: str) -> tuple[frozenset[str], ...]:
  """The class sets filled for a base scene of `geometry` in `layout`: every prebuilt set
  (`class_sets.PREBUILT`), with visibility those of `class_sets.VISIBLE`, with deformables those of
  `class_sets.FLEX`, a MuJoCo copy (aliased poses) those of `class_sets.ALIASED` (several: those in
  each); every class always. On OptiX their distinct code-split parts (`tiers.served`: the bound
  classes are in every build; visibility and pose aliasing are bound)."""
  from raywarp.core.build.variants import class_sets, tiers

  sets = set(class_sets.PREBUILT)
  flex = geometry in ("flex", "flexplain")
  if flex:
    sets &= set(class_sets.FLEX)
  if backend == "optix":
    return tuple(sorted({s & tiers.SPLIT_CLASSES for s in sets} | {tiers.SPLIT_CLASSES}, key=sorted))
  if LAYOUTS[layout][1]:
    sets &= set(class_sets.VISIBLE)
  if geometry in ("mjmesh", "flex", "flexplain"):
    sets &= set(class_sets.ALIASED)
  return (*sorted(sets, key=sorted), frozenset(tiers.CLASSES))


def _points(
  backend: str,
  *,
  modules: bool = True,
  only: list[dict[str, str]] | None = None,
  profile: str | None = None,
  part: tuple[int, int] = (0, 1),
) -> Iterator[tuple[dict[str, object], Build | None]]:
  """Every fill point: each base scene (`feature_space.base_scenes`) in each of its sharing layouts
  (`GEOMETRY_LAYOUTS`), tier, GI on / off and class set (`_sets`), resolved on the CPU: its census
  row, and on OptiX the build its programs compile (None for settings a scene cannot honor, or on
  the software backend); `modules`: list each point's Warp modules; `only`: the points whose row
  has the values of one of these selections (`_matches`: a dry run's, or a listed shard's points,
  `listing`); `profile`: the GPU (`harness.PROFILES`, default the backend's `PROFILES` entry);
  `part` (I, N): every N-th base scene from the I-th (a listing part)."""
  from raywarp.core.build.variants import class_sets
  from tools.prebuilt import feature_space
  from tools.prebuilt import probe as harness

  profile = profile or PROFILES[backend]
  fill = _Fill(backend=backend, profile=profile, arch=harness.PROFILES[profile][0], modules=modules, only=only)
  tables = class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED
  # Before the loop: `_sets` reads the tables, which each point overwrites with its own set.
  plan = {(g, layout): _sets(g, layout, backend) for g in GEOMETRY_LAYOUTS for layout in geometry_layouts(g, backend)}
  try:
    if part[0] == 0:
      yield from _optics_points(fill, backend)
    for point in itertools.islice(feature_space.base_scenes(), part[0], None, part[1]):
      code: str = point.code()
      if only is not None and all(o.get("scene", code) != code for o in only):
        continue
      scene, cams, _ = harness.load_point(code)
      for layout in geometry_layouts(point.geometry, backend):
        if only is not None and all(o.get("scene", code) != code or o.get("layout", layout) != layout for o in only):
          continue
        r = _renderer(scene, cams, layout)
        try:
          yield from _layout_points(fill, harness.Probe(code, r), layout, plan[point.geometry, layout])
        finally:
          r.close()
  finally:
    class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED = tables


def _optics_points(fill: _Fill, backend: str) -> Iterator[tuple[dict[str, object], Build | None]]:
  """The fill points of the per-world optics glass point (`feature_space.OPTICS_POINT`, layout
  "optics")."""
  import raywarp as rw
  from tools.prebuilt import feature_space
  from tools.prebuilt import probe as harness

  code: str = feature_space.OPTICS_POINT.code()
  only = fill.only
  if only is not None and all(o.get("scene", code) != code or o.get("layout", "optics") != "optics" for o in only):
    return
  scene, cams, _ = harness.load_point(code)
  # The sets with its classes: the translucent layers' code split (OptiX: the glass class is bound).
  need = {"surface"} if backend == "optix" else {"surface", "glass"}
  sets = tuple(s for s in _sets("mesh", "world", backend) if need <= s)
  r = _renderer(scene, cams, rw.PerWorld(material=list(feature_space.OPTICS_PER_WORLD["material"])))
  try:
    yield from _layout_points(fill, harness.Probe(code, r), "optics", sets)
  finally:
    r.close()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Fill:
  """What `_points` resolves the points for: the backend, GPU profile and its SM version, whether
  to list each point's Warp modules, the selections of the points wanted (None: all)."""

  backend: str
  profile: str
  arch: int
  modules: bool
  only: list[dict[str, str]] | None


def _renderer(scene: object, cameras: object, layout: str | PerWorld) -> Renderer:
  """A CPU renderer of `scene` (no device work) in sharing layout `layout` (`LAYOUTS`), or with
  per-world declaration `layout`."""
  import warp as wp

  import raywarp as rw
  from raywarp.core.pipeline.renderer import Renderer
  from tools.prebuilt import probe as harness

  with harness.no_device_work(), wp.ScopedDevice("cpu"):
    if isinstance(layout, str):
      pose, visible = LAYOUTS[layout]
      per_world = rw.PerWorld(pose=list(pose), visible=list(visible)) if pose or visible else None
    else:
      per_world = layout
    return Renderer(scene, nworld=2, cameras=cameras, resolution=(64, 64), per_world=per_world)  # pyrefly: ignore[bad-argument-type]


def _layout_points(
  fill: _Fill, probe: Resolver, layout: str, sets: tuple[frozenset[str], ...]
) -> Iterator[tuple[dict[str, object], Build | None]]:
  """The fill points of one base scene (`probe`, a `harness.Probe`) in sharing layout `layout`:
  every tier, GI on / off, class set of `sets` (`_tier_sets`) and output form (`_points`)."""
  from raywarp.core.build.variants import class_sets, tiers

  # A complete tier's build serves every class whatever the set (`tiers.COMPLETE`): one set.
  axes = ((t, g, c, o) for t in bundled_tiers() for g in (False, True) for c in _tier_sets(t, sets) for o in bundled_forms(t))
  for tier, gi, classes, form in axes:
    # This set served (`tiers.cover`: unless the scene needs more).
    class_sets.PREBUILT = class_sets.VISIBLE = class_sets.FLEX = class_sets.ALIASED = (classes,)
    s = tiers.TIERS[tier].replace(**{"gi.enabled": gi, "performance.traversal": fill.backend})
    s = s.replace(outputs=_form_outputs(form, s.outputs))
    row: dict[str, object] = {
      "scene": probe.name,
      "layout": layout,
      "tier": tier,
      "gi": gi,
      "set": sorted(classes),
      "outputs": form,
    }
    if fill.only is not None and not any(_matches(row, o) for o in fill.only):
      continue
    try:
      resolution = probe.resolution(s, fill.profile)
    except ValueError as e:
      yield row | {"error": str(e)}, None
      continue
    f = resolution.features
    build_id = hashlib.sha1(repr(f).encode()).hexdigest()[:16]
    BUILDS.setdefault(build_id, {k: repr(v) for k, v in vars(f).items()})
    row |= {
      "class": [int(v) for v in tiers.feature_class(f)],
      "axes": {k: repr(getattr(f, k)) for k in tiers.SCENE_AXES},
      "build": build_id,
    }
    build = _build(probe, f, s, fill.arch, lenses=bundled_lenses(tier)) if fill.backend == "optix" else None
    if build is not None:
      row["ptx_key"] = build.key
      row["flags"], row["supersampled"] = repr(build.flags), build.supersampled
    if fill.modules and build is not None:
      row["warp_modules"] = _warp_modules(probe, resolution, s, fill.backend)
    elif fill.modules and fill.backend == "software":
      named = (_warp_modules(probe, twin, s, fill.backend) for twin in literal_twins(resolution))
      row["warp_modules"] = list(dict.fromkeys(name for names in named for name in names))
    yield row, build


def literal_twins(resolution: Resolution) -> list[Resolution]:
  """`resolution` of a software fill point with each value of the scene word bits its build binds
  as module literals (`toggles.SCENE_ALL` but `toggles.RUN_TIME`: the visible lights, the spot
  cone), whatever the point's own: each value is a megakernel and probe-ray module of its own
  (`toggles.preamble`)."""
  from raywarp.core.kernels.abi import toggles

  f = resolution.features
  literal = f.bound & toggles.SCENE_ALL & ~toggles.RUN_TIME
  bits = _bits(literal)
  subsets = (sum(c) for n in range(len(bits) + 1) for c in itertools.combinations(bits, n))
  return [resolution._replace(features=dataclasses.replace(f, word=f.word & ~literal | v)) for v in subsets]


def _bits(mask: int) -> list[int]:
  """The single bits of `mask`."""
  return [1 << i for i in range(mask.bit_length()) if mask >> i & 1]


def _tier_sets(tier: str, sets: tuple[frozenset[str], ...]) -> tuple[frozenset[str], ...]:
  """The class sets filled for `tier` among a point's `sets`: every class alone for a complete tier
  (`tiers.COMPLETE`: its build is the same for each; with and without the surface split for a
  `tiers.SURFACE_SPLIT` one), else all of them."""
  from raywarp.core.build.variants import tiers

  every = frozenset(tiers.CLASSES)
  if tier in tiers.SURFACE_SPLIT:
    return (every, every - {"surface"})
  return (every,) if tier in tiers.COMPLETE else sets


def _matches(row: dict[str, object], only: dict[str, str]) -> bool:
  """Whether census `row` has the values `only` (as text; the class set as "a+b")."""
  named = {k: str(v) for k, v in row.items()} | {"set": "+".join(row.get("set", []))}  # pyrefly: ignore[no-matching-overload]
  return all(named.get(k) == v for k, v in only.items())


# The output forms of the fill points: what of the outputs a tier build still compiles (every
# other output change is the output word's, run-time data: `kernels/abi/output_layout.py`), by name
# -> the outputs from the tier settings'. The body (`Features.any_rgb`): their shaded outputs, or a
# geometry-only request (depth, segmentation, hit outputs: the unshaded body, a handful of builds
# every tier shares); and the "motion" output (`Features.motion`: the temporal history, a build
# of its own) beside either (its status, `MotionOutput.status`, is a word bit).
OUTPUT_FORMS: Final = ("shaded", "geometry", "motion", "geometry_motion")


# Not in the bundle unless `NICHE` (user decisions, 2026-11-11, W14 and 2026-11-16: they compile
# once on first use, or ahead in `Renderer.warm`): variants of the default tier (reflection-hit sun
# shadows, a shadow budget below the exact lights), 2 x 2 supersampling outside the quality recipe
# (`default_ss2_counts` stays; `default_counts` is the per-frame RTX look's), the sampled-shadow
# variants of the fast and
# default tiers (the counted ones serve more samples too), the unlit tier's run-time texture taps,
# and the motion-vector output forms of every tier but `MOTION_TIERS`. The zero-compile walks list
# them as documented exceptions (`zero_holes.json`).
NICHE_TIERS: Final = frozenset(
  {
    "default_hit_sun",
    "default_casters",
    "fast_ss2",
    "default_ss2",
    "fast_sampled",
    "default_sampled",
    "unlit_ss2",
    "unlit_sampled",
  }
)
NICHE_FORMS: Final = frozenset({"motion", "geometry_motion"})
# The tiers whose motion-vector forms are bundled (SLAM and visual odometry: the fast form) and
# the lens models whose render programs are bundled beside their builds' (`lens_keys`: the fisheye
# / OpenCV distortion program of the fast tier; every other lens program compiles on first use).
MOTION_TIERS: Final = frozenset({"fast"})
LENS_TIERS: Final[dict[str, tuple[str, ...]]] = {"fast": ("distortion",)}
# Whether the fill compiles the niche tiers and forms too (`fill --niche`, `list --niche`).
NICHE = False


def bundled_tiers() -> tuple[str, ...]:
  """The tiers the fill compiles (`tiers.TIERS`, less `NICHE_TIERS` unless `NICHE`)."""
  from raywarp.core.build.variants import tiers

  return tuple(t for t in tiers.TIERS if NICHE or t not in NICHE_TIERS)


def bundled_forms(tier: str) -> tuple[str, ...]:
  """The output forms the fill compiles for `tier` (`OUTPUT_FORMS`, less `NICHE_FORMS` unless
  `NICHE` or a `MOTION_TIERS` tier)."""
  return tuple(o for o in OUTPUT_FORMS if NICHE or tier in MOTION_TIERS or o not in NICHE_FORMS)


def bundled_lenses(tier: str) -> tuple[str, ...]:
  """The lens models (`primary.LENS_MODELS` names) whose programs the fill compiles beside `tier`'s
  builds (`LENS_TIERS`; every one with `NICHE`)."""
  from raywarp.core.trace.rays import primary

  return tuple(m.name for m in primary.LENS_MODELS) if NICHE else LENS_TIERS.get(tier, ())


def _form_outputs(form: str, outputs: Outputs) -> Outputs:
  """The outputs of fill-point output form `form` (`OUTPUT_FORMS`) from the tier settings'
  `outputs`."""
  from raywarp.settings.outputs import MotionOutput
  from raywarp.settings.outputs import Outputs as OutputsOf

  base = outputs if form in ("shaded", "motion") else OutputsOf.of("depth")
  return dataclasses.replace(base, motion=MotionOutput()) if form.endswith("motion") else base


# The Features of every distinct tier build `_points` resolved, by build id (its census rows'
# "build"): `census` writes them beside the rows (a coverage miss is diffed against them).
BUILDS: dict[str, dict[str, str]] = {}


@dataclasses.dataclass(frozen=True)
class Build:
  """What one OptiX build compiles (`launch.OptixRenderer`'s programs, resolved without a device):
  its PTX key, program Features and flags, whether it renders supersampled, its register caps
  (shaded, light: `registers.register_caps`; each module's cap: `registers.optix_registers` of its
  word), target and the lens models whose programs are bundled beside it (`bundled_lenses`)."""

  key: str
  features: Features
  flags: SceneFlags
  supersampled: bool
  registers: tuple[int, int]
  arch: int
  lenses: tuple[str, ...] = ()


def _build(probe: Resolver, f: Features, settings: RenderSettings, arch: int, *, lenses: tuple[str, ...] = ()) -> Build:
  """The OptiX build of Features `f` (`settings`) over the probe's scene (`lenses`: its bundled
  lens programs)."""
  from raywarp.core.build.variants import registers
  from raywarp.core.trace.optix import supersample
  from raywarp.core.trace.optix.context import MAX_PTX_ARCH
  from raywarp.core.trace.optix.device import programs
  from tools.prebuilt import probe as harness

  supersampled = f.spp > 1 and f.any_rgb
  f_prog = supersample.program_features(f) if supersampled else f
  flags = programs.program_flags(f_prog, harness.scene_flags(probe, f_prog))
  # The PTX target a device of `arch` compiles for (`trace/optix/context.ptx_arch`: at most sm_89),
  # which its key names: a build keyed or compiled for the device's own architecture would never
  # hit.
  target = min(arch, MAX_PTX_ARCH)
  key = programs.ptx_key(f_prog, flags, supersampled=supersampled, arch=target)
  caps = registers.register_caps(settings)
  return Build(key=key, features=f_prog, flags=flags, supersampled=supersampled, registers=caps, arch=target, lenses=lenses)


# Warp modules of the fill points (`_warp_modules`), by name: what a frame launches beside the
# OptiX programs (structures, GI bake, sensors, display), or a software build's megakernels and
# the rest, compiled once each (`_compile_warp`).
_MODULES: dict[str, wp.Module] = {}


def _warp_modules(scene: object, resolution: Resolution, settings: RenderSettings, backend: str) -> list[str]:
  """The Warp modules of a first frame of the probe `scene` resolved to `resolution` with
  `settings` on `backend` (`warmup.first_frame`: as far as it lists them; its software helper
  kernels with the scene's ray-query layout, `features.software_view`, as a frame's; the relight
  kernel of a GI build; a shaded build's deferred display variants, `_post_modules`), recorded in
  `_MODULES` with the library's module-level ones (`_library_modules`); their names."""
  from raywarp.core.pipeline import baked, warmup
  from raywarp.core.trace.optix_backend import OptixBackend
  from raywarp.core.trace.software_backend import SoftwareBackend

  ctx = dataclasses.replace(
    scene.renderer.context(),  # pyrefly: ignore[missing-attribute]
    settings=settings,
    features=resolution.features,
    effective=resolution.effective,
    relight=resolution.relight,
    camera_model=resolution.camera,
  )
  # The backend class: `first_frame` reads only its static `first_frame_kernels`.
  traced = OptixBackend if backend == "optix" else SoftwareBackend
  default, render = warmup.first_frame(ctx, traced).modules()
  # The relight kernel of a GI configuration (`baked.relight_kernel`, as a relit bake launches it:
  # one per probe-relevant settings, `resolve.relight_build`): the warm-up lists it for scenes
  # whose worlds' GI inputs differ (`warmup._relights`: declared or batched per-world tables),
  # which the fill scenes lack.
  relight = (baked.relight_kernel(ctx).module,) if resolution.relight is not None else ()
  post = _post_modules() if resolution.features.any_rgb else ()
  lens = _lens_megakernels(resolution.features) if backend == "software" else ()
  area = _area_kernel(resolution.features) if backend == "software" else ()
  names = []
  for module in (*_library_modules(), *_simulator_modules(), *default, *render, *relight, *post, *lens, *area):
    _MODULES.setdefault(module.name, module)
    names.append(module.name)
  return names


def _lens_megakernels(f: Features) -> tuple[wp.Module, ...]:
  """The lens camera models' megakernels of software build `f` (`pixel.megakernel` of each
  `primary.LENS_MODELS` model, beside the probe cameras' pinhole one): each model is a megakernel
  of its own, as on OptiX a program of its own, so a lens camera's first frame loads, never
  compiles."""
  from raywarp.core.kernels import pixel
  from raywarp.core.trace.rays import primary

  return tuple(pixel.megakernel(f, model).module for model in primary.LENS_MODELS)


def _area_kernel(f: Features) -> tuple[wp.Module, ...]:
  """The soft area shadows' tracing kernel of software build `f` (`area_shadows.software_kernel`;
  none without LTC area lights or shading): `shadows.area_rays` turned on loads, never compiles."""
  from raywarp.core.shade.shadows import area as area_shadows

  return (area_shadows.software_kernel(f).module,) if f.area_lights and f.any_rgb else ()


@functools.cache
def _post_modules() -> tuple[wp.Module, ...]:
  """The deferred display's modules of every source, graded and not, and their meters
  (`deferred.reachable_kernels`): a run-time post-effect change loads, never compiles."""
  from raywarp.core.post.display import deferred

  return tuple(dict.fromkeys(k.module for k in deferred.reachable_kernels()))


# The library's packages whose module-level kernels a scene's upload and frames launch beside the
# warm-up's (`_library_modules`), with every other module defining device code outside the OptiX
# programs (`kernel_code.WARP_MODULES`: the Newton and Isaac Lab adapters' too, whose packages
# import their simulators).
LIBRARY: Final = ("raywarp._base", "raywarp.core", "raywarp.scene", "raywarp.integrations.mujoco")
# MuJoCo Warp's package whose module-level kernels the bundle holds too (`_simulator_modules`):
# they are the same for every model (its per-model specialisations are kernels of their own
# modules, `module="unique"`, not bundled), so a fresh machine's first `mjw.forward` / `mjw.step`
# after a renderer's seed loads them instead of compiling them (`smooth`, `forward`, `sensor`,
# `solver`, `collision_*`: seconds each). Named by their content like every Warp module: another
# mujoco-warp release than the fill's misses them and compiles, as it would without them.
SIMULATOR: Final = "mujoco_warp._src"
# Name endings of `SIMULATOR`'s modules that are no simulation code: its tests, its CLI tool.
_NOT_SIMULATION: Final = ("_test", ".cli")
# The kinds of the fill's Warp modules (`module_kind`): counted apart in the listing and the bundle.
KINDS: Final = ("renderer", "simulator")


def module_kind(name: str) -> str:
  """The kind of the fill's Warp module `name` (a module's name, or its cache directory's,
  "wp_<name>_<hash>"): "simulator" for one of `SIMULATOR` (`_simulator_modules`), else "renderer"
  (raywarp's and the Warp builtins it launches)."""
  return "simulator" if name.removeprefix("wp_").startswith(f"{SIMULATOR}.") else "renderer"


def kind_counts(names: Iterable[str]) -> dict[str, int]:
  """The number of modules `names` of each kind (`KINDS`)."""
  kinds = [module_kind(n) for n in names]
  return {k: kinds.count(k) for k in KINDS}


@functools.cache
def _library_modules() -> tuple[wp.Module, ...]:
  """Every Warp module of `LIBRARY` and `kernel_code.WARP_MODULES` holding compilable
  module-level kernels (`_compilable`; its Python modules imported): the upload's (environment
  maps, textures, MuJoCo capabilities), the structures' and the frame's helpers launched outside
  `warmup.first_frame`, the adapters' bindings and sensors. A module that cannot be imported (the
  fill environment lacks Newton) raises: the bundle would lack its kernels."""
  import warp as wp

  from raywarp.core.build.kernel_cache import kernel_code

  names: list[str] = []
  for package in LIBRARY:
    names += _package_modules(package)
  names += sorted(set(kernel_code.WARP_MODULES) - set(names))
  for name in names:
    importlib.import_module(name)
  return tuple(m for m in (wp.get_module(name) for name in names) if _compilable(m))


@functools.cache
def _simulator_modules() -> tuple[wp.Module, ...]:
  """MuJoCo Warp's module-level Warp modules holding compilable kernels (`SIMULATOR`'s modules,
  imported), but its tests' and its command-line tool's (`_NOT_SIMULATION`)."""
  import warp as wp

  names = [n for n in _package_modules(SIMULATOR) if not n.endswith(_NOT_SIMULATION)]
  for name in names:
    importlib.import_module(name)
  return tuple(m for m in (wp.get_module(name) for name in names) if _compilable(m))


def _package_modules(package: str) -> list[str]:
  """The names of `package` and of every module under it (imported to be listed)."""
  import pkgutil

  root = importlib.import_module(package)
  return [package, *(m.name for m in pkgutil.walk_packages(root.__path__, f"{package}."))]


def _compilable(module: wp.Module) -> bool:
  """Whether `module` holds a kernel Warp compiles as it is: a plain one, or a generic one with
  overloads declared at import (`per_world.COLUMN_DTYPES`; a generic kernel without any compiles
  nothing)."""
  return any(not k.is_generic or k.overloads for k in module.kernels.values())


def _owner(module: wp.Module) -> str:
  """What assigns `module` to a shard (`listing`): a word module's group (its LTO-IR and every
  link stay together, `software.group_key`), else its name (which holds its content hash)."""
  from tools.prebuilt import software

  return software.group_key(module) if software.words_of(module) else module.name


def _compile_warp(
  root: Path, arch: int, workers: int, names: frozenset[str] | None, seed: str | None, kept: BuildCache | None
) -> int:
  """The modules of `_MODULES` named in `names` (None: all; a listed shard's, `listing`), compiled
  for sm_`arch` into the fill's Warp kernel cache (`root`/warp/<Warp release>/<module id>/, the
  layout a load reads; `build.py collect` takes it), each reported as it ends; the module count.
  Skipped: the modules whose entry the cache holds (a resumed shard's snapshot), the previous
  bundle `seed` holds (`seed_bundle`; copied in: same id, same code) or the build cache `kept`
  holds; each module compiled here is kept there. The software word modules first
  (`software.fill_group`: one LTO-IR per group, a link per word), then the others (NVRTC), never
  both at once (a group's capture swaps Warp's binary step). Compiled: their count."""
  import warp as wp

  cache = root / "warp" / wp.config.version
  wanted = [m for m in _MODULES.values() if names is None or m.name in names]
  ids = {m.get_module_identifier() for m in wanted}
  if seed is not None:
    print(f"FILL seeded {_take_modules(seed_bundle(seed), cache, ids, arch)} Warp modules from {seed}", flush=True)
  if kept is not None:
    print(f"FILL cached {len(kept.take_warp(wp.config.version, ids, cache))} Warp modules from {kept.root}", flush=True)
  held = _held(cache, arch)
  groups: dict[str, list[wp.Module]] = {}
  plain: list[wp.Module] = []
  for module in wanted:
    key = _owner(module)
    if module.get_module_identifier() in held:
      continue
    if key == module.name:
      plain.append(module)
    else:
      groups.setdefault(key, []).append(module)
  t0 = time.perf_counter()

  def keep() -> None:
    """Keeps the modules compiled so far in the build cache (`held`: those found, not compiled)."""
    if kept is not None:
      for name in sorted(ids & _held(cache, arch) - held):
        kept.keep_warp(wp.config.version, cache / name)
        held.add(name)

  # A group's links run one after another (9-55 s each): its modules in chunks of `GROUP_CHUNK`,
  # each a job of its own (its own LTO-IR), so `workers` link at once.
  chunks = [ms[i : i + GROUP_CHUNK] for ms in groups.values() for i in range(0, len(ms), GROUP_CHUNK)]
  n = _run(workers, [functools.partial(_fill_group, ms, arch, cache) for ms in chunks], "group", t0, keep)
  return n + _run(workers, [functools.partial(_compile_module, m, arch, cache) for m in plain], "module", t0, keep)


def _take_modules(seed: Bundle, cache: Path, ids: set[str], arch: int) -> int:
  """Copies the Warp modules of `ids` that bundle `seed` holds (of sm_`arch` and this Warp
  release: none of another's) and `cache` lacks into `cache` (fetched first when `seed` is remote;
  whole entries: copied under a temporary name, then renamed); their number. Only those: a module
  of the previous tree that this fill does not list is not carried over."""
  import warp as wp

  if (seed.manifest.arch, seed.manifest.warp) != (arch, wp.config.version):
    print(f"FILL seed {seed.root}: sm{seed.manifest.arch} Warp {seed.manifest.warp}, no module of it taken", flush=True)
    return 0
  listed = seed.module_files()
  todo = sorted((ids & set(listed)) - _held(cache, arch))
  failed = seed.fetch(rel for name in todo for rel in listed[name])
  cache.mkdir(parents=True, exist_ok=True)
  taken = 0
  for name in todo:
    if any(rel in failed for rel in listed[name]):
      continue  # compiled instead
    partial = cache / f".{name}.{os.getpid()}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    for rel in listed[name]:
      shutil.copyfile(seed.root / rel, partial / Path(rel).name)
    shutil.rmtree(cache / name, ignore_errors=True)  # an entry without a binary (an interrupted move)
    partial.rename(cache / name)
    taken += 1
  return taken


def seed_bundle(seed: str) -> Bundle:
  """The previous bundle `seed`: a complete directory (`<package>/sm<arch>/`), or a published
  bundle's URL (`<kernels>/sm<arch>/<content>`: its file list fetched into a scratch store of this
  machine, each file when first read), read once per process.

  Raises:
    SystemExit: the published bundle cannot be fetched.
  """
  return _seed_bundle(seed)


@functools.cache
def _seed_bundle(seed: str) -> Bundle:
  """`seed_bundle`'s (once per process)."""
  from raywarp.core.build.kernel_cache import prebuilt
  from raywarp.core.build.kernel_cache import store as kernel_store

  if "://" not in seed:
    return prebuilt.Bundle(Path(seed), prebuilt.Manifest.read(Path(seed) / prebuilt.MANIFEST))
  local = Path(tempfile.gettempdir()) / "raywarp_fill_seed" / hashlib.sha1(seed.encode()).hexdigest()[:16]
  try:
    files = kernel_store.listing(local, seed, None)
    first = prebuilt.Bundle(local, prebuilt.Manifest(arch=0, warp="", content=""), seed, files)
    return prebuilt.Bundle(local, prebuilt.Manifest.read(first.path(prebuilt.MANIFEST)), seed, files)
  except kernel_store.FetchError as e:
    raise SystemExit(f"FILL the seed bundle {seed}: {e}") from e


def _held(cache: Path, arch: int) -> set[str]:
  """The module ids Warp kernel cache `cache` holds an sm_`arch` binary of."""
  return {p.name for p in cache.iterdir() if any(p.glob(f"*.sm{arch}*.cubin"))} if cache.is_dir() else set()


def _fill_group(modules: list[wp.Module], arch: int, cache: Path) -> int:
  """One software word group (`software.fill_group`); its module count."""
  from tools.prebuilt import software

  return software.fill_group(modules, arch, cache)


def _compile_module(module: wp.Module, arch: int, cache: Path) -> int:
  """One module compiled (NVRTC) in a scratch directory, its files moved into its cache entry; 1."""
  import warp as wp

  from tools.prebuilt import software

  scratch = Path(tempfile.mkdtemp(prefix="raywarp_fill_"))
  try:
    (cubin,) = wp.compile_aot_module(module, arch=arch, module_dir=str(scratch))
    software.move_entry(scratch, cache / cubin.name.split(".sm")[0])
  finally:
    shutil.rmtree(scratch, ignore_errors=True)
  return 1


def _run(workers: int, jobs: list[Callable[[], int]], what: str, t0: float, after: Callable[[], None]) -> int:
  """`jobs` on `workers` threads (NVRTC and nvJitLink release the GIL), `after` and one line as
  each ends (the heartbeat); the modules they filled."""
  done = 0
  with concurrent.futures.ThreadPoolExecutor(workers) as pool:
    futures = [pool.submit(job) for job in jobs]
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
      done += future.result()
      after()
      print(f"FILL warp {what} {i}/{len(jobs)}, {done} modules (elapsed {time.perf_counter() - t0:.0f} s)", flush=True)
  return done


def census(backend: str, out: Path) -> None:
  """Resolves every fill point (see the module doc) and prints the bundle's build count."""
  t0 = time.perf_counter()
  rows: list[dict[str, object]] = []
  scene = None
  for row, _ in _points(backend, modules=False):  # the build count needs no Warp-module lists
    rows.append(row)
    if row["scene"] != scene:
      scene = row["scene"]
      print(f"FILL {scene} {len(rows)} rows {time.perf_counter() - t0:.0f} s", flush=True)
  # Compressed (a job's output mirror skips large files): OUT.json.gz, OUT.builds.json.gz.
  with gzip.open(out.with_suffix(".json.gz"), "wt") as f:
    json.dump(rows, f)
  with gzip.open(out.with_suffix(".builds.json.gz"), "wt") as f:
    json.dump(BUILDS, f)
  _report(rows)


def fill(
  root: Path,
  workers: int,
  *,
  warp: bool = False,
  backend: str = "optix",
  only: list[dict[str, str]] | None = None,
  profile: str | None = None,
  census: tuple[Path, ...] = (),
  shard: Path | None = None,
  seed: str | None = None,
  cache: Path | None = None,
  lens: bool = False,
) -> None:
  """Compiles the PTX of every distinct OptiX build of the fill points (and its GI programs' PTX,
  its bundled lens programs, `Build.lenses`; with `lens` every lens model's program too,
  `lens_keys`: a lens camera of another build compiles its model's program once per build, user
  decisions 2026-11-11 and 2026-11-16)
  into `root`/raywarp/ptx, each with the kernel code its generation used (`reach`) and the module
  records per tier the census files `census` reach (`census_records`; `build.py collect` then
  makes the bundle): `workers` processes, NVRTC only (no GPU). `warp` (always on the software
  backend, whose bundle is Warp modules only): instead, the Warp modules the fill points' frames
  launch on `backend` (`_warp_fill`). `shard`: a shard file of `spread` (its points and Warp
  modules: a shard resolves only those points); `only`: the points whose census row has these
  values (scene, tier, layout, gi, set as "a+b": a dry run's few builds); `profile`: the GPU (the
  backend's `PROFILES` entry by default). What `root` already holds (a resumed shard's snapshot)
  is not compiled again, nor what the previous bundle `seed` (`seed_bundle`: a directory or a
  published bundle's URL) or the build cache `cache` (`build_cache.BuildCache`: the mounted cache
  bucket's root) holds for this tree: the PTX of each listed key whose reached code is unchanged
  (`prebuilt.Entry.fresh`) and the listed Warp modules (same id, same code), each copied in
  (`root` holds only this fill's keys and modules: `build.py collect` bundles nothing else); what
  the cache lacks is kept there as it is made. Prints the build count, the wall time and the PTX
  bytes."""
  t0 = time.perf_counter()
  listed = None if shard is None else json.loads(shard.read_text())
  only = listed["points"] if listed is not None else only
  if warp or backend == "software":
    names = None if listed is None else frozenset(listed["modules"])
    _warp_fill(root, workers, names, backend=backend, only=only, profile=profile, seed=seed, cache=cache)
    return
  records, nrow, old = census_records(list(census))
  builds: dict[str, Build] = {}
  for _, build in _points("optix", modules=False, only=only, profile=profile):
    if build is not None:
      builds.setdefault(build.key, build)
  jobs = list(builds.values())
  reached = {k for b in jobs for u in _units(b, {}, lens=lens) for k in (u.key,)}
  outside = sorted(set(records) - reached)
  nmodule = sum(len(rs) for groups in records.values() for rs in groups.values())
  print(
    f"FILL census: {nrow} rows ({old} without units: regenerate with internal/benchmarks/lab/scene_keys.py), "
    f"{nmodule} modules over {len(records)} PTX keys; {len(outside)} keys no fill point "
    f"reaches{f' (e.g. {outside[0]}): those scenes compile' if outside else ''}",
    flush=True,
  )
  print(f"FILL {len(jobs)} builds resolved in {time.perf_counter() - t0:.0f} s", flush=True)
  ptx = root / "raywarp" / "ptx"
  ptx.mkdir(parents=True, exist_ok=True)
  t1 = time.perf_counter()
  context = multiprocessing.get_context("spawn")
  # One process per PTX (a build's render, lens and GI programs in parallel): its reach is recorded
  # from the process start (`reach`: a factory cached by an earlier build would hide the code that
  # made its result).
  with concurrent.futures.ProcessPoolExecutor(
    workers, mp_context=context, initializer=_worker, initargs=(str(root),), max_tasks_per_child=1
  ) as pool:
    # As each build finishes (`map` yields in order: one long build held back every later line and
    # tripped the stall watchdog while the other workers went on).
    units = [(job, unit) for job in jobs for unit in _units(job, records, lens=lens)]
    futures = [pool.submit(_compile, job, [unit], seed, None if cache is None else str(cache)) for job, unit in units]
    hows: list[str] = []
    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
      key, seconds, how = future.result()
      hows.append(how)
      print(f"FILL {i}/{len(futures)} {key[:16]} {how} {seconds:.0f} s (elapsed {time.perf_counter() - t1:.0f} s)", flush=True)
  files = list(ptx.glob("*.ptx"))
  size = sum(f.stat().st_size for f in files)
  print(
    f"FILL done: {len(jobs)} builds, {len(files)} PTX ({size / 2**20:.0f} MB) in {time.perf_counter() - t1:.0f} s "
    f"on {workers} processes (resolve {t1 - t0:.0f} s); {_tally(hows)}",
    flush=True,
  )


def _tally(hows: list[str]) -> str:
  """ "compiled N, kept N, seeded N, cached N" of the units' `_compile` outcomes."""
  return ", ".join(f"{how} {hows.count(how)}" for how in ("compiled", "kept", "seeded", "cached"))


def _warp_fill(
  root: Path,
  workers: int,
  names: frozenset[str] | None,
  *,
  backend: str,
  only: list[dict[str, str]] | None,
  profile: str | None,
  seed: str | None,
  cache: Path | None,
) -> None:
  """`fill`'s Warp modules: the points listed (one pass, cheap: no compile), then the modules
  `names` (None: all) compiled (`_compile_warp`; `seed`: a previous bundle's, `cache`: the build
  cache's taken first).

  Raises:
    SystemExit: a listed module the points no longer make (the shard files are another tree's).
  """
  from tools.prebuilt import probe as harness

  t0 = time.perf_counter()
  for i, _ in enumerate(_points(backend, modules=True, only=only, profile=profile), 1):
    if i % 1000 == 0:  # the heartbeat of the listing
      print(f"FILL listing {i} points, {len(_MODULES)} Warp modules ({time.perf_counter() - t0:.0f} s)", flush=True)
  print(f"FILL {len(_MODULES)} Warp modules listed {kind_counts(_MODULES)} in {time.perf_counter() - t0:.0f} s", flush=True)
  if names is not None and not names <= set(_MODULES):
    raise SystemExit(f"FILL {len(names - set(_MODULES))} listed modules not made here: list again on this tree")
  arch = harness.PROFILES[profile or PROFILES[backend]][0]
  kept = None if cache is None else BuildCache(cache, arch)
  nmod = _compile_warp(root, arch, workers, names, seed, kept)
  print(f"FILL done: compiled {nmod} Warp modules in {time.perf_counter() - t0:.0f} s", flush=True)


def listing(
  backend: str,
  out: Path,
  part: tuple[int, int],
  only: list[dict[str, str]] | None = None,
  profile: str | None = None,
) -> None:
  """Lists the fill points of base-scene part `part` (I, N: every N-th base scene from the I-th,
  `_points`) once, with their Warp modules, into `out`/part-<I>.json: one point of each build and
  of each module, with the module's shard owner and source size (`_source_size`); `spread` makes
  the shard files of the parts. Parts run as parallel processes (a listing is one core). `only`:
  just the points of these selections (`_points`: a dry run's few builds); `profile`: the GPU (the
  backend's `PROFILES` entry by default; one listing per architecture, the shards' `fill --profile`
  the same)."""
  t0 = time.perf_counter()
  builds: dict[str, dict[str, str]] = {}
  made: dict[str, dict[str, str]] = {}
  for i, (row, build) in enumerate(_points(backend, modules=True, part=part, only=only, profile=profile), 1):
    point = _selection(row)
    if build is not None:
      builds.setdefault(build.key, point)
    for name in row.get("warp_modules", ()):  # pyrefly: ignore[not-iterable]
      made.setdefault(name, point)
    if i % 1000 == 0:  # the heartbeat
      print(f"FILL listing {part[0]}/{part[1]}: {i} points, {len(made)} modules ({time.perf_counter() - t0:.0f} s)", flush=True)
  modules = [[name, _owner(_MODULES[name]), _source_size(_MODULES[name]), point] for name, point in made.items()]
  out.mkdir(parents=True, exist_ok=True)
  _write(out / f"part-{part[0]}.json", {"builds": list(builds.items()), "modules": modules})
  print(
    f"FILL listed {part[0]}/{part[1]}: {len(builds)} builds, {len(made)} modules {kind_counts(made)} in "
    f"{time.perf_counter() - t0:.0f} s",
    flush=True,
  )


# The most modules of one software word group compiled as one job (`_compile_warp`, `spread`): one
# LTO-IR (minutes) and as many links (9-55 s each on cpu-upgrade, one after another).
GROUP_CHUNK: Final = 12
# Shard wall per software word module on cpu-upgrade (4 workers): a link (9-55 s, dry run
# 43978938600) and its share of its chunk's LTO-IR, over the workers: the sizing estimate `spread`
# prints.
WORD_WALL_S: Final = 17


def spread(out: Path, ptx_shards: int, warp_shards: int, *, chunk: int = GROUP_CHUNK, limit: int = 0) -> None:
  """The shard files of a sharded fill (`fill --shard`) from the listing parts in `out`
  (`listing`): `ptx-<i>.json` (one point of each build, every `ptx_shards`-th build from the i-th),
  `warp-<i>.json` (the modules, their owners spread over `warp_shards` shards by size, `_spread`,
  each `SHARD_ALONE` owner in a shard of its own, and one point of each) and `listing.json` (the
  counts and each Warp shard's size). `chunk`: a software word group of more modules is split into
  owners of at most `chunk` (each its own LTO-IR and links: a group's links run one after another,
  9-55 s each, so a whole group of hundreds outlasted a shard's cap; dry run 43978938600).
  `limit`: only that many modules (a dry run's proof of the path: the smallest word group's first
  two and the smallest other modules)."""
  builds: dict[str, dict[str, str]] = {}
  made: dict[str, tuple[str, int, dict[str, str]]] = {}
  for path in sorted(out.glob("part-*.json"), key=lambda p: int(p.stem.split("-")[1])):
    data = json.loads(path.read_text())
    for key, point in data["builds"]:
      builds.setdefault(key, point)
    for name, owner, size, point in data["modules"]:
      made.setdefault(name, (owner, size, point))
  made = _chunked(made, chunk)
  if limit:
    made = _limited(made, limit)
  keys = list(builds)
  for i in range(ptx_shards if keys else 0):
    _write(out / f"ptx-{i}.json", {"points": _unique(builds[k] for k in keys[i::ptx_shards])})
  owners: dict[str, list[str]] = {}
  sizes: dict[str, int] = {}
  for name, (owner, size, _) in made.items():
    owners.setdefault(owner, []).append(name)
    sizes[owner] = sizes.get(owner, 0) + size
  # The relight kernels each compile alone for ~13 min (2026-11-11 w10r-probe-warp: 759 s, the
  # other modules of a fill point ~2 s each): one shard each, the rest spread by size.
  alone = sorted(o for o in owners if SHARD_ALONE in o and "__locals__" in o)  # a factory's
  if warp_shards > len(alone):
    shards = [[o] for o in alone] + _spread({o: n for o, n in sizes.items() if o not in alone}, warp_shards - len(alone))
  else:
    shards = _spread(sizes, warp_shards)
  words: list[int] = []
  for i, members in enumerate(shards):
    names = [n for owner in members for n in owners[owner]]
    words.append(sum(_word_owner(n, made[n][0].split("#")[0]) for n in names))
    _write(out / f"warp-{i}.json", {"points": _unique(made[n][2] for n in names), "modules": names})
  summary = {
    "builds": len(keys),
    "modules": len(made),
    "kinds": kind_counts(made),
    "owners": len(owners),
    "alone": len(alone),
    "warp_bytes": [sum(sizes[o] for o in members) for members in shards],
    "largest": sorted(((sizes[o], o) for o in sizes), reverse=True)[:10],
    "word_modules": words,
    "word_wall_min": [round(n * WORD_WALL_S / 60) for n in words],
  }
  _write(out / "listing.json", summary)
  print(
    f"FILL spread {len(keys)} builds, {len(made)} modules {summary['kinds']} ({len(owners)} owners); shard bytes "
    f"{summary['warp_bytes']}",
    flush=True,
  )
  if any(words):
    print(
      f"FILL spread software word modules per shard {words}: the longest ~{max(summary['word_wall_min'])} min of "
      f"links at ~{WORD_WALL_S} s each (more than the shard cap: raise WARP_SOFTWARE)",
      flush=True,
    )


def _word_owner(name: str, owner: str) -> bool:
  """Whether module `name`'s shard owner `owner` is a software word group (`_owner`: a group key,
  not the module's own name)."""
  return owner != name


def _chunked(made: dict[str, tuple[str, int, dict[str, str]]], chunk: int) -> dict[str, tuple[str, int, dict[str, str]]]:
  """`made` (module -> owner, size, point) with each word group of more than `chunk` modules (0:
  none split) split into owners `<group>#<i>` of at most `chunk` modules each, by name."""
  if chunk <= 0:
    return made
  groups: dict[str, list[str]] = {}
  for name, (owner, _, _) in made.items():
    if _word_owner(name, owner):
      groups.setdefault(owner, []).append(name)
  out = dict(made)
  for owner, names in groups.items():
    if len(names) > chunk:
      for i, name in enumerate(sorted(names)):
        _, size, point = made[name]
        out[name] = (f"{owner}#{i // chunk}", size, point)
  return out


def _limited(made: dict[str, tuple[str, int, dict[str, str]]], limit: int) -> dict[str, tuple[str, int, dict[str, str]]]:
  """`limit` modules of `made`: the first two (by name) of its smallest word-group owner, then the
  smallest other modules (a dry run: every compile path, in minutes)."""
  sizes: dict[str, int] = {}
  for name, (owner, size, _) in made.items():
    if _word_owner(name, owner):
      sizes[owner] = sizes.get(owner, 0) + size
  words = sorted(n for n, (o, _, _) in made.items() if sizes and o == min(sizes, key=lambda k: (sizes[k], k)))[:2]
  plain = sorted((s, n) for n, (o, s, _) in made.items() if not _word_owner(n, o))
  keep = words + [n for _, n in plain][: max(0, limit - len(words))]
  return {n: made[n] for n in keep}


# The Warp modules a Warp shard compiles alone (`spread`): the GI relight kernels (made by a
# factory: `make_relight...__locals__...`; the package module `...gi.relight` is small).
SHARD_ALONE: Final = "relight"


def _selection(row: dict[str, object]) -> dict[str, str]:
  """The selection (`_points` `only`) of fill point `row` alone."""
  return {k: "+".join(row["set"]) if k == "set" else str(row[k]) for k in _POINT_KEYS}  # pyrefly: ignore[no-matching-overload]


# A fill point's identity: the census row fields `_points` enumerates.
_POINT_KEYS: Final = ("scene", "layout", "tier", "gi", "set", "outputs")


def _unique(points: Iterator[dict[str, str]]) -> list[dict[str, str]]:
  """`points` without repeats, in order."""
  return [dict(p) for p in dict.fromkeys(tuple(p.items()) for p in points)]


def _write(path: Path, data: object) -> None:
  """`data` as JSON at `path`."""
  path.write_text(json.dumps(data, indent=1))


def _spread(sizes: dict[str, int], count: int) -> list[list[str]]:
  """The owners `sizes` (owner -> size) in `count` shards of about equal size: largest first, each
  into the smallest shard so far (longest-processing-time first), so that a module larger than a
  shard's share is alone in its shard."""
  shards: list[tuple[int, int, list[str]]] = [(0, i, []) for i in range(count)]
  for owner in sorted(sizes, key=lambda o: (-sizes[o], o)):
    size, i, members = min(shards)
    members.append(owner)
    shards[i] = (size + sizes[owner], i, members)
  return [members for _, _, members in shards]


def _source_size(module: wp.Module) -> int:
  """The Python source length of every kernel of `module` and of every function its kernels reach
  (each overload once, `reach.walk`): the size of the module's code, a proxy of its compile time."""
  from tools.prebuilt import reach

  return sum(len(adj.source) for adj in reach.walk(module)[0])


def _worker(root: str) -> None:
  """A fill process: its reach recorded from here (`reach.start`, before raywarp is imported),
  raywarp's cache under `root` (its PTX cache is the bundle's source)."""
  from tools.prebuilt import reach

  reach.start()
  os.environ["RAYWARP_CACHE_DIR"] = str(Path(root) / "raywarp")
  import warp as wp

  wp.config.log_level = wp.LOG_WARNING  # no init banner / module-load lines from every process
  wp.init()


@dataclasses.dataclass(frozen=True)
class Unit:
  """One PTX of a build (its render programs, a lens model's render program, its GI programs' or
  its soft area shadows' program): its key, whether it is the GI module, its lens model's name
  ("": none), whether it is the area pass' module, and the module records the census reaches per
  tier (`prebuilt.Entry.groups`)."""

  key: str
  gi: bool
  groups: dict[str, tuple[ModuleRecord, ...]]
  lens: str = ""
  area: bool = False


def _gi_key(build: Build) -> str | None:
  """The PTX key of `build`'s GI programs (None: it has none)."""
  from raywarp.core.trace.optix.device import programs

  f = build.features
  return programs.ptx_key(f, build.flags, supersampled=False, arch=build.arch, gi=True) if programs.gi_programs(f) else None


def area_key(build: Build) -> str | None:
  """The PTX key of `build`'s soft area shadows' program (`programs.build_area`; None: none, a
  build without LTC area lights or shading, or a supersampled one: `shadows.area_rays` needs LTC
  area lights and aa.spp=1, `RenderSettings._check_area_rays`). Its census records stay out of the
  pack: the setting on creates its module once."""
  from raywarp.core.trace.optix.device import programs

  f = build.features
  if not (f.area_lights and f.any_rgb) or build.supersampled:
    return None
  return programs.ptx_key(f, build.flags, supersampled=False, arch=build.arch, area=True)


def unit_records(build: Build, word: int, outputs: OutputLayout) -> list[tuple[str, ModuleRecord]]:
  """(PTX key, module record) of every module a scene of `build` whose word is `word`
  (`trace/optix/device/programs.build_word`) creates with the output word `outputs`
  (`output_layout.layout_of` of the written outputs): its render programs', and its GI programs' (no
  output word), each over both micromap options where the build has the alpha test (a scene's
  cut-outs may or may not have micromaps: `programs.scene_free_flags`). A census row's `units`
  (`scene_keys.py`)."""
  from raywarp.core.trace.optix import launch
  from raywarp.core.trace.optix.device import programs

  f, flags = build.features, build.flags
  omms = (False, True) if f.alpha_cutout else (False,)

  def records(layout: object) -> list[ModuleRecord]:
    """The records of the programs of launch-parameter layout `layout` over each micromap option
    (`launch.expected_record`: the rule of a build's modules created ahead)."""
    return [
      launch.expected_record(f, dataclasses.replace(flags, has_omm=omm), layout, word, outputs, build.registers)  # pyrefly: ignore[bad-specialization, bad-argument-type]
      for omm in omms
    ]

  out = [(build.key, r) for r in records(programs.layout(f, flags))]
  gi = _gi_key(build)
  return out + ([(gi, r) for r in records(programs.gi_layout(f, flags))] if gi is not None else [])


# A census: PTX key -> tier -> its module records, most used first (`census_records`).
type Census = dict[str, dict[str, tuple[ModuleRecord, ...]]]


# A census selection group: the rows of these recipes (None: every row), their own units ("own":
# each recipe word's) or their single-change neighbours' ("near").
type Group = tuple[frozenset[str] | None, str]
# Every row's own units, then every row's neighbours (the bundle index's records).
EVERY: Final[tuple[Group, ...]] = ((None, "own"), (None, "near"))


def groups_of(text: str) -> tuple[Group, ...]:
  """`a+b:own,c:near`: groups (`Group`) in order.

  Raises:
    ValueError: a part other than own / near.
  """
  out: list[Group] = []
  for item in filter(None, text.split(",")):
    recipes, _, part = item.partition(":")
    if part not in ("own", "near"):
      raise ValueError(f"{item!r}: RECIPE[+RECIPE]:own|near")
    out.append((frozenset(recipes.split("+")), part))
  return tuple(out)


def census_ranked(paths: list[Path], groups: tuple[Group, ...] = EVERY) -> tuple[list[tuple[str, str, ModuleRecord]], int, int]:
  """The module units the census files `paths` reach (`word_reach.py --units` rows' `units`), each
  once as (PTX key, tier, record), group by group (`groups`: a row's own units are its first `own`,
  the others its single-change neighbours'), within a group the most used first (ties by key,
  tier, record); a unit in several groups is in the first; (units, rows read, rows without units:
  an older census)."""
  from collections import Counter

  from raywarp.core.build.kernel_cache import prebuilt

  uses: list[Counter[tuple[str, str, ModuleRecord]]] = [Counter() for _ in groups]
  nrow = old = 0
  for path in paths:
    for row in json.loads(path.read_text()):
      chosen = [g for g, (recipes, _) in enumerate(groups) if recipes is None or row.get("recipe") in recipes]
      if not chosen:
        continue
      nrow += 1
      if "units" not in row:
        old += 1
        continue
      mine = row.get("own", len(row["units"]))
      parts = {"own": row["units"][:mine], "near": row["units"][mine:]}
      for g in chosen:
        uses[g].update((u["key"], u["group"], prebuilt.ModuleRecord.of(u["module"])) for u in parts[groups[g][1]])
  ranked = list(
    dict.fromkeys(unit for part in uses for unit, _ in sorted(part.items(), key=lambda kv: (-kv[1], kv[0][:2], repr(kv[0][2]))))
  )
  return ranked, nrow, old


def census_records(paths: list[Path], groups: tuple[Group, ...] = EVERY) -> tuple[Census, int, int]:
  """The module records the census files `paths` reach per PTX key and tier, in `census_ranked`'s
  order; (records, rows read, rows without units)."""
  ranked, nrow, old = census_ranked(paths, groups)
  out: dict[str, dict[str, list[ModuleRecord]]] = {}
  for key, group, record in ranked:
    out.setdefault(key, {}).setdefault(group, []).append(record)
  return {k: {g: tuple(rs) for g, rs in groups.items()} for k, groups in out.items()}, nrow, old


def lens_keys(build: Build, names: tuple[str, ...] | None = None) -> dict[str, str]:
  """Lens model name -> the PTX key of `build`'s render program for that model
  (`programs.build_lens`): every lens model (`primary.LENS_MODELS`; `names`: those), so that a
  lens camera's renderer finds its program bundled beside the build's."""
  from raywarp.core.trace.optix.device import programs
  from raywarp.core.trace.rays import primary

  f = build.features
  return {
    m.name: programs.ptx_key(f, build.flags, supersampled=build.supersampled, arch=build.arch, lens=m)
    for m in primary.LENS_MODELS
    if names is None or m.name in names
  }


def _units(build: Build, census: Census, *, lens: bool) -> list[Unit]:
  """The PTX units of `build` with the module records `census` holds for them: its bundled lens
  programs (`Build.lenses`; `lens`: every lens model's), with those of census scenes with such
  cameras."""
  units = [Unit(build.key, False, census.get(build.key, {}))]
  names = None if lens else build.lenses
  units += [Unit(key, False, census.get(key, {}), lens=name) for name, key in lens_keys(build, names).items()]
  gi = _gi_key(build)
  units += [Unit(gi, True, census.get(gi, {}))] if gi is not None else []
  area = area_key(build)
  return units + ([Unit(area, False, {}, area=True)] if area is not None else [])


def _compile(build: Build, units: list[Unit], seed: str | None, cache: str | None) -> tuple[str, float, str]:
  """The PTX of `build`'s `units`, each unless the PTX cache holds it, the previous bundle `seed`
  (`seed_bundle`) holds it for this kernel code (`_take`) or the build cache at `cache` does
  (`BuildCache.ptx_entry`), each stored with its entry (`<key>.json`: the reach of its generation,
  `reach.of`, its digest, and its module records per tier; `build.py collect` indexes them), and
  each compiled one kept in the build cache; (the last unit's key, seconds, per unit what was
  done: compiled, kept, seeded or cached)."""
  import warp as wp

  from raywarp.core.build.kernel_cache import kernel_code, prebuilt
  from raywarp.core.trace.optix import pipeline
  from raywarp.core.trace.optix.device import programs
  from raywarp.core.trace.rays import primary
  from tools.prebuilt import reach

  t0 = time.perf_counter()
  f, flags = build.features, build.flags
  done = []
  for unit in units:
    path = pipeline._ptx_path(unit.key)
    if path is None:
      raise SystemExit("RAYWARP_CACHE_DIR is empty: the fill has no PTX cache")
    kept = None if cache is None else BuildCache(Path(cache), build.arch)
    entry, how = _cached(path), "kept"
    if entry is None and seed is not None:
      entry, how = _take(seed_bundle(seed), unit.key, path), "seeded"
    if entry is None and kept is not None:
      entry, how = kept.ptx_entry(unit.key, path), "cached"
    if entry is None:
      if unit.gi:
        module = programs.build_gi(f, flags)
      elif unit.area:
        module = programs.build_area(f, flags)
      elif unit.lens:
        model = next(m for m in primary.LENS_MODELS if m.name == unit.lens)
        module = programs.build_lens(f, flags, model, supersampled=build.supersampled)
      else:
        module = programs.build(f, flags, supersampled=build.supersampled)
      ptx = wp.compile_aot_module(module, arch=build.arch, use_ptx=True)[0].read_bytes()
      found = reach.of([module])
      entry, how = prebuilt.Entry(ptx="", code=kernel_code.digest(found), reach=found, groups={}), "compiled"
      _write_bytes(path, ptx)
      if kept is not None:
        kept.keep_ptx(unit.key, path, entry)
    done.append(how)
    path.with_suffix(".json").write_text(json.dumps(dataclasses.replace(entry, ptx="", groups=unit.groups).json()))
  return units[-1].key, time.perf_counter() - t0, "+".join(done)


def _cached(path: Path) -> Entry | None:
  """The entry of the fill's PTX file `path` when it holds the PTX of this kernel code (a resumed
  shard's)."""
  from raywarp.core.build.kernel_cache import prebuilt

  try:
    entry = prebuilt.Entry.of(json.loads(path.with_suffix(".json").read_text()))
  except (OSError, ValueError, KeyError, TypeError):
    return None
  return entry if path.is_file() and entry.fresh() else None


def _take(seed: Bundle, key: str, path: Path) -> Entry | None:
  """The entry of PTX key `key` in the previous bundle `seed` when it holds the PTX of this kernel
  code (`Bundle.entry`; fetched when `seed` is remote; None too when a fetch fails: compiled
  instead), its PTX written at `path`."""
  import lzma

  from raywarp.core.build.kernel_cache import store as kernel_store

  try:
    entry = seed.entry(key)
    if entry is not None:
      _write_bytes(path, lzma.decompress(seed.path(f"ptx/{entry.ptx}").read_bytes()))
  except kernel_store.FetchError:
    return None  # warned once (`prebuilt.unreachable`)
  return entry


def _write_bytes(path: Path, data: bytes) -> None:
  """`data` written whole at `path` (a temporary name, then renamed)."""
  partial = path.with_suffix(f".{os.getpid()}.partial")
  partial.write_bytes(data)
  partial.replace(path)


def _report(rows: list[dict[str, object]]) -> None:
  """Per tier and in total: distinct builds, PTX keys, and the scene axes' values reached."""
  ok = [r for r in rows if "error" not in r]
  for tier in sorted({str(r["tier"]) for r in ok}):
    sel = [r for r in ok if r["tier"] == tier]
    axes = {k: sorted({str(r["axes"][k]) for r in sel}) for k in sel[0]["axes"]}  # pyrefly: ignore[bad-index, not-iterable]
    print(
      f"CENSUS {tier}: {len({r['build'] for r in sel})} builds, {len(_keys(sel))} PTX keys, "
      f"{len({json.dumps(r['class']) for r in sel})} class / loop values; axes {json.dumps(axes)}"
    )
  print(
    f"CENSUS total: {len({r['build'] for r in ok})} builds, {len(_keys(ok))} PTX keys over "
    f"{len(ok)} fill points ({len(rows) - len(ok)} settings a scene cannot honor)"
  )


def _keys(rows: list[dict[str, object]]) -> set[object]:
  """The PTX keys of census `rows`."""
  return {r["ptx_key"] for r in rows if "ptx_key" in r}


def fraction(text: str) -> tuple[int, int]:
  """`I/N` (a listing part, a pack shard): (I, N) with 0 <= I < N.

  Raises:
    argparse.ArgumentTypeError: another text.
  """
  index, _, count = text.partition("/")
  if not (index.isdigit() and count.isdigit() and int(index) < int(count)):
    raise argparse.ArgumentTypeError(f"{text!r}: I/N with 0 <= I < N, e.g. 3/16")
  return int(index), int(count)


def main(argv: list[str]) -> None:
  """`census --backend B --out OUT.json`, `list` / `spread --out D ...`, `fill --root R ...`."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  c = sub.add_parser("census")
  c.add_argument("--backend", choices=tuple(PROFILES), required=True)
  c.add_argument("--out", type=Path, required=True)
  ls = sub.add_parser("list")
  ls.add_argument("--backend", choices=tuple(PROFILES), required=True)
  ls.add_argument("--out", type=Path, required=True, help="the directory of the listing parts and shard files")
  ls.add_argument("--part", type=fraction, default=(0, 1), help="I/N: every N-th base scene from the I-th (parallel processes)")
  ls.add_argument("--niche", action="store_true", help="the niche tiers and output forms too (`NICHE_TIERS`, `NICHE_FORMS`)")
  ls.add_argument("--only", action="append", default=[], help="KEY=VALUE of the points to list (repeat; a dry run)")
  ls.add_argument("--profile", default=None, help="the GPU profile (tools/prebuilt/probe.PROFILES), e.g. l4")
  sp = sub.add_parser("spread")
  sp.add_argument("--out", type=Path, required=True, help="the directory of the listing parts")
  sp.add_argument("--ptx-shards", type=int, required=True, help="OptiX PTX shards (0 on the software backend)")
  sp.add_argument("--warp-shards", type=int, required=True, help="Warp module shards")
  sp.add_argument(
    "--chunk",
    type=int,
    default=GROUP_CHUNK,
    help="split software word groups into owners of at most this many modules (0: whole)",
  )
  sp.add_argument("--limit", type=int, default=0, help="only this many Warp modules (a dry run)")
  fl = sub.add_parser("fill")
  fl.add_argument("--root", type=Path, required=True, help="the cache root (`build.py collect --root`)")
  fl.add_argument("--workers", type=int, required=True, help="NVRTC processes (~3 GB each; a job's cores, not the host's)")
  fl.add_argument("--shard", type=Path, default=None, help="a shard file of `list`: its points (and Warp modules) only")
  fl.add_argument("--warp", action="store_true", help="the Warp modules instead (a job of its own)")
  fl.add_argument(
    "--backend", choices=tuple(PROFILES), default="optix", help="software: its Warp modules (an A100 / H100 bundle)"
  )
  fl.add_argument("--only", action="append", default=[], help="KEY=VALUE of the census rows to fill (repeat)")
  fl.add_argument(
    "--seed",
    default=None,
    help="a previous bundle (sm<arch>/ or a published one's URL): take its listed keys of unchanged code and listed modules",
  )
  fl.add_argument(
    "--cache", type=Path, default=None, help="the build cache's root (tools/prebuilt/build_cache.py): take, then keep"
  )
  fl.add_argument("--profile", default=None, help="the GPU profile (tools/prebuilt/probe.PROFILES), e.g. l4")
  fl.add_argument("--niche", action="store_true", help="the niche tiers and output forms too (`NICHE_TIERS`, `NICHE_FORMS`)")
  fl.add_argument("--lens", action="store_true", help="the lens models' programs too (`lens_keys`; not bundled by default)")
  fl.add_argument(
    "--census",
    type=Path,
    action="extend",
    nargs="+",
    default=[],
    help="scene_keys.py / word_reach.py --units outputs whose rows' units the bundle records",
  )
  a = p.parse_args(argv)
  global NICHE  # the CLI's choice for the module's enumeration
  NICHE = bool(getattr(a, "niche", False))
  if a.cmd == "census":
    census(a.backend, a.out)
  elif a.cmd == "list":
    listing(a.backend, a.out, a.part, [dict(kv.split("=", 1) for kv in a.only)] if a.only else None, a.profile)
  elif a.cmd == "spread":
    spread(a.out, a.ptx_shards, a.warp_shards, chunk=a.chunk, limit=a.limit)
  else:
    only = [dict(kv.split("=", 1) for kv in a.only)] if a.only else None
    fill(
      a.root,
      a.workers,
      warp=a.warp,
      backend=a.backend,
      only=only,
      profile=a.profile,
      census=tuple(a.census),
      shard=a.shard,
      seed=a.seed,
      cache=a.cache,
      lens=a.lens,
    )


if __name__ == "__main__":
  main(sys.argv[1:])
