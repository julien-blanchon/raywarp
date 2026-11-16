"""The compile-trigger inventory's table: for every compile-key input `triggers.derive` reads from
the code, its value domain, whether the prebuilt bundle covers it and what a miss costs. Coverage
of the Features fields is computed (`render`: the tiers' run-time mechanisms, the fill's builds
and the route walk's holes, `zero_holes.json`); the rest is stated here.
`tests/contracts/test_compile_triggers.py` keeps the table and the code in step: a new input fails
until its row exists. Miss costs: NVRTC of a build's PTX 90 s (fast) / 300 s (default) /
450 s (the former quality tier) on 32 CPU cores (W8-FINAL); an OptiX module 3-34 s on RTX
(W9-VIEWER; 115 s for the showcase room's 4, W8-DEMO); a Warp module 1-60 s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

PTX_S: Final = "90-450 s (NVRTC)"
MODULE_S: Final = "3-34 s per module (OptiX), x4 per build (render + GI, with motion)"
WARP_S: Final = "1-60 s (Warp)"

# Features field -> its value domain (the type's, narrowed where the code bounds it).
FEATURES: Final[dict[str, str]] = {
  "shape_types": "subset of 10 ShapeType (OptiX tier: every type)",
  "traversal": "optix / software",
  "ordered_traversal": "bool (tuning)",
  "max_registers": "int (tuning; 0 on CPU)",
  "block_dim": "int (tuning)",
  "spp": (
    "1 / 2 (OptiX supersampled: the factor aa.spp run time, Lens.scale; software per-frame body: Params.spp);"
    " int >= 1 in the software motion body"
  ),
  "glossy": "0 / 1 (reflections.enabled)",
  "reflection_shadows": "off / sun (a word bit: SETTING_FIELDS)",
  "sun_disc": "bool (soft sun shadows: shadow_samples > 1)",
  "nlight": "-1 (run time) / 1-4 (small form)",
  "light_forms": "0 / 1-4 / general",
  "shadow_samples": "int >= 1 (sampled tiers: run time)",
  "budget": "0-16 (counted tiers: run time, capacity 16)",
  "light_cap": "0 / 1-16 (counted tiers: run time)",
  "hit_light_cap": "0 / 1-16 (counted tiers: run time)",
  "anisotropy": "1-16 (sampled tiers: run time)",
  "planes": "none / some",
  "switches": "bits (derived: the tier's SWITCHES)",
  "bound": "bits (derived: the tier's bound bits)",
  "word": "bits (software word, derived)",
  "out_word": "bits (software output word, derived from outputs)",
}
# The Features fields every other row of which is a bool.
BOOL: Final = "bool"
# What to do with a Features field the bundle does not cover (or covers partly), one action each:
# (a) run-time data, (b) prebuild, (c) remove, (d) another compile strategy. Owners where another
# lane holds it.
ACTIONS: Final[dict[str, str]] = {
  "counters": "a profiling build (`core.pipeline.platform.tuning.override(counters=True)`, not a setting): compiles by design",
  "spp": (
    "done (W9-HOLES, W12-SOFT): the factor / sample count run time, spp 1 / 2 in the Features;"
    " software spp >= 2 + motion compiles (OptiX refuses it)"
  ),
  "shadow_samples": "done (W9-HOLES): sampled / counted variants of every lit tier family",
  "anisotropy": "done (W9-HOLES): run-time taps in the unlit, sampled and counted variants",
  "budget": "done (W9-HOLES): counted variants (run time up to 16); max_casters <= 16",
  "light_cap": "done (W9-HOLES): counted variants (run time up to 16)",
  "hit_light_cap": "done (W9-HOLES): as light_cap",
  "reflection_shadows": "done (W9-HOLES): a word bit (REFLECTION_SHADOWS); the default's counted variants",
  "area_lights": "covered on OptiX by the area class bit",
  "tinted_shadows": "done (W9-HOLES): a tinted build serves gray / clear casters (run-time transmittance)",
  "transparency": "done (W10-COMPILE-BITS): a setting bit (TRANSPARENCY); unread on a scene without translucent content",
  "live_material_uv": "done (W10-COMPILE-BITS): OptiX tier builds bind it (BOUND_FREE, MAP_LIVE_UV); software: the scene's",
  "textures": "done (W9-HOLES): folded by the slots' word bits",
  "unlit": "done (W9-HOLES): unlit ss2 / sampled variants",
  "coat_normal": "done (W9-HOLES): folded with the normal and layered maps",
  "detail_normal": "done (W9-HOLES): folded with the layered maps' class bit",
  "alpha_cutout": "unread without cut-out material; textures off on a cut-out scene: pairs",
  "world_optics": (
    "opt-in (user, 2026-11-13, BUG-487): only a renderer declaring instance.material / material.specular_ior per"
    " world on a scene with see-through content; prebuilt for the fill's glass point (`fill._optics_points`)"
  ),
}

# Device-code factory arguments that are not named after a Features field (primitive types): their
# source, read off the call sites (`triggers.factory_inputs` lists the expressions).
FACTORY_INPUTS: Final[dict[str, str]] = {
  "run_budget": "a tier build reading `toggles.RUN_COUNTS` (`Features.bound` / `switches`, `shadow_budget`)",
  "ao": "settings lighting.contact_ao > 0 (the deferred display's kernel variant; render kernels: none)",
  "area": "Features.area_lights",
  "banded": "device: launches are serial (the CPU's software megakernel in bands of rays, `trace.software_backend._banded`)",
  "camera_rays": "code (call site literal)",
  "cap": "Features.light_cap",
  "dual": "Features.run_counts or a tier build's `toggles.RUN_COUNTS` (both cap readers, `light_sources._make_run_cap`)",
  "coat": "Features.coat_normal_map",
  "column": "code (counter columns)",
  "compact": "Features.compact_poses / scene flags (OptiX hit programs)",
  "counted": "Features.counters",
  "cull_backfaces": "Features.backface_culling",
  "depth": "code (call site literal)",
  "detail": "Features.detail_normal",
  "direct": "settings outputs (direct / indirect split)",
  "enabled": "Features.diffuse_roughness / specular_weight",
  "every": "code (bound gate form)",
  "first_hit": "code (call site literal)",
  "first_tree": "Features.trees",
  "fisheye": "camera model (`rays.CameraModel.distortion`: a module of its own beside every build, `programs.build_lens`)",
  "distortion": "camera model (`rays.CameraModel.distortion`: a module of its own beside every build, `programs.build_lens`)",
  "fold": "Features.transparency and backface_culling",
  "graded": "settings grading.enabled (the deferred display's kernel; render kernels: False)",
  "gtype": "scene flags primitive_types (OptiX intersection programs)",
  "half": "code (word half)",
  "has_box": "scene flags has_box",
  "has_mesh": "scene flags has_mesh",
  "has_primitive": "scene flags primitive_types",
  "high": "code (word half)",
  "hit": "code (call site literal)",
  "instance_ray_types": "Features.shape_types",
  "instance_types": "Features.shape_types",
  "kind": "Features.look / settings exposure metering (display, meter modules: listed)",
  "layered": "Features.layered",
  "layered_coat": "Features.coat_layer / coat_normal",
  "layers": "Features layered lobes",
  "live_uv": "Features.live_material_uv",
  "meshes": "Features.shape_types",
  "ntree": "Features.trees",
  "ordered": "Features.ordered_traversal",
  "supersampled": "Features.spp > 1 (the OptiX supersampled image)",
  "prepared": "code (light source form)",
  "runtime": "code (call site literal: the run-time shadow sampling of a build reading `toggles.SOFT_SHADOWS`)",
  "scaled": "Features.scaled_normal_map",
  "shading": "code (layer shading form)",
  "shadow": "code (shadow ray form)",
  "shared_tree": "Features.trees",
  "slot": "Features.any_rgb / transparency (the light visibility call's slot after the layer program's)",
  "specular_layers": "Features.transmissive",
  "split": "code (shading split)",
  "sun_only": "code (light loop form)",
  "term_column": "Features.counters",
  "tinted": "Features.tinted_shadows",
  "toggle": "code (call site literal: the camera rays' or the reflection hits' cap)",
  "types": "Features.shape_types",
  "uv": "code (call site literal)",
  "visible": "Features.visibility / scene flags shared_visible",
  "wrapped": "Features.fuzz",
}
# The non-primitive factory argument types: Features projections, Warp structs and the device
# functions other factories build (closures: their compile inputs are those factories').
FACTORY_TYPES: Final = frozenset(
  {
    "Aim", "AlphaTest", "AreaVisibility", "Background", "Begin", "Body", "BodyAny", "BodyClosest", "BodyLeave", "BsdfEval",
    "BvhTrees", "Callable", "CameraModel", "Candidate", "CastRay", "CellLight", "CellRange", "CenterTrace", "Centre",
    "Columns", "Context", "CutAway", "CutMesh", "DirectLighting", "DirectLights", "End", "EvalSurface",
    "Evaluators", "Features", "FetchTexel", "FirstHitWriters", "GeometryOnly", "GeometryPixel",
    "GeometryRay", "Glossy", "LayerShade", "LightLoop", "LightVisibility", "LightVisibilityAt",
    "LookFlags", "N", "ParamsLayout", "PixelCone", "PlaneHit", "PlaneQueries", "PrimaryRay", "Queries",
    "RayQuery", "Remainder", "RenderLocate", "RenderPixel", "RenderPrepare", "SampleRayOf", "ShadeHit",
    "ShadeLayer", "ShadeProbeRay", "ShadeSurface", "ShadeSurfaceSplit", "ShadowPass", "ShadowRay",
    "Small", "Source", "ThroughAmbient", "TracePixel", "TraceSample", "TraceSlots", "TraceSpec", "Tracer", "TreeOn",
    "Trees", "type",
    "Unshadowed", "WriteHistory", "_TraceFlags", "Counts",
  }
)  # fmt: skip

# The other triggers: id -> (source, domain, bundle, pack, miss, action).
OTHER: Final[dict[str, tuple[str, str, str, str, str, str]]] = {
  # PTX key parts (`programs.ptx_parts`).
  "ptx program": (
    "code; a lens camera's model (`resolve.camera_model`); `shadows.area_rays` > 0 (the area pass)",
    "render / gi / area / render_<lens model> (distortion, bundle)",
    "render, gi: yes; area: PTX yes (`fill.area_key`), module not packed (created on the pass' first "
    "launch); lens models: no by default (`fill.fill(lens=True)`: `fill.lens_keys`)",
    "-",
    PTX_S,
    "-",
  ),
  "ptx repr(key_features(f))": (
    "the build's Features (rows above) but block_dim, max_registers, ordered_traversal (`programs.KEY_UNREAD`: "
    "no generated code reads them, BUG-404)",
    "the fill's 113 PTX (W10-REFILL)",
    "yes (routed)",
    "-",
    PTX_S,
    "see Features",
  ),
  "ptx tuple((getattr(scene, k) for k in _PTX_FLAGS))": (
    "scene flags (rows below)",
    "scene-free: f",
    "yes",
    "-",
    PTX_S,
    "-",
  ),
  "ptx supersampled": ("settings aa.spp > 1", "bool", "yes", "-", PTX_S, "see spp"),
  "ptx arch": ("device: PTX target min(sm, 89)", "sm_80 / sm_89", "yes (per bundle arch)", "-", PTX_S, "-"),
  "ptx wp.config.version": (
    "the Warp release (MuJoCo Warp not in the key: BUG-400)",
    "pinned",
    "yes (pinned release)",
    "-",
    PTX_S,
    "-",
  ),
  "ptx (wp.config.mode, wp.config.optimization_level, wp.config.verify_fp, wp.config.verify_cuda, wp.config.lineinfo)": (
    "Warp config (user process)",
    "defaults",
    "yes (defaults only)",
    "-",
    PTX_S,
    "document: a debug Warp config compiles",
  ),
  "ptx intrinsics.header_version(inc) if inc is not None else None": (
    "OptiX headers (bundled)",
    "pinned",
    "yes",
    "-",
    PTX_S,
    "-",
  ),
  # Scene flags: a scene-free build's are derived from its Features (`programs.scene_free_flags`).
  **{
    f"flags {k}": ("scene structures; scene-free: Features / bound GEO bits", "bool / types", "yes", "-", PTX_S, "-")
    for k in (
      "has_mesh",
      "has_box",
      "has_instances",
      "has_shared",
      "translucent_triangles",
      "translucent_curves",
      "shared_visible",
      "primitive_types",
      "shared_primitive_types",
      "has_world_curves",
      "has_shared_curves",
    )
  },
  "flags has_omm": ("scene: opacity micromaps", "bool", "yes (module option, not PTX)", "no: per word", MODULE_S, "-"),
  # Module record fields (`prebuilt.ModuleRecord`).
  "record has_curves": (
    "scene: curves traced (spheres, capsules) x word shape bits",
    "bool",
    "yes",
    "census words only",
    MODULE_S,
    "enumerate with the bound set",
  ),
  "record has_custom": (
    "scene: custom primitives x word shape bits",
    "bool",
    "yes",
    "census words only",
    MODULE_S,
    "enumerate with the bound set",
  ),
  "record opacity_micromaps": (
    "scene: micromaps",
    "bool",
    "yes",
    "census words only",
    MODULE_S,
    "enumerate with the bound set",
  ),
  "record max_registers": (
    "the tier family's fixed caps (`core/build/variants/registers.OPTIX_REGISTERS`) x word heaviness",
    "64-128",
    "every tier family's",
    "census words only",
    MODULE_S,
    "done (W14-HOST): internal per-tier constants, no setting",
  ),
  "record bound": (
    "the build word's bits and the output word (rows below)",
    "16-51 bits per build: 3.3e16",
    "no",
    "census words only (7 800)",
    MODULE_S,
    "the bound-set design (W9-ZERO-design.md)",
  ),
  # Software path.
  "software word": (
    "the software build word (#define, `toggles.preamble`)",
    "served class sets",
    "software bundle (not shipped)",
    "-",
    WARP_S,
    "W9-HUNT-D BUG-402",
  ),
  "mdl per-material": (
    "USD MDL materials: SDK compile + distill to the canonical BSDF (CPU, convert time)",
    "every MDL material",
    "n/a: data at render time",
    "-",
    "seconds per material, cached",
    "-",
  ),
  "constant wp.constant": ("module-level code", "fixed", "yes (in the module hash)", "-", WARP_S, "-"),
  "relight trees": (
    "scene trees in the GI relight kernel (`software_view`)",
    "2^5 x builds: 1 604 modules",
    "yes (listed), 2.3 GB",
    "-",
    WARP_S,
    "owned by W9-RELIGHT",
  ),
}


def bit_row(name: str) -> tuple[str, str, str, str, str, str]:
  """The row of build-word bit `name` (`triggers.word_bits`): every bit is a module key input."""
  kind = name.split(" ")[0]
  source = {
    "class": "scene content (a feature class used)",
    "field": "scene content (a material / map field used)",
    "bound": "scene properties and GI / env settings",
    "setting": "settings (shadows, fog, culling, translucent layers)",
    "shape": "scene shape types",
    "sampling": "settings (samples, denoiser, anisotropy) in *_sampled",
  }.get(kind, "scene layout (structures, poses, visibility, planes, deformables)")
  return (source, "on / off", "PTX: yes", "census words only", MODULE_S, "measured per bit: bound set or run-time data")


def feature_coverage(field: str, fill: dict[str, set[str]], holes: dict[str, int]) -> str:
  """The bundle coverage of Features field `field`: "yes (run time)" where a tier build reads it
  at run time (scene-free, switch, word bit), "yes (tier axis)" where the fill builds several
  values of it, "partial" / "no" where the route walk finds configurations it sends outside."""
  from raywarp.core.build.variants import tiers

  runtime = tiers.SCENE_FREE | tiers.BOUND_FREE | set(tiers.SWITCHES) | set(tiers.BOUND_FIELDS) | set(tiers.FIELD_BITS)
  how = "run time" if field in runtime else "tier axis" if len(fill.get(field, ())) > 1 else "fixed"
  if not holes.get(field):
    return f"yes ({how})"
  return f"partial ({how}; {holes[field]} hole points)" if how != "fixed" else f"no ({holes[field]} hole points)"


def render(derived: dict[str, set[str]], sources: dict[str, set[str]]) -> str:
  """The inventory as markdown: Features fields (sources from the code, coverage computed), the
  factory inputs, the PTX / module / word rows."""
  from tools.prebuilt import zero

  walk = json.loads(Path(zero.__file__).with_name("zero_holes.json").read_text())
  holes: dict[str, int] = {}
  for h in walk["holes"]:
    for f in h["why"]:
      holes[f] = holes.get(f, 0) + 1
  fill: dict[str, set[str]] = {}
  for f in zero.fill_builds().values():
    for k, v in vars(f).items():
      fill.setdefault(k, set()).add(repr(v))
  head = "| trigger | source (from the code) | domain | bundle | module pack | miss | action |\n|---|---|---|---|---|---|---|"
  lines = [
    "# Compile-trigger inventory (generated: `python -m tools.prebuilt.triggers report "
    "internal/research/kb/COMPILE_TRIGGERS.md`)",
    "",
  ]
  lines += [
    f"Route walk: {walk['points']} points, {walk['served']} served, {len(walk['holes'])} holes.",
    "",
    "## Features fields (PTX key)",
    "",
    head,
  ]
  for field in sorted(derived["features"]):
    src = ", ".join(sorted(sources.get(field, ()))) or "routing"
    cov = feature_coverage(field, fill, holes)
    lines.append(f"| {field} | {src} | {FEATURES.get(field, BOOL)} | {cov} | per word | {PTX_S} | {ACTIONS.get(field, '-')} |")
  lines += ["", "## Device-code factory inputs not named after a Features field", "", "| argument | source |", "|---|---|"]
  lines += [f"| {k} | {v} |" for k, v in sorted(FACTORY_INPUTS.items())]
  lines += ["", "## OptiX keys, words, software, MDL", "", head]
  for key in sorted(OTHER):
    lines.append(f"| {key} | " + " | ".join(OTHER[key]) + " |")
  for name in sorted(derived["word"]):
    lines.append(f"| word {name} | " + " | ".join(bit_row(name)) + " |")
  return "\n".join(lines) + "\n"
