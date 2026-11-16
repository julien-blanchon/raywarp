"""The raywarp code a prebuilt OptiX program depends on: the one list of kernel-generating code.

A build's PTX is valid for the code it was generated from: `digest(reach)` of the kernel-code
functions its generation used (`Reach`) plus every kernel-code module's other code (imports,
constants, classes, module-level code) and the host constants kernel code reads (`HOST_CONSTANTS`,
by value). A PTX cache entry and a bundle index entry carry their reach and digest
(`prebuilt.Entry`); an entry whose digest differs from the installed code's is not used. The bundle
fill records each build's reach (`tools/prebuilt/reach.py`), so a kernel edit invalidates only the
builds whose generation used the edited function; a build compiled at run time records every
function (`ALL`). The kernel code: the modules whose code reaches the OptiX programs' PTX
(`PTX_PACKAGES`, `PTX_MODULES`: the render and GI kernels, shading, the light loop, the traversal,
the programs and their launch parameters, the device preamble). Everything else is host code:
routing data (`class_sets.py`, `tiers.py`), resolve, upload, the scene loaders, the adapters,
settings beyond those constants. Its edits keep every entry, so a host fix or a new class set keeps
the bundle. What a build's code depends on beyond the code (its Features, scene flags, output scale,
target, the Warp / MuJoCo Warp releases) is in its key (`programs.ptx_key`).

The other modules that define device code (`WARP_MODULES`: structure builds, uploads, sensors, the
adapters' bindings) compile as plain Warp modules, which Warp's kernel cache names by a content
hash of their generated code: a bundled entry serves exactly its code, nothing needs a digest.

`tests/contracts/test_kernel_code.py` keeps the list whole: every module defining device code
(`wp.func`, `wp.kernel`, `wp.struct`, a native builtin, CUDA source) is in the PTX list or in
`WARP_MODULES`; every name a PTX module imports from another raywarp module is in `HOST_CONSTANTS`
(data, hashed by value) or `HOST_ONLY` (functions and classes used on the host); the OptiX
programs' generated code calls functions of the PTX list only.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Final

from raywarp.cache.code_version import CODE_SUFFIXES, code_files, code_units

# The packages whose every module is kernel code (device functions, their factories and constants),
# subpackages included: the device band of `core` (tracing, shading, the display passes in the
# kernels, the megakernel assembly and its ABI).
PTX_PACKAGES: Final = (
  "raywarp.core.kernels",
  "raywarp.core.post.display",
  "raywarp.core.post.exposure",
  "raywarp.core.post.grading",
  "raywarp.core.post.temporal",
  "raywarp.core.shade",
  "raywarp.core.trace.accel",
  "raywarp.core.trace.optix.device",
  "raywarp.core.trace.rays",
)
# Kernel-code modules outside those packages: the shape types.
PTX_MODULES: Final = ("raywarp._base.types",)
# Modules with device code that never reaches the OptiX programs: plain Warp modules, named in
# Warp's kernel cache by their generated code's hash.
WARP_MODULES: Final = frozenset({
  "raywarp.core.outputs.camera_export",
  "raywarp.core.outputs.ground_truth",
  "raywarp.core.outputs.pointcloud",
  "raywarp.core.pipeline.baked",
  "raywarp.core.pipeline.cameras",
  "raywarp.core.pipeline.platform.cadence",
  "raywarp.core.pipeline.worlds.active",
  "raywarp.core.pipeline.derive.kernels",
  "raywarp.core.pipeline.worlds.optics",
  "raywarp.core.pipeline.worlds.tables",
  "raywarp.core.pipeline.worlds.validator_kernels",
  "raywarp.core.post.aa.fxaa",
  "raywarp.core.post.sensor.effects",
  "raywarp.core.post.sensor.inputs",
  "raywarp.core.trace.optix.accel",
  "raywarp.core.trace.optix.dynamic",
  "raywarp.core.trace.optix.instances",
  "raywarp.core.trace.optix.mesh_inputs",
  "raywarp.core.trace.optix.omm.kernels",
  "raywarp.core.trace.optix.supersample",
  "raywarp.integrations.isaaclab.kernels",
  "raywarp.integrations.mujoco.capabilities",
  "raywarp.integrations.mujoco.hfield",
  "raywarp.integrations.mujoco.render_context",
  "raywarp.integrations.mujoco.skybox",
  "raywarp.integrations.newton.binding",
  "raywarp.integrations.newton.deformables",
  "raywarp.integrations.newton.headless.frames",
  "raywarp.integrations.newton.sensor.kernels",
  "raywarp.video.colour",
})  # fmt: skip
# Data that kernel code reads from host modules (enum codes, colour constants): module -> names,
# hashed by value (`repr`), so their modules stay host code.
HOST_CONSTANTS: Final = {
  "raywarp.core.trace.optix.api": ("VERSION",),
  "raywarp.scene.images.colorspace": ("REC709_Y",),
  "raywarp.scene.lighting.ies_profile": ("COARSE_KNOTS", "KIND_C"),
  "raywarp.scene.lighting.photometry": (
    "DAYLIGHT_RANGE",
    "DAYLIGHT_SPLIT",
    "DAYLIGHT_X_HIGH",
    "DAYLIGHT_X_LOW",
    "PLANCKIAN_RANGE",
    "PLANCKIAN_U",
    "PLANCKIAN_V",
  ),
  "raywarp.settings.fields": ("LAYOUTS", "MISS_DEPTHS", "REFLECTION_SHADOWS"),
  "raywarp.settings.imaging": ("ADAPTATION", "METERING"),
}
# Host objects (functions, classes, modules) that kernel-code modules import and use on the host
# only (bakes, caches, launch helpers, device queries, the settings types an output word is
# derived from: the word is run-time data, a module's bound value): module -> names ("*": the
# module itself, imported whole).
HOST_ONLY: Final = {
  "raywarp._base.records": ("record_field",),
  "raywarp._base.warp_structs": ("struct_mirror", "warp_struct"),
  "raywarp._native": ("OPTIX_INCLUDE_DIR",),
  "raywarp.core.build.variants.tiers": ("*",),
  "raywarp.core.pipeline.frame": ("Frame",),
  "raywarp.core.pipeline.gpu_scene.deformables": ("surface_triangles",),
  "raywarp.core.pipeline.platform.bake_cache": ("*",),
  "raywarp.core.pipeline.platform.cadence": ("Cadence",),
  "raywarp.core.pipeline.platform.device": ("device_caps", "host_copy", "launch_bands"),
  "raywarp.core.pipeline.platform.launches": ("CachedLaunch",),
  "raywarp.core.pipeline.textures.uploader": ("MipTexture2D", "box_chain"),
  "raywarp.core.trace.optix.api": ("*",),
  "raywarp.core.trace.optix.strides": ("misread",),
  "raywarp.scene.images.latlong": ("latlong_directions", "latlong_uv", "texel_solid_angles"),
  "raywarp.scene.lighting.ies_profile": ("*", "IesGrid"),
  "raywarp.settings": ("GI",),
  "raywarp.settings.outputs": (
    "DepthOutput",
    "FloatOutput",
    "LuminanceOutput",
    "NormalOutput",
    "OutputSpec",
    "Outputs",
    "ShadowOutput",
    "is_kind",
  ),
}


def ptx_modules() -> tuple[str, ...]:
  """Every kernel-code module (`PTX_PACKAGES`' modules, `PTX_MODULES`): dotted names, sorted."""
  dotted = set(PTX_MODULES)
  for name, _ in code_files(*PTX_PACKAGES):
    if name.endswith(".py"):
      package, rel = name.split("/", 1)
      dotted.add(".".join((package, *rel.removesuffix(".py").split("/"))).removesuffix(".__init__"))
  return tuple(sorted(dotted))


# The kernel-code functions a build's code generation used, as `units` names
# ("<file>:<function>"); `ALL`: every function (a build compiled at run time records no reach).
type Reach = frozenset[str]
ALL: Final[Reach] = frozenset({"*"})


def sources() -> dict[str, Path]:
  """Every kernel-code file by name (`code_files`' names: "<package>/<path>" of the packages'
  files, "<module path>.py" of `PTX_MODULES`): its path."""
  out: dict[str, Path] = {}
  for package in PTX_PACKAGES:
    spec = importlib.util.find_spec(package)
    roots = None if spec is None else spec.submodule_search_locations
    if not roots:
      raise ModuleNotFoundError(f"{package} (kernel_code.PTX_PACKAGES) is not a package")
    for root in map(Path, roots):
      out |= {f"{package}/{f.relative_to(root).as_posix()}": f for f in sorted(root.rglob("*")) if _code(f)}
  for name in PTX_MODULES:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
      raise ModuleNotFoundError(f"{name} (kernel_code.PTX_MODULES) is not a module")
    out[name.replace(".", "/") + ".py"] = Path(spec.origin)
  return out


def _code(file: Path) -> bool:
  """Whether `file` is a code file `code_files` reads."""
  return file.suffix in CODE_SUFFIXES and "__pycache__" not in file.parts and file.is_file()


def _files() -> list[tuple[str, bytes]]:
  """(name, content) of the kernel-code files (`sources`)."""
  return [(name, path.read_bytes()) for name, path in sorted(sources().items())]


def _constants() -> list[tuple[str, bytes]]:
  """("<module>:<name>", repr bytes) of every `HOST_CONSTANTS` value."""
  return [
    (f"{module}:{name}", repr(getattr(importlib.import_module(module), name)).encode())
    for module, names in HOST_CONSTANTS.items()
    for name in names
  ]


@functools.cache
def units() -> dict[str, str]:
  """The kernel code's parts and their digests (`code_units` of `sources`: comments, docstrings
  and formatting do not count): "<file>:<function>" of each top-level function, "<file>:" of the
  rest of each module, the other files whole."""
  return code_units(_files())


def function(unit: str) -> bool:
  """Whether `units` name `unit` is a top-level function (a part a reach selects)."""
  return not unit.endswith(":") and ":" in unit


def scope(reach: Reach, table: dict[str, str]) -> list[str]:
  """The parts of code `table` (`units`) a build of reach `reach` depends on: the functions in it
  (`ALL`: every one) and every other part."""
  every = reach == ALL
  return sorted(u for u in table if every or not function(u) or u in reach)


def digest(reach: Reach, table: dict[str, str] | None = None) -> str:
  """The digest of the kernel code a build of reach `reach` depends on: its `scope` of the
  installed code's parts (`table`: another tree's, `units` of its files), a function no longer
  defined counting as gone, and the host constants' values."""
  table = units() if table is None else table
  parts = [(u, table[u]) for u in scope(reach, table)]
  gone = [] if reach == ALL else sorted(u for u in reach if u not in table)
  h = hashlib.blake2b(digest_size=12)
  for name, part in [*parts, *((u, "gone") for u in gone)]:
    h.update(f"{name}\0{part}\0".encode())
  for name, value in _constants():
    h.update(f"{name}\0".encode() + value + b"\0")
  return h.hexdigest()


def version() -> str:
  """The digest of all the kernel code (`digest(ALL)`)."""
  return digest(ALL)
