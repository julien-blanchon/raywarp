"""The zero-compile route model: every scene a renderer can be given, as the values the kernel
resolution reads of it, and the check that a configuration's kernels are a prebuilt fill point.

`resolve.resolve_features` reads a scene only through a fixed set of values: its capabilities
(`scene.Capabilities`), light count and shadow casters, environment, headlight, deformables,
material inputs (derive inputs, tables), instance tables (optics, two-sided, shadow
transmittance), the per-world declaration, the sharing layout and the cameras. `Scene` is that set,
each value over its whole domain (booleans; counts by the thresholds the code compares them to).
Whatever adapter made a scene (MuJoCo, mjlab, Newton, Isaac Lab, USD, glTF), its kernels are those
of its `Scene` point: `stub` builds a `GpuScene` carrying exactly these values (over a plain base
scene, `feature_space.base_scenes`), so the real resolution and routing run on it, on the CPU,
in about a millisecond.

`FillModel` holds the OptiX fill's PTX keys (`fill._layout_points` over the stubs of the fill's
base scenes: its own enumeration of tiers, GI, class sets and output forms) and `Probe.route`
resolves one configuration on one `Scene` point to its build and key: a configuration compiles
nothing when its route lands on a tier and its key is a fill key.

  python -m tools.prebuilt.zero holes [--write]     # the depth-1 walk's holes (zero_holes.json)
  python -m tools.prebuilt.zero sample 120 S.json   # a stratified sample of the proven space
  python -m tools.prebuilt.zero walk 2 3/8 P3.json  # shard 3 of 8 of the pair walk (2 values)
  python -m tools.prebuilt.zero random 20000 7 3-10 R7.json  # 20 000 random points of 3 to 10 axes
  python -m tools.prebuilt.zero lattice 3/64 L3.json  # shard 3 of 64 of the tier-selection lattice
  python -m tools.prebuilt.zero merge P.json P*.json  # the shards merged, holes by axes
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from tools.prebuilt import fill

if TYPE_CHECKING:
  from raywarp._base.dims import Ids, Mask
  from raywarp.core.build.variants.resolve import Resolution
  from raywarp.core.kernels.abi.features import Features
  from raywarp.core.pipeline.gpu_scene.gpu_scene import Capabilities, GpuScene
  from raywarp.settings import RenderSettings
  from tools.prebuilt.probe import Resolver

# The GPU of the OptiX bundle's census (`fill.PROFILES`).
PROFILE: Final = fill.PROFILES["optix"]
# The per-world declared fields the resolution reads (`resolve.py`: `"..." in declared`), each a
# compile-relevant declaration; every other declared field is run-time data.
DECLARED: Final = (
  "camera.distortion",
  "fog.range",
  "instance.material",
  "instance.shading",
  "instance.visible",
  "light.cast_shadow",
  "light.radius",
  "light.shape",
  "light.type",
  "material.alpha_cutoff",
  "material.base_diffuse_roughness",
  "material.coat_weight",
  "material.coat_ior",
  "material.coat_color",
  "material.emission_luminance",
  "material.fuzz",
  "material.fuzz_color",
  "material.mirror",
  "material.specular_anisotropy",
  "material.specular_ior",
  "material.specular_weight",
  "material.tex_affine",
  "material.texture",
  "material.uv_world",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Scene:
  """Every value of a scene the kernel resolution reads (`stub` makes the scene).

  Attributes:
    nlight: lights (the light loop's form: 0, 1-4 small, 8+ general with the grid by default).
    casters: lights that can cast a shadow (the shadow budget compares its cap to it).
    caps: the content's capabilities (`scene.Capabilities`; shape types, materials, lights).
    env_lit: an environment map lights the scene; env_maps: any environment map.
    headlight: the scene authors a camera headlight (MuJoCo).
    flex: deformables: 0 none, 2 surfaces (cloth, volumes), 1 segments (cables).
    flex_material: a deformable has a material.
    medium: a rendered instance has a solid-glass optics row.
    twosided: a rendered instance is two-sided.
    transmit: shadow transmittance: 0 opaque casters, 1 gray, 2 tinted.
    material_uv: live material UV tables (MuJoCo: `material_uv`).
    curve: a material has the Unreal specular curve or a dielectric specular level flag.
    weight: a canonical material's specular weight differs from 1 (or its IOR scale is 0).
    base_lit: a material's per-world luminance needs an emission tint map.
    diffuse_rows: the scene has a per-material diffuse-roughness table.
    diffuse_role: the material texture table has the diffuse-roughness role column.
    declared: the per-world declared fields (`DECLARED` names) and `alpha`.
    alpha: `PerWorld.alpha` (per-world alpha of a declared color table).
    shared, world: the layout's shared instances, per-world instances (`sharing.Layout`).
    planes: rendered analytic planes.
    compact, visible: compact pose columns, a per-world visibility table.
    bundle, distortion: the cameras' ray bundles, lens distortion.
    uniform: every camera at one resolution.
  """

  nlight: int = 1
  casters: int = 1
  caps: Capabilities | None = None  # None: plain meshes
  env_lit: bool = False
  env_maps: bool = False
  headlight: bool = False
  flex: int = 0
  flex_material: bool = False
  medium: bool = False
  twosided: bool = False
  transmit: int = 0
  material_uv: bool = False
  curve: bool = False
  weight: bool = False
  base_lit: bool = False
  diffuse_rows: bool = False
  diffuse_role: bool = False
  declared: frozenset[str] = frozenset()
  alpha: bool = False
  shared: bool = True
  world: bool = False
  planes: int = 0
  compact: bool = True
  visible: bool = False
  bundle: bool = False
  distortion: bool = False
  uniform: bool = True


@functools.cache
def _base() -> GpuScene:
  """The container scene the stubs copy (a plain base scene of the fill, on the CPU)."""
  from tools.prebuilt import feature_space
  from tools.prebuilt import probe as harness

  point = next(p for p in feature_space.base_scenes() if p.nlight == 1 and p.planes == 0 and p.geometry == "mesh")
  scene, _cams, _ = harness.load_point(point.code())
  return scene


def capabilities(**fields: object) -> Capabilities:
  """`scene.Capabilities` of plain meshes with `fields` (each an attribute of it)."""
  from raywarp._base.types import ShapeType
  from raywarp.core.pipeline.gpu_scene.gpu_scene import Capabilities

  return Capabilities(**{"shape_types": (int(ShapeType.MESH),), **fields})  # pyrefly: ignore[bad-argument-type]


def stub(point: Scene) -> GpuScene:
  """A `GpuScene` whose every value the resolution reads is `point`'s (a shallow copy of the base
  scene: the arrays the resolution never reads are the base's)."""
  import warp as wp

  from raywarp._base.types import ROW_MEDIUM, TWO_SIDED, ShadingFlag, ShapeType, TexRole

  base = _base()
  s = copy.copy(base)
  caps = point.caps if point.caps is not None else capabilities()
  types_ = [int(ShapeType.MESH)] + [int(ShapeType.PLANE)] * point.planes
  n = len(types_)
  with wp.ScopedDevice("cpu"):
    s.instance_type = wp.array(np.asarray(types_, np.int32), dtype=int)
  s.ninstance = n
  s.rendered = np.arange(n).astype(np.int32)
  s.nlight = point.nlight
  s.capabilities = caps
  s.environment = dataclasses.replace(base.environment, lit=point.env_lit, has_maps=point.env_maps)
  s.headlight = dataclasses.replace(base.headlight, active=point.headlight)
  optics = np.zeros((n, *base.instance_optics.shape[1:]), np.float32)
  optics[0, ROW_MEDIUM] = 1.5 if point.medium else 0.0
  s.instance_optics = optics
  s.instance_twosided = np.full(n, TWO_SIDED if point.twosided else 0, np.int32)
  transmit = {0: (0.0, 0.0, 0.0), 1: (0.5, 0.5, 0.5), 2: (0.8, 0.2, 0.1)}[point.transmit]
  s.instance_shadow_transmit = np.tile(np.asarray(transmit, np.float32), (n, 1))
  s.material_uv = types.SimpleNamespace() if point.material_uv else None  # pyrefly: ignore[bad-assignment]
  dim = np.asarray([point.flex] if point.flex else [], np.int32)
  s.deformables = types.SimpleNamespace(nprim=1 if point.flex else 0, dim=dim, has_material=point.flex_material, count=len(dim))  # pyrefly: ignore[bad-assignment]
  curve = int(ShadingFlag.SPECULAR_CURVE) if point.curve else 0
  values = dict(base.derive_inputs.values)
  values["material.specular_weight"] = np.asarray([0.5 if point.weight else 1.0], np.float32)
  s.derive_inputs = types.SimpleNamespace(  # pyrefly: ignore[bad-assignment]
    shading=np.asarray([curve], np.int32),
    values=values,
    ior_scale=np.asarray([1.0], np.float64),
    base_lit=np.asarray([point.base_lit]),
  )
  tables = dict(base.tables)
  tables.pop("material.base_diffuse_roughness", None)
  if point.diffuse_rows:
    tables["material.base_diffuse_roughness"] = np.zeros((1, 1), np.float32)
  texture = np.zeros((1, 1, int(TexRole.DIFFUSE_ROUGHNESS) + (1 if point.diffuse_role else 0)), np.int32)
  tables["material.texture"] = texture
  s.tables = tables  # pyrefly: ignore[bad-assignment]
  return s


@dataclasses.dataclass(frozen=True)
class StubLayout:
  """The `sharing.Layout` values the resolution and the scene flags read (`_layout`)."""

  nshared: int
  planes: Ids[int]
  compact: bool
  nvisible: int
  visible_slot: Ids[int]
  per_world: Mask[int]
  shared_ids: Ids[int]


@dataclasses.dataclass(frozen=True)
class StubRenderer:
  """What `probe.scene_flags` reads of a renderer: the stub scene and its layout."""

  scene: GpuScene
  layout: StubLayout


def _layout(point: Scene) -> StubLayout:
  """The `sharing.Layout` values the resolution and the scene flags read."""
  n = 1 + point.planes
  per_world = np.zeros(n, bool)
  per_world[0] = point.world
  # The per-world visibility table declares the mesh (instance 0): shared, it makes the scene's
  # shared instances per-world visible (`OptixScene.shared_visible`).
  visible_slot = np.full(n, -1, np.int32)
  if point.visible:
    visible_slot[0] = 0
  return StubLayout(
    nshared=1 if point.shared else 0,
    planes=np.zeros(0, np.int32),
    compact=point.compact,
    nvisible=1 if point.visible else 0,
    visible_slot=visible_slot,
    per_world=per_world,
    shared_ids=np.asarray([0] if point.shared else [], np.int32),
  )


@dataclasses.dataclass
class Probe:
  """One `Scene` point resolved like `probe.Probe` (the fill's), with a memo of tier builds per
  class-set table (`resolve_features(tier_builds=)`)."""

  name: str
  point: Scene
  memo: dict[object, dict] = dataclasses.field(default_factory=dict)

  @functools.cached_property
  def scene(self) -> GpuScene:
    """The stub scene."""
    return stub(self.point)

  @functools.cached_property
  def renderer(self) -> StubRenderer:
    """What `probe.scene_flags` reads of a renderer (the scene and its layout)."""
    return StubRenderer(self.scene, _layout(self.point))

  def resolution(self, settings: RenderSettings, profile: str) -> Resolution:
    """The resolution of `settings` on `profile`'s GPU (ValueError: settings the scene cannot
    honor), as `probe.Probe.resolution`."""
    from raywarp.core.build.variants import class_sets
    from raywarp.core.build.variants.resolve import RenderCaps, resolve_features
    from tools.prebuilt import probe as harness

    p = self.point
    caps = RenderCaps(
      scene=self.scene,
      layout=_layout(p),  # pyrefly: ignore[bad-argument-type]
      ncam=1,
      bundle=p.bundle,
      distortion=p.distortion,
      uniform=p.uniform,
    )
    decl = types.SimpleNamespace(fields=lambda: sorted(p.declared), alpha=p.alpha) if p.declared or p.alpha else None
    memo = self.memo.setdefault((class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED), {})
    device = harness.device_caps(profile)
    return resolve_features(settings, caps, decl, device, casters=p.casters, tier_builds=memo)  # pyrefly: ignore[bad-argument-type]


@dataclasses.dataclass(frozen=True)
class Route:
  """One configuration's kernels: its tier (None: outside every tier), the fields that leave the
  nearest tier, its build's PTX key (render programs; a lens camera's: "<render key>+<its lens
  model's key>", `lens_key`) and the build."""

  tier: str | None
  outside: tuple[str, ...]
  key: str | None
  build: Features | None


def key_of(probe: Resolver, f: Features, settings: RenderSettings) -> str:
  """The OptiX PTX key of build `f` (`fill._build`'s: the target of the census GPU)."""
  return fill._build(probe, f, settings, fill_arch()).key


def lens_key(build: fill.Build, model: str) -> str:
  """The key of a lens camera's programs: the build's render PTX and its lens model's
  (`programs.build_lens`, `fill.lens_keys`), both of which the fill must hold."""
  return f"{build.key}+{fill.lens_keys(build)[model]}"


@functools.cache
def fill_arch() -> int:
  """The SM version of the census GPU."""
  from tools.prebuilt import probe as harness

  return harness.PROFILES[PROFILE][0]


def route(probe: Probe, settings: RenderSettings) -> Route:
  """The route of `settings` on the probe's scene (OptiX; ValueError: settings it cannot honor)."""
  s = settings.replace(**{"performance.traversal": "optix"})
  r = probe.resolution(s, PROFILE)
  if r.choice is None or r.choice.tier is None:
    outside = r.choice.outside if r.choice is not None else ("performance.kernels",)
    return Route(None, outside, None, r.features)
  build = fill._build(probe, r.features, s, fill_arch())
  key = build.key if r.camera.name == "pinhole" else lens_key(build, r.camera.name)
  return Route(r.choice.tier, (), key, r.features)


# --- The fill ------------------------------------------------------------------------------------


def base_points() -> Iterator[tuple[str, Scene]]:
  """The fill's base scenes as `Scene` points (`feature_space.base_scenes`: plain opaque meshes lit
  by point lights; "pointfill": two of the lights cast shadows), in the shared layout: on OptiX
  the geometry, layout and planes are bound (`tiers.BOUND_FREE`), checked by
  `test_zero_compile.py::test_optix_fill_keys_ignore_geometry_and_layout`."""
  from tools.prebuilt import feature_space

  for p in feature_space.base_scenes():
    if p.geometry != "mesh" or p.planes:
      continue
    casters = 2 if p.lights == "pointfill" else p.nlight
    yield p.code(), Scene(nlight=p.nlight, casters=casters)


@functools.cache
def fill_builds() -> dict[str, Features]:
  """Every OptiX fill point's build by PTX key (`fill._layout_points` on `base_points`)."""
  from raywarp.core.build.variants import class_sets

  out: dict[str, Features] = {}
  f = fill._Fill(backend="optix", profile=PROFILE, arch=fill_arch(), modules=False, only=None)
  tables = class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED
  sets = fill._sets("mesh", "shared", "optix")
  try:
    for name, point in base_points():
      for _row, build in fill._layout_points(f, Probe(name, point), "shared", sets):
        if build is not None:
          out[build.key] = build.features
  finally:
    class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED = tables
  return out


def fill_keys() -> frozenset[str]:
  """The PTX keys of every OptiX fill point."""
  return frozenset(fill_builds())


def digest(keys: frozenset[str]) -> str:
  """A short digest of a key set (reports)."""
  return hashlib.sha1("\n".join(sorted(keys)).encode()).hexdigest()[:12]


def main(argv: list[str]) -> int:
  """CLI: `holes` prints the exhaustive check's holes; `sample N OUT` writes a stratified sample
  of the proven space (`zero_space.sample`, the GPU check's `tiers_fuzz.py --sample`)."""
  from tools.prebuilt import zero_space

  if argv[:1] == ["holes"]:
    return zero_space.report(str(Path(__file__).with_name("zero_holes.json")) if "--write" in argv else None)
  if argv[:1] == ["walk"] and len(argv) == 4:  # walk DEPTH I/N OUT.json: one shard, exactly DEPTH values
    i, n = (int(x) for x in argv[2].split("/"))
    tally = zero_space.walk(int(argv[1]), shard=(i, n), exact=True)
    Path(argv[3]).write_text(json.dumps(zero_space.holes_file(tally), indent=1) + "\n")
    print(f"shard {i}/{n}: points {tally.points}, refused {tally.refused}, holes {len(tally.holes)}")
    return 0
  if argv[:1] == ["random"] and len(argv) == 5:  # random N SEED KMIN-KMAX OUT.json: K random axes per point
    kmin, kmax = (int(x) for x in argv[3].split("-"))
    tally = zero_space.random_walk(int(argv[1]), (kmin, kmax), seed=int(argv[2]))
    Path(argv[4]).write_text(json.dumps(zero_space.holes_file(tally), indent=1) + "\n")
    print(f"random {argv[1]} seed {argv[2]}: points {tally.points}, refused {tally.refused}, holes {len(tally.holes)}")
    return 0
  if argv[:1] == ["lattice"] and len(argv) == 3:  # lattice I/N OUT.json: shard I of N of the lattice walk
    i, n = (int(x) for x in argv[1].split("/"))
    tally = zero_space.lattice_walk((i, n))
    Path(argv[2]).write_text(json.dumps(zero_space.holes_file(tally), indent=1) + "\n")
    print(f"lattice {i}/{n}: points {tally.points}, refused {tally.refused}, holes {len(tally.holes)}")
    return 0
  if argv[:1] == ["merge"] and len(argv) > 2:  # merge OUT.json SHARD.json...
    merged = zero_space.merge(argv[2:], argv[1])
    print(json.dumps({k: v for k, v in merged.items() if k not in ("holes", "anomalies")}, indent=1))
    return 0
  if argv[:1] == ["sample"] and len(argv) == 3:  # sample N OUT.json
    Path(argv[2]).write_text(json.dumps(zero_space.sample(int(argv[1])), indent=1, default=str))
    return 0
  print(__doc__)
  return 2


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
