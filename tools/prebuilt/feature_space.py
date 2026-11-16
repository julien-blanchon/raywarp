"""Synthetic scenes covering the renderer's feature space, for the bundle-vs-own speed parity check.

A `Point` picks one value per axis: the light kinds and count, the material model, the texture
roles, the geometry kinds, instancing, an environment and the camera headlight. `build` makes its
scene (a `rw.SceneBuilder` scene; height fields go through MuJoCo, the only height-field
source). `catalogue` is the stratified sample the check renders: every feature-class combination
(`raywarp.core.build.variants.tiers.feature_class`: surface / glass, coat, layers, headlight,
primitives) once, the light counts (so the light-loop forms) and the other axes spread over it
round-robin, plus the extremes (everything at once, the leanest scene).
The prebuilt kernels' fill points are `base_scenes` (`tools/prebuilt/fill.py`; loaded by
`tools/prebuilt/probe.load_point`); the development tree registers every point as a benchmark scene
and times the parity check over them (internal/benchmarks/feature_space_parity.py).

Nothing here is imported by `raywarp`.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

import raywarp as rw
from raywarp.scene import (
  AlphaMode,
  EnvironmentLight,
  LightKind,
  Material,
  OpenPbrExtras,
  RaywarpExt,
  Shaping,
  Slot,
  TextureRef,
)
from raywarp.scene.model.types import MeshId

HERE = Path(__file__).resolve().parent
IES = HERE.parents[1] / "tests" / "data" / "scenes" / "ies" / "wallwash_2002.ies"

LIGHT_KINDS: Final = ("point", "spot", "rect", "disk", "sphere", "tube", "ies", "sun", "env")
LIGHT_COUNTS: Final = (1, 2, 4, 8, 64)
# Material models (the glass / translucent / cut-out ones drive the surface and glass classes).
MATERIALS: Final = ("plain", "eon", "glass", "thin", "cutout")
TEXTURES: Final = ("none", "base", "pbr", "normal")
GEOMETRY: Final = ("mesh", "prim", "hfield")
# The geometries built through MuJoCo (`build_mjcf`): height fields (its only source), and the
# bundle fill's MuJoCo copies of the base scenes ("mjmesh": a MuJoCo binding aliases its poses, so
# its shared instances read world 0's row, `Features.compact_poses` off; "flex": with a deformable
# cloth, "flexplain": the cloth without a material, `Features.flex_materials` off).
MJCF_GEOMETRY: Final = ("hfield", "mjmesh", "flex", "flexplain")
# Analytic planes of a scene (`Features.planes` none / some): the floor as a mesh quad, the floor
# plane; 6 (the floor plane and five more: walls, a ceiling, a lower floor) in the timing sample's
# extremes only (the plane loop over several planes).
PLANE_COUNTS: Final = (0, 1)
MANY_PLANES: Final = 6
# The light loop's form for each light count: the small table sized 1 to 4, the general loop.
LOOP_OF_COUNT: Final = {1: "small1", 2: "small2", 3: "small3", 4: "small4", 8: "general", 64: "general"}
# Floor, a 3 x 3 m stage; views (eye, target, fovy) around it, four for the multi-camera configs.
VIEWS: Final = (
  ((0.0, -4.2, 2.4), (0.0, 0.0, 0.5), 55.0),
  ((3.6, -2.4, 1.8), (0.0, 0.0, 0.5), 60.0),
  ((-3.2, 2.8, 2.6), (0.0, 0.0, 0.4), 55.0),
  ((0.3, -1.6, 4.2), (0.0, 0.0, 0.0), 70.0),
)


@dataclasses.dataclass(frozen=True, slots=True)
class Point:
  """One synthetic scene of the feature space.

  Attributes:
    lights: light kinds ("mix": every kind of `LIGHT_KINDS` in turn), placed over the stage.
    nlight: lights (an "env" light is the environment: it counts once).
    material: the props' material model (`MATERIALS`); the floor stays plain.
    coat: some props add the clear coat.
    layers: some props add the layered lobes (anisotropy, fuzz).
    textures: texture roles of the props and floor (`TEXTURES`).
    geometry: "mesh" (meshes on a plane), "prim" (+ boxes, spheres, capsules, ellipsoids,
      cylinders), "hfield" (a MuJoCo height field under the props); the base scenes' MuJoCo copies
      "mjmesh" and "flex" (`MJCF_GEOMETRY`).
    instancing: the props as many instances of one mesh (else a few distinct meshes).
    env: a sky environment besides the lights.
    headlight: the camera headlight.
    planes: analytic planes (`PLANE_COUNTS`; height-field scenes have none).
  """

  lights: str = "point"
  nlight: int = 1
  material: str = "plain"
  coat: bool = False
  layers: bool = False
  textures: str = "none"
  geometry: str = "mesh"
  instancing: bool = False
  env: bool = False
  headlight: bool = False
  planes: int = 1

  def code(self) -> str:
    """The scene's registry name (`parse` inverts it)."""
    named = (
      ("c", self.coat),
      ("l", self.layers),
      ("i", self.instancing),
      ("e", self.env),
      ("h", self.headlight),
      ("n", self.planes == 0),
      ("m", self.planes > 1),
    )
    flags = "".join(c for c, on in named if on) or "x"
    return f"fs_{self.lights}{self.nlight}_{self.material}_{self.textures}_{self.geometry}_{flags}"


def parse(code: str) -> Point:
  """The `Point` of registry name `code` (`Point.code`)."""
  _, lights, material, textures, geometry, flags = code.split("_")
  kind = lights.rstrip("0123456789")
  return Point(
    lights=kind,
    nlight=int(lights[len(kind) :]),
    material=material,
    coat="c" in flags,
    layers="l" in flags,
    textures=textures,
    geometry=geometry,
    instancing="i" in flags,
    env="e" in flags,
    headlight="h" in flags,
    planes=0 if "n" in flags else MANY_PLANES if "m" in flags else 1,
  )


def bits(p: Point) -> dict[str, object]:
  """The bundle bits `p` is meant to select: the surface (opaque / translucent / glass), coat,
  layers, headlight, primitives, and the light-loop form."""
  surface = {"glass": "glass", "thin": "translucent", "cutout": "translucent"}.get(p.material, "opaque")
  return {
    "surface": surface,
    "coat": p.coat,
    "layers": p.layers,
    "headlight": p.headlight,
    "primitives": p.geometry == "prim",
    "loop": LOOP_OF_COUNT[p.nlight],
    "planes": 0 if p.geometry == "hfield" else p.planes,
  }


def catalogue() -> list[Point]:
  """The stratified sample: one scene per feature-class combination (surface x coat x layers x
  headlight x primitives: 48), the light counts (and so the loop forms) cycled over them, the other
  axes (light kinds, textures, instancing,
  environment, EON, height fields) assigned round-robin so that each value appears several
  times, then the extremes."""
  surfaces = {"opaque": ("plain", "eon"), "translucent": ("thin", "cutout"), "glass": ("glass",)}
  kinds = itertools.cycle((*LIGHT_KINDS[:-1], "mix"))
  textures = itertools.cycle(TEXTURES)
  out: list[Point] = []
  nlights = itertools.cycle(LIGHT_COUNTS)
  for i, (surface, coat, layers, headlight, prim) in enumerate(
    itertools.product(surfaces, (False, True), (False, True), (False, True), (False, True))
  ):
    nlight = next(nlights)
    models = surfaces[surface]
    material = models[i % len(models)]
    # Height fields come from MuJoCo, whose materials hold neither the lobes, solid glass, cut-outs
    # nor EON: only the points it can express get one.
    mjcf = material in ("plain", "thin") and not coat and not layers
    geometry = "prim" if prim else ("hfield" if mjcf and i % 3 == 0 else "mesh")
    out.append(
      Point(
        lights=next(kinds),
        nlight=nlight,
        material=material,
        coat=coat,
        layers=layers,
        textures=next(textures),
        geometry=geometry,
        instancing=i % 3 == 0,
        env=i % 4 == 1,
        headlight=headlight,
      )
    )
  out += [
    Point(
      lights="mix",
      nlight=64,
      material="glass",
      coat=True,
      layers=True,
      textures="pbr",
      geometry="prim",
      instancing=True,
      env=True,
      headlight=True,
      planes=MANY_PLANES,
    ),
    Point(lights="ies", nlight=64, material="eon", coat=True, layers=True, textures="pbr", geometry="prim", env=True),
    Point(lights="env", nlight=1, material="plain", textures="none", geometry="mesh", env=True),
    Point(lights="point", nlight=1),
  ]
  return list(dict.fromkeys(out))


# Rare class combinations for the wide prebuilt-vs-scene evaluation
# (internal/benchmarks/specs/w6_final/ specs.py): each is meant to miss the prebuilt class sets
# (`raywarp.core.build.variants.class_sets.PREBUILT`) and so be routed to a superset
# (`tiers.cover`), which the evaluation prices against its scene build. Codes (`Point.code`): IES in
# a small loop, normal maps without a base map, solid glass alone, layers alone, coat with cut-outs,
# and a few mixed ones.
RARE: Final = (
  "fs_ies1_plain_none_mesh_x",
  "fs_env1_plain_normal_mesh_e",
  "fs_point2_glass_none_mesh_x",
  "fs_point1_plain_normal_prim_h",
  "fs_spot4_eon_pbr_mesh_l",
  "fs_point8_cutout_base_mesh_c",
  "fs_rect1_thin_normal_prim_lie",
  "fs_sun1_glass_pbr_prim_lh",
  "fs_disk2_plain_base_mesh_ce",
  "fs_mix8_eon_normal_mesh_lm",
)


def rare() -> list[Point]:
  """The `RARE` points (registered in `internal/benchmarks/scenes.py` with the catalogue)."""
  return [parse(code) for code in RARE]


# The random-scene fuzz (internal/benchmarks/specs/w6_final.md, "any arbitrary scene"): light counts
# beyond the catalogue's (3, 5, 6, 16: the small forms' edges and the general loop below the grid),
# geometry weighted towards the host kinds, MuJoCo kinds as their builders express them.
RANDOM_LIGHT_COUNTS: Final = (1, 2, 3, 4, 5, 6, 8, 16, 64)
RANDOM_GEOMETRY: Final = ("mesh", "mesh", "prim", "prim", "hfield", "mjmesh", "flex", "flexplain")
RANDOM_SEED: Final = 20261107
RANDOM_COUNT: Final = 40


def _pick[T](rng: np.random.Generator, options: Sequence[T]) -> T:
  """One of `options`, uniformly."""
  return options[rng.integers(len(options), size=1).tolist()[0]]


def random_point(rng: np.random.Generator) -> Point:
  """One point drawn over every axis; MuJoCo geometries keep what their MJCF builders express
  (height fields: point / spot / sun lights, plain or thin props, the headlight; the base copies:
  point lights only)."""
  geometry = _pick(rng, RANDOM_GEOMETRY)
  nlight = _pick(rng, RANDOM_LIGHT_COUNTS)
  if geometry == "hfield":
    lights = _pick(rng, ("point", "spot", "sun"))
    return Point(lights, nlight, _pick(rng, ("plain", "thin")), geometry=geometry, headlight=rng.random() < 0.3, planes=0)
  if geometry in MJCF_GEOMETRY:
    return Point("point", nlight, geometry=geometry, planes=_pick(rng, PLANE_COUNTS))
  lights = _pick(rng, (*LIGHT_KINDS, "mix"))
  p = Point(
    lights=lights,
    nlight=1 if lights == "env" else nlight,
    material=_pick(rng, MATERIALS),
    coat=rng.random() < 0.4,
    layers=rng.random() < 0.3,
    textures=_pick(rng, TEXTURES),
    geometry=geometry,
    instancing=rng.random() < 0.4,
    env=rng.random() < 0.4,
    headlight=rng.random() < 0.2,
    planes=_pick(rng, (0, 1, 1, MANY_PLANES)),
  )
  return dataclasses.replace(p, env=True) if sealed(p) else p


def sealed(p: Point) -> bool:
  """Whether `p`'s stage is dark: its only lights are suns (outside the stage) and its planes
  close it from above (`MANY_PLANES`: the ceiling at z = 8 shadows every sun; the environment
  reaches the stage through the open side)."""
  return p.lights == "sun" and p.planes >= MANY_PLANES and not p.env


def random_points(n: int = RANDOM_COUNT, seed: int = RANDOM_SEED) -> list[Point]:
  """`n` distinct seeded random points, none of the catalogue, the base scenes or `RARE` (scenes
  the release census never saw)."""
  rng = np.random.default_rng(seed)
  known = {p.code() for p in (*catalogue(), *base_scenes(), *rare())}
  out: dict[str, Point] = {}
  while len(out) < n:
    p = random_point(rng)
    if p.code() not in known:
      out.setdefault(p.code(), p)
  return list(out.values())


# The light counts of the bundle fill's base scenes: every loop form (`LOOP_OF_COUNT`; 64: the
# general loop with the light grid; 0: the general loop without lights or grid, a scene lit by its
# environment or emission only).
BASE_LIGHT_COUNTS: Final = (0, 1, 2, 3, 4, 8, 64)


# The base scenes' light kinds: point lights casting shadows ("point"); three or four lights of
# which two cast shadows ("pointfill": a shadow budget covering every caster resolves to none, so
# the lights take the small loop of their count at every tier, `tiers.class_free`).
BASE_FILL_LIGHTS: Final = (3, 4)


# The bundle's per-world optics point (`tools/prebuilt/fill._optics_points`, Features.world_optics):
# glass props whose material and refraction index are chosen per world (domain randomisation), its
# census form with that declaration (`internal/benchmarks/scenes.py`, `CodeScene.per_world`).
OPTICS_POINT: Final = Point(lights="point", nlight=2, material="glass", textures="none", geometry="mesh")
OPTICS_PER_WORLD: Final[dict[str, tuple[str, ...]]] = {"material": ("instance.material", "specular_ior")}


def base_scenes() -> list[Point]:
  """The bundle fill's base scenes (`tools/prebuilt/fill.py`): plain opaque meshes lit by point
  lights, no texture, environment, headlight or primitive, so that each uses no feature class, at
  every light count and plane form; as host scenes ("mesh") and as MuJoCo copies ("mjmesh": its
  pose layout; "flex": with a deformable). A class set compiles in on any of them (`tiers.cover`),
  so every build of the bundle is one base scene's tier build with one class set."""
  return [
    Point(
      lights="point",
      nlight=n,
      material="plain",
      coat=False,
      layers=False,
      textures="none",
      geometry=geometry,
      instancing=False,
      env=False,
      headlight=False,
      planes=planes,
    )
    for geometry in ("mesh", "mjmesh", "flex", "flexplain")
    for n in BASE_LIGHT_COUNTS
    for planes in PLANE_COUNTS
  ] + [
    Point(lights="pointfill", nlight=n, material="plain", textures="none", geometry=geometry, planes=planes)
    for geometry in ("mesh", "mjmesh", "flex", "flexplain")
    for n in BASE_FILL_LIGHTS
    for planes in PLANE_COUNTS
  ]


# --- Scene construction ---------------------------------------------------------------------------


def _xform(x: float, y: float, z: float, rot: np.ndarray | None = None) -> np.ndarray:
  """(3, 4) frame at (x, y, z) (`rot`: its rotation, else identity)."""
  return np.c_[np.eye(3) if rot is None else rot, [x, y, z]].astype(np.float32)


def _xform_rx(a: float) -> np.ndarray:
  """Rotation by `a` about x."""
  c, s = math.cos(a), math.sin(a)
  return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _xform_ry(a: float) -> np.ndarray:
  """Rotation by `a` about y."""
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _down() -> np.ndarray:
  """(3, 3) rotation whose -Z (a light's emission axis) points down: the identity (Z up)."""
  return np.eye(3)


def _sun_rotation() -> np.ndarray:
  """(3, 3) rotation of a low sun from the south-west (emission along -Z)."""
  zx, zy, zz = 0.4, 0.5, 0.75
  zn = math.sqrt(zx * zx + zy * zy + zz * zz)
  zx, zy, zz = zx / zn, zy / zn, zz / zn
  z = np.array([zx, zy, zz])
  # (0, 0, 1) x z, normalised.
  xn = math.hypot(zx, zy)
  x = np.array([-zy / xn, zx / xn, 0.0])
  return np.stack([x, np.cross(z, x), z], axis=1)


def _light(kind: str, i: int, n: int) -> rw.Light:
  """Light `i` of `n` of `kind` over the stage (the total power shared: the exposure barely
  moves)."""
  side = math.ceil(math.sqrt(n))
  x, y = ((i % side) + 0.5) / side * 3.0 - 1.5, ((i // side) + 0.5) / side * 3.0 - 1.5
  at = _xform(x, y, 2.4, _down())
  share = 1.0 / n
  if kind in ("point", "pointfill"):
    # "pointfill": the lights past the first two cast no shadow.
    shadows = kind == "point" or i < 2
    return rw.Light(
      kind=LightKind.SPHERE, intensity=300.0 * share, radius=0.0, treat_as_point=True, shadow_enable=shadows, xform=at
    )
  if kind == "spot":
    return rw.Light(
      kind=LightKind.SPHERE, intensity=600.0 * share, radius=0.02, shaping=Shaping(cone_angle=35.0, cone_softness=0.3), xform=at
    )
  if kind == "rect":
    return rw.Light(kind=LightKind.RECT, intensity=900.0 * share, width=0.4, height=0.3, xform=at)
  if kind == "disk":
    return rw.Light(kind=LightKind.DISK, intensity=900.0 * share, radius=0.2, xform=at)
  if kind == "sphere":
    return rw.Light(kind=LightKind.SPHERE, intensity=900.0 * share, radius=0.08, xform=at)
  if kind == "tube":
    return rw.Light(kind=LightKind.CYLINDER, intensity=900.0 * share, radius=0.02, length=0.6, xform=at)
  if kind == "ies":
    return rw.Light(
      kind=LightKind.SPHERE, intensity=600.0 * share, radius=0.0, treat_as_point=True,
      shaping=Shaping(ies=IES.read_text(), ies_mode="measured"), xform=at,
    )  # fmt: skip
  if kind == "sun":
    return rw.Light(kind=LightKind.DISTANT, intensity=3.0 * share, angle=0.53, xform=_xform(0.0, 0.0, 0.0, _sun_rotation()))
  raise ValueError(f"light kind {kind!r}")


def _checker(rgb: bool, cells: int = 8, texels: int = 16) -> np.ndarray:
  """(N, N, 3) uint8 checker (`rgb`: coloured, else grey)."""
  i = (np.arange(cells * texels) // texels) % 2
  board = (i[:, None] ^ i[None, :]).astype(np.float32)
  colour = np.array([0.85, 0.55, 0.3]) if rgb else np.array([0.8, 0.8, 0.8])
  image = (0.25 + 0.75 * board[..., None]) * colour
  return (image * 255).astype(np.uint8)


def _bumps(texels: int = 128) -> np.ndarray:
  """(N, N, 3) uint8 tangent-space normal map of a grid of bumps."""
  u = np.linspace(0.0, 8.0 * math.pi, texels)
  nx, ny = 0.3 * np.cos(u)[None, :] * np.ones((texels, 1)), 0.3 * np.cos(u)[:, None] * np.ones((1, texels))
  n = np.stack([nx, ny, np.ones_like(nx)], -1)
  n /= np.linalg.norm(n, axis=-1, keepdims=True)
  return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _textured(b: rw.SceneBuilder, base: Material, roles: str) -> Material:
  """`base` with the texture roles `roles` (`TEXTURES`)."""
  if roles == "none":
    return base
  mat = base.with_texture(Slot.BASE_COLOR, TextureRef(b.add_texture(_checker(True))))
  if roles in ("pbr", "normal"):
    normal = b.add_texture(_bumps(), colorspace="linear")
    mat = mat.with_texture(Slot.GEOMETRY_NORMAL, TextureRef(normal, scale=(2.0, 2.0, 2.0, 1.0), bias=(-1.0, -1.0, -1.0, 0.0)))
  if roles == "pbr":
    data = b.add_texture(_checker(False), colorspace="linear")
    mat = mat.with_texture(Slot.SPECULAR_ROUGHNESS, TextureRef(data, channel=1))
    mat = mat.with_texture(Slot.BASE_METALNESS, TextureRef(data, channel=2))
    mat = mat.with_texture(Slot.OCCLUSION, TextureRef(data, channel=0))
    mat = mat.with_texture(Slot.EMISSION_COLOR, TextureRef(b.add_texture(_checker(True))))
  return mat


def _material(b: rw.SceneBuilder, p: Point, k: int) -> Material:
  """Prop material `k` of `p`: its model (`MATERIALS`), the coat on the odd props and the layered
  lobes on props 2 and 3 when `p` has them, its texture roles."""
  hue = (0.3 + 0.6 * ((k * 0.37) % 1.0), 0.35 + 0.5 * ((k * 0.61) % 1.0), 0.3 + 0.5 * ((k * 0.83) % 1.0))
  base = rw.Material(base_color=hue, base_metalness=float(k % 4 == 0), specular_roughness=0.25 + 0.15 * (k % 4))
  extras = OpenPbrExtras()
  if p.coat and k % 2:
    base = dataclasses.replace(base, coat_weight=1.0, coat_roughness=0.1)
  if p.layers and k >= 2:
    extras = dataclasses.replace(extras, specular_roughness_anisotropy=0.6, fuzz_weight=0.5)
  if p.material == "eon":
    extras = dataclasses.replace(extras, base_diffuse_roughness=0.8)
  elif p.material == "glass":
    base = dataclasses.replace(base, transmission_weight=1.0, specular_roughness=0.02, specular_ior=1.5)
    extras = dataclasses.replace(extras, transmission_depth=0.3)
  elif p.material == "thin":
    base = dataclasses.replace(
      base, geometry_opacity=0.4, geometry_thin_walled=True, ext=RaywarpExt(alpha_mode=AlphaMode.BLEND)
    )
  elif p.material == "cutout":
    mask = b.add_texture(_checker(False, cells=6), colorspace="linear")
    base = dataclasses.replace(base, ext=RaywarpExt(alpha_mode=AlphaMode.MASK, alpha_cutoff=0.5)).with_texture(
      Slot.GEOMETRY_OPACITY, TextureRef(mask, channel=0)
    )
  if extras != OpenPbrExtras():
    base = dataclasses.replace(base, extras=extras)
  return _textured(b, base, p.textures)


def _mesh(b: rw.SceneBuilder, k: int) -> MeshId:
  """A distinct prop mesh `k` (an n-gon prism with UVs)."""
  sides = 5 + 2 * (k % 5)
  a = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
  ring = np.stack([0.25 * np.cos(a), 0.25 * np.sin(a)], -1)
  points = np.concatenate([np.c_[ring, np.zeros(sides)], np.c_[ring, np.full(sides, 0.5)]]).astype(np.float32)
  tris = []
  for i in range(sides):
    j = (i + 1) % sides
    tris += [(i, j, sides + j), (i, sides + j, sides + i)]
  for i in range(1, sides - 1):
    tris += [(0, i + 1, i), (sides, sides + i, sides + i + 1)]
  uv = (points[:, :2] + 0.25) * 2.0
  return b.add_mesh(points, np.asarray(tris, np.int32), uvs=uv.astype(np.float32))


def build(p: Point) -> rw.Scene:
  """The scene of `p`: a 3 x 3 m stage of props lit by `p`'s lights (height fields:
  `build_mjcf`)."""
  b = rw.SceneBuilder()
  floor = b.add_material(_textured(b, rw.Material(base_color=(0.5, 0.5, 0.48), specular_roughness=0.7), p.textures))
  if p.planes:
    b.add_instance("plane", size=(4.0, 4.0, 0.0), material=floor)
  else:
    quad = np.array([[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]], np.float32)
    uv = (quad[:, :2] + 4.0) / 8.0
    b.add_instance("mesh", mesh=b.add_mesh(quad, np.array([[0, 1, 2], [0, 2, 3]], np.int32), uvs=uv), material=floor)
  # More planes: walls on three sides of the stage, a ceiling, a second floor below the first.
  walls = ((0.0, 6.0, 3.0, _xform_rx(math.pi / 2)), (6.0, 0.0, 3.0, _xform_ry(-math.pi / 2)),
           (-6.0, 0.0, 3.0, _xform_ry(math.pi / 2)), (0.0, 0.0, 8.0, _xform_rx(math.pi)), (0.0, 0.0, -1.0, None))  # fmt: skip
  for x, y, z, rot in walls[: max(p.planes - 1, 0)]:
    b.add_instance("plane", size=(8.0, 8.0, 0.0), material=floor, xform=_xform(x, y, z, rot))
  mats = [b.add_material(_material(b, p, k)) for k in range(4)]
  grid = [(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
  if p.instancing:
    mesh = _mesh(b, 0)
    xforms = np.stack([_xform(x + 0.25 * dx, y + 0.25 * dy, 0.0) for x, y in grid for dx in (-1, 1) for dy in (-1, 1)])
    for m, chunk in enumerate(np.array_split(xforms, len(mats))):
      b.add_instances("mesh", xforms=chunk, mesh=mesh, material=mats[m])
  else:
    for k, (x, y) in enumerate(grid[:5]):
      b.add_instance("mesh", mesh=_mesh(b, k), material=mats[k % len(mats)], xform=_xform(x, y, 0.0))
  if p.geometry == "prim":
    shapes = (("box", (0.2, 0.2, 0.2)), ("sphere", (0.22, 0.0, 0.0)), ("capsule", (0.1, 0.2, 0.0)),
              ("ellipsoid", (0.25, 0.15, 0.2)), ("cylinder", (0.15, 0.2, 0.0)))  # fmt: skip
    for k, ((kind, size), (x, y)) in enumerate(zip(shapes, grid[5:] + grid[:1], strict=False)):
      b.add_instance(kind, size=size, material=mats[k % len(mats)], xform=_xform(x, y, 0.3))
  kinds = [k for k in LIGHT_KINDS if k != "env"] if p.lights == "mix" else [p.lights]
  if p.lights != "env":
    for i in range(p.nlight):
      b.add_light(_light(kinds[i % len(kinds)], i, p.nlight))
  if p.env or p.lights == "env":
    b.add_environment(EnvironmentLight(color=(0.55, 0.65, 0.8), intensity=0.6))
  if p.headlight:
    b.add_light(rw.Light(kind=LightKind.CAMERA, color=(0.3, 0.3, 0.3)))
  return b.build()


def build_mjcf(p: Point) -> str:
  """The MJCF of a MuJoCo `p` (`MJCF_GEOMETRY`): a height field (`_hfield_mjcf`), or a base
  scene's copy (`_base_mjcf`)."""
  return _hfield_mjcf(p) if p.geometry == "hfield" else _base_mjcf(p)


def _base_mjcf(p: Point) -> str:
  """A base scene's MuJoCo copy (`base_scenes`): the floor (a plane, or a slab mesh without
  planes), five prism props in the world body, `p.nlight` point lights in a grid over the stage,
  no headlight, no texture; "flex" / "flexplain": with a 9 x 9 cloth over the props (with / without
  its own material)."""
  slab = "-4 -4 -.02  4 -4 -.02  4 4 -.02  -4 4 -.02  -4 -4 0  4 -4 0  4 4 0  -4 4 0"
  floor = (
    '<geom name="floor" type="plane" size="4 4 0.1" material="floor"/>'
    if p.planes
    else '<geom name="floor" type="mesh" mesh="slab" material="floor"/>'
  )
  side = max(1, math.ceil(math.sqrt(p.nlight)))
  lights = "".join(
    f'<light name="l{i}" type="point" pos="{((i % side) + 0.5) / side * 3.0 - 1.5:.3f} '
    f'{((i // side) + 0.5) / side * 3.0 - 1.5:.3f} 2.4" diffuse="{1.0 / p.nlight:.4f} {1.0 / p.nlight:.4f} '
    f'{1.0 / p.nlight:.4f}" castshadow="{"true" if p.lights == "point" or i < 2 else "false"}"/>'
    for i in range(p.nlight)
  )
  meshes = "".join(f'<mesh name="prop{k}" vertex="{_prism_vertices(5 + 2 * k)}"/>' for k in range(5))
  props = "".join(
    f'<geom name="prop{k}" type="mesh" mesh="prop{k}" pos="{x} {y} 0" material="m{k % 4}"/>'
    for k, (x, y) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0)))
  )
  materials = "".join(
    f'<material name="m{k}" rgba="{0.3 + 0.6 * ((k * 0.37) % 1.0):.3f} {0.35 + 0.5 * ((k * 0.61) % 1.0):.3f} '
    f'{0.3 + 0.5 * ((k * 0.83) % 1.0):.3f} 1"/>'
    for k in range(4)
  )
  cloth = (
    '<flexcomp name="cloth" type="grid" count="9 9 1" spacing="0.12 0.12 0.12" pos="0.6 0.6 1.2" dim="2" '
    f'radius="0.006" mass="0.24"{' material="cloth"' if p.geometry == "flex" else ""}>'
    '<edge equality="true" damping="0.1"/><contact selfcollide="none"/><pin id="0 8"/></flexcomp>'
    if p.geometry in ("flex", "flexplain")
    else ""
  )
  return f"""<mujoco><visual><headlight active="0"/></visual>
  <asset>
    <material name="floor" rgba="0.5 0.5 0.48 1"/>{materials}
    <material name="cloth" rgba="0.7 0.3 0.25 1"/>
    <mesh name="slab" vertex="{slab}"/>{meshes}
  </asset>
  <worldbody>{lights}
    <camera name="cam" pos="0 -4.2 2.4" xyaxes="1 0 0 0 0.5 0.87"/>
    {floor}{props}{cloth}
  </worldbody>
</mujoco>"""


def _prism_vertices(sides: int) -> str:
  """The vertices of `_mesh`'s n-gon prism (MuJoCo takes their convex hull), as MJCF text."""
  a = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
  ring = np.stack([0.25 * np.cos(a), 0.25 * np.sin(a)], -1)
  points = np.concatenate([np.c_[ring, np.zeros(sides)], np.c_[ring, np.full(sides, 0.5)]])
  return "  ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in points)


def _hfield_mjcf(p: Point) -> str:
  """The MJCF of a height-field `p` (MuJoCo is the height-field source): a sine terrain under
  mesh props (no analytic shape: the primitives bit stays off), MuJoCo point / spot / directional
  lights (`p`'s nearest kinds), its headlight as `p` says."""
  kinds = {"spot": "spot", "sun": "directional"}
  lights = "".join(
    f'<light pos="{-1.2 + 2.4 * ((i * 0.618) % 1.0):.3f} {-1.2 + 2.4 * ((i * 0.381) % 1.0):.3f} 2.4" dir="0 0 -1" '
    f'type="{kinds.get(p.lights, "spot" if i % 2 else "point")}" castshadow="true"/>'
    for i in range(p.nlight)
  )
  head = "" if p.headlight else '<visual><headlight active="0"/></visual>'
  props = "".join(
    f'<geom type="mesh" mesh="prop" pos="{x} {y} 0.3" rgba="{0.3 + 0.1 * k:.2f} 0.5 0.6 '
    f'{0.4 if p.material in ("thin", "glass") else 1.0}"/>'
    for k, (x, y) in enumerate(((-1, -1), (0, 0), (1, 1), (1, -1)))
  )
  u = np.linspace(0.0, 4.0 * math.pi, 64)
  heights = 0.5 + 0.25 * np.sin(u)[None, :] + 0.25 * np.cos(u)[:, None]
  elevation = " ".join(f"{h:.3f}" for h in heights.ravel())
  return f"""<mujoco>{head}
  <asset>
    <hfield name="terrain" nrow="64" ncol="64" size="2 2 0.3 0.05" elevation="{elevation}"/>
    <texture name="grid" type="2d" builtin="checker" width="64" height="64" rgb1=".8 .8 .8" rgb2=".3 .3 .3"/>
    <material name="ground" texture="grid" texrepeat="6 6"/>
    <mesh name="prop" vertex="-.2 -.2 0  .2 -.2 0  .2 .2 0  -.2 .2 0  0 0 .4"/>
  </asset>
  <worldbody>{lights}
    <camera name="cam" pos="0 -4.2 2.4" xyaxes="1 0 0 0 0.5 0.87"/>
    <geom type="hfield" hfield="terrain" material="ground"/>{props}
  </worldbody>
</mujoco>"""
