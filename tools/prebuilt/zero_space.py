"""The input space of the zero-compile check (`zero.py`): every compile-relevant settings field and
every scene value the resolution reads, each over its domain, and the walks over it.

Settings (`SETTINGS`): every field `RenderSettings.fields_table()` marks compile-time, plus the
run-time fields that change what the render kernels store (`exposure.mode`,
`sensor.aberration`), each at every choice, the integers at the values around each threshold the
code compares them to: the library's domain of single changes
(`raywarp/core/build/variants/nearby.py`), and the outputs. Scene (`SCENE`): every `zero.Scene`
value. `test_settings_space_is_every_compile_time_field` keeps `SETTINGS` complete.

`walk(depth)` routes every base (`BASES`: the recipes of `tools/recipes.py`) with every combination
of up to `depth` axis values (settings and scene together) and yields the points that compile.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from raywarp.core.build.variants import nearby
from tools import recipes
from tools.prebuilt import zero

if TYPE_CHECKING:
  from raywarp.settings import RenderSettings

# The settings the walk starts from: the default, the fast field set, unlit shading and the
# documented looks (`tools/recipes.py`).
BASES: Final = tuple(recipes.RECIPES)
# Compile-time settings field -> its values: the library's domain of single changes
# (`core/build/variants/nearby.py`, the one `Renderer.warm(nearby=...)` creates the modules of) and
# the outputs as 7 sets.
SETTINGS: Final[dict[str, tuple[object, ...]]] = {
  **nearby.DOMAIN,
  "outputs": ("rgb", "depth", "rgb+motion", "depth+motion", "rgb+direct", "rgb+guides", "all"),
}
EXCLUDED: Final = nearby.EXCLUDED


def outputs(name: str) -> object:
  """The `Outputs` of output-set `name` (`SETTINGS["outputs"]`)."""
  from raywarp.settings.fields import OutputKind
  from raywarp.settings.outputs import KINDS, Outputs

  sets: dict[str, tuple[OutputKind, ...]] = {
    "rgb": ("rgb",),
    "depth": ("depth",),
    "rgb+motion": ("rgb", "motion"),
    "depth+motion": ("depth", "motion"),
    "rgb+direct": ("rgb", "direct", "indirect"),
    "rgb+guides": ("rgb", "hdr", "albedo", "normal"),
    "all": KINDS,
  }
  return Outputs.of(*sets[name])


def settings_of(base: RenderSettings, changes: dict[str, object]) -> RenderSettings:
  """`base` with `changes` (the "outputs" axis a whole group; ValueError: an invalid
  combination)."""
  rest = {k: v for k, v in changes.items() if k != "outputs"}
  s = base.replace(**rest) if rest else base
  return s.replace(outputs=outputs(str(changes["outputs"]))) if "outputs" in changes else s


def _caps(**fields: object) -> dict[str, object]:
  """A scene axis value changing the capabilities (`zero.capabilities`)."""
  return {"caps": fields}


def _shapes(*names: str) -> dict[str, object]:
  """A scene axis value with the meshes and the shape types `names` (`ShapeType` members)."""
  from raywarp._base.types import ShapeType

  return _caps(shape_types=(int(ShapeType.MESH), *(int(ShapeType[n]) for n in names)))


def _textures(**slots: bool) -> dict[str, object]:
  """A scene axis value binding texture slots `slots` (`scene.TextureSlots`)."""
  from raywarp.core.pipeline.gpu_scene.gpu_scene import TextureSlots

  return _caps(textures=TextureSlots(**slots))


def _layers(**lobes: bool) -> dict[str, object]:
  """A scene axis value with layered lobes `lobes` (`scene.LayerUse`)."""
  from raywarp.core.pipeline.gpu_scene.gpu_scene import LayerUse

  return _caps(layers=LayerUse(**lobes))


def _prepared(**fields: object) -> dict[str, object]:
  """A scene axis value with a prepared-material superset (read with "instance.material")."""
  from raywarp.core.pipeline.gpu_scene.gpu_scene import PreparedMaterialCapabilities, TextureSlots

  base = {"textures": TextureSlots(), "world_uv": False, "alpha_cutout": False, "alpha_map": False, "has_alpha": False}
  base |= {"has_mirror": False, "specular_curve": False, "clearcoat": False, "transmissive": False, "vertex_colors": False}
  return _caps(prepared_materials=PreparedMaterialCapabilities(**(base | fields)))  # pyrefly: ignore[bad-argument-type]


def scene_axes() -> dict[str, dict[str, dict[str, object]]]:
  """Scene axis -> value name -> the `zero.Scene` fields it sets ("caps": `Capabilities` fields).
  Each axis's first value is the plain base's."""
  bools = (
    "has_alpha alpha_map has_mirror has_spot has_area_lights visible_lights mesh_frames uv_maps world_uv "
    "physical_emission specular_curve transmissive light_panels alpha_cutout vertex_colors diffuse_roughness clearcoat "
    "sun_disc has_ies"
  ).split()
  slots = "base_color normal orm emissive opacity scaled_normal diffuse_roughness layered coat_normal detail_normal".split()
  lights: dict[str, dict[str, object]] = {
    f"{n}/{c}": {"nlight": n, "casters": c}
    for n, c in ((1, 1), (0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (64, 64), (3, 2), (8, 2), (64, 1))
  }
  axes: dict[str, dict[str, dict[str, object]]] = {
    "lights": lights,
    "shapes": {
      "mesh": {},
      "box": _shapes("BOX"),
      "round": _shapes("SPHERE", "CAPSULE"),
      "custom": _shapes("ELLIPSOID", "CYLINDER"),
      "hfield": _shapes("HFIELD"),
      "sdf": _shapes("SDF"),
      "all": _shapes("BOX", "SPHERE", "CAPSULE", "ELLIPSOID", "CYLINDER", "HFIELD", "SDF"),
    },
    "env": {"none": {}, "background": {"env_maps": True}, "lit": {"env_lit": True, "env_maps": True}},
    "flex": {"none": {}, "surface": {"flex": 2}, "material": {"flex": 2, "flex_material": True}, "segments": {"flex": 1}},
    "transmit": {"opaque": {}, "gray": {"transmit": 1}, "tinted": {"transmit": 2}},
    "layout": {
      "shared": {},
      "both": {"world": True},
      "world": {"world": True, "shared": False},
      "mujoco": {"compact": False},
      "visible": {"visible": True},
      "plane": {"planes": 1},
    },
    "camera": {
      "pinhole": {},
      "distortion": {"distortion": True},
      "bundle": {"bundle": True},
      "mixed_resolutions": {"uniform": False},
    },
    "prepared": {
      "none": {},
      "plain": _prepared(),
      "rich": _prepared(
        clearcoat=True,
        has_mirror=True,
        transmissive=True,
        has_alpha=True,
        alpha_cutout=True,
        alpha_map=True,
        vertex_colors=True,
        world_uv=True,
        specular_curve=True,
      ),
    },
  }
  axes |= {f"cap.{b}": {"off": {}, "on": _caps(**{b: True})} for b in bools}
  axes |= {f"tex.{t}": {"off": {}, "on": _textures(**{t: True})} for t in slots}
  axes |= {f"layer.{t}": {"off": {}, "on": _layers(**{t: True})} for t in ("anisotropy", "fuzz", "coat", "coat_own_normal")}
  for flag in ("headlight", "medium", "twosided", "material_uv", "curve", "weight", "base_lit", "diffuse_rows", "diffuse_role"):
    axes[flag] = {"off": {}, "on": {flag: True}}
  for name in zero.DECLARED:
    axes[f"declared.{name}"] = {"off": {}, "on": {"declared": frozenset({name})}}
  axes["declared.alpha"] = {"off": {}, "on": {"declared": frozenset({"material.base_color"}), "alpha": True}}
  return axes


def scene_of(values: dict[str, dict[str, object]]) -> zero.Scene:
  """The `zero.Scene` of scene axis values (their fields merged: capabilities, declarations)."""
  caps: dict[str, object] = {}
  declared: set[str] = set()
  fields: dict[str, object] = {}
  for v in values.values():
    for k, x in v.items():
      if k == "caps" and isinstance(x, dict):
        caps |= x
      elif k == "declared" and isinstance(x, (set, frozenset)):
        declared |= {str(d) for d in x}
      else:
        fields[k] = x
  return zero.Scene(**fields, caps=zero.capabilities(**caps), declared=frozenset(declared))  # pyrefly: ignore[bad-argument-type]


@dataclasses.dataclass(frozen=True)
class Hole:
  """A configuration that compiles: its base (`BASES`), settings and scene changes, the route."""

  base: str
  settings: tuple[tuple[str, object], ...]
  scene: tuple[tuple[str, str], ...]
  tier: str | None
  outside: tuple[str, ...]


@dataclasses.dataclass
class Tally:
  """What a walk routed: points, refused (settings the renderer raises for), on a fill key, the
  holes, and the served points whose build costs more than their own (`cost_anomaly`)."""

  points: int = 0
  refused: int = 0
  served: int = 0
  holes: list[Hole] = dataclasses.field(default_factory=list)
  anomalies: list[Hole] = dataclasses.field(default_factory=list)


def _values() -> list[tuple[str, str, object]]:
  """Every non-default axis value: ("settings" | "scene", axis, value)."""
  out: list[tuple[str, str, object]] = [("settings", a, v) for a, vs in SETTINGS.items() for v in vs]
  out += [("scene", a, v) for a, vs in scene_axes().items() for v in list(vs)[1:]]
  return out


def walk(
  depth: int,
  bases: tuple[str, ...] = BASES,
  *,
  keep: Callable[[Hole], bool] = lambda h: True,
  shard: tuple[int, int] = (0, 1),
  exact: bool = False,
) -> Tally:
  """Routes every base with every combination of up to `depth` axis values (distinct axes);
  `shard` (i, n): only every n-th combination from the i-th (parallel walks); `exact`: only the
  combinations of exactly `depth` values."""
  keys, axes, tally = zero.fill_keys(), scene_axes(), Tally()
  probes: dict[zero.Scene, zero.Probe] = {}
  values = _values()
  index = -1
  for name in bases:
    for k in range(depth if exact else 0, depth + 1):
      for combo in itertools.combinations(values, k):
        if len({(kind, a) for kind, a, _ in combo}) < k:
          continue
        index += 1
        if index % shard[1] != shard[0]:
          continue
        _route_one(name, list(combo), keys, axes, probes, tally, keep=keep)
  return tally


def random_walk(n: int, k: tuple[int, int], *, seed: int, bases: tuple[str, ...] = BASES) -> Tally:
  """Routes `n` random points: a base of `bases`, then K in [k[0], k[1]] distinct axes (settings and
  scene together), each at a random non-default value; the cross-check of the all-K proof
  (`zero_proof.py`) beyond the exhaustive single and pair walks. Seeded: shards take distinct
  seeds."""
  import random

  rng, keys, axes, tally = random.Random(seed), zero.fill_keys(), scene_axes(), Tally()
  by_axis: dict[tuple[str, str], list[object]] = collections.defaultdict(list)
  for kind, a, v in _values():
    by_axis[kind, a].append(v)
  names = sorted(by_axis, key=str)
  probes: dict[zero.Scene, zero.Probe] = {}
  while tally.points < n:
    base = rng.choice(bases)
    combo = [(kind, a, rng.choice(by_axis[kind, a])) for kind, a in rng.sample(names, rng.randint(*k))]
    _route_one(base, combo, keys, axes, probes, tally)
  return tally


# The settings that select a tier variant (`tiers.VARIANTS`, the run-time sampling and counts, the
# body forms of the outputs), at their thresholds: the lattice whose every combination
# `lattice_walk` routes. The light caps and shadow budget compressed to none / below the lights /
# at the counted capacity.
LATTICE: Final[dict[str, tuple[object, ...]]] = {
  "aa.spp": (1, 2, 3),
  "shadows.samples": (1, 2, 3, 4, 8),
  "surfaces.anisotropy": (1, 2, 4, 8, 16),
  "reflections.shadows": ("off", "sun"),
  "shadows.max_casters": (None, 1, 4, 16),
  "lighting.max_exact": (None, 1, 4, 16),
  "reflections.max_exact": (None, 1, 16),
  "surfaces.shading": ("pbr", "unlit"),
  "outputs": ("rgb", "depth", "rgb+motion", "all"),
}
# The scenes of the lattice walk: the plain one, and a rich one (8 casting lights, area lights,
# an environment, textures, translucent and cut-out materials, primitives).
LATTICE_SCENES: Final = {
  "plain": {},
  "rich": {
    "lights": "8/8",
    "cap.has_area_lights": "on",
    "env": "lit",
    "tex.base_color": "on",
    "tex.opacity": "on",
    "cap.has_alpha": "on",
    "cap.alpha_cutout": "on",
    "shapes": "all",
  },
}


def lattice_walk(shard: tuple[int, int] = (0, 1), bases: tuple[str, ...] = BASES) -> Tally:
  """Routes every base x every combination of the `LATTICE` settings x every `LATTICE_SCENES`
  scene (`shard` (i, n): every n-th point from the i-th): the tier selection is exhaustive, any
  number of its settings changed together."""
  keys, axes, tally, probes = zero.fill_keys(), scene_axes(), Tally(), {}
  names = list(LATTICE)
  index = -1
  for base in bases:
    for scene in LATTICE_SCENES.values():
      for values in itertools.product(*LATTICE.values()):
        index += 1
        if index % shard[1] != shard[0]:
          continue
        combo = [("settings", a, v) for a, v in zip(names, values, strict=True)]
        combo += [("scene", a, v) for a, v in scene.items()]
        _route_one(base, combo, keys, axes, probes, tally)
  return tally


def _route_one(  # noqa: PLR0913  # the walk's state
  base: str,
  combo: list[tuple[str, str, object]],
  keys: frozenset[str],
  axes: dict[str, dict[str, dict[str, object]]],
  probes: dict[zero.Scene, zero.Probe],
  tally: Tally,
  *,
  keep: Callable[[Hole], bool] = lambda h: True,
) -> None:
  """Routes one point (recipe `base` with the axis values `combo`) into `tally` (an invalid
  settings combination is not a point; `keep`: the holes recorded)."""
  changes = {a: v for kind, a, v in combo if kind == "settings"}
  try:
    s = settings_of(recipes.settings(base), changes)
  except ValueError:
    return
  point = scene_of({a: axes[a][str(v)] for kind, a, v in combo if kind == "scene"})
  probe = probes.setdefault(point, zero.Probe(repr(point), point))
  if len(probes) > 4096:
    probes.clear()
  tally.points += 1
  try:
    r = zero.route(probe, s)
  except ValueError:
    tally.refused += 1
    return
  names = tuple((a, str(v)) for kind, a, v in combo if kind == "scene")
  if r.tier is not None and r.key in keys:
    tally.served += 1
    why = cost_anomaly(s, r.tier)
    if why:
      tally.anomalies.append(Hole(base, tuple(changes.items()), names, r.tier, (why,)))
    return
  hole = Hole(base, tuple(changes.items()), names, r.tier, r.outside)
  if keep(hole):
    tally.holes.append(hole)


# The settings a counted / sampled variant reads at run time (`tiers.COUNTED`, `SAMPLED`): its
# compiled capacities cost frames, so a configuration asking no more than its tier family's base
# values should not land there.
COUNT_FIELDS: Final = (
  "shadows.max_casters",
  "lighting.max_exact",
  "reflections.max_exact",
  "shadows.samples",
  "surfaces.anisotropy",
)


def cost_anomaly(s: RenderSettings, tier: str) -> str:
  """Why serving settings `s` with `tier` costs more than `s` asks for ("" if not): a counted or
  sampled variant serving count values no larger than its family's base tier compiles (`None`:
  every light, the largest)."""
  from raywarp.core.build.variants import tiers

  family = tier.split("_")[0]
  if tier not in tiers.COUNTED | tiers.SAMPLED or family not in tiers.TIERS:
    return ""
  mine, theirs = s.flat(), tiers.TIERS[family].flat()

  def at_most(a: object, b: object) -> bool:
    """Whether count `a` asks no more than `b` (None: unbounded)."""
    return b is None or (a is not None and a <= b)  # pyrefly: ignore[unsupported-operation]

  if all(at_most(mine[f], theirs[f]) for f in COUNT_FIELDS):
    changed = ", ".join(f"{f}={mine[f]!r}" for f in COUNT_FIELDS if mine[f] != theirs[f])
    return f"{tier} for counts within {family}'s: {changed}"
  return ""


def holes_file(tally: Tally) -> dict[str, object]:
  """The holes as `zero_holes.json` lists them: each with its route (the fields that leave the
  nearest tier, or the tier whose key the fill lacks); the cost anomalies alike."""

  def record(h: Hole, why: object) -> dict[str, object]:
    """One hole or anomaly."""
    return {"base": h.base, "settings": {k: repr(v) for k, v in h.settings}, "scene": dict(h.scene), "why": why}

  holes = [record(h, list(h.outside) if h.outside else f"tier {h.tier}: key not filled") for h in tally.holes]
  anomalies = [record(h, h.outside[0]) for h in tally.anomalies]
  return {"points": tally.points, "refused": tally.refused, "served": tally.served, "holes": holes, "anomalies": anomalies}


def report(write: str | None = None) -> int:
  """Prints the depth-1 walk's holes, grouped by what leaves the tiers."""
  t0 = time.time()
  tally = walk(1)
  if write:
    Path(write).write_text(json.dumps(holes_file(tally), indent=1) + "\n")
  groups: dict[tuple[str, ...], list[Hole]] = collections.defaultdict(list)
  for h in tally.holes:
    groups[h.outside or ("(tier " + str(h.tier) + ": key not filled)",)].append(h)
  for why, hs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"{len(hs):5d}  {why}")
    for h in hs[:4]:
      print(f"         {h.base} {dict(h.settings)} {dict(h.scene)} -> {h.tier}")
  print(
    f"points {tally.points}, refused {tally.refused}, served {tally.served}, holes {len(tally.holes)}: {time.time() - t0:.0f} s"
  )
  return 0 if not tally.holes else 1


def merge(paths: list[str], out: str) -> dict[str, object]:
  """The `holes_file`s of a sharded walk (`walk(shard=)`) at `paths` merged into one, written to
  `out`, with the holes counted by the axes they change (`by_axes`) and by what leaves the
  tiers (`by_why`)."""
  parts = [json.loads(Path(p).read_text()) for p in paths]
  holes = [h for part in parts for h in part["holes"]]
  merged: dict[str, object] = {k: sum(part[k] for part in parts) for k in ("points", "refused", "served")}
  axes = collections.Counter(" + ".join(sorted([*h["settings"], *h["scene"]])) for h in holes)
  why = collections.Counter(" ".join(h["why"]) if isinstance(h["why"], list) else h["why"] for h in holes)
  anomalies = [a for part in parts for a in part.get("anomalies", [])]
  kinds = collections.Counter(a["why"].split(" for ")[0] for a in anomalies)
  merged |= {"holes": holes, "by_axes": dict(axes.most_common()), "by_why": dict(why.most_common())}
  merged |= {"anomalies": anomalies, "anomalies_by_tier": dict(kinds.most_common())}
  Path(out).write_text(json.dumps(merged, indent=1) + "\n")
  return merged


def sample(n: int, seed: int = 0, bases: tuple[str, ...] = ("fast", "default", "quality", "unlit")) -> list[dict[str, object]]:
  """A stratified sample of the configurations the route proves served on the plain scene: `n`
  entries ({"base", "changes"}), each settings value in turn as the stratum (cycling), with up
  to two random values of other fields; a combination that compiles or is refused is redrawn."""
  import random

  rng, keys, probe = random.Random(seed), zero.fill_keys(), zero.Probe("plain", zero.Scene())
  values = [(a, v) for a, vs in SETTINGS.items() for v in vs]
  out: list[dict[str, object]] = []
  for i in range(n):
    name = bases[i % len(bases)]
    for _ in range(8):
      changes = dict([values[i % len(values)], *rng.sample(values, rng.randint(0, 2))])
      try:
        s = settings_of(recipes.settings(name), changes)
        r = zero.route(probe, s)
      except ValueError:
        continue
      if r.tier is not None and r.key in keys:
        out.append({"base": name, "changes": changes, "tier": r.tier})
        break
  return out


__all__ = [
  "BASES",
  "SETTINGS",
  "Hole",
  "Tally",
  "cost_anomaly",
  "lattice_walk",
  "merge",
  "random_walk",
  "sample",
  "scene_axes",
  "settings_of",
  "walk",
]
