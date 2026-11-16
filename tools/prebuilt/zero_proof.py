"""The all-K part of the zero-compile proof: the tier route's served test decomposes into
independent groups of Features fields, so a configuration of any number of changed values is
served exactly when each group is.

A configuration renders with tier T's prebuilt build `B` when `tiers.differences(B, F, E,
unread=M)` is empty, F its Features resolved for T (scene-free, on T's axes), E its Features
specialised on the scene, M the fields no frame of the scene reads (`resolve._scene_moot`).
`differences` is a loop over the Features fields; each field's verdict reads its own two values
and a few others (the moot rules of shadows, reflections, sampling and translucent casters, a
class's other fields, the texture slots, the scene's light count). So:

1. **Hubs** (`HUBS`): the few fields `differences` reads beside the one it judges (reflections,
   shadows, textures, normal maps, translucent casters) and the scene inputs of the test as
   pseudo-fields (`SCENE_NLIGHT`, `MOOT`). Everything below is conditioned on their values.
2. **Coupling** (`couplings`): under each hub assignment, non-hub field m couples to n in build B
   when changing both changes the verdict otherwise than the union of each alone. The groups
   (`groups`) are the connected components over every hub assignment.
3. **Decomposition** (`check_decomposition`): for random targets, the verdict equals the union of
   the verdicts of the target's projections on each group (with its hubs, the other fields at the
   build's own values). Given the hubs, the set of targets B serves is then a product of
   per-group sets: a box.
4. **Boxes** (`box`): each group's served value combinations, exhaustively.

The layer below (settings and scene values -> F, E, M: `resolve.resolve_features`) is checked by
the exhaustive single and pair walks (`zero_space.walk`) and the random walk of 3 to 10 values
(`zero_space.random_walk`, 10^6 points on HF): a hole is a reachable F outside every tier's box.

  python -m tools.prebuilt.zero_proof OUT.json [WORKERS]   # couplings, groups, decomposition, boxes
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import functools
import itertools
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from tools.prebuilt import fill, zero

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features

# Pseudo-fields: the scene's light count (`differences`' `scene`: light caps, small forms) and the
# fields no frame of the scene reads (`unread`).
SCENE_NLIGHT: Final = "scene.nlight"
MOOT: Final = "scene.moot"
# The hub fields: those `differences` reads beside the field it judges (the moot rules of
# reflections, shadows, sampling, texture taps, normal maps, translucent casters; the scene's light
# count and moot set). The analysis is conditioned on each assignment of them (`hub_values`).
HUBS: Final = ("glossy", "shadows", "textures", "normal_map", "translucent_shadows", SCENE_NLIGHT, MOOT)
# The scene light counts and moot sets the hub assignments take (the thresholds of the caps' and
# small forms' tests: one light, past the small form, past the caps).
HUB_NLIGHT: Final = (1, 64)
# Random contexts (the non-hub fields at random values) per hub assignment in which couplings are
# searched, beside the build's own target.
CONTEXTS: Final = 3
# Field values beyond those the fill's builds hold (the settings' domains,
# `core/build/variants/nearby.DOMAIN`).
EXTRA: Final[dict[str, tuple[object, ...]]] = {
  "shadow_samples": (1, 2, 3, 4, 8),
  "anisotropy": (1, 2, 4, 8, 16),
  "budget": (0, 1, 2, 4, 8, 16),
  "light_cap": (0, 1, 2, 4, 8, 16),
  "hit_light_cap": (0, 1, 2, 4, 8, 16),
  "light_forms": (0, 1, 2, 3, 4, 5),
  "glossy": (0, 1),
  "reflection_shadows": (0, 1),
  "sun_disc": (False, True),
  "spp": (1, 2),
}
# The largest group product `box` enumerates.
BOX_LIMIT: Final = 1 << 16
# Fields the route never compares (`tiers._UNCOMPARED`) or that are a build's own properties.
SKIP: Final = frozenset({"switches", "bound", "word", "out_word", "scene_free", "traversal"})


@functools.cache
def tier_builds() -> dict[str, tuple[Features, ...]]:
  """Every OptiX fill build by tier (`fill._layout_points` on `zero.base_points`)."""
  from raywarp.core.build.variants import class_sets

  out: dict[str, dict[str, Features]] = collections.defaultdict(dict)
  f = fill._Fill(backend="optix", profile=zero.PROFILE, arch=zero.fill_arch(), modules=False, only=None)
  tables = class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED
  sets = fill._sets("mesh", "shared", "optix")
  try:
    for name, point in zero.base_points():
      probe = zero.Probe(name, point)
      for row, build in fill._layout_points(f, probe, "shared", sets):
        if build is not None:
          out[str(row["tier"])][build.key] = build.features
  finally:
    class_sets.PREBUILT, class_sets.VISIBLE, class_sets.FLEX, class_sets.ALIASED = tables
  return {t: tuple(bs.values()) for t, bs in out.items()}


def domains(builds: tuple[Features, ...]) -> dict[str, tuple[object, ...]]:
  """Each compared field's values: booleans both, the others those of `builds` and `EXTRA`; the
  pseudo-fields' (light counts around the thresholds, the moot sets)."""
  from raywarp.core.build.variants import resolve

  out: dict[str, tuple[object, ...]] = {}
  for field in dataclasses.fields(builds[0]):
    name = field.name
    if name in SKIP:
      continue
    seen = {getattr(b, name) for b in builds} | set(EXTRA.get(name, ()))
    if all(isinstance(v, bool) for v in seen):
      seen = {False, True}
    out[name] = tuple(sorted(seen, key=repr))
  out[SCENE_NLIGHT] = (0, 1, 2, 3, 4, 5, 8, 64)
  translucent = resolve._TRANSLUCENT_FIELDS
  out[MOOT] = (frozenset(), frozenset({"alpha_cutout"}), translucent, translucent | {"alpha_cutout"})
  return out


def verdict(build: Features, values: dict[str, object]) -> frozenset[str]:
  """The fields of target `values` (Features fields and pseudo-fields over `build`'s own) that
  `build` cannot render (`tiers.differences`)."""
  from raywarp.core.build.variants import tiers

  target = with_fields(build, {k: v for k, v in values.items() if k not in (SCENE_NLIGHT, MOOT)})
  nlight = values.get(SCENE_NLIGHT, 1)
  scene = dataclasses.replace(target, nlight=nlight if isinstance(nlight, int) else 1, light_cap=0, hit_light_cap=0)
  moot = values.get(MOOT, frozenset())
  unread = frozenset(str(m) for m in moot) if isinstance(moot, frozenset) else frozenset()
  return frozenset(tiers.differences(build, target, scene, unread=unread))


def with_fields(build: Features, values: Mapping[str, object]) -> Features:
  """`build` with Features fields `values` (the walk's domains: each a value its field takes,
  `domains`), set on a copy (the values come from the fields' own domains, so no field takes
  another type)."""
  out = copy.copy(build)
  for name, value in values.items():
    object.__setattr__(out, name, value)
  return out


class Decomposition(TypedDict):
  """One build's decomposition (`tier_report`): its coupling groups of several fields, the count
  of single-field groups, each group's box (served, total), the counter-examples."""

  groups: list[list[str]]
  singles: int
  boxes: dict[str, tuple[int, int]]
  decomposition_failures: int
  examples: list[dict[str, str]]


def _own(build: Features, doms: dict[str, tuple[object, ...]]) -> dict[str, object]:
  """`build`'s own target: its values, the pseudo-fields at 1 light and nothing moot."""
  return {**{n: getattr(build, n) for n in doms if not n.startswith("scene.")}, SCENE_NLIGHT: 1, MOOT: frozenset()}


def hub_values(doms: dict[str, tuple[object, ...]]) -> list[dict[str, object]]:
  """Every assignment of the hub fields (`HUBS`; the light count at `HUB_NLIGHT`, the moot set
  empty or the translucent fields with the alpha test)."""
  values = {h: doms[h] for h in HUBS}
  values[SCENE_NLIGHT] = HUB_NLIGHT
  values[MOOT] = (doms[MOOT][0], doms[MOOT][-1])
  return [dict(zip(HUBS, c, strict=True)) for c in itertools.product(*(values[h] for h in HUBS))]


def couplings(build: Features, doms: dict[str, tuple[object, ...]], seed: int = 0) -> set[frozenset[str]]:
  """The pairs {m, n} of non-hub fields whose verdicts in `build` are not independent under some
  hub assignment: in the build's own target or one of `CONTEXTS` random ones (the non-hub fields
  at random values), the verdict of m and n changed together is not the union of each changed
  alone."""
  rng, names = random.Random(seed), [n for n in doms if n not in HUBS]
  edges: set[frozenset[str]] = set()
  for hubs in hub_values(doms):
    contexts = [_own(build, doms), *({n: rng.choice(doms[n]) for n in names} for _ in range(CONTEXTS))]
    for ctx in ({**c, **hubs} for c in contexts):
      alone = {(n, v): verdict(build, {**ctx, n: v}) for n in names for v in doms[n]}
      for m, n in itertools.combinations(names, 2):
        if frozenset((m, n)) in edges:
          continue
        if any(
          verdict(build, {**ctx, m: vm, n: vn}) != alone[m, vm] | alone[n, vn] for vm, vn in itertools.product(doms[m], doms[n])
        ):
          edges.add(frozenset((m, n)))
  return edges


def groups(names: list[str], edges: set[frozenset[str]]) -> list[frozenset[str]]:
  """The connected components of the coupling graph over `names`."""
  parent = {n: n for n in names}

  def root(n: str) -> str:
    """Union-find root of `n`."""
    while parent[n] != n:
      parent[n] = parent[parent[n]]
      n = parent[n]
    return n

  for e in edges:
    a, b = sorted(e)
    parent[root(a)] = root(b)
  out: dict[str, set[str]] = collections.defaultdict(set)
  for n in names:
    out[root(n)].add(n)
  return sorted((frozenset(g) for g in out.values()), key=lambda g: (-len(g), sorted(g)))


def check_decomposition(
  build: Features, doms: dict[str, tuple[object, ...]], parts: list[frozenset[str]], n: int, seed: int = 1
) -> list[dict[str, str]]:
  """Random targets (every field, hubs too, at a random value) whose verdict is not the union of
  their projections' on `parts` (each part's values and the target's hubs, the others the
  build's own): empty when the test decomposes given the hubs."""
  rng, own = random.Random(seed), _own(build, doms)
  bad: list[dict[str, str]] = []
  for _ in range(n):
    t = {k: rng.choice(vs) for k, vs in doms.items()}
    base = {**own, **{h: t[h] for h in HUBS}}
    joined = frozenset().union(verdict(build, base), *(verdict(build, {**base, **{k: t[k] for k in g}}) for g in parts))
    if verdict(build, t) != joined:
      bad.append({k: repr(v) for k, v in t.items()})
  return bad


def box(build: Features, doms: dict[str, tuple[object, ...]], part: frozenset[str]) -> tuple[int, int]:
  """(served, total) value combinations of group `part` in `build` (the others its own); (-1,
  total) when the product exceeds `BOX_LIMIT` (not enumerated)."""
  own, names = _own(build, doms), sorted(part)
  total = 1
  for k in names:
    total *= len(doms[k])
  if total > BOX_LIMIT:
    return -1, total
  combos = itertools.product(*(doms[k] for k in names))
  served = sum(not verdict(build, {**own, **dict(zip(names, c, strict=True))}) for c in combos)
  return served, total


def tier_report(tier: str, per_tier: int = 1, samples: int = 4000) -> list[Decomposition]:
  """Tier `tier`'s first `per_tier` fill builds: the coupling groups of the non-hub fields, the
  decomposition's counter-examples among `samples` random targets, each group's box size (at the
  build's own hubs)."""
  builds = tier_builds()[tier]
  doms = domains(builds)
  rows: list[Decomposition] = []
  for build in builds[:per_tier]:
    parts = groups([n for n in doms if n not in HUBS], couplings(build, doms))
    bad = check_decomposition(build, doms, parts, samples)
    multi = [g for g in parts if len(g) > 1]
    rows.append(
      Decomposition(
        groups=[sorted(g) for g in multi],
        singles=sum(len(g) == 1 for g in parts),
        boxes={"+".join(sorted(g)): box(build, doms, g) for g in multi},
        decomposition_failures=len(bad),
        examples=bad[:3],
      )
    )
  return rows


def report(workers: int) -> dict[str, list[Decomposition]]:
  """Every tier's `tier_report`, `workers` processes forked after the fill's builds."""
  import multiprocessing

  names = list(tier_builds())
  with multiprocessing.get_context("fork").Pool(workers) as pool:
    return dict(zip(names, pool.map(tier_report, names), strict=True))


def main(argv: list[str]) -> int:
  """CLI: the report (`argv[1]` worker processes, default 8) as JSON to `argv[0]`; non-zero when
  a decomposition fails."""
  r = report(int(argv[1]) if len(argv) > 1 else 8)
  Path(argv[0]).write_text(json.dumps(r, indent=1, default=str) + "\n")
  for tier, rows in r.items():
    for row in rows:
      print(tier, [len(g) for g in row["groups"]], row["singles"], "failures", row["decomposition_failures"])
  failures = sum(row["decomposition_failures"] for rows in r.values() for row in rows)
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
