"""The prebuilt class sets (`raywarp/core/build/variants/class_sets.py`, `tiers.cover`) from a
class census.

  python -m tools.prebuilt.classes --census A.json [--census B.json ...] [--axes C.json] [--write]

Each census is a `internal/benchmarks/lab/scene_keys.py --untiered` output (every row's `class`: the
`tiers.CLASSES` flags of the scene's own build, then its loop and plane forms). The sets:
every class combination a scene uses, then up to a budget their one-class neighbours (one class
added or removed: what a new scene most likely uses) in order of what they save, kept consistent
(glass only with the translucent surface, the layered maps only with the coat or the layers);
every class is always prebuilt beside them. The sets with per-world visibility builds
(`class_sets.VISIBLE`), with deformables (`class_sets.FLEX`) and with aliased poses
(`class_sets.ALIASED`: MuJoCo's layout, shared instances without compact poses): the sets serving
the census rows with visibility / deformables / aliased poses (their `visibility` / `flex` /
`compact` false; `--axes`: a census read for these tables only, e.g. the declared per-world
probes, so that it does not weigh the sets), the bundle's other such scenes routing to their
cheapest superset among them (every class at worst). Prints the counts per source; `--write`
regenerates class_sets.py.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Final

from raywarp.core.build.variants import tiers

TARGET: Final = Path(__file__).resolve().parents[2] / "raywarp" / "core" / "build" / "variants" / "class_sets.py"


def used(row: dict[str, object]) -> frozenset[str]:
  """The classes a census row's own build uses, as `tiers.used_classes` counts them (the headlight
  in a small-loop build)."""
  flags = row["class"]
  if not isinstance(flags, list):
    raise TypeError(f"census row of {row.get('scene')}: class {flags!r} is not a list")
  small = tiers.small_loop(flags[len(tiers.CLASSES)])
  return frozenset(n for n, v in zip(tiers.CLASSES, flags, strict=False) if v or (n == "headlight" and small))


def consistent(s: frozenset[str]) -> bool:
  """Whether a class set can be a scene's: glass is a translucent surface; the layered maps belong
  to the coat or the layers."""
  return ("glass" not in s or "surface" in s) and ("layer_maps" not in s or bool({"coat", "layers"} & s))


def neighbours(s: frozenset[str]) -> set[frozenset[str]]:
  """`s` with one class added or removed (consistent ones)."""
  out = {s ^ {c} for c in tiers.CLASSES}
  return {n for n in out if consistent(n)}


def _cheapest(s: frozenset[str], prebuilt: set[frozenset[str]]) -> frozenset[str]:
  """The set serving a scene using exactly the classes `s` when the bundle holds `prebuilt` (and
  every class): its cheapest superset (`tiers.cover`)."""
  every = frozenset(tiers.CLASSES)
  return min((t for t in (*prebuilt, every) if s <= t), key=lambda t: (tiers.cover_cost(s, t), len(t), sorted(t)))


def _penalty(s: frozenset[str], prebuilt: set[frozenset[str]]) -> int:
  """What a scene using exactly the classes `s` pays when the bundle holds `prebuilt` (and every
  class): the `CLASS_COST` of its cheapest superset's extra classes."""
  return tiers.cover_cost(s, _cheapest(s, prebuilt))


def served(rows: list[dict[str, object]], key: str, prebuilt: set[frozenset[str]]) -> set[frozenset[str]]:
  """The prebuilt sets serving the census rows whose `key` ("visibility", "flex") is set (every
  class not counted: always prebuilt with both)."""
  every = frozenset(tiers.CLASSES)
  return {t for r in rows if r.get(key) for t in (_cheapest(used(r), prebuilt),) if t != every}


def sets(seen: Counter[frozenset[str]], budget: int) -> set[frozenset[str]]:
  """The prebuilt sets: every class set the scenes use (`seen`: how many scenes each), then, up to
  `budget` sets in all, the one-class neighbours (`neighbours`) in order of what they save: the
  scenes of the sets they neighbour times the penalty a scene using exactly the neighbour would pay
  without it. Every class is always prebuilt beside them (not counted)."""
  every = frozenset(tiers.CLASSES)
  chosen = {s for s in seen if s != every}
  weight: Counter[frozenset[str]] = Counter()
  for s, n in seen.items():
    for m in neighbours(s):
      weight[m] += n
  candidates = {m for m in weight if m not in chosen and m != every}
  while len(chosen) < budget and candidates:
    best = max(candidates, key=lambda m: (weight[m] * _penalty(m, chosen), sorted(m)))
    if _penalty(best, chosen) == 0:
      break
    chosen.add(best)
    candidates.remove(best)
  return chosen


# The comments of class_sets.py's tables.
_COMMENTS: Final = {
  "PREBUILT": "# Every class set the bundle holds (each in every sharing layout, tier, GI and loop / plane form).",
  "VISIBLE": "# The sets of PREBUILT also built with per-world visibility (`tiers.cover` of a visibility scene).",
  "FLEX": "# The sets of PREBUILT also built with deformables (`tiers.cover` of a scene with FLEX shapes).",
  "ALIASED": "# The sets of PREBUILT also built with aliased poses (`tiers.cover`: MuJoCo, shared instances).",
}


def render(tables: dict[str, set[frozenset[str]]]) -> str:
  """class_sets.py holding `tables` (name -> sets, sorted: one line per set)."""
  head = TARGET.read_text().split(_COMMENTS["PREBUILT"])[0].split("PREBUILT: Final")[0]
  out = head
  for name, sets_ in tables.items():
    lines = [
      f"  frozenset({{{', '.join(repr(c) for c in sorted(s))}}})," if s else "  frozenset()," for s in sorted(sets_, key=sorted)
    ]
    out += f"{_COMMENTS[name]}\n{name}: Final[tuple[frozenset[str], ...]] = (\n" + "\n".join(lines) + "\n)\n"
    out += "\n" if name != list(tables)[-1] else ""
  return out


def main() -> None:
  """Prints the class sets of the census files; `--write` regenerates class_sets.py."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--census", type=Path, action="append", required=True)
  p.add_argument("--axes", type=Path, action="append", default=[], help="a census for VISIBLE / FLEX / ALIASED only")
  p.add_argument("--budget", type=int, default=170, help="prebuilt sets at most (~115 builds each)")
  p.add_argument("--write", action="store_true")
  a = p.parse_args()
  seen: Counter[frozenset[str]] = Counter()
  every: list[dict[str, object]] = []
  for path in a.census:
    rows = [r for r in json.loads(path.read_text()) if "class" in r]
    mine = Counter(used(r) for r in rows)
    print(f"{path}: {len({r['scene'] for r in rows})} scenes, {len(mine)} class sets")
    seen += mine
    every += rows
  prebuilt = sets(seen, a.budget)
  print(f"class sets: {len(seen)} used, {len(prebuilt)} prebuilt with the neighbours (+ every class)")
  tables = {"PREBUILT": prebuilt}
  every += [r for path in a.axes for r in json.loads(path.read_text()) if "class" in r]
  for r in every:  # aliased poses: shared instances without compact poses (MuJoCo's layout)
    r["aliased"] = r.get("compact") is False
  for name, key in (("VISIBLE", "visibility"), ("FLEX", "flex"), ("ALIASED", "aliased")):
    tables[name] = served(every, key, prebuilt)
    scenes = sorted({str(r["scene"]) for r in every if r.get(key)})
    print(f"{name}: {len(tables[name])} sets (+ every class) serving {len(scenes)} scenes: {', '.join(scenes)}")
  if a.write:
    TARGET.write_text(render(tables))
    print(f"wrote {TARGET}")


if __name__ == "__main__":
  main()
