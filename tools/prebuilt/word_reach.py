"""Which OptiX modules a settings change alone can reach, the scene fixed (CPU; tooling).

For each scene and recipe: the tier build and module word of the recipe (the build word,
`trace/optix/device/programs.build_word`, and above bit 64 the output word,
`output_layout.OutputLayout.word` of the written outputs: `module_word`), and of every single-field
change a user can make (`probe.toggles`). Every census form of a scene (`forms`: a feature-space
scene of `feature_space.py`, `probe.point_forms`; in the development tree also the benchmark
registry's scenes, its `common.forms`: as the registry renders it, an
MJCF scene also with its own lights only, a code scene also with its per-world declarations) is a
scene of its own here. A change that keeps the
build changes at most the module word (an output change: the output word): the bits it flips are
settings-controlled for that scene and build. The reachable words of a build are then the closure:
every combination of the flipped bits applied to the recipe's word (an upper bound of what combined
changes reach; its size only, `closure`). A change that leaves for another build (another tier, a
code split such as transparency) counts that build's word too. Rows (JSON) per scene and recipe: the
builds reached, the controlled bits, the single changes' words; with `--units U.json`, census rows
(`scene_keys.py`'s form) holding the module records of the recipe's word and of each single change's
word on its build, an output change's only for `PACK_OUTPUTS` (the census and the neighbour records
of the bundle index, `fill
--census`; the units are listed in that order, the recipe word's own first: `own` of them).

  python -m tools.prebuilt.word_reach --scenes anymal_c,office_usd --out OUT.json --units U.json
  python -m tools.prebuilt.word_reach --scenes anymal_c --profiles rtx_pro_6000,l4,a10g \
    --out 'OUT.sm{arch}.json' --units 'sm{arch}/U.json'

`--profiles`: the GPU profiles (`probe.PROFILES`, default rtx_pro_6000) the rows resolve
on, each scene probed once; `{arch}` in `--out` / `--units` names each profile's SM version
(required with several profiles: one census per architecture of the multi-architecture fill,
`tools/prebuilt/fill_all.sh`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from raywarp.core.kernels.abi.output_layout import OutputLayout
from raywarp.settings import RenderSettings
from tools.prebuilt import probe as harness

# The development tree's benchmark registry (its census scenes' forms): absent from a public
# checkout, where the census takes the feature-space scenes only.
_REGISTRY = Path(__file__).resolve().parents[2] / "internal" / "benchmarks" / "lab" / "compile"

RECIPES = ("fast", "default", "quality")


class Unit(TypedDict):
  """A census unit: a PTX key, its module group (the tier) and the module's record
  (`prebuilt.ModuleRecord.json`)."""

  key: str
  group: str
  module: dict[str, object]


class Row(TypedDict, total=False):
  """A scene's reach under one recipe (`_row`): outside every tier, or its build, module word, the
  bits single changes flip, the builds and words they reach, the closure's size and the census
  units (`own`: the recipe word's, first)."""

  scene: str
  recipe: str
  outside: bool
  build: str
  word: int
  flips: dict[str, int]
  builds: dict[str, list[int]]
  own: int
  closure: int
  units: list[Unit]


# The builds reached (their Features as `fill census` writes them, JSON) and the changes reaching
# them (the current profile's: `main` clears it per profile).
BUILDS: dict[str, set[str]] = {}
# The GPU profile the rows resolve on (`probe.PROFILES`; `main` sets it per profile).
PROFILE = "rtx_pro_6000"


# The single output changes whose module words the units list (the module pack's neighbours): the
# outputs a policy adds to the color (depth, segmentation, semantic ids, normal, albedo), the linear
# HDR beside it (evaluation, comparisons: `rgb + hdr`) and the observation-tensor color layouts
# (RGB bytes, planes). Every output change stays on the build (the output word); the others are
# index records only, created on first use (1-21 s). Packing one more output form costs one module
# per fast / default own render word (490 in the w13rel2 census, ~30 % of the pack's 1 652 own
# modules): not done (lanes/W14-FIX.md).
PACK_OUTPUTS = frozenset(
  {
    "outputs.depth",
    "outputs.hdr",
    "outputs.seg",
    "outputs.semantic",
    "outputs.normal",
    "outputs.albedo",
    "outputs.rgb.channels",
    "outputs.rgb.planar",
  }
)
# The output word's place in a module word: above the build word's two 32-bit halves
# (`toggles.HIGH`: its second half's bits start at 32; a shift of 32 dropped that half from every
# census record and OR'ed it into the output word).
OUTPUT_SHIFT = 64


def module_word(word: int, outputs: OutputLayout) -> int:
  """The module word of build word `word` (both halves) and output word `outputs`."""
  return word | outputs.word << OUTPUT_SHIFT


def split_word(module: int) -> tuple[int, int]:
  """(build word, output word) of module word `module` (`module_word`'s inverse)."""
  return module & ((1 << OUTPUT_SHIFT) - 1), module >> OUTPUT_SHIFT


def _key(probe: harness.Probe, s: RenderSettings, path: str = "") -> tuple[str, int] | None:
  """(build id, module word) of settings `s` on the probe's scene (None: outside every tier, or
  settings it cannot honor)."""

  from raywarp.core.kernels.abi.output_layout import layout_of
  from raywarp.core.trace.optix.device import programs
  from raywarp.core.trace.optix.supersample import module_outputs

  try:
    r = probe.resolution(s, PROFILE)
  except ValueError:
    return None
  if r.choice is None or r.choice.tier is None or r.features.traversal != "optix":
    return None
  f = r.features
  word = programs.build_word(f, r.effective, harness.scene_flags(probe, f))
  key = json.dumps({k: repr(v) for k, v in vars(f).items()}, sort_keys=True)
  BUILDS.setdefault(key, set()).add(path)
  return key, module_word(word, layout_of(module_outputs(f, s.written_outputs())))


def _units(probe: harness.Probe, s: RenderSettings, words: set[int]) -> list[Unit]:
  """The census units of module words `words` on the tier build of settings `s`
  (`fill.unit_records`)."""

  from tools.prebuilt import fill

  r = probe.resolution(s, PROFILE)
  build = fill._build(probe, r.features, s, harness.PROFILES[PROFILE][0])
  tier = r.choice.tier if r.choice is not None else None
  if tier is None:
    raise ValueError(f"{probe.name}: settings outside every tier have no census units")
  return [
    Unit(key=k, group=tier, module=rec.json())
    for w in sorted(words)
    for k, rec in fill.unit_records(build, split_word(w)[0], OutputLayout.of_word(split_word(w)[1]))
  ]


def _row(name: str, probe: harness.Probe, recipe: str, required: Mapping[str, object]) -> Row:
  """The reach of `recipe` (with the scene's own `required` settings) on scene `name` (see the
  module doc)."""
  base = harness.recipe(recipe).replace(required)
  home = _key(probe, base)
  if home is None:
    return Row(scene=name, recipe=recipe, outside=True)
  flips: dict[str, int] = {}
  builds: dict[str, set[int]] = {home[0]: {home[1]}}
  for path, _value, s in harness.toggles(base):
    if isinstance(s, str):
      continue  # a value the settings refuse
    k = _key(probe, s, path)
    if k is None:
      continue
    builds.setdefault(k[0], set()).add(k[1])
    if k[0] == home[0] and k[1] != home[1]:
      flips[path] = k[1] ^ home[1]
  bits = 0
  for v in flips.values():
    bits |= v
  words = {b: sorted(ws) for b, ws in builds.items()}
  near = {home[1] ^ v for path, v in flips.items() if not path.startswith("outputs.") or path in PACK_OUTPUTS}
  own = _units(probe, base, {home[1]})
  units = own + _units(probe, base, near - {home[1]})
  # The closure's size (every combination of the flipped bits; not enumerated: with the output
  # bits it is millions of words).
  closure = 1 << bin(bits).count("1")
  return Row(
    scene=name,
    recipe=recipe,
    build=home[0],
    word=home[1],
    flips=flips,
    builds=words,
    own=len(own),
    closure=closure,
    units=units,
  )


def forms(name: str) -> list[harness.Form]:
  """The census forms of scene `name`: a feature-space scene's (`probe.point_forms`), else the
  development tree's benchmark registry's (its `common.forms`).

  Raises:
    SystemExit: a registry scene in a checkout without the development tree.
  """
  if harness.is_point(name):
    return harness.point_forms(name)
  if not _REGISTRY.is_dir():
    raise SystemExit(f"{name}: a benchmark registry scene, only in the development tree (internal/)")
  sys.path[:0] = [str(_REGISTRY)]
  import common  # pyrefly: ignore[missing-import]  # the development tree's compile lab (_REGISTRY)

  return common.forms(name)


def main() -> None:
  """Rows for the scenes (see the module doc)."""
  global PROFILE  # the profile the rows resolve on
  import warp as wp

  p = argparse.ArgumentParser()
  p.add_argument("--scenes", required=True)
  p.add_argument("--out", required=True, help="the rows' file ({arch}: the profile's SM version)")
  p.add_argument("--units", default=None, help="census rows of the single changes' modules ({arch} as --out)")
  p.add_argument("--profiles", default="rtx_pro_6000", help="GPU profiles, comma separated (compile/harness.PROFILES)")
  a = p.parse_args()
  profiles = a.profiles.split(",")
  if len(profiles) > 1 and not ("{arch}" in a.out and (a.units is None or "{arch}" in a.units)):
    raise SystemExit("several --profiles: name each one's files with {arch} in --out and --units")
  rows: dict[str, list[Row]] = {pr: [] for pr in profiles}
  builds: dict[str, dict[str, set[str]]] = {pr: {} for pr in profiles}
  for name in a.scenes.split(","):
    for form in forms(name):
      probe = harness.probe(form.label, form.scene, form.cameras, form.per_world)
      for PROFILE in profiles:
        BUILDS.clear()
        BUILDS.update(builds[PROFILE])
        with wp.ScopedDevice("cpu"):
          for recipe in form.recipes or RECIPES:
            row = _row(form.label, probe, recipe, form.required)
            rows[PROFILE].append(row)
            n = sum(len(w) for w in row.get("builds", {}).values())
            flips = sorted(row.get("flips", {}))
            print(
              f"REACH {PROFILE} {form.label} {recipe} builds {len(row.get('builds', {}))} words {n} flips {flips}", flush=True
            )
        builds[PROFILE] = dict(BUILDS)
  for pr in profiles:
    arch = str(harness.PROFILES[pr][0])
    out = Path(a.out.replace("{arch}", arch))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows[pr], "builds": {k: sorted(v) for k, v in builds[pr].items()}}))
    if a.units is not None:
      keep = ("scene", "recipe", "own", "units")
      units = Path(a.units.replace("{arch}", arch))
      units.parent.mkdir(parents=True, exist_ok=True)
      units.write_text(json.dumps([{k: r.get(k) for k in keep} for r in rows[pr] if r.get("units")]))


if __name__ == "__main__":
  main()
