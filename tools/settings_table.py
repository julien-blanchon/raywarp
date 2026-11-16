"""Generated tables of the docs: the source of truth is the code (the settings' field metadata,
`RenderSettings.fields_table()`, so the docs cannot list a compiled-in setting as runtime (D1);
the OpenPBR inputs' coverage, `scene.model.openpbr.COVERAGE`; the MuJoCo material mapping of
`raywarp.integrations.mujoco.materials`).

  python tools/settings_table.py          # check: exit 1 when a generated region is stale
  python tools/settings_table.py --write  # rewrite the regions in place

Regions (between `<!-- <name>:begin ... -->` and `<!-- <name>:end -->` lines):
  settings-table  docs/settings.md: every field, its default, compiled in or runtime, rebuilds
  runtime-fields  docs/settings.md: the runtime fields, by group
  openpbr-inputs    docs/scenes.md: every OpenPBR input, its record, default and coverage
  mujoco-materials  docs/scenes.md: MuJoCo material fields -> OpenPBR (`openpbr_materials`'s table)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Final

from raywarp.settings import RenderSettings

ROOT = Path(__file__).resolve().parent.parent
FILES = {
  ROOT / "docs" / "settings.md": ("settings-table", "runtime-fields"),
  ROOT / "docs" / "scenes.md": ("openpbr-inputs", "mujoco-materials"),
}
# Column where the meaning starts in the mapping table of `raywarp.integrations.mujoco.materials`'
# docstring.
_MEANING: Final = 27
_REBUILDS = {"gi": "rebuilds the GI probe volume", "light_grid": "rebuilds the light grid"}
# Compile-time fields (part of `Features`) that the prebuilt tier builds compile in and read at run
# time: a change switches code on or off without another kernel (`core/build/variants/tiers.py`
# `SWITCHES`, `SETTING_FIELDS`, the class bits; `reflections.enabled` a launch parameter; the
# grading a deferred display module only, no render kernel).
_TIER_SWITCHES: Final = frozenset({
  "shadows.enabled",
  "environment.background",
  "lighting.emissive_lights",
  "reflections.enabled",
  "transparency.enabled",
  "surfaces.textures",
  "surfaces.backface_culling",
  "grading.enabled",
})  # fmt: skip
# The output requests that change the build: the motion history (every other output is a bit of
# the run-time output word).
_BUILD_OUTPUTS: Final = frozenset({"outputs.motion"})


def _value(value: object) -> str:
  """A default as code: frozensets as sorted sets, everything else as its repr."""
  if isinstance(value, frozenset):
    return "{" + ", ".join(repr(v) for v in sorted(value)) + "}"
  return repr(value)


def settings_table() -> str:
  """The markdown table of every field."""
  rows = ["| setting | default | kind | a change also |", "| --- | --- | --- | --- |"]
  for f in RenderSettings.fields_table():
    kind = "runtime" if f.runtime else _compiled_kind(f.name)
    rebuild = "" if f.rebuild is None else _REBUILDS[f.rebuild]
    rows.append(f"| `{f.name}` | `{_value(f.default)}` | {kind} | {rebuild} |")
  return "\n".join(rows)


def _compiled_kind(name: str) -> str:
  """The kind of compile-time field `name`: an output (run-time output word but `_BUILD_OUTPUTS`),
  a tier switch (`_TIER_SWITCHES`), else compiled in."""
  if name.startswith("outputs.") and not name.startswith(tuple(_BUILD_OUTPUTS)):
    return "output word"
  return "tier switch" if name in _TIER_SWITCHES else "compiled in"


def runtime_fields() -> str:
  """The runtime fields as one bullet per group."""
  groups: dict[str, list[str]] = {}
  for f in RenderSettings.fields_table():
    if f.runtime:
      group, _, name = f.name.partition(".")
      groups.setdefault(group, []).append(f"`{name}`" + (" (rebuilds)" if f.rebuild else ""))
  return "\n".join(f"- `{group}`: {', '.join(names)}" for group, names in groups.items())


def openpbr_inputs() -> str:
  """The markdown table of every OpenPBR input: where it lives (`Material` field, `extras`, or a
  texture only), its default and how the core renders it (`openpbr.COVERAGE`)."""
  from raywarp.scene.model.material import GEOMETRY_INPUTS, Material, OpenPbrExtras
  from raywarp.scene.model.openpbr import COVERAGE

  core, extras = vars(Material()), {f.name: f.default for f in dataclasses.fields(OpenPbrExtras)}
  rows = ["| input | in | default | status | rendered as |", "| --- | --- | --- | --- | --- |"]
  for name, c in COVERAGE.items():
    where, default = ("`Material`", f"`{core[name]!r}`") if name in core else ("`extras`", f"`{extras.get(name)!r}`")
    if name in GEOMETRY_INPUTS:
      where, default = ("`Slot.GEOMETRY_NORMAL`", "") if name == "geometry_normal" else ("`extras.textures`", "")
    rows.append(f"| `{name}` | {where} | {default} | {c.status} | {c.how} |")
  return "\n".join(rows)


def mujoco_materials() -> str:
  """The markdown table of the mapping table in `raywarp.integrations.mujoco.materials`' docstring:
  a row is indented 2 spaces, its meaning starts at column `_MEANING` (on the next line after a
  longer field), continuation lines are indented more."""
  from raywarp.integrations.mujoco import materials

  doc = materials.__doc__ or ""
  start = doc.index("\n\n", doc.index("`openpbr_materials` gives")) + 2
  rows: list[list[str]] = []
  for line in doc[start : doc.index("\n\n", start)].splitlines():
    if line.startswith("   "):
      rows[-1][1] = f"{rows[-1][1]} {line.strip()}".strip()
    elif line[_MEANING - 1 : _MEANING] == " ":
      rows.append([line[:_MEANING].strip(), line[_MEANING:].strip()])
    else:
      rows.append([line.strip(), ""])
  out = ["| MuJoCo | OpenPBR `rw.Material` |", "| --- | --- |"]
  return "\n".join(out + [f"| `{key}` | {value.replace('|', '/')} |" for key, value in rows])


GENERATORS = {
  "settings-table": settings_table,
  "runtime-fields": runtime_fields,
  "openpbr-inputs": openpbr_inputs,
  "mujoco-materials": mujoco_materials,
}


def regenerate(text: str, name: str) -> str:
  """`text` with region `name` replaced by its generated content.

  Raises:
    ValueError: when the region's markers are missing.
  """
  begin, end = f"<!-- {name}:begin", f"<!-- {name}:end -->"
  start = text.find(begin)
  stop = text.find(end)
  if start < 0 or stop < start:
    raise ValueError(f"region {name!r} not found")
  head = text.index("\n", start) + 1  # keep the begin marker line
  return text[:head] + GENERATORS[name]() + "\n" + text[stop:]


def stale() -> list[str]:
  """The (file, region) pairs whose content differs from the generated one."""
  out = []
  for path, names in FILES.items():
    text = path.read_text()
    out += [f"{path.relative_to(ROOT)}: {name}" for name in names if regenerate(text, name) != text]
  return out


def main() -> None:
  """Checks (default) or rewrites (--write) the generated regions."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--write", action="store_true", help="rewrite the regions in place")
  args = parser.parse_args()
  if args.write:
    for path, names in FILES.items():
      text = path.read_text()
      for name in names:
        text = regenerate(text, name)
      path.write_text(text)
    return
  bad = stale()
  if bad:
    print("stale generated regions (run python tools/settings_table.py --write):\n  " + "\n  ".join(bad))
    sys.exit(1)


if __name__ == "__main__":
  main()
