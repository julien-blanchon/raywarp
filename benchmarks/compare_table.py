"""Markdown tables (and bar-chart data) of a renderer comparison's results JSON
(`benchmarks/results/compare.json`; method: docs/benchmark.md, "Comparison").

  python benchmarks/compare_table.py benchmarks/results/compare.json            # the README tables
  python benchmarks/compare_table.py benchmarks/results/compare.json --bars     # bar data

Standard library only. Tables: (1) franka, RGB: images/s of every renderer and mode at every
(resolution, worlds); (2) franka at 512 px x 512 worlds: ms/frame for RGB and RGB + depth, first
frame, peak GPU memory, FLIP against the Mitsuba reference (512 px, 1 world; renderers that render
the same MJCF with the same lights only); (3) raywarp on every scene: images/s, FAST and default.
(4) the Newton + USD model (`g1_usd`): images/s and the 512 px x 512 worlds table. A cell
without a measurement says why: "err" (the renderer failed: the JSON's `error`), "skip"
(the stage ran out of time), "-" (not run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Renderers whose franka image is the same MJCF under the same lights as the Mitsuba reference: its
# FLIP is a quality number. The others rebuild the scene or light it differently (the JSON's
# `renderers` notes): their FLIP is not shown.
SAME_SCENE = ("raywarp", "mujoco_warp")
HEADLINE = (512, 512)


def config_label(res: int, worlds: int) -> str:
  return f"{res}² x {worlds}"


def number(v: float) -> str:
  """Compact: 12.3k, 1.2M, 950."""
  if v >= 1e6:
    return f"{v / 1e6:.2f}M"
  if v >= 1e4:
    return f"{v / 1e3:.0f}k"
  if v >= 1e3:
    return f"{v / 1e3:.1f}k"
  return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def cell(row: dict | None, field: str, fmt) -> str:
  if row is None:
    return "-"
  if row.get("status") == "error":
    return "err"
  if row.get("status") == "skipped":
    return "skip"
  value = row.get(field)
  return "-" if value is None else fmt(value)


def label(data: dict, renderer: str, mode: str) -> str:
  return f"{data['renderers'].get(renderer, {}).get('label', renderer)} {mode}"


def index(rows: list[dict]) -> dict[tuple, dict]:
  return {(r["renderer"], r["mode"], r["scene"], r["res"], r["worlds"], r["outputs"]): r for r in rows}


def series(rows: list[dict], scene: str) -> list[tuple[str, str]]:
  """(renderer, mode) pairs of `scene` in first-seen order, raywarp first."""
  seen: list[tuple[str, str]] = []
  for r in rows:
    k = (r["renderer"], r["mode"])
    if r["scene"] == scene and k not in seen:
      seen.append(k)
  return sorted(seen, key=lambda k: k[0] != "raywarp")


def configs(rows: list[dict], scene: str) -> list[tuple[int, int]]:
  return sorted({(r["res"], r["worlds"]) for r in rows if r["scene"] == scene}, key=lambda c: (c[0], c[1]))


def throughput_table(data: dict, scene: str, outputs: str, only: str | None = None) -> str:
  rows = [r for r in data["rows"] if only is None or r["renderer"] == only]
  ix, grid = index(rows), configs(rows, scene)
  head = "| renderer | " + " | ".join(config_label(*c) for c in grid) + " |"
  lines = [head, "|---|" + "---:|" * len(grid)]
  for renderer, mode in series(rows, scene):
    cells = [cell(ix.get((renderer, mode, scene, res, worlds, outputs)), "images_per_s", number) for res, worlds in grid]
    lines.append(f"| {label(data, renderer, mode)} | " + " | ".join(cells) + " |")
  return "\n".join(lines)


def headline_table(data: dict, scene: str = "franka") -> str:
  ix, res, worlds = index(data["rows"]), *HEADLINE
  first: dict[tuple[str, str], float] = {}
  for r in data["rows"]:
    if r["scene"] == scene and r.get("first_frame_s") is not None:
      first.setdefault((r["renderer"], r["mode"]), r["first_frame_s"])
  flips = (data.get("quality") or {}).get("flip", {})
  lines = [
    f"| renderer ({config_label(res, worlds)}) | ms/frame RGB | images/s RGB | ms/frame RGB+depth | first frame"
    " | peak GPU memory | FLIP vs Mitsuba ↓ |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for renderer, mode in series(data["rows"], scene):
    rgb = ix.get((renderer, mode, scene, res, worlds, "rgb"))
    depth = ix.get((renderer, mode, scene, res, worlds, "rgb+depth"))
    f = first.get((renderer, mode))
    flip = flips.get(f"{renderer}/{mode}") if renderer in SAME_SCENE and scene == "franka" else None
    lines.append(
      f"| {label(data, renderer, mode)} | {cell(rgb, 'ms_per_frame', lambda v: f'{v:.1f}')} | "
      f"{cell(rgb, 'images_per_s', number)} | "
      f"{cell(depth, 'ms_per_frame', lambda v: f'{v:.1f}')} | {'-' if f is None else f'{f:.1f} s'} | "
      f"{cell(rgb, 'peak_mem_mb', lambda v: f'{v / 1024:.1f} GB')} | {'-' if flip is None else f'{flip:.3f}'} |"
    )
  return "\n".join(lines)


def tables(data: dict) -> str:
  gpu = ", ".join(data["meta"].get("gpu") or []) or "GPU not recorded"
  parts = [
    f"Franka (MJCF), RGB, images/s ({gpu}):\n\n" + throughput_table(data, "franka", "rgb"),
    f"Franka at {config_label(*HEADLINE)}:\n\n" + headline_table(data),
  ]
  if any(r["scene"] == "g1_usd" for r in data["rows"]):
    parts += [
      "Newton + USD (Unitree G1 from its USD stage, replicated), RGB, images/s:\n\n" + throughput_table(data, "g1_usd", "rgb"),
      f"Newton + USD at {config_label(*HEADLINE)}:\n\n" + headline_table(data, "g1_usd"),
    ]
  scenes = [s for s in ("showcase", "sponza") if any(r["scene"] == s for r in data["rows"])]
  parts += [f"Raywarp, {s}, RGB, images/s:\n\n" + throughput_table(data, s, "rgb", only="raywarp") for s in scenes]
  return "\n\n".join(parts) + "\n"


def bars(data: dict) -> list[dict]:
  """Flat records for bar charts: every ok row's label, scene, res, worlds, outputs, images/s,
  ms/frame."""
  return [
    {
      "label": label(data, r["renderer"], r["mode"]),
      "renderer": r["renderer"],
      "mode": r["mode"],
      "scene": r["scene"],
      "res": r["res"],
      "worlds": r["worlds"],
      "outputs": r["outputs"],
      "images_per_s": r["images_per_s"],
      "ms_per_frame": r["ms_per_frame"],
    }
    for r in data["rows"]
    if r.get("status") == "ok"
  ]


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("results", type=Path)
  p.add_argument("--bars", action="store_true", help="bar-chart records as JSON instead of the tables")
  a = p.parse_args(argv)
  data = json.loads(a.results.read_text())
  print(json.dumps(bars(data), indent=1) if a.bars else tables(data), end="")


if __name__ == "__main__":
  main()
