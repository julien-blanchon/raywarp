"""Pins warp-lang in pyproject.toml to the exact Warp release of the prebuilt kernels the package
pins (raywarp/core/build/kernel_cache/kernels.json: a bundle serves one Warp release only), for a
release build.

  python3 tools/release/pin_warp.py          # rewrites the dependency, prints it

Standard library only (it runs before any environment exists, in the manylinux image too).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_DEPENDENCY = re.compile(r'"warp-lang[^"]*"')


def main() -> None:
  bundles = json.loads((ROOT / "raywarp/core/build/kernel_cache/kernels.json").read_text())["bundles"]
  releases = {b["warp"] for b in bundles.values()}
  if len(releases) != 1:
    raise SystemExit(f"the pinned bundles name {sorted(releases) or 'no'} Warp releases: one is needed")
  pin = f'"warp-lang=={releases.pop()}"'
  path = ROOT / "pyproject.toml"
  text = path.read_text()
  head, sep, rest = text.partition("dependencies = [")
  body, end, tail = rest.partition("]")
  if not sep or len(_DEPENDENCY.findall(body)) != 1:
    raise SystemExit("pyproject.toml: [project] dependencies hold no single warp-lang requirement")
  path.write_text(head + sep + _DEPENDENCY.sub(pin, body) + end + tail)
  print(f"pyproject.toml: {pin}")


if __name__ == "__main__":
  main()
