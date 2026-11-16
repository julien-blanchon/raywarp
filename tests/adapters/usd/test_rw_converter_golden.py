"""The converter version guard: fixture stages convert to the checked-in golden digests.

`raywarp.formats.usd.version.CONVERTER` is the `.rayw` cache key's converter version (no
source-code hash: a refactor keeps every cache). Any change of what a stage converts to (the USD
adapter, the scene builder, the core modules the build imports, a library) changes a digest here:
bump CONVERTER and regenerate the golden with

  PYTHONPATH=. python tests/adapters/usd/test_rw_converter_golden.py

(the MDL and MaterialX fixtures need the bundled MDL SDK, `raywarp._native`: the script refuses
to write a golden without their digests).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pxr")

import raywarp.formats.usd
from raywarp.formats.usd.version import CONVERTER
from raywarp.scene.model.scene import Scene

# Its own path, not conftest: the file also runs as a script (to rewrite the golden).
ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "data" / "converter_golden.json"
# Fixture stages: preview materials, textures and cut-outs, area / punctual / IES lights,
# environments, instancing and mesh attributes.
FIXTURES = ("materials", "lights_area", "cutouts", "instancing", "environment", "ies_fixtures", "meshes")
# The MDL SDK paths: MDL (library materials, a baked module, a glass, MDL over a preview
# approximation) and MaterialX (networks and a referenced document, through the MaterialX MDL
# generator; and the layered-coat and fuzz feature stages). Their output depends on the MDL SDK, the
# MaterialX library and raywarp's MDL library, all in the cache key.
CONVERTER_DATA = ROOT / "tests" / "data" / "converter"
SDK_FIXTURES = ("mdl", "mtlx", "layered_coat", "layered_fuzz")


def digest(scene: Scene) -> str:
  """The digest of a converted scene: every array and the metadata, file paths by name (the
  machine's absolute paths are not converter output)."""
  h = hashlib.sha256()
  for name, array in sorted(scene.arrays().items()):
    if name.startswith("images.paths."):
      continue
    h.update(f"{name}{array.dtype.str}{array.shape}".encode())
    h.update(array.tobytes())
  h.update(json.dumps([os.path.basename(p) for p in scene.images.paths]).encode())
  meta = {k: v for k, v in scene.meta.items() if k != "source"}
  h.update(json.dumps(meta, sort_keys=True).encode())
  return h.hexdigest()[:24]


def convert(name: str) -> str:
  """The digest of fixture stage `name` (tests/data/converter's, else a benchmark fixture scene)."""
  own = CONVERTER_DATA / f"{name}.usda"
  path = own if own.is_file() else ROOT / "tests" / "data" / "scenes" / f"{name}.usda"
  return digest(raywarp.formats.usd.read(str(path)).build(threads=1))


def test_fixture_conversions_match_the_golden() -> None:
  golden = json.loads(GOLDEN.read_text())
  assert golden["converter"] == CONVERTER, "CONVERTER was bumped: regenerate the golden (module docstring)"
  changed = [name for name in FIXTURES if convert(name) != golden["scenes"].get(name)]
  assert not changed, f"converter output changed for {changed}: bump raywarp.formats.usd.version.CONVERTER (module docstring)"


def test_sdk_fixture_conversions_match_the_golden() -> None:
  pytest.importorskip("MaterialX")
  golden = json.loads(GOLDEN.read_text())
  assert golden["converter"] == CONVERTER, "CONVERTER was bumped: regenerate the golden (module docstring)"
  changed = [name for name in SDK_FIXTURES if convert(name) != golden["scenes"].get(name)]
  assert not changed, f"MDL / MaterialX converter output changed for {changed}: bump raywarp.formats.usd.version.CONVERTER"


if __name__ == "__main__":
  from raywarp.formats.usd.materials.mdl import sdk

  if sdk.api() is None:
    sys.exit("the MDL / MaterialX fixtures need the bundled MDL SDK: golden not written")
  scenes = {n: convert(n) for n in (*FIXTURES, *SDK_FIXTURES)}
  GOLDEN.write_text(json.dumps({"converter": CONVERTER, "scenes": scenes}, indent=1) + "\n")
  sys.stdout.write(f"wrote {GOLDEN}\n")
