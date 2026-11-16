"""The release smoke of a built wheel's native components: the installed raywarp converts the MDL /
MaterialX converter fixtures through its bundled MDL SDK (CPU, no CUDA driver needed) and each
conversion matches the golden digest (tests/adapters/usd/test_rw_converter_golden.py).

  cd /tmp && python -I tools/release/mdl_smoke.py REPO

REPO: the checkout holding the fixtures and the golden (data only: the package is the installed
one, -I and another working directory keep the checkout's off sys.path). Exits non-zero on a
digest that differs or a missing SDK.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from raywarp.scene.model.scene import Scene

FIXTURES = ("mdl", "mtlx", "layered_coat", "layered_fuzz")


def digest(scene: Scene) -> str:
  """The golden test's digest of a converted scene (every array and the metadata, paths by name)."""
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


def main() -> None:
  repo = Path(sys.argv[1])
  import raywarp
  import raywarp.formats.usd
  from raywarp.formats.usd.materials.mdl import sdk

  if Path(raywarp.__file__).is_relative_to(repo):
    raise SystemExit(f"mdl_smoke: raywarp was imported from the checkout ({raywarp.__file__}), not the wheel")
  golden = json.loads((repo / "tests/data/converter_golden.json").read_text())["scenes"]
  failed = []
  for name in FIXTURES:
    own = repo / "tests/data/converter" / f"{name}.usda"
    path = own if own.is_file() else repo / "tests/data/scenes" / f"{name}.usda"
    start = time.perf_counter()
    got = digest(raywarp.formats.usd.read(str(path)).build(threads=1))
    ok = got == golden.get(name)
    print(f"mdl_smoke {name}: {got} {'ok' if ok else f'differs from {golden.get(name)}'} ({time.perf_counter() - start:.1f} s)")
    failed += [] if ok else [name]
  print(f"mdl_smoke: MDL SDK {sdk.version()}, raywarp from {raywarp.__file__}")
  if failed:
    raise SystemExit(f"mdl_smoke: converter output differs for {failed}")


if __name__ == "__main__":
  main()
