"""The keys of a kernel fill's inputs: whether an architecture's bundle, a listing part or a scene's
census must be made again, computed in seconds on any machine (git and the standard library only:
the GitHub runner, before any job).

A bundle is a pure function of the committed tree's code that reaches it and of the compilers and
releases it was built with; its key digests exactly those (`parts`):
- `code`: the git tree ids of raywarp/ (kernel code, routing, the Warp modules' sources) and of
  the fill tooling (tools/prebuilt but its orchestration and scene list, tools/release/kernels.py:
  the layout), tools/native/hook.py (the OptiX headers' commit and the native inputs) and
  tools/hf/pins.env (the job image: its CUDA);
- `locked`: the uv.lock entries of the releases a bundle depends on (Warp: its NVRTC and module
  hashes; MuJoCo Warp: the simulator modules it bundles; MuJoCo);
- the architecture's own inputs: SM version, backend, GPU profile, the pack's driver and groups, the
  fill's options (ONLY, FILL_ARGS, NICHE);
- on an OptiX architecture, the census's (`census_key`: the scene list, the census code
  internal/benchmarks/ when the tree has it), whose module records the bundle carries.
A raw tree id (not a syntax digest): a comment inside a Warp function changes Warp's module hash.

  python tools/prebuilt/buildkey.py bundle --arch 120 --input backend=optix ... # one line: the key
  python tools/prebuilt/buildkey.py census --scenes FILE                         # the census key
  python tools/prebuilt/buildkey.py list --arch 120 --input ...                  # a listing's key
  python tools/prebuilt/buildkey.py changed --keys K.json --pins kernels.json --bucket B

`changed`: the architectures to build, those of K.json (arch -> key) whose key differs from the
one the pins record for them (format 4 `key`; none recorded, or pins of another bucket than B:
changed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
# The code a bundle is made from (git paths: trees or files).
CODE: Final = ("raywarp", "tools/prebuilt", "tools/release/kernels.py", "tools/native/hook.py", "tools/hf/pins.env")
# Files under those paths that do not change a bundle: the orchestration (job counts, flavors, caps)
# and the census scene list (an input of its own, `census_key`).
NOT_CODE: Final = frozenset({
  "tools/prebuilt/fill_all.sh",
  "tools/prebuilt/buildkey.py",
  "tools/prebuilt/buckets.env",
  "tools/prebuilt/census_scenes.txt",
})  # fmt: skip
# The census's own code (absent from a public checkout: the census then probes fs_* scenes only).
CENSUS_CODE: Final = ("internal/benchmarks",)
# The locked releases a bundle depends on (uv.lock package names).
LOCKED: Final = ("warp-lang", "mujoco-warp", "mujoco")


def _git(*args: str) -> str:
  """`git <args>` in the repository (stdout, stripped)."""
  return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def tree_digest(paths: tuple[str, ...], rev: str = "HEAD") -> str:
  """The digest of the committed content under `paths` at `rev` (every blob id with its path, but
  `NOT_CODE`; a path the tree lacks counts as absent)."""
  listed = _git("ls-tree", "-r", rev, "--", *paths).splitlines()
  h = hashlib.blake2b(digest_size=12)
  for line in sorted(listed):
    meta, path = line.split("\t", 1)
    if path not in NOT_CODE:
      h.update(f"{path}\0{meta.split()[2]}\0".encode())
  return h.hexdigest()


def locked(lock: Path = ROOT / "uv.lock") -> str:
  """The locked versions (and sources) of `LOCKED`: every resolution of each, sorted (each
  `[[package]]` table's name, version and source lines: any Python 3, no TOML parser)."""
  rows = []
  for table in lock.read_text().split("\n[[package]]\n")[1:]:
    fields = {k: v.strip('"') for k, v in _FIELD.findall(table.split("\n\n", 1)[0])}
    if fields.get("name") in LOCKED:
      rows.append(f"{fields['name']}=={fields.get('version', '')} {fields.get('source', '')}")
  return ";".join(sorted(rows))


# A `[[package]]` table's top-level `key = value` line (name, version, source).
_FIELD: Final = re.compile(r"^(name|version|source) = (.*)$", re.MULTILINE)


def digest(parts: dict[str, str]) -> str:
  """The key of `parts` (name -> value)."""
  return hashlib.blake2b(json.dumps(parts, sort_keys=True).encode(), digest_size=12).hexdigest()


def code_parts(rev: str = "HEAD") -> dict[str, str]:
  """The parts every key shares: the code and the locked releases."""
  return {"code": tree_digest(CODE, rev), "locked": locked()}


def census_key(scenes: Path, rev: str = "HEAD") -> str:
  """The census's key: `code_parts`, its own code and its scene list (`scenes`, whitespace
  separated, order-free)."""
  names = sorted(set(scenes.read_text().split()))
  return digest({**code_parts(rev), "census": tree_digest(CENSUS_CODE, rev), "scenes": " ".join(names)})


def keyed(inputs: list[str], rev: str = "HEAD") -> str:
  """The key of `code_parts` and `inputs` ("NAME=VALUE")."""
  extra = dict(i.split("=", 1) for i in inputs)
  return digest({**code_parts(rev), **{f"input.{k}": v for k, v in extra.items()}})


def changed(keys: dict[str, str], pins: Path, bucket: str) -> list[str]:
  """The architectures of `keys` whose bundle the pins `pins` do not record with that key on
  storage bucket `bucket` (pins of another bucket: every one)."""
  try:
    data = json.loads(pins.read_text())
  except (OSError, ValueError):
    data = {}
  bundles = data.get("bundles", {}) if data.get("bucket") == bucket else {}
  return [a for a, k in keys.items() if not isinstance(bundles.get(a), dict) or bundles[a].get("key") != k]


def main(argv: list[str]) -> None:
  """`bundle`, `list`, `census`, `changed` (see the module doc)."""
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="cmd", required=True)
  for name in ("bundle", "list"):
    s = sub.add_parser(name)
    s.add_argument("--arch", required=True)
    s.add_argument("--input", action="append", default=[], metavar="NAME=VALUE")
  c = sub.add_parser("census")
  c.add_argument("--scenes", type=Path, required=True)
  ch = sub.add_parser("changed")
  ch.add_argument("--keys", type=Path, required=True, help="JSON: arch -> bundle key")
  ch.add_argument("--pins", type=Path, required=True)
  ch.add_argument("--bucket", required=True, help="the kernels bucket the release publishes to")
  a = p.parse_args(argv)
  if a.cmd in ("bundle", "list"):
    print(keyed([f"kind={a.cmd}", f"arch={a.arch}", *a.input]))
  elif a.cmd == "census":
    print(census_key(a.scenes))
  else:
    print(" ".join(changed(json.loads(a.keys.read_text()), a.pins, a.bucket)))


if __name__ == "__main__":
  main(sys.argv[1:])
