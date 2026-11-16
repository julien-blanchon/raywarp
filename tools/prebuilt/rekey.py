"""Re-key a kernels bundle across a change of the PTX key or of host code inside kernel-code
modules, without compiling anything (CPU).

A bundle's index names each PTX by its key (`programs.ptx_key`) and vouches for it with the digest
of the kernel code its generation used (`prebuilt.Entry`). A change of what the key holds (its
parts, `programs.ptx_parts`), or an edit of kernel-code units that generate no device code (a host
function such as `ptx_key` itself, imports), leaves every PTX valid but no longer found. `rekey`
writes the bundle again with each entry under its new key and its digest of this tree's code:

- the builds are the fill points' (`fill._points`, resolved on the CPU; their GI programs too),
  each mapped from its key under the old parts (`old`: the parts
  replaced, e.g. {"warp": (("warp-lang", "1.18.0"), ("mujoco-warp", "3.17.0"))}) to its key now;
- an entry is kept only when it was generated from the tree `since` (its digest of that tree's
  code, `units_at`) and every unit its digest covers (`kernel_code.scope`) that changed since is
  one of `ignore` (declared host code). Every other entry is dropped (reported: a refill compiles
  it).

  python -m tools.prebuilt.build rekey --bundle B/sm120 --out OUT --since REV \\
    --old warp='(("warp-lang", "1.18.0"), ("mujoco-warp", "3.17.0"))' \\
    --ignore raywarp/core/trace/optix/device/programs.py: --ignore
    raywarp/core/trace/optix/device/programs.py:ptx_key
"""

from __future__ import annotations

import ast
import dataclasses
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
  from tools.prebuilt.fill import Build

ROOT: Final = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True, slots=True)
class Report:
  """What `rekey` did.

  Attributes:
    kept: entries written under their new key.
    unmapped: index keys no fill build has under the old parts (a census outside this tree's).
    stale: entries not generated from the `since` tree's code.
    changed: entries whose code changed since beyond `ignore`, by key: the changed units.
  """

  kept: int
  unmapped: list[str]
  stale: list[str]
  changed: dict[str, list[str]]


def units_at(rev: str) -> dict[str, str]:
  """The kernel code's parts and digests (`kernel_code.units`) at git revision `rev` of this
  repository (a file the revision lacks has no parts)."""
  from raywarp.cache.code_version import code_units
  from raywarp.core.build.kernel_cache import kernel_code

  files = []
  for name, path in sorted(kernel_code.sources().items()):
    rel = path.resolve().relative_to(ROOT).as_posix()
    shown = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT, capture_output=True, check=False)
    if shown.returncode == 0:
      files.append((name, shown.stdout))
  return code_units(files)


def key_map(builds: Iterable[Build], old: Mapping[str, object]) -> dict[str, str]:
  """Old key -> new key of `builds` and their GI programs (`programs.ptx_parts`, the parts `old`
  replaced in the old keys)."""
  from raywarp.core.trace.optix.device import programs

  out: dict[str, str] = {}
  for b in builds:
    for gi in (False, True) if programs.gi_programs(b.features) else (False,):
      parts = programs.ptx_parts(b.features, b.flags, supersampled=not gi and b.supersampled, arch=b.arch, gi=gi)
      before = parts._replace(**old)  # pyrefly: ignore[bad-argument-type]  # the old parts' own types
      out[programs.key_of(before)] = programs.key_of(parts)
  return out


def rekey(
  bundle: Path, out: Path, keys: Mapping[str, str], since: Mapping[str, str], ignore: frozenset[str]
) -> tuple[Path, Report]:
  """`bundle` (`<package>/sm<arch>/`) written to `out`/<package>/sm<arch>/ with each entry of its
  entries under its new key (`keys`, old -> new) and its digest of this tree's code, when it is the
  code of tree `since` (`units_at`) but for the units `ignore`; its Warp modules and PTX files
  copied as they are. Returns the bundle and the report."""
  from raywarp.core.build.kernel_cache import kernel_code, prebuilt
  from tools.prebuilt.build import _seal, read_entries, write_entries

  now = kernel_code.units()
  moved = {u for u in set(since) | set(now) if since.get(u) != now.get(u)}
  index = read_entries(bundle)
  new: dict[str, object] = {}
  unmapped, stale, changed = [], [], {}
  for key, raw in index.items():
    entry = prebuilt.Entry.of(raw)
    if key not in keys:
      unmapped.append(key)
    elif kernel_code.digest(entry.reach, dict(since)) != entry.code:
      stale.append(key)
    elif extra := sorted(_covered(entry.reach, moved) - ignore):
      changed[key] = extra
    else:
      new[keys[key]] = dataclasses.replace(entry, code=kernel_code.digest(entry.reach)).json()
  manifest = prebuilt.Manifest.read(bundle / prebuilt.MANIFEST)
  target = out / bundle.parent.name / bundle.name
  shutil.rmtree(target, ignore_errors=True)
  shutil.copytree(bundle, target)
  write_entries(target, new)
  _seal(target, dataclasses.replace(manifest, content=""))
  return target, Report(kept=len(new), unmapped=unmapped, stale=stale, changed=changed)


def _covered(reach: frozenset[str], moved: set[str]) -> set[str]:
  """The units of `moved` a digest of reach `reach` covers (`kernel_code.scope`: functions in it,
  every other part; a function of the reach either tree lacks too)."""
  from raywarp.core.build.kernel_cache import kernel_code

  return {u for u in moved if reach == kernel_code.ALL or not kernel_code.function(u) or u in reach}


def parse_old(items: list[str]) -> dict[str, object]:
  """`--old NAME=VALUE` items as key parts (VALUE a Python literal, `ast.literal_eval`)."""
  out: dict[str, object] = {}
  for item in items:
    name, sep, value = item.partition("=")
    if not sep:
      raise SystemExit(f"--old {item!r}: NAME=VALUE")
    out[name] = ast.literal_eval(value)
  return out
