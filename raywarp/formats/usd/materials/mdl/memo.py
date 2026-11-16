"""A disk cache of MDL conversions: a material read again from unchanged files skips the SDK
(module load, distill, the DAG reads).

An entry, `<cache>/mdl/conv-<key>.json`, holds the canonical material (its texture ids as indices
into its texture ops), the mesh UV sets it reads, its texture ops (a file bound as is, or a disk
bake), its report entries and the files it was made from with their stats. The key: the
converter version (`version.CONVERTER`), raywarp's MDL library content, the SDK version and
distiller target, the shader's results key (module, material, authored inputs), the stage frame,
the texture size limit, the search roots. An entry whose files changed (or whose bakes are gone)
is a miss.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Final, final

import numpy as np

from raywarp._base.named import construct_named
from raywarp.cache.paths import write_atomic
from raywarp.formats.usd import version as usd_version
from raywarp.formats.usd.materials.context import MeshNeeds, ShadeContext, Where
from raywarp.formats.usd.materials.mdl import load, sdk
from raywarp.formats.usd.materials.mdl.bake import BAKE_VERSION, Stored
from raywarp.scene.images import files
from raywarp.scene.images.colorspace import ColorSpace
from raywarp.scene.model.material import Material, OpenPbrExtras, RaywarpExt
from raywarp.scene.model.report import LoadReport
from raywarp.scene.model.types import AlphaMode, ImageId, Json, Slot, TextureId, TextureRef, Wrap

# A texture op: ("file", path, slot, colorspace or "", wrap u, wrap v) or ("baked", the bake's
# .npy, transfer, colorspace, peak).
type TextureOp = tuple[str | int | float, ...]
_REPEAT: Final = (int(Wrap.REPEAT), int(Wrap.REPEAT))
# Days a code version's entries may go unused before a read of another version removes them.
_STALE_DAYS: Final = 7


@functools.cache
def _code(version: str) -> str:
  """The code a conversion depends on: the `.rayw` key's converter version (`version.CONVERTER`) and
  MDL library digest, the SDK's `version`, the distiller target and the bake version."""
  text = f"{usd_version.CONVERTER}|{usd_version.mdl_library_digest()}|{version}|{load.TARGET}|{BAKE_VERSION}"
  return hashlib.sha256(text.encode()).hexdigest()[:16]


def path(directory: Path, ctx: ShadeContext, shading_key: tuple[str, ...], roots: list[str]) -> Path:
  """The entry file of a conversion (under its code's directory: a code change leaves the old
  entries apart)."""
  frame = [ctx.frame.meters_per_unit, np.asarray(ctx.frame.stage_from_scene).round(9).tolist()]
  text = json.dumps([list(shading_key), frame, ctx.max_texture_size, roots, load.env_paths()])
  code = _code(sdk.version())
  return directory / "mdl" / f"conv-{code}" / f"{hashlib.sha256(text.encode()).hexdigest()}.json"


def prune(directory: Path) -> None:
  """Removes the conversion entries of other code versions left unused for `_STALE_DAYS` (a
  directory per code version: another tree or an older build may still read its own), and the
  generated MaterialX modules no read used for as long."""
  current = f"conv-{_code(sdk.version())}"
  with contextlib.suppress(OSError):
    for d in (directory / "mdl").glob("conv-*"):
      if d.name != current and time.time() - d.stat().st_mtime > _STALE_DAYS * 86400:
        shutil.rmtree(d, ignore_errors=True)
  # The generated MaterialX modules (`materialx.module_file` touches the ones a read uses).
  for f in (directory / "mdl" / "mtlx").glob("*.mdl"):
    with contextlib.suppress(OSError):
      if time.time() - f.stat().st_mtime > _STALE_DAYS * 86400:
        f.unlink()


@final
@dataclasses.dataclass(slots=True)
class Recorder:
  """The texture ops of one conversion, by the texture id each made."""

  ops: list[TextureOp] = dataclasses.field(default_factory=list)
  index: dict[TextureId, int] = dataclasses.field(default_factory=dict)
  complete: bool = True  # False: a texture with no op (a bake without a disk file)

  def add(self, texture: TextureId, op: TextureOp | None) -> None:
    """Records the op that made `texture` (None: not replayable)."""
    if op is None:
      self.complete = False
    elif texture not in self.index:
      self.index[texture] = len(self.ops)
      self.ops.append(op)


def save(  # noqa: PLR0913  # an entry's parts
  file: Path,
  result: tuple[Material, MeshNeeds],
  rec: Recorder,
  report: LoadReport,
  deps: set[str],
  paths: list[str],
  stats: Stats,
) -> None:
  """Writes the entry of a conversion (best effort: nothing when it is not replayable; a cache
  that cannot be written is warned about, as the bake cache's). `paths`: its module's search
  paths (an entry replays only on the same ones)."""
  mat, needs = result
  if not rec.complete:
    return
  try:
    data = {"material": _material(mat, rec.index), "uv": list(needs.uv), "ops": [list(o) for o in rec.ops]}
  except KeyError:  # a texture no op made
    return
  record = report.to_dict() | {"examples": {}}  # prim paths are the stage's: the replay's own
  data |= {"report": record, "deps": [stats.of(d) for d in sorted(deps)], "paths": paths}
  text = json.dumps(data).encode()
  write_atomic(file, lambda f: f.write(text), "MDL conversion cache")


def read(file: Path, paths: list[str], stats: Stats) -> dict[str, Json] | None:
  """The entry at `file` when its files are unchanged and its module's search paths are
  `paths`, else None."""
  try:
    data = json.loads(file.read_text())
  except (OSError, ValueError):
    return None
  deps = data.get("deps") if isinstance(data, dict) else None
  if not isinstance(deps, list) or any(not isinstance(d, list) or stats.of(str(d[0])) != d for d in deps):
    return None
  return data if data.get("paths") == paths else None


def replay(
  ctx: ShadeContext, prim: Where, data: dict[str, Json], images: dict[str, ImageId]
) -> tuple[Material, MeshNeeds] | None:
  """The material of an entry, its textures made again in `ctx` (`images`: the bake images of
  the read, by bake key: shared with its live bakes), its report entries at `prim`; None for an
  entry that does not replay (a bake file gone, a schema it does not read)."""
  try:
    ops = [op for op in _list(data["ops"]) if isinstance(op, list)]
    texels = {str(op[1]): _texels(str(op[1]), float(str(op[4]))) for op in ops if op[0] == "baked"}
    ids: list[TextureId] = []
    for op in ops:
      if op[0] == "file":
        _, p, slot, cs, wu, wv = op
        ids.append(ctx.texture(str(p), Slot(int(str(slot))), ColorSpace(cs) if cs else None, (int(str(wu)), int(str(wv)))))
        continue
      _, npy, transfer, cs, _ = op
      key = Path(str(npy)).stem  # a bake's file is <key>.npy
      if key not in images:
        images[key] = ctx.builder.add_image(texels[str(npy)], transfer="srgb" if transfer == "srgb" else "linear")
      ids.append(ctx.builder.add_texture(images[key], colorspace=ColorSpace(cs), wrap=_REPEAT))
    result = _unmaterial(data["material"], ids), MeshNeeds(tuple(str(u) for u in _list(data["uv"])))
  # A malformed entry, or a bake file gone or unreadable, is a miss, never the error material.
  except (KeyError, IndexError, TypeError, ValueError, OSError, EOFError):
    return None
  record = LoadReport.from_dict(_dict(data.get("report")))
  for name in (*record.dropped, *record.approximated):
    record.examples[name] = ctx.report.describe(prim)
  ctx.report.merge(record)
  return result


def _texels(npy: str, peak: float) -> Stored:
  """A bake's texels from its file, an emission's divided by its peak."""
  texels = np.load(npy)
  return np.asarray(texels / peak, np.float32) if peak not in (0.0, 1.0) else texels


@final
@dataclasses.dataclass(slots=True)
class Stats:
  """The file stats of one stage read, each file's taken once (its entries share modules and
  textures; a stat on a network mount costs milliseconds)."""

  taken: dict[str, list[Json]] = dataclasses.field(default_factory=dict)

  def of(self, file: str) -> list[Json]:
    """[file, mtime_ns, size] of a file (a packaged one: its package's); -1, -1 when missing (a
    would-be module or resource: one appearing later is a change)."""
    if file not in self.taken:
      try:
        st = Path(files.container(file)).stat()
        self.taken[file] = [file, st.st_mtime_ns, st.st_size]
      except OSError:
        self.taken[file] = [file, -1, -1]
    return self.taken[file]


def _ref(ref: TextureRef, index: dict[TextureId, int]) -> dict[str, Json]:
  """A TextureRef as JSON, its texture as an op index."""
  return {"op": index[ref.texture], "channel": ref.channel, "scale": list(ref.scale), "bias": list(ref.bias),
          "uv_xform": [list(r) for r in ref.uv_xform], "uv_set": ref.uv_set}  # fmt: skip


def _unref(d: Json, ids: list[TextureId]) -> TextureRef:
  """`_ref`'s inverse."""
  assert isinstance(d, dict)
  (s0, s1, s2, s3), (b0, b1, b2, b3) = _floats(d["scale"]), _floats(d["bias"])
  (a, b, c), (e, f, g) = _floats(_at(d["uv_xform"], 0)), _floats(_at(d["uv_xform"], 1))
  return TextureRef(ids[int(str(d["op"]))], channel=int(str(d["channel"])), scale=(s0, s1, s2, s3),
                    bias=(b0, b1, b2, b3), uv_xform=((a, b, c), (e, f, g)), uv_set=int(str(d["uv_set"])))  # fmt: skip


def _material(mat: Material, index: dict[TextureId, int]) -> dict[str, object]:
  """A Material as JSON-serialisable values (enums as their values, textures as op indices)."""
  out: dict[str, object] = {**dataclasses.asdict(mat)}
  out["textures"] = [[int(s), _ref(r, index)] for s, r in mat.textures]
  extras = out["extras"]
  if mat.extras is not None and isinstance(extras, dict):
    extras["textures"] = [[n, _ref(r, index)] for n, r in mat.extras.textures]
  return out


def _unmaterial(d: Json, ids: list[TextureId]) -> Material:
  """`_material`'s inverse."""
  d = _dict(d)
  plain = {k: _tuples(v) for k, v in d.items() if k not in ("textures", "ext", "extras")}
  ext: dict[str, object] = {k: _tuples(v) for k, v in _dict(d["ext"]).items()}
  ext["alpha_mode"] = AlphaMode(int(str(ext["alpha_mode"])))
  extras = None
  if d.get("extras") is not None:
    e = _dict(d["extras"])
    refs = tuple((str(_at(t, 0)), _unref(_at(t, 1), ids)) for t in _list(e["textures"]))
    extras = construct_named(OpenPbrExtras, {k: _tuples(v) for k, v in e.items() if k != "textures"}, textures=refs)
  textures = tuple((Slot(int(str(_at(t, 0)))), _unref(_at(t, 1), ids)) for t in _list(d["textures"]))
  return construct_named(Material, plain, textures=textures, extras=extras, ext=construct_named(RaywarpExt, ext))


def _tuples(v: Json) -> object:
  """Lists back as tuples (recursively)."""
  return tuple(_tuples(x) for x in v) if isinstance(v, list) else v


def _floats(v: Json) -> tuple[float, ...]:
  """A JSON list of numbers as floats."""
  return tuple(float(str(x)) for x in _list(v))


def _list(v: Json) -> list[Json]:
  """`v` as a list ([] otherwise)."""
  return v if isinstance(v, list) else []


def _dict(v: Json) -> dict[str, Json]:
  """`v` as a dict ({} otherwise)."""
  return v if isinstance(v, dict) else {}


def _at(v: Json, i: int) -> Json:
  """Element `i` of a JSON list."""
  return _list(v)[i]
