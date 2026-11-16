"""Varying distilled expressions baked to textures by the SDK's CPU baker, cached on disk.

The domain follows the expression's lookups. When they read one axis-aligned map of one UV set
(or of the world position: the core's world projection), and integer tilings of it, the bake
covers the window that map sends to [0, 1]^2 and is read through it: tiled textures keep their
texel density, exactly. An expression that reads world lookups of incommensurate tilings
(a road's world-projected detail under coarser stain noises) is baked over one period of the
coarsest world map that the finest one tiles an integer number of times while keeping at least
`MIN_FINE_TEXELS` texels per period, and read through the world projection: the finest detail at
its density, the other lookups repeating with that window (reported). Otherwise it covers the
mesh's UV square (reported unless the map is the identity; a position read there is (u, v, 0)).
Its side is the largest texture read times its tiling (`PROCEDURAL_SIZE` without
one), at most `PROCEDURAL_SIZE` (the 512-1024 px training target) and the texture size limit.
An expression with nothing varying below it (no varying state, no valid texture's lookup) is
a constant, baked as one texel. Bakes are cached in `<cache>/mdl/<sha256>.npy` by the SDK
version, the expression's structure (full signatures), the domain, the kind and the stats of the
texture files below it and of the material's module closure (the bodies of its non-inlined
functions), and looked up before the SDK compiles a baker (the costly step).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, final

import numpy as np

from raywarp.cache.paths import write_atomic
from raywarp.formats.usd.materials.context import ShadeContext, StageFrame, UvRows
from raywarp.formats.usd.materials.mdl import keys, sdk
from raywarp.formats.usd.materials.mdl.dag import Dag
from raywarp.formats.usd.materials.mdl.keys import Lookups
from raywarp.formats.usd.materials.mdl.values import WORLD_SET, UnsupportedError, UvMap
from raywarp.scene.images import colorspace, files
from raywarp.scene.images import decode as imaging
from raywarp.scene.model.types import UV_SET_WORLD, ImageId

if TYPE_CHECKING:
  from concurrent.futures import Executor

  from raywarp.formats.usd.materials.mdl import sdk_types as st
  from raywarp.scene.build.builder import SceneBuilder

type Texels = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
# Texels as stored: 8-bit, float16 / float32 (emission), or one float64 texel (a constant).
type Stored = np.ndarray[tuple[int, int, int], np.dtype[np.generic]]
# What a bake holds: colors (sRGB), scalars, roughness (the perceptual sqrt of the GGX alpha the
# distiller writes), normals ([0, 1]-packed tangent-space vectors) or emission (float).
type Kind = Literal["color", "scalar", "roughness", "normal", "emission"]
# Bumped when the bake's arithmetic or encoding changes (part of the disk-cache key).
BAKE_VERSION: Final = 1
# Side of a bake that reads the texture coordinates (a procedural) or tiles a lookup [texels]:
# the top of the 512-1024 px range policies train at, capped by the bake size limit.
PROCEDURAL_SIZE: Final = 1024
# The probe of a mean [texels per side: prime, so that no integer tiling of a texture aliases
# to one phase]; the 16-bit sRGB table; the key's version.
_PROBE: Final = 13
# Texels per period the finest world lookup keeps at least in a world-window bake: a quarter of
# the 1024 training target, the density of a 256 px texture tiled at its own period.
MIN_FINE_TEXELS: Final = 256
_KEY_VERSION: Final = "sdk3"
_SRGB_STEPS: Final = 65536
_EPS: Final = 1e-6
_IDENTITY_ROWS: Final = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def encoding(kind: Kind) -> imaging.Encoding:
  """How a kind is stored: colors as sRGB, emission as float, the others linear."""
  return "srgb" if kind == "color" else "float" if kind == "emission" else "linear"


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Domain:
  """Where a bake is evaluated: `window` (min_u, max_u, min_v, max_v) of the SDK's texture
  space (UVs, or the stage's world x, y), read through TextureRef `uv_set` / `rows`, `size`
  texels per side; `exact` False for a UV-square bake of lookups it cannot fold; `position` for
  such a bake of an expression that reads the position (a world lookup it cannot fold, a 3D
  noise, a triplanar projection): the SDK bakes the UV square with the position (u, v, 0)."""

  window: tuple[float, float, float, float]
  uv_set: int
  rows: UvRows
  size: int
  exact: bool
  position: bool = False


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Baked:
  """A bake: its texels and domain (`const`: the value of a constant, no texels; `binary`: every
  texel 0 or 1, a cut-out; `peak`: an emission's largest texel, divided out of the texels;
  `key`: its cache key; `file`: its disk cache file)."""

  domain: Domain
  key: str
  texels: Stored | None = None
  const: tuple[float, ...] | None = None
  binary: bool = False
  peak: float = 1.0
  file: Path | None = None
  gamuts: tuple[str, ...] = ()  # the non-Rec.709 primaries of the textures it reads (kept as is)


@final
@dataclasses.dataclass
class Bakes:
  """A stage read's bakes: the disk cache directory (None: off), the bakes made and the ones
  that failed, by key; the builder images of the bakes bound (an image is added when a slot
  binds it: a prefetched bake no slot binds adds none); the jobs planned (their keys walk the
  SDK expression: computed once)."""

  directory: Path | None
  made: dict[str, Baked] = dataclasses.field(default_factory=dict)
  failed: set[str] = dataclasses.field(default_factory=set)
  images: dict[str, ImageId] = dataclasses.field(default_factory=dict)
  jobs: dict[tuple[Dag, str, str, bool], _Job] = dataclasses.field(default_factory=dict)
  # The bakes whose SDK texels held NaN / inf (stored as 0; the field reports them).
  nonfinite: set[str] = dataclasses.field(default_factory=set)

  def forget(self, dag: Dag) -> None:
    """Drops the jobs of a mapped material's reader (they keep it alive)."""
    for k in [k for k in self.jobs if k[0] is dag]:
      del self.jobs[k]

  def image(self, builder: SceneBuilder, baked: Baked, kind: Kind) -> ImageId | None:
    """The builder image of a bake (added once; None when the builder rejects its texels)."""
    if baked.key not in self.images and baked.texels is not None:
      try:
        self.images[baked.key] = builder.add_image(baked.texels, transfer="srgb" if encoding(kind) == "srgb" else "linear")
      except (ValueError, TypeError):  # non-finite texels
        return None
      self.made[baked.key] = dataclasses.replace(baked, texels=None)  # the builder holds them now
    return self.images.get(baked.key)


def rows_of(uv: UvMap, frame: StageFrame) -> tuple[int, UvRows] | None:
  """(TextureRef uv_set, uv_xform) reading map `uv` (mesh sets above 1: set 0). A world map is
  composed with the stage frame (the core projects scene coordinates; None when the stage's x, y
  are not the scene's horizontal plane)."""
  if uv.uv_set != WORLD_SET:
    return (uv.uv_set if uv.uv_set <= 1 else 0), uv.xform()
  m = frame.stage_from_scene
  if abs(m[0, 2]) > _EPS or abs(m[1, 2]) > _EPS:
    return None
  a, b, c, d, e, f = uv.rows
  r = np.array([[a, b], [d, e]]) @ m[:2, :2]
  return UV_SET_WORLD, ((float(r[0, 0]), float(r[0, 1]), c), (float(r[1, 0]), float(r[1, 1]), f))


def domain(lookups: Lookups, frame: StageFrame, limit: int | None) -> Domain:
  """The domain of a bake of an expression with these `Dag.lookups`."""
  cap = min(limit or PROCEDURAL_SIZE, PROCEDURAL_SIZE)
  maps, side = lookups.maps or frozenset(), max(lookups.sizes, default=0)
  tiled = None if lookups.uv_read or lookups.maps is None else _tiling(maps)
  read = rows_of(tiled[0], frame) if tiled is not None else None
  if tiled is not None and read is not None:
    return Domain(_window(tiled[0]), read[0], read[1], min(side * tiled[1] or PROCEDURAL_SIZE, cap), exact=True)
  cover = None if lookups.maps is None else _world_cover(maps, cap)
  read = rows_of(cover[0], frame) if cover is not None else None
  if cover is not None and read is not None:
    return Domain(_window(cover[0]), read[0], read[1], min(side * cover[1] or PROCEDURAL_SIZE, cap), exact=False)
  tiling = max((abs(x) for uv in maps for x in (uv.rows[0], uv.rows[4])), default=1.0)
  size = min(max(PROCEDURAL_SIZE, side * math.ceil(max(tiling, 1.0))), cap)
  identity = all(uv.uv_set >= 0 and uv.rows == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0) for uv in maps)
  position = lookups.position or any(uv.uv_set == WORLD_SET for uv in maps)
  exact = lookups.maps is not None and identity and (bool(maps) or not lookups.uv_read) and not position
  uv_set = min((uv.uv_set for uv in maps if uv.uv_set >= 0), default=0)
  return Domain((0.0, 1.0, 0.0, 1.0), uv_set, _IDENTITY_ROWS, size, exact, position=position)


def _window(uv: UvMap) -> tuple[float, float, float, float]:
  """The SDK texture-space window that axis-aligned map `uv` sends to [0, 1]^2."""
  a, _, c, _, e, f = uv.rows
  return (-c / a, (1.0 - c) / a, -f / e, (1.0 - f) / e)


def _world_cover(maps: frozenset[UvMap], cap: int) -> tuple[UvMap, int] | None:
  """(the coarsest world map that the finest world map tiles an integer number of times, at most
  cap / MIN_FINE_TEXELS, and that tiling) for maps whose world lookups are not all integer tilings
  of one (None: no such lookups, or one not axis-aligned)."""
  world = [uv for uv in maps if uv.uv_set == WORLD_SET]
  if not world or _tiling(frozenset(world)) is not None:
    return None
  if any(abs(uv.rows[1]) > _EPS or abs(uv.rows[3]) > _EPS or abs(uv.rows[0] * uv.rows[4]) < _EPS for uv in world):
    return None
  finest = max(world, key=lambda uv: abs(uv.rows[0] * uv.rows[4]))
  most = max(1, cap // MIN_FINE_TEXELS)
  covers = [t for base in world if (t := _tiling(frozenset({base, finest}))) is not None and t[1] <= most]
  return max(covers, key=lambda t: t[1], default=None)


def _tiling(maps: frozenset[UvMap]) -> tuple[UvMap, int] | None:
  """(the coarsest map, the largest tiling of the others over it) when every map is
  axis-aligned, of one UV set, and an integer tiling of it."""
  if not maps or len({uv.uv_set for uv in maps}) != 1:
    return None
  if any(abs(uv.rows[1]) > _EPS or abs(uv.rows[3]) > _EPS or abs(uv.rows[0] * uv.rows[4]) < _EPS for uv in maps):
    return None
  base = min(maps, key=lambda uv: abs(uv.rows[0] * uv.rows[4]))
  factor = 1
  for uv in maps:
    for i, j in ((0, 2), (4, 5)):  # (scale, offset) of u, then v
      k = uv.rows[i] / base.rows[i]
      if abs(k - round(k)) > _EPS or abs(uv.rows[j] - k * base.rows[j] - round(uv.rows[j] - k * base.rows[j])) > _EPS:
        return None
      factor = max(factor, abs(round(k)))
  return base, factor


def stored(kind: Kind, x: Texels) -> Texels:
  """Baked linear values (h, w, c) as a slot stores them: 1 channel for scalars, the perceptual
  sqrt of a GGX alpha, unit normals packed to [0, 1], RGB else."""
  if kind == "normal":
    n = x[..., :3]
    length = np.sqrt(np.maximum(np.einsum("...i,...i->...", n, n), 1e-24))[..., None]
    return (0.5 * n / length + 0.5).astype(np.float32)
  if kind == "roughness":
    return np.sqrt(np.maximum(x[..., :1], 0.0)).astype(np.float32)
  return x[..., :1] if kind == "scalar" else x[..., :3]


def _bake(api: sdk.Api, baker: st.IBaker, window: tuple[float, float, float, float], size: int) -> Texels | None:
  """`baker` over `window` at size x size texels, row 0 at the top (None when it fails)."""
  canvas = api.images.create_canvas(baker.get_pixel_type(), size, size)
  if baker.bake_texture(canvas, *window, 0.0, 1) != 0:
    return None
  tile = canvas.get_tile()
  # The SDK view borrows tile storage. Even a flipped 1x1 view is contiguous, so
  # ascontiguousarray can leave it borrowed after the local SDK handles are released.
  view = tile.get_data_numpy().reshape(size, size, -1)[::-1]
  return np.array(view, dtype=np.float32, order="C", copy=True)


def _finite(texels: Texels) -> tuple[Texels, bool]:
  """`texels` with NaN / inf as 0 (a 0 / 0 in the material's expressions), and whether it had
  any."""
  bad = ~np.isfinite(texels)
  if not bad.any():
    return texels, False
  texels[bad] = 0.0
  return texels, True


def bake(ctx: ShadeContext, dag: Dag, path: str, *, kind: Kind, bakes: Bakes, mean: bool = False) -> Baked | None:
  """The expression at `path` baked (None when the SDK cannot bake it), from `bakes` when a
  prefetch or an earlier field made (or failed) it; `mean`: only the probe's mean (a field
  without a texture slot)."""
  job = _job(ctx, dag, path, kind, bakes, mean=mean)
  if job.key in bakes.made or job.key in bakes.failed:
    return bakes.made.get(job.key)
  _store(bakes, job, _texels(dag.api, dag.material, job, bakes.nonfinite))
  return bakes.made.get(job.key)


# A bake request: the expression's path, its kind, whether only its mean is needed.
type Request = tuple[str, Kind, bool]


def prefetch(ctx: ShadeContext, plans: list[tuple[Dag, list[Request]]], bakes: Bakes, pool: Executor) -> None:
  """Bakes the requests of materials on `pool` (the SDK compiles bakers and bakes with the GIL
  released) into `bakes`, for `bake` to find: one task per transaction lane, its jobs in turn (a
  transaction is not thread-safe); every task done on return. A job that fails is a failed bake,
  reported by the field that asks for it."""
  lanes: dict[int, dict[str, tuple[Dag, _Job]]] = {}
  seen: set[str] = set()  # a key once, in the first lane that asks for it
  for dag, requests in plans:
    for path, kind, mean in requests:
      try:
        j = _job(ctx, dag, path, kind, bakes, mean=mean)
      except UnsupportedError:  # the field that asks for it bakes (and reports) it
        continue
      if j.key not in bakes.made and j.key not in bakes.failed and j.key not in seen:
        seen.add(j.key)
        lanes.setdefault(dag.lane, {})[j.key] = (dag, j)
  tasks = [pool.submit(_lane, list(jobs.values()), bakes.nonfinite) for jobs in lanes.values()]
  for task in tasks:
    for j, data in task.result():  # a job the SDK cannot bake is None (reported by its field)
      _store(bakes, j, data)


def _lane(jobs: list[tuple[Dag, _Job]], nonfinite: set[str]) -> list[tuple[_Job, Stored | None]]:
  """The texels of one lane's jobs, in turn (None for a job that fails; `nonfinite` gets the keys
  of those with NaN / inf texels)."""
  return [(j, _texels(dag.api, dag.material, j, nonfinite)) for dag, j in jobs]


def _store(bakes: Bakes, job: _Job, data: Stored | None) -> None:
  """Records a job's bake in `bakes` (failed: None texels, or texels `_finish` rejects)."""
  baked = None
  if data is not None:
    try:
      baked = _finish(job, data)
    except (ValueError, TypeError):
      baked = None
  if baked is None:
    bakes.failed.add(job.key)
  else:
    bakes.made[job.key] = baked


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _Job:
  """A bake to make: the expression, its domain and kind, its cache key and file (None: off)."""

  path: str
  domain: Domain
  kind: Kind
  mean: bool
  varies: bool
  key: str
  cached: Path | None
  gamuts: tuple[str, ...]


def _job(ctx: ShadeContext, dag: Dag, path: str, kind: Kind, bakes: Bakes, *, mean: bool) -> _Job:  # noqa: PLR0913  # a bake's inputs, keyword `mean`
  """The job of a bake (its key read on the caller's thread: SDK expression walks), once."""
  memo = (dag, path, kind, mean)
  if memo not in bakes.jobs:
    lookups = keys.lookups(dag, path)
    dom = domain(lookups, ctx.frame, ctx.max_texture_size)
    gamuts = tuple(sorted(sp.value for sp in lookups.spaces if not colorspace.rec709_primaries(sp)))
    key = _key(dag, path, dom, kind, mean=mean)
    cached = None if bakes.directory is None or key is None else bakes.directory / "mdl" / f"{key}.npy"
    # An expression too large to key is baked once per read under a key of its own.
    unkeyed = f"unkeyed:{len(bakes.jobs)}"
    bakes.jobs[memo] = _Job(path, dom, kind, mean, dag.varies(path), key or unkeyed, cached, gamuts)
  return bakes.jobs[memo]


def _texels(api: sdk.Api, material: st.ICompiled_material, job: _Job, nonfinite: set[str]) -> Stored | None:
  """The stored texels of a bake (the disk cache's when it reads), one float64 texel for a
  constant (an expression with nothing varying below it) or a mean job (its probe's mean); None
  when the SDK cannot bake it. Thread-safe within a lane: SDK calls of one transaction, numpy."""
  if job.cached is not None and job.cached.is_file():
    try:
      data = np.load(job.cached)
    except (OSError, ValueError, EOFError):  # an empty, truncated or corrupt cache file is a miss
      job.cached.unlink(missing_ok=True)
    else:
      if _nonfinite_marker(job.cached).is_file():  # the first bake's NaN / inf texels (read as 0)
        nonfinite.add(job.key)
      return data
  baker = api.distiller.create_baker(material, job.path, api.bake_on_cpu)
  if baker is None or not sdk.valid(baker):
    return None
  side = _PROBE if job.mean and job.varies else 1 if not job.varies else job.domain.size
  texels = _bake(api, baker, job.domain.window, side)
  if texels is None:
    return None
  texels, bad = _finite(texels)
  if bad:
    nonfinite.add(job.key)  # a set add: atomic across lanes
  if job.cached is not None:
    _mark_nonfinite(job.cached, bad=bad)
  values = stored(job.kind, texels)
  if job.mean or not job.varies:
    data: Stored = np.zeros((1, 1, values.shape[2]), np.float64)
    data[0, 0] = values.reshape(-1, values.shape[2]).mean(axis=0)
  else:
    data = _encode(values, job.kind)
  if job.cached is not None:
    save(job.cached, data)
  return data


def _finish(job: _Job, data: Stored) -> Baked:
  """The bake of a job's texels: a constant (one float64 texel), else texels (an emission's
  divided by their peak)."""
  if data.dtype == np.float64:
    return Baked(domain=job.domain, key=job.key, const=tuple(float(x) for x in data.reshape(-1)), gamuts=job.gamuts)
  binary = job.kind == "scalar" and data.dtype == np.uint8 and not bool(((data > 0) & (data < 255)).any())
  peak = float(data.max()) if job.kind == "emission" else 1.0
  if peak > 0.0 and peak != 1.0:
    data = np.asarray(data / peak, np.float32)
  peak = peak if peak > 0.0 else 1.0
  return Baked(domain=job.domain, key=job.key, texels=data, binary=binary, peak=peak, file=job.cached, gamuts=job.gamuts)


def _encode(values: Texels, kind: Kind) -> Stored:
  """Stored texels: colors as 8-bit sRGB through a 16-bit table (the hot path), the others by
  `images.encode`."""
  if encoding(kind) != "srgb":
    return imaging.encode(values, encoding(kind), values.shape[2])
  index = np.rint(np.clip(values, 0.0, 1.0) * float(_SRGB_STEPS - 1)).astype(np.int32)
  return np.asarray(_srgb_table()[index], np.uint8).reshape(values.shape)


@functools.cache
def _srgb_table() -> np.ndarray[[int], np.dtype[np.uint8]]:
  """8-bit sRGB codes of `_SRGB_STEPS` linear values in [0, 1]."""
  return imaging.encode(np.linspace(0.0, 1.0, _SRGB_STEPS, dtype=np.float32)[None, :, None], "srgb", 1).reshape(-1)


def _key(dag: Dag, path: str, dom: Domain, kind: Kind, *, mean: bool) -> str | None:
  """The cache key of a bake: the expression's structure (constants, compounds and resources
  included), the domain, the kind and the stats of every texture file below it, read or not
  (None: an expression too large to key, a texture file that is missing)."""
  structure = keys.key(dag, path)
  stats = []
  for file in sorted(keys.files(dag, path) | dag.modules):
    try:
      st_ = Path(files.container(file)).stat()  # a packaged texture: its package's
    except OSError:
      return None
    stats.append([file, st_.st_mtime_ns, st_.st_size])
  text = json.dumps([BAKE_VERSION, _KEY_VERSION, sdk.version(), structure, dom.window, dom.size, kind, mean, stats])
  return hashlib.sha256(text.encode()).hexdigest() if structure else None


def _nonfinite_marker(cached: Path) -> Path:
  """The empty file next to a cached bake that records its texels were not finite."""
  return cached.with_suffix(".nonfinite")


def _mark_nonfinite(cached: Path, *, bad: bool) -> None:
  """Records (or clears) a bake's non-finite flag next to its cache file, before the file itself
  is written (a reader of the file sees its flag)."""
  marker = _nonfinite_marker(cached)
  try:
    if bad:
      marker.parent.mkdir(parents=True, exist_ok=True)
      marker.touch()
    else:
      marker.unlink(missing_ok=True)
  except OSError as error:
    warnings.warn(f"raywarp: bake cache flag not written ({marker}): {error}", RuntimeWarning, stacklevel=2)


def save(path: Path, data: np.ndarray[[int, int, int], np.dtype[np.generic]]) -> None:
  """Writes bake texels `data` to `path` (`write_atomic`)."""
  write_atomic(path, lambda f: np.save(f, data), "bake cache")
