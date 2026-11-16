"""Host side of the GI feature: the probe volume (`ProbeVolume`), the grid of light probes whose
maps give indirect light to the shading, placed over the scene and updated on the device (its bake
/ update policy is `schedule.Schedule`).

`ProbeVolume` owns the maps (one set of irradiance maps per group of worlds, `groups.Groups`;
the placement and the distance maps are shared), the ray buffers and the update state, all
persistent: an update allocates nothing and reads nothing back. The ray rotation of update k is
entry k of a table drawn once from a seeded generator (`ROTATIONS` entries, then it repeats),
indexed on the device by the update counter, so a CUDA graph's replays keep rotating the rays.
Its probe launches run in bands over the CPU's cores (`device.launch_bands`).

Placement (`place`, static geometry only, once): relocation and classification of the probes,
the ambient cube's walls (`kernels.walls_kernel`) and contact occlusion (`kernels.ao_kernel`),
and the static bake (`bake_static`):
STATIC_RAYS geometry-only rays per probe give the distance moments (then frozen: the updates
trace the same static geometry). The updates' own rays (gi.rays) only carry radiance.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, final

import numpy as np
import warp as wp

from raywarp._base.types import ShapeType
from raywarp.core.kernels.abi import rays as trace
from raywarp.core.pipeline.platform.device import host_copy, launch_bands
from raywarp.core.shade.gi import kernels
from raywarp.core.shade.gi.atlas import Atlas
from raywarp.core.shade.gi.cube import FACES, MAX_TEXELS, SUBDIV, CubeTexture
from raywarp.core.shade.gi.groups import Groups
from raywarp.core.shade.gi.probes import DIST_TEXELS, IRR_TEXELS, Volume, attach, no_relight
from raywarp.core.shade.lights.punctual import Lights
from raywarp.core.shade.materials.material import MaterialTables
from raywarp.settings import GI

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp._base.dims import AnyDeviceArray, AnyHostArray, Floats
  from raywarp.core.kernels.abi.structs import EnvironmentArrays
  from raywarp.core.pipeline.frame import Frame
  from raywarp.core.shade.gi.relight import Relight

# Relocation iterations before the classification pass.
RELOCATE_ITERATIONS: Final = 4
# Geometry-only rays per probe of the static bake (distance moments): about ten
# rays in each distance texel's power-50 lobe (~0.12 sr of 4 pi: N / 105 rays), where the
# updates' gi.rays left < 1 and the Chebyshev test became a random step.
STATIC_RAYS: Final = 1024
# Default bounds: instances larger than this along an axis [m] are unbounded (skyboxes, terrain).
UNBOUNDED_EXTENT_M: Final = 1.0e4
# Ray rotations drawn per volume (update k uses entry k % ROTATIONS).
ROTATIONS: Final = 256
# Fewest probes per CPU band of a launch (`device.launch_bands`).
PROBE_BAND: Final = 64
# Fewest ambient-cube texel rows per CPU band of the cube's resample (`ProbeVolume.refresh`).
CUBE_ROW_BAND: Final = 8
# Most ray results held at once (20 B each: 80 MB); the groups of an update are traced in chunks.
RAY_BUDGET: Final = 1 << 22
# Survey points per axis of the free-path survey (`free_path`), and the fewest enclosed ones that
# give the scene an interior scale.
SURVEY_CELLS: Final = 8
MIN_ENCLOSED: Final = 8
# Probe spacings per median instance diagonal at most, in an open scene (`target_spacing`).
OPEN_CELLS: Final = 2.0
# Probe spacings per free distance of the derived spacing (`target_spacing`): 0.5 m in the
# furnished room the fixed 0.5 m spacing was validated on (the diag corner room, 4 x 3 x 2.5 m
# with a table: free distance 0.81 m).
FREE_PATH_CELLS: Final = 1.6
# Smallest extent of the probe box per axis, relative to its largest (a flat scene still gets one
# layer of probes).
_MIN_EXTENT: Final = 1.0e-6
# Growth of the probe spacing per step until the grid fits gi.max_probes.
_SPACING_GROWTH: Final = 1.1
# Contact occlusion reach [cube texels] (`kernels.ao_kernel`): geometry within one texel of a
# texel face shades it (contacts, corners). Two texels over-darkened shelf edges and corners
# (the gi_stress shelves case against its Mitsuba reference at 512 px: blotch 0.48 -> 1.25); one
# keeps the contacts and corners.
CONTACT_AO_TEXELS: Final = 1.0
# Dilation passes of the cube (outside, then inside solids: `cube.Stage`): the crevice under a
# curved object resting on a floor holds no probe in sight for up to about a probe spacing
# (cube.SUBDIV texels) per side.
CONTACT_PASSES: Final = 2 * SUBDIV
# Distance maps clamp hit distances to this many probe-cell diagonals (`Volume.max_distance`).
_MAX_DISTANCE_CELLS: Final = 1.5

# A host point (box corner); a (1,) device gate flag.
type Point = Floats[[3]]
type Gate = wp.array[int, [1]]


# probe_rays(launch, n, lit, env): `n` units of a probe-ray launch on the backend's hardware
# (`GiRays.probe` over the inputs' frame), lit by `lit` in environment `env`.
type ProbeRays = Callable[[kernels.ProbeLaunch, int, Lights, EnvironmentArrays], None]


class PlacementRays(Protocol):
  """The placement's geometry-only rays on a backend's hardware (`GiRays.placement`): `n` units of
  a launch of the software kernels' bodies (`kernels.make_static_ray`, `make_survey_ray`,
  `make_walls_texel`, `make_ao_texel`) over its traversal."""

  def static(self, launch: kernels.StaticLaunch, n: int) -> None:
    """`n` (ray, probe) units of the static bake."""
    ...

  def survey(self, launch: kernels.SurveyLaunch, n: int) -> None:
    """`n` (ray, point) units of a survey."""
    ...

  def walls(self, launch: kernels.WallsLaunch, n: int) -> None:
    """`n` texels of the walls (x fastest)."""
    ...

  def ao(self, launch: kernels.AoLaunch, n: int) -> None:
    """`n` listed texels of the contact occlusion."""
    ...


class GiRays(Protocol):
  """The GI's rays on a backend's hardware (`TraceBackend.gi_tracer`: OptiX's
  `launch.OptixGiRays`), with the probe geometry's colors (`colors`: the moving instances
  hidden; the deformables are never traced, as in the software bake): the probe rays
  (`kernels.make_probe_ray`) with the materials of frame `source`, and the placement's rays."""

  def probe(
    self,
    launch: kernels.ProbeLaunch,
    n: int,
    *,
    source: Frame,
    colors: wp.array[wp.vec4, [int, int]],
    lit: Lights,
    env: EnvironmentArrays,
  ) -> None:
    """`n` (group, ray, probe) units of a probe-ray launch."""
    ...

  def placement(self, colors: wp.array[wp.vec4, [int, int]]) -> PlacementRays:
    """The placement's rays over geometry of these colors."""
    ...


class ProbeInputs(NamedTuple):
  """The frame for the probe rays (each group reads its representative world's rows).

  `lights()` / `environment()` return a new light / environment struct of the frame each call
  (callers may modify them). `worlds`: a relit volume's (`relight.Relight`) frame of the worlds
  themselves, whose tints it computes (these inputs are then the authored ones the shared maps
  are baked with); None: these. `rays`: the probe rays on the backend's hardware (None: the
  software kernel). `relit`: a relit volume's inputs as its relight kernel reads them (its own
  build's material carrier and word, `baked._relit_inputs`); None: not relit.
  """

  geo: trace.SoftwareGeometry
  mats: MaterialTables
  env: EnvironmentArrays
  lights: Callable[[], Lights]
  environment: Callable[[], EnvironmentArrays]
  worlds: ProbeInputs | None = None
  rays: ProbeRays | None = None
  relit: ProbeInputs | None = None


def random_rotations(count: int) -> np.ndarray[[int, 3, 3], np.dtype[np.float32]]:
  """(count, 3, 3) float32 uniformly random rotations (unit quaternions from 4 normals each,
  seeded: every volume draws the same table)."""
  rng = np.random.default_rng(0)
  out = np.empty((count, 3, 3), np.float32)
  for k in range(count):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q, keepdims=True)
    out[k] = np.array(
      [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
      ],
      dtype=np.float32,
    )
  return out


def scene_bounds[N: IntVar](
  lower: np.ndarray[[N, 3], np.dtype[np.float32]],
  upper: np.ndarray[[N, 3], np.dtype[np.float32]],
  types: np.ndarray[[N], np.dtype[np.int32]],
  *,
  spacing: float,
  max_probes: int,
) -> tuple[Point, Point]:
  """Default probe box: the bounds of world 0's instances, or of their largest cluster when
  those need more than `max_probes` probes at `spacing`.

  Planes and unbounded instances (larger than UNBOUNDED_EXTENT_M: skyboxes, terrain) are
  ignored. Instances whose boxes are more than two spacings apart along an axis (they share no
  probe cell) belong to different clusters (`_cluster`): far-away scenery (surrounding
  buildings hundreds of metres out) would otherwise stretch the grid over empty space; every
  instance of the kept cluster is inside (a percentile trim left perimeter walls outside the box,
  where their lookups read texels computed elsewhere). Points outside the box read its boundary
  texels.
  """
  lo_all, hi_all = _bounded(lower, upper, types)
  if not len(lo_all):
    return np.full(3, -1.0), np.full(3, 1.0)
  lo, hi = lo_all.min(0), hi_all.max(0)
  if np.prod(np.maximum(np.ceil((hi - lo) / spacing), 1.0)) > max_probes:
    main = _cluster(lo_all, hi_all, 2.0 * spacing)
    lo, hi = lo_all[main].min(0), hi_all[main].max(0)
  return lo, np.maximum(hi, lo + _MIN_EXTENT * float(np.max(hi - lo)))


def cluster_bounds[N: IntVar](
  lower: np.ndarray[[N, 3], np.dtype[np.float32]],
  upper: np.ndarray[[N, 3], np.dtype[np.float32]],
  types: np.ndarray[[N], np.dtype[np.int32]],
  *,
  gap: float,
) -> tuple[Point, Point]:
  """The bounds of the main cluster (`_cluster`, boxes `gap` apart) of the instances
  `scene_bounds` keeps."""
  lo_all, hi_all = _bounded(lower, upper, types)
  if not len(lo_all):
    return np.full(3, -1.0), np.full(3, 1.0)
  main = _cluster(lo_all, hi_all, gap)
  lo, hi = lo_all[main].min(0), hi_all[main].max(0)
  return lo, np.maximum(hi, lo + _MIN_EXTENT * float(np.max(hi - lo)))


def _bounded[N: IntVar](
  lower: np.ndarray[[N, 3], np.dtype[np.float32]],
  upper: np.ndarray[[N, 3], np.dtype[np.float32]],
  types: np.ndarray[[N], np.dtype[np.int32]],
) -> tuple[np.ndarray[[int, 3], np.dtype[np.float32]], np.ndarray[[int, 3], np.dtype[np.float32]]]:
  """(lower, upper) of the instances that bound the probes: planes, ground planes authored as
  meshes (`_ground_meshes`) and unbounded instances (larger than UNBOUNDED_EXTENT_M) left out,
  unless nothing else is left."""
  ok = np.all(np.isfinite(lower), axis=1) & np.all(upper - lower < UNBOUNDED_EXTENT_M, axis=1)
  keep = ok & (types != int(ShapeType.PLANE))
  keep &= ~_ground_meshes(lower, upper, keep)
  if not np.any(keep):
    keep = ok
  return lower[keep], upper[keep]


def _ground_meshes[N: IntVar](
  lower: np.ndarray[[N, 3], np.dtype[np.float32]],
  upper: np.ndarray[[N, 3], np.dtype[np.float32]],
  keep: np.ndarray[[N], np.dtype[np.bool_]],
) -> np.ndarray[[N], np.dtype[np.bool_]]:
  """(N,) the kept instances that are planes in all but name: flat (no thickness) and reaching
  past the bounds of every other kept instance (a 222 m floor quad under a classroom, a ground
  quad under a robot); the walls and floors of a room lie within the bounds of the others."""
  ext = upper - lower
  flat = keep & (ext.min(1) <= _MIN_EXTENT * ext.max(1))
  if not np.any(flat) or keep.sum() < 2:
    return np.zeros(len(lower), bool)
  # The others' bounds per instance: the kept extremes, or the second ones for the instance that
  # holds an extreme.
  lo_k = np.where(keep[:, None], lower, np.inf)
  hi_k = np.where(keep[:, None], upper, -np.inf)
  lo_sorted, hi_sorted = np.sort(lo_k, axis=0), np.sort(hi_k, axis=0)
  lo_other = np.where(lo_k == lo_sorted[0], lo_sorted[1], lo_sorted[0])
  hi_other = np.where(hi_k == hi_sorted[-1], hi_sorted[-2], hi_sorted[-1])
  beyond = np.any((lower < lo_other) | (upper > hi_other), axis=1)
  return flat & beyond


def _cluster[N: IntVar](
  lower: np.ndarray[[N, 3], np.dtype[np.float32]], upper: np.ndarray[[N, 3], np.dtype[np.float32]], gap: float
) -> np.ndarray[[N], np.dtype[np.bool_]]:
  """(N,) the boxes of the cluster with the most boxes: per axis, the sorted intervals split
  where one starts more than `gap` past the end of all before it, the largest part kept, until
  no axis splits it."""
  keep = np.ones(len(lower), bool)
  changed = True
  while changed:
    changed = False
    for a in range(3):
      idx = np.flatnonzero(keep)
      order = idx[np.argsort(lower[idx, a], kind="stable")]
      ends = np.maximum.accumulate(upper[order, a])
      cuts = np.flatnonzero(lower[order[1:], a] > ends[:-1] + gap) + 1
      if len(cuts):
        part = max(np.split(order, cuts.tolist()), key=len)
        keep[:] = False
        keep[part] = True
        changed = True
  return keep


def free_path[NInstance: IntVar](
  geo: trace.SoftwareGeometry,
  see_through: wp.array[wp.int32, [NInstance]],
  lo: Point,
  hi: Point,
  *,
  device: wp.Device,
  shape_types: tuple[int, ...],
  trees: trace.Trees,
  alpha_cutout: bool,
  live_uv: bool = False,
  world_uv: bool = False,
  rays: PlacementRays | None = None,
) -> float | None:
  """The scene's free distance [m] (`rays`: on the backend's hardware): over the enclosed points
  (`kernels.survey_kernel`: front faces all around, the inside of rooms) of SURVEY_CELLS^3 cell
  centres of [lo, hi], the median of each point's harmonic mean distance to the static geometry
  along `kernels.RELOCATE_RAYS` fixed directions; None with fewer than MIN_ENCLOSED enclosed
  points (an open scene: no interior scale). Scale-free (a scene scaled by s has s times its free
  distance) and set by the nearest geometry (a harmonic mean), not by how the scene is split into
  instances."""
  kernel = kernels.survey_kernel(shape_types, trees=trees, alpha_cutout=alpha_cutout, live_uv=live_uv, world_uv=world_uv)
  n = SURVEY_CELLS
  step = (np.asarray(hi, np.float64) - np.asarray(lo, np.float64)) / n
  k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
  centres = np.asarray(lo, np.float64) + (np.stack([i, j, k], -1).reshape(-1, 3) + 0.5) * step
  shape = (n**3, kernels.RELOCATE_RAYS)
  q = kernels.SurveyLaunch()
  q.see_through, q.first, q.npoint = see_through, 0, n**3
  q.origins = wp.array(centres.astype(np.float32), dtype=wp.vec3, device=device)
  q.moving = wp.ones(n**3, dtype=wp.int32, device=device)
  q.kinds = kinds = wp.empty(shape, dtype=wp.int32, device=device)
  q.dists = dists = wp.empty(shape, dtype=float, device=device)
  if rays is not None:
    rays.survey(q, kernels.RELOCATE_RAYS * n**3)
  else:
    wp.launch(kernel, dim=(kernels.RELOCATE_RAYS, n**3), inputs=[geo, q], device=device)
  kind, dist = host_copy(kinds), host_copy(dists)
  front = kind == kernels.SURVEY_FRONT
  fronts = front.sum(1)
  # Enclosed: front faces in at least ENCLOSED_FRACTION of the directions and not inside geometry
  # (back faces above BACKFACE_FRACTION): a room's interior, not the open air around the scenery.
  enclosed = ((kind == kernels.SURVEY_BACK).sum(1) <= kernels.BACKFACE_FRACTION * kernels.RELOCATE_RAYS) & (
    fronts >= kernels.ENCLOSED_FRACTION * kernels.RELOCATE_RAYS
  )
  if int(enclosed.sum()) < MIN_ENCLOSED:
    return None
  inverse = np.where(front, 1.0 / np.maximum(dist, 1.0e-30), 0.0).sum(1)[enclosed] / fronts[enclosed]
  return float(np.median(1.0 / inverse))


def target_spacing(free: float | None, lo: Point, hi: Point, *, max_probes: int, diagonal: float) -> float:
  """The derived probe spacing [m] of a scene of free distance `free` (`free_path`) in [lo, hi]:
  free / FREE_PATH_CELLS. An open scene (None: no interior scale) gets the spacing that fills
  `max_probes` over the box, but no finer than its median instance `diagonal` / OPEN_CELLS (a
  robot over a ground plane: a few probes per object, not the whole budget around it)."""
  if free is None:
    volume = float(np.prod(np.maximum(np.asarray(hi) - np.asarray(lo), 1.0e-30)))
    return max((volume / max_probes) ** (1.0 / 3.0), diagonal / OPEN_CELLS)
  return free / FREE_PATH_CELLS


def grid_counts(extent: Point, *, spacing: float, max_probes: int) -> tuple[tuple[int, int, int], Point]:
  """(probes per axis, probe spacing per axis [m]) of a box of size `extent` [m]: cells of about
  `spacing`, grown by _SPACING_GROWTH until at most `max_probes` (>= 1) probes."""
  while True:
    counts = np.maximum(np.ceil(extent / spacing).astype(int), 1)
    if int(np.prod(counts)) <= max_probes:
      break
    spacing *= _SPACING_GROWTH
  nx, ny, nz = (int(c) for c in counts)
  return (nx, ny, nz), extent / counts


@final
class ProbeVolume:
  """Probe grid, maps, ray buffers and update state of a renderer (host side of the GI feature).

  Args:
    lo, hi: box covered by the probes (probes at the centres of cells of about `spacing`).
    spacing: target probe spacing [m] (gi.spacing, or `target_spacing`); it grows until the grid
      has at most gi.max_probes probes.
    cell: largest ambient-cube texel [m] (`cube.subdivision`): the derived spacing / SUBDIV.
    settings: the renderer's GI settings: max_probes, rays (per probe per update), normal_bias
      (lookup offset along the normal, as a fraction of the smallest probe spacing).
    scale: map units per radiance unit (`Volume.scale`): the exposure, so the float16 maps hold
      display-relative values.
    device: where the maps and buffers live and the launches run.
    groups: the world -> group table (`groups.Groups`); None: one group shared by every world.
  """

  def __init__(
    self,
    lo: Point,
    hi: Point,
    *,
    spacing: float,
    cell: float,
    settings: GI,
    scale: float,
    device: wp.Device,
    groups: Groups | None = None,
  ) -> None:
    """Sizes the grid over [lo, hi] and allocates every buffer of the volume."""
    lo = np.asarray(lo, dtype=np.float64)
    self.box = np.stack([lo, np.asarray(hi, dtype=np.float64)])
    extent = self.box[1] - lo
    extent = np.maximum(extent, _MIN_EXTENT * float(np.max(extent)))
    self.target = spacing
    self.cell = cell
    self.counts, step = grid_counts(extent, spacing=spacing, max_probes=settings.max_probes)
    self.nprobe: int = math.prod(self.counts)  # math.prod is typed Literal[1] | int
    self.spacing = step
    self.rays = settings.rays
    self.device = device
    self.groups = groups if groups is not None else Groups(1, digests=None, device=device)
    self.ngroup = self.groups.ngroup
    # Groups traced per launch: their ray results fit RAY_BUDGET rays.
    self.chunk = max(1, min(self.ngroup, RAY_BUDGET // max(self.nprobe * self.rays, 1)))
    self.struct = self._volume(lo, step, normal_bias=settings.normal_bias, scale=scale)
    # Ray results of an update (one chunk of groups).
    self.ray_radiance: wp.array[wp.vec4, [int, int]] | None = None
    self.ray_sky: wp.array[float, [int, int]] | None = None
    # Update state (device): rotation table, update counter, this update's ray directions and
    # lobe weights, the gate of host-driven updates (always open).
    self.rotations = wp.array(random_rotations(ROTATIONS), dtype=wp.mat33, device=device)
    self.updates = wp.zeros(1, dtype=int, device=device)
    self.dirs = wp.zeros(self.rays, dtype=wp.vec3, device=device)
    self.irradiance_lobes = self._lobes(IRR_TEXELS * IRR_TEXELS)
    self.distance_lobes = self._lobes(DIST_TEXELS * DIST_TEXELS)
    self.always = wp.ones(1, dtype=int, device=device)
    # The periodic updates record no ray hits (`kernels.RayHits`, empty; allocated once: an update
    # allocates nothing).
    self.no_hits = self._hits(0)
    # The per-world tint of a relit volume (`relight.Relight`, set by the renderer's bake).
    self.relight: Relight | None = None

  def _lobes(self, ntexel: int) -> kernels.LobeRays:
    """Lobe lists of `ntexel` texels over this volume's rays (`kernels.lobe_rays`)."""
    lobes = kernels.LobeRays()
    lobes.ray = wp.zeros((ntexel, self.rays), dtype=wp.int32, device=self.device)
    lobes.weight = wp.zeros((ntexel, self.rays), dtype=float, device=self.device)
    lobes.count = wp.zeros(ntexel, dtype=wp.int32, device=self.device)
    return lobes

  def _volume(self, lo: Point, step: Point, *, normal_bias: float, scale: float) -> Volume:
    """The `Volume` struct of the grid (probe (0, 0, 0) at the centre of the first cell, `step`
    [m] from `lo`): zeroed maps (one set of irradiance maps per group), no offsets, every probe
    active."""
    dev = self.device
    vol = Volume()
    vol.origin = wp.vec3(*(lo + 0.5 * step))
    vol.spacing = wp.vec3(*step)
    vol.counts = wp.vec3i(*self.counts)
    smallest = min(step.tolist())
    vol.normal_bias = normal_bias * smallest
    vol.max_distance = _MAX_DISTANCE_CELLS * math.hypot(*step.tolist())
    vol.scale = scale
    vol.nprobe = self.nprobe
    vol.world_group = self.groups.world_group
    vol.irradiance = wp.zeros((self.ngroup * self.nprobe, IRR_TEXELS * IRR_TEXELS), dtype=wp.vec4h, device=dev)
    vol.distance = wp.zeros((self.nprobe, DIST_TEXELS * DIST_TEXELS), dtype=wp.vec2h, device=dev)
    vol.offset = wp.zeros(self.nprobe, dtype=wp.vec3, device=dev)
    vol.active = wp.ones(self.nprobe, dtype=wp.int32, device=dev)
    self.irradiance_atlas = Atlas(self.ngroup * self.nprobe, IRR_TEXELS, 4, dev)
    self.distance_atlas = Atlas(self.nprobe, DIST_TEXELS, 2, dev)
    attach(vol, self.irradiance_atlas, self.distance_atlas)
    self.cube = CubeTexture(
      tuple(lo.tolist()),
      tuple(step.tolist()),
      self.counts,
      FACES * self.ngroup,
      dev,
      budget=MAX_TEXELS,
      cell=self.cell,
      passes=(CONTACT_PASSES, CONTACT_PASSES),
    )
    vol.cube = self.cube.struct
    vol.relight = no_relight(dev)
    return vol

  def refresh(self, gate: Gate | None = None) -> None:
    """Copies the maps into the probe lookup's texture atlases, then resamples the groups that
    ran (`gate`: this update's gate, before the groups' ages count it; None: every used group)
    into the ambient cube, dilated (after every blend or restore)."""
    self.irradiance_atlas.refresh(self.struct.irradiance)
    self.distance_atlas.refresh(self.struct.distance)
    g = self.groups
    c = self.cube.struct.counts
    staging = self.cube.staging()
    ao = self.cube.ao
    gate = self.always if gate is None else gate

    def run(a: int, b: int) -> None:
      """The cube's texel rows [a, b) (row k * ny + j; each texel is written once: the bands'
      results are the single launch's)."""
      wp.launch(
        kernels.resample_cube,
        dim=(self.ngroup, b - a, c[0]),
        inputs=[self.struct, gate, g.rep, ao, g.nworld, a],
        outputs=[staging],
        device=self.device,
      )

    launch_bands(wp.get_module(kernels.__name__), device=self.device, n=c[2] * c[1], launch=run, min_band=CUBE_ROW_BAND)
    self.cube.finish((gate, g.rep, g.nworld))

  @property
  def nbytes(self) -> int:
    """Device memory of the volume (maps, offsets, flags, groups) and of the ray buffers."""
    vol = self.struct
    arrays = (vol.irradiance, vol.distance, vol.offset, vol.active, self.ray_radiance, self.ray_sky)
    atlases = self.irradiance_atlas.nbytes + self.distance_atlas.nbytes + self.cube.nbytes
    relight = self.relight.nbytes if self.relight is not None else 0
    return int(sum(a.capacity for a in arrays if a is not None)) + self.groups.nbytes + atlases + relight

  def ray_buffers(self) -> tuple[wp.array[wp.vec4, [int, int]], wp.array[float, [int, int]]]:
    """The per-ray results of a chunk of groups (radiance + hit distance, sky weight): update
    scratch, allocated on first use and freed by `release`."""
    if self.ray_radiance is None or self.ray_sky is None:
      shape = (self.chunk * self.nprobe, self.rays)
      self.ray_radiance = wp.zeros(shape, dtype=wp.vec4, device=self.device)
      self.ray_sky = wp.zeros(shape, dtype=float, device=self.device)
    return self.ray_radiance, self.ray_sky

  def _persistent(self) -> dict[str, AnyDeviceArray]:
    """The placed, baked device state by name: maps, relocation offsets, active flags, update
    counter, the cube's walls."""
    vol = self.struct
    return {
      "irradiance": vol.irradiance,
      "distance": vol.distance,
      "offset": vol.offset,
      "active": vol.active,
      "updates": self.updates,
      "walls": self.cube.wall_array(),
      "ao": self.cube.ao,
    }

  def state(self) -> dict[str, AnyHostArray]:
    """The placed, baked state as host arrays, with the box: the bake cache's entry (`restore`)."""
    if self.cube.walls is None:
      raise RuntimeError("the volume's staging arrays were released (a frozen bake): its state is in the bake cache")
    return {"box": self.box, "grid": np.asarray([self.target, self.cell])} | {
      k: host_copy(a) for k, a in self._persistent().items()
    }

  def restore(self, state: Mapping[str, AnyHostArray]) -> None:
    """Copies a `state` of a volume over the same box into this one (bit-identical)."""
    for name, dst in self._persistent().items():
      dst.assign(state[name])
    self.cube.set_walls()
    self.refresh()

  @property
  def released(self) -> bool:
    """Whether the cube's staging arrays were released (`release`)."""
    return self.cube.walls is None

  def release(self) -> None:
    """Frees the update scratch (ray buffers) and the cube's staging arrays: a frozen bake (no
    update, no tracked group) keeps what the lookups read. An explicit update allocates the
    scratch and staging texels again (and keeps them); periodic updates re-place the volume
    (`baked.Baked.probes`)."""
    self.ray_radiance = None
    self.ray_sky = None
    self.cube.release()

  def bands(self, module: wp.Module, launch: Callable[[int, int], None]) -> None:
    """`launch(a, b)` (kernels of `module`) over the probes [a, b) (`device.launch_bands`)."""
    launch_bands(module, device=self.device, n=self.nprobe, launch=launch, min_band=PROBE_BAND)

  def place[NInstance: IntVar](
    self,
    geo: trace.SoftwareGeometry,
    see_through: wp.array[wp.int32, [NInstance]],
    open_to: wp.array[wp.int32, [NInstance]],
    *,
    trees: trace.Trees,
    shape_types: tuple[int, ...],
    alpha_cutout: bool,
    live_uv: bool = False,
    world_uv: bool = False,
    rays: PlacementRays | None = None,
  ) -> None:
    """Relocation (at most RELOCATE_ITERATIONS moves), then classification (probes still inside
    geometry: inactive) (`_relocate`); the cube's walls (`kernels.walls_kernel`), compact solids
    (`kernels.compact_solids`) and contact occlusion (`_contact`); the static bake
    (`bake_static`).

    `see_through`: 1 for geometry the placement ignores (window glass, moving instances);
    `open_to`: 1 for geometry the contact occlusion ignores (those and emitters); `trees` /
    `shape_types` / `alpha_cutout`: the ray queries' Features; `rays`: the rays on the backend's
    hardware (None: the software kernels).
    """
    q = {"trees": trees, "alpha_cutout": alpha_cutout, "live_uv": live_uv, "world_uv": world_uv}
    self._relocate(geo, see_through, kernels.survey_kernel(shape_types, **q), rays)
    c = self.cube.struct.counts
    wall_array = self.cube.wall_array()
    w = kernels.WallsLaunch()
    w.vol, w.see_through, w.walls = self.struct, see_through, wall_array
    dim = (c[2], c[1], c[0])
    if rays is not None:
      rays.walls(w, c[0] * c[1] * c[2])
    else:
      wp.launch(kernels.walls_kernel(shape_types, **q), dim=dim, inputs=[geo, w], device=self.device)
    wp.launch(kernels.compact_solids, dim=dim, outputs=[wall_array], device=self.device)
    wp.launch(kernels.compact_neighbours, dim=dim, outputs=[wall_array], device=self.device)
    self._contact(geo, open_to, kernels.ao_kernel(shape_types, **q), rays)
    self.cube.set_walls()
    self.bake_static(geo, see_through, kernels.static_kernel(shape_types, **q), rays)

  def _contact[NInstance: IntVar](
    self,
    geo: trace.SoftwareGeometry,
    open_to: wp.array[wp.int32, [NInstance]],
    kernel: wp.Kernel,
    rays: PlacementRays | None,
  ) -> None:
    """The contact occlusion (`kernels.ao_kernel`) of the texels near a wall, listed first
    (`kernels.near_walls`, `kernels.list_cells`: dense threads, the open space skipped)."""
    dev = self.device
    walls = self.cube.wall_array()
    nz, ny, nx = walls.shape[:3]
    flags = wp.empty(nz * ny * nx, dtype=wp.int32, device=dev)
    wp.launch(kernels.near_walls, dim=(nz, ny, nx), inputs=[walls], outputs=[self.cube.ao, flags], device=dev)
    index = wp.empty_like(flags)
    wp.utils.array_scan(flags, index, inclusive=False)
    last = np.concatenate([host_copy(index[-1:]), host_copy(flags[-1:])])
    count = int(last.sum())
    if count == 0:
      return
    cells = wp.empty(count, dtype=wp.vec3i, device=dev)
    wp.launch(kernels.list_cells, dim=nz * ny * nx, inputs=[flags, index, nx, ny], outputs=[cells], device=dev)
    q = kernels.AoLaunch()
    q.cube, q.see_through, q.cells, q.ao = self.cube.struct, open_to, cells, self.cube.ao
    q.reach = CONTACT_AO_TEXELS * float(np.max(1.0 / np.asarray(self.cube.struct.inv_cell, np.float64)))
    if rays is not None:
      rays.ao(q, count)
    else:
      wp.launch(kernel, dim=count, inputs=[geo, q], device=dev)

  def _relocate[NInstance: IntVar](
    self,
    geo: trace.SoftwareGeometry,
    see_through: wp.array[wp.int32, [NInstance]],
    kernel: wp.Kernel,
    rays: PlacementRays | None,
  ) -> None:
    """At most RELOCATE_ITERATIONS moves of every probe, then its classification: per step a
    survey of the probes still moving (`kernel`, from `kernels.survey_kernel`) and their move
    (`kernels.relocate_step`)."""
    dev = self.device
    moving = wp.ones(self.nprobe, dtype=wp.int32, device=dev)
    origins = wp.empty(self.nprobe, dtype=wp.vec3, device=dev)
    wp.launch(kernels.probe_positions, dim=self.nprobe, inputs=[self.struct], outputs=[origins], device=dev)
    shape = (self.nprobe, kernels.RELOCATE_RAYS)
    kinds = wp.empty(shape, dtype=wp.int32, device=dev)
    dists = wp.empty(shape, dtype=float, device=dev)
    for step in range(RELOCATE_ITERATIONS + 1):
      last = int(step == RELOCATE_ITERATIONS)

      def survey(a: int, b: int) -> kernels.SurveyLaunch:
        """The survey launch of probes [a, b)."""
        q = kernels.SurveyLaunch()
        q.see_through, q.origins, q.moving, q.first, q.npoint = see_through, origins, moving, a, b - a
        q.kinds, q.dists = kinds, dists
        return q

      def move(a: int, b: int, last: int = last) -> None:
        """The move of probes [a, b)."""
        wp.launch(
          kernels.relocate_step,
          dim=b - a,
          inputs=[kinds, dists, last, a],
          outputs=[moving, origins, self.struct],
          device=dev,
        )

      def run(a: int, b: int) -> None:
        """The survey and move of probes [a, b) (the software kernel)."""
        wp.launch(kernel, dim=(kernels.RELOCATE_RAYS, b - a), inputs=[geo, survey(a, b)], device=dev)
        move(a, b)

      if rays is not None:
        rays.survey(survey(0, self.nprobe), kernels.RELOCATE_RAYS * self.nprobe)
        move(0, self.nprobe)
      else:
        self.bands(kernel.module, run)

  def bake_static[NInstance: IntVar](
    self,
    geo: trace.SoftwareGeometry,
    see_through: wp.array[wp.int32, [NInstance]],
    kernel: wp.Kernel,
    rays: PlacementRays | None = None,
  ) -> None:
    """STATIC_RAYS geometry-only rays per probe (`kernels.static_kernel`, or `rays` on the
    backend's hardware), in rotated sets of gi.rays, pooled into the distance moments."""
    dev = self.device
    ndist = DIST_TEXELS * DIST_TEXELS
    distance_sum = wp.zeros((self.nprobe, ndist), dtype=wp.vec3, device=dev)
    ray_radiance, _ray_sky = self.ray_buffers()
    length = self.struct.max_distance
    for _ in range(math.ceil(STATIC_RAYS / self.rays)):
      self.directions()

      def launch(a: int, b: int) -> kernels.StaticLaunch:
        """The launch of the rays of probes [a, b)."""
        q = kernels.StaticLaunch()
        vol = self.struct
        q.origin, q.spacing, q.counts, q.offset, q.active = vol.origin, vol.spacing, vol.counts, vol.offset, vol.active
        q.see_through, q.dirs, q.length = see_through, self.dirs, length
        q.first, q.nprobe, q.nray, q.ray_radiance = a, b - a, self.rays, ray_radiance
        return q

      def accumulate(a: int, b: int) -> None:
        """The sums of the rays of probes [a, b)."""
        wp.launch(
          kernels.accumulate_static,
          dim=(b - a, ndist),
          inputs=[self.struct, ray_radiance, self.distance_lobes, a],
          outputs=[distance_sum],
          device=dev,
        )

      def run(a: int, b: int) -> None:
        """The rays of probes [a, b) and their sums (the software kernel)."""
        wp.launch(kernel, dim=(self.rays, b - a), inputs=[geo, launch(a, b)], device=dev)
        accumulate(a, b)

      if rays is not None:
        rays.static(launch(0, self.nprobe), self.rays * self.nprobe)
        accumulate(0, self.nprobe)
      else:
        self.bands(kernel.module, run)
      wp.launch(kernels.advance, dim=1, inputs=[self.always, 1], outputs=[self.updates], device=dev)
    wp.launch(
      kernels.resolve_static,
      dim=(self.nprobe, ndist),
      inputs=[distance_sum],
      outputs=[self.struct],
      device=dev,
    )

  def directions(self, gate: Gate | None = None) -> None:
    """This update's ray directions and lobe lists (`kernels.rotate_rays`, `kernels.lobe_rays`)."""
    gate = self.always if gate is None else gate
    dev = self.device
    wp.launch(
      kernels.rotate_rays,
      dim=self.rays,
      inputs=[self.rotations, self.updates, gate, self.rays],
      outputs=[self.dirs],
      device=dev,
    )
    ntexel = IRR_TEXELS * IRR_TEXELS + DIST_TEXELS * DIST_TEXELS
    wp.launch(
      kernels.lobe_rays,
      dim=ntexel,
      inputs=[self.dirs, gate, self.rays],
      outputs=[self.irradiance_lobes, self.distance_lobes],
      device=dev,
    )

  def update(
    self,
    kernel: wp.Kernel,
    inputs: ProbeInputs,
    lit: Lights,
    *,
    step: UpdateStep,
    gate: Gate | None = None,
    counted: bool = True,
  ) -> None:
    """One update of the groups that run (`groups.group_used`): trace the probe rays (`kernel`,
    from `kernels.make_trace_kernel`) one chunk of groups at a time, blend the maps.

    `inputs` / `lit`: the frame (env.gi must be this volume); `step`: the update's parameters;
    `gate`: a device flag skipping the update when 0 (None: always run); `counted` False: the
    caller ends it (`count`, after a CUDA graph's conditional body: `schedule.Schedule.step`).
    """
    gate = self.always if gate is None else gate
    self.directions(gate)
    for first_group in range(0, self.ngroup, self.chunk):
      n = min(self.chunk, self.ngroup - first_group)
      self._trace_chunk(kernel, inputs, lit, gate, first_group, n, self.no_hits)
      self._blend_chunk(first_group, n, step.hysteresis, gate=gate)
    if counted:
      self.count(gate)

  def bake(self, kernel: wp.Kernel, inputs: ProbeInputs, lit: Lights, updates: int) -> None:
    """The initial bake: `updates` updates of every group, each adding a bounce (the maps
    replaced). The first traces the rays (`kernel`) and keeps their hits (`kernels.RayHits`); the
    others light the same hits again from the previous update's irradiance
    (`kernels.bounce_hits`): one update's rays and light loops for the whole bake. After each
    update the probes are filtered with their neighbours (`kernels.filter_irradiance`: one set of
    rays, no temporal mean). One chunk of groups at a time (the hits of one chunk are held: bake
    scratch)."""
    gate = self.always
    self.directions(gate)
    hits = self._hits(self.chunk * self.nprobe)
    unfiltered = wp.empty_like(self.struct.irradiance)
    for first_group in range(0, self.ngroup, self.chunk):
      n = min(self.chunk, self.ngroup - first_group)
      self._trace_chunk(kernel, inputs, lit, gate, first_group, n, hits)
      self._blend_chunk(first_group, n, 0.0, gate=gate)
      self._filter_chunk(unfiltered, first_group, n)
      for _ in range(updates - 1):
        self.refresh(gate)
        self._bounce_chunk(hits, gate, first_group, n)
        self._blend_chunk(first_group, n, 0.0, gate=gate)
        self._filter_chunk(unfiltered, first_group, n)
    self.count(gate)

  def _filter_chunk(self, unfiltered: wp.array[wp.vec4h, [int, int]], first_group: int, n: int) -> None:
    """The bake's probe filter over groups [first_group, first_group + n)
    (`kernels.filter_irradiance`, from a copy of the maps in `unfiltered`)."""
    irradiance = self.struct.irradiance
    wp.copy(unfiltered, irradiance)
    g = self.groups
    wp.launch(
      kernels.filter_irradiance,
      dim=(n, self.nprobe, IRR_TEXELS * IRR_TEXELS),
      inputs=[self.struct, unfiltered, self.always, g.rep, g.nworld, first_group],
      outputs=[irradiance],
      device=self.device,
    )

  def _hits(self, rows: int) -> kernels.RayHits:
    """`rows` rows of ray hits (0: none recorded)."""
    shape = (rows, self.rays if rows else 0)
    hits = kernels.RayHits()
    hits.fixed = wp.empty(shape, dtype=wp.vec3, device=self.device)
    hits.weight = wp.empty(shape, dtype=wp.vec3h, device=self.device)
    hits.at = wp.empty(shape, dtype=float, device=self.device)
    hits.normals = wp.empty(shape, dtype=wp.vec2ui, device=self.device)
    return hits

  def _trace_chunk(
    self,
    kernel: wp.Kernel,
    inputs: ProbeInputs,
    lit: Lights,
    gate: Gate,
    first_group: int,
    n: int,
    hits: kernels.RayHits,
  ) -> None:
    """Traces the rays of groups [first_group, first_group + n) into the ray buffers (and `hits`
    unless empty)."""
    ray_radiance, ray_sky = self.ray_buffers()
    g = self.groups

    def launch(a: int, b: int) -> kernels.ProbeLaunch:
      """The launch of the rays of probes [a, b) of the chunk's groups."""
      q = kernels.ProbeLaunch()
      q.dirs, q.updates, q.gate, q.rep, q.nworld = self.dirs, self.updates, gate, g.rep, g.nworld
      q.nray, q.spread = self.rays, math.sqrt(4.0 * math.pi / self.rays)
      q.first_group, q.first, q.nprobe = first_group, a, b - a
      q.ray_radiance, q.ray_sky, q.hits = ray_radiance, ray_sky, hits
      return q

    if inputs.rays is not None:
      inputs.rays(launch(0, self.nprobe), n * self.rays * self.nprobe, lit, inputs.env)
      return

    def run(a: int, b: int) -> None:
      """The rays of probes [a, b) of the chunk's groups."""
      wp.launch(
        kernel,
        dim=(n, self.rays, b - a),
        inputs=[inputs.geo, inputs.mats, lit, inputs.env, launch(a, b)],
        device=self.device,
      )

    self.bands(kernel.module, run)

  def _bounce_chunk(self, hits: kernels.RayHits, gate: Gate, first_group: int, n: int) -> None:
    """Lights the hits of groups [first_group, first_group + n) again (`kernels.bounce_hits`)."""
    ray_radiance, _ = self.ray_buffers()
    g = self.groups

    def run(a: int, b: int) -> None:
      """The rays of probes [a, b)."""
      wp.launch(
        kernels.bounce_hits,
        dim=(n, b - a, self.rays),
        inputs=[self.struct, self.dirs, hits, gate, g.rep, g.nworld, first_group, a],
        outputs=[ray_radiance],
        device=self.device,
      )

    self.bands(wp.get_module(kernels.__name__), run)

  def _blend_chunk(self, first_group: int, n: int, hysteresis: float, *, gate: Gate) -> None:
    """Blends the ray results of groups [first_group, first_group + n) into their irradiance
    maps."""
    irradiance = self.struct.irradiance
    nirr = IRR_TEXELS * IRR_TEXELS
    g = self.groups
    ray_radiance, ray_sky = self.ray_buffers()

    def run(a: int, b: int) -> None:
      """The maps of probes [a, b)."""
      wp.launch(
        kernels.blend_irradiance,
        dim=(n, b - a, nirr),
        inputs=[
          self.struct,
          ray_radiance,
          ray_sky,
          self.irradiance_lobes,
          gate,
          g.rep,
          g.nworld,
          first_group,
          a,
          hysteresis,
        ],
        outputs=[irradiance],
        device=self.device,
      )

    self.bands(wp.get_module(kernels.__name__), run)

  def count(self, gate: Gate) -> None:
    """Ends a gated update: the maps resampled (`refresh`), the update counted (the next one
    uses the next rotation and seed). Every launch tests `gate`; the texture uploads (CUDA array
    copies) run whatever it holds, so this may not go in a CUDA graph's conditional body (CUDA
    rejects array copies there: the graph does not instantiate)."""
    self.refresh(gate)
    wp.launch(kernels.advance, dim=1, inputs=[gate, 1], outputs=[self.updates], device=self.device)


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class UpdateStep:
  """The parameters of one probe update (`ProbeVolume.update`).

  hysteresis: weight of the previous irradiance (the maps are read by the probe hits: each update
  adds a bounce).
  """

  hysteresis: float
