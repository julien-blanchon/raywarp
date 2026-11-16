"""The light grid's visibility test (`grid`) as a flood over the grid's half-cell lattice.

Each light's lattice is flooded outwards from the light instead of testing every (point, light)
pair: the flood starts at the points around the light's bounding box (SEED_STEPS lattice steps) and
tests, round after round, the untested 26-neighbours of the points tested in the previous round
that are fewer than REACH steps from a point the light reaches. The points a point light reaches
form a star-shaped, hence connected, region around it (an area light: one per sample point), so
the flood finds every lattice point it reaches except beyond a gap of REACH or more blocked
lattice steps (light through a slit narrower than the lattice); each light costs its lit region
and a REACH-step shell around it, not the whole lattice times the lights (an interior: most
(point, light) pairs are behind walls). Directional lights are tested from every point.

Rounds run as three launches: `settle_kernel` (pairs whose cells are marked already need only the
centre ray: it decides the flood), the trace (`grid.make_visit`: the software kernel, on the
CPU split over the cores, or the OptiX program) and `spread_kernel` (marks, hops, next round).
The result does not depend on the order pairs run in (deterministic on every device).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, final

import numpy as np
import warp as wp

from raywarp.core.pipeline.platform.device import host_copy, launch_bands

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.rays import SoftwareGeometry
  from raywarp.core.shade.lights.punctual import Lights

wp.set_module_options({"enable_backward": False})

# Encoded (lattice point, light of the chunk) pairs: q * nchunk + c.
type Pairs = np.ndarray[[int], np.dtype[np.int32]]
# A box corner (float32 or float64).
type Corner = np.ndarray[[3], np.dtype[np.float32]] | np.ndarray[[3], np.dtype[np.float64]]

# The flood (module doc): seeds within SEED_STEPS lattice steps of a light's bounding box; an
# occluded point passes it on while fewer than REACH steps from a point the light reaches.
SEED_STEPS: Final = 2
REACH: Final = 2
# Lights are flooded in chunks of at most CHUNK_PAIRS (lattice point, light) pairs (the flood's
# per-pair state: one int32 each); a round's queue starts at QUEUE pairs (grown when it overflows).
CHUNK_PAIRS: Final = 1 << 26
QUEUE: Final = 1 << 20
# CPU traces split into launches of at least this many pairs (`software_trace`).
CPU_SLICE: Final = 256
# Flag of a pair whose cells all see its light already (`settle_kernel`; pairs stay below it).
MARKED: Final = 1 << 30


@wp.struct
class Pass:
  """One round of the flood (module doc): the (lattice point, light) pairs to trace.

  `pairs[i]` = q * nchunk + c: lattice point q and light light0 + c; the trace (`make_visit`)
  replaces it by -1 - pair where the light reaches the point. Kept small: it is a field of the
  OptiX launch parameters (`trace.optix.device.params`).
  """

  origin: wp.vec3
  cell: float
  dims: wp.vec3i
  pairs: wp.array[wp.int32]
  light0: int
  nchunk: int


@wp.func
def lattice_ijk(
  # In:
  dims: wp.vec3i,
  q: int,
) -> wp.vec3i:
  """The (i, j, k) of lattice point q ((2 dims + 1) points per axis, x fastest)."""
  nx = 2 * dims[0] + 1
  ny = 2 * dims[1] + 1
  return wp.vec3i(q % nx, (q // nx) % ny, q // (nx * ny))


@wp.kernel
def seed_kernel(
  # In:
  pairs: wp.array[wp.int32],
  nchunk: int,
  # Out:
  state: wp.array2d[wp.int32],
):
  """Claims the seed pairs for round 0 (`spread_kernel`'s state)."""
  i = wp.tid()
  pair = pairs[i]
  q = pair // nchunk
  state[q, pair - q * nchunk] = 0


@wp.func
def cells_of(
  # In:
  dims: wp.vec3i,
  ijk: wp.vec3i,
) -> tuple[wp.vec3i, wp.vec3i]:
  """(lo, hi) cells holding lattice point `ijk` (a corner is shared by 8 cells, a face point by 2;
  per axis cells (i + 1) // 2 - 1 and i // 2, the same if i is odd)."""
  lo = wp.vec3i(wp.max((ijk[0] + 1) // 2 - 1, 0), wp.max((ijk[1] + 1) // 2 - 1, 0), wp.max((ijk[2] + 1) // 2 - 1, 0))
  hi = wp.vec3i(wp.min(ijk[0] // 2, dims[0] - 1), wp.min(ijk[1] // 2, dims[1] - 1), wp.min(ijk[2] // 2, dims[2] - 1))
  return lo, hi


@wp.kernel
def settle_kernel(
  # In:
  g: Pass,
  visible: wp.array2d[wp.int32],
):
  """Flags pair i MARKED when every cell holding its point sees its light already (marked by
  earlier rounds: deterministic): its trace then only decides whether the flood passes on."""
  i = wp.tid()
  pair = g.pairs[i]
  q = pair // g.nchunk
  li = g.light0 + pair - q * g.nchunk
  ijk = lattice_ijk(g.dims, q)
  lo, hi = cells_of(g.dims, ijk)
  for z in range(lo[2], hi[2] + 1):
    for y in range(lo[1], hi[1] + 1):
      for x in range(lo[0], hi[0] + 1):
        if visible[(z * g.dims[1] + y) * g.dims[0] + x, li] == 0:
          return
  g.pairs[i] = pair | wp.static(MARKED)


@wp.func
def occluded_hop(
  # In:
  state: wp.array2d[wp.int32],
  n: wp.vec3i,
  lo: wp.vec3i,
  hi: wp.vec3i,
  c: int,
  rnd: int,
) -> int:
  """The hop of an occluded pair of round `rnd` (light c, neighbours lo..hi of `n` lattice points
  per axis): 1 + the smallest hop of its neighbours of earlier rounds (none, a seed: 1)."""
  best = int(0)  # 1 + the smallest settled neighbour hop (0: none)
  for z in range(lo[2], hi[2] + 1):
    for y in range(lo[1], hi[1] + 1):
      for x in range(lo[0], hi[0] + 1):
        sm = state[(z * n[1] + y) * n[0] + x, c]
        if sm >= 0 and sm < rnd * 4 and (best == 0 or sm % 4 + 1 < best):
          best = sm % 4 + 1
  return wp.where(best == 0, 1, best)


@wp.kernel
def spread_kernel(
  # In:
  g: Pass,
  rnd: int,
  # Out:
  visible: wp.array2d[wp.int32],
  state: wp.array2d[wp.int32],
  queued: wp.array[wp.int32],
  count: wp.array[wp.int32],
):
  """Settles traced pair i of round `rnd` and queues its lattice point's neighbours.

  `state` per (lattice point, light): -1 untested, else round * 4 + hop (claimed for a round: hop
  0 until settled). A lit pair marks its cells (`visible`), hop 0; an occluded one's hop is 1 +
  the smallest hop of its neighbours of earlier rounds (seeds: 1). A pair under REACH claims each
  untested 26-neighbour for the next round (first claim wins) and queues it (`queued` past its
  size: only counted, `gather_kernel` lists them).
  """
  i = wp.tid()
  pair = g.pairs[i]
  reached = pair < 0
  pair = wp.where(reached, -1 - pair, pair & wp.static(MARKED - 1))
  q = pair // g.nchunk
  c = pair - q * g.nchunk
  ijk = lattice_ijk(g.dims, q)
  n = wp.vec3i(2 * g.dims[0] + 1, 2 * g.dims[1] + 1, 2 * g.dims[2] + 1)
  lo = wp.vec3i(wp.max(ijk[0] - 1, 0), wp.max(ijk[1] - 1, 0), wp.max(ijk[2] - 1, 0))
  hi = wp.vec3i(wp.min(ijk[0] + 1, n[0] - 1), wp.min(ijk[1] + 1, n[1] - 1), wp.min(ijk[2] + 1, n[2] - 1))
  h = int(0)
  if reached:
    clo, chi = cells_of(g.dims, ijk)
    for z in range(clo[2], chi[2] + 1):
      for y in range(clo[1], chi[1] + 1):
        for x in range(clo[0], chi[0] + 1):
          visible[(z * g.dims[1] + y) * g.dims[0] + x, g.light0 + c] = 1
  else:
    h = occluded_hop(state, n, lo, hi, c, rnd)
  state[q, c] = rnd * 4 + h
  if h >= wp.static(REACH):
    return
  for z in range(lo[2], hi[2] + 1):
    for y in range(lo[1], hi[1] + 1):
      for x in range(lo[0], hi[0] + 1):
        m = (z * n[1] + y) * n[0] + x
        if state[m, c] == -1:
          if wp.atomic_cas(state, m, c, -1, (rnd + 1) * 4) == -1:
            slot = wp.atomic_add(count, 0, 1)
            if slot < queued.shape[0]:
              queued[slot] = m * g.nchunk + c


@wp.kernel
def gather_kernel(
  # In:
  state: wp.array2d[wp.int32],
  rnd: int,
  # Out:
  queued: wp.array[wp.int32],
  count: wp.array[wp.int32],
):
  """Lists the pairs claimed for round `rnd` (a queue that overflowed in `spread_kernel`)."""
  q, c = wp.tid()
  if state[q, c] == rnd * 4:
    queued[wp.atomic_add(count, 0, 1)] = q * state.shape[1] + c


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Boxes:
  """World 0's lights as the flood seeds them: (nlight,) on / directional flags, (nlight, 3)
  centres, (nlight,) bounding-sphere radii [m] (area: half extents + radius)."""

  active: np.ndarray[[int], np.dtype[np.bool_]]
  directional: np.ndarray[[int], np.dtype[np.bool_]]
  centre: np.ndarray[[int, 3], np.dtype[np.float64]]
  extent: np.ndarray[[int], np.dtype[np.float64]]


def seeds(boxes: Boxes, *, light0: int, nchunk: int, lo: Corner, cell: float, dims: Sequence[int]) -> Pairs:
  """Round 0 of the flood of lights light0 .. light0 + nchunk - 1: the lattice points within
  SEED_STEPS steps of each active light's bounding box (directional lights: every point), as
  encoded pairs (`Pass.pairs`)."""
  n = [2 * d + 1 for d in dims]
  step = 0.5 * cell
  out = []
  for c in range(nchunk):
    li = light0 + c
    if not bool(boxes.active[li]):
      continue
    if bool(boxes.directional[li]):
      q = np.arange(n[0] * n[1] * n[2]).astype(np.int64)
    else:
      r = float(boxes.extent[li])
      # Per axis: the lattice range of the box, SEED_STEPS more on each side, clipped.
      span = []
      for d in range(3):
        x = (float(boxes.centre[li, d]) - float(lo[d])) / step
        first = min(max(math.floor(x - r / step) - SEED_STEPS, 0), n[d] - 1)
        last = min(max(math.ceil(x + r / step) + SEED_STEPS, 0), n[d] - 1)
        span.append(np.arange(last - first + 1).astype(np.int64) + first)
      i, j, k = np.meshgrid(*span, indexing="ij")
      q = ((k * n[1] + j) * n[0] + i).ravel()
    out.append(q * nchunk + c)
  return np.concatenate(out).astype(np.int32) if out else np.zeros(0, np.int32)


def software_trace(kernel: wp.Kernel, geo: SoftwareGeometry, lit: Lights, g: Pass, npair: int) -> None:
  """Traces the `npair` pairs of `g` with the software kernel, on the CPU split over the cores
  (`device.launch_bands`: each band writes its own slice of `g.pairs`, the same result)."""
  dev = g.pairs.device

  def run(a: int, b: int) -> None:
    """Pairs [a, b)."""
    part = Pass()
    part.origin, part.cell, part.dims, part.light0, part.nchunk = g.origin, g.cell, g.dims, g.light0, g.nchunk
    part.pairs = g.pairs[a:b]
    wp.launch(kernel, dim=b - a, inputs=[geo, lit, part], device=dev)

  launch_bands(kernel.module, device=dev, n=npair, launch=run, min_band=CPU_SLICE)


def flood(
  trace: Callable[[Pass, int], None], g: Pass, *, first: Pairs, npoint: int, visible: wp.array[wp.int32, [int, int]]
) -> None:
  """Floods the chunk of lights of `g` (`light0`, `nchunk`) from the `first` round's pairs
  (`seeds`), marking `visible` (on its device): per round, flags the pairs whose cells are marked
  (`settle_kernel`), traces them (`trace(g, npair)`), settles them and queues the next round
  (`spread_kernel`)."""
  npair = len(first)
  if npair == 0:
    return
  dev = visible.device
  state = wp.full((npoint, g.nchunk), -1, dtype=wp.int32, device=dev)
  size = max(npair, QUEUE)
  pairs, queued = wp.empty(size, dtype=wp.int32, device=dev), wp.empty(size, dtype=wp.int32, device=dev)
  count = wp.zeros(1, dtype=wp.int32, device=dev)
  pairs[:npair].assign(first)
  wp.launch(seed_kernel, dim=npair, inputs=[pairs, g.nchunk], outputs=[state], device=dev)
  rnd = 0
  while npair:
    g.pairs = pairs
    wp.launch(settle_kernel, dim=npair, inputs=[g, visible], device=dev)
    trace(g, npair)
    count.zero_()
    wp.launch(spread_kernel, dim=npair, inputs=[g, rnd], outputs=[visible, state, queued, count], device=dev)
    npair = int(host_copy(count)[0])
    if npair > queued.shape[0]:  # the queue overflowed: list the round's claims from `state`
      size = npair
      pairs, queued = wp.empty(size, dtype=wp.int32, device=dev), wp.empty(size, dtype=wp.int32, device=dev)
      count.zero_()
      wp.launch(gather_kernel, dim=(npoint, g.nchunk), inputs=[state, rnd + 1], outputs=[queued, count], device=dev)
    pairs, queued = queued, pairs
    rnd += 1
