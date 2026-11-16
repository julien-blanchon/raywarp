"""The lights a shading point evaluates exactly: every light of its cell, or the K strongest (caps).

A light loop (`loop.py`, `shadows/budget.py`) asks its `Source` for the lights to evaluate:

* uncapped: the point's light-grid cell (every light without the grid), each light exactly, plus
  the cell's far field (`grid`: its weak distant lights' unshadowed irradiance, through the
  lobes' albedo as `remainder`: `lobe_remainder`);
* capped (`lighting.max_exact` / `reflections.max_exact`, every light form): the
  cell's lights are ranked by a cheap estimate of their irradiance, the punctual form's
  `eval_light` x NoL (no BRDF, no LTC, no shadow ray; an area light's punctual form is its
  Lambertian point-light equivalent), the K strongest are evaluated exactly (LTC for area lights,
  shadow rays per the loop), the others' estimated irradiance reaches the point through the
  diffuse lobe and the specular lobe's albedo spread over the hemisphere, unshadowed
  (`remainder`: `lobe_remainder`), with the cell's far field.

`begin(geo, lit, worldid, point, normal) -> (slots, first, last, rest)`: the loop runs entries
first..last-1 (`light_of(lit, slots, j)` -> light id); `rest` is the irradiance left to
`remainder(rest, lobes)` (the far field; capped, plus the capped-out lights'); `Source.rest`
says whether the loop adds it (capped or culled). With counters (`kernels/abi/counters.py`) `begin`
counts ranking entries and evaluator candidates, before the evaluator rejects inactive,
out-of-range, back-facing or zero terms. These are not counts of actual BRDF/LTC executions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple, final

import warp as wp

from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi.features import RUNTIME_LIGHTS, Features
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.post.grading import color
from raywarp.core.shade.lights import ies as ies_lib
from raywarp.core.shade.lights import punctual
from raywarp.core.shade.lights.punctual import Light, Lights
from raywarp.core.shade.materials import brdf
from raywarp.core.shade.materials.material import Lobes

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Counters columns of a source: (evaluator candidate entries, lights ranked cheaply); -1: none.
type Columns = tuple[int, int]

# cell_range(lit, worldid, point, normal) -> (cell, first, last, far field); cell_light(lit, cell,
# j) -> light id (`make_cell`).
type CellRange = Callable[[Lights, int, wp.vec3, wp.vec3], tuple[int, int, int, wp.vec3]]
type CellLight = Callable[[Lights, int, int], int]
# estimate(lit, worldid, li, point, normal) -> cheap irradiance (`_make_estimate`); with IES
# profiles (ranking, remainder) irradiance (`_make_ies_estimate`).
type Estimate = Callable[[Lights, int, int, wp.vec3, wp.vec3], wp.vec3]
# A `Source`'s functions (module doc): begin, light_of, remainder.
type Begin[S] = Callable[[GeometryFields, Lights, int, wp.vec3, wp.vec3], tuple[S, int, int, wp.vec3]]
type LightOf[S] = Callable[[Lights, S, int], int]
type Remainder = Callable[[wp.vec3, Lobes], wp.vec3]


@wp.func
def far_always(
  # Model:
  _lit: Lights,
) -> bool:
  """A source whose `rest` is always added (`Source.far`)."""
  return True


@final
class Source[S](NamedTuple):
  """The lights of a shading point (module doc): `begin`, `light_of`, `remainder`; `capped`: K >
  0; `rest`: the loop adds `remainder(rest, lobes)` (capped, or a light grid's far fields) where
  `far(lit)` (an uncapped source: the light grid's lookup on, its build-word bit,
  `make_source(bound=...)`; a capped one: always)."""

  begin: Begin[S]
  light_of: LightOf[S]
  remainder: Remainder
  capped: bool
  rest: bool
  # A counted build's source (`Counts.dual`): capped at the run-time cap, every light
  # of the cell at cap 0; `is_capped(slots)` tells which (`capped` is True: the loops' capped form).
  dual: bool = False
  is_capped: Callable[..., bool] | None = None
  far: Callable[[Lights], bool] = far_always


type Sources = Source[int] | Source[wp.Vector[int, int]]


# `Counts.toggle`: the run-time cap of the camera rays / of the glossy-reflection hits
# (`Toggles.light_cap` / `hit_light_cap`); STATIC_CAP: the compiled cap alone (the GI probe rays).
STATIC_CAP: Final = -1
CAMERA_CAP: Final = 0
HIT_CAP: Final = 1


@final
class Counts(NamedTuple):
  """How a capped source reads its cap (`make_source`).

  Attributes:
    toggle: STATIC_CAP (the compiled cap alone), CAMERA_CAP or HIT_CAP (the `Toggles` cap read).
    dual: a counted build's: the run-time cap as is, 0 every light.
    run: a tier build's: the run-time cap within the compiled one, where `toggles.RUN_COUNTS` is on.
    switches: the build's run-time switches (`toggles.switches_of`; the bit's `toggles.setting`).
  """

  toggle: int = STATIC_CAP
  dual: bool = False
  run: bool = False
  switches: int = 0

  def reads(self) -> bool:
    """Whether the source reads a run-time cap at all."""
    return self.toggle != STATIC_CAP and (self.dual or self.run)


STATIC: Final = Counts()


def counts_of(f: Features, toggle: int) -> Counts:
  """How the capped sources of build `f` read their cap `toggle` at run time: a counted tier
  build as is (`Features.run_counts`), a build reading `toggles.RUN_COUNTS` within its
  compiled cap, else not."""
  run = toggles.reads(f, toggles.RUN_COUNTS)
  return Counts(toggle=toggle, dual=f.run_counts, run=run, switches=toggles.switches_of(f))


# entries(lit, head) -> the loop's entries after the source's lights (the headlight: 0 or 1);
# is_head(lit, j, last) -> whether entry `j` of a loop over the source's entries first..last-1 is
# the headlight's; light_at(lit, worldid, li, head_dir, head) -> the light of entry `li`;
# with_head(lit, total, head_c) -> a loop's result plus the headlight's term (`make_entries`).
type Entries = Callable[[Lights, wp.vec3], int]
type IsHead = Callable[[Lights, int, int], bool]
type LightAt = Callable[[Lights, int, int, wp.vec3, wp.vec3], Light]
type WithHead = Callable[[Lights, wp.vec3, wp.vec3], wp.vec3]


@final
class HeadEntries(NamedTuple):
  """A light loop's entries past the source's lights (`make_entries`)."""

  entries: Entries
  is_head: IsHead
  light_at: LightAt
  with_head: WithHead


@specialize
def _make_run_cap(cap: int, *, toggle: int, dual: bool, switches: int, bound: int):
  """run_cap(rt, bits2) -> the cap a source of compiled cap `cap` ranks with (`make_source`; not
  STATIC_CAP): the run-time cap of `toggle` (`dual`: as is, 0 every light), else, where the build
  word or switch `toggles.RUN_COUNTS` is on (`switches`, `bound`: `toggles.setting`), that cap
  when it is below `cap`, else `cap`."""
  setting = toggles.setting(switches, bound)

  @wp.func
  def run_cap(
    # In:
    rt: toggles.Toggles,
    bits2: int,
  ) -> int:
    """The cap of the shading point's ranking."""
    k = rt.light_cap
    if wp.static(toggle == HIT_CAP):
      k = rt.hit_light_cap
    if wp.static(dual):
      return k
    if setting(rt.flags, bits2, wp.static(toggles.RUN_COUNTS)):
      if k > 0 and k < wp.static(cap):
        return k
    return wp.static(cap)

  return run_cap


@specialize
def make_entries(*, headlight: bool, bound: int = 0):
  """The entries of a light loop past the source's lights: the camera headlight as one more entry
  (`headlight`, Features.headlight; light id `punctual.HEADLIGHT`, `punctual.light_at`) or none (the
  loop's code is the plain fetch). The loop picks entry j's id: the source's `light_of` unless
  `is_head` (j past `last`). `bound` (Features.bound): the entry reads the build word's headlight
  bit (`toggles.CLASS_HEADLIGHT`), off: the loop is the headlight-free one."""
  gate = toggles.bound_on(bound)

  @wp.func(inline=True)
  def on(
    # Model:
    lit: Lights,
  ) -> bool:
    """Whether the headlight's entry is compiled in and its class bit on."""
    if wp.static(headlight):
      return gate(lit.bits, wp.static(toggles.CLASS_HEADLIGHT))
    return False

  @wp.func(inline=True)
  def entries(
    # Model:
    lit: Lights,
    # In:
    head: wp.vec3,
  ) -> int:
    """1 for the headlight's entry when it is on (`head` nonzero), else 0."""
    if on(lit):
      if wp.length_sq(head) > 0.0:
        return 1
    return 0

  @wp.func(inline=True)
  def is_head(
    # Model:
    lit: Lights,
    # In:
    j: int,
    last: int,
  ) -> bool:
    """Whether entry `j` (of first..last-1 and the `entries` past them) is the headlight's."""
    if on(lit):
      return j >= last
    return False

  @wp.func(inline=True)
  def light_at(
    # Model:
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    head_dir: wp.vec3,
    head: wp.vec3,
  ) -> Light:
    """Light `li` (`punctual.light_at`; without the headlight: `punctual.fetch`)."""
    if on(lit):
      return punctual.light_at(lit, worldid, li, head_dir, head)
    return punctual.fetch(lit, worldid, li)

  @wp.func(inline=True)
  def with_head(
    # Model:
    lit: Lights,
    # In:
    total: wp.vec3,
    head_c: wp.vec3,
  ) -> wp.vec3:
    """`total` plus the headlight's term `head_c` (without the headlight: `total`)."""
    if on(lit):
      return total + head_c
    return total

  return HeadEntries(entries, is_head, light_at, with_head)


def make_source(
  *,
  nlight: int,
  culling: bool,
  cap: int,
  has_spot: bool,
  columns: Columns,
  ies: bool = False,
  counts: Counts = STATIC,
  bound: int = 0,
):
  """The source of a light loop: the cell's lights (`cap` 0: `resolve` zeroes a cap not below the
  light count), else the `cap` strongest of them (`has_spot`: Features.has_spot for the
  cheap estimate; `ies`: Features.ies, profiled lights ranked by their coarse profile's bound and
  their remainder by its azimuthal mean, `_make_ies_estimate`). `counts` (`Counts`): with `toggle`
  CAMERA_CAP / HIT_CAP and `dual` (a counted tier build, `Features.run_counts`)
  `cap` is the capacity, the cap read at run time (`Toggles`), 0 every light of the cell; with
  `run` (a tier build reading `toggles.RUN_COUNTS`, `switches`: its run-time switches) a run-time
  cap below `cap` ranks with it where the bit is on (0 or above: `cap`, whose slots then hold
  every light a configuration's cap at or above the scene's light count selects). `bound`
  (Features.bound): the IES ranking, the spot cones of the cheap estimate and the grid's lookup
  read the build word's bits (`toggles.CLASS_IES`, `SCENE_SPOT`, `CLASS_CULLING`)."""
  cell_range, cell_light = make_cell(nlight, culling=culling, bound=bound)
  if cap <= 0:
    return _uncapped(cell_range, cell_light, columns[0], culling=culling, bound=bound)
  return _capped(cell_range, cell_light, cap=cap, has_spot=has_spot, columns=columns, ies=ies, counts=counts, bound=bound)


@specialize
def make_exact_test(
  *,
  nlight: int,
  culling: bool,
  cap: int,
  has_spot: bool,
  ies: bool = False,
  counts: Counts = STATIC,
  bound: int = 0,
):
  """exact(geo, lit, worldid, li, point, normal): whether the light loop of `make_source`'s
  arguments shades light `li` exactly at (`point`, `normal`): uncapped, every light of the cell
  is; capped, `li` is in the cell and fewer than the cap of its lights rank strictly above it
  (the same cheap estimate as `_capped`'s ranking; a tie counts as exact). For the reflection
  hits' emissive panels (`reflections.panel_light`): a panel's emission is seen in a mirror
  unless the shading point's exact light term already integrates it. One pass over the cell,
  only where a reflection ray meets a panel light under a cap."""
  cell_range, cell_light = make_cell(nlight, culling=culling, bound=bound)
  estimate = _make_estimate(has_spot=has_spot, bound=bound)
  ies_estimate = _make_ies_estimate(has_spot=has_spot, bound=bound)
  reads = counts.reads()
  run_cap = _make_run_cap(max(cap, 0), toggle=counts.toggle, dual=counts.dual, switches=counts.switches, bound=bound)
  gate = toggles.bound_on(bound)  # the IES class bit (`bound`: Features.bound)

  @wp.func(inline=True)
  def rank_y(
    # Model:
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> float:
    """The ranking luminance of light `li` at the point (`_capped`'s)."""
    if wp.static(ies):
      if gate(lit.bits, wp.static(toggles.CLASS_IES)):
        rank, _e = ies_estimate(lit, worldid, li, point, normal)
        return color.luminance(rank)
    return color.luminance(estimate(lit, worldid, li, point, normal))

  @wp.func
  @device_generic
  def exact[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> bool:
    """Whether light `li` is shaded exactly at the point (see `make_exact_test`)."""
    k_cap = int(wp.static(max(cap, 0)))  # pyrefly: ignore[unnecessary-type-conversion]  # a variable (reassigned below), not a constant
    if wp.static(reads):
      k_cap = run_cap(geo.rt, geo.bits2)
    cell, first, last, _far = cell_range(lit, worldid, point, normal)
    found = bool(False)
    # A cell within the cap shades every light exactly (`_capped`); past it, only lights of a
    # positive estimate take a slot, ranked until `k_cap` of them rank above `li`.
    ranked = k_cap > 0 and last - first > k_cap
    y = float(0.0)
    if ranked:
      y = rank_y(lit, worldid, li, point, normal)
    above = int(0)
    for j in range(first, last):
      other = cell_light(lit, cell, j)
      if other == li:
        found = True
      elif ranked and above < k_cap:
        if rank_y(lit, worldid, other, point, normal) > y:
          above += 1
    return found and (not ranked or (y > 0.0 and above < k_cap))

  return exact


@wp.func
def grid_free(
  # Model:
  lit: Lights,
  # In:
  worldid: int,
) -> bool:
  """Whether world `worldid` shades without the light grid: none built (`Grid.valid` empty), or
  its light row changed since the build (`Grid.valid` 0). Every point then takes the whole table
  (`make_cell`)."""
  rows = lit.grid_valid.shape[0] // 2
  if rows == 0:
    return True
  return lit.grid_valid[worldid % rows] == 0


# The factories below are cached: a megakernel builds its camera rays' and its reflection hits'
# sources from the same arguments, and Warp emits one copy per wp.func object (no deduplication).
@specialize
def make_cell(nlight: int, *, culling: bool, bound: int = 0):
  """The point's light-grid entries and entry-to-light lookup (all lights without culling).

  `cell_range(lit, worldid, point, normal)` returns (cell, first, last, the cell's far-field
  irradiance at `normal`, `far_field`); `cell_light(lit, cell, j)` returns its light id. `nlight`:
  the light count compiled in, or RUNTIME_LIGHTS (the light table's, `Lights.nlight`). `bound`
  (Features.bound): the grid's lookup reads the build word's culling bit (`toggles.CLASS_CULLING`).
  """
  table = _table_count(nlight)
  gate = toggles.bound_on(bound)

  @wp.func
  def cell_range(
    # Model:
    lit: Lights,
    # In:
    worldid: wp.int32,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> tuple[int, int, int, wp.vec3]:
    """(cell id, first, last, far-field irradiance): cell -1 the whole light table (entry j is
    light j), -2 a cell without movable lights (entries first..last-1 of `grid_lights`), else the
    cell's static lights then the movable ones (entries 0..last-1, `cell_light`)."""
    zero = wp.vec3(0.0)
    if wp.static(culling):
      if gate(lit.bits, wp.static(toggles.CLASS_CULLING)):
        if grid_free(lit, worldid):
          return -1, 0, table(lit), zero
        q = (point - lit.grid_origin) * lit.grid_inv_cell
        if q[0] < 0.0 or q[1] < 0.0 or q[2] < 0.0:
          return -1, 0, table(lit), zero
        if q[0] >= float(lit.grid_dims[0]) or q[1] >= float(lit.grid_dims[1]) or q[2] >= float(lit.grid_dims[2]):
          return -1, 0, table(lit), zero
        c = punctual.cell_of(lit.grid_origin, lit.grid_inv_cell, lit.grid_dims, point, normal)
        far = far_field(lit, worldid, c, normal)
        # The movable lights: the list after the last cell's (`Grid.start`).
        ncell = lit.grid_dims[0] * lit.grid_dims[1] * lit.grid_dims[2]
        ndynamic = lit.grid_start[ncell + 1] - lit.grid_start[ncell]
        if ndynamic == 0:
          return -2, lit.grid_start[c], lit.grid_start[c + 1], far
        count = lit.grid_start[c + 1] - lit.grid_start[c]
        return c, 0, count + ndynamic, far
    return -1, 0, table(lit), zero

  @wp.func
  def cell_light(
    # Model:
    lit: Lights,
    # In:
    cell: int,
    j: int,
  ) -> int:
    """Light of slot `j`: static cell lights, then the movable list exactly once."""
    if wp.static(culling):
      if gate(lit.bits, wp.static(toggles.CLASS_CULLING)):
        if cell == -2:
          return lit.grid_lights[j]
        if cell >= 0:
          start = lit.grid_start[cell]
          count = lit.grid_start[cell + 1] - start
          if j >= count:
            return lit.grid_lights[lit.grid_start[lit.grid_dims[0] * lit.grid_dims[1] * lit.grid_dims[2]] + j - count]
          return lit.grid_lights[start + j]
    return j

  return cell_range, cell_light


@specialize
def _table_count(nlight: int):
  """count(lit): the light table's entries: `nlight`, a constant, or with RUNTIME_LIGHTS
  `lit.nlight`."""

  @wp.func
  def count(
    # Model:
    lit: Lights,
  ) -> int:
    """The light table's entry count."""
    if wp.static(nlight == RUNTIME_LIGHTS):
      return lit.nlight
    return wp.static(nlight)

  return count


@wp.func
def far_field(
  # Model:
  lit: Lights,
  # In:
  worldid: int,
  cell: int,
  normal: wp.vec3,
) -> wp.vec3:
  """Irradiance of light-grid cell `cell`'s far field (`grid.Grid.far`) on a surface of
  normal `normal`: its ambient cube, sum of n_i^2 x the luminance on the axis normal of n_i's sign
  (exact on the axis normals: floors, walls, ceilings), in the far field's color; zero without far
  fields or when the world's lights differ from the build's (`Grid.valid`'s second half)."""
  if lit.grid_far.shape[0] == 0:
    return wp.vec3(0.0)
  rows = lit.grid_valid.shape[0] // 2
  if lit.grid_valid[rows + worldid % rows] == 0:
    return wp.vec3(0.0)
  rgb = lit.grid_far[3 * cell]
  y = color.luminance(rgb)
  if y <= 0.0:
    return wp.vec3(0.0)
  pos = lit.grid_far[3 * cell + 1]
  neg = lit.grid_far[3 * cell + 2]
  e = float(0.0)
  for i in range(3):
    e += normal[i] * normal[i] * wp.where(normal[i] > 0.0, pos[i], neg[i])
  return rgb * (e / y)


@wp.func
def no_remainder(
  # In:
  rest: wp.vec3,
  _lobes: Lobes,
) -> wp.vec3:
  """Nothing is left over (uncapped without a light grid: `rest` is zero)."""
  return rest


@wp.func
def lobe_remainder(
  # In:
  rest: wp.vec3,
  lobes: Lobes,
) -> wp.vec3:
  """The irradiance left over (the capped-out lights', the light grid's far field) through the
  diffuse lobe (the coupled diffuse albedo) and the specular one spread over the hemisphere: the
  flux the specular lobe reflects of an irradiance E is E x its albedo (spec_albedo x the
  multiscatter compensation), returned as uniform radiance E x albedo / pi. Its highlights are
  lost (no direction), not its energy: a glossy floor under capped-out lamps kept 0.934 of the
  all-exact mean with the diffuse lobe alone."""
  albedo = lobes.diffuse_color + wp.cw_mul(lobes.spec_albedo, lobes.energy_comp)
  return wp.cw_mul(wp.max(rest, wp.vec3(0.0)), albedo) * brdf.diffuse_lambert()


@specialize
def _uncapped(cell_range: CellRange, cell_light: CellLight, column: int, *, culling: bool, bound: int = 0) -> Source[int]:
  """Every light of the cell, exactly, and its far field (`culling`; `bound`: Features.bound, the
  far field added with the build word's culling bit on, `Source.far`)."""
  gate = toggles.bound_on(bound)

  @wp.func
  def far_on(
    # Model:
    lit: Lights,
  ) -> bool:
    """Whether the light grid's lookup is on (its far field added)."""
    return gate(lit.bits, wp.static(toggles.CLASS_CULLING))

  @wp.func
  @device_generic
  def begin[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> tuple[int, int, int, wp.vec3]:
    """(cell id, the cell's [first, last), its far field)."""
    cell, first, last, far = cell_range(lit, worldid, point, normal)
    if wp.static(column >= 0):
      counters.add(geo, worldid, wp.static(column), last - first)
    return cell, first, last, far

  @wp.func
  def light_of(
    # Model:
    lit: Lights,
    # In:
    slots: int,
    j: int,
  ) -> int:
    """Light of cell entry `j`."""
    return cell_light(lit, slots, j)

  return Source(
    begin=begin,
    light_of=light_of,
    remainder=lobe_remainder if culling else no_remainder,
    capped=False,
    rest=culling,
    far=far_on,
  )


@specialize
def _make_estimate(*, has_spot: bool, bound: int):
  """estimate(lit, worldid, li, point, normal) -> the cheap irradiance that ranks the lights of
  `_capped`, from the lights' hot records (`punctual.LightHot`; `bound`: Features.bound, the spot
  cones' bit)."""
  gate = toggles.bound_on(bound)

  @wp.func
  def estimate(
    # Model:
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> wp.vec3:
    """Unshadowed irradiance of light `li`'s punctual form at the point (0: off, behind, out of
    range)."""
    hot = punctual.fetch_hot(lit, worldid, li)
    # Evaluated whatever the active bit (selected after): no branch between the record's loads.
    spotted = False
    if wp.static(has_spot):
      spotted = gate(lit.bits, wp.static(toggles.SCENE_SPOT))
    l, _dist, scale = punctual.eval_hot(lit, worldid, li, hot, spotted, point)
    e = scale * wp.max(wp.dot(normal, l), 0.0)
    return wp.where((hot.flags & wp.static(punctual.HOT_ACTIVE)) != 0, e, wp.vec3(0.0))

  return estimate


@specialize
def _make_ies_estimate(*, has_spot: bool, bound: int):
  """estimate(lit, worldid, li, point, normal) -> (ranking irradiance, remainder irradiance) of
  `_capped` with IES profiles: the hot record's estimate times the light's coarse profile toward
  the point (`ies.coarse`: a bound of the profile for the ranking, never below it, and its
  azimuthal mean for the remainder), not the exact profile (acos, atan2 and four loads per cell
  light): the exact set evaluates it exactly. `bound`: Features.bound (the spot cones' bit)."""
  gate = toggles.bound_on(bound)

  @wp.func(inline=True)
  def estimate(
    # Model:
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> tuple[wp.vec3, wp.vec3]:
    """(bound-weighted, mean-weighted) unshadowed irradiance of light `li`'s punctual form."""
    hot = punctual.fetch_hot(lit, worldid, li)
    spotted = False
    if wp.static(has_spot):
      spotted = gate(lit.bits, wp.static(toggles.SCENE_SPOT))
    l, _dist, scale = punctual.eval_hot(lit, worldid, li, hot, spotted, point)
    e = wp.where((hot.flags & wp.static(punctual.HOT_ACTIVE)) != 0, scale * wp.max(wp.dot(normal, l), 0.0), wp.vec3(0.0))
    instance = hot.flags >> wp.static(punctual.HOT_IES_SHIFT)
    # A tube's nadir is not in the hot record: its peak bounds it (and estimates its remainder).
    if instance == 0 or (hot.flags & wp.static(punctual.HOT_TYPE_MASK)) == wp.static(punctual.LIGHT_TUBE):
      return e, e
    w = ies_lib.coarse(lit.ies, instance, wp.dot(-l, hot.dir))
    return e * w[0], e * w[1]

  return estimate


@specialize
def _capped[N: int](  # noqa: C901, PLR0915  # the IES ranking and the run-time cap are compile-time branches
  cell_range: CellRange,
  cell_light: CellLight,
  *,
  cap: N,
  has_spot: bool,
  columns: Columns,
  ies: bool = False,
  counts: Counts = STATIC,
  bound: int = 0,
) -> Source[wp.Vector[int, N]]:
  """The `cap` strongest lights of the cell by cheap irradiance, the rest diffuse (a cell within the
  cap: every light, in order, as uncapped). `counts` (`make_source`): at most `cap`, the run-time
  cap's strongest (slots past it never take a light); `dual`: every light of the cell at a run-time
  cap 0 (the slots then hold the cell, `Ids[cap]`, and `top[0]` is -1)."""
  dual, reads = counts.dual, counts.reads()
  run_cap = _make_run_cap(cap, toggle=counts.toggle, dual=dual, switches=counts.switches, bound=bound)
  gate = toggles.bound_on(bound)  # the IES class bit (`bound`: Features.bound)
  Ids = wp.types.vector(length=cap + 1 if dual else cap, dtype=int)
  Lums = wp.types.vector(length=cap, dtype=float)
  exact_col, cheap_col = columns
  estimate = _make_estimate(has_spot=has_spot, bound=bound)
  ies_estimate = _make_ies_estimate(has_spot=has_spot, bound=bound)

  @wp.func
  @device_generic
  @concrete(**{"return": tuple[Ids, int, int, wp.vec3]})
  def begin[G: GeometryFields](  # noqa: C901  # compile-time branches: IES ranking, run-time cap
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
  ) -> tuple[wp.Vector[int, N], int, int, wp.vec3]:
    """(the slots' light ids, 0, occupied slots, the others' estimated irradiance).

    The slots are indexed by constants only (unrolled loops over the `cap` slots, the weakest
    slot's luminance kept as a scalar): a dynamically indexed slot vector lives in local memory
    on the GPU (as `shadows.budget`'s). Empty slots (luminance 0) are the weakest, the first of
    them first: the occupied slots are a prefix, in the order they were filled.
    """
    top = Ids(-1)
    top_y = Lums(0.0)
    # Each slot's estimate (by channel): subtracted from the total without a second fetch.
    top_r, top_g, top_b = Lums(0.0), Lums(0.0), Lums(0.0)
    weakest = int(0)  # the first slot of the smallest luminance
    weak_y = float(0.0)  # top_y[weakest]
    total = wp.vec3(0.0)
    cell, first, last, far = cell_range(lit, worldid, point, normal)
    k_cap = int(wp.static(cap))
    if wp.static(reads):
      k_cap = run_cap(geo.rt, geo.bits2)
    if wp.static(dual):
      if k_cap == 0:
        # Every light of the cell: the slots hold the cell (`light_of`), the far field is left.
        top[wp.static(cap)] = cell
        if wp.static(exact_col >= 0):
          counters.add(geo, worldid, wp.static(exact_col), last - first)
        return top, first, last, far  # pyrefly: ignore[bad-return]  # a dual source: cap + 1 slots
    if wp.static(reads):
      # Slots past the run-time cap never take a light (never the weakest).
      for k in range(1, wp.static(cap)):
        if k >= k_cap:
          top_y[k] = 3.4e38
    if last - first <= k_cap:
      # The cap selects nothing: every light of the cell, in its order, as the uncapped source
      # (none dropped by the cheap estimate: an area light's punctual form can rank 0 where its
      # LTC form lights the point; a scene's own build has no cap then).
      n = last - first
      for k in range(wp.static(cap)):
        if k < n:
          top[k] = cell_light(lit, cell, first + k)
      if wp.static(exact_col >= 0):
        counters.add(geo, worldid, wp.static(exact_col), n)
      return top, 0, n, far  # pyrefly: ignore[bad-return]  # a dual source: cap + 1 slots
    total += far
    for j in range(first, last):
      li = cell_light(lit, cell, j)
      profiled = False
      if wp.static(ies):
        profiled = gate(lit.bits, wp.static(toggles.CLASS_IES))
      if profiled:
        # Ranked by the profile's bound, the remainder by its mean (`_make_ies_estimate`).
        rank, e = ies_estimate(lit, worldid, li, point, normal)
        y = color.luminance(rank)
      else:
        e = estimate(lit, worldid, li, point, normal)
        y = color.luminance(e)
      if y > 0.0:
        total += e
        if y > weak_y:
          for k in range(wp.static(cap)):
            if k == weakest:
              top_y[k] = y
              top[k] = li
              top_r[k] = e[0]
              top_g[k] = e[1]
              top_b[k] = e[2]
          weakest = int(0)
          weak_y = top_y[0]
          for k in range(1, wp.static(cap)):
            if top_y[k] < weak_y:
              weakest = k
              weak_y = top_y[k]
    # The occupied prefix: the loop's trip count; the others' irradiance is what is left.
    n = int(0)
    for k in range(wp.static(cap)):
      if top[k] >= 0:
        total -= wp.vec3(top_r[k], top_g[k], top_b[k])
        n += 1
    if wp.static(cheap_col >= 0):
      counters.add(geo, worldid, wp.static(cheap_col), last - first)
      counters.add(geo, worldid, wp.static(exact_col), n)
    return top, 0, n, total  # pyrefly: ignore[bad-return]  # a dual source: cap + 1 slots

  @wp.func
  @device_generic
  @concrete(slots=Ids)
  def light_of(
    # Model:
    lit: Lights,
    # In:
    slots: wp.Vector[int, N],
    j: int,
  ) -> int:
    """Light of slot `j` (picked by constant indices, as `begin`'s; a dual source at cap 0: cell
    entry `j`)."""
    li = slots[0]
    if wp.static(dual):
      if li < 0:
        return cell_light(lit, slots[wp.static(cap)], j)
    for i in range(1, wp.static(cap)):
      if j == i:
        li = slots[i]
    return li

  @wp.func
  @concrete(slots=Ids)
  def is_capped(
    # In:
    slots: wp.Vector[int, N],
  ) -> bool:
    """Whether the slots hold the strongest lights (else the cell's every light, or none)."""
    return slots[0] >= 0

  return Source(
    begin=begin,
    light_of=light_of,
    remainder=lobe_remainder,
    capped=True,
    rest=True,
    dual=dual,
    is_capped=is_capped if dual else None,
  )
