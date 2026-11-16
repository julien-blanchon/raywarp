"""The light loop of a shading point: one loop for every light form, with the shadow budget.

`make_light_loop` wires a per-light evaluator pair, `unshadowed` (the light's radiance at the point
without visibility, and whether it casts shadows) and `visibility` (its shadow rays), into the
loop over the lights the point sees (all of them, or the light grid cell's list with culling):

* no shadows: the sum of the unshadowed terms;
* shadows without a budget: each caster's term times its visibility;
* `sun_only` (reflection hits with reflections.shadows="sun"): shadow rays for directional
  lights only, the other lights unshadowed;
* untraced directional casters times the loop's `sun` input: their visibility as the caller
  estimates it without a ray (reflection hits without sun shadow rays: the GI's sky visibility,
  `reflections.sun_visibility`; 1 elsewhere);
* a budget of K shadow rays (`shadows.max_casters` below the light count): the K strongest casters
  by unshadowed luminance are traced, the others scaled by the traced ones' luminance-weighted
  visibility.

The factories live in `evaluators.py`: punctual lights (`make_punctual`, with BRDF and
shadow rays) and area lights (`make_area_evaluators`: LTC shapes and punctual lights in one
table). The ordinary loop reads each light once per point (`punctual.fetch`: one packed struct)
and hands it to both evaluators. The budget variant (`shadows/budget.py`) keeps
the traced casters' unshadowed terms from selection and smooths its transitions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import warp as wp

from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi.features import SMALL_LIGHTS_MAX
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.shade.lights import area, punctual
from raywarp.core.shade.lights.punctual import Light, Lights, Visibility
from raywarp.core.shade.lights.sources import Begin, Source, make_entries
from raywarp.core.shade.materials.material import Lobes
from raywarp.core.shade.shadows import budget as shadow_budget

if TYPE_CHECKING:
  from raywarp.core.shade.shadows.visibility import LightVisibility

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


# Producer contracts shared by camera, reflection and shadow-budget light loops.
# fmt: off
type LightLoop = Callable[[GeometryFields, Lights, int, Lobes, wp.vec3, wp.vec3, wp.vec3, wp.vec3,
  float, int, int, float, wp.vec3, wp.vec3], wp.vec3]
type Context[C] = Callable[[Lights, Lobes, wp.vec3, wp.vec3, float], C]
type Unshadowed[C] = Callable[[Lights, Light, Lobes, C, wp.vec3, wp.vec3, wp.vec3, float], tuple[wp.vec3, bool]]
type PreparedTerm[C] = Callable[[Lights, Light, Lobes, C, wp.vec3, wp.vec3, wp.vec3, float],
  tuple[wp.vec3, bool, wp.vec3, float]]
type LightVisibilityAt = Callable[[GeometryFields, Lights, int, Light, int, wp.vec3, wp.vec3, wp.vec3,
  int, int], Visibility]
type AreaVisibility = Callable[[GeometryFields, int, area.AreaLight, wp.vec3, int, int, int, int], float]
# fmt: on


@final
class Prepared[C](NamedTuple):
  """A punctual evaluator's prepared term: `term` -> (unshadowed radiance, casts shadows, direction
  to the light, distance), and `visibility`, the shading's shadow rays taking that direction and
  distance (the same rays and random streams as `Evaluators.visibility`, without evaluating the
  light again). Only the uncapped and small loops use it; area evaluators have none."""

  term: PreparedTerm[C]
  visibility: LightVisibility[Visibility]


def _require_prepared[C](prepared: Prepared[C] | None) -> Prepared[C]:
  """`prepared`, known present (read only under a `wp.static(prepared is not None)` branch)."""
  if prepared is None:
    raise ValueError("the selected evaluator has no prepared punctual term")
  return prepared


@final
class Evaluators[C](NamedTuple):
  """The per-light evaluators of a light loop, their per-point context and (punctual lights) the
  prepared term.

  context(lit, lobes, normal, view, nov) -> C: per-point state shared by the lights (the LTC
  frame; punctual lights: none); unshadowed(lit, light, lobes, ctx, point, normal, view, nov) ->
  (radiance without visibility, casts shadows) of a fetched `punctual.Light`; visibility(geo, lit,
  worldid, light, li, point, normal, origin, seed, stream) -> its visibility (float, or RGB
  with tinted shadows) from shading point `point`, the rays leaving from `origin`
  (`Surface.origin`); `li` and the table's count `lit.nlight` key its random streams.
  """

  context: Context[C]
  unshadowed: Unshadowed[C]
  visibility: LightVisibilityAt
  context_type: type[C]
  prepared: Prepared[C] | None = None


@final
class Small(NamedTuple):
  """The small form of a light loop (`make_light_loop`): every light of the table (`source`:
  `sources.make_source` uncapped, without the grid), `size` <= SMALL_LIGHTS_MAX straight-line
  entries, each taken when the table holds it; `lights_column`: the counter of the lights evaluated
  (-1: none), counted once per loop (`source` itself counts nothing: it is read per entry)."""

  source: Source[int]
  size: int
  lights_column: int = -1


def make_light_loop[C, S](
  *,
  source: Source[S],
  shadows: bool,
  budget: int,
  sun_only: bool,
  evaluators: Evaluators[C],
  term_column: int = -1,
  run_counts: bool = False,
  run_budget: bool = False,
  hit: bool = False,
  headlight: bool = False,
  small: Small | None = None,
  bound: int = 0,
  switches: int = 0,
):
  """light_loop(geo, lit, worldid, lobes, point, normal, origin, view, nov, seed, stream, sun,
  head_dir, head) -> radiance.

  `origin`: the shading point's secondary-ray origin (`Surface.origin`), where shadow rays start;
  `sun`: the visibility of the directional casters the loop traces no ray for (module doc);
  `head_dir` / `head`: the camera headlight's direction and radiance scale (zero: off), read
  with `headlight` only.

  `source`: the lights evaluated exactly and the capped-out rest (`sources.make_source`);
  `shadows`: shadow rays at all; `budget`: K > 0 traced casters (below the light count), 0 every
  caster; `sun_only`: see the module doc (no budget);
  `evaluators`: the lights' (`Evaluators`); `term_column`: optional term/caster columns
  (-1 compiles counting out), before visibility and budget selection.

  `small` (`Small`, shadowed camera-ray loops: `kernels/lighting._camera_loop`): a table of at
  most `small.size` lights and no light grid takes the straight-line exact loop over them
  (`_make_small_loop`; the budget, the cap and the grid select nothing: every light is traced).

  `run_counts` (a counted tier build, `Features.run_counts`): `shadows` / `budget` / `sun_only` are
  what it compiles (`budget`: the capacity) and the loop reads the shadows, the traced casters, the
  remainder and (`hit`: the glossy-reflection hits' loop, without a budget) the reflection hits'
  shadow rays from `Toggles` at run time. `run_budget` (a tier build's camera rays, `switches` its
  run-time switches): K is `Toggles.budget` within the compiled `budget` where
  `toggles.RUN_COUNTS` is on (`shadow_budget.make_budget_loop`).

  `headlight`: the loop evaluates the camera headlight as one more entry after the cell's lights
  (`punctual.light_at`; no shadow ray), so that the light evaluation is compiled once (an inlined
  term beside the loop cost a translucent room 10 % unused); its term is added after the remainder
  (it takes no part in the visibility share). `bound` (Features.bound): the headlight's entry reads
  the build word's bit (`sources.make_entries`).
  """
  context, unshadowed = evaluators.context, evaluators.unshadowed
  if not shadows:
    return _make_unshadowed_loop(source, context, unshadowed, term_column=term_column, headlight=headlight, bound=bound)
  if small is not None:
    if run_counts or hit or small.source.rest or not 0 < small.size <= SMALL_LIGHTS_MAX:
      raise ValueError("the small form takes every light of the table (no cap, no grid), at most SMALL_LIGHTS_MAX")
    loop = _make_small_loop(small, evaluators, term_column=term_column, headlight=headlight, bound=bound)
    return loop if small.lights_column < 0 else _counted(loop, small.source.begin, small.lights_column)
  if budget <= 0 or hit:
    return _make_shadowed_loop(
      source,
      evaluators,
      sun_only=sun_only,
      term_column=term_column,
      run_counts=run_counts,
      hit=hit,
      headlight=headlight,
      bound=bound,
      switches=switches,
    )
  return shadow_budget.make_budget_loop(
    source,
    evaluators,
    budget=budget,
    term_column=term_column,
    run_counts=run_counts,
    run_budget=run_budget,
    headlight=headlight,
    bound=bound,
    switches=switches,
  )


def _make_unshadowed_loop[C, S](
  source: Source[S], context: Context[C], unshadowed: Unshadowed[C], *, term_column: int, headlight: bool, bound: int
):
  """The light loop without shadow rays (`headlight`, `bound`: `make_light_loop`)."""
  begin, light_of, remainder, far = source.begin, source.light_of, source.remainder, source.far
  entries, is_head, light_at, with_head = make_entries(headlight=headlight, bound=bound)

  @wp.func
  @device_generic
  def light_loop[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    _origin: wp.vec3,
    view: wp.vec3,
    nov: float,
    _seed: int,
    _stream: int,
    sun: float,
    head_dir: wp.vec3,
    head: wp.vec3,
  ) -> wp.vec3:
    """Radiance from every light the point sees, unshadowed (directional casters times `sun`)."""
    ctx = context(lit, lobes, normal, view, nov)
    total = wp.vec3(0.0)
    head_c = wp.vec3(0.0)
    slots, first, last, rest = begin(geo, lit, worldid, point, normal)
    for j in range(first, last + entries(lit, head)):
      li = wp.static(punctual.HEADLIGHT)
      if not is_head(lit, j, last):
        li = light_of(lit, slots, j)
      light = light_at(lit, worldid, li, head_dir, head)
      c, caster = unshadowed(lit, light, lobes, ctx, point, normal, view, nov)
      if is_head(lit, j, last):
        head_c = c  # added last: `make_light_loop`
        continue
      if wp.static(term_column >= 0):
        counters.term(geo, worldid, wp.static(term_column), c, caster)
      if caster and light.type == wp.static(punctual.LIGHT_DIRECTIONAL):
        c = c * sun
      total += c
    if wp.static(source.rest):
      if far(lit):
        total += remainder(rest, lobes)
    return with_head(lit, total, head_c)

  return light_loop


@specialize
def _make_light_term[C](
  evaluators: Evaluators[C],
  *,
  prepared: bool,
  sun_only: bool,
  term_column: int,
  run_counts: bool = False,
  hit: bool = False,
  headlight: bool = False,
  bound: int = 0,
  switches: int = 0,
):
  """light_term(geo, lit, worldid, lobes, ctx, point, normal, origin, view, nov, seed, stream,
  sun, head_dir, head, li) -> (shadowed, unshadowed) term of entry `li` with its shadow ray (set)
  (`prepared`: the evaluators' prepared punctual term, an uncapped source's; the rest:
  `_make_shadowed_loop`)."""
  unshadowed, visibility = evaluators.unshadowed, evaluators.visibility
  ready = evaluators.prepared if prepared else None
  light_at = make_entries(headlight=headlight, bound=bound).light_at
  setting = toggles.setting(switches, bound)  # the reflection hits' shadow rays (`sun_only`)

  @wp.func(inline=True)
  @device_generic
  @concrete(ctx=evaluators.context_type)
  def light_term[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    lobes: Lobes,
    ctx: C,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
    sun: float,
    head_dir: wp.vec3,
    head: wp.vec3,
    li: int,
  ) -> tuple[wp.vec3, wp.vec3]:
    """(shadowed, unshadowed) term of entry `li` (an untraced directional caster's both times
    `sun`)."""
    light = light_at(lit, worldid, li, head_dir, head)
    direction, distance = wp.vec3(0.0), float(0.0)
    if wp.static(ready is not None):
      c, caster, direction, distance = wp.static(_require_prepared(ready).term)(
        lit, light, lobes, ctx, point, normal, view, nov
      )
    else:
      c, caster = unshadowed(lit, light, lobes, ctx, point, normal, view, nov)
    if wp.static(term_column >= 0):
      if li != wp.static(punctual.HEADLIGHT):
        counters.term(geo, worldid, wp.static(term_column), c, caster)
    traced = caster and wp.length_sq(c) > 0.0
    if wp.static(run_counts and hit):
      # The reflection hits' shadow rays: off or directional lights.
      traced = traced and geo.rt.reflection_shadows != 0 and light.type == wp.static(punctual.LIGHT_DIRECTIONAL)
    elif wp.static(run_counts):
      traced = traced and (geo.rt.flags & wp.static(toggles.SHADOWS)) != 0
    elif wp.static(sun_only):
      traced = traced and light.type == wp.static(punctual.LIGHT_DIRECTIONAL)
      traced = traced and setting(geo.rt.flags, geo.bits2, wp.static(toggles.REFLECTION_SHADOWS))
    if traced:
      if wp.static(ready is not None):
        v = wp.static(_require_prepared(ready).visibility)(
          geo, worldid, light, origin, normal, direction, distance, seed, stream, lit.nlight, li
        )
      else:
        v = visibility(geo, lit, worldid, light, li, point, normal, origin, seed, stream)
      return punctual.attenuate(c, v), c
    if caster and light.type == wp.static(punctual.LIGHT_DIRECTIONAL):
      c = c * sun
    return c, c

  return light_term


def _make_shadowed_loop[C, S](
  source: Source[S],
  evaluators: Evaluators[C],
  *,
  sun_only: bool,
  term_column: int,
  run_counts: bool = False,
  hit: bool = False,
  headlight: bool = False,
  bound: int = 0,
  switches: int = 0,
):
  """The light loop with a shadow ray (set) for every caster (`sun_only`: directional ones only;
  `run_counts` / `hit` / `headlight` / `bound` / `switches`: `make_light_loop`)."""
  context = evaluators.context
  begin, light_of, remainder, far = source.begin, source.light_of, source.remainder, source.far
  is_capped = cast("Callable[..., bool]", source.is_capped)  # read only by a dual source
  entries, is_head, _light_at, with_head = make_entries(headlight=headlight, bound=bound)
  light_term = _make_light_term(
    evaluators,
    prepared=not source.capped,
    sun_only=sun_only,
    term_column=term_column,
    run_counts=run_counts,
    hit=hit,
    headlight=headlight,
    bound=bound,
    switches=switches,
  )

  @wp.func
  @device_generic
  def light_loop[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
    sun: float,
    head_dir: wp.vec3,
    head: wp.vec3,
  ) -> wp.vec3:
    """Radiance from every light the point sees, the casters times their visibility."""
    ctx = context(lit, lobes, normal, view, nov)
    total = wp.vec3(0.0)
    slots, first, last, rest = begin(geo, lit, worldid, point, normal)
    unshadowed_total = wp.vec3(0.0)
    head_c = wp.vec3(0.0)
    for j in range(first, last + entries(lit, head)):
      li = wp.static(punctual.HEADLIGHT)
      if not is_head(lit, j, last):
        li = light_of(lit, slots, j)
      c, c0 = light_term(geo, lit, worldid, lobes, ctx, point, normal, origin, view, nov, seed, stream, sun, head_dir, head, li)
      if is_head(lit, j, last):
        head_c = c  # added last: `make_light_loop`
        continue
      total += c
      unshadowed_total += c0
    if wp.static(source.dual):
      if is_capped(slots):
        total += remainder(rest, lobes)
      else:
        total += remainder(rest, lobes) * shadow_budget.visible_share(total, unshadowed_total)
    elif wp.static(source.capped):
      # Unshadowed (the capped-out lights' shadows are soft: `shadow_budget`'s loop).
      total += remainder(rest, lobes)
    elif wp.static(source.rest):
      # Uncapped: the far field alone takes the evaluated lights' mean visibility.
      if far(lit):
        total += remainder(rest, lobes) * shadow_budget.visible_share(total, unshadowed_total)
    return with_head(lit, total, head_c)

  return light_loop


# A small-form entry with no light (past the table's count, the headlight off;
# `punctual.HEADLIGHT` is -1).
_NO_LIGHT: Final = -2


def _entry_evaluators[C](
  evaluators: Evaluators[C],
):
  """(entry_term, entry_visibility) of the small form's two passes (`_make_small_loop`): a light's
  unshadowed term with its direction and distance (a prepared punctual term's, else unused) and
  its visibility from them."""
  prepared = evaluators.prepared

  @wp.func(inline=True)
  @concrete(ctx=evaluators.context_type)
  def entry_term(
    # Model:
    lit: Lights,
    # In:
    light: Light,
    lobes: Lobes,
    ctx: C,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
  ) -> tuple[wp.vec3, bool, wp.vec3, float]:
    """(unshadowed radiance, casts shadows, direction, distance) of `light` (the direction and
    distance: a prepared punctual term's, else unused)."""
    if wp.static(prepared is not None):
      return wp.static(_require_prepared(prepared).term)(lit, light, lobes, ctx, point, normal, view, nov)
    else:
      c, caster = evaluators.unshadowed(lit, light, lobes, ctx, point, normal, view, nov)
      return c, caster, wp.vec3(0.0), 0.0

  @wp.func(inline=True)
  @device_generic
  def entry_visibility[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    light: Light,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
    distance: float,
    seed: int,
    stream: int,
  ):
    """The visibility of `light` (entry light `li`) from `point`, its rays from `origin`."""
    if wp.static(prepared is not None):
      return wp.static(_require_prepared(prepared).visibility)(
        geo, worldid, light, origin, normal, direction, distance, seed, stream, lit.nlight, li
      )
    else:
      return evaluators.visibility(geo, lit, worldid, light, li, point, normal, origin, seed, stream)

  return entry_term, entry_visibility


def _make_small_loop[C](small: Small, evaluators: Evaluators[C], *, term_column: int, headlight: bool, bound: int):
  """The light loop of `small`: its entries unrolled, the headlight as one more
  (`make_light_loop`). In two passes: every entry's unshadowed term first, then the shadow rays
  (a site each, no loop state across them). Only the terms (and a punctual light's direction and
  distance, `Evaluators.prepared`) are live across the rays, not the lobes (the BSDF inputs:
  every lobe compiled in, e.g. EON's view factors in a tier build) they were evaluated with; each
  traced light is fetched again for its rays. The same operations in the same order as one pass."""
  context, begin, light_of = evaluators.context, small.source.begin, small.source.light_of
  entries, _is_head, light_at, _with_head = make_entries(headlight=headlight, bound=bound)
  # The entries: the table's, then the headlight's.
  n = small.size + (1 if headlight else 0)
  Rgb = wp.types.matrix(shape=(n, 3), dtype=float)
  Floats = wp.types.vector(length=n, dtype=float)

  @wp.func(inline=True)
  @device_generic
  def entry_light[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
    head: wp.vec3,
    j: int,
  ) -> int:
    """Entry `j`'s light id: the table's j-th light below its count, the headlight's entry past
    them (`punctual.HEADLIGHT`); _NO_LIGHT: none."""
    slots, _first, last, _rest = begin(geo, lit, worldid, point, normal)
    if j < wp.static(small.size):
      if j < last:
        return light_of(lit, slots, j)
      return wp.static(_NO_LIGHT)
    if entries(lit, head) > 0:
      return wp.static(punctual.HEADLIGHT)
    return wp.static(_NO_LIGHT)

  entry_term, entry_visibility = _entry_evaluators(evaluators)

  @wp.func
  @device_generic
  def light_loop[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
    sun: float,
    head_dir: wp.vec3,
    head: wp.vec3,
  ) -> wp.vec3:
    """Radiance from every light of the table, the casters times their visibility."""
    ctx = context(lit, lobes, normal, view, nov)
    terms = Rgb()
    dirs = Rgb()
    dists = Floats()
    traced = int(0)  # bit j: entry j's shadow ray(s) are traced
    # A literal bound: straight-line entries (a constant table count, a scene's own build, folds
    # the tests); each fetches its world's live light row.
    for j in range(wp.static(n)):
      li = entry_light(geo, lit, worldid, point, normal, head, j)
      if li != wp.static(_NO_LIGHT):
        light = light_at(lit, worldid, li, head_dir, head)
        c, caster, direction, distance = entry_term(lit, light, lobes, ctx, point, normal, view, nov)
        if wp.static(term_column >= 0):
          if li != wp.static(punctual.HEADLIGHT):
            counters.term(geo, worldid, wp.static(term_column), c, caster)
        if caster and wp.length_sq(c) > 0.0:
          traced = traced | (1 << j)
          dirs[j] = direction
          dists[j] = distance
        elif caster and light.type == wp.static(punctual.LIGHT_DIRECTIONAL):
          c = c * sun
        terms[j] = c
    total = wp.vec3(0.0)
    for j in range(wp.static(n)):
      li = entry_light(geo, lit, worldid, point, normal, head, j)
      if li != wp.static(_NO_LIGHT):
        c = terms[j]
        if (traced & (1 << j)) != 0:
          light = light_at(lit, worldid, li, head_dir, head)
          v = entry_visibility(geo, lit, worldid, light, li, point, normal, origin, dirs[j], dists[j], seed, stream)
          c = punctual.attenuate(c, v)
        total += c
    return total

  return light_loop


def _counted(light_loop: LightLoop, begin: Begin[int], column: int):
  """`light_loop` counting the lights it evaluates in counter `column`, once per call (a build
  without counters takes `light_loop` itself)."""

  @wp.func
  @device_generic
  def counted_loop[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
    sun: float,
    head_dir: wp.vec3,
    head: wp.vec3,
  ) -> wp.vec3:
    """`light_loop`, after counting the table's lights at the point."""
    _slots, first, last, _rest = begin(geo, lit, worldid, point, normal)
    counters.add(geo, worldid, wp.static(column), last - first)
    return light_loop(geo, lit, worldid, lobes, point, normal, origin, view, nov, seed, stream, sun, head_dir, head)

  return counted_loop
