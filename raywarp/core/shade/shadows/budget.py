"""The shadow budget of the light loop: shadow rays for the K strongest casters of a shading point.

`shadows.max_casters` K below the light count: every light is evaluated unshadowed, the K
strongest casters by unshadowed luminance hold slots (filled in place by the loop) and get shadow
rays (`_make_trace_slots`; their radiance is kept from the selection, not evaluated again), the
others are scaled by the traced ones' visibility (the luminance-weighted ratio).

Budget transitions: the set of the K strongest casters changes across a surface (where two lights'
contributions cross, and at light-grid cells), and an untraced caster takes the traced ones'
visibility, so the shading jumped at those borders (polygonal patches on floors). The strongest
untraced caster is traced too, and the result blends from the K + 1 estimate (where it
is as strong as the weakest traced caster) to the K estimate (where it is negligible) by their
relative luminance margin, which both sides of a border share: the image is continuous, at one more
shadow ray wherever an untraced caster lights the point (`shadows.max_casters` = K traces up to
K + 1 rays). Tracing it over the whole margin, rather than within a band of it, removed the third
caster's shadow streaks on a large interior's fast floor (FLIP vs every caster traced
0.071 -> 0.059 for +22 % shadow rays; a 0.25 band before).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, cast

import warp as wp

from raywarp.core.kernels.abi import counters, toggles
from raywarp.core.kernels.abi.rays import GeometryFields
from raywarp.core.kernels.generic import concrete, device_generic, specialize
from raywarp.core.post.grading import color
from raywarp.core.shade.lights import punctual
from raywarp.core.shade.lights.punctual import Lights, Visibility
from raywarp.core.shade.lights.sources import Source, make_entries
from raywarp.core.shade.materials.material import Lobes

if TYPE_CHECKING:
  from raywarp.core.shade.lights.loop import Evaluators, LightVisibilityAt

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# "ratio": the untraced casters' visibility is FLOOR + (1 - FLOOR) x the traced ones' (luminance
# weighted). The traced casters are the strongest, usually the nearest and highest: when an object
# blocks them all (a desk, a rack) the far, low untraced ones mostly still reach the point, and
# the plain ratio (FLOOR 0) turned those points pitch black with the shadow edges of K lights
# (fast's K = 2 on the Isaac Warehouse floor). FLOOR 0.5 (open_plan / many_lights at
# fast, FLIP vs every caster traced: 0.073 / 0.031 vs plain ratio 0.070 / 0.029, no black
# region; unshadowed, FLOOR 1: 0.092 / 0.034).
BUDGET_REMAINDER_FLOOR: Final = 0.5

# fmt: off
type Visible = Callable[[GeometryFields, Lights, int, int, wp.vec3, wp.vec3, wp.vec3, int, int], Visibility]
type TraceSlots[N: int] = Callable[[GeometryFields, Lights, int, wp.Vector[int, N], wp.Vector[float, N],
  wp.Matrix[float, N, Literal[3]], wp.vec3, wp.vec3, wp.vec3, int, int], tuple[wp.vec3, wp.vec3, float]]
type Remainder = Callable[[GeometryFields, Lights, int, wp.vec3, wp.vec3, wp.vec3, int, int,
  wp.vec3, wp.vec3, wp.vec3, wp.vec3, float, int, float, wp.vec3], wp.vec3]
# fmt: on


@specialize
def _slot_types[N: int](
  budget: N,
) -> tuple[type[wp.Vector[int, N]], type[wp.Vector[float, N]], type[wp.Matrix[float, N, Literal[3]]]]:
  """(lights, luminances, radiances) vector / matrix types of the budget's K slots."""
  return (
    wp.types.vector(length=budget, dtype=int),
    wp.types.vector(length=budget, dtype=float),
    wp.types.matrix(shape=(budget, 3), dtype=float),
  )


@specialize  # the budget's factories share one `visible` per visibility (one copy)
def _make_visible(visibility: LightVisibilityAt):
  """visible(geo, lit, worldid, li, point, normal, origin, seed, stream):
  `visibility` of light `li` (an `Evaluators.visibility`) fetched by id (the slots keep ids)."""

  @wp.func(inline=True)
  @device_generic
  def visible[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    li: int,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    seed: int,
    stream: int,
  ):
    """Visibility of light `li` from `point` (rays from `origin`)."""
    return visibility(geo, lit, worldid, punctual.fetch(lit, worldid, li), li, point, normal, origin, seed, stream)

  return visible


def _make_trace_slots[N: int](budget: N, visibility: LightVisibilityAt):
  """trace_slots(geo, lit, worldid, top, top_y, top_c, point, normal, origin, seed, stream) ->
  (traced, lit_traced, weakest_y): the occupied slots' radiance unshadowed / times their visibility,
  and the weakest traced luminance."""
  visible = _make_visible(visibility)
  Ids, Floats, Rgb = _slot_types(budget)

  @wp.func(inline=True)
  @device_generic
  @concrete(top=Ids, top_y=Floats, top_c=Rgb)
  def trace_slots[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    top: wp.Vector[int, N],
    top_y: wp.Vector[float, N],
    top_c: wp.Matrix[float, N, Literal[3]],
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    seed: int,
    stream: int,
  ) -> tuple[wp.vec3, wp.vec3, float]:
    """Shadow rays of the occupied slots."""
    traced = wp.vec3(0.0)
    lit_traced = wp.vec3(0.0)
    weakest_y = float(3.4e38)
    nslot = int(0)
    for k in range(wp.static(budget)):
      if top[k] >= 0:
        nslot += 1
        weakest_y = wp.min(weakest_y, top_y[k])
    # A runtime trip count: a constant one would be unrolled (one shadow-ray call site per slot).
    # The occupied slots are a prefix (filled in order); slot k is picked by constant indices (a
    # dynamically indexed slot vector lives in local memory on the GPU, across the shadow rays).
    for k in range(nslot):
      li = top[0]
      c = top_c[0]
      for i in range(1, wp.static(budget)):
        if k == i:
          li = top[i]
          c = top_c[i]
      traced += c
      lit_traced += punctual.attenuate(c, visible(geo, lit, worldid, li, point, normal, origin, seed, stream))
    return traced, lit_traced, weakest_y

  return trace_slots


@wp.func(inline=True)
def ratio_estimate(
  # In:
  total: wp.vec3,
  casters: wp.vec3,
  traced: wp.vec3,
  lit_traced: wp.vec3,
) -> wp.vec3:
  """The remainder: the untraced casters take BUDGET_REMAINDER_FLOOR + (1 - FLOOR) x the
  traced ones' luminance-weighted visibility (never darker than FLOOR x their unshadowed light)."""
  weight = color.luminance(traced)
  ratio = wp.where(weight > 0.0, color.luminance(lit_traced) / wp.max(weight, 1.0e-30), 1.0)
  visible = wp.static(BUDGET_REMAINDER_FLOOR) + wp.static(1.0 - BUDGET_REMAINDER_FLOOR) * ratio
  return total - casters + lit_traced + (casters - traced) * visible


def _make_ratio_remainder(visibility: LightVisibilityAt):
  """remainder(geo, lit, worldid, point, normal, origin, seed, stream, total, casters, traced,
  lit_traced, weakest_y, next_li, next_y, next_c) -> radiance (the ratio with the margin blend:
  module doc)."""
  visible = _make_visible(visibility)

  @wp.func(inline=True)
  @device_generic
  def remainder[G: GeometryFields](
    # Model:
    geo: G,
    lit: Lights,
    # In:
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    seed: int,
    stream: int,
    total: wp.vec3,
    casters: wp.vec3,
    traced: wp.vec3,
    lit_traced: wp.vec3,
    weakest_y: float,
    next_li: int,
    next_y: float,
    next_c: wp.vec3,
  ) -> wp.vec3:
    """The ratio estimate, blended towards the one with the strongest untraced caster traced by
    their luminance margin (module doc)."""
    shaded = ratio_estimate(total, casters, traced, lit_traced)
    if next_li >= 0 and next_y > 0.0:
      # In [0, 1]: the untraced caster is never stronger than the weakest traced one (clamped).
      margin = wp.clamp((weakest_y - next_y) / wp.max(weakest_y, 1.0e-30), 0.0, 1.0)
      traced_more = traced + next_c
      lit_more = lit_traced + punctual.attenuate(
        next_c, visible(geo, lit, worldid, next_li, point, normal, origin, seed, stream)
      )
      shaded_more = ratio_estimate(total, casters, traced_more, lit_more)
      shaded = shaded_more + (shaded - shaded_more) * margin
    return shaded

  return remainder


@wp.func(inline=True)
def keep_untraced(
  # In:
  li: int,
  y: float,
  c: wp.vec3,
  next_li: int,
  next_y: float,
  next_c: wp.vec3,
) -> tuple[int, float, wp.vec3]:
  """The strongest untraced caster (next_li, next_y, next_c) with caster `li` (luminance `y`,
  radiance `c`) offered."""
  if y > next_y:
    next_li = li
    next_y = y
    next_c = c
  return next_li, next_y, next_c


def make_budget_loop[C, S](
  source: Source[S],
  evaluators: Evaluators[C],
  *,
  budget: int,
  term_column: int = -1,
  run_counts: bool = False,
  run_budget: bool = False,
  headlight: bool = False,
  bound: int = 0,
  switches: int = 0,
):
  """The light loop (`lights.loop.make_light_loop`'s signature) with shadow rays for the `budget`
  strongest casters (module doc); `source`: the lights evaluated exactly
  (`lights.sources.make_source`); `evaluators`: the lights' (`lights.loop.Evaluators`).
  `run_counts`: `budget` is the capacity, the shadows and K are read at run time (`Toggles`).
  `run_budget` (a tier build reading `toggles.RUN_COUNTS`; `switches`: its run-time switches): K
  is `Toggles.budget` where the bit is on and K is below `budget` (0: `budget`, whose slots then
  hold every caster a budget at or above the scene's casters traces).
  `headlight`: the camera headlight as one more entry, `bound` its class bit
  (`lights.loop.make_light_loop`)."""
  visibility = evaluators.visibility
  trace_slots = _make_trace_slots(budget, visibility)
  remainder = _make_ratio_remainder(visibility)
  return _make_light_loop(
    source,
    evaluators,
    budget=budget,
    trace_slots=trace_slots,
    remainder=remainder,
    term_column=term_column,
    run_counts=run_counts,
    run_budget=run_budget,
    headlight=headlight,
    bound=bound,
    switches=switches,
  )


@wp.func
def visible_share(
  # In:
  shaded: wp.vec3,
  unshadowed: wp.vec3,
) -> float:
  """Luminance share of the evaluated lights' light that reaches the point (1 without any): the
  visibility an uncapped loop's light-grid far field takes, as the ratio estimator's untraced
  casters do (the share of the full BSDF's light, applied to the far field's direction-less
  remainder: an approximation for at most `lights.grid.FAR_FRACTION` of the light)."""
  y = color.luminance(unshadowed)
  return wp.where(y > 0.0, wp.clamp(color.luminance(shaded) / wp.max(y, 1.0e-30), 0.0, 1.0), 1.0)


def _make_light_loop[C, S, N: int](  # noqa: C901, PLR0915  # one device loop: its compile-time branches (counted, sources)
  source: Source[S],
  evaluators: Evaluators[C],
  *,
  budget: N,
  trace_slots: TraceSlots[N],
  remainder: Remainder,
  term_column: int,
  run_counts: bool = False,
  run_budget: bool = False,
  headlight: bool = False,
  bound: int = 0,
  switches: int = 0,
):
  """`make_budget_loop`'s loop over the source's lights: the slot selection, then `trace_slots`
  and `remainder` (`run_counts` / `run_budget` / `headlight` / `bound` / `switches`:
  `make_budget_loop`)."""
  context, unshadowed, visibility = evaluators.context, evaluators.unshadowed, evaluators.visibility
  begin, light_of, remainder_of, far_on = source.begin, source.light_of, source.remainder, source.far
  is_capped = cast("Callable[..., bool]", source.is_capped)  # read only by a dual source
  Ids, Floats, Rgb = _slot_types(budget)
  entries, is_head, light_at, with_head = make_entries(headlight=headlight, bound=bound)
  setting = toggles.setting(switches, bound)  # `toggles.RUN_COUNTS` (`run_budget`)

  @wp.func
  @device_generic
  def light_loop[G: GeometryFields](  # noqa: C901, PLR0915  # slots indexed by constants only (unrolled), compile-time branches
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
    _sun: float,
    head_dir: wp.vec3,
    head: wp.vec3,
  ) -> wp.vec3:
    """Radiance from every light the point sees, shadow rays for the strongest casters only
    (camera rays: every caster traced or taking the traced ones' visibility, `_sun` unused)."""
    ctx = context(lit, lobes, normal, view, nov)
    # The K slots: caster, its unshadowed luminance and radiance (kept: not evaluated again).
    top = Ids(-1)
    # Every slot starts empty at luminance 0: strict comparisons pick the first, and a caster of
    # no light (a zero BSDF towards it) never takes a slot and its shadow ray.
    top_y = Floats(0.0)
    top_c = Rgb()
    weakest = int(0)
    weak_y = float(0.0)  # top_y[weakest]
    total = wp.vec3(0.0)
    casters = wp.vec3(0.0)
    head_c = wp.vec3(0.0)  # the headlight's term, added last (`lights.loop.make_light_loop`)
    # The strongest untraced caster (the blend; the lights the slots reject or evict are exactly
    # the untraced ones). Another build's loop has none of the counted build's run-time state
    # below (its code is the loop's without the switches: with them, a sun-lit room's fast frame
    # was 12 % slower on the RTX PRO 6000).
    next_li = int(-1)
    next_y = float(-1.0)
    next_c = wp.vec3(0.0)
    if wp.static(run_counts):
      shadowed = (geo.rt.flags & wp.static(toggles.SHADOWS)) != 0
      every = shadowed and geo.rt.budget == 0  # every caster traced in turn (`budget` 0), no slots
      unshadowed_total = wp.vec3(0.0)
      # The settings' K: the slots past it never take a caster (never the weakest).
      for k in range(1, wp.static(budget)):
        if k >= geo.rt.budget:
          top_y[k] = 3.4e38
    else:
      if wp.static(run_budget):
        # The settings' K below the compiled one: the slots past it never take a caster.
        if setting(geo.rt.flags, geo.bits2, wp.static(toggles.RUN_COUNTS)):
          k_run = geo.rt.budget
          for k in range(1, wp.static(budget)):
            if k_run > 0 and k >= k_run:
              top_y[k] = 3.4e38
    slots, first, last, rest = begin(geo, lit, worldid, point, normal)
    for j in range(first, last + entries(lit, head)):
      li = wp.static(punctual.HEADLIGHT)
      if not is_head(lit, j, last):
        li = light_of(lit, slots, j)
      c, caster = unshadowed(lit, light_at(lit, worldid, li, head_dir, head), lobes, ctx, point, normal, view, nov)
      if is_head(lit, j, last):
        head_c = c  # no shadow ray, no slot
        continue
      if wp.static(term_column >= 0):
        counters.term(geo, worldid, wp.static(term_column), c, caster)
      if wp.static(run_counts):
        if every:  # pyrefly: ignore[unbound-name]  # bound under `wp.static(run_counts)`
          unshadowed_total += c  # pyrefly: ignore[unbound-name]  # bound under `wp.static(run_counts)`
          if caster and wp.length_sq(c) > 0.0:
            v = visibility(geo, lit, worldid, punctual.fetch(lit, worldid, li), li, point, normal, origin, seed, stream)
            c = punctual.attenuate(c, v)
        caster = caster and shadowed and not every  # pyrefly: ignore[unbound-name]  # a slot caster (traced later)
      total += c
      if caster:
        casters += c
        y = color.luminance(c)
        # The slots are searched and filled in place: passing them through a function per light
        # copied them (dynamically indexed: local memory on the GPU), many_lights software -29 %
        # on the RTX PRO 6000. They are indexed by constants only (unrolled loops over the K
        # slots): a dynamic index also puts them in local memory.
        # The light that stays untraced: this one, or the slot's caster it evicts.
        out_li = li
        out_y = y
        out_c = c
        if y > weak_y:
          for k in range(wp.static(budget)):
            if k == weakest:
              out_li = top[k]
              out_y = top_y[k]
              out_c = top_c[k]
              top_y[k] = y
              top[k] = li
              top_c[k] = c
          # Only a replacement changes the minimum: searched again from slot 0 (ties: the first
          # slot); a rejected caster leaves the weakest slot as it was.
          weakest = int(0)
          weak_y = top_y[0]
          for k in range(1, wp.static(budget)):
            if top_y[k] < weak_y:
              weakest = k
              weak_y = top_y[k]
        if out_li >= 0:
          next_li, next_y, next_c = keep_untraced(out_li, out_y, out_c, next_li, next_y, next_c)
    # The lights past the cap or the cell's far field (unshadowed terms), before the shadow rays:
    # no lobes live across them.
    far = wp.vec3(0.0)
    if wp.static(source.dual or source.capped or source.rest):
      far = remainder_of(rest, lobes)
    if wp.static(run_counts):
      if every:  # pyrefly: ignore[unbound-name]  # bound under `wp.static(run_counts)`
        # Every caster traced in turn (the shadowed loop's result).
        if wp.static(source.dual):
          if is_capped(slots):
            return with_head(lit, total + far, head_c)
        if wp.static(source.capped and not source.dual):
          return with_head(lit, total + far, head_c)
        if wp.static(source.rest):
          return with_head(lit, total + far * visible_share(total, unshadowed_total), head_c)  # pyrefly: ignore[unbound-name]  # bound under `wp.static(run_counts)`
        return with_head(lit, total, head_c)
      if not shadowed:  # pyrefly: ignore[unbound-name]  # bound under `wp.static(run_counts)`
        # Shadows off (`toggles.SHADOWS`): every light unshadowed, as the unshadowed loop.
        if wp.static(source.rest):
          return with_head(lit, total + far, head_c)
        return with_head(lit, total, head_c)
    traced, lit_traced, weakest_y = trace_slots(geo, lit, worldid, top, top_y, top_c, point, normal, origin, seed, stream)
    # The shading point, the slots' totals, the strongest untraced caster.
    shaded = remainder(
      geo, lit, worldid, point, normal, origin, seed, stream,
      total, casters, traced, lit_traced, weakest_y,
      next_li, next_y, next_c,
    )  # fmt: skip
    if wp.static(source.dual):
      if is_capped(slots):
        shaded += far
      else:
        shaded += far * visible_share(shaded, total)
    elif wp.static(source.capped):
      # Unshadowed, as the capped-out lights' estimate always was (their shadows are soft: a
      # visibility share of the strongest casters' made hard, dark shadows).
      shaded += far
    elif wp.static(source.rest):
      # Uncapped: the far field alone (a few % of the light) takes the evaluated lights' mean
      # visibility (it does not light the inside of a shadow the near lights cast).
      if far_on(lit):
        shaded += far * visible_share(shaded, total)
    return with_head(lit, shaded, head_c)

  return light_loop
