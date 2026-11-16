"""Which ray-tracing calls a render kernel makes (closest hit, shadow tests), picked once per
`Features` from a trace backend's `Tracer`, counted when on."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast, final

from raywarp.core.kernels.abi import counters
from raywarp.core.kernels.abi.features import Features
from raywarp.core.kernels.abi.rays import BodyAny, BodyClosest, BodyLeave, BodyShadow, Tracer, widen
from raywarp.core.kernels.generic import specialize

if TYPE_CHECKING:
  from raywarp.core.shade.lights.loop import LightVisibilityAt


@final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Queries:
  """Ray-query funcs of a render body (each `(geo, worldid, origin, direction, max_dist)`).

  Attributes:
    closest: closest hit (`Hit`), back faces culled as the camera rays need.
    any_hit: whether anything blocks the ray (shadow rays).
    shadow: a shadow ray's RGB transmittance through translucent casters (0: blocked; one
      terminating walk, `Tracer.shadow`); None: shadow rays are `any_hit` queries (translucent
      shadows off).
    light_visibility: the light loops' visibility of one light (`light_loop.Evaluators.visibility`),
      a call of a program of its own (`trace/optix/device/programs.light_visibility_program`); None:
      the evaluators' own, inline.
    leave: the exit face of a primitive a camera ray travels inside (`Tracer.leave`); None: the
      closest queries report it.
  """

  closest: BodyClosest
  any_hit: BodyAny
  shadow: BodyShadow | None = None
  light_visibility: LightVisibilityAt | None = None
  leave: BodyLeave | None = None


def with_visibility(q: Queries, light_visibility: LightVisibilityAt | None) -> Queries:
  """`q` whose light loops take `light_visibility` (`Queries.light_visibility`; None: `q`)."""
  return q if light_visibility is None else dataclasses.replace(q, light_visibility=light_visibility)


@specialize
def make_queries(f: Features, tracer: Tracer):
  """The queries of `tracer` for `f` (both backends' queries alpha-test their candidates
  themselves, `Features.alpha_cutout`). With `f.counters` every query counts its calls
  (`counters.counted`).
  """
  closest, any_hit = widen(tracer.closest), widen(tracer.any_hit)
  shadow = None if tracer.shadow is None else widen(tracer.shadow)
  if f.counters:
    closest = counters.counted(closest, counters.CLOSEST)
    any_hit = counters.counted(any_hit, counters.SHADOW)
    shadow = None if shadow is None else counters.counted(shadow, counters.SHADOW)
  # The one cast between the backend's geometry and the generic view, as `widen`'s.
  leave = None if tracer.leave is None else cast("BodyLeave", tracer.leave)
  return Queries(closest=closest, any_hit=any_hit, shadow=shadow, leave=leave)
