"""The settings one change away from a configuration: every compile-time settings field over its
domain (each choice, the integers at the values around each threshold the code compares them to),
plus the run-time fields that change what the render kernels store (`exposure.mode`,
`sensor.aberration`: the deferred display's HDR output, an output word bit of the OptiX modules),
plus the output sets a caller switches between (`outputs`: a viewer's output views; each set is an
output word of the OptiX modules and of the software megakernels, so it is one change away like a
field).

`Renderer.warm(nearby=...)` creates the OptiX modules of these neighbours ahead, or on the
software traversal loads their Warp modules (`core/pipeline/warm.py`), and the zero-compile walk of
the tooling (`tools/prebuilt/zero_space.py`) routes the same space from the default, the fast field
set and unlit shading: one domain for both.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Sequence
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
  from raywarp.core.build.variants.resolve import Resolution
  from raywarp.settings import Outputs, RenderSettings

# Light counts and caps around the thresholds the resolution compares them to (the small loop's
# `SMALL_LIGHTS_MAX`, the light grid's `resolve.GRID_MIN_LIGHTS`, the default's caps and budgets,
# the counted tiers' capacity 16).
_COUNTS: Final = (None, 1, 2, 3, 4, 5, 8, 16)
# Settings field -> its values.
DOMAIN: Final[dict[str, tuple[object, ...]]] = {
  "aa.spp": (1, 2, 3),
  "shadows.enabled": (True, False),
  "shadows.samples": (1, 2, 3, 4, 8),
  "shadows.max_casters": _COUNTS,
  "shadows.translucent": ("off", "transmittance", "clear"),
  "lighting.area_lights": ("point", "ltc"),
  "lighting.light_grid": ("auto", True, False),
  "lighting.camera_light": ("auto", True, False),
  "lighting.emissive_lights": (True, False),
  "lighting.max_exact": _COUNTS,
  "environment.enabled": ("auto", True, False),
  "environment.background": (True, False),
  "reflections.enabled": (True, False),
  "reflections.shadows": ("off", "sun"),
  "reflections.max_exact": (None, 1, 2, 4, 8, 16),
  "transparency.enabled": ("auto", True, False),
  "gi.enabled": (True, False),
  "openpbr.diffuse_roughness": ("auto", True, False),
  "openpbr.layers": ("auto", False),
  "surfaces.textures": ("auto", True, False),
  "surfaces.backface_culling": (True, False),
  "surfaces.anisotropy": (1, 2, 4, 8, 16),
  "surfaces.shading": ("pbr", "unlit"),
  "exposure.mode": ("auto", "manual"),
  "grading.enabled": (True, False),
  "sensor.aberration": (0.0, 0.01),
}
# Compile-time fields outside the domain: `performance.kernels="scene"` compiles by definition, the
# trace backend is an axis of its own; the outputs are the caller's (`Outputs`).
EXCLUDED: Final = frozenset({"performance.kernels", "performance.traversal"})


def changes(settings: RenderSettings, outputs: Sequence[Outputs] = ()) -> list[tuple[str, object, RenderSettings]]:
  """The settings one change from `settings`: ("outputs", names, settings) for each output set of
  `outputs` but the current one, first, then (field, value, settings) for every value of `DOMAIN`
  but the field's current one, in `DOMAIN`'s order (a combination the settings refuse,
  ValueError, left out)."""
  flat = settings.flat()
  out: list[tuple[str, object, RenderSettings]] = [
    ("outputs", ",".join(sorted(o.kinds)), settings.replace(outputs=o)) for o in dict.fromkeys(outputs) if o != settings.outputs
  ]
  for field, values in DOMAIN.items():
    for value in values:
      if value == flat[field] and type(value) is type(flat[field]):
        continue
      try:
        out.append((field, value, settings.replace(**{field: value})))
      except ValueError:
        continue
  return out


# Modules `Renderer.warm(nearby=...)` creates at most by default (each 2-12 s of OptiX's
# compile, `ESTIMATE_S`), and at once (each one OptiX task graph on every core).
LIMIT: Final = 64
WORKERS: Final = 8
# Seconds OptiX takes to create one module of a tier family (RTX PRO 6000, 192 cores, measured per
# specialised module: 1.5-2.7 s on the fast field set, 10-12 s on the default; the unlit and
# fast tiers are the light family, every other tier the full one); "warp": one software
# megakernel's Warp module compiled (7-10 s each on cpu-upgrade, an output view's).
ESTIMATE_S: Final = {"light": 2.5, "full": 11.0, "warp": 9.0}


def family(tier: str) -> str:
  """The `ESTIMATE_S` family of tier `tier`."""
  return "light" if tier.startswith(("unlit", "fast")) else "full"


@final
@dataclasses.dataclass(frozen=True, slots=True)
class Plan[U: Hashable]:
  """The modules of the settings one change away from a renderer's that its own frames do not
  bind (`Renderer.nearby_modules`), with their cost: OptiX modules, or on the software traversal
  Warp modules (`kind` "warp").

  Attributes:
    modules: (change "field=value", tier, unit) of each module to create, at most the limit, in
      `DOMAIN`'s order (a unit: `trace.optix.launch.Unit`, `warmup.WarpModule`; tier "" outside the
      tiers).
    outside: the changes no tier serves (their first frame compiles: the documented exceptions).
    skipped: the modules past the limit.
    create: creates the units with PTX at hand into OptiX's disk cache
      (`OptixRenderer.create_modules`), or loads the Warp modules, compiled into Warp's kernel
      cache when missing (`warmup.load`), `workers` at a time: (created, without PTX, seconds).
    kind: "optix" or "warp" (`ESTIMATE_S`).
  """

  modules: tuple[tuple[str, str, U], ...]
  outside: tuple[str, ...]
  skipped: int
  create: Callable[[list[U], int], tuple[int, int, float]] = dataclasses.field(repr=False, compare=False)
  kind: str = "optix"

  def create_all(self, workers: int) -> tuple[int, int, float]:
    """Creates (or loads) every module of the plan, `workers` at a time (`create`)."""
    return self.create([unit for _, _, unit in self.modules], workers)

  @property
  def count(self) -> int:
    """The modules to create."""
    return len(self.modules)

  def seconds(self, workers: int) -> float:
    """The estimated wall seconds of their creation `workers` at a time (`ESTIMATE_S`; each one
    OptiX task graph: at least the slowest)."""
    each = [ESTIMATE_S["warp" if self.kind == "warp" else family(tier)] for _, tier, _ in self.modules]
    return max(max(each, default=0.0), sum(each) / max(1, workers))

  def __str__(self) -> str:
    """E.g. "38 modules (~52 s at 8 at once), 2 changes outside the tiers, 0 past the limit"."""
    head = f"{self.count} modules (~{self.seconds(WORKERS):.0f} s at {WORKERS} at once)"
    return f"{head}, {len(self.outside)} changes outside the tiers, {self.skipped} past the limit"


def plan[U: Hashable](  # noqa: PLR0913  # the plan's inputs, keyword-only
  settings: RenderSettings,
  resolve: Callable[[RenderSettings], Resolution],
  units: Callable[[RenderSettings, Resolution], tuple[U, ...]],
  *,
  own: tuple[U, ...],
  limit: int,
  create: Callable[[list[U], int], tuple[int, int, float]],
  outputs: Sequence[Outputs] = (),
  kind: str = "optix",
) -> Plan[U]:
  """The `Plan` of the settings one change from `settings` (`changes`, with the output sets
  `outputs` first): each resolved (`resolve`; ValueError: a change the scene or device cannot
  render, left out), the modules its frame binds (`units`) that neither `own` (the configuration's)
  nor an earlier change binds, at most `limit`. A change no tier serves is `outside`; its modules
  are planned too for Warp modules (`kind` "warp": loading compiles what its first frame would),
  not for OptiX ones (their PTX is the bundle's or a frame's, never compiled by a warm)."""
  seen = set(own)
  modules: list[tuple[str, str, U]] = []
  outside: list[str] = []
  skipped = 0
  for field, value, s in changes(settings, outputs):
    try:
      r = resolve(s)
    except ValueError:
      continue
    change = f"{field}={value!r}"
    tier = r.choice.tier if r.choice is not None else None
    if tier is None:
      outside.append(change)
      if kind != "warp":
        continue
    for unit in units(s, r):
      if unit in seen:
        continue
      seen.add(unit)
      if len(modules) < limit:
        modules.append((change, tier or "", unit))
      else:
        skipped += 1
  return Plan(tuple(modules), tuple(outside), skipped, create=create, kind=kind)
