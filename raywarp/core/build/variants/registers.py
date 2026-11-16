"""The OptiX register caps of a build: per tier family (`OPTIX_REGISTERS`), overridden by the
current tuning's (`core/pipeline/platform/tuning.py`), per module from its build word
(`optix_registers`: a module option, one PTX serves every cap).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Final

from raywarp.core.build.variants import tiers
from raywarp.core.kernels.abi import toggles
from raywarp.core.pipeline.platform import tuning

if TYPE_CHECKING:
  from raywarp.core.kernels.abi.features import Features
  from raywarp.settings import RenderSettings

# The OptiX register cap of geometry-only builds: one closest hit, no shading, 1.4x faster at 64
# than at 128 (4096 x 64^2 depth + seg, RTX PRO 6000).
OPTIX_GEOMETRY_REGISTERS: Final = 64
# The OptiX register caps (shaded, light) of each tier family's shaded builds (a module option: one
# PTX serves every cap). Shaded: 128 keeps the default's heavy shading fast (a large interior, RTX
# PRO 6000: 4526 vs 3161 at 96, 2181 at 64); 96 for the fast and unlit tiers' light shading
# (+12-14 % over 128 on the RTX PRO 6000, -3-4 % on an L4). Light (0: the shaded cap): the cap of
# the builds without heavy shading (clear coat, layered lobes, light culling); the fast tier's 80
# against 96: 4-7 % faster on robot scenes and interiors (RTX PRO 6000, 512 worlds x 512^2), while
# builds with those features were 10-46 % slower at 80 and keep the shaded cap.
OPTIX_REGISTERS: Final = {"unlit": (96, 0), "fast": (96, 80), "default": (128, 0)}


# Settings no tier fixes (`tiers.tier_settings` keeps the configuration's): not compared.
_UNCOMPARED: Final = ("outputs.", "performance.", "gi.enabled")


@functools.lru_cache(maxsize=64)
def family(settings: RenderSettings) -> str:
  """The tier family `settings` shade like (`tiers.TIERS`' bases "unlit", "fast", "default"):
  "unlit" for unlit shading (the unlit tier serves it), else of "fast" and "default" the base whose
  compile-time settings differ from them in the fewest fields (the default on a tie). A tier
  variant's settings are its base's family ("fast_ltc": "fast")."""
  if settings.surfaces.shading == "unlit":
    return "unlit"
  own = settings.flat()
  fields = [f.name for f in settings.fields_table() if not f.runtime and not f.name.startswith(_UNCOMPARED)]

  def distance(base: str) -> int:
    """The compile-time fields `settings` and tier base `base` differ in."""
    theirs = tiers.TIERS[base].flat()
    return sum(own[name] != theirs[name] for name in fields)

  return min(("default", "fast"), key=distance)


def register_caps(settings: RenderSettings) -> tuple[int, int]:
  """The OptiX register caps (shaded, light) of a build of `settings`: its tier family's
  (`family`, `OPTIX_REGISTERS`), each overridden by the current tuning's."""
  shaded, light = OPTIX_REGISTERS[family(settings)]
  t = tuning.current()
  return (
    shaded if t.optix_max_registers is None else t.optix_max_registers,
    light if t.optix_light_registers is None else t.optix_light_registers,
  )


def optix_registers(caps: tuple[int, int], f: Features, word: int) -> int:
  """The register cap of the OptiX module of build `f` binding build word `word` (its geometry
  copy, `trace/optix/device/programs.build_word`) under caps (shaded, light) `caps`
  (`register_caps`): a geometry-only build's `OPTIX_GEOMETRY_REGISTERS`, a shaded build's light cap
  when set and the module has no heavy shading (clear coat, layered lobes, light culling: compiled
  in and, where bound, on in `word`), else the shaded cap."""
  if not f.any_rgb:
    return OPTIX_GEOMETRY_REGISTERS
  on = toggles.word_on(f.bound, word)
  heavy = (
    (f.clearcoat and on(toggles.CLASS_COAT))
    or (f.layered and on(toggles.CLASS_LAYERS))
    or (f.culling and on(toggles.CLASS_CULLING))
  )
  shaded, light = caps
  return light if light and not heavy else shaded
