"""The documented settings recipes by name, for the tooling (benchmarks, the zero-compile walk, the
web UI): the default (`RenderSettings()`), the fast field set (`rw.FAST`) and the looks of
docs/settings.md "Recipes", each as dotted changes of the default. The library has no named
configurations: users write these fields (docs/settings.md). `tests/unit/test_default_settings.py`
snapshots each recipe's settings and resolved features; the zero-compile walk
(`tools/prebuilt/zero_space.py`) starts from each and proves it served by the prebuilt tiers.

  from tools.recipes import settings
  s = settings("quality")  # RenderSettings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
  from raywarp.settings import RenderSettings

# Recipe -> its dotted changes of the default ("fast": `rw.FAST`; "rtx_look": the USD map's base
# for RTX's RealTimePathTracing, `raywarp/formats/usd/render_settings.py`).
RECIPES: Final[dict[str, dict[str, object]]] = {
  "default": {},
  "fast": {},  # `rw.FAST`
  "unlit": {"surfaces.shading": "unlit", "surfaces.anisotropy": 1},
  "quality": {
    "aa.spp": 2,
    "shadows.samples": 2,
    "shadows.max_casters": 16,
    "lighting.max_exact": 8,
    "reflections.max_exact": 4,
    "reflections.shadows": "sun",
    "transparency.max_layers": 8,
  },
  "reference": {
    "aa.spp": 2,
    "shadows.samples": 8,
    "shadows.max_casters": 16,
    "lighting.max_exact": None,
    "reflections.max_exact": None,
    "reflections.shadows": "sun",
    "transparency.max_layers": 8,
  },
  "rtx_look": {
    "gi.normal_bias": 0.3,
    "lighting.max_exact": None,
    "lighting.light_grid_cell": 0.5,
    "reflections.max_exact": None,
    "reflections.shadows": "sun",
    "shadows.max_casters": 8,
    "shadows.translucent": "clear",
    "post.tone_mapper": "aces_rtx",
  },
}


def settings(name: str) -> RenderSettings:
  """The settings of recipe `name` (`RECIPES`).

  Raises:
    ValueError: for an unknown recipe.
  """
  from raywarp.settings import FAST, RenderSettings  # tooling imports raywarp late

  if name not in RECIPES:
    raise ValueError(f"unknown recipe {name!r}: one of {list(RECIPES)}")
  return RenderSettings().replace(FAST if name == "fast" else RECIPES[name])
