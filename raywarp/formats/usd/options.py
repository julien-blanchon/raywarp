"""USD adapter options (pxr-free: a warm cache load reads them without importing pxr)."""

from __future__ import annotations

import dataclasses
import enum
import math
from pathlib import Path
from typing import Final, final


class Dialect(enum.Enum):
  """The UsdLux convention of a light's inputs (`raywarp.formats.usd.stage.lux`)."""

  SPEC = "spec"  # the UsdLux spec (Omniverse RTX `omni:rtx:usdluxVersion` >= 2505)
  RTX_LEGACY = "rtx_legacy"  # Omniverse RTX before 2505 (what Kit assumes without the attribute)


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
  """USD adapter options (they are part of the cache key); the options every adapter shares
  (texture storage and size cap, emissive panels, the environment sun) are `rw.SceneOptions`.

  Attributes:
    time: time code read (None: the earliest).
    purposes: rendered UsdGeom purposes.
    double_sided: every gprim double-sided (True), or `doubleSided` decides (False); None: the
      stage's, as Omniverse RTX: every gprim double-sided unless its RTX settings enable face
      culling (`hydra:faceCulling:enabled`).
    invisible_lights: lights under an invisible prim emit (Omniverse: they do not).
    material_contexts: render contexts tried, in order, before a material's universal surface
      (USD's rule: an MDL material wins over its UsdPreviewSurface approximation): "mtlx"
      (MaterialX surface networks: any surfaceshader nodedef of the MaterialX standard
      libraries, converted through the MaterialX MDL generator and the MDL SDK) and "mdl"
      (Omniverse MDL). MaterialX first by default, as SimReady declares OpenPBR the target;
      ("mdl", "mtlx") prefers the MDL material; ("mtlx",) never reads MDL (a universal MDL
      shader is then reported). Both need the bundled MDL SDK (`raywarp._native`).
    displacement: prepare supported scalar height into shared mesh geometry (False: reported
      and omitted). Bump-only and vector displacement are not implemented.
    displacement_max_level: maximum four-way linear triangle subdivision levels, >= 0.
    displacement_max_triangles: total newly displaced triangles across shared mesh variants, >= 0.
    lux_dialect: the UsdLux dialect of lights without an authored `omni:rtx:usdluxVersion` (None:
      detected per light from Kit layer metadata and attribute naming; set it for a stage whose
      layers carry no such marks, e.g. one built live in an Omniverse Kit app).

  Emissive surfaces are not an option here: the upload fits lights to their emitting parts
  (`SceneOptions.emissive_lights`), switched by the `lighting.emissive_lights` render setting.

  Raises:
    ValueError: for a count below its minimum, a value of the wrong type, unknown or repeated
      render contexts, or a non-finite time.
  """

  time: float | None = None
  purposes: tuple[str, ...] = ("default", "render")
  double_sided: bool | None = None
  invisible_lights: bool = False
  material_contexts: tuple[str, ...] = ("mtlx", "mdl")
  displacement: bool = True
  displacement_max_level: int = 6
  displacement_max_triangles: int = 16_000_000
  lux_dialect: Dialect | None = None

  def __post_init__(self) -> None:
    """Checks the values (at construction: not deep in a stage read)."""
    for name, minimum in (("displacement_max_level", 0), ("displacement_max_triangles", 0)):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"usd.Options.{name}: an int of at least {minimum}, got {value!r}")
    for name in ("invisible_lights", "displacement"):
      if not isinstance(getattr(self, name), bool):
        raise ValueError(f"usd.Options.{name}: a bool, got {getattr(self, name)!r}")
    if self.double_sided is not None and not isinstance(self.double_sided, bool):
      raise ValueError(f"usd.Options.double_sided: a bool or None, got {self.double_sided!r}")
    for name in ("purposes", "material_contexts"):
      value = getattr(self, name)
      if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"usd.Options.{name}: a tuple of str, got {value!r}")
    contexts = self.material_contexts
    if any(c not in MATERIAL_CONTEXTS for c in contexts) or len(set(contexts)) != len(contexts):
      raise ValueError(f"usd.Options.material_contexts: distinct names among {MATERIAL_CONTEXTS}, got {contexts!r}")
    if self.lux_dialect is not None and not isinstance(self.lux_dialect, Dialect):
      raise ValueError(f"usd.Options.lux_dialect: a usd.Dialect or None, got {self.lux_dialect!r}")
    if self.time is not None and not math.isfinite(self.time):
      raise ValueError(f"usd.Options.time: a finite time code or None, got {self.time!r}")


# The environment variables of the MDL search paths (part of the cache key: `api.cache_key`).
MDL_PATH_VARIABLES: Final = ("MDL_USER_PATH", "MDL_SYSTEM_PATH")
# Omniverse's own MDL library shipped with raywarp (`raywarp.formats.usd.materials.mdl.load`; its
# content: in the cache key).
MDL_LIBRARY: Final = Path(__file__).resolve().parent / "materials" / "mdl" / "library"

# Render contexts `Options.material_contexts` may name.
MATERIAL_CONTEXTS: Final = ("mtlx", "mdl")
# The default options (a shared immutable value: argument defaults).
DEFAULT_OPTIONS: Final = Options()
