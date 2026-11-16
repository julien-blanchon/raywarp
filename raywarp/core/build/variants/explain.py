"""What a settings object resolves to on a scene (`Renderer.explain()`): every feature the
kernels run with the setting that asked for it and why it is on or off ("auto" choices, settings
the scene or the other settings make moot), the frame's cost class and the kernel tier serving it
(`core/build/variants/tiers.py`).

It reads the resolved `Features` (`core/build/variants/resolve.py`): what it reports is what
compiles and runs, never a second derivation of the rules.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, final

from raywarp.core.kernels.abi.features import Features
from raywarp.settings import RenderSettings
from raywarp.settings.fields import REFLECTION_SHADOWS

if TYPE_CHECKING:
  from raywarp.core.build.variants.tiers import Choice

_REFLECTION_SHADOW_NAMES: Final = {v: k for k, v in REFLECTION_SHADOWS.items()}


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Row:
  """One feature of an `Explanation`.

  Attributes:
    feature: what runs ("shadows", "gi", ...).
    setting: the setting(s) that ask for it, with their values.
    resolved: what runs ("off", "on", a mode or a count).
    why: the reason (the setting itself, the scene's data, or another setting that makes it
      moot); "" when the setting says it all.
  """

  feature: str
  setting: str
  resolved: str
  why: str = ""


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Explanation:
  """What a Renderer's settings resolve to (`explain`; `str()` is a table).

  Attributes:
    cost: the cost class, cheapest first "geometry", "unlit", "fast", "default" or "quality"
      (`cost_class`).
    cost_why: the features that put the frame in that class.
    rows: every feature, in pipeline order.
    kernels: the kernel tier serving the settings and the switches they turn off at run time, or
      why they compile kernels of their own (`kernels_line`).
  """

  cost: str
  cost_why: str
  rows: tuple[Row, ...]
  kernels: str = ""

  def row(self, feature: str) -> Row:
    """The row of `feature` (KeyError for an unknown one)."""
    for r in self.rows:
      if r.feature == feature:
        return r
    raise KeyError(feature)

  def __str__(self) -> str:
    """A plain-text table: the cost class, then one line per feature."""
    width = max(len(r.feature) for r in self.rows)
    lines = [f"cost: {self.cost} ({self.cost_why})"]
    if self.kernels:
      lines.append(f"kernels: {self.kernels}")
    for r in self.rows:
      why = f"  -- {r.why}" if r.why else ""
      lines.append(f"  {r.feature:<{width}}  {r.resolved:<12} [{r.setting}]{why}")
    return "\n".join(lines)


def explain(settings: RenderSettings, f: Features, choice: Choice | None = None) -> Explanation:
  """The explanation of `settings` resolved to `f` (`Renderer.features`), served by the kernels
  of `choice` (`tiers.route`; None: `performance.kernels="scene"`)."""
  cost, why = cost_class(f)
  if choice is not None:
    kernels = kernels_line(choice, shaded=f.any_rgb)
  else:
    scene = settings.performance.kernels == "scene"
    kernels = "scene (specialised on this scene: performance.kernels='scene')" if scene else ""
  return Explanation(cost=cost, cost_why=why, rows=_rows(settings, f), kernels=kernels)


def kernels_line(choice: Choice, *, shaded: bool) -> str:
  """The tier serving a configuration (prebuilt, no compile on a change within it), the switches
  it turns off at run time, or the fields that leave the nearest tier (own kernels: they compile
  unless cached)."""
  if not shaded:
    return "geometry (every tier's unshaded kernel)"
  if choice.tier is not None:
    off = f", switched off at run time: {', '.join(choice.off)}" if choice.off else ""
    return f"tier {choice.tier!r}{off}"
  if choice.nearest is None and choice.outside:
    return f"own (outside the tiers: its {', '.join(choice.outside)} are not served by this backend's tiers)"
  if choice.nearest is None:
    return "own (outside the tiers: no tier fits this renderer)"
  return f"own (outside the tiers; nearest {choice.nearest!r} differs in {', '.join(choice.outside)})"


def cost_class(f: Features) -> tuple[str, str]:
  """(class, reason): "geometry" (no shaded output: no shading is compiled), "unlit" (base colors
  only), "quality" (several samples per pixel or per shadowed light), "default" (traced
  reflections or area-light shading, as `RenderSettings()`), else "fast" (as `FAST`)."""
  if not f.any_rgb:
    return "geometry", "only geometry outputs: no shading"
  if f.unlit:
    return "unlit", 'surfaces.shading="unlit": base colors, no light'
  heavy = [n for n, on in (("aa.spp", f.spp > 1), ("shadows.samples", f.shadows and f.shadow_samples > 1)) if on]
  if heavy:
    return "quality", " and ".join(f"{n} > 1" for n in heavy)
  medium = [n for n, on in (("reflection rays", f.glossy != 0), ("LTC area lights", f.area_lights)) if on]
  if medium:
    return "default", " and ".join(medium)
  return "fast", "point lights, no reflection ray, one sample"


def _rows(s: RenderSettings, f: Features) -> tuple[Row, ...]:
  """Every feature's row (see `explain`)."""
  if not f.any_rgb:
    moot = "no shaded output: shading is not compiled"
    return (
      Row(feature="traversal", setting=f"performance.traversal={s.performance.traversal!r}", resolved=f.traversal),
      Row(feature="shading", setting=f"outputs={sorted(s.outputs.kinds)}", resolved="off", why=moot),
    )
  return (
    Row(feature="traversal", setting=f"performance.traversal={s.performance.traversal!r}", resolved=f.traversal),
    _auto("textures", "surfaces.textures", s.surfaces.textures, f.textures, "the scene binds a texture"),
    _auto("transparency", "transparency.enabled", s.transparency.enabled, f.transparency, "translucent data in the scene"),
    _lights(s, f),
    _auto("light grid", "lighting.light_grid", s.lighting.light_grid, f.culling, "8 lights or more"),
    _auto("headlight", "lighting.camera_light", s.lighting.camera_light, f.headlight, "the scene authors one"),
    _area_lights(s, f),
    _shadows(s, f),
    _auto("environment", "environment.enabled", s.environment.enabled, f.env_lit, "the scene has a lighting map"),
    _reflections(s, f),
    _gi(s, f),
    _sampling(s, f),
    Row(
      feature="display",
      setting=f"exposure.mode={s.exposure.mode!r}",
      resolved="deferred" if s.deferred_display() else "in kernel",
      why="automatic exposure meters every image" if s.auto_exposure() else "",
    ),
  )


def _auto(feature: str, name: str, value: object, on: bool, reason: str) -> Row:
  """The row of a True / False / "auto" setting resolved to `on`."""
  why = (reason if on else f"not: {reason}") if value == "auto" else ""
  if value is True and not on:
    why = "nothing in the scene uses it"
  return Row(feature=feature, setting=f"{name}={value!r}", resolved="on" if on else "off", why=why)


def _lights(s: RenderSettings, f: Features) -> Row:
  """The light loop: the light count and the per-point cap."""
  setting = f"lighting.max_exact={s.lighting.max_exact!r}"
  if not f.nlight:
    return Row(feature="lights", setting=setting, resolved="0", why="the scene has no light")
  if f.light_cap:
    return Row(feature="lights", setting=setting, resolved=f"{f.light_cap} of {f.nlight} exact", why="the rest diffusely")
  why = "no cap" if s.lighting.max_exact is None else "the cap is not below the light count"
  return Row(feature="lights", setting=setting, resolved=f"{f.nlight} exact", why=why)


def _area_lights(s: RenderSettings, f: Features) -> Row:
  """LTC area lights: asked for, and present in the scene."""
  setting = f"lighting.area_lights={s.lighting.area_lights!r}"
  if s.lighting.area_lights == "point":
    return Row(feature="area lights", setting=setting, resolved="point form")
  return Row(
    feature="area lights",
    setting=setting,
    resolved="ltc" if f.area_lights else "off",
    why="" if f.area_lights else "no area light",
  )


def _shadows(s: RenderSettings, f: Features) -> Row:
  """Shadow rays: on, the caster budget, the samples and their targets."""
  sh = s.shadows
  setting = f"shadows.enabled={sh.enabled}, samples={sh.samples}, max_casters={sh.max_casters!r}"
  if not f.shadows or not f.nlight:
    return Row(feature="shadows", setting=setting, resolved="off", why="" if not sh.enabled else _no_light(f))
  parts = [f"{f.shadow_samples} ray(s)"]
  parts.append(f"{f.budget} casters" if f.budget else "every caster")
  targets = "fixed light targets (per frame)" if f.area_lights or f.sun_disc else ""
  why = "the cap covers every potential caster" if sh.max_casters is not None and not f.budget else ""
  return Row(feature="shadows", setting=setting, resolved=", ".join(parts), why="; ".join(w for w in (targets, why) if w))


def _no_light(f: Features) -> str:
  """Why a lit build traces no shadow ray: unlit shading, or a scene without lights."""
  return 'surfaces.shading="unlit"' if f.unlit else "no light"


def _reflections(s: RenderSettings, f: Features) -> Row:
  """Traced reflections and their shadows."""
  r = s.reflections
  setting = f"reflections.enabled={r.enabled}"
  if not f.glossy:
    why = "" if not r.enabled else 'surfaces.shading="unlit"' if s.surfaces.shading == "unlit" else "no shaded output"
    return Row(feature="reflections", setting=setting, resolved="off", why=why)
  shadows = _REFLECTION_SHADOW_NAMES[f.reflection_shadows]
  return Row(feature="reflections", setting=setting, resolved="dominant", why=f"shadows {shadows}")


def _gi(s: RenderSettings, f: Features) -> Row:
  """Probe GI."""
  setting = f"gi.enabled={s.gi.enabled}"
  return Row(feature="gi", setting=setting, resolved="probes" if f.gi else "off")


def _sampling(s: RenderSettings, f: Features) -> Row:
  """Anti-aliasing samples."""
  n = s.aa.spp if f.spp > 1 else 1  # Features.spp 2: any n > 1 at run time (`resolve._spp`)
  return Row(feature="sampling", setting=f"aa.spp={s.aa.spp}", resolved=f"{n}x{n} spp")
