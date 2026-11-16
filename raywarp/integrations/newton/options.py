"""The extra keyword options of SensorCamera / ViewerRaywarpHeadless: `RenderSettings.replace` keys,
checked before any scene is built (a typo fails at construction, not at the first frame)."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from raywarp.settings import RenderSettings


def _setting_keys() -> frozenset[str]:
  """Every `RenderSettings.replace` key: "group.field" names and group names."""
  names = {f.name for f in RenderSettings.fields_table()}
  return frozenset(names | {k.partition(".")[0] for k in names})


def render_options(options: Mapping[str, object], reserved: Collection[str], owner: str) -> dict[str, object]:
  """A front end's keyword options, validated as `RenderSettings.replace` keys.

  Raises:
    TypeError: for a reserved option (the front end sets it itself) or an unknown one.
  """
  # A reserved group also reserves its dotted fields ("outputs" -> "outputs.depth").
  taken = [key for key in options if any(key == name or key.startswith(name + ".") for name in reserved)]
  if taken:
    raise TypeError(f"{owner} sets {taken} itself; remove them from the render options")
  keys = _setting_keys()
  unknown = [name for name in options if name not in keys]
  if unknown:
    raise TypeError(
      f"unknown render options {unknown}: use RenderSettings groups or 'group.field' keys (lights and envmaps: lighting=)"
    )
  return dict(options)


def emissive_lights(options: Mapping[str, object]) -> bool:
  """Whether renderers of the validated keyword `options` light with emissive panels (a front
  end's options are fixed: without them the upload skips the panels' fit)."""
  return RenderSettings().replace(**options).lighting.emissive_lights
