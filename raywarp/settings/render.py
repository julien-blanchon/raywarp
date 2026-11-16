"""The complete configuration of a render, `RenderSettings`: one frozen group per subsystem, with
the runtime "group.field" addressing (`replace`, `flat`, `diff`, `fields_table`), and `FAST`, the
field set for maximum training throughput.

The default (`RenderSettings()`, every group at its own defaults), measured choices:
- LTC area lights and the light grid;
- one fixed center shadow ray for each of the 4 strongest casters, tinted sun shadows through
  glass;
- glossy reflections along the GGX dominant direction;
- 4 translucent layers, anisotropic texture filtering (8 taps at grazing angles: +2-5 % on L4);
- light caps: the 4 strongest lights of a point shaded exactly, the others' irradiance diffusely
  (a large interior, L4 16 x 256^2: -44 %; an open-plan interior's FLIP 0.06 vs every light
  exact); 2 at reflection hits, unshadowed (the sun then lights every indoor wall a reflection
  sees, e.g. steel table legs glow in a room with a sunlit window; reflections.shadows "sun"
  removes it for +10-49 % frame time on RTX);
- probe GI resampled into the ambient cube (0.5 m / 256 rays / 8 updates, `settings.GI`: 16 or 32
  updates matched the references no better; vs Mitsuba at 512 px mean FLIP: window-lit room 0.70
  -> 0.12, closed workcell 0.55 -> 0.14, sunlit tabletop 0.071 -> 0.038);
- automatic exposure (`exposure.mode` "auto": each camera image metered, `settings.Exposure`),
  locked after the first metered frame per world and camera (a reset world meters its next frame;
  `exposure.adaptation="per_frame"` is strictly per frame);
- PBR Neutral as the tone mapper (RTX's ACES approximation, "aces_rtx", turned HDRI-lit tabletops
  milky, +0.3..+0.9 EV, FLIP 0.39-0.48);
- the environment's sun a shadowed directional light (`SceneOptions.env_sun`, an upload option).
Single-field opt-ins served by the prebuilt kernels (`core/build/variants/tiers.py`, no compile):
`aa.spp=2`, `shadows.samples`, `reflections.shadows="sun"`, the light caps and the caster budget
up to 16; `reflections.enabled=False` (run time: -29 % frame time on a
large office interior, RTX PRO 6000 512^2 x 512 worlds).
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Mapping
from typing import Final, final, get_args

from raywarp._base.named import field_value, replace_named
from raywarp.settings.atmosphere import Atmosphere
from raywarp.settings.effects import GI, Reflections, Transparency
from raywarp.settings.fields import Rebuild
from raywarp.settings.imaging import Exposure, Grading, deferred
from raywarp.settings.lighting import Environment, Lights, Shadows
from raywarp.settings.openpbr import OpenPBR
from raywarp.settings.outputs import ColorOutput, DepthOutput, FloatOutput, NormalOutput, Outputs
from raywarp.settings.performance import Performance
from raywarp.settings.sampling import AntiAliasing
from raywarp.settings.sensor import Sensor
from raywarp.settings.surfaces import Post, Surfaces

type SettingsGroup = (
  Outputs
  | AntiAliasing
  | Shadows
  | Lights
  | Environment
  | Reflections
  | Transparency
  | GI
  | Surfaces
  | Post
  | Atmosphere
  | Exposure
  | Grading
  | Sensor
  | Performance
  | OpenPBR
)
# The group classes, for `isinstance` (the members of `SettingsGroup`).
# Final[...]: `get_args` is typed as returning `Any`s.
_GROUP_TYPES: Final[tuple[type[SettingsGroup], ...]] = get_args(SettingsGroup.__value__)


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class FieldInfo:
  """One row of `RenderSettings.fields_table()`.

  Attributes:
    name: "group.field".
    runtime: R (True: never recompiles) or C (False: part of `Features`).
    rebuild: what a change rebuilds besides the kernels (None: nothing).
    default: the default value.
  """

  name: str
  runtime: bool
  rebuild: Rebuild | None
  default: object


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class RenderSettings:
  """Feature switches of a `Renderer`, one frozen group per subsystem (see the module docstring).
  `RenderSettings()` is the one default configuration, each group at its own defaults; `FAST` is
  the documented field set for maximum training throughput (`RenderSettings().replace(FAST)`).

  Attributes:
    outputs: what is written (`Outputs`).
    aa: anti-aliasing (`AntiAliasing`).
    shadows: shadow rays (`Shadows`).
    lighting: light shading (`Lights`).
    environment: image-based lighting and background (`Environment`).
    reflections: traced reflections (`Reflections`).
    transparency: translucent layers (`Transparency`).
    gi: probe GI (`GI`).
    openpbr: local OpenPBR lobes and their angular integration (`OpenPBR`).
    surfaces: textures, normals, culling (`Surfaces`).
    post: the tone mapper (`Post`).
    atmosphere: distance fog and haze (`Atmosphere`).
    exposure: automatic or manual exposure (`Exposure`).
    grading: white balance and colour grading (`Grading`).
    sensor: camera and depth sensor effects (chromatic aberration among them), runtime and off by
      default (`Sensor`).
    performance: kernel build, traversal and rebuild choices that never change pixels
      (`Performance`).
  """

  outputs: Outputs = dataclasses.field(default=Outputs(rgb=ColorOutput()))
  aa: AntiAliasing = dataclasses.field(default=AntiAliasing())
  shadows: Shadows = dataclasses.field(default=Shadows())
  lighting: Lights = dataclasses.field(default=Lights())
  environment: Environment = dataclasses.field(default=Environment())
  reflections: Reflections = dataclasses.field(default=Reflections())
  transparency: Transparency = dataclasses.field(default=Transparency())
  gi: GI = dataclasses.field(default=GI())
  openpbr: OpenPBR = dataclasses.field(default=OpenPBR())
  surfaces: Surfaces = dataclasses.field(default=Surfaces())
  post: Post = dataclasses.field(default=Post())
  atmosphere: Atmosphere = dataclasses.field(default=Atmosphere())
  exposure: Exposure = dataclasses.field(default=Exposure())
  grading: Grading = dataclasses.field(default=Grading())
  sensor: Sensor = dataclasses.field(default=Sensor())
  performance: Performance = dataclasses.field(default=Performance())

  def __post_init__(self) -> None:
    """Checks that every group is an instance of its class (TypeError) and the checks across
    groups (ValueError): the sensor's effects and the post anti-aliasing need their outputs
    (`Sensor.check_outputs`)."""
    for f in dataclasses.fields(self):
      want = type(f.default)
      value = field_value(self, f.name)
      if not isinstance(value, want):
        raise TypeError(f"{f.name} must be a {want.__name__}, got {type(value).__name__}")
    self.sensor.check_outputs(self.outputs)
    if self.aa.post and self.outputs.rgb is None and self.outputs.luminance is None:
      raise ValueError("aa.post filters the display color: request outputs.rgb or outputs.luminance")
    if self.aa.post and self.outputs.depth is None:
      raise ValueError("aa.post finds the geometric edges in the depth: request outputs.depth")
    self._check_area_rays()
    if self.lighting.contact_ao and self.outputs.depth is not None and self.outputs.depth.convention != "planar":
      # The contact AO places points from planar depth (`shade/ao/contact.py`).
      raise ValueError('lighting.contact_ao > 0 needs a planar depth output (outputs.depth.convention="planar")')

  def _check_area_rays(self) -> None:
    """The soft area shadows' pass (`shadows.area_rays`) needs LTC area lights, one sample per
    pixel and a float32 position output if one is requested (ValueError)."""
    if not self.shadows.area_rays:
      return
    needs = {
      'lighting.area_lights="ltc"': self.lighting.area_lights == "ltc",
      "aa.spp=1": self.aa.spp == 1,
      'a float32 "position" output': self.outputs.position is None or self.outputs.position.dtype == "float32",
    }
    missing = [name for name, ok in needs.items() if not ok]
    if missing:
      raise ValueError(f"shadows.area_rays > 0 needs {', '.join(missing)}")

  def replace(self, changes: Mapping[str, object] | None = None, /, **groups: object) -> RenderSettings:
    """A validated copy with changes: dotted fields in a mapping (`replace({"shadows.samples":
    4})`) and / or groups by keyword, each a whole group (`shadows=Shadows(samples=4)`) or a
    mapping of its fields (`shadows={"samples": 4}`, over the current group); dotted keywords
    (`**{"shadows.samples": 4}`) work too. Dotted fields apply over a group given whole.

    Raises:
      ValueError: for an unknown group or field, or an invalid value.
      TypeError: for a group that is not an instance of its class.
    """
    changes = {**(changes or {}), **groups}
    for key, value in list(changes.items()):
      if key in _GROUPS and isinstance(value, Mapping):
        del changes[key]
        changes |= {f"{key}.{name}": v for name, v in value.items()}
    whole: dict[str, object] = {}
    fields: dict[str, dict[str, object]] = {}
    for key, value in changes.items():
      group, dotted, name = key.partition(".")
      if group not in _GROUPS:
        raise ValueError(f"unknown setting {key!r}: groups are {sorted(_GROUPS)} (or 'group.field')")
      if dotted:
        if name not in _GROUPS[group]:
          raise ValueError(f"unknown setting {key!r}: {group} has {sorted(_GROUPS[group])}")
        fields.setdefault(group, {})[name] = value
      else:
        whole[group] = value
    for group, values in fields.items():
      base = whole.get(group, self.group(group))
      if not isinstance(base, _GROUP_TYPES):
        raise TypeError(f"{group} must be a settings group, got {type(base).__name__}")
      whole[group] = replace_named(base, values)
    return replace_named(self, whole)

  def written_outputs(self) -> Outputs:
    """The outputs the render writes: `outputs` plus the deferred display's "hdr" (`outputs`
    itself without the deferred display). That "hdr", read only by the deferred display, is
    float16 storing sqrt(radiance) (`FloatOutput.sqrt`: 6 instead of 12 B per pixel, at most one
    display level of error, no exposure needed beforehand); with the contact AO on, also a
    planar "depth" when the outputs have none."""
    if not self.deferred_display():
      return self.outputs
    added: dict[str, object] = {}
    if "hdr" not in self.outputs.kinds:
      added["hdr"] = FloatOutput(dtype="float16", sqrt=True)
    if self.contact_ao() and "depth" not in self.outputs.kinds:
      added["depth"] = DepthOutput()  # the contact AO's (`shade/ao/contact.py`)
    if self.area_rays():  # the soft area shadows' inputs (`shade/shadows/area.py`)
      kinds = self.outputs.kinds
      if "position" not in kinds:
        added["position"] = FloatOutput()
      if "normal" not in kinds:
        added["normal"] = NormalOutput(frame="geometric", dtype="float16")
    return replace_named(self.outputs, added) if added else self.outputs

  def deferred_display(self) -> bool:
    """Whether the display is written by the deferred display pass from the frame's HDR image
    (`core/post/display/deferred.py`): automatic exposure, chromatic aberration or the grading, with
    a shaded output (`post.display.deferred`)."""
    return self.outputs.shaded and deferred(
      auto_exposure=self.auto_exposure(),
      aberration=self.sensor.aberration,
      grading=self.grading,
      contact_ao=self.contact_ao(),
      area_rays=self.area_rays(),
    )

  def area_rays(self) -> int:
    """The soft area shadows' rays per light and pixel (`shadows.area_rays`; 0 unlit, without
    shadows or without a shaded output)."""
    lit = self.outputs.shaded and self.surfaces.shading != "unlit" and self.shadows.enabled
    return self.shadows.area_rays if lit else 0

  def contact_ao(self) -> float:
    """The contact AO's strength (`lighting.contact_ao`; 0 unlit or without a shaded output)."""
    return self.lighting.contact_ao if self.outputs.shaded and self.surfaces.shading != "unlit" else 0.0

  def depth_with_rgb(self) -> bool:
    """Whether every camera writing the color writes its depth too (`CameraSpec.outputs` leaving
    it out): the input of the contact AO and of the post anti-aliasing's edges (`aa.post`)."""
    return self.contact_ao() > 0.0 or self.aa.post

  def auto_exposure(self) -> bool:
    """Whether every camera image is metered (`exposure.mode` "auto"): a shaded output of lit
    shading ("unlit" shows the base colors as they are)."""
    return self.exposure.mode == "auto" and self.outputs.shaded and self.surfaces.shading != "unlit"

  def group(self, name: str) -> SettingsGroup:
    """The settings group `name` (a name known only at runtime: "shadows", ...)."""
    value = field_value(self, name)
    if not isinstance(value, _GROUP_TYPES):  # `__post_init__` checked every group's type
      raise TypeError(f"{name} is not a settings group")
    return value

  def flat(self) -> dict[str, object]:
    """The "group.field" -> value table of every field."""
    return {
      f"{group}.{f.name}": field_value(self.group(group), f.name)
      for group in _GROUPS
      for f in dataclasses.fields(self.group(group))
    }

  def diff(self, other: RenderSettings) -> dict[str, object]:
    """The "group.field" values of `other` that differ from this one's (`self.replace(**d) ==
    other`)."""
    mine = self.flat()
    return {k: v for k, v in other.flat().items() if mine[k] != v}

  @classmethod
  def fields_table(cls) -> list[FieldInfo]:
    """Every field, walked from the groups' metadata: name, runtime, rebuild, default (the value
    in `RenderSettings()`: e.g. `outputs.rgb` is a `ColorOutput()` there, None in `Outputs()`)."""
    defaults = cls()
    return [
      FieldInfo(
        name=f"{group}.{f.name}",
        runtime=bool(f.metadata.get("runtime")),
        rebuild=f.metadata.get("rebuild"),
        default=field_value(defaults.group(group), f.name),
      )
      for group in _GROUPS
      for f in dataclasses.fields(defaults.group(group))
    ]

  def rebuild_inputs(self, rebuild: Rebuild) -> tuple[object, ...]:
    """The values of the fields whose change rebuilds `rebuild` (part of that structure's key:
    `core/pipeline/baked.py`)."""
    return tuple(field_value(self.group(group), name) for group, name in _rebuild_fields(rebuild))


@functools.cache
def _rebuild_fields(rebuild: Rebuild) -> tuple[tuple[str, str], ...]:
  """The (group, field) names whose metadata names `rebuild`."""
  rows = RenderSettings.fields_table()
  return tuple((g, n) for g, _, n in (f.name.partition(".") for f in rows if f.rebuild == rebuild))


# Group name -> its field names.
_GROUPS: Final = {
  g.name: frozenset(f.name for f in dataclasses.fields(RenderSettings().group(g.name)))
  for g in dataclasses.fields(RenderSettings)
}


# Maximum training throughput, still PBR (`RenderSettings().replace(FAST)`; the prebuilt kernels
# hold its own build, `core/build/variants/tiers.py` "fast"): every light point-form and exact (no
# cap: cheaper than the cap's ranking for point lights), shadow rays for the 2 strongest casters
# only, no reflection rays (a MuJoCo `reflectance` stays a specular F0; its ray: +50 % GPU), opaque
# shadows, 2 translucent layers and isotropic texture filtering. Against the default (RTX PRO
# 6000, 512^2 x 512 worlds): a large office interior -58 % frame time (70.7 vs 169.7 ms), other
# interiors -63..-67 %, a feature-dense room -38 %, an open courtyard -29 %, robot / HDRI tabletops
# about -20 %. Its tier's OptiX register caps are lower
# (`core/build/variants/registers.OPTIX_REGISTERS`).
FAST: Final[Mapping[str, object]] = types.MappingProxyType(
  {
    "lighting.area_lights": "point",
    "lighting.max_exact": None,
    "lighting.visible_lights": False,
    "shadows.max_casters": 2,
    "shadows.translucent": "off",
    "reflections.enabled": False,
    "reflections.max_exact": None,
    "transparency.max_layers": 2,
    "surfaces.anisotropy": 1,
  }
)


def mapped(
  base: RenderSettings, fields: Mapping[str, object | None], overrides: Mapping[str, object] | None = None
) -> RenderSettings:
  """How a foreign render config becomes `RenderSettings`, one rule for every adapter (MuJoCo
  Warp's `create_render_context`, Newton's `RenderConfig`, Isaac Lab's `RaywarpRendererCfg`,
  mjlab's sensor flags): `base` (the default, `RenderSettings()`, unless the caller gives its
  own), then the config's mapped `fields` ("group.field" keys; None: the config leaves it unset,
  `base`'s kept), then the user's dotted `overrides` (their keys win)."""
  given = {key: value for key, value in fields.items() if value is not None}
  return base.replace(**given).replace(**dict(overrides or {}))
