"""Render settings: everything a user can switch or tune about a render, as frozen typed groups
of fields (`RenderSettings`: `RenderSettings()` is the default; `FAST` the maximum-throughput field
set).

Rule: a setting that decides which code a kernel holds is compile-time (C): it is resolved into
`Features` (`core/build/variants/resolve.py`), the key of the process-global kernel cache, and code
it switches off is not emitted. A value the kernels read (the numbers, and the switches and choices
a uniform branch selects: the tone mapper, the render order, the sensor effects, ...) is runtime
(R): it reaches them through the small `Params` struct (or per-world arrays) and changing it never
recompiles. Which is which is field metadata: `runtime_field` marks R, `setting` / `rebuild_field`
and a plain default mark C (with the numeric limits that `validate` checks), as is what a change
rebuilds (`rebuild`: "gi" the probe volume of GI, "light_grid" the light visibility grid);
`RenderSettings.fields_table()` walks it. Under `performance.kernels="prebuilt"` a C change
compiles nothing either while a kernel tier still serves it
(`core/build/variants/tiers.py`: the default's, `FAST`'s or unlit compile-time settings, some of
their effects turned off or read at run time). "auto" values resolve against the scene's
capabilities.

A setting is addressed as `group.field` ("shadows.samples"): `RenderSettings(shadows=Shadows(
samples=4))`, `settings.replace(**{"shadows.samples": 4})`, `Renderer.update_settings(...)`.

Layout: `fields` (the `Literal` choices and their id tables, field kinds, `validate`), one module
per family of groups (`outputs`, `sampling`, `lighting`, `effects`, `openpbr`, `surfaces`,
`atmosphere`, `imaging`, `sensor`, `performance`) and `render` (`RenderSettings`).

The package (`rw.settings`) exports `RenderSettings`, `FAST`, `Outputs`, the settings groups
(`rw.settings.Shadows(samples=4)`) and the output specs (`rw.settings.ColorOutput(channels=3)`);
the rest is imported from its module. Most code never needs the groups: `RenderSettings.replace`
takes a group's fields as a mapping (`rw.RenderSettings().replace(shadows={"max_casters": 2})`)
and `Outputs.of` names outputs with their default specs. The table of every field:
docs/settings.md.
"""

from __future__ import annotations

from raywarp.settings.atmosphere import Atmosphere as Atmosphere
from raywarp.settings.effects import GI as GI
from raywarp.settings.effects import Reflections as Reflections
from raywarp.settings.effects import Transparency as Transparency
from raywarp.settings.imaging import Exposure as Exposure
from raywarp.settings.imaging import Grading as Grading
from raywarp.settings.lighting import Environment as Environment
from raywarp.settings.lighting import Lights as Lights
from raywarp.settings.lighting import Shadows as Shadows
from raywarp.settings.openpbr import OpenPBR as OpenPBR
from raywarp.settings.outputs import ColorOutput as ColorOutput
from raywarp.settings.outputs import DepthOutput as DepthOutput
from raywarp.settings.outputs import FloatOutput as FloatOutput
from raywarp.settings.outputs import IdOutput as IdOutput
from raywarp.settings.outputs import LuminanceOutput as LuminanceOutput
from raywarp.settings.outputs import MotionOutput as MotionOutput
from raywarp.settings.outputs import NormalOutput as NormalOutput
from raywarp.settings.outputs import Outputs as Outputs
from raywarp.settings.outputs import ShadowOutput as ShadowOutput
from raywarp.settings.performance import Performance as Performance
from raywarp.settings.render import FAST as FAST
from raywarp.settings.render import RenderSettings as RenderSettings
from raywarp.settings.sampling import AntiAliasing as AntiAliasing
from raywarp.settings.sensor import Sensor as Sensor
from raywarp.settings.surfaces import Post as Post
from raywarp.settings.surfaces import Surfaces as Surfaces
