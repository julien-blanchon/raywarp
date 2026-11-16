# Settings

`rw.RenderSettings` is a frozen dataclass of frozen groups, one per subsystem. There is one
default, `rw.RenderSettings()`, for every entry point (the Renderer, MuJoCo Warp, Newton, Isaac
Lab, mjlab, the viewer, the webui); `rw.FAST` is the documented field set for maximum training
throughput (`rw.RenderSettings().replace(rw.FAST)`), and the other looks are field recipes
([Recipes](#recipes)). Every one of them renders on prebuilt kernels: nothing compiles. The
exceptions, built on first use (or ahead with `Renderer.warm`) and cached: the motion output, lens
cameras' programs, and three default-tier variants (below and `performance.md`).

```python
import raywarp as rw
import raywarp.integrations.mujoco
import raywarp.formats.usd

s = rw.RenderSettings(aa=rw.settings.AntiAliasing(spp=2), shadows=rw.settings.Shadows(samples=4))  # whole groups
s = s.replace(**{"shadows.samples": 4, "exposure.mode": "manual", "exposure.ev100": 12.0})  # or "group.field" keys
fast = rw.RenderSettings().replace(rw.FAST).replace(outputs=rw.Outputs.of("rgb", "depth"))
print(rw.RenderSettings().diff(fast))  # the fields that differ, as "group.field" -> value
```

`settings.flat()` lists every field as `"group.field" -> value`; `RenderSettings.fields_table()`
lists each field's kind (below). Invalid values raise `ValueError` at construction, naming the
field and the allowed values.

## Compiled in or runtime

Every discrete setting is **compiled in**: it is resolved, with the scene's capabilities, into a
`Features` key (`raywarp/core/build/variants/resolve.py`). Continuous values are **runtime**: they reach the
kernels through a small parameter struct or per-world tables, and changing them never recompiles.
Some fields also rebuild a baked structure (the GI / sky probe volume, the light grid) on the next
render.

**Kernel tiers.** The kernels compiled are a few fixed **tiers** (`raywarp/core/build/variants/tiers.py`): the
default's (`default`), the fast field set's (`fast`) and unlit shading's (`unlit`), each with a few
variants (2 x 2 supersampling, sampled shadows, the caps and budgets read at run
time, ...; the geometry-only outputs share one unshaded kernel). A tier compiles its settings'
effects; each frame can turn two of them off without another kernel: GI (in scenes without a
lighting environment, where GI off shows the plain ambient) and traced reflections. So any
configuration that keeps a tier's compile-time settings, or only turns those switches off,
renders with that tier's kernels: switching between the default, `FAST` and the recipes,
toggling a switch or changing any runtime field never compiles, and a prebuilt
bundle (one per GPU architecture, downloaded on first use: only what a renderer uses) ships every tier. The outputs
(`outputs.*`, their dtypes and conventions) and the exposure mode are run-time data: every build
compiles every output store and each frame selects from an output word, so requesting another
output or dtype never changes the kernels (on OptiX a module of the same PTX; the bundle's module
pack holds the tiers' outputs and the fast tier's single-output changes). A compile-time value
no tier uses (e.g. `shadows.max_casters=8`, `shadows.samples=4`, fog, the grading) lands on a tier
variant that reads it at run time or compiles it in. The variants and programs the bundle leaves
out compile on first use: `outputs.motion` (but on the fast tier), the lens cameras' render
programs (but the fast tier's fisheye / OpenCV-distortion program), the sampled-shadow variants
(`shadows.samples` > 1 without the counted recipes' other values) and, on the default tier,
`reflections.shadows="sun"` and a caster budget below the default's (the "Zero compiles" section of `performance.md`).
`Renderer.explain()` says which: `kernels: tier 'default', switched off at run time: gi`, or
`kernels: own (outside the tiers; nearest 'default' differs in budget)`.

**Prebuilt or scene kernels (`performance.kernels`).** The default, `"prebuilt"`, serves the tier's
build resolved without the scene: counts, ids and table sizes are read at run time, and the scene
only picks one of the build's prebuilt variants (its feature classes: translucent surfaces, clear
coat, layered lobes, texture maps, environment, analytic primitives, ...). A new scene or a scene
change then compiles nothing, and with the bundle installed a cold start compiles nothing either; a
build missing from the bundle warns and compiles once (cached). `"scene"` compiles the settings'
own build specialised on this scene (its light count and kinds, its materials and shapes) on first
use, cached on disk: a few percent faster per frame (the prebuilt build is 7-13 % slower on RTX,
512 worlds, the `FAST` field set, over six interiors and robot scenes), at the cost of a compile for every
new scene or compile-time change. `Renderer.explain()` names the build (`kernels: scene ...`).

`Renderer.update_settings(**changes)` takes the same keys as `replace` and changes either kind;
a captured CUDA graph is stale afterwards (`RenderGraph.launch` raises `rw.StaleBinding`;
`render` runs one eager frame and records its graph again).

```python
import mujoco
import mujoco_warp as mjw

mjm = mujoco.MjModel.from_xml_string("""
<mujoco><worldbody>
  <light pos="0 0 2" dir="0 0 -1"/>
  <geom type="plane" size="2 2 0.1"/><geom type="sphere" pos="0 0 0.3" size="0.3" rgba="0.8 0.3 0.2 1"/>
  <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
</worldbody></mujoco>""")
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=2)
mjw.forward(m, d)
r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=2, cameras=["cam"], resolution=(32, 24))
r.render(d)
before = r.features
r.update_settings(**{"exposure.compensation": 0.5, "reflections.max_roughness": 0.3})  # runtime: same kernel
assert r.features == before
r.update_settings(**{"shadows.enabled": False})  # compiled in: another prebuilt build
assert r.features != before
```

Values that a kernel cannot see collapse to one `Features` (one build for all of them): every
shading field of a depth / segmentation-only render, shadow fields with shadows off, a shadow
budget covering every light, light caps where no capped loop
is built, and so on (`canonical.canonical`, tested by `tests/unit/test_canonical_features.py`). Values
`"auto"` resolve against the scene: `environment.enabled="auto"` is on when the scene has an
environment map that lights it, `transparency.enabled="auto"` when something is translucent,
`surfaces.textures="auto"` when a material binds a texture.

The table is generated from the field metadata (`python tools/settings_table.py --write`; the
docs test fails when it is stale), so it is the source of truth. Kinds: "runtime" (a parameter or
table, never a kernel), "output word" (run-time output selection: no kernel change), "tier switch"
(compile-time in the settings' own build, but compiled into every prebuilt tier build and switched
at run time), "compiled in" (a compile-time field, served by the prebuilt tiers as above):

<!-- settings-table:begin (generated by tools/settings_table.py from RenderSettings.fields_table(); do not edit) -->
| setting | default | kind | a change also |
| --- | --- | --- | --- |
| `outputs.rgb` | `ColorOutput(channels=4, dtype='uint8', planar=False, alpha=False)` | output word |  |
| `outputs.depth` | `None` | output word |  |
| `outputs.seg` | `None` | output word |  |
| `outputs.semantic` | `None` | output word |  |
| `outputs.hdr` | `None` | output word |  |
| `outputs.albedo` | `None` | output word |  |
| `outputs.normal` | `None` | output word |  |
| `outputs.position` | `None` | output word |  |
| `outputs.uv` | `None` | output word |  |
| `outputs.material` | `None` | output word |  |
| `outputs.roughness` | `None` | output word |  |
| `outputs.metallic` | `None` | output word |  |
| `outputs.emission` | `None` | output word |  |
| `outputs.direct` | `None` | output word |  |
| `outputs.indirect` | `None` | output word |  |
| `outputs.shadow` | `None` | output word |  |
| `outputs.luminance` | `None` | output word |  |
| `outputs.motion` | `None` | compiled in |  |
| `aa.spp` | `1` | compiled in |  |
| `aa.post` | `False` | runtime |  |
| `shadows.enabled` | `True` | tier switch |  |
| `shadows.samples` | `1` | compiled in |  |
| `shadows.max_casters` | `4` | compiled in |  |
| `shadows.translucent` | `'transmittance'` | compiled in |  |
| `shadows.area_rays` | `0` | runtime |  |
| `lighting.area_lights` | `'ltc'` | compiled in |  |
| `lighting.light_grid` | `'auto'` | compiled in |  |
| `lighting.light_grid_cell` | `1.0` | runtime | rebuilds the light grid |
| `lighting.camera_light` | `'auto'` | compiled in |  |
| `lighting.fill` | `0.0` | runtime |  |
| `lighting.fill_color` | `(1.0, 1.0, 1.0)` | runtime |  |
| `lighting.emissive_lights` | `True` | tier switch |  |
| `lighting.max_exact` | `4` | compiled in |  |
| `lighting.visible_lights` | `True` | runtime |  |
| `lighting.contact_ao` | `0.0` | runtime |  |
| `environment.enabled` | `'auto'` | compiled in |  |
| `environment.background` | `True` | tier switch |  |
| `environment.background_color` | `(0.0, 0.0, 0.0)` | runtime |  |
| `reflections.enabled` | `True` | tier switch |  |
| `reflections.max_roughness` | `0.5` | runtime |  |
| `reflections.shadows` | `'off'` | compiled in |  |
| `reflections.max_exact` | `2` | compiled in |  |
| `transparency.enabled` | `'auto'` | tier switch |  |
| `transparency.max_layers` | `4` | runtime | rebuilds the GI probe volume |
| `gi.enabled` | `True` | compiled in | rebuilds the GI probe volume |
| `gi.spacing` | `None` | runtime | rebuilds the GI probe volume |
| `gi.max_probes` | `32768` | runtime | rebuilds the GI probe volume |
| `gi.bounds` | `None` | runtime | rebuilds the GI probe volume |
| `gi.rays` | `256` | runtime | rebuilds the GI probe volume |
| `gi.normal_bias` | `0.1` | runtime | rebuilds the GI probe volume |
| `gi.update_every` | `0` | runtime |  |
| `gi.bake_updates` | `8` | runtime |  |
| `gi.relight` | `'on_change'` | runtime |  |
| `openpbr.diffuse_roughness` | `'auto'` | compiled in |  |
| `openpbr.layers` | `'auto'` | compiled in |  |
| `surfaces.textures` | `'auto'` | tier switch |  |
| `surfaces.backface_culling` | `True` | tier switch |  |
| `surfaces.anisotropy` | `8` | compiled in |  |
| `surfaces.shading` | `'pbr'` | compiled in |  |
| `post.tone_mapper` | `'pbr_neutral'` | runtime |  |
| `atmosphere.color` | `(0.5, 0.5, 0.5)` | runtime |  |
| `atmosphere.start` | `0.0` | runtime |  |
| `atmosphere.end` | `0.0` | runtime |  |
| `atmosphere.haze` | `0.0` | runtime |  |
| `exposure.mode` | `'auto'` | runtime |  |
| `exposure.ev100` | `14.96578428466209` | runtime |  |
| `exposure.metering` | `'average'` | runtime |  |
| `exposure.key` | `0.18` | runtime |  |
| `exposure.compensation` | `0.0` | runtime |  |
| `exposure.min_ev` | `-6.0` | runtime |  |
| `exposure.max_ev` | `20.0` | runtime |  |
| `exposure.adaptation` | `'locked'` | runtime |  |
| `exposure.frame_time` | `0.03333333333333333` | runtime |  |
| `grading.enabled` | `False` | runtime |  |
| `grading.kelvin` | `6504.0` | runtime |  |
| `grading.tint` | `0.0` | runtime |  |
| `grading.lift` | `(0.0, 0.0, 0.0)` | runtime |  |
| `grading.gamma` | `(1.0, 1.0, 1.0)` | runtime |  |
| `grading.gain` | `(1.0, 1.0, 1.0)` | runtime |  |
| `grading.saturation` | `1.0` | runtime |  |
| `sensor.rolling_shutter` | `0.0` | runtime |  |
| `sensor.motion_blur` | `0.0` | runtime |  |
| `sensor.vignette` | `(0.0, 0.0, 0.0)` | runtime |  |
| `sensor.noise` | `False` | runtime |  |
| `sensor.full_well` | `4000.0` | runtime |  |
| `sensor.read_noise` | `3.0` | runtime |  |
| `sensor.bits` | `0` | runtime |  |
| `sensor.response` | `(1.0, 1.0, 0.5, 1.0)` | runtime |  |
| `sensor.aberration` | `0.0` | runtime |  |
| `sensor.depth` | `'off'` | runtime |  |
| `sensor.baseline` | `0.05` | runtime |  |
| `sensor.disparity_noise` | `0.08` | runtime |  |
| `sensor.disparity_step` | `0.03125` | runtime |  |
| `sensor.range_noise` | `(0.0012, 0.4, 0.0019)` | runtime |  |
| `sensor.min_depth` | `0.1` | runtime |  |
| `sensor.max_depth` | `10.0` | runtime |  |
| `sensor.dropout` | `0.0` | runtime |  |
| `sensor.edge_dropout` | `0.0` | runtime |  |
| `sensor.edge_threshold` | `0.05` | runtime |  |
| `performance.kernels` | `'prebuilt'` | compiled in |  |
| `performance.traversal` | `'auto'` | compiled in |  |
| `performance.render_order` | `'auto'` | runtime |  |
| `performance.rebuild_every` | `64` | runtime |  |
<!-- settings-table:end -->

Color settings (`lighting.fill_color`, `background_color`, `atmosphere.color`, `grading.lift` /
`gamma` / `gain`) take any three numbers: a tuple, a list or a (3,) NumPy array, stored as a
tuple of floats (settings stay hashable and compare by value).

The runtime fields, by group:

<!-- runtime-fields:begin (generated by tools/settings_table.py; do not edit) -->
- `aa`: `post`
- `shadows`: `area_rays`
- `lighting`: `light_grid_cell` (rebuilds), `fill`, `fill_color`, `visible_lights`, `contact_ao`
- `environment`: `background_color`
- `reflections`: `max_roughness`
- `transparency`: `max_layers` (rebuilds)
- `gi`: `spacing` (rebuilds), `max_probes` (rebuilds), `bounds` (rebuilds), `rays` (rebuilds), `normal_bias` (rebuilds), `update_every`, `bake_updates`, `relight`
- `post`: `tone_mapper`
- `atmosphere`: `color`, `start`, `end`, `haze`
- `exposure`: `mode`, `ev100`, `metering`, `key`, `compensation`, `min_ev`, `max_ev`, `adaptation`, `frame_time`
- `grading`: `enabled`, `kelvin`, `tint`, `lift`, `gamma`, `gain`, `saturation`
- `sensor`: `rolling_shutter`, `motion_blur`, `vignette`, `noise`, `full_well`, `read_noise`, `bits`, `response`, `aberration`, `depth`, `baseline`, `disparity_noise`, `disparity_step`, `range_noise`, `min_depth`, `max_depth`, `dropout`, `edge_dropout`, `edge_threshold`
- `performance`: `render_order`, `rebuild_every`
<!-- runtime-fields:end -->

Also runtime, outside `RenderSettings`: cameras' clip planes (`Renderer.set_clip`), camera
rays (`set_camera_rays`), the semantic table (`set_semantics`), the rendered worlds
(`render(active=...)`), every per-world table (`Renderer.per_world`, [robotics](robotics.md)) and
the MuJoCo model's aliased arrays (`m.geom_rgba`, `m.geom_matid`, ...).

## Default

`rw.RenderSettings()`, every group at its own defaults; the measured reasons:

| group | default | why |
| --- | --- | --- |
| `lighting` | `area_lights="ltc"`, `max_exact=4`, `light_grid="auto"` | LTC area lights; the 4 strongest lights of a point shaded exactly, the others' irradiance diffusely (a large interior, L4 16 x 256^2: -44 % frame time; an open-plan interior's FLIP 0.06 vs every light exact); the light grid from 8 lights |
| `shadows` | `max_casters=4`, `samples=1`, `translucent="transmittance"` | one fixed center shadow ray for each of the 4 strongest casters; tinted sun shadows through glass |
| `reflections` | `enabled=True`, `max_exact=2`, `shadows="off"` | glossy reflections along the GGX dominant direction; 2 lights at reflection hits, unshadowed (`shadows="sun"` keeps sunlight out of indoor reflections for +10-49 % frame time on RTX) |
| `transparency` | `max_layers=4` | 4 translucent layers |
| `surfaces` | `anisotropy=8` | anisotropic texture filtering, 8 taps at grazing angles (+2-5 % on an L4) |
| `aa` | `spp=1` | deterministic per frame: an observation never depends on earlier frames |
| `gi` | on | probe GI resampled into the ambient cube (0.5 m / 256 rays / 8 updates; 16 or 32 updates matched the references no better); vs Mitsuba at 512 px mean FLIP: window-lit room 0.70 -> 0.12, closed workcell 0.55 -> 0.14, sunlit tabletop 0.071 -> 0.038 |
| `post` | `tone_mapper="pbr_neutral"` | PBR Neutral keeps base colors (RTX's ACES approximation turned HDRI-lit tabletops milky, +0.3..+0.9 EV, FLIP 0.39-0.48) |
| `exposure` | `mode="auto"`, `adaptation="locked"` | automatic exposure, locked after the first metered frame per world and camera |

The environment map's sun is a shadowed directional light by default (`SceneOptions.env_sun`, an
upload option). Single-field opt-ins served by the prebuilt kernels (no compile):
the light caps and the caster budget up to 16 (on `FAST` and the default). `shadows.samples` alone, and on the
default `aa.spp=2`, `reflections.shadows="sun"` and a caster budget below its own route to tier
variants the bundle leaves out:
their first use compiles (NVRTC, about 10-25 min, cached; `Renderer.warm` ahead), unless they come
with the recipes' other values (quality, reference, RTX look are prebuilt; `performance.md`, "Not
included in the prebuilt kernels"). One more exception on the software traversal (A100 / H100, CPU): its megakernel
compiles the sample loop of the motion output's body, so `aa.spp >= 3`, or `aa.spp=2` together
with `outputs.motion`, compiles that configuration's own kernels on its first frame
(seconds to a minute, once per machine: Warp's kernel cache); `Renderer.warm` ahead of the run
compiles them off the critical path. On OptiX `aa.spp` is run-time supersampling and every value
is served.

Frame time against the fast field set (RTX PRO 6000, 512^2 x 512 worlds): Office 2.4x, other
interiors 2.7-3.1x, the showcase room 1.6x, Sponza 1.4x, robot / HDRI tabletops 1.2-1.3x.

## Fast field set

`rw.FAST` is the field set for maximum training throughput, still PBR:

```python
fast = rw.RenderSettings().replace(rw.FAST)
print(dict(rw.FAST))
```

| field | `FAST` | default |
| --- | --- | --- |
| `lighting.area_lights` | `"point"` | `"ltc"` |
| `lighting.max_exact` | `None` (every light exact: cheaper than the cap's ranking for point lights) | 4 |
| `lighting.visible_lights` | `False` (area lights in view show no emitting surface; RTX: -1.4 to -2.2 % on Franka, Office, the showcase room, images identical where no light is in view) | `True` |
| `shadows.max_casters` | 2 | 4 |
| `shadows.translucent` | `"off"` (opaque shadows) | `"transmittance"` |
| `reflections.enabled` | `False` (a MuJoCo `reflectance` stays a specular F0; its ray: +50 % GPU) | `True` |
| `reflections.max_exact` | `None` | 2 |
| `transparency.max_layers` | 2 | 4 |
| `surfaces.anisotropy` | 1 (isotropic filtering) | 8 |

Measured against the default (RTX PRO 6000, 512^2 x 512 worlds): Office -58 % frame time (70.7
vs 169.7 ms), other interiors -63..-67 %, the showcase room -38 %, Sponza -29 %, robot / HDRI
tabletops about -20 %. The bundle holds its own build (`fast` tier), so `FAST` compiles nothing.

A run-time speed knob on the default: `reflections.enabled=False` (Office -29 %, 169.7 -> 119.8
ms; the image is the same apart from the reflections). `gi.enabled=False` on the default is
within run-to-run drift on static scenes; on `FAST` it saves 3.5-3.8 % (Office, the showcase room)
to 12.7 % (Franka) (RTX PRO 6000, 512 worlds), and under domain randomisation
also the relight.

## Recipes

The other looks are plain field changes of the default. Each routes to a prebuilt tier (zero
compiles: `python -m tools.prebuilt.zero holes` walks them; `tools/recipes.py` holds the same
table for the tooling).

<!-- SHOWCASE: settings/recipes.webp (docs/assets/SHOTS.md) -->
![unlit, FAST, default, quality](assets/settings/recipes.webp)

```python
# Shape and color, no lighting cues: the base color (material x texture x instance color) as is,
# plus the emission; no lights, shadows, environment, reflections or GI (1.5-13x faster than PBR),
# isotropic texture filtering (the unlit tier's own; anisotropy > 1 takes the sampled variant).
unlit = rw.RenderSettings().replace({"surfaces.shading": "unlit", "surfaces.anisotropy": 1})

# Evaluation, dataset generation, high-resolution policies: 2 x 2 samples per pixel, 2 fixed
# soft-shadow rays for 16 casters, 8 lights exact (4 at reflection hits), sun shadows in reflections.
quality = rw.RenderSettings().replace(
  {
    "aa.spp": 2,
    "shadows.samples": 2,
    "shadows.max_casters": 16,
    "lighting.max_exact": 8,
    "reflections.max_exact": 4,
    "reflections.shadows": "sun",
    "transparency.max_layers": 8,
  }
)

# Per-frame comparison target (a finite-sample approximation, not physical ground truth): 2 x 2
# samples per pixel, 8 fixed soft-shadow rays for every caster, every light exact.
reference = rw.RenderSettings().replace(
  {
    "aa.spp": 2,
    "shadows.samples": 8,
    "shadows.max_casters": 16,
    "lighting.max_exact": None,
    "reflections.max_exact": None,
    "reflections.shadows": "sun",
    "transparency.max_layers": 8,
  }
)

# Side by side with Omniverse RTX RealTimePathTracing (Isaac Sim, Newton ViewerRTX): RTX's ACES
# approximation, clear glass shadows, per frame; the USD settings map's base.
rtx_look = rw.RenderSettings().replace(
  {
    "gi.normal_bias": 0.3,
    "lighting.max_exact": None,
    "lighting.light_grid_cell": 0.5,
    "reflections.max_exact": None,
    "reflections.shadows": "sun",
    "shadows.max_casters": 8,
    "shadows.translucent": "clear",
    "post.tone_mapper": "aces_rtx",
  }
)
```

Every recipe is deterministic per frame: an observation never depends on earlier frames.

Any settings take any `Outputs`: a depth / segmentation-only request (`rw.Outputs.of("depth",
"seg")`) compiles one geometry kernel whatever the shading settings. `Renderer.explain()` prints
what a settings object resolves to on the scene: its cost class (geometry, unlit, fast, default,
quality, converged) and every feature with the setting that asked for it and why it is on or off.

**Choosing.** Start from the observation the policy needs:

1. Depth / segmentation only: any settings with `outputs=rw.Outputs.of("depth", "seg")`.
2. Shape and color, no lighting cues: the unlit recipe, rendered at the policy's resolution.
3. RGB for training: the default; `FAST` for maximum throughput, then add back only what the task
   needs to see (what a mirror or a glossy floor reflects: `reflections.enabled=True`; soft
   shadows: `lighting.area_lights="ltc"`).
4. RGB for evaluation, demos, sim-to-real checks: the default (probe GI, its Mitsuba comparison
   in [Default](#default)); the quality recipe for spp 2.
5. Matching an Omniverse RTX render: the RTX look recipe.

## The groups: what each costs, when to turn it off

Costs are relative to the same frame without the feature; evidence paths in brackets.

**`outputs`.** One field per output (`rgb`, `depth`, `seg`, `semantic`, `hdr`, `albedo`,
`normal`, `position`, `uv`, `material`, `roughness`, `metallic`, `emission`, `direct`,
`indirect`, `shadow`, `luminance`, `motion`), each its typed spec or None: only requested
outputs are allocated and written (whether another output set compiles: [Compiled in or
runtime](#compiled-in-or-runtime)). `rw.Outputs.of("rgb", "depth")` requests by
name with the default specs, `rw.Outputs.parse` from plain data (JSON, a command line). The
specs carry the conventions: `DepthOutput(convention="planar" | "radial", miss="zero" | "inf" |
"far")` (MuJoCo Warp / Isaac Lab / `mujoco.Renderer`), `ColorOutput(channels=3, planar=True,
dtype="float16")` (the color channels-first for a policy's input, written by the kernel:
-0.3..+1.9 % frame time on an RTX PRO 6000; every layout world-major), `NormalOutput(frame="shading" |
"geometric")`, `ShadowOutput(light=)`, and `dtype="float16"` on every float output. Without a
shaded output (depth, seg, semantic, position, uv, material, a geometric normal, motion) the
render takes the geometry-only kernel: one closest hit per pixel, no shading. What each output
holds and what it costs: [robotics.md](robotics.md#outputs-what-you-can-request-and-what-it-costs).

**`aa`** (anti-aliasing). The guidance:

- `aa.spp=2` for small clean images: 2 x 2 samples everywhere (the best quality of every
  variant studied at 128 x 128).
- Every frame renders on its own (no temporal accumulation, reprojection or denoiser): a frame
  never depends on earlier ones.
- Specular anti-aliasing is always on: it widens roughness by the normal variance in the pixel
  (cheap; keeps glints from flickering).
- `aa.post=True` (off by default, runtime) runs FXAA 3.11 (quality) on the final color across
  the geometric edges only: where the depth output is not one plane over a pixel and its
  neighbours (silhouettes, the sky's edge, thin wires, alpha cut-outs, creases). It smooths those
  at one sample per pixel and leaves texture and shadow edges exactly as rendered (a checker
  floor keeps its lines). It needs a `depth` output (ValueError otherwise: the depth is the edge
  source; float32 finds shallower creases than float16) and an `rgb` or `luminance` output, and
  runs in the `sensor` pass before its other effects: two launches over the image that read the
  depth and the display in place, no copy of the image. Off: no pass, no memory; switching
  never compiles.

<!-- SHOWCASE: settings/aa.webp (docs/assets/SHOTS.md) -->
![Anti-aliasing: aa.spp=1, aa.post (FXAA), aa.spp=2 (2x crops)](assets/settings/aa.webp)

**`shadows`.** One shadow ray per light per sample by default, only for the `max_casters`
strongest casters of each point (the rest scaled by the traced ones' visibility).
`samples > 1` samples soft shadows from the light's size (ray cost grows with the sample count).
A small budget leaves visible variance at partially occluded edges. Area lights and finite-angle
suns use fixed targets: the light center at one sample, producing a hard shadow, else a fixed
low-discrepancy set shifted per pixel and light (larger sets can still show spatial sampling
error). Punctual bulb shadows use random points per pixel when `samples > 1`.
Off (`enabled=False`) when the task does
not need shadows: the largest single saving of a lit render after reflections.

<!-- SHOWCASE: settings/shadows.webp (docs/assets/SHOTS.md) -->
![Shadows: hard (FAST), soft LTC area light, shadows.samples=8](assets/settings/shadows.webp)

`area_rays` (0 = off, the default) softens the shadows of large, near area lights by a pass after
the render (`core/shade/shadows/area.py`): per pixel, the pass shades the pixel's surface as the
render does (its camera ray traced again: every lobe and layer), and for each area light of angular
radius >= 0.05 rad that brings at least 1 / `max_casters` of those lights' light, `area_rays`
shadow rays to points of the light (interleaved over 5 x 5 pixel blocks, the visible share then
filtered over the pixel's surface) replace the centre ray's hard shadow. The render kernels are the
same with it on or off (off costs nothing: their PTX is byte-identical); on, the render also writes
the position (float32) and normal outputs and shows the image through the deferred display; it
needs `lighting.area_lights="ltc"` and `aa.spp=1` (ValueError otherwise). Its
programs are in the prebuilt bundle (the fill lists them); turning it on creates its OptiX module
once. Cost on (RTX PRO 6000, 512 worlds x 512^2, default, 4 rays): about 2.5-3x the frame where
near area lights light most pixels.

**`lighting`.** `area_lights="ltc"` shades rect / disk / sphere / tube lights as area lights
(LTC highlights, soft shadows); `"point"` shades them as points (cheaper, `FAST`).
`max_exact` (default 4; None: every light, `FAST`; 8 in the quality recipe) shades only the strongest lights of a point exactly and the rest through their
diffuse irradiance (Office, 16 x 256 x 256, L4: -44 % frame time, FLIP 0.06 against every light
exact; Office candidate light entries per pixel 49.6 -> 3.2). `light_grid` limits each point to the lights its cell
can see (needed with dozens of lights; built on the first render, cached on disk). `"auto"` (the
default) builds it from 8 lights: below, its lookup costs more than it culls (robotics scenes
with 1-5 lights: off 0-4 % faster; 8 lights in 4 closed rooms: on 7 % faster at 512 px on an
L4); `True` forces it at any count.
`emissive_lights` turns flat emissive panels into rect lights. `camera_light` is MuJoCo's headlight, a
camera-attached light (`"auto"`: as the model says).

<!-- SHOWCASE: settings/contact_ao.webp (docs/assets/SHOTS.md) -->
![lighting.contact_ao off / on](assets/settings/contact_ao.webp)

`contact_ao` (0 = off, the default; up to 1) darkens the indirect light of the static scenery
around the dynamic objects: the per-world instances (a robot's links, grasped or randomised
objects), which the GI probes and the environment's visibility do not see. Each such instance is
a box proxy (a mesh's box, a primitive's extents; spheres stay spheres), posed every frame, and
each pixel's occlusion is the analytic form factor of the proxies around it (from the depth
output: contacts under objects, the robot's base and gripper on the table, at any distance on or
off screen). It is deterministic (no taps, no noise, no history), applied by the deferred display,
and adds a planar depth output if the outputs hold none. The dynamic objects themselves are not
darkened, and static contacts stay the probes'. Off it costs nothing (its code is in a display
kernel variant of its own, no proxies are gathered and no depth is written for it). On, at 512
worlds x 512^2 on an RTX PRO 6000: +0.6-0.8 ms where nothing moves (the depth output alone: Office,
the showcase room); on the Franka arm (about 60 moving parts) the first version measured +15 ms,
the grouped and tiled one about 3x less on the CPU (re-measured at the next RTX gate). The cost
grows with the moving parts near what the pixels see. It reconstructs points as a pinhole
camera's: orthographic, fisheye and distorted cameras get an approximate occlusion. Turning it on
without a requested depth changes the output word: the software path compiles that word's
megakernel once, OptiX creates its module (request a depth output to keep the word).

**`environment`.** Image-based lighting from the scene's environment map (`enabled`), the map as
background (`background`, else `background_color`). Without GI the environment lights every
surface unoccluded (the GI probes keep the sky out of rooms).

**`reflections`.** Traced reflections (on by default, off in `FAST`: a run-time switch): one
deterministic ray along the GGX dominant direction for surfaces smoother than `max_roughness`. A MuJoCo `reflectance` is the
material's specular F0, reflected like any glossy surface (MuJoCo's default shininess 0.5 is
roughness 0.5, at the default `max_roughness`: no ray; a mirror needs a low roughness). `shadows` at
reflection hits (`"sun"`: keeps sunlight out of indoor reflections), `max_exact` at reflection
hits. The traced reflection fades to the environment's specular term from half `max_roughness`;
the dominant ray stands for the lobe's sharp core only (its share of the GGX lobe: 99.8 % for
chrome at roughness 0.05, 39 % at 0.25), the rest is the blurred environment / GI term, so rough
metals do not read as mirrors.

<!-- SHOWCASE: settings/reflections.webp (docs/assets/SHOTS.md) -->
![reflections.enabled off / on](assets/settings/reflections.webp)

**`transparency`.** Translucent layers composited front to back, up to `max_layers` (run-time
data: the layer loops' bound; a change re-bakes the GI probes); compiled in only when the scene
has something translucent (`"auto"`). Turning them off on such a scene renders every surface
opaque with the same kernels (a run-time setting of the prebuilt tier build). Past the budget the last surface
closes the ray and the background along the ray (not traced: environment or background color)
shows through its transmittance, so stacked panes stay see-through at 2 layers (`FAST`).
A closed pane's or a solid's exit face is no layer: two layers see what is behind a glass block.

`shadows.translucent` lets the shadow rays of every light through translucent casters, attenuated
by each caster's transmittance once (one terminating walk per ray: free where a ray meets none).

**`gi`.** Shared irradiance probes (DDGI-style), baked from static scenery (or updated every
`update_every` frames): bounce light, emissive surfaces lighting the room. On by default and
in `FAST` (probe spacing derived from the scene, about 0.5 m in a furnished room; 256 rays, 8 bake
updates): against Mitsuba (512 px) it cut the mean FLIP of a window-lit room 0.70 -> 0.12, a
closed workcell 0.55 -> 0.14 and a sunlit tabletop 0.071 -> 0.038 for a frame cost within noise. Worlds whose lights, materials,
appearance and environment are equal share one set of probe maps; randomised worlds get their
own (K stacked sets for K variants; under continuous randomisation one shared volume tinted per
world), found from the data and tracked every frame (`raywarp/core/shade/gi/groups.py`).
Dynamic/per-world geometry and deformables do not enter the bake or its placement bounds. Its
field changes re-bake the volume.

<!-- SHOWCASE: settings/gi.webp (docs/assets/SHOTS.md) -->
![gi.enabled off / on](assets/settings/gi.webp)

`gi.relight` (runtime) decides when a world of continuous randomisation (the shared volume tinted
per world) whose GI inputs changed is re-lit by rays. `"on_change"` (the default) re-lights it on
the frame of the change. `"on_request"` keeps its tint until `Renderer.request_gi_relight(worlds)`
names it (a (nworld,) bool device mask: one launch, no host read, captured graphs stay valid),
e.g. once per episode reset of a dataset generator that randomises materials every frame. In
both, changes of the lights' colours, intensities and on / off state and of the environment
intensity are recombined from the last re-light without rays, and the exposure is no GI input
(it scales the image after the GI); material, light position and environment map changes need
rays.

*GI in `FAST` (on).* Without GI, `FAST` would light only what the lights and
the sky reach directly: interiors lit through their openings get black
ceilings and arcades (Office, Sponza). On static scenes the probes cost `FAST` 3.5-3.8 % on
Office and the showcase room and 12.7 % on Franka (RTX PRO 6000, 512^2 x 512 worlds, GI off vs on;
an earlier build measured no difference); the first frame bakes, and every world whose lights / materials are
randomised gets its own bake. Turn it off for open outdoor scenes, or to skip the bake:

```python
outdoor_fast = rw.RenderSettings().replace(rw.FAST).replace(gi={"enabled": False})
assert rw.RenderSettings().replace(rw.FAST).diff(outdoor_fast) == {"gi.enabled": False}
```

The default with and without GI at 512^2 x 512 worlds (RTX PRO 6000): Office 198.1 vs 198.3 ms,
Bistro 288.0 vs 287.2, Hospital 176.5 vs 176.3 (within run-to-run drift); +0.3 GiB of device memory
for the probe volume.

**`surfaces`.** `textures` (`"auto"`), `backface_culling`, `anisotropy` (8
texture taps at grazing angles against moire, +2-5 % on an L4, B66), `shading="unlit"` (the
unlit recipe).

**`post`.** `tone_mapper` (`"pbr_neutral"` Khronos PBR Neutral, which keeps base colors: the default;
`"aces_rtx"` RTX's ACES approximation, the RTX look recipe's: closest to
Omniverse RTX renders, but it turns HDRI-lit tabletops milky (+0.3..+0.9 EV);
`"aces"`; `"clamp"`), runtime.

<!-- SHOWCASE: settings/tone_mapper.webp (docs/assets/SHOTS.md) -->
![post.tone_mapper: pbr_neutral, aces_rtx, aces](assets/settings/tone_mapper.webp)

**`exposure`, `grading`.** `exposure.mode`: `"auto"` (the default: each camera image
metered: middle grey 0.18 on the 10-90 % histogram band, highlights protected, temporal adaptation
reset per world) or `"manual"` (`exposure.ev100`, the camera exposure, higher is darker, also per
world `"exposure"`); switching between the two or changing the number never recompiles; `grading` (white balance, the ASC
CDL, saturation; applied by the deferred display pass, its values per world). Lateral chromatic
aberration (`sensor.aberration`, run in the deferred display: 0: off; per world, a
camera-randomisation knob; with manual exposure a nonzero value turns the deferred display on),
lens vignetting, sensor noise and the camera response are the `sensor` group's.

**`sensor`** (all run time, all off by default: off runs no pass and allocates nothing). A pass after
the display, on the final `rgb` / `luminance` and `depth` outputs (the scene-referred outputs stay
clean); its kernels are prebuilt with every build's, so switching an effect never compiles.
Camera: `rolling_shutter`, `motion_blur` (along `outputs.motion`), `vignette`, `noise` (Poisson shot
noise of `full_well` electrons at white plus `read_noise` electrons rms, seeded per world, pixel
and frame), `bits` (the ADC's quantisation of the linear signal; 0: none), `response` (the camera
response curve). Per world: `env=["sensor.noise", "sensor.vignette", "sensor.response"]`.
Depth: `depth="stereo"` (an active-stereo pair: disparity noise and quantisation, the occlusion
band, RealSense D400-like defaults) or `depth="range"` (time of flight / structured light:
noise of sigma0 + k (z - z0)^2 metres, Kinect v1's fit by default, any camera), with `min_depth`,
`max_depth`, random `dropout`, and `edge_dropout` (pixels next to a depth jump of more than
`edge_threshold` times their depth, the sensors' flying pixels). Cost: see `docs/performance.md`.

**`atmosphere`.** Distance fog and exponential haze: on where `end > start` or `haze > 0` (the
default 0 / 0 / 0 is off), or where `fog.range` is declared per world; color, range and haze are
runtime and per world (a domain-randomisation knob; every tier build compiles the fog in, off at
run time: free on the RTX PRO 6000).

<!-- SHOWCASE: settings/fog.webp (docs/assets/SHOTS.md) -->
![atmosphere off / fog / haze](assets/settings/fog.webp)

**`performance`** (never changes pixels). `kernels` (`"prebuilt"` or `"scene"`, above), `traversal` (`"auto"`: OptiX on GPUs with RT
cores, the software BVH elsewhere), `render_order` (`"world"` helps when every world renders nearly the same
image), `rebuild_every` (rebuild the per-world trees every N frames, default 64, instead of refitting them
forever: refit trees degrade as objects travel; 0: refit only).

Not settings (never change pixels): the OptiX register caps, fixed per tier family (the default
128; `FAST` 96, and 80 for its builds without clear coat, layered lobes or light culling; the
unlit tier 96; geometry-only builds 64), and pure launch tuning (block size, the Warp register
cap, the BVH builder, OptiX micromaps, the validation cadence): measured constants in
`raywarp/core/pipeline/platform/tuning.py`, which benchmarks override with `tuning.override(...)`. The debug
build counting lights, rays and layers per pixel (`Renderer.counters()`) is
`tuning.override(counters=True)` around the Renderer's creation.

## Texture storage (GPU memory)

Material textures are the largest GPU allocation of big scenes. Each texture stores only its
distinct channels (no constant-1 or copied channels), textures of equal texels are stored
once, and a texture whose every texel is one value is stored as a single texel (sampled exactly
as before). `rw.upload(host, options=rw.SceneOptions(textures=raywarp.core.pipeline.textures.formats.TextureStorage(...)))` picks how 8-bit sRGB color is stored
and caps the resolution; data textures (ORM, opacity, roughness, linear color) and normal maps
are always exact (8-bit texels as uint8, float textures as float16):

- `color` (8-bit sRGB base color, emission): `"srgb8"` (default on CUDA: 8-bit sRGB RGBA, 4 bytes
  / texel, 1 for a grey image without alpha, decoded by the texture unit per texel before the filter, so filtering stays linear and
  mip 0 is exact; the mips are filtered in linear values) or `"float16"` (linear, 8 bytes / texel
  for 4 channels: exact).
- `max_size=N`: textures larger than N texels (longer side) are halved until they fit, as if
  their finest mips were dropped. Default `raywarp.core.pipeline.textures.formats.MAX_TEXTURE_SIZE` (1024), the same cap the Newton
  and USD adapters apply at decode (`SceneOptions.textures` is the one knob), so MuJoCo models
  (whose textures arrive decoded in `MjModel`) are capped alike; `max_size=0` keeps every level.

On the CPU 8-bit color is float16 (the CPU has no sRGB texture read). HDR color (texels outside
[0, 1]) is always exact float16. `raywarp.core.pipeline.textures.formats.EXACT_STORAGE` keeps every stored texel exact (same size
cap).

Block compression (BC7 / BC4 / BC5) is not offered: on an L4 (the default, 128 worlds x 512^2) it
saved 1.1-1.2 GB of fixed memory on the Isaac Office and Bistro (about 5 % of the worlds that fit
in 24 GB) at the same frame time, but changed Bistro visibly (2.3 % of pixels more than 16 levels
off, FLIP 0.035 against exact) and its encode made the cold upload 2.6x slower; 8-bit sRGB color
stays within 3 levels of exact everywhere.

Every built texture's levels are cached on disk by content (`$RAYWARP_CACHE_DIR/textures`; an
empty value turns the cache off), with an entry per texture keyed by its every input (source
images, channels, color spaces, wraps, storage, the code), so a repeat upload of the same scene
reads the stored levels instead of converting, packing and filtering the images again.

The option is `upload`'s (host scenes: USD, `SceneBuilder`, Newton's `scene`) and, as
`options=rw.SceneOptions(textures=...)`, `rw.integrations.mujoco.scene`'s.

```python
host = rw.SceneBuilder().build()
gpu = rw.upload(host, options=rw.SceneOptions(textures=raywarp.core.pipeline.textures.formats.EXACT_STORAGE))
```

## USD stages

`rw.formats.usd.settings(stage)` maps a stage's Omniverse render settings (`rtx:*` in the layer's
`renderSettings` metadata, `omni:rtx:*` on RenderSettings / RenderProduct prims) onto a
`RenderSettings`: the base settings from `rtx:rendermode` (RealTimePathTracing and PathTracing ->
the RTX look recipe, also when unset; RaytracedLighting -> the default; MinimalRendering ->
`FAST`; or your own `base=`), then the authored shadows,
reflections, translucency, GI, sampled lighting, tone mapper, face culling and
background, camera exposure (f-stop, shutter, ISO -> a manual `exposure.ev100`). It is explicit: a stage's
settings never apply by themselves. Keys it ignores (AA, DLSS, denoiser, caches) go to
`report=` (a `rw.scene.LoadReport`). Its keys never touch `outputs` or `performance`.

```python
# docs: requires pxr
from pxr import Usd

stage = Usd.Stage.CreateInMemory()
stage.GetRootLayer().customLayerData = {"renderSettings": {"rtx:rendermode": "RaytracedLighting"}}
assert rw.formats.usd.settings(stage) == rw.RenderSettings()
```
