# Settings

Choose how a frame looks and what it costs. `rw.RenderSettings` holds every field, grouped by
subsystem. Every entry point (Newton, Isaac Lab, MuJoCo Warp, mjlab, the viewer) uses the same
default.

## Quick start

```python
import raywarp as rw
import raywarp.integrations.mujoco
import raywarp.formats.usd

s = rw.RenderSettings(aa=rw.settings.AntiAliasing(spp=2), shadows=rw.settings.Shadows(samples=4))  # whole groups
s = s.replace(**{"shadows.samples": 4, "exposure.mode": "manual", "exposure.ev100": 12.0})  # or "group.field" keys
fast = rw.RenderSettings().replace(rw.FAST).replace(outputs=rw.Outputs.of("rgb", "depth"))
print(rw.RenderSettings().diff(fast))  # the fields that differ, as "group.field" -> value
```

- `RenderSettings` is frozen: `replace` returns a new value.
- `settings.flat()` lists every field as `"group.field" -> value`.
- An invalid value raises `ValueError` at construction, naming the field and the allowed values.
- Color fields take any three numbers (tuple, list or (3,) array); they are stored as a tuple.

> **New in 0.4.0:** temporal accumulation is gone. Every frame renders on its own and is
> deterministic per pixel and seed: an observation never depends on earlier frames.

## Choose settings for your task

Start from the observation the policy needs:

| task | settings |
| --- | --- |
| depth / segmentation only | any settings with `outputs=rw.Outputs.of("depth", "seg")`: one closest hit per pixel, no shading |
| shape and color, no lighting cues | the [unlit recipe](#recipes), at the policy's resolution |
| RGB for training | `rw.FAST`, then add back only what the task must see (mirrors and glossy floors: `reflections.enabled=True`; soft shadows: `lighting.area_lights="ltc"`) |
| RGB for evaluation, demos, sim-to-real checks | the [default](#default); the quality recipe for 2 x 2 samples per pixel |
| matching an Omniverse RTX render | the RTX look recipe |

The default, `FAST` and every recipe render on prebuilt kernels: switching between them compiles
nothing ([how the kernels work](performance.md#how-the-kernels-work)). `Renderer.explain()` prints
what a settings object resolves to on a scene: its cost class (geometry, unlit, fast, default,
quality, converged) and each feature with the setting that turned it on or off.

### Default

`rw.RenderSettings()`, every group at its own defaults, chosen by measurement:

| group | default | why |
| --- | --- | --- |
| `lighting` | `area_lights="ltc"`, `max_exact=4`, `light_grid="auto"` | LTC area lights. The 4 strongest lights of a point are shaded exactly, the others diffusely (large interior, L4, 16 x 256^2: -44 % frame time, FLIP 0.06 vs every light exact). The light grid from 8 lights. |
| `shadows` | `max_casters=4`, `samples=1`, `translucent="transmittance"` | one fixed shadow ray to each of the 4 strongest casters; tinted sun shadows through glass |
| `reflections` | `enabled=True`, `max_exact=2`, `shadows="off"` | glossy reflections, 2 unshadowed lights at reflection hits (`shadows="sun"` keeps sunlight out of indoor reflections for +10-49 % frame time on RTX) |
| `transparency` | `max_layers=4` | 4 translucent layers |
| `surfaces` | `anisotropy=8` | anisotropic texture filtering, 8 taps at grazing angles (+2-5 % on an L4) |
| `aa` | `spp=1` | one sample per pixel |
| `gi` | on | probe GI (0.5 m, 256 rays, 8 updates). Mean FLIP vs Mitsuba at 512 px: window-lit room 0.70 -> 0.12, closed workcell 0.55 -> 0.14, sunlit tabletop 0.071 -> 0.038. |
| `post` | `tone_mapper="pbr_neutral"` | keeps base colors (RTX's ACES approximation turned HDRI-lit tabletops milky: +0.3 to +0.9 EV) |
| `exposure` | `mode="auto"`, `adaptation="locked"` | automatic exposure, locked after the first metered frame per world and camera |

An environment map's sun is a shadowed directional light by default (`rw.SceneOptions(env_sun=...)`,
an upload option). Frame time against `FAST` (RTX PRO 6000, 512 worlds x 512^2): Office 2.4x,
other interiors 2.7-3.1x, the showcase room 1.6x, Sponza 1.4x, robot / HDRI tabletops 1.2-1.3x.

### Fast field set

`rw.FAST` is the field set for maximum training throughput, still PBR:

```python
fast = rw.RenderSettings().replace(rw.FAST)
print(dict(rw.FAST))
```

| field | `FAST` | default |
| --- | --- | --- |
| `lighting.area_lights` | `"point"` | `"ltc"` |
| `lighting.max_exact` | `None` (every light exact: cheaper than ranking for point lights) | 4 |
| `lighting.visible_lights` | `False` (area lights show no emitting surface; -1.4 to -2.2 % on RTX) | `True` |
| `shadows.max_casters` | 2 | 4 |
| `shadows.translucent` | `"off"` (opaque shadows) | `"transmittance"` |
| `reflections.enabled` | `False` (a MuJoCo `reflectance` stays a specular F0) | `True` |
| `reflections.max_exact` | `None` | 2 |
| `transparency.max_layers` | 2 | 4 |
| `surfaces.anisotropy` | 1 (isotropic filtering) | 8 |

Against the default (RTX PRO 6000, 512 worlds x 512^2): Office -58 % frame time (70.7 vs 169.7
ms), other interiors -63 to -67 %, the showcase room -38 %, Sponza -29 %, robot / HDRI tabletops
about -20 %.

Two run-time speed knobs:

- `reflections.enabled=False` on the default: Office -29 % (169.7 -> 119.8 ms). Only the
  reflections change.
- `gi.enabled=False`: on `FAST` it saves 3.5-3.8 % (Office, showcase room) to 12.7 % (Franka). On
  the default it is within run-to-run drift on static scenes. Keep GI on indoors
  ([dark interiors](faq.md#dark-interiors)); turn it off for open outdoor scenes:

```python
outdoor_fast = rw.RenderSettings().replace(rw.FAST).replace(gi={"enabled": False})
assert rw.RenderSettings().replace(rw.FAST).diff(outdoor_fast) == {"gi.enabled": False}
```

### Recipes

The other looks are plain field changes of the default. Each is prebuilt.

<!-- SHOWCASE: settings/recipes.webp (docs/assets/SHOTS.md) -->
![unlit, FAST, default, quality](assets/settings/recipes.webp)

```python
# Shape and color, no lighting cues: base color x texture x instance color, plus emission.
# No lights, shadows, environment, reflections or GI (1.5-13x faster than PBR); isotropic filtering.
unlit = rw.RenderSettings().replace({"surfaces.shading": "unlit", "surfaces.anisotropy": 1})

# Evaluation, dataset generation, high-resolution policies: 2 x 2 samples per pixel, 2 soft-shadow
# rays for 16 casters, 8 lights exact (4 at reflection hits), sun shadows in reflections.
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

# Per-frame comparison target (a finite-sample approximation, not ground truth): 2 x 2 samples
# per pixel, 8 soft-shadow rays for every caster, every light exact.
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
# approximation, clear glass shadows; the base of the USD settings mapping.
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

Single-field changes off these recipes are mostly prebuilt too. The few that compile once on first
use (for example `shadows.samples` alone, or `aa.spp=2` on the default) are listed in
[Zero compiles](performance.md#zero-compiles).

## Compiled in or runtime

Change settings on a live renderer with `Renderer.update_settings(**changes)`. It takes the same
keys as `replace`.

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

Each field has a kind, shown in the [field table](#all-fields):

| kind | changing it |
| --- | --- |
| runtime | a number read every frame: never compiles |
| output word | picks which outputs a frame writes: never compiles |
| tier switch | an effect every prebuilt build carries, turned on or off per frame |
| compiled in | picks another prebuilt build (or, off the documented settings, compiles once) |
| "rebuilds ..." | also re-bakes the GI probes or the light grid on the next render |

The terms are explained once in [how the kernels work](performance.md#how-the-kernels-work).

> **Requires an RT-core GPU** (OptiX) for output changes without any compile: on the software
> path (A100, H100, CPU) a new output set can compile its kernel once.

- After `update_settings`, a captured CUDA graph is stale: `RenderGraph.launch` raises
  `rw.StaleBinding`, and `render` runs one eager frame and records again.
- Values a kernel cannot see collapse to one build: shading fields of a depth-only render, shadow
  fields with shadows off, a shadow budget covering every light, and so on.
- `"auto"` values resolve against the scene. `environment.enabled="auto"` is on when an environment
  map lights the scene; `transparency.enabled="auto"` when something is translucent;
  `surfaces.textures="auto"` when a material binds a texture.
- `performance.kernels="prebuilt"` (the default) serves every scene from the same builds, so a new
  scene compiles nothing. `"scene"` compiles a build specialised on this scene, once (cached): a
  few percent faster per frame (prebuilt is 7-13 % slower on RTX, 512 worlds, `FAST`, six
  interiors and robot scenes), at the cost of a compile per new scene.

Also run time, outside `RenderSettings`: camera clip planes (`Renderer.set_clip`), camera rays
(`set_camera_rays`), the semantic table (`set_semantics`), the rendered worlds
(`render(active=...)`), every per-world table (`Renderer.per_world`, [robotics](robotics.md)) and
the MuJoCo model's aliased arrays (`m.geom_rgba`, `m.geom_matid`, ...).

## What each group does

Costs are relative to the same frame without the feature.

**`outputs`.** One field per output, each a typed spec or None. Only requested outputs are
allocated and written. `rw.Outputs.of("rgb", "depth")` requests by name; `rw.Outputs.parse`
reads plain data (JSON, a command line). The specs carry the conventions:

| spec | options |
| --- | --- |
| `DepthOutput` | `convention="planar" \| "radial"`, `miss="zero" \| "inf" \| "far"` (MuJoCo Warp / Isaac Lab / `mujoco.Renderer`) |
| `ColorOutput` | `channels=3, planar=True, dtype="float16"`: channels-first for a policy, written by the kernel (-0.3 to +1.9 % frame time, RTX PRO 6000) |
| `NormalOutput` | `frame="shading" \| "geometric"` |
| `ShadowOutput` | `light=` |
| every float output | `dtype="float16"` |

Without a shaded output (only depth, seg, semantic, position, uv, material, a geometric normal,
motion), the render takes the geometry-only kernel. What each output holds and costs:
[robotics](robotics.md#outputs-what-you-can-request-and-what-it-costs).

**`aa`** (anti-aliasing).

- `aa.spp=2`: 2 x 2 samples per pixel everywhere. The best quality of every variant studied at
  128 x 128.
- No temporal accumulation, reprojection or denoiser: a frame never depends on earlier ones.
- Specular anti-aliasing is always on. It widens roughness by the normal variance in the pixel,
  which keeps glints from flickering.
- `aa.post=True` (run time, off by default) runs FXAA on geometric edges only: silhouettes, the
  sky's edge, thin wires, cut-outs, creases. Texture and shadow edges stay as rendered. It needs a
  `depth` output (float32 finds shallower creases than float16) and an `rgb` or `luminance`
  output; otherwise `ValueError`. Off, it costs nothing.

<!-- SHOWCASE: settings/aa.webp (docs/assets/SHOTS.md) -->
![Anti-aliasing: aa.spp=1, aa.post (FXAA), aa.spp=2 (2x crops)](assets/settings/aa.webp)

**`shadows`.** One shadow ray per light per sample, for the `max_casters` strongest casters of each
point. The others are scaled by the traced ones' visibility.

- `samples > 1` samples soft shadows over the light's size. Ray cost grows with the count.
- Area lights and sized suns use fixed targets: the light's center at one sample (a hard shadow),
  else a fixed pattern shifted per pixel and light. Point-bulb shadows use random points.
- `enabled=False` is the largest single saving of a lit render after reflections.

<!-- SHOWCASE: settings/shadows.webp (docs/assets/SHOTS.md) -->
![Shadows: hard (FAST), soft LTC area light, shadows.samples=8](assets/settings/shadows.webp)

`shadows.area_rays` (0 = off, the default) softens the shadows of large, near area lights in a pass
after the render. For each area light that matters at a pixel, it traces `area_rays` shadow rays
to points of the light. They replace the hard center-ray shadow.

- It needs `lighting.area_lights="ltc"` and `aa.spp=1` (`ValueError` otherwise).
- On, the render also writes the position and normal outputs.
- Off, it costs nothing. On, about 2.5-3x the frame where near area lights light most pixels
  (RTX PRO 6000, 512 worlds x 512^2, default, 4 rays).

`shadows.translucent` lets shadow rays through translucent casters, attenuated by each caster's
transmittance once. It is free where a ray meets none.

**`lighting`.**

| field | what |
| --- | --- |
| `area_lights` | `"ltc"`: rect, disk, sphere and tube lights as area lights (soft highlights and shadows). `"point"` (`FAST`): as points, cheaper. |
| `max_exact` | shade only the strongest lights of a point exactly, the rest diffusely. Default 4; `None` (`FAST`): every light; 8 in the quality recipe. Office, L4, 16 x 256^2: -44 % frame time, FLIP 0.06. |
| `light_grid` | limits each point to the lights its cell can see; built on the first render, cached. `"auto"` builds it from 8 lights (below that its lookup costs more than it culls); `True` always. |
| `emissive_lights` | flat emissive panels become rect lights |
| `camera_light` | MuJoCo's headlight, a camera-attached light (`"auto"`: as the model says) |
| `fill`, `fill_color` | a run-time ambient fill ([dark interiors](faq.md#dark-interiors)) |
| `contact_ao` | contact shadows under moving objects (below) |

<!-- SHOWCASE: settings/contact_ao.webp (docs/assets/SHOTS.md) -->
![lighting.contact_ao off / on](assets/settings/contact_ao.webp)

`lighting.contact_ao` (0 = off, up to 1) darkens the static scenery's indirect light around moving
objects: a robot's links, grasped or randomised objects. The GI probes do not see those objects.

- Each moving instance is a box proxy (spheres stay spheres), posed every frame.
- The occlusion is analytic: deterministic, no noise, no history.
- It adds a planar depth output if none is requested. The moving objects themselves are not darkened.
- Off, it costs nothing. On, RTX PRO 6000, 512 worlds x 512^2: +0.6-0.8 ms where nothing moves.
  The cost grows with the moving parts near what the pixels see.
- Orthographic, fisheye and distorted cameras get an approximate occlusion.
- Request a depth output yourself to avoid an output change on first use.

**`environment`.** Image-based lighting from the scene's environment map (`enabled`), and the map
as background (`background`, else `background_color`). Without GI the environment lights every
surface unoccluded; the GI probes keep the sky out of rooms.

**`reflections`.** One traced ray along the GGX dominant direction, for surfaces smoother than
`max_roughness`. On by default, off in `FAST` (a run-time switch).

- `shadows="sun"` keeps sunlight out of indoor reflections. `max_exact` caps lights at hits.
- The traced ray stands for the lobe's sharp core only (99.8 % of chrome at roughness 0.05, 39 %
  at 0.25). The rest is the blurred environment / GI term, so rough metals do not read as mirrors.
- A MuJoCo `reflectance` is a specular F0, reflected like any glossy surface. MuJoCo's default
  shininess 0.5 is roughness 0.5: no ray. A mirror needs a low roughness.

<!-- SHOWCASE: settings/reflections.webp (docs/assets/SHOTS.md) -->
![reflections.enabled off / on](assets/settings/reflections.webp)

**`transparency`.** Translucent layers composited front to back, up to `max_layers` (run time; a
change re-bakes GI). Compiled in only when the scene has something translucent.

- `enabled=False` renders every surface opaque with the same kernels.
- Past the budget, the last surface shows the background along the ray through its
  transmittance. So stacked panes stay see-through at 2 layers (`FAST`).
- A closed pane's exit face is no layer: two layers see through a glass block.

**`gi`.** Shared irradiance probes, baked from static scenery on the first frame (or updated every
`update_every` frames). They bring bounce light and let emissive surfaces light a room. On by
default and in `FAST`; spacing about 0.5 m in a furnished room, 256 rays, 8 bake updates.

- Worlds with equal lights, materials and environment share one probe set. A few randomised
  variants get one set each. Continuous randomisation shares one set, tinted per world.
- Moving and per-world geometry stays out of the bake.
- GI field changes re-bake the probes. Memory: +0.3 GiB for the probe volume (512 worlds).

<!-- SHOWCASE: settings/gi.webp (docs/assets/SHOTS.md) -->
![gi.enabled off / on](assets/settings/gi.webp)

`gi.relight` (run time) decides when a continuously randomised world is re-lit by rays:

- `"on_change"` (the default): on the frame its GI inputs change.
- `"on_request"`: only when `Renderer.request_gi_relight(worlds)` names it (a (nworld,) bool device
  mask; captured graphs stay valid). Use it once per episode reset of a generator that randomises
  materials every frame.

In both modes, changes of light colours, intensities, on / off and environment intensity need no
rays. Material, light-position and environment-map changes do. Exposure is not a GI input.

**`surfaces`.** `textures` (`"auto"`), `backface_culling`, `anisotropy` (8 texture taps at grazing
angles against moire, +2-5 % on an L4), `shading="unlit"` (the unlit recipe).

**`post`.** `tone_mapper`, run time:

| value | look |
| --- | --- |
| `"pbr_neutral"` (default) | Khronos PBR Neutral: keeps base colors |
| `"aces_rtx"` | RTX's ACES approximation (the RTX look recipe): closest to Omniverse RTX, but HDRI-lit tabletops turn milky (+0.3 to +0.9 EV) |
| `"aces"` | filmic ACES |
| `"clamp"` | no curve |

<!-- SHOWCASE: settings/tone_mapper.webp (docs/assets/SHOTS.md) -->
![post.tone_mapper: pbr_neutral, aces_rtx, aces](assets/settings/tone_mapper.webp)

**`exposure`.** `mode="auto"` (the default) meters each camera image: middle grey 0.18 on the
10-90 % histogram band, highlights protected. `mode="manual"` uses `ev100` (higher is darker;
per world: `"exposure"`). Switching never recompiles. Light units: [scenes](scenes.md#light-units-and-exposure).

**`grading`.** White balance, the ASC CDL and saturation, applied by the display pass. Its values
can be set per world.

**`sensor`** (all run time, all off by default; off runs no pass and allocates nothing). A pass
after the display, on the final `rgb` / `luminance` and `depth`. The scene-referred outputs stay
clean. Per world: `env=["sensor.noise", "sensor.vignette", "sensor.response"]`. Costs:
[performance](performance.md#sensor-effects-and-multi-gpu).

| fields | effect |
| --- | --- |
| `rolling_shutter`, `motion_blur` | along `outputs.motion` |
| `vignette`, `aberration` | lens vignetting, lateral chromatic aberration |
| `noise`, `full_well`, `read_noise` | Poisson shot noise and read noise, seeded per world, pixel and frame |
| `bits` | the ADC's quantisation (0: none) |
| `response` | the camera response curve |
| `depth="stereo"`, `baseline`, `disparity_noise`, `disparity_step` | active-stereo depth: disparity noise, quantisation, occlusion band (RealSense D400-like defaults) |
| `depth="range"`, `range_noise` | time of flight / structured light: noise sigma0 + k (z - z0)^2 m (Kinect v1's fit) |
| `min_depth`, `max_depth`, `dropout`, `edge_dropout`, `edge_threshold` | depth range, random dropout, flying pixels at depth jumps |

**`atmosphere`.** Distance fog and exponential haze. On where `end > start` or `haze > 0`, or
where `fog.range` is declared per world. Color, range and haze are run time and per world (a
domain-randomisation knob). Off, it is free.

<!-- SHOWCASE: settings/fog.webp (docs/assets/SHOTS.md) -->
![atmosphere off / fog / haze](assets/settings/fog.webp)

**`performance`** (never changes pixels).

| field | what |
| --- | --- |
| `kernels` | `"prebuilt"` or `"scene"` ([above](#compiled-in-or-runtime)) |
| `traversal` | `"auto"`: OptiX on GPUs with RT cores, the software BVH elsewhere ([OptiX](performance.md#optix)) |
| `render_order` | `"world"` helps when every world renders nearly the same image |
| `rebuild_every` | rebuild per-world trees every N frames (default 64) instead of refitting forever; 0: refit only |

Launch tuning (block sizes, register caps, the BVH builder) is not a setting: it never changes
pixels.

## Store textures in less GPU memory

Material textures are the largest GPU allocation of big scenes. Each texture stores only its
distinct channels. Textures with equal texels are stored once. A one-value texture is stored as a
single texel.

```python
host = rw.SceneBuilder().build()
gpu = rw.upload(host, options=rw.SceneOptions(textures=raywarp.core.pipeline.textures.formats.EXACT_STORAGE))
```

`rw.SceneOptions(textures=TextureStorage(...))` (from `raywarp.core.pipeline.textures.formats`)
picks how 8-bit sRGB color is stored and caps the resolution:

| option | values |
| --- | --- |
| `color` | `"srgb8"` (default on CUDA): 4 bytes per texel, 1 for grey without alpha; filtered in linear values, mip 0 exact. `"float16"`: linear, 8 bytes per texel, exact. |
| `max_size` | textures larger than N texels are halved until they fit. Default `MAX_TEXTURE_SIZE` (1024), applied by every adapter; 0 keeps every level. |

- Data textures (ORM, opacity, roughness, linear color) and normal maps are always exact.
- HDR color is always float16. On the CPU, 8-bit color is float16.
- `EXACT_STORAGE` keeps every texel exact, with the same size cap.
- Built textures are cached on disk ([caches](performance.md#disk-caches)).
- The option is `upload`'s, and `rw.integrations.mujoco.scene`'s `options=`.

Block compression (BC7 / BC4 / BC5) is not offered. On an L4 it saved 1.1-1.2 GB on the Isaac
Office and Bistro, but changed Bistro visibly (FLIP 0.035) and made the cold upload 2.6x slower.
8-bit sRGB color stays within 3 levels of exact everywhere.

## USD stages

`rw.formats.usd.settings(stage)` maps a stage's Omniverse render settings onto a
`RenderSettings`. It is explicit: a stage's settings never apply by themselves.

```python
# docs: requires pxr
from pxr import Usd

stage = Usd.Stage.CreateInMemory()
stage.GetRootLayer().customLayerData = {"renderSettings": {"rtx:rendermode": "RaytracedLighting"}}
assert rw.formats.usd.settings(stage) == rw.RenderSettings()
```

- It reads `rtx:*` in the layer's `renderSettings` metadata and `omni:rtx:*` on RenderSettings /
  RenderProduct prims.
- The base comes from `rtx:rendermode`: RealTimePathTracing and PathTracing (or unset) -> the RTX
  look recipe; RaytracedLighting -> the default; MinimalRendering -> `FAST`; or your own `base=`.
- Then it applies the authored shadows, reflections, translucency, GI, sampled lighting, tone
  mapper, face culling, background and camera exposure (f-stop, shutter, ISO -> a manual `ev100`).
- Ignored keys (AA, DLSS, denoiser, caches) go to `report=` (a `rw.scene.LoadReport`). It never
  touches `outputs` or `performance`.

## All fields

The table is generated from the field metadata, so it is the source of truth. Kinds:
[above](#compiled-in-or-runtime).

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

## See also

- [Performance and kernels](performance.md): the max-speed checklist, what compiles
- [Scenes](scenes.md): lights, exposure, materials, cameras
- [Robotics](robotics.md): outputs and per-world tables
- [Troubleshooting](faq.md): dark images, changes that did not show
