# FAQ and troubleshooting

## My scene is dark: why?

**Quickest fix:** add the studio rig (`rw.Lighting.studio()`: a sky dome, a key and a fill
light, exposed for the default `exposure.ev100`): `lighting=rw.Lighting.studio()` on
`rw.formats.usd.scene` / `rw.integrations.newton.scene` / the Newton front ends / `RaywarpRendererCfg`,
`host.with_lighting(rw.Lighting.studio())` on a built scene, and for MuJoCo
`rest = rw.integrations.mujoco.add_lights(spec, rw.Lighting.studio())` before `spec.compile()`, then
`rw.integrations.mujoco.scene(mjm, m, lighting=rest)` ([Lighting a scene](scenes.md#lighting-a-scene)).

Each cause, with its fix:

1. **No light reaches the camera.** A scene needs lights, an ambient fill or an environment map.
   MuJoCo: a model without `<light>` still has its headlight unless `visual/headlight` turns it
   off; USD: a stage without UsdLux lights or a DomeLight is dark; a Newton model carries no
   lights (`SensorCamera` / `ViewerRaywarpHeadless` add a default rig, `rw.integrations.newton.scene` does not). Fix: add
   them from Python with `lighting=rw.Lighting(lights=..., envmap=...)` on the adapter, or
   `scene.with_lighting(...)` on a built host scene; or the fill knobs of
   [dark interiors](#dark-interiors).
2. **Exposure.** See [dark interiors](#dark-interiors), item 5.
3. **The output you read is off.** `out.rgb` is None unless `outputs.rgb` is requested (every
   output not requested is None); a geometry-only `Outputs` (depth, seg) writes no color at all.
4. **Metals with nothing to reflect.** A metallic surface shows the environment; without an
   environment map it is dark except for highlights (`lighting.fill` lights only the diffuse
   part: metals stay dark).

## Dark interiors

A room lit through a window, a ceiling above downlights, the inside of a cabinet: what no light
reaches directly. raywarp never adds light behind your back (no hidden per-scene constants):
these are your knobs, from the most physical to the plainest fill. They combine.

1. **GI on** (`gi.enabled`, on by default and in `FAST`, off under unlit shading): the light that comes in through the windows or falls on the
   floor bounces around the room, from a probe grid baked on the first frame (per group of
   worlds with equal materials and lights; under continuous per-world randomisation one shared
   grid tinted per world). It is the physically right answer for interiors.
2. **An ambient fill** (`lighting.fill`, runtime, default 0): a constant term added to every
   lit surface, display-relative: a white diffuse surface shows `lighting.fill` x
   `lighting.fill_color` of display white at its world's exposure, whatever the scene's light
   units. Diffuse only (base color x (1 - metalness) x ambient occlusion x fill, as HyperLux's
   ambient), added on top of GI, the environment and the lights: it replaces none of them, never
   recompiles and re-bakes nothing. 0.03-0.1 keeps shadows readable; more flattens the image.
3. **A headlight** (`lighting.camera_light=True`, unshadowed, along each camera's view): what a camera
   sees is lit, as MuJoCo's default viewer. Its color: `Lighting(headlight=rw.Headlight(
   ambient=..., diffuse=...))` on the adapter (MuJoCo: `visual/headlight`), 0.4 grey without one.
4. **Fill lights** (`rw.Lighting.fill_lights(n)`): `n` unshadowed directional lights from every side and
   above, display-relative (`brightness` split among them), added like any light:
   `Lighting(lights=rw.Lighting.fill_lights(3))` (MuJoCo: through `rw.integrations.mujoco.add_lights(spec, ...)`
   before compiling). Unlike the ambient they have a direction (shading, specular highlights),
   and GI bounces them.
5. **Exposure.** Photometric lights are in physical units (lux, candela, nits); the default
   exposure is automatic (each camera image metered: `exposure.mode="auto"`, `exposure.*`). With a
   manual `exposure.ev100` (e.g. 14.97, a sunny-day camera) a 3 lux MJCF light or a 100 nit USD
   rect light is nearly black: lower it (each -1 doubles the brightness) or keep the automatic
   default. Display-relative light (MuJoCo's classic lights, `Lighting`'s ambient, headlight,
   fill lights and envmaps without an intensity) is calibrated for `Lighting.ev100`: keep
   `exposure.ev100` equal to it.

Keep GI on indoors: an environment map lights only what the probes see of the sky; with GI off,
the sky lights every room as if it had no roof (the Isaac Office comes out +2.6 EV, washed out).
Off only makes sense for an open scene, where it changes little.

```python
import mujoco
import mujoco_warp as mjw

import raywarp as rw
import raywarp.integrations.mujoco

spec = mujoco.MjSpec.from_string("""
<mujoco><visual><headlight active="0"/></visual><worldbody>
  <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
  <geom type="plane" size="2 2 0.1"/><geom type="box" pos="0 0 0.6" size="0.6 0.6 0.05"/>
  <camera name="cam" pos="0 -2 0.3" xyaxes="1 0 0 0 0.2 1"/>
</worldbody></mujoco>""")  # a table-top in shadow from below, one lamp above
fill = rw.integrations.mujoco.add_lights(spec, rw.Lighting(lights=rw.Lighting.fill_lights(3, brightness=0.2)))
mjm = spec.compile()
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=1)
mjw.forward(m, d)
settings = rw.RenderSettings().replace(rw.FAST).replace(**{"lighting.fill": 0.05, "lighting.camera_light": True})
r = rw.Renderer(
  rw.integrations.mujoco.scene(mjm, m, lighting=fill), nworld=1, cameras=["cam"], resolution=(32, 24), settings=settings
)
lit = r.render(d).rgb.numpy()[..., :3].mean()
r.update_settings(**{"lighting.fill": 0.1})  # runtime: no recompilation, no re-bake
assert r.render(d).rgb.numpy()[..., :3].mean() > lit
```

Estimating the exposure from the HDR output:

```python
import mujoco
import mujoco_warp as mjw
import numpy as np

import raywarp as rw
import raywarp.integrations.mujoco

mjm = mujoco.MjModel.from_xml_string("""
<mujoco><worldbody>
  <light type="directional" dir="0 0 -1" intensity="2"/>
  <geom type="plane" size="2 2 0.1"/><geom type="sphere" pos="0 0 0.3" size="0.3" rgba="0.8 0.3 0.2 1"/>
  <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
</worldbody></mujoco>""")
m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=1)
mjw.forward(m, d)
settings = rw.RenderSettings(outputs=rw.Outputs.of("rgb", "hdr"))
r = rw.Renderer(rw.integrations.mujoco.scene(mjm, m), nworld=1, cameras=["cam"], resolution=(32, 24), settings=settings)
hdr = r.render(d).hdr.numpy()  # linear radiance before exposure, (nworld, ncam, H, W, 3)
luminance = float((hdr @ np.array([0.2126, 0.7152, 0.0722])).mean())
ev100 = float(np.log2(luminance * 100.0 / 12.5))  # the reflected-light meter equation
r.update_settings(**{"exposure.mode": "manual", "exposure.ev100": ev100})  # runtime: no recompilation
assert r.render(d).rgb.numpy()[..., :3].mean() > 20
```

## Too dark or too bright

`exposure.ev100` is the manual camera exposure (`exposure.mode="manual"`; higher is darker, per world via `PerWorld(env=["exposure"])`).
`post.tone_mapper` shapes the highlights (`"pbr_neutral"`, the default, keeps base colors;
`"aces"` and `"aces_rtx"` are filmic). An environment map that is too bright or dark:
`lighting=rw.Lighting(envmap_intensity=...)` or the per-world `"env.intensity"`. For a side-by-side
with Omniverse RTX use the RTX look recipe ([settings](settings.md#recipes)) for its `aces_rtx` curve. Every tone mapper uses the same
USD/RTX exposure convention: linear radiance multiplied by `2 ** -exposure.ev100`.

A Renderer's first frame (outside a graph capture) is checked once: a `RuntimeWarning` names each
camera whose first image has more than 20 % of its pixels clipped to white, or is black. It reads
back up to 16 worlds once; no later frame is read back.

## Textures are missing / a material looks wrong

Print the load report (`host.report` for USD and host scenes, [scenes](scenes.md#the-load-report)):
every dropped or approximated attribute is counted with the first prim path that hit it.
Common causes: a texture path that does not resolve (`DomeLight.texture:file.unresolved`, ...),
an MDL module that was not downloaded (`internal/tools/fetch_assets.py` follows MDL imports), a UV set
other than the first two. `surfaces.textures="auto"` turns textures on only when a material
binds one. MuJoCo: a geom with a material and a non-default `rgba` shows the `rgba` (MuJoCo's
rule), not the material's color.

## The first frame is slow

On a GPU the first render of a configuration downloads its prebuilt kernels (only those it
uses), creates its OptiX modules (from the driver's module pack when the bundle has one, else
seconds each, once per machine and driver) and runs the scene's bakes (light grid, GI,
environment prefilter); everything is cached on disk, so the second run starts in seconds. A
warning that a prebuilt file could not be fetched means a compile (no network: see
[prebuilt kernels](performance.md#first-frame-prebuilt-kernels-caches-and-warm)); on the CPU the kernels compile
once (tens of seconds per megakernel, then cached). Warm up before training (`Renderer.warm`)
with the run's world count, cameras and resolution. A big USD stage spends most of its first load parsing:
its `.wpbr` is kept on the first load (the default `cache=True`; `rw.formats.usd.bake` ahead of time).

## A change did not take effect

- **`RenderSettings` is frozen**: `replace` returns a new value; give it to a new `Renderer` or
  call `r.update_settings(**changes)` on the existing one.
- **Per-world overrides win**: once a field is declared in `PerWorld`, its table wins over the
  MuJoCo model's array (`m.geom_rgba` changes no longer show for that field).
- **Live vs read once (MuJoCo)**: the arrays of `mjw.Model` (geom colors, sizes, material ids,
  light poses, camera tables) are aliased and re-read every frame; what MuJoCo Warp does not hold
  (material metallic / roughness, light intensity / range / bulb radius, textures) is read from
  `mjm` once, in `rw.integrations.mujoco.scene`. Rebuild the scene after changing those in `mjm`, or declare
  them per world.
- **Baked structures**: the light grid and the GI probe volume are built on the first render.
  Moving lights automatically fall back to all-light evaluation in the same frame; call
  `r.rebuild_light_grid()` to restore culling for their new positions. A known movable subset
  can use `PerWorld(dynamic_lights=[...])`: those light indices always bypass the static grid,
  while undeclared footprint changes retain the automatic fallback. Moving shared walls require
  refreshed bakes; shared scenery moved by hand is detected and raises (shared scenery is static). `r.update_settings(...)` re-bakes GI
  only for the GI fields themselves: after changing what the probes saw (`lighting.area_lights`,
  `lighting.emissive_lights`, the scene's envmap intensity, ...) call `r.rebuild_gi()` (a frozen bake keeps the old light). Instances
  that move or change per world never occlude the light grid's rays (only their shadow rays).
- **A per-world alpha** written without `PerWorld(alpha=True)` renders opaque: colors are RGB
  tints by default. Hide instances with `PerWorld(visible=[...])`.
- `r.features` is the resolved `Features`: everything the serving kernel build compiles in.

## `StaleBinding`

`RenderGraph.launch` checks that the arrays it reads are the captured ones. It raises when you
pass another simulator state (Newton swaps `state_0` / `state_1` every substep: capture an even
number of substeps), re-bind outputs, or change the renderer (`update_settings`,
`set_camera_rays`, a new per-world table): capture again after such a change.

## OptiX errors

There is no silent fallback. `performance.traversal="optix"` raises when the driver does not serve
OptiX 9.0 (R580 or newer), the bundled OptiX headers are missing (reinstall raywarp) or the scene
has geometry OptiX does not trace (dim-1 flex: segments); `"auto"` (the default) raises on a
GPU with RT cores where OptiX cannot run (for a scene
OptiX would trace), and
otherwise picks by device: OptiX with RT cores, the software BVH on the CPU and on A100 / H100.
Set `performance.traversal="software"` to run without OptiX; the images are the same.

## The GPU and CPU images differ slightly

They are within a few 1/255 per channel: fast-math and fused-multiply-add differences change
the last bits, which can flip a pixel on an edge (a depth or segmentation tie on a box edge).
The regression suite records references per platform for that reason.

## Newton colors look different from the MuJoCo adapter

Newton defines `shape_color` as sRGB (its viewer and sensor decode it) while its MJCF importer
stores MuJoCo's linear `rgba` unchanged, so an MJCF loaded through Newton renders its colors
sRGB-decoded (darker). The Newton adapter follows Newton, so the drop-ins keep parity with
Newton's own sensor.

## Why do soft shadows look dotted?

A finite light source is partly visible along a soft shadow's edge. Each shadow ray samples
one point on that source; at low sample counts, neighboring pixels estimate different visible
fractions. This produces grain even with correct geometry and light size. Glossy reflections
and pixel antialiasing can contribute separate noise.

Each frame stands alone (no temporal accumulation): more shadow samples (`shadows.samples`) or
the soft area-shadow pass (`shadows.area_rays`) reduce the grain, at the cost of rays. A fixed
one-point light sample makes a hard shadow and changes the intended appearance; it is not an
equivalent soft-shadow solution.

`shadows.samples` concentrates shadow rays at an already shaded primary surface. This avoids
repeating primary tracing and material evaluation for every shadow sample.

## Why ray tracing (and not rasterisation or path tracing)?

**What raywarp is:** a batched ray tracer, not a path tracer and not a rasteriser. Every pixel
traces a camera ray (more with AA), shades the hit with a PBR BRDF and casts shadow rays to
the selected lights. LTC approximates direct area-light response; image-based lighting,
single-bounce reflections and optional ray-baked GI probes supply indirect approximations.
The reference recipe takes more samples per frame. Increasing samples converges those existing
approximations;
it does not turn this renderer into a recursive path tracer or physical ground truth.

**Why rays fit many-world RL rendering:**

- one shared, instanced BVH (OptiX instance tree or our software BVH) for every world and
  camera; a step refits it once, there is no per-view setup;
- one launch renders every (world, camera, pixel); a rasteriser processes every triangle again
  for each view;
- depth, segmentation, shadows and reflections come from the same rays: no shadow-map or
  G-buffer pass per view and per light;
- the cost is nearly flat in triangle count (OptiX: 0.53 -> 0.66 ms from 10k to 10M triangles,
  `internal/results/history/benchmark_matrix`).

**What research and industry use:**

- bps3d (Shacklett et al., ICLR 2021): Vulkan raster, the whole batch as one tiled frame;
- Madrona batch renderer (Rosenzweig, Shacklett et al., SIGGRAPH Asia 2024): raster and a
  software ray tracer, the ray tracer wins beyond a few views on dense scenes;
- MuJoCo Playground / MuJoCo Warp, Newton `SensorCamera`: software ray tracers;
- Isaac Lab `TiledCamera`: Omniverse RTX, hardware ray tracing with a large fixed cost per step;
- ManiSkill3 / SAPIEN, Habitat, Genesis, PyBatchRender: GL / Vulkan rasterisers; their
  ray-traced modes render one environment at a time.

The field moved from raster batching (bps3d) to ray tracing (Madrona, MJWarp, Newton). Sources
and numbers: `internal/results/history/research/survey_2026_10_28`.

**Measured (one RTX PRO 6000; `internal/results/history/research/raster_vs_rt`):** against MuJoCo's
OpenGL rasteriser (EGL on the GPU), at the training resolutions 512^2 and 1024^2:

| scene | cameras x res | ours unlit | ours `FAST` | MuJoCo EGL (no readback) |
| --- | --- | ---: | ---: | ---: |
| Unitree G1 (393k triangles) | 1 x 1024^2 | 8.3k | 5.6k | 3.2k |
| Unitree G1 | 256 x 512^2 | 110k | 43.7k | 3.7k |
| Unitree G1 | 256 x 1024^2 | 30.7k | 12.1k | 3.2k |
| ALOHA clutter (1.19M triangles) | 256 x 1024^2 | 18.3k | 5.8k | 0.59k |

(camera-frames/s). The rasteriser draws one view at a time; eight processes sharing the GPU give
at most 1.3x one (on ALOHA it stays at ~660 views/s: every view re-processes every triangle). On
the same GPU we are also 8-11x Omniverse RTX's Minimal mode at 64^2-128^2 and 3.7x at 1080p
(25x its real-time path tracer).

**Where a rasteriser would still win:** a single camera at high resolution (2048^2 and up) on a
low-poly scene, where its per-view cost no longer matters and its fill rate does (our closest
measured case: 52 primitives, 1 x 1024^2, the rasteriser at 0.69x our `FAST`). A
batched rasteriser (ManiSkill3, PyBatchRender) would remove the per-view CPU cost, not the
per-view triangle work; we did not measure one, as it is a new dependency.
