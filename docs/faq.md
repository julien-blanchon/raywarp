# Troubleshooting

Find your symptom, then its cause, fix and links. The last section explains why raywarp
ray-traces.

## Quick start

| symptom | try first | section |
| --- | --- | --- |
| black or dark images | `lighting=rw.Lighting.studio()` | [My images are black or dark](#my-images-are-black-or-dark) |
| a dark interior | keep GI on, add `lighting.fill` | [Light a dark interior](#light-a-dark-interior) |
| too dark or washed out | `exposure.mode="auto"`, or lower `exposure.ev100` | [Too dark or too bright](#too-dark-or-too-bright) |
| something compiles | `print(r.explain())` | [It compiles on the first frame](#it-compiles-on-the-first-frame) |
| the first frame is slow | `Renderer.warm` | [The first frame is slow](#the-first-frame-is-slow) |
| out of GPU memory | fewer worlds, smaller textures | [Out of memory](#out-of-memory) |
| not like Isaac Sim / Blender / MuJoCo | the RTX look recipe, `rw.formats.usd.settings` | [It looks different](#it-looks-different-from-isaac-sim--blender--mujoco) |
| a texture or material is wrong | `print(host.report)` | [Textures are missing](#textures-are-missing--a-material-looks-wrong) |
| a change did not show | `r.update_settings(...)` | [A change did not take effect](#a-change-did-not-take-effect) |
| `StaleBinding` | capture the graph again | [`StaleBinding`](#stalebinding) |
| an OptiX error | `performance.traversal="software"` | [OptiX errors](#optix-errors) |
| dotted soft shadows | more `shadows.samples` | [Soft shadows look dotted](#soft-shadows-look-dotted) |

<a id="my-scene-is-dark-why"></a>

## My images are black or dark

**Quickest fix:** add the studio rig, `rw.Lighting.studio()`. It is a sky dome plus a key and a
fill light, exposed for the default exposure. Where it goes:

| scene | how |
| --- | --- |
| USD, Newton, Isaac Lab | `lighting=rw.Lighting.studio()` on `rw.formats.usd.scene`, `rw.integrations.newton.scene`, the Newton front ends or `RaywarpRendererCfg` |
| a built host scene | `host.with_lighting(rw.Lighting.studio())` |
| MuJoCo | `rest = rw.integrations.mujoco.add_lights(spec, rw.Lighting.studio())` before `spec.compile()`, then `rw.integrations.mujoco.scene(mjm, m, lighting=rest)` |

More in [Lighting a scene](scenes.md#lighting-a-scene). The causes, with their fixes:

1. **No light reaches the camera.** A scene needs lights, an ambient fill or an environment map.
   - MuJoCo: a model without `<light>` keeps its headlight unless `visual/headlight` turns it off.
   - USD: a stage without UsdLux lights or a DomeLight is dark.
   - Newton: a model carries no lights. `SensorCamera` / `ViewerRaywarpHeadless` add a default
     rig; `rw.integrations.newton.scene` does not.

   Fix: `lighting=rw.Lighting(lights=..., envmap=...)` on the adapter, `scene.with_lighting(...)`
   on a built scene, or the fills of [dark interiors](#light-a-dark-interior).
2. **Exposure.** See [dark interiors](#light-a-dark-interior), item 5.
3. **You read an output that is off.** Every output not requested is None. A geometry-only
   `Outputs` (depth, seg) writes no color at all.
4. **Metals with nothing to reflect.** A metal shows the environment. Without an environment map
   it is dark except for highlights. `lighting.fill` lights only the diffuse part.

<a id="dark-interiors"></a>

## Light a dark interior

A room lit through a window, a ceiling above downlights, the inside of a cabinet: places no light
reaches directly. raywarp never adds hidden light. These are your knobs, from the most physical to
the plainest. They combine.

<!-- SHOWCASE: faq/dark_interior.webp (docs/assets/SHOTS.md) -->
![A dark interior: no fill, lighting.fill, GI](assets/faq/dark_interior.webp)

1. **GI** (`gi.enabled`, on by default and in `FAST`, off under unlit shading). Light from windows
   and lamps bounces around the room, from probes baked on the first frame. This is the
   physically right answer for interiors.
2. **An ambient fill** (`lighting.fill`, run time, default 0). A constant diffuse term on every
   lit surface. A white surface shows `lighting.fill` x `lighting.fill_color` of display white,
   whatever the light units. It adds to GI, the environment and the lights. It never recompiles
   or re-bakes. 0.03-0.1 keeps shadows readable; more flattens the image.
3. **A headlight** (`lighting.camera_light=True`). An unshadowed light along each camera's view,
   as in MuJoCo's default viewer. Set its color with `Lighting(headlight=rw.Headlight(ambient=...,
   diffuse=...))` (MuJoCo: `visual/headlight`); 0.4 grey without one.
4. **Fill lights** (`rw.Lighting.fill_lights(n)`). `n` unshadowed directional lights from every
   side and above, sharing `brightness`. Use `Lighting(lights=rw.Lighting.fill_lights(3))`
   (MuJoCo: through `rw.integrations.mujoco.add_lights(spec, ...)`). Unlike the ambient fill they
   have a direction, give highlights, and GI bounces them.
5. **Exposure.** Exposure is automatic by default (`exposure.mode="auto"`). Photometric lights are
   in physical units (lux, candela, nits). With a manual `exposure.ev100` of 14.97 (a sunny-day
   camera), a 3 lux MJCF light or a 100 nit USD rect light is nearly black. Lower `ev100` (each -1
   doubles the brightness) or keep auto. Display-relative light (MuJoCo's classic lights, the
   ambient, headlight, fill lights, envmaps without an intensity) is calibrated for
   `Lighting.ev100`: keep `exposure.ev100` equal to it.

Keep GI on indoors. Without GI, the sky lights every room as if it had no roof: the Isaac Office
comes out +2.6 EV, washed out. GI off only suits open scenes, where it changes little.

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

To estimate a manual exposure from the HDR output:

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

- `exposure.ev100` is the manual exposure (`exposure.mode="manual"`). Higher is darker. Per world:
  `PerWorld(env=["exposure"])`.
- `post.tone_mapper` shapes the highlights. `"pbr_neutral"` (the default) keeps base colors;
  `"aces"` and `"aces_rtx"` are filmic.
- An environment map that is too bright or dark: `rw.Lighting(envmap_intensity=...)`, or the
  per-world `"env.intensity"`.
- To compare with Omniverse RTX, use the RTX look recipe ([settings](settings.md#recipes)).
- Every tone mapper uses the USD / RTX convention: linear radiance times `2 ** -exposure.ev100`.

The first frame is checked once (outside a graph capture). A `RuntimeWarning` names each camera
whose first image is black or has more than 20 % of its pixels clipped to white. The check reads
back up to 16 worlds once; later frames are not read back.

## It compiles on the first frame

**Cause.** The configuration is outside the prebuilt kernels. Usual cases: a lens camera, the
motion output on the default, `shadows.samples` alone, `performance.kernels="scene"`, the CPU
device, or a GPU architecture without a prebuilt bundle. The full list:
[zero compiles](performance.md#zero-compiles).

**Fix.**

- `print(r.explain())` names the build. `kernels: own (outside the tiers; nearest 'default'
  differs in ...)` names the setting that left the prebuilt builds: change it back.
- Use a [documented recipe](settings.md#recipes): every one is prebuilt.
- Compile once ahead with `Renderer.warm`; the result is cached on disk.
- Make compiles an error in production:
  `rw.kernels.configure(compile=False)` (or `RAYWARP_KERNELS_COMPILE=0`) raises
  `rw.kernels.KernelsNotPrebuilt`, naming the configuration ([install](install.md#prebuilt-kernels)).

## The first frame is slow

The first render of a configuration downloads its prebuilt kernels, creates its OptiX modules and
runs the scene's bakes. Everything is cached, so the second run starts in seconds. Warm up before
training with `Renderer.warm`, at the run's world count, cameras and resolution. Details:
[first frame](performance.md#first-frame-prebuilt-kernels-caches-and-warm).

- A warning that a prebuilt file could not be fetched means a compile. Without network, see
  [offline machines](install.md#run-offline-or-mirror-the-kernels).
- On the CPU the kernels compile once (tens of seconds), then are cached.
- A big USD stage spends its first load parsing. Its `.rayw` is kept (`cache=True`, the default);
  `rw.formats.usd.bake` makes it ahead of time.

## Out of memory

**Cause.** GPU memory grows with worlds x cameras x pixels x outputs, plus the scene's textures.

**Fix.**

- Request only the outputs you read: unrequested outputs are not allocated. Use
  `dtype="float16"` on float outputs and `ColorOutput(channels=3)` where alpha is not needed.
- Cap texture size: `rw.SceneOptions(textures=TextureStorage(max_size=512))`
  ([texture storage](settings.md#store-textures-in-less-gpu-memory)). The default caps at 1024 px.
- Render fewer worlds per renderer, or one renderer per GPU.
- Lower the GI probe budget, `gi.max_probes` (default 32768), or turn GI off for open scenes.
- `CameraSpec(outputs=...)` saves time, not memory: buffers stay `(nworld, ncam, H, W)`.

## It looks different from Isaac Sim / Blender / MuJoCo

**Isaac Sim / Omniverse RTX.** Use the RTX look recipe ([settings](settings.md#recipes)) for
RTX's ACES curve and glass shadows. `rw.formats.usd.settings(stage)` maps a stage's authored RTX
settings ([USD stages](settings.md#usd-stages)). Kit-saved lights use RTX's legacy units
([USD light dialects](scenes.md#usd-light-dialects)).

**Blender, Mitsuba and other path tracers.** raywarp is a real-time ray tracer, not a path tracer
([why](#why-ray-tracing-and-not-rasterisation-or-path-tracing)). The reference recipe is the
closest comparison. Keep GI on for interiors.

**MuJoCo.** A geom with a material and a non-default `rgba` shows the `rgba` (MuJoCo's rule).
MuJoCo's classic lights are display-relative, calibrated for `Lighting.ev100` (14.97). Shininess
0.5 is roughness 0.5: no mirror reflection
([how MuJoCo lights are read](scenes.md#how-each-formats-lights-are-read)).

### Newton colors look darker than the MuJoCo adapter's

Newton defines `shape_color` as sRGB, but its MJCF importer stores MuJoCo's linear `rgba`
unchanged. So an MJCF loaded through Newton renders its colors sRGB-decoded (darker). The Newton
adapter follows Newton, to match Newton's own sensor.

### The GPU and CPU images differ slightly

They are within a few 1/255 per channel. Fast-math and fused-multiply-add change the last bits.
That can flip an edge pixel (a depth or segmentation tie on a box edge). The regression suite
keeps references per platform for that reason.

## Textures are missing / a material looks wrong

Print the load report: `host.report` for USD and host scenes
([load report](scenes.md#the-load-report)). It counts every dropped or approximated attribute,
with the first prim path that hit it. Common causes:

- a texture path that does not resolve (`DomeLight.texture:file.unresolved`, ...);
- an MDL module that was not downloaded beside the stage;
- a UV set other than the first two.

`surfaces.textures="auto"` turns textures on only when a material binds one. In MuJoCo, a geom
with a material and a non-default `rgba` shows the `rgba` (MuJoCo's rule).

## A change did not take effect

- **`RenderSettings` is frozen.** `replace` returns a new value. Give it to a new `Renderer`, or
  call `r.update_settings(**changes)`.
- **Per-world tables win.** Once a field is declared in `PerWorld`, its table wins over the MuJoCo
  model's array: `m.geom_rgba` changes no longer show for that field.
- **Live vs read once (MuJoCo).** The arrays of `mjw.Model` (geom colors, sizes, material ids,
  light poses, camera tables) are re-read every frame. What MuJoCo Warp does not hold (material
  metallic / roughness, light intensity / range / bulb radius, textures) is read from `mjm` once.
  Rebuild the scene after changing those, or declare them per world.
- **Baked structures.** The light grid and GI probes are built on the first render.
  - Moving lights fall back to evaluating every light in the same frame.
    `r.rebuild_light_grid()` restores the culling. `PerWorld(dynamic_lights=[...])` declares a
    known movable subset.
  - After changing what the probes saw (`lighting.area_lights`, `lighting.emissive_lights`, the
    envmap intensity, ...), call `r.rebuild_gi()`. `update_settings` re-bakes only for the GI
    fields themselves.
  - Shared scenery is static: moving it by hand raises. Moving shared walls needs refreshed bakes.
  - Instances that move or change per world never block the light grid's rays (only their
    shadow rays).
- **Per-world alpha.** Written without `PerWorld(alpha=True)` it renders opaque: colors are RGB
  tints by default. Hide instances with `PerWorld(visible=[...])`.
- `r.features` holds everything the serving kernel build compiles in.

## `StaleBinding`

`RenderGraph.launch` checks that it reads the captured arrays. It raises when you:

- pass another simulator state (Newton swaps `state_0` / `state_1` every substep: capture an even
  number of substeps);
- re-bind outputs;
- change the renderer (`update_settings`, `set_camera_rays`, a new per-world table).

Capture again after such a change.

## OptiX errors

There is no silent fallback. `performance.traversal="optix"` raises when:

- the driver does not serve OptiX 9.0 (R580 or newer);
- the bundled OptiX headers are missing (reinstall raywarp);
- the scene has geometry OptiX does not trace (1-D flex: segments).

`"auto"` (the default) raises on an RT-core GPU where OptiX cannot run. Elsewhere it picks the
software BVH (CPU, A100 / H100). Set `performance.traversal="software"` to run without OptiX; the
images are the same.

<a id="why-do-soft-shadows-look-dotted"></a>

## Soft shadows look dotted

Along a soft shadow's edge, the light is partly visible. Each shadow ray samples one point on the
light. At low sample counts, neighbouring pixels estimate different visible fractions: grain.
Glossy reflections and antialiasing can add their own noise.

Each frame stands alone (no temporal accumulation). To reduce the grain, raise `shadows.samples`
or turn on the soft area-shadow pass (`shadows.area_rays`), at the cost of rays. `shadows.samples`
adds shadow rays at the shaded surface only, without re-tracing the camera ray. A one-point light
sample gives a hard shadow: a different look, not a cheaper soft shadow.

## Why ray tracing (and not rasterisation or path tracing)?

**What raywarp is:** a batched ray tracer, not a path tracer and not a rasteriser.

- Every pixel traces a camera ray (more with AA), shades the hit with a PBR BRDF and casts shadow
  rays to the selected lights.
- LTC approximates area lights. Image-based lighting, single-bounce reflections and optional
  ray-baked GI probes approximate indirect light.
- More samples (the reference recipe) converge these approximations. They do not make it a
  recursive path tracer or physical ground truth.

**Why rays fit many-world rendering:**

- One shared, instanced BVH serves every world and camera. A step refits it once; there is no
  per-view setup.
- One launch renders every (world, camera, pixel). A rasteriser processes every triangle again for
  each view.
- Depth, segmentation, shadows and reflections come from the same rays: no shadow-map or G-buffer
  pass per view and per light.
- The cost is nearly flat in triangle count (OptiX: 0.53 -> 0.66 ms from 10k to 10M triangles).

**What research and industry use:**

| system | approach |
| --- | --- |
| bps3d (Shacklett et al., ICLR 2021) | Vulkan raster, the batch as one tiled frame |
| Madrona batch renderer (SIGGRAPH Asia 2024) | raster and a software ray tracer; the ray tracer wins beyond a few views on dense scenes |
| MuJoCo Playground / MuJoCo Warp, Newton `SensorCamera` | software ray tracers |
| Isaac Lab `TiledCamera` | Omniverse RTX, hardware ray tracing with a large fixed cost per step |
| ManiSkill3 / SAPIEN, Habitat, Genesis, PyBatchRender | GL / Vulkan rasterisers; ray-traced modes render one environment at a time |

The field moved from raster batching (bps3d) to ray tracing (Madrona, MJWarp, Newton).

**Measured** on one RTX PRO 6000 against MuJoCo's OpenGL rasteriser (EGL on the GPU), in
camera-frames/s:

| scene | cameras x res | ours unlit | ours `FAST` | MuJoCo EGL (no readback) |
| --- | --- | ---: | ---: | ---: |
| Unitree G1 (393k triangles) | 1 x 1024^2 | 8.3k | 5.6k | 3.2k |
| Unitree G1 | 256 x 512^2 | 110k | 43.7k | 3.7k |
| Unitree G1 | 256 x 1024^2 | 30.7k | 12.1k | 3.2k |
| ALOHA clutter (1.19M triangles) | 256 x 1024^2 | 18.3k | 5.8k | 0.59k |

The rasteriser draws one view at a time. Eight processes sharing the GPU give at most 1.3x one;
on ALOHA it stays near 660 views/s. On the same GPU raywarp is also 8-11x faster than Omniverse
RTX's Minimal mode at 64^2-128^2, 3.7x at 1080p, and 25x its real-time path tracer.

**Where a rasteriser would still win:** one camera at high resolution (2048^2 and up) on a
low-poly scene. There its per-view cost no longer matters and its fill rate does. Our closest
measured case: 52 primitives, 1 x 1024^2, the rasteriser at 0.69x our `FAST`. A batched
rasteriser (ManiSkill3, PyBatchRender) would remove the per-view CPU cost, not the per-view
triangle work; we did not measure one.

## See also

- [Scenes](scenes.md): lights, environments, materials, the load report
- [Settings](settings.md): every field and the recipes
- [Performance and kernels](performance.md): first frames and speed
