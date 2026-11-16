# Changelog

## 0.4.0 (unreleased)

### Breaking changes

- **The project is renamed Raywarp.** No alias package, no shims: update imports, commands and
  environment variables.

  | | Before | Now |
  |---|---|---|
  | Distribution | `pip install warp-pbr[...]` | `pip install raywarp[...]` |
  | Import package | `warp_pbr` | `raywarp` |
  | Documented alias | `import warp_pbr as wpr` | `import raywarp as rw` |
  | Command | `warp-pbr` | `raywarp` |
  | Environment variables | `WARP_PBR_*` (`WARP_PBR_CACHE_DIR`, `WARP_PBR_KERNELS`, `WARP_PBR_KERNELS_URL`, ...) | `RAYWARP_*` |
  | Cache directory | `~/.cache/warp_pbr` | `~/.cache/raywarp` |
  | USD attributes | `warp_pbr:unlit`, `warp_pbr:vertexColor`, `warp_pbr:displayEmission` | `raywarp:*` |
  | mjlab renderer entry point | `warp_pbr` | `raywarp` |
  | Isaac Lab | `WarpPbrRenderer`, `WarpPbrRendererCfg` | `RaywarpRenderer`, `RaywarpRendererCfg` |
  | mjlab | `WarpPbrCameraSensorCfg` | `RaywarpCameraSensorCfg` |
  | Newton | `ViewerPbrHeadless` | `ViewerRaywarpHeadless` |
  | Material extension | `wpr.scene.WarpPbrExt` | `rw.scene.RaywarpExt` |
  | Prebuilt kernels bucket | `blanchon/warp-pbr-kernels` | `blanchon/raywarp-kernels` (a release's pins name its bucket) |

- **Distinct class names:** the host scene's environment light `wpr.scene.Environment` is
  `rw.scene.EnvironmentLight`; the internal GPU-scene `Environment` / `Headlight` are
  `GpuEnvironment` / `GpuHeadlight` and the kernel ABI struct `Environment` is
  `EnvironmentArrays`. The settings group `rw.settings.Environment` and `rw.Headlight` keep
  their names.
- **Temporal accumulation is removed.** `aa.temporal` and `aa.accumulate_max` are gone (passing
  them raises like any unknown setting), with the running-mean store, the per-frame Halton
  jitter, the progressive (Owen-Sobol) shadow samples, the stochastic sun-disc / area-light
  targets and the budget's sampled remainder that existed only to be averaged. Every frame
  renders on its own and is deterministic per pixel and seed: soft shadows (`shadows.samples`,
  `shadows.area_rays`) keep their fixed per-pixel patterns. For clean stills use `aa.spp=2`; the
  `reference` recipe (docs/settings.md) is now per frame at `aa.spp=2`, and the RTX look (and the
  USD `rtx:rendermode` mapping) per frame at `aa.spp=1`. The motion output keeps its history
  (`Renderer.reset_history` restarts it). `ViewerRaywarpHeadless(accumulate=...)` is gone (the
  viewer renders every frame at its size). The prebuilt bundle has no accumulate builds any
  more, and the OptiX layer / light-visibility call payloads are 2 registers narrower.
- **One command, `raywarp`, replaces `warp-pbr-viewer` and `python -m warp_pbr.viewer`:**
  `raywarp view [SCENE ...]` (same options). `python -m warp_pbr.cache.prune --max-gb N` is
  `raywarp cache prune --max-gb N` (`raywarp.cache.prune.prune` is unchanged). The command's
  other subcommands are new (Added). Without the console script: `python -m raywarp.cli`.
- **The generated API reference is gone:** docs/api.md, `tools/api_doc.py` and the static audit
  `tools/audit.py` (and their tests) are removed; the docstrings and the user guide are the
  reference.
- **The repository is split into the public tree and `internal/`:** the large benchmark suite
  (`benchmarks/` -> `internal/benchmarks/`), the knowledge base (`research/` ->
  `internal/research/`), the archived results (`internal/results/`), the dev web UI
  (`internal/webui/`, docs/webui.md -> `internal/docs/webui.md`), their tests (`internal/tests/`)
  and the development tools move under `internal/`: `tools/autotune`, `tools/dev`,
  `tools/mitsuba_usd`, `tools/mdl_corpus*`, `tools/viewer` (its stream client and delay proxy:
  `tests/fixtures/`), `tools/fetch_assets.py`, `tools/intel_samples.py`,
  `tools/polyhaven_apartment.py`, `tools/video/bench.py` and the dev parts of `tools/hf` (`sandbox`, `test`: `python -m
  internal.tools.hf`, a superset of `python -m tools.hf`). `internal/` stays in the repository
  (worktrees and HF jobs see it; `tools/hf/runs.py` ships it) but never in the wheel, the sdist or
  the public export (`.gitattributes` `export-ignore`). The committed fixture scenes moved from
  `benchmarks/scenes/` to `tests/data/scenes/`; `tests/conftest.py` is the root `conftest.py`
  (it serves `tests/` and `internal/tests/`); `tools/test_cache.py` / `tools/test_precompile.py`
  are `tests/kernel_snapshot.py` / `tests/precompile.py`. Module and path renames:
  `internal/tools/dev/internal_split.py` (the generated move, re-runnable).
- **`tools/video` is the library module `raywarp.video`** (`from raywarp.video import
  recorder`, unchanged API; NVENC through the new `video` extra); `tools/gltf_to_usd.py` and
  `tools/gltf_to_mjcf.py` are `raywarp convert` and the module `raywarp.integrations.mujoco.from_gltf`.
- **No lint / test / type-check workflows in GitHub CI** (`.github/workflows/ci.yml` removed):
  the gates are run by developers (docs/DEVELOPMENT.md section 4.3); the release workflow builds
  and publishes the wheels.

- **Packaging: `dev` and `bench` are dependency groups (PEP 735), no longer extras.** `pip install
  "raywarp[dev]"` / `uv sync --extra dev` become `uv sync` (the `dev` group is uv's default) and
  `uv sync --group bench`; the `dev` group no longer pulls in the `mujoco` extra (`uv sync
  --all-extras --group bench` is the full development environment).
- **Package restructure: the integrations and formats moved, the core is laid out by pipeline
  stage.** `import raywarp as rw` and its names, `rw.scene`, `rw.settings` and `rw.randomize`
  are unchanged. The simulator / framework integrations are `raywarp.integrations.<x>` and the
  scene formats `raywarp.formats.<x>` (no aliases at the old paths): `import
  raywarp.integrations.mujoco` then `rw.integrations.mujoco.scene(mjm, m)`, `import
  raywarp.formats.usd` then `rw.formats.usd.scene(path)`. `core/` holds packages only, by stage:
  `trace` (`rays`, `accel`, `optix` and the trace backends), `shade` (`materials`, `lights`,
  `shadows`, `gi`, `ao`, `environment`), `post` (`exposure`, `grading`, `display`, `aa`,
  `sensor`, `temporal`), `outputs`, `pipeline` (`Renderer`, frames, graphs, bakes, `gpu_scene`,
  `upload`, `textures`, `worlds`, `platform`), `build` (`variants`, `kernel_cache`) and `kernels`
  (the megakernel and its `abi`); `scene/` is `model`, `images`, `lighting`, `build`, and the leaf
  packages are `raywarp._base` (shared types) and `raywarp.cache` (the disk caches).
  `rw.settings` is the one settings package (`raywarp.core.settings` is gone) and also exports
  `RenderSettings`, `FAST` and `Outputs`. Every module, old -> new: research/kb/RESTRUCTURE_MAP.md
  (one table per restructure). The paths the docs cited, from 0.3.0:

  | Old (0.3.0) | New |
  |---|---|
  | `raywarp.mujoco` | `raywarp.integrations.mujoco` |
  | `raywarp.newton` | `raywarp.integrations.newton` |
  | `raywarp.newton.sensor.SensorCamera` | `raywarp.integrations.newton.sensor.SensorCamera` |
  | `raywarp.newton.headless.ViewerRaywarpHeadless` | `raywarp.integrations.newton.headless.ViewerRaywarpHeadless` |
  | `raywarp.mjlab` (entry point `raywarp.mjlab:create_groups`) | `raywarp.integrations.mjlab` (`raywarp.integrations.mjlab:create_groups`) |
  | `raywarp.isaaclab` (`class_type` `"raywarp.isaaclab.renderer:RaywarpRenderer"`) | `raywarp.integrations.isaaclab` (`"raywarp.integrations.isaaclab.renderer:RaywarpRenderer"`) |
  | `raywarp.usd` | `raywarp.formats.usd` |
  | `raywarp.usd.mdl` | `raywarp.formats.usd.materials.mdl` |
  | `raywarp.gltf` | `raywarp.formats.gltf` |
  | `raywarp.core.types` | `raywarp._base.types` |
  | `raywarp.core.settings` | `raywarp.settings` |
  | `raywarp.core.scene` (`GpuScene`, `CameraSpec`) | `raywarp.core.pipeline.gpu_scene.gpu_scene` |
  | `raywarp.core.renderer` | `raywarp.core.pipeline.renderer` |
  | `raywarp.core.texture_formats` | `raywarp.core.pipeline.textures.formats` |
  | `raywarp.core.tuning` (`override`) | `raywarp.core.build.variants.tuning` |
  | `raywarp.core.validate` | `raywarp.core.pipeline.worlds.validator` |
  | `raywarp.core.semantics` | `raywarp.core.outputs.semantic_classes` |
  | `raywarp.core.prebuilt` | `raywarp.core.build.kernel_cache.prebuilt` |
  | `raywarp.scene.lighting` (`with_lighting`) | `raywarp.scene.lighting.rig` |
  | `raywarp.scene.cache` (`.wpbr` files) | `raywarp.scene.model.wpbr` |
  | `raywarp.scene.cache.cache_dir` / `CACHE_ENV` | `raywarp.cache.paths` |
  | `raywarp.scene.prune` (`python -m`) | `raywarp.cache.prune` |

  Caches: `.wpbr` scene files, prepared scenes and the MDL conversion memo stay valid; bakes,
  textures and emissive fits miss once; the prebuilt kernel bundle is refilled (the Warp module
  names changed; until then the kernels compile at run time).
- **`Renderer.set_world_skip` is removed: `render(sim, active=ids)` / `capture(..., active=ids)`
  render only the listed worlds.** `active` is a (k,) int32 device array of distinct world ids
  (1 <= k <= nworld; -1 entries are holes); None renders every world. A short list launches k
  worlds' work; a fixed (nworld,) array rewritten in place each step (e.g. `where(done, -1,
  arange)`) suits captured graphs. Neither rebuilds nor compiles anything (a list of another
  length re-records the graph). Outputs stay (nworld, ...): unlisted rows keep their values.
  Out-of-range or repeated ids raise ValueError on the next render. **A world listed again after
  a skipped frame restarts its temporal state** (accumulation, previous poses and motion,
  metered exposure, sensor noise sequence), as does every skipped world when `active` goes back
  to None. Migration: `set_world_skip(mask)` -> `render(active=ids)` with the ids of the worlds
  to render. Newton's `SensorCamera` skips its invalid views this way.
- **`lighting.contact_ao_radius` is removed.** `lighting.contact_ao` is now computed from
  analytic box / sphere proxies of the moving instances (robot links, free or randomised
  objects), which need no radius: it darkens only the indirect light of the static scenery around
  them, never the moving objects themselves, without noise or history (docs/settings.md,
  `lighting`). A radial depth output with the contact AO on raises ValueError.
- **Per-world `camera.distortion` rows are 12 floats** (were 5): OpenCV's order k1, k2, p1, p2,
  k3, k4, k5, k6, s1, s2, s3, s4. `CameraSpec.distortion` / `CameraSpec.from_opencv` take 4, 5,
  8 or 12 coefficients, so OpenCV's rational (k4-k6) and thin prism (s1-s4) models render;
  existing 5-coefficient cameras render as before.
- **A mesh instance's uniform scale is part of its pose**: a scaled mesh instance's `instance.rot`
  holds s R (s its uniform scale, R its rotation) instead of reading a scaled copy of the mesh, so
  one mesh serves every size. Code writing per-world `instance.rot` columns of such instances
  writes s R (the column's initial values hold it); a plain R renders the mesh at scale 1.
  Non-uniform, sheared and mirrored scales are still baked into a mesh copy (one per distinct
  shape). Bindings that write rigid rotations (Newton, Isaac Lab) keep every scale baked.
- **`.wpbr` format 14, scene converter 13: prepare scenes again.** A file of format 13 is refused,
  never read wrongly: a `rw.usd.load` cache is rebuilt from its sources on the next load,
  prepared scenes must be prepared again (`rw.usd.bake`, `python -m tools.hf prepare`). Images of
  equal content are stored once and the second UV set's / vertex colors' corner indices only for
  the meshes that have them (smaller files). An MDL parameter without a default and without an
  authored value now takes its type's default-constructed value (reported
  `mdl.parameter_without_default:<name>(default-constructed)`) instead of failing the material
  (SimPBR road markings rendered as the magenta error material).
- **One default, no presets.** `rw.presets` (`fastest`, `training_fast`, `balanced`, `quality`,
  `omniverse`, `reference`) is removed. `rw.RenderSettings()` is the one default for every entry
  point, now the former `balanced` values (LTC area lights, 4 shadow casters, glossy reflections,
  4 layers, anisotropic filtering 8, register caps 128 / 0; probe GI and the environment sun as
  before): every group at its own defaults. `rw.FAST` (dotted fields) is the maximum-throughput
  field set: `rw.RenderSettings().replace(rw.FAST)` equals the former `training_fast` (point
  lights, no light cap, 2 casters, no reflection rays, opaque shadows, 2 layers, anisotropy 1,
  register caps 96 / 80; -29 to -67 % frame time against the default on RTX). The unlit, quality,
  reference and Omniverse RTX looks are field recipes (docs/settings.md "Recipes"); every one,
  `FAST` and the default render on prebuilt kernel tiers (`default`, `fast`, `unlit` and their
  variants), compiling nothing. Migration: `presets.balanced` -> `RenderSettings()`,
  `presets.training_fast` -> `RenderSettings().replace(rw.FAST)`, `presets.fastest` ->
  `RenderSettings().replace({"surfaces.shading": "unlit", "surfaces.anisotropy": 1})`, the others -> their recipe.
- **Adapters default to the default.** MuJoCo Warp's render context, Newton's `SensorCamera`,
  mjlab's sensors and Isaac Lab's renderer render `RenderSettings()` (they rendered
  `training_fast`): pass `rw.FAST` for the previous speed. Isaac Lab's `RaywarpRendererCfg.preset`
  is removed (`settings={**rw.FAST}` for maximum throughput). `rw.usd.settings` maps
  `rtx:rendermode` RealTimePathTracing / PathTracing (and unset) to the RTX look recipe,
  RaytracedLighting to the default and MinimalRendering to `FAST`.
- **Viewer and explain.** `python -m raywarp.viewer --preset` is `--base {default,fast}` (its
  command `{"type": "base"}`); the Scene Studio's API field `preset` is `recipe` (a
  `tools/recipes.py` name). `Renderer.explain()`'s cost classes are geometry, unlit, fast,
  default, quality, converged (`training` / `balanced` renamed).
- **The mjlab adapter is `raywarp.mjlab`** (was the root module `raywarp.mjlab_rgb`):
  `from raywarp.mjlab import RaywarpCameraSensorCfg`; the `mjlab.rgb_renderers` entry point is
  `raywarp.mjlab:create_groups` (reinstall the package to refresh it).
- **One light description, `rw.Light`** (the UsdLux record): `rw.LightSpec` is gone; make lights
  with `Light.sun(direction, illuminance=)`, `Light.point(position, intensity= | power=,
  radius=)`, `Light.spot(position, direction, cone_angle=, ...)` (IES profiles:
  `ies=rw.Ies(path, mode=, axis=, direction=)`), `Light.sphere / rect / disk / tube` (radiance in
  nits). Intensities are UsdLux's (intensity x color; `LightSpec` divided by max(color)); a sun /
  point / spot without one is display-relative. `rw.SceneLighting` is `rw.Lighting`,
  `rw.HeadlightSpec` is `rw.Headlight`; the settings group formerly `rw.Lighting` is
  `rw.Lights` (`RenderSettings(lighting=rw.Lights(...))`). `rw.mujoco.add_lights` raises for a
  light MJCF cannot hold (area, IES, a soft cone) instead of dropping it.
- **One adapter entry shape and `rw.SceneOptions`**: `rw.<adapter>.scene(source, *, options=,
  adapter=, lighting=, device=)` returns the uploaded, bound GpuScene; `rw.usd.load` /
  `rw.newton.load` return the host Scene (`rw.usd.scene` / `rw.newton.scene` did);
  `rw.newton.scene_from_usd` is `load_usd`. The shared options (texture storage and the one
  texture size cap, emissive panel lights, the environment sun) are `rw.SceneOptions`
  (`rw.upload(host, options=...)` replaces `textures=` / `env_sun=` / `emissive_lights=`; the
  USD / Newton `max_texture_size` is `SceneOptions.textures.max_size`); adapter options are typed
  objects: `rw.usd.Options(...)` (no `**options`; `displacement` is a bool),
  `rw.newton.Options(merge_worlds=, shapes=, checkerboard=, load_textures=)`,
  `rw.mujoco.Options(geom_groups=, use_skybox=, unset_metallic=, unset_roughness=)`. The Newton
  viewers take `scene_options=` instead of `max_texture_size=`.
- **`Instances.world` / `Lights.world` removed** (no adapter wrote them; `SceneBuilder`
  `world=` and `ALL_WORLDS` with them). Converted scenes are rebuilt once (converter version 4).
- **Unused features removed (no shims)**: the plugin system (`raywarp.plugins`,
  `RenderSettings.plugins`, `Renderer.plugin_field`, the `raywarp.plugins` entry points; the
  image blur plugin is gone), adaptive edge AA (`aa.adaptive`), the lens flare (`lens.flare*`),
  IES profiles on LTC area lights (`lighting.ies_area_lights`: profiled area lights are shaded
  in their point form, as by default), bounded GI emission bounces (`gi.bounces`; the USD map's
  `rtpt:maxBounces` is now reported as dropped), multiple stochastic glossy rays per sample
  (`reflections.samples`), BC1/BC3 colour storage (`TextureStorage(color="bc1")`), CPU emulation
  of block textures (`TextureStorage(emulate=)`), and `upload(transmission_alpha=)`.
- **TAA and the OptiX AI denoiser removed (no shims)**: `aa.temporal="taa"` and `post.denoise`
  (`"optix"`, `"optix_temporal"`) are gone, with the `Tuning.denoise_workers` knob. `aa.temporal`
  is `"off"` or `"accumulate"` (static views, unchanged); moving views render per frame (`aa.spp`
  for anti-aliasing). No preset used either: every preset renders as before.
- **Optional effects removed (no shims)**: the reflection filter (`reflections.filter`), bloom
  (the `bloom` group), the 3D LUT (`grading.lut`, `lut_strength`), the lens vignette
  (`lens.vignette`, `vignette_strength`: use `sensor.vignette`), the EV false colour
  (`exposure.false_color`: a host-side `ev_false_color` lives in the diagnostics), the sun-disc
  specular (`lighting.sun_specular`; a sized sun keeps its widened highlight), `aa.jitter`
  (jitter is implied by `aa.temporal="accumulate"`), `reflections.shadows="all"`, stochastic
  glossy reflections (`reflections.glossy="stochastic"`, `reflections.clamp`; the USD
  `rtpt:fireflyFilter` key is reported as ignored), the height fog and its sun in-scattering
  (`atmosphere.height_fog`, `density`, `height`, `falloff`, `max_opacity`, `environment_tint`,
  `inscatter`, `sun_light`, `sun_*`; distance fog and haze stay) and the sky occlusion
  (`environment.sky_occlusion`: with GI off the environment lights every surface unoccluded).
  `lens.aberration` is now the strength itself (a float, 0: off; `aberration_strength` is gone):
  run-time data of the deferred display, per world, compiling nothing. No preset changes look or
  speed (`omniverse` loses its moot `reflections.clamp`).
- **The thin lens is removed**: `CameraSpec.aperture` / `focus_distance`, `Renderer.set_lens` and
  `Renderer.optics` are gone (defocus at 1 sample per pixel is noise; no adapter mapped it). USD
  `fStop` / `focusDistance` are still read into the scene's camera data and render as a pinhole.
- **`rw.randomize`** replaces `raywarp.core.variation` (`jitter`, `contrast`); `contrast`
  accepts layered (15-role) texel-map tables. The adapter conformance kit moved to the tests
  (`tests/fixtures/conformance.py`).
- **Material textures: 8-bit sRGB color by default on CUDA, everything else exact; block
  compression is gone** (docs/settings.md "Texture storage"): `rw.TextureStorage(color=,
  max_size=)`; `color="srgb8"` (the default, `rw.DEFAULT_STORAGE`) is decoded by the texture
  unit before filtering, `"float16"` is exact (`rw.EXACT_STORAGE`, and the CPU). The BC7 / BC4 /
  BC5 storage, its GPU encoder and the `data=` / `normal=` options are removed (BC changed Bistro
  visibly and made the cold upload 2.6x slower for ~5 % more worlds). Every texture's levels are
  cached on disk (`$RAYWARP_CACHE_DIR/textures`), so a repeat upload reads them.
- **`RaywarpExt.normal_scale` is gone: a normal map's strength and flips are its binding's
  per-axis scale / bias** (`rw.scene.normal_map`, docs/scenes.md "Normal maps"). The core now
  honours any finite per-component scale / bias on `Slot.GEOMETRY_NORMAL` exactly (it used to read
  every binding as the plain unpack, reported `gpu.material.normal_scale_bias`): +Y-down (DirectX)
  maps authored as UsdUVTexture scale (2, -2, 2, 1) / bias (-1, 1, -1, 0), MaterialX vector2
  `normalmap` scales (were averaged, `mtlx.normalmap.scale(mean of x and y)`) and glTF
  `normalTexture.scale` render as authored; strengths and offsets compile a scaled decode
  (`Features.scaled_normal_map`: Toksvig on the fetched texel, the texel map unclamped). A MaterialX `normalmap` with frame inputs or a non-tangent
  `space` goes through the node graph (reported) instead of being read as a tangent-space image.
  `.wpbr` version 9.
- **`rw.Material` is an OpenPBR Surface 1.1.1 material** (docs/scenes.md, "The material"): the
  spec's input names and defaults, the other OpenPBR inputs in `extras=rw.OpenPbrExtras(...)`,
  the raywarp extensions in `ext=rw.RaywarpExt(...)`. Every adapter writes it; the device values
  are derived from it at upload. Renames: `metallic` -> `base_metalness`, `roughness` ->
  `specular_roughness`, `specular` (level s) -> `specular_weight` (w = 4 s²), `ior` ->
  `specular_ior`, `transmission` -> `transmission_weight`, `emissive` / `emissive_strength` ->
  `emission_color` / `emission_luminance`, `opacity` (and the base color's alpha: `base_color`
  is RGB) -> `geometry_opacity`, `clearcoat` / `clearcoat_roughness` -> `coat_weight` /
  `coat_roughness` (a glTF-style coat: `MaterialFlags.COAT_GLTF` at `coat_ior` 1.5),
  `MaterialFlags.THIN_WALLED` -> `geometry_thin_walled`; `alpha_mode`, `alpha_cutoff`,
  `normal_scale`, `occlusion_strength` and `flags` move to `ext`. `rw.Slot` members follow the
  inputs (`METALLIC` -> `BASE_METALNESS`, `ROUGHNESS` -> `SPECULAR_ROUGHNESS`, `NORMAL` ->
  `GEOMETRY_NORMAL`, `EMISSIVE` -> `EMISSION_COLOR`, `OPACITY` -> `GEOMETRY_OPACITY`,
  `TRANSMISSION` -> `TRANSMISSION_WEIGHT`, `SPECULAR` -> `SPECULAR_WEIGHT`).
  `rw.openpbr.OPENPBR_DEFAULT` is gone (it is `rw.Material()`). Load-report keys: the
  `openpbr.<input>` drops of what the core does not render are `gpu.material.<input>` (reported
  by the upload for every source; `openpbr.<input>` now only counts clamped values).
  `.wpbr` caches are rebuilt.
- **The defaults are OpenPBR's, not glTF's**: `rw.Material()` is base color 0.8 (was 1),
  `specular_roughness` 0.3 (was 1), `specular_ior` 1.5, `coat_ior` 1.6. A host material that
  relied on the old defaults renders differently: write `base_color=(1.0, 1.0, 1.0),
  specular_roughness=1.0` for the old look. The adapters set every value explicitly (no pixel
  change for MuJoCo, USD, MaterialX or Newton content).
- **Per-world material fields are OpenPBR's**: `PerWorld(material=[...])` / `Renderer.per_world`
  take `base_metalness`, `specular_roughness`, `specular_weight` and `emission_luminance`
  (were `metallic`, `roughness`, `specular`, `emission`). `specular_weight` and
  `emission_luminance` are derived into the device's `material.reflectance` / `material.emission`
  by one kernel per frame (equal to a fresh upload at float32 values, within 1 ulp for
  float64-authored weights); those device slots (renamed from `material.specular`;
  `material.metallic` / `material.roughness` are `material.base_metalness` /
  `material.specular_roughness`) are not declarable themselves. MuJoCo: per-world
  `material.specular` held MuJoCo's level s; `specular_weight` holds w = 4 s² (the device
  reflectance stays s), starting from the live `m.mat_specular` / `m.mat_emission`.
  `light.intensity` is `light.strength` (peak cd / lux). Caveat: a material emitting nothing at
  upload has no emission map, so a per-world `emission_luminance` L lights it as base color x L x
  max(`emission_color`), not `emission_color` x L (the declaration warns); on an emissive panel
  made into a rect light a per-world emission changes the visible surface only, not the light or
  the GI.
- **Lights are UsdLux records**: `SceneBuilder.add_light(rw.Light(kind=rw.LightKind...,
  intensity=, exposure=, color=, ..., shaping=rw.Shaping(...), ext=rw.LightExt(...)))` and
  `add_environment(rw.scene.EnvironmentLight(...))` replace the keyword light / environment rows; the
  host tables hold the UsdLux inputs and the device values (peak cd / lux, radii, cones) are
  derived at upload. Authored values outside UsdLux's ranges are clamped and reported (a
  DistantLight `angle` above 180 deg, cone angles / softness, color temperatures), and a light
  that cannot be derived is dropped and reported (`light.invalid`) instead of failing the stage.
  `.wpbr` is version 8 (rebuilt on load).
- **Color spaces use the Color Interop Forum / OCIO names** (`scene.colorspace.ColorSpace`:
  `srgb_rec709_scene`, `lin_rec709_scene`, `lin_ap1_scene` (ACEScg), `lin_ap0_scene`, the
  P3-D65 and Rec.2020 spaces, `data`, with the usual aliases); the SRGB / LINEAR enum is gone.
  Textures in other scene-referred spaces are converted to Rec.709 at upload (8-bit kept as 8-bit
  sRGB, float as linear float; above-1 emission as float16) and reported, instead of being read
  as linear.
- **Adapters are top-level subpackages, imported on their own**: `import raywarp.mujoco`, then
  `rw.mujoco.scene(...)` (likewise `raywarp.usd`, `raywarp.newton`, `raywarp.isaaclab`);
  `raywarp.adapters` is gone and `import raywarp` imports the core eagerly (no lazy
  attributes). The Newton drop-ins and the Isaac Lab classes are imported from their modules:
  `from raywarp.newton.sensor import SensorCamera`, `raywarp.newton.viewer.ViewerRaywarp`,
  `raywarp.newton.headless.ViewerRaywarpHeadless`, `from raywarp.isaaclab.cfg import
  RaywarpRendererCfg`. The conformance kit is `raywarp._common.conformance`.
- **One lighting input for every adapter**: `lighting=rw.SceneLighting(lights=[rw.LightSpec(...)],
  ambient=..., envmap=..., envmap_intensity=..., envmap_rotation=..., headlight=rw.HeadlightSpec(...),
  ev100=...)` replaces the `lights` / `envmap` / `envmap_intensity` / `envmap_rotation` /
  `headlight` / `ev100` keywords of `rw.mujoco.scene`, `rw.newton.scene`, `scene_from_usd`,
  `SensorCamera`, `ViewerRaywarp` / `ViewerRaywarpHeadless` (whose `**render_options` are now
  `RenderSettings` keys only) and the `default_lights` / `envmap` fields of Isaac Lab's
  `RaywarpRendererCfg`. `rw.newton.Light` / `Headlight` are `rw.LightSpec` / `rw.HeadlightSpec`.
  `RaywarpRendererCfg(default_lights=False)` is `lighting=rw.SceneLighting()` (no light; None
  keeps the key and fill lights).
- `SensorCamera.Light` (an alias of `rw.newton.Light`, which Newton's sensor does not have) is
  removed: use `rw.LightSpec`.
- `ViewerRaywarp` / `ViewerRaywarpHeadless`: any `lighting=` replaces the default ViewerGL look (a headlight
  alone included). `SensorCamera`, the viewers and Isaac Lab's renderer render at
  `post.ev100 = lighting.ev100` unless a setting gives `post.ev100`.
- **Typed output requests**: `rw.Outputs` has one field per output holding its spec
  (`rw.ColorOutput`, `DepthOutput`, `IdOutput`, `FloatOutput`, `NormalOutput`, `ShadowOutput`,
  `LuminanceOutput`, `MotionOutput`) or None; `Outputs.of("rgb", "depth")` requests by name,
  `Outputs.parse` from plain data. It replaces `Outputs(kinds=..., depth=, miss_depth=,
  segmentation=, layout=)` and the `"outputs.kinds"` keys: `"alpha"` is
  `ColorOutput(alpha=True)`, `layout="rgb8_chw"` / `"rgb16f_chw"` is `ColorOutput(layout="chw",
  dtype=...)`, `segmentation="semantic"` is the `semantic` output (`RenderOutputs.semantic`),
  depth conventions are `DepthOutput(convention=, miss=)`. `RenderOutputs` is generic over
  (NWorld, NCam, H, W) with one field per output. Requesting albedo or normal alone writes (and
  allocates) only that one.
- **`gi.emission` is removed** (user decision, robotics defaults): emitted light always bounces
  like any other light in the probes (the former `"full"`). The `"glossy"` mode (RTX
  RealTimePathTracing's shortcut, `omniverse`'s until now) and `"off"` are gone; `omniverse` now
  bakes emission like every preset (its Office look moves away from RTX's).
- **The `lens` settings group is folded into `sensor`:** `lens.aberration` /
  `rw.settings.Lens(aberration=)` are `sensor.aberration` / `rw.settings.Sensor(aberration=)`
  (same values and behaviour; per world `PerWorld(env=["sensor.aberration"])`).
- **`gi.hysteresis` is removed:** the periodic GI updates keep 0.97 of the previous maps
  (`features/gi/schedule.HYSTERESIS`).

### Added

- **`raywarp`**, a small command over the Python API (docs/cli.md): `view` (the web viewer),
  `record SCENE OUT.mp4` (an MP4 of a scene file, the camera orbiting; several worlds in one
  mosaic), `convert IN.gltf OUT.usdc|OUT.xml` (glTF -> USD / MJCF), `prepare IN.usd -o OUT.wpbr`
  (a stage baked to a scene file), `kernels info | fetch [--arch --driver]`, `cache info | prune
  --max-gb N`, `info` (GPU, compute capability, RT cores, driver, OptiX, Warp, the traversal
  "auto" picks, the kernels and whether their remote answers).
- **`rw.kernels`**: `configure(download=..., compile=...)` and the environment variables
  `RAYWARP_KERNELS_DOWNLOAD` / `RAYWARP_KERNELS_COMPILE` (on by default). Download off: nothing
  is fetched, the local store only. Compile off: a configuration whose kernels are not prebuilt
  (settings outside every tier, naming the differing fields and the nearest tier;
  `performance.kernels="scene"`; the CPU; a PTX or first-frame Warp module the bundle lacks)
  raises `rw.kernels.KernelsNotPrebuilt` instead of compiling. `info()`, `detect()`, `fetch(arch)`
  (an offline mirror; replaces `prebuilt.prefetch` in the docs), `pinned_architectures()`. The
  first renderer on a GPU picks its architecture's pinned bundle (one per sm_80, 86, 89, 90, 120
  in `kernels.json`) and the driver's OptiX module pack (docs/install.md).
- **A minimal public benchmark**, `python -m benchmarks.run --scene franka|showcase|sponza
  [--worlds N --res R --settings fast|default]`: ms/frame, images/s, first frame, compile count, one
  PNG; the scene files (Menagerie's Franka, the showcase room, Intel's New Sponza: CC-BY 4.0)
  download on first use, versioned by scene-file format and USD converter
  (`tools/release/benchmark_scenes.py` publishes them; benchmarks/README.md).
- The `video` extra (PyNvVideoCodec, not on macOS); docs/install.md, docs/cli.md, docs/benchmark.md.

- Extras `gltf` (`raywarp.gltf`: pygltflib and the `usd` extra; pygltflib was only in `bench`) and
  `all` (mujoco, newton, usd, gltf, viewer). Python 3.13 classifier; Documentation and Changelog
  project URLs.
- `gi.relight` (runtime): `"on_change"` (the default) re-lights a randomised world's GI by rays on
  the frame its GI inputs change; `"on_request"` waits for `Renderer.request_gi_relight(worlds)`
  (a (nworld,) bool device mask: no host read, captured graphs stay valid), e.g. once per episode.
  Changes of the lights' colours, intensities and on / off state and of the environment intensity
  are recombined from the last re-light without rays (they used to re-light by rays after any
  earlier material or ambient change), and the exposure is no longer a GI input (a per-world
  exposure re-lights nothing).
- Dataset ground truth, computed only when called: `Renderer.camera_info(out=)` (per world and
  camera: intrinsics K in pixels, lens model and coefficients, OpenCV camera-to-world pose,
  resolution, frame index; one launch, no host read), `Renderer.camera_normals(out=)` (the normal
  output in each camera's OpenCV frame) and `Renderer.nocs(sim, out=)` (normalized object
  coordinates of each pixel's hit). `benchmarks.datagen.sinks.CameraLog` records `camera_info`
  per frame to `cameras.npz` / `cameras.json` (benchmark tooling, not the library).
- Per-world uniform mesh scale as pose data: declare the instance's pose (`PerWorld(pose=[...])`)
  and write `instance.rot` as s R (s > 0) per world and frame: no recompile, no mesh copy, one
  refit per frame as any pose.
- `aa.post` (runtime, off by default): FXAA 3.11 post anti-aliasing of the final color across the
  geometric edges only (silhouettes, thin wires, alpha cut-outs, creases: found in the depth
  output), in the `sensor` pass; texture and shadow edges stay as rendered. Needs an `rgb` or
  `luminance` output and a `depth` output (ValueError). Off it costs nothing.
- `shadows.area_rays` (runtime, 0: off): soft shadows of large, near area lights by a pass after
  the render (needs LTC area lights, `aa.spp=1`, `aa.temporal="off"`; off, the render kernels are
  unchanged).
- `rw.scene.sun_sky(elevation, azimuth, *, turbidity=, ground_albedo=, height=)`: a procedural
  clear sky (Hosek-Wilkie) with a physical sun disc as an equirectangular radiance map for
  `rw.Lighting(envmap=...)`; the upload cuts the sun into a shadowed directional light. One sky
  per world through several maps and `per_world["env.id"]`.
- Glass switched per world: `PerWorld(material=["instance.material"])` or a per-world
  `specular_ior` in a scene with see-through content now switches the instance's refraction,
  shadow transmittance and two-sided / closed-pane behaviour per world too (it kept the authored
  material's). It compiles a variant of its own (`Features.world_optics`) only for such
  renderers; every other renderer runs unchanged kernels.
- Different objects in different worlds: Newton models whose worlds hold different shapes (also
  different height fields) are pooled instead of raising, each world showing its own objects (no
  recompile when `world_indices` remaps them); Isaac Lab multi-asset spawning shows each env its
  own prototype. `rw.newton.load_usd` and `rw.newton.binding` take `world_homes=` (per world, the
  world root whose world-local content it shows).
- `raywarp.mujoco.mjx`: `feed(d, mjx_data)` copies an MJX / MuJoCo Playground (JAX) state's poses
  into MuJoCo Warp `Data` (stable pointers: captured graphs replay), `to_jax(r.outputs)` views the
  outputs as JAX arrays without a copy or host sync.
- Motion blur and rolling shutter (`sensor.motion_blur`, `sensor.rolling_shutter`) streak moving
  objects over the static background behind them (they stayed inside their own silhouette).
- Test scenes: `polyhaven_apartment` (built here from CC0 Poly Haven assets), `isaac_warehouse`,
  `rivermark` (outdoor town), `moore_lane` (house and garden, 230k plant instances) and three
  scenes of Bitterli's rendering resources in the reference corpus (`bitterli_*`).
- MaterialX Standard Surface and glTF PBR surfaces map to OpenPBR input by input
  (`raywarp.scene.surfaces`), and MaterialX node graphs are bound (affine functions of one
  texture) or baked into textures (75 nodes, `scene.mtlx_graph.SUPPORTED`), for USD and `.mtlx`
  sources.
- `.mtlx` documents referenced from USD (sublayers, references, payloads, in prototypes too)
  provide the materials their prims bind; read once through the asset resolver with size limits;
  unmapped materials, path collisions and unfollowed arcs are reported; they are `.wpbr`
  dependencies.
- USD DDS textures (decoded with Pillow, sRGB / BGRA remaps, BC5 normals; cube, volume, array,
  BC6H, BC4 SNORM and truncated files are rejected and reported).
- New outputs, each compiled in only when requested: `position`, `uv`, `material`, `roughness`,
  `metallic`, `emission`, `direct` / `indirect` (the lighting split: hdr = direct + indirect +
  emission on the hits of an opaque, unfogged spp-1 render), `shadow` (the visibility of one light), `luminance` (a grey
  camera), `semantic` together with `seg`, a geometric `normal`, and float16 variants of every
  float output (`dtype="float16"`). Definitions and costs: docs/robotics.md, "Outputs: what you
  can request and what it costs".
- The geometry-only kernel: a render without shaded outputs traces one closest hit per pixel
  through its own kernel / OptiX program (no material, light or environment tables, no shading
  code), and collapses the color's anti-aliasing and temporal settings (one kernel for every
  preset). Its OptiX programs always use 64 registers (`performance.optix_max_registers` caps
  the shaded programs only).

- `rw.SceneLighting.studio(up=, fill=, replace=)`: a neutral studio rig (sky dome, shadowed key,
  weak fill) exposed for the default `post.ev100`, for any scene; `SceneLighting(replace=True)`
  drops the scene's own lighting; `rw.mujoco.add_lights(spec, lighting)` adds lights to a MuJoCo
  spec before compiling.
- `rw.usd.scene(..., lighting=)` and `Scene.with_lighting(lighting)` (a built or cached host
  scene, before `rw.upload`); `SceneLighting.ambient` (a constant ambient fill); MuJoCo
  envmaps in any image format (8-bit sRGB, EXR), not only `.hdr`. Docs: "Lighting a scene"
  (scenes.md) and "My scene is dark: why?" (faq.md).
- `rw.TextureStorage(blocks=True)` (opt-in): block-compressed material textures (BC1 / BC3 sRGB
  color, BC4 / BC5 data; normal maps as BC5 plus a BC4 companion, z and Toksvig kept), sampled by
  the CUDA texture unit (BC CUDA arrays created through the driver, no new dependency). Isaac
  Office / Hospital: 3.8-3.9x less GPU memory than exact, 720p frame time within -4 to +1 % and
  the training batch +4-6 %, the same first-frame compile, FLIP 0.02-0.06 (docs/settings.md
  "Texture storage"). Alpha-tested
  opacity maps are compressed too. On the CPU the blocks are decoded at upload. The encoder (BC1:
  principal axis, cluster fit, optimal single-color endpoints; BC4 min / max) runs on the upload
  device; encodes are cached on disk (keyed by texels, format, encoder sources, Warp version and
  device kind; damaged files are rebuilt, failed writes warn).

### Fixed

- A directional light turned per world (an environment sun under a per-world `env.yaw`) kept no
  light-grid culling: every other world evaluated every light (randomised steady frames: Office
  311 -> 61 ms, Bistro 186 -> 124 ms at `FAST`, 512 worlds).
- The light grid no longer floods lights that emit nothing (unless their emission is written per
  world or by the simulator), and its bake is bounded (the cell grows until at most 2^29 tests):
  about 29x less bake work on the Rivermark town, whose first frame spent minutes there.
  `lighting.light_grid_cell` is a minimum.
- Thin-walled transmissive materials reflect at both interfaces of the pane (OpenPBR's thin wall,
  2F / (1 + F)); they reflected at one.
- `sun_sky` no longer raises at small map heights; `light.range` accepts +inf per world; the
  per-world range error names the failing field, world and value.
- `update_settings` with unchanged values returns at once (captured graphs stay valid), and
  per-world `set_semantics` reads nothing back from the device.
- A MaterialX bake that reads the position is reported `mdl.baked_position.<slot>`, no longer as an
  exact UV bake.
- USD cameras: `fStop <= 0` means no depth of field; invalid camera values are clamped and
  reported, and a camera that cannot be built is dropped and reported (every stage of the
  Automotive, Marbles and SimReady corpus loads).
- A DomeLight's pole is its local +Z in any stage (Omniverse RTX's convention; a Y-up stage's
  dome stood sideways).
- Newton mesh textures from HDR / EXR files were decoded as sRGB (they are linear).
- Texture uploads on worker threads no longer change process-wide Warp state: device -> host reads
  go through `core.device.to_host` (Warp 1.18's `array.numpy()` swaps the device's current stream,
  so threads interleaving could leave the null stream current), the texture pool and the OptiX PTX
  compile thread pass devices explicitly instead of entering `wp.ScopedDevice` (process-wide
  default device). Upstream report draft: `research/upstream/warp_numpy_thread_safety.md`.

### Changed

- **MuJoCo Warp: a batched `m.mat_shininess` is read per world** (was only warned about): the
  roughness 1 - shininess of the materials that leave `roughness` unset, as the scene upload maps
  it, re-read every frame (one small kernel per frame, only for such renderers).
- **`ViewerRaywarpHeadless` accumulates nothing by default** (`accumulate=0`, was 32): the default
  is the prebuilt build, so a Newton user's first frame loads instead of compiling; pass
  `accumulate=N` for converged still views (another build, compiled on its first frame).
- **Inactive worlds cost less.** Under `render(active=...)` the deferred display, the exposure
  metering and the exposure solve launch over the listed worlds only, and `RenderOrder.AUTO`
  compares the listed worlds' cameras only; the render kernels are unchanged.
- **Less host memory and fewer mesh copies at upload**: uniformly scaled mesh instances share one
  mesh (Office 580 -> 53 mesh copies, the Isaac warehouse 81 -> 2, Rivermark 38,290 -> 10,936), and
  the mesh normal / UV / color pools build without float64 temporaries (Rivermark's upload peak
  host memory about halved).
- **Faster `sensor` pass**: the 8-bit display is decoded through a 256-entry table (the camera
  effects up to about 1 ms cheaper at 512 worlds x 512^2 on an RTX PRO 6000).
- **Prebuilt bundle**: MuJoCo Warp's model-independent module-level kernels (21 modules, for the
  bundle's mujoco-warp release) ship in the common archive, so a first `mjw.forward` after the
  renderer's construction loads them instead of compiling. OptiX module names in the bundle are
  stable across refills (the normalised PTX is reproducible: no program hash suffix, programs in
  a fixed order), so a refill can reuse the published pack's unchanged modules.
- **Less GPU memory for textures and GI structures.** A material texture of one value is stored
  as a single texel (same sampling), a grey sRGB color texture as one 8-bit channel with the
  texture unit's sRGB decode, and the OptiX path no longer builds the software mesh BVHs for the
  GI bake / updates its hardware rays trace (only for the relight kernel).
- **The bake cache no longer sweeps legacy `*.tmp` files** left by writers older than its
  per-writer temporary names (`core/bake_cache.py`): delete stale `*.tmp` files under
  `<cache dir>/bakes/` (`~/.cache/raywarp` by default) by hand if an old version left any.
- **`quality`: 2 shadow samples instead of 8** (user decision after the post-release preset study;
  spp 2, the fine GI grid and the rest unchanged): the same FLIP vs `reference` (Office 0.083 =
  0.083, Sponza 0.062 vs 0.059) for 44 % less frame time on the Office and 27 % on Sponza
  (`results/history/post_release/presets/` item 3, `results/history/post_release/quality_preset/`).
  `reference` is unchanged (its values are now explicit instead of inherited from `quality`).
- **`balanced` (the `RenderSettings()` default) tone maps with PBR Neutral again** (user decision,
  robotics defaults): `aces_rtx` turned every HDRI-lit tabletop milky (+0.3..+0.9 EV, FLIP
  0.39-0.48 vs PBR Neutral, `results/history/review_c/robotics_fit/`). Only `omniverse` keeps
  `post.tone_mapper="aces_rtx"`; every other preset now shares one tone curve.
- **`lighting.light_grid` defaults to `"auto"`: the light grid is built from 8 lights** (user
  decision, robotics defaults; generalises the single-sun rule). With fewer lights the grid's
  lookup and dynamic light loop cost more than they cull (robotics scenes with 1-5 lights at
  512 px on an L4: grid off 0-4 % faster; 8 lights in closed rooms: grid on 7 % faster);
  `light_grid=True` forces it. Pixels: identical on open scenes; few-light scenes with walls move
  by FLIP <= 0.014 (lights the grid culled now compete for the shadow budget and light caps).
- **`balanced` (the `RenderSettings()` default) renders without probe GI** (user decision,
  robotics defaults): against a Mitsuba path tracer at 512 px, after the probe-placement fix, GI
  off is closer on the robotics tabletops (pick table FLIP 0.09 vs 0.17, with the Panda 0.13 vs
  0.18, env-lit bin 0.46 vs 0.56; G1 outdoors 0.061 vs 0.053) and 12-27 % cheaper per frame
  (`results/history/post_release/robotics_defaults/`). `gi.enabled` defaults to False; the grid
  fields keep the 1 m / 64 rays / 8 updates opt-in for interiors lit through their windows
  (`settings.replace(**{"gi.enabled": True})`). `quality`, `omniverse` and `reference` keep GI.

## 0.3.0 (2026-10-28)

The `core-refactor` release: correctness, typing, structure, performance and the robotics features of
the refactor plan (`research/plan/quality.md`). Milestone reports: `results/history/<milestone>/`.
Backward compatibility was not kept (no shims): see "Breaking changes".

### Highlights

- **Faster**: on an RTX PRO 6000, the default `balanced` preset at 4096 worlds x 64^2 went from 129k to
  626k camera-frames/s (adaptive AA off below 128 px); OptiX compile time per program halved; OptiX
  helmet balanced +20 % (one shading copy per adaptive-AA build); `training_fast` +12-14 % on RTX
  (OptiX register cap 96); light-heavy scenes several times faster (light caps for every light form,
  packed lights: Office 960 x 540 balanced 104 -> 7.8 ms on an L4); GI bake ~7x faster on CPU; light
  grid build by flood fill (Hospital 1609 -> 8 s on CPU); OptiX frame structs cached (host time per
  frame 1065 -> 5 us).
- **Looks right**: `balanced` has coarse GI (no black indirect-only areas); the Isaac Hospital matches
  Omniverse RTX within 0.1 EV in `omniverse` (emissive fixtures become lights); spot cones follow
  UsdLux; OmniPBR emission scaled per MDL; world-projected MDL UVs; GI leak at thin features fixed.
- **Correct**: about 100 bugs fixed across M1-M10, Phase A and the Phase B per-file pass (33 there),
  each with a regression test; pixel changes recorded and attributed in the regression references
  (`results/regression/manifest_m*.json`, latest m7).
- **Typed and checked**: `pyrefly check warp_pbr` 0 errors, no `Any`, every host array shaped, ruff S7
  rule set, `tools/audit.py --fail-on-any` (function / file size, docstrings, In/Out groups); CI runs
  them with the CPU suite.
- **Robotics features**: depth conventions (planar / radial, miss value), semantic segmentation,
  per-camera lens (resolution, clip, exposure, K, OpenCV distortion), `znear` near clip, fog / haze,
  CHW training layouts, per-world randomisation of materials / lights / visibility (a cheap mask on
  shared instances) / texture variation, `fastest` (unlit) and `depth_seg` presets, a MuJoCo
  Warp-style API, adapter conformance kit, `wpr.prewarm` + bake disk caches, Newton / MuJoCo zero-copy
  residency (no per-frame host transfer, one CUDA graph).
- **Docs**: `docs/` user guide (snippets executed by the tests), `docs/DEVELOPMENT.md` with diagrams.

### Breaking changes

- Package layout: `warp_pbr.scene` (host scene, `.wpbr`), `warp_pbr.core`, `warp_pbr.adapters.{mujoco,usd,newton}`;
  `newton-pbr` is folded into `warp-pbr[newton]` (`wpr.newton.SensorCamera`, `ViewerPbr`).
- Settings are grouped dataclasses (`RenderSettings(aa=..., shadows=..., ...)`, `presets.*`); one
  default (`RenderSettings()` is `presets.balanced`).
- `wpr.newton.scene` / `scene_from_usd` return a host `Scene` (upload with `wpr.upload`).
- `wpr.prewarm(scene, settings, *, nworld=, resolution=, ...)`: one scene per call.
- `PerWorld(alpha=True)` declares a per-world alpha (colour tables are RGB tints).
- Removed: `wpr.load_hdr` (use `wpr.read_hdr`), Metal support (until warp-metal supports Warp >= 1.18),
  the OptiX register auto-tune, research knobs and environment switches.
- Requires Warp >= 1.18, Python >= 3.12.

### Milestones

M0 safety net (regression suite, audit), M1 severe bugs, M2 foundation (types, settings API,
adapters), M3 pilot pass, M4 subsystem passes, M5 features (lighting, outputs + API, caches), M6
integration (S7 gates, GPU-4), M7 kernel lab (one shading copy, register caps; SER rejected), M8 look
development (RTX reference tooling, Hospital, emission, spot cones, GI), M9 docs, M10 bug hunt + visual
sweeps, Phase A (user decisions, W17 preset validation), Phase B (per-file quality pass of every file).
