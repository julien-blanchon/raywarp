# Shot list: the docs' images and clips

Every image or clip the docs reference, for the SHOWCASE lane to record in one GPU job. Paths are
under `docs/assets/`. Stills: lossy WebP (quality 85, under 150 KB) unless noted. Clips: H.264 MP4
(CRF 28, no audio, under 2 MB) plus an animated WebP preview of the same name (12 fps, 640 px wide,
under 1 MB) that the page shows and links to the MP4. Tiles are separated by 4 px of white and
labelled in the caption, not in the image. Scenes: the `benchmarks/run.py` scenes (franka, showcase,
sponza), the fixtures in `tests/data/scenes/` and Poly Haven CC0 HDRIs; no NVIDIA assets.
Every image is looked at before it is committed. The pages already reference each file under a
`<!-- SHOWCASE: <file> -->` comment; a clip's WebP preview links to its MP4. "front" is the scene's
benchmark view (`benchmarks/run.py`) or its MJCF camera `cam`.

| file | kind | page, section | scene | settings | camera | size |
| --- | --- | --- | --- | --- | --- | --- |
| `readme/hero.webp` | still | README.md, Raywarp | franka (benchmarks/run.py scene, MuJoCo Menagerie) + showcase (benchmarks/run.py scene) | quality recipe, per-world material / light / environment randomisation | the scene's front view, one per world | 1600x600 (a 4 x 2 mosaic of 400x300 worlds) |
| `features/batch_mosaic.mp4` + `features/batch_mosaic.webp` | clip | features.md, One call, many worlds | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default, per-world randomisation of materials and lights | front view per world | 8 s, 1280x720 (8 x 8 tiles of 160x90), mp4 + webp preview |
| `features/outputs.webp` | still | features.md, Outputs | franka (benchmarks/run.py scene, MuJoCo Menagerie) (arm in motion) | default; outputs rgb, depth, seg, normal, motion; r.nocs() | front | 1920x320 (6 tiles of 320x320) |
| `features/randomisation.webp` | still | features.md, Per-world randomisation | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; PerWorld material / light / env, random per world | front | 1920x270 (8 tiles of 240x270) |
| `quickstart/mujoco.webp` | still | quickstart.md, MuJoCo Warp | the snippet's MJCF | the snippet's | front and side | 960x360 (rgb / depth / seg rows, worlds 0-3) |
| `quickstart/usd.webp` | still | quickstart.md, USD and glTF without a simulator | the snippet's demo.usda | default | the stage's camera | 960x540 |
| `quickstart/newton.webp` | still | quickstart.md, Quick start | the snippet's Newton ant (ant.usda), 4 worlds | default + rw.Lighting.studio() | the snippet's pose | 960x540 (2 x 2 worlds) |
| `concepts/worlds.webp` | still | concepts.md, Worlds | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default, per-world materials | front | 960x540 (4 x 4 worlds) |
| `concepts/recipes.webp` | still | concepts.md, Settings | showcase (benchmarks/run.py scene) | FAST / default / quality | the scene's view | 1440x270 (3 tiles of 480x270) |
| `scenes/lights.webp` | still | scenes.md, Light types | tests/data/scenes/lights_punctual.usda, lights_area.usda, ies_fixtures.usda | default | View0 | 1920x270 (4 tiles) |
| `scenes/envmaps.webp` | still | scenes.md, Environment maps (HDRI) | tests/data/scenes/environment.usda with sky.hdr / studio.hdr + a Poly Haven CC0 HDRI | default | View0 | 1440x270 (3 tiles) |
| `scenes/exposure.webp` | still | scenes.md, Light units and exposure | showcase (benchmarks/run.py scene) | exposure.mode auto, manual ev100 8, 12, 15 | the scene's view | 1920x270 (4 tiles) |
| `scenes/materials.webp` | still | scenes.md, Materials and textures | tests/data/scenes/materials.usda | quality | View0 | 960x540 |
| `scenes/glass.webp` | still | scenes.md, The material: OpenPBR Surface | tests/data/scenes/glass.usda | default / default with shadows.translucent='clear' | View0 | 960x270 (2 tiles) |
| `scenes/camera_models.webp` | still | scenes.md, Cameras | showcase (benchmarks/run.py scene) | default | one mount, four CameraSpec projections | 1920x270 (4 tiles of 480x270) |
| `scenes/lens_distortion.mp4` + `scenes/lens_distortion.webp` | clip | scenes.md, Use fisheye and OpenCV cameras | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default, PerWorld(camera=['distortion']) | front, k1 swept over the clip | 6 s, 1280x720, mp4 + webp preview |
| `scenes/segmentation.webp` | still | scenes.md, Segmentation | franka (benchmarks/run.py scene, MuJoCo Menagerie) | outputs rgb, seg, semantic (set_semantics) | front | 1440x270 (3 tiles) |
| `settings/recipes.webp` | still | settings.md, Recipes | showcase (benchmarks/run.py scene) | the four recipes | the scene's view | 1920x270 (4 tiles of 480x270) |
| `settings/aa.webp` | still | settings.md, What each group does | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; aa.spp 1 / aa.post True / aa.spp 2 | front, crop on the gripper edges, nearest upscale x2 | 1440x360 (3 tiles) |
| `settings/shadows.webp` | still | settings.md, What each group does | showcase (benchmarks/run.py scene) | FAST / default / shadows.samples 8 | the scene's view | 1440x270 (3 tiles) |
| `settings/contact_ao.webp` | still | settings.md, What each group does | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; lighting.contact_ao 0 / 1 | close-up of the cubes on the floor | 960x270 (2 tiles) |
| `settings/reflections.webp` | still | settings.md, What each group does | showcase (benchmarks/run.py scene) | default; reflections.enabled False / True | the scene's view | 960x270 (2 tiles) |
| `settings/gi.webp` | still | settings.md, What each group does | tests/data/scenes/gi_window.usda and showcase (benchmarks/run.py scene) | default; gi.enabled False / True | View0 / the scene's view | 960x540 (2 x 2 tiles) |
| `settings/tone_mapper.webp` | still | settings.md, What each group does | franka (benchmarks/run.py scene, MuJoCo Menagerie) under an HDRI | default with each tone mapper | front | 1440x270 (3 tiles) |
| `settings/fog.webp` | still | settings.md, What each group does | sponza (benchmarks/run.py scene, Intel CC-BY 4.0) | default; atmosphere end 30 / haze 0.05 | courtyard view | 1440x270 (3 tiles) |
| `robotics/outputs.webp` | still | robotics.md, Outputs: what you can request and what it costs | franka (benchmarks/run.py scene, MuJoCo Menagerie) (arm in motion) | default; rgb, depth, seg, semantic, normal, albedo, roughness, metallic, direct, indirect, motion, shadow | front | 1920x540 (6 x 2 tiles of 320x270) |
| `robotics/point_cloud.webp` | still | robotics.md, Get point clouds and per-world labels | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; r.point_cloud over front + side, 16k points | a third view of the points (matplotlib 3D scatter) | 960x540 |
| `robotics/rolling_shutter.mp4` + `robotics/rolling_shutter.webp` | clip | robotics.md, Add camera sensor effects | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; sensor.rolling_shutter 1 / sensor.motion_blur 0.5 / both | front, orbiting fast | 6 s, 1280x360 (3 tiles), mp4 + webp preview |
| `robotics/sensor_noise.webp` | still | robotics.md, Add camera sensor effects | showcase (benchmarks/run.py scene) | sensor off / sensor.noise / vignette + response / sensor.depth stereo (depth tile) | the scene's view | 1920x270 (4 tiles) |
| `robotics/batch_mask.mp4` + `robotics/batch_mask.webp` | clip | robotics.md, Skip finished episodes with the batch mask | franka (benchmarks/run.py scene, MuJoCo Menagerie) (simulated) | FAST; active = even worlds after 2 s | front per world | 8 s, 1280x720 (4 x 4 worlds), mp4 + webp preview |
| `domain_randomisation/grid.mp4` + `domain_randomisation/grid.webp` | clip | domain_randomisation.md, How it works | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; PerWorld material / light / env / camera fovy | front per world | 8 s, 1280x720 (4 x 4 worlds), mp4 + webp preview |
| `domain_randomisation/materials.webp` | still | domain_randomisation.md, Randomise materials per world | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default | front | 1920x270 (8 worlds) |
| `domain_randomisation/lights.webp` | still | domain_randomisation.md, Randomise lights: count, intensity, colour | showcase (benchmarks/run.py scene) | default | the scene's view | 1920x270 (8 worlds) |
| `domain_randomisation/environment.webp` | still | domain_randomisation.md, Randomise exposure and sensor noise | franka (benchmarks/run.py scene, MuJoCo Menagerie) + Poly Haven CC0 HDRIs | default | front | 1920x270 (8 worlds) |
| `domain_randomisation/camera.webp` | still | domain_randomisation.md, Jitter the camera | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default | front, per-world offsets | 1920x270 (8 worlds) |
| `randomization/ground_truth.webp` | still | randomization.md, Export ground truth for a dataset | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; r.nocs(), r.camera_normals(), MotionOutput(status=True), r.camera_info() | front | 1600x270 (5 tiles) |
| `recording/mosaic.mp4` + `recording/mosaic.webp` | clip | recording.md, Every view on one frame: 8 x 8 tiles of 512² = 4096², so HEVC (NVENC's H.264 stops at 4096 px): | franka (benchmarks/run.py scene, MuJoCo Menagerie) | FAST; Recorder mosaic 8 x 8 | front per world | 10 s, 1280x720, mp4 + webp preview |
| `datagen/scene_switch.mp4` + `datagen/scene_switch.webp` | clip | datagen.md, What a switch costs | franka (benchmarks/run.py scene, MuJoCo Menagerie), showcase (benchmarks/run.py scene), sponza (benchmarks/run.py scene, Intel CC-BY 4.0), tests/data/scenes/* | FAST, 16 worlds, 1 s per scene | each scene's view | 10 s, 1280x720 (4 x 4 worlds), mp4 + webp preview |
| `performance/traversal.webp` | still | performance.md, OptiX | showcase (benchmarks/run.py scene) | default; performance.traversal 'optix' / 'software' / abs difference x10 | the scene's view | 1440x270 (3 tiles) |
| `faq/dark_interior.webp` | still | faq.md, Light a dark interior | tests/data/scenes/gi_window.usda | gi off / lighting.fill 0.05 / gi on | View0 | 1440x270 (3 tiles) |
| `cli/viewer.webp` | still | cli.md, Web viewer | showcase (benchmarks/run.py scene) | the viewer's default | the viewer page (screenshot) | 1280x720 |

## Demo gallery

The clips of [docs/gallery.md](../gallery.md), recorded by the scripts of `examples/demos/` in one GPU job on an RTX
GPU (OptiX: RT cores, NVENC) from a repository checkout with `pip install "raywarp[all,video]"`. Per demo:
`gallery/<name>.mp4` (6-8 s, 1280x720 unless noted, H.264 CRF 28, no audio, under ~2 MB) and `gallery/<name>.webp`
(animated preview, 640 px wide, 12 fps, looping, under ~1 MB). The demos record with `raywarp.video.recorder`
(NVENC at its default bitrate): re-encode each MP4 to CRF 28 before committing. Look at every clip (first, middle
and last frame at native size) before it is committed. Order: Newton and USD first, as in the gallery.

| files | recording command (GPU) | scene, credits, licence | notes |
| --- | --- | --- | --- |
| `gallery/newton_usd_robots.mp4` + `.webp` | `python examples/demos/newton_usd_robots.py --worlds 64 --size 1280x720 --out newton_usd_robots.mp4` | Newton example asset `cartpole.usda`, Newton project, Apache-2.0; sky `rw.scene.sun_sky` | 8 x 8 worlds, stage materials and lights (`load_usd`), per-world colours and start poses; Newton SolverFeatherstone |
| `gallery/newton_falling_objects.mp4` + `.webp` | `python examples/demos/newton_falling_objects.py --worlds 16 --size 1280x720 --out newton_falling_objects.mp4` | Stanford Bunny from Newton's example assets (`bunny.usd`), Newton project, Apache-2.0 package | 4 x 4 worlds, XPBD, per-world `shape_color` and drop layout |
| `gallery/newton_headless_viewer.mp4` + `.webp` | `python examples/demos/newton_headless_viewer.py --size 1280x720 --out newton_headless_viewer.mp4` | Newton example `basic_shapes` and its assets, Newton project, Apache-2.0 | `ViewerRaywarpHeadless`, aa.spp 2; needs pyglet |
| `gallery/usd_stages.mp4` + `.webp` | `python examples/demos/scene_tour.py --scenes materials_hdri,layered_coat,instancing,textures --size 1280x720 --out usd_stages.mp4` | Raywarp fixtures, Apache-2.0 | 2 s per stage, quality recipe |
| `gallery/parallel_worlds.mp4` + `.webp` | `python examples/demos/parallel_worlds.py --worlds 256 --size 1280x720 --out parallel_worlds.mp4` | franka: MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | 16 x 16 tiles of 80x44; each world's camera on its own orbit phase |
| `gallery/domain_randomisation.mp4` + `.webp` | `python examples/demos/domain_randomisation.py --worlds 16 --size 1280x720 --out domain_randomisation.mp4` | franka: MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | re-drawn every 15 frames (0.5 s) |
| `gallery/batch_mask.mp4` + `.webp` | `python examples/demos/batch_mask.py --worlds 16 --size 1280x720 --out batch_mask.mp4` | franka: MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | thirds: all worlds, even worlds, a sweeping band |
| `gallery/robot_multicam.mp4` + `.webp` | `python examples/demos/robot_multicam.py --worlds 4 --size 1280x720 --out robot_multicam.mp4` | franka: MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | 4 worlds x (front, side, top, wrist), one row per world |
| `gallery/material_studio.mp4` + `.webp` | `python examples/demos/material_studio.py --size 1280x720 --out material_studio.mp4` | Raywarp fixture `tests/data/scenes/textures/studio.hdr`, Apache-2.0 | SceneBuilder sphere wall; the HDRI turns once (`env.yaw`) |
| `gallery/daylight_timelapse.mp4` + `.webp` | `python examples/demos/daylight_timelapse.py --size 1280x720 --out daylight_timelapse.mp4` | sponza: Intel New Sponza, Intel Corporation, CC BY 4.0 | 36 `sun_sky` maps, `env.id` per frame, exposure.adaptation continuous |
| `gallery/lights_tour.mp4` + `.webp` | `python examples/demos/scene_tour.py --scenes lights_area,ies_fixtures,many_lights --size 1280x720 --out lights_tour.mp4` | Raywarp fixtures, Apache-2.0 | quality recipe |
| `gallery/before_after_gi.mp4` + `.webp` | `python examples/demos/before_after.py --setting gi --size 1280x360 --out before_after_gi.mp4` | Raywarp fixture `gi_window`, Apache-2.0 | 1280x360 (two 638x360 panels); also writes `before_after_gi_before.mp4` / `_after.mp4` |
| `gallery/sponza_flythrough.mp4` + `.webp` | `python examples/demos/sponza_flythrough.py --size 1280x720 --out sponza_flythrough.mp4` | sponza: Intel New Sponza, Intel Corporation, CC BY 4.0 | quality recipe (aa.spp 2, shadows.samples 2, 16 casters) |
| `gallery/scene_tour.mp4` + `.webp` | `python examples/demos/scene_tour.py --size 1280x720 --out scene_tour.mp4` | Intel New Sponza, Intel Corporation, CC BY 4.0; Raywarp showcase room, fixtures `open_plan`, `cornell_rect`, Apache-2.0 | 2 s per scene |
| `gallery/gltf_asset.mp4` + `.webp` | `python examples/demos/gltf_asset.py --size 1280x720 --out gltf_asset.mp4` | Khronos glTF sample `ToyCar`, Guido Odendahl, CC0 1.0 (downloaded at run time, sha256 pinned) | full turntable turn under studio.hdr, aa.spp 2 |
| `gallery/ground_truth.mp4` + `.webp` | `python examples/demos/ground_truth.py --size 1280x720 --out ground_truth.mp4` | franka: MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | 3 x 2: rgb, depth, seg, camera normals, NOCS, motion |
| `gallery/camera_models.mp4` + `.webp` | `python examples/demos/camera_models.py --size 1280x720 --out camera_models.mp4` | showcase: Raywarp showcase room, Apache-2.0 | 2 x 2: pinhole, ortho, fisheye 200 degrees, OpenCV k1 sweep; lens cameras compile once |
| `gallery/sensor_effects.mp4` + `.webp` | `python examples/demos/sensor_effects.py --worlds 4 --size 1280x720 --out sensor_effects.mp4` | showcase: Raywarp showcase room, Apache-2.0 | 2 x 2 per-world sensor variants; rolling shutter + motion blur from 4 s |
| `gallery/compare.mp4` + `.webp` | `MUJOCO_GL=egl python examples/demos/compare_renderers.py --size 1920x360 --out compare.mp4` | franka: MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | 1920x360 (3 panels); raw clips `compare_raywarp.mp4`, `compare_mujoco_warp.mp4`, `compare_mujoco_opengl.mp4` |
| `gallery/web_viewer.mp4` + `.webp` | manual screen capture of `raywarp view` on the showcase room (no script) | showcase: Raywarp showcase room, Apache-2.0 | 8 s, 1280x720: orbit, a settings toggle, a scene switch |

Previews from the MP4s, one shot each (palette per clip, 640 px, 12 fps, looping):

```bash
ffmpeg -i gallery/NAME.mp4 -vf "fps=12,scale=640:-2:flags=lanczos" -c:v libwebp -lossless 0 -q:v 60 -compression_level 6 -loop 0 -an gallery/NAME.webp
```

(If a preview exceeds ~1 MB, lower `-q:v` to 45 or cut to 6 s with `-t 6`.) Re-encode a recorder MP4 for the
docs with `ffmpeg -i NAME.mp4 -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p -movflags +faststart -an gallery/NAME.mp4`.

### Montages

Edited by the SHOWCASE lane from the demos' raw clips: 8-12 s, 1280x720, H.264 CRF 28 under ~2 MB, preview
`gallery/montage_<name>.webp` as above. Captions and labels go in the page, not in the image, except the short
setting / renderer labels named below.

| file | what it shows | illustrates (page, section) | raw clips (demo commands, GPU) | scene, credits, licence | edit notes |
| --- | --- | --- | --- | --- | --- |
| `gallery/montage_renderers.mp4` | raywarp next to MuJoCo Warp's renderer and MuJoCo OpenGL, one simulation and camera | features.md, Quick start (or performance.md, comparison) | `MUJOCO_GL=egl python examples/demos/compare_renderers.py --size 1920x360 --out compare.mp4` (raw `compare_<renderer>.mp4`, 638x360 each) | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | 3 s each renderer full frame, then a vertical wipe raywarp / MuJoCo Warp, then the 3-panel row; small renderer labels |
| `gallery/montage_batch_zoom.mp4` | 1 world zooming out to 1024 | features.md, One launch, many worlds; performance.md, batching | `python examples/demos/parallel_worlds.py --worlds 1024 --size 2560x1440 --out batch_1024.mp4` (HEVC beyond 4096 px is not needed at 2560) | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | start on one 80x44 tile scaled to full frame, zoom out over 6 s to the whole 32 x 32 mosaic, hold 2 s |
| `gallery/montage_multicam.mp4` | front / side / top / wrist cameras per world | robotics.md, cameras (several cameras per world) | `python examples/demos/robot_multicam.py --worlds 1 --size 1280x720 --out multicam_1.mp4` and `--worlds 4 --out multicam_4.mp4` | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | 2 x 2 of one world for 4 s, cut to 4 x 4 (4 worlds) for 4 s |
| `gallery/montage_ground_truth.mp4` | rgb wiped into each ground-truth output, then the grid | randomization.md, Ground truth; features.md, Outputs | `python examples/demos/ground_truth.py --size 1280x720 --out ground_truth.mp4` | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | crop each panel from the 3 x 2 grid, scale to full frame, horizontal wipe rgb -> depth -> seg -> normals -> NOCS -> motion (1.2 s each), end on the grid |
| `gallery/montage_before_after.mp4` | five settings off / on | settings.md, The groups (aa, gi, lighting, shadows, sensor) | `python examples/demos/before_after.py --setting S --size 1280x720 --out before_after_S.mp4` for S in fxaa, gi, contact_ao, glass, noise (raw `before_after_S_before.mp4` / `_after.mp4`) | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0; Raywarp showcase room and fixtures `gi_window`, `glass`, Apache-2.0 | per setting 2 s: a slow wipe from the before to the after raw clip, label "S off / on"; FXAA on a 2x nearest crop |
| `gallery/montage_randomisation.mp4` | one world, then 16, then 64, re-randomised | domain_randomisation.md, How it works | `python examples/demos/domain_randomisation.py --worlds 1 --out dr_1.mp4`, `--worlds 16 --out dr_16.mp4`, `--worlds 64 --out dr_64.mp4` | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | cuts on the re-randomisation beat (every 15 frames), 3 s per batch size |
| `gallery/montage_batch_mask.mp4` | active worlds vs frozen ones | robotics.md, Selected worlds only | `python examples/demos/batch_mask.py --worlds 64 --size 1280x720 --out batch_mask_64.mp4` | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0 | outline the active tiles (the schedule is in `active_worlds`), frame time from the run printed in a corner |
| `gallery/montage_scene_tours.mp4` | Sponza, the showcase room, fixtures | scenes.md, top; datagen.md, What a switch costs | `python examples/demos/sponza_flythrough.py --out sponza_flythrough.mp4`; `python examples/demos/scene_tour.py --out scene_tour.mp4`; `python examples/demos/scene_tour.py --scenes lights_area,many_lights,glass,materials_hdri --out fixtures_tour.mp4` | Intel New Sponza, Intel Corporation, CC BY 4.0; Raywarp showcase room and fixtures, Apache-2.0 | 4 s Sponza fly-through, then 1.5 s per scene, cut on camera moves |
| `gallery/montage_daylight.mp4` | sunrise to sunset, shadows sweeping | scenes.md, Environment maps (HDRI); domain_randomisation.md, Environment | `python examples/demos/daylight_timelapse.py --out daylight_sponza.mp4`; `python examples/demos/daylight_timelapse.py --scene shadows --out daylight_shadows.mp4` | Intel New Sponza, Intel Corporation, CC BY 4.0; Raywarp fixture `shadows`, Apache-2.0 | Sponza 7 s, the fixture 3 s picture-in-picture or cut |
| `gallery/montage_robotics.mp4` | MuJoCo Warp Panda and Newton USD robots / bodies | robotics.md, top; features.md, Quick start | `python examples/demos/robot_multicam.py --out robot_multicam.mp4`; `python examples/demos/newton_usd_robots.py --out newton_usd_robots.mp4`; `python examples/demos/newton_falling_objects.py --out newton_falling_objects.mp4` | MuJoCo Menagerie `franka_emika_panda`, Apache-2.0; Newton example assets, Apache-2.0 | 3-4 s each, cross-fades, Newton first |
