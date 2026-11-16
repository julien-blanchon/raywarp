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
| `readme/hero.webp` | still | README.md, top | franka (benchmarks/run.py scene, MuJoCo Menagerie) + showcase (benchmarks/run.py scene) | quality recipe, per-world material / light / environment randomisation | the scene's front view, one per world | 1600x600 (a 4 x 2 mosaic of 400x300 worlds) |
| `features/batch_mosaic.mp4` + `features/batch_mosaic.webp` | clip | features.md, One launch, many worlds | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default, per-world randomisation of materials and lights | front view per world | 8 s, 1280x720 (8 x 8 tiles of 160x90), mp4 + webp preview |
| `features/outputs.webp` | still | features.md, Outputs | franka (benchmarks/run.py scene, MuJoCo Menagerie) (arm in motion) | default; outputs rgb, depth, seg, normal, motion; r.nocs() | front | 1920x320 (6 tiles of 320x320) |
| `features/randomisation.webp` | still | features.md, Per-world randomisation | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; PerWorld material / light / env, random per world | front | 1920x270 (8 tiles of 240x270) |
| `quickstart/mujoco.webp` | still | quickstart.md, MuJoCo Warp | the snippet's MJCF | the snippet's | front and side | 960x360 (rgb / depth / seg rows, worlds 0-3) |
| `quickstart/usd.webp` | still | quickstart.md, USD stages and glTF | the snippet's demo.usda | default | the stage's camera | 960x540 |
| `quickstart/newton.webp` | still | quickstart.md, Newton | the snippet's Newton model | default + rw.Lighting.studio() | the snippet's pose | 960x540 (2 x 2 worlds) |
| `concepts/worlds.webp` | still | concepts.md, Worlds | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default, per-world materials | front | 960x540 (4 x 4 worlds) |
| `concepts/recipes.webp` | still | concepts.md, Settings (after the recipes table) | showcase (benchmarks/run.py scene) | FAST | default | quality | the scene's view | 1440x270 (3 tiles of 480x270) |
| `scenes/lights.webp` | still | scenes.md, Lights | tests/data/scenes/lights_punctual.usda, lights_area.usda, ies_fixtures.usda | default | View0 | 1920x270 (4 tiles) |
| `scenes/envmaps.webp` | still | scenes.md, Environment maps (HDRI) | tests/data/scenes/environment.usda with sky.hdr / studio.hdr + a Poly Haven CC0 HDRI | default | View0 | 1440x270 (3 tiles) |
| `scenes/exposure.webp` | still | scenes.md, Light units and exposure | showcase (benchmarks/run.py scene) | exposure.mode auto, manual ev100 8, 12, 15 | the scene's view | 1920x270 (4 tiles) |
| `scenes/materials.webp` | still | scenes.md, The material: OpenPBR Surface | tests/data/scenes/materials.usda | quality | View0 | 960x540 |
| `scenes/glass.webp` | still | scenes.md, Materials and textures (after the OpenPBR inputs table) | tests/data/scenes/glass.usda | default | default with shadows.translucent='clear' | View0 | 960x270 (2 tiles) |
| `scenes/camera_models.webp` | still | scenes.md, Cameras | showcase (benchmarks/run.py scene) | default | one mount, four CameraSpec projections | 1920x270 (4 tiles of 480x270) |
| `scenes/lens_distortion.mp4` + `scenes/lens_distortion.webp` | clip | scenes.md, Cameras (after the lens bullets) | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default, PerWorld(camera=['distortion']) | front, k1 swept over the clip | 6 s, 1280x720, mp4 + webp preview |
| `scenes/segmentation.webp` | still | scenes.md, Segmentation | franka (benchmarks/run.py scene, MuJoCo Menagerie) | outputs rgb, seg, semantic (set_semantics) | front | 1440x270 (3 tiles) |
| `settings/recipes.webp` | still | settings.md, Recipes | showcase (benchmarks/run.py scene) | the four recipes | the scene's view | 1920x270 (4 tiles of 480x270) |
| `settings/aa.webp` | still | settings.md, The groups: `aa` | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; aa.spp 1 | aa.post True | aa.spp 2 | front, crop on the gripper edges, nearest upscale x2 | 1440x360 (3 tiles) |
| `settings/shadows.webp` | still | settings.md, The groups: `shadows` | showcase (benchmarks/run.py scene) | FAST | default | shadows.samples 8 | the scene's view | 1440x270 (3 tiles) |
| `settings/contact_ao.webp` | still | settings.md, The groups: `lighting` | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; lighting.contact_ao 0 | 1 | close-up of the cubes on the floor | 960x270 (2 tiles) |
| `settings/reflections.webp` | still | settings.md, The groups: `reflections` | showcase (benchmarks/run.py scene) | default; reflections.enabled False | True | the scene's view | 960x270 (2 tiles) |
| `settings/gi.webp` | still | settings.md, The groups: `gi` | tests/data/scenes/gi_window.usda and showcase (benchmarks/run.py scene) | default; gi.enabled False | True | View0 / the scene's view | 960x540 (2 x 2 tiles) |
| `settings/tone_mapper.webp` | still | settings.md, The groups: `post` | franka (benchmarks/run.py scene, MuJoCo Menagerie) under an HDRI | default with each tone mapper | front | 1440x270 (3 tiles) |
| `settings/fog.webp` | still | settings.md, The groups: `atmosphere` | sponza (benchmarks/run.py scene, Intel CC-BY 4.0) | default; atmosphere end 30 | haze 0.05 | courtyard view | 1440x270 (3 tiles) |
| `robotics/outputs.webp` | still | robotics.md, Outputs: what you can request and what it costs | franka (benchmarks/run.py scene, MuJoCo Menagerie) (arm in motion) | default; rgb, depth, seg, semantic, normal, albedo, roughness, metallic, direct, indirect, motion, shadow | front | 1920x540 (6 x 2 tiles of 320x270) |
| `robotics/point_cloud.webp` | still | robotics.md, Point clouds and per-world labels | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; r.point_cloud over front + side, 16k points | a third view of the points (matplotlib 3D scatter) | 960x540 |
| `robotics/rolling_shutter.mp4` + `robotics/rolling_shutter.webp` | clip | robotics.md, Sensor effects (after the bullets) | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; sensor.rolling_shutter 1 | sensor.motion_blur 0.5 | both | front, orbiting fast | 6 s, 1280x360 (3 tiles), mp4 + webp preview |
| `robotics/sensor_noise.webp` | still | robotics.md, Sensor effects (after the bullets) | showcase (benchmarks/run.py scene) | sensor off | sensor.noise | vignette + response | sensor.depth stereo (depth tile) | the scene's view | 1920x270 (4 tiles) |
| `robotics/batch_mask.mp4` + `robotics/batch_mask.webp` | clip | robotics.md, Selected worlds only | franka (benchmarks/run.py scene, MuJoCo Menagerie) (simulated) | FAST; active = even worlds after 2 s | front per world | 8 s, 1280x720 (4 x 4 worlds), mp4 + webp preview |
| `domain_randomisation/grid.mp4` + `domain_randomisation/grid.webp` | clip | domain_randomisation.md, How it works | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; PerWorld material / light / env / camera fovy | front per world | 8 s, 1280x720 (4 x 4 worlds), mp4 + webp preview |
| `domain_randomisation/materials.webp` | still | domain_randomisation.md, Materials and textures | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default | front | 1920x270 (8 worlds) |
| `domain_randomisation/lights.webp` | still | domain_randomisation.md, Lights | showcase (benchmarks/run.py scene) | default | the scene's view | 1920x270 (8 worlds) |
| `domain_randomisation/environment.webp` | still | domain_randomisation.md, Environment, background, atmosphere | franka (benchmarks/run.py scene, MuJoCo Menagerie) + Poly Haven CC0 HDRIs | default | front | 1920x270 (8 worlds) |
| `domain_randomisation/camera.webp` | still | domain_randomisation.md, Camera and sensor | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default | front, per-world offsets | 1920x270 (8 worlds) |
| `randomization/ground_truth.webp` | still | randomization.md, Ground truth | franka (benchmarks/run.py scene, MuJoCo Menagerie) | default; r.nocs(), r.camera_normals(), MotionOutput(status=True), r.camera_info() | front | 1600x270 (5 tiles) |
| `recording/mosaic.mp4` + `recording/mosaic.webp` | clip | recording.md, Batches: many worlds and cameras | franka (benchmarks/run.py scene, MuJoCo Menagerie) | FAST; Recorder mosaic 8 x 8 | front per world | 10 s, 1280x720, mp4 + webp preview |
| `datagen/scene_switch.mp4` + `datagen/scene_switch.webp` | clip | datagen.md, What a switch costs | franka (benchmarks/run.py scene, MuJoCo Menagerie), showcase (benchmarks/run.py scene), sponza (benchmarks/run.py scene, Intel CC-BY 4.0), tests/data/scenes/* | FAST, 16 worlds, 1 s per scene | each scene's view | 10 s, 1280x720 (4 x 4 worlds), mp4 + webp preview |
| `performance/traversal.webp` | still | performance.md, OptiX | showcase (benchmarks/run.py scene) | default; performance.traversal 'optix' | 'software' | abs difference x10 | the scene's view | 1440x270 (3 tiles) |
| `faq/dark_interior.webp` | still | faq.md, Dark interiors | tests/data/scenes/gi_window.usda | gi off | lighting.fill 0.05 | gi on | View0 | 1440x270 (3 tiles) |
| `cli/viewer.webp` | still | cli.md, Web viewer | showcase (benchmarks/run.py scene) | the viewer's default | the viewer page (screenshot) | 1280x720 |
