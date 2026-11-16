# Command line

The Python API is the interface; `raywarp` is a small command over it for what is handy from a
shell. `raywarp COMMAND --help` lists every option.

| command | what |
| --- | --- |
| `raywarp view [SCENE ...] [--scenes-dir DIR]` | the [web viewer](../README.md#web-viewer): `.wpbr`, USD, glTF and MJCF files; prints its URL |
| `raywarp record SCENE OUT.mp4 [--frames 150 --fps 30 --size 1024x576 --worlds 1 --orbit 360 --settings default\|fast]` | an MP4 of a scene, the camera orbiting its opening view (`raywarp.video`, [recording](recording.md)); several worlds: one mosaic |
| `raywarp convert IN.gltf OUT.usdc` | a glTF asset to the USD stage `rw.formats.gltf.load` reads (`.usd` / `.usda` / `.usdc`), for other renderers |
| `raywarp convert IN.glb OUT_DIR/NAME.xml [--scale S --normalize SIZE]` | a glTF asset to a visual-only MJCF model, meshes and textures beside it (`raywarp.integrations.mujoco.from_gltf`) |
| `raywarp prepare IN.usd [-o OUT.wpbr]` | a USD stage baked to a `.wpbr` scene file (`rw.formats.usd.bake`): `rw.Scene.load` maps it in milliseconds |
| `raywarp kernels info [--arch N]` | what serves this machine's kernels: architecture, pinned bundle, local store, remote, driver, the download / compile switches ([install](install.md#prebuilt-kernels)) |
| `raywarp kernels fetch [--arch N --driver D]` | a whole bundle and its OptiX module pack into the local store: an offline mirror |
| `raywarp cache info` | the cache directory (`RAYWARP_CACHE_DIR`) and each cache's size |
| `raywarp cache prune --max-gb N` | deletes the least recently used cache entries until they fit ([performance](performance.md#first-frame-prebuilt-kernels-caches-and-warm)) |
| `raywarp info` | the environment: GPU, compute capability, RT cores, driver, OptiX, Warp, the traversal "auto" picks, the kernels and whether their remote answers |

```bash
raywarp info
raywarp record robot/scene.xml robot.mp4 --worlds 16 --size 512x512
raywarp convert helmet.glb helmet.usdc && raywarp prepare helmet.usdc
raywarp view helmet.wpbr
```

Without the console script (a source checkout): `python -m raywarp.cli COMMAND`. Each command
needs the extras of what it reads (`mujoco` for MJCF, `usd` for USD, `gltf` for glTF).
