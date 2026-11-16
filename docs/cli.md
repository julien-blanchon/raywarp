# Command line

The Python API is the interface; `raywarp` is a small command over it for what is handy from a
shell. `raywarp COMMAND --help` lists every option.

| command | what |
| --- | --- |
| `raywarp view [SCENE ...] [--scenes-dir DIR]` | the [web viewer](#web-viewer): `.rayw`, USD, glTF and MJCF files; prints its URL |
| `raywarp record SCENE OUT.mp4 [--frames 150 --fps 30 --size 1024x576 --worlds 1 --orbit 360 --settings default\|fast]` | an MP4 of a scene, the camera orbiting its opening view (`raywarp.video`, [recording](recording.md)); several worlds: one mosaic |
| `raywarp convert IN.gltf OUT.usdc` | a glTF asset to the USD stage `rw.formats.gltf.load` reads (`.usd` / `.usda` / `.usdc`), for other renderers |
| `raywarp convert IN.glb OUT_DIR/NAME.xml [--scale S --normalize SIZE]` | a glTF asset to a visual-only MJCF model, meshes and textures beside it (`raywarp.integrations.mujoco.from_gltf`) |
| `raywarp prepare IN.usd [-o OUT.rayw]` | a USD stage baked to a `.rayw` scene file (`rw.formats.usd.bake`): `rw.Scene.load` maps it in milliseconds |
| `raywarp kernels info [--arch N]` | what serves this machine's kernels: architecture, pinned bundle, local store, remote, driver, the download / compile switches ([install](install.md#prebuilt-kernels)) |
| `raywarp kernels fetch [--arch N --driver D]` | a whole bundle and its OptiX module pack into the local store: an offline mirror |
| `raywarp cache info` | the cache directory (`RAYWARP_CACHE_DIR`) and each cache's size |
| `raywarp cache prune --max-gb N` | deletes the least recently used cache entries until they fit ([performance](performance.md#first-frame-prebuilt-kernels-caches-and-warm)) |
| `raywarp info` | the environment: GPU, compute capability, RT cores, driver, OptiX, Warp, the traversal "auto" picks, the kernels and whether their remote answers |

```bash
raywarp info
raywarp record robot/scene.xml robot.mp4 --worlds 16 --size 512x512
raywarp convert helmet.glb helmet.usdc && raywarp prepare helmet.usdc
raywarp view helmet.rayw
```

Without the console script (a source checkout): `python -m raywarp.cli COMMAND`. Each command
needs the extras of what it reads (`mujoco` for MJCF, `usd` for USD, `gltf` for glTF).

## Web viewer

`raywarp view` serves an interactive viewer from the GPU
machine: pick a scene, change any setting, fly through it, and see every frame's time and whether
anything compiled. Install the `viewer` extra (Pillow; the server is the standard library's and
the page has no build step) plus the extras of your scene formats; the prebuilt kernels for your
GPU download on first use.

<!-- SHOWCASE: cli/viewer.webp (docs/assets/SHOTS.md) -->
![raywarp view in a browser](assets/cli/viewer.webp)

```bash
raywarp view office.usd robot/scene.xml helmet.glb --scenes-dir ~/scenes   # any .rayw / USD / glTF / MJCF
# raywarp viewer: http://127.0.0.1:8080/?token=...
#   device cuda:0 (NVIDIA RTX PRO 6000 ...); prebuilt kernels: sm120, Warp 1.18.0, content ...
```

The server binds the loopback interface and prints a URL with a session token. From a laptop,
forward the port over SSH and open that URL locally:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-host     # then open http://127.0.0.1:8080/?token=... on the laptop
```

- **Scenes:** the built-in demo, the files and directories of the command line, or any path on
  the GPU machine typed in the page. A scene that fails to load is reported; the previous one
  stays.
- **Settings:** every `RenderSettings` field, grouped as in `rw.RenderSettings`, starting from the
  default or `rw.FAST`, with each field's kind (R run time, B run time with a rebake, K another
  prebuilt kernel build) and the fields changed from that base highlighted. Changes apply between two
  frames on the live renderer; a refused value is reported and the renderer keeps its settings.
  The view menu picks the image shown (rgb, depth, normal, albedo, ...; on OptiX the outputs switch at run
  time: [settings](settings.md#compiled-in-or-runtime)).
- **Camera:** drag to look, W A S D to move, Q / E down / up, Shift faster, the wheel sets the
  speed, H returns to the scene's view. FOV and image size in the side panel.
- **Compiles, live:** the status line counts kernel compiles, OptiX PTX compiles and OptiX module
  creations since the start, and names the kernel tier serving the settings. With the prebuilt
  bundle, kernel and PTX compiles stay at 0; any compile is listed and flagged in red, and
  settings outside the prebuilt tiers are named before a frame renders. An OptiX module created
  on first use (no module pack for this driver) is a stall of seconds, shown in amber. While the
  renderer is blocked (a scene load, a kernel compile, an OptiX module creation), the page shows
  what it waits for, for how long, and the estimate.
- **Background warm:** after a scene opens (and after a change of a kernel-build field), the
  viewer creates in the background the OptiX modules of every output view and of the settings one
  change away (`Renderer.warm(..., nearby=..., nearby_outputs=...)`), so the next click finds them
  ready; the frames keep streaming meanwhile and the status line says when it is done.

Flags: `--device cuda:1`, `--port 0` (any free port), `--size 1280x720`, `--base fast`,
`--warm N` (modules created ahead at most; 0: none), `--host 0.0.0.0` (other machines; prefer the
SSH forward), `--no-token`. The frames are JPEGs pushed over one WebSocket on the same port: the
frame rate follows the renderer and the link's bandwidth, not its round trip, and a credit window
of unacknowledged frames (grown while the link shows no queue, shrunk when it does) keeps the
newest camera at most about one round trip away.
