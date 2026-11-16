# Command line

The `raywarp` command does from a shell what is handy there: view, record, convert and prepare
scenes, and check the machine. The Python API is the main interface.

> **New in 0.4.0:** the `raywarp` command. It replaces `warp-pbr-viewer`; `raywarp view` takes
> the same options.

## Quick start

```bash
raywarp info                                                    # the GPU, driver, OptiX, kernels
raywarp record robot/scene.xml robot.mp4 --worlds 16 --size 512x512
raywarp convert helmet.glb helmet.usdc && raywarp prepare helmet.usdc
raywarp view helmet.rayw                                        # prints the viewer's URL
```

`raywarp COMMAND --help` lists every option. In a source checkout without the console script,
run `python -m raywarp.cli COMMAND`. Each command needs the extras of what it reads (`mujoco` for
MJCF, `usd` for USD, `gltf` for glTF; [install](install.md)).

## Commands

| command | what |
| --- | --- |
| `raywarp view [SCENE ...] [--scenes-dir DIR]` | the [web viewer](#web-viewer) for `.rayw`, USD, glTF and MJCF files |
| `raywarp record SCENE OUT.mp4 [--frames 150 --fps 30 --size 1024x576 --worlds 1 --orbit 360 --settings default\|fast]` | an MP4 with the camera orbiting the scene's opening view; several worlds make a mosaic ([recording](recording.md)) |
| `raywarp convert IN.gltf OUT.usdc` | a glTF asset to a USD stage (`.usd` / `.usda` / `.usdc`), the one `rw.formats.gltf.load` reads |
| `raywarp convert IN.glb OUT_DIR/NAME.xml [--scale S --normalize SIZE]` | a glTF asset to a visual-only MJCF model, meshes and textures beside it |
| `raywarp prepare IN.usd [-o OUT.rayw]` | a USD stage baked to a `.rayw` scene file, which `rw.Scene.load` maps in milliseconds |
| `raywarp kernels info [--arch N]` | what serves this machine's kernels: architecture, bundle, local store, remote, driver, switches ([install](install.md#prebuilt-kernels)) |
| `raywarp kernels fetch [--arch N --driver D]` | a whole kernel bundle and its OptiX modules into the local store, for an offline machine |
| `raywarp cache info` | the cache directory (`RAYWARP_CACHE_DIR`) and each cache's size |
| `raywarp cache prune --max-gb N` | deletes the least recently used cache entries until they fit ([caches](performance.md#first-frame-prebuilt-kernels-caches-and-warm)) |
| `raywarp info` | the GPU, compute capability, RT cores, driver, OptiX, Warp, the traversal `"auto"` picks, the kernels and whether their download answers |

## Web viewer

`raywarp view` serves an interactive viewer from the GPU machine. Pick a scene, change any
setting, fly through it, and see each frame's time and whether anything compiled.

<!-- SHOWCASE: cli/viewer.webp (docs/assets/SHOTS.md) -->
![raywarp view in a browser](assets/cli/viewer.webp)

```bash
pip install "raywarp[viewer,usd,gltf,mujoco]"
raywarp view office.usd robot/scene.xml helmet.glb --scenes-dir ~/scenes   # any .rayw / USD / glTF / MJCF
# raywarp viewer: http://127.0.0.1:8080/?token=...
#   device cuda:0 (NVIDIA RTX PRO 6000 ...); prebuilt kernels: sm120, Warp 1.18.0, content ...
```

The server listens on the loopback interface and prints a URL with a session token. From a
laptop, forward the port over SSH and open that URL locally:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-host     # then open http://127.0.0.1:8080/?token=... on the laptop
```

What the page does:

- **Scenes.** The built-in demo, the files and directories on the command line, or any path on the
  GPU machine typed in the page. A scene that fails to load is reported; the previous one stays.
- **Settings.** Every `RenderSettings` field, grouped as in `rw.RenderSettings`, from the default
  or `rw.FAST`. Changed fields are highlighted. Each field shows its kind: R (run time), B (run
  time, re-bakes) or K (another kernel build). Changes apply between two frames. A refused value
  is reported and the renderer keeps its settings.
- **Views.** The view menu picks the image shown: rgb, depth, normal, albedo, ...

  > **Requires an RT-core GPU** (OptiX) to switch views without a compile. On the software path
  > a new view can compile its kernel once ([settings](settings.md#compiled-in-or-runtime)).

- **Camera.** Drag to look. W A S D move, Q / E go down / up, Shift is faster, the wheel sets the
  speed, H returns to the scene's view. FOV and image size are in the side panel.
- **Compiles, live.** The status line counts kernel compiles, OptiX program compiles and OptiX
  module creations. With the prebuilt kernels the compiles stay at 0. Any compile shows in red.
  Settings outside the prebuilt builds are named before a frame renders. An OptiX module created
  on first use (seconds) shows in amber. While the renderer is blocked, the page shows what it
  waits for and for how long.
- **Background warm.** After a scene opens, the viewer prepares in the background the OptiX
  modules of every view and of the settings one change away (`Renderer.warm(..., nearby=...)`).
  The next click finds them ready. Frames keep streaming meanwhile.

| flag | effect |
| --- | --- |
| `--device cuda:1` | the GPU |
| `--port 0` | any free port (default 8080) |
| `--size 1280x720` | the image size |
| `--base fast` | start from `rw.FAST` |
| `--warm N` | modules prepared ahead, at most N (0: none) |
| `--host 0.0.0.0` | serve other machines (prefer the SSH forward) |
| `--no-token` | no session token |

Frames are JPEGs pushed over one WebSocket on the same port. The frame rate follows the renderer
and the link's bandwidth, not its round trip. The newest camera stays at most about one round
trip away.

## See also

- [Install](install.md): extras and prebuilt kernels
- [Recording video](recording.md): `raywarp record` and recording from Python
- [Performance and kernels](performance.md): caches and first frames
- [Scenes](scenes.md): what each format brings
