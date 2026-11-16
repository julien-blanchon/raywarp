# Recording video

Save MP4 files of one view or a whole batch without slowing the render loop. From the shell,
`raywarp record SCENE OUT.mp4` records a scene file with an orbiting camera ([CLI](cli.md)).

## Quick start

```bash
pip install "raywarp[video]"   # GPU encoding (PyNvVideoCodec, optional); both paths need the ffmpeg CLI on PATH
```

Record the four Newton worlds of the [quickstart](quickstart.md#quick-start), tiled on one frame:

```python
# docs: skip (needs ffmpeg and the quickstart's Newton scene; tests/unit/test_video.py runs the recorder)
from raywarp.video import recorder

r = rw.Renderer(scene, nworld=model.world_count, cameras=[rw.CameraSpec(fovy=60)], resolution=(512, 512))
r.camera_pose.fill_(eye)
with recorder("ants.mp4", (model.world_count, 1, 512, 512), fps=30, layout="mosaic") as rec:
  for i in range(300):
    # step the simulation here: Newton's solver writes state
    r.render(state)
    rec.write(r.outputs.rgb)  # returns at once; encoding runs beside the next frame
print(rec.stats)  # frames, blocked_s (0: never slowed the loop), backend, reason
```

- `shape` is the `(nworld, ncam, H, W)` of `r.outputs.rgb`.
- Call `write` on the thread that rendered, after the render returns.
- The recorder needs the default `rgb` output (RGBA bytes, `ColorOutput()`).

> **New in 0.4.0:** `raywarp.video` is part of the library (it was a development tool), with
> the `video` extra for GPU encoding.

## Record a batch to MP4

One file per world and camera, or every view tiled on one frame:

```python
# docs: skip (needs a scene and ffmpeg)
r = rw.Renderer(scene, nworld=64, cameras=[rw.CameraSpec(fovy=60)], resolution=(512, 512))
graph = r.capture(sim=state)

# One file per world and camera ({world}, {cam}, {index} = world * ncam + cam):
with recorder("videos/world{world:03d}_cam{cam}.mp4", (64, 1, 512, 512), fps=30) as rec:
  for i in range(150):
    graph.launch(state)
    rec.write(r.outputs.rgb)

# Every view on one frame: 8 x 8 tiles of 512² = 4096², so HEVC (NVENC's H.264 stops at 4096 px):
with recorder("videos/mosaic.mp4", (64, 1, 512, 512), fps=30, layout="mosaic", codec="hevc") as rec:
  for i in range(150):
    graph.launch(state)
    rec.write(r.outputs.rgb)
```

<!-- SHOWCASE: recording/mosaic.webp (docs/assets/SHOTS.md) -->
[![A 64-view mosaic recorded at render speed](assets/recording/mosaic.webp)](assets/recording/mosaic.mp4)

| option | default | effect |
| --- | --- | --- |
| `layout` | `"streams"` | `"streams"`: one file per world and camera; `"mosaic"`: all views tiled on one frame |
| `codec` | `"h264"` | `"h264"` or `"hevc"` (mosaics wider than 4096 px need HEVC on NVENC) |
| `backend` | `"auto"` | `"nvenc"` (GPU, raises without one), `"x264"` (CPU encoder processes) or `"auto"` |
| `bits_per_pixel` | 0.15 | bitrate = bpp x W x H x fps |
| `nvenc_preset` / `x264_preset` | `"P3"` / `"veryfast"` | `"P1"` fastest ... `"P7"`; any x264 preset |
| `gop_seconds` | 2.0 | keyframe interval |
| `depth` | 6 | frames in flight before `write` blocks |
| `workers` | 4 | encoder processes |
| `cols` | 0 (square) | mosaic columns |

`backend="auto"` uses NVENC when the GPU has one and PyNvVideoCodec imports. Otherwise it encodes
on the CPU and says why in `rec.stats.reason`.

## Record a CUDA graph loop

A single 1024² view gives the GPU little work, so an eager loop is limited by the host launching
kernels. Replaying a captured graph is faster: on Franka, one render took 1.51 ms eagerly and
0.61 ms replayed.

```python
# docs: skip (needs a renderer `r`, a MuJoCo Warp model and ffmpeg)
graph = r.capture(lambda: mjw.step(m, d), sim=d)  # step + render as one CUDA graph
with recorder("run.mp4", (1, 1, 1024, 1024), fps=30) as rec:
  for i in range(300):
    graph.launch(d)
    rec.write(r.outputs.rgb)  # outside the graph, on the same stream
```

Keep `write` **outside** the capture. It uses host queues and a rotating ring slot, which a
graph would freeze. Called after `graph.launch` on the same stream, its work queues behind the
replay.

## How the recorder keeps the loop fast

The usual loop reads every frame back and pipes it to ffmpeg:

```python
# docs: skip (the slow way, for contrast)
for i in range(n):
  out = r.render()
  ffmpeg.stdin.write(out.rgb.numpy()[0, 0, ..., :3].tobytes())  # sync + 4 B/px over PCIe + CPU x264
```

Each `.numpy()` waits for the GPU and copies 4 bytes per pixel. The CPU then converts and encodes
on the render thread. On the showcase room (1 world at 1024², RTX PRO 6000) this loop rendered at
296 frames/s but recorded at 64.

`recorder` keeps the render thread free:

1. **On the GPU:** a small kernel converts the `rgb` output to NV12 video format in a ring slot.
   `write` costs the render thread a few tens of microseconds.
2. **In separate encoder processes:** each waits for its slot, encodes it and frees it.
   - **NVENC:** the GPU encoder reads the frame from device memory. Only the compressed stream,
     a few kB per frame, reaches the host.
   - **CPU** (`backend="x264"`): for GPUs without NVENC (A100, H100) and the CPU device. The
     frame is copied to pinned host memory at 1.5 bytes per pixel, and ffmpeg processes encode on
     every core.
3. **Backpressure, not drops:** the render thread blocks only when every slot is still being
   encoded. `rec.stats.blocked_s` says for how long.

## Measured throughput

RTX PRO 6000 Blackwell (4 NVENC engines, 192 vCPUs), showcase room, default settings, graph
replay, the robot posed from NumPy every frame (a host-bound loop), `bits_per_pixel=0.15`, 30 fps:

| case | frames/s | vs render only | file size | quality |
|---|---:|---:|---|---|
| 1 x 1024², render only, eager | 327.9 | | | |
| 1 x 1024², render only, graph replay | 411.7 | | | |
| + naive `.numpy()` + ffmpeg pipe (eager) | 64 | | | |
| + NVENC H.264 P3 | 365.9 | -11.1 % | 6.0 MB (5.1 Mbit/s) | PSNR 42.9 dB |
| + NVENC H.264 P1 | 379.2 | -7.9 % | 6.0 MB | |
| + CPU libx264 veryfast | 327.5 | -20.5 % | 5.7 MB (4.7 Mbit/s) | PSNR 41.4 dB |
| 64 x 512², render only, graph replay | 50.5 (3 232 images/s) | | | |
| + NVENC H.264, 64 files | 41.1 | -18.6 % | 48.6 MB (1.3 Mbit/s each) | |
| + NVENC HEVC mosaic 4096² | 45.0 | -10.9 % | 48.7 MB (82 Mbit/s) | |
| + CPU libx264 mosaic 4096² | 46.9 | -7.1 % | 74.1 MB | |

These rows were measured with an earlier design that ran the encoders in threads of the render
process, where they competed for the GIL. The current design runs them in separate processes;
its frame rates are **not measured yet**.

The encoders alone are faster than the renderer (synthetic frames, same GPU):

| stream | NVENC | CPU libx264 veryfast |
|---|---:|---:|
| 1 x 1024² H.264 | 1 130 frames/s (P3), 1 245 (P1) | 760 frames/s |
| 64 x 512² H.264, 64 files | 211 batches/s = 13 500 frames/s | |
| 64 x 512² HEVC mosaic 4096² | 115 frames/s | |

To measure your own loop, time it with and without the recorder (one warm-up frame,
`wp.synchronize_device()` before reading the clock) and read `rec.stats.blocked_s`.

## Pitfalls

| pitfall | what to do |
| --- | --- |
| a sync in the loop (`.numpy()`, `wp.synchronize()`, `torch.cuda.synchronize()`) | remove it: it serialises the GPU and the host. `write` does none of them. |
| hand-written readback into ordinary memory | copy into `pinned=True` buffers on their own stream: pageable copies are synchronous and half as fast |
| NVENC session limits | each file is one encoder session. GeForce drivers cap concurrent sessions (8 on recent drivers); past the cap, `recorder` raises and suggests a mosaic. |
| no NVENC on A100 / H100 | `auto` encodes on the CPU, limited by cores: record a subset of worlds (`rgb[:k]`) or every k-th step |
| odd resolutions | an odd width or height is padded by one row or column. NVENC's H.264 stops at 4096 px per side; HEVC goes to 8192. |
| `write` inside a captured graph | do not: it would bake one ring slot into the graph. Launch the graph, then `write`. |
| colours look slightly off when decoded | the files are tagged BT.709, limited range. Decode with exact rounding: `ffmpeg ... -sws_flags accurate_rnd+full_chroma_int`. |
| `rec.stats.blocked_s > 0` | the encoder is slower than the renderer: use a faster preset (`nvenc_preset="P1"`, `x264_preset="ultrafast"`), a mosaic, fewer worlds, or more `depth` |

## See also

- [Data generation](datagen.md): saving frames as images instead.
- [Command line](cli.md): `raywarp record`.
- [Robotics recipes: CUDA graphs](robotics.md#step-and-render-in-one-cuda-graph).
