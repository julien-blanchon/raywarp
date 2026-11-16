# Recording video

raywarp renders into device arrays; `raywarp.video` saves MP4 files of any batch **without
slowing the render loop** (`from raywarp.video import recorder`), and `raywarp record SCENE
OUT.mp4` records a scene file from the shell, the camera orbiting ([CLI](cli.md)).

```bash
pip install "raywarp[video]"   # NVENC from device memory (PyNvVideoCodec, optional); the ffmpeg CLI must be on PATH for both paths
```

## Why a naive loop is slow

The usual loop reads every frame back and pipes RGB to ffmpeg:

```python
# docs: skip (the slow way, for contrast)
for i in range(n):
  out = r.render()
  ffmpeg.stdin.write(out.rgb.numpy()[0, 0, ..., :3].tobytes())  # sync + 4 B/px over PCIe + CPU x264
```

Each `.numpy()` waits for the GPU, copies 4 bytes per pixel into pageable memory, and then the CPU
converts RGB to YUV and encodes on the render thread. On the showcase room (1 world at 1024^2,
the default settings, RTX PRO 6000) this loop rendered at 296 frames/s but recorded at only 64 frames/s:
recording cost 4.6 times as much as rendering.

## The recipe

`recorder` keeps the render thread free:

1. **On the GPU, on the render stream:** a small Warp kernel converts the `rgb` output (sRGB
   RGBA8) to NV12 (BT.709, limited range) in a ring slot, then records an event. The launch is
   recorded once and replayed, so `write` costs the render thread a few tens of microseconds.
   The render loop never waits for the kernel.
2. **In separate encoder processes:** each process maps the ring and the events through CUDA
   IPC, waits for the slot's event, encodes, and reports the slot free. Processes, not threads:
   in-process encoder threads took the render process's GIL. That slowed a host-bound loop (Python
   posing per frame) by 8-19 % on an RTX PRO 6000, worst with 64 NVENC sessions. Each process
   owns a fixed subset of the files:
   - **NVENC** (`backend="nvenc"`): PyNvVideoCodec reads the NV12 frame straight from device
     memory. No raw frame crosses PCIe; only the compressed bitstream, a few kB per frame,
     reaches the host. An `ffmpeg -c:v copy` process per file muxes it into MP4 and writes the
     colour tags.
   - **CPU** (`backend="x264"`): for GPUs without NVENC (A100, H100) or the CPU device. A copy
     stream reads the NV12 slot (1.5 bytes per pixel instead of 4) into **pinned** host buffers.
     A pool of ffmpeg libx264 / libx265 processes, one per file, encodes on every core.
3. **Backpressure, not drops:** with `depth` slots in flight (6 by default), the render thread
   blocks only when every slot is still being encoded. `RecordStats.blocked_s` reports how long.

`backend="auto"` (the default) picks NVENC when the device has one and PyNvVideoCodec imports.
Otherwise it uses the CPU path and says why in `RecordStats.reason`. `backend="nvenc"` raises
instead of switching.

## Minimal: one world, one camera

```python
# docs: skip (needs a renderer `r` and ffmpeg; tests/unit/test_video.py runs the recorder)
from raywarp.video import recorder

r = rw.Renderer(scene, nworld=1, cameras=["front"], resolution=(1024, 1024))
graph = r.capture(lambda: mjw.step(m, d), sim=d)  # step + render as one CUDA graph
with recorder("run.mp4", (1, 1, 1024, 1024), fps=30) as rec:
  for i in range(300):
    graph.launch(d)
    rec.write(r.outputs.rgb)  # returns at once; encoding runs beside the next frame
print(rec.stats)  # frames, blocked_s (0: never slowed the loop), backend, reason
```

The `shape` argument is the `(nworld, ncam, H, W)` of `out.rgb`. Call `write` on the thread and
stream that rendered, after the render (or the graph launch) returns. The recorder needs the
default `rgb` output (vec4ub RGBA, `ColorOutput()`).

**Replay a CUDA graph for the render step.** A single 1024^2 view gives the GPU little work per
kernel. Eagerly, the frame time is then set by the host launching kernels one by one. On Franka,
an eager render takes 1.51 ms; the same render replayed from a capture takes 0.61 ms. In the
showcase room loop below, replay lifts 1 x 1024^2 from 328 to 412 frames/s. Recording
cannot speed up a launch-bound loop, but it can hide behind a fast one. Keep `write` **outside**
the capture: it uses host queues and a rotating ring slot, which a graph would freeze. Called
after `graph.launch` on the same stream, its conversion kernel queues behind the replay and its
event marks the frame ready, so the capture stays valid. Host-side work that cannot be captured,
such as posing from NumPy, runs between launches as usual.

## Batches: many worlds and cameras

```python
# docs: skip (needs a scene and ffmpeg)
r = rw.Renderer(scene, nworld=64, cameras=["front"], resolution=(512, 512))

graph = r.capture()

# One file per world and camera ({world}, {cam}, {index} = world * ncam + cam):
with recorder("videos/world{world:03d}_cam{cam}.mp4", (64, 1, 512, 512), fps=30) as rec:
  for i in range(150):
    graph.launch()
    rec.write(r.outputs.rgb)

# Or every view tiled on one frame (8 x 8 tiles of 512^2 = 4096^2: hevc, since NVENC's H.264 stops at 4096):
with recorder("videos/mosaic.mp4", (64, 1, 512, 512), fps=30, layout="mosaic", codec="hevc") as rec:
  for i in range(150):
    graph.launch()
    rec.write(r.outputs.rgb)
```

<!-- SHOWCASE: recording/mosaic.webp (docs/assets/SHOTS.md) -->
[![A 64-view mosaic recorded at render speed](assets/recording/mosaic.webp)](assets/recording/mosaic.mp4)

Options: `codec` (`"h264"`, `"hevc"`), `bits_per_pixel` (0.15: bitrate = bpp x W x H x fps),
`nvenc_preset` (`"P1"` fastest ... `"P7"`), `x264_preset` (`"veryfast"`), `gop_seconds`,
`depth`, `workers` (encoder processes, 4), `cols` (mosaic columns).

## Measured throughput

RTX PRO 6000 Blackwell (driver 580.178, 4 NVENC engines, 192 vCPUs), showcase room, the default settings. The
render step is replayed from a CUDA graph and the robot is posed from NumPy every frame (a
host-bound loop). Render-only runs came before and after the recorded ones, 300 frames (150 at
64 worlds), `bits_per_pixel=0.15`, 30 fps, files of 10 s (5 s at 64 worlds). Run
`2026-11-10-w9video-rtx`, commit c3b8da176.

| case | frames/s | vs render only | file size | quality |
|---|---:|---:|---|---|
| 1 x 1024^2, render only, eager | 327.9 | | | |
| 1 x 1024^2, render only, graph replay | 411.7 | | | |
| + naive `.numpy()` + ffmpeg pipe (eager) | 64 | | | |
| + NVENC H.264 P3 | 365.9 | -11.1 % | 6.0 MB (5.1 Mbit/s) | PSNR 42.9 dB, bias <= 0.31 / 255 |
| + NVENC H.264 P1 | 379.2 | -7.9 % | 6.0 MB | |
| + CPU libx264 veryfast | 327.5 | -20.5 % | 5.7 MB (4.7 Mbit/s) | PSNR 41.4 dB, bias <= 0.46 / 255 |
| 64 x 512^2, render only, graph replay | 50.5 (3 232 images/s) | | | |
| + NVENC H.264, 64 files | 41.1 | -18.6 % | 48.6 MB (1.3 Mbit/s each) | |
| + NVENC HEVC mosaic 4096^2 | 45.0 | -10.9 % | 48.7 MB (82 Mbit/s) | |
| + CPU libx264 mosaic 4096^2 | 46.9 | -7.1 % | 74.1 MB | |

These rows were measured with the encoders in **threads** of the render process, the first
design. They miss the 5 % target because NVENC calls in those threads took the GIL from a
host-bound render loop. The worse the loop is host-bound and the more sessions there are, the
larger the loss. Even so, recording at 1024^2 ran at 366-379 frames/s against the naive loop's
64. The recipe now runs the encoders in separate processes that share the ring through CUDA
IPC, so the render process only replays one launch and writes one pipe message per frame. That
design passes the NVENC round trip on the RTX PRO 6000 (`tests/tooling/test_video.py`), but its
frame rates are **not measured yet**.

Encoder ceilings (synthetic frames, no renderer; RTX PRO 6000):

| stream | NVENC | CPU libx264 veryfast |
|---|---:|---:|
| 1 x 1024^2 H.264 | 1 130 frames/s (P3), 1 245 (P1) | 760 frames/s |
| 64 x 512^2 H.264, 64 files | 211 batches/s = 13 500 frames/s | |
| 64 x 512^2 HEVC mosaic 4096^2 | 115 frames/s | |

The encoders are faster than the renderer in both cases (411 and 50 frames/s). Recording
overhead therefore comes from the host side (GIL, launches, pipes), not from encoder
throughput. One libx264 process at 1024^2 is the exception: it is close to its limit at 410
frames/s (render-thread blocked 0.5 ms per frame).

## Pitfalls

- **Readback sync.** `.numpy()`, `wp.synchronize()` or `torch.cuda.synchronize()` in the loop
  serialises the GPU and the host. `write` does none of them: it only launches a kernel and
  records an event.
- **Pageable memory.** A device-to-host copy into ordinary (pageable) memory is synchronous and
  runs at about half the PCIe rate. The CPU path copies into `pinned=True` buffers on its own
  stream. Any hand-written readback should do the same.
- **NVENC session limits.** Each output file is one encoder session. GeForce drivers cap
  concurrent sessions per system (8 on recent drivers); RTX PRO / Quadro / data-centre boards with
  NVENC have no cap. Past the cap `recorder` raises with the hint to record a mosaic. An
  RTX PRO 6000 Blackwell has 4 NVENC engines, an L4 2; many sessions share them.
- **No NVENC on A100 / H100** (nor GA100 / GH100 / GB100 derivatives): `auto` uses the CPU path,
  which is limited by cores. It can encode 64 streams of 512^2 only at a few hundred frames/s in
  total, so record a subset of worlds (`rgb[:k]`) or every k-th step.
- **Odd resolutions.** 4:2:0 video needs even sizes: an odd width or height is padded by one
  repeated row / column (the file is one pixel larger). NVENC's H.264 encoder stops at 4096 px
  per side; HEVC goes to 8192.
- **CUDA graphs.** Capture the render step and do not capture `write`: it uses host queues and
  a ring index. Launch the graph, then call `write(r.outputs.rgb)` on the same stream; the
  conversion kernel queues behind the replay. `rec.write` inside `capture(after=...)` would bake
  one ring slot into the graph and skip the hand-off to the encoder processes.
- **Colour.** raywarp's `rgb` is gamma-encoded sRGB, so it is matrixed to Y'CbCr directly, with
  no transfer function applied. The files are tagged BT.709 primaries, transfer and matrix,
  limited range, centre chroma siting, so players convert back exactly. To decode for analysis,
  use exact rounding (`ffmpeg ... -sws_flags accurate_rnd+full_chroma_int`). ffmpeg's default
  scaler truncates, about -1 per channel.
- **Encoder slower than the renderer.** `RecordStats.blocked_s > 0` means the render loop
  waited. Use a faster preset (`nvenc_preset="P1"`, `x264_preset="ultrafast"`), a mosaic (one
  session instead of N), fewer recorded worlds, or more `depth` (absorbs bursts only).

## Measuring it yourself

Time the same loop with and without the recorder (one warm-up frame, `wp.synchronize_device()`
before reading the clock) and read `RecordStats.blocked_s`: the time the render loop waited for
the encoders.
