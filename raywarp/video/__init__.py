"""Fast MP4 recording of raywarp renders (docs/recording.md; `raywarp record` from the shell).

  from raywarp.video import recorder
  with recorder("run.mp4", (nworld, ncam, h, w), fps=30) as rec:
    for _ in range(n):
      rec.write(renderer.render().rgb)

NVENC (the `video` extra, PyNvVideoCodec, + the ffmpeg CLI) when the GPU has one, else pipelined CPU
encoding (ffmpeg libx264); see `recorder`.
"""

from raywarp.video.colour import Tiling
from raywarp.video.encoders import Encoding
from raywarp.video.recorder import Recorder, RecordStats, recorder

__all__ = ["Encoding", "RecordStats", "Recorder", "Tiling", "recorder"]
