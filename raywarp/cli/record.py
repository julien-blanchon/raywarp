"""`raywarp record SCENE OUT.mp4`: an MP4 of a scene file, the camera orbiting its opening view
(`raywarp.video.recorder`; several worlds: one mosaic file).

The scene opens as in the viewer (`raywarp.viewer.scenes`: its first authored camera, else a
view framing it); each frame turns the camera `--orbit` / `--frames` degrees around the view's
target, about the vertical axis. NVENC when the GPU has one and PyNvVideoCodec is installed (the
`video` extra), else CPU encoding (ffmpeg libx264); the ffmpeg CLI must be on PATH.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from raywarp.cli.common import SETTINGS, Sub, settings, size

if TYPE_CHECKING:
  from raywarp.video.recorder import Rgb


def register(sub: Sub) -> None:
  """Adds `record`."""
  p = sub.add_parser("record", help="record an MP4 of a scene file, the camera orbiting", description=__doc__)
  p.add_argument("scene", type=Path, help="a scene file: .rayw, USD, glTF or MJCF")
  p.add_argument("out", type=Path, help="the MP4 file")
  p.add_argument("--frames", type=int, default=150, help="frames (default: 150)")
  p.add_argument("--fps", type=int, default=30, help="frame rate (default: 30)")
  p.add_argument("--size", type=size, default=(1024, 576), help="WIDTHxHEIGHT (default: 1024x576)")
  p.add_argument("--worlds", type=int, default=1, help="worlds rendered per frame, tiled in one file (default: 1)")
  p.add_argument("--orbit", type=float, default=360.0, help="degrees the camera turns over the clip (default: 360)")
  p.add_argument("--settings", choices=SETTINGS, default="default", help="default, or fast (rw.FAST)")
  p.add_argument("--backend", choices=("auto", "nvenc", "x264"), default="auto", help="the encoder (default: auto)")
  p.add_argument("--device", default="cuda", help="Warp device (default: cuda)")
  p.set_defaults(run=run)


def orbit(view: tuple[tuple[float, float, float], tuple[float, float, float], float], degrees: float) -> np.ndarray:
  """The (7,) camera pose of `view` (eye, target, fovy) with its eye turned `degrees` about the
  vertical axis through the target."""
  from raywarp.viewer.camera import as_view, view_camera  # noqa: PLC0415

  eye, target = np.asarray(view[0], float), np.asarray(view[1], float)
  a = math.radians(degrees)
  rel = eye - target
  turned = np.array([rel[0] * math.cos(a) - rel[1] * math.sin(a), rel[0] * math.sin(a) + rel[1] * math.cos(a), rel[2]])
  return view_camera(as_view(target + turned, target, view[2])).pose()


def run(args: argparse.Namespace) -> int:
  """Records the clip; prints the recorder's stats."""
  import warp as wp  # noqa: PLC0415

  import raywarp as rw  # noqa: PLC0415
  from raywarp.video import recorder  # noqa: PLC0415
  from raywarp.viewer.scenes import file_entry, load  # noqa: PLC0415

  if args.frames < 1 or args.worlds < 1:
    raise SystemExit("raywarp record: --frames and --worlds must be positive")
  width, height = args.size
  with wp.ScopedDevice(args.device):
    loaded = load(file_entry(args.scene), args.device)
    r = rw.Renderer(
      loaded.gpu,
      nworld=args.worlds,
      cameras=[rw.CameraSpec(fovy=loaded.view[2])],
      resolution=(width, height),
      settings=settings(args.settings),
    )
    rgb = r.outputs.rgb
    if rgb is None or rgb.dtype is not wp.vec4ub:
      raise SystemExit("raywarp record: the renderer's rgb output is not the default RGBA8 (ColorOutput())")
    layout = "mosaic" if args.worlds > 1 else "streams"
    with recorder(args.out, (args.worlds, 1, height, width), fps=args.fps, layout=layout, backend=args.backend) as rec:
      for i in range(args.frames):
        pose = orbit(loaded.view, args.orbit * i / args.frames)
        r.camera_pose.assign(np.tile(pose, (args.worlds, 1, 1)))
        r.render(loaded.sim)
        rec.write(cast("Rgb", rgb))
    r.close()
  print(f"{args.out}: {rec.stats}")
  return 0
