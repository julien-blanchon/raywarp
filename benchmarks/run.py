"""raywarp's public benchmark: one scene, N worlds, one camera; prints ms/frame, images/s, the
first-frame time and the compile count, and saves one PNG.

  python -m benchmarks.run --scene franka|showcase|sponza [--worlds N] [--res R] [--settings
  fast|default]

Scenes (downloaded on first use into the cache directory, `<cache>/benchmarks/<key>/`, from the
public assets bucket `blanchon/raywarp-assets`, `benchmarks/v<format>-c<converter>/`, every file
checked against its sha256; README.md: sources and licences):
  franka    MuJoCo Menagerie's MJX Panda pick scene with three cubes (MJCF: the `mujoco`
            extra; Apache-2.0)
  showcase  raywarp's showcase room: a furnished room, a window, a robot (`.rayw`)
  sponza    Intel's New Sponza, the full scene (`.rayw`; CC-BY 4.0, Intel Corporation)

The scene files are converted ahead per scene-file format and USD converter version (`KEY`); the
release pipeline publishes them (`tools/release/benchmark_scenes.py`). The first frame includes
the kernel load (the prebuilt kernels download on a machine's first run: run twice for the warm
number); "compiles" counts Warp modules, OptiX PTX compiled here and OptiX modules created (0 with
the prebuilt kernels and the driver's module pack).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import struct
import tarfile
import time
import zlib
from pathlib import Path
from typing import Final

import numpy as np

# The scene files on the remote, by scene name.
FILES: Final = {"franka": "franka.tar", "showcase": "showcase.rayw", "sponza": "sponza.rayw"}
# The MJCF model inside franka.tar.
FRANKA_XML: Final = "franka/scene.xml"
# Each scene's benchmark camera: {scene: [eye, target, fovy]} (world units, degrees).
CAMERAS: Final = "cameras.json"
# The public bucket of the scenes (tools/prebuilt/buckets.env's ASSETS_BUCKET), and a mirror of it.
ASSETS_BUCKET: Final = "blanchon/raywarp-assets"
URL_ENV: Final = "RAYWARP_BENCHMARKS_URL"


def key() -> str:
  """The scene files' version: the `.rayw` format and the USD converter
  (`v<format>-c<converter>`)."""
  from raywarp.formats.usd import version
  from raywarp.scene.model import rayw

  return f"v{rayw.VERSION}-c{version.CONVERTER}"


def scene_file(name: str) -> tuple[Path, tuple[tuple[float, float, float], tuple[float, float, float], float]]:
  """The local scene file of benchmark scene `name`, downloaded (and unpacked) on first use, and
  its camera (eye, target, fovy).

  Raises:
    RuntimeError: caching is off, or the file could not be fetched.
  """
  from raywarp.cache.paths import cache_dir
  from raywarp.core.build.kernel_cache import store

  root = cache_dir()
  if root is None:
    raise RuntimeError("caching is off (RAYWARP_CACHE_DIR is empty): the benchmark scenes need a cache directory")
  remote = os.environ.get(URL_ENV) or store.bucket_url(ASSETS_BUCKET)
  local, url = root / "benchmarks" / key(), f"{remote.rstrip('/')}/benchmarks/{key()}"
  rel = FILES[name]
  failed = store.fetch(local, url, store.listing(local, url, None), [rel, CAMERAS])
  if failed:
    raise RuntimeError(f"benchmark scene {name}: {next(iter(failed.values()))}")
  eye, target, fovy = json.loads((local / CAMERAS).read_text())[name]
  view = ((float(eye[0]), float(eye[1]), float(eye[2])), (float(target[0]), float(target[1]), float(target[2])), float(fovy))
  if rel.endswith(".tar"):
    path = local / FRANKA_XML
    if not path.is_file():
      with tarfile.open(local / rel) as tar:
        tar.extractall(local, filter="data")
    return path, view
  return local / rel, view


@dataclasses.dataclass(frozen=True)
class Result:
  """One benchmark run.

  Attributes:
    first_frame_s: seconds from the renderer's construction to its first frame on the device.
    ms_per_frame: mean milliseconds of a steady render (all worlds).
    images_per_s: images (worlds x cameras) per second.
    compiles: Warp modules, PTX and OptiX modules compiled or created in this process.
  """

  first_frame_s: float
  ms_per_frame: float
  images_per_s: float
  compiles: dict[str, int]


def measure(  # noqa: PLR0913  # the scene and keyword-only run parameters
  path: Path,
  view: tuple[tuple[float, float, float], tuple[float, float, float], float],
  *,
  worlds: int,
  res: int,
  settings: str,
  frames: int,
  device: str,
  png: Path | None,
) -> Result:
  """Loads scene file `path`, renders it `frames` times at `worlds` x the camera `view` (eye,
  target, fovy) at `res` x `res`; saves world 0's image as `png`."""
  import warp as wp

  import raywarp as rw
  from raywarp.viewer.camera import view_camera
  from raywarp.viewer.scenes import file_entry, load
  from raywarp.viewer.watch import WATCH

  WATCH.install()
  base = rw.RenderSettings()
  with wp.ScopedDevice(device):
    loaded = load(file_entry(path), device)
    start = time.perf_counter()
    r = rw.Renderer(
      loaded.gpu,
      nworld=worlds,
      cameras=[rw.CameraSpec(fovy=view[2])],
      resolution=(res, res),
      settings=base.replace(rw.FAST) if settings == "fast" else base,
    )
    r.camera_pose.assign(np.tile(view_camera(view).pose(), (worlds, 1, 1)))
    r.render(loaded.sim)
    wp.synchronize_device(device)
    first = time.perf_counter() - start
    r.render(loaded.sim)  # one more before timing (the steady state)
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(frames):
      r.render(loaded.sim)
    wp.synchronize_device(device)
    seconds = time.perf_counter() - start
    rgb = r.outputs.rgb
    if png is not None and rgb is not None:
      write_png(png, rgb.numpy()[0, 0, ..., :3])
    r.close()
  ms = seconds / frames * 1e3
  return Result(first, ms, worlds * frames / seconds, {str(kind): n for kind, n in WATCH.counts().items()})


def write_png(path: Path, rgb: np.ndarray) -> None:
  """An 8-bit RGB PNG of the (H, W, 3) uint8 image `rgb` (standard library only)."""

  def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

  h, w, _ = rgb.shape
  rows = b"".join(b"\0" + np.ascontiguousarray(rgb[y], dtype=np.uint8).tobytes() for y in range(h))
  header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def main(argv: list[str] | None = None) -> None:
  """The command line."""
  p = argparse.ArgumentParser(
    prog="python -m benchmarks.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--scene", choices=sorted(FILES), required=True)
  p.add_argument("--worlds", type=int, default=64, help="worlds per frame (default: 64)")
  p.add_argument("--res", type=int, default=512, help="square image side in pixels (default: 512)")
  p.add_argument("--settings", choices=("default", "fast"), default="default", help="RenderSettings(), or rw.FAST over it")
  p.add_argument("--frames", type=int, default=100, help="timed frames (default: 100)")
  p.add_argument("--device", default="cuda", help="Warp device (default: cuda)")
  p.add_argument("--png", type=Path, default=None, help="the saved image (default: benchmark-<scene>.png)")
  args = p.parse_args(argv)
  png = args.png or Path(f"benchmark-{args.scene}.png")
  path, view = scene_file(args.scene)
  result = measure(
    path,
    view,
    worlds=args.worlds,
    res=args.res,
    settings=args.settings,
    frames=args.frames,
    device=args.device,
    png=png,
  )
  print(f"scene {args.scene}: {args.worlds} worlds x {args.res}^2, settings {args.settings}, {args.frames} frames")
  print(f"  {result.ms_per_frame:.2f} ms/frame, {result.images_per_s:.0f} images/s")
  print(f"  first frame {result.first_frame_s:.2f} s, compiles {result.compiles}")
  print(f"  image: {png}")


if __name__ == "__main__":
  main()
