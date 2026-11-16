"""`raywarp view [SCENE ...] [--scenes-dir DIR] [--device D] [--port P]`: serves the web viewer
and prints its URL (`raywarp.viewer`)."""

from __future__ import annotations

import argparse
import secrets

from raywarp.viewer.session import BASES, Options


def _size(text: str) -> tuple[int, int]:
  """A "WIDTHxHEIGHT" command-line value."""
  from raywarp.viewer.camera import resolution  # noqa: PLC0415  # imports raywarp: after --help

  try:
    width, height = (int(v) for v in text.lower().split("x"))
    return resolution(width, height)
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"size {text!r}: {e}") from None


def add_arguments(p: argparse.ArgumentParser) -> None:
  """The viewer's options on parser `p` (`raywarp view`)."""
  p.add_argument("scenes", nargs="*", help="scene files: .wpbr, USD (.usd/.usda/.usdc/.usdz), glTF (.gltf/.glb), MJCF (.xml)")
  p.add_argument("--scenes-dir", action="append", default=[], help="a directory of scene files (and its subdirectories)")
  p.add_argument("--open", default=None, help="the scene shown first (a file given; default: the first file, else the demo)")
  p.add_argument("--device", default="cuda", help="Warp device: cuda, cuda:N or cpu (default: cuda)")
  p.add_argument("--host", default="127.0.0.1", help="address to bind (default: loopback, for an SSH port forward)")
  p.add_argument("--port", type=int, default=8080, help="port (default: 8080; 0: any free port)")
  p.add_argument("--size", type=_size, default=(1024, 576), help="first image WIDTHxHEIGHT (default: 1024x576)")
  p.add_argument("--base", default="default", choices=list(BASES), help="first settings: default or fast (rw.FAST)")
  p.add_argument(
    "--warm",
    type=int,
    default=None,
    help="after a scene opens, prepare in the background the modules (OptiX, or the software traversal's Warp modules)"
    " of every view and of the settings one change away, at most this many (default: raywarp.core.build.variants.nearby.LIMIT;"
    " 0: no background warm)",
  )
  p.add_argument("--no-token", action="store_true", help="serve the API without the session token")


def _report(url: str, port: int, info: dict[str, object], device: str) -> None:
  """Prints where the viewer is and what serves its kernels."""
  kernels = info["kernels"] or (
    "none: the CPU compiles its kernels" if info["arch"] is None else "NOT INSTALLED: kernels compile"
  )
  print(f"raywarp viewer: {url}", flush=True)
  print(f"  device {device}; prebuilt kernels: {kernels}", flush=True)
  if info["arch"] is not None:
    print(f"  OptiX module pack: {info['optix_modules'] or 'none (OptiX creates each module once on first use)'}", flush=True)
  print(f"  from another machine: ssh -N -L {port}:127.0.0.1:{port} <this host>, then open the URL there", flush=True)


def run(args: argparse.Namespace) -> int:
  """Serves the viewer of the parsed `args` until interrupted."""
  import warp as wp  # noqa: PLC0415  # after the command line (fast --help)

  from raywarp.viewer.scenes import DEMO, catalogue, file_entry  # noqa: PLC0415
  from raywarp.viewer.server import RenderThread, ViewerServer, serve  # noqa: PLC0415
  from raywarp.viewer.session import Session, bundle_info  # noqa: PLC0415
  from raywarp.viewer.watch import WATCH  # noqa: PLC0415

  try:
    files = [*args.scenes, *([args.open] if args.open else [])]
    scenes = catalogue(files, args.scenes_dir)
    first = file_entry(args.open or args.scenes[0]).id if files else DEMO
  except ValueError as e:
    raise SystemExit(f"raywarp viewer: {e}") from None
  from raywarp.core.build.variants.nearby import LIMIT  # noqa: PLC0415

  warm = LIMIT if args.warm is None else args.warm
  wp.init()
  WATCH.install()
  device = wp.get_device(args.device)
  session = Session(scenes, Options(device=str(device), size=args.size, base=args.base, scene=first, warm=warm))
  token = "" if args.no_token else secrets.token_urlsafe(18)
  try:
    server = ViewerServer((args.host, args.port), RenderThread(session), token)
  except OSError as e:
    raise SystemExit(f"raywarp viewer: cannot listen on {args.host}:{args.port}: {e} (try --port 0)") from None
  port = server.server_address[1]
  host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
  url = f"http://{host}:{port}/" + (f"?token={token}" if token else "")
  _report(url, port, bundle_info(device), f"{device} ({device.name})")
  serve(server)
  return 0
