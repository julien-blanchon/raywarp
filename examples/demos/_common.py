"""What every gallery demo shares (docs/gallery.md): the command line, scene loading, camera paths,
the recorder and host-composed frames.

Every demo has the same options: `--device` (cuda), `--worlds`, `--size` (the video frame: `WxH` or
one number for a square; a batch is tiled on it), `--frames` (30 per second) and `--out` (the MP4).
Scenes are a benchmark scene of `benchmarks/run.py` (franka, showcase, sponza: downloaded on first
use), a fixture name of `tests/data/scenes/` (cornell_rect, open_plan, ...) or a scene file path.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import warp as wp

import raywarp as rw
from raywarp.cli.record import orbit
from raywarp.video import Recorder, recorder
from raywarp.viewer.camera import as_view, authored_view, bounds, home_view, view_camera
from raywarp.viewer.scenes import mjcf_views

if TYPE_CHECKING:
  import mujoco
  import mujoco_warp as mjw
  from shape_extensions import IntVar

REPO: Final = Path(__file__).resolve().parents[2]
FIXTURES: Final = REPO / "tests" / "data" / "scenes"
BENCHMARK_SCENES: Final = ("franka", "showcase", "sponza")
FPS: Final = 30
# A camera view: (eye, target, fovy [degrees]).
type View = tuple[tuple[float, float, float], tuple[float, float, float], float]
# 8-bit RGB(A) host images.
type Image[H: IntVar, W: IntVar] = np.ndarray[tuple[H, W, int], np.dtype[np.uint8]]
type Rgb = wp.array[wp.vec4ub, [int, int, int, int]]


def cli(main: Callable[..., None], doc: str | None) -> None:
  """Runs `main` with its keyword arguments as `--options` (their defaults and types)."""
  p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
  for name, param in inspect.signature(main).parameters.items():
    default = param.default
    p.add_argument(f"--{name}", type=type(default), default=default, help=f"(default: {default})")
  main(**vars(p.parse_args()))


def parse_size(size: str) -> tuple[int, int]:
  """(width, height) of `WxH` or of one number (a square)."""
  w, _, h = size.lower().partition("x")
  return int(w), int(h or w)


def tile(size: str, count: int, cols: int = 0) -> tuple[tuple[int, int], int]:
  """The (width, height) of each of `count` images tiled `cols` per row (0: a square grid) on a
  `size` frame (even sizes), and the tiles per row."""
  width, height = parse_size(size)
  cols = cols or math.ceil(math.sqrt(count))
  rows = math.ceil(count / cols)
  return (max(2, width // cols // 2 * 2), max(2, height // rows // 2 * 2)), cols


def record(out: str, rgb: Rgb, cols: int = 0) -> Recorder:
  """A recorder of the renderer output `rgb`: every world and camera tiled on one frame."""
  return recorder(out, tuple(rgb.shape), fps=FPS, layout="mosaic", cols=cols)


def scene_path(scene: str) -> tuple[Path, View | None]:
  """The file of `scene` (benchmark scene, fixture name or path) and its benchmark view."""
  if scene in BENCHMARK_SCENES:
    sys.path.insert(0, str(REPO))  # benchmarks/ lives in the repository, beside examples/
    from benchmarks.run import scene_file  # noqa: PLC0415  # the repository's, on use

    return scene_file(scene)
  for suffix in (".usda", ".xml"):
    if (FIXTURES / f"{scene}{suffix}").is_file():
      return FIXTURES / f"{scene}{suffix}", None
  return Path(scene), None


@dataclasses.dataclass
class Stage:
  """A scene on the device, ready to render.

  Attributes:
    gpu: the device scene.
    view: its opening view.
    sim: the simulator state `render` reads (MuJoCo Warp data; None: a static scene).
    mjm: the MuJoCo model (MJCF scenes).
    m: the MuJoCo Warp model (MJCF scenes).
    extent: the scene's bounding radius (world units).
  """

  gpu: rw.GpuScene
  view: View
  extent: float = 1.0
  sim: mjw.Data | None = None
  mjm: mujoco.MjModel | None = None
  m: mjw.Model | None = None

  def step(self) -> None:
    """Advances the simulation one step (nothing for a static scene)."""
    if self.m is not None and self.sim is not None:
      import mujoco_warp as mjw  # noqa: PLC0415  # the mujoco extra, MJCF scenes only

      mjw.step(self.m, self.sim)


def load_scene(scene: str, nworld: int, lighting: rw.Lighting | None = None) -> Stage:
  """`scene` on Warp's current device, `lighting` added (USD, glTF, `.rayw`); an MJCF model gets
  `nworld` worlds of MuJoCo Warp data."""
  path, view = scene_path(scene)
  if path.suffix == ".xml":
    return _load_mjcf(path, nworld, view)
  host = host_scene(path)
  if lighting is not None:
    host = host.with_lighting(lighting)
  box = bounds(host)
  extent = 1.0 if box is None else float(np.linalg.norm(box[1] - box[0])) / 2
  return Stage(rw.upload(host), view or authored_view(host) or home_view(host), extent)


def host_scene(path: Path) -> rw.Scene:
  """The host scene of a `.rayw`, USD or glTF file (cached as `.rayw` after the first read)."""
  if path.suffix == ".rayw":
    return rw.Scene.load(path)
  if path.suffix in (".gltf", ".glb"):
    import raywarp.formats.gltf  # noqa: PLC0415  # the gltf extra, on use

    return raywarp.formats.gltf.load(path, lighting=rw.Lighting.studio())
  import raywarp.formats.usd  # noqa: PLC0415  # the usd extra, on use

  return raywarp.formats.usd.load(path)


def _load_mjcf(path: Path, nworld: int, view: View | None) -> Stage:
  """An MJCF model in `nworld` worlds of MuJoCo Warp, at rest."""
  import mujoco  # noqa: PLC0415  # the mujoco extra, on use
  import mujoco_warp as mjw  # noqa: PLC0415

  import raywarp.integrations.mujoco  # noqa: PLC0415

  mjm = mujoco.MjModel.from_xml_path(str(path))
  m, d = mjw.put_model(mjm), mjw.make_data(mjm, nworld=nworld)
  mjw.forward(m, d)
  gpu = raywarp.integrations.mujoco.scene(mjm, m, d)
  return Stage(gpu, view or mjcf_view(mjm), float(mjm.stat.extent), d, mjm, m)


def mjcf_view(mjm: mujoco.MjModel) -> View:
  """The view of a MuJoCo model's first camera at rest (else one framing the model)."""
  import mujoco  # noqa: PLC0415  # the mujoco extra, on use

  host = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, host)
  return mjcf_views(mjm, host)[0]


def look_at(eye: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray) -> np.ndarray:
  """(7,) camera pose (position, quaternion xyzw) at `eye` looking at `target`, z up."""
  return view_camera(as_view(np.asarray(eye, float), np.asarray(target, float), 60.0)).pose()


def orbit_pose(view: View, frame: int, frames: int, degrees: float = 60.0) -> np.ndarray:
  """(7,) pose of `view` turned about the vertical through its target: a back-and-forth swing of
  `degrees` over the clip (the clip loops)."""
  return orbit(view, 0.5 * degrees * math.sin(2.0 * math.pi * frame / max(frames, 1)))


def pose_cameras(r: rw.Renderer, pose: np.ndarray) -> None:
  """Poses every free camera of every world at `pose` (7,)."""
  r.camera_pose.assign(np.broadcast_to(pose, (*r.camera_pose.shape, 7)).astype(np.float32))


def renderer(
  stage: Stage,
  nworld: int,
  resolution: tuple[int, int],
  settings: rw.RenderSettings | None = None,
  per_world: rw.PerWorld | None = None,
  cameras: list[rw.CameraSpec] | None = None,
) -> rw.Renderer:
  """A Renderer of `stage` with free cameras (one at the view's field of view by default)."""
  return rw.Renderer(
    stage.gpu,
    nworld=nworld,
    cameras=cameras or [rw.CameraSpec(fovy=stage.view[2])],
    resolution=resolution,
    settings=settings,
    per_world=per_world,
  )


def rgb_of(r: rw.Renderer) -> Rgb:
  """The renderer's display color (RGBA8, (nworld, ncam, H, W))."""
  rgb = r.outputs.rgb
  if rgb is None or rgb.dtype is not wp.vec4ub:
    raise ValueError("the demo needs the default rgb output (ColorOutput())")
  return cast("Rgb", rgb)


class Canvas:
  """A host-composed frame (side-by-side panels): `show(image)`, then `rec.write(canvas.rgb)`."""

  def __init__(self, width: int, height: int) -> None:
    """A black (1, 1, height, width) RGBA8 frame on Warp's current device."""
    self.rgb: Rgb = wp.zeros((1, 1, height, width), dtype=wp.vec4ub)

  def show(self, image: Image) -> None:
    """Shows the (h, w, 3 or 4) uint8 `image` centred on the frame (cropped if larger)."""
    h, w = self.rgb.shape[2:]
    image = image[:h, :w, :3]
    y, x = (h - image.shape[0]) // 2, (w - image.shape[1]) // 2
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[y : y + image.shape[0], x : x + image.shape[1], :3] = image
    self.rgb.assign(rgba[None, None])


def row(images: list[Image], gap: int = 4) -> Image:
  """Images of one height side by side, `gap` white pixels apart."""
  h = images[0].shape[0]
  white = np.full((h, gap, 3), 255, np.uint8)
  parts = [p for im in images for p in (im[..., :3], white)][:-1]
  return np.concatenate(parts, axis=1)


def grid(rows: list[list[Image]], gap: int = 4) -> Image:
  """Rows of images (`row`) stacked, `gap` white pixels apart."""
  lines = [row(r, gap) for r in rows]
  white = np.full((gap, lines[0].shape[1], 3), 255, np.uint8)
  return np.concatenate([p for line in lines for p in (line, white)][:-1], axis=0)


def colorize(values: np.ndarray, lo: float, hi: float) -> Image:
  """A float image on a blue-yellow ramp from `lo` to `hi` (0, NaN, inf: black)."""
  t = np.clip((np.nan_to_num(values, nan=lo, posinf=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
  ramp = np.stack([0.27 + 0.72 * t**1.5, 0.0 + 0.9 * t, 0.33 + 0.4 * np.sin(math.pi * t) - 0.2 * t], axis=-1)
  out = (np.clip(ramp, 0.0, 1.0) * 255).astype(np.uint8)
  out[~np.isfinite(values) | (values == 0)] = 0
  return out


def label_colors(ids: np.ndarray) -> Image:
  """A distinct color per integer id (-1, a miss: black)."""
  h = (ids.astype(np.int64) * 2654435761) & 0xFFFFFF
  out = np.stack([(h >> 16) & 255, (h >> 8) & 255, h & 255], axis=-1).astype(np.uint8)
  out[ids < 0] = 0
  return out
