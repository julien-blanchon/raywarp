"""The viewer's scenes: a built-in demo, scene files and scene directories, and their loading.

A scene file is a prepared `.rayw` host scene, a USD stage (read through the `.rayw` cache, the
`usd` extra), a glTF asset (converted to USD once and cached; lit by the studio rig) or an MJCF
model (MuJoCo Warp at its initial state, the `mujoco` extra). Every scene is Z-up. A scene opens on
its first authored perspective camera, else a view framing its bounds.
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import warp as wp

import raywarp as rw
from raywarp.viewer.camera import View, as_view, authored_view, bounds, fit, home_view, norm

if TYPE_CHECKING:
  import mujoco

type Format = Literal["demo", "rayw", "usd", "gltf", "mjcf"]

# Scene file suffix -> its format.
FORMATS: Final[dict[str, Format]] = {
  ".rayw": "rayw",
  ".usd": "usd",
  ".usda": "usd",
  ".usdc": "usd",
  ".usdz": "usd",
  ".gltf": "gltf",
  ".glb": "gltf",
  ".xml": "mjcf",
  ".mjcf": "mjcf",
}
# Scenes listed from directories at most (a directory of thousands of layers is not a catalogue).
MAX_LISTED: Final = 500
DEMO: Final = "demo"


@dataclasses.dataclass(frozen=True)
class Entry:
  """A selectable scene.

  Attributes:
    id: its catalogue key ("demo", else the file's absolute path).
    name: the name shown.
    group: the group shown (the directory it was listed from, "files", "built-in").
    path: the scene file (None: the demo).
  """

  id: str
  name: str
  group: str
  path: Path | None = None

  @property
  def format(self) -> Format:
    """The scene's format (`FORMATS`)."""
    return "demo" if self.path is None else FORMATS[self.path.suffix.lower()]


@dataclasses.dataclass
class Loaded:
  """A scene on the device.

  Attributes:
    entry: the catalogue entry.
    gpu: the device scene.
    sim: the simulator state `Renderer.render` takes (MuJoCo Warp data; None: static).
    view: the view it opens on.
    extent: its bounding radius (world units; the fly speed scales with it).
    seconds: the load time (read, convert or cache hit, and upload).
  """

  entry: Entry
  gpu: rw.GpuScene
  sim: object
  view: View
  extent: float
  seconds: float = 0.0


def file_entry(text: str | os.PathLike[str], group: str = "files") -> Entry:
  """The entry of scene file `text` (user home expanded, made absolute).

  Raises:
    ValueError: for a missing file or a suffix outside `FORMATS`.
  """
  path = Path(text).expanduser()
  if not path.is_file():
    raise ValueError(f"no scene file at {str(path)!r}")
  if path.suffix.lower() not in FORMATS:
    raise ValueError(f"{path.name}: not a scene file (one of {', '.join(sorted(FORMATS))})")
  path = path.resolve()
  return Entry(str(path), path.name, group, path)


def _listed(directory: Path) -> Iterable[Path]:
  """The scene files of `directory` and of its direct subdirectories, sorted."""
  found = [p for p in directory.iterdir() if p.is_file()]
  found += [p for d in sorted(directory.iterdir()) if d.is_dir() for p in d.iterdir() if p.is_file()]
  return sorted((p for p in found if p.suffix.lower() in FORMATS), key=lambda p: str(p).lower())


def catalogue(files: Iterable[str | os.PathLike[str]], directories: Iterable[str | os.PathLike[str]]) -> dict[str, Entry]:
  """The demo, the scene files and the scene files found in `directories` (at most `MAX_LISTED`).

  Raises:
    ValueError: for a scene file that is missing or of no known format, or a missing directory.
  """
  entries = {DEMO: Entry(DEMO, "PBR demo (built in)", "built-in")}
  for f in files:
    e = file_entry(f)
    entries[e.id] = e
  for d in directories:
    root = Path(d).expanduser()
    if not root.is_dir():
      raise ValueError(f"no scene directory at {str(root)!r}")
    for p in _listed(root):
      if len(entries) >= MAX_LISTED:
        raise ValueError(f"more than {MAX_LISTED} scenes listed: give fewer or narrower directories")
      e = file_entry(p, group=root.name)
      entries.setdefault(e.id, e)
  return entries


def demo() -> rw.Scene:
  """A red rough and a blue metallic sphere on a floor, an area light and a grey environment."""
  builder = rw.SceneBuilder()
  for x, color, metal in [(-0.9, (0.7, 0.12, 0.06), 0.0), (0.9, (0.15, 0.4, 0.8), 0.8)]:
    mat = builder.add_material(rw.Material(base_color=color, base_metalness=metal, specular_roughness=0.3))
    builder.add_instance("sphere", material=mat, size=(0.65, 0, 0), xform=np.c_[np.eye(3), [x, 0, 0.65]])
  floor = builder.add_material(rw.Material(base_color=(0.55, 0.6, 0.65), specular_roughness=0.8))
  builder.add_instance("box", material=floor, size=(3, 3, 0.1), xform=np.c_[np.eye(3), [0, 0, -0.1]])
  builder.add_light(rw.Light(kind=rw.scene.LightKind.SPHERE, intensity=30000, radius=0.3, xform=np.c_[np.eye(3), [1, -2, 4]]))
  return builder.build().with_lighting(rw.Lighting(envmap=np.full((8, 16, 3), 0.2, np.float32)))


def demo_view() -> View:
  """The demo's close view."""
  return fit(np.array([0.0, 0.0, 0.6]), 1.8).view()


def _host(entry: Entry) -> rw.Scene:
  """The host scene of a demo, `.rayw`, USD or glTF entry (adapters imported on use)."""
  path = entry.path
  if path is None:
    return demo()
  if entry.format == "rayw":
    return rw.Scene.load(path)
  if entry.format == "usd":
    import raywarp.formats.usd  # noqa: PLC0415  # the usd extra, on use

    return raywarp.formats.usd.load(path, cache=True)
  import raywarp.formats.gltf  # noqa: PLC0415  # the usd extra and pygltflib, on use

  # A glTF asset is usually unlit: the studio rig lights it (as the adapter's documented example).
  return raywarp.formats.gltf.load(path, cache=True, lighting=rw.Lighting.studio())


def mjcf_views(mjm: mujoco.MjModel, data: mujoco.MjData) -> tuple[View, ...]:
  """The views of a MuJoCo model's cameras (target 1 unit ahead), else one framing its extent."""
  views = []
  for cam in range(mjm.ncam):
    eye = np.asarray(data.cam_xpos[cam], float)
    forward = -np.asarray(data.cam_xmat[cam], float).reshape(3, 3)[:, 2]
    views.append(as_view(eye, eye + forward, float(mjm.cam_fovy[cam])))
  return tuple(views) or (fit(np.asarray(mjm.stat.center, float), float(mjm.stat.extent)).view(),)


def _load_mjcf(entry: Entry, device: wp.Device) -> Loaded:
  """An MJCF model on `device` at its initial state (MuJoCo Warp, the mujoco extra)."""
  import mujoco  # noqa: PLC0415  # the mujoco extra, on use
  import mujoco_warp as mjw  # noqa: PLC0415

  import raywarp.integrations.mujoco  # noqa: PLC0415

  if entry.path is None:
    raise ValueError("an MJCF entry has a path")
  mjm = mujoco.MjModel.from_xml_path(os.fspath(entry.path))
  data = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, data)
  with wp.ScopedDevice(device):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, data)
    gpu = raywarp.integrations.mujoco.scene(mjm, m, d)
  return Loaded(entry, gpu, d, mjcf_views(mjm, data)[0], max(float(mjm.stat.extent), 0.25))


def load(entry: Entry, device: wp.DeviceLike) -> Loaded:
  """Loads `entry` onto `device` (the adapter's exceptions propagate: the viewer reports them)."""
  start = time.perf_counter()
  dev = wp.get_device(device)
  if entry.format == "mjcf":
    loaded = _load_mjcf(entry, dev)
  else:
    host = _host(entry)
    box = bounds(host)
    extent = 2.0 if box is None else max(norm(box[1] - box[0]) * 0.5, 0.25)
    view = demo_view() if entry.path is None else authored_view(host) or home_view(host)
    loaded = Loaded(entry, rw.upload(host, dev), None, view, extent)
  wp.synchronize_device(dev)
  loaded.seconds = time.perf_counter() - start
  return loaded
