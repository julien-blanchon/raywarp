"""The viewer's state and its device work: one scene, one renderer, one camera.

Every method runs on the server's single render thread, so a command (scene, base, settings,
view, size) always lands between two frames. A settings change goes to the live renderer
(`Renderer.update_settings`: runtime fields on the next frame, compile-time fields switch to a
prebuilt tier variant); a change the renderer refuses leaves it on its previous settings and is
reported. A scene that fails to load is reported and the previous scene stays. The renderer is
rebuilt only for a new scene, image size or field of view. A scene opened and a compile-time
change start the background warm of the settings one change away and of every view (`warm.py`).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Final

import warp as wp

import raywarp as rw
from raywarp.core.build.kernel_cache import prebuilt
from raywarp.core.build.variants import nearby
from raywarp.viewer import controls, display
from raywarp.viewer.camera import Camera, resolution, view_camera
from raywarp.viewer.scenes import DEMO, Entry, Loaded, file_entry, load
from raywarp.viewer.warm import Request, Warmer
from raywarp.viewer.watch import WATCH

# The settings a viewer starts from (and resets to): the default and the fast set (`rw.FAST`).
BASES: Final = {"default": rw.RenderSettings(), "fast": rw.RenderSettings().replace(rw.FAST)}
# The settings groups (`RenderSettings`' fields); `outputs` is the viewer's own (the view).
GROUPS: Final = tuple(f.name for f in dataclasses.fields(rw.RenderSettings))
# Frame times kept for the averages.
_HISTORY: Final = 30


@dataclasses.dataclass(frozen=True)
class Options:
  """How a viewer starts.

  Attributes:
    device: the Warp device ("cuda", "cuda:1", "cpu").
    size: the first image (width, height).
    base: the first settings (a `BASES` name).
    scene: the first scene's catalogue id.
    warm: the modules one change away the background warm creates at most (0: no warm).
  """

  device: str = "cuda"
  size: tuple[int, int] = (1024, 576)
  base: str = "default"
  scene: str = DEMO
  warm: int = nearby.LIMIT


def bundle_info(device: wp.Device) -> dict[str, object]:
  """The prebuilt kernels serving `device`: the bundle's Warp release and content digest, and the
  OptiX module pack of this driver (None fields: none installed; the CPU has none)."""
  if not device.is_cuda:
    return {"kernels": None, "optix_modules": None, "arch": None}
  arch = int(device.arch)
  found = prebuilt.bundle(arch)
  pack = prebuilt.module_pack(arch, prebuilt.driver_version())
  kernels = None if found is None else f"sm{arch}, Warp {found.manifest.warp}, content {found.manifest.content[:12]}"
  return {"kernels": kernels, "optix_modules": None if pack is None else str(pack.root), "arch": arch}


def _without_outputs(patch: object) -> object:
  """A browser patch, refused when it names an output (the view menu owns the outputs).

  Raises:
    ValueError: for an `outputs.*` path.
  """
  if isinstance(patch, dict) and any(str(k).startswith("outputs.") for k in patch):
    raise ValueError("the outputs follow the view menu: pick a view instead of patching outputs.*")
  return patch


class Session:
  """The viewer: scene, base settings, settings, view, size, camera and the renderer."""

  def __init__(self, scenes: dict[str, Entry], options: Options) -> None:
    """Cheap: the first scene loads on the first command or frame (on the render thread).

    Raises:
      ValueError: for an unknown base, scene or an invalid size.
    """
    if options.base not in BASES:
      raise ValueError(f"unknown base {options.base!r}; one of {list(BASES)}")
    if options.scene not in scenes:
      raise ValueError(f"unknown scene {options.scene!r}")
    if not WATCH.installed:
      raise RuntimeError("install the compile watch first (raywarp.viewer.watch.WATCH.install())")
    self.scenes = scenes
    self.device = wp.get_device(options.device)
    self.size = resolution(*options.size)
    self.base = options.base
    self.settings: rw.RenderSettings = BASES[options.base]
    self.view = "rgb"
    self.camera = Camera()
    self.scene_id = options.scene
    self.loaded: Loaded | None = None
    self.renderer: rw.Renderer | None = None
    self.kernels = ""
    self.frame = 0
    self.times: list[float] = []
    self.message = ""
    self.warmer = Warmer(options.warm)

  # State -----------------------------------------------------------------------------------------

  def state(self) -> dict[str, object]:
    """A JSON snapshot for the browser (no device work)."""
    rows = [r for r in controls.schema(self.settings) if r["group"] != "outputs"]
    changed = controls.changed(BASES[self.base], self.settings)
    return {
      "scenes": [{"id": e.id, "name": e.name, "group": e.group} for e in self.scenes.values()],
      "scene": self.scene_id,
      "loaded": None if self.loaded is None else {"seconds": self.loaded.seconds, "extent": self.loaded.extent},
      "bases": list(BASES),
      "base": self.base,
      "settings": rows,
      "changed": sorted(changed),
      "views": list(display.VIEWS),
      "view": self.view,
      "size": list(self.size),
      "camera": controls.plain(self.camera),
      "device": str(self.device),
      "bundle": bundle_info(self.device),
      "kernels": self.kernels,
      "compiles": WATCH.counts(),
      "warm": self.warmer.status(),
      "message": self.message,
    }

  # Commands --------------------------------------------------------------------------------------

  def command(self, message: object) -> dict[str, object]:
    """Runs one browser command and returns the new state.

    Raises:
      ValueError: for a malformed command or one that failed (the state is unchanged).
    """
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
      raise ValueError("a command is a JSON object with a string 'type'")
    kind = message["type"]
    self.message = ""
    with wp.ScopedDevice(self.device):
      if kind == "scene":
        self.open(str(message.get("id")))
      elif kind == "path":
        self.open_path(str(message.get("path", "")))
      elif kind == "base":
        self.set_base(message.get("name"))
      elif kind == "settings":
        self.apply(controls.patch_settings(self.settings, _without_outputs(message.get("patch"))))
      elif kind == "view":
        self.set_view(message.get("name"))
      elif kind == "size":
        self.set_size(message.get("width"), message.get("height"))
      elif kind == "home":
        self.home()
      elif kind != "state":
        raise ValueError(f"unknown command {kind!r}")
    return self.state()

  def open(self, scene_id: str) -> Loaded:
    """Loads catalogue scene `scene_id` and frames its view (returned); on failure the previous
    scene stays.

    Raises:
      ValueError: for an unknown id or a scene that failed to load (its error named).
    """
    entry = self.scenes.get(scene_id)
    if entry is None:
      raise ValueError(f"unknown scene {scene_id!r}")
    try:
      loaded = load(entry, self.device)
      camera = view_camera(loaded.view)
      renderer = self._new_renderer(loaded, self.size, camera.fov)
    except Exception as e:  # any adapter or renderer failure is the user's to see, not a crash
      raise ValueError(f"{entry.name}: could not load: {type(e).__name__}: {e}") from e
    self._swap(renderer)
    self.loaded, self.scene_id, self.camera = loaded, scene_id, camera
    self.message = f"loaded {entry.name} in {loaded.seconds:.2f} s"
    self.warm()
    return loaded

  def open_path(self, text: str) -> None:
    """Adds scene file `text` (a path on the viewer's machine) to the catalogue and opens it.

    Raises:
      ValueError: for a missing file, an unknown format or a failed load.
    """
    entry = file_entry(text)
    known = entry.id in self.scenes
    self.scenes.setdefault(entry.id, entry)
    try:
      self.open(entry.id)
    except ValueError:
      if not known:
        del self.scenes[entry.id]  # a file that does not load is not listed
      raise

  def set_base(self, name: object) -> None:
    """Starts over from base settings `name` (`BASES`: every setting its).

    Raises:
      ValueError: for an unknown base or settings the renderer refuses.
    """
    if not isinstance(name, str) or name not in BASES:
      raise ValueError(f"unknown base {name!r}; one of {list(BASES)}")
    self.apply(BASES[name])
    self.base = name

  def apply(self, settings: rw.RenderSettings, view: str | None = None) -> None:
    """Switches to `settings` and view `view` (None: the current one) on the live renderer
    (made first, with the scene, when there is none yet), between two frames.

    Raises:
      ValueError: when the renderer refuses them (it stays on the previous settings and view).
    """
    view = self.view if view is None else view
    r = self.renderer if self.renderer is not None else self._renderer(self._scene())
    new = display.with_outputs(settings, [view])
    groups = {g: new.group(g) for g in GROUPS if new.group(g) != r.settings.group(g)}
    if groups:
      try:
        r.update_settings(**groups)
      except (ValueError, TypeError) as e:
        raise ValueError(f"the renderer refused the change: {e}") from e
    self.kernels = r.explain().kernels
    compiled = any(controls.change_kind(p) == "recompile" for p in controls.changed(self.settings, settings))
    self.settings, self.view = settings, view
    if compiled:
      self.warm()

  def warm(self) -> None:
    """Starts the background warm of the current scene, settings and view: the settings one change
    away and every view (`warm.Warmer`)."""
    if self.loaded is None:
      return
    outputs = tuple(dict.fromkeys(display.with_outputs(self.settings, [v]).outputs for v in display.VIEWS))
    settings = display.with_outputs(self.settings, [self.view])
    self.warmer.request(Request(self.loaded, self.size, self.camera.fov, settings, outputs))

  def set_view(self, name: object) -> None:
    """Shows view `name` (`display.VIEWS`: its outputs are switched at run time).

    Raises:
      ValueError: for an unknown view.
    """
    if name not in display.VIEWS:
      raise ValueError(f"unknown view {name!r}")
    self.apply(self.settings, str(name))

  def set_size(self, width: object, height: object) -> None:
    """Renders `width` x `height` images from the next frame (a new renderer).

    Raises:
      ValueError: for a size outside `camera.MAX_SIZE`.
    """
    size = resolution(width, height)
    loaded = self._scene()
    self._swap(self._new_renderer(loaded, size, self.camera.fov))
    self.size = size

  def home(self) -> None:
    """Back to the scene's opening view."""
    if self.loaded is not None:
      self.camera = view_camera(self.loaded.view)

  # Frames ----------------------------------------------------------------------------------------

  def _scene(self) -> Loaded:
    """The current scene (loaded on first use)."""
    return self.loaded if self.loaded is not None else self.open(self.scene_id)

  def _new_renderer(self, loaded: Loaded, size: tuple[int, int], fov: float) -> rw.Renderer:
    """A renderer of `loaded` at `size` and vertical FOV `fov` with the current settings and view.

    The FOV is the camera spec's (a new renderer: its kernels are the process's, nothing
    compiles); the per-world "camera.fovy" table would change it at run time, but declaring it
    enters the resolved Features of every build, which the prebuilt tiers are not built for.
    """
    camera = rw.CameraSpec(fovy=fov)
    settings = display.with_outputs(self.settings, [self.view])
    return rw.Renderer(loaded.gpu, nworld=1, cameras=[camera], resolution=size, settings=settings)

  def _swap(self, renderer: rw.Renderer) -> None:
    """Makes `renderer` the live one (the previous one released)."""
    self._close_renderer()
    self.renderer, self.kernels = renderer, renderer.explain().kernels

  def _renderer(self, loaded: Loaded) -> rw.Renderer:
    """The live renderer (a new one for another field of view)."""
    r = self.renderer
    if r is not None and abs(r.cameras[0].fovy - self.camera.fov) < 1e-6:
      return r
    renderer = self._new_renderer(loaded, self.size, self.camera.fov)
    self._swap(renderer)
    return renderer

  def _close_renderer(self) -> None:
    """Releases the renderer (the next frame makes a new one)."""
    if self.renderer is not None:
      self.renderer.close()
    self.renderer = None

  def render(self, camera: object) -> tuple[display.Image, dict[str, object]]:
    """One frame from fly camera `camera` (JSON; None: the current one): its (H, W, 3) uint8
    display image and its info.

    Raises:
      ValueError: for a malformed camera or a scene that failed to load.
    """
    if camera is not None:
      self.camera = Camera.parse(camera)
    with wp.ScopedDevice(self.device):
      loaded = self._scene()
      r = self._renderer(loaded)
      seq = WATCH.total()
      r.camera_pose.assign(self.camera.pose().reshape(1, 1, 7))
      start = time.perf_counter()
      r.render(loaded.sim, seed=self.frame)
      wp.synchronize_device(self.device)
      render_ms = (time.perf_counter() - start) * 1e3
      start = time.perf_counter()
      data = display.read(r, display.needed(self.view))
      image = display.view_image(self.view, data, r.settings)
      read_ms = (time.perf_counter() - start) * 1e3
    self.frame += 1
    self.times = [*self.times[-(_HISTORY - 1) :], render_ms]
    status = r.compile_status()
    return image, {
      "frame": self.frame,
      "size": list(self.size),
      "view": self.view,
      "render_ms": render_ms,
      "render_ms_avg": sum(self.times) / len(self.times),
      "read_ms": read_ms,
      "kernels": self.kernels,
      "pending": len(status.pending),
      "compiles": WATCH.counts(),
      "new_events": WATCH.total() - seq,
    }

  def close(self) -> None:
    """Releases the renderer."""
    self._close_renderer()
