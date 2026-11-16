"""ViewerRaywarpHeadless: a Newton viewer that ray traces its frames with no window and no OpenGL.

Composition: Newton's `ViewerNull` (the viewer API, a frame budget, no-op logging) + Newton's
viewer `Camera` + `PbrFrames`. Needs pyglet only for the camera's vector maths (no display).
"""

from __future__ import annotations

from typing import override

import newton
import warp as wp
from newton._src.viewer.camera import Camera
from newton.viewer import ViewerNull

from raywarp._base import arguments
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.integrations.newton.build import Stage
from raywarp.integrations.newton.headless.frames import PbrFrames, copy_frame, drop_scene
from raywarp.scene.lighting.rig import Lighting


class ViewerRaywarpHeadless(ViewerNull):
  """A Newton viewer whose frames are rendered by raywarp, without window or OpenGL.

  A drop-in for :class:`~newton.viewer.ViewerNull` (runs ``num_frames`` frames, logging is a
  no-op) that renders, at every :meth:`end_frame`, the last logged state from its camera
  (:meth:`set_camera`, ``width`` x ``height``); :meth:`get_frame` returns it, e.g. to record
  videos. The visibility toggles (`show_*`, `set_visible_worlds`) apply to the model's shapes (a
  ``usd=`` stage's visuals stay shown). Without ``lighting`` it reproduces ViewerGL's look
  (`raywarp.integrations.newton.headless.look`).

  Attributes:
    camera: Newton's viewer camera (pose, field of view, clip planes, resolution).
    pbr: the PBR frames (scene, renderers, last images).
    seed: Random seed of the next frame (soft shadows).
  """

  def __init__(  # noqa: PLR0913  # ViewerNull's signature, positional order kept
    self,
    num_frames: int = 1000,
    benchmark: bool = False,  # noqa: FBT001, FBT002  # ViewerNull's signature
    benchmark_timeout: float | None = None,
    benchmark_start_frame: int = 3,
    *,
    width: int = 1920,
    height: int = 1080,
    lighting: Lighting | None = None,
    render_device: wp.DeviceLike = None,
    usd: Stage | None = None,
    scene_options: SceneOptions = DEFAULT_SCENE_OPTIONS,
    **render_options: object,
  ) -> None:
    """Create the viewer (same arguments as ViewerNull, plus the frame size and PBR options).

    Args:
      num_frames: Frames before :meth:`is_running` returns False.
      benchmark: ViewerNull's benchmark timing (FPS after the warm-up frames).
      benchmark_timeout: Stop after this many seconds of timed frames (enables ``benchmark``).
      benchmark_start_frame: Warm-up frames before the benchmark timing starts.
      width: Frame width [px].
      height: Frame height [px].
      lighting: Lights, ambient, environment and headlight of the scene (``rw.Lighting``; its
        ``ev100`` is the viewer's manual ``exposure.ev100`` unless a render option sets it). None:
        the stage's UsdLux lights (``usd``), else ViewerGL's look.
      render_device: Warp device that renders (default: the model's).
      usd: The stage (path or ``Usd.Stage``) the model was loaded from: its visuals, materials
        and, without ``lighting``, UsdLux lights (`load_usd`).
      scene_options: The scene's texture storage and size cap, emissive panels and environment
        sun (``rw.SceneOptions``).
      **render_options: `RenderSettings.replace` keys: groups (``aa={"spp": 2}``) or
        ``"group.field"`` keys (``**{"performance.traversal": "optix"}``).

    Raises:
      ValueError: for a width or height that is not an integer >= 1.
    """
    width = arguments.check_int("ViewerRaywarpHeadless: width", width, low=1)
    height = arguments.check_int("ViewerRaywarpHeadless: height", height, low=1)
    self.pbr = PbrFrames(
      lighting,
      render_options,
      owner="ViewerRaywarpHeadless",
      particles="not rendered (headless)",
      device=render_device,
      usd=usd,
      scene_options=scene_options,
    )
    super().__init__(
      num_frames=num_frames,
      benchmark=benchmark,
      benchmark_timeout=benchmark_timeout,
      benchmark_start_frame=benchmark_start_frame,
    )
    self.camera = Camera(width=width, height=height, up_axis="Z")
    self.seed = 0
    self._last_state: newton.State | None = None

  @override
  def set_model(self, model: newton.Model | None) -> None:
    """ViewerBase's `set_model`, the camera on the model's up axis, then the PBR scene."""
    super().set_model(model)
    if model is not None:
      up_axis = int(newton.Axis.from_any(model.up_axis))
      if up_axis != self.camera.up_axis:
        self.camera = Camera(width=self.camera.width, height=self.camera.height, up_axis=up_axis)
    self.pbr.set_model(model)

  @override
  def clear_model(self) -> None:
    """ViewerBase's `clear_model`, and the PBR scene dropped."""
    super().clear_model()
    self._last_state = None
    drop_scene(self)

  @override
  def set_camera(self, pos: wp.vec3, pitch: float | None = None, yaw: float | None = None) -> None:
    """Camera position [m], pitch and yaw [deg] (ViewerGL's conventions)."""
    from pyglet.math import Vec3  # noqa: PLC0415  # the camera's vector type (pyglet: optional)

    self.camera.pos = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
    if pitch is not None:
      self.camera.pitch = max(min(pitch, 89.0), -89.0)
    if yaw is not None:
      self.camera.yaw = (yaw + 180.0) % 360.0 - 180.0
    self.camera.sync_pivot_to_view()

  @override
  def log_state(self, state: newton.State) -> None:
    """Keeps the state rendered at the next :meth:`end_frame` (no per-shape batches)."""
    self._last_state = state

  @override
  def end_frame(self) -> None:
    """Renders the PBR frame of the last logged state, then counts the frame."""
    if not self.is_rendering_paused():
      self.render_pbr()
    super().end_frame()

  def render_pbr(self) -> wp.array[wp.uint8, [int, int, 3]] | None:
    """Renders the PBR frame of the last logged state from the camera (or None)."""
    size = (self.camera.width, self.camera.height)
    return self.pbr.render_viewer(self, self._last_state, self.camera, size=size, seed=self.seed)

  @override
  def get_frame(
    self, *, output: wp.array[wp.uint8, [int, int, int]] | None = None, render_ui: bool = False
  ) -> wp.array[wp.uint8, [int, int, int]]:
    """The last PBR frame, (height, width, 3) uint8 with a top-left origin.

    Raises:
      NotImplementedError: for `render_ui` (a headless viewer draws no UI: `ViewerBase`'s
        contract for an unsupported capture option).
      RuntimeError: when no frame was rendered yet.
      ValueError: for an output of another shape.
    """
    if render_ui:
      raise NotImplementedError("ViewerRaywarpHeadless.get_frame: render_ui=True (a headless viewer draws no UI)")
    frame = self.pbr.frame
    if frame is None:
      raise RuntimeError("ViewerRaywarpHeadless.get_frame: no frame rendered yet (log_state and end_frame first)")
    return copy_frame(frame, output)

  @override
  def close(self) -> None:
    """Drops the PBR scene."""
    self.pbr.set_model(None)
    super().close()
