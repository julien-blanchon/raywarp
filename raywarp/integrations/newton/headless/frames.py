"""The PBR frames of the headless viewer: one merged scene of a Newton model rendered from a
viewer camera.

`PbrFrames` is the component `ViewerRaywarpHeadless` (a ViewerNull) is composed with: it builds the
scene once per model (every shape a viewer toggle can show; the toggles write a per-instance
visibility mask, no rebuild and no stage re-read), renders it from the viewer's camera and
presents the result as a GPU RGB image.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from typing import Final, Protocol, cast, final

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerBase

from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, GpuScene
from raywarp.core.pipeline.platform.device import host_copy
from raywarp.core.pipeline.renderer import Renderer
from raywarp.core.pipeline.upload.build import upload
from raywarp.core.pipeline.upload.options import DEFAULT_SCENE_OPTIONS, SceneOptions
from raywarp.core.pipeline.worlds.sharing import PerWorld
from raywarp.core.pipeline.worlds.tables import InstanceColumns
from raywarp.integrations.newton import from_usd
from raywarp.integrations.newton.api import binding as model_binding
from raywarp.integrations.newton.binding import NewtonBinding
from raywarp.integrations.newton.build import Stage
from raywarp.integrations.newton.build import load as model_load
from raywarp.integrations.newton.headless.look import GL_CHECKER_COLORS, gl_look
from raywarp.integrations.newton.options import emissive_lights
from raywarp.integrations.newton.options import render_options as render_options_of
from raywarp.integrations.newton.types import Checkerboard, NewtonState, Options, StateView
from raywarp.scene.lighting.rig import Lighting
from raywarp.settings import Outputs, RenderSettings

# RenderSettings fields the viewers set themselves.
_RESERVED_OPTIONS: Final = ("outputs",)

type Size = tuple[int, int]
type RgbImage = wp.array[wp.uint8, [int, int, 3]]


@wp.kernel(enable_backward=False)
def _present(
  # In:
  rgba: wp.array4d[wp.vec4ub],
  # Out:
  frame_out: wp.array3d[wp.uint8],
) -> None:
  """Camera 0 of world 0 as a (height, width, 3) RGB uint8 image."""
  y, x = wp.tid()
  c = rgba[0, 0, y, x]
  for k in range(3):
    frame_out[y, x, k] = c[k]


class ViewerCamera(Protocol):
  """What the frames read of a Newton viewer camera (ViewerGL's `Camera`, a private class)."""

  @property
  def pos(self) -> Sequence[float]:
    """Position [m]."""
    ...

  @property
  def fov(self) -> float:
    """Vertical field of view [deg]."""
    ...

  @property
  def near(self) -> float:
    """Near clip distance [m]."""
    ...

  @property
  def far(self) -> float:
    """Far clip distance [m]."""
    ...

  def get_front(self) -> Sequence[float]:
    """Unit view direction."""
    ...

  def get_up(self) -> Sequence[float]:
    """Unit up direction."""
    ...


def camera_pose(camera: ViewerCamera) -> wp.transformf:
  """World pose of a Newton viewer camera (columns right, up, back), as a Renderer camera pose."""
  front = np.array([float(v) for v in camera.get_front()][:3])
  up = np.array([float(v) for v in camera.get_up()][:3])
  right = np.cross(front, up)
  right = right / max(float(np.sqrt(right @ right)), 1e-12)
  up = np.cross(right, front)
  rot = np.stack([right, up, -front], axis=1).astype(np.float32)
  pos = [float(v) for v in camera.pos][:3]
  return wp.transformf(wp.vec3f(*pos), wp.quat_from_matrix(wp.mat33f(*rot.ravel())))


@final
@dataclasses.dataclass(frozen=True)
class ShapeKinds:
  """Per shape what ViewerBase's visibility rule reads (flags, static, type, world)."""

  flags: np.ndarray[[int], np.dtype[np.int32]]
  static: np.ndarray[[int], np.dtype[np.bool_]]
  types: np.ndarray[[int], np.dtype[np.int32]]
  worlds: np.ndarray[[int], np.dtype[np.int32]]

  @staticmethod
  def of(model: newton.Model) -> ShapeKinds | None:
    """The model's shape kinds (None for a model without shape arrays)."""
    if model.shape_flags is None or model.shape_body is None or model.shape_type is None or model.shape_world is None:
      return None
    static = cast("np.ndarray[[int], np.dtype[np.bool_]]", host_copy(model.shape_body) < 0)
    return ShapeKinds(_ints(model.shape_flags), static, _ints(model.shape_type), _ints(model.shape_world))


def _ints(array: wp.array) -> np.ndarray[[int], np.dtype[np.int32]]:
  """A (shape_count,) Newton model array as int32 (Newton's model arrays are unshaped)."""
  return cast("np.ndarray[[int], np.dtype[np.int32]]", host_copy(array).astype(np.int32))


def visible_shapes(viewer: ViewerBase, kinds: ShapeKinds) -> frozenset[int]:
  """Shapes a Newton viewer shows: ViewerBase's rule, evaluated once per distinct shape kind.

  Show Visual / Collision / Ground / Static toggles, the visible worlds and the layer's
  visibility (ViewerBase internals: `_should_show_shape`, `_layer_force_hidden`, `_visible_worlds`).
  """
  if viewer._layer_force_hidden():  # noqa: SLF001  # ViewerBase's rule (private: the repository audit lists it)
    return frozenset()
  keys = np.stack([kinds.flags, kinds.static.astype(np.int64), kinds.types], axis=1)
  unique, inverse = np.unique(keys, axis=0, return_inverse=True)
  shown = np.array([viewer._should_show_shape(int(f), bool(st), int(t)) for f, st, t in unique], dtype=bool)  # noqa: SLF001
  mask = shown[inverse.ravel()]
  worlds = viewer._visible_worlds  # noqa: SLF001
  if worlds is not None:
    mask &= (kinds.worlds < 0) | np.isin(kinds.worlds, sorted(worlds))
  return frozenset(np.flatnonzero(mask).tolist())


def copy_frame(
  frame: wp.array[wp.uint8, [int, int, 3]], output: wp.array[wp.uint8, [int, int, int]] | None
) -> wp.array[wp.uint8, [int, int, int]]:
  """A viewer's `get_frame`: `frame` copied into `output` (a new array when None).

  Raises:
    ValueError: for an `output` of another shape.
  """
  if output is None:
    return wp.clone(frame)
  if tuple(output.shape) != tuple(frame.shape):
    raise ValueError(f"Shape of `output` must be {tuple(frame.shape)}, got {tuple(output.shape)}")
  wp.copy(output, frame)
  return output


@final
@dataclasses.dataclass
class _View:
  """A renderer of one resolution and the images it presents."""

  size: Size
  r: Renderer[NewtonState]
  frame: RgbImage


def _close(views: list[_View]) -> None:
  """Closes the renderers of dropped `views` (their OptiX build and compile threads: the
  garbage collector releases none of them), every one, then raises the failures together."""
  errors: list[Exception] = []
  for view in views:
    try:
      view.r.close()
    except Exception as error:  # noqa: BLE001  # every renderer closed first
      errors.append(error)
  if errors:
    raise ExceptionGroup("closing the dropped renderers failed", errors)


@final
class PbrFrames:
  """raywarp renders of a Newton model (all worlds in one scene) from a viewer camera.

  Attributes:
    frame: the last presented image (None before the first frame).
    serial: number of presented frames (one per `render` with a scene).
  """

  def __init__(  # noqa: PLR0913  # the viewer's PBR options, keyword-only
    self,
    lighting: Lighting | None,
    render_options: Mapping[str, object],
    *,
    owner: str,
    particles: str,
    device: wp.DeviceLike = None,
    usd: Stage | None = None,
    scene_options: SceneOptions = DEFAULT_SCENE_OPTIONS,
  ) -> None:
    """Validates the options (see `ViewerRaywarpHeadless`)."""
    self._options = render_options_of(render_options, _RESERVED_OPTIONS, owner)
    self._particles = particles
    self._kinds: ShapeKinds | None = None
    self._lighting = lighting
    self._usd = usd
    self._scene_options = scene_options
    self._device = None if device is None else wp.get_device(device)
    self._scene: GpuScene[NewtonState] | None = None
    self._look: Mapping[str, object] = {}
    self._shown: frozenset[int] | None = None
    self._mask: np.ndarray[[int], np.dtype[np.int32]] | None = None
    self._views: dict[Size, _View] = {}
    self._view: _View | None = None
    self.frame: RgbImage | None = None
    self.serial = 0

  @property
  def scene(self) -> GpuScene[NewtonState] | None:
    """The PBR scene (None without a model or visible shapes)."""
    return self._scene

  def set_model(self, model: newton.Model | None) -> None:
    """Builds the scene of `model` (every shape a toggle can show), or drops it (None); warns once
    per model that its particles, cloth and soft bodies are not ray traced."""
    views = list(self._views.values())
    self._scene, self._views, self._view, self._mask = None, {}, None, None
    _close(views)
    self.frame = None
    self._kinds = None if model is None else ShapeKinds.of(model)
    if model is None:
      return
    if model.particle_count > 0 or model.tri_count > 0:
      warnings.warn(f"raywarp.integrations.newton: particles, cloth and soft bodies are {self._particles}", stacklevel=3)
    self._scene, self._look = self._build(model)
    if self._scene.ninstance == 0:
      self._scene = None
    self.show(self._shown)

  def _build(self, model: newton.Model) -> tuple[GpuScene[NewtonState], Mapping[str, object]]:
    """The model's scene and its look's settings: the user's lighting, the stage's or ViewerGL's."""
    device = model.device if self._device is None else self._device
    # The viewer's render options are fixed: without emissive lights the upload skips the fit.
    scene_options = self._scene_options
    options = dataclasses.replace(scene_options, emissive_lights=scene_options.fit_emissive and emissive_lights(self._options))
    everything = range(model.shape_count)  # shown or hidden later by the visibility mask
    lighting = self._lighting
    user_lit = lighting is not None
    stage_lit = self._usd is not None and not user_lit and from_usd.stage_has_lighting(self._usd)
    # The display-relative lighting is calibrated for the exposure the viewer renders at.
    look: dict[str, object] = {} if lighting is None else {"exposure.mode": "manual", "exposure.ev100": lighting.ev100}
    if not user_lit and not stage_lit:
      # ViewerGL's look; a settings group the user gives wins over its fields.
      up_axis = int(newton.Axis.from_any(model.up_axis))
      gl = gl_look(up_axis)
      lighting = Lighting(gl.lights, envmap=gl.ambient.envmap(up_axis), envmap_intensity=1.0, ev100=gl.ev100)
      look = {k: v for k, v in gl.settings.items() if k.partition(".")[0] not in self._options}
      look |= {"exposure.mode": "manual", "exposure.ev100": gl.ev100}
    if self._usd is not None:
      # Particles, cloth and soft bodies are not ray traced (`set_model` warns).
      adapter = Options(merge_worlds=True, shapes=everything, particle_meshes=False)
      host = from_usd.load_usd(model, self._usd, options=options, adapter=adapter, lighting=lighting)
      bound = model_binding(host, model, adapter=adapter, stage=self._usd, device=device)
      return upload(host, device, binding=bound, options=options), look
    checker = None
    if not user_lit:
      # Planes in ViewerGL's checker (1 m squares).
      planes = np.flatnonzero(host_copy(model.shape_type) == int(newton.GeoType.PLANE)) if model.shape_type is not None else []
      checker = Checkerboard(frozenset(int(p) for p in planes), colors=GL_CHECKER_COLORS)
    adapter = Options(merge_worlds=True, shapes=everything, checkerboard=checker, particle_meshes=False)
    host = model_load(model, options=options, adapter=adapter, lighting=lighting)
    bound = model_binding(host, model, adapter=adapter, device=device)
    return upload(host, device, binding=bound, options=options), look

  def show(self, shapes: frozenset[int] | None) -> None:
    """Shows only the Newton shapes `shapes` (None: all); stage visuals stay unless it is empty."""
    self._shown = shapes
    if self._scene is None:
      return
    binding = self._scene.binding
    if not isinstance(binding, NewtonBinding):
      raise TypeError("PbrFrames needs a scene bound by raywarp.integrations.newton")
    shape0 = binding.shape0
    shown = np.ones(len(shape0), bool) if shapes is None else np.isin(shape0, sorted(shapes)) | ((shape0 < 0) & bool(shapes))
    mask = shown.astype(np.int32)
    if self._mask is not None and np.array_equal(mask, self._mask):
      return
    self._mask = mask
    for view in self._views.values():
      self._apply_mask(view.r)

  def _apply_mask(self, r: Renderer[NewtonState]) -> None:
    """Writes the visibility mask into a renderer (one small copy, on a change only)."""
    if self._mask is not None:
      r.per_world["instance.visible"].assign(self._mask[None])
      r.reset_history()

  def _settings(self) -> RenderSettings:
    """RenderSettings of the viewer's renderers: look, render options and the outputs it needs."""
    own = {"outputs": Outputs.of("rgb")}
    return RenderSettings().replace(**self._look).replace(**self._options).replace(**own)

  def _renderer(self, size: Size) -> _View:
    """The renderer of `size` (the renderers of other sizes closed)."""
    view = self._views.get(size)
    if view is not None:
      return view
    _close([self._views.pop(old) for old in [s for s in self._views if s != size]])
    if self._scene is None:
      raise RuntimeError("PbrFrames: no scene (set_model first)")
    device = self._scene.device
    per_world = PerWorld(camera=["fovy"], visible=["*"])
    r = Renderer(
      self._scene, nworld=1, cameras=[CameraSpec(fovy=45.0)], resolution=size, settings=self._settings(), per_world=per_world
    )
    w, h = size
    view = self._views[size] = _View(size, r, wp.zeros((h, w, 3), dtype=wp.uint8, device=device))
    self._apply_mask(r)
    return view

  def render_viewer(
    self, viewer: ViewerBase, state: newton.State | None, camera: ViewerCamera, *, size: Size, seed: int
  ) -> RgbImage | None:
    """A Newton viewer's frame of `state` (None before a state or a model): the shapes its
    visibility toggles show, at its world offsets and layer root."""
    if state is None or viewer.model is None:
      return None
    if self._kinds is not None:
      self.show(visible_shapes(viewer, self._kinds))
    # The (world_count,) vec3 world offsets (typed (int, 3) from their NumPy source).
    offsets = cast("wp.array[wp.vec3, [int]] | None", viewer.world_offsets)
    root = wp.transformf(*[float(v) for v in viewer.layer.xform])  # the active layer's transform
    return self.render(state, camera, size=size, world_offsets=offsets, root=root, seed=seed)

  def render(  # noqa: PLR0913  # one frame's inputs, keyword-only
    self,
    state: newton.State,
    camera: ViewerCamera,
    *,
    size: Size,
    world_offsets: wp.array[wp.vec3, [int]] | None,
    root: wp.transformf | None,
    seed: int,
  ) -> RgbImage | None:
    """Renders the scene from a Newton viewer camera at `size` (width, height); None for a model
    without visible shapes."""
    if self._scene is None:
      return None
    pose = camera_pose(camera)
    view = self._view = self._renderer((max(size[0], 1), max(size[1], 1)))
    r = view.r
    r.camera_pose.fill_(pose)
    fovy = r.per_world["camera.fovy"]
    if isinstance(fovy, InstanceColumns):
      raise TypeError("per_world['camera.fovy'] must be the (nworld, ncamera) table")
    fovy.fill_(float(camera.fov))
    r.set_clip(znear=float(camera.near), zfar=float(camera.far))  # as ViewerGL's projection
    r.render(StateView(state, world_offsets=world_offsets, root=root), seed=seed)
    self._present(view)
    return view.frame

  def _present(self, view: _View) -> None:
    """Copies the rendered frame into the presented image."""
    out = view.r.outputs
    if out.rgb is None:
      raise RuntimeError("PbrFrames: the renderer wrote no rgb (requested)")
    h, w = view.frame.shape[0], view.frame.shape[1]
    device = view.frame.device
    wp.launch(_present, dim=(h, w), inputs=[out.rgb], outputs=[view.frame], device=device)
    self.serial += 1
    self.frame = view.frame

  @property
  def renderer(self) -> Renderer[NewtonState] | None:
    """The renderer of the last frame."""
    return None if self._view is None else self._view.r

  @property
  def sizes(self) -> frozenset[Size]:
    """The resolutions of the renderers kept."""
    return frozenset(self._views)


def drop_scene(viewer: ViewerBase) -> None:
  """Drops a PBR viewer's scene (`clear_model`), also when the base viewer's constructor calls it
  before the viewer made its `PbrFrames`."""
  frames = vars(viewer).get("pbr")
  if isinstance(frames, PbrFrames):
    frames.set_model(None)
