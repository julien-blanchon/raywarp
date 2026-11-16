"""Drop-in replacement for `newton.sensors.SensorCamera`, rendered by raywarp.

Each view is one render world of a `rw.integrations.newton.scene` whose instances take the poses of
the view's Newton world (`StateView.world_indices`), so views can show any world (several views per
world, stereo pairs, ...). The view's camera is a free camera rendered from Newton's camera-space
ray bundle (raywarp's ray-bundle mode); outputs are converted to Newton's formats (`kernels.py`).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera as NewtonSensorCamera

from raywarp.core.pipeline import graph as render_graph
from raywarp.core.pipeline.upload.options import SceneOptions
from raywarp.core.trace.optix.graph import Segments
from raywarp.integrations.newton.api import scene as newton_scene
from raywarp.integrations.newton.options import emissive_lights
from raywarp.integrations.newton.options import render_options as render_options_of
from raywarp.integrations.newton.sensor.call import Call, Rays, UpdateImages, validate_call
from raywarp.integrations.newton.sensor.frame import (
  SensorScene,
  SensorViews,
  SettingsKey,
  convert_outputs,
  load_bundle,
  sensor_settings,
)
from raywarp.integrations.newton.types import (
  NEWTON_LIGHT_DIRECTION,
  Checkerboard,
  Hemisphere,
  NewtonState,
  Options,
  StateView,
  Vec3,
  default_lights,
  from_z_up,
)
from raywarp.scene.lighting.rig import Lighting
from raywarp.scene.model.lights import Light
from raywarp.settings import RenderSettings

if TYPE_CHECKING:
  from shape_extensions import IntVar

  from raywarp.core.pipeline.renderer import Renderer

ClearData = NewtonSensorCamera.ClearData
RenderConfig = NewtonSensorCamera.RenderConfig
TextureProjectionMode = NewtonSensorCamera.TextureProjectionMode
# RenderSettings fields the sensor sets itself (after the render options).
_RESERVED_OPTIONS: Final = ("outputs", "environment.background_color")


def _vec3(v: Sequence[float] | wp.vec3f) -> Vec3:
  """A Newton vector argument (a sequence or a Warp vec3) as three floats."""
  return (float(v[0]), float(v[1]), float(v[2]))


class SensorCamera:
  """Physically based drop-in replacement for :class:`newton.sensors.SensorCamera`.

  Same constructor, :meth:`update` signature and output formats as Newton's ray-traced camera
  sensor (1.7 and main): one view per camera transform, any of Newton's
  ``compute_camera_rays_*`` bundles (pinhole, OpenCV, fisheye; camera frame -Z forward, +Y up),
  ``world_indices`` with :class:`WorldRenderFlag`, :class:`ClearData` values where nothing is hit
  and the :class:`RenderConfig` fields that have an equivalent. Switch with one import::

      # not: from newton.sensors import SensorCamera
      from raywarp.integrations.newton.sensor import SensorCamera

  Rendering follows raywarp: PBR shading of the shape colors, mesh textures and scalar
  metallic / roughness, ray-traced shadows, optional image-based lighting (``lighting=``).
  Only the images passed to :meth:`update` are rendered (each a compile-time output of the
  renderer), into the caller's arrays where the layouts match (color, depth or forward depth,
  shape index, HDR, normal); an in-place pass converts what differs from Newton's values.
  :meth:`update` makes no host synchronisation or allocation after its first call with the same
  arrays and settings, so it can be captured in a CUDA graph: :meth:`capture` (also with OptiX
  traversal). Differences to Newton's sensor:

  * Lights: without :meth:`create_default_light` or ``lighting``, the scene is lit by
    :func:`raywarp.integrations.newton.default_lights` (Newton's sensor then only has ambient
    light); no hemispheric ambient unless :meth:`set_ambient_light` is called. Lights and ambient
    follow ``model.up_axis``.
  * Worlds may hold different shapes (a different object per world): the scene pools them and
    each world shows its own (instances are built once and posed per view); shapes of opacity 0
    are not rendered (Newton main's rule). A shape whose ``shape_color`` differs between worlds
    shows each world's color, read from ``model.shape_color`` at every update (opacity: that of
    the first world showing the shape).
  * Height fields render as Newton's watertight mesh of them; the model's triangles over particles
    (cloth, soft-body skins) render as deformable surfaces read from ``state.particle_q`` at every
    update (cloth thickened by the particle radius); loose particles and Gaussians are not rendered.
  * ``RenderConfig``: ``enable_textures``, ``enable_shadows``, ``enable_ambient_lighting``,
    ``enable_backface_culling``, ``output_color_space``, ``anti_aliasing`` and
    ``max_distance`` map to raywarp settings (fast math is always on: ``enable_fast_math`` is
    ignored; ``max_distance`` must be > 0, ``inf`` unbounded, where Newton renders nothing at 0;
    geometry nearer than the camera's near plane is clipped, Newton has no near clip);
    ``block_dim`` and ``render_order`` are accepted but not used (raywarp tunes its launch
    shape; its ``performance.render_order`` setting chooses the order, default tiled);
    ``enable_global_world=False`` and ``TextureProjectionMode.TRIPLANAR`` are not supported
    (warned once); tile sizes, particle and Gaussian settings are ignored.
  * Multisample bundles (``sample_count > 1``, SSAA or MSAA): each pixel's ray is the mean of
    its samples and anti-aliasing uses ``ceil(sqrt(sample_count))²`` stratified samples
    interpolated between neighbouring pixel rays (raywarp's supersampling). Depth, shape index,
    normal and albedo come from the first sample.
  * ``hdr_color_image`` is exposed linear radiance (before tone mapping); ``color_image`` is
    tone mapped (raywarp's ``tone_mapper``) and sRGB encoded.
  * Misses show the environment map in ``color_image`` / ``hdr_color_image`` when one is set
    with ``envmap_background`` (default), else the clear color.

  Extra keyword-only constructor options: ``lighting`` (``rw.Lighting``: lights, ambient,
  environment, headlight; its ``ev100`` is the sensor's ``exposure.ev100`` unless a render option
  sets it) and :meth:`raywarp.RenderSettings.replace` keys: groups
  (``shadows=rw.settings.Shadows(samples=4)``, replacing the whole group) or ``"group.field"`` keys
  (``**{"aa.spp": 2, "performance.traversal": "optix"}``), which take precedence over the
  :class:`RenderConfig` mapping.
  """

  AntiAliasing = NewtonSensorCamera.AntiAliasing
  ClearData = NewtonSensorCamera.ClearData
  GaussianRenderMode = NewtonSensorCamera.GaussianRenderMode
  RenderConfig = NewtonSensorCamera.RenderConfig
  RenderOrder = NewtonSensorCamera.RenderOrder
  TextureProjectionMode = NewtonSensorCamera.TextureProjectionMode
  Utils = NewtonSensorCamera.Utils
  WorldRenderFlag = NewtonSensorCamera.WorldRenderFlag

  # Newton's ray-bundle helpers (camera-space rays, (height, width, sample_count, 2) vec3).
  compute_camera_rays_pinhole = staticmethod(NewtonSensorCamera.compute_camera_rays_pinhole)
  compute_camera_rays_pinhole_usd = staticmethod(NewtonSensorCamera.compute_camera_rays_pinhole_usd)
  compute_camera_rays_pinhole_opencv = staticmethod(NewtonSensorCamera.compute_camera_rays_pinhole_opencv)
  compute_camera_rays_fisheye_opencv = staticmethod(NewtonSensorCamera.compute_camera_rays_fisheye_opencv)
  compute_camera_rays_fisheye_ftheta = staticmethod(NewtonSensorCamera.compute_camera_rays_fisheye_ftheta)
  compute_camera_rays_fisheye_kannala_brandt = staticmethod(NewtonSensorCamera.compute_camera_rays_fisheye_kannala_brandt)

  def __init__(
    self,
    model: newton.Model,
    *,
    default_clear_data: ClearData | None = None,
    default_render_config: RenderConfig | None = None,
    load_textures: bool = True,
    lighting: Lighting | None = None,
    **render_options: object,
  ) -> None:
    """Construct a camera sensor for a model.

    Args:
      model: Newton simulation model to render.
      default_clear_data: Clear values used by :meth:`update` when its ``clear_data`` argument
        is ``None``. Defaults to ``ClearData()``.
      default_render_config: Render settings used by :meth:`update` when its
        ``render_config`` argument is ``None``. Defaults to ``RenderConfig()``.
      load_textures: Load mesh textures (shown with ``RenderConfig.enable_textures``).
      lighting: Lights, ambient, environment and headlight of the scene (Newton models carry
        none); None: :func:`raywarp.integrations.newton.default_lights` (or
        :meth:`create_default_light`).
      **render_options: `RenderSettings.replace` keys (not ``outputs`` and
        ``environment.background_color``, which the sensor sets: misses take the clear color).

    Raises:
      TypeError: for a render option the sensor sets itself, or an unknown one.
    """
    options = render_options_of(render_options, _RESERVED_OPTIONS, "SensorCamera")
    self.default_clear_data: ClearData = default_clear_data if default_clear_data is not None else ClearData()
    """Clear values used by :meth:`update` when its ``clear_data`` argument is ``None``."""
    self.default_render_config: RenderConfig = default_render_config if default_render_config is not None else RenderConfig()
    """Render settings used by :meth:`update` when its ``render_config`` argument is ``None``."""
    self.model = model
    self._up_axis = int(newton.Axis.from_any(model.up_axis))
    self._load_textures = load_textures
    self._lighting = Lighting(default_lights(self._up_axis)) if lighting is None else lighting
    # The display-relative lighting (the user's or the default rig) is calibrated for the exposure
    # the sensor renders at (an `exposure` group the user gives wins).
    manual = {} if "exposure" in options else {"exposure.mode": "manual", "exposure.ev100": self._lighting.ev100}
    self._options = {**manual, **options}
    self._ambient: Hemisphere | None = None
    self._checkerboard: Checkerboard | None = None
    self._scene: SensorScene | None = None
    self._settings_cache: dict[SettingsKey, RenderSettings] = {}
    self._warned: set[str] = set()
    self._renderer: Renderer[NewtonState] | None = None
    gaussians = model.gaussians_data
    if gaussians is not None and len(gaussians) > 0:
      warnings.warn("raywarp.integrations.newton.sensor.SensorCamera does not render Gaussians", stacklevel=2)
    # Fails early for models whose worlds differ.
    self._get_scene()

  @property
  def device(self) -> wp.Device:
    """Device of the model this sensor renders."""
    return self.model.device

  @property
  def renderer(self) -> Renderer[NewtonState] | None:
    """The raywarp Renderer of the last `update` (its settings, outputs and per-world tables;
    one per output set and resolution), None before the first."""
    return self._renderer

  def allocate_image[D](self, view_count: int, width: int, height: int, dtype: type[D]) -> wp.array[D, [int, int, int]]:
    """Allocate a zeroed (view_count, height, width) output image array on the model device."""
    return wp.zeros((view_count, height, width), dtype=dtype, device=self.device)

  def allocate_image_color(self, view_count: int, width: int, height: int) -> wp.array[wp.uint32, [int, int, int]]:
    """Allocate an RGBA color output array (packed ``uint32``)."""
    return self.allocate_image(view_count, width, height, wp.uint32)

  def allocate_image_depth(self, view_count: int, width: int, height: int) -> wp.array[wp.float32, [int, int, int]]:
    """Allocate a ray-distance depth output array [m]."""
    return self.allocate_image(view_count, width, height, wp.float32)

  def allocate_image_depth_forward(self, view_count: int, width: int, height: int) -> wp.array[wp.float32, [int, int, int]]:
    """Allocate a forward (planar) depth output array [m]."""
    return self.allocate_image_depth(view_count, width, height)

  def allocate_image_shape_index(self, view_count: int, width: int, height: int) -> wp.array[wp.uint32, [int, int, int]]:
    """Allocate a shape-index output array."""
    return self.allocate_image(view_count, width, height, wp.uint32)

  def allocate_image_normal(self, view_count: int, width: int, height: int) -> wp.array[wp.vec3f, [int, int, int]]:
    """Allocate a world-space surface-normal output array (``vec3f``)."""
    return self.allocate_image(view_count, width, height, wp.vec3f)

  def allocate_image_albedo(self, view_count: int, width: int, height: int) -> wp.array[wp.uint32, [int, int, int]]:
    """Allocate an RGBA albedo output array (packed ``uint32``)."""
    return self.allocate_image(view_count, width, height, wp.uint32)

  def allocate_image_color_hdr(self, view_count: int, width: int, height: int) -> wp.array[wp.vec3f, [int, int, int]]:
    """Allocate a linear HDR color output array (``vec3f``)."""
    return self.allocate_image(view_count, width, height, wp.vec3f)

  def compute_camera_transforms_usd(
    self, cameras: Sequence[object], *, time: float | None = None, xform: wp.transformf | None = None
  ) -> wp.array[wp.transformf, [int]]:
    """Read world-space camera transforms from one USD camera per view (Newton's implementation)."""
    # Newton's private implementation, imported where it is used (the repository audit lists it).
    from newton._src.core import Axis  # noqa: PLC0415
    from newton._src.sensors.sensor_camera_render import camera_utils  # noqa: PLC0415

    return camera_utils.compute_camera_transforms_usd(
      cameras, device=self.model.device, target_up_axis=Axis(self._up_axis), time=time, xform=xform
    )

  def create_default_light(
    self,
    *,
    enable_shadows: bool = True,
    direction: wp.vec3f | None = None,
    color: Sequence[float] | wp.vec3f | None = None,
  ) -> None:
    """Replace the lights with one directional light, as Newton's sensor does.

    Args:
      enable_shadows: Cast shadows (rendered when ``RenderConfig.enable_shadows`` is set too).
      direction: Normalized direction the light travels; defaults to normalized ``(-1, 1, -1)``
        for a Z-up model, rotated to the model's up axis otherwise (the light points down).
      color: Linear RGB color of the light (default white).
    """
    d = from_z_up(NEWTON_LIGHT_DIRECTION, self._up_axis) if direction is None else _vec3(direction)
    rgb = (1.0, 1.0, 1.0) if color is None else _vec3(color)
    light = Light.sun(d, color=rgb, shadows=enable_shadows)
    self._lighting = dataclasses.replace(self._lighting, lights=[light])
    self._drop_scene()

  def set_ambient_light(
    self, sky_color: Sequence[float] | wp.vec3f, ground_color: Sequence[float] | wp.vec3f | None = None
  ) -> None:
    """Set the hemispheric ambient light used when ``RenderConfig.enable_ambient_lighting`` is set.

    Surfaces facing the model's up axis receive ``sky_color``, surfaces facing away
    ``ground_color``, others a blend: a display-relative environment of that radiance (it
    replaces the ``lighting`` environment; it is not shown as the background).

    Args:
      sky_color: Linear RGB ambient radiance from above.
      ground_color: Linear RGB ambient radiance from below (default: ``sky_color``).
    """
    sky = _vec3(sky_color)
    self._ambient = Hemisphere(sky, sky if ground_color is None else _vec3(ground_color))
    self._drop_scene()

  def assign_checkerboard_material(
    self,
    *,
    shape_indices: Sequence[int] | np.ndarray[[int], np.dtype[np.integer]],
    resolution: int = 64,
    checker_size: int = 32,
  ) -> None:
    """Assign Newton's grey checkerboard texture to selected shapes (multiplying their color).

    The texture is mapped with MuJoCo's texture coordinates (mesh UVs, primitive mapping), not
    Newton's cubic projection. Shown with ``RenderConfig.enable_textures``.

    Raises:
      ValueError: for a shape index out of range.
    """
    indices = np.asarray(shape_indices, dtype=np.int64).reshape(-1)
    if np.any((indices < 0) | (indices >= self.model.shape_count)):
      raise ValueError("shape_indices contains an out-of-range shape index")
    self._checkerboard = Checkerboard(frozenset(indices.tolist()), resolution, checker_size)
    self._drop_scene()

  def capture(self, step: Callable[[], None]) -> Segments:
    """Captures `step()` (e.g. physics + :meth:`update`) in a CUDA graph, after one eager run.

    Returns an object whose ``launch()`` replays it on the current stream without synchronising
    (with OptiX traversal the graph is cut around the OptiX calls, see
    ``raywarp.core.trace.optix.graph``; a plain ``wp.ScopedCapture`` cannot record them). Capture
    again after changing the outputs, settings or arrays passed to :meth:`update`, the lights,
    the ambient light or the checkerboard materials (a replay then raises RuntimeError), or after
    :meth:`close`.

    Raises:
      RuntimeError: on a device without CUDA graphs (CPU).
    """
    segments = render_graph.capture(self.device, step)
    scene = self._scene
    # The renderers the graph may have used, with their versions (`Renderer.debug.graph_version`).
    captured = [] if scene is None else [(v.r, v.r.debug.graph_version()) for v in scene.renderers.values()]

    def check() -> None:
      """Refuses a replay over a rebuilt scene or a changed renderer (released or stale buffers)."""
      if self._scene is not scene or any(r.debug.graph_version() != version for r, version in captured):
        raise RuntimeError(
          "SensorCamera: the scene or a renderer changed since the capture (lights, ambient, checkerboard, outputs, "
          "settings or close): capture again"
        )

    segments.checks.append(check)
    return segments

  def close(self) -> None:
    """Releases the renderers (their OptiX structures and programs); a later
    :meth:`update` builds the scene again. Graphs captured before then refuse to replay."""
    self._drop_scene()

  def _drop_scene(self) -> None:
    """Drops the scene (rebuilt by the next update) and closes its renderers."""
    scene, self._scene, self._renderer = self._scene, None, None
    if scene is not None:
      scene.close()

  def sync_deformable_meshes(self, state: newton.State) -> None:
    """No-op: every update reads the particle meshes from its state (Newton's sensor API)."""

  def _warn_once(self, key: str, message: str) -> None:
    """Warns `message` the first time `key` is seen."""
    if key not in self._warned:
      self._warned.add(key)
      warnings.warn(message, stacklevel=3)

  def _get_scene(self) -> SensorScene:
    """The GPU scene (rebuilt after a light, ambient or material change) and its renderers."""
    if self._scene is None:
      lighting = self._lighting
      if self._ambient is not None:
        # The hemispheric ambient, display-relative like Newton's ambient term.
        hemisphere = self._ambient.envmap(self._up_axis)
        lighting = dataclasses.replace(lighting, envmap=hemisphere, envmap_intensity=None, envmap_rotation=0.0)
      # The render options are fixed (RenderConfig never maps `lighting`): without emissive
      # lights no call reads the panel lights, so the upload skips their fit.
      adapter = Options(checkerboard=self._checkerboard, load_textures=self._load_textures)
      options = SceneOptions(emissive_lights=emissive_lights(self._options))
      scene = newton_scene(self.model, options=options, adapter=adapter, lighting=lighting, device=self.model.device)
      self._scene = SensorScene.of(scene, ambient_background=self._ambient is None)
    return self._scene

  def _apply_config(self, views: SensorViews, config: RenderConfig) -> None:
    """Sets the runtime settings of `config` (far clip, ambient light) on the renderer."""
    far = float(config.max_distance)
    if not far > 0.0:
      # Newton's rays then hit nothing; raywarp's zfar 0 is "unbounded": no silent opposite.
      raise ValueError(f"RenderConfig.max_distance must be > 0 (inf: unbounded), got {config.max_distance}")
    views.r.set_clip(zfar=0.0 if far == math.inf else far)  # Newton's inf: unbounded (zfar 0)
    views.light_ambient(lit=config.enable_ambient_lighting)
    if not config.enable_global_world:
      self._warn_once(
        "global", "raywarp.integrations.newton.sensor.SensorCamera: RenderConfig.enable_global_world=False is not supported"
      )
    if int(config.texture_projection_mode) != int(TextureProjectionMode.CUBIC):
      self._warn_once(
        "projection",
        "raywarp.integrations.newton.sensor.SensorCamera: only MuJoCo's texture mapping is supported (no TRIPLANAR)",
      )

  def update[NV: IntVar, NH: IntVar, NW: IntVar](  # noqa: PLR0913  # Newton's signature
    self,
    state: newton.State,
    camera_transforms: wp.array[wp.transformf, [NV]],
    camera_rays: Rays,
    *,
    color_image: wp.array[wp.uint32, [NV, NH, NW]] | None = None,
    depth_image: wp.array[wp.float32, [NV, NH, NW]] | None = None,
    forward_depth_image: wp.array[wp.float32, [NV, NH, NW]] | None = None,
    shape_index_image: wp.array[wp.uint32, [NV, NH, NW]] | None = None,
    normal_image: wp.array[wp.vec3f, [NV, NH, NW]] | None = None,
    albedo_image: wp.array[wp.uint32, [NV, NH, NW]] | None = None,
    hdr_color_image: wp.array[wp.vec3f, [NV, NH, NW]] | None = None,
    world_indices: wp.array[wp.int32, [NV]] | None = None,
    clear_data: ClearData | None = None,
    render_config: RenderConfig | None = None,
    sync_deformables: bool = True,
    seed: int = 0,
  ) -> None:
    """Render this camera sensor (see :meth:`newton.sensors.SensorCamera.update`).

    The number of views is ``camera_transforms.shape[0]``; all non-``None`` outputs have shape
    ``(view_count, height, width)`` with height, width from ``camera_rays``. Unlike Newton's
    sensor, no BVH refit of the model is needed (the sensor refits its own).

    Args:
      state: Simulation state with body transforms.
      camera_transforms: World-space camera transform per view, shape ``(view_count,)``.
      camera_rays: Camera-space ray origins ``[..., 0]`` and directions ``[..., 1]``, shape
        ``(height, width, sample_count, 2)`` (Newton's ``compute_camera_rays_*``).
      color_image: Output RGBA color (packed ``uint32``, Newton's byte order).
      depth_image: Output ray-distance depth from each ray origin [m].
      forward_depth_image: Output depth along the camera's forward axis from its origin [m].
      shape_index_image: Output shape index.
      normal_image: Output world-space shading normals.
      albedo_image: Output albedo (packed ``uint32``): the surface's base color.
      hdr_color_image: Output exposed linear HDR color (before tone mapping).
      world_indices: Optional per-view world, shape ``(view_count,)`` int32; defaults to view
        ``i`` rendering world ``i``. ``WorldRenderFlag.DISABLE_PRESERVE`` leaves a view's outputs
        unchanged; any other value outside ``[0, model.world_count)`` clears them.
      clear_data: Clear values for this call (default :attr:`default_clear_data`).
      render_config: Render settings for this call (default :attr:`default_render_config`).
      sync_deformables: Newton's flag; the particle meshes are always read from `state`.
      seed: Random seed of stochastic effects (soft shadows, path tracing).

    Raises:
      ValueError, TypeError, RuntimeError: for arrays of the wrong shape, dtype or device, as
        Newton's sensor.
    """
    del sync_deformables
    images = UpdateImages(
      color_image, depth_image, forward_depth_image, shape_index_image, normal_image, albedo_image, hdr_color_image
    )
    config = render_config if render_config is not None else self.default_render_config
    clear = clear_data if clear_data is not None else self.default_clear_data
    call = validate_call(
      self.model,
      camera_transforms=camera_transforms,
      camera_rays=camera_rays,
      images=images,
      world_indices=world_indices,
      config=config,
      clear=clear,
    )
    if call is not None:
      self._render(state, call, seed)

  def _render(self, state: newton.State, call: Call, seed: int) -> None:
    """Renders a validated call into the caller's arrays, then converts them to Newton's values."""
    scene = self._get_scene()
    settings = sensor_settings(call, self._options, env_background=scene.ambient_background, cache=self._settings_cache)
    views = scene.views(call.view_count, call.width, call.height, settings, call.images.arrays())
    self._apply_config(views, call.config)
    r = self._renderer = views.r
    load_bundle(views, call.camera_rays)
    active = views.active_views(call.world_indices, call.world_count)
    if call.images.shape is not None:
      views.write_shapes(scene.shapes, call.world_indices, scene.no_world)
    wp.copy(r.camera_pose.flatten(), call.camera_transforms)
    r.render(StateView(state, world_indices=call.world_indices), seed=seed, active=active)
    convert_outputs(scene, views, call)
