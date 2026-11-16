"""The steps of one `SensorCamera.update` after validation (`call.py`): settings, renderer,
bundle, conversions.

Typed intermediate results (`UpdateImages`, `Call`, `SensorScene`, `SensorViews`); after the first
call with the same arrays and settings nothing here allocates or synchronises.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast, final

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera as NewtonSensorCamera
from newton.utils import ColorSpace

from raywarp.core.pipeline.gpu_scene.gpu_scene import CameraSpec, GpuScene
from raywarp.core.pipeline.renderer import Renderer
from raywarp.core.pipeline.worlds.sharing import PerWorld
from raywarp.core.pipeline.worlds.tables import InstanceColumns
from raywarp.integrations.newton.binding import NewtonBinding
from raywarp.integrations.newton.sensor import kernels as sk
from raywarp.integrations.newton.sensor.call import Call, Image, NewtonImage, Rays
from raywarp.integrations.newton.types import NewtonState
from raywarp.settings import Outputs, RenderSettings
from raywarp.settings.render import mapped

if TYPE_CHECKING:
  from raywarp._base.dims import AnyDeviceArray

ClearData = NewtonSensorCamera.ClearData
RenderConfig = NewtonSensorCamera.RenderConfig
# Placeholder vertical field of view of the scene's camera (rays replace its projection).
_FOVY: Final = 60.0


def _flag(value: object) -> bool:
  """A RenderConfig flag as a settings bool (Newton types its flags bool but takes truthy ints)."""
  return bool(value)


def _mapped(config: RenderConfig, samples: int, *, env_background: bool) -> dict[str, object]:
  """The RenderConfig fields as RenderSettings keys."""
  fields: dict[str, object] = {
    "surfaces.textures": "auto" if config.enable_textures else False,
    "surfaces.backface_culling": _flag(config.enable_backface_culling),
    "shadows.enabled": _flag(config.enable_shadows),
    "aa.spp": max(1, math.ceil(math.sqrt(samples) - 1e-9)),
    # Newton's sensor uses direct lights and hemispheric ambient per world. A shared GI bake is an
    # optional extension, not part of that ambient term.
    "gi.enabled": False,
  }
  if not env_background:
    fields["environment.background"] = False  # the hemispheric ambient is not a background
  return fields


# Newton's `ClearData.clear_shape_index` of "no shape" (its default: nothing cleared to an id).
NO_SHAPE: Final = 0xFFFFFFFF

type SettingsKey = tuple[bool, bool, bool, int, Outputs, tuple[float, ...], bool]


def sensor_settings(
  call: Call, options: Mapping[str, object], *, env_background: bool, cache: dict[SettingsKey, RenderSettings]
) -> RenderSettings:
  """A call's RenderSettings (cached): its RenderConfig and outputs, then the render options."""
  config = call.config
  clear = call.clear.clear_color
  background = tuple((clear >> s & 0xFF) / 255.0 for s in (0, 8, 16))
  outputs = call.images.outputs()
  key: SettingsKey = (
    config.enable_textures,
    config.enable_backface_culling,
    config.enable_shadows,
    call.sample_count,
    outputs,
    background,
    env_background,
  )
  settings = cache.get(key)
  if settings is None:
    # The sensor's own fields last: misses show the clear color (exact for its rgb; the output
    # pass fixes the rest).
    own = {"outputs": outputs, "environment.background_color": background}
    fields = _mapped(config, call.sample_count, env_background=env_background)
    settings = cache[key] = mapped(RenderSettings(), fields, options).replace(**own)
  return settings


@final
@dataclasses.dataclass
class SensorViews:
  """Renderer of view_count views at one resolution and its per-frame buffers.

  Attributes:
    r: the renderer (one render world per view, one ray-bundle camera).
    bundle: the pixel rays and depth scale of the current bundle (`kernels.load_rays`).
    active: the views the renderer renders, -1 for those it skips (`Renderer.render(active=...)`,
      written from world_indices on the GPU).
    segments: (view_count, ninstance) Newton shape of each instance in each view's world: the
      renderer's segment table (`write_shapes`), so "seg" is Newton's shape index output.
    exposure, ambient, env_intensity: the renderer's per-world values; `*_on` their lit values.
    lit: whether the ambient light is on.
  """

  r: Renderer[NewtonState]
  bundle: sk.Bundle
  active: wp.array[wp.int32, [int]]
  segments: wp.array[wp.int32, [int, int]]
  exposure: wp.array[float, [int]]
  ambient: wp.array[wp.vec3, [int]]
  ambient_on: wp.array[wp.vec3, [int]]
  env_intensity: wp.array[float, [int]]
  env_intensity_on: wp.array[float, [int]]
  lit: bool = True

  @staticmethod
  def create(  # noqa: PLR0913  # the renderer's inputs, keyword-only
    scene: GpuScene[NewtonState],
    *,
    view_count: int,
    width: int,
    height: int,
    settings: RenderSettings,
    arrays: dict[str, NewtonImage],
  ) -> SensorViews:
    """A renderer of `view_count` ray-bundle views rendering into `arrays`."""
    rays = np.zeros((height, width, 2, 3), dtype=np.float32)
    rays[..., 1, 2] = -1.0
    r = Renderer(
      scene,
      nworld=view_count,
      cameras=[CameraSpec(fovy=_FOVY, rays=rays)],
      resolution=(width, height),
      settings=settings,
      output_arrays=arrays,
      per_world=PerWorld(env=["exposure", "ambient", "intensity"]),
    )
    npix, device = width * height, scene.device
    bundle = sk.Bundle()
    bundle.scale, bundle.coverage = wp.zeros(npix, dtype=float, device=device), wp.zeros(npix, dtype=float, device=device)
    bundle.flags = wp.zeros(1, dtype=int, device=device)
    ambient, intensity = _world_table(r, "ambient"), _world_table(r, "env.intensity")
    segments = wp.full((view_count, scene.ninstance), -1, dtype=wp.int32, device=device)
    if scene.ninstance:
      r.set_segment_table(segments)
    return SensorViews(
      r,
      bundle,
      wp.zeros(view_count, dtype=wp.int32, device=device),
      segments,
      _world_table(r, "exposure"),
      ambient,
      wp.clone(ambient),
      intensity,
      wp.clone(intensity),
    )

  def light_ambient(self, *, lit: bool) -> None:
    """Switches the ambient light (constant ambient and environment) on or off.

    Only on a change: a device copy (graph capturable), no host round trip.
    """
    if lit == self.lit:
      return
    if lit:
      wp.copy(self.ambient, self.ambient_on)
      wp.copy(self.env_intensity, self.env_intensity_on)
    else:
      self.ambient.zero_()
      self.env_intensity.zero_()
    self.lit = lit

  def active_views(self, world_indices: wp.array[wp.int32, [int]] | None, world_count: int) -> wp.array[wp.int32, [int]] | None:
    """The render's active views (`Renderer.render(active=...)`): those of `world_indices` that
    show a valid world (None without it: every view)."""
    if world_indices is None:
      return None
    active = self.active  # (view_count,): world_indices' shape
    wp.launch(sk.view_active, dim=active.shape[0], inputs=[world_indices, world_count], outputs=[active], device=active.device)
    return active

  def write_shapes(
    self, shapes: sk.Shapes, world_indices: wp.array[wp.int32, [int]] | None, no_world: wp.array[wp.int32, [int]]
  ) -> None:
    """The segment table of a call: each view's Newton shapes (its world: `world_indices[v]`,
    else v); device work only."""
    seg = self.segments
    if seg.shape[1]:
      world = world_indices if world_indices is not None else no_world
      inputs = [world, world_indices is not None, shapes]
      wp.launch(sk.view_shapes, dim=seg.shape, inputs=inputs, outputs=[seg], device=seg.device)


def _world_table(r: Renderer[NewtonState], field: str) -> AnyDeviceArray:
  """The renderer's (nworld, ...) per-world table of a declared non-instance `field`."""
  table = r.per_world[field]
  if isinstance(table, InstanceColumns):
    raise TypeError(f"per_world[{field!r}] is an instance field, not a per-world table")
  return table


def load_bundle(views: SensorViews, camera_rays: Rays) -> None:
  """Writes the call's pixel rays into the renderer's ray-bundle camera."""
  b = views.bundle
  rays = views.r.camera_rays
  if rays is None:
    raise RuntimeError("raywarp.integrations.newton.sensor.SensorCamera: the renderer lost its camera rays")
  b.origin, b.direction = rays[0], rays[1]
  b.flags.zero_()
  height, width = int(camera_rays.shape[0]), int(camera_rays.shape[1])
  wp.launch(
    sk.load_rays,
    dim=(height, width),
    inputs=[camera_rays],
    outputs=[b.origin, b.direction, b.scale, b.coverage, b.flags],
    device=camera_rays.device,
  )


@final
@dataclasses.dataclass
class SensorScene:
  """GPU scene of the Newton model (rebuilt when lights or materials change) and its renderers.

  Attributes:
    scene: the GPU scene.
    shapes: Newton shape per instance and world (shape index output).
    ambient_background: misses may show the environment (False for the hemispheric ambient).
    no_world: (1,) int32 placeholder for an absent world-index array.
    renderers: renderer per (view count, width, height).
    placeholders: per output dtype, the (1, 1, 1) array bound to an output not asked for.
  """

  scene: GpuScene[NewtonState]
  shapes: sk.Shapes
  ambient_background: bool
  no_world: wp.array[wp.int32, [int]]
  renderers: dict[tuple[int, int, int], SensorViews] = dataclasses.field(default_factory=dict)
  placeholders: dict[type, Image[object]] = dataclasses.field(default_factory=dict)

  @staticmethod
  def of(scene: GpuScene[NewtonState], *, ambient_background: bool) -> SensorScene:
    """The sensor state of a Newton GpuScene."""
    binding = scene.binding
    if not isinstance(binding, NewtonBinding):
      raise TypeError("SensorCamera needs a scene bound by raywarp.integrations.newton")
    shape0, col, rows = binding.shape_table()
    device = scene.device
    shapes = sk.Shapes()
    shapes.shape0 = wp.array(shape0 if len(shape0) else np.zeros(1, np.int32), dtype=wp.int32, device=device)
    shapes.column = wp.array(col if len(col) else np.zeros(1, np.int32), dtype=wp.int32, device=device)
    row_table = np.ascontiguousarray(rows) if rows.shape[1] else np.full((1, 1), -1, np.int32)
    shapes.rows = wp.array(row_table, dtype=wp.int32, device=device)
    shapes.ninstance = scene.ninstance
    return SensorScene(scene, shapes, ambient_background, wp.zeros(1, dtype=wp.int32, device=device))

  def views(
    self, view_count: int, width: int, height: int, settings: RenderSettings, arrays: dict[str, NewtonImage]
  ) -> SensorViews:
    """The renderer of this view count and resolution, with `settings`, rendering into `arrays`."""
    key = (view_count, width, height)
    views = self.renderers.get(key)
    if views is None:
      views = self.renderers[key] = SensorViews.create(
        self.scene, view_count=view_count, width=width, height=height, settings=settings, arrays=arrays
      )
      return views
    r = views.r
    if r.settings is not settings and r.settings != settings:
      r.update_settings(**r.settings.diff(settings))
    r.bind_outputs(**arrays)
    return views

  def close(self) -> None:
    """Closes every renderer (their OptiX build and compile threads: the garbage
    collector releases none of them), then raises the failures together."""
    views, self.renderers = list(self.renderers.values()), {}
    errors: list[Exception] = []
    for v in views:
      try:
        v.r.close()
      except Exception as error:  # noqa: BLE001  # every renderer closed first
        errors.append(error)
    if errors:
      raise ExceptionGroup("closing the sensor's renderers failed", errors)

  def placeholder[D](self, dtype: type[D]) -> wp.array[D, [int, int, int]]:
    """A (1, 1, 1) array bound to the outputs an update is not asked for."""
    if dtype not in self.placeholders:
      self.placeholders[dtype] = wp.empty((1, 1, 1), dtype=dtype, device=self.scene.device)
    return cast("Image[D]", self.placeholders[dtype])  # keyed by its dtype


def _srgb_packed_to_linear(packed: int) -> int:
  """Newton's conversion of packed sRGB clear colors for ColorSpace.LINEAR outputs."""
  r, g, b, a = (packed & 0xFF), (packed >> 8) & 0xFF, (packed >> 16) & 0xFF, (packed >> 24) & 0xFF
  lr, lg, lb = (min(max(int(c * 255.0), 0), 255) for c in newton.utils.color_srgb_to_linear((r / 255.0, g / 255.0, b / 255.0)))
  return (a << 24) | (lb << 16) | (lg << 8) | lr


def _clear(clear: ClearData, *, linear: bool) -> sk.Clear:
  """Newton's clear values in the output formats."""
  out = sk.Clear()
  color, albedo = clear.clear_color, clear.clear_albedo
  if linear:
    color, albedo = _srgb_packed_to_linear(color), _srgb_packed_to_linear(albedo)
  out.color, out.albedo = wp.uint32(color), wp.uint32(albedo)
  out.depth = float(clear.clear_depth)
  out.shape = wp.uint32(clear.clear_shape_index)
  out.normal = wp.vec3(*clear.clear_normal)
  return out


def _wanted(call: Call, clear: sk.Clear, *, linear: bool, env_background: bool) -> sk.OutputSwitches:
  """Which outputs the conversion writes, and whether any pixel differs from the renderer's."""
  im, c = call.images, call.clear
  want = sk.OutputSwitches()
  want.color, want.depth, want.forward = im.color is not None, im.depth is not None, im.forward is not None
  want.forward_in_depth = want.forward and not want.depth
  want.shape, want.normal = im.shape is not None, im.normal is not None
  want.albedo, want.hdr = im.albedo is not None, im.hdr is not None
  want.shaded = want.color or want.hdr or want.normal or want.albedo
  want.multisample, want.background, want.linear = call.sample_count > 1, env_background, linear
  # Everything but plain color (in sRGB, no background, clear color 0), shape indices (written by
  # the megakernel: `sk.view_shapes`) with the default clear index, plain normals (clear
  # normal 0) and plain depth (clear depth 0) needs a pass over the pixels.
  want.pixel_pass = (
    (want.shape and c.clear_shape_index != NO_SHAPE)
    or want.albedo
    or want.hdr
    or want.forward
    or (want.normal and tuple(c.clear_normal) != (0.0, 0.0, 0.0))
    or (want.color and (linear or env_background or want.multisample or int(clear.color) != 0))
    or (want.depth and float(c.clear_depth) != 0.0)
  )
  return want


def convert_outputs(scene: SensorScene, views: SensorViews, call: Call) -> None:
  """Newton's values in the caller's arrays: one in-place pass, skipped when nothing differs."""
  r = views.r
  linear = int(call.config.output_color_space) == int(ColorSpace.LINEAR)
  clear = _clear(call.clear, linear=linear)
  want = _wanted(call, clear, linear=linear, env_background=r.features.env_background)
  if not want.pixel_pass and call.world_indices is None and not want.depth:
    return  # the renderer wrote Newton's values
  im = call.images
  images = sk.ViewArrays()
  images.color = im.color if im.color is not None else scene.placeholder(wp.uint32)
  images.depth = im.depth if im.depth is not None else scene.placeholder(wp.float32)
  images.forward = im.forward if im.forward is not None else scene.placeholder(wp.float32)
  images.shape = im.shape if im.shape is not None else scene.placeholder(wp.uint32)
  images.normal = im.normal if im.normal is not None else scene.placeholder(wp.vec3f)
  images.albedo = im.albedo if im.albedo is not None else scene.placeholder(wp.uint32)
  images.hdr = im.hdr if im.hdr is not None else scene.placeholder(wp.vec3f)
  view = sk.Views()
  view.world = call.world_indices if call.world_indices is not None else scene.no_world
  view.has_world, view.world_count, view.exposure = call.world_indices is not None, call.world_count, views.exposure
  out = r.outputs
  rgba = out.rgb if out.rgb is not None else _placeholder4(scene, wp.vec4ub)
  albedo = out.albedo if out.albedo is not None else _placeholder4(scene, wp.vec3)
  normal = out.normal if out.normal is not None else _placeholder4(scene, wp.vec3)
  wp.launch(
    sk.newton_outputs,
    dim=(call.view_count, call.width * call.height),
    inputs=[rgba, albedo, normal, call.width, view, views.bundle, clear, want],
    outputs=[images],
    device=scene.scene.device,
  )


def _placeholder4[D](scene: SensorScene, dtype: type[D]) -> wp.array[D, [int, int, int, int]]:
  """A (1, 1, 1, 1) array standing in for a renderer output that does not exist."""
  return scene.placeholder(dtype).reshape((1, 1, 1, 1))
