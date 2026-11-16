"""MuJoCo Warp's renderer API on raywarp: the one-import switch.

Replace MuJoCo Warp's render calls by this module's, same names, arguments and output dtypes::

  from raywarp.integrations import mujoco as mjr  # instead of mujoco_warp's render functions

  rc = mjr.create_render_context(
    mjm,
    nworld=n,
    cam_res=(64, 64),
    render_rgb=True,
    render_depth=True,
  )
  mjr.render(m, d, rc)  # every step (after mjw.step)
  mjr.get_rgb(rc, 0, rgb)  # (nworld, H, W) vec3 in [0, 1]
  mjr.get_depth(rc, 0, 10.0, depth)  # (nworld, H, W) float, depth / scale clamped to [0, 1]
  mjr.get_segmentation(rc, 0, seg)  # (nworld, H, W) vec2i (object id, mjtObj type)

The context builds the raywarp scene and `Renderer` at its first `render(m, d, rc)` (the scene
aliases `m`, MuJoCo Warp's `RenderContext` needs only `mjm`); `rc.renderer` is the Renderer for
everything beyond MuJoCo Warp's API (settings, per-world tables, outputs). Arguments map to
`RenderSettings` over `settings` (default `RenderSettings()`; `RenderSettings().replace(rw.FAST)`
for maximum throughput): `use_shadows`, `use_textures`, `samples_per_pixel`,
`enable_backface_culling`, `background_color`, `render_skybox` and `enabled_geom_groups` when
given; its execution hints (`use_fast_math`, `use_precomputed_rays`, `deterministic`) have no
effect, its shading-model options are accepted at their defaults only and splats raise
(`create_render_context`). Per-camera `render_rgb` /
`render_depth` / `render_seg` lists render the union for every camera (one output set per
Renderer). `refit_bvh` is not needed: `render` refits.

The policy path skips `get_rgb`'s float vec3 expansion: `color=ColorOutput(channels=3,
planar=True)` renders channels-first uint8 (or float16) planes and `bind_outputs(m, rc,
rgb=obs)` writes them straight into the caller's observation tensor (`depth=` likewise);
`capture(m, d, rc, step)` records step + render as a CUDA graph (`Renderer.capture`)::

  rc = mjr.create_render_context(
    mjm,
    nworld=n,
    cam_res=(128, 128),
    render_rgb=True,
    color=rw.settings.ColorOutput(channels=3, planar=True),
  )
  obs = torch.empty((n * ncam, 3, 128, 128), dtype=torch.uint8, device="cuda")
  mjr.bind_outputs(m, rc, rgb=wp.from_torch(obs))
  graph = mjr.capture(m, d, rc, step=lambda: mjw.step(m, d))
  graph.launch()  # every step: physics, then the observation in `obs`
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Callable, Sequence
from typing import Final, final

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from raywarp._base import arguments
from raywarp._base.dims import AnyDeviceArray, AnyHostArray
from raywarp.core.outputs.render_outputs import RenderOutputs
from raywarp.core.pipeline.graph import RenderGraph
from raywarp.core.pipeline.renderer import Renderer
from raywarp.integrations.mujoco.build import Options as MujocoOptions
from raywarp.integrations.mujoco.build import scene as mujoco_scene
from raywarp.settings import ColorOutput, Outputs, RenderSettings
from raywarp.settings.fields import OutputKind
from raywarp.settings.render import mapped

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# mjtObj of the segmentation pairs.
_GEOM: Final = wp.constant(int(mujoco.mjtObj.mjOBJ_GEOM))
_FLEX: Final = wp.constant(int(mujoco.mjtObj.mjOBJ_FLEX))

# The `mjm.cam_output` bit of each output kind (read for a `render_*` flag left None).
_CAMOUT_BITS: Final = {
  "rgb": int(mujoco.mjtCamOutBit.mjCAMOUT_RGB),
  "depth": int(mujoco.mjtCamOutBit.mjCAMOUT_DEPTH),
  "seg": int(mujoco.mjtCamOutBit.mjCAMOUT_SEG),
}

type Flags = Sequence[bool] | bool | None
# Channels of MuJoCo Warp's `background_color`.
_RGBA: Final = 4

# MuJoCo Warp's shading-model options and their defaults (`create_render_context` accepts these
# values only: they leave MuJoCo Warp's full model on, which raywarp's PBR shading replaces).
# Final[dict[str, bool | float]]: the values mix bools and a float; the annotation types them.
_MJW_SHADING: Final[dict[str, bool | float]] = {
  "use_ambient_lighting": True,
  "enable_per_light_ambient": True,
  "flex_render_smooth": True,
  "shadow_light_fraction": 0.3,
  "enable_vertex_normals": True,
  "enable_specular": True,
  "enable_emission": True,
}


@final
@dataclasses.dataclass
class RenderContext:
  """A raywarp render context with MuJoCo Warp's interface (`create_render_context`).

  Attributes:
    mjm: the host model.
    nworld: worlds.
    cameras: the rendered MuJoCo camera ids.
    cam_res: (width, height) per rendered camera.
    settings: the Renderer's settings.
    geom_groups: the rendered geom groups.
    use_skybox: fill the background with the model's skybox.
    renderer: the Renderer, built by the first `render` (None before).
    model: the MuJoCo Warp model the Renderer aliases (the first call's `m`; None before).
  """

  mjm: mujoco.MjModel
  nworld: int
  cameras: list[int]
  cam_res: list[tuple[int, int]]
  settings: RenderSettings
  geom_groups: tuple[int, ...]
  use_skybox: bool
  renderer: Renderer | None = None
  model: mjw.Model | None = None


def _flag(value: Flags, mjm: mujoco.MjModel, cams: list[int], kind: OutputKind) -> bool:
  """Whether any rendered camera `cams` wants output `kind`, as MuJoCo Warp reads its `render_*`
  flag: None, the model's `cam_output` bits; a bool, every camera; a list, one flag per active
  camera, one per model camera (the active ones read), or one for all.

  Raises:
    ValueError: for a list of another length.
  """
  if value is None:
    return any(int(mjm.cam_output[c]) & _CAMOUT_BITS[kind] for c in cams)
  if isinstance(value, (bool, np.bool_)):
    return value
  flags = list(value)
  if len(flags) == mjm.ncam and len(cams) != mjm.ncam:
    flags = [flags[c] for c in cams]
  if len(flags) not in {1, len(cams)}:
    raise ValueError(f"render_{kind} length ({len(flags)}) does not match active camera count ({len(cams)}).")
  return any(flags)


def _truthy(value: int | None) -> bool | None:
  """A MuJoCo Warp flag as a bool (it takes truthy ints, e.g. use_shadows=1; None: unset)."""
  return None if value is None else bool(value)


def _active(mjm: mujoco.MjModel, cam_active: Sequence[bool] | Sequence[str] | Sequence[int] | None) -> list[int]:
  """MuJoCo Warp's camera selection: booleans, names or ids (None: every camera).

  Raises:
    ValueError: for an unknown camera name or id, or a mask of the wrong length.
  """
  if cam_active is None:
    return list(range(mjm.ncam))
  items = list(cam_active)
  if items and isinstance(items[0], (bool, np.bool_)):
    if len(items) != mjm.ncam:
      raise ValueError(f"cam_active must have length {mjm.ncam} (got {len(items)})")
    return [int(i) for i in np.flatnonzero(items)]
  ids = []
  for c in items:
    if not isinstance(c, str | int | np.integer) or isinstance(c, bool | np.bool_):
      raise ValueError(f"Invalid cam_active format: {list(cam_active)}")
    i = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, c) if isinstance(c, str) else operator.index(c)
    if not 0 <= i < mjm.ncam:
      raise ValueError(f"Camera '{c}' not found in model.")
    ids.append(i)
  return ids


def _resolutions(
  mjm: mujoco.MjModel, cams: list[int], cam_res: list[tuple[int, int]] | tuple[int, int] | None
) -> list[tuple[int, int]]:
  """(width, height) per rendered camera, as MuJoCo Warp reads `cam_res` (None: `cam_resolution`).

  Raises:
    ValueError: for a count that matches neither one nor every camera.
  """
  if cam_res is None:
    return [(int(mjm.cam_resolution[c][0]), int(mjm.cam_resolution[c][1])) for c in cams]
  res = [tuple(cam_res)] * len(cams) if isinstance(cam_res, tuple) else [tuple(r) for r in cam_res]
  if len(res) == 1 and len(cams) > 1:
    res = res * len(cams)
  if len(res) != len(cams):
    raise ValueError(f"Camera resolutions count ({len(res)}) does not match active camera count ({len(cams)}).")
  return [(w, h) for w, h in res]


def _check_unsupported(
  shading: dict[str, bool | float],
  splats: tuple[AnyHostArray | None, ...],
  background_color: tuple[float, float, float, float] | None,
) -> None:
  """Raises for what `create_render_context` accepts but raywarp does not render.

  Raises:
    ValueError: for a shading-model option off its MuJoCo Warp default (`_MJW_SHADING`), splat
      data, or a background color that is not RGBA with alpha 1.
  """
  for name, value in shading.items():
    if value != _MJW_SHADING[name]:
      raise ValueError(
        f"create_render_context: {name}={value!r} configures MuJoCo Warp's shading model; raywarp renders with"
        f" its own (only the default {_MJW_SHADING[name]!r} is accepted): set `settings` instead"
      )
  if any(v is not None for v in splats):
    raise ValueError("create_render_context: raywarp does not render Gaussian splats (splat_* given)")
  if background_color is not None and (len(background_color) != _RGBA or float(background_color[3]) != 1.0):
    raise ValueError(
      f"create_render_context: background_color is opaque RGBA: its alpha must be 1, got {tuple(background_color)}"
    )


def _requested_outputs(
  mjm: mujoco.MjModel, cams: list[int], flags: dict[OutputKind, Flags], color: ColorOutput | None
) -> Outputs:
  """The outputs any active camera `cams` renders by its `render_*` `flags` (their union), the
  color stored as `color` (None: packed RGBA).

  Raises:
    ValueError: no camera renders rgb, depth or segmentation, or a flag list of the wrong length.
  """
  kinds = [k for k, value in flags.items() if _flag(value, mjm, cams, k)]
  if not kinds:
    raise ValueError("create_render_context: no camera renders rgb, depth or segmentation")
  outputs = Outputs.of(*kinds)
  return dataclasses.replace(outputs, rgb=color) if color is not None and "rgb" in kinds else outputs


def create_render_context(  # noqa: PLR0913, PLR0917  # MuJoCo Warp's first 7 parameters, in its order
  mjm: mujoco.MjModel,
  nworld: int = 1,
  cam_res: list[tuple[int, int]] | tuple[int, int] | None = None,
  render_rgb: Flags = None,
  render_depth: Flags = None,
  render_seg: Flags = None,
  use_textures: bool | int | None = None,  # noqa: FBT001  # MuJoCo Warp's signature
  # Keyword-only from here: MuJoCo Warp's 8th parameter is use_fast_math, so a positional call
  # would bind the wrong parameter.
  *,
  use_shadows: bool | int | None = None,
  enabled_geom_groups: Sequence[int] = (0, 1, 2),
  cam_active: Sequence[bool] | Sequence[str] | Sequence[int] | None = None,
  background_color: tuple[float, float, float, float] | None = None,
  render_skybox: bool = True,
  enable_backface_culling: bool | int | None = None,
  samples_per_pixel: int | None = None,
  settings: RenderSettings | None = None,
  color: ColorOutput | None = None,
  # MuJoCo Warp's execution hints: no effect on what raywarp renders.
  use_fast_math: bool = True,  # noqa: ARG001  # MuJoCo Warp's signature
  use_precomputed_rays: bool = True,  # noqa: ARG001  # MuJoCo Warp's signature
  deterministic: bool = False,  # noqa: ARG001  # MuJoCo Warp's signature
  # MuJoCo Warp's shading model: only its defaults (`_MJW_SHADING`).
  use_ambient_lighting: bool = True,  # MuJoCo Warp's signature
  enable_per_light_ambient: bool = True,  # MuJoCo Warp's signature
  flex_render_smooth: bool = True,  # MuJoCo Warp's signature
  shadow_light_fraction: float = 0.3,
  enable_vertex_normals: bool = True,  # MuJoCo Warp's signature
  enable_specular: bool = True,  # MuJoCo Warp's signature
  enable_emission: bool = True,  # MuJoCo Warp's signature
  splat_position: AnyHostArray | None = None,
  splat_rotation: AnyHostArray | None = None,
  splat_scale: AnyHostArray | None = None,
  splat_rgba: AnyHostArray | None = None,
  splat_adr: AnyHostArray | None = None,
  splat_group_id: AnyHostArray | None = None,
) -> RenderContext:
  """MuJoCo Warp's `create_render_context` (same parameters), rendering with raywarp (see the
  module doc).

  `settings`: the base settings (default `RenderSettings()`; `RenderSettings().replace(rw.FAST)`
  for maximum throughput); the mapped arguments given (not None) override them. `color`: the
  color storage (e.g. channels-first planes for `bind_outputs(rgb=...)`; default packed RGBA,
  read by `get_rgb`). Per-camera `render_*` lists render their union for every camera (one output
  set per Renderer). `render_skybox` defaults to True (MuJoCo Warp: False): the model's skybox is
  the background. MuJoCo Warp's execution hints (`use_fast_math`, `use_precomputed_rays`,
  `deterministic`) have no effect. Its shading-model options (ambient terms, flex smoothing,
  shadow light fraction, vertex normals, specular, emission) are accepted at MuJoCo Warp's
  defaults only: raywarp shades with its own PBR model, configured by `settings`. Gaussian splats
  are not rendered.

  Raises:
    ValueError: for nworld that is not an integer >= 1, an unknown camera, mismatched
      resolutions or `render_*` lists, nothing to render, a geom group outside [0, mjNGROUP), a
      shading-model option off its MuJoCo Warp default, splat data, or a background color that is
      not RGBA with alpha 1 (the background is opaque).
  """
  shading = {
    "use_ambient_lighting": use_ambient_lighting,
    "enable_per_light_ambient": enable_per_light_ambient,
    "flex_render_smooth": flex_render_smooth,
    "shadow_light_fraction": shadow_light_fraction,
    "enable_vertex_normals": enable_vertex_normals,
    "enable_specular": enable_specular,
    "enable_emission": enable_emission,
  }
  splats = (splat_position, splat_rotation, splat_scale, splat_rgba, splat_adr, splat_group_id)
  _check_unsupported(shading, splats, background_color)
  nworld = arguments.check_int("create_render_context: nworld", nworld, low=1)
  adapter = MujocoOptions(geom_groups=tuple(enabled_geom_groups), use_skybox=render_skybox)  # validates the groups
  cams = _active(mjm, cam_active)
  if not cams:
    raise ValueError("create_render_context needs at least one active camera")
  flags: dict[OutputKind, Flags] = {"rgb": render_rgb, "depth": render_depth, "seg": render_seg}
  fields: dict[str, object | None] = {
    "outputs": _requested_outputs(mjm, cams, flags, color),
    "shadows.enabled": _truthy(use_shadows),
    "surfaces.textures": _truthy(use_textures),
    "surfaces.backface_culling": _truthy(enable_backface_culling),
    "aa.spp": samples_per_pixel,
    "environment.background_color": None if background_color is None else tuple(float(c) for c in background_color[:3]),
  }
  base = settings if settings is not None else RenderSettings()
  return RenderContext(
    mjm=mjm,
    nworld=nworld,
    cameras=cams,
    cam_res=_resolutions(mjm, cams, cam_res),
    settings=mapped(base, fields),
    geom_groups=adapter.geom_groups,
    use_skybox=adapter.use_skybox,
  )


def context_renderer(m: mjw.Model, rc: RenderContext, d: mjw.Data | None = None) -> Renderer:
  """The context's Renderer, built at the first call (the scene aliases `m`; `d`'s mocap poses
  decide the shared mocap bodies, `rw.integrations.mujoco.scene`): `rc.renderer`.

  Raises:
    ValueError: `m` is another model than the one the Renderer was built with.
  """
  if rc.model is not None and m is not rc.model:
    raise ValueError("this render context's Renderer reads the model it was built with: create a context per model")
  if rc.renderer is None:
    rc.model = m
    gpu = mujoco_scene(rc.mjm, m, d, adapter=MujocoOptions(geom_groups=rc.geom_groups, use_skybox=rc.use_skybox))
    names = gpu.camera_names
    specs = [dataclasses.replace(gpu.cameras[names[c]], resolution=res) for c, res in zip(rc.cameras, rc.cam_res, strict=True)]
    rc.renderer = Renderer(gpu, nworld=rc.nworld, cameras=specs, resolution=rc.cam_res[0], settings=rc.settings)
  return rc.renderer


def render(m: mjw.Model, d: mjw.Data, rc: RenderContext) -> None:
  """MuJoCo Warp's `render`: renders the current frame of `d` into the context's outputs.

  The first call builds the scene (aliasing `m`) and the Renderer (`rc.renderer`).
  """
  context_renderer(m, rc, d).render(d)


def bind_outputs(m: mjw.Model, rc: RenderContext, **arrays: AnyDeviceArray) -> None:
  """Renders into the caller's arrays (`Renderer.bind_outputs`: e.g. `rgb` observation planes,
  `depth`), zero copy; builds the Renderer first if needed. Captured graphs become stale."""
  context_renderer(m, rc).bind_outputs(**arrays)


def capture(
  m: mjw.Model,
  d: mjw.Data,
  rc: RenderContext,
  step: Callable[[], None] | None = None,
  *,
  after: Callable[[], None] | None = None,
) -> RenderGraph[mjw.Data]:
  """A CUDA graph of `step()` (e.g. `mjw.step(m, d)`) + `render(m, d, rc)` + `after()`
  (`Renderer.capture`: runs the work once eagerly first; CUDA only)."""
  return context_renderer(m, rc, d).capture(step, d, after=after)


def _outputs(rc: RenderContext, camera_index: int) -> RenderOutputs:
  """Camera `camera_index`'s outputs, (nworld, 1, H, W) views.

  Raises:
    ValueError: before the first `render`, or for a camera index outside the context's cameras.
  """
  if rc.renderer is None:
    raise ValueError("render(m, d, rc) must run before reading the outputs")
  cameras = rc.renderer.outputs.cameras
  arguments.check_int("camera_index", camera_index, low=0, high=len(cameras) - 1)
  return cameras[camera_index]


def _rendered(view: AnyDeviceArray | None, name: str, out: AnyDeviceArray) -> AnyDeviceArray:
  """`view`, an output of the context (named `name`), checked against the caller's `out`.

  Raises:
    ValueError: for an output the context does not render (None), or an `out` not shaped
      (nworld, height, width) like it.
  """
  if view is None:
    raise ValueError(f"the render context does not render {name} (create_render_context(render_...=True))")
  expected = (view.shape[0], view.shape[2], view.shape[3])
  if tuple(out.shape) != expected:
    raise ValueError(f"the {name} output array must have shape {expected} (nworld, height, width), got {tuple(out.shape)}")
  return view


@wp.kernel
def _rgb_kernel(
  # In:
  rgba: wp.array4d[wp.vec4ub],
  # Out:
  rgb_out: wp.array3d[wp.vec3],
):
  """8-bit sRGB bytes -> vec3 in [0, 1] (MuJoCo Warp's `get_rgb`)."""
  w, y, x = wp.tid()
  c = rgba[w, 0, y, x]
  rgb_out[w, y, x] = wp.vec3(float(c[0]), float(c[1]), float(c[2])) * wp.static(1.0 / 255.0)


@wp.kernel
def _depth_kernel(
  # In:
  depth: wp.array4d[float],
  depth_scale: float,
  # Out:
  depth_out: wp.array3d[float],
):
  """Depth / scale, clamped to [0, 1] (MuJoCo Warp's `get_depth`)."""
  w, y, x = wp.tid()
  depth_out[w, y, x] = wp.clamp(depth[w, 0, y, x] / depth_scale, 0.0, 1.0)


@wp.kernel
def _seg_kernel(
  # In:
  seg: wp.array4d[int],
  ngeom: int,
  # Out:
  seg_out: wp.array3d[wp.vec2i],
):
  """Geom id (flex f: ngeom + f) -> MuJoCo Warp's (object id, mjtObj type); (-1, -1) on a miss."""
  w, y, x = wp.tid()
  s = seg[w, 0, y, x]
  out = wp.vec2i(-1, -1)
  if s >= ngeom:
    out = wp.vec2i(s - ngeom, _FLEX)
  elif s >= 0:
    out = wp.vec2i(s, _GEOM)
  seg_out[w, y, x] = out


def get_rgb(rc: RenderContext, camera_index: int, rgb_out: wp.array[wp.vec3, [int, int, int]]) -> None:
  """MuJoCo Warp's `get_rgb`: camera `camera_index`'s color, (nworld, height, width) vec3 in
  [0, 1]."""
  view = _rendered(_outputs(rc, camera_index).rgb, "rgb", rgb_out)
  if view.dtype != wp.vec4ub:
    raise ValueError(
      "get_rgb reads the RGBA color (ColorOutput(channels=4)); read a 3-channel or planar color from outputs.rgb"
    )
  wp.launch(_rgb_kernel, dim=rgb_out.shape, inputs=[view], outputs=[rgb_out], device=view.device)


def get_depth(rc: RenderContext, camera_index: int, depth_scale: float, depth_out: wp.array[float, [int, int, int]]) -> None:
  """MuJoCo Warp's `get_depth`: camera `camera_index`'s depth / `depth_scale`, clamped to [0, 1]."""
  # float32: `create_render_context` requests the default DepthOutput.
  view = _rendered(_outputs(rc, camera_index).depth, "depth", depth_out)
  wp.launch(_depth_kernel, dim=depth_out.shape, inputs=[view, float(depth_scale)], outputs=[depth_out], device=view.device)


def get_segmentation(rc: RenderContext, camera_index: int, seg_out: wp.array[wp.vec2i, [int, int, int]]) -> None:
  """MuJoCo Warp's `get_segmentation`: (object id, mjtObj type) per pixel, (-1, -1) on a miss."""
  view = _rendered(_outputs(rc, camera_index).seg, "seg", seg_out)
  wp.launch(_seg_kernel, dim=seg_out.shape, inputs=[view, rc.mjm.ngeom], outputs=[seg_out], device=view.device)
