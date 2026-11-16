"""raywarp as an RGB backend of mjlab's camera sensors (the ``mjlab.rgb_renderers`` entry point).

mjlab loads this module when a camera sensor asks for ``rgb_backend="raywarp"``. Configure the
sensors with `RaywarpCameraSensorCfg` (a `CameraSensorCfg` with raywarp's fields)::

  from raywarp.integrations.mjlab import RaywarpCameraSensorCfg

  cam = RaywarpCameraSensorCfg(
    name="front",
    camera_name="robot/front",
    width=256,
    height=256,
    raywarp_settings=rw.RenderSettings().replace(rw.FAST),  # max throughput (None: the default)
    raywarp_per_world=rw.PerWorld(pose=["table/*"]),  # mocap entities posed per world
  )

A plain `CameraSensorCfg` with ``rgb_backend="raywarp"`` renders the default settings with its
`use_textures` / `use_shadows`; `raywarp_settings` must agree with those flags (else it raises).
`RaywarpCameraSensorCfg` defaults `use_shadows` to True. The renderers are built with their final
settings (no recompilation after start).

Memory: the sensors of one `enabled_geom_groups` share one scene (textures, meshes, BLASes);
the sensors of one resolution and configuration share one Renderer, which writes each camera's
RGB straight into the tensor mjlab reads (`ColorOutput(channels=3)`: no RGBA, no copy); each
sensor's tensor is a strided (nworld, H, W, 3) view of the world-major color (mjlab hands it on as
is, or a contiguous `.clone()` with `clone_data`; flatten it with `.reshape`, not `.view`).

mjlab wraps fixed-base entities (rooms, tables) in mocap bodies. One posed the same in every
world when the sensors initialise is shared scenery (`rw.integrations.mujoco.scene`);
`raywarp_per_world` declares the ones reset events later pose per world (env origins of a terrain,
randomised placements).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.sim.sim_data import WarpBridge

import raywarp as rw
import raywarp.integrations.mujoco
from raywarp.settings.render import mapped

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor

# The color every renderer writes: world-major RGB bytes bound to mjlab's tensors.
_OBSERVATION: Final = rw.Outputs(rgb=rw.settings.ColorOutput(channels=3))


@dataclasses.dataclass
class RaywarpCameraSensorCfg(CameraSensorCfg):
  """A mjlab camera sensor rendered by raywarp.

  Attributes:
    use_shadows: mjlab's flag, True by default here (raywarp's default renders shadows).
    raywarp_settings: the Renderer settings (default `RenderSettings()` with the sensor's flags;
      given, they must agree with `use_shadows` / `use_textures`); outputs are the adapter's
      (module doc).
    raywarp_per_world: the mocap entities posed per world (`rw.PerWorld`), or None.
  """

  rgb_backend: str = "raywarp"
  use_shadows: bool = True  # raywarp's default shades with shadows (mjlab's default is False)
  raywarp_settings: rw.RenderSettings | None = None
  raywarp_per_world: rw.PerWorld | None = None


def _raw[T](value: T | WarpBridge[T]) -> T:
  """The zero-copy Warp struct under mjlab's Torch-facing proxy."""
  return value.struct if isinstance(value, WarpBridge) else value


def _settings(cfg: CameraSensorCfg) -> rw.RenderSettings:
  """The Renderer settings of a sensor, with the observation-only color, by the adapters' rule
  (`settings.render.mapped`): the default with mjlab's `use_shadows` / `use_textures` mapped.
  Those flags are always set, so a sensor's own `raywarp_settings` (the base) must agree with
  them (textures "auto" agrees with use_textures=True) instead of being overwritten by them.

  Raises:
    TypeError: for a `raywarp_settings` that is not a RenderSettings.
    ValueError: for settings that contradict the sensor's flags.
  """
  explicit = cfg.raywarp_settings if isinstance(cfg, RaywarpCameraSensorCfg) else None
  if explicit is None:
    flags = {"surfaces.textures": cfg.use_textures, "shadows.enabled": cfg.use_shadows}
    return mapped(rw.RenderSettings(), flags).replace(outputs=_OBSERVATION)
  if not isinstance(explicit, rw.RenderSettings):
    raise TypeError(f"raywarp_settings must be a rw.RenderSettings, got {type(explicit).__name__}")
  textures = explicit.surfaces.textures is not False
  if explicit.shadows.enabled != cfg.use_shadows or textures != cfg.use_textures:
    raise ValueError(
      f"sensor {cfg.name!r}: raywarp_settings (shadows.enabled={explicit.shadows.enabled!r},"
      f" surfaces.textures={explicit.surfaces.textures!r}) contradict use_shadows={cfg.use_shadows},"
      f" use_textures={cfg.use_textures}: make them agree"
    )
  return explicit.replace(outputs=_OBSERVATION)


def _per_world(cfg: CameraSensorCfg) -> rw.PerWorld | None:
  """The sensor's `raywarp_per_world` declaration, if it has one.

  Raises:
    TypeError: for a declaration that is not a PerWorld.
  """
  declared = cfg.raywarp_per_world if isinstance(cfg, RaywarpCameraSensorCfg) else None
  if declared is not None and not isinstance(declared, rw.PerWorld):
    raise TypeError(f"raywarp_per_world must be a rw.PerWorld, got {type(declared).__name__}")
  return declared


@final
class MjlabRgbRenderer:
  """One Renderer for mjlab RGB cameras of one resolution and configuration (`create_groups`).

  Attributes:
    rgb: MuJoCo camera id -> its (nworld, height, width, 3) uint8 tensor, written in place by
      every `render` (a view of the world-major color; mjlab reads it after `wait_for_rgb`).
    renderer: the raywarp Renderer.
    data: the MuJoCo Warp data every `render` reads.
  """

  def __init__(
    self,
    scene: rw.GpuScene[mjw.Data],
    data: mjw.Data,
    sensors: Sequence[CameraSensor],
    *,
    settings: rw.RenderSettings,
    per_world: rw.PerWorld | None,
  ) -> None:
    """Builds the Renderer of `sensors` (one resolution) over `scene`, its color bound to `rgb`.

    Raises:
      ValueError: for no sensor or sensors of different resolutions.
    """
    if not sensors:
      raise ValueError("at least one RGB sensor is required")
    if len({(s.cfg.width, s.cfg.height) for s in sensors}) != 1:
      raise ValueError("RGB sensors in one group must have the same resolution")
    width, height = sensors[0].cfg.width, sensors[0].cfg.height
    device = wp.get_device(scene.device)
    self.data = data
    self._storage = torch.empty((data.nworld, len(sensors), height, width, 3), dtype=torch.uint8, device=str(device))
    binding = cast("wp.array[wp.vec3ub, [int, int, int, int]]", wp.from_torch(self._storage, dtype=wp.vec3ub))
    self.renderer = rw.Renderer(
      scene,
      nworld=data.nworld,
      cameras=[sensor.camera_idx for sensor in sensors],
      resolution=(width, height),
      settings=settings,
      per_world=per_world,
      output_arrays={"rgb": binding},
    )
    # (nworld, H, W, 3) per sensor: strided views of the world-major color.
    self.rgb: dict[int, torch.Tensor] = {sensor.camera_idx: self._storage[:, i] for i, sensor in enumerate(sensors)}
    self._pending: torch.cuda.Stream | None = None

  def render(self) -> None:
    """Renders the current mjlab state into `rgb`, ordered after Torch's current stream (mjlab's
    physics); `wait_for_rgb` orders the readers after it.

    Raises:
      RuntimeError: when the renderer's color is no longer bound to `rgb` (rebound elsewhere).
    """
    view = self.renderer.outputs.rgb
    if view is None or view.ptr != self._storage.data_ptr():
      raise RuntimeError("the raywarp mjlab RGB binding changed; recreate the sensor context")
    stream = torch.cuda.current_stream()
    producer = wp.get_stream(self.renderer.device)
    producer.wait_stream(wp.stream_from_torch(stream))
    self.renderer.render(self.data)
    stream.wait_stream(wp.stream_to_torch(producer))
    self._pending = stream

  def wait_for_rgb(self) -> None:
    """Orders the caller's Torch stream after the last `render`."""
    if self._pending is not None:
      torch.cuda.current_stream().wait_stream(self._pending)
      self._pending = None

  def close(self) -> None:
    """Finishes the pending render before releasing the Renderer (released also when the wait
    fails)."""
    pending, self._pending = self._pending, None
    try:
      if pending is not None:
        pending.synchronize()
    finally:
      self.renderer.close()


class _GroupKey(NamedTuple):
  """The sensors sharing one Renderer: geom groups, resolution and configuration (PerWorld holds
  lists: keyed by its repr, the settings by theirs, like it)."""

  geom_groups: tuple[int, ...]
  width: int
  height: int
  settings: str
  per_world: str


def create_groups(
  mj_model: mujoco.MjModel,
  model: mjw.Model | WarpBridge[mjw.Model],
  data: mjw.Data | WarpBridge[mjw.Data],
  sensors: list[CameraSensor],
  wp_device: wp.Device,
) -> list[MjlabRgbRenderer]:
  """The factory of mjlab's RGB renderers (the entry point): one scene per geom-group set, one
  `MjlabRgbRenderer` per resolution, settings and per-world declaration over it, on the device of
  `model` (mjlab's `wp_device`).

  Raises:
    TypeError, ValueError: for invalid raywarp fields of a sensor (`RaywarpCameraSensorCfg`),
      or a `wp_device` that is not the model's.
  """
  raw_model, raw_data = _raw(model), _raw(data)
  if wp.get_device(wp_device) != raw_model.geom_type.device:
    raise ValueError(f"mjlab's device {wp_device} is not the model's ({raw_model.geom_type.device})")
  groups: dict[_GroupKey, list[CameraSensor]] = {}
  configs: dict[_GroupKey, tuple[rw.RenderSettings, rw.PerWorld | None]] = {}
  for sensor in sensors:
    cfg = sensor.cfg
    settings, per_world = _settings(cfg), _per_world(cfg)
    key = _GroupKey(tuple(sorted(set(cfg.enabled_geom_groups))), cfg.width, cfg.height, repr(settings), repr(per_world))
    groups.setdefault(key, []).append(sensor)
    configs[key] = (settings, per_world)
  scenes: dict[tuple[int, ...], rw.GpuScene[mjw.Data]] = {}
  renderers: list[MjlabRgbRenderer] = []
  try:
    for key, members in groups.items():
      if key.geom_groups not in scenes:
        adapter = rw.integrations.mujoco.Options(geom_groups=key.geom_groups)
        scenes[key.geom_groups] = rw.integrations.mujoco.scene(mj_model, raw_model, raw_data, adapter=adapter)
      settings, per_world = configs[key]
      renderers.append(MjlabRgbRenderer(scenes[key.geom_groups], raw_data, members, settings=settings, per_world=per_world))
  except BaseException:
    for r in renderers:  # the groups built before the failure (nothing else owns them)
      r.renderer.close()
    raise
  return renderers
