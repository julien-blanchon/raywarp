"""The checked arguments of one `SensorCamera.update` (step 1 of `frame.py`).

`validate_call` checks the caller's arrays as Newton's sensor does and returns the typed `Call`
(`UpdateImages`: the requested output arrays, Newton's names and dtypes).
"""

from __future__ import annotations

import dataclasses
from typing import Final, NamedTuple, final

import newton
import warp as wp
from newton.sensors import SensorCamera as NewtonSensorCamera

from raywarp.settings import ColorOutput, Outputs
from raywarp.settings.fields import OutputKind

ClearData = NewtonSensorCamera.ClearData
RenderConfig = NewtonSensorCamera.RenderConfig
AntiAliasing = NewtonSensorCamera.AntiAliasing

type Image[D] = wp.array[D, [int, int, int]]
# One of Newton's output images (`UpdateImages` fields).
type NewtonImage = Image[wp.uint32] | Image[wp.float32] | Image[wp.vec3f]
# Newton's camera-space ray bundle: (height, width, sample_count, origin / direction).
type Rays = wp.array[wp.vec3f, [int, int, int, 2]]
# A ray bundle's rank and its last axis: (height, width, samples, (origin, direction)).
RAYS_NDIM: Final = 4
RAY_PAIR: Final = 2


class UpdateImages(NamedTuple):
  """The output arrays of one update (None: not requested), Newton's names and dtypes."""

  color: Image[wp.uint32] | None
  depth: Image[wp.float32] | None
  forward: Image[wp.float32] | None
  shape: Image[wp.uint32] | None
  normal: Image[wp.vec3f] | None
  albedo: Image[wp.uint32] | None
  hdr: Image[wp.vec3f] | None

  def named(self) -> dict[str, tuple[NewtonImage | None, type]]:
    """Newton's argument name -> (array, dtype)."""
    dtypes = (wp.uint32, wp.float32, wp.float32, wp.uint32, wp.vec3f, wp.uint32, wp.vec3f)
    names = (
      "color_image",
      "depth_image",
      "forward_depth_image",
      "shape_index_image",
      "normal_image",
      "albedo_image",
      "hdr_color_image",
    )
    return {name: (image, dtype) for name, image, dtype in zip(names, self, dtypes, strict=True)}

  def outputs(self) -> Outputs:
    """The renderer outputs these need (the color's coverage alpha marks the shaded misses)."""
    shaded: dict[OutputKind, NewtonImage | None] = {"hdr": self.hdr, "normal": self.normal, "albedo": self.albedo}
    kinds: set[OutputKind] = {k for k, image in shaded.items() if image is not None}
    if self.albedo is not None:
      # The albedo's hit test reads the first sample's normal.
      kinds.add("normal")
    if self.depth is not None or self.forward is not None:
      kinds.add("depth")
    if self.shape is not None:
      kinds.add("seg")
    rgb = ColorOutput(alpha=True) if kinds & {"hdr", "normal", "albedo"} or self.color is not None else None
    return Outputs.of(*sorted(kinds), **({"rgb": rgb} if rgb is not None else {}))

  def arrays(self) -> dict[str, NewtonImage]:
    """Renderer buffer -> caller array, where the layouts match (rendered in place)."""
    bound = {"rgb": self.color, "depth": self.depth if self.depth is not None else self.forward}
    bound |= {"seg": self.shape, "hdr": self.hdr, "normal": self.normal}
    return {name: image for name, image in bound.items() if image is not None}


@final
@dataclasses.dataclass(frozen=True)
class Call:
  """A validated update: its arrays, sizes, settings and clear values."""

  camera_transforms: wp.array[wp.transformf, [int]]
  camera_rays: Rays
  images: UpdateImages
  world_indices: wp.array[wp.int32, [int]] | None
  config: RenderConfig
  clear: ClearData
  view_count: int
  height: int
  width: int
  sample_count: int
  world_count: int


def _check_array(name: str, array: object, dtype: type, device: wp.Device) -> None:
  """Newton's array checks: a Warp array of `dtype` on `device`."""
  if not isinstance(array, wp.array):
    raise TypeError(f"{name} must be a Warp array, got {type(array).__name__}.")
  if array.dtype != dtype:
    raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}.")
  if array.device != device:
    raise RuntimeError(f"{name} must be on the model device ({device}), got {array.device}.")


def _check_rays(camera_rays: Rays, config: RenderConfig, device: wp.Device) -> int:
  """The bundle's sample count (raises for a malformed bundle or a missing anti-aliasing mode)."""
  _check_array("camera_rays", camera_rays, wp.vec3f, device)
  shaped = camera_rays.ndim == RAYS_NDIM and camera_rays.shape[2] > 0 and camera_rays.shape[3] == RAY_PAIR
  if not shaped or camera_rays.shape[0] <= 0 or camera_rays.shape[1] <= 0:
    raise ValueError(f"camera_rays must have shape (height, width, sample_count, 2), got {tuple(camera_rays.shape)}.")
  try:
    anti_aliasing = AntiAliasing(config.anti_aliasing)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"Invalid anti_aliasing mode: {config.anti_aliasing!r}") from exc
  samples = int(camera_rays.shape[2])
  if samples > 1 and anti_aliasing == AntiAliasing.NONE:
    raise ValueError("anti_aliasing must be SSAA or MSAA for camera_rays with more than one sample")
  return samples


def _check_worlds(
  world_indices: wp.array[wp.int32, [int]] | None, view_count: int, world_count: int, device: wp.Device
) -> None:
  """Newton's `world_indices` checks (and views beyond the worlds need it)."""
  if world_indices is None:
    if view_count > world_count:
      raise ValueError(
        f"view_count ({view_count}) exceeds model.world_count ({world_count}); "
        "pass an explicit world_indices mapping for the extra views."
      )
    return
  if world_indices.shape != (view_count,):
    raise ValueError(f"world_indices shape must be ({view_count},), got {tuple(world_indices.shape)}")
  if world_indices.dtype != wp.int32:
    raise ValueError(f"world_indices dtype must be int32, got {world_indices.dtype}")
  if world_indices.device != device:
    raise ValueError(f"world_indices device must be {device}, got {world_indices.device}")


def validate_call(  # noqa: PLR0913  # Newton's update() arguments, keyword-only
  model: newton.Model,
  *,
  camera_transforms: wp.array[wp.transformf, [int]],
  camera_rays: Rays,
  images: UpdateImages,
  world_indices: wp.array[wp.int32, [int]] | None,
  config: RenderConfig,
  clear: ClearData,
) -> Call | None:
  """The checked call (as Newton's sensor checks it), None when no output is requested."""
  device = model.device
  _check_array("camera_transforms", camera_transforms, wp.transformf, device)
  if camera_transforms.ndim != 1 or camera_transforms.shape[0] <= 0:
    raise ValueError(f"camera_transforms must have shape (view_count,), got {tuple(camera_transforms.shape)}.")
  samples = _check_rays(camera_rays, config, device)
  height, width = int(camera_rays.shape[0]), int(camera_rays.shape[1])
  view_count, world_count = int(camera_transforms.shape[0]), model.world_count
  _check_worlds(world_indices, view_count, world_count, device)
  expected = (view_count, height, width)
  for name, (image, dtype) in images.named().items():
    if image is not None:
      _check_array(name, image, dtype, device)
      if tuple(image.shape) != expected:
        raise ValueError(f"{name} shape must be {expected}, got {tuple(image.shape)}")
  if all(image is None for image in images):
    return None
  return Call(
    camera_transforms, camera_rays, images, world_indices, config, clear, view_count, height, width, samples, world_count
  )
