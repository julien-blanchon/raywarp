"""Value types of the Newton adapter: default lights, hemispheric ambient, checkerboard, the
simulator state a Newton scene renders."""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Collection
from typing import Final, final

import newton
import numpy as np
import warp as wp

from raywarp.scene.images.latlong import latlong_directions
from raywarp.scene.model.lights import Light

type Rgb = tuple[float, float, float]
type Vec3 = tuple[float, float, float]

# Newton's default light travels along normalised (-1, 1, -1) in a Z-up model.
NEWTON_LIGHT_DIRECTION: Final = (-0.57735026, 0.57735026, -0.57735026)


def from_z_up(v: Vec3, up_axis: int) -> Vec3:
  """`v` given for a Z-up model, rotated to a model whose up axis is `up_axis` (0, 1, 2).

  Newton's own rule for its default light (`Axis.Z.quat_between_axes(up)`).
  """
  if up_axis == int(newton.Axis.Z):
    return v
  q = newton.Axis.Z.quat_between_axes(newton.Axis(up_axis))
  r = wp.quat_rotate(q, wp.vec3(*v))
  return (float(r[0]), float(r[1]), float(r[2]))


def default_lights(up_axis: int = int(newton.Axis.Z)) -> list[Light]:
  """A key light with shadows and a dim fill (for scenes without an environment), for `up_axis`."""
  return [
    Light.sun(from_z_up((-0.4, 0.5, -0.75), up_axis), color=(1.0, 0.97, 0.92)),
    Light.sun(from_z_up((0.5, -0.3, -0.6), up_axis), color=(0.25, 0.27, 0.3), shadows=False),
  ]


@final
@dataclasses.dataclass(frozen=True)
class Hemisphere:
  """Hemispheric ambient light: radiance `sky` seen along the up axis, `ground` along -up.

  A Lambertian surface of normal n shows albedo * mix(ground, sky, (n . up + 1) / 2): Newton's
  sensor ambient (`set_ambient_light`) and ViewerGL's ambient.
  """

  sky: Rgb
  ground: Rgb

  def envmap(self, up_axis: int, height: int = 32) -> np.ndarray[[int, int, 3], np.dtype[np.float32]]:
    """Equirectangular radiance a + b (d . up) of that irradiance (the core's layout, `latlong`).

    A Lambertian surface under L(d) = a + b (d . up) shows albedo * (a + 2/3 b (n . up)), so
    a = (sky + ground) / 2 and b = 3/2 (sky - ground) / 2.
    """
    sky, ground = np.asarray(self.sky, np.float64), np.asarray(self.ground, np.float64)
    a = 0.5 * (sky + ground)
    b = 0.75 * (sky - ground)
    directions = latlong_directions(height, 2 * height)
    return (a + b * directions[..., up_axis : up_axis + 1]).astype(np.float32)


@final
@dataclasses.dataclass(frozen=True)
class Checkerboard:
  """Grey checkerboard texture multiplying the color of some shapes (Newton's sensor material).

  Attributes:
    shapes: indices of the textured shapes.
    resolution: texture size [px] (square).
    checker_size: size of a checker square [px].
    colors: sRGB grey of the even and odd squares.
  """

  shapes: frozenset[int]
  resolution: int = 64
  checker_size: int = 32
  colors: tuple[int, int] = (0x80, 0xBF)

  def __post_init__(self) -> None:
    """Validates the sizes and greys (ValueError)."""
    for name in ("resolution", "checker_size"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"Checkerboard.{name} must be a positive integer [px], got {value!r}")
    if len(self.colors) != 2 or not all(  # the even and odd squares
      isinstance(c, numbers.Integral) and not isinstance(c, bool) and 0 <= c <= 0xFF for c in self.colors
    ):
      raise ValueError(f"Checkerboard.colors must be two sRGB greys in [0, 255], got {self.colors!r}")

  def image(self) -> np.ndarray[[int, int, 3], np.dtype[np.uint8]]:
    """The (resolution, resolution, 3) uint8 sRGB texture."""
    cells = np.arange(self.resolution) // self.checker_size
    even = (cells[:, None] + cells[None, :]) % 2 == 0
    grey = np.where(even, *self.colors).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


@final
@dataclasses.dataclass
class StateView:
  """What `Renderer.render` takes for a Newton scene besides a bare `newton.State`.

  Attributes:
    state: the Newton state (its `body_q` is read on every call).
    world_indices: (nworld,) int32 Newton world shown by each render world (None: world v shows
      Newton world v); out-of-range values show world 0.
    world_offsets: (world_count,) vec3 translation added to each Newton world's instances.
    root: transform applied to every instance last (e.g. a viewer layer).
  """

  state: newton.State
  world_indices: wp.array[wp.int32, [int]] | None = None
  world_offsets: wp.array[wp.vec3, [int]] | None = None
  root: wp.transformf | None = None


# What `Renderer.render(sim)` takes for a Newton scene (the `NewtonBinding`'s simulator state).
type NewtonState = newton.State | StateView


@final
@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
  """Newton adapter options (`rw.integrations.newton.scene` / `load` / `binding`); the options
  every adapter shares are `rw.SceneOptions`.

  Attributes:
    merge_worlds: one render world holding every Newton world's shapes (viewer, no shared
      instances; bind with the same options); else render world v shows Newton world v (worlds of
      different shapes: their pooled shapes, each world showing its own, `build.pooled`).
    shapes: shape indices to render (default the visible, non-site shapes; shapes of opacity 0
      are never rendered); a world none of whose shapes is listed renders only the global ones.
    checkerboard: shapes given a grey checkerboard texture.
    load_textures: load mesh textures.
    particle_meshes: render the model's triangles over particles (cloth, soft-body skins) from
      `state.particle_q` (`raywarp.integrations.newton.deformables`).
    world_colors: the shapes whose `shape_color` differs between Newton worlds show each world's
      color, re-read from `model.shape_color` on every render (a per-world "instance.color"
      table, `binding.WorldColors`); False: world 0's color everywhere (reported).
  """

  merge_worlds: bool = False
  shapes: Collection[int] | None = None
  checkerboard: Checkerboard | None = None
  load_textures: bool = True
  particle_meshes: bool = True
  world_colors: bool = True


# The default options (a shared immutable value: argument defaults).
DEFAULT_OPTIONS: Final = Options()
